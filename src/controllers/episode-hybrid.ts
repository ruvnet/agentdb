/**
 * Hybrid (dense + BM25) episode retrieval for ReflexionMemory (ADR-016).
 *
 * Dense search alone misses episodes whose wording matches the query but whose embedding does not (names, error
 * codes, numbers). This keeps a BM25 index over the most recent episodes, rebuilt only when the episodes table
 * changes, and fuses its ranking with the dense ranking by reciprocal-rank fusion. On LongMemEval session retrieval
 * (benchmarks/longmemeval, which runs these ranking functions over turn chunks, not the episode store), the same
 * fusion lifts recall_all@5 over dense-only ranking of the same chunks from 0.883 to 0.900 (S) and from 0.630 to
 * 0.719 (M), with the default MiniLM embedder. See ADR-016.
 */
import type { IDatabaseConnection } from '../types/database.types.js';
import { Bm25Index, tokenize } from '../search/conversation-retrieval.js';
import { crossEncoderScores } from '../search/cross-encoder.js';

/**
 * Episodes indexed for BM25, newest first; older episodes are still found by dense search. The first search in a
 * process builds the index (about 200 ms at 10k episodes, 1 s at 50k); later writes are appended incrementally.
 */
export const HYBRID_MAX_INDEXED = clampInt(process.env.AGENTDB_HYBRID_MAX_EPISODES, 10_000, 1, 1_000_000);

function clampInt(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = Math.floor(Number(raw));
  return Number.isFinite(n) && n >= min ? Math.min(n, max) : fallback;
}
/** The candidate pool each side contributes before fusion. */
export const HYBRID_MAX_POOL = 500;
/** Weight of the BM25 ranking against the dense ranking. Equal weights won on LongMemEval_M; see ADR-016. */
export const HYBRID_LEXICAL_WEIGHT = 1;
const RRF_K = 60;

interface EpisodeTextRow {
  id: number;
  task: string | null;
  input: string | null;
  output: string | null;
  critique: string | null;
}

/**
 * A BM25 index over episode text. New episodes are appended incrementally (the common write-then-read pattern);
 * the index is rebuilt only when rows were deleted or replaced, or it has grown past 1.25x its cap.
 */
export class EpisodeLexicalIndex {
  private count = -1;
  private maxId = 0;
  private ids: number[] = [];
  private index: Bm25Index | null = null;

  constructor(
    private readonly db: IDatabaseConnection,
    private readonly maxDocs = HYBRID_MAX_INDEXED,
  ) {}

  private static textOf(r: EpisodeTextRow): string[] {
    return tokenize([r.task, r.input, r.output, r.critique].filter(Boolean).join('\n'));
  }

  private refresh(): void {
    const sig = this.db.prepare<{ c: number; m: number | null }>('SELECT COUNT(*) AS c, MAX(id) AS m FROM episodes').get();
    const count = sig?.c ?? 0;
    const maxId = sig?.m ?? 0;
    if (this.index && count === this.count && maxId === this.maxId) return;
    if (this.index && count > this.count && maxId > this.maxId && this.ids.length < this.maxDocs * 1.25) {
      const added = this.db
        .prepare<EpisodeTextRow>('SELECT id, task, input, output, critique FROM episodes WHERE id > ? ORDER BY id ASC')
        .all(this.maxId);
      if (this.count + added.length === count) {
        for (const r of added) {
          this.index.add(EpisodeLexicalIndex.textOf(r));
          this.ids.push(r.id);
        }
        this.count = count;
        this.maxId = maxId;
        return;
      }
    }
    const rows = this.db
      .prepare<EpisodeTextRow>('SELECT id, task, input, output, critique FROM episodes ORDER BY id DESC LIMIT ?')
      .all(this.maxDocs);
    this.index = new Bm25Index(rows.map(r => EpisodeLexicalIndex.textOf(r)));
    this.ids = rows.map(r => r.id);
    this.count = count;
    this.maxId = maxId;
  }

  /** Episode ids with a BM25 match, best first, at most `limit`. */
  search(query: string, limit: number): { id: number; score: number }[] {
    this.refresh();
    if (!this.index || this.ids.length === 0) return [];
    const scores = this.index.score(tokenize(query, 256));
    const hits: { id: number; score: number }[] = [];
    for (let i = 0; i < scores.length; i++) if (scores[i] > 0) hits.push({ id: this.ids[i], score: scores[i] });
    hits.sort((a, z) => z.score - a.score || z.id - a.id);
    return hits.slice(0, limit);
  }
}

/**
 * Weighted RRF scores of a dense ranking (best first) and a lexical ranking (best first) over episode ids.
 * Episodes only in the lexical list are ranked just past the dense pool on the dense side.
 */
export function fuseEpisodeScores(denseIds: readonly number[], lexical: readonly { id: number }[], lexicalWeight = HYBRID_LEXICAL_WEIGHT): Map<number, number> {
  const score = new Map<number, number>();
  const denseMiss = denseIds.length + 1;
  denseIds.forEach((id, r) => score.set(id, 1 / (RRF_K + r + 1)));
  lexical.forEach(({ id }, r) => {
    const base = score.get(id) ?? 1 / (RRF_K + denseMiss);
    score.set(id, base + lexicalWeight / (RRF_K + r + 1));
  });
  return score;
}

/** Ids ordered by score, ties broken by dense order, then by newer id. */
export function orderByScore(score: ReadonlyMap<number, number>, denseIds: readonly number[] = []): number[] {
  const order = new Map<number, number>();
  denseIds.forEach((id, i) => order.set(id, i));
  return [...score.entries()]
    .sort((a, z) => z[1] - a[1] || (order.get(a[0]) ?? Infinity) - (order.get(z[0]) ?? Infinity) || z[0] - a[0])
    .map(([id]) => id);
}

/** Fused ranking of episode ids, best first. */
export function fuseEpisodeRankings(denseIds: readonly number[], lexical: readonly { id: number }[], lexicalWeight = HYBRID_LEXICAL_WEIGHT): number[] {
  return orderByScore(fuseEpisodeScores(denseIds, lexical, lexicalWeight), denseIds);
}

/** Candidates the cross-encoder reads: the top `RERANK_DEPTH` of each ranking (the configuration benchmarked in ADR-016). */
export const RERANK_DEPTH = 60;
/** Weight of the cross-encoder rank in the fusion; best on LongMemEval_M (ADR-016). */
export const RERANK_WEIGHT = 8;

/**
 * Adds `RERANK_WEIGHT`/(k + cross-encoder rank) to each candidate's fused score, in place. Candidates without text
 * are skipped. Throws if the model cannot be loaded; the caller keeps the hybrid order.
 */
export async function applyCrossEncoder(
  query: string,
  score: Map<number, number>,
  candidates: readonly number[],
  textOf: (id: number) => string | undefined,
  model?: string,
): Promise<void> {
  const ids = candidates.filter(id => score.has(id) && textOf(id));
  const ce = await crossEncoderScores(query, ids.map(id => textOf(id)!), model);
  const byCe = ids.map((_, i) => i).sort((a, z) => ce[z] - ce[a] || a - z);
  byCe.forEach((i, r) => score.set(ids[i], score.get(ids[i])! + RERANK_WEIGHT / (RRF_K + r + 1)));
}
