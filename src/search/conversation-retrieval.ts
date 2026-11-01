/**
 * Conversation-memory retrieval: hybrid dense + BM25 ranking of chunks, aggregated to the memory unit a caller
 * asks for (a session, an episode). ADR-016.
 *
 * Pure functions over plain arrays, so the MCP search path and the LongMemEval harness run the same code.
 * Every input that can come from a tool call is bounded: text is cut before tokenising, the token count per
 * document and per query is capped, and the tokeniser regex has no nesting (linear time).
 */

/** Limits applied to untrusted text before it is tokenised. */
export const RETRIEVAL_LIMITS = Object.freeze({
  maxCharsPerText: 32_768,
  maxTokensPerDoc: 4_096,
  maxQueryTokens: 256,
});

const STOPWORDS = new Set(
  ('a an and are as at be been but by can could did do does doing for from had has have having he her hers him his how i if in into is it its ' +
    'me my myself no nor not of off on once only or other our ours out over own same she should so some such than that the their theirs them ' +
    'then there these they this those through to too under until up very was we were what when where which while who whom why will with would you ' +
    'your yours yourself just also about any all am again against because before being below between both down during each few further here more ' +
    'most s t don now ll re ve d m o y')
    .split(' '),
);

/** Lower-cased word tokens without stopwords, capped at `maxTokens`. Linear in the (already capped) input. */
export function tokenize(text: string, maxTokens: number = RETRIEVAL_LIMITS.maxTokensPerDoc): string[] {
  const src = text.length > RETRIEVAL_LIMITS.maxCharsPerText ? text.slice(0, RETRIEVAL_LIMITS.maxCharsPerText) : text;
  const out: string[] = [];
  for (const m of src.toLowerCase().matchAll(/[\p{L}\p{N}]+/gu)) {
    const tok = m[0];
    if (tok.length < 2 && !/\d/.test(tok)) continue;
    if (STOPWORDS.has(tok)) continue;
    out.push(tok.length > 6 ? lightStem(tok) : tok);
    if (out.length >= maxTokens) break;
  }
  return out;
}

/** A small suffix stripper: plural and common verb endings only, so "hiking"/"hikes"/"hiked" share a term. */
function lightStem(tok: string): string {
  if (tok.endsWith('ies')) return tok.slice(0, -3) + 'y';
  if (tok.endsWith('ing')) return tok.slice(0, -3);
  if (tok.endsWith('ed')) return tok.slice(0, -2);
  if (tok.endsWith('es')) return tok.slice(0, -2);
  if (tok.endsWith('s') && !tok.endsWith('ss')) return tok.slice(0, -1);
  return tok;
}

/** Okapi BM25 over a document set that can grow by appending (`add`). */
export class Bm25Index {
  private readonly postings = new Map<string, { doc: number; tf: number }[]>();
  private readonly lengths: number[] = [];
  private total = 0;
  private readonly k1: number;
  private readonly b: number;

  constructor(docs: readonly string[][] = [], k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    for (const tokens of docs) this.add(tokens);
  }

  get size(): number {
    return this.lengths.length;
  }

  /** Appends a document; its index is the previous `size`. */
  add(tokens: readonly string[]): number {
    const doc = this.lengths.length;
    this.lengths.push(tokens.length);
    this.total += tokens.length;
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [term, n] of tf) {
      let list = this.postings.get(term);
      if (!list) this.postings.set(term, (list = []));
      list.push({ doc, tf: n });
    }
    return doc;
  }

  /** BM25 score of every document for the query tokens (0 where no term matches). */
  score(query: readonly string[]): Float64Array {
    const n = this.lengths.length;
    const scores = new Float64Array(n);
    const avgLen = n ? this.total / n || 1 : 1;
    for (const term of new Set(query.slice(0, RETRIEVAL_LIMITS.maxQueryTokens))) {
      const list = this.postings.get(term);
      if (!list) continue;
      const idf = Math.log(1 + (n - list.length + 0.5) / (list.length + 0.5));
      for (const { doc, tf } of list) {
        const norm = tf + this.k1 * (1 - this.b + (this.b * this.lengths[doc]) / avgLen);
        scores[doc] += idf * ((tf * (this.k1 + 1)) / norm);
      }
    }
    return scores;
  }
}

/** Dot product of a query vector with each row; equal to cosine when all vectors are L2-normalised. */
export function dotScores(query: ArrayLike<number>, rows: readonly ArrayLike<number>[]): Float64Array {
  const out = new Float64Array(rows.length);
  const d = query.length;
  for (let r = 0; r < rows.length; r++) {
    const row = rows[r];
    if (row.length !== d) throw new Error(`dimension mismatch: query ${d}, row ${r} ${row.length}`);
    let s = 0;
    for (let i = 0; i < d; i++) s += query[i] * row[i];
    out[r] = s;
  }
  return out;
}

/** 1-based rank of each entry under a descending sort; entries that must not count (score <= floor) get rank Infinity. */
export function ranksOf(scores: ArrayLike<number>, floor = -Infinity): Float64Array {
  const idx = Array.from({ length: scores.length }, (_, i) => i).filter(i => scores[i] > floor);
  idx.sort((a, z) => scores[z] - scores[a] || a - z);
  const ranks = new Float64Array(scores.length).fill(Infinity);
  idx.forEach((doc, r) => (ranks[doc] = r + 1));
  return ranks;
}

export interface HybridOptions {
  /** RRF constant (default 60). */
  rrfK?: number;
  /** Weight of the dense ranking in the fusion (default 1). */
  denseWeight?: number;
  /** Weight of the BM25 ranking in the fusion (default 1). 0 gives dense-only ordering. */
  lexicalWeight?: number;
}

/** Weighted reciprocal-rank fusion of a dense and a lexical score list over the same documents. */
export function fuseHybrid(dense: ArrayLike<number>, lexical: ArrayLike<number> | null, opts: HybridOptions = {}): Float64Array {
  const k = opts.rrfK ?? 60;
  const wd = opts.denseWeight ?? 1;
  const wl = lexical ? opts.lexicalWeight ?? 1 : 0;
  const rd = ranksOf(dense);
  const rl = lexical && wl > 0 ? ranksOf(lexical, 0) : null;
  const out = new Float64Array(dense.length);
  for (let i = 0; i < dense.length; i++) {
    out[i] = wd / (k + rd[i]) + (rl ? wl / (k + rl[i]) : 0);
  }
  return out;
}

export interface GroupOptions {
  /** How chunk scores become a group score: the best chunk (`max`), or best plus a decayed sum of the next ones (`max-plus`). */
  aggregation?: 'max' | 'max-plus';
  /** Weight of each further chunk in `max-plus` (default 0.1, halving each step). */
  tailWeight?: number;
}

/** Rank groups (sessions, episodes) by the scores of their chunks. Ties keep first-seen order. */
export function rankGroups(groupOf: readonly string[], scores: ArrayLike<number>, opts: GroupOptions = {}): { group: string; score: number }[] {
  const per = new Map<string, number[]>();
  const order: string[] = [];
  groupOf.forEach((g, i) => {
    let list = per.get(g);
    if (!list) {
      per.set(g, (list = []));
      order.push(g);
    }
    list.push(scores[i]);
  });
  const tail = opts.tailWeight ?? 0.1;
  const ranked = order.map((group, firstSeen) => {
    const s = per.get(group)!.sort((a, z) => z - a);
    let score = s[0];
    if (opts.aggregation === 'max-plus') for (let i = 1, w = tail; i < s.length && i < 4; i++, w /= 2) score += w * s[i];
    return { group, score, firstSeen };
  });
  ranked.sort((a, z) => z.score - a.score || a.firstSeen - z.firstSeen);
  return ranked.map(({ group, score }) => ({ group, score }));
}
