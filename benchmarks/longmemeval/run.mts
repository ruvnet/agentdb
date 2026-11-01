/**
 * LongMemEval session-level retrieval benchmark for agentdb's conversation retrieval (ADR-016).
 *
 * For each question, the haystack's turns are chunked (load.ts `windows`), scored with the same functions agentdb's
 * search path uses (src/search/conversation-retrieval.ts), aggregated to sessions, and scored against
 * `answer_session_ids`. Metrics follow the LongMemEval paper and code: session-level recall_any@k, recall_all@k and
 * NDCG@k (binary relevance), k in {5, 10}; the 30 `_abs` questions are skipped, as the official retrieval eval does.
 *
 *   node --experimental-strip-types benchmarks/longmemeval/run.mts --split s [--model Xenova/all-MiniLM-L6-v2] [--json out.json]
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { streamArray } from './load.mts';
import { loadCache, prepare, type Prepared } from './pipeline.mts';
import { candidates, loadPairScores, pairKey } from './rerank.mts';
import type { Instance } from './load.mts';
import { Bm25Index, tokenize, fuseHybrid, rankGroups, ranksOf, type GroupOptions, type HybridOptions } from '../../src/search/conversation-retrieval.ts';

const { values: args } = parseArgs({
  options: {
    split: { type: 'string', default: 's' },
    model: { type: 'string', default: 'Xenova/all-MiniLM-L6-v2' },
    data: { type: 'string', default: '/data/scratch/longmemeval' },
    'query-prefix': { type: 'string', default: '' },
    'doc-prefix': { type: 'string', default: '' },
    json: { type: 'string' },
    limit: { type: 'string' },
    reranker: { type: 'string', default: 'Xenova/ms-marco-MiniLM-L-6-v2' },
    depth: { type: 'string', default: '50' },
    only: { type: 'string' },
    'include-abs': { type: 'boolean', default: false },
  },
});

export interface Ctx { it: Instance; p: Prepared; ce: Map<string, number> }
export interface Config { name: string; group: GroupOptions; score(c: Ctx): Float64Array }

const hybrid = (h: HybridOptions) => (c: Ctx): Float64Array => fuseHybrid(c.p.dense, c.p.lexical, h);

/** Hybrid RRF plus a cross-encoder rank over the candidate chunks (weight `w`). Chunks without a CE score keep their hybrid score. */
const reranked = (h: HybridOptions, w: number, depth: number) => (c: Ctx): Float64Array => {
  const base = fuseHybrid(c.p.dense, c.p.lexical, h);
  const cand = candidates(c.p.dense, c.p.lexical, depth);
  const ce = new Float64Array(c.p.texts.length).fill(-Infinity);
  for (const i of cand) {
    const s = c.ce.get(pairKey(c.it.question, c.p.texts[i]));
    if (s !== undefined) ce[i] = s;
  }
  const r = ranksOf(ce, -Infinity);
  const k = h.rrfK ?? 60;
  return base.map((b, i) => b + (Number.isFinite(r[i]) ? w / (k + r[i]) : 0));
};

/** Adds a session-level BM25 ranking (whole session as one document) as a third RRF term on every chunk. */
const withSessionBm25 = (inner: (c: Ctx) => Float64Array, w: number) => (c: Ctx): Float64Array => {
  const s = inner(c);
  const ids = [...new Set(c.p.groupOf)];
  const text = new Map<string, string[]>();
  c.p.groupOf.forEach((g, i) => (text.get(g) ?? text.set(g, []).get(g)!).push(c.p.texts[i]));
  const sess = new Bm25Index(ids.map(id => tokenize(text.get(id)!.join(' '), 20_000))).score(tokenize(c.it.question));
  const r = ranksOf(sess, 0);
  const rankOf = new Map(ids.map((id, j) => [id, r[j]]));
  return s.map((v, i) => v + (Number.isFinite(rankOf.get(c.p.groupOf[i])!) ? w / (60 + rankOf.get(c.p.groupOf[i])!) : 0));
};

const DEPTH = Number(args.depth);
export const CONFIGS: Config[] = [
  { name: 'dense (agentdb 3.1.0 exact search, turn chunks, max)', group: { aggregation: 'max' }, score: hybrid({ lexicalWeight: 0 }) },
  { name: 'bm25 only', group: { aggregation: 'max' }, score: c => c.p.lexical },
  { name: 'hybrid rrf 1:1 max', group: { aggregation: 'max' }, score: hybrid({ lexicalWeight: 1 }) },
  { name: 'hybrid rrf 1:0.5 max', group: { aggregation: 'max' }, score: hybrid({ lexicalWeight: 0.5 }) },
  { name: 'hybrid rrf 1:1 max-plus', group: { aggregation: 'max-plus' }, score: hybrid({ lexicalWeight: 1 }) },
  { name: 'hybrid rrf 1:0.5 max-plus', group: { aggregation: 'max-plus' }, score: hybrid({ lexicalWeight: 0.5 }) },
  { name: 'hybrid 1:1 + CE x1', group: { aggregation: 'max' }, score: reranked({ lexicalWeight: 1 }, 1, DEPTH) },
  { name: 'hybrid 1:1 + CE x2', group: { aggregation: 'max' }, score: reranked({ lexicalWeight: 1 }, 2, DEPTH) },
  { name: 'hybrid 1:1 + CE x4', group: { aggregation: 'max' }, score: reranked({ lexicalWeight: 1 }, 4, DEPTH) },
  // The configuration agentdb ships for agentdb_search rerank:true (depth 60, weight 8): src/controllers/episode-hybrid.ts.
  { name: 'hybrid 1:1 + CE x8', group: { aggregation: 'max' }, score: reranked({ lexicalWeight: 1 }, 8, DEPTH) },
  { name: 'hybrid 1:1 + CE x16', group: { aggregation: 'max' }, score: reranked({ lexicalWeight: 1 }, 16, DEPTH) },
  { name: 'hybrid 1:1 + session bm25 x1', group: { aggregation: 'max' }, score: withSessionBm25(hybrid({ lexicalWeight: 1 }), 1) },
  { name: 'hybrid 1:1 + session bm25 x0.5', group: { aggregation: 'max' }, score: withSessionBm25(hybrid({ lexicalWeight: 1 }), 0.5) },
  { name: 'hybrid+sess + CE x8', group: { aggregation: 'max' }, score: withSessionBm25(reranked({ lexicalWeight: 1 }, 8, DEPTH), 1) },
  { name: 'hybrid+sess + CE x16', group: { aggregation: 'max' }, score: withSessionBm25(reranked({ lexicalWeight: 1 }, 16, DEPTH), 1) },
  { name: 'hybrid 1:1 + CE x2 max-plus', group: { aggregation: 'max-plus' }, score: reranked({ lexicalWeight: 1 }, 2, DEPTH) },
];

const KS = [5, 10] as const;
type Metrics = Record<string, number>;

export function sessionMetrics(ranked: string[], gold: Set<string>): Metrics {
  const m: Metrics = {};
  for (const k of KS) {
    const top = ranked.slice(0, k);
    const hits = top.filter(s => gold.has(s)).length;
    m[`recall_any@${k}`] = hits > 0 ? 1 : 0;
    m[`recall_all@${k}`] = hits === gold.size ? 1 : 0;
    let dcg = 0;
    top.forEach((s, i) => gold.has(s) && (dcg += 1 / Math.log2(i + 2)));
    let idcg = 0;
    for (let i = 0; i < Math.min(gold.size, k); i++) idcg += 1 / Math.log2(i + 2);
    m[`ndcg@${k}`] = idcg ? dcg / idcg : 0;
  }
  return m;
}

async function main(): Promise<void> {
  const cache = loadCache(args.model!, args.data!);
  const ce = loadPairScores(args.data!, args.reranker!);
  const configs = args.only ? CONFIGS.filter(c => new RegExp(args.only!).test(c.name)) : CONFIGS;
  const usesCe = configs.some(c => c.name.includes('CE'));
  if (usesCe && ce.size === 0) throw new Error('no cross-encoder scores; run rerank.mts first or pass --only');
  const sums = configs.map(() => ({ all: {} as Metrics, byType: new Map<string, { n: number; m: Metrics }>() }));
  let n = 0;
  const limit = args.limit ? Number(args.limit) : Infinity;
  const t0 = Date.now();
  for await (const it of streamArray(join(args.data!, `longmemeval_${args.split}_cleaned.json`))) {
    if (it.question_id.endsWith('_abs') && !args['include-abs']) continue;
    if (it.answer_session_ids.length === 0) continue;
    if (n >= limit) break;
    n++;
    const p = prepare(it, cache, args['query-prefix']!, args['doc-prefix']!);
    const gold = new Set(it.answer_session_ids);
    configs.forEach((cfg, c) => {
      const ranked = rankGroups(p.groupOf, cfg.score({ it, p, ce }), cfg.group).map(g => g.group);
      const m = sessionMetrics(ranked, gold);
      const acc = sums[c];
      for (const [k, v] of Object.entries(m)) acc.all[k] = (acc.all[k] ?? 0) + v;
      let t = acc.byType.get(it.question_type);
      if (!t) acc.byType.set(it.question_type, (t = { n: 0, m: {} }));
      t.n++;
      for (const [k, v] of Object.entries(m)) t.m[k] = (t.m[k] ?? 0) + v;
    });
  }
  const report = {
    split: args.split, model: args.model, questions: n, seconds: (Date.now() - t0) / 1000,
    configs: configs.map((cfg, c) => ({
      name: cfg.name,
      metrics: Object.fromEntries(Object.entries(sums[c].all).map(([k, v]) => [k, +(v / n).toFixed(4)])),
      byType: Object.fromEntries([...sums[c].byType].map(([t, { n: tn, m }]) => [t, { n: tn, ...Object.fromEntries(Object.entries(m).map(([k, v]) => [k, +(v / tn).toFixed(3)])) }])),
    })),
  };
  console.log(`${report.split} ${report.model} n=${n} (${report.seconds.toFixed(0)} s)`);
  console.log('config'.padEnd(58), 'R_any@5 R_all@5 NDCG@5  R_any@10 R_all@10 NDCG@10');
  for (const c of report.configs) {
    const m = c.metrics;
    console.log(c.name.padEnd(58), [m['recall_any@5'], m['recall_all@5'], m['ndcg@5'], m['recall_any@10'], m['recall_all@10'], m['ndcg@10']].map(v => v.toFixed(3).padStart(7)).join(' '));
  }
  if (args.json) writeFileSync(args.json, JSON.stringify(report, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
