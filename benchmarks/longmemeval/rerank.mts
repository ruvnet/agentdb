/**
 * Cross-encoder scores for the LongMemEval harness (ADR-016). For each question, the candidate chunks are the union
 * of the top `--depth` chunks by dense score and by BM25 score; each (question, chunk) pair is scored once by a
 * cross-encoder and cached, keyed by sha1(question + U+0001 + chunk).
 *
 *   node --experimental-strip-types benchmarks/longmemeval/rerank.mts --split s [--reranker Xenova/ms-marco-MiniLM-L-6-v2] [--depth 50]
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { streamArray } from './load.mts';
import { keyOf, loadCache, prepare, slug } from './pipeline.mts';
import { ranksOf } from '../../src/search/conversation-retrieval.ts';

export const pairKey = (q: string, chunk: string): string => keyOf(q + '\u0001' + chunk);

/** Indices of the candidate chunks: top `depth` by each ranker. */
export function candidates(dense: ArrayLike<number>, lexical: ArrayLike<number>, depth: number): number[] {
  const rd = ranksOf(dense);
  const rl = ranksOf(lexical, 0);
  const out: number[] = [];
  for (let i = 0; i < rd.length; i++) if (rd[i] <= depth || rl[i] <= depth) out.push(i);
  return out;
}

export function loadPairScores(data: string, reranker: string): Map<string, number> {
  const path = join(data, 'ce', `${slug(reranker)}.tsv`);
  const map = new Map<string, number>();
  if (!existsSync(path)) return map;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const tab = line.indexOf('\t');
    if (tab > 0) map.set(line.slice(0, tab), Number(line.slice(tab + 1)));
  }
  return map;
}

async function main(): Promise<void> {
  const { values: args } = parseArgs({
    options: {
      split: { type: 'string', default: 's' },
      model: { type: 'string', default: 'Xenova/all-MiniLM-L6-v2' },
      reranker: { type: 'string', default: 'Xenova/ms-marco-MiniLM-L-6-v2' },
      depth: { type: 'string', default: '50' },
      device: { type: 'string', default: 'cuda' },
      data: { type: 'string', default: '/data/scratch/longmemeval' },
      batch: { type: 'string', default: '64' },
    },
  });
  const cache = loadCache(args.model!, args.data!);
  mkdirSync(join(args.data!, 'ce'), { recursive: true });
  const out = join(args.data!, 'ce', `${slug(args.reranker!)}.tsv`);
  const have = loadPairScores(args.data!, args.reranker!);
  const { AutoTokenizer, AutoModelForSequenceClassification } = await import('@huggingface/transformers');
  const tokenizer = await AutoTokenizer.from_pretrained(args.reranker!);
  const model = await AutoModelForSequenceClassification.from_pretrained(args.reranker!, { device: args.device as 'cuda', dtype: 'fp32' });
  const depth = Number(args.depth);
  const batch = Number(args.batch);
  const t0 = Date.now();
  let pairs = 0;
  let q = 0;
  for await (const it of streamArray(join(args.data!, `longmemeval_${args.split}_cleaned.json`))) {
    if (it.question_id.endsWith('_abs')) continue;
    q++;
    const p = prepare(it, cache);
    const todo = [...new Set(candidates(p.dense, p.lexical, depth).map(i => p.texts[i]))].filter(t => !have.has(pairKey(it.question, t)));
    for (let i = 0; i < todo.length; i += batch) {
      const docs = todo.slice(i, i + batch);
      const enc = tokenizer(new Array(docs.length).fill(it.question), { text_pair: docs, padding: true, truncation: true, max_length: 512 });
      const { logits } = await model(enc);
      const scores = logits.data as Float32Array;
      const width = logits.dims[1] ?? 1;
      const lines = docs.map((d, j) => {
        const k = pairKey(it.question, d);
        const s = scores[j * width + (width - 1)];
        have.set(k, s);
        return `${k}\t${s}`;
      });
      appendFileSync(out, lines.join('\n') + '\n');
      pairs += docs.length;
    }
    if (q % 50 === 0) console.log(`${q} questions, ${pairs} pairs, ${(pairs / ((Date.now() - t0) / 1000)).toFixed(0)} pairs/s`);
  }
  console.log(`done: ${q} questions, ${pairs} new pairs in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
