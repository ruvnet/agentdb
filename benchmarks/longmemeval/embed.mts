/**
 * Embeds every LongMemEval chunk and question once and caches the vectors on disk (ADR-016), so retrieval
 * configurations can be compared without re-embedding. Same call agentdb's EmbeddingService makes
 * (`feature-extraction`, mean pooling, L2-normalised); only the device differs (CUDA when available).
 *
 *   node --experimental-strip-types benchmarks/longmemeval/embed.mts --split s [--model Xenova/all-MiniLM-L6-v2] [--device cuda]
 *
 * Cache: <dir>/<model-slug>/{keys.txt, vectors.f32, meta.json}; keys are sha1(prefix+text) prefixes, one per row.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync, openSync, writeSync, closeSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { streamArray, windows } from './load.mts';
import { keyOf, slug } from './pipeline.mts';

const { values: args } = parseArgs({
  strict: false,
  options: {
    split: { type: 'string', default: 's' },
    model: { type: 'string', default: 'Xenova/all-MiniLM-L6-v2' },
    device: { type: 'string', default: 'cuda' },
    dtype: { type: 'string', default: 'fp32' },
    data: { type: 'string', default: '/data/scratch/longmemeval' },
    batch: { type: 'string', default: '256' },
    'query-prefix': { type: 'string', default: '' },
    'doc-prefix': { type: 'string', default: '' },
  },
});


async function main(): Promise<void> {
  const dir = join(args.data!, 'emb', slug(args.model!));
  mkdirSync(dir, { recursive: true });
  const keysPath = join(dir, 'keys.txt');
  const vecPath = join(dir, 'vectors.f32');
  const have = new Set(existsSync(keysPath) ? readFileSync(keysPath, 'utf8').split('\n').filter(Boolean) : []);

  const todo = new Map<string, string>();
  const add = (text: string, prefix: string): void => {
    const k = keyOf(prefix + text);
    if (!have.has(k) && !todo.has(k)) todo.set(k, prefix + text);
  };
  for await (const it of streamArray(join(args.data!, `longmemeval_${args.split}_cleaned.json`))) {
    add(it.question, args['query-prefix']!);
    for (const session of it.haystack_sessions) for (const t of session) for (const w of windows(t.content)) add(w, args['doc-prefix']!);
  }
  console.log(`${args.model}: ${have.size} cached, ${todo.size} to embed`);
  if (todo.size === 0) return;

  const { pipeline } = await import('@huggingface/transformers');
  const extractor = await pipeline('feature-extraction', args.model!, { device: args.device as 'cuda', dtype: args.dtype as 'fp32' });
  const entries = [...todo.entries()].sort((a, z) => a[1].length - z[1].length); // length-sorted batches pad less
  const batch = Number(args.batch);
  const fd = openSync(vecPath, 'a');
  const t0 = Date.now();
  let dim = 0;
  // Batch by an estimated token budget: attention memory grows with batch x length^2, so long chunks get small batches.
  const tokenBudget = batch * 64;
  for (let i = 0, step = 0; i < entries.length; i += step) {
    const estTokens = Math.min(512, Math.ceil(entries[Math.min(i + batch, entries.length) - 1][1].length / 3.5));
    step = Math.max(8, Math.min(batch, Math.floor(tokenBudget / estTokens)));
    const slice = entries.slice(i, i + step);
    const out = await extractor(slice.map(e => e[1]), { pooling: 'mean', normalize: true });
    const data = out.data as Float32Array;
    dim = out.dims[1];
    writeSync(fd, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
    appendFileSync(keysPath, slice.map(e => e[0]).join('\n') + '\n');
    if (Math.floor(i / 20000) !== Math.floor((i + slice.length) / 20000)) {
      const done = i + slice.length;
      console.log(`${done}/${entries.length}  ${(done / ((Date.now() - t0) / 1000)).toFixed(0)}/s`);
    }
  }
  closeSync(fd);
  writeFileSync(join(dir, 'meta.json'), JSON.stringify({ model: args.model, dim, dtype: args.dtype, device: args.device }, null, 2));
  console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s, dim ${dim}`);
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
