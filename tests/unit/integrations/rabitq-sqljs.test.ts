/**
 * SqlJsRvfBackend with the opt-in @ruvector/rabitq-wasm candidate index (ADR-018).
 *
 * The package is imported directly first: if it is missing this file fails
 * instead of passing on the exact scan.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { createRequire } from 'module';
import { SqlJsRvfBackend } from '../../../src/backends/rvf/SqlJsRvfBackend.js';

const require = createRequire(import.meta.url);
const { RabitqIndex } = require('@ruvector/rabitq-wasm/node/ruvector_rabitq_wasm.js');

const DIM = 64;
const N = 3000;

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
}

const CENTERS = (() => {
  const rnd = rng(1);
  return Array.from({ length: 32 }, () => Float32Array.from({ length: DIM }, rnd));
})();

/** Clustered vectors; queries drawn the same way are "in distribution". */
function dataset(n: number, seed: number, centers = CENTERS): Float32Array[] {
  const rnd = rng(seed);
  return Array.from({ length: n }, (_, i) =>
    Float32Array.from({ length: DIM }, (_, d) => centers[i % 32][d] + 0.35 * rnd())
  );
}

async function backend(extra: Record<string, unknown> = {}, metric: 'cosine' | 'l2' | 'ip' = 'cosine') {
  const b = new SqlJsRvfBackend({ dimension: DIM, metric, ...extra } as any);
  await b.initialize();
  return b;
}

async function fill(b: SqlJsRvfBackend, vectors: Float32Array[]) {
  await b.insertBatchAsync(vectors.map((embedding, i) => ({ id: `v${i}`, embedding, metadata: { group: i % 2 } })));
}

describe('@ruvector/rabitq-wasm in SqlJsRvfBackend', () => {
  const vectors = dataset(N, 3);
  const queries = dataset(40, 99);
  let exact: SqlJsRvfBackend;
  let quantized: SqlJsRvfBackend;

  beforeAll(async () => {
    exact = await backend();
    quantized = await backend({ quantizedIndex: 'rabitq', rabitq: { minVectors: 500 } });
    await fill(exact, vectors);
    await fill(quantized, vectors);
  });

  it('exposes RabitqIndex.build/search', () => {
    expect(typeof RabitqIndex.build).toBe('function');
  });

  it('is off unless requested', async () => {
    expect(exact.getQuantizedIndexStats()).toBeNull();
    expect(quantized.getQuantizedIndexStats()?.enabled).toBe(true);
  });

  it('recall@10 >= 0.95 against the exact scan, with exact similarities', async () => {
    let hits = 0;
    for (const q of queries) {
      const truth = await exact.searchAsync(q, 10);
      const approx = await quantized.searchAsync(q, 10);
      const truthIds = new Set(truth.map((r) => r.id));
      hits += approx.filter((r) => truthIds.has(r.id)).length;
      const bySim = new Map(truth.map((r) => [r.id, r.similarity]));
      for (const r of approx) if (bySim.has(r.id)) expect(r.similarity).toBeCloseTo(bySim.get(r.id)!, 6);
    }
    expect(hits / (queries.length * 10)).toBeGreaterThanOrEqual(0.95);
    const stats = quantized.getQuantizedIndexStats()!;
    expect(stats.indexed).toBe(N);
    expect(stats.builds).toBe(1);
  });

  it('recall@10 >= 0.9 for queries unrelated to the data (near-tied neighbours)', async () => {
    const rnd = rng(77);
    let hits = 0;
    for (let i = 0; i < 40; i++) {
      const q = Float32Array.from({ length: DIM }, rnd);
      const truth = new Set((await exact.searchAsync(q, 10)).map((r) => r.id));
      hits += (await quantized.searchAsync(q, 10)).filter((r) => truth.has(r.id)).length;
    }
    expect(hits / 400).toBeGreaterThanOrEqual(0.9);
  });

  it('sees writes and removals made after the index was built', async () => {
    const fresh = Float32Array.from({ length: DIM }, (_, d) => (d % 2 ? 5 : -5));
    await quantized.insertAsync('fresh', fresh);
    expect((await quantized.searchAsync(fresh, 1))[0].id).toBe('fresh');

    const moved = Float32Array.from({ length: DIM }, (_, d) => (d % 3 ? -4 : 4));
    await quantized.insertAsync('v7', moved); // update in place
    expect((await quantized.searchAsync(moved, 1))[0].id).toBe('v7');

    await quantized.removeAsync('fresh');
    expect((await quantized.searchAsync(fresh, 5)).map((r) => r.id)).not.toContain('fresh');
    expect(quantized.getQuantizedIndexStats()!.builds).toBe(1);
  });

  it('rebuilds once enough vectors change', async () => {
    const more = dataset(400, 7);
    await quantized.insertBatchAsync(more.map((embedding, i) => ({ id: `m${i}`, embedding })));
    await quantized.searchAsync(queries[0], 10);
    const stats = quantized.getQuantizedIndexStats()!;
    expect(stats.builds).toBe(2);
    expect(stats.dirty).toBe(0);
  });

  it('keeps filtered queries exact', async () => {
    const q = queries[1];
    const a = await quantized.searchAsync(q, 10, { filter: { group: 1 } });
    const b = await exact.searchAsync(q, 10, { filter: { group: 1 } });
    expect(a.map((r) => r.id)).toEqual(b.map((r) => r.id));
  });

  it('skips the index below minVectors and for the ip metric', async () => {
    const small = await backend({ quantizedIndex: 'rabitq' });
    await fill(small, vectors.slice(0, 100));
    await small.searchAsync(queries[0], 5);
    expect(small.getQuantizedIndexStats()!.indexed).toBe(0);

    const ip = await backend({ quantizedIndex: 'rabitq' }, 'ip');
    expect(ip.getQuantizedIndexStats()).toBeNull();
  });

  it('supports the l2 metric', async () => {
    const l2Exact = await backend({}, 'l2');
    const l2Quant = await backend({ quantizedIndex: 'rabitq', rabitq: { minVectors: 500 } }, 'l2');
    await fill(l2Exact, vectors);
    await fill(l2Quant, vectors);
    let hits = 0;
    for (const q of queries.slice(0, 10)) {
      const truth = new Set((await l2Exact.searchAsync(q, 10)).map((r) => r.id));
      hits += (await l2Quant.searchAsync(q, 10)).filter((r) => truth.has(r.id)).length;
    }
    expect(hits / 100).toBeGreaterThanOrEqual(0.95);
  });
});
