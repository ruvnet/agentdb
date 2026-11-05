/**
 * Ecosystem integration benchmark (ADR-018).
 *
 *   npm run benchmark:ecosystem
 *
 * Compares each optional package against the code path AgentDB uses without it:
 *   - @ruvector/mincut-wasm      vs the JavaScript half-split heuristic (cut quality + time)
 *   - @ruvector/graph-transformer vs the JavaScript PPR power iteration (time + max |diff|)
 *   - @ruvector/rabitq-wasm      vs SqlJsRvfBackend's exact top-k scan (time + recall@10)
 */
import { MincutService } from '../src/controllers/MincutService.js';
import { SparsificationService } from '../src/controllers/SparsificationService.js';
import { SqlJsRvfBackend } from '../src/backends/rvf/SqlJsRvfBackend.js';

function rng(seed: number) {
  let s = seed;
  return () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
}

function graph(n: number, extraPerNode: number, seed: number): number[][] {
  const r = rng(seed);
  const adj = Array.from({ length: n }, () => new Set<number>());
  for (let i = 0; i < n; i++) {
    adj[i].add((i + 1) % n);
    adj[(i + 1) % n].add(i);
    for (let k = 0; k < extraPerNode; k++) {
      const j = Math.floor(r() * n);
      if (j !== i) {
        adj[i].add(j);
        adj[j].add(i);
      }
    }
  }
  return adj.map((a) => [...a]);
}

const time = async <T>(fn: () => Promise<T> | T): Promise<[T, number]> => {
  const t0 = performance.now();
  const out = await fn();
  return [out, performance.now() - t0];
};

async function mincut() {
  console.log('\n## Min cut (stoer-wagner): @ruvector/mincut-wasm vs JS half-split');
  const svc = new MincutService({ algorithm: 'stoer-wagner' });
  await svc.initialize();
  console.log(`backend: ${svc.getBackend()}`);
  for (const n of [1000, 2000, 4000]) {
    const edges = graph(n, 2, n);
    const capped = new MincutService({ algorithm: 'stoer-wagner', wasmMaxNodes: Infinity });
    await capped.initialize();
    const [wasm, tw] = await time(() => capped.stoerWagnerMincut(edges));
    const [js, tj] = await time(() => (svc as any).stoerWagnerFallback(edges));
    console.log(`n=${n}: wasm cut=${wasm.cutSize} (${tw.toFixed(1)} ms) | fallback cut=${js.cutSize} (${tj.toFixed(1)} ms)`);
  }
}

async function ppr() {
  console.log('\n## Personalized PageRank: @ruvector/graph-transformer vs JS power iteration');
  const svc = new SparsificationService({ method: 'ppr', topK: 20, maxIterations: 20 });
  await svc.initialize();
  console.log(`runtime: ${svc.getPprRuntime()}`);
  for (const n of [500, 2000, 5000]) {
    const edges = graph(n, 3, n);
    const [nat, tn] = await time(() => svc.pprSparsification(0, edges, 20, 0.15));
    const [js, tj] = await time(() => (svc as any).pprFallback(0, edges, 20, 0.15));
    let diff = 0;
    for (let i = 0; i < n; i++) diff = Math.max(diff, Math.abs(nat.scores[i] - js.scores[i]));
    console.log(`n=${n}: ${nat.method} ${tn.toFixed(2)} ms | ppr-fallback ${tj.toFixed(1)} ms | speedup ${(tj / tn).toFixed(0)}x | max|diff| ${diff.toExponential(1)}`);
  }
}

async function rabitq() {
  console.log('\n## Vector search: SqlJsRvfBackend exact top-k vs quantizedIndex: rabitq (dim 384, cosine)');
  const dim = 384;
  const r = rng(5);
  const centers = Array.from({ length: 64 }, () => Float32Array.from({ length: dim }, () => r() - 0.5));
  const sample = (i: number) => Float32Array.from({ length: dim }, (_, d) => centers[i % 64][d] + 0.35 * (r() - 0.5));
  for (const n of [5000, 20000, 50000]) {
    const vectors = Array.from({ length: n }, (_, i) => sample(i));
    const queries = Array.from({ length: 50 }, (_, i) => sample(i * 7));
    const exact = new SqlJsRvfBackend({ dimension: dim, metric: 'cosine' } as any);
    const quant = new SqlJsRvfBackend({ dimension: dim, metric: 'cosine', quantizedIndex: 'rabitq' } as any);
    await exact.initialize();
    await quant.initialize();
    for (let off = 0; off < n; off += 10000) {
      const batch = vectors.slice(off, off + 10000).map((embedding, i) => ({ id: `v${off + i}`, embedding }));
      await exact.insertBatchAsync(batch);
      await quant.insertBatchAsync(batch);
    }
    const [, buildMs] = await time(() => quant.searchAsync(queries[0], 10)); // first query builds the index
    let te = 0, tq = 0, hits = 0;
    for (const q of queries) {
      const [truth, a] = await time(() => exact.searchAsync(q, 10));
      const [approx, b] = await time(() => quant.searchAsync(q, 10));
      te += a;
      tq += b;
      const ids = new Set(truth.map((x) => x.id));
      hits += approx.filter((x) => ids.has(x.id)).length;
    }
    const Q = queries.length;
    console.log(`n=${n}: exact ${(te / Q).toFixed(2)} ms/q | rabitq ${(tq / Q).toFixed(2)} ms/q (${(te / tq).toFixed(1)}x) | recall@10 ${(hits / (Q * 10)).toFixed(3)} | index build ${buildMs.toFixed(0)} ms`);
    exact.close();
    quant.close();
  }
}

await mincut();
await ppr();
await rabitq();
