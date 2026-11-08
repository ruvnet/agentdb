/**
 * SparsificationService PPR on @ruvector/graph-transformer (ADR-018).
 *
 * Both packages are imported directly first: a missing package fails here
 * instead of passing on the JavaScript fallback.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { GraphTransformer } from '@ruvector/graph-transformer';
import * as gtWasm from 'ruvector-graph-transformer-wasm';
import { SparsificationService } from '../../../src/controllers/SparsificationService.js';
import { loadNapi, loadWasm, toNativeAdjacency } from '../../../src/controllers/integrations/PprNativeAdapter.js';
import { resetSublinearCore } from '../../../src/controllers/integrations/SublinearCoreAdapter.js';
import type { GraphEdges } from '../../../src/types/graph.js';

function randomGraph(n: number, deg: number, seed: number): GraphEdges {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const adj = Array.from({ length: n }, () => new Set<number>());
  for (let i = 0; i < n; i++) {
    adj[i].add((i + 1) % n);
    adj[(i + 1) % n].add(i);
    for (let k = 0; k < deg; k++) {
      const j = Math.floor(rnd() * n);
      if (j !== i) {
        adj[i].add(j);
        adj[j].add(i);
      }
    }
  }
  return adj.map((a) => [...a]);
}

const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

function service(topK = 10) {
  // maxIterations 20 matches the native 20-step power iteration.
  return new SparsificationService({ method: 'ppr', topK, maxIterations: 20, convergenceThreshold: 0 });
}

describe('@ruvector/graph-transformer PPR integration', () => {
  // @ruvnet/sublinear-core is preferred when installed; hide it so these cases reach graph-transformer.
  beforeAll(() => resetSublinearCore(null));
  afterAll(() => resetSublinearCore());

  it('exposes pprScores on the NAPI class and ppr_scores on the WASM class', () => {
    expect(typeof new GraphTransformer().pprScores).toBe('function');
    expect(typeof gtWasm.JsGraphTransformer).toBe('function');
  });

  it('SparsificationService uses the NAPI backend', async () => {
    const svc = service();
    await svc.initialize();
    expect(svc.getPprRuntime()).toBe('napi');
    const result = await svc.sparsify(0, randomGraph(50, 3, 1));
    expect(result.method).toBe('ppr-napi');
    expect(result.scores).toBeInstanceOf(Float32Array);
    expect(result.topKIndices[0]).toBe(0);
  });

  it('matches the JavaScript power iteration on graphs without sinks', async () => {
    const svc = service(15);
    await svc.initialize();
    for (const seed of [1, 2, 3, 4, 5]) {
      const edges = randomGraph(120, 3, seed);
      const source = seed * 7;
      const native = await svc.pprSparsification(source, edges, 15, 0.15);
      const js = (svc as any).pprFallback(source, edges, 15, 0.15);
      expect(native.method).toBe('ppr-napi');
      expect(js.method).toBe('ppr-fallback');
      expect(maxAbsDiff(native.scores, js.scores)).toBeLessThan(1e-5);
      expect(native.topKIndices.slice(0, 5)).toEqual(js.topKIndices.slice(0, 5));
    }
  });

  it('WASM variant gives the same scores as NAPI', async () => {
    const napi = await loadNapi();
    const wasm = await loadWasm();
    expect(napi?.runtime).toBe('napi');
    expect(wasm?.runtime).toBe('wasm');
    const adj = toNativeAdjacency(randomGraph(200, 4, 9));
    expect(maxAbsDiff(napi!.scores(3, adj, 0.15), wasm!.scores(3, adj, 0.15))).toBeLessThan(1e-12);
  });

  it('cleans holes and out-of-range ids before calling native code', async () => {
    const edges: GraphEdges = [[1, 99], undefined, [0, -1, 1]];
    expect(toNativeAdjacency(edges)).toEqual([[1], [], [0, 1]]);
    const result = await service(2).pprSparsification(0, edges, 2, 0.15);
    expect(result.method).toBe('ppr-napi');
    expect(Array.from(result.scores).every(Number.isFinite)).toBe(true);
  });

  it('falls back to JavaScript for an out-of-range source', async () => {
    const result = await service(2).pprSparsification(10, [[1], [0]], 2, 0.15);
    expect(result.method).toBe('ppr-fallback');
  });

  // The JavaScript fallback is now O(iterations * (n + edges)) like the native code,
  // so a speed race between them is noise; agreement on a large graph is what matters.
  it('matches the JavaScript fallback on a 2,000-node graph without sinks', async () => {
    const svc = service(20);
    await svc.initialize();
    const edges = randomGraph(2000, 3, 11);
    const native = await svc.pprSparsification(0, edges, 20, 0.15);
    const js = (svc as any).pprFallback(0, edges, 20, 0.15);
    expect(native.method).toBe('ppr-napi');
    expect(maxAbsDiff(native.scores, js.scores)).toBeLessThan(1e-5);
  });
});
