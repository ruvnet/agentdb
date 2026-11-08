/**
 * SparsificationService PPR on @ruvnet/sublinear-core, and the in-tree fallback.
 *
 * The package is imported directly first: if it is missing this file fails
 * instead of passing on the fallback.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { createRequire } from 'module';
import * as sublinearCore from '@ruvnet/sublinear-core';
import { SparsificationService } from '../../../src/controllers/SparsificationService.js';
import { resetPprBackend } from '../../../src/controllers/integrations/PprNativeAdapter.js';
import {
  loadSublinearCore,
  personalizedPageRankInTree,
  resetSublinearCore,
} from '../../../src/controllers/integrations/SublinearCoreAdapter.js';

const require = createRequire(import.meta.url);

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Directed graph, out-degree 1..9, `sinkRate` of the nodes have no out-edges. */
function directedGraph(n: number, seed: number, sinkRate = 0.1): number[][] {
  const r = rng(seed);
  const g: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    if (r() >= sinkRate) {
      const deg = 1 + Math.floor(r() * 9);
      for (let k = 0; k < deg; k++) row.push(Math.floor(r() * n));
    }
    g.push(row);
  }
  return g;
}

/** Independent reference: pull over in-edges, dangling mass to the source, L1 stop. */
function referencePpr(adj: number[][], src: number, alpha: number, maxIt: number, tol: number) {
  const n = adj.length;
  const inEdges: Array<Array<[number, number]>> = Array.from({ length: n }, () => []);
  for (let u = 0; u < n; u++) for (const v of adj[u]) inEdges[v].push([u, 1 / adj[u].length]);
  let x = new Float64Array(n);
  x[src] = 1;
  for (let it = 0; it < maxIt; it++) {
    let dangling = 0;
    for (let u = 0; u < n; u++) if (adj[u].length === 0) dangling += x[u];
    const y = new Float64Array(n);
    let delta = 0;
    for (let v = 0; v < n; v++) {
      let s = 0;
      for (const [u, w] of inEdges[v]) s += x[u] * w;
      y[v] = (1 - alpha) * s + (v === src ? alpha + (1 - alpha) * dangling : 0);
      delta += Math.abs(y[v] - x[v]);
    }
    x = y;
    if (delta <= tol) break;
  }
  return x;
}

/** The pre-3.7.1 JavaScript fallback, kept here only to pin the bug it had. */
function oldFallback(src: number, edges: number[][], alpha: number, maxIt = 20, threshold = 1e-6) {
  const n = edges.length;
  const scores = new Float32Array(n);
  scores[src] = 1.0;
  for (let iter = 0; iter < maxIt; iter++) {
    const next = new Float32Array(n);
    let maxDelta = 0;
    for (let i = 0; i < n; i++) {
      next[i] = alpha * (i === src ? 1.0 : 0.0);
      for (let j = 0; j < n; j++) {
        const nb = edges[j] || [];
        if (nb.includes(i) && nb.length > 0) next[i] += ((1 - alpha) * scores[j]) / nb.length;
      }
      maxDelta = Math.max(maxDelta, Math.abs(next[i] - scores[i]));
    }
    scores.set(next);
    if (maxDelta < threshold) break;
  }
  return scores;
}

const sum = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
};
const maxAbsDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

const TIGHT = { maxIterations: 2000, tolerance: 1e-14 };

afterEach(() => {
  resetSublinearCore();
  resetPprBackend();
});

describe('@ruvnet/sublinear-core PPR (real import)', () => {
  it('is version 1.x with the PageRank exports', () => {
    const version: string = require('@ruvnet/sublinear-core/package.json').version;
    expect(version).toMatch(/^1\./);
    expect(typeof sublinearCore.personalizedPageRank).toBe('function');
    expect(typeof sublinearCore.graphFromAdjacency).toBe('function');
  });

  it('SparsificationService prefers sublinear-core', async () => {
    const svc = new SparsificationService({ method: 'ppr', topK: 5 });
    await svc.initialize();
    expect(svc.getPprRuntime()).toBe('sublinear-core');
    const result = await svc.sparsify(0, directedGraph(200, 1));
    expect(result.method).toBe('ppr-sublinear-core');
    expect(result.scores).toBeInstanceOf(Float32Array);
    expect(result.metadata?.iterations).toBeGreaterThan(0);
  });

  it('scores sum to 1 and match power iteration within 1e-9 on graphs with sinks', async () => {
    const core = await loadSublinearCore();
    expect(core?.runtime).toBe('sublinear-core');
    for (const seed of [1, 2, 3]) {
      const adj = directedGraph(1000, seed, 0.15);
      const run = core!.run(seed * 11, adj, 0.15, TIGHT);
      expect(run.converged).toBe(true);
      expect(Math.abs(sum(run.scores) - 1)).toBeLessThan(1e-12);
      expect(maxAbsDiff(run.scores, referencePpr(adj, seed * 11, 0.15, 2000, 1e-14))).toBeLessThan(1e-9);
    }
  });

  it('public Float32 scores sum to 1 through the service at the default budget', async () => {
    const svc = new SparsificationService({ method: 'ppr', topK: 10 });
    const result = await svc.pprSparsification(3, directedGraph(1000, 4, 0.2), 10, 0.15);
    expect(result.method).toBe('ppr-sublinear-core');
    expect(Math.abs(sum(result.scores) - 1)).toBeLessThan(1e-5);
  });

  it('accepts holes and out-of-range neighbour ids', async () => {
    const svc = new SparsificationService({ method: 'ppr', topK: 2 });
    const result = await svc.pprSparsification(0, [[1, 99], undefined, [0, -1, 1]] as any, 2, 0.15);
    expect(result.method).toBe('ppr-sublinear-core');
    expect(Math.abs(sum(result.scores) - 1)).toBeLessThan(1e-6);
  });
});

describe('in-tree PPR fallback (sublinear-core absent)', () => {
  it('the loader returns null when the package does not resolve or lacks the exports', async () => {
    resetSublinearCore(null);
    expect(await loadSublinearCore()).toBeNull();
    resetSublinearCore({});
    expect(await loadSublinearCore()).toBeNull();
    const svc = new SparsificationService({ method: 'ppr', topK: 3 });
    await svc.initialize();
    expect(svc.getPprRuntime()).not.toBe('sublinear-core');
  });

  it('uses the in-tree fallback when no backend is installed', async () => {
    resetSublinearCore(null);
    const svc = new SparsificationService({ method: 'ppr', topK: 3 });
    await svc.initialize();
    (svc as any).pprBackend = null; // graph-transformer absent too
    const result = await svc.pprSparsification(0, directedGraph(500, 9), 3, 0.15);
    expect(result.method).toBe('ppr-fallback');
    expect(Math.abs(sum(result.scores) - 1)).toBeLessThan(1e-5);
  });

  it('sums to 1 and matches power iteration within 1e-9', () => {
    for (const seed of [5, 6, 7]) {
      const adj = directedGraph(1000, seed, 0.25);
      const run = personalizedPageRankInTree(seed, adj, 0.2, TIGHT);
      expect(Math.abs(sum(run.scores) - 1)).toBeLessThan(1e-12);
      expect(maxAbsDiff(run.scores, referencePpr(adj, seed, 0.2, 2000, 1e-14))).toBeLessThan(1e-9);
    }
  });

  it('agrees with sublinear-core within 1e-12 at the same budget', async () => {
    const core = (await loadSublinearCore())!;
    for (const [maxIterations, tolerance] of [[20, 1e-6], [20, 0], [500, 1e-12]] as const) {
      const adj = directedGraph(800, maxIterations + 1, 0.1);
      const a = personalizedPageRankInTree(4, adj, 0.15, { maxIterations, tolerance });
      const b = core.run(4, adj, 0.15, { maxIterations, tolerance });
      expect(a.iterations).toBe(b.iterations);
      expect(maxAbsDiff(a.scores, b.scores)).toBeLessThan(1e-12);
    }
  });

  it('is linear in the graph size (10,000 nodes well under a second)', () => {
    const adj = directedGraph(10_000, 3);
    const t0 = performance.now();
    const run = personalizedPageRankInTree(0, adj, 0.15, { maxIterations: 20, tolerance: 1e-6 });
    expect(performance.now() - t0).toBeLessThan(1000);
    expect(Math.abs(sum(run.scores) - 1)).toBeLessThan(1e-12);
  });

  it('alpha = 1 puts all mass on the source; an out-of-range source scores nothing', () => {
    const adj = directedGraph(50, 2);
    const one = personalizedPageRankInTree(7, adj, 1, TIGHT);
    expect(one.scores[7]).toBe(1);
    expect(sum(one.scores)).toBe(1);
    expect(sum(personalizedPageRankInTree(50, adj, 0.15, TIGHT).scores)).toBe(0);
  });
});

describe('regression: pre-3.7.1 fallback lost the mass of sink nodes', () => {
  it('old fallback sums well below 1 where the new one sums to 1', async () => {
    const adj = directedGraph(300, 8, 0.15);
    const old = oldFallback(0, adj, 0.15);
    expect(sum(old)).toBeLessThan(0.9);

    resetSublinearCore(null);
    const svc = new SparsificationService({ method: 'ppr', topK: 5 });
    const fixed = (svc as any).pprFallback(0, adj, 5, 0.15);
    expect(fixed.method).toBe('ppr-fallback');
    expect(Math.abs(sum(fixed.scores) - 1)).toBeLessThan(1e-5);
  });
});
