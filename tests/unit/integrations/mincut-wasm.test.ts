/**
 * MincutService on @ruvector/mincut-wasm (ADR-018).
 *
 * The package is imported directly first: if it is missing or its API moved,
 * this file fails instead of silently testing the JavaScript fallback.
 */
import { describe, it, expect } from 'vitest';
import * as mincutWasm from '@ruvector/mincut-wasm';
import { MincutService } from '../../../src/controllers/MincutService.js';
import { undirectedEdges } from '../../../src/controllers/integrations/MincutWasmAdapter.js';
import type { GraphEdges } from '../../../src/types/graph.js';

/** Exact global min cut by enumerating every bipartition (n <= 14). */
function bruteForceMincut(edges: GraphEdges): number {
  const n = edges.length;
  const list = undirectedEdges(edges);
  let best = Infinity;
  // Fix node 0 on side A; every non-empty proper subset of the rest goes to side B.
  for (let mask = 1; mask < 1 << (n - 1); mask++) {
    let cut = 0;
    for (const [u, v] of list) {
      const su = u === 0 ? 0 : (mask >> (u - 1)) & 1;
      const sv = v === 0 ? 0 : (mask >> (v - 1)) & 1;
      if (su !== sv) cut++;
    }
    if (cut < best) best = cut;
  }
  return best;
}

function symmetric(n: number, pairs: Array<[number, number]>): GraphEdges {
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const [u, v] of pairs) {
    adj[u].push(v);
    adj[v].push(u);
  }
  return adj;
}

function randomConnected(n: number, extra: number, seed: number): GraphEdges {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const pairs: Array<[number, number]> = [];
  for (let i = 1; i < n; i++) pairs.push([i, Math.floor(rnd() * i)]); // spanning tree
  for (let e = 0; e < extra; e++) {
    const u = Math.floor(rnd() * n);
    const v = Math.floor(rnd() * n);
    if (u !== v) pairs.push([u, v]);
  }
  return symmetric(n, pairs);
}

function assertValidCut(edges: GraphEdges, result: { partitions: number[][]; cutSize: number; cutEdges: Array<[number, number]> }) {
  const n = edges.length;
  const all = result.partitions.flat().sort((a, b) => a - b);
  expect(all).toEqual(Array.from({ length: n }, (_, i) => i));
  expect(result.partitions.every((p) => p.length > 0)).toBe(true);
  const side = new Map<number, number>();
  result.partitions.forEach((p, idx) => p.forEach((node) => side.set(node, idx)));
  const crossing = undirectedEdges(edges).filter(([u, v]) => side.get(u) !== side.get(v));
  expect(result.cutEdges).toEqual(crossing);
  expect(result.cutSize).toBe(crossing.length);
}

async function wasmService(): Promise<MincutService> {
  const service = new MincutService({ algorithm: 'stoer-wagner' });
  await service.initialize();
  return service;
}

describe('@ruvector/mincut-wasm integration', () => {
  it('exposes the API the adapter relies on', () => {
    expect(typeof mincutWasm.WasmMinCut.fromEdges).toBe('function');
    const g = mincutWasm.WasmMinCut.fromEdges([[0, 1, 1], [1, 2, 1], [2, 0, 1]]);
    expect(g.minCutValue()).toBe(2);
    expect(g.partition()).toHaveProperty('s');
    g.free();
  });

  it('MincutService loads the WASM backend', async () => {
    const service = await wasmService();
    expect(service.getBackend()).toBe('wasm');
  });

  it.each([
    ['two triangles joined by a bridge', symmetric(6, [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [2, 3]]), 1],
    ['6-cycle', symmetric(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]), 2],
    ['K5', symmetric(5, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]), 4],
    ['two components', symmetric(4, [[0, 1], [2, 3]]), 0],
    ['isolated vertex', [[1], [0, 2], [1], []] as GraphEdges, 0],
    ['path', symmetric(5, [[0, 1], [1, 2], [2, 3], [3, 4]]), 1],
  ])('%s: exact cut value', async (_name, edges, expected) => {
    const result = await (await wasmService()).stoerWagnerMincut(edges);
    expect(result.algorithm).toBe('stoer-wagner-wasm');
    expect(result.cutSize).toBe(expected);
    assertValidCut(edges, result);
  });

  it('equals the brute-force minimum cut on 40 random graphs', async () => {
    const service = await wasmService();
    for (let seed = 1; seed <= 40; seed++) {
      const n = 6 + (seed % 7); // 6..12 nodes
      const edges = randomConnected(n, n + (seed % 9), seed);
      const result = await service.stoerWagnerMincut(edges);
      expect(result.algorithm).toBe('stoer-wagner-wasm');
      expect(result.cutSize).toBe(bruteForceMincut(edges));
      assertValidCut(edges, result);
    }
  });

  it('is never worse than the JavaScript half-split fallback', async () => {
    const service = await wasmService();
    for (let seed = 100; seed < 120; seed++) {
      const edges = randomConnected(30, 40, seed);
      const wasm = await service.stoerWagnerMincut(edges);
      const fallback = (service as any).stoerWagnerFallback(edges);
      expect(fallback.algorithm).toBe('stoer-wagner-fallback');
      expect(wasm.cutSize).toBeLessThanOrEqual(fallback.cutSize);
    }
  });

  it('does not reuse a cached cut for a different graph of the same size', async () => {
    const service = await wasmService();
    const bridge = symmetric(6, [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [2, 3]]);
    const cycle = symmetric(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]]);
    expect((await service.stoerWagnerMincut(bridge)).cutSize).toBe(1);
    expect((await service.stoerWagnerMincut(cycle)).cutSize).toBe(2);
  });

  it('keeps the JavaScript handling for graphs with fewer than two nodes', async () => {
    const service = await wasmService();
    expect((await service.stoerWagnerMincut([])).partitions).toHaveLength(2);
    expect((await service.stoerWagnerMincut([[]])).partitions).toHaveLength(2);
  });

  it('keeps the JavaScript path above wasmMaxNodes', async () => {
    const service = new MincutService({ algorithm: 'stoer-wagner', wasmMaxNodes: 50 });
    await service.initialize();
    expect((await service.stoerWagnerMincut(randomConnected(60, 60, 3))).algorithm).toBe('stoer-wagner-fallback');
    expect((await service.stoerWagnerMincut(randomConnected(50, 60, 3))).algorithm).toBe('stoer-wagner-wasm');
  });

  it('handles a 2,000-node graph', async () => {
    const service = await wasmService();
    const edges = randomConnected(2000, 6000, 7);
    const t0 = performance.now();
    const result = await service.stoerWagnerMincut(edges);
    const ms = performance.now() - t0;
    assertValidCut(edges, result);
    expect(result.cutSize).toBeLessThanOrEqual(Math.min(...edges.map((a) => a!.length)));
    expect(ms).toBeLessThan(10_000);
  });
});
