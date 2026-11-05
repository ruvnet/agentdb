/**
 * Adapter from AgentDB's adjacency lists to @ruvector/mincut-wasm (ADR-018).
 *
 * @ruvector/mincut-wasm computes the exact global minimum cut
 * (`WasmMinCut.fromEdges([[u, v, w], ...])`, `.minCutValue()`, `.partition()`
 * -> `{ s, t }`, `.cutEdges()`). This file maps that onto MincutService's
 * `MincutResult` (`partitions: number[][]`).
 *
 * Mapping rules:
 * - Adjacency lists are treated as undirected unit-weight edges, de-duplicated,
 *   so `[[1], [0]]` is one edge of weight 1 (not two). Self loops and
 *   out-of-range ids are ignored.
 * - The WASM graph only sees vertices that have an edge. A vertex with no
 *   edges makes the minimum cut 0; it is returned as its own side so every
 *   node 0..n-1 appears in exactly one partition.
 * - `cutSize` counts undirected cut edges once and equals `minCutValue()`.
 */

import type { GraphEdges } from '../../types/graph.js';
import { optionalImport, pickExport } from '../../utils/optional-import.js';

export const MINCUT_WASM_PACKAGE = '@ruvector/mincut-wasm';

export interface MincutWasmModule {
  WasmMinCut: {
    fromEdges(edges: Array<[number, number, number]>): WasmMinCutInstance;
  };
  getVersion?: () => string;
}

interface WasmMinCutInstance {
  minCutValue(): number;
  partition(): { s: number[]; t: number[] };
  free(): void;
}

export interface WasmMincutResult {
  partitions: number[][];
  cutSize: number;
  cutEdges: Array<[number, number]>;
  algorithm: 'stoer-wagner-wasm';
}

/** Loads @ruvector/mincut-wasm, or null when it is not installed or has no WasmMinCut. */
export async function loadMincutWasm(): Promise<MincutWasmModule | null> {
  const mod = await optionalImport(MINCUT_WASM_PACKAGE);
  const WasmMinCut = pickExport<MincutWasmModule['WasmMinCut']>(mod, 'WasmMinCut');
  if (!WasmMinCut || typeof WasmMinCut.fromEdges !== 'function') return null;
  return { WasmMinCut, getVersion: pickExport(mod, 'getVersion') };
}

/** Undirected, de-duplicated unit-weight edge list (u < v). */
export function undirectedEdges(edges: GraphEdges): Array<[number, number]> {
  const n = edges.length;
  const seen = new Set<number>();
  const out: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const neighbors = edges[i];
    if (!neighbors) continue;
    for (const raw of neighbors) {
      const j = Number(raw);
      if (!Number.isInteger(j) || j < 0 || j >= n || j === i) continue;
      const u = i < j ? i : j;
      const v = i < j ? j : i;
      const key = u * n + v;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push([u, v]);
    }
  }
  return out;
}

/**
 * Exact global minimum cut via WASM. Returns null for graphs with fewer than
 * two nodes so the caller can keep its own handling of those.
 */
export function wasmGlobalMincut(mod: MincutWasmModule, edges: GraphEdges): WasmMincutResult | null {
  const n = edges.length;
  if (n < 2) return null;

  const list = undirectedEdges(edges);
  const touched = new Uint8Array(n);
  for (const [u, v] of list) {
    touched[u] = 1;
    touched[v] = 1;
  }
  const isolated: number[] = [];
  const connected: number[] = [];
  for (let i = 0; i < n; i++) (touched[i] ? connected : isolated).push(i);

  // A vertex with no edges is a cut of value 0 on its own.
  if (isolated.length > 0) {
    const partitions = connected.length > 0
      ? [isolated, connected]
      : [[isolated[0]], isolated.slice(1)];
    return { partitions, cutSize: 0, cutEdges: [], algorithm: 'stoer-wagner-wasm' };
  }

  const graph = mod.WasmMinCut.fromEdges(list.map(([u, v]) => [u, v, 1] as [number, number, number]));
  try {
    const { s, t } = graph.partition();
    const side = new Uint8Array(n);
    for (const node of s) side[node] = 1;
    const cutEdges = list.filter(([u, v]) => side[u] !== side[v]);
    return {
      partitions: [Array.from(s).sort((a, b) => a - b), Array.from(t).sort((a, b) => a - b)],
      cutSize: cutEdges.length,
      cutEdges,
      algorithm: 'stoer-wagner-wasm',
    };
  } finally {
    graph.free();
  }
}
