/**
 * Personalized PageRank through @ruvector/graph-transformer (ADR-018).
 *
 * Order of preference:
 *   1. `@ruvector/graph-transformer` (NAPI): `new GraphTransformer().pprScores(source, adjacency, alpha)`
 *   2. `ruvector-graph-transformer-wasm`: `new JsGraphTransformer().ppr_scores(source, adjacency, alpha)`
 *      (a web-target build: in Node the .wasm bytes are read and passed to initSync,
 *      in a browser the default init fetches them next to the module)
 *   3. null — SparsificationService keeps its JavaScript power iteration.
 *
 * Both native variants run 20 power iterations and keep the mass of a node
 * with no out-edges on that node (a self loop). The adjacency handed to them
 * has every `undefined` entry replaced by `[]` and out-of-range ids removed.
 */

import type { GraphEdges } from '../../types/graph.js';
import { optionalImport, pickExport } from '../../utils/optional-import.js';

export const GRAPH_TRANSFORMER_PACKAGE = '@ruvector/graph-transformer';
export const GRAPH_TRANSFORMER_WASM_PACKAGE = 'ruvector-graph-transformer-wasm';

export interface PprBackend {
  runtime: 'napi' | 'wasm';
  scores(source: number, adjacency: number[][], alpha: number): number[];
}

let backendPromise: Promise<PprBackend | null> | null = null;

/** Loads the fastest available PPR backend once per process. */
export function loadPprBackend(): Promise<PprBackend | null> {
  if (!backendPromise) {
    backendPromise = (async () => (await loadNapi()) ?? (await loadWasm()))();
  }
  return backendPromise;
}

/** Test hook. */
export function resetPprBackend(): void {
  backendPromise = null;
}

export async function loadNapi(): Promise<PprBackend | null> {
  const mod = await optionalImport(GRAPH_TRANSFORMER_PACKAGE);
  const GraphTransformer = pickExport<any>(mod, 'GraphTransformer');
  if (typeof GraphTransformer !== 'function') return null;
  try {
    const gt = new GraphTransformer();
    if (typeof gt.pprScores !== 'function') return null;
    return {
      runtime: 'napi',
      scores: (source, adjacency, alpha) => gt.pprScores(source, adjacency, alpha),
    };
  } catch {
    return null; // native binary missing for this platform
  }
}

export async function loadWasm(): Promise<PprBackend | null> {
  const mod = await optionalImport(GRAPH_TRANSFORMER_WASM_PACKAGE);
  const JsGraphTransformer = pickExport<any>(mod, 'JsGraphTransformer');
  if (typeof JsGraphTransformer !== 'function') return null;
  try {
    const isNode = typeof process !== 'undefined' && !!process.versions?.node;
    if (isNode) {
      const { createRequire } = await import('module');
      const { readFileSync } = await import('fs');
      const require = createRequire(import.meta.url);
      const wasmPath = require.resolve(`${GRAPH_TRANSFORMER_WASM_PACKAGE}/ruvector_graph_transformer_wasm_bg.wasm`);
      pickExport<any>(mod, 'initSync')({ module: readFileSync(wasmPath) });
    } else {
      await (mod.default as () => Promise<unknown>)();
    }
    const gt = new JsGraphTransformer({});
    return {
      runtime: 'wasm',
      scores: (source, adjacency, alpha) => Array.from(gt.ppr_scores(source, adjacency, alpha) as number[]),
    };
  } catch {
    return null;
  }
}

/** Dense number[][] adjacency with no holes and no out-of-range ids. */
export function toNativeAdjacency(edges: GraphEdges): number[][] {
  const n = edges.length;
  const out: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const neighbors = edges[i];
    out[i] = neighbors
      ? neighbors.filter((j) => Number.isInteger(j) && j >= 0 && j < n)
      : [];
  }
  return out;
}
