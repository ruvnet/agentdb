/**
 * Personalized PageRank through @ruvnet/sublinear-core, with an in-tree
 * implementation of the same algorithm when the package is not installed.
 *
 * Both paths run power iteration in float64 on the out-edges, send the mass of
 * a node with no out-edges back to the source (sublinear-core's
 * `dangling: 'teleport'`), and stop when the L1 change between iterates is at
 * or below the tolerance or after `maxIterations`. The scores therefore sum to
 * 1 at every iteration, and the two paths agree to rounding error.
 *
 * The previous JavaScript fallback scanned every adjacency list for every node
 * (O(n^2 * degree) per iteration) and dropped dangling mass, so its scores
 * summed to roughly 0.73-0.86 on graphs with sink nodes.
 *
 * `alpha` is the teleport (restart) probability, as everywhere in
 * SparsificationService; sublinear-core's `damping` is `1 - alpha`.
 * The adjacency must already be clean (see toNativeAdjacency): integer ids in
 * [0, n), no holes. Duplicate neighbours count as parallel edges.
 */

import { optionalImport, pickExport } from '../../utils/optional-import.js';

export const SUBLINEAR_CORE_PACKAGE = '@ruvnet/sublinear-core';

export interface PprRunOptions {
  /** Iteration cap. */
  maxIterations: number;
  /** Stop when the L1 change between iterates is <= tolerance. */
  tolerance: number;
}

export interface PprRun {
  scores: Float64Array;
  iterations: number;
  /** L1 change of the last iteration. */
  delta: number;
  converged: boolean;
}

export interface SublinearCorePpr {
  runtime: 'sublinear-core';
  run(source: number, adjacency: number[][], alpha: number, options: PprRunOptions): PprRun;
}

let corePromise: Promise<SublinearCorePpr | null> | null = null;

/** Loads @ruvnet/sublinear-core once per process; null when it is not installed. */
export function loadSublinearCore(): Promise<SublinearCorePpr | null> {
  if (!corePromise) corePromise = probe(optionalImport(SUBLINEAR_CORE_PACKAGE));
  return corePromise;
}

/**
 * Test hook. With no argument the next load probes the package again; with a
 * module (or null for "not installed") the next load uses that instead.
 */
export function resetSublinearCore(mod?: unknown): void {
  corePromise = mod === undefined ? null : probe(Promise.resolve(mod));
}

async function probe(pending: Promise<unknown>): Promise<SublinearCorePpr | null> {
  const mod = await pending;
  const graphFromCsr = pickExport<(csr: CsrAdjacency) => unknown>(mod, 'graphFromCsr');
  const personalizedPageRank = pickExport<
    (g: unknown, seed: number, o: Record<string, unknown>) => PprRun
  >(mod, 'personalizedPageRank');
  if (typeof graphFromCsr !== 'function' || typeof personalizedPageRank !== 'function') {
    return null;
  }
  return {
    runtime: 'sublinear-core',
    run(source, adjacency, alpha, { maxIterations, tolerance }) {
      const r = personalizedPageRank(graphFromCsr(toCsrAdjacency(adjacency)), source, {
        damping: 1 - alpha,
        // sublinear-core requires a positive tolerance; 0 means "run maxIterations".
        tolerance: tolerance > 0 ? tolerance : Number.MIN_VALUE,
        maxIterations,
        dangling: 'teleport',
      });
      return { scores: r.scores, iterations: r.iterations, delta: r.delta, converged: r.converged };
    },
  };
}

/** sublinear-core's CsrMatrix: row = source, columns strictly increasing per row. */
export interface CsrAdjacency {
  rows: number;
  cols: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  values: Float64Array;
}

/**
 * CSR out-adjacency with unit weights; duplicate neighbours are merged into one
 * entry whose weight is the multiplicity, so they still count as parallel edges.
 * Built directly because sublinear-core's graphFromAdjacency goes through an
 * array of [u, v] tuples and costs more than the power iteration itself.
 */
export function toCsrAdjacency(adjacency: number[][]): CsrAdjacency {
  const n = adjacency.length;
  let nnz = 0;
  for (let u = 0; u < n; u++) nnz += adjacency[u].length;
  const rowPtr = new Uint32Array(n + 1);
  const colIdx = new Uint32Array(nnz);
  const values = new Float64Array(nnz);
  let k = 0;
  for (let u = 0; u < n; u++) {
    const row = adjacency[u];
    const sorted = row.length > 1 ? Uint32Array.from(row).sort() : row;
    for (let i = 0; i < sorted.length; i++) {
      const v = sorted[i];
      if (k > rowPtr[u] && colIdx[k - 1] === v) {
        values[k - 1] += 1;
      } else {
        colIdx[k] = v;
        values[k] = 1;
        k++;
      }
    }
    rowPtr[u + 1] = k;
  }
  return { rows: n, cols: n, rowPtr, colIdx: colIdx.subarray(0, k), values: values.subarray(0, k) };
}

/**
 * In-tree personalized PageRank, same algorithm and stopping rule as
 * sublinear-core. O(iterations * (n + edges)).
 */
export function personalizedPageRankInTree(
  source: number,
  adjacency: number[][],
  alpha: number,
  { maxIterations, tolerance }: PprRunOptions
): PprRun {
  const n = adjacency.length;
  const d = 1 - alpha;
  let x = new Float64Array(n);
  let y = new Float64Array(n);
  if (!(Number.isInteger(source) && source >= 0 && source < n)) {
    return { scores: x, iterations: 0, delta: 0, converged: true };
  }
  x[source] = 1;
  let delta = Infinity;
  let it = 0;
  while (it < maxIterations) {
    y.fill(0);
    let danglingMass = 0;
    for (let u = 0; u < n; u++) {
      const xu = x[u];
      if (xu === 0) continue;
      const row = adjacency[u];
      if (row.length === 0) {
        danglingMass += xu;
        continue;
      }
      const f = (d * xu) / row.length;
      for (let k = 0; k < row.length; k++) y[row[k]] += f;
    }
    y[source] += 1 - d + d * danglingMass;
    delta = 0;
    for (let i = 0; i < n; i++) delta += Math.abs(y[i] - x[i]);
    const t = x;
    x = y;
    y = t;
    it++;
    if (delta <= tolerance) break;
  }
  return { scores: x, iterations: it, delta, converged: delta <= tolerance };
}
