/**
 * Graph input normalisation for SparsificationService and MincutService.
 *
 * The canonical adjacency list is an array indexed by node id
 * (`GraphEdges`). The documentation also showed an object keyed by node id
 * (`{ 0: [1, 2], 1: [0] }`), which JavaScript callers pass; it used to fail
 * with "edges.reduce is not a function" or silently score an empty graph.
 * Both forms are accepted at the public boundary and converted here.
 */

import type { GraphEdges, GraphInput } from '../types/graph.js';

/** Largest node id accepted in an object-keyed graph (bounds the allocation). */
export const MAX_GRAPH_NODES = 10_000_000;

/**
 * Return `edges` as a dense `GraphEdges` array. Arrays are returned as is.
 * For an object keyed by node id, node count is one more than the largest id
 * seen as a key or as a neighbour, and ids without a key get no neighbours.
 *
 * @throws TypeError for anything other than an array or a plain object with
 *   non-negative integer keys and array values
 * @throws RangeError when a node id is at or above MAX_GRAPH_NODES
 */
export function normalizeGraphEdges(edges: GraphInput): GraphEdges {
  if (Array.isArray(edges)) return edges;
  if (edges === null || typeof edges !== 'object') {
    throw new TypeError('Graph edges must be an adjacency array or an object keyed by node id');
  }

  const entries: Array<[number, number[] | undefined]> = [];
  let n = 0;
  for (const [key, neighbors] of Object.entries(edges)) {
    const id = Number(key);
    if (!Number.isInteger(id) || id < 0 || String(id) !== key) {
      throw new TypeError(`Graph node id must be a non-negative integer, got "${key}"`);
    }
    if (id >= MAX_GRAPH_NODES) {
      throw new RangeError(`Graph node id ${id} exceeds the limit of ${MAX_GRAPH_NODES - 1}`);
    }
    if (neighbors !== undefined && !Array.isArray(neighbors)) {
      throw new TypeError(`Neighbours of node ${id} must be an array`);
    }
    entries.push([id, neighbors as number[] | undefined]);
    n = Math.max(n, id + 1);
    for (const j of neighbors ?? []) {
      if (Number.isInteger(j) && j >= 0 && j < MAX_GRAPH_NODES) n = Math.max(n, j + 1);
    }
  }

  const out: GraphEdges = new Array(n);
  for (let i = 0; i < n; i++) out[i] = [];
  for (const [id, neighbors] of entries) out[id] = neighbors ? [...neighbors] : [];
  return out;
}
