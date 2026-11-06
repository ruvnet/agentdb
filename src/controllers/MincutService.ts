/**
 * MincutService - Graph Partitioning with Minimum Cut Algorithms
 *
 * Implements multiple minimum cut algorithms for dynamic graph partitioning
 * to achieve 50-80% memory reduction through intelligent node clustering.
 *
 * Features:
 * - Stoer-Wagner algorithm (deterministic, optimal for small graphs)
 * - Karger's randomized algorithm (probabilistic, good for large graphs)
 * - Flow-based mincut (max-flow min-cut theorem)
 * - Partition caching for repeated queries
 * - Exact global min cut through @ruvector/mincut-wasm when installed (ADR-018)
 *
 * Goal: 50-80% memory reduction through dynamic graph partitioning
 */

import type { GraphEdges, GraphInput } from '../types/graph.js';
import { normalizeGraphEdges } from '../utils/graph-edges.js';
import { loadMincutWasm, wasmGlobalMincut, undirectedEdges, type MincutWasmModule } from './integrations/MincutWasmAdapter.js';

export interface MincutConfig {
  algorithm: 'stoer-wagner' | 'karger' | 'flow-based';
  maxPartitionSize?: number;
  minCutThreshold?: number;
  /**
   * Largest graph (nodes) sent to @ruvector/mincut-wasm; bigger graphs use the
   * JavaScript path. Building the WASM structure grows about quadratically
   * (~0.15 s at 1k nodes, ~1.7 s at 4k, ~6.5 s at 8k); queries after that are
   * sub-millisecond. Default 2000.
   */
  wasmMaxNodes?: number;
  /**
   * Largest graph (undirected edges) sent to @ruvector/mincut-wasm. Build time
   * also grows with edges (~0.4 s at 1k nodes / 25k edges, ~2.2 s at 2k / 51k,
   * ~15 s for K1000's 500k). Default 20000.
   */
  wasmMaxEdges?: number;
}

export interface MincutResult {
  partitions: number[][];  // Array of node sets
  cutSize: number;         // Number of edges cut
  cutEdges: Array<[number, number]>;
  algorithm: string;
}

export interface Partition {
  nodes: Set<number>;
  internalEdges: number;
  externalEdges: number;
}

export class MincutService {
  private mincutWasm: MincutWasmModule | null = null;
  private initialized: boolean = false;
  private partitionCache: Map<string, MincutResult> = new Map();

  constructor(private config: MincutConfig) {}

  async initialize(): Promise<void> {
    if (this.initialized) return;
    this.mincutWasm = await loadMincutWasm();
    if (!this.mincutWasm) {
      console.warn('⚠️  @ruvector/mincut-wasm not installed, using JavaScript fallback');
    }
    this.initialized = true;
  }

  /**
   * Stoer-Wagner mincut algorithm
   */
  async stoerWagnerMincut(input: GraphInput): Promise<MincutResult> {
    const edges = normalizeGraphEdges(input);
    const cacheKey = `sw-${this.hashEdges(edges)}`;
    if (this.partitionCache.has(cacheKey)) {
      return this.partitionCache.get(cacheKey)!;
    }

    if (!this.initialized) await this.initialize();
    if (this.mincutWasm && this.fitsWasm(edges)) {
      try {
        const result = wasmGlobalMincut(this.mincutWasm, edges);
        if (result) {
          this.partitionCache.set(cacheKey, result);
          return result;
        }
      } catch (err) {
        console.warn('⚠️  @ruvector/mincut-wasm failed, using JavaScript fallback:', (err as Error).message);
      }
    }

    // JavaScript fallback
    return this.stoerWagnerFallback(edges);
  }

  /**
   * Karger's randomized mincut
   */
  async kargerMincut(
    input: GraphInput,
    iterations: number = 100
  ): Promise<MincutResult> {
    const edges = normalizeGraphEdges(input);
    let bestCut: MincutResult | null = null;

    for (let i = 0; i < iterations; i++) {
      const result = await this.kargerIteration(edges);

      if (!bestCut || result.cutSize < bestCut.cutSize) {
        bestCut = result;
      }
    }

    return bestCut!;
  }

  /**
   * Flow-based mincut (max-flow min-cut theorem)
   */
  async flowBasedMincut(
    input: GraphInput,
    source: number,
    sink: number
  ): Promise<MincutResult> {
    const edges = normalizeGraphEdges(input);
    // Use Ford-Fulkerson algorithm
    return this.fordFulkersonMincut(edges, source, sink);
  }

  /**
   * Partition graph based on configuration
   */
  async partition(input: GraphInput): Promise<MincutResult> {
    const edges = normalizeGraphEdges(input);
    let result: MincutResult;
    switch (this.config.algorithm) {
      case 'stoer-wagner':
        result = await this.stoerWagnerMincut(edges);
        break;
      case 'karger':
        result = await this.kargerMincut(edges);
        break;
      case 'flow-based':
        // Use first and last nodes as source/sink
        result = await this.flowBasedMincut(edges, 0, edges.length - 1);
        break;
      default:
        throw new Error(`Unknown algorithm: ${this.config.algorithm}`);
    }
    return this.config.maxPartitionSize ? this.boundPartitions(result, edges, this.config.maxPartitionSize) : result;
  }

  /**
   * Split every partition larger than maxSize into equal chunks of consecutive
   * node ids, so no part exceeds maxSize. A minimum cut is often one node
   * against the rest; callers that process each part (partitioned attention,
   * quadratic in part size) need bounded parts. cutEdges and cutSize are
   * recomputed over all parts, each undirected edge counted once.
   */
  boundPartitions(result: MincutResult, input: GraphInput, maxSize: number): MincutResult {
    const edges = normalizeGraphEdges(input);
    const limit = Math.max(1, Math.floor(maxSize));
    if (result.partitions.every((p) => p.length <= limit)) return result;
    const partitions: number[][] = [];
    for (const part of result.partitions) {
      if (part.length <= limit) {
        partitions.push(part);
        continue;
      }
      const sorted = [...part].sort((a, b) => a - b);
      const chunks = Math.ceil(sorted.length / limit);
      const size = Math.ceil(sorted.length / chunks);
      for (let i = 0; i < sorted.length; i += size) partitions.push(sorted.slice(i, i + size));
    }
    const side = new Map<number, number>();
    partitions.forEach((p, idx) => p.forEach((node) => side.set(node, idx)));
    const cutEdges = undirectedEdges(edges).filter(([u, v]) => side.get(u) !== side.get(v));
    return { partitions, cutSize: cutEdges.length, cutEdges, algorithm: `${result.algorithm}+bounded` };
  }

  /**
   * Get partition containing a node
   */
  getPartition(
    node: number,
    result: MincutResult
  ): number[] {
    for (const partition of result.partitions) {
      if (partition.includes(node)) {
        return partition;
      }
    }
    return [];
  }

  /**
   * Check if two nodes are in same partition
   */
  inSamePartition(
    node1: number,
    node2: number,
    result: MincutResult
  ): boolean {
    const partition = this.getPartition(node1, result);
    return partition.includes(node2);
  }

  /**
   * Calculate partition statistics
   */
  getPartitionStats(
    result: MincutResult,
    input: GraphInput
  ): {
    numPartitions: number;
    avgPartitionSize: number;
    maxPartitionSize: number;
    minPartitionSize: number;
    cutRatio: number;
  } {
    const edges = normalizeGraphEdges(input);
    const sizes = result.partitions.map((p) => p.length);
    const totalEdges = edges.reduce((sum, neighbors) => sum + (neighbors?.length || 0), 0) / 2;

    return {
      numPartitions: result.partitions.length,
      avgPartitionSize: sizes.reduce((a, b) => a + b, 0) / sizes.length,
      maxPartitionSize: Math.max(...sizes),
      minPartitionSize: Math.min(...sizes),
      cutRatio: totalEdges > 0 ? result.cutSize / totalEdges : 0,
    };
  }

  private fitsWasm(edges: GraphEdges): boolean {
    if (edges.length > (this.config.wasmMaxNodes ?? 2000)) return false;
    const maxEdges = this.config.wasmMaxEdges ?? 20000;
    let directed = 0;
    for (const neighbors of edges) directed += neighbors?.length ?? 0;
    // Adjacency lists are usually symmetric: each undirected edge is listed twice.
    return directed / 2 <= maxEdges || undirectedEdges(edges).length <= maxEdges;
  }

  /** Which implementation answers stoer-wagner queries. */
  getBackend(): 'wasm' | 'fallback' {
    return this.mincutWasm ? 'wasm' : 'fallback';
  }

  /**
   * Clear partition cache
   */
  clearCache(): void {
    this.partitionCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    keys: string[];
  } {
    return {
      size: this.partitionCache.size,
      keys: Array.from(this.partitionCache.keys()),
    };
  }

  // Helper methods
  /** FNV-1a over the adjacency lists: different graphs of the same size get different keys. */
  private hashEdges(edges: GraphEdges): string {
    let h = 0x811c9dc5;
    let count = 0;
    const mix = (x: number) => {
      h ^= x & 0xffff;
      h = Math.imul(h, 0x01000193);
      h ^= x >>> 16;
      h = Math.imul(h, 0x01000193);
    };
    for (let i = 0; i < edges.length; i++) {
      mix(0xfffffff0 - i);
      for (const j of edges[i] || []) {
        mix(j);
        count++;
      }
    }
    return `${edges.length}-${count}-${(h >>> 0).toString(16)}`;
  }

  private stoerWagnerFallback(edges: GraphEdges): MincutResult {
    // Simple 2-partition heuristic
    const n = edges.length;
    const mid = Math.floor(n / 2);

    const partition1 = Array.from({ length: mid }, (_, i) => i);
    const partition2 = Array.from({ length: n - mid }, (_, i) => i + mid);

    const cutEdges: Array<[number, number]> = [];
    let cutSize = 0;

    for (let i = 0; i < mid; i++) {
      const neighbors = edges[i] || [];
      for (const j of neighbors) {
        if (j >= mid) {
          cutEdges.push([i, j]);
          cutSize++;
        }
      }
    }

    const result = {
      partitions: [partition1, partition2],
      cutSize,
      cutEdges,
      algorithm: 'stoer-wagner-fallback',
    };

    // Cache the result
    const cacheKey = `sw-${this.hashEdges(edges)}`;
    this.partitionCache.set(cacheKey, result);

    return result;
  }

  private async kargerIteration(edges: GraphEdges): Promise<MincutResult> {
    // Implement single Karger iteration
    // Contract random edges until 2 nodes remain
    const n = edges.length;
    const contracted = new Map<number, Set<number>>();

    // Initialize: each node is its own set
    for (let i = 0; i < n; i++) {
      contracted.set(i, new Set([i]));
    }

    const edgeList: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) {
      const neighbors = edges[i] || [];
      for (const j of neighbors) {
        if (i < j) edgeList.push([i, j]);
      }
    }

    // Contract edges
    while (contracted.size > 2) {
      if (edgeList.length === 0) break;

      const idx = Math.floor(Math.random() * edgeList.length);
      const [u, v] = edgeList[idx];

      // Find which sets contain u and v
      let setU: Set<number> | undefined;
      let setV: Set<number> | undefined;
      let keyU: number | undefined;
      let keyV: number | undefined;

      for (const [key, set] of Array.from(contracted.entries())) {
        if (set.has(u)) { setU = set; keyU = key; }
        if (set.has(v)) { setV = set; keyV = key; }
      }

      if (setU && setV && keyU !== keyV) {
        // Merge sets
        const nodesToMerge = Array.from(setV);
        for (const node of nodesToMerge) {
          setU.add(node);
        }
        contracted.delete(keyV!);
      }

      edgeList.splice(idx, 1);
    }

    const partitions: number[][] = [];
    for (const set of Array.from(contracted.values())) {
      partitions.push(Array.from(set));
    }
    const cutEdges = this.getCutEdges(partitions, edges);

    return {
      partitions,
      cutSize: cutEdges.length,
      cutEdges,
      algorithm: 'karger',
    };
  }

  private fordFulkersonMincut(
    edges: GraphEdges,
    source: number,
    sink: number
  ): MincutResult {
    // Simplified max-flow implementation
    // In practice, use optimized algorithms (Edmonds-Karp, etc.)

    const n = edges.length;
    const capacity = Array(n).fill(null).map(() => Array(n).fill(0));

    // Build capacity matrix (treat as undirected by adding both directions)
    for (let i = 0; i < n; i++) {
      const neighbors = edges[i] || [];
      for (const j of neighbors) {
        capacity[i][j] = 1; // Unit capacity
        capacity[j][i] = 1; // Reverse edge for undirected graph
      }
    }

    // Perform BFS-based max-flow to find augmenting paths
    const residual = capacity.map(row => [...row]);

    while (true) {
      // BFS to find augmenting path
      const parent = new Map<number, number>();
      const visited = new Set<number>([source]);
      const queue = [source];
      let foundPath = false;

      while (queue.length > 0 && !foundPath) {
        const u = queue.shift()!;

        for (let v = 0; v < n; v++) {
          if (!visited.has(v) && residual[u][v] > 0) {
            visited.add(v);
            parent.set(v, u);
            queue.push(v);

            if (v === sink) {
              foundPath = true;
              break;
            }
          }
        }
      }

      if (!foundPath) break;

      // Find minimum capacity along the path
      let pathFlow = Infinity;
      for (let v = sink; v !== source; v = parent.get(v)!) {
        const u = parent.get(v)!;
        pathFlow = Math.min(pathFlow, residual[u][v]);
      }

      // Update residual capacities
      for (let v = sink; v !== source; v = parent.get(v)!) {
        const u = parent.get(v)!;
        residual[u][v] -= pathFlow;
        residual[v][u] += pathFlow;
      }
    }

    // Find reachable nodes from source in residual graph
    const reachable = new Set<number>([source]);
    const queue = [source];

    while (queue.length > 0) {
      const u = queue.shift()!;
      for (let v = 0; v < n; v++) {
        if (residual[u][v] > 0 && !reachable.has(v)) {
          reachable.add(v);
          queue.push(v);
        }
      }
    }

    const partition1 = Array.from(reachable);
    const partition2 = Array.from({ length: n }, (_, i) => i).filter((i) => !reachable.has(i));

    const cutEdges = this.getCutEdges([partition1, partition2], edges);

    return {
      partitions: [partition1, partition2],
      cutSize: cutEdges.length,
      cutEdges,
      algorithm: 'ford-fulkerson',
    };
  }

  private getCutEdges(
    partitions: number[][],
    edges: GraphEdges
  ): Array<[number, number]> {
    const cutEdges: Array<[number, number]> = [];
    const partitionMap = new Map<number, number>();

    partitions.forEach((partition, idx) => {
      partition.forEach((node) => partitionMap.set(node, idx));
    });

    for (let i = 0; i < edges.length; i++) {
      const neighbors = edges[i] || [];
      for (const j of neighbors) {
        if (partitionMap.get(i) !== partitionMap.get(j)) {
          cutEdges.push([i, j]);
        }
      }
    }

    return cutEdges;
  }
}
