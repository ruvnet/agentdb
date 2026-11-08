/**
 * SparsificationService - Graph Sparsification for AgentDB
 *
 * Implements Personalized PageRank (PPR), random walk sampling, and spectral
 * sparsification for reducing graph size while preserving structure.
 *
 * Features:
 * - 10-100x speedup for large graphs
 * - PPR-based node importance scoring
 * - Random walk sampling
 * - Spectral sparsification
 * - PPR through @ruvnet/sublinear-core, then @ruvector/graph-transformer (NAPI) or
 *   ruvector-graph-transformer-wasm, with an in-tree JavaScript fallback (ADR-018).
 *   sublinear-core and the fallback compute the same standard PPR (dangling mass
 *   returns to the source, scores sum to 1); graph-transformer keeps dangling mass
 *   on the sink and always runs 20 iterations.
 * - Zero-copy operations where supported
 *
 * Based on:
 * - "Fast Personalized PageRank on MapReduce" (Bahmani et al., 2011)
 * - "Graph Sparsification by Effective Resistances" (Spielman & Srivastava, 2011)
 * - "Local Graph Partitioning using PageRank Vectors" (Andersen et al., 2006)
 *
 * @version 3.0.0-alpha.5
 */

import type { GraphEdges, GraphInput } from '../types/graph.js';
import { normalizeGraphEdges } from '../utils/graph-edges.js';
import { loadPprBackend, toNativeAdjacency, type PprBackend } from './integrations/PprNativeAdapter.js';
import {
  loadSublinearCore,
  personalizedPageRankInTree,
  type PprRun,
  type SublinearCorePpr,
} from './integrations/SublinearCoreAdapter.js';

// Re-export for convenience
export type { GraphEdges, GraphInput };

export interface SparsificationConfig {
  /** Sparsification method */
  method: 'ppr' | 'random-walk' | 'spectral' | 'degree-based';
  /** Number of top nodes to keep */
  topK: number;
  /** PPR teleport probability (default: 0.15) */
  alpha?: number;
  /** Number of random walks (default: 100) */
  numWalks?: number;
  /** Random walk length (default: 10) */
  walkLength?: number;
  /** Convergence threshold for PPR: L1 change between iterations (default: 1e-6) */
  convergenceThreshold?: number;
  /** Maximum PPR iterations (default: 20) */
  maxIterations?: number;
}

export interface SparsificationResult {
  /** Top-k node indices by importance */
  topKIndices: number[];
  /** Importance scores for all nodes */
  scores: Float32Array;
  /** Ratio of edges retained (edges_kept / total_edges) */
  sparsityRatio: number;
  /** Method used for sparsification */
  method: string;
  /** Execution time in milliseconds */
  executionTimeMs?: number;
  /** Additional metadata */
  metadata?: {
    iterations?: number;
    convergence?: number;
    totalNodes?: number;
    totalEdges?: number;
  };
}

/**
 * SparsificationService - Reduces graph size while preserving structure
 */
export class SparsificationService {
  private sublinearCore: SublinearCorePpr | null = null;
  private pprBackend: PprBackend | null = null;
  private initialized: boolean = false;
  private config: SparsificationConfig;

  constructor(config: SparsificationConfig) {
    this.config = {
      alpha: 0.15,
      numWalks: 100,
      walkLength: 10,
      convergenceThreshold: 1e-6,
      maxIterations: 20,
      ...config,
    };
  }

  /**
   * Initialize WASM/NAPI bindings
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    this.sublinearCore = await loadSublinearCore();
    this.pprBackend = await loadPprBackend();
    if (!this.sublinearCore && !this.pprBackend) {
      console.warn('⚠️  SparsificationService: @ruvnet/sublinear-core and @ruvector/graph-transformer not available, using JavaScript fallback');
    }

    this.initialized = true;
  }

  /**
   * Personalized PageRank sparsification
   *
   * Computes importance scores for nodes based on random walk with restart.
   * Nodes with higher PPR scores are more important relative to the source.
   *
   * @param sourceNode - Starting node for PPR
   * @param edges - Graph adjacency list
   * @param topK - Number of top nodes to return
   * @param alpha - Teleport probability (1-alpha = continue walk)
   * @returns Sparsification result with top-k nodes
   */
  async pprSparsification(
    sourceNode: number,
    input: GraphInput,
    topK: number,
    alpha: number = 0.15
  ): Promise<SparsificationResult> {
    const edges = normalizeGraphEdges(input);
    const startTime = performance.now();

    if (!this.initialized) await this.initialize();
    const n = edges.length;
    const validSource = n > 0 && Number.isInteger(sourceNode) && sourceNode >= 0 && sourceNode < n;
    if (this.sublinearCore && validSource) {
      try {
        const run = this.sublinearCore.run(sourceNode, toNativeAdjacency(edges), alpha, this.pprRunOptions());
        return this.pprResult(run, edges, topK, 'ppr-sublinear-core', startTime);
      } catch (err) {
        console.warn('sublinear-core PPR failed, falling back:', (err as Error).message);
      }
    }
    if (this.pprBackend && n > 0 && Number.isInteger(sourceNode) && sourceNode >= 0 && sourceNode < n) {
      try {
        const raw = this.pprBackend.scores(sourceNode, toNativeAdjacency(edges), alpha);
        const scores = Float32Array.from(raw);
        const totalEdges = this.countTotalEdges(edges);
        return {
          topKIndices: this.getTopK(scores, topK),
          scores,
          sparsityRatio: totalEdges > 0 ? topK / totalEdges : 0,
          method: `ppr-${this.pprBackend.runtime}`,
          executionTimeMs: performance.now() - startTime,
          metadata: { iterations: 20, totalNodes: n, totalEdges },
        };
      } catch (err) {
        console.warn(`${this.pprBackend.runtime} PPR failed, falling back:`, (err as Error).message);
      }
    }

    // JavaScript fallback
    return this.pprFallback(sourceNode, edges, topK, alpha);
  }

  /**
   * Random walk sampling sparsification
   *
   * Performs multiple random walks from source node and counts visit frequencies.
   * More frequently visited nodes are considered more important.
   *
   * @param sourceNode - Starting node for walks
   * @param edges - Graph adjacency list
   * @param topK - Number of top nodes to return
   * @param numWalks - Number of random walks
   * @param walkLength - Length of each walk
   * @returns Sparsification result with top-k nodes
   */
  async randomWalkSparsification(
    sourceNode: number,
    input: GraphInput,
    topK: number,
    numWalks: number = 100,
    walkLength: number = 10
  ): Promise<SparsificationResult> {
    const edges = normalizeGraphEdges(input);
    const startTime = performance.now();
    const visitCounts = new Map<number, number>();

    for (let i = 0; i < numWalks; i++) {
      let current = sourceNode;

      for (let step = 0; step < walkLength; step++) {
        visitCounts.set(current, (visitCounts.get(current) || 0) + 1);

        const neighbors = edges[current] || [];
        if (neighbors.length === 0) break;

        // Random walk step
        current = neighbors[Math.floor(Math.random() * neighbors.length)];
      }
    }

    // Get top-k most visited
    const sorted = Array.from(visitCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK);

    const topKIndices = sorted.map(([node]) => node);
    const n = edges.length;
    const scores = new Float32Array(n);

    // Normalize by total visits (not numWalks) to keep scores <= 1.0
    const totalVisits = Array.from(visitCounts.values()).reduce((sum, count) => sum + count, 0);
    sorted.forEach(([node, count]) => {
      scores[node] = count / totalVisits;
    });

    const executionTimeMs = performance.now() - startTime;

    return {
      topKIndices,
      scores,
      sparsityRatio: topK / this.countTotalEdges(edges),
      method: 'random-walk',
      executionTimeMs,
      metadata: {
        totalNodes: edges.length,
        totalEdges: this.countTotalEdges(edges),
      },
    };
  }

  /**
   * Spectral sparsification
   *
   * Uses graph spectrum (eigenvalues/eigenvectors) to identify important edges.
   * Falls back to degree-based approximation if spectral methods unavailable.
   *
   * @param edges - Graph adjacency list
   * @param topK - Number of top nodes to return
   * @returns Sparsification result with top-k nodes
   */
  async spectralSparsification(
    input: GraphInput,
    topK: number
  ): Promise<SparsificationResult> {
    const edges = normalizeGraphEdges(input);
    const startTime = performance.now();

    // No published spectral sparsifier binding yet (ruvector-sparsifier is crates.io only):
    // degree-based approximation
    return this.degreeBasedSparsification(edges, topK);
  }

  /**
   * Sparsify graph based on configuration
   *
   * @param sourceNode - Source node (required for PPR and random-walk)
   * @param edges - Graph adjacency list
   * @returns Sparsification result
   */
  async sparsify(
    sourceNode: number,
    input: GraphInput
  ): Promise<SparsificationResult> {
    const edges = normalizeGraphEdges(input);
    if (!this.initialized) {
      await this.initialize();
    }

    switch (this.config.method) {
      case 'ppr':
        return this.pprSparsification(
          sourceNode,
          edges,
          this.config.topK,
          this.config.alpha
        );
      case 'random-walk':
        return this.randomWalkSparsification(
          sourceNode,
          edges,
          this.config.topK,
          this.config.numWalks,
          this.config.walkLength
        );
      case 'spectral':
        return this.spectralSparsification(edges, this.config.topK);
      case 'degree-based':
        return this.degreeBasedSparsification(edges, this.config.topK);
      default:
        throw new Error(`Unknown sparsification method: ${this.config.method}`);
    }
  }

  /**
   * Extract top-k indices from scores
   */
  private getTopK(scores: Float32Array, k: number): number[] {
    const indexed = Array.from(scores).map((score, idx) => ({ score, idx }));
    return indexed
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(({ idx }) => idx);
  }

  private pprRunOptions(): { maxIterations: number; tolerance: number } {
    return {
      maxIterations: this.config.maxIterations ?? 20,
      tolerance: this.config.convergenceThreshold ?? 1e-6,
    };
  }

  private pprResult(
    run: PprRun,
    edges: GraphEdges,
    topK: number,
    method: string,
    startTime: number
  ): SparsificationResult {
    const scores = Float32Array.from(run.scores);
    const totalEdges = this.countTotalEdges(edges);
    return {
      topKIndices: this.getTopK(scores, topK),
      scores,
      sparsityRatio: totalEdges > 0 ? topK / totalEdges : 0,
      method,
      executionTimeMs: performance.now() - startTime,
      metadata: {
        iterations: run.iterations,
        convergence: run.delta,
        totalNodes: edges.length,
        totalEdges,
      },
    };
  }

  /**
   * PPR JavaScript fallback implementation
   *
   * Power iteration over out-edges in float64, dangling mass back to the
   * source; the same algorithm as @ruvnet/sublinear-core.
   */
  private pprFallback(
    sourceNode: number,
    edges: GraphEdges,
    topK: number,
    alpha: number
  ): SparsificationResult {
    const startTime = performance.now();
    const run = personalizedPageRankInTree(sourceNode, toNativeAdjacency(edges), alpha, this.pprRunOptions());
    return this.pprResult(run, edges, topK, 'ppr-fallback', startTime);
  }

  /**
   * Degree-based sparsification fallback
   *
   * Simple heuristic: keep nodes with highest degree (most connections).
   */
  private degreeBasedSparsification(
    edges: GraphEdges,
    topK: number
  ): SparsificationResult {
    const startTime = performance.now();
    const n = edges.length;
    const degrees = new Float32Array(n);

    // Compute degrees
    for (let i = 0; i < n; i++) {
      const neighbors = edges[i];
      degrees[i] = neighbors ? neighbors.length : 0;
    }

    const topKIndices = this.getTopK(degrees, topK);
    const executionTimeMs = performance.now() - startTime;

    return {
      topKIndices,
      scores: degrees,
      sparsityRatio: topK / this.countTotalEdges(edges),
      method: 'degree-based',
      executionTimeMs,
      metadata: {
        totalNodes: n,
        totalEdges: this.countTotalEdges(edges),
      },
    };
  }

  /**
   * Count total edges in graph
   */
  private countTotalEdges(edges: GraphEdges): number {
    return edges.reduce((sum, neighbors) => sum + (neighbors?.length || 0), 0);
  }

  /** Which implementation answers PPR queries. */
  getPprRuntime(): 'sublinear-core' | 'napi' | 'wasm' | 'fallback' {
    return this.sublinearCore?.runtime ?? this.pprBackend?.runtime ?? 'fallback';
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<SparsificationConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): SparsificationConfig {
    return { ...this.config };
  }

  /**
   * Reset to default configuration
   */
  resetConfig(): void {
    this.config = {
      method: this.config.method,
      topK: this.config.topK,
      alpha: 0.15,
      numWalks: 100,
      walkLength: 10,
      convergenceThreshold: 1e-6,
      maxIterations: 20,
    };
  }
}
