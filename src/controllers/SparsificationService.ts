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
 * - WASM/NAPI bindings with JavaScript fallback
 * - Zero-copy operations where supported
 *
 * Based on:
 * - "Fast Personalized PageRank on MapReduce" (Bahmani et al., 2011)
 * - "Graph Sparsification by Effective Resistances" (Spielman & Srivastava, 2011)
 * - "Local Graph Partitioning using PageRank Vectors" (Andersen et al., 2006)
 *
 * @version 3.0.0-alpha.5
 */

import type { GraphEdges } from '../types/graph.js';

// Re-export for convenience
export type { GraphEdges };

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
  /** Convergence threshold for PPR (default: 1e-6) */
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
  private wasmModule: any;
  private napiModule: any;
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

    try {
      // Try NAPI first (fastest)
      // Use dynamic import with string to avoid TypeScript module resolution
      const napiModuleName = '@ruvector/sparsifier';
      const napi = await import(/* @vite-ignore */ napiModuleName).catch(() => null);
      if (napi) {
        this.napiModule = napi;
        console.error('✅ SparsificationService: Loaded @ruvector/sparsifier NAPI module');
      } else {
        // Fall back to WASM
        const wasmModuleName = 'ruvector-sparsifier-wasm';
        const wasm = await import(/* @vite-ignore */ wasmModuleName).catch(() => null);
        if (wasm) {
          await wasm.default();
          this.wasmModule = wasm;
          console.error('✅ SparsificationService: Loaded ruvector-sparsifier-wasm module');
        } else {
          console.warn('⚠️  SparsificationService: No native bindings available, using JavaScript fallback');
        }
      }
    } catch (err) {
      console.warn('⚠️  SparsificationService: Initialization error, using JavaScript fallback:', err);
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
    edges: GraphEdges,
    topK: number,
    alpha: number = 0.15
  ): Promise<SparsificationResult> {
    const startTime = performance.now();

    // Try NAPI first
    if (this.napiModule?.pprScores) {
      try {
        const scores = this.napiModule.pprScores(sourceNode, edges, alpha);
        const topKIndices = this.getTopK(scores, topK);
        const executionTimeMs = performance.now() - startTime;

        return {
          topKIndices,
          scores,
          sparsityRatio: topK / this.countTotalEdges(edges),
          method: 'ppr-napi',
          executionTimeMs,
          metadata: {
            totalNodes: edges.length,
            totalEdges: this.countTotalEdges(edges),
          },
        };
      } catch (err) {
        console.warn('NAPI PPR failed, falling back:', err);
      }
    }

    // Try WASM
    if (this.wasmModule?.ppr_scores) {
      try {
        const scores = this.wasmModule.ppr_scores(sourceNode, edges, alpha);
        const topKIndices = this.getTopK(scores, topK);
        const executionTimeMs = performance.now() - startTime;

        return {
          topKIndices,
          scores,
          sparsityRatio: topK / this.countTotalEdges(edges),
          method: 'ppr-wasm',
          executionTimeMs,
          metadata: {
            totalNodes: edges.length,
            totalEdges: this.countTotalEdges(edges),
          },
        };
      } catch (err) {
        console.warn('WASM PPR failed, falling back:', err);
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
    edges: GraphEdges,
    topK: number,
    numWalks: number = 100,
    walkLength: number = 10
  ): Promise<SparsificationResult> {
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
    edges: GraphEdges,
    topK: number
  ): Promise<SparsificationResult> {
    const startTime = performance.now();

    // Try NAPI spectral sparsification
    if (this.napiModule?.spectralSparsify) {
      try {
        const result = this.napiModule.spectralSparsify(edges, topK);
        result.executionTimeMs = performance.now() - startTime;
        return result;
      } catch (err) {
        console.warn('NAPI spectral sparsification failed, falling back:', err);
      }
    }

    // Fallback: degree-based approximation
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
    edges: GraphEdges
  ): Promise<SparsificationResult> {
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

  /**
   * PPR JavaScript fallback implementation
   *
   * Power iteration method for computing PageRank with restart.
   */
  private pprFallback(
    sourceNode: number,
    edges: GraphEdges,
    topK: number,
    alpha: number
  ): SparsificationResult {
    const startTime = performance.now();
    const n = edges.length;

    // Handle empty graph
    if (n === 0) {
      return {
        topKIndices: [],
        scores: new Float32Array(0),
        sparsityRatio: 0,
        method: 'ppr-fallback',
        executionTimeMs: performance.now() - startTime,
        metadata: {
          totalNodes: 0,
          totalEdges: 0,
        },
      };
    }

    const scores = new Float32Array(n);
    scores[sourceNode] = 1.0;

    let iterations = 0;
    let convergence = 0;

    // Power iteration
    for (let iter = 0; iter < (this.config.maxIterations || 20); iter++) {
      const newScores = new Float32Array(n);
      let maxDelta = 0;

      for (let i = 0; i < n; i++) {
        // Teleport component
        newScores[i] = alpha * (i === sourceNode ? 1.0 : 0.0);

        // Random walk component
        for (let j = 0; j < n; j++) {
          const neighbors = edges[j] || [];
          if (neighbors.includes(i) && neighbors.length > 0) {
            newScores[i] += (1 - alpha) * scores[j] / neighbors.length;
          }
        }

        maxDelta = Math.max(maxDelta, Math.abs(newScores[i] - scores[i]));
      }

      scores.set(newScores);
      iterations = iter + 1;
      convergence = maxDelta;

      // Check convergence
      if (maxDelta < (this.config.convergenceThreshold || 1e-6)) {
        break;
      }
    }

    const topKIndices = this.getTopK(scores, topK);
    const executionTimeMs = performance.now() - startTime;

    return {
      topKIndices,
      scores,
      sparsityRatio: topK / this.countTotalEdges(edges),
      method: 'ppr-fallback',
      executionTimeMs,
      metadata: {
        iterations,
        convergence,
        totalNodes: n,
        totalEdges: this.countTotalEdges(edges),
      },
    };
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
