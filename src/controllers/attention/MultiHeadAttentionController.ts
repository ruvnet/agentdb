/**
 * MultiHeadAttentionController - Multi-Head Attention for Memory Systems
 *
 * Implements multi-head attention that projects queries and memory entries
 * into multiple subspaces, computes attention in each, and aggregates the
 * results. This allows the model to attend to different aspects of the
 * information simultaneously.
 *
 * Features:
 * - Configurable number of attention heads
 * - Multiple aggregation strategies (average, max, concat)
 * - Per-head dimension control
 * - Parallel attention computation
 */

import type { VectorBackend } from '../../backends/VectorBackend.js';

/**
 * Configuration for multi-head attention computation
 */
export interface MultiHeadAttentionConfig {
  /** Number of attention heads */
  numHeads?: number;
  /** Dimension per head (auto-calculated if not specified) */
  headDim?: number;
  /** Number of top results to return per head */
  topK?: number;
  /** Minimum attention score threshold (0-1) */
  minScore?: number;
  /** Temperature for softmax scaling */
  temperature?: number;
  /** Aggregation strategy for combining head outputs */
  aggregation?: 'average' | 'max' | 'concat' | 'weighted';
}

/**
 * Attention output from a single head
 */
export interface HeadAttentionOutput {
  /** Head index */
  headIndex: number;
  /** Attended output from this head */
  attended: number[];
  /** Top attention scores for this head */
  topScores: { id: string; score: number }[];
}

/**
 * Result of multi-head attention computation
 */
export interface MultiHeadAttentionResult {
  /** Per-head attention outputs */
  heads: HeadAttentionOutput[];
  /** Aggregated attended output */
  attended: number[];
  /** Overall top scores across all heads */
  aggregatedScores?: { id: string; score: number }[];
  /** Execution time in milliseconds */
  executionTimeMs: number;
}

/**
 * Memory entry with embedding for multi-head attention
 */
export interface MemoryEntry {
  id: string;
  embedding: number[];
  content?: string;
  metadata?: Record<string, any>;
}

/**
 * Multi-head attention controller for computing parallel attention patterns
 */
export class MultiHeadAttentionController {
  private vectorBackend: VectorBackend | null;
  private memoryStore: Map<string, MemoryEntry>;
  private dimension: number;
  private config: MultiHeadAttentionConfig;
  private headProjections: Float32Array[];

  constructor(
    vectorBackend: VectorBackend | null = null,
    config: MultiHeadAttentionConfig = {}
  ) {
    this.vectorBackend = vectorBackend;
    this.memoryStore = new Map();
    this.dimension = 0;
    this.headProjections = [];
    this.config = {
      numHeads: 8,
      topK: 10,
      minScore: 0.0,
      temperature: 1.0,
      aggregation: 'average',
      ...config
    };
    if (config.numHeads !== undefined && (!Number.isInteger(config.numHeads) || config.numHeads <= 0)) {
      throw new Error('numHeads must be a positive integer');
    }
    if (config.headDim !== undefined && (!Number.isInteger(config.headDim) || config.headDim <= 0)) {
      throw new Error('headDim must be a positive integer');
    }
  }

  /**
   * Per-head subspace size. Clamped to >= 1 so a vector narrower than the
   * head count still yields a non-empty projection instead of all-zero heads.
   */
  private resolveHeadDim(dimension: number): number {
    return this.config.headDim || Math.max(1, Math.floor(dimension / (this.config.numHeads || 8)));
  }

  /**
   * Initialize head projections for a given dimension
   */
  private initializeProjections(dimension: number): void {
    const numHeads = this.config.numHeads || 8;
    const headDim = this.resolveHeadDim(dimension);

    this.headProjections = [];

    // Create random projection matrices for each head
    // In practice, these would be learned parameters
    for (let h = 0; h < numHeads; h++) {
      const projection = new Float32Array(dimension * headDim);

      // Xavier initialization
      const scale = Math.sqrt(2.0 / (dimension + headDim));
      for (let i = 0; i < projection.length; i++) {
        projection[i] = (Math.random() * 2 - 1) * scale;
      }

      this.headProjections.push(projection);
    }
  }

  /**
   * Add a memory entry to the attention context
   */
  addMemory(entry: MemoryEntry): void {
    if (!entry.id || !entry.embedding || entry.embedding.length === 0) {
      throw new Error('Memory entry must have id and non-empty embedding');
    }

    if (this.dimension === 0) {
      this.dimension = entry.embedding.length;
      this.initializeProjections(this.dimension);
    } else if (entry.embedding.length !== this.dimension) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimension}, got ${entry.embedding.length}`
      );
    }

    this.memoryStore.set(entry.id, entry);

    if (this.vectorBackend) {
      const embedding = new Float32Array(entry.embedding);
      this.vectorBackend.insert(entry.id, embedding, entry.metadata);
    }
  }

  /**
   * Remove a memory entry
   */
  removeMemory(id: string): boolean {
    return this.memoryStore.delete(id);
  }

  /**
   * Clear all memories
   */
  clearMemories(): void {
    this.memoryStore.clear();
    this.dimension = 0;
    this.headProjections = [];
  }

  /**
   * Project a vector using a head's projection matrix
   */
  private projectVector(vector: number[], headIndex: number): number[] {
    const projection = this.headProjections[headIndex];
    if (!projection) {
      // If no projection available, return sliced view
      const numHeads = this.config.numHeads || 8;
      const headDim = Math.floor(vector.length / numHeads);
      const start = headIndex * headDim;
      return vector.slice(start, start + headDim);
    }

    const headDim = this.resolveHeadDim(this.dimension);
    const result = new Array(headDim).fill(0);

    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < vector.length; j++) {
        result[i] += vector[j] * projection[j * headDim + i];
      }
    }

    return result;
  }

  /**
   * Compute multi-head attention for a query vector
   *
   * @param query - Query vector for attention computation
   * @param options - Override default configuration options
   * @returns Multi-head attention result with per-head outputs and aggregated result
   */
  async computeMultiHeadAttention(
    query: number[],
    options: MultiHeadAttentionConfig = {}
  ): Promise<MultiHeadAttentionResult> {
    const startTime = performance.now();

    // Validate input
    if (!query || !Array.isArray(query)) {
      throw new Error('Query must be a non-null array');
    }

    const config = { ...this.config, ...options };
    const {
      numHeads = 8,
      topK = 10,
      minScore = 0.0,
      temperature = 1.0,
      aggregation = 'average'
    } = config;

    // Handle empty memory
    if (this.memoryStore.size === 0) {
      const emptyHeads: HeadAttentionOutput[] = Array(numHeads).fill(null).map((_, i) => ({
        headIndex: i,
        attended: [...query].slice(
          i * Math.floor(query.length / numHeads),
          (i + 1) * Math.floor(query.length / numHeads)
        ),
        topScores: []
      }));

      return {
        heads: emptyHeads,
        attended: [...query],
        executionTimeMs: performance.now() - startTime
      };
    }

    // Validate dimension
    if (this.dimension > 0 && query.length !== this.dimension) {
      throw new Error(
        `Query dimension mismatch: expected ${this.dimension}, got ${query.length}`
      );
    }

    // Initialize projections if needed
    if (this.headProjections.length === 0) {
      this.initializeProjections(query.length);
    }

    // Compute attention for each head
    const heads: HeadAttentionOutput[] = [];
    const allScores: Map<string, number> = new Map();

    for (let h = 0; h < numHeads; h++) {
      const headResult = await this.computeHeadAttention(
        query, h, topK, minScore, temperature
      );
      heads.push(headResult);

      // Aggregate scores across heads
      for (const score of headResult.topScores) {
        const existing = allScores.get(score.id) || 0;
        allScores.set(score.id, existing + score.score);
      }
    }

    // Aggregate attended outputs from all heads
    const attended = this.aggregateHeadOutputs(query, heads, aggregation);

    // Sort aggregated scores
    const aggregatedScores = Array.from(allScores.entries())
      .map(([id, score]) => ({ id, score: score / numHeads }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    const executionTimeMs = performance.now() - startTime;

    return {
      heads,
      attended,
      aggregatedScores,
      executionTimeMs
    };
  }

  /**
   * Compute attention for a single head
   */
  private async computeHeadAttention(
    query: number[],
    headIndex: number,
    topK: number,
    minScore: number,
    temperature: number
  ): Promise<HeadAttentionOutput> {
    // Project query to head subspace
    const projectedQuery = this.projectVector(query, headIndex);
    const scale = 1.0 / Math.sqrt(projectedQuery.length);

    // Compute scores for all memories
    const scores: { id: string; score: number; projectedEmbedding: number[] }[] = [];

    for (const [id, entry] of this.memoryStore.entries()) {
      const projectedKey = this.projectVector(entry.embedding, headIndex);

      // Scaled dot-product attention
      let dotProduct = 0;
      for (let i = 0; i < projectedQuery.length; i++) {
        dotProduct += projectedQuery[i] * projectedKey[i];
      }

      const scaledScore = dotProduct * scale / temperature;
      scores.push({ id, score: scaledScore, projectedEmbedding: projectedKey });
    }

    // Apply softmax
    const normalizedScores = this.applySoftmax(scores);

    // Filter and sort
    const filteredScores = normalizedScores
      .filter(item => item.normalizedScore >= minScore)
      .sort((a, b) => b.normalizedScore - a.normalizedScore)
      .slice(0, topK);

    // Compute attended output for this head
    const headDim = projectedQuery.length;
    const attended = new Array(headDim).fill(0);

    for (const item of normalizedScores) {
      for (let i = 0; i < headDim; i++) {
        attended[i] += item.normalizedScore * item.projectedEmbedding[i];
      }
    }

    return {
      headIndex,
      attended,
      topScores: filteredScores.map(item => ({
        id: item.id,
        score: item.normalizedScore
      }))
    };
  }

  /**
   * Apply softmax to scores
   */
  private applySoftmax(
    items: { id: string; score: number; projectedEmbedding: number[] }[]
  ): { id: string; normalizedScore: number; projectedEmbedding: number[] }[] {
    if (items.length === 0) return [];

    const maxScore = Math.max(...items.map(item => item.score));
    const expScores = items.map(item => ({
      ...item,
      expScore: Math.exp(item.score - maxScore)
    }));

    const sumExp = expScores.reduce((sum, item) => sum + item.expScore, 0);

    if (sumExp === 0 || !isFinite(sumExp)) {
      return items.map(item => ({
        ...item,
        normalizedScore: 1 / items.length
      }));
    }

    return expScores.map(item => ({
      id: item.id,
      normalizedScore: item.expScore / sumExp,
      projectedEmbedding: item.projectedEmbedding
    }));
  }

  /**
   * Aggregate outputs from all attention heads
   */
  private aggregateHeadOutputs(
    query: number[],
    heads: HeadAttentionOutput[],
    strategy: 'average' | 'max' | 'concat' | 'weighted'
  ): number[] {
    if (heads.length === 0) {
      return [...query];
    }

    const dimension = query.length;
    const numHeads = heads.length;
    const headDim = heads[0]?.attended.length || Math.floor(dimension / numHeads);

    switch (strategy) {
      case 'concat':
        // Concatenate all head outputs (may change dimension)
        const concatResult: number[] = [];
        for (const head of heads) {
          concatResult.push(...head.attended);
        }
        // Pad or truncate to original dimension
        if (concatResult.length < dimension) {
          return [...concatResult, ...new Array(dimension - concatResult.length).fill(0)];
        }
        return concatResult.slice(0, dimension);

      case 'max':
        // Take element-wise max across reconstructed head outputs
        const maxResult = new Array(dimension).fill(-Infinity);
        for (const head of heads) {
          const reconstructed = this.reconstructFromHead(head.attended, head.headIndex, dimension);
          for (let i = 0; i < dimension; i++) {
            maxResult[i] = Math.max(maxResult[i], reconstructed[i]);
          }
        }
        return maxResult.map(v => isFinite(v) ? v : 0);

      case 'weighted':
        // Weight by average attention scores
        const weightedResult = new Array(dimension).fill(0);
        const weights: number[] = [];

        for (const head of heads) {
          const avgScore = head.topScores.reduce((sum, s) => sum + s.score, 0) /
            Math.max(1, head.topScores.length);
          weights.push(avgScore);
        }

        const totalWeight = weights.reduce((sum, w) => sum + w, 0) || 1;

        for (let h = 0; h < heads.length; h++) {
          const reconstructed = this.reconstructFromHead(heads[h].attended, h, dimension);
          const weight = weights[h] / totalWeight;
          for (let i = 0; i < dimension; i++) {
            weightedResult[i] += weight * reconstructed[i];
          }
        }
        return weightedResult;

      case 'average':
      default:
        // Average across reconstructed head outputs
        const avgResult = new Array(dimension).fill(0);
        for (const head of heads) {
          const reconstructed = this.reconstructFromHead(head.attended, head.headIndex, dimension);
          for (let i = 0; i < dimension; i++) {
            avgResult[i] += reconstructed[i] / numHeads;
          }
        }
        return avgResult;
    }
  }

  /**
   * Reconstruct full-dimension vector from head output
   */
  private reconstructFromHead(
    headOutput: number[],
    headIndex: number,
    targetDim: number
  ): number[] {
    // Simple reconstruction by placing head output in appropriate position
    const numHeads = this.config.numHeads || 8;
    const headDim = Math.max(1, Math.floor(targetDim / numHeads));
    const result = new Array(targetDim).fill(0);

    const start = headIndex * headDim;
    const len = Math.min(headOutput.length, targetDim - start);

    for (let i = 0; i < len; i++) {
      result[start + i] = headOutput[i];
    }

    return result;
  }

  /**
   * Get statistics about the controller
   */
  getStats(): {
    memoryCount: number;
    dimension: number;
    numHeads: number;
    headDim: number;
    hasVectorBackend: boolean;
  } {
    const numHeads = this.config.numHeads || 8;
    const headDim = this.config.headDim ||
      (this.dimension > 0 ? Math.floor(this.dimension / numHeads) : 0);

    return {
      memoryCount: this.memoryStore.size,
      dimension: this.dimension,
      numHeads,
      headDim,
      hasVectorBackend: this.vectorBackend !== null
    };
  }
}
