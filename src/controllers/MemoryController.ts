/**
 * MemoryController - Unified Memory Management with Attention Mechanisms
 *
 * Provides a unified interface for memory storage, retrieval, and search
 * with integrated attention mechanisms for enhanced relevance scoring.
 *
 * Features:
 * - CRUD operations for memory entries
 * - Vector similarity search
 * - Attention-enhanced retrieval
 * - Temporal and importance weighting
 * - Multiple namespace support
 * - Integration with vector backends
 */

import type { VectorBackend } from '../backends/VectorBackend.js';
import {
  assertEmbeddingSpaceCompatible,
  type EmbeddingSpaceIdentity,
  type MemoryProvenance
} from '../embedding/EmbeddingSpaceIdentity.js';
import { SelfAttentionController } from './attention/SelfAttentionController.js';
import { CrossAttentionController } from './attention/CrossAttentionController.js';
import { MultiHeadAttentionController } from './attention/MultiHeadAttentionController.js';

/**
 * Configuration for the MemoryController
 */
export interface MemoryControllerConfig {
  /** Default namespace for memories */
  namespace?: string;
  /** Enable attention-enhanced retrieval */
  enableAttention?: boolean;
  /** Number of attention heads for multi-head attention */
  numHeads?: number;
  /** Default top-k for search operations */
  defaultTopK?: number;
  /** Default similarity threshold */
  defaultThreshold?: number;
  /** Authoritative embedding space for corpus mutations */
  embeddingSpaceIdentity?: EmbeddingSpaceIdentity;
}

/**
 * Memory entry structure
 */
export interface Memory {
  /** Unique identifier */
  id: string;
  /** Text content */
  content?: string;
  /** Embedding vector */
  embedding: number[];
  /** Importance score (0-1) */
  importance?: number;
  /** Creation timestamp */
  timestamp?: number;
  /** Additional metadata */
  metadata?: Record<string, any>;
  /** Typed origin and trust information for this memory */
  provenance?: MemoryProvenance;
  /** Space in which embedding was produced */
  embeddingSpaceIdentity?: EmbeddingSpaceIdentity;
}

/**
 * Search options for memory retrieval
 */
export interface SearchOptions {
  /** Number of top results to return */
  topK?: number;
  /** Minimum similarity threshold */
  threshold?: number;
  /** Metadata filters */
  filter?: Record<string, any>;
  /** Use attention for ranking */
  useAttention?: boolean;
  /** Weight recent memories higher */
  temporalWeight?: number;
  /** Weight by importance score */
  weighByImportance?: boolean;
  /**
   * Identity of the supplied query vector. A mismatch is allowed for
   * vector-only reads, but never for corpus mutation or cache reuse.
   */
  embeddingSpaceIdentity?: EmbeddingSpaceIdentity;
}

/**
 * Search result with scores
 */
export interface SearchResult extends Memory {
  /** Similarity score */
  score: number;
  /** Attention score (if attention enabled) */
  attentionScore?: number;
}

/**
 * Result from attention-enhanced retrieval
 */
export interface AttentionRetrievalResult extends SearchResult {
  /** Combined attention score from all mechanisms */
  attentionScore: number;
}

/**
 * MemoryController - Main class for memory management
 */
export class MemoryController {
  private vectorBackend: VectorBackend | null;
  private memories: Map<string, Memory>;
  private config: MemoryControllerConfig;

  // Attention controllers
  private selfAttention: SelfAttentionController;
  private crossAttention: CrossAttentionController;
  private multiHeadAttention: MultiHeadAttentionController;

  constructor(
    vectorBackend: VectorBackend | null = null,
    config: MemoryControllerConfig = {}
  ) {
    this.vectorBackend = vectorBackend;
    this.memories = new Map();
    this.config = {
      namespace: 'default',
      enableAttention: true,
      numHeads: 8,
      defaultTopK: 10,
      defaultThreshold: 0.0,
      ...config
    };

    // Initialize attention controllers
    this.selfAttention = new SelfAttentionController(vectorBackend);
    this.crossAttention = new CrossAttentionController(vectorBackend);
    this.multiHeadAttention = new MultiHeadAttentionController(vectorBackend, {
      numHeads: this.config.numHeads
    });
  }

  /**
   * Store a memory entry
   */
  async store(memory: Memory, namespace?: string): Promise<void> {
    if (!memory.id || !memory.embedding || memory.embedding.length === 0) {
      throw new Error('Memory must have id and non-empty embedding');
    }
    this.assertMutationIdentity(memory.embeddingSpaceIdentity);

    // Add timestamp if not provided
    const storedMemory: Memory = {
      ...memory,
      timestamp: memory.timestamp ?? Date.now()
    };

    // Store in local map
    this.memories.set(memory.id, storedMemory);

    // Store in vector backend if available
    if (this.vectorBackend) {
      const embedding = new Float32Array(memory.embedding);
      this.vectorBackend.insert(memory.id, embedding, {
        ...memory.metadata,
        provenance: memory.provenance,
        embeddingSpaceIdentity: memory.embeddingSpaceIdentity,
        namespace: namespace || this.config.namespace
      });
    }

    // Add to attention controllers
    const attentionEntry = {
      id: memory.id,
      embedding: memory.embedding,
      content: memory.content,
      metadata: memory.metadata
    };

    this.selfAttention.addMemory(attentionEntry);

    if (namespace) {
      this.crossAttention.addToContext(namespace, attentionEntry);
    } else {
      this.crossAttention.addToContext(this.config.namespace || 'default', attentionEntry);
    }

    this.multiHeadAttention.addMemory(attentionEntry);
  }

  /**
   * Retrieve a memory by ID
   */
  async retrieve(id: string): Promise<Memory | undefined> {
    return this.memories.get(id);
  }

  /**
   * Update an existing memory
   */
  async update(id: string, updates: Partial<Memory>): Promise<boolean> {
    const existing = this.memories.get(id);
    if (!existing) {
      return false;
    }
    this.assertMutationIdentity(
      updates.embeddingSpaceIdentity ?? existing.embeddingSpaceIdentity
    );

    const updated: Memory = {
      ...existing,
      ...updates,
      id // Ensure ID cannot be changed
    };

    // Re-store with updates
    await this.delete(id);
    await this.store(updated);

    return true;
  }

  /**
   * Delete a memory by ID
   */
  async delete(id: string): Promise<boolean> {
    const existed = this.memories.delete(id);

    // Remove from attention controllers
    this.selfAttention.removeMemory(id);
    this.multiHeadAttention.removeMemory(id);

    // Remove from all cross-attention contexts
    for (const contextName of this.crossAttention.listContexts()) {
      this.crossAttention.removeFromContext(contextName, id);
    }

    return existed;
  }

  /**
   * Search for similar memories
   */
  async search(
    query: number[],
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    if (this.config.embeddingSpaceIdentity && options.embeddingSpaceIdentity) {
      assertEmbeddingSpaceCompatible(
        this.config.embeddingSpaceIdentity,
        options.embeddingSpaceIdentity,
        'vector-read'
      );
    }
    const {
      topK = this.config.defaultTopK || 10,
      threshold = this.config.defaultThreshold || 0.0,
      filter,
      temporalWeight = 0,
      weighByImportance = false
    } = options;

    // Compute similarity scores
    const results: SearchResult[] = [];

    for (const [id, memory] of this.memories.entries()) {
      // Apply metadata filter if provided
      if (filter && !this.matchesFilter(memory.metadata || {}, filter)) {
        continue;
      }

      // Compute cosine similarity
      let score = this.cosineSimilarity(query, memory.embedding);

      // Apply temporal weighting
      if (temporalWeight > 0 && memory.timestamp) {
        const age = Date.now() - memory.timestamp;
        const decayFactor = Math.exp(-temporalWeight * age / (24 * 60 * 60 * 1000));
        score = score * (1 - temporalWeight) + score * temporalWeight * decayFactor;
      }

      // Apply importance weighting
      if (weighByImportance && memory.importance !== undefined) {
        score = score * (0.5 + 0.5 * memory.importance);
      }

      if (score >= threshold) {
        results.push({
          ...memory,
          score
        });
      }
    }

    // Sort by score and return top-k
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private assertMutationIdentity(identity?: EmbeddingSpaceIdentity): void {
    const expected = this.config.embeddingSpaceIdentity;
    if (!expected) return;
    if (!identity) {
      throw new Error(
        'Memory mutation requires embeddingSpaceIdentity when the controller has an authoritative space'
      );
    }
    assertEmbeddingSpaceCompatible(expected, identity, 'corpus-mutation');
  }

  /**
   * Retrieve memories with attention-enhanced scoring
   */
  async retrieveWithAttention(
    query: number[],
    options: SearchOptions = {}
  ): Promise<AttentionRetrievalResult[]> {
    const {
      topK = this.config.defaultTopK || 10,
      threshold = this.config.defaultThreshold || 0.0,
      useAttention = true,
      temporalWeight = 0,
      weighByImportance = false
    } = options;

    // First get base search results
    const baseResults = await this.search(query, {
      ...options,
      topK: topK * 2 // Get more candidates for attention ranking
    });

    if (!useAttention || !this.config.enableAttention) {
      return baseResults.map(r => ({
        ...r,
        attentionScore: r.score
      }));
    }

    // Compute attention scores
    const selfAttentionResult = await this.selfAttention.computeAttention(query, {
      topK: topK * 2,
      minScore: 0
    });

    const multiHeadResult = await this.multiHeadAttention.computeMultiHeadAttention(
      query,
      { topK: topK * 2 }
    );

    // Build attention score map
    const attentionScores = new Map<string, number>();

    for (const score of selfAttentionResult.scores) {
      attentionScores.set(score.id, (attentionScores.get(score.id) || 0) + score.score);
    }

    if (multiHeadResult.aggregatedScores) {
      for (const score of multiHeadResult.aggregatedScores) {
        attentionScores.set(score.id, (attentionScores.get(score.id) || 0) + score.score);
      }
    }

    // Combine base similarity with attention scores
    const results: AttentionRetrievalResult[] = [];

    for (const baseResult of baseResults) {
      const attentionScore = attentionScores.get(baseResult.id) || 0;

      // Combine scores (weighted average)
      const combinedScore = 0.5 * baseResult.score + 0.5 * (attentionScore / 2);

      // Apply temporal weighting
      let finalScore = combinedScore;
      if (temporalWeight > 0 && baseResult.timestamp) {
        const age = Date.now() - baseResult.timestamp;
        const decayFactor = Math.exp(-temporalWeight * age / (24 * 60 * 60 * 1000));
        finalScore = combinedScore * (1 - temporalWeight) + combinedScore * temporalWeight * decayFactor;
      }

      // Apply importance weighting
      if (weighByImportance && baseResult.importance !== undefined) {
        finalScore = finalScore * (0.5 + 0.5 * baseResult.importance);
      }

      if (finalScore >= threshold) {
        results.push({
          ...baseResult,
          score: finalScore,
          attentionScore
        });
      }
    }

    // Sort by combined score and return top-k
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Compute cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;

    return dotProduct / denominator;
  }

  /**
   * Check if metadata matches filter criteria
   */
  private matchesFilter(
    metadata: Record<string, any>,
    filter: Record<string, any>
  ): boolean {
    for (const [key, value] of Object.entries(filter)) {
      if (metadata[key] !== value) {
        return false;
      }
    }
    return true;
  }

  /**
   * Get all memories (for iteration/export)
   */
  getAllMemories(): Memory[] {
    return Array.from(this.memories.values());
  }

  /**
   * Get memory count
   */
  get count(): number {
    return this.memories.size;
  }

  /**
   * Clear all memories
   */
  clear(): void {
    this.memories.clear();
    this.selfAttention.clearMemories();
    this.crossAttention.clearAllContexts();
    this.multiHeadAttention.clearMemories();
  }

  /**
   * Get the self-attention controller for direct access
   */
  getSelfAttentionController(): SelfAttentionController {
    return this.selfAttention;
  }

  /**
   * Get the cross-attention controller for direct access
   */
  getCrossAttentionController(): CrossAttentionController {
    return this.crossAttention;
  }

  /**
   * Get the multi-head attention controller for direct access
   */
  getMultiHeadAttentionController(): MultiHeadAttentionController {
    return this.multiHeadAttention;
  }

  /**
   * Get controller statistics
   */
  getStats(): {
    memoryCount: number;
    selfAttention: ReturnType<SelfAttentionController['getStats']>;
    crossAttention: ReturnType<CrossAttentionController['getStats']>;
    multiHeadAttention: ReturnType<MultiHeadAttentionController['getStats']>;
  } {
    return {
      memoryCount: this.memories.size,
      selfAttention: this.selfAttention.getStats(),
      crossAttention: this.crossAttention.getStats(),
      multiHeadAttention: this.multiHeadAttention.getStats()
    };
  }
}
