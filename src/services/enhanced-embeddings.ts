/**
 * Enhanced Embedding Service with RuVector Integration
 *
 * A comprehensive embedding service supporting:
 * - Multiple providers (@huggingface/transformers, OpenAI, Cohere, custom)
 * - LRU cache with O(1) operations using doubly-linked list
 * - Batch processing with semaphore-controlled parallelism
 * - Text pre-processing pipeline (normalization, chunking, deduplication)
 * - Direct RuVector storage and hybrid search
 * - Lazy model loading with runtime switching
 */

import type { VectorBackend, SearchResult as VectorSearchResult } from '../backends/VectorBackend.js';

// ============================================================================
// Performance & Security Constants
// ============================================================================

/** @inline Maximum allowed vector dimension for bounds checking */
export const MAX_VECTOR_DIMENSION = 4096;

/** @inline Maximum allowed cache size to prevent memory exhaustion */
export const MAX_CACHE_SIZE = 100_000;

/** @inline Default cache size for embedding storage */
export const DEFAULT_CACHE_SIZE = 10000;

/** @inline Maximum text length to prevent DoS via large inputs */
const MAX_TEXT_LENGTH = 100_000;

/** @inline Maximum batch size to prevent memory exhaustion */
export const MAX_BATCH_SIZE = 10000;

/** @inline API request timeout in milliseconds */
const API_TIMEOUT_MS = 30_000;

/** @inline Maximum concurrent API requests */
const MAX_CONCURRENT_REQUESTS = 10;

/** @inline Small epsilon for numerical stability */
const EPSILON = 1e-10;

/** Allowed API hosts for external calls */
const ALLOWED_API_HOSTS = [
  'api.openai.com',
  'api.cohere.ai',
];

// ============================================================================
// Security Helpers
// ============================================================================

/**
 * Mask sensitive data in error messages
 */
function maskApiKey(key: string | undefined): string {
  if (!key) return '[not set]';
  if (key.length <= 8) return '***';
  return `${key.slice(0, 4)}...${key.slice(-4)}`;
}

/**
 * Validate URL is allowed for external API calls
 */
function validateApiUrl(url: string): void {
  try {
    const parsed = new URL(url);
    if (!ALLOWED_API_HOSTS.includes(parsed.host)) {
      throw new Error(`API host not allowed: ${parsed.host}`);
    }
    if (parsed.protocol !== 'https:') {
      throw new Error('Only HTTPS connections are allowed');
    }
  } catch (e) {
    if (e instanceof Error && e.message.includes('not allowed')) {
      throw e;
    }
    throw new Error('Invalid API URL');
  }
}

/**
 * Create a fetch request with timeout
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = API_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Supported embedding models
 */
export type SupportedModel =
  | 'all-MiniLM-L6-v2'
  | 'all-mpnet-base-v2'
  | 'bge-small-en-v1.5'
  | string;

/**
 * Embedding provider types
 */
export type EmbeddingProvider = 'transformers' | 'openai' | 'cohere' | 'custom';

/**
 * Configuration for the enhanced embedding service (user input)
 */
export interface EnhancedEmbeddingConfig {
  /** Embedding provider (default: 'transformers') */
  provider?: EmbeddingProvider;

  /** Model name (default: 'all-MiniLM-L6-v2') */
  model?: SupportedModel;

  /** Vector dimension (auto-detected from model if not specified) */
  dimension?: number;

  /** API key for OpenAI/Cohere providers */
  apiKey?: string;

  /** Custom embedding function for 'custom' provider */
  customEmbedder?: CustomEmbedder;

  /** LRU cache configuration */
  cache?: Partial<CacheConfig>;

  /** Batch processing configuration */
  batch?: Partial<BatchConfig>;

  /** Pre-processing pipeline configuration */
  preprocessing?: Partial<PreprocessingConfig>;

  /** RuVector backend for direct storage */
  vectorBackend?: VectorBackend;
}

/**
 * Internal resolved configuration (all defaults applied)
 */
interface ResolvedEmbeddingConfig {
  provider: EmbeddingProvider;
  model: SupportedModel;
  dimension: number;
  apiKey: string;
  customEmbedder?: CustomEmbedder;
  cache: CacheConfig;
  batch: BatchConfig;
  preprocessing: PreprocessingConfig;
  vectorBackend?: VectorBackend;
}

/**
 * Cache configuration
 */
export interface CacheConfig {
  /** Maximum number of cached embeddings (default: 10000) */
  maxSize: number;

  /** Enable cache statistics tracking (default: true) */
  trackStats: boolean;
}

/**
 * Batch processing configuration
 */
export interface BatchConfig {
  /** Batch size for processing (default: 32) */
  batchSize: number;

  /** Maximum concurrent batches (default: 4) */
  maxConcurrency: number;

  /** Enable progress callbacks (default: true) */
  enableProgress: boolean;
}

/**
 * Pre-processing pipeline configuration
 */
export interface PreprocessingConfig {
  /** Enable text normalization (default: true) */
  normalize: boolean;

  /** Maximum text length before chunking (default: 512 tokens approx) */
  maxLength: number;

  /** Chunk overlap for long texts (default: 50 chars) */
  chunkOverlap: number;

  /** Enable deduplication (default: true) */
  deduplicate: boolean;
}

/**
 * Custom embedding function signature
 */
export type CustomEmbedder = (text: string) => Promise<Float32Array>;

/**
 * Search result from the embedding service
 */
export interface SearchResult {
  /** Unique identifier */
  id: string;

  /** Original text (if stored) */
  text?: string;

  /** Similarity score (0-1, higher is better) */
  similarity: number;

  /** Raw distance from vector search */
  distance: number;

  /** Associated metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Statistics for the embedding service
 */
export interface EmbeddingStats {
  /** Total embeddings generated */
  totalEmbeddings: number;

  /** Cache statistics */
  cache: {
    hits: number;
    misses: number;
    hitRate: number;
    size: number;
    maxSize: number;
  };

  /** Batch processing statistics */
  batch: {
    totalBatches: number;
    averageBatchSize: number;
  };

  /** Model information */
  model: {
    name: string;
    dimension: number;
    provider: EmbeddingProvider;
    loaded: boolean;
  };

  /** Vector backend statistics (if connected) */
  vectorBackend?: {
    count: number;
    backend: string;
  };
}

/**
 * Progress callback for batch operations
 */
export type ProgressCallback = (progress: {
  current: number;
  total: number;
  percentage: number;
}) => void;

// ============================================================================
// LRU Cache Implementation with O(1) Operations
// ============================================================================

/**
 * Doubly-linked list node for LRU cache
 */
interface CacheNode {
  key: string;
  value: Float32Array;
  prev: CacheNode | null;
  next: CacheNode | null;
}

/**
 * LRU Cache with O(1) get/put operations using doubly-linked list + hash map
 */
class LRUCache {
  private capacity: number;
  private map: Map<string, CacheNode> = new Map();
  private head: CacheNode | null = null;
  private tail: CacheNode | null = null;

  // Statistics
  private _hits = 0;
  private _misses = 0;
  private trackStats: boolean;

  constructor(capacity: number, trackStats = true) {
    // Enforce maximum cache size for security
    this.capacity = Math.min(Math.max(1, capacity), MAX_CACHE_SIZE);
    this.trackStats = trackStats;
  }

  /**
   * Get value from cache - O(1)
   */
  get(key: string): Float32Array | undefined {
    const node = this.map.get(key);
    if (!node) {
      if (this.trackStats) this._misses++;
      return undefined;
    }

    if (this.trackStats) this._hits++;

    // Move to front (most recently used)
    this.moveToFront(node);
    return node.value;
  }

  /**
   * Put value in cache - O(1)
   */
  put(key: string, value: Float32Array): void {
    const existingNode = this.map.get(key);

    if (existingNode) {
      // Update existing node and move to front
      existingNode.value = value;
      this.moveToFront(existingNode);
      return;
    }

    // Create new node
    const newNode: CacheNode = {
      key,
      value,
      prev: null,
      next: this.head,
    };

    // Add to front
    if (this.head) {
      this.head.prev = newNode;
    }
    this.head = newNode;

    if (!this.tail) {
      this.tail = newNode;
    }

    this.map.set(key, newNode);

    // Evict if over capacity
    if (this.map.size > this.capacity) {
      this.evictLRU();
    }
  }

  /**
   * Check if key exists - O(1)
   */
  has(key: string): boolean {
    return this.map.has(key);
  }

  /**
   * Delete key from cache - O(1)
   */
  delete(key: string): boolean {
    const node = this.map.get(key);
    if (!node) return false;

    this.removeNode(node);
    this.map.delete(key);
    return true;
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.map.clear();
    this.head = null;
    this.tail = null;
    this._hits = 0;
    this._misses = 0;
  }

  /**
   * Get current size
   */
  get size(): number {
    return this.map.size;
  }

  /**
   * Get cache statistics
   */
  get stats(): { hits: number; misses: number; hitRate: number; size: number; maxSize: number } {
    const total = this._hits + this._misses;
    return {
      hits: this._hits,
      misses: this._misses,
      hitRate: total > 0 ? this._hits / total : 0,
      size: this.map.size,
      maxSize: this.capacity,
    };
  }

  /**
   * Move node to front of list (most recently used)
   */
  private moveToFront(node: CacheNode): void {
    if (node === this.head) return;

    // Remove from current position
    this.removeNode(node);

    // Add to front
    node.prev = null;
    node.next = this.head;
    if (this.head) {
      this.head.prev = node;
    }
    this.head = node;

    if (!this.tail) {
      this.tail = node;
    }
  }

  /**
   * Remove node from list
   */
  private removeNode(node: CacheNode): void {
    if (node.prev) {
      node.prev.next = node.next;
    } else {
      this.head = node.next;
    }

    if (node.next) {
      node.next.prev = node.prev;
    } else {
      this.tail = node.prev;
    }
  }

  /**
   * Evict least recently used entry
   */
  private evictLRU(): void {
    if (!this.tail) return;

    const lruKey = this.tail.key;
    this.removeNode(this.tail);
    this.map.delete(lruKey);
  }
}

// ============================================================================
// Semaphore for Concurrency Control
// ============================================================================

/**
 * Simple semaphore for controlling concurrent operations
 */
class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];
  private maxWaiting: number;

  constructor(permits: number, maxWaiting: number = 1000) {
    // Enforce maximum concurrent requests for security
    this.permits = Math.min(Math.max(1, permits), MAX_CONCURRENT_REQUESTS);
    this.maxWaiting = maxWaiting;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    // Prevent unbounded queue growth (deadlock protection)
    if (this.waiting.length >= this.maxWaiting) {
      throw new Error('Too many pending requests - semaphore queue full');
    }

    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    const next = this.waiting.shift();
    if (next) {
      next();
    } else {
      this.permits++;
    }
  }

  async withLock<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// ============================================================================
// Model Dimension Mapping
// ============================================================================

const MODEL_DIMENSIONS: Record<string, number> = {
  'all-MiniLM-L6-v2': 384,
  'Xenova/all-MiniLM-L6-v2': 384,
  'all-mpnet-base-v2': 768,
  'Xenova/all-mpnet-base-v2': 768,
  'bge-small-en-v1.5': 384,
  'BAAI/bge-small-en-v1.5': 384,
  'text-embedding-ada-002': 1536, // OpenAI
  'text-embedding-3-small': 1536, // OpenAI
  'text-embedding-3-large': 3072, // OpenAI
  'embed-english-v3.0': 1024, // Cohere
  'embed-multilingual-v3.0': 1024, // Cohere
};

/**
 * Get model dimension from model name
 */
function getModelDimension(model: string): number {
  return MODEL_DIMENSIONS[model] || MODEL_DIMENSIONS[`Xenova/${model}`] || 384;
}

// ============================================================================
// Enhanced Embedding Service Implementation
// ============================================================================

/**
 * Enhanced Embedding Service with RuVector Integration
 */
export class EnhancedEmbeddingService {
  private config: ResolvedEmbeddingConfig;
  private cache: LRUCache;
  private semaphore: Semaphore;
  private pipeline: any = null;
  private modelLoaded = false;

  // Statistics
  private totalEmbeddings = 0;
  private totalBatches = 0;
  private totalBatchItems = 0;

  constructor(config: EnhancedEmbeddingConfig = {}) {
    // Set defaults
    this.config = {
      provider: config.provider ?? 'transformers',
      model: config.model ?? 'all-MiniLM-L6-v2',
      dimension: config.dimension ?? getModelDimension(config.model ?? 'all-MiniLM-L6-v2'),
      apiKey: config.apiKey ?? '',
      customEmbedder: config.customEmbedder,
      cache: {
        maxSize: config.cache?.maxSize ?? 10000,
        trackStats: config.cache?.trackStats ?? true,
      },
      batch: {
        batchSize: config.batch?.batchSize ?? 32,
        maxConcurrency: config.batch?.maxConcurrency ?? 4,
        enableProgress: config.batch?.enableProgress ?? true,
      },
      preprocessing: {
        normalize: config.preprocessing?.normalize ?? true,
        maxLength: config.preprocessing?.maxLength ?? 2000,
        chunkOverlap: config.preprocessing?.chunkOverlap ?? 50,
        deduplicate: config.preprocessing?.deduplicate ?? true,
      },
      vectorBackend: config.vectorBackend,
    };

    // Initialize cache
    this.cache = new LRUCache(
      this.config.cache.maxSize,
      this.config.cache.trackStats
    );

    // Initialize semaphore for concurrency control
    this.semaphore = new Semaphore(this.config.batch.maxConcurrency);
  }

  // ==========================================================================
  // Core Embedding Methods
  // ==========================================================================

  /**
   * Generate embedding for a single text
   */
  async embed(text: string): Promise<Float32Array> {
    // Pre-process text
    const processedText = this.preprocessText(text);

    // Check cache
    const cacheKey = this.getCacheKey(processedText);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return cached;
    }

    // Generate embedding
    const embedding = await this.generateEmbedding(processedText);

    // Cache result
    this.cache.put(cacheKey, embedding);
    this.totalEmbeddings++;

    return embedding;
  }

  /**
   * Generate embeddings for multiple texts with batch processing
   */
  async embedBatch(
    texts: string[],
    onProgress?: ProgressCallback
  ): Promise<Float32Array[]> {
    if (texts.length === 0) return [];

    // Enforce maximum batch size
    if (texts.length > MAX_BATCH_SIZE) {
      throw new Error(`Batch size ${texts.length} exceeds maximum of ${MAX_BATCH_SIZE}`);
    }

    // Pre-process all texts
    const processedTexts = texts.map((t) => this.preprocessText(t));

    // Check cache and separate cached/uncached
    const results: Float32Array[] = new Array(texts.length);
    const uncachedIndices: number[] = [];
    const uncachedTexts: string[] = [];

    for (let i = 0; i < processedTexts.length; i++) {
      const cacheKey = this.getCacheKey(processedTexts[i]);
      const cached = this.cache.get(cacheKey);
      if (cached) {
        results[i] = cached;
      } else {
        uncachedIndices.push(i);
        uncachedTexts.push(processedTexts[i]);
      }
    }

    // If all cached, return immediately
    if (uncachedTexts.length === 0) {
      if (onProgress) {
        onProgress({ current: texts.length, total: texts.length, percentage: 100 });
      }
      return results;
    }

    // Deduplicate uncached texts if enabled
    let textsToEmbed = uncachedTexts;
    let indexMapping: Map<string, number[]> | null = null;

    if (this.config.preprocessing.deduplicate) {
      const { unique, mapping } = this.deduplicateTexts(uncachedTexts);
      textsToEmbed = unique;
      indexMapping = mapping;
    }

    // Split into batches
    const batchSize = this.config.batch.batchSize;
    const batches: string[][] = [];
    for (let i = 0; i < textsToEmbed.length; i += batchSize) {
      batches.push(textsToEmbed.slice(i, i + batchSize));
    }

    // Process batches with semaphore-controlled concurrency
    let processed = texts.length - uncachedTexts.length; // Start with cached count
    const batchResults: Float32Array[][] = [];

    await Promise.all(
      batches.map((batch, batchIndex) =>
        this.semaphore.withLock(async () => {
          const embeddings = await this.processBatch(batch);
          batchResults[batchIndex] = embeddings;

          processed += batch.length;
          this.totalBatches++;
          this.totalBatchItems += batch.length;

          if (onProgress) {
            onProgress({
              current: processed,
              total: texts.length,
              percentage: Math.round((processed / texts.length) * 100),
            });
          }
        })
      )
    );

    // Flatten batch results
    const allNewEmbeddings = batchResults.flat();

    // Map embeddings back to original indices
    if (indexMapping) {
      // With deduplication
      let embeddingIndex = 0;
      for (const uniqueText of textsToEmbed) {
        const embedding = allNewEmbeddings[embeddingIndex++];
        const originalIndices = indexMapping.get(uniqueText) || [];

        // Cache the embedding
        const cacheKey = this.getCacheKey(uniqueText);
        this.cache.put(cacheKey, embedding);

        // Assign to all original positions
        for (const origIdx of originalIndices) {
          results[uncachedIndices[origIdx]] = embedding;
        }
      }
    } else {
      // Without deduplication
      for (let i = 0; i < allNewEmbeddings.length; i++) {
        const embedding = allNewEmbeddings[i];
        const originalIndex = uncachedIndices[i];

        // Cache the embedding
        const cacheKey = this.getCacheKey(uncachedTexts[i]);
        this.cache.put(cacheKey, embedding);

        results[originalIndex] = embedding;
      }
    }

    this.totalEmbeddings += uncachedTexts.length;

    return results;
  }

  // ==========================================================================
  // Search Methods (RuVector Integration)
  // ==========================================================================

  /**
   * Search for similar texts using RuVector backend
   */
  async search(query: string, k: number = 10): Promise<SearchResult[]> {
    if (!this.config.vectorBackend) {
      throw new Error('Vector backend not configured. Pass vectorBackend in config.');
    }

    // Generate query embedding
    const queryEmbedding = await this.embed(query);

    // Search using RuVector
    const results = this.config.vectorBackend.search(queryEmbedding, k);

    // Convert to SearchResult format
    return results.map((r: VectorSearchResult) => ({
      id: r.id,
      text: r.metadata?.text as string | undefined,
      similarity: r.similarity,
      distance: r.distance,
      metadata: r.metadata,
    }));
  }

  /**
   * Store text with embedding in RuVector
   */
  async store(
    id: string,
    text: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.config.vectorBackend) {
      throw new Error('Vector backend not configured. Pass vectorBackend in config.');
    }

    const embedding = await this.embed(text);

    this.config.vectorBackend.insert(id, embedding, {
      ...metadata,
      text, // Store original text in metadata
    });
  }

  /**
   * Store multiple texts with embeddings in RuVector (batch operation)
   */
  async storeBatch(
    items: Array<{ id: string; text: string; metadata?: Record<string, unknown> }>,
    onProgress?: ProgressCallback
  ): Promise<void> {
    if (!this.config.vectorBackend) {
      throw new Error('Vector backend not configured. Pass vectorBackend in config.');
    }

    // Extract texts for batch embedding
    const texts = items.map((item) => item.text);
    const embeddings = await this.embedBatch(texts, onProgress);

    // Insert into vector backend
    const insertItems = items.map((item, i) => ({
      id: item.id,
      embedding: embeddings[i],
      metadata: {
        ...item.metadata,
        text: item.text,
      },
    }));

    this.config.vectorBackend.insertBatch(insertItems);
  }

  /**
   * Hybrid search combining embedding similarity with metadata filters
   */
  async hybridSearch(
    query: string,
    k: number = 10,
    filter?: Record<string, unknown>
  ): Promise<SearchResult[]> {
    if (!this.config.vectorBackend) {
      throw new Error('Vector backend not configured. Pass vectorBackend in config.');
    }

    const queryEmbedding = await this.embed(query);

    const results = this.config.vectorBackend.search(queryEmbedding, k, {
      filter: filter as Record<string, any>,
    });

    return results.map((r: VectorSearchResult) => ({
      id: r.id,
      text: r.metadata?.text as string | undefined,
      similarity: r.similarity,
      distance: r.distance,
      metadata: r.metadata,
    }));
  }

  // ==========================================================================
  // Model Management
  // ==========================================================================

  /**
   * Lazy load the embedding model
   */
  async loadModel(): Promise<void> {
    if (this.modelLoaded && this.pipeline) return;

    if (this.config.provider === 'transformers') {
      await this.loadTransformersModel();
    }
    // OpenAI and Cohere don't need preloading

    this.modelLoaded = true;
  }

  /**
   * Switch to a different model at runtime
   */
  async switchModel(model: SupportedModel, provider?: EmbeddingProvider): Promise<void> {
    // Update config
    this.config.model = model;
    if (provider) {
      this.config.provider = provider;
    }
    this.config.dimension = getModelDimension(model);

    // Reset model state
    this.pipeline = null;
    this.modelLoaded = false;

    // Clear cache (embeddings are model-specific)
    this.cache.clear();

    // Load new model if transformers
    if (this.config.provider === 'transformers') {
      await this.loadTransformersModel();
    }

    this.modelLoaded = true;
  }

  // ==========================================================================
  // Statistics
  // ==========================================================================

  /**
   * Get service statistics
   */
  getStats(): EmbeddingStats {
    const cacheStats = this.cache.stats;

    return {
      totalEmbeddings: this.totalEmbeddings,
      cache: cacheStats,
      batch: {
        totalBatches: this.totalBatches,
        averageBatchSize:
          this.totalBatches > 0
            ? Math.round(this.totalBatchItems / this.totalBatches)
            : 0,
      },
      model: {
        name: this.config.model,
        dimension: this.config.dimension,
        provider: this.config.provider,
        loaded: this.modelLoaded,
      },
      vectorBackend: this.config.vectorBackend
        ? {
            count: this.config.vectorBackend.getStats().count,
            backend: this.config.vectorBackend.getStats().backend,
          }
        : undefined,
    };
  }

  /**
   * Clear the embedding cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get current model dimension
   */
  getDimension(): number {
    return this.config.dimension;
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  /**
   * Pre-process text according to config
   */
  private preprocessText(text: string): string {
    if (!text) return '';

    // Early length check to prevent DoS
    if (text.length > MAX_TEXT_LENGTH) {
      text = text.slice(0, MAX_TEXT_LENGTH);
    }

    let processed = text;

    if (this.config.preprocessing.normalize) {
      // Normalize whitespace
      processed = processed.replace(/\s+/g, ' ').trim();

      // Normalize unicode
      processed = processed.normalize('NFKC');

      // Remove control characters
      processed = processed.replace(/[\x00-\x1F\x7F]/g, '');
    }

    // Truncate if too long
    if (processed.length > this.config.preprocessing.maxLength) {
      processed = processed.slice(0, this.config.preprocessing.maxLength);
    }

    return processed;
  }

  /**
   * Chunk long text into smaller pieces with overlap
   */
  chunkText(text: string): string[] {
    const maxLength = this.config.preprocessing.maxLength;
    const overlap = this.config.preprocessing.chunkOverlap;

    if (text.length <= maxLength) {
      return [text];
    }

    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
      const end = Math.min(start + maxLength, text.length);
      chunks.push(text.slice(start, end));

      if (end >= text.length) break;

      start = end - overlap;
    }

    return chunks;
  }

  /**
   * Generate cache key for text
   */
  private getCacheKey(text: string): string {
    return `${this.config.provider}:${this.config.model}:${text}`;
  }

  /**
   * Deduplicate texts and create mapping
   */
  private deduplicateTexts(texts: string[]): {
    unique: string[];
    mapping: Map<string, number[]>;
  } {
    const mapping = new Map<string, number[]>();
    const unique: string[] = [];

    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const existing = mapping.get(text);
      if (existing) {
        existing.push(i);
      } else {
        mapping.set(text, [i]);
        unique.push(text);
      }
    }

    return { unique, mapping };
  }

  /**
   * Generate embedding using configured provider
   */
  private async generateEmbedding(text: string): Promise<Float32Array> {
    switch (this.config.provider) {
      case 'transformers':
        return this.embedWithTransformers(text);
      case 'openai':
        return this.embedWithOpenAI(text);
      case 'cohere':
        return this.embedWithCohere(text);
      case 'custom':
        return this.embedWithCustom(text);
      default:
        throw new Error(`Unknown provider: ${this.config.provider}`);
    }
  }

  /**
   * Process a batch of texts
   */
  private async processBatch(texts: string[]): Promise<Float32Array[]> {
    switch (this.config.provider) {
      case 'transformers':
        return this.batchEmbedWithTransformers(texts);
      case 'openai':
        return this.batchEmbedWithOpenAI(texts);
      case 'cohere':
        return this.batchEmbedWithCohere(texts);
      case 'custom':
        return Promise.all(texts.map((t) => this.embedWithCustom(t)));
      default:
        throw new Error(`Unknown provider: ${this.config.provider}`);
    }
  }

  /**
   * Load transformers.js model
   */
  private async loadTransformersModel(): Promise<void> {
    try {
      const transformers = await import('@huggingface/transformers');

      // Set HF token if available
      const hfToken = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN;
      if (hfToken && transformers.env) {
        (transformers.env as any).HF_TOKEN = hfToken;
      }

      // Determine model path
      const modelPath = this.config.model.includes('/')
        ? this.config.model
        : `Xenova/${this.config.model}`;

      this.pipeline = await transformers.pipeline(
        'feature-extraction',
        modelPath
      );
    } catch (error) {
      console.warn(
        `Failed to load transformers model: ${(error as Error).message}`
      );
      console.warn('Falling back to mock embeddings');
      this.pipeline = null;
    }
  }

  /**
   * Embed with @huggingface/transformers (single)
   */
  private async embedWithTransformers(text: string): Promise<Float32Array> {
    if (!this.pipeline) {
      await this.loadTransformersModel();
    }

    if (this.pipeline) {
      const output = await this.pipeline(text, {
        pooling: 'mean',
        normalize: true,
      });
      return new Float32Array(output.data);
    }

    // Fallback to mock
    return this.mockEmbedding(text);
  }

  /**
   * Batch embed with transformers
   */
  private async batchEmbedWithTransformers(
    texts: string[]
  ): Promise<Float32Array[]> {
    if (!this.pipeline) {
      await this.loadTransformersModel();
    }

    if (this.pipeline) {
      // Process one at a time for transformers.js (batch not well supported)
      return Promise.all(
        texts.map(async (text) => {
          const output = await this.pipeline(text, {
            pooling: 'mean',
            normalize: true,
          });
          return new Float32Array(output.data);
        })
      );
    }

    return texts.map((t) => this.mockEmbedding(t));
  }

  /**
   * Embed with OpenAI API
   */
  private async embedWithOpenAI(text: string): Promise<Float32Array> {
    if (!this.config.apiKey) {
      throw new Error('OpenAI API key required');
    }

    const apiUrl = 'https://api.openai.com/v1/embeddings';
    validateApiUrl(apiUrl);

    try {
      const response = await fetchWithTimeout(apiUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model || 'text-embedding-ada-002',
          input: text,
        }),
      });

      if (!response.ok) {
        // Don't expose API key in error message
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const data = (await response.json()) as any;

      // Validate response structure
      if (!data?.data?.[0]?.embedding || !Array.isArray(data.data[0].embedding)) {
        throw new Error('Invalid response format from OpenAI API');
      }

      return new Float32Array(data.data[0].embedding);
    } catch (error) {
      if (error instanceof Error) {
        // Sanitize error message to not expose API key
        if (error.name === 'AbortError') {
          throw new Error('OpenAI API request timed out');
        }
        // Re-throw without exposing internal details
        throw new Error(`OpenAI embedding failed: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * Batch embed with OpenAI API
   */
  private async batchEmbedWithOpenAI(texts: string[]): Promise<Float32Array[]> {
    if (!this.config.apiKey) {
      throw new Error('OpenAI API key required');
    }

    const apiUrl = 'https://api.openai.com/v1/embeddings';
    validateApiUrl(apiUrl);

    try {
      const response = await fetchWithTimeout(apiUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model || 'text-embedding-ada-002',
          input: texts,
        }),
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const data = (await response.json()) as any;

      // Validate response structure
      if (!data?.data || !Array.isArray(data.data)) {
        throw new Error('Invalid response format from OpenAI API');
      }

      // Sort by index to maintain order
      const sorted = data.data.sort(
        (a: any, b: any) => a.index - b.index
      );

      return sorted.map((item: any) => {
        if (!item?.embedding || !Array.isArray(item.embedding)) {
          throw new Error('Invalid embedding format in OpenAI response');
        }
        return new Float32Array(item.embedding);
      });
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('OpenAI API request timed out');
        }
        throw new Error(`OpenAI batch embedding failed: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * Embed with Cohere API
   */
  private async embedWithCohere(text: string): Promise<Float32Array> {
    if (!this.config.apiKey) {
      throw new Error('Cohere API key required');
    }

    const apiUrl = 'https://api.cohere.ai/v1/embed';
    validateApiUrl(apiUrl);

    try {
      const response = await fetchWithTimeout(apiUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model || 'embed-english-v3.0',
          texts: [text],
          input_type: 'search_document',
        }),
      });

      if (!response.ok) {
        throw new Error(`Cohere API error: ${response.status} ${response.statusText}`);
      }

      const data = (await response.json()) as any;

      // Validate response structure
      if (!data?.embeddings?.[0] || !Array.isArray(data.embeddings[0])) {
        throw new Error('Invalid response format from Cohere API');
      }

      return new Float32Array(data.embeddings[0]);
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Cohere API request timed out');
        }
        throw new Error(`Cohere embedding failed: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * Batch embed with Cohere API
   */
  private async batchEmbedWithCohere(texts: string[]): Promise<Float32Array[]> {
    if (!this.config.apiKey) {
      throw new Error('Cohere API key required');
    }

    const apiUrl = 'https://api.cohere.ai/v1/embed';
    validateApiUrl(apiUrl);

    try {
      const response = await fetchWithTimeout(apiUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.config.model || 'embed-english-v3.0',
          texts,
          input_type: 'search_document',
        }),
      });

      if (!response.ok) {
        throw new Error(`Cohere API error: ${response.status} ${response.statusText}`);
      }

      const data = (await response.json()) as any;

      // Validate response structure
      if (!data?.embeddings || !Array.isArray(data.embeddings)) {
        throw new Error('Invalid response format from Cohere API');
      }

      return data.embeddings.map((emb: unknown) => {
        if (!Array.isArray(emb)) {
          throw new Error('Invalid embedding format in Cohere response');
        }
        return new Float32Array(emb as number[]);
      });
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Cohere API request timed out');
        }
        throw new Error(`Cohere batch embedding failed: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * Embed with custom function
   */
  private async embedWithCustom(text: string): Promise<Float32Array> {
    if (!this.config.customEmbedder) {
      throw new Error('Custom embedder function required');
    }

    return this.config.customEmbedder(text);
  }

  /**
   * Generate mock embedding for testing/fallback
   */
  private mockEmbedding(text: string): Float32Array {
    const embedding = new Float32Array(this.config.dimension);

    if (!text || text.length === 0) {
      return embedding;
    }

    // Deterministic hash-based generation
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = (hash << 5) - hash + text.charCodeAt(i);
      hash = hash & hash;
    }

    // Fill with pseudo-random values
    for (let i = 0; i < this.config.dimension; i++) {
      const seed = hash + i * 31;
      embedding[i] = Math.sin(seed) * Math.cos(seed * 0.5);
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < embedding.length; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm);

    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create an enhanced embedding service with default configuration
 */
export function createEmbeddingService(
  config?: EnhancedEmbeddingConfig
): EnhancedEmbeddingService {
  return new EnhancedEmbeddingService(config);
}

// ============================================================================
// Default Export
// ============================================================================

export default EnhancedEmbeddingService;
