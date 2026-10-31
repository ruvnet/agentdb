/**
 * VectorBackend - Unified interface for vector database backends
 *
 * Provides abstraction over different vector search implementations
 * (RuVector, hnswlib-node) for AgentDB v2.
 *
 * Design:
 * - String-based IDs for all operations (backends handle label mapping internally)
 * - Consistent SearchResult format across backends
 * - Save/load with metadata persistence
 * - Backend-specific optimizations hidden behind interface
 */

export interface VectorConfig {
  /** Vector dimension (e.g., 384, 768, 1536) */
  dimension?: number;
  /** Alias for dimension (backward compatibility) */
  dimensions?: number;

  /** Distance metric: 'cosine', 'l2' (Euclidean), 'ip' (inner product) */
  metric: 'cosine' | 'l2' | 'ip';

  /** Maximum number of elements (default: 100000) */
  maxElements?: number;

  /** HNSW M parameter - connections per layer (default: 16) */
  M?: number;

  /** HNSW efConstruction - build quality (default: 200) */
  efConstruction?: number;

  /** HNSW efSearch - search quality (default: 100) */
  efSearch?: number;
}

export interface SearchResult {
  /** String ID of the vector */
  id: string;

  /** Raw distance value from backend */
  distance: number;

  /** Normalized similarity (0-1, higher is more similar) */
  similarity: number;

  /** Optional metadata attached to vector */
  metadata?: Record<string, unknown>;
}

export interface SearchOptions {
  /** Minimum similarity threshold (0-1) */
  threshold?: number;

  /** Override efSearch for this query */
  efSearch?: number;

  /** Metadata filters (post-filtering) */
  filter?: Record<string, unknown>;

  /**
   * Caller-owned opaque correlation key for later feedback (issue #14).
   *
   * Backends that support async feedback (e.g. `SelfLearningRvfBackend`)
   * use this — instead of an internal id the caller has no way to obtain —
   * to correlate a later `recordFeedback(feedbackId, quality)` call back to
   * this search. Must be unique among currently-outstanding searches; reusing
   * an id that hasn't been resolved via `recordFeedback` yet throws.
   */
  feedbackId?: string;

  /**
   * searchVectors() only: collections with fewer rows than this are always
   * answered by exact cosine over the stored embeddings (ADR-015). Default
   * AGENTDB_EXACT_SEARCH_THRESHOLD or 1000; 0 disables the small-collection
   * path (the short-result path still applies).
   */
  exactSearchThreshold?: number;
}

/**
 * The caller's own copy of a collection, used by searchVectors() to answer
 * exactly when the approximate index cannot be trusted (ADR-015).
 */
export interface ExactSearchSource {
  /** Rows in the collection (not the backend, which may be shared) */
  count(): number;
  /** Exact top-k by cosine over the stored embeddings, best first */
  scan(query: Float32Array, k: number, threshold: number): SearchResult[];
  /** Index hits that belong to this collection; default: all */
  owns?(result: SearchResult): boolean;
  /** Identity used to dedupe index hits against scan results; default: id */
  key?(result: SearchResult): string;
}

export interface VectorStats {
  /** Number of vectors in index */
  count: number;

  /** Vector dimension */
  dimension: number;

  /** Distance metric */
  metric: string;

  /** Backend name */
  backend: 'ruvector' | 'hnswlib' | 'rvf';

  /** Memory usage in bytes (0 if not available) */
  memoryUsage: number;
}

/**
 * VectorBackend - Interface for vector search implementations
 *
 * All backends must:
 * 1. Accept string IDs and handle label mapping internally
 * 2. Normalize distances to similarities (0-1 range)
 * 3. Support save/load with metadata persistence
 * 4. Provide stats for monitoring
 */
export interface VectorBackend {
  /** Backend name for detection and logging */
  readonly name: 'ruvector' | 'hnswlib' | 'rvf';

  /**
   * Insert a single vector with optional metadata
   * @param id - Unique string identifier
   * @param embedding - Vector as Float32Array
   * @param metadata - Optional metadata to store with vector
   */
  insert(id: string, embedding: Float32Array, metadata?: Record<string, unknown>): void;

  /**
   * Insert multiple vectors in batch (more efficient)
   * @param items - Array of {id, embedding, metadata?}
   */
  insertBatch(items: Array<{
    id: string;
    embedding: Float32Array;
    metadata?: Record<string, unknown>;
  }>): void;

  /**
   * Search for k-nearest neighbors
   * @param query - Query vector
   * @param k - Number of results
   * @param options - Search options (threshold, efSearch, filters)
   * @returns Sorted results (most similar first)
   */
  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[];

  /**
   * Remove a vector by ID
   * @param id - Vector ID to remove
   * @returns true if removed, false if not found
   */
  remove(id: string): boolean;

  /**
   * Get backend statistics
   * @returns Stats object with count, dimension, backend name, etc.
   */
  getStats(): VectorStats;

  /**
   * Save index to disk
   * @param path - File path (backend adds extensions as needed)
   */
  save(path: string): Promise<void>;

  /**
   * Load index from disk
   * @param path - File path (backend looks for associated files)
   */
  load(path: string): Promise<void>;

  /**
   * Close and cleanup resources
   */
  close(): void;
}

/**
 * VectorBackendAsync - Extended interface for async-native backends (e.g., RVF)
 *
 * Backends like @ruvector/rvf are fully async. This interface provides
 * native async methods while maintaining backward compatibility with
 * the sync VectorBackend interface.
 */
export interface VectorBackendAsync extends VectorBackend {
  /** Async single-vector insert */
  insertAsync(id: string, embedding: Float32Array, metadata?: Record<string, unknown>): Promise<void>;

  /** Async batch insert (preferred for RVF) */
  insertBatchAsync(items: Array<{
    id: string;
    embedding: Float32Array;
    metadata?: Record<string, unknown>;
  }>): Promise<void>;

  /** Async k-NN search with native filter support */
  searchAsync(query: Float32Array, k: number, options?: SearchOptions): Promise<SearchResult[]>;

  /** Async remove by ID */
  removeAsync(id: string): Promise<boolean>;

  /** Async stats with live store status */
  getStatsAsync(): Promise<VectorStats>;

  /** Flush any pending writes (for backends that batch sync calls) */
  flush(): Promise<void>;
}

/** True when the backend exposes the async API (RuVector, RVF backends). */
export function isAsyncVectorBackend(backend: VectorBackend): backend is VectorBackendAsync {
  return typeof (backend as Partial<VectorBackendAsync>).searchAsync === 'function';
}

/** Collections smaller than this are searched exactly (ADR-015) */
export const DEFAULT_EXACT_SEARCH_THRESHOLD = 1000;

function exactSearchThreshold(options?: SearchOptions): number {
  const configured = options?.exactSearchThreshold
    ?? (typeof process !== 'undefined' ? Number(process.env?.AGENTDB_EXACT_SEARCH_THRESHOLD) : NaN);
  return Number.isFinite(configured) && configured >= 0 ? configured : DEFAULT_EXACT_SEARCH_THRESHOLD;
}

/**
 * k-NN search that works with both sync and async-native backends.
 * Async-native backends cannot answer the sync search() — use this instead.
 *
 * With an `exact` source, the answer falls back to exact cosine over the
 * caller's stored embeddings when the index cannot be trusted:
 * - the collection has fewer than `exactSearchThreshold` rows (HNSW on tiny
 *   graphs misses neighbours, ruvnet/RuVector#1298), or
 * - the index returned fewer of the collection's vectors than
 *   min(k, collection size).
 * Exact hits come first, in exact order; index hits the scan did not see
 * (stale rows) are appended after deduplication.
 */
export async function searchVectors(
  backend: VectorBackend,
  query: Float32Array,
  k: number,
  options?: SearchOptions,
  exact?: ExactSearchSource
): Promise<SearchResult[]> {
  let hits: SearchResult[];
  try {
    hits = isAsyncVectorBackend(backend)
      ? await backend.searchAsync(query, k, options)
      : backend.search(query, k, options);
  } catch (error) {
    if (!exact) throw error;
    hits = [];
  }
  if (!exact) return hits;

  let owned: SearchResult[];
  let rows: number;
  let exactHits: SearchResult[];
  try {
    owned = exact.owns ? hits.filter((hit) => exact.owns!(hit)) : hits;
    rows = exact.count();
    if (rows >= exactSearchThreshold(options) && owned.length >= Math.min(k, rows)) return hits;
    // No threshold means no cut-off, as for the index (negative cosines included)
    exactHits = exact.scan(query, k, options?.threshold ?? -Infinity);
  } catch {
    // The caller's table is unavailable: the index answer is all there is
    return hits;
  }

  const keyOf = exact.key ?? ((hit: SearchResult) => hit.id);
  const seen = new Set(exactHits.map(keyOf));
  const merged = [...exactHits];
  for (const hit of owned) {
    if (merged.length >= k) break;
    const key = keyOf(hit);
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(hit);
    }
  }
  return merged.slice(0, Math.max(0, k));
}
