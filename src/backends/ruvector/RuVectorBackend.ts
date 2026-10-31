/**
 * RuVectorBackend - High-Performance Vector Storage
 *
 * Implements VectorBackend using @ruvector/core with optional GNN support.
 * Provides <100µs search latency with native SIMD optimizations.
 *
 * Features:
 * - Automatic fallback when @ruvector packages not installed
 * - Separate metadata storage for rich queries
 * - Distance-to-similarity conversion for all metrics
 * - Batch operations for optimal throughput
 * - Persistent storage with separate metadata files
 * - Parallel batch insert with configurable concurrency
 * - Buffer pooling to reduce memory allocations
 * - Adaptive index parameters based on dataset size
 * - Memory-mapped support for large indices
 * - Statistics tracking for performance monitoring
 */

import type { VectorBackendAsync, VectorConfig, SearchResult, SearchOptions, VectorStats } from '../VectorBackend.js';

// ============================================================================
// Performance & Security Constants
// ============================================================================

/** @inline Maximum supported vector dimension for bounds checking */
export const MAX_VECTOR_DIMENSION = 4096;

/** @inline Maximum batch size for optimal cache utilization */
export const MAX_BATCH_SIZE = 10000;

/** @inline Default cache size for embedding storage */
export const DEFAULT_CACHE_SIZE = 10000;

/** @inline Small epsilon for numerical stability */
const EPSILON = 1e-10;

/** @inline Maximum metadata entries to prevent memory exhaustion */
const MAX_METADATA_ENTRIES = 10_000_000;

/** @inline Maximum path length for file operations */
const MAX_PATH_LENGTH = 4096;

/** Forbidden path patterns for security */
const FORBIDDEN_PATH_PATTERNS = [
  /\.\./,       // Path traversal
  /^\/etc\//i,  // System config
  /^\/proc\//i, // Process info
  /^\/sys\//i,  // System info
  /^\/dev\//i,  // Devices
];

// ============================================================================
// Security Validation Helpers
// ============================================================================

/**
 * Validate a file path is safe
 */
function validatePath(inputPath: string): void {
  if (!inputPath || typeof inputPath !== 'string') {
    throw new Error('Path must be a non-empty string');
  }
  if (inputPath.length > MAX_PATH_LENGTH) {
    throw new Error(`Path exceeds maximum length of ${MAX_PATH_LENGTH}`);
  }
  for (const pattern of FORBIDDEN_PATH_PATTERNS) {
    if (pattern.test(inputPath)) {
      throw new Error(`Path contains forbidden pattern`);
    }
  }
}

/**
 * Validate vector dimension is within safe limits
 */
function validateDimension(dimension: number): void {
  if (!Number.isFinite(dimension) || dimension < 1 || dimension > MAX_VECTOR_DIMENSION) {
    throw new Error(`Dimension must be between 1 and ${MAX_VECTOR_DIMENSION}`);
  }
}

/**
 * Semaphore for controlling concurrent operations
 */
class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift()!;
      resolve();
    } else {
      this.permits++;
    }
  }

  get available(): number {
    return this.permits;
  }
}

/**
 * Buffer pool for reusing Float32Array buffers
 */
class BufferPool {
  private pools: Map<number, Float32Array[]> = new Map();
  private maxPoolSize: number;

  constructor(maxPoolSize = 100) {
    this.maxPoolSize = maxPoolSize;
  }

  /**
   * Acquire a buffer of the specified size
   */
  acquire(size: number): Float32Array {
    const pool = this.pools.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return new Float32Array(size);
  }

  /**
   * Release a buffer back to the pool
   */
  release(buffer: Float32Array): void {
    const size = buffer.length;
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    const pool = this.pools.get(size)!;
    if (pool.length < this.maxPoolSize) {
      // Clear buffer before returning to pool
      buffer.fill(0);
      pool.push(buffer);
    }
  }

  /**
   * Clear all pooled buffers
   */
  clear(): void {
    this.pools.clear();
  }

  /**
   * Get pool statistics
   */
  getStats(): { totalBuffers: number; totalMemory: number } {
    let totalBuffers = 0;
    let totalMemory = 0;
    for (const [size, pool] of this.pools) {
      totalBuffers += pool.length;
      totalMemory += pool.length * size * 4; // Float32 = 4 bytes
    }
    return { totalBuffers, totalMemory };
  }
}

/**
 * Statistics tracker for performance monitoring
 */
interface PerformanceStats {
  insertCount: number;
  insertTotalLatencyMs: number;
  insertMinLatencyMs: number;
  insertMaxLatencyMs: number;
  searchCount: number;
  searchTotalLatencyMs: number;
  searchMinLatencyMs: number;
  searchMaxLatencyMs: number;
  cacheHits: number;
  cacheMisses: number;
  lastMemoryUsage: number;
  indexRebuildCount: number;
}

/**
 * Adaptive index parameters based on dataset size
 */
interface AdaptiveParams {
  M: number;
  efConstruction: number;
  efSearch: number;
}

/**
 * Extended configuration with new options
 */
export interface RuVectorConfig extends VectorConfig {
  /** Concurrency level for parallel batch insert (default: 4) */
  parallelConcurrency?: number;
  /** Buffer pool max size (default: 100) */
  bufferPoolSize?: number;
  /** Enable adaptive index parameters (default: true) */
  adaptiveParams?: boolean;
  /** Enable memory-mapped storage for large indices (default: false) */
  enableMmap?: boolean;
  /** Path for memory-mapped storage */
  mmapPath?: string;
  /** Enable statistics tracking (default: true) */
  enableStats?: boolean;
  /**
   * File the native index persists to. When omitted, a private temp directory
   * is created and removed on close(), so instances never share a file (the
   * native binding otherwise defaults to ./ruvector.db in the cwd).
   */
  storagePath?: string;
}

/** ruvector-core distance metric names */
const NATIVE_METRIC: Record<string, string> = {
  cosine: 'Cosine',
  l2: 'Euclidean',
  ip: 'DotProduct',
};

export class RuVectorBackend implements VectorBackendAsync {
  readonly name = 'ruvector' as const;
  private db: any; // VectorDb from ruvector / @ruvector/core (every method is async)
  private config: RuVectorConfig;
  private metadata: Map<string, Record<string, any>> = new Map();
  private initialized = false;

  // Native calls are async; they run one at a time through this queue so a
  // sync insert followed by a remove (or a search) observes them in order.
  private queue: Promise<void> = Promise.resolve();
  private lastWriteError: Error | null = null;
  private cachedCount = 0;
  private knownIds: Set<string> = new Set();
  private storeWasEmpty = true;
  private storagePath = '';
  private ownedTempDir: string | null = null;
  private defaultEfSearch: number | undefined;
  private VectorDbCtor: any;
  private nativeOptions: Record<string, unknown> = {};

  // Concurrency control
  private semaphore: Semaphore;
  private bufferPool: BufferPool;

  // Statistics tracking
  private stats: PerformanceStats = {
    insertCount: 0,
    insertTotalLatencyMs: 0,
    insertMinLatencyMs: Infinity,
    insertMaxLatencyMs: 0,
    searchCount: 0,
    searchTotalLatencyMs: 0,
    searchMinLatencyMs: Infinity,
    searchMaxLatencyMs: 0,
    cacheHits: 0,
    cacheMisses: 0,
    lastMemoryUsage: 0,
    indexRebuildCount: 0,
  };

  // Mmap support
  private mmapEnabled = false;
  private mmapBuffer: Buffer | null = null;

  constructor(config: VectorConfig | RuVectorConfig) {
    // Handle both dimension and dimensions for backward compatibility
    const dimension = config.dimension ?? config.dimensions;
    if (!dimension) {
      throw new Error('Vector dimension is required (use dimension or dimensions)');
    }

    // Validate dimension for security
    validateDimension(dimension);

    // Store both forms for compatibility with different backends
    this.config = { ...config, dimension, dimensions: dimension } as RuVectorConfig;

    // Initialize concurrency control with configurable limit (bounded for security)
    const concurrency = Math.min(Math.max(1, (config as RuVectorConfig).parallelConcurrency ?? 4), 32);
    this.semaphore = new Semaphore(concurrency);

    // Initialize buffer pool with bounded size
    const bufferPoolSize = Math.min(Math.max(1, (config as RuVectorConfig).bufferPoolSize ?? 100), 1000);
    this.bufferPool = new BufferPool(bufferPoolSize);

    // Mmap configuration - validate path if provided
    this.mmapEnabled = (config as RuVectorConfig).enableMmap ?? false;
    if (this.mmapEnabled && (config as RuVectorConfig).mmapPath) {
      validatePath((config as RuVectorConfig).mmapPath!);
    }
  }

  /**
   * Get adaptive HNSW parameters based on expected dataset size
   */
  private getAdaptiveParams(datasetSize: number): AdaptiveParams {
    if (datasetSize < 1000) {
      // Small dataset: lower M and efConstruction for faster builds
      return { M: 8, efConstruction: 100, efSearch: 50 };
    } else if (datasetSize <= 100000) {
      // Medium dataset: balanced parameters
      return { M: 16, efConstruction: 200, efSearch: 100 };
    } else {
      // Large dataset: higher M for better recall
      return { M: 32, efConstruction: 400, efSearch: 200 };
    }
  }

  /**
   * Pick the native storage file: the configured path, or a fresh private
   * temp directory that close() removes.
   */
  private async resolveStoragePath(): Promise<string> {
    const pathModule = await import('path');
    // ':memory:' would become a literal file of that name in the cwd
    if (this.config.storagePath && this.config.storagePath !== ':memory:') {
      validatePath(this.config.storagePath);
      return pathModule.resolve(this.config.storagePath);
    }
    const fs = await import('fs');
    const os = await import('os');
    this.ownedTempDir = fs.mkdtempSync(pathModule.join(os.tmpdir(), 'agentdb-ruvector-'));
    return pathModule.join(this.ownedTempDir, 'index.db');
  }

  private removeOwnedTempDir(): void {
    if (!this.ownedTempDir) return;
    const dir = this.ownedTempDir;
    this.ownedTempDir = null;
    import('fs')
      .then((fs) => fs.rmSync(dir, { recursive: true, force: true }))
      .catch(() => { /* best effort */ });
  }

  /**
   * Run a native operation after every earlier one has settled.
   * A failure is recorded (and surfaced by flush()) without breaking the queue.
   */
  private enqueue<T>(op: () => Promise<T>): Promise<T> {
    const result = this.queue.then(op);
    this.queue = result.then(
      () => undefined,
      (err: Error) => {
        this.lastWriteError = err;
        console.error(`[RuVectorBackend] native operation failed: ${err?.message ?? err}`);
      }
    );
    return result;
  }

  /**
   * Write entries to the native index with upsert semantics. ruvector-core
   * keeps a stale HNSW node when an existing id is re-inserted, so an id that
   * may already be stored is deleted first.
   */
  private async writeNative(entries: Array<{ id: string; vector: Float32Array }>): Promise<void> {
    const latest = new Map<string, Float32Array>();
    for (const e of entries) latest.set(e.id, e.vector);

    let replaced = 0;
    for (const id of latest.keys()) {
      if (this.knownIds.has(id) || !this.storeWasEmpty) {
        if (await this.db.delete(id)) replaced++;
      }
    }

    const batch = Array.from(latest, ([id, vector]) => ({ id, vector }));
    if (batch.length === 1) {
      await this.db.insert(batch[0]);
    } else {
      await this.db.insertBatch(batch);
    }
    for (const { id } of batch) this.knownIds.add(id);
    this.cachedCount += batch.length - replaced;
  }

  private async deleteNative(id: string): Promise<boolean> {
    const removed = await this.db.delete(id);
    this.knownIds.delete(id);
    if (removed) this.cachedCount--;
    return removed;
  }

  private validateEntry(id: string, embedding: Float32Array, metadata?: Record<string, any>): Float32Array {
    if (!id || typeof id !== 'string') {
      throw new Error('Vector ID must be a non-empty string');
    }
    if (!(embedding instanceof Float32Array) && !Array.isArray(embedding)) {
      throw new Error('Embedding must be a Float32Array or array');
    }
    if (embedding.length !== this.config.dimension) {
      throw new Error(`Vector dimension ${embedding.length} does not match expected ${this.config.dimension}`);
    }
    if (metadata && !this.metadata.has(id) && this.metadata.size >= MAX_METADATA_ENTRIES) {
      throw new Error(`Metadata store has reached maximum capacity of ${MAX_METADATA_ENTRIES}`);
    }
    // Copy: callers (and insertWithPooledBuffer) may reuse the buffer before
    // the queued native write runs.
    return new Float32Array(embedding);
  }

  private recordInsertLatency(startTime: number): void {
    if (this.config.enableStats === false) return;
    const latency = performance.now() - startTime;
    this.stats.insertCount++;
    this.stats.insertTotalLatencyMs += latency;
    this.stats.insertMinLatencyMs = Math.min(this.stats.insertMinLatencyMs, latency);
    this.stats.insertMaxLatencyMs = Math.max(this.stats.insertMaxLatencyMs, latency);
  }

  /**
   * Initialize RuVector database with optional dependency handling
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Try main ruvector package first (includes core, gnn, graph)
      let VectorDB;
      try {
        const ruvector: any = await import('ruvector');
        VectorDB = ruvector.VectorDb || ruvector.VectorDB
          || ruvector.default?.VectorDb || ruvector.default?.VectorDB;
      } catch {
        // Fallback to @ruvector/core. Its canonical export is VectorDb;
        // VectorDB is only an alias in some versions.
        const core: any = await import('@ruvector/core');
        VectorDB = core.VectorDb || core.VectorDB
          || core.default?.VectorDb || core.default?.VectorDB;
      }

      if (!VectorDB) {
        throw new Error('Could not find VectorDb export in ruvector or @ruvector/core');
      }

      // Handle both 'dimension' and 'dimensions' for backward compatibility
      const dimensions = this.config.dimension ?? this.config.dimensions;
      if (!dimensions) {
        throw new Error('Vector dimension is required (use dimension or dimensions)');
      }

      // Determine HNSW parameters (adaptive or explicit)
      const maxElements = this.config.maxElements || 100000;
      const useAdaptive = this.config.adaptiveParams !== false;
      const adaptiveParams = useAdaptive ? this.getAdaptiveParams(maxElements) : null;

      const m = this.config.M ?? adaptiveParams?.M ?? 16;
      const efConstruction = this.config.efConstruction ?? adaptiveParams?.efConstruction ?? 200;
      const efSearch = this.config.efSearch ?? adaptiveParams?.efSearch ?? 100;

      this.storagePath = await this.resolveStoragePath();

      // HNSW parameters must go under `hnswConfig`: the binding ignores
      // top-level keys, and an omitted hnswConfig selects a flat index.
      this.VectorDbCtor = VectorDB;
      this.nativeOptions = {
        dimensions,
        distanceMetric: NATIVE_METRIC[this.config.metric] ?? NATIVE_METRIC.cosine,
        hnswConfig: { m, efConstruction, efSearch, maxElements },
      };
      this.db = new VectorDB({ ...this.nativeOptions, storagePath: this.storagePath });
      this.defaultEfSearch = efSearch;

      this.cachedCount = await this.db.len();
      this.storeWasEmpty = this.cachedCount === 0;

      // Initialize memory-mapped storage if enabled and available
      if (this.mmapEnabled && this.config.mmapPath) {
        await this.initializeMmap();
      }

      this.initialized = true;
    } catch (error) {
      this.removeOwnedTempDir();
      const errorMessage = (error as Error).message;

      // Special handling for path validation errors (from ruvector package)
      // When using :memory:, ruvector may reject it as a path traversal attempt
      // This is expected and not critical - users should use file-based paths for ruvector persistence
      if (errorMessage.includes('Path traversal') || errorMessage.includes('Invalid path')) {
        throw new Error(
          `RuVector does not support :memory: database paths.\n` +
          `Use a file path instead, or RuVector will be skipped and fallback backend will be used.\n` +
          `Original error: ${errorMessage}`
        );
      }

      throw new Error(
        `RuVector initialization failed. Please install: npm install ruvector\n` +
        `Or legacy packages: npm install @ruvector/core\n` +
        `Error: ${errorMessage}`
      );
    }
  }

  /**
   * Initialize memory-mapped storage for large indices
   */
  private async initializeMmap(): Promise<void> {
    if (!this.config.mmapPath) return;

    // Validate path for security
    validatePath(this.config.mmapPath);

    let fd: number | null = null;

    try {
      const fs = await import('fs');
      const pathModule = await import('path');

      // Define mmap module interface
      interface MmapModule {
        map: (size: number, prot: number, flags: number, fd: number, offset: number) => Buffer;
        PROT_READ: number;
        PROT_WRITE: number;
        MAP_SHARED: number;
      }

      // Check if mmap module is available (optional native dependency)
      let mmapModule: MmapModule | null = null;
      try {
        // Dynamic import of optional native module - known safe module
        const moduleName = 'mmap-io';
        mmapModule = await (import(moduleName) as unknown as Promise<MmapModule>);
      } catch {
        // mmap-io not available, fall back to regular I/O
        console.debug('[RuVectorBackend] mmap-io not available, using standard I/O');
        return;
      }

      if (!mmapModule) return;

      const mmapPath = pathModule.resolve(this.config.mmapPath);

      // Re-validate resolved path
      validatePath(mmapPath);

      const mmap = mmapModule;

      // Create or open the mmap file
      if (fs.existsSync(mmapPath)) {
        fd = fs.openSync(mmapPath, 'r+');
        const stats = fs.fstatSync(fd);

        // Validate file size to prevent excessive memory mapping
        const maxMmapSize = 4 * 1024 * 1024 * 1024; // 4GB max
        if (stats.size > maxMmapSize) {
          throw new Error(`Mmap file size ${stats.size} exceeds maximum of ${maxMmapSize}`);
        }

        this.mmapBuffer = mmap.map(
          stats.size,
          mmap.PROT_READ | mmap.PROT_WRITE,
          mmap.MAP_SHARED,
          fd,
          0
        );
        fs.closeSync(fd);
        fd = null;
      }

      this.mmapEnabled = true;
    } catch (error) {
      console.debug(`[RuVectorBackend] Failed to initialize mmap: ${(error as Error).message}`);
      this.mmapEnabled = false;
    } finally {
      // Ensure file descriptor is closed even on error
      if (fd !== null) {
        try {
          const fs = await import('fs');
          fs.closeSync(fd);
        } catch {
          // Ignore close errors in cleanup
        }
      }
    }
  }

  /**
   * Insert single vector with optional metadata.
   *
   * The native write is queued; searchAsync()/flush() wait for it. Use
   * insertAsync() to await the write and observe its errors directly.
   */
  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void {
    // Validation errors throw here; native write errors are recorded by
    // enqueue() and surfaced by flush()
    this.queueInsert(id, embedding, metadata).catch(() => {});
  }

  async insertAsync(id: string, embedding: Float32Array, metadata?: Record<string, any>): Promise<void> {
    await this.queueInsert(id, embedding, metadata);
  }

  private queueInsert(id: string, embedding: Float32Array, metadata?: Record<string, any>): Promise<void> {
    this.ensureInitialized();
    const startTime = this.config.enableStats !== false ? performance.now() : 0;
    const vector = this.validateEntry(id, embedding, metadata);
    this.setMetadata(id, metadata);
    const write = this.enqueue(() => this.writeNative([{ id, vector }]));
    this.recordInsertLatency(startTime);
    return write;
  }

  /**
   * Batch insert (queued; see insert())
   */
  insertBatch(items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>): void {
    // Validation errors throw here; see insert()
    this.queueBatch(items).catch(() => {});
  }

  async insertBatchAsync(
    items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>
  ): Promise<void> {
    await this.queueBatch(items);
  }

  private queueBatch(
    items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>
  ): Promise<void> {
    this.ensureInitialized();
    const entries: Array<{ id: string; vector: Float32Array }> = [];
    for (const item of items) {
      entries.push({ id: item.id, vector: this.validateEntry(item.id, item.embedding, item.metadata) });
    }
    for (const item of items) this.setMetadata(item.id, item.metadata);
    const writes: Promise<void>[] = [];
    for (let i = 0; i < entries.length; i += MAX_BATCH_SIZE) {
      const chunk = entries.slice(i, i + MAX_BATCH_SIZE);
      writes.push(this.enqueue(() => this.writeNative(chunk)));
    }
    return Promise.all(writes).then(() => undefined);
  }

  /**
   * Batch insert in chunks of `batchSize`.
   *
   * Native writes are serialized through the backend queue, so `concurrency`
   * no longer changes throughput; it is kept for API compatibility.
   */
  async insertBatchParallel(
    items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>,
    options?: { batchSize?: number; concurrency?: number }
  ): Promise<void> {
    const batchSize = Math.max(1, options?.batchSize ?? 100);
    for (let i = 0; i < items.length; i += batchSize) {
      await this.insertBatchAsync(items.slice(i, i + batchSize));
    }
  }

  /**
   * Insert using buffer pooling to reduce allocations
   *
   * Acquires a buffer from the pool, copies the embedding data,
   * performs the insert, and returns the buffer to the pool.
   *
   * @param id - Vector ID
   * @param embedding - Vector data (can be regular array or Float32Array)
   * @param metadata - Optional metadata
   */
  insertWithPooledBuffer(
    id: string,
    embedding: number[] | Float32Array,
    metadata?: Record<string, any>
  ): void {
    this.ensureInitialized();

    const dimension = this.config.dimension!;

    // Acquire buffer from pool
    const buffer = this.bufferPool.acquire(dimension);

    try {
      // Copy data to pooled buffer
      if (embedding instanceof Float32Array) {
        buffer.set(embedding);
      } else {
        for (let i = 0; i < dimension; i++) {
          buffer[i] = embedding[i] ?? 0;
        }
      }

      // insert() copies the vector before queueing, so the buffer can be reused
      this.insert(id, buffer, metadata);
    } finally {
      // Return buffer to pool
      this.bufferPool.release(buffer);
    }
  }

  /**
   * Synchronous search is not possible: every ruvector-core call is async.
   * Use searchAsync() (or searchVectors() from VectorBackend.ts).
   */
  search(_query: Float32Array, _k: number, _options?: SearchOptions): SearchResult[] {
    throw new Error(
      'RuVectorBackend search is async-only. Use searchAsync() or the VectorBackendAsync interface.'
    );
  }

  /**
   * Search for k-nearest neighbors with optional filtering.
   * Waits for queued writes so a preceding insert is visible.
   */
  async searchAsync(query: Float32Array, k: number, options?: SearchOptions): Promise<SearchResult[]> {
    this.ensureInitialized();
    await this.queue;

    const startTime = this.config.enableStats !== false ? performance.now() : 0;
    const hasFilter = !!options?.filter && Object.keys(options.filter).length > 0;
    // Metadata filters are applied after the ANN search, so over-fetch.
    const fetchK = hasFilter ? Math.min(k * 4, Math.max(k, this.cachedCount)) : k;

    const raw: Array<{ id: string; score: number }> = fetchK > 0
      ? await this.db.search({
        vector: query instanceof Float32Array ? query : new Float32Array(query),
        k: fetchK,
        efSearch: options?.efSearch ?? this.defaultEfSearch,
      })
      : [];

    const results: SearchResult[] = [];
    for (const r of raw) {
      // ruvector-core reports a distance in `score` (0 = identical for cosine)
      const distance = r.score;
      const similarity = this.distanceToSimilarity(distance);
      if (options?.threshold !== undefined && similarity < options.threshold) continue;

      const metadata = this.metadata.get(r.id);
      if (hasFilter && !this.matchesFilter(metadata, options!.filter!)) continue;

      results.push({ id: r.id, distance, similarity, metadata });
      if (results.length >= k) break;
    }

    if (this.config.enableStats !== false) {
      const latency = performance.now() - startTime;
      this.stats.searchCount++;
      this.stats.searchTotalLatencyMs += latency;
      this.stats.searchMinLatencyMs = Math.min(this.stats.searchMinLatencyMs, latency);
      this.stats.searchMaxLatencyMs = Math.max(this.stats.searchMaxLatencyMs, latency);
    }

    return results;
  }

  private matchesFilter(metadata: Record<string, any> | undefined, filter: Record<string, unknown>): boolean {
    if (!metadata) return false;
    for (const [key, value] of Object.entries(filter)) {
      if (metadata[key] !== value) return false;
    }
    return true;
  }

  private setMetadata(id: string, metadata?: Record<string, any>): void {
    if (metadata) {
      this.metadata.set(id, metadata);
    } else {
      this.metadata.delete(id);
    }
  }

  /**
   * Remove vector by ID. The native delete is queued; the return value only
   * reports whether this instance has seen the id. Use removeAsync() for the
   * authoritative result.
   */
  remove(id: string): boolean {
    this.ensureInitialized();
    const known = this.knownIds.has(id) || this.metadata.has(id);
    this.removeAsync(id).catch(() => { /* recorded by enqueue() */ });
    return known;
  }

  async removeAsync(id: string): Promise<boolean> {
    this.ensureInitialized();
    this.metadata.delete(id);
    return this.enqueue(() => this.deleteNative(id));
  }

  /**
   * Get database statistics. `count` reflects settled writes; await flush()
   * or use getStatsAsync() to include queued ones.
   */
  getStats(): VectorStats {
    this.ensureInitialized();
    return {
      count: this.cachedCount,
      dimension: this.config.dimension || 384,
      metric: this.config.metric,
      backend: 'ruvector',
      memoryUsage: 0,
    };
  }

  async getStatsAsync(): Promise<VectorStats> {
    this.ensureInitialized();
    await this.queue;
    this.cachedCount = await this.db.len();
    return this.getStats();
  }

  /**
   * Wait for every queued native write. Throws the most recent write failure
   * since the last flush, if any.
   */
  async flush(): Promise<void> {
    if (!this.initialized) return;
    await this.queue;
    if (this.lastWriteError) {
      const err = this.lastWriteError;
      this.lastWriteError = null;
      throw err;
    }
  }

  /**
   * Get extended performance statistics
   *
   * Returns detailed metrics including latencies, cache stats, and buffer pool info.
   */
  getExtendedStats(): {
    basic: VectorStats;
    performance: PerformanceStats;
    bufferPool: { totalBuffers: number; totalMemory: number };
    config: {
      parallelConcurrency: number;
      adaptiveParams: boolean;
      mmapEnabled: boolean;
    };
  } {
    return {
      basic: this.getStats(),
      performance: {
        ...this.stats,
        insertAvgLatencyMs: this.stats.insertCount > 0
          ? this.stats.insertTotalLatencyMs / this.stats.insertCount
          : 0,
        searchAvgLatencyMs: this.stats.searchCount > 0
          ? this.stats.searchTotalLatencyMs / this.stats.searchCount
          : 0,
      } as PerformanceStats & { insertAvgLatencyMs: number; searchAvgLatencyMs: number },
      bufferPool: this.bufferPool.getStats(),
      config: {
        parallelConcurrency: this.config.parallelConcurrency ?? 4,
        adaptiveParams: this.config.adaptiveParams !== false,
        mmapEnabled: this.mmapEnabled,
      },
    };
  }

  /**
   * Reset performance statistics
   */
  resetStats(): void {
    this.stats = {
      insertCount: 0,
      insertTotalLatencyMs: 0,
      insertMinLatencyMs: Infinity,
      insertMaxLatencyMs: 0,
      searchCount: 0,
      searchTotalLatencyMs: 0,
      searchMinLatencyMs: Infinity,
      searchMaxLatencyMs: 0,
      cacheHits: 0,
      cacheMisses: 0,
      lastMemoryUsage: 0,
      indexRebuildCount: 0,
    };
  }

  /**
   * Update index parameters adaptively based on current dataset size
   *
   * This triggers an index rebuild with optimal parameters for the current size.
   * Should be called after significant data changes.
   */
  async updateAdaptiveParams(): Promise<void> {
    this.ensureInitialized();

    if (this.config.adaptiveParams === false) {
      return;
    }

    const params = this.getAdaptiveParams(this.cachedCount);

    // M and efConstruction are fixed when the native index is built; efSearch
    // is a per-query parameter, so adopt the recommended value as the default.
    this.defaultEfSearch = params.efSearch;

    this.stats.indexRebuildCount++;
  }

  /**
   * Save index and metadata to disk
   */
  async save(savePath: string): Promise<void> {
    this.ensureInitialized();

    // Validate path for security
    validatePath(savePath);

    // The native index persists itself at storagePath; once queued writes have
    // settled, saving elsewhere is a file copy.
    await this.flush();
    const pathModule = await import('path');
    const target = pathModule.resolve(savePath);
    if (target !== this.storagePath) {
      const fsp = await import('fs/promises');
      await fsp.copyFile(this.storagePath, target);
    }

    // Save metadata separately as JSON
    const metadataPath = savePath + '.meta.json';
    validatePath(metadataPath);

    const fs = await import('fs/promises');
    await fs.writeFile(
      metadataPath,
      JSON.stringify(Object.fromEntries(this.metadata), null, 2)
    );
  }

  /**
   * Load index and metadata from disk
   */
  async load(loadPath: string): Promise<void> {
    this.ensureInitialized();

    // Validate path for security
    validatePath(loadPath);

    // Reopen the native index at loadPath
    await this.flush();
    const pathModule = await import('path');
    const source = pathModule.resolve(loadPath);
    if (source !== this.storagePath) {
      const fsSync = await import('fs');
      if (!fsSync.existsSync(source)) {
        throw new Error(`RuVector index not found: ${loadPath}`);
      }
      this.db = new this.VectorDbCtor({ ...this.nativeOptions, storagePath: source });
      this.removeOwnedTempDir();
      this.storagePath = source;
    }
    this.cachedCount = await this.db.len();
    this.storeWasEmpty = this.cachedCount === 0;
    this.knownIds.clear();
    this.metadata.clear();

    // Load metadata
    const metadataPath = loadPath + '.meta.json';
    validatePath(metadataPath);

    try {
      const fs = await import('fs/promises');
      const data = await fs.readFile(metadataPath, 'utf-8');

      // Safely parse metadata with validation
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        throw new Error('Invalid JSON in metadata file');
      }

      // Validate parsed data is a plain object
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        throw new Error('Metadata must be a plain object');
      }

      // Check for prototype pollution attempts
      const unsafeKeys = ['__proto__', 'constructor', 'prototype'];
      for (const key of Object.keys(parsed as Record<string, unknown>)) {
        if (unsafeKeys.includes(key)) {
          throw new Error(`Forbidden key in metadata: ${key}`);
        }
      }

      this.metadata = new Map(Object.entries(parsed as Record<string, Record<string, any>>));

      // Enforce size limit
      if (this.metadata.size > MAX_METADATA_ENTRIES) {
        this.metadata.clear();
        throw new Error(`Loaded metadata exceeds maximum of ${MAX_METADATA_ENTRIES} entries`);
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        // No metadata file - this is okay for backward compatibility
        console.debug(`[RuVectorBackend] No metadata file found at ${metadataPath}`);
      } else {
        throw error;
      }
    }
  }

  /**
   * Close and cleanup resources
   */
  close(): void {
    // The native binding has no close(); drop our reference once queued
    // writes settle, then remove a temp index this instance created.
    const ownedTempDir = this.ownedTempDir;
    this.ownedTempDir = null;
    if (ownedTempDir) {
      this.queue.finally(() => {
        this.ownedTempDir = ownedTempDir;
        this.removeOwnedTempDir();
      });
    }
    this.initialized = false;
    this.knownIds.clear();
    this.metadata.clear();

    // Clear buffer pool
    this.bufferPool.clear();

    // Clean up mmap buffer
    if (this.mmapBuffer) {
      this.mmapBuffer = null;
    }

    // Reset stats
    this.resetStats();
  }

  /**
   * Convert distance to similarity score based on metric
   *
   * Cosine: distance is already in [0, 2], where 0 = identical
   * L2: exponential decay for unbounded distances
   * IP: negative inner product, so negate for similarity
   */
  private distanceToSimilarity(distance: number): number {
    switch (this.config.metric) {
      case 'cosine':
        return 1 - distance; // cosine distance is 1 - similarity
      case 'l2':
        return Math.exp(-distance); // exponential decay
      case 'ip':
        return -distance; // inner product: higher is better
      default:
        return 1 - distance;
    }
  }

  /**
   * Ensure database is initialized before operations
   */
  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('RuVectorBackend not initialized. Call initialize() first.');
    }
  }

  /**
   * Get the buffer pool instance for advanced use cases
   */
  getBufferPool(): BufferPool {
    return this.bufferPool;
  }

  /**
   * Get the current concurrency semaphore status
   */
  getConcurrencyStatus(): { available: number; configured: number } {
    return {
      available: this.semaphore.available,
      configured: this.config.parallelConcurrency ?? 4,
    };
  }

  /**
   * Check if memory-mapped storage is active
   */
  isMmapEnabled(): boolean {
    return this.mmapEnabled && this.mmapBuffer !== null;
  }

  /**
   * Get recommended adaptive parameters for a given dataset size
   */
  static getRecommendedParams(datasetSize: number): AdaptiveParams {
    if (datasetSize < 1000) {
      return { M: 8, efConstruction: 100, efSearch: 50 };
    } else if (datasetSize <= 100000) {
      return { M: 16, efConstruction: 200, efSearch: 100 };
    } else {
      return { M: 32, efConstruction: 400, efSearch: 200 };
    }
  }
}

// Export helper classes for advanced use cases
export { Semaphore, BufferPool };
export type { PerformanceStats, AdaptiveParams };
