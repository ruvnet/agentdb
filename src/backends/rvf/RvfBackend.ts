/**
 * RvfBackend - RVF Format Vector Storage for AgentDB
 *
 * Implements VectorBackend and VectorBackendAsync using the @ruvector/rvf SDK.
 * Uses N-API backend (fast) on Node.js with WASM fallback for browser/edge.
 *
 * Features:
 * - Single-file .rvf persistence with crash safety
 * - Native async API with sync compatibility layer
 * - Typed metadata filter expressions (11 operators)
 * - Lineage tracking (fileId, parentId, derive)
 * - COW branching for agent experiments
 * - Progressive indexing (3-layer HNSW quality tiers)
 * - Witness chain integrity verification
 * - Auto-compaction with dead space tracking
 * - Performance statistics tracking
 *
 * Security:
 * - Path validation (reuses RuVectorBackend patterns)
 * - Bounded batch sizes and metadata limits
 * - No prototype pollution in metadata handling
 */

import type {
  VectorBackendAsync,
  VectorConfig,
  SearchResult,
  SearchOptions,
  VectorStats,
} from '../VectorBackend.js';
import { FilterBuilder, type RvfFilterExpr } from './FilterBuilder.js';
import {
  validatePath,
  validateId,
  validateMetadata,
  validateDimension,
  MAX_BATCH_SIZE,
  MAX_PENDING_WRITES,
  MAX_SEARCH_K,
  DEFAULT_BATCH_THRESHOLD,
} from './validation.js';

/** RVF-specific configuration options */
export interface RvfConfig extends VectorConfig {
  /** Path to .rvf store file */
  storagePath?: string;
  /** RVF sub-backend: 'auto' | 'node' | 'wasm' */
  rvfBackend?: 'auto' | 'node' | 'wasm';
  /** Batch threshold for auto-flushing queued sync inserts */
  batchThreshold?: number;
  /** Enable performance statistics tracking */
  enableStats?: boolean;
  /** Compression profile: 'none' (fp32), 'scalar' (int8), 'product' (PQ) */
  compression?: 'none' | 'scalar' | 'product';
  /** Hardware profile: 0=Generic, 1=Core, 2=Hot, 3=Full */
  hardwareProfile?: 0 | 1 | 2 | 3;
}

/** Re-export FilterBuilder for external use */
export { FilterBuilder, type RvfFilterExpr } from './FilterBuilder.js';

/** HNSW index statistics (AGI introspection) */
export interface IndexStats {
  indexedVectors: number;
  layers: number;
  m: number;
  efConstruction: number;
  needsRebuild: boolean;
}

/** Witness chain verification result */
export interface WitnessVerification {
  valid: boolean;
  entries: number;
  error?: string;
  /** True when the installed @ruvector/rvf cannot verify witnesses at all */
  unsupported?: boolean;
}

function unsupportedWitness(reason: string): WitnessVerification {
  return { valid: false, entries: 0, unsupported: true, error: `unsupported: ${reason}` };
}

function toWitnessVerification(raw: any): WitnessVerification {
  return {
    valid: raw?.valid === true,
    entries: typeof raw?.entries === 'number' ? raw.entries : 0,
    error: raw?.error,
  };
}

/** Performance statistics */
interface PerfStats {
  insertCount: number;
  insertTotalMs: number;
  searchCount: number;
  searchTotalMs: number;
  flushCount: number;
  compactionCount: number;
}

/**
 * RvfBackend - VectorBackend + VectorBackendAsync implementation using @ruvector/rvf
 */
export class RvfBackend implements VectorBackendAsync {
  readonly name = 'rvf' as const;

  // RVF database handle (unknown since @ruvector/rvf types not available at compile-time)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private db: any = null;
  private dim: number;
  private metricType: 'cosine' | 'l2' | 'ip';
  private config: RvfConfig;
  private initialized = false;
  private storagePath: string;

  // Sync insert queue: buffers sync insert() calls, flushed on threshold or explicit flush()
  private pending: Array<{ id: string; vector: Float32Array; metadata?: Record<string, unknown> }> = [];
  private batchThreshold: number;

  // Cached stats (updated on flush/search)
  private cachedCount = 0;

  // Performance tracking
  private stats: PerfStats = {
    insertCount: 0, insertTotalMs: 0,
    searchCount: 0, searchTotalMs: 0,
    flushCount: 0, compactionCount: 0,
  };

  constructor(config: VectorConfig | RvfConfig) {
    const dimension = config.dimension ?? config.dimensions;
    if (!dimension) {
      throw new Error('Vector dimension is required (use dimension or dimensions)');
    }
    validateDimension(dimension);

    this.dim = dimension;
    this.metricType = config.metric ?? 'cosine';
    this.config = { ...config, dimension, dimensions: dimension } as RvfConfig;
    this.storagePath = (config as RvfConfig).storagePath ?? 'agentdb.rvf';
    this.batchThreshold = Math.min(
      Math.max(1, (config as RvfConfig).batchThreshold ?? DEFAULT_BATCH_THRESHOLD),
      MAX_BATCH_SIZE,
    );
  }

  /**
   * Initialize the RVF database connection.
   * Lazy-loads @ruvector/rvf to avoid hard dependency.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const storagePath = this.storagePath;
    if (storagePath !== ':memory:') {
      validatePath(storagePath);
    }

    try {
      const { RvfDatabase } = await import('@ruvector/rvf');
      const rvfBackendType = this.config.rvfBackend ?? 'auto';

      // Map AgentDB metric names to RVF metric names
      const rvfMetric = this.metricType === 'ip' ? 'dotproduct' : this.metricType;

      const fs = await import('fs');
      const fileExists = storagePath !== ':memory:' && fs.existsSync(storagePath);

      if (fileExists) {
        this.db = await RvfDatabase.open(storagePath, rvfBackendType);
      } else {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const createOpts: any = {
          dimensions: this.dim,
          metric: rvfMetric,
          m: this.config.M ?? 16,
          efConstruction: this.config.efConstruction ?? 200,
        };
        if (this.config.compression && this.config.compression !== 'none') {
          createOpts.compression = this.config.compression;
        }
        if (this.config.hardwareProfile !== undefined) {
          createOpts.profile = this.config.hardwareProfile;
        }
        this.db = await RvfDatabase.create(storagePath, createOpts, rvfBackendType);
      }

      // Get initial count
      try {
        const status = await this.db.status();
        this.cachedCount = status.totalVectors ?? 0;
      } catch {
        this.cachedCount = 0;
      }

      this.initialized = true;
    } catch (error) {
      const msg = (error as Error).message;
      throw new Error(
        `RVF backend initialization failed.\n` +
        `Install with: npm install @ruvector/rvf\n` +
        `Error: ${msg}`,
      );
    }
  }

  // ─── Sync VectorBackend interface (queues writes, cached reads) ───

  insert(id: string, embedding: Float32Array, metadata?: Record<string, unknown>): void {
    this.ensureInitialized();
    validateId(id);
    if (embedding.length !== this.dim) {
      throw new Error(`Vector dimension ${embedding.length} does not match expected ${this.dim}`);
    }
    if (this.pending.length >= MAX_PENDING_WRITES) {
      throw new Error(`Pending write queue full (${MAX_PENDING_WRITES}). Call flush() first.`);
    }

    this.pending.push({
      id,
      vector: embedding instanceof Float32Array ? embedding : new Float32Array(embedding),
      metadata: validateMetadata(metadata),
    });

    if (this.pending.length >= this.batchThreshold) {
      // Fire-and-forget flush (sync interface cannot await)
      this.flush().catch((err) => {
        console.error('[RvfBackend] Auto-flush failed:', err.message);
      });
    }
  }

  insertBatch(items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, unknown> }>): void {
    this.ensureInitialized();
    if (items.length > MAX_BATCH_SIZE) {
      throw new Error(`Batch size ${items.length} exceeds maximum of ${MAX_BATCH_SIZE}`);
    }
    for (const item of items) {
      this.insert(item.id, item.embedding, item.metadata);
    }
  }

  search(_query: Float32Array, _k: number, _options?: SearchOptions): SearchResult[] {
    // Sync search is not natively supported by RVF.
    // Throw with guidance to use searchAsync().
    throw new Error(
      'RVF backend is async-only for search. Use searchAsync() or the VectorBackendAsync interface.',
    );
  }

  remove(id: string): boolean {
    // Queue a delete -- sync interface cannot await
    this.ensureInitialized();
    validateId(id);
    // Fire-and-forget with error logging
    this.db.delete([id]).catch((err: Error) => {
      console.error('[RvfBackend] Delete failed:', err.message);
    });
    return true;
  }

  getStats(): VectorStats {
    return {
      count: this.cachedCount + this.pending.length,
      dimension: this.dim,
      metric: this.metricType,
      backend: 'rvf',
      memoryUsage: 0,
    };
  }

  async save(_path: string): Promise<void> {
    this.ensureInitialized();
    await this.flush();
    await this.db.compact();
  }

  async load(path: string): Promise<void> {
    validatePath(path);
    const { RvfDatabase } = await import('@ruvector/rvf');
    const rvfBackendType = this.config.rvfBackend ?? 'auto';
    if (this.db) {
      await this.db.close();
    }
    this.db = await RvfDatabase.open(path, rvfBackendType);
    const status = await this.db.status();
    this.cachedCount = status.totalVectors ?? 0;
    this.storagePath = path;
    this.initialized = true;
  }

  close(): void {
    if (this.db) {
      this.db.close().catch((err: Error) => {
        console.warn('[RvfBackend] Close error:', err.message);
      });
      this.db = null;
    }
    this.pending = [];
    this.initialized = false;
    this.cachedCount = 0;
  }

  // ─── Async VectorBackendAsync interface (native RVF operations) ───

  async insertAsync(id: string, embedding: Float32Array, metadata?: Record<string, unknown>): Promise<void> {
    this.ensureInitialized();
    validateId(id);
    if (embedding.length !== this.dim) {
      throw new Error(`Vector dimension ${embedding.length} does not match expected ${this.dim}`);
    }
    const start = this.config.enableStats !== false ? performance.now() : 0;

    await this.db.ingestBatch([{
      id,
      vector: embedding instanceof Float32Array ? embedding : new Float32Array(embedding),
      metadata: validateMetadata(metadata),
    }]);
    this.cachedCount++;

    if (this.config.enableStats !== false) {
      this.stats.insertCount++;
      this.stats.insertTotalMs += performance.now() - start;
    }
  }

  async insertBatchAsync(items: Array<{
    id: string;
    embedding: Float32Array;
    metadata?: Record<string, unknown>;
  }>): Promise<void> {
    this.ensureInitialized();
    if (items.length > MAX_BATCH_SIZE) {
      throw new Error(`Batch size ${items.length} exceeds maximum of ${MAX_BATCH_SIZE}`);
    }
    if (items.length === 0) return;

    const start = this.config.enableStats !== false ? performance.now() : 0;

    const entries = items.map((item) => ({
      id: item.id,
      vector: item.embedding instanceof Float32Array
        ? item.embedding
        : new Float32Array(item.embedding),
      metadata: item.metadata,
    }));

    const result = await this.db.ingestBatch(entries);
    this.cachedCount += result.accepted ?? items.length;

    if (this.config.enableStats !== false) {
      this.stats.insertCount += items.length;
      this.stats.insertTotalMs += performance.now() - start;
    }
  }

  async searchAsync(query: Float32Array, k: number, options?: SearchOptions): Promise<SearchResult[]> {
    this.ensureInitialized();
    if (!Number.isFinite(k) || k < 1) {
      throw new Error('k must be a positive finite integer');
    }
    const safeK = Math.min(Math.floor(k), MAX_SEARCH_K);
    const queryVec = query instanceof Float32Array ? query : new Float32Array(query);
    if (queryVec.length !== this.dim) {
      throw new Error(`Query dimension ${queryVec.length} does not match expected ${this.dim}`);
    }

    // Flush pending writes before searching so results are current
    if (this.pending.length > 0) {
      await this.flush();
    }

    const start = this.config.enableStats !== false ? performance.now() : 0;

    const rvfOpts: Record<string, unknown> = {};
    if (options?.efSearch) rvfOpts.efSearch = options.efSearch;
    if (options?.threshold) rvfOpts.minScore = options.threshold;
    // Wire filter expressions through to RVF query options
    if (options?.filter && typeof options.filter === 'object') {
      const builtFilter = FilterBuilder.buildFilter(options.filter as Parameters<typeof FilterBuilder.buildFilter>[0]);
      if (builtFilter) rvfOpts.filter = builtFilter;
    }
    const results = await this.db.query(queryVec, safeK, Object.keys(rvfOpts).length > 0 ? rvfOpts : undefined);

    if (this.config.enableStats !== false) {
      this.stats.searchCount++;
      this.stats.searchTotalMs += performance.now() - start;
    }

    const mapped: SearchResult[] = [];
    for (const r of results) {
      const similarity = this.distanceToSimilarity(r.distance);
      if (options?.threshold && similarity < options.threshold) continue;
      mapped.push({
        id: r.id,
        distance: r.distance,
        similarity,
        metadata: r.metadata,
      });
    }

    return mapped;
  }

  async removeAsync(id: string): Promise<boolean> {
    this.ensureInitialized();
    if (!id || typeof id !== 'string') return false;
    validateId(id);
    const result = await this.db.delete([id]);
    if (result.deleted > 0) {
      this.cachedCount = Math.max(0, this.cachedCount - result.deleted);
      return true;
    }
    return false;
  }

  async getStatsAsync(): Promise<VectorStats> {
    this.ensureInitialized();
    const status = await this.db.status();
    this.cachedCount = status.totalVectors ?? 0;
    return {
      count: this.cachedCount + this.pending.length,
      dimension: this.dim,
      metric: this.metricType,
      backend: 'rvf',
      memoryUsage: 0,
    };
  }

  async flush(): Promise<void> {
    if (this.pending.length === 0) return;
    this.ensureInitialized();

    const batch = this.pending.splice(0, this.pending.length);
    const entries = batch.map((item) => ({
      id: item.id,
      vector: item.vector,
      metadata: item.metadata,
    }));

    // Flush in sub-batches of MAX_BATCH_SIZE for safety
    for (let i = 0; i < entries.length; i += MAX_BATCH_SIZE) {
      const chunk = entries.slice(i, i + MAX_BATCH_SIZE);
      const result = await this.db.ingestBatch(chunk);
      this.cachedCount += result.accepted ?? chunk.length;
    }

    this.stats.flushCount++;
  }

  // ─── RVF-specific extensions ───

  /** Get the cryptographic file identity */
  async fileId(): Promise<string> {
    this.ensureInitialized();
    return this.db.fileId();
  }

  /** Get parent store ID (all zeros for root) */
  async parentId(): Promise<string> {
    this.ensureInitialized();
    return this.db.parentId();
  }

  /** Get lineage depth (0 for root) */
  async lineageDepth(): Promise<number> {
    this.ensureInitialized();
    return this.db.lineageDepth();
  }

  /** Create a COW branch */
  async derive(childPath: string): Promise<RvfBackend> {
    this.ensureInitialized();
    validatePath(childPath);

    const childDb = await this.db.derive(childPath);
    const child = new RvfBackend({ ...this.config, storagePath: childPath });
    child.db = childDb;
    child.initialized = true;
    try {
      const status = await childDb.status();
      child.cachedCount = status.totalVectors ?? 0;
    } catch {
      child.cachedCount = this.cachedCount;
    }
    return child;
  }

  /** Compact the store (reclaim dead space) */
  async compact(): Promise<{ segmentsCompacted: number; bytesReclaimed: number }> {
    this.ensureInitialized();
    await this.flush();
    const result = await this.db.compact();
    this.stats.compactionCount++;
    return {
      segmentsCompacted: result.segmentsCompacted ?? 0,
      bytesReclaimed: result.bytesReclaimed ?? 0,
    };
  }

  /** Get segment introspection */
  async segments(): Promise<Array<{ id: number; segType: string; payloadLength: number }>> {
    this.ensureInitialized();
    return this.db.segments();
  }

  /** Get RVF store status */
  async status(): Promise<{ totalVectors: number; totalSegments: number }> {
    this.ensureInitialized();
    const s = await this.db.status();
    this.cachedCount = s.totalVectors ?? 0;
    return { totalVectors: s.totalVectors ?? 0, totalSegments: s.totalSegments ?? 0 };
  }

  /** Get dimension */
  async getDimension(): Promise<number> {
    this.ensureInitialized();
    try {
      return await this.db.dimension();
    } catch {
      return this.dim;
    }
  }

  /** Get the storage path */
  getStoragePath(): string {
    return this.storagePath;
  }

  /** Check if the backend is initialized */
  isInitialized(): boolean {
    return this.initialized;
  }

  /** Get performance statistics */
  getPerformanceStats(): PerfStats & { avgInsertMs: number; avgSearchMs: number } {
    return {
      ...this.stats,
      avgInsertMs: this.stats.insertCount > 0
        ? this.stats.insertTotalMs / this.stats.insertCount
        : 0,
      avgSearchMs: this.stats.searchCount > 0
        ? this.stats.searchTotalMs / this.stats.searchCount
        : 0,
    };
  }

  // ─── AGI Capability Extensions (ADR-004) ───

  /** Get the distance metric name from the RVF store */
  metric(): string {
    this.ensureInitialized();
    if (typeof this.db.metric !== 'function') return this.metricType;
    try {
      const raw = this.db.metric();
      return typeof raw === 'string' ? raw : this.metricType;
    } catch {
      return this.metricType;
    }
  }

  /** Get HNSW index statistics */
  indexStats(): IndexStats {
    this.ensureInitialized();
    try {
      if (typeof this.db.indexStats !== 'function') {
        throw new Error('indexStats unsupported');
      }
      const raw = this.db.indexStats();
      return {
        indexedVectors: raw.indexedVectors ?? 0,
        layers: raw.layers ?? 0,
        m: raw.m ?? 16,
        efConstruction: raw.efConstruction ?? 200,
        needsRebuild: raw.needsRebuild ?? false,
      };
    } catch {
      return {
        indexedVectors: this.cachedCount,
        layers: 0,
        m: 16,
        efConstruction: 200,
        needsRebuild: false,
      };
    }
  }

  /**
   * Verify SHAKE-256 witness chain integrity.
   *
   * Fails closed: `valid` is true only when the store itself reports
   * `valid: true`. When the installed @ruvector/rvf has no verifyWitness
   * (0.2.x), or only an async one, the result is `unsupported`.
   */
  verifyWitness(): WitnessVerification {
    this.ensureInitialized();
    if (typeof this.db.verifyWitness !== 'function') {
      return unsupportedWitness('this @ruvector/rvf build does not provide verifyWitness()');
    }
    try {
      const raw = this.db.verifyWitness();
      if (raw && typeof raw.then === 'function') {
        raw.then(undefined, () => { /* result discarded; see verifyWitnessAsync() */ });
        return unsupportedWitness('verifyWitness() is async in this @ruvector/rvf build; use verifyWitnessAsync()');
      }
      return toWitnessVerification(raw);
    } catch (err) {
      return { valid: false, entries: 0, error: (err as Error).message };
    }
  }

  /** verifyWitness() that also supports async @ruvector/rvf implementations */
  async verifyWitnessAsync(): Promise<WitnessVerification> {
    this.ensureInitialized();
    if (typeof this.db.verifyWitness !== 'function') {
      return unsupportedWitness('this @ruvector/rvf build does not provide verifyWitness()');
    }
    try {
      return toWitnessVerification(await this.db.verifyWitness());
    } catch (err) {
      return { valid: false, entries: 0, error: (err as Error).message };
    }
  }

  /**
   * Snapshot-freeze state, returns epoch number.
   * Async because @ruvector/rvf 0.4.x freezes asynchronously.
   */
  async freeze(): Promise<number> {
    this.ensureInitialized();
    if (typeof this.db.freeze !== 'function') {
      throw new Error('unsupported: this @ruvector/rvf build does not provide freeze()');
    }
    const epoch = await this.db.freeze();
    if (typeof epoch !== 'number') {
      throw new Error(`freeze() returned ${typeof epoch}, expected an epoch number`);
    }
    return epoch;
  }

  // ─── ADR-007 Phase 1: Extended RVF APIs ───

  /**
   * Open an existing RVF store for read-only access (no lock required).
   * Enables concurrent reader patterns.
   */
  static async openReadonly(path: string, config?: Partial<RvfConfig>): Promise<RvfBackend> {
    if (path !== ':memory:') {
      validatePath(path);
    }

    const { RvfDatabase } = await import('@ruvector/rvf');
    const backendType = config?.rvfBackend ?? 'auto';
    const db = await RvfDatabase.openReadonly(path, backendType);

    // Probe dimension from the store
    let dim = config?.dimension ?? config?.dimensions ?? 0;
    try {
      dim = await db.dimension();
    } catch {
      if (!dim) throw new Error('Cannot determine dimension from read-only store');
    }

    const backend = new RvfBackend({
      dimension: dim,
      metric: config?.metric ?? 'cosine',
      storagePath: path,
      ...config,
    });
    backend.db = db;
    backend.initialized = true;
    try {
      const status = await db.status();
      backend.cachedCount = status.totalVectors ?? 0;
    } catch {
      backend.cachedCount = 0;
    }
    return backend;
  }

  /**
   * Delete vectors matching a filter expression.
   * @param filter - RvfFilterExpr or predicate DSL object
   * @returns delete result with count and epoch
   */
  async deleteByFilter(
    filter: RvfFilterExpr | Record<string, unknown>,
  ): Promise<{ deleted: number; epoch: number }> {
    this.ensureInitialized();

    // Accept either a raw RvfFilterExpr or a predicate DSL object
    let rvfFilter: RvfFilterExpr | null;
    if ('op' in filter) {
      rvfFilter = filter as RvfFilterExpr;
    } else {
      rvfFilter = FilterBuilder.buildFilter(filter as Parameters<typeof FilterBuilder.buildFilter>[0]);
    }

    if (!rvfFilter) {
      throw new Error('Cannot build filter expression from provided predicates');
    }

    try {
      const result = await this.db.deleteByFilter(rvfFilter);
      if (result.deleted > 0) {
        this.cachedCount = Math.max(0, this.cachedCount - result.deleted);
      }
      return { deleted: result.deleted ?? 0, epoch: result.epoch ?? 0 };
    } catch (err) {
      throw new Error(`deleteByFilter failed: ${(err as Error).message}`);
    }
  }

  /**
   * Embed a kernel image into the RVF store.
   * @returns segment ID of the embedded kernel
   */
  async embedKernel(
    arch: number,
    kernelType: number,
    flags: number,
    image: Uint8Array,
    apiPort: number,
    cmdline?: string,
  ): Promise<number> {
    this.ensureInitialized();
    try {
      return await this.db.embedKernel(arch, kernelType, flags, image, apiPort, cmdline);
    } catch (err) {
      throw new Error(`embedKernel failed: ${(err as Error).message}`);
    }
  }

  /**
   * Extract the kernel image from the RVF store.
   * @returns kernel data or null if not present
   */
  async extractKernel(): Promise<{ header: Uint8Array; image: Uint8Array } | null> {
    this.ensureInitialized();
    try {
      return await this.db.extractKernel();
    } catch {
      return null;
    }
  }

  /**
   * Embed an eBPF program into the RVF store.
   * @returns segment ID of the embedded program
   */
  async embedEbpf(
    programType: number,
    attachType: number,
    maxDimension: number,
    bytecode: Uint8Array,
    btf?: Uint8Array,
  ): Promise<number> {
    this.ensureInitialized();
    try {
      return await this.db.embedEbpf(programType, attachType, maxDimension, bytecode, btf);
    } catch (err) {
      throw new Error(`embedEbpf failed: ${(err as Error).message}`);
    }
  }

  /**
   * Extract the eBPF program from the RVF store.
   * @returns eBPF data or null if not present
   */
  async extractEbpf(): Promise<{ header: Uint8Array; payload: Uint8Array } | null> {
    this.ensureInitialized();
    try {
      return await this.db.extractEbpf();
    } catch {
      return null;
    }
  }

  // ─── Private helpers ───

  private distanceToSimilarity(distance: number): number {
    switch (this.metricType) {
      case 'cosine': return 1 - distance;
      case 'l2': return Math.exp(-distance);
      case 'ip': return -distance;
      default: return 1 - distance;
    }
  }

  private ensureInitialized(): void {
    if (!this.initialized || !this.db) {
      throw new Error('RvfBackend not initialized. Call initialize() first.');
    }
  }
}
