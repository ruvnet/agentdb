/**
 * HNSWIndex - Hierarchical Navigable Small World Index
 *
 * High-performance approximate nearest neighbor (ANN) search using HNSW algorithm.
 * Provides 10-100x speedup over brute-force search for large vector datasets.
 *
 * Features:
 * - HNSW indexing for sub-millisecond search
 * - Automatic index building and management
 * - Configurable M and efConstruction parameters
 * - Persistent index storage
 * - Graceful fallback to brute-force
 * - Multi-distance metric support (cosine, euclidean, ip)
 *
 * Note: hnswlib-node is lazy-loaded to avoid import failures on systems
 * without C++ build tools. Use forceWasm: true in AgentDB config to skip
 * hnswlib entirely and use pure WASM backends.
 */

import * as fs from 'fs';
import * as path from 'path';
import { validateFilterColumns } from '../security/input-validation.js';

// Lazy-loaded hnswlib-node to avoid import failures on systems without build tools
let HierarchicalNSW: any = null;
let hnswlibLoadAttempted = false;
let hnswlibLoadError: Error | null = null;

/**
 * Lazy-load hnswlib-node module.
 * Only loads when actually needed, avoiding import failures on systems without build tools.
 */
async function loadHnswlib(): Promise<boolean> {
  if (hnswlibLoadAttempted) {
    if (hnswlibLoadError) throw hnswlibLoadError;
    return HierarchicalNSW !== null;
  }
  hnswlibLoadAttempted = true;

  try {
    const hnswlibNode = await import('hnswlib-node');
    HierarchicalNSW = (hnswlibNode as any).default?.HierarchicalNSW
      || (hnswlibNode as any).HierarchicalNSW;
    return true;
  } catch (error) {
    hnswlibLoadError = new Error(
      `hnswlib-node failed to load: ${(error as Error).message}\n` +
      'This usually means native dependencies are missing.\n' +
      'Solutions:\n' +
      '  1. Install build tools and run: npm rebuild hnswlib-node\n' +
      '  2. Use AgentDB with forceWasm: true to skip hnswlib entirely\n' +
      '  3. Import from "agentdb/wasm" for WASM-only mode'
    );
    throw hnswlibLoadError;
  }
}

/**
 * Check if hnswlib-node is available without throwing.
 * Useful for conditional logic before attempting to use HNSWIndex.
 */
export async function isHnswlibAvailable(): Promise<boolean> {
  try {
    await loadHnswlib();
    return true;
  } catch {
    return false;
  }
}

// Database type from db-fallback
type Database = any;

export interface HNSWConfig {
  /** Maximum number of connections per layer (default: 16) */
  M: number;

  /** Size of dynamic candidate list during construction (default: 200) */
  efConstruction: number;

  /** Size of dynamic candidate list during search (default: 100) */
  efSearch: number;

  /** Distance metric: 'cosine', 'euclidean', 'ip' (inner product) */
  metric: 'cosine' | 'l2' | 'ip';

  /** Vector dimension */
  dimension: number;

  /** Maximum number of elements in index */
  maxElements: number;

  /** Enable persistent index storage */
  persistIndex: boolean;

  /** Path to store index file */
  indexPath?: string;

  /** Rebuild index threshold (rebuild when updates exceed this percentage) */
  rebuildThreshold: number;
}

export interface HNSWSearchResult {
  id: number;
  distance: number;
  similarity: number;
  metadata?: any;
}

export interface HNSWStats {
  enabled: boolean;
  indexBuilt: boolean;
  numElements: number;
  dimension: number;
  metric: string;
  M: number;
  efConstruction: number;
  efSearch: number;
  lastBuildTime: number | null;
  lastSearchTime: number | null;
  totalSearches: number;
  avgSearchTimeMs: number;
}

export class HNSWIndex {
  private db: Database;
  private config: HNSWConfig;
  private index: any | null = null;
  private vectorCache: Map<number, Float32Array> = new Map();
  private idToLabel: Map<number, number> = new Map();
  private labelToId: Map<number, number> = new Map();
  private nextLabel: number = 0;
  private indexBuilt: boolean = false;
  private updatesSinceLastBuild: number = 0;
  private totalSearches: number = 0;
  private totalSearchTime: number = 0;
  private lastBuildTime: number | null = null;
  private lastSearchTime: number | null = null;

  // Deferred loading: constructor can't be async, so we mark for lazy load
  private pendingPersistentLoad: boolean = false;
  private initializePromise: Promise<void> | null = null;

  constructor(db: Database, config?: Partial<HNSWConfig>) {
    this.db = db;
    this.config = {
      M: 16,
      efConstruction: 200,
      efSearch: 100,
      metric: 'cosine',
      dimension: 1536,
      maxElements: 100000,
      persistIndex: true,
      rebuildThreshold: 0.1, // Rebuild after 10% updates
      ...config,
    };

    // Mark for deferred loading (can't load in constructor as hnswlib is lazy-loaded)
    if (this.config.persistIndex && this.config.indexPath) {
      this.pendingPersistentLoad = true;
    }
  }

  /**
   * Initialize the index asynchronously.
   * Call this after construction if you need to load a persisted index.
   * This is automatically called by buildIndex() and search() if needed.
   */
  async initialize(): Promise<void> {
    if (this.initializePromise) {
      return this.initializePromise;
    }

    this.initializePromise = this.doInitialize();
    return this.initializePromise;
  }

  private async doInitialize(): Promise<void> {
    // Load hnswlib if we have a pending persistent load
    if (this.pendingPersistentLoad && this.config.indexPath) {
      this.pendingPersistentLoad = false;
      await this.loadIndexAsync();
    }
  }

  /**
   * Build HNSW index from database vectors
   */
  async buildIndex(tableName: string = 'pattern_embeddings'): Promise<void> {
    const start = Date.now();
    console.error(`[HNSWIndex] Building HNSW index from ${tableName}...`);

    // Ensure any pending initialization is complete
    await this.initialize();

    // Lazy-load hnswlib-node (avoids import failures on systems without build tools)
    await loadHnswlib();

    try {
      // Fetch all vectors from database
      const stmt = this.db.prepare(`
        SELECT pattern_id as id, embedding
        FROM ${tableName}
      `);

      const rows = stmt.all() as any[];

      if (rows.length === 0) {
        console.warn('[HNSWIndex] No vectors found in database');
        return;
      }

      // Create new HNSW index
      this.index = new HierarchicalNSW(this.config.metric, this.config.dimension);
      this.index.initIndex(
        Math.max(rows.length, this.config.maxElements),
        this.config.M,
        this.config.efConstruction
      );
      this.index.setEf(this.config.efSearch);

      // Clear mappings
      this.vectorCache.clear();
      this.idToLabel.clear();
      this.labelToId.clear();
      this.nextLabel = 0;

      // Add vectors to index
      console.error(`[HNSWIndex] Adding ${rows.length} vectors to index...`);

      for (const row of rows) {
        const id = row.id;
        const embedding = new Float32Array(
          (row.embedding as Buffer).buffer,
          (row.embedding as Buffer).byteOffset,
          (row.embedding as Buffer).byteLength / 4
        );

        // Add to index with label (convert Float32Array to number[])
        const label = this.nextLabel++;
        this.index.addPoint(Array.from(embedding), label);

        // Store mappings
        this.idToLabel.set(id, label);
        this.labelToId.set(label, id);
        this.vectorCache.set(id, embedding);
      }

      this.indexBuilt = true;
      this.updatesSinceLastBuild = 0;
      this.lastBuildTime = Date.now();

      const duration = (Date.now() - start) / 1000;
      console.error(`[HNSWIndex] ✅ Index built successfully in ${duration.toFixed(2)}s`);
      console.error(`[HNSWIndex] - Elements: ${rows.length}`);
      console.error(`[HNSWIndex] - Dimension: ${this.config.dimension}`);
      console.error(`[HNSWIndex] - M: ${this.config.M}`);
      console.error(`[HNSWIndex] - efConstruction: ${this.config.efConstruction}`);

      // Persist index if enabled
      if (this.config.persistIndex && this.config.indexPath) {
        await this.saveIndex();
      }
    } catch (error) {
      console.error('[HNSWIndex] Failed to build index:', error);
      this.indexBuilt = false;
      throw error;
    }
  }

  /**
   * Search HNSW index for k-nearest neighbors
   */
  async search(
    query: Float32Array,
    k: number,
    options?: {
      threshold?: number;
      filters?: Record<string, any>;
    }
  ): Promise<HNSWSearchResult[]> {
    // Ensure any pending initialization is complete (may load persisted index)
    await this.initialize();

    if (!this.index || !this.indexBuilt) {
      throw new Error('Index not built. Call buildIndex() first.');
    }

    const searchStart = Date.now();

    try {
      // Perform HNSW search (convert Float32Array to number[])
      const result = this.index.searchKnn(Array.from(query), k);

      const searchTime = Date.now() - searchStart;
      this.lastSearchTime = searchTime;
      this.totalSearches++;
      this.totalSearchTime += searchTime;

      // Convert results to our format
      const results: HNSWSearchResult[] = [];

      for (let i = 0; i < result.neighbors.length; i++) {
        const label = result.neighbors[i];
        const distance = result.distances[i];
        const id = this.labelToId.get(label);

        if (id === undefined) {
          console.warn(`[HNSWIndex] Label ${label} not found in mapping`);
          continue;
        }

        // Convert distance to similarity based on metric
        const similarity = this.distanceToSimilarity(distance);

        // Apply threshold if specified
        if (options?.threshold !== undefined && similarity < options.threshold) {
          continue;
        }

        results.push({
          id,
          distance,
          similarity,
        });
      }

      // Apply filters if specified (post-filtering)
      if (options?.filters) {
        return this.applyFilters(results, options.filters);
      }

      return results;
    } catch (error) {
      console.error('[HNSWIndex] Search failed:', error);
      throw error;
    }
  }

  /**
   * Add a single vector to the index
   */
  addVector(id: number, embedding: Float32Array): void {
    if (!this.index || !this.indexBuilt) {
      throw new Error('Index not built. Call buildIndex() first.');
    }

    const label = this.nextLabel++;
    this.index.addPoint(Array.from(embedding), label);

    this.idToLabel.set(id, label);
    this.labelToId.set(label, id);
    this.vectorCache.set(id, embedding);

    this.updatesSinceLastBuild++;

    // Check if rebuild is needed
    const totalElements = this.labelToId.size;
    const updatePercentage = this.updatesSinceLastBuild / totalElements;

    if (updatePercentage > this.config.rebuildThreshold) {
      console.error(`[HNSWIndex] Rebuild threshold reached (${(updatePercentage * 100).toFixed(1)}%)`);
    }
  }

  /**
   * Remove a vector from the index
   */
  removeVector(id: number): void {
    if (!this.index || !this.indexBuilt) {
      throw new Error('Index not built. Call buildIndex() first.');
    }

    const label = this.idToLabel.get(id);
    if (label === undefined) {
      console.warn(`[HNSWIndex] ID ${id} not found in index`);
      return;
    }

    // Note: hnswlib doesn't support deletion, so we mark for rebuild
    this.idToLabel.delete(id);
    this.labelToId.delete(label);
    this.vectorCache.delete(id);

    this.updatesSinceLastBuild++;
  }

  /**
   * Check if index needs rebuilding
   */
  needsRebuild(): boolean {
    if (!this.indexBuilt) return true;

    const totalElements = this.labelToId.size;
    if (totalElements === 0) return false;

    const updatePercentage = this.updatesSinceLastBuild / totalElements;
    return updatePercentage > this.config.rebuildThreshold;
  }

  /**
   * Save index to disk
   */
  private async saveIndex(): Promise<void> {
    if (!this.index || !this.config.indexPath) return;

    try {
      const indexDir = path.dirname(this.config.indexPath);
      if (!fs.existsSync(indexDir)) {
        fs.mkdirSync(indexDir, { recursive: true });
      }

      // Save HNSW index
      this.index.writeIndex(this.config.indexPath);

      // Save mappings
      const mappingsPath = this.config.indexPath + '.mappings.json';
      const mappings = {
        idToLabel: Array.from(this.idToLabel.entries()),
        labelToId: Array.from(this.labelToId.entries()),
        nextLabel: this.nextLabel,
        config: this.config,
      };

      fs.writeFileSync(mappingsPath, JSON.stringify(mappings, null, 2));

      console.error(`[HNSWIndex] Index saved to ${this.config.indexPath}`);
    } catch (error) {
      console.error('[HNSWIndex] Failed to save index:', error);
    }
  }

  /**
   * Load index from disk (async version for lazy loading)
   */
  private async loadIndexAsync(): Promise<void> {
    if (!this.config.indexPath || !fs.existsSync(this.config.indexPath)) {
      return;
    }

    try {
      console.error(`[HNSWIndex] Loading index from ${this.config.indexPath}...`);

      // Lazy-load hnswlib-node first
      await loadHnswlib();

      // Load HNSW index
      this.index = new HierarchicalNSW(this.config.metric, this.config.dimension);
      this.index.readIndex(this.config.indexPath);
      this.index.setEf(this.config.efSearch);

      // Load mappings
      const mappingsPath = this.config.indexPath + '.mappings.json';
      if (fs.existsSync(mappingsPath)) {
        const mappingsData = JSON.parse(fs.readFileSync(mappingsPath, 'utf-8'));

        this.idToLabel = new Map(mappingsData.idToLabel);
        this.labelToId = new Map(mappingsData.labelToId);
        this.nextLabel = mappingsData.nextLabel;
      }

      this.indexBuilt = true;
      console.error(`[HNSWIndex] ✅ Index loaded successfully (${this.labelToId.size} elements)`);
    } catch (error) {
      console.warn('[HNSWIndex] Failed to load index:', error);
      this.index = null;
      this.indexBuilt = false;
    }
  }

  /**
   * Convert distance to similarity based on metric
   */
  private distanceToSimilarity(distance: number): number {
    switch (this.config.metric) {
      case 'cosine':
        // Cosine distance is 1 - similarity
        return 1 - distance;

      case 'l2':
        // Euclidean distance: convert to similarity (0-1 range)
        // Using exponential decay: e^(-distance)
        return Math.exp(-distance);

      case 'ip':
        // Inner product: higher is more similar
        // Negate distance to get similarity
        return -distance;

      default:
        return 1 - distance;
    }
  }

  /**
   * Apply post-filtering to search results
   */
  private applyFilters(
    results: HNSWSearchResult[],
    filters: Record<string, any>
  ): HNSWSearchResult[] {
    const entries = Object.entries(filters);
    if (entries.length === 0 || results.length === 0) return results;

    // Keys are interpolated into SQL: only real pattern_embeddings columns
    const columns = validateFilterColumns(this.db, 'pattern_embeddings', entries.map(([key]) => key));
    const whereClause = columns.map((column) => `${column} = ?`).join(' AND ');
    const params = entries.map(([, value]) => value);

    // One query for all candidates instead of one per result
    const matching = new Set<number>();
    for (let i = 0; i < results.length; i += 500) {
      const ids = results.slice(i, i + 500).map((r) => r.id);
      const rows = this.db.prepare(`
        SELECT pattern_id FROM pattern_embeddings
        WHERE pattern_id IN (${ids.map(() => '?').join(',')}) AND ${whereClause}
      `).all(...ids, ...params) as Array<{ pattern_id: number }>;
      for (const row of rows) matching.add(Number(row.pattern_id));
    }

    return results.filter((result) => matching.has(Number(result.id)));
  }

  /**
   * Get index statistics
   */
  getStats(): HNSWStats {
    return {
      enabled: this.indexBuilt,
      indexBuilt: this.indexBuilt,
      numElements: this.labelToId.size,
      dimension: this.config.dimension,
      metric: this.config.metric,
      M: this.config.M,
      efConstruction: this.config.efConstruction,
      efSearch: this.config.efSearch,
      lastBuildTime: this.lastBuildTime,
      lastSearchTime: this.lastSearchTime,
      totalSearches: this.totalSearches,
      avgSearchTimeMs: this.totalSearches > 0 ? this.totalSearchTime / this.totalSearches : 0,
    };
  }

  /**
   * Update efSearch parameter for search quality/speed tradeoff
   */
  setEfSearch(ef: number): void {
    if (this.index) {
      this.index.setEf(ef);
      this.config.efSearch = ef;
      console.error(`[HNSWIndex] efSearch updated to ${ef}`);
    }
  }

  /**
   * Clear index and free memory
   */
  clear(): void {
    this.index = null;
    this.vectorCache.clear();
    this.idToLabel.clear();
    this.labelToId.clear();
    this.nextLabel = 0;
    this.indexBuilt = false;
    this.updatesSinceLastBuild = 0;
    console.error('[HNSWIndex] Index cleared');
  }

  /**
   * Check if index is built and ready
   */
  isReady(): boolean {
    return this.indexBuilt && this.index !== null;
  }
}
