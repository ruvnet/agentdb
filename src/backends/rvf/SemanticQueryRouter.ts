/**
 * SemanticQueryRouter - Intent-Based Query Routing for AgentDB
 *
 * Wraps @ruvector/router's SemanticRouter for intelligent query routing
 * based on learned intent embeddings. Routes queries to appropriate
 * handlers (e.g., different memory stores, search strategies) based
 * on semantic similarity.
 *
 * Security:
 * - Intent names validated (length, characters)
 * - Embedding dimensions validated
 * - Route count bounded
 * - No file I/O (uses in-memory VectorDb)
 */

/** Route match result */
export interface RouteMatch {
  intent: string;
  score: number;
  metadata: Record<string, unknown>;
}

/** Intent configuration */
export interface IntentConfig {
  name: string;
  exemplars: Float32Array[];
  metadata?: Record<string, unknown>;
}

/** Router configuration */
export interface RouterConfig {
  /** Embedding dimension */
  dimension: number;
  /** Minimum similarity score for matches (default: 0.0 for distance metrics) */
  threshold?: number;
  /** Maximum number of intents (default: 1000) */
  maxIntents?: number;
  /** Path for router state persistence (ADR-007 Phase 1) */
  persistencePath?: string;
}

/** Router statistics */
export interface RouterStats {
  intentCount: number;
  dimension: number;
  totalQueries: number;
  avgLatencyMs: number;
}

// Bounds
const MAX_INTENTS = 10000;
const MAX_INTENT_NAME_LENGTH = 256;
const MAX_DIMENSION = 4096;
const MAX_EXEMPLARS = 100;
const MAX_PATH_LENGTH = 4096;

const FORBIDDEN_PATH_PATTERNS = [
  /\.\./,       // Path traversal
  /^\/etc\//i,  // System config
  /^\/proc\//i, // Process info
  /^\/sys\//i,  // System info
  /^\/dev\//i,  // Devices
];

function validatePath(inputPath: string): void {
  if (!inputPath || typeof inputPath !== 'string') {
    throw new Error('Path must be a non-empty string');
  }
  if (inputPath.length > MAX_PATH_LENGTH) {
    throw new Error(`Path exceeds maximum length of ${MAX_PATH_LENGTH}`);
  }
  if (inputPath.includes('\0')) {
    throw new Error('Path must not contain null bytes');
  }
  if (/\.\.[\\/]/.test(inputPath)) {
    throw new Error('Path contains forbidden traversal pattern');
  }
  for (const pattern of FORBIDDEN_PATH_PATTERNS) {
    if (pattern.test(inputPath)) {
      throw new Error('Path contains forbidden pattern');
    }
  }
}

/**
 * SemanticQueryRouter - Route queries to intents via learned embeddings
 *
 * Uses @ruvector/router's N-API VectorDb (HNSW + SIMD) for sub-millisecond
 * routing. Falls back to built-in brute-force search if @ruvector/router
 * is not available.
 */
export class SemanticQueryRouter {
  private dim: number;
  private threshold: number;
  private maxIntents: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private router: any = null;
  private fallbackIntents = new Map<string, { centroid: Float32Array; norm: number; metadata: Record<string, unknown> }>();
  private _totalQueries = 0;
  private _totalLatencyMs = 0;
  private _destroyed = false;
  private _useNative: boolean;
  private _persistencePath: string | undefined;
  private _persistTimer: ReturnType<typeof setTimeout> | null = null;
  private _persistDebounceMs = 5000;
  private _persistDirty = false;

  private constructor(config: RouterConfig) {
    this.dim = config.dimension;
    this.threshold = config.threshold ?? 0.0;
    this.maxIntents = Math.min(Math.max(1, config.maxIntents ?? 1000), MAX_INTENTS);
    this._useNative = false;
    if (config.persistencePath) validatePath(config.persistencePath);
    this._persistencePath = config.persistencePath;
  }

  /**
   * Create a new semantic query router.
   * Lazy-loads @ruvector/router; falls back to built-in search.
   */
  static async create(config: RouterConfig): Promise<SemanticQueryRouter> {
    if (!Number.isFinite(config.dimension) || config.dimension < 1 || config.dimension > MAX_DIMENSION) {
      throw new Error(`dimension must be between 1 and ${MAX_DIMENSION}`);
    }

    const instance = new SemanticQueryRouter(config);

    try {
      const { SemanticRouter } = await import('@ruvector/router');
      instance.router = new SemanticRouter({
        dimension: config.dimension,
        threshold: config.threshold ?? 0.0,
      });
      instance._useNative = true;
    } catch {
      // Fallback to built-in brute-force
      instance._useNative = false;
    }

    // ADR-007 Phase 1: attempt to restore persisted state
    if (config.persistencePath) {
      try {
        const loaded = await instance.load(config.persistencePath);
        if (loaded) instance._persistDirty = false;
      } catch { /* fresh start if load fails */ }
    }

    return instance;
  }

  /**
   * Check if @ruvector/router is available.
   */
  static async isAvailable(): Promise<boolean> {
    try {
      await import('@ruvector/router');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Add an intent with exemplar embeddings.
   * The centroid of exemplars is used for routing.
   */
  addIntent(config: IntentConfig): void {
    this.ensureAlive();

    if (!config.name || config.name.length > MAX_INTENT_NAME_LENGTH) {
      throw new Error(`Intent name must be 1-${MAX_INTENT_NAME_LENGTH} characters`);
    }
    if (!config.exemplars || config.exemplars.length === 0) {
      throw new Error('At least one exemplar embedding is required');
    }
    if (config.exemplars.length > MAX_EXEMPLARS) {
      throw new Error(`Maximum ${MAX_EXEMPLARS} exemplars per intent`);
    }

    // Validate dimensions
    for (const emb of config.exemplars) {
      if (emb.length !== this.dim) {
        throw new Error(`Exemplar dimension ${emb.length} does not match router dimension ${this.dim}`);
      }
    }

    const intentCount = this._useNative
      ? (this.router?.getIntents()?.length ?? 0)
      : this.fallbackIntents.size;
    if (intentCount >= this.maxIntents) {
      throw new Error(`Maximum ${this.maxIntents} intents reached`);
    }

    // Compute centroid
    const centroid = new Float32Array(this.dim);
    for (const emb of config.exemplars) {
      for (let i = 0; i < this.dim; i++) {
        centroid[i] += emb[i] / config.exemplars.length;
      }
    }

    const metadata = config.metadata ?? {};

    if (this._useNative && this.router) {
      this.router.addIntent({
        name: config.name,
        utterances: ['__embedding__'], // Required but unused for embedding routing
        embedding: centroid,
        metadata,
      });
    } else {
      let norm = 0;
      for (let i = 0; i < this.dim; i++) norm += centroid[i] * centroid[i];
      norm = Math.sqrt(norm);
      this.fallbackIntents.set(config.name, { centroid, norm, metadata });
    }

    this.schedulePersist();
  }

  /**
   * Route a query embedding to the best matching intent(s).
   */
  route(query: Float32Array, k: number = 1): RouteMatch[] {
    this.ensureAlive();
    if (query.length !== this.dim) {
      throw new Error(`Query dimension ${query.length} does not match router dimension ${this.dim}`);
    }

    const start = performance.now();
    let results: RouteMatch[];

    if (this._useNative && this.router) {
      results = this.router.routeWithEmbedding(query, k) as RouteMatch[];
    } else {
      results = this.fallbackRoute(query, k);
    }

    const elapsed = performance.now() - start;
    this._totalQueries++;
    this._totalLatencyMs += elapsed;

    return results;
  }

  /**
   * Remove an intent.
   */
  removeIntent(name: string): boolean {
    this.ensureAlive();
    let removed: boolean;
    if (this._useNative && this.router) {
      removed = this.router.removeIntent(name) as boolean;
    } else {
      removed = this.fallbackIntents.delete(name);
    }
    if (removed) this.schedulePersist();
    return removed;
  }

  /**
   * Get all intent names.
   */
  getIntents(): string[] {
    if (this._useNative && this.router) {
      return this.router.getIntents() as string[];
    }
    return Array.from(this.fallbackIntents.keys());
  }

  /**
   * Get router statistics.
   */
  getStats(): RouterStats {
    return {
      intentCount: this.getIntents().length,
      dimension: this.dim,
      totalQueries: this._totalQueries,
      avgLatencyMs: this._totalQueries > 0 ? this._totalLatencyMs / this._totalQueries : 0,
    };
  }

  /** Whether using native @ruvector/router */
  get isNative(): boolean {
    return this._useNative;
  }

  /** Check if destroyed */
  get isDestroyed(): boolean {
    return this._destroyed;
  }

  /**
   * Save router state to disk (ADR-007 Phase 1 persistence).
   * Requires @ruvector/router native save support or falls back to JSON.
   */
  async save(path: string): Promise<boolean> {
    this.ensureAlive();
    validatePath(path);
    if (this._useNative && this.router) {
      try {
        const { NativeAccelerator: N } = await import('./NativeAccelerator.js');
        const accel = new N();
        await accel.initialize();
        if (accel.routerPersistAvailable) return accel.routerSave(this.router, path);
      } catch { /* fall through to JSON */ }
    }
    // JSON fallback for fallback intents
    const { writeFile } = await import('node:fs/promises');
    const data: Array<{ name: string; centroid: number[]; metadata: Record<string, unknown> }> = [];
    for (const [name, { centroid, metadata }] of this.fallbackIntents) {
      data.push({ name, centroid: Array.from(centroid), metadata });
    }
    await writeFile(path, JSON.stringify({ dim: this.dim, threshold: this.threshold, intents: data }));
    return true;
  }

  /**
   * Load router state from disk (ADR-007 Phase 1 persistence).
   */
  async load(path: string): Promise<boolean> {
    this.ensureAlive();
    validatePath(path);
    if (this._useNative) {
      try {
        const { NativeAccelerator: N } = await import('./NativeAccelerator.js');
        const accel = new N();
        await accel.initialize();
        if (accel.routerPersistAvailable) {
          const loaded = await accel.routerLoad(path, { dimension: this.dim });
          if (loaded) { this.router = loaded; return true; }
        }
      } catch { /* fall through to JSON */ }
    }
    // JSON fallback
    try {
      const { readFile } = await import('node:fs/promises');
      const raw = JSON.parse(await readFile(path, 'utf-8'));
      if (raw.intents && Array.isArray(raw.intents)) {
        for (const intent of raw.intents) {
          const centroid = new Float32Array(intent.centroid);
          let norm = 0;
          for (let i = 0; i < centroid.length; i++) norm += centroid[i] * centroid[i];
          norm = Math.sqrt(norm);
          this.fallbackIntents.set(intent.name, { centroid, norm, metadata: intent.metadata ?? {} });
        }
        return true;
      }
    } catch { /* path not found or invalid */ }
    return false;
  }

  /** Whether a persistence path is configured */
  get persistencePath(): string | undefined {
    return this._persistencePath;
  }

  /**
   * Persist router state immediately (ADR-007 Phase 1).
   * Saves to the configured persistencePath using native or JSON fallback.
   */
  async persist(): Promise<boolean> {
    if (!this._persistencePath || this._destroyed) return false;
    this._persistDirty = false;
    return this.save(this._persistencePath);
  }

  /** Destroy the router */
  destroy(): void {
    if (!this._destroyed) {
      if (this._persistTimer) { clearTimeout(this._persistTimer); this._persistTimer = null; }
      this.router = null;
      this.fallbackIntents.clear();
      this._destroyed = true;
    }
  }

  // --- Private helpers ---

  private fallbackRoute(query: Float32Array, k: number): RouteMatch[] {
    // Pre-compute query norm once
    let qNorm = 0;
    for (let i = 0; i < query.length; i++) qNorm += query[i] * query[i];
    qNorm = Math.sqrt(qNorm);
    if (qNorm < 1e-10) return [];

    const scored: RouteMatch[] = [];
    for (const [name, { centroid, norm, metadata }] of this.fallbackIntents) {
      if (norm < 1e-10) continue;
      // Cosine similarity with pre-computed centroid norm
      let dot = 0;
      for (let i = 0; i < query.length; i++) dot += query[i] * centroid[i];
      const sim = dot / (qNorm * norm);
      if (sim >= this.threshold) {
        scored.push({ intent: name, score: sim, metadata });
      }
    }

    // For small k, use partial sort (selection) instead of full sort
    if (k < scored.length / 4) {
      return this.topK(scored, k);
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  private topK(items: RouteMatch[], k: number): RouteMatch[] {
    // Min-heap for top-k selection: O(n*log(k)) vs O(n*log(n)) for full sort
    const heap: RouteMatch[] = [];
    for (const item of items) {
      if (heap.length < k) {
        heap.push(item);
        if (heap.length === k) heap.sort((a, b) => a.score - b.score);
      } else if (item.score > heap[0].score) {
        heap[0] = item;
        // Bubble down to maintain min-heap property
        let i = 0;
        let settled = false;
        while (!settled) {
          const l = 2 * i + 1, r = 2 * i + 2;
          let smallest = i;
          if (l < k && heap[l].score < heap[smallest].score) smallest = l;
          if (r < k && heap[r].score < heap[smallest].score) smallest = r;
          if (smallest === i) {
            settled = true;
          } else {
            [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
            i = smallest;
          }
        }
      }
    }
    heap.sort((a, b) => b.score - a.score);
    return heap;
  }

  /**
   * Schedule a debounced persist (ADR-007 Phase 1).
   * Coalesces rapid route changes into a single write after 5s of quiet.
   */
  private schedulePersist(): void {
    if (!this._persistencePath || this._destroyed) return;
    this._persistDirty = true;
    if (this._persistTimer) clearTimeout(this._persistTimer);
    this._persistTimer = setTimeout(() => {
      this._persistTimer = null;
      this.persist().catch(() => { /* non-blocking */ });
    }, this._persistDebounceMs);
  }

  private ensureAlive(): void {
    if (this._destroyed) {
      throw new Error('SemanticQueryRouter has been destroyed. Create a new instance.');
    }
  }
}
