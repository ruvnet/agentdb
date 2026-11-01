/**
 * ReasoningBank Controller - Pattern Storage and Retrieval
 *
 * Manages reasoning patterns with embeddings for semantic similarity search.
 * Integrates with ReasoningBank WASM for high-performance pattern matching.
 *
 * Pattern Structure:
 * - taskType: Type of task (e.g., "code_review", "data_analysis")
 * - approach: Description of the reasoning approach used
 * - successRate: Success rate of this pattern (0-1)
 * - embedding: Vector embedding of the pattern for similarity search
 * - metadata: Additional contextual information
 *
 * AgentDB v2 Migration:
 * - Uses VectorBackend abstraction for 8x faster search (RuVector/hnswlib)
 * - Optional GNN enhancement via LearningBackend
 * - 100% backward compatible with v1 API
 * - New features: useGNN option, recordOutcome for learning
 */

import type { IDatabaseConnection, DatabaseRows } from '../types/database.types.js';
import { normalizeRowId } from '../types/database.types.js';
import { EmbeddingService } from './EmbeddingService.js';
import type { VectorBackend, SearchResult } from '../backends/VectorBackend.js';
import { searchVectors } from '../backends/VectorBackend.js';
import { sqlExactSource } from '../backends/ExactSearch.js';
import {
  assertMetadata,
  assertTags,
  embeddingBlob,
  inTransaction,
  mergeMetadata,
  toPage,
  toRecordId,
  CrudValidationError,
} from './crud-helpers.js';
import { TopK, squaredNorm, blobToFloat32 } from '../utils/top-k.js';

/** Parse JSON from DB row values without throwing on malformed data. */
function safeJsonParse<T>(value: string | null | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

export interface ReasoningPattern {
  id?: number;
  taskType: string;
  approach: string;
  successRate: number;
  embedding?: Float32Array;
  uses?: number;
  avgReward?: number;
  tags?: string[];
  metadata?: Record<string, any>;
  createdAt?: number;
  similarity?: number; // Cosine similarity score (for search results)
  /** Rank score when reward-aware reranking is on (ADR-015) */
  score?: number;
}

/**
 * Reward-aware rerank weights (ADR-015):
 * score = similarity·w.similarity + successRate·w.successRate + avgReward·w.reward
 * with successRate and avgReward clamped to [0, 1].
 */
export interface RerankWeights {
  similarity: number;
  successRate: number;
  reward: number;
}

export const DEFAULT_RERANK_WEIGHTS: Readonly<RerankWeights> = Object.freeze({
  similarity: 0.7,
  successRate: 0.2,
  reward: 0.1,
});

export interface ReasoningBankOptions {
  /** Rerank search results by outcome as well as similarity (default true) */
  rerank?: boolean;
  /** Override the default weights; similarity should stay the largest */
  rerankWeights?: Partial<RerankWeights>;
}

function resolveWeights(base: RerankWeights, override?: Partial<RerankWeights>): RerankWeights {
  const weights = { ...base, ...override };
  for (const [name, value] of Object.entries(weights)) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`rerank weight ${name} must be a non-negative finite number`);
    }
  }
  return weights;
}

const clamp01 = (x: number | undefined): number =>
  Number.isFinite(x) ? Math.min(1, Math.max(0, x as number)) : 0;

/** Rank score for a pattern with a similarity (ADR-015) */
export function rerankScore(
  similarity: number,
  successRate: number | undefined,
  avgReward: number | undefined,
  weights: RerankWeights = DEFAULT_RERANK_WEIGHTS
): number {
  return weights.similarity * similarity
    + weights.successRate * clamp01(successRate)
    + weights.reward * clamp01(avgReward);
}

export interface PatternSearchQuery {
  /** v1 API: Task string (will be embedded automatically) */
  task?: string;
  /** v2 API: Pre-computed embedding */
  taskEmbedding?: Float32Array;
  k?: number;
  threshold?: number;
  /** Enable GNN-based query enhancement (requires LearningBackend) */
  useGNN?: boolean;
  /** Rerank by outcome as well as similarity; default from the constructor (true) */
  rerank?: boolean;
  /** Per-query rerank weights */
  rerankWeights?: Partial<RerankWeights>;
  filters?: {
    taskType?: string;
    minSuccessRate?: number;
    tags?: string[];
  };
}

export interface PatternStats {
  totalPatterns: number;
  avgSuccessRate: number;
  avgUses: number;
  topTaskTypes: Array<{ taskType: string; count: number }>;
  recentPatterns: number;
  highPerformingPatterns: number;
}

/**
 * Optional GNN Learning Backend for query enhancement
 */
export interface LearningBackend {
  /**
   * Enhance query embedding using GNN and neighbor context
   */
  enhance(query: Float32Array, neighbors: Float32Array[], weights: number[]): Float32Array;

  /**
   * Add training sample for future learning
   */
  addSample(embedding: Float32Array, success: boolean): void;

  /**
   * Train the GNN model
   */
  train(options?: { epochs?: number; batchSize?: number }): Promise<{
    epochs: number;
    finalLoss: number;
  }>;
}

/** How long getPatternStats() results are reused */
const PATTERN_STATS_TTL_MS = 5 * 60 * 1000;

export class ReasoningBank {
  private db: IDatabaseConnection;
  private embedder: EmbeddingService;
  private cache: Map<string, any>;

  // v2: Optional vector backend (uses legacy if not provided)
  private vectorBackend?: VectorBackend;
  private learningBackend?: LearningBackend;

  // Maps pattern ID (number) to vector backend ID (string) for hybrid mode
  private idMapping: Map<number, string> = new Map();
  private nextVectorId = 0;

  private rerankByDefault: boolean;
  private rerankWeights: RerankWeights;

  /**
   * Constructor supports both legacy (v1) and new (v2) modes
   *
   * Legacy mode (v1 - backward compatible):
   *   new ReasoningBank(db, embedder)
   *
   * New mode (v2 - with VectorBackend):
   *   new ReasoningBank(db, embedder, vectorBackend, learningBackend?)
   */
  constructor(
    db: IDatabaseConnection,
    embedder: EmbeddingService,
    vectorBackend?: VectorBackend,
    learningBackend?: LearningBackend,
    options: ReasoningBankOptions = {}
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend;
    this.learningBackend = learningBackend;
    this.rerankByDefault = options.rerank ?? true;
    this.rerankWeights = resolveWeights(DEFAULT_RERANK_WEIGHTS, options.rerankWeights);
    this.cache = new Map();
    this.initializeSchema();
  }

  /**
   * Initialize reasoning patterns schema
   */
  private initializeSchema(): void {
    // Create patterns table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS reasoning_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER DEFAULT (strftime('%s', 'now')),
        task_type TEXT NOT NULL,
        approach TEXT NOT NULL,
        success_rate REAL NOT NULL DEFAULT 0.0,
        uses INTEGER DEFAULT 0,
        avg_reward REAL DEFAULT 0.0,
        tags TEXT,
        metadata TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_patterns_task_type ON reasoning_patterns(task_type);
      CREATE INDEX IF NOT EXISTS idx_patterns_success_rate ON reasoning_patterns(success_rate);
      CREATE INDEX IF NOT EXISTS idx_patterns_uses ON reasoning_patterns(uses);
    `);

    // Create pattern embeddings table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS pattern_embeddings (
        pattern_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        FOREIGN KEY (pattern_id) REFERENCES reasoning_patterns(id) ON DELETE CASCADE
      );
    `);
  }

  /**
   * Store a reasoning pattern with embedding
   *
   * v1 (legacy): Stores in SQLite with pattern_embeddings table
   * v2 (VectorBackend): Stores metadata in SQLite, vectors in VectorBackend
   */
  async storePattern(pattern: ReasoningPattern): Promise<number> {
    // Generate embedding from approach text
    const embedding = await this.embedder.embedPassage(
      `${pattern.taskType}: ${pattern.approach}`
    );

    // Insert pattern metadata into SQLite
    const stmt = this.db.prepare(`
      INSERT INTO reasoning_patterns (
        task_type, approach, success_rate, uses, avg_reward, tags, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      pattern.taskType,
      pattern.approach,
      pattern.successRate,
      pattern.uses || 0,
      pattern.avgReward || 0.0,
      pattern.tags ? JSON.stringify(pattern.tags) : null,
      pattern.metadata ? JSON.stringify(pattern.metadata) : null
    );

    const patternId = normalizeRowId(result.lastInsertRowid);

    // Store embedding based on mode
    if (this.vectorBackend) {
      // v2: Use VectorBackend for high-performance search
      const vectorId = `pattern_${this.nextVectorId++}`;
      this.idMapping.set(patternId, vectorId);

      this.vectorBackend.insert(vectorId, embedding, {
        patternId,
        taskType: pattern.taskType,
        successRate: pattern.successRate,
      });
    }

    // Always keep the SQL copy: v1 searches it, and v2 falls back to it when
    // the index is small or comes back short (ADR-015)
    this.storePatternEmbedding(patternId, embedding);

    // Invalidate cache
    this.cache.clear();

    return patternId;
  }

  /**
   * Store pattern embedding
   */
  private storePatternEmbedding(patternId: number, embedding: Float32Array): void {
    const blob = embeddingBlob(embedding);

    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO pattern_embeddings (pattern_id, embedding)
      VALUES (?, ?)
    `);

    stmt.run(patternId, blob);
  }

  /**
   * Search patterns by semantic similarity
   *
   * v1 (legacy): Uses SQLite with cosine similarity computation
   * v2 (VectorBackend): Uses high-performance vector search (8x faster)
   * v2 + GNN: Optionally enhances query with learned patterns
   */
  async searchPatterns(query: PatternSearchQuery): Promise<ReasoningPattern[]> {
    const k = query.k || 10;
    const threshold = query.threshold || 0.0;

    // Generate embedding if task string provided (v1 API compatibility)
    let queryEmbedding: Float32Array;
    if (query.task && !query.taskEmbedding) {
      queryEmbedding = await this.embedder.embedQuery(query.task);
    } else if (query.taskEmbedding) {
      queryEmbedding = query.taskEmbedding;
    } else {
      throw new Error('PatternSearchQuery must provide either task (v1) or taskEmbedding (v2)');
    }

    // Create enriched query with embedding (ensure taskEmbedding is always defined)
    const enrichedQuery: PatternSearchQuery & { taskEmbedding: Float32Array } = {
      ...query,
      taskEmbedding: queryEmbedding
    };

    // Use VectorBackend if available (v2 mode)
    if (this.vectorBackend) {
      return this.searchPatternsV2(enrichedQuery);
    }

    // Legacy v1 search (100% backward compatible)
    return this.searchPatternsLegacy(enrichedQuery);
  }

  /**
   * v2: Search using VectorBackend with optional GNN enhancement
   */
  private async searchPatternsV2(query: PatternSearchQuery & { taskEmbedding: Float32Array }): Promise<ReasoningPattern[]> {
    const k = query.k || 10;
    const threshold = query.threshold || 0.0;
    let queryEmbedding = query.taskEmbedding;

    // Optional: Apply GNN enhancement
    if (query.useGNN && this.learningBackend) {
      // Get initial candidates for GNN context
      const candidates = await searchVectors(this.vectorBackend!, queryEmbedding, k * 3, { threshold: 0.0 });

      if (candidates.length > 0) {
        // Retrieve neighbor embeddings for GNN
        const neighborEmbeddings = await this.getEmbeddingsForVectorIds(
          candidates.map(c => c.id)
        );
        const weights = candidates.map(c => c.similarity);

        // Enhance query using GNN
        queryEmbedding = this.learningBackend.enhance(queryEmbedding, neighborEmbeddings, weights);
      }
    }

    // Reranking needs a candidate pool wider than k
    const weights = this.weightsFor(query);
    const poolSize = weights ? k * 3 : k;

    // Perform vector search; small or short index answers are completed by
    // exact cosine over pattern_embeddings (ADR-015)
    const results = await searchVectors(
      this.vectorBackend!,
      queryEmbedding,
      poolSize,
      { threshold },
      this.exactPatternSource()
    );

    // Hydrate with metadata from SQLite. A shared backend can return other
    // controllers' vectors, which carry no patternId.
    const patterns = this.hydratePatterns(results.filter((r) => r.metadata?.patternId != null));
    if (!weights) return patterns.slice(0, k);

    for (const p of patterns) p.score = rerankScore(p.similarity ?? 0, p.successRate, p.avgReward, weights);
    return patterns.sort((a, b) => b.score! - a.score!).slice(0, k);
  }

  /** Rerank weights for this query, or null when reranking is off */
  private weightsFor(query: PatternSearchQuery): RerankWeights | null {
    if (!(query.rerank ?? this.rerankByDefault)) return null;
    return query.rerankWeights ? resolveWeights(this.rerankWeights, query.rerankWeights) : this.rerankWeights;
  }

  /** pattern_embeddings as an exact-search source; index ids are per-process, so dedupe on patternId */
  private exactPatternSource() {
    return sqlExactSource(this.db, {
      table: 'pattern_embeddings',
      idColumn: 'pattern_id',
      vectorId: (id) => `pattern:${id}`,
      metadata: (id) => ({ patternId: id }),
      owns: (hit) => hit.metadata?.patternId != null,
      key: (hit) => String(hit.metadata?.patternId),
    });
  }

  /**
   * v1: Legacy search using SQLite (backward compatible)
   */
  private async searchPatternsLegacy(query: PatternSearchQuery & { taskEmbedding: Float32Array }): Promise<ReasoningPattern[]> {
    const k = query.k || 10;
    const threshold = query.threshold || 0.0;

    // Build WHERE clause for filters
    const conditions: string[] = [];
    const params: any[] = [];

    if (query.filters?.taskType) {
      conditions.push('rp.task_type = ?');
      params.push(query.filters.taskType);
    }

    if (query.filters?.minSuccessRate !== undefined) {
      conditions.push('rp.success_rate >= ?');
      params.push(query.filters.minSuccessRate);
    }

    if (query.filters?.tags && query.filters.tags.length > 0) {
      // Check if any of the tags match
      const tagConditions = query.filters.tags.map(() => 'rp.tags LIKE ?').join(' OR ');
      conditions.push(`(${tagConditions})`);
      query.filters.tags.forEach(tag => {
        params.push(`%"${tag}"%`);
      });
    }

    const whereClause = conditions.length > 0
      ? `WHERE ${conditions.join(' AND ')}`
      : '';

    // Retrieve all candidate patterns
    const stmt = this.db.prepare<DatabaseRows.ReasoningPattern & { embedding: Buffer }>(`
      SELECT
        rp.id,
        rp.ts,
        rp.task_type,
        rp.approach,
        rp.success_rate,
        rp.uses,
        rp.avg_reward,
        rp.tags,
        rp.metadata,
        pe.embedding
      FROM reasoning_patterns rp
      JOIN pattern_embeddings pe ON rp.id = pe.pattern_id
      ${whereClause}
    `);

    const rows = stmt.all(...params);

    // Threshold on similarity, then rank by the rerank score (or similarity
    // when reranking is off) in a bounded top-k; only winners are converted.
    const weights = this.weightsFor(query);
    const queryNorm = Math.sqrt(squaredNorm(query.taskEmbedding));
    const top = new TopK<{ row: (typeof rows)[number]; embedding: Float32Array; similarity: number }>(k);
    for (const row of rows) {
      const embedding = blobToFloat32(row.embedding as Buffer);
      const similarity = this.cosineSimilarity(query.taskEmbedding, embedding, queryNorm);
      if (!(similarity >= threshold)) continue;
      const score = weights ? rerankScore(similarity, row.success_rate, row.avg_reward, weights) : similarity;
      top.push(score, { row, embedding, similarity });
    }

    return top.sorted().map(({ score, item: { row, embedding, similarity } }) => ({
      id: row.id,
      taskType: row.task_type,
      approach: row.approach,
      successRate: row.success_rate,
      uses: row.uses,
      avgReward: row.avg_reward,
      tags: safeJsonParse(row.tags, []),
      metadata: safeJsonParse(row.metadata, {}),
      createdAt: row.ts,
      embedding,
      similarity,
      ...(weights ? { score } : {}),
    }));
  }

  /**
   * Hydrate search results with metadata from SQLite
   */
  private hydratePatterns(results: SearchResult[]): ReasoningPattern[] {
    // Prepare statement OUTSIDE loop for better-sqlite3 best practice
    const stmt = this.db.prepare(`
      SELECT * FROM reasoning_patterns WHERE id = ?
    `);

    return results.flatMap(result => {
      const patternId = result.metadata?.patternId;
      if (!patternId) {
        throw new Error(`VectorBackend result missing patternId: ${result.id}`);
      }

      const row = stmt.get(patternId) as any;

      // Deleted pattern still present in the index: skip it
      if (!row) return [];

      return [{
        id: row.id,
        taskType: row.task_type,
        approach: row.approach,
        successRate: row.success_rate,
        uses: row.uses,
        avgReward: row.avg_reward,
        tags: safeJsonParse(row.tags, []),
        metadata: safeJsonParse(row.metadata, {}),
        createdAt: row.ts,
        similarity: result.similarity,
      }];
    });
  }

  /**
   * Get embeddings for vector IDs (for GNN)
   */
  private async getEmbeddingsForVectorIds(vectorIds: string[]): Promise<Float32Array[]> {
    // In a full implementation, this would retrieve embeddings from VectorBackend
    // For now, we regenerate them from the database
    const embeddings: Float32Array[] = [];

    for (const vectorId of vectorIds) {
      // Find pattern ID from mapping
      let patternId: number | undefined;
      for (const [pid, vid] of this.idMapping.entries()) {
        if (vid === vectorId) {
          patternId = pid;
          break;
        }
      }

      if (patternId) {
        const pattern = this.getPattern(patternId);
        if (pattern?.approach) {
          const embedding = await this.embedder.embedPassage(
            `${pattern.taskType}: ${pattern.approach}`
          );
          embeddings.push(embedding);
        }
      }
    }

    return embeddings;
  }

  /**
   * Get pattern statistics
   */
  getPatternStats(): PatternStats {
    // Check cache first
    const cacheKey = 'pattern_stats';
    const cached = this.cache.get(cacheKey) as { value: PatternStats; expiresAt: number } | undefined;
    if (cached && cached.expiresAt > Date.now()) {
      return cached.value;
    }

    // Total patterns
    const totalRow = this.db.prepare<DatabaseRows.CountResult>(
      'SELECT COUNT(*) as count FROM reasoning_patterns'
    ).get();

    // Average success rate and uses
    const avgRow = this.db.prepare<DatabaseRows.AverageResult>(`
      SELECT
        AVG(success_rate) as avg_success_rate,
        AVG(uses) as avg_uses
      FROM reasoning_patterns
    `).get();

    // Top task types
    const topTaskTypes = this.db.prepare(`
      SELECT
        task_type,
        COUNT(*) as count
      FROM reasoning_patterns
      GROUP BY task_type
      ORDER BY count DESC
      LIMIT 10
    `).all() as any[];

    // Recent patterns (last 7 days)
    const recentRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM reasoning_patterns
      WHERE ts >= strftime('%s', 'now', '-7 days')
    `).get() as any;

    // High performing patterns (success_rate >= 0.8)
    const highPerfRow = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM reasoning_patterns
      WHERE success_rate >= 0.8
    `).get() as any;

    const stats: PatternStats = {
      totalPatterns: totalRow?.count ?? 0,
      avgSuccessRate: avgRow?.avg_success_rate ?? 0,
      avgUses: avgRow?.avg_uses ?? 0,
      topTaskTypes: topTaskTypes.map(row => ({
        taskType: row.task_type,
        count: row.count,
      })),
      recentPatterns: recentRow?.count ?? 0,
      highPerformingPatterns: highPerfRow?.count ?? 0,
    };

    // Cache for 5 minutes. Expiry is checked on read: a timer here kept every
    // short-lived process that called getPatternStats() alive for 5 minutes.
    this.cache.set(cacheKey, { value: stats, expiresAt: Date.now() + PATTERN_STATS_TTL_MS });

    return stats;
  }

  /**
   * Update pattern statistics after use
   */
  updatePatternStats(
    patternId: number,
    success: boolean,
    reward: number
  ): boolean {
    const stmt = this.db.prepare(`
      UPDATE reasoning_patterns
      SET
        uses = uses + 1,
        success_rate = (success_rate * uses + ?) / (uses + 1),
        avg_reward = (avg_reward * uses + ?) / (uses + 1)
      WHERE id = ?
    `);

    const result = stmt.run(success ? 1 : 0, reward, patternId);

    // Invalidate cache
    this.cache.clear();
    return result.changes > 0;
  }

  /**
   * Record pattern outcome for GNN learning (v2 feature)
   *
   * Updates pattern stats and adds training sample to LearningBackend
   * for future GNN model improvements.
   *
   * The updated success rate and average reward feed the reward-aware
   * rerank in searchPatterns() (ADR-015).
   *
   * @param patternId - Pattern ID to update
   * @param success - Whether the pattern was successful
   * @param reward - Optional reward value (default: 1 for success, 0 for failure)
   * @returns false when no pattern has that id
   */
  async recordOutcome(
    patternId: number,
    success: boolean,
    reward?: number
  ): Promise<boolean> {
    if (reward !== undefined && !Number.isFinite(reward)) {
      throw new Error('reward must be a finite number');
    }
    // Update pattern statistics
    const actualReward = reward !== undefined ? reward : (success ? 1.0 : 0.0);
    if (!this.updatePatternStats(patternId, success, actualReward)) return false;

    // Add to GNN training buffer if available
    if (this.learningBackend) {
      const pattern = this.getPattern(patternId);
      if (pattern?.approach) {
        const embedding = await this.embedder.embedPassage(
          `${pattern.taskType}: ${pattern.approach}`
        );
        this.learningBackend.addSample(embedding, success);
      }
    }
    return true;
  }

  /**
   * Train GNN model on collected samples (v2 feature)
   *
   * Trains the learning backend using accumulated pattern outcomes.
   * Requires LearningBackend to be configured.
   *
   * @param options - Training options (epochs, batchSize)
   * @returns Training results with epochs and final loss
   * @throws Error if LearningBackend not available
   */
  async trainGNN(options?: { epochs?: number; batchSize?: number }): Promise<{
    epochs: number;
    finalLoss: number;
  }> {
    if (!this.learningBackend) {
      throw new Error('GNN learning not available. Initialize ReasoningBank with LearningBackend.');
    }

    return this.learningBackend.train(options);
  }

  /**
   * Get pattern by ID
   */
  getPattern(patternId: number): ReasoningPattern | null {
    const stmt = this.db.prepare(`
      SELECT
        rp.id,
        rp.ts,
        rp.task_type,
        rp.approach,
        rp.success_rate,
        rp.uses,
        rp.avg_reward,
        rp.tags,
        rp.metadata,
        pe.embedding
      FROM reasoning_patterns rp
      LEFT JOIN pattern_embeddings pe ON rp.id = pe.pattern_id
      WHERE rp.id = ?
    `);

    const row = stmt.get(patternId) as any;
    if (!row) return null;

    return {
      id: row.id,
      taskType: row.task_type,
      approach: row.approach,
      successRate: row.success_rate,
      uses: row.uses,
      avgReward: row.avg_reward,
      tags: safeJsonParse(row.tags, []),
      metadata: safeJsonParse(row.metadata, {}),
      createdAt: row.ts,
      embedding: row.embedding
        ? new Float32Array(
            row.embedding.buffer,
            row.embedding.byteOffset,
            row.embedding.byteLength / 4
          )
        : undefined,
    };
  }

  /** Fields a pattern update may change. uses / successRate / avgReward belong to updatePatternStats(). */
  private static readonly PATTERN_PATCH_FIELDS = new Set(['taskType', 'approach', 'tags', 'metadata']);

  /**
   * Update a pattern's content in place, keeping its id (ADR-017).
   * Re-embeds only when taskType or approach changes; the SQL row, pattern_embeddings and the vector
   * backend move together. Usage statistics are not writable here — use updatePatternStats()/recordOutcome().
   * Metadata is merged by default (null removes a key); `{ replaceMetadata: true }` replaces it.
   * Returns null when the pattern does not exist.
   */
  async updatePattern(
    patternId: number | string,
    patch: Partial<{ taskType: string; approach: string; tags: string[] | null; metadata: Record<string, unknown> | null }>,
    options: { replaceMetadata?: boolean } = {},
  ): Promise<{ pattern: ReasoningPattern; reembedded: boolean; changedFields: string[] } | null> {
    const id = toRecordId(patternId, 'patternId');
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new CrudValidationError('patch must be an object', 'patch');
    const keys = Object.keys(patch).filter(k => (patch as Record<string, unknown>)[k] !== undefined);
    for (const k of keys) {
      if (!ReasoningBank.PATTERN_PATCH_FIELDS.has(k)) throw new CrudValidationError(`"${k}" cannot be updated`, k);
    }
    const current = this.db
      .prepare<{ task_type: string; approach: string; metadata: string | null }>(
        'SELECT task_type, approach, metadata FROM reasoning_patterns WHERE id = ?',
      )
      .get(id);
    if (!current) return null;
    if (keys.length === 0) return { pattern: this.getPattern(id)!, reembedded: false, changedFields: [] };

    const sets: string[] = [];
    const values: unknown[] = [];
    let taskType = current.task_type;
    let approach = current.approach;
    for (const field of ['taskType', 'approach'] as const) {
      const v = patch[field];
      if (v === undefined) continue;
      if (typeof v !== 'string' || !v.trim()) throw new CrudValidationError(`${field} must be a non-empty string`, field);
      if (field === 'taskType') taskType = v; else approach = v;
      sets.push(`${field === 'taskType' ? 'task_type' : 'approach'} = ?`);
      values.push(v);
    }
    if (patch.tags !== undefined) {
      sets.push('tags = ?');
      values.push(patch.tags === null ? null : JSON.stringify(assertTags(patch.tags)));
    }
    if (patch.metadata !== undefined) {
      const merged = patch.metadata === null ? null
        : options.replaceMetadata ? assertMetadata(patch.metadata)
        : mergeMetadata(safeJsonParse(current.metadata, {}), assertMetadata(patch.metadata));
      sets.push('metadata = ?');
      values.push(merged === null ? null : JSON.stringify(merged));
    }

    const reembed = taskType !== current.task_type || approach !== current.approach;
    const embedding = reembed ? await this.embedder.embedPassage(`${taskType}: ${approach}`) : undefined;

    inTransaction(this.db, () => {
      if (reembed) {
        const now = this.db
          .prepare<{ task_type: string; approach: string }>('SELECT task_type, approach FROM reasoning_patterns WHERE id = ?')
          .get(id);
        if (!now) throw new CrudValidationError(`pattern ${id} was deleted during the update`, 'patternId');
        if (now.task_type !== current.task_type || now.approach !== current.approach) {
          throw new CrudValidationError(`pattern ${id} was modified concurrently; retry the update`, 'patternId');
        }
      }
      this.db.prepare(`UPDATE reasoning_patterns SET ${sets.join(', ')} WHERE id = ?`).run(...values, id);
      if (embedding) this.storePatternEmbedding(id, embedding);
    });

    if (embedding && this.vectorBackend) {
      this.removePatternVector(id);
      const vectorId = `pattern_${this.nextVectorId++}`;
      this.idMapping.set(id, vectorId);
      const row = this.db.prepare<{ success_rate: number }>('SELECT success_rate FROM reasoning_patterns WHERE id = ?').get(id);
      this.vectorBackend.insert(vectorId, embedding, { patternId: id, taskType, successRate: row?.success_rate ?? 0 });
    }
    this.cache.clear();
    return { pattern: this.getPattern(id)!, reembedded: reembed, changedFields: keys };
  }

  /** Drop a pattern's entry from the vector backend so it can no longer occupy a top-k slot. */
  private removePatternVector(patternId: number): void {
    const vectorId = this.idMapping.get(patternId);
    if (!vectorId || !this.vectorBackend) return;
    try {
      this.vectorBackend.remove(vectorId);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // eslint-disable-next-line no-console
      console.warn(`[ReasoningBank] vector remove failed for pattern ${patternId}: ${msg}`);
    }
    this.idMapping.delete(patternId);
  }

  /**
   * List patterns, newest first, optionally filtered by task type. `total` counts every match.
   */
  listPatterns(options: { taskType?: string; limit?: number; offset?: number } = {}): {
    items: ReasoningPattern[]; total: number; limit: number; offset: number;
  } {
    const { limit, offset } = toPage(options.limit, options.offset);
    const where = options.taskType !== undefined ? 'WHERE task_type = ?' : '';
    const params = options.taskType !== undefined ? [String(options.taskType)] : [];
    const total = this.db.prepare<{ c: number }>(`SELECT COUNT(*) AS c FROM reasoning_patterns ${where}`).get(...params)?.c ?? 0;
    const rows = this.db
      .prepare<any>(
        `SELECT id, ts, task_type, approach, success_rate, uses, avg_reward, tags, metadata
         FROM reasoning_patterns ${where} ORDER BY id DESC LIMIT ? OFFSET ?`,
      )
      .all(...params, limit, offset);
    const items = rows.map((row: any) => ({
      id: row.id,
      taskType: row.task_type,
      approach: row.approach,
      successRate: row.success_rate,
      uses: row.uses,
      avgReward: row.avg_reward,
      tags: safeJsonParse(row.tags, []),
      metadata: safeJsonParse(row.metadata, {}),
      createdAt: row.ts,
    }));
    return { items, total, limit, offset };
  }

  /**
   * Delete pattern by ID
   */
  deletePattern(patternId: number): boolean {
    // Foreign-key cascades are off unless the connection enables them
    this.db.prepare('DELETE FROM pattern_embeddings WHERE pattern_id = ?').run(patternId);
    const stmt = this.db.prepare('DELETE FROM reasoning_patterns WHERE id = ?');
    const result = stmt.run(patternId);
    this.removePatternVector(patternId);

    // Invalidate cache
    this.cache.clear();

    return result.changes > 0;
  }

  /**
   * Clear query cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Calculate cosine similarity between two vectors (`normA` is a's
   * precomputed Math.sqrt(squaredNorm(a)))
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array, normA: number): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normB += b[i] * b[i];
    }

    const denom = normA * Math.sqrt(normB);
    return denom === 0 ? 0 : dotProduct / denom;
  }
}
