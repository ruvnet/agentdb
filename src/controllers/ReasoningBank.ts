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
    learningBackend?: LearningBackend
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend;
    this.learningBackend = learningBackend;
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
    const embedding = await this.embedder.embed(
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
    } else {
      // v1: Use legacy SQLite storage (backward compatible)
      this.storePatternEmbedding(patternId, embedding);
    }

    // Invalidate cache
    this.cache.clear();

    return patternId;
  }

  /**
   * Store pattern embedding
   */
  private storePatternEmbedding(patternId: number, embedding: Float32Array): void {
    const blob = Buffer.from(embedding.buffer);

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
      queryEmbedding = await this.embedder.embed(query.task);
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
      const candidates = this.vectorBackend!.search(queryEmbedding, k * 3, { threshold: 0.0 });

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

    // Perform vector search
    const results = this.vectorBackend!.search(queryEmbedding, k, { threshold });

    // Hydrate with metadata from SQLite
    return this.hydratePatterns(results);
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

    // Calculate similarities
    const candidates = rows.map(row => {
      const embedding = new Float32Array(
        (row.embedding as Buffer).buffer,
        (row.embedding as Buffer).byteOffset,
        (row.embedding as Buffer).byteLength / 4
      );

      const similarity = this.cosineSimilarity(query.taskEmbedding, embedding);

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
        embedding,
        similarity,
      };
    });

    // Filter by threshold and sort by similarity
    const filtered = candidates
      .filter(c => c.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);

    return filtered;
  }

  /**
   * Hydrate search results with metadata from SQLite
   */
  private hydratePatterns(results: SearchResult[]): ReasoningPattern[] {
    // Prepare statement OUTSIDE loop for better-sqlite3 best practice
    const stmt = this.db.prepare(`
      SELECT * FROM reasoning_patterns WHERE id = ?
    `);

    return results.map(result => {
      const patternId = result.metadata?.patternId;
      if (!patternId) {
        throw new Error(`VectorBackend result missing patternId: ${result.id}`);
      }

      const row = stmt.get(patternId) as any;

      if (!row) {
        throw new Error(`Pattern ${patternId} not found in database`);
      }

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
        similarity: result.similarity,
      };
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
          const embedding = await this.embedder.embed(
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
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
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

    // Cache for 5 minutes
    this.cache.set(cacheKey, stats);
    setTimeout(() => this.cache.delete(cacheKey), 5 * 60 * 1000);

    return stats;
  }

  /**
   * Update pattern statistics after use
   */
  updatePatternStats(
    patternId: number,
    success: boolean,
    reward: number
  ): void {
    const stmt = this.db.prepare(`
      UPDATE reasoning_patterns
      SET
        uses = uses + 1,
        success_rate = (success_rate * uses + ?) / (uses + 1),
        avg_reward = (avg_reward * uses + ?) / (uses + 1)
      WHERE id = ?
    `);

    stmt.run(success ? 1 : 0, reward, patternId);

    // Invalidate cache
    this.cache.clear();
  }

  /**
   * Record pattern outcome for GNN learning (v2 feature)
   *
   * Updates pattern stats and adds training sample to LearningBackend
   * for future GNN model improvements.
   *
   * @param patternId - Pattern ID to update
   * @param success - Whether the pattern was successful
   * @param reward - Optional reward value (default: 1 for success, 0 for failure)
   */
  async recordOutcome(
    patternId: number,
    success: boolean,
    reward?: number
  ): Promise<void> {
    // Update pattern statistics
    const actualReward = reward !== undefined ? reward : (success ? 1.0 : 0.0);
    this.updatePatternStats(patternId, success, actualReward);

    // Add to GNN training buffer if available
    if (this.learningBackend) {
      const pattern = this.getPattern(patternId);
      if (pattern?.approach) {
        const embedding = await this.embedder.embed(
          `${pattern.taskType}: ${pattern.approach}`
        );
        this.learningBackend.addSample(embedding, success);
      }
    }
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

  /**
   * Delete pattern by ID
   */
  deletePattern(patternId: number): boolean {
    const stmt = this.db.prepare('DELETE FROM reasoning_patterns WHERE id = ?');
    const result = stmt.run(patternId);

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
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dotProduct / denom;
  }
}
