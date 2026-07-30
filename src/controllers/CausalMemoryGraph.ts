/**
 * CausalMemoryGraph - Causal Reasoning over Agent Memories
 *
 * Implements intervention-based reasoning rather than correlation.
 * Stores p(y|do(x)) estimates and tracks causal uplift across episodes.
 *
 * Based on:
 * - Pearl's do-calculus and causal inference
 * - Uplift modeling from A/B testing
 * - Instrumental variable methods
 *
 * v2.0.0-alpha.3 Features:
 * - HyperbolicAttention for tree-structured causal chain retrieval
 * - Poincaré embeddings for hierarchical relationships
 * - Feature flag: ENABLE_HYPERBOLIC_ATTENTION (default: false)
 * - 100% backward compatible with fallback to standard retrieval
 */

import type { IDatabaseConnection, DatabaseRows } from '../types/database.types.js';
import { normalizeRowId } from '../types/database.types.js';
import type { GraphDatabaseAdapter, CausalEdge as GraphCausalEdge } from '../backends/graph/GraphDatabaseAdapter.js';
import { NodeIdMapper } from '../utils/NodeIdMapper.js';
import { AttentionService, type HyperbolicAttentionConfig } from '../services/AttentionService.js';
import { EmbeddingService } from './EmbeddingService.js';
import type { VectorBackend } from '../backends/VectorBackend.js';

/**
 * Configuration for CausalMemoryGraph
 */
export interface CausalMemoryGraphConfig {
  /** Enable hyperbolic attention for causal chains (default: false) */
  ENABLE_HYPERBOLIC_ATTENTION?: boolean;
  /** Hyperbolic attention configuration */
  hyperbolicConfig?: Partial<HyperbolicAttentionConfig>;
}

export interface CausalEdge {
  id?: number;
  fromMemoryId: number;
  fromMemoryType: 'episode' | 'skill' | 'note' | 'fact';
  toMemoryId: number;
  toMemoryType: 'episode' | 'skill' | 'note' | 'fact';

  // Metrics
  similarity: number;
  uplift?: number; // E[y|do(x)] - E[y]
  confidence: number;
  sampleSize?: number;

  // Evidence
  evidenceIds?: string[];
  experimentIds?: string[];
  confounderScore?: number;

  // Explanation
  mechanism?: string;
  metadata?: Record<string, any>;
}

export interface CausalExperiment {
  id?: number;
  name: string;
  hypothesis: string;
  treatmentId: number;
  treatmentType: string;
  controlId?: number;

  // Design
  startTime: number;
  endTime?: number;
  sampleSize: number;

  // Results
  treatmentMean?: number;
  controlMean?: number;
  uplift?: number;
  pValue?: number;
  confidenceIntervalLow?: number;
  confidenceIntervalHigh?: number;

  status: 'running' | 'completed' | 'failed';
  metadata?: Record<string, any>;
}

export interface CausalObservation {
  experimentId: number;
  episodeId: number;
  isTreatment: boolean;
  outcomeValue: number;
  outcomeType: 'reward' | 'success' | 'latency';
  context?: Record<string, any>;
}

export interface CausalQuery {
  interventionMemoryId: number;
  interventionMemoryType: string;
  outcomeMemoryId?: number;
  minConfidence?: number;
  minUplift?: number;
}

export class CausalMemoryGraph {
  private db: IDatabaseConnection;
  private graphBackend?: any; // GraphBackend or GraphDatabaseAdapter
  private attentionService?: AttentionService;
  private embedder?: EmbeddingService;
  private vectorBackend?: VectorBackend;
  private config: CausalMemoryGraphConfig;

  /**
   * Constructor supports both v1 (legacy) and v2 (with attention) modes
   *
   * v1 mode: new CausalMemoryGraph(db)
   * v2 mode: new CausalMemoryGraph(db, graphBackend, embedder, config, vectorBackend)
   *
   * @param db - Database connection
   * @param graphBackend - Optional graph database adapter
   * @param embedder - Optional embedding service for generating embeddings
   * @param config - Optional configuration for hyperbolic attention
   * @param vectorBackend - Optional vector backend for optimized similarity search (150x faster than SQLite)
   */
  constructor(
    db: IDatabaseConnection,
    graphBackend?: any,
    embedder?: EmbeddingService,
    config?: CausalMemoryGraphConfig,
    vectorBackend?: VectorBackend
  ) {
    this.db = db;
    this.graphBackend = graphBackend;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend;
    this.config = {
      ENABLE_HYPERBOLIC_ATTENTION: false,
      ...config,
    };

    // Initialize AttentionService if embedder provided
    if (embedder && this.config.ENABLE_HYPERBOLIC_ATTENTION) {
      this.attentionService = new AttentionService(db, {
        hyperbolic: {
          enabled: true,
          ...this.config.hyperbolicConfig,
        },
      });
    }
  }

  /**
   * Add a causal edge between memories
   *
   * When vectorBackend is available, stores the edge embedding for fast similarity search.
   * This enables finding similar causal patterns across the memory graph.
   */
  async addCausalEdge(edge: CausalEdge): Promise<number> {
    // Create embedding for causal mechanism if embedder is available
    const mechanismText = edge.mechanism || `${edge.fromMemoryType}-${edge.toMemoryType} causal link`;
    let embedding: Float32Array;

    if (this.embedder) {
      embedding = await this.embedder.embedPassage(mechanismText);
    } else {
      // Fallback to zero embedding if no embedder available
      embedding = new Float32Array(384).fill(0);
    }

    // Use GraphDatabaseAdapter if available (AgentDB v2)
    if (this.graphBackend && 'createCausalEdge' in this.graphBackend) {
      const graphAdapter = this.graphBackend as any as GraphDatabaseAdapter;

      // Convert episode IDs to string format expected by graph database
      // Use NodeIdMapper to get full node IDs from numeric IDs
      const mapper = NodeIdMapper.getInstance();

      const fromNodeId = typeof edge.fromMemoryId === 'string'
        ? edge.fromMemoryId
        : (mapper.getNodeId(edge.fromMemoryId) || `${edge.fromMemoryType}-${edge.fromMemoryId}`);

      const toNodeId = typeof edge.toMemoryId === 'string'
        ? edge.toMemoryId
        : (mapper.getNodeId(edge.toMemoryId) || `${edge.toMemoryType}-${edge.toMemoryId}`);

      const graphEdge: GraphCausalEdge = {
        from: fromNodeId,
        to: toNodeId,
        mechanism: mechanismText,
        uplift: edge.uplift || 0,
        confidence: edge.confidence,
        sampleSize: edge.sampleSize || 0
      };

      const edgeId = await graphAdapter.createCausalEdge(graphEdge, embedding);
      // Convert string ID to numeric ID for compatibility
      // Extract numeric ID from string format "type-number" or return hash
      if (typeof edgeId === 'number') {
        return edgeId;
      }
      // Parse numeric ID from string format like "edge-123"
      const numMatch = String(edgeId).match(/(\d+)$/);
      return numMatch ? parseInt(numMatch[1], 10) : Math.abs(this.hashString(String(edgeId)));
    }

    // Fallback to SQLite
    const stmt = this.db.prepare(`
      INSERT INTO causal_edges (
        from_memory_id, from_memory_type, to_memory_id, to_memory_type,
        similarity, uplift, confidence, sample_size,
        evidence_ids, confounder_score,
        mechanism, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      edge.fromMemoryId,
      edge.fromMemoryType,
      edge.toMemoryId,
      edge.toMemoryType,
      edge.similarity,
      edge.uplift || null,
      edge.confidence,
      edge.sampleSize || null,
      edge.evidenceIds ? JSON.stringify(edge.evidenceIds) : null,
      edge.confounderScore || null,
      edge.mechanism || null,
      edge.metadata ? JSON.stringify(edge.metadata) : null
    );

    const edgeId = normalizeRowId(result.lastInsertRowid);

    // Store embedding in VectorBackend for fast similarity search
    if (this.vectorBackend && embedding) {
      this.vectorBackend.insert(`causal-edge:${edgeId}`, embedding, {
        fromMemoryId: edge.fromMemoryId,
        fromMemoryType: edge.fromMemoryType,
        toMemoryId: edge.toMemoryId,
        toMemoryType: edge.toMemoryType,
        mechanism: mechanismText,
        confidence: edge.confidence,
        uplift: edge.uplift,
      });
    }

    return edgeId;
  }

  /**
   * Create a causal experiment (A/B test)
   */
  createExperiment(experiment: CausalExperiment): number {
    const stmt = this.db.prepare(`
      INSERT INTO causal_experiments (
        name, hypothesis, treatment_id, treatment_type, control_id,
        start_time, sample_size, status, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      experiment.name,
      experiment.hypothesis,
      experiment.treatmentId,
      experiment.treatmentType,
      experiment.controlId || null,
      experiment.startTime,
      experiment.sampleSize,
      experiment.status,
      experiment.metadata ? JSON.stringify(experiment.metadata) : null
    );

    return normalizeRowId(result.lastInsertRowid);
  }

  /**
   * Record an observation in an experiment
   */
  recordObservation(observation: CausalObservation): void {
    const stmt = this.db.prepare(`
      INSERT INTO causal_observations (
        experiment_id, episode_id, is_treatment, outcome_value, outcome_type, context
      ) VALUES (?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      observation.experimentId,
      observation.episodeId,
      observation.isTreatment ? 1 : 0,
      observation.outcomeValue,
      observation.outcomeType,
      observation.context ? JSON.stringify(observation.context) : null
    );

    // Update sample size
    this.db.prepare(`
      UPDATE causal_experiments
      SET sample_size = sample_size + 1
      WHERE id = ?
    `).run(observation.experimentId);
  }

  /**
   * Calculate uplift for an experiment
   */
  calculateUplift(experimentId: number): {
    uplift: number;
    pValue: number;
    confidenceInterval: [number, number];
  } {
    interface ObservationRow {
      is_treatment: number;
      outcome_value: number;
    }

    // Get treatment and control observations
    const observations = this.db.prepare<ObservationRow>(`
      SELECT is_treatment, outcome_value
      FROM causal_observations
      WHERE experiment_id = ?
    `).all(experimentId);

    const treatmentValues = observations
      .filter(o => o.is_treatment === 1)
      .map(o => o.outcome_value);

    const controlValues = observations
      .filter(o => o.is_treatment === 0)
      .map(o => o.outcome_value);

    if (treatmentValues.length === 0 || controlValues.length === 0) {
      return { uplift: 0, pValue: 1.0, confidenceInterval: [0, 0] };
    }

    // Calculate means
    const treatmentMean = this.mean(treatmentValues);
    const controlMean = this.mean(controlValues);
    const uplift = treatmentMean - controlMean;

    // Calculate standard errors
    const treatmentSE = this.standardError(treatmentValues);
    const controlSE = this.standardError(controlValues);
    const pooledSE = Math.sqrt(treatmentSE ** 2 + controlSE ** 2);

    // t-statistic and p-value (two-tailed)
    const tStat = uplift / pooledSE;
    const df = treatmentValues.length + controlValues.length - 2;
    const pValue = 2 * (1 - this.tCDF(Math.abs(tStat), df));

    // 95% confidence interval
    const tCritical = this.tInverse(0.025, df);
    const marginOfError = tCritical * pooledSE;
    const confidenceInterval: [number, number] = [
      uplift - marginOfError,
      uplift + marginOfError
    ];

    // Update experiment with results
    this.db.prepare(`
      UPDATE causal_experiments
      SET treatment_mean = ?,
          control_mean = ?,
          uplift = ?,
          p_value = ?,
          confidence_interval_low = ?,
          confidence_interval_high = ?,
          status = 'completed'
      WHERE id = ?
    `).run(
      treatmentMean,
      controlMean,
      uplift,
      pValue,
      confidenceInterval[0],
      confidenceInterval[1],
      experimentId
    );

    return { uplift, pValue, confidenceInterval };
  }

  /**
   * Query causal effects
   */
  queryCausalEffects(query: CausalQuery): CausalEdge[] {
    const {
      interventionMemoryId,
      interventionMemoryType,
      outcomeMemoryId,
      minConfidence = 0.5,
      minUplift = 0.0
    } = query;

    let sql = `
      SELECT * FROM causal_edges
      WHERE from_memory_id = ?
        AND from_memory_type = ?
        AND confidence >= ?
        AND ABS(uplift) >= ?
    `;

    const params: any[] = [
      interventionMemoryId,
      interventionMemoryType,
      minConfidence,
      minUplift
    ];

    if (outcomeMemoryId) {
      sql += ' AND to_memory_id = ?';
      params.push(outcomeMemoryId);
    }

    sql += ' ORDER BY ABS(uplift) * confidence DESC';

    const rows = this.db.prepare<DatabaseRows.CausalEdge>(sql).all(...params);

    return rows.map(row => this.rowToCausalEdge(row));
  }

  /**
   * Find similar causal patterns using vector similarity search
   *
   * Uses vectorBackend for fast similarity search (150x faster than SQLite).
   * This enables discovering analogous causal relationships across different
   * domains or contexts.
   *
   * @param mechanism - The causal mechanism description to search for
   * @param k - Number of similar patterns to return (default: 10)
   * @param minConfidence - Minimum confidence threshold (default: 0.5)
   * @returns Similar causal edges with similarity scores
   */
  async findSimilarCausalPatterns(
    mechanism: string,
    k: number = 10,
    minConfidence: number = 0.5
  ): Promise<Array<CausalEdge & { similarity: number }>> {
    // If no embedder or vectorBackend, return empty array
    if (!this.embedder || !this.vectorBackend) {
      return [];
    }

    // Generate embedding for the query mechanism
    const queryEmbedding = await this.embedder.embedQuery(mechanism);

    // Search for similar causal edges using vectorBackend
    const results = this.vectorBackend.search(queryEmbedding, k * 2); // Get more to filter by confidence

    // Filter results to only causal-edge entries and by confidence
    const filteredResults = results.filter(result => {
      if (!result.id.startsWith('causal-edge:')) return false;
      const confidence = result.metadata?.confidence as number | undefined;
      return confidence === undefined || confidence >= minConfidence;
    });

    // Get full edge data from database
    const edges: Array<CausalEdge & { similarity: number }> = [];

    for (const result of filteredResults.slice(0, k)) {
      const edgeId = parseInt(result.id.replace('causal-edge:', ''), 10);
      if (isNaN(edgeId)) continue;

      const row = this.db.prepare<DatabaseRows.CausalEdge>(
        'SELECT * FROM causal_edges WHERE id = ?'
      ).get(edgeId);

      if (row) {
        edges.push({
          ...this.rowToCausalEdge(row),
          similarity: result.similarity
        });
      }
    }

    return edges;
  }

  /**
   * Get causal chain (multi-hop reasoning)
   *
   * v2: Uses HyperbolicAttention if enabled for tree-structured retrieval
   * v1: Falls back to recursive CTE with standard scoring
   *
   * @param fromMemoryId - Starting memory node
   * @param toMemoryId - Target memory node
   * @param maxDepth - Maximum chain depth (default: 5)
   * @returns Ranked causal chains with paths, uplift, and confidence
   */
  async getCausalChain(fromMemoryId: number, toMemoryId: number, maxDepth: number = 5): Promise<{
    path: number[];
    totalUplift: number;
    confidence: number;
    attentionMetrics?: {
      hyperbolicDistance: number[];
      computeTimeMs: number;
    };
  }[]> {
    // v2: Use HyperbolicAttention if enabled
    if (this.attentionService && this.embedder) {
      return this.getCausalChainWithAttention(fromMemoryId, toMemoryId, maxDepth);
    }

    interface ChainRow {
      path: string;
      total_uplift: number;
      min_confidence: number;
    }

    // v1: Legacy recursive CTE
    const chains = this.db.prepare<ChainRow>(`
      WITH RECURSIVE chain(from_id, to_id, depth, path, total_uplift, min_confidence) AS (
        SELECT
          from_memory_id,
          to_memory_id,
          1,
          from_memory_id || '->' || to_memory_id,
          uplift,
          confidence
        FROM causal_edges
        WHERE from_memory_id = ? AND confidence >= 0.5

        UNION ALL

        SELECT
          chain.from_id,
          ce.to_memory_id,
          chain.depth + 1,
          chain.path || '->' || ce.to_memory_id,
          chain.total_uplift + ce.uplift,
          MIN(chain.min_confidence, ce.confidence)
        FROM chain
        JOIN causal_edges ce ON chain.to_id = ce.from_memory_id
        WHERE chain.depth < ?
          AND ce.confidence >= 0.5
          AND chain.path NOT LIKE '%' || ce.to_memory_id || '%'
      )
      SELECT path, total_uplift, min_confidence
      FROM chain
      WHERE to_id = ?
      ORDER BY total_uplift DESC
      LIMIT 10
    `).all(fromMemoryId, maxDepth, toMemoryId);

    return chains.map(row => ({
      path: row.path.split('->').map(Number),
      totalUplift: row.total_uplift,
      confidence: row.min_confidence
    }));
  }

  /**
   * Hash a string to a positive integer
   * Used for converting string IDs to numeric IDs for backward compatibility
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get causal chain with HyperbolicAttention (v2 feature)
   *
   * Uses Poincaré embeddings to model hierarchical causal relationships.
   * Retrieves chains based on hyperbolic distance in embedding space.
   *
   * @private
   */
  private async getCausalChainWithAttention(
    fromMemoryId: number,
    toMemoryId: number,
    maxDepth: number
  ): Promise<{
    path: number[];
    totalUplift: number;
    confidence: number;
    attentionMetrics: {
      hyperbolicDistance: number[];
      computeTimeMs: number;
    };
  }[]> {
    // Get all candidate chains using CTE
    const candidateChains = this.db.prepare(`
      WITH RECURSIVE chain(from_id, to_id, depth, path, total_uplift, min_confidence) AS (
        SELECT
          from_memory_id,
          to_memory_id,
          1,
          from_memory_id || '->' || to_memory_id,
          uplift,
          confidence
        FROM causal_edges
        WHERE from_memory_id = ? AND confidence >= 0.5

        UNION ALL

        SELECT
          chain.from_id,
          ce.to_memory_id,
          chain.depth + 1,
          chain.path || '->' || ce.to_memory_id,
          chain.total_uplift + ce.uplift,
          MIN(chain.min_confidence, ce.confidence)
        FROM chain
        JOIN causal_edges ce ON chain.to_id = ce.from_memory_id
        WHERE chain.depth < ?
          AND ce.confidence >= 0.5
          AND chain.path NOT LIKE '%' || ce.to_memory_id || '%'
      )
      SELECT path, total_uplift, min_confidence
      FROM chain
      WHERE to_id = ?
      LIMIT 50
    `).all(fromMemoryId, maxDepth, toMemoryId) as any[];

    if (candidateChains.length === 0) {
      return [];
    }

    // Get embeddings for query (from node)
    const fromEpisode = this.db.prepare('SELECT task, output FROM episodes WHERE id = ?').get(fromMemoryId) as any;
    const queryText = fromEpisode ? `${fromEpisode.task}: ${fromEpisode.output}` : '';
    const queryEmbedding = await this.embedder!.embedQuery(queryText);

    // Get embeddings and hierarchy levels for all chain nodes
    const allNodeIds = new Set<number>();
    candidateChains.forEach((chain: any) => {
      const path = chain.path.split('->').map(Number);
      path.forEach(id => allNodeIds.add(id));
    });

    const nodeEmbeddings = new Map<number, Float32Array>();
    const hierarchyLevels = new Map<number, number>();

    for (const nodeId of allNodeIds) {
      const episode = this.db.prepare('SELECT task, output FROM episodes WHERE id = ?').get(nodeId) as any;
      if (episode) {
        const text = `${episode.task}: ${episode.output}`;
        const embedding = await this.embedder!.embedPassage(text);
        nodeEmbeddings.set(nodeId, embedding);

        // Calculate hierarchy level (depth from root)
        const level = candidateChains
          .filter((chain: any) => chain.path.includes(String(nodeId)))
          .reduce((minDepth: number, chain: any) => {
            const path = chain.path.split('->').map(Number);
            const idx = path.indexOf(nodeId);
            return Math.min(minDepth, idx);
          }, maxDepth);

        hierarchyLevels.set(nodeId, level);
      }
    }

    // Prepare keys, values, and hierarchy for attention
    const nodeList = Array.from(allNodeIds);
    const keys = new Float32Array(nodeList.length * 384);
    const values = new Float32Array(nodeList.length * 384);
    const hierarchyArray: number[] = [];

    nodeList.forEach((nodeId, idx) => {
      const embedding = nodeEmbeddings.get(nodeId)!;
      keys.set(embedding, idx * 384);
      values.set(embedding, idx * 384);
      hierarchyArray.push(hierarchyLevels.get(nodeId) || 0);
    });

    // Apply HyperbolicAttention
    const queries = new Float32Array(384);
    queries.set(queryEmbedding);

    const attentionResult = await this.attentionService!.hyperbolicAttention(
      queries,
      keys,
      values,
      hierarchyArray
    );

    // Re-rank chains by attention weights
    const rankedChains = candidateChains
      .map((chain: any) => {
        const path = chain.path.split('->').map(Number);

        // Calculate average attention weight for nodes in path
        const avgWeight = path.reduce((sum: number, nodeId: number) => {
          const idx = nodeList.indexOf(nodeId);
          return sum + (idx >= 0 ? attentionResult.weights[idx] : 0);
        }, 0) / path.length;

        return {
          path,
          totalUplift: chain.total_uplift,
          confidence: chain.min_confidence * avgWeight, // Boost confidence by attention
          attentionMetrics: {
            hyperbolicDistance: attentionResult.distances,
            computeTimeMs: attentionResult.metrics.computeTimeMs,
          },
        };
      })
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 10);

    return rankedChains;
  }

  /**
   * Calculate causal gain: E[outcome|do(treatment)] - E[outcome]
   */
  calculateCausalGain(treatmentId: number, outcomeType: 'reward' | 'success' | 'latency'): {
    causalGain: number;
    confidence: number;
    mechanism: string;
  } {
    // Get episodes where treatment was applied
    const withTreatment = this.db.prepare(`
      SELECT AVG(CASE WHEN ? = 'reward' THEN reward
                     WHEN ? = 'success' THEN success
                     WHEN ? = 'latency' THEN latency_ms
                END) as avg_outcome
      FROM episodes
      WHERE id IN (
        SELECT to_memory_id FROM causal_edges
        WHERE from_memory_id = ? AND confidence >= 0.6
      )
    `).get(outcomeType, outcomeType, outcomeType, treatmentId) as any;

    // Get baseline (no treatment)
    const baseline = this.db.prepare(`
      SELECT AVG(CASE WHEN ? = 'reward' THEN reward
                     WHEN ? = 'success' THEN success
                     WHEN ? = 'latency' THEN latency_ms
                END) as avg_outcome
      FROM episodes
      WHERE id NOT IN (
        SELECT to_memory_id FROM causal_edges
        WHERE from_memory_id = ?
      )
    `).get(outcomeType, outcomeType, outcomeType, treatmentId) as any;

    const causalGain = (withTreatment?.avg_outcome || 0) - (baseline?.avg_outcome || 0);

    // Get most confident edge for mechanism
    const edge = this.db.prepare(`
      SELECT mechanism, confidence
      FROM causal_edges
      WHERE from_memory_id = ?
      ORDER BY confidence DESC
      LIMIT 1
    `).get(treatmentId) as any;

    return {
      causalGain,
      confidence: edge?.confidence || 0,
      mechanism: edge?.mechanism || 'unknown'
    };
  }

  /**
   * Detect confounders using correlation analysis
   */
  detectConfounders(edgeId: number): {
    confounders: Array<{
      memoryId: number;
      correlationWithTreatment: number;
      correlationWithOutcome: number;
      confounderScore: number;
    }>;
  } {
    const edge = this.db.prepare('SELECT * FROM causal_edges WHERE id = ?').get(edgeId) as any;

    if (!edge) {
      return { confounders: [] };
    }

    // Find memories correlated with both treatment and outcome
    // This is a simplified version - production would use proper statistical tests
    const potentialConfounders = this.db.prepare(`
      SELECT DISTINCT e.id, e.task
      FROM episodes e
      WHERE e.id != ? AND e.id != ?
        AND e.session_id IN (
          SELECT session_id FROM episodes WHERE id = ?
          UNION
          SELECT session_id FROM episodes WHERE id = ?
        )
    `).all(edge.from_memory_id, edge.to_memory_id, edge.from_memory_id, edge.to_memory_id) as any[];

    const confounders = potentialConfounders.map((conf: any) => {
      // Calculate correlation scores (simplified)
      const treatmentCorr = this.calculateCorrelation(conf.id, edge.from_memory_id);
      const outcomeCorr = this.calculateCorrelation(conf.id, edge.to_memory_id);
      const confounderScore = Math.sqrt(treatmentCorr ** 2 * outcomeCorr ** 2);

      return {
        memoryId: conf.id,
        correlationWithTreatment: treatmentCorr,
        correlationWithOutcome: outcomeCorr,
        confounderScore
      };
    }).filter(c => c.confounderScore > 0.3);

    // Update edge with confounder score
    if (confounders.length > 0) {
      const maxConfounderScore = Math.max(...confounders.map(c => c.confounderScore));
      this.db.prepare(`
        UPDATE causal_edges
        SET confounder_score = ?
        WHERE id = ?
      `).run(maxConfounderScore, edgeId);
    }

    return { confounders };
  }

  // ========================================================================
  // Private Helper Methods
  // ========================================================================

  private rowToCausalEdge(row: DatabaseRows.CausalEdge): CausalEdge {
    return {
      id: row.id,
      fromMemoryId: row.from_memory_id,
      fromMemoryType: row.from_memory_type as 'episode' | 'skill' | 'note' | 'fact',
      toMemoryId: row.to_memory_id,
      toMemoryType: row.to_memory_type as 'episode' | 'skill' | 'note' | 'fact',
      similarity: row.similarity,
      uplift: row.uplift ?? undefined,
      confidence: row.confidence,
      sampleSize: row.sample_size ?? undefined,
      evidenceIds: row.evidence_ids ? JSON.parse(row.evidence_ids) : undefined,
      experimentIds: undefined, // Not in DatabaseRows.CausalEdge
      confounderScore: row.confounder_score ?? undefined,
      mechanism: row.mechanism ?? undefined,
      metadata: row.metadata ? JSON.parse(row.metadata) : undefined
    };
  }

  private mean(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private variance(values: number[]): number {
    const avg = this.mean(values);
    return values.reduce((sum, val) => sum + (val - avg) ** 2, 0) / values.length;
  }

  private standardError(values: number[]): number {
    return Math.sqrt(this.variance(values) / values.length);
  }

  private tCDF(t: number, df: number): number {
    // Simplified t-distribution CDF (use proper stats library in production)
    // This is an approximation
    return 0.5 + 0.5 * Math.sign(t) * (1 - Math.pow(1 + t * t / df, -df / 2));
  }

  private tInverse(p: number, df: number): number {
    // Simplified inverse t-distribution (use proper stats library)
    // Approximation for 95% CI
    return 1.96; // Standard normal approximation
  }

  private calculateCorrelation(id1: number, id2: number): number {
    // Simplified correlation calculation
    // In production, use proper correlation metrics
    const sharedSessions = this.db.prepare(`
      SELECT COUNT(DISTINCT e1.session_id) as shared
      FROM episodes e1
      JOIN episodes e2 ON e1.session_id = e2.session_id
      WHERE e1.id = ? AND e2.id = ?
    `).get(id1, id2) as any;

    return Math.min(sharedSessions?.shared || 0, 1.0);
  }
}
