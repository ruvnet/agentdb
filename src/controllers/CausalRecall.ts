/**
 * CausalRecall - Utility-Based Reranking + Certificate Issuer
 *
 * Combines:
 * 1. Vector similarity search
 * 2. Causal uplift from CausalMemoryGraph
 * 3. Utility-based reranking: U = α*similarity + β*uplift − γ*latencyCost
 * 4. Automatic certificate issuance via ExplainableRecall
 *
 * This is the main entry point for production retrieval with:
 * - Causal-aware ranking
 * - Explainable provenance
 * - Policy compliance
 */

// Database type from db-fallback
type Database = any;
import { CausalMemoryGraph, CausalEdge } from './CausalMemoryGraph.js';
import { ExplainableRecall, RecallCertificate } from './ExplainableRecall.js';
import { EmbeddingService } from './EmbeddingService.js';
import type { VectorBackend } from '../backends/VectorBackend.js';

export interface RerankConfig {
  alpha: number; // Similarity weight (default: 0.7)
  beta: number;  // Uplift weight (default: 0.2)
  gamma: number; // Latency penalty (default: 0.1)
  minConfidence?: number; // Min causal confidence (default: 0.6)
}

export interface RerankCandidate {
  id: string;
  type: 'episode' | 'skill' | 'note' | 'fact';
  content: string;
  similarity: number;
  uplift?: number;
  causalConfidence?: number;
  latencyMs?: number;
  utilityScore: number;
  rank: number;
}

export interface CausalRecallResult {
  candidates: RerankCandidate[];
  certificate: RecallCertificate;
  queryId: string;
  totalLatencyMs: number;
  metrics: {
    vectorSearchMs: number;
    causalLookupMs: number;
    rerankMs: number;
    certificateMs: number;
  };
}

export class CausalRecall {
  private db: Database;
  private causalGraph: CausalMemoryGraph;
  private explainableRecall: ExplainableRecall;
  private embedder: EmbeddingService;
  private vectorBackend?: VectorBackend;

  constructor(
    db: Database,
    embedder: EmbeddingService,
    vectorBackend?: VectorBackend,
    private config: RerankConfig = {
      alpha: 0.7,
      beta: 0.2,
      gamma: 0.1,
      minConfidence: 0.6
    }
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend;
    this.causalGraph = new CausalMemoryGraph(db);
    this.explainableRecall = new ExplainableRecall(db);
  }

  /**
   * Main recall function with utility-based reranking and certificate issuance
   *
   * @param queryId Unique query identifier
   * @param queryText Natural language query
   * @param k Number of results to return (default: 12)
   * @param requirements Optional list of requirements for completeness checking
   * @param accessLevel Security access level for certificate
   * @returns Reranked results with certificate
   */
  async recall(
    queryId: string,
    queryText: string,
    k: number = 12,
    requirements?: string[],
    accessLevel: 'public' | 'internal' | 'confidential' | 'restricted' = 'internal'
  ): Promise<CausalRecallResult> {
    const startTime = Date.now();
    const metrics = {
      vectorSearchMs: 0,
      causalLookupMs: 0,
      rerankMs: 0,
      certificateMs: 0
    };

    // Step 1: Vector similarity search
    const vectorStart = Date.now();
    const queryEmbedding = await this.embedder.embedQuery(queryText);
    const candidates = await this.vectorSearch(queryEmbedding, k * 2); // Fetch 2k for reranking
    metrics.vectorSearchMs = Date.now() - vectorStart;

    // Step 2: Load causal edges for candidates
    const causalStart = Date.now();
    const causalEdges = await this.loadCausalEdges(candidates.map(c => c.id));
    metrics.causalLookupMs = Date.now() - causalStart;

    // Step 3: Rerank by utility
    const rerankStart = Date.now();
    const reranked = this.rerankByUtility(candidates, causalEdges);
    const topK = reranked.slice(0, k);
    metrics.rerankMs = Date.now() - rerankStart;

    // Step 4: Issue certificate
    const certStart = Date.now();
    const certificate = await this.issueCertificate({
      queryId,
      queryText,
      candidates: topK,
      requirements: requirements || this.extractRequirements(queryText),
      accessLevel
    });
    metrics.certificateMs = Date.now() - certStart;

    const totalLatencyMs = Date.now() - startTime;

    return {
      candidates: topK,
      certificate,
      queryId,
      totalLatencyMs,
      metrics
    };
  }

  /**
   * Vector similarity search using cosine similarity
   */
  private async vectorSearch(
    queryEmbedding: Float32Array,
    k: number
  ): Promise<Array<{ id: string; type: string; content: string; similarity: number; latencyMs: number }>> {
    // Use optimized vector backend if available (100x faster)
    if (this.vectorBackend && typeof this.vectorBackend.search === 'function') {
      const searchResults = this.vectorBackend.search(queryEmbedding, k, {
        threshold: 0.0
      });

      // Fetch episode content from DB
      if (searchResults.length === 0) {
        return [];
      }

      const episodeIds = searchResults.map(r => r.id);
      const placeholders = episodeIds.map(() => '?').join(',');
      const episodes = this.db.prepare(`
        SELECT
          id,
          task || ' ' || COALESCE(output, '') as content,
          latency_ms
        FROM episodes
        WHERE id IN (${placeholders})
      `).all(...episodeIds) as any[];

      const episodeMap = new Map(episodes.map((e: any) => [e.id, e]));

      return searchResults.map(result => {
        const ep = episodeMap.get(result.id);
        return {
          id: result.id.toString(),
          type: 'episode',
          content: ep?.content || '',
          similarity: result.similarity,
          latencyMs: ep?.latency_ms || 0
        };
      });
    }

    // Fallback to SQL-based similarity search
    const results: any[] = [];
    const episodes = this.db.prepare(`
      SELECT
        e.id,
        'episode' as type,
        e.task || ' ' || COALESCE(e.output, '') as content,
        ee.embedding,
        e.latency_ms
      FROM episodes e
      JOIN episode_embeddings ee ON e.id = ee.episode_id
      ORDER BY e.ts DESC
      LIMIT ?
    `).all(k * 2);

    for (const ep of episodes) {
      const episodeRow = ep as any;
      const embedding = this.deserializeEmbedding(episodeRow.embedding);
      const similarity = this.cosineSimilarity(queryEmbedding, embedding);
      results.push({
        id: episodeRow.id.toString(),
        type: episodeRow.type,
        content: episodeRow.content,
        similarity,
        latencyMs: episodeRow.latency_ms || 0
      });
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  }

  /**
   * Load causal edges for candidates
   */
  private async loadCausalEdges(candidateIds: string[]): Promise<Map<string, CausalEdge[]>> {
    const edgeMap = new Map<string, CausalEdge[]>();

    if (candidateIds.length === 0) {
      return edgeMap;
    }

    const placeholders = candidateIds.map(() => '?').join(',');
    const edges = this.db.prepare(`
      SELECT * FROM causal_edges
      WHERE from_memory_id IN (${placeholders})
        AND confidence >= ?
    `).all(...candidateIds.map(id => parseInt(id)), this.config.minConfidence || 0.6) as any[];

    for (const edge of edges) {
      const fromId = edge.from_memory_id.toString();
      if (!edgeMap.has(fromId)) {
        edgeMap.set(fromId, []);
      }
      edgeMap.get(fromId)!.push({
        id: edge.id,
        fromMemoryId: edge.from_memory_id,
        fromMemoryType: edge.from_memory_type,
        toMemoryId: edge.to_memory_id,
        toMemoryType: edge.to_memory_type,
        similarity: edge.similarity,
        uplift: edge.uplift,
        confidence: edge.confidence,
        sampleSize: edge.sample_size,
        evidenceIds: edge.evidence_ids ? JSON.parse(edge.evidence_ids) : undefined,
        mechanism: edge.mechanism
      });
    }

    return edgeMap;
  }

  /**
   * Rerank by utility: U = α*similarity + β*uplift − γ*latencyCost
   */
  private rerankByUtility(
    candidates: Array<{ id: string; type: string; content: string; similarity: number; latencyMs: number }>,
    causalEdges: Map<string, CausalEdge[]>
  ): RerankCandidate[] {
    const { alpha, beta, gamma } = this.config;

    const reranked = candidates.map(candidate => {
      // Get causal uplift (average if multiple edges)
      const edges = causalEdges.get(candidate.id) || [];
      const avgUplift = edges.length > 0
        ? edges.reduce((sum, e) => sum + (e.uplift || 0), 0) / edges.length
        : 0;

      const avgConfidence = edges.length > 0
        ? edges.reduce((sum, e) => sum + e.confidence, 0) / edges.length
        : 0;

      // Normalize latency (assume max 1000ms)
      const latencyCost = Math.min(candidate.latencyMs / 1000, 1.0);

      // Calculate utility
      const utilityScore = alpha * candidate.similarity + beta * avgUplift - gamma * latencyCost;

      return {
        id: candidate.id,
        type: candidate.type as any,
        content: candidate.content,
        similarity: candidate.similarity,
        uplift: avgUplift,
        causalConfidence: avgConfidence,
        latencyMs: candidate.latencyMs,
        utilityScore,
        rank: 0 // Will be set after sorting
      };
    });

    // Sort by utility score descending
    reranked.sort((a, b) => b.utilityScore - a.utilityScore);

    // Assign ranks
    reranked.forEach((candidate, idx) => {
      candidate.rank = idx + 1;
    });

    return reranked;
  }

  /**
   * Issue certificate for the retrieval
   */
  private async issueCertificate(params: {
    queryId: string;
    queryText: string;
    candidates: RerankCandidate[];
    requirements: string[];
    accessLevel: 'public' | 'internal' | 'confidential' | 'restricted';
  }): Promise<RecallCertificate> {
    const { queryId, queryText, candidates, requirements, accessLevel } = params;

    const chunks = candidates.map(c => ({
      id: c.id,
      type: c.type,
      content: c.content,
      relevance: c.similarity
    }));

    return await this.explainableRecall.createCertificate({
      queryId,
      queryText,
      chunks,
      requirements,
      accessLevel
    });
  }

  /**
   * Extract requirements from query text (simple keyword extraction)
   */
  private extractRequirements(queryText: string): string[] {
    // Simple extraction: split on common words and filter
    const stopWords = new Set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'from', 'for', 'with', 'how', 'what', 'where', 'when', 'why', 'who']);

    const words = queryText
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 3 && !stopWords.has(w));

    // Return unique words
    return [...new Set(words)];
  }

  /**
   * Deserialize embedding from Buffer
   */
  private deserializeEmbedding(buffer: Buffer): Float32Array {
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vector dimensions must match');
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magnitudeA += a[i] * a[i];
      magnitudeB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  /**
   * Batch recall for multiple queries
   */
  async batchRecall(
    queries: Array<{ queryId: string; queryText: string; k?: number }>,
    requirements?: string[],
    accessLevel: 'public' | 'internal' | 'confidential' | 'restricted' = 'internal'
  ): Promise<CausalRecallResult[]> {
    const results: CausalRecallResult[] = [];

    for (const query of queries) {
      const result = await this.recall(
        query.queryId,
        query.queryText,
        query.k || 12,
        requirements,
        accessLevel
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Get recall statistics
   */
  getStats(): {
    totalCausalEdges: number;
    totalCertificates: number;
    avgRedundancyRatio: number;
    avgCompletenessScore: number;
  } {
    const causalEdges = this.db.prepare('SELECT COUNT(*) as count FROM causal_edges').get() as any;
    const certificates = this.db.prepare('SELECT COUNT(*) as count FROM recall_certificates').get() as any;

    const avgStats = this.db.prepare(`
      SELECT
        AVG(redundancy_ratio) as avg_redundancy,
        AVG(completeness_score) as avg_completeness
      FROM recall_certificates
    `).get() as any;

    return {
      totalCausalEdges: causalEdges.count,
      totalCertificates: certificates.count,
      avgRedundancyRatio: avgStats?.avg_redundancy || 0,
      avgCompletenessScore: avgStats?.avg_completeness || 0
    };
  }

  /**
   * Update rerank configuration
   */
  updateConfig(config: Partial<RerankConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Search for memories with semantic similarity and causal utility ranking
   *
   * @param params Search parameters
   * @returns Array of ranked search results with similarity and causal uplift scores
   */
  async search(params: {
    query: string;
    k?: number;
    includeEvidence?: boolean;
    alpha?: number;
    beta?: number;
    gamma?: number;
  }): Promise<Array<{
    id: number;
    type: string;
    content: string;
    similarity: number;
    causalUplift: number;
    utilityScore: number;
  }>> {
    const {
      query,
      k = 12,
      includeEvidence = false,
      alpha = this.config.alpha,
      beta = this.config.beta,
      gamma = this.config.gamma
    } = params;

    // Temporarily override config for this search
    const originalConfig = { ...this.config };
    this.config = { ...this.config, alpha, beta, gamma };

    try {
      // Step 1: Generate query embedding
      const queryEmbedding = await this.embedder.embedQuery(query);

      // Step 2: Vector similarity search
      const candidates = await this.vectorSearch(queryEmbedding, k * 2);

      // Step 3: Load causal edges for uplift scoring
      const causalEdges = await this.loadCausalEdges(candidates.map(c => c.id));

      // Step 4: Rerank by utility
      const reranked = this.rerankByUtility(candidates, causalEdges);

      // Step 5: Format results for search interface
      const results = reranked.slice(0, k).map(candidate => ({
        id: parseInt(candidate.id),
        type: candidate.type,
        content: candidate.content,
        similarity: candidate.similarity,
        causalUplift: candidate.uplift || 0,
        utilityScore: candidate.utilityScore
      }));

      return results;
    } finally {
      // Restore original config
      this.config = originalConfig;
    }
  }
}
