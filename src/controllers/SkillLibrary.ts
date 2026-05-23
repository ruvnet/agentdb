/**
 * SkillLibrary - Lifelong Learning Skill Management
 *
 * Promotes high-reward trajectories into reusable skills.
 * Manages skill composition, relationships, and adaptive selection.
 *
 * Based on: "Voyager: An Open-Ended Embodied Agent with Large Language Models"
 * https://arxiv.org/abs/2305.16291
 */

import type { IDatabaseConnection, DatabaseRows } from '../types/database.types.js';
import { normalizeRowId } from '../types/database.types.js';
import { EmbeddingService } from './EmbeddingService.js';
import { VectorBackend } from '../backends/VectorBackend.js';
import type { GraphDatabaseAdapter } from '../backends/graph/GraphDatabaseAdapter.js';
import { NodeIdMapper } from '../utils/NodeIdMapper.js';
import { QueryCache, type QueryCacheConfig } from '../core/QueryCache.js';

/** Parse JSON from DB row values without throwing on malformed data. */
function safeJsonParse<T>(value: string | null | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

export interface Skill {
  id?: number;
  name: string;
  description?: string;
  signature?: {
    // v1 API: optional
    inputs: Record<string, any>;
    outputs: Record<string, any>;
  };
  code?: string;
  successRate: number;
  uses?: number; // v1 API: optional (defaults to 0)
  avgReward?: number; // v1 API: optional (defaults to 0)
  avgLatencyMs?: number; // v1 API: optional (defaults to 0)
  createdFromEpisode?: number;
  metadata?: Record<string, any>;
}

export interface SkillLink {
  parentSkillId: number;
  childSkillId: number;
  relationship: 'prerequisite' | 'alternative' | 'refinement' | 'composition';
  weight: number;
  metadata?: Record<string, any>;
}

export interface SkillQuery {
  /** v2 API: task description */
  task?: string;
  /** v1 API: query string (alias for task) */
  query?: string;
  k?: number;
  minSuccessRate?: number;
  preferRecent?: boolean;
}

export class SkillLibrary {
  private db: IDatabaseConnection;
  private embedder: EmbeddingService;
  private vectorBackend: VectorBackend | null;
  private graphBackend?: any; // GraphBackend or GraphDatabaseAdapter
  private queryCache: QueryCache;

  constructor(
    db: IDatabaseConnection,
    embedder: EmbeddingService,
    vectorBackend?: VectorBackend,
    graphBackend?: any,
    cacheConfig?: QueryCacheConfig
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vectorBackend = vectorBackend || null;
    this.graphBackend = graphBackend;
    this.queryCache = new QueryCache(cacheConfig);
  }

  /**
   * Create a new skill manually or from an episode
   * Invalidates skill cache
   */
  async createSkill(skill: Skill): Promise<number> {
    // Invalidate skills cache on write
    this.queryCache.invalidateCategory('skills');
    // Use GraphDatabaseAdapter if available (AgentDB v2)
    if (this.graphBackend && 'storeSkill' in this.graphBackend) {
      const graphAdapter = this.graphBackend as any as GraphDatabaseAdapter;

      const text = this.buildSkillText(skill);
      const embedding = await this.embedder.embed(text);

      const nodeId = await graphAdapter.storeSkill(
        {
          id: skill.id ? `skill-${skill.id}` : `skill-${Date.now()}-${Math.random()}`,
          name: skill.name,
          description: skill.description || '',
          code: skill.code || '',
          usageCount: skill.uses ?? 0,
          avgReward: skill.avgReward ?? 0,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          tags: JSON.stringify(skill.metadata || {}),
        },
        embedding
      );

      const numericId = parseInt(nodeId.split('-').pop() || '0', 36);
      NodeIdMapper.getInstance().register(numericId, nodeId);
      return numericId;
    }

    // Fallback to SQLite
    const stmt = this.db.prepare(`
      INSERT INTO skills (
        name, description, signature, code, success_rate, uses,
        avg_reward, avg_latency_ms, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    // v1 API compatibility: provide defaults for optional fields
    const signature = skill.signature || { inputs: {}, outputs: {} };
    const uses = skill.uses ?? 0;
    const avgReward = skill.avgReward ?? 0;
    const avgLatencyMs = skill.avgLatencyMs ?? 0;

    const result = stmt.run(
      skill.name,
      skill.description || null,
      JSON.stringify(signature),
      skill.code || null,
      skill.successRate,
      uses,
      avgReward,
      avgLatencyMs,
      skill.metadata ? JSON.stringify(skill.metadata) : null
    );

    const skillId = normalizeRowId(result.lastInsertRowid);

    // Generate and store embedding in VectorBackend
    const text = this.buildSkillText(skill);
    const embedding = await this.embedder.embed(text);

    // Store in VectorBackend with skill metadata (if available)
    if (this.vectorBackend) {
      this.vectorBackend.insert(`skill:${skillId}`, embedding, {
        name: skill.name,
        description: skill.description,
        successRate: skill.successRate,
        avgReward: skill.avgReward,
      });
    } else {
      // Legacy: store in database
      this.storeSkillEmbeddingLegacy(skillId, embedding);
    }

    return skillId;
  }

  /**
   * Update skill statistics after use
   * Invalidates skill cache
   */
  updateSkillStats(skillId: number, success: boolean, reward: number, latencyMs: number): void {
    // Invalidate skills cache on update
    this.queryCache.invalidateCategory('skills');
    const stmt = this.db.prepare(`
      UPDATE skills
      SET
        uses = uses + 1,
        success_rate = (success_rate * uses + ?) / (uses + 1),
        avg_reward = (avg_reward * uses + ?) / (uses + 1),
        avg_latency_ms = (avg_latency_ms * uses + ?) / (uses + 1)
      WHERE id = ?
    `);

    stmt.run(success ? 1 : 0, reward, latencyMs, skillId);
  }

  /**
   * Retrieve skills relevant to a task
   */
  async searchSkills(query: SkillQuery): Promise<Skill[]> {
    return this.retrieveSkills(query);
  }

  async retrieveSkills(query: SkillQuery): Promise<Skill[]> {
    // v1 API compatibility: accept both 'query' and 'task'
    const task = query.task || query.query;
    if (!task) {
      throw new Error('SkillQuery must provide either task (v2) or query (v1)');
    }

    const { k = 5, minSuccessRate = 0.5, preferRecent = true } = query;

    // Check cache first
    const cacheKey = this.queryCache.generateKey(
      'retrieveSkills',
      [task, k, minSuccessRate, preferRecent],
      'skills'
    );

    const cached = this.queryCache.get<Skill[]>(cacheKey);
    if (cached) {
      return cached;
    }

    // Generate query embedding
    const queryEmbedding = await this.embedder.embed(task);

    // Use GraphDatabaseAdapter if available (AgentDB v2)
    if (this.graphBackend && 'searchSkills' in this.graphBackend) {
      const graphAdapter = this.graphBackend as any as GraphDatabaseAdapter;

      const searchResults = await graphAdapter.searchSkills(queryEmbedding, k);

      const results = searchResults
        .map((result) => {
          // Handle metadata/tags parsing
          let metadata: any = undefined;
          if (result.tags) {
            if (typeof result.tags === 'string') {
              // Skip parsing if it's a String object representation
              if (!result.tags.startsWith('String(')) {
                try {
                  metadata = JSON.parse(result.tags);
                } catch (e) {
                  // Invalid JSON, skip
                  metadata = undefined;
                }
              }
            } else {
              // Already an object
              metadata = result.tags;
            }
          }

          return {
            id: parseInt(result.id.split('-').pop() || '0', 36),
            name: result.name,
            description: result.description,
            code: result.code,
            successRate: result.avgReward, // Use avgReward as successRate proxy
            uses: result.usageCount,
            avgReward: result.avgReward,
            metadata,
          };
        })
        .filter((skill) => skill.successRate >= minSuccessRate);

      // Cache the results
      this.queryCache.set(cacheKey, results);
      return results;
    }

    // Use VectorBackend for semantic search (if available)
    if (this.vectorBackend) {
      const searchResults = this.vectorBackend.search(queryEmbedding, k * 3);

      // Map results back to skill IDs and fetch full skill data
      const skillsWithSimilarity: (Skill & { similarity: number })[] = [];

      // Prepare statement ONCE outside loop (better-sqlite3 best practice)
      const getSkillStmt = this.db.prepare<DatabaseRows.Skill>('SELECT * FROM skills WHERE id = ?');

      for (const result of searchResults) {
        // Extract skill ID from vector ID (format: "skill:123")
        const skillId = parseInt(result.id.replace('skill:', ''));

        // Fetch full skill data from database
        const row = getSkillStmt.get(skillId);

        if (!row) continue;

        // Apply filters
        if (row.success_rate < minSuccessRate) continue;

        skillsWithSimilarity.push({
          id: row.id,
          name: row.name,
          description: row.description ?? undefined,
          signature: safeJsonParse(row.signature, undefined) as Skill["signature"],
          code: row.code ?? undefined,
          successRate: row.success_rate,
          uses: row.uses,
          avgReward: row.avg_reward,
          avgLatencyMs: row.avg_latency_ms,
          createdFromEpisode: row.created_from_episode ?? undefined,
          metadata: safeJsonParse(row.metadata, undefined),
          similarity: result.similarity,
        });
      }

      // Compute composite scores
      skillsWithSimilarity.sort((a, b) => {
        const scoreA = this.computeSkillScore(a);
        const scoreB = this.computeSkillScore(b);
        return scoreB - scoreA;
      });

      const results = skillsWithSimilarity.slice(0, k);

      // Cache the results
      this.queryCache.set(cacheKey, results);
      return results;
    } else {
      // Legacy: use SQL-based similarity search
      return this.retrieveSkillsLegacy(query);
    }
  }

  /**
   * Legacy SQL-based skill retrieval (fallback when VectorBackend not available)
   */
  private async retrieveSkillsLegacy(query: SkillQuery): Promise<Skill[]> {
    // v1 API compatibility: accept both 'query' and 'task'
    const task = query.task || query.query;
    if (!task) {
      throw new Error('SkillQuery must provide either task (v2) or query (v1)');
    }

    const { k = 5, minSuccessRate = 0.5 } = query;
    const queryEmbedding = await this.embedder.embed(task);

    // Fetch all skills with embeddings
    const stmt = this.db.prepare<DatabaseRows.Skill & { embedding: Buffer }>(`
      SELECT s.*, e.embedding
      FROM skills s
      LEFT JOIN skill_embeddings e ON s.id = e.skill_id
      WHERE s.success_rate >= ?
    `);
    const rows = stmt.all(minSuccessRate);

    // Compute similarities
    const skillsWithSimilarity: (Skill & { similarity: number })[] = [];
    for (const row of rows) {
      if (!row.embedding) continue;

      const embedding = new Float32Array(row.embedding.buffer);
      const similarity = this.cosineSimilarity(queryEmbedding, embedding);

      skillsWithSimilarity.push({
        id: row.id,
        name: row.name,
        description: row.description ?? undefined,
        signature: safeJsonParse(row.signature, undefined) as Skill["signature"],
        code: row.code ?? undefined,
        successRate: row.success_rate,
        uses: row.uses,
        avgReward: row.avg_reward,
        avgLatencyMs: row.avg_latency_ms,
        createdFromEpisode: row.created_from_episode ?? undefined,
        metadata: safeJsonParse(row.metadata, undefined),
        similarity,
      });
    }

    // Sort by composite score
    skillsWithSimilarity.sort((a, b) => {
      const scoreA = this.computeSkillScore(a);
      const scoreB = this.computeSkillScore(b);
      return scoreB - scoreA;
    });

    return skillsWithSimilarity.slice(0, k);
  }

  /**
   * Store skill embedding (legacy fallback)
   */
  private storeSkillEmbeddingLegacy(skillId: number, embedding: Float32Array): void {
    const stmt = this.db.prepare(`
      INSERT INTO skill_embeddings (skill_id, embedding)
      VALUES (?, ?)
      ON CONFLICT(skill_id) DO UPDATE SET embedding = excluded.embedding
    `);
    const buffer = Buffer.from(embedding.buffer);
    stmt.run(skillId, buffer);
  }

  /**
   * Cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Link two skills with a relationship
   */
  linkSkills(link: SkillLink): void {
    const stmt = this.db.prepare(`
      INSERT INTO skill_links (parent_skill_id, child_skill_id, relationship, weight, metadata)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(parent_skill_id, child_skill_id, relationship)
      DO UPDATE SET weight = excluded.weight
    `);

    stmt.run(
      link.parentSkillId,
      link.childSkillId,
      link.relationship,
      link.weight,
      link.metadata ? JSON.stringify(link.metadata) : null
    );
  }

  /**
   * Get skill composition plan (prerequisites and alternatives)
   */
  getSkillPlan(skillId: number): {
    skill: Skill;
    prerequisites: Skill[];
    alternatives: Skill[];
    refinements: Skill[];
  } {
    // Get main skill
    const skill = this.getSkillById(skillId);

    // Get prerequisites
    const prereqStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'prerequisite'
      ORDER BY sl.weight DESC
    `);
    const prerequisites = prereqStmt.all(skillId).map(this.rowToSkill);

    // Get alternatives
    const altStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'alternative'
      ORDER BY sl.weight DESC, s.success_rate DESC
    `);
    const alternatives = altStmt.all(skillId).map(this.rowToSkill);

    // Get refinements
    const refStmt = this.db.prepare(`
      SELECT s.* FROM skills s
      JOIN skill_links sl ON s.id = sl.child_skill_id
      WHERE sl.parent_skill_id = ? AND sl.relationship = 'refinement'
      ORDER BY sl.weight DESC, s.created_at DESC
    `);
    const refinements = refStmt.all(skillId).map(this.rowToSkill);

    return { skill, prerequisites, alternatives, refinements };
  }

  /**
   * Consolidate high-reward episodes into skills with ML pattern extraction
   * This is the core learning mechanism enhanced with pattern analysis
   */
  async consolidateEpisodesIntoSkills(config: {
    minAttempts?: number;
    minReward?: number;
    timeWindowDays?: number;
    extractPatterns?: boolean;
  }): Promise<{
    created: number;
    updated: number;
    patterns: Array<{
      task: string;
      commonPatterns: string[];
      successIndicators: string[];
      avgReward: number;
    }>;
  }> {
    const { minAttempts = 3, minReward = 0.7, timeWindowDays = 7, extractPatterns = true } = config;

    interface ConsolidationCandidate {
      task: string;
      attempt_count: number;
      avg_reward: number;
      success_rate: number;
      avg_latency: number | null;
      latest_episode_id: number;
      episode_ids: string;
    }

    const stmt = this.db.prepare<ConsolidationCandidate>(`
      SELECT
        task,
        COUNT(*) as attempt_count,
        AVG(reward) as avg_reward,
        AVG(success) as success_rate,
        AVG(latency_ms) as avg_latency,
        MAX(id) as latest_episode_id,
        GROUP_CONCAT(id) as episode_ids
      FROM episodes
      WHERE ts > strftime('%s', 'now') - ?
        AND reward >= ?
      GROUP BY task
      HAVING attempt_count >= ?
    `);

    const candidates = stmt.all(timeWindowDays * 86400, minReward, minAttempts);
    let created = 0;
    let updated = 0;
    const patterns: Array<{
      task: string;
      commonPatterns: string[];
      successIndicators: string[];
      avgReward: number;
    }> = [];

    for (const candidate of candidates) {
      const episodeIds = candidate.episode_ids.split(',').map(Number);

      // Extract patterns from successful episodes if requested
      let extractedPatterns: string[] = [];
      let successIndicators: string[] = [];
      let enhancedDescription = `Auto-generated skill from successful episodes`;

      if (extractPatterns) {
        const patternData = await this.extractPatternsFromEpisodes(episodeIds);
        extractedPatterns = patternData.commonPatterns;
        successIndicators = patternData.successIndicators;

        if (extractedPatterns.length > 0) {
          enhancedDescription = `Skill learned from ${episodeIds.length} successful episodes. Common patterns: ${extractedPatterns.slice(0, 3).join(', ')}`;
        }

        patterns.push({
          task: candidate.task,
          commonPatterns: extractedPatterns,
          successIndicators: successIndicators,
          avgReward: candidate.avg_reward,
        });
      }

      // Check if skill already exists
      const existing = this.db.prepare('SELECT id FROM skills WHERE name = ?').get(candidate.task);

      if (!existing) {
        // Create new skill with extracted patterns
        const skill: Skill = {
          name: candidate.task,
          description: enhancedDescription,
          signature: {
            inputs: { task: 'string' },
            outputs: { result: 'any' },
          },
          successRate: candidate.success_rate,
          uses: candidate.attempt_count,
          avgReward: candidate.avg_reward,
          avgLatencyMs: candidate.avg_latency ?? 0,
          createdFromEpisode: candidate.latest_episode_id,
          metadata: {
            sourceEpisodes: episodeIds,
            autoGenerated: true,
            consolidatedAt: Date.now(),
            extractedPatterns: extractedPatterns,
            successIndicators: successIndicators,
            patternConfidence: this.calculatePatternConfidence(
              episodeIds.length,
              candidate.success_rate
            ),
          },
        };

        await this.createSkill(skill);
        created++;
      } else {
        // Update existing skill stats
        this.updateSkillStats(
          (existing as any).id,
          candidate.success_rate > 0.5,
          candidate.avg_reward,
          candidate.avg_latency ?? 0
        );
        updated++;
      }
    }

    return { created, updated, patterns };
  }

  /**
   * Extract common patterns from successful episodes using ML-inspired analysis
   */
  private async extractPatternsFromEpisodes(episodeIds: number[]): Promise<{
    commonPatterns: string[];
    successIndicators: string[];
  }> {
    // Retrieve episodes with their outputs and critiques
    const episodes = this.db
      .prepare(
        `
      SELECT id, task, input, output, critique, reward, success, metadata
      FROM episodes
      WHERE id IN (${episodeIds.map(() => '?').join(',')})
      AND success = 1
    `
      )
      .all(...episodeIds) as any[];

    if (episodes.length === 0) {
      return { commonPatterns: [], successIndicators: [] };
    }

    const commonPatterns: string[] = [];
    const successIndicators: string[] = [];

    // Pattern 1: Analyze output text for common keywords and phrases
    const outputTexts = episodes.map((ep) => ep.output).filter(Boolean);

    if (outputTexts.length > 0) {
      const keywordFrequency = this.extractKeywordFrequency(outputTexts);
      const topKeywords = this.getTopKeywords(keywordFrequency, 5);

      if (topKeywords.length > 0) {
        commonPatterns.push(`Common techniques: ${topKeywords.join(', ')}`);
      }
    }

    // Pattern 2: Analyze critique patterns for successful strategies
    const critiques = episodes.map((ep) => ep.critique).filter(Boolean);

    if (critiques.length > 0) {
      const critiqueKeywords = this.extractKeywordFrequency(critiques);
      const topCritiquePatterns = this.getTopKeywords(critiqueKeywords, 3);

      if (topCritiquePatterns.length > 0) {
        successIndicators.push(...topCritiquePatterns);
      }
    }

    // Pattern 3: Analyze reward distribution
    const avgReward = episodes.reduce((sum, ep) => sum + ep.reward, 0) / episodes.length;
    const highRewardCount = episodes.filter((ep) => ep.reward > avgReward).length;
    const highRewardRatio = highRewardCount / episodes.length;

    if (highRewardRatio > 0.6) {
      successIndicators.push(
        `High consistency (${(highRewardRatio * 100).toFixed(0)}% above average)`
      );
    }

    // Pattern 4: Analyze metadata for common parameters
    const metadataPatterns = this.extractMetadataPatterns(episodes);
    if (metadataPatterns.length > 0) {
      commonPatterns.push(...metadataPatterns);
    }

    // Pattern 5: Temporal analysis - learning curve
    const learningTrend = this.analyzeLearningTrend(episodes);
    if (learningTrend) {
      successIndicators.push(learningTrend);
    }

    return { commonPatterns, successIndicators };
  }

  /**
   * Extract keyword frequency from text array using NLP-inspired techniques
   */
  private extractKeywordFrequency(texts: string[]): Map<string, number> {
    const frequency = new Map<string, number>();

    // Common stop words to filter out
    const stopWords = new Set([
      'the',
      'a',
      'an',
      'and',
      'or',
      'but',
      'in',
      'on',
      'at',
      'to',
      'for',
      'of',
      'with',
      'by',
      'from',
      'as',
      'is',
      'was',
      'are',
      'were',
      'been',
      'be',
      'have',
      'has',
      'had',
      'do',
      'does',
      'did',
      'will',
      'would',
      'should',
      'could',
      'may',
      'might',
      'must',
      'can',
      'this',
      'that',
      'these',
      'those',
    ]);

    for (const text of texts) {
      // Extract words (alphanumeric sequences)
      const words = text.toLowerCase().match(/\b[a-z0-9_-]+\b/g) || [];

      for (const word of words) {
        if (word.length > 3 && !stopWords.has(word)) {
          frequency.set(word, (frequency.get(word) || 0) + 1);
        }
      }
    }

    return frequency;
  }

  /**
   * Get top N keywords by frequency
   */
  private getTopKeywords(frequency: Map<string, number>, n: number): string[] {
    return Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .filter(([_, count]) => count >= 2) // Only keywords appearing at least twice
      .map(([word, _]) => word);
  }

  /**
   * Extract common patterns from episode metadata
   */
  private extractMetadataPatterns(episodes: any[]): string[] {
    const patterns: string[] = [];
    const metadataFields = new Map<string, Set<any>>();

    for (const episode of episodes) {
      if (episode.metadata) {
        try {
          const metadata =
            typeof episode.metadata === 'string' ? JSON.parse(episode.metadata) : episode.metadata;

          for (const [key, value] of Object.entries(metadata)) {
            if (!metadataFields.has(key)) {
              metadataFields.set(key, new Set());
            }
            metadataFields.get(key)!.add(value);
          }
        } catch (e) {
          // Skip invalid metadata
        }
      }
    }

    // Find fields with consistent values
    metadataFields.forEach((values, field) => {
      if (values.size === 1) {
        // All episodes have the same value for this field
        const value = Array.from(values)[0];
        patterns.push(`Consistent ${field}: ${value}`);
      }
    });

    return patterns;
  }

  /**
   * Analyze learning trend across episodes
   */
  private analyzeLearningTrend(episodes: any[]): string | null {
    if (episodes.length < 3) return null;

    // Sort by episode ID (temporal order)
    const sorted = [...episodes].sort((a, b) => a.id - b.id);

    const firstHalfReward =
      sorted.slice(0, Math.floor(sorted.length / 2)).reduce((sum, ep) => sum + ep.reward, 0) /
      Math.floor(sorted.length / 2);

    const secondHalfReward =
      sorted.slice(Math.floor(sorted.length / 2)).reduce((sum, ep) => sum + ep.reward, 0) /
      (sorted.length - Math.floor(sorted.length / 2));

    const improvement = ((secondHalfReward - firstHalfReward) / firstHalfReward) * 100;

    if (improvement > 10) {
      return `Strong learning curve (+${improvement.toFixed(0)}% improvement)`;
    } else if (improvement > 5) {
      return `Moderate learning curve (+${improvement.toFixed(0)}% improvement)`;
    } else if (Math.abs(improvement) < 5) {
      return `Stable performance (±${Math.abs(improvement).toFixed(0)}%)`;
    }

    return null;
  }

  /**
   * Calculate pattern confidence score based on sample size and success rate
   */
  private calculatePatternConfidence(sampleSize: number, successRate: number): number {
    // Confidence increases with sample size and success rate
    // Using a sigmoid-like function for smooth scaling
    const sampleFactor = Math.min(sampleSize / 10, 1.0); // Saturates at 10 samples
    const successFactor = successRate;

    return Math.min(sampleFactor * successFactor, 0.99);
  }

  /**
   * Prune underperforming skills
   * Invalidates cache on completion
   */
  pruneSkills(config: { minUses?: number; minSuccessRate?: number; maxAgeDays?: number }): number {
    const { minUses = 3, minSuccessRate = 0.4, maxAgeDays = 60 } = config;

    const stmt = this.db.prepare(`
      DELETE FROM skills
      WHERE uses < ?
        AND success_rate < ?
        AND created_at < strftime('%s', 'now') - ?
    `);

    const result = stmt.run(minUses, minSuccessRate, maxAgeDays * 86400);

    // Invalidate cache after pruning
    if (result.changes > 0) {
      this.queryCache.invalidateCategory('skills');
    }

    return result.changes;
  }

  /**
   * Get query cache statistics
   */
  getCacheStats() {
    return this.queryCache.getStatistics();
  }

  /**
   * Clear query cache
   */
  clearCache(): void {
    this.queryCache.clear();
  }

  /**
   * Prune expired cache entries
   */
  pruneCache(): number {
    return this.queryCache.pruneExpired();
  }

  /**
   * Warm cache with common skill queries
   */
  async warmCache(commonTasks: string[]): Promise<void> {
    await this.queryCache.warm(async (cache) => {
      // Pre-load common skill queries
      for (const task of commonTasks) {
        await this.retrieveSkills({ task, k: 5 });
      }
    });
  }

  // ========================================================================
  // Private Helper Methods
  // ========================================================================

  private getSkillById(id: number): Skill {
    const stmt = this.db.prepare('SELECT * FROM skills WHERE id = ?');
    const row = stmt.get(id);
    if (!row) throw new Error(`Skill ${id} not found`);
    return this.rowToSkill(row);
  }

  private rowToSkill(row: any): Skill {
    return {
      id: row.id,
      name: row.name,
      description: row.description,
      signature: safeJsonParse(row.signature, undefined) as Skill["signature"],
      code: row.code,
      successRate: row.success_rate,
      uses: row.uses,
      avgReward: row.avg_reward,
      avgLatencyMs: row.avg_latency_ms,
      createdFromEpisode: row.created_from_episode,
      metadata: safeJsonParse(row.metadata, undefined),
    };
  }

  private buildSkillText(skill: Skill): string {
    const parts = [skill.name];
    if (skill.description) parts.push(skill.description);
    parts.push(JSON.stringify(skill.signature));
    return parts.join('\n');
  }

  /**
   * Compute composite skill score from similarity and metadata
   * VectorBackend provides normalized similarity (0-1)
   */
  private computeSkillScore(skill: Skill & { similarity: number }): number {
    // Composite score: similarity * 0.4 + success_rate * 0.3 + (uses/1000) * 0.1 + avg_reward * 0.2
    const uses = skill.uses ?? 0;
    const avgReward = skill.avgReward ?? 0;

    return (
      skill.similarity * 0.4 +
      skill.successRate * 0.3 +
      Math.min(uses / 1000, 1.0) * 0.1 +
      avgReward * 0.2
    );
  }
}
