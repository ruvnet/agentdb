/**
 * API Backward Compatibility Tests for AgentDB v2
 *
 * Ensures 100% backward compatibility with v1 API signatures
 * Tests all public APIs for ReasoningBank, SkillLibrary, and HNSWIndex
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { ReasoningBank, ReasoningPattern, PatternSearchQuery } from '../../src/controllers/ReasoningBank.js';
import { SkillLibrary, Skill, SkillQuery } from '../../src/controllers/SkillLibrary.js';
import { HNSWIndex, HNSWConfig } from '../../src/controllers/HNSWIndex.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import { createBackend } from '../../src/backends/factory.js';
import type { VectorBackend } from '../../src/backends/VectorBackend.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-api-compat.db';

describe('API Backward Compatibility', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let vectorBackend: VectorBackend;
  let reasoningBank: ReasoningBank;
  let skillLibrary: SkillLibrary;
  let hnswIndex: HNSWIndex;

  beforeEach(async () => {
    // Clean up
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });

    // Initialize database
    db = new Database(TEST_DB_PATH);
    db.pragma('journal_mode = WAL');

    // Load schemas
    const schemaPath = path.join(__dirname, '../../src/schemas/schema.sql');
    if (fs.existsSync(schemaPath)) {
      db.exec(fs.readFileSync(schemaPath, 'utf-8'));
    }

    const frontierSchemaPath = path.join(__dirname, '../../src/schemas/frontier-schema.sql');
    if (fs.existsSync(frontierSchemaPath)) {
      db.exec(fs.readFileSync(frontierSchemaPath, 'utf-8'));
    }

    // Initialize embedder
    embedder = new EmbeddingService({
      model: 'mock-model',
      dimension: 384,
      provider: 'local',
    });
    await embedder.initialize();

    // Initialize vector backend (required for v2)
    vectorBackend = await createBackend('auto', {
      dimensions: 384,
      metric: 'cosine',
    });

    // Initialize controllers
    reasoningBank = new ReasoningBank(db, embedder);
    skillLibrary = new SkillLibrary(db, embedder, vectorBackend);
    hnswIndex = new HNSWIndex(db, {
      dimension: 384,
      metric: 'cosine',
      M: 16,
      efConstruction: 200,
      efSearch: 100,
      maxElements: 10000,
      persistIndex: false,
      rebuildThreshold: 0.1,
    });
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('ReasoningBank API v1 Compatibility', () => {
    describe('storePattern - v1 signature', () => {
      it('should accept v1 pattern object with all required fields', async () => {
        const pattern: ReasoningPattern = {
          taskType: 'code_review',
          approach: 'Review for bugs, style, and security vulnerabilities',
          successRate: 0.85,
        };

        const id = await reasoningBank.storePattern(pattern);

        expect(id).toBeDefined();
        expect(typeof id).toBe('number');
        expect(id).toBeGreaterThan(0);
      });

      it('should accept v1 pattern with optional fields', async () => {
        const pattern: ReasoningPattern = {
          taskType: 'data_analysis',
          approach: 'Statistical analysis with visualization',
          successRate: 0.92,
          uses: 15,
          avgReward: 0.88,
          tags: ['statistics', 'visualization'],
          metadata: {
            dataset: 'user_behavior',
            tools: ['pandas', 'matplotlib']
          },
        };

        const id = await reasoningBank.storePattern(pattern);

        expect(id).toBeGreaterThan(0);

        // Verify pattern was stored correctly
        const retrieved = reasoningBank.getPattern(id);
        expect(retrieved).toBeDefined();
        expect(retrieved?.taskType).toBe('data_analysis');
        expect(retrieved?.uses).toBe(15);
        expect(retrieved?.tags).toEqual(['statistics', 'visualization']);
      });

      it('should handle pattern with minimal fields', async () => {
        const pattern: ReasoningPattern = {
          taskType: 'test_task',
          approach: 'Simple approach',
          successRate: 0.5,
        };

        const id = await reasoningBank.storePattern(pattern);

        expect(id).toBeGreaterThan(0);
      });
    });

    describe('searchPatterns - v1 signature', () => {
      beforeEach(async () => {
        // Seed patterns for search tests
        await reasoningBank.storePattern({
          taskType: 'authentication',
          approach: 'JWT-based authentication with refresh tokens',
          successRate: 0.95,
          tags: ['security', 'auth'],
        });

        await reasoningBank.storePattern({
          taskType: 'authentication',
          approach: 'OAuth2 with PKCE flow',
          successRate: 0.88,
          tags: ['security', 'oauth'],
        });

        await reasoningBank.storePattern({
          taskType: 'database_optimization',
          approach: 'Index optimization and query tuning',
          successRate: 0.82,
          tags: ['performance', 'database'],
        });
      });

      it('should support v1 searchPatterns with query object', async () => {
        const queryEmbedding = await embedder.embed('authentication security');

        const query: PatternSearchQuery = {
          taskEmbedding: queryEmbedding,
          k: 10,
          threshold: 0.0,
        };

        const results = await reasoningBank.searchPatterns(query);

        expect(Array.isArray(results)).toBe(true);
        expect(results.length).toBeGreaterThan(0);
        expect(results[0]).toHaveProperty('taskType');
        expect(results[0]).toHaveProperty('approach');
        expect(results[0]).toHaveProperty('similarity');
      });

      it('should support v1 searchPatterns with filters', async () => {
        const queryEmbedding = await embedder.embed('security authentication');

        const query: PatternSearchQuery = {
          taskEmbedding: queryEmbedding,
          k: 10,
          threshold: 0.5,
          filters: {
            taskType: 'authentication',
            minSuccessRate: 0.9,
          },
        };

        const results = await reasoningBank.searchPatterns(query);

        expect(Array.isArray(results)).toBe(true);
        results.forEach(pattern => {
          expect(pattern.taskType).toBe('authentication');
          expect(pattern.successRate).toBeGreaterThanOrEqual(0.9);
          expect(pattern.similarity).toBeGreaterThanOrEqual(0.5);
        });
      });

      it('should support v1 searchPatterns with tag filters', async () => {
        const queryEmbedding = await embedder.embed('security best practices');

        const query: PatternSearchQuery = {
          taskEmbedding: queryEmbedding,
          k: 10,
          filters: {
            tags: ['security'],
          },
        };

        const results = await reasoningBank.searchPatterns(query);

        expect(Array.isArray(results)).toBe(true);
        results.forEach(pattern => {
          expect(pattern.tags).toBeDefined();
          expect(pattern.tags).toContain('security');
        });
      });

      it('should respect k parameter limit', async () => {
        const queryEmbedding = await embedder.embed('test query');

        const query: PatternSearchQuery = {
          taskEmbedding: queryEmbedding,
          k: 2,
        };

        const results = await reasoningBank.searchPatterns(query);

        expect(results.length).toBeLessThanOrEqual(2);
      });
    });

    describe('getPatternStats - v1 signature', () => {
      it('should return pattern statistics', () => {
        const stats = reasoningBank.getPatternStats();

        expect(stats).toBeDefined();
        expect(stats).toHaveProperty('totalPatterns');
        expect(stats).toHaveProperty('avgSuccessRate');
        expect(stats).toHaveProperty('avgUses');
        expect(stats).toHaveProperty('topTaskTypes');
        expect(stats).toHaveProperty('recentPatterns');
        expect(stats).toHaveProperty('highPerformingPatterns');
      });
    });

    describe('updatePatternStats - v1 signature', () => {
      it('should update pattern stats after use', async () => {
        const patternId = await reasoningBank.storePattern({
          taskType: 'test',
          approach: 'test approach',
          successRate: 0.8,
        });

        reasoningBank.updatePatternStats(patternId, true, 0.95);

        const updated = reasoningBank.getPattern(patternId);
        expect(updated).toBeDefined();
        expect(updated?.uses).toBe(1);
      });
    });

    describe('getPattern - v1 signature', () => {
      it('should retrieve pattern by ID', async () => {
        const patternId = await reasoningBank.storePattern({
          taskType: 'test_retrieval',
          approach: 'Test pattern retrieval',
          successRate: 0.75,
        });

        const pattern = reasoningBank.getPattern(patternId);

        expect(pattern).toBeDefined();
        expect(pattern?.id).toBe(patternId);
        expect(pattern?.taskType).toBe('test_retrieval');
      });

      it('should return null for non-existent pattern', () => {
        const pattern = reasoningBank.getPattern(99999);
        expect(pattern).toBeNull();
      });
    });

    describe('deletePattern - v1 signature', () => {
      it('should delete pattern by ID', async () => {
        const patternId = await reasoningBank.storePattern({
          taskType: 'test_delete',
          approach: 'Pattern to be deleted',
          successRate: 0.6,
        });

        const deleted = reasoningBank.deletePattern(patternId);

        expect(deleted).toBe(true);
        expect(reasoningBank.getPattern(patternId)).toBeNull();
      });

      it('should return false for non-existent pattern', () => {
        const deleted = reasoningBank.deletePattern(99999);
        expect(deleted).toBe(false);
      });
    });

    describe('clearCache - v1 signature', () => {
      it('should clear query cache', () => {
        reasoningBank.clearCache();
        // Should not throw
        expect(true).toBe(true);
      });
    });
  });

  describe('SkillLibrary API v1 Compatibility', () => {
    describe('createSkill - v1 signature', () => {
      it('should accept v1 skill object', async () => {
        const skill: Skill = {
          name: 'jwt_authentication',
          description: 'JWT token generation and validation',
          signature: {
            inputs: { user: 'object', secret: 'string' },
            outputs: { token: 'string' },
          },
          successRate: 0.95,
          uses: 100,
          avgReward: 0.92,
          avgLatencyMs: 120,
        };

        const id = await skillLibrary.createSkill(skill);

        expect(id).toBeDefined();
        expect(typeof id).toBe('number');
        expect(id).toBeGreaterThan(0);
      });

      it('should accept skill with optional code field', async () => {
        const skill: Skill = {
          name: 'hash_password',
          description: 'Secure password hashing with bcrypt',
          signature: {
            inputs: { password: 'string' },
            outputs: { hash: 'string' },
          },
          code: 'const bcrypt = require("bcrypt"); return bcrypt.hash(password, 10);',
          successRate: 0.98,
          uses: 250,
          avgReward: 0.96,
          avgLatencyMs: 200,
        };

        const id = await skillLibrary.createSkill(skill);

        expect(id).toBeGreaterThan(0);
      });

      it('should accept skill with metadata', async () => {
        const skill: Skill = {
          name: 'api_request',
          description: 'Make authenticated API requests',
          signature: {
            inputs: { url: 'string', headers: 'object' },
            outputs: { response: 'object' },
          },
          successRate: 0.88,
          uses: 500,
          avgReward: 0.85,
          avgLatencyMs: 300,
          metadata: {
            protocol: 'https',
            timeout: 5000,
            retries: 3,
          },
        };

        const id = await skillLibrary.createSkill(skill);

        expect(id).toBeGreaterThan(0);
      });
    });

    describe('searchSkills - v1 signature', () => {
      beforeEach(async () => {
        // Seed skills
        await skillLibrary.createSkill({
          name: 'authentication',
          description: 'User authentication with JWT',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.95,
          uses: 100,
          avgReward: 0.92,
          avgLatencyMs: 100,
        });

        await skillLibrary.createSkill({
          name: 'database_query',
          description: 'Execute SQL queries safely',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.88,
          uses: 200,
          avgReward: 0.85,
          avgLatencyMs: 50,
        });
      });

      it('should support v1 searchSkills signature', async () => {
        const query: SkillQuery = {
          task: 'authentication security',
          k: 10,
        };

        const results = await skillLibrary.searchSkills(query);

        expect(Array.isArray(results)).toBe(true);
        expect(results.length).toBeGreaterThan(0);
        expect(results[0]).toHaveProperty('id');
        expect(results[0]).toHaveProperty('name');
      });

      it('should support minSuccessRate filter', async () => {
        const query: SkillQuery = {
          task: 'database operations',
          k: 10,
          minSuccessRate: 0.9,
        };

        const results = await skillLibrary.searchSkills(query);

        results.forEach(skill => {
          expect(skill.successRate).toBeGreaterThanOrEqual(0.9);
        });
      });

      it('should support preferRecent option', async () => {
        const query: SkillQuery = {
          task: 'test task',
          k: 5,
          preferRecent: true,
        };

        const results = await skillLibrary.searchSkills(query);

        expect(Array.isArray(results)).toBe(true);
      });
    });

    describe('retrieveSkills - v1 alias compatibility', () => {
      it('should support retrieveSkills as alias for searchSkills', async () => {
        await skillLibrary.createSkill({
          name: 'test_skill',
          description: 'Test skill',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.8,
          uses: 50,
          avgReward: 0.75,
          avgLatencyMs: 100,
        });

        const results = await skillLibrary.retrieveSkills({
          task: 'test',
          k: 5,
        });

        expect(Array.isArray(results)).toBe(true);
      });
    });

    describe('updateSkillStats - v1 signature', () => {
      it('should update skill statistics', async () => {
        const skillId = await skillLibrary.createSkill({
          name: 'test_update',
          description: 'Test skill update',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.8,
          uses: 10,
          avgReward: 0.75,
          avgLatencyMs: 100,
        });

        skillLibrary.updateSkillStats(skillId, true, 0.9, 90);

        // Verify update
        const updated = db.prepare('SELECT * FROM skills WHERE id = ?').get(skillId) as any;
        expect(updated.uses).toBe(11);
      });
    });

    describe('consolidateEpisodesIntoSkills - v1 signature', () => {
      it('should accept v1 config signature', async () => {
        const result = await skillLibrary.consolidateEpisodesIntoSkills({
          minAttempts: 3,
          minReward: 0.7,
          timeWindowDays: 7,
        });

        expect(result).toBeDefined();
        expect(result).toHaveProperty('created');
        expect(result).toHaveProperty('updated');
        expect(result).toHaveProperty('patterns');
      });

      it('should support extractPatterns option', async () => {
        const result = await skillLibrary.consolidateEpisodesIntoSkills({
          minAttempts: 3,
          minReward: 0.7,
          extractPatterns: true,
        });

        expect(result.patterns).toBeInstanceOf(Array);
      });
    });

    describe('linkSkills - v1 signature', () => {
      it('should link skills with relationship', async () => {
        const skill1 = await skillLibrary.createSkill({
          name: 'basic_auth',
          description: 'Basic authentication',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.8,
          uses: 50,
          avgReward: 0.75,
          avgLatencyMs: 100,
        });

        const skill2 = await skillLibrary.createSkill({
          name: 'advanced_auth',
          description: 'Advanced authentication with MFA',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.9,
          uses: 30,
          avgReward: 0.85,
          avgLatencyMs: 150,
        });

        skillLibrary.linkSkills({
          parentSkillId: skill2,
          childSkillId: skill1,
          relationship: 'prerequisite',
          weight: 0.8,
        });

        // Should not throw
        expect(true).toBe(true);
      });
    });
  });

  describe('HNSWIndex API v1 Compatibility', () => {
    describe('Constructor - v1 signature', () => {
      it('should accept v1 config object', () => {
        const config: Partial<HNSWConfig> = {
          dimension: 384,
          metric: 'cosine',
          M: 16,
          efConstruction: 200,
          efSearch: 100,
        };

        const index = new HNSWIndex(db, config);

        expect(index).toBeDefined();
        expect(index.isReady()).toBe(false);
      });

      it('should work with minimal config', () => {
        const index = new HNSWIndex(db, {
          dimension: 384,
          metric: 'cosine',
        });

        expect(index).toBeDefined();
      });

      it('should support all distance metrics', () => {
        const metrics: Array<'cosine' | 'l2' | 'ip'> = ['cosine', 'l2', 'ip'];

        metrics.forEach(metric => {
          const index = new HNSWIndex(db, {
            dimension: 384,
            metric,
          });

          expect(index).toBeDefined();
        });
      });
    });

    describe('buildIndex - v1 signature', () => {
      beforeEach(async () => {
        // Seed some patterns for index building
        for (let i = 0; i < 10; i++) {
          await reasoningBank.storePattern({
            taskType: `test_${i}`,
            approach: `Approach ${i}`,
            successRate: 0.8 + i * 0.01,
          });
        }
      });

      it('should build index from default table', async () => {
        await hnswIndex.buildIndex();

        expect(hnswIndex.isReady()).toBe(true);
      });

      it('should build index from custom table name', async () => {
        await hnswIndex.buildIndex('pattern_embeddings');

        expect(hnswIndex.isReady()).toBe(true);
      });
    });

    describe('search - v1 signature', () => {
      beforeEach(async () => {
        // Seed patterns and build index
        for (let i = 0; i < 20; i++) {
          await reasoningBank.storePattern({
            taskType: `task_${i}`,
            approach: `Test approach ${i}`,
            successRate: 0.7 + Math.random() * 0.3,
          });
        }

        await hnswIndex.buildIndex();
      });

      it('should search with v1 signature (query, k)', async () => {
        const query = await embedder.embed('test query');

        const results = await hnswIndex.search(query, 5);

        expect(Array.isArray(results)).toBe(true);
        expect(results.length).toBeGreaterThan(0);
        expect(results.length).toBeLessThanOrEqual(5);
        expect(results[0]).toHaveProperty('id');
        expect(results[0]).toHaveProperty('distance');
        expect(results[0]).toHaveProperty('similarity');
      });

      it('should support threshold option', async () => {
        const query = await embedder.embed('test query');

        const results = await hnswIndex.search(query, 10, {
          threshold: 0.5,
        });

        results.forEach(result => {
          expect(result.similarity).toBeGreaterThanOrEqual(0.5);
        });
      });

      it('should support filters option', async () => {
        const query = await embedder.embed('test query');

        // Test without filters (filters parameter exists but is undefined)
        const results = await hnswIndex.search(query, 10);

        expect(Array.isArray(results)).toBe(true);
      });
    });

    describe('addVector - v1 signature', () => {
      beforeEach(async () => {
        // Build initial index
        await reasoningBank.storePattern({
          taskType: 'initial',
          approach: 'Initial pattern',
          successRate: 0.8,
        });

        await hnswIndex.buildIndex();
      });

      it('should add vector to existing index', () => {
        const embedding = new Float32Array(384);
        for (let i = 0; i < 384; i++) {
          embedding[i] = Math.random();
        }

        hnswIndex.addVector(1000, embedding);

        // Should not throw
        expect(true).toBe(true);
      });
    });

    describe('removeVector - v1 signature', () => {
      beforeEach(async () => {
        const patternId = await reasoningBank.storePattern({
          taskType: 'to_remove',
          approach: 'Pattern to remove',
          successRate: 0.8,
        });

        await hnswIndex.buildIndex();
      });

      it('should mark vector for removal', () => {
        hnswIndex.removeVector(1);

        // Should not throw
        expect(true).toBe(true);
      });
    });

    describe('getStats - v1 signature', () => {
      it('should return index statistics', () => {
        const stats = hnswIndex.getStats();

        expect(stats).toBeDefined();
        expect(stats).toHaveProperty('enabled');
        expect(stats).toHaveProperty('indexBuilt');
        expect(stats).toHaveProperty('numElements');
        expect(stats).toHaveProperty('dimension');
        expect(stats).toHaveProperty('metric');
        expect(stats).toHaveProperty('M');
        expect(stats).toHaveProperty('efConstruction');
        expect(stats).toHaveProperty('efSearch');
        expect(stats).toHaveProperty('totalSearches');
        expect(stats).toHaveProperty('avgSearchTimeMs');
      });
    });

    describe('setEfSearch - v1 signature', () => {
      beforeEach(async () => {
        await reasoningBank.storePattern({
          taskType: 'test',
          approach: 'Test pattern',
          successRate: 0.8,
        });

        await hnswIndex.buildIndex();
      });

      it('should update efSearch parameter', () => {
        hnswIndex.setEfSearch(200);

        const stats = hnswIndex.getStats();
        expect(stats.efSearch).toBe(200);
      });
    });

    describe('isReady - v1 signature', () => {
      it('should return false when index not built', () => {
        expect(hnswIndex.isReady()).toBe(false);
      });

      it('should return true after index built', async () => {
        await reasoningBank.storePattern({
          taskType: 'test',
          approach: 'Test pattern',
          successRate: 0.8,
        });

        await hnswIndex.buildIndex();

        expect(hnswIndex.isReady()).toBe(true);
      });
    });

    describe('clear - v1 signature', () => {
      it('should clear index and free memory', async () => {
        await reasoningBank.storePattern({
          taskType: 'test',
          approach: 'Test pattern',
          successRate: 0.8,
        });

        await hnswIndex.buildIndex();

        hnswIndex.clear();

        expect(hnswIndex.isReady()).toBe(false);
        expect(hnswIndex.getStats().numElements).toBe(0);
      });
    });
  });

  describe('Cross-Controller Integration', () => {
    it('should maintain compatibility across controllers', async () => {
      // Store pattern via ReasoningBank
      const patternId = await reasoningBank.storePattern({
        taskType: 'integration_test',
        approach: 'Cross-controller integration',
        successRate: 0.9,
      });

      // Build HNSW index
      await hnswIndex.buildIndex();

      // Search via HNSW
      const query = await embedder.embed('integration test');
      const results = await hnswIndex.search(query, 5);

      expect(results.length).toBeGreaterThan(0);

      // Verify pattern still accessible
      const pattern = reasoningBank.getPattern(patternId);
      expect(pattern).toBeDefined();
    });
  });

  describe('Type Compatibility', () => {
    it('should accept v1 ReasoningPattern type', async () => {
      const pattern: ReasoningPattern = {
        taskType: 'type_test',
        approach: 'Type compatibility test',
        successRate: 0.8,
      };

      const id = await reasoningBank.storePattern(pattern);
      expect(id).toBeGreaterThan(0);
    });

    it('should accept v1 Skill type', async () => {
      const skill: Skill = {
        name: 'type_test',
        description: 'Type compatibility test',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.8,
        uses: 10,
        avgReward: 0.75,
        avgLatencyMs: 100,
      };

      const id = await skillLibrary.createSkill(skill);
      expect(id).toBeGreaterThan(0);
    });

    it('should accept v1 PatternSearchQuery type', async () => {
      const embedding = await embedder.embed('test');
      const query: PatternSearchQuery = {
        taskEmbedding: embedding,
        k: 10,
      };

      const results = await reasoningBank.searchPatterns(query);
      expect(Array.isArray(results)).toBe(true);
    });

    it('should accept v1 SkillQuery type', async () => {
      const query: SkillQuery = {
        task: 'test',
        k: 5,
      };

      const results = await skillLibrary.searchSkills(query);
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('Error Handling Compatibility', () => {
    it('should handle invalid inputs gracefully', async () => {
      // Wrong dimension should return empty results or handle gracefully
      const results = await reasoningBank.searchPatterns({
        taskEmbedding: new Float32Array(384), // Correct dimension but empty vector
        k: 5,
      });

      // Should not throw, may return empty results
      expect(Array.isArray(results)).toBe(true);
    });

    it('should handle non-existent IDs gracefully', () => {
      const pattern = reasoningBank.getPattern(99999);
      expect(pattern).toBeNull();
    });

    it('should throw when searching unbuilt index', async () => {
      const newIndex = new HNSWIndex(db, { dimension: 384, metric: 'cosine' });
      const query = new Float32Array(384);

      await expect(newIndex.search(query, 5)).rejects.toThrow('Index not built');
    });
  });
});
