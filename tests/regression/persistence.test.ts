/**
 * Persistence and Data Migration Tests for AgentDB v2
 *
 * Ensures data survives restarts and is compatible between backends
 * Tests save/load cycles, cross-session persistence, and v1 database compatibility
 */

import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from 'vitest';
import Database from 'better-sqlite3';
import { ReasoningBank, ReasoningPattern } from '../../src/controllers/ReasoningBank.js';
import { SkillLibrary, Skill } from '../../src/controllers/SkillLibrary.js';
import { ReflexionMemory, Episode } from '../../src/controllers/ReflexionMemory.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import { createBackend } from '../../src/backends/factory.js';
import type { VectorBackend } from '../../src/backends/VectorBackend.js';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const TEST_DB_PATH = './tests/fixtures/test-persistence.db';
const TEMP_DIR = `./tests/fixtures/temp-${Date.now()}`;

describe('Persistence and Data Migration', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let vectorBackend: VectorBackend;

  beforeAll(() => {
    // Create temp directory
    if (!fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
    }
  });

  afterAll(() => {
    // Clean up temp directory
    if (fs.existsSync(TEMP_DIR)) {
      fs.rmSync(TEMP_DIR, { recursive: true, force: true });
    }
  });

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
  });

  afterEach(() => {
    if (db) {
      db.close();
    }
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('ReasoningBank Persistence', () => {
    it('should persist patterns across database restarts', async () => {
      // First session: store patterns
      let reasoningBank = new ReasoningBank(db, embedder);

      const patterns: ReasoningPattern[] = [
        {
          taskType: 'authentication',
          approach: 'JWT with refresh tokens',
          successRate: 0.95,
          tags: ['security', 'auth'],
        },
        {
          taskType: 'database_optimization',
          approach: 'Index-based query optimization',
          successRate: 0.88,
          tags: ['performance', 'database'],
        },
        {
          taskType: 'api_design',
          approach: 'RESTful API with versioning',
          successRate: 0.92,
          tags: ['api', 'design'],
        },
      ];

      const patternIds: number[] = [];
      for (const pattern of patterns) {
        const id = await reasoningBank.storePattern(pattern);
        patternIds.push(id);
      }

      // Close database
      db.close();

      // Second session: reopen and verify
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reasoningBank = new ReasoningBank(db, embedder);

      // Verify all patterns exist
      for (let i = 0; i < patternIds.length; i++) {
        const pattern = reasoningBank.getPattern(patternIds[i]);
        expect(pattern).toBeDefined();
        expect(pattern?.taskType).toBe(patterns[i].taskType);
        expect(pattern?.approach).toBe(patterns[i].approach);
        expect(pattern?.successRate).toBe(patterns[i].successRate);
      }

      // Verify stats
      const stats = reasoningBank.getPatternStats();
      expect(stats.totalPatterns).toBe(patterns.length);
    });

    it('should preserve embeddings across sessions', async () => {
      // First session: store pattern with embedding
      let reasoningBank = new ReasoningBank(db, embedder);

      const patternId = await reasoningBank.storePattern({
        taskType: 'embedding_test',
        approach: 'Test embedding persistence',
        successRate: 0.85,
      });

      // Retrieve embedding
      const firstPattern = reasoningBank.getPattern(patternId);
      expect(firstPattern?.embedding).toBeDefined();
      const firstEmbedding = firstPattern!.embedding!;

      db.close();

      // Second session: verify embedding
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reasoningBank = new ReasoningBank(db, embedder);

      const secondPattern = reasoningBank.getPattern(patternId);
      expect(secondPattern?.embedding).toBeDefined();
      const secondEmbedding = secondPattern!.embedding!;

      // Embeddings should be identical
      expect(firstEmbedding.length).toBe(secondEmbedding.length);
      for (let i = 0; i < firstEmbedding.length; i++) {
        expect(Math.abs(firstEmbedding[i] - secondEmbedding[i])).toBeLessThan(0.0001);
      }
    });

    it('should maintain pattern statistics across restarts', async () => {
      // First session: create and update pattern
      let reasoningBank = new ReasoningBank(db, embedder);

      const patternId = await reasoningBank.storePattern({
        taskType: 'stats_test',
        approach: 'Test statistics persistence',
        successRate: 0.8,
      });

      // Update stats multiple times
      for (let i = 0; i < 10; i++) {
        reasoningBank.updatePatternStats(patternId, true, 0.9);
      }

      const firstPattern = reasoningBank.getPattern(patternId);
      const firstUses = firstPattern!.uses;
      const firstSuccessRate = firstPattern!.successRate;

      db.close();

      // Second session: verify stats
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reasoningBank = new ReasoningBank(db, embedder);

      const secondPattern = reasoningBank.getPattern(patternId);
      expect(secondPattern!.uses).toBe(firstUses);
      expect(secondPattern!.successRate).toBeCloseTo(firstSuccessRate!, 5);
    });

    it('should handle large pattern datasets', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Store 100 patterns
      const patternCount = 100;
      for (let i = 0; i < patternCount; i++) {
        await reasoningBank.storePattern({
          taskType: `task_${i}`,
          approach: `Approach for task ${i}`,
          successRate: 0.7 + Math.random() * 0.3,
          tags: [`tag_${i % 5}`],
        });
      }

      db.close();

      // Reopen and verify count
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      const newReasoningBank = new ReasoningBank(db, embedder);
      const stats = newReasoningBank.getPatternStats();

      expect(stats.totalPatterns).toBe(patternCount);
    }, 30000);
  });

  describe('SkillLibrary Persistence', () => {
    it('should persist skills across database restarts', async () => {
      // First session: store skills
      let skillLibrary = new SkillLibrary(db, embedder, vectorBackend);

      const skills: Skill[] = [
        {
          name: 'jwt_auth',
          description: 'JWT authentication',
          signature: { inputs: { user: 'object' }, outputs: { token: 'string' } },
          successRate: 0.95,
          uses: 100,
          avgReward: 0.92,
          avgLatencyMs: 120,
        },
        {
          name: 'data_validation',
          description: 'Input data validation',
          signature: { inputs: { data: 'any' }, outputs: { valid: 'boolean' } },
          successRate: 0.88,
          uses: 200,
          avgReward: 0.85,
          avgLatencyMs: 50,
        },
      ];

      const skillIds: number[] = [];
      for (const skill of skills) {
        const id = await skillLibrary.createSkill(skill);
        skillIds.push(id);
      }

      db.close();

      // Second session: verify
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      skillLibrary = new SkillLibrary(db, embedder, vectorBackend);

      // Verify skills exist via search
      const results = await skillLibrary.searchSkills({
        task: 'authentication',
        k: 10,
      });

      expect(results.length).toBeGreaterThan(0);
    });

    it('should preserve skill relationships across sessions', async () => {
      // First session: create skills and link them
      let skillLibrary = new SkillLibrary(db, embedder, vectorBackend);

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
        description: 'Advanced authentication',
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

      db.close();

      // Second session: verify relationship
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      skillLibrary = new SkillLibrary(db, embedder, vectorBackend);

      const plan = skillLibrary.getSkillPlan(skill2);
      expect(plan.prerequisites.length).toBeGreaterThan(0);
      expect(plan.prerequisites[0].id).toBe(skill1);
    });

    it('should persist skill metadata correctly', async () => {
      // First session: create skill with complex metadata
      let skillLibrary = new SkillLibrary(db, embedder, vectorBackend);

      const metadata = {
        version: '2.1.0',
        author: 'test_author',
        dependencies: ['dep1', 'dep2'],
        config: {
          timeout: 5000,
          retries: 3,
        },
      };

      const skillId = await skillLibrary.createSkill({
        name: 'metadata_test',
        description: 'Test metadata persistence',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.85,
        uses: 25,
        avgReward: 0.80,
        avgLatencyMs: 125,
        metadata,
      });

      db.close();

      // Second session: verify metadata
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      const row = db.prepare('SELECT * FROM skills WHERE id = ?').get(skillId) as any;
      const retrievedMetadata = JSON.parse(row.metadata);

      expect(retrievedMetadata).toEqual(metadata);
    });
  });

  describe('ReflexionMemory Persistence', () => {
    it('should persist episodes across restarts', async () => {
      // First session: store episodes
      let reflexion = new ReflexionMemory(db, embedder);

      const episodes: Episode[] = [
        {
          sessionId: 'session_1',
          task: 'task_1',
          input: 'input data 1',
          output: 'output data 1',
          critique: 'critique 1',
          reward: 0.85,
          success: true,
          latencyMs: 100,
        },
        {
          sessionId: 'session_1',
          task: 'task_2',
          input: 'input data 2',
          output: 'output data 2',
          critique: 'critique 2',
          reward: 0.92,
          success: true,
          latencyMs: 150,
        },
      ];

      for (const episode of episodes) {
        await reflexion.storeEpisode(episode);
      }

      db.close();

      // Second session: verify
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reflexion = new ReflexionMemory(db, embedder);

      const retrieved = await reflexion.getRecentEpisodes('session_1', 10);
      expect(retrieved.length).toBe(episodes.length);
    });

    it('should maintain episode trajectory history', async () => {
      // First session: create trajectory
      let reflexion = new ReflexionMemory(db, embedder);

      const sessionId = `traj_${Date.now()}`;

      // Store 5 episodes in sequence
      for (let i = 0; i < 5; i++) {
        await reflexion.storeEpisode({
          sessionId,
          task: 'sequential_task',
          output: `output_${i}`,
          reward: 0.7 + i * 0.05,
          success: true,
        });
      }

      db.close();

      // Second session: verify trajectory
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reflexion = new ReflexionMemory(db, embedder);

      const episodes = await reflexion.getRecentEpisodes(sessionId, 10);
      expect(episodes.length).toBe(5);

      // Verify order (most recent first)
      for (let i = 0; i < episodes.length - 1; i++) {
        expect(episodes[i].reward).toBeGreaterThanOrEqual(episodes[i + 1].reward!);
      }
    });
  });

  describe('Database File Integrity', () => {
    it('should handle database file corruption gracefully', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      await reasoningBank.storePattern({
        taskType: 'test',
        approach: 'test',
        successRate: 0.8,
      });

      db.close();

      // Corrupt the database file header ("SQLite format 3\0" magic)
      const dbBuffer = fs.readFileSync(TEST_DB_PATH);
      const corrupted = Buffer.from(dbBuffer);
      corrupted.write('CORRUPTED', 0);
      fs.writeFileSync(TEST_DB_PATH, corrupted);

      // SQLite opens lazily, so the corruption surfaces on first access:
      // it must raise a catchable error rather than crash or return garbage.
      db = new Database(TEST_DB_PATH);
      expect(() => {
        db.prepare('SELECT COUNT(*) FROM reasoning_patterns').get();
      }).toThrow(/not a database|malformed/i);
    });

    it('should verify database schema integrity', () => {
      // reasoning_patterns / pattern_embeddings are owned by ReasoningBank,
      // which creates them on construction (they are not in schema.sql).
      new ReasoningBank(db, embedder);

      const tables = db.prepare(`
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
      `).all() as any[];

      const tableNames = tables.map(t => t.name);

      // Essential tables must exist
      expect(tableNames).toContain('reasoning_patterns');
      expect(tableNames).toContain('pattern_embeddings');
      expect(tableNames).toContain('skills');
      expect(tableNames).toContain('skill_embeddings');
      expect(tableNames).toContain('episodes');
    });

    it('should maintain indexes after restart', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Store patterns
      for (let i = 0; i < 50; i++) {
        await reasoningBank.storePattern({
          taskType: `task_${i % 5}`,
          approach: `approach_${i}`,
          successRate: 0.7 + Math.random() * 0.3,
        });
      }

      db.close();

      // Reopen
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      // Verify indexes exist
      const indexes = db.prepare(`
        SELECT name FROM sqlite_master
        WHERE type='index'
      `).all() as any[];

      const indexNames = indexes.map(i => i.name);
      expect(indexNames.length).toBeGreaterThan(0);
    });
  });

  describe('WAL Mode Persistence', () => {
    it('should maintain data consistency with WAL mode', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Verify WAL mode is active
      const walMode = db.pragma('journal_mode', { simple: true });
      expect(walMode).toBe('wal');

      // Store pattern
      const patternId = await reasoningBank.storePattern({
        taskType: 'wal_test',
        approach: 'WAL mode test',
        successRate: 0.85,
      });

      // Force checkpoint
      db.pragma('wal_checkpoint(FULL)');

      // Verify pattern persists
      const pattern = reasoningBank.getPattern(patternId);
      expect(pattern).toBeDefined();
      expect(pattern?.taskType).toBe('wal_test');
    });

    it('should handle concurrent access in WAL mode', async () => {
      // This test verifies WAL allows concurrent reads
      const reasoningBank = new ReasoningBank(db, embedder);

      await reasoningBank.storePattern({
        taskType: 'concurrent_test',
        approach: 'Concurrent access test',
        successRate: 0.8,
      });

      // Open second connection (WAL allows this)
      const db2 = new Database(TEST_DB_PATH, { readonly: true });

      const count = db2.prepare('SELECT COUNT(*) as count FROM reasoning_patterns').get() as any;
      expect(count.count).toBeGreaterThan(0);

      db2.close();
    });
  });

  describe('Cross-Session State Management', () => {
    it('should maintain cache invalidation across sessions', async () => {
      // First session: populate cache
      let reasoningBank = new ReasoningBank(db, embedder);

      await reasoningBank.storePattern({
        taskType: 'cache_test',
        approach: 'Cache invalidation test',
        successRate: 0.8,
      });

      // Trigger stats (which caches results)
      const stats1 = reasoningBank.getPatternStats();

      db.close();

      // Second session: cache should be empty
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reasoningBank = new ReasoningBank(db, embedder);

      // Add new pattern
      await reasoningBank.storePattern({
        taskType: 'cache_test_2',
        approach: 'Second pattern',
        successRate: 0.85,
      });

      const stats2 = reasoningBank.getPatternStats();

      // Stats should reflect new pattern
      expect(stats2.totalPatterns).toBeGreaterThan(stats1.totalPatterns);
    });
  });

  describe('Data Migration Scenarios', () => {
    it('should handle empty database gracefully', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);
      const skillLibrary = new SkillLibrary(db, embedder, vectorBackend);
      const reflexion = new ReflexionMemory(db, embedder);

      // All stats should handle empty database
      const patternStats = reasoningBank.getPatternStats();
      expect(patternStats.totalPatterns).toBe(0);

      const episodes = await reflexion.getRecentEpisodes('empty_session', 10);
      expect(episodes.length).toBe(0);

      const skills = await skillLibrary.searchSkills({ task: 'anything', k: 10 });
      expect(skills.length).toBe(0);
    });

    it('should handle incremental data additions', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Add patterns incrementally
      const counts: number[] = [];
      for (let batch = 0; batch < 5; batch++) {
        for (let i = 0; i < 10; i++) {
          await reasoningBank.storePattern({
            taskType: `batch_${batch}`,
            approach: `pattern_${i}`,
            successRate: 0.8,
          });
        }

        const stats = reasoningBank.getPatternStats();
        counts.push(stats.totalPatterns);
      }

      // Verify incremental growth
      for (let i = 1; i < counts.length; i++) {
        expect(counts[i]).toBeGreaterThan(counts[i - 1]);
      }

      expect(counts[counts.length - 1]).toBe(50);
    });

    it('should handle data deletion and recreation', async () => {
      let reasoningBank = new ReasoningBank(db, embedder);

      // Create and delete pattern
      const patternId = await reasoningBank.storePattern({
        taskType: 'delete_test',
        approach: 'Pattern to delete',
        successRate: 0.8,
      });

      const deleted = reasoningBank.deletePattern(patternId);
      expect(deleted).toBe(true);

      db.close();

      // Reopen and verify deletion persists
      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      reasoningBank = new ReasoningBank(db, embedder);

      const pattern = reasoningBank.getPattern(patternId);
      expect(pattern).toBeNull();
    });
  });

  describe('Performance Under Persistence', () => {
    it('should maintain performance with large datasets', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Store 500 patterns
      for (let i = 0; i < 500; i++) {
        await reasoningBank.storePattern({
          taskType: `perf_task_${i % 10}`,
          approach: `Approach ${i}`,
          successRate: 0.7 + Math.random() * 0.3,
        });
      }

      db.close();

      // Measure reload time
      const startTime = Date.now();

      db = new Database(TEST_DB_PATH);
      db.pragma('journal_mode = WAL');

      const reloadTime = Date.now() - startTime;

      // Database should reload quickly
      expect(reloadTime).toBeLessThan(1000); // Less than 1 second

      // Verify all data accessible
      const newReasoningBank = new ReasoningBank(db, embedder);
      const stats = newReasoningBank.getPatternStats();

      expect(stats.totalPatterns).toBe(500);
    }, 60000);

    it('should handle checkpoint operations efficiently', async () => {
      const reasoningBank = new ReasoningBank(db, embedder);

      // Store data
      for (let i = 0; i < 100; i++) {
        await reasoningBank.storePattern({
          taskType: 'checkpoint_test',
          approach: `Pattern ${i}`,
          successRate: 0.8,
        });
      }

      // Measure checkpoint time
      const startTime = Date.now();
      db.pragma('wal_checkpoint(FULL)');
      const checkpointTime = Date.now() - startTime;

      // Checkpoint should be fast
      expect(checkpointTime).toBeLessThan(500); // Less than 500ms
    });
  });
});
