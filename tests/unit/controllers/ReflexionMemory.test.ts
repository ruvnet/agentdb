/**
 * Unit Tests for ReflexionMemory Controller
 *
 * Tests episodic replay memory, self-critique storage, and similarity-based retrieval
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { ReflexionMemory, Episode } from '../../../src/controllers/ReflexionMemory.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-reflexion.db';

describe('ReflexionMemory', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let reflexion: ReflexionMemory;

  beforeEach(async () => {
    // Clean up
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });

    // Initialize
    db = new Database(TEST_DB_PATH);
    db.pragma('journal_mode = WAL');

    // Load schemas
    const schemaPath = path.join(__dirname, '../../../src/schemas/schema.sql');
    if (fs.existsSync(schemaPath)) {
      db.exec(fs.readFileSync(schemaPath, 'utf-8'));
    }

    const frontierSchemaPath = path.join(__dirname, '../../../src/schemas/frontier-schema.sql');
    if (fs.existsSync(frontierSchemaPath)) {
      db.exec(fs.readFileSync(frontierSchemaPath, 'utf-8'));
    }

    embedder = new EmbeddingService({
      model: 'mock-model',
      dimension: 384,
      provider: 'local',
    });
    await embedder.initialize();

    reflexion = new ReflexionMemory(db, embedder);
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('storeEpisode', () => {
    it('should store episode with all fields', async () => {
      const episode: Episode = {
        sessionId: 'session-1',
        task: 'implement_auth',
        input: 'Need OAuth2',
        output: 'Implemented OAuth2 with PKCE',
        critique: 'Works well, good security',
        reward: 0.95,
        success: true,
        latencyMs: 1200,
        tokensUsed: 500,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
      expect(typeof episodeId).toBe('number');
    });

    it('should store episode with minimal fields', async () => {
      const episode: Episode = {
        sessionId: 'session-2',
        task: 'fix_bug',
        reward: 0.7,
        success: true,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should store failed episode with critique', async () => {
      const episode: Episode = {
        sessionId: 'session-3',
        task: 'implement_cache',
        critique: 'Redis timeout issues - need retry logic',
        reward: 0.2,
        success: false,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should generate and store embeddings', async () => {
      const episode: Episode = {
        sessionId: 'session-4',
        task: 'test task with embedding',
        reward: 0.8,
        success: true,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      // Verify embedding was stored
      const embedding = db.prepare('SELECT embedding FROM episode_embeddings WHERE episode_id = ?')
        .get(episodeId) as any;

      expect(embedding).toBeDefined();
      expect(embedding.embedding).toBeInstanceOf(Buffer);
    });
  });

  describe('retrieveRelevant', () => {
    beforeEach(async () => {
      // Seed test data
      const episodes: Episode[] = [
        {
          sessionId: 'seed-1',
          task: 'implement JWT authentication',
          reward: 0.95,
          success: true,
          critique: 'JWT with 24h expiry works well',
        },
        {
          sessionId: 'seed-2',
          task: 'fix OAuth2 timeout bug',
          reward: 0.88,
          success: true,
          critique: 'Added retry logic for token refresh',
        },
        {
          sessionId: 'seed-3',
          task: 'implement database caching',
          reward: 0.65,
          success: false,
          critique: 'Redis connection issues',
        },
        {
          sessionId: 'seed-4',
          task: 'add authentication middleware',
          reward: 0.92,
          success: true,
        },
      ];

      for (const ep of episodes) {
        await reflexion.storeEpisode(ep);
      }
    });

    it('should retrieve relevant episodes by similarity', async () => {
      const episodes = await reflexion.retrieveRelevant({
        task: 'authentication issues',
        k: 5,
      });

      expect(episodes.length).toBeGreaterThan(0);
      expect(episodes.length).toBeLessThanOrEqual(5);
      expect(episodes[0]).toHaveProperty('id');
      expect(episodes[0]).toHaveProperty('task');
      expect(episodes[0]).toHaveProperty('similarity');
      expect(episodes[0].similarity).toBeGreaterThanOrEqual(0);
      expect(episodes[0].similarity).toBeLessThanOrEqual(1);
    });

    it('should filter by minimum reward', async () => {
      const episodes = await reflexion.retrieveRelevant({
        task: 'authentication',
        k: 10,
        minReward: 0.9,
      });

      episodes.forEach(ep => {
        expect(ep.reward).toBeGreaterThanOrEqual(0.9);
      });
    });

    it('should filter for failures only', async () => {
      const episodes = await reflexion.retrieveRelevant({
        task: 'database caching',
        k: 10,
        onlyFailures: true,
      });

      episodes.forEach(ep => {
        expect(ep.success).toBe(false);
      });
    });

    it('should filter for successes only', async () => {
      const episodes = await reflexion.retrieveRelevant({
        task: 'authentication',
        k: 10,
        onlySuccesses: true,
      });

      episodes.forEach(ep => {
        expect(ep.success).toBe(true);
      });
    });

    it('should return episodes sorted by similarity when hybrid is off', async () => {
      // The default fuses BM25 with the vector ranking (ADR-016); hybrid: false is the pure-vector order.
      const episodes = await reflexion.retrieveRelevant({
        task: 'authentication security',
        k: 5,
        hybrid: false,
      });

      // Verify descending similarity order
      for (let i = 0; i < episodes.length - 1; i++) {
        expect(episodes[i].similarity || 0).toBeGreaterThanOrEqual(episodes[i + 1].similarity || 0);
      }
    });

    it('should respect k limit', async () => {
      const episodes = await reflexion.retrieveRelevant({
        task: 'authentication',
        k: 2,
      });

      expect(episodes.length).toBeLessThanOrEqual(2);
    });
  });

  describe('getTaskStats', () => {
    beforeEach(async () => {
      // Seed data for specific task
      for (let i = 0; i < 10; i++) {
        await reflexion.storeEpisode({
          sessionId: `stats-${i}`,
          task: 'implement_api',
          reward: 0.7 + i * 0.02,
          success: i > 3,
          latencyMs: 100 + i * 10,
        });
      }
    });

    it('should calculate task statistics', () => {
      const stats = reflexion.getTaskStats('implement_api');

      expect(stats.totalAttempts).toBe(10);
      expect(stats.successRate).toBeGreaterThan(0);
      expect(stats.successRate).toBeLessThanOrEqual(1);
      expect(stats.avgReward).toBeGreaterThan(0);
      expect(stats.avgLatency).toBeGreaterThan(0);
    });

    it('should return zero stats for non-existent task', () => {
      const stats = reflexion.getTaskStats('non_existent_task');

      expect(stats.totalAttempts).toBe(0);
      expect(stats.successRate).toBe(0);
      expect(stats.avgReward).toBe(0);
    });
  });

  describe('getCritiqueSummary', () => {
    beforeEach(async () => {
      await reflexion.storeEpisode({
        sessionId: 'critique-1',
        task: 'implement caching',
        critique: 'Redis timeout - need better error handling',
        reward: 0.3,
        success: false,
      });

      await reflexion.storeEpisode({
        sessionId: 'critique-2',
        task: 'cache layer implementation',
        critique: 'Connection pool exhausted - increase pool size',
        reward: 0.4,
        success: false,
      });
    });

    it('should generate critique summary from failures', async () => {
      const summary = await reflexion.getCritiqueSummary({
        task: 'caching',
        onlyFailures: true,
      });

      expect(summary).toBeTruthy();
      expect(summary).toContain('failures');
      expect(summary).toContain('lessons learned');
    });

    it('should return message when no failures found', async () => {
      const summary = await reflexion.getCritiqueSummary({
        task: 'completely unknown task',
        onlyFailures: true,
      });

      expect(summary).toContain('No prior failures');
    });
  });

  describe('getSuccessStrategies', () => {
    beforeEach(async () => {
      await reflexion.storeEpisode({
        sessionId: 'success-1',
        task: 'implement OAuth',
        output: 'Used PKCE flow with refresh tokens',
        reward: 0.95,
        success: true,
      });

      await reflexion.storeEpisode({
        sessionId: 'success-2',
        task: 'OAuth implementation',
        output: 'Implemented OAuth2 with state parameter for CSRF protection',
        reward: 0.92,
        success: true,
      });
    });

    it('should generate success strategies summary', async () => {
      const strategies = await reflexion.getSuccessStrategies({
        task: 'OAuth',
        minReward: 0.8,
      });

      expect(strategies).toBeTruthy();
      expect(strategies).toContain('Successful strategies');
    });

    it('should return message when no successes found', async () => {
      const strategies = await reflexion.getSuccessStrategies({
        task: 'unknown task',
        minReward: 0.9,
      });

      expect(strategies).toContain('No successful strategies');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty strings gracefully', async () => {
      const episode: Episode = {
        sessionId: '',
        task: '',
        reward: 0.5,
        success: true,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should handle Unicode characters', async () => {
      const episode: Episode = {
        sessionId: '测试会话',
        task: 'Implement 🚀 authentication 🔐',
        critique: 'Works perfectly! ✅',
        reward: 0.9,
        success: true,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should handle reward boundaries', async () => {
      const ep1: Episode = {
        sessionId: 's1',
        task: 't1',
        reward: 0.0,
        success: false,
      };

      const ep2: Episode = {
        sessionId: 's2',
        task: 't2',
        reward: 1.0,
        success: true,
      };

      const id1 = await reflexion.storeEpisode(ep1);
      const id2 = await reflexion.storeEpisode(ep2);

      expect(id1).toBeGreaterThan(0);
      expect(id2).toBeGreaterThan(0);
    });

    it('should handle null/undefined optional fields', async () => {
      const episode: Episode = {
        sessionId: 'test',
        task: 'test task',
        reward: 0.8,
        success: true,
        input: undefined,
        output: undefined,
        critique: undefined,
        latencyMs: undefined,
        tokensUsed: undefined,
      };

      const episodeId = await reflexion.storeEpisode(episode);

      expect(episodeId).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('should store 100 episodes efficiently', async () => {
      const startTime = Date.now();

      for (let i = 0; i < 100; i++) {
        await reflexion.storeEpisode({
          sessionId: `perf-${i}`,
          task: `task ${i}`,
          reward: Math.random(),
          success: Math.random() > 0.5,
        });
      }

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(5000); // Should complete in less than 5 seconds
    }, 10000);

    it('should retrieve episodes efficiently', async () => {
      // Seed data
      for (let i = 0; i < 50; i++) {
        await reflexion.storeEpisode({
          sessionId: `search-${i}`,
          task: `task ${i % 10}`,
          reward: Math.random(),
          success: true,
        });
      }

      const startTime = Date.now();

      await reflexion.retrieveRelevant({
        task: 'task search',
        k: 10,
      });

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(200); // Should complete in less than 200ms
    });
  });
});
