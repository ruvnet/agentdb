/**
 * @test Attention Mechanism Regression Tests
 * @description Ensure attention support doesn't break existing AgentDB functionality
 * @coverage
 *   - MemoryController behavior with attention disabled vs enabled
 *   - Existing AgentDB controllers (reflexion, skills) unaffected
 *   - MemoryController API stability
 *   - Performance regression
 *
 * Attention lives in MemoryController (enableAttention + retrieveWithAttention,
 * plus getSelf/Cross/MultiHeadAttentionController). AgentDB's getController()
 * only exposes 'memory'|'reflexion' (ReflexionMemory), 'skills' and 'causal';
 * it has never registered attention controllers, listControllers(), query(),
 * migrations or transactions, so those speculative cases were removed.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AgentDB } from '../../src/index';
import { MemoryController } from '../../src/controllers/MemoryController';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory';
import { SkillLibrary } from '../../src/controllers/SkillLibrary';

describe('Attention Mechanism Regression Tests', () => {
  describe('Backward Compatibility - Attention Disabled', () => {
    let memoryController: MemoryController;

    beforeEach(() => {
      memoryController = new MemoryController(null, {
        namespace: 'regression-test',
        enableAttention: false
      });
    });

    it('should store and retrieve memories normally', async () => {
      const memory = {
        id: 'test-memory',
        content: 'Test content',
        embedding: [0.1, 0.2, 0.3]
      };

      await memoryController.store(memory);
      const retrieved = await memoryController.retrieve(memory.id);

      expect(retrieved).toBeDefined();
      expect(retrieved!.id).toBe(memory.id);
      expect(retrieved!.content).toBe(memory.content);
    });

    it('should perform vector search without attention', async () => {
      await memoryController.store({ id: 'm1', embedding: [0.1, 0.2, 0.3] });
      await memoryController.store({ id: 'm2', embedding: [0.4, 0.5, 0.6] });

      const results = await memoryController.search([0.1, 0.2, 0.3], { topK: 2 });

      expect(results).toHaveLength(2);
      expect(results[0]).toHaveProperty('score');
      expect(results[0]).not.toHaveProperty('attentionScore');
      expect(results[0].id).toBe('m1');
    });

    it('should fall back to similarity scores in retrieveWithAttention', async () => {
      await memoryController.store({ id: 'm1', embedding: [0.1, 0.2, 0.3] });

      const results = await memoryController.retrieveWithAttention([0.1, 0.2, 0.3]);

      expect(results).toHaveLength(1);
      expect(results[0].attentionScore).toBe(results[0].score);
    });
  });

  describe('Existing AgentDB controllers', () => {
    let db: AgentDB;

    beforeEach(async () => {
      db = new AgentDB({ dbPath: ':memory:', namespace: 'regression-test' });
      await db.initialize();
    });

    afterEach(async () => {
      await db.close();
    });

    it('should maintain existing ReflexionMemory functionality', async () => {
      const reflexion = db.getController('reflexion') as ReflexionMemory;

      await reflexion.storeEpisode({
        sessionId: 'session1',
        task: 'Complete task',
        reward: 0.8,
        success: true
      });
      const [retrieved] = await reflexion.getRecentEpisodes('session1', 1);

      expect(retrieved).toBeDefined();
      expect(retrieved.reward).toBe(0.8);
      expect(retrieved.success).toBe(true);
    });

    it('should maintain existing SkillLibrary functionality', async () => {
      const skillLib = db.getController('skills') as SkillLibrary;

      const skill = {
        name: 'test-skill',
        description: 'Test skill',
        code: 'function test() { return true; }',
        successRate: 0.9
      };

      const id = await skillLib.createSkill(skill);
      const retrieved = skillLib.getSkill(id);

      expect(retrieved).toBeDefined();
      expect(retrieved!.name).toBe(skill.name);
      expect(retrieved!.successRate).toBe(skill.successRate);
    });

    it('should not create attention-specific tables', () => {
      const tables = db.database
        .prepare("SELECT name FROM sqlite_master WHERE type='table'")
        .all() as Array<{ name: string }>;
      const tableNames = tables.map(t => t.name);

      expect(tableNames).toContain('episodes');
      expect(tableNames).not.toContain('attention_scores');
      expect(tableNames).not.toContain('attention_cache');
    });
  });

  describe('Attention Enabled', () => {
    let memoryController: MemoryController;

    beforeEach(() => {
      memoryController = new MemoryController(null, {
        namespace: 'feature-flag-test',
        enableAttention: true
      });
    });

    it('should expose all attention controllers', () => {
      expect(memoryController.getSelfAttentionController()).toBeDefined();
      expect(memoryController.getCrossAttentionController()).toBeDefined();
      expect(memoryController.getMultiHeadAttentionController()).toBeDefined();
    });

    it('should enhance memory retrieval with attention scores', async () => {
      await memoryController.store({ id: 'enhanced-mem', embedding: [0.1, 0.2, 0.3] });

      const results = await memoryController.retrieveWithAttention([0.1, 0.2, 0.3]);

      expect(results[0]).toHaveProperty('attentionScore');
      expect(results[0].attentionScore).toBeGreaterThanOrEqual(0);
    });

    it('should still support legacy search API', async () => {
      await memoryController.store({ id: 'legacy-search', embedding: [0.1, 0.2, 0.3] });

      const results = await memoryController.search([0.1, 0.2, 0.3]);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).not.toHaveProperty('attentionScore');
    });
  });

  describe('API Stability', () => {
    let memoryController: MemoryController;

    beforeEach(() => {
      memoryController = new MemoryController(null, { enableAttention: true });
    });

    it('should maintain stable AgentDB initialization API', () => {
      const db = new AgentDB({ dbPath: ':memory:' });
      expect(db.initialize).toBeInstanceOf(Function);
      expect(db.getController).toBeInstanceOf(Function);
      expect(db.close).toBeInstanceOf(Function);
    });

    it('should maintain stable MemoryController API', () => {
      expect(memoryController.store).toBeInstanceOf(Function);
      expect(memoryController.retrieve).toBeInstanceOf(Function);
      expect(memoryController.search).toBeInstanceOf(Function);
      expect(memoryController.delete).toBeInstanceOf(Function);
      expect(memoryController.update).toBeInstanceOf(Function);
      expect(memoryController.retrieveWithAttention).toBeInstanceOf(Function);
    });

    it('should not break existing method signatures', async () => {
      const memory = {
        id: 'api-test',
        content: 'Test',
        embedding: [0.1, 0.2, 0.3],
        metadata: { key: 'value' }
      };

      await expect(memoryController.store(memory)).resolves.toBeUndefined();
      await expect(memoryController.retrieve('api-test')).resolves.toBeDefined();
      await expect(memoryController.search([0.1, 0.2, 0.3])).resolves.toHaveLength(1);
    });

    it('should maintain backward-compatible search options', async () => {
      await memoryController.store({ id: 'compat-test', embedding: [0.1, 0.2, 0.3] });

      const results = await memoryController.search([0.1, 0.2, 0.3], {
        topK: 10,
        threshold: 0.5,
        filter: {}
      });

      expect(results.map(r => r.id)).toEqual(['compat-test']);
    });
  });

  describe('Performance Regression', () => {
    it('should not slow down initialization', async () => {
      const start = performance.now();
      const db = new AgentDB({ dbPath: ':memory:', enableAttention: true });
      await db.initialize();
      const duration = performance.now() - start;
      await db.close();

      expect(duration).toBeLessThan(1000);
    });

    it('should not impact memory storage performance', async () => {
      const memoryController = new MemoryController(null, { enableAttention: true });
      const start = performance.now();

      for (let i = 0; i < 100; i++) {
        await memoryController.store({
          id: `perf-test-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      expect(performance.now() - start).toBeLessThan(2000);
    });

    it('should not increase memory footprint significantly', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      const memoryController = new MemoryController(null, { enableAttention: true });

      for (let i = 0; i < 1000; i++) {
        await memoryController.store({
          id: `mem-footprint-${i}`,
          embedding: Array(128).fill(0).map(() => Math.random())
        });
      }

      global.gc && global.gc();
      const increase = process.memoryUsage().heapUsed - initialMemory;

      expect(increase).toBeLessThan(100 * 1024 * 1024);
    });
  });
});
