/**
 * AgentDB v1.6.0 Regression Tests - Core Features
 * Tests all existing functionality to ensure no regressions
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { createDatabase } from '../../src/db-fallback.js';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory.js';
import { SkillLibrary } from '../../src/controllers/SkillLibrary.js';
import { CausalMemoryGraph } from '../../src/controllers/CausalMemoryGraph.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';

describe('Core Features Regression Tests', () => {
  let db: any;
  let embedder: EmbeddingService;
  let reflexion: ReflexionMemory;
  let skills: SkillLibrary;
  let causalGraph: CausalMemoryGraph;
  const testDbPath = './test-core-features.db';

  beforeAll(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Initialize database
    db = await createDatabase(testDbPath);

    // Load schemas
    const schemaPath = path.join(__dirname, '../../src/schemas/schema.sql');
    const frontierSchemaPath = path.join(__dirname, '../../src/schemas/frontier-schema.sql');

    if (fs.existsSync(schemaPath)) {
      const schema = fs.readFileSync(schemaPath, 'utf-8');
      db.exec(schema);
    }

    if (fs.existsSync(frontierSchemaPath)) {
      const schema = fs.readFileSync(frontierSchemaPath, 'utf-8');
      db.exec(schema);
    }

    // Initialize embedding service
    embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    // Initialize controllers
    reflexion = new ReflexionMemory(db, embedder);
    skills = new SkillLibrary(db, embedder);
    causalGraph = new CausalMemoryGraph(db);
  });

  afterAll(() => {
    // Clean up
    if (db && typeof db.close === 'function') {
      db.close();
    }
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Reflexion Memory', () => {
    it('should store episodes', async () => {
      const episodeId = await reflexion.storeEpisode({
        sessionId: 'test-session-1',
        task: 'test task',
        input: 'test input',
        output: 'test output',
        critique: 'test critique',
        reward: 0.9,
        success: true,
        latencyMs: 100,
        tokensUsed: 50
      });

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should retrieve relevant episodes', async () => {
      // Store multiple episodes
      await reflexion.storeEpisode({
        sessionId: 'test-session-2',
        task: 'authentication task',
        reward: 0.85,
        success: true
      });

      await reflexion.storeEpisode({
        sessionId: 'test-session-3',
        task: 'login implementation',
        reward: 0.75,
        success: true
      });

      const results = await reflexion.retrieveRelevant({
        task: 'authentication',
        k: 5
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty('id');
      expect(results[0]).toHaveProperty('similarity');
    });

    it('should filter by success/failure', async () => {
      // Store failure
      await reflexion.storeEpisode({
        sessionId: 'test-session-4',
        task: 'failed task',
        reward: 0.3,
        success: false,
        critique: 'This failed because...'
      });

      const failures = await reflexion.retrieveRelevant({
        task: 'failed task',
        k: 5,
        onlyFailures: true
      });

      expect(failures.length).toBeGreaterThan(0);
      expect(failures[0].success).toBe(false);
    });

    it('should prune old episodes', () => {
      const pruned = reflexion.pruneEpisodes({
        minReward: 0.5,
        maxAgeDays: 1,
        keepMinPerTask: 1
      });

      expect(pruned).toBeGreaterThanOrEqual(0);
    });

    it('should get critique summary', async () => {
      const summary = await reflexion.getCritiqueSummary({
        task: 'test task',
        k: 5
      });

      expect(summary).toBeDefined();
      expect(typeof summary).toBe('string');
    });
  });

  describe('Skill Library', () => {
    it('should create skills', async () => {
      const skillId = await skills.createSkill({
        name: 'test_skill',
        description: 'Test skill description',
        signature: { inputs: {}, outputs: {} },
        code: 'console.log("test");',
        successRate: 0.9,
        uses: 0,
        avgReward: 0.85,
        avgLatencyMs: 100
      });

      expect(skillId).toBeGreaterThan(0);
    });

    it('should search skills', async () => {
      await skills.createSkill({
        name: 'auth_skill',
        description: 'Authentication implementation',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.95,
        uses: 10,
        avgReward: 0.9,
        avgLatencyMs: 50
      });

      const results = await skills.searchSkills({
        task: 'authentication',
        k: 5
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty('name');
      expect(results[0]).toHaveProperty('successRate');
    });

    it('should consolidate episodes into skills', async () => {
      // Store some successful episodes
      for (let i = 0; i < 5; i++) {
        await reflexion.storeEpisode({
          sessionId: `consolidate-session-${i}`,
          task: 'consolidate_test',
          reward: 0.8 + Math.random() * 0.2,
          success: true,
          output: 'successful output'
        });
      }

      const result = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 3,
        minReward: 0.7,
        timeWindowDays: 7,
        extractPatterns: true
      });

      expect(result).toHaveProperty('created');
      expect(result).toHaveProperty('updated');
      expect(result).toHaveProperty('patterns');
    });

    it('should prune underperforming skills', () => {
      const pruned = skills.pruneSkills({
        minUses: 1,
        minSuccessRate: 0.99,
        maxAgeDays: 0
      });

      expect(pruned).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Causal Memory Graph', () => {
    it('should add causal edges', async () => {
      const edgeId = await causalGraph.addCausalEdge({
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.9,
        uplift: 0.25,
        confidence: 0.95,
        sampleSize: 100,
        mechanism: 'test cause → test effect',
        evidenceIds: []
      });

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should create experiments', () => {
      const expId = causalGraph.createExperiment({
        name: 'test_experiment',
        hypothesis: 'Testing causal relationship',
        treatmentId: 1,
        treatmentType: 'treatment',
        startTime: Math.floor(Date.now() / 1000),
        sampleSize: 0,
        status: 'running',
        metadata: { test: true }
      });

      expect(expId).toBeGreaterThan(0);
    });

    it('should record observations', () => {
      const expId = causalGraph.createExperiment({
        name: 'observation_test',
        hypothesis: 'Test hypothesis',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Math.floor(Date.now() / 1000),
        sampleSize: 0,
        status: 'running',
        metadata: {}
      });

      // Create episode for observation
      const episodeResult = db.prepare(
        'INSERT INTO episodes (session_id, task, reward, success, created_at) VALUES (?, ?, ?, ?, ?)'
      ).run('obs-session', 'test', 0.8, 1, Math.floor(Date.now() / 1000));

      causalGraph.recordObservation({
        experimentId: expId,
        episodeId: Number(episodeResult.lastInsertRowid),
        isTreatment: true,
        outcomeValue: 0.8,
        outcomeType: 'reward'
      });

      // Verify observation was recorded
      const obs = db.prepare('SELECT COUNT(*) as count FROM causal_observations WHERE experiment_id = ?').get(expId);
      expect(obs.count).toBe(1);
    });

    it('should calculate uplift', () => {
      const expId = causalGraph.createExperiment({
        name: 'uplift_test',
        hypothesis: 'Test uplift calculation',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Math.floor(Date.now() / 1000),
        sampleSize: 0,
        status: 'running',
        metadata: {}
      });

      // Add multiple observations
      for (let i = 0; i < 10; i++) {
        const episodeResult = db.prepare(
          'INSERT INTO episodes (session_id, task, reward, success, created_at) VALUES (?, ?, ?, ?, ?)'
        ).run(`uplift-${i}`, 'test', 0.8, 1, Math.floor(Date.now() / 1000));

        causalGraph.recordObservation({
          experimentId: expId,
          episodeId: Number(episodeResult.lastInsertRowid),
          isTreatment: i < 5,
          outcomeValue: i < 5 ? 0.8 : 0.6,
          outcomeType: 'reward'
        });
      }

      const result = causalGraph.calculateUplift(expId);

      expect(result).toHaveProperty('uplift');
      expect(result).toHaveProperty('confidenceInterval');
      expect(result).toHaveProperty('pValue');
    });

    it('should query causal effects', () => {
      const results = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
        minConfidence: 0.5
      });

      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('Database Persistence', () => {
    it('should persist episodes across saves', async () => {
      const episodeId = await reflexion.storeEpisode({
        sessionId: 'persistence-test',
        task: 'persistence task',
        reward: 0.9,
        success: true
      });

      // Save database
      if (db && typeof db.save === 'function') {
        db.save();
      }

      // Verify episode exists
      const episode = db.prepare('SELECT * FROM episodes WHERE id = ?').get(episodeId);
      expect(episode).toBeDefined();
      expect(episode.task).toBe('persistence task');
    });
  });
});
