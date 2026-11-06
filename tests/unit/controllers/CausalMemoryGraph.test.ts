/**
 * Unit Tests for CausalMemoryGraph Controller
 *
 * Tests causal reasoning, intervention-based analysis, and uplift modeling
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { CausalMemoryGraph, CausalEdge, CausalExperiment, CausalObservation } from '../../../src/controllers/CausalMemoryGraph.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-causal.db';

describe('CausalMemoryGraph', () => {
  let db: Database.Database;
  let causalGraph: CausalMemoryGraph;

  beforeEach(() => {
    // Clean up existing test database
    if (fs.existsSync(TEST_DB_PATH)) {
      fs.unlinkSync(TEST_DB_PATH);
    }
    if (fs.existsSync(`${TEST_DB_PATH}-wal`)) {
      fs.unlinkSync(`${TEST_DB_PATH}-wal`);
    }
    if (fs.existsSync(`${TEST_DB_PATH}-shm`)) {
      fs.unlinkSync(`${TEST_DB_PATH}-shm`);
    }

    // Initialize database
    db = new Database(TEST_DB_PATH);
    db.pragma('journal_mode = WAL');

    // Load base schema first (contains episodes, skills, patterns tables)
    const baseSchemaPath = path.join(__dirname, '../../../src/schemas/schema.sql');
    if (fs.existsSync(baseSchemaPath)) {
      const baseSchema = fs.readFileSync(baseSchemaPath, 'utf-8');
      db.exec(baseSchema);
    }

    // Load frontier schema (contains causal_edges, experiments, observations)
    const frontierSchemaPath = path.join(__dirname, '../../../src/schemas/frontier-schema.sql');
    if (fs.existsSync(frontierSchemaPath)) {
      const frontierSchema = fs.readFileSync(frontierSchemaPath, 'utf-8');
      db.exec(frontierSchema);
    }

    causalGraph = new CausalMemoryGraph(db);
  });

  afterEach(() => {
    db.close();
    if (fs.existsSync(TEST_DB_PATH)) {
      fs.unlinkSync(TEST_DB_PATH);
    }
    if (fs.existsSync(`${TEST_DB_PATH}-wal`)) {
      fs.unlinkSync(`${TEST_DB_PATH}-wal`);
    }
    if (fs.existsSync(`${TEST_DB_PATH}-shm`)) {
      fs.unlinkSync(`${TEST_DB_PATH}-shm`);
    }
  });

  describe('addCausalEdge', () => {
    it('should add causal edge with all required fields', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.85,
        uplift: 0.25,
        confidence: 0.95,
        sampleSize: 100,
        evidenceIds: ['e1', 'e2', 'e3'],
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
      expect(typeof edgeId).toBe('number');
    });

    it('should add causal edge with minimal fields', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'skill',
        toMemoryId: 2,
        toMemoryType: 'skill',
        similarity: 0.7,
        confidence: 0.8,
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should handle negative uplift (harmful effects)', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.9,
        uplift: -0.3,
        confidence: 0.9,
        sampleSize: 50,
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should store edge with mechanism explanation', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.88,
        uplift: 0.2,
        confidence: 0.87,
        mechanism: 'Adding tests reduces bugs by catching errors early',
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });
  });

  describe('createExperiment', () => {
    it('should create A/B test experiment', () => {
      const experiment: CausalExperiment = {
        name: 'Test Impact of Code Reviews',
        hypothesis: 'Code reviews reduce bug rate',
        treatmentId: 100,
        treatmentType: 'code_review',
        controlId: 101,
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      };

      const expId = causalGraph.createExperiment(experiment);

      expect(expId).toBeGreaterThan(0);
    });

    it('should create experiment without control group', () => {
      const experiment: CausalExperiment = {
        name: 'Test New Feature',
        hypothesis: 'Feature improves UX',
        treatmentId: 200,
        treatmentType: 'feature_flag',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      };

      const expId = causalGraph.createExperiment(experiment);

      expect(expId).toBeGreaterThan(0);
    });
  });

  describe('recordObservation', () => {
    it('should record treatment observation', () => {
      // Create episode first (required by foreign key constraint)
      db.prepare(`
        INSERT INTO episodes (id, ts, session_id, task, reward, success)
        VALUES (1, ?, 'test-session', 'test task', 0.85, 1)
      `).run(Date.now());

      const expId = causalGraph.createExperiment({
        name: 'Test',
        hypothesis: 'Tests help',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      });

      const observation: CausalObservation = {
        experimentId: expId,
        episodeId: 1,
        isTreatment: true,
        outcomeValue: 0.85,
        outcomeType: 'reward',
      };

      expect(() => causalGraph.recordObservation(observation)).not.toThrow();
    });

    it('should record control observation', () => {
      // Create episode first (required by foreign key constraint)
      db.prepare(`
        INSERT INTO episodes (id, ts, session_id, task, reward, success)
        VALUES (2, ?, 'test-session', 'test task', 0.65, 1)
      `).run(Date.now());

      const expId = causalGraph.createExperiment({
        name: 'Test',
        hypothesis: 'Tests help',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      });

      const observation: CausalObservation = {
        experimentId: expId,
        episodeId: 2,
        isTreatment: false,
        outcomeValue: 0.65,
        outcomeType: 'reward',
      };

      expect(() => causalGraph.recordObservation(observation)).not.toThrow();
    });
  });

  describe('calculateUplift', () => {
    it('should calculate positive uplift', () => {
      // Create all episodes first (required by foreign key constraint)
      const insertEpisode = db.prepare(`
        INSERT INTO episodes (id, ts, session_id, task, reward, success)
        VALUES (?, ?, 'test-session', 'test task', ?, 1)
      `);

      for (let i = 0; i < 20; i++) {
        insertEpisode.run(i, Date.now(), i < 10 ? 0.85 : 0.65);
      }

      const expId = causalGraph.createExperiment({
        name: 'Test',
        hypothesis: 'Tests help',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      });

      // Add treatment observations (higher rewards)
      for (let i = 0; i < 10; i++) {
        causalGraph.recordObservation({
          experimentId: expId,
          episodeId: i,
          isTreatment: true,
          outcomeValue: 0.8 + Math.random() * 0.1,
          outcomeType: 'reward',
        });
      }

      // Add control observations (lower rewards)
      for (let i = 10; i < 20; i++) {
        causalGraph.recordObservation({
          experimentId: expId,
          episodeId: i,
          isTreatment: false,
          outcomeValue: 0.6 + Math.random() * 0.1,
          outcomeType: 'reward',
        });
      }

      const result = causalGraph.calculateUplift(expId);

      expect(result.uplift).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThanOrEqual(0);
      expect(result.pValue).toBeLessThanOrEqual(1);
      expect(result.confidenceInterval).toHaveLength(2);
      expect(result.confidenceInterval[0]).toBeLessThan(result.confidenceInterval[1]);
    });

    it('should handle experiment with no observations', () => {
      const expId = causalGraph.createExperiment({
        name: 'Test',
        hypothesis: 'Tests help',
        treatmentId: 1,
        treatmentType: 'test',
        startTime: Date.now(),
        sampleSize: 0,
        status: 'running',
      });

      const result = causalGraph.calculateUplift(expId);

      expect(result.uplift).toBe(0);
      expect(result.pValue).toBe(1.0);
    });
  });

  describe('queryCausalEffects', () => {
    beforeEach(async () => {
      // Seed some causal edges
      await causalGraph.addCausalEdge({
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.9,
        uplift: 0.3,
        confidence: 0.95,
        sampleSize: 100,
      });

      await causalGraph.addCausalEdge({
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 3,
        toMemoryType: 'episode',
        similarity: 0.85,
        uplift: 0.15,
        confidence: 0.88,
        sampleSize: 80,
      });

      await causalGraph.addCausalEdge({
        fromMemoryId: 2,
        fromMemoryType: 'episode',
        toMemoryId: 4,
        toMemoryType: 'episode',
        similarity: 0.7,
        uplift: 0.05,
        confidence: 0.6,
        sampleSize: 30,
      });
    });

    it('should query causal effects by intervention', () => {
      const effects = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
      });

      expect(effects.length).toBeGreaterThan(0);
      effects.forEach(edge => {
        expect(edge.fromMemoryId).toBe(1);
        expect(edge.fromMemoryType).toBe('episode');
      });
    });

    it('should filter by minimum confidence', () => {
      const effects = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
        minConfidence: 0.9,
      });

      effects.forEach(edge => {
        expect(edge.confidence).toBeGreaterThanOrEqual(0.9);
      });
    });

    it('should filter by minimum uplift', () => {
      const effects = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
        minUplift: 0.2,
      });

      effects.forEach(edge => {
        expect(Math.abs(edge.uplift || 0)).toBeGreaterThanOrEqual(0.2);
      });
    });

    it('should sort by impact (uplift * confidence)', () => {
      const effects = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
        minConfidence: 0.5,
      });

      // Verify descending order
      for (let i = 0; i < effects.length - 1; i++) {
        const impact1 = Math.abs(effects[i].uplift || 0) * effects[i].confidence;
        const impact2 = Math.abs(effects[i + 1].uplift || 0) * effects[i + 1].confidence;
        expect(impact1).toBeGreaterThanOrEqual(impact2);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero uplift', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.9,
        uplift: 0.0,
        confidence: 0.95,
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should handle very high confidence', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.99,
        uplift: 0.5,
        confidence: 0.99,
        sampleSize: 1000,
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should handle low confidence edges', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.5,
        uplift: 0.1,
        confidence: 0.5,
        sampleSize: 10,
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });

    it('should handle empty evidence IDs', async () => {
      const edge: CausalEdge = {
        fromMemoryId: 1,
        fromMemoryType: 'episode',
        toMemoryId: 2,
        toMemoryType: 'episode',
        similarity: 0.8,
        confidence: 0.85,
        evidenceIds: [],
      };

      const edgeId = await causalGraph.addCausalEdge(edge);

      expect(edgeId).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('should add 100 causal edges efficiently', async () => {
      const startTime = Date.now();

      for (let i = 0; i < 100; i++) {
        await causalGraph.addCausalEdge({
          fromMemoryId: i,
          fromMemoryType: 'episode',
          toMemoryId: i + 1,
          toMemoryType: 'episode',
          similarity: Math.random(),
          uplift: Math.random() * 0.4 - 0.2,
          confidence: 0.5 + Math.random() * 0.5,
          sampleSize: Math.floor(Math.random() * 100) + 10,
        });
      }

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(1000); // Should complete in less than 1 second
    });

    it('should query causal effects efficiently', async () => {
      // Add test data
      for (let i = 0; i < 50; i++) {
        await causalGraph.addCausalEdge({
          fromMemoryId: 1,
          fromMemoryType: 'episode',
          toMemoryId: i + 2,
          toMemoryType: 'episode',
          similarity: Math.random(),
          uplift: Math.random() * 0.4 - 0.2,
          confidence: 0.5 + Math.random() * 0.5,
        });
      }

      const startTime = Date.now();

      const effects = causalGraph.queryCausalEffects({
        interventionMemoryId: 1,
        interventionMemoryType: 'episode',
        minConfidence: 0.7,
      });

      const duration = Date.now() - startTime;

      expect(effects.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(100); // Should complete in less than 100ms
    });
  });
});
