/**
 * Regression tests for controller defects found in the 3.0.0 review.
 *
 * - AgentDB hands one vector backend to episodes ("123"), skills ("skill:N")
 *   and causal edges ("causal-edge:N"). SkillLibrary parsed every hit with
 *   parseInt(id.replace('skill:', '')), so episode 1 was loaded as skill 1.
 * - LearningSystem.updatePolicyIncremental() mutated a fresh copy of the
 *   policy and never saved it; train() re-read the policy for every batch and
 *   saved an unmodified copy.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { SkillLibrary } from '../../src/controllers/SkillLibrary.js';
import { LearningSystem } from '../../src/controllers/LearningSystem.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import type { SearchResult, VectorBackend } from '../../src/backends/VectorBackend.js';

const DIM = 8;
const SCHEMA = fs.readFileSync(path.join(__dirname, '../../src/schemas/schema.sql'), 'utf-8');

const embedder = {
  embedQuery: async () => new Float32Array(DIM).fill(1),
  embedPassage: async () => new Float32Array(DIM).fill(1),
  embed: async () => new Float32Array(DIM).fill(1),
} as unknown as EmbeddingService;

/** A sync backend that returns a fixed hit list, as a shared index would. */
function sharedBackend(hits: SearchResult[]): VectorBackend {
  return {
    name: 'hnswlib',
    insert() {},
    insertBatch() {},
    search: () => hits,
    remove: () => false,
    getStats: () => ({ count: hits.length, dimension: DIM, metric: 'cosine', backend: 'hnswlib', memoryUsage: 0 }),
    save: async () => {},
    load: async () => {},
    close() {},
  };
}

describe('SkillLibrary on a vector index shared with episodes', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(SCHEMA);
    const insert = db.prepare(
      `INSERT INTO skills (id, name, description, signature, code, success_rate, uses, avg_reward, avg_latency_ms)
       VALUES (?, ?, 'd', '{}', 'c', 0.9, 1, 0.5, 1)`
    );
    insert.run(1, 'skill-one');
    insert.run(7, 'skill-seven');
  });

  afterEach(() => db.close());

  it('ignores hits that are not "skill:N" vector ids', async () => {
    // "1" is episode 1's vector, not skill 1
    const library = new SkillLibrary(
      db as any,
      embedder,
      sharedBackend([
        { id: '1', distance: 0.01, similarity: 0.99 },
        { id: 'causal-edge:1', distance: 0.2, similarity: 0.8 },
        { id: 'skill:7', distance: 0.6, similarity: 0.4 },
      ])
    );

    const skills = await library.retrieveSkills({ task: 'episode text', k: 5, minSuccessRate: 0 });

    expect(skills.map((s) => s.id)).toEqual([7]);
    expect(skills[0].similarity).toBeCloseTo(0.4);
  });

  it('falls back to SQL embeddings when the index returns no skill hits', async () => {
    const library = new SkillLibrary(db as any, embedder, sharedBackend([]));
    const id = await library.createSkill({
      name: 'stored-skill',
      description: 'd',
      signature: { inputs: {}, outputs: {} },
      code: 'c',
      successRate: 0.9,
      uses: 0,
      avgReward: 0,
      avgLatencyMs: 0,
    });

    const skills = await library.retrieveSkills({ task: 'anything', k: 5, minSuccessRate: 0 });

    expect(skills.map((s) => s.id)).toContain(id);
  });
});

describe('LearningSystem policy persistence', () => {
  let db: Database.Database;
  let learning: LearningSystem;

  beforeEach(async () => {
    db = new Database(':memory:');
    const localEmbedder = new EmbeddingService({ model: 'mock-model', dimension: 384, provider: 'local' });
    await localEmbedder.initialize();
    learning = new LearningSystem(db as any, localEmbedder);
  });

  afterEach(() => db.close());

  it('submitFeedback() updates the Q-value that predict() reads', async () => {
    const sessionId = await learning.startSession('u', 'q-learning', {
      learningRate: 0.1,
      discountFactor: 0.9,
      explorationRate: 1e-9,
    });

    await learning.submitFeedback({
      sessionId,
      state: 's',
      action: 'a',
      reward: 1,
      success: true,
      timestamp: Date.now(),
    });

    const prediction = await learning.predict(sessionId, 's');
    expect(prediction.action).toBe('a');
    expect(prediction.qValue).toBeGreaterThan(0);
  });

  it('train() saves the policy it trained', async () => {
    const sessionId = await learning.startSession('u', 'q-learning', {
      learningRate: 0.1,
      discountFactor: 0.9,
      explorationRate: 1e-9,
    });
    await learning.submitFeedback({
      sessionId,
      state: 's',
      action: 'a',
      reward: 1,
      success: true,
      timestamp: Date.now(),
    });

    await learning.train(sessionId, 5, 1, 0.5);

    const prediction = await learning.predict(sessionId, 's');
    // one feedback step gives 0.1; five training steps at lr 0.5 approach 1
    expect(prediction.qValue).toBeGreaterThan(0.9);
  });
});
