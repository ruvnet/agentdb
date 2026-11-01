/**
 * ADR-016: opt-in cross-encoder reranking of the hybrid candidates. The model is mocked: these tests check the
 * fusion and the fallbacks, not the model.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';

const ce = vi.hoisted(() => ({ fail: false, calls: 0, sizes: [] as number[] }));
vi.mock('../../../src/search/cross-encoder.js', () => ({
  // Scores a candidate by whether it mentions "gold"; fails on demand to exercise the fallback.
  crossEncoderScores: async (_q: string, docs: string[]) => {
    ce.calls++;
    ce.sizes.push(docs.length);
    if (ce.fail) throw new Error('model unavailable');
    return Float64Array.from(docs.map(d => (d.includes('gold') ? 10 : 0)));
  },
}));

import { applyCrossEncoder, fuseEpisodeScores, orderByScore, RERANK_DEPTH } from '../../../src/controllers/episode-hybrid.js';
import { ReflexionMemory } from '../../../src/controllers/ReflexionMemory.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';

describe('applyCrossEncoder', () => {
  beforeEach(() => Object.assign(ce, { fail: false, calls: 0, sizes: [] }));

  it('moves the candidate the cross-encoder prefers to the top', async () => {
    const score = fuseEpisodeScores([1, 2, 3], []);
    const text = new Map([[1, 'a'], [2, 'b'], [3, 'the gold answer']]);
    await applyCrossEncoder('q', score, [1, 2, 3], id => text.get(id));
    expect(orderByScore(score, [1, 2, 3])[0]).toBe(3);
  });

  it('skips candidates without text and leaves non-candidates untouched', async () => {
    const score = fuseEpisodeScores([1, 2, 3], []);
    const before = score.get(3);
    await applyCrossEncoder('q', score, [1, 2], id => (id === 1 ? 'gold' : undefined));
    expect(ce.sizes).toEqual([1]);
    expect(score.get(3)).toBe(before);
  });
});

describe('ReflexionMemory rerank option', () => {
  let db: Database.Database;
  let reflexion: ReflexionMemory;

  beforeEach(async () => {
    Object.assign(ce, { fail: false, calls: 0, sizes: [] });
    db = new Database(':memory:');
    db.exec(fs.readFileSync(path.join(__dirname, '../../../src/schemas/schema.sql'), 'utf-8'));
    const embedder = new EmbeddingService({ model: 'mock-model', dimension: 384, provider: 'local' });
    await embedder.initialize();
    reflexion = new ReflexionMemory(db, embedder);
    for (let i = 0; i < 150; i++) {
      await reflexion.storeEpisode({ sessionId: `s${i}`, task: `deploy service ${i} to staging`, reward: 0.5, success: true });
    }
    await reflexion.storeEpisode({ sessionId: 'gold', task: 'deploy service gold to staging', reward: 0.5, success: true });
  });

  afterEach(() => db.close());

  it('is off by default', async () => {
    await reflexion.retrieveRelevant({ task: 'deploy service to staging', k: 3 });
    expect(ce.calls).toBe(0);
  });

  it('reranks a bounded candidate set when asked', async () => {
    const r = await reflexion.retrieveRelevant({ task: 'deploy service to staging', k: 3, rerank: true });
    expect(ce.calls).toBe(1);
    expect(ce.sizes[0]).toBeLessThanOrEqual(2 * RERANK_DEPTH);
    expect(r[0].sessionId).toBe('gold');
  });

  it('keeps the hybrid order when the model cannot run', async () => {
    ce.fail = true;
    const r = await reflexion.retrieveRelevant({ task: 'deploy service to staging', k: 3, rerank: true });
    expect(r).toHaveLength(3);
  });

  it('does nothing with hybrid off', async () => {
    await reflexion.retrieveRelevant({ task: 'deploy service to staging', k: 3, rerank: true, hybrid: false });
    expect(ce.calls).toBe(0);
  });
});
