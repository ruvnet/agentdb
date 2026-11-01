/**
 * ADR-016: ReflexionMemory.retrieveRelevant fuses BM25 keyword ranking with the dense ranking.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { ReflexionMemory } from '../../../src/controllers/ReflexionMemory.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';
import { EpisodeLexicalIndex, fuseEpisodeRankings } from '../../../src/controllers/episode-hybrid.js';

describe('fuseEpisodeRankings', () => {
  it('keeps the dense order when there is no lexical hit', () => {
    expect(fuseEpisodeRankings([3, 1, 2], [])).toEqual([3, 1, 2]);
  });

  it('lifts an episode both rankings share above a dense-only one', () => {
    expect(fuseEpisodeRankings([1, 2, 3], [{ id: 2 }])[0]).toBe(2);
  });

  it('adds lexical-only episodes after the dense ones they do not beat', () => {
    // A lexical-only hit is ranked just past the dense pool on the dense side, so a small lexical weight leaves it last.
    const out = fuseEpisodeRankings([1, 2], [{ id: 9 }], 0.01);
    expect(out).toEqual([1, 2, 9]);
  });
});

describe('ReflexionMemory hybrid retrieval', () => {
  let db: Database.Database;
  let reflexion: ReflexionMemory;

  beforeEach(async () => {
    db = new Database(':memory:');
    db.exec(fs.readFileSync(path.join(__dirname, '../../../src/schemas/schema.sql'), 'utf-8'));
    const frontier = path.join(__dirname, '../../../src/schemas/frontier-schema.sql');
    if (fs.existsSync(frontier)) db.exec(fs.readFileSync(frontier, 'utf-8'));
    const embedder = new EmbeddingService({ model: 'mock-model', dimension: 384, provider: 'local' });
    await embedder.initialize();
    reflexion = new ReflexionMemory(db, embedder);
    for (let i = 0; i < 80; i++) {
      await reflexion.storeEpisode({
        sessionId: `s${i}`,
        task: `routine maintenance task number ${i} about general upkeep`,
        output: `completed step ${i}`,
        reward: 0.5,
        success: true,
      });
    }
    await reflexion.storeEpisode({
      sessionId: 'needle',
      task: 'fixed the ERR_SQLITE_BUSY lock in the zephyrine migration runner',
      reward: 0.9,
      success: true,
    });
  });

  afterEach(() => db.close());

  it('finds a keyword-only match that pure vector order does not rank first', async () => {
    const q = { task: 'zephyrine ERR_SQLITE_BUSY', k: 3 };
    const hybrid = await reflexion.retrieveRelevant(q);
    expect(hybrid[0].sessionId).toBe('needle');
    expect(hybrid[0].similarity).toBeTypeOf('number');
    const dense = await reflexion.retrieveRelevant({ ...q, hybrid: false });
    expect(dense.length).toBe(3);
  });

  it('returns at most k episodes and honours filters on keyword-only hits', async () => {
    const r = await reflexion.retrieveRelevant({ task: 'zephyrine', k: 2, minReward: 0.95 });
    expect(r.length).toBeLessThanOrEqual(2);
    expect(r.every(e => e.reward >= 0.95)).toBe(true);
  });

  it('sees an episode stored after the index was built', async () => {
    await reflexion.retrieveRelevant({ task: 'zephyrine', k: 1 });
    await reflexion.storeEpisode({ sessionId: 'late', task: 'quokkaflux deployment rollback', reward: 0.7, success: true });
    const r = await reflexion.retrieveRelevant({ task: 'quokkaflux', k: 1 });
    expect(r[0].sessionId).toBe('late');
  });

  it('caps the indexed episodes', () => {
    const idx = new EpisodeLexicalIndex(db, 10);
    expect(idx.search('routine maintenance', 100).length).toBeLessThanOrEqual(10);
  });
});
