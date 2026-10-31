/**
 * Regression tests for RuVectorBackend (the factory's default backend).
 *
 * Before 3.0.0 every search returned []: the native search() promise was never
 * awaited, results were read from `distance` although the binding reports
 * `score`, writes were fire-and-forget, remove()/getStats() called methods the
 * binding does not have, HNSW parameters were passed at the wrong level, and
 * every instance shared ./ruvector.db in the cwd.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { RuVectorBackend } from '../../src/backends/ruvector/RuVectorBackend.js';
import { searchVectors } from '../../src/backends/VectorBackend.js';

const DIM = 8;

function unit(axis: number, wobble = 0): Float32Array {
  const v = new Float32Array(DIM);
  v[axis] = 1;
  v[(axis + 1) % DIM] = wobble;
  return v;
}

describe('RuVectorBackend', () => {
  let dir: string;
  let backend: RuVectorBackend;

  beforeEach(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rv-test-'));
    backend = new RuVectorBackend({
      dimension: DIM,
      metric: 'cosine',
      storagePath: path.join(dir, 'index.db'),
    });
    await backend.initialize();
  });

  afterEach(async () => {
    await backend.flush().catch(() => {});
    backend.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('finds an inserted vector with a finite distance', async () => {
    backend.insert('a', unit(0), { tag: 'a' });
    backend.insert('b', unit(3));
    backend.insert('c', unit(5));

    const results = await searchVectors(backend, unit(0, 0.05), 1);

    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('a');
    expect(Number.isFinite(results[0].distance)).toBe(true);
    expect(results[0].similarity).toBeGreaterThan(0.9);
    expect(results[0].metadata).toEqual({ tag: 'a' });
  });

  it('removes vectors and reports stats', async () => {
    await backend.insertBatchAsync([
      { id: 'a', embedding: unit(0) },
      { id: 'b', embedding: unit(3) },
      { id: 'c', embedding: unit(5) },
    ]);
    expect((await backend.getStatsAsync()).count).toBe(3);

    expect(await backend.removeAsync('a')).toBe(true);
    expect(await backend.removeAsync('missing')).toBe(false);

    const stats = await backend.getStatsAsync();
    expect(stats.count).toBe(2);
    expect(stats.backend).toBe('ruvector');
    expect(backend.getStats().count).toBe(2);

    const ids = (await backend.searchAsync(unit(0), 3)).map((r) => r.id);
    expect(ids).not.toContain('a');
  });

  it('re-inserting an id replaces it instead of leaving a stale vector', async () => {
    await backend.insertAsync('a', unit(0));
    await backend.insertAsync('a', unit(4));

    expect((await backend.getStatsAsync()).count).toBe(1);
    const results = await backend.searchAsync(unit(0), 5);
    expect(results.map((r) => r.id)).toEqual(['a']);
    expect(results[0].similarity).toBeLessThan(0.5);
  });

  it('does not write ./ruvector.db into the working directory', async () => {
    const cwd = process.cwd();
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rv-cwd-'));
    process.chdir(scratch);
    try {
      const temp = new RuVectorBackend({ dimension: DIM, metric: 'cosine' });
      await temp.initialize();
      await temp.insertAsync('x', unit(1));
      expect((await temp.searchAsync(unit(1), 1))[0]?.id).toBe('x');
      temp.close();
      expect(fs.readdirSync(scratch)).toEqual([]);
    } finally {
      process.chdir(cwd);
      fs.rmSync(scratch, { recursive: true, force: true });
    }
  });

  it('rejects the sync search() loudly instead of returning []', () => {
    expect(() => backend.search(unit(0), 1)).toThrow(/async-only/);
  });

  it('sync insert() still throws on invalid input', () => {
    expect(() => backend.insert('', unit(0))).toThrow(/non-empty string/);
    expect(() => backend.insert('x', new Float32Array(DIM + 1))).toThrow(/dimension/);
  });
});

describe('controllers sharing a RuVectorBackend', () => {
  /** Deterministic bag-of-words embedder: distinct texts get distinct vectors. */
  const embed = async (text: string) => {
    const v = new Float32Array(DIM);
    for (const word of text.toLowerCase().split(/\W+/).filter(Boolean)) {
      let h = 0;
      for (const ch of word) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
      v[h % DIM] += 1;
    }
    return v;
  };
  const embedder = { embedQuery: embed, embedPassage: embed, embed } as any;

  let dir: string;
  let backend: RuVectorBackend;
  let db: any;

  beforeEach(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rv-ctl-'));
    backend = new RuVectorBackend({ dimension: DIM, metric: 'cosine', storagePath: path.join(dir, 'index.db') });
    await backend.initialize();
    const Database = (await import('better-sqlite3')).default;
    db = new Database(':memory:');
    db.exec(fs.readFileSync(path.join(__dirname, '../../src/schemas/schema.sql'), 'utf-8'));
  });

  afterEach(async () => {
    await backend.flush().catch(() => {});
    backend.close();
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('ReflexionMemory and SkillLibrary retrieve their own rows through the async backend', async () => {
    const { ReflexionMemory } = await import('../../src/controllers/ReflexionMemory.js');
    const { SkillLibrary } = await import('../../src/controllers/SkillLibrary.js');
    const reflexion = new ReflexionMemory(db, embedder, backend);
    const skills = new SkillLibrary(db, embedder, backend);

    const episodeId = await reflexion.storeEpisode({
      sessionId: 's', task: 'parse csv files quickly', reward: 0.9, success: true,
    });
    const skillId = await skills.createSkill({
      name: 'deploy', description: 'ship docker containers', signature: { inputs: {}, outputs: {} },
      code: 'c', successRate: 0.9, uses: 0, avgReward: 0, avgLatencyMs: 0,
    });

    const episodes = await reflexion.retrieveRelevant({ task: 'parse csv files quickly', k: 3 });
    expect(episodes.map((e) => e.id)).toContain(episodeId);

    const found = await skills.retrieveSkills({ task: 'deploy ship docker containers', k: 3, minSuccessRate: 0 });
    expect(found.map((s) => s.id)).toEqual([skillId]);
  });
});
