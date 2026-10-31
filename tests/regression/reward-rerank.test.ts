/**
 * ADR-015 / E1: outcome feedback and reward-aware rerank.
 *
 * Pattern search ranked by similarity only; success rate and reward could
 * filter but never reorder, and there was no way to feed an outcome back for
 * a skill. Patterns are now ranked by
 *   0.7·similarity + 0.2·successRate + 0.1·avgReward
 * and recordOutcome() exists on both ReasoningBank and SkillLibrary.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { ReasoningBank } from '../../src/controllers/ReasoningBank.js';
import { SkillLibrary } from '../../src/controllers/SkillLibrary.js';
import type { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import type { SearchResult, VectorBackend } from '../../src/backends/VectorBackend.js';

const SCHEMA = fs.readFileSync(path.join(__dirname, '../../src/schemas/schema.sql'), 'utf-8');
const DIM = 16;

/** Vectors keyed by the text before the first ":" or newline */
const vectors = new Map<string, Float32Array>();
const keyOf = (text: string) => text.split(/[:\n]/)[0].trim();
const embedder = {
  embedQuery: async (t: string) => vectors.get(keyOf(t))!,
  embedPassage: async (t: string) => vectors.get(keyOf(t))!,
  embed: async (t: string) => vectors.get(keyOf(t))!,
} as unknown as EmbeddingService;

function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

/** An exact in-memory index, so the v2 path is exercised without HNSW noise */
function exactBackend(): VectorBackend {
  const stored = new Map<string, { v: Float32Array; meta?: Record<string, unknown> }>();
  return {
    name: 'hnswlib',
    insert(id, v, meta) { stored.set(id, { v: new Float32Array(v), meta }); },
    insertBatch() {},
    search(q, k): SearchResult[] {
      return [...stored]
        .map(([id, e]) => ({ id, similarity: cosine(q, e.v), distance: 1 - cosine(q, e.v), metadata: e.meta }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);
    },
    remove: (id) => stored.delete(id),
    getStats: () => ({ count: stored.size, dimension: DIM, metric: 'cosine', backend: 'hnswlib', memoryUsage: 0 }),
    save: async () => {},
    load: async () => {},
    close() {},
  };
}

function vec(values: number[]): Float32Array {
  const v = new Float32Array(DIM);
  values.forEach((x, i) => (v[i] = x));
  return v;
}

for (const mode of ['legacy (SQL)', 'v2 (vector backend)'] as const) {
  describe(`ReasoningBank rerank, ${mode}`, () => {
    let db: Database.Database;
    let bank: ReasoningBank;
    beforeEach(() => {
      db = new Database(':memory:');
      db.exec(SCHEMA);
      bank = mode === 'legacy (SQL)' ? new ReasoningBank(db as any, embedder) : new ReasoningBank(db as any, embedder, exactBackend());
    });
    afterEach(() => db.close());

    it('swaps two near-duplicate patterns once their feedback disagrees', async () => {
      vectors.set('query', vec([1, 0]));
      vectors.set('closer', vec([1, 0.05])); // marginally more similar
      vectors.set('further', vec([1, 0.08]));
      const closer = await bank.storePattern({ taskType: 'closer', approach: 'a', successRate: 0.5 });
      const further = await bank.storePattern({ taskType: 'further', approach: 'b', successRate: 0.5 });

      const before = await bank.searchPatterns({ task: 'query', k: 2 });
      expect(before.map((p) => p.id)).toEqual([closer, further]);

      for (let i = 0; i < 3; i++) {
        expect(await bank.recordOutcome(closer, false, 0)).toBe(true);
        expect(await bank.recordOutcome(further, true, 1)).toBe(true);
      }

      const after = await bank.searchPatterns({ task: 'query', k: 2 });
      expect(after.map((p) => p.id)).toEqual([further, closer]);
      expect(after[0].score).toBeGreaterThan(after[1].score!);
      // similarity is still reported unchanged; only the order moved
      expect(after[0].similarity!).toBeLessThan(after[1].similarity!);

      // Opting out restores similarity order
      const plain = await bank.searchPatterns({ task: 'query', k: 2, rerank: false });
      expect(plain.map((p) => p.id)).toEqual([closer, further]);
    });

    it('keeps similarity dominant: feedback does not lift an unrelated pattern', async () => {
      vectors.set('query', vec([1, 0]));
      vectors.set('relevant', vec([1, 0.1]));
      vectors.set('unrelated', vec([0, 1]));
      const relevant = await bank.storePattern({ taskType: 'relevant', approach: 'a', successRate: 0 });
      const unrelated = await bank.storePattern({ taskType: 'unrelated', approach: 'b', successRate: 1 });
      for (let i = 0; i < 5; i++) await bank.recordOutcome(unrelated, true, 1);

      const got = await bank.searchPatterns({ task: 'query', k: 2 });
      expect(got.map((p) => p.id)).toEqual([relevant, unrelated]);
    });

    it('precision@5 on a seeded set is no worse than similarity-only', async () => {
      // 4 topics x 6 patterns. Each topic has 3 approaches that work and 3
      // near-duplicates that failed in use. Relevant = same topic AND works.
      let seed = 11;
      const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
      const relevantIds = new Map<number, Set<number>>();
      for (let topic = 0; topic < 4; topic++) {
        const centre = vec([]);
        centre[topic * 4] = 1;
        vectors.set(`q${topic}`, centre);
        relevantIds.set(topic, new Set());
        for (let j = 0; j < 6; j++) {
          const v = new Float32Array(centre);
          for (let d = 0; d < DIM; d++) v[d] += (rnd() - 0.5) * 0.4;
          const key = `t${topic}p${j}`;
          vectors.set(key, v);
          const works = j % 2 === 0;
          const id = await bank.storePattern({ taskType: key, approach: 'x', successRate: 0.5 });
          for (let n = 0; n < 4; n++) await bank.recordOutcome(id, works, works ? 1 : 0);
          if (works) relevantIds.get(topic)!.add(id);
        }
      }

      let simOnly = 0;
      let reranked = 0;
      for (let topic = 0; topic < 4; topic++) {
        const rel = relevantIds.get(topic)!;
        const a = await bank.searchPatterns({ task: `q${topic}`, k: 5, rerank: false });
        const b = await bank.searchPatterns({ task: `q${topic}`, k: 5 });
        simOnly += a.filter((p) => rel.has(p.id!)).length / 5;
        reranked += b.filter((p) => rel.has(p.id!)).length / 5;
      }
      expect(reranked / 4).toBeGreaterThanOrEqual(simOnly / 4);
    });

    it('rejects invalid weights and reports unknown pattern ids', async () => {
      expect(() => new ReasoningBank(db as any, embedder, undefined, undefined, { rerankWeights: { reward: -1 } })).toThrow(/non-negative/);
      expect(await bank.recordOutcome(987654, true)).toBe(false);
      await expect(bank.recordOutcome(1, true, Number.NaN)).rejects.toThrow(/finite/);
    });
  });
}

describe('SkillLibrary.recordOutcome', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(SCHEMA);
  });
  afterEach(() => db.close());

  it('updates success rate, reward and uses, and reorders retrieval', async () => {
    vectors.set('query', vec([1, 0]));
    vectors.set('skill-a', vec([1, 0.05]));
    vectors.set('skill-b', vec([1, 0.08]));
    const library = new SkillLibrary(db as any, embedder);
    const a = await library.createSkill({ name: 'skill-a', successRate: 0.5, uses: 0, avgReward: 0, avgLatencyMs: 40 });
    const b = await library.createSkill({ name: 'skill-b', successRate: 0.5, uses: 0, avgReward: 0, avgLatencyMs: 40 });

    expect((await library.retrieveSkills({ task: 'query', k: 2, minSuccessRate: 0 })).map((s) => s.id)).toEqual([a, b]);

    expect(library.recordOutcome(a, false)).toBe(true);
    expect(library.recordOutcome(b, true, 0.9, 100)).toBe(true);
    expect(library.recordOutcome(424242, true)).toBe(false);

    const row = (id: number) => db.prepare('SELECT uses, success_rate, avg_reward, avg_latency_ms FROM skills WHERE id = ?').get(id) as any;
    expect(row(a)).toEqual({ uses: 1, success_rate: 0, avg_reward: 0, avg_latency_ms: 40 });
    expect(row(b)).toEqual({ uses: 1, success_rate: 1, avg_reward: 0.9, avg_latency_ms: 100 });

    // retrieveSkills caches by query; recordOutcome must invalidate it
    expect((await library.retrieveSkills({ task: 'query', k: 2, minSuccessRate: 0 })).map((s) => s.id)).toEqual([b, a]);
    expect(() => library.recordOutcome(a, true, Infinity)).toThrow(/finite/);
    expect(() => library.recordOutcome(a, true, 1, -5)).toThrow(/non-negative/);
  });
});
