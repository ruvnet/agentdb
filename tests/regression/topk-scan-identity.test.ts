/**
 * ADR-015 / E7: full-table scans use a bounded top-k heap.
 *
 * ReflexionMemory, SkillLibrary, WASMVectorSearch and SqlJsRvfBackend used to
 * score every row, build an object for it, sort them all and slice k. They now
 * keep the k best in a heap and build result objects for the winners only.
 *
 * Two things are checked against each scan site:
 * - results (ids, order, bit-exact similarity) equal the old
 *   sort-then-slice reference on a seeded set with deliberate exact ties;
 * - only the winners are materialised (these assertions fail on 3.0.0).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { TopK } from '../../src/utils/top-k.js';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory.js';
import { SkillLibrary } from '../../src/controllers/SkillLibrary.js';
import { WASMVectorSearch } from '../../src/controllers/WASMVectorSearch.js';
import { SqlJsRvfBackend } from '../../src/backends/rvf/SqlJsRvfBackend.js';
import { cosineSimilaritySIMD } from '../../src/simd/simd-vector-ops.js';
import type { EmbeddingService } from '../../src/controllers/EmbeddingService.js';

const SCHEMA = fs.readFileSync(path.join(__dirname, '../../src/schemas/schema.sql'), 'utf-8');
const DIM = 19; // not a multiple of 4 or 8: exercises the unrolled remainders
const N = 300;
const K = 10;

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** N seeded vectors; rows in TIE_ROWS are exact copies of the query. */
const TIE_ROWS = new Set([3, 50, 120, 200, 299, 17, 18]);
function seededVectors(): { query: Float32Array; rows: Float32Array[] } {
  const rand = mulberry32(42);
  const vec = () => Float32Array.from({ length: DIM }, () => rand() * 2 - 1);
  const query = vec();
  const rows = Array.from({ length: N }, (_, i) => (TIE_ROWS.has(i) ? new Float32Array(query) : vec()));
  return { query, rows };
}

function blob(v: Float32Array): Buffer {
  return Buffer.from(v.buffer, v.byteOffset, v.byteLength);
}

/** The controllers' cosine as it was in 3.0.0 */
function naiveCosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const { query, rows: vectors } = seededVectors();
const embedder = {
  embedQuery: async () => query,
  embedPassage: async () => query,
  embed: async () => query,
} as unknown as EmbeddingService;

describe('TopK', () => {
  it('equals a stable sort-then-slice, ties in push order', () => {
    const rand = mulberry32(7);
    // Coarse scores so many are exactly equal
    const scores = Array.from({ length: 2000 }, () => Math.round(rand() * 50) / 50);
    for (const k of [0, 1, 5, 64, 2000, 5000]) {
      const top = new TopK<number>(k);
      scores.forEach((s, i) => top.push(s, i));
      const reference = scores
        .map((score, item) => ({ score, item }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
      expect(top.sorted()).toEqual(reference);
    }
  });
});

describe('ReflexionMemory SQL scan', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(SCHEMA);
    const ep = db.prepare(`INSERT INTO episodes (id, session_id, task, reward, success, tags, metadata)
                           VALUES (?, 's', ?, ?, 1, '["t"]', '{"m":1}')`);
    const emb = db.prepare('INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)');
    vectors.forEach((v, i) => {
      ep.run(i + 1, `task ${i}`, (i % 5) / 5); // repeated rewards: ORDER BY reward has ties too
      emb.run(i + 1, blob(v));
    });
  });
  afterEach(() => db.close());

  it('returns the old ranking and converts only k rows', async () => {
    const rowsInScanOrder = db
      .prepare(`SELECT e.id, ee.embedding FROM episodes e JOIN episode_embeddings ee ON e.id = ee.episode_id ORDER BY e.reward DESC`)
      .all() as Array<{ id: number; embedding: Buffer }>;
    const reference = rowsInScanOrder
      .map((r) => ({ id: r.id, similarity: naiveCosine(query, new Float32Array(r.embedding.buffer, r.embedding.byteOffset, DIM)) }))
      .sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
      .slice(0, K);

    const memory = new ReflexionMemory(db as any, embedder);
    const convert = vi.spyOn(ReflexionMemory.prototype as any, 'convertDatabaseEpisode');
    const got = await memory.retrieveRelevant({ task: 'q', k: K });

    expect(got.map((e) => ({ id: e.id, similarity: e.similarity }))).toEqual(reference);
    expect(convert).toHaveBeenCalledTimes(K);
  });
});

describe('SkillLibrary SQL scan', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(SCHEMA);
    const skill = db.prepare(`INSERT INTO skills (id, name, description, signature, code, success_rate, uses, avg_reward, avg_latency_ms, metadata)
                              VALUES (?, ?, 'd', '{"inputs":{},"outputs":{}}', 'c', ?, ?, ?, 1, '{"m":1}')`);
    const emb = db.prepare('INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)');
    vectors.forEach((v, i) => {
      skill.run(i + 1, `skill-${i}`, 0.5 + (i % 3) / 10, i % 4, (i % 2) / 2);
      emb.run(i + 1, blob(v));
    });
  });
  afterEach(() => db.close());

  it('returns the old composite ranking and parses JSON for k rows only', async () => {
    const scan = db
      .prepare(`SELECT s.*, e.embedding FROM skills s LEFT JOIN skill_embeddings e ON s.id = e.skill_id WHERE s.success_rate >= ?`)
      .all(0.5) as any[];
    const score = (r: any, sim: number) =>
      sim * 0.4 + r.success_rate * 0.3 + Math.min(r.uses / 1000, 1.0) * 0.1 + r.avg_reward * 0.2;
    const reference = scan
      .map((r) => {
        const similarity = naiveCosine(query, new Float32Array(r.embedding.buffer, r.embedding.byteOffset, DIM));
        return { id: r.id, similarity, s: score(r, similarity) };
      })
      .sort((a, b) => b.s - a.s)
      .slice(0, K)
      .map(({ id, similarity }) => ({ id, similarity }));

    const library = new SkillLibrary(db as any, embedder);
    const parse = vi.spyOn(JSON, 'parse');
    const got = await library.retrieveSkills({ task: 'q', k: K, minSuccessRate: 0.5 });

    expect(got.map((s: any) => ({ id: s.id, similarity: s.similarity }))).toEqual(reference);
    // signature + metadata per returned skill, not per scanned row
    expect(parse.mock.calls.length).toBeLessThanOrEqual(2 * K);
  });
});

describe('WASMVectorSearch.findKNN', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = new Database(':memory:');
    db.exec('CREATE TABLE pattern_embeddings (pattern_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)');
    const emb = db.prepare('INSERT INTO pattern_embeddings (pattern_id, embedding) VALUES (?, ?)');
    vectors.forEach((v, i) => emb.run(i + 1, blob(v)));
  });
  afterEach(() => db.close());

  it('returns the old ranking without a per-row cosineSimilarity call', async () => {
    const search = new WASMVectorSearch(db, { enableWASM: false, enableSIMD: false });
    const reference = vectors
      .map((v, i) => {
        const similarity = search.cosineSimilarity(query, v);
        return { id: i + 1, distance: 1 - similarity, similarity };
      })
      .filter((c) => c.similarity >= 0.1)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, K);

    const perRow = vi.spyOn(search, 'cosineSimilarity');
    const got = await search.findKNN(query, K, 'pattern_embeddings', { threshold: 0.1 });

    expect(got).toEqual(reference);
    expect(perRow).not.toHaveBeenCalled();
  });
});

describe('SqlJsRvfBackend brute-force search', () => {
  it('returns the old ranking and scores only the winners twice', async () => {
    const backend = new SqlJsRvfBackend({ dimension: DIM, metric: 'cosine' } as any);
    await backend.initialize();
    try {
      await backend.insertBatchAsync(vectors.map((v, i) => ({ id: `v${i}`, embedding: v, metadata: { i } })));
      const reference = vectors
        .map((v, i) => {
          const similarity = cosineSimilaritySIMD(query, v);
          return { id: `v${i}`, distance: 1 - similarity, similarity, metadata: { i } };
        })
        .filter((r) => r.similarity >= 0)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, K);

      const computeScore = vi.spyOn(backend as any, 'computeScore');
      const got = await backend.searchAsync(query, K);

      expect(got).toEqual(reference);
      // 3.0.0 called computeScore once per cached vector
      expect(computeScore).toHaveBeenCalledTimes(K);
    } finally {
      backend.close();
    }
  });
});
