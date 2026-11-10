import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Database from 'better-sqlite3';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ReflexionMemory, type ReflexionQuery } from '../../../src/controllers/ReflexionMemory.js';
import type { NeuralReranker } from '../../../src/controllers/NeuralReranker.js';

let db: Database.Database;
const embedder = () => ({ embedQuery: vi.fn(async () => Float32Array.of(1, 0)) }) as any;
function controller(rank = vi.fn(async (_id, candidates, options) => ({
  order: options.mode === 'active' ? candidates.map(item => item.id).reverse() : candidates.map(item => item.id),
  proposedOrder: candidates.map(item => item.id).reverse(), scores: candidates.map((item, index) => ({ id: item.id, score: index })),
  revision: 2, mode: options.mode, modelHash: 'a'.repeat(64), available: true
}))) { return { rank } as unknown as NeuralReranker; }
const strip = (values: any[]) => values.map(({ neuralRerank, ...value }) => value);
beforeEach(() => {
  db = new Database(':memory:');
  db.exec(readFileSync(join(__dirname, '../../../src/schemas/schema.sql'), 'utf8'));
  for (let index = 1; index <= 90; index++) {
    const id = db.prepare('INSERT INTO episodes (session_id, task, reward, success) VALUES (?, ?, ?, ?)')
      .run(`session-${index}`, index % 3 ? `deploy service ${index}` : `rare quokkaflux incident ${index}`, index % 2 ? 0.9 : 0.2, index % 2).lastInsertRowid;
    const vector = Float32Array.of(1, index / 90);
    db.prepare('INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)').run(id, Buffer.from(vector.buffer));
  }
});
afterEach(() => db.close());

describe('learned reranking retrieval opt-in', () => {
  it('leaves ordinary retrieval unchanged and shadow keeps the exact original pool/order/cosines', async () => {
    const learned = controller(), memory = new ReflexionMemory(db as any, embedder(), undefined, undefined, undefined, undefined, learned);
    const query = { task: 'quokkaflux deploy', k: 3 };
    const baseline = await memory.retrieveRelevant(query);
    expect(learned.rank).not.toHaveBeenCalled();
    const shadow = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker' } });
    expect(strip(shadow)).toEqual(baseline);
    expect((learned.rank as any).mock.calls[0][1].map(item => item.id)).toEqual(baseline.map(item => String(item.id)));
    expect((learned.rank as any).mock.calls[0][1]).toHaveLength(3);
    expect(shadow.every(item => item.neuralRerank?.mode === 'shadow' && item.neuralRerank.status === 'scored')).toBe(true);
    expect(baseline.every(item => item.neuralRerank === undefined)).toBe(true);
  });
  it('active mode reranks at most 64 filtered candidates and preserves actual cosine fields', async () => {
    const learned = controller(), memory = new ReflexionMemory(db as any, embedder(), undefined, undefined, undefined, undefined, learned);
    const query = { task: 'deploy', k: 3, minReward: 0.8, onlySuccesses: true, timeWindowDays: 1 };
    const full = await memory.retrieveRelevant({ ...query, k: 64 });
    const active = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker', mode: 'active' } });
    expect(active.map(item => item.id)).toEqual(full.map(item => item.id).reverse().slice(0, 3));
    const call = (learned.rank as any).mock.calls[0];
    expect(call[1].length).toBeLessThanOrEqual(64); expect(call[2]).toEqual({ mode: 'active' });
    expect(active.every(item => item.reward >= 0.8 && item.success)).toBe(true);
    const originals = new Map(full.map(item => [item.id, item.similarity]));
    for (const item of active) expect(item.similarity).toBe(originals.get(item.id));
  });
  it('active errors and unavailable models restore the normal original-k baseline, with one query embedding', async () => {
    for (const unavailable of [false, true]) {
      const rank = vi.fn(async (_id, candidates, options) => {
        if (!unavailable) throw new Error('runtime failed');
        return { order: candidates.map(item => item.id), scores: [], revision: 1, mode: options.mode, modelHash: null, available: false };
      });
      const embedding = embedder(), memory = new ReflexionMemory(db as any, embedding, undefined, undefined, undefined, undefined, controller(rank));
      const query = { task: 'quokkaflux deploy', k: 3 };
      const baseline = await memory.retrieveRelevant(query);
      embedding.embedQuery.mockClear();
      const restored = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker', mode: 'active' } });
      expect(strip(restored)).toEqual(baseline);
      expect(restored.every(item => item.neuralRerank?.status === (unavailable ? 'unavailable' : 'error'))).toBe(true);
      expect(embedding.embedQuery).toHaveBeenCalledTimes(1);
      expect(rank.mock.calls[0][1].length).toBeGreaterThan(3);
    }
  });
  it('no controller and graph backends return annotated baseline without constructing a controller', async () => {
    const memory = new ReflexionMemory(db as any, embedder());
    const query = { task: 'deploy', k: 3 };
    const baseline = await memory.retrieveRelevant(query);
    const missing = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker', mode: 'active' } });
    expect(strip(missing)).toEqual(baseline); expect(missing.every(item => item.neuralRerank?.status === 'unavailable')).toBe(true);
    const learned = controller();
    const graph = { execute: vi.fn(async () => []) } as any;
    const graphMemory = new ReflexionMemory(db as any, embedder(), undefined, undefined, graph, undefined, learned);
    vi.spyOn(graphMemory as any, 'retrieveFromGenericGraph').mockResolvedValue(baseline);
    const unsupported = await graphMemory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker', mode: 'active' } });
    expect(strip(unsupported)).toEqual(baseline); expect(unsupported.every(item => item.neuralRerank?.status === 'unsupported')).toBe(true);
    expect(learned.rank).not.toHaveBeenCalled();
  });
  it('neural queries bypass read/write result cache so promotion is visible on the next request', async () => {
    let revision = 2;
    const rank = vi.fn(async (_id, candidates, options) => ({ order: revision === 2 ? candidates.map(item => item.id).reverse() : candidates.map(item => item.id),
      scores: candidates.map(item => ({ id: item.id, score: revision })), revision, modelHash: 'a'.repeat(64), available: true, mode: options.mode }));
    const memory = new ReflexionMemory(db as any, embedder(), undefined, undefined, undefined, undefined, controller(rank));
    const query: ReflexionQuery = { task: 'deploy', k: 3, neuralRerank: { id: 'ranker', mode: 'active' } };
    const first = await memory.retrieveRelevant(query); revision = 3;
    const next = await memory.retrieveRelevant(query);
    expect(rank).toHaveBeenCalledTimes(2); expect(first[0].id).not.toBe(next[0].id);
    expect(next.every(item => item.neuralRerank?.revision === 3)).toBe(true);
  });
  it('snapshots neural query settings before awaiting the embedder and rejects malformed configs explicitly', async () => {
    const learned = controller(), embedding = embedder();
    const memory = new ReflexionMemory(db as any, embedding, undefined, undefined, undefined, undefined, learned);
    const query: ReflexionQuery = { task: 'deploy', k: 3, neuralRerank: { id: 'ranker', mode: 'shadow', candidateId: 'candidate' } };
    const pending = memory.retrieveRelevant(query);
    query.neuralRerank!.id = 'changed'; query.neuralRerank!.mode = 'active'; query.k = 64;
    const result = await pending;
    expect(result).toHaveLength(3); expect((learned.rank as any).mock.calls[0][0]).toBe('ranker');
    expect((learned.rank as any).mock.calls[0][2]).toEqual({ mode: 'shadow', candidateId: 'candidate' });
    for (const config of [null, {}, { id: 'x', mode: 'automatic' }, { id: 'x', extra: 1 }, { id: 'x', mode: 'active', candidateId: 'candidate' }])
      await expect(memory.retrieveRelevant({ task: 'deploy', neuralRerank: config } as any)).rejects.toThrow();
    await expect(memory.retrieveRelevant({ task: 'deploy', k: 65, neuralRerank: { id: 'x' } })).rejects.toThrow('k');
    let calls = 0;
    const config = Object.defineProperty({}, 'id', { get() { calls++; return 'x'; } });
    await expect(memory.retrieveRelevant({ task: 'deploy', neuralRerank: config } as any)).rejects.toThrow('accessors');
    expect(calls).toBe(0);
  });
});


it('filters minSimilarity before active scoring so ineligible high-score rows cannot displace eligible results', async () => {
  const learned = controller(), memory = new ReflexionMemory(db as any, embedder(), undefined, undefined, undefined, undefined, learned);
  const query = { task: 'deploy', k: 3, hybrid: false, minSimilarity: 0.95, neuralRerank: { id: 'ranker', mode: 'active' as const } };
  const ranked = await memory.retrieveRelevant(query);
  const scored = (learned.rank as any).mock.calls[0][1];
  expect(scored.length).toBeGreaterThan(3);
  expect(scored.every(item => item.features.similarity >= (0.95 + 1) / 2)).toBe(true);
  expect(ranked).toHaveLength(3); expect(ranked.every(item => item.similarity! >= 0.95)).toBe(true);
  const baseline = await memory.retrieveRelevant({ task: 'deploy', k: 3, hybrid: false });
  const shadow = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker', mode: 'shadow' } });
  expect(strip(shadow)).toEqual(baseline.filter(item => item.similarity! >= 0.95));
  await expect(memory.retrieveRelevant({ ...query, minSimilarity: NaN })).rejects.toThrow('minSimilarity');
});


it('default shadow keeps negative-cosine baseline rows when no positive similarity filter is requested', async () => {
  db.prepare('UPDATE episode_embeddings SET embedding = ?').run(Buffer.from(Float32Array.of(-1, 0).buffer));
  const learned = controller(), memory = new ReflexionMemory(db as any, embedder(), undefined, undefined, undefined, undefined, learned);
  const query = { task: 'unmatchedword', k: 3, hybrid: false };
  const baseline = await memory.retrieveRelevant(query);
  expect(baseline).toHaveLength(3); expect(baseline.every(item => item.similarity === -1)).toBe(true);
  const shadow = await memory.retrieveRelevant({ ...query, neuralRerank: { id: 'ranker' } });
  expect(strip(shadow)).toEqual(baseline);
  expect((learned.rank as any).mock.calls[0][1].every(item => item.features.similarity === 0)).toBe(true);
});
