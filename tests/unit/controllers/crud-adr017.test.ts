/**
 * ADR-017: get / list / update / delete for episodes, reasoning patterns and skills.
 * Every update must move the SQL row, the embedding table, the vector index, the caches and
 * (for episodes) the BM25 index together; deletes must leave no orphans even with FK cascades off.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { ReflexionMemory } from '../../../src/controllers/ReflexionMemory.js';
import { ReasoningBank } from '../../../src/controllers/ReasoningBank.js';
import { SkillLibrary } from '../../../src/controllers/SkillLibrary.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';
import { CrudValidationError, mergeMetadata, toRecordId } from '../../../src/controllers/crud-helpers.js';

const schemaDir = path.join(__dirname, '../../../src/schemas');

function openDb(foreignKeys: boolean): Database.Database {
  const db = new Database(':memory:');
  db.exec(fs.readFileSync(path.join(schemaDir, 'schema.sql'), 'utf-8'));
  const frontier = path.join(schemaDir, 'frontier-schema.sql');
  if (fs.existsSync(frontier)) db.exec(fs.readFileSync(frontier, 'utf-8'));
  // schema.sql turns cascades on; the orphan tests run with them off, which is the driver default.
  db.pragma(`foreign_keys = ${foreignKeys ? 'ON' : 'OFF'}`);
  return db;
}

async function mockEmbedder(): Promise<EmbeddingService> {
  const e = new EmbeddingService({ model: 'mock-model', dimension: 384, provider: 'local' });
  await e.initialize();
  return e;
}

/** Minimal in-memory VectorBackend that records removals. */
function fakeVectorBackend() {
  const store = new Map<string, Float32Array>();
  return {
    store,
    removed: [] as string[],
    insert(id: string, v: Float32Array) { store.set(id, v); },
    insertBatch(items: { id: string; embedding: Float32Array }[]) { for (const i of items) store.set(i.id, i.embedding); },
    search() { return []; },
    remove(id: string) { this.removed.push(id); return store.delete(id); },
    getStats() { return { count: store.size, dimension: 384, metric: 'cosine', backend: 'fake', memoryUsage: 0 }; },
    save: async () => {},
    load: async () => {},
    close() {},
    name: 'fake',
  } as any;
}

const count = (db: Database.Database, sql: string, ...p: unknown[]) => (db.prepare(sql).get(...p) as { c: number }).c;

describe('crud helpers', () => {
  it('rejects ids parseInt would accept', () => {
    expect(() => toRecordId('12-notes')).toThrow(CrudValidationError);
    expect(() => toRecordId(0)).toThrow();
    expect(() => toRecordId(1.5)).toThrow();
    expect(toRecordId('42')).toBe(42);
  });

  it('merges metadata, removes null keys and blocks prototype pollution', () => {
    expect(mergeMetadata({ a: 1, b: 2 }, { b: null, c: 3 })).toEqual({ a: 1, c: 3 });
    expect(() => mergeMetadata({}, JSON.parse('{"__proto__":{"polluted":true}}'))).toThrow(/forbidden key/);
    expect(() => mergeMetadata({}, { nested: JSON.parse('{"constructor":{"x":1}}') })).toThrow(/forbidden key/);
    expect(({} as any).polluted).toBeUndefined();
  });

  it('checks the size of the merged metadata, not just the patch', () => {
    const big = 'x'.repeat(40_000);
    expect(() => mergeMetadata({ a: big }, { b: big })).toThrow(/exceeds/);
  });
});

describe('ReflexionMemory CRUD', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let reflexion: ReflexionMemory;

  beforeEach(async () => {
    db = openDb(false);
    embedder = await mockEmbedder();
    reflexion = new ReflexionMemory(db, embedder);
  });
  afterEach(() => db.close());

  const store = (task: string, extra: Record<string, unknown> = {}) =>
    reflexion.storeEpisode({ sessionId: 's1', task, reward: 0.5, success: true, ...extra } as any);

  it('gets an episode by id, without the embedding unless asked', async () => {
    const id = await store('alpha task', { tags: ['a'], metadata: { k: 1 } });
    const ep = reflexion.getEpisode(id)!;
    expect(ep.task).toBe('alpha task');
    expect(ep.tags).toEqual(['a']);
    expect(ep.embedding).toBeUndefined();
    expect(reflexion.getEpisode(id, { includeEmbedding: true })!.embedding).toHaveLength(384);
    expect(reflexion.getEpisode(99_999)).toBeNull();
    expect(() => reflexion.getEpisode('12-notes')).toThrow(CrudValidationError);
  });

  it('lists with paging, totals, ordering and a session filter', async () => {
    for (let i = 0; i < 7; i++) await store(`task ${i}`);
    await reflexion.storeEpisode({ sessionId: 's2', task: 'other', reward: 0, success: false });
    const page = reflexion.listEpisodes({ sessionId: 's1', limit: 3, offset: 0 });
    expect(page.total).toBe(7);
    expect(page.items.map(e => e.task)).toEqual(['task 6', 'task 5', 'task 4']);
    expect(reflexion.listEpisodes({ sessionId: 's1', limit: 3, offset: 6 }).items).toHaveLength(1);
    expect(reflexion.listEpisodes({ order: 'oldest', limit: 1 }).items[0].task).toBe('task 0');
    expect(reflexion.listEpisodes({ success: false }).total).toBe(1);
    expect(() => reflexion.listEpisodes({ limit: 10_000 })).toThrow(/limit/);
  });

  it('does not call the embedder for a metadata, tag or reward update', async () => {
    const id = await store('keep my vector', { metadata: { a: 1, b: 2 } });
    const before = reflexion.getEpisode(id, { includeEmbedding: true })!.embedding!;
    const spy = vi.spyOn(embedder, 'embedPassage');
    const res = (await reflexion.updateEpisode(id, { metadata: { b: null, c: 3 }, tags: ['x', 'x', ' y '], reward: 0.9 }))!;
    expect(spy).not.toHaveBeenCalled();
    expect(res.reembedded).toBe(false);
    expect(res.episode.metadata).toEqual({ a: 1, c: 3 });
    expect(res.episode.tags).toEqual(['x', 'y']);
    expect(res.episode.reward).toBe(0.9);
    expect(reflexion.getEpisode(id, { includeEmbedding: true })!.embedding).toEqual(before);
  });

  it('re-embeds on a text change and the keyword index serves the new text, not the old', async () => {
    for (let i = 0; i < 30; i++) await store(`routine upkeep item ${i}`);
    const id = await store('zorblax quarantine procedure');
    // Warm the BM25 index on the old text.
    expect((await reflexion.retrieveRelevant({ task: 'zorblax', k: 1 }))[0].id).toBe(id);
    const before = reflexion.getEpisode(id, { includeEmbedding: true })!.embedding!;

    const res = (await reflexion.updateEpisode(id, { task: 'quuxifier calibration procedure' }))!;
    expect(res.reembedded).toBe(true);
    expect(reflexion.getEpisode(id, { includeEmbedding: true })!.embedding).not.toEqual(before);
    expect(count(db, 'SELECT COUNT(*) AS c FROM episode_embeddings WHERE episode_id = ?', id)).toBe(1);

    const hits = await reflexion.retrieveRelevant({ task: 'quuxifier', k: 1 });
    expect(hits[0].id).toBe(id);
    expect(hits[0].task).toBe('quuxifier calibration procedure');
    const stale = await reflexion.retrieveRelevant({ task: 'zorblax', k: 3 });
    expect(stale.every(h => !h.task.includes('zorblax'))).toBe(true);
  });

  it('clears nullable fields with null and refuses immutable or unknown fields', async () => {
    const id = await store('t', { critique: 'meh', latencyMs: 5 });
    const res = (await reflexion.updateEpisode(id, { critique: null, latencyMs: null }))!;
    expect(res.episode.critique).toBeUndefined();
    expect(res.episode.latencyMs).toBeUndefined();
    await expect(reflexion.updateEpisode(id, { sessionId: 'x' } as any)).rejects.toThrow(/cannot be updated/);
    await expect(reflexion.updateEpisode(id, { id: 7 } as any)).rejects.toThrow(/cannot be updated/);
    await expect(reflexion.updateEpisode(id, { task: '   ' })).rejects.toThrow(/non-empty/);
    await expect(reflexion.updateEpisode(id, { tokensUsed: -1 })).rejects.toThrow(/non-negative/);
    await expect(reflexion.updateEpisode(id, { metadata: JSON.parse('{"__proto__":{"p":1}}') })).rejects.toThrow(/forbidden/);
    expect(await reflexion.updateEpisode(424242, { reward: 1 })).toBeNull();
  });

  it('refuses to store a vector computed from text that changed while it was embedding', async () => {
    const id = await store('original text');
    const real = embedder.embedPassage.bind(embedder);
    vi.spyOn(embedder, 'embedPassage').mockImplementationOnce(async (t: string) => {
      db.prepare('UPDATE episodes SET task = ? WHERE id = ?').run('someone else wrote this', id);
      return real(t);
    });
    await expect(reflexion.updateEpisode(id, { task: 'my edit' })).rejects.toThrow(/modified concurrently/);
    expect(reflexion.getEpisode(id)!.task).toBe('someone else wrote this');
  });

  it('keeps the vector backend in step on update and delete', async () => {
    const vb = fakeVectorBackend();
    const r = new ReflexionMemory(db, embedder, vb);
    const id = await r.storeEpisode({ sessionId: 's', task: 'vector one', reward: 1, success: true });
    const first = vb.store.get(String(id));
    await r.updateEpisode(id, { task: 'vector two' });
    expect(vb.removed).toContain(String(id));
    expect(vb.store.get(String(id))).not.toEqual(first);
    expect(await r.deleteEpisode(id)).toBe(true);
    expect(vb.store.has(String(id))).toBe(false);
  });

  it('delete leaves no orphaned embedding and never coerces a malformed id', async () => {
    const ids: number[] = [];
    for (let i = 0; i < 13; i++) ids.push(await store(`ep ${i}`));
    const twelve = ids[11];
    expect(await reflexion.deleteEpisode(`${twelve}-notes`)).toBe(false);
    expect(reflexion.getEpisode(twelve)).not.toBeNull();
    expect(await reflexion.deleteEpisode(twelve)).toBe(true);
    expect(count(db, 'SELECT COUNT(*) AS c FROM episode_embeddings WHERE episode_id = ?', twelve)).toBe(0);
  });

  it('prune removes the embeddings of pruned episodes', async () => {
    for (let i = 0; i < 3; i++) await store(`old ${i}`, { reward: 0 });
    db.prepare("UPDATE episodes SET ts = ts - 1000 * 86400").run();
    const pruned = reflexion.pruneEpisodes({ minReward: 0.3, maxAgeDays: 1, keepMinPerTask: 0 });
    expect(pruned).toBe(3);
    expect(count(db, 'SELECT COUNT(*) AS c FROM episode_embeddings')).toBe(0);
  });
});

describe('ReasoningBank CRUD', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;

  beforeEach(async () => {
    db = openDb(false);
    embedder = await mockEmbedder();
  });
  afterEach(() => db.close());

  it('updates content in place, re-embeds only on text change and never touches stats', async () => {
    const bank = new ReasoningBank(db, embedder);
    const id = await bank.storePattern({ taskType: 'debug', approach: 'bisect', successRate: 0.7, uses: 4, metadata: { a: 1 } });
    const spy = vi.spyOn(embedder, 'embedPassage');
    const meta = (await bank.updatePattern(id, { metadata: { b: 2 }, tags: ['t'] }))!;
    expect(spy).not.toHaveBeenCalled();
    expect(meta.pattern.metadata).toEqual({ a: 1, b: 2 });
    const text = (await bank.updatePattern(id, { approach: 'git bisect run' }))!;
    expect(spy).toHaveBeenCalledTimes(1);
    expect(text.reembedded).toBe(true);
    expect(text.pattern.successRate).toBe(0.7);
    expect(text.pattern.uses).toBe(4);
    await expect(bank.updatePattern(id, { successRate: 1 } as any)).rejects.toThrow(/cannot be updated/);
    expect(await bank.updatePattern(9999, { approach: 'x' })).toBeNull();
  });

  it('delete removes the pattern from the vector index too', async () => {
    const vb = fakeVectorBackend();
    const bank = new ReasoningBank(db, embedder, vb);
    const id = await bank.storePattern({ taskType: 'debug', approach: 'bisect', successRate: 0.7 });
    expect(vb.store.size).toBe(1);
    await bank.updatePattern(id, { approach: 'printf' });
    expect(vb.store.size).toBe(1);
    expect(bank.deletePattern(id)).toBe(true);
    expect(vb.store.size).toBe(0);
    expect(count(db, 'SELECT COUNT(*) AS c FROM pattern_embeddings')).toBe(0);
  });

  it('lists patterns with a task-type filter', async () => {
    const bank = new ReasoningBank(db, embedder);
    await bank.storePattern({ taskType: 'a', approach: '1', successRate: 0 });
    await bank.storePattern({ taskType: 'b', approach: '2', successRate: 0 });
    await bank.storePattern({ taskType: 'a', approach: '3', successRate: 0 });
    const page = bank.listPatterns({ taskType: 'a' });
    expect(page.total).toBe(2);
    expect(page.items.map(p => p.approach)).toEqual(['3', '1']);
  });
});

describe('SkillLibrary CRUD', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let skills: SkillLibrary;

  beforeEach(async () => {
    db = openDb(false);
    embedder = await mockEmbedder();
    skills = new SkillLibrary(db, embedder);
  });
  afterEach(() => db.close());

  const create = (name: string) => skills.createSkill({ name, description: `${name} skill`, successRate: 0.5, uses: 3 });

  it('gets, lists and updates a skill without touching usage stats', async () => {
    const id = await create('deploy');
    await create('rollback');
    expect(skills.getSkill(id)!.name).toBe('deploy');
    expect(skills.getSkill(777)).toBeNull();
    expect(skills.listSkills({ limit: 1 }).total).toBe(2);

    const spy = vi.spyOn(embedder, 'embedPassage');
    const codeOnly = (await skills.updateSkill(id, { code: 'kubectl apply' }))!;
    expect(spy).not.toHaveBeenCalled();
    expect(codeOnly.reembedded).toBe(false);
    const renamed = (await skills.updateSkill(id, { name: 'deploy-v2', description: 'blue/green deploy' }))!;
    expect(spy).toHaveBeenCalledTimes(1);
    expect(renamed.skill.name).toBe('deploy-v2');
    expect(renamed.skill.uses).toBe(3);
    await expect(skills.updateSkill(id, { uses: 99 } as any)).rejects.toThrow(/cannot be updated/);
  });

  it('rejects a rename onto an existing name with a validation error', async () => {
    const a = await create('a');
    await create('b');
    await expect(skills.updateSkill(a, { name: 'b' })).rejects.toThrow(/already exists/);
    expect(skills.getSkill(a)!.name).toBe('a');
  });

  it('delete removes the embedding, links in both directions and the vector entry', async () => {
    const vb = fakeVectorBackend();
    const lib = new SkillLibrary(db, embedder, vb);
    const a = await lib.createSkill({ name: 'a', successRate: 0 });
    const b = await lib.createSkill({ name: 'b', successRate: 0 });
    const c = await lib.createSkill({ name: 'c', successRate: 0 });
    lib.linkSkills({ parentSkillId: a, childSkillId: b, relationship: 'prerequisite', weight: 1 });
    lib.linkSkills({ parentSkillId: c, childSkillId: a, relationship: 'composition', weight: 1 });
    expect(lib.deleteSkill(a)).toBe(true);
    expect(lib.deleteSkill(a)).toBe(false);
    expect(count(db, 'SELECT COUNT(*) AS c FROM skill_embeddings WHERE skill_id = ?', a)).toBe(0);
    expect(count(db, 'SELECT COUNT(*) AS c FROM skill_links WHERE parent_skill_id = ? OR child_skill_id = ?', a, a)).toBe(0);
    expect(vb.store.has(`skill:${a}`)).toBe(false);
    expect(lib.getSkill(b)).not.toBeNull();
  });

  it('prune also removes the dependents of pruned skills', async () => {
    const id = await skills.createSkill({ name: 'stale', successRate: 0, uses: 0 });
    db.prepare('UPDATE skills SET created_at = created_at - 1000 * 86400 WHERE id = ?').run(id);
    expect(skills.pruneSkills({ minUses: 3, minSuccessRate: 0.4, maxAgeDays: 1 })).toBe(1);
    expect(count(db, 'SELECT COUNT(*) AS c FROM skill_embeddings')).toBe(0);
  });
});

describe('Bm25Index.replace (ADR-017 incremental keyword index)', () => {
  it('scores exactly like an index rebuilt from the edited corpus', async () => {
    const { Bm25Index } = await import('../../../src/search/conversation-retrieval.js');
    const docs = [['a', 'b', 'c'], ['b', 'b', 'd'], ['c', 'e'], ['a', 'f', 'f', 'g']];
    const live = new Bm25Index(docs);
    live.replace(1, docs[1], ['z', 'a']);
    live.replace(3, docs[3], ['b']);
    const edited = [docs[0], ['z', 'a'], docs[2], ['b']];
    const rebuilt = new Bm25Index(edited);
    for (const q of [['a'], ['b'], ['z'], ['f'], ['c', 'e', 'a']]) {
      expect(Array.from(live.score(q))).toEqual(Array.from(rebuilt.score(q)));
    }
    expect(() => live.replace(9, [], [])).toThrow(RangeError);
  });
});
