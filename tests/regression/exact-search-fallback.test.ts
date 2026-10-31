/**
 * ADR-015: exact-search fallback in searchVectors().
 *
 * @ruvector/core 0.1.30's HNSW misses neighbours on tiny graphs
 * (ruvnet/RuVector#1298: 14/150 trials at M=8). The controllers trusted the
 * index whenever it returned anything, so a stored vector could silently
 * never come back. ReflexionMemory and SkillLibrary only fell back to SQL
 * when the index returned *zero* usable hits; ReasoningBank never did (and in
 * v2 mode kept no SQL copy to fall back to).
 *
 * The stub index here returns real results but drops the best match, which
 * is what a partial HNSW miss looks like. Every assertion below fails on
 * 3.0.0.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { searchVectors, type ExactSearchSource, type SearchResult, type VectorBackend } from '../../src/backends/VectorBackend.js';
import { SkillLibrary } from '../../src/controllers/SkillLibrary.js';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory.js';
import { ReasoningBank } from '../../src/controllers/ReasoningBank.js';
import type { EmbeddingService } from '../../src/controllers/EmbeddingService.js';

const SCHEMA = fs.readFileSync(path.join(__dirname, '../../src/schemas/schema.sql'), 'utf-8');
const DIM = 8;

function unit(values: number[]): Float32Array {
  const v = new Float32Array(DIM);
  values.forEach((x, i) => (v[i] = x));
  return v;
}
function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

const QUERY = unit([1, 0]);
/** Token in the stored text -> its vector; alpha is the true nearest neighbour */
const VECTORS: Record<string, Float32Array> = {
  alpha: unit([0.99, 0.1]),
  beta: unit([0.9, 0.4]),
  gamma: unit([0.7, 0.7]),
  delta: unit([0.1, 1]),
};
const NAMES = Object.keys(VECTORS);
/** Names in exact-cosine order against QUERY */
const EXACT_ORDER = [...NAMES].sort((a, b) => cosine(QUERY, VECTORS[b]) - cosine(QUERY, VECTORS[a]));

function vectorFor(text: string): Float32Array {
  const name = NAMES.find((n) => text.includes(n));
  return name ? new Float32Array(VECTORS[name]) : new Float32Array(QUERY);
}
const embedder = {
  embedQuery: async (t: string) => vectorFor(t),
  embedPassage: async (t: string) => vectorFor(t),
  embed: async (t: string) => vectorFor(t),
} as unknown as EmbeddingService;

/**
 * Sync index that stores everything but never returns the vectors whose
 * metadata/text marks them as dropped, like an HNSW graph that cannot reach
 * a node.
 */
function lossyBackend(drop: (id: string, meta?: Record<string, unknown>) => boolean): VectorBackend & { stored: Map<string, any> } {
  const stored = new Map<string, { v: Float32Array; meta?: Record<string, unknown> }>();
  return {
    name: 'ruvector',
    stored,
    insert(id, v, meta) { stored.set(id, { v: new Float32Array(v), meta }); },
    insertBatch(items) { items.forEach((i) => stored.set(i.id, { v: new Float32Array(i.embedding), meta: i.metadata })); },
    search(q, k) {
      return [...stored]
        .filter(([id, e]) => !drop(id, e.meta))
        .map(([id, e]) => ({ id, similarity: cosine(q, e.v), distance: 1 - cosine(q, e.v), metadata: e.meta }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);
    },
    remove: (id) => stored.delete(id),
    getStats: () => ({ count: stored.size, dimension: DIM, metric: 'cosine', backend: 'ruvector', memoryUsage: 0 }),
    save: async () => {},
    load: async () => {},
    close() {},
  };
}

describe('searchVectors with an exact source', () => {
  const rows: Array<{ id: string; v: Float32Array }> = NAMES.map((n) => ({ id: n, v: VECTORS[n] }));
  const exactScan = (q: Float32Array, k: number): SearchResult[] =>
    rows
      .map(({ id, v }) => ({ id, similarity: cosine(q, v), distance: 1 - cosine(q, v) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  const source = (): ExactSearchSource & { scan: ReturnType<typeof vi.fn> } => ({
    count: () => rows.length,
    scan: vi.fn(exactScan),
  });
  const backendReturning = (hits: SearchResult[]): VectorBackend =>
    ({ name: 'ruvector', search: () => hits } as unknown as VectorBackend);

  it('answers a small collection exactly, recovering a dropped vector', async () => {
    const lossy = exactScan(QUERY, 4).filter((r) => r.id !== 'alpha');
    const got = await searchVectors(backendReturning(lossy), QUERY, 3, undefined, source());
    expect(got.map((r) => r.id)).toEqual(EXACT_ORDER.slice(0, 3));
  });

  it('keeps the index answer for a large collection that returned k hits', async () => {
    const hits = exactScan(QUERY, 2).reverse(); // index order is preserved as-is
    const src = source();
    const got = await searchVectors(backendReturning(hits), QUERY, 2, { exactSearchThreshold: 2 }, src);
    expect(got).toBe(hits);
    expect(src.scan).not.toHaveBeenCalled();
  });

  it('falls back on a large collection when the index comes back short', async () => {
    const src = source();
    const got = await searchVectors(backendReturning([exactScan(QUERY, 4)[3]]), QUERY, 3, { exactSearchThreshold: 2 }, src);
    expect(src.scan).toHaveBeenCalledOnce();
    expect(got.map((r) => r.id)).toEqual(EXACT_ORDER.slice(0, 3));
  });

  it('answers exactly when the index throws', async () => {
    const broken = { name: 'ruvector', search: () => { throw new Error('native index gone'); } } as unknown as VectorBackend;
    const got = await searchVectors(broken, QUERY, 2, undefined, source());
    expect(got.map((r) => r.id)).toEqual(EXACT_ORDER.slice(0, 2));
  });
});

describe('controllers on a lossy index', () => {
  let db: Database.Database;
  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(SCHEMA);
  });
  afterEach(() => db.close());

  it('SkillLibrary returns the stored skill the index dropped, in exact order', async () => {
    const backend = lossyBackend((_id, meta) => String(meta?.name).includes('alpha'));
    const library = new SkillLibrary(db as any, embedder, backend);
    for (const name of NAMES) {
      await library.createSkill({ name: `skill-${name}`, description: name, successRate: 0.9, uses: 1, avgReward: 0.5 });
    }
    const got = await library.retrieveSkills({ task: 'query', k: 4, minSuccessRate: 0 });
    expect(got.map((s) => s.name)).toEqual(EXACT_ORDER.map((n) => `skill-${n}`));
  });

  it('ReflexionMemory returns the stored episode the index dropped, in exact order', async () => {
    const ids = new Map<string, number>();
    const backend = lossyBackend((id) => id === String(ids.get('alpha')));
    const memory = new ReflexionMemory(db as any, embedder, backend);
    for (const name of NAMES) {
      ids.set(name, await memory.storeEpisode({ sessionId: 's', task: `task ${name}`, reward: 1, success: true }));
    }
    const got = await memory.retrieveRelevant({ task: 'query', k: 4 });
    expect(got.map((e) => e.task)).toEqual(EXACT_ORDER.map((n) => `task ${n}`));
  });

  it('ReasoningBank (v2) returns the stored pattern the index dropped, in exact order', async () => {
    const backend = lossyBackend((_id, meta) => meta?.taskType === 'alpha');
    const bank = new ReasoningBank(db as any, embedder, backend);
    for (const name of NAMES) {
      await bank.storePattern({ taskType: name, approach: `approach ${name}`, successRate: 0.5 });
    }
    // rerank off: this test is about recall, not reward weighting
    const got = await bank.searchPatterns({ taskEmbedding: QUERY, k: 4, rerank: false } as any);
    expect(got.map((p) => p.taskType)).toEqual(EXACT_ORDER);
  });
});

/**
 * Same check against the real backends, in a child process because the
 * `ruvector` package reads RUVECTOR_BACKEND when it is first imported. With
 * the default native HNSW on 3.0.0 this missed 32 of 225 self-lookups on
 * 3..12-vector collections; RUVECTOR_BACKEND=rvf makes the factory fall
 * through to the RVF backend.
 */
describe('tiny collections on the real vector backends', () => {
  const ROOT = path.resolve(__dirname, '../..');
  const TSX = path.join(ROOT, 'node_modules/.bin/tsx');

  async function runChild(env: Record<string, string>): Promise<{ backend: string; trials: number; misses: number }> {
    const { spawn } = await import('node:child_process');
    const os = await import('node:os');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-exact-'));
    const script = path.join(dir, 'child.mts');
    fs.writeFileSync(script, `
      import Database from '${ROOT}/node_modules/better-sqlite3/lib/index.js';
      import * as fs from 'node:fs';
      import { createBackend } from '${ROOT}/src/backends/factory.ts';
      import { SkillLibrary } from '${ROOT}/src/controllers/SkillLibrary.ts';
      const schema = fs.readFileSync('${ROOT}/src/schemas/schema.sql', 'utf-8');
      let seed = 3;
      const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff * 2 - 1; };
      const vectors = new Map();
      const vectorFor = (text) => vectors.get(text.split('\\n')[0]);
      const embedder = { embedQuery: async (t) => vectorFor(t), embedPassage: async (t) => vectorFor(t) };
      let trials = 0, misses = 0, backendName = '';
      for (let t = 0; t < 30; t++) {
        const n = 3 + (t % 10);
        const backend = await createBackend('auto', { dimensions: 64, metric: 'cosine', M: 8, storagePath: '${dir}/t' + t + '.idx' });
        backendName = backend.constructor.name;
        const db = new Database(':memory:');
        db.exec(schema);
        const library = new SkillLibrary(db, embedder, backend, undefined, { enabled: false });
        for (let i = 0; i < n; i++) {
          vectors.set('s' + t + '_' + i, Float32Array.from({ length: 64 }, rnd));
          await library.createSkill({ name: 's' + t + '_' + i, successRate: 0.9, uses: 1, avgReward: 0.5 });
        }
        for (let i = 0; i < n; i++) {
          trials++;
          const got = await library.retrieveSkills({ task: 's' + t + '_' + i, k: n, minSuccessRate: 0 });
          if (got.length !== n || got[0].name !== 's' + t + '_' + i) misses++;
        }
        backend.close();
        db.close();
      }
      process.stderr.write('RESULT ' + JSON.stringify({ backend: backendName, trials, misses }) + '\\n');
    `);
    try {
      const child = spawn(TSX, [script], {
        cwd: dir,
        env: { ...process.env, AGENTDB_DISABLE_TRANSFORMERS: '1', ...env },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stderr = '';
      child.stderr.on('data', (d) => { stderr += d; });
      const code = await new Promise<number | null>((resolve) => child.once('exit', resolve));
      const line = stderr.split('\n').find((l) => l.startsWith('RESULT '));
      if (code !== 0 || !line) throw new Error(`child failed (${code}):\n${stderr}`);
      return JSON.parse(line.slice('RESULT '.length));
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }

  it('returns every stored skill for its own embedding (default backend)', async () => {
    const result = await runChild({});
    expect(result.trials).toBe(225);
    expect(result.misses).toBe(0);
  }, 120_000);

  it('returns every stored skill for its own embedding (RUVECTOR_BACKEND=rvf)', async () => {
    const result = await runChild({ RUVECTOR_BACKEND: 'rvf' });
    expect(result.trials).toBe(225);
    expect(result.misses).toBe(0);
  }, 120_000);
});
