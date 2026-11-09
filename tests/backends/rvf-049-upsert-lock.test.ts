/**
 * RvfBackend on the real @ruvector/rvf >= 0.4.10 / rvf-node >= 0.4.5.
 *
 * - Re-ingesting a live id keeps the HNSW index (rvf 0.4.7 dropped it, so the
 *   next query rebuilt the graph: 1.7 s at 2,000 x 384).
 * - In-place upsert re-links the node (rvf 0.4.10), so the controllers no
 *   longer remove() before re-inserting an id on RVF. On the same churn
 *   (3,000 x 384, 300 updates) in-place reaches recall@10 1.0 where
 *   remove + insert reaches 0.9955 (0.4.9: in-place 0.9675).
 * - An upsert does not count as a new vector in getStats().
 * - A writer that was SIGKILLed no longer holds the lock (0.4.7 kept it 30 s).
 * - rvf 0.4.10 applies a batch that names an id twice last-write-wins (0.4.9
 *   rejected it with InvalidManifest). RvfBackend still collapses repeats
 *   itself, so the sync write queue never wedges on an older rvf.
 * - remove() must not let a queued write resurrect the id.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';
import { closeRvfBackendsAfterEach } from './helpers/close-rvf-backends.js';

closeRvfBackendsAfterEach();

const D = 384;
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32 - 0.5;
function unit(): Float32Array {
  const v = new Float32Array(D);
  let s = 0;
  for (let i = 0; i < D; i++) { v[i] = rnd(); s += v[i] * v[i]; }
  s = Math.sqrt(s);
  for (let i = 0; i < D; i++) v[i] /= s;
  return v;
}
const axis = (i: number, dim = 8) => { const v = new Float32Array(dim); v[i] = 1; return v; };

async function memStore(dim = D): Promise<RvfBackend> {
  const b = new RvfBackend({ dimension: dim, metric: 'cosine', storagePath: ':memory:', batchThreshold: 1000 });
  await b.initialize();
  return b;
}

async function fill(b: RvfBackend, vs: Map<string, Float32Array>): Promise<void> {
  const all = [...vs].map(([id, embedding]) => ({ id, embedding }));
  for (let i = 0; i < all.length; i += 500) await b.insertBatchAsync(all.slice(i, i + 500));
}

function exactTop(vs: Map<string, Float32Array>, q: Float32Array, k: number): string[] {
  return [...vs]
    .map(([id, v]) => { let d = 0; for (let i = 0; i < D; i++) d += v[i] * q[i]; return [id, d] as const; })
    .sort((a, b) => b[1] - a[1]).slice(0, k).map(([id]) => id);
}

let tmpDir: string;
beforeAll(() => { tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rvf049-')); });
afterAll(() => { fs.rmSync(tmpDir, { recursive: true, force: true }); });

describe('rvf 0.4.10: upsert of a live id', () => {
  it('the query after an upsert does not rebuild the graph', async () => {
    const vs = new Map<string, Float32Array>();
    for (let i = 0; i < 2000; i++) vs.set(`v${i}`, unit());
    const b = await memStore();
    await fill(b, vs);
    await b.searchAsync(unit(), 10); // build the graph

    const moved = unit();
    await b.insertAsync('v7', moved);
    const t0 = performance.now();
    const hits = await b.searchAsync(moved, 10);
    const ms = performance.now() - t0;

    expect(hits[0].id).toBe('v7');
    expect(new Set(hits.map((h) => h.id)).size).toBe(hits.length);
    expect((await b.getStatsAsync()).count).toBe(2000);
    // rvf 0.4.7: ~1,700 ms (full rebuild). 0.4.9: < 1 ms.
    expect(ms).toBeLessThan(250);
  }, 60_000);

  it('recall@10 after churn: in-place upsert (what the controllers do on RVF) is not worse than remove + insert', async () => {
    const N = 3000, CHURN = 300, Q = 200;
    const base = new Map<string, Float32Array>();
    for (let i = 0; i < N; i++) base.set(`v${i}`, unit());
    const churn = Array.from({ length: CHURN }, (_, k) => [`v${(k * 37) % N}`, unit()] as const);
    const queries = Array.from({ length: Q }, unit);

    const latest = new Map(base);
    for (const [id, v] of churn) latest.set(id, v);

    const upsert = await memStore();
    const reinsert = await memStore();
    await fill(upsert, base);
    await fill(reinsert, base);
    await upsert.searchAsync(queries[0], 10);
    await reinsert.searchAsync(queries[0], 10);
    for (const [id, v] of churn) {
      await upsert.insertAsync(id, v);
      await reinsert.removeAsync(id);
      await reinsert.insertAsync(id, v);
    }

    async function recall(b: RvfBackend): Promise<number> {
      let hit = 0;
      for (const q of queries) {
        const ids = (await b.searchAsync(q, 10)).map((h) => h.id);
        expect(new Set(ids).size).toBe(ids.length);
        const truth = exactTop(latest, q, 10);
        hit += ids.filter((id) => truth.includes(id)).length;
      }
      return hit / (Q * 10);
    }
    const rUpsert = await recall(upsert);
    const rReinsert = await recall(reinsert);
    expect((await upsert.getStatsAsync()).count).toBe(N);
    expect((await reinsert.getStatsAsync()).count).toBe(N);
    // rvf 0.4.9: upsert 0.9675, remove + insert 0.9955. 0.4.10: 1.0 and 0.9955.
    expect(rUpsert).toBeGreaterThanOrEqual(rReinsert);
    expect(rUpsert).toBeGreaterThanOrEqual(0.99);
    console.log(`[rvf-0410] recall@10 after ${CHURN} updates: upsert ${rUpsert}, remove+insert ${rReinsert}`);
  }, 120_000);

  it('an upsert does not count as a new vector', async () => {
    // rvf reports an upsert of a stored id as accepted: 1. Summing that made
    // the sync getStats() grow by one per update (5 for 2 vectors here).
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    await b.insertAsync('y', axis(1));
    b.insert('x', axis(4));
    await b.flush();
    await b.insertAsync('y', axis(5));
    await b.insertBatchAsync([{ id: 'x', embedding: axis(6) }]);
    expect(b.getStats().count).toBe(2);
    expect((await b.getStatsAsync()).count).toBe(2);
  });
});

describe('rvf 0.4.10: a batch that names an id twice', () => {
  it('is applied last-write-wins by rvf itself and does not poison the store', async () => {
    const { RvfDatabase } = await import('@ruvector/rvf');
    const file = path.join(tmpDir, 'dup-batch.rvf');
    const db = await RvfDatabase.create(file, { dimensions: 8, metric: 'cosine' });
    try {
      // rvf 0.4.9: rejected with InvalidManifest (0x0105) and nothing stored.
      await db.ingestBatch([{ id: 'x', vector: axis(1) }, { id: 'y', vector: axis(2) }, { id: 'x', vector: axis(5) }]);
      expect((await db.status()).totalVectors).toBe(2);
      const hits = await db.query(axis(5), 2);
      expect(hits[0].id).toBe('x');
      expect(hits[0].distance).toBeCloseTo(0, 5);
      await db.ingestBatch([{ id: 'z', vector: axis(3) }]);
      expect((await db.query(axis(3), 1))[0].id).toBe('z');
    } finally {
      await db.close();
    }
  });
});

describe('rvf 0.4.10: a SIGKILLed writer does not hold the lock', () => {
  it('reopens within 2 s of the writer being killed', async () => {
    const file = path.join(tmpDir, 'killed.rvf');
    const seedStore = new RvfBackend({ dimension: 8, metric: 'cosine', storagePath: file });
    await seedStore.initialize();
    await seedStore.insertAsync('a', axis(0));
    await seedStore.closeAsync();

    const child = spawn(process.execPath, ['--input-type=module', '-e', `
      import { RvfDatabase } from '@ruvector/rvf';
      const db = await RvfDatabase.open(${JSON.stringify(file)});
      console.log('READY');
      setInterval(() => {}, 1000);
    `], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'inherit'] });
    await new Promise<void>((resolve, reject) => {
      child.stdout!.on('data', (d) => { if (String(d).includes('READY')) resolve(); });
      child.on('exit', (code) => reject(new Error(`writer exited early (${code})`)));
    });
    child.kill('SIGKILL');
    await new Promise((r) => child.once('exit', r));

    const t0 = performance.now();
    let reopened: RvfBackend | null = null;
    while (performance.now() - t0 < 5000) {
      const b = new RvfBackend({ dimension: 8, metric: 'cosine', storagePath: file });
      try { await b.initialize(); reopened = b; break; } catch { await new Promise((r) => setTimeout(r, 50)); }
    }
    const ms = performance.now() - t0;
    expect(reopened).not.toBeNull();
    expect((await reopened!.getStatsAsync()).count).toBe(1);
    // rvf 0.4.7: 30,000 ms (lock kept until 30 s old). 0.4.9: ~10 ms.
    expect(ms).toBeLessThan(2000);
  }, 15_000);
});

describe('RvfBackend sync write queue', () => {
  it('the same id queued twice flushes, last write wins', async () => {
    const b = await memStore(8);
    b.insert('x', axis(1));
    b.insert('y', axis(2));
    b.insert('x', axis(5));
    await expect(b.flush()).resolves.toBeUndefined();
    const hits = await b.searchAsync(axis(5), 2);
    expect(hits[0].id).toBe('x');
    expect(hits[0].similarity).toBeCloseTo(1, 5);
    expect((await b.getStatsAsync()).count).toBe(2);
  });

  it('insertBatchAsync with a repeated id keeps the last one', async () => {
    const b = await memStore(8);
    await b.insertBatchAsync([{ id: 'x', embedding: axis(0) }, { id: 'x', embedding: axis(3) }]);
    const hits = await b.searchAsync(axis(3), 1);
    expect(hits[0].id).toBe('x');
    expect(hits[0].similarity).toBeCloseTo(1, 5);
  });

  it('remove() of a queued id is not undone by the flush', async () => {
    const b = await memStore(8);
    await b.insertAsync('keep', axis(0));
    b.insert('gone', axis(1));
    b.remove('gone');
    const ids = (await b.searchAsync(axis(1), 5)).map((h) => h.id);
    expect(ids).not.toContain('gone');
    expect((await b.getStatsAsync()).count).toBe(1);
  });

  it('remove() then insert() of a stored id leaves exactly the new copy', async () => {
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    b.remove('x');
    b.insert('x', axis(4));
    const hits = await b.searchAsync(axis(4), 3);
    expect(hits.filter((h) => h.id === 'x')).toHaveLength(1);
    expect(hits[0].id).toBe('x');
    expect(hits[0].similarity).toBeCloseTo(1, 5);
    expect((await b.getStatsAsync()).count).toBe(1);
  });

  it('remove() then insert() while a flush is in flight keeps the new copy', async () => {
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    b.insert('filler', axis(1));
    const inFlight = b.flush();
    b.remove('x');
    b.insert('x', axis(4));
    await inFlight;
    const hits = await b.searchAsync(axis(4), 3);
    expect(hits[0]?.id).toBe('x');
    expect(hits[0].similarity).toBeCloseTo(1, 5);
    expect(hits.filter((h) => h.id === 'x')).toHaveLength(1);
  });

  it('remove() then insertAsync() keeps the new copy', async () => {
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    b.remove('x');
    await b.insertAsync('x', axis(4));
    const hits = await b.searchAsync(axis(4), 3);
    expect(hits[0]?.id).toBe('x');
    expect(hits[0].similarity).toBeCloseTo(1, 5);
  });

  it('remove() then insertBatchAsync() keeps the new copy', async () => {
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    b.remove('x');
    await b.insertBatchAsync([{ id: 'x', embedding: axis(4) }]);
    const hits = await b.searchAsync(axis(4), 3);
    expect(hits[0]?.id).toBe('x');
  });

  it('removeAsync of a stored id deletes it before the next search', async () => {
    const b = await memStore(8);
    await b.insertAsync('x', axis(0));
    b.remove('x');
    const ids = (await b.searchAsync(axis(0), 3)).map((h) => h.id);
    expect(ids).not.toContain('x');
    await expect(b.removeAsync('x')).resolves.toBe(false);
  });
});
