/**
 * RvfBackend against the real @ruvector/rvf >= 0.4.7 native backend (no mocks).
 *
 * 0.4.7 stores per-vector metadata durably and resolves filter leaves by field
 * name. Before it, agentdb stayed on 0.2.3, which dropped metadata and threw on
 * every filtered query ("filter requires string field 'valueType'").
 *
 * The legacy fixture was written by agentdb 3.6.1 + @ruvector/rvf 0.2.3:
 * 50 batch inserts `doc-i` and 10 sync inserts `sync-i`, dimension 8, cosine,
 * vector i = sin(i*7 + j). See tests/fixtures/rvf-0.2.3.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import {
  RvfBackend,
  RvfMetadataTypeConflictError,
  RVF_METADATA_TYPE_CONFLICT,
  minEfSearch,
} from '../../src/backends/rvf/RvfBackend.js';

const DIM = 8;
const vec = (i: number) => new Float32Array(DIM).map((_, j) => Math.sin(i * 7 + j));
const FIXTURE_DIR = path.join(__dirname, '..', 'fixtures', 'rvf-0.2.3');

let tmpDir: string;
const open: RvfBackend[] = [];

async function backendAt(file: string): Promise<RvfBackend> {
  const b = new RvfBackend({ dimension: DIM, metric: 'cosine', storagePath: file });
  await b.initialize();
  open.push(b);
  return b;
}

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rvf04-'));
});

afterEach(async () => {
  for (const b of open.splice(0)) {
    if (b.isInitialized()) await b.closeAsync().catch(() => undefined);
  }
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('RvfBackend on @ruvector/rvf 0.4.7: metadata', () => {
  it('round-trips metadata through insert, close and reopen', async () => {
    const file = path.join(tmpDir, 'meta.rvf');
    const b = await backendAt(file);
    await b.insertAsync('a', vec(1), { kind: 'note', n: 7, tags: ['x', 'y'], nested: { ok: true } });
    await b.insertBatchAsync([
      { id: 'b', embedding: vec(2), metadata: { kind: 'task', n: 8 } },
      { id: 'c', embedding: vec(3), metadata: { kind: 'note', n: 9 } },
    ]);
    b.insert('d', vec(4), { kind: 'sync', n: 10 });
    await b.closeAsync(); // flushes the queued sync insert

    const r = await backendAt(file);
    expect((await r.getStatsAsync()).count).toBe(4);
    const hits = await r.searchAsync(vec(1), 4, { forceExact: true });
    const byId = Object.fromEntries(hits.map((h) => [h.id, h.metadata]));
    expect(byId.a).toEqual({ kind: 'note', n: 7, tags: ['x', 'y'], nested: { ok: true } });
    expect(byId.b).toEqual({ kind: 'task', n: 8 });
    expect(byId.d).toEqual({ kind: 'sync', n: 10 });
  });

  it('filters by field name, including after reopen', async () => {
    const file = path.join(tmpDir, 'filter.rvf');
    const b = await backendAt(file);
    await b.insertBatchAsync([...Array(20)].map((_, i) => ({
      id: `v${i}`,
      embedding: vec(i),
      metadata: { kind: i % 2 ? 'odd' : 'even', n: i, flag: i % 3 === 0 },
    })));
    await b.closeAsync();

    const r = await backendAt(file);
    const ids = async (filter: Record<string, unknown>) =>
      (await r.searchAsync(vec(0), 20, { filter, forceExact: true })).map((h) => h.id).sort();

    expect(await ids({ kind: 'odd' })).toHaveLength(10);
    expect((await ids({ kind: 'odd' })).every((id) => Number(id.slice(1)) % 2 === 1)).toBe(true);
    // $range is inclusive on both ends (rvf's native range excludes high)
    expect(await ids({ n: { $range: [3, 6] } })).toEqual(['v3', 'v4', 'v5', 'v6']);
    expect(await ids({ n: { $in: [1, 2] } })).toEqual(['v1', 'v2']);
    expect(await ids({ flag: true })).toHaveLength(7);
    expect(await ids({ kind: 'even', n: { $gt: 14 } })).toEqual(['v16', 'v18']);
    // Unknown field and type-mismatched literal match nothing (no throw)
    expect(await ids({ missing: 'x' })).toEqual([]);
    expect(await ids({ kind: 5 })).toEqual([]);
    // A raw RvfFilterExpr passes through unchanged
    const raw = await r.searchAsync(vec(0), 20, { filter: { op: 'eq', field: 'n', value: 4 } });
    expect(raw.map((h) => h.id)).toEqual(['v4']);
    // The HNSW path (no forceExact) honours the filter too
    const approx = await r.searchAsync(vec(0), 5, { filter: { kind: 'even' } });
    expect(approx.length).toBeGreaterThan(0);
    expect(approx.every((h) => (h.metadata as { kind: string }).kind === 'even')).toBe(true);
  });

  it('deleteByFilter resolves fields by name', async () => {
    const b = await backendAt(path.join(tmpDir, 'del.rvf'));
    await b.insertBatchAsync([...Array(6)].map((_, i) => ({
      id: `v${i}`, embedding: vec(i), metadata: { kind: i < 2 ? 'gone' : 'kept' },
    })));
    const res = await b.deleteByFilter({ kind: 'gone' });
    expect(res.deleted).toBe(2);
    const left = await b.searchAsync(vec(0), 10, { forceExact: true });
    expect(left.map((h) => h.id).sort()).toEqual(['v2', 'v3', 'v4', 'v5']);
  });
});

describe('RvfBackend on @ruvector/rvf 0.4.7: metadata type conflicts', () => {
  it('insertAsync throws RvfMetadataTypeConflictError naming the field', async () => {
    const b = await backendAt(path.join(tmpDir, 'conflict.rvf'));
    await b.insertAsync('a', vec(1), { n: 1 });
    const err = await b.insertAsync('b', vec(2), { n: 'one' }).catch((e) => e);
    expect(err).toBeInstanceOf(RvfMetadataTypeConflictError);
    expect(err.code).toBe(RVF_METADATA_TYPE_CONFLICT);
    expect(err.field).toBe('n');
    expect(err.ids).toEqual(['b']);
    expect(err.message).toMatch(/field "n"/);
    expect((await b.getStatsAsync()).count).toBe(1);
  });

  it('a conflicting batch is not partially applied', async () => {
    const b = await backendAt(path.join(tmpDir, 'batch.rvf'));
    await b.insertAsync('a', vec(1), { n: 1 });
    const err = await b.insertBatchAsync([
      { id: 'g1', embedding: vec(2), metadata: { n: 2 } },
      { id: 'bad', embedding: vec(3), metadata: { n: 'x' } },
    ]).catch((e) => e);
    expect(err).toBeInstanceOf(RvfMetadataTypeConflictError);
    expect(err.ids).toEqual(['g1', 'bad']);
    expect((await b.getStatsAsync()).count).toBe(1);
  });

  it('queued sync writes: good entries land, conflicting ones are kept, flush throws', async () => {
    const b = new RvfBackend({ dimension: DIM, storagePath: path.join(tmpDir, 'queue.rvf'), batchThreshold: 1000 });
    await b.initialize();
    open.push(b);
    b.insert('a', vec(1), { n: 1 });
    b.insert('bad', vec(2), { n: 'two' });
    b.insert('c', vec(3), { n: 3 });

    const err = await b.flush().catch((e) => e);
    expect(err).toBeInstanceOf(RvfMetadataTypeConflictError);
    expect(err.ids).toEqual(['bad']);

    const hits = await b.searchAsync(vec(1), 10, { forceExact: true });
    expect(hits.map((h) => h.id).sort()).toEqual(['a', 'c']);

    const rejected = b.takeRejectedWrites();
    expect(rejected).toHaveLength(1);
    expect(rejected[0].id).toBe('bad');
    expect(rejected[0].metadata).toEqual({ n: 'two' });
    expect(Array.from(rejected[0].vector)).toEqual(Array.from(vec(2)));
    // Queue is clear: the next flush does not re-raise
    await expect(b.flush()).resolves.toBeUndefined();
    expect(b.getRejectedWrites()).toHaveLength(0);
  });
});

describe('RvfBackend on @ruvector/rvf 0.4.7: close and locking', () => {
  it('a second writer gets a clear LockHeld error', async () => {
    const file = path.join(tmpDir, 'lock.rvf');
    await backendAt(file);
    const second = new RvfBackend({ dimension: DIM, storagePath: file });
    await expect(second.initialize()).rejects.toThrow(/locked by another writer/);
  });

  it('reopening right after sync close() waits for the lock to be released', async () => {
    const file = path.join(tmpDir, 'reopen.rvf');
    const b = await backendAt(file);
    await b.insertAsync('a', vec(1), { n: 1 });
    b.close();
    const r = await backendAt(file);
    expect((await r.getStatsAsync()).count).toBe(1);
  });

  it('openReadonly works while a writer is open', async () => {
    const file = path.join(tmpDir, 'ro.rvf');
    const w = await backendAt(file);
    await w.insertAsync('a', vec(1), { kind: 'x' });
    const ro = await RvfBackend.openReadonly(file, { dimension: DIM, metric: 'cosine' });
    open.push(ro);
    const hits = await ro.searchAsync(vec(1), 1);
    expect(hits[0].id).toBe('a');
  });
});

describe('RvfBackend on @ruvector/rvf 0.4.7: store written by agentdb 3.6.1 / rvf 0.2.3', () => {
  function copyFixture(): string {
    const file = path.join(tmpDir, 'legacy.rvf');
    fs.copyFileSync(path.join(FIXTURE_DIR, 'legacy.rvf'), file);
    fs.copyFileSync(path.join(FIXTURE_DIR, 'legacy.rvf.idmap.json'), `${file}.idmap.json`);
    return file;
  }

  it('opens and every vector is readable by its original string id', async () => {
    const b = await backendAt(copyFixture());
    expect((await b.getStatsAsync()).count).toBe(60);
    for (const id of ['doc-0', 'doc-3', 'doc-49', 'sync-50', 'sync-59']) {
      const i = Number(id.split('-')[1]);
      const [top] = await b.searchAsync(vec(i), 1, { forceExact: true });
      expect(top.id).toBe(id);
      expect(top.distance).toBeCloseTo(0, 5);
      // rvf 0.2.3 never stored metadata, so legacy hits carry none
      expect(top.metadata).toBeUndefined();
    }
  });

  it('accepts new writes with metadata and filters them; legacy rows stay', async () => {
    const file = copyFixture();
    const b = await backendAt(file);
    await b.insertAsync('new-1', vec(100), { kind: 'fresh', n: 1 });
    await b.closeAsync();

    const r = await backendAt(file);
    expect((await r.getStatsAsync()).count).toBe(61);
    const hits = await r.searchAsync(vec(100), 5, { filter: { kind: 'fresh' }, forceExact: true });
    expect(hits.map((h) => h.id)).toEqual(['new-1']);
    expect(hits[0].metadata).toEqual({ kind: 'fresh', n: 1 });
    const [legacy] = await r.searchAsync(vec(7), 1, { forceExact: true });
    expect(legacy.id).toBe('doc-7');
  });

  it('documents the 0.2.x metric caveat: the reopened store ranks by L2', async () => {
    // rvf 0.2.x did not persist the metric; 0.2.3 itself reopens these files
    // as L2, and so does 0.4.7. For doc-3 vs doc-38 cosine distance is 0.00103,
    // squared L2 is 0.00812. Stores created with rvf >= 0.4 keep cosine.
    const b = await backendAt(copyFixture());
    const hits = await b.searchAsync(vec(3), 3, { forceExact: true });
    const d38 = hits.find((h) => h.id === 'doc-38');
    expect(d38?.distance).toBeCloseTo(0.00812, 4);

    const fresh = await backendAt(path.join(tmpDir, 'fresh.rvf'));
    await fresh.insertBatchAsync([3, 38].map((i) => ({ id: `doc-${i}`, embedding: vec(i) })));
    await fresh.closeAsync();
    const reopened = await backendAt(path.join(tmpDir, 'fresh.rvf'));
    const fh = await reopened.searchAsync(vec(3), 2, { forceExact: true });
    expect(fh.find((h) => h.id === 'doc-38')?.distance).toBeCloseTo(0.00103, 4);
  });
});

describe('minEfSearch (dimension-aware ef floor, RuVector #1401)', () => {
  it('is 256 up to 128 dims and grows with sqrt(dim/128) above', () => {
    expect(minEfSearch(8)).toBe(256);
    expect(minEfSearch(128)).toBe(256);
    expect(minEfSearch(256)).toBe(363);
    expect(minEfSearch(384)).toBe(444);
    expect(minEfSearch(768)).toBe(628);
    expect(minEfSearch(1536)).toBe(887);
  });
});
