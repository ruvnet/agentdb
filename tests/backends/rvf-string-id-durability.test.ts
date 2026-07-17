/**
 * Regression test for issue #13: AgentDB 3.0.0-alpha.17 pinned
 * `@ruvector/rvf@^0.1.9`, whose Node backend converted external IDs with
 * `Number(e.id)` — arbitrary strings became `NaN` and collided at native ID
 * 0, silently dropping data. Fixed by bumping the dependency to `^0.2.3`
 * (RVF's persistent string-to-i64 label map).
 *
 * This exercises the real `@ruvector/rvf` native backend (no mocks) with
 * the exact ID shapes from the issue report: SKU-style, purely numeric
 * strings, and hex/UUID-like IDs — insert, self-query, close, and reopen.
 */
import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { SelfLearningRvfBackend } from '../../src/backends/rvf/SelfLearningRvfBackend.js';

const DIM = 4;
const IDS = ['SKU-alpha', 'SKU-beta', '12345', 'e17e8d3bddb7057a1fcef6e3489b1b85'];

function orthogonalVector(index: number): Float32Array {
  const v = new Float32Array(DIM);
  v[index % DIM] = 1;
  return v;
}

describe('Issue #13: arbitrary string IDs survive ingest, self-query, and reopen', () => {
  let tmpDir: string;

  afterEach(() => {
    if (tmpDir) fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('stores all 4 vectors (not 2) and each self-query returns its own ID', async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-id-durability-'));
    const storagePath = path.join(tmpDir, 'test.rvf');

    const backend = await SelfLearningRvfBackend.create({
      dimension: DIM,
      storagePath,
      learning: false,
    });

    for (let i = 0; i < IDS.length; i++) {
      await backend.insertAsync(IDS[i], orthogonalVector(i));
    }

    const stats = await backend.getStatsAsync();
    expect(stats.count).toBe(IDS.length);

    for (let i = 0; i < IDS.length; i++) {
      const results = await backend.searchAsync(orthogonalVector(i), 1);
      expect(results.length).toBe(1);
      expect(results[0].id).toBe(IDS[i]);
    }

    backend.close();
  });

  it('preserves all IDs across close and reopen', async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-id-durability-reopen-'));
    const storagePath = path.join(tmpDir, 'test.rvf');

    const backend = await SelfLearningRvfBackend.create({
      dimension: DIM,
      storagePath,
      learning: false,
    });
    for (let i = 0; i < IDS.length; i++) {
      await backend.insertAsync(IDS[i], orthogonalVector(i));
    }
    backend.close();

    // Reopen from disk (RvfBackend.initialize() detects the existing file
    // and calls RvfDatabase.open() rather than create()).
    const reopened = await SelfLearningRvfBackend.create({
      dimension: DIM,
      storagePath,
      learning: false,
    });

    const stats = await reopened.getStatsAsync();
    expect(stats.count).toBe(IDS.length);

    for (let i = 0; i < IDS.length; i++) {
      const results = await reopened.searchAsync(orthogonalVector(i), 1);
      expect(results.length).toBe(1);
      expect(results[0].id).toBe(IDS[i]);
    }

    reopened.close();
  });

  it('does not collide purely-numeric-looking string IDs at native ID 0', async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-id-durability-numeric-'));
    const storagePath = path.join(tmpDir, 'test.rvf');

    const backend = await SelfLearningRvfBackend.create({
      dimension: DIM,
      storagePath,
      learning: false,
    });

    // Two different numeric-looking string IDs: under the old Number(e.id)
    // conversion these are both well-defined numbers (not NaN), but the bug
    // report's core claim is that mixing them with non-numeric IDs corrupted
    // the whole batch. Confirm both are distinguishable and independently
    // retrievable in the same store as non-numeric IDs.
    await backend.insertAsync('0', orthogonalVector(0));
    await backend.insertAsync('12345', orthogonalVector(1));
    await backend.insertAsync('SKU-gamma', orthogonalVector(2));

    const stats = await backend.getStatsAsync();
    expect(stats.count).toBe(3);

    const r0 = await backend.searchAsync(orthogonalVector(0), 1);
    expect(r0[0].id).toBe('0');
    const r1 = await backend.searchAsync(orthogonalVector(1), 1);
    expect(r1[0].id).toBe('12345');
    const r2 = await backend.searchAsync(orthogonalVector(2), 1);
    expect(r2[0].id).toBe('SKU-gamma');

    backend.close();
  });
});
