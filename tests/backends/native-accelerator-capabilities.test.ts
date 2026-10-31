/**
 * NativeAccelerator capability probes against the installed @ruvector packages.
 *
 * The probes looked for names the packages do not export: a static
 * TensorCompress.compress and SemanticRouter.load (both are instance
 * methods), GraphDatabase beginTransaction/cypher (the binding has
 * begin/query) and VectorDb.batchInsert (it is insertBatch). Every one of
 * those features therefore reported unavailable even when installed.
 */
import { describe, it, expect } from 'vitest';
import { createRequire } from 'module';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { NativeAccelerator } from '../../src/backends/rvf/NativeAccelerator.js';

const require = createRequire(import.meta.url);
const installed = (name: string): boolean => {
  try { require.resolve(name); return true; } catch { return false; }
};

describe('NativeAccelerator.capabilities()', () => {
  it.skipIf(!installed('@ruvector/core'))('reports core batch insert when @ruvector/core is installed', async () => {
    const accel = new NativeAccelerator();
    await accel.initialize();
    const caps = accel.capabilities();

    expect(caps.coreBatchInsert).toBe(true);
    expect(Object.values(caps).some(Boolean)).toBe(true);
  });

  it.skipIf(!installed('@ruvector/graph-node'))('reports graph tx/batch/cypher from graph-node begin/batchInsert/query', async () => {
    const accel = new NativeAccelerator();
    await accel.initialize();
    const caps = accel.capabilities();

    expect(caps.graphTx).toBe(true);
    expect(caps.graphBatchInsert).toBe(true);
    expect(caps.graphCypher).toBe(true);
  });

  it.skipIf(!installed('@ruvector/gnn') || !installed('@ruvector/router'))(
    'reports tensor compression and router persistence from instance methods',
    async () => {
      const accel = new NativeAccelerator();
      await accel.initialize();
      const caps = accel.capabilities();

      expect(caps.nativeTensorCompress).toBe(true);
      expect(caps.routerPersist).toBe(true);
    }
  );

  it.skipIf(!installed('@ruvector/gnn'))('round-trips a vector through native half-precision compression', async () => {
    const accel = new NativeAccelerator();
    await accel.initialize();
    const vec = Float32Array.from({ length: 32 }, (_, i) => Math.sin(i));

    const compressed = accel.tensorCompress(vec, 1);
    expect(compressed).toBeInstanceOf(Uint8Array);
    const restored = accel.tensorDecompress(compressed!, vec.length)!;
    expect(restored).toHaveLength(vec.length);
    for (let i = 0; i < vec.length; i++) expect(restored[i]).toBeCloseTo(vec[i], 2);
  });

  it.skipIf(!installed('@ruvector/core'))('coreBatchInsert writes through VectorDb.insertBatch', async () => {
    const accel = new NativeAccelerator();
    await accel.initialize();
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-accel-'));
    try {
      const { VectorDb } = require('@ruvector/core');
      const db = new VectorDb({
        dimensions: 4,
        storagePath: path.join(dir, 'core.db'),
        hnswConfig: { m: 8, efConstruction: 50, efSearch: 50, maxElements: 100 },
      });
      const ok = await accel.coreBatchInsert(db, [
        { id: 'a', vector: new Float32Array([1, 0, 0, 0]), metadata: { k: 1 } },
        { id: 'b', vector: new Float32Array([0, 1, 0, 0]) },
      ]);
      expect(ok).toBe(true);
      expect(await db.len()).toBe(2);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
