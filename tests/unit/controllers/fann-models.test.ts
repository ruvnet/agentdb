import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import initSqlJs from 'sql.js';
import NativeDatabase from 'better-sqlite3';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadRuntime } from 'ruv-fann';
import { FannModels, type DenseModel, type FannModelsOptions } from '../../../src/controllers/FannModels.js';
import { wrapExistingSqlJsDatabase } from '../../../src/db-fallback.js';
import type { IDatabaseConnection } from '../../../src/types/database.types.js';

const linear = (): DenseModel => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 },
] });
const data = () => ({ inputs: new Float64Array([-1, 0, 1, 2]), targets: new Float64Array([-1, 1, 3, 5]), rows: 4, epochs: 250, learningRate: 0.05, optimizer: 'adamw' as const });
let SQL: Awaited<ReturnType<typeof initSqlJs>>;
const resources: { dispose?: () => void; close?: () => void }[] = [];
const directories: string[] = [];
beforeAll(async () => {
  SQL = await initSqlJs();
});
afterEach(() => {
  for (const value of resources.splice(0).reverse()) { value.dispose?.(); value.close?.(); }
  for (const path of directories.splice(0)) rmSync(path, { recursive: true, force: true });
});
function db() {
  const connection = wrapExistingSqlJsDatabase(new SQL.Database()) as IDatabaseConnection & { close(): void };
  resources.push(connection);
  return connection;
}
function controller(connection = db(), options: FannModelsOptions = {}) {
  const models = new FannModels(connection, options);
  resources.push(models);
  return models;
}

describe('persisted FANN models with actual WASM', () => {
  it('trains real weights, decreases loss and predicts a batch after reopening sql.js', async () => {
    const raw = new SQL.Database();
    const connection = wrapExistingSqlJsDatabase(raw);
    resources.push(connection);
    const models = controller(connection);
    const created = await models.create('linear', JSON.stringify(linear()));
    expect(created.revision).toBe(1);
    const trained = await models.train('linear', data());
    expect(trained.finalLoss).toBeLessThan(trained.initialLoss * 0.001);
    expect(trained.backend).toMatch(/^wasm(?:-simd)?$/);
    const output = await models.predict('linear', new Float64Array([-1, 0, 1, 2]), 4);
    expect(output.outputs[2]).toBeCloseTo(3, 1);
    const bytes = raw.export();
    models.dispose();
    const restored = wrapExistingSqlJsDatabase(new SQL.Database(bytes));
    resources.push(restored);
    const reopened = controller(restored, { readonly: true });
    expect(reopened.get('linear')!.revision).toBe(2);
    expect(await reopened.predict('linear', new Float64Array([-1, 0, 1, 2]), 4)).toEqual(output);
    expect(reopened.list()).toEqual([expect.objectContaining({ id: 'linear', parameterCount: 2 })]);
    expect(reopened.list()[0]).not.toHaveProperty('model');
    await expect(reopened.train('linear', data())).rejects.toThrow('read-only');
    await expect(reopened.create('new', linear())).rejects.toThrow('read-only');
    expect(() => reopened.delete('linear')).toThrow('read-only');
  });

  it('invalidates another controller cache after training and delete/recreate with the same revision', async () => {
    const connection = db();
    const writer = controller(connection);
    const reader = controller(connection);
    await writer.create('shared', linear());
    expect((await reader.predict('shared', new Float64Array([1]))).outputs).toEqual([0]);
    await writer.train('shared', data());
    expect((await reader.predict('shared', new Float64Array([1]))).outputs[0]).toBeCloseTo(3, 1);
    expect(writer.delete('shared')).toBe(true);
    await expect(reader.predict('shared', new Float64Array([1]))).rejects.toThrow('not found');
    await writer.create('shared', linear());
    const first = await reader.predict('shared', new Float64Array([1]));
    writer.delete('shared');
    const changed = linear();
    (changed.layers[0].weights as number[])[0] = 7;
    await writer.create('shared', changed);
    const second = await reader.predict('shared', new Float64Array([1]));
    expect(first.revision).toBe(second.revision);
    expect(second.outputs).toEqual([7]);
  });

  it('leaves persisted state unchanged after malformed or unbounded work', async () => {
    const models = controller();
    await models.create('safe', linear());
    const before = models.get('safe');
    for (const patch of [
      { inputs: new Float64Array([NaN, 0, 1, 2]) },
      { targets: new Float64Array([1]) },
      { epochs: 1001 }, { rows: 129 }, { learningRate: Infinity },
      { optimizer: 'bogus' }, { rows: 128, epochs: 1000 },
    ]) {
      await expect(models.train('safe', { ...data(), ...patch } as any)).rejects.toThrow();
      expect(models.get('safe')).toEqual(before);
    }
    await expect(models.predict('safe', new Float64Array([Infinity]))).rejects.toThrow('finite');
    await expect(models.predict('safe', new Float64Array([1, 2]))).rejects.toThrow('length');
    await expect(models.create('bad id', linear())).rejects.toThrow('id');
    await expect(models.create('invalid', { ...linear(), schema: 'unknown' } as any)).rejects.toThrow('schema');
    await expect(models.create('invalid', { ...linear(), layers: [{ ...linear().layers[0], activation: 'unknown' }] } as any)).rejects.toThrow();
    expect(models.list()).toHaveLength(1);
    expect(models.get('safe')).toEqual(before);
  });

  it('accepts omitted bias as a bias-free dense layer', async () => {
    const models = controller();
    const model = linear();
    delete (model.layers[0] as any).bias;
    const record = await models.create('bias-free', model);
    expect(record.parameterCount).toBe(1);
    expect(record.model.layers[0].bias).toBeNull();
    expect((await models.predict('bias-free', new Float64Array([2]))).outputs).toEqual([0]);
  });

  it('snapshots caller inputs before async initialization and rejects shared memory', async () => {
    const models = controller();
    const model = linear();
    (model.layers[0].weights as number[])[0] = 2;
    await models.create('owned', model);
    const input = new Float64Array([3]);
    const pending = models.predict('owned', input);
    input[0] = 99;
    expect((await pending).outputs).toEqual([6]);
    await expect(models.predict('owned', new Float64Array(new SharedArrayBuffer(8)))).rejects.toThrow();
    await expect(models.train('owned', { ...data(), inputs: new Float64Array(new SharedArrayBuffer(32)) })).rejects.toThrow();
  });

  it('enforces the total training compute budget before touching runtime or checkpoint', async () => {
    const models = controller(db(), { maxTrainingCompute: 10 });
    await models.create('bounded', linear());
    await expect(models.train('bounded', data())).rejects.toThrow('compute budget');
    expect(models.get('bounded')!.revision).toBe(1);
  });

  it('preserves checkpoints on real WASM numerical failure', async () => {
    const models = controller();
    await models.create('overflow', linear());
    const before = models.get('overflow');
    await expect(models.train('overflow', {
      inputs: new Float64Array([1e308]), targets: new Float64Array([1e308]),
      rows: 1, epochs: 1, learningRate: 1,
    })).rejects.toThrow();
    expect(models.get('overflow')).toEqual(before);
  });

  it('reuses a prepared WASM workspace and disposes LRU evictions', async () => {
    const runtime = await loadRuntime({ simd: false });
    // Observe real SDK calls without replacing numerical execution.
    const models = controller(db(), { simd: false, cacheSize: 1 });
    await models.create('one', linear());
    await models.create('two', linear());
    await models.predict('one', new Float64Array([1, 2]), 2);
    const cache = (models as any).cache as Map<string, { prepared: any }>;
    const prepared = cache.get('one')!.prepared;
    await models.predict('one', new Float64Array([3, 4]), 2);
    expect(cache.get('one')!.prepared).toBe(prepared);
    await models.predict('two', new Float64Array([1]));
    expect(() => prepared.predict(new Float64Array([1]))).toThrow('disposed');
    expect(runtime.backend).toBe('wasm');
    const second = cache.get('two')!.prepared;
    models.dispose();
    expect(() => second.predict(new Float64Array([1]))).toThrow('disposed');
  });

  it('rejects stale concurrent training using persisted compare-and-swap', async () => {
    const connection = db();
    const first = controller(connection);
    const second = controller(connection);
    await first.create('race', linear());
    // Both calls capture revision 1 before awaiting the shared runtime.
    const results = await Promise.allSettled([first.train('race', data()), second.train('race', data())]);
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
    const failure = results.find(result => result.status === 'rejected') as PromiseRejectedResult;
    expect(failure.reason.message).toContain('changed during training');
    expect(first.get('race')!.revision).toBe(2);
  });

  it('does not mutate after disposal during lazy initialization and disposes idempotently', async () => {
    const models = controller();
    const creating = models.create('cancelled', linear());
    models.dispose();
    models.dispose();
    await expect(creating).rejects.toThrow('disposed');
    await expect(models.predict('missing', new Float64Array([1]))).rejects.toThrow('disposed');
    expect(() => models.get('missing')).toThrow('disposed');
    expect(() => models.list()).toThrow('disposed');
    expect(() => models.delete('missing')).toThrow('disposed');
  });

  it('bounds model dimensions, parameter storage, serialized size and identifiers', async () => {
    const models = controller(db(), { maxModelBytes: 8, maxSerializedBytes: 512 });
    await expect(models.create('large', linear())).rejects.toThrow('memory budget');
    await expect(models.create('x'.repeat(129), linear())).rejects.toThrow('id');
    await expect(models.create('large-json', ' '.repeat(513))).rejects.toThrow('JSON byte');
    await expect(models.create('dims', { ...linear(), layers: [{ ...linear().layers[0], inputs: 4097 }] } as any)).rejects.toThrow('layer inputs');
    expect(models.list()).toEqual([]);
  });

  it('finalizes statements on failed writes', async () => {
    const connection = db();
    const finalize = vi.fn();
    const original = connection.prepare.bind(connection);
    connection.prepare = ((sql: string) => {
      const stmt = original(sql);
      return { get: stmt.get.bind(stmt), all: stmt.all.bind(stmt), run: stmt.run.bind(stmt), finalize };
    }) as any;
    const models = controller(connection);
    await models.create('duplicate', linear());
    finalize.mockClear();
    await expect(models.create('duplicate', linear())).rejects.toThrow();
    expect(finalize).toHaveBeenCalledTimes(1);
  });

  it('persists through a real native SQLite file and predicts through a read-only connection', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'agentdb-fann-'));
    directories.push(directory);
    const filename = join(directory, 'models.db');
    const connection = new NativeDatabase(filename);
    const models = controller(connection);
    await models.create('file', linear());
    await models.train('file', data());
    models.dispose();
    connection.close();
    const reopened = new NativeDatabase(filename, { readonly: true });
    resources.push(reopened);
    const reader = controller(reopened, { readonly: true });
    expect((await reader.predict('file', new Float64Array([1]))).outputs[0]).toBeCloseTo(3, 1);
    expect(reopened.prepare('SELECT revision FROM fann_models').get()).toEqual({ revision: 2 });
  });
});
