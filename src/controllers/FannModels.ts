/** Persisted neural models backed exclusively by the real ruv-fann WASM runtime. */
import { randomUUID } from 'node:crypto';
import { loadRuntime, type DenseModel, type FannRuntime, type PreparedModel } from 'ruv-fann';
import type { IDatabaseConnection, IPreparedStatement } from '../types/database.types.js';
import { inTransaction } from './crud-helpers.js';

export type { DenseModel } from 'ruv-fann';

export interface ModelMetadata {
  id: string;
  revision: number;
  inputSize: number;
  outputSize: number;
  parameterCount: number;
  createdAt: string;
  updatedAt: string;
}
export interface ModelRecord extends ModelMetadata { model: DenseModel }
export interface FannTrainOptions {
  inputs: Float64Array;
  targets: Float64Array;
  rows: number;
  epochs: number;
  learningRate: number;
  optimizer?: 'sgd' | 'adamw';
}
export interface FannModelsOptions {
  /** Existing databases can be opened without schema writes. Mutations are rejected. */
  readonly?: boolean;
  simd?: boolean | 'auto';
  /** Limits can be lowered, never raised above these defaults. */
  maxRows?: number;
  maxEpochs?: number;
  maxTrainingCompute?: number;
  maxModelBytes?: number;
  maxSerializedBytes?: number;
  maxParameters?: number;
  cacheSize?: number;
  cacheBytes?: number;
}

const DEFAULTS = Object.freeze({
  maxRows: 128,
  maxEpochs: 1_000,
  maxTrainingCompute: 50_000_000,
  maxModelBytes: 8 * 1024 * 1024,
  maxSerializedBytes: 8 * 1024 * 1024,
  maxParameters: 262_144,
  cacheSize: 16,
  cacheBytes: 32 * 1024 * 1024,
});
type Limits = { -readonly [Key in keyof typeof DEFAULTS]: number };
type Versioned = ModelMetadata & { version_token: string };
type Stored = Versioned & { model_json: string };
type CacheEntry = { token: string; prepared: PreparedModel };
const COLUMNS = `id, revision, input_size AS inputSize, output_size AS outputSize,
  parameter_count AS parameterCount, created_at AS createdAt, updated_at AS updatedAt`;
const runtimes = new Map<boolean | 'auto', Promise<FannRuntime>>();

function sharedRuntime(simd: boolean | 'auto'): Promise<FannRuntime> {
  let pending = runtimes.get(simd);
  if (!pending) {
    pending = loadRuntime({ simd }).catch(error => {
      runtimes.delete(simd);
      throw error;
    });
    runtimes.set(simd, pending);
  }
  return pending;
}

function integer(value: unknown, name: string, max: number): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > max) {
    throw new Error(`${name} must be an integer between 1 and ${max}`);
  }
}
function idCheck(id: string): void {
  if (typeof id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(id)) {
    throw new Error('id must contain 1–128 safe ASCII letters, numbers, dots, colons, underscores or hyphens');
  }
}
function finiteArray(values: unknown, length: number, name: string): asserts values is Float64Array {
  if (!(values instanceof Float64Array) || values.buffer instanceof SharedArrayBuffer || values.length !== length) {
    throw new Error(`${name} must be a Float64Array of length ${length}`);
  }
  for (const value of values) if (!Number.isFinite(value)) throw new Error(`${name} must contain finite numbers`);
}

/**
 * Checkpoint persistence includes weights and biases, not optimizer moments. Each
 * train call starts a fresh optimizer. Training work is capped at 50 million
 * parameters × rows × epochs, with at most 128 rows, 1,000 epochs, 262,144
 * parameters, 64 layers, 4,096 neurons per layer, and 8 MiB each for serialized
 * JSON and parameter storage. Prepared inference storage has a 16 MiB per-model
 * cap and a 16-entry / 32 MiB LRU; repeated batches reuse their WASM workspace.
 */
export class FannModels {
  private readonly limits: Limits;
  private readonly readonly: boolean;
  private readonly simd: boolean | 'auto';
  private readonly cache = new Map<string, CacheEntry>();
  private cacheBytes = 0;
  private disposed = false;

  constructor(private readonly db: IDatabaseConnection, options: FannModelsOptions = {}) {
    const limits: Limits = { ...DEFAULTS };
    for (const key of Object.keys(DEFAULTS) as (keyof Limits)[]) {
      const value = options[key];
      if (value !== undefined) {
        integer(value, key, DEFAULTS[key]);
        limits[key] = value;
      }
    }
    this.limits = limits;
    this.readonly = options.readonly ?? false;
    this.simd = options.simd ?? 'auto';
    if (this.simd !== true && this.simd !== false && this.simd !== 'auto') throw new Error('Invalid simd option');
    if (!this.readonly) db.exec(`CREATE TABLE IF NOT EXISTS fann_models (
      id TEXT PRIMARY KEY, model_json TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision >= 1),
      version_token TEXT NOT NULL, input_size INTEGER NOT NULL, output_size INTEGER NOT NULL,
      parameter_count INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )`);
  }

  async create(id: string, model: DenseModel | string): Promise<ModelRecord> {
    this.assertWritable();
    idCheck(id);
    const checked = this.checkModel(model);
    const runtime = await this.runtime();
    // Let the runtime validate activation names and all numerical semantics.
    const prepared = runtime.prepare(checked.model, { maxRows: 1 });
    try { this.checkPrepared(prepared); } finally { prepared.dispose(); }
    this.assertWritable();
    const now = new Date().toISOString();
    this.statement('INSERT INTO fann_models (id, model_json, revision, version_token, input_size, output_size, parameter_count, created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)',
      stmt => stmt.run(id, checked.json, randomUUID(), checked.inputSize, checked.outputSize, checked.parameterCount, now, now));
    this.invalidate(id);
    return { id, revision: 1, inputSize: checked.inputSize, outputSize: checked.outputSize,
      parameterCount: checked.parameterCount, createdAt: now, updatedAt: now, model: checked.model };
  }

  get(id: string): ModelRecord | null {
    const row = this.read(id);
    if (!row) return null;
    const { model_json, version_token, ...metadata } = row;
    return { ...metadata, model: this.checkModel(model_json).model };
  }

  list(limit = 50): ModelMetadata[] {
    this.assertAlive();
    integer(limit, 'limit', 500);
    return this.statement<ModelMetadata, ModelMetadata[]>(`SELECT ${COLUMNS} FROM fann_models ORDER BY id LIMIT ?`, stmt => stmt.all(limit));
  }

  async train(id: string, options: FannTrainOptions) {
    this.assertWritable();
    const row = this.required(id);
    const checked = this.checkModel(row.model_json);
    integer(options.rows, 'rows', this.limits.maxRows);
    integer(options.epochs, 'epochs', this.limits.maxEpochs);
    const { rows, epochs, learningRate } = options;
    const optimizer = options.optimizer ?? 'adamw';
    if (typeof learningRate !== 'number' || !Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      throw new Error('learningRate must be finite and greater than 0 and at most 1');
    }
    if (optimizer !== 'sgd' && optimizer !== 'adamw') throw new Error('optimizer must be sgd or adamw');
    if (checked.parameterCount * rows * epochs > this.limits.maxTrainingCompute) throw new Error('Training compute budget exceeded');
    finiteArray(options.inputs, rows * checked.inputSize, 'inputs');
    finiteArray(options.targets, rows * checked.outputSize, 'targets');
    // Own buffers and checkpoint across the async initialization boundary.
    const inputs = options.inputs.slice();
    const targets = options.targets.slice();
    const runtime = await this.runtime();
    const result = runtime.train(checked.model, { inputs, targets, rows, epochs, learningRate, optimizer });
    const trained = this.checkModel(result.model);
    if (!Number.isFinite(result.initialLoss) || !Number.isFinite(result.finalLoss) || result.initialLoss < 0 || result.finalLoss < 0) {
      throw new Error('Training returned non-finite or negative loss');
    }
    if (trained.inputSize !== checked.inputSize || trained.outputSize !== checked.outputSize || trained.parameterCount !== checked.parameterCount) {
      throw new Error('Training changed model dimensions');
    }
    this.assertWritable();
    const now = new Date().toISOString();
    const revision = row.revision + 1;
    if (!Number.isSafeInteger(revision)) throw new Error('Model revision exhausted');
    inTransaction(this.db, () => {
      const changes = this.statement('UPDATE fann_models SET model_json = ?, revision = ?, version_token = ?, updated_at = ? WHERE id = ? AND revision = ? AND version_token = ?',
        stmt => stmt.run(trained.json, revision, randomUUID(), now, id, row.revision, row.version_token).changes);
      if (changes !== 1) throw new Error('Model changed during training; retry with the latest revision');
    });
    this.invalidate(id);
    return { id, revision, initialLoss: result.initialLoss, finalLoss: result.finalLoss,
      epochs, parameterCount: trained.parameterCount, backend: runtime.backend };
  }

  async predict(id: string, inputs: Float64Array, rows = 1) {
    this.assertAlive();
    idCheck(id);
    integer(rows, 'rows', this.limits.maxRows);
    if (!(inputs instanceof Float64Array) || inputs.length > this.limits.maxRows * 4096) throw new Error('Invalid inputs length');
    finiteArray(inputs, inputs.length, 'inputs');
    // Own the input across initialization; shared memory is rejected above.
    const ownedInputs = inputs.slice();
    // Reading after initialization ensures another controller's latest checkpoint is used.
    const runtime = await this.runtime();
    let row = this.requiredMetadata(id);
    finiteArray(ownedInputs, rows * row.inputSize, 'inputs');
    let entry = this.cache.get(id);
    if (entry && (entry.token !== row.version_token || entry.prepared.maxRows < rows)) {
      this.invalidate(id);
      entry = undefined;
    }
    if (!entry) {
      // Fetch weights only on cache misses, not on repeated batch inference.
      const checkpoint = this.required(id);
      row = checkpoint;
      finiteArray(ownedInputs, rows * row.inputSize, 'inputs');
      const checked = this.checkModel(checkpoint.model_json);
      const capacity = Math.min(this.limits.maxRows, 2 ** Math.ceil(Math.log2(rows)));
      const prepared = runtime.prepare(checked.model, { maxRows: capacity });
      try { this.checkPrepared(prepared); } catch (error) { prepared.dispose(); throw error; }
      entry = { token: row.version_token, prepared };
      if (prepared.estimatedBytes <= this.limits.cacheBytes) {
        while (this.cache.size >= this.limits.cacheSize || this.cacheBytes + prepared.estimatedBytes > this.limits.cacheBytes) {
          this.invalidate(this.cache.keys().next().value!);
        }
        this.cache.set(id, entry);
        this.cacheBytes += prepared.estimatedBytes;
      }
    } else {
      this.cache.delete(id);
      this.cache.set(id, entry);
    }
    try {
      const outputs = entry.prepared.predict(ownedInputs, rows);
      finiteArray(outputs, rows * row.outputSize, 'outputs');
      return { id, revision: row.revision, rows, outputs: Array.from(outputs), backend: runtime.backend };
    } finally {
      if (this.cache.get(id) !== entry) entry.prepared.dispose();
    }
  }

  delete(id: string): boolean {
    this.assertWritable();
    idCheck(id);
    const changes = this.statement('DELETE FROM fann_models WHERE id = ?', stmt => stmt.run(id).changes);
    this.invalidate(id);
    return changes > 0;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const id of this.cache.keys()) this.invalidate(id);
  }

  private async runtime(): Promise<FannRuntime> {
    this.assertAlive();
    const runtime = await sharedRuntime(this.simd);
    this.assertAlive();
    return runtime;
  }
  private assertAlive(): void { if (this.disposed) throw new Error('FannModels is disposed'); }
  private assertWritable(): void {
    this.assertAlive();
    if (this.readonly) throw new Error('FannModels is read-only');
  }
  private statement<T = any, R = any>(sql: string, fn: (stmt: IPreparedStatement<T>) => R): R {
    const stmt = this.db.prepare<T>(sql);
    try { return fn(stmt); } finally { stmt.finalize?.(); }
  }
  private read(id: string): Stored | undefined {
    this.assertAlive();
    idCheck(id);
    return this.statement<Stored, Stored | undefined>(`SELECT ${COLUMNS}, model_json, version_token FROM fann_models WHERE id = ?`, stmt => stmt.get(id));
  }
  private requiredMetadata(id: string): Versioned {
    const row = this.statement<Versioned, Versioned | undefined>(`SELECT ${COLUMNS}, version_token FROM fann_models WHERE id = ?`, stmt => stmt.get(id));
    if (!row) { this.invalidate(id); throw new Error(`Model ${id} not found`); }
    return row;
  }
  private required(id: string): Stored {
    const row = this.read(id);
    if (!row) { this.invalidate(id); throw new Error(`Model ${id} not found`); }
    return row;
  }
  private invalidate(id: string): void {
    const entry = this.cache.get(id);
    if (!entry) return;
    this.cache.delete(id);
    this.cacheBytes -= entry.prepared.estimatedBytes;
    entry.prepared.dispose();
  }
  private checkPrepared(prepared: PreparedModel): void {
    if (!Number.isSafeInteger(prepared.estimatedBytes) || prepared.estimatedBytes < 0 || prepared.estimatedBytes > 16 * 1024 * 1024) {
      throw new Error('Prepared model memory budget exceeded');
    }
  }
  private checkModel(input: DenseModel | string) {
    if (typeof input === 'string' && Buffer.byteLength(input) > this.limits.maxSerializedBytes) throw new Error('Model JSON byte budget exceeded');
    const value = typeof input === 'string' ? JSON.parse(input) : input;
    if (!value || value.schema !== 'ruv-fann.dense-layers/1' || !Array.isArray(value.layers)) throw new Error('Invalid dense model schema');
    integer(value.layers.length, 'layers', 64);
    let parameterCount = 0;
    let previous = 0;
    const layers = value.layers.map((layer: DenseModel['layers'][number], index: number) => {
      if (!layer || typeof layer !== 'object') throw new Error('Invalid model layer');
      integer(layer.inputs, 'layer inputs', 4096);
      integer(layer.outputs, 'layer outputs', 4096);
      if (index && layer.inputs !== previous) throw new Error('Model layers have incompatible dimensions');
      previous = layer.outputs;
      const count = layer.inputs * layer.outputs;
      parameterCount += count + (layer.bias == null ? 0 : layer.outputs);
      if (parameterCount > this.limits.maxParameters || parameterCount * 8 > this.limits.maxModelBytes) throw new Error('Model parameter or memory budget exceeded');
      if (!Array.isArray(layer.weights) || layer.weights.length !== count || !layer.weights.every(Number.isFinite)) throw new Error('Invalid finite model weights');
      if (layer.bias != null && (!Array.isArray(layer.bias) || layer.bias.length !== layer.outputs || !layer.bias.every(Number.isFinite))) throw new Error('Invalid finite model bias');
      if (typeof layer.activation !== 'string' || layer.activation.length > 64 || typeof layer.steepness !== 'number' || !Number.isFinite(layer.steepness) || layer.steepness <= 0) throw new Error('Invalid model activation or steepness');
      return { inputs: layer.inputs, outputs: layer.outputs, weights: [...layer.weights], bias: layer.bias == null ? null : [...layer.bias], activation: layer.activation, steepness: layer.steepness };
    });
    const model: DenseModel = { schema: 'ruv-fann.dense-layers/1', layers };
    const json = JSON.stringify(model);
    if (Buffer.byteLength(json) > this.limits.maxSerializedBytes) throw new Error('Model JSON byte budget exceeded');
    return { model, json, parameterCount, inputSize: layers[0].inputs, outputSize: layers[layers.length - 1].outputs };
  }
}
