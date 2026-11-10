/**
 * Real-WASM prepared workspace benchmark, with identical persisted checkpoints.
 * Run: npx tsx benchmarks/fann-models.mts
 * Optional: FANN_BENCH_SAMPLES=1000 FANN_BENCH_WARMUP=100
 * Prints environment and median timings. Ratios are observations, not CI gates.
 */
import assert from 'node:assert/strict';
import { cpus, platform, arch } from 'node:os';
import Database from 'better-sqlite3';
import { FannModels, type DenseModel } from '../src/controllers/FannModels.js';

function boundedCount(value: string | undefined, fallback: number): number {
  const number = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(number) || number < 1 || number > 10_000) throw new Error('Benchmark counts must be integers from 1 to 10000');
  return number;
}
const samples = boundedCount(process.env.FANN_BENCH_SAMPLES, 1_000);
const warmup = boundedCount(process.env.FANN_BENCH_WARMUP, 100);
const rows = 32;
const model: DenseModel = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 16, outputs: 16, weights: Array.from({ length: 256 }, (_, i) => Math.sin(i * 0.37) * 0.1), bias: Array.from({ length: 16 }, (_, i) => i * 0.001), activation: 'Tanh', steepness: 1 },
  { inputs: 16, outputs: 1, weights: Array.from({ length: 16 }, (_, i) => Math.cos(i * 0.71) * 0.2), bias: [0.01], activation: 'Linear', steepness: 1 },
] };
const inputs = Float64Array.from({ length: rows * 16 }, (_, i) => Math.sin(i * 0.11));
const database = new Database(':memory:');
const cached = new FannModels(database);
try {
  await cached.create('benchmark', model);
  const expected = await cached.predict('benchmark', inputs, rows);
  const freshPredict = async () => {
    // Both lanes use the same existing table and persisted checkpoint. The fresh
    // lane reloads and validates the checkpoint, prepares WASM, then disposes it.
    const fresh = new FannModels(database, { readonly: true });
    try { return await fresh.predict('benchmark', inputs, rows); }
    finally { fresh.dispose(); }
  };
  assert.deepEqual(await freshPredict(), expected);
  for (let i = 0; i < warmup; i++) {
    assert.deepEqual(await cached.predict('benchmark', inputs, rows), expected);
    assert.deepEqual(await freshPredict(), expected);
  }
  const cachedTimes: number[] = [];
  const freshTimes: number[] = [];
  async function measure(lane: 'cached' | 'fresh') {
    const start = performance.now();
    const output = lane === 'cached' ? await cached.predict('benchmark', inputs, rows) : await freshPredict();
    const elapsed = performance.now() - start;
    // Correctness checks stay outside the measured operation.
    assert.deepEqual(output, expected);
    (lane === 'cached' ? cachedTimes : freshTimes).push(elapsed);
  }
  for (let i = 0; i < samples; i++) {
    // Alternate order to reduce systematic warmup / scheduler bias.
    await measure(i % 2 ? 'fresh' : 'cached');
    await measure(i % 2 ? 'cached' : 'fresh');
  }
  function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  const cachedMedian = median(cachedTimes);
  const freshMedian = median(freshTimes);
  console.log(JSON.stringify({
    node: process.version, platform: platform(), arch: arch(), cpu: cpus()[0]?.model ?? 'unknown',
    backend: expected.backend, sqlite: 'better-sqlite3 :memory:', model: '16x16 Tanh / 16x1 Linear',
    parameterCount: cached.get('benchmark')!.parameterCount, rows, warmup, samplesPerLane: samples,
    checkpointJsonBytes: Buffer.byteLength(JSON.stringify(model)), identicalOutputs: true,
    cachedMedianMs: cachedMedian, freshControllerMedianMs: freshMedian,
    freshOverCachedRatio: freshMedian / cachedMedian,
  }, null, 2));
} finally {
  cached.dispose();
  database.close();
}
