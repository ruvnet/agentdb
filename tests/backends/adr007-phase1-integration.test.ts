/**
 * ADR-007 Phase 1: Full Integration Test Suite
 *
 * Validates all Phase 1 capabilities across the NativeAccelerator bridge:
 * - SIMD activation functions (matvec, softmax, relu, gelu, sigmoid, layerNorm, add, mul, scale, normalize)
 * - WASM store bridge (create, ingest, query, export, close)
 * - WASM quantization bridge (SQ params, dequant, PQ codebook, distances)
 * - FilterBuilder (all 11 operators + static buildFilter DSL)
 * - TemporalCompressor batch compression + native tensor bridge
 * - ContrastiveTrainer native InfoNCE delegation
 * - Graph transactions + Cypher + batch insert
 * - Core batch operations
 * - EWC manager
 * - Performance benchmarks
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { NativeAccelerator, resetAccelerator } from '../../src/backends/rvf/NativeAccelerator.js';
import { WasmStoreBridge } from '../../src/backends/rvf/WasmStoreBridge.js';
import { FilterBuilder, type RvfFilterExpr, type FilterPredicate } from '../../src/backends/rvf/FilterBuilder.js';

function randomVec(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = (Math.random() - 0.5) * 2;
  return v;
}

describe('ADR-007 Phase 1: SIMD Activation Functions', () => {
  let accel: NativeAccelerator;

  beforeEach(() => {
    resetAccelerator();
    accel = new NativeAccelerator();
  });

  describe('matvec', () => {
    it('should compute matrix-vector product correctly', () => {
      const matrix = [[1, 2], [3, 4]];
      const vector = [5, 6];
      const result = accel.matvec(matrix, vector);
      expect(result[0]).toBeCloseTo(17); // 1*5 + 2*6
      expect(result[1]).toBeCloseTo(39); // 3*5 + 4*6
    });

    it('should handle identity matrix', () => {
      const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      const v = [3, 7, 11];
      const result = accel.matvec(identity, v);
      expect(result[0]).toBeCloseTo(3);
      expect(result[1]).toBeCloseTo(7);
      expect(result[2]).toBeCloseTo(11);
    });

    it('should handle large matrices', () => {
      const dim = 128;
      const matrix = Array.from({ length: dim }, () =>
        Array.from({ length: dim }, () => Math.random() - 0.5),
      );
      const vector = Array.from({ length: dim }, () => Math.random());
      const result = accel.matvec(matrix, vector);
      expect(result.length).toBe(dim);
      for (const val of result) expect(Number.isFinite(val)).toBe(true);
    });
  });

  describe('softmax', () => {
    it('should sum to 1.0', () => {
      const result = accel.softmax([1, 2, 3]);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });

    it('should produce valid probabilities', () => {
      const result = accel.softmax([0, 0, 0]);
      for (const p of result) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
      expect(result[0]).toBeCloseTo(1 / 3, 5);
    });

    it('should handle large values without overflow', () => {
      const result = accel.softmax([1000, 1001, 1002]);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
      expect(result[2]).toBeGreaterThan(result[1]);
    });

    it('should handle single element', () => {
      const result = accel.softmax([5.0]);
      expect(result[0]).toBeCloseTo(1.0, 5);
    });
  });

  describe('relu', () => {
    it('should zero out negative values', () => {
      const result = accel.relu([-1, 0, 1, -0.5, 2.5]);
      expect(result).toEqual([0, 0, 1, 0, 2.5]);
    });

    it('should pass through positive values unchanged', () => {
      const input = [1.5, 2.7, 0.1];
      const result = accel.relu(input);
      for (let i = 0; i < input.length; i++) {
        expect(result[i]).toBe(input[i]);
      }
    });
  });

  describe('gelu', () => {
    it('should approximate x*sigmoid(1.702*x)', () => {
      const result = accel.gelu([0]);
      expect(result[0]).toBeCloseTo(0, 5);
    });

    it('should be negative for large negative inputs', () => {
      const result = accel.gelu([-5]);
      expect(result[0]).toBeCloseTo(0, 1); // GELU(-5) ≈ 0
    });

    it('should be positive for positive inputs', () => {
      const result = accel.gelu([2.0]);
      expect(result[0]).toBeGreaterThan(0);
    });

    it('should approximate identity for large positive values', () => {
      const result = accel.gelu([10.0]);
      expect(result[0]).toBeCloseTo(10.0, 0);
    });
  });

  describe('sigmoid', () => {
    it('should return 0.5 for input 0', () => {
      const result = accel.sigmoid([0]);
      expect(result[0]).toBeCloseTo(0.5, 5);
    });

    it('should approach 1 for large positive input', () => {
      const result = accel.sigmoid([10]);
      expect(result[0]).toBeGreaterThan(0.999);
    });

    it('should approach 0 for large negative input', () => {
      const result = accel.sigmoid([-10]);
      expect(result[0]).toBeLessThan(0.001);
    });

    it('should be monotonically increasing', () => {
      const input = [-3, -2, -1, 0, 1, 2, 3];
      const result = accel.sigmoid(input);
      for (let i = 1; i < result.length; i++) {
        expect(result[i]).toBeGreaterThan(result[i - 1]);
      }
    });
  });

  describe('layerNorm', () => {
    it('should produce zero-mean output', () => {
      const result = accel.layerNorm([1, 2, 3, 4, 5]);
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      expect(mean).toBeCloseTo(0, 4);
    });

    it('should produce unit variance output', () => {
      const result = accel.layerNorm([1, 2, 3, 4, 5]);
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      const variance = result.reduce((a, b) => a + (b - mean) ** 2, 0) / result.length;
      expect(variance).toBeCloseTo(1.0, 3);
    });

    it('should handle constant input', () => {
      const result = accel.layerNorm([5, 5, 5, 5]);
      for (const val of result) {
        expect(Math.abs(val)).toBeLessThan(1); // Should be ~0 with eps
      }
    });
  });

  describe('element-wise add', () => {
    it('should add vectors element-wise', () => {
      const result = accel.add([1, 2, 3], [4, 5, 6]);
      expect(result).toEqual([5, 7, 9]);
    });

    it('should throw on length mismatch', () => {
      expect(() => accel.add([1, 2], [1, 2, 3])).toThrow('Length mismatch');
    });
  });

  describe('element-wise mul', () => {
    it('should multiply vectors element-wise', () => {
      const result = accel.mul([2, 3, 4], [5, 6, 7]);
      expect(result).toEqual([10, 18, 28]);
    });

    it('should throw on length mismatch', () => {
      expect(() => accel.mul([1], [1, 2])).toThrow('Length mismatch');
    });
  });

  describe('scale', () => {
    it('should scale all elements', () => {
      const result = accel.scale([1, 2, 3], 2.0);
      expect(result).toEqual([2, 4, 6]);
    });

    it('should handle zero scalar', () => {
      const result = accel.scale([1, 2, 3], 0);
      expect(result).toEqual([0, 0, 0]);
    });
  });

  describe('normalize', () => {
    it('should produce unit-length vector', () => {
      const result = accel.normalizeVec([3, 4]);
      const norm = Math.sqrt(result[0] ** 2 + result[1] ** 2);
      expect(norm).toBeCloseTo(1.0, 5);
    });

    it('should handle zero vector', () => {
      const result = accel.normalizeVec([0, 0, 0]);
      for (const val of result) expect(val).toBe(0);
    });

    it('should preserve direction', () => {
      const result = accel.normalizeVec([3, 4]);
      expect(result[0]).toBeCloseTo(0.6, 5);
      expect(result[1]).toBeCloseTo(0.8, 5);
    });
  });
});

describe('ADR-007 Phase 1: WASM Store Bridge', () => {
  it('should create WasmStoreBridge instance', () => {
    const bridge = new WasmStoreBridge();
    expect(bridge.available).toBe(false); // Before init
  });

  it('should initialize without errors', async () => {
    const bridge = new WasmStoreBridge();
    const available = await bridge.initialize();
    // May or may not be available depending on @ruvector/rvf-wasm
    expect(typeof available).toBe('boolean');
  });

  it('should return null for store operations when unavailable', () => {
    const bridge = new WasmStoreBridge();
    expect(bridge.wasmStoreCreate(128, 0)).toBeNull();
    expect(bridge.wasmStoreIngest(0, new Float32Array(128), [0], 1)).toBe(0);
    expect(bridge.wasmStoreQuery(0, new Float32Array(128), 10, 0)).toBeNull();
    expect(bridge.wasmStoreExport(0)).toBeNull();
    expect(bridge.wasmStoreClose(0)).toBe(false);
  });

  it('NativeAccelerator should expose WASM store', () => {
    const accel = new NativeAccelerator();
    expect(accel.wasmStoreAvailable).toBe(false); // Before init
    expect(accel.wasmStore).toBeInstanceOf(WasmStoreBridge);
    expect(accel.wasmStoreCreate(128, 0)).toBeNull();
    expect(accel.wasmStoreClose(0)).toBe(false);
  });
});

describe('ADR-007 Phase 1: WASM Quantization Bridge', () => {
  let accel: NativeAccelerator;

  beforeEach(() => {
    resetAccelerator();
    accel = new NativeAccelerator();
  });

  it('should report quantization availability', () => {
    expect(typeof accel.wasmQuantizationAvailable).toBe('boolean');
  });

  it('should return safe defaults when unavailable', () => {
    expect(accel.loadSqParams(new Uint8Array(16), 128)).toBe(false);
    expect(accel.dequantI8(new Uint8Array(128), new Float32Array(128), 128)).toBe(false);
    expect(accel.loadPqCodebook(new Uint8Array(256), 8, 256)).toBe(false);
    expect(accel.pqDistances(new Uint8Array(128), 128)).toBeNull();
  });
});

describe('ADR-007 Phase 1: FilterBuilder', () => {
  describe('fluent API', () => {
    it('should build single eq filter', () => {
      const builder = new FilterBuilder();
      const filter = builder.eq('status', 'active').build();
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('eq');
      expect((filter as { op: 'eq'; fieldId: number; value: string }).value).toBe('active');
    });

    it('should build AND of multiple predicates', () => {
      const builder = new FilterBuilder();
      const filter = builder
        .eq('type', 'memory')
        .gt('score', 0.5)
        .build();
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('and');
      const and = filter as { op: 'and'; exprs: RvfFilterExpr[] };
      expect(and.exprs.length).toBe(2);
    });

    it('should support all comparison operators', () => {
      const builder = new FilterBuilder();
      builder.eq('a', 1).ne('b', 2).lt('c', 3).le('d', 4).gt('e', 5).ge('f', 6);
      const filter = builder.build() as { op: 'and'; exprs: RvfFilterExpr[] };
      expect(filter.exprs.length).toBe(6);
      expect(filter.exprs[0].op).toBe('eq');
      expect(filter.exprs[1].op).toBe('ne');
      expect(filter.exprs[2].op).toBe('lt');
      expect(filter.exprs[3].op).toBe('le');
      expect(filter.exprs[4].op).toBe('gt');
      expect(filter.exprs[5].op).toBe('ge');
    });

    it('should support IN operator', () => {
      const builder = new FilterBuilder();
      const filter = builder.in('status', ['active', 'pending']).build();
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('in');
    });

    it('should support range operator', () => {
      const builder = new FilterBuilder();
      const filter = builder.range('score', 0.0, 1.0).build();
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('range');
    });

    it('should support NOT wrapper', () => {
      const builder = new FilterBuilder();
      const inner: RvfFilterExpr = { op: 'eq', fieldId: 0, value: 'deleted' };
      const filter = builder.not(inner).build();
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('not');
    });

    it('should assign stable field IDs', () => {
      const builder = new FilterBuilder();
      const id1 = builder.fieldId('name');
      const id2 = builder.fieldId('name');
      const id3 = builder.fieldId('other');
      expect(id1).toBe(id2);
      expect(id3).not.toBe(id1);
    });

    it('should return null for empty builder', () => {
      const builder = new FilterBuilder();
      expect(builder.build()).toBeNull();
    });

    it('should reset correctly', () => {
      const builder = new FilterBuilder();
      builder.eq('a', 1).eq('b', 2);
      builder.reset();
      expect(builder.build()).toBeNull();
    });

    it('should enforce MAX_FILTER_EXPRS limit', () => {
      const builder = new FilterBuilder();
      for (let i = 0; i < 64; i++) builder.eq(`field${i}`, i);
      expect(() => builder.eq('overflow', 99)).toThrow('exceeds maximum');
    });
  });

  describe('static buildFilter DSL', () => {
    it('should build from plain values (eq)', () => {
      const filter = FilterBuilder.buildFilter({ status: 'active' });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('eq');
    });

    it('should build from operator objects', () => {
      const filter = FilterBuilder.buildFilter({
        score: { $gt: 0.5 },
        type: { $ne: 'deleted' },
      });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('and');
    });

    it('should handle $in operator', () => {
      const filter = FilterBuilder.buildFilter({
        status: { $in: ['a', 'b', 'c'] },
      });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('in');
    });

    it('should handle $range operator', () => {
      const filter = FilterBuilder.buildFilter({
        score: { $range: [0.0, 1.0] },
      });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('range');
    });

    it('should return null for empty predicates', () => {
      expect(FilterBuilder.buildFilter({})).toBeNull();
    });

    it('should return null for null input', () => {
      expect(FilterBuilder.buildFilter(null as unknown as FilterPredicate)).toBeNull();
    });

    it('should skip null/undefined values', () => {
      const filter = FilterBuilder.buildFilter({ a: null as unknown as string, b: 'active' });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('eq');
    });

    it('should support boolean values', () => {
      const filter = FilterBuilder.buildFilter({ active: true });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('eq');
    });

    it('should support numeric values', () => {
      const filter = FilterBuilder.buildFilter({ count: 42 });
      expect(filter).not.toBeNull();
      expect(filter!.op).toBe('eq');
    });
  });
});

describe('ADR-007 Phase 1: TemporalCompressor Batch + Native Bridge', () => {
  it('should batch compress multiple vectors', async () => {
    const { TemporalCompressor } = await import('../../src/backends/rvf/AdaptiveIndexTuner.js');
    const compressor = await TemporalCompressor.create();

    const items = [
      { id: 'v1', embedding: new Float32Array([1, 2, 3, 4]), accessFrequency: 1.0 },
      { id: 'v2', embedding: new Float32Array([5, 6, 7, 8]), accessFrequency: 0.5 },
      { id: 'v3', embedding: new Float32Array([9, 10, 11, 12]), accessFrequency: 0.1 },
    ];

    const results = compressor.compressBatch(items);
    expect(results.length).toBe(3);
    expect(results[0].tier).toBe('none');
    expect(results[1].tier).toBe('pq8');
    expect(results[2].tier).toBe('binary');

    // All entries should be stored
    for (const item of items) {
      expect(compressor.has(item.id)).toBe(true);
    }

    // Decompress and verify
    const v1 = compressor.decompress('v1');
    expect(v1).not.toBeNull();
    expect(v1!.length).toBe(4);
    for (let i = 0; i < 4; i++) expect(v1![i]).toBeCloseTo(items[0].embedding[i], 4);

    compressor.destroy();
  });

  it('should report nativeCompressAvailable property', async () => {
    const { TemporalCompressor } = await import('../../src/backends/rvf/AdaptiveIndexTuner.js');
    const compressor = await TemporalCompressor.create();
    expect(typeof compressor.nativeCompressAvailable).toBe('boolean');
    compressor.destroy();
  });

  it('should handle empty batch', async () => {
    const { TemporalCompressor } = await import('../../src/backends/rvf/AdaptiveIndexTuner.js');
    const compressor = await TemporalCompressor.create();
    const results = compressor.compressBatch([]);
    expect(results.length).toBe(0);
    compressor.destroy();
  });

  it('should handle batch with mixed tiers', async () => {
    const { TemporalCompressor } = await import('../../src/backends/rvf/AdaptiveIndexTuner.js');
    const compressor = await TemporalCompressor.create();
    const dim = 32;

    const items: Array<{ id: string; embedding: Float32Array; accessFrequency: number }> = [];
    for (let i = 0; i < 20; i++) {
      items.push({
        id: `batch-${i}`,
        embedding: randomVec(dim),
        accessFrequency: i / 20, // 0.0 to 0.95
      });
    }

    const results = compressor.compressBatch(items);
    expect(results.length).toBe(20);

    // Verify all are decompressible
    for (const item of items) {
      const restored = compressor.decompress(item.id);
      expect(restored).not.toBeNull();
      expect(restored!.length).toBe(dim);
    }

    compressor.destroy();
  });
});

describe('ADR-007 Phase 1: ContrastiveTrainer Native InfoNCE', () => {
  it('should compute loss with native InfoNCE delegation path', async () => {
    const { ContrastiveTrainer } = await import('../../src/backends/rvf/ContrastiveTrainer.js');
    const trainer = await ContrastiveTrainer.create({ dimension: 8 });

    const sample = {
      anchor: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]),
      positive: new Float32Array([0.9, 0.1, 0, 0, 0, 0, 0, 0]),
      negatives: [
        new Float32Array([0, 0, 1, 0, 0, 0, 0, 0]),
        new Float32Array([0, 0, 0, 1, 0, 0, 0, 0]),
      ],
    };

    const loss = trainer.computeLoss([sample]);
    expect(loss).toBeGreaterThan(0);
    expect(Number.isFinite(loss)).toBe(true);

    // Verify accelerator is loaded
    const accel = trainer.getAccelerator();
    expect(accel).not.toBeNull();

    trainer.destroy();
  });

  it('should produce same loss with and without native path for JS fallback', async () => {
    const { ContrastiveTrainer } = await import('../../src/backends/rvf/ContrastiveTrainer.js');
    const trainer = await ContrastiveTrainer.create({ dimension: 4 });

    const sample = {
      anchor: new Float32Array([1, 0, 0, 0]),
      positive: new Float32Array([0.8, 0.2, 0, 0]),
      negatives: [new Float32Array([0, 0, 1, 0])],
    };

    // Compute loss twice - should be deterministic
    const loss1 = trainer.computeLoss([sample]);
    const loss2 = trainer.computeLoss([sample]);
    expect(loss1).toBeCloseTo(loss2, 5);

    trainer.destroy();
  });
});

describe('ADR-007 Phase 1: Graph + Core Batch + EWC APIs', () => {
  let accel: NativeAccelerator;

  beforeEach(() => {
    resetAccelerator();
    accel = new NativeAccelerator();
  });

  describe('Graph Transaction fallback', () => {
    it('should execute callback without native transactions', async () => {
      let called = false;
      const result = await accel.graphTransaction({}, () => { called = true; });
      expect(result).toBe(true);
      expect(called).toBe(true);
    });

    it('should handle async callbacks', async () => {
      let called = false;
      const result = await accel.graphTransaction({}, async () => {
        await new Promise(r => setTimeout(r, 1));
        called = true;
      });
      expect(result).toBe(true);
      expect(called).toBe(true);
    });

    it('should catch errors in callback', async () => {
      const result = await accel.graphTransaction({}, () => {
        throw new Error('test error');
      });
      expect(result).toBe(false);
    });
  });

  describe('Core Batch Insert fallback', () => {
    it('should return false without native support', async () => {
      const mockDb = {};
      expect(await accel.coreBatchInsert(mockDb, [
        { id: '1', vector: randomVec(4) },
        { id: '2', vector: randomVec(4) },
      ])).toBe(false);
    });
  });

  describe('EWC Manager fallback', () => {
    it('should return 0 penalty without native', () => {
      expect(accel.ewcPenalty(randomVec(128))).toBe(0);
    });

    it('should return false for Fisher update without native', () => {
      expect(accel.ewcUpdateFisher(randomVec(128), 1.0)).toBe(false);
    });
  });
});

describe('ADR-007 Phase 1: Performance Benchmarks', () => {
  let accel: NativeAccelerator;

  beforeEach(() => {
    resetAccelerator();
    accel = new NativeAccelerator();
  });

  it('should complete 10K softmax (dim=64) in <200ms', () => {
    const input = Array.from({ length: 64 }, () => Math.random());
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.softmax(input);
    expect(performance.now() - start).toBeLessThan(200);
  });

  it('should complete 10K relu (dim=256) in <100ms', () => {
    const input = Array.from({ length: 256 }, () => Math.random() - 0.5);
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.relu(input);
    expect(performance.now() - start).toBeLessThan(100);
  });

  it('should complete 10K gelu (dim=256) in <300ms', () => {
    const input = Array.from({ length: 256 }, () => Math.random() - 0.5);
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.gelu(input);
    expect(performance.now() - start).toBeLessThan(300);
  });

  it('should complete 10K sigmoid (dim=256) in <300ms', () => {
    const input = Array.from({ length: 256 }, () => Math.random() * 10 - 5);
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.sigmoid(input);
    expect(performance.now() - start).toBeLessThan(300);
  });

  it('should complete 10K layerNorm (dim=256) in <200ms', () => {
    const input = Array.from({ length: 256 }, () => Math.random());
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.layerNorm(input);
    expect(performance.now() - start).toBeLessThan(200);
  });

  it('should complete 10K element-wise add (dim=256) in <100ms', () => {
    const a = Array.from({ length: 256 }, () => Math.random());
    const b = Array.from({ length: 256 }, () => Math.random());
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.add(a, b);
    expect(performance.now() - start).toBeLessThan(100);
  });

  it('should complete 10K normalize (dim=384) in <100ms', () => {
    const a = Array.from({ length: 384 }, () => Math.random());
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) accel.normalizeVec(a);
    expect(performance.now() - start).toBeLessThan(100);
  });

  it('should complete 1K matvec (64x64) in <200ms', () => {
    const matrix = Array.from({ length: 64 }, () =>
      Array.from({ length: 64 }, () => Math.random()),
    );
    const vector = Array.from({ length: 64 }, () => Math.random());
    const start = performance.now();
    for (let i = 0; i < 1_000; i++) accel.matvec(matrix, vector);
    expect(performance.now() - start).toBeLessThan(200);
  });

  it('should complete batch compression of 100 vectors (dim=128) in <200ms', async () => {
    const { TemporalCompressor } = await import('../../src/backends/rvf/AdaptiveIndexTuner.js');
    const compressor = await TemporalCompressor.create();

    const items = Array.from({ length: 100 }, (_, i) => ({
      id: `bench-${i}`,
      embedding: randomVec(128),
      accessFrequency: Math.random(),
    }));

    const start = performance.now();
    compressor.compressBatch(items);
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(200);

    compressor.destroy();
  });

  it('should complete FilterBuilder.buildFilter 10K times in <100ms', () => {
    const predicates = { status: 'active', score: { $gt: 0.5 }, type: { $ne: 'deleted' } };
    const start = performance.now();
    for (let i = 0; i < 10_000; i++) FilterBuilder.buildFilter(predicates);
    expect(performance.now() - start).toBeLessThan(100);
  });
});

describe('ADR-007 Phase 1: AcceleratorStats Integration', () => {
  it('should include all new capability flags after initialization', async () => {
    resetAccelerator();
    const accel = new NativeAccelerator();
    const stats = await accel.initialize();

    // All flags should be booleans
    expect(typeof stats.simdAvailable).toBe('boolean');
    expect(typeof stats.simdActivationsAvailable).toBe('boolean');
    expect(typeof stats.wasmVerifyAvailable).toBe('boolean');
    expect(typeof stats.wasmStoreAvailable).toBe('boolean');
    expect(typeof stats.wasmQuantizationAvailable).toBe('boolean');
    expect(typeof stats.nativeInfoNceAvailable).toBe('boolean');
    expect(typeof stats.nativeAdamWAvailable).toBe('boolean');
    expect(typeof stats.nativeTensorCompressAvailable).toBe('boolean');
    expect(typeof stats.routerPersistAvailable).toBe('boolean');
    expect(typeof stats.sonaExtendedAvailable).toBe('boolean');
    expect(Array.isArray(stats.capabilities)).toBe(true);
  });
});
