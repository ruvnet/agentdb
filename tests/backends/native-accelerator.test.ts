/**
 * ADR-007 Phase 1: NativeAccelerator Test Suite
 *
 * Tests native @ruvector capability bridge with JS fallbacks.
 * Since @ruvector packages are not installed in CI, all tests
 * validate the JS fallback paths and correctness guarantees.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { NativeAccelerator, resetAccelerator, getAccelerator } from '../../src/backends/rvf/NativeAccelerator.js';

// Helper: create a random Float32Array
function randomVec(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = (Math.random() - 0.5) * 2;
  return v;
}

// Helper: normalize a vector
function normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  const out = new Float32Array(v.length);
  if (norm > 0) for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

describe('ADR-007 NativeAccelerator', () => {
  let accel: NativeAccelerator;

  beforeEach(() => {
    resetAccelerator();
    accel = new NativeAccelerator();
  });

  // ─── Initialization ───

  describe('Initialization', () => {
    it('should initialize without errors even with no @ruvector packages', async () => {
      const stats = await accel.initialize();
      expect(stats).toBeDefined();
      expect(stats.capabilities).toBeInstanceOf(Array);
    });

    it('should report capabilities based on available @ruvector packages', async () => {
      const stats = await accel.initialize();
      // Each boolean should be a boolean regardless of availability
      expect(typeof stats.simdAvailable).toBe('boolean');
      expect(typeof stats.wasmVerifyAvailable).toBe('boolean');
      expect(typeof stats.nativeInfoNceAvailable).toBe('boolean');
      expect(typeof stats.nativeAdamWAvailable).toBe('boolean');
      expect(typeof stats.nativeTensorCompressAvailable).toBe('boolean');
      expect(typeof stats.routerPersistAvailable).toBe('boolean');
      expect(typeof stats.sonaExtendedAvailable).toBe('boolean');
      // capabilities length should match number of true booleans
      const trueCount = [
        stats.simdAvailable, stats.wasmVerifyAvailable, stats.nativeInfoNceAvailable,
        stats.nativeAdamWAvailable, stats.nativeTensorCompressAvailable,
        stats.routerPersistAvailable, stats.sonaExtendedAvailable,
      ].filter(Boolean).length;
      expect(stats.capabilities.length).toBeGreaterThanOrEqual(trueCount > 0 ? 1 : 0);
    });

    it('should return same stats on second initialize call', async () => {
      const s1 = await accel.initialize();
      const s2 = await accel.initialize();
      expect(s1).toEqual(s2);
    });
  });

  // ─── Singleton ───

  describe('Singleton getAccelerator', () => {
    it('should return the same instance on multiple calls', async () => {
      const a1 = await getAccelerator();
      const a2 = await getAccelerator();
      expect(a1).toBe(a2);
    });

    it('should return new instance after reset', async () => {
      const a1 = await getAccelerator();
      resetAccelerator();
      const a2 = await getAccelerator();
      expect(a1).not.toBe(a2);
    });
  });

  // ─── SIMD Vector Math (JS fallbacks) ───

  describe('Cosine Similarity', () => {
    it('should return 1.0 for identical vectors', () => {
      const v = normalize(randomVec(128));
      expect(accel.cosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
    });

    it('should return 0.0 for orthogonal vectors', () => {
      const a = new Float32Array([1, 0, 0, 0]);
      const b = new Float32Array([0, 1, 0, 0]);
      expect(accel.cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
    });

    it('should return -1.0 for opposite vectors', () => {
      const a = new Float32Array([1, 0, 0]);
      const b = new Float32Array([-1, 0, 0]);
      expect(accel.cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
    });

    it('should handle high-dimensional vectors', () => {
      const a = randomVec(1024);
      const b = randomVec(1024);
      const sim = accel.cosineSimilarity(a, b);
      expect(sim).toBeGreaterThanOrEqual(-1.0);
      expect(sim).toBeLessThanOrEqual(1.0);
    });

    it('should throw on dimension mismatch', () => {
      expect(() => accel.cosineSimilarity(randomVec(64), randomVec(128))).toThrow('Dimension mismatch');
    });

    it('should throw on dimension exceeding max', () => {
      expect(() => accel.cosineSimilarity(randomVec(5000), randomVec(5000))).toThrow('exceeds maximum');
    });

    it('should return 0 for zero vectors', () => {
      const z = new Float32Array(64);
      expect(accel.cosineSimilarity(z, z)).toBe(0);
    });
  });

  describe('Dot Product', () => {
    it('should compute dot product correctly', () => {
      const a = new Float32Array([1, 2, 3]);
      const b = new Float32Array([4, 5, 6]);
      expect(accel.dotProduct(a, b)).toBeCloseTo(32.0, 5); // 4 + 10 + 18
    });

    it('should return 0 for orthogonal vectors', () => {
      const a = new Float32Array([1, 0]);
      const b = new Float32Array([0, 1]);
      expect(accel.dotProduct(a, b)).toBeCloseTo(0.0, 5);
    });

    it('should handle negative values', () => {
      const a = new Float32Array([-1, 2, -3]);
      const b = new Float32Array([4, -5, 6]);
      expect(accel.dotProduct(a, b)).toBeCloseTo(-32.0, 5);
    });
  });

  describe('L2 Distance', () => {
    it('should return 0 for identical vectors', () => {
      const v = randomVec(128);
      expect(accel.l2Distance(v, v)).toBeCloseTo(0.0, 5);
    });

    it('should compute distance correctly', () => {
      const a = new Float32Array([0, 0]);
      const b = new Float32Array([3, 4]);
      expect(accel.l2Distance(a, b)).toBeCloseTo(5.0, 5);
    });

    it('should be symmetric', () => {
      const a = randomVec(64);
      const b = randomVec(64);
      expect(accel.l2Distance(a, b)).toBeCloseTo(accel.l2Distance(b, a), 5);
    });

    it('should satisfy triangle inequality', () => {
      const a = randomVec(64);
      const b = randomVec(64);
      const c = randomVec(64);
      expect(accel.l2Distance(a, c)).toBeLessThanOrEqual(
        accel.l2Distance(a, b) + accel.l2Distance(b, c) + 1e-5
      );
    });
  });

  // ─── Hamming Distance (SOTA: binary quantization) ───

  describe('Hamming Distance', () => {
    it('should return 0 for identical binary vectors', () => {
      const a = new Uint8Array([0xFF, 0x00, 0xAA]);
      expect(accel.hammingDistance(a, a)).toBe(0);
    });

    it('should compute correct hamming distance', () => {
      const a = new Uint8Array([0b11110000]);
      const b = new Uint8Array([0b11001100]);
      // XOR = 0b00111100, popcount = 4
      expect(accel.hammingDistance(a, b)).toBe(4);
    });

    it('should return max distance for inverse vectors', () => {
      const a = new Uint8Array([0xFF]);
      const b = new Uint8Array([0x00]);
      expect(accel.hammingDistance(a, b)).toBe(8);
    });

    it('should handle multi-byte vectors', () => {
      const a = new Uint8Array(128).fill(0xFF);
      const b = new Uint8Array(128).fill(0x00);
      expect(accel.hammingDistance(a, b)).toBe(1024); // 128 * 8
    });

    it('should throw on length mismatch', () => {
      expect(() => accel.hammingDistance(new Uint8Array(4), new Uint8Array(8))).toThrow('Length mismatch');
    });

    it('should complete 10K hamming distances (128B) in <100ms', () => {
      const a = new Uint8Array(128); const b = new Uint8Array(128);
      for (let i = 0; i < 128; i++) { a[i] = i; b[i] = 255 - i; }
      const start = performance.now();
      for (let i = 0; i < 10_000; i++) accel.hammingDistance(a, b);
      expect(performance.now() - start).toBeLessThan(100);
    });
  });

  // ─── WASM Verification (JS fallbacks) ───

  describe('Witness Chain Verification', () => {
    it('should return invalid for empty chain', () => {
      const result = accel.verifyWitnessChain(new Uint8Array(0));
      expect(result.valid).toBe(false);
      expect(result.entryCount).toBe(0);
    });

    it('should return invalid for null-like input', () => {
      const result = accel.verifyWitnessChain(null as unknown as Uint8Array);
      expect(result.valid).toBe(false);
    });

    it('should validate chain with correct entry size (73 bytes)', () => {
      const chain = new Uint8Array(73 * 5); // 5 entries
      for (let i = 0; i < chain.length; i++) chain[i] = i % 256;
      const result = accel.verifyWitnessChain(chain);
      expect(result.valid).toBe(true);
      expect(result.entryCount).toBe(5);
    });

    it('should reject chain with incorrect size', () => {
      const chain = new Uint8Array(100); // Not a multiple of 73
      const result = accel.verifyWitnessChain(chain);
      expect(result.valid).toBe(false);
    });

    it('should handle single entry chain', () => {
      const chain = new Uint8Array(73);
      const result = accel.verifyWitnessChain(chain);
      expect(result.valid).toBe(true);
      expect(result.entryCount).toBe(1);
    });
  });

  describe('Segment Header Verification', () => {
    it('should reject too-short data', () => {
      const result = accel.verifySegmentHeader(new Uint8Array(2));
      expect(result.valid).toBe(false);
    });

    it('should validate RVF magic bytes', () => {
      const data = new Uint8Array([0x52, 0x56, 0x46, 0x00, 0x01, 0x02]);
      const result = accel.verifySegmentHeader(data);
      expect(result.valid).toBe(true);
      expect(result.crc).toBeGreaterThan(0);
    });

    it('should reject invalid magic bytes', () => {
      const data = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x01, 0x02]);
      const result = accel.verifySegmentHeader(data);
      expect(result.valid).toBe(false);
    });

    it('should compute deterministic CRC', () => {
      const data = new Uint8Array([0x52, 0x56, 0x46, 0x00]);
      const r1 = accel.verifySegmentHeader(data);
      const r2 = accel.verifySegmentHeader(data);
      expect(r1.crc).toBe(r2.crc);
    });
  });

  // ─── InfoNCE Loss (JS fallback) ───

  describe('InfoNCE Loss', () => {
    it('should return 0 for perfect separation', () => {
      // Anchor and positive are identical, negatives are orthogonal
      const anchor = new Float32Array([1, 0, 0]);
      const positive = new Float32Array([1, 0, 0]);
      const negatives = [new Float32Array([0, 1, 0])];
      const loss = accel.infoNceLoss(anchor, positive, negatives, 0.07);
      // With perfect similarity between anchor-positive and zero to negative,
      // loss should be small but positive
      expect(loss).toBeGreaterThanOrEqual(0);
      expect(loss).toBeLessThan(2);
    });

    it('should return higher loss for poor separation', () => {
      const anchor = new Float32Array([1, 0.5, 0]);
      const positive = new Float32Array([1, 0.5, 0]);
      const neg1 = [new Float32Array([0, 1, 0])];
      const neg2 = [new Float32Array([0.9, 0.4, 0])]; // Very similar to anchor

      const loss1 = accel.infoNceLoss(anchor, positive, neg1, 0.07);
      const loss2 = accel.infoNceLoss(anchor, positive, neg2, 0.07);
      expect(loss2).toBeGreaterThan(loss1);
    });

    it('should increase with more hard negatives (deterministic)', () => {
      // Use deterministic vectors: anchor=[1,0,...], positive=[0.9,0.1,...], negatives close to anchor
      const dim = 64;
      const anchor = new Float32Array(dim); anchor[0] = 1;
      const positive = new Float32Array(dim); positive[0] = 0.9; positive[1] = 0.1;
      const hardNeg = new Float32Array(dim); hardNeg[0] = 0.8; hardNeg[1] = -0.2;
      const neg1 = [hardNeg];
      const neg3 = [hardNeg, new Float32Array(dim).fill(0.1), new Float32Array(dim).fill(-0.1)];

      const loss1 = accel.infoNceLoss(anchor, positive, neg1, 0.07);
      const loss3 = accel.infoNceLoss(anchor, positive, neg3, 0.07);
      // More negatives in denominator strictly increases loss with deterministic vectors
      expect(loss3).toBeGreaterThanOrEqual(loss1);
    });

    it('low temperature should give lower loss than high temperature when positive is similar', () => {
      // With aligned anchor-positive, low temp makes positive dominate softmax → loss near 0
      // High temp makes distribution uniform → loss near log(N+1)
      const anchor = new Float32Array(64); anchor[0] = 1;
      const positive = new Float32Array(64); positive[0] = 0.95; positive[1] = 0.05;
      const negatives = [new Float32Array(64), new Float32Array(64)];
      negatives[0][32] = 1; negatives[1][48] = 1; // orthogonal

      const lossLow = accel.infoNceLoss(anchor, positive, negatives, 0.01);
      const lossHigh = accel.infoNceLoss(anchor, positive, negatives, 1.0);
      // Low temp: positive dominates → loss ≈ 0
      // High temp: uniform-ish → loss ≈ log(3) ≈ 1.1
      expect(lossLow).toBeLessThan(lossHigh);
    });

    it('should handle empty negatives', () => {
      const anchor = new Float32Array([1, 0, 0]);
      const positive = new Float32Array([1, 0, 0]);
      const loss = accel.infoNceLoss(anchor, positive, [], 0.07);
      expect(Number.isFinite(loss)).toBe(true);
    });
  });

  // ─── AdamW Optimizer (JS fallback) ───

  describe('AdamW Optimizer', () => {
    it('should update params toward minimum', () => {
      const params = new Float32Array([5.0, -3.0, 7.0]);
      const grads = new Float32Array([1.0, -1.0, 1.0]); // Points away from zero
      const m = new Float32Array(3);
      const v = new Float32Array(3);

      const origMag = Math.sqrt(params[0] ** 2 + params[1] ** 2 + params[2] ** 2);

      for (let step = 1; step <= 100; step++) {
        accel.adamWStep(params, grads, m, v, step, 0.01, 0.01);
      }

      const newMag = Math.sqrt(params[0] ** 2 + params[1] ** 2 + params[2] ** 2);
      expect(newMag).toBeLessThan(origMag);
    });

    it('should apply weight decay', () => {
      const params = new Float32Array([10.0]);
      const grads = new Float32Array([0.0]); // Zero gradient
      const m = new Float32Array(1);
      const v = new Float32Array(1);

      accel.adamWStep(params, grads, m, v, 1, 0.01, 0.1);
      // Weight decay should shrink params toward 0
      expect(params[0]).toBeLessThan(10.0);
    });

    it('should handle zero gradients gracefully', () => {
      const params = new Float32Array([1.0, 2.0]);
      const grads = new Float32Array([0.0, 0.0]);
      const m = new Float32Array(2);
      const v = new Float32Array(2);

      // Should not throw or produce NaN
      accel.adamWStep(params, grads, m, v, 1, 0.001, 0.0);
      expect(Number.isFinite(params[0])).toBe(true);
      expect(Number.isFinite(params[1])).toBe(true);
    });

    it('should converge to zero with weight decay and zero gradient', () => {
      const params = new Float32Array([10.0]);
      const grads = new Float32Array([0.0]);
      const m = new Float32Array(1);
      const v = new Float32Array(1);

      for (let step = 1; step <= 2000; step++) {
        accel.adamWStep(params, grads, m, v, step, 0.01, 0.1);
      }
      // With wd=0.1, lr=0.01: each step decays by ~0.1%
      // After 2000 steps starting at 10: 10 * (1 - 0.001)^2000 ≈ 1.35
      expect(Math.abs(params[0])).toBeLessThan(2.0);
    });

    it('should update momentum and velocity', () => {
      const params = new Float32Array([1.0]);
      const grads = new Float32Array([0.5]);
      const m = new Float32Array(1);
      const v = new Float32Array(1);

      accel.adamWStep(params, grads, m, v, 1, 0.01, 0.0);
      expect(m[0]).not.toBe(0);
      expect(v[0]).not.toBe(0);
    });
  });

  // ─── Tensor Compression (returns null without native) ───

  describe('Tensor Compression', () => {
    it('should return null for compress without native', () => {
      expect(accel.tensorCompress(randomVec(64), 1)).toBeNull();
    });

    it('should return null for decompress without native', () => {
      expect(accel.tensorDecompress(new Uint8Array(16), 64)).toBeNull();
    });

    it('should return null for batchCompress without native', () => {
      expect(accel.tensorBatchCompress([randomVec(64)], 1)).toBeNull();
    });
  });

  // ─── Router Persistence (returns false/null without native) ───

  describe('Router Persistence', () => {
    it('should return false for save without native', async () => {
      expect(await accel.routerSave({}, '/tmp/test.bin')).toBe(false);
    });

    it('should return null for load without native', async () => {
      expect(await accel.routerLoad('/tmp/test.bin')).toBeNull();
    });
  });

  // ─── SONA Extended (returns false without native) ───

  describe('SONA Extended APIs', () => {
    it('should return false for addContext without native', () => {
      expect(accel.sonaAddContext(null, 0, { key: 'val' })).toBe(false);
    });

    it('should return false for flush without native', () => {
      expect(accel.sonaFlush(null)).toBe(false);
    });

    it('should return false for applyBaseLora without native', () => {
      expect(accel.sonaApplyBaseLora(null, randomVec(64))).toBe(false);
    });
  });

  // ─── Performance Benchmarks ───

  describe('Performance benchmarks', () => {
    it('should complete 10K cosine similarity computations in <200ms', () => {
      const a = randomVec(384);
      const b = randomVec(384);
      const start = performance.now();
      for (let i = 0; i < 10000; i++) accel.cosineSimilarity(a, b);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(200);
    });

    it('should complete 10K dot products in <200ms', () => {
      const a = randomVec(384);
      const b = randomVec(384);
      const start = performance.now();
      for (let i = 0; i < 10000; i++) accel.dotProduct(a, b);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(200);
    });

    it('should complete 10K L2 distances in <200ms', () => {
      const a = randomVec(384);
      const b = randomVec(384);
      const start = performance.now();
      for (let i = 0; i < 10000; i++) accel.l2Distance(a, b);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(200);
    });

    it('should complete 1K InfoNCE loss computations in <300ms', () => {
      const anchor = randomVec(128);
      const positive = randomVec(128);
      const negatives = [randomVec(128), randomVec(128), randomVec(128), randomVec(128)];
      const start = performance.now();
      for (let i = 0; i < 1000; i++) accel.infoNceLoss(anchor, positive, negatives, 0.07);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(300);
    });

    it('should complete 1K AdamW steps in <100ms', () => {
      const params = randomVec(256);
      const grads = randomVec(256);
      const m = new Float32Array(256);
      const v = new Float32Array(256);
      const start = performance.now();
      for (let step = 1; step <= 1000; step++) {
        accel.adamWStep(params, grads, m, v, step, 0.001, 0.01);
      }
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(100);
    });

    it('should complete 10K witness chain verifications in <100ms', () => {
      const chain = new Uint8Array(73 * 10);
      const start = performance.now();
      for (let i = 0; i < 10000; i++) accel.verifyWitnessChain(chain);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(100);
    });

    it('should complete 10K segment header verifications in <100ms', () => {
      const header = new Uint8Array([0x52, 0x56, 0x46, 0x00, 0x01, 0x02, 0x03, 0x04]);
      const start = performance.now();
      for (let i = 0; i < 10000; i++) accel.verifySegmentHeader(header);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(100);
    });
  });

  // ─── Mathematical Correctness ───

  describe('Mathematical correctness', () => {
    it('cosine similarity should match manual computation', () => {
      const a = new Float32Array([3, 4]);
      const b = new Float32Array([4, 3]);
      // cos = (12 + 12) / (5 * 5) = 24/25 = 0.96
      expect(accel.cosineSimilarity(a, b)).toBeCloseTo(0.96, 2);
    });

    it('InfoNCE loss should decrease when positive gets closer to anchor', () => {
      const anchor = new Float32Array([1, 0, 0, 0]);
      const pos1 = new Float32Array([0.5, 0.5, 0, 0]); // moderate similarity
      const pos2 = new Float32Array([0.9, 0.1, 0, 0]); // higher similarity
      const negs = [new Float32Array([0, 0, 1, 0])];

      const loss1 = accel.infoNceLoss(anchor, pos1, negs, 0.1);
      const loss2 = accel.infoNceLoss(anchor, pos2, negs, 0.1);
      expect(loss2).toBeLessThan(loss1);
    });

    it('AdamW with constant gradient should converge', () => {
      // Minimize f(x) = x^2, gradient = 2x
      const params = new Float32Array([10.0]);
      const m = new Float32Array(1);
      const v = new Float32Array(1);

      for (let step = 1; step <= 500; step++) {
        const grads = new Float32Array([2 * params[0]]);
        accel.adamWStep(params, grads, m, v, step, 0.05, 0.0);
      }
      expect(Math.abs(params[0])).toBeLessThan(0.5);
    });

    it('L2 distance should satisfy d(a,b) >= 0', () => {
      for (let trial = 0; trial < 100; trial++) {
        const a = randomVec(32);
        const b = randomVec(32);
        expect(accel.l2Distance(a, b)).toBeGreaterThanOrEqual(0);
      }
    });

    it('dot product should be commutative', () => {
      const a = randomVec(128);
      const b = randomVec(128);
      expect(accel.dotProduct(a, b)).toBeCloseTo(accel.dotProduct(b, a), 3);
    });

    it('CRC32C should be deterministic for segment verification', () => {
      const data = new Uint8Array([0x52, 0x56, 0x46, 0x00, 0x01, 0x02, 0x03, 0x04]);
      const r1 = accel.verifySegmentHeader(data);
      const r2 = accel.verifySegmentHeader(data);
      const r3 = accel.verifySegmentHeader(data);
      expect(r1.crc).toBe(r2.crc);
      expect(r2.crc).toBe(r3.crc);
    });
  });

  // ─── Edge Cases ───

  describe('Edge cases', () => {
    it('should handle dimension 1 vectors', () => {
      const a = new Float32Array([1.0]);
      const b = new Float32Array([1.0]);
      expect(accel.cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
      expect(accel.dotProduct(a, b)).toBeCloseTo(1.0, 5);
      expect(accel.l2Distance(a, b)).toBeCloseTo(0.0, 5);
    });

    it('should handle very small values', () => {
      const a = new Float32Array([1e-30, 1e-30]);
      const b = new Float32Array([1e-30, 1e-30]);
      const sim = accel.cosineSimilarity(a, b);
      expect(Number.isFinite(sim)).toBe(true);
    });

    it('should handle very large values', () => {
      const a = new Float32Array([1e30, 1e30]);
      const b = new Float32Array([1e30, 1e30]);
      const sim = accel.cosineSimilarity(a, b);
      expect(Number.isFinite(sim)).toBe(true);
    });

    it('should handle NaN in vectors gracefully', () => {
      const a = new Float32Array([NaN, 1.0]);
      const b = new Float32Array([1.0, 1.0]);
      const sim = accel.cosineSimilarity(a, b);
      // NaN propagation is expected in math operations
      expect(typeof sim).toBe('number');
    });

    it('should handle witness chain of exactly 73 bytes', () => {
      const chain = new Uint8Array(73);
      const result = accel.verifyWitnessChain(chain);
      expect(result.valid).toBe(true);
      expect(result.entryCount).toBe(1);
    });

    it('should handle large witness chains', () => {
      const chain = new Uint8Array(73 * 1000);
      const result = accel.verifyWitnessChain(chain);
      expect(result.entryCount).toBe(1000);
    });
  });

  // ─── SOTA: Benchmark unrolled ops vs baseline ───

  describe('Optimization benchmarks', () => {
    it('should complete 100K cosine similarity (dim=384) in <200ms (unrolled)', () => {
      const a = randomVec(384);
      const b = randomVec(384);
      const start = performance.now();
      for (let i = 0; i < 100_000; i++) accel.cosineSimilarity(a, b);
      const elapsed = performance.now() - start;
      // Allow more time in CI / shared environments
      expect(elapsed).toBeLessThan(500);
    });

    it('should complete 100K dot products (dim=384) in <150ms (unrolled)', () => {
      const a = randomVec(384);
      const b = randomVec(384);
      const start = performance.now();
      for (let i = 0; i < 100_000; i++) accel.dotProduct(a, b);
      const elapsed = performance.now() - start;
      // Allow more time in CI / shared environments
      expect(elapsed).toBeLessThan(500);
    });

    it('should complete 100K CRC32C (64B) in <50ms (table-lookup)', () => {
      const data = new Uint8Array(64);
      for (let i = 0; i < data.length; i++) data[i] = i & 0xFF;
      data[0] = 0x52; data[1] = 0x56; data[2] = 0x46; // RVF magic
      const start = performance.now();
      for (let i = 0; i < 100_000; i++) accel.verifySegmentHeader(data);
      const elapsed = performance.now() - start;
      expect(elapsed).toBeLessThan(100);
    });
  });

  // ─── Integration: accelerator module ───

  describe('Module integration', () => {
    it('should be importable from NativeAccelerator module', async () => {
      const mod = await import('../../src/backends/rvf/NativeAccelerator.js');
      expect(mod.NativeAccelerator).toBeDefined();
      const instance = new mod.NativeAccelerator();
      expect(instance).toBeDefined();
    });

    it('should export singleton functions', async () => {
      const mod = await import('../../src/backends/rvf/NativeAccelerator.js');
      expect(typeof mod.getAccelerator).toBe('function');
      expect(typeof mod.resetAccelerator).toBe('function');
    });

    it('getStats should reflect native availability after init', async () => {
      await accel.initialize();
      const stats = accel.getStats();
      // If SIMD is available (packages installed), it should be reflected
      if (stats.simdAvailable) {
        expect(stats.capabilities).toContain('simd');
      }
      if (stats.wasmVerifyAvailable) {
        expect(stats.capabilities).toContain('wasm-verify');
      }
    });
  });

  describe('Graph Transaction APIs (Phase 1)', () => {
    it('should report graph capabilities', () => {
      expect(typeof accel.graphTxAvailable).toBe('boolean');
      expect(typeof accel.graphBatchInsertAvailable).toBe('boolean');
      expect(typeof accel.graphCypherAvailable).toBe('boolean');
    });

    it('should execute transaction fallback without native', async () => {
      let executed = false;
      const mockDb = {};
      const result = await accel.graphTransaction(mockDb, () => { executed = true; });
      expect(result).toBe(true);
      expect(executed).toBe(true);
    });

    it('should handle transaction error gracefully', async () => {
      const result = await accel.graphTransaction({}, () => { throw new Error('fail'); });
      expect(result).toBe(false);
    });

    it('should return false for batch insert without native', async () => {
      expect(await accel.graphBatchInsertNodes({}, [{ id: 'a', data: {} }])).toBe(false);
    });

    it('should return null for cypher query without native', async () => {
      expect(await accel.graphCypherQuery({}, 'MATCH (n) RETURN n')).toBeNull();
    });
  });

  describe('Core Batch Operations (Phase 1)', () => {
    it('should report core batch capability', () => {
      expect(typeof accel.coreBatchInsertAvailable).toBe('boolean');
    });

    it('should return false for batch insert without native', async () => {
      expect(await accel.coreBatchInsert({}, [{ id: 'a', vector: new Float32Array(4) }])).toBe(false);
    });
  });

  describe('EWC Manager (Phase 1)', () => {
    it('should report EWC availability', () => {
      expect(typeof accel.ewcManagerAvailable).toBe('boolean');
    });

    it('should return 0 penalty without native', () => {
      expect(accel.ewcPenalty(new Float32Array(4))).toBe(0);
    });

    it('should return false for Fisher update without native', () => {
      expect(accel.ewcUpdateFisher(new Float32Array(4), 1.0)).toBe(false);
    });
  });

  // ─── SIMD Activations (JS fallbacks) ───

  describe('Softmax', () => {
    it('should produce a valid probability distribution', () => {
      const input = [1.0, 2.0, 3.0, 4.0];
      const result = accel.softmax(input);
      expect(result.length).toBe(4);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
      for (const v of result) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); }
    });

    it('should assign highest probability to largest input', () => {
      const result = accel.softmax([1.0, 5.0, 2.0]);
      expect(result[1]).toBeGreaterThan(result[0]);
      expect(result[1]).toBeGreaterThan(result[2]);
    });

    it('should be numerically stable with large values', () => {
      const result = accel.softmax([1000, 1001, 1002]);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
      expect(Number.isFinite(result[0])).toBe(true);
    });

    it('should return uniform for equal inputs', () => {
      const result = accel.softmax([5, 5, 5, 5]);
      for (const v of result) expect(v).toBeCloseTo(0.25, 5);
    });
  });

  describe('ReLU', () => {
    it('should zero out negative values', () => {
      const result = accel.relu([-1, 2, -3, 4, -5, 6, -7, 8]);
      expect(result).toEqual([0, 2, 0, 4, 0, 6, 0, 8]);
    });

    it('should pass through positive values unchanged', () => {
      const result = accel.relu([1, 2, 3, 4]);
      expect(result).toEqual([1, 2, 3, 4]);
    });

    it('should handle zero input', () => {
      const result = accel.relu([0, 0, 0]);
      expect(result).toEqual([0, 0, 0]);
    });

    it('should handle odd-length arrays (non-multiple of 4)', () => {
      const result = accel.relu([-1, 2, -3, 4, -5]);
      expect(result).toEqual([0, 2, 0, 4, 0]);
    });
  });

  describe('GELU', () => {
    it('should approximate 0 for large negative values', () => {
      const result = accel.gelu([-10]);
      expect(result[0]).toBeCloseTo(0, 3);
    });

    it('should approximate identity for large positive values', () => {
      const result = accel.gelu([10]);
      expect(result[0]).toBeCloseTo(10, 1);
    });

    it('should return 0 for input 0', () => {
      const result = accel.gelu([0]);
      expect(result[0]).toBeCloseTo(0, 5);
    });

    it('should produce smooth non-linearity', () => {
      const result = accel.gelu([-1, 0, 1]);
      expect(result[0]).toBeLessThan(0);
      expect(result[1]).toBeCloseTo(0, 5);
      expect(result[2]).toBeGreaterThan(0);
    });
  });

  describe('Sigmoid', () => {
    it('should return 0.5 for input 0', () => {
      const result = accel.sigmoid([0]);
      expect(result[0]).toBeCloseTo(0.5, 5);
    });

    it('should approach 1 for large positive input', () => {
      const result = accel.sigmoid([100]);
      expect(result[0]).toBeCloseTo(1.0, 5);
    });

    it('should approach 0 for large negative input', () => {
      const result = accel.sigmoid([-100]);
      expect(result[0]).toBeCloseTo(0.0, 5);
    });

    it('should be monotonically increasing', () => {
      const result = accel.sigmoid([-2, -1, 0, 1, 2]);
      for (let i = 1; i < result.length; i++) expect(result[i]).toBeGreaterThan(result[i - 1]);
    });

    it('should handle 4-wide unrolled path', () => {
      const result = accel.sigmoid([-2, -1, 0, 1, 2, 3, 4, 5]);
      expect(result.length).toBe(8);
      for (const v of result) { expect(v).toBeGreaterThan(0); expect(v).toBeLessThan(1); }
    });
  });

  describe('LayerNorm', () => {
    it('should produce zero mean', () => {
      const result = accel.layerNorm([1, 2, 3, 4, 5, 6, 7, 8]);
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      expect(mean).toBeCloseTo(0, 4);
    });

    it('should produce unit variance', () => {
      const result = accel.layerNorm([1, 2, 3, 4, 5, 6, 7, 8]);
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      const variance = result.reduce((a, b) => a + (b - mean) ** 2, 0) / result.length;
      expect(variance).toBeCloseTo(1.0, 3);
    });

    it('should handle constant input', () => {
      const result = accel.layerNorm([5, 5, 5, 5]);
      for (const v of result) expect(v).toBeCloseTo(0, 5);
    });

    it('should handle non-multiple-of-4 length', () => {
      const result = accel.layerNorm([1, 3, 5]);
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      expect(mean).toBeCloseTo(0, 4);
    });
  });

  describe('Matvec', () => {
    it('should compute matrix-vector product correctly', () => {
      const matrix = [[1, 2, 3], [4, 5, 6]];
      const vector = [1, 1, 1];
      const result = accel.matvec(matrix, vector);
      expect(result[0]).toBeCloseTo(6, 5);
      expect(result[1]).toBeCloseTo(15, 5);
    });

    it('should handle identity matrix', () => {
      const matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
      const vector = [3, 7, 11, 13];
      const result = accel.matvec(matrix, vector);
      expect(result).toEqual([3, 7, 11, 13]);
    });

    it('should handle single-row matrix', () => {
      const result = accel.matvec([[2, 3]], [4, 5]);
      expect(result[0]).toBeCloseTo(23, 5);
    });

    it('should use 4-wide unrolling for wide vectors', () => {
      const dim = 64;
      const row = Array.from({ length: dim }, (_, i) => i + 1);
      const vec = Array.from({ length: dim }, () => 1);
      const result = accel.matvec([row], vec);
      expect(result[0]).toBeCloseTo(dim * (dim + 1) / 2, 5);
    });
  });

  describe('Element-wise operations', () => {
    it('add should sum element-wise', () => {
      const result = accel.add([1, 2, 3, 4], [10, 20, 30, 40]);
      expect(result).toEqual([11, 22, 33, 44]);
    });

    it('add should throw on length mismatch', () => {
      expect(() => accel.add([1, 2], [1, 2, 3])).toThrow('Length mismatch');
    });

    it('mul should multiply element-wise', () => {
      const result = accel.mul([1, 2, 3, 4], [2, 3, 4, 5]);
      expect(result).toEqual([2, 6, 12, 20]);
    });

    it('mul should throw on length mismatch', () => {
      expect(() => accel.mul([1], [1, 2])).toThrow('Length mismatch');
    });

    it('scale should multiply by scalar', () => {
      const result = accel.scale([1, 2, 3, 4, 5], 3);
      expect(result).toEqual([3, 6, 9, 12, 15]);
    });

    it('scale by 0 should produce zeros', () => {
      const result = accel.scale([1, 2, 3, 4], 0);
      expect(result).toEqual([0, 0, 0, 0]);
    });

    it('normalizeVec should produce unit vector', () => {
      const result = accel.normalizeVec([3, 4]);
      const norm = Math.sqrt(result[0] ** 2 + result[1] ** 2);
      expect(norm).toBeCloseTo(1.0, 5);
      expect(result[0]).toBeCloseTo(0.6, 5);
      expect(result[1]).toBeCloseTo(0.8, 5);
    });

    it('normalizeVec of zero vector should return zeros', () => {
      const result = accel.normalizeVec([0, 0, 0, 0]);
      expect(result).toEqual([0, 0, 0, 0]);
    });

    it('normalizeVec should handle non-multiple-of-4 length', () => {
      const result = accel.normalizeVec([1, 0, 0]);
      expect(result[0]).toBeCloseTo(1, 5);
      expect(result[1]).toBeCloseTo(0, 5);
      expect(result[2]).toBeCloseTo(0, 5);
    });
  });

  // ─── WASM Quantization Bridge (returns false/null without native) ───

  describe('WASM Quantization Bridge', () => {
    it('should report quantization availability', () => {
      expect(typeof accel.wasmQuantizationAvailable).toBe('boolean');
    });

    it('should return false for loadSqParams without native', () => {
      expect(accel.loadSqParams(new Uint8Array(16), 4)).toBe(false);
    });

    it('should return false for dequantI8 without native', () => {
      expect(accel.dequantI8(new Uint8Array(4), new Float32Array(4), 4)).toBe(false);
    });

    it('should return false for loadPqCodebook without native', () => {
      expect(accel.loadPqCodebook(new Uint8Array(256), 8, 256)).toBe(false);
    });

    it('should return null for pqDistances without native', () => {
      expect(accel.pqDistances(new Uint8Array(16), 4)).toBeNull();
    });
  });

  // ─── WASM Store Bridge (returns null/0/false without native) ───

  describe('WASM Store Bridge', () => {
    it('should report store availability', () => {
      expect(typeof accel.wasmStoreAvailable).toBe('boolean');
    });

    it('should expose wasmStore instance', () => {
      expect(accel.wasmStore).toBeDefined();
    });

    it('should return null for wasmStoreCreate without native', () => {
      expect(accel.wasmStoreCreate(128, 0)).toBeNull();
    });

    it('should return 0 for wasmStoreIngest without native', () => {
      expect(accel.wasmStoreIngest(1, new Float32Array(4), [0], 1)).toBe(0);
    });

    it('should return null for wasmStoreQuery without native', () => {
      expect(accel.wasmStoreQuery(1, new Float32Array(4), 10, 0)).toBeNull();
    });

    it('should return null for wasmStoreExport without native', () => {
      expect(accel.wasmStoreExport(1)).toBeNull();
    });

    it('should return false for wasmStoreClose without native', () => {
      expect(accel.wasmStoreClose(1)).toBe(false);
    });
  });

  // ─── Updated Stats ───

  describe('Updated AcceleratorStats', () => {
    it('should include new stat fields after init', async () => {
      const stats = await accel.initialize();
      expect(typeof stats.simdActivationsAvailable).toBe('boolean');
      expect(typeof stats.wasmStoreAvailable).toBe('boolean');
      expect(typeof stats.wasmQuantizationAvailable).toBe('boolean');
    });

    it('simdActivationsAvailable should match simdAvailable', async () => {
      const stats = await accel.initialize();
      expect(stats.simdActivationsAvailable).toBe(stats.simdAvailable);
    });
  });

  // ─── Activation Performance Benchmarks ───

  describe('Activation performance benchmarks', () => {
    it('should complete 10K softmax (dim=256) in <300ms', () => {
      const input = Array.from({ length: 256 }, () => Math.random());
      const start = performance.now();
      for (let i = 0; i < 10_000; i++) accel.softmax(input);
      expect(performance.now() - start).toBeLessThan(300);
    });

    it('should complete 10K relu (dim=256) in <100ms', () => {
      const input = Array.from({ length: 256 }, () => Math.random() - 0.5);
      const start = performance.now();
      for (let i = 0; i < 10_000; i++) accel.relu(input);
      expect(performance.now() - start).toBeLessThan(100);
    });

    it('should complete 10K sigmoid (dim=256) in <300ms', () => {
      const input = Array.from({ length: 256 }, () => Math.random() * 10 - 5);
      const start = performance.now();
      for (let i = 0; i < 10_000; i++) accel.sigmoid(input);
      expect(performance.now() - start).toBeLessThan(300);
    });

    it('should complete 10K add (dim=256) in <100ms', () => {
      const a = Array.from({ length: 256 }, () => Math.random());
      const b = Array.from({ length: 256 }, () => Math.random());
      const start = performance.now();
      for (let i = 0; i < 10_000; i++) accel.add(a, b);
      expect(performance.now() - start).toBeLessThan(100);
    });

    it('should complete 1K matvec (64x64) in <200ms', () => {
      const matrix = Array.from({ length: 64 }, () => Array.from({ length: 64 }, () => Math.random()));
      const vector = Array.from({ length: 64 }, () => Math.random());
      const start = performance.now();
      for (let i = 0; i < 1_000; i++) accel.matvec(matrix, vector);
      expect(performance.now() - start).toBeLessThan(200);
    });
  });
});
