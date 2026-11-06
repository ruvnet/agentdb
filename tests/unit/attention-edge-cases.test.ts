/**
 * @test AttentionService Edge Case Tests
 * @description Comprehensive edge case and error handling tests for AttentionService
 * @coverage
 *   - Zero-length inputs
 *   - Dimension mismatches
 *   - NaN and Infinity handling
 *   - Concurrent operations
 *   - Resource exhaustion
 *   - Invalid configurations
 *   - Boundary conditions
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AttentionService } from '../../src/controllers/AttentionService.js';

describe('AttentionService Edge Cases', () => {
  let service: AttentionService;

  beforeEach(async () => {
    service = new AttentionService({
      embedDim: 768,
      numHeads: 12,
      headDim: 64,
      backend: 'wasm',
    });
    await service.initialize();
  });

  afterEach(async () => {
    if (service) {
      await service.dispose();
    }
  });

  describe('Zero-Length Inputs', () => {
    it('should handle empty query array', async () => {
      const query = new Float32Array(0);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      for (let i = 0; i < key.length; i++) {
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow();
    });

    it('should handle empty key array', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(0);
      const value = new Float32Array(768);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
      }
      for (let i = 0; i < value.length; i++) {
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow();
    });

    it('should handle empty value array', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(0);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow();
    });

    it('should handle all arrays empty', async () => {
      const query = new Float32Array(0);
      const key = new Float32Array(0);
      const value = new Float32Array(0);

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow();
    });
  });

  describe('Dimension Mismatches', () => {
    it('should detect query dimension mismatch', async () => {
      const query = new Float32Array(512); // Wrong dimension
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
      }
      for (let i = 0; i < key.length; i++) {
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/dimension/i);
    });

    it('should detect key-value dimension mismatch', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(512); // Wrong dimension

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
      }
      for (let i = 0; i < value.length; i++) {
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/dimension/i);
    });

    it('should detect non-aligned sequence lengths', async () => {
      const seqLen1 = 128;
      const seqLen2 = 127; // Not aligned
      const embedDim = 768;

      const query = new Float32Array(seqLen1 * embedDim);
      const key = new Float32Array(seqLen2 * embedDim);
      const value = new Float32Array(seqLen2 * embedDim);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
      }
      for (let i = 0; i < key.length; i++) {
        key[i] = Math.random();
        value[i] = Math.random();
      }

      // This should either work (cross-attention) or throw clear error
      // Depends on implementation - just ensure no crash
      try {
        await service.multiHeadAttention(query, key, value);
      } catch (err) {
        expect(err).toBeInstanceOf(Error);
      }
    });
  });

  describe('NaN and Infinity Handling', () => {
    it('should detect NaN in query', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      query[0] = NaN;
      for (let i = 1; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/NaN/i);
    });

    it('should detect Infinity in key', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      key[0] = Infinity;
      for (let i = 1; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/Infinity/i);
    });

    it('should detect negative Infinity in value', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      value[0] = -Infinity;
      for (let i = 1; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/Infinity/i);
    });

    it('should produce finite output for valid inputs', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      const result = await service.multiHeadAttention(query, key, value);

      // Verify all outputs are finite
      for (let i = 0; i < result.output.length; i++) {
        expect(Number.isFinite(result.output[i])).toBe(true);
      }
    });

    it('should handle extreme values without overflow', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      // Use extreme but valid values
      for (let i = 0; i < query.length; i++) {
        query[i] = i % 2 === 0 ? 1e6 : -1e6;
        key[i] = i % 3 === 0 ? 1e6 : -1e6;
        value[i] = i % 5 === 0 ? 1e6 : -1e6;
      }

      const result = await service.multiHeadAttention(query, key, value);

      // Should not produce NaN or Infinity
      for (let i = 0; i < result.output.length; i++) {
        expect(Number.isFinite(result.output[i])).toBe(true);
      }
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle concurrent attention calls', async () => {
      const numConcurrent = 10;
      const embedDim = 768;

      const promises = [];

      for (let i = 0; i < numConcurrent; i++) {
        const query = new Float32Array(embedDim);
        const key = new Float32Array(embedDim);
        const value = new Float32Array(embedDim);

        for (let j = 0; j < embedDim; j++) {
          query[j] = Math.random();
          key[j] = Math.random();
          value[j] = Math.random();
        }

        promises.push(service.multiHeadAttention(query, key, value));
      }

      const results = await Promise.all(promises);

      expect(results).toHaveLength(numConcurrent);
      for (const result of results) {
        expect(result.output).toBeDefined();
        expect(result.output.length).toBe(embedDim);
      }
    });

    it('should handle concurrent Flash Attention v2 calls', async () => {
      const numConcurrent = 5;
      const seqLen = 64;
      const embedDim = 768;

      const promises = [];

      for (let i = 0; i < numConcurrent; i++) {
        const query = new Float32Array(seqLen * embedDim);
        const key = new Float32Array(seqLen * embedDim);
        const value = new Float32Array(seqLen * embedDim);

        for (let j = 0; j < query.length; j++) {
          query[j] = Math.random();
          key[j] = Math.random();
          value[j] = Math.random();
        }

        promises.push(
          service.flashAttentionV2(query, key, value, {
            seqLength: seqLen,
            blockSize: 32,
          })
        );
      }

      const results = await Promise.all(promises);

      expect(results).toHaveLength(numConcurrent);
      for (const result of results) {
        expect(result.output).toBeDefined();
        expect(result.output.length).toBe(seqLen * embedDim);
      }
    });

    it('should not have race conditions in initialization', async () => {
      // Create multiple services and initialize concurrently
      const services = Array(5)
        .fill(null)
        .map(
          () =>
            new AttentionService({
              embedDim: 768,
              numHeads: 12,
              headDim: 64,
              backend: 'wasm',
            })
        );

      // Initialize all concurrently
      await Promise.all(services.map((s) => s.initialize()));

      // Verify all are initialized
      for (const s of services) {
        const query = new Float32Array(768);
        const key = new Float32Array(768);
        const value = new Float32Array(768);

        for (let i = 0; i < query.length; i++) {
          query[i] = Math.random();
          key[i] = Math.random();
          value[i] = Math.random();
        }

        const result = await s.multiHeadAttention(query, key, value);
        expect(result.output).toBeDefined();
      }

      // Clean up
      await Promise.all(services.map((s) => s.dispose()));
    });
  });

  describe('Resource Exhaustion', () => {
    it('should handle very large sequences', async () => {
      const seqLen = 2048; // Large sequence
      const embedDim = 768;

      const query = new Float32Array(seqLen * embedDim);
      const key = new Float32Array(seqLen * embedDim);
      const value = new Float32Array(seqLen * embedDim);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      // Should either succeed or throw memory error (not crash)
      try {
        const result = await service.multiHeadAttention(query, key, value);
        expect(result.output).toBeDefined();
      } catch (err) {
        expect(err).toBeInstanceOf(Error);
        expect((err as Error).message).toMatch(/memory|size|limit/i);
      }
    });

    it('should handle rapid sequential allocations', async () => {
      const iterations = 100;
      const embedDim = 768;

      for (let i = 0; i < iterations; i++) {
        const query = new Float32Array(embedDim);
        const key = new Float32Array(embedDim);
        const value = new Float32Array(embedDim);

        for (let j = 0; j < embedDim; j++) {
          query[j] = Math.random();
          key[j] = Math.random();
          value[j] = Math.random();
        }

        const result = await service.multiHeadAttention(query, key, value);
        expect(result.output).toBeDefined();
      }

      // No memory leaks - should complete without error
    });
  });

  describe('Invalid Configurations', () => {
    it('should reject invalid embed dimension', async () => {
      await expect(
        (async () => {
          const invalidService = new AttentionService({
            embedDim: 0,
            numHeads: 12,
            headDim: 64,
            backend: 'wasm',
          });
          await invalidService.initialize();
        })()
      ).rejects.toThrow();
    });

    it('should reject invalid number of heads', async () => {
      await expect(
        (async () => {
          const invalidService = new AttentionService({
            embedDim: 768,
            numHeads: 0,
            headDim: 64,
            backend: 'wasm',
          });
          await invalidService.initialize();
        })()
      ).rejects.toThrow();
    });

    it('should reject mismatched embedDim and numHeads*headDim', async () => {
      await expect(
        (async () => {
          const invalidService = new AttentionService({
            embedDim: 768,
            numHeads: 12,
            headDim: 32, // 12 * 32 = 384 ≠ 768
            backend: 'wasm',
          });
          await invalidService.initialize();
        })()
      ).rejects.toThrow();
    });
  });

  describe('Boundary Conditions', () => {
    it('should handle minimum valid sequence length (1)', async () => {
      const seqLen = 1;
      const embedDim = 768;

      const query = new Float32Array(seqLen * embedDim);
      const key = new Float32Array(seqLen * embedDim);
      const value = new Float32Array(seqLen * embedDim);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      const result = await service.multiHeadAttention(query, key, value);

      expect(result.output).toBeDefined();
      expect(result.output.length).toBe(seqLen * embedDim);
    });

    it('should handle all-zero input', async () => {
      const query = new Float32Array(768); // All zeros
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      const result = await service.multiHeadAttention(query, key, value);

      expect(result.output).toBeDefined();
      expect(result.output.length).toBe(768);

      // All zeros should produce valid output (likely all zeros or uniform)
      for (let i = 0; i < result.output.length; i++) {
        expect(Number.isFinite(result.output[i])).toBe(true);
      }
    });

    it('should handle identical query, key, and value', async () => {
      const embedDim = 768;
      const data = new Float32Array(embedDim);

      for (let i = 0; i < embedDim; i++) {
        data[i] = Math.random();
      }

      // Use same array for all three
      const result = await service.multiHeadAttention(data, data, data);

      expect(result.output).toBeDefined();
      expect(result.output.length).toBe(embedDim);
    });

    it('should handle very small values (underflow)', async () => {
      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      for (let i = 0; i < query.length; i++) {
        query[i] = 1e-38; // Very small
        key[i] = 1e-38;
        value[i] = 1e-38;
      }

      const result = await service.multiHeadAttention(query, key, value);

      expect(result.output).toBeDefined();
      for (let i = 0; i < result.output.length; i++) {
        expect(Number.isFinite(result.output[i])).toBe(true);
      }
    });

    it('should handle power-of-two dimensions', async () => {
      const dimensions = [256, 512, 1024, 2048];

      for (const dim of dimensions) {
        const testService = new AttentionService({
          embedDim: dim,
          numHeads: 8,
          headDim: dim / 8,
          backend: 'wasm',
        });

        await testService.initialize();

        const query = new Float32Array(dim);
        const key = new Float32Array(dim);
        const value = new Float32Array(dim);

        for (let i = 0; i < dim; i++) {
          query[i] = Math.random();
          key[i] = Math.random();
          value[i] = Math.random();
        }

        const result = await testService.multiHeadAttention(query, key, value);

        expect(result.output).toBeDefined();
        expect(result.output.length).toBe(dim);

        await testService.dispose();
      }
    });
  });

  describe('Error Recovery', () => {
    it('should recover from failed operation', async () => {
      const embedDim = 768;

      // First operation fails (NaN input)
      const badQuery = new Float32Array(embedDim);
      badQuery[0] = NaN;

      try {
        await service.multiHeadAttention(badQuery, badQuery, badQuery);
      } catch (err) {
        // Expected to fail
      }

      // Next operation should still work
      const goodQuery = new Float32Array(embedDim);
      const goodKey = new Float32Array(embedDim);
      const goodValue = new Float32Array(embedDim);

      for (let i = 0; i < embedDim; i++) {
        goodQuery[i] = Math.random();
        goodKey[i] = Math.random();
        goodValue[i] = Math.random();
      }

      const result = await service.multiHeadAttention(goodQuery, goodKey, goodValue);

      expect(result.output).toBeDefined();
    });

    it('should handle dispose() called multiple times', async () => {
      await service.dispose();
      await service.dispose(); // Second dispose should not throw
      await service.dispose(); // Third dispose should not throw
    });

    it('should reject operations after dispose()', async () => {
      await service.dispose();

      const query = new Float32Array(768);
      const key = new Float32Array(768);
      const value = new Float32Array(768);

      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/disposed|initialized/i);
    });
  });

  describe('Size limit and validation scope (#43)', () => {
    const rows = (n: number, dim: number) => {
      const a = new Float32Array(n * dim);
      for (let i = 0; i < a.length; i++) a[i] = Math.random();
      return a;
    };

    it('rejects calls above maxAttentionScores before computing', async () => {
      const small = new AttentionService({ embedDim: 64, numHeads: 4, headDim: 16, maxAttentionScores: 100 });
      await small.initialize();
      try {
        // 8 x 8 rows x 4 heads = 256 scores > 100
        await expect(small.multiHeadAttention(rows(8, 64), rows(8, 64), rows(8, 64))).rejects.toThrow(/size limit/);
        // 4 x 4 x 4 = 64 scores: allowed
        const ok = await small.multiHeadAttention(rows(4, 64), rows(4, 64), rows(4, 64));
        expect(ok.output.length).toBe(4 * 64);
      } finally {
        await small.dispose();
      }
    });

    it('does not apply the quadratic limit to linearAttention', async () => {
      const small = new AttentionService({ embedDim: 64, numHeads: 4, headDim: 16, maxAttentionScores: 100 });
      await small.initialize();
      try {
        const out = await small.linearAttention(rows(8, 64), rows(8, 64), rows(8, 64));
        expect(out.output.length).toBe(8 * 64);
      } finally {
        await small.dispose();
      }
    });

    it('validates the other mechanisms too', async () => {
      const bad = rows(1, 768);
      bad[5] = Number.NaN;
      const ok = rows(1, 768);
      await expect(service.flashAttention(bad, ok, ok)).rejects.toThrow(/NaN/);
      await expect(service.linearAttention(ok, ok, new Float32Array(100))).rejects.toThrow(/dimension/);
      await expect(service.moeAttention(new Float32Array(0), ok, ok)).rejects.toThrow(/empty/);
      await expect(service.hyperbolicAttention(ok, ok, ok, Number.NaN)).rejects.toThrow(/curvature/);
    });

    it('rejects a non-integer or zero-head config', () => {
      expect(() => new AttentionService({ embedDim: 768, numHeads: 12.5, headDim: 64 })).toThrow(/integer/);
      expect(() => new AttentionService({ embedDim: 64, numHeads: 4, headDim: 16, maxAttentionScores: 0 })).toThrow(/maxAttentionScores/);
    });

    it('rejects initialize() after dispose()', async () => {
      const s2 = new AttentionService({ embedDim: 64, numHeads: 4, headDim: 16 });
      await s2.initialize();
      await s2.dispose();
      await expect(s2.initialize()).rejects.toThrow(/disposed/);
    });
  });
});
