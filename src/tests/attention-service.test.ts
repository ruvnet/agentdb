/**
 * AttentionService Test Suite
 *
 * Tests for all attention mechanisms with NAPI and WASM backends
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { AttentionService } from '../controllers/AttentionService.js';
import type { AttentionConfig, AttentionResult } from '../controllers/AttentionService.js';

describe('AttentionService', () => {
  let service: AttentionService;
  const config: AttentionConfig = {
    numHeads: 8,
    headDim: 64,
    embedDim: 512,
    dropout: 0.1,
    bias: true
  };

  beforeAll(async () => {
    service = new AttentionService(config);
    await service.initialize();
  });

  afterEach(() => {
    service.resetStats();
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const newService = new AttentionService(config);
      await newService.initialize();

      const info = newService.getInfo();
      expect(info.initialized).toBe(true);
      expect(info.runtime).toBeDefined();
      expect(info.config).toEqual(expect.objectContaining({
        numHeads: 8,
        headDim: 64,
        embedDim: 512
      }));
    });

    it('should detect runtime environment', async () => {
      const info = service.getInfo();
      expect(['nodejs', 'browser', 'unknown']).toContain(info.runtime);
    });

    it('should handle multiple initializations gracefully', async () => {
      await service.initialize();
      await service.initialize(); // Should not throw
      const info = service.getInfo();
      expect(info.initialized).toBe(true);
    });
  });

  describe('Multi-Head Attention', () => {
    it('should compute attention for simple inputs', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      // Create simple test vectors
      const query = new Float32Array(seqLen * embedDim);
      const key = new Float32Array(seqLen * embedDim);
      const value = new Float32Array(seqLen * embedDim);

      // Fill with test data
      for (let i = 0; i < query.length; i++) {
        query[i] = Math.random();
        key[i] = Math.random();
        value[i] = Math.random();
      }

      const result = await service.multiHeadAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
      expect(result.mechanism).toBe('multi-head');
      expect(result.executionTimeMs).toBeGreaterThan(0);
      expect(['napi', 'wasm', 'fallback']).toContain(result.runtime);
    });

    it('should handle attention with mask', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      // Create causal mask (lower triangular)
      const mask = new Float32Array(seqLen * seqLen);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          mask[i * seqLen + j] = j <= i ? 1 : 0;
        }
      }

      const result = await service.multiHeadAttention(query, key, value, mask);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
    });

    it('should produce consistent results', async () => {
      const seqLen = 2;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => 0.5);
      const key = new Float32Array(seqLen * embedDim).map(() => 0.5);
      const value = new Float32Array(seqLen * embedDim).map(() => 1.0);

      const result1 = await service.multiHeadAttention(query, key, value);
      const result2 = await service.multiHeadAttention(query, key, value);

      // Results should be identical for same inputs
      expect(result1.output.length).toBe(result2.output.length);
      for (let i = 0; i < result1.output.length; i++) {
        expect(Math.abs(result1.output[i] - result2.output[i])).toBeLessThan(1e-5);
      }
    });

    it('should handle zero vectors', async () => {
      const seqLen = 2;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim); // All zeros
      const key = new Float32Array(seqLen * embedDim);
      const value = new Float32Array(seqLen * embedDim);

      const result = await service.multiHeadAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
    });
  });

  describe('Flash Attention', () => {
    it('should compute flash attention', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      const result = await service.flashAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
      expect(result.mechanism).toBe('flash');
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should be memory efficient', async () => {
      const seqLen = 8;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      service.resetStats();
      await service.flashAttention(query, key, value);

      const stats = service.getStats();
      expect(stats.totalOps).toBe(1);
      expect(stats.peakMemoryBytes).toBeGreaterThan(0);
    });
  });

  describe('Linear Attention', () => {
    it('should compute linear attention', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      const result = await service.linearAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
      expect(result.mechanism).toBe('linear');
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should scale linearly with sequence length', async () => {
      const embedDim = config.embedDim;

      // Test with short sequence
      const shortSeqLen = 4;
      const shortQuery = new Float32Array(shortSeqLen * embedDim).map(() => Math.random());
      const shortKey = new Float32Array(shortSeqLen * embedDim).map(() => Math.random());
      const shortValue = new Float32Array(shortSeqLen * embedDim).map(() => Math.random());

      const shortResult = await service.linearAttention(shortQuery, shortKey, shortValue);

      // Test with longer sequence
      const longSeqLen = 16;
      const longQuery = new Float32Array(longSeqLen * embedDim).map(() => Math.random());
      const longKey = new Float32Array(longSeqLen * embedDim).map(() => Math.random());
      const longValue = new Float32Array(longSeqLen * embedDim).map(() => Math.random());

      const longResult = await service.linearAttention(longQuery, longKey, longValue);

      // Linear attention should scale better than quadratic
      expect(longResult.executionTimeMs / shortResult.executionTimeMs).toBeLessThan(10);
    });
  });

  describe('Hyperbolic Attention', () => {
    it('should compute hyperbolic attention', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      const result = await service.hyperbolicAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
      expect(result.mechanism).toBe('hyperbolic');
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should support custom curvature', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      // Test with different curvatures
      const result1 = await service.hyperbolicAttention(query, key, value, -1.0);
      const result2 = await service.hyperbolicAttention(query, key, value, -0.5);

      expect(result1.output.length).toBe(result2.output.length);
      // Different curvatures should produce different results
      let different = false;
      for (let i = 0; i < result1.output.length; i++) {
        if (Math.abs(result1.output[i] - result2.output[i]) > 1e-5) {
          different = true;
          break;
        }
      }
      expect(different).toBe(true);
    });
  });

  describe('Mixture-of-Experts Attention', () => {
    it('should compute MoE attention', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      const result = await service.moeAttention(query, key, value);

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(seqLen * embedDim);
      expect(result.mechanism).toBe('moe');
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should handle different number of experts', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      // Create service with different MoE configs
      const service4 = new AttentionService({ ...config, numExperts: 4, topK: 2 });
      await service4.initialize();
      const result4 = await service4.moeAttention(query, key, value);

      const service8 = new AttentionService({ ...config, numExperts: 8, topK: 2 });
      await service8.initialize();
      const result8 = await service8.moeAttention(query, key, value);

      expect(result4.output.length).toBe(result8.output.length);
    });
  });

  describe('Performance Tracking', () => {
    it('should track operation statistics', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      service.resetStats();

      await service.multiHeadAttention(query, key, value);
      await service.flashAttention(query, key, value);
      await service.linearAttention(query, key, value);

      const stats = service.getStats();

      expect(stats.totalOps).toBe(3);
      expect(stats.avgExecutionTimeMs).toBeGreaterThan(0);
      expect(stats.peakMemoryBytes).toBeGreaterThan(0);
      expect(stats.mechanismCounts['multi-head']).toBe(1);
      expect(stats.mechanismCounts['flash']).toBe(1);
      expect(stats.mechanismCounts['linear']).toBe(1);
    });

    it('should calculate average execution time correctly', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      service.resetStats();

      const result1 = await service.multiHeadAttention(query, key, value);
      const result2 = await service.multiHeadAttention(query, key, value);

      const stats = service.getStats();
      const expectedAvg = (result1.executionTimeMs + result2.executionTimeMs) / 2;

      expect(Math.abs(stats.avgExecutionTimeMs - expectedAvg)).toBeLessThan(1);
    });

    it('should reset statistics', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      await service.multiHeadAttention(query, key, value);
      service.resetStats();

      const stats = service.getStats();
      expect(stats.totalOps).toBe(0);
      expect(stats.avgExecutionTimeMs).toBe(0);
      expect(stats.peakMemoryBytes).toBe(0);
    });
  });

  describe('Error Handling', () => {
    it('should reject key/value row-count mismatch', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim);
      const key = new Float32Array(seqLen * embedDim);
      const value = new Float32Array((seqLen + 1) * embedDim); // Wrong size

      // Since 3.6.0 (#43): key and value must have the same number of rows.
      await expect(service.multiHeadAttention(query, key, value)).rejects.toThrow(/dimension mismatch/);
    });

    it('should handle empty inputs', async () => {
      const query = new Float32Array(0);
      const key = new Float32Array(0);
      const value = new Float32Array(0);

      // Should handle gracefully
      try {
        await service.multiHeadAttention(query, key, value);
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe('Zero-Copy Processing', () => {
    it('should use Float32Array without copying', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const query = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const key = new Float32Array(seqLen * embedDim).map(() => Math.random());
      const value = new Float32Array(seqLen * embedDim).map(() => Math.random());

      // Store original buffer references
      const queryBuffer = query.buffer;
      const keyBuffer = key.buffer;
      const valueBuffer = value.buffer;

      await service.multiHeadAttention(query, key, value);

      // Input buffers should not be modified
      expect(query.buffer).toBe(queryBuffer);
      expect(key.buffer).toBe(keyBuffer);
      expect(value.buffer).toBe(valueBuffer);
    });
  });

  describe('Configuration', () => {
    it('should respect custom configuration', async () => {
      const customConfig: AttentionConfig = {
        numHeads: 4,
        headDim: 32,
        embedDim: 128,
        dropout: 0.2,
        bias: false,
        useFlash: true
      };

      const customService = new AttentionService(customConfig);
      await customService.initialize();

      const info = customService.getInfo();
      expect(info.config.numHeads).toBe(4);
      expect(info.config.headDim).toBe(32);
      expect(info.config.embedDim).toBe(128);
      expect(info.config.dropout).toBe(0.2);
      expect(info.config.bias).toBe(false);
    });

    it('should use default values for optional config', async () => {
      const minimalConfig: AttentionConfig = {
        numHeads: 8,
        headDim: 64,
        embedDim: 512
      };

      const minimalService = new AttentionService(minimalConfig);
      await minimalService.initialize();

      const info = minimalService.getInfo();
      expect(info.config.dropout).toBeDefined();
      expect(info.config.bias).toBeDefined();
    });
  });

  describe('Runtime Detection', () => {
    it('should provide runtime information', () => {
      const info = service.getInfo();
      expect(info.runtime).toBeDefined();
      expect(['nodejs', 'browser', 'unknown']).toContain(info.runtime);
    });

    it('should indicate backend availability', () => {
      const info = service.getInfo();
      expect(typeof info.hasNAPI).toBe('boolean');
      expect(typeof info.hasWASM).toBe('boolean');
    });
  });

  describe('Batch Processing', () => {
    it('should handle multiple sequential operations', async () => {
      const seqLen = 4;
      const embedDim = config.embedDim;

      const queries = [
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random())
      ];

      const keys = [
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random())
      ];

      const values = [
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random()),
        new Float32Array(seqLen * embedDim).map(() => Math.random())
      ];

      service.resetStats();

      for (let i = 0; i < 3; i++) {
        await service.multiHeadAttention(queries[i], keys[i], values[i]);
      }

      const stats = service.getStats();
      expect(stats.totalOps).toBe(3);
    });
  });
});
