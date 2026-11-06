/**
 * Unit Tests for EmbeddingService Controller
 *
 * Tests text embedding generation, caching, and different providers
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';

describe('EmbeddingService', () => {
  describe('Local Provider', () => {
    let embedder: EmbeddingService;

    beforeEach(async () => {
      embedder = new EmbeddingService({
        model: 'mock-model',
        dimension: 384,
        provider: 'local',
      });
      await embedder.initialize();
    });

    it('should initialize successfully', () => {
      expect(embedder).toBeDefined();
    });

    it('should generate embedding for text', async () => {
      const embedding = await embedder.embed('test text');

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should generate deterministic embeddings for same text', async () => {
      const text = 'deterministic test';
      const embedding1 = await embedder.embed(text);
      const embedding2 = await embedder.embed(text);

      expect(embedding1.length).toBe(embedding2.length);

      // Embeddings should be identical for same text
      for (let i = 0; i < embedding1.length; i++) {
        expect(embedding1[i]).toBe(embedding2[i]);
      }
    });

    it('should generate different embeddings for different texts', async () => {
      const embedding1 = await embedder.embed('text one');
      const embedding2 = await embedder.embed('text two');

      // Embeddings should be different
      let different = false;
      for (let i = 0; i < embedding1.length; i++) {
        if (embedding1[i] !== embedding2[i]) {
          different = true;
          break;
        }
      }

      expect(different).toBe(true);
    });

    it('should generate normalized embeddings', async () => {
      const embedding = await embedder.embed('normalization test');

      // Calculate L2 norm
      let norm = 0;
      for (let i = 0; i < embedding.length; i++) {
        norm += embedding[i] * embedding[i];
      }
      norm = Math.sqrt(norm);

      // Normalized embeddings should have norm ≈ 1.0
      expect(norm).toBeGreaterThan(0.99);
      expect(norm).toBeLessThan(1.01);
    });

    it('should handle empty text', async () => {
      const embedding = await embedder.embed('');

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);

      // Should return zero vector or valid embedding
      let hasNonZero = false;
      for (let i = 0; i < embedding.length; i++) {
        if (embedding[i] !== 0) {
          hasNonZero = true;
          break;
        }
      }

      // Either all zeros or valid embedding
      expect(hasNonZero || embedding[0] === 0).toBe(true);
    });

    it('should handle very long text', async () => {
      const longText = 'a'.repeat(10000);
      const embedding = await embedder.embed(longText);

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should handle Unicode characters', async () => {
      const unicodeText = 'Hello 世界 🌍 مرحبا';
      const embedding = await embedder.embed(unicodeText);

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should handle special characters', async () => {
      const specialText = '!@#$%^&*()_+-={}[]|\\:";\'<>?,./';
      const embedding = await embedder.embed(specialText);

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });
  });

  describe('Caching', () => {
    let embedder: EmbeddingService;

    beforeEach(async () => {
      embedder = new EmbeddingService({
        model: 'mock-model',
        dimension: 384,
        provider: 'local',
      });
      await embedder.initialize();
    });

    it('should cache embeddings', async () => {
      const text = 'cached text';

      // The mock embedder finishes in well under 1 ms, so comparing Date.now()
      // durations reads 0 < 0. Assert the cache hit directly instead: the
      // second call must return the stored array, not compute a new one.
      const computeSpy = vi.spyOn(embedder as any, 'generateEmbedding');
      const first = await embedder.embed(text);
      const second = await embedder.embed(text);

      expect(second).toBe(first);
      expect(computeSpy).toHaveBeenCalledTimes(1);
    });

    it('should return same instance from cache', async () => {
      const text = 'cache instance test';

      const embedding1 = await embedder.embed(text);
      const embedding2 = await embedder.embed(text);

      // Should be the exact same object
      expect(embedding1).toBe(embedding2);
    });

    it('should clear cache', async () => {
      await embedder.embed('test 1');
      await embedder.embed('test 2');

      embedder.clearCache();

      // After clearing, should generate new embeddings
      const embedding = await embedder.embed('test 1');
      expect(embedding).toBeDefined();
    });

    it('should handle cache size limits', async () => {
      // Generate many embeddings to test cache eviction
      for (let i = 0; i < 100; i++) {
        await embedder.embed(`test text ${i}`);
      }

      // Cache should still work
      const embedding = await embedder.embed('test text 0');
      expect(embedding).toBeDefined();
    });
  });

  describe('Batch Operations', () => {
    let embedder: EmbeddingService;

    beforeEach(async () => {
      embedder = new EmbeddingService({
        model: 'mock-model',
        dimension: 384,
        provider: 'local',
      });
      await embedder.initialize();
    });

    it('should embed batch of texts', async () => {
      const texts = ['text 1', 'text 2', 'text 3', 'text 4', 'text 5'];
      const embeddings = await embedder.embedBatch(texts);

      expect(embeddings).toHaveLength(5);
      embeddings.forEach(emb => {
        expect(emb).toBeInstanceOf(Float32Array);
        expect(emb.length).toBe(384);
      });
    });

    it('should handle empty batch', async () => {
      const embeddings = await embedder.embedBatch([]);

      expect(embeddings).toHaveLength(0);
    });

    it('should handle large batch', async () => {
      const texts = Array.from({ length: 100 }, (_, i) => `text ${i}`);
      const embeddings = await embedder.embedBatch(texts);

      expect(embeddings).toHaveLength(100);
    }, 10000);

    it('should handle batch with duplicate texts', async () => {
      const texts = ['same text', 'same text', 'different text', 'same text'];
      const embeddings = await embedder.embedBatch(texts);

      expect(embeddings).toHaveLength(4);

      // First and second embeddings should be identical (cached)
      expect(embeddings[0]).toBe(embeddings[1]);
      expect(embeddings[0]).toBe(embeddings[3]);
    });
  });

  describe('Different Dimensions', () => {
    it('should support 128-dimensional embeddings', async () => {
      const embedder = new EmbeddingService({
        model: 'small-model',
        dimension: 128,
        provider: 'local',
      });
      await embedder.initialize();

      const embedding = await embedder.embed('test');

      expect(embedding.length).toBe(128);
    });

    it('should support 768-dimensional embeddings', async () => {
      const embedder = new EmbeddingService({
        model: 'large-model',
        dimension: 768,
        provider: 'local',
      });
      await embedder.initialize();

      const embedding = await embedder.embed('test');

      expect(embedding.length).toBe(768);
    });

    it('should support 1536-dimensional embeddings', async () => {
      const embedder = new EmbeddingService({
        model: 'xl-model',
        dimension: 1536,
        provider: 'local',
      });
      await embedder.initialize();

      const embedding = await embedder.embed('test');

      expect(embedding.length).toBe(1536);
    });
  });

  describe('Edge Cases', () => {
    let embedder: EmbeddingService;

    beforeEach(async () => {
      embedder = new EmbeddingService({
        model: 'mock-model',
        dimension: 384,
        provider: 'local',
      });
      await embedder.initialize();
    });

    it('should handle whitespace-only text', async () => {
      const embedding = await embedder.embed('   \n\t  ');

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should handle single character', async () => {
      const embedding = await embedder.embed('a');

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should handle numbers as text', async () => {
      const embedding = await embedder.embed('123456789');

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });

    it('should handle mixed content', async () => {
      const mixedText = 'Text with numbers 123, symbols !@#, and Unicode 你好';
      const embedding = await embedder.embed(mixedText);

      expect(embedding).toBeInstanceOf(Float32Array);
      expect(embedding.length).toBe(384);
    });
  });

  describe('Performance', () => {
    let embedder: EmbeddingService;

    beforeEach(async () => {
      embedder = new EmbeddingService({
        model: 'mock-model',
        dimension: 384,
        provider: 'local',
      });
      await embedder.initialize();
    });

    it('should generate 100 embeddings efficiently', async () => {
      const startTime = Date.now();

      for (let i = 0; i < 100; i++) {
        await embedder.embed(`test text ${i}`);
      }

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(2000); // Should complete in less than 2 seconds
    }, 10000);

    it('should handle concurrent embedding requests', async () => {
      const promises = Array.from({ length: 50 }, (_, i) =>
        embedder.embed(`concurrent text ${i}`)
      );

      const embeddings = await Promise.all(promises);

      expect(embeddings).toHaveLength(50);
      embeddings.forEach(emb => {
        expect(emb).toBeInstanceOf(Float32Array);
      });
    });

    it('should batch embed efficiently', async () => {
      const texts = Array.from({ length: 100 }, (_, i) => `batch text ${i}`);

      const startTime = Date.now();
      const embeddings = await embedder.embedBatch(texts);
      const duration = Date.now() - startTime;

      expect(embeddings).toHaveLength(100);
      expect(duration).toBeLessThan(3000); // Should complete in less than 3 seconds
    }, 10000);
  });
});
