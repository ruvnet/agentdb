/**
 * HNSWLibBackend Tests
 *
 * Comprehensive test suite for HNSWLib backend wrapper
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { HNSWLibBackend } from '../../src/backends/hnswlib/HNSWLibBackend.js';
import type { VectorConfig } from '../../src/backends/VectorBackend.js';
import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as os from 'os';
import * as path from 'path';

describe('HNSWLibBackend', () => {
  let backend: HNSWLibBackend;
  let tempDir: string;

  const config: VectorConfig = {
    dimension: 384,
    metric: 'cosine',
    maxElements: 1000,
    M: 16,
    efConstruction: 200,
    efSearch: 100,
  };

  beforeEach(async () => {
    backend = new HNSWLibBackend(config);
    await backend.initialize();

    // Create temp directory for save/load tests
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'hnswlib-test-'));
  });

  afterEach(async () => {
    backend.close();

    // Cleanup temp directory
    if (fsSync.existsSync(tempDir)) {
      await fs.rm(tempDir, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    it('should initialize with correct config', () => {
      const stats = backend.getStats();
      expect(stats.dimension).toBe(384);
      expect(stats.metric).toBe('cosine');
      expect(stats.backend).toBe('hnswlib');
      expect(stats.count).toBe(0);
    });

    it('should be ready after initialization', () => {
      expect(backend.isReady()).toBe(true);
    });

    it('should have correct backend name', () => {
      expect(backend.name).toBe('hnswlib');
    });
  });

  describe('Insert Operations', () => {
    it('should insert a single vector', () => {
      const embedding = new Float32Array(384).fill(0.1);
      backend.insert('test-1', embedding);

      const stats = backend.getStats();
      expect(stats.count).toBe(1);
    });

    it('should insert vector with metadata', () => {
      const embedding = new Float32Array(384).fill(0.1);
      const metadata = { type: 'test', category: 'example' };

      backend.insert('test-1', embedding, metadata);

      const results = backend.search(embedding, 1);
      expect(results[0].metadata).toEqual(metadata);
    });

    it('should reject duplicate IDs', () => {
      const embedding = new Float32Array(384).fill(0.1);
      backend.insert('test-1', embedding);

      expect(() => {
        backend.insert('test-1', embedding);
      }).toThrow(/already exists/);
    });

    it('should insert batch of vectors', () => {
      const items = Array.from({ length: 10 }, (_, i) => ({
        id: `vec-${i}`,
        embedding: new Float32Array(384).fill(i * 0.1),
        metadata: { index: i },
      }));

      backend.insertBatch(items);

      const stats = backend.getStats();
      expect(stats.count).toBe(10);
    });
  });

  describe('Search Operations', () => {
    beforeEach(() => {
      // Insert test vectors
      for (let i = 0; i < 100; i++) {
        const embedding = new Float32Array(384);
        for (let j = 0; j < 384; j++) {
          embedding[j] = Math.random();
        }
        backend.insert(`vec-${i}`, embedding, { index: i });
      }
    });

    it('should search and return k results', () => {
      const query = new Float32Array(384).fill(0.5);
      const results = backend.search(query, 10);

      expect(results.length).toBeLessThanOrEqual(10);
      expect(results.length).toBeGreaterThan(0);
    });

    it('should return results sorted by similarity', () => {
      const query = new Float32Array(384).fill(0.5);
      const results = backend.search(query, 10);

      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(results[i].similarity);
      }
    });

    it('should return identical vector with high similarity', () => {
      const embedding = new Float32Array(384).fill(0.7);
      backend.insert('identical', embedding);

      const results = backend.search(embedding, 1);
      expect(results[0].id).toBe('identical');
      expect(results[0].similarity).toBeGreaterThan(0.99);
    });

    it('should apply similarity threshold', () => {
      const query = new Float32Array(384).fill(0.5);
      const results = backend.search(query, 10, { threshold: 0.8 });

      results.forEach((result) => {
        expect(result.similarity).toBeGreaterThanOrEqual(0.8);
      });
    });

    it('should allow efSearch override', () => {
      const query = new Float32Array(384).fill(0.5);

      // Search with default efSearch
      const results1 = backend.search(query, 10);

      // Search with higher efSearch (better quality)
      const results2 = backend.search(query, 10, { efSearch: 200 });

      expect(results1.length).toBeGreaterThan(0);
      expect(results2.length).toBeGreaterThan(0);
    });

    it('should include metadata in results', () => {
      const query = new Float32Array(384).fill(0.5);
      const results = backend.search(query, 5);

      results.forEach((result) => {
        expect(result.metadata).toBeDefined();
        expect(result.metadata!.index).toBeGreaterThanOrEqual(0);
      });
    });

    it('should apply metadata filters', () => {
      // Insert vectors with specific metadata
      const embedding1 = new Float32Array(384).fill(0.1);
      const embedding2 = new Float32Array(384).fill(0.2);

      backend.insert('cat-1', embedding1, { category: 'A', type: 'test' });
      backend.insert('cat-2', embedding2, { category: 'B', type: 'test' });

      const query = new Float32Array(384).fill(0.15);
      const results = backend.search(query, 10, {
        filter: { category: 'A' },
      });

      results.forEach((result) => {
        expect(result.metadata!.category).toBe('A');
      });
    });
  });

  describe('Remove Operations', () => {
    beforeEach(() => {
      const embedding = new Float32Array(384).fill(0.1);
      backend.insert('test-1', embedding, { data: 'value' });
    });

    it('should remove vector by ID', () => {
      const removed = backend.remove('test-1');
      expect(removed).toBe(true);

      const stats = backend.getStats();
      expect(stats.count).toBe(0);
    });

    it('should return false for non-existent ID', () => {
      const removed = backend.remove('non-existent');
      expect(removed).toBe(false);
    });

    it('should not return removed vectors in search', () => {
      const embedding = new Float32Array(384).fill(0.1);
      backend.insert('test-2', embedding);

      backend.remove('test-1');

      const results = backend.search(embedding, 10);
      const ids = results.map((r) => r.id);

      expect(ids).not.toContain('test-1');
    });

    it('should allow reinserting removed ID', () => {
      backend.remove('test-1');

      const newEmbedding = new Float32Array(384).fill(0.2);
      backend.insert('test-1', newEmbedding, { data: 'new-value' });

      const results = backend.search(newEmbedding, 1);
      expect(results[0].id).toBe('test-1');
      expect(results[0].metadata!.data).toBe('new-value');
    });
  });

  describe('Save and Load', () => {
    const indexPath = () => path.join(tempDir, 'test-index.bin');

    beforeEach(() => {
      // Insert test data
      for (let i = 0; i < 50; i++) {
        const embedding = new Float32Array(384).fill(i * 0.01);
        backend.insert(`vec-${i}`, embedding, { index: i, category: 'test' });
      }
    });

    it('should save index to disk', async () => {
      await backend.save(indexPath());

      expect(fsSync.existsSync(indexPath())).toBe(true);
      expect(fsSync.existsSync(indexPath() + '.mappings.json')).toBe(true);
    });

    it('should load index from disk', async () => {
      await backend.save(indexPath());

      // Create new backend and load
      const newBackend = new HNSWLibBackend(config);
      await newBackend.initialize();
      await newBackend.load(indexPath());

      const stats = newBackend.getStats();
      expect(stats.count).toBe(50);

      // Verify search works
      const query = new Float32Array(384).fill(0.25);
      const results = newBackend.search(query, 5);
      expect(results.length).toBeGreaterThan(0);

      newBackend.close();
    });

    it('should preserve metadata after save/load', async () => {
      await backend.save(indexPath());

      const newBackend = new HNSWLibBackend(config);
      await newBackend.initialize();
      await newBackend.load(indexPath());

      const query = new Float32Array(384).fill(0.25);
      const results = newBackend.search(query, 5);

      results.forEach((result) => {
        expect(result.metadata).toBeDefined();
        expect(result.metadata!.category).toBe('test');
      });

      newBackend.close();
    });

    it('should throw error loading non-existent file', async () => {
      const newBackend = new HNSWLibBackend(config);
      await newBackend.initialize();

      await expect(
        newBackend.load('/non/existent/path.bin')
      ).rejects.toThrow(/not found/);

      newBackend.close();
    });
  });

  describe('Statistics', () => {
    it('should return accurate count', () => {
      for (let i = 0; i < 25; i++) {
        const embedding = new Float32Array(384).fill(i * 0.1);
        backend.insert(`vec-${i}`, embedding);
      }

      const stats = backend.getStats();
      expect(stats.count).toBe(25);
    });

    it('should update count after removals', () => {
      for (let i = 0; i < 10; i++) {
        const embedding = new Float32Array(384).fill(i * 0.1);
        backend.insert(`vec-${i}`, embedding);
      }

      backend.remove('vec-0');
      backend.remove('vec-1');

      const stats = backend.getStats();
      expect(stats.count).toBe(8);
    });
  });

  describe('Similarity Conversion', () => {
    it('should convert cosine distance to similarity', () => {
      const embedding = new Float32Array(384).fill(0.5);
      backend.insert('test', embedding);

      const results = backend.search(embedding, 1);
      expect(results[0].similarity).toBeCloseTo(1.0, 1); // Identical vector
    });

    it('should handle L2 metric', async () => {
      const l2Backend = new HNSWLibBackend({ ...config, metric: 'l2' });
      await l2Backend.initialize();

      const embedding = new Float32Array(384).fill(0.5);
      l2Backend.insert('test', embedding);

      const results = l2Backend.search(embedding, 1);
      expect(results[0].similarity).toBeGreaterThan(0.5); // Exp decay

      l2Backend.close();
    });
  });

  describe('Rebuild Detection', () => {
    it('should detect when rebuild needed', () => {
      for (let i = 0; i < 100; i++) {
        const embedding = new Float32Array(384).fill(i * 0.1);
        backend.insert(`vec-${i}`, embedding);
      }

      // Remove 15% (above 10% threshold)
      for (let i = 0; i < 15; i++) {
        backend.remove(`vec-${i}`);
      }

      expect(backend.needsRebuild(0.1)).toBe(true);
    });

    it('should not require rebuild for small deletes', () => {
      for (let i = 0; i < 100; i++) {
        const embedding = new Float32Array(384).fill(i * 0.1);
        backend.insert(`vec-${i}`, embedding);
      }

      // Remove 5% (below 10% threshold)
      for (let i = 0; i < 5; i++) {
        backend.remove(`vec-${i}`);
      }

      expect(backend.needsRebuild(0.1)).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should throw error if not initialized before insert', () => {
      const uninitBackend = new HNSWLibBackend(config);
      const embedding = new Float32Array(384).fill(0.1);

      expect(() => {
        uninitBackend.insert('test', embedding);
      }).toThrow(/not initialized/);
    });

    it('should throw error if not initialized before search', () => {
      const uninitBackend = new HNSWLibBackend(config);
      const query = new Float32Array(384).fill(0.1);

      expect(() => {
        uninitBackend.search(query, 10);
      }).toThrow(/not initialized/);
    });
  });

  describe('Performance', () => {
    it('should handle large batch inserts efficiently', () => {
      const items = Array.from({ length: 1000 }, (_, i) => ({
        id: `vec-${i}`,
        embedding: new Float32Array(384).fill(Math.random()),
      }));

      const start = performance.now();
      backend.insertBatch(items);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
      expect(backend.getStats().count).toBe(1000);
    });

    it('should search efficiently on large dataset', async () => {
      // 10K vectors exceed the shared config's maxElements (1000), which hnswlib
      // enforces; give this test a backend sized for the dataset.
      backend.close();
      backend = new HNSWLibBackend({ ...config, maxElements: 10000 });
      await backend.initialize();

      // Insert 10K vectors
      for (let i = 0; i < 10000; i++) {
        const embedding = new Float32Array(384);
        for (let j = 0; j < 384; j++) {
          embedding[j] = Math.random();
        }
        backend.insert(`vec-${i}`, embedding);
      }

      const query = new Float32Array(384).fill(0.5);
      const start = performance.now();
      backend.search(query, 10);
      const duration = performance.now() - start;

      // Should be fast even with 10K vectors
      expect(duration).toBeLessThan(100); // < 100ms
    });
  });
});
