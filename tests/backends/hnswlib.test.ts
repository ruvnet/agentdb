/**
 * HNSW Backend Tests
 *
 * Tests hnswlib-specific functionality including index building,
 * search quality, persistence, and HNSW algorithm parameters.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { HNSWIndex } from '../../src/controllers/HNSWIndex.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

// Mock database interface
interface MockDatabase {
  prepare: (sql: string) => {
    all: (...params: any[]) => any[];
    get: (...params: any[]) => any;
  };
}

function generateVector(dimension: number): Float32Array {
  const vec = new Float32Array(dimension);
  for (let i = 0; i < dimension; i++) {
    vec[i] = Math.random() * 2 - 1;
  }
  return normalizeVector(vec);
}

function normalizeVector(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

interface TestVector {
  id: string;
  embedding: Float32Array;
}

function generateTestVectors(count: number, dimension: number): TestVector[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `vec-${i}`,
    embedding: generateVector(dimension),
  }));
}

function createMockDatabase(vectors: TestVector[]): MockDatabase {
  const vectorMap = new Map(
    vectors.map((v, idx) => [
      idx + 1,
      {
        id: idx + 1,
        pattern_id: idx + 1,
        embedding: Buffer.from(v.embedding.buffer),
      },
    ])
  );

  return {
    prepare: (sql: string) => ({
      all: (...params: any[]) => {
        if (sql.includes('SELECT pattern_id')) {
          return Array.from(vectorMap.values());
        }
        return [];
      },
      get: (id: number, ...params: any[]) => {
        const result = vectorMap.get(id);
        if (result && params.length > 0) {
          // Simple filter matching for testing
          return result;
        }
        return result;
      },
    }),
  };
}

describe('HNSW Backend Tests', () => {
  const DIMENSION = 384;
  const TEST_DIR = path.join(os.tmpdir(), 'agentdb-hnsw-tests');

  let testVectors: TestVector[];
  let mockDb: MockDatabase;

  beforeAll(() => {
    // Create test directory
    if (!fs.existsSync(TEST_DIR)) {
      fs.mkdirSync(TEST_DIR, { recursive: true });
    }

    testVectors = generateTestVectors(1000, DIMENSION);
    mockDb = createMockDatabase(testVectors) as any;
  });

  afterAll(() => {
    // Cleanup test directory
    if (fs.existsSync(TEST_DIR)) {
      fs.rmSync(TEST_DIR, { recursive: true, force: true });
    }
  });

  describe('Initialization', () => {
    it('should initialize with default config', () => {
      const index = new HNSWIndex(mockDb as any);
      const stats = index.getStats();

      expect(stats.M).toBe(16);
      expect(stats.efConstruction).toBe(200);
      expect(stats.efSearch).toBe(100);
      expect(stats.metric).toBe('cosine');
      expect(stats.dimension).toBe(1536); // Default
    });

    it('should initialize with custom config', () => {
      const index = new HNSWIndex(mockDb as any, {
        M: 32,
        efConstruction: 400,
        efSearch: 200,
        metric: 'l2',
        dimension: DIMENSION,
        maxElements: 50000,
        persistIndex: false,
        rebuildThreshold: 0.2,
      });

      const stats = index.getStats();

      expect(stats.M).toBe(32);
      expect(stats.efConstruction).toBe(400);
      expect(stats.efSearch).toBe(200);
      expect(stats.metric).toBe('l2');
      expect(stats.dimension).toBe(DIMENSION);
    });

    it('should not build index on initialization', () => {
      const index = new HNSWIndex(mockDb as any, { dimension: DIMENSION });
      const stats = index.getStats();

      expect(stats.indexBuilt).toBe(false);
      expect(stats.numElements).toBe(0);
    });
  });

  describe('Index Building', () => {
    it('should build index from database', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      const stats = index.getStats();
      expect(stats.indexBuilt).toBe(true);
      expect(stats.numElements).toBe(testVectors.length);
      expect(stats.lastBuildTime).toBeGreaterThan(0);
    });

    it('should handle empty database gracefully', async () => {
      const emptyDb = createMockDatabase([]) as any;
      const index = new HNSWIndex(emptyDb, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      const stats = index.getStats();
      expect(stats.indexBuilt).toBe(false);
      expect(stats.numElements).toBe(0);
    });

    it('should track build time', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      const startTime = Date.now();
      await index.buildIndex('pattern_embeddings');
      const endTime = Date.now();

      const stats = index.getStats();
      expect(stats.lastBuildTime).toBeGreaterThanOrEqual(startTime);
      expect(stats.lastBuildTime).toBeLessThanOrEqual(endTime);
    });

    it('should rebuild index when called multiple times', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');
      const firstBuildTime = index.getStats().lastBuildTime;

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 10));

      await index.buildIndex('pattern_embeddings');
      const secondBuildTime = index.getStats().lastBuildTime;

      expect(secondBuildTime).toBeGreaterThan(firstBuildTime!);
    });
  });

  describe('Search Operations', () => {
    let index: HNSWIndex;

    beforeEach(async () => {
      index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });
      await index.buildIndex('pattern_embeddings');
    });

    it('should search and return k results', async () => {
      const query = generateVector(DIMENSION);
      const k = 10;

      const results = await index.search(query, k);

      expect(results.length).toBeLessThanOrEqual(k);
    });

    it('should return results sorted by similarity', async () => {
      const query = generateVector(DIMENSION);

      const results = await index.search(query, 20);

      // Check descending similarity order
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(results[i].similarity);
      }
    });

    it('should find exact match with high similarity', async () => {
      const exactVector = testVectors[0].embedding;

      const results = await index.search(exactVector, 1);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].similarity).toBeGreaterThan(0.99); // Should be very close to 1.0
    });

    it('should respect threshold parameter', async () => {
      const query = generateVector(DIMENSION);
      const threshold = 0.7;

      const results = await index.search(query, 50, { threshold });

      for (const result of results) {
        expect(result.similarity).toBeGreaterThanOrEqual(threshold);
      }
    });

    it('should throw error when index not built', async () => {
      const newIndex = new HNSWIndex(mockDb as any, { dimension: DIMENSION });
      const query = generateVector(DIMENSION);

      await expect(async () => {
        await newIndex.search(query, 10);
      }).rejects.toThrow('Index not built');
    });

    it('should include distance in results', async () => {
      const query = generateVector(DIMENSION);

      const results = await index.search(query, 10);

      for (const result of results) {
        expect(result).toHaveProperty('distance');
        expect(result).toHaveProperty('similarity');
        expect(typeof result.distance).toBe('number');
        expect(typeof result.similarity).toBe('number');
      }
    });

    it('should track search statistics', async () => {
      const query = generateVector(DIMENSION);

      await index.search(query, 10);
      await index.search(query, 10);
      await index.search(query, 10);

      const stats = index.getStats();
      expect(stats.totalSearches).toBe(3);
      expect(stats.avgSearchTimeMs).toBeGreaterThan(0);
      expect(stats.lastSearchTime).toBeGreaterThan(0);
    });
  });

  describe('Dynamic Updates', () => {
    let index: HNSWIndex;

    beforeEach(async () => {
      index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
        rebuildThreshold: 0.1, // 10% update threshold
      });
      await index.buildIndex('pattern_embeddings');
    });

    it('should add new vectors to index', () => {
      const initialCount = index.getStats().numElements;
      const newVector = generateVector(DIMENSION);
      const newId = testVectors.length + 1;

      index.addVector(newId, newVector);

      const finalCount = index.getStats().numElements;
      expect(finalCount).toBe(initialCount + 1);
    });

    it('should mark vectors for removal', () => {
      const initialCount = index.getStats().numElements;
      const removeId = 1;

      index.removeVector(removeId);

      const finalCount = index.getStats().numElements;
      expect(finalCount).toBe(initialCount - 1);
    });

    it('should detect when rebuild is needed', () => {
      const initialCount = index.getStats().numElements;
      const updateCount = Math.ceil(initialCount * 0.15); // 15% updates

      expect(index.needsRebuild()).toBe(false);

      // Add enough vectors to trigger rebuild threshold
      for (let i = 0; i < updateCount; i++) {
        index.addVector(testVectors.length + i + 1, generateVector(DIMENSION));
      }

      expect(index.needsRebuild()).toBe(true);
    });

    it('should handle duplicate IDs', () => {
      const duplicateId = 999;
      const vec1 = generateVector(DIMENSION);
      const vec2 = generateVector(DIMENSION);

      index.addVector(duplicateId, vec1);
      const countAfterFirst = index.getStats().numElements;

      index.addVector(duplicateId, vec2);
      const countAfterSecond = index.getStats().numElements;

      expect(countAfterSecond).toBeGreaterThan(countAfterFirst);
    });
  });

  describe('Distance Metrics', () => {
    it('should use cosine metric correctly', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: 3,
        metric: 'cosine',
        persistIndex: false,
      });

      // Create simple test database
      const simpleVectors = [
        { id: 'x-axis', embedding: new Float32Array([1, 0, 0]) },
        { id: 'y-axis', embedding: new Float32Array([0, 1, 0]) },
        { id: 'diagonal', embedding: new Float32Array([0.707, 0.707, 0]) },
      ];

      const simpleDb = createMockDatabase(simpleVectors) as any;
      const simpleIndex = new HNSWIndex(simpleDb, {
        dimension: 3,
        metric: 'cosine',
        persistIndex: false,
      });

      await simpleIndex.buildIndex('pattern_embeddings');

      const query = new Float32Array([1, 0, 0]);
      const results = await simpleIndex.search(query, 3);

      // x-axis should be most similar to itself
      expect(results[0].id).toBe(1); // x-axis vector
      expect(results[0].similarity).toBeCloseTo(1.0, 2);
    });

    it('should use L2 metric correctly', async () => {
      const simpleVectors = [
        { id: 'origin', embedding: new Float32Array([0, 0, 0]) },
        { id: 'close', embedding: new Float32Array([0.1, 0.1, 0.1]) },
        { id: 'far', embedding: new Float32Array([1, 1, 1]) },
      ];

      const simpleDb = createMockDatabase(simpleVectors) as any;
      const index = new HNSWIndex(simpleDb, {
        dimension: 3,
        metric: 'l2',
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      const query = new Float32Array([0, 0, 0]);
      const results = await index.search(query, 3);

      // Closest vector should be origin itself
      expect(results[0].distance).toBeCloseTo(0, 2);
    });

    it('should use inner product metric correctly', async () => {
      const simpleVectors = [
        { id: 'high', embedding: new Float32Array([1, 1, 1]) },
        { id: 'medium', embedding: new Float32Array([0.5, 0.5, 0.5]) },
        { id: 'low', embedding: new Float32Array([0.1, 0.1, 0.1]) },
      ];

      const simpleDb = createMockDatabase(simpleVectors) as any;
      const index = new HNSWIndex(simpleDb, {
        dimension: 3,
        metric: 'ip',
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      const query = new Float32Array([1, 1, 1]);
      const results = await index.search(query, 3);

      // Higher inner product should rank first
      expect(results[0].id).toBe(1); // 'high' vector
    });
  });

  describe('Performance Tuning', () => {
    it('should update efSearch parameter', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        efSearch: 100,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      index.setEfSearch(200);

      const stats = index.getStats();
      expect(stats.efSearch).toBe(200);
    });

    it('should maintain search quality with higher efSearch', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');

      const query = generateVector(DIMENSION);

      // Search with default efSearch
      const results1 = await index.search(query, 10);

      // Increase efSearch for better quality
      index.setEfSearch(300);
      const results2 = await index.search(query, 10);

      // Results should be similar or better quality
      expect(results2.length).toBeGreaterThanOrEqual(results1.length);
    });
  });

  describe('Persistence', () => {
    it('should save and load index', async () => {
      const indexPath = path.join(TEST_DIR, 'test-index.hnsw');

      const index1 = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: true,
        indexPath,
      });

      await index1.buildIndex('pattern_embeddings');
      const stats1 = index1.getStats();

      // Create new instance and load
      const index2 = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: true,
        indexPath,
      });
      // Persisted indexes load lazily; initialize() performs the load.
      await index2.initialize();

      const stats2 = index2.getStats();

      expect(stats2.indexBuilt).toBe(true);
      expect(stats2.numElements).toBe(stats1.numElements);
    });

    it('should create index directory if not exists', async () => {
      const nestedPath = path.join(TEST_DIR, 'nested', 'path', 'index.hnsw');

      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: true,
        indexPath: nestedPath,
      });

      await index.buildIndex('pattern_embeddings');

      expect(fs.existsSync(path.dirname(nestedPath))).toBe(true);
    });

    it('should save mappings with index', async () => {
      const indexPath = path.join(TEST_DIR, 'mappings-test.hnsw');

      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: true,
        indexPath,
      });

      await index.buildIndex('pattern_embeddings');

      const mappingsPath = indexPath + '.mappings.json';
      expect(fs.existsSync(mappingsPath)).toBe(true);

      const mappingsData = JSON.parse(fs.readFileSync(mappingsPath, 'utf-8'));
      expect(mappingsData).toHaveProperty('idToLabel');
      expect(mappingsData).toHaveProperty('labelToId');
      expect(mappingsData).toHaveProperty('nextLabel');
      expect(mappingsData).toHaveProperty('config');
    });
  });

  describe('Error Handling', () => {
    it('should handle corrupted index files gracefully', async () => {
      const indexPath = path.join(TEST_DIR, 'corrupted.hnsw');

      // Create corrupted file
      fs.writeFileSync(indexPath, 'corrupted data');

      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: true,
        indexPath,
      });

      // Should not throw, but index won't be built
      expect(index.isReady()).toBe(false);
    });

    it('should clear index resources', () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      index.clear();

      const stats = index.getStats();
      expect(stats.indexBuilt).toBe(false);
      expect(stats.numElements).toBe(0);
    });
  });

  describe('Statistics and Monitoring', () => {
    it('should report comprehensive statistics', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      await index.buildIndex('pattern_embeddings');
      await index.search(generateVector(DIMENSION), 10);

      const stats = index.getStats();

      expect(stats).toHaveProperty('enabled');
      expect(stats).toHaveProperty('indexBuilt');
      expect(stats).toHaveProperty('numElements');
      expect(stats).toHaveProperty('dimension');
      expect(stats).toHaveProperty('metric');
      expect(stats).toHaveProperty('M');
      expect(stats).toHaveProperty('efConstruction');
      expect(stats).toHaveProperty('efSearch');
      expect(stats).toHaveProperty('lastBuildTime');
      expect(stats).toHaveProperty('lastSearchTime');
      expect(stats).toHaveProperty('totalSearches');
      expect(stats).toHaveProperty('avgSearchTimeMs');
    });

    it('should check readiness status', async () => {
      const index = new HNSWIndex(mockDb as any, {
        dimension: DIMENSION,
        persistIndex: false,
      });

      expect(index.isReady()).toBe(false);

      await index.buildIndex('pattern_embeddings');

      expect(index.isReady()).toBe(true);
    });
  });
});
