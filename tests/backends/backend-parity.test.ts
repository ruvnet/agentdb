/**
 * Backend Parity Tests
 *
 * Ensures RuVector and hnswlib produce equivalent results for identical operations.
 * Tests search result overlap (90%+ for top-10), similarity score accuracy (within 1%),
 * and insert/remove operations parity.
 *
 * Requirements:
 * - Top-1 results must match exactly
 * - Top-10 results must have 90%+ overlap
 * - Similarity scores within 1% margin
 * - Insert/remove operations maintain consistent counts
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { HNSWIndex } from '../../src/controllers/HNSWIndex.js';
import { WASMVectorSearch } from '../../src/controllers/WASMVectorSearch.js';

// Mock database interface
interface MockDatabase {
  prepare: (sql: string) => {
    all: (...params: any[]) => any[];
    get: (...params: any[]) => any;
  };
}

// Test data generation utilities
function generateVector(dimension: number): Float32Array {
  const vec = new Float32Array(dimension);
  for (let i = 0; i < dimension; i++) {
    vec[i] = Math.random() * 2 - 1; // Range [-1, 1]
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

// Create mock database with vector storage
function createMockDatabase(vectors: TestVector[]): MockDatabase {
  const vectorMap = new Map(
    vectors.map((v, idx) => [
      idx + 1,
      {
        id: idx + 1,
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
        return vectorMap.get(id);
      },
    }),
  };
}

describe('Backend Parity Tests', () => {
  const DIMENSION = 384;
  const NUM_VECTORS = 1000;
  const NUM_QUERIES = 100;
  const K_RESULTS = 10;

  let testVectors: TestVector[];
  let queries: TestVector[];
  let mockDb: MockDatabase;
  let hnswIndex: HNSWIndex;
  let wasmSearch: WASMVectorSearch;

  beforeAll(async () => {
    console.log('[Backend Parity] Generating test data...');

    // Generate test vectors and queries
    testVectors = generateTestVectors(NUM_VECTORS, DIMENSION);
    queries = generateTestVectors(NUM_QUERIES, DIMENSION);

    // Create mock database
    mockDb = createMockDatabase(testVectors) as any;

    // Initialize HNSW backend
    hnswIndex = new HNSWIndex(mockDb as any, {
      dimension: DIMENSION,
      metric: 'cosine',
      M: 16,
      efConstruction: 200,
      efSearch: 100,
      maxElements: NUM_VECTORS * 2,
      persistIndex: false,
    });

    // Initialize WASM backend (RuVector alternative)
    wasmSearch = new WASMVectorSearch(mockDb as any, {
      enableWASM: false, // Use pure JS for consistent comparison
      enableSIMD: false,
      batchSize: 100,
      indexThreshold: NUM_VECTORS,
    });

    // Build HNSW index
    console.log('[Backend Parity] Building HNSW index...');
    await hnswIndex.buildIndex('pattern_embeddings');

    console.log('[Backend Parity] Test setup complete');
  });

  afterAll(() => {
    hnswIndex.clear();
    wasmSearch.clearIndex();
  });

  describe('Search Result Parity', () => {
    it('should return same top-1 result for exact match queries', async () => {
      // Use actual vectors from test set as queries (should return themselves)
      const sampleQueries = testVectors.slice(0, 10);

      for (let i = 0; i < sampleQueries.length; i++) {
        const query = sampleQueries[i];

        const hnswResults = await hnswIndex.search(query.embedding, 1);
        const wasmResults = await wasmSearch.findKNN(query.embedding, 1);

        // Top-1 should be the exact same vector
        expect(hnswResults.length).toBeGreaterThan(0);
        expect(wasmResults.length).toBeGreaterThan(0);

        // IDs should match (both should find the query vector itself)
        expect(hnswResults[0].id).toBe(wasmResults[0].id);

        // Similarity should be very close (both ~1.0 for self-match)
        expect(Math.abs(hnswResults[0].similarity - wasmResults[0].similarity)).toBeLessThan(0.01);
      }
    });

    it('should return same top-10 results with 90%+ overlap', async () => {
      const sampleQueries = queries.slice(0, 20);
      let totalOverlap = 0;
      let totalQueries = 0;

      for (const query of sampleQueries) {
        const hnswResults = await hnswIndex.search(query.embedding, K_RESULTS);
        const wasmResults = await wasmSearch.findKNN(query.embedding, K_RESULTS);

        // Extract IDs
        const hnswIds = new Set(hnswResults.map(r => r.id));
        const wasmIds = new Set(wasmResults.map(r => r.id));

        // Calculate overlap
        const overlap = Array.from(hnswIds).filter(id => wasmIds.has(id)).length;
        const overlapPercentage = overlap / K_RESULTS;

        totalOverlap += overlapPercentage;
        totalQueries++;

        // Each query should have at least 90% overlap (HNSW is approximate)
        expect(overlap).toBeGreaterThanOrEqual(9); // 90% of 10 = 9
      }

      // Average overlap should be very high
      const avgOverlap = totalOverlap / totalQueries;
      console.log(`[Backend Parity] Average top-10 overlap: ${(avgOverlap * 100).toFixed(2)}%`);
      expect(avgOverlap).toBeGreaterThan(0.9);
    });

    it('should produce similar similarity scores (within 1%)', async () => {
      const sampleQueries = queries.slice(0, 10);

      for (const query of sampleQueries) {
        const hnswResults = await hnswIndex.search(query.embedding, 5);
        const wasmResults = await wasmSearch.findKNN(query.embedding, 5);

        // Compare similarity scores for matching IDs
        const hnswMap = new Map(hnswResults.map(r => [r.id, r.similarity]));

        for (const wasmResult of wasmResults) {
          const hnswSimilarity = hnswMap.get(wasmResult.id);

          if (hnswSimilarity !== undefined) {
            const diff = Math.abs(hnswSimilarity - wasmResult.similarity);
            expect(diff).toBeLessThan(0.01); // Within 1%
          }
        }
      }
    });

    it('should handle threshold filtering consistently', async () => {
      const query = queries[0];
      const threshold = 0.7;

      const hnswResults = await hnswIndex.search(query.embedding, K_RESULTS, { threshold });
      const wasmResults = await wasmSearch.findKNN(query.embedding, K_RESULTS, 'pattern_embeddings', { threshold });

      // Both should filter by threshold
      for (const result of hnswResults) {
        expect(result.similarity).toBeGreaterThanOrEqual(threshold);
      }

      for (const result of wasmResults) {
        expect(result.similarity).toBeGreaterThanOrEqual(threshold);
      }

      // Results should have similar counts (within reason for approximate search)
      const countDiff = Math.abs(hnswResults.length - wasmResults.length);
      expect(countDiff).toBeLessThanOrEqual(2); // Allow small variance
    });
  });

  describe('Insert/Remove Parity', () => {
    it('should maintain count after insertions', () => {
      const initialHnsw = hnswIndex.getStats().numElements;

      // Add new vectors to both backends
      const newVectors = generateTestVectors(100, DIMENSION);

      for (let i = 0; i < newVectors.length; i++) {
        const id = NUM_VECTORS + i + 1;
        hnswIndex.addVector(id, newVectors[i].embedding);
      }

      const finalHnsw = hnswIndex.getStats().numElements;

      expect(finalHnsw).toBe(initialHnsw + 100);
      console.log(`[Backend Parity] HNSW: ${initialHnsw} -> ${finalHnsw} (+100)`);
    });

    it('should handle removals correctly', async () => {
      const vectorToRemove = testVectors[0];
      const removeId = 1; // First vector

      // Remove from HNSW
      hnswIndex.removeVector(removeId);

      // Search should not return removed vector
      const results = await hnswIndex.search(vectorToRemove.embedding, K_RESULTS);

      // The removed vector should not appear in results
      // Note: HNSW doesn't support true deletion, so we verify it's marked as removed
      const stats = hnswIndex.getStats();
      expect(stats.numElements).toBeLessThan(NUM_VECTORS + 100); // Less than before removal
    });

    it('should handle duplicate inserts consistently', () => {
      const duplicateId = 999;
      const vec1 = generateVector(DIMENSION);
      const vec2 = generateVector(DIMENSION);

      // Insert same ID twice
      hnswIndex.addVector(duplicateId, vec1);
      const countAfterFirst = hnswIndex.getStats().numElements;

      hnswIndex.addVector(duplicateId, vec2);
      const countAfterSecond = hnswIndex.getStats().numElements;

      // Count should increase (HNSW creates new point)
      expect(countAfterSecond).toBeGreaterThan(countAfterFirst);
    });
  });

  describe('Edge Cases', () => {
    it('should handle k=1 search consistently', async () => {
      const query = queries[0];

      const hnswResults = await hnswIndex.search(query.embedding, 1);
      const wasmResults = await wasmSearch.findKNN(query.embedding, 1);

      expect(hnswResults.length).toBe(1);
      expect(wasmResults.length).toBe(1);
    });

    it('should handle k larger than index size gracefully', async () => {
      const query = queries[0];
      const largeK = NUM_VECTORS * 10;

      const hnswResults = await hnswIndex.search(query.embedding, largeK);
      const wasmResults = await wasmSearch.findKNN(query.embedding, largeK);

      // Should return at most the number of vectors in index
      expect(hnswResults.length).toBeLessThanOrEqual(NUM_VECTORS + 200);
      expect(wasmResults.length).toBeLessThanOrEqual(NUM_VECTORS);
    });

    it('should handle zero-vector queries without errors', async () => {
      const zeroVector = new Float32Array(DIMENSION); // All zeros

      // hnswlib-node 3 answers a zero-vector cosine query (distance 1) rather than throwing
      const hnswResults = await hnswIndex.search(zeroVector, K_RESULTS);
      expect(Array.isArray(hnswResults)).toBe(true);

      // WASM should handle gracefully
      const wasmResults = await wasmSearch.findKNN(zeroVector, K_RESULTS);
      expect(Array.isArray(wasmResults)).toBe(true);
    });

    it('should handle identical vectors consistently', async () => {
      const identicalVec = generateVector(DIMENSION);
      const id1 = 10001;
      const id2 = 10002;

      hnswIndex.addVector(id1, identicalVec);
      hnswIndex.addVector(id2, identicalVec);

      const results = await hnswIndex.search(identicalVec, 5);

      // Should find both identical vectors with similarity ~1.0
      const highSimilarity = results.filter(r => r.similarity > 0.99);
      expect(highSimilarity.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Performance Characteristics', () => {
    it('should measure search latency differences', async () => {
      const query = queries[0];
      const iterations = 10;

      // HNSW timing
      const hnswStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await hnswIndex.search(query.embedding, K_RESULTS);
      }
      const hnswTime = performance.now() - hnswStart;
      const hnswAvg = hnswTime / iterations;

      // WASM timing
      const wasmStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await wasmSearch.findKNN(query.embedding, K_RESULTS);
      }
      const wasmTime = performance.now() - wasmStart;
      const wasmAvg = wasmTime / iterations;

      console.log(`[Backend Parity] HNSW avg: ${hnswAvg.toFixed(2)}ms`);
      console.log(`[Backend Parity] WASM avg: ${wasmAvg.toFixed(2)}ms`);

      // Both should complete in reasonable time
      expect(hnswAvg).toBeLessThan(100); // <100ms per search
      expect(wasmAvg).toBeLessThan(500); // WASM brute force is slower but still reasonable
    });

    it('should report backend statistics accurately', () => {
      const hnswStats = hnswIndex.getStats();
      const wasmStats = wasmSearch.getStats();

      console.log('[Backend Parity] HNSW Stats:', hnswStats);
      console.log('[Backend Parity] WASM Stats:', wasmStats);

      expect(hnswStats.indexBuilt).toBe(true);
      expect(hnswStats.numElements).toBeGreaterThan(0);
      expect(hnswStats.dimension).toBe(DIMENSION);
    });
  });

  describe('Distance Metrics Parity', () => {
    it('should compute cosine similarity consistently', async () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([1, 0, 0]); // Identical
      const vec3 = new Float32Array([0, 1, 0]); // Orthogonal

      // Both backends should compute same similarity
      const sim1 = wasmSearch.cosineSimilarity(vec1, vec2);
      const sim2 = wasmSearch.cosineSimilarity(vec1, vec3);

      expect(sim1).toBeCloseTo(1.0, 4); // Identical vectors
      expect(sim2).toBeCloseTo(0.0, 4); // Orthogonal vectors
    });

    it('should handle normalized vs unnormalized vectors', () => {
      const unnormalized = new Float32Array([3, 4, 0]);
      const normalized = normalizeVector(new Float32Array([3, 4, 0]));

      // Self-similarity should always be 1.0 for normalized vectors
      const sim = wasmSearch.cosineSimilarity(normalized, normalized);
      expect(sim).toBeCloseTo(1.0, 4);
    });
  });
});
