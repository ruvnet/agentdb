/**
 * Unit Tests - SparsificationService
 *
 * Tests graph sparsification methods:
 * - Personalized PageRank (PPR)
 * - Random walk sampling
 * - Spectral sparsification
 * - Degree-based fallback
 * - Top-k selection
 * - Convergence and correctness
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SparsificationService,
  type GraphEdges,
  type GraphInput,
  type SparsificationConfig,
  type SparsificationResult,
} from '../../src/controllers/SparsificationService.js';
import { normalizeGraphEdges } from '../../src/utils/graph-edges.js';

describe('SparsificationService', () => {
  let service: SparsificationService;

  // Test graph: Simple linear chain
  // 0 -> 1 -> 2 -> 3 -> 4
  const linearChain: GraphInput = {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [],
  };

  // Test graph: Star topology
  // 0 is hub connected to 1, 2, 3, 4
  const starGraph: GraphInput = {
    0: [1, 2, 3, 4],
    1: [0],
    2: [0],
    3: [0],
    4: [0],
  };

  // Test graph: Cycle
  // 0 -> 1 -> 2 -> 3 -> 0
  const cycleGraph: GraphInput = {
    0: [1],
    1: [2],
    2: [3],
    3: [0],
  };

  // Test graph: Dense graph with clear clustering
  const denseGraph: GraphInput = {
    0: [1, 2, 3],
    1: [0, 2],
    2: [0, 1, 3],
    3: [0, 2, 4, 5],
    4: [3, 5],
    5: [3, 4],
  };

  beforeEach(async () => {
    const config: SparsificationConfig = {
      method: 'ppr',
      topK: 3,
      alpha: 0.15,
      numWalks: 100,
      walkLength: 10,
    };
    service = new SparsificationService(config);
    await service.initialize();
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const newService = new SparsificationService({
        method: 'ppr',
        topK: 5,
      });
      await newService.initialize();
      expect(newService).toBeDefined();
    });

    it('should set default configuration values', () => {
      const config = service.getConfig();
      expect(config.alpha).toBe(0.15);
      expect(config.numWalks).toBe(100);
      expect(config.walkLength).toBe(10);
      expect(config.convergenceThreshold).toBe(1e-6);
      expect(config.maxIterations).toBe(20);
    });

    it('should allow configuration updates', () => {
      service.updateConfig({ alpha: 0.2, topK: 5 });
      const config = service.getConfig();
      expect(config.alpha).toBe(0.2);
      expect(config.topK).toBe(5);
    });

    it('should reset to default configuration', () => {
      service.updateConfig({ alpha: 0.5 });
      service.resetConfig();
      const config = service.getConfig();
      expect(config.alpha).toBe(0.15);
    });
  });

  describe('PPR Sparsification', () => {
    it('should compute PPR scores for linear chain', async () => {
      const result = await service.pprSparsification(0, linearChain, 3, 0.15);

      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(3);
      expect(result.topKIndices).toContain(0);
      // The source outranks every node that has out-edges. Node 4 is a sink:
      // the native backends keep a dangling node's mass on it (ADR-018), so it
      // can outrank the source there; the JavaScript fallback drops that mass.
      for (const node of [1, 2, 3]) {
        expect(result.scores[0]).toBeGreaterThan(result.scores[node]);
      }
      expect(result.scores).toBeInstanceOf(Float32Array);
      expect(result.method).toMatch(/ppr/);
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should give source node highest PPR score', async () => {
      const result = await service.pprSparsification(0, starGraph, 3, 0.15);

      expect(result.topKIndices[0]).toBe(0);
      expect(result.scores[0]).toBeGreaterThan(result.scores[1]);
      expect(result.scores[0]).toBeGreaterThan(result.scores[2]);
    });

    it('should compute different scores for different source nodes', async () => {
      const result1 = await service.pprSparsification(0, denseGraph, 3, 0.15);
      const result2 = await service.pprSparsification(3, denseGraph, 3, 0.15);

      expect(result1.topKIndices).not.toEqual(result2.topKIndices);
      expect(result1.scores[0]).not.toBe(result2.scores[0]);
    });

    it('should respect alpha parameter (teleport probability)', async () => {
      const lowAlpha = await service.pprSparsification(0, linearChain, 3, 0.01);
      const highAlpha = await service.pprSparsification(0, linearChain, 3, 0.9);

      // High alpha means more restarts -> source node dominates more
      expect(highAlpha.scores[0]).toBeGreaterThan(lowAlpha.scores[0]);
    });

    it('should handle disconnected nodes gracefully', async () => {
      const disconnectedGraph: GraphInput = {
        0: [1],
        1: [],
        2: [3], // Disconnected component
        3: [],
      };

      const result = await service.pprSparsification(0, disconnectedGraph, 2, 0.15);
      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(2);
    });

    it('should converge within max iterations', async () => {
      const result = await service.pprSparsification(0, denseGraph, 3, 0.15);

      if (result.metadata?.iterations) {
        expect(result.metadata.iterations).toBeLessThanOrEqual(20);
      }
    });

    it('should compute correct sparsity ratio', async () => {
      const result = await service.pprSparsification(0, starGraph, 2, 0.15);

      // starGraph has 8 edges total (4 outgoing from 0, 4 returning)
      expect(result.sparsityRatio).toBeCloseTo(2 / 8, 2);
    });

    it('should include metadata', async () => {
      const result = await service.pprSparsification(0, denseGraph, 3, 0.15);

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.totalNodes).toBeGreaterThan(0);
      expect(result.metadata?.totalEdges).toBeGreaterThan(0);
    });
  });

  describe('Random Walk Sparsification', () => {
    it('should perform random walk sampling', async () => {
      const result = await service.randomWalkSparsification(0, linearChain, 3, 100, 10);

      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(3);
      expect(result.method).toBe('random-walk');
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should visit source node most frequently', async () => {
      const result = await service.randomWalkSparsification(0, starGraph, 3, 100, 10);

      // In star graph, walks often return to hub
      expect(result.topKIndices).toContain(0);
      expect(result.scores[0]).toBeGreaterThan(0);
    });

    it('should scale visit counts by number of walks', async () => {
      const result = await service.randomWalkSparsification(0, cycleGraph, 2, 100, 5);

      // All scores should be <= 1.0 (normalized by numWalks)
      for (let i = 0; i < result.scores.length; i++) {
        expect(result.scores[i]).toBeLessThanOrEqual(1.0);
      }
    });

    it('should explore local neighborhood with short walks', async () => {
      const result = await service.randomWalkSparsification(0, linearChain, 2, 100, 2);

      // Short walks (length 2) should stay close to source
      expect(result.topKIndices).toContain(0);
      expect(result.topKIndices).toContain(1);
    });

    it('should handle varying number of walks', async () => {
      const few = await service.randomWalkSparsification(0, denseGraph, 3, 10, 5);
      const many = await service.randomWalkSparsification(0, denseGraph, 3, 1000, 5);

      // More walks should give more stable results
      expect(few.scores).toBeInstanceOf(Float32Array);
      expect(many.scores).toBeInstanceOf(Float32Array);
    });

    it('should terminate on isolated nodes', async () => {
      const isolatedGraph: GraphInput = {
        0: [1],
        1: [], // Dead end
      };

      const result = await service.randomWalkSparsification(0, isolatedGraph, 2, 50, 10);
      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(2);
    });
  });

  describe('Spectral Sparsification', () => {
    it('should perform spectral sparsification', async () => {
      const result = await service.spectralSparsification(denseGraph, 3);

      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(3);
      expect(result.method).toMatch(/spectral|degree-based/);
    });

    it('should fall back to degree-based if spectral unavailable', async () => {
      // Spectral methods likely not available, should use degree-based
      const result = await service.spectralSparsification(starGraph, 2);

      expect(result.method).toMatch(/degree-based/);
      expect(result.topKIndices).toContain(0); // Hub has highest degree
    });
  });

  describe('Degree-Based Sparsification', () => {
    it('should rank nodes by degree', async () => {
      service.updateConfig({ method: 'degree-based' });
      const result = await service.sparsify(0, starGraph);

      // Node 0 (hub) has highest degree
      expect(result.topKIndices[0]).toBe(0);
    });

    it('should handle uniform degree graphs', async () => {
      const uniformGraph: GraphInput = {
        0: [1],
        1: [2],
        2: [0],
      };

      service.updateConfig({ method: 'degree-based', topK: 2 });
      const result = await service.sparsify(0, uniformGraph);

      expect(result).toBeDefined();
      expect(result.topKIndices).toHaveLength(2);
    });

    it('should compute correct degrees', async () => {
      service.updateConfig({ method: 'degree-based', topK: 3 });
      const result = await service.sparsify(0, denseGraph);

      // Node 0 has 3 neighbors, node 3 has 4 neighbors
      expect(result.scores[3]).toBe(4);
      expect(result.scores[0]).toBe(3);
    });
  });

  describe('Sparsify Method', () => {
    it('should route to correct method based on config', async () => {
      service.updateConfig({ method: 'ppr', topK: 2 });
      const pprResult = await service.sparsify(0, linearChain);
      expect(pprResult.method).toMatch(/ppr/);

      service.updateConfig({ method: 'random-walk' });
      const rwResult = await service.sparsify(0, linearChain);
      expect(rwResult.method).toBe('random-walk');

      service.updateConfig({ method: 'degree-based' });
      const degreeResult = await service.sparsify(0, linearChain);
      expect(degreeResult.method).toBe('degree-based');
    });

    it('should throw error for unknown method', async () => {
      service.updateConfig({ method: 'invalid' as any });

      await expect(service.sparsify(0, linearChain)).rejects.toThrow(
        /Unknown sparsification method/
      );
    });

    it('should initialize automatically if not done', async () => {
      const newService = new SparsificationService({
        method: 'ppr',
        topK: 2,
      });

      // Should auto-initialize
      const result = await newService.sparsify(0, linearChain);
      expect(result).toBeDefined();
    });
  });

  describe('Top-K Selection', () => {
    it('should return exactly k nodes', async () => {
      const k = 3;
      service.updateConfig({ topK: k });
      const result = await service.sparsify(0, denseGraph);

      expect(result.topKIndices).toHaveLength(k);
    });

    it('should return nodes in descending score order', async () => {
      const result = await service.pprSparsification(0, denseGraph, 3, 0.15);

      for (let i = 0; i < result.topKIndices.length - 1; i++) {
        const score1 = result.scores[result.topKIndices[i]];
        const score2 = result.scores[result.topKIndices[i + 1]];
        expect(score1).toBeGreaterThanOrEqual(score2);
      }
    });

    it('should handle k larger than graph size', async () => {
      const smallGraph: GraphInput = {
        0: [1],
        1: [],
      };

      const result = await service.pprSparsification(0, smallGraph, 10, 0.15);
      expect(result.topKIndices.length).toBeLessThanOrEqual(10);
    });

    it('should handle k = 0', async () => {
      service.updateConfig({ topK: 0 });
      const result = await service.sparsify(0, linearChain);

      expect(result.topKIndices).toHaveLength(0);
    });

    it('should handle k = 1 (single node)', async () => {
      const result = await service.pprSparsification(0, starGraph, 1, 0.15);

      expect(result.topKIndices).toHaveLength(1);
      expect(result.topKIndices[0]).toBe(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty graph', async () => {
      const emptyGraph: GraphInput = {};

      const result = await service.pprSparsification(0, emptyGraph, 1, 0.15);
      expect(result).toBeDefined();
    });

    it('should handle single-node graph', async () => {
      const singleNode: GraphInput = {
        0: [],
      };

      const result = await service.pprSparsification(0, singleNode, 1, 0.15);
      expect(result.topKIndices).toContain(0);
    });

    it('should handle self-loops', async () => {
      const selfLoopGraph: GraphInput = {
        0: [0, 1],
        1: [1],
      };

      const result = await service.pprSparsification(0, selfLoopGraph, 2, 0.15);
      expect(result).toBeDefined();
    });

    it('should handle large node IDs', async () => {
      const sparseGraph: GraphInput = {
        100: [200],
        200: [300],
        300: [],
      };

      const result = await service.pprSparsification(100, sparseGraph, 2, 0.15);
      expect(result).toBeDefined();
      expect(result.topKIndices).toContain(100);
    });
  });

  describe('Performance Metrics', () => {
    it('should track execution time', async () => {
      const result = await service.pprSparsification(0, denseGraph, 3, 0.15);

      expect(result.executionTimeMs).toBeDefined();
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should compute sparsity ratio correctly', async () => {
      const result = await service.pprSparsification(0, linearChain, 2, 0.15);

      // linearChain has 4 edges total
      const expectedRatio = 2 / 4;
      expect(result.sparsityRatio).toBeCloseTo(expectedRatio, 2);
    });

    it('should track total nodes and edges', async () => {
      const result = await service.pprSparsification(0, denseGraph, 3, 0.15);

      expect(result.metadata?.totalNodes).toBe(6);
      expect(result.metadata?.totalEdges).toBeGreaterThan(0);
    });

    it('should track PPR convergence', async () => {
      const result = await service.pprSparsification(0, cycleGraph, 2, 0.15);

      if (result.method === 'ppr-fallback') {
        expect(result.metadata?.convergence).toBeDefined();
        expect(result.metadata?.convergence).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('Correctness Validation', () => {
    it('PPR scores should sum to approximately 1', async () => {
      const result = await service.pprSparsification(0, cycleGraph, 4, 0.15);

      const sum = Array.from(result.scores).reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 1); // Within 0.1
    });

    it('PPR with alpha=1 should give all weight to source', async () => {
      const result = await service.pprSparsification(0, linearChain, 5, 1.0);

      // With alpha=1, all weight stays at source (no random walk)
      expect(result.scores[0]).toBeGreaterThan(0.9);
    });

    it('Random walk visit counts should be reasonable', async () => {
      const result = await service.randomWalkSparsification(0, starGraph, 3, 100, 5);

      // Each score should represent visits normalized by numWalks
      for (const idx of result.topKIndices) {
        expect(result.scores[idx]).toBeGreaterThan(0);
        expect(result.scores[idx]).toBeLessThanOrEqual(1.0);
      }
    });

    it('Degree scores should match actual degrees', async () => {
      service.updateConfig({ method: 'degree-based', topK: 3 });
      const result = await service.sparsify(0, starGraph);

      // Node 0 has degree 4
      expect(result.scores[0]).toBe(4);
      // Leaf nodes have degree 1
      expect(result.scores[1]).toBe(1);
    });
  });

  describe('Graph input forms', () => {
    const asArray: GraphEdges = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2, 4, 5], [3, 5], [3, 4]];

    it('scores an object-keyed graph and the same adjacency array identically', async () => {
      for (const method of ['ppr', 'random-walk', 'spectral', 'degree-based'] as const) {
        service.updateConfig({ method, topK: 3 });
        const fromObject = await service.sparsify(0, denseGraph);
        const fromArray = await service.sparsify(0, asArray);
        if (method !== 'random-walk') {
          expect(Array.from(fromObject.scores)).toEqual(Array.from(fromArray.scores));
        }
        expect(fromObject.scores).toHaveLength(6);
      }
    });

    it('counts nodes that appear only as neighbours', () => {
      expect(normalizeGraphEdges({ 0: [3] })).toEqual([[3], [], [], []]);
    });

    it('rejects input that is neither an array nor an object keyed by node id', () => {
      expect(() => normalizeGraphEdges('0:1' as any)).toThrow(TypeError);
      expect(() => normalizeGraphEdges({ a: [1] } as any)).toThrow(/non-negative integer/);
      expect(() => normalizeGraphEdges({ 0: 1 } as any)).toThrow(/must be an array/);
      expect(() => normalizeGraphEdges({ 2e7: [] })).toThrow(RangeError);
    });
  });
});
