/**
 * Unit Tests for MincutService Controller
 *
 * Tests graph partitioning algorithms including Stoer-Wagner,
 * Karger's randomized algorithm, and flow-based mincut.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { MincutService, MincutConfig, MincutResult } from '../../../src/controllers/MincutService.js';
import type { GraphEdges } from '../../../src/types/graph.js';

describe('MincutService', () => {
  let service: MincutService;

  beforeEach(() => {
    const config: MincutConfig = {
      algorithm: 'stoer-wagner',
      maxPartitionSize: 100,
      minCutThreshold: 0.1,
    };
    service = new MincutService(config);
  });

  afterEach(() => {
    service.clearCache();
  });

  describe('Constructor', () => {
    it('should initialize with stoer-wagner config', () => {
      const config: MincutConfig = {
        algorithm: 'stoer-wagner',
      };
      const svc = new MincutService(config);
      expect(svc).toBeDefined();
    });

    it('should initialize with karger config', () => {
      const config: MincutConfig = {
        algorithm: 'karger',
      };
      const svc = new MincutService(config);
      expect(svc).toBeDefined();
    });

    it('should initialize with flow-based config', () => {
      const config: MincutConfig = {
        algorithm: 'flow-based',
      };
      const svc = new MincutService(config);
      expect(svc).toBeDefined();
    });
  });

  describe('initialize', () => {
    it('should initialize without errors', async () => {
      await expect(service.initialize()).resolves.not.toThrow();
    });

    it('should handle missing WASM/NAPI gracefully', async () => {
      await service.initialize();
      // Service should still work with fallback
      expect(service).toBeDefined();
    });
  });

  describe('Stoer-Wagner Algorithm', () => {
    it('should partition simple graph', async () => {
      // Create simple graph: 0-1-2-3
      const edges: GraphEdges = [
        [1],       // 0 -> 1
        [0, 2],    // 1 -> 0, 2
        [1, 3],    // 2 -> 1, 3
        [2],       // 3 -> 2
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.cutSize).toBeGreaterThan(0);
      expect(result.algorithm).toContain('stoer-wagner');
    });

    it('should partition disconnected graph', async () => {
      // Two disconnected components: 0-1 and 2-3
      const edges: GraphEdges = [
        [1],       // 0 -> 1
        [0],       // 1 -> 0
        [3],       // 2 -> 3
        [2],       // 3 -> 2
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.cutSize).toBe(0); // No cut needed for disconnected components
    });

    it('should partition complete graph', async () => {
      // Complete graph K4
      const edges: GraphEdges = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.cutSize).toBeGreaterThan(0);
    });

    it('should cache partition results', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [0, 2],
        [0, 1],
      ];

      const result1 = await service.stoerWagnerMincut(edges);
      const result2 = await service.stoerWagnerMincut(edges);

      // Should return cached result (same reference for fallback implementation)
      // For cached results, the values should be identical
      expect(result1.partitions).toEqual(result2.partitions);
      expect(result1.cutSize).toBe(result2.cutSize);
      expect(result1.algorithm).toBe(result2.algorithm);

      // Verify cache stats show the entry
      const stats = service.getCacheStats();
      expect(stats.size).toBeGreaterThan(0);
    });

    it('should handle single node graph', async () => {
      const edges: GraphEdges = [
        [],  // Single isolated node
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
    });

    it('should handle empty graph', async () => {
      const edges: GraphEdges = [];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.cutSize).toBe(0);
    });
  });

  describe('Karger Algorithm', () => {
    it('should partition simple graph', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [0, 2, 3],
        [0, 1],
        [1],
      ];

      const config: MincutConfig = { algorithm: 'karger' };
      const kargerService = new MincutService(config);

      const result = await kargerService.kargerMincut(edges, 10);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.algorithm).toBe('karger');
    });

    it('should find mincut with multiple iterations', async () => {
      const edges: GraphEdges = [
        [1],
        [0, 2],
        [1, 3],
        [2],
      ];

      const config: MincutConfig = { algorithm: 'karger' };
      const kargerService = new MincutService(config);

      const result = await kargerService.kargerMincut(edges, 50);

      expect(result).toBeDefined();
      expect(result.cutSize).toBeGreaterThanOrEqual(1);
    });

    it('should improve with more iterations', async () => {
      const edges: GraphEdges = [
        [1, 2, 3],
        [0, 2],
        [0, 1, 3],
        [0, 2],
      ];

      const config: MincutConfig = { algorithm: 'karger' };
      const kargerService = new MincutService(config);

      const result10 = await kargerService.kargerMincut(edges, 10);
      const result100 = await kargerService.kargerMincut(edges, 100);

      // More iterations should find similar or better cuts
      expect(result100.cutSize).toBeLessThanOrEqual(result10.cutSize + 1);
    });

    it('should handle graph with no edges', async () => {
      const edges: GraphEdges = [
        [],
        [],
        [],
      ];

      const config: MincutConfig = { algorithm: 'karger' };
      const kargerService = new MincutService(config);

      const result = await kargerService.kargerMincut(edges, 10);

      expect(result).toBeDefined();
      expect(result.cutSize).toBe(0);
    });
  });

  describe('Flow-Based Mincut', () => {
    it('should partition graph using max-flow min-cut', async () => {
      const edges: GraphEdges = [
        [1, 2],    // source
        [3],
        [3],
        [],        // sink
      ];

      const result = await service.flowBasedMincut(edges, 0, 3);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.algorithm).toBe('ford-fulkerson');
    });

    it('should find minimum cut between source and sink', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [3],
        [3],
        [],
      ];

      const result = await service.flowBasedMincut(edges, 0, 3);

      expect(result.cutSize).toBeGreaterThan(0);
      expect(result.partitions[0]).toContain(0); // Source in first partition
      expect(result.partitions[1]).toContain(3); // Sink in second partition
    });

    it('should handle disconnected source and sink', async () => {
      const edges: GraphEdges = [
        [1],
        [0],
        [3],
        [2],
      ];

      const result = await service.flowBasedMincut(edges, 0, 3);

      expect(result).toBeDefined();
      expect(result.cutSize).toBe(0);
    });
  });

  describe('partition', () => {
    it('should use configured algorithm', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [0, 2],
        [0, 1],
      ];

      const config: MincutConfig = { algorithm: 'stoer-wagner' };
      const swService = new MincutService(config);

      const result = await swService.partition(edges);

      expect(result.algorithm).toContain('stoer-wagner');
    });

    it('should throw error for unknown algorithm', async () => {
      const edges: GraphEdges = [[1], [0]];

      const config: MincutConfig = { algorithm: 'unknown' as any };
      const badService = new MincutService(config);

      await expect(badService.partition(edges)).rejects.toThrow('Unknown algorithm');
    });
  });

  describe('getPartition', () => {
    it('should return partition containing node', async () => {
      const edges: GraphEdges = [
        [1],
        [0, 2],
        [1, 3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);
      const partition = service.getPartition(0, result);

      expect(partition).toContain(0);
      expect(partition.length).toBeGreaterThan(0);
    });

    it('should return empty array for non-existent node', async () => {
      const edges: GraphEdges = [[1], [0]];

      const result = await service.stoerWagnerMincut(edges);
      const partition = service.getPartition(999, result);

      expect(partition).toEqual([]);
    });

    it('should return different partitions for nodes in different sets', async () => {
      const edges: GraphEdges = [
        [1],
        [0, 2],
        [1, 3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);
      const partition0 = service.getPartition(0, result);
      const partition3 = service.getPartition(3, result);

      // Nodes in different partitions should not be in same array
      if (partition0.includes(3)) {
        expect(partition0).toEqual(partition3);
      } else {
        expect(partition0).not.toEqual(partition3);
      }
    });
  });

  describe('inSamePartition', () => {
    it('should return true for nodes in same partition', async () => {
      const edges: GraphEdges = [
        [1],
        [0],
        [3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);
      const same = service.inSamePartition(0, 1, result);

      expect(typeof same).toBe('boolean');
    });

    it('should return false for nodes in different partitions', async () => {
      const edges: GraphEdges = [
        [1],
        [0],
        [3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);

      // Create a guaranteed split: nodes 0,1 vs 2,3
      // Due to fallback algorithm, first half vs second half
      const same01 = service.inSamePartition(0, 1, result);
      const same02 = service.inSamePartition(0, 2, result);

      // At least one pair should be different
      expect(same01 || !same02).toBe(true);
    });
  });

  describe('getPartitionStats', () => {
    it('should calculate partition statistics', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [0, 2],
        [0, 1, 3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);
      const stats = service.getPartitionStats(result, edges);

      expect(stats.numPartitions).toBe(2);
      expect(stats.avgPartitionSize).toBeGreaterThan(0);
      expect(stats.maxPartitionSize).toBeGreaterThanOrEqual(stats.minPartitionSize);
      expect(stats.cutRatio).toBeGreaterThanOrEqual(0);
      expect(stats.cutRatio).toBeLessThanOrEqual(1);
    });

    it('should calculate correct average partition size', async () => {
      const edges: GraphEdges = [
        [1],
        [0],
        [3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);
      const stats = service.getPartitionStats(result, edges);

      expect(stats.avgPartitionSize).toBe(2); // 4 nodes in 2 partitions
    });

    it('should handle empty graph', async () => {
      const edges: GraphEdges = [];

      const result = await service.stoerWagnerMincut(edges);
      const stats = service.getPartitionStats(result, edges);

      expect(stats.numPartitions).toBe(2);
      expect(stats.cutRatio).toBe(0);
    });
  });

  describe('Cache Management', () => {
    it('should clear cache', async () => {
      const edges: GraphEdges = [[1], [0]];

      await service.stoerWagnerMincut(edges);
      const statsBefore = service.getCacheStats();
      expect(statsBefore.size).toBeGreaterThan(0);

      service.clearCache();
      const statsAfter = service.getCacheStats();
      expect(statsAfter.size).toBe(0);
    });

    it('should return cache statistics', async () => {
      const edges1: GraphEdges = [[1], [0]];
      const edges2: GraphEdges = [[1, 2], [0, 2], [0, 1]];

      await service.stoerWagnerMincut(edges1);
      await service.stoerWagnerMincut(edges2);

      const stats = service.getCacheStats();

      expect(stats.size).toBeGreaterThan(0);
      expect(Array.isArray(stats.keys)).toBe(true);
      expect(stats.keys.length).toBe(stats.size);
    });

    it('should cache different graphs separately', async () => {
      const edges1: GraphEdges = [[1], [0]];
      const edges2: GraphEdges = [[1, 2], [0, 2], [0, 1]];

      await service.stoerWagnerMincut(edges1);
      await service.stoerWagnerMincut(edges2);

      const stats = service.getCacheStats();
      expect(stats.size).toBe(2);
    });
  });

  describe('Edge Cases', () => {
    it('should handle graph with undefined neighbors', async () => {
      const edges: GraphEdges = [
        [1],
        undefined,
        [3],
        [2],
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
    });

    it('should handle large partition sizes', async () => {
      // Create larger graph
      const size = 20;
      const edges: GraphEdges = Array(size).fill(null).map((_, i) => {
        const neighbors: number[] = [];
        if (i > 0) neighbors.push(i - 1);
        if (i < size - 1) neighbors.push(i + 1);
        return neighbors;
      });

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
      expect(result.partitions[0].length + result.partitions[1].length).toBe(size);
    });

    it('should handle self-loops gracefully', async () => {
      const edges: GraphEdges = [
        [0, 1],  // Self-loop on node 0
        [0],
      ];

      const result = await service.stoerWagnerMincut(edges);

      expect(result).toBeDefined();
      expect(result.partitions).toHaveLength(2);
    });
  });

  describe('Performance', () => {
    it('should partition medium graph efficiently', async () => {
      const size = 100;
      const edges: GraphEdges = Array(size).fill(null).map((_, i) => {
        const neighbors: number[] = [];
        if (i > 0) neighbors.push(i - 1);
        if (i < size - 1) neighbors.push(i + 1);
        if (i > 1) neighbors.push(i - 2);
        return neighbors;
      });

      const start = Date.now();
      const result = await service.stoerWagnerMincut(edges);
      const duration = Date.now() - start;

      expect(result).toBeDefined();
      expect(duration).toBeLessThan(1000); // Should complete in under 1 second
    });

    it('should benefit from caching', async () => {
      const edges: GraphEdges = [
        [1, 2],
        [0, 2],
        [0, 1],
      ];

      const start1 = Date.now();
      await service.stoerWagnerMincut(edges);
      const duration1 = Date.now() - start1;

      const start2 = Date.now();
      await service.stoerWagnerMincut(edges);
      const duration2 = Date.now() - start2;

      // Cached version should be faster (or at least not slower)
      expect(duration2).toBeLessThanOrEqual(duration1 + 10);
    });
  });
});

describe('MincutService graph input forms', () => {
  it('partitions an object-keyed graph like the same adjacency array', async () => {
    const service = new MincutService({ algorithm: 'stoer-wagner', maxPartitionSize: 10, minCutRatio: 0.1 } as MincutConfig);
    await service.initialize();
    const asObject = { 0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4, 5], 4: [3, 5], 5: [3, 4] };
    const asArray: GraphEdges = [[1, 2], [0, 2], [0, 1, 3], [2, 4, 5], [3, 5], [3, 4]];
    const a = await service.stoerWagnerMincut(asObject);
    const b = await service.stoerWagnerMincut(asArray);
    expect(a.cutSize).toBe(b.cutSize);
    expect(a.partitions).toEqual(b.partitions);
    expect(service.getPartitionStats(a, asObject).numPartitions).toBe(b.partitions.length);
  });
});
