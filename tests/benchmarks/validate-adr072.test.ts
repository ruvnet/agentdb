/**
 * ADR-072 Validation Test - Quick smoke test
 *
 * Validates that all ADR-072 components can be imported and initialized
 */

import { describe, it, expect } from 'vitest';
import {
  generateRandomGraph,
  generateScaleFreeGraph,
  generateSmallWorldGraph,
  calculateGraphStats
} from './helpers/graph-generator.js';

describe('ADR-072 Component Validation', () => {
  it('should generate random graphs', () => {
    const graph = generateRandomGraph({
      numNodes: 100,
      avgDegree: 4,
      seed: 42
    });

    // Adjacency array (GraphEdges): what SparsificationService and
    // MincutService accept. Undirected, so every edge appears both ways.
    expect(graph.length).toBe(100);
    expect(graph.some((neighbors) => (neighbors?.length ?? 0) > 0)).toBe(true);
    graph.forEach((neighbors, u) => {
      for (const v of neighbors ?? []) expect(graph[v]).toContain(u);
    });

    const stats = calculateGraphStats(graph);
    console.log(`✅ Random graph: ${stats.numNodes} nodes, ${stats.numEdges} edges`);
    expect(stats.numNodes).toBe(100);
  });

  it('should generate scale-free graphs', () => {
    const graph = generateScaleFreeGraph({
      numNodes: 100,
      m0: 5,
      m: 3,
      seed: 42
    });

    const stats = calculateGraphStats(graph);
    console.log(`✅ Scale-free graph: ${stats.numNodes} nodes, ${stats.numEdges} edges`);
    expect(stats.numNodes).toBe(100);
  });

  it('should generate small-world graphs', () => {
    const graph = generateSmallWorldGraph({
      numNodes: 100,
      avgDegree: 4,
      seed: 42
    });

    const stats = calculateGraphStats(graph);
    console.log(`✅ Small-world graph: ${stats.numNodes} nodes, ${stats.numEdges} edges`);
    expect(stats.numNodes).toBe(100);
  });

  it('should calculate graph statistics correctly', () => {
    const graph = generateRandomGraph({
      numNodes: 50,
      avgDegree: 4,
      seed: 42
    });

    const stats = calculateGraphStats(graph);

    expect(stats.numNodes).toBe(50);
    expect(stats.numEdges).toBeGreaterThan(0);
    expect(stats.avgDegree).toBeGreaterThan(0);
    expect(stats.density).toBeGreaterThan(0);
    expect(stats.density).toBeLessThan(1);
    expect(stats.maxDegree).toBeGreaterThanOrEqual(stats.avgDegree);
    expect(stats.minDegree).toBeLessThanOrEqual(stats.avgDegree);

    console.log('✅ Graph stats:', stats);
  });
});
