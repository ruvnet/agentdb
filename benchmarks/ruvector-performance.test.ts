/**
 * RuVector Performance Benchmarks
 *
 * Comprehensive benchmark suite validating optimal RuVector integration
 * across all AgentDB components:
 * - @ruvector/core: Vector database performance
 * - @ruvector/graph-node: Graph database performance
 * - @ruvector/gnn: GNN operations performance
 * - SDK integration: ReflexionMemory, SkillLibrary, etc.
 */

import { describe, it, expect, afterAll } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { performance } from 'perf_hooks';
import { SPEED_GATES } from '../tests/benchmarks/helpers/speed-gate';

const BENCH_DIR = path.join(process.cwd(), 'bench-data');
const RESULTS: Record<string, any> = {};

// Ensure benchmark directory exists (results JSON is written there)
if (!fs.existsSync(BENCH_DIR)) {
  fs.mkdirSync(BENCH_DIR, { recursive: true });
}

// Databases go in a fresh directory per run. Fixed paths kept state between
// runs (duplicate node ids), and VectorDb without storagePath writes
// ./ruvector.db in the working directory.
const DB_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-ruvector-bench-'));

/** Import a CommonJS native package's exports under either interop shape. */
async function load(name: string): Promise<any> {
  const mod: any = await import(name);
  return mod.default && Object.keys(mod).length <= 2 ? { ...mod.default, ...mod } : mod;
}

const randomVector = (dim: number) => new Float32Array(dim).map(() => Math.random());

/** Throughput floor; machine speed, so skipped under CI (#44). */
function expectThroughput(opsPerSec: number, floor: number) {
  expect(Number.isFinite(opsPerSec)).toBe(true);
  expect(opsPerSec).toBeGreaterThan(0);
  if (SPEED_GATES) expect(opsPerSec).toBeGreaterThan(floor);
}

/**
 * Benchmark helper. A plain async function: the old `new Promise(async
 * resolve => ...)` wrapper swallowed any throw from `fn` and never
 * resolved, so one failing call hung the suite until the 600 s timeout.
 */
async function benchmark(name: string, fn: () => Promise<void> | void, iterations: number = 1): Promise<number> {
  const start = performance.now();

  for (let i = 0; i < iterations; i++) {
    await fn();
  }

  const end = performance.now();
  const duration = end - start;
  const avgDuration = duration / iterations;
  const opsPerSec = iterations / (duration / 1000);

  RESULTS[name] = {
    iterations,
    totalDurationMs: duration.toFixed(2),
    avgDurationMs: avgDuration.toFixed(4),
    opsPerSec: Math.floor(opsPerSec)
  };

  console.log(`\n📊 ${name}`);
  console.log(`   Iterations: ${iterations}`);
  console.log(`   Total: ${duration.toFixed(2)}ms`);
  console.log(`   Average: ${avgDuration.toFixed(4)}ms`);
  console.log(`   Throughput: ${Math.floor(opsPerSec).toLocaleString()} ops/sec`);

  return opsPerSec;
}

// @ruvector/core 0.1.x exports VectorDb (the VectorDB alias exists only in
// index.d.ts), with insert({ id, vector }) and search({ vector, k }).
describe('RuVector Core (@ruvector/core) Performance', () => {
  it('should benchmark vector insert performance', async () => {
    const { VectorDb } = await load('@ruvector/core');

    const db = new VectorDb({
      dimensions: 384,
      storagePath: path.join(DB_DIR, 'core-insert.db'),
      distanceMetric: 'Cosine'
    });

    // Single insert
    let n = 0;
    const singleInsert = await benchmark('Vector Insert (single)', async () => {
      const id = await db.insert({ id: `bench-single-${n++}`, vector: randomVector(384) });
      expect(typeof id).toBe('string');
    }, 100);

    expectThroughput(singleInsert, 1000); // At least 1K ops/sec

    // Batch insert
    let round = 0;
    const batchInsert = await benchmark('Vector Insert (batch 100)', async () => {
      const entries = Array.from({ length: 100 }, (_, i) => ({
        id: `batch-${round}-${i}`,
        vector: randomVector(384)
      }));
      round++;
      const ids = await db.insertBatch(entries);
      expect(ids).toHaveLength(100);
    }, 10);

    expect(await db.len()).toBe(100 + 10 * 100);

    // Batch throughput should be 10K+ ops/sec (100 vectors per batch * throughput)
    const batchThroughput = batchInsert * 100;
    expectThroughput(batchThroughput, 10000);

    console.log(`   → Batch throughput: ${batchThroughput.toLocaleString()} vectors/sec`);
  });

  it('should benchmark vector search performance', async () => {
    const { VectorDb } = await load('@ruvector/core');

    const db = new VectorDb({
      dimensions: 384,
      storagePath: path.join(DB_DIR, 'core-search.db'),
      distanceMetric: 'Cosine'
    });

    // Insert test data
    await db.insertBatch(
      Array.from({ length: 1000 }, (_, i) => ({ id: `search-${i}`, vector: randomVector(384) }))
    );

    const queryVector = randomVector(384);
    const checkResults = (results: Array<{ id: string; score: number }>, k: number) => {
      expect(results.length).toBeGreaterThan(0);
      expect(results.length).toBeLessThanOrEqual(k);
      for (const r of results) {
        expect(r.id.startsWith('search-')).toBe(true);
        expect(Number.isFinite(r.score)).toBe(true);
      }
    };

    // Search k=10
    const search10 = await benchmark('Vector Search (k=10)', async () => {
      checkResults(await db.search({ vector: queryVector, k: 10 }), 10);
    }, 100);

    expectThroughput(search10, 1000); // At least 1K searches/sec

    // Search k=100
    const search100 = await benchmark('Vector Search (k=100)', async () => {
      checkResults(await db.search({ vector: queryVector, k: 100 }), 100);
    }, 100);

    expectThroughput(search100, 500); // At least 500 searches/sec
  });
});

describe('RuVector Graph Database (@ruvector/graph-node) Performance', () => {
  it('should benchmark node creation performance', async () => {
    const { GraphDatabase } = await load('@ruvector/graph-node');

    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: path.join(DB_DIR, 'bench-graph.db')
    });

    const embedding = randomVector(384);

    // Single node creation
    let n = 0;
    const singleNode = await benchmark('Graph Node Create (single)', async () => {
      const id = await db.createNode({
        id: `node-${n++}`,
        embedding,
        labels: ['Test'],
        properties: { type: 'benchmark' }
      });
      expect(typeof id).toBe('string');
    }, 100);

    expectThroughput(singleNode, 1000); // At least 1K ops/sec

    // Batch node creation (fresh ids each round)
    let round = 0;
    const batchCreate = await benchmark('Graph Node Create (batch 100)', async () => {
      const nodes = Array.from({ length: 100 }, (_, i) => ({
        id: `batch-node-${round}-${i}`,
        embedding: randomVector(384),
        labels: ['Test'],
        properties: { type: 'benchmark', index: i.toString() }
      }));
      round++;
      const result = await db.batchInsert({ nodes, edges: [] });
      expect(result.nodeIds).toHaveLength(100);
    }, 10);

    const batchThroughput = batchCreate * 100;
    expectThroughput(batchThroughput, 50000); // At least 50K nodes/sec

    console.log(`   → Batch throughput: ${batchThroughput.toLocaleString()} nodes/sec`);
  });

  it('should benchmark Cypher query performance', async () => {
    const { GraphDatabase } = await load('@ruvector/graph-node');

    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: path.join(DB_DIR, 'bench-cypher.db')
    });

    // Insert test data
    for (let i = 0; i < 100; i++) {
      await db.createNode({
        id: `cypher-${i}`,
        embedding: randomVector(384),
        labels: ['Test'],
        properties: { index: i.toString(), type: 'benchmark' }
      });
    }

    // Simple query
    const simpleQuery = await benchmark('Cypher Query (MATCH simple)', async () => {
      const result = await db.query('MATCH (n:Test) RETURN n LIMIT 10');
      // @ruvector/graph-node 2.1.2 does not apply LIMIT (returns all 100).
      expect(result.nodes.length).toBeGreaterThan(0);
      expect(result.nodes.every((node: any) => node.labels.includes('Test'))).toBe(true);
    }, 100);

    expectThroughput(simpleQuery, 500); // At least 500 queries/sec

    // Complex query with filtering
    const complexQuery = await benchmark('Cypher Query (MATCH with WHERE)', async () => {
      const result = await db.query('MATCH (n:Test) WHERE n.type = "benchmark" RETURN n LIMIT 10');
      expect(Array.isArray(result.nodes)).toBe(true);
    }, 100);

    expectThroughput(complexQuery, 300); // At least 300 queries/sec
  });

  it('should benchmark graph traversal performance', async () => {
    const { GraphDatabase } = await load('@ruvector/graph-node');

    const db = new GraphDatabase({
      dimensions: 384,
      distanceMetric: 'Cosine',
      storagePath: path.join(DB_DIR, 'bench-traversal.db')
    });

    // Create a small graph
    const nodeIds: string[] = [];
    for (let i = 0; i < 50; i++) {
      const id = await db.createNode({
        id: `trav-${i}`,
        embedding: randomVector(384),
        labels: ['Traversal'],
        properties: { index: i.toString() }
      });
      nodeIds.push(id);
    }

    // Create edges
    for (let i = 0; i < 45; i++) {
      await db.createEdge({
        from: nodeIds[i],
        to: nodeIds[i + 1],
        description: 'next',
        embedding: randomVector(384),
        confidence: 0.9
      });
    }

    // Benchmark k-hop traversal (kHop was renamed kHopNeighbors)
    const traversal = await benchmark('Graph Traversal (2-hop)', async () => {
      const neighbors: string[] = await db.kHopNeighbors(nodeIds[0], 2);
      expect(neighbors).toContain(nodeIds[1]);
    }, 100);

    expectThroughput(traversal, 100); // At least 100 traversals/sec
  });
});

describe('RuVector GNN (@ruvector/gnn) Performance', () => {
  it('should benchmark GNN forward pass performance', async () => {
    const { RuvectorLayer } = await load('@ruvector/gnn');

    const layer = new RuvectorLayer(384, 512, 4, 0.1);
    const nodeEmbedding = randomVector(384);
    const neighborEmbeddings = [randomVector(384), randomVector(384), randomVector(384)];
    // forward() takes the edge weights as a Float32Array
    const edgeWeights = new Float32Array([0.8, 0.6, 0.9]);

    const output: Float32Array = layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);
    expect(output.length).toBe(512);
    expect(Array.from(output).every(Number.isFinite)).toBe(true);

    const forwardPass = await benchmark('GNN Forward Pass (3 neighbors)', async () => {
      layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);
    }, 1000);

    expectThroughput(forwardPass, 500); // At least 500 forward passes/sec
  });

  it('should benchmark tensor compression performance', async () => {
    const { TensorCompress } = await load('@ruvector/gnn');

    const compressor = new TensorCompress();
    const tensor = randomVector(384);

    const compression = await benchmark('Tensor Compression', async () => {
      compressor.compress(tensor, 0.5);
    }, 1000);

    expectThroughput(compression, 1000); // At least 1K compressions/sec

    const compressed: string = compressor.compress(tensor, 0.5);
    expect(typeof compressed).toBe('string');
    const restored: Float32Array = compressor.decompress(compressed);
    expect(restored.length).toBe(384);
    expect(Array.from(restored).every(Number.isFinite)).toBe(true);

    const decompression = await benchmark('Tensor Decompression', async () => {
      compressor.decompress(compressed);
    }, 1000);

    expectThroughput(decompression, 1000); // At least 1K decompressions/sec
  });

  // The DifferentiableSearch class is now the differentiableSearch(query,
  // candidates, k, temperature) function.
  it('should benchmark differentiable search performance', async () => {
    const { differentiableSearch } = await load('@ruvector/gnn');

    const query = randomVector(128);
    const candidates = Array.from({ length: 100 }, () => randomVector(128));

    const result = differentiableSearch(query, candidates, 10, 1.0);
    expect(result.indices).toHaveLength(10);
    expect(result.weights).toHaveLength(10);
    for (const idx of result.indices) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(100);
    }
    expect(result.weights.every(Number.isFinite)).toBe(true);

    const search = await benchmark('Differentiable Search (100 candidates)', async () => {
      differentiableSearch(query, candidates, 10, 1.0);
    }, 100);

    expectThroughput(search, 100); // At least 100 searches/sec
  });
});

describe('AgentDB SDK Integration Performance', () => {
  it('should benchmark ReflexionMemory with GraphDatabase', async () => {
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const { ReflexionMemory } = await import('../src/controllers/ReflexionMemory.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    const db = await createUnifiedDatabase(
      path.join(DB_DIR, 'bench-reflexion.graph'),
      embedder,
      { forceMode: 'graph' }
    );

    const graphDb = db.getGraphDatabase()!;
    const reflexion = new ReflexionMemory(graphDb as any, embedder, undefined, undefined, graphDb as any);

    // Benchmark episode storage
    const storeEpisode = await benchmark('ReflexionMemory Store Episode', async () => {
      await reflexion.storeEpisode({
        sessionId: 'bench',
        task: `task-${Math.random()}`,
        reward: Math.random(),
        success: true,
        input: 'benchmark input',
        output: 'benchmark output'
      });
    }, 50);

    expectThroughput(storeEpisode, 10); // At least 10 episodes/sec

    // Benchmark episode retrieval
    const retrieveEpisode = await benchmark('ReflexionMemory Retrieve Episodes', async () => {
      const episodes = await reflexion.retrieveRelevant({ task: 'benchmark', k: 10 });
      expect(episodes.length).toBeGreaterThan(0);
      expect(episodes.length).toBeLessThanOrEqual(10);
    }, 50);

    expectThroughput(retrieveEpisode, 10); // At least 10 retrievals/sec

    db.close();
  });

  it('should benchmark SkillLibrary with GraphDatabase', async () => {
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const { SkillLibrary } = await import('../src/controllers/SkillLibrary.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    const db = await createUnifiedDatabase(
      path.join(DB_DIR, 'bench-skills.graph'),
      embedder,
      { forceMode: 'graph' }
    );

    const graphDb = db.getGraphDatabase()!;
    // The graph backend is the 4th argument (3rd is the vector backend);
    // passed 3rd, createSkill fell through to SQLite (this.db.prepare).
    const skills = new SkillLibrary(graphDb as any, embedder, undefined, graphDb as any);

    // Benchmark skill creation
    const createSkill = await benchmark('SkillLibrary Create Skill', async () => {
      await skills.createSkill({
        name: `skill-${Math.random()}`,
        description: 'benchmark skill',
        code: 'function bench() { return true; }',
        successRate: Math.random()
      });
    }, 50);

    expectThroughput(createSkill, 10); // At least 10 skills/sec

    // Benchmark skill search
    const searchSkill = await benchmark('SkillLibrary Search Skills', async () => {
      // The graph path filters on avgReward (stored as 0 here) as the success
      // rate, so the default minSuccessRate of 0.5 would drop every skill.
      // Not bounded by k: the adapter's Cypher LIMIT is ignored by
      // @ruvector/graph-node 2.1.2, so all stored skills come back.
      const found = await skills.searchSkills({ query: 'benchmark', k: 10, minSuccessRate: 0 });
      expect(found.length).toBeGreaterThan(0);
      expect(found.every((skill) => skill.name.startsWith('skill-'))).toBe(true);
    }, 50);

    expectThroughput(searchSkill, 10); // At least 10 searches/sec

    db.close();
  });
});

// After all tests, write results to file
afterAll(() => {
  fs.rmSync(DB_DIR, { recursive: true, force: true });
  const resultsPath = path.join(BENCH_DIR, 'benchmark-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify(RESULTS, null, 2));

  console.log(`\n\n📊 BENCHMARK RESULTS SUMMARY\n`);
  console.log('═'.repeat(70));

  for (const [name, data] of Object.entries(RESULTS)) {
    console.log(`\n${name}:`);
    console.log(`  Throughput: ${data.opsPerSec.toLocaleString()} ops/sec`);
    console.log(`  Avg Latency: ${data.avgDurationMs}ms`);
  }

  console.log(`\n${'═'.repeat(70)}`);
  console.log(`\n✅ Results saved to: ${resultsPath}\n`);
});
