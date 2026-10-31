/**
 * ADR-015 / E7 benchmark: full-scan search at N=10k, 384-dim, k=10.
 *
 * Times the four brute-force scan sites (ReflexionMemory SQL fallback,
 * SkillLibrary legacy, WASMVectorSearch.findKNN, SqlJsRvfBackend search).
 * Everything is in memory; nothing is written to the working directory.
 *
 *   npx tsx benchmarks/topk-scan-bench.mts            # N=10000
 *   BENCH_N=20000 npx tsx benchmarks/topk-scan-bench.mts
 *
 * For a before/after comparison, run it once on the current tree and once
 * with the four source files checked out from the previous release.
 */
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { ReflexionMemory } from '../src/controllers/ReflexionMemory.js';
import { SkillLibrary } from '../src/controllers/SkillLibrary.js';
import { WASMVectorSearch } from '../src/controllers/WASMVectorSearch.js';
import { SqlJsRvfBackend } from '../src/backends/rvf/SqlJsRvfBackend.js';
import type { EmbeddingService } from '../src/controllers/EmbeddingService.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA = fs.readFileSync(path.join(here, '../src/schemas/schema.sql'), 'utf-8');
const N = Number(process.env.BENCH_N ?? 10_000);
const DIM = 384;
const K = 10;
const RUNS = Number(process.env.BENCH_RUNS ?? 20);

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(1);
const vec = () => Float32Array.from({ length: DIM }, () => rand() * 2 - 1);
const vectors = Array.from({ length: N }, vec);
const queries = Array.from({ length: RUNS }, vec);
const blob = (v: Float32Array) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);

let qi = 0;
const embedder = {
  embedQuery: async () => queries[qi++ % RUNS],
  embedPassage: async () => queries[0],
} as unknown as EmbeddingService;

async function time(label: string, run: () => Promise<unknown>): Promise<void> {
  await run(); // warm-up
  const samples: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const t0 = process.hrtime.bigint();
    await run();
    samples.push(Number(process.hrtime.bigint() - t0) / 1e6);
  }
  samples.sort((a, b) => a - b);
  const median = samples[Math.floor(samples.length / 2)];
  const p95 = samples[Math.floor(samples.length * 0.95) - 1] ?? samples[samples.length - 1];
  console.log(`${label.padEnd(34)} median ${median.toFixed(2).padStart(8)} ms   p95 ${p95.toFixed(2).padStart(8)} ms`);
}

const db = new Database(':memory:');
db.exec(SCHEMA);
db.exec('CREATE TABLE IF NOT EXISTS pattern_embeddings (pattern_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)');
const insert = db.transaction(() => {
  const ep = db.prepare(`INSERT INTO episodes (id, session_id, task, reward, success, tags, metadata) VALUES (?, 's', ?, ?, 1, '["a","b"]', '{"k":"v"}')`);
  const epEmb = db.prepare('INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)');
  const sk = db.prepare(`INSERT INTO skills (id, name, description, signature, code, success_rate, uses, avg_reward, avg_latency_ms, metadata) VALUES (?, ?, 'd', '{"inputs":{},"outputs":{}}', 'c', 0.9, 1, 0.5, 1, '{"k":"v"}')`);
  const skEmb = db.prepare('INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)');
  const pe = db.prepare('INSERT INTO pattern_embeddings (pattern_id, embedding) VALUES (?, ?)');
  vectors.forEach((v, i) => {
    ep.run(i + 1, `task ${i}`, rand());
    epEmb.run(i + 1, blob(v));
    sk.run(i + 1, `skill-${i}`);
    skEmb.run(i + 1, blob(v));
    pe.run(i + 1, blob(v));
  });
});
insert();

console.log(`N=${N} dim=${DIM} k=${K} runs=${RUNS} node=${process.version}`);

// Query cache disabled so every call scans
const reflexion = new ReflexionMemory(db as any, embedder, undefined, undefined, undefined, { enabled: false } as any);
await time('ReflexionMemory SQL fallback', () => reflexion.retrieveRelevant({ task: 'q', k: K }));

const skills = new SkillLibrary(db as any, embedder, undefined, undefined, { enabled: false } as any);
await time('SkillLibrary legacy scan', () => skills.retrieveSkills({ task: 'q', k: K, minSuccessRate: 0 }));

const wasm = new WASMVectorSearch(db, { enableWASM: false, enableSIMD: false });
await time('WASMVectorSearch.findKNN', () => wasm.findKNN(queries[qi++ % RUNS], K));

const rvf = new SqlJsRvfBackend({ dimension: DIM, metric: 'cosine' } as any);
await rvf.initialize();
await rvf.insertBatchAsync(vectors.map((v, i) => ({ id: `v${i}`, embedding: v, metadata: { i } })));
await time('SqlJsRvfBackend.searchAsync', () => rvf.searchAsync(queries[qi++ % RUNS], K));
rvf.close();
db.close();
