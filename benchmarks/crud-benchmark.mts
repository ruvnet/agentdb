/**
 * ADR-017 CRUD benchmark: get / list / update / delete over N episodes, patterns and skills.
 *
 *   npx tsx benchmarks/crud-benchmark.mts            # mock embedder: measures DB + index work only
 *   CRUD_BENCH_REAL=1 npx tsx benchmarks/crud-benchmark.mts   # real MiniLM: includes re-embed cost
 *   CRUD_BENCH_N=10000 ...                          # dataset size (default 10000)
 *
 * The headline comparison is an in-place update against what callers had to do before 3.3.0:
 * delete + insert, which also changes the record's id.
 */
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { ReflexionMemory } from '../src/controllers/ReflexionMemory.js';
import { ReasoningBank } from '../src/controllers/ReasoningBank.js';
import { SkillLibrary } from '../src/controllers/SkillLibrary.js';
import { EmbeddingService } from '../src/controllers/EmbeddingService.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const N = Number(process.env.CRUD_BENCH_N ?? 10_000);
const OPS = Number(process.env.CRUD_BENCH_OPS ?? 500);
const REAL = process.env.CRUD_BENCH_REAL === '1';

function pct(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

async function time(label: string, ops: number, fn: (i: number) => unknown | Promise<unknown>) {
  const t: number[] = [];
  for (let i = 0; i < ops; i++) {
    const t0 = performance.now();
    await fn(i);
    t.push(performance.now() - t0);
  }
  const row = { op: label, ops, p50_ms: +pct(t, 50).toFixed(4), p99_ms: +pct(t, 99).toFixed(4), mean_ms: +(t.reduce((a, b) => a + b, 0) / ops).toFixed(4) };
  console.log(JSON.stringify(row));
  return row;
}

const db = new Database(':memory:');
db.exec(fs.readFileSync(path.join(here, '../src/schemas/schema.sql'), 'utf-8'));
db.pragma('foreign_keys = OFF');
const embedder = REAL
  ? new EmbeddingService({ model: 'Xenova/all-MiniLM-L6-v2', dimension: 384, provider: 'transformers' })
  : new EmbeddingService({ model: 'mock-model', dimension: 384, provider: 'local' });
await embedder.initialize();
const reflexion = new ReflexionMemory(db, embedder);
const bank = new ReasoningBank(db, embedder);
const skills = new SkillLibrary(db, embedder);

console.log(JSON.stringify({ N, OPS, embedder: REAL ? 'MiniLM-L6-v2 (transformers.js)' : 'mock (hash)', node: process.version }));
const seedStart = performance.now();
const ids: number[] = [];
for (let i = 0; i < N; i++) {
  ids.push(await reflexion.storeEpisode({ sessionId: `s${i % 50}`, task: `task ${i} about subsystem ${i % 97}`, output: `out ${i}`, reward: 0.5, success: true, metadata: { i } }));
}
const pids: number[] = [];
const sids: number[] = [];
for (let i = 0; i < Math.min(N, 2000); i++) {
  pids.push(await bank.storePattern({ taskType: `type${i % 20}`, approach: `approach ${i}`, successRate: 0.5 }));
  sids.push(await skills.createSkill({ name: `skill-${i}`, description: `does thing ${i}`, successRate: 0.5 }));
}
console.log(JSON.stringify({ seeded_ms: Math.round(performance.now() - seedStart) }));
const pick = (arr: number[], i: number) => arr[(i * 7919) % arr.length];

const results = [];
results.push(await time('episode.get', OPS, i => reflexion.getEpisode(pick(ids, i))));
results.push(await time('episode.list(page=50)', OPS, i => reflexion.listEpisodes({ sessionId: `s${i % 50}`, limit: 50 })));
results.push(await time('episode.update(metadata only)', OPS, i => reflexion.updateEpisode(pick(ids, i), { metadata: { touched: i } })));
results.push(await time('episode.update(text, re-embed)', OPS, i => reflexion.updateEpisode(pick(ids, i), { task: `rewritten task ${i} v2` })));
// What callers did before: delete then insert (new id, full re-embed)
const victims = ids.slice(0, OPS);
results.push(await time('episode.delete+insert (pre-3.3 "update")', OPS, async i => {
  await reflexion.deleteEpisode(victims[i]);
  await reflexion.storeEpisode({ sessionId: 's0', task: `replacement ${i}`, reward: 0.5, success: true });
}));
results.push(await time('episode.delete', OPS, i => reflexion.deleteEpisode(ids[OPS + i])));
await reflexion.retrieveRelevant({ task: "warm", k: 5 }); // build the keyword index once
results.push(await time('hybrid search (no writes)', Math.min(OPS, 50), i => reflexion.retrieveRelevant({ task: `fresh ${i}`, k: 5 })));
// In-place edits patch the keyword index (Bm25Index.replace); a search right after must not pay a rebuild

results.push(await time('hybrid search after text update', Math.min(OPS, 50), async i => {
  await reflexion.updateEpisode(pick(ids, i + 1), { task: `fresh ${i}` });
  return reflexion.retrieveRelevant({ task: `fresh ${i}`, k: 5 });
}));
results.push(await time('pattern.update(approach, re-embed)', OPS, i => bank.updatePattern(pick(pids, i), { approach: `better approach ${i}` })));
results.push(await time('skill.update(description, re-embed)', OPS, i => skills.updateSkill(pick(sids, i), { description: `does it better ${i}` })));
results.push(await time('skill.delete', Math.min(OPS, sids.length), i => skills.deleteSkill(sids[i])));

const outDir = path.join(here, 'results');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `crud-${REAL ? 'real' : 'mock'}-${N}.json`), JSON.stringify({ N, OPS, real: REAL, results }, null, 2));
db.close();
