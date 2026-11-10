/** Run in an empty consumer directory after installing the sealed agentdb tarball. */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
const require = createRequire(resolve(process.cwd(), 'package.json'));
const { WeightPrograms } = await import(pathToFileURL(require.resolve('agentdb/controllers/WeightPrograms')).href);
const { FannModels } = await import(pathToFileURL(require.resolve('agentdb/controllers/FannModels')).href);
const { NeuralReranker } = await import(pathToFileURL(require.resolve('agentdb/controllers/NeuralReranker')).href);
const { createDatabase } = await import(pathToFileURL(require.resolve('agentdb/db-fallback')).href);
const pkg = require('agentdb/package.json');
assert.equal(pkg.version, '3.10.0');
assert.equal(pkg.dependencies['ruv-fann'], '0.1.2');
assert.equal(pkg.dependencies['ruv-fann-wsp'], '0.1.0');
const typeFixture = join(process.cwd(), 'neural-reranker-types.mts');
writeFileSync(typeFixture, `
import { NeuralReranker, RERANK_FEATURE_SCHEMA, episodeRerankFeatures,
  type RerankFeatures, type RerankGroup, type RankingPolicy, type TrainingCandidate,
  type RankingEvaluation, type RankResult, type RerankerRecord, type NeuralRerankerOptions,
  type WeightInspection, type WeightPatch } from 'agentdb/controllers/NeuralReranker';
declare const database: ConstructorParameters<typeof NeuralReranker>[0];
const options: NeuralRerankerOptions = { readonly: true, simd: false };
const reranker = new NeuralReranker(database, options);
const features: RerankFeatures = episodeRerankFeatures({ similarity: 0.4, reward: 0.7, success: true, ts: 1 }, 2);
const groups: RerankGroup[] = [{ queryId: 'q1', candidates: [
  { id: 'a', features, relevance: 0 },
  { id: 'b', features: { similarity: 0.5, reward: 1, success: 1, recency: 0.9 }, relevance: 3 },
] }];
const policy: RankingPolicy = { k: 2, minNdcgImprovement: 0.01, maxMrrRegression: 0, maxPerQueryNdcgRegression: 0.1 };
const created: RerankerRecord = reranker.create('reranker');
const candidate: Promise<TrainingCandidate> = reranker.train('reranker', groups);
const evaluated: Promise<RankingEvaluation> = reranker.evaluate('reranker', 'candidate', groups, policy);
const ranked: Promise<RankResult> = reranker.rank('reranker', [{ id: 'a', features }], { mode: 'shadow' });
const found: RerankerRecord | null = reranker.get('reranker');
const inspected: Promise<WeightInspection> = reranker.inspectWeights('reranker');
const patch: WeightPatch = { schema: 'ruv-fann.weight-patch/1', baseHash: '0'.repeat(64), edits: [{ layer: 1, parameter: 'bias', index: 0, value: 0 }] };
const patched: Promise<TrainingCandidate> = reranker.proposePatch('reranker', patch);
if (RERANK_FEATURE_SCHEMA !== 'agentdb.rerank-features/1') throw new Error('feature schema export');
void created; void candidate; void evaluated; void ranked; void found; void inspected; void patched;
`);
const tsc = fileURLToPath(new URL('../node_modules/typescript/bin/tsc', import.meta.url));
execFileSync(process.execPath, [tsc, '--noEmit', '--strict', '--target', 'ES2022', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--lib', 'ESNext,DOM,ESNext.Disposable', '--typeRoots', fileURLToPath(new URL('../node_modules/@types', import.meta.url)), '--types', 'node', typeFixture], { cwd: process.cwd(), stdio: 'inherit' });
process.env.AGENTDB_FORCE_SQLJS = '1';
const dir = mkdtempSync(join(tmpdir(), 'agentdb-packed-fann-'));
const path = join(dir, 'models.db');
let db, models;
try {
  db = await createDatabase(path); models = new FannModels(db);
  await models.create('linear', { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 },
  ] });
  const result = await models.train('linear', { inputs: new Float64Array([0, 1, 2]), targets: new Float64Array([1, 3, 5]), rows: 3, epochs: 300, learningRate: 0.05, optimizer: 'sgd' });
  assert.ok(result.finalLoss < result.initialLoss * 0.01);
  const prediction = await models.predict('linear', new Float64Array([1, 2]), 2);
  assert.ok(Math.abs(prediction.outputs[0] - 3) < 0.05);
  assert.ok(Math.abs(prediction.outputs[1] - 5) < 0.05);
  models.dispose(); db.close();
  db = await createDatabase(path); models = new FannModels(db);
  const reopened = await models.predict('linear', new Float64Array([1, 2]), 2);
  assert.deepEqual(reopened.outputs, prediction.outputs);
  assert.equal(reopened.revision, prediction.revision);
  assert.equal(models.delete('linear'), true);
  console.log(JSON.stringify({ status: 'pass', package: `agentdb@${pkg.version}`, tests: 8, backend: result.backend, initialLoss: result.initialLoss, finalLoss: result.finalLoss }));
} finally { models?.dispose(); db?.close(); rmSync(dir, { recursive: true, force: true }); }

// Exercise the installed neural reranker through actual WASM training/inference, isolated holdout evaluation,
// explicit promotion, active ranking, durable reopen and rollback. This fixture is independent of the CI benchmark.
const rerankDir = mkdtempSync(join(tmpdir(), 'agentdb-packed-rerank-'));
const rerankDbPath = join(rerankDir, 'reranker.db');
const makeGroups = (prefix, count) => Array.from({ length: count }, (_, i) => ({
  queryId: `${prefix}-query-${i}`,
  candidates: [
    { id: `${prefix}-${i}-baseline-first`, features: { similarity: 0.96 - i * 0.001, reward: 0, success: 0, recency: 0.1 }, relevance: 0 },
    { id: `${prefix}-${i}-partial`, features: { similarity: 0.62, reward: 0.5, success: 0.5, recency: 0.5 }, relevance: 1 },
    { id: `${prefix}-${i}-relevant`, features: { similarity: 0.18 + i * 0.001, reward: 1, success: 1, recency: 0.9 }, relevance: 3 },
  ],
}));
const trainGroups = makeGroups('train', 16);
const heldoutGroups = makeGroups('holdout', 6);
const rankCandidates = group => group.candidates.map(({ id, features }) => ({ id, features }));
let rerankDb, reranker;
try {
  rerankDb = await createDatabase(rerankDbPath);
  reranker = new NeuralReranker(rerankDb);
  assert.equal(reranker.create('episodes').revision, 1);
  const training = await reranker.train('episodes', trainGroups, { epochs: 250, learningRate: 0.01 });
  const evaluation = await reranker.evaluate('episodes', training.candidateId, heldoutGroups, {
    k: 3, minNdcgImprovement: 0.01, maxMrrRegression: 0, maxPerQueryNdcgRegression: 0.1,
  });
  assert.equal(evaluation.accepted, true);
  assert.ok(evaluation.candidate.ndcg > evaluation.baseline.ndcg);
  const shadow = await reranker.rank('episodes', rankCandidates(heldoutGroups[0]), { mode: 'shadow', candidateId: training.candidateId });
  assert.equal(shadow.available, true);
  assert.deepEqual(shadow.order, rankCandidates(heldoutGroups[0]).map(row => row.id));
  assert.notDeepEqual(shadow.proposedOrder, shadow.order);
  reranker.dispose(); rerankDb.close(); reranker = undefined; rerankDb = undefined;

  rerankDb = await createDatabase(rerankDbPath);
  reranker = new NeuralReranker(rerankDb);
  const durable = reranker.get('episodes');
  assert.ok(durable);
  assert.ok(durable.candidates.some(row => row.candidateId === training.candidateId && row.modelHash === training.modelHash));
  assert.ok(durable.evaluations.some(row => row.evaluationId === evaluation.evaluationId && row.accepted));
  const promoted = await reranker.promote('episodes', evaluation.evaluationId, 1);
  assert.equal(promoted.revision, 2);
  const active = await reranker.rank('episodes', rankCandidates(heldoutGroups[0]), { mode: 'active' });
  assert.equal(active.available, true);
  assert.equal(active.revision, 2);
  assert.equal(active.order[0], heldoutGroups[0].candidates[2].id);
  const restored = reranker.rollback('episodes', 1, 2);
  assert.equal(restored.revision, 3);
  assert.equal(restored.model, null);
  const baselineAgain = await reranker.rank('episodes', rankCandidates(heldoutGroups[0]), { mode: 'active' });
  assert.equal(baselineAgain.available, false);
  assert.deepEqual(baselineAgain.order, rankCandidates(heldoutGroups[0]).map(row => row.id));
  reranker.dispose(); rerankDb.close(); reranker = undefined; rerankDb = undefined;

  rerankDb = await createDatabase(rerankDbPath);
  reranker = new NeuralReranker(rerankDb);
  assert.equal(reranker.get('episodes').revision, 3);
  assert.equal(reranker.get('episodes').history[0].operation, 'rollback');
  console.log(JSON.stringify({ status: 'pass', controller: 'NeuralReranker', trainQueries: trainGroups.length, heldoutQueries: heldoutGroups.length,
    trainingBackend: training.backend, evaluationBackend: evaluation.backend, baselineNdcg: evaluation.baseline.ndcg,
    candidateNdcg: evaluation.candidate.ndcg, promotedRevision: promoted.revision, rollbackRevision: restored.revision }));
} finally { reranker?.dispose(); rerankDb?.close(); rmSync(rerankDir, { recursive: true, force: true }); }

// A direct 49-parameter WSP edit maps reward through Tanh hidden unit 0 to the linear output.
// It has no training-query provenance and must still pass independent held-out evaluation.
const patchDir = mkdtempSync(join(tmpdir(), 'agentdb-packed-rerank-patch-'));
let patchDb, patchController;
try {
  patchDb = await createDatabase(join(patchDir, 'reranker-patch.db'));
  patchController = new NeuralReranker(patchDb);
  const record = patchController.create('gradient-free');
  const seed = await patchController.inspectWeights('gradient-free');
  assert.equal(seed.source, 'seed');
  assert.equal(seed.revision, record.revision);
  const edits = [
    ...Array.from({ length: 32 }, (_, index) => ({ layer: 0, parameter: 'weights', index, value: index === 1 ? 2 : 0 })),
    ...Array.from({ length: 8 }, (_, index) => ({ layer: 0, parameter: 'bias', index, value: 0 })),
    ...Array.from({ length: 8 }, (_, index) => ({ layer: 1, parameter: 'weights', index, value: index === 0 ? 1 : 0 })),
    { layer: 1, parameter: 'bias', index: 0, value: 0 },
  ];
  assert.equal(edits.length, 49);
  const patchCandidate = await patchController.proposePatch('gradient-free', {
    schema: 'ruv-fann.weight-patch/1', baseHash: seed.baseHash, edits,
  });
  assert.equal(patchCandidate.origin, 'weight-patch');
  assert.equal(patchCandidate.baseHash, seed.baseHash);
  assert.equal(patchCandidate.trainingQueryIds.length, 0);
  assert.equal(Object.hasOwn(patchCandidate, 'epochs'), false);
  const patchGroups = Array.from({ length: 6 }, (_, i) => ({ queryId: `patch-holdout-${i}`, candidates: [
    { id: `patch-${i}-low`, features: { similarity: 0.5, reward: 0, success: 0, recency: 0.5 }, relevance: 0 },
    { id: `patch-${i}-mid`, features: { similarity: 0.5, reward: 0.5, success: 0, recency: 0.5 }, relevance: 1 },
    { id: `patch-${i}-high`, features: { similarity: 0.5, reward: 1, success: 0, recency: 0.5 }, relevance: 3 },
  ] }));
  const patchEvaluation = await patchController.evaluate('gradient-free', patchCandidate.candidateId, patchGroups, {
    k: 3, minNdcgImprovement: 0.01, maxMrrRegression: 0, maxPerQueryNdcgRegression: 0,
  });
  assert.equal(patchEvaluation.accepted, true);
  assert.equal(patchEvaluation.perQuery.length, 6);
  assert.ok(patchEvaluation.candidate.ndcg > patchEvaluation.baseline.ndcg);
  const promotedPatch = await patchController.promote('gradient-free', patchEvaluation.evaluationId, record.revision);
  const patchedRank = await patchController.rank('gradient-free', patchGroups[0].candidates.map(({ id, features }) => ({ id, features })), { mode: 'active' });
  assert.equal(patchedRank.available, true);
  assert.equal(patchedRank.order[0], 'patch-0-high');
  assert.equal(patchedRank.revision, promotedPatch.revision);
  console.log(JSON.stringify({ status: 'pass', controller: 'NeuralReranker', origin: patchCandidate.origin,
    parameterEdits: edits.length, heldoutQueries: patchGroups.length, baselineNdcg: patchEvaluation.baseline.ndcg,
    candidateNdcg: patchEvaluation.candidate.ndcg, promotedRevision: promotedPatch.revision, topRanked: patchedRank.order[0] }));
} finally { patchController?.dispose(); patchDb?.close(); rmSync(patchDir, { recursive: true, force: true }); }

const wspDir = mkdtempSync(join(tmpdir(), 'agentdb-packed-wsp-'));
let programDb, programs;
try {
  programDb = await createDatabase(join(wspDir, 'programs.db')); programs = new WeightPrograms(programDb);
  const initial = await programs.create('linear', { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 1, outputs: 1, weights: [0], activation: 'Linear', steepness: 1 },
  ] });
  const proposed = await programs.propose('linear', {
    schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash,
    edits: [{ layer: 0, parameter: 'weights', index: 0, value: 2 }],
  }, { inputs: [1, 2], targets: [2, 4], rows: 2 }, { maxMSE: 0, maxRegressionMSE: 0 });
  assert.equal(proposed.evaluation.accepted, true);
  programs.dispose(); programDb.close();
  programDb = await createDatabase(join(wspDir, 'programs.db')); programs = new WeightPrograms(programDb);
  assert.equal(programs.get('linear').proposals[0].id, proposed.proposalId);
  const promoted = await programs.promote('linear', proposed.proposalId, 1);
  assert.equal(promoted.revision, 2); assert.deepEqual(promoted.model.layers[0].weights, [2]);
  assert.equal(promoted.history[0].evaluation.candidateHash, proposed.evaluation.candidateHash);
  const restored = programs.rollback('linear', 1, 2);
  assert.equal(restored.revision, 3); assert.deepEqual(restored.model, initial.model);
  console.log('PASS: installed WeightPrograms export, real WASM evaluation, durable outstanding proposal, promotion and rollback.');
} finally { programs?.dispose(); programDb?.close(); rmSync(wspDir, { recursive: true, force: true }); }

const cli = require.resolve('agentdb/cli');
assert.equal(execFileSync(process.execPath, [cli, '--version'], { encoding: 'utf8' }).trim(), 'agentdb v3.10.0');
execFileSync(process.execPath, [fileURLToPath(new URL('../tests/mcp-fann-e2e.mjs', import.meta.url))], {
  cwd: process.cwd(), stdio: 'inherit', timeout: 120000,
  env: { ...process.env, AGENTDB_TEST_CLI: cli, AGENTDB_TEST_DRIVERS: 'sqljs' },
});
console.log('PASS: installed CLI version and real MCP training/persistence/policies.');

execFileSync(process.execPath, [fileURLToPath(new URL('../tests/mcp-wsp-e2e.mjs', import.meta.url))], {
  cwd: process.cwd(), stdio: 'inherit', timeout: 120000,
  env: { ...process.env, AGENTDB_TEST_CLI: cli, AGENTDB_TEST_DRIVERS: 'sqljs' },
});
console.log('PASS: installed CLI durable weight-program MCP proposals, promotion, rollback and policies.');

execFileSync(process.execPath, [fileURLToPath(new URL('../tests/mcp-rerank-e2e.mjs', import.meta.url))], {
  cwd: process.cwd(), stdio: 'inherit', timeout: 120000,
  env: { ...process.env, AGENTDB_TEST_CLI: cli, AGENTDB_TEST_DRIVERS: 'sqljs' },
});
console.log('PASS: installed CLI neural reranker training, held-out evaluation, promotion, active ranking, rollback and policies.');
