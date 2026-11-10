/** Run in an empty consumer directory after installing the sealed agentdb tarball. */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
const require = createRequire(resolve(process.cwd(), 'package.json'));
const { WeightPrograms } = await import(pathToFileURL(require.resolve('agentdb/controllers/WeightPrograms')).href);
const { FannModels } = await import(pathToFileURL(require.resolve('agentdb/controllers/FannModels')).href);
const { createDatabase } = await import(pathToFileURL(require.resolve('agentdb/db-fallback')).href);
const pkg = require('agentdb/package.json');
assert.equal(pkg.version, '3.9.0');
assert.equal(pkg.dependencies['ruv-fann'], '0.1.2');
assert.equal(pkg.dependencies['ruv-fann-wsp'], '0.1.0');
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
assert.equal(execFileSync(process.execPath, [cli, '--version'], { encoding: 'utf8' }).trim(), 'agentdb v3.9.0');
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
