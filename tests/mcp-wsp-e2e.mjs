// Real weight-program MCP round trips over both SQLite drivers; no gradient training.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';

function startServer(env, dir) {
  const child = spawn(process.execPath, [process.env.AGENTDB_TEST_CLI ?? 'dist/src/cli/agentdb-cli.js', 'mcp', 'start'], {
    cwd: process.cwd(),
    env: { ...process.env, AGENTDB_PATH: join(dir, 'agentdb.db'), AGENTDB_DISABLE_TRANSFORMERS: '1', ...env },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let buf = '';
  let nextId = 1;
  const waiters = new Map();
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (c) => { stderr += c; });
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    buf += chunk;
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line); // throws on non-JSON stdout: protocol purity
      const w = waiters.get(msg.id);
      if (w) { waiters.delete(msg.id); w(msg); }
    }
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => reject(new Error(`timeout on ${method}\n${stderr}`)), 60_000);
    waiters.set(id, (m) => { clearTimeout(timer); resolve(m); });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
  const call = async (name, args = {}) => {
    const res = await request('tools/call', { name, arguments: args });
    const text = res.result?.content?.[0]?.text ?? '';
    return { isError: Boolean(res.result?.isError), text, json: () => JSON.parse(text) };
  };
  const stop = () => new Promise((resolve, reject) => {
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(new Error('shutdown timeout')); }, 10000);
    child.once('exit', code => { clearTimeout(timer); code === 0 ? resolve() : reject(new Error(`exit ${code}: ${stderr}`)); });
    child.kill('SIGTERM');
  });
  return { request, call, stop };
}

async function init(server) {
  await server.request('initialize', {
    protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'crud-e2e', version: '1' },
  });
}


const blank = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [0, 0, 0, 0], bias: [0, 0], activation: 'Threshold', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Threshold', steepness: 1 },
] };
const xor = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [1, 1, -1, -1], bias: [-0.5, 1.5], activation: 'Threshold', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [1, 1], bias: [-1.5], activation: 'Threshold', steepness: 1 },
] };
const dataset = { inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4 };
const policy = { maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0 };
const drivers = (process.env.AGENTDB_TEST_DRIVERS ?? 'sqljs,native').split(',');
assert.ok(drivers.length > 0 && drivers.every(driver => ['sqljs', 'native'].includes(driver)));
for (const driver of drivers) {
  const dir = mkdtempSync(join(tmpdir(), 'agentdb-wsp-e2e-'));
  const env = { AGENTDB_FORCE_SQLJS: driver === 'sqljs' ? '1' : '' };
  let s;
  const ok = async (name, args) => { const r = await s.call(name, args); assert.equal(r.isError, false, r.text); return r.json(); };
  try {
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'learning' }, dir); await init(s);
    const capabilities = await ok('agentdb_capabilities', {});
    assert.equal(capabilities.sqlDriver.isNative, driver === 'native', `Expected actual ${driver} SQL driver`);
    assert.equal(capabilities.weightPrograms.gradients, false);
    const names = (await s.request('tools/list', {})).result.tools.map(t => t.name);
    for (const action of ['create', 'get', 'list', 'propose', 'promote']) assert.ok(names.includes(`agentdb_wsp_${action}`));
    assert.ok(!names.includes('agentdb_wsp_rollback'));
    const initial = await ok('agentdb_wsp_create', { id: 'xor', model: blank });
    const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash, edits: xor.layers.flatMap((layer, index) => [
      ...layer.weights.map((value, location) => ({ layer: index, parameter: 'weights', index: location, value })),
      ...layer.bias.map((value, location) => ({ layer: index, parameter: 'bias', index: location, value })),
    ]) };
    for (const args of [
      { id: 'xor', patch: { ...patch, extra: true }, dataset, policy },
      { id: 'xor', patch, dataset: { ...dataset, trusted: true }, policy },
      { id: 'xor', patch, dataset, policy: { ...policy, accepted: true } },
      { id: 'xor', patch: { ...patch, edits: [patch.edits[0], patch.edits[0]] }, dataset, policy },
    ]) assert.ok((await s.call('agentdb_wsp_propose', args)).isError);
    assert.ok((await s.call('agentdb_wsp_list', JSON.parse('{"constructor":{}}'))).isError);
    const proposal = await ok('agentdb_wsp_propose', { id: 'xor', patch, dataset, policy });
    assert.equal(proposal.evaluation.accepted, true);
    assert.equal(proposal.evaluation.candidate.mse, 0);
    assert.ok(proposal.evaluation.baseline.mse > 0);
    for (const args of [
      { id: 'xor', proposalId: 'forged', expectedRevision: 1 },
      { id: 'xor', proposalId: proposal.proposalId, expectedRevision: 1, evaluation: { accepted: true } },
      { id: 'xor', proposalId: proposal.evaluation, expectedRevision: 1 },
    ]) assert.ok((await s.call('agentdb_wsp_promote', args)).isError);
    await s.stop(); s = undefined;
    // Promotion authority and model survive a process restart, including outstanding proposals.
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'learning' }, dir); await init(s);
    const reopened = await ok('agentdb_wsp_get', { id: 'xor' });
    assert.equal(reopened.proposals[0].id, proposal.proposalId);
    const promoted = await ok('agentdb_wsp_promote', { id: 'xor', proposalId: proposal.proposalId, expectedRevision: 1 });
    assert.equal(promoted.revision, 2); assert.deepEqual(promoted.model, xor);
    assert.equal(promoted.history[0].evaluation.candidateHash, proposal.evaluation.candidateHash);
    assert.ok((await s.call('agentdb_wsp_promote', { id: 'xor', proposalId: proposal.proposalId, expectedRevision: 1 })).isError);
    // A rejected proposal remains nonpromotable even with an attached forged receipt.
    const rejected = await ok('agentdb_wsp_propose', {
      id: 'xor', patch: { schema: patch.schema, baseHash: promoted.hash, edits: [{ layer: 1, parameter: 'weights', index: 1, value: 0 }] }, dataset, policy,
    });
    assert.equal(rejected.evaluation.accepted, false);
    assert.ok((await s.call('agentdb_wsp_promote', { id: 'xor', proposalId: rejected.proposalId, expectedRevision: 2 })).isError);
    assert.ok((await s.call('agentdb_wsp_rollback', { id: 'xor', targetRevision: 1, expectedRevision: 2 })).isError);
    await s.stop(); s = undefined;
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'readonly' }, dir); await init(s);
    assert.equal((await ok('agentdb_wsp_get', { id: 'xor' })).hash, promoted.hash);
    const listed = await ok('agentdb_wsp_list', {}); assert.equal(listed.length, 1); assert.equal(listed[0].model, undefined);
    for (const action of ['create', 'propose', 'promote', 'rollback']) assert.ok((await s.call(`agentdb_wsp_${action}`, { id: 'xor' })).isError);
    await s.stop(); s = undefined;
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'admin' }, dir); await init(s);
    const rollback = await ok('agentdb_wsp_rollback', { id: 'xor', targetRevision: 1, expectedRevision: 2 });
    assert.equal(rollback.revision, 3); assert.deepEqual(rollback.model, blank); assert.equal(rollback.hash, initial.hash);
    assert.ok((await s.call('agentdb_wsp_rollback', { id: 'xor', targetRevision: 1, expectedRevision: 2 })).isError);
    await ok('agentdb_neural_create', { id: 'independent', model: xor });
    assert.deepEqual((await ok('agentdb_neural_predict', { id: 'independent', inputs: dataset.inputs, rows: 4 })).outputs, dataset.targets);
    const reset = await s.call('agentdb_init', { reset: true }); assert.equal(reset.isError, false, reset.text);
    assert.deepEqual(await ok('agentdb_wsp_list', {}), []);
    await ok('agentdb_wsp_create', { id: 'after-reset', model: blank });
    console.log(`PASS ${driver}: real WASM XOR construction, durable proposal/promotion/rollback, forged receipt rejection, strict policies and reset`);
  } finally { if (s) await s.stop(); rmSync(dir, { recursive: true, force: true }); }
}
