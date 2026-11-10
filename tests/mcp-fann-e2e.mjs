// Real neural MCP round trips, including persistence, policies and actual SQL driver checks.
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


const model = { schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear', steepness: 1 },
] };
const drivers = (process.env.AGENTDB_TEST_DRIVERS ?? 'sqljs,native').split(',');
assert.ok(drivers.length > 0 && drivers.every(driver => ['sqljs', 'native'].includes(driver)));
for (const driver of drivers) {
  const dir = mkdtempSync(join(tmpdir(), 'agentdb-fann-e2e-'));
  const env = { AGENTDB_FORCE_SQLJS: driver === 'sqljs' ? '1' : '' };
  let s;
  const ok = async (name, args) => { const r = await s.call(name, args); assert.equal(r.isError, false, r.text); return r.json(); };
  try {
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'learning' }, dir); await init(s);
    const capabilities = await ok('agentdb_capabilities', {});
    assert.equal(capabilities.sqlDriver.isNative, driver === 'native', `Expected actual ${driver} SQL backend`);
    const names = (await s.request('tools/list', {})).result.tools.map(t => t.name);
    assert.ok(names.includes('agentdb_neural_train'));
    assert.ok(!names.includes('agentdb_neural_delete'));
    for (const invalid of [{ ...model, extra: true }, { ...model, layers: [{ ...model.layers[0], extra: true }] }]) {
      assert.ok((await s.call('agentdb_neural_create', { id: 'invalid', model: invalid })).isError);
    }
    assert.deepEqual(await ok('agentdb_neural_list', {}), []);
    await ok('agentdb_neural_create', { id: 'linear', model });
    const before = await ok('agentdb_neural_predict', { id: 'linear', inputs: [1, 2], rows: 2 });
    assert.deepEqual(before.outputs, [0, 0]);
    const trained = await ok('agentdb_neural_train', { id: 'linear', inputs: [0, 1, 2], targets: [1, 3, 5], rows: 3, epochs: 300, learningRate: 0.05, optimizer: 'sgd' });
    assert.ok(trained.finalLoss < trained.initialLoss * 0.01);
    const after = await ok('agentdb_neural_predict', { id: 'linear', inputs: [1, 2], rows: 2 });
    assert.ok(Math.abs(after.outputs[0] - 3) < 0.05);
    assert.ok(after.revision > before.revision);
    assert.ok((await s.call('agentdb_neural_delete', { id: 'linear' })).isError);
    for (const args of [{ id: 'linear', inputs: [1], rows: 129 }, { id: 'linear', inputs: ['bad'] }, { id: 'linear', inputs: [1], extra: true }]) {
      assert.ok((await s.call('agentdb_neural_predict', args)).isError);
    }
    assert.ok((await s.call('agentdb_neural_list', JSON.parse('{"constructor":1}'))).isError);
    const metadata = await ok('agentdb_neural_list', {});
    assert.equal(metadata.length, 1); assert.equal(metadata[0].model, undefined);
    await s.stop(); s = undefined;
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'readonly' }, dir); await init(s);
    const reopened = await ok('agentdb_neural_predict', { id: 'linear', inputs: [1, 2], rows: 2 });
    assert.deepEqual(reopened.outputs, after.outputs);
    assert.equal(reopened.revision, after.revision);
    for (const action of ['create', 'train', 'delete']) assert.ok((await s.call(`agentdb_neural_${action}`, { id: 'linear', model })).isError);
    await s.stop(); s = undefined;
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'admin' }, dir); await init(s);
    assert.equal((await ok('agentdb_neural_delete', { id: 'linear' })).deleted, true);
    assert.equal(await ok('agentdb_neural_get', { id: 'linear' }), null);
    await ok('agentdb_neural_create', { id: 'reset', model });
    const reset = await s.call('agentdb_init', { reset: true }); assert.equal(reset.isError, false, reset.text);
    assert.deepEqual(await ok('agentdb_neural_list', {}), []);
    await ok('agentdb_neural_create', { id: 'after-reset', model });
    console.log(`PASS ${driver}: real WASM training, cached inference, durable reopen, policies, validation and reset`);
  } finally { if (s) await s.stop(); rmSync(dir, { recursive: true, force: true }); }
}
