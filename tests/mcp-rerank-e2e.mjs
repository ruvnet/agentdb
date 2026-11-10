// Real learned-reranker WASM training, held-out gates and MCP policy round trips on both SQL drivers.
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
    protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'rerank-e2e', version: '1' },
  });
}


// Explicit synthetic judgments only. These fixtures do not measure production relevance.
const groups = prefix => Array.from({ length: 8 }, (_, query) => ({ queryId: `${prefix}-${query}`, candidates: [
  { id: 'low', features: { similarity: 0.95, reward: 0, success: 0, recency: 0.5 }, relevance: 0 },
  { id: 'fair', features: { similarity: 0.8, reward: 1 / 3, success: 0, recency: 0.5 }, relevance: 1 },
  { id: 'good', features: { similarity: 0.65, reward: 2 / 3, success: 1, recency: 0.5 }, relevance: 2 },
  { id: 'best', features: { similarity: 0.5, reward: 1, success: 1, recency: 0.5 }, relevance: 3 }
] }));
const candidates = groups('sample')[0].candidates.map(({ id, features }) => ({ id, features }));
const baselineOrder = candidates.map(item => item.id);
const drivers = (process.env.AGENTDB_TEST_DRIVERS ?? 'sqljs,native').split(',');
assert.ok(drivers.length > 0 && drivers.every(driver => ['sqljs', 'native'].includes(driver)));
for (const driver of drivers) {
  const dir = mkdtempSync(join(tmpdir(), 'agentdb-rerank-e2e-'));
  const env = { AGENTDB_FORCE_SQLJS: driver === 'sqljs' ? '1' : '' };
  let s;
  const ok = async (name, args = {}) => { const result = await s.call(name, args); assert.equal(result.isError, false, result.text); return result.json(); };
  const search = async args => { const result = await s.call('agentdb_search', args); assert.equal(result.isError, false, result.text); return result.text; };
  const ids = text => [...text.matchAll(/\[ID: (\d+)\]/g)].map(match => match[1]);
  try {
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'learning' }, dir); await init(s);
    const capability = await ok('agentdb_capabilities');
    assert.equal(capability.sqlDriver.isNative, driver === 'native', `Expected actual ${driver} SQL backend`);
    assert.equal(capability.learnedReranking.defaultMode, 'shadow');
    assert.equal(capability.learnedReranking.weightSpaceProgramming.provider, null);
    assert.equal(capability.learnedReranking.weightSpaceProgramming.gradients, false);
    const names = (await s.request('tools/list', {})).result.tools.map(tool => tool.name);
    for (const action of ['create', 'train', 'evaluate', 'get', 'list', 'rank', 'inspect', 'patch']) assert.ok(names.includes(`agentdb_rerank_${action}`));
    for (const action of ['promote', 'rollback']) assert.ok(!names.includes(`agentdb_rerank_${action}`));
    // A separate model is authored entirely by 49 explicit weight replacements.
    // This path never calls train; the same held-out gates govern promotion.
    await ok('agentdb_rerank_create', { id: 'weight-space' });
    const seed = await ok('agentdb_rerank_inspect', { id: 'weight-space' });
    assert.equal(seed.source, 'seed'); assert.match(seed.baseHash, /^[a-f0-9]{64}$/);
    const edits = seed.model.layers.flatMap((layer, layerIndex) => [
      ...layer.weights.map((_value, index) => ({ layer: layerIndex, parameter: 'weights', index,
        value: (layerIndex === 0 && index === 1) || (layerIndex === 1 && index === 0) ? 1 : 0 })),
      ...layer.bias.map((_value, index) => ({ layer: layerIndex, parameter: 'bias', index, value: 0 }))
    ]);
    assert.equal(edits.length, 49);
    const directPatch = { schema: 'ruv-fann.weight-patch/1', baseHash: seed.baseHash, edits };
    const patched = await ok('agentdb_rerank_patch', { id: 'weight-space', patch: directPatch });
    assert.equal(patched.origin, 'weight-patch'); assert.equal(patched.parameterCount, 49);
    for (const key of ['initialLoss', 'finalLoss', 'epochs']) assert.equal(Object.hasOwn(patched, key), false);
    const prospective = await ok('agentdb_rerank_inspect', { id: 'weight-space', candidateId: patched.candidateId });
    assert.equal(prospective.source, 'candidate'); assert.equal(prospective.model.layers[0].weights[1], 1);
    assert.equal(prospective.model.layers[1].weights[0], 1);
    assert.notEqual(prospective.baseHash, seed.baseHash);
    for (const args of [
      { id: 'weight-space', patch: { ...directPatch, baseHash: '0'.repeat(64) } },
      { id: 'weight-space', patch: { ...directPatch, accepted: true } },
      { id: 'weight-space', patch: { ...directPatch, edits: [edits[0], edits[0]] } },
      { id: 'weight-space', patch: directPatch, evaluation: { accepted: true } }
    ]) assert.ok((await s.call('agentdb_rerank_patch', args)).isError);
    const weightEvaluation = await ok('agentdb_rerank_evaluate', { id: 'weight-space', candidateId: patched.candidateId, groups: groups('wsp-heldout') });
    assert.equal(weightEvaluation.accepted, true, weightEvaluation.reasons.join(', '));
    assert.equal(weightEvaluation.candidate.ndcg, 1);
    assert.equal((await ok('agentdb_rerank_rank', { id: 'weight-space', candidates, options: { mode: 'active' } })).available, false);
    assert.ok((await s.call('agentdb_rerank_promote', { id: 'weight-space', evaluationId: weightEvaluation.evaluationId, expectedRevision: 1 })).isError);
    const initial = await ok('agentdb_rerank_create', { id: 'memory-ranker' });
    assert.equal(initial.revision, 1); assert.equal(initial.modelHash, null);
    for (let index = 0; index < 4; index++) {
      const stored = await s.call('reflexion_store', { session_id: `rerank-${index}`, task: `deploy service example ${index}`, reward: index / 3, success: index >= 2 });
      assert.equal(stored.isError, false, stored.text);
    }
    const normal = await search({ query: 'deploy service', k: 3 });
    const allCosines = await search({ query: 'deploy service', k: 4, hybrid: false });
    assert.match(allCosines, /Similarity: -/, 'Fixture must include a negative-cosine result');
    const zeroThresholdShadow = await search({ query: 'deploy service', k: 4, hybrid: false, min_similarity: 0, neuralRerank: { id: 'memory-ranker' } });
    assert.deepEqual(ids(zeroThresholdShadow), ids(allCosines), 'MCP default/zero threshold must not remove negative-cosine shadow results');
    const unavailable = await search({ query: 'deploy service', k: 3, neuralRerank: { id: 'memory-ranker', mode: 'active' } });
    assert.deepEqual(ids(unavailable), ids(normal)); assert.match(unavailable, /active\/unavailable/);
    const trained = await ok('agentdb_rerank_train', { id: 'memory-ranker', groups: groups('train'), options: { epochs: 250, learningRate: 0.01 } });
    assert.ok(trained.finalLoss < trained.initialLoss); assert.equal(trained.parameterCount, 49);
    assert.ok((await s.call('agentdb_rerank_evaluate', { id: 'memory-ranker', candidateId: trained.candidateId, groups: groups('train') })).isError);
    const evaluated = await ok('agentdb_rerank_evaluate', { id: 'memory-ranker', candidateId: trained.candidateId, groups: groups('heldout') });
    assert.equal(evaluated.accepted, true, evaluated.reasons.join(', ')); assert.ok(evaluated.candidate.ndcg > evaluated.baseline.ndcg);
    const shadow = await ok('agentdb_rerank_rank', { id: 'memory-ranker', candidates, options: { candidateId: trained.candidateId } });
    assert.deepEqual(shadow.order, baselineOrder); assert.equal(shadow.proposedOrder[0], 'best');
    const shadowSearch = await search({ query: 'deploy service', k: 3, neuralRerank: { id: 'memory-ranker', candidateId: trained.candidateId } });
    assert.deepEqual(ids(shadowSearch), ids(normal)); assert.match(shadowSearch, /shadow\/scored/);
    for (const args of [
      { id: 'memory-ranker', candidates, options: { mode: 'active', candidateId: trained.candidateId } },
      { id: 'memory-ranker', candidates: [{ id: 'bad', features: { ...candidates[0].features, label: 3 } }] }
    ]) assert.ok((await s.call('agentdb_rerank_rank', args)).isError);
    assert.ok((await s.call('agentdb_rerank_promote', { id: 'memory-ranker', evaluationId: evaluated.evaluationId, expectedRevision: 1 })).isError);
    await s.stop(); s = undefined;

    // Stored evaluations are the authority and survive a process restart.
    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'admin' }, dir); await init(s);
    const reopened = await ok('agentdb_rerank_get', { id: 'memory-ranker' });
    assert.equal(reopened.evaluations[0].evaluationId, evaluated.evaluationId);
    assert.ok((await s.call('agentdb_rerank_promote', { id: 'memory-ranker', evaluationId: evaluated.evaluationId, expectedRevision: 1, evaluation: { accepted: true } })).isError);
    const directPromoted = await ok('agentdb_rerank_promote', { id: 'weight-space', evaluationId: weightEvaluation.evaluationId, expectedRevision: 1 });
    assert.equal(directPromoted.revision, 2); assert.equal(directPromoted.modelHash, patched.modelHash);
    const directRank = await ok('agentdb_rerank_rank', { id: 'weight-space', candidates, options: { mode: 'active' } });
    assert.deepEqual(directRank.order, ['best', 'good', 'fair', 'low']);
    const activeWeights = await ok('agentdb_rerank_inspect', { id: 'weight-space' });
    assert.equal(activeWeights.source, 'active'); assert.equal(activeWeights.baseHash, prospective.baseHash);
    assert.ok((await s.call('agentdb_rerank_patch', { id: 'weight-space', patch: directPatch })).isError, 'old seed hash must not patch the promoted checkpoint');
    const promoted = await ok('agentdb_rerank_promote', { id: 'memory-ranker', evaluationId: evaluated.evaluationId, expectedRevision: 1 });
    assert.equal(promoted.revision, 2); assert.equal(promoted.modelHash, trained.modelHash);
    const active = await ok('agentdb_rerank_rank', { id: 'memory-ranker', candidates, options: { mode: 'active' } });
    assert.equal(active.order[0], 'best'); assert.ok(active.scores.every(item => Number.isFinite(item.score)));
    const activeSearch = await search({ query: 'deploy service', k: 3, neuralRerank: { id: 'memory-ranker', mode: 'active' }, min_similarity: 0 });
    assert.match(activeSearch, /active\/scored/);
    assert.ok((await s.call('agentdb_search', { query: 'deploy service', neuralRerank: { id: 'memory-ranker', mode: 'active', candidateId: trained.candidateId } })).isError);
    assert.ok((await s.call('agentdb_rerank_promote', { id: 'memory-ranker', evaluationId: evaluated.evaluationId, expectedRevision: 1 })).isError);
    await s.stop(); s = undefined;

    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'readonly' }, dir); await init(s);
    const readable = (await s.request('tools/list', {})).result.tools.map(tool => tool.name);
    for (const action of ['get', 'list', 'rank', 'inspect']) assert.ok(readable.includes(`agentdb_rerank_${action}`));
    for (const action of ['create', 'train', 'evaluate', 'patch', 'promote', 'rollback']) {
      assert.ok(!readable.includes(`agentdb_rerank_${action}`));
      assert.ok((await s.call(`agentdb_rerank_${action}`, { id: 'memory-ranker' })).isError);
    }
    assert.equal((await ok('agentdb_rerank_get', { id: 'memory-ranker' })).revision, 2);
    assert.equal((await ok('agentdb_rerank_rank', { id: 'memory-ranker', candidates, options: { mode: 'active' } })).order[0], 'best');
    await s.stop(); s = undefined;

    s = startServer({ ...env, AGENTDB_MCP_PROFILE: 'admin' }, dir); await init(s);
    const directRestored = await ok('agentdb_rerank_rollback', { id: 'weight-space', targetRevision: 1, expectedRevision: 2 });
    assert.equal(directRestored.revision, 3); assert.equal(directRestored.modelHash, null);
    assert.equal((await ok('agentdb_rerank_inspect', { id: 'weight-space' })).baseHash, seed.baseHash);
    const restored = await ok('agentdb_rerank_rollback', { id: 'memory-ranker', targetRevision: 1, expectedRevision: 2 });
    assert.equal(restored.revision, 3); assert.equal(restored.modelHash, null);
    const baseline = await ok('agentdb_rerank_rank', { id: 'memory-ranker', candidates, options: { mode: 'active' } });
    assert.equal(baseline.available, false); assert.deepEqual(baseline.order, baselineOrder);
    assert.ok((await s.call('agentdb_rerank_rollback', { id: 'memory-ranker', targetRevision: 1, expectedRevision: 2 })).isError);
    const reset = await s.call('agentdb_init', { reset: true }); assert.equal(reset.isError, false, reset.text);
    assert.deepEqual(await ok('agentdb_rerank_list'), []);
    await ok('agentdb_rerank_create', { id: 'after-reset' });
    const inserted = await s.call('reflexion_store', { session_id: 'new', task: 'new deploy service', reward: 1, success: true });
    assert.equal(inserted.isError, false, inserted.text);
    const afterReset = await search({ query: 'new deploy service', k: 1, neuralRerank: { id: 'after-reset', mode: 'active' } });
    assert.match(afterReset, /active\/unavailable/);
    console.log(`PASS ${driver}: direct 49-parameter WSP patch + actual WASM candidate training, held-out ranking gate, strict schemas, shadow/active search, durable admin promotion/rollback, readonly policy and reset`);
  } finally { if (s) await s.stop(); rmSync(dir, { recursive: true, force: true }); }
}
