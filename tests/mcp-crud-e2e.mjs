// ADR-017 end-to-end: drive every CRUD tool through the real MCP server over stdio.
// Runs once on sql.js (BEGIN/COMMIT path) and once on better-sqlite3 (driver transaction path),
// and checks that the default `learning` profile refuses the destructive tools.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';

function startServer(env) {
  const dir = mkdtempSync(join(tmpdir(), 'agentdb-crud-e2e-'));
  const child = spawn(process.execPath, ['dist/src/cli/agentdb-cli.js', 'mcp', 'start'], {
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
  const stop = () => { child.kill('SIGTERM'); rmSync(dir, { recursive: true, force: true }); };
  return { request, call, stop };
}

async function init(server) {
  await server.request('initialize', {
    protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'crud-e2e', version: '1' },
  });
}

async function runAdmin(label, env) {
  const s = startServer({ AGENTDB_MCP_PROFILE: 'admin', ...env });
  try {
    await init(s);
    const tools = (await s.request('tools/list', {})).result.tools.map((t) => t.name);
    for (const t of ['agentdb_get', 'agentdb_list', 'agentdb_update', 'agentdb_pattern_get', 'agentdb_pattern_list',
      'agentdb_pattern_update', 'agentdb_pattern_delete', 'skill_get', 'skill_list', 'skill_update', 'skill_delete']) {
      assert.ok(tools.includes(t), `${label}: tools/list is missing ${t}`);
    }

    const initRes = await s.call('agentdb_init', {});
    assert.equal(initRes.isError, false, initRes.text);

    // Episodes / vector records
    const ins = await s.call('agentdb_insert', { text: 'zorblax quarantine runbook', metadata: { owner: 'ops', tmp: 1 }, tags: ['ops'] });
    assert.equal(ins.isError, false, ins.text);
    const id = Number(/ID: (\d+)/.exec(ins.text)[1]);
    assert.equal(s.call && (await s.call('agentdb_get', { id: 999999 })).text, 'null');
    const got = (await s.call('agentdb_get', { id })).json();
    assert.equal(got.task, 'zorblax quarantine runbook');
    assert.equal(got.embedding, undefined);
    assert.equal((await s.call('agentdb_get', { id, include_embedding: true })).json().embedding.length, 384);

    const metaOnly = (await s.call('agentdb_update', { id, metadata: { tmp: null, team: 'sre' }, tags: ['ops', 'runbook'] })).json();
    assert.equal(metaOnly.updated, true);
    assert.equal(metaOnly.reembedded, false);
    assert.deepEqual(metaOnly.episode.metadata, { owner: 'ops', team: 'sre' });

    const textUpd = (await s.call('agentdb_update', { id, text: 'quuxifier calibration runbook' })).json();
    assert.equal(textUpd.reembedded, true);
    const search = await s.call('agentdb_search', { query: 'quuxifier', k: 1 });
    assert.match(search.text, /quuxifier calibration runbook/, `${label}: search should see the updated text`);

    for (let i = 0; i < 4; i++) await s.call('agentdb_insert', { text: `filler ${i}`, session_id: 'other' });
    const page = (await s.call('agentdb_list', { session_id: 'other', limit: 2 })).json();
    assert.equal(page.total, 4);
    assert.equal(page.items.length, 2);

    // Security at the boundary
    assert.ok((await s.call('agentdb_update', { id, metadata: JSON.parse('{"__proto__":{"x":1}}') })).isError);
    assert.ok((await s.call('agentdb_update', { id, session_id: 'x' })).isError, 'unknown args must be rejected');
    assert.ok((await s.call('agentdb_get', { id: '12-notes' })).isError);
    assert.ok((await s.call('agentdb_list', { limit: 100000 })).isError);
    assert.ok((await s.call('agentdb_insert', { text: '' })).isError);
    assert.ok((await s.call('agentdb_insert', { text: 'x', metadata: 'not-an-object' })).isError);
    const leak = await s.call('agentdb_update', { id: -1 });
    assert.ok(leak.isError && !/\n\s+at /.test(leak.text), 'errors must not carry a stack trace');

    const del = await s.call('agentdb_delete', { id });
    assert.match(del.text, /Deleted: 1/);
    assert.equal((await s.call('agentdb_get', { id })).text, 'null');

    // Patterns
    const ps = await s.call('agentdb_pattern_store', { taskType: 'debug', approach: 'bisect the regression', successRate: 0.8 });
    assert.equal(ps.isError, false, ps.text);
    const pid = (await s.call('agentdb_pattern_list', {})).json().items[0].id;
    const pu = (await s.call('agentdb_pattern_update', { pattern_id: pid, approach: 'git bisect run with a test' })).json();
    assert.equal(pu.reembedded, true);
    assert.equal(pu.pattern.successRate, 0.8);
    assert.equal((await s.call('agentdb_pattern_get', { pattern_id: pid })).json().approach, 'git bisect run with a test');
    assert.ok((await s.call('agentdb_pattern_update', { pattern_id: pid, successRate: 1 })).isError);
    assert.equal((await s.call('agentdb_pattern_delete', { pattern_id: pid })).json().deleted, true);
    assert.equal((await s.call('agentdb_pattern_get', { pattern_id: pid })).text, 'null');

    // Skills
    assert.equal((await s.call('skill_create', { name: 'deploy', description: 'ship it', code: 'make deploy' })).isError, false);
    await s.call('skill_create', { name: 'rollback', description: 'undo it' });
    const sk = (await s.call('skill_list', {})).json();
    assert.equal(sk.total, 2);
    const deploy = sk.items.find((x) => x.name === 'deploy');
    assert.ok((await s.call('skill_update', { skill_id: deploy.id, name: 'rollback' })).isError, 'rename clash');
    const su = (await s.call('skill_update', { skill_id: deploy.id, description: 'blue/green deploy' })).json();
    assert.equal(su.reembedded, true);
    assert.equal((await s.call('skill_get', { skill_id: deploy.id })).json().description, 'blue/green deploy');
    assert.equal((await s.call('skill_delete', { skill_id: deploy.id })).json().deleted, true);
    assert.equal((await s.call('skill_delete', { skill_id: deploy.id })).json().deleted, false);
    console.log(`ok - ${label}: CRUD round trip over stdio`);
  } finally {
    s.stop();
  }
}

async function runDefaultProfile() {
  const s = startServer({ AGENTDB_FORCE_SQLJS: '1' });
  try {
    await init(s);
    const tools = (await s.request('tools/list', {})).result.tools.map((t) => t.name);
    assert.ok(tools.includes('agentdb_update') && tools.includes('skill_get'), 'learning profile keeps reads and updates');
    for (const t of ['agentdb_pattern_delete', 'skill_delete', 'agentdb_delete']) {
      assert.ok(!tools.includes(t), `learning profile must not list ${t}`);
      assert.ok((await s.call(t, { pattern_id: 1, skill_id: 1, id: 1 })).isError, `learning profile must refuse ${t}`);
    }
    console.log('ok - default learning profile refuses deletes, allows get/list/update');
  } finally {
    s.stop();
  }
}

await runAdmin('sql.js', { AGENTDB_FORCE_SQLJS: '1' });
await runAdmin('better-sqlite3', {});
await runDefaultProfile();
process.exit(0);
