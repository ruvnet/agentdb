import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const directory = mkdtempSync(join(tmpdir(), 'agentdb-mcp-stdio-'));
const child = spawn(process.execPath, ['dist/src/cli/agentdb-cli.js', 'mcp', 'start'], {
  cwd: process.cwd(),
  detached: process.platform !== 'win32',
  env: {
    ...process.env,
    AGENTDB_PATH: join(directory, 'agentdb.db'),
    AGENTDB_FORCE_SQLJS: '1',
    AGENTDB_DISABLE_TRANSFORMERS: '1',
  },
  stdio: ['pipe', 'pipe', 'pipe'],
});

let stdoutBuffer = '';
let stderr = '';
const messages = [];
const waiters = new Map();

function fail(error) {
  for (const waiter of waiters.values()) waiter.reject(error);
  waiters.clear();
}

child.stderr.setEncoding('utf8');
child.stderr.on('data', (chunk) => { stderr += chunk; });
child.stdout.setEncoding('utf8');
child.stdout.on('data', (chunk) => {
  stdoutBuffer += chunk;
  while (stdoutBuffer.includes('\n')) {
    const newline = stdoutBuffer.indexOf('\n');
    const line = stdoutBuffer.slice(0, newline).trim();
    stdoutBuffer = stdoutBuffer.slice(newline + 1);
    if (!line) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      fail(new Error(`MCP stdout contained non-JSON protocol data: ${line}`));
      return;
    }
    if (message.jsonrpc !== '2.0') {
      fail(new Error(`MCP stdout contained a non-JSON-RPC message: ${line}`));
      return;
    }
    messages.push(message);
    const waiter = waiters.get(message.id);
    if (waiter) {
      waiters.delete(message.id);
      waiter.resolve(message);
    }
  }
});

function send(message) {
  child.stdin.write(`${JSON.stringify(message)}\n`);
}

function response(id, timeoutMs = 15_000) {
  const existing = messages.find((message) => message.id === id);
  if (existing) return Promise.resolve(existing);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      waiters.delete(id);
      reject(new Error(`Timed out waiting for MCP response ${id}\nstderr:\n${stderr}`));
    }, timeoutMs);
    waiters.set(id, {
      resolve: (message) => {
        clearTimeout(timer);
        resolve(message);
      },
      reject: (error) => {
        clearTimeout(timer);
        reject(error);
      },
    });
  });
}

async function terminate() {
  child.kill('SIGTERM');
  await Promise.race([
    new Promise((resolve) => child.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 3_000)),
  ]);
  if (child.exitCode === null) {
    if (process.platform === 'win32') child.kill('SIGKILL');
    else {
      try { process.kill(-child.pid, 'SIGKILL'); } catch { /* already exited */ }
    }
  }
}

try {
  send({
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2025-06-18',
      capabilities: {},
      clientInfo: { name: 'stdio-regression', version: '1.0.0' },
    },
  });
  const initialized = await response(1);
  if (initialized.error) throw new Error(`MCP initialize failed: ${JSON.stringify(initialized.error)}`);

  send({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} });
  send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
  const tools = await response(2);
  if (!Array.isArray(tools.result?.tools)) {
    throw new Error(`MCP tools/list returned an invalid response: ${JSON.stringify(tools)}`);
  }

  send({
    jsonrpc: '2.0',
    id: 3,
    method: 'tools/call',
    params: {
      name: 'learning_train',
      arguments: { session_id: 'missing-session', epochs: 1 },
    },
  });
  await response(3);

  await new Promise((resolve) => setTimeout(resolve, 100));
  if (stdoutBuffer.trim()) {
    throw new Error(`MCP stdout ended with incomplete protocol data: ${stdoutBuffer}`);
  }
  console.error(`MCP stdio purity verified across ${messages.length} JSON-RPC responses`);
} finally {
  await terminate();
  rmSync(directory, { recursive: true, force: true });
}
