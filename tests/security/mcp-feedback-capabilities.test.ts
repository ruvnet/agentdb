/**
 * ADR-015 MCP surface, through the real stdio server:
 * - agentdb_capabilities (read-only, default and readonly profiles)
 * - agentdb_pattern_feedback / agentdb_skill_feedback (default learning
 *   profile), with input validation at the boundary.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawn, type ChildProcess } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

const ROOT = path.resolve(__dirname, '../..');
const TSX = path.join(ROOT, 'node_modules/.bin/tsx');
const SERVER = path.join(ROOT, 'src/mcp/agentdb-mcp-server.ts');

class McpClient {
  private buffer = '';
  private waiters = new Map<number, (msg: any) => void>();
  private nextId = 1;
  stderr = '';

  constructor(private child: ChildProcess) {
    child.stdout!.setEncoding('utf8');
    child.stdout!.on('data', (chunk: string) => {
      this.buffer += chunk;
      let nl: number;
      while ((nl = this.buffer.indexOf('\n')) >= 0) {
        const line = this.buffer.slice(0, nl).trim();
        this.buffer = this.buffer.slice(nl + 1);
        if (!line) continue;
        const msg = JSON.parse(line);
        this.waiters.get(msg.id)?.(msg);
      }
    });
    child.stderr!.setEncoding('utf8');
    child.stderr!.on('data', (chunk: string) => { this.stderr += chunk; });
  }

  request(method: string, params: unknown): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`timeout on ${method}\n${this.stderr}`)), 60_000);
      this.waiters.set(id, (msg) => { clearTimeout(timer); resolve(msg); });
      this.child.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    });
  }

  async start(): Promise<void> {
    const init = await this.request('initialize', {
      protocolVersion: '2025-06-18',
      capabilities: {},
      clientInfo: { name: 'feedback-capabilities-test', version: '1.0.0' },
    });
    if (init.error) throw new Error(JSON.stringify(init.error));
    this.child.stdin!.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');
  }

  callTool(name: string, args: Record<string, unknown>): Promise<any> {
    return this.request('tools/call', { name, arguments: args });
  }
}

function startServer(dir: string, profile?: string): { child: ChildProcess; client: McpClient } {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    AGENTDB_PATH: path.join(dir, 'agentdb.db'),
    AGENTDB_FORCE_SQLJS: '1',
    AGENTDB_DISABLE_TRANSFORMERS: '1',
  };
  delete env.AGENTDB_MCP_PROFILE;
  delete env.AGENTDB_MCP_ALLOW;
  delete env.AGENTDB_MCP_DENY;
  if (profile) env.AGENTDB_MCP_PROFILE = profile;
  const child = spawn(TSX, [SERVER], { cwd: dir, env, stdio: ['pipe', 'pipe', 'pipe'] });
  return { child, client: new McpClient(child) };
}

function text(response: any): string {
  return response.result?.content?.map((c: any) => c.text).join('\n') ?? JSON.stringify(response.error);
}


async function stop(child: ChildProcess | undefined): Promise<void> {
  if (child && child.exitCode === null) {
    child.kill('SIGKILL');
    await new Promise((r) => child.once('exit', r));
  }
}

const NEW_TOOLS = ['agentdb_capabilities', 'agentdb_pattern_feedback', 'agentdb_skill_feedback'];

describe('ADR-015 MCP tools (default profile)', () => {
  let dir: string;
  let server: { child: ChildProcess; client: McpClient };

  beforeAll(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-mcp-adr015-'));
    server = startServer(dir);
    await server.client.start();
  }, 90_000);

  afterAll(async () => {
    await stop(server?.child);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('lists the capability and feedback tools', async () => {
    const list = await server.client.request('tools/list', {});
    const names = list.result.tools.map((t: any) => t.name);
    for (const tool of NEW_TOOLS) expect(names).toContain(tool);
  });

  it('agentdb_capabilities reports accelerator, backends, SQL driver and embedding', async () => {
    const res = await server.client.callTool('agentdb_capabilities', {});
    expect(res.result.isError).toBeFalsy();
    const caps = JSON.parse(text(res));

    expect(Object.values(caps.nativeAccelerator).every((v) => typeof v === 'boolean')).toBe(true);
    expect(caps.nativeAccelerator).toHaveProperty('simd');
    expect(caps.vectorBackend.inUse).toBe('sql-exact-scan');
    expect(['ruvector', 'rvf', 'hnswlib', 'sqljsrvf', 'none']).toContain(caps.vectorBackend.available);
    expect(caps.sqlDriver.implementation).toBe('sql.js (WASM)'); // AGENTDB_FORCE_SQLJS
    expect(caps.embedding).toEqual({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers',
      active: 'mock', // AGENTDB_DISABLE_TRANSFORMERS
    });
    expect(caps.retrieval.patternRerankWeights).toEqual({ similarity: 0.7, successRate: 0.2, reward: 0.1 });
    expect(caps.mcpProfile).toBe('learning');
  }, 60_000);

  it('agentdb_pattern_feedback updates the pattern and validates input', async () => {
    const stored = text(await server.client.callTool('agentdb_pattern_store', {
      taskType: 'code_review', approach: 'read the diff twice', successRate: 0.5,
    }));
    const id = Number(/Pattern ID: (\d+)/.exec(stored)![1]);

    const ok = await server.client.callTool('agentdb_pattern_feedback', { pattern_id: id, success: true, reward: 0.8 });
    expect(ok.result.isError).toBeFalsy();
    expect(JSON.parse(text(ok))).toMatchObject({ patternId: id, uses: 1, successRate: 1, avgReward: 0.8 });

    // A reward alone: >= 0.5 counts as success
    const rewardOnly = JSON.parse(text(await server.client.callTool('agentdb_pattern_feedback', { pattern_id: id, reward: 0.2 })));
    expect(rewardOnly).toMatchObject({ recorded: { success: false, reward: 0.2 }, uses: 2, successRate: 0.5 });

    const bad: Array<[Record<string, unknown>, RegExp]> = [
      [{ pattern_id: id }, /success or reward is required/],
      [{ pattern_id: id, reward: 1.5 }, /reward/],
      [{ pattern_id: id, reward: 'high' }, /reward/],
      [{ pattern_id: id, success: 'yes' }, /success/],
      [{ pattern_id: -1, success: true }, /pattern_id/],
      [{ pattern_id: 1.5, success: true }, /pattern_id/],
      [{ pattern_id: 999999, success: true }, /not found/],
    ];
    for (const [args, message] of bad) {
      const res = await server.client.callTool('agentdb_pattern_feedback', args);
      expect(res.result.isError, JSON.stringify(args)).toBe(true);
      expect(text(res)).toMatch(message);
    }
  }, 60_000);

  it('agentdb_skill_feedback updates the skill and validates input', async () => {
    const created = text(await server.client.callTool('skill_create', {
      name: 'csv-parser', description: 'parse csv files', success_rate: 0.5,
    }));
    const id = Number(/#(\d+)/.exec(created)![1]);

    const ok = JSON.parse(text(await server.client.callTool('agentdb_skill_feedback', {
      skill_id: id, success: false, latency_ms: 120,
    })));
    expect(ok).toMatchObject({ skillId: id, uses: 1, successRate: 0, avgReward: 0, avgLatencyMs: 120 });

    for (const args of [
      { skill_id: id },
      { skill_id: id, success: true, latency_ms: -1 },
      { skill_id: id, reward: 2 },
      { skill_id: 'x', success: true },
      { skill_id: 424242, success: true },
    ]) {
      const res = await server.client.callTool('agentdb_skill_feedback', args);
      expect(res.result.isError, JSON.stringify(args)).toBe(true);
    }
  }, 60_000);
});

describe('ADR-015 MCP tools (readonly profile)', () => {
  it('exposes agentdb_capabilities but not the feedback writers', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-mcp-adr015-ro-'));
    const server = startServer(dir, 'readonly');
    try {
      await server.client.start();
      const names = (await server.client.request('tools/list', {})).result.tools.map((t: any) => t.name);
      expect(names).toContain('agentdb_capabilities');
      expect(names).not.toContain('agentdb_pattern_feedback');
      expect(names).not.toContain('agentdb_skill_feedback');
      const res = await server.client.callTool('agentdb_skill_feedback', { skill_id: 1, success: true });
      expect(text(res)).toMatch(/not permitted/);
    } finally {
      await stop(server.child);
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 90_000);
});
