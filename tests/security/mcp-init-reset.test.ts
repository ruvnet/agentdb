/**
 * agentdb_init took a `db_path` and, with reset=true, ran
 * fs.unlinkSync(db_path) with no validation; the default MCP policy allowed
 * every tool. Any MCP client could delete any file the server could write.
 *
 * Now: no db_path, reset drops tables on the open handle, and destructive
 * tools need AGENTDB_MCP_PROFILE=admin.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
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
      clientInfo: { name: 'init-reset-test', version: '1.0.0' },
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

describe('agentdb_init over MCP', () => {
  let dir: string;
  let child: ChildProcess | undefined;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-mcp-init-'));
  });

  afterEach(async () => {
    if (child && child.exitCode === null) {
      child.kill('SIGKILL');
      await new Promise((r) => child!.once('exit', r));
    }
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('never deletes a caller-supplied db_path, even as admin', async () => {
    const victim = path.join(dir, 'victim.txt');
    fs.writeFileSync(victim, 'do not delete');
    const server = startServer(dir, 'admin');
    child = server.child;
    await server.client.start();

    await server.client.callTool('agentdb_init', { db_path: victim, reset: true });

    expect(fs.existsSync(victim)).toBe(true);
    expect(fs.readFileSync(victim, 'utf8')).toBe('do not delete');
  }, 90_000);

  it('reset clears the open database and keeps it usable', async () => {
    const server = startServer(dir, 'admin');
    child = server.child;
    await server.client.start();

    await server.client.callTool('reflexion_store', {
      session_id: 's', task: 'remember me', reward: 1, success: true,
    });
    const before = text(await server.client.callTool('db_stats', {}));
    expect(before).toMatch(/Episodes \(Vectors\): 1\b/);

    const init = await server.client.callTool('agentdb_init', { reset: true });
    expect(text(init)).toMatch(/initialized successfully/);

    const after = text(await server.client.callTool('db_stats', {}));
    expect(after).not.toMatch(/Episodes \(Vectors\): 1\b/);
  }, 90_000);

  it('is not available without the admin profile', async () => {
    const victim = path.join(dir, 'victim.txt');
    fs.writeFileSync(victim, 'x');
    const server = startServer(dir);
    child = server.child;
    await server.client.start();

    const list = await server.client.request('tools/list', {});
    const names = list.result.tools.map((t: any) => t.name);
    expect(names).not.toContain('agentdb_init');
    expect(names).not.toContain('agentdb_delete');
    expect(names).toContain('reflexion_store');

    const res = await server.client.callTool('agentdb_init', { db_path: victim, reset: true });
    expect(text(res)).toMatch(/not permitted/);
    expect(fs.existsSync(victim)).toBe(true);
  }, 90_000);
});
