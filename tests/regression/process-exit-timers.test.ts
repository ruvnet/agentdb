/**
 * ReasoningBank.getPatternStats() armed a 5-minute setTimeout to expire its
 * cache, so any short-lived process that called it (e.g. ruflo
 * `autopilot learn` via agentic-flow AgentDBService.getMetrics) stayed
 * alive for exactly 5 minutes. The sql.js database's leak-warning interval
 * likewise held open any process that never called close().
 *
 * Each case runs in a child process that must exit on its own within 5 s.
 */
import { describe, it, expect } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

const ROOT = path.resolve(__dirname, '../..');
const TSX = path.join(ROOT, 'node_modules/.bin/tsx');

function exitsWithin(script: string, ms: number, env: NodeJS.ProcessEnv = {}): Promise<{ exited: boolean; code: number | null; output: string }> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-exit-'));
  const file = path.join(dir, 'child.mts');
  fs.writeFileSync(file, script.replaceAll('@ROOT@', ROOT));
  const child = spawn(TSX, [file], { cwd: dir, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
  let output = '';
  child.stdout.on('data', (d) => { output += d; });
  child.stderr.on('data', (d) => { output += d; });
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      fs.rmSync(dir, { recursive: true, force: true });
      resolve({ exited: false, code: null, output });
    }, ms);
    child.once('exit', (code) => {
      clearTimeout(timer);
      fs.rmSync(dir, { recursive: true, force: true });
      resolve({ exited: true, code, output });
    });
  });
}

describe('library timers do not keep the process alive', () => {
  it('a process that calls ReasoningBank.getPatternStats() exits promptly', async () => {
    const result = await exitsWithin(`
      import Database from '@ROOT@/node_modules/better-sqlite3/lib/index.js';
      import { ReasoningBank } from '@ROOT@/src/controllers/ReasoningBank.ts';
      const db = new Database(':memory:');
      const embedder = { embed: async () => new Float32Array(8), embedQuery: async () => new Float32Array(8), embedPassage: async () => new Float32Array(8) };
      const bank = new ReasoningBank(db as any, embedder as any);
      const stats = bank.getPatternStats();
      if (bank.getPatternStats() !== stats) throw new Error('cached stats not reused');
      console.log('stats', stats.totalPatterns);
    `, 5000);

    expect(result.output).toContain('stats 0');
    expect(result.exited).toBe(true);
    expect(result.code).toBe(0);
  }, 20_000);

  it('a process that opens a sql.js database and never closes it exits promptly', async () => {
    const result = await exitsWithin(`
      import { createDatabase } from '@ROOT@/src/db-fallback.ts';
      const db = await createDatabase(':memory:');
      db.exec('CREATE TABLE t (x INTEGER)');
      console.log('opened');
    `, 5000, { AGENTDB_FORCE_SQLJS: '1' });

    expect(result.output).toContain('opened');
    expect(result.exited).toBe(true);
  }, 20_000);
});
