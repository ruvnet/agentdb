/**
 * In-memory RVF stores are private and leave nothing behind.
 *
 * Before 3.7.1 an RvfBackend without a storagePath opened `./agentdb.rvf` in
 * the working directory (so an in-memory AgentDB on rvf reopened whatever an
 * earlier run left there), and `storagePath: ':memory:'` created a file
 * literally named `:memory:`. Now both use a fresh temp directory per
 * instance, removed on close.
 */
import { describe, it, expect } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';

const ROOT = path.resolve(__dirname, '../..');
const TSX = path.join(ROOT, 'node_modules/.bin/tsx');
const DIM = 4;
const vec = (...v: number[]) => new Float32Array(v);

async function waitGone(p: string, ms = 5000): Promise<boolean> {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (!fs.existsSync(p)) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return !fs.existsSync(p);
}

describe('RvfBackend in-memory stores', () => {
  it('two instances without a path share no state and live in separate temp dirs', async () => {
    const a = new RvfBackend({ dimension: DIM, metric: 'cosine' });
    const b = new RvfBackend({ dimension: DIM, metric: 'cosine', storagePath: ':memory:' });
    await a.initialize();
    await b.initialize();
    try {
      await a.insertAsync('only-in-a', vec(1, 0, 0, 0));
      expect((await a.status()).totalVectors).toBe(1);
      expect((await b.status()).totalVectors).toBe(0);
      expect(await b.searchAsync(vec(1, 0, 0, 0), 5)).toEqual([]);

      const pa = a.getStoragePath();
      const pb = b.getStoragePath();
      expect(pa).not.toBe(pb);
      for (const p of [pa, pb]) {
        expect(path.resolve(p).startsWith(fs.realpathSync(os.tmpdir())) || p.startsWith(os.tmpdir())).toBe(true);
        expect(path.dirname(path.resolve(p))).not.toBe(process.cwd());
      }
    } finally {
      await a.closeAsync();
      await b.closeAsync();
    }
  });

  it("':memory:' and an omitted path create nothing in the working directory", async () => {
    const before = new Set(fs.readdirSync(process.cwd()));
    const stores = [
      new RvfBackend({ dimension: DIM, storagePath: ':memory:' }),
      new RvfBackend({ dimension: DIM }),
    ];
    for (const s of stores) {
      await s.initialize();
      await s.insertAsync('x', vec(0, 1, 0, 0));
    }
    expect(fs.existsSync(path.join(process.cwd(), ':memory:'))).toBe(false);
    expect(fs.existsSync(path.join(process.cwd(), 'agentdb.rvf'))).toBe(false);
    for (const s of stores) await s.closeAsync();
    const added = fs.readdirSync(process.cwd()).filter((f) => !before.has(f));
    expect(added).toEqual([]);
  });

  it('closeAsync() removes the temp directory; close() removes it once rvf lets go', async () => {
    const s1 = new RvfBackend({ dimension: DIM });
    await s1.initialize();
    await s1.insertAsync('a', vec(1, 1, 0, 0));
    const dir1 = path.dirname(s1.getStoragePath());
    expect(fs.existsSync(dir1)).toBe(true);
    await s1.closeAsync();
    expect(fs.existsSync(dir1)).toBe(false);

    const s2 = new RvfBackend({ dimension: DIM, storagePath: ':memory:' });
    await s2.initialize();
    const dir2 = path.dirname(s2.getStoragePath());
    s2.close();
    expect(await waitGone(dir2)).toBe(true);
  });

  it('re-initializing after close starts empty', async () => {
    const s = new RvfBackend({ dimension: DIM });
    await s.initialize();
    await s.insertAsync('a', vec(1, 0, 0, 1));
    await s.closeAsync();
    await s.initialize();
    try {
      expect((await s.status()).totalVectors).toBe(0);
    } finally {
      await s.closeAsync();
    }
  });

  it('a given path is still used as is', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rvf-path-'));
    const file = path.join(dir, 'kept.rvf');
    try {
      const s = new RvfBackend({ dimension: DIM, storagePath: file });
      await s.initialize();
      await s.insertAsync('a', vec(1, 0, 0, 0));
      await s.closeAsync();
      expect(fs.existsSync(file)).toBe(true);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('in-memory AgentDB on the rvf backend', () => {
  it('writes nothing to the working directory and two instances share no vectors', async () => {
    const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-inmem-cwd-'));
    const scriptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-inmem-script-'));
    const script = path.join(scriptDir, 'child.mts');
    fs.writeFileSync(script, `
      import { AgentDB } from '${ROOT}/src/core/AgentDB.ts';
      import * as fs from 'node:fs';
      import * as path from 'node:path';
      const a = new AgentDB({ vectorBackend: 'rvf' });
      const b = new AgentDB({ dbPath: ':memory:', vectorBackend: 'rvf' });
      await a.initialize();
      await b.initialize();
      await a.getController('reflexion').storeEpisode({ sessionId: 's', task: 'parse csv files', reward: 1, success: true });
      await a.vectorBackend.flush?.();
      const pa = a.vectorBackend.getStoragePath?.();
      const pb = b.vectorBackend.getStoragePath?.();
      const out = {
        backend: a.vectorBackend.name,
        countA: (await a.vectorBackend.status()).totalVectors,
        countB: (await b.vectorBackend.status()).totalVectors,
        pa, pb,
      };
      await a.close();
      await b.close();
      out.dirsGone = !fs.existsSync(path.dirname(pa)) && !fs.existsSync(path.dirname(pb));
      process.stderr.write('RESULT ' + JSON.stringify(out) + '\\n');
    `);
    try {
      const child = spawn(TSX, [script], {
        cwd,
        env: { ...process.env, AGENTDB_DISABLE_TRANSFORMERS: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stderr = '';
      child.stderr.on('data', (d) => (stderr += d));
      child.stdout.on('data', () => {});
      const code = await new Promise<number | null>((resolve) => child.on('close', resolve));
      const line = stderr.split('\n').find((l) => l.startsWith('RESULT '));
      expect(code, stderr).toBe(0);
      expect(line, stderr).toBeDefined();
      const r = JSON.parse(line!.slice('RESULT '.length));
      expect(r.backend).toBe('rvf');
      expect(r.countA).toBe(1);
      expect(r.countB).toBe(0);
      expect(r.pa).not.toBe(r.pb);
      expect(r.dirsGone).toBe(true);
      expect(fs.readdirSync(cwd)).toEqual([]);
    } finally {
      fs.rmSync(cwd, { recursive: true, force: true });
      fs.rmSync(scriptDir, { recursive: true, force: true });
    }
  }, 60_000);
});

/**
 * Run a child with a private TMPDIR that opens in-memory stores and never
 * closes them, then ends the way `mode` says. Returns what is left in TMPDIR.
 */
async function runUnclosedChild(mode: 'exit' | 'return' | 'SIGTERM' | 'SIGINT') {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-exit-hook-'));
  const scriptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-exit-script-'));
  const script = path.join(scriptDir, 'child.mts');
  fs.writeFileSync(script, `
    import { RvfBackend } from '${ROOT}/src/backends/rvf/RvfBackend.ts';
    const stores = [new RvfBackend({ dimension: 4 }), new RvfBackend({ dimension: 4, storagePath: ':memory:' })];
    for (const s of stores) {
      await s.initialize();
      await s.insertAsync('x', new Float32Array([1, 0, 0, 0]));
    }
    // Many instances, one set of listeners.
    for (let i = 0; i < 20; i++) await new RvfBackend({ dimension: 4 }).initialize();
    const listeners = { exit: process.listenerCount('exit'), SIGINT: process.listenerCount('SIGINT'), SIGTERM: process.listenerCount('SIGTERM') };
    process.stderr.write('READY ' + JSON.stringify(listeners) + '\\n');
    const mode = ${JSON.stringify(mode)};
    if (mode === 'exit') process.exit(0);
    if (mode !== 'return') setInterval(() => {}, 1000); // wait for the signal
  `);
  try {
    const child = spawn(TSX, [script], {
      env: { ...process.env, TMPDIR: tmp, TMP: tmp, TEMP: tmp },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stderr = '';
    let listeners: Record<string, number> | null = null;
    child.stdout.on('data', () => {});
    child.stderr.on('data', (d) => {
      stderr += d;
      const line = stderr.split('\n').find((l) => l.startsWith('READY '));
      if (line && !listeners) {
        listeners = JSON.parse(line.slice('READY '.length));
        if (mode === 'SIGTERM' || mode === 'SIGINT') child.kill(mode);
      }
    });
    const timer = setTimeout(() => child.kill('SIGKILL'), 45_000);
    const [code, signal] = await new Promise<[number | null, NodeJS.Signals | null]>((resolve) =>
      child.on('close', (c, s) => resolve([c, s])),
    );
    clearTimeout(timer);
    // tsx's own cache may land in TMPDIR; only our store directories matter.
    const left = fs.readdirSync(tmp).filter((f) => f.startsWith('agentdb-rvf-'));
    return { code, signal, listeners, left, stderr };
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
    fs.rmSync(scriptDir, { recursive: true, force: true });
  }
}

describe('in-memory stores a process never closes', () => {
  it.each(['exit', 'return'] as const)('are removed when the process ends (%s)', async (mode) => {
    const r = await runUnclosedChild(mode);
    expect(r.code, r.stderr).toBe(0);
    expect(r.listeners, r.stderr).toEqual({ exit: expect.any(Number), SIGINT: 1, SIGTERM: 1 });
    expect(r.left).toEqual([]);
  }, 60_000);

  it.each(['SIGTERM', 'SIGINT'] as const)('are removed on %s, and the signal still terminates', async (sig) => {
    const r = await runUnclosedChild(sig);
    expect(r.listeners, r.stderr).not.toBeNull();
    expect(r.left).toEqual([]);
    // The default action still runs: tsx reports the child's signal as 128 + signo.
    const signo = sig === 'SIGTERM' ? 15 : 2;
    expect(r.signal === sig || r.code === 128 + signo, `code=${r.code} signal=${r.signal}`).toBe(true);
  }, 60_000);
});
