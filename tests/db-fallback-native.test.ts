// Regression test for ruflo#2235(A) — db-fallback used to be sql.js-only by
// construction, so the bundled better-sqlite3 native binary was never engaged
// by the MCP memory bridge. The fix tries better-sqlite3 first, falling back
// to sql.js when the native module isn't installed.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getDatabaseImplementation,
  getDatabaseInfo,
  createDatabase,
  _resetDatabaseImplementationForTests,
} from '../src/db-fallback.js';

let nativeAvailable = false;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const BetterSqlite3 = require('better-sqlite3');
  const probe = new BetterSqlite3(':memory:');
  probe.close();
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe('db-fallback resolution order (ruflo#2235 A)', () => {
  beforeEach(() => {
    _resetDatabaseImplementationForTests();
    delete process.env.AGENTDB_FORCE_SQLJS;
  });

  it.skipIf(!nativeAvailable)('prefers better-sqlite3 when the native module is loadable', async () => {
    const Impl = await getDatabaseImplementation();
    expect(typeof Impl).toBe('function'); // class constructor
    const info = getDatabaseInfo();
    expect(info.implementation).toMatch(/better-sqlite3/);
    expect(info.isNative).toBe(true);
  });

  it('falls back to sql.js when AGENTDB_FORCE_SQLJS is set', async () => {
    process.env.AGENTDB_FORCE_SQLJS = '1';
    _resetDatabaseImplementationForTests();
    await getDatabaseImplementation();
    const info = getDatabaseInfo();
    expect(info.implementation).toMatch(/sql\.js/);
    expect(info.isNative).toBe(false);
  });

  it('exposes the same prepare/run/all API regardless of backend', async () => {
    const db = await createDatabase(':memory:');
    db.exec('CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)');
    db.prepare('INSERT INTO t (name) VALUES (?)').run('alpha');
    db.prepare('INSERT INTO t (name) VALUES (?)').run('beta');
    const rows = db.prepare('SELECT name FROM t ORDER BY id').all() as Array<{ name: string }>;
    expect(rows.map((r) => r.name)).toEqual(['alpha', 'beta']);
  });
});

// ruflo#3175 — a better-sqlite3 that is installed but cannot load (e.g. a
// NODE_MODULE_VERSION mismatch after a Node upgrade with a reused npx cache)
// used to be swallowed by a bare catch and reported with a green sql.js
// success line. Each case loads a fresh copy of db-fallback with a mocked
// better-sqlite3 so the module-level cache and captured error start clean.
describe('db-fallback native load diagnostics (ruflo#3175)', () => {
  const ABI_MESSAGE =
    "The module '/x/node_modules/better-sqlite3/build/Release/better_sqlite3.node'\n" +
    'was compiled against a different Node.js version using\n' +
    'NODE_MODULE_VERSION 137. This version of Node.js requires\n' +
    'NODE_MODULE_VERSION 141. Please try re-compiling or re-installing\n' +
    'the module (for instance, using `npm rebuild` or `npm install`).';

  async function loadWithNative(factory: () => unknown) {
    vi.resetModules();
    vi.doMock('better-sqlite3', factory);
    return import('../src/db-fallback.js');
  }

  let warnSpy: ReturnType<typeof vi.spyOn>;
  let errorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    delete process.env.AGENTDB_FORCE_SQLJS;
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.doUnmock('better-sqlite3');
    vi.resetModules();
    vi.restoreAllMocks();
  });

  const allOutput = () =>
    [...warnSpy.mock.calls, ...errorSpy.mock.calls].map((c) => c.join(' ')).join('\n');

  it('ABI mismatch: warns once, records nativeError, and still serves sql.js', async () => {
    const mod = await loadWithNative(() => ({
      default: class {
        constructor() {
          throw Object.assign(new Error(ABI_MESSAGE), { code: 'ERR_DLOPEN_FAILED' });
        }
      },
    }));

    const Impl = await mod.getDatabaseImplementation();
    const db = new Impl(':memory:');
    db.exec('CREATE TABLE t (v INTEGER)');
    db.prepare('INSERT INTO t (v) VALUES (?)').run(7);
    expect(db.prepare('SELECT v FROM t').get()).toEqual({ v: 7 });
    db.close();

    const info = mod.getDatabaseInfo();
    expect(info.implementation).toMatch(/sql\.js/);
    expect(info.isNative).toBe(false);
    expect(info.fallback).toBe(true);
    expect(info.nativeStatus).toBe('load-failed');
    expect(info.nativeError).toEqual({ code: 'ERR_DLOPEN_FAILED', message: ABI_MESSAGE });
    expect(Object.isFrozen(info.nativeError)).toBe(true);

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const warning = String(warnSpy.mock.calls[0][0]);
    expect(warning).toContain('⚠');
    expect(warning).toContain('NODE_MODULE_VERSION 137');
    expect(warning).toContain('141');
    expect(warning).toContain('npm rebuild better-sqlite3');
    expect(warning).toMatch(/npx cache/);
    // No green success line for an unhealthy fallback.
    expect(allOutput()).not.toContain('✅');
  });

  it('ABI mismatch without an error code is still classified (code synthesised)', async () => {
    const mod = await loadWithNative(() => ({
      default: class {
        constructor() {
          throw new Error(ABI_MESSAGE);
        }
      },
    }));
    await mod.getDatabaseImplementation();
    expect(mod.getDatabaseInfo().nativeError?.code).toBe('ERR_DLOPEN_FAILED');
    expect(String(warnSpy.mock.calls[0][0])).toContain('npm rebuild better-sqlite3');
  });

  it('missing native binding (no code) warns with NATIVE_LOAD_FAILED', async () => {
    const mod = await loadWithNative(() => ({
      default: class {
        constructor() {
          throw new Error('Could not locate the bindings file. Tried:\n → /x/build/better_sqlite3.node');
        }
      },
    }));
    await mod.getDatabaseImplementation();
    const info = mod.getDatabaseInfo();
    expect(info.nativeStatus).toBe('load-failed');
    expect(info.nativeError?.code).toBe('NATIVE_LOAD_FAILED');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain('Could not locate the bindings file');
  });

  it('not installed (MODULE_NOT_FOUND): neutral info, no warning, no nativeError', async () => {
    const mod = await loadWithNative(() => {
      throw Object.assign(
        new Error("Cannot find package 'better-sqlite3' imported from /x/dist/src/db-fallback.js"),
        { code: 'ERR_MODULE_NOT_FOUND' }
      );
    });
    await mod.getDatabaseImplementation();
    const info = mod.getDatabaseInfo();
    expect(info.implementation).toMatch(/sql\.js/);
    expect(info.fallback).toBe(true);
    expect(info.nativeStatus).toBe('not-installed');
    expect(info.nativeError).toBeUndefined();
    expect(warnSpy).not.toHaveBeenCalled();
    expect(allOutput()).toMatch(/ℹ.*not installed/);
    expect(allOutput()).not.toContain('⚠');
  });

  it('missing transitive dependency is a broken install, not "not installed"', async () => {
    const mod = await loadWithNative(() => {
      throw Object.assign(
        new Error("Cannot find package 'bindings' imported from /x/node_modules/better-sqlite3/lib/database.js"),
        { code: 'ERR_MODULE_NOT_FOUND' }
      );
    });
    await mod.getDatabaseImplementation();
    expect(mod.getDatabaseInfo().nativeStatus).toBe('load-failed');
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it('native OK: no fallback, no nativeError, no warning', async () => {
    const mod = await loadWithNative(() => ({
      default: class {
        close() {}
      },
    }));
    await mod.getDatabaseImplementation();
    const info = mod.getDatabaseInfo();
    expect(info.isNative).toBe(true);
    expect(info.fallback).toBe(false);
    expect(info.nativeStatus).toBe('loaded');
    expect(info.nativeError).toBeUndefined();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('AGENTDB_FORCE_SQLJS is intentional: not a fallback, no warning', async () => {
    process.env.AGENTDB_FORCE_SQLJS = '1';
    const mod = await loadWithNative(() => {
      throw new Error('must not be imported');
    });
    await mod.getDatabaseImplementation();
    const info = mod.getDatabaseInfo();
    expect(info.fallback).toBe(false);
    expect(info.nativeStatus).toBe('disabled');
    expect(info.nativeError).toBeUndefined();
    expect(warnSpy).not.toHaveBeenCalled();
    delete process.env.AGENTDB_FORCE_SQLJS;
  });

  it('reports no fallback before the implementation is selected, and reset clears the error', async () => {
    const mod = await loadWithNative(() => ({
      default: class {
        constructor() {
          throw Object.assign(new Error(ABI_MESSAGE), { code: 'ERR_DLOPEN_FAILED' });
        }
      },
    }));
    expect(mod.getDatabaseInfo().fallback).toBe(false);
    expect(mod.getDatabaseInfo().nativeError).toBeUndefined();
    await mod.getDatabaseImplementation();
    expect(mod.getDatabaseInfo().nativeError).toBeDefined();
    mod._resetDatabaseImplementationForTests();
    expect(mod.getDatabaseInfo().nativeError).toBeUndefined();
    expect(mod.getDatabaseInfo().fallback).toBe(false);
  });
});
