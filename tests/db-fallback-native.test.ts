// Regression test for ruflo#2235(A) — db-fallback used to be sql.js-only by
// construction, so the bundled better-sqlite3 native binary was never engaged
// by the MCP memory bridge. The fix tries better-sqlite3 first, falling back
// to sql.js when the native module isn't installed.

import { describe, it, expect, beforeEach } from 'vitest';
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
