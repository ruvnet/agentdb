// Regression test for the Node 24 abort in better-sqlite3 Statement GC
// (`Assertion failed: (env) != nullptr` in node::RemoveEnvironmentCleanupHook,
// from Statement::~Statement()).
//
// Node >= 24.19 headers ship a partial `node::ObjectWrap` cleanup-hook backport
// (nodejs/node#65446): an addon compiled against them aborts when V8 collects a
// wrapped object in an allocation-driven GC. better-sqlite3 11.x has no Node 24
// prebuild, so `^11` was always compiled locally against those headers; 12.x
// ships a Node 24 prebuild built before the regression.

import { describe, it, expect } from 'vitest';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const pkg = JSON.parse(readFileSync(join(__dirname, '..', 'package.json'), 'utf8'));

let nativeAvailable = false;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const BetterSqlite3 = require('better-sqlite3');
  new BetterSqlite3(':memory:').close();
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

// Prepare and drop many statements while allocating, so they are collected by
// allocation-driven GC rather than an explicit gc() call.
const STATEMENT_CHURN = `
  const Database = require('better-sqlite3');
  const db = new Database(':memory:');
  db.exec('CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)');
  let junk = [];
  for (let i = 0; i < 200000; i++) {
    db.prepare('SELECT v FROM t WHERE id = ?').get(i);
    junk.push({ i, s: 'x'.repeat(16) });
    if (junk.length > 1000) junk = [];
  }
  db.close();
  console.log('SURVIVED');
`;

describe('better-sqlite3 statement GC (Node 24 abort)', () => {
  it('declares a better-sqlite3 range with Node 24 prebuilds (>= 12.8.0)', () => {
    const range: string = pkg.optionalDependencies['better-sqlite3'];
    const match = /^\^(\d+)\.(\d+)\.(\d+)$/.exec(range);

    expect(match, `unexpected range ${range}`).not.toBeNull();
    const [major, minor] = [Number(match![1]), Number(match![2])];
    expect(major > 12 || (major === 12 && minor >= 8)).toBe(true);
  });

  it.skipIf(!nativeAvailable)('survives statements collected by allocation-driven GC', () => {
    const child = spawnSync(process.execPath, ['-e', STATEMENT_CHURN], {
      cwd: join(__dirname, '..'),
      encoding: 'utf8',
      timeout: 120_000,
    });

    expect(child.stderr).not.toContain('(env) != nullptr');
    expect(child.signal).toBeNull();
    expect(child.stdout).toContain('SURVIVED');
  }, 150_000);
});
