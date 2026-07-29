import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import initSqlJs from 'sql.js';
import {
  _resetDatabaseImplementationForTests,
  createDatabase,
  SqlJsConcurrentModificationError,
} from '../src/db-fallback.js';
import { SqlJsRvfBackend } from '../src/backends/rvf/SqlJsRvfBackend.js';

let testDirectory: string;

async function rewriteFromAnotherProcess(filename: string, value: string): Promise<void> {
  const SQL = await initSqlJs();
  const external = new SQL.Database(readFileSync(filename));
  external.exec(`INSERT INTO events (value) VALUES ('${value.replaceAll("'", "''")}')`);
  writeFileSync(filename, Buffer.from(external.export()));
  external.close();
}

describe('sql.js file persistence', () => {
  beforeEach(() => {
    testDirectory = mkdtempSync(join(tmpdir(), 'agentdb-sqljs-'));
    process.env.AGENTDB_FORCE_SQLJS = '1';
    _resetDatabaseImplementationForTests();
  });

  afterEach(() => {
    delete process.env.AGENTDB_FORCE_SQLJS;
    _resetDatabaseImplementationForTests();
    rmSync(testDirectory, { recursive: true, force: true });
  });

  it('atomically persists an uncontended database', async () => {
    const filename = join(testDirectory, 'clean.db');
    const db = await createDatabase(filename);
    db.exec('CREATE TABLE events (value TEXT)');
    db.prepare('INSERT INTO events (value) VALUES (?)').run('local');
    db.close();

    const reopened = await createDatabase(filename);
    expect(reopened.prepare('SELECT value FROM events').all()).toEqual([{ value: 'local' }]);
    reopened.close();
  });

  it('rejects a stale save without destroying an external writer update', async () => {
    const filename = join(testDirectory, 'conflict.db');
    const seed = await createDatabase(filename);
    seed.exec('CREATE TABLE events (value TEXT)');
    seed.prepare('INSERT INTO events (value) VALUES (?)').run('seed');
    seed.close();

    const stale = await createDatabase(filename);
    stale.prepare('INSERT INTO events (value) VALUES (?)').run('stale');
    await rewriteFromAnotherProcess(filename, 'external');

    expect(() => stale.save()).toThrowError(SqlJsConcurrentModificationError);
    expect(() => stale.close()).toThrowError(SqlJsConcurrentModificationError);

    const SQL = await initSqlJs();
    const disk = new SQL.Database(readFileSync(filename));
    const values = disk.exec('SELECT value FROM events ORDER BY rowid')[0].values.flat();
    expect(values).toEqual(['seed', 'external']);
    disk.close();
  });

  it('treats WAL sidecar creation as a concurrent modification', async () => {
    const filename = join(testDirectory, 'wal-conflict.db');
    const seed = await createDatabase(filename);
    seed.exec('CREATE TABLE events (value TEXT)');
    seed.close();

    const stale = await createDatabase(filename);
    writeFileSync(`${filename}-wal`, 'external WAL activity');

    expect(() => stale.save()).toThrowError(SqlJsConcurrentModificationError);
    expect(() => stale.close()).toThrowError(SqlJsConcurrentModificationError);
  });

  it('rejects an existing database with an active rollback journal', async () => {
    const filename = join(testDirectory, 'journal-conflict.db');
    const seed = await createDatabase(filename);
    seed.exec('CREATE TABLE events (value TEXT)');
    seed.close();
    writeFileSync(`${filename}-journal`, 'uncheckpointed transaction');

    await expect(createDatabase(filename)).rejects.toThrow(/WAL or journal/);
  });

  it('allows only the first of two stale sql.js handles to persist', async () => {
    const filename = join(testDirectory, 'two-handles.db');
    const seed = await createDatabase(filename);
    seed.exec('CREATE TABLE events (value TEXT)');
    seed.close();

    const first = await createDatabase(filename);
    const second = await createDatabase(filename);
    first.prepare('INSERT INTO events (value) VALUES (?)').run('first');
    second.prepare('INSERT INTO events (value) VALUES (?)').run('second');
    first.save();

    expect(() => second.save()).toThrowError(SqlJsConcurrentModificationError);
    first.close();
    expect(() => second.close()).toThrowError(SqlJsConcurrentModificationError);
  });

  it('never replaces a corrupt existing database with a fresh one', async () => {
    const filename = join(testDirectory, 'corrupt.db');
    writeFileSync(filename, 'not a sqlite database');

    await expect(createDatabase(filename)).rejects.toThrow();
    expect(readFileSync(filename, 'utf8')).toBe('not a sqlite database');
  });

  it('applies the same conflict contract to the sql.js RVF backend', async () => {
    const filename = join(testDirectory, 'vectors.rvf');
    const seed = new SqlJsRvfBackend({ dimension: 2, storagePath: filename } as any);
    await seed.initialize();
    await seed.insertAsync('seed', new Float32Array([1, 0]));
    seed.close();

    const stale = new SqlJsRvfBackend({ dimension: 2, storagePath: filename } as any);
    await stale.initialize();
    const SQL = await initSqlJs();
    const external = new SQL.Database(readFileSync(filename));
    external.exec('CREATE TABLE external_update (value TEXT)');
    writeFileSync(filename, Buffer.from(external.export()));
    external.close();

    expect(() => stale.close()).toThrowError(SqlJsConcurrentModificationError);
    expect(stale.isInitialized()).toBe(false);

    const disk = new SQL.Database(readFileSync(filename));
    expect(disk.exec(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='external_update'"
    )[0].values[0][0]).toBe('external_update');
    disk.close();
  });
});
