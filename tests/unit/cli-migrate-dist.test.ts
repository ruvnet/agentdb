/**
 * `agentdb migrate <legacy.db> --target <new.db>` from the built package.
 *
 * The command loaded its schema from ../../schemas/ relative to
 * dist/src/cli/commands/, i.e. dist/src/schemas/, which the package does not
 * ship (the .sql files live in dist/schemas/). The target was left without
 * tables and the command failed with "no such table: episodes". It now uses
 * the schemas inlined at build time. The test runs the built CLI because the
 * source tree hides the bug (src/schemas/ does exist).
 */
import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import initSqlJs from 'sql.js';

const cli = resolve('dist/src/cli/agentdb-cli.js');

describe.skipIf(!existsSync(cli))('agentdb migrate (built CLI)', () => {
  it('migrates v1 episodes into a fresh target database', async () => {
    const SQL = await initSqlJs();
    const dir = mkdtempSync(join(tmpdir(), 'agentdb-migrate-'));
    try {
      const legacy = new SQL.Database();
      legacy.run(`CREATE TABLE episodes (
        id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, task TEXT NOT NULL,
        output TEXT, reward REAL, success BOOLEAN)`);
      legacy.run(`INSERT INTO episodes (session_id, task, output, reward, success) VALUES
        ('s1', 'parse config', 'ok', 0.9, 1), ('s1', 'write tests', 'ok', 0.7, 1)`);
      writeFileSync(join(dir, 'legacy.db'), Buffer.from(legacy.export()));
      legacy.close();

      const r = spawnSync(process.execPath, [cli, 'migrate', 'legacy.db', '--target', 'out.db', '--no-optimize'], {
        cwd: dir, encoding: 'utf8', timeout: 120_000,
      });
      const out = `${r.stdout}\n${r.stderr}`;
      expect(out).not.toMatch(/no such table/);
      expect(r.status).toBe(0);
      expect(out).toMatch(/Migrated 2 records from episodes/);

      const target = new SQL.Database(readFileSync(join(dir, 'out.db')));
      const rows = target.exec('SELECT task FROM episodes ORDER BY id')[0].values.map((v) => v[0]);
      target.close();
      expect(rows).toEqual(['parse config', 'write tests']);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
