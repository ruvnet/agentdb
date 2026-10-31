/**
 * validatePragmaCommand only checked the first token, and the sql.js
 * wrapper ran the result through db.exec(), which executes stacked
 * statements: "journal_mode = WAL; DROP TABLE t" passed and dropped t.
 */
import { describe, it, expect } from 'vitest';
import { validatePragmaCommand } from '../../src/security/input-validation.js';

describe('PRAGMA validation', () => {
  it('rejects stacked statements and junk after the value', () => {
    expect(() => validatePragmaCommand('journal_mode = WAL; DROP TABLE episodes')).toThrow();
    expect(() => validatePragmaCommand('foreign_keys=ON;DELETE FROM skills')).toThrow();
    expect(() => validatePragmaCommand('cache_size = -64000 -- x')).toThrow();
  });

  it('accepts plain, assigned and table_info pragmas', () => {
    expect(validatePragmaCommand('journal_mode = WAL')).toBe('journal_mode = WAL');
    expect(validatePragmaCommand('cache_size = -64000')).toBe('cache_size = -64000');
    expect(validatePragmaCommand('page_count')).toBe('page_count');
    expect(validatePragmaCommand('table_info(episodes)')).toBe('table_info(episodes)');
    expect(() => validatePragmaCommand('table_info(episodes); DROP TABLE x')).toThrow();
  });

  it('sql.js wrapper never runs a stacked statement and returns table_info rows', async () => {
    const saved = process.env.AGENTDB_FORCE_SQLJS;
    process.env.AGENTDB_FORCE_SQLJS = '1';
    try {
      const { createDatabase, wrapExistingSqlJsDatabase } = await import('../../src/db-fallback.js');
      const SQL = await (await import('sql.js')).default();
      const raw = new SQL.Database();
      raw.run('CREATE TABLE victim (id INTEGER, name TEXT)');
      const db = wrapExistingSqlJsDatabase(raw);

      expect(() => db.pragma('user_version; DROP TABLE victim')).toThrow();
      expect(raw.exec("SELECT name FROM sqlite_master WHERE name = 'victim'")[0]?.values).toEqual([['victim']]);

      const columns = db.pragma('table_info(victim)') as Array<{ name: string }>;
      expect(columns.map((c) => c.name)).toEqual(['id', 'name']);
      expect(createDatabase).toBeTypeOf('function');
    } finally {
      if (saved === undefined) delete process.env.AGENTDB_FORCE_SQLJS;
      else process.env.AGENTDB_FORCE_SQLJS = saved;
    }
  });
});
