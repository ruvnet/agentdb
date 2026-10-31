/**
 * Reset an open AgentDB database in place.
 *
 * agentdb_init used to unlink a caller-supplied path (arbitrary file
 * deletion) and, for the default path, the database the server still held
 * open. Resetting through the open handle touches nothing but this database.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- better-sqlite3 or the sql.js wrapper
type Database = any;

function quoteIdentifier(name: string): string {
  return `"${name.replace(/"/g, '""')}"`;
}

/**
 * Drop every user table in one transaction. Returns the dropped table names.
 * The caller re-creates the schema afterwards.
 */
export function resetDatabaseTables(db: Database): string[] {
  const rows = db
    .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'`)
    .all() as Array<{ name: string }>;
  const names = rows.map((row) => row.name);

  const fk = db.prepare('PRAGMA foreign_keys').get() as { foreign_keys?: number } | undefined;
  const foreignKeysOn = fk?.foreign_keys === 1;
  // Cannot change inside a transaction; off so drop order does not matter
  if (foreignKeysOn) db.exec('PRAGMA foreign_keys = OFF');
  try {
    db.exec('BEGIN');
    try {
      for (const name of names) {
        // IF EXISTS: dropping a virtual table also drops its shadow tables
        db.exec(`DROP TABLE IF EXISTS ${quoteIdentifier(name)}`);
      }
      db.exec('COMMIT');
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  } finally {
    if (foreignKeysOn) db.exec('PRAGMA foreign_keys = ON');
  }
  return names;
}
