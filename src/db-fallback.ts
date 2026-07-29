/**
 * Database System backed by SQLite.
 *
 * Resolution order (ruflo #2235 A):
 *   1. **better-sqlite3** if the optional peer is loadable — native, faster,
 *      what most callers actually want when they install the native module.
 *   2. **sql.js** (WASM) — pure-JS fallback, requires no build tools.
 *
 * Both implementations expose the same `db.prepare(sql).run/get/all(...)`
 * interface (the sql.js wrapper below was designed to mimic better-sqlite3),
 * so callers don't care which one served them.
 *
 * SECURITY: Fixed SQL injection vulnerabilities:
 * - PRAGMA commands validated against whitelist
 * - Removed eval() usage (replaced with async import)
 */

import { validatePragmaCommand, ValidationError } from './security/input-validation.js';
import { createHash, randomUUID } from 'node:crypto';
import * as fs from 'fs';
import * as path from 'path';

// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unused-vars -- sql.js Database has no shared TS type with better-sqlite3; kept for documentation
type Database = any;

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- cached impl (better-sqlite3 class or sql.js wrapper class)
let cachedImpl: any = null;
let cachedImplKind: 'better-sqlite3' | 'sql.js' | null = null;

export interface FileFingerprint {
  exists: boolean;
  device?: number;
  inode?: number;
  mode?: number;
  size?: number;
  mtimeMs?: number;
  ctimeMs?: number;
  sha256?: string;
}

export interface DatabaseFileState {
  database: FileFingerprint;
  wal: FileFingerprint;
  shm: FileFingerprint;
  journal: FileFingerprint;
}

export class SqlJsConcurrentModificationError extends Error {
  readonly code = 'AGENTDB_SQLJS_CONCURRENT_MODIFICATION';

  constructor(filename: string) {
    super(
      `Refusing to overwrite "${filename}" because it changed after this sql.js instance opened. ` +
      'Another process may have written to the database; reopen it before retrying.'
    );
    this.name = 'SqlJsConcurrentModificationError';
  }
}

function fingerprintFile(filename: string): FileFingerprint {
  let fd: number | undefined;
  try {
    fd = fs.openSync(filename, 'r');
    const before = fs.fstatSync(fd);
    const data = fs.readFileSync(fd);
    const after = fs.fstatSync(fd);
    if (
      before.dev !== after.dev ||
      before.ino !== after.ino ||
      before.size !== after.size ||
      before.mtimeMs !== after.mtimeMs ||
      before.ctimeMs !== after.ctimeMs
    ) {
      throw new SqlJsConcurrentModificationError(filename);
    }
    return {
      exists: true,
      device: after.dev,
      inode: after.ino,
      mode: after.mode,
      size: after.size,
      mtimeMs: after.mtimeMs,
      ctimeMs: after.ctimeMs,
      sha256: createHash('sha256').update(data).digest('hex'),
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { exists: false };
    throw error;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

export function captureDatabaseFileState(filename: string): DatabaseFileState {
  return {
    database: fingerprintFile(filename),
    wal: fingerprintFile(`${filename}-wal`),
    shm: fingerprintFile(`${filename}-shm`),
    journal: fingerprintFile(`${filename}-journal`),
  };
}

export function loadSqlJsDatabaseFile(
  filename: string
): { data: Uint8Array | null; state: DatabaseFileState } {
  const state = captureDatabaseFileState(filename);
  if (
    (state.wal.exists && state.wal.size! > 0) ||
    (state.journal.exists && state.journal.size! > 0)
  ) {
    throw new Error(
      `Refusing to open "${filename}" with sql.js while an active SQLite WAL or journal exists. ` +
      'Checkpoint/close the other SQLite writer first.'
    );
  }
  if (!state.database.exists) return { data: null, state };

  // fingerprintFile hashes the exact bytes read from a stable descriptor.
  // Read once more and verify the digest so the returned bytes and baseline
  // are guaranteed to describe the same database generation.
  const data = fs.readFileSync(filename);
  const digest = createHash('sha256').update(data).digest('hex');
  if (digest !== state.database.sha256) {
    throw new SqlJsConcurrentModificationError(filename);
  }
  if (
    data.length < 16 ||
    data.subarray(0, 16).toString('utf8') !== 'SQLite format 3\u0000'
  ) {
    throw new Error(`Refusing to open invalid SQLite database file "${filename}"`);
  }
  return { data: new Uint8Array(data), state };
}

function sameDatabaseFileState(left: DatabaseFileState, right: DatabaseFileState): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function fsyncDirectory(directory: string): void {
  let fd: number | undefined;
  try {
    fd = fs.openSync(directory, 'r');
    fs.fsyncSync(fd);
  } catch (error) {
    // Some platforms/filesystems do not permit opening or syncing directories.
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== 'EINVAL' && code !== 'EPERM' && code !== 'EISDIR') throw error;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

export function persistSqlJsDatabase(
  filename: string,
  data: Uint8Array,
  expectedState: DatabaseFileState
): DatabaseFileState {
  const directory = path.dirname(filename);
  fs.mkdirSync(directory, { recursive: true });
  if (fs.existsSync(filename) && fs.lstatSync(filename).isSymbolicLink()) {
    throw new Error(`Refusing to replace symbolic-link database path "${filename}"`);
  }

  const lockPath = `${filename}.agentdb.lock`;
  let lockFd: number | undefined;
  let tempPath: string | undefined;
  try {
    try {
      lockFd = fs.openSync(lockPath, 'wx', 0o600);
      fs.writeSync(lockFd, `${process.pid}\n`);
      fs.fsyncSync(lockFd);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EEXIST') {
        throw new SqlJsConcurrentModificationError(filename);
      }
      throw error;
    }

    if (!sameDatabaseFileState(captureDatabaseFileState(filename), expectedState)) {
      throw new SqlJsConcurrentModificationError(filename);
    }

    tempPath = path.join(directory, `.${path.basename(filename)}.${process.pid}.${randomUUID()}.tmp`);
    const tempFd = fs.openSync(tempPath, 'wx', 0o600);
    try {
      fs.writeFileSync(tempFd, Buffer.from(data));
      if (expectedState.database.exists && expectedState.database.mode !== undefined) {
        fs.fchmodSync(tempFd, expectedState.database.mode);
      }
      fs.fsyncSync(tempFd);
    } finally {
      fs.closeSync(tempFd);
    }

    // Recheck after export/write so a non-cooperating SQLite writer cannot be
    // silently overwritten merely because it raced with the temporary write.
    if (!sameDatabaseFileState(captureDatabaseFileState(filename), expectedState)) {
      throw new SqlJsConcurrentModificationError(filename);
    }

    fs.renameSync(tempPath, filename);
    tempPath = undefined;
    fsyncDirectory(directory);
    return captureDatabaseFileState(filename);
  } finally {
    if (tempPath) {
      try { fs.unlinkSync(tempPath); } catch { /* best-effort cleanup */ }
    }
    if (lockFd !== undefined) {
      fs.closeSync(lockFd);
      try { fs.unlinkSync(lockPath); } catch { /* preserve the original error */ }
    }
  }
}

/**
 * Get the SQLite database implementation. Prefers native `better-sqlite3`
 * when loadable; falls back to WASM `sql.js` (no build tools required).
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- returns dynamically constructed class
export async function getDatabaseImplementation(): Promise<any> {
  if (cachedImpl) return cachedImpl;

  // 1. Try native better-sqlite3 (optional peer; same prepare/run API).
  if (!process.env.AGENTDB_FORCE_SQLJS) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic optional import
      const mod: any = await import('better-sqlite3');
      const BetterSqlite3 = mod.default ?? mod;
      if (typeof BetterSqlite3 === 'function') {
        // Importing the JS wrapper can succeed even when its native binding is
        // missing or incompatible with the current Node ABI. Probe an actual
        // in-memory connection before selecting it.
        const probe = new BetterSqlite3(':memory:');
        probe.close();
        cachedImpl = BetterSqlite3;
        cachedImplKind = 'better-sqlite3';
        console.error('✅ Using native better-sqlite3');
        return cachedImpl;
      }
    } catch {
      // Not installed or failed to load → fall through to sql.js.
    }
  }

  // 2. Fall back to sql.js (WASM, no build tools).
  try {
    console.error('✅ Using sql.js (WASM SQLite, no build tools required)');

    // sql.js requires async initialization
    const mod = await import('sql.js');
    const SQL = await mod.default();

    cachedImpl = createSqlJsWrapper(SQL);
    cachedImplKind = 'sql.js';
    return cachedImpl;
  } catch (error) {
    console.error('❌ Failed to initialize sql.js:', (error as Error).message);
    throw new Error(
      'Failed to initialize SQLite. Install one of:\n' +
      '  npm install better-sqlite3  # native, faster (recommended)\n' +
      '  npm install sql.js          # pure-JS fallback, no build tools'
    );
  }
}

/** Reset the cached implementation (intended for tests). */
export function _resetDatabaseImplementationForTests(): void {
  cachedImpl = null;
  cachedImplKind = null;
}

/**
 * Create a better-sqlite3 compatible wrapper around sql.js
 * This allows AgentDB to work (with reduced performance) without native compilation
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js SQL factory type
function createSqlJsWrapper(SQL: any) {
  return class SqlJsDatabase {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js Database instance
    private db: any;
    private filename: string;
    private fileState: DatabaseFileState | null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js Statement instances
    private activeStatements: Map<number, any> = new Map();
    // Cache wrappers by SQL text so repeated db.prepare(sameSql) returns the
    // same underlying sql.js Statement and doesn't add a new entry to
    // activeStatements. Without this, callers using the
    // db.prepare(sql).run(...)-and-discard pattern (idiomatic in
    // better-sqlite3, where V8 GCs the handle) leak forever under sql.js.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- prepared-statement wrapper objects
    private statementCache: Map<string, any> = new Map();
    private statementCounter: number = 0;
    private intervalId: NodeJS.Timeout | null = null;

    constructor(filename: string, _options?: unknown) {
      this.filename = filename;
      const loaded = filename === ':memory:' ? null : loadSqlJsDatabaseFile(filename);
      this.fileState = loaded?.state ?? null;

      // In-memory database
      if (filename === ':memory:') {
        this.db = new SQL.Database();
      } else {
        // File-based database - use safe fs module (no eval)
        if (loaded!.data) {
          this.db = new SQL.Database(loaded!.data);
        } else {
          this.db = new SQL.Database();
        }
      }

      // Warn if too many active statements (memory leak detection)
      this.intervalId = setInterval(() => {
        if (this.activeStatements.size > 50) {
          console.warn(`⚠️  Detected ${this.activeStatements.size} active SQL statements - possible memory leak`);
        }
      }, 10000);
    }

    prepare(sql: string) {
      // Reuse the cached wrapper for an identical SQL string. The cache owns
      // the underlying stmt's lifetime; finalize() is a no-op on a cached
      // wrapper so concurrent callers that all called prepare(sql) keep
      // working. close() drops everything.
      const cached = this.statementCache.get(sql);
      if (cached) return cached;

      const stmt = this.db.prepare(sql);
      let isFinalized = false;
      const stmtId = ++this.statementCounter;
      const self = this;

      // Track active statement
      this.activeStatements.set(stmtId, stmt);

      const evictOnError = () => {
        if (!isFinalized) {
          stmt.free();
          isFinalized = true;
          self.activeStatements.delete(stmtId);
          self.statementCache.delete(sql);
        }
      };

      const wrapper = {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        run: (...params: any[]) => {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            stmt.step();
            stmt.reset();

            return {
              changes: this.db.getRowsModified(),
              lastInsertRowid: this.db.exec('SELECT last_insert_rowid()')[0]?.values[0]?.[0] || 0
            };
          } catch (error) {
            evictOnError();
            throw error;
          }
        },

        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        get: (...params: any[]) => {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            const hasRow = stmt.step();

            if (!hasRow) {
              stmt.reset();
              return undefined;
            }

            const columns = stmt.getColumnNames();
            const values = stmt.get();
            stmt.reset();

            const result: Record<string, unknown> = {};
            columns.forEach((col: string, idx: number) => {
              result[col] = values[idx];
            });

            return result;
          } catch (error) {
            evictOnError();
            throw error;
          }
        },

        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        all: (...params: any[]) => {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            const results: Record<string, unknown>[] = [];

            while (stmt.step()) {
              const columns = stmt.getColumnNames();
              const values = stmt.get();

              const result: Record<string, unknown> = {};
              columns.forEach((col: string, idx: number) => {
                result[col] = values[idx];
              });

              results.push(result);
            }

            stmt.reset();
            return results;
          } catch (error) {
            evictOnError();
            throw error;
          }
        },

        finalize: () => {
          // No-op while cached: the wrapper is shared and finalize() must not
          // tear out a stmt another caller is about to use. close() handles
          // the real teardown.
        }
      };

      this.statementCache.set(sql, wrapper);
      return wrapper;
    }

    exec(sql: string) {
      return this.db.exec(sql);
    }

    save() {
      // Save to file if needed
      if (this.filename !== ':memory:') {
        try {
          const data = this.db.export();
          this.fileState = persistSqlJsDatabase(this.filename, data, this.fileState!);
        } catch (error) {
          console.error('❌ Could not save database to file:', (error as Error).message);
          throw error;
        }
      }
    }

    close() {
      let saveError: unknown;
      try {
        this.save();
      } catch (error) {
        saveError = error;
      }

      // Clear interval timer
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }

      // Free all active statements to prevent memory leaks
      for (const [, stmt] of this.activeStatements.entries()) {
        try {
          stmt.free();
        } catch {
          // Statement may already be freed
        }
      }
      this.activeStatements.clear();
      this.statementCache.clear();

      this.db.close();
      if (saveError) throw saveError;
    }

    pragma(pragma: string, _options?: unknown) {
      try {
        // SECURITY: Validate PRAGMA command against whitelist to prevent SQL injection
        const validatedPragma = validatePragmaCommand(pragma);

        // Execute validated PRAGMA
        const result = this.db.exec(`PRAGMA ${validatedPragma}`);
        return result[0]?.values[0]?.[0];
      } catch (error) {
        if (error instanceof ValidationError) {
          console.error(`❌ Invalid PRAGMA command: ${error.message}`);
          throw error;
        }
        throw error;
      }
    }

    transaction(fn: () => unknown) {
      // Return a function that executes the transaction when called
      // This matches better-sqlite3 API where transaction() returns a callable function
      return () => {
        try {
          this.db.exec('BEGIN TRANSACTION');
          const result = fn();
          this.db.exec('COMMIT');
          return result;
        } catch (error) {
          this.db.exec('ROLLBACK');
          throw error;
        }
      };
    }
  };
}

/**
 * Create a database instance using sql.js
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- returns dynamically constructed SqlJsDatabase
export async function createDatabase(filename: string, options?: unknown): Promise<any> {
  const DatabaseImpl = await getDatabaseImplementation();
  return new DatabaseImpl(filename, options);
}

/**
 * Wrap an EXISTING sql.js raw database with the better-sqlite3-compatible API.
 * Used by AgentDB unified mode to share one sql.js Database instance for both
 * vector (rvf) and relational tables in a single .rvf file.
 *
 * Unlike createDatabase(), this does NOT create a new SQL.Database — it wraps
 * the one already held by SqlJsRvfBackend.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js raw database and returned wrapper
export function wrapExistingSqlJsDatabase(rawDb: any, filename: string = ':memory:'): any {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js Statement instances
  const activeStatements = new Map<number, any>();
  // SQL-text cache; see SqlJsDatabase.statementCache for rationale.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- prepared-statement wrapper objects
  const statementCache = new Map<string, any>();
  let statementCounter = 0;
  const loaded = filename === ':memory:' ? null : loadSqlJsDatabaseFile(filename);
  let fileState = loaded?.state ?? null;
  if (loaded?.data) {
    const rawDigest = createHash('sha256').update(rawDb.export()).digest('hex');
    if (rawDigest !== fileState!.database.sha256) {
      throw new SqlJsConcurrentModificationError(filename);
    }
  }

  return {
    prepare(sql: string) {
      const cached = statementCache.get(sql);
      if (cached) return cached;

      const stmt = rawDb.prepare(sql);
      let isFinalized = false;
      const stmtId = ++statementCounter;
      activeStatements.set(stmtId, stmt);

      const evictOnError = () => {
        if (!isFinalized) {
          stmt.free();
          isFinalized = true;
          activeStatements.delete(stmtId);
          statementCache.delete(sql);
        }
      };

      const wrapper = {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        run(...params: any[]) {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            stmt.step();
            stmt.reset();
            return {
              changes: rawDb.getRowsModified(),
              lastInsertRowid: rawDb.exec('SELECT last_insert_rowid()')[0]?.values[0]?.[0] || 0,
            };
          } catch (error) {
            evictOnError();
            throw error;
          }
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        get(...params: any[]) {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            const hasRow = stmt.step();
            if (!hasRow) { stmt.reset(); return undefined; }
            const columns = stmt.getColumnNames();
            const values = stmt.get();
            stmt.reset();
            const result: Record<string, unknown> = {};
            columns.forEach((col: string, idx: number) => { result[col] = values[idx]; });
            return result;
          } catch (error) {
            evictOnError();
            throw error;
          }
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js bind accepts heterogeneous params
        all(...params: any[]) {
          if (isFinalized) throw new Error('Statement already finalized');
          try {
            stmt.bind(params);
            const results: Record<string, unknown>[] = [];
            while (stmt.step()) {
              const columns = stmt.getColumnNames();
              const values = stmt.get();
              const result: Record<string, unknown> = {};
              columns.forEach((col: string, idx: number) => { result[col] = values[idx]; });
              results.push(result);
            }
            stmt.reset();
            return results;
          } catch (error) {
            evictOnError();
            throw error;
          }
        },
        finalize() {
          // No-op while cached; close() handles teardown.
        },
      };

      statementCache.set(sql, wrapper);
      return wrapper;
    },

    exec(sql: string) {
      return rawDb.exec(sql);
    },

    save() {
      if (filename !== ':memory:') {
        const data = rawDb.export();
        fileState = persistSqlJsDatabase(filename, data, fileState!);
      }
    },

    close() {
      let saveError: unknown;
      try {
        this.save();
      } catch (error) {
        saveError = error;
      }
      for (const [, stmt] of activeStatements.entries()) {
        try { stmt.free(); } catch { /* already freed */ }
      }
      activeStatements.clear();
      statementCache.clear();
      rawDb.close();
      if (saveError) throw saveError;
    },

    pragma(pragma: string, _options?: unknown) {
      try {
        const validatedPragma = validatePragmaCommand(pragma);
        const result = rawDb.exec(`PRAGMA ${validatedPragma}`);
        return result[0]?.values[0]?.[0];
      } catch (error) {
        if (error instanceof ValidationError) {
          console.error(`Invalid PRAGMA command: ${error.message}`);
          throw error;
        }
        throw error;
      }
    },

    transaction(fn: () => unknown) {
      return () => {
        try {
          rawDb.exec('BEGIN TRANSACTION');
          const result = fn();
          rawDb.exec('COMMIT');
          return result;
        } catch (error) {
          rawDb.exec('ROLLBACK');
          throw error;
        }
      };
    },
  };
}

/**
 * Get information about current database implementation
 */
export function getDatabaseInfo(): {
  implementation: string;
  isNative: boolean;
  performance: 'high' | 'medium' | 'low';
  requiresBuildTools: boolean;
} {
  // Reports the currently-loaded backend; falls back to the sql.js description
  // before getDatabaseImplementation() has run.
  if (cachedImplKind === 'better-sqlite3') {
    return {
      implementation: 'better-sqlite3 (native)',
      isNative: true,
      performance: 'high',
      requiresBuildTools: true,
    };
  }
  return {
    implementation: 'sql.js (WASM)',
    isNative: false,
    performance: 'medium',
    requiresBuildTools: false,
  };
}
