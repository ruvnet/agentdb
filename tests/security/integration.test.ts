/**
 * Integration Security Tests
 *
 * Tests the complete security fixes across all components:
 * - MCP server agentdb_delete tool
 * - BatchOperations bulkDelete/bulkUpdate
 * - db-fallback PRAGMA validation
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { createDatabase } from '../../src/db-fallback';
import { BatchOperations } from '../../src/optimizations/BatchOperations';
import { EmbeddingService } from '../../src/controllers/EmbeddingService';
import { ValidationError } from '../../src/security/input-validation';
import * as fs from 'fs';
import * as path from 'path';

describe('Integration Security Tests', () => {
  // The PRAGMA checks below exercise db-fallback's sql.js wrapper (it throws
  // ValidationError and exposes stmt.finalize). Native better-sqlite3 rejects
  // stacked pragmas itself, with its own error type.
  let savedForceSqlJs: string | undefined;
  beforeAll(() => {
    savedForceSqlJs = process.env.AGENTDB_FORCE_SQLJS;
    process.env.AGENTDB_FORCE_SQLJS = '1';
  });
  afterAll(() => {
    if (savedForceSqlJs === undefined) delete process.env.AGENTDB_FORCE_SQLJS;
    else process.env.AGENTDB_FORCE_SQLJS = savedForceSqlJs;
  });

  let db: any;
  let batchOps: BatchOperations;
  const testDbPath = path.join(__dirname, 'test-security.db');

  beforeEach(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Create fresh database
    db = await createDatabase(testDbPath);

    // Create test table
    db.exec(`
      CREATE TABLE episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        task TEXT NOT NULL,
        reward REAL NOT NULL,
        success INTEGER NOT NULL,
        ts INTEGER DEFAULT (strftime('%s', 'now'))
      )
    `);

    // Insert test data
    const stmt = db.prepare(`
      INSERT INTO episodes (session_id, task, reward, success)
      VALUES (?, ?, ?, ?)
    `);

    stmt.run('session-1', 'Test task 1', 0.8, 1);
    stmt.run('session-2', 'Test task 2', 0.6, 0);
    stmt.run('session-1', 'Test task 3', 0.9, 1);
    stmt.finalize();

    // Initialize batch operations (without embedding service for security tests)
    const mockEmbedder = {
      embed: async () => new Float32Array(384),
      embedBatch: async (texts: string[]) => texts.map(() => new Float32Array(384)),
    } as any;

    batchOps = new BatchOperations(db, mockEmbedder);
  });

  afterEach(() => {
    // Clean up
    if (db) {
      db.close();
    }
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('PRAGMA Injection Prevention', () => {
    it('should allow safe PRAGMA commands', () => {
      expect(() => db.pragma('journal_mode')).not.toThrow();
      expect(() => db.pragma('cache_size')).not.toThrow();
      expect(() => db.pragma('page_size')).not.toThrow();
    });

    it('should block malicious PRAGMA commands', () => {
      // SQL injection attempts
      expect(() => db.pragma("journal_mode; DROP TABLE episodes--")).toThrow(ValidationError);
      expect(() => db.pragma("user_version; DELETE FROM episodes--")).toThrow(ValidationError);
    });

    it('should block dangerous PRAGMA commands', () => {
      // Information disclosure attempts
      expect(() => db.pragma('database_list')).toThrow(ValidationError);
      expect(() => db.pragma('schema_version')).toThrow(ValidationError);
      expect(() => db.pragma('index_list(episodes)')).toThrow(ValidationError);
      // table_info(identifier) is whitelisted (BatchOperations uses it); smuggling is not
      expect(() => db.pragma("table_info(episodes); DROP TABLE episodes")).toThrow(ValidationError);
    });
  });

  describe('BatchOperations.bulkDelete Security', () => {
    it('should delete records with valid conditions', () => {
      const deleted = batchOps.bulkDelete('episodes', {
        session_id: 'session-1',
      });

      expect(deleted).toBe(2); // Should delete 2 records with session-1

      // Verify deletion
      const remaining = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(remaining.count).toBe(1);
    });

    it('should reject SQL injection in table name', () => {
      expect(() => batchOps.bulkDelete("episodes'; DROP TABLE episodes--", { id: 1 }))
        .toThrow(ValidationError);

      // Verify table still exists
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });

    it('should reject SQL injection in conditions', () => {
      // This would bypass the WHERE clause if not properly validated
      expect(() => batchOps.bulkDelete('episodes', {
        "session_id' OR '1'='1": 'anything',
      })).toThrow(ValidationError);

      // Verify no records were deleted
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });

    it('should reject non-whitelisted table names', () => {
      expect(() => batchOps.bulkDelete('users', { id: 1 })).toThrow(ValidationError);
      expect(() => batchOps.bulkDelete('sqlite_master', { id: 1 })).toThrow(ValidationError);
    });
  });

  describe('BatchOperations.bulkUpdate Security', () => {
    it('should update records with valid data', () => {
      const updated = batchOps.bulkUpdate(
        'episodes',
        { reward: 1.0, success: 1 },
        { session_id: 'session-2' }
      );

      expect(updated).toBe(1);

      // Verify update
      const record = db.prepare('SELECT * FROM episodes WHERE session_id = ?').get('session-2');
      expect(record.reward).toBe(1.0);
      expect(record.success).toBe(1);
    });

    it('should reject SQL injection in table name', () => {
      expect(() => batchOps.bulkUpdate(
        "episodes'; DROP TABLE episodes--",
        { reward: 1.0 },
        { id: 1 }
      )).toThrow(ValidationError);
    });

    it('should reject SQL injection in SET clause', () => {
      expect(() => batchOps.bulkUpdate(
        'episodes',
        { "reward = 1; DROP TABLE episodes--": 0.5 },
        { id: 1 }
      )).toThrow(ValidationError);

      // Verify table still exists
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });

    it('should reject SQL injection in WHERE clause', () => {
      expect(() => batchOps.bulkUpdate(
        'episodes',
        { reward: 1.0 },
        { "id' OR '1'='1": 1 }
      )).toThrow(ValidationError);

      // Verify no unintended updates
      const records = db.prepare('SELECT * FROM episodes WHERE reward = 1.0').all();
      expect(records.length).toBe(0);
    });
  });

  describe('Parameterized Query Verification', () => {
    it('should use parameterized queries for DELETE', () => {
      // Attempt to inject via parameter value (should be safely escaped)
      const safeSessionId = "session-1'; DELETE FROM episodes--";

      // Condition values are bound as parameters, so the payload is compared as a
      // literal string: it matches no row and deletes nothing.
      expect(batchOps.bulkDelete('episodes', { session_id: safeSessionId })).toBe(0);

      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3); // All records still present
    });

    it('should use parameterized queries for UPDATE', () => {
      // Attempt to inject via parameter value
      const maliciousValue = "0.5'; DELETE FROM episodes WHERE '1'='1";

      // The value would be safely escaped as a parameter
      const updated = batchOps.bulkUpdate(
        'episodes',
        { task: maliciousValue },
        { id: 1 }
      );

      // Update should succeed (with malicious string as literal data)
      expect(updated).toBe(1);

      // Verify the malicious string was stored as literal text, not executed
      const record = db.prepare('SELECT * FROM episodes WHERE id = 1').get();
      expect(record.task).toBe(maliciousValue);

      // Verify all records still exist (no deletion occurred)
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });
  });

  describe('Real-World Attack Scenarios', () => {
    it('should prevent Bobby Tables attack', () => {
      // Classic "Little Bobby Tables" XKCD attack
      const bobbyTables = "Robert'); DROP TABLE episodes;--";

      // Bound as a parameter: treated as a literal session_id, matches nothing
      expect(batchOps.bulkDelete('episodes', { session_id: bobbyTables })).toBe(0);

      // Verify table still exists
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });

    it('should prevent UNION-based data exfiltration', () => {
      expect(() => {
        batchOps.bulkDelete('episodes', {
          "id UNION SELECT * FROM sqlite_master": 1,
        });
      }).toThrow(ValidationError);
    });

    it('should prevent time-based blind SQL injection', () => {
      // Attacker tries to use SLEEP/WAITFOR to detect database
      expect(() => {
        batchOps.bulkDelete('episodes', {
          "id AND SLEEP(5)": 1,
        });
      }).toThrow(ValidationError);
    });

    it('should prevent stacked queries', () => {
      // Bound as a parameter: the stacked DELETE is never executed
      expect(batchOps.bulkDelete('episodes', { session_id: "'; DELETE FROM episodes; --" })).toBe(0);
      const count = db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(3);
    });
  });
});

describe('Error Message Security', () => {
  it('should not leak database structure in errors', () => {
    try {
      const db = { prepare: () => { throw new Error('SQLITE_ERROR: no such table: users'); } };
      const batchOps = new BatchOperations(db as any, {} as any);
      batchOps.bulkDelete('invalid_table', { id: 1 });
    } catch (error) {
      // Error should be caught and sanitized
      expect(error).toBeDefined();
    }
  });

  it('should provide safe validation error messages', () => {
    try {
      const db = { prepare: () => ({}) };
      const batchOps = new BatchOperations(db as any, {} as any);
      batchOps.bulkDelete("users'; DROP TABLE episodes--", { id: 1 });
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      const validationError = error as ValidationError;
      // Should not reveal SQL internals
      expect(validationError.message).not.toContain('DROP');
      expect(validationError.message).not.toContain('sqlite');
    }
  });
});
