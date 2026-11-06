/**
 * SQL Injection Security Test Suite
 *
 * Tests all SQL injection vulnerabilities that were fixed:
 * 1. Table name injection in BatchOperations
 * 2. PRAGMA command injection in db-fallback
 * 3. DELETE operation injection in MCP server
 * 4. Column name injection in WHERE/SET clauses
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  validateTableName,
  validateColumnName,
  validatePragmaCommand,
  validateSessionId,
  validateId,
  validateTimestamp,
  buildSafeWhereClause,
  buildSafeSetClause,
  ValidationError,
} from '../../src/security/input-validation';

describe('SQL Injection Prevention - Input Validation', () => {
  describe('Table Name Validation', () => {
    it('should accept valid table names', () => {
      expect(validateTableName('episodes')).toBe('episodes');
      expect(validateTableName('skills')).toBe('skills');
      expect(validateTableName('causal_edges')).toBe('causal_edges');
      expect(validateTableName(' EPISODES ')).toBe('episodes'); // Case-insensitive, trimmed
    });

    it('should reject SQL injection attempts in table names', () => {
      // SQL injection payloads
      const maliciousInputs = [
        "episodes; DROP TABLE users--",
        "episodes' OR '1'='1",
        "episodes UNION SELECT * FROM users",
        "episodes'; DELETE FROM episodes--",
        "../../../etc/passwd",
        "episodes`; DELETE FROM skills;--",
        "episodes\x00malicious",
      ];

      maliciousInputs.forEach(input => {
        expect(() => validateTableName(input)).toThrow(ValidationError);
      });
    });

    it('should reject non-whitelisted table names', () => {
      expect(() => validateTableName('users')).toThrow(ValidationError);
      expect(() => validateTableName('admin')).toThrow(ValidationError);
      expect(() => validateTableName('sqlite_master')).toThrow(ValidationError);
    });

    it('should reject empty or invalid table names', () => {
      expect(() => validateTableName('')).toThrow(ValidationError);
      expect(() => validateTableName(null as any)).toThrow(ValidationError);
      expect(() => validateTableName(undefined as any)).toThrow(ValidationError);
      expect(() => validateTableName(123 as any)).toThrow(ValidationError);
    });
  });

  describe('Column Name Validation', () => {
    it('should accept valid column names for episodes table', () => {
      expect(validateColumnName('episodes', 'id')).toBe('id');
      expect(validateColumnName('episodes', 'session_id')).toBe('session_id');
      expect(validateColumnName('episodes', 'task')).toBe('task');
      expect(validateColumnName('episodes', ' REWARD ')).toBe('reward'); // Case-insensitive
    });

    it('should reject SQL injection in column names', () => {
      const maliciousInputs = [
        "id; DROP TABLE episodes--",
        "id' OR '1'='1",
        "id, (SELECT password FROM users)",
        "id UNION SELECT * FROM admin",
      ];

      maliciousInputs.forEach(input => {
        expect(() => validateColumnName('episodes', input)).toThrow(ValidationError);
      });
    });

    it('should reject non-whitelisted column names', () => {
      expect(() => validateColumnName('episodes', 'password')).toThrow(ValidationError);
      expect(() => validateColumnName('episodes', 'credit_card')).toThrow(ValidationError);
    });
  });

  describe('PRAGMA Command Validation', () => {
    it('should accept valid PRAGMA commands', () => {
      expect(validatePragmaCommand('journal_mode')).toBe('journal_mode');
      expect(validatePragmaCommand('journal_mode = WAL')).toBe('journal_mode = WAL');
      expect(validatePragmaCommand('cache_size')).toBe('cache_size');
      expect(validatePragmaCommand('page_size')).toBe('page_size');
      expect(validatePragmaCommand(' FOREIGN_KEYS ')).toBe('FOREIGN_KEYS');
    });

    it('should reject SQL injection in PRAGMA commands', () => {
      const maliciousInputs = [
        "journal_mode; DROP TABLE episodes--",
        "user_version; DELETE FROM skills--",
        "database_list; ATTACH 'malicious.db' AS evil",
        "compile_options'; DROP TABLE users--",
      ];

      maliciousInputs.forEach(input => {
        expect(() => validatePragmaCommand(input)).toThrow(ValidationError);
      });
    });

    it('should reject dangerous PRAGMA commands', () => {
      // These PRAGMAs could be used for attacks
      expect(() => validatePragmaCommand('database_list')).toThrow(ValidationError);
      expect(() => validatePragmaCommand('table_info')).toThrow(ValidationError);
      expect(() => validatePragmaCommand('schema_version')).toThrow(ValidationError);
    });

    it('should reject empty PRAGMA commands', () => {
      expect(() => validatePragmaCommand('')).toThrow(ValidationError);
      expect(() => validatePragmaCommand(null as any)).toThrow(ValidationError);
    });
  });

  describe('Session ID Validation', () => {
    it('should accept valid session IDs', () => {
      expect(validateSessionId('session-123')).toBe('session-123');
      expect(validateSessionId('user_session_abc')).toBe('user_session_abc');
      expect(validateSessionId('a1b2c3d4e5')).toBe('a1b2c3d4e5');
    });

    it('should reject SQL injection in session IDs', () => {
      const maliciousInputs = [
        "session' OR '1'='1",
        "session; DROP TABLE episodes--",
        "session' UNION SELECT * FROM users--",
        "session\x00malicious",
      ];

      maliciousInputs.forEach(input => {
        expect(() => validateSessionId(input)).toThrow(ValidationError);
      });
    });

    it('should reject session IDs with special characters', () => {
      expect(() => validateSessionId("session'; DELETE FROM episodes--")).toThrow(ValidationError);
      expect(() => validateSessionId("session OR 1=1")).toThrow(ValidationError);
      expect(() => validateSessionId("session<script>")).toThrow(ValidationError);
    });

    it('should reject overly long session IDs', () => {
      const longId = 'a'.repeat(256);
      expect(() => validateSessionId(longId)).toThrow(ValidationError);
    });
  });

  describe('ID Validation', () => {
    it('should accept valid numeric IDs', () => {
      expect(validateId(1)).toBe(1);
      expect(validateId(999)).toBe(999);
      expect(validateId('42')).toBe(42);
      expect(validateId(0)).toBe(0);
    });

    it('should reject SQL injection via numeric IDs', () => {
      expect(() => validateId("1 OR 1=1")).toThrow(ValidationError);
      expect(() => validateId("1; DROP TABLE episodes--")).toThrow(ValidationError);
      expect(() => validateId("1 UNION SELECT * FROM users")).toThrow(ValidationError);
    });

    it('should reject negative and non-integer IDs', () => {
      expect(() => validateId(-1)).toThrow(ValidationError);
      expect(() => validateId(3.14)).toThrow(ValidationError);
      expect(() => validateId(NaN)).toThrow(ValidationError);
      expect(() => validateId(Infinity)).toThrow(ValidationError);
    });
  });

  describe('Timestamp Validation', () => {
    it('should accept valid timestamps', () => {
      const now = Math.floor(Date.now() / 1000);
      expect(validateTimestamp(now)).toBe(now);
      expect(validateTimestamp(1609459200)).toBe(1609459200); // 2021-01-01
    });

    it('should reject SQL injection via timestamps', () => {
      expect(() => validateTimestamp("1609459200 OR 1=1")).toThrow(ValidationError);
      expect(() => validateTimestamp("1609459200; DROP TABLE episodes--")).toThrow(ValidationError);
    });

    it('should reject unreasonable timestamps', () => {
      expect(() => validateTimestamp(0)).toThrow(ValidationError); // Before 2000
      expect(() => validateTimestamp(5000000000)).toThrow(ValidationError); // After 2100
      expect(() => validateTimestamp(-1)).toThrow(ValidationError);
    });
  });

  describe('Safe WHERE Clause Building', () => {
    it('should build safe parameterized WHERE clauses', () => {
      const result = buildSafeWhereClause('episodes', {
        session_id: 'test-session',
        success: 1,
      });

      expect(result.clause).toBe('session_id = ? AND success = ?');
      expect(result.values).toEqual(['test-session', 1]);
    });

    it('should prevent SQL injection in WHERE clauses', () => {
      // Malicious table name
      expect(() => buildSafeWhereClause("episodes'; DROP TABLE users--", { id: 1 }))
        .toThrow(ValidationError);

      // Malicious column name
      expect(() => buildSafeWhereClause('episodes', { "id' OR '1'='1": 1 }))
        .toThrow(ValidationError);
    });

    it('should reject empty conditions', () => {
      expect(() => buildSafeWhereClause('episodes', {})).toThrow(ValidationError);
      expect(() => buildSafeWhereClause('episodes', null as any)).toThrow(ValidationError);
    });
  });

  describe('Safe SET Clause Building', () => {
    it('should build safe parameterized SET clauses', () => {
      const result = buildSafeSetClause('episodes', {
        reward: 0.95,
        success: 1,
      });

      expect(result.clause).toBe('reward = ?, success = ?');
      expect(result.values).toEqual([0.95, 1]);
    });

    it('should prevent SQL injection in SET clauses', () => {
      // Malicious column name
      expect(() => buildSafeSetClause('episodes', { "reward = 1; DROP TABLE users--": 0.5 }))
        .toThrow(ValidationError);
    });

    it('should reject empty updates', () => {
      expect(() => buildSafeSetClause('episodes', {})).toThrow(ValidationError);
    });
  });
});

describe('SQL Injection Prevention - Real-World Attack Scenarios', () => {
  describe('Attack Scenario 1: Table Name Injection', () => {
    it('should prevent deletion of wrong tables', () => {
      // Attacker tries to delete from users table instead of episodes
      expect(() => validateTableName("users; DROP TABLE episodes--")).toThrow(ValidationError);
      expect(() => validateTableName("episodes' UNION SELECT * FROM users--")).toThrow(ValidationError);
    });
  });

  describe('Attack Scenario 2: PRAGMA Database Manipulation', () => {
    it('should prevent database file attachment', () => {
      // Attacker tries to attach malicious database
      expect(() => validatePragmaCommand("database_list; ATTACH 'evil.db' AS attack--"))
        .toThrow(ValidationError);
    });

    it('should prevent schema information leakage', () => {
      // Attacker tries to get table structure
      expect(() => validatePragmaCommand("table_info(users)")).toThrow(ValidationError);
    });
  });

  describe('Attack Scenario 3: DELETE Bypass', () => {
    it('should prevent unauthorized deletions', () => {
      // Attacker tries to delete all records
      expect(() => validateSessionId("' OR '1'='1")).toThrow(ValidationError);
      expect(() => validateId("1 OR 1=1")).toThrow(ValidationError);
    });
  });

  describe('Attack Scenario 4: Data Exfiltration', () => {
    it('should prevent UNION-based data exfiltration', () => {
      expect(() => validateColumnName('episodes', "id UNION SELECT password FROM users"))
        .toThrow(ValidationError);
    });
  });

  describe('Attack Scenario 5: Second-Order Injection', () => {
    it('should sanitize stored data used in queries', () => {
      // Even if malicious data was stored, it should be validated when used
      const storedMaliciousSessionId = "session'; DELETE FROM episodes--";
      expect(() => validateSessionId(storedMaliciousSessionId)).toThrow(ValidationError);
    });
  });
});

describe('SQL Injection Prevention - Error Handling', () => {
  it('should not leak sensitive information in error messages', () => {
    try {
      validateTableName("admin");
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      const validationError = error as ValidationError;
      // Error message should be informative but not leak database structure
      expect(validationError.message).not.toContain('sqlite');
      expect(validationError.message).not.toContain('password');
    }
  });

  it('should provide safe error messages', () => {
    try {
      validateTableName("users; DROP TABLE episodes--");
    } catch (error) {
      const validationError = error as ValidationError;
      expect(validationError.getSafeMessage()).toContain('Invalid input');
    }
  });
});
