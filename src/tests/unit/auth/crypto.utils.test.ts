/**
 * Unit tests for crypto.utils.ts
 * Tests all cryptographic operations for security and correctness
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  generateSecureRandom,
  generateSecureRandomHex,
  generateSecureRandomBase64,
  hashPassword,
  verifyPassword,
  generateApiKey,
  hashApiKey,
  verifyApiKey,
  generateSessionId,
  generateJWTSecret,
  constantTimeCompare,
  generateHMAC,
  verifyHMAC,
  sha256,
  generateUUID,
  validatePasswordStrength,
} from '../../../utils/crypto.utils.js';

describe('Crypto Utils', () => {
  describe('generateSecureRandom', () => {
    it('should generate buffer of correct length', () => {
      const buffer = generateSecureRandom(32);
      expect(buffer).toBeInstanceOf(Buffer);
      expect(buffer.length).toBe(32);
    });

    it('should generate unique values', () => {
      const buffer1 = generateSecureRandom(32);
      const buffer2 = generateSecureRandom(32);
      expect(buffer1.equals(buffer2)).toBe(false);
    });

    it('should throw error for length < 16', () => {
      expect(() => generateSecureRandom(15)).toThrow('at least 16 bytes');
    });

    it('should throw error for length > 1024', () => {
      expect(() => generateSecureRandom(1025)).toThrow('too large');
    });
  });

  describe('generateSecureRandomHex', () => {
    it('should generate hex string of correct length', () => {
      const hex = generateSecureRandomHex(32);
      expect(hex).toMatch(/^[0-9a-f]+$/);
      expect(hex.length).toBe(64); // 32 bytes = 64 hex chars
    });

    it('should generate unique values', () => {
      const hex1 = generateSecureRandomHex(32);
      const hex2 = generateSecureRandomHex(32);
      expect(hex1).not.toBe(hex2);
    });
  });

  describe('generateSecureRandomBase64', () => {
    it('should generate base64url string', () => {
      const base64 = generateSecureRandomBase64(32);
      expect(base64).toMatch(/^[A-Za-z0-9_-]+$/);
      expect(base64.length).toBeGreaterThan(0);
    });

    it('should not contain +, /, or = characters', () => {
      const base64 = generateSecureRandomBase64(32);
      expect(base64).not.toMatch(/[+/=]/);
    });
  });

  describe('hashPassword', () => {
    it('should hash password successfully', async () => {
      const password = 'TestPassword123!';
      const hash = await hashPassword(password);

      expect(hash).toBeDefined();
      expect(hash.startsWith('$argon2id$')).toBe(true);
    });

    it('should generate different hashes for same password', async () => {
      const password = 'TestPassword123!';
      const hash1 = await hashPassword(password);
      const hash2 = await hashPassword(password);

      expect(hash1).not.toBe(hash2);
    });

    it('should throw error for password < 8 chars', async () => {
      await expect(hashPassword('short')).rejects.toThrow('at least 8 characters');
    });

    it('should throw error for password > 128 chars', async () => {
      const longPassword = 'a'.repeat(129);
      await expect(hashPassword(longPassword)).rejects.toThrow('exceeds maximum length');
    });

    it('should throw error for non-string password', async () => {
      await expect(hashPassword(null as any)).rejects.toThrow('non-empty string');
    });
  });

  describe('verifyPassword', () => {
    it('should verify correct password', async () => {
      const password = 'TestPassword123!';
      const hash = await hashPassword(password);
      const valid = await verifyPassword(password, hash);

      expect(valid).toBe(true);
    });

    it('should reject incorrect password', async () => {
      const password = 'TestPassword123!';
      const hash = await hashPassword(password);
      const valid = await verifyPassword('WrongPassword', hash);

      expect(valid).toBe(false);
    });

    it('should return false for invalid hash', async () => {
      const valid = await verifyPassword('password', 'invalid-hash');
      expect(valid).toBe(false);
    });

    it('should return false for empty inputs', async () => {
      expect(await verifyPassword('', 'hash')).toBe(false);
      expect(await verifyPassword('password', '')).toBe(false);
    });
  });

  describe('generateApiKey', () => {
    it('should generate live API key', () => {
      const result = generateApiKey('live');

      expect(result.key).toMatch(/^agdb_live_[0-9a-f]{64}$/);
      expect(result.hash).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should generate test API key', () => {
      const result = generateApiKey('test');

      expect(result.key).toMatch(/^agdb_test_[0-9a-f]{64}$/);
      expect(result.hash).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should generate unique keys', () => {
      const result1 = generateApiKey('live');
      const result2 = generateApiKey('live');

      expect(result1.key).not.toBe(result2.key);
      expect(result1.hash).not.toBe(result2.hash);
    });

    it('should default to live environment', () => {
      const result = generateApiKey();
      expect(result.key).toMatch(/^agdb_live_/);
    });
  });

  describe('hashApiKey', () => {
    it('should hash API key consistently', () => {
      const apiKey = 'agdb_live_' + '0'.repeat(64);
      const hash1 = hashApiKey(apiKey);
      const hash2 = hashApiKey(apiKey);

      expect(hash1).toBe(hash2);
      expect(hash1).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should throw error for invalid prefix', () => {
      expect(() => hashApiKey('invalid_key')).toThrow('Invalid API key format');
    });

    it('should throw error for non-string key', () => {
      expect(() => hashApiKey(null as any)).toThrow('non-empty string');
    });
  });

  describe('verifyApiKey', () => {
    it('should verify correct API key', () => {
      const { key, hash } = generateApiKey('live');
      const valid = verifyApiKey(key, hash);

      expect(valid).toBe(true);
    });

    it('should reject incorrect API key', () => {
      const { hash } = generateApiKey('live');
      const wrongKey = generateApiKey('live').key;

      const valid = verifyApiKey(wrongKey, hash);
      expect(valid).toBe(false);
    });

    it('should return false for empty inputs', () => {
      expect(verifyApiKey('', 'hash')).toBe(false);
      expect(verifyApiKey('key', '')).toBe(false);
    });

    it('should return false for invalid key format', () => {
      const valid = verifyApiKey('invalid', '0'.repeat(64));
      expect(valid).toBe(false);
    });
  });

  describe('generateSessionId', () => {
    it('should generate valid session ID', () => {
      const sessionId = generateSessionId();

      expect(sessionId).toMatch(/^[A-Za-z0-9_-]+$/);
      expect(sessionId.length).toBeGreaterThan(0);
    });

    it('should generate unique session IDs', () => {
      const id1 = generateSessionId();
      const id2 = generateSessionId();

      expect(id1).not.toBe(id2);
    });
  });

  describe('generateJWTSecret', () => {
    it('should generate valid JWT secret', () => {
      const secret = generateJWTSecret();

      expect(secret).toMatch(/^[A-Za-z0-9_-]+$/);
      expect(secret.length).toBeGreaterThan(64);
    });

    it('should generate unique secrets', () => {
      const secret1 = generateJWTSecret();
      const secret2 = generateJWTSecret();

      expect(secret1).not.toBe(secret2);
    });
  });

  describe('constantTimeCompare', () => {
    it('should return true for equal strings', () => {
      const str = 'test-string-12345';
      expect(constantTimeCompare(str, str)).toBe(true);
    });

    it('should return false for different strings', () => {
      expect(constantTimeCompare('string1', 'string2')).toBe(false);
    });

    it('should return false for different length strings', () => {
      expect(constantTimeCompare('short', 'longer-string')).toBe(false);
    });

    it('should return false for empty inputs', () => {
      expect(constantTimeCompare('', 'test')).toBe(false);
      expect(constantTimeCompare('test', '')).toBe(false);
    });
  });

  describe('generateHMAC', () => {
    it('should generate HMAC signature', () => {
      const data = 'test data';
      const secret = 'secret-key';
      const hmac = generateHMAC(data, secret);

      expect(hmac).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should be deterministic', () => {
      const data = 'test data';
      const secret = 'secret-key';
      const hmac1 = generateHMAC(data, secret);
      const hmac2 = generateHMAC(data, secret);

      expect(hmac1).toBe(hmac2);
    });

    it('should change with different data', () => {
      const secret = 'secret-key';
      const hmac1 = generateHMAC('data1', secret);
      const hmac2 = generateHMAC('data2', secret);

      expect(hmac1).not.toBe(hmac2);
    });

    it('should change with different secret', () => {
      const data = 'test data';
      const hmac1 = generateHMAC(data, 'secret1');
      const hmac2 = generateHMAC(data, 'secret2');

      expect(hmac1).not.toBe(hmac2);
    });
  });

  describe('verifyHMAC', () => {
    it('should verify correct HMAC', () => {
      const data = 'test data';
      const secret = 'secret-key';
      const hmac = generateHMAC(data, secret);

      expect(verifyHMAC(data, hmac, secret)).toBe(true);
    });

    it('should reject incorrect HMAC', () => {
      const data = 'test data';
      const secret = 'secret-key';
      const wrongHMAC = generateHMAC('wrong data', secret);

      expect(verifyHMAC(data, wrongHMAC, secret)).toBe(false);
    });

    it('should reject with wrong secret', () => {
      const data = 'test data';
      const hmac = generateHMAC(data, 'secret1');

      expect(verifyHMAC(data, hmac, 'secret2')).toBe(false);
    });
  });

  describe('sha256', () => {
    it('should generate SHA-256 hash', () => {
      const data = 'test data';
      const hash = sha256(data);

      expect(hash).toMatch(/^[0-9a-f]{64}$/);
    });

    it('should be deterministic', () => {
      const data = 'test data';
      const hash1 = sha256(data);
      const hash2 = sha256(data);

      expect(hash1).toBe(hash2);
    });

    it('should produce different hashes for different data', () => {
      const hash1 = sha256('data1');
      const hash2 = sha256('data2');

      expect(hash1).not.toBe(hash2);
    });
  });

  describe('generateUUID', () => {
    it('should generate valid UUID v4', () => {
      const uuid = generateUUID();

      expect(uuid).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    });

    it('should generate unique UUIDs', () => {
      const uuid1 = generateUUID();
      const uuid2 = generateUUID();

      expect(uuid1).not.toBe(uuid2);
    });
  });

  describe('validatePasswordStrength', () => {
    it('should accept strong password', () => {
      // No ascending run like '123': the policy rejects sequential characters.
      const result = validatePasswordStrength('StrongPass975!');

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.score).toBeGreaterThan(4);
    });

    it('should reject short password', () => {
      const result = validatePasswordStrength('Short1!');

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Password must be at least 8 characters');
    });

    it('should reject password without numbers', () => {
      const result = validatePasswordStrength('OnlyLetters');

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject password with only numbers', () => {
      const result = validatePasswordStrength('12345678');

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Password should contain letters');
    });

    it('should detect repeated characters', () => {
      const result = validatePasswordStrength('Passsss111');

      expect(result.errors).toContain('Password contains repeated characters');
    });

    it('should detect sequential characters', () => {
      const result = validatePasswordStrength('Password123');

      expect(result.errors).toContain('Password contains sequential characters');
    });

    it('should return error for empty password', () => {
      const result = validatePasswordStrength('');

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Password is required');
    });

    it('should give higher score for longer passwords', () => {
      const short = validatePasswordStrength('Pass123!');
      const long = validatePasswordStrength('LongerPassword123!');

      expect(long.score).toBeGreaterThan(short.score);
    });
  });
});
