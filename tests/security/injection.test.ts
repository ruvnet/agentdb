/**
 * AgentDB v2 Security Injection Test Suite
 *
 * Tests for:
 * - Vector input validation (NaN/Infinity)
 * - ID path traversal prevention
 * - Cypher injection prevention
 * - Metadata sanitization
 * - Graph query security
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  validateVector,
  validateVectorId,
  validateSearchOptions,
  validateHNSWParams,
  validateCypherParams,
  validateLabel,
  validateBatchSize,
  sanitizeMetadata,
  SECURITY_LIMITS,
} from '../../src/security/validation';
import { ValidationError } from '../../src/security/input-validation';

describe('AgentDB v2 Security: Vector Validation', () => {
  describe('validateVector', () => {
    it('should accept valid vectors', () => {
      const vector = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      expect(() => validateVector(vector, 4)).not.toThrow();

      const arrayVector = [1.0, -0.5, 0.0, 0.75];
      expect(() => validateVector(arrayVector, 4)).not.toThrow();
    });

    it('should reject NaN values', () => {
      const vector = new Float32Array([0.1, NaN, 0.3]);
      expect(() => validateVector(vector, 3)).toThrow(ValidationError);
      expect(() => validateVector(vector, 3)).toThrow(/NaN or Infinity not allowed/);
    });

    it('should reject Infinity values', () => {
      const vector = new Float32Array([0.1, Infinity, 0.3]);
      expect(() => validateVector(vector, 3)).toThrow(ValidationError);

      const negInf = new Float32Array([0.1, -Infinity, 0.3]);
      expect(() => validateVector(negInf, 3)).toThrow(ValidationError);
    });

    it('should reject dimension mismatch', () => {
      const vector = new Float32Array([0.1, 0.2, 0.3]);
      expect(() => validateVector(vector, 4)).toThrow(ValidationError);
      expect(() => validateVector(vector, 4)).toThrow(/expected 4, got 3/);
    });

    it('should reject extreme values', () => {
      const vector = new Float32Array([1e15, 0.2, 0.3]);
      expect(() => validateVector(vector, 3)).toThrow(/magnitude too large/);
    });

    it('should reject oversized vectors', () => {
      const dimension = SECURITY_LIMITS.MAX_DIMENSION + 1;
      expect(() => validateVector(new Float32Array(dimension), dimension))
        .toThrow(/must be between/);
    });

    it('should reject missing vectors', () => {
      expect(() => validateVector(null as any, 3)).toThrow(/is required/);
      expect(() => validateVector(undefined as any, 3)).toThrow(/is required/);
    });
  });
});

describe('AgentDB v2 Security: ID Validation', () => {
  describe('validateVectorId', () => {
    it('should accept valid IDs', () => {
      expect(validateVectorId('doc-123')).toBe('doc-123');
      expect(validateVectorId('user_abc_def')).toBe('user_abc_def');
      expect(validateVectorId('valid-id-12345')).toBe('valid-id-12345');
    });

    it('should reject path traversal attempts', () => {
      expect(() => validateVectorId('../../../etc/passwd')).toThrow(ValidationError);
      expect(() => validateVectorId('..\\..\\windows\\system32')).toThrow(/path characters/);
      expect(() => validateVectorId('../../secret')).toThrow(/path characters/);
      expect(() => validateVectorId('id/../other')).toThrow(/path characters/);
    });

    it('should reject IDs with slashes', () => {
      expect(() => validateVectorId('path/to/file')).toThrow(/path characters/);
      expect(() => validateVectorId('windows\\path')).toThrow(/path characters/);
    });

    it('should reject IDs with dangerous characters', () => {
      // Cypher injection attempts
      expect(() => validateVectorId("id'; DROP DATABASE--")).toThrow(/dangerous characters/);
      expect(() => validateVectorId('id" OR 1=1')).toThrow(/dangerous characters/);
      expect(() => validateVectorId('id`malicious')).toThrow(/dangerous characters/);
      expect(() => validateVectorId('id{prop:value}')).toThrow(/dangerous characters/);
      expect(() => validateVectorId('id[0]')).toThrow(/dangerous characters/);
    });

    it('should reject control characters', () => {
      expect(() => validateVectorId('id\x00null')).toThrow(/control characters/);
      expect(() => validateVectorId('id\x01\x02')).toThrow(/control characters/);
      expect(() => validateVectorId('id\n\r')).toThrow(/control characters/);
    });

    it('should reject empty IDs', () => {
      expect(() => validateVectorId('')).toThrow(/cannot be empty/);
    });

    it('should reject oversized IDs', () => {
      const longId = 'a'.repeat(SECURITY_LIMITS.MAX_ID_LENGTH + 1);
      expect(() => validateVectorId(longId)).toThrow(/exceeds maximum length/);
    });

    it('should reject non-string IDs', () => {
      expect(() => validateVectorId(123 as any)).toThrow(/must be a string/);
      expect(() => validateVectorId(null as any)).toThrow(/must be a string/);
      expect(() => validateVectorId(undefined as any)).toThrow(/must be a string/);
    });
  });
});

describe('AgentDB v2 Security: Search Options Validation', () => {
  describe('validateSearchOptions', () => {
    it('should accept valid search options', () => {
      const options = validateSearchOptions({
        k: 10,
        threshold: 0.8,
        efSearch: 50,
      });

      expect(options.k).toBe(10);
      expect(options.threshold).toBe(0.8);
      expect(options.efSearch).toBe(50);
    });

    it('should reject invalid k values', () => {
      expect(() => validateSearchOptions({ k: 0 })).toThrow(/must be between/);
      expect(() => validateSearchOptions({ k: -1 })).toThrow(/must be between/);
      expect(() => validateSearchOptions({ k: SECURITY_LIMITS.MAX_K + 1 }))
        .toThrow(/must be between/);
      expect(() => validateSearchOptions({ k: 3.14 as any })).toThrow(/must be an integer/);
      expect(() => validateSearchOptions({ k: NaN })).toThrow(/must be an integer/);
    });

    it('should reject invalid threshold values', () => {
      expect(() => validateSearchOptions({ threshold: -0.1 }))
        .toThrow(/must be between 0 and 1/);
      expect(() => validateSearchOptions({ threshold: 1.5 }))
        .toThrow(/must be between 0 and 1/);
      expect(() => validateSearchOptions({ threshold: NaN }))
        .toThrow(/must be a finite number/);
      expect(() => validateSearchOptions({ threshold: Infinity }))
        .toThrow(/must be a finite number/);
    });

    it('should reject invalid efSearch values', () => {
      expect(() => validateSearchOptions({ efSearch: 0 }))
        .toThrow(/must be between/);
      expect(() => validateSearchOptions({ efSearch: SECURITY_LIMITS.MAX_EF_SEARCH + 1 }))
        .toThrow(/must be between/);
      expect(() => validateSearchOptions({ efSearch: 3.5 as any }))
        .toThrow(/must be an integer/);
    });

    it('should sanitize filter metadata', () => {
      const options = validateSearchOptions({
        k: 5,
        filter: { category: 'test', password: 'secret123' },
      });

      expect(options.filter).toBeDefined();
      expect(options.filter!.category).toBe('test');
      expect(options.filter!.password).toBeUndefined(); // Sanitized
    });

    it('should handle boolean flags', () => {
      const options = validateSearchOptions({
        includeMetadata: true,
        includeVectors: false,
      });

      expect(options.includeMetadata).toBe(true);
      expect(options.includeVectors).toBe(false);
    });
  });

  describe('validateHNSWParams', () => {
    it('should accept valid HNSW parameters', () => {
      const params = validateHNSWParams({
        M: 16,
        efConstruction: 200,
        efSearch: 50,
      });

      expect(params.M).toBe(16);
      expect(params.efConstruction).toBe(200);
      expect(params.efSearch).toBe(50);
    });

    it('should reject invalid M values', () => {
      expect(() => validateHNSWParams({ M: 1 })).toThrow(/must be between/);
      expect(() => validateHNSWParams({ M: SECURITY_LIMITS.MAX_M + 1 }))
        .toThrow(/must be between/);
      expect(() => validateHNSWParams({ M: 3.14 as any }))
        .toThrow(/must be an integer/);
    });

    it('should reject invalid efConstruction values', () => {
      expect(() => validateHNSWParams({ efConstruction: 3 }))
        .toThrow(/must be between/);
      expect(() => validateHNSWParams({ efConstruction: SECURITY_LIMITS.MAX_EF_CONSTRUCTION + 1 }))
        .toThrow(/must be between/);
    });
  });
});

describe('AgentDB v2 Security: Graph Query Security', () => {
  describe('validateCypherParams', () => {
    it('should accept valid Cypher parameters', () => {
      const params = validateCypherParams({
        name: 'test',
        age: 25,
        active: true,
      });

      expect(params.name).toBe('test');
      expect(params.age).toBe(25);
      expect(params.active).toBe(true);
    });

    it('should reject invalid parameter names', () => {
      expect(() => validateCypherParams({ 'invalid-name': 'value' }))
        .toThrow(/must be alphanumeric/);
      expect(() => validateCypherParams({ '123invalid': 'value' }))
        .toThrow(/must be alphanumeric/);
      expect(() => validateCypherParams({ 'name;DROP': 'value' }))
        .toThrow(/must be alphanumeric/);
      expect(() => validateCypherParams({ "name' OR '1'='1": 'value' }))
        .toThrow(/must be alphanumeric/);
    });

    it('should reject oversized parameter values', () => {
      const longValue = 'a'.repeat(10001);
      expect(() => validateCypherParams({ name: longValue }))
        .toThrow(/too long/);
    });

    it('should reject null bytes in parameters', () => {
      expect(() => validateCypherParams({ name: 'test\x00malicious' }))
        .toThrow(/null bytes/);
    });

    it('should reject too many parameters', () => {
      const params: Record<string, any> = {};
      for (let i = 0; i <= SECURITY_LIMITS.MAX_CYPHER_PARAMS; i++) {
        params[`param${i}`] = 'value';
      }

      expect(() => validateCypherParams(params))
        .toThrow(/Too many Cypher parameters/);
    });
  });

  describe('validateLabel', () => {
    it('should accept valid labels', () => {
      expect(validateLabel('User')).toBe('User');
      expect(validateLabel('Document_Type')).toBe('Document_Type');
      expect(validateLabel('_PrivateNode')).toBe('_PrivateNode');
    });

    it('should reject invalid label formats', () => {
      expect(() => validateLabel('123Invalid')).toThrow(/must be alphanumeric/);
      expect(() => validateLabel('Label-Name')).toThrow(/must be alphanumeric/);
      expect(() => validateLabel('Label Name')).toThrow(/must be alphanumeric/);
      expect(() => validateLabel("Label'; DROP")).toThrow(/must be alphanumeric/);
    });

    it('should reject empty labels', () => {
      expect(() => validateLabel('')).toThrow(/cannot be empty/);
    });

    it('should reject oversized labels', () => {
      const longLabel = 'a'.repeat(SECURITY_LIMITS.MAX_LABEL_LENGTH + 1);
      expect(() => validateLabel(longLabel)).toThrow(/exceeds maximum length/);
    });
  });
});

describe('AgentDB v2 Security: Metadata Sanitization', () => {
  describe('sanitizeMetadata', () => {
    it('should remove sensitive fields', () => {
      const metadata = {
        title: 'Document',
        password: 'secret123',
        apiKey: 'sk-abc123',
        token: 'bearer-xyz',
        secretKey: 'hidden',
      };

      const sanitized = sanitizeMetadata(metadata);

      expect(sanitized.title).toBe('Document');
      expect(sanitized.password).toBeUndefined();
      expect(sanitized.apiKey).toBeUndefined();
      expect(sanitized.token).toBeUndefined();
      expect(sanitized.secretKey).toBeUndefined();
    });

    it('should remove case-insensitive sensitive fields', () => {
      const metadata = {
        PASSWORD: 'secret',
        ApiKey: 'key123',
        Social_Security_Number: '123-45-6789',
        credit_card: '4111111111111111',
      };

      const sanitized = sanitizeMetadata(metadata);

      expect(sanitized.PASSWORD).toBeUndefined();
      expect(sanitized.ApiKey).toBeUndefined();
      expect(sanitized.Social_Security_Number).toBeUndefined();
      expect(sanitized.credit_card).toBeUndefined();
    });

    it('should reject oversized metadata', () => {
      const largeMetadata: Record<string, any> = {};
      const longString = 'a'.repeat(SECURITY_LIMITS.MAX_METADATA_SIZE);
      largeMetadata.data = longString;

      expect(() => sanitizeMetadata(largeMetadata))
        .toThrow(/exceeds maximum size/);
    });

    it('should reject overly long property keys', () => {
      const metadata: Record<string, any> = {};
      const longKey = 'a'.repeat(SECURITY_LIMITS.MAX_PROPERTY_KEY_LENGTH + 1);
      metadata[longKey] = 'value';

      expect(() => sanitizeMetadata(metadata))
        .toThrow(/property key exceeds maximum length/);
    });

    it('should handle empty metadata', () => {
      expect(sanitizeMetadata({})).toEqual({});
      expect(sanitizeMetadata(null as any)).toEqual({});
      expect(sanitizeMetadata(undefined as any)).toEqual({});
    });
  });
});

describe('AgentDB v2 Security: Batch Operations', () => {
  describe('validateBatchSize', () => {
    it('should accept valid batch sizes', () => {
      expect(validateBatchSize(1)).toBe(1);
      expect(validateBatchSize(100)).toBe(100);
      expect(validateBatchSize(SECURITY_LIMITS.MAX_BATCH_SIZE)).toBe(SECURITY_LIMITS.MAX_BATCH_SIZE);
    });

    it('should reject invalid batch sizes', () => {
      expect(() => validateBatchSize(0)).toThrow(/must be between 1 and/);
      expect(() => validateBatchSize(-1)).toThrow(/must be between 1 and/);
      expect(() => validateBatchSize(SECURITY_LIMITS.MAX_BATCH_SIZE + 1))
        .toThrow(/must be between 1 and/);
      expect(() => validateBatchSize(3.14)).toThrow(/must be an integer/);
      expect(() => validateBatchSize(NaN)).toThrow(/must be an integer/);
    });
  });
});

describe('AgentDB v2 Security: Real-World Attack Scenarios', () => {
  it('should prevent Cypher injection via IDs', () => {
    // Attacker tries to inject Cypher via ID
    const maliciousIds = [
      "id'; MATCH (n) DETACH DELETE n--",
      'id") RETURN n--',
      "id' OR '1'='1",
      'id{admin:true}',
    ];

    maliciousIds.forEach(id => {
      expect(() => validateVectorId(id)).toThrow(ValidationError);
    });
  });

  it('should prevent data exfiltration via metadata', () => {
    // Attacker tries to store sensitive data
    const maliciousMetadata = {
      data: 'normal',
      stolen_password: 'user123',
      api_key_leaked: 'sk-secret',
    };

    const sanitized = sanitizeMetadata(maliciousMetadata);
    expect(sanitized.data).toBe('normal');
    expect(sanitized.stolen_password).toBeUndefined();
    expect(sanitized.api_key_leaked).toBeUndefined();
  });

  it('should prevent DoS via oversized vectors', () => {
    // Attacker tries to consume memory
    const oversized = new Float32Array(SECURITY_LIMITS.MAX_DIMENSION + 1);
    expect(() => validateVector(oversized, oversized.length))
      .toThrow(/must be between/);
  });

  it('should prevent DoS via excessive batch size', () => {
    // Attacker tries to overload system
    expect(() => validateBatchSize(SECURITY_LIMITS.MAX_BATCH_SIZE * 10))
      .toThrow(/must be between/);
  });

  it('should prevent path traversal via IDs in file operations', () => {
    // Attacker tries to access arbitrary files
    const pathTraversalAttempts = [
      '../../../etc/passwd',
      '..\\..\\windows\\system32\\config\\sam',
      'id/../../../secret.txt',
      '..\\..\\.ssh\\id_rsa',
    ];

    pathTraversalAttempts.forEach(id => {
      expect(() => validateVectorId(id)).toThrow(/path characters/);
    });
  });
});
