/**
 * AgentDB v2 Security Validation
 *
 * Comprehensive input validation for RuVector integration:
 * - Vector dimension and value validation (NaN/Infinity prevention)
 * - ID sanitization (path traversal prevention)
 * - Search options validation (bounds checking)
 * - Cypher query parameter validation (injection prevention)
 * - Metadata sanitization (sensitive data protection)
 */

import { ValidationError } from './input-validation.js';

/**
 * Security limits for AgentDB v2
 */
export const SECURITY_LIMITS = {
  MAX_VECTORS: 10_000_000,        // 10M vectors max
  MAX_DIMENSION: 4096,            // Maximum vector dimension
  MAX_BATCH_SIZE: 10_000,         // Batch insert limit
  MAX_K: 10_000,                  // Search result limit
  QUERY_TIMEOUT_MS: 30_000,       // 30s query timeout
  MAX_MEMORY_MB: 16_384,          // 16GB memory limit
  MAX_ID_LENGTH: 256,             // ID string length
  MAX_METADATA_SIZE: 65_536,      // 64KB metadata per vector
  MAX_LABEL_LENGTH: 128,          // Graph node label length
  MAX_PROPERTY_KEY_LENGTH: 128,   // Property key length
  MAX_CYPHER_PARAMS: 100,         // Maximum Cypher parameters
  MIN_DIMENSION: 1,               // Minimum vector dimension
  MIN_K: 1,                       // Minimum search results
  MIN_THRESHOLD: 0.0,             // Minimum similarity threshold
  MAX_THRESHOLD: 1.0,             // Maximum similarity threshold
  MIN_EF_SEARCH: 1,               // Minimum HNSW efSearch
  MAX_EF_SEARCH: 1000,            // Maximum HNSW efSearch
  MIN_EF_CONSTRUCTION: 4,         // Minimum HNSW efConstruction
  MAX_EF_CONSTRUCTION: 500,       // Maximum HNSW efConstruction
  MAX_M: 64,                      // Maximum HNSW M parameter
  MIN_M: 2,                       // Minimum HNSW M parameter
} as const;

/**
 * Validate vector embedding data
 * Prevents NaN, Infinity, and dimension mismatches
 */
export function validateVector(
  embedding: Float32Array | number[],
  expectedDim: number,
  fieldName: string = 'vector'
): void {
  // Validate dimension bounds
  if (expectedDim < SECURITY_LIMITS.MIN_DIMENSION || expectedDim > SECURITY_LIMITS.MAX_DIMENSION) {
    throw new ValidationError(
      `Invalid expected dimension: ${expectedDim} (must be between ${SECURITY_LIMITS.MIN_DIMENSION} and ${SECURITY_LIMITS.MAX_DIMENSION})`,
      'INVALID_DIMENSION',
      fieldName
    );
  }

  // Check if embedding exists
  if (!embedding) {
    throw new ValidationError(
      `${fieldName} is required`,
      'MISSING_VECTOR',
      fieldName
    );
  }

  // Validate dimension match
  if (embedding.length !== expectedDim) {
    throw new ValidationError(
      `Invalid ${fieldName} dimension: expected ${expectedDim}, got ${embedding.length}`,
      'DIMENSION_MISMATCH',
      fieldName
    );
  }

  // Validate each value for NaN and Infinity
  for (let i = 0; i < embedding.length; i++) {
    const value = embedding[i];

    if (!Number.isFinite(value)) {
      throw new ValidationError(
        `Invalid value in ${fieldName} at index ${i}: ${value} (NaN or Infinity not allowed)`,
        'INVALID_VECTOR_VALUE',
        `${fieldName}[${i}]`
      );
    }

    // Optional: Check for extreme values that might indicate errors
    if (Math.abs(value) > 1e10) {
      throw new ValidationError(
        `Extreme value in ${fieldName} at index ${i}: ${value} (magnitude too large)`,
        'EXTREME_VECTOR_VALUE',
        `${fieldName}[${i}]`
      );
    }
  }
}

/**
 * Validate vector ID
 * Prevents path traversal, excessive length, and malicious characters
 */
export function validateVectorId(id: string, fieldName: string = 'id'): string {
  if (typeof id !== 'string') {
    throw new ValidationError(
      `${fieldName} must be a string`,
      'INVALID_ID_TYPE',
      fieldName
    );
  }

  if (id.length === 0) {
    throw new ValidationError(
      `${fieldName} cannot be empty`,
      'EMPTY_ID',
      fieldName
    );
  }

  if (id.length > SECURITY_LIMITS.MAX_ID_LENGTH) {
    throw new ValidationError(
      `${fieldName} exceeds maximum length (${SECURITY_LIMITS.MAX_ID_LENGTH} chars)`,
      'ID_TOO_LONG',
      fieldName
    );
  }

  // Prevent path traversal attacks
  if (id.includes('..') || id.includes('/') || id.includes('\\')) {
    throw new ValidationError(
      `${fieldName} contains invalid path characters (., /, \\)`,
      'PATH_TRAVERSAL_ATTEMPT',
      fieldName
    );
  }

  // Prevent null bytes and control characters
  if (/[\x00-\x1F\x7F]/.test(id)) {
    throw new ValidationError(
      `${fieldName} contains control characters`,
      'INVALID_CHARACTERS',
      fieldName
    );
  }

  // Prevent potential Cypher injection via IDs used in graph queries
  const cypherDangerousChars = /['"`;{}[\]()]/;
  if (cypherDangerousChars.test(id)) {
    throw new ValidationError(
      `${fieldName} contains potentially dangerous characters for graph queries`,
      'DANGEROUS_ID_CHARACTERS',
      fieldName
    );
  }

  return id;
}

/**
 * Validate search options
 * Ensures k, threshold, and other parameters are within safe bounds
 */
export interface SearchOptions {
  k?: number;
  threshold?: number;
  efSearch?: number;
  filter?: Record<string, any>;
  includeMetadata?: boolean;
  includeVectors?: boolean;
}

export function validateSearchOptions(options: SearchOptions): SearchOptions {
  const validated: SearchOptions = {};

  // Validate k parameter
  if (options.k !== undefined) {
    if (!Number.isInteger(options.k)) {
      throw new ValidationError(
        'k must be an integer',
        'INVALID_K_TYPE',
        'k'
      );
    }

    if (options.k < SECURITY_LIMITS.MIN_K || options.k > SECURITY_LIMITS.MAX_K) {
      throw new ValidationError(
        `k must be between ${SECURITY_LIMITS.MIN_K} and ${SECURITY_LIMITS.MAX_K}`,
        'K_OUT_OF_BOUNDS',
        'k'
      );
    }

    validated.k = options.k;
  }

  // Validate threshold parameter
  if (options.threshold !== undefined) {
    if (typeof options.threshold !== 'number') {
      throw new ValidationError(
        'threshold must be a number',
        'INVALID_THRESHOLD_TYPE',
        'threshold'
      );
    }

    if (!Number.isFinite(options.threshold)) {
      throw new ValidationError(
        'threshold must be a finite number',
        'THRESHOLD_NOT_FINITE',
        'threshold'
      );
    }

    if (options.threshold < SECURITY_LIMITS.MIN_THRESHOLD ||
        options.threshold > SECURITY_LIMITS.MAX_THRESHOLD) {
      throw new ValidationError(
        `threshold must be between ${SECURITY_LIMITS.MIN_THRESHOLD} and ${SECURITY_LIMITS.MAX_THRESHOLD}`,
        'THRESHOLD_OUT_OF_BOUNDS',
        'threshold'
      );
    }

    validated.threshold = options.threshold;
  }

  // Validate efSearch parameter (HNSW specific)
  if (options.efSearch !== undefined) {
    if (!Number.isInteger(options.efSearch)) {
      throw new ValidationError(
        'efSearch must be an integer',
        'INVALID_EFSEARCH_TYPE',
        'efSearch'
      );
    }

    if (options.efSearch < SECURITY_LIMITS.MIN_EF_SEARCH ||
        options.efSearch > SECURITY_LIMITS.MAX_EF_SEARCH) {
      throw new ValidationError(
        `efSearch must be between ${SECURITY_LIMITS.MIN_EF_SEARCH} and ${SECURITY_LIMITS.MAX_EF_SEARCH}`,
        'EFSEARCH_OUT_OF_BOUNDS',
        'efSearch'
      );
    }

    validated.efSearch = options.efSearch;
  }

  // Validate filter object
  if (options.filter !== undefined) {
    if (typeof options.filter !== 'object' || options.filter === null) {
      throw new ValidationError(
        'filter must be an object',
        'INVALID_FILTER_TYPE',
        'filter'
      );
    }

    validated.filter = sanitizeMetadata(options.filter);
  }

  // Copy boolean flags
  if (options.includeMetadata !== undefined) {
    validated.includeMetadata = Boolean(options.includeMetadata);
  }

  if (options.includeVectors !== undefined) {
    validated.includeVectors = Boolean(options.includeVectors);
  }

  return validated;
}

/**
 * Validate HNSW index parameters
 */
export interface HNSWParams {
  M?: number;
  efConstruction?: number;
  efSearch?: number;
}

export function validateHNSWParams(params: HNSWParams): HNSWParams {
  const validated: HNSWParams = {};

  if (params.M !== undefined) {
    if (!Number.isInteger(params.M)) {
      throw new ValidationError(
        'M must be an integer',
        'INVALID_M_TYPE',
        'M'
      );
    }

    if (params.M < SECURITY_LIMITS.MIN_M || params.M > SECURITY_LIMITS.MAX_M) {
      throw new ValidationError(
        `M must be between ${SECURITY_LIMITS.MIN_M} and ${SECURITY_LIMITS.MAX_M}`,
        'M_OUT_OF_BOUNDS',
        'M'
      );
    }

    validated.M = params.M;
  }

  if (params.efConstruction !== undefined) {
    if (!Number.isInteger(params.efConstruction)) {
      throw new ValidationError(
        'efConstruction must be an integer',
        'INVALID_EFCONSTRUCTION_TYPE',
        'efConstruction'
      );
    }

    if (params.efConstruction < SECURITY_LIMITS.MIN_EF_CONSTRUCTION ||
        params.efConstruction > SECURITY_LIMITS.MAX_EF_CONSTRUCTION) {
      throw new ValidationError(
        `efConstruction must be between ${SECURITY_LIMITS.MIN_EF_CONSTRUCTION} and ${SECURITY_LIMITS.MAX_EF_CONSTRUCTION}`,
        'EFCONSTRUCTION_OUT_OF_BOUNDS',
        'efConstruction'
      );
    }

    validated.efConstruction = params.efConstruction;
  }

  if (params.efSearch !== undefined) {
    const searchOpts = validateSearchOptions({ efSearch: params.efSearch });
    validated.efSearch = searchOpts.efSearch;
  }

  return validated;
}

/**
 * Sanitize metadata to prevent sensitive data exposure
 * Removes fields that commonly contain secrets or PII
 */
export function sanitizeMetadata(
  metadata: Record<string, any>
): Record<string, any> {
  if (!metadata || typeof metadata !== 'object') {
    return {};
  }

  const sanitized = { ...metadata };

  // List of sensitive field names (case-insensitive)
  const sensitivePatterns = [
    /password/i,
    /secret/i,
    /token/i,
    /key/i,
    /credential/i,
    /api[_-]?key/i,
    /auth/i,
    /private/i,
    /ssn/i,
    /social[_-]?security/i,
    /credit[_-]?card/i,
    /card[_-]?number/i,
    /cvv/i,
    /pin/i,
  ];

  // Check metadata size
  const metadataStr = JSON.stringify(metadata);
  if (metadataStr.length > SECURITY_LIMITS.MAX_METADATA_SIZE) {
    throw new ValidationError(
      `Metadata exceeds maximum size (${SECURITY_LIMITS.MAX_METADATA_SIZE} bytes)`,
      'METADATA_TOO_LARGE',
      'metadata'
    );
  }

  // Remove sensitive fields
  for (const key of Object.keys(sanitized)) {
    if (sensitivePatterns.some(pattern => pattern.test(key))) {
      delete sanitized[key];
      console.warn(`[Security] Removed potentially sensitive metadata field: ${key}`);
    }

    // Validate property key length
    if (key.length > SECURITY_LIMITS.MAX_PROPERTY_KEY_LENGTH) {
      throw new ValidationError(
        `Metadata property key exceeds maximum length: ${key.substring(0, 50)}...`,
        'PROPERTY_KEY_TOO_LONG',
        'metadata'
      );
    }
  }

  return sanitized;
}

/**
 * Validate Cypher query parameters for graph operations
 * Prevents Cypher injection attacks
 */
export function validateCypherParams(
  params: Record<string, any>
): Record<string, any> {
  if (!params || typeof params !== 'object') {
    return {};
  }

  if (Object.keys(params).length > SECURITY_LIMITS.MAX_CYPHER_PARAMS) {
    throw new ValidationError(
      `Too many Cypher parameters (max ${SECURITY_LIMITS.MAX_CYPHER_PARAMS})`,
      'TOO_MANY_PARAMS',
      'cypherParams'
    );
  }

  const validated: Record<string, any> = {};

  for (const [key, value] of Object.entries(params)) {
    // Validate parameter key format (must be alphanumeric + underscore)
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(key)) {
      throw new ValidationError(
        `Invalid Cypher parameter name: ${key} (must be alphanumeric with underscores)`,
        'INVALID_PARAM_NAME',
        key
      );
    }

    // Validate string values aren't suspiciously long
    if (typeof value === 'string' && value.length > 10000) {
      throw new ValidationError(
        `Cypher parameter value too long: ${key}`,
        'PARAM_VALUE_TOO_LONG',
        key
      );
    }

    // Prevent null bytes in string parameters
    if (typeof value === 'string' && value.includes('\x00')) {
      throw new ValidationError(
        `Cypher parameter contains null bytes: ${key}`,
        'NULL_BYTE_IN_PARAM',
        key
      );
    }

    validated[key] = value;
  }

  return validated;
}

/**
 * Validate graph node label
 */
export function validateLabel(label: string): string {
  if (typeof label !== 'string') {
    throw new ValidationError(
      'Label must be a string',
      'INVALID_LABEL_TYPE',
      'label'
    );
  }

  if (label.length === 0) {
    throw new ValidationError(
      'Label cannot be empty',
      'EMPTY_LABEL',
      'label'
    );
  }

  if (label.length > SECURITY_LIMITS.MAX_LABEL_LENGTH) {
    throw new ValidationError(
      `Label exceeds maximum length (${SECURITY_LIMITS.MAX_LABEL_LENGTH} chars)`,
      'LABEL_TOO_LONG',
      'label'
    );
  }

  // Label must be alphanumeric + underscore (Cypher safe)
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(label)) {
    throw new ValidationError(
      'Label must be alphanumeric with underscores, starting with letter or underscore',
      'INVALID_LABEL_FORMAT',
      'label'
    );
  }

  return label;
}

/**
 * Validate batch size for bulk operations
 */
export function validateBatchSize(batchSize: number): number {
  if (!Number.isInteger(batchSize)) {
    throw new ValidationError(
      'Batch size must be an integer',
      'INVALID_BATCH_SIZE_TYPE',
      'batchSize'
    );
  }

  if (batchSize < 1 || batchSize > SECURITY_LIMITS.MAX_BATCH_SIZE) {
    throw new ValidationError(
      `Batch size must be between 1 and ${SECURITY_LIMITS.MAX_BATCH_SIZE}`,
      'BATCH_SIZE_OUT_OF_BOUNDS',
      'batchSize'
    );
  }

  return batchSize;
}

/**
 * Validate vector count doesn't exceed limits
 */
export function validateVectorCount(count: number): void {
  if (count > SECURITY_LIMITS.MAX_VECTORS) {
    throw new ValidationError(
      `Vector count exceeds maximum (${SECURITY_LIMITS.MAX_VECTORS})`,
      'TOO_MANY_VECTORS',
      'vectorCount'
    );
  }
}

/**
 * Safe logging that doesn't expose vectors or sensitive data
 */
export function safeLog(message: string, data?: any): void {
  if (!data) {
    console.error(message);
    return;
  }

  if (typeof data === 'object') {
    const safe = { ...data };

    // Remove sensitive fields
    delete safe.embedding;
    delete safe.vector;
    delete safe.metadata;
    delete safe.password;
    delete safe.token;
    delete safe.apiKey;

    // Truncate IDs if they're arrays
    if (Array.isArray(safe.ids) && safe.ids.length > 5) {
      safe.ids = `[${safe.ids.length} IDs...]`;
    }

    console.error(message, safe);
  } else {
    console.error(message, data);
  }
}
