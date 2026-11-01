/**
 * Shared helpers for the get/update/list/delete paths of the episode, pattern and skill controllers
 * (ADR-017). Patch semantics used by every update:
 *   - a field that is `undefined` (or absent) is left alone;
 *   - `null` clears a nullable column;
 *   - ids, creation timestamps and usage statistics are never written by a content update.
 */
import type { IDatabaseConnection } from '../types/database.types.js';
import { ValidationError } from '../security/input-validation.js';

/** Keys that turn a plain JSON merge into a prototype-pollution sink. */
const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

/** Upper bound on serialized metadata, matching SECURITY_LIMITS.MAX_METADATA_SIZE. */
export const MAX_METADATA_BYTES = 65_536;
const MAX_METADATA_DEPTH = 16;
export const MAX_TAGS = 64;
export const MAX_TAG_LENGTH = 128;
export const MAX_LIST_LIMIT = 500;

/** A ValidationError, so the MCP layer reports its message instead of a generic error. */
export class CrudValidationError extends ValidationError {
  constructor(message: string, field: string) {
    super(message, 'CRUD_VALIDATION_ERROR', field);
    this.name = 'CrudValidationError';
  }
}

function assertSafeValue(value: unknown, field: string, depth: number): void {
  if (depth > MAX_METADATA_DEPTH) throw new CrudValidationError(`${field} is nested too deeply`, field);
  if (value === null || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const v of value) assertSafeValue(v, field, depth + 1);
    return;
  }
  for (const key of Object.keys(value)) {
    if (FORBIDDEN_KEYS.has(key)) throw new CrudValidationError(`${field} contains a forbidden key "${key}"`, field);
    assertSafeValue((value as Record<string, unknown>)[key], field, depth + 1);
  }
}

/**
 * Validate a metadata object: a plain JSON object, no prototype-pollution keys at any depth,
 * and at most MAX_METADATA_BYTES once serialized.
 */
export function assertMetadata(value: unknown, field = 'metadata'): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new CrudValidationError(`${field} must be an object`, field);
  }
  assertSafeValue(value, field, 0);
  const size = Buffer.byteLength(JSON.stringify(value), 'utf8');
  if (size > MAX_METADATA_BYTES) {
    throw new CrudValidationError(`${field} exceeds ${MAX_METADATA_BYTES} bytes`, field);
  }
  return value as Record<string, unknown>;
}

/** Validate a tag list: strings, trimmed, non-empty, de-duplicated, bounded in count and length. */
export function assertTags(value: unknown, field = 'tags'): string[] {
  if (!Array.isArray(value)) throw new CrudValidationError(`${field} must be an array of strings`, field);
  if (value.length > MAX_TAGS) throw new CrudValidationError(`${field} has more than ${MAX_TAGS} entries`, field);
  const out: string[] = [];
  for (const t of value) {
    if (typeof t !== 'string') throw new CrudValidationError(`${field} must contain only strings`, field);
    const s = t.trim();
    if (!s) continue;
    if (s.length > MAX_TAG_LENGTH) throw new CrudValidationError(`${field} entry exceeds ${MAX_TAG_LENGTH} characters`, field);
    if (!out.includes(s)) out.push(s);
  }
  return out;
}

/**
 * Merge a metadata patch into the stored metadata. A `null` value in the patch removes that key.
 * The merged result is validated (size and keys), not just the patch.
 */
export function mergeMetadata(
  existing: Record<string, unknown> | null | undefined,
  patch: Record<string, unknown>,
): Record<string, unknown> {
  assertSafeValue(patch, 'metadata', 0);
  const merged: Record<string, unknown> = Object.create(null);
  for (const [k, v] of Object.entries(existing ?? {})) merged[k] = v;
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) delete merged[k];
    else merged[k] = v;
  }
  return assertMetadata({ ...merged });
}

/** Strict positive-integer id; rejects strings such as "12-notes" that parseInt would accept. */
export function toRecordId(id: unknown, field = 'id'): number {
  const n = typeof id === 'string' && /^\d+$/.test(id.trim()) ? Number(id) : id;
  if (typeof n !== 'number' || !Number.isSafeInteger(n) || n < 1) {
    throw new CrudValidationError(`${field} must be a positive integer`, field);
  }
  return n;
}

/** Clamp a list page request. */
export function toPage(limit: unknown, offset: unknown): { limit: number; offset: number } {
  const l = limit === undefined ? 50 : Number(limit);
  const o = offset === undefined ? 0 : Number(offset);
  if (!Number.isInteger(l) || l < 1 || l > MAX_LIST_LIMIT) {
    throw new CrudValidationError(`limit must be an integer between 1 and ${MAX_LIST_LIMIT}`, 'limit');
  }
  if (!Number.isInteger(o) || o < 0) throw new CrudValidationError('offset must be a non-negative integer', 'offset');
  return { limit: l, offset: o };
}

/**
 * Run `fn` atomically. Uses the driver's transaction() when it has one (better-sqlite3),
 * otherwise explicit BEGIN/COMMIT/ROLLBACK (sql.js).
 */
export function inTransaction<T>(db: IDatabaseConnection, fn: () => T): T {
  const tx = (db as unknown as { transaction?: (f: () => T) => () => T }).transaction;
  if (typeof tx === 'function') return tx.call(db, fn)();
  db.exec('BEGIN');
  try {
    const out = fn();
    db.exec('COMMIT');
    return out;
  } catch (err) {
    try { db.exec('ROLLBACK'); } catch { /* already rolled back */ }
    throw err;
  }
}

/** Serialize a Float32Array for a BLOB column without copying. */
export function embeddingBlob(embedding: Float32Array): Buffer {
  return Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
}
