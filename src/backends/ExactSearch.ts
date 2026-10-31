/**
 * Exact-search sources over the controllers' SQL embedding tables (ADR-015).
 *
 * searchVectors() uses one of these to answer by exact cosine when the
 * approximate index is small or came back short.
 */
import type { IDatabaseConnection } from '../types/database.types.js';
import type { ExactSearchSource, SearchResult } from './VectorBackend.js';
import { TopK, squaredNorm, cosineWithQueryNorm, blobToFloat32 } from '../utils/top-k.js';

export interface SqlEmbeddingTable {
  /** Table holding one BLOB embedding per row (a trusted constant, not input) */
  table: string;
  /** Integer row-id column in that table */
  idColumn: string;
  /** Vector id the controller uses in the index for a row id */
  vectorId(rowId: number): string;
  /** Metadata attached to scan results (e.g. { patternId }) */
  metadata?(rowId: number): Record<string, unknown>;
  owns?(result: SearchResult): boolean;
  key?(result: SearchResult): string;
}

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function sqlExactSource(db: IDatabaseConnection, spec: SqlEmbeddingTable): ExactSearchSource {
  if (!IDENTIFIER.test(spec.table) || !IDENTIFIER.test(spec.idColumn)) {
    throw new Error(`Invalid embedding table spec ${spec.table}.${spec.idColumn}`);
  }
  return {
    owns: spec.owns,
    key: spec.key,
    count(): number {
      const row = db.prepare<{ n: number }>(`SELECT COUNT(*) AS n FROM ${spec.table}`).get();
      return Number(row?.n ?? 0);
    },
    scan(query: Float32Array, k: number, threshold: number): SearchResult[] {
      const rows = db
        .prepare<{ id: number; embedding: Uint8Array }>(
          `SELECT ${spec.idColumn} AS id, embedding FROM ${spec.table}`
        )
        .all();
      const queryNorm = Math.sqrt(squaredNorm(query));
      const top = new TopK<number>(k);
      for (const row of rows) {
        if (!row.embedding || row.embedding.byteLength !== query.length * 4) continue;
        const similarity = cosineWithQueryNorm(query, queryNorm, blobToFloat32(row.embedding));
        if (similarity >= threshold) top.push(similarity, Number(row.id));
      }
      return top.sorted().map(({ score, item }) => ({
        id: spec.vectorId(item),
        distance: 1 - score,
        similarity: score,
        metadata: spec.metadata?.(item),
      }));
    },
  };
}
