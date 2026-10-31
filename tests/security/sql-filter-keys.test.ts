/**
 * Filter keys were spliced into SQL as `${key} = ?` (HNSWIndex.applyFilters,
 * WASMVectorSearch.findKNN) and metadata paths as `'$.${path}'`
 * (MetadataFilter.toSQL). Keys must now be identifiers naming real columns,
 * and JSON paths are bound as parameters.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { HNSWIndex } from '../../src/controllers/HNSWIndex.js';
import { WASMVectorSearch } from '../../src/controllers/WASMVectorSearch.js';
import { MetadataFilter } from '../../src/controllers/MetadataFilter.js';

const DIM = 8;
const INJECTION = '1=1 UNION SELECT';
// Syntactically valid once spliced in: `1=1 OR kind = ?` matches every row
const BYPASS = '1=1 OR kind';

function vec(seed: number): Float32Array {
  return Float32Array.from({ length: DIM }, (_, i) => Math.sin(seed + i));
}

describe('SQL filter key validation', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = new Database(':memory:');
    db.exec(`
      CREATE TABLE pattern_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        kind TEXT
      );
    `);
    const insert = db.prepare('INSERT INTO pattern_embeddings (pattern_id, embedding, kind) VALUES (?, ?, ?)');
    for (let i = 1; i <= 6; i++) {
      const v = vec(i);
      insert.run(i, Buffer.from(v.buffer), i % 2 ? 'odd' : 'even');
    }
  });

  afterEach(() => db.close());

  it('HNSWIndex rejects an injected filter key and still filters by real columns', async () => {
    const index = new HNSWIndex(db as any, { dimension: DIM, metric: 'cosine', maxElements: 100, persistIndex: false });
    await index.buildIndex('pattern_embeddings');

    await expect(index.search(vec(1), 6, { filters: { [INJECTION]: 1 } })).rejects.toThrow();
    await expect(index.search(vec(1), 6, { filters: { no_such_column: 1 } })).rejects.toThrow();
    await expect(index.search(vec(1), 6, { filters: { [BYPASS]: 'none' } })).rejects.toThrow();

    const odd = await index.search(vec(1), 6, { filters: { kind: 'odd' } });
    expect(odd.length).toBeGreaterThan(0);
    expect(odd.every((r) => r.id % 2 === 1)).toBe(true);
  });

  it('WASMVectorSearch rejects an injected filter key or table name', async () => {
    const search = new WASMVectorSearch(db as any, { enableWASM: false } as any);

    await expect(search.findKNN(vec(1), 3, 'pattern_embeddings', { filters: { [INJECTION]: 1 } })).rejects.toThrow();
    await expect(search.findKNN(vec(1), 3, 'pattern_embeddings; DROP TABLE x')).rejects.toThrow();
    await expect(search.findKNN(vec(1), 3, 'pattern_embeddings', { filters: { [BYPASS]: 'none' } })).rejects.toThrow();

    const odd = await search.findKNN(vec(1), 6, 'pattern_embeddings', { filters: { kind: 'odd' } });
    expect(odd.length).toBeGreaterThan(0);
    expect(odd.every((r: any) => Number(r.id) % 2 === 1)).toBe(true);
  });

  it('MetadataFilter.toSQL rejects injected fields and binds the JSON path', () => {
    expect(() => MetadataFilter.toSQL({ [INJECTION]: 1 } as any)).toThrow();
    expect(() => MetadataFilter.toSQL({ "metadata.a') OR 1=1 --": 1 } as any)).toThrow();

    const { where, params } = MetadataFilter.toSQL({ 'metadata.user.id': { $in: [1, 2] } } as any);
    expect(where).toBe('json_extract(metadata, ?) IN (?, ?)');
    expect(params).toEqual(['$.user.id', 1, 2]);
  });
});
