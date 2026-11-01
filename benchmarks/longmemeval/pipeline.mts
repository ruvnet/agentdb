/**
 * Shared per-question pipeline for the LongMemEval harness (ADR-016): chunking, the embedding cache, and the
 * dense + BM25 scores every configuration starts from.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { windows, type Instance } from './load.mts';
import { Bm25Index, dotScores, tokenize } from '../../src/search/conversation-retrieval.ts';

export const keyOf = (text: string): string => createHash('sha1').update(text).digest('hex').slice(0, 20);
export const slug = (model: string): string => model.replace(/[^a-zA-Z0-9.-]+/g, '_');

export interface EmbeddingCache {
  dim: number;
  row(text: string): Float32Array;
}

export function loadCache(model: string, data: string): EmbeddingCache {
  const dir = join(data, 'emb', slug(model));
  const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
  const keys = readFileSync(join(dir, 'keys.txt'), 'utf8').split('\n').filter(Boolean);
  const buf = readFileSync(join(dir, 'vectors.f32'));
  const vec = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  // The embedder may be appending while this reads: use only rows that have both a key and a full vector.
  const complete = Math.min(keys.length, Math.floor(vec.length / meta.dim));
  const rows = new Map(keys.slice(0, complete).map((k, i) => [k, i]));
  return {
    dim: meta.dim,
    row(text: string): Float32Array {
      const r = rows.get(keyOf(text));
      if (r === undefined) throw new Error('missing embedding; run embed.mts first');
      return vec.subarray(r * meta.dim, (r + 1) * meta.dim);
    },
  };
}

export interface Prepared {
  texts: string[];
  groupOf: string[];
  sessionDate: Map<string, string>;
  dense: Float64Array;
  lexical: Float64Array;
}

export function prepare(it: Instance, cache: EmbeddingCache, queryPrefix = '', docPrefix = ''): Prepared {
  const groupOf: string[] = [];
  const texts: string[] = [];
  const sessionDate = new Map<string, string>();
  it.haystack_sessions.forEach((session, s) => {
    const id = it.haystack_session_ids[s];
    sessionDate.set(id, it.haystack_dates[s]);
    for (const t of session) for (const w of windows(t.content)) {
      groupOf.push(id);
      texts.push(w);
    }
  });
  const dense = dotScores(cache.row(queryPrefix + it.question), texts.map(t => cache.row(docPrefix + t)));
  const lexical = new Bm25Index(texts.map(t => tokenize(t))).score(tokenize(it.question));
  return { texts, groupOf, sessionDate, dense, lexical };
}
