/**
 * ADR-016: hybrid dense + BM25 conversation retrieval.
 */
import { describe, expect, it } from 'vitest';
import {
  Bm25Index,
  dotScores,
  fuseHybrid,
  rankGroups,
  ranksOf,
  RETRIEVAL_LIMITS,
  tokenize,
} from '../../src/search/conversation-retrieval';

describe('tokenize', () => {
  it('lower-cases, drops stopwords and single letters, keeps digits, stems long words', () => {
    expect(tokenize('I graduated with a Business Administration degree in 2019, Q4')).toEqual([
      'graduat',
      'business',
      'administration',
      'degree',
      '2019',
      'q4',
    ]);
  });

  it('handles non-Latin letters', () => {
    expect(tokenize('Zürich 東京 café')).toEqual(['zürich', '東京', 'café']);
  });

  it('caps tokens and characters on hostile input, in linear time', () => {
    const huge = 'word '.repeat(2_000_000);
    const t0 = performance.now();
    const toks = tokenize(huge);
    const ms = performance.now() - t0;
    expect(toks.length).toBeLessThanOrEqual(RETRIEVAL_LIMITS.maxTokensPerDoc);
    expect(ms).toBeLessThan(500);
    const t1 = performance.now();
    tokenize('a'.repeat(5_000_000) + '!');
    expect(performance.now() - t1).toBeLessThan(500);
  });
});

describe('Bm25Index', () => {
  const docs = [
    ['coffee', 'shop', 'morning'],
    ['hiking', 'trip', 'yosemite', 'hiking'],
    ['tax', 'return', 'deadline'],
  ];
  const idx = new Bm25Index(docs);

  it('scores matching documents above zero and others at zero', () => {
    const s = idx.score(['hiking']);
    expect(s[1]).toBeGreaterThan(0);
    expect(s[0]).toBe(0);
    expect(s[2]).toBe(0);
  });

  it('rewards rarer terms more (idf)', () => {
    const multi = new Bm25Index([['a1', 'common'], ['b1', 'common'], ['c1', 'rare']]);
    const s = multi.score(['common', 'rare']);
    expect(s[2]).toBeGreaterThan(s[0]);
  });

  it('counts a repeated query term once', () => {
    expect(Array.from(idx.score(['hiking', 'hiking']))).toEqual(Array.from(idx.score(['hiking'])));
  });

  it('handles an empty corpus', () => {
    expect(new Bm25Index([]).score(['x']).length).toBe(0);
  });
});

describe('dotScores / ranksOf', () => {
  it('computes dot products and rejects a dimension mismatch', () => {
    expect(Array.from(dotScores([1, 0], [[1, 0], [0, 1], [0.5, 0.5]]))).toEqual([1, 0, 0.5]);
    expect(() => dotScores([1, 0], [[1, 0, 0]])).toThrow(/dimension mismatch/);
  });

  it('ranks descending, breaks ties by index, and leaves entries at or under the floor unranked', () => {
    expect(Array.from(ranksOf([0.2, 0.9, 0.2]))).toEqual([2, 1, 3]);
    expect(Array.from(ranksOf([0, 3, 0], 0))).toEqual([Infinity, 1, Infinity]);
  });
});

describe('fuseHybrid', () => {
  it('with lexicalWeight 0 keeps the dense order exactly', () => {
    const dense = [0.1, 0.8, 0.5];
    const fused = fuseHybrid(dense, [9, 0, 0], { lexicalWeight: 0 });
    expect(Array.from(ranksOf(fused))).toEqual(Array.from(ranksOf(dense)));
  });

  it('lifts a document both rankers agree on above one only dense likes', () => {
    const dense = [0.9, 0.85, 0.1];
    const lexical = [0, 5, 0];
    const fused = fuseHybrid(dense, lexical);
    expect(fused[1]).toBeGreaterThan(fused[0]);
  });

  it('gives no lexical credit to documents with no term match', () => {
    const fused = fuseHybrid([0.5, 0.5], [0, 0], { rrfK: 60 });
    expect(fused[0]).toBeCloseTo(1 / 61);
    expect(fused[1]).toBeCloseTo(1 / 62);
  });
});

describe('rankGroups', () => {
  const groups = ['s1', 's1', 's2', 's3', 's3'];
  const scores = [0.2, 0.9, 0.5, 0.6, 0.6];

  it('max: a group scores as its best chunk', () => {
    expect(rankGroups(groups, scores).map(g => g.group)).toEqual(['s1', 's3', 's2']);
  });

  it('max-plus: further chunks add a decayed bonus', () => {
    const r = rankGroups(groups, scores, { aggregation: 'max-plus', tailWeight: 0.5 });
    expect(r[0]).toEqual({ group: 's1', score: 0.9 + 0.5 * 0.2 });
    expect(r[1]).toEqual({ group: 's3', score: 0.6 + 0.5 * 0.6 });
  });

  it('breaks ties by first appearance', () => {
    expect(rankGroups(['b', 'a'], [1, 1]).map(g => g.group)).toEqual(['b', 'a']);
  });
});
