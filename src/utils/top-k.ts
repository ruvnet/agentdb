/**
 * Bounded top-k selection for brute-force vector scans (ADR-015).
 *
 * A full scan used to build a scored object per row, sort the whole array and
 * slice k. TopK keeps only the k best seen so far in a min-heap, so a scan of
 * N rows costs O(N log k) comparisons and allocates k entries instead of N.
 *
 * Ordering is identical to `rows.sort((a, b) => b.score - a.score).slice(0, k)`:
 * Array.prototype.sort is stable, so equal scores keep scan order. The heap
 * breaks ties the same way (earlier push wins).
 */

interface Entry<T> {
  score: number;
  index: number;
  item: T;
}

/** a ranks below b: lower score, or same score but pushed later */
function worse<T>(a: Entry<T>, b: Entry<T>): boolean {
  return a.score < b.score || (a.score === b.score && a.index > b.index);
}

export class TopK<T> {
  private readonly heap: Entry<T>[] = [];
  private readonly k: number;
  private pushed = 0;

  constructor(k: number) {
    this.k = Number.isFinite(k) ? Math.max(0, Math.floor(k)) : 0;
  }

  get size(): number {
    return this.heap.length;
  }

  /** Lowest score currently kept, or -Infinity while the heap is not full. */
  get floor(): number {
    return this.heap.length < this.k ? -Infinity : this.heap[0].score;
  }

  /**
   * Offer a candidate. `item` may be a thunk's input (a raw row); convert it
   * only after `sorted()` so losers are never materialised.
   */
  push(score: number, item: T): void {
    const entry: Entry<T> = { score, index: this.pushed++, item };
    if (this.k === 0) return;
    const heap = this.heap;
    if (heap.length < this.k) {
      heap.push(entry);
      this.siftUp(heap.length - 1);
    } else if (worse(heap[0], entry)) {
      heap[0] = entry;
      this.siftDown(0);
    }
  }

  /** Winners, best first (score desc, scan order on ties). */
  sorted(): Array<{ score: number; item: T }> {
    return [...this.heap]
      .sort((a, b) => (worse(a, b) ? 1 : worse(b, a) ? -1 : 0))
      .map(({ score, item }) => ({ score, item }));
  }

  private siftUp(i: number): void {
    const heap = this.heap;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!worse(heap[i], heap[parent])) break;
      [heap[i], heap[parent]] = [heap[parent], heap[i]];
      i = parent;
    }
  }

  private siftDown(i: number): void {
    const heap = this.heap;
    const n = heap.length;
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let smallest = i;
      if (left < n && worse(heap[left], heap[smallest])) smallest = left;
      if (right < n && worse(heap[right], heap[smallest])) smallest = right;
      if (smallest === i) break;
      [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
      i = smallest;
    }
  }
}

/** Sum of squares, accumulated in index order like the controllers' cosine. */
export function squaredNorm(a: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
  return sum;
}

/**
 * Cosine similarity with the query norm precomputed (`queryNorm` is
 * `Math.sqrt(squaredNorm(query))`). Bit-identical to the controllers'
 * `dot / (Math.sqrt(normA) * Math.sqrt(normB))`, which recomputed the query
 * norm for every row.
 */
export function cosineWithQueryNorm(query: Float32Array, queryNorm: number, b: Float32Array): number {
  let dot = 0;
  let normB = 0;
  for (let i = 0; i < query.length; i++) {
    dot += query[i] * b[i];
    normB += b[i] * b[i];
  }
  return dot / (queryNorm * Math.sqrt(normB));
}

/** Float32Array view over a BLOB column value (copied only if misaligned). */
export function blobToFloat32(blob: Uint8Array): Float32Array {
  if (blob.byteOffset % 4 !== 0) {
    return new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength));
  }
  return new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
}
