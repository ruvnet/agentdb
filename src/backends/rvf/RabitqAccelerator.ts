/**
 * Optional RaBitQ candidate index for SqlJsRvfBackend (ADR-018).
 *
 * @ruvector/rabitq-wasm builds a 1-bit quantized index over a snapshot of the
 * vectors (`RabitqIndex.build(flat, dim, seed, rerankFactor)`) and answers
 * `search(query, k)` with positional ids. It has no incremental insert, so:
 *
 * - the index covers a snapshot; ids written or removed after the snapshot are
 *   "dirty": their index hits are dropped and their current vectors are scanned
 *   exactly on every query;
 * - the snapshot is rebuilt on the next query once the dirty set passes
 *   `max(256, 10%)` of the snapshot;
 * - candidates are rescored by the backend with its exact metric, so returned
 *   similarities are exact; only membership of the top-k is approximate.
 *
 * Only cosine (vectors normalised before indexing) and l2 are served; inner
 * product and filtered queries stay on the exact scan.
 */

import { optionalImport, pickExport } from '../../utils/optional-import.js';

export const RABITQ_PACKAGE = '@ruvector/rabitq-wasm';
/** Node build (CommonJS, reads its .wasm synchronously). The package root is the web build. */
const RABITQ_NODE_ENTRY = '@ruvector/rabitq-wasm/node/ruvector_rabitq_wasm.js';

export interface RabitqOptions {
  /** Smallest collection the index is used for (default 2000). */
  minVectors?: number;
  /** Candidates re-ranked exactly inside RaBitQ per result (default 20). */
  rerankFactor?: number;
  /**
   * Candidates requested per result, all rescored exactly (default 10).
   * Measured recall@10 on 5,000 clustered vectors: 1.000 for in-distribution
   * queries at any setting; 0.74-0.76 at 2, 0.96-0.98 at 10 and 1.000 at 50 for
   * queries unrelated to the data (near-tied neighbours).
   */
  overfetch?: number;
  seed?: number;
}

interface RabitqIndexInstance {
  search(query: Float32Array, k: number): Array<{ id: number; distance: number; free?: () => void }>;
  free(): void;
}

interface RabitqIndexStatic {
  build(vectors: Float32Array, dim: number, seed: bigint, rerankFactor: number): RabitqIndexInstance;
}

/** Loads the RaBitQ index class, or null when the package is not installed. */
export async function loadRabitq(): Promise<RabitqIndexStatic | null> {
  const isNode = typeof process !== 'undefined' && !!process.versions?.node;
  const mod = await optionalImport(isNode ? RABITQ_NODE_ENTRY : RABITQ_PACKAGE);
  const RabitqIndex = pickExport<RabitqIndexStatic>(mod, 'RabitqIndex');
  return RabitqIndex && typeof RabitqIndex.build === 'function' ? RabitqIndex : null;
}

export interface RabitqStats {
  enabled: boolean;
  indexed: number;
  dirty: number;
  builds: number;
  lastBuildMs: number;
}

export class RabitqAccelerator {
  private index: RabitqIndexInstance | null = null;
  private ids: string[] = [];
  private dirty = new Set<string>();
  private builds = 0;
  private lastBuildMs = 0;
  private readonly minVectors: number;
  private readonly rerankFactor: number;
  private readonly overfetch: number;
  private readonly seed: bigint;

  constructor(
    private readonly Rabitq: RabitqIndexStatic,
    private readonly dim: number,
    private readonly normalize: boolean,
    options: RabitqOptions = {}
  ) {
    this.minVectors = Math.max(1, options.minVectors ?? 2000);
    this.rerankFactor = Math.max(1, Math.floor(options.rerankFactor ?? 20));
    this.overfetch = Math.max(1, options.overfetch ?? 10);
    this.seed = BigInt(Math.floor(options.seed ?? 42));
  }

  /** A vector was written or removed. */
  touch(id: string): void {
    if (this.index) this.dirty.add(id);
  }

  reset(): void {
    this.index?.free();
    this.index = null;
    this.ids = [];
    this.dirty.clear();
  }

  stats(): RabitqStats {
    return {
      enabled: true,
      indexed: this.index ? this.ids.length : 0,
      dirty: this.dirty.size,
      builds: this.builds,
      lastBuildMs: this.lastBuildMs,
    };
  }

  /**
   * Candidate ids for a query, or null when the caller should scan exactly
   * (collection too small). Dirty ids that still exist are always included.
   */
  candidates(
    query: Float32Array,
    k: number,
    vectors: Map<string, { embedding: Float32Array }>
  ): Set<string> | null {
    if (vectors.size < this.minVectors) {
      if (this.index) this.reset();
      return null;
    }
    if (!this.index || this.dirty.size > Math.max(256, this.ids.length * 0.1)) {
      this.build(vectors);
    }

    const want = Math.min(this.ids.length, k * this.overfetch + this.dirty.size);
    const out = new Set<string>();
    const hits = this.index!.search(this.prepare(query), want);
    for (const hit of hits) {
      const id = this.ids[hit.id];
      hit.free?.();
      if (id !== undefined && !this.dirty.has(id)) out.add(id);
    }
    for (const id of this.dirty) if (vectors.has(id)) out.add(id);
    return out;
  }

  private build(vectors: Map<string, { embedding: Float32Array }>): void {
    const t0 = performance.now();
    this.reset();
    const ids = Array.from(vectors.keys());
    const flat = new Float32Array(ids.length * this.dim);
    ids.forEach((id, i) => flat.set(this.prepare(vectors.get(id)!.embedding), i * this.dim));
    this.index = this.Rabitq.build(flat, this.dim, this.seed, this.rerankFactor);
    this.ids = ids;
    this.builds++;
    this.lastBuildMs = performance.now() - t0;
  }

  private prepare(v: Float32Array): Float32Array {
    if (!this.normalize) return v;
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    if (s === 0) return v;
    const inv = 1 / Math.sqrt(s);
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
  }
}
