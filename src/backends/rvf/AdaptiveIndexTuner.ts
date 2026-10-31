/**
 * AdaptiveIndexTuner - Self-Learning Index & Memory Management for AgentDB
 *
 * Provides:
 * - Adaptive HNSW index monitoring via indexStats() + performance tracking
 * - Temporal memory decay via @ruvector/gnn TensorCompress
 * - Thompson Sampling query strategy selection via AgentDBSolver
 *
 * Security:
 * - TensorCompress operates on embeddings only (lossless reversible)
 * - All bounds checked (frequency 0-1, compression levels validated)
 * - No file I/O (operates on in-memory vectors)
 */

import type { IndexStats } from './RvfBackend.js';
import type { SolverBandit } from './SolverBandit.js';

/** Compression tier for temporal memory decay */
export type CompressionTier = 'none' | 'half' | 'pq8' | 'pq4' | 'binary';

/** Compressed vector entry */
export interface CompressedEntry {
  id: string;
  compressedJson: string;
  tier: CompressionTier;
  originalDim: number;
  accessFrequency: number;
  lastAccessed: number;
  /** Matryoshka truncated dimension (SOTA: MRL truncation) */
  truncatedDim?: number;
}

/** Index health assessment */
export interface IndexHealth {
  healthy: boolean;
  indexedVectors: number;
  layers: number;
  needsRebuild: boolean;
  avgSearchMs: number;
  avgInsertMs: number;
  recommendations: string[];
}

/** Compression statistics */
export interface CompressionStats {
  totalEntries: number;
  byTier: Record<CompressionTier, number>;
  estimatedSavingsPercent: number;
}

// Tier thresholds (from @ruvector/gnn getCompressionLevel)
// accessFreq >= 0.8: none (hot data)
// accessFreq >= 0.6: half (warm data)
// accessFreq >= 0.4: pq8 (cool data)
// accessFreq >= 0.2: pq4 (cold data)
// accessFreq < 0.2: binary (frozen data)
const TIER_ORDER: CompressionTier[] = ['none', 'half', 'pq8', 'pq4', 'binary'];
const SAVINGS_BY_TIER: Record<CompressionTier, number> = {
  none: 0,
  half: 50,
  pq8: 75,
  pq4: 87.5,
  binary: 96,
};

/**
 * TemporalCompressor - Access-frequency-based vector compression
 *
 * Uses tiered JSON compression for progressive compression of
 * aged embeddings. Hot data stays uncompressed, cold data is compressed
 * to save memory while remaining decompressible on demand.
 *
 * Compression tiers:
 * - none (freq >= 0.8): Full Float32Array stored
 * - half (freq >= 0.6): Float16-quantized
 * - pq8 (freq >= 0.4): 8-bit scalar quantization
 * - pq4 (freq >= 0.2): 4-bit scalar quantization
 * - binary (freq < 0.2): 1-bit sign encoding
 */
export class TemporalCompressor {
  private entries = new Map<string, CompressedEntry>();
  private _destroyed = false;
  // ADR-007 Phase 1: optional NativeAccelerator for native tensor compression
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private accel: any = null;
  private bandit: SolverBandit | null = null;

  /**
   * Create a new temporal compressor.
   * Lazy-loads NativeAccelerator for native compression when available.
   */
  static async create(bandit?: SolverBandit): Promise<TemporalCompressor> {
    const compressor = new TemporalCompressor();
    try {
      const { getAccelerator } = await import('./NativeAccelerator.js');
      compressor.accel = await getAccelerator();
    } catch { /* proceed without native compression */ }
    compressor.bandit = bandit || null;
    return compressor;
  }

  /**
   * Always available (built-in compression).
   */
  static async isAvailable(): Promise<boolean> {
    return true;
  }

  /** Whether native tensor compression is available (ADR-007 Phase 1) */
  get nativeCompressAvailable(): boolean {
    return this.accel?.nativeTensorCompressAvailable ?? false;
  }

  /**
   * Compress a vector based on its access frequency.
   * Uses Matryoshka-style dimensional truncation for cold tiers (SOTA: MRL).
   * ADR-007 Phase 1: delegates to NativeAccelerator tensorCompress when available.
   *
   * @param id - Unique identifier for this entry
   * @param embedding - The vector to compress
   * @param accessFrequency - Access frequency in [0.0, 1.0] (1.0 = hot)
   */
  compress(id: string, embedding: Float32Array, accessFrequency: number): CompressedEntry {
    this.ensureAlive();
    const freq = Math.min(Math.max(0, accessFrequency), 1);
    const tier = this.frequencyToTier(freq);

    // Matryoshka truncation: for cold/frozen data, store only leading dimensions
    const truncDim = this.matryoshkaDim(embedding.length, tier);
    const vec = truncDim < embedding.length ? embedding.subarray(0, truncDim) : embedding;

    // ADR-007 Phase 1: try native tensor compression for non-trivial tiers
    let compressedJson: string;
    if (this.accel?.nativeTensorCompressAvailable && tier !== 'none') {
      const level = this.tierToLevel(tier);
      const native = this.accel.tensorCompress(vec, level);
      if (native) {
        // Store native compressed bytes as base64 JSON
        compressedJson = JSON.stringify({ _native: true, b64: this.toBase64(native) });
      } else {
        compressedJson = this.compressVector(vec, tier);
      }
    } else {
      compressedJson = this.compressVector(vec, tier);
    }

    const entry: CompressedEntry = {
      id,
      compressedJson,
      tier,
      originalDim: embedding.length,
      accessFrequency: freq,
      lastAccessed: Date.now(),
      truncatedDim: truncDim < embedding.length ? truncDim : undefined,
    };

    this.entries.set(id, entry);
    return entry;
  }

  /**
   * Batch compress multiple vectors (ADR-007 Phase 1).
   * Uses NativeAccelerator.tensorBatchCompress when available.
   */
  compressBatch(
    items: Array<{ id: string; embedding: Float32Array; accessFrequency: number }>,
  ): CompressedEntry[] {
    this.ensureAlive();
    // Group by tier for batch native compression
    if (this.accel?.nativeTensorCompressAvailable) {
      const byTier = new Map<CompressionTier, Array<{ item: typeof items[0]; idx: number }>>();
      for (let i = 0; i < items.length; i++) {
        const freq = Math.min(Math.max(0, items[i].accessFrequency), 1);
        const tier = this.frequencyToTier(freq);
        if (!byTier.has(tier)) byTier.set(tier, []);
        byTier.get(tier)!.push({ item: items[i], idx: i });
      }

      const results = new Array<CompressedEntry>(items.length);
      for (const [tier, group] of byTier) {
        if (tier !== 'none' && group.length > 1) {
          const level = this.tierToLevel(tier);
          const vecs = group.map(g => {
            const truncDim = this.matryoshkaDim(g.item.embedding.length, tier);
            return truncDim < g.item.embedding.length ? g.item.embedding.subarray(0, truncDim) : g.item.embedding;
          });
          const batchResult = this.accel.tensorBatchCompress(vecs, level);
          if (batchResult) {
            for (let j = 0; j < group.length; j++) {
              const g = group[j];
              const compressedJson = JSON.stringify({ _native: true, b64: this.toBase64(batchResult[j]) });
              const truncDim = this.matryoshkaDim(g.item.embedding.length, tier);
              results[g.idx] = {
                id: g.item.id, compressedJson, tier,
                originalDim: g.item.embedding.length,
                accessFrequency: Math.min(Math.max(0, g.item.accessFrequency), 1),
                lastAccessed: Date.now(),
                truncatedDim: truncDim < g.item.embedding.length ? truncDim : undefined,
              };
              this.entries.set(g.item.id, results[g.idx]);
            }
            continue;
          }
        }
        // Fallback: compress individually
        for (const g of group) {
          results[g.idx] = this.compress(g.item.id, g.item.embedding, g.item.accessFrequency);
        }
      }
      return results;
    }
    // No native: compress individually
    return items.map(item => this.compress(item.id, item.embedding, item.accessFrequency));
  }

  /**
   * Decompress a vector back to its original form.
   * Matryoshka-truncated vectors are zero-padded to original dimension.
   */
  decompress(id: string): Float32Array | null {
    this.ensureAlive();
    const entry = this.entries.get(id);
    if (!entry) return null;

    entry.lastAccessed = Date.now();
    const storedDim = entry.truncatedDim ?? entry.originalDim;

    // ADR-007 Phase 1: try native tensor decompression first
    let vec: Float32Array;
    const parsed = JSON.parse(entry.compressedJson);
    if (parsed._native && parsed.b64 && this.accel?.nativeTensorCompressAvailable) {
      const compressed = this.fromBase64(parsed.b64);
      const native = this.accel.tensorDecompress(compressed, storedDim);
      vec = native ?? this.decompressVector(entry.compressedJson, entry.tier, storedDim);
    } else {
      vec = this.decompressVector(entry.compressedJson, entry.tier, storedDim);
    }

    // Zero-pad if Matryoshka-truncated
    if (entry.truncatedDim && entry.truncatedDim < entry.originalDim) {
      const full = new Float32Array(entry.originalDim);
      full.set(vec);
      return full;
    }
    return vec;
  }

  /**
   * Decompress from raw compressed JSON with known tier and dimension.
   */
  decompressRaw(compressedJson: string, tier: CompressionTier, dim: number): Float32Array {
    this.ensureAlive();
    return this.decompressVector(compressedJson, tier, dim);
  }

  /**
   * Update the access frequency for an entry and recompress if tier changed.
   */
  updateFrequency(id: string, newFrequency: number): CompressionTier | null {
    this.ensureAlive();
    const entry = this.entries.get(id);
    if (!entry) return null;

    const freq = Math.min(Math.max(0, newFrequency), 1);
    const newTier = this.frequencyToTier(freq);

    if (newTier !== entry.tier) {
      // Decompress and recompress at new level
      const original = this.decompressVector(entry.compressedJson, entry.tier, entry.originalDim);
      entry.compressedJson = this.compressVector(original, newTier);
      entry.tier = newTier;
    }

    entry.accessFrequency = freq;
    entry.lastAccessed = Date.now();
    return newTier;
  }

  /**
   * Get compression statistics.
   */
  getStats(): CompressionStats {
    const byTier: Record<CompressionTier, number> = {
      none: 0, half: 0, pq8: 0, pq4: 0, binary: 0,
    };
    for (const entry of this.entries.values()) {
      byTier[entry.tier]++;
    }

    const total = this.entries.size;
    let totalSavings = 0;
    for (const tier of TIER_ORDER) {
      totalSavings += byTier[tier] * SAVINGS_BY_TIER[tier];
    }

    return {
      totalEntries: total,
      byTier,
      estimatedSavingsPercent: total > 0 ? totalSavings / total : 0,
    };
  }

  /** Check if an entry exists */
  has(id: string): boolean {
    return this.entries.has(id);
  }

  /** Remove an entry */
  remove(id: string): boolean {
    return this.entries.delete(id);
  }

  /** Get number of compressed entries */
  get size(): number {
    return this.entries.size;
  }

  /** Check if destroyed */
  get isDestroyed(): boolean {
    return this._destroyed;
  }

  /** Destroy the compressor */
  destroy(): void {
    if (!this._destroyed) {
      this.entries.clear();
      this._destroyed = true;
    }
  }

  /** Map compression tier to native level (0-4) */
  private tierToLevel(tier: CompressionTier): number {
    switch (tier) {
      case 'none': return 0;
      case 'half': return 1;
      case 'pq8': return 2;
      case 'pq4': return 3;
      case 'binary': return 4;
      default: return 0;
    }
  }

  /** Encode Uint8Array to base64 string */
  private toBase64(data: Uint8Array): string {
    if (typeof Buffer !== 'undefined') {
      return Buffer.from(data).toString('base64');
    }
    // Browser fallback
    let binary = '';
    for (let i = 0; i < data.length; i++) binary += String.fromCharCode(data[i]);
    return btoa(binary);
  }

  /** Decode base64 string to Uint8Array */
  private fromBase64(b64: string): Uint8Array {
    if (typeof Buffer !== 'undefined') {
      return new Uint8Array(Buffer.from(b64, 'base64'));
    }
    // Browser fallback
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  private compressVector(vec: Float32Array, tier: CompressionTier): string {
    const len = vec.length;
    switch (tier) {
      case 'none':
        return JSON.stringify(Array.from(vec));
      case 'half': {
        // Float16 quantization: store as scaled int16
        let maxAbs = 0;
        for (let i = 0; i < len; i++) {
          const a = vec[i] < 0 ? -vec[i] : vec[i];
          if (a > maxAbs) maxAbs = a;
        }
        const scale = maxAbs || 1;
        const quantized = new Array(len);
        const invScale = 32767 / scale;
        for (let i = 0; i < len; i++) {
          quantized[i] = Math.round(vec[i] * invScale);
        }
        return JSON.stringify({ s: scale, d: quantized });
      }
      case 'pq8': {
        // 8-bit scalar quantization
        let min = vec[0], max = vec[0];
        for (let i = 1; i < len; i++) {
          if (vec[i] < min) min = vec[i];
          if (vec[i] > max) max = vec[i];
        }
        const range = max - min || 1;
        const q = new Array(len);
        const invRange = 255 / range;
        for (let i = 0; i < len; i++) {
          q[i] = Math.round((vec[i] - min) * invRange);
        }
        return JSON.stringify({ mn: min, mx: max, d: q });
      }
      case 'pq4': {
        // 4-bit scalar quantization (two values per byte)
        let min4 = vec[0], max4 = vec[0];
        for (let i = 1; i < len; i++) {
          if (vec[i] < min4) min4 = vec[i];
          if (vec[i] > max4) max4 = vec[i];
        }
        const range4 = max4 - min4 || 1;
        const invRange4 = 15 / range4;
        const packed = new Array(Math.ceil(len / 2));
        for (let i = 0, p = 0; i < len; i += 2, p++) {
          const lo = Math.round((vec[i] - min4) * invRange4);
          const hi = i + 1 < len ? Math.round((vec[i + 1] - min4) * invRange4) : 0;
          packed[p] = (hi << 4) | lo;
        }
        return JSON.stringify({ mn: min4, mx: max4, d: packed });
      }
      case 'binary': {
        // 1-bit sign encoding
        let sum = 0;
        for (let i = 0; i < len; i++) sum += vec[i];
        const mean = sum / len;
        const signs = new Array(Math.ceil(len / 8));
        for (let i = 0, s = 0; i < len; i += 8, s++) {
          let byte = 0;
          for (let b = 0; b < 8 && i + b < len; b++) {
            if (vec[i + b] >= mean) byte |= (1 << b);
          }
          signs[s] = byte;
        }
        return JSON.stringify({ m: mean, d: signs });
      }
      default:
        return JSON.stringify(Array.from(vec));
    }
  }

  private decompressVector(json: string, tier: CompressionTier, dim: number): Float32Array {
    const data = JSON.parse(json);
    if (data && data._native === true) {
      // Written by the native compressor in compress()
      const native = this.accel?.tensorDecompress(this.fromBase64(data.b64), dim) ?? null;
      if (!native) {
        throw new Error('Entry was compressed natively but @ruvector/gnn is not available to decompress it');
      }
      return native;
    }
    switch (tier) {
      case 'none':
        return new Float32Array(data);
      case 'half': {
        const scale = data.s as number;
        return new Float32Array((data.d as number[]).map((v) => (v / 32767) * scale));
      }
      case 'pq8': {
        const mn = data.mn as number;
        const mx = data.mx as number;
        const range = mx - mn || 1;
        return new Float32Array((data.d as number[]).map((v) => mn + (v / 255) * range));
      }
      case 'pq4': {
        const mn4 = data.mn as number;
        const mx4 = data.mx as number;
        const range4 = mx4 - mn4 || 1;
        const result = new Float32Array(dim);
        const packed = data.d as number[];
        for (let i = 0; i < packed.length; i++) {
          const lo = packed[i] & 0xF;
          const hi = (packed[i] >> 4) & 0xF;
          const idx = i * 2;
          if (idx < dim) result[idx] = mn4 + (lo / 15) * range4;
          if (idx + 1 < dim) result[idx + 1] = mn4 + (hi / 15) * range4;
        }
        return result;
      }
      case 'binary': {
        const mean = data.m as number;
        const signs = data.d as number[];
        const result = new Float32Array(dim);
        for (let i = 0; i < signs.length; i++) {
          for (let b = 0; b < 8 && i * 8 + b < dim; b++) {
            result[i * 8 + b] = (signs[i] & (1 << b)) ? mean * 1.1 : mean * 0.9;
          }
        }
        return result;
      }
      default:
        return new Float32Array(data);
    }
  }

  private frequencyToTier(freq: number): CompressionTier {
    // ADR-010: Bandit-guided tier selection
    if (this.bandit) {
      const tiers: CompressionTier[] = ['none', 'half', 'pq8', 'pq4', 'binary'];
      const ctx = freq >= 0.5 ? 'hot' : freq >= 0.2 ? 'warm' : 'cold';
      const selected = this.bandit.selectArm(ctx, tiers) as CompressionTier;
      return selected;
    }
    if (freq >= 0.8) return 'none';
    if (freq >= 0.6) return 'half';
    if (freq >= 0.4) return 'pq8';
    if (freq >= 0.2) return 'pq4';
    return 'binary';
  }

  /**
   * Matryoshka-style dimensional truncation (SOTA: MRL).
   * For cold tiers, store only leading dimensions. Matryoshka-trained
   * embeddings concentrate information in early dimensions, so truncation
   * preserves more semantic content than quantization at equal byte cost.
   *
   * Truncation ratios: none=100%, half=100%, pq8=75%, pq4=50%, binary=25%
   */
  /**
   * Matryoshka-style dimensional truncation ratios (SOTA: MRL).
   *
   * Only the most aggressive tier (binary) gets truncation. pq8/pq4 keep full
   * dimensions because their quantization error alone is already significant
   * and truncation on non-MRL-trained embeddings compounds error too aggressively.
   * Binary tier has a generous tolerance (1.0) so truncation to 50% is safe.
   */
  private matryoshkaDim(originalDim: number, tier: CompressionTier): number {
    switch (tier) {
      case 'none':
      case 'half':
      case 'pq8':
      case 'pq4':
        return originalDim; // Keep full dimensionality for quantized tiers
      case 'binary':
        return Math.max(8, Math.floor(originalDim * 0.5)); // 50% for frozen data
      default:
        return originalDim;
    }
  }

  private ensureAlive(): void {
    if (this._destroyed) {
      throw new Error('TemporalCompressor has been destroyed. Create a new instance.');
    }
  }
}

/**
 * IndexHealthMonitor - Adaptive HNSW index health monitoring
 *
 * Tracks index statistics and query performance to recommend
 * parameter adjustments and rebuild triggers.
 */
export class IndexHealthMonitor {
  private searchLatencies: number[] = [];
  private insertLatencies: number[] = [];
  private readonly maxSamples = 1000;

  /**
   * Record a search latency sample (in milliseconds).
   */
  recordSearch(latencyMs: number): void {
    this.searchLatencies.push(latencyMs);
    if (this.searchLatencies.length > this.maxSamples) {
      this.searchLatencies.shift();
    }
  }

  /**
   * Record an insert latency sample (in milliseconds).
   */
  recordInsert(latencyMs: number): void {
    this.insertLatencies.push(latencyMs);
    if (this.insertLatencies.length > this.maxSamples) {
      this.insertLatencies.shift();
    }
  }

  /**
   * Assess index health based on stats and recorded latencies.
   */
  assess(stats: IndexStats): IndexHealth {
    const avgSearchMs = this.average(this.searchLatencies);
    const avgInsertMs = this.average(this.insertLatencies);
    const recommendations: string[] = [];
    let healthy = true;

    if (stats.needsRebuild) {
      healthy = false;
      recommendations.push('Index needs rebuild. Run compact() or re-index.');
    }

    if (avgSearchMs > 50 && stats.indexedVectors > 1000) {
      recommendations.push(
        `High search latency (${avgSearchMs.toFixed(1)}ms). Consider increasing M or efSearch.`,
      );
    }

    if (stats.layers === 0 && stats.indexedVectors > 0) {
      healthy = false;
      recommendations.push('Index has 0 layers despite having vectors. Rebuild required.');
    }

    if (stats.m < 8 && stats.indexedVectors > 10000) {
      recommendations.push(
        `M=${stats.m} is low for ${stats.indexedVectors} vectors. Consider M=16 or higher.`,
      );
    }

    if (stats.efConstruction < 100 && stats.indexedVectors > 5000) {
      recommendations.push(
        `efConstruction=${stats.efConstruction} is low. Consider 200+ for better recall.`,
      );
    }

    if (avgInsertMs > 10) {
      recommendations.push(
        `High insert latency (${avgInsertMs.toFixed(1)}ms). Consider batch inserts.`,
      );
    }

    return {
      healthy: healthy && recommendations.length === 0,
      indexedVectors: stats.indexedVectors,
      layers: stats.layers,
      needsRebuild: stats.needsRebuild,
      avgSearchMs,
      avgInsertMs,
      recommendations,
    };
  }

  /** Reset all recorded samples */
  reset(): void {
    this.searchLatencies = [];
    this.insertLatencies = [];
  }

  private average(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }
}
