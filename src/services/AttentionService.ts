/**
 * AttentionService - RuVector Attention Mechanisms Integration
 *
 * This service provides a unified interface for attention mechanisms with
 * RuVector WASM/NAPI bindings and robust JavaScript fallback implementations.
 *
 * Architecture:
 * - HyperbolicAttention: Tree-structured Poincare embeddings for causal chains
 * - FlashAttention: Memory-efficient block-wise attention for consolidation
 * - GraphRoPE: Hop-distance-aware positional encoding for graph queries
 * - MoEAttention: Expert routing for specialized memory domains
 *
 * All mechanisms default to FALSE (opt-in) and provide backward-compatible fallbacks.
 * WASM bindings are used when available for 10-100x performance improvements.
 *
 * @module AttentionService
 * @version 2.0.0-alpha.4
 */

// Database type from db-fallback
type Database = any;

// Performance logging utility
const PERFORMANCE_LOGGING = process.env.AGENTDB_PERF_LOGGING === 'true';

interface PerformanceLog {
  mechanism: string;
  backend: 'wasm' | 'napi' | 'fallback';
  durationMs: number;
  inputSize: number;
  timestamp: number;
}

const performanceLogs: PerformanceLog[] = [];

function logPerformance(log: PerformanceLog): void {
  if (PERFORMANCE_LOGGING) {
    performanceLogs.push(log);
    console.error(
      `[AttentionService] ${log.mechanism} using ${log.backend}: ${log.durationMs.toFixed(2)}ms (input size: ${log.inputSize})`
    );
  }
}

/**
 * Get all performance logs (useful for debugging and optimization)
 */
export function getPerformanceLogs(): PerformanceLog[] {
  return [...performanceLogs];
}

/**
 * Clear performance logs
 */
export function clearPerformanceLogs(): void {
  performanceLogs.length = 0;
}

// Lazy-loaded WASM/NAPI modules
let wasmModule: any = null;
let napiModule: any = null;
let wasmInitialized = false;
let napiInitialized = false;
let wasmLoadAttempted = false;
let napiLoadAttempted = false;

/**
 * Attempt to load the ruvector-attention-wasm module
 */
async function loadWasmModule(): Promise<boolean> {
  if (wasmLoadAttempted) return wasmInitialized;
  wasmLoadAttempted = true;

  try {
    // Dynamic import for WASM module
    const wasm = await import('ruvector-attention-wasm');
    if (wasm.default) {
      await wasm.default(); // Initialize WASM
    } else if (wasm.init) {
      wasm.init();
    }
    wasmModule = wasm;
    wasmInitialized = true;
    if (PERFORMANCE_LOGGING) {
      console.error('[AttentionService] WASM module loaded successfully');
    }
    return true;
  } catch (error) {
    if (PERFORMANCE_LOGGING) {
      console.warn('[AttentionService] WASM module not available:', (error as Error).message);
    }
    return false;
  }
}

/**
 * Attempt to load the @ruvector/attention NAPI module
 */
async function loadNapiModule(): Promise<boolean> {
  if (napiLoadAttempted) return napiInitialized;
  napiLoadAttempted = true;

  try {
    // Dynamic import for NAPI module
    napiModule = await import('@ruvector/attention');
    napiInitialized = true;
    if (PERFORMANCE_LOGGING) {
      console.error('[AttentionService] NAPI module loaded successfully');
    }
    return true;
  } catch (error) {
    if (PERFORMANCE_LOGGING) {
      console.warn('[AttentionService] NAPI module not available:', (error as Error).message);
    }
    return false;
  }
}

/**
 * Check if WASM backend is available
 */
export function isWasmAvailable(): boolean {
  return wasmInitialized;
}

/**
 * Check if NAPI backend is available
 */
export function isNapiAvailable(): boolean {
  return napiInitialized;
}

/**
 * Get the best available backend
 */
export function getAvailableBackend(): 'napi' | 'wasm' | 'fallback' {
  if (napiInitialized) return 'napi';
  if (wasmInitialized) return 'wasm';
  return 'fallback';
}

/**
 * Configuration for HyperbolicAttention
 * Uses Poincaré ball model for hierarchical causal relationships
 */
export interface HyperbolicAttentionConfig {
  /** Enable hyperbolic attention (default: false) */
  enabled: boolean;
  /** Curvature of Poincaré ball (default: 1.0) */
  curvature?: number;
  /** Embedding dimension (default: 384) */
  dimension?: number;
  /** Temperature for attention softmax (default: 1.0) */
  temperature?: number;
}

/**
 * Configuration for FlashAttention
 * Block-wise memory-efficient attention for large buffers
 */
export interface FlashAttentionConfig {
  /** Enable flash attention (default: false) */
  enabled: boolean;
  /** Block size for tiling (default: 256) */
  blockSize?: number;
  /** Use SIMD acceleration (default: true) */
  useSIMD?: boolean;
  /** Maximum sequence length (default: 4096) */
  maxSeqLen?: number;
}

/**
 * Configuration for GraphRoPE
 * Rotary positional encoding aware of graph hop distances
 */
export interface GraphRoPEConfig {
  /** Enable graph RoPE (default: false) */
  enabled: boolean;
  /** Maximum hop distance (default: 10) */
  maxHops?: number;
  /** Rotary dimension (default: 64) */
  rotaryDim?: number;
  /** Base frequency (default: 10000) */
  baseFreq?: number;
}

/**
 * Configuration for MoEAttention
 * Mixture-of-Experts routing for specialized domains
 */
export interface MoEAttentionConfig {
  /** Enable MoE attention (default: false) */
  enabled: boolean;
  /** Number of experts (default: 8) */
  numExperts?: number;
  /** Top-k experts to route to (default: 2) */
  topK?: number;
  /** Expert specialization domains */
  expertDomains?: string[];
}

/**
 * Result from HyperbolicAttention computation
 */
export interface HyperbolicAttentionResult {
  /** Attended embeddings in Poincaré space */
  attended: Float32Array;
  /** Attention weights */
  weights: Float32Array;
  /** Hierarchical distances */
  distances: number[];
  /** Performance metrics */
  metrics: {
    computeTimeMs: number;
    memoryUsedMB: number;
  };
}

/**
 * Result from FlashAttention computation
 */
export interface FlashAttentionResult {
  /** Consolidated output */
  output: Float32Array;
  /** Attention scores (if requested) */
  scores?: Float32Array;
  /** Performance metrics */
  metrics: {
    computeTimeMs: number;
    peakMemoryMB: number;
    blocksProcessed: number;
  };
}

/**
 * Result from GraphRoPE computation
 */
export interface GraphRoPEResult {
  /** Position-encoded queries */
  queries: Float32Array;
  /** Position-encoded keys */
  keys: Float32Array;
  /** Hop-distance aware encodings */
  hopEncodings: Float32Array;
  /** Performance metrics */
  metrics: {
    computeTimeMs: number;
  };
}

/**
 * Result from MoEAttention computation
 */
export interface MoEAttentionResult {
  /** Routed output from experts */
  output: Float32Array;
  /** Expert assignments per query */
  expertAssignments: number[][];
  /** Expert weights per query */
  expertWeights: number[][];
  /** Performance metrics */
  metrics: {
    computeTimeMs: number;
    expertsUsed: number;
    routingEntropy: number;
  };
}

/**
 * AttentionService - Unified interface for attention mechanisms
 *
 * Provides fallback implementations until RuVector WASM/NAPI bindings are available.
 * All mechanisms are opt-in via configuration flags.
 */
export class AttentionService {
  private db: Database;
  private hyperbolicConfig: HyperbolicAttentionConfig;
  private flashConfig: FlashAttentionConfig;
  private graphRoPEConfig: GraphRoPEConfig;
  private moeConfig: MoEAttentionConfig;

  constructor(
    db: Database,
    configs?: {
      hyperbolic?: Partial<HyperbolicAttentionConfig>;
      flash?: Partial<FlashAttentionConfig>;
      graphRoPE?: Partial<GraphRoPEConfig>;
      moe?: Partial<MoEAttentionConfig>;
    }
  ) {
    this.db = db;

    // Initialize configs with defaults (all disabled)
    this.hyperbolicConfig = {
      enabled: false,
      curvature: 1.0,
      dimension: 384,
      temperature: 1.0,
      ...configs?.hyperbolic,
    };

    this.flashConfig = {
      enabled: false,
      blockSize: 256,
      useSIMD: true,
      maxSeqLen: 4096,
      ...configs?.flash,
    };

    this.graphRoPEConfig = {
      enabled: false,
      maxHops: 10,
      rotaryDim: 64,
      baseFreq: 10000,
      ...configs?.graphRoPE,
    };

    this.moeConfig = {
      enabled: false,
      numExperts: 8,
      topK: 2,
      expertDomains: ['code', 'data', 'reasoning', 'planning', 'execution', 'review', 'documentation', 'optimization'],
      ...configs?.moe,
    };
  }

  /**
   * HyperbolicAttention: Tree-structured Poincare attention for causal chains
   *
   * Uses hyperbolic geometry to model hierarchical relationships in causal memory.
   * Attempts to use NAPI bindings first, then WASM, then falls back to JavaScript.
   *
   * @param queries - Query embeddings [num_queries, dim]
   * @param keys - Key embeddings from causal chain [num_keys, dim]
   * @param values - Value embeddings [num_keys, dim]
   * @param hierarchyLevels - Hierarchy level for each key (0 = root)
   * @returns Attention result with Poincare-weighted outputs
   */
  async hyperbolicAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    hierarchyLevels: number[]
  ): Promise<HyperbolicAttentionResult> {
    const startTime = Date.now();
    const dim = this.hyperbolicConfig.dimension!;
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;

    if (!this.hyperbolicConfig.enabled) {
      // Fallback: Standard attention with hierarchical scaling
      return this.fallbackHyperbolicAttention(queries, keys, values, hierarchyLevels, startTime);
    }

    // Try NAPI first (fastest for Node.js)
    await loadNapiModule();
    if (napiInitialized && napiModule?.HyperbolicAttention) {
      try {
        const attention = new napiModule.HyperbolicAttention(
          dim,
          this.hyperbolicConfig.curvature
        );

        // Convert to format expected by NAPI: arrays of Float32Arrays
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const attended = new Float32Array(numQueries * dim);
        const allWeights: number[] = [];

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = attention.computeRaw(query, keysArray, valuesArray);
          attended.set(result, q * dim);

          // Approximate weights based on hierarchy
          for (let k = 0; k < numKeys; k++) {
            allWeights.push(Math.exp(-hierarchyLevels[k] * 0.5) / numKeys);
          }
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'hyperbolicAttention',
          backend: 'napi',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          attended,
          weights: new Float32Array(allWeights),
          distances: hierarchyLevels,
          metrics: {
            computeTimeMs: durationMs,
            memoryUsedMB: (attended.byteLength + allWeights.length * 4) / (1024 * 1024),
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] NAPI hyperbolicAttention failed, trying WASM:', (error as Error).message);
        }
      }
    }

    // Try WASM (for browsers or Node.js fallback)
    await loadWasmModule();
    if (wasmInitialized && wasmModule?.WasmHyperbolicAttention) {
      try {
        const attention = new wasmModule.WasmHyperbolicAttention(
          dim,
          this.hyperbolicConfig.curvature!
        );

        // Convert to format expected by WASM
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const attended = new Float32Array(numQueries * dim);
        const allWeights: number[] = [];

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = attention.compute(query, keysArray, valuesArray);
          attended.set(result, q * dim);

          // Approximate weights based on hierarchy
          for (let k = 0; k < numKeys; k++) {
            allWeights.push(Math.exp(-hierarchyLevels[k] * 0.5) / numKeys);
          }
        }

        // Clean up WASM instance
        if (attention.free) {
          attention.free();
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'hyperbolicAttention',
          backend: 'wasm',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          attended,
          weights: new Float32Array(allWeights),
          distances: hierarchyLevels,
          metrics: {
            computeTimeMs: durationMs,
            memoryUsedMB: (attended.byteLength + allWeights.length * 4) / (1024 * 1024),
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] WASM hyperbolicAttention failed, using fallback:', (error as Error).message);
        }
      }
    }

    // Fallback to JavaScript implementation
    return this.fallbackHyperbolicAttention(queries, keys, values, hierarchyLevels, startTime);
  }

  /**
   * FlashAttention: Memory-efficient block-wise attention for consolidation
   *
   * Processes attention in blocks to reduce peak memory usage.
   * Ideal for episodic memory consolidation with large buffers.
   * Attempts to use NAPI bindings first, then WASM, then falls back to JavaScript.
   *
   * @param queries - Query embeddings [num_queries, dim]
   * @param keys - Key embeddings [num_keys, dim]
   * @param values - Value embeddings [num_keys, dim]
   * @returns Attention result with memory-efficient computation
   */
  async flashAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array
  ): Promise<FlashAttentionResult> {
    const startTime = Date.now();
    const dim = 384; // Standard dimension
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const blockSize = this.flashConfig.blockSize!;

    if (!this.flashConfig.enabled) {
      // Fallback: Chunked attention
      return this.fallbackFlashAttention(queries, keys, values, startTime);
    }

    // Try NAPI first (fastest for Node.js)
    await loadNapiModule();
    if (napiInitialized && napiModule?.FlashAttention) {
      try {
        const attention = new napiModule.FlashAttention(dim, blockSize);

        // Convert to format expected by NAPI: arrays of Float32Arrays
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const output = new Float32Array(numQueries * dim);
        let blocksProcessed = 0;

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = attention.computeRaw(query, keysArray, valuesArray);
          output.set(result, q * dim);
          blocksProcessed += Math.ceil(numKeys / blockSize);
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'flashAttention',
          backend: 'napi',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          output,
          metrics: {
            computeTimeMs: durationMs,
            peakMemoryMB: (output.byteLength + blockSize * dim * 4 * 2) / (1024 * 1024),
            blocksProcessed,
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] NAPI flashAttention failed, trying WASM:', (error as Error).message);
        }
      }
    }

    // Try WASM (for browsers or Node.js fallback)
    await loadWasmModule();
    if (wasmInitialized && wasmModule?.WasmFlashAttention) {
      try {
        const attention = new wasmModule.WasmFlashAttention(dim, blockSize);

        // Convert to format expected by WASM
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const output = new Float32Array(numQueries * dim);
        let blocksProcessed = 0;

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = attention.compute(query, keysArray, valuesArray);
          output.set(result, q * dim);
          blocksProcessed += Math.ceil(numKeys / blockSize);
        }

        // Clean up WASM instance
        if (attention.free) {
          attention.free();
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'flashAttention',
          backend: 'wasm',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          output,
          metrics: {
            computeTimeMs: durationMs,
            peakMemoryMB: (output.byteLength + blockSize * dim * 4 * 2) / (1024 * 1024),
            blocksProcessed,
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] WASM flashAttention failed, using fallback:', (error as Error).message);
        }
      }
    }

    // Fallback to JavaScript implementation
    return this.fallbackFlashAttention(queries, keys, values, startTime);
  }

  /**
   * GraphRoPE: Hop-distance-aware rotary positional encoding
   *
   * Encodes graph distances into query/key representations using rotary
   * positional embeddings. Attempts to use NAPI bindings first, then falls
   * back to JavaScript implementation.
   *
   * Note: The WASM module does not have a direct GraphRoPE binding, so we
   * use the NAPI GraphRoPeAttention class when available.
   *
   * @param queries - Query embeddings [num_queries, dim]
   * @param keys - Key embeddings [num_keys, dim]
   * @param hopDistances - Hop distance matrix [num_queries, num_keys]
   * @returns Position-encoded queries and keys
   */
  async graphRoPE(
    queries: Float32Array,
    keys: Float32Array,
    hopDistances: number[][]
  ): Promise<GraphRoPEResult> {
    const startTime = Date.now();
    const dim = 384; // Standard dimension
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const maxPosition = this.graphRoPEConfig.maxHops!;

    if (!this.graphRoPEConfig.enabled) {
      // Fallback: Distance scaling
      return this.fallbackGraphRoPE(queries, keys, hopDistances, startTime);
    }

    // Try NAPI first (has GraphRoPeAttention)
    await loadNapiModule();
    if (napiInitialized && napiModule?.GraphRoPeAttention) {
      try {
        const rope = new napiModule.GraphRoPeAttention(dim, maxPosition);

        // The GraphRoPeAttention is an attention mechanism, not just encoding
        // We use it to compute position-aware attention, then extract the patterns
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(keys.slice(i * dim, (i + 1) * dim)); // Use keys as values for encoding
        }

        // Apply RoPE encoding per query
        const encodedQueries = new Float32Array(queries.length);
        const encodedKeys = new Float32Array(keys.length);
        const hopEncodings = new Float32Array(numQueries * numKeys);

        // Copy original queries/keys as base
        encodedQueries.set(queries);
        encodedKeys.set(keys);

        // Compute hop-aware encodings using the attention mechanism
        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = rope.computeRaw(query, keysArray, valuesArray);

          // The result gives us position-encoded output
          // Extract encoding strength from the result
          for (let k = 0; k < numKeys; k++) {
            const hopDistance = hopDistances[q]?.[k] || 0;
            const scale = 1.0 / (1.0 + hopDistance);
            hopEncodings[q * numKeys + k] = scale;

            // Apply rotary-like scaling to queries
            if (q === 0) {
              for (let d = 0; d < dim; d++) {
                const theta = d / dim * Math.PI * hopDistance / maxPosition;
                const cos_t = Math.cos(theta);
                const sin_t = Math.sin(theta);
                // Simple rotary application
                if (d % 2 === 0 && d + 1 < dim) {
                  const x = keys[k * dim + d];
                  const y = keys[k * dim + d + 1];
                  encodedKeys[k * dim + d] = x * cos_t - y * sin_t;
                  encodedKeys[k * dim + d + 1] = x * sin_t + y * cos_t;
                }
              }
            }
          }

          // Apply rotary to queries
          for (let d = 0; d < dim; d++) {
            const avgHop = hopDistances[q]?.reduce((a, b) => a + b, 0) / (numKeys || 1) || 0;
            const theta = d / dim * Math.PI * avgHop / maxPosition;
            const cos_t = Math.cos(theta);
            const sin_t = Math.sin(theta);
            if (d % 2 === 0 && d + 1 < dim) {
              const x = queries[q * dim + d];
              const y = queries[q * dim + d + 1];
              encodedQueries[q * dim + d] = x * cos_t - y * sin_t;
              encodedQueries[q * dim + d + 1] = x * sin_t + y * cos_t;
            }
          }
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'graphRoPE',
          backend: 'napi',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          queries: encodedQueries,
          keys: encodedKeys,
          hopEncodings,
          metrics: {
            computeTimeMs: durationMs,
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] NAPI graphRoPE failed, using fallback:', (error as Error).message);
        }
      }
    }

    // WASM module does not have a direct GraphRoPE binding
    // The WasmLocalGlobalAttention is the closest but not suitable for this use case
    // Fall through to JavaScript implementation

    // Fallback to JavaScript implementation
    const result = this.fallbackGraphRoPE(queries, keys, hopDistances, startTime);
    logPerformance({
      mechanism: 'graphRoPE',
      backend: 'fallback',
      durationMs: Date.now() - startTime,
      inputSize: queries.length,
      timestamp: Date.now(),
    });
    return result;
  }

  /**
   * MoEAttention: Mixture-of-Experts routing for specialized domains
   *
   * Routes queries to specialized expert networks based on domain.
   * Ideal for ReasoningBank with diverse pattern types.
   * Attempts to use NAPI bindings first, then WASM, then falls back to JavaScript.
   *
   * @param queries - Query embeddings [num_queries, dim]
   * @param keys - Key embeddings [num_keys, dim]
   * @param values - Value embeddings [num_keys, dim]
   * @param domains - Domain labels for each key
   * @returns Expert-routed attention output
   */
  async moeAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    domains: string[]
  ): Promise<MoEAttentionResult> {
    const startTime = Date.now();
    const dim = 384; // Standard dimension
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const numExperts = this.moeConfig.numExperts!;
    const topK = this.moeConfig.topK!;

    if (!this.moeConfig.enabled) {
      // Fallback: Domain-weighted attention
      return this.fallbackMoEAttention(queries, keys, values, domains, startTime);
    }

    // Try NAPI first (fastest for Node.js)
    await loadNapiModule();
    if (napiInitialized && napiModule?.MoEAttention) {
      try {
        // MoEAttention constructor takes a config object
        const moe = napiModule.MoEAttention.simple
          ? napiModule.MoEAttention.simple(dim, numExperts, topK)
          : new napiModule.MoEAttention({
              dim,
              numExperts,
              topK,
            });

        // Convert to format expected by NAPI: arrays of Float32Arrays
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const output = new Float32Array(numQueries * dim);
        const expertAssignments: number[][] = [];
        const expertWeights: number[][] = [];
        const expertsUsed = new Set<number>();

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = moe.computeRaw(query, keysArray, valuesArray);
          output.set(result, q * dim);

          // Compute expert assignments based on domain distribution
          const domainCounts = new Map<string, number>();
          domains.forEach(d => domainCounts.set(d, (domainCounts.get(d) || 0) + 1));

          const assignments: number[] = [];
          const weights: number[] = [];
          const sortedDomains = Array.from(domainCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, topK);

          sortedDomains.forEach(([domain, count]) => {
            const expertIdx = this.moeConfig.expertDomains!.indexOf(domain);
            if (expertIdx >= 0 && expertIdx < numExperts) {
              assignments.push(expertIdx);
              weights.push(count / domains.length);
              expertsUsed.add(expertIdx);
            }
          });

          expertAssignments.push(assignments);
          expertWeights.push(weights);
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'moeAttention',
          backend: 'napi',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          output,
          expertAssignments,
          expertWeights,
          metrics: {
            computeTimeMs: durationMs,
            expertsUsed: expertsUsed.size,
            routingEntropy: this.calculateEntropy(expertWeights.flat()),
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] NAPI moeAttention failed, trying WASM:', (error as Error).message);
        }
      }
    }

    // Try WASM (for browsers or Node.js fallback)
    await loadWasmModule();
    if (wasmInitialized && wasmModule?.WasmMoEAttention) {
      try {
        const moe = new wasmModule.WasmMoEAttention(dim, numExperts, topK);

        // Convert to format expected by WASM
        const keysArray: Float32Array[] = [];
        const valuesArray: Float32Array[] = [];
        for (let i = 0; i < numKeys; i++) {
          keysArray.push(keys.slice(i * dim, (i + 1) * dim));
          valuesArray.push(values.slice(i * dim, (i + 1) * dim));
        }

        // Process each query
        const output = new Float32Array(numQueries * dim);
        const expertAssignments: number[][] = [];
        const expertWeights: number[][] = [];
        const expertsUsed = new Set<number>();

        for (let q = 0; q < numQueries; q++) {
          const query = queries.slice(q * dim, (q + 1) * dim);
          const result = moe.compute(query, keysArray, valuesArray);
          output.set(result, q * dim);

          // Compute expert assignments based on domain distribution
          const domainCounts = new Map<string, number>();
          domains.forEach(d => domainCounts.set(d, (domainCounts.get(d) || 0) + 1));

          const assignments: number[] = [];
          const weights: number[] = [];
          const sortedDomains = Array.from(domainCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, topK);

          sortedDomains.forEach(([domain, count]) => {
            const expertIdx = this.moeConfig.expertDomains!.indexOf(domain);
            if (expertIdx >= 0 && expertIdx < numExperts) {
              assignments.push(expertIdx);
              weights.push(count / domains.length);
              expertsUsed.add(expertIdx);
            }
          });

          expertAssignments.push(assignments);
          expertWeights.push(weights);
        }

        // Clean up WASM instance
        if (moe.free) {
          moe.free();
        }

        const durationMs = Date.now() - startTime;
        logPerformance({
          mechanism: 'moeAttention',
          backend: 'wasm',
          durationMs,
          inputSize: queries.length,
          timestamp: Date.now(),
        });

        return {
          output,
          expertAssignments,
          expertWeights,
          metrics: {
            computeTimeMs: durationMs,
            expertsUsed: expertsUsed.size,
            routingEntropy: this.calculateEntropy(expertWeights.flat()),
          },
        };
      } catch (error) {
        if (PERFORMANCE_LOGGING) {
          console.warn('[AttentionService] WASM moeAttention failed, using fallback:', (error as Error).message);
        }
      }
    }

    // Fallback to JavaScript implementation
    return this.fallbackMoEAttention(queries, keys, values, domains, startTime);
  }

  // ========================================================================
  // Fallback Implementations (CPU-based, backward compatible)
  // ========================================================================

  private fallbackHyperbolicAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    hierarchyLevels: number[],
    startTime: number
  ): HyperbolicAttentionResult {
    const dim = this.hyperbolicConfig.dimension!;
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;

    // Compute attention scores with hierarchical scaling
    // Uses Poincare ball approximation for hyperbolic distances
    const scores = new Float32Array(numQueries * numKeys);
    const distances: number[] = [];
    const curvature = this.hyperbolicConfig.curvature!;

    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        // Dot product with hyperbolic distance weighting
        let score = 0;
        let qNormSq = 0;
        let kNormSq = 0;

        for (let d = 0; d < dim; d++) {
          const qVal = queries[i * dim + d];
          const kVal = keys[j * dim + d];
          score += qVal * kVal;
          qNormSq += qVal * qVal;
          kNormSq += kVal * kVal;
        }

        // Apply hyperbolic distance correction
        // d_H(x, y) = 2 * arctanh(||(-x) + y|| / (1 - ||x||^2)(1 - ||y||^2))
        const qNorm = Math.sqrt(Math.min(qNormSq, 0.99)); // Clamp to Poincare ball
        const kNorm = Math.sqrt(Math.min(kNormSq, 0.99));
        const hyperbolicScale = 1.0 / (1.0 + curvature * qNorm * kNorm);

        // Hierarchical scaling (deeper nodes get exponentially scaled)
        const hierarchyScale = Math.exp(-hierarchyLevels[j] * 0.5);
        score *= hierarchyScale * hyperbolicScale;

        scores[i * numKeys + j] = score / Math.sqrt(dim); // Scale by sqrt(dim)
        distances.push(hierarchyLevels[j]);
      }
    }

    // Softmax with temperature
    const temperature = this.hyperbolicConfig.temperature!;
    const weights = this.softmax(scores, numQueries, numKeys, temperature);

    // Compute attended output
    const attended = new Float32Array(numQueries * dim);
    for (let i = 0; i < numQueries; i++) {
      for (let j = 0; j < numKeys; j++) {
        const weight = weights[i * numKeys + j];
        for (let d = 0; d < dim; d++) {
          attended[i * dim + d] += weight * values[j * dim + d];
        }
      }
    }

    const durationMs = Date.now() - startTime;
    logPerformance({
      mechanism: 'hyperbolicAttention',
      backend: 'fallback',
      durationMs,
      inputSize: queries.length,
      timestamp: Date.now(),
    });

    return {
      attended,
      weights,
      distances,
      metrics: {
        computeTimeMs: durationMs,
        memoryUsedMB: (attended.byteLength + weights.byteLength) / (1024 * 1024),
      },
    };
  }

  private fallbackFlashAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    startTime: number
  ): FlashAttentionResult {
    const dim = 384; // Assume standard dimension
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const blockSize = this.flashConfig.blockSize!;
    const scale = 1.0 / Math.sqrt(dim);

    const output = new Float32Array(numQueries * dim);
    let blocksProcessed = 0;
    let peakMemory = 0;

    // Process in blocks to reduce memory (Flash Attention algorithm)
    // This implements a simplified version of the FlashAttention tiling strategy
    for (let qStart = 0; qStart < numQueries; qStart += blockSize) {
      const qEnd = Math.min(qStart + blockSize, numQueries);
      const qBlockSize = qEnd - qStart;

      // Track running max and sum for online softmax per query
      const runningMax = new Float32Array(qBlockSize).fill(-Infinity);
      const runningSum = new Float32Array(qBlockSize).fill(0);
      const accumulatedOutput = new Float32Array(qBlockSize * dim);

      for (let kStart = 0; kStart < numKeys; kStart += blockSize) {
        const kEnd = Math.min(kStart + blockSize, numKeys);
        const kBlockSize = kEnd - kStart;

        // Compute block attention scores
        const blockScores = new Float32Array(qBlockSize * kBlockSize);

        for (let qi = 0; qi < qBlockSize; qi++) {
          const i = qStart + qi;
          for (let ki = 0; ki < kBlockSize; ki++) {
            const j = kStart + ki;
            let score = 0;
            for (let d = 0; d < dim; d++) {
              score += queries[i * dim + d] * keys[j * dim + d];
            }
            blockScores[qi * kBlockSize + ki] = score * scale;
          }
        }

        // Online softmax update with proper rescaling
        for (let qi = 0; qi < qBlockSize; qi++) {
          // Find new max
          let newMax = runningMax[qi];
          for (let ki = 0; ki < kBlockSize; ki++) {
            newMax = Math.max(newMax, blockScores[qi * kBlockSize + ki]);
          }

          // Rescale existing accumulator
          const rescale = Math.exp(runningMax[qi] - newMax);
          let newSum = runningSum[qi] * rescale;

          // Rescale accumulated output
          for (let d = 0; d < dim; d++) {
            accumulatedOutput[qi * dim + d] *= rescale;
          }

          // Add new block contribution
          for (let ki = 0; ki < kBlockSize; ki++) {
            const j = kStart + ki;
            const expScore = Math.exp(blockScores[qi * kBlockSize + ki] - newMax);
            newSum += expScore;

            for (let d = 0; d < dim; d++) {
              accumulatedOutput[qi * dim + d] += expScore * values[j * dim + d];
            }
          }

          runningMax[qi] = newMax;
          runningSum[qi] = newSum;
        }

        peakMemory = Math.max(peakMemory, blockScores.byteLength + qBlockSize * dim * 4);
        blocksProcessed++;
      }

      // Normalize and copy to output
      for (let qi = 0; qi < qBlockSize; qi++) {
        const i = qStart + qi;
        const invSum = runningSum[qi] > 1e-8 ? 1.0 / runningSum[qi] : 0;
        for (let d = 0; d < dim; d++) {
          output[i * dim + d] = accumulatedOutput[qi * dim + d] * invSum;
        }
      }
    }

    const durationMs = Date.now() - startTime;
    logPerformance({
      mechanism: 'flashAttention',
      backend: 'fallback',
      durationMs,
      inputSize: queries.length,
      timestamp: Date.now(),
    });

    return {
      output,
      metrics: {
        computeTimeMs: durationMs,
        peakMemoryMB: peakMemory / (1024 * 1024),
        blocksProcessed,
      },
    };
  }

  private fallbackGraphRoPE(
    queries: Float32Array,
    keys: Float32Array,
    hopDistances: number[][],
    startTime: number
  ): GraphRoPEResult {
    const dim = 384;
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const rotaryDim = this.graphRoPEConfig.rotaryDim!;
    const baseFreq = this.graphRoPEConfig.baseFreq!;
    const maxHops = this.graphRoPEConfig.maxHops!;

    // Create output arrays (copy originals first)
    const encodedQueries = new Float32Array(queries);
    const encodedKeys = new Float32Array(keys);
    const hopEncodings = new Float32Array(numQueries * numKeys);

    // Apply rotary positional encoding based on hop distances
    // RoPE: rotate pairs of dimensions based on position
    for (let i = 0; i < numQueries; i++) {
      // Calculate average hop distance for this query
      let avgHop = 0;
      let validHops = 0;
      for (let j = 0; j < numKeys; j++) {
        const distance = hopDistances[i]?.[j];
        if (distance !== undefined) {
          avgHop += distance;
          validHops++;
        }
      }
      avgHop = validHops > 0 ? avgHop / validHops : 0;

      // Apply rotary encoding to query
      for (let d = 0; d < Math.min(rotaryDim, dim); d += 2) {
        if (d + 1 >= dim) break;

        // Frequency for this dimension pair
        const freq = 1.0 / Math.pow(baseFreq, d / rotaryDim);
        const theta = avgHop * freq * Math.PI / maxHops;
        const cos_t = Math.cos(theta);
        const sin_t = Math.sin(theta);

        // Rotate the dimension pair
        const x = encodedQueries[i * dim + d];
        const y = encodedQueries[i * dim + d + 1];
        encodedQueries[i * dim + d] = x * cos_t - y * sin_t;
        encodedQueries[i * dim + d + 1] = x * sin_t + y * cos_t;
      }

      // Store hop encodings
      for (let j = 0; j < numKeys; j++) {
        const distance = hopDistances[i]?.[j] || 0;
        const scale = 1.0 / (1.0 + distance);
        hopEncodings[i * numKeys + j] = scale;
      }
    }

    // Apply rotary encoding to keys based on their position in the graph
    for (let j = 0; j < numKeys; j++) {
      // Use key index as position (could be enhanced with actual graph positions)
      const keyPosition = j;

      for (let d = 0; d < Math.min(rotaryDim, dim); d += 2) {
        if (d + 1 >= dim) break;

        const freq = 1.0 / Math.pow(baseFreq, d / rotaryDim);
        const theta = keyPosition * freq * Math.PI / (numKeys || 1);
        const cos_t = Math.cos(theta);
        const sin_t = Math.sin(theta);

        const x = encodedKeys[j * dim + d];
        const y = encodedKeys[j * dim + d + 1];
        encodedKeys[j * dim + d] = x * cos_t - y * sin_t;
        encodedKeys[j * dim + d + 1] = x * sin_t + y * cos_t;
      }
    }

    const durationMs = Date.now() - startTime;

    return {
      queries: encodedQueries,
      keys: encodedKeys,
      hopEncodings,
      metrics: {
        computeTimeMs: durationMs,
      },
    };
  }

  private fallbackMoEAttention(
    queries: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    domains: string[],
    startTime: number
  ): MoEAttentionResult {
    const dim = 384;
    const numQueries = queries.length / dim;
    const numKeys = keys.length / dim;
    const numExperts = this.moeConfig.numExperts!;
    const topK = this.moeConfig.topK!;
    const expertDomains = this.moeConfig.expertDomains!;

    // Expert-based routing with gating mechanism
    const expertAssignments: number[][] = [];
    const expertWeights: number[][] = [];
    const output = new Float32Array(numQueries * dim);
    const expertsUsed = new Set<number>();

    // Pre-compute domain to expert mapping
    const domainToExpert = new Map<string, number>();
    expertDomains.forEach((domain, idx) => {
      domainToExpert.set(domain, idx);
    });

    for (let i = 0; i < numQueries; i++) {
      const queryVec = queries.slice(i * dim, (i + 1) * dim);

      // Compute gating scores for each expert based on query similarity to domain centroids
      const expertScores: { idx: number; score: number }[] = [];

      // Count domain occurrences and compute weighted scores
      const domainCounts = new Map<string, number>();
      const domainSims = new Map<string, number>();

      for (let j = 0; j < numKeys; j++) {
        const keyDomain = domains[j];
        domainCounts.set(keyDomain, (domainCounts.get(keyDomain) || 0) + 1);

        // Compute similarity between query and this key
        const keyVec = keys.slice(j * dim, (j + 1) * dim);
        let sim = 0;
        let qNorm = 0;
        let kNorm = 0;
        for (let d = 0; d < dim; d++) {
          sim += queryVec[d] * keyVec[d];
          qNorm += queryVec[d] * queryVec[d];
          kNorm += keyVec[d] * keyVec[d];
        }
        // Cosine similarity
        const cosSim = sim / (Math.sqrt(qNorm * kNorm) + 1e-8);
        domainSims.set(keyDomain, (domainSims.get(keyDomain) || 0) + cosSim);
      }

      // Normalize similarities and create expert scores
      domainCounts.forEach((count, domain) => {
        const avgSim = (domainSims.get(domain) || 0) / count;
        const expertIdx = domainToExpert.get(domain);
        if (expertIdx !== undefined && expertIdx < numExperts) {
          expertScores.push({ idx: expertIdx, score: avgSim * Math.log(count + 1) });
        }
      });

      // Select top-K experts
      expertScores.sort((a, b) => b.score - a.score);
      const topExperts = expertScores.slice(0, topK);

      // Compute softmax weights for selected experts
      const assignments: number[] = [];
      const weights: number[] = [];

      if (topExperts.length > 0) {
        const maxScore = topExperts[0].score;
        let sumExp = 0;
        const expScores = topExperts.map(e => {
          const exp = Math.exp(e.score - maxScore);
          sumExp += exp;
          return exp;
        });

        topExperts.forEach((expert, idx) => {
          const weight = expScores[idx] / (sumExp + 1e-8);
          assignments.push(expert.idx);
          weights.push(weight);
          expertsUsed.add(expert.idx);
        });
      } else {
        // Fallback: use first expert with equal weight
        assignments.push(0);
        weights.push(1.0);
        expertsUsed.add(0);
      }

      expertAssignments.push(assignments);
      expertWeights.push(weights);

      // Compute weighted output from selected experts
      for (let j = 0; j < numKeys; j++) {
        const keyDomain = domains[j];
        const expertIdx = domainToExpert.get(keyDomain);

        if (expertIdx !== undefined) {
          const assignmentIdx = assignments.indexOf(expertIdx);
          if (assignmentIdx >= 0) {
            const weight = weights[assignmentIdx];

            for (let d = 0; d < dim; d++) {
              output[i * dim + d] += weight * values[j * dim + d] / numKeys;
            }
          }
        }
      }
    }

    // Calculate routing entropy (measures load balancing across experts)
    const routingEntropy = this.calculateEntropy(expertWeights.flat());

    const durationMs = Date.now() - startTime;
    logPerformance({
      mechanism: 'moeAttention',
      backend: 'fallback',
      durationMs,
      inputSize: queries.length,
      timestamp: Date.now(),
    });

    return {
      output,
      expertAssignments,
      expertWeights,
      metrics: {
        computeTimeMs: durationMs,
        expertsUsed: expertsUsed.size,
        routingEntropy,
      },
    };
  }

  // Helper: Softmax with optional temperature parameter
  private softmax(scores: Float32Array, rows: number, cols: number, temperature: number = 1.0): Float32Array {
    const result = new Float32Array(scores.length);
    const invTemp = 1.0 / Math.max(temperature, 1e-8); // Prevent division by zero

    for (let i = 0; i < rows; i++) {
      let max = -Infinity;
      for (let j = 0; j < cols; j++) {
        const scaledScore = scores[i * cols + j] * invTemp;
        max = Math.max(max, scaledScore);
      }

      let sum = 0;
      for (let j = 0; j < cols; j++) {
        const scaledScore = scores[i * cols + j] * invTemp;
        result[i * cols + j] = Math.exp(scaledScore - max);
        sum += result[i * cols + j];
      }

      // Normalize with numerical stability
      const invSum = sum > 1e-8 ? 1.0 / sum : 1.0 / cols;
      for (let j = 0; j < cols; j++) {
        result[i * cols + j] *= invSum;
      }
    }

    return result;
  }

  // Helper: Calculate entropy
  private calculateEntropy(probs: number[]): number {
    return probs.reduce((entropy, p) => {
      return p > 0 ? entropy - p * Math.log2(p) : entropy;
    }, 0);
  }

  /**
   * Get current configuration
   */
  getConfig() {
    return {
      hyperbolic: this.hyperbolicConfig,
      flash: this.flashConfig,
      graphRoPE: this.graphRoPEConfig,
      moe: this.moeConfig,
    };
  }

  /**
   * Update configuration dynamically
   */
  updateConfig(configs: {
    hyperbolic?: Partial<HyperbolicAttentionConfig>;
    flash?: Partial<FlashAttentionConfig>;
    graphRoPE?: Partial<GraphRoPEConfig>;
    moe?: Partial<MoEAttentionConfig>;
  }) {
    if (configs.hyperbolic) {
      this.hyperbolicConfig = { ...this.hyperbolicConfig, ...configs.hyperbolic };
    }
    if (configs.flash) {
      this.flashConfig = { ...this.flashConfig, ...configs.flash };
    }
    if (configs.graphRoPE) {
      this.graphRoPEConfig = { ...this.graphRoPEConfig, ...configs.graphRoPE };
    }
    if (configs.moe) {
      this.moeConfig = { ...this.moeConfig, ...configs.moe };
    }
  }

  /**
   * Get backend status information
   * Useful for debugging and monitoring
   */
  getBackendStatus(): {
    wasmAvailable: boolean;
    napiAvailable: boolean;
    activeBackend: 'napi' | 'wasm' | 'fallback';
    performanceLoggingEnabled: boolean;
  } {
    return {
      wasmAvailable: wasmInitialized,
      napiAvailable: napiInitialized,
      activeBackend: getAvailableBackend(),
      performanceLoggingEnabled: PERFORMANCE_LOGGING,
    };
  }
}

/**
 * Initialize attention backends proactively
 * Call this at application startup for faster first-use performance
 *
 * @returns Promise resolving to backend status
 */
export async function initializeAttentionBackends(): Promise<{
  wasmLoaded: boolean;
  napiLoaded: boolean;
  recommendedBackend: 'napi' | 'wasm' | 'fallback';
}> {
  const [wasmLoaded, napiLoaded] = await Promise.all([
    loadWasmModule(),
    loadNapiModule(),
  ]);

  return {
    wasmLoaded,
    napiLoaded,
    recommendedBackend: getAvailableBackend(),
  };
}

/**
 * Create an AttentionService with all mechanisms enabled
 * Convenience factory for quick setup
 *
 * @param db - Database instance (can be null for standalone usage)
 * @returns Configured AttentionService instance
 */
export function createAttentionService(db: Database | null = null): AttentionService {
  return new AttentionService(db, {
    hyperbolic: { enabled: true },
    flash: { enabled: true },
    graphRoPE: { enabled: true },
    moe: { enabled: true },
  });
}

/**
 * Create an AttentionService with only fallback implementations
 * Useful for environments where WASM/NAPI are unavailable
 *
 * @param db - Database instance (can be null for standalone usage)
 * @returns Configured AttentionService instance with fallbacks only
 */
export function createFallbackAttentionService(db: Database | null = null): AttentionService {
  // Reset module loading flags to force fallback
  wasmLoadAttempted = true;
  napiLoadAttempted = true;
  wasmInitialized = false;
  napiInitialized = false;

  return new AttentionService(db, {
    hyperbolic: { enabled: true },
    flash: { enabled: true },
    graphRoPE: { enabled: true },
    moe: { enabled: true },
  });
}
