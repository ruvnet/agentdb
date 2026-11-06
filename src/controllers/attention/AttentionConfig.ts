/**
 * AttentionConfig - Configuration management for attention mechanisms
 *
 * Handles:
 * - Config validation
 * - Constants
 * - Default values
 */

/**
 * Configuration for attention mechanisms
 */
export interface AttentionConfig {
  /** Number of attention heads */
  numHeads: number;
  /** Dimension of each head */
  headDim: number;
  /** Total embedding dimension (usually numHeads * headDim) */
  embedDim: number;
  /** Dropout probability (0-1) */
  dropout?: number;
  /** Whether to use bias in linear projections */
  bias?: boolean;
  /** Use Flash Attention optimization if available */
  useFlash?: boolean;
  /** Use Linear Attention for O(n) complexity */
  useLinear?: boolean;
  /** Use Hyperbolic space for hierarchical data */
  useHyperbolic?: boolean;
  /** Use Mixture-of-Experts routing */
  useMoE?: boolean;
  /** Number of experts for MoE (default: 8) */
  numExperts?: number;
  /** Top-k experts to activate in MoE (default: 2) */
  topK?: number;
  /** Sparsification configuration */
  sparsification?: {
    enabled: boolean;
    method: 'ppr' | 'random-walk' | 'spectral';
    topK: number;
  };
  /**
   * Upper bound on attention scores per call (queryRows × keyRows × numHeads).
   * Calls above it are rejected with a size-limit error before any buffer is
   * allocated. Default: 2^25 (about 128 MB of f32 scores).
   */
  maxAttentionScores?: number;
  /** Graph partitioning configuration */
  partitioning?: {
    enabled: boolean;
    method: 'stoer-wagner' | 'karger' | 'flow-based';
    maxPartitionSize: number;
  };
}

/**
 * Options for attention operations (alias for AttentionConfig)
 */
export type AttentionOptions = AttentionConfig;

/**
 * Result from attention computation
 */
export interface AttentionResult {
  /** Output embeddings after attention */
  output: Float32Array;
  /** Attention weights (optional, for visualization) */
  weights?: Float32Array;
  /** Execution time in milliseconds */
  executionTimeMs: number;
  /** Which mechanism was used */
  mechanism: 'multi-head' | 'flash' | 'linear' | 'hyperbolic' | 'moe' | 'sparse' | 'partitioned';
  /** Runtime environment */
  runtime: 'napi' | 'wasm' | 'fallback';
  /** Sparsification metadata (for sparse attention) */
  sparsityMetadata?: {
    method?: string;
    topKNodes?: number;
    sparsityRatio?: number;
  };
  /** Partitioning metadata (for partitioned attention) */
  partitioningMetadata?: {
    numPartitions?: number;
    cutSize?: number;
    avgPartitionSize?: number;
  };
}

/**
 * AttentionConfigManager - Manages configuration and constants
 */
export class AttentionConfigManager {
  // Performance targets (ADR-071)
  static readonly FLASH_V2_MIN_SPEEDUP = 2.49;
  static readonly FLASH_V2_MAX_SPEEDUP = 7.47;

  // Attention computation constants
  static readonly MASKED_SCORE = -Infinity;

  // Buffer pool limits
  static readonly MAX_POOLED_BUFFERS = 10;

  // Mask cache limits
  static readonly MAX_CACHED_MASKS = 50;

  private config: AttentionConfig;

  constructor(config: AttentionConfig) {
    this.config = this.applyDefaults(config);
    this.validateConfig(this.config);
  }

  /**
   * Apply default values to configuration
   */
  private applyDefaults(config: AttentionConfig): AttentionConfig {
    const defaults: AttentionConfig = {
      ...config,
      dropout: config.dropout ?? 0.1,
      bias: config.bias ?? true,
      useFlash: config.useFlash ?? true,
      useLinear: config.useLinear ?? false,
      useHyperbolic: config.useHyperbolic ?? false,
      useMoE: config.useMoE ?? false,
      numExperts: config.numExperts ?? 8,
      topK: config.topK ?? 2,
    };

    if (config.sparsification) {
      defaults.sparsification = {
        enabled: config.sparsification.enabled ?? false,
        method: config.sparsification.method ?? 'ppr',
        topK: config.sparsification.topK ?? 100,
      };
    }

    if (config.partitioning) {
      defaults.partitioning = {
        enabled: config.partitioning.enabled ?? false,
        method: config.partitioning.method ?? 'stoer-wagner',
        maxPartitionSize: config.partitioning.maxPartitionSize ?? 1000,
      };
    }

    return defaults;
  }

  /**
   * Validate configuration values
   */
  private validateConfig(config: AttentionConfig): void {
    if (!Number.isInteger(config.numHeads) || config.numHeads <= 0) {
      throw new Error('numHeads must be a positive integer');
    }
    if (!Number.isInteger(config.headDim) || config.headDim <= 0) {
      throw new Error('headDim must be a positive integer');
    }
    if (!Number.isInteger(config.embedDim) || config.embedDim <= 0) {
      throw new Error('embedDim must be a positive integer');
    }
    if (config.embedDim !== config.numHeads * config.headDim) {
      throw new Error(
        `embedDim (${config.embedDim}) must equal numHeads × headDim (${config.numHeads} × ${config.headDim} = ${config.numHeads * config.headDim})`
      );
    }
    if (
      config.maxAttentionScores !== undefined &&
      (!Number.isFinite(config.maxAttentionScores) || config.maxAttentionScores <= 0)
    ) {
      throw new Error('maxAttentionScores must be a positive finite number');
    }
    if (config.dropout !== undefined && (config.dropout < 0 || config.dropout > 1)) {
      throw new Error('dropout must be between 0 and 1');
    }
    if (config.numExperts !== undefined && config.numExperts <= 0) {
      throw new Error('numExperts must be positive');
    }
    if (config.topK !== undefined && config.topK <= 0) {
      throw new Error('topK must be positive');
    }
  }

  /**
   * Get the configuration
   */
  getConfig(): AttentionConfig {
    return { ...this.config };
  }

  /**
   * Get number of heads
   */
  getNumHeads(): number {
    return this.config.numHeads;
  }

  /**
   * Get head dimension
   */
  getHeadDim(): number {
    return this.config.headDim;
  }

  /**
   * Get embedding dimension
   */
  getEmbedDim(): number {
    return this.config.embedDim;
  }

  /**
   * Get dropout rate
   */
  getDropout(): number {
    return this.config.dropout || 0.0;
  }

  /**
   * Get number of experts for MoE
   */
  getNumExperts(): number {
    return this.config.numExperts || 8;
  }

  /**
   * Get top-k for MoE
   */
  getTopK(): number {
    return this.config.topK || 2;
  }
}
