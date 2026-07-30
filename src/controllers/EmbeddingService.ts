/**
 * EmbeddingService - Text Embedding Generation
 *
 * Handles text-to-vector embedding generation using various models.
 * Supports both local (transformers.js) and remote (OpenAI, etc.) embeddings.
 */

import {
  applyRoleTemplate,
  createEmbeddingSpaceIdentity,
  type EmbeddingRole,
  type EmbeddingRolePolicy,
  type EmbeddingSpaceIdentity,
  type EmbeddingSpaceIdentityInput
} from '../embedding/EmbeddingSpaceIdentity.js';

export interface EmbeddingConfig {
  model: string;
  dimension: number;
  provider: 'transformers' | 'openai' | 'local';
  apiKey?: string;
  /**
   * Required for unregistered models. Registered model capabilities are
   * authoritative and a conflicting declaration is rejected.
   */
  rolePolicy?: EmbeddingRolePolicy;
  /**
   * Immutable production identity. Legacy callers receive an explicitly
   * unverified identity until they provide artifact fingerprints.
   */
  embeddingSpace?: EmbeddingSpaceIdentityInput;
}

const symmetricModels = [
  'Xenova/all-MiniLM-L6-v2',
  'all-MiniLM-L6-v2',
  'text-embedding-3-small',
  'text-embedding-3-large',
  'text-embedding-ada-002',
  'mock',
  'mock-model',
  'small-model',
  'large-model',
  'xl-model'
];

const modelRoleRegistry = new Map<string, EmbeddingRolePolicy>(
  symmetricModels.map(model => [model, { kind: 'symmetric' }])
);

function policiesEqual(left: EmbeddingRolePolicy, right: EmbeddingRolePolicy): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

/**
 * Register an exact model revision's role policy. Family-name inference is
 * deliberately unsupported because variants can use different instructions.
 */
export function registerEmbeddingModelRolePolicy(
  modelId: string,
  policy: EmbeddingRolePolicy
): void {
  if (!modelId.trim()) throw new Error('Embedding modelId cannot be empty');
  const existing = modelRoleRegistry.get(modelId);
  if (existing && !policiesEqual(existing, policy)) {
    throw new Error(`Cannot replace authoritative role policy for registered model ${modelId}`);
  }
  modelRoleRegistry.set(modelId, Object.freeze({ ...policy }));
}

export class EmbeddingService {
  private config: EmbeddingConfig;
  private readonly rolePolicy: EmbeddingRolePolicy;
  private readonly embeddingSpaceIdentity: EmbeddingSpaceIdentity;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- transformers.js pipeline has no exported type
  private pipeline: any;
  private cache: Map<string, Float32Array>;
  private inFlightCache: Map<string, Promise<Float32Array>>;

  constructor(config: EmbeddingConfig) {
    this.config = config;
    this.cache = new Map();
    this.inFlightCache = new Map();

    const registeredPolicy = modelRoleRegistry.get(config.model);
    if (registeredPolicy && config.rolePolicy && !policiesEqual(registeredPolicy, config.rolePolicy)) {
      throw new Error(
        `Embedding role policy for registered model ${config.model} disagrees with the model registry`
      );
    }
    if (!registeredPolicy && !config.rolePolicy) {
      throw new Error(
        `Unregistered embedding model ${config.model} requires an explicit rolePolicy`
      );
    }
    this.rolePolicy = registeredPolicy ?? Object.freeze({ ...config.rolePolicy! });
    if (config.embeddingSpace && !policiesEqual(config.embeddingSpace.rolePolicy, this.rolePolicy)) {
      throw new Error('Embedding space rolePolicy disagrees with the model capability policy');
    }
    this.embeddingSpaceIdentity = createEmbeddingSpaceIdentity(
      config.embeddingSpace ?? {
        modelId: config.model,
        modelArtifactHash: `legacy-unverified:${config.model}`,
        tokenizerHash: `legacy-unverified:${config.model}`,
        promptTemplateHash: `legacy-unverified:${JSON.stringify(this.rolePolicy)}`,
        poolingStrategy: 'mean',
        truncationLength: 512,
        outputDimension: config.dimension,
        outputDtype: 'float32',
        normalization: 'l2',
        runtimeRevision: `legacy-unverified:${config.provider}`,
        distanceMetric: 'cosine',
        rolePolicy: this.rolePolicy
      }
    );
  }

  /**
   * Initialize the embedding service
   */
  async initialize(): Promise<void> {
    if (this.config.provider === 'transformers') {
      if (process.env.AGENTDB_DISABLE_TRANSFORMERS === '1') {
        console.error('Transformers.js disabled by AGENTDB_DISABLE_TRANSFORMERS');
        this.pipeline = null;
        return;
      }
      // Use transformers.js for local embeddings
      try {
        const transformers = await import('@huggingface/transformers');

        const env = transformers.env as Record<string, unknown>;

        // Try to load model from bundled .rvf or local cache first
        try {
          const { ModelCacheLoader } = await import('../model/ModelCacheLoader.js');
          const cached = await ModelCacheLoader.resolve(this.config.model);

          if (cached) {
            env.localModelPath = cached.localPath;
            env.allowRemoteModels = false;
            env.cacheDir = cached.localPath;
          }
        } catch {
          // ModelCacheLoader not available — fall through to network download
        }

        // Set Hugging Face token if available from environment
        const hfToken = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN;
        if (hfToken && typeof env === 'object') {
          env.HF_TOKEN = hfToken;
        }

        this.pipeline = await transformers.pipeline('feature-extraction', this.config.model);
        console.error(`Transformers.js loaded: ${this.config.model}`);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.warn(`Transformers.js initialization failed: ${errorMessage}`);
        console.warn('   Falling back to mock embeddings for testing');
        console.warn('   This is normal if:');
        console.warn('     - Running offline/without internet access');
        console.warn('     - Model not yet downloaded (~90MB on first use)');
        console.warn('     - Network connectivity issues');
        console.warn('   To use real embeddings:');
        console.warn('     - Ensure internet connectivity for first-time model download');
        console.warn('     - Or pre-download: npx agentdb install-embeddings');
        this.pipeline = null;
      }
    }
  }

  /**
   * Generate embedding for text
   */
  async embed(text: string): Promise<Float32Array> {
    if (this.rolePolicy.kind === 'asymmetric') {
      throw new Error(
        'embed() is ambiguous for an asymmetric model; use embedQuery() or embedPassage()'
      );
    }
    return this.embedFor('passage', text);
  }

  /** Embed retrieval query text using the model's exact query template. */
  async embedQuery(text: string): Promise<Float32Array> {
    return this.embedFor('query', text);
  }

  /** Embed corpus text using the model's exact passage template. */
  async embedPassage(text: string): Promise<Float32Array> {
    return this.embedFor('passage', text);
  }

  /** Role-authoritative embedding entry point. */
  async embedFor(role: EmbeddingRole, text: string): Promise<Float32Array> {
    const processedText = this.rolePolicy.kind === 'asymmetric'
      ? applyRoleTemplate(
          role === 'query'
            ? this.rolePolicy.queryTemplate
            : this.rolePolicy.passageTemplate,
          text
        )
      : text;

    // Check cache
    const cacheKey = `${this.embeddingSpaceIdentity.hash}:${role}:${processedText}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    const existing = this.inFlightCache.get(cacheKey);
    if (existing) {
      return existing;
    }

    const generation = this.generateEmbedding(processedText).then(embedding => {
      if (this.inFlightCache.get(cacheKey) === generation) {
        if (this.cache.size > 10000) {
          const keysToDelete = Array.from(this.cache.keys()).slice(0, 5000);
          keysToDelete.forEach(k => this.cache.delete(k));
        }
        this.cache.set(cacheKey, embedding);
      }
      return embedding;
    });
    this.inFlightCache.set(cacheKey, generation);

    try {
      return await generation;
    } finally {
      if (this.inFlightCache.get(cacheKey) === generation) {
        this.inFlightCache.delete(cacheKey);
      }
    }
  }

  getEmbeddingSpaceIdentity(): Readonly<EmbeddingSpaceIdentity> {
    return this.embeddingSpaceIdentity;
  }

  /**
   * Batch embed multiple texts
   */
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map(text => this.embed(text)));
  }

  async embedQueryBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map(text => this.embedQuery(text)));
  }

  async embedPassageBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map(text => this.embedPassage(text)));
  }

  /**
   * Clear embedding cache
   */
  clearCache(): void {
    this.cache.clear();
    this.inFlightCache.clear();
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private async generateEmbedding(text: string): Promise<Float32Array> {
    if (this.config.provider === 'transformers' && this.pipeline) {
      const output = await this.pipeline(text, { pooling: 'mean', normalize: true });
      return new Float32Array(output.data);
    }
    if (this.config.provider === 'openai' && this.config.apiKey) {
      return this.embedOpenAI(text);
    }
    return this.mockEmbedding(text);
  }

  private async embedOpenAI(text: string): Promise<Float32Array> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.config.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: this.config.model,
        input: text
      })
    });

    const data = await response.json() as { data: Array<{ embedding: number[] }> };
    return new Float32Array(data.data[0].embedding);
  }

  private mockEmbedding(text: string): Float32Array {
    // Simple deterministic mock embedding for testing
    const embedding = new Float32Array(this.config.dimension);

    // Handle null/undefined/empty text
    if (!text || text.length === 0) {
      return new Float32Array(this.config.dimension);
    }

    // Use simple hash-based generation
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash) + text.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }

    // Fill embedding with pseudo-random values based on hash
    for (let i = 0; i < this.config.dimension; i++) {
      const seed = hash + i * 31;
      embedding[i] = Math.sin(seed) * Math.cos(seed * 0.5);
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < embedding.length; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm);

    for (let i = 0; i < embedding.length; i++) {
      embedding[i] /= norm;
    }

    return embedding;
  }
}
