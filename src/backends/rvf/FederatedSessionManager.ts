/**
 * FederatedSessionManager - Cross-Session Federated Learning for AgentDB
 *
 * Wraps @ruvector/ruvllm's EphemeralAgent + FederatedCoordinator to provide:
 * - Agent-scoped trajectory recording per session
 * - Federated aggregation across all agent sessions
 * - Warm-start pattern loading for new sessions
 * - LoRA adapter management for task-specific fine-tuning
 * - Continual learning pipeline with EWC++ protection
 *
 * Architecture:
 *   Session 1 → EphemeralAgent → aggregate() → FederatedCoordinator → getInitialPatterns()
 *   Session 2 → EphemeralAgent → aggregate() ↗                      → warm-start new sessions
 *   Session N → EphemeralAgent → aggregate() ↗
 *
 * Security:
 * - Operates on embeddings only (no user text stored in trajectories)
 * - Agent IDs validated (length, format)
 * - Dimension bounded (1-4096)
 * - Max agents bounded (1-1000)
 * - Quality scores clamped to [0, 1]
 * - LoRA rank bounded (1-64)
 * - Trajectory state is serializable JSON (no executable code)
 */

/** Configuration for the federated session manager */
export interface FederatedConfig {
  /** Embedding dimension (must match across all sessions) */
  dimension: number;
  /** Maximum number of concurrent agents (default: 100) */
  maxAgents?: number;
  /** Quality threshold for pattern inclusion (default: 0.4) */
  qualityThreshold?: number;
  /** Number of trajectories before auto-consolidation (default: 50) */
  consolidationInterval?: number;
  /** LoRA rank for adapter fine-tuning (default: 4) */
  loraRank?: number;
  /** LoRA alpha scaling (default: 8) */
  loraAlpha?: number;
  /** EWC lambda for catastrophic forgetting prevention (default: 100) */
  ewcLambda?: number;
}

/** Exported session state for persistence */
export interface SessionState {
  agentId: string;
  trajectories: TrajectoryRecord[];
  stats: SessionStats;
  sessionDurationMs: number;
  timestamp: string;
}

/** Individual trajectory record */
export interface TrajectoryRecord {
  embedding: number[];
  quality: number;
  route?: string;
  context?: string[];
  timestamp: number;
}

/** Session statistics */
export interface SessionStats {
  trajectoryCount: number;
  avgQuality: number;
  patternsLearned: number;
  sessionDurationMs: number;
}

/** Federated coordinator statistics */
export interface FederatedStats {
  totalAgents: number;
  totalTrajectories: number;
  patternsLearned: number;
  avgQuality: number;
  qualityThreshold: number;
  contributions: Record<string, number>;
}

/** Learned pattern from federated aggregation */
export interface FederatedPattern {
  id: string;
  type: string;
  embedding: number[];
  successRate: number;
  useCount: number;
  lastUsed: string;
}

// Bounds
const MAX_DIMENSION = 4096;
const MAX_AGENTS = 1000;
const MAX_LORA_RANK = 64;
const MAX_AGENT_ID_LENGTH = 256;
const MAX_PATTERNS_K = 100;

/**
 * FederatedSessionManager - Orchestrates cross-session federated learning
 */
export class FederatedSessionManager {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private coordinator: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private activeSessions = new Map<string, any>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private loraManager: any = null;
  private dim: number;
  private maxAgents = 100;
  private config: FederatedConfig;
  private _destroyed = false;

  private constructor(config: FederatedConfig) {
    this.dim = config.dimension;
    this.config = config;
  }

  /**
   * Create a new federated session manager.
   * Lazy-loads @ruvector/ruvllm to avoid hard dependency.
   */
  static async create(config: FederatedConfig): Promise<FederatedSessionManager> {
    if (!Number.isFinite(config.dimension) || config.dimension < 1 || config.dimension > MAX_DIMENSION) {
      throw new Error(`dimension must be between 1 and ${MAX_DIMENSION}`);
    }

    const instance = new FederatedSessionManager(config);
    const maxAgents = Math.min(Math.max(1, config.maxAgents ?? 100), MAX_AGENTS);
    const loraRank = Math.min(Math.max(1, config.loraRank ?? 4), MAX_LORA_RANK);

    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ruvllm: any = await import('@ruvector/ruvllm');

      // Signature is (coordinatorId, config); passing the config first made
      // it the id and left hiddenDim/embeddingDim at ruvllm's default of 256.
      instance.coordinator = new ruvllm.FederatedCoordinator(`agentdb-${Date.now().toString(36)}`, {
        hiddenDim: config.dimension,
        embeddingDim: config.dimension,
        ...(config.ewcLambda !== undefined ? { ewcLambda: config.ewcLambda } : {}),
      });
      instance.maxAgents = maxAgents;

      if (config.qualityThreshold !== undefined) {
        instance.coordinator.setQualityThreshold(
          Math.min(Math.max(0, config.qualityThreshold), 1),
        );
      }

      if (config.consolidationInterval !== undefined) {
        instance.coordinator.setConsolidationInterval(
          Math.max(1, config.consolidationInterval),
        );
      }

      // Initialize LoRA manager for task-specific adapters
      instance.loraManager = new ruvllm.LoraManager();
      // Create a default adapter
      instance.loraManager.create('default', {
        inputDim: config.dimension,
        outputDim: config.dimension,
        rank: loraRank,
        alpha: config.loraAlpha ?? loraRank * 2,
      });
      instance.loraManager.activate('default');
    } catch (error) {
      throw new Error(
        `Federated session manager initialization failed.\n` +
        `Install with: npm install @ruvector/ruvllm\n` +
        `Error: ${(error as Error).message}`,
      );
    }

    return instance;
  }

  /**
   * Check if @ruvector/ruvllm is available.
   */
  static async isAvailable(): Promise<boolean> {
    try {
      await import('@ruvector/ruvllm');
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Begin a new agent session.
   * Returns a session handle for recording trajectories.
   *
   * @param agentId - Unique identifier for this agent session
   * @param warmStart - Whether to load initial patterns from coordinator (default: true)
   */
  async beginSession(agentId: string, warmStart: boolean = true): Promise<SessionHandle> {
    this.ensureAlive();
    this.validateAgentId(agentId);

    if (this.activeSessions.has(agentId)) {
      throw new Error(`Session already active for agent: ${agentId}`);
    }
    if (this.activeSessions.size >= this.maxAgents) {
      throw new Error(`Maximum concurrent sessions reached (${this.maxAgents})`);
    }

    const agent = this.coordinator.createAgent(agentId);
    this.activeSessions.set(agentId, agent);

    // Warm-start with coordinator patterns
    let initialPatterns: FederatedPattern[] = [];
    if (warmStart) {
      initialPatterns = this.coordinator.getInitialPatterns(10) as FederatedPattern[];
    }

    return new SessionHandle(
      agentId,
      agent,
      this,
      initialPatterns,
      this.dim,
    );
  }

  /**
   * End a session, aggregate its trajectories, and clean up.
   *
   * @param agentId - The agent session to end
   * @returns Session statistics
   */
  endSession(agentId: string): SessionStats {
    this.ensureAlive();
    const agent = this.activeSessions.get(agentId);
    if (!agent) {
      throw new Error(`No active session for agent: ${agentId}`);
    }

    // Force learn to extract patterns
    agent.forceLearn();

    // Aggregate into coordinator
    this.coordinator.aggregate(agent);

    // Extract stats before cleanup
    const stats: SessionStats = {
      trajectoryCount: agent.trajectoryCount(),
      avgQuality: agent.avgQuality() || 0,
      patternsLearned: (agent.getPatterns(MAX_PATTERNS_K) as FederatedPattern[]).length,
      sessionDurationMs: agent.uptimeSeconds() * 1000,
    };

    this.activeSessions.delete(agentId);

    // Auto-consolidate if coordinator says it's time
    if (this.coordinator.shouldConsolidate()) {
      this.coordinator.forceConsolidate();
    }

    return stats;
  }

  /**
   * Export a session's state for persistence.
   * The state can be re-imported in a future session.
   */
  exportSession(agentId: string): SessionState {
    this.ensureAlive();
    const agent = this.activeSessions.get(agentId);
    if (!agent) {
      throw new Error(`No active session for agent: ${agentId}`);
    }
    return agent.exportState() as SessionState;
  }

  /**
   * Force consolidation across all aggregated agent data.
   * Useful for periodic maintenance (e.g., NightlyLearner).
   */
  consolidate(): void {
    this.ensureAlive();
    this.coordinator.forceConsolidate();
  }

  /**
   * Get warm-start patterns for a new session.
   */
  getInitialPatterns(k: number = 10): FederatedPattern[] {
    this.ensureAlive();
    const safeK = Math.min(Math.max(1, k), MAX_PATTERNS_K);
    return this.coordinator.getInitialPatterns(safeK) as FederatedPattern[];
  }

  /**
   * Find patterns similar to a query embedding.
   */
  findPatterns(query: number[], k: number = 5): FederatedPattern[] {
    this.ensureAlive();
    if (query.length !== this.dim) {
      throw new Error(`Query dimension ${query.length} does not match configured dimension ${this.dim}`);
    }
    const safeK = Math.min(Math.max(1, k), MAX_PATTERNS_K);
    return this.coordinator.findPatterns(query, safeK) as FederatedPattern[];
  }

  /**
   * Apply the global LoRA transformation to an embedding.
   */
  applyLora(embedding: number[]): number[] {
    this.ensureAlive();
    if (!this.loraManager) {
      return embedding;
    }
    return this.loraManager.forward(embedding) as number[];
  }

  /**
   * Create a task-specific LoRA adapter.
   */
  createAdapter(name: string, rank?: number): void {
    this.ensureAlive();
    if (!this.loraManager) return;

    const safeRank = Math.min(Math.max(1, rank ?? this.config.loraRank ?? 4), MAX_LORA_RANK);
    this.loraManager.create(name, {
      inputDim: this.dim,
      outputDim: this.dim,
      rank: safeRank,
      alpha: safeRank * 2,
    });
  }

  /**
   * Activate a specific LoRA adapter.
   */
  activateAdapter(name: string): void {
    this.ensureAlive();
    if (!this.loraManager) return;
    this.loraManager.activate(name);
  }

  /**
   * List all registered LoRA adapters.
   */
  listAdapters(): string[] {
    if (!this.loraManager) return [];
    return this.loraManager.list() as string[];
  }

  /**
   * Get federated coordinator statistics.
   */
  getStats(): FederatedStats {
    this.ensureAlive();
    const raw = this.coordinator.stats() as {
      totalAgents: number;
      totalTrajectories: number;
      patternsLearned: number;
      avgQuality: number;
      qualityThreshold: number;
    };
    const contributions = this.coordinator.getContributions() as Record<string, number>;

    return {
      totalAgents: raw.totalAgents,
      totalTrajectories: raw.totalTrajectories,
      patternsLearned: raw.patternsLearned,
      avgQuality: raw.avgQuality,
      qualityThreshold: raw.qualityThreshold,
      contributions,
    };
  }

  /** Get the number of active sessions */
  get activeSessionCount(): number {
    return this.activeSessions.size;
  }

  /** Get the configured dimension */
  get dimension(): number {
    return this.dim;
  }

  /** Check if destroyed */
  get isDestroyed(): boolean {
    return this._destroyed;
  }

  /** Destroy the manager and all active sessions */
  destroy(): void {
    if (!this._destroyed) {
      this.activeSessions.clear();
      if (this.loraManager) {
        this.loraManager.clear();
        this.loraManager = null;
      }
      if (this.coordinator) {
        this.coordinator.clear();
        this.coordinator = null;
      }
      this._destroyed = true;
    }
  }

  // --- Internal ---

  /** @internal Record a trajectory from a session handle */
  _recordTrajectory(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    agent: any,
    embedding: number[],
    quality: number,
    route?: string,
  ): void {
    const safeQuality = Math.min(Math.max(0, quality), 1);
    if (route) {
      agent.processTaskWithRoute(embedding, route, safeQuality);
    } else {
      agent.processTask(embedding, safeQuality);
    }
  }

  private validateAgentId(id: string): void {
    if (!id || id.length === 0 || id.length > MAX_AGENT_ID_LENGTH) {
      throw new Error(`Agent ID must be 1-${MAX_AGENT_ID_LENGTH} characters`);
    }
    if (id.includes('\0')) {
      throw new Error('Agent ID must not contain null bytes');
    }
  }

  private ensureAlive(): void {
    if (this._destroyed) {
      throw new Error('FederatedSessionManager has been destroyed. Create a new instance.');
    }
  }
}

/**
 * SessionHandle - Lightweight handle for recording trajectories within a session
 */
export class SessionHandle {
  private agentId: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private agent: any;
  private manager: FederatedSessionManager;
  private _initialPatterns: FederatedPattern[];
  private _dim: number;
  private _ended = false;

  /** @internal */
  constructor(
    agentId: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    agent: any,
    manager: FederatedSessionManager,
    initialPatterns: FederatedPattern[],
    dim: number,
  ) {
    this.agentId = agentId;
    this.agent = agent;
    this.manager = manager;
    this._initialPatterns = initialPatterns;
    this._dim = dim;
  }

  /**
   * Record a trajectory (embedding + quality score).
   *
   * @param embedding - The query/action embedding
   * @param quality - Quality score in [0.0, 1.0]
   * @param route - Optional route label (e.g., "math", "code")
   */
  recordTrajectory(embedding: number[] | Float32Array, quality: number, route?: string): void {
    this.ensureActive();
    const arr = embedding instanceof Float32Array ? Array.from(embedding) : embedding;
    if (arr.length !== this._dim) {
      throw new Error(`Embedding dimension ${arr.length} does not match configured dimension ${this._dim}`);
    }
    this.manager._recordTrajectory(this.agent, arr, quality, route);
  }

  /**
   * Get patterns learned during this session.
   */
  getPatterns(k: number = 10): FederatedPattern[] {
    this.ensureActive();
    this.agent.forceLearn();
    return this.agent.getPatterns(Math.min(Math.max(1, k), MAX_PATTERNS_K)) as FederatedPattern[];
  }

  /**
   * Get the initial warm-start patterns loaded at session start.
   */
  get initialPatterns(): FederatedPattern[] {
    return this._initialPatterns;
  }

  /**
   * Get current session statistics.
   */
  getStats(): SessionStats {
    this.ensureActive();
    return {
      trajectoryCount: this.agent.trajectoryCount() as number,
      avgQuality: this.agent.avgQuality() as number || 0,
      patternsLearned: (this.agent.getPatterns(MAX_PATTERNS_K) as FederatedPattern[]).length,
      sessionDurationMs: (this.agent.uptimeSeconds() as number) * 1000,
    };
  }

  /**
   * End this session and aggregate into the coordinator.
   */
  end(): SessionStats {
    this.ensureActive();
    this._ended = true;
    return this.manager.endSession(this.agentId);
  }

  /** The agent ID for this session */
  get id(): string {
    return this.agentId;
  }

  /** Whether this session has been ended */
  get isEnded(): boolean {
    return this._ended;
  }

  private ensureActive(): void {
    if (this._ended) {
      throw new Error(`Session ${this.agentId} has already ended.`);
    }
  }
}
