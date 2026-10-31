/**
 * AgentDB - Main database wrapper class
 *
 * Provides a unified interface to all AgentDB controllers with:
 * - sql.js WASM for relational storage (with better-sqlite3 fallback)
 * - RuVector for optimized vector search (150x faster than SQLite)
 * - Unified integration passing vector backend to all controllers
 */
import { ReflexionMemory } from '../controllers/ReflexionMemory.js';
import { SkillLibrary } from '../controllers/SkillLibrary.js';
import { CausalMemoryGraph } from '../controllers/CausalMemoryGraph.js';
import { EmbeddingService } from '../controllers/EmbeddingService.js';
import { createBackend } from '../backends/factory.js';
import type { VectorBackend, VectorConfig } from '../backends/VectorBackend.js';
import type { IDatabaseConnection } from '../types/database.types.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export interface AgentDBConfig {
  dbPath?: string;
  namespace?: string;
  enableAttention?: boolean;
  attentionConfig?: Record<string, any>;
  /** Force use of sql.js WASM even if better-sqlite3 is available */
  forceWasm?: boolean;
  /** Vector backend type: 'auto' | 'ruvector' | 'hnswlib' */
  vectorBackend?: 'auto' | 'ruvector' | 'hnswlib';
  /** Vector dimension (default: 384 for MiniLM) */
  vectorDimension?: number;
}

export class AgentDB {
  private db!: IDatabaseConnection;
  private reflexion!: ReflexionMemory;
  private skills!: SkillLibrary;
  private causalGraph!: CausalMemoryGraph;
  private embedder!: EmbeddingService;
  public vectorBackend!: VectorBackend;
  private initialized = false;
  private config: AgentDBConfig;
  private usingWasm = false;

  constructor(config: AgentDBConfig = {}) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    const dbPath = this.config.dbPath || ':memory:';
    const vectorDimension = this.config.vectorDimension || 384;

    // Initialize database with unified fallback system
    this.db = await this.initializeDatabase(dbPath);

    // Load schemas
    await this.loadSchemas();

    // Initialize embedder with default Xenova model
    this.embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: vectorDimension,
      provider: 'transformers'
    });
    await this.embedder.initialize();

    // Initialize vector backend (RuVector preferred, HNSWLib fallback)
    // A file-backed AgentDB keeps its vector index beside the database; an
    // in-memory one lets the backend use a private temp location.
    this.vectorBackend = await createBackend(this.config.vectorBackend || 'auto', {
      dimensions: vectorDimension,
      metric: 'cosine',
      storagePath: dbPath === ':memory:' ? undefined : `${dbPath}.ruvector`,
    } as VectorConfig);

    // Initialize controllers WITH vector backend for optimized search
    // This enables 150x faster vector search via RuVector instead of SQLite brute-force
    this.reflexion = new ReflexionMemory(this.db, this.embedder, this.vectorBackend);
    this.skills = new SkillLibrary(this.db, this.embedder, this.vectorBackend);
    this.causalGraph = new CausalMemoryGraph(
      this.db,
      undefined, // graphBackend - not used in default initialization
      this.embedder,
      undefined, // config - use defaults
      this.vectorBackend
    );

    this.initialized = true;

    console.log(`[AgentDB] Initialized with ${this.usingWasm ? 'sql.js WASM' : 'better-sqlite3'} + ${this.vectorBackend.name} vector backend`);
  }

  /**
   * Initialize database with automatic fallback:
   * 1. Try better-sqlite3 (native, fastest)
   * 2. Fallback to sql.js WASM (no build tools required)
   */
  private async initializeDatabase(dbPath: string): Promise<IDatabaseConnection> {
    // Force WASM if requested
    if (this.config.forceWasm) {
      return this.initializeSqlJsWasm(dbPath);
    }

    // Try better-sqlite3 first (native performance)
    try {
      const Database = (await import('better-sqlite3')).default;
      const db = new Database(dbPath);
      db.pragma('journal_mode = WAL');
      this.usingWasm = false;
      return db as unknown as IDatabaseConnection;
    } catch (error) {
      // better-sqlite3 not available or failed, try sql.js WASM
      console.log('[AgentDB] better-sqlite3 not available, using sql.js WASM');
      return this.initializeSqlJsWasm(dbPath);
    }
  }

  /**
   * Initialize sql.js WASM database
   */
  private async initializeSqlJsWasm(dbPath: string): Promise<IDatabaseConnection> {
    const { createDatabase } = await import('../db-fallback.js');
    const db = await createDatabase(dbPath);
    this.usingWasm = true;
    return db as IDatabaseConnection;
  }

  /**
   * Load database schemas.
   *
   * Uses inlined SQL constants (src/schemas/inline.ts, generated from the .sql
   * files at build time) so this works in:
   *   - the browser (no `fs` available) — fixes #2
   *   - globally-installed CLIs where __dirname resolution misses dist/schemas
   *     (the "Schema file not found" warning at install time) — fixes #1
   */
  private async loadSchemas(): Promise<void> {
    const { SCHEMA_SQL, FRONTIER_SCHEMA_SQL } = await import('../schemas/inline.js');
    if (SCHEMA_SQL) this.db.exec(SCHEMA_SQL);
    if (FRONTIER_SCHEMA_SQL) this.db.exec(FRONTIER_SCHEMA_SQL);
  }

  getController(name: string): any {
    if (!this.initialized) {
      throw new Error('AgentDB not initialized. Call initialize() first.');
    }

    switch (name) {
      case 'memory':
      case 'reflexion':
        return this.reflexion;
      case 'skills':
        return this.skills;
      case 'causal':
      case 'causalGraph':
        return this.causalGraph;
      default:
        throw new Error(`Unknown controller: ${name}`);
    }
  }

  async close(): Promise<void> {
    if (this.vectorBackend) {
      const backend = this.vectorBackend as VectorBackend & { flush?: () => Promise<void> };
      await backend.flush?.().catch(() => { /* already logged by the backend */ });
      backend.close();
    }
    if (this.db) {
      this.db.close();
    }
  }

  // Expose database for advanced usage
  get database(): IDatabaseConnection {
    return this.db;
  }

  // Check if using WASM backend
  get isWasm(): boolean {
    return this.usingWasm;
  }

  // Get vector backend info
  get vectorBackendName(): string {
    return this.vectorBackend?.name || 'none';
  }
}
