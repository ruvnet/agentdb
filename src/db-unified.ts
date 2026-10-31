/**
 * Unified Database Layer for AgentDB v2
 *
 * Architecture:
 * - PRIMARY: RuVector GraphDatabase (@ruvector/graph-node) for new databases
 * - FALLBACK: SQLite (sql.js) for legacy databases
 *
 * Detection Logic:
 * 1. Check if database file exists
 * 2. If exists, check file signature to determine type
 * 3. If new database or .graph extension → use GraphDatabase
 * 4. If .db extension and SQLite signature → use SQLite (legacy mode)
 *
 * Migration:
 * - Provides migration tool to convert SQLite → GraphDatabase
 * - Maintains backward compatibility with existing databases
 */

import { GraphDatabaseAdapter, type GraphDatabaseConfig } from './backends/graph/GraphDatabaseAdapter.js';
import { getDatabaseImplementation } from './db-fallback.js';
import * as fs from 'fs';
import * as path from 'path';

export type DatabaseMode = 'graph' | 'sqlite-legacy';

export interface UnifiedDatabaseConfig {
  path: string;
  dimensions?: number;       // Default: 384 (sentence-transformers standard)
  forceMode?: DatabaseMode;  // Force specific mode
  autoMigrate?: boolean;     // Auto-migrate SQLite → Graph
}

/**
 * Unified Database - Smart detection and mode selection
 */
export class UnifiedDatabase {
  private mode: DatabaseMode;
  private graphDb?: GraphDatabaseAdapter;
  private sqliteDb?: any;
  private config: UnifiedDatabaseConfig;

  constructor(config: UnifiedDatabaseConfig) {
    this.config = config;
    this.mode = 'graph'; // Default to graph mode
  }

  /**
   * Initialize database with automatic mode detection
   */
  async initialize(embedder: any): Promise<void> {
    const dbPath = this.config.path;

    // Check if user forced a specific mode
    if (this.config.forceMode) {
      this.mode = this.config.forceMode;
      await this.initializeMode(embedder);
      return;
    }

    // Auto-detect based on file extension and content
    if (fs.existsSync(dbPath)) {
      const ext = path.extname(dbPath);

      // .graph extension = always use graph mode
      if (ext === '.graph') {
        this.mode = 'graph';
        console.error('🔍 Detected .graph extension → Using RuVector GraphDatabase');
      }
      // .db extension = check if it's SQLite
      else if (ext === '.db') {
        const isLegacySQLite = await this.isSQLiteDatabase(dbPath);

        if (isLegacySQLite) {
          this.mode = 'sqlite-legacy';
          console.error('🔍 Detected legacy SQLite database');

          // Offer migration if autoMigrate is enabled
          if (this.config.autoMigrate) {
            console.error('🔄 Auto-migration enabled, will migrate to GraphDatabase...');
            await this.migrateSQLiteToGraph(dbPath, embedder);
            this.mode = 'graph';
            // Migration already initialized graphDb, skip initializeMode
            return;
          } else {
            console.error('ℹ️  Running in legacy SQLite mode');
            console.error('💡 To migrate to RuVector Graph: set autoMigrate: true');
          }
        } else {
          // Not SQLite, use graph mode
          this.mode = 'graph';
          console.error('🔍 Using RuVector GraphDatabase');
        }
      } else {
        // Unknown extension, default to graph
        this.mode = 'graph';
      }
    } else {
      // New database - use graph mode (recommended)
      this.mode = 'graph';
      console.error('✨ Creating new RuVector GraphDatabase');

      // Suggest .graph extension if not using it
      if (!dbPath.endsWith('.graph') && !dbPath.endsWith('.db')) {
        console.error('💡 Tip: Use .graph extension for clarity (e.g., agentdb.graph)');
      }
    }

    await this.initializeMode(embedder);
  }

  /**
   * Initialize the selected database mode
   */
  private async initializeMode(embedder: any): Promise<void> {
    if (this.mode === 'graph') {
      // Use RuVector GraphDatabase
      const config: GraphDatabaseConfig = {
        storagePath: this.config.path,
        dimensions: this.config.dimensions || 384,
        distanceMetric: 'Cosine'
      };

      this.graphDb = new GraphDatabaseAdapter(config, embedder);
      await this.graphDb.initialize();

      console.error('✅ RuVector GraphDatabase ready (Primary Mode)');
      console.error('   - Cypher queries enabled');
      console.error('   - Hypergraph support active');
      console.error('   - ACID transactions available');
      console.error('   - 131K+ ops/sec batch inserts');
    } else {
      // Use legacy SQLite — db-fallback prefers native better-sqlite3 when
      // loadable and falls back to sql.js (WASM) otherwise (ruflo #2235 A).
      const { createDatabase, getDatabaseInfo } = await import('./db-fallback.js');
      this.sqliteDb = await createDatabase(this.config.path);
      const info = getDatabaseInfo();

      console.error(`⚠️  Using legacy SQLite mode (${info.implementation})`);
      console.error('   - Limited to SQL queries');
      console.error('   - No hypergraph support');
      console.error('   - Consider migration to GraphDatabase');
    }
  }

  /**
   * Check if file is a SQLite database
   */
  private async isSQLiteDatabase(filePath: string): Promise<boolean> {
    try {
      const buffer = fs.readFileSync(filePath);

      // SQLite databases start with "SQLite format 3\0"
      const signature = buffer.slice(0, 16).toString();
      return signature.startsWith('SQLite format 3');
    } catch {
      return false;
    }
  }

  /**
   * Migrate SQLite database to RuVector GraphDatabase
   */
  private async migrateSQLiteToGraph(sqlitePath: string, embedder: any): Promise<void> {
    console.error('🔄 Starting migration from SQLite to RuVector Graph...');

    const startTime = Date.now();

    // Load SQLite database using ESM import
    const { createDatabase } = await import('./db-fallback.js');
    const sqliteDb = await createDatabase(sqlitePath);

    // Create new GraphDatabase
    const graphPath = sqlitePath.replace(/\.db$/, '.graph');
    const graphConfig: GraphDatabaseConfig = {
      storagePath: graphPath,
      dimensions: this.config.dimensions || 384,
      distanceMetric: 'Cosine'
    };

    const graphDb = new GraphDatabaseAdapter(graphConfig, embedder);
    await graphDb.initialize();

    // Migrate episodes
    console.error('   📦 Migrating episodes...');
    const episodes = sqliteDb.prepare('SELECT * FROM episodes').all();

    for (const ep of episodes) {
      // Generate embedding for episode
      const text = `${ep.task} ${ep.input || ''} ${ep.output || ''}`;
      const embedding = await embedder.embed(text);

      await graphDb.storeEpisode({
        id: `ep-${ep.id}`,
        sessionId: ep.session_id,
        task: ep.task,
        reward: ep.reward,
        success: ep.success === 1,
        input: ep.input,
        output: ep.output,
        critique: ep.critique,
        createdAt: ep.created_at,
        tokensUsed: ep.tokens_used,
        latencyMs: ep.latency_ms
      }, embedding);
    }
    console.error(`   ✅ Migrated ${episodes.length} episodes`);

    // Migrate skills
    console.error('   📦 Migrating skills...');
    const skills = sqliteDb.prepare('SELECT * FROM skills').all();

    for (const skill of skills) {
      const text = `${skill.name} ${skill.description} ${skill.code}`;
      const embedding = await embedder.embed(text);

      await graphDb.storeSkill({
        id: `skill-${skill.id}`,
        name: skill.name,
        description: skill.description,
        code: skill.code,
        usageCount: skill.usage_count,
        avgReward: skill.avg_reward,
        createdAt: skill.created_at,
        updatedAt: skill.updated_at,
        tags: skill.tags
      }, embedding);
    }
    console.error(`   ✅ Migrated ${skills.length} skills`);

    // Migrate causal edges
    console.error('   📦 Migrating causal relationships...');
    const edges = sqliteDb.prepare('SELECT * FROM causal_edges').all();

    for (const edge of edges) {
      const text = edge.mechanism;
      const embedding = await embedder.embed(text);

      await graphDb.createCausalEdge({
        from: `ep-${edge.from_memory_id}`,
        to: `ep-${edge.to_memory_id}`,
        mechanism: edge.mechanism,
        uplift: edge.uplift,
        confidence: edge.confidence,
        sampleSize: edge.sample_size
      }, embedding);
    }
    console.error(`   ✅ Migrated ${edges.length} causal edges`);

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`\n🎉 Migration complete in ${duration}s!`);
    console.error(`   Old SQLite: ${sqlitePath}`);
    console.error(`   New Graph: ${graphPath}`);
    console.error(`\n💡 Backup your SQLite file and update path to use .graph`);

    // Close SQLite
    sqliteDb.close();

    // Update config to use new graph database
    this.config.path = graphPath;
    this.graphDb = graphDb;
  }

  /**
   * Get the active database mode
   */
  getMode(): DatabaseMode {
    return this.mode;
  }

  /**
   * Get the graph database (if in graph mode)
   */
  getGraphDatabase(): GraphDatabaseAdapter | undefined {
    return this.graphDb;
  }

  /**
   * Get the SQLite database (if in legacy mode)
   */
  getSQLiteDatabase(): any | undefined {
    return this.sqliteDb;
  }

  /**
   * Execute a query (auto-routes to correct database)
   */
  async query(queryOrCypher: string): Promise<any> {
    if (this.mode === 'graph') {
      // Execute Cypher query
      return await this.graphDb!.query(queryOrCypher);
    } else {
      // Execute SQL query
      return this.sqliteDb!.prepare(queryOrCypher).all();
    }
  }

  /**
   * Close database
   */
  close(): void {
    if (this.graphDb) {
      this.graphDb.close();
    }
    if (this.sqliteDb) {
      this.sqliteDb.close();
    }
  }
}

/**
 * Create unified database (smart mode detection)
 */
export async function createUnifiedDatabase(
  path: string,
  embedder: any,
  options?: Partial<UnifiedDatabaseConfig>
): Promise<UnifiedDatabase> {
  const config: UnifiedDatabaseConfig = {
    path,
    dimensions: options?.dimensions || 384,
    forceMode: options?.forceMode,
    autoMigrate: options?.autoMigrate ?? false  // Default: manual migration
  };

  const db = new UnifiedDatabase(config);
  await db.initialize(embedder);

  return db;
}
