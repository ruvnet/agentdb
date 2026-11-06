/**
 * AgentDB Init Command - Initialize database with backend detection
 */

import { detectBackend, formatDetectionResult, type DetectionResult } from '../../backends/detector.js';
import { createDatabase } from '../../db-fallback.js';

// Color codes for beautiful output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
};

interface InitOptions {
  backend?: 'auto' | 'ruvector' | 'hnswlib';
  dimension?: number;
  model?: string;
  preset?: 'small' | 'medium' | 'large';
  inMemory?: boolean;
  dryRun?: boolean;
  dbPath?: string;
}

function printDetectionInfo(detection: DetectionResult): void {
  console.log(`\n${colors.bright}${colors.cyan}🔍 AgentDB v2 - Backend Detection${colors.reset}\n`);
  console.log(formatDetectionResult(detection));
}

function getBackendColor(backend: 'ruvector' | 'hnswlib'): string {
  return backend === 'ruvector' ? colors.green : colors.yellow;
}

export async function initCommand(options: InitOptions = {}): Promise<void> {
  const {
    backend = 'auto',
    dimension = 384,
    model,
    preset,
    inMemory = false,
    dryRun = false,
    dbPath = './agentdb.db'
  } = options;

  try {
    // Detect available backends
    const detection = await detectBackend();

    if (dryRun) {
      printDetectionInfo(detection);
      return;
    }

    // Validate backend selection
    if (backend === 'ruvector' && detection.backend !== 'ruvector') {
      console.error(`${colors.red}❌ Error: RuVector not available${colors.reset}`);
      console.error(`   Install with: ${colors.cyan}npm install @ruvector/core${colors.reset}`);
      process.exit(1);
    }

    if (backend === 'hnswlib' && detection.backend !== 'hnswlib') {
      console.error(`${colors.red}❌ Error: HNSWLib not available${colors.reset}`);
      console.error(`   Install with: ${colors.cyan}npm install hnswlib-node${colors.reset}`);
      process.exit(1);
    }

    // Determine actual backend to use
    const selectedBackend = backend === 'auto' ? detection.backend : backend;

    // Determine actual database path (handle in-memory)
    const actualDbPath = inMemory ? ':memory:' : dbPath;

    // Determine embedding model (with dimension-aware defaults)
    const embeddingModel = model || (dimension === 768 ? 'Xenova/bge-base-en-v1.5' : 'Xenova/all-MiniLM-L6-v2');

    console.log(`\n${colors.bright}${colors.cyan}🚀 Initializing AgentDB${colors.reset}\n`);
    console.log(`  Database:      ${colors.blue}${actualDbPath}${colors.reset}`);
    console.log(`  Backend:       ${getBackendColor(selectedBackend)}${selectedBackend}${colors.reset}`);
    console.log(`  Dimension:     ${colors.blue}${dimension}${colors.reset}`);
    console.log(`  Model:         ${colors.blue}${embeddingModel}${colors.reset}`);
    if (preset) {
      console.log(`  Preset:        ${colors.blue}${preset}${colors.reset}`);
    }
    console.log('');

    // Initialize database
    const db = await createDatabase(actualDbPath);

    // Configure for performance
    db.pragma('journal_mode = WAL');
    db.pragma('synchronous = NORMAL');
    db.pragma('cache_size = -64000');

    // Load schemas from the inlined constants (generated from src/schemas/*.sql
    // at build time), as AgentDB.loadSchemas() does: resolving dist/schemas
    // relative to this file misses when run from source (tsx) or other layouts.
    const { SCHEMA_SQL, FRONTIER_SCHEMA_SQL } = await import('../../schemas/inline.js');
    db.exec(SCHEMA_SQL);
    db.exec(FRONTIER_SCHEMA_SQL);

    // Store backend configuration
    db.prepare(`
      CREATE TABLE IF NOT EXISTS agentdb_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER DEFAULT (strftime('%s', 'now'))
      )
    `).run();

    db.prepare(`
      INSERT OR REPLACE INTO agentdb_config (key, value)
      VALUES (?, ?)
    `).run('backend', selectedBackend);

    db.prepare(`
      INSERT OR REPLACE INTO agentdb_config (key, value)
      VALUES (?, ?)
    `).run('dimension', dimension.toString());

    db.prepare(`
      INSERT OR REPLACE INTO agentdb_config (key, value)
      VALUES (?, ?)
    `).run('embedding_model', embeddingModel);

    if (preset) {
      db.prepare(`
        INSERT OR REPLACE INTO agentdb_config (key, value)
        VALUES (?, ?)
      `).run('preset', preset);
    }

    db.prepare(`
      INSERT OR REPLACE INTO agentdb_config (key, value)
      VALUES (?, ?)
    `).run('version', '2.0.0');

    db.close();

    console.log(`${colors.green}✅ AgentDB initialized successfully${colors.reset}\n`);

    if (selectedBackend === 'ruvector' && detection.features.gnn) {
      console.log(`${colors.bright}${colors.magenta}🧠 Bonus:${colors.reset} GNN self-learning available`);
      console.log(`   Use ${colors.cyan}agentdb train${colors.reset} to enable adaptive patterns\n`);
    }

    if (selectedBackend === 'hnswlib') {
      console.log(`${colors.yellow}💡 Tip:${colors.reset} Install RuVector for 150x performance boost`);
      console.log(`   ${colors.cyan}npm install @ruvector/core${colors.reset}\n`);
    }

  } catch (error) {
    console.error(`${colors.red}❌ Initialization failed:${colors.reset}`);
    console.error(`   ${(error as Error).message}`);
    process.exit(1);
  }
}
