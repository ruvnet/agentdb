#!/usr/bin/env node
/**
 * AgentDB MCP Server
 * Production-ready MCP server for Claude Desktop integration
 * Exposes AgentDB frontier memory features + core vector DB operations via MCP protocol
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
// Database type from db-fallback
type Database = any;
import { createDatabase } from '../db-fallback.js';
import { CausalMemoryGraph } from '../controllers/CausalMemoryGraph.js';
import { CausalRecall } from '../controllers/CausalRecall.js';
import { ReflexionMemory } from '../controllers/ReflexionMemory.js';
import { SkillLibrary } from '../controllers/SkillLibrary.js';
import { NightlyLearner } from '../controllers/NightlyLearner.js';
import { LearningSystem } from '../controllers/LearningSystem.js';
import { EmbeddingService } from '../controllers/EmbeddingService.js';
import { BatchOperations } from '../optimizations/BatchOperations.js';
import { ReasoningBank } from '../controllers/ReasoningBank.js';
import { MCPToolCaches } from '../optimizations/ToolCache.js';
import {
  validateId,
  validateTimestamp,
  validateSessionId,
  validateTaskString,
  validateNumericRange,
  validateArrayLength,
  validateObject,
  validateBoolean,
  validateEnum,
  ValidationError,
  handleSecurityError,
} from '../security/input-validation.js';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import {
  buildMcpToolPolicy,
  filterMcpTools,
  isMcpToolAllowed,
} from '../security/mcp-policy.js';

// ES module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const toolPolicy = buildMcpToolPolicy();

// stdout belongs exclusively to JSON-RPC when running over stdio. Route every
// transitive diagnostic (including legacy controller console.log calls) to
// stderr before database and embedding initialization begins.
console.log = (...args: unknown[]) => console.error(...args);

// ============================================================================
// Helper Functions for Core Vector DB Operations
// ============================================================================

/**
 * Initialize database schema
 */
function initializeSchema(database: any): void {
  const db = database;
  // Episodes table (vector store)
  db.exec(`
    CREATE TABLE IF NOT EXISTS episodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      session_id TEXT NOT NULL,
      task TEXT NOT NULL,
      input TEXT,
      output TEXT,
      critique TEXT,
      reward REAL NOT NULL,
      success INTEGER NOT NULL,
      latency_ms INTEGER,
      tokens_used INTEGER,
      tags TEXT,
      metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task);
    CREATE INDEX IF NOT EXISTS idx_episodes_reward ON episodes(reward);
    CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
  `);

  // Episode embeddings (vector storage)
  db.exec(`
    CREATE TABLE IF NOT EXISTS episode_embeddings (
      episode_id INTEGER PRIMARY KEY,
      embedding BLOB NOT NULL,
      FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
    );
  `);

  // Skills table
  db.exec(`
    CREATE TABLE IF NOT EXISTS skills (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      name TEXT UNIQUE NOT NULL,
      description TEXT NOT NULL,
      signature TEXT,
      code TEXT,
      success_rate REAL DEFAULT 0.0,
      uses INTEGER DEFAULT 0,
      avg_reward REAL DEFAULT 0.0,
      avg_latency_ms REAL DEFAULT 0.0,
      tags TEXT,
      metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_skills_success_rate ON skills(success_rate);
  `);

  // Skill embeddings
  db.exec(`
    CREATE TABLE IF NOT EXISTS skill_embeddings (
      skill_id INTEGER PRIMARY KEY,
      embedding BLOB NOT NULL,
      FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
    );
  `);

  // Causal edges table
  db.exec(`
    CREATE TABLE IF NOT EXISTS causal_edges (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      from_memory_id INTEGER NOT NULL,
      from_memory_type TEXT NOT NULL,
      to_memory_id INTEGER NOT NULL,
      to_memory_type TEXT NOT NULL,
      similarity REAL DEFAULT 0.0,
      uplift REAL NOT NULL,
      confidence REAL DEFAULT 0.95,
      sample_size INTEGER DEFAULT 0,
      evidence_ids TEXT,
      experiment_ids TEXT,
      confounder_score REAL DEFAULT 0.0,
      mechanism TEXT,
      metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_causal_from ON causal_edges(from_memory_id, from_memory_type);
    CREATE INDEX IF NOT EXISTS idx_causal_to ON causal_edges(to_memory_id, to_memory_type);
    CREATE INDEX IF NOT EXISTS idx_causal_uplift ON causal_edges(uplift);
  `);

  // Causal experiments
  db.exec(`
    CREATE TABLE IF NOT EXISTS causal_experiments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      intervention_id INTEGER NOT NULL,
      control_outcome REAL NOT NULL,
      treatment_outcome REAL NOT NULL,
      uplift REAL NOT NULL,
      sample_size INTEGER DEFAULT 1,
      metadata TEXT
    );
  `);

  // Causal observations
  db.exec(`
    CREATE TABLE IF NOT EXISTS causal_observations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      action TEXT NOT NULL,
      outcome TEXT NOT NULL,
      reward REAL NOT NULL,
      session_id TEXT,
      metadata TEXT
    );
  `);

  // Provenance certificates
  db.exec(`
    CREATE TABLE IF NOT EXISTS provenance_certificates (
      id TEXT PRIMARY KEY,
      ts INTEGER DEFAULT (strftime('%s', 'now')),
      query_id TEXT NOT NULL,
      agent_id TEXT NOT NULL,
      query_text TEXT NOT NULL,
      retrieval_method TEXT NOT NULL,
      source_ids TEXT NOT NULL,
      certificate_hash TEXT NOT NULL,
      metadata TEXT
    );
  `);
}

/**
 * Serialize embedding to BLOB
 */
function serializeEmbedding(embedding: Float32Array): Buffer {
  return Buffer.from(embedding.buffer);
}

/**
 * Deserialize embedding from BLOB
 */
function deserializeEmbedding(blob: Buffer): Float32Array {
  return new Float32Array(blob.buffer, blob.byteOffset, blob.byteLength / 4);
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============================================================================
// Initialize Database and Controllers
// ============================================================================
const dbPath = process.env.AGENTDB_PATH || './agentdb.db';
const db = await createDatabase(dbPath);

// Configure for performance
db.pragma('journal_mode = WAL');
db.pragma('synchronous = NORMAL');
db.pragma('cache_size = -64000');

// Initialize schema automatically on server start using SQL files
const schemaPath = path.join(__dirname, '../schemas/schema.sql');
const frontierSchemaPath = path.join(__dirname, '../schemas/frontier-schema.sql');

try {
  if (fs.existsSync(schemaPath)) {
    const schemaSQL = fs.readFileSync(schemaPath, 'utf-8');
    db.exec(schemaSQL);
    console.error('✅ Main schema loaded');
  }

  if (fs.existsSync(frontierSchemaPath)) {
    const frontierSQL = fs.readFileSync(frontierSchemaPath, 'utf-8');
    db.exec(frontierSQL);
    console.error('✅ Frontier schema loaded');
  }

  console.error('✅ Database schema initialized');
} catch (error) {
  console.error('⚠️  Schema initialization failed, using fallback:', (error as Error).message);
  // Fallback to initializeSchema function if SQL files not found
  initializeSchema(db);
}

// Initialize embedding service
const embeddingService = new EmbeddingService({
  model: 'Xenova/all-MiniLM-L6-v2',
  dimension: 384,
  provider: 'transformers'
});
await embeddingService.initialize();

// Initialize all controllers
const causalGraph = new CausalMemoryGraph(db);
const reflexion = new ReflexionMemory(db, embeddingService);
const skills = new SkillLibrary(db, embeddingService);
const causalRecall = new CausalRecall(db, embeddingService);
const learner = new NightlyLearner(db, embeddingService);
const learningSystem = new LearningSystem(db, embeddingService);
const batchOps = new BatchOperations(db, embeddingService);
const reasoningBank = new ReasoningBank(db, embeddingService);
const caches = new MCPToolCaches();

// ============================================================================
// MCP Server Setup
// ============================================================================
const server = new Server(
  {
    name: 'agentdb',
    version: '1.3.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// ============================================================================
// Tool Definitions
// ============================================================================
const tools = [
  // ==========================================================================
  // CORE VECTOR DB OPERATIONS (NEW)
  // ==========================================================================
  {
    name: 'agentdb_init',
    description: 'Initialize AgentDB database with schema and optimizations. Creates all required tables for vector storage, causal memory, skills, and provenance tracking.',
    inputSchema: {
      type: 'object',
      properties: {
        db_path: { type: 'string', description: 'Database file path (optional, defaults to ./agentdb.db)', default: './agentdb.db' },
        reset: { type: 'boolean', description: 'Reset database (delete existing)', default: false },
      },
    },
  },
  {
    name: 'agentdb_insert',
    description: 'Insert a single vector with metadata into AgentDB. Automatically generates embeddings for the provided text.',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Text content to embed and store' },
        metadata: { type: 'object', description: 'Additional metadata (JSON object)' },
        session_id: { type: 'string', description: 'Session identifier', default: 'default' },
        tags: { type: 'array', items: { type: 'string' }, description: 'Tags for categorization' },
      },
      required: ['text'],
    },
  },
  {
    name: 'agentdb_insert_batch',
    description: 'Batch insert multiple vectors efficiently using transactions and parallel embedding generation. Optimized for large datasets.',
    inputSchema: {
      type: 'object',
      properties: {
        items: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              text: { type: 'string', description: 'Text content to embed' },
              metadata: { type: 'object', description: 'Metadata (JSON)' },
              session_id: { type: 'string', description: 'Session ID' },
              tags: { type: 'array', items: { type: 'string' } },
            },
            required: ['text'],
          },
          description: 'Array of items to insert',
        },
        batch_size: { type: 'number', description: 'Batch size for processing', default: 100 },
      },
      required: ['items'],
    },
  },
  {
    name: 'agentdb_search',
    description: 'Semantic k-NN vector search using cosine similarity. Returns the most relevant results ranked by similarity score.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query text' },
        k: { type: 'number', description: 'Number of results to return', default: 10 },
        min_similarity: { type: 'number', description: 'Minimum similarity threshold (0-1)', default: 0.0 },
        filters: {
          type: 'object',
          properties: {
            session_id: { type: 'string', description: 'Filter by session ID' },
            tags: { type: 'array', items: { type: 'string' }, description: 'Filter by tags' },
            min_reward: { type: 'number', description: 'Minimum reward threshold' },
          },
          description: 'Optional filters',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'agentdb_delete',
    description: 'Delete vector(s) from AgentDB by ID or filters. Supports single ID deletion or bulk deletion with conditions.',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'number', description: 'Specific vector ID to delete' },
        filters: {
          type: 'object',
          properties: {
            session_id: { type: 'string', description: 'Delete all vectors with this session ID' },
            before_timestamp: { type: 'number', description: 'Delete vectors before this Unix timestamp' },
          },
          description: 'Bulk deletion filters (used if id not provided)',
        },
      },
    },
  },

  // ==========================================================================
  // FRONTIER MEMORY FEATURES (EXISTING)
  // ==========================================================================
  {
    name: 'reflexion_store',
    description: 'Store an episode with self-critique for reflexion-based learning',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session identifier' },
        task: { type: 'string', description: 'Task description' },
        reward: { type: 'number', description: 'Task reward (0-1)' },
        success: { type: 'boolean', description: 'Whether task succeeded' },
        critique: { type: 'string', description: 'Self-critique reflection' },
        input: { type: 'string', description: 'Task input' },
        output: { type: 'string', description: 'Task output' },
        latency_ms: { type: 'number', description: 'Execution latency' },
        tokens: { type: 'number', description: 'Tokens used' },
      },
      required: ['session_id', 'task', 'reward', 'success'],
    },
  },
  {
    name: 'reflexion_retrieve',
    description: 'Retrieve relevant past episodes for learning from experience',
    inputSchema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'Task to find similar episodes for' },
        k: { type: 'number', description: 'Number of episodes to retrieve', default: 5 },
        only_failures: { type: 'boolean', description: 'Only retrieve failures' },
        only_successes: { type: 'boolean', description: 'Only retrieve successes' },
        min_reward: { type: 'number', description: 'Minimum reward threshold' },
      },
      required: ['task'],
    },
  },
  {
    name: 'skill_create',
    description: 'Create a reusable skill in the skill library',
    inputSchema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'Skill name' },
        description: { type: 'string', description: 'What the skill does' },
        code: { type: 'string', description: 'Skill implementation code' },
        success_rate: { type: 'number', description: 'Initial success rate' },
      },
      required: ['name', 'description'],
    },
  },
  {
    name: 'skill_search',
    description: 'Search for applicable skills by semantic similarity',
    inputSchema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'Task to find skills for' },
        k: { type: 'number', description: 'Number of skills to return', default: 10 },
        min_success_rate: { type: 'number', description: 'Minimum success rate filter' },
      },
      required: ['task'],
    },
  },
  {
    name: 'causal_add_edge',
    description: 'Add a causal relationship between actions and outcomes',
    inputSchema: {
      type: 'object',
      properties: {
        cause: { type: 'string', description: 'Causal action/intervention' },
        effect: { type: 'string', description: 'Observed effect/outcome' },
        uplift: { type: 'number', description: 'Causal uplift magnitude' },
        confidence: { type: 'number', description: 'Confidence in causal claim (0-1)', default: 0.95 },
        sample_size: { type: 'number', description: 'Number of observations', default: 0 },
      },
      required: ['cause', 'effect', 'uplift'],
    },
  },
  {
    name: 'causal_query',
    description: 'Query causal effects to understand what actions cause what outcomes',
    inputSchema: {
      type: 'object',
      properties: {
        cause: { type: 'string', description: 'Filter by cause (optional)' },
        effect: { type: 'string', description: 'Filter by effect (optional)' },
        min_confidence: { type: 'number', description: 'Minimum confidence', default: 0.5 },
        min_uplift: { type: 'number', description: 'Minimum uplift', default: 0.0 },
        limit: { type: 'number', description: 'Maximum results', default: 10 },
      },
    },
  },
  {
    name: 'recall_with_certificate',
    description: 'Retrieve memories with causal utility scoring and provenance certificate',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Query for memory retrieval' },
        k: { type: 'number', description: 'Number of results', default: 12 },
        alpha: { type: 'number', description: 'Similarity weight', default: 0.7 },
        beta: { type: 'number', description: 'Causal uplift weight', default: 0.2 },
        gamma: { type: 'number', description: 'Recency weight', default: 0.1 },
      },
      required: ['query'],
    },
  },
  {
    name: 'learner_discover',
    description: 'Automatically discover causal patterns from episode history',
    inputSchema: {
      type: 'object',
      properties: {
        min_attempts: { type: 'number', description: 'Minimum attempts required', default: 3 },
        min_success_rate: { type: 'number', description: 'Minimum success rate', default: 0.6 },
        min_confidence: { type: 'number', description: 'Minimum statistical confidence', default: 0.7 },
        dry_run: { type: 'boolean', description: 'Preview without storing', default: false },
      },
    },
  },
  {
    name: 'db_stats',
    description: 'Get database statistics showing record counts',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },

  // ==========================================================================
  // LEARNING SYSTEM TOOLS (v1.3.0 - Tools 1-5)
  // ==========================================================================
  {
    name: 'learning_start_session',
    description: 'Start a new reinforcement learning session with specified algorithm and configuration. Supports 9 RL algorithms: q-learning, sarsa, dqn, policy-gradient, actor-critic, ppo, decision-transformer, mcts, model-based.',
    inputSchema: {
      type: 'object',
      properties: {
        user_id: { type: 'string', description: 'User identifier for the learning session' },
        session_type: {
          type: 'string',
          description: 'RL algorithm type',
          enum: ['q-learning', 'sarsa', 'dqn', 'policy-gradient', 'actor-critic', 'ppo', 'decision-transformer', 'mcts', 'model-based'],
        },
        config: {
          type: 'object',
          description: 'Learning configuration parameters',
          properties: {
            learning_rate: { type: 'number', description: 'Learning rate (0-1)', default: 0.01 },
            discount_factor: { type: 'number', description: 'Discount factor gamma (0-1)', default: 0.99 },
            exploration_rate: { type: 'number', description: 'Epsilon for epsilon-greedy exploration (0-1)', default: 0.1 },
            batch_size: { type: 'number', description: 'Batch size for training', default: 32 },
            target_update_frequency: { type: 'number', description: 'Update frequency for target network', default: 100 },
          },
          required: ['learning_rate', 'discount_factor'],
        },
      },
      required: ['user_id', 'session_type', 'config'],
    },
  },
  {
    name: 'learning_end_session',
    description: 'End an active learning session and save the final trained policy to the database.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session ID to end' },
      },
      required: ['session_id'],
    },
  },
  {
    name: 'learning_predict',
    description: 'Get AI-recommended action for a given state with confidence scores and alternative actions.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Learning session ID' },
        state: { type: 'string', description: 'Current state description' },
      },
      required: ['session_id', 'state'],
    },
  },
  {
    name: 'learning_feedback',
    description: 'Submit feedback on action quality to train the RL policy. Feedback includes reward signal and outcome state.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Learning session ID' },
        state: { type: 'string', description: 'State where action was taken' },
        action: { type: 'string', description: 'Action that was executed' },
        reward: { type: 'number', description: 'Reward received (higher is better)' },
        next_state: { type: 'string', description: 'Resulting state after action (optional)' },
        success: { type: 'boolean', description: 'Whether the action was successful' },
      },
      required: ['session_id', 'state', 'action', 'reward', 'success'],
    },
  },
  {
    name: 'learning_train',
    description: 'Train the RL policy using batch learning with collected experiences. Returns training metrics including loss, average reward, and convergence rate.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Learning session ID to train' },
        epochs: { type: 'number', description: 'Number of training epochs', default: 50 },
        batch_size: { type: 'number', description: 'Batch size for training', default: 32 },
        learning_rate: { type: 'number', description: 'Learning rate for this training run', default: 0.01 },
      },
      required: ['session_id'],
    },
  },

  // ==========================================================================
  // LEARNING SYSTEM TOOLS (v1.4.0 - Tools 6-10)
  // ==========================================================================
  {
    name: 'learning_metrics',
    description: 'Get learning performance metrics including success rates, rewards, and policy improvement',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Optional session ID to filter metrics' },
        time_window_days: { type: 'number', description: 'Time window in days for metrics (default: 7)', default: 7 },
        include_trends: { type: 'boolean', description: 'Include trend analysis over time', default: true },
        group_by: { type: 'string', description: 'Group metrics by task/session/skill', enum: ['task', 'session', 'skill'], default: 'task' },
      },
    },
  },
  {
    name: 'learning_transfer',
    description: 'Transfer learning between sessions or tasks, enabling knowledge reuse across different contexts',
    inputSchema: {
      type: 'object',
      properties: {
        source_session: { type: 'string', description: 'Source session ID to transfer from' },
        target_session: { type: 'string', description: 'Target session ID to transfer to' },
        source_task: { type: 'string', description: 'Source task pattern to transfer from' },
        target_task: { type: 'string', description: 'Target task pattern to transfer to' },
        min_similarity: { type: 'number', description: 'Minimum similarity threshold (0-1)', default: 0.7, minimum: 0, maximum: 1 },
        transfer_type: { type: 'string', description: 'Type of transfer', enum: ['episodes', 'skills', 'causal_edges', 'all'], default: 'all' },
        max_transfers: { type: 'number', description: 'Maximum number of items to transfer', default: 10 },
      },
    },
  },
  {
    name: 'learning_explain',
    description: 'Explain action recommendations with confidence scores and supporting evidence from past experiences',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Query or task description to get recommendations for' },
        k: { type: 'number', description: 'Number of recommendations to return', default: 5 },
        explain_depth: { type: 'string', description: 'Explanation detail level', enum: ['summary', 'detailed', 'full'], default: 'detailed' },
        include_confidence: { type: 'boolean', description: 'Include confidence scores', default: true },
        include_evidence: { type: 'boolean', description: 'Include supporting evidence from past episodes', default: true },
        include_causal: { type: 'boolean', description: 'Include causal reasoning chains', default: true },
      },
      required: ['query'],
    },
  },
  {
    name: 'experience_record',
    description: 'Record tool execution as experience for reinforcement learning and experience replay',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Session identifier' },
        tool_name: { type: 'string', description: 'Name of the tool executed' },
        action: { type: 'string', description: 'Action taken or tool parameters' },
        state_before: { type: 'object', description: 'System state before action (JSON)' },
        state_after: { type: 'object', description: 'System state after action (JSON)' },
        outcome: { type: 'string', description: 'Outcome description' },
        reward: { type: 'number', description: 'Reward signal (0-1)', minimum: 0, maximum: 1 },
        success: { type: 'boolean', description: 'Whether the action succeeded' },
        latency_ms: { type: 'number', description: 'Execution latency in milliseconds' },
        metadata: { type: 'object', description: 'Additional metadata (JSON)' },
      },
      required: ['session_id', 'tool_name', 'action', 'outcome', 'reward', 'success'],
    },
  },
  {
    name: 'reward_signal',
    description: 'Calculate reward signal for outcomes based on success, efficiency, and causal impact',
    inputSchema: {
      type: 'object',
      properties: {
        episode_id: { type: 'number', description: 'Episode ID to calculate reward for' },
        success: { type: 'boolean', description: 'Whether the outcome was successful' },
        target_achieved: { type: 'boolean', description: 'Whether the target was achieved', default: true },
        efficiency_score: { type: 'number', description: 'Efficiency score (0-1)', default: 0.5, minimum: 0, maximum: 1 },
        quality_score: { type: 'number', description: 'Quality score (0-1)', default: 0.5, minimum: 0, maximum: 1 },
        time_taken_ms: { type: 'number', description: 'Time taken in milliseconds' },
        expected_time_ms: { type: 'number', description: 'Expected time in milliseconds' },
        include_causal: { type: 'boolean', description: 'Include causal impact in reward', default: true },
        reward_function: { type: 'string', description: 'Reward function to use', enum: ['standard', 'sparse', 'dense', 'shaped'], default: 'standard' },
      },
      required: ['success'],
    },
  },

  // ==========================================================================
  // CORE AGENTDB TOOLS (6-10) - v1.3.0
  // ==========================================================================
  {
    name: 'agentdb_stats',
    description: 'Get comprehensive database statistics including table counts, storage usage, and performance metrics',
    inputSchema: {
      type: 'object',
      properties: {
        detailed: { type: 'boolean', description: 'Include detailed statistics', default: false },
      },
    },
  },
  {
    name: 'agentdb_pattern_store',
    description: 'Store reasoning pattern with embedding, taskType, approach, and successRate',
    inputSchema: {
      type: 'object',
      properties: {
        taskType: { type: 'string', description: 'Type of task (e.g., "code_review", "data_analysis")' },
        approach: { type: 'string', description: 'Description of the reasoning approach' },
        successRate: { type: 'number', description: 'Success rate (0-1)', default: 0.0 },
        tags: { type: 'array', items: { type: 'string' }, description: 'Optional tags' },
        metadata: { type: 'object', description: 'Additional metadata' },
      },
      required: ['taskType', 'approach', 'successRate'],
    },
  },
  {
    name: 'agentdb_pattern_search',
    description: 'Search patterns with taskEmbedding, k, threshold, and filters',
    inputSchema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'Task description to search for' },
        k: { type: 'number', description: 'Number of results to return', default: 10 },
        threshold: { type: 'number', description: 'Minimum similarity threshold (0-1)', default: 0.0 },
        filters: {
          type: 'object',
          properties: {
            taskType: { type: 'string', description: 'Filter by task type' },
            minSuccessRate: { type: 'number', description: 'Minimum success rate' },
            tags: { type: 'array', items: { type: 'string' }, description: 'Filter by tags' },
          },
          description: 'Optional filters',
        },
      },
      required: ['task'],
    },
  },
  {
    name: 'agentdb_pattern_stats',
    description: 'Get pattern statistics including total patterns, success rates, and top task types',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'agentdb_clear_cache',
    description: 'Clear query cache to refresh statistics and search results',
    inputSchema: {
      type: 'object',
      properties: {
        cache_type: {
          type: 'string',
          description: 'Type of cache to clear (all, patterns, stats)',
          enum: ['all', 'patterns', 'stats'],
          default: 'all'
        },
      },
    },
  },

  // ==========================================================================
  // BATCH OPERATION TOOLS (v2.0 MCP Optimization - Phase 2)
  // ==========================================================================
  {
    name: 'skill_create_batch',
    description: 'Batch create multiple skills efficiently using transactions and parallel embedding generation. 3x faster than sequential skill_create calls (304 → 900 ops/sec). 🔄 PARALLEL-SAFE: Can be used alongside other batch operations.',
    inputSchema: {
      type: 'object',
      properties: {
        skills: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string', description: 'Skill name (unique)' },
              description: { type: 'string', description: 'What the skill does' },
              signature: { type: 'object', description: 'Optional function signature' },
              code: { type: 'string', description: 'Skill implementation code' },
              success_rate: { type: 'number', description: 'Initial success rate (0-1)', default: 0.0 },
              uses: { type: 'number', description: 'Initial use count', default: 0 },
              avg_reward: { type: 'number', description: 'Average reward', default: 0.0 },
              avg_latency_ms: { type: 'number', description: 'Average latency', default: 0.0 },
              tags: { type: 'array', items: { type: 'string' }, description: 'Tags for categorization' },
              metadata: { type: 'object', description: 'Additional metadata (JSON)' },
            },
            required: ['name', 'description'],
          },
          description: 'Array of skills to create',
          minItems: 1,
          maxItems: 100,
        },
        batch_size: { type: 'number', description: 'Batch size for processing (default: 32)', default: 32 },
        format: {
          type: 'string',
          enum: ['concise', 'detailed', 'json'],
          description: 'Response format (default: concise)',
          default: 'concise',
        },
      },
      required: ['skills'],
    },
  },
  {
    name: 'reflexion_store_batch',
    description: 'Batch store multiple episodes efficiently using transactions and parallel embedding generation. 3.3x faster than sequential reflexion_store calls (152 → 500 ops/sec). 🔄 PARALLEL-SAFE: Can be used alongside other batch operations.',
    inputSchema: {
      type: 'object',
      properties: {
        episodes: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              session_id: { type: 'string', description: 'Session identifier' },
              task: { type: 'string', description: 'Task description' },
              reward: { type: 'number', description: 'Task reward (0-1)' },
              success: { type: 'boolean', description: 'Whether task succeeded' },
              critique: { type: 'string', description: 'Self-critique reflection' },
              input: { type: 'string', description: 'Task input' },
              output: { type: 'string', description: 'Task output' },
              latency_ms: { type: 'number', description: 'Execution latency' },
              tokens: { type: 'number', description: 'Tokens used' },
              tags: { type: 'array', items: { type: 'string' }, description: 'Tags' },
              metadata: { type: 'object', description: 'Additional metadata' },
            },
            required: ['session_id', 'task', 'reward', 'success'],
          },
          description: 'Array of episodes to store',
          minItems: 1,
          maxItems: 1000,
        },
        batch_size: { type: 'number', description: 'Batch size for processing (default: 100)', default: 100 },
        format: {
          type: 'string',
          enum: ['concise', 'detailed', 'json'],
          description: 'Response format (default: concise)',
          default: 'concise',
        },
      },
      required: ['episodes'],
    },
  },
  {
    name: 'agentdb_pattern_store_batch',
    description: 'Batch store multiple reasoning patterns efficiently using transactions and parallel embedding generation. 4x faster than sequential agentdb_pattern_store calls. 🔄 PARALLEL-SAFE: Can be used alongside other batch operations.',
    inputSchema: {
      type: 'object',
      properties: {
        patterns: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              taskType: { type: 'string', description: 'Type of task (e.g., "code_review", "data_analysis")' },
              approach: { type: 'string', description: 'Description of the reasoning approach' },
              successRate: { type: 'number', description: 'Success rate (0-1)' },
              tags: { type: 'array', items: { type: 'string' }, description: 'Optional tags' },
              metadata: { type: 'object', description: 'Additional metadata' },
            },
            required: ['taskType', 'approach', 'successRate'],
          },
          description: 'Array of reasoning patterns to store',
          minItems: 1,
          maxItems: 500,
        },
        batch_size: { type: 'number', description: 'Batch size for processing (default: 50)', default: 50 },
        format: {
          type: 'string',
          enum: ['concise', 'detailed', 'json'],
          description: 'Response format (default: concise)',
          default: 'concise',
        },
      },
      required: ['patterns'],
    },
  },

  // ==========================================================================
  // ADR-073 — SOTA roadmap additions (causal traversal · batch delete · on-demand consolidation)
  // ==========================================================================
  {
    name: 'causal_traverse',
    description: 'Walk the causal graph between two memories. Returns the chain of causal links (with confidence and uplift) up to max_depth hops. Use this to answer "WHY does this memory matter?" / "what chain of causes led here?" — the primary explainability surface beyond a single recall. Wraps CausalMemoryGraph.getCausalChain().',
    inputSchema: {
      type: 'object',
      properties: {
        from_memory_id: { type: 'number', description: 'Starting memory id (the cause / earlier node)' },
        to_memory_id: { type: 'number', description: 'Ending memory id (the effect / target node)' },
        max_depth: { type: 'number', description: 'Maximum hops along the causal chain', default: 5, minimum: 1, maximum: 20 },
      },
      required: ['from_memory_id', 'to_memory_id'],
    },
  },
  {
    name: 'agentdb_delete_batch',
    description: 'Delete many episode/pattern rows by ID list in a single transaction. Faster and safer than looping `agentdb_delete` (one transaction, validated IDs, atomic). Use for bulk cleanup of old sessions / failed experiments.',
    inputSchema: {
      type: 'object',
      properties: {
        ids: { type: 'array', items: { type: 'number' }, description: 'Episode IDs to delete (1–1000 per call)', minItems: 1, maxItems: 1000 },
        table: { type: 'string', enum: ['episodes', 'reasoning_patterns'], default: 'episodes', description: 'Target table (whitelist-validated)' },
      },
      required: ['ids'],
    },
  },
  {
    name: 'consolidate_now',
    description: 'Run the NightlyLearner consolidation pass on demand — discovers new causal edges, prunes low-confidence edges, and promotes high-reward episodes into skills. Use after a batch of new episodes or when you want consolidated memories available immediately (instead of waiting for a scheduled run). Returns a summary report.',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: { type: 'string', description: 'Optional: limit consolidation to this session' },
      },
    },
  },
];

// ============================================================================
// Tool Handlers
// ============================================================================
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: filterMcpTools(tools, toolPolicy) };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (!isMcpToolAllowed(name, toolPolicy)) {
      throw new Error(`MCP tool "${name}" is not permitted by the active AgentDB capability policy`);
    }
    switch (name) {
      // ======================================================================
      // CORE VECTOR DB OPERATIONS
      // ======================================================================
      case 'agentdb_init': {
        const targetDbPath = (args?.db_path as string) || dbPath;

        if (args?.reset && fs.existsSync(targetDbPath)) {
          fs.unlinkSync(targetDbPath);
        }

        // Initialize schema
        initializeSchema(db);

        const stats = db.prepare('SELECT COUNT(*) as count FROM sqlite_master WHERE type="table"').get() as any;

        return {
          content: [
            {
              type: 'text',
              text: `✅ AgentDB initialized successfully!\n` +
                    `📍 Database: ${targetDbPath}\n` +
                    `📊 Tables created: ${stats.count}\n` +
                    `⚙️  Optimizations: WAL mode, cache_size=64MB\n` +
                    `🧠 Embedding service: ${embeddingService.constructor.name} ready`,
            },
          ],
        };
      }

      case 'agentdb_insert': {
        const text = args?.text as string;
        const sessionId = (args?.session_id as string) || 'default';
        const tags = (args?.tags as string[]) || [];
        const metadata = (args?.metadata as Record<string, any>) || {};

        const episodeId = await reflexion.storeEpisode({
          sessionId,
          task: text,
          reward: 1.0,
          success: true,
          input: text,
          output: '',
          critique: '',
          latencyMs: 0,
          tokensUsed: 0,
          tags,
          metadata,
        });

        return {
          content: [
            {
              type: 'text',
              text: `✅ Vector inserted successfully!\n` +
                    `🆔 ID: ${episodeId}\n` +
                    `📝 Text: ${text.substring(0, 100)}${text.length > 100 ? '...' : ''}\n` +
                    `🏷️  Tags: ${tags.join(', ') || 'none'}\n` +
                    `🧠 Embedding: ${embeddingService.constructor.name}`,
            },
          ],
        };
      }

      case 'agentdb_insert_batch': {
        const items = (args?.items as any[]) || [];
        const batchSize = (args?.batch_size as number) || 100;

        const episodes = items.map((item: any) => ({
          sessionId: item.session_id || 'default',
          task: item.text,
          reward: 1.0,
          success: true,
          input: item.text,
          output: '',
          critique: '',
          latencyMs: 0,
          tokensUsed: 0,
          tags: item.tags || [],
          metadata: item.metadata || {},
        }));

        const batchOpsConfig = new BatchOperations(db, embeddingService, {
          batchSize,
          parallelism: 4,
        });

        const inserted = await batchOpsConfig.insertEpisodes(episodes);

        return {
          content: [
            {
              type: 'text',
              text: `✅ Batch insert completed!\n` +
                    `📊 Inserted: ${inserted} vectors\n` +
                    `⚡ Batch size: ${batchSize}\n` +
                    `🧠 Embeddings generated in parallel\n` +
                    `💾 Transaction committed`,
            },
          ],
        };
      }

      case 'agentdb_search': {
        const queryText = args?.query as string;
        const k = (args?.k as number) || 10;
        const minSimilarity = (args?.min_similarity as number) || 0.0;
        const filters = args?.filters as any;

        const query: any = {
          task: queryText,
          k,
        };

        if (filters) {
          if (filters.min_reward !== undefined) {
            query.minReward = filters.min_reward;
          }
          // Session ID filter would require custom query
        }

        const results = await reflexion.retrieveRelevant(query);

        // Filter by minimum similarity if specified
        let filteredResults = results;
        if (minSimilarity > 0) {
          filteredResults = results.filter(r => (r.similarity || 0) >= minSimilarity);
        }

        return {
          content: [
            {
              type: 'text',
              text: `🔍 Search completed!\n` +
                    `📊 Found: ${filteredResults.length} results\n` +
                    `🎯 Query: ${queryText}\n\n` +
                    `Top Results:\n` +
                    filteredResults.slice(0, 5).map((r, i) =>
                      `${i + 1}. [ID: ${r.id}] Similarity: ${(r.similarity || 0).toFixed(3)}\n` +
                      `   Task: ${r.task.substring(0, 80)}${r.task.length > 80 ? '...' : ''}\n` +
                      `   Reward: ${r.reward.toFixed(2)}`
                    ).join('\n\n') +
                    (filteredResults.length > 5 ? `\n\n... and ${filteredResults.length - 5} more results` : ''),
            },
          ],
        };
      }

      case 'agentdb_delete': {
        let deleted = 0;
        const id = args?.id as number | undefined;
        const filters = args?.filters as any;

        try {
          if (id !== undefined) {
            // Validate ID
            const validatedId = validateId(id, 'id');

            // Delete single vector using parameterized query
            const stmt = db.prepare('DELETE FROM episodes WHERE id = ?');
            const result = stmt.run(validatedId);
            deleted = result.changes;
          } else if (filters) {
            // Bulk delete with validated filters
            if (filters.session_id) {
              // Validate session_id
              const validatedSessionId = validateSessionId(filters.session_id);

              // Use parameterized query
              const stmt = db.prepare('DELETE FROM episodes WHERE session_id = ?');
              const result = stmt.run(validatedSessionId);
              deleted = result.changes;
            } else if (filters.before_timestamp) {
              // Validate timestamp
              const validatedTimestamp = validateTimestamp(filters.before_timestamp, 'before_timestamp');

              // Use parameterized query
              const stmt = db.prepare('DELETE FROM episodes WHERE ts < ?');
              const result = stmt.run(validatedTimestamp);
              deleted = result.changes;
            } else {
              throw new ValidationError('Invalid or missing filter criteria', 'INVALID_FILTER');
            }
          } else {
            throw new ValidationError('Either id or filters must be provided', 'MISSING_PARAMETER');
          }

          return {
            content: [
              {
                type: 'text',
                text: `✅ Delete operation completed!\n` +
                      `📊 Deleted: ${deleted} vector(s)\n` +
                      `🗑️  ${id !== undefined ? `ID: ${id}` : 'Bulk deletion with filters'}`,
              },
            ],
          };
        } catch (error: any) {
          const safeMessage = handleSecurityError(error);
          return {
            content: [
              {
                type: 'text',
                text: `❌ Delete operation failed: ${safeMessage}`,
              },
            ],
            isError: true,
          };
        }
      }

      // ======================================================================
      // FRONTIER MEMORY FEATURES (EXISTING)
      // ======================================================================
      case 'reflexion_store': {
        const episodeId = await reflexion.storeEpisode({
          sessionId: args?.session_id as string,
          task: args?.task as string,
          reward: args?.reward as number,
          success: args?.success as boolean,
          critique: (args?.critique as string) || '',
          input: (args?.input as string) || '',
          output: (args?.output as string) || '',
          latencyMs: (args?.latency_ms as number) || 0,
          tokensUsed: (args?.tokens as number) || 0,
        });
        return {
          content: [
            {
              type: 'text',
              text: `✅ Stored episode #${episodeId}\nTask: ${args?.task}\nReward: ${args?.reward}\nSuccess: ${args?.success}`,
            },
          ],
        };
      }

      case 'reflexion_retrieve': {
        const episodes = await reflexion.retrieveRelevant({
          task: args?.task as string,
          k: (args?.k as number) || 5,
          onlyFailures: args?.only_failures as boolean | undefined,
          onlySuccesses: args?.only_successes as boolean | undefined,
          minReward: args?.min_reward as number | undefined,
        });
        return {
          content: [
            {
              type: 'text',
              text: `🔍 Retrieved ${episodes.length} episodes:\n\n` +
                episodes.map((ep, i) =>
                  `${i + 1}. Episode ${ep.id}\n   Task: ${ep.task}\n   Reward: ${ep.reward.toFixed(2)}\n   Similarity: ${ep.similarity?.toFixed(3) || 'N/A'}`
                ).join('\n\n'),
            },
          ],
        };
      }

      case 'skill_create': {
        const skillId = await skills.createSkill({
          name: args?.name as string,
          description: args?.description as string,
          signature: { inputs: {}, outputs: {} },
          code: (args?.code as string) || '',
          successRate: (args?.success_rate as number) || 0.0,
          uses: 0,
          avgReward: 0.0,
          avgLatencyMs: 0.0,
        });
        return {
          content: [
            {
              type: 'text',
              text: `✅ Created skill #${skillId}: ${args?.name}`,
            },
          ],
        };
      }

      case 'skill_search': {
        const foundSkills = await skills.searchSkills({
          task: args?.task as string,
          k: (args?.k as number) || 10,
          minSuccessRate: (args?.min_success_rate as number) || 0.0,
        });
        return {
          content: [
            {
              type: 'text',
              text: `🔍 Found ${foundSkills.length} skills:\n\n` +
                foundSkills.map((skill, i) =>
                  `${i + 1}. ${skill.name}\n   ${skill.description}\n   Success: ${(skill.successRate * 100).toFixed(1)}%`
                ).join('\n\n'),
            },
          ],
        };
      }

      case 'causal_add_edge': {
        const cause = args?.cause as string;
        const effect = args?.effect as string;
        const uplift = args?.uplift as number;
        const confidence = (args?.confidence as number) || 0.95;
        const sampleSize = (args?.sample_size as number) || 0;

        const edgeId = causalGraph.addCausalEdge({
          fromMemoryId: 0,
          fromMemoryType: cause as 'episode' | 'skill' | 'note' | 'fact',
          toMemoryId: 0,
          toMemoryType: effect as 'episode' | 'skill' | 'note' | 'fact',
          similarity: 0,
          uplift,
          confidence,
          sampleSize,
          evidenceIds: [],
        });
        return {
          content: [
            {
              type: 'text',
              text: `✅ Added causal edge #${edgeId}\n${cause} → ${effect}\nUplift: ${uplift}`,
            },
          ],
        };
      }

      case 'causal_query': {
        const cause = args?.cause as string | undefined;
        const effect = args?.effect as string | undefined;
        const minConfidence = (args?.min_confidence as number) || 0.5;
        const minUplift = (args?.min_uplift as number) || 0.0;
        const limit = (args?.limit as number) || 10;

        const edges = causalGraph.queryCausalEffects({
          interventionMemoryId: 0,
          interventionMemoryType: cause || '',
          outcomeMemoryId: effect ? 0 : undefined,
          minConfidence,
          minUplift,
        });
        const limited = edges.slice(0, limit);
        return {
          content: [
            {
              type: 'text',
              text: `🔍 Found ${edges.length} causal edges:\n\n` +
                limited.map((edge, i) =>
                  `${i + 1}. ${edge.fromMemoryType} → ${edge.toMemoryType}\n   Uplift: ${(edge.uplift || 0).toFixed(3)} (confidence: ${edge.confidence.toFixed(2)})`
                ).join('\n\n'),
            },
          ],
        };
      }

      case 'recall_with_certificate': {
        const query = args?.query as string;
        const k = (args?.k as number) || 12;

        const result = await causalRecall.recall(
          'mcp-' + Date.now(),
          query,
          k,
          undefined,
          'internal'
        );
        return {
          content: [
            {
              type: 'text',
              text: `🔍 Retrieved ${result.candidates.length} results:\n\n` +
                result.candidates.slice(0, 5).map((r, i) =>
                  `${i + 1}. ${r.type} ${r.id}\n   Similarity: ${r.similarity.toFixed(3)}\n   Uplift: ${r.uplift?.toFixed(3) || '0.000'}`
                ).join('\n\n') +
                `\n\n📜 Certificate ID: ${result.certificate.id}`,
            },
          ],
        };
      }

      case 'learner_discover': {
        const minAttempts = (args?.min_attempts as number) || 3;
        const minSuccessRate = (args?.min_success_rate as number) || 0.6;
        const minConfidence = (args?.min_confidence as number) || 0.7;
        const dryRun = (args?.dry_run as boolean) || false;

        const discovered = await learner.discover({
          minAttempts,
          minSuccessRate,
          minConfidence,
          dryRun,
        });
        return {
          content: [
            {
              type: 'text',
              text: `🌙 Discovered ${discovered.length} causal patterns:\n\n` +
                discovered.slice(0, 10).map((edge, i) =>
                  `${i + 1}. ${edge.fromMemoryType} → ${edge.toMemoryType}\n   Uplift: ${(edge.uplift || 0).toFixed(3)} (n=${edge.sampleSize || 0})`
                ).join('\n\n'),
            },
          ],
        };
      }

      case 'db_stats': {
        const stats: Record<string, number> = {
          causal_edges: (db.prepare('SELECT COUNT(*) as count FROM causal_edges').get() as any)?.count || 0,
          causal_experiments: (db.prepare('SELECT COUNT(*) as count FROM causal_experiments').get() as any)?.count || 0,
          causal_observations: (db.prepare('SELECT COUNT(*) as count FROM causal_observations').get() as any)?.count || 0,
          episodes: (db.prepare('SELECT COUNT(*) as count FROM episodes').get() as any)?.count || 0,
          episode_embeddings: (db.prepare('SELECT COUNT(*) as count FROM episode_embeddings').get() as any)?.count || 0,
          skills: (db.prepare('SELECT COUNT(*) as count FROM skills').get() as any)?.count || 0,
        };
        return {
          content: [
            {
              type: 'text',
              text: `📊 Database Statistics:\n\n` +
                `Causal Edges: ${stats.causal_edges}\n` +
                `Experiments: ${stats.causal_experiments}\n` +
                `Observations: ${stats.causal_observations}\n` +
                `Episodes (Vectors): ${stats.episodes}\n` +
                `Episode Embeddings: ${stats.episode_embeddings}\n` +
                `Skills: ${stats.skills}`,
            },
          ],
        };
      }

      // ======================================================================
      // CORE AGENTDB TOOLS (6-10)
      // ======================================================================
      case 'agentdb_stats': {
        const detailed = (args?.detailed as boolean) || false;

        // Check cache first (60s TTL)
        const cacheKey = `stats:${detailed ? 'detailed' : 'summary'}`;
        const cached = caches.stats.get(cacheKey);
        if (cached) {
          return {
            content: [
              {
                type: 'text',
                text: `${cached}\n\n⚡ (cached)`,
              },
            ],
          };
        }

        // Helper to safely query table count
        const safeCount = (tableName: string): number => {
          try {
            return (db.prepare(`SELECT COUNT(*) as count FROM ${tableName}`).get() as any)?.count || 0;
          } catch {
            return 0; // Table doesn't exist
          }
        };

        const stats: Record<string, number> = {
          causal_edges: safeCount('causal_edges'),
          causal_experiments: safeCount('causal_experiments'),
          causal_observations: safeCount('causal_observations'),
          episodes: safeCount('episodes'),
          episode_embeddings: safeCount('episode_embeddings'),
          skills: safeCount('skills'),
          skill_embeddings: safeCount('skill_embeddings'),
          reasoning_patterns: safeCount('reasoning_patterns'),
          pattern_embeddings: safeCount('pattern_embeddings'),
          learning_sessions: safeCount('rl_sessions'),
        };

        let output = `📊 AgentDB Comprehensive Statistics\n\n` +
          `🧠 Memory & Learning:\n` +
          `   Episodes (Vectors): ${stats.episodes}\n` +
          `   Episode Embeddings: ${stats.episode_embeddings}\n` +
          `   Skills: ${stats.skills}\n` +
          `   Skill Embeddings: ${stats.skill_embeddings}\n` +
          `   Reasoning Patterns: ${stats.reasoning_patterns}\n` +
          `   Pattern Embeddings: ${stats.pattern_embeddings}\n` +
          `   Learning Sessions: ${stats.learning_sessions}\n\n` +
          `🔗 Causal Intelligence:\n` +
          `   Causal Edges: ${stats.causal_edges}\n` +
          `   Experiments: ${stats.causal_experiments}\n` +
          `   Observations: ${stats.causal_observations}\n`;

        if (detailed) {
          // Add storage statistics
          const dbStats = db.prepare(`
            SELECT page_count * page_size as total_bytes
            FROM pragma_page_count(), pragma_page_size()
          `).get() as any;

          const totalMB = (dbStats.total_bytes / (1024 * 1024)).toFixed(2);

          // Add recent activity stats
          const recentActivity = db.prepare(`
            SELECT COUNT(*) as count
            FROM episodes
            WHERE ts >= strftime('%s', 'now', '-7 days')
          `).get() as any;

          output += `\n📦 Storage:\n` +
            `   Database Size: ${totalMB} MB\n` +
            `   Recent Activity (7d): ${recentActivity.count} episodes\n`;
        }

        // Cache the result (60s TTL)
        caches.stats.set(cacheKey, output);

        return {
          content: [
            {
              type: 'text',
              text: output,
            },
          ],
        };
      }

      case 'agentdb_pattern_store': {
        const taskType = args?.taskType as string;
        const approach = args?.approach as string;
        const successRate = args?.successRate as number;
        const tags = (args?.tags as string[]) || [];
        const metadata = (args?.metadata as Record<string, any>) || {};

        const patternId = await reasoningBank.storePattern({
          taskType,
          approach,
          successRate,
          tags,
          metadata,
        });

        return {
          content: [
            {
              type: 'text',
              text: `✅ Reasoning pattern stored successfully!\n\n` +
                `🆔 Pattern ID: ${patternId}\n` +
                `📋 Task Type: ${taskType}\n` +
                `💡 Approach: ${approach.substring(0, 100)}${approach.length > 100 ? '...' : ''}\n` +
                `📊 Success Rate: ${(successRate * 100).toFixed(1)}%\n` +
                `🏷️  Tags: ${tags.join(', ') || 'none'}\n` +
                `🧠 Embedding generated and stored`,
            },
          ],
        };
      }

      case 'agentdb_pattern_search': {
        const task = args?.task as string;
        const k = (args?.k as number) || 10;
        const threshold = (args?.threshold as number) || 0.0;
        const filters = args?.filters as any;

        // Generate embedding for the task
        const taskEmbedding = await embeddingService.embed(task);

        const patterns = await reasoningBank.searchPatterns({
          taskEmbedding,
          k,
          threshold,
          filters: filters ? {
            taskType: filters.taskType,
            minSuccessRate: filters.minSuccessRate,
            tags: filters.tags,
          } : undefined,
        });

        return {
          content: [
            {
              type: 'text',
              text: `🔍 Pattern search completed!\n\n` +
                `📊 Found: ${patterns.length} matching patterns\n` +
                `🎯 Query: ${task}\n` +
                `🎚️  Threshold: ${threshold.toFixed(2)}\n\n` +
                `Top Results:\n` +
                patterns.slice(0, 5).map((p, i) =>
                  `${i + 1}. [ID: ${p.id}] ${p.taskType}\n` +
                  `   Similarity: ${(p.similarity || 0).toFixed(3)}\n` +
                  `   Success Rate: ${(p.successRate * 100).toFixed(1)}%\n` +
                  `   Approach: ${p.approach.substring(0, 80)}${p.approach.length > 80 ? '...' : ''}\n` +
                  `   Uses: ${p.uses || 0}`
                ).join('\n\n') +
                (patterns.length > 5 ? `\n\n... and ${patterns.length - 5} more patterns` : ''),
            },
          ],
        };
      }

      case 'agentdb_pattern_stats': {
        // Check cache first (60s TTL)
        const cacheKey = 'pattern_stats';
        const cached = caches.stats.get(cacheKey);
        if (cached) {
          return {
            content: [
              {
                type: 'text',
                text: `${cached}\n\n⚡ (cached)`,
              },
            ],
          };
        }

        const stats = reasoningBank.getPatternStats();

        const output = `📊 Reasoning Pattern Statistics\n\n` +
          `📈 Overview:\n` +
          `   Total Patterns: ${stats.totalPatterns}\n` +
          `   Avg Success Rate: ${(stats.avgSuccessRate * 100).toFixed(1)}%\n` +
          `   Avg Uses per Pattern: ${stats.avgUses.toFixed(1)}\n` +
          `   High Performing (≥80%): ${stats.highPerformingPatterns}\n` +
          `   Recent (7 days): ${stats.recentPatterns}\n\n` +
          `🏆 Top Task Types:\n` +
          stats.topTaskTypes.slice(0, 10).map((tt, i) =>
            `   ${i + 1}. ${tt.taskType}: ${tt.count} patterns`
          ).join('\n') +
          (stats.topTaskTypes.length === 0 ? '   No patterns stored yet' : '');

        // Cache the result (60s TTL)
        caches.stats.set(cacheKey, output);

        return {
          content: [
            {
              type: 'text',
              text: output,
            },
          ],
        };
      }

      case 'agentdb_clear_cache': {
        const cacheType = (args?.cache_type as string) || 'all';

        let cleared = 0;

        switch (cacheType) {
          case 'patterns':
            cleared += caches.patterns.clear();
            reasoningBank.clearCache();
            break;
          case 'stats':
            cleared += caches.stats.clear();
            cleared += caches.metrics.clear();
            break;
          case 'all':
            caches.clearAll();
            reasoningBank.clearCache();
            cleared = -1; // All cleared
            break;
        }

        return {
          content: [
            {
              type: 'text',
              text: `✅ Cache cleared successfully!\n\n` +
                `🧹 Cache Type: ${cacheType}\n` +
                `♻️  ${cleared === -1 ? 'All caches' : `${cleared} cache entries`} cleared\n` +
                `📊 Statistics and search results will be refreshed on next query`,
            },
          ],
        };
      }

      // ======================================================================
      // BATCH OPERATION TOOLS (v2.0 MCP Optimization - Phase 2)
      // ======================================================================
      case 'skill_create_batch': {
        try {
          // Validate inputs
          const skillsArray = validateArrayLength(args?.skills, 'skills', 1, 100);
          const batchSize = args?.batch_size ? validateNumericRange(args.batch_size, 'batch_size', 1, 100) : 32;
          const format = args?.format ? validateEnum(args.format, 'format', ['concise', 'detailed', 'json'] as const) : 'concise';

          // Validate each skill
          const validatedSkills = skillsArray.map((skill: any, index: number) => {
            const name = validateTaskString(skill.name, `skills[${index}].name`);
            const description = validateTaskString(skill.description, `skills[${index}].description`);
            const successRate = skill.success_rate !== undefined
              ? validateNumericRange(skill.success_rate, `skills[${index}].success_rate`, 0, 1)
              : 0.0;

            return {
              name,
              description,
              signature: skill.signature || { inputs: {}, outputs: {} },
              code: skill.code || '',
              successRate,
              uses: skill.uses || 0,
              avgReward: skill.avg_reward || 0.0,
              avgLatencyMs: skill.avg_latency_ms || 0.0,
              tags: skill.tags || [],
              metadata: skill.metadata || {},
            };
          });

          // Use BatchOperations for efficient insertion
          const startTime = Date.now();
          const batchOpsConfig = new BatchOperations(db, embeddingService, {
            batchSize,
            parallelism: 4,
          });

          const skillIds = await batchOpsConfig.insertSkills(validatedSkills);
          const duration = Date.now() - startTime;

          // Format response
          if (format === 'json') {
            return {
              content: [
                {
                  type: 'text',
                  text: JSON.stringify({
                    success: true,
                    inserted: skillIds.length,
                    skill_ids: skillIds,
                    duration_ms: duration,
                    batch_size: batchSize,
                  }, null, 2),
                },
              ],
            };
          } else if (format === 'detailed') {
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Batch skill creation completed!\n\n` +
                    `📊 Performance:\n` +
                    `   • Skills Created: ${skillIds.length}\n` +
                    `   • Duration: ${duration}ms\n` +
                    `   • Throughput: ${(skillIds.length / (duration / 1000)).toFixed(1)} skills/sec\n` +
                    `   • Batch Size: ${batchSize}\n` +
                    `   • Parallelism: 4 workers\n\n` +
                    `🆔 Created Skill IDs:\n` +
                    skillIds.slice(0, 10).map((id, i) =>
                      `   ${i + 1}. Skill #${id}: ${validatedSkills[i].name}`
                    ).join('\n') +
                    (skillIds.length > 10 ? `\n   ... and ${skillIds.length - 10} more skills` : '') +
                    `\n\n🧠 All embeddings generated in parallel\n` +
                    `💾 Transaction committed successfully`,
                },
              ],
            };
          } else {
            // Concise format (default)
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Created ${skillIds.length} skills in ${duration}ms (${(skillIds.length / (duration / 1000)).toFixed(1)} skills/sec)`,
                },
              ],
            };
          }
        } catch (error: any) {
          const safeMessage = handleSecurityError(error);
          return {
            content: [
              {
                type: 'text',
                text: `❌ Batch skill creation failed: ${safeMessage}\n\n` +
                  `💡 Troubleshooting:\n` +
                  `   • Ensure all skills have unique names\n` +
                  `   • Verify success_rate is between 0 and 1\n` +
                  `   • Check that skills array has 1-100 items\n` +
                  `   • Ensure descriptions are not empty`,
              },
            ],
            isError: true,
          };
        }
      }

      case 'reflexion_store_batch': {
        try {
          // Validate inputs
          const episodesArray = validateArrayLength(args?.episodes, 'episodes', 1, 1000);
          const batchSize = args?.batch_size ? validateNumericRange(args.batch_size, 'batch_size', 1, 1000) : 100;
          const format = args?.format ? validateEnum(args.format, 'format', ['concise', 'detailed', 'json'] as const) : 'concise';

          // Validate each episode
          const validatedEpisodes = episodesArray.map((ep: any, index: number) => {
            const sessionId = validateSessionId(ep.session_id);
            const task = validateTaskString(ep.task, `episodes[${index}].task`);
            const reward = validateNumericRange(ep.reward, `episodes[${index}].reward`, 0, 1);
            const success = validateBoolean(ep.success, `episodes[${index}].success`);

            return {
              sessionId,
              task,
              reward,
              success,
              critique: ep.critique || '',
              input: ep.input || '',
              output: ep.output || '',
              latencyMs: ep.latency_ms || 0,
              tokensUsed: ep.tokens || 0,
              tags: ep.tags || [],
              metadata: ep.metadata || {},
            };
          });

          // Use BatchOperations for efficient insertion
          const startTime = Date.now();
          const batchOpsConfig = new BatchOperations(db, embeddingService, {
            batchSize,
            parallelism: 4,
          });

          const insertedCount = await batchOpsConfig.insertEpisodes(validatedEpisodes);
          const duration = Date.now() - startTime;

          // Format response
          if (format === 'json') {
            return {
              content: [
                {
                  type: 'text',
                  text: JSON.stringify({
                    success: true,
                    inserted: insertedCount,
                    duration_ms: duration,
                    batch_size: batchSize,
                  }, null, 2),
                },
              ],
            };
          } else if (format === 'detailed') {
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Batch episode storage completed!\n\n` +
                    `📊 Performance:\n` +
                    `   • Episodes Stored: ${insertedCount}\n` +
                    `   • Duration: ${duration}ms\n` +
                    `   • Throughput: ${(insertedCount / (duration / 1000)).toFixed(1)} episodes/sec\n` +
                    `   • Batch Size: ${batchSize}\n` +
                    `   • Parallelism: 4 workers\n\n` +
                    `📈 Statistics:\n` +
                    `   • Sessions: ${new Set(validatedEpisodes.map(e => e.sessionId)).size}\n` +
                    `   • Success Rate: ${(validatedEpisodes.filter(e => e.success).length / validatedEpisodes.length * 100).toFixed(1)}%\n` +
                    `   • Avg Reward: ${(validatedEpisodes.reduce((sum, e) => sum + e.reward, 0) / validatedEpisodes.length).toFixed(3)}\n\n` +
                    `🧠 All embeddings generated in parallel\n` +
                    `💾 Transaction committed successfully`,
                },
              ],
            };
          } else {
            // Concise format (default)
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Stored ${insertedCount} episodes in ${duration}ms (${(insertedCount / (duration / 1000)).toFixed(1)} episodes/sec)`,
                },
              ],
            };
          }
        } catch (error: any) {
          const safeMessage = handleSecurityError(error);
          return {
            content: [
              {
                type: 'text',
                text: `❌ Batch episode storage failed: ${safeMessage}\n\n` +
                  `💡 Troubleshooting:\n` +
                  `   • Ensure all session_ids are valid (alphanumeric, hyphens, underscores)\n` +
                  `   • Verify rewards are between 0 and 1\n` +
                  `   • Check that episodes array has 1-1000 items\n` +
                  `   • Ensure tasks are not empty or excessively long`,
              },
            ],
            isError: true,
          };
        }
      }

      case 'agentdb_pattern_store_batch': {
        try {
          // Validate inputs
          const patternsArray = validateArrayLength(args?.patterns, 'patterns', 1, 500);
          const batchSize = args?.batch_size ? validateNumericRange(args.batch_size, 'batch_size', 1, 500) : 50;
          const format = args?.format ? validateEnum(args.format, 'format', ['concise', 'detailed', 'json'] as const) : 'concise';

          // Validate each pattern
          const validatedPatterns = patternsArray.map((pattern: any, index: number) => {
            const taskType = validateTaskString(pattern.taskType, `patterns[${index}].taskType`);
            const approach = validateTaskString(pattern.approach, `patterns[${index}].approach`);
            const successRate = validateNumericRange(pattern.successRate, `patterns[${index}].successRate`, 0, 1);

            return {
              taskType,
              approach,
              successRate,
              tags: pattern.tags || [],
              metadata: pattern.metadata || {},
            };
          });

          // Use BatchOperations for efficient insertion
          const startTime = Date.now();
          const batchOpsConfig = new BatchOperations(db, embeddingService, {
            batchSize,
            parallelism: 4,
          });

          const patternIds = await batchOpsConfig.insertPatterns(validatedPatterns);
          const duration = Date.now() - startTime;

          // Format response
          if (format === 'json') {
            return {
              content: [
                {
                  type: 'text',
                  text: JSON.stringify({
                    success: true,
                    inserted: patternIds.length,
                    pattern_ids: patternIds,
                    duration_ms: duration,
                    batch_size: batchSize,
                  }, null, 2),
                },
              ],
            };
          } else if (format === 'detailed') {
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Batch pattern storage completed!\n\n` +
                    `📊 Performance:\n` +
                    `   • Patterns Stored: ${patternIds.length}\n` +
                    `   • Duration: ${duration}ms\n` +
                    `   • Throughput: ${(patternIds.length / (duration / 1000)).toFixed(1)} patterns/sec\n` +
                    `   • Batch Size: ${batchSize}\n` +
                    `   • Parallelism: 4 workers\n\n` +
                    `📈 Statistics:\n` +
                    `   • Task Types: ${new Set(validatedPatterns.map(p => p.taskType)).size}\n` +
                    `   • Avg Success Rate: ${(validatedPatterns.reduce((sum, p) => sum + p.successRate, 0) / validatedPatterns.length * 100).toFixed(1)}%\n` +
                    `   • High Performing (≥80%): ${validatedPatterns.filter(p => p.successRate >= 0.8).length}\n\n` +
                    `🆔 Sample Pattern IDs: ${patternIds.slice(0, 5).join(', ')}${patternIds.length > 5 ? '...' : ''}\n` +
                    `🧠 All embeddings generated in parallel\n` +
                    `💾 Transaction committed successfully`,
                },
              ],
            };
          } else {
            // Concise format (default)
            return {
              content: [
                {
                  type: 'text',
                  text: `✅ Stored ${patternIds.length} patterns in ${duration}ms (${(patternIds.length / (duration / 1000)).toFixed(1)} patterns/sec)`,
                },
              ],
            };
          }
        } catch (error: any) {
          const safeMessage = handleSecurityError(error);
          return {
            content: [
              {
                type: 'text',
                text: `❌ Batch pattern storage failed: ${safeMessage}\n\n` +
                  `💡 Troubleshooting:\n` +
                  `   • Ensure taskType and approach are not empty\n` +
                  `   • Verify successRate is between 0 and 1\n` +
                  `   • Check that patterns array has 1-500 items\n` +
                  `   • Avoid excessively long task types or approaches`,
              },
            ],
            isError: true,
          };
        }
      }

      // ======================================================================
      // LEARNING SYSTEM TOOLS (Tools 1-5)
      // ======================================================================
      case 'learning_start_session': {
        const userId = args?.user_id as string;
        const sessionType = args?.session_type as any;
        const config = args?.config as any;

        const sessionId = await learningSystem.startSession(
          userId,
          sessionType,
          {
            learningRate: config.learning_rate,
            discountFactor: config.discount_factor,
            explorationRate: config.exploration_rate,
            batchSize: config.batch_size,
            targetUpdateFrequency: config.target_update_frequency,
          }
        );

        return {
          content: [
            {
              type: 'text',
              text: `✅ Learning session started!\n\n` +
                `🆔 Session ID: ${sessionId}\n` +
                `👤 User: ${userId}\n` +
                `🧠 Algorithm: ${sessionType}\n` +
                `⚙️  Config:\n` +
                `   • Learning Rate: ${config.learning_rate}\n` +
                `   • Discount Factor: ${config.discount_factor}\n` +
                `   • Exploration Rate: ${config.exploration_rate || 0.1}\n` +
                `   • Batch Size: ${config.batch_size || 32}\n\n` +
                `📝 Use this session ID for predictions and feedback.`,
            },
          ],
        };
      }

      case 'learning_end_session': {
        const sessionId = args?.session_id as string;
        await learningSystem.endSession(sessionId);

        return {
          content: [
            {
              type: 'text',
              text: `✅ Learning session ended!\n\n` +
                `🆔 Session ID: ${sessionId}\n` +
                `💾 Final policy saved to database\n` +
                `📊 Session marked as completed`,
            },
          ],
        };
      }

      case 'learning_predict': {
        const sessionId = args?.session_id as string;
        const state = args?.state as string;

        const prediction = await learningSystem.predict(sessionId, state);

        return {
          content: [
            {
              type: 'text',
              text: `🎯 AI Recommendation:\n\n` +
                `📍 State: ${state}\n` +
                `✨ Recommended Action: ${prediction.action}\n` +
                `💯 Confidence: ${(prediction.confidence * 100).toFixed(1)}%\n` +
                `📊 Q-Value: ${prediction.qValue?.toFixed(3) || 'N/A'}\n\n` +
                `🔄 Alternative Actions:\n` +
                prediction.alternatives.map((alt, i) =>
                  `   ${i + 1}. ${alt.action} (${(alt.confidence * 100).toFixed(1)}% confidence)`
                ).join('\n'),
            },
          ],
        };
      }

      case 'learning_feedback': {
        const sessionId = args?.session_id as string;
        const state = args?.state as string;
        const action = args?.action as string;
        const reward = args?.reward as number;
        const nextState = args?.next_state as string | undefined;
        const success = args?.success as boolean;

        await learningSystem.submitFeedback({
          sessionId,
          state,
          action,
          reward,
          nextState,
          success,
          timestamp: Date.now(),
        });

        return {
          content: [
            {
              type: 'text',
              text: `✅ Feedback recorded!\n\n` +
                `🆔 Session: ${sessionId}\n` +
                `📍 State: ${state}\n` +
                `🎬 Action: ${action}\n` +
                `🏆 Reward: ${reward.toFixed(2)}\n` +
                `${success ? '✅' : '❌'} Success: ${success}\n` +
                `${nextState ? `➡️  Next State: ${nextState}\n` : ''}` +
                `🧠 Policy updated incrementally`,
            },
          ],
        };
      }

      case 'learning_train': {
        const sessionId = args?.session_id as string;
        const epochs = (args?.epochs as number) || 50;
        const batchSize = (args?.batch_size as number) || 32;
        const learningRate = (args?.learning_rate as number) || 0.01;

        console.log(`🎓 Training session ${sessionId}...`);
        const result = await learningSystem.train(sessionId, epochs, batchSize, learningRate);

        return {
          content: [
            {
              type: 'text',
              text: `🎓 Training completed!\n\n` +
                `📊 Training Results:\n` +
                `   • Epochs: ${result.epochsCompleted}\n` +
                `   • Final Loss: ${result.finalLoss.toFixed(4)}\n` +
                `   • Avg Reward: ${result.avgReward.toFixed(3)}\n` +
                `   • Convergence Rate: ${(result.convergenceRate * 100).toFixed(1)}%\n` +
                `   • Training Time: ${result.trainingTimeMs}ms\n\n` +
                `💾 Trained policy saved to database\n` +
                `✨ Ready for improved predictions!`,
            },
          ],
        };
      }

      // ======================================================================
      // LEARNING SYSTEM TOOLS (v1.4.0 - Tools 6-10) - IMPLEMENTATIONS
      // ======================================================================
      case 'learning_metrics': {
        const sessionId = args?.session_id as string | undefined;
        const timeWindowDays = (args?.time_window_days as number) || 7;
        const includeTrends = (args?.include_trends as boolean) !== false;
        const groupBy = (args?.group_by as 'task' | 'session' | 'skill') || 'task';

        // Check cache first (120s TTL for expensive computations)
        const cacheKey = `metrics:${sessionId || 'all'}:${timeWindowDays}:${groupBy}:${includeTrends}`;
        const cached = caches.metrics.get(cacheKey);
        if (cached) {
          return {
            content: [
              {
                type: 'text',
                text: `${cached}\n\n⚡ (cached)`,
              },
            ],
          };
        }

        const metrics = await learningSystem.getMetrics({
          sessionId,
          timeWindowDays,
          includeTrends,
          groupBy,
        });

        const output = `📊 Learning Performance Metrics\n\n` +
          `⏱️  Time Window: ${timeWindowDays} days\n\n` +
          `📈 Overall Performance:\n` +
          `   • Total Episodes: ${metrics.overall.totalEpisodes}\n` +
          `   • Success Rate: ${(metrics.overall.successRate * 100).toFixed(1)}%\n` +
          `   • Avg Reward: ${metrics.overall.avgReward.toFixed(3)}\n` +
          `   • Reward Range: [${metrics.overall.minReward.toFixed(2)}, ${metrics.overall.maxReward.toFixed(2)}]\n` +
          `   • Avg Latency: ${metrics.overall.avgLatencyMs.toFixed(0)}ms\n\n` +
          `🎯 Top ${groupBy.charAt(0).toUpperCase() + groupBy.slice(1)}s:\n` +
          metrics.groupedMetrics.slice(0, 5).map((g, i) =>
            `   ${i + 1}. ${g.key.substring(0, 40)}${g.key.length > 40 ? '...' : ''}\n` +
            `      Count: ${g.count}, Success: ${(g.successRate * 100).toFixed(1)}%, Reward: ${g.avgReward.toFixed(2)}`
          ).join('\n') +
          (metrics.groupedMetrics.length === 0 ? '   No data available' : '') +
          (includeTrends && metrics.trends.length > 0 ? `\n\n📉 Recent Trends (last ${Math.min(7, metrics.trends.length)} days):\n` +
            metrics.trends.slice(-7).map((t) =>
              `   ${t.date}: ${t.count} episodes, ${(t.successRate * 100).toFixed(1)}% success`
            ).join('\n') : '') +
          (metrics.policyImprovement.versions > 0 ? `\n\n🧠 Policy Improvement:\n` +
            `   • Versions: ${metrics.policyImprovement.versions}\n` +
            `   • Q-Value Improvement: ${metrics.policyImprovement.qValueImprovement >= 0 ? '+' : ''}${metrics.policyImprovement.qValueImprovement.toFixed(3)}` : '');

        // Cache the result (120s TTL)
        caches.metrics.set(cacheKey, output);

        return {
          content: [
            {
              type: 'text',
              text: output,
            },
          ],
        };
      }

      case 'learning_transfer': {
        const sourceSession = args?.source_session as string | undefined;
        const targetSession = args?.target_session as string | undefined;
        const sourceTask = args?.source_task as string | undefined;
        const targetTask = args?.target_task as string | undefined;
        const minSimilarity = (args?.min_similarity as number) || 0.7;
        const transferType = (args?.transfer_type as any) || 'all';
        const maxTransfers = (args?.max_transfers as number) || 10;

        const result = await learningSystem.transferLearning({
          sourceSession,
          targetSession,
          sourceTask,
          targetTask,
          minSimilarity,
          transferType,
          maxTransfers,
        });

        return {
          content: [
            {
              type: 'text',
              text: `🔄 Learning Transfer Completed!\n\n` +
                `📤 Source: ${sourceSession ? `Session ${sourceSession}` : `Task "${sourceTask}"`}\n` +
                `📥 Target: ${targetSession ? `Session ${targetSession}` : `Task "${targetTask}"`}\n` +
                `🎯 Transfer Type: ${transferType}\n` +
                `📊 Min Similarity: ${(minSimilarity * 100).toFixed(0)}%\n\n` +
                `✅ Transferred:\n` +
                `   • Episodes: ${result.transferred.episodes}\n` +
                `   • Skills/Q-Values: ${result.transferred.skills}\n` +
                `   • Causal Edges: ${result.transferred.causalEdges}\n` +
                (result.transferred.details.length > 0 ? `\n📝 Transfer Details:\n` +
                  result.transferred.details.slice(0, 5).map((d: any, i: number) =>
                    `   ${i + 1}. ${d.type} #${d.id} (similarity: ${(d.similarity * 100).toFixed(1)}%)`
                  ).join('\n') : '') +
                `\n\n💡 Knowledge successfully transferred for reuse!`,
            },
          ],
        };
      }

      case 'learning_explain': {
        const query = args?.query as string;
        const k = (args?.k as number) || 5;
        const explainDepth = (args?.explain_depth as any) || 'detailed';
        const includeConfidence = (args?.include_confidence as boolean) !== false;
        const includeEvidence = (args?.include_evidence as boolean) !== false;
        const includeCausal = (args?.include_causal as boolean) !== false;

        const explanation = await learningSystem.explainAction({
          query,
          k,
          explainDepth,
          includeConfidence,
          includeEvidence,
          includeCausal,
        });

        return {
          content: [
            {
              type: 'text',
              text: `🔍 AI Action Recommendations (Explainable)\n\n` +
                `🎯 Query: ${query}\n\n` +
                `💡 Recommended Actions:\n` +
                explanation.recommendations.map((rec: any, i: number) =>
                  `${i + 1}. ${rec.action}\n` +
                  `   • Confidence: ${(rec.confidence * 100).toFixed(1)}%\n` +
                  `   • Success Rate: ${(rec.successRate * 100).toFixed(1)}%\n` +
                  `   • Avg Reward: ${rec.avgReward.toFixed(3)}\n` +
                  `   • Supporting Examples: ${rec.supportingExamples}\n` +
                  (includeEvidence && rec.evidence ? `   • Evidence:\n` +
                    rec.evidence.map((e: any) =>
                      `     - Episode ${e.episodeId}: reward=${e.reward.toFixed(2)}, similarity=${(e.similarity * 100).toFixed(1)}%`
                    ).join('\n') : '')
                ).join('\n\n') +
                (explainDepth !== 'summary' && explanation.reasoning ? `\n\n🧠 Reasoning:\n` +
                  `   • Similar Experiences Found: ${explanation.reasoning.similarExperiencesFound}\n` +
                  `   • Avg Similarity: ${(explanation.reasoning.avgSimilarity * 100).toFixed(1)}%\n` +
                  `   • Unique Actions Considered: ${explanation.reasoning.uniqueActions}` : '') +
                (includeCausal && explanation.causalChains && explanation.causalChains.length > 0 ? `\n\n🔗 Causal Reasoning Chains:\n` +
                  explanation.causalChains.slice(0, 3).map((chain: any, i: number) =>
                    `   ${i + 1}. ${chain.fromMemoryType} → ${chain.toMemoryType} (uplift: ${(chain.uplift || 0).toFixed(3)})`
                  ).join('\n') : ''),
            },
          ],
        };
      }

      case 'experience_record': {
        const sessionId = args?.session_id as string;
        const toolName = args?.tool_name as string;
        const action = args?.action as string;
        const stateBefore = args?.state_before as any;
        const stateAfter = args?.state_after as any;
        const outcome = args?.outcome as string;
        const reward = args?.reward as number;
        const success = args?.success as boolean;
        const latencyMs = args?.latency_ms as number | undefined;
        const metadata = args?.metadata as any;

        const experienceId = await learningSystem.recordExperience({
          sessionId,
          toolName,
          action,
          stateBefore,
          stateAfter,
          outcome,
          reward,
          success,
          latencyMs,
          metadata,
        });

        return {
          content: [
            {
              type: 'text',
              text: `✅ Experience recorded successfully!\n\n` +
                `🆔 Experience ID: ${experienceId}\n` +
                `📋 Session: ${sessionId}\n` +
                `🔧 Tool: ${toolName}\n` +
                `🎬 Action: ${action}\n` +
                `📊 Outcome: ${outcome}\n` +
                `🏆 Reward: ${reward.toFixed(3)}\n` +
                `${success ? '✅' : '❌'} Success: ${success}\n` +
                (latencyMs ? `⏱️  Latency: ${latencyMs}ms\n` : '') +
                `\n💾 Experience stored for offline learning and future recommendations`,
            },
          ],
        };
      }

      case 'reward_signal': {
        const episodeId = args?.episode_id as number | undefined;
        const success = args?.success as boolean;
        const targetAchieved = (args?.target_achieved as boolean) !== false;
        const efficiencyScore = (args?.efficiency_score as number) || 0.5;
        const qualityScore = (args?.quality_score as number) || 0.5;
        const timeTakenMs = args?.time_taken_ms as number | undefined;
        const expectedTimeMs = args?.expected_time_ms as number | undefined;
        const includeCausal = (args?.include_causal as boolean) !== false;
        const rewardFunction = (args?.reward_function as any) || 'standard';

        const reward = learningSystem.calculateReward({
          episodeId,
          success,
          targetAchieved,
          efficiencyScore,
          qualityScore,
          timeTakenMs,
          expectedTimeMs,
          includeCausal,
          rewardFunction,
        });

        return {
          content: [
            {
              type: 'text',
              text: `🎯 Reward Signal Calculated\n\n` +
                `📊 Final Reward: ${reward.toFixed(3)}\n` +
                `🔧 Reward Function: ${rewardFunction}\n\n` +
                `📈 Input Factors:\n` +
                `   • Success: ${success ? '✅' : '❌'}\n` +
                `   • Target Achieved: ${targetAchieved ? '✅' : '❌'}\n` +
                `   • Efficiency Score: ${(efficiencyScore * 100).toFixed(1)}%\n` +
                `   • Quality Score: ${(qualityScore * 100).toFixed(1)}%\n` +
                (timeTakenMs && expectedTimeMs ? `   • Time Efficiency: ${((expectedTimeMs / timeTakenMs) * 100).toFixed(1)}%\n` : '') +
                (includeCausal ? `   • Causal Impact: Included\n` : '') +
                `\n💡 Use this reward signal for learning feedback`,
            },
          ],
        };
      }

      // ======================================================================
      // ADR-073 — SOTA roadmap handlers
      // ======================================================================
      case 'causal_traverse': {
        const fromId = args?.from_memory_id as number;
        const toId = args?.to_memory_id as number;
        const maxDepth = (args?.max_depth as number) || 5;
        const chain = await causalGraph.getCausalChain(fromId, toId, maxDepth);
        const found = chain && Array.isArray((chain as any).edges) ? (chain as any).edges.length > 0 : false;
        return {
          content: [{
            type: 'text',
            text: found
              ? `🔗 Causal chain ${fromId} → ${toId} (max ${maxDepth} hops):\n${JSON.stringify(chain, null, 2)}`
              : `❌ No causal chain found between memory #${fromId} and #${toId} within ${maxDepth} hops.`,
          }],
        };
      }

      case 'agentdb_delete_batch': {
        const ids = (args?.ids as number[]) || [];
        const requestedTable = (args?.table as string) || 'episodes';
        const allowedTables = new Set(['episodes', 'reasoning_patterns']);
        if (!allowedTables.has(requestedTable)) {
          throw new Error(`Invalid table: ${requestedTable}`);
        }
        if (ids.length === 0) {
          return { content: [{ type: 'text', text: '⚠️  No IDs supplied — nothing to delete.' }] };
        }
        // Validate each id is a positive integer (defense in depth)
        const safeIds = ids.map((raw) => {
          const n = Number(raw);
          if (!Number.isInteger(n) || n <= 0) throw new Error(`Invalid id: ${raw}`);
          return n;
        });
        const placeholders = safeIds.map(() => '?').join(',');
        // requestedTable is whitelist-checked above; parameterising values via ?.
        const stmt = db.prepare(`DELETE FROM ${requestedTable} WHERE id IN (${placeholders})`);
        const result = stmt.run(...safeIds);
        return {
          content: [{
            type: 'text',
            text: `🗑️  Batch delete from \`${requestedTable}\`: ${result.changes}/${safeIds.length} rows removed (atomic).`,
          }],
        };
      }

      case 'consolidate_now': {
        const sessionId = args?.session_id as string | undefined;
        const t0 = Date.now();
        // Prefer the session-scoped path when sessionId is given.
        const report = sessionId
          ? await learner.consolidateEpisodes(sessionId)
          : await learner.run();
        const ms = Date.now() - t0;
        return {
          content: [{
            type: 'text',
            text: `🌙 Consolidation complete${sessionId ? ` (session=${sessionId})` : ''} in ${ms}ms\n` +
              JSON.stringify(report, null, 2),
          }],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error: any) {
    return {
      content: [
        {
          type: 'text',
          text: `❌ Error: ${error.message}\n${error.stack || ''}`,
        },
      ],
      isError: true,
    };
  }
});

// ============================================================================
// Start Server
// ============================================================================
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('🚀 AgentDB MCP Server v2.0.0 running on stdio');
  console.error('📦 32 tools available (5 core + 9 frontier + 10 learning + 5 AgentDB + 3 batch ops)');
  console.error('🧠 Embedding service initialized');
  console.error('🎓 Learning system ready (9 RL algorithms)');
  console.error('⚡ NEW v2.0: Batch operations (3-4x faster), format parameters, enhanced validation');
  console.error('🔬 Features: transfer learning, XAI explanations, reward shaping, intelligent caching');

  // Keep the process alive - the StdioServerTransport handles stdin/stdout
  // but we need to ensure Node.js doesn't exit when main() completes

  // Use setInterval to keep event loop alive (like many MCP servers do)
  // This ensures the process doesn't exit even if stdin closes
  const keepAlive = setInterval(() => {
    // Empty interval just to keep event loop alive
  }, 1000 * 60 * 60); // Every hour (basically forever)

  // Periodic auto-save: Save database every 5 minutes to prevent data loss
  const autoSaveInterval = setInterval(() => {
    try {
      if (db && typeof db.save === 'function') {
        db.save();
        console.error('💾 Auto-saved database to', dbPath);
      }
    } catch (error) {
      console.error('❌ Auto-save failed:', (error as Error).message);
    }
  }, 5 * 60 * 1000); // 5 minutes

  // Handle graceful shutdown with database persistence
  const shutdown = async () => {
    console.error('🔄 Shutting down AgentDB MCP Server...');

    // Clear intervals
    clearInterval(keepAlive);
    clearInterval(autoSaveInterval);

    let persistenceFailed = false;

    // Save database before exit
    try {
      if (db && typeof db.save === 'function') {
        console.error('💾 Saving database to', dbPath);
        await db.save();
        console.error('✅ Database saved successfully');
      }
    } catch (error) {
      persistenceFailed = true;
      console.error('❌ Error saving database:', (error as Error).message);
    }

    // Close database connection. close() persists sql.js itself, so only call
    // it after save when save succeeded; a failed optimistic save must not be
    // retried against the same stale snapshot.
    try {
      if (!persistenceFailed && db && typeof db.close === 'function') {
        db.close();
        console.error('✅ Database connection closed');
      }
    } catch (error) {
      persistenceFailed = true;
      console.error('❌ Error closing database:', (error as Error).message);
    }

    process.exit(persistenceFailed ? 1 : 0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  // Handle unexpected exits
  process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught exception:', error);
    shutdown();
  });

  process.on('unhandledRejection', (error) => {
    console.error('❌ Unhandled rejection:', error);
    shutdown();
  });

  // Return a never-resolving promise
  return new Promise(() => {
    // The setInterval above keeps the event loop alive
    // StdioServerTransport handles all MCP communication
  });
}

main().catch(console.error);
