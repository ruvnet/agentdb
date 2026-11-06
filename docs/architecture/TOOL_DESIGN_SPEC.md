# AgentDB MCP Tools Design Specification

**Version:** 2.0.0
**Date:** October 22, 2025
**Status:** Design Phase - Implementation Ready
**Target Release:** AgentDB 1.3.0

---

## Executive Summary

This document provides comprehensive design specifications for **11 missing MCP tools** identified in the verification report. These tools will complete the AgentDB MCP interface, providing full programmatic access to:

- **Core Vector Database Operations** (10 tools)
- **Enhanced Frontier Features** (1 tool)

**Current State:** 9 tools implemented (Frontier features only)
**Target State:** 20 tools implemented (Full coverage)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Vector DB Tools (10 Tools)](#core-vector-db-tools)
3. [Enhanced Frontier Tools (1 Tool)](#enhanced-frontier-tools)
4. [Integration Points](#integration-points)
5. [Error Handling Strategy](#error-handling-strategy)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### Design Principles

1. **Consistency:** All tools follow the same signature pattern as existing 9 tools
2. **Type Safety:** Leverage TypeScript interfaces from controllers
3. **Performance:** Batch operations where possible, use prepared statements
4. **Error Transparency:** Detailed error messages with recovery suggestions
5. **Backward Compatibility:** New tools don't break existing functionality

### Technology Stack

- **Runtime:** Node.js 20.9.0+ with ESM
- **Database:** SQLite 3.35+ via better-sqlite3
- **Embeddings:** Xenova/all-MiniLM-L6-v2 (384-dim) via @xenova/transformers
- **MCP Protocol:** @modelcontextprotocol/sdk v1.0+

### Integration with Existing Tools

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentDB MCP Server                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Existing 9 Tools│  │  New 11 Tools    │                │
│  │  (Frontier)      │  │  (Core + More)   │                │
│  └──────────────────┘  └──────────────────┘                │
│           │                      │                           │
│           └──────────┬───────────┘                           │
│                      │                                       │
│  ┌─────────────────────────────────────────────┐            │
│  │         Shared Controllers Layer             │            │
│  ├─────────────────────────────────────────────┤            │
│  │  • EmbeddingService                          │            │
│  │  • ReflexionMemory                           │            │
│  │  • SkillLibrary                              │            │
│  │  • CausalMemoryGraph                         │            │
│  │  • CausalRecall                              │            │
│  │  • ExplainableRecall                         │            │
│  │  • NightlyLearner                            │            │
│  └─────────────────────────────────────────────┘            │
│                      │                                       │
│  ┌─────────────────────────────────────────────┐            │
│  │         SQLite Database Layer                │            │
│  │  • schema.sql (core tables)                  │            │
│  │  • frontier-schema.sql (causal + certs)      │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Vector DB Tools

### 1. agentdb_init

**Purpose:** Initialize AgentDB with custom configuration

**Status:** 🔴 Not Implemented
**Priority:** High (P0)
**Complexity:** Low (2-3 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_init',
  description: 'Initialize AgentDB with custom configuration and schema',
  inputSchema: {
    type: 'object',
    properties: {
      db_path: {
        type: 'string',
        description: 'Path to SQLite database file',
        default: './agentdb.db'
      },
      embedding_model: {
        type: 'string',
        description: 'Embedding model to use',
        default: 'Xenova/all-MiniLM-L6-v2',
        enum: ['Xenova/all-MiniLM-L6-v2', 'Xenova/all-mpnet-base-v2', 'Xenova/bge-small-en-v1.5']
      },
      cache_dir: {
        type: 'string',
        description: 'Embedding cache directory',
        default: './.agentdb/embeddings'
      },
      enable_frontier: {
        type: 'boolean',
        description: 'Enable frontier features (causal graph + certificates)',
        default: true
      },
      vector_dimensions: {
        type: 'number',
        description: 'Vector embedding dimensions',
        default: 384
      },
      pragmas: {
        type: 'object',
        description: 'SQLite PRAGMA settings',
        properties: {
          journal_mode: { type: 'string', default: 'WAL' },
          synchronous: { type: 'string', default: 'NORMAL' },
          cache_size: { type: 'number', default: -2000 },
          foreign_keys: { type: 'boolean', default: true }
        }
      }
    }
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `✅ AgentDB initialized successfully

📊 Configuration:
  Database: ${db_path}
  Model: ${embedding_model}
  Dimensions: ${vector_dimensions}
  Frontier Features: ${enable_frontier ? 'Enabled' : 'Disabled'}

📁 Schema Status:
  Core Tables: ${core_count} tables created
  Frontier Tables: ${frontier_count} tables created
  Indexes: ${index_count} indexes created

⚡ Performance Settings:
  Journal Mode: ${journal_mode}
  Cache Size: ${cache_size} pages

🔗 Connection: Ready`
  }]
}
```

#### Implementation Notes

1. Check if database already exists - warn if overwriting
2. Apply schema.sql and frontier-schema.sql
3. Set SQLite PRAGMAs for optimal performance
4. Initialize EmbeddingService with specified model
5. Verify schema integrity with SELECT COUNT(*) queries
6. Create cache directory if needed
7. Return detailed initialization report

#### Error Scenarios

- **Database locked:** Return error with suggestion to close other connections
- **Invalid model name:** List available models
- **Schema creation failed:** Return SQL error with line number
- **Insufficient permissions:** Suggest chmod/ownership fixes

---

### 2. agentdb_insert

**Purpose:** Insert a single vector with metadata into AgentDB

**Status:** 🔴 Not Implemented
**Priority:** High (P0)
**Complexity:** Medium (3-4 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_insert',
  description: 'Insert a single vector with metadata into AgentDB',
  inputSchema: {
    type: 'object',
    properties: {
      collection: {
        type: 'string',
        description: 'Collection name',
        enum: ['episodes', 'skills', 'notes', 'facts', 'events']
      },
      text: {
        type: 'string',
        description: 'Text content to embed and store'
      },
      metadata: {
        type: 'object',
        description: 'Additional metadata (JSON)',
        properties: {
          // Collection-specific fields
          task: { type: 'string' },
          reward: { type: 'number' },
          success: { type: 'boolean' },
          tags: { type: 'array', items: { type: 'string' } }
        }
      },
      embedding: {
        type: 'array',
        description: 'Pre-computed embedding (optional, will auto-generate if missing)',
        items: { type: 'number' }
      },
      skip_embedding: {
        type: 'boolean',
        description: 'Skip embedding generation for text-only storage',
        default: false
      }
    },
    required: ['collection', 'text']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `✅ Inserted into ${collection}

Record ID: ${record_id}
Embedding ID: ${embedding_id}
Vector Dimension: ${dimension}
Text Length: ${text.length} chars
Metadata Fields: ${Object.keys(metadata).length}

Processing Time: ${latency_ms}ms
  - Text Processing: ${text_processing_ms}ms
  - Embedding Generation: ${embedding_ms}ms
  - Database Write: ${db_write_ms}ms`
  }]
}
```

#### Implementation Notes

1. Validate collection name against schema
2. Generate embedding if not provided using EmbeddingService
3. Insert into appropriate table (episodes, skills, notes, facts)
4. Store embedding in corresponding _embeddings table
5. Handle collection-specific required fields
6. Return detailed insertion report with timing breakdown

#### Collection-Specific Handling

**Episodes:**
- Required: session_id, task, reward, success
- Auto-generate: ts, created_at
- Embedding: task + output + critique

**Skills:**
- Required: name, description, signature
- Auto-generate: uses=0, success_rate=0, created_at
- Embedding: name + description + signature

**Notes:**
- Required: text
- Optional: title, note_type, importance
- Embedding: summary or text

**Facts:**
- Required: subject, predicate, object
- Optional: source_type, confidence
- No embedding (use graph queries)

**Events:**
- Required: session_id, step, content
- Optional: phase, role, tool_calls
- No embedding (temporal queries)

---

### 3. agentdb_insert_batch

**Purpose:** Batch insert multiple vectors for high-performance bulk loading

**Status:** 🔴 Not Implemented
**Priority:** High (P0)
**Complexity:** Medium (4-5 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_insert_batch',
  description: 'Batch insert multiple vectors with metadata for high-performance bulk loading',
  inputSchema: {
    type: 'object',
    properties: {
      collection: {
        type: 'string',
        description: 'Collection name',
        enum: ['episodes', 'skills', 'notes', 'facts', 'events']
      },
      records: {
        type: 'array',
        description: 'Array of records to insert',
        items: {
          type: 'object',
          properties: {
            text: { type: 'string' },
            metadata: { type: 'object' },
            embedding: { type: 'array', items: { type: 'number' } }
          },
          required: ['text']
        },
        minItems: 1,
        maxItems: 1000
      },
      batch_size: {
        type: 'number',
        description: 'Records per transaction (for memory control)',
        default: 100,
        minimum: 10,
        maximum: 1000
      },
      on_error: {
        type: 'string',
        description: 'Error handling strategy',
        enum: ['abort', 'skip', 'collect'],
        default: 'abort'
      },
      parallel_embedding: {
        type: 'boolean',
        description: 'Generate embeddings in parallel (faster but more memory)',
        default: false
      }
    },
    required: ['collection', 'records']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `✅ Batch insert completed

📊 Summary:
  Collection: ${collection}
  Total Records: ${records.length}
  Successfully Inserted: ${success_count}
  Failed: ${error_count}
  Skipped: ${skip_count}

⚡ Performance:
  Total Time: ${total_ms}ms
  Avg Per Record: ${avg_ms}ms
  Throughput: ${records_per_sec} records/sec

💾 Storage:
  Records Written: ${success_count}
  Embeddings Generated: ${embeddings_generated}
  Total Vectors: ${total_vectors}
  Database Size Change: +${size_delta_mb}MB

${error_count > 0 ? `
❌ Errors (${error_count}):
${errors.slice(0, 5).map((e, i) => `  ${i + 1}. Record ${e.index}: ${e.message}`).join('\n')}
${errors.length > 5 ? `  ... and ${errors.length - 5} more errors` : ''}
` : ''}`
  }]
}
```

#### Implementation Notes

1. **Transaction Management:**
   - Use BEGIN TRANSACTION for each batch
   - Commit after batch_size records
   - Rollback on error if on_error='abort'

2. **Embedding Generation:**
   - Sequential: Generate embeddings one-by-one (lower memory)
   - Parallel: Use Promise.all() for batch embedding (faster)
   - Cache embeddings in memory during batch

3. **Error Handling:**
   - **abort:** Stop on first error, rollback transaction
   - **skip:** Continue processing, log error
   - **collect:** Process all, return detailed error list

4. **Memory Management:**
   - Process in batches to avoid OOM
   - Clear embedding cache after each batch
   - Use prepared statements for speed

5. **Progress Reporting:**
   - Optional callback for progress updates
   - Emit progress events during long operations

#### Performance Benchmarks (Expected)

- **Small Batch (10 records):** 50-100ms
- **Medium Batch (100 records):** 500-800ms
- **Large Batch (1000 records):** 5-8 seconds
- **Throughput:** 100-200 records/sec (sequential), 300-500 records/sec (parallel)

---

### 4. agentdb_search

**Purpose:** Perform k-NN vector similarity search with advanced filtering

**Status:** 🔴 Not Implemented
**Priority:** High (P0)
**Complexity:** High (6-8 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_search',
  description: 'Perform k-NN vector similarity search with advanced filtering',
  inputSchema: {
    type: 'object',
    properties: {
      query: {
        type: 'string',
        description: 'Query text to search for'
      },
      query_embedding: {
        type: 'array',
        description: 'Pre-computed query embedding (optional)',
        items: { type: 'number' }
      },
      collections: {
        type: 'array',
        description: 'Collections to search (default: all)',
        items: {
          type: 'string',
          enum: ['episodes', 'skills', 'notes', 'facts']
        },
        default: ['episodes', 'skills', 'notes']
      },
      k: {
        type: 'number',
        description: 'Number of results to return',
        default: 10,
        minimum: 1,
        maximum: 1000
      },
      similarity_threshold: {
        type: 'number',
        description: 'Minimum cosine similarity (0-1)',
        default: 0.0,
        minimum: 0,
        maximum: 1
      },
      filters: {
        type: 'object',
        description: 'Metadata filters for results',
        properties: {
          min_reward: { type: 'number' },
          max_latency_ms: { type: 'number' },
          success_only: { type: 'boolean' },
          tags: { type: 'array', items: { type: 'string' } },
          time_range: {
            type: 'object',
            properties: {
              start: { type: 'number' },
              end: { type: 'number' }
            }
          }
        }
      },
      rerank: {
        type: 'object',
        description: 'Reranking configuration (causal-aware)',
        properties: {
          enabled: { type: 'boolean', default: false },
          alpha: { type: 'number', default: 0.7 },
          beta: { type: 'number', default: 0.2 },
          gamma: { type: 'number', default: 0.1 }
        }
      },
      include_embeddings: {
        type: 'boolean',
        description: 'Include embeddings in response',
        default: false
      }
    },
    required: ['query']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `🔍 Search Results (k=${k})

Query: "${query}"
Collections: ${collections.join(', ')}
Similarity Threshold: ${similarity_threshold}

📊 Results (${results.length}):

${results.slice(0, 10).map((r, i) => `
${i + 1}. ${r.collection} #${r.id} (similarity: ${r.similarity.toFixed(3)})
   ${r.title || r.task || r.text?.substring(0, 80)}...
   ${r.metadata ? `Metadata: ${JSON.stringify(r.metadata).substring(0, 100)}` : ''}
   ${rerank.enabled ? `Utility Score: ${r.utility_score.toFixed(3)} (uplift: ${r.uplift?.toFixed(3) || 'N/A'})` : ''}
`).join('\n')}

${results.length > 10 ? `... and ${results.length - 10} more results` : ''}

⚡ Performance:
  Query Time: ${query_time_ms}ms
  Embedding Generation: ${embedding_ms}ms
  Vector Search: ${vector_search_ms}ms
  ${rerank.enabled ? `Reranking: ${rerank_ms}ms` : ''}
  Results Filtered: ${filtered_count}

${filters ? `🔧 Filters Applied:\n${JSON.stringify(filters, null, 2)}` : ''}`
  }]
}
```

#### Implementation Notes

1. **Query Embedding:**
   - Generate if query_embedding not provided
   - Validate dimensions match database vectors

2. **Multi-Collection Search:**
   - Query each collection's embedding table
   - Union results and sort by similarity
   - Apply collection-specific filters

3. **Similarity Computation:**
   - Use cosine similarity: dot(a, b) / (norm(a) * norm(b))
   - Pre-filter by threshold before sorting
   - Use SQLite BLOB operations for efficiency

4. **Filtering:**
   - Apply SQL WHERE clauses for metadata filters
   - Support complex time range queries
   - Tag filtering with JSON_EXTRACT (SQLite 3.38+)

5. **Reranking (Optional):**
   - Load causal edges for top results
   - Apply utility formula: U = α*similarity + β*uplift - γ*latency
   - Re-sort by utility score

6. **Result Formatting:**
   - Include all metadata by default
   - Optionally include raw embeddings
   - Truncate text fields for readability

#### Search Strategies

**Exact Similarity Search:**
```sql
SELECT * FROM (
  SELECT id, embedding FROM episode_embeddings
  UNION ALL
  SELECT id, embedding FROM skill_embeddings
  UNION ALL
  SELECT id, embedding FROM note_embeddings
) WHERE cosine_similarity(embedding, ?) > threshold
ORDER BY cosine_similarity(embedding, ?) DESC
LIMIT k
```

**Filtered Search:**
```sql
SELECT e.*, ee.embedding, cosine_similarity(ee.embedding, ?) as similarity
FROM episodes e
JOIN episode_embeddings ee ON e.id = ee.episode_id
WHERE e.reward >= ? AND e.success = 1
  AND cosine_similarity(ee.embedding, ?) > ?
ORDER BY similarity DESC
LIMIT ?
```

**Hybrid Search (with causal reranking):**
1. Vector search → top 2k candidates
2. Load causal edges for candidates
3. Compute utility scores
4. Re-sort and return top k

---

### 5. agentdb_delete

**Purpose:** Delete vectors by ID with cascade options

**Status:** 🔴 Not Implemented
**Priority:** Medium (P1)
**Complexity:** Low (2-3 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_delete',
  description: 'Delete vectors by ID with cascade options for related records',
  inputSchema: {
    type: 'object',
    properties: {
      collection: {
        type: 'string',
        description: 'Collection name',
        enum: ['episodes', 'skills', 'notes', 'facts', 'events']
      },
      ids: {
        type: 'array',
        description: 'Record IDs to delete',
        items: { type: 'number' },
        minItems: 1,
        maxItems: 100
      },
      cascade: {
        type: 'boolean',
        description: 'Delete related records (embeddings, causal edges, etc.)',
        default: true
      },
      soft_delete: {
        type: 'boolean',
        description: 'Mark as deleted instead of removing (if collection supports it)',
        default: false
      },
      dry_run: {
        type: 'boolean',
        description: 'Preview what would be deleted without actually deleting',
        default: false
      }
    },
    required: ['collection', 'ids']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `${dry_run ? '🔍 DRY RUN - No records were deleted' : '✅ Deletion completed'}

📊 Summary:
  Collection: ${collection}
  Records Deleted: ${deleted_count}/${ids.length}
  Cascade Enabled: ${cascade}
  Soft Delete: ${soft_delete}

${cascade ? `
🗑️ Cascade Deletions:
  Embeddings: ${embeddings_deleted}
  Causal Edges: ${causal_edges_deleted}
  Skill Links: ${skill_links_deleted}
  Consolidated Memories: ${consolidated_deleted}
  Provenance Records: ${provenance_deleted}
` : ''}

${errors.length > 0 ? `
❌ Errors (${errors.length}):
${errors.slice(0, 5).map((e, i) => `  ${i + 1}. ID ${e.id}: ${e.message}`).join('\n')}
` : ''}

💾 Storage:
  Database Size Change: ${size_delta_mb}MB
  Space Reclaimed: ${space_reclaimed_mb}MB (after VACUUM)`
  }]
}
```

#### Implementation Notes

1. **Validation:**
   - Check IDs exist before deletion
   - Warn if deleting records with high importance/usage

2. **Cascade Deletion:**
   - Use SQLite foreign keys (ON DELETE CASCADE)
   - Manual cleanup for non-FK relationships
   - Delete embeddings from _embeddings tables
   - Remove causal edges referencing deleted records

3. **Soft Delete:**
   - Add deleted_at timestamp
   - Filter soft-deleted records in queries
   - Implement purge_soft_deleted() maintenance function

4. **Dry Run:**
   - Query what would be deleted
   - Calculate size savings
   - Show dependency tree

5. **Transaction Safety:**
   - Wrap in transaction
   - Rollback on error
   - Return deleted record count

#### Cascade Logic by Collection

**Episodes:**
- Delete episode_embeddings (FK cascade)
- Remove causal_observations (FK cascade)
- Remove causal edges where from_memory_id or to_memory_id matches
- Update skills.created_from_episode to NULL

**Skills:**
- Delete skill_embeddings (FK cascade)
- Delete skill_links (FK cascade)
- Remove causal edges referencing skill

**Notes:**
- Delete note_embeddings (FK cascade)
- Remove provenance_sources
- Remove causal edges

**Facts:**
- No embeddings (no cascade needed)
- Remove from causal edges

---

### 6. agentdb_stats

**Purpose:** Enhanced database statistics with collection-specific metrics

**Status:** 🟡 Partially Implemented (db_stats exists but limited)
**Priority:** Medium (P1)
**Complexity:** Low (2 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_stats',
  description: 'Get comprehensive database statistics with collection-specific metrics',
  inputSchema: {
    type: 'object',
    properties: {
      detailed: {
        type: 'boolean',
        description: 'Include detailed per-collection stats',
        default: false
      },
      collections: {
        type: 'array',
        description: 'Specific collections to analyze (default: all)',
        items: { type: 'string' }
      },
      include_disk_usage: {
        type: 'boolean',
        description: 'Calculate disk space usage',
        default: true
      },
      include_performance: {
        type: 'boolean',
        description: 'Include query performance stats',
        default: false
      }
    }
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `📊 AgentDB Statistics

💾 Database Info:
  Path: ${db_path}
  Size: ${db_size_mb}MB
  Page Size: ${page_size} bytes
  Page Count: ${page_count}
  Fragmentation: ${fragmentation}%

📈 Record Counts:
  Episodes: ${episodes_count} (${episodes_with_embeddings} with embeddings)
  Skills: ${skills_count} (avg success rate: ${avg_skill_success}%)
  Notes: ${notes_count}
  Facts: ${facts_count}
  Events: ${events_count}
  Consolidated Memories: ${consolidated_count}

🔗 Graph Statistics:
  Causal Edges: ${causal_edges_count}
  Exp Nodes: ${exp_nodes_count}
  Exp Edges: ${exp_edges_count}
  Skill Links: ${skill_links_count}

🧠 Frontier Features:
  Causal Experiments: ${experiments_count}
  Recall Certificates: ${certificates_count}
  Provenance Sources: ${provenance_count}
  Justification Paths: ${justification_count}

📐 Embedding Stats:
  Total Vectors: ${total_vectors}
  Dimensions: ${embedding_dimensions}
  Storage Space: ${vector_storage_mb}MB
  Models Used: ${models.join(', ')}

${detailed ? `
📋 Collection Details:

Episodes:
  Total: ${episodes_count}
  Success Rate: ${episode_success_rate}%
  Avg Reward: ${avg_episode_reward}
  Avg Latency: ${avg_episode_latency}ms
  Sessions: ${unique_sessions}
  Tasks: ${unique_tasks}

Skills:
  Total: ${skills_count}
  Avg Success Rate: ${avg_skill_success}%
  Total Uses: ${total_skill_uses}
  Most Used: ${top_skill.name} (${top_skill.uses} uses)

Notes:
  Total: ${notes_count}
  Types: ${note_types.join(', ')}
  Avg Importance: ${avg_importance}
  Most Accessed: ${top_note.title}

Facts:
  Total: ${facts_count}
  Unique Subjects: ${unique_subjects}
  Unique Predicates: ${unique_predicates}
  Avg Confidence: ${avg_fact_confidence}
` : ''}

${include_performance ? `
⚡ Performance Metrics:
  Avg Query Time: ${avg_query_ms}ms
  Cache Hit Rate: ${cache_hit_rate}%
  Index Usage: ${index_usage}%
  Slow Queries: ${slow_query_count}
` : ''}`
  }]
}
```

#### Implementation Notes

1. **Basic Stats:**
   - SELECT COUNT(*) from all tables
   - PRAGMA page_count, page_size
   - Database file size from filesystem

2. **Detailed Stats:**
   - Aggregate statistics per collection
   - Top performers (skills, notes)
   - Time-series trends

3. **Performance Stats:**
   - Query memory_access_log
   - Analyze query plans
   - Index usage statistics

4. **Disk Usage:**
   - Use PRAGMA freelist_count
   - Calculate fragmentation
   - Estimate vector storage overhead

---

### 7. agentdb_pattern_store

**Purpose:** Store reasoning patterns for meta-learning

**Status:** 🔴 Not Implemented
**Priority:** Medium (P1)
**Complexity:** Medium (4-5 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_pattern_store',
  description: 'Store reasoning patterns for meta-learning and strategy optimization',
  inputSchema: {
    type: 'object',
    properties: {
      pattern_type: {
        type: 'string',
        description: 'Type of reasoning pattern',
        enum: [
          'problem_solving',
          'decomposition',
          'debugging',
          'optimization',
          'abstraction',
          'analogy',
          'backtracking',
          'heuristic'
        ]
      },
      name: {
        type: 'string',
        description: 'Pattern name/identifier'
      },
      description: {
        type: 'string',
        description: 'Human-readable pattern description'
      },
      pattern_data: {
        type: 'object',
        description: 'Structured pattern data',
        properties: {
          trigger_conditions: { type: 'array' },
          steps: { type: 'array' },
          success_indicators: { type: 'array' },
          failure_modes: { type: 'array' },
          prerequisites: { type: 'array' }
        }
      },
      source_episodes: {
        type: 'array',
        description: 'Episode IDs that generated this pattern',
        items: { type: 'number' }
      },
      effectiveness_score: {
        type: 'number',
        description: 'Estimated effectiveness (0-1)',
        minimum: 0,
        maximum: 1
      },
      domain: {
        type: 'string',
        description: 'Problem domain (e.g., coding, data_analysis, debugging)'
      },
      tags: {
        type: 'array',
        description: 'Searchable tags',
        items: { type: 'string' }
      }
    },
    required: ['pattern_type', 'name', 'description', 'pattern_data']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `✅ Stored reasoning pattern #${pattern_id}

🧠 Pattern Details:
  Name: ${name}
  Type: ${pattern_type}
  Domain: ${domain}
  Effectiveness: ${effectiveness_score.toFixed(2)}

📋 Pattern Structure:
  Trigger Conditions: ${trigger_conditions.length}
  Steps: ${steps.length}
  Success Indicators: ${success_indicators.length}
  Failure Modes: ${failure_modes.length}

🔗 Provenance:
  Source Episodes: ${source_episodes.length}
  Tags: ${tags.join(', ')}

💾 Storage:
  Embedding Generated: ${embedding_generated}
  Vector Dimension: ${dimension}
  Pattern ID: ${pattern_id}`
  }]
}
```

#### Implementation Notes

1. **Storage Strategy:**
   - Store in notes table with note_type='reasoning_pattern'
   - JSON encode pattern_data in metadata field
   - Generate embedding from name + description + steps

2. **Pattern Structure:**
   ```typescript
   interface ReasoningPattern {
     trigger_conditions: string[];  // When to apply pattern
     steps: {                        // Execution steps
       step_number: number;
       action: string;
       expected_outcome: string;
     }[];
     success_indicators: string[];   // How to know it worked
     failure_modes: string[];        // Common failure patterns
     prerequisites: string[];        // Required knowledge/skills
   }
   ```

3. **Meta-Learning Integration:**
   - Link to source episodes
   - Track pattern usage in memory_access_log
   - Update effectiveness_score based on outcomes

4. **Searchability:**
   - Embed pattern description
   - Index tags for filtering
   - Link to domain/task taxonomy

---

### 8. agentdb_pattern_search

**Purpose:** Search for applicable reasoning patterns

**Status:** 🔴 Not Implemented
**Priority:** Medium (P1)
**Complexity:** Medium (3-4 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_pattern_search',
  description: 'Search for applicable reasoning patterns by task description',
  inputSchema: {
    type: 'object',
    properties: {
      task: {
        type: 'string',
        description: 'Task or problem description'
      },
      pattern_types: {
        type: 'array',
        description: 'Filter by pattern types',
        items: { type: 'string' }
      },
      domain: {
        type: 'string',
        description: 'Problem domain filter'
      },
      k: {
        type: 'number',
        description: 'Number of patterns to return',
        default: 5,
        minimum: 1,
        maximum: 50
      },
      min_effectiveness: {
        type: 'number',
        description: 'Minimum effectiveness score',
        default: 0.5,
        minimum: 0,
        maximum: 1
      },
      sort_by: {
        type: 'string',
        description: 'Sorting strategy',
        enum: ['similarity', 'effectiveness', 'recent_usage', 'composite'],
        default: 'composite'
      }
    },
    required: ['task']
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `🔍 Found ${results.length} reasoning patterns for: "${task}"

${results.map((p, i) => `
${i + 1}. ${p.name} (${p.pattern_type})
   Similarity: ${p.similarity.toFixed(3)}
   Effectiveness: ${p.effectiveness_score.toFixed(2)}
   Composite Score: ${p.composite_score.toFixed(3)}

   Description: ${p.description}

   Trigger Conditions:
   ${p.pattern_data.trigger_conditions.map(c => `    • ${c}`).join('\n')}

   Steps (${p.pattern_data.steps.length}):
   ${p.pattern_data.steps.slice(0, 3).map((s, j) => `    ${j + 1}. ${s.action}`).join('\n')}
   ${p.pattern_data.steps.length > 3 ? `    ... and ${p.pattern_data.steps.length - 3} more steps` : ''}

   Success Indicators:
   ${p.pattern_data.success_indicators.map(i => `    ✓ ${i}`).join('\n')}

   Domain: ${p.domain || 'General'}
   Used: ${p.usage_count} times
   Last Used: ${p.last_used ? formatDate(p.last_used) : 'Never'}

`).join('\n')}

⚡ Search Performance:
  Query Time: ${query_ms}ms
  Patterns Evaluated: ${evaluated_count}`
  }]
}
```

#### Implementation Notes

1. **Search Strategy:**
   - Embed task description
   - Search notes with note_type='reasoning_pattern'
   - Filter by pattern_types and domain
   - Apply min_effectiveness threshold

2. **Composite Scoring:**
   ```typescript
   composite_score =
     0.4 * similarity +
     0.3 * effectiveness_score +
     0.2 * (usage_count / max_usage) +
     0.1 * recency_factor
   ```

3. **Result Enrichment:**
   - Load full pattern_data from metadata
   - Include usage statistics from memory_access_log
   - Format steps for readability

4. **Integration with Causal Graph:**
   - Optionally include causal evidence
   - Show patterns that causally lead to success
   - Highlight prerequisite patterns

---

### 9. agentdb_pattern_stats

**Purpose:** Get statistics about stored reasoning patterns

**Status:** 🔴 Not Implemented
**Priority:** Low (P2)
**Complexity:** Low (1-2 hours)

#### Tool Definition

```typescript
{
  name: 'agentdb_pattern_stats',
  description: 'Get statistics about stored reasoning patterns and their usage',
  inputSchema: {
    type: 'object',
    properties: {
      pattern_type: {
        type: 'string',
        description: 'Filter by pattern type'
      },
      domain: {
        type: 'string',
        description: 'Filter by domain'
      },
      include_usage: {
        type: 'boolean',
        description: 'Include usage statistics',
        default: true
      }
    }
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `📊 Reasoning Pattern Statistics

📈 Overview:
  Total Patterns: ${total_patterns}
  Pattern Types: ${pattern_types.length}
  Domains: ${domains.length}
  Avg Effectiveness: ${avg_effectiveness.toFixed(2)}

🏆 Top Patterns by Type:
${pattern_type_stats.map(t => `
  ${t.type}: ${t.count} patterns (avg effectiveness: ${t.avg_effectiveness.toFixed(2)})
    Most Used: ${t.top_pattern.name} (${t.top_pattern.uses} uses)
`).join('\n')}

🌍 Domains:
${domain_stats.map(d => `
  ${d.domain}: ${d.count} patterns (avg effectiveness: ${d.avg_effectiveness.toFixed(2)})
`).join('\n')}

${include_usage ? `
📊 Usage Statistics:
  Total Applications: ${total_applications}
  Success Rate: ${pattern_success_rate}%
  Avg Application Time: ${avg_application_time}ms
  Most Applied: ${most_applied.name} (${most_applied.uses} times)
  Highest Success Rate: ${best_pattern.name} (${best_pattern.success_rate}%)
` : ''}`
  }]
}
```

#### Implementation Notes

1. Query notes table with note_type='reasoning_pattern'
2. Group by pattern_type and domain
3. Calculate aggregate statistics
4. Join with memory_access_log for usage stats
5. Rank patterns by effectiveness and usage

---

### 10. agentdb_clear_cache

**Purpose:** Clear query cache and rebuild indexes

**Status:** 🔴 Not Implemented
**Priority:** Low (P2)
**Complexity:** Low (1 hour)

#### Tool Definition

```typescript
{
  name: 'agentdb_clear_cache',
  description: 'Clear query cache, temporary data, and optionally rebuild indexes',
  inputSchema: {
    type: 'object',
    properties: {
      cache_type: {
        type: 'string',
        description: 'Type of cache to clear',
        enum: ['embedding', 'query', 'all'],
        default: 'all'
      },
      rebuild_indexes: {
        type: 'boolean',
        description: 'Rebuild database indexes',
        default: false
      },
      vacuum: {
        type: 'boolean',
        description: 'Run VACUUM to reclaim space',
        default: false
      },
      analyze: {
        type: 'boolean',
        description: 'Run ANALYZE to update statistics',
        default: true
      }
    }
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `✅ Cache cleared successfully

🧹 Cleanup Summary:
  Embedding Cache: ${embedding_cache_cleared ? 'Cleared' : 'Skipped'}
  Query Cache: ${query_cache_cleared ? 'Cleared' : 'Skipped'}

${rebuild_indexes ? `
🔨 Index Rebuild:
  Indexes Rebuilt: ${indexes_rebuilt}
  Time Taken: ${rebuild_time_ms}ms
` : ''}

${vacuum ? `
💾 VACUUM:
  Space Reclaimed: ${space_reclaimed_mb}MB
  Database Size: ${db_size_before_mb}MB → ${db_size_after_mb}MB
  Time Taken: ${vacuum_time_ms}ms
` : ''}

${analyze ? `
📊 ANALYZE:
  Tables Analyzed: ${tables_analyzed}
  Statistics Updated: ${stats_updated}
  Time Taken: ${analyze_time_ms}ms
` : ''}

⚡ Performance Impact:
  Before: ${perf_before.avg_query_ms}ms avg query
  After: ${perf_after.avg_query_ms}ms avg query
  Improvement: ${improvement_percent}%`
  }]
}
```

#### Implementation Notes

1. **Embedding Cache:**
   - Clear EmbeddingService cache directory
   - Delete cached model files if requested

2. **Query Cache:**
   - Clear SQLite query cache
   - Reset prepared statement cache

3. **Index Rebuild:**
   ```sql
   REINDEX;
   ```

4. **VACUUM:**
   ```sql
   VACUUM;
   ```

5. **ANALYZE:**
   ```sql
   ANALYZE;
   ```

---

## Enhanced Frontier Tools

### 11. skill_consolidate

**Purpose:** Automatically create skills from successful episodes

**Status:** 🔴 Not Implemented
**Priority:** High (P0)
**Complexity:** Medium (4-5 hours)

#### Tool Definition

```typescript
{
  name: 'skill_consolidate',
  description: 'Automatically create reusable skills from successful episode patterns',
  inputSchema: {
    type: 'object',
    properties: {
      min_attempts: {
        type: 'number',
        description: 'Minimum attempts for a task to be considered',
        default: 3,
        minimum: 2
      },
      min_success_rate: {
        type: 'number',
        description: 'Minimum success rate (0-1) to promote to skill',
        default: 0.6,
        minimum: 0,
        maximum: 1
      },
      min_reward: {
        type: 'number',
        description: 'Minimum average reward',
        default: 0.7,
        minimum: 0,
        maximum: 1
      },
      time_window_days: {
        type: 'number',
        description: 'Only consider episodes from last N days',
        default: 7,
        minimum: 1
      },
      auto_link: {
        type: 'boolean',
        description: 'Automatically create skill relationships',
        default: true
      },
      dry_run: {
        type: 'boolean',
        description: 'Preview skills without creating them',
        default: false
      }
    }
  }
}
```

#### Response Format

```typescript
{
  content: [{
    type: 'text',
    text: `${dry_run ? '🔍 DRY RUN - Skill Consolidation Preview' : '✅ Skill Consolidation Complete'}

📊 Analysis:
  Episodes Analyzed: ${episodes_analyzed}
  Time Window: ${time_window_days} days
  Candidates Found: ${candidates_found}
  Skills Created: ${skills_created}
  Skills Updated: ${skills_updated}

${candidates.map((c, i) => `
${i + 1}. ${dry_run ? '[PREVIEW]' : '✅'} ${c.task}
   Attempts: ${c.attempt_count}
   Success Rate: ${(c.success_rate * 100).toFixed(1)}%
   Avg Reward: ${c.avg_reward.toFixed(2)}
   Avg Latency: ${c.avg_latency}ms
   Source Episodes: ${c.episode_ids.length}
   ${dry_run ? 'Would create new skill' : `Created skill #${c.skill_id}`}
   ${auto_link && c.prerequisites.length > 0 ? `Prerequisites: ${c.prerequisites.join(', ')}` : ''}
`).join('\n')}

${auto_link ? `
🔗 Skill Relationships:
  Prerequisites Created: ${prereq_links}
  Alternatives Created: ${alternative_links}
  Refinements Created: ${refinement_links}
` : ''}

⚡ Performance:
  Analysis Time: ${analysis_ms}ms
  Skill Creation Time: ${creation_ms}ms
  Total Time: ${total_ms}ms`
  }]
}
```

#### Implementation Notes

1. **Candidate Discovery:**
   - Use skill_candidates view from schema.sql
   - Query episodes grouped by task
   - Apply filters: min_attempts, min_success_rate, min_reward
   - Consider time window

2. **Skill Creation:**
   - Use SkillLibrary.consolidateEpisodesIntoSkills()
   - Extract metadata from source episodes
   - Generate composite code from successful outputs
   - Compute initial statistics

3. **Auto-Linking:**
   - Analyze task dependencies
   - Find prerequisite patterns (tasks that frequently occur before)
   - Identify alternatives (tasks with similar outcomes)
   - Detect refinements (improved versions of existing skills)

4. **Linking Algorithm:**
   ```typescript
   // Prerequisites: Task A frequently precedes Task B
   if (cooccurrence_rate(A, B) > 0.7 && temporal_order(A, B)) {
     linkSkills(B, A, 'prerequisite');
   }

   // Alternatives: Tasks A and B solve same problem
   if (similarity(A, B) > 0.8 && same_outcome(A, B)) {
     linkSkills(A, B, 'alternative');
   }

   // Refinements: Task B is improved version of Task A
   if (similarity(A, B) > 0.7 && reward(B) > reward(A)) {
     linkSkills(A, B, 'refinement');
   }
   ```

5. **Dry Run:**
   - Perform all analysis
   - Don't write to database
   - Return detailed preview

#### Integration with Existing Tools

- Uses ReflexionMemory to query episodes
- Uses SkillLibrary to create and link skills
- Uses CausalMemoryGraph to infer causal relationships
- Updates skill statistics automatically

---

## Integration Points

### Integration with Existing 9 Tools

#### Shared Controllers

All 11 new tools integrate with existing controllers:

1. **EmbeddingService:**
   - Used by: agentdb_insert, agentdb_insert_batch, agentdb_search, agentdb_pattern_store
   - Functionality: Generate embeddings for text

2. **ReflexionMemory:**
   - Used by: skill_consolidate, agentdb_insert (episodes)
   - Functionality: Episode storage and retrieval

3. **SkillLibrary:**
   - Used by: skill_consolidate, agentdb_insert (skills)
   - Functionality: Skill management

4. **CausalMemoryGraph:**
   - Used by: agentdb_search (reranking), skill_consolidate
   - Functionality: Causal relationships

5. **CausalRecall:**
   - Used by: agentdb_search (reranking mode)
   - Functionality: Utility-based ranking

6. **ExplainableRecall:**
   - Used by: agentdb_search (certificate generation)
   - Functionality: Provenance tracking

#### Data Flow Diagram

```
User Query
    │
    ├──> agentdb_insert ──┐
    ├──> agentdb_search ──┤
    ├──> agentdb_delete ──┤
    │                      │
    │                      v
    │              Controllers Layer
    │           ┌──────────────────┐
    │           │ EmbeddingService │
    │           │ ReflexionMemory  │
    │           │ SkillLibrary     │
    │           │ CausalGraph      │
    │           └──────────────────┘
    │                      │
    │                      v
    │                  Database
    │            ┌──────────────────┐
    │            │ Episodes         │
    │            │ Skills           │
    │            │ Notes            │
    │            │ Causal Edges     │
    │            │ Embeddings       │
    │            └──────────────────┘
    │                      │
    └──────────────────────┴──> Response
```

#### Compatibility Matrix

| New Tool | Uses Embedding | Uses Causal | Uses Recall | Uses Skills |
|----------|----------------|-------------|-------------|-------------|
| agentdb_init | ✅ | ✅ | ✅ | ✅ |
| agentdb_insert | ✅ | ❌ | ❌ | Optional |
| agentdb_insert_batch | ✅ | ❌ | ❌ | ❌ |
| agentdb_search | ✅ | Optional | Optional | ❌ |
| agentdb_delete | ❌ | ✅ | ❌ | ✅ |
| agentdb_stats | ❌ | ✅ | ✅ | ✅ |
| agentdb_pattern_store | ✅ | ❌ | ❌ | ❌ |
| agentdb_pattern_search | ✅ | Optional | ❌ | ❌ |
| agentdb_pattern_stats | ❌ | ❌ | ❌ | ❌ |
| agentdb_clear_cache | ✅ | ❌ | ❌ | ❌ |
| skill_consolidate | ✅ | ✅ | ❌ | ✅ |

---

## Error Handling Strategy

### Error Categories

#### 1. Database Errors

**Causes:**
- Database locked
- Disk full
- Corrupted database
- Schema mismatch

**Handling:**
```typescript
try {
  // Database operation
} catch (error) {
  if (error.code === 'SQLITE_BUSY') {
    return {
      content: [{
        type: 'text',
        text: '❌ Database is locked by another process\n\n' +
              '💡 Suggestions:\n' +
              '  • Close other AgentDB connections\n' +
              '  • Wait a few seconds and retry\n' +
              '  • Check for zombie processes'
      }],
      isError: true
    };
  }
  // Handle other errors...
}
```

#### 2. Validation Errors

**Causes:**
- Invalid parameters
- Missing required fields
- Out-of-range values

**Handling:**
```typescript
if (!collections.includes(collection)) {
  return {
    content: [{
      type: 'text',
      text: `❌ Invalid collection: ${collection}\n\n` +
            `Valid collections: ${collections.join(', ')}\n\n` +
            `💡 Did you mean: ${suggestCorrection(collection, collections)}?`
    }],
    isError: true
  };
}
```

#### 3. Embedding Errors

**Causes:**
- Model not loaded
- OOM during embedding
- Invalid text encoding

**Handling:**
```typescript
try {
  const embedding = await embedder.embed(text);
} catch (error) {
  return {
    content: [{
      type: 'text',
      text: '❌ Embedding generation failed\n\n' +
            `Error: ${error.message}\n\n` +
            '💡 Suggestions:\n' +
            '  • Ensure embedding model is initialized\n' +
            '  • Check available memory\n' +
            '  • Try shorter text (current: ${text.length} chars)\n' +
            '  • Use skip_embedding=true for text-only storage'
    }],
    isError: true
  };
}
```

#### 4. Performance Errors

**Causes:**
- Query timeout
- Too many results
- Memory limit exceeded

**Handling:**
```typescript
const MAX_RESULTS = 10000;
if (k > MAX_RESULTS) {
  return {
    content: [{
      type: 'text',
      text: `❌ Requested too many results: k=${k}\n\n` +
            `Maximum allowed: ${MAX_RESULTS}\n\n` +
            `💡 Suggestions:\n` +
            `  • Reduce k to ${MAX_RESULTS} or less\n` +
            `  • Use filters to narrow results\n` +
            `  • Use pagination with offset/limit`
    }],
    isError: true
  };
}
```

### Error Response Format

All errors follow this format:

```typescript
{
  content: [{
    type: 'text',
    text: `❌ ${error_title}

${error_description}

💡 Suggestions:
  • ${suggestion_1}
  • ${suggestion_2}
  • ${suggestion_3}

🔍 Debug Info:
  Tool: ${tool_name}
  Timestamp: ${timestamp}
  Error Code: ${error_code}
  ${additional_context}`
  }],
  isError: true
}
```

---

## Implementation Roadmap

### Phase 1: Core Operations (Weeks 1-2)

**Priority:** P0 - Critical for basic functionality

1. **agentdb_init** (2-3 hours)
   - Database initialization
   - Schema setup
   - Configuration management

2. **agentdb_insert** (3-4 hours)
   - Single vector insertion
   - Embedding generation
   - Metadata handling

3. **agentdb_insert_batch** (4-5 hours)
   - Batch insertion
   - Transaction management
   - Performance optimization

4. **agentdb_search** (6-8 hours)
   - k-NN search
   - Multi-collection queries
   - Filtering and reranking

**Deliverables:**
- Core CRUD operations functional
- Unit tests with 80% coverage
- Integration tests with existing tools
- Performance benchmarks documented

### Phase 2: Management Tools (Week 3)

**Priority:** P1 - Important for production use

1. **agentdb_delete** (2-3 hours)
   - Record deletion
   - Cascade logic
   - Soft delete support

2. **agentdb_stats** (2 hours)
   - Enhanced statistics
   - Performance metrics
   - Collection details

3. **agentdb_clear_cache** (1 hour)
   - Cache management
   - Index rebuild
   - VACUUM support

**Deliverables:**
- Database maintenance tools
- Administrative documentation
- Monitoring dashboards

### Phase 3: Advanced Features (Week 4)

**Priority:** P1 - Enhanced functionality

1. **agentdb_pattern_store** (4-5 hours)
   - Pattern storage
   - Meta-learning support
   - Provenance tracking

2. **agentdb_pattern_search** (3-4 hours)
   - Pattern retrieval
   - Composite scoring
   - Usage analytics

3. **agentdb_pattern_stats** (1-2 hours)
   - Pattern statistics
   - Usage metrics
   - Effectiveness tracking

4. **skill_consolidate** (4-5 hours)
   - Automatic skill creation
   - Relationship inference
   - Dry-run preview

**Deliverables:**
- Meta-learning capabilities
- Auto-consolidation system
- Pattern library documentation

### Phase 4: Testing & Documentation (Week 5)

**Priority:** P0 - Quality assurance

1. **Comprehensive Testing**
   - Unit tests (target: 90% coverage)
   - Integration tests
   - End-to-end scenarios
   - Performance benchmarks
   - Load testing

2. **Documentation**
   - Tool reference guide
   - Usage examples
   - Best practices
   - Migration guide
   - API documentation

3. **Performance Optimization**
   - Query optimization
   - Index tuning
   - Memory profiling
   - Bottleneck resolution

**Deliverables:**
- Test suite with 90% coverage
- Complete documentation
- Performance baseline
- Production-ready release

### Phase 5: Deployment (Week 6)

**Priority:** P0 - Release preparation

1. **Packaging**
   - NPM package update
   - Version bump (1.2.1 → 1.3.0)
   - Changelog preparation
   - Migration scripts

2. **Deployment**
   - Release to npm
   - Update documentation site
   - Announce new tools
   - Gather feedback

3. **Monitoring**
   - Usage analytics
   - Error tracking
   - Performance monitoring
   - User feedback collection

**Deliverables:**
- Released package
- Updated documentation
- Monitoring dashboards
- Support channels

---

## Implementation Complexity Estimates

| Tool | Complexity | Time Estimate | Dependencies |
|------|-----------|---------------|--------------|
| agentdb_init | Low | 2-3 hours | Database, Schema |
| agentdb_insert | Medium | 3-4 hours | EmbeddingService |
| agentdb_insert_batch | Medium | 4-5 hours | EmbeddingService, Transactions |
| agentdb_search | High | 6-8 hours | EmbeddingService, CausalRecall |
| agentdb_delete | Low | 2-3 hours | Database, Foreign Keys |
| agentdb_stats | Low | 2 hours | Database |
| agentdb_pattern_store | Medium | 4-5 hours | EmbeddingService, Notes |
| agentdb_pattern_search | Medium | 3-4 hours | EmbeddingService |
| agentdb_pattern_stats | Low | 1-2 hours | Database |
| agentdb_clear_cache | Low | 1 hour | Filesystem, SQLite |
| skill_consolidate | Medium | 4-5 hours | SkillLibrary, Causal |
| **TOTAL** | **Mixed** | **33-42 hours** | **4-5 weeks** |

---

## Database Schema Requirements

### New Tables Required

None - all tools use existing schema from schema.sql and frontier-schema.sql

### Schema Extensions Needed

1. **Add deleted_at column for soft deletes (optional):**
   ```sql
   ALTER TABLE episodes ADD COLUMN deleted_at INTEGER;
   ALTER TABLE skills ADD COLUMN deleted_at INTEGER;
   ALTER TABLE notes ADD COLUMN deleted_at INTEGER;
   ```

2. **Add pattern_type to notes for pattern storage:**
   ```sql
   -- Already supported via note_type field
   -- Just need to add 'reasoning_pattern' as valid type
   ```

3. **Add indexes for performance:**
   ```sql
   CREATE INDEX IF NOT EXISTS idx_notes_type_importance
     ON notes(note_type, importance DESC);

   CREATE INDEX IF NOT EXISTS idx_episodes_deleted
     ON episodes(deleted_at) WHERE deleted_at IS NOT NULL;
   ```

### Migration Script

```sql
-- Migration: Add soft delete support
-- Version: 1.3.0
-- Date: 2025-10-22

BEGIN TRANSACTION;

-- Add deleted_at columns
ALTER TABLE episodes ADD COLUMN deleted_at INTEGER;
ALTER TABLE skills ADD COLUMN deleted_at INTEGER;
ALTER TABLE notes ADD COLUMN deleted_at INTEGER;
ALTER TABLE facts ADD COLUMN deleted_at INTEGER;

-- Add indexes for soft delete queries
CREATE INDEX IF NOT EXISTS idx_episodes_deleted
  ON episodes(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_skills_deleted
  ON skills(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_notes_deleted
  ON notes(deleted_at) WHERE deleted_at IS NOT NULL;

-- Add composite index for pattern search
CREATE INDEX IF NOT EXISTS idx_notes_type_importance
  ON notes(note_type, importance DESC);

-- Update schema version
CREATE TABLE IF NOT EXISTS schema_version (
  version TEXT PRIMARY KEY,
  applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
  description TEXT
);

INSERT INTO schema_version (version, description)
VALUES ('1.3.0', 'Add soft delete support and pattern search indexes');

COMMIT;
```

---

## Testing Strategy

### Unit Tests

Each tool needs comprehensive unit tests:

```typescript
describe('agentdb_insert', () => {
  it('should insert episode with embedding', async () => {
    const result = await tools.agentdb_insert({
      collection: 'episodes',
      text: 'Test task',
      metadata: { reward: 0.9, success: true }
    });
    expect(result.record_id).toBeDefined();
  });

  it('should handle duplicate inserts', async () => {
    // Test idempotency
  });

  it('should validate collection name', async () => {
    // Test error handling
  });
});
```

### Integration Tests

Test interactions between tools:

```typescript
describe('Tool Integration', () => {
  it('should insert and search', async () => {
    // Insert episode
    await tools.agentdb_insert({
      collection: 'episodes',
      text: 'Build authentication system',
      metadata: { reward: 0.95, success: true }
    });

    // Search for it
    const results = await tools.agentdb_search({
      query: 'authentication',
      k: 5
    });

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].similarity).toBeGreaterThan(0.7);
  });
});
```

### Performance Tests

Benchmark each tool:

```typescript
describe('Performance', () => {
  it('batch insert should handle 1000 records < 10s', async () => {
    const records = generateRecords(1000);
    const start = Date.now();

    await tools.agentdb_insert_batch({
      collection: 'episodes',
      records
    });

    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(10000);
  });
});
```

---

## Success Metrics

### Functional Metrics

- ✅ All 11 tools implemented and tested
- ✅ Integration with existing 9 tools verified
- ✅ Unit test coverage > 90%
- ✅ Integration test coverage > 80%
- ✅ All error scenarios handled gracefully

### Performance Metrics

- ✅ agentdb_insert: < 100ms per record
- ✅ agentdb_insert_batch: > 100 records/sec
- ✅ agentdb_search: < 200ms for k=10
- ✅ agentdb_search with reranking: < 500ms for k=10
- ✅ skill_consolidate: < 5s for 1000 episodes

### Quality Metrics

- ✅ Zero critical bugs in production
- ✅ Documentation completeness: 100%
- ✅ Code review approval
- ✅ Performance benchmarks documented
- ✅ Migration path from 1.2.1 → 1.3.0

---

## Appendix A: MCP Protocol Compliance

All tools follow MCP protocol standards:

```typescript
// Tool definition format
interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
}

// Response format
interface MCPResponse {
  content: Array<{
    type: 'text' | 'image' | 'resource';
    text?: string;
    data?: string;
    mimeType?: string;
  }>;
  isError?: boolean;
}
```

---

## Appendix B: References

- **Existing Implementation:** `/workspaces/agentic-flow/packages/agentdb/dist/mcp/agentdb-mcp-server.js`
- **Schema:** `/workspaces/agentic-flow/packages/agentdb/src/schemas/`
- **Controllers:** `/workspaces/agentic-flow/packages/agentdb/src/controllers/`
- **Verification Report:** `/workspaces/agentic-flow/docs/agentdb-tools-verification.md`
- **MCP SDK:** https://github.com/modelcontextprotocol/sdk

---

**Document Version:** 1.0.0
**Last Updated:** October 22, 2025
**Prepared By:** System Architecture Designer
**Status:** Ready for Implementation
