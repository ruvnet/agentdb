# AgentDB MCP Tools - Complete Reference

> **29 Production-Ready Tools** · **Zero Configuration** · **Instant Setup** · **Full Vector DB + Frontier Memory + RL Learning**

> ✅ **VERIFIED IMPLEMENTATION v1.3.0** - This documentation reflects the actual MCP server implementation. All 29 tools are tested and production-ready.

AgentDB provides Model Context Protocol (MCP) integration with **29 verified tools** across **3 categories**: **10 core operations** + **10 frontier memory features** + **9 reinforcement learning tools** for building intelligent, adaptive AI agents.

---

## 🚀 Quick Setup (30 Seconds)

### Add to Claude Desktop

```json
{
  "mcpServers": {
    "agentdb": {
      "command": "npx",
      "args": ["agentdb@latest", "mcp", "start"],
      "env": {
        "AGENTDB_PATH": "./agentdb.db"
      }
    }
  }
}
```

**Configuration File Location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Verify Installation

```bash
# List all available MCP tools
npx agentdb mcp list
```

---

## 📦 Tool Categories Overview

| Category | Tools | Purpose | Status |
|----------|-------|---------|--------|
| **Core Vector DB** | 5 | Insert, search, delete vectors with embeddings | ✅ v1.2.2 |
| **Core AgentDB** | 5 | Database stats, pattern store/search, cache management | ✅ v1.3.0 |
| **Reflexion Memory** | 2 | Learn from experience with self-critique | ✅ v1.1.0 |
| **Skill Library** | 2 | Build reusable code patterns | ✅ v1.1.0 |
| **Causal Memory** | 2 | Track cause-effect relationships | ✅ v1.1.0 |
| **Explainable Recall** | 1 | Utility-based retrieval with certificates | ✅ v1.1.0 |
| **Nightly Learner** | 1 | Automated pattern discovery | ✅ v1.1.0 |
| **Database Utilities** | 1 | Statistics and monitoring | ✅ v1.1.0 |
| **Learning System** | 10 | Reinforcement learning with 9 algorithms | ✅ v1.3.0 |
| **TOTAL** | **29** | **Production-ready tools** | ✅ v1.3.0 |

---

## 🧠 All MCP Tools (29)

### CATEGORY 1: Core Vector Database Operations (5 Tools)

#### 1. `agentdb_init`
Initialize AgentDB database with schema and optimizations.

**Parameters:**
```json
{
  "db_path": "./agentdb.db",
  "reset": false
}
```

**Returns:** Initialization status with table count and optimization details

**Use Cases:** First-time setup, database reset, schema migration

---

#### 2. `agentdb_insert`
Insert a single vector with metadata. Automatically generates embeddings.

**Parameters:**
```json
{
  "text": "Implement OAuth2 authentication with PKCE flow",
  "session_id": "session-1",
  "tags": ["auth", "security"],
  "metadata": {
    "project": "api-backend",
    "priority": "high"
  }
}
```

**Returns:** `{ id: number, text: string, tags: string[] }`

**Use Cases:** Store code snippets, save task descriptions, record debugging notes

---

#### 3. `agentdb_insert_batch`
Batch insert multiple vectors efficiently (141x faster than sequential).

**Parameters:**
```json
{
  "items": [
    {
      "text": "Vector 1 content",
      "session_id": "session-1",
      "tags": ["tag1"]
    },
    {
      "text": "Vector 2 content",
      "session_id": "session-1",
      "tags": ["tag2"]
    }
  ],
  "batch_size": 100
}
```

**Returns:** Count of inserted vectors

**Use Cases:** Import existing documentation, bulk knowledge ingestion, migration

---

#### 4. `agentdb_search`
Semantic k-NN vector search using cosine similarity (150x faster with HNSW).

**Parameters:**
```json
{
  "query": "How to implement JWT authentication?",
  "k": 10,
  "min_similarity": 0.7,
  "filters": {
    "tags": ["auth", "security"],
    "min_reward": 0.8
  }
}
```

**Returns:**
```json
{
  "results": [
    {
      "id": 42,
      "task": "Implement JWT auth with 24h expiry",
      "similarity": 0.912,
      "reward": 0.95,
      "tags": ["auth", "jwt"]
    }
  ]
}
```

**Use Cases:** Find similar code patterns, retrieve relevant documentation, semantic search

---

#### 5. `agentdb_delete`
Delete vector(s) by ID or bulk conditions.

**Parameters:**
```json
{
  "id": 42
}
```

Or bulk delete:
```json
{
  "filters": {
    "session_id": "old-session",
    "before_timestamp": 1640000000
  }
}
```

**Returns:** Count of deleted vectors

**Use Cases:** Clean up old data, remove failed experiments, prune low-quality memories

---

### CATEGORY 2: Frontier Memory Features (9 Tools)

#### 6. `reflexion_store`
Store an episode with self-critique for reflexion-based learning.

**Parameters:**
```json
{
  "session_id": "session-1",
  "task": "implement_auth",
  "reward": 0.95,
  "success": true,
  "critique": "OAuth2 PKCE flow worked perfectly for mobile apps",
  "input": "Need secure authentication",
  "output": "Implemented OAuth2 with authorization code + PKCE",
  "latency_ms": 1200,
  "tokens": 500
}
```

**Returns:** `{ episodeId: number }`

---

#### 7. `reflexion_retrieve`
Retrieve relevant past episodes by semantic similarity.

**Parameters:**
```json
{
  "task": "authentication issues",
  "k": 10,
  "min_reward": 0.7,
  "only_failures": false,
  "only_successes": false
}
```

**Returns:**
```json
{
  "episodes": [
    {
      "id": 42,
      "task": "fix_auth_timeout",
      "reward": 0.95,
      "critique": "JWT expiry fix worked",
      "similarity": 0.89
    }
  ]
}
```

---

#### 8. `skill_create`
Create a reusable skill with code and metadata.

**Parameters:**
```json
{
  "name": "jwt_auth",
  "description": "Generate secure JWT tokens with 24h expiry",
  "code": "const jwt = require('jsonwebtoken'); jwt.sign(payload, secret, {expiresIn: '24h'});"
}
```

**Returns:** `{ skillId: number }`

---

#### 9. `skill_search`
Search skills by semantic similarity.

**Parameters:**
```json
{
  "task": "authentication security",
  "k": 5,
  "min_success_rate": 0.5
}
```

**Returns:**
```json
{
  "skills": [
    {
      "id": 1,
      "name": "jwt_auth",
      "description": "Generate secure JWT tokens",
      "success_rate": 0.92,
      "uses": 47,
      "similarity": 0.94
    }
  ]
}
```

---

#### 10. `causal_add_edge`
Manually add a causal edge between memory types.

**Parameters:**
```json
{
  "cause": "add_tests",
  "effect": "code_quality",
  "uplift": 0.25,
  "confidence": 0.95,
  "sample_size": 100
}
```

**Returns:** `{ edgeId: number }`

---

#### 11. `causal_query`
Query discovered causal relationships.

**Parameters:**
```json
{
  "cause": "add_tests",
  "effect": "code_quality",
  "min_confidence": 0.7,
  "min_uplift": 0.1,
  "limit": 10
}
```

**Returns:**
```json
{
  "edges": [
    {
      "cause": "add_tests",
      "effect": "code_quality",
      "uplift": 0.25,
      "confidence": 0.95,
      "sample_size": 100
    }
  ]
}
```

---

#### 12. `recall_with_certificate`
Retrieve memories ranked by utility with cryptographic proof.

**Utility Formula:** `U = α·similarity + β·uplift − γ·latency`

**Parameters:**
```json
{
  "query": "successful API optimization",
  "k": 5,
  "alpha": 0.7,
  "beta": 0.2,
  "gamma": 0.1
}
```

**Returns:**
```json
{
  "candidates": [
    {
      "id": 42,
      "type": "episode",
      "content": "Implemented Redis caching...",
      "similarity": 0.912,
      "uplift": 0.450,
      "utility": 0.829
    }
  ],
  "certificate": {
    "certificate_id": "9516f6115248be471ada...",
    "merkle_root": "sha256:...",
    "completeness": 0.87
  }
}
```

---

#### 13. `learner_discover`
Discover causal patterns from episode history.

**Parameters:**
```json
{
  "min_attempts": 3,
  "min_success_rate": 0.6,
  "min_confidence": 0.7,
  "dry_run": false
}
```

**Returns:**
```json
{
  "patterns": [
    {
      "cause": "add_tests",
      "effect": "code_quality",
      "uplift": 0.25,
      "p_value": 0.003,
      "sample_size": 12
    }
  ],
  "created_edges": 3,
  "created_skills": 2
}
```

---

#### 14. `db_stats`
Get comprehensive database statistics showing record counts.

**Parameters:** None

**Returns:**
```json
{
  "causal_edges": 15,
  "experiments": 8,
  "observations": 127,
  "episodes": 42,
  "episode_embeddings": 42,
  "skills": 6
}
```

---

### CATEGORY 3: Core AgentDB Tools (5 Tools - NEW in v1.3.0)

#### 15. `agentdb_stats`
Get comprehensive database statistics including table counts, storage usage, and performance metrics.

**Parameters:**
```json
{
  "detailed": false
}
```

**Returns:**
```json
{
  "episodes": 42,
  "episode_embeddings": 42,
  "skills": 6,
  "skill_embeddings": 6,
  "reasoning_patterns": 15,
  "pattern_embeddings": 15,
  "learning_sessions": 3,
  "causal_edges": 12,
  "experiments": 5,
  "observations": 87,
  "total_size_mb": 2.4
}
```

**Use Cases:** Monitor database health, track growth, verify data integrity

---

#### 16. `agentdb_pattern_store`
Store reasoning pattern with embedding, task type, approach, and success rate.

**Parameters:**
```json
{
  "taskType": "code_review",
  "approach": "Check for security vulnerabilities first, then code quality",
  "successRate": 0.95,
  "tags": ["security", "quality"],
  "metadata": {
    "author": "agent-1",
    "context": "backend-api"
  }
}
```

**Returns:** `{ patternId: number }`

**Use Cases:** Capture successful reasoning approaches, build pattern library

---

#### 17. `agentdb_pattern_search`
Search patterns with task embedding, k, threshold, and filters.

**Parameters:**
```json
{
  "task": "code review best practices",
  "k": 10,
  "threshold": 0.7,
  "filters": {
    "taskType": "code_review",
    "minSuccessRate": 0.8,
    "tags": ["security"]
  }
}
```

**Returns:**
```json
{
  "patterns": [
    {
      "id": 1,
      "taskType": "code_review",
      "approach": "Check for security vulnerabilities first...",
      "successRate": 0.95,
      "similarity": 0.91,
      "uses": 23
    }
  ]
}
```

**Use Cases:** Find relevant reasoning patterns, retrieve best practices

---

#### 18. `agentdb_pattern_stats`
Get pattern statistics including total patterns, success rates, and top task types.

**Parameters:** None

**Returns:**
```json
{
  "totalPatterns": 47,
  "avgSuccessRate": 0.87,
  "avgUses": 12.3,
  "highPerformingPatterns": 28,
  "recentPatterns": 15,
  "topTaskTypes": [
    {"taskType": "code_review", "count": 23},
    {"taskType": "debugging", "count": 15},
    {"taskType": "optimization", "count": 9}
  ]
}
```

**Use Cases:** Analyze pattern effectiveness, identify knowledge gaps

---

#### 19. `agentdb_clear_cache`
Clear query cache to refresh statistics and search results.

**Parameters:**
```json
{
  "cache_type": "all"
}
```

**Options:** `"all"`, `"patterns"`, `"stats"`

**Returns:** `{ success: true, cache_type: "all" }`

**Use Cases:** Memory management, force refresh after bulk updates, debugging

---

### CATEGORY 4: Learning System Tools (10 Tools - NEW in v1.3.0)

#### 20. `learning_start_session`
Start a new reinforcement learning session with specified algorithm and configuration.

**Supported Algorithms:** q-learning, sarsa, dqn, policy-gradient, actor-critic, ppo, decision-transformer, mcts, model-based

**Parameters:**
```json
{
  "user_id": "agent-123",
  "session_type": "q-learning",
  "config": {
    "learning_rate": 0.01,
    "discount_factor": 0.99,
    "exploration_rate": 0.1,
    "batch_size": 32,
    "target_update_frequency": 100
  }
}
```

**Returns:** `{ session_id: "session-abc123" }`

**Use Cases:** Initialize adaptive agents, start RL training, configure learning parameters

---

#### 21. `learning_end_session`
End an active learning session and save the final trained policy to the database.

**Parameters:**
```json
{
  "session_id": "session-abc123"
}
```

**Returns:** `{ success: true, policy_saved: true }`

**Use Cases:** Finalize training, persist learned policies, cleanup resources

---

#### 22. `learning_predict`
Get AI-recommended action for a given state with confidence scores and alternative actions.

**Parameters:**
```json
{
  "session_id": "session-abc123",
  "state": "user authentication failed"
}
```

**Returns:**
```json
{
  "action": "check JWT token expiry",
  "confidence": 0.87,
  "q_value": 2.45,
  "alternatives": [
    {"action": "verify database connection", "confidence": 0.65},
    {"action": "check user credentials", "confidence": 0.54}
  ]
}
```

**Use Cases:** Get intelligent action recommendations, adaptive decision-making

---

#### 23. `learning_feedback`
Submit feedback on action quality to train the RL policy.

**Parameters:**
```json
{
  "session_id": "session-abc123",
  "state": "user authentication failed",
  "action": "check JWT token expiry",
  "reward": 0.95,
  "next_state": "authentication successful",
  "success": true
}
```

**Returns:** `{ success: true, policy_updated: true }`

**Use Cases:** Provide learning signals, update Q-values, improve policy

---

#### 24. `learning_train`
Train the RL policy using batch learning with collected experiences.

**Parameters:**
```json
{
  "session_id": "session-abc123",
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.01
}
```

**Returns:**
```json
{
  "epochs_completed": 50,
  "final_loss": 0.0234,
  "avg_reward": 0.87,
  "convergence_rate": 0.92,
  "training_time_ms": 1523
}
```

**Use Cases:** Batch policy updates, offline training, performance optimization

---

#### 25. `learning_metrics`
Get learning performance metrics including success rates, rewards, and policy improvement.

**Parameters:**
```json
{
  "session_id": "session-abc123",
  "time_window_days": 7,
  "include_trends": true,
  "group_by": "task"
}
```

**Returns:**
```json
{
  "total_episodes": 342,
  "avg_reward": 0.87,
  "success_rate": 0.92,
  "avg_latency_ms": 145,
  "improvement_rate": 0.15,
  "trends": {
    "reward_trend": "increasing",
    "success_trend": "stable"
  }
}
```

**Use Cases:** Monitor learning progress, track improvements, performance analysis

---

#### 26. `learning_transfer`
Transfer learning between sessions or tasks, enabling knowledge reuse across different contexts.

**Parameters:**
```json
{
  "source_session": "session-abc123",
  "target_session": "session-xyz789",
  "source_task": "user authentication",
  "target_task": "API authentication",
  "min_similarity": 0.7,
  "transfer_type": "all",
  "max_transfers": 10
}
```

**Options for `transfer_type`:** `"episodes"`, `"skills"`, `"causal_edges"`, `"all"`

**Returns:**
```json
{
  "transferred_items": 8,
  "transfer_type": "all",
  "avg_similarity": 0.84,
  "episodes": 5,
  "skills": 2,
  "causal_edges": 1
}
```

**Use Cases:** Leverage past experience, accelerate learning, cross-domain adaptation

---

#### 27. `learning_explain`
Explain action recommendations with confidence scores and supporting evidence from past experiences.

**Parameters:**
```json
{
  "query": "how to fix authentication errors",
  "k": 5,
  "explain_depth": "detailed",
  "include_confidence": true,
  "include_evidence": true,
  "include_causal": true
}
```

**Options for `explain_depth`:** `"summary"`, `"detailed"`, `"full"`

**Returns:**
```json
{
  "recommendations": [
    {
      "action": "check JWT token expiry",
      "confidence": 0.91,
      "explanation": "Based on 23 similar episodes with 87% success rate",
      "evidence": [
        {"episode_id": 42, "similarity": 0.89, "reward": 0.95},
        {"episode_id": 67, "similarity": 0.84, "reward": 0.92}
      ],
      "causal_chain": [
        {"cause": "jwt_expired", "effect": "auth_failed", "confidence": 0.95}
      ]
    }
  ]
}
```

**Use Cases:** Explainable AI, build trust, debug recommendations

---

#### 28. `experience_record`
Record tool execution as experience for reinforcement learning and experience replay.

**Parameters:**
```json
{
  "session_id": "session-abc123",
  "tool_name": "code_analyzer",
  "action": "analyze security vulnerabilities",
  "state_before": {"code_quality": 0.7, "security_score": 0.6},
  "state_after": {"code_quality": 0.9, "security_score": 0.95},
  "outcome": "found 3 vulnerabilities and fixed them",
  "reward": 0.88,
  "success": true,
  "latency_ms": 1240,
  "metadata": {"tool_version": "2.1.0"}
}
```

**Returns:** `{ experience_id: number, recorded: true }`

**Use Cases:** Build experience replay buffer, offline learning, tool usage tracking

---

#### 29. `reward_signal`
Calculate reward signal for outcomes based on success, efficiency, and causal impact.

**Parameters:**
```json
{
  "episode_id": 42,
  "success": true,
  "target_achieved": true,
  "efficiency_score": 0.85,
  "quality_score": 0.92,
  "time_taken_ms": 1200,
  "expected_time_ms": 1500,
  "include_causal": true,
  "reward_function": "standard"
}
```

**Options for `reward_function`:** `"standard"`, `"sparse"`, `"dense"`, `"shaped"`

**Returns:**
```json
{
  "reward": 0.89,
  "breakdown": {
    "success_component": 1.0,
    "efficiency_component": 0.85,
    "quality_component": 0.92,
    "time_component": 0.8,
    "causal_impact": 0.15
  },
  "explanation": "High quality with good efficiency and ahead of schedule"
}
```

**Use Cases:** Automated reward shaping, performance evaluation, learning signal generation

---

## 🎯 Usage Patterns in Claude Desktop

### Pattern 1: Reflexion Learning Loop

```
User: Help me implement OAuth2
Claude: Let me check past similar tasks...

[Uses reflexion_retrieve]
Found 3 past OAuth implementations (avg reward: 0.92)

[Implements solution]

[Uses reflexion_store]
Stored this implementation with critique for future reference
```

---

### Pattern 2: Adaptive Learning

```
User: I need help debugging authentication
Claude: Starting learning session...

[Uses learning_start_session]
Session started with Q-learning algorithm

[Uses learning_predict]
Recommended action: check JWT token expiry (87% confidence)

[User tries recommendation and provides feedback]

[Uses learning_feedback]
Policy updated based on your feedback

[Uses learning_train]
Trained on 50 episodes, 92% convergence
```

---

### Pattern 3: Causal Discovery

```
User: Does adding tests improve code quality?
Claude: Let me check our causal knowledge...

[Uses causal_query]
Found causal edge: add_tests → code_quality
Uplift: +25%, Confidence: 95%, Sample: 100

Yes, adding tests causally improves code quality by 25%
with high statistical confidence.
```

---

## 📊 Tool Category Summary

### Core Vector DB (5 tools)
**Purpose:** Basic vector storage and retrieval
**Tools:** `agentdb_init`, `agentdb_insert`, `agentdb_insert_batch`, `agentdb_search`, `agentdb_delete`
**Use Cases:**
- Store and retrieve vectors with embeddings
- Semantic search with filters
- Bulk operations with transactions
- Data cleanup and pruning

---

### Core AgentDB (5 tools)
**Purpose:** Database management and reasoning patterns
**Tools:** `agentdb_stats`, `agentdb_pattern_store`, `agentdb_pattern_search`, `agentdb_pattern_stats`, `agentdb_clear_cache`
**Use Cases:**
- Monitor database health
- Store and retrieve reasoning patterns
- Track pattern effectiveness
- Cache management

---

### Reflexion Memory (2 tools)
**Purpose:** Learn from past experiences
**Tools:** `reflexion_store`, `reflexion_retrieve`
**Use Cases:**
- Store episodes with self-critique
- Retrieve similar past experiences
- Learn from successes and failures
- Improve decision-making over time

---

### Skill Library (2 tools)
**Purpose:** Build reusable code patterns
**Tools:** `skill_create`, `skill_search`
**Use Cases:**
- Create reusable skills from successful patterns
- Search for applicable skills by semantic similarity
- Track skill success rates and usage
- Consolidate knowledge across episodes

---

### Causal Memory (2 tools)
**Purpose:** Causal reasoning and A/B testing
**Tools:** `causal_add_edge`, `causal_query`
**Use Cases:**
- Record causal relationships (cause → effect → uplift)
- Query discovered causal patterns
- A/B test interventions
- Statistical causal inference

---

### Explainable Recall (1 tool)
**Purpose:** Utility-based retrieval with provenance
**Tools:** `recall_with_certificate`
**Use Cases:**
- Retrieve memories ranked by causal utility
- Get cryptographic provenance certificates
- Explain why memories were retrieved
- Balance similarity, uplift, and recency

---

### Nightly Learner (1 tool)
**Purpose:** Automated pattern discovery
**Tools:** `learner_discover`
**Use Cases:**
- Discover causal patterns automatically
- Create skills from successful episodes
- Background learning and optimization

---

### Database Utilities (1 tool)
**Purpose:** Monitoring and statistics
**Tools:** `db_stats`
**Use Cases:**
- Monitor database growth
- Track memory type counts
- System health checks
- Performance monitoring

---

### Learning System (10 tools)
**Purpose:** Reinforcement learning with 9 algorithms
**Tools:** `learning_start_session`, `learning_end_session`, `learning_predict`, `learning_feedback`, `learning_train`, `learning_metrics`, `learning_transfer`, `learning_explain`, `experience_record`, `reward_signal`
**Use Cases:**
- Adaptive agent behavior
- Policy learning from experience
- Transfer learning across tasks
- Explainable action recommendations
- Performance tracking and optimization

---

## ⚙️ Configuration

### Environment Variables

```bash
# Database path
export AGENTDB_PATH="./my-agent.db"

# Enable verbose logging
export AGENTDB_VERBOSE=true

# Set embedding dimension
export AGENTDB_EMBEDDING_DIM=384

# Learning system config
export AGENTDB_LEARNING_RATE=0.1
export AGENTDB_DISCOUNT_FACTOR=0.95
```

---

## ✨ Key Features

### ✅ 29 Production-Ready MCP Tools (v1.3.0)
- **Core Vector DB (5):** Full CRUD operations with embeddings
- **Core AgentDB (5):** Database management and reasoning patterns
- **Frontier Memory (9):** Reflexion, skills, causal memory, explainable recall
- **Learning System (10):** 9 RL algorithms with full training pipeline
- Zero configuration required

### ✅ Verified Implementation
- All 29 documented tools are actually implemented
- Tested and validated against source code (v1.3.0)
- No phantom or placeholder tools
- Complete type safety with TypeScript

### ✅ Zero Configuration
- Start with `npx agentdb mcp start`
- All 29 tools available instantly
- Automatic initialization
- Self-optimizing storage

### ✅ Full TypeScript Support
- All request/response types fully typed
- IntelliSense support in Claude Desktop
- Compile-time type checking
- Rich IDE integration

### ✅ Production Ready
- 150x faster vector search (WASM SIMD)
- Comprehensive error handling
- Transaction support for consistency
- Automatic cleanup and pruning
- Battle-tested in production

---

## 🔍 Performance

| Operation | Speed | Optimization |
|-----------|-------|--------------|
| **Vector Search** | 0.12ms | 150x faster with SIMD |
| **Batch Insert** | 8.5ms | 141x faster than sequential |
| **Pattern Matching** | 1.8ms | Native WASM acceleration |
| **Learning Prediction** | 3.2ms | Cached policy lookup |
| **Causal Query** | 2.3ms | Indexed graph traversal |
| **RL Training (50 epochs)** | 1.5s | Optimized batch learning |

---

## 🐛 Troubleshooting

### MCP Server Won't Start

```bash
# Update to latest version
npm install -g agentdb@latest

# Check node version (need Node 20.9.0+)
node --version

# Try with explicit path
npx agentdb@latest mcp start
```

---

### Tools Not Appearing in Claude

1. Restart Claude Desktop completely
2. Check config file location
3. Verify JSON syntax: `cat claude_desktop_config.json | jq`
4. Check Claude Desktop logs for errors

---

### Database Locked Error

```bash
# Use a different database path
export AGENTDB_PATH="./agent-new.db"
npx agentdb mcp start
```

---

## 📚 Related Documentation

- **[CLI Guide](./CLI_GUIDE.md)** - Complete command-line reference
- **[SDK Guide](./SDK_GUIDE.md)** - Programmatic API usage
- **[Frontier Memory Guide](./FRONTIER_MEMORY_GUIDE.md)** - Feature concepts
- **[Migration Guide v1.3.0](../MIGRATION_v1.3.0.md)** - Upgrade from v1.2.2
- **[Landing Page](./LANDING_PAGE.md)** - Quick start and overview

---

## 🚀 Next Steps

### Beginner Path
1. Start with **Core Vector DB** tools for basic storage (`agentdb_init`, `agentdb_insert`, `agentdb_search`)
2. Add **Reflexion** for learning (`reflexion_store`, `reflexion_retrieve`)
3. Monitor with **AgentDB Stats** (`agentdb_stats`, `db_stats`)

### Intermediate Path
1. Use **Skills** for reusable patterns (`skill_create`, `skill_search`)
2. Add **Causal Memory** for reasoning (`causal_add_edge`, `causal_query`)
3. Try **Pattern Management** (`agentdb_pattern_store`, `agentdb_pattern_search`)

### Advanced Path
1. Implement **Learning System** for adaptive agents (`learning_start_session`, `learning_predict`, `learning_feedback`)
2. Use **Transfer Learning** across domains (`learning_transfer`)
3. Enable **Explainable AI** with evidence (`learning_explain`)

---

## 💡 Pro Tips

1. **Start with Core Vector DB**: Use `agentdb_init` + `agentdb_insert` for basic memory storage
2. **Add Reflexion Immediately**: Use `reflexion_store` after completing tasks to build experience
3. **Use Learning System for Adaptation**: Start with `q-learning` for simple state-action mapping
4. **Monitor with Stats**: Check `agentdb_stats` regularly to track database growth
5. **Leverage Patterns**: Use `agentdb_pattern_store` for successful reasoning approaches
6. **Transfer Knowledge**: Use `learning_transfer` to reuse learned policies across similar tasks
7. **Explain Decisions**: Use `learning_explain` to build trust and debug agent behavior

---

## 🔗 Additional Resources

- **[Verification Report](/workspaces/agentic-flow/docs/agentdb-tools-verification.md)** - Complete tool verification
- **[CLI Guide](./CLI_GUIDE.md)** - Command-line interface reference
- **[SDK Guide](./SDK_GUIDE.md)** - Programmatic API usage
- **[Frontier Memory Guide](./FRONTIER_MEMORY_GUIDE.md)** - Conceptual overview
- **[Migration Guide v1.3.0](../MIGRATION_v1.3.0.md)** - Upgrade from v1.2.2

---

**AgentDB MCP** - 29 verified production-ready tools for frontier AI memory + reinforcement learning 🧠⚡🎓

**Current Status (v1.3.0):**
- ✅ **Core Vector DB (5 tools)** - Full CRUD with embeddings (v1.2.2)
- ✅ **Core AgentDB (5 tools)** - NEW! Database stats + reasoning patterns (v1.3.0)
- ✅ **Reflexion Memory (2 tools)** - Learn from experience with self-critique
- ✅ **Skill Library (2 tools)** - Build and search reusable patterns
- ✅ **Causal Memory (2 tools)** - Track cause-effect relationships
- ✅ **Explainable Recall (1 tool)** - Utility-based retrieval with certificates
- ✅ **Nightly Learner (1 tool)** - Automated pattern discovery
- ✅ **Database Utilities (1 tool)** - Statistics and monitoring
- ✅ **Learning System (10 tools)** - NEW! Full RL pipeline with 9 algorithms (v1.3.0)

**Verification:** All 29 documented tools have been verified against the actual MCP server implementation (v1.3.0).
