# AgentDB MCP Integration - Claude Desktop Guide

> **15 Production-Ready Tools** · **Zero Configuration** · **Instant Setup** · **Full Memory Features**

AgentDB provides complete Model Context Protocol (MCP) integration, exposing all frontier memory features as tools for Claude Desktop and other AI assistants.

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

## 📦 Available MCP Tools (15 Total)

### 🔄 Reflexion Memory (3 tools)
**Learn from experience with self-critique**

#### 1. `reflexion_store`
Store complete task episodes with self-generated critiques.

```json
{
  "sessionId": "session-1",
  "task": "implement_auth",
  "reward": 0.95,
  "success": true,
  "critique": "OAuth2 PKCE flow worked perfectly for mobile apps",
  "input": "Need secure authentication",
  "output": "Implemented OAuth2 with authorization code + PKCE",
  "latencyMs": 1200,
  "tokens": 500
}
```

**Returns:** `{ episodeId: number }`

---

#### 2. `reflexion_retrieve`
Retrieve relevant past episodes by semantic similarity.

```json
{
  "task": "authentication issues",
  "k": 10,
  "minReward": 0.7,
  "onlyFailures": false,
  "onlySuccesses": false
}
```

**Returns:** `{ episodes: Episode[] }` with similarity scores

---

#### 3. `reflexion_critique_summary`
Get aggregated critique lessons from past episodes.

```json
{
  "task": "authentication",
  "onlyFailures": true
}
```

**Returns:** `{ critiques: string[], successRate: number, avgReward: number }`

---

### 🎓 Skill Library (3 tools)
**Auto-consolidate successful patterns**

#### 4. `skill_create`
Create a reusable skill with code and metadata.

```json
{
  "name": "jwt_auth",
  "description": "Generate secure JWT tokens with 24h expiry",
  "code": "const jwt = require('jsonwebtoken'); jwt.sign(payload, secret, {expiresIn: '24h'});"
}
```

**Returns:** `{ skillId: number }`

---

#### 5. `skill_search`
Search skills by semantic similarity.

```json
{
  "query": "authentication security",
  "k": 5,
  "minSuccessRate": 0.5
}
```

**Returns:** `{ skills: Skill[] }` with success rates and usage stats

---

#### 6. `skill_consolidate`
Auto-create skills from successful episode patterns.

```json
{
  "minAttempts": 3,
  "minReward": 0.7,
  "timeWindowDays": 7
}
```

**Returns:** `{ createdSkills: number, skills: Skill[] }`

---

### 🔗 Causal Memory (4 tools)
**Intervention-based causality with p(y|do(x))**

#### 7. `causal_add_edge`
Manually add a causal edge between memory types.

```json
{
  "cause": "add_tests",
  "effect": "code_quality",
  "uplift": 0.25,
  "confidence": 0.95,
  "sampleSize": 100
}
```

**Returns:** `{ edgeId: number }`

---

#### 8. `causal_query`
Query discovered causal relationships.

```json
{
  "cause": "add_tests",
  "effect": "code_quality",
  "minConfidence": 0.7,
  "minUplift": 0.1,
  "limit": 10
}
```

**Returns:** `{ edges: CausalEdge[] }` with uplift and confidence

---

#### 9. `causal_experiment_create`
Create A/B experiment to measure causal effects.

```json
{
  "name": "test_coverage_quality",
  "cause": "add_tests",
  "effect": "code_quality"
}
```

**Returns:** `{ experimentId: number }`

---

#### 10. `causal_experiment_calculate`
Calculate uplift with statistical significance.

```json
{
  "experimentId": 1
}
```

**Returns:**
```json
{
  "treatmentMean": 0.877,
  "controlMean": 0.625,
  "uplift": 0.252,
  "pValue": 0.003,
  "confidenceInterval": [0.210, 0.293],
  "significant": true
}
```

---

### 🎯 Causal Recall (1 tool)
**Utility-based retrieval with provenance**

#### 11. `recall_with_certificate`
Retrieve memories ranked by utility with cryptographic proof.

**Utility Formula:** `U = α·similarity + β·uplift − γ·latency`

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
    "certificateId": "9516f6115248be471ada...",
    "merkleRoot": "sha256:...",
    "completeness": 0.87,
    "parameters": { "k": 5, "alpha": 0.7, "beta": 0.2, "gamma": 0.1 }
  },
  "latencyMs": 12
}
```

---

### 🌙 Nightly Learner (2 tools)
**Automated pattern discovery**

#### 12. `learner_discover`
Discover causal patterns from episode history.

```json
{
  "minAttempts": 3,
  "minSuccessRate": 0.6,
  "minConfidence": 0.7,
  "dryRun": false
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
      "pValue": 0.003,
      "sampleSize": 12,
      "confidence": 0.95
    }
  ],
  "createdEdges": 3,
  "createdSkills": 2
}
```

---

#### 13. `learner_prune`
Remove low-quality causal edges.

```json
{
  "minConfidence": 0.5,
  "minUplift": 0.05,
  "maxAgeDays": 90
}
```

**Returns:** `{ prunedCount: number }`

---

### 📊 Database (2 tools)

#### 14. `db_stats`
Get comprehensive database statistics.

```json
{}
```

**Returns:**
```json
{
  "episodes": 145,
  "skills": 23,
  "causalEdges": 12,
  "experiments": 5,
  "observations": 47,
  "certificates": 89
}
```

---

#### 15. `db_export`
Export all data to JSON format.

```json
{
  "format": "json",
  "includeBinary": false
}
```

**Returns:** Complete database dump in JSON

---

## 🎯 Usage Patterns in Claude Desktop

### Pattern 1: Learning from Experience

```
User: Help me implement OAuth2 authentication
Claude: I'll search for similar past episodes first...

[Uses reflexion_retrieve]
Found 3 similar episodes with high success rates.
Episode #42 used OAuth2 PKCE flow with 95% success.

[Uses skill_search]
Found skill "jwt_auth" that handles token generation.

[Implements solution]

[Uses reflexion_store]
Stored this episode for future learning.
```

---

### Pattern 2: Causal Experimentation

```
User: Does adding tests actually improve code quality?
Claude: Let me check our causal knowledge...

[Uses causal_query]
Found existing causal edge: add_tests → code_quality
Uplift: 0.25, Confidence: 95%, Sample: 100

Based on past experiments, adding tests causally improves
code quality by 25% with high statistical confidence.
```

---

### Pattern 3: Smart Retrieval

```
User: Show me the most effective API optimization approaches
Claude: I'll retrieve based on effectiveness, not just similarity...

[Uses recall_with_certificate with α=0.5, β=0.4, γ=0.1]

Top 3 optimizations by utility:
1. Redis caching (uplift: 0.45, similarity: 0.91)
2. Database indexing (uplift: 0.38, similarity: 0.89)
3. Connection pooling (uplift: 0.32, similarity: 0.86)

Certificate ID: 9516f6115248be471...
Completeness: 87%
```

---

### Pattern 4: Automated Discovery

```
User: What patterns have you learned overnight?
Claude: Running pattern discovery...

[Uses learner_discover]
Discovered 5 new causal patterns:
• code_review → bug_rate (-30% bugs, p=0.001)
• add_caching → response_time (-120ms, p=0.008)
• refactor_types → maintainability (+18%, p=0.03)

Created 3 causal edges and 2 new skills automatically.
```

---

## ⚙️ Configuration Options

### Environment Variables

```bash
# Database path
export AGENTDB_PATH="./my-agent.db"

# Enable verbose logging
export AGENTDB_VERBOSE=true

# Set embedding dimension (default: 384)
export AGENTDB_EMBEDDING_DIM=384
```

### Advanced MCP Configuration

```json
{
  "mcpServers": {
    "agentdb": {
      "command": "npx",
      "args": ["agentdb@latest", "mcp", "start"],
      "env": {
        "AGENTDB_PATH": "/path/to/agent.db",
        "AGENTDB_VERBOSE": "true"
      }
    }
  }
}
```

---

## 🔍 Tool Categories Summary

| Category | Tools | Purpose |
|----------|-------|---------|
| **Reflexion** | 3 | Learn from successes and failures |
| **Skills** | 3 | Build reusable code library |
| **Causal** | 4 | A/B testing and causality |
| **Recall** | 1 | Smart retrieval with utility |
| **Learner** | 2 | Automated pattern discovery |
| **Database** | 2 | Stats and export |
| **TOTAL** | **15** | **Complete memory system** |

---

## ✨ Key Features

### ✅ Zero Configuration
- All tools available immediately after `npx agentdb mcp start`
- No initialization required for basic usage
- Temporary in-memory database until you specify a path

### ✅ Full TypeScript Support
- All request/response types fully typed
- IntelliSense support in Claude Desktop
- Compile-time type checking

### ✅ Explainable Results
- Every recall includes provenance certificate
- Causal edges include statistical confidence
- Utility scores show why memories were selected

### ✅ Production Ready
- 150x faster vector search (WASM SIMD)
- Comprehensive error handling
- Transaction support for consistency
- Automatic cleanup and pruning

---

## 🐛 Troubleshooting

### MCP Server Won't Start

**Problem:** `npx agentdb mcp start` shows error

**Solution:**
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

**Problem:** MCP tools not showing up

**Solution:**
1. Restart Claude Desktop completely
2. Check config file location (see Quick Setup)
3. Verify JSON syntax with: `cat claude_desktop_config.json | jq`
4. Check Claude Desktop logs for errors

---

### Database Locked Error

**Problem:** "database is locked" error

**Solution:**
```bash
# Close any other processes using the database
# Or use a different database path
export AGENTDB_PATH="./agent-${Date.now()}.db"
```

---

## 📚 Related Documentation

- **[CLI Guide](./CLI_GUIDE.md)** - Complete command-line reference
- **[SDK Guide](./SDK_GUIDE.md)** - Programmatic API usage
- **[Frontier Memory Guide](./FRONTIER_MEMORY_GUIDE.md)** - Feature concepts
- **[Test Report](./CLI_TEST_REPORT.md)** - Validation results

---

## 🚀 Next Steps

1. **Start Simple**: Begin with `reflexion_store` and `reflexion_retrieve`
2. **Build Skills**: Use `skill_create` and `skill_search` for reusable patterns
3. **Run Experiments**: Test causal hypotheses with A/B experiments
4. **Automate**: Use `learner_discover` for overnight pattern discovery
5. **Optimize**: Tune α, β, γ in `recall_with_certificate` for your use case

---

## 💡 Pro Tips

1. **Use Dry-Run First**: Always run `learner_discover` with `dryRun: true` first
2. **Check Certificates**: Verify `completeness > 0.8` for comprehensive retrieval
3. **Monitor Stats**: Regularly check `db_stats` to understand your memory system
4. **Prune Regularly**: Use `learner_prune` weekly to maintain quality
5. **Store Critiques**: Always include detailed critiques in `reflexion_store`

---

**AgentDB MCP** - Production-ready frontier memory for Claude Desktop 🧠⚡
