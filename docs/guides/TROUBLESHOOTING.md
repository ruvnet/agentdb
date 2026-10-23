# AgentDB v2 Troubleshooting Guide

Common issues and solutions for AgentDB v2

---

## Quick Diagnosis

Run these commands to diagnose issues:

```bash
# 1. Check installed version
npm list agentdb

# 2. Detect available backends
npx agentdb detect

# 3. Run benchmarks
npx agentdb benchmark --backend=auto

# 4. Check database stats
npx agentdb stats ./your-db.db
```

---

## Installation Issues

### Error: Cannot find module 'agentdb'

**Cause**: Package not installed or incorrect version

**Solution**:
```bash
# Install latest version
npm install agentdb@2.0.0-alpha.1

# Or add to package.json
npm install --save agentdb@2.0.0-alpha.1

# Verify installation
npm list agentdb
```

### Error: Native module build failed (better-sqlite3)

**Cause**: Missing build tools or incompatible platform

**Solution 1 - Install build tools**:
```bash
# Windows
npm install --global windows-build-tools

# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential python3
```

**Solution 2 - Use different backend**:
```bash
# Use RuVector instead (no native build required)
npm install @ruvector/core

# Or fallback to SQLite (built-in)
```

### Error: @ruvector/core not found

**Cause**: Optional dependency not installed

**Solution**:
```bash
# Install RuVector backend
npm install @ruvector/core

# Or explicitly install as dependency (not optional)
npm install --save @ruvector/core
```

---

## Backend Detection Issues

### No backend detected

**Error**:
```
Error: No vector backend available.
Install one of:
  - npm install @ruvector/core (recommended)
  - npm install hnswlib-node (fallback)
```

**Diagnosis**:
```bash
npx agentdb detect
```

**Solution**:
```bash
# Install recommended backend
npm install @ruvector/core

# Verify detection
npx agentdb detect
# Should show: Backend: ruvector
```

### RuVector not detected but installed

**Cause**: Import resolution issue or version mismatch

**Diagnosis**:
```bash
# Check if package is actually installed
npm list @ruvector/core

# Try direct import
node -e "import('@ruvector/core').then(console.log)"
```

**Solution**:
```bash
# Reinstall package
npm uninstall @ruvector/core
npm install @ruvector/core

# Clear npm cache
npm cache clean --force
npm install
```

### Backend detection shows WASM instead of native

**Detection output**:
```
Backend:     ruvector
Native:      ❌ (using WASM)
```

**Cause**: Platform-specific native build not available

**Is this a problem?**: No! WASM is still 130x faster than SQLite.

**To get native** (optional, for maximum performance):
```bash
# Install platform-specific package
# Linux x64
npm install @ruvector/core-linux-x64

# macOS ARM64
npm install @ruvector/core-darwin-arm64

# Windows x64
npm install @ruvector/core-win32-x64
```

---

## Runtime Errors

### Error: Database locked

**Cause**: Multiple processes accessing SQLite database

**Solution 1 - Enable WAL mode**:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'better-sqlite3',
  wal: true  // Write-Ahead Logging for better concurrency
});
```

**Solution 2 - Use RuVector** (no locking issues):
```bash
npm install @ruvector/core
```

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector'  // No lock contention
});
```

### Error: Out of memory

**Cause**: Too many vectors loaded, insufficient memory

**Diagnosis**:
```typescript
const stats = await db.getStats();
console.log(`Memory usage: ${stats.memoryUsage / 1024 / 1024} MB`);
console.log(`Vector count: ${stats.count}`);
```

**Solution 1 - Use RuVector with compression**:
```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  compression: true  // 4-8x memory reduction
});
```

**Solution 2 - Limit max elements**:
```typescript
const db = await createVectorDB({
  maxElements: 10000  // Prevent unbounded growth
});
```

**Solution 3 - Use pagination**:
```typescript
// Instead of loading all at once
const page1 = await db.search({ query: 'test', k: 100, offset: 0 });
const page2 = await db.search({ query: 'test', k: 100, offset: 100 });
```

### Error: Embedding dimension mismatch

**Error**:
```
Error: Expected dimension 384, got 768
```

**Cause**: Vectors have different dimensions than configured

**Solution**:
```typescript
// Check your embedding model's output dimension
const embedding = await embedder.embed('test');
console.log(`Embedding dimension: ${embedding.length}`);

// Match database configuration
const db = await createVectorDB({
  dimension: embedding.length  // Use actual dimension
});
```

### Error: Invalid distance metric

**Error**:
```
Error: Unsupported metric: euclidean
```

**Cause**: Using incorrect metric name

**Solution**:
```typescript
// Valid metrics: 'cosine', 'l2', 'ip'
const db = await createVectorDB({
  metric: 'cosine'  // Not 'euclidean'
});

// Metric mapping:
// - 'cosine': Cosine similarity
// - 'l2': Euclidean distance
// - 'ip': Inner product
```

---

## Performance Issues

### Search is slower than expected

**Expected performance**:
- RuVector native: 50-100µs
- RuVector WASM: 80-150µs
- better-sqlite3: 3-5ms
- SQLite: 10-15ms

**Diagnosis**:
```bash
# Run benchmark on your data
npx agentdb benchmark --backend=ruvector --vectors=10000
```

**Solution 1 - Check backend**:
```typescript
const stats = await db.getStats();
console.log(`Using backend: ${stats.backend}`);
// If not 'ruvector', switch backends
```

**Solution 2 - Tune HNSW parameters**:
```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  efSearch: 50  // Lower = faster (default: 100)
});

// Or per-query
const results = await db.search({
  query: 'test',
  k: 10,
  efSearch: 50
});
```

**Solution 3 - Use batch operations**:
```typescript
// ❌ Slow: Individual operations
for (const item of items) {
  await db.insert(item);
}

// ✅ Fast: Batch operation (141x faster)
await db.insertBatch(items);
```

### High memory usage

**Diagnosis**:
```typescript
const stats = await db.getStats();
const memoryMB = stats.memoryUsage / 1024 / 1024;
console.log(`Memory: ${memoryMB.toFixed(1)} MB`);
console.log(`Vectors: ${stats.count}`);
console.log(`Per vector: ${(memoryMB / stats.count * 1024).toFixed(2)} KB`);
```

**Expected memory per vector**:
- RuVector: ~170 bytes (384 dims)
- SQLite: ~1.5 KB (384 dims)

**Solution - Enable compression** (RuVector only):
```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  compression: true  // 4-8x reduction
});
```

### Slow database initialization

**Cause**: Large existing database being loaded

**Solution 1 - Use lazy loading**:
```typescript
const db = await createVectorDB({
  path: './large-db.db',
  lazyLoad: true  // Load index on first query
});
```

**Solution 2 - Monitor loading**:
```typescript
const db = await createVectorDB({
  path: './large-db.db',
  onProgress: (loaded, total) => {
    console.log(`Loading: ${loaded}/${total} vectors`);
  }
});
```

---

## GNN Learning Issues

### Error: GNN not available

**Error**:
```
Error: GNN learning requires @ruvector/gnn
Install with: npm install @ruvector/gnn
```

**Solution**:
```bash
# Install GNN package
npm install @ruvector/gnn

# Verify installation
npx agentdb detect
# Should show: GNN: ✅
```

### Error: Not enough training samples

**Error**:
```
Error: Minimum 10 samples required for training (found: 5)
```

**Solution**:
```typescript
// Check current sample count
const stats = await db.getGNNStats();
console.log(`Samples: ${stats.sampleCount}`);

// Collect more feedback
await db.learn({
  query: 'test query',
  results: searchResults,
  feedback: 'success'
});

// Try training again
if (stats.sampleCount >= 10) {
  await db.trainGNN({ epochs: 50 });
}
```

### GNN training loss not decreasing

**Diagnosis**:
```typescript
const result = await db.trainGNN({
  epochs: 100,
  onEpoch: (epoch, loss) => {
    console.log(`Epoch ${epoch}: Loss = ${loss}`);
  }
});
```

**Solution 1 - Lower learning rate**:
```typescript
const db = await createVectorDB({
  enableGNN: true,
  gnn: {
    inputDim: 384,
    outputDim: 384,
    learningRate: 0.0001  // 10x lower
  }
});
```

**Solution 2 - More epochs**:
```typescript
await db.trainGNN({ epochs: 500 });  // Increase iterations
```

**Solution 3 - Check data quality**:
```typescript
// Ensure feedback is meaningful
await db.learn({
  query: 'specific query',
  results: actuallyRelevantResults,  // Not random
  feedback: 'success'
});
```

---

## Database Corruption

### Error: Database file is corrupted

**Diagnosis**:
```bash
# Try to open and check
npx agentdb stats ./corrupted-db.db
```

**Solution 1 - Recover from backup**:
```bash
# If you have backups
cp ./backup/agent-memory.db ./agent-memory.db
```

**Solution 2 - Export/reimport** (if partially readable):
```bash
# Export what's recoverable
npx agentdb export ./corrupted-db.db --output=vectors.json --ignore-errors

# Create new database
npx agentdb import ./new-db.db --input=vectors.json
```

**Solution 3 - Start fresh** (last resort):
```bash
# Backup old database
mv ./agent-memory.db ./agent-memory.db.corrupted

# Create new database
npx agentdb init ./agent-memory.db
```

### Database file size keeps growing

**Cause**: Not vacuuming after deletions (SQLite only)

**Solution**:
```typescript
// For SQLite backends
await db.vacuum();  // Reclaim space after deletions

// Or enable auto-vacuum
const db = await createVectorDB({
  backend: 'sqlite',
  autoVacuum: true
});
```

**RuVector note**: RuVector automatically manages space, no vacuum needed.

---

## MCP Integration Issues

### MCP server not starting

**Error**:
```
Failed to start MCP server: agentdb
```

**Diagnosis**:
```bash
# Test MCP server manually
npx agentdb@2.0.0-alpha.1 mcp start
```

**Solution 1 - Check Node.js version**:
```bash
node --version  # Should be >= 20.9.0

# Update Node.js if needed
nvm install 24.22.1
nvm use 24.22.1
```

**Solution 2 - Reinstall package**:
```bash
npm uninstall -g agentdb
npm install -g agentdb@2.0.0-alpha.1
```

**Solution 3 - Check Claude Desktop config**:
```json
{
  "mcpServers": {
    "agentdb": {
      "command": "npx",
      "args": ["agentdb@2.0.0-alpha.1", "mcp", "start"]
    }
  }
}
```

### MCP tools not appearing in Claude Desktop

**Diagnosis**:
```bash
# Check MCP server logs
tail -f ~/.config/claude/logs/mcp-server-agentdb.log
```

**Solution 1 - Restart Claude Desktop**:
1. Quit Claude Desktop completely
2. Restart application
3. Tools should appear in ~30 seconds

**Solution 2 - Verify MCP server config**:
```bash
# Check config file
cat ~/.config/claude/claude_desktop_config.json

# Validate JSON syntax
npx ajv-cli validate -s claude-mcp-schema.json -d claude_desktop_config.json
```

---

## Platform-Specific Issues

### Windows: npm ERR! code ELIFECYCLE

**Cause**: Build tools missing or permission issues

**Solution**:
```bash
# Run as administrator
npm install --global windows-build-tools

# Or use WSL2 (recommended for development)
wsl --install
```

### macOS: Code signing issues

**Error**:
```
"@ruvector/core" cannot be opened because the developer cannot be verified
```

**Solution**:
```bash
# Allow unsigned binaries (macOS)
xattr -d com.apple.quarantine node_modules/@ruvector/core/bin/*

# Or disable Gatekeeper temporarily
sudo spctl --master-disable
```

### Linux: Permission denied

**Error**:
```
EACCES: permission denied, open '/root/.agentdb/cache.db'
```

**Solution**:
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/.agentdb

# Or use different path
const db = await createVectorDB({
  path: './data/agent-memory.db'  // User-writable location
});
```

---

## Debugging Tips

### Enable debug logging

```typescript
// Set debug environment variable
process.env.AGENTDB_DEBUG = 'true';

const db = await createVectorDB({
  path: './agent-memory.db',
  logLevel: 'debug'  // 'error' | 'warn' | 'info' | 'debug'
});

// Or via CLI
AGENTDB_DEBUG=true npx agentdb stats ./agent-memory.db
```

### Collect diagnostic info

```bash
# Create diagnostic report
npx agentdb diagnose ./agent-memory.db --output=diagnostic.json

# Report includes:
# - AgentDB version
# - Node.js version
# - Backend detection results
# - Database stats
# - Recent errors
# - Performance metrics
```

### Test backend directly

```typescript
import { detectBackends, createBackend } from 'agentdb/backends';

// Detect backends
const detection = await detectBackends();
console.log('Detection:', detection);

// Test backend directly
const backend = await createBackend('ruvector', {
  dimension: 384,
  metric: 'cosine'
});

// Insert test vector
backend.insert('test-1', new Float32Array(384).fill(0.1));

// Search
const results = backend.search(new Float32Array(384).fill(0.1), 5);
console.log('Results:', results);
```

---

## Getting Help

### Before filing an issue

1. **Check version**: `npm list agentdb`
2. **Run detection**: `npx agentdb detect`
3. **Create diagnostic**: `npx agentdb diagnose ./your-db.db`
4. **Test minimal example**: Use code from [examples/](../examples/)

### Filing an issue

Include:
- AgentDB version
- Node.js version (`node --version`)
- Platform (OS, arch)
- Backend detection output (`npx agentdb detect`)
- Error message and stack trace
- Minimal reproducible example

**GitHub Issues**: https://github.com/ruvnet/agentic-flow/issues

### Community support

- **Discussions**: https://github.com/ruvnet/agentic-flow/discussions
- **Discord**: [Join server](https://discord.gg/ruvnet) (if available)

---

## Common Error Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `Cannot find module 'agentdb'` | Not installed | `npm install agentdb@2.0.0-alpha.1` |
| `No vector backend available` | No backend installed | `npm install @ruvector/core` |
| `Database locked` | SQLite concurrency | Use RuVector or enable WAL mode |
| `Out of memory` | Too many vectors | Enable compression or limit elements |
| `Dimension mismatch` | Config doesn't match embeddings | Update `dimension` config |
| `GNN not available` | Package missing | `npm install @ruvector/gnn` |
| `Not enough samples` | Need 10+ for training | Collect more feedback |
| `Native build failed` | Build tools missing | Use WASM fallback (automatic) |

---

## Summary

Most issues can be resolved by:

1. **Installing the right backend**: `npm install @ruvector/core`
2. **Running detection**: `npx agentdb detect`
3. **Checking configuration**: Verify dimension, metric match embeddings
4. **Using appropriate backend**: RuVector for production, SQLite for simple cases

**Still stuck?** File an issue with diagnostic output: `npx agentdb diagnose ./your-db.db`

---

**Next**: [Migration Guide](./MIGRATION_V2.md) | [Backend Configuration](./BACKENDS.md)
