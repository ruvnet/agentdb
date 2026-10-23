# AgentDB Performance Benchmarks

**Version:** 2.0.0 (v2 compatible)
**Last Updated:** 2025-11-29

Comprehensive performance testing suite for AgentDB v2 to validate performance claims and identify optimization opportunities.

## Overview

This benchmark suite tests:

1. **Vector Search Performance**: Tests with 100, 1K, 10K, and 100K vectors
2. **HNSW Indexing**: 150x faster validation with hierarchical indexing
3. **Quantization Performance**: 4-bit and 8-bit quantization (4-32x memory reduction)
4. **Batch Operations**: Batch vs individual operations comparison
5. **Database Backends**: better-sqlite3 vs sql.js vs @ruvector/core
6. **Memory Systems**: Causal graph, reflexion, skill library, ReasoningBank
7. **Self-Learning**: Advanced reasoning and adaptation benchmarks

## Running Benchmarks

```bash
# Run all benchmarks
npm run benchmark

# Build benchmark suite
npm run benchmark:build

# Run specific benchmark directly
npm run benchmark
```

## Benchmark Categories

### 1. Vector Search Performance

Tests vector similarity search with different dataset sizes:

- **100 vectors**: Baseline small dataset
- **1K vectors**: Typical small application
- **10K vectors**: Medium-sized application
- **100K vectors**: Large-scale application

Each test measures:
- Average search time
- Operations per second
- Query latency (p50, p95, p99)

### 2. HNSW Indexing

Validates the "150x faster" claim by comparing:

- **With HNSW**: Hierarchical Navigable Small World indexing
- **Without HNSW**: Brute force cosine similarity search

### 3. Quantization

Tests memory reduction and accuracy tradeoffs:

- **4-bit quantization**: ~8x memory reduction
- **8-bit quantization**: ~4x memory reduction
- **Accuracy analysis**: Measures impact on search quality

### 4. Batch Operations

Compares batch vs individual insert performance:

- **Batch insert**: Insert 1000 vectors at once
- **Individual insert**: Insert vectors one by one
- **Memory usage**: Track memory consumption during batching

### 5. Database Backends

Compares better-sqlite3 vs sql.js:

- **Initialization time**: Database setup performance
- **Insert performance**: Bulk insert operations
- **Query performance**: Search and retrieval speed

### 6. Memory Systems

Tests specialized memory systems:

- **Causal Graph**: Query performance for causal reasoning
- **Reflexion Memory**: Episode retrieval and filtering
- **Skill Library**: Semantic search and categorization

## Report Formats

The benchmark suite generates three report formats:

1. **HTML Report**: `benchmarks/reports/performance-report.html`
   - Interactive visualizations
   - Detailed metrics tables
   - Bottleneck analysis

2. **JSON Report**: `benchmarks/reports/performance-report.json`
   - Machine-readable results
   - Complete benchmark data
   - API integration friendly

3. **Markdown Report**: `benchmarks/reports/performance-report.md`
   - GitHub-friendly format
   - Summary tables
   - Recommendations

## Performance Metrics

Each benchmark tracks:

- **Duration**: Total time to complete benchmark
- **Operations per Second**: Throughput measurement
- **Memory Usage**: Heap memory consumption
- **Success Rate**: Percentage of successful operations
- **Custom Metrics**: Benchmark-specific measurements

## Bottleneck Detection

The suite automatically identifies:

- **Slow Operations**: Operations exceeding 10 seconds
- **Low Throughput**: Operations below 10 ops/sec
- **Memory Leaks**: Abnormal memory growth patterns
- **Performance Degradation**: Comparison with baselines

## Optimization Recommendations

Based on benchmark results, the suite provides:

- **HNSW Usage**: When to enable HNSW indexing
- **Batch Size**: Optimal batch sizes for bulk operations
- **Quantization**: When to use 4-bit vs 8-bit quantization
- **Database Backend**: better-sqlite3 vs sql.js recommendations

## Baseline Comparisons

The "150x faster" claim is validated by:

1. Running identical searches with HNSW enabled
2. Running identical searches with brute force (no HNSW)
3. Calculating speedup factor
4. Comparing against 150x threshold

## Example Output

```
🚀 AgentDB Performance Benchmark Suite
================================================================================

📊 Vector Search Performance
   Test vector similarity search with different dataset sizes
--------------------------------------------------------------------------------
   ✅ Vector Search (100 vectors): 245.32ms (408 ops/sec)
   ✅ Vector Search (1K vectors): 1,234.56ms (81 ops/sec)
   ✅ Vector Search (10K vectors): 3,456.78ms (29 ops/sec)
   ✅ Vector Search (100K vectors): 12,345.67ms (8 ops/sec)
   ✅ Vector Search with HNSW (10K vectors): 234.56ms (426 ops/sec)
   ✅ Vector Search without HNSW (10K vectors): 35,234.56ms (3 ops/sec)
   ✅ 150x Faster Claim Verification: 1,234.56ms
      speedupFactor: 150.23
      claimVerified: true

================================================================================
📈 Benchmark Summary
================================================================================
Total Duration: 45.23s
Total Benchmarks: 18
Successful: 18
Failed: 0

📄 Reports generated:
   - HTML: benchmarks/reports/performance-report.html
   - JSON: benchmarks/reports/performance-report.json
   - Markdown: benchmarks/reports/performance-report.md
```

## Configuration

Benchmark parameters can be customized:

```typescript
// Vector search dataset sizes
const vectorCounts = [100, 1000, 10000, 100000];

// Quantization bits
const quantizationBits = [4, 8];

// Batch sizes
const batchSizes = [10, 100, 1000, 5000];

// Query iterations
const queryCount = 100;
```

## Performance Targets

Expected performance ranges:

| Metric | Target | Actual |
|--------|--------|--------|
| Vector Search (10K) | < 1s | Measured |
| HNSW Speedup | > 100x | Verified |
| Batch Insert | > 1000 vectors/sec | Measured |
| Memory Reduction (4-bit) | > 70% | Measured |
| Initialization Time | < 100ms | Measured |

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run Performance Benchmarks
  run: npm run benchmark

- name: Upload Benchmark Reports
  uses: actions/upload-artifact@v3
  with:
    name: performance-reports
    path: packages/agentdb/benchmarks/reports/
```

## Directory Structure

```
benchmarks/
├── README.md                           # This file
├── package.json                        # Benchmark dependencies
├── tsconfig.json                       # TypeScript configuration
├── vitest.config.ts                    # Main Vitest config
├── vitest.quick.config.ts              # Quick benchmark config
│
├── configs/                            # Configuration files
│   └── baseline.json                   # Baseline metrics
│
├── results/                            # Benchmark results (gitignored)
│   └── .gitkeep
│
├── archive/                            # Historical reports
│   └── old-benchmarks/                 # Archived benchmark reports
│
├── Core Benchmarks (Root Level)
├── simple-benchmark.ts                 # Simple performance test
├── benchmark-runner.ts                 # Main benchmark orchestrator
├── runner.ts                           # Test runner
├── comparison.ts                       # Backend comparison
├── regression-check.ts                 # Regression detection
│
├── Specialized Benchmarks
├── memory.bench.ts                     # Memory system benchmarks
├── vector-search.bench.ts              # Vector search performance
├── benchmark-reasoningbank.js          # ReasoningBank benchmarks
├── benchmark-self-learning.js          # Self-learning benchmarks
├── advanced-reasoning-benchmark.js     # Advanced reasoning tests
├── advanced-self-learning-benchmark.js # Advanced self-learning
│
└── Category-Specific Benchmarks/
    ├── batch-ops/                      # Batch operation benchmarks
    │   └── batch-ops-bench.ts
    ├── database/                       # Database backend benchmarks
    │   └── database-bench.ts
    ├── hnsw/                           # HNSW indexing benchmarks
    │   └── hnsw-benchmark.ts
    ├── memory-systems/                 # Memory system benchmarks
    │   └── memory-bench.ts
    ├── quantization/                   # Quantization benchmarks
    │   └── quantization-bench.ts
    ├── vector-search/                  # Vector search benchmarks
    │   └── vector-search-bench.ts
    └── reports/                        # Report generation
        └── performance-reporter.ts
```

## Benchmark Categories

### Core Performance Tests

#### simple-benchmark.ts
**Purpose:** Quick performance validation
**Tests:**
- Basic vector search
- Database operations
- Memory usage

**Usage:**
```bash
npm run benchmark
# or
tsx simple-benchmark.ts
```

#### benchmark-runner.ts
**Purpose:** Comprehensive benchmark orchestration
**Features:**
- Runs all benchmark suites
- Generates HTML/JSON/Markdown reports
- Bottleneck detection
- Performance regression analysis

**Usage:**
```bash
npm run benchmark:full
```

### Specialized Benchmarks

#### ReasoningBank (benchmark-reasoningbank.js)
**Tests:**
- Pattern storage and retrieval
- Similarity search performance
- Verdict judgment accuracy
- Memory distillation efficiency

**Usage:**
```bash
node benchmark-reasoningbank.js
```

#### Self-Learning (benchmark-self-learning.js)
**Tests:**
- Adaptive learning performance
- Strategy optimization
- Pattern recognition
- Experience replay

**Usage:**
```bash
node benchmark-self-learning.js
```

#### Advanced Reasoning (advanced-reasoning-benchmark.js)
**Tests:**
- Complex causal reasoning
- Multi-step inference
- Graph traversal performance

#### Advanced Self-Learning (advanced-self-learning-benchmark.js)
**Tests:**
- Meta-learning capabilities
- Transfer learning
- Continuous adaptation

### Category-Specific Tests

#### batch-ops/
**Validates:** Batch insert and update operations
**Metrics:**
- Throughput (vectors/sec)
- Memory efficiency
- Transaction performance

#### database/
**Validates:** Database backend comparison
**Backends Tested:**
- better-sqlite3 (native, fastest)
- sql.js (WASM, portable)
- @ruvector/core (optimized)

#### hnsw/
**Validates:** HNSW indexing performance
**Claims Verified:**
- 150x speedup vs brute force
- Logarithmic search complexity
- Index build time

#### memory-systems/
**Validates:** Specialized memory systems
**Systems Tested:**
- Causal Memory Graph
- Reflexion Memory
- Skill Library
- ReasoningBank

#### quantization/
**Validates:** Memory reduction through quantization
**Tests:**
- 4-bit quantization (8x memory reduction)
- 8-bit quantization (4x memory reduction)
- Accuracy degradation analysis

#### vector-search/
**Validates:** Vector similarity search
**Dataset Sizes:**
- 100 vectors (baseline)
- 1K vectors (small)
- 10K vectors (medium)
- 100K vectors (large)

## Running Benchmarks

### Quick Tests
```bash
# Fast benchmarks only
npm run bench:quick

# Memory-specific benchmarks
npm run bench:memory

# Regression detection
npm run bench:regression
```

### Full Suite
```bash
# All benchmarks with reporting
npm run bench

# Watch mode for development
npm run bench:watch

# Generate JSON report
npm run bench:report
```

### Individual Benchmarks
```bash
# Run specific benchmark
tsx simple-benchmark.ts
node benchmark-reasoningbank.js
node benchmark-self-learning.js

# Category-specific
tsx hnsw/hnsw-benchmark.ts
tsx vector-search/vector-search-bench.ts
```

## Configuration

### Baseline Metrics (configs/baseline.json)
Reference performance metrics for regression detection:
```json
{
  "vectorSearch": {
    "1000": { "p50": 10, "p95": 20, "p99": 30 },
    "10000": { "p50": 50, "p95": 100, "p99": 150 }
  },
  "hnswSpeedup": 150,
  "batchThroughput": 1000
}
```

### Vitest Configuration
- **vitest.config.ts:** Full benchmark suite
- **vitest.quick.config.ts:** Fast benchmarks only

### TypeScript Configuration (tsconfig.json)
Optimized for benchmark compilation with proper type checking.

## Performance Targets

### AgentDB v2.0.0 Targets

| Metric | Target | Verification |
|--------|--------|--------------|
| HNSW Speedup | 150x+ | hnsw-benchmark.ts |
| Vector Search (10K) | < 1s | vector-search-bench.ts |
| Batch Insert | > 1000/sec | batch-ops-bench.ts |
| Memory (4-bit) | 75%+ reduction | quantization-bench.ts |
| Backend (SQLite) | < 100ms init | database-bench.ts |
| ReasoningBank | < 50ms search | benchmark-reasoningbank.js |
| Self-Learning | 90%+ accuracy | benchmark-self-learning.js |

## Reports

### Generated Reports (in reports/)
1. **performance-report.html** - Interactive visualizations
2. **performance-report.json** - Machine-readable data
3. **performance-report.md** - GitHub-friendly summary

### Report Contents
- Performance metrics (duration, throughput, memory)
- Bottleneck identification
- Regression detection
- Optimization recommendations
- Baseline comparisons

## Regression Detection

The `regression-check.ts` script compares current performance against baseline:

```bash
npm run bench:regression
```

**Detects:**
- Performance degradation > 20%
- Memory usage increases > 30%
- Throughput reductions > 15%

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Dependencies
        run: |
          cd packages/agentdb
          npm install
          cd benchmarks
          npm install

      - name: Run Benchmarks
        run: |
          cd packages/agentdb/benchmarks
          npm run bench

      - name: Regression Check
        run: |
          cd packages/agentdb/benchmarks
          npm run bench:regression

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-reports
          path: packages/agentdb/benchmarks/reports/
```

## Adding New Benchmarks

### 1. Choose Category
Determine if your benchmark fits into an existing category or needs a new one.

### 2. Create Benchmark File
```typescript
// benchmarks/new-category/my-benchmark.ts
import { bench, describe } from 'vitest';

describe('My Feature Benchmarks', () => {
  bench('operation name', async () => {
    // Your benchmark code
    const result = await myOperation();
    return result;
  }, {
    iterations: 100,
    warmup: 10
  });
});
```

### 3. Update Configuration
Add baseline metrics to `configs/baseline.json`:
```json
{
  "myFeature": {
    "targetLatency": 50,
    "targetThroughput": 1000
  }
}
```

### 4. Document
Update this README with:
- Benchmark description
- Performance targets
- Usage instructions

### 5. Add to CI
Include in benchmark suite if appropriate.

## Troubleshooting

### Out of Memory Errors
```bash
# Increase heap size
node --max-old-space-size=8192 benchmark-runner.ts
```

### Slow Benchmarks
```bash
# Use quick config
npm run bench:quick
```

### Inconsistent Results
- Close other applications
- Run multiple times and average
- Check system resource usage

## Best Practices

1. **Warm-up Iterations:** Use 10-20 warm-up runs before measuring
2. **Multiple Runs:** Average results from 3-5 runs
3. **Isolated Environment:** Minimize background processes
4. **Baseline Comparison:** Always compare against baseline
5. **Document Changes:** Update baseline when intentional changes occur

## Dependencies

### Required
- **Node.js:** >= 20.9.0
- **vitest:** ^1.1.0
- **typescript:** ^5.3.3
- **ts-node:** ^10.9.2

### Optional (for specific benchmarks)
- **better-sqlite3:** Native performance
- **@ruvector/core:** RuVector backend
- **@xenova/transformers:** Embedding generation

## Contributing

To add new benchmarks:

1. Create benchmark file in appropriate category
2. Follow naming convention: `feature-bench.ts` or `benchmark-feature.js`
3. Add baseline metrics to `configs/baseline.json`
4. Update this README with benchmark description
5. Test locally with `npm run bench`
6. Submit PR with performance results

## Support

- **Issues:** https://github.com/ruvnet/agentic-flow/issues
- **Documentation:** See [main docs](../docs/README.md)
- **Performance Reports:** Check `archive/old-benchmarks/` for historical data

## License

MIT

---

**AgentDB v2.0.0 Benchmarks** | Last Updated: 2025-11-29
