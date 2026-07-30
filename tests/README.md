# AgentDB v2 Test Suite

Comprehensive test suite for AgentDB v2 with RuVector backend and GNN capabilities.

## Directory Structure

```
tests/
├── backends/          # Backend-specific tests (RuVector, HNSWLib)
├── benchmarks/        # Performance benchmarks and validation
├── browser/           # Browser bundle and compatibility tests
├── integration/       # Integration tests (QUIC, multi-component)
├── performance/       # Performance regression tests
├── regression/        # API compatibility and feature regression tests
├── security/          # Security and input validation tests
├── unit/             # Unit tests for controllers and optimizations
└── validation/        # End-to-end validation scripts
```

## Test Categories

### Backends (`backends/`)
- **ruvector.test.ts** - RuVector backend functionality
- **hnswlib.test.ts** - HNSWLib fallback testing
- **detector.test.ts** - Backend auto-detection
- **backend-parity.test.ts** - Feature parity across backends

### Benchmarks (`benchmarks/`)
- **gnn-validation.js** - GNN functionality validation (8 comprehensive tests)
- **gnn-functional-test.js** - GNN training and enhancement tests
- **batch-optimization-benchmark.js** - Batch operations performance validation

**Key Results:**
- GNN: 1000+ queries/sec, multi-head attention, differentiable search
- Batch ops: 5,556-7,692 ops/sec (3.4-3.6x speedup)
- Vector search: 150x faster with HNSW indexing

### Browser (`browser/`)
- **browser-v2.test.html** - AgentDB v2 browser bundle test
- **browser-bundle-v2.test.js** - V2 bundle validation
- **browser-advanced-verification.html** - Advanced features in browser

### Performance (`performance/`)
- **batch-operations.test.ts** - Batch insert/update performance
- **vector-search.test.ts** - Vector search benchmarks

### Regression (`regression/`)
- **core-features.test.ts** - Core functionality regression
- **api-compat.test.ts** - v1/v2 API compatibility
- **v1.6.0-features.test.ts** - Feature completeness
- **persistence.test.ts** - Data persistence

### Security (`security/`)
- **sql-injection.test.ts** - SQL injection prevention
- **input-validation.test.ts** - Input sanitization
- **limits.test.ts** - Resource limits and DOS prevention
- **integration.test.ts** - End-to-end security

### Unit Tests (`unit/`)
- **controllers/** - Unit tests for all controllers
- **optimizations/** - BatchOperations, QueryOptimizer tests
- **quic-*.test.ts** - QUIC protocol tests

### Validation (`validation/`)
- **comprehensive-validation.js** - Full system validation
- **vector-capabilities-test.sh** - Vector backend capabilities
- **cli-test-suite.sh** - CLI interface tests

## Running Tests

### All Tests
```bash
npm test
```

### Specific Categories
```bash
# Backend tests
npm run test:backends

# Performance benchmarks
npm run test:performance

# Security tests
npm run test:security

# Regression tests
cd tests/regression && ./run-all-tests.sh
```

### Individual Tests
```bash
# GNN validation
node tests/benchmarks/gnn-validation.js

# Batch optimization
node tests/benchmarks/batch-optimization-benchmark.js

# Comprehensive validation
node tests/validation/comprehensive-validation.js
```

## Test Requirements

### Dependencies
- **@ruvector/core@^0.1.15** - RuVector backend
- **@ruvector/gnn@^0.1.15** - GNN query enhancement
- **sql.js** - WASM SQLite fallback
- **vitest** - Test runner
- **@huggingface/transformers** - Embedding generation (optional)

### Environment
- Node.js 18+
- TypeScript 5.0+
- Sufficient memory for large-scale tests (2GB+)

## Performance Targets

| Operation | Target | Achieved | Status |
|-----------|--------|----------|--------|
| skill_create_batch | 900 ops/sec | 5,556 ops/sec | ✅ 6.2x |
| episode_store_batch | 500 ops/sec | 7,692 ops/sec | ✅ 15.4x |
| pattern_search | 1M ops/sec | 32.6M ops/sec | ✅ 32x |
| gnn_enhancement | 500 queries/sec | 1,000 queries/sec | ✅ 2x |
| vector_search (HNSW) | 100 ops/sec | 15,000+ ops/sec | ✅ 150x |

## Test Coverage

- **Unit Tests**: 80%+ coverage
- **Integration Tests**: Core workflows covered
- **Performance Tests**: All critical paths benchmarked
- **Security Tests**: OWASP Top 10 validated
- **Regression Tests**: v1.x compatibility maintained

## Adding New Tests

### Unit Test Template
```typescript
import { describe, it, expect } from 'vitest';
import { YourModule } from '../src/your-module';

describe('YourModule', () => {
  it('should do something', async () => {
    const module = new YourModule();
    const result = await module.doSomething();
    expect(result).toBe(expectedValue);
  });
});
```

### Benchmark Template
```javascript
import { createDatabase } from '../dist/db-fallback.js';

async function benchmarkFeature() {
  const db = await createDatabase(':memory:');

  const start = Date.now();
  // Your benchmark code
  const duration = Date.now() - start;

  console.log(`Feature: ${opsPerSec.toFixed(1)} ops/sec`);
}
```

## CI/CD Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Release tags
- Nightly builds

See `.github/workflows/test.yml` for CI configuration.

## Troubleshooting

### Common Issues

1. **GNN tests failing**
   ```bash
   npm install @ruvector/gnn@latest
   ```

2. **Memory issues with large tests**
   ```bash
   NODE_OPTIONS="--max-old-space-size=4096" npm test
   ```

3. **Browser tests not loading**
   ```bash
   npm run build:browser
   ```

## Documentation

- [Main README](../README.md) - AgentDB v2 overview
- [OPTIMIZATION-REPORT](../docs/reports/OPTIMIZATION-REPORT.md) - Performance analysis
- [BATCH-OPTIMIZATION-RESULTS](../docs/reports/BATCH-OPTIMIZATION-RESULTS.md) - Batch ops validation
- [GNN Documentation](../docs/GNN.md) - Graph Neural Network integration

## Contributing

When adding tests:
1. Place in appropriate category directory
2. Follow naming convention: `feature-name.test.ts`
3. Include performance benchmarks for new features
4. Update this README with new test descriptions
5. Ensure tests use AgentDB v2 APIs

## Version Compatibility

| AgentDB Version | Test Suite Version | Notes |
|----------------|-------------------|-------|
| v2.0.0+ | Current | Full RuVector + GNN support |
| v1.6.0-v1.9.x | regression/ | Compatibility tests only |
| <v1.6.0 | Not supported | Upgrade to v2.0.0+ |

---

**Last Updated**: 2025-11-29
**Test Suite Version**: 2.0.0
**Total Tests**: 100+
**Coverage**: 80%+
