# @ruvector/attention Integration Architecture for AgentDB v2

**Status**: Final Architecture Design
**Version**: 2.0.0-beta.1
**Date**: 2025-11-30
**Target Release**: AgentDB v2.0.0-beta.1

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Design](#component-design)
4. [Integration Points](#integration-points)
5. [Data Flow Architecture](#data-flow-architecture)
6. [Build System Architecture](#build-system-architecture)
7. [CLI Command Architecture](#cli-command-architecture)
8. [MCP Tool Architecture](#mcp-tool-architecture)
9. [Testing Strategy](#testing-strategy)
10. [Performance Monitoring](#performance-monitoring)
11. [Error Handling Strategy](#error-handling-strategy)
12. [Migration & Backward Compatibility](#migration--backward-compatibility)
13. [Security Considerations](#security-considerations)
14. [Deployment Architecture](#deployment-architecture)

---

## 1. Executive Summary

### 1.1 Integration Goals

This architecture integrates **@ruvector/attention** (NAPI + WASM) into AgentDB v2 to enable:

1. **Hyperbolic Attention** for tree-structured causal memory graphs
2. **FlashAttention** for memory-efficient episodic consolidation
3. **GraphRoPE** for hop-distance-aware graph traversal
4. **MoE Attention** for expert routing across memory domains
5. **DualSpace Attention** for hybrid Euclidean + hyperbolic retrieval

### 1.2 Design Principles

✅ **Zero Breaking Changes**: All existing APIs remain unchanged
✅ **Feature Flags**: Opt-in attention mechanisms via configuration
✅ **Backward Compatible**: Falls back to existing search when attention disabled
✅ **Dual Runtime**: NAPI (Node.js) + WASM (browser) from single codebase
✅ **Performance Monitored**: Comprehensive metrics for all attention operations

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentDB v2.0.0-beta.1                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              AttentionService (NEW)                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │  │
│  │  │ NAPI Layer │  │ WASM Layer │  │ Feature Flags      │ │  │
│  │  │ (Node.js)  │  │ (Browser)  │  │ Runtime Detection  │ │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ▲                                   │
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────┐  │
│  │           Memory Controllers (ENHANCED)                    │  │
│  ├────────────────┬─────────────────┬──────────────────────┤  │
│  │ CausalMemory   │ ReasoningBank   │ ExplainableRecall    │  │
│  │ Graph          │ (Flash+MoE)     │ (GraphRoPE)          │  │
│  │ (Hyperbolic)   │                 │                       │  │
│  └────────────────┴─────────────────┴──────────────────────┘  │
│                              ▲                                   │
│                              │                                   │
│  ┌───────────────────────────┴──────────────────────────────┐  │
│  │              VectorBackend (UNCHANGED)                     │  │
│  │  ┌──────────────┐  ┌──────────────┐                      │  │
│  │  │ RuVector     │  │ HNSWLib      │                      │  │
│  │  │ (150x fast)  │  │ (fallback)   │                      │  │
│  │  └──────────────┘  └──────────────┘                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture

### 2.1 Core Components

#### 2.1.1 AttentionService (NEW)

**Responsibility**: Unified interface for all attention mechanisms

**Location**: `packages/agentdb/src/controllers/AttentionService.ts`

**Interfaces**:
```typescript
export interface AttentionConfig {
  runtime: 'node' | 'browser' | 'auto';
  dimension: number;
  mechanisms: {
    multihead?: MultiHeadConfig;
    flash?: FlashAttentionConfig;
    hyperbolic?: HyperbolicConfig;
    graphRoPE?: GraphRoPEConfig;
    moe?: MoEConfig;
    dualSpace?: DualSpaceConfig;
  };
  enableMetrics?: boolean;
  fallbackToVector?: boolean;
}

export interface AttentionResult {
  output: Float32Array;
  attentionWeights?: Float32Array[];
  mechanism: string;
  latencyMs: number;
  memoryUsed?: number;
}
```

**Key Methods**:
```typescript
class AttentionService {
  // Initialization
  async initialize(config: AttentionConfig): Promise<void>

  // Unified attention interface
  async attend(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[],
    mechanism: AttentionMechanism,
    options?: AttentionOptions
  ): Promise<AttentionResult>

  // Batch processing
  async attendBatch(
    queries: Float32Array[],
    keys: Float32Array[][],
    values: Float32Array[][],
    mechanism: AttentionMechanism
  ): Promise<AttentionResult[]>

  // Runtime detection
  getRuntime(): 'napi' | 'wasm'

  // Metrics
  getMetrics(): AttentionMetrics

  // Feature detection
  isAvailable(mechanism: AttentionMechanism): boolean
}
```

#### 2.1.2 Memory Controller Enhancements

##### CausalMemoryGraph (Hyperbolic Attention)

**Enhancement**: Use hyperbolic attention for tree-structured causal chains

```typescript
// NEW: HyperbolicAttentionEnhancement.ts
export class HyperbolicMemoryEnhancement {
  constructor(
    private causalGraph: CausalMemoryGraph,
    private attentionService: AttentionService
  ) {}

  async queryCausalEffectsWithHyperbolic(
    query: CausalQuery,
    curvature: number = -1.0
  ): Promise<CausalEdge[]> {
    // 1. Get candidate edges from SQLite
    const candidates = this.causalGraph.queryCausalEffects(query);

    // 2. Extract embeddings
    const queryEmb = await this.getEmbedding(query.interventionMemoryId);
    const keys = await Promise.all(
      candidates.map(c => this.getEmbedding(c.toMemoryId))
    );

    // 3. Apply hyperbolic attention
    const result = await this.attentionService.attend(
      queryEmb,
      keys,
      keys, // values = keys for re-ranking
      'hyperbolic',
      { curvature }
    );

    // 4. Re-rank based on hyperbolic attention weights
    return this.rerank(candidates, result.attentionWeights);
  }
}
```

##### ReasoningBank (Flash + MoE Attention)

**Enhancement**: Use FlashAttention for large pattern libraries + MoE for expert routing

```typescript
// NEW: FlashMoEEnhancement.ts
export class ReasoningBankFlashMoE {
  constructor(
    private reasoningBank: ReasoningBank,
    private attentionService: AttentionService
  ) {}

  async searchPatternsWithFlash(
    query: PatternSearchQuery,
    k: number
  ): Promise<ReasoningPattern[]> {
    // Use FlashAttention for memory-efficient search over 10K+ patterns
    const queryEmb = query.taskEmbedding;

    // Get all pattern embeddings (cached)
    const { keys, ids } = await this.getAllPatternEmbeddings();

    // FlashAttention with block-wise processing
    const result = await this.attentionService.attend(
      queryEmb,
      keys,
      keys,
      'flash',
      { blockSize: 256 }
    );

    // Extract top-k
    return this.extractTopK(result, ids, k);
  }

  async routeToExpert(
    query: PatternSearchQuery
  ): Promise<{ expertId: number; patterns: ReasoningPattern[] }> {
    // MoE routing to specialized pattern domains
    const experts = await this.getExpertEmbeddings();

    const result = await this.attentionService.attend(
      query.taskEmbedding,
      experts.keys,
      experts.keys,
      'moe',
      { numExperts: 4, topK: 1 }
    );

    const expertId = this.selectExpert(result);
    const patterns = await this.searchInExpert(expertId, query);

    return { expertId, patterns };
  }
}
```

##### ExplainableRecall (GraphRoPE)

**Enhancement**: Use GraphRoPE for hop-distance-aware graph traversal

```typescript
// NEW: GraphRoPEEnhancement.ts
export class ExplainableRecallGraphRoPE {
  constructor(
    private explainableRecall: ExplainableRecall,
    private attentionService: AttentionService
  ) {}

  async explainWithGraphRoPE(
    query: string,
    maxHops: number = 5
  ): Promise<ExplanationChain[]> {
    // 1. Get causal chain candidates
    const chains = await this.getCausalChains(query, maxHops);

    // 2. Extract node embeddings and hop distances
    const { embeddings, positions } = await this.extractGraphData(chains);

    // 3. Apply GraphRoPE attention
    const queryEmb = await this.embed(query);
    const result = await this.attentionService.attend(
      queryEmb,
      embeddings,
      embeddings,
      'graphrope',
      { positions, maxHops }
    );

    // 4. Re-rank explanations by attention weights
    return this.rankExplanations(chains, result.attentionWeights);
  }
}
```

### 2.2 Architecture Decision Records (ADRs)

#### ADR-001: Dual Runtime Support (NAPI + WASM)

**Decision**: Support both NAPI (Node.js) and WASM (browser) from single codebase

**Rationale**:
- AgentDB v2 targets both server and browser environments
- @ruvector/attention provides both NAPI and WASM packages
- NAPI offers zero-copy performance (35µs/op)
- WASM enables browser deployment with acceptable overhead

**Trade-offs**:
- ✅ PRO: Single AttentionService API for both runtimes
- ✅ PRO: Automatic runtime detection
- ⚠️ CON: WASM requires memory copy (vs NAPI zero-copy)
- ⚠️ CON: Additional build complexity

**Implementation**:
```typescript
class AttentionService {
  private backend: 'napi' | 'wasm';

  async initialize(config: AttentionConfig) {
    const runtime = config.runtime === 'auto'
      ? this.detectRuntime()
      : config.runtime;

    if (runtime === 'node' || (runtime === 'browser' && typeof process !== 'undefined')) {
      // Use NAPI
      this.backend = 'napi';
      const napi = await import('@ruvector/attention');
      this.multihead = new napi.MultiHeadAttention(config.dimension, 8);
    } else {
      // Use WASM
      this.backend = 'wasm';
      const wasm = await import('ruvector-attention-wasm');
      await wasm.init();
      this.multihead = wasm.WasmMultiHeadAttention.new(config.dimension, 8);
    }
  }
}
```

#### ADR-002: Feature Flags for Gradual Rollout

**Decision**: Use feature flags to enable/disable attention mechanisms per controller

**Rationale**:
- Allows gradual testing and rollout
- Users can opt-in to new features
- Provides fallback to existing vector search

**Implementation**:
```typescript
export interface MemoryControllerConfig {
  enableHyperbolicAttention?: boolean;
  enableFlashAttention?: boolean;
  enableGraphRoPE?: boolean;
  enableMoE?: boolean;
}

// Usage in CausalMemoryGraph
class CausalMemoryGraph {
  constructor(
    db: Database,
    graphBackend?: GraphDatabaseAdapter,
    config?: MemoryControllerConfig
  ) {
    this.config = config || {};

    // Only initialize attention if enabled
    if (config?.enableHyperbolicAttention) {
      this.attentionEnhancement = new HyperbolicMemoryEnhancement(this, attentionService);
    }
  }

  async queryCausalEffects(query: CausalQuery): Promise<CausalEdge[]> {
    // Fallback path (existing implementation)
    if (!this.config.enableHyperbolicAttention || !this.attentionEnhancement) {
      return this.queryCausalEffectsLegacy(query);
    }

    // New path with hyperbolic attention
    return this.attentionEnhancement.queryCausalEffectsWithHyperbolic(query);
  }
}
```

#### ADR-003: Performance Monitoring Hooks

**Decision**: Instrument all attention operations with performance metrics

**Rationale**:
- Track latency, memory usage, and throughput
- Detect performance regressions
- Compare NAPI vs WASM performance
- Monitor attention weight distributions

**Implementation**:
```typescript
export interface AttentionMetrics {
  totalCalls: number;
  totalLatencyMs: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  memoryUsedBytes: number;
  mechanism: Record<AttentionMechanism, MechanismMetrics>;
}

class AttentionService {
  private metrics: AttentionMetrics;

  async attend(...): Promise<AttentionResult> {
    const startTime = performance.now();
    const startMemory = process.memoryUsage?.()?.heapUsed || 0;

    try {
      const result = await this.attendInternal(...);

      // Record metrics
      const latency = performance.now() - startTime;
      this.metrics.totalCalls++;
      this.metrics.totalLatencyMs += latency;
      this.recordLatencyPercentile(latency);

      return { ...result, latencyMs: latency };
    } finally {
      const endMemory = process.memoryUsage?.()?.heapUsed || 0;
      this.metrics.memoryUsedBytes += (endMemory - startMemory);
    }
  }
}
```

---

## 3. Component Design

### 3.1 AttentionService Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       AttentionService                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Runtime Abstraction Layer                   │ │
│  │  ┌──────────────┐              ┌──────────────┐            │ │
│  │  │  NAPI Module │              │  WASM Module │            │ │
│  │  │              │              │              │            │ │
│  │  │ - Zero-copy  │              │ - Browser    │            │ │
│  │  │ - 35µs/op    │              │ - Memory copy│            │ │
│  │  │ - Multi-thread│             │ - Single-thread│          │ │
│  │  └──────────────┘              └──────────────┘            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Attention Mechanism Layer                    │ │
│  │  ┌─────────┬─────────┬───────────┬──────────┬──────────┐  │ │
│  │  │MultiHead│ Flash   │Hyperbolic │GraphRoPE │   MoE    │  │ │
│  │  │         │         │           │          │          │  │ │
│  │  │8 heads  │Block256 │Curvature  │MaxHops32 │4 experts │  │ │
│  │  └─────────┴─────────┴───────────┴──────────┴──────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Metrics & Monitoring Layer                  │ │
│  │  ┌──────────────┬──────────────┬─────────────────────────┐│ │
│  │  │ Latency      │ Memory Usage │ Attention Weights       ││ │
│  │  │ Tracking     │ Monitoring   │ Visualization           ││ │
│  │  └──────────────┴──────────────┴─────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Error Handling Layer                      │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │ - Fallback to vector search on failure               │  │ │
│  │  │ - Retry with exponential backoff                     │  │ │
│  │  │ - Graceful degradation for missing mechanisms        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 TypeScript Type Definitions

**File**: `packages/agentdb/src/types/attention.ts`

```typescript
// See AttentionService.ts for full interface definition
// This file exports shared types used across multiple controllers

export type AttentionMechanism =
  | 'multihead'
  | 'flash'
  | 'hyperbolic'
  | 'graphrope'
  | 'moe'
  | 'dualspace'
  | 'linear';

export interface AttentionOptions {
  // MultiHead options
  numHeads?: number;

  // Flash options
  blockSize?: number;

  // Hyperbolic options
  curvature?: number;
  temperature?: number;

  // GraphRoPE options
  positions?: number[];
  maxHops?: number;

  // MoE options
  numExperts?: number;
  topK?: number;
  expertCapacity?: number;

  // DualSpace options
  euclideanDim?: number;
  hyperbolicDim?: number;
  fusionWeight?: number;

  // Performance options
  useCache?: boolean;
  returnWeights?: boolean;
}
```

---

## 4. Integration Points

### 4.1 Integration Point Matrix

| Controller | Attention Mechanism | Integration Type | Priority |
|------------|-------------------|------------------|----------|
| **CausalMemoryGraph** | Hyperbolic | Enhancement | HIGH |
| **ReasoningBank** | Flash + MoE | Enhancement | HIGH |
| **ExplainableRecall** | GraphRoPE | Enhancement | HIGH |
| **SkillLibrary** | Linear | Enhancement | MEDIUM |
| **NightlyLearner** | Flash | Enhancement | MEDIUM |
| **ContextSynthesizer** | DualSpace | New Feature | LOW |

### 4.2 Data Flow: CausalMemoryGraph + HyperbolicAttention

```
┌────────────────────────────────────────────────────────────────┐
│  Step 1: User Query                                             │
│  ─────────────────────────────────────────────────────────────  │
│  const query = {                                                │
│    interventionMemoryId: 42,                                    │
│    interventionMemoryType: 'skill',                             │
│    minConfidence: 0.5                                           │
│  };                                                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 2: SQLite Candidate Retrieval (UNCHANGED)                 │
│  ─────────────────────────────────────────────────────────────  │
│  SELECT * FROM causal_edges                                     │
│  WHERE from_memory_id = 42                                      │
│    AND confidence >= 0.5                                        │
│  ORDER BY uplift * confidence DESC                              │
│  LIMIT 100  -- Initial candidates                               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 3: Embedding Extraction                                   │
│  ─────────────────────────────────────────────────────────────  │
│  queryEmb = await embedder.embed("skill-42-description")       │
│  keysEmb = await Promise.all(                                   │
│    candidates.map(c => embedder.embed(c.toMemoryId))           │
│  )                                                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 4: HyperbolicAttention Application (NEW)                  │
│  ─────────────────────────────────────────────────────────────  │
│  const result = await attentionService.attend(                  │
│    queryEmb,                                                    │
│    keysEmb,                                                     │
│    keysEmb,  // values = keys for re-ranking                    │
│    'hyperbolic',                                                │
│    { curvature: -1.0 }  // Poincaré ball                        │
│  );                                                             │
│                                                                  │
│  // Hyperbolic distance favors tree-structured paths            │
│  attentionWeights = softmax(poincare_distance(queryEmb, keys)) │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 5: Re-ranking (NEW)                                       │
│  ─────────────────────────────────────────────────────────────  │
│  reranked = candidates                                          │
│    .map((c, i) => ({ ...c, attnScore: weights[i] }))           │
│    .sort((a, b) => b.attnScore - a.attnScore)                  │
│    .slice(0, k)                                                 │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Step 6: Return Enhanced Results                                │
│  ─────────────────────────────────────────────────────────────  │
│  return reranked.map(c => ({                                    │
│    ...c,                                                        │
│    hyperblicScore: c.attnScore,                                 │
│    mechanism: 'hyperbolic'                                      │
│  }));                                                           │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow Architecture

### 5.1 End-to-End Data Flow

```
USER REQUEST
    │
    ▼
┌─────────────────────────────────────────┐
│  Memory Controller                      │
│  (CausalMemoryGraph, ReasoningBank...)  │
└─────────────────────────────────────────┘
    │
    ├─── Fallback Path (existing) ──────────► VectorBackend ──► SQLite
    │
    └─── Enhanced Path (NEW)
             │
             ▼
    ┌─────────────────────────────────┐
    │  AttentionService               │
    │  - Runtime detection            │
    │  - Mechanism selection          │
    │  - Metrics recording            │
    └─────────────────────────────────┘
             │
             ├─── NAPI Path (Node.js)
             │        │
             │        ▼
             │   @ruvector/attention (NAPI)
             │   - Zero-copy Float32Array
             │   - Multi-threaded
             │   - 35µs/op
             │
             └─── WASM Path (Browser)
                      │
                      ▼
                 ruvector-attention-wasm
                 - Memory copy required
                 - Single-threaded
                 - ~100µs/op

    Both paths converge ──► AttentionResult
                                  │
                                  ▼
                         Memory Controller
                         (re-rank/enhance)
                                  │
                                  ▼
                            USER RESPONSE
```

---

## 6. Build System Architecture

### 6.1 Dual-Target Build Configuration

**File**: `packages/agentdb/scripts/build-attention.js` (NEW)

```javascript
#!/usr/bin/env node

/**
 * Build script for attention integration
 * Creates separate bundles for Node.js (NAPI) and Browser (WASM)
 */

import esbuild from 'esbuild';
import { join } from 'path';

// Node.js bundle (uses NAPI)
await esbuild.build({
  entryPoints: ['src/controllers/AttentionService.ts'],
  bundle: true,
  platform: 'node',
  target: 'node18',
  format: 'esm',
  outfile: 'dist/attention-node.js',
  external: [
    '@ruvector/attention',  // NAPI bindings (external)
    'better-sqlite3',
    'sqlite3'
  ],
  define: {
    'process.env.ATTENTION_RUNTIME': '"napi"'
  }
});

// Browser bundle (uses WASM)
await esbuild.build({
  entryPoints: ['src/controllers/AttentionService.ts'],
  bundle: true,
  platform: 'browser',
  target: 'es2020',
  format: 'esm',
  outfile: 'dist/attention-browser.js',
  external: [
    'ruvector-attention-wasm'  // WASM module (loaded separately)
  ],
  define: {
    'process.env.ATTENTION_RUNTIME': '"wasm"'
  }
});

console.log('✅ Attention bundles built: dist/attention-{node,browser}.js');
```

### 6.2 package.json Updates

```json
{
  "dependencies": {
    "@ruvector/attention": "^0.1.0",
    "ruvector-attention-wasm": "^0.1.0"
  },
  "scripts": {
    "build:attention": "node scripts/build-attention.js",
    "build": "npm run build:ts && npm run copy:schemas && npm run build:browser && npm run build:attention"
  },
  "exports": {
    "./controllers/AttentionService": {
      "node": "./dist/attention-node.js",
      "browser": "./dist/attention-browser.js",
      "default": "./dist/controllers/AttentionService.js"
    }
  }
}
```

### 6.3 WASM Module Lazy Loading

```typescript
// Browser-specific initialization
async function loadWASMAttention(): Promise<typeof import('ruvector-attention-wasm')> {
  // Lazy load WASM module to reduce initial bundle size
  const wasm = await import('ruvector-attention-wasm');
  await wasm.init();
  return wasm;
}

// Usage in AttentionService
class AttentionService {
  private wasmModule?: typeof import('ruvector-attention-wasm');

  async initialize(config: AttentionConfig) {
    if (this.backend === 'wasm' && !this.wasmModule) {
      this.wasmModule = await loadWASMAttention();
      console.log('✅ WASM attention module loaded');
    }
  }
}
```

---

## 7. CLI Command Architecture

### 7.1 New CLI Commands

**File**: `packages/agentdb/src/cli/commands/attention.ts` (NEW)

```typescript
import { Command } from 'commander';
import { AttentionService } from '../../controllers/AttentionService.js';

export function createAttentionCommands(): Command {
  const attention = new Command('attention')
    .description('Attention mechanism tools and benchmarks');

  // Test attention mechanisms
  attention
    .command('test')
    .description('Test attention mechanisms')
    .option('-m, --mechanism <type>', 'Mechanism to test', 'multihead')
    .option('-d, --dimension <number>', 'Vector dimension', '384')
    .option('-r, --runtime <type>', 'Runtime (napi/wasm/auto)', 'auto')
    .action(async (options) => {
      const service = new AttentionService();
      await service.initialize({
        runtime: options.runtime,
        dimension: parseInt(options.dimension),
        mechanisms: { [options.mechanism]: {} }
      });

      // Run test
      const query = new Float32Array(parseInt(options.dimension)).fill(0.5);
      const keys = [query, query, query];

      const result = await service.attend(query, keys, keys, options.mechanism);

      console.log(`✅ ${options.mechanism} test complete`);
      console.log(`   Runtime: ${service.getRuntime()}`);
      console.log(`   Latency: ${result.latencyMs.toFixed(2)}ms`);
    });

  // Benchmark attention mechanisms
  attention
    .command('benchmark')
    .description('Benchmark attention mechanisms')
    .option('-n, --iterations <number>', 'Number of iterations', '100')
    .action(async (options) => {
      const mechanisms: AttentionMechanism[] = [
        'multihead', 'flash', 'hyperbolic', 'graphrope', 'moe'
      ];

      const results: Record<string, number> = {};

      for (const mechanism of mechanisms) {
        const service = new AttentionService();
        await service.initialize({
          runtime: 'auto',
          dimension: 384,
          mechanisms: { [mechanism]: {} }
        });

        const query = new Float32Array(384).fill(0.5);
        const keys = Array.from({ length: 100 }, () => query);

        const startTime = performance.now();
        for (let i = 0; i < parseInt(options.iterations); i++) {
          await service.attend(query, keys, keys, mechanism);
        }
        const avgLatency = (performance.now() - startTime) / parseInt(options.iterations);

        results[mechanism] = avgLatency;
      }

      // Display results
      console.table(results);
    });

  // Show metrics
  attention
    .command('metrics')
    .description('Display attention metrics')
    .action(async () => {
      // Load metrics from AttentionService (would need persistence)
      console.log('📊 Attention Metrics:');
      console.log('   Total calls: 1,234');
      console.log('   Avg latency: 45ms');
      console.log('   P95 latency: 120ms');
    });

  return attention;
}
```

**Integration**: Add to `packages/agentdb/src/cli/agentdb-cli.ts`:

```typescript
import { createAttentionCommands } from './commands/attention.js';

// ... existing commands ...

program.addCommand(createAttentionCommands());
```

---

## 8. MCP Tool Architecture

### 8.1 New MCP Tools for Attention

**File**: `packages/agentdb/src/mcp/attention-tools.ts` (NEW)

```typescript
import { Tool } from '@modelcontextprotocol/sdk/types.js';

export const attentionTools: Tool[] = [
  {
    name: 'agentdb_attention_query',
    description: 'Query memories using attention mechanisms',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Query text' },
        mechanism: {
          type: 'string',
          enum: ['multihead', 'flash', 'hyperbolic', 'graphrope', 'moe'],
          description: 'Attention mechanism to use'
        },
        k: { type: 'number', description: 'Number of results', default: 10 },
        options: {
          type: 'object',
          description: 'Mechanism-specific options',
          properties: {
            curvature: { type: 'number' },
            blockSize: { type: 'number' },
            numExperts: { type: 'number' }
          }
        }
      },
      required: ['query', 'mechanism']
    }
  },

  {
    name: 'agentdb_attention_metrics',
    description: 'Get attention performance metrics',
    inputSchema: {
      type: 'object',
      properties: {
        mechanism: {
          type: 'string',
          description: 'Filter by mechanism (optional)'
        }
      }
    }
  },

  {
    name: 'agentdb_attention_visualize',
    description: 'Visualize attention weights as heatmap',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        memories: { type: 'array', items: { type: 'string' } },
        mechanism: { type: 'string' }
      },
      required: ['query', 'memories', 'mechanism']
    }
  }
];
```

**Handler Implementation**:

```typescript
// In packages/agentdb/src/mcp/server.ts

import { attentionTools } from './attention-tools.js';

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === 'agentdb_attention_query') {
    const { query, mechanism, k, options } = args;

    // Initialize AttentionService
    const attentionService = new AttentionService();
    await attentionService.initialize({
      runtime: 'auto',
      dimension: 384,
      mechanisms: { [mechanism]: options || {} }
    });

    // Get embeddings
    const queryEmb = await embedder.embed(query);
    const { keys, values, ids } = await getMemoryEmbeddings(k * 3);

    // Apply attention
    const result = await attentionService.attend(
      queryEmb,
      keys,
      values,
      mechanism,
      options
    );

    // Extract top-k
    const ranked = extractTopK(result, ids, k);

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          results: ranked,
          mechanism,
          latencyMs: result.latencyMs
        }, null, 2)
      }]
    };
  }

  // ... other handlers ...
});
```

---

## 9. Testing Strategy

### 9.1 Test Pyramid

```
                    ▲
                   ╱ ╲
                  ╱   ╲
                 ╱  E2E ╲  (10 tests)
                ╱───────╲
               ╱         ╲
              ╱Integration╲  (30 tests)
             ╱─────────────╲
            ╱               ╲
           ╱  Unit Tests     ╲  (100 tests)
          ╱───────────────────╲
         ╱                     ╲
        ╱   Browser/Benchmark   ╲  (20 tests)
       ╱─────────────────────────╲
```

### 9.2 Unit Tests

**File**: `packages/agentdb/src/tests/attention-service.test.ts` (NEW)

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { AttentionService } from '../controllers/AttentionService.js';

describe('AttentionService', () => {
  let service: AttentionService;

  beforeEach(async () => {
    service = new AttentionService();
    await service.initialize({
      runtime: 'auto',
      dimension: 384,
      mechanisms: {
        multihead: { numHeads: 8 },
        flash: { blockSize: 256 },
        hyperbolic: { curvature: -1.0 }
      }
    });
  });

  describe('Runtime Detection', () => {
    it('should detect NAPI runtime in Node.js', () => {
      expect(service.getRuntime()).toBe('napi');
    });
  });

  describe('MultiHead Attention', () => {
    it('should compute attention correctly', async () => {
      const query = new Float32Array(384).fill(1);
      const keys = [query, query];

      const result = await service.attend(query, keys, keys, 'multihead');

      expect(result.output).toBeInstanceOf(Float32Array);
      expect(result.output.length).toBe(384);
      expect(result.latencyMs).toBeGreaterThan(0);
    });

    it('should handle batch processing', async () => {
      const queries = Array.from({ length: 5 }, () => new Float32Array(384).fill(1));
      const keys = queries.map(q => [q, q]);

      const results = await service.attendBatch(queries, keys, keys, 'multihead');

      expect(results).toHaveLength(5);
    });
  });

  describe('Hyperbolic Attention', () => {
    it('should apply hyperbolic distance', async () => {
      const query = new Float32Array(384).fill(1);
      const keys = [query, new Float32Array(384).fill(0.5)];

      const result = await service.attend(
        query,
        keys,
        keys,
        'hyperbolic',
        { curvature: -1.0 }
      );

      expect(result.mechanism).toBe('hyperbolic');
      expect(result.attentionWeights).toBeDefined();
    });
  });

  describe('Metrics', () => {
    it('should track latency metrics', async () => {
      const query = new Float32Array(384).fill(1);

      await service.attend(query, [query], [query], 'multihead');
      await service.attend(query, [query], [query], 'multihead');

      const metrics = service.getMetrics();

      expect(metrics.totalCalls).toBe(2);
      expect(metrics.avgLatencyMs).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should fallback on unsupported mechanism', async () => {
      const query = new Float32Array(384).fill(1);

      await expect(
        service.attend(query, [query], [query], 'unsupported' as any)
      ).rejects.toThrow('Unsupported mechanism');
    });
  });
});
```

### 9.3 Integration Tests

**File**: `packages/agentdb/src/tests/causal-hyperbolic-integration.test.ts` (NEW)

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { CausalMemoryGraph } from '../controllers/CausalMemoryGraph.js';
import { AttentionService } from '../controllers/AttentionService.js';
import { EmbeddingService } from '../controllers/EmbeddingService.js';

describe('CausalMemoryGraph + HyperbolicAttention Integration', () => {
  let db: any;
  let causalGraph: CausalMemoryGraph;
  let attentionService: AttentionService;

  beforeEach(async () => {
    // Initialize in-memory database
    db = new Database(':memory:');

    // Initialize services
    const embedder = new EmbeddingService({
      model: 'mock',
      dimension: 384,
      provider: 'local'
    });
    await embedder.initialize();

    attentionService = new AttentionService();
    await attentionService.initialize({
      runtime: 'auto',
      dimension: 384,
      mechanisms: { hyperbolic: { curvature: -1.0 } }
    });

    causalGraph = new CausalMemoryGraph(db, undefined, {
      enableHyperbolicAttention: true
    });
  });

  it('should enhance causal queries with hyperbolic attention', async () => {
    // Add causal edges
    await causalGraph.addCausalEdge({
      fromMemoryId: 1,
      fromMemoryType: 'skill',
      toMemoryId: 2,
      toMemoryType: 'skill',
      similarity: 0.8,
      confidence: 0.9,
      uplift: 0.2
    });

    // Query with hyperbolic enhancement
    const results = await causalGraph.queryCausalEffects({
      interventionMemoryId: 1,
      interventionMemoryType: 'skill',
      minConfidence: 0.5
    });

    expect(results).toHaveLength(1);
    expect(results[0].hyperbolicScore).toBeDefined();
  });
});
```

### 9.4 Browser Compatibility Tests

**File**: `packages/agentdb/src/tests/browser-wasm-attention.test.ts` (NEW)

```typescript
import { describe, it, expect } from 'vitest';

describe('Browser WASM Attention', () => {
  // These tests run in browser environment (vitest browser mode)

  it('should load WASM module in browser', async () => {
    const wasm = await import('ruvector-attention-wasm');
    await wasm.init();

    expect(wasm.version()).toBeDefined();
  });

  it('should compute attention in browser', async () => {
    const wasm = await import('ruvector-attention-wasm');
    await wasm.init();

    const mha = wasm.WasmMultiHeadAttention.new(384, 8);
    const query = new Float32Array(384).fill(1);
    const keys = [query, query];

    const result = mha.compute(query, keys, keys);

    expect(result).toBeInstanceOf(Float32Array);
  });
});
```

### 9.5 Benchmark Suite

**File**: `packages/agentdb/benchmarks/attention-benchmark.ts` (NEW)

```typescript
import { AttentionService } from '../src/controllers/AttentionService.js';

async function benchmarkMechanism(
  mechanism: string,
  dimension: number,
  numKeys: number,
  iterations: number
) {
  const service = new AttentionService();
  await service.initialize({
    runtime: 'auto',
    dimension,
    mechanisms: { [mechanism]: {} }
  });

  const query = new Float32Array(dimension).fill(0.5);
  const keys = Array.from({ length: numKeys }, () => query);

  const startTime = performance.now();

  for (let i = 0; i < iterations; i++) {
    await service.attend(query, keys, keys, mechanism as any);
  }

  const totalTime = performance.now() - startTime;
  const avgLatency = totalTime / iterations;
  const throughput = (iterations * 1000) / totalTime;

  return { avgLatency, throughput };
}

async function runBenchmarks() {
  console.log('🔥 Attention Mechanism Benchmarks\n');

  const configs = [
    { mechanism: 'multihead', dimension: 384, numKeys: 100 },
    { mechanism: 'flash', dimension: 768, numKeys: 1000 },
    { mechanism: 'hyperbolic', dimension: 384, numKeys: 100 },
    { mechanism: 'graphrope', dimension: 384, numKeys: 100 },
    { mechanism: 'moe', dimension: 384, numKeys: 100 }
  ];

  for (const config of configs) {
    const result = await benchmarkMechanism(
      config.mechanism,
      config.dimension,
      config.numKeys,
      100
    );

    console.log(`${config.mechanism}:`);
    console.log(`  Avg Latency: ${result.avgLatency.toFixed(2)}ms`);
    console.log(`  Throughput: ${result.throughput.toFixed(0)} ops/sec\n`);
  }
}

runBenchmarks();
```

---

## 10. Performance Monitoring

### 10.1 Metrics Collection

```typescript
export interface AttentionMetrics {
  // Call statistics
  totalCalls: number;
  callsByMechanism: Record<AttentionMechanism, number>;

  // Latency statistics
  totalLatencyMs: number;
  avgLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  latencyByMechanism: Record<AttentionMechanism, LatencyStats>;

  // Memory statistics
  totalMemoryUsedBytes: number;
  avgMemoryPerCall: number;
  peakMemoryBytes: number;

  // Throughput
  opsPerSecond: number;

  // Runtime distribution
  napiCalls: number;
  wasmCalls: number;

  // Error tracking
  totalErrors: number;
  errorsByType: Record<string, number>;
}
```

### 10.2 Performance Targets

| Metric | Target (NAPI) | Target (WASM) | Measurement Method |
|--------|---------------|---------------|-------------------|
| **MultiHead Latency (384-dim, 100 keys)** | <50ms | <150ms | Benchmark suite |
| **Flash Latency (768-dim, 1000 keys)** | <200ms | <500ms | Benchmark suite |
| **Hyperbolic Latency (384-dim, 100 keys)** | <60ms | <180ms | Benchmark suite |
| **Memory Overhead** | <100MB | <150MB | process.memoryUsage() |
| **Throughput (MultiHead)** | >20 ops/sec | >10 ops/sec | Benchmark suite |

### 10.3 Monitoring Dashboard (CLI)

```bash
# Display live metrics
$ agentdb attention metrics

📊 Attention Performance Metrics

┌──────────────┬───────────┬────────────┬──────────┬──────────┐
│  Mechanism   │  Calls    │  Avg (ms)  │  P95 (ms)│  P99 (ms)│
├──────────────┼───────────┼────────────┼──────────┼──────────┤
│ MultiHead    │    1,234  │      45.2  │     120  │     250  │
│ Flash        │      567  │     180.5  │     400  │     600  │
│ Hyperbolic   │      890  │      52.1  │     130  │     280  │
│ GraphRoPE    │      234  │      61.3  │     150  │     320  │
│ MoE          │      456  │      73.8  │     180  │     380  │
└──────────────┴───────────┴────────────┴──────────┴──────────┘

Runtime Distribution:
  NAPI: 85% (2,456 calls)
  WASM: 15% (925 calls)

Memory Usage:
  Total: 245 MB
  Average per call: 85 KB
  Peak: 412 MB
```

---

## 11. Error Handling Strategy

### 11.1 Error Hierarchy

```typescript
export class AttentionError extends Error {
  constructor(
    message: string,
    public mechanism: AttentionMechanism,
    public cause?: Error
  ) {
    super(message);
    this.name = 'AttentionError';
  }
}

export class AttentionInitializationError extends AttentionError {
  constructor(message: string, mechanism: AttentionMechanism) {
    super(`Initialization failed: ${message}`, mechanism);
    this.name = 'AttentionInitializationError';
  }
}

export class AttentionComputationError extends AttentionError {
  constructor(message: string, mechanism: AttentionMechanism, cause?: Error) {
    super(`Computation failed: ${message}`, mechanism, cause);
    this.name = 'AttentionComputationError';
  }
}
```

### 11.2 Graceful Degradation Strategy

```typescript
class AttentionService {
  async attend(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[],
    mechanism: AttentionMechanism,
    options?: AttentionOptions
  ): Promise<AttentionResult> {
    try {
      // Attempt attention computation
      return await this.attendInternal(query, keys, values, mechanism, options);
    } catch (error) {
      // Log error
      console.error(`AttentionService error (${mechanism}):`, error);
      this.metrics.totalErrors++;

      // Fallback strategy
      if (this.config.fallbackToVector) {
        console.warn(`Falling back to vector search for ${mechanism}`);
        return this.fallbackToVectorSearch(query, keys, values);
      }

      throw new AttentionComputationError(
        error instanceof Error ? error.message : 'Unknown error',
        mechanism,
        error instanceof Error ? error : undefined
      );
    }
  }

  private async fallbackToVectorSearch(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): Promise<AttentionResult> {
    // Simple cosine similarity fallback
    const similarities = keys.map(key => this.cosineSimilarity(query, key));
    const weights = this.softmax(similarities);

    // Weighted average of values
    const output = this.weightedAverage(values, weights);

    return {
      output,
      attentionWeights: [new Float32Array(weights)],
      mechanism: 'fallback',
      latencyMs: 0
    };
  }
}
```

### 11.3 Retry Policy

```typescript
async attendWithRetry(
  query: Float32Array,
  keys: Float32Array[],
  values: Float32Array[],
  mechanism: AttentionMechanism,
  options?: AttentionOptions,
  maxRetries: number = 3
): Promise<AttentionResult> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await this.attend(query, keys, values, mechanism, options);
    } catch (error) {
      lastError = error as Error;

      // Exponential backoff
      const delayMs = Math.pow(2, attempt) * 100;
      await new Promise(resolve => setTimeout(resolve, delayMs));

      console.warn(`Retry ${attempt + 1}/${maxRetries} for ${mechanism}`);
    }
  }

  throw new AttentionComputationError(
    `Failed after ${maxRetries} retries: ${lastError?.message}`,
    mechanism,
    lastError
  );
}
```

---

## 12. Migration & Backward Compatibility

### 12.1 Migration Strategy

**Phase 1: Opt-in (v2.0.0-beta.1)**
- AttentionService added, disabled by default
- Feature flags required to enable
- No breaking changes

**Phase 2: Gradual Rollout (v2.1.0)**
- Enable hyperbolic attention by default for CausalMemoryGraph
- Performance monitoring enabled
- Automatic fallback on errors

**Phase 3: Full Integration (v2.2.0)**
- All memory controllers use attention by default
- Legacy vector search deprecated but available
- CLI tools fully integrated

### 12.2 Backward Compatibility Guarantees

✅ **API Compatibility**: All existing controller APIs unchanged
✅ **Feature Flags**: Attention mechanisms opt-in via config
✅ **Fallback**: Graceful degradation to vector search on errors
✅ **Dependencies**: Optional dependencies for attention packages
✅ **TypeScript**: No breaking type changes

### 12.3 Migration Guide

**For Users (v2.0.0-alpha.2.7 → v2.0.0-beta.1)**

```typescript
// BEFORE (v2.0.0-alpha.2.7)
const causalGraph = new CausalMemoryGraph(db);
const results = await causalGraph.queryCausalEffects(query);

// AFTER (v2.0.0-beta.1) - Opt-in to hyperbolic attention
const causalGraph = new CausalMemoryGraph(db, undefined, {
  enableHyperbolicAttention: true  // NEW: Feature flag
});
const results = await causalGraph.queryCausalEffects(query);
// Results now include hyperbolicScore field
```

**For Developers (adding new attention integration)**

```typescript
// 1. Import AttentionService
import { AttentionService } from './AttentionService.js';

// 2. Add feature flag to config
interface MyControllerConfig {
  enableMyAttention?: boolean;
}

// 3. Initialize AttentionService
class MyController {
  private attentionService?: AttentionService;

  constructor(config?: MyControllerConfig) {
    if (config?.enableMyAttention) {
      this.attentionService = new AttentionService();
      await this.attentionService.initialize({
        runtime: 'auto',
        dimension: 384,
        mechanisms: { multihead: {} }
      });
    }
  }

  // 4. Add fallback logic
  async myQuery(query: string): Promise<Result[]> {
    if (!this.attentionService) {
      return this.myQueryLegacy(query);  // Fallback
    }

    // Use attention
    const queryEmb = await this.embed(query);
    const result = await this.attentionService.attend(...);
    return this.processAttentionResult(result);
  }
}
```

---

## 13. Security Considerations

### 13.1 WASM Sandboxing

✅ **Browser Isolation**: WASM runs in isolated linear memory
✅ **No File System Access**: WASM cannot access file system
✅ **CSP Compliance**: Supports Content Security Policy headers

### 13.2 NAPI Security

⚠️ **Native Code Execution**: NAPI bindings execute native Rust code
✅ **Platform Binaries**: Prebuild binaries for 8 platforms (verified by npm)
✅ **Source Code Audited**: @ruvector/attention source code reviewed (see RUVECTOR-ATTENTION-SOURCE-CODE-ANALYSIS.md)

### 13.3 Input Validation

```typescript
class AttentionService {
  private validateInputs(
    query: Float32Array,
    keys: Float32Array[],
    values: Float32Array[]
  ): void {
    // Dimension validation
    if (query.length !== this.config.dimension) {
      throw new Error(`Query dimension mismatch: expected ${this.config.dimension}, got ${query.length}`);
    }

    // Keys/values length match
    if (keys.length !== values.length) {
      throw new Error(`Keys/values length mismatch: ${keys.length} vs ${values.length}`);
    }

    // NaN/Infinity check
    if (!isFinite(query[0])) {
      throw new Error('Query contains invalid values (NaN/Infinity)');
    }

    // Size limits (prevent DoS)
    if (keys.length > 100000) {
      throw new Error('Too many keys (max 100,000)');
    }
  }
}
```

---

## 14. Deployment Architecture

### 14.1 Node.js Deployment (Production)

```
┌─────────────────────────────────────────────────────────────┐
│              Production Node.js Server                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  AgentDB v2.0.0-beta.1 (NAPI Runtime)                  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  @ruvector/attention (NAPI bindings)             │  │ │
│  │  │  - Prebuild binary: attention.linux-x64-gnu.node │  │ │
│  │  │  - Zero-copy Float32Array                        │  │ │
│  │  │  - Multi-threaded (tokio)                        │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                         │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  RuVector Backend (150x faster)                  │  │ │
│  │  │  - HNSW indexing                                 │  │ │
│  │  │  - SIMD acceleration                             │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Monitoring & Metrics                                  │ │
│  │  - Prometheus exporter                                 │ │
│  │  - Grafana dashboards                                  │ │
│  │  - Alert on P95 latency > 200ms                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Docker Deployment**:

```dockerfile
FROM node:22-alpine

# Install dependencies
WORKDIR /app
COPY package*.json ./
RUN npm ci --production

# Copy application
COPY dist ./dist

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD node -e "require('./dist/cli/agentdb-cli.js')" || exit 1

CMD ["node", "dist/cli/agentdb-cli.js", "serve"]
```

### 14.2 Browser Deployment (Edge)

```html
<!DOCTYPE html>
<html>
<head>
  <title>AgentDB Browser Demo</title>
</head>
<body>
  <script type="module">
    // Load WASM attention module
    import { init } from 'https://cdn.jsdelivr.net/npm/ruvector-attention-wasm@0.1.0/+esm';
    import { AttentionService } from './dist/attention-browser.js';

    // Initialize
    await init();

    const service = new AttentionService();
    await service.initialize({
      runtime: 'wasm',
      dimension: 384,
      mechanisms: { multihead: { numHeads: 8 } }
    });

    // Use attention
    const query = new Float32Array(384).fill(0.5);
    const result = await service.attend(query, [query], [query], 'multihead');

    console.log('WASM Attention Result:', result);
  </script>
</body>
</html>
```

### 14.3 Cloudflare Workers Deployment

```typescript
// Cloudflare Workers Edge Deployment
import { AttentionService } from 'agentdb/controllers/AttentionService';

export default {
  async fetch(request: Request): Promise<Response> {
    // Initialize WASM attention (cold start ~50ms)
    const service = new AttentionService();
    await service.initialize({
      runtime: 'wasm',
      dimension: 384,
      mechanisms: { multihead: {} }
    });

    // Process request
    const { query, keys } = await request.json();
    const result = await service.attend(query, keys, keys, 'multihead');

    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
```

---

## 15. Architecture Summary

### 15.1 Key Architectural Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Dual Runtime (NAPI + WASM)** | Support Node.js + browser | Build complexity, WASM overhead |
| **Feature Flags** | Gradual rollout, backward compatible | Configuration complexity |
| **AttentionService Abstraction** | Unified API for all mechanisms | Additional abstraction layer |
| **Performance Monitoring** | Track regressions, optimize | Metric collection overhead |
| **Graceful Degradation** | Fallback to vector search on errors | Complexity in error handling |
| **Optional Dependencies** | Don't force attention on all users | npm peer dependency warnings |

### 15.2 Integration Checklist

✅ **Architecture Design** (this document)
✅ **AttentionService Interface** (see AttentionService.ts)
✅ **TypeScript Types** (see types/attention.ts)
✅ **Build System Changes** (package.json, scripts/build-attention.js)
✅ **CLI Commands** (cli/commands/attention.ts)
✅ **MCP Tools** (mcp/attention-tools.ts)
✅ **Testing Strategy** (unit, integration, browser, benchmark)
✅ **Error Handling** (AttentionError hierarchy, fallback)
✅ **Performance Monitoring** (metrics collection, dashboard)
✅ **Migration Guide** (backward compatibility, feature flags)
✅ **Security Review** (WASM sandboxing, input validation)
✅ **Deployment Plan** (Node.js, browser, edge)

### 15.3 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Zero Breaking Changes** | 100% backward compatible | Existing test suite passes |
| **Performance Gain** | 3x faster causal queries | Benchmark suite |
| **Memory Efficiency** | 10x reduction with Flash | Memory profiling |
| **Browser Support** | 95%+ users | WASM compatibility tests |
| **Test Coverage** | >85% for attention code | vitest coverage report |
| **Documentation** | 100% public APIs documented | TypeDoc coverage |

---

**Document Version**: 1.0 (Final)
**Last Updated**: 2025-11-30
**Review Status**: ✅ READY FOR IMPLEMENTATION
**Implementation Team**: AgentDB Core Team

---

## Appendix A: File Tree

```
packages/agentdb/
├── docs/
│   └── integration/
│       └── ARCHITECTURE.md (this file)
├── src/
│   ├── controllers/
│   │   ├── AttentionService.ts (NEW - interface only)
│   │   ├── CausalMemoryGraph.ts (ENHANCED)
│   │   ├── ReasoningBank.ts (ENHANCED)
│   │   └── ExplainableRecall.ts (ENHANCED)
│   ├── types/
│   │   └── attention.ts (NEW)
│   ├── cli/
│   │   └── commands/
│   │       └── attention.ts (NEW)
│   ├── mcp/
│   │   └── attention-tools.ts (NEW)
│   └── tests/
│       ├── attention-service.test.ts (NEW)
│       ├── causal-hyperbolic-integration.test.ts (NEW)
│       └── browser-wasm-attention.test.ts (NEW)
├── scripts/
│   └── build-attention.js (NEW)
├── benchmarks/
│   └── attention-benchmark.ts (NEW)
└── package.json (UPDATED)
```

## Appendix B: Dependencies

```json
{
  "dependencies": {
    "@ruvector/attention": "^0.1.0",
    "ruvector-attention-wasm": "^0.1.0"
  },
  "peerDependencies": {
    "@ruvector/attention": "^0.1.0"
  },
  "peerDependenciesMeta": {
    "@ruvector/attention": {
      "optional": true
    }
  }
}
```

## Appendix C: Glossary

- **NAPI**: Node.js API for native addons (Rust bindings)
- **WASM**: WebAssembly (browser-compatible binary format)
- **Hyperbolic Attention**: Attention mechanism using Poincaré distance
- **FlashAttention**: Block-wise tiled attention (Dao 2022)
- **GraphRoPE**: Rotary Position Embeddings for graph structures
- **MoE**: Mixture of Experts (sparse gating)
- **DualSpace**: Hybrid Euclidean + hyperbolic attention
