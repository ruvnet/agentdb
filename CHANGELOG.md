# Changelog

All notable changes to AgentDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-10-31

Hybrid episode retrieval and a LongMemEval benchmark (docs/adrs/ADR-016). No API removals. The default order of `retrieveRelevant()` / `agentdb_search` results changes; `hybrid: false` restores the 3.1.0 order.

### Added

- **Hybrid retrieval, on by default.** `ReflexionMemory.retrieveRelevant()` (and `agentdb_search`) fuse BM25 keyword ranking over episode text with the vector ranking by reciprocal-rank fusion. Keyword-only hits are fetched, filtered like vector hits, and carry their real cosine `similarity`. `hybrid: false` (MCP: `"hybrid": false`) gives pure vector order. Not applied with a graph backend.
- **Opt-in cross-encoder reranking.** `rerank: true` (MCP: `"rerank": true`) rereads the top 60 vector and top 60 keyword candidates with a local cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, q8; `AGENTDB_RERANKER_MODEL` to change). About 200 ms per call on CPU and a 23 MB download on first use; any failure keeps the hybrid order.
- **`benchmarks/longmemeval/`**: a LongMemEval session-retrieval harness over the same ranking functions (streaming loader for the 2.7 GB M split, cached embeddings, cached cross-encoder scores). Results for S and M are in `benchmarks/longmemeval/results/`.

### Measured (LongMemEval session retrieval, MiniLM embedder, official protocol: 470 questions, abstention skipped)

| | S recall_all@5 | S NDCG@5 | M recall_all@5 | M NDCG@5 | M recall_all@10 |
|---|---|---|---|---|---|
| dense only (same chunks) | 0.883 | 0.904 | 0.630 | 0.692 | 0.751 |
| hybrid (new default) | 0.900 | 0.924 | 0.719 | 0.750 | 0.806 |
| hybrid + rerank (MiniLM-L-6) | 0.934 | 0.943 | 0.789 | 0.828 | 0.881 |

On M, the best published retrieval in the LongMemEval paper (Table 9: Contriever with LLM-extracted fact keys) is recall 0.762, NDCG 0.632 at 5 and recall 0.862 at 10. The harness runs the ranking functions over turn chunks, not the episode store; see ADR-016 for the protocol and its limits.

### Changed

- `ruvector` optional dependency `^0.2.40` → `^0.3.5`.

### Security

- `agentdb_search` validates `query` (string, non-empty, at most 10,000 characters); it was passed through unchecked. Tokenising caps text at 32 KB and 4,096 tokens per document, 256 per query; the tokeniser regex is linear. `AGENTDB_HYBRID_MAX_EPISODES` is clamped to [1, 1,000,000].

### Performance

- The BM25 index covers the newest 10,000 episodes (`AGENTDB_HYBRID_MAX_EPISODES`). The first search in a process builds it (about 200 ms at 10k, 1 s at 50k); later writes are appended, so a write-then-search costs about 8 ms at 50k. Deletes trigger a rebuild.

## [3.1.0] - 2026-10-31

Retrieval quality and observability (docs/adrs/ADR-015). No breaking changes.

### Added

- **Outcome feedback.** `SkillLibrary.recordOutcome(skillId, success, reward?, latencyMs?)`; `ReasoningBank.recordOutcome()` now returns `false` for an unknown id and rejects a non-finite reward. MCP tools `agentdb_pattern_feedback` and `agentdb_skill_feedback` (default `learning` profile) take an id plus `success`, a `reward` in [0, 1], or both; inputs are validated at the boundary.
- **Reward-aware pattern ranking.** `searchPatterns()` ranks by `0.7·similarity + 0.2·successRate + 0.1·avgReward` in both the SQL and vector-backend paths. Weights are configurable (constructor `{ rerankWeights }` or per query); `rerank: false` restores similarity order. Results carry `score`; `similarity` is unchanged.
- **`agentdb_capabilities` MCP tool** (read-only, default and readonly profiles): native accelerator features, the vector backend in use and the ones available, the SQL driver, the embedding model and dimension and whether the hash fallback is active, and the ranking defaults. `EmbeddingService.getRuntimeInfo()` backs the embedding part.
- `searchVectors(backend, query, k, options, exact?)` takes an optional exact-search source; `SearchOptions.exactSearchThreshold` and `AGENTDB_EXACT_SEARCH_THRESHOLD` configure it.

### Fixed

- **Stored vectors missing from small indexes.** `@ruvector/core` 0.1.30's HNSW misses neighbours on tiny graphs (ruvnet/RuVector#1298); on 3.0.0 the native backend missed 41–85 of 225 self-lookups on 3–12-vector collections. Collections under 1000 rows are now answered by exact cosine over the stored embeddings, as is any search where the index returns fewer of the collection's vectors than `min(k, rows)` or throws. ReflexionMemory, SkillLibrary and ReasoningBank (vector-backend mode) use it; 3.0.0 only fell back when the index returned nothing.
- ReasoningBank in vector-backend mode keeps the `pattern_embeddings` copy, skips deleted patterns instead of throwing, ignores other controllers' hits in a shared index, and `deletePattern()` removes the embedding row.

### Performance

- Full-table scans (ReflexionMemory SQL fallback, SkillLibrary legacy, `WASMVectorSearch.findKNN`, `SqlJsRvfBackend` search) keep a bounded top-k heap, compute the query norm once and build result objects for the winners only; `SqlJsRvfBackend` caches each vector's norm. Results are bit-identical. N=10k, 384-dim, k=10, best of three median: SqlJsRvfBackend 6.7 → 2.3 ms, SkillLibrary 44.5 → 35.4 ms, ReflexionMemory 44.8 → 38.3 ms, findKNN 23.3 → 19.6 ms (`benchmarks/topk-scan-bench.mts`).

## [3.0.0] - 2026-10-31

First stable 3.x release, from a security, integration, self-learning and performance review (docs/adrs/ADR-014). Published to `latest` only; no alpha dist-tag.

### Breaking

- `RuVectorBackend.search()` is async-only and throws if called synchronously (it used to return `[]`). Use `searchAsync()`, or `searchVectors()` from `agentdb/backends`, which every controller now uses.
- `RvfBackend.freeze()` and `NativeAccelerator`'s `graphBatchInsertNodes`, `graphCypherQuery` and `coreBatchInsert` return promises.
- With `AGENTDB_MCP_PROFILE` unset, the MCP server uses the `learning` profile (now including the memory-write tools) instead of allowing every tool. `agentdb_init`, `agentdb_delete`, `agentdb_delete_batch` and `agentdb_clear_cache` need `AGENTDB_MCP_PROFILE=admin` or `AGENTDB_MCP_ALLOW`.
- The QUIC server rejects every request when no token is configured, binds `127.0.0.1` by default, and the CLI no longer prints its token (`--auth-token` / `AGENTDB_QUIC_AUTH_TOKEN`).
- `AgentDB.close()` also flushes and closes the vector backend.
- Library diagnostics go to stderr, never stdout (same text).

### Security

- `agentdb_init` no longer deletes files: `db_path` is gone, and `reset` drops the tables on the open handle. It used to `unlinkSync` any caller-supplied path, under an allow-all default policy.
- SQL injection through filter keys (`HNSWIndex`, `WASMVectorSearch`) and metadata paths (`MetadataFilter`) is closed: keys must be real column identifiers and JSON paths are bound parameters.
- The model cache moved from shared `/tmp/agentdb-models` to a private `$XDG_CACHE_HOME/agentdb-models` (0700, owner-checked, not a symlink), and `.rvf` extraction can no longer write outside its directory.
- `LLMRouter` no longer loads `.env` files implicitly (it read a hardcoded `/workspaces/agentic-flow/.env`); pass `envFile`.
- PRAGMA commands are validated whole and run as one statement; Cypher numbers are coerced to finite values; MCP batch items are capped at 1000 and `k` at 100.
- OpenTelemetry advisories: `@opentelemetry/sdk-node` ^0.220 (propagator-jaeger <2.9) and `@opentelemetry/resources` ^2.8 (core <2.8).

### Fixed

- **The default vector backend returns results.** `RuVectorBackend` never awaited `VectorDB.search()`, so every search was empty. It also called methods `VectorDB` lacks, ignored HNSW options and shared one `./ruvector.db` in the cwd. Searches are awaited, writes are ordered, storage is `<dbPath>.ruvector` (or private temp for `:memory:`), and Reflexion and SkillLibrary fall back to SQL when the index returns nothing.
- Episodes no longer load as skills from the shared index.
- `LearningSystem` saves feedback and trained policies, so `predict()` reflects them.
- RVF witness verification fails closed with an explicit `unsupported` result on @ruvector/rvf builds without it (it threw "is not a function").
- `NativeAccelerator` probes the APIs the native packages actually export, and reports them through `capabilities()`.
- `FederatedCoordinator` is constructed with `(id, config)` and enforces `maxAgents`.
- sql.js: statements freed by `save()`/`export()` are re-prepared, so the next write no longer throws.
- Library timers no longer keep short-lived processes alive. `getPatternStats()` held one for 5 minutes, and the sql.js leak-warning interval held one for as long as the database was not closed.

### Performance

- sql.js vector upsert reuses one statement: 20k inserts 85 → 35 ms.
- db-fallback reads column names once per query: 40k rows 45 → 21 ms (5 columns), 130 → 55 ms (16 columns).
- SkillLibrary fetches hits with one `IN` query (300 hits 0.42 → 0.32 ms).

### Removed

- The unused `jsonwebtoken` dependency, and the `sharp` override that forced a vulnerable downgrade under `@huggingface/transformers` 4.3.

## [3.0.0-alpha.19] - 2026-07-29

### Fixed

- Prevented sql.js fallback instances from silently overwriting database updates
  made by another process. File-backed saves now use SHA-256 generation checks,
  WAL/journal detection, a cooperative writer lock, atomic replacement, and
  durable file/directory synchronization.
- Applied the same fail-closed persistence contract to the built-in sql.js RVF
  backend and surfaced concurrent-modification failures during shutdown.
- Reserved MCP stdout for JSON-RPC messages; CLI, database, embedding, and
  transitive controller diagnostics now remain on stderr.
- Restored reproducible clean installs by pinning available ruvllm platform
  packages and limited the release TypeScript build to publishable source.

### Tested

- Added concurrent writer, WAL/journal, corrupt-file, clean persistence, and
  two-handle regression coverage.
- Added a post-build MCP handshake/tool-call test that rejects any non-JSON
  stdout.

## [2.0.0-alpha.2.16] - 2025-12-03

### Added
- **SONA v0.1.4 Federated Learning Integration**: Complete TypeScript implementation
  - `EphemeralLearningAgent`: Lightweight distributed learning nodes (~5MB footprint)
  - `FederatedLearningCoordinator`: Central aggregation with quality-weighted consolidation
  - `FederatedLearningManager`: Multi-agent coordination with automatic aggregation
  - Quality-based filtering for high-quality agent selection
  - Large-scale federation support (50+ agents with configurable limits)
  - Comprehensive documentation in `docs/SONA_FEDERATED_LEARNING_v0.1.4.md`
  - Working examples in `examples/federated-learning-example.ts`

### Changed
- **Dependencies Updated**:
  - `@ruvector/sona`: ^0.1.3 → ^0.1.4 (federated learning features)
  - `@ruvector/gnn`: ^0.1.21 → ^0.1.22 (array conversion fixes)
  - `ruvector`: ^0.1.24 → ^0.1.29 (improved performance)

### Fixed
- **AgentDB Fast API**: Fixed `db.insert is not a function` error
  - Made `vectorBackend` public in AgentDB.ts for wrapper access
  - Updated wrapper to use `vectorBackend.insert(id, embedding, metadata)` signature
  - Fixed import path from `'agentdb'` to `'../core/AgentDB.js'`
  - Verified working at 0.09ms average insert time

- **TypeScript Compilation**: Fixed AttentionBrowser.ts WASM loader import error
  - Added `@ts-ignore` for dynamically generated WASM loader

### Performance
- **Native Attention Mechanisms**: All components verified working with native bindings
  - MultiHeadAttention: 0.018ms avg with native Float32Array
  - LinearAttention: Working with 2D arrays [seqLen][dim]
  - scaledDotProductAttention: 0.018ms avg
  - AgentDB Fast: 0.09ms avg insert time

### Documentation
- Complete federated learning guide with 5 detailed use cases
- API documentation for all 3 core federated classes
- Performance characteristics and tuning recommendations
- Migration guide from v0.1.3 to v0.1.4
- Example implementations and test results

### Tested
- ✅ EphemeralLearningAgent: Task processing, quality filtering, state export
- ✅ FederatedLearningCoordinator: Multi-agent aggregation, weighted consolidation
- ✅ Quality filtering: High-quality agent selection (threshold-based)
- ✅ Large-scale federation: 50+ agents → 20 kept (configurable limits)
- ✅ FederatedLearningManager: Complete workflow coordination

---

## [1.6.0] - 2025-10-25

### Added
- **Direct Vector Search CLI**: New `vector-search` command for raw vector similarity queries
  - Three distance metrics: cosine similarity, euclidean distance, dot product
  - Configurable k (number of results) and threshold parameters
  - JSON and table output formats
- **MMR Diversity Ranking**: Maximal Marginal Relevance algorithm for diverse result sets
  - CLI flag: `--mmr [lambda]` where lambda balances relevance vs diversity (0-1)
  - JavaScript API: `MMRDiversityRanker.selectDiverse()`
  - Works with all distance metrics
- **Context Synthesis**: Intelligent summarization and insight generation from memories
  - CLI flag: `--synthesize-context` for query and reflexion retrieve commands
  - JavaScript API: `ContextSynthesizer.synthesize()`
  - Generates summaries, patterns, insights, and actionable recommendations
- **Advanced Metadata Filtering**: MongoDB-style query operators
  - 10 operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains, $exists
  - CLI parameter: `--filters <json>`
  - JavaScript API: `MetadataFilter.apply()` and `MetadataFilter.toSQL()`
  - Nested path support for complex queries
- **Enhanced Init Command**: More initialization options
  - Custom embedding dimensions: `--dimension 384|768|1536`
  - Database presets: `--preset small|medium|large`
  - In-memory databases: `--in-memory`
- **Export/Import with Compression**: Data portability features
  - Export command with `--compress` flag (gzip compression)
  - Import command with `--decompress` flag
  - Automatic compression ratio reporting
  - Supports both .json and .json.gz formats
- **Stats Command**: Comprehensive database analytics
  - Episode, embedding, skill, and causal edge counts
  - Database size and storage metrics
  - Coverage statistics
  - Top 5 domain listing

### Fixed
- **Database Persistence**: Critical fix for sql.js WASM data persistence
  - Added `db.save()` calls after all write operations
  - Episodes, skills, and causal edges now correctly persist to disk
  - Affected methods: reflexion store, skill create, skill consolidate, causal observation
- **Browser Bundle WASM Loading**: Resolved test failures in browser environment
  - Added environment-aware path detection
  - Local `node_modules` path for Node.js testing
  - CDN fallback for browser environments
  - Quieted unnecessary WASM fallback logs
- **Package Exports**: All new controllers properly exported
  - MMRDiversityRanker accessible via package API
  - ContextSynthesizer accessible via package API
  - MetadataFilter accessible via package API
  - TypeScript types included for all controllers

### Changed
- **Documentation**: Updated to accurately reflect v1.6.0 features
  - Removed unimplemented QUIC synchronization from main features
  - Removed unimplemented HNSW performance claims
  - Clarified performance characteristics without specific benchmarks
  - Updated version references throughout documentation
- **API Improvements**: Enhanced controller interfaces
  - All controllers follow consistent patterns
  - Comprehensive error handling
  - TypeScript strict mode compliance

### Developer Notes
- **Test Coverage**: 38/38 landing page features verified (100%)
- **Backward Compatibility**: 100% compatible with v1.5.9
- **Breaking Changes**: None
- **Migration**: Drop-in replacement, no code changes required

---

## [1.5.9] - 2025-10-24

### Fixed
- Transaction handling in batch operations
- Hook integration for agentic-flow npm/npx

### Verified
- All core frontier memory features functional
- MCP integration with 29 tools operational
- Database persistence working correctly

---

## [1.3.0] - 2025-10-22

### Added
- **Learning System Tools**: 10 new MCP tools for reinforcement learning
  - Session management: start_session, end_session
  - Adaptive intelligence: predict, feedback, train
  - Analytics: metrics, explain
  - Advanced features: transfer, experience_record, reward_signal
  - 9 RL algorithms: Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, PPO, Decision Transformer, MCTS, Model-Based

- **Core AgentDB Tools**: 5 new MCP tools for advanced database management
  - agentdb_stats: Comprehensive database statistics
  - agentdb_pattern_store: Store reasoning patterns with embeddings
  - agentdb_pattern_search: Semantic pattern search with filters
  - agentdb_pattern_stats: Pattern analytics and top task types
  - agentdb_clear_cache: Cache management for optimal performance

### Changed
- **MCP Tool Count**: Increased from 14 to 29 tools
- **Documentation**: Added comprehensive learning system guides

---

## [1.2.2] - 2025-10-20

### Added
- **Core Vector DB Tools**: 5 fundamental MCP tools
  - agentdb_init: Initialize database with schema
  - agentdb_insert: Insert single vector with metadata
  - agentdb_insert_batch: Optimized batch insert with transactions
  - agentdb_search: Semantic k-NN vector search with filters
  - agentdb_delete: Delete vectors by ID or filters

---

## [1.1.0] - 2025-10-15

### Added
- **Frontier Memory Features**: 6 advanced memory patterns
  - Reflexion Memory: Episodic replay with self-critique
  - Skill Library: Lifelong learning with pattern consolidation
  - Causal Memory Graph: Intervention-based causality with p(y|do(x))
  - Explainable Recall: Provenance certificates with Merkle proofs
  - Causal Recall: Utility-based reranking (α·similarity + β·uplift − γ·latency)
  - Nightly Learner: Automated causal discovery with doubly robust estimation

- **MCP Integration**: 9 frontier memory tools for Claude Desktop
- **CLI Commands**: 17 commands for reflexion, skill, causal, learner, and recall operations
- **ReasoningBank Integration**: Pattern matching, experience curation, memory optimization

---

## [1.0.7] - 2025-10-10

### Added
- Browser bundle with sql.js WASM backend
- Universal runtime support (Node.js, browser, edge)
- Basic vector database operations

---

## Roadmap

### v1.7.0 (Planned)
- **QUIC Synchronization**: Real-time multi-node coordination
  - Architecture designed and documented
  - Implementation in progress
  - 8-week development timeline estimated

### v2.0.0 (Future)
- **HNSW Indexing**: Hierarchical Navigable Small World graphs
  - 10-100x performance improvements for large datasets
  - Sub-millisecond search at 100K+ vectors
- **Hybrid Search**: Combined vector + keyword search
- **Quantization**: 4-bit, 8-bit vector compression for memory efficiency
- **Multi-database Coordination**: Cross-instance synchronization
- **Performance Benchmarking Suite**: Comprehensive real-world performance metrics

### Under Consideration
- Distributed consensus protocols (Raft, Byzantine)
- CRDT-based conflict-free synchronization
- WebTransport for browser-to-browser sync
- GPU acceleration for vector operations
- Streaming query results
- Advanced caching strategies

---

## Version Support

- **v1.6.x**: Active development, recommended for production
- **v1.5.x**: Maintenance mode, security fixes only
- **v1.3.x**: Legacy support, critical fixes only
- **v1.0-1.2**: No longer supported

---

## Migration Guides

- [Upgrading from v1.3.0 to v1.6.0](docs/V1.6.0_MIGRATION.md)
- [Upgrading from v1.2.2 to v1.3.0](MIGRATION_v1.3.0.md)
- [Upgrading from v1.2.1 to v1.2.2](docs/MIGRATION_v1.2.2.md)

---

**Note**: This CHANGELOG was created on 2025-10-25 to provide accurate version history. Prior to v1.6.0, version documentation was inconsistent. All features and fixes have been verified against actual implementation and test results.
