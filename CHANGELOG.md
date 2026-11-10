# Changelog

All notable changes to AgentDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.9.0] - 2026-11-10

### Added

- **Durable weight-space programs:** `WeightPrograms` from `agentdb/controllers/WeightPrograms` stores dense checkpoints, evaluates caller-authored weight patches with real WASM, persists internal proposal evidence, and promotes accepted proposals by ID. Model generation, base hash and revision checks protect concurrent updates and deleted/recreated programs. Rollback restores a retained checkpoint as a new revision; history retains 32 checkpoints and proposals retain at most 16 entries with a 24-hour expiry.
- **Six MCP tools:** `agentdb_wsp_create/get/list/propose/promote/rollback` validate strict nested schemas. Reads work in readonly/retrieval profiles, creation/evaluation/promotion require learning, and rollback requires admin or an explicit allow. Returned evaluation receipts do not authorize promotion. Evaluation data is caller-supplied; acceptance does not prove unseen-data generalization. No provider or gradient training runs.
- Real native SQLite, sql.js and stdio tests cover hand-authored XOR construction/repair, rejected proposals, mutation isolation, forged receipts, concurrent staleness, ABA, persistence, checkpoint integrity and rollback.

### Changed

- Exact required dependencies: `ruv-fann-wsp@0.1.0` and `ruv-fann@0.1.2`. Existing `FannModels` checkpoints and APIs remain independent. Protected packed-consumer qualification now exercises durable WSP proposals and MCP workflows alongside existing neural training.

## [3.8.2] - 2026-11-10

### Changed

- **Exact pins replace the remaining ruvector `^0.1.x` ranges:** `@ruvector/rvf-wasm` 0.1.13 (was ^0.1.7), `@ruvector/rvf-solver` 0.1.11 (was ^0.1.7) and `@ruvector/rabitq-wasm` 0.1.2 (was ^0.1.2). Each is the latest published version. All three stay optional and lazily imported, with their fallbacks unchanged. A new test covers each through the path agentdb uses: rvf-wasm `init()` resolves to the RVF exports, `AgentDBSolver.create()` runs on rvf-solver, and `loadRabitq()` returns `RabitqIndex.build` from the rabitq-wasm node entry.

### Fixed

- **CLI write commands exited 1 after a successful write on native SQLite.** `reflexion store`, `skill create`, `skill consolidate`, `causal experiment create` and `causal experiment add-observation` called `this.db.save()`, which only the sql.js wrapper has. With better-sqlite3 (used when it loads since 3.7.7), the row was written, then the command printed `this.db.save is not a function` and exited 1. `save()` is now called only when it exists, as the CLI's shutdown path already did.
- **README Quick Start runs as written.** Path A imported `SelfLearningRvfBackend` from the package root, which does not export it, and used an undefined `queryEmbedding`. It now imports from `agentdb/backends/self-learning` and sets `dimension: 384`. Path B used `add` and `search` commands that don't exist. It now uses `init`, `reflexion store` and `reflexion retrieve`. `tests/unit/readme-quickstart.test.ts` pulls both paths out of README.md and runs them against the build.

## [3.8.1] - 2026-11-10

### Fixed

- **Pi 4 / Cortex-A72 crash floors.** The optional native packages are raised to the first builds whose linux-arm64 binaries contain no `LDAPR` instructions (an ARMv8.3 instruction that Cortex-A72, the Raspberry Pi 4 CPU, does not have). A SIGILL kills the process, so agentdb's `try { await import() }` fallbacks cannot catch it. The old floors still let a lockfile, an offline cache or a parent package's pin resolve the crashing builds. This repo's own lockfile had resolved `@ruvector/attention` 2.2.3 and `@ruvector/core` 0.1.30. A fresh install of 3.8.0 already picked up the fixed builds through its ranges. Measured on a `npm install --cpu=arm64 --os=linux` tree, loading each package with node 22 arm64 under `qemu-aarch64 -cpu cortex-a72`:

  | package | old floor | LDAPR | load on A72 | new floor | LDAPR | load on A72 |
  |---|---|---|---|---|---|---|
  | `@ruvector/attention` | 2.2.3 | 305 | SIGILL | ^2.3.1 | 0 | ok |
  | `@ruvector/graph-node` | 2.1.2 | 2464 | SIGILL | ^2.1.4 | 0 | ok |
  | `@ruvector/rvf-node` | 0.4.5 | 100 | SIGILL | 0.4.7 | 0 | ok |
  | `@ruvector/sona` | 0.1.11 | 34 | SIGILL | 0.1.13 | 0 | ok |
  | `@ruvector/gnn` | 0.1.23 | no arm64 binary | not found | 0.1.28 | 0 | ok |
  | `@ruvector/router` | 0.1.15 | no arm64 binary | not found | 0.1.34 | 0 | ok |
  | `@ruvector/ruvllm` | 4.1.0 | 0 | ok | ^4.2.2 | 0 | ok |
  | `@ruvector/rvf` | 0.4.10 | (JS, uses rvf-node) | | 0.4.12 | | |
  | `ruvector` | 0.3.5 | | | 0.3.20 | | |

- **`@ruvector/core` 0.1.38 is now declared** (optional). agentdb imports it directly in the backend factory, detector, `RuVectorBackend` and `NativeAccelerator`, but only got it through `ruvector`, whose range (`^0.1.36`) still admits builds before 0.1.38. 0.1.38 stops re-inserted ids from returning ghost HNSW results and rejects NaN/Inf vectors.
- **The WASM attention fallback now loads in Node.** `ruvector-attention-wasm` is a wasm-bindgen web build, and its default init `fetch()`es a `file://` URL, which Node rejects ("fetch failed"). So `AttentionService` never used WASM in Node, on 0.1.32 or 2.x. Under Node it now reads the `.wasm` bytes and calls `initSync`. Browsers keep the default init.

### Changed

- **`ruvector-attention-wasm` ^0.1.32 → ^2.1.0.** A `^0.1.x` range never reached the 2.x line. The three classes agentdb constructs (`WasmFlashAttention(dim, blockSize)`, `WasmHyperbolicAttention(dim, curvature)`, `WasmMoEAttention(dim, experts, topK)`) and their `compute(query, keys, values)` signatures are unchanged. Speed is about the same (dim 64, 32 keys, 2,000 calls, per call, 0.1.32 vs 2.1.0): flash 342 vs 313 us, hyperbolic 800 vs 761 us, MoE 322 vs 307 us. That is within run-to-run noise, so this release makes no performance claim.
- The 0.x edges (`@ruvector/core`, `gnn`, `router`, `sona`, `rvf`, `rvf-node`, `ruvector`) are exact pins instead of `^0.x` ranges, as `ruv-fann` already was. `@ruvector/attention`, `graph-node`, `ruvllm` and `ruvector-attention-wasm` use caret ranges on their major. All of these stay optional.

### Tests

- `tests/unit/optional-native-floors.test.ts` (in the release test list): checks each declared floor; checks that each installed package satisfies its floor and imports; with the NAPI binding unavailable, `AttentionService.flashAttention` runs on the `wasm` backend (this fails on 3.8.0's loader); and checks that the Flash, Hyperbolic and MoE WASM classes compute finite output.

## [3.7.7] - 2026-11-09

### Fixed

- **A failed native `better-sqlite3` load is no longer reported as success** (ruvnet/ruflo#3175, #58). After a Node ABI change (e.g. a reused npx cache), `getDatabaseImplementation()` used to swallow the load error and print a green "✅ Using sql.js" line. It now prints one yellow warning on stderr. For `NODE_MODULE_VERSION` / `ERR_DLOPEN_FAILED` the warning names both module versions and gives the fix (`npm rebuild better-sqlite3` or clear the npx cache). A missing `better-sqlite3` package still gets a neutral info line. `getDatabaseInfo()` adds `fallback`, `nativeStatus` and a read-only `nativeError: { code, message }`. sql.js is still used as the fallback.

## [3.7.6] - 2026-11-09

### Changed

- **`@ruvector/rvf` ^0.4.10 and `@ruvector/rvf-node` ^0.4.5** (were ^0.4.9 / ^0.4.4). rvf 0.4.10 re-links a node upserted in place and applies a batch that names an id twice last-write-wins.
- **`updateEpisode`/`updateSkill` no longer `remove()` before re-inserting on RVF.** 3.7.4 kept the remove because in-place upserts lost recall. Same churn as 3.7.4 through `RvfBackend` (3,000 x 384 cosine, 300 updates, 200 queries, recall@10 vs exact), in-place vs remove + insert:

  | data (seed) | rvf 0.4.9 | rvf 0.4.10 |
  |---|---|---|
  | lcg 12345 | 0.9675 vs 0.9955 | **1.0** vs 0.9955 |
  | lcg 777 | 0.966 vs 0.9965 | **1.0** vs 0.9965 |
  | mulberry32 1 | 1.0 vs 1.0 | 1.0 vs 1.0 |
  | mulberry32 2 | 0.999 vs 0.9995 | **1.0** vs 0.9995 |
  | mulberry32 3 | 1.0 vs 0.9995 | 1.0 vs 0.9995 |

  In-place is at least remove + insert on every seed on 0.4.10. (lcg 4242 is a degenerate data set, about 0.39 on both paths and both versions, 0.3975 vs 0.3885 on 0.4.10.) Update latency, median per update: 3.6-3.9 ms before (remove + insert, 3.7.5 on 0.4.9), 2.9-3.4 ms now (in-place on 0.4.10); p95 4.4-4.7 ms before, 3.5-4.2 ms now. An in-place upsert costs about 3.1 ms on 0.4.10 against 2.2 ms on 0.4.9; that is the re-linking.
- The skip is gated on a new optional `VectorBackend.upsertsInPlace` flag, set by `RvfBackend` and `SelfLearningRvfBackend`. `HNSWLibBackend` rejects a re-inserted id and the other backends keep remove + insert.
- `RvfBackend` still collapses a repeated id in a batch to its last write. rvf 0.4.10 does the same on its own (tested), so on the supported floor this changes nothing; it costs one Map pass and keeps the write queue from wedging if an override or lockfile resolves an older rvf.
- The call-order guarantees from 3.7.4/3.7.5 (one chain for writes, deletes and flushes; `remove()` drops queued writes) are unchanged.

### Fixed

- **`getStats().count` no longer grows on updates.** rvf reports an in-place upsert of a stored id as `accepted: 1`, and `RvfBackend` added that up, so the sync count rose by one per update until the next `getStatsAsync()` (5 for a store holding 2). The count is now read back from rvf after each ingest (`status()`, about 3 us).

### Tests

- `tests/backends/rvf-049-upsert-lock.test.ts`: the recall test now asserts in-place >= remove + insert and >= 0.99 (fails on 0.4.9: 0.966 vs 0.9985); a raw rvf batch with a repeated id stores the last write and later batches still work (fails on 0.4.9: InvalidManifest); an upsert does not change `getStats().count`.
- `tests/unit/controllers/crud-adr017.test.ts`: on a backend with `upsertsInPlace`, `updateEpisode` and `updateSkill` replace the vector without calling `remove()`.
- The meta-snapshot cost per commit in rvf (O(n)) is not fixed in 0.4.10; nothing here depends on it.

## [3.7.5] - 2026-11-09

### Fixed

- **RVF writes and deletes now reach rvf in call order.** In 3.7.4 the order could still break in three ways, each of which removed the new copy of an id:
  - a `remove()` then `insert()` made while a flush was in flight: the running flush picked up the new write before the chained delete ran;
  - a `remove()` followed by `insertAsync()`;
  - a `remove()` followed by `insertBatchAsync()`.

  A flush now covers only the writes queued before it was called. `insertAsync`, `insertBatchAsync`, deletes and flushes share one chain. The auto-flush at the batch threshold made the first case reachable from `updateEpisode`/`updateSkill`.

### Corrected

- The 3.7.4 entry said its 384-d recall@10 figures used an efSearch floor of 100. `RvfBackend` applies 444 at 384-d (`minEfSearch`), which is what was measured; the entry now says 444.

## [3.7.4] - 2026-11-09

### Fixed

- **The same id written twice before a flush no longer wedges the RVF write queue.** rvf (0.4.7 and 0.4.9) rejects an `ingestBatch` that names an id twice (`InvalidManifest`, 0x0105). A sync `insert()` of an id that was already queued, which is what `updateEpisode`/`updateSkill` do right after a store, made every later `flush()` and `searchAsync()` fail until the store was closed, and the queued writes were then lost. A batch now keeps each id's last write; `insertBatchAsync` with a repeated id does the same.
- **`remove()` is ordered with queued writes.** It drops queued writes of the id and deletes the stored copy after any flush in flight, and `searchAsync()` waits for that delete. Before, the delete was fire-and-forget: an `insert()` then `remove()` before a flush left the id in the store, and the controllers' update path (`remove()` then `insert()` of a stored id) could delete the new copy. `removeAsync()` returns `true` for an id that was only queued.

### Changed

- **`@ruvector/rvf` ^0.4.9 and `@ruvector/rvf-node` ^0.4.4** (were ^0.4.7 / ^0.4.2). Measured through `RvfBackend`, 2,000 x 384 cosine:
  - re-ingesting a live id no longer drops the HNSW index: the query after an upsert takes 0.41 ms (median; 0.4.7: 1,686 ms, a full rebuild). Upsert itself: 5.0 ms (0.4.7: 41 ms).
  - a writer killed with SIGKILL no longer holds the lock: reopen takes 7-14 ms on the first attempt (0.4.7: 30,048 ms, 300 attempts).
  - recall@10 vs exact at 5,000 x 384 (efSearch floor 444): 0.868 with no updates; 0.806 after 400 remove + insert updates and 0.792 after 400 in-place upserts.
- The controllers keep calling `remove()` before re-inserting an id. On the same churn (3,000 x 384, 300 updates, 200 queries), in-place upserts reached recall@10 0.966 where remove + insert reached 0.9985 and 0.9965 on two of three seeds, so the workaround is kept until rvf closes that gap.
- agentdb has no lock retry or sleep loop to remove: a live second writer still gets the `locked by another writer` error.

### Tests

- `tests/backends/rvf-049-upsert-lock.test.ts` (real package, in the Library workflow): upsert keeps the index, a SIGKILLed writer's store reopens within 2 s, the queue cases above, and recall after churn for both update paths. On rvf 0.4.7 the first two fail (1,702 ms; no reopen within 5 s). On 0.4.9 without this release's `RvfBackend` changes the four queue tests fail.
- `tests/regression/core-features.test.ts` runs again (it passed `dimensions` to `EmbeddingService`, which takes `dimension`, and did not await `addCausalEdge`).

## [3.7.2] - 2026-11-08

### Fixed

- **In-memory RVF stores that are never closed no longer leak temp directories.** Since 3.7.1 each in-memory `RvfBackend` owns a private temp directory, removed on close; a process that exited without closing left it behind (two test files alone left 37 per run). Open in-memory stores are now tracked and removed synchronously when the process exits or receives SIGINT/SIGTERM. One set of listeners is registered for all instances (not one per instance). The cleanup is idempotent, and when the host has no handler of its own for the signal, the signal's default action (terminate) still runs. Stores with an explicit `storagePath` are never touched.
- Closing an in-memory store removes its directory synchronously (it was a deferred dynamic import).
- `tests/backends/rvf-backend-extended.test.ts` and `rvf-flush-retry.test.ts` close every store they open.

## [3.7.1] - 2026-11-08

### Fixed

- **Personalized PageRank scores sum to 1.** The JavaScript fallback in `SparsificationService` dropped the mass of nodes with no out-edges, so its scores summed to about 0.73-0.86 on graphs with sinks (0.729 at n=1,000 and 0.756 at n=10,000 on a random graph with 10% sinks). Dangling mass now returns to the source, as in standard PPR.
- **In-memory RVF stores are in memory.** An `RvfBackend` with no `storagePath` opened `./agentdb.rvf` in the working directory, so an in-memory `AgentDB` on the rvf backend reopened whatever an earlier run left there and shared it with every other in-memory instance. `storagePath: ':memory:'` created a file literally named `:memory:` (plus `:memory:.idmap.json`). rvf has no in-memory mode, so both now use a fresh private temp directory per instance, removed by `closeAsync()` (and by `close()` once rvf releases the file). Nothing is written to the working directory unless a path is given. `AgentDB.close()` now awaits `closeAsync()` when the backend has it.
- **The fallback is linear in the graph size.** It scanned every adjacency list for every node (`O(n^2 * degree)` per iteration): 87 ms at n=1,000 and 11.9 s at n=10,000. It now pushes over out-edges in float64: 0.25 ms and 2.9 ms at the default 20 iterations.

### Changed

- **`@ruvnet/sublinear-core` ^1.0.1** (new optional dependency, zero dependencies, lazy-loaded) is the first PPR backend. Order: sublinear-core, then `@ruvector/graph-transformer` (NAPI / WASM), then the in-tree fallback. sublinear-core and the fallback compute the same thing (agreement within 1e-12 at the same budget; within 1e-9 of a converged power iteration). n=1,000 / 10,000: 0.53 ms / 4.8 ms.
- `result.method` can be `'ppr-sublinear-core'`, and `getPprRuntime()` can return `'sublinear-core'`.
- `convergenceThreshold` is now compared with the L1 change between iterations (was the largest per-node change), the same rule as sublinear-core. `metadata.convergence` reports that L1 change. Defaults (20 iterations, 1e-6) are unchanged.
- Out-of-range neighbour ids are ignored by every PPR path (the old fallback counted them in the out-degree and lost their share).

### Known

- `@ruvector/graph-transformer` (used only when sublinear-core is not installed) keeps a sink's mass on the sink and always runs 20 iterations, so on graphs with sinks its scores differ from standard PPR (L1 ≈ 0.5 with 10% sinks). On graphs without sinks it agrees with the fallback.

## [3.7.0] - 2026-11-06

### Changed

- **`@ruvector/rvf` ^0.4.7** (was ^0.2.3), `@ruvector/rvf-node` ^0.4.2 (was ^0.1.7), `@ruvector/rvf-wasm` ^0.1.7. Still optional and lazy-loaded; without it `createBackend('rvf')` falls back to the sql.js RVF backend. This brings the 0.4.x filtered-recall fix, the #1382 file-growth fix and `forceExact`.
- **Metadata is stored.** rvf 0.2.3 dropped the `metadata` passed to `RvfBackend.insert`/`insertAsync`/`insertBatchAsync`; 0.4.7 stores it with the vector and search hits carry it, including after reopen. Numbers are stored as f64, objects and arrays as JSON.
- **Filtered search works.** On 0.2.3 every `searchAsync` with a `filter` threw (`filter requires string field 'valueType'`). `FilterBuilder` now emits leaves as `{ op, field: <name>, value }`, which rvf resolves against the store's own field registry; it no longer emits per-instance numeric `fieldId`s, which never matched that registry. `fieldId()`/`getFieldMap()` remain but are deprecated. `RvfFilterExpr` accepts either `field` or `fieldId` leaves.
- `$range` / `range()` stay inclusive (`low <= v <= high`). rvf's native `range` excludes `high`, so they are emitted as `and(ge, le)`.
- A raw `RvfFilterExpr` passed as `SearchOptions.filter` is forwarded unchanged (it was run through the predicate DSL).
- New `SearchOptions.forceExact`: exact brute-force scan on the RVF backend; other backends ignore it.
- **Metadata type conflicts throw.** A field keeps the type of its first value; writing another type now throws `RvfMetadataTypeConflictError` (`code` 0xff09, `field`, `ids`) instead of being silently dropped. rvf applies a batch all-or-nothing, so `insertAsync`/`insertBatchAsync` write nothing on a conflict. For queued sync `insert()`s, `flush()` writes the other entries, keeps the conflicting ones with their data in `getRejectedWrites()`/`takeRejectedWrites()`, and throws. Both are exported from `agentdb/backends`.
- `insertBatchAsync` validates ids, dimensions and metadata like `insert` (prototype-pollution keys are stripped before metadata is persisted).
- **Locking.** rvf 0.4.x holds a writer lock until close completes. New `closeAsync()` flushes queued writes and resolves once the lock is released. A new `RvfBackend` on a path whose sync `close()` is still in flight waits for it. A second writer gets an error that says the store is locked and points to `closeAsync()`/`openReadonly()`.
- **Dimension-aware ef floor.** Above 128 dims, queries use `efSearch >= ceil(256 * sqrt(dim / 128))` (384-d: 444). rvf 0.4.7 floors ef at 256 for every dimension; on 5k × 384-d cosine, recall@10 at efSearch 100 was 0.965 and is 0.995 with the floor. This mirrors RuVector #1401, which is not yet in a release. Larger caller values are kept.

### Migration

- Stores written by agentdb 3.6.x / rvf 0.2.3 open and read under 0.4.7: all vectors and string ids are intact (fixture in `tests/fixtures/rvf-0.2.3`), and new writes with metadata and filters work on them. Legacy rows have no metadata, because 0.2.3 never stored it.
- rvf 0.2.x did not persist the distance metric, so a 0.2.x store reopens as L2 (squared) under 0.2.3 itself and under 0.4.7. For unit-normalized embeddings the ranking is the same as cosine, but `distance`/`similarity` values and `threshold` filters differ. To get cosine back, re-ingest into a new store; stores created with rvf 0.4.x keep their metric.
- A store opened for writing by 0.4.7 may not be readable by 0.2.x again.

## [3.6.1] - 2026-11-06

### Fixed

- **`simulation/tests/latent-space/self-organizing-hnsw.test.ts` no longer hangs.** The scenario picked each node's neighbors by scanning and sorting every vector, so building the graph was O(n² log n); at the configured 100k/1M nodes (`AGENTDB_BENCH_FULL=1`, or any checkout before #44) it never finished. The loop was synchronous, so the 600 s hook timeout could not fire and the run held its vitest forks at 100% CPU (one host: four forks for 10 h). Neighbors are now chosen from at most 512 sampled candidates, the configured `graphSizes` are `[5000, 15000]`, and the scenario yields to the event loop and throws once its 540 s budget (`SIMULATION_TIME_BUDGET_MS`) is spent. With `AGENTDB_BENCH_FULL=1` the file now finishes in 179 s on a 32-thread host at load average 42; the default run takes about 10 s. The test's assertions get a 10 s per-test timeout.
- `vitest.bench.config.ts` runs one file at a time (`fileParallelism: false`): the suites are CPU-bound and measure speed, and a runaway suite now holds one core. Both vitest configs set `teardownTimeout: 30000`.
- `WASMVectorSearch` dropped a `require.resolve('@ruvector/wasm/reasoningbank_wasm.js')` probe. `require` is undefined in this ES module, so the probe always threw into its catch, and `@ruvector/wasm` does not ship that file. The other WASM search paths are unchanged.

### Security

- `ui/` (the demo app, not part of the npm package): `@remix-run/router` 1.23.0 → 1.23.4 and `react-router`/`react-router-dom` 6.30.1 → 6.30.6 via `npm audit fix` (high: open redirect via protocol-relative paths). `vite` and `zod` advisories in `ui/` need major upgrades and are not changed here.

### Not changed

- `@ruvector/rvf` stays at `^0.2.3`. 0.4.x rejects per-vector metadata in `ingestBatch` (`RvfErrorCode.MetadataNotSupported`, ruvector#704), and `RvfBackend.insert` passes metadata, so a plain insert-with-metadata fails on 0.4.6.

## [3.6.0] - 2026-11-06

Follow-ups to 3.5.0 (#43, #44, #46).

### Changed

- **`AttentionService` validates its inputs** (#43). `multiHeadAttention`, `flashAttention`, `flashAttentionV2`, `linearAttention`, `hyperbolicAttention`, `fusedAttention` and `moeAttention` now reject, instead of returning output for:
  - an empty query, key or value;
  - a buffer whose length is not a whole number of `embedDim` rows;
  - key and value with different row counts (query may differ: cross-attention). Previously this returned output; it now throws `dimension mismatch`;
  - NaN or ±Infinity anywhere in the input, and a non-finite hyperbolic `curvature`;
  - more than `maxAttentionScores` scores (queryRows × keyRows × numHeads, default 2^25) for the quadratic mechanisms. A 2048 × 2048 × 12-head call used to run for minutes and end in a stack error; it now fails at once with a size-limit error. `linearAttention` is not capped. Raise the limit with the new `maxAttentionScores` config option.
  - any call after `dispose()` (previously the service silently re-initialized).
  The config must satisfy `embedDim === numHeads × headDim` with positive integers. `sparseAttention` and `partitionedAttention` check their query for emptiness and finiteness only; their internal dense fallback is unchanged, so a query sized to the node count still works.
- `AttentionService` times each call with `performance.now()` instead of shared `performance.mark` names, so concurrent calls and concurrent `initialize()` across instances no longer read each other's measurements.
- **`@ruvector/sona` ^0.1.11** (was ^0.1.4). 0.1.11 only adds `SonaEngine.forceLearnAsync`.
- Benchmark and simulation suites (`benchmarks/`, `simulation/`, `tests/benchmarks/`) are no longer part of the default `vitest` run (#44). `npm run test:bench` and the `benchmark:adr072*` scripts run them with `vitest.bench.config.ts`. The suites were brought up to date: the ADR-072 graph generator emits the adjacency array `SparsificationService`/`MincutService` accept (it built a COO object they never took); `ruvector-performance` uses the current exports (`VectorDb`, `kHopNeighbors`, `differentiableSearch()`) and no longer hangs on a swallowed error; the simulation tests read the fields the scenarios actually report. Absolute speed thresholds are skipped when `CI` (or `AGENTDB_SKIP_SPEED_GATES=1`) is set; long simulations use small sizes unless `AGENTDB_BENCH_FULL=1`. Tests the scenarios cannot satisfy are skipped with the reason. With `CI=1` all 12 files pass (287 passed, 16 skipped).
- Claude Code plugin 3.6.0: `.mcp.json` starts `agentdb@3.6.0` (was 3.4.0).

### Fixed

- `RvfBackend.flush()` removed queued writes before the ingest ran, so a failed `ingestBatch` (including the fire-and-forget auto-flush from `insert()`) dropped them. Items now leave the queue only after their batch is accepted, and flushes are serialized so overlapping flushes don't ingest an item twice (#46).

### Not changed: `@ruvector/rvf` stays ^0.2.3

Re-evaluated against 0.4.3 (#46). Measured with the 0.2.3 and 0.4.3 SDKs side by side on the native backend:

- 0.4.3 still rejects non-empty per-vector metadata (`RvfError` 65284, "Per-vector metadata is not yet supported by this SDK (see issue #704)"). 0.2.3 accepts it and silently drops it, so no metadata is lost by moving, but every `metadata:` passed to `RvfBackend` would have to be stripped and the metadata filter path stays unusable.
- The first 0.4.3 write to a 0.2.3 store still makes it unreadable to 0.2.3 (`Segment hash verification failed: InvalidChecksum`). A store 0.4.3 creates is also unreadable to 0.2.3.
- 0.4.3 persists the cosine metric for a store it creates (distance 0 for a scaled copy of the query), but a 0.2.3 store upgraded in place keeps scoring with squared L2 (distance 5 for the same pair, before and after the 0.4.3 write).
- The SDK has no call that returns stored vectors: `query()` returns `{ id, distance }` and `exportBytes()` returns the file. A read-old, write-new migration, which is the only way to give old stores the cosine metric, can't be written against it.

So moving now would be a one-way format change that doesn't fix old stores' metric, and still drops metadata. The move waits for an rvf release with durable metadata (#704) and a way to read vectors back. Note that `ruvector` 0.3.12+ (an optional dependency) already installs its own nested `@ruvector/rvf` ^0.4.2; agentdb's `RvfBackend` doesn't use it.

## [3.5.0] - 2026-11-05

Dependency floor release. agentdb is a dependency of agentic-flow and ruflo, so its ranges decide what the ecosystem installs.

### Breaking

- **Node.js 20.9.0 or newer** (`engines.node`, `agentdb doctor`). Node 18 is end of life, and sharp 0.35.5 (the GHSA-wq5f-xc86-pv6w fix, reached through `@huggingface/transformers`) and better-sqlite3 12 refuse it (#35).
- **`@ruvector/ruvllm` ^4.1.0** (was ^2.5.1). 4.1.0 is the restored runtime; 4.0.0 and 4.0.1 were a wrong-repository publish and are excluded. RuvLLM 4 throws `RUVLLM_NO_LANGUAGE_MODEL` from `query()`/`generate()` when no model is loaded instead of returning placeholder text, so `LLMRouter` uses ruvllm for text generation only when `isModelLoaded()`; embeddings, similarity and memory still use it.

### Added

- `SparsificationService` and `MincutService` accept `GraphInput`: the adjacency array, or an object keyed by node id (`{ 0: [1, 2] }`) as the docs showed. `normalizeGraphEdges()` converts and validates it.
- `tests/unit/integrations/ruvllm-api.test.ts` requires the real `@ruvector/ruvllm` and asserts the installed version and every symbol agentdb calls.
- CI job that installs with `--omit=optional` and type-checks.

### Changed

- `@ruvector/attention` ^2.2.3 (was ^0.1.2, which already resolved 0.1.32 on fresh installs; that release aborts Node on a Rust panic, 2.2.3 throws). `@ruvector/graph-node` ^2.1.2 (Cypher depth fix). `better-sqlite3` ^12.8.0: 11.x has no Node 24 prebuild and its local builds abort when a statement is garbage-collected on Node 24.19+ (#29). `semver` is now a declared dependency.
- `@ruvector/rvf` stays ^0.2.3. rvf 0.4 can open a 0.2 store, but its first write makes the file unreadable to 0.2/0.3, and its SDK rejects per-vector metadata. Moving to it needs a migration path (back up `.rvf` and `.rvf.idmap.json`), so it is left for a dedicated release.
- The TypeScript build no longer depends on optional dependencies' type declarations (`src/types/optional-modules.d.ts`); a failed optional install no longer fails `tsc` with TS2307.

### Fixed

- `LLMRouter`: `getEmbedding()` always returned an empty vector (`embedding.values` on an array is `Array.prototype.values`); ruvllm response text was read from fields that do not exist; feedback used a shape ruvllm rejects.
- `FederatedSessionManager`: LoRA adapters ignored the configured dimension and were 256 x 256, so `applyLora()` changed the embedding length.
- `NativeAccelerator.ewcUpdateFisher()` called a method ruvllm never had and always returned false; it now registers the weights with `EwcManager.registerTask()`. Native tensor compression works with `@ruvector/gnn` 0.1.25 (`TensorCompress`).
- `SemanticQueryRouter`: with `@ruvector/router` 0.1.32 (which now loads natively), a threshold of 0 became 0.7 and dropped matches.
- `AttentionService` never reached the native `@ruvector/attention` classes from ESM (they sit on `default`), and called the removed `computeRaw`.
- `reasoning_patterns` gains the `context` and `outcome` columns `BatchOperations.insertPatterns` writes, with a migration for existing databases; `agentdb_pattern_store_batch` always failed (#17, fixes #16).
- `getRecentEpisodes` returned episodes stored in the same second in arbitrary order.
- Multi-head attention with fewer dimensions than heads produced all-zero heads.
- Importing `agentdb/cli` ran the CLI and exited the host process.
- CLI `vector-search` ignored its database path and decoded 384-dim embeddings as 1536 bytes; `import` dropped embeddings written by `export`.
- The SQL validator accepts `table_info` only as `table_info(identifier)`.
- `HNSWLibBackend` did not await hnswlib-node 3's async index read/write, refused to re-insert a removed id, and recorded sub-millisecond searches as 0 ms. `HNSWIndex.search` threw when k exceeded the index size; k is now capped.
- `agentdb init` from a source checkout looked for schema files that only exist in the built package and created a database without the `episodes` table; it now uses the inlined schemas.
- Test suite: 38 sparsification tests passed object graphs to the array API; other suites used config keys, controllers and imports (`@jest/globals`) that do not exist. Plugin and real-browser suites are excluded from the default vitest run.

## [3.4.0] - 2026-11-05

Ecosystem integrations (docs/adrs/ADR-018). Native ruvector packages now back code paths that previously probed for packages that do not exist. No API removals.

### Added

- **Exact minimum cut via `@ruvector/mincut-wasm`.** `MincutService` (`algorithm: 'stoer-wagner'`) returns the true global min cut (`algorithm: 'stoer-wagner-wasm'`) for graphs up to `wasmMaxNodes` (default 2000) and `wasmMaxEdges` (default 20,000). Before, it split node ids in half: on 1,000 nodes that cut 1,003 edges where the minimum is 4. `getBackend()` reports which path ran.
- **Native Personalized PageRank via `@ruvector/graph-transformer`** (NAPI), with `ruvector-graph-transformer-wasm` as the WASM rung. `SparsificationService` PPR is 40x (500 nodes) to 780x (5,000 nodes) faster, with scores within 4e-7 of the JavaScript path. `getPprRuntime()` reports `napi`, `wasm` or `fallback`.
- **Opt-in RaBitQ index for `SqlJsRvfBackend`** (`quantizedIndex: 'rabitq'` or `AGENTDB_QUANTIZED_INDEX=rabitq`) via `@ruvector/rabitq-wasm`. At 50,000 384-dim vectors, a query takes 5.2 ms instead of 33.3 ms with recall@10 1.000. Similarities stay exact: candidates are rescored. Writes after the index is built are scanned exactly until the next rebuild. `SqlJsRvfBackend` is now exported from `agentdb/backends`.
- **Opt-in prompt-injection screening of memory writes** via `aidefence` (`AGENTDB_INJECTION_GUARD=warn|block`, `setInjectionGuard()`, `scanForInjection()`). Covers episode, pattern and skill creates, updates and batch inserts, and therefore every MCP write tool for them. `block` throws `ValidationError` `INJECTION_DETECTED` and fails closed if aidefence is missing.
- Skills `agentdb-graph-partition`, `agentdb-quantized-search` and `agentdb-memory-guard` for Claude Code (in the agentdb-graph, agentdb-search and agentdb-memory plugins) and Codex (`.agents/skills/`). The agentdb plugin (3.4.0) gains the `injection_guard` option.
- `npm run benchmark:ecosystem`.

### Changed

- `@ruvector/graph-transformer` moves from `dependencies` to `optionalDependencies`. New optional dependencies: `@ruvector/mincut-wasm`, `@ruvector/rabitq-wasm`, `aidefence`.
- `stoer-wagner` partitions are now minimum cuts, often one low-degree node against the rest, not halves. Native PPR keeps the mass of nodes with no out-edges, so scores differ from 3.3.0 on graphs with such nodes.

### Fixed

- `MincutService` cached cuts by node and edge count only, so a different graph of the same size got the previous graph's cut.
- `MincutService.partition()` ignored `maxPartitionSize`. It now splits larger parts into equal chunks, which keeps `partitionedAttention` bounded.

## [3.3.0] - 2026-11-01

Full CRUD for episodes, reasoning patterns and skills (docs/adrs/ADR-017). No API removals.

### Added

- **Get, list and update for every record type, delete for patterns and skills.** Library: `getEpisode`, `listEpisodes`, `updateEpisode`; `listPatterns`, `updatePattern`; `getSkill`, `listSkills`, `updateSkill`, `deleteSkill`. MCP: `agentdb_get`, `agentdb_list`, `agentdb_update`, `agentdb_pattern_get`, `agentdb_pattern_list`, `agentdb_pattern_update`, `agentdb_pattern_delete`, `skill_get`, `skill_list`, `skill_update`, `skill_delete`.
- **Updates keep the id and every index consistent.** The SQL row, embedding, vector-index entry, caches and BM25 keyword index move together in one transaction. The embedder runs only when embedded text changes, so a metadata, tag or reward edit is ~30x cheaper than the old delete + insert (0.038 ms vs 1.13 ms p50 with MiniLM). Usage statistics stay with the feedback tools and cannot be overwritten by an edit.
- `Bm25Index.replace()`: in-place keyword-index edits, so the first hybrid search after an update no longer pays a full rebuild (33.4 ms vs 46.8 ms p50 at 10,000 episodes).
- `npm run test:mcp-crud` (stdio end-to-end on both SQL drivers), `npm run benchmark:crud`, and a `Library` CI workflow.

### Fixed

- `deleteEpisode` never removed the episode from the vector index (called `delete()`, the interface has `remove()`).
- `deleteEpisode("12-notes")` deleted episode 12 (`parseInt`); ids are now strict integers everywhere.
- `deletePattern` left the pattern in the vector index.
- `pruneEpisodes`, `pruneSkills`, `agentdb_delete` with filters and `agentdb_delete_batch` left orphaned embeddings (and skill links).
- `agentdb_init` failed on better-sqlite3 (double-quoted SQL string literal).

### Security

- `agentdb_insert` / `agentdb_insert_batch` validate text, session id, tags and metadata (size cap, no `__proto__`/`constructor`/`prototype` keys).
- Error responses no longer include stack traces.
- New delete tools are admin-only; get/list are read-only; updates are in the `learning` profile. The plugin mod screens updates for secrets and refuses the new deletes unless `allow_destructive` is on.

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
