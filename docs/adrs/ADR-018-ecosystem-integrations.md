# ADR-018: Ecosystem integrations: min cut, PPR, RaBitQ, injection screening (3.4.0)

- **Status:** Accepted
- **Date:** 2026-11-05
- **Release:** agentdb 3.4.0

## Context

agentdb already had code paths meant to use native ruvector packages, but two of them could never run, and one listed dependency was never imported:

| Code | Probed for | On npm? | Effect |
|---|---|---|---|
| `MincutService.initialize()` | `@ruvector/mincut`, `ruvector-mincut-wasm` | no (404) | Always printed "No mincut bindings available". The "stoer-wagner" path was a half split of node ids, not a min cut |
| `SparsificationService.initialize()` | `@ruvector/sparsifier`, `ruvector-sparsifier-wasm` | no (404) | PPR always ran in JavaScript, O(n²·d) per iteration |
| `dependencies` | `@ruvector/graph-transformer` | yes | Installed for every user and imported nowhere. It exposes `GraphTransformer.pprScores(source, adjacency, alpha)`, the exact call SparsificationService was probing for |

The goal of this release is to connect agentdb to packages that do this work, through real, tested code paths. Every integration follows one shape:

1. An optional dependency, installed by default and allowed to fail.
2. A lazy `import()` through a variable (`src/utils/optional-import.ts`), so it is not resolved at build time.
3. A graceful fallback to the existing JavaScript path, with an accessor that reports which path ran.
4. A test that imports the package directly, so a missing or changed package fails CI instead of silently testing the fallback.

No candidate depends on agentdb. Every candidate's `dependencies`, `optionalDependencies` and `peerDependencies` were checked on the registry.

## Decision

### Wired

| Package (weekly downloads) | Code path | Default | Test |
|---|---|---|---|
| `@ruvector/mincut-wasm` 2.3.0 (47k) | `MincutService.stoerWagnerMincut()` via `controllers/integrations/MincutWasmAdapter.ts`: `WasmMinCut.fromEdges` + `partition()` → `partitions: number[][]` | on, graphs ≤ `wasmMaxNodes` (2000) and ≤ `wasmMaxEdges` (20,000) | `tests/unit/integrations/mincut-wasm.test.ts` |
| `@ruvector/graph-transformer` 2.0.4 (65k) | `SparsificationService.pprSparsification()` via `controllers/integrations/PprNativeAdapter.ts` (NAPI) | on | `tests/unit/integrations/ppr-graph-transformer.test.ts` |
| `ruvector-graph-transformer-wasm` 2.0.4 (60k) | same adapter, WASM rung (`initSync` with the .wasm bytes in Node, `init()` in browsers) | when NAPI is missing | same file |
| `@ruvector/rabitq-wasm` 0.1.2 (51k) | `SqlJsRvfBackend` search via `backends/rvf/RabitqAccelerator.ts` | **off**: `quantizedIndex: 'rabitq'` or `AGENTDB_QUANTIZED_INDEX=rabitq` | `tests/unit/integrations/rabitq-sqljs.test.ts` |
| `aidefence` 3.0.1 (0.9k) | `security/injection-guard.ts`, called by every episode, pattern and skill create, update and batch insert | **off**: `AGENTDB_INJECTION_GUARD=warn\|block` or `setInjectionGuard()` | `tests/unit/integrations/injection-guard.test.ts` |

`@ruvector/graph-transformer` moves from `dependencies` to `optionalDependencies`. It now has a use, and its native binary can be missing on some platforms.

Adapter rules:

- **Min cut.** Adjacency is treated as undirected unit-weight edges, de-duplicated, ignoring self loops and out-of-range ids. A vertex with no edges is returned as its own side with cut 0, because WASM never sees it. Graphs with fewer than 2 nodes keep the JavaScript handling. The partition cache key is now an FNV-1a hash of the adjacency; the old `${n}-${edgeCount}` key returned one graph's cut for another graph of the same size.
- **PPR.** `undefined` entries become `[]` and out-of-range ids are dropped before native code runs. The scores come back as a `Float32Array`. An out-of-range source falls back to JavaScript.
- **RaBitQ.** The index covers a snapshot. Writes and removals after it are dirty and scanned exactly. The snapshot is rebuilt once dirty ids pass max(256, 10%). Candidates (`k × overfetch`, default 10) are rescored with the backend's exact metric. cosine (vectors normalised before indexing) and l2 only; `ip` and filtered queries stay exact. Off by default because results become approximate and each rebuild blocks a query for seconds at 50k vectors.
- **Injection guard.** Uses aidefence's side-effect-free `dist/detection` entry, so the HTTP gateway and express are never loaded. `warn` logs to stderr, which keeps MCP stdio clean. `block` throws `ValidationError` `INJECTION_DETECTED` and fails closed (`INJECTION_GUARD_UNAVAILABLE`) when aidefence is missing. Off by default: with the guard off, aidefence is never imported.

The agentdb Claude Code plugin gains `injection_guard` (→ `AGENTDB_INJECTION_GUARD`). New skills, `agentdb-graph-partition`, `agentdb-quantized-search` and `agentdb-memory-guard`, ship in the agentdb-graph, agentdb-search and agentdb-memory plugins and are mirrored for Codex under `.agents/skills/`. `tests/unit/integrations/skills-mirror.test.ts` keeps the copies identical.

### Evaluated and not wired

| Package | Finding |
|---|---|
| `sublinear-time-solver` 1.7.2 | `computePageRank` builds a dense n×n system and calls `getEntry` for every pair. PPR at n=1,000 took **29.7 s with 3.6% max error**, against 0.49 ms and 2e-7 for `@ruvector/graph-transformer`. A default install pulls in 220 packages and 123 MB (express, typescript, tsx). Rejected |
| `@ruvnet/bmssp` 1.0.0 | Single-source shortest paths only. agentdb's only shortest-path API is `GraphBackend.shortestPath`, served by `@ruvector/graph-node`. No PPR or sparsification. No consumer |
| `ruvector-sparsifier` (crate) | Exists at `ruvector/crates/ruvector-sparsifier{,-wasm}` and is on crates.io (2.3.0). There is **no npm package** (`@ruvector/sparsifier`, `ruvector-sparsifier-wasm`, `@ruvector/sparsifier-wasm` all 404), so `spectralSparsification` keeps the degree-based approximation. Publishing it to npm would give that method a native path |
| `@ruvector/graph-wasm` 2.0.5 | `GraphDB.createNode(labels, props)`, no `open(path)` or storage path. `GraphDatabaseAdapter` needs graph-node's `GraphDatabase` (`createNode({id, embedding, labels, properties})`, `open`, `begin/commit`, `batchInsert`). Not a drop-in fallback without a persistence shim |
| `@ruvector/diskann` 0.1.3 | NAPI Vamana index with string ids, insert/build/search/delete/save/load. A possible large-scale backend, but it needs its own `VectorBackend` implementation and measurement. Deferred |
| `@ruvector/emergent-time`, `@ruvector/learning-wasm`, `@ruvector/cnn`, `@ruvector/kge` (11 MB, 11/week), `@ruvector/typesafe` (136 MB), `strange-loops`, `@ruvector/pi-brain`, `@ruvector/rvlite` | No agentdb code path they would serve (agent clocks, MicroLoRA, image CNN, KG embeddings, type checking, consciousness loops, a remote brain client, a separate database). Listing them would be dead weight |

## Measured

`npm run benchmark:ecosystem` (Node 22, x64 Linux):

| | Without | With | |
|---|---|---|---|
| Min cut, 1,000 / 2,000 / 4,000 nodes (3 edges per node) | cut 1003 / 2018 / 4015 (half split), 0.4–3.3 ms | cut **4 / 3 / 3**, 148 / 355 / 1694 ms | exact, equals brute force on 40 random graphs |
| PPR, 500 / 2,000 / 5,000 nodes | 48.7 / 652 / 5339 ms | 1.21 / 2.58 / 6.85 ms (NAPI) | 40x / 253x / 780x faster, max abs diff 3.8e-7 |
| SqlJsRvfBackend top-10, 384-dim, 5k / 20k / 50k vectors | 3.3 / 8.5 / 33.3 ms | 3.1 / 3.4 / 5.2 ms | recall@10 1.000; index build 0.5 / 1.4 / 5.1 s |

RaBitQ recall for queries unrelated to the data (near-tied neighbours, 5k vectors): 0.74–0.76 at `overfetch` 2, 0.96–0.98 at 10 (the default), 1.000 at 50.

Min cut construction grows about quadratically with nodes (6.5 s at 8,000 nodes, 83 s at 20,000) and with edges (2.2 s at 2,000 nodes / 51k edges, 15 s for K1000's 500k), which is why `wasmMaxNodes` defaults to 2000 and `wasmMaxEdges` to 20,000. The first version capped nodes only; `tests/unit/attention-sparse.test.ts` caught K1000 timing out, and the edge cap fixes it. `WasmThreeLevelHierarchy` builds faster but returned cut values of 0–1 where the true value was 3–4, so it is not used.

aidefence detection: ~100 ms for the first scan (pattern compile), then 0.1–10 ms per field.

## Verification

- `tests/unit/integrations/` runs in the Library workflow, as does `tests/unit/controllers/MincutService.test.ts`. That workflow now also triggers on `.agents/**` and `plugins/*/skills/**`.
- `npm run build` (including both browser bundles) and `npm run verify:bundle` pass.
- `bash plugins/agentdb/scripts/smoke.sh`, `claude plugin validate` for the marketplace and the three updated plugins, and `claude plugin test plugins/agentdb` pass.

## Consequences

- A default install also gets `@ruvector/mincut-wasm` (1.2 MB), `@ruvector/rabitq-wasm` (0.3 MB) and `aidefence` (0.3 MB plus dependencies, mostly already present through `@modelcontextprotocol/sdk`). Any of them can fail to install without breaking agentdb.
- `MincutService` `stoer-wagner` results change for graphs of 2 to 2000 nodes: they are now true minimum cuts, often one low-degree node against the rest, instead of a half split. `algorithm` is `stoer-wagner-wasm`. `MincutService.partition()` now applies its `maxPartitionSize` option (declared before but ignored), splitting larger sides into equal chunks of consecutive ids (`algorithm` gains `+bounded`). Without that, partitionedAttention, which is quadratic in part size, timed out on a true one-node-versus-rest cut; with it, its tests run in the same time as on 3.3.0.
- PPR results now come from 20 fixed native iterations, which keep the mass of a node with no out-edges. Scores on graphs with such nodes differ from the old JavaScript output. `method` is `ppr-napi` or `ppr-wasm`.
- Pre-existing, not changed here: 38 tests in `tests/unit/sparsification.test.ts` fail on main because they pass object-keyed adjacency (`{0: [1]}`) to an array API.
