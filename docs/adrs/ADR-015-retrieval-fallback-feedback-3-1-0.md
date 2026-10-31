# ADR-015: Exact-search fallback, outcome feedback and top-k scans (3.1.0)

- Status: Accepted
- Date: 2026-10-31

## Decision

Release `3.1.0` (minor, no breaking changes) with four changes. Publish to `latest` only, as in ADR-014.

1. **Exact-search fallback.** `searchVectors()` takes an optional exact source, the caller's SQL embedding table. It answers by exact cosine when the collection has fewer than 1000 rows (`exactSearchThreshold`, `AGENTDB_EXACT_SEARCH_THRESHOLD`), when the index returns fewer of the collection's vectors than `min(k, rows)`, or when the index throws. Exact hits come first in exact order; index hits the scan did not see are appended after dedupe. ReflexionMemory, SkillLibrary and ReasoningBank pass their tables.
2. **Outcome feedback and reward-aware pattern ranking.** Patterns rank by `0.7·similarity + 0.2·successRate + 0.1·avgReward`, with both rates clamped to [0, 1], from a candidate pool of 3k. `recordOutcome()` exists on ReasoningBank and SkillLibrary. The MCP tools `agentdb_pattern_feedback` and `agentdb_skill_feedback` are in the default `learning` profile.
3. **Bounded top-k for full scans.** The four brute-force scan sites keep the k best in a heap, compute the query norm once and build result objects only for the winners.
4. **`agentdb_capabilities`.** A read-only MCP tool in the default profile that reports what the server is running on.

## Why

- `@ruvector/core` 0.1.30's HNSW misses neighbours on tiny graphs (ruvnet/RuVector#1298). ADR-014's fallback only covered searches that returned nothing, so a partial miss lost a stored vector silently. In vector-backend mode ReasoningBank never wrote `pattern_embeddings`, so it had no copy to fall back to.
- Reward and success rate could filter results but never reorder them, and skills had no outcome API. Agents could record outcomes but retrieval did not learn from them.
- Each scan scored every row, built an object for it, sorted the whole array and sliced k.
- Hosts could not tell which backend, SQL driver or embedding model they actually got. The embedding service silently uses a hash fallback when the model fails to load.

## Measured

- **Fallback.** 30 collections of 3–12 vectors (64-dim, M=8), 225 lookups of each skill by its own embedding through SkillLibrary. Native `@ruvector/core` on 4fc0610 missed 41–85 of 225 across runs (HNSW build is not deterministic); 3.1.0 misses 0. With `RUVECTOR_BACKEND=rvf` the factory falls through to `RvfBackend`, which missed 0 on both. A stub index that drops the best match fails all three controllers on 4fc0610 and passes on 3.1.0.
- **Ranking.** Two near-duplicate patterns swap order after opposite feedback; with `rerank: false` the similarity order returns. A pattern with perfect feedback but an orthogonal embedding does not overtake a relevant one. On a seeded set of 4 topics × 6 patterns, half of them failed near-duplicates, precision@5 is 0.55 by similarity only and 0.60 reranked (0.60 is the ceiling: 3 relevant per topic).
- **Top-k scans.** Rankings and similarities are bit-identical to the 3.0.0 code on a seeded set with exact ties; ties keep scan order. The query norm and, in SqlJsRvfBackend, the stored norms use the original accumulation order, so scores do not move in the last bit. N=10k, 384-dim, k=10, 40 queries, median of each run, best of three alternating runs (the host had load average ~27 on 32 cores, so single runs varied up to 2×):

  | Scan | 3.0.0 | 3.1.0 |
  |---|---|---|
  | SqlJsRvfBackend.searchAsync | 6.7 ms | 2.3 ms |
  | SkillLibrary legacy scan | 44.5 ms | 35.4 ms |
  | ReflexionMemory SQL fallback | 44.8 ms | 38.3 ms |
  | WASMVectorSearch.findKNN | 23.3 ms | 19.6 ms |

  The SQL-backed scans are dominated by reading 10k 1.5 KB BLOBs, which this does not change.
- **Gate.** `npm run build`, `npx tsc --noEmit` and `npm run test:mcp-stdio` (now also calling `agentdb_capabilities`) pass. The targeted set (tests/regression, tests/security, tests/backends) has no new failures: the 114–115 tests that fail on this branch (114 with `dist/` built) all fail on 4fc0610 in the same environment too (117 there). In a child process, `AgentDB` initialize, store, search and close wrote 0 bytes to stdout and exited in 0.2 s (hash embeddings) and 0.6 s (Transformers.js model).

## Not done

- SkillLibrary keeps its existing composite score (0.4 similarity, 0.3 success rate, 0.1 uses, 0.2 reward). Moving it to the 0.7/0.2/0.1 weights would reorder existing results; feedback already moves skills through the success rate and reward terms.
- ReflexionMemory episodes are not reranked by reward; reward still only filters them.
- The fallback does not apply the controllers' SQL filters (task type, minimum reward); like the index path, it returns candidates the controller then filters.
- The MCP server's controllers still run without a vector index, so the MCP tools always search exactly. `agentdb_capabilities` reports that as `sql-exact-scan`.
- Exact fallback costs a full scan of the collection per search below the threshold; collections at or above it pay only when the index comes back short.
- The upstream HNSW defect is not fixed here; this works around it.
- The package description's MCP tool count ("41") is unchanged; the server lists 38 tools under the admin profile.
