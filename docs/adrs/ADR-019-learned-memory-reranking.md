# ADR-019: Evaluated learned memory reranking

Status: Implemented
Date: 2026-11-10
Updated: 2026-11-10
Related: ADR-016 (hybrid retrieval), ADR-012 (MCP capabilities)

## Context

AgentDB persists ruv-FANN models and weight programs, but its episode retrieval
has no learned scorer trained on caller-labeled relevance judgments. Existing
cross-encoder reranking is a separate optional path. Directly retraining an active
model would change retrieval before its ranking quality had been evaluated.

## Decision

Add a `NeuralReranker` bounded context with separate candidates, stored evaluations
and a revisioned active-model pointer. Reuse the public ruv-FANN WASM runtime for
deterministic small-network training and cached inference. Keep model shape and
four normalized features versioned and fixed; relevance labels are never inputs.

Support direct hash-bound WSP patches as a second candidate source. Inspection
exposes seed, active or candidate weights without activating them. Preserve training
query provenance across patches and checkpoints. Omit training metrics for direct
edits. Task-embedding model registries and automatic agent optimization are outside
this change.

Evaluate whole query groups using NDCG and MRR against both the supplied baseline
and current active model. Enforce disjoint training/evaluation query IDs, positive
improvement, bounded regressions and at least five judged evaluation groups.
Only an internally persisted accepted evaluation can authorize promotion. All
state transitions bind generation, revision and model hash, with retained rollback.

Opt-in SQL episode queries default to shadow scoring. Shadow preserves ordinary
retrieval exactly; active mode reorders a bounded pool only after explicit
promotion. Missing models and inference failure preserve ordinary results. Neural
queries bypass the result cache. Keep graph retrieval and cross-encoder behavior
independent. Expose strict MCP tools with admin-only promotion and rollback.

## Validation and limitations

Test real WASM and both SQLite drivers, independent known ranking metrics,
mutation isolation, stale asynchronous updates, reset/ABA, corruption, retention,
readonly behavior, exact shadow preservation and active fallback. Exercise actual
MCP stdio and installed package entrypoints. Bind the synthetic evaluation report
and no-skip test report to the exact source and packed archive in the protected
release lane.

Synthetic results demonstrate the pipeline, not real-world relevance. Query-ID
disjointness cannot establish semantic independence; dataset provenance and
judgments remain caller responsibilities. Scores are not probabilities. A
reranker cannot recover candidates absent from retrieval. Feature changes require
a new feature schema and retraining; there is no silent schema migration.
