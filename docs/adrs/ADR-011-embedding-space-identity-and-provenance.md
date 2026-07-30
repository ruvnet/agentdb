# ADR-011: Canonical Embedding-Space Identity and Typed Provenance

- Status: Accepted
- Date: 2026-07-29

## Context

Model names do not identify vector spaces. A tokenizer, query instruction,
pooling implementation, truncation limit, dtype, normalization, or runtime
revision can change vectors while retaining the same model ID. Generic
`embed(text)` APIs also let asymmetric retrieval models silently encode a
query as a passage. That can corrupt caches and mix incompatible vectors in a
corpus without producing an error.

AgentDB additionally needs memory provenance that can be consumed by RVF,
MetaHarness, Flywheel, and research manifests without parsing arbitrary
metadata.

## Decision

AgentDB defines one canonical `EmbeddingSpaceIdentity`. Its deterministic
SHA-256 includes:

- model artifact, tokenizer, and prompt-template hashes;
- exact role policy and query/passage templates;
- pooling, truncation, dimension, dtype, and normalization;
- runtime/graph revision and distance metric.

The hash is the compatibility key for caches, stores, migrations, receipts,
and benchmark manifests. Serialization sorts object keys and is portable
across Node, browsers, WASM, and edge runtimes.

Embedding providers have an explicit `symmetric` or `asymmetric` role policy.
The registry is keyed by exact model ID; family-name inference is forbidden.
For a registered model, a conflicting caller declaration is rejected.
Unregistered models must declare their role policy. With asymmetric models:

- `embedQuery` applies the pinned query template;
- `embedPassage` applies the pinned passage template;
- generic `embed` fails as ambiguous;
- caches include identity hash, role, and processed text.

Concurrent requests for the same cache key share one in-flight computation.
This avoids duplicate local inference or API charges without allowing results
to cross identity or role boundaries.

AgentDB uses the maintained `@huggingface/transformers` runtime. The abandoned
`@xenova/transformers` dependency and its legacy `onnx-proto` chain are not
part of the production graph. Existing v2 model-cache directories remain
readable so this migration does not force a model download.

Existing symmetric callers remain source-compatible. Their generated identity
is visibly marked `legacy-unverified`; production persistence and research
workflows should provide immutable hashes.

Memories may carry a typed provenance record:
`user_claim`, `agent_output`, `system_observation`, `tool_result`, or
`unknown`, plus source/run/receipt and trust information.

An authoritative MemoryController enforces these compatibility modes:

- `corpus-mutation`: identity must be present and equal;
- `cache-reuse`: identity must be equal;
- `vector-read`: remains available on mismatch for inspection, export, and
  explicit migration.

Text embedding and corpus mutation are disabled on mismatch. Existing vector
reads are not presented as semantically compatible; they remain an operational
escape hatch for recovery and migration.

## Consequences

Changing only a query prefix creates a different embedding-space hash even
when the model ID is unchanged. Cache reuse and corpus mutation then fail.
Research manifests can use the same hash to require a new experimental
revision.

Callers introducing custom model IDs must explicitly classify role behavior.
Production callers must supply immutable artifact fingerprints to receive a
verified, reproducible identity.

## Acceptance

Given one corpus, changing only the query prefix template while retaining the
same model ID must:

1. produce a new embedding-space identity;
2. reject cache reuse;
3. prevent corpus mutation;
4. preserve vector-only reads; and
5. give manifest validators a different benchmark identity.
