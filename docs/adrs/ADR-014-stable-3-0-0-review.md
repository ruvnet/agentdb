# ADR-014: Stable 3.0.0 from a four-part review

- Status: Accepted
- Date: 2026-10-31

## Decision

Release main as `3.0.0`, the first stable 3.x, with every finding below fixed and regression-tested. Publish to `latest` only; do not add or move the `alpha` or `next` dist-tags. The ruvnet ecosystem is moving to stable versions only (ruflo ADR-513).

## Why

Consumers declared prerelease ranges (`^3.0.0-alpha.20`, `>=2.0.0-alpha.1`). Under semver a prerelease only matches ranges that name the same `major.minor.patch`, so one consumer (ruflo's `@claude-flow/aidefence`, `>=2.0.0-alpha.1`) could not accept `3.0.0-alpha.20`, and npm installed `2.0.0-alpha.3.7` beside it with OpenTelemetry 0.52 and sharp 0.32. A stable `3.0.0` satisfies every one of those ranges and leaves one copy.

A security review, an integration and self-learning review, and a performance review of `3.0.0-alpha.20` found:
- **The default vector backend returned no results.** RuVector search was never awaited, and the backend called methods VectorDB does not have.
- **Learning was lost.** LearningSystem dropped its policy updates.
- **Episodes loaded as skills** from the shared index.
- **`agentdb_init` deleted arbitrary files** under the allow-all default MCP policy.
- **SQL injection** through filter keys.
- **A model cache in shared `/tmp`.**
- **Library timers kept CLIs alive.** `getPatternStats` held one for 5 minutes.
- **Library logs on stdout** corrupted stdio MCP JSON-RPC for hosts that embed AgentDB.

The CHANGELOG lists each fix; each has a regression test that fails on the base commit.

## Gate

Before publishing, the candidate tarball was installed under a fresh `ruflo@3.66.2` through a root `overrides`:
- one AgentDB copy;
- production advisories 70 → 11, with 0 critical;
- `ruflo doctor` 29 passed, 0 failed;
- memory store and search work;
- `autopilot learn` exits in 6 s;
- a stdio MCP session with `autopilot_learn` and `memory_search` produced JSON-only stdout.

The full test suite hangs past 45 minutes on the base commit too, so the gate is the targeted set: 477/477.

## Not done

- `@ruvector/core` 0.1.30 HNSW misses results on very small graphs. The SQL fallback covers zero-hit searches but not partial misses; to be reported upstream.
- There is no pinned SHA-256 manifest for bundled models.
- `RvfBackend` still defaults its storage to `./agentdb.rvf`.
- `@ruvector/attention` 2.x and `ruvector` 0.3 are left to a separate migration.
