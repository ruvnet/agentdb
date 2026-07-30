# ADR-013: RVF COW Research Generations

- Status: Accepted
- Date: 2026-07-29

## Decision

Research, consolidation and re-embedding candidates run in RVF copy-on-write
branches derived from an authoritative parent. Candidate code can request
evaluation but cannot replace the parent or move its active alias.

Each experiment binds:

- repository and commit SHA;
- benchmark artifact hash;
- embedding-space identity hash;
- experimental revision;
- parent and candidate RVF fingerprints;
- verified parent/child lineage;
- candidate witness validity;
- the independent Flywheel promotion decision.

The parent fingerprint is checked again before authorization. Any parent change
causes a stale-baseline failure. Successful authorization returns a deterministic
receipt; a separate trusted workflow performs atomic promotion.

A manifest transition that changes the canonical embedding-space hash must
also advance the experimental revision. This prevents a query-template change
from silently reusing results produced in a different embedding space.

## Consequences

Candidates cannot corrupt the live corpus, silently change their baseline or
gain release credentials. The fingerprint reads only store identity, status and
segment metadata, so authorization cost is bounded by segment count rather than
vector count.
