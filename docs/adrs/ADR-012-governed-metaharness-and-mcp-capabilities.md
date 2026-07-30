# ADR-012: Governed MetaHarness Verification and MCP Capabilities

- Status: Accepted
- Date: 2026-07-29

## Context

AgentDB is the durable evidence plane for agentic systems. It must be able to
verify Flywheel evidence and replay bundles without becoming a code-execution
surface. Its MCP server also exposes retrieval, learning, administrative and
destructive operations through one undifferentiated tool list.

Ruvector 0.2.40 provides a pinned, lazy MetaHarness facade for capability
discovery, promotion evaluation, replay verification, workspace scoring,
reward-hack detection and credential guards. Darwin execution is explicitly
opt-in and is not exposed by ruvector MCP.

## Decision

1. AgentDB consumes MetaHarness only through the lazy `ruvector` facade.
2. AgentDB exposes evidence validation, replay verification and reward-hack
   scanning through its SDK; it does not expose Darwin/candidate execution.
3. Promotion evidence is validated for finite values and scanned for live
   credentials before evaluation.
4. MCP tools support `readonly`, `retrieval`, `learning` and `admin` profiles,
   plus explicit allow and deny lists.
5. Deny rules always override profiles and allow lists. Authorization is
   applied both to discovery and dispatch.
6. An unconfigured server remains backward compatible. Production deployments
   SHOULD select an explicit profile.

## Consequences

- Ordinary AgentDB use does not load MetaHarness packages.
- Research execution remains in a separate secret-free candidate workflow.
- Operators can expose retrieval without exposing training, deletion or schema
  mutation.
- The ruvector version becomes part of AgentDB's compatibility contract.

## Acceptance criteria

- Importing AgentDB does not import ruvector or MetaHarness eagerly.
- A readonly MCP client cannot discover or invoke mutation tools.
- Explicit deny wins over profile and allow.
- Invalid/non-finite evidence is rejected locally.
- Replay verification may pin the expected gate fingerprint.
- No AgentDB MCP tool can invoke Darwin.

