# Durable weight-space programs

AgentDB 3.9.0 adds `WeightPrograms` with exact required dependencies
`ruv-fann-wsp@0.1.0` and `ruv-fann@0.1.2`. A caller or an external LLM authors a
strict weight patch; real ruv-FANN WASM evaluates the original and candidate
against caller-supplied examples. The controller persists the proposal and its
evidence, then promotes it only on an explicit request. There are no provider
calls, embedded credentials, gradients or execution of caller code.

Passing an evaluation policy establishes behavior only on the supplied dataset.
The controller cannot establish that data is held out or that the candidate will
generalize. Evaluation hashes identify evidence; they are not signatures.

## Library example

```ts
import { createDatabase } from 'agentdb/db-fallback';
import { WeightPrograms } from 'agentdb/controllers/WeightPrograms';
import { loadRuntime } from 'ruv-fann';

const db = await createDatabase('./programs.db');
const programs = new WeightPrograms(db);
try {
  const initial = await programs.create('double', {
    schema: 'ruv-fann.dense-layers/1',
    layers: [{ inputs: 1, outputs: 1, weights: [0], bias: [0],
      activation: 'Linear', steepness: 1 }],
  });
  const proposal = await programs.propose('double', {
    schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash,
    edits: [{ layer: 0, parameter: 'weights', index: 0, value: 2 }],
  }, { inputs: [1, 2, 3], targets: [2, 4, 6], rows: 3 },
  { maxMSE: 0, maxRegressionMSE: 0 });

  if (proposal.evaluation.accepted) {
    const promoted = await programs.promote('double', proposal.proposalId,
      initial.revision);
    const prepared = (await loadRuntime()).prepare(promoted.model);
    try { console.log(prepared.predict(new Float64Array([4]))); } // [8]
    finally { prepared.dispose(); }
    const restored = programs.rollback('double', initial.revision,
      promoted.revision); // revision 3, original weights
    console.log(restored.revision);
  }
} finally {
  programs.dispose();
  db.close(); // also persists the sql.js backend
}
```

`create` rejects an existing ID. `get(id)` returns the current model, hash,
revision, bounded history and proposal evidence. `list(limit)` returns metadata
only. `propose` and `promote` are asynchronous; `get`, `list` and `rollback` are
synchronous. Disposing a controller does not close its database.

A new controller on the same database can promote an outstanding proposal after
a process restart. Promotion accepts only a stored proposal ID. Passing a
caller-authored evaluation receipt or changing a returned object cannot authorize
promotion or modify persisted state. Writable access to the database itself is a
trusted boundary; hashes do not authenticate arbitrary database writes.

## Revision and retention rules

Dedicated `wsp_programs`, `wsp_proposals` and `wsp_checkpoints` tables leave existing
`FannModels` checkpoints independent. Creation assigns a random generation token.
A proposal records that generation, source revision, source hash, candidate hash,
dataset hash, policy hash, numerical metrics and acceptance reasons. Model bias
omission/null normalizes to explicit zeros, and negative zero normalizes to zero.

If the program changes while evaluation runs, proposal persistence rejects the
stale work. Promotion recomputes the stored candidate fingerprint and uses a SQL
transaction with generation/hash/revision compare-and-swap. It rejects rejected,
expired, stale, cross-program or missing proposals. Rollback restores an exact
retained checkpoint and increments the current revision; it never reuses an old
revision. Deleting and recreating a program through administrative SQL assigns a
new generation, so old proposals cannot promote into it.

Defaults retain 16 proposals and 32 checkpoints per program. Proposals expire in
24 hours; new proposals prune expired/stale proposals and evict the oldest excess
entries. Promoted evaluation evidence remains in its retained checkpoint even
when the proposal record is evicted. A rollback target must still be retained.
Per-controller in-flight proposals are bounded at the proposal limit to bound
concurrent evaluation allocations. Retention can be tightened through
`maxProposals`, `maxHistory` and `maxProposalAgeMs` constructor options.

`new WeightPrograms(db, { readonly: true })` performs no schema initialization and
rejects every mutation. Use it with an existing database and an appropriately
read-only database connection. `simd: false | true | 'auto'` selects the SDK backend.

## Safety and evaluation limits

The SDK accepts only dense models and edits to existing weight/bias locations;
patches cannot change shapes, activations or topology. Unknown fields, duplicate
locations, non-finite values, malformed shapes and excessive work are rejected.
Caller getters, custom iterators and `toJSON` are not used to snapshot models.

| Bound | Default |
|---|---:|
| Model JSON | 1 MiB |
| Model parameters, including implicit biases | 65,536 |
| Layers / dimension | 64 / 65,536 |
| Edits per proposal | 128 |
| Absolute edited value / absolute delta | 1,000,000 / 1,000 |
| Evaluation rows | 1,024 |
| Elements in each flat input/target array | 65,536 |
| Rows × model parameters | 10,000,000 |
| Program ID length | 128 safe ASCII characters |

Acceptance requires candidate MSE ≤ `maxMSE`, candidate MSE minus baseline MSE ≤
`maxRegressionMSE`, and optional candidate maximum absolute error ≤
`maxAbsoluteError`. All thresholds must be finite and nonnegative. Numerical
overflow rejects evaluation rather than creating accepted evidence.

## MCP

All six tools reject unknown fields at every structured level. Arrays are flat,
row-major primitive numbers. These tools use the same model, patch, dataset and
policy shapes as the library.

| Tool | Required arguments | Capability profile |
|---|---|---|
| `agentdb_wsp_create` | `id`, `model` | learning/admin |
| `agentdb_wsp_get` | `id` | readonly/retrieval/learning/admin |
| `agentdb_wsp_list` | optional `limit` (1–100) | readonly/retrieval/learning/admin |
| `agentdb_wsp_propose` | `id`, `patch`, `dataset`, `policy` | learning/admin |
| `agentdb_wsp_promote` | `id`, `proposalId`, `expectedRevision` | learning/admin |
| `agentdb_wsp_rollback` | `id`, `targetRevision`, `expectedRevision` | admin |

Explicit MCP allow/deny settings retain their existing semantics. An admin
`agentdb_init` reset drops WSP records together with other user tables and
recreates the controller after a successful reset transaction.

## Verification

`tests/unit/controllers/weight-programs.test.ts` uses actual native SQLite,
sql.js and WASM. It constructs and repairs XOR with hand-authored threshold
weights, without gradients; checks rejected policies, mutation isolation,
forged receipts, stale concurrent updates, ABA, expiry, bounded retention,
checkpoint corruption, transactional failure and file reopen. MCP dispatch
schemas and profiles have their own unit tests. `npm run test:mcp-wsp` repeats
durable proposal/promotion/rollback through real stdio servers on both drivers.
The protected library lane and installed packed-consumer checks include WSP.
