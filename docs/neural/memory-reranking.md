# Learned memory reranking

AgentDB 3.10.0 adds `NeuralReranker`: a small ruv-FANN network that learns to order
already-retrieved episodes from numeric features. Models, training candidates,
evaluations and promotion history persist in SQLite. Inference and training use
real WASM; no external model provider or additional model download is required.

Existing retrieval is unchanged unless a query opts in. **Shadow is the default
for an opted-in query:** it adds diagnostic scores while preserving the ordinary
retrieval results and order. Active mode uses only an explicitly promoted model.

## Features and labels

The fixed feature version is `agentdb.rerank-features/1`. Each candidate supplies
exactly four finite numbers between zero and one:

| Feature | Episode adapter |
|---|---|
| `similarity` | Cosine similarity mapped from [-1, 1] to [0, 1], then clamped |
| `reward` | Stored episode reward clamped to [0, 1] |
| `success` | Stored episode success as 0 or 1 |
| `recency` | Exponential decay with a 30-day half-life; missing timestamps map to 0 |

`episodeRerankFeatures(episode, nowSeconds)` produces the same feature mapping as
retrieval. Use a fixed time when recording a labeled dataset. Preserve the features
that were available when the query ran; do not reconstruct them from later outcomes.

A labeled query group looks like this:

```json
{
  "queryId": "support-case-104",
  "candidates": [
    { "id": "episode-23", "features": { "similarity": 0.93, "reward": 0.4, "success": 0, "recency": 0.9 }, "relevance": 0 },
    { "id": "episode-51", "features": { "similarity": 0.79, "reward": 0.8, "success": 1, "recency": 0.6 }, "relevance": 3 }
  ]
}
```

Candidate array order is the baseline ranking. Relevance is a separate, explicit
integer judgment from 0 (irrelevant) to 3 (highly relevant). Do not manufacture
relevance labels from reward or success: those are input features, and doing so
would create a circular evaluation.

Split data by whole query groups, preferably by session/user/time as appropriate.
The controller rejects overlapping training/evaluation query IDs. It cannot prove
that differently named queries are independent, that judgments are reliable, or
that data has never been used to choose a model. Keep a separate validation split
for tuning and an untouched test split for the final evaluation.

## Train, evaluate and promote

```ts
import { readFile } from 'node:fs/promises';
import { createDatabase } from 'agentdb/db-fallback';
import { NeuralReranker } from 'agentdb/controllers/NeuralReranker';

const db = await createDatabase('./memory.db');
const reranker = new NeuralReranker(db);
try {
  // Arrays of labeled query groups with the shape above.
  const training = JSON.parse(await readFile('./training.json', 'utf8'));
  const heldout = JSON.parse(await readFile('./heldout.json', 'utf8'));
  const initial = reranker.create('episode-relevance');
  const candidate = await reranker.train(initial.id, training);
  const evaluation = await reranker.evaluate(initial.id, candidate.candidateId, heldout);
  console.log(evaluation.baseline, evaluation.candidate, evaluation.reasons);
  if (evaluation.accepted) {
    const active = await reranker.promote(initial.id, evaluation.evaluationId, initial.revision);
    console.log('Promoted revision', active.revision);
    // If required later: reranker.rollback(initial.id, initial.revision, active.revision).
  }
} finally {
  reranker.dispose();
  db.close();
}
```

Training creates a separate candidate and never changes the active model. The
network is fixed at 4 inputs, 8 Tanh hidden units and one linear output, with
49 parameters. Initialization is deterministic. Training uses AdamW and MSE on
relevance/3; defaults are 250 epochs and a 0.01 learning rate. Scores are ranking
signals, not calibrated probabilities.

Evaluation measures macro-averaged NDCG@k and MRR@k on the exact supplied candidate
sets. NDCG uses graded gains `2^relevance - 1`; MRR treats relevance greater than
zero as relevant. Defaults require:

- At least five evaluation query groups, each with a positive judgment.
- Mean NDCG improvement of at least 0.01 over the input baseline **and** the current
  active model, when one exists.
- No mean MRR regression.
- No individual-query NDCG regression greater than 0.1.

`RankingPolicy` exposes `k` (default 10), `minNdcgImprovement`, `maxMrrRegression`
and `maxPerQueryNdcgRegression`. Improvement must remain strictly positive.
Acceptance applies to these examples and thresholds; it does not establish
unseen-query generalization or prove that retrieval recall improves.

Promotion accepts an internally stored evaluation ID, not a caller-supplied
receipt. Model hashes, generation tokens and revisions bind the evaluation to
its source state. A stale candidate or concurrent update cannot overwrite a newer
model. Rollback verifies a retained checkpoint and creates a new revision.

## Direct weight-space programming

An agent can inspect a seed, active model or candidate and submit explicit weight
edits instead of training. Both paths use the same ranking evaluation and promotion
gates. Inspection of a new reranker exposes a deterministic seed; it does not
activate that seed.

```ts
const inspection = await reranker.inspectWeights('episode-relevance');
const candidate = await reranker.proposePatch('episode-relevance', {
  schema: 'ruv-fann.weight-patch/1',
  baseHash: inspection.baseHash,
  edits: [{ layer: 0, parameter: 'weights', index: 1, value: 0.5 }]
});
const evaluation = await reranker.evaluate('episode-relevance', candidate.candidateId, heldout);
if (evaluation.accepted) {
  await reranker.promote('episode-relevance', evaluation.evaluationId, inspection.revision);
}
```

This example is a proposal, not a claim that this edit improves ranking. Inspect
another candidate with `inspectWeights(id, candidateId)` and pass that same ID as
the third argument to `proposePatch` when branching from it. Patches bind to the
exact model fingerprint and permit at most 128 finite, bounded edits. Stale hashes
and topology changes are rejected. Patch candidates record `origin: 'weight-patch'`
and their patch; training losses, epochs and training dataset hashes are absent.
Training-query provenance survives patching, promotion, rollback and candidate
retention, so editing a trained model does not bypass the disjointness check.

This implements the inspect–edit–evaluate–promote portion of an adaptive weight
architecture. It does not automatically retrieve model templates by task embedding,
run an LLM optimizer, change topology or update weights during retrieval. Those
orchestration decisions belong to the caller. Execution uses the published
ruv-FANN WASM runtime; no native N-API or WebGPU performance claim is implied.

## Use with retrieval

Pass the controller as the seventh optional `ReflexionMemory` constructor argument:

```ts
import { ReflexionMemory } from 'agentdb/controllers/ReflexionMemory';

const memory = new ReflexionMemory(db, embedder,
  undefined, undefined, undefined, undefined, reranker);

const shadow = await memory.retrieveRelevant({
  task: 'recover a failed deployment', k: 5,
  neuralRerank: { id: 'episode-relevance' }
});
const active = await memory.retrieveRelevant({
  task: 'recover a failed deployment', k: 5,
  neuralRerank: { id: 'episode-relevance', mode: 'active' }
});
```

Shadow preserves the normal top-k results and exposes optional `neuralRerank`
metadata on each returned episode. A `candidateId` may be used in shadow to inspect
an unpromoted candidate. Active mode widens the existing SQL retrieval path to a
bounded pool of 64, ranks that pool, and returns k results. It cannot recover a
relevant episode missing from that pool. Existing filters still apply, and the
original `similarity` field remains cosine similarity.

A missing model or inference failure preserves the ordinary retrieval result with
a diagnostic status. Graph retrieval is not neurally reranked. Neural queries
bypass the result cache so promotion, rollback and recency cannot silently reuse
old neural results. Prepared neural models have a separate bounded LRU cache.

The controller can also rank supplied feature vectors directly with `rank`.
Its result includes baseline/shadow order, proposed order, finite scores,
revision and model identity. Active ranking never accepts an unpromoted candidate.

## MCP and bounds

`agentdb_search` accepts the same `neuralRerank` option. Dedicated tools are:

| Tools | Default capability profiles |
|---|---|
| `agentdb_rerank_get`, `list`, `rank`, `inspect` | readonly, retrieval, learning, admin |
| `agentdb_rerank_create`, `train`, `evaluate`, `patch` | learning, admin |
| `agentdb_rerank_promote`, `rollback` | admin |

All names in the table use the `agentdb_rerank_` prefix. Explicit allow/deny policy
retains its existing semantics. Schemas reject unknown fields and invalid numeric
values. Administrative database reset disposes and reconstructs the controllers.

Bounds are 64 candidates per query, 128 query groups and 512 candidates total per
training/evaluation call, at most 1,000 epochs and 50 million training operations
(parameters × rows × epochs). Retention keeps at most 16 candidates, 16 evaluations
and 32 checkpoints per reranker; pending asynchronous work is bounded. Read-only
controllers reject mutations and do not initialize tables. Call `dispose()` to
release prepared WASM models; it does not close the database.

## Reproduce the evaluation

From the AgentDB source checkout:

```bash
npm ci
npm run build
npm run eval:neural-rerank
```

The report uses deterministic synthetic data with separate training, validation
and held-out query IDs. It compares fixed training budgets using validation NDCG
only, then checks the selected candidate on the held-out split. It also verifies
shadow order, promotion, rejected-evaluation handling, rollback, stable ties and
finite scores. A separate predetermined `tanh(reward)` toy model verifies the
gradient-free patch path without calling training. Timings describe warmed model ranking on that machine, excluding
embedding and candidate retrieval. These results validate the implementation;
they do not measure production memory relevance.

A reference run of `synthetic-v1` selected 250 epochs using validation queries:

| Held-out metric | Input baseline | Selected candidate |
|---|---:|---:|
| NDCG@10 | 0.691862 | 1.000000 |
| MRR@10 | 0.875000 | 1.000000 |

The eight held-out groups contain eight candidates each. The synthetic generator
intentionally supplies a learnable relationship between historical signals and
scenario relevance. The release attaches the exact source-bound report so these
numbers can be checked without interpreting them as a production benchmark.
