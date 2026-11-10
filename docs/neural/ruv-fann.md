# Persisted neural models

AgentDB 3.8 adds `ruv-fann` as a runtime dependency. Its `ruv-fann-wasm`
dependency supplies the actual portable core and SIMD binaries. No native build
or optional accelerator is required for dense inference or training.

```js
import { createDatabase } from 'agentdb/db-fallback';
import { FannModels } from 'agentdb/controllers/FannModels';

const db = await createDatabase('models.db');
const models = new FannModels(db);
await models.create('linear', {
  schema: 'ruv-fann.dense-layers/1',
  layers: [{ inputs: 1, outputs: 1, weights: [0], bias: [0],
    activation: 'Linear', steepness: 1 }],
});
await models.train('linear', {
  inputs: new Float64Array([0, 1, 2]),
  targets: new Float64Array([1, 3, 5]), rows: 3,
  epochs: 300, learningRate: 0.05, optimizer: 'sgd',
});
console.log(await models.predict('linear', new Float64Array([1, 2]), 2));
models.dispose();
db.close();
```

Checkpoints persist weights, biases, dimensions and revisions. Optimizer moments
are not persisted; each training call starts a fresh optimizer. Training uses MSE
and supports Linear, Sigmoid, SigmoidSymmetric, Tanh and ReLU activations. It
commits only a complete finite checkpoint. A concurrent update rejects the
stale training commit. Model IDs cannot overwrite an existing checkpoint.

Inference reuses prepared WASM models and batch workspaces in a bounded LRU.
Every call checks the persisted version, including delete/recreate cycles, before
reusing cached state. Call `dispose()` before closing the database to free cached
WASM allocations. The default cache holds at most 16 models and 32 MiB.

MCP tools use flat row-major numeric arrays:

| Tool | Minimum profile | Purpose |
| --- | --- | --- |
| `agentdb_neural_get` | readonly | Read checkpoint and revision |
| `agentdb_neural_list` | readonly | List metadata without weights |
| `agentdb_neural_predict` | readonly | Batched inference |
| `agentdb_neural_create` | learning | Create a checkpoint |
| `agentdb_neural_train` | learning | Train and commit a revision |
| `agentdb_neural_delete` | admin | Delete a checkpoint |

Explicit allow/deny settings apply as for other MCP tools. Unknown arguments,
non-finite numbers and invalid dimensions are rejected. Requests are bounded to
128 rows, 1,000 epochs and 50 million parameter × row × epoch operations. Model
JSON is limited to 8 MiB. These limits constrain synchronous WASM work; training
is intended for small dense models rather than foundation-model fine tuning.
