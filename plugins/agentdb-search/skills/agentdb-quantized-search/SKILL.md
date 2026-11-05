---
name: agentdb-quantized-search
description: Speed up AgentDB's built-in sql.js vector backend on large collections with the opt-in RaBitQ 1-bit quantized index (@ruvector/rabitq-wasm). Use when SqlJsRvfBackend search gets slow past ~10,000 vectors, when choosing between exact and approximate search for a browser or no-native install, or when the user asks about RaBitQ or quantized search in AgentDB.
---

# Quantized search (RaBitQ)

`SqlJsRvfBackend`, the built-in `.rvf` backend used when no native vector engine is available, scans every vector on every query. With `quantizedIndex: 'rabitq'`, a 1-bit RaBitQ index from `@ruvector/rabitq-wasm` (an optional dependency, installed by default) narrows each query to a candidate set. The candidates are then rescored exactly, so returned similarities are exact. Only which ids reach the top-k is approximate.

## Turn it on

```ts
import { SqlJsRvfBackend } from 'agentdb/backends';

const backend = new SqlJsRvfBackend({
  dimension: 384,
  metric: 'cosine',               // or 'l2'; 'ip' always stays exact
  quantizedIndex: 'rabitq',
  rabitq: { minVectors: 2000, overfetch: 10, rerankFactor: 20 },  // defaults shown
} as any);
await backend.initialize();
backend.getQuantizedIndexStats(); // { enabled, indexed, dirty, builds, lastBuildMs } or null when off
```

The environment variable `AGENTDB_QUANTIZED_INDEX=rabitq` turns it on for every SqlJsRvfBackend in the process. If the package is missing, the backend logs one warning and keeps the exact scan.

## Measured (`npm run benchmark:ecosystem`; 384-dim, clustered data, cosine, top-10)

| Vectors | Exact scan | RaBitQ | Recall@10 | Index build (first query) |
|---|---|---|---|---|
| 5,000 | 3.3 ms | 3.1 ms | 1.000 | 0.5 s |
| 20,000 | 8.5 ms | 3.4 ms | 1.000 | 1.4 s |
| 50,000 | 33.3 ms | 5.2 ms | 1.000 | 5.1 s |

For queries unrelated to the data, where neighbours are near-tied, recall@10 is 0.96 to 0.98 at the default `overfetch: 10` and 1.000 at 50.

## How it stays correct

- The index covers a snapshot. Ids written or removed after it are dirty: their stale index hits are dropped and their current vectors are scanned exactly on every query.
- The snapshot is rebuilt on the next query once dirty ids pass max(256, 10% of the snapshot). The rebuild blocks that query, about 1 s per 10,000 vectors at 384 dimensions.
- Queries with a metadata `filter` use the exact scan.
- Below `minVectors` there is no index. The exact scan is as fast there.

## Don't

- Don't enable it for write-heavy workloads that query after every few writes on a large collection. Each rebuild costs seconds.
- Don't expect it to change other backends. RuVector, native RVF and hnswlib already have their own ANN indexes.
