---
name: agentdb-graph-partition
description: Partition or prune an AgentDB graph in code — exact minimum cut with MincutService (backed by @ruvector/mincut-wasm) and Personalized PageRank sparsification with SparsificationService (backed by @ruvector/graph-transformer). Use when splitting a memory or causal graph at its weakest link, finding the bottleneck between two clusters, keeping only the nodes most relevant to a seed node before attention, or when the user asks which native graph packages AgentDB uses.
---

# Graph partition and PPR sparsification

Two library services (no MCP tool) for graphs given as adjacency lists (`number[][]`, node ids `0..n-1`, treated as undirected and unweighted).

| Service | Answers | Native package (optional dependency, installed by default) | Without it |
|---|---|---|---|
| `MincutService` (`algorithm: 'stoer-wagner'`) | Exact global minimum cut: the fewest edges whose removal disconnects the graph | `@ruvector/mincut-wasm` | Splits nodes `0..n/2` from the rest. Not a min cut; can cut ~n edges where the true answer is 3 |
| `SparsificationService` (`method: 'ppr'`) | Personalized PageRank from a seed node, top-k nodes | `@ruvector/graph-transformer` (NAPI), then `ruvector-graph-transformer-wasm` | JavaScript power iteration, same scores within 4e-7, 23x to 400x slower |

## Use

```ts
import { MincutService, SparsificationService } from 'agentdb/controllers';

const mincut = new MincutService({ algorithm: 'stoer-wagner' });
await mincut.initialize();
const cut = await mincut.stoerWagnerMincut(adjacency);
// cut.algorithm === 'stoer-wagner-wasm' when the native path ran
// cut.partitions: [sideA, sideB], cut.cutSize, cut.cutEdges: [u, v][] (u < v, each edge once)

const ppr = new SparsificationService({ method: 'ppr', topK: 20 });
const top = await ppr.sparsify(seedNode, adjacency);
// top.method === 'ppr-napi' | 'ppr-wasm' | 'ppr-fallback'; top.topKIndices; top.scores (Float32Array)
```

Check which path ran: `mincut.getBackend()` returns `'wasm'` or `'fallback'`; `ppr.getPprRuntime()` returns `'napi'`, `'wasm'` or `'fallback'`.

## Know the costs

- Min cut construction grows about quadratically: ~0.1 s at 1,000 nodes, ~1.4 s at 4,000, ~6.5 s at 8,000. Graphs above `wasmMaxNodes` (default 2000) use the fallback. Raise it only for one-off analysis: `new MincutService({ algorithm: 'stoer-wagner', wasmMaxNodes: 8000 })`.
- A global min cut is often one low-degree node against the rest. That is the correct answer, not a balanced split. For balanced chunks, cut recursively or use PPR neighbourhoods.
- Results are cached by graph content, so a repeated query on the same graph is free.
- Native PPR runs 20 power iterations and keeps the mass of a node with no edges on that node. The fallback drops it, so scores differ on graphs with dead-end nodes.
- `AttentionService.sparseAttention()` and `partitionedAttention()` call these services, so they pick up the native paths too.

## Don't

- Don't read `cutSize` from `algorithm: 'karger'` or `'flow-based'` as a global min cut. Karger is randomized, and flow-based is an s-t cut between nodes 0 and n-1.
- Don't pass object-keyed adjacency (`{0: [1]}`). Pass arrays.
