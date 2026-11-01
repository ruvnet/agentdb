# ADR-016: Hybrid episode retrieval, opt-in cross-encoder, LongMemEval benchmark (3.2.0)

- Status: Accepted
- Date: 2026-10-31

## Decision

Release `3.2.0` (minor) with:

1. **Hybrid retrieval by default.** `ReflexionMemory.retrieveRelevant()`, and so `agentdb_search`, fuse a BM25 ranking over episode text (task, input, output, critique) with the vector ranking by reciprocal-rank fusion, `1/(60+rank_dense) + 1/(60+rank_bm25)`. The dense pool is widened to `max(4k, 50)` (cap 500). The top keyword hits (as many as the dense pool) are fetched, filtered like dense hits and given their real cosine similarity. The dense side is then the cosine order over the union, so no episode gets an invented rank. `hybrid: false` restores the 3.1.0 order. Not applied with a graph backend, whose episode ids are not SQL row ids.
2. **Opt-in cross-encoder reranking.** `rerank: true` scores the union of the top 60 dense and top 60 keyword candidates with a local cross-encoder, and adds `8/(60+rank_ce)` to the fused score. Default model `Xenova/ms-marco-MiniLM-L-6-v2` (22M parameters, q8), overridable with `AGENTDB_RERANKER_MODEL`. It is off by default. It costs about 200 ms per call on CPU (8 cores, 60 pairs) and downloads 23 MB on first use, which is over the MCP 100 ms target.
3. **A reproducible benchmark**, `benchmarks/longmemeval/`, and its results in `benchmarks/longmemeval/results/`.
4. `ruvector` optional dependency `^0.2.40` → `^0.3.5`.

## Why

Vector search alone misses episodes whose wording matches but whose embedding does not: names, error codes, numbers, rare terms. agentdb shipped a `HybridSearch` class (BM25 + RRF), but no search path used it, and there was no public benchmark of agentdb's retrieval at all. LongMemEval (Wu et al., ICLR 2025) is the standard long-term chat-memory benchmark. It labels the evidence sessions for 500 questions, so retrieval can be scored without an LLM.

## Protocol

- Data: `longmemeval_s_cleaned.json` (~50 sessions per question) and `longmemeval_m_cleaned.json` (~500 sessions, 2.7 GB) from `huggingface.co/datasets/xiaowu0162/longmemeval-cleaned`.
- Unit: every turn is a chunk, and turns longer than 160 words are split into overlapping 160-word windows (S: 327k unique chunks, M: 850k). A session scores as its best chunk.
- Metrics: session-level `recall_any@k` (any gold session in the top k), `recall_all@k` (all of them), NDCG@k with binary relevance, k ∈ {5, 10}.
- Official protocol: the 30 `_abs` (abstention) questions are skipped, leaving 470, as the LongMemEval retrieval eval does. `--include-abs` scores all 500, which is the protocol SelRoute (arXiv 2604.02431) reports.
- Embedder: `Xenova/all-MiniLM-L6-v2`, agentdb's default, with the same `feature-extraction` call agentdb makes. Embeddings were computed on GPU in fp32; agentdb's CPU path yields the same vectors to rounding.
- What the harness exercises: the ranking functions agentdb ships (`src/search/conversation-retrieval.ts`, and the fusion and cross-encoder settings in `src/controllers/episode-hybrid.ts`). It does not exercise `storeEpisode()` or the episode store; stored episodes are one vector per episode text, not turn chunks. The comparisons below are therefore between ranking strategies over the same chunks and the same embedder, not against agentdb 3.1.0 end to end.

## Results

`recall_all@5 / NDCG@5 / recall_all@10`, cross-encoders at depth 60:

| | S (470) | M (470) | M with abstention (500) |
|---|---|---|---|
| dense only | 0.883 / 0.904 / 0.947 | 0.630 / 0.692 / 0.751 | 0.616 / 0.679 / 0.734 |
| BM25 only | 0.830 / 0.874 / 0.908 | 0.702 / 0.749 / 0.766 | 0.684 / 0.735 / 0.754 |
| **hybrid (default)** | **0.900 / 0.924 / 0.949** | **0.719 / 0.750 / 0.806** | 0.700 / 0.736 / 0.790 |
| **+ CE MiniLM-L-6 ×8 (`rerank: true`)** | **0.934 / 0.943 / 0.968** | **0.789 / 0.828 / 0.881** | 0.766 / 0.810 / 0.860 |
| + CE MiniLM-L-12 ×16 | 0.921* / 0.942 / 0.970 | 0.802 / 0.837 / 0.872 | 0.778 / 0.818 / 0.852 |
| + CE mxbai-rerank-xsmall ×16 | 0.962 / 0.965 / 0.985 | 0.791 / 0.844 / 0.898 | 0.768 / 0.825 / 0.876 |

\* MiniLM-L-12 ×8 on S: 0.923 / 0.941 / 0.970. Full per-type breakdowns are in the JSON files.

Published comparisons:

- **LongMemEval paper, Table 9 (M, value = session).** Best row: Contriever with key = value + LLM-extracted user facts, Recall@5 0.762, NDCG@5 0.632, Recall@10 0.862. Stella V5 1.5B on the same key design scores 0.732 / 0.520 / 0.862. The paper does not say whether Recall means any or all gold sessions. Our numbers clear 0.762 under both readings: `recall_all@5` 0.789–0.802, `recall_any@5` 0.938–0.953. That uses no LLM at indexing time and models of 22–70M parameters.
- **SelRoute (M, 500 questions including abstention), best row** (bge-base embedder, 109M): recall_all@5 0.800, NDCG@5 0.800. On that protocol our best NDCG@5 is higher (0.825 with mxbai, 0.818 with MiniLM-L-12). Our best recall_all@5 is lower (0.778). This is a mixed result, not a win.
- **gbrain-evals (S, 470 questions, strict recall@5).** Hybrid with OpenAI `text-embedding-3-large` plus reranker scores 0.955. Ours with mxbai-rerank-xsmall is 0.962. Our shipped default reranker (MiniLM-L-6) scores 0.934, which is lower.

Negative results, not shipped:

- **Time-window boosting** (rule-based parsing of "two weeks ago", "last Saturday" and month names, then boosting sessions in that window). It fired on 57 of 470 S questions, and on 20 of those the gold sessions fell outside the window. Net effect: S recall_all@5 0.900 → 0.898. Removed.
- **Session-level BM25** (whole session as one document) as a third RRF term: M recall_all@5 0.719 → 0.736 without a reranker, but no gain on top of the cross-encoder (0.789 → 0.791). The product has no session grouping, so not shipped.
- **Chunk aggregation** "max plus a decayed sum of the next chunks" made no consistent difference against max.

## Consequences

- `retrieveRelevant()` results are no longer sorted by `similarity` by default; `similarity` is still each episode's cosine. Callers that depend on cosine order pass `hybrid: false`.
- **BM25 index cost.** The index covers the newest 10,000 episodes (`AGENTDB_HYBRID_MAX_EPISODES`, clamped to [1, 1,000,000]); older episodes are reached by vector search only. The first search in a process builds it: about 200 ms at 10k and 1,061 ms at 50k on ruvultra. Later inserts are appended, at a warm query cost of 8.4 ms and write-then-search of 7.9 ms at 50k. A delete, or growth past 1.25× the cap, triggers a full rebuild. An in-place `UPDATE` of an episode's text, with count and max id unchanged, is not detected until the next rebuild; episodes are not updated in place by agentdb itself.
- **Short result lists.** Only the top `2k` of the fused order are guaranteed loaded. When filters reject many of them, a call can return fewer than `k` results.
- **Cached failures.** If the cross-encoder fails to load, that query's hybrid-order result is cached under the `rerank: true` key until the query cache expires.
- **Input bounds.** `agentdb_search` now validates `query` (non-empty string, at most 10,000 characters). Tokenising caps text at 32 KB, 4,096 tokens per document and 256 per query. The tokeniser regex `[\p{L}\p{N}]+` has no nesting, and a test bounds 10 MB of input at under 500 ms. The cross-encoder caps candidates at 128 and each text at 4 KB.
- **ruflo.** ruflo stores episodes through ReflexionMemory but does not call `retrieveRelevant()`; its `memory_search` has its own hybrid and cross-encoder pipeline (`cli/src/memory/hybrid-retrieval.ts`). ruflo users get this change through the `agentdb` plugin's `agentdb_search` and through agentdb's own API. Running ruflo's pipeline through this harness is follow-up work.
- **ruvector 0.3.5.** It exports everything agentdb loads (`VectorDb`, `isNative` and the five MetaHarness functions). On the ruvector, backend and MetaHarness test files it has 20 failures against 21 on 3.1.0 with ruvector 0.2.40, and every 0.3.5 failure also fails on 0.2.40.

## Reproduce

```bash
# data (~3 GB) and embeddings (GPU: about 15 minutes for S + M; --device cpu works, slowly)
curl -L -o /data/scratch/longmemeval/longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
node --experimental-strip-types benchmarks/longmemeval/embed.mts --split s
# cross-encoder scores; the --depth must match the run below
node --experimental-strip-types benchmarks/longmemeval/rerank.mts --split s --depth 60 --reranker Xenova/ms-marco-MiniLM-L-6-v2
node --experimental-strip-types benchmarks/longmemeval/run.mts --split s --depth 60 --reranker Xenova/ms-marco-MiniLM-L-6-v2 --json out.json
# M needs --max-old-space-size=16000; --include-abs scores all 500 questions
```
