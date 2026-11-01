/**
 * Opt-in cross-encoder reranking (ADR-016). A cross-encoder reads the query and a candidate together, so it ranks
 * far better than either embedding similarity or BM25, at about 200 ms per 60 candidates on CPU for the default
 * 22M-parameter model (q8). It is therefore applied only to a bounded candidate set, and only when a caller asks.
 *
 * The model is downloaded once by @huggingface/transformers. Every input is bounded: at most MAX_CANDIDATES
 * candidates, each cut to MAX_CHARS before tokenising, and the tokenizer truncates pairs at 512 tokens.
 */

export const DEFAULT_RERANKER = process.env.AGENTDB_RERANKER_MODEL || 'Xenova/ms-marco-MiniLM-L-6-v2';
/** Two candidate lists of RERANK_DEPTH (60) each, deduplicated, fit. */
export const MAX_CANDIDATES = 128;
const MAX_CHARS = 4_096;
const BATCH = 32;

type Loaded = {
  tokenizer: (a: string[], o: Record<string, unknown>) => unknown;
  model: (enc: unknown) => Promise<{ logits: { data: ArrayLike<number>; dims: number[] } }>;
};

const loading = new Map<string, Promise<Loaded>>();

async function load(modelId: string): Promise<Loaded> {
  let p = loading.get(modelId);
  if (!p) {
    p = (async () => {
      const t = await import('@huggingface/transformers');
      const tokenizer = await t.AutoTokenizer.from_pretrained(modelId);
      const model = await t.AutoModelForSequenceClassification.from_pretrained(modelId, { dtype: 'q8' });
      return { tokenizer: tokenizer as unknown as Loaded['tokenizer'], model: model as unknown as Loaded['model'] };
    })();
    // A failed load is not cached, so a later call can retry (e.g. after the network returns).
    p.catch(() => loading.delete(modelId));
    loading.set(modelId, p);
  }
  return p;
}

/**
 * Relevance logits of each candidate for the query, in input order. Throws if the model cannot be loaded;
 * callers fall back to their unreranked order.
 */
export async function crossEncoderScores(query: string, candidates: readonly string[], modelId = DEFAULT_RERANKER): Promise<Float64Array> {
  if (candidates.length > MAX_CANDIDATES) throw new RangeError(`at most ${MAX_CANDIDATES} candidates`);
  const out = new Float64Array(candidates.length);
  if (candidates.length === 0) return out;
  const { tokenizer, model } = await load(modelId);
  const q = query.slice(0, MAX_CHARS);
  for (let i = 0; i < candidates.length; i += BATCH) {
    const docs = candidates.slice(i, i + BATCH).map(d => d.slice(0, MAX_CHARS));
    const enc = tokenizer(new Array(docs.length).fill(q), { text_pair: docs, padding: true, truncation: true, max_length: 512 });
    const { logits } = await model(enc);
    const width = logits.dims[1] ?? 1;
    for (let j = 0; j < docs.length; j++) out[i + j] = Number(logits.data[j * width + (width - 1)]);
  }
  return out;
}

