/** Explicitly trained, held-out evaluated memory reranking over real ruv-FANN WASM. */
import { createHash, randomUUID } from 'node:crypto';
import { loadRuntime, type DenseModel, type FannRuntime, type PreparedModel } from 'ruv-fann';
import { applyWeightPatch, fingerprintModel, type WeightPatch } from 'ruv-fann-wsp';
export type { WeightPatch } from 'ruv-fann-wsp';
import type { IDatabaseConnection, IPreparedStatement } from '../types/database.types.js';
import { inTransaction } from './crud-helpers.js';

export const RERANK_FEATURE_SCHEMA = 'agentdb.rerank-features/1' as const;
export interface RerankFeatures { similarity: number; reward: number; success: number; recency: number }
export interface RerankCandidate { id: string; features: RerankFeatures }
export interface LabeledRerankCandidate extends RerankCandidate { relevance: number }
export interface RerankGroup { queryId: string; candidates: LabeledRerankCandidate[] }
export interface TrainingOptions { epochs?: number; learningRate?: number }
export interface RankingPolicy { k?: number; minNdcgImprovement?: number; maxMrrRegression?: number; maxPerQueryNdcgRegression?: number }
export interface NeuralRerankerOptions { readonly?: boolean; simd?: boolean | 'auto' }
export interface RankingMetrics { ndcg: number; mrr: number }
export interface RerankerMetadata { featureSchema: string; id: string; revision: number; modelHash: string | null; createdAt: string; updatedAt: string }
export interface TrainingCandidate {
  featureSchema: string; candidateId: string; rerankerId: string; sourceRevision: number; modelHash: string;
  origin: 'training' | 'weight-patch'; patch?: WeightPatch; baseHash?: string;
  trainingQueryIds: string[]; datasetHash?: string; initialLoss?: number; finalLoss?: number;
  epochs?: number; parameterCount: number; createdAt: string; backend: 'wasm' | 'wasm-simd';
}
export interface RankingEvaluation {
  featureSchema: string; evaluationId: string; rerankerId: string; candidateId: string; sourceRevision: number;
  baseModelHash: string | null; candidateHash: string; datasetHash: string; policyHash: string;
  policy: Required<RankingPolicy>; baseline: RankingMetrics; active: RankingMetrics | null; candidate: RankingMetrics;
  perQuery: { queryId: string; baseline: RankingMetrics; active: RankingMetrics | null; candidate: RankingMetrics }[];
  accepted: boolean; reasons: string[]; createdAt: string; backend: 'wasm' | 'wasm-simd';
}
export interface RerankerHistory {
  revision: number; modelHash: string | null; operation: 'create' | 'promote' | 'rollback';
  sourceRevision: number | null; evaluationId: string | null; createdAt: string;
  evaluation: RankingEvaluation | null; trainingQueryIds: string[];
}
export interface RerankerRecord extends RerankerMetadata {
  model: DenseModel | null; trainingQueryIds: string[]; history: RerankerHistory[]; candidates: TrainingCandidate[]; evaluations: RankingEvaluation[];
}
export interface RankResult {
  order: string[]; scores: { id: string; score: number }[]; revision: number;
  mode: 'shadow' | 'active'; modelHash: string | null; available: boolean; proposedOrder?: string[];
}
export interface WeightInspection {
  rerankerId: string; revision: number; featureSchema: string; model: DenseModel;
  baseHash: string; source: 'seed' | 'active' | 'candidate'; candidateId?: string;
}
export interface RankOptions { mode?: 'shadow' | 'active'; candidateId?: string }
export interface RerankEpisode { similarity?: number; reward?: number; success?: boolean; ts?: number }

const clamp = (value: number) => Math.min(1, Math.max(0, value));
/** Timestamp and now are Unix seconds; a 30-day half-life is fixed for labeling parity. */
export function episodeRerankFeatures(episode: RerankEpisode, nowSeconds = Date.now() / 1000): RerankFeatures {
  if (typeof nowSeconds !== 'number' || !Number.isFinite(nowSeconds)) throw new Error('nowSeconds must be finite');
  return {
    similarity: typeof episode.similarity === 'number' && Number.isFinite(episode.similarity) ? clamp((episode.similarity + 1) / 2) : 0,
    reward: typeof episode.reward === 'number' && Number.isFinite(episode.reward) ? clamp(episode.reward) : 0,
    success: episode.success === true ? 1 : 0,
    recency: typeof episode.ts === 'number' && Number.isFinite(episode.ts) && episode.ts >= 0 ? 2 ** (-Math.max(0, nowSeconds - episode.ts) / 2_592_000) : 0,
  };
}

const FEATURE_KEYS = ['similarity', 'reward', 'success', 'recency'] as const;
const META = 'feature_schema AS featureSchema, id, revision, model_hash AS modelHash, created_at AS createdAt, updated_at AS updatedAt';
type Source = RerankerMetadata & { generation: string; model_json: string | null; training_query_ids: string };
type CandidateRow = { candidateId: string; rerankerId: string; generation: string; sourceRevision: number; modelHash: string; model_json: string; metadata_json: string };
type EvaluationRow = { evaluationId: string; rerankerId: string; generation: string; sourceRevision: number; candidateId: string; candidateHash: string; baseModelHash: string | null; accepted: number; metadata_json: string };
const CANDIDATE = 'id AS candidateId, reranker_id AS rerankerId, generation, source_revision AS sourceRevision, model_hash AS modelHash, model_json, metadata_json';
const EVALUATION = 'id AS evaluationId, reranker_id AS rerankerId, generation, source_revision AS sourceRevision, candidate_id AS candidateId, candidate_hash AS candidateHash, base_model_hash AS baseModelHash, accepted, metadata_json';
const runtimePromises = new Map<boolean | 'auto', Promise<FannRuntime>>();
function sharedRuntime(simd: boolean | 'auto'): Promise<FannRuntime> {
  let pending = runtimePromises.get(simd);
  if (!pending) {
    pending = loadRuntime({ simd }).catch(error => { runtimePromises.delete(simd); throw error; });
    runtimePromises.set(simd, pending);
  }
  return pending;
}
function hash(json: string): string { return createHash('sha256').update(json).digest('hex'); }
function encoded(value: unknown): string {
  const json = JSON.stringify(value);
  if (Buffer.byteLength(json) > 262_144) throw new Error('Reranker metadata exceeds byte limit');
  return json;
}
function integer(value: unknown, field: string, min = 1, max = Number.MAX_SAFE_INTEGER): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) throw new Error(`${field} must be an integer from ${min} to ${max}`);
}
function finite(value: unknown, field: string, min = 0, max = 1): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new Error(`${field} must be finite in [${min}, ${max}]`);
}
function identifier(value: unknown, field = 'id'): asserts value is string {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value)) throw new Error(`${field} must contain 1–128 safe ASCII characters`);
}
/** No user getters, iterators, toJSON, or prototype fields participate in snapshots. */
function object(value: unknown, keys: readonly string[], required: readonly string[], field: string): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error(`${field} must be a plain object`);
  const names = Reflect.ownKeys(value);
  if (names.length > keys.length) throw new Error(`Unknown ${field} keys`);
  const result: Record<string, any> = Object.create(null);
  for (const key of names) {
    if (typeof key !== 'string' || !keys.includes(key)) throw new Error(`Unknown ${field} key`);
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!Object.hasOwn(descriptor, 'value')) throw new Error(`${field} accessors are not allowed`);
    result[key] = descriptor.value;
  }
  for (const key of required) if (!Object.hasOwn(result, key)) throw new Error(`Missing ${field}.${key}`);
  return result;
}
function array(value: unknown, min: number, max: number, field: string): unknown[] {
  if (!Array.isArray(value) || ![Array.prototype, null].includes(Object.getPrototypeOf(value))) throw new Error(`${field} must be an array`);
  const length = Object.getOwnPropertyDescriptor(value, 'length')!.value;
  integer(length, `${field} length`, min, max);
  if (Reflect.ownKeys(value).length !== length + 1) throw new Error(`${field} must contain only dense data elements`);
  const result: unknown[] = [];
  for (let index = 0; index < length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw new Error(`${field} holes/accessors are not allowed`);
    result.push(descriptor.value);
  }
  return result;
}
function candidateSnapshot(value: unknown, labeled: boolean): RerankCandidate | LabeledRerankCandidate {
  const candidate = object(value, labeled ? ['id', 'features', 'relevance'] : ['id', 'features'], labeled ? ['id', 'features', 'relevance'] : ['id', 'features'], 'candidate');
  identifier(candidate.id, 'candidate.id');
  const features = object(candidate.features, FEATURE_KEYS, FEATURE_KEYS, 'features');
  const owned = {} as RerankFeatures;
  for (const key of FEATURE_KEYS) { finite(features[key], key); owned[key] = features[key]; }
  if (labeled) { integer(candidate.relevance, 'relevance', 0, 3); return { id: candidate.id, features: owned, relevance: candidate.relevance }; }
  return { id: candidate.id, features: owned };
}
function candidatesSnapshot(value: unknown, min: number, labeled: boolean): (RerankCandidate | LabeledRerankCandidate)[] {
  const candidates = array(value, min, 64, 'candidates').map(item => candidateSnapshot(item, labeled));
  if (new Set(candidates.map(item => item.id)).size !== candidates.length) throw new Error('Duplicate candidate ID');
  return candidates;
}
function groupsSnapshot(value: unknown, evaluation: boolean): RerankGroup[] {
  let count = 0;
  const groups = array(value, evaluation ? 5 : 1, 128, 'groups').map(item => {
    const group = object(item, ['queryId', 'candidates'], ['queryId', 'candidates'], 'group');
    identifier(group.queryId, 'queryId');
    const candidates = candidatesSnapshot(group.candidates, 2, true) as LabeledRerankCandidate[];
    count += candidates.length;
    if (count > 512) throw new Error('Total candidate limit is 512');
    if (evaluation && !candidates.some(candidate => candidate.relevance > 0)) throw new Error('Each evaluation group requires a positive judgment');
    return { queryId: group.queryId, candidates };
  });
  if (new Set(groups.map(group => group.queryId)).size !== groups.length) throw new Error('Duplicate query ID');
  return groups;
}
function rankingPolicy(value: unknown): Required<RankingPolicy> {
  const source = object(value === undefined ? {} : value, ['k', 'minNdcgImprovement', 'maxMrrRegression', 'maxPerQueryNdcgRegression'], [], 'policy');
  const policy = { k: source.k === undefined ? 10 : source.k, minNdcgImprovement: source.minNdcgImprovement === undefined ? 0.01 : source.minNdcgImprovement, maxMrrRegression: source.maxMrrRegression === undefined ? 0 : source.maxMrrRegression, maxPerQueryNdcgRegression: source.maxPerQueryNdcgRegression === undefined ? 0.1 : source.maxPerQueryNdcgRegression };
  integer(policy.k, 'k', 1, 64);
  finite(policy.minNdcgImprovement, 'minNdcgImprovement', Number.MIN_VALUE, 1);
  finite(policy.maxMrrRegression, 'maxMrrRegression'); finite(policy.maxPerQueryNdcgRegression, 'maxPerQueryNdcgRegression');
  return policy;
}
function seededModel(): DenseModel {
  let state = 0x5eed1234;
  const weight = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return ((state >>> 0) / 0x1_0000_0000 - 0.5) * 0.5; };
  return { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 4, outputs: 8, weights: Array.from({ length: 32 }, weight), bias: Array(8).fill(0), activation: 'Tanh', steepness: 1 },
    { inputs: 8, outputs: 1, weights: Array.from({ length: 8 }, weight), bias: [0], activation: 'Linear', steepness: 1 },
  ] };
}
function metrics(candidates: LabeledRerankCandidate[], order: number[], k: number): RankingMetrics {
  const cutoff = Math.min(k, candidates.length);
  let dcg = 0; let idealDcg = 0; let mrr = 0;
  const ideal = candidates.map(candidate => candidate.relevance).sort((a, b) => b - a);
  for (let index = 0; index < cutoff; index++) {
    const relevance = candidates[order[index]].relevance;
    dcg += (2 ** relevance - 1) / Math.log2(index + 2);
    idealDcg += (2 ** ideal[index] - 1) / Math.log2(index + 2);
    if (relevance > 0 && mrr === 0) mrr = 1 / (index + 1);
  }
  return { ndcg: idealDcg ? dcg / idealDcg : 0, mrr };
}
function average(values: RankingMetrics[]): RankingMetrics {
  return { ndcg: values.reduce((sum, item) => sum + item.ndcg, 0) / values.length, mrr: values.reduce((sum, item) => sum + item.mrr, 0) / values.length };
}

export class NeuralReranker {
  private disposed = false;
  private inFlight = 0;
  private readonly readonly: boolean;
  private readonly simd: boolean | 'auto';
  private readonly cache = new Map<string, PreparedModel>();
  constructor(private readonly db: IDatabaseConnection, options: NeuralRerankerOptions = {}) {
    const config = object(options, ['readonly', 'simd'], [], 'options');
    if (config.readonly !== undefined && typeof config.readonly !== 'boolean') throw new Error('readonly must be boolean');
    this.readonly = config.readonly ?? false; this.simd = config.simd === undefined ? 'auto' : config.simd;
    if (![true, false, 'auto'].includes(this.simd)) throw new Error('Invalid SIMD option');
    if (!this.readonly) db.exec(`
      CREATE TABLE IF NOT EXISTS neural_rerankers (id TEXT PRIMARY KEY, feature_schema TEXT NOT NULL, generation TEXT NOT NULL, revision INTEGER NOT NULL, model_hash TEXT, model_json TEXT, training_query_ids TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS neural_rerank_candidates (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, reranker_id TEXT NOT NULL, generation TEXT NOT NULL, source_revision INTEGER NOT NULL, model_hash TEXT NOT NULL, model_json TEXT NOT NULL, metadata_json TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS neural_rerank_candidates_owner ON neural_rerank_candidates(reranker_id,generation,sequence);
      CREATE TABLE IF NOT EXISTS neural_rerank_evaluations (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, reranker_id TEXT NOT NULL, generation TEXT NOT NULL, source_revision INTEGER NOT NULL, candidate_id TEXT NOT NULL, candidate_hash TEXT NOT NULL, base_model_hash TEXT, accepted INTEGER NOT NULL, metadata_json TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS neural_rerank_evaluations_owner ON neural_rerank_evaluations(reranker_id,generation,sequence);
      CREATE TABLE IF NOT EXISTS neural_rerank_checkpoints (reranker_id TEXT NOT NULL,generation TEXT NOT NULL,revision INTEGER NOT NULL,model_hash TEXT,model_json TEXT,operation TEXT NOT NULL,source_revision INTEGER,evaluation_id TEXT,evaluation_json TEXT,training_query_ids TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(reranker_id,generation,revision));
    `);
  }

  create(id: string): RerankerRecord {
    this.writable(); identifier(id);
    const now = new Date().toISOString(); const generation = randomUUID();
    return inTransaction(this.db, () => {
      this.statement('INSERT INTO neural_rerankers (id,feature_schema,generation,revision,model_hash,model_json,training_query_ids,created_at,updated_at) VALUES (?,?,?,1,NULL,NULL,?,?,?)', stmt => stmt.run(id, RERANK_FEATURE_SCHEMA, generation, '[]', now, now));
      for (const table of ['neural_rerank_candidates', 'neural_rerank_evaluations', 'neural_rerank_checkpoints']) this.statement(`DELETE FROM ${table} WHERE reranker_id = ? AND generation <> ?`, stmt => stmt.run(id, generation));
      this.checkpoint(id, generation, 1, null, null, [], 'create', null, null, null, now);
      return this.record(this.required(id));
    });
  }
  get(id: string): RerankerRecord | null {
    this.alive(); identifier(id);
    return inTransaction(this.db, () => { const source = this.read(id); return source ? this.record(source) : null; });
  }
  list(limit = 50): RerankerMetadata[] {
    this.alive(); integer(limit, 'limit', 1, 100);
    return this.statement<RerankerMetadata, RerankerMetadata[]>(`SELECT ${META} FROM neural_rerankers ORDER BY id LIMIT ?`, stmt => stmt.all(limit));
  }

  async train(id: string, groups: RerankGroup[], options: TrainingOptions = {}): Promise<TrainingCandidate> {
    this.writable(); identifier(id); this.enter();
    try {
      const owned = groupsSnapshot(groups, false);
      const config = object(options, ['epochs', 'learningRate'], [], 'training options');
      const epochs = config.epochs === undefined ? 250 : config.epochs; const learningRate = config.learningRate === undefined ? 0.01 : config.learningRate;
      integer(epochs, 'epochs', 1, 1000); finite(learningRate, 'learningRate', Number.MIN_VALUE, 1);
      const source = this.required(id);
      const labeled = owned.flatMap(group => group.candidates); const rows = labeled.length;
      if (49 * rows * epochs > 50_000_000) throw new Error('Training compute budget exceeded');
      const inputs = Float64Array.from(labeled.flatMap(candidate => FEATURE_KEYS.map(key => candidate.features[key])));
      const targets = Float64Array.from(labeled, candidate => candidate.relevance / 3);
      const datasetHash = hash(encoded({ schema: RERANK_FEATURE_SCHEMA, groups: owned }));
      const runtime = await this.runtime();
      const result = runtime.train(seededModel(), { inputs, targets, rows, epochs, learningRate, optimizer: 'adamw' });
      finite(result.initialLoss, 'initial loss', 0, Number.MAX_VALUE); finite(result.finalLoss, 'final loss', 0, Number.MAX_VALUE);
      const modelJson = encoded(result.model); const modelHash = hash(modelJson);
      const prepared = this.prepared(runtime, modelHash, modelJson);
      if (prepared.inputSize !== 4 || prepared.outputSize !== 1 || result.parameterCount !== 49) throw new Error('Invalid trained reranker dimensions');
      this.writable();
      const candidate: TrainingCandidate = { origin: 'training', featureSchema: RERANK_FEATURE_SCHEMA, candidateId: randomUUID(), rerankerId: id, sourceRevision: source.revision, modelHash,
        trainingQueryIds: owned.map(group => group.queryId), datasetHash, initialLoss: result.initialLoss, finalLoss: result.finalLoss,
        epochs, parameterCount: result.parameterCount, createdAt: new Date().toISOString(), backend: runtime.backend };
      this.persistCandidate(source, candidate, modelJson);
      return candidate;
    } finally { this.inFlight--; }
  }

  /** Inspect seed/active/candidate weights without activating a model or training. */
  async inspectWeights(id: string, candidateId?: string): Promise<WeightInspection> {
    this.alive(); identifier(id); if (candidateId !== undefined) identifier(candidateId, 'candidateId'); this.enter();
    try {
      const source = this.required(id); const base = this.weightSource(source, candidateId);
      const baseHash = await fingerprintModel(base.model);
      this.alive(); this.current(source);
      if (base.candidate) this.sameCandidate(base.candidate, this.candidate(source, base.candidate.candidateId));
      return { rerankerId: id, revision: source.revision, featureSchema: RERANK_FEATURE_SCHEMA,
        model: base.model, baseHash, source: base.source, ...(candidateId === undefined ? {} : { candidateId }) };
    } finally { this.inFlight--; }
  }

  /** Direct bounded WSP edits produce a candidate, never implicit gradients or promotion. */
  async proposePatch(id: string, patch: WeightPatch, baseCandidateId?: string): Promise<TrainingCandidate> {
    this.writable(); identifier(id); if (baseCandidateId !== undefined) identifier(baseCandidateId, 'baseCandidateId'); this.enter();
    try {
      const source = this.required(id); const base = this.weightSource(source, baseCandidateId);
      const owned = this.patchSnapshot(patch);
      // Published WSP validates hash, duplicate locations, shapes, magnitude and delta,
      // and synchronously snapshots everything before its first await.
      const model = await applyWeightPatch(base.model, owned);
      this.alive();
      const modelHash = await fingerprintModel(model);
      const modelJson = encoded(model);
      if (modelHash !== hash(modelJson)) throw new Error('WSP canonical model hash mismatch');
      const runtime = await this.runtime();
      this.prepared(runtime, modelHash, modelJson);
      this.writable();
      const candidate: TrainingCandidate = { origin: 'weight-patch', featureSchema: RERANK_FEATURE_SCHEMA,
        candidateId: randomUUID(), rerankerId: id, sourceRevision: source.revision, modelHash,
        trainingQueryIds: [...base.trainingQueryIds], patch: owned, baseHash: owned.baseHash,
        parameterCount: 49, createdAt: new Date().toISOString(), backend: runtime.backend };
      this.persistCandidate(source, candidate, modelJson, base.candidate);
      return candidate;
    } finally { this.inFlight--; }
  }

  async evaluate(id: string, candidateId: string, groups: RerankGroup[], policy?: RankingPolicy): Promise<RankingEvaluation> {
    this.writable(); identifier(id); identifier(candidateId, 'candidateId'); this.enter();
    try {
      const owned = groupsSnapshot(groups, true); const criteria = rankingPolicy(policy);
      const source = this.required(id); const stored = this.candidate(source, candidateId);
      const trained = JSON.parse(stored.metadata_json) as TrainingCandidate;
      const trainingIds = new Set(trained.trainingQueryIds);
      if (owned.some(group => trainingIds.has(group.queryId))) throw new Error('Evaluation query IDs overlap training; held-out groups required');
      const runtime = await this.runtime();
      const candidateModel = this.prepared(runtime, stored.modelHash, stored.model_json);
      const activeModel = source.modelHash === null ? null : this.prepared(runtime, source.modelHash, source.model_json!);
      const perQuery = owned.map(group => {
        const baseline = metrics(group.candidates, group.candidates.map((_, index) => index), criteria.k);
        const candidate = metrics(group.candidates, this.scored(candidateModel, group.candidates).order, criteria.k);
        const active = activeModel ? metrics(group.candidates, this.scored(activeModel, group.candidates).order, criteria.k) : null;
        return { queryId: group.queryId, baseline, candidate, active };
      });
      const baseline = average(perQuery.map(group => group.baseline));
      const candidate = average(perQuery.map(group => group.candidate));
      const active = activeModel ? average(perQuery.map(group => group.active!)) : null;
      const reasons: string[] = [];
      for (const comparison of ['baseline', 'active'] as const) {
        const reference = comparison === 'baseline' ? baseline : active;
        if (!reference) continue;
        if (candidate.ndcg - reference.ndcg < criteria.minNdcgImprovement) reasons.push(`Insufficient mean NDCG improvement versus ${comparison}`);
        if (reference.mrr - candidate.mrr > criteria.maxMrrRegression) reasons.push(`MRR regression versus ${comparison}`);
        for (const group of perQuery) if (group[comparison]!.ndcg - group.candidate.ndcg > criteria.maxPerQueryNdcgRegression) reasons.push(`Per-query NDCG regression versus ${comparison}: ${group.queryId}`);
      }
      const evaluation: RankingEvaluation = { featureSchema: RERANK_FEATURE_SCHEMA, evaluationId: randomUUID(), rerankerId: id, candidateId, sourceRevision: source.revision,
        baseModelHash: source.modelHash, candidateHash: stored.modelHash, datasetHash: hash(encoded({ schema: RERANK_FEATURE_SCHEMA, groups: owned })),
        policyHash: hash(encoded(criteria)), policy: criteria, baseline, active, candidate, perQuery, accepted: reasons.length === 0, reasons,
        createdAt: new Date().toISOString(), backend: runtime.backend };
      this.writable();
      inTransaction(this.db, () => {
        this.current(source); this.sameCandidate(stored, this.candidate(source, candidateId));
        this.statement('INSERT INTO neural_rerank_evaluations (id,reranker_id,generation,source_revision,candidate_id,candidate_hash,base_model_hash,accepted,metadata_json) VALUES (?,?,?,?,?,?,?,?,?)', stmt => stmt.run(evaluation.evaluationId, id, source.generation, source.revision, candidateId, stored.modelHash, source.modelHash, evaluation.accepted ? 1 : 0, encoded(evaluation)));
        this.statement('DELETE FROM neural_rerank_evaluations WHERE reranker_id = ? AND sequence NOT IN (SELECT sequence FROM neural_rerank_evaluations WHERE reranker_id = ? AND generation = ? ORDER BY sequence DESC LIMIT 16)', stmt => stmt.run(id, id, source.generation));
      });
      return evaluation;
    } finally { this.inFlight--; }
  }

  async promote(id: string, evaluationId: string, expectedRevision: number): Promise<RerankerRecord> {
    this.writable(); identifier(id); identifier(evaluationId, 'evaluationId'); integer(expectedRevision, 'expectedRevision'); this.enter();
    try {
      const source = this.required(id);
      if (source.revision !== expectedRevision) throw new Error('Stale promotion revision');
      const receipt = this.evaluation(source, evaluationId);
      const evidence = JSON.parse(receipt.metadata_json) as RankingEvaluation;
      if (evidence.featureSchema !== RERANK_FEATURE_SCHEMA) throw new Error('Unsupported evaluation feature schema');
      if (receipt.accepted !== 1 || evidence.accepted !== true) throw new Error('Evaluation rejected promotion');
      const stored = this.candidate(source, receipt.candidateId);
      if (receipt.baseModelHash !== source.modelHash || receipt.candidateHash !== stored.modelHash || evidence.evaluationId !== evaluationId || evidence.candidateId !== stored.candidateId || evidence.candidateHash !== stored.modelHash || evidence.baseModelHash !== source.modelHash || evidence.sourceRevision !== source.revision) throw new Error('Evaluation lineage mismatch');
      const runtime = await this.runtime();
      this.prepared(runtime, stored.modelHash, stored.model_json);
      this.writable();
      return inTransaction(this.db, () => {
        this.current(source); this.sameCandidate(stored, this.candidate(source, stored.candidateId));
        const latest = this.evaluation(source, evaluationId);
        if (encoded(latest) !== encoded(receipt)) throw new Error('Evaluation changed during promotion');
        return this.advance(source, stored.modelHash, stored.model_json, this.trainingIds(JSON.parse(stored.metadata_json).trainingQueryIds), 'promote', source.revision, evaluationId, receipt.metadata_json);
      });
    } finally { this.inFlight--; }
  }

  rollback(id: string, targetRevision: number, expectedRevision: number): RerankerRecord {
    this.writable(); identifier(id); integer(targetRevision, 'targetRevision'); integer(expectedRevision, 'expectedRevision');
    return inTransaction(this.db, () => {
      const source = this.required(id);
      if (source.revision !== expectedRevision) throw new Error('Stale rollback revision');
      const checkpoint = this.statement<{ model_hash: string | null; model_json: string | null; training_query_ids: string }, { model_hash: string | null; model_json: string | null; training_query_ids: string } | undefined>('SELECT model_hash,model_json,training_query_ids FROM neural_rerank_checkpoints WHERE reranker_id = ? AND generation = ? AND revision = ?', stmt => stmt.get(id, source.generation, targetRevision));
      if (!checkpoint) throw new Error('Rollback checkpoint is not retained');
      if ((checkpoint.model_hash === null) !== (checkpoint.model_json === null) || (checkpoint.model_json !== null && hash(checkpoint.model_json) !== checkpoint.model_hash)) throw new Error('Checkpoint hash mismatch');
      return this.advance(source, checkpoint.model_hash, checkpoint.model_json, this.trainingIds(JSON.parse(checkpoint.training_query_ids)), 'rollback', targetRevision, null, null);
    });
  }

  async rank(id: string, candidates: RerankCandidate[], options: RankOptions = {}): Promise<RankResult> {
    this.alive(); identifier(id); this.enter();
    try {
      const owned = candidatesSnapshot(candidates, 0, false) as RerankCandidate[];
      const config = object(options, ['mode', 'candidateId'], [], 'rank options');
      const mode = config.mode === undefined ? 'shadow' : config.mode;
      if (mode !== 'shadow' && mode !== 'active') throw new Error('Invalid reranking mode');
      if (config.candidateId !== undefined) identifier(config.candidateId, 'candidateId');
      if (mode === 'active' && config.candidateId !== undefined) throw new Error('candidateId is only permitted in shadow mode');
      const source = this.required(id);
      const candidate = config.candidateId === undefined ? null : this.candidate(source, config.candidateId);
      const modelHash = candidate ? candidate.modelHash : source.modelHash;
      const modelJson = candidate ? candidate.model_json : source.model_json;
      const baseline = owned.map(item => item.id);
      if (modelHash === null) return { order: baseline, scores: [], revision: source.revision, mode, modelHash: null, available: false, ...(mode === 'shadow' ? { proposedOrder: [...baseline] } : {}) };
      const runtime = await this.runtime();
      const prepared = this.prepared(runtime, modelHash, modelJson!);
      const scored = this.scored(prepared, owned);
      this.alive(); this.current(source);
      if (candidate) this.sameCandidate(candidate, this.candidate(source, candidate.candidateId));
      const proposedOrder = scored.order.map(index => owned[index].id);
      return { order: mode === 'active' ? proposedOrder : baseline, scores: owned.map((item, index) => ({ id: item.id, score: scored.values[index] })), revision: source.revision, mode, modelHash, available: true, ...(mode === 'shadow' ? { proposedOrder } : {}) };
    } finally { this.inFlight--; }
  }

  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    for (const prepared of this.cache.values()) prepared.dispose(); this.cache.clear();
  }
  private trainingIds(value: unknown): string[] {
    const ids = array(value, 0, 128, 'trainingQueryIds');
    const owned = ids.map(id => { identifier(id, 'training query ID'); return id; });
    if (new Set(owned).size !== owned.length) throw new Error('Duplicate training query ID');
    return owned;
  }
  private weightSource(source: Source, candidateId?: string): { model: DenseModel; source: WeightInspection['source']; trainingQueryIds: string[]; candidate?: CandidateRow } {
    const candidate = candidateId === undefined ? undefined : this.candidate(source, candidateId);
    if (candidate) return { model: JSON.parse(candidate.model_json), source: 'candidate', candidate,
      trainingQueryIds: this.trainingIds(JSON.parse(candidate.metadata_json).trainingQueryIds) };
    if (source.modelHash === null) {
      if (source.model_json !== null) throw new Error('Baseline model binding mismatch');
      return { model: seededModel(), source: 'seed', trainingQueryIds: [] };
    }
    if (source.model_json === null || hash(source.model_json) !== source.modelHash) throw new Error('Active reranker model hash mismatch');
    return { model: JSON.parse(source.model_json), source: 'active', trainingQueryIds: this.trainingIds(JSON.parse(source.training_query_ids)) };
  }
  private patchSnapshot(value: unknown): WeightPatch {
    const patch = object(value, ['schema', 'baseHash', 'edits'], ['schema', 'baseHash', 'edits'], 'patch');
    if (patch.schema !== 'ruv-fann.weight-patch/1' || typeof patch.baseHash !== 'string' || !/^[a-f0-9]{64}$/.test(patch.baseHash)) throw new Error('Invalid weight patch schema or baseHash');
    const edits = array(patch.edits, 1, 128, 'edits').map(item => {
      const edit = object(item, ['layer', 'parameter', 'index', 'value'], ['layer', 'parameter', 'index', 'value'], 'edit');
      integer(edit.layer, 'edit layer', 0, 63); integer(edit.index, 'edit index', 0, 65535);
      if (edit.parameter !== 'weights' && edit.parameter !== 'bias') throw new Error('Invalid patch parameter');
      finite(edit.value, 'edit value', -1_000_000, 1_000_000);
      return { layer: edit.layer, parameter: edit.parameter as 'weights' | 'bias', index: edit.index, value: edit.value };
    });
    return { schema: 'ruv-fann.weight-patch/1', baseHash: patch.baseHash, edits };
  }
  private persistCandidate(source: Source, candidate: TrainingCandidate, modelJson: string, baseCandidate?: CandidateRow): void {
    inTransaction(this.db, () => {
      this.current(source);
      if (baseCandidate) this.sameCandidate(baseCandidate, this.candidate(source, baseCandidate.candidateId));
      this.statement('INSERT INTO neural_rerank_candidates (id,reranker_id,generation,source_revision,model_hash,model_json,metadata_json) VALUES (?,?,?,?,?,?,?)', stmt => stmt.run(candidate.candidateId, source.id, source.generation, source.revision, candidate.modelHash, modelJson, encoded(candidate)));
      this.statement('DELETE FROM neural_rerank_candidates WHERE reranker_id = ? AND sequence NOT IN (SELECT sequence FROM neural_rerank_candidates WHERE reranker_id = ? AND generation = ? ORDER BY sequence DESC LIMIT 16)', stmt => stmt.run(source.id, source.id, source.generation));
      this.statement('DELETE FROM neural_rerank_evaluations WHERE reranker_id = ? AND candidate_id NOT IN (SELECT id FROM neural_rerank_candidates WHERE reranker_id = ? AND generation = ?)', stmt => stmt.run(source.id, source.id, source.generation));
    });
  }
  private prepared(runtime: FannRuntime, modelHash: string, json: string): PreparedModel {
    if (typeof json !== 'string' || Buffer.byteLength(json) > 65_536 || hash(json) !== modelHash) throw new Error('Stored reranker model hash mismatch');
    const cached = this.cache.get(modelHash);
    if (cached) { this.cache.delete(modelHash); this.cache.set(modelHash, cached); return cached; }
    const prepared = runtime.prepare(json, { maxRows: 64 });
    if (prepared.inputSize !== 4 || prepared.outputSize !== 1 || prepared.estimatedBytes > 1024 * 1024) { prepared.dispose(); throw new Error('Invalid reranker workspace'); }
    while (this.cache.size >= 8) { const oldest = this.cache.keys().next().value!; this.cache.get(oldest)!.dispose(); this.cache.delete(oldest); }
    this.cache.set(modelHash, prepared); return prepared;
  }
  private scored(prepared: PreparedModel, candidates: RerankCandidate[]): { values: number[]; order: number[] } {
    if (!candidates.length) return { values: [], order: [] };
    const inputs = Float64Array.from(candidates.flatMap(candidate => FEATURE_KEYS.map(key => candidate.features[key])));
    const values = Array.from(prepared.predict(inputs, candidates.length));
    if (values.length !== candidates.length || values.some(value => !Number.isFinite(value))) throw new Error('Non-finite reranker scores');
    return { values, order: values.map((_, index) => index).sort((a, b) => values[b] - values[a] || a - b) };
  }
  private advance(source: Source, modelHash: string | null, json: string | null, trainingQueryIds: string[], operation: 'promote' | 'rollback', sourceRevision: number, evaluationId: string | null, evaluationJson: string | null): RerankerRecord {
    const revision = source.revision + 1; integer(revision, 'next revision'); const now = new Date().toISOString();
    const changes = this.statement('UPDATE neural_rerankers SET revision = ?,model_hash = ?,model_json = ?,training_query_ids = ?,updated_at = ? WHERE id = ? AND generation = ? AND revision = ? AND model_hash IS ?', stmt => stmt.run(revision, modelHash, json, encoded(trainingQueryIds), now, source.id, source.generation, source.revision, source.modelHash).changes);
    if (changes !== 1) throw new Error('Reranker changed during update');
    this.checkpoint(source.id, source.generation, revision, modelHash, json, trainingQueryIds, operation, sourceRevision, evaluationId, evaluationJson, now);
    this.statement('DELETE FROM neural_rerank_checkpoints WHERE reranker_id = ? AND generation = ? AND revision NOT IN (SELECT revision FROM neural_rerank_checkpoints WHERE reranker_id = ? AND generation = ? ORDER BY revision DESC LIMIT 32)', stmt => stmt.run(source.id, source.generation, source.id, source.generation));
    return this.record(this.required(source.id));
  }
  private checkpoint(id: string, generation: string, revision: number, modelHash: string | null, json: string | null, trainingQueryIds: string[], operation: RerankerHistory['operation'], sourceRevision: number | null, evaluationId: string | null, evaluation: string | null, now: string): void {
    this.statement('INSERT INTO neural_rerank_checkpoints (reranker_id,generation,revision,model_hash,model_json,operation,source_revision,evaluation_id,evaluation_json,training_query_ids,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)', stmt => stmt.run(id, generation, revision, modelHash, json, operation, sourceRevision, evaluationId, evaluation, encoded(trainingQueryIds), now));
  }
  private record(source: Source): RerankerRecord {
    const candidates = this.statement<{ metadata_json: string }, { metadata_json: string }[]>('SELECT metadata_json FROM neural_rerank_candidates WHERE reranker_id = ? AND generation = ? ORDER BY sequence DESC LIMIT 16', stmt => stmt.all(source.id, source.generation));
    const evaluations = this.statement<{ metadata_json: string }, { metadata_json: string }[]>('SELECT metadata_json FROM neural_rerank_evaluations WHERE reranker_id = ? AND generation = ? ORDER BY sequence DESC LIMIT 16', stmt => stmt.all(source.id, source.generation));
    const history = this.statement<any, any[]>('SELECT revision,model_hash AS modelHash,operation,source_revision AS sourceRevision,evaluation_id AS evaluationId,evaluation_json,training_query_ids,created_at AS createdAt FROM neural_rerank_checkpoints WHERE reranker_id = ? AND generation = ? ORDER BY revision DESC LIMIT 32', stmt => stmt.all(source.id, source.generation));
    const { generation, model_json, training_query_ids, ...metadata } = source;
    return { ...metadata, trainingQueryIds: this.trainingIds(JSON.parse(training_query_ids)), model: model_json === null ? null : JSON.parse(model_json), candidates: candidates.map(row => JSON.parse(row.metadata_json)), evaluations: evaluations.map(row => JSON.parse(row.metadata_json)), history: history.map(({ evaluation_json, training_query_ids, ...entry }) => ({ ...entry, trainingQueryIds: this.trainingIds(JSON.parse(training_query_ids)), evaluation: evaluation_json === null ? null : JSON.parse(evaluation_json) })) };
  }
  private read(id: string): Source | undefined { return this.statement<Source, Source | undefined>(`SELECT ${META},generation,model_json,training_query_ids FROM neural_rerankers WHERE id = ?`, stmt => stmt.get(id)); }
  private required(id: string): Source { const source = this.read(id); if (!source) throw new Error(`Reranker ${id} not found`); if (source.featureSchema !== RERANK_FEATURE_SCHEMA) throw new Error('Unsupported reranker feature schema'); return source; }
  private candidate(source: Source, id: string): CandidateRow {
    const row = this.statement<CandidateRow, CandidateRow | undefined>(`SELECT ${CANDIDATE} FROM neural_rerank_candidates WHERE id = ? AND reranker_id = ? AND generation = ?`, stmt => stmt.get(id, source.id, source.generation));
    if (!row) throw new Error('Training candidate not found in this generation');
    if (row.sourceRevision !== source.revision) throw new Error('Training candidate is stale');
    if (hash(row.model_json) !== row.modelHash) throw new Error('Training candidate model hash mismatch');
    if (JSON.parse(row.metadata_json).featureSchema !== RERANK_FEATURE_SCHEMA) throw new Error('Unsupported candidate feature schema');
    return row;
  }
  private evaluation(source: Source, id: string): EvaluationRow {
    const row = this.statement<EvaluationRow, EvaluationRow | undefined>(`SELECT ${EVALUATION} FROM neural_rerank_evaluations WHERE id = ? AND reranker_id = ? AND generation = ?`, stmt => stmt.get(id, source.id, source.generation));
    if (!row) throw new Error('Evaluation not found in this generation');
    if (row.sourceRevision !== source.revision) throw new Error('Evaluation is stale');
    return row;
  }
  private sameCandidate(expected: CandidateRow, actual: CandidateRow): void {
    if (encoded(expected) !== encoded(actual)) throw new Error('Training candidate changed during operation');
  }
  private current(source: Source): void {
    const current = this.read(source.id);
    if (!current || current.generation !== source.generation || current.featureSchema !== source.featureSchema || current.revision !== source.revision || current.modelHash !== source.modelHash || current.model_json !== source.model_json || current.training_query_ids !== source.training_query_ids) throw new Error('Reranker changed during operation; stale generation/revision');
  }
  private async runtime(): Promise<FannRuntime> { this.alive(); const runtime = await sharedRuntime(this.simd); this.alive(); return runtime; }
  private statement<T = any, R = any>(sql: string, fn: (stmt: IPreparedStatement<T>) => R): R { const statement = this.db.prepare<T>(sql); try { return fn(statement); } finally { statement.finalize?.(); } }
  private enter(): void { this.alive(); if (this.inFlight >= 4) throw new Error('Too many in-flight reranker operations'); this.inFlight++; }
  private alive(): void { if (this.disposed) throw new Error('NeuralReranker is disposed'); }
  private writable(): void { this.alive(); if (this.readonly) throw new Error('NeuralReranker is read-only'); }
}
