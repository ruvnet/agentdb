/** Durable, gradient-free weight proposals evaluated by real ruv-FANN WASM. */
import { createHash, randomUUID } from 'node:crypto';
import { fingerprintModel, evaluateWeightPatch, type WeightPatch, type EvaluationDataset,
  type EvaluationPolicy, type WeightEvaluation } from 'ruv-fann-wsp';
import type { DenseModel } from 'ruv-fann';
import type { IDatabaseConnection, IPreparedStatement } from '../types/database.types.js';
import { inTransaction } from './crud-helpers.js';

export type { WeightPatch, PatchLimits, EvaluationDataset, EvaluationPolicy, WeightEvaluation } from 'ruv-fann-wsp';
export interface WeightProgramsOptions {
  readonly?: boolean;
  simd?: boolean | 'auto';
  /** Retention limits may only be tightened. */
  maxProposals?: number;
  maxHistory?: number;
  maxProposalAgeMs?: number;
}
export interface WeightProgramMetadata {
  id: string; revision: number; hash: string; createdAt: string; updatedAt: string;
}
export type StoredWeightEvaluation = Omit<WeightEvaluation, 'model'>;
export interface WeightProgramHistory {
  revision: number; hash: string; operation: 'create' | 'promote' | 'rollback';
  sourceRevision: number | null; proposalId: string | null; createdAt: string;
  evaluation: StoredWeightEvaluation | null;
}
export interface WeightProgramProposal {
  id: string; revision: number; baseHash: string; candidateHash: string;
  accepted: boolean; status: 'pending' | 'rejected' | 'stale' | 'promoted' | 'expired';
  createdAt: string; expiresAt: string; evaluation: StoredWeightEvaluation;
}
export interface WeightProgramRecord extends WeightProgramMetadata {
  model: DenseModel; history: WeightProgramHistory[]; proposals: WeightProgramProposal[];
}
export interface WeightProgramProposed {
  id: string; proposalId: string; revision: number; evaluation: WeightEvaluation;
  createdAt: string; expiresAt: string;
}
type ProgramRow = WeightProgramMetadata & { generation: string; model_json: string };
type CheckpointRow = {
  revision: number; hash: string; model_json: string; operation: WeightProgramHistory['operation'];
  sourceRevision: number | null; proposalId: string | null; createdAt: string; evaluation_json: string | null;
};
type ProposalRow = {
  id: string; revision: number; baseHash: string; candidateHash: string; accepted: number;
  candidate_json: string; evaluation_json: string; createdAt: string; expiresAt: string;
};
const PROGRAM_COLUMNS = 'id, revision, model_hash AS hash, created_at AS createdAt, updated_at AS updatedAt';
const CHECKPOINT_COLUMNS = 'revision, model_hash AS hash, model_json, operation, source_revision AS sourceRevision, proposal_id AS proposalId, created_at AS createdAt, evaluation_json';
const PROPOSAL_COLUMNS = 'id, base_revision AS revision, base_hash AS baseHash, candidate_hash AS candidateHash, accepted, candidate_json, evaluation_json, created_at AS createdAt, expires_at AS expiresAt';
const MAX_JSON_BYTES = 1024 * 1024;

function integer(value: unknown, field: string, max = Number.MAX_SAFE_INTEGER): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > max) throw new Error(`${field} must be an integer from 1 to ${max}`);
}
function identifier(value: unknown, field = 'id'): asserts value is string {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value)) throw new Error(`${field} must contain 1–128 safe ASCII characters`);
}
/** Preserve unknown keys for SDK validation; reject JSON-unsafe values instead of silently dropping them. */
function snapshotModel(input: DenseModel | string): DenseModel {
  if (typeof input === 'string' && Buffer.byteLength(input) > MAX_JSON_BYTES) throw new Error('Model exceeds 1 MiB JSON limit');
  const source: unknown = typeof input === 'string' ? JSON.parse(input) : input;
  let nodes = 0;
  let estimatedBytes = 0;
  function budget(bytes: number): void {
    estimatedBytes += bytes;
    if (estimatedBytes > MAX_JSON_BYTES) throw new Error('Model exceeds 1 MiB JSON limit');
  }
  // Exact JSON string byte accounting without allocating its escaped form.
  function stringBudget(value: string): void {
    budget(2);
    for (let index = 0; index < value.length; index++) {
      const code = value.charCodeAt(index);
      if (code === 34 || code === 92) budget(2);
      else if (code < 32) budget([8, 9, 10, 12, 13].includes(code) ? 2 : 6);
      else if (code < 128) budget(1);
      else if (code < 2048) budget(2);
      else if (code >= 0xd800 && code <= 0xdbff) {
        const next = value.charCodeAt(index + 1);
        if (next >= 0xdc00 && next <= 0xdfff) { budget(4); index++; }
        else budget(6);
      } else budget(code >= 0xdc00 && code <= 0xdfff ? 6 : 3);
    }
  }
  function clone(value: unknown, depth: number): unknown {
    if (++nodes > 140_000 || depth > 8) throw new Error('Model exceeds structural limits');
    if (typeof value === 'string') {
      if (value.length > MAX_JSON_BYTES) throw new Error('Model string limit exceeded');
      stringBudget(value);
      return value;
    }
    if (value === null || typeof value === 'boolean') { budget(value === null ? 4 : value ? 4 : 5); return value; }
    if (typeof value === 'number' && Number.isFinite(value)) { budget(String(value).length); return value === 0 ? 0 : value; }
    if (Array.isArray(value)) {
      if (Object.getPrototypeOf(value) !== Array.prototype) throw new Error('Model arrays must have the default prototype');
      const length = Object.getOwnPropertyDescriptor(value, 'length')!.value as number;
      if (length > 65_536) throw new Error('Model array limit exceeded');
      budget(length ? length + 1 : 2);
      const keys = Reflect.ownKeys(value);
      if (keys.length !== length + 1) throw new Error('Model arrays must contain only dense indexed data');
      const copy: unknown[] = [];
      for (let index = 0; index < length; index++) {
        const property = Object.getOwnPropertyDescriptor(value, String(index));
        if (!property || !Object.hasOwn(property, 'value')) throw new Error('Model array accessors and holes are not supported');
        copy.push(clone(property.value, depth + 1));
      }
      return copy;
    }
    if (value && typeof value === 'object' && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null)) {
      const keys = Reflect.ownKeys(value);
      if (keys.length > 16) throw new Error('Model object has too many keys');
      budget(keys.length ? keys.length + 1 : 2);
      const copy: Record<string, unknown> = Object.create(null);
      for (const key of keys) {
        if (typeof key !== 'string') throw new Error('Model symbols are not supported');
        if (key.length > 128) throw new Error('Model key limit exceeded');
        stringBudget(key); budget(1);
        if (key === '__proto__' || key === 'constructor' || key === 'prototype') throw new Error('Unsafe model key');
        const property = Object.getOwnPropertyDescriptor(value, key)!;
        if (!Object.hasOwn(property, 'value')) throw new Error('Model accessors are not supported');
        copy[key] = clone(property.value, depth + 1);
      }
      return copy;
    }
    throw new Error('Model must contain finite JSON values');
  }
  const owned = clone(source, 0) as DenseModel;
  if (Buffer.byteLength(JSON.stringify(owned)) > MAX_JSON_BYTES) throw new Error('Model exceeds 1 MiB JSON limit');
  return owned;
}
function canonicalModel(validated: DenseModel): DenseModel {
  return { schema: 'ruv-fann.dense-layers/1', layers: validated.layers.map(layer => ({
    inputs: layer.inputs, outputs: layer.outputs, weights: [...layer.weights],
    bias: layer.bias == null ? Array(layer.outputs).fill(0) : [...layer.bias],
    activation: layer.activation, steepness: layer.steepness,
  })) };
}

/**
 * The database is the trusted authority: promote accepts an internally stored
 * proposal ID, never a caller-authored evaluation receipt. Evaluation datasets
 * are caller-supplied; acceptance does not establish unseen-data generalization.
 * History retains 32 checkpoints and 16 proposals per program by default.
 * Proposals expire after 24 hours; retained promoted evidence also lives in the
 * checkpoint lineage after a proposal is evicted. No gradients or providers run.
 */
export class WeightPrograms {
  private disposed = false;
  private inFlightProposals = 0;
  private readonly readonly: boolean;
  private readonly simd: boolean | 'auto';
  private readonly maxProposals: number;
  private readonly maxHistory: number;
  private readonly maxProposalAgeMs: number;

  constructor(private readonly db: IDatabaseConnection, options: WeightProgramsOptions = {}) {
    for (const key of Object.keys(options)) if (!['readonly', 'simd', 'maxProposals', 'maxHistory', 'maxProposalAgeMs'].includes(key)) throw new Error(`Unknown WeightPrograms option: ${key}`);
    if (options.readonly !== undefined && typeof options.readonly !== 'boolean') throw new Error('readonly must be boolean');
    this.readonly = options.readonly ?? false;
    this.simd = options.simd ?? 'auto';
    if (![true, false, 'auto'].includes(this.simd)) throw new Error('Invalid simd option');
    this.maxProposals = options.maxProposals ?? 16;
    this.maxHistory = options.maxHistory ?? 32;
    this.maxProposalAgeMs = options.maxProposalAgeMs ?? 86_400_000;
    integer(this.maxProposals, 'maxProposals', 16);
    integer(this.maxHistory, 'maxHistory', 32);
    integer(this.maxProposalAgeMs, 'maxProposalAgeMs', 86_400_000);
    if (!this.readonly) db.exec(`
      CREATE TABLE IF NOT EXISTS wsp_programs (
        id TEXT PRIMARY KEY, generation TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision >= 1),
        model_hash TEXT NOT NULL, model_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS wsp_checkpoints (
        program_id TEXT NOT NULL, generation TEXT NOT NULL, revision INTEGER NOT NULL,
        model_hash TEXT NOT NULL, model_json TEXT NOT NULL, operation TEXT NOT NULL,
        source_revision INTEGER, proposal_id TEXT, evaluation_json TEXT, created_at TEXT NOT NULL,
        PRIMARY KEY(program_id, generation, revision)
      );
      CREATE TABLE IF NOT EXISTS wsp_proposals (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
        program_id TEXT NOT NULL, generation TEXT NOT NULL, base_revision INTEGER NOT NULL,
        base_hash TEXT NOT NULL, candidate_hash TEXT NOT NULL, accepted INTEGER NOT NULL,
        candidate_json TEXT NOT NULL, evaluation_json TEXT NOT NULL,
        created_at TEXT NOT NULL, expires_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS wsp_proposals_program ON wsp_proposals(program_id, generation, sequence);
    `);
  }

  async create(id: string, model: DenseModel | string): Promise<WeightProgramRecord> {
    this.assertWritable(); identifier(id);
    const owned = snapshotModel(model);
    const hash = await fingerprintModel(owned);
    this.assertWritable();
    const checkpoint = canonicalModel(owned);
    const json = JSON.stringify(checkpoint);
    if (Buffer.byteLength(json) > MAX_JSON_BYTES) throw new Error('Normalized model exceeds 1 MiB JSON limit');
    const now = new Date().toISOString();
    const generation = randomUUID();
    return inTransaction(this.db, () => {
      this.statement('INSERT INTO wsp_programs (id, generation, revision, model_hash, model_json, created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?, ?)', stmt => stmt.run(id, generation, hash, json, now, now));
      this.statement('DELETE FROM wsp_checkpoints WHERE program_id = ? AND generation <> ?', stmt => stmt.run(id, generation));
      this.statement('DELETE FROM wsp_proposals WHERE program_id = ? AND generation <> ?', stmt => stmt.run(id, generation));
      this.insertCheckpoint(id, generation, 1, hash, json, 'create', null, null, null, now);
      return this.record(this.required(id));
    });
  }

  get(id: string): WeightProgramRecord | null {
    this.assertAlive(); identifier(id);
    return inTransaction(this.db, () => {
      const row = this.read(id);
      return row ? this.record(row) : null;
    });
  }

  list(limit = 50): WeightProgramMetadata[] {
    this.assertAlive(); integer(limit, 'limit', 100);
    return this.statement<WeightProgramMetadata, WeightProgramMetadata[]>(`SELECT ${PROGRAM_COLUMNS} FROM wsp_programs ORDER BY id LIMIT ?`, stmt => stmt.all(limit));
  }

  async propose(id: string, patch: WeightPatch, dataset: EvaluationDataset, policy: EvaluationPolicy): Promise<WeightProgramProposed> {
    this.assertWritable(); identifier(id);
    if (this.inFlightProposals >= this.maxProposals) throw new Error('Too many in-flight weight proposals');
    this.inFlightProposals++;
    try {
      const source = this.required(id);
      // The SDK synchronously snapshots and validates all caller inputs before its
      // first await, then evaluates baseline and candidate in actual WASM.
      const evaluation = await evaluateWeightPatch(source.model_json, patch, dataset, policy, { simd: this.simd });
      this.assertWritable();
      if (evaluation.baseHash !== source.hash) throw new Error('Evaluation source hash mismatch');
      const { model, ...evidence } = evaluation;
      const candidateJson = JSON.stringify(model);
      if (Buffer.byteLength(candidateJson) > MAX_JSON_BYTES) throw new Error('Candidate model exceeds 1 MiB JSON limit');
      const proposalId = randomUUID();
      const now = new Date().toISOString();
      const expiresAt = new Date(Date.now() + this.maxProposalAgeMs).toISOString();
      inTransaction(this.db, () => {
        this.assertCurrent(source);
        this.statement('DELETE FROM wsp_proposals WHERE program_id = ? AND (generation <> ? OR expires_at <= ? OR base_revision <> ?)', stmt => stmt.run(id, source.generation, now, source.revision));
        this.statement('INSERT INTO wsp_proposals (id, program_id, generation, base_revision, base_hash, candidate_hash, accepted, candidate_json, evaluation_json, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
          stmt => stmt.run(proposalId, id, source.generation, source.revision, evaluation.baseHash, evaluation.candidateHash, evaluation.accepted ? 1 : 0, candidateJson, JSON.stringify(evidence), now, expiresAt));
        this.statement('DELETE FROM wsp_proposals WHERE program_id = ? AND generation = ? AND sequence NOT IN (SELECT sequence FROM wsp_proposals WHERE program_id = ? AND generation = ? ORDER BY sequence DESC LIMIT ?)',
          stmt => stmt.run(id, source.generation, id, source.generation, this.maxProposals));
      });
      // No shared references with the stored JSON checkpoint or evaluation.
      return { id, proposalId, revision: source.revision, evaluation, createdAt: now, expiresAt };
    } finally { this.inFlightProposals--; }
  }

  async promote(id: string, proposalId: string, expectedRevision: number): Promise<WeightProgramRecord> {
    this.assertWritable(); identifier(id); identifier(proposalId, 'proposalId'); integer(expectedRevision, 'expectedRevision');
    const source = this.required(id);
    if (source.revision !== expectedRevision) throw new Error('Program revision changed; stale promotion');
    const proposal = this.statement<ProposalRow, ProposalRow | undefined>(`SELECT ${PROPOSAL_COLUMNS} FROM wsp_proposals WHERE id = ? AND program_id = ? AND generation = ?`, stmt => stmt.get(proposalId, id, source.generation));
    if (!proposal) throw new Error('Proposal not found in this program generation');
    if (proposal.revision !== source.revision || proposal.baseHash !== source.hash) throw new Error('Proposal is stale');
    if (proposal.expiresAt <= new Date().toISOString()) throw new Error('Proposal expired');
    const evaluation = JSON.parse(proposal.evaluation_json) as StoredWeightEvaluation;
    if (proposal.accepted !== 1 || evaluation.accepted !== true) throw new Error('Proposal evaluation rejected promotion');
    if (evaluation.schema !== 'ruv-fann.weight-evaluation/1' || evaluation.baseHash !== source.hash || evaluation.candidateHash !== proposal.candidateHash) throw new Error('Proposal lineage mismatch');
    // Recompute the stored candidate hash rather than trusting a detached receipt.
    const candidateHash = await fingerprintModel(proposal.candidate_json);
    this.assertWritable();
    if (candidateHash !== proposal.candidateHash) throw new Error('Stored candidate hash mismatch');
    return inTransaction(this.db, () => {
      this.assertCurrent(source);
      const latest = this.statement<ProposalRow, ProposalRow | undefined>(`SELECT ${PROPOSAL_COLUMNS} FROM wsp_proposals WHERE id = ? AND program_id = ? AND generation = ?`, stmt => stmt.get(proposalId, id, source.generation));
      if (!latest || latest.candidate_json !== proposal.candidate_json || latest.evaluation_json !== proposal.evaluation_json || latest.candidateHash !== candidateHash || latest.accepted !== 1) throw new Error('Stored proposal changed during promotion');
      if (latest.expiresAt <= new Date().toISOString()) throw new Error('Proposal expired');
      return this.advance(source, candidateHash, proposal.candidate_json, 'promote', source.revision, proposalId, proposal.evaluation_json);
    });
  }

  rollback(id: string, targetRevision: number, expectedRevision: number): WeightProgramRecord {
    this.assertWritable(); identifier(id); integer(targetRevision, 'targetRevision'); integer(expectedRevision, 'expectedRevision');
    return inTransaction(this.db, () => {
      const source = this.required(id);
      if (source.revision !== expectedRevision) throw new Error('Program revision changed; stale rollback');
      const checkpoint = this.statement<CheckpointRow, CheckpointRow | undefined>(`SELECT ${CHECKPOINT_COLUMNS} FROM wsp_checkpoints WHERE program_id = ? AND generation = ? AND revision = ?`, stmt => stmt.get(id, source.generation, targetRevision));
      if (!checkpoint) throw new Error('Target revision is not retained in this program generation');
      // All checkpoint writers persist canonical SDK JSON. Check its bytes
      // synchronously so corruption cannot activate an inconsistent model/hash.
      if (createHash('sha256').update(checkpoint.model_json).digest('hex') !== checkpoint.hash) throw new Error('Retained checkpoint hash mismatch');
      return this.advance(source, checkpoint.hash, checkpoint.model_json, 'rollback', targetRevision, null, null);
    });
  }

  dispose(): void { this.disposed = true; }

  private advance(source: ProgramRow, hash: string, json: string, operation: 'promote' | 'rollback', sourceRevision: number, proposalId: string | null, evaluationJson: string | null): WeightProgramRecord {
    const revision = source.revision + 1;
    integer(revision, 'next revision');
    const now = new Date().toISOString();
    const changes = this.statement('UPDATE wsp_programs SET revision = ?, model_hash = ?, model_json = ?, updated_at = ? WHERE id = ? AND generation = ? AND revision = ? AND model_hash = ?',
      stmt => stmt.run(revision, hash, json, now, source.id, source.generation, source.revision, source.hash).changes);
    if (changes !== 1) throw new Error('Program changed during update; retry with latest revision');
    this.insertCheckpoint(source.id, source.generation, revision, hash, json, operation, sourceRevision, proposalId, evaluationJson, now);
    this.statement('DELETE FROM wsp_checkpoints WHERE program_id = ? AND generation = ? AND revision NOT IN (SELECT revision FROM wsp_checkpoints WHERE program_id = ? AND generation = ? ORDER BY revision DESC LIMIT ?)',
      stmt => stmt.run(source.id, source.generation, source.id, source.generation, this.maxHistory));
    return this.record(this.required(source.id));
  }
  private insertCheckpoint(id: string, generation: string, revision: number, hash: string, json: string, operation: WeightProgramHistory['operation'], sourceRevision: number | null, proposalId: string | null, evaluation: string | null, now: string): void {
    this.statement('INSERT INTO wsp_checkpoints (program_id, generation, revision, model_hash, model_json, operation, source_revision, proposal_id, evaluation_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
      stmt => stmt.run(id, generation, revision, hash, json, operation, sourceRevision, proposalId, evaluation, now));
  }
  private record(row: ProgramRow): WeightProgramRecord {
    const history = this.statement<CheckpointRow, CheckpointRow[]>(`SELECT ${CHECKPOINT_COLUMNS} FROM wsp_checkpoints WHERE program_id = ? AND generation = ? ORDER BY revision DESC LIMIT ?`, stmt => stmt.all(row.id, row.generation, this.maxHistory));
    const proposalRows = this.statement<ProposalRow, ProposalRow[]>(`SELECT ${PROPOSAL_COLUMNS} FROM wsp_proposals WHERE program_id = ? AND generation = ? ORDER BY sequence DESC LIMIT ?`, stmt => stmt.all(row.id, row.generation, this.maxProposals));
    const promoted = new Set(history.filter(item => item.proposalId).map(item => item.proposalId));
    const now = new Date().toISOString();
    const { generation, model_json, ...metadata } = row;
    return { ...metadata, model: JSON.parse(model_json),
      history: history.map(({ model_json: checkpoint, evaluation_json, ...entry }) => ({ ...entry, evaluation: evaluation_json ? JSON.parse(evaluation_json) : null })),
      proposals: proposalRows.map(({ candidate_json, evaluation_json, accepted, ...proposal }) => ({ ...proposal, accepted: accepted === 1,
        status: promoted.has(proposal.id) ? 'promoted' : proposal.expiresAt <= now ? 'expired' : proposal.revision !== row.revision || proposal.baseHash !== row.hash ? 'stale' : accepted !== 1 ? 'rejected' : 'pending',
        evaluation: JSON.parse(evaluation_json),
      })),
    };
  }
  private assertCurrent(source: ProgramRow): void {
    const current = this.read(source.id);
    if (!current || current.generation !== source.generation || current.revision !== source.revision || current.hash !== source.hash) throw new Error('Program changed during evaluation; proposal is stale');
  }
  private read(id: string): ProgramRow | undefined {
    return this.statement<ProgramRow, ProgramRow | undefined>(`SELECT ${PROGRAM_COLUMNS}, generation, model_json FROM wsp_programs WHERE id = ?`, stmt => stmt.get(id));
  }
  private required(id: string): ProgramRow {
    const row = this.read(id);
    if (!row) throw new Error(`Weight program ${id} not found`);
    return row;
  }
  private statement<T = any, R = any>(sql: string, fn: (stmt: IPreparedStatement<T>) => R): R {
    const stmt = this.db.prepare<T>(sql);
    try { return fn(stmt); } finally { stmt.finalize?.(); }
  }
  private assertAlive(): void { if (this.disposed) throw new Error('WeightPrograms is disposed'); }
  private assertWritable(): void {
    this.assertAlive();
    if (this.readonly) throw new Error('WeightPrograms is read-only');
  }
}
