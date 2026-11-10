import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import Database from 'better-sqlite3';
import initSqlJs from 'sql.js';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadRuntime, type DenseModel } from 'ruv-fann';
import { fingerprintModel, type WeightPatch } from 'ruv-fann-wsp';
import { WeightPrograms } from '../../../src/controllers/WeightPrograms.js';
import { FannModels } from '../../../src/controllers/FannModels.js';
import { wrapExistingSqlJsDatabase } from '../../../src/db-fallback.js';
import { resetDatabaseTables } from '../../../src/mcp/reset-database.js';
import type { IDatabaseConnection } from '../../../src/types/database.types.js';

const dataset = () => ({ inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4 });
const policy = () => ({ maxMSE: 0, maxRegressionMSE: 0, maxAbsoluteError: 0 });
const blank = (): DenseModel => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [0, 0, 0, 0], bias: [0, 0], activation: 'Threshold', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [0, 0], bias: [0], activation: 'Threshold', steepness: 1 },
] });
const xor = (): DenseModel => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [1, 1, -1, -1], bias: [-0.5, 1.5], activation: 'Threshold', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [1, 1], bias: [-1.5], activation: 'Threshold', steepness: 1 },
] });
async function construction(model = blank()): Promise<WeightPatch> {
  const target = xor();
  return { schema: 'ruv-fann.weight-patch/1', baseHash: await fingerprintModel(model), edits: target.layers.flatMap((layer, index) => [
    ...layer.weights.map((value, location) => ({ layer: index, parameter: 'weights' as const, index: location, value })),
    ...layer.bias!.map((value, location) => ({ layer: index, parameter: 'bias' as const, index: location, value })),
  ]) };
}
let SQL: Awaited<ReturnType<typeof initSqlJs>>;
const cleanup: (() => void)[] = [];
beforeAll(async () => { SQL = await initSqlJs(); });
afterEach(() => { vi.useRealTimers(); for (const fn of cleanup.splice(0).reverse()) fn(); });

for (const driver of ['native', 'sqljs'] as const) describe(`WeightPrograms actual ${driver} SQL + WASM`, () => {
  function setup(options: ConstructorParameters<typeof WeightPrograms>[1] = {}) {
    const raw = driver === 'native' ? new Database(':memory:') : new SQL.Database();
    const db: IDatabaseConnection = driver === 'native' ? raw as any : wrapExistingSqlJsDatabase(raw);
    cleanup.push(() => (db as any).close());
    const programs = new WeightPrograms(db, options);
    cleanup.push(() => programs.dispose());
    return { db, programs, raw };
  }
  it('constructs XOR without gradients and persists exact promotion/rollback lineage', async () => {
    const { programs } = setup();
    const original = await programs.create('xor', blank());
    const proposal = await programs.propose('xor', await construction(), dataset(), policy());
    expect(proposal.evaluation.accepted).toBe(true);
    expect(proposal.evaluation.candidate.mse).toBe(0);
    expect(proposal.evaluation.baseline.mse).toBeGreaterThan(0);
    expect(programs.get('xor')!.model).toEqual(original.model);
    const promoted = await programs.promote('xor', proposal.proposalId, 1);
    expect(promoted.revision).toBe(2);
    expect(promoted.hash).toBe(proposal.evaluation.candidateHash);
    expect(promoted.model).toEqual(xor());
    expect(promoted.history[0]).toMatchObject({ operation: 'promote', proposalId: proposal.proposalId, sourceRevision: 1, evaluation: { accepted: true } });
    const prepared = (await loadRuntime()).prepare(promoted.model, { maxRows: 4 });
    try { expect(Array.from(prepared.predict(Float64Array.from(dataset().inputs), 4))).toEqual(dataset().targets); }
    finally { prepared.dispose(); }
    const rolled = programs.rollback('xor', 1, 2);
    expect(rolled.revision).toBe(3);
    expect(rolled.hash).toBe(original.hash);
    expect(rolled.model).toEqual(original.model);
    expect(rolled.history[0]).toMatchObject({ operation: 'rollback', sourceRevision: 1 });
    expect(programs.list()).toEqual([expect.objectContaining({ id: 'xor', revision: 3 })]);
    expect(programs.list()[0]).not.toHaveProperty('model');
  });

  it('repairs a single XOR weight and rejects regression or unmet thresholds', async () => {
    const { programs } = setup();
    const broken = xor();
    (broken.layers[1].weights as number[])[1] = 0;
    const initial = await programs.create('repair', broken);
    const repair: WeightPatch = { schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash, edits: [{ layer: 1, parameter: 'weights', index: 1, value: 1 }] };
    const good = await programs.propose('repair', repair, dataset(), policy());
    await programs.promote('repair', good.proposalId, 1);
    const regression: WeightPatch = { ...repair, baseHash: programs.get('repair')!.hash, edits: [{ layer: 1, parameter: 'weights', index: 1, value: 0 }] };
    const bad = await programs.propose('repair', regression, dataset(), { maxMSE: 1, maxRegressionMSE: 0 });
    expect(bad.evaluation.accepted).toBe(false);
    await expect(programs.promote('repair', bad.proposalId, 2)).rejects.toThrow('rejected');
    const threshold = await programs.propose('repair', regression, dataset(), { maxMSE: 0, maxRegressionMSE: 1 });
    expect(threshold.evaluation.accepted).toBe(false);
    expect(programs.get('repair')!.revision).toBe(2);
  });

  it('owns request snapshots and does not trust mutations of returned receipts or records', async () => {
    const { programs } = setup();
    const model = blank();
    const creating = programs.create('owned', model);
    (model.layers[0].weights as number[])[0] = 900;
    await creating;
    expect(programs.get('owned')!.model).toEqual(blank());
    const patch = await construction(); const data = dataset(); const rule = policy();
    const pending = programs.propose('owned', patch, data, rule);
    patch.edits[0].value = 900; data.targets[0] = 99; rule.maxMSE = 999;
    const proposal = await pending;
    expect(proposal.evaluation.accepted).toBe(true);
    (proposal.evaluation.model.layers[0].weights as number[])[0] = 999;
    proposal.evaluation.accepted = false;
    const record = programs.get('owned')!;
    record.proposals[0].evaluation.accepted = false;
    (record.model.layers[0].weights as number[])[0] = 123;
    const promoted = await programs.promote('owned', proposal.proposalId, 1);
    expect(promoted.model).toEqual(xor());
  });

  it('rejects forged receipts, cross-program IDs and forged acceptance', async () => {
    const { programs } = setup();
    await programs.create('one', xor()); await programs.create('two', blank());
    const bad = await programs.propose('one', { schema: 'ruv-fann.weight-patch/1', baseHash: await fingerprintModel(xor()), edits: [{ layer: 1, parameter: 'weights', index: 1, value: 0 }] }, dataset(), policy());
    bad.evaluation.accepted = true;
    await expect(programs.promote('one', bad as any, 1)).rejects.toThrow('proposalId');
    await expect(programs.promote('one', bad.proposalId, 1)).rejects.toThrow('rejected');
    await expect(programs.promote('two', bad.proposalId, 1)).rejects.toThrow('not found');
    expect(programs.get('one')!.revision).toBe(1);
  });

  it('rejects malformed patches, non-finite and overflowing evaluation without persistence', async () => {
    const { programs } = setup();
    await programs.create('safe', blank());
    const patch = await construction();
    for (const invalid of [
      { ...patch, baseHash: 'a'.repeat(64) }, { ...patch, edits: [] },
      { ...patch, edits: [patch.edits[0], patch.edits[0]] },
      { ...patch, edits: [{ ...patch.edits[0], value: Infinity }] },
      { ...patch, edits: [{ ...patch.edits[0], layer: 64 }] },
      { ...patch, edits: [{ ...patch.edits[0], index: 99 }] },
      { ...patch, edits: [{ ...patch.edits[0], value: 1001 }] },
      { ...patch, evaluation: { accepted: true } },
    ]) await expect(programs.propose('safe', invalid as any, dataset(), policy())).rejects.toThrow();
    await expect(programs.propose('safe', patch, { ...dataset(), targets: [1e308, 1e308, 1e308, 1e308] }, policy())).rejects.toThrow();
    await expect(programs.propose('safe', patch, { ...dataset(), inputs: [NaN] }, policy())).rejects.toThrow();
    await expect(programs.propose('safe', patch, dataset(), { ...policy(), maxMSE: Infinity })).rejects.toThrow();
    expect(programs.get('safe')!.proposals).toEqual([]);
    expect(programs.get('safe')!.revision).toBe(1);
  });

  it('rejects simultaneous evaluations after a concurrent revision change', async () => {
    const { programs, db } = setup();
    const other = new WeightPrograms(db); cleanup.push(() => other.dispose());
    await programs.create('race', blank());
    const patch = await construction();
    const results = Promise.allSettled([programs.propose('race', patch, dataset(), policy()), other.propose('race', patch, dataset(), policy())]);
    programs.rollback('race', 1, 1);
    expect((await results).every(result => result.status === 'rejected')).toBe(true);
    expect(programs.get('race')!.proposals).toHaveLength(0);
  });

  it('allows one competing promotion and rejects stale proposal/revision reuse', async () => {
    const { programs, db } = setup();
    const other = new WeightPrograms(db); cleanup.push(() => other.dispose());
    await programs.create('race', blank());
    const one = await programs.propose('race', await construction(), dataset(), policy());
    const two = await other.propose('race', await construction(), dataset(), policy());
    const results = await Promise.allSettled([programs.promote('race', one.proposalId, 1), other.promote('race', two.proposalId, 1)]);
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
    expect(programs.get('race')!.revision).toBe(2);
    await expect(other.promote('race', two.proposalId, 2)).rejects.toThrow('stale');
    expect(() => other.rollback('race', 1, 1)).toThrow('stale');
  });

  it('caps in-flight proposals and releases slots after validation failures', async () => {
    const { programs } = setup({ maxProposals: 2 });
    await programs.create('concurrency', blank());
    const patch = await construction();
    const first = programs.propose('concurrency', patch, dataset(), policy());
    const second = programs.propose('concurrency', patch, dataset(), policy());
    await expect(programs.propose('concurrency', patch, dataset(), policy())).rejects.toThrow('in-flight');
    await Promise.all([first, second]);
    await expect(programs.propose('concurrency', { ...patch, edits: [] }, dataset(), policy())).rejects.toThrow();
    expect((await programs.propose('concurrency', patch, dataset(), policy())).evaluation.accepted).toBe(true);
  });

  it('does not invoke array iterators, element getters or model accessors', async () => {
    const { programs } = setup();
    const iterator = vi.fn(() => { throw new Error('iterator ran'); });
    const custom = blank();
    Object.defineProperty(custom.layers[0].weights, Symbol.iterator, { value: iterator });
    await expect(programs.create('iterator', custom)).rejects.toThrow();
    expect(iterator).not.toHaveBeenCalled();
    const getter = vi.fn(() => 1);
    const accessor = blank(); Object.defineProperty(accessor.layers[0].weights, '0', { get: getter });
    await expect(programs.create('accessor', accessor)).rejects.toThrow('accessors');
    expect(getter).not.toHaveBeenCalled();
    const object = blank(); Object.defineProperty(object, 'layers', { get: getter });
    await expect(programs.create('object', object)).rejects.toThrow('accessors');
    expect(getter).not.toHaveBeenCalled();
    const symbol = blank(); Object.defineProperty(symbol, Symbol('hidden'), { value: 1 });
    await expect(programs.create('symbol', symbol)).rejects.toThrow('symbols');
    const hidden = blank(); Object.defineProperty(hidden, 'unknown', { value: true });
    await expect(programs.create('hidden', hidden)).rejects.toThrow();
    const huge = blank(); (huge.layers[0] as any).activation = 'x'.repeat(1024 * 1024 + 1);
    await expect(programs.create('huge', huge)).rejects.toThrow('string limit');
    expect(programs.list()).toEqual([]);
  });

  it('protects against deletion/recreation ABA while preserving monotonic fresh generations', async () => {
    const { programs, db } = setup();
    await programs.create('aba', blank());
    const proposal = await programs.propose('aba', await construction(), dataset(), policy());
    db.prepare('DELETE FROM wsp_programs WHERE id = ?').run('aba');
    await programs.create('aba', blank());
    await expect(programs.promote('aba', proposal.proposalId, 1)).rejects.toThrow('not found');
    expect(programs.get('aba')!.proposals).toEqual([]);
    const patch = await construction();
    const pending = programs.propose('aba', patch, dataset(), policy()).then(() => 'accepted', error => String(error));
    db.prepare('DELETE FROM wsp_programs WHERE id = ?').run('aba');
    await programs.create('aba', blank());
    expect(await pending).toContain('stale');
    expect(programs.get('aba')!.revision).toBe(1);
  });

  it('bounds proposal/checkpoint retention and preserves promoted evidence in lineage', async () => {
    const { programs } = setup({ maxProposals: 2, maxHistory: 3 });
    await programs.create('bounded', blank());
    const patch = await construction();
    const first = await programs.propose('bounded', patch, dataset(), policy());
    await programs.propose('bounded', patch, dataset(), policy());
    const latest = await programs.propose('bounded', patch, dataset(), policy());
    expect(programs.get('bounded')!.proposals).toHaveLength(2);
    await expect(programs.promote('bounded', first.proposalId, 1)).rejects.toThrow('not found');
    await programs.promote('bounded', latest.proposalId, 1);
    programs.rollback('bounded', 1, 2);
    programs.rollback('bounded', 2, 3);
    const record = programs.get('bounded')!;
    expect(record.history).toHaveLength(3);
    expect(record.history.find(row => row.revision === 2)!.evaluation!.candidateHash).toBe(latest.evaluation.candidateHash);
    expect(() => programs.rollback('bounded', 1, 4)).toThrow('not retained');
  });

  it('expires proposals without authorizing their later promotion', async () => {
    const { programs } = setup({ maxProposalAgeMs: 100 });
    await programs.create('expiry', blank());
    const result = await programs.propose('expiry', await construction(), dataset(), policy());
    vi.useFakeTimers({ toFake: ['Date'] }); vi.setSystemTime(Date.now() + 1000);
    await expect(programs.promote('expiry', result.proposalId, 1)).rejects.toThrow('expired');
    expect(programs.get('expiry')!.proposals[0].status).toBe('expired');
  });

  it('rejects candidate corruption and SQL failure atomically', async () => {
    const { programs, db } = setup();
    await programs.create('integrity', blank());
    const result = await programs.propose('integrity', await construction(), dataset(), policy());
    db.prepare('UPDATE wsp_proposals SET candidate_json = ? WHERE id = ?').run(JSON.stringify(blank()), result.proposalId);
    await expect(programs.promote('integrity', result.proposalId, 1)).rejects.toThrow('hash mismatch');
    expect(programs.get('integrity')!.revision).toBe(1);
    const valid = await programs.propose('integrity', await construction(), dataset(), policy());
    db.exec("CREATE TRIGGER fail_wsp_checkpoint BEFORE INSERT ON wsp_checkpoints BEGIN SELECT RAISE(ABORT, 'checkpoint failure'); END");
    await expect(programs.promote('integrity', valid.proposalId, 1)).rejects.toThrow('checkpoint failure');
    expect(programs.get('integrity')!.revision).toBe(1);
    expect(programs.get('integrity')!.history).toHaveLength(1);
  });

  it('rejects corrupted retained checkpoint bytes without changing the active program', async () => {
    const { programs, db } = setup();
    await programs.create('rollback-integrity', blank());
    const proposal = await programs.propose('rollback-integrity', await construction(), dataset(), policy());
    const active = await programs.promote('rollback-integrity', proposal.proposalId, 1);
    db.prepare('UPDATE wsp_checkpoints SET model_json = ? WHERE program_id = ? AND revision = 1').run(JSON.stringify(xor()), 'rollback-integrity');
    expect(() => programs.rollback('rollback-integrity', 1, 2)).toThrow('checkpoint hash mismatch');
    const after = programs.get('rollback-integrity')!;
    expect(after.revision).toBe(active.revision);
    expect(after.hash).toBe(active.hash);
    expect(after.model).toEqual(active.model);
    expect(after.history).toHaveLength(2);
  });

  it('rejects revision overflow and keeps FannModels independent', async () => {
    const { programs, db } = setup();
    const fann = new FannModels(db); cleanup.push(() => fann.dispose());
    await fann.create('neural', blank());
    await programs.create('overflow', blank());
    db.prepare('UPDATE wsp_programs SET revision = ? WHERE id = ?').run(Number.MAX_SAFE_INTEGER, 'overflow');
    expect(() => programs.rollback('overflow', 1, Number.MAX_SAFE_INTEGER)).toThrow('next revision');
    expect(fann.get('neural')!.revision).toBe(1);
    expect(fann.get('neural')!.model).toEqual(blank());
  });

  it('enforces readonly with no schema writes, disposal and reset recreation', async () => {
    const { programs, db } = setup();
    await programs.create('readonly', blank());
    const exec = vi.spyOn(db, 'exec');
    const reader = new WeightPrograms(db, { readonly: true }); cleanup.push(() => reader.dispose());
    expect(exec).not.toHaveBeenCalled();
    expect(reader.get('readonly')!.revision).toBe(1);
    await expect(reader.create('denied', blank())).rejects.toThrow('read-only');
    await expect(reader.propose('readonly', await construction(), dataset(), policy())).rejects.toThrow('read-only');
    await expect(reader.promote('readonly', 'forged', 1)).rejects.toThrow('read-only');
    expect(() => reader.rollback('readonly', 1, 1)).toThrow('read-only');
    exec.mockRestore();
    resetDatabaseTables(db); programs.dispose(); reader.dispose();
    const reopened = new WeightPrograms(db); cleanup.push(() => reopened.dispose());
    expect(reopened.list()).toEqual([]); await reopened.create('after-reset', blank());
    programs.dispose(); expect(() => programs.list()).toThrow('disposed');
    const pending = reopened.create('cancelled', blank()); reopened.dispose();
    await expect(pending).rejects.toThrow('disposed');
  });

  it('persists outstanding proposals and rollback checkpoints through a real file reopen', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'agentdb-wsp-persist-')); cleanup.push(() => rmSync(directory, { recursive: true, force: true }));
    const file = join(directory, 'programs.db');
    const raw: any = driver === 'native' ? new Database(file) : new SQL.Database();
    const db: any = driver === 'native' ? raw : wrapExistingSqlJsDatabase(raw);
    const first = new WeightPrograms(db);
    await first.create('durable', blank());
    const proposal = await first.propose('durable', await construction(), dataset(), policy());
    if (driver === 'sqljs') writeFileSync(file, raw.export());
    first.dispose(); db.close();
    const restored: any = driver === 'native' ? new Database(file) : wrapExistingSqlJsDatabase(new SQL.Database(readFileSync(file)));
    cleanup.push(() => restored.close());
    const reopened = new WeightPrograms(restored); cleanup.push(() => reopened.dispose());
    expect(reopened.get('durable')!.proposals[0].evaluation.candidateHash).toBe(proposal.evaluation.candidateHash);
    expect((await reopened.promote('durable', proposal.proposalId, 1)).model).toEqual(xor());
    expect(reopened.rollback('durable', 1, 2).model).toEqual(blank());
    if (driver === 'native') {
      const readonly = new Database(file, { readonly: true }); cleanup.push(() => readonly.close());
      const reader = new WeightPrograms(readonly, { readonly: true }); cleanup.push(() => reader.dispose());
      expect(reader.get('durable')!.revision).toBe(3);
    }
  });
});
