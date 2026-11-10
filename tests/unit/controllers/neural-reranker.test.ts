import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import Database from 'better-sqlite3';
import initSqlJs from 'sql.js';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fingerprintModel, type WeightPatch } from 'ruv-fann-wsp';
import { NeuralReranker, RERANK_FEATURE_SCHEMA, episodeRerankFeatures, type RerankCandidate, type RerankGroup, type RankingEvaluation, type WeightInspection } from '../../../src/controllers/NeuralReranker.js';
import { wrapExistingSqlJsDatabase } from '../../../src/db-fallback.js';
import { resetDatabaseTables } from '../../../src/mcp/reset-database.js';
import type { IDatabaseConnection } from '../../../src/types/database.types.js';

// Observe actual SDK training calls while delegating every numerical operation
// to the real runtime. The patch path must never invoke the gradient trainer.
const trainingCalls = vi.hoisted(() => ({ count: 0 }));
vi.mock('ruv-fann', async importOriginal => {
  const actual = await importOriginal<typeof import('ruv-fann')>();
  return { ...actual, async loadRuntime(options?: { simd?: boolean | 'auto' }) {
    const runtime = await actual.loadRuntime(options);
    return { backend: runtime.backend, prepare: runtime.prepare.bind(runtime), train(model: any, options: any) {
      trainingCalls.count++; return runtime.train(model, options);
    } };
  } };
});
function rewardPatch(inspection: WeightInspection): WeightPatch {
  return { schema: 'ruv-fann.weight-patch/1', baseHash: inspection.baseHash,
    edits: inspection.model.layers.flatMap((layer, index) => [
      ...layer.weights.map((_, location) => ({ layer: index, parameter: 'weights' as const, index: location,
        value: index === 0 ? location === 1 ? 1 : 0 : location === 0 ? 1 : 0 })),
      ...layer.bias!.map((_, location) => ({ layer: index, parameter: 'bias' as const, index: location, value: 0 })),
    ]) };
}

// Controlled deterministic synthetic judgments; these are not production-effectiveness measurements.
function groups(prefix: string, count = 8): RerankGroup[] {
  return Array.from({ length: count }, (_, query) => ({ queryId: `${prefix}-${query}`, candidates: [0, 1, 2, 3].map(relevance => ({
    id: `grade-${relevance}`, relevance, features: { similarity: 0.8, reward: relevance / 3, success: relevance >= 2 ? 1 : 0, recency: 0.5 },
  })) }));
}
const unlabeled = (group: RerankGroup): RerankCandidate[] => group.candidates.map(({ id, features }) => ({ id, features }));
let SQL: Awaited<ReturnType<typeof initSqlJs>>;
const cleanup: (() => void)[] = [];
beforeAll(async () => { SQL = await initSqlJs(); });
afterEach(() => { for (const fn of cleanup.splice(0).reverse()) fn(); });

describe('episode feature adapter', () => {
  it('uses fixed feature order, bounded values and seconds-based 30-day recency', () => {
    const now = 10_000_000;
    expect(RERANK_FEATURE_SCHEMA).toBe('agentdb.rerank-features/1');
    expect(episodeRerankFeatures({ similarity: -1, reward: 4, success: true, ts: now - 2_592_000 }, now)).toEqual({ similarity: 0, reward: 1, success: 1, recency: 0.5 });
    expect(episodeRerankFeatures({ similarity: 0, reward: -1, success: false, ts: now + 10 }, now)).toEqual({ similarity: 0.5, reward: 0, success: 0, recency: 1 });
    expect(episodeRerankFeatures({ similarity: Infinity, reward: NaN, ts: NaN }, now)).toEqual({ similarity: 0, reward: 0, success: 0, recency: 0 });
    expect(episodeRerankFeatures({}, now).recency).toBe(0);
    expect(() => episodeRerankFeatures({}, Infinity)).toThrow('finite');
  });
});

for (const driver of ['native', 'sqljs'] as const) describe(`NeuralReranker actual ${driver} SQL + WASM`, () => {
  function setup() {
    const raw: any = driver === 'native' ? new Database(':memory:') : new SQL.Database();
    const db: IDatabaseConnection = driver === 'native' ? raw : wrapExistingSqlJsDatabase(raw);
    cleanup.push(() => (db as any).close());
    const controller = new NeuralReranker(db, { simd: false }); cleanup.push(() => controller.dispose());
    return { db, controller, raw };
  }
  async function accepted(controller: NeuralReranker, id = 'memory') {
    controller.create(id);
    const candidate = await controller.train(id, groups('train'));
    const evaluation = await controller.evaluate(id, candidate.candidateId, groups('heldout', 5));
    expect(evaluation.accepted, evaluation.reasons.join(', ')).toBe(true);
    return { candidate, evaluation };
  }

  it('authors a 49-edit seed candidate and promotes held-out improvement without any gradient training', async () => {
    const { controller } = setup(); controller.create('weights');
    const inspection = await controller.inspectWeights('weights');
    expect(inspection.source).toBe('seed'); expect(inspection.baseHash).toBe(await fingerprintModel(inspection.model));
    expect(controller.get('weights')!.model).toBeNull();
    const patch = rewardPatch(inspection); expect(patch.edits).toHaveLength(49);
    const before = trainingCalls.count;
    const candidate = await controller.proposePatch('weights', patch);
    expect(trainingCalls.count).toBe(before);
    expect(candidate).toMatchObject({ origin: 'weight-patch', parameterCount: 49, backend: 'wasm', trainingQueryIds: [], baseHash: inspection.baseHash });
    for (const field of ['epochs', 'initialLoss', 'finalLoss', 'datasetHash']) expect(candidate).not.toHaveProperty(field);
    const candidateWeights = await controller.inspectWeights('weights', candidate.candidateId);
    expect(candidateWeights.source).toBe('candidate'); expect(candidateWeights.baseHash).toBe(candidate.modelHash);
    expect(candidateWeights.model.layers[0].weights[1]).toBe(1);
    const evaluation = await controller.evaluate('weights', candidate.candidateId, groups('heldout', 5));
    expect(evaluation.accepted).toBe(true); expect(evaluation.candidate.ndcg).toBe(1);
    await controller.promote('weights', evaluation.evaluationId, 1);
    const active = await controller.inspectWeights('weights'); expect(active.source).toBe('active');
    expect(active.baseHash).toBe(candidate.modelHash);
    expect((await controller.rank('weights', unlabeled(groups('rank', 1)[0]), { mode: 'active' })).order).toEqual(['grade-3', 'grade-2', 'grade-1', 'grade-0']);
    expect(trainingCalls.count).toBe(before);
    controller.rollback('weights', 1, 2);
    expect((await controller.inspectWeights('weights')).source).toBe('seed');
    expect((await controller.rank('weights', [], { mode: 'active' })).available).toBe(false);
  });

  it('owns direct patch metadata and rejects stale hashes, invalid edits and concurrent inspection/patch staleness', async () => {
    const { controller } = setup(); controller.create('patches');
    const seed = await controller.inspectWeights('patches'); const patch = rewardPatch(seed);
    const pending = controller.proposePatch('patches', patch); patch.edits[1].value = 900;
    const candidate = await pending;
    expect(candidate.patch!.edits[1].value).toBe(1);
    candidate.patch!.edits[1].value = 800;
    expect(controller.get('patches')!.candidates[0].patch!.edits[1].value).toBe(1);
    const inspected = await controller.inspectWeights('patches', candidate.candidateId);
    (inspected.model.layers[0].weights as number[])[1] = 700;
    expect((await controller.inspectWeights('patches', candidate.candidateId)).model.layers[0].weights[1]).toBe(1);
    for (const invalid of [
      { ...rewardPatch(seed), baseHash: 'a'.repeat(64) },
      { ...rewardPatch(seed), edits: [] },
      { ...rewardPatch(seed), edits: [rewardPatch(seed).edits[0], rewardPatch(seed).edits[0]] },
      { ...rewardPatch(seed), edits: [{ layer: 0, parameter: 'weights', index: 0, value: 1001 }] },
      { ...rewardPatch(seed), edits: [{ layer: 0, parameter: 'weights', index: 99, value: 1 }] },
      { ...rewardPatch(seed), topology: [4, 9, 1] },
    ]) await expect(controller.proposePatch('patches', invalid as any)).rejects.toThrow();
    const getter = vi.fn(() => 1); const accessors = rewardPatch(seed);
    Object.defineProperty(accessors.edits[0], 'value', { get: getter });
    await expect(controller.proposePatch('patches', accessors)).rejects.toThrow('accessors'); expect(getter).not.toHaveBeenCalled();
    const inspecting = controller.inspectWeights('patches');
    const proposing = controller.proposePatch('patches', rewardPatch(seed));
    controller.rollback('patches', 1, 1);
    await expect(inspecting).rejects.toThrow('stale'); await expect(proposing).rejects.toThrow('stale');
  });

  it('carries ancestor training IDs through candidate patches, active promotion, eviction and rollback', async () => {
    const { controller } = setup(); controller.create('lineage');
    const trained = await controller.train('lineage', groups('ancestor'));
    const weights = await controller.inspectWeights('lineage', trained.candidateId);
    const patch: WeightPatch = { schema: 'ruv-fann.weight-patch/1', baseHash: weights.baseHash,
      edits: [{ layer: 1, parameter: 'bias', index: 0, value: weights.model.layers[1].bias![0] + 0.01 }] };
    const derived = await controller.proposePatch('lineage', patch, trained.candidateId);
    expect(derived.trainingQueryIds).toEqual(trained.trainingQueryIds);
    await expect(controller.evaluate('lineage', derived.candidateId, groups('ancestor', 5))).rejects.toThrow('overlap');
    const evaluation = await controller.evaluate('lineage', derived.candidateId, groups('heldout', 5));
    await controller.promote('lineage', evaluation.evaluationId, 1);
    expect(controller.get('lineage')!.trainingQueryIds).toEqual(trained.trainingQueryIds);
    expect(controller.get('lineage')!.history[0].trainingQueryIds).toEqual(trained.trainingQueryIds);
    for (let index = 0; index < 16; index++) await controller.train('lineage', groups(`eviction-${index}`), { epochs: 1 });
    expect(controller.get('lineage')!.candidates.some(candidate => candidate.candidateId === trained.candidateId || candidate.candidateId === derived.candidateId)).toBe(false);
    const active = await controller.inspectWeights('lineage');
    const next = await controller.proposePatch('lineage', { schema: patch.schema, baseHash: active.baseHash,
      edits: [{ layer: 1, parameter: 'bias', index: 0, value: active.model.layers[1].bias![0] + 0.01 }] });
    expect(next.trainingQueryIds).toEqual(trained.trainingQueryIds);
    await expect(controller.evaluate('lineage', next.candidateId, groups('ancestor', 5))).rejects.toThrow('overlap');
    controller.rollback('lineage', 1, 2); expect(controller.get('lineage')!.trainingQueryIds).toEqual([]);
    controller.rollback('lineage', 2, 3); expect(controller.get('lineage')!.trainingQueryIds).toEqual(trained.trainingQueryIds);
  });

  it('requires explicit training/evaluation/promotion and preserves exact shadow baseline order', async () => {
    const { controller } = setup();
    const initial = controller.create('memory');
    expect(initial).toMatchObject({ featureSchema: RERANK_FEATURE_SCHEMA, revision: 1, modelHash: null, model: null });
    const inputs = unlabeled(groups('rank', 1)[0]);
    expect(await controller.rank('memory', inputs, { mode: 'active' })).toMatchObject({ order: inputs.map(item => item.id), available: false, scores: [] });
    const trained = await controller.train('memory', groups('train'));
    expect(trained.backend).toBe('wasm'); expect(trained.parameterCount).toBe(49);
    expect(trained.finalLoss).toBeLessThan(trained.initialLoss * 0.01);
    expect(controller.get('memory')!.modelHash).toBeNull();
    const shadow = await controller.rank('memory', inputs, { candidateId: trained.candidateId });
    expect(shadow.order).toEqual(inputs.map(item => item.id));
    expect(shadow.proposedOrder).toEqual(['grade-3', 'grade-2', 'grade-1', 'grade-0']);
    const evaluation = await controller.evaluate('memory', trained.candidateId, groups('heldout', 5));
    expect(evaluation.accepted).toBe(true); expect(evaluation.candidate).toEqual({ ndcg: 1, mrr: 1 });
    const promoted = await controller.promote('memory', evaluation.evaluationId, 1);
    expect(promoted.revision).toBe(2); expect(promoted.history[0].evaluation!.evaluationId).toBe(evaluation.evaluationId);
    const active = await controller.rank('memory', inputs, { mode: 'active' });
    expect(active.order).toEqual(shadow.proposedOrder); expect(active.modelHash).toBe(trained.modelHash);
    expect(active.revision).toBe(2); expect(active.scores.every(row => Number.isFinite(row.score))).toBe(true);
    expect(controller.list()[0]).not.toHaveProperty('model');
    expect(controller.rollback('memory', 1, 2)).toMatchObject({ revision: 3, modelHash: null, model: null });
    expect((await controller.rank('memory', inputs, { mode: 'active' })).order).toEqual(inputs.map(item => item.id));
  });

  it('matches an independent numerical oracle for graded NDCG@3 and binary MRR@3', async () => {
    const { controller } = setup(); controller.create('oracle');
    const candidate = await controller.train('oracle', groups('train'));
    const heldout = groups('oracle', 5).map(group => ({ ...group, candidates: [group.candidates[0], group.candidates[3], group.candidates[1], group.candidates[2]] }));
    const evaluation = await controller.evaluate('oracle', candidate.candidateId, heldout, { k: 3 });
    // For grades [0,3,1,2], DCG@3=4.916508275; ideal DCG@3=9.392789261.
    // Precomputed oracle values avoid reimplementing the controller's metric formula in the test.
    expect(evaluation.baseline.ndcg).toBeCloseTo(0.5234343216411389, 12);
    expect(evaluation.baseline.mrr).toBe(0.5);
    expect(evaluation.candidate.ndcg).toBe(1); expect(evaluation.candidate.mrr).toBe(1);
    expect(evaluation.perQuery).toHaveLength(5);
  });

  it('compares both the input baseline and current active model before accepting', async () => {
    const { controller } = setup(); const first = await accepted(controller);
    await controller.promote('memory', first.evaluation.evaluationId, 1);
    const identical = await controller.train('memory', groups('other-training'));
    const evaluation = await controller.evaluate('memory', identical.candidateId, groups('new-heldout', 5));
    expect(evaluation.candidate.ndcg).toBeGreaterThan(evaluation.baseline.ndcg);
    expect(evaluation.active).toEqual({ ndcg: 1, mrr: 1 });
    expect(evaluation.accepted).toBe(false); expect(evaluation.reasons.join(' ')).toContain('versus active');
    await expect(controller.promote('memory', evaluation.evaluationId, 2)).rejects.toThrow('rejected');
  });

  it('gates MRR and per-query regressions independently of mean NDCG improvement', async () => {
    const { controller } = setup(); controller.create('gates');
    const trained = await controller.train('gates', groups('train'));
    const heldout = groups('gates', 5).map(group => ({ ...group, candidates: [group.candidates[1], group.candidates[0], group.candidates[2], group.candidates[3]] }));
    heldout[0].candidates = groups('bad', 1)[0].candidates.map(candidate => ({ ...candidate, relevance: candidate.relevance === 0 ? 3 : 0 }));
    const mrr = await controller.evaluate('gates', trained.candidateId, heldout, { maxPerQueryNdcgRegression: 1 });
    expect(mrr.candidate.ndcg).toBeGreaterThan(mrr.baseline.ndcg + 0.01);
    expect(mrr.reasons.join(' ')).toContain('MRR regression'); expect(mrr.accepted).toBe(false);
    const perQuery = await controller.evaluate('gates', trained.candidateId, heldout, { maxMrrRegression: 1, maxPerQueryNdcgRegression: 0 });
    expect(perQuery.reasons.join(' ')).toContain('Per-query'); expect(perQuery.accepted).toBe(false);
  });

  it('rejects train/evaluation overlap, undersized evaluation and all-zero judgments', async () => {
    const { controller } = setup(); controller.create('holdout');
    const trained = await controller.train('holdout', groups('train'));
    await expect(controller.evaluate('holdout', trained.candidateId, groups('train', 5))).rejects.toThrow('overlap');
    await expect(controller.evaluate('holdout', trained.candidateId, groups('small', 4))).rejects.toThrow('groups length');
    const zero = groups('zero', 5); zero[2].candidates.forEach(candidate => { candidate.relevance = 0; });
    await expect(controller.evaluate('holdout', trained.candidateId, zero)).rejects.toThrow('positive judgment');
    expect(controller.get('holdout')!.evaluations).toEqual([]);
  });

  it('owns all request/result snapshots and never accepts caller-forged evaluation receipts', async () => {
    const { controller } = setup(); controller.create('owned');
    const input = groups('train'); const options = { epochs: 250, learningRate: 0.01 };
    const pending = controller.train('owned', input, options);
    input[0].candidates[0].features.reward = 99; input[0].queryId = 'corrupted'; options.epochs = 1;
    const trained = await pending; expect(trained.epochs).toBe(250); expect(trained.trainingQueryIds).toContain('train-0');
    trained.trainingQueryIds[0] = 'caller';
    const heldout = groups('heldout', 5); const policy = { minNdcgImprovement: 0.01 };
    const checking = controller.evaluate('owned', trained.candidateId, heldout, policy);
    heldout[0].candidates[0].relevance = 3; policy.minNdcgImprovement = 1;
    const evaluated = await checking; expect(evaluated.accepted).toBe(true);
    evaluated.accepted = false; evaluated.candidateHash = 'fake';
    const inspected = controller.get('owned')!; inspected.evaluations[0].accepted = false; inspected.candidates[0].trainingQueryIds[0] = 'tampered';
    await expect(controller.promote('owned', evaluated as any, 1)).rejects.toThrow('evaluationId');
    await expect(controller.promote('owned', 'invented', 1)).rejects.toThrow('not found');
    const promoted = await controller.promote('owned', evaluated.evaluationId, 1);
    expect(promoted.revision).toBe(2);
    (promoted.model!.layers[0].weights as number[])[0] = 999;
    expect(controller.get('owned')!.model!.layers[0].weights[0]).not.toBe(999);
  });

  it('cannot promote a rejected evaluation even after caller changes its acceptance', async () => {
    const { controller } = setup(); controller.create('rejected');
    const candidate = await controller.train('rejected', groups('train'));
    const evaluation = await controller.evaluate('rejected', candidate.candidateId, groups('heldout', 5), { minNdcgImprovement: 1 });
    expect(evaluation.accepted).toBe(false); evaluation.accepted = true;
    await expect(controller.promote('rejected', evaluation.evaluationId, 1)).rejects.toThrow('rejected');
    expect(controller.get('rejected')!.revision).toBe(1);
  });

  it('preserves stable ties, supports empty batches and rejects active unpromoted candidates', async () => {
    const { controller } = setup(); const { candidate, evaluation } = await accepted(controller);
    expect(await controller.rank('memory', [], { candidateId: candidate.candidateId })).toMatchObject({ order: [], proposedOrder: [], scores: [], available: true });
    await expect(controller.rank('memory', [], { mode: 'active', candidateId: candidate.candidateId })).rejects.toThrow('only permitted');
    await controller.promote('memory', evaluation.evaluationId, 1);
    const features = groups('ties', 1)[0].candidates[0].features;
    const ties = [{ id: 'z', features }, { id: 'a', features }];
    expect((await controller.rank('memory', ties, { mode: 'active' })).order).toEqual(['z', 'a']);
    await expect(controller.rank('memory', [], { candidateId: candidate.candidateId })).rejects.toThrow('stale');
  });

  it('rejects malformed data, labels-as-features, oversized work and null options before persistence', async () => {
    const { controller } = setup(); controller.create('bounded');
    const original = groups('train');
    for (const input of [[], [...original, original[0]], groups('x', 129), [{ ...original[0], candidates: [original[0].candidates[0], original[0].candidates[0]] }], [{ ...original[0], extra: true }], [{ ...original[0], candidates: [{ ...original[0].candidates[0], features: { ...original[0].candidates[0].features, relevance: 3 } }, original[0].candidates[1]] }]]) {
      await expect(controller.train('bounded', input as any)).rejects.toThrow();
    }
    const over = groups('over', 128); over[0].candidates.push({ ...over[0].candidates[0], id: 'extra' });
    await expect(controller.train('bounded', over)).rejects.toThrow('512');
    const invalid = groups('bad'); invalid[0].candidates[0].features.similarity = NaN;
    await expect(controller.train('bounded', invalid)).rejects.toThrow('finite');
    for (const options of [{ epochs: 1001 }, { epochs: null }, { learningRate: Infinity }, { learningRate: 0 }, { extra: true }, null]) await expect(controller.train('bounded', original, options as any)).rejects.toThrow();
    for (const options of [{ mode: null }, { mode: 'invalid' }, null, { extra: true }]) await expect(controller.rank('bounded', [], options as any)).rejects.toThrow();
    expect(controller.get('bounded')!.candidates).toEqual([]);
    const trained = await controller.train('bounded', original, { epochs: 1 });
    for (const policy of [null, { k: null }, { minNdcgImprovement: 0 }, { minNdcgImprovement: 2 }, { maxMrrRegression: NaN }, { extra: true }]) await expect(controller.evaluate('bounded', trained.candidateId, groups('eval', 5), policy as any)).rejects.toThrow();
  });

  it('rejects holes, getters, symbols and custom iterators without invoking them', async () => {
    const { controller } = setup(); controller.create('data');
    const getter = vi.fn(() => 0.5); const input = groups('getter');
    Object.defineProperty(input[0].candidates[0].features, 'reward', { get: getter });
    await expect(controller.train('data', input)).rejects.toThrow('accessors'); expect(getter).not.toHaveBeenCalled();
    const iterator = vi.fn(() => { throw new Error('iterator invoked'); }); const custom = groups('iterator');
    Object.defineProperty(custom, Symbol.iterator, { value: iterator });
    await expect(controller.train('data', custom)).rejects.toThrow('dense'); expect(iterator).not.toHaveBeenCalled();
    const hole = groups('holes'); delete (hole[0].candidates as any)[0];
    await expect(controller.train('data', hole)).rejects.toThrow();
    const indexed = groups('indexed'); Object.defineProperty(indexed, '0', { get: getter });
    await expect(controller.train('data', indexed)).rejects.toThrow('accessors'); expect(getter).not.toHaveBeenCalled();
    const symbol = groups('symbol'); Object.defineProperty(symbol[0], Symbol('hidden'), { value: true });
    await expect(controller.train('data', symbol)).rejects.toThrow('Unknown');
  });

  it('rejects stale training/evaluation and concurrent competing promotion', async () => {
    const { controller, db } = setup(); controller.create('race');
    const pending = controller.train('race', groups('train')); controller.rollback('race', 1, 1);
    await expect(pending).rejects.toThrow('stale'); expect(controller.get('race')!.candidates).toEqual([]);
    const candidate = await controller.train('race', groups('fresh'));
    const checking = controller.evaluate('race', candidate.candidateId, groups('heldout', 5)); controller.rollback('race', 1, 2);
    await expect(checking).rejects.toThrow('stale'); expect(controller.get('race')!.evaluations).toEqual([]);
    const fresh = await controller.train('race', groups('fresh2'));
    const accepted = await controller.evaluate('race', fresh.candidateId, groups('heldout2', 5));
    const other = new NeuralReranker(db); cleanup.push(() => other.dispose());
    const results = await Promise.allSettled([controller.promote('race', accepted.evaluationId, 3), other.promote('race', accepted.evaluationId, 3)]);
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
    await expect(other.promote('race', accepted.evaluationId, 4)).rejects.toThrow('stale');
  });

  it('rechecks candidate/evaluation bindings after async preparation', async () => {
    const { controller, db } = setup(); const { evaluation, candidate } = await accepted(controller);
    const pending = controller.promote('memory', evaluation.evaluationId, 1);
    const tampered = { ...evaluation, reasons: ['changed after receipt read'] };
    db.prepare('UPDATE neural_rerank_evaluations SET metadata_json = ? WHERE id = ?').run(JSON.stringify(tampered), evaluation.evaluationId);
    await expect(pending).rejects.toThrow('changed during promotion'); expect(controller.get('memory')!.revision).toBe(1);
    db.prepare('UPDATE neural_rerank_candidates SET model_json = ? WHERE id = ?').run('{}', candidate.candidateId);
    await expect(controller.rank('memory', [], { candidateId: candidate.candidateId })).rejects.toThrow('hash mismatch');
    await expect(controller.promote('memory', evaluation.evaluationId, 1)).rejects.toThrow('hash mismatch');
  });

  it('rejects reset/recreation ABA and disposal across asynchronous operations', async () => {
    const { controller, db } = setup(); controller.create('aba');
    const pending = controller.train('aba', groups('train'));
    db.prepare('DELETE FROM neural_rerankers WHERE id = ?').run('aba'); controller.create('aba');
    await expect(pending).rejects.toThrow('stale');
    const candidate = await controller.train('aba', groups('fresh'));
    const evaluation = await controller.evaluate('aba', candidate.candidateId, groups('heldout', 5));
    db.prepare('DELETE FROM neural_rerankers WHERE id = ?').run('aba'); controller.create('aba');
    await expect(controller.promote('aba', evaluation.evaluationId, 1)).rejects.toThrow('not found');
    const cancelling = controller.train('aba', groups('cancel'));
    resetDatabaseTables(db); controller.dispose();
    const rebuilt = new NeuralReranker(db); cleanup.push(() => rebuilt.dispose()); rebuilt.create('aba');
    await expect(cancelling).rejects.toThrow('disposed');
    expect(rebuilt.get('aba')!.revision).toBe(1); expect(rebuilt.get('aba')!.candidates).toEqual([]);
  });

  it('rejects stale inference and disposed evaluation instead of returning old-source success', async () => {
    const { controller } = setup(); const { evaluation } = await accepted(controller);
    await controller.promote('memory', evaluation.evaluationId, 1);
    const pending = controller.rank('memory', unlabeled(groups('rank', 1)[0]), { mode: 'active' });
    controller.rollback('memory', 1, 2);
    await expect(pending).rejects.toThrow('stale');
    const candidate = await controller.train('memory', groups('fresh'));
    const checking = controller.evaluate('memory', candidate.candidateId, groups('heldout', 5));
    controller.dispose(); await expect(checking).rejects.toThrow('disposed');
    expect(() => controller.get('memory')).toThrow('disposed');
  });

  it('caps in-flight work, candidate/evaluation retention and prepared LRU while disposing evictions', async () => {
    const { controller } = setup(); controller.create('bounded');
    const four = [1, 2, 3, 4].map(epochs => controller.train('bounded', groups(`train-${epochs}`), { epochs }));
    await expect(controller.train('bounded', groups('excess'))).rejects.toThrow('in-flight'); await Promise.all(four);
    const cache = (controller as any).cache as Map<string, any>; const evicted = cache.values().next().value;
    for (let epochs = 5; epochs <= 17; epochs++) await controller.train('bounded', groups(`train-${epochs}`), { epochs });
    expect(controller.get('bounded')!.candidates).toHaveLength(16); expect(cache.size).toBe(8);
    expect(() => evicted.predict(new Float64Array([0, 0, 0, 0]))).toThrow('disposed');
    const candidate = controller.get('bounded')!.candidates[0];
    for (let i = 0; i < 17; i++) await controller.evaluate('bounded', candidate.candidateId, groups(`eval-${i}`, 5), { minNdcgImprovement: 1 });
    expect(controller.get('bounded')!.evaluations).toHaveLength(16);
    const retained = [...cache.values()]; controller.dispose(); controller.dispose();
    for (const prepared of retained) expect(() => prepared.predict(new Float64Array([0, 0, 0, 0]))).toThrow('disposed');
  });

  it('hash-checks rollback, retains only 32 checkpoints and prevents revision overflow', async () => {
    const { controller, db } = setup(); const { evaluation } = await accepted(controller);
    await controller.promote('memory', evaluation.evaluationId, 1);
    const active = controller.get('memory')!;
    db.prepare('UPDATE neural_rerank_checkpoints SET model_json = ? WHERE reranker_id = ? AND revision = 2').run('{}', 'memory');
    expect(() => controller.rollback('memory', 2, 2)).toThrow('hash mismatch'); expect(controller.get('memory')!.model).toEqual(active.model);
    db.prepare('UPDATE neural_rerank_checkpoints SET model_json = ? WHERE reranker_id = ? AND revision = 2').run(JSON.stringify(active.model), 'memory');
    for (let revision = 2; revision < 35; revision++) controller.rollback('memory', revision, revision);
    expect(controller.get('memory')!.history).toHaveLength(32);
    expect(() => controller.rollback('memory', 1, 35)).toThrow('not retained');
    db.prepare('UPDATE neural_rerankers SET revision = ? WHERE id = ?').run(Number.MAX_SAFE_INTEGER, 'memory');
    expect(() => controller.rollback('memory', 35, Number.MAX_SAFE_INTEGER)).toThrow('next revision');
    expect(controller.get('memory')!.revision).toBe(Number.MAX_SAFE_INTEGER);
  });

  it('rejects unsupported persisted feature versions on operations while allowing inspection', async () => {
    const { controller, db } = setup(); const { candidate, evaluation } = await accepted(controller);
    db.prepare('UPDATE neural_rerankers SET feature_schema = ? WHERE id = ?').run('agentdb.rerank-features/999', 'memory');
    expect(controller.get('memory')!.featureSchema).toBe('agentdb.rerank-features/999');
    await expect(controller.inspectWeights('memory')).rejects.toThrow('feature schema');
    await expect(controller.proposePatch('memory', {} as any)).rejects.toThrow('feature schema');
    await expect(controller.rank('memory', [], { candidateId: candidate.candidateId })).rejects.toThrow('feature schema');
    await expect(controller.train('memory', groups('other'))).rejects.toThrow('feature schema');
    await expect(controller.evaluate('memory', candidate.candidateId, groups('other', 5))).rejects.toThrow('feature schema');
    await expect(controller.promote('memory', evaluation.evaluationId, 1)).rejects.toThrow('feature schema');
    expect(() => controller.rollback('memory', 1, 1)).toThrow('feature schema');
  });

  it('keeps SQL failures atomic and readonly construction free from schema writes', async () => {
    const { controller, db } = setup(); const { evaluation } = await accepted(controller);
    const exec = vi.spyOn(db, 'exec'); const reader = new NeuralReranker(db, { readonly: true }); cleanup.push(() => reader.dispose());
    expect(exec).not.toHaveBeenCalled(); exec.mockRestore();
    expect(reader.get('memory')!.revision).toBe(1);
    expect((await reader.inspectWeights('memory')).source).toBe('seed');
    await expect(reader.proposePatch('memory', {} as any)).rejects.toThrow('read-only');
    expect(() => reader.create('denied')).toThrow('read-only');
    await expect(reader.train('memory', groups('denied'))).rejects.toThrow('read-only');
    await expect(reader.evaluate('memory', 'missing', groups('denied', 5))).rejects.toThrow('read-only');
    await expect(reader.promote('memory', evaluation.evaluationId, 1)).rejects.toThrow('read-only');
    expect(() => reader.rollback('memory', 1, 1)).toThrow('read-only');
    db.exec("CREATE TRIGGER fail_rerank_checkpoint BEFORE INSERT ON neural_rerank_checkpoints BEGIN SELECT RAISE(ABORT, 'checkpoint failure'); END");
    await expect(controller.promote('memory', evaluation.evaluationId, 1)).rejects.toThrow('checkpoint failure');
    expect(controller.get('memory')!.revision).toBe(1); expect(controller.get('memory')!.history).toHaveLength(1);
  });

  it('persists candidates and accepted evaluations through real file reopen', async () => {
    const directory = mkdtempSync(join(tmpdir(), 'agentdb-reranker-')); cleanup.push(() => rmSync(directory, { recursive: true, force: true }));
    const filename = join(directory, 'rank.db');
    const raw: any = driver === 'native' ? new Database(filename) : new SQL.Database();
    const db: any = driver === 'native' ? raw : wrapExistingSqlJsDatabase(raw);
    const original = new NeuralReranker(db); const { candidate } = await accepted(original);
    const weights = await original.inspectWeights('memory', candidate.candidateId);
    const patched = await original.proposePatch('memory', { schema: 'ruv-fann.weight-patch/1', baseHash: weights.baseHash,
      edits: [{ layer: 1, parameter: 'bias', index: 0, value: weights.model.layers[1].bias![0] + 0.01 }] }, candidate.candidateId);
    const evaluation = await original.evaluate('memory', patched.candidateId, groups('patch-heldout', 5));
    expect(evaluation.accepted).toBe(true);
    if (driver === 'sqljs') writeFileSync(filename, raw.export()); original.dispose(); db.close();
    const restored: any = driver === 'native' ? new Database(filename) : wrapExistingSqlJsDatabase(new SQL.Database(readFileSync(filename)));
    cleanup.push(() => restored.close()); const reopened = new NeuralReranker(restored); cleanup.push(() => reopened.dispose());
    expect(reopened.get('memory')!.candidates[0].modelHash).toBe(patched.modelHash);
    expect(reopened.get('memory')!.candidates[0].trainingQueryIds).toEqual(candidate.trainingQueryIds);
    expect((await reopened.promote('memory', evaluation.evaluationId, 1)).revision).toBe(2);
    expect((await reopened.rank('memory', unlabeled(groups('new', 1)[0]), { mode: 'active' })).order[0]).toBe('grade-3');
    expect(reopened.get('memory')!.trainingQueryIds).toEqual(candidate.trainingQueryIds);
    const active = await reopened.inspectWeights('memory');
    const descendant = await reopened.proposePatch('memory', { schema: 'ruv-fann.weight-patch/1', baseHash: active.baseHash,
      edits: [{ layer: 1, parameter: 'bias', index: 0, value: active.model.layers[1].bias![0] + 0.01 }] });
    await expect(reopened.evaluate('memory', descendant.candidateId, groups('train', 5))).rejects.toThrow('overlap');
    reopened.rollback('memory', 1, 2); expect(reopened.get('memory')!.trainingQueryIds).toEqual([]);
    reopened.rollback('memory', 2, 3); expect(reopened.get('memory')!.trainingQueryIds).toEqual(candidate.trainingQueryIds);
    if (driver === 'native') {
      const readonly = new Database(filename, { readonly: true }); cleanup.push(() => readonly.close());
      const reader = new NeuralReranker(readonly, { readonly: true }); cleanup.push(() => reader.dispose());
      expect((await reader.rank('memory', [], { mode: 'active' })).available).toBe(true);
    }
  });
});
