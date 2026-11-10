#!/usr/bin/env node
/** Reproducible synthetic smoke benchmark. It does not establish production relevance. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createDatabase } from '../dist/src/db-fallback.js';
import { NeuralReranker, RERANK_FEATURE_SCHEMA } from '../dist/src/controllers/NeuralReranker.js';

const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== '--output')) throw new Error('Usage: node scripts/neural-rerank-eval.mjs [--output path]');
const output = resolve(args[1] ?? '/tmp/neural-rerank-evaluation.json');
function random(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}
// An explicit synthetic data-generating process: relevance is a scenario grade;
// reward is a noisy historical signal. This is NOT automatically labeled user data.
function groups(prefix, count, seed) {
  const rng = random(seed);
  return Array.from({ length: count }, (_, query) => ({
    queryId: `${prefix}-${query}`,
    candidates: Array.from({ length: 8 }, (_, index) => {
      const relevance = Math.floor(index / 2);
      return { id: `${prefix}-${query}-${index}`, relevance, features: {
        similarity: rng(), reward: 0.75 * relevance / 3 + 0.25 * rng(),
        success: rng() < (relevance >= 2 ? 0.8 : 0.2) ? 1 : 0, recency: rng(),
      } };
    }).sort((a, b) => b.features.similarity - a.features.similarity),
  }));
}
const training = groups('train', 16, 42);
const validation = groups('validation', 8, 8128);
const heldout = groups('heldout', 8, 4001);
const features = group => group.candidates.map(({ id, features }) => ({ id, features }));
const trainIds = new Set(training.map(group => group.queryId));
const validationIds = new Set(validation.map(group => group.queryId));
assert(heldout.every(group => !trainIds.has(group.queryId) && !validationIds.has(group.queryId)));
const db = await createDatabase(':memory:');
const controller = new NeuralReranker(db, { simd: false });
try {
  const initial = controller.create('synthetic-evaluation');
  // Fixed candidates, chosen only on validation groups. Never tune from heldout outcomes.
  const candidates = [];
  for (const epochs of [1, 250]) {
    const trained = await controller.train(initial.id, training, { epochs, learningRate: 0.01 });
    const evaluation = await controller.evaluate(initial.id, trained.candidateId, validation);
    candidates.push({ trained, evaluation });
  }
  candidates.sort((a, b) => b.evaluation.candidate.ndcg - a.evaluation.candidate.ndcg || a.trained.epochs - b.trained.epochs);
  const selected = candidates[0];
  const evaluation = await controller.evaluate(initial.id, selected.trained.candidateId, heldout);
  assert.equal(evaluation.accepted, true, `Heldout gate rejected: ${evaluation.reasons.join(', ')}`);
  assert(evaluation.candidate.ndcg > evaluation.baseline.ndcg);
  assert(selected.trained.finalLoss < selected.trained.initialLoss);

  const rejected = await controller.evaluate(initial.id, selected.trained.candidateId, validation, { minNdcgImprovement: 1 });
  assert.equal(rejected.accepted, false);
  await assert.rejects(controller.promote(initial.id, rejected.evaluationId, initial.revision));
  const sample = features(heldout[0]);
  const shadow = await controller.rank(initial.id, sample, { candidateId: selected.trained.candidateId });
  assert.deepEqual(shadow.order, sample.map(row => row.id));
  const promoted = await controller.promote(initial.id, evaluation.evaluationId, initial.revision);
  let changed = false;
  for (const group of heldout) {
    const ranked = await controller.rank(initial.id, features(group), { mode: 'active' });
    assert(ranked.scores.every(row => Number.isFinite(row.score)));
    changed ||= ranked.order.join('|') !== group.candidates.map(row => row.id).join('|');
  }
  assert(changed);
  const ties = [{ id: 'tie-z', features: sample[0].features }, { id: 'tie-a', features: sample[0].features }];
  assert.deepEqual((await controller.rank(initial.id, ties, { mode: 'active' })).order, ['tie-z', 'tie-a']);
  const times = [];
  for (let i = 0; i < 60; i++) {
    const start = performance.now();
    await controller.rank(initial.id, sample, { mode: 'active' });
    if (i >= 10) times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  const restored = controller.rollback(initial.id, initial.revision, promoted.revision);
  assert.equal(restored.modelHash, null);
  assert.deepEqual((await controller.rank(initial.id, sample, { mode: 'active' })).order, sample.map(row => row.id));
  // Predetermined analytic toy scorer: tanh(reward), with no training or tuning.
  const patchInitial = controller.create('synthetic-weight-patch');
  const inspection = await controller.inspectWeights(patchInitial.id);
  const edits = inspection.model.layers.flatMap((layer, layerIndex) => [
    ...layer.weights.map((_, index) => ({ layer: layerIndex, parameter: 'weights', index,
      value: (layerIndex === 0 && index === 1) || (layerIndex === 1 && index === 0) ? 1 : 0 })),
    ...layer.bias.map((_, index) => ({ layer: layerIndex, parameter: 'bias', index, value: 0 })),
  ]);
  const patched = await controller.proposePatch(patchInitial.id, {
    schema: 'ruv-fann.weight-patch/1', baseHash: inspection.baseHash, edits,
  });
  assert.equal(patched.origin, 'weight-patch');
  for (const key of ['epochs', 'initialLoss', 'finalLoss', 'datasetHash']) assert.equal(patched[key], undefined);
  const patchEvaluation = await controller.evaluate(patchInitial.id, patched.candidateId, heldout);
  assert.equal(patchEvaluation.accepted, true);
  await controller.promote(patchInitial.id, patchEvaluation.evaluationId, patchInitial.revision);
  const patchRank = await controller.rank(patchInitial.id, sample, { mode: 'active' });
  assert.notDeepEqual(patchRank.order, sample.map(row => row.id));
  assert(patchRank.scores.every(row => Number.isFinite(row.score)));
  const report = {
    schema: 'agentdb.neural-rerank-eval/1', status: 'passed', dataset: 'synthetic-v1',
    sourceSha: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    featureSchema: RERANK_FEATURE_SCHEMA, runtime: process.version, backend: evaluation.backend,
    trainingQueries: training.length, validationQueries: validation.length, heldoutQueries: heldout.length,
    selection: { method: 'validation NDCG only; fewer epochs break ties', epochs: selected.trained.epochs,
      candidates: candidates.map(({ trained, evaluation }) => ({ epochs: trained.epochs, initialLoss: trained.initialLoss,
        finalLoss: trained.finalLoss, validation: evaluation.candidate })) },
    baseline: evaluation.baseline, candidate: evaluation.candidate,
    improvement: evaluation.candidate.ndcg - evaluation.baseline.ndcg,
    weightPatch: { method: 'Predetermined tanh(reward) toy scorer; no training', edits: edits.length,
      baseline: patchEvaluation.baseline, candidate: patchEvaluation.candidate },
    inference: { candidates: sample.length, warmup: 10, samples: times.length, medianMs: times[25], p95Ms: times[47] },
    checks: { gradientFreePatchCandidate: true, heldoutQueryIdsDisjoint: true, shadowPreservesOrder: true, promotionChangesOrder: true,
      rollbackRestoresBaseline: true, deterministicTieOrder: true, finiteScores: true, rejectedEvaluationCannotPromote: true },
    limitation: 'Controlled synthetic feature-ranking benchmark; not evidence of production relevance, generalization, or end-to-end retrieval latency.',
  };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report));
} finally { controller.dispose(); db.close(); }
