/**
 * Regression tests for issue #14: SelfLearningRvfBackend's feedback and
 * contrastive-learning loop was disconnected from retrieval in three ways:
 *
 *   1. searchAsync() generated an internal "q_N" correlation key but never
 *      returned it, so recordFeedback() was unreachable through any public
 *      API.
 *   2. recordFeedback() never wrote the matching `recentSearches[].quality`,
 *      so createSamples()'s search for negative examples
 *      (`r.quality < negativeThreshold`) always matched zero entries and no
 *      contrastive sample was ever created.
 *   3. ContrastiveTrainer.project() was never called from the search path,
 *      so even a trained projection could not affect ranking.
 *
 * Fixed by: accepting a caller-supplied `feedbackId` in SearchOptions,
 * propagating recorded quality onto the matching recentSearches entry, and
 * applying trainer.project() to the query once at least one batch has
 * trained.
 */
import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { SelfLearningRvfBackend } from '../../src/backends/rvf/SelfLearningRvfBackend.js';

const DIM = 8;

function vec(seed: number): Float32Array {
  const v = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) v[i] = Math.sin(seed * 7.13 + i * 1.7);
  return v;
}

describe('Issue #14: feedback correlation and contrastive learning', () => {
  let tmpDirs: string[] = [];

  afterEach(() => {
    for (const d of tmpDirs) fs.rmSync(d, { recursive: true, force: true });
    tmpDirs = [];
  });

  async function createBackend(overrides: Record<string, unknown> = {}) {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-feedback-'));
    tmpDirs.push(tmpDir);
    return SelfLearningRvfBackend.create({
      dimension: DIM,
      storagePath: path.join(tmpDir, 'test.rvf'),
      learning: true,
      trainingBatchSize: 4,
      positiveThreshold: 0.6,
      negativeThreshold: 0.4,
      ...overrides,
    });
  }

  it('recordFeedback works with a caller-supplied feedbackId (was unreachable before the fix)', async () => {
    const backend = await createBackend();
    for (let i = 0; i < 5; i++) await backend.insertAsync(`doc_${i}`, vec(i));

    await backend.searchAsync(vec(0), 3, { feedbackId: 'my-feedback-id' });
    // Before the fix, "my-feedback-id" was never a real key in
    // activeTrajectories (only the internal "q_N" format was), so this
    // would silently no-op and trajectoriesRecorded would stay 0.
    backend.recordFeedback('my-feedback-id', 0.9);

    expect(backend.getLearningStats().trajectoriesRecorded).toBe(1);
    backend.close();
  });

  it('rejects a feedbackId reused while still outstanding', async () => {
    const backend = await createBackend();
    for (let i = 0; i < 5; i++) await backend.insertAsync(`doc_${i}`, vec(i));

    await backend.searchAsync(vec(0), 3, { feedbackId: 'dup' });
    await expect(backend.searchAsync(vec(1), 3, { feedbackId: 'dup' })).rejects.toThrow(/already outstanding/);

    backend.recordFeedback('dup', 0.8);
    // Now that "dup" has been resolved, it's free to reuse.
    await expect(backend.searchAsync(vec(1), 3, { feedbackId: 'dup' })).resolves.toBeDefined();
    backend.close();
  });

  it('handles concurrent/out-of-order feedback without cross-talk', async () => {
    const backend = await createBackend();
    for (let i = 0; i < 5; i++) await backend.insertAsync(`doc_${i}`, vec(i));

    await backend.searchAsync(vec(0), 3, { feedbackId: 'first' });
    await backend.searchAsync(vec(1), 3, { feedbackId: 'second' });
    await backend.searchAsync(vec(2), 3, { feedbackId: 'third' });

    // Resolve out of order: third, then first, then second.
    backend.recordFeedback('third', 0.9);
    backend.recordFeedback('first', 0.2);
    backend.recordFeedback('second', 0.7);

    expect(backend.getLearningStats().trajectoriesRecorded).toBe(3);

    // Recording feedback again for an already-resolved id is a no-op, not a
    // crash or a double-count.
    backend.recordFeedback('first', 0.5);
    expect(backend.getLearningStats().trajectoriesRecorded).toBe(3);

    backend.close();
  });

  it('feedback quality actually produces contrastive samples and trained batches (was always 0 before the fix)', async () => {
    const backend = await createBackend({ trainingBatchSize: 2 });
    for (let i = 0; i < 10; i++) await backend.insertAsync(`doc_${i}`, vec(i));

    // A spread of searches with both low- and high-quality feedback: low
    // quality entries become negative-example candidates for later
    // high-quality trajectories, which is what unlocks createSamples().
    for (let round = 0; round < 6; round++) {
      const fid = `round_${round}`;
      await backend.searchAsync(vec(round), 4, { feedbackId: fid });
      // Alternate low/high quality so recentSearches accumulates entries
      // below negativeThreshold that later high-quality rounds can use.
      backend.recordFeedback(fid, round % 2 === 0 ? 0.15 : 0.9);
    }
    await backend.forceLearn();

    const stats = backend.getLearningStats();
    expect(stats.contrastiveSamples).toBeGreaterThan(0);
    expect(stats.contrastiveBatches).toBeGreaterThan(0);

    backend.close();
  });

  it('a trained contrastive projection measurably changes retrieval for a held-out query', async () => {
    const backend = await createBackend({ trainingBatchSize: 2 });
    for (let i = 0; i < 10; i++) await backend.insertAsync(`doc_${i}`, vec(i));

    const heldOut = vec(100); // never used for training feedback

    const before = await backend.searchAsync(heldOut, 5);
    const beforeSims = before.map((r) => r.similarity);

    for (let round = 0; round < 8; round++) {
      const fid = `train_${round}`;
      await backend.searchAsync(vec(round), 4, { feedbackId: fid });
      backend.recordFeedback(fid, round % 2 === 0 ? 0.1 : 0.95);
    }
    // Drive enough gradient steps that the projection moves measurably off
    // its identity-plus-noise initialization.
    for (let i = 0; i < 5; i++) await backend.forceLearn();

    expect(backend.getLearningStats().contrastiveBatches).toBeGreaterThan(0);

    const after = await backend.searchAsync(heldOut, 5);
    const afterSims = after.map((r) => r.similarity);

    // Before the fix, trainer.project() was never called from the search
    // path, so training could never change retrieval for ANY query,
    // held-out or not. Assert the held-out query's results actually moved.
    expect(afterSims).not.toEqual(beforeSims);

    backend.close();
  });
});
