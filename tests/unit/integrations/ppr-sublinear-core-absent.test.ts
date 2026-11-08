/**
 * @ruvnet/sublinear-core is optional. When it cannot be imported,
 * SparsificationService answers PPR from graph-transformer or the in-tree
 * fallback, and the fallback still gives scores that sum to 1.
 */
import { describe, it, expect, vi } from 'vitest';

vi.mock('@ruvnet/sublinear-core', () => {
  throw new Error("Cannot find package '@ruvnet/sublinear-core'");
});

import { SparsificationService } from '../../../src/controllers/SparsificationService.js';
import { loadSublinearCore } from '../../../src/controllers/integrations/SublinearCoreAdapter.js';

const sinkGraph: number[][] = [[1, 2], [2, 3], [], [0, 4], [], [0]];

describe('@ruvnet/sublinear-core absent', () => {
  it('the loader resolves to null instead of throwing', async () => {
    expect(await loadSublinearCore()).toBeNull();
  });

  it('PPR still runs and its scores sum to 1', async () => {
    const svc = new SparsificationService({ method: 'ppr', topK: 3, maxIterations: 200, convergenceThreshold: 1e-12 });
    await svc.initialize();
    expect(svc.getPprRuntime()).not.toBe('sublinear-core');
    (svc as any).pprBackend = null; // graph-transformer absent as well
    const result = await svc.pprSparsification(0, sinkGraph, 3, 0.15);
    expect(result.method).toBe('ppr-fallback');
    const total = Array.from(result.scores).reduce((a, b) => a + b, 0);
    expect(Math.abs(total - 1)).toBeLessThan(1e-6);
    expect(result.topKIndices[0]).toBe(0);
  });
});
