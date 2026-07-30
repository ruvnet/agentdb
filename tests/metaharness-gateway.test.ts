import { describe, expect, it, vi } from 'vitest';
import { MetaHarnessGateway } from '../src/governance/MetaHarnessGateway.js';

const evidence = {
  baseline: { primary: 0.5, noopRate: 0.1, costPerWin: 2 },
  candidate: { primary: 0.7, noopRate: 0.05, costPerWin: 1 },
};

describe('MetaHarnessGateway', () => {
  it('loads lazily, guards payloads and delegates promotion', async () => {
    const assertSafe = vi.fn(async () => undefined);
    const evaluate = vi.fn(async () => ({ promote: true, reasons: ['better'] }));
    const loader = vi.fn(async () => ({
      getMetaHarnessCapabilities: async () => [],
      evaluateMetaHarnessPromotion: evaluate,
      verifyMetaHarnessReplay: async () => ({}),
      scanMetaHarnessRewardHacks: async () => ({ clean: true, findings: [] }),
      assertMetaHarnessSafePayload: assertSafe,
    }));
    const gateway = new MetaHarnessGateway(loader);
    expect(loader).not.toHaveBeenCalled();

    await expect(gateway.evaluatePromotion(evidence)).resolves.toEqual({
      promote: true,
      reasons: ['better'],
    });
    expect(loader).toHaveBeenCalledTimes(1);
    expect(assertSafe).toHaveBeenCalledWith(evidence);
    expect(evaluate).toHaveBeenCalledWith(evidence);
  });

  it('rejects invalid evidence before loading optional dependencies', async () => {
    const loader = vi.fn();
    const gateway = new MetaHarnessGateway(loader as never);
    await expect(gateway.evaluatePromotion({
      ...evidence,
      candidate: { ...evidence.candidate, primary: Number.NaN },
    })).rejects.toThrow(/candidate.primary must be finite/);
    expect(loader).not.toHaveBeenCalled();
  });
});

