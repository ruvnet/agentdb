import { describe, expect, it } from 'vitest';
import { MetaHarnessGateway } from '../src/governance/MetaHarnessGateway.js';
import {
  RvfExperimentBranch,
  StaleExperimentBaselineError,
} from '../src/governance/RvfExperimentBranch.js';

function store(options: {
  id: string;
  parent?: string;
  depth?: number;
  path: string;
  segments?: number;
}) {
  let segmentCount = options.segments ?? 1;
  const value: any = {
    fileId: async () => options.id,
    parentId: async () => options.parent ?? '0'.repeat(64),
    lineageDepth: async () => options.depth ?? 0,
    status: async () => ({ totalVectors: 10, totalSegments: segmentCount }),
    segments: async () => Array.from({ length: segmentCount }, (_, id) => ({
      id,
      segType: 'DATA',
      payloadLength: 100,
    })),
    verifyWitness: () => ({ valid: true, entries: 2 }),
    getStoragePath: () => options.path,
    mutate: () => { segmentCount++; },
  };
  return value;
}

const anchor = {
  repository: 'ruvnet/agentdb',
  commitSha: 'abc123',
  benchmarkHash: 'bench-sha256',
  embeddingSpaceHash: 'space-sha256',
};
const evidence = {
  baseline: { primary: 0.5, noopRate: 0.1, costPerWin: 2 },
  candidate: { primary: 0.7, noopRate: 0.05, costPerWin: 1 },
};

describe('RvfExperimentBranch', () => {
  it('binds lineage, anchors and deterministic promotion evidence', async () => {
    const parent = store({ id: 'parent', path: '/store/live.rvf' });
    const child = store({
      id: 'child',
      parent: 'parent',
      depth: 1,
      path: '/store/candidate.rvf',
    });
    parent.derive = async () => child;
    const gateway = new MetaHarnessGateway(async () => ({
      getMetaHarnessCapabilities: async () => [],
      assertMetaHarnessSafePayload: async () => undefined,
      evaluateMetaHarnessPromotion: async () => ({ promote: true, reasons: ['improved'] }),
      verifyMetaHarnessReplay: async () => ({}),
      scanMetaHarnessRewardHacks: async () => ({ clean: true, findings: [] }),
    }));

    const branch = await RvfExperimentBranch.create(parent, child.getStoragePath(), anchor);
    const receipt = await branch.authorizePromotion(gateway, evidence);
    expect(receipt.decision.promote).toBe(true);
    expect(receipt.anchor).toEqual(anchor);
    expect(receipt.receiptHash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('fails closed when the parent changes during evaluation', async () => {
    const parent = store({ id: 'parent', path: '/store/live.rvf' });
    const child = store({
      id: 'child',
      parent: 'parent',
      depth: 1,
      path: '/store/candidate.rvf',
    });
    parent.derive = async () => child;
    const branch = await RvfExperimentBranch.create(parent, child.getStoragePath(), anchor);
    parent.mutate();

    await expect(branch.authorizePromotion({} as MetaHarnessGateway, evidence))
      .rejects.toBeInstanceOf(StaleExperimentBaselineError);
  });
});

