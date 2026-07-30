export interface PromotionEvidence {
  baseline: {
    primary: number;
    noopRate: number;
    costPerWin: number;
    regressed?: boolean;
  };
  candidate: {
    primary: number;
    noopRate: number;
    costPerWin: number;
    regressed?: boolean;
  };
  anchor?: { baseline: number; candidate: number };
}

export interface PromotionDecision {
  promote: boolean;
  reasons: string[];
}

type RuvectorMetaHarness = {
  getMetaHarnessCapabilities(): Promise<unknown[]>;
  evaluateMetaHarnessPromotion(evidence: PromotionEvidence): Promise<PromotionDecision>;
  verifyMetaHarnessReplay(
    bundle: unknown,
    options?: { pinnedGateFingerprint?: string },
  ): Promise<Record<string, unknown>>;
  scanMetaHarnessRewardHacks(
    trajectory: Record<string, unknown>,
  ): Promise<{ clean: boolean; findings: unknown[] }>;
  assertMetaHarnessSafePayload(payload: unknown): Promise<void>;
};

type ModuleLoader = () => Promise<RuvectorMetaHarness>;

const defaultLoader: ModuleLoader = async () => {
  const loaded = await import('ruvector') as unknown as RuvectorMetaHarness;
  const required = [
    'getMetaHarnessCapabilities',
    'evaluateMetaHarnessPromotion',
    'verifyMetaHarnessReplay',
    'scanMetaHarnessRewardHacks',
    'assertMetaHarnessSafePayload',
  ] as const;
  for (const name of required) {
    if (typeof loaded[name] !== 'function') {
      throw new Error(
        `Installed ruvector does not expose ${name}; install ruvector >= 0.2.40`,
      );
    }
  }
  return loaded;
};

/**
 * Lazy governance boundary around ruvector's pinned MetaHarness integration.
 * Candidate/Darwin execution is intentionally absent: AgentDB stores evidence
 * and verifies promotion, but never executes research candidates over MCP.
 */
export class MetaHarnessGateway {
  private pending: Promise<RuvectorMetaHarness> | null = null;

  constructor(private readonly loader: ModuleLoader = defaultLoader) {}

  private load(): Promise<RuvectorMetaHarness> {
    this.pending ??= this.loader();
    return this.pending;
  }

  async capabilities(): Promise<unknown[]> {
    return (await this.load()).getMetaHarnessCapabilities();
  }

  async evaluatePromotion(evidence: PromotionEvidence): Promise<PromotionDecision> {
    validatePromotionEvidence(evidence);
    const api = await this.load();
    await api.assertMetaHarnessSafePayload(evidence);
    return api.evaluateMetaHarnessPromotion(evidence);
  }

  async verifyReplay(
    bundle: unknown,
    pinnedGateFingerprint?: string,
  ): Promise<Record<string, unknown>> {
    const api = await this.load();
    await api.assertMetaHarnessSafePayload(bundle);
    return api.verifyMetaHarnessReplay(
      bundle,
      pinnedGateFingerprint ? { pinnedGateFingerprint } : {},
    );
  }

  async scanRewardHacks(
    trajectory: Record<string, unknown>,
  ): Promise<{ clean: boolean; findings: unknown[] }> {
    const api = await this.load();
    return api.scanMetaHarnessRewardHacks(trajectory);
  }
}

function validatePromotionEvidence(evidence: PromotionEvidence): void {
  for (const side of ['baseline', 'candidate'] as const) {
    for (const field of ['primary', 'noopRate', 'costPerWin'] as const) {
      if (!Number.isFinite(evidence[side][field])) {
        throw new TypeError(`${side}.${field} must be finite`);
      }
    }
  }
  if (evidence.anchor) {
    if (!Number.isFinite(evidence.anchor.baseline) || !Number.isFinite(evidence.anchor.candidate)) {
      throw new TypeError('anchor values must be finite');
    }
  }
}
