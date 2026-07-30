import { createHash } from 'node:crypto';
import type { PromotionDecision, PromotionEvidence } from './MetaHarnessGateway.js';
import { MetaHarnessGateway } from './MetaHarnessGateway.js';

interface GovernedRvfStore {
  fileId(): Promise<string>;
  parentId(): Promise<string>;
  lineageDepth(): Promise<number>;
  status(): Promise<{ totalVectors: number; totalSegments: number }>;
  segments(): Promise<Array<{ id: number; segType: string; payloadLength: number }>>;
  verifyWitness(): { valid: boolean; entries: number; error?: string };
  derive(childPath: string): Promise<GovernedRvfStore>;
  getStoragePath(): string;
}

export interface ExperimentAnchor {
  repository: string;
  commitSha: string;
  benchmarkHash: string;
  embeddingSpaceHash: string;
}

export interface ExperimentReceipt {
  schemaVersion: 1;
  parentPath: string;
  childPath: string;
  parentFingerprint: string;
  childFingerprint: string;
  anchor: ExperimentAnchor;
  decision: PromotionDecision;
  receiptHash: string;
}

export class StaleExperimentBaselineError extends Error {
  readonly code = 'AGENTDB_STALE_EXPERIMENT_BASELINE';

  constructor() {
    super('The parent RVF generation changed after the experiment branch was created');
    this.name = 'StaleExperimentBaselineError';
  }
}

/**
 * Creates an isolated RVF COW candidate and authorizes it without mutating the
 * authoritative parent. Actual path/alias promotion remains a trusted external
 * transaction so candidate code never receives release authority.
 */
export class RvfExperimentBranch {
  private constructor(
    private readonly parent: GovernedRvfStore,
    readonly candidate: GovernedRvfStore,
    readonly anchor: ExperimentAnchor,
    private readonly parentFingerprint: string,
  ) {}

  static async create(
    parent: GovernedRvfStore,
    childPath: string,
    anchor: ExperimentAnchor,
  ): Promise<RvfExperimentBranch> {
    validateAnchor(anchor);
    const parentFingerprint = await fingerprintStore(parent);
    const parentId = await parent.fileId();
    const candidate = await parent.derive(childPath);
    if (await candidate.parentId() !== parentId) {
      throw new Error('Derived RVF candidate does not identify the expected parent');
    }
    if (await candidate.lineageDepth() < 1) {
      throw new Error('Derived RVF candidate has invalid lineage depth');
    }
    return new RvfExperimentBranch(parent, candidate, anchor, parentFingerprint);
  }

  async authorizePromotion(
    gateway: MetaHarnessGateway,
    evidence: PromotionEvidence,
  ): Promise<ExperimentReceipt> {
    if (await fingerprintStore(this.parent) !== this.parentFingerprint) {
      throw new StaleExperimentBaselineError();
    }
    const witness = this.candidate.verifyWitness();
    if (!witness.valid) {
      throw new Error(`Candidate RVF witness verification failed: ${witness.error ?? 'unknown error'}`);
    }

    const decision = await gateway.evaluatePromotion(evidence);
    const childFingerprint = await fingerprintStore(this.candidate);
    const unsigned = {
      schemaVersion: 1 as const,
      parentPath: this.parent.getStoragePath(),
      childPath: this.candidate.getStoragePath(),
      parentFingerprint: this.parentFingerprint,
      childFingerprint,
      anchor: this.anchor,
      decision,
    };
    return {
      ...unsigned,
      receiptHash: hashCanonical(unsigned),
    };
  }
}

async function fingerprintStore(store: GovernedRvfStore): Promise<string> {
  const [fileId, parentId, lineageDepth, status, segments] = await Promise.all([
    store.fileId(),
    store.parentId(),
    store.lineageDepth(),
    store.status(),
    store.segments(),
  ]);
  return hashCanonical({
    fileId,
    parentId,
    lineageDepth,
    status,
    segments: [...segments].sort((a, b) => a.id - b.id),
  });
}

function hashCanonical(value: unknown): string {
  return createHash('sha256').update(canonicalJson(value)).digest('hex');
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

function validateAnchor(anchor: ExperimentAnchor): void {
  for (const [name, value] of Object.entries(anchor)) {
    if (typeof value !== 'string' || value.trim().length === 0) {
      throw new TypeError(`anchor.${name} must be a non-empty string`);
    }
  }
}

