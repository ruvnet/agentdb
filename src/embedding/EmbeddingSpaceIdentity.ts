/**
 * Canonical identity for an embedding space.
 *
 * The hash is intentionally independent of object insertion order. It is safe
 * to persist in RVF metadata, cache keys, experiment manifests, and receipts.
 */

export type EmbeddingRole = 'query' | 'passage';
export type EmbeddingRolePolicy =
  | { kind: 'symmetric' }
  | {
      kind: 'asymmetric';
      prefixPolicy: string;
      prefixPolicyVersion: string;
      queryTemplate: string;
      passageTemplate: string;
    };

export type EmbeddingProvenanceType =
  | 'user_claim'
  | 'agent_output'
  | 'system_observation'
  | 'tool_result'
  | 'unknown';

export interface MemoryProvenance {
  type: EmbeddingProvenanceType;
  sourceId?: string;
  runId?: string;
  receiptId?: string;
  createdAt?: string;
  trust?: 'unverified' | 'verified' | 'signed';
  signature?: string;
}

export interface EmbeddingSpaceIdentityInput {
  modelId: string;
  modelArtifactHash: string;
  tokenizerHash: string;
  promptTemplateHash: string;
  poolingStrategy: string;
  truncationLength: number;
  outputDimension: number;
  outputDtype: string;
  normalization: string;
  runtimeRevision: string;
  distanceMetric: string;
  rolePolicy: EmbeddingRolePolicy;
}

export interface EmbeddingSpaceIdentity extends EmbeddingSpaceIdentityInput {
  schema: 'agentdb.embedding-space.v1';
  hashAlgorithm: 'sha256';
  hash: string;
}

export type EmbeddingCompatibilityOperation =
  | 'cache-reuse'
  | 'corpus-mutation'
  | 'vector-read';

export class EmbeddingSpaceMismatchError extends Error {
  readonly code = 'AGENTDB_EMBEDDING_SPACE_MISMATCH';

  constructor(
    readonly expectedHash: string,
    readonly actualHash: string,
    readonly operation: EmbeddingCompatibilityOperation
  ) {
    super(
      `Embedding space mismatch for ${operation}: expected ${expectedHash}, received ${actualHash}`
    );
    this.name = 'EmbeddingSpaceMismatchError';
  }
}

function canonicalize(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(',')}]`;
  }
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record)
    .sort()
    .map(key => `${JSON.stringify(key)}:${canonicalize(record[key])}`)
    .join(',')}}`;
}

// Small, dependency-free SHA-256 keeps identity creation synchronous in Node,
// browsers, WASM hosts, and edge runtimes.
function sha256(input: string): string {
  const bytes = new TextEncoder().encode(input);
  const bitLength = bytes.length * 8;
  const paddedLength = (((bytes.length + 9 + 63) >> 6) << 6);
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000), false);

  const constants = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  ]);
  const state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
  ]);
  const words = new Uint32Array(64);
  const rotate = (value: number, bits: number): number =>
    (value >>> bits) | (value << (32 - bits));

  for (let offset = 0; offset < padded.length; offset += 64) {
    for (let i = 0; i < 16; i++) words[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const a = words[i - 15];
      const b = words[i - 2];
      const s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >>> 3);
      const s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >>> 10);
      words[i] = (words[i - 16] + s0 + words[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = state;
    for (let i = 0; i < 64; i++) {
      const s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const choose = (e & f) ^ (~e & g);
      const t1 = (h + s1 + choose + constants[i] + words[i]) >>> 0;
      const s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (s0 + majority) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    state[0] = (state[0] + a) >>> 0; state[1] = (state[1] + b) >>> 0;
    state[2] = (state[2] + c) >>> 0; state[3] = (state[3] + d) >>> 0;
    state[4] = (state[4] + e) >>> 0; state[5] = (state[5] + f) >>> 0;
    state[6] = (state[6] + g) >>> 0; state[7] = (state[7] + h) >>> 0;
  }
  return Array.from(state, word => word.toString(16).padStart(8, '0')).join('');
}

export function createEmbeddingSpaceIdentity(
  input: EmbeddingSpaceIdentityInput
): EmbeddingSpaceIdentity {
  if (!input.modelArtifactHash || !input.tokenizerHash || !input.promptTemplateHash) {
    throw new Error('Embedding space identity requires immutable model, tokenizer, and prompt hashes');
  }
  if (!Number.isInteger(input.truncationLength) || input.truncationLength <= 0) {
    throw new Error('Embedding truncationLength must be a positive integer');
  }
  if (!Number.isInteger(input.outputDimension) || input.outputDimension <= 0) {
    throw new Error('Embedding outputDimension must be a positive integer');
  }
  const payload = {
    schema: 'agentdb.embedding-space.v1' as const,
    ...input,
    rolePolicy: Object.freeze({ ...input.rolePolicy })
  };
  return Object.freeze({
    ...payload,
    hashAlgorithm: 'sha256' as const,
    hash: sha256(canonicalize(payload))
  });
}

export function embeddingSpacesEqual(
  left: Pick<EmbeddingSpaceIdentity, 'hash'>,
  right: Pick<EmbeddingSpaceIdentity, 'hash'>
): boolean {
  return left.hash === right.hash;
}

export function assertEmbeddingSpaceCompatible(
  expected: Pick<EmbeddingSpaceIdentity, 'hash'>,
  actual: Pick<EmbeddingSpaceIdentity, 'hash'>,
  operation: EmbeddingCompatibilityOperation
): void {
  if (operation !== 'vector-read' && !embeddingSpacesEqual(expected, actual)) {
    throw new EmbeddingSpaceMismatchError(expected.hash, actual.hash, operation);
  }
}

export function applyRoleTemplate(template: string, text: string): string {
  if (!template.includes('{text}')) {
    throw new Error('Embedding role template must contain {text}');
  }
  return template.replaceAll('{text}', text);
}
