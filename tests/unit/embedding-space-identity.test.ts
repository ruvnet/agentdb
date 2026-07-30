import { describe, expect, it } from 'vitest';
import {
  EmbeddingSpaceMismatchError,
  assertEmbeddingSpaceCompatible,
  createEmbeddingSpaceIdentity,
  type EmbeddingRolePolicy,
  type EmbeddingSpaceIdentityInput
} from '../../src/embedding/EmbeddingSpaceIdentity.js';
import {
  EmbeddingService,
  registerEmbeddingModelRolePolicy
} from '../../src/controllers/EmbeddingService.js';
import { MemoryController } from '../../src/controllers/MemoryController.js';

const rolePolicy = (queryTemplate: string): EmbeddingRolePolicy => ({
  kind: 'asymmetric',
  prefixPolicy: 'e5-retrieval',
  prefixPolicyVersion: 'artifact-v1',
  queryTemplate,
  passageTemplate: 'passage: {text}'
});

const identityInput = (policy: EmbeddingRolePolicy): EmbeddingSpaceIdentityInput => ({
  modelId: 'example/e5-pinned-revision',
  modelArtifactHash: 'sha256:model',
  tokenizerHash: 'sha256:tokenizer',
  promptTemplateHash: 'sha256:pinned-prompt-artifact',
  poolingStrategy: 'mean',
  truncationLength: 512,
  outputDimension: 8,
  outputDtype: 'float32',
  normalization: 'l2',
  runtimeRevision: 'onnx:graph-sha256',
  distanceMetric: 'cosine',
  rolePolicy: policy
});

describe('EmbeddingSpaceIdentity', () => {
  it('is deterministic and changes when only the query template changes', () => {
    const first = createEmbeddingSpaceIdentity(identityInput(rolePolicy('query: {text}')));
    const reordered = createEmbeddingSpaceIdentity({
      ...identityInput(rolePolicy('query: {text}'))
    });
    const changed = createEmbeddingSpaceIdentity(
      identityInput(rolePolicy('query_instruction: {text}'))
    );

    expect(first.hash).toMatch(/^[a-f0-9]{64}$/);
    expect(reordered.hash).toBe(first.hash);
    expect(changed.hash).not.toBe(first.hash);
  });

  it('rejects reuse and mutation but preserves vector-only reads', () => {
    const expected = createEmbeddingSpaceIdentity(identityInput(rolePolicy('query: {text}')));
    const changed = createEmbeddingSpaceIdentity(
      identityInput(rolePolicy('changed query: {text}'))
    );

    expect(() =>
      assertEmbeddingSpaceCompatible(expected, changed, 'cache-reuse')
    ).toThrow(EmbeddingSpaceMismatchError);
    expect(() =>
      assertEmbeddingSpaceCompatible(expected, changed, 'corpus-mutation')
    ).toThrow(EmbeddingSpaceMismatchError);
    expect(() =>
      assertEmbeddingSpaceCompatible(expected, changed, 'vector-read')
    ).not.toThrow();
  });
});

describe('role-aware embeddings', () => {
  it('makes role authoritative for asymmetric models', async () => {
    const policy = rolePolicy('query: {text}');
    registerEmbeddingModelRolePolicy('example/e5-pinned-revision', policy);
    const service = new EmbeddingService({
      model: 'example/e5-pinned-revision',
      dimension: 8,
      provider: 'local',
      embeddingSpace: identityInput(policy)
    });

    await expect(service.embed('same text')).rejects.toThrow(/ambiguous/);
    const query = await service.embedQuery('same text');
    const passage = await service.embedPassage('same text');
    expect(Array.from(query)).not.toEqual(Array.from(passage));
    expect(service.getEmbeddingSpaceIdentity().rolePolicy).toEqual(policy);
  });

  it('rejects provider declarations that conflict with the exact model registry', () => {
    expect(() =>
      new EmbeddingService({
        model: 'Xenova/all-MiniLM-L6-v2',
        dimension: 384,
        provider: 'local',
        rolePolicy: rolePolicy('query: {text}')
      })
    ).toThrow(/disagrees with the model registry/);
  });

  it('coalesces concurrent requests for the same identity, role, and text', async () => {
    const service = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 2,
      provider: 'transformers'
    });
    let calls = 0;
    (service as unknown as { pipeline: Function }).pipeline = async () => {
      calls++;
      await Promise.resolve();
      return { data: [1, 0] };
    };

    const [first, second] = await Promise.all([
      service.embedQuery('one request'),
      service.embedQuery('one request')
    ]);
    expect(calls).toBe(1);
    expect(second).toBe(first);
  });
});

describe('memory compatibility and provenance', () => {
  it('blocks foreign mutation while retaining vector reads', async () => {
    const expected = createEmbeddingSpaceIdentity(identityInput(rolePolicy('query: {text}')));
    const changed = createEmbeddingSpaceIdentity(
      identityInput(rolePolicy('changed query: {text}'))
    );
    const controller = new MemoryController(null, {
      embeddingSpaceIdentity: expected
    });

    await controller.store({
      id: 'trusted',
      embedding: [1, 0],
      embeddingSpaceIdentity: expected,
      provenance: {
        type: 'tool_result',
        runId: 'run-1',
        trust: 'signed'
      }
    });
    await expect(
      controller.store({
        id: 'foreign',
        embedding: [0, 1],
        embeddingSpaceIdentity: changed
      })
    ).rejects.toThrow(EmbeddingSpaceMismatchError);

    const reads = await controller.search([1, 0], {
      embeddingSpaceIdentity: changed
    });
    expect(reads.map(result => result.id)).toEqual(['trusted']);
    expect(reads[0].provenance?.type).toBe('tool_result');
  });
});
