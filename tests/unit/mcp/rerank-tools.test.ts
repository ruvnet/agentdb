import { describe, expect, it, vi } from 'vitest';
import { dispatchRerankTool, isRerankTool, RERANK_TOOLS, validateNeuralRerankQuery } from '../../../src/mcp/rerank-tools.js';
import { buildMcpToolPolicy, isMcpToolAllowed } from '../../../src/security/mcp-policy.js';
const features = () => ({ similarity: 0.5, reward: 0.8, success: 1, recency: 0.5 });
const groups = (count = 5) => Array.from({ length: count }, (_, index) => ({ queryId: `query-${index}`, candidates: [
  { id: 'a', features: features(), relevance: 0 }, { id: 'b', features: features(), relevance: 3 }
] }));
const fake = () => Object.fromEntries(['create', 'get', 'list', 'train', 'evaluate', 'promote', 'rollback', 'rank', 'inspectWeights', 'proposePatch'].map(action => [action, vi.fn((...args) => ({ action, args }))])) as any;

describe('learned reranker strict MCP tools', () => {
  it('exposes exactly ten tools with readonly/learning/admin separation and explicit deny precedence', () => {
    expect(RERANK_TOOLS).toHaveLength(10);
    for (const tool of RERANK_TOOLS) { expect(isRerankTool(tool.name)).toBe(true); expect(tool.inputSchema.additionalProperties).toBe(false); }
    expect(isRerankTool('agentdb_rerank_accept_receipt')).toBe(false);
    for (const profile of ['readonly', 'retrieval', 'learning', 'admin']) {
      const policy = buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: profile });
      for (const action of ['get', 'list', 'rank', 'inspect']) expect(isMcpToolAllowed(`agentdb_rerank_${action}`, policy)).toBe(true);
      for (const action of ['create', 'train', 'evaluate', 'patch']) expect(isMcpToolAllowed(`agentdb_rerank_${action}`, policy)).toBe(profile === 'learning' || profile === 'admin');
      for (const action of ['promote', 'rollback']) expect(isMcpToolAllowed(`agentdb_rerank_${action}`, policy)).toBe(profile === 'admin');
    }
    expect(isMcpToolAllowed('agentdb_rerank_promote', buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: 'admin', AGENTDB_MCP_DENY: 'agentdb_rerank_promote' }))).toBe(false);
  });
  it('dispatches all ten exact argument shapes without accepting evaluation receipts', async () => {
    const controller = fake();
    await dispatchRerankTool(controller, 'agentdb_rerank_create', { id: 'ranker' });
    await dispatchRerankTool(controller, 'agentdb_rerank_get', { id: 'ranker' });
    await dispatchRerankTool(controller, 'agentdb_rerank_list', undefined);
    const input = groups();
    await dispatchRerankTool(controller, 'agentdb_rerank_train', { id: 'ranker', groups: input, options: { epochs: 100, learningRate: 0.01 } });
    await dispatchRerankTool(controller, 'agentdb_rerank_evaluate', { id: 'ranker', candidateId: 'candidate', groups: input, policy: { k: 3 } });
    await dispatchRerankTool(controller, 'agentdb_rerank_promote', { id: 'ranker', evaluationId: 'evaluation', expectedRevision: 1 });
    await dispatchRerankTool(controller, 'agentdb_rerank_rollback', { id: 'ranker', targetRevision: 1, expectedRevision: 2 });
    const candidates = [{ id: 'episode', features: features() }];
    await dispatchRerankTool(controller, 'agentdb_rerank_rank', { id: 'ranker', candidates, options: { mode: 'shadow', candidateId: 'candidate' } });
    expect(controller.train).toHaveBeenCalledWith('ranker', input, { epochs: 100, learningRate: 0.01 });
    expect(controller.promote).toHaveBeenCalledWith('ranker', 'evaluation', 1);
    expect(controller.rank).toHaveBeenCalledWith('ranker', candidates, { mode: 'shadow', candidateId: 'candidate' });
    expect(controller.list).toHaveBeenCalledWith(undefined);
    await dispatchRerankTool(controller, 'agentdb_rerank_inspect', { id: 'ranker', candidateId: 'candidate' });
    const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: 'a'.repeat(64), edits: [{ layer: 0, parameter: 'weights', index: 0, value: 1 }] };
    await dispatchRerankTool(controller, 'agentdb_rerank_patch', { id: 'ranker', patch, baseCandidateId: 'candidate' });
    expect(controller.inspectWeights).toHaveBeenCalledWith('ranker', 'candidate');
    expect(controller.proposePatch).toHaveBeenCalledWith('ranker', patch, 'candidate');
  });
  it('rejects malformed scalars, unknown nested keys, nonfinite features and aggregate overflow before dispatch', async () => {
    const controller = fake();
    for (const raw of [null, [], true, { limit: 0 }, { limit: 101 }, { limit: '2' }, JSON.parse('{"constructor":1}'), JSON.parse('{"__proto__":{}}')])
      await expect(dispatchRerankTool(controller, 'agentdb_rerank_list', raw)).rejects.toThrow('Invalid');
    const valid = { id: 'ranker', groups: groups(1) };
    for (const raw of [{ ...valid, groups: [] }, { ...valid, extra: 1 }, { ...valid, options: { epochs: 1001 } },
      { ...valid, options: { learningRate: Infinity } }, { ...valid, options: { provider: 'external' } },
      { ...valid, groups: [{ ...groups(1)[0], split: 'trusted' }] },
      { ...valid, groups: [{ queryId: 'q', candidates: [{ id: 'a', relevance: 3, features: { ...features(), label: 3 } }, groups(1)[0].candidates[1]] }] },
      { ...valid, groups: groups(9).map((group, index) => ({ queryId: group.queryId, candidates: Array.from({ length: 64 }, (_, row) => ({ id: `c${index}-${row}`, features: features(), relevance: 1 })) })) }
    ]) await expect(dispatchRerankTool(controller, 'agentdb_rerank_train', raw)).rejects.toThrow('Invalid');
    for (const raw of [{ id: 'x', candidates: Array(65).fill({ id: 'c', features: features() }) },
      { id: 'x', candidates: [{ id: 'c', features: { ...features(), reward: NaN } }] },
      { id: 'x', candidates: [{ id: 'c', features: { ...features(), recency: 2 } }] },
      { id: 'x', candidates: [], options: { mode: 'active', candidateId: 'candidate' } }
    ]) await expect(dispatchRerankTool(controller, 'agentdb_rerank_rank', raw)).rejects.toThrow('Invalid');
    for (const raw of [{ id: 'x', evaluationId: { accepted: true }, expectedRevision: 1 },
      { id: 'x', evaluationId: 'e', expectedRevision: 1, evaluation: { accepted: true } },
      { id: 'x', evaluationId: 'e', expectedRevision: Number.MAX_SAFE_INTEGER + 1 }
    ]) await expect(dispatchRerankTool(controller, 'agentdb_rerank_promote', raw)).rejects.toThrow('Invalid');
    expect(controller.train).not.toHaveBeenCalled(); expect(controller.rank).not.toHaveBeenCalled(); expect(controller.promote).not.toHaveBeenCalled();
  });
  it('requires five evaluation groups and validates the nested search option', async () => {
    const controller = fake();
    await expect(dispatchRerankTool(controller, 'agentdb_rerank_evaluate', { id: 'x', candidateId: 'c', groups: groups(4) })).rejects.toThrow('Invalid');
    expect(() => validateNeuralRerankQuery({ id: 'x' })).not.toThrow();
    expect(() => validateNeuralRerankQuery({ id: 'x', mode: 'shadow', candidateId: 'c' })).not.toThrow();
    for (const raw of [null, {}, { id: 'x', trusted: true }, { id: 'x', mode: 'active', candidateId: 'c' }, { id: 'x', mode: 'automatic' }])
      expect(() => validateNeuralRerankQuery(raw)).toThrow('Invalid');
  });
});


describe('direct weight-space patch MCP validation', () => {
  it('rejects unknown keys, malformed fingerprints, unbounded edits and invalid locations before dispatch', async () => {
    const controller = fake();
    const patch = { schema: 'ruv-fann.weight-patch/1', baseHash: 'a'.repeat(64), edits: [{ layer: 0, parameter: 'weights', index: 0, value: 1 }] };
    for (const invalid of [{ ...patch, schema: 'other' }, { ...patch, baseHash: 'not-a-hash' },
      { ...patch, edits: [] }, { ...patch, edits: Array(129).fill(patch.edits[0]) },
      { ...patch, accepted: true }, { ...patch, edits: [{ ...patch.edits[0], value: Infinity }] },
      { ...patch, edits: [{ ...patch.edits[0], value: 1_000_001 }] },
      { ...patch, edits: [{ ...patch.edits[0], parameter: 'activation' }] },
      { ...patch, edits: [{ ...patch.edits[0], index: -1 }] },
      { ...patch, edits: [{ ...patch.edits[0], layer: 64 }] },
      { ...patch, edits: [{ ...patch.edits[0], code: 'execute' }] }
    ]) await expect(dispatchRerankTool(controller, 'agentdb_rerank_patch', { id: 'ranker', patch: invalid })).rejects.toThrow('Invalid');
    for (const raw of [{ id: 'ranker', patch, evaluation: { accepted: true } }, { id: 'ranker', patch, baseCandidateId: { accepted: true } }])
      await expect(dispatchRerankTool(controller, 'agentdb_rerank_patch', raw)).rejects.toThrow('Invalid');
    for (const raw of [{ id: 'ranker', candidateId: 1 }, { id: 'ranker', trusted: true }])
      await expect(dispatchRerankTool(controller, 'agentdb_rerank_inspect', raw)).rejects.toThrow('Invalid');
    expect(controller.proposePatch).not.toHaveBeenCalled(); expect(controller.inspectWeights).not.toHaveBeenCalled();
  });
});
