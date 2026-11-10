import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import initSqlJs from 'sql.js';
import { WeightPrograms } from '../../../src/controllers/WeightPrograms.js';
import { wrapExistingSqlJsDatabase } from '../../../src/db-fallback.js';
import { WSP_TOOLS, dispatchWspTool, isWspTool } from '../../../src/mcp/wsp-tools.js';
import { buildMcpToolPolicy, isMcpToolAllowed } from '../../../src/security/mcp-policy.js';

let programs: WeightPrograms;
let db: any;
const model = { schema: 'ruv-fann.dense-layers/1' as const, layers: [
  { inputs: 1, outputs: 1, weights: [0], bias: [0], activation: 'Linear' as const, steepness: 1 },
] };
beforeEach(async () => {
  const SQL = await initSqlJs(); db = wrapExistingSqlJsDatabase(new SQL.Database()); programs = new WeightPrograms(db);
});
afterEach(() => { programs.dispose(); db.close(); });

describe('weight program MCP dispatch and policies', () => {
  it('exposes exactly six strict tools with correct capability profiles', () => {
    expect(WSP_TOOLS).toHaveLength(6);
    for (const tool of WSP_TOOLS) {
      expect(isWspTool(tool.name)).toBe(true);
      expect(tool.inputSchema.additionalProperties).toBe(false);
    }
    expect(isWspTool('agentdb_wsp_eval_receipt')).toBe(false);
    for (const profile of ['readonly', 'retrieval', 'learning', 'admin']) {
      const policy = buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: profile });
      for (const action of ['get', 'list']) expect(isMcpToolAllowed(`agentdb_wsp_${action}`, policy)).toBe(true);
      for (const action of ['create', 'propose', 'promote']) expect(isMcpToolAllowed(`agentdb_wsp_${action}`, policy)).toBe(profile === 'learning' || profile === 'admin');
      expect(isMcpToolAllowed('agentdb_wsp_rollback', policy)).toBe(profile === 'admin');
    }
  });

  it('dispatches an actual evaluated proposal through promotion and rollback', async () => {
    const initial: any = await dispatchWspTool(programs, 'agentdb_wsp_create', { id: 'linear', model });
    const proposed: any = await dispatchWspTool(programs, 'agentdb_wsp_propose', {
      id: 'linear', patch: { schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash, edits: [{ layer: 0, parameter: 'weights', index: 0, value: 2 }] },
      dataset: { inputs: [1, 2], targets: [2, 4], rows: 2 }, policy: { maxMSE: 0, maxRegressionMSE: 0 },
    });
    expect(proposed.evaluation.accepted).toBe(true);
    expect((await dispatchWspTool(programs, 'agentdb_wsp_promote', { id: 'linear', proposalId: proposed.proposalId, expectedRevision: 1 }) as any).revision).toBe(2);
    expect((await dispatchWspTool(programs, 'agentdb_wsp_get', { id: 'linear' }) as any).model.layers[0].weights).toEqual([2]);
    expect((await dispatchWspTool(programs, 'agentdb_wsp_rollback', { id: 'linear', targetRevision: 1, expectedRevision: 2 }) as any).revision).toBe(3);
    expect(await dispatchWspTool(programs, 'agentdb_wsp_list', undefined)).toEqual([expect.objectContaining({ id: 'linear', revision: 3 })]);
  });

  it('accepts finite zero and negative steepness permitted by the SDK', async () => {
    for (const steepness of [0, -1]) {
      const checkpoint = { ...model, layers: [{ ...model.layers[0], steepness }] };
      const record: any = await dispatchWspTool(programs, 'agentdb_wsp_create', { id: `finite${steepness}`, model: checkpoint });
      expect(record.model.layers[0].steepness).toBe(steepness);
    }
  });

  it('rejects unknown/prototype keys, forged receipts and invalid scalar schemas', async () => {
    for (const args of [null, [], true, { limit: 0 }, { limit: 101 }, { limit: '1' }, JSON.parse('{"constructor":1}'), JSON.parse('{"__proto__":{}}'), { evaluation: { accepted: true } }]) {
      await expect(dispatchWspTool(programs, 'agentdb_wsp_list', args)).rejects.toThrow('Invalid');
    }
    for (const args of [
      { id: 'x', proposalId: 'invented', expectedRevision: 0 },
      { id: 'x', proposalId: { accepted: true }, expectedRevision: 1 },
      { id: 'x', proposalId: 'invented', expectedRevision: 1, evaluation: { accepted: true } },
    ]) await expect(dispatchWspTool(programs, 'agentdb_wsp_promote', args)).rejects.toThrow('Invalid');
    await expect(dispatchWspTool(programs, 'unknown', {})).rejects.toThrow('Unknown');
  });

  it('rejects nested unknown keys, nonfinite values and unbounded arrays before evaluation', async () => {
    const initial = await programs.create('safe', model);
    const valid = {
      id: 'safe', patch: { schema: 'ruv-fann.weight-patch/1', baseHash: initial.hash, edits: [{ layer: 0, parameter: 'weights', index: 0, value: 2 }] },
      dataset: { inputs: [1], targets: [2], rows: 1 }, policy: { maxMSE: 0, maxRegressionMSE: 0 },
    };
    for (const args of [
      { ...valid, patch: { ...valid.patch, receipt: true } },
      { ...valid, patch: { ...valid.patch, edits: [{ ...valid.patch.edits[0], activation: 'Linear' }] } },
      { ...valid, dataset: { ...valid.dataset, trusted: true } },
      { ...valid, policy: { ...valid.policy, accepted: true } },
      { ...valid, policy: { ...valid.policy, maxMSE: Infinity } },
      { ...valid, dataset: { ...valid.dataset, inputs: [NaN] } },
      { ...valid, dataset: { ...valid.dataset, inputs: Array(65537).fill(1) } },
      { ...valid, patch: { ...valid.patch, edits: Array(129).fill(valid.patch.edits[0]) } },
    ]) await expect(dispatchWspTool(programs, 'agentdb_wsp_propose', args)).rejects.toThrow('Invalid');
    for (const invalid of [{ ...model, unknown: true }, { ...model, layers: [{ ...model.layers[0], unknown: true }] }]) {
      await expect(dispatchWspTool(programs, 'agentdb_wsp_create', { id: 'invalid', model: invalid })).rejects.toThrow('Invalid');
    }
    expect(programs.get('safe')!.proposals).toEqual([]);
  });
});
