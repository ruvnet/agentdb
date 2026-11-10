import Ajv from 'ajv';
import type { WeightPrograms } from '../controllers/WeightPrograms.js';

const id = { type: 'string', minLength: 1, maxLength: 128, pattern: '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$' };
const integer = (minimum: number, maximum: number) => ({ type: 'integer', minimum, maximum });
const numbers = { type: 'array', minItems: 1, maxItems: 65536, items: { type: 'number' } };
const checkpoint = {
  type: 'object', additionalProperties: false, required: ['schema', 'layers'], properties: {
    schema: { const: 'ruv-fann.dense-layers/1' },
    layers: { type: 'array', minItems: 1, maxItems: 64, items: {
      type: 'object', additionalProperties: false, required: ['inputs', 'outputs', 'weights', 'activation', 'steepness'], properties: {
        inputs: integer(1, 65536), outputs: integer(1, 65536), weights: numbers,
        bias: { anyOf: [{ type: 'null' }, { ...numbers, maxItems: 65536 }] },
        activation: { type: 'string', minLength: 1, maxLength: 64 }, steepness: { type: 'number' },
      },
    } },
  },
};
const patch = {
  type: 'object', additionalProperties: false, required: ['schema', 'baseHash', 'edits'], properties: {
    schema: { const: 'ruv-fann.weight-patch/1' }, baseHash: { type: 'string', pattern: '^[a-f0-9]{64}$' },
    edits: { type: 'array', minItems: 1, maxItems: 128, items: {
      type: 'object', additionalProperties: false, required: ['layer', 'parameter', 'index', 'value'], properties: {
        layer: integer(0, 63), parameter: { type: 'string', enum: ['weights', 'bias'] }, index: integer(0, 65535),
        value: { type: 'number', minimum: -1_000_000, maximum: 1_000_000 },
      },
    } },
  },
};
const dataset = {
  type: 'object', additionalProperties: false, required: ['inputs', 'targets', 'rows'],
  properties: { inputs: numbers, targets: numbers, rows: integer(1, 1024) },
};
const policy = {
  type: 'object', additionalProperties: false, required: ['maxMSE', 'maxRegressionMSE'], properties: {
    maxMSE: { type: 'number', minimum: 0 }, maxRegressionMSE: { type: 'number', minimum: 0 }, maxAbsoluteError: { type: 'number', minimum: 0 },
  },
};
const define = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  name: `agentdb_wsp_${name}`, description,
  inputSchema: { type: 'object' as const, additionalProperties: false, properties, required },
});
export const WSP_TOOLS = [
  define('create', 'Create a durable weight program from a dense checkpoint. Caller or LLM authors proposals; no provider or gradient training runs.', { id, model: checkpoint }, ['id', 'model']),
  define('get', 'Read a durable weight program, bounded checkpoint lineage and proposal evaluation evidence.', { id }, ['id']),
  define('list', 'List weight program metadata without model weights or proposal payloads.', { limit: integer(1, 100) }, []),
  define('propose', 'Evaluate a strict weight patch against caller-supplied data in real WASM and persist internal evidence. Acceptance does not prove generalization.', { id, patch, dataset, policy }, ['id', 'patch', 'dataset', 'policy']),
  define('promote', 'Promote an internally evaluated accepted proposal using its ID and expected program revision. Caller-authored receipts cannot authorize promotion.', { id, proposalId: id, expectedRevision: integer(1, Number.MAX_SAFE_INTEGER) }, ['id', 'proposalId', 'expectedRevision']),
  define('rollback', 'Restore a retained checkpoint as a new revision. Admin permission required.', { id, targetRevision: integer(1, Number.MAX_SAFE_INTEGER), expectedRevision: integer(1, Number.MAX_SAFE_INTEGER) }, ['id', 'targetRevision', 'expectedRevision']),
];
const ajv = new Ajv({ strict: true, strictNumbers: true, ownProperties: true });
const validators = new Map(WSP_TOOLS.map(tool => [tool.name, ajv.compile(tool.inputSchema)]));
export function isWspTool(name: string): boolean { return validators.has(name); }

export async function dispatchWspTool(programs: WeightPrograms, name: string, raw: unknown): Promise<unknown> {
  const validate = validators.get(name);
  if (!validate) throw new Error('Unknown weight program tool');
  const args = raw === undefined ? {} : raw;
  if (!validate(args)) throw new Error(`Invalid weight program arguments: ${ajv.errorsText(validate.errors)}`);
  const value = args as Record<string, any>;
  switch (name) {
    case 'agentdb_wsp_create': return programs.create(value.id, value.model);
    case 'agentdb_wsp_get': return programs.get(value.id);
    case 'agentdb_wsp_list': return programs.list(value.limit);
    case 'agentdb_wsp_propose': return programs.propose(value.id, value.patch, value.dataset, value.policy);
    case 'agentdb_wsp_promote': return programs.promote(value.id, value.proposalId, value.expectedRevision);
    case 'agentdb_wsp_rollback': return programs.rollback(value.id, value.targetRevision, value.expectedRevision);
    default: throw new Error('Unknown weight program tool');
  }
}
