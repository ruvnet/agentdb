import Ajv from 'ajv';
import type { NeuralReranker } from '../controllers/NeuralReranker.js';

const id = { type: 'string', minLength: 1, maxLength: 128, pattern: '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$' };
const integer = (minimum: number, maximum: number) => ({ type: 'integer', minimum, maximum });
const unit = { type: 'number', minimum: 0, maximum: 1 };
const features = { type: 'object', additionalProperties: false,
  required: ['similarity', 'reward', 'success', 'recency'],
  properties: { similarity: unit, reward: unit, success: unit, recency: unit } };
const candidate = { type: 'object', additionalProperties: false, required: ['id', 'features'], properties: { id, features } };
const labeled = { ...candidate, required: ['id', 'features', 'relevance'],
  properties: { id, features, relevance: integer(0, 3) } };
const groups = (minimum: number) => ({ type: 'array', minItems: minimum, maxItems: 128, items: {
  type: 'object', additionalProperties: false, required: ['queryId', 'candidates'], properties: {
    queryId: id, candidates: { type: 'array', minItems: 2, maxItems: 64, items: labeled }
  }
} });
// Same bounded WeightPatch interchange as the standalone WSP MCP tools. The
// controller additionally checks actual model indices, base hash and delta caps.
const weightPatch = {
  type: 'object', additionalProperties: false, required: ['schema', 'baseHash', 'edits'], properties: {
    schema: { const: 'ruv-fann.weight-patch/1' }, baseHash: { type: 'string', pattern: '^[a-f0-9]{64}$' },
    edits: { type: 'array', minItems: 1, maxItems: 128, items: {
      type: 'object', additionalProperties: false, required: ['layer', 'parameter', 'index', 'value'], properties: {
        layer: integer(0, 63), parameter: { type: 'string', enum: ['weights', 'bias'] }, index: integer(0, 65535),
        value: { type: 'number', minimum: -1_000_000, maximum: 1_000_000 }
      }
    } }
  }
};
const define = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  name: `agentdb_rerank_${name}`, description,
  inputSchema: { type: 'object' as const, additionalProperties: false, properties, required }
});
export const NEURAL_RERANK_QUERY_SCHEMA = { type: 'object', additionalProperties: false, required: ['id'], properties: {
  id, mode: { type: 'string', enum: ['shadow', 'active'] }, candidateId: id
}, allOf: [{ if: { required: ['mode'], properties: { mode: { const: 'active' } } }, then: { properties: { candidateId: false } } }] };
export const RERANK_TOOLS = [
  define('create', 'Create a learned reranker with baseline ordering and no active model. No automatic training or labels.', { id }, ['id']),
  define('get', 'Read reranker metadata, bounded checkpoints, candidates and internal ranking evaluations.', { id }, ['id']),
  define('list', 'List learned reranker metadata without model weights or labeled query groups.', { limit: integer(1, 100) }, []),
  define('inspect', 'Inspect a canonical seed, active or prospective candidate checkpoint and its WSP base fingerprint. Read-only.', { id, candidateId: id }, ['id']),
  define('patch', 'Propose bounded direct weight edits using a WSP base fingerprint. No gradient updates or provider calls; held-out evaluation and explicit admin promotion remain required.', { id, patch: weightPatch, baseCandidateId: id }, ['id', 'patch']),
  define('train', 'Train a separate real-WASM neural candidate using caller-labeled query groups. Active retrieval is unchanged.', {
    id, groups: groups(1), options: { type: 'object', additionalProperties: false, properties: {
      epochs: integer(1, 1000), learningRate: { type: 'number', exclusiveMinimum: 0, maximum: 1 }
    } }
  }, ['id', 'groups']),
  define('evaluate', 'Evaluate a candidate against baseline and active model on disjoint caller-labeled held-out query IDs. No generalization guarantee.', {
    id, candidateId: id, groups: groups(5), policy: { type: 'object', additionalProperties: false, properties: {
      k: integer(1, 64), minNdcgImprovement: { type: 'number', exclusiveMinimum: 0, maximum: 1 },
      maxMrrRegression: unit, maxPerQueryNdcgRegression: unit
    } }
  }, ['id', 'candidateId', 'groups']),
  define('promote', 'Admin: promote an internally accepted ranking evaluation using its ID and expected revision.', {
    id, evaluationId: id, expectedRevision: integer(1, Number.MAX_SAFE_INTEGER)
  }, ['id', 'evaluationId', 'expectedRevision']),
  define('rollback', 'Admin: restore a retained reranker checkpoint as a new monotonic revision.', {
    id, targetRevision: integer(1, Number.MAX_SAFE_INTEGER), expectedRevision: integer(1, Number.MAX_SAFE_INTEGER)
  }, ['id', 'targetRevision', 'expectedRevision']),
  define('rank', 'Score at most 64 candidates with a promoted model, or shadow-score a current candidate. Shadow returns the input order.', {
    id, candidates: { type: 'array', minItems: 0, maxItems: 64, items: candidate },
    options: { type: 'object', additionalProperties: false, properties: {
      mode: { type: 'string', enum: ['shadow', 'active'] }, candidateId: id
    }, allOf: NEURAL_RERANK_QUERY_SCHEMA.allOf }
  }, ['id', 'candidates'])
];
const ajv = new Ajv({ strict: true, strictNumbers: true, ownProperties: true });
const validators = new Map(RERANK_TOOLS.map(tool => [tool.name, ajv.compile(tool.inputSchema)]));
const queryValidator = ajv.compile(NEURAL_RERANK_QUERY_SCHEMA);
export function isRerankTool(name: string): boolean { return validators.has(name); }
export function validateNeuralRerankQuery(raw: unknown): void {
  if (!queryValidator(raw)) throw new Error(`Invalid neuralRerank arguments: ${ajv.errorsText(queryValidator.errors)}`);
}
export async function dispatchRerankTool(rerankers: NeuralReranker, name: string, raw: unknown): Promise<unknown> {
  const validate = validators.get(name);
  if (!validate) throw new Error('Unknown neural reranker tool');
  const args = raw === undefined ? {} : raw;
  if (!validate(args)) throw new Error(`Invalid neural reranker arguments: ${ajv.errorsText(validate.errors)}`);
  const value = args as Record<string, any>;
  // Schema bounds each list; enforce the aggregate row budget before dispatch.
  if (Array.isArray(value.groups) && value.groups.reduce((sum: number, group: any) => sum + group.candidates.length, 0) > 512)
    throw new Error('Invalid neural reranker arguments: total candidates exceed 512');
  switch (name) {
    case 'agentdb_rerank_create': return rerankers.create(value.id);
    case 'agentdb_rerank_get': return rerankers.get(value.id);
    case 'agentdb_rerank_list': return rerankers.list(value.limit);
    case 'agentdb_rerank_inspect': return rerankers.inspectWeights(value.id, value.candidateId);
    case 'agentdb_rerank_patch': return rerankers.proposePatch(value.id, value.patch, value.baseCandidateId);
    case 'agentdb_rerank_train': return rerankers.train(value.id, value.groups, value.options);
    case 'agentdb_rerank_evaluate': return rerankers.evaluate(value.id, value.candidateId, value.groups, value.policy);
    case 'agentdb_rerank_promote': return rerankers.promote(value.id, value.evaluationId, value.expectedRevision);
    case 'agentdb_rerank_rollback': return rerankers.rollback(value.id, value.targetRevision, value.expectedRevision);
    case 'agentdb_rerank_rank': return rerankers.rank(value.id, value.candidates, value.options);
    default: throw new Error('Unknown neural reranker tool');
  }
}
