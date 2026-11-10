import type { FannModels } from '../controllers/FannModels.js';

const id = { type: 'string', minLength: 1, maxLength: 128, pattern: '^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$' };
const numbers = { type: 'array', minItems: 1, maxItems: 524288, items: { type: 'number' } };
const integer = (maximum: number) => ({ type: 'integer', minimum: 1, maximum });
const checkpoint = {
  type: 'object', additionalProperties: false, required: ['schema', 'layers'],
  properties: {
    schema: { const: 'ruv-fann.dense-layers/1' },
    layers: { type: 'array', minItems: 1, maxItems: 64, items: {
      type: 'object', additionalProperties: false,
      required: ['inputs', 'outputs', 'weights', 'activation', 'steepness'],
      properties: { inputs: integer(4096), outputs: integer(4096),
        weights: { ...numbers, maxItems: 262144 },
        bias: { anyOf: [{ type: 'null' }, { ...numbers, maxItems: 4096 }] },
        activation: { type: 'string', maxLength: 64, description: 'Case-sensitive FANN activation, e.g. Linear, Sigmoid, Tanh or ReLU.' },
        steepness: { type: 'number', exclusiveMinimum: 0 },
      },
    } },
  },
};
const define = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  name: `agentdb_neural_${name}`, description,
  inputSchema: { type: 'object' as const, properties, required, additionalProperties: false },
});

export const FANN_TOOLS = [
  define('create', 'Persist a dense ruv-FANN model checkpoint. Existing model IDs are rejected.', {
    id, model: checkpoint,
  }, ['id', 'model']),
  define('get', 'Read a persisted neural model and its revision.', { id }, ['id']),
  define('list', 'List neural model metadata without checkpoint weights.', { limit: integer(100) }, []),
  define('predict', 'Run batched inference using a cached prepared WASM model. Inputs are flat row-major numbers.', {
    id, inputs: numbers, rows: integer(128),
  }, ['id', 'inputs']),
  define('train', 'Train a persisted neural model with bounded MSE optimization; commit a new revision only on success.', {
    id, inputs: numbers, targets: numbers, rows: integer(128), epochs: integer(1000),
    learningRate: { type: 'number', exclusiveMinimum: 0, maximum: 1 }, optimizer: { type: 'string', enum: ['sgd', 'adamw'] },
  }, ['id', 'inputs', 'targets', 'rows', 'epochs', 'learningRate']),
  define('delete', 'Delete a persisted neural model and release its cached workspace. Admin permission required.', { id }, ['id']),
];

export function isFannTool(name: string): boolean { return FANN_TOOLS.some(tool => tool.name === name); }

export async function dispatchFannTool(models: FannModels, name: string, raw: unknown): Promise<unknown> {
  const tool = FANN_TOOLS.find(candidate => candidate.name === name);
  if (!tool) throw new Error('Unknown neural tool');
  if (raw !== undefined && (raw === null || typeof raw !== 'object' || Array.isArray(raw))) throw new Error('Arguments must be an object');
  const args = (raw ?? {}) as Record<string, unknown>;
  for (const key of Object.keys(args)) if (!(Object.hasOwn(tool.inputSchema.properties, key))) throw new Error(`Unknown argument: ${key}`);
  for (const key of tool.inputSchema.required) if (args[key] === undefined) throw new Error(`Missing argument: ${key}`);
  if ('id' in args && (typeof args.id !== 'string' || args.id.length < 1 || args.id.length > 128)) throw new Error('id must contain 1–128 characters');
  const bounded = (key: string, max: number): number | undefined => {
    const value = args[key];
    if (value === undefined) return undefined;
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1 || value > max) throw new Error(`${key} must be an integer from 1 to ${max}`);
    return value;
  };
  const vector = (key: string): Float64Array => {
    const value = args[key];
    if (!Array.isArray(value) || value.length < 1 || value.length > 524288 || !value.every(x => typeof x === 'number' && Number.isFinite(x))) throw new Error(`${key} must be a finite numeric array with 1–524288 entries`);
    return Float64Array.from(value);
  };
  const modelId = args.id as string;
  switch (name) {
    case 'agentdb_neural_create':
      if (!args.model || typeof args.model !== 'object' || Array.isArray(args.model)) throw new Error('model must be a dense checkpoint object');
      for (const key of Object.keys(args.model)) if (!['schema', 'layers'].includes(key)) throw new Error(`Unknown model field: ${key}`);
      const layers = (args.model as Record<string, unknown>).layers;
      if (Array.isArray(layers)) for (const layer of layers) {
        if (layer && typeof layer === 'object') for (const key of Object.keys(layer)) {
          if (!['inputs', 'outputs', 'weights', 'bias', 'activation', 'steepness'].includes(key)) throw new Error(`Unknown layer field: ${key}`);
        }
      }
      return models.create(modelId, JSON.stringify(args.model));
    case 'agentdb_neural_get': return models.get(modelId);
    case 'agentdb_neural_list': return models.list(bounded('limit', 100));
    case 'agentdb_neural_predict': return models.predict(modelId, vector('inputs'), bounded('rows', 128));
    case 'agentdb_neural_train': {
      if (typeof args.learningRate !== 'number' || !Number.isFinite(args.learningRate) || args.learningRate <= 0) throw new Error('learningRate must be finite and positive');
      if (args.optimizer !== undefined && args.optimizer !== 'sgd' && args.optimizer !== 'adamw') throw new Error('optimizer must be sgd or adamw');
      return models.train(modelId, { inputs: vector('inputs'), targets: vector('targets'), rows: bounded('rows', 128)!, epochs: bounded('epochs', 1000)!, learningRate: args.learningRate, optimizer: args.optimizer });
    }
    case 'agentdb_neural_delete': return { id: modelId, deleted: models.delete(modelId) };
    default: throw new Error('Unknown neural tool');
  }
}
