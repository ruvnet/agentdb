/**
 * @ruvector/ruvllm API contract.
 *
 * agentdb declares `@ruvector/ruvllm` as an optional dependency, and agentdb is
 * itself a dependency of agentic-flow and ruflo, so this range decides which
 * ruvllm the ecosystem installs. 4.0.0/4.0.1 were published from the wrong
 * repository; 4.1.0 is the restored runtime.
 *
 * The package is required directly: if it is missing this file fails instead
 * of passing on the "not installed" branch. Every symbol agentdb calls is
 * asserted on the real package, then exercised through agentdb's own wrappers.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { createRequire } from 'module';
import { readFileSync } from 'fs';
import * as path from 'path';
import { LLMRouter } from '../../../src/services/LLMRouter.js';
import { FederatedSessionManager } from '../../../src/backends/rvf/FederatedSessionManager.js';
import { NativeAccelerator } from '../../../src/backends/rvf/NativeAccelerator.js';

const require = createRequire(import.meta.url);
const ruvllm = require('@ruvector/ruvllm');
const installed: string = require('@ruvector/ruvllm/package.json').version;
const declared: string = JSON.parse(
  readFileSync(path.join(__dirname, '../../../package.json'), 'utf8'),
).optionalDependencies['@ruvector/ruvllm'];

/** Methods agentdb calls, per class (src/services, src/backends/rvf). */
const USED: Record<string, string[]> = {
  RuvLLM: [
    'query', 'generate', 'feedback', 'embed', 'similarity',
    'addMemory', 'searchMemory', 'stats', 'isModelLoaded',
  ],
  FederatedCoordinator: [
    'setQualityThreshold', 'setConsolidationInterval', 'createAgent', 'aggregate',
    'getInitialPatterns', 'findPatterns', 'shouldConsolidate', 'forceConsolidate',
    'stats', 'getContributions', 'clear',
  ],
  EphemeralAgent: [
    'processTask', 'processTaskWithRoute', 'forceLearn', 'getPatterns',
    'trajectoryCount', 'avgQuality', 'uptimeSeconds',
  ],
  LoraManager: ['create', 'activate', 'forward', 'list', 'clear'],
  SimdOps: [
    'cosineSimilarity', 'dotProduct', 'l2Distance', 'matvec', 'softmax', 'relu',
    'gelu', 'sigmoid', 'layerNorm', 'add', 'mul', 'scale', 'normalize',
  ],
  EwcManager: ['registerTask', 'computePenalty'],
};

function parse(v: string): [number, number, number] {
  const m = /^(\d+)\.(\d+)\.(\d+)/.exec(v);
  if (!m) throw new Error(`unparseable version ${v}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

describe('@ruvector/ruvllm version', () => {
  it('declares ^4.1.0', () => {
    expect(declared).toBe('^4.1.0');
  });

  it('installed version satisfies ^4.1.0 (never the 4.0.x wrong-repo publish)', () => {
    const [major, minor] = parse(installed);
    expect(major).toBe(4);
    expect(minor).toBeGreaterThanOrEqual(1);
  });
});

describe('@ruvector/ruvllm symbols used by agentdb', () => {
  for (const [cls, methods] of Object.entries(USED)) {
    it(`${cls} exposes ${methods.length} methods`, () => {
      expect(typeof ruvllm[cls]).toBe('function');
      const missing = methods.filter((m) => typeof ruvllm[cls].prototype[m] !== 'function');
      expect(missing).toEqual([]);
    });
  }
});

describe('@ruvector/ruvllm behaviour agentdb depends on', () => {
  it('RuvLLM embeds, compares, stores and searches without a model', () => {
    const engine = new ruvllm.RuvLLM({ embeddingDim: 768 });
    expect(engine.embed('hello world')).toHaveLength(768);
    expect(typeof engine.similarity('a b', 'a c')).toBe('number');
    const id = engine.addMemory('agentdb memory', { source: 'test' });
    expect(typeof id).toBe('string');
    const hits = engine.searchMemory('agentdb memory', 1);
    expect(hits[0].content).toBe('agentdb memory');
    expect(engine.feedback({ requestId: 'r-1', rating: 4 })).toBe(true);
    expect(typeof engine.stats().totalQueries).toBe('number');
  });

  it('RuvLLM refuses text generation when no language model is loaded', () => {
    const engine = new ruvllm.RuvLLM({ embeddingDim: 64 });
    if (engine.isModelLoaded()) return; // a model-capable native binary is present
    expect(() => engine.query('hi', { maxTokens: 4 })).toThrow(/RUVLLM_NO_LANGUAGE_MODEL/);
  });

  it('FederatedCoordinator aggregates an agent into reusable patterns', () => {
    const coordinator = new ruvllm.FederatedCoordinator('agentdb-test', { hiddenDim: 16, embeddingDim: 16 });
    const agent = coordinator.createAgent('a1');
    const v = Array.from({ length: 16 }, (_, i) => Math.sin(i + 1));
    agent.processTask(v, 0.9);
    agent.processTaskWithRoute(v, 'route-a', 0.8);
    agent.forceLearn();
    coordinator.aggregate(agent);
    expect(agent.trajectoryCount()).toBe(2);
    expect(coordinator.getInitialPatterns(5).length).toBeGreaterThan(0);
    expect(coordinator.stats().totalAgents).toBe(1);
  });

  it('LoraManager takes dimensions as positional arguments', () => {
    const lora = new ruvllm.LoraManager();
    lora.create('d', { rank: 2, alpha: 4 }, 16, 16);
    lora.activate('d');
    expect(lora.forward(new Array(16).fill(0.1))).toHaveLength(16);
    expect(lora.list()).toEqual(['d']);
  });

  it('SimdOps and EwcManager compute', () => {
    const simd = new ruvllm.SimdOps();
    expect(simd.cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1);
    expect(simd.dotProduct([1, 2], [3, 4])).toBe(11);
    expect(simd.l2Distance([0, 0], [3, 4])).toBeCloseTo(5);
    const ewc = new ruvllm.EwcManager();
    ewc.registerTask('t', [1, 2, 3]);
    expect(ewc.computePenalty([2, 3, 4])).toBeGreaterThan(0);
  });
});

describe('agentdb wrappers on @ruvector/ruvllm', () => {
  let manager: FederatedSessionManager | null = null;
  afterEach(() => {
    if (manager && !manager.isDestroyed) manager.destroy();
    manager = null;
  });

  it('LLMRouter keeps ruvllm for embeddings but only generates text with a loaded model', async () => {
    const router = new LLMRouter({ provider: 'onnx' });
    await router.initialize();
    expect(router.isRuvLLMAvailable()).toBe(true);
    const embedding = await router.getEmbedding('hello');
    expect(embedding).toBeInstanceOf(Float32Array);
    expect(embedding!.length).toBe(768);

    const canGenerate = router.isProviderAvailable('ruvllm');
    expect(canGenerate).toBe(new ruvllm.RuvLLM({ embeddingDim: 8 }).isModelLoaded());
    if (!canGenerate) {
      expect(router.getAvailableProviders()).not.toContain('ruvllm');
      expect(router.optimizeModelSelection('t', 'privacy').provider).toBe('onnx');
      const res = await router.generate('summarize this strategy', { provider: 'ruvllm' });
      expect(res.content.length).toBeGreaterThan(0);
    }
  });

  it('FederatedSessionManager LoRA keeps the embedding dimension', async () => {
    manager = await FederatedSessionManager.create({ dimension: 32 });
    const out = manager.applyLora(new Array(32).fill(0.5));
    expect(out).toHaveLength(32);
    manager.createAdapter('task-a', 2);
    manager.activateAdapter('task-a');
    expect(manager.applyLora(new Array(32).fill(0.5))).toHaveLength(32);
  });

  it('NativeAccelerator EWC registers weights and penalises drift', async () => {
    const accel = new NativeAccelerator();
    await accel.initialize();
    expect(accel.ewcManagerAvailable).toBe(true);
    expect(accel.ewcUpdateFisher(Float32Array.from([1, 2, 3, 4]), 1.0)).toBe(true);
    expect(accel.ewcPenalty(Float32Array.from([2, 3, 4, 5]))).toBeGreaterThan(0);
  });
});
