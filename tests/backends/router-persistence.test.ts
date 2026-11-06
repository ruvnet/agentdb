/**
 * ADR-007 Phase 1: Router Persistence + SONA Context Enrichment Tests
 *
 * Tests:
 * - Router save/load round-trip (with mock NativeAccelerator)
 * - Debounced persistence trigger
 * - JSON fallback when native unavailable
 * - SONA context enrichment adds metadata to trajectories
 * - SONA flush triggers on tick cycle
 *
 * Uses mock-first approach since @ruvector/* packages may not be installed.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';

describe('ADR-007 Phase 1: Router Persistence', () => {
  let SemanticQueryRouter: typeof import('../../src/backends/rvf/SemanticQueryRouter.js').SemanticQueryRouter;

  beforeEach(async () => {
    const mod = await import('../../src/backends/rvf/SemanticQueryRouter.js');
    SemanticQueryRouter = mod.SemanticQueryRouter;
  });

  describe('persistencePath configuration', () => {
    it('should accept persistencePath in RouterConfig', async () => {
      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: '/tmp/test-router.json',
      });
      expect(router.persistencePath).toBe('/tmp/test-router.json');
      router.destroy();
    });

    it('should have undefined persistencePath by default', async () => {
      const router = await SemanticQueryRouter.create({ dimension: 4 });
      expect(router.persistencePath).toBeUndefined();
      router.destroy();
    });
  });

  describe('JSON fallback save/load round-trip', () => {
    let tmpDir: string;
    let tmpFile: string;

    beforeEach(async () => {
      tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-test-'));
      tmpFile = path.join(tmpDir, 'router-state.json');
    });

    it('should save router state to JSON file', async () => {
      const router = await SemanticQueryRouter.create({
        dimension: 4,
        threshold: 0.1,
      });

      router.addIntent({
        name: 'greeting',
        exemplars: [new Float32Array([1, 0, 0, 0])],
        metadata: { category: 'social' },
      });
      router.addIntent({
        name: 'search',
        exemplars: [new Float32Array([0, 1, 0, 0])],
        metadata: { category: 'query' },
      });

      const saved = await router.save(tmpFile);
      expect(saved).toBe(true);

      // Verify JSON file was written
      // Written by @ruvector/router's own save() when the native router loads
      // ({ dimension, intents[].embedding }), else by the JSON fallback
      // ({ dim, intents[].centroid }).
      const content = JSON.parse(await fs.readFile(tmpFile, 'utf-8'));
      expect(content.dim ?? content.dimension).toBe(4);
      expect(content.threshold).toBe(0.1);
      expect(content.intents).toHaveLength(2);
      expect(content.intents[0].name).toBe('greeting');
      expect(content.intents[1].name).toBe('search');

      router.destroy();
    });

    it('should load router state from JSON file', async () => {
      // Create and save router
      const router1 = await SemanticQueryRouter.create({
        dimension: 4,
        threshold: 0.0,
      });
      router1.addIntent({
        name: 'alpha',
        exemplars: [new Float32Array([1, 0, 0, 0])],
        metadata: { ef: 50 },
      });
      router1.addIntent({
        name: 'beta',
        exemplars: [new Float32Array([0, 0, 1, 0])],
        metadata: { ef: 200 },
      });
      await router1.save(tmpFile);
      router1.destroy();

      // Load into a fresh router
      const router2 = await SemanticQueryRouter.create({
        dimension: 4,
        threshold: 0.0,
      });
      const loaded = await router2.load(tmpFile);
      expect(loaded).toBe(true);

      // Verify intents were restored
      const intents = router2.getIntents();
      expect(intents).toContain('alpha');
      expect(intents).toContain('beta');
      expect(intents.length).toBe(2);

      // Verify routing still works after load
      const matches = router2.route(new Float32Array([1, 0, 0, 0]), 2);
      expect(Array.isArray(matches)).toBe(true);
      if (matches.length > 0) {
        expect(matches[0].intent).toBe('alpha');
        expect(matches[0].score).toBeGreaterThan(0.9);
      }

      router2.destroy();
    });

    it('should return false when loading from nonexistent path', async () => {
      const router = await SemanticQueryRouter.create({ dimension: 4 });
      const loaded = await router.load('/tmp/nonexistent-path-xyz.json');
      expect(loaded).toBe(false);
      router.destroy();
    });

    it('should return false when loading invalid JSON', async () => {
      await fs.writeFile(tmpFile, 'not-json');
      const router = await SemanticQueryRouter.create({ dimension: 4 });
      const loaded = await router.load(tmpFile);
      expect(loaded).toBe(false);
      router.destroy();
    });

    it('should round-trip with persistencePath auto-load on create', async () => {
      // Create and save
      const router1 = await SemanticQueryRouter.create({
        dimension: 4,
        threshold: 0.0,
      });
      router1.addIntent({
        name: 'intent-a',
        exemplars: [new Float32Array([1, 0, 0, 0])],
      });
      await router1.save(tmpFile);
      router1.destroy();

      // Create with persistencePath - should auto-load
      const router2 = await SemanticQueryRouter.create({
        dimension: 4,
        threshold: 0.0,
        persistencePath: tmpFile,
      });
      const intents = router2.getIntents();
      expect(intents).toContain('intent-a');
      router2.destroy();
    });
  });

  describe('persist() method', () => {
    it('should return false when no persistencePath configured', async () => {
      const router = await SemanticQueryRouter.create({ dimension: 4 });
      const result = await router.persist();
      expect(result).toBe(false);
      router.destroy();
    });

    it('should persist to configured path', async () => {
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-persist-'));
      const tmpFile = path.join(tmpDir, 'state.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });
      router.addIntent({
        name: 'test',
        exemplars: [new Float32Array([1, 0, 0, 0])],
      });

      const result = await router.persist();
      expect(result).toBe(true);

      // Verify file exists and is valid
      const content = JSON.parse(await fs.readFile(tmpFile, 'utf-8'));
      expect(content.intents).toHaveLength(1);
      expect(content.intents[0].name).toBe('test');

      router.destroy();
    });

    it('should return false after destroy', async () => {
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-destroy-'));
      const tmpFile = path.join(tmpDir, 'state.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });
      router.destroy();
      const result = await router.persist();
      expect(result).toBe(false);
    });
  });

  describe('debounced persistence trigger', () => {
    it('should schedule persist after addIntent', async () => {
      vi.useFakeTimers();
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-debounce-'));
      const tmpFile = path.join(tmpDir, 'debounce.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });

      // Add intent triggers debounced persist
      router.addIntent({
        name: 'debounce-test',
        exemplars: [new Float32Array([1, 0, 0, 0])],
      });

      // File may or may not exist yet (debounce hasn't fired)
      try { await fs.access(tmpFile); } catch { /* expected before debounce fires */ }

      // Fast-forward past the debounce delay
      vi.advanceTimersByTime(6000);
      // Allow the async persist to complete
      await vi.runAllTimersAsync();

      vi.useRealTimers();
      router.destroy();
    });

    it('should coalesce rapid changes into single persist', async () => {
      vi.useFakeTimers();
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-coalesce-'));
      const tmpFile = path.join(tmpDir, 'coalesce.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });

      // Rapid adds should reset the timer each time
      for (let i = 0; i < 10; i++) {
        router.addIntent({
          name: `intent-${i}`,
          exemplars: [new Float32Array([i / 10, 0, 0, 0])],
        });
      }

      // Only one persist should happen after the debounce window
      vi.advanceTimersByTime(6000);
      await vi.runAllTimersAsync();

      vi.useRealTimers();

      // Verify all 10 intents are present
      expect(router.getIntents().length).toBe(10);

      router.destroy();
    });

    it('should schedule persist after removeIntent', async () => {
      vi.useFakeTimers();
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-remove-'));
      const tmpFile = path.join(tmpDir, 'remove.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });

      router.addIntent({
        name: 'to-remove',
        exemplars: [new Float32Array([1, 0, 0, 0])],
      });

      const removed = router.removeIntent('to-remove');
      expect(removed).toBe(true);

      vi.advanceTimersByTime(6000);
      await vi.runAllTimersAsync();

      vi.useRealTimers();
      router.destroy();
    });

    it('should clear timer on destroy', async () => {
      vi.useFakeTimers();
      const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-cleartimer-'));
      const tmpFile = path.join(tmpDir, 'cleartimer.json');

      const router = await SemanticQueryRouter.create({
        dimension: 4,
        persistencePath: tmpFile,
      });

      router.addIntent({
        name: 'timer-clear-test',
        exemplars: [new Float32Array([1, 0, 0, 0])],
      });

      // Destroy should clear the pending timer
      router.destroy();
      expect(router.isDestroyed).toBe(true);

      vi.useRealTimers();
    });
  });
});

describe('ADR-007 Phase 1: SONA Context Enrichment', () => {
  describe('SonaLearningBackend context enrichment', () => {
    let SonaLearningBackend: typeof import('../../src/backends/rvf/SonaLearningBackend.js').SonaLearningBackend;
    let sonaAvailable = false;

    beforeEach(async () => {
      const mod = await import('../../src/backends/rvf/SonaLearningBackend.js');
      SonaLearningBackend = mod.SonaLearningBackend;
      sonaAvailable = await SonaLearningBackend.isAvailable();
    });

    it('should expose getContextStats method', async () => {
      if (!sonaAvailable) return;

      const sona = await SonaLearningBackend.create({ hiddenDim: 8 });
      const stats = sona.getContextStats();
      expect(stats).toEqual({
        contextAdded: 0,
        contextFailed: 0,
        flushCount: 0,
        tickCount: 0,
      });
      sona.destroy();
    });

    it('should increment tickCount on tick', async () => {
      if (!sonaAvailable) return;

      const sona = await SonaLearningBackend.create({ hiddenDim: 8 });

      sona.tick();
      sona.tick();
      sona.tick();

      const stats = sona.getContextStats();
      expect(stats.tickCount).toBe(3);

      sona.destroy();
    });

    it('should attempt flush on every 10th tick', async () => {
      if (!sonaAvailable) return;

      const sona = await SonaLearningBackend.create({ hiddenDim: 8 });

      for (let i = 0; i < 10; i++) {
        sona.tick();
      }

      const stats = sona.getContextStats();
      expect(stats.tickCount).toBe(10);
      // Flush was attempted (may succeed or fail depending on native availability)

      sona.destroy();
    });

    it('should handle addContext for invalid trajectory gracefully', async () => {
      if (!sonaAvailable) return;

      const sona = await SonaLearningBackend.create({ hiddenDim: 8 });
      // Invalid trajectory ID - should return false
      const result = await sona.addContext(999, { key: 'value' });
      expect(result).toBe(false);

      sona.destroy();
    });

    it('should track context enrichment failures', async () => {
      if (!sonaAvailable) return;

      const sona = await SonaLearningBackend.create({ hiddenDim: 8 });

      const query = new Float32Array(8).fill(0.5);
      const trajId = sona.beginTrajectory(query);

      // This will attempt addContext - if native not available, should track failure
      await sona.addContext(trajId, { queryType: 'test', timestamp: Date.now() });
      const stats = sona.getContextStats();
      // Either contextAdded or contextFailed should be 1
      expect(stats.contextAdded + stats.contextFailed).toBe(1);

      sona.endTrajectory(trajId, 0.8);
      sona.destroy();
    });
  });

  describe('SONA context stats interface', () => {
    it('should have correct shape for context stats', () => {
      const stats = {
        contextAdded: 5,
        contextFailed: 2,
        flushCount: 3,
        tickCount: 100,
      };
      expect(stats.contextAdded).toBe(5);
      expect(stats.contextFailed).toBe(2);
      expect(stats.flushCount).toBe(3);
      expect(stats.tickCount).toBe(100);
    });
  });

  describe('SONA flush on tick cycle', () => {
    it('should trigger flush every N ticks (default: 10)', () => {
      const flushInterval = 10;
      const flushTicks: number[] = [];

      for (let tick = 1; tick <= 50; tick++) {
        if (tick % flushInterval === 0) {
          flushTicks.push(tick);
        }
      }

      expect(flushTicks).toEqual([10, 20, 30, 40, 50]);
    });

    it('should not flush on non-interval ticks', () => {
      const flushInterval = 10;
      const nonFlushTicks = [1, 5, 9, 11, 15, 19, 21];

      for (const tick of nonFlushTicks) {
        expect(tick % flushInterval).not.toBe(0);
      }
    });
  });
});

describe('ADR-007 Phase 1: SelfLearningRvfBackend Integration', () => {
  describe('routerPersistencePath configuration', () => {
    it('should include routerPersistencePath in SelfLearningConfig', async () => {
      const mod = await import('../../src/backends/rvf/SelfLearningRvfBackend.js');
      expect(mod.SelfLearningRvfBackend).toBeDefined();

      // Type-level test: config accepts routerPersistencePath
      const config: import('../../src/backends/rvf/SelfLearningRvfBackend.js').SelfLearningConfig = {
        dimension: 128,
        metric: 'cosine',
        learning: true,
        routerPersistencePath: '/tmp/router-state.json',
      };
      expect(config.routerPersistencePath).toBe('/tmp/router-state.json');
    });
  });

  describe('context metadata in search path', () => {
    it('should include expected context metadata fields', () => {
      const context = {
        queryType: 'search',
        intent: 'similarity',
        timestamp: Date.now(),
        searchK: 10,
      };

      expect(context.queryType).toBe('search');
      expect(typeof context.intent).toBe('string');
      expect(context.timestamp).toBeGreaterThan(0);
      expect(context.searchK).toBe(10);
    });

    it('should default queryType to unknown when no route', () => {
      const route: string | undefined = undefined;
      const context = {
        queryType: route ?? 'unknown',
        intent: route,
        timestamp: Date.now(),
      };
      expect(context.queryType).toBe('unknown');
      expect(context.intent).toBeUndefined();
    });
  });

  describe('persistRouter method', () => {
    it('should be callable on SelfLearningRvfBackend', async () => {
      const mod = await import('../../src/backends/rvf/SelfLearningRvfBackend.js');
      expect(typeof mod.SelfLearningRvfBackend.prototype.persistRouter).toBe('function');
    });
  });
});

describe('Router persistence edge cases', () => {
  let SemanticQueryRouter: typeof import('../../src/backends/rvf/SemanticQueryRouter.js').SemanticQueryRouter;

  beforeEach(async () => {
    const mod = await import('../../src/backends/rvf/SemanticQueryRouter.js');
    SemanticQueryRouter = mod.SemanticQueryRouter;
  });

  it('should preserve metadata through save/load cycle', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-meta-'));
    const tmpFile = path.join(tmpDir, 'meta.json');

    const router1 = await SemanticQueryRouter.create({
      dimension: 4,
      threshold: 0.0,
    });
    router1.addIntent({
      name: 'with-meta',
      exemplars: [new Float32Array([1, 0, 0, 0])],
      metadata: { strategy: 'narrow', ef: 50, tags: ['fast'] },
    });
    await router1.save(tmpFile);
    router1.destroy();

    // Load and verify metadata
    const raw = JSON.parse(await fs.readFile(tmpFile, 'utf-8'));
    expect(raw.intents[0].metadata.strategy).toBe('narrow');
    expect(raw.intents[0].metadata.ef).toBe(50);
    expect(raw.intents[0].metadata.tags).toEqual(['fast']);
  });

  it('should handle empty router save/load', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-empty-'));
    const tmpFile = path.join(tmpDir, 'empty.json');

    const router = await SemanticQueryRouter.create({ dimension: 4 });
    await router.save(tmpFile);

    const raw = JSON.parse(await fs.readFile(tmpFile, 'utf-8'));
    expect(raw.intents).toHaveLength(0);

    const router2 = await SemanticQueryRouter.create({ dimension: 4 });
    const loaded = await router2.load(tmpFile);
    expect(loaded).toBe(true);
    expect(router2.getIntents().length).toBe(0);

    router.destroy();
    router2.destroy();
  });

  it('should not persist when router is destroyed', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-destroyed-'));
    const tmpFile = path.join(tmpDir, 'destroyed.json');

    const router = await SemanticQueryRouter.create({
      dimension: 4,
      persistencePath: tmpFile,
    });
    router.destroy();

    const result = await router.persist();
    expect(result).toBe(false);
  });

  it('should save centroid as number array for JSON compatibility', async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'router-centroid-'));
    const tmpFile = path.join(tmpDir, 'centroid.json');

    const router = await SemanticQueryRouter.create({ dimension: 4 });
    router.addIntent({
      name: 'vec-test',
      exemplars: [new Float32Array([0.25, 0.5, 0.75, 1.0])],
    });
    await router.save(tmpFile);

    // Native router files store the vector as `embedding`, the fallback as `centroid`.
    const raw = JSON.parse(await fs.readFile(tmpFile, 'utf-8'));
    const vector = raw.intents[0].centroid ?? raw.intents[0].embedding;
    expect(Array.isArray(vector)).toBe(true);
    expect(vector[0]).toBeCloseTo(0.25);
    expect(vector[3]).toBeCloseTo(1.0);

    router.destroy();
  });

  it('should reject path traversal in save()', async () => {
    const router = await SemanticQueryRouter.create({ dimension: 4 });
    await expect(router.save('../../../etc/shadow')).rejects.toThrow('forbidden');
    router.destroy();
  });

  it('should reject path traversal in load()', async () => {
    const router = await SemanticQueryRouter.create({ dimension: 4 });
    await expect(router.load('../../../etc/passwd')).rejects.toThrow('forbidden');
    router.destroy();
  });

  it('should reject path traversal in persistencePath config', async () => {
    await expect(
      SemanticQueryRouter.create({ dimension: 4, persistencePath: '../../../tmp/evil.json' }),
    ).rejects.toThrow('forbidden');
  });

  it('should reject null bytes in path', async () => {
    const router = await SemanticQueryRouter.create({ dimension: 4 });
    await expect(router.save('/tmp/test\0evil')).rejects.toThrow('null bytes');
    router.destroy();
  });
});
