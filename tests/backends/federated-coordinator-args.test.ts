/**
 * FederatedSessionManager constructed @ruvector/ruvllm's FederatedCoordinator
 * as `new FederatedCoordinator({ dimension, maxAgents })`, but the signature
 * is `(coordinatorId, config)`. The config object became the id and the
 * coordinator kept ruvllm's default hiddenDim/embeddingDim of 256.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { FederatedSessionManager } from '../../src/backends/rvf/FederatedSessionManager.js';

let manager: FederatedSessionManager | null = null;

afterEach(() => {
  if (manager && !manager.isDestroyed) manager.destroy();
  manager = null;
});

describe('FederatedSessionManager coordinator construction', () => {
  it('passes a string id and hiddenDim = dimension', async () => {
    if (!(await FederatedSessionManager.isAvailable())) return;

    manager = await FederatedSessionManager.create({ dimension: 32 });
    const coordinator = (manager as any).coordinator;

    expect(typeof coordinator.getCoordinatorId()).toBe('string');
    expect(coordinator.config.hiddenDim).toBe(32);
    expect(coordinator.config.embeddingDim).toBe(32);
  });

  it('enforces maxAgents concurrent sessions', async () => {
    if (!(await FederatedSessionManager.isAvailable())) return;

    manager = await FederatedSessionManager.create({ dimension: 8, maxAgents: 1 });
    await manager.beginSession('a');
    await expect(manager.beginSession('b')).rejects.toThrow(/Maximum concurrent sessions/);
  });
});
