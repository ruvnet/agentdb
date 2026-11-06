/**
 * @ruvector/rvf is optional. When it cannot be imported, createBackend('rvf')
 * falls back to the built-in sql.js RVF backend, and RvfBackend itself fails
 * with an install hint instead of a raw module error.
 */
import { describe, it, expect, vi } from 'vitest';

vi.mock('@ruvector/rvf', () => {
  throw new Error("Cannot find package '@ruvector/rvf'");
});

import { createBackend } from '../../src/backends/factory.js';
import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';

describe('@ruvector/rvf absent', () => {
  it("createBackend('rvf') falls back to the sql.js RVF backend", async () => {
    const backend = await createBackend('rvf', { dimension: 4, metric: 'cosine' });
    expect(backend.constructor.name).toBe('SqlJsRvfBackend');
    backend.close();
  });

  it('RvfBackend.initialize() explains how to install rvf', async () => {
    const b = new RvfBackend({ dimension: 4, storagePath: ':memory:' });
    await expect(b.initialize()).rejects.toThrow(/npm install @ruvector\/rvf/);
  });
});
