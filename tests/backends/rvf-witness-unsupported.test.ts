/**
 * RvfBackend.verifyWitness()/freeze() against the installed @ruvector/rvf.
 *
 * @ruvector/rvf 0.2.x has no verifyWitness, indexStats, metric or freeze.
 * verifyWitness() used to report "this.db.verifyWitness is not a function"
 * and freeze() threw a TypeError. Both must now say "unsupported" explicitly,
 * and witness verification must never report valid without the store's word.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';

describe('RvfBackend witness and freeze on an rvf build without them', () => {
  let dir: string;
  let backend: RvfBackend;

  beforeEach(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-rvf-witness-'));
    backend = new RvfBackend({ dimension: 4, metric: 'cosine', storagePath: path.join(dir, 'store.rvf') } as any);
    await backend.initialize();
  });

  afterEach(() => {
    backend.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('verifyWitness() fails closed with an explicit unsupported result', async () => {
    const native = (backend as any).db;
    if (typeof native.verifyWitness === 'function') {
      return; // a newer @ruvector/rvf is installed; covered by the mock test below
    }

    for (const result of [backend.verifyWitness(), await backend.verifyWitnessAsync()]) {
      expect(result.valid).toBe(false);
      expect(result.unsupported).toBe(true);
      expect(result.error).toMatch(/^unsupported:/);
      expect(result.error).not.toMatch(/is not a function/);
    }
  });

  it('freeze() rejects with an unsupported error', async () => {
    const native = (backend as any).db;
    if (typeof native.freeze === 'function') return;

    await expect(Promise.resolve().then(() => backend.freeze())).rejects.toThrow(/^unsupported:/);
  });

  it('only a literal valid: true counts, and an async verifier is not trusted synchronously', async () => {
    const native = (backend as any).db;

    native.verifyWitness = () => ({ valid: 'yes', entries: 3 });
    expect(backend.verifyWitness().valid).toBe(false);

    native.verifyWitness = async () => ({ valid: true, entries: 3 });
    const syncResult = backend.verifyWitness();
    expect(syncResult.valid).toBe(false);
    expect(syncResult.unsupported).toBe(true);
    expect(await backend.verifyWitnessAsync()).toEqual({ valid: true, entries: 3, error: undefined });
  });
});
