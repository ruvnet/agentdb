import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';

vi.mock('../../src/db-fallback.js', () => ({ createDatabase: vi.fn() }));
vi.mock('../../src/backends/detector.js', () => ({ detectBackend: vi.fn() }));
import { isSupportedNodeVersion } from '../../src/cli/commands/doctor.js';

describe('Node runtime required by patched sharp and better-sqlite3 12', () => {
  it.each(['v18.20.8', 'v20.0.0', 'v20.8.1', 'invalid', 'v22.invalid.0'])('refuses unsupported runtime %s', version => {
    expect(isSupportedNodeVersion(version)).toBe(false);
  });
  it.each(['v20.9.0', '20.9.1', 'v22.23.4', 'v24.22.1'])('accepts supported runtime %s', version => {
    expect(isSupportedNodeVersion(version)).toBe(true);
  });
  it('advertises the same minimum used by the diagnostic', () => {
    const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
    expect(pkg.engines.node).toBe('>=20.9.0');
    // No override may pin sharp below the patched 0.35.5 (GHSA-wq5f-xc86-pv6w).
    expect(pkg.overrides?.['@huggingface/transformers']?.sharp).toBeUndefined();
    expect(pkg.optionalDependencies['better-sqlite3']).toBe('^12.8.0');
  });
});
