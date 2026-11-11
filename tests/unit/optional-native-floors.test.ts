/**
 * Optional native/WASM dependency floors (3.8.1).
 *
 * - ruvector-attention-wasm 2.x is a wasm-bindgen "web" build. Its default
 *   init fetch()es a file:// URL, which Node rejects, so AttentionService's
 *   WASM path never loaded in Node. It now initialises from the .wasm bytes.
 * - The @ruvector native floors are the first builds whose linux-arm64
 *   binaries contain 0 LDAPR instructions (older ones SIGILL on Cortex-A72,
 *   i.e. Raspberry Pi 4). These tests pin the declared floors and check that
 *   whatever is installed satisfies them and really imports.
 */
import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import semver from 'semver';

const require = createRequire(import.meta.url);
const pkg = JSON.parse(readFileSync('package.json', 'utf8'));

const FLOORS: Record<string, string> = {
  'ruvector-attention-wasm': '2.1.0',
  '@ruvector/attention': '^2.3.1',
  '@ruvector/graph-node': '^2.1.4',
  '@ruvector/ruvllm': '^4.2.2',
  '@ruvector/core': '0.1.38',
  '@ruvector/gnn': '0.1.28',
  '@ruvector/router': '0.1.34',
  '@ruvector/sona': '0.1.13',
  '@ruvector/rvf': '0.4.12',
  '@ruvector/rvf-node': '0.4.7',
  ruvector: '0.3.20',
  // 3.8.2: exact pins instead of ^0.1.x ranges
  '@ruvector/rvf-wasm': '0.1.13',
  '@ruvector/rvf-solver': '0.1.11',
  '@ruvector/rabitq-wasm': '0.1.2',
};

function installedVersion(name: string): string | null {
  try {
    return require(`${name}/package.json`).version;
  } catch {
    return null;
  }
}

describe('optional dependency floors', () => {
  it.each(Object.entries(FLOORS))('%s is declared as optional at %s', (name, range) => {
    expect(pkg.optionalDependencies[name]).toBe(range);
    expect(pkg.dependencies?.[name]).toBeUndefined();
  });

  it('declares no ^0.x range for these edges', () => {
    for (const name of Object.keys(FLOORS)) {
      expect(pkg.optionalDependencies[name]).not.toMatch(/^\^0\./);
    }
  });

  it.each(Object.entries(FLOORS))('installed %s satisfies %s and imports', async (name, range) => {
    const version = installedVersion(name);
    if (version === null) {
      console.warn(`skipping ${name}: optional package not installed`);
      return;
    }
    expect(semver.satisfies(version, range)).toBe(true);
    const mod = await import(name);
    const exported = new Set([...Object.keys(mod), ...Object.keys(mod.default ?? {})]);
    exported.delete('default');
    expect(exported.size).toBeGreaterThan(0);
  });
});

describe('AttentionService WASM path in Node (ruvector-attention-wasm 2.x)', () => {
  it('loads the WASM module and computes flash attention through it', async () => {
    if (installedVersion('ruvector-attention-wasm') === null) {
      console.warn('skipping: ruvector-attention-wasm not installed');
      return;
    }
    // Make the NAPI binding unavailable so the service falls through to WASM,
    // as it does in browsers and on platforms without a prebuilt binary.
    vi.doMock('@ruvector/attention', () => {
      throw new Error('NAPI binding unavailable (test)');
    });
    process.env.AGENTDB_PERF_LOGGING = 'true';
    const svc = await import('../../src/services/AttentionService.js');
    const status = await svc.initializeAttentionBackends();
    expect(status.wasmLoaded).toBe(true);
    expect(status.napiLoaded).toBe(false);
    expect(svc.isWasmAvailable()).toBe(true);

    const dim = 384;
    const nKeys = 8;
    const fill = (n: number, seed: number) =>
      Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37 + seed));
    const service = new svc.AttentionService(null as any, { flash: { enabled: true, blockSize: 4 } });
    svc.clearPerformanceLogs();
    const result = await service.flashAttention(fill(dim, 1), fill(dim * nKeys, 2), fill(dim * nKeys, 3));

    expect(result.output.length).toBe(dim);
    expect(Array.from(result.output).every(Number.isFinite)).toBe(true);
    const log = svc.getPerformanceLogs().find(l => l.mechanism === 'flashAttention');
    expect(log?.backend).toBe('wasm');
  });

  it('exposes the classes AttentionService constructs, with the same signatures', async () => {
    if (installedVersion('ruvector-attention-wasm') === null) return;
    const wasm: any = await import('ruvector-attention-wasm');
    const dim = 16;
    const q = Float32Array.from({ length: dim }, (_, i) => i / dim);
    const K = [q, q.map(v => 1 - v)];
    for (const make of [
      () => new wasm.WasmFlashAttention(dim, 4),
      () => new wasm.WasmHyperbolicAttention(dim, 1.0),
      () => new wasm.WasmMoEAttention(dim, 4, 2),
    ]) {
      const out: Float32Array = make().compute(q, K, K);
      expect(out.length).toBe(dim);
      expect(Array.from(out).every(Number.isFinite)).toBe(true);
    }
  });
});

describe('pinned WASM edges load through the paths agentdb uses (3.8.2)', () => {
  it('@ruvector/rvf-wasm init() resolves to the raw RVF exports', async () => {
    if (installedVersion('@ruvector/rvf-wasm') === null) return;
    const mod: any = await import('@ruvector/rvf-wasm');
    expect(typeof mod.default).toBe('function');
    const exportsObj = await mod.default();
    expect(typeof exportsObj.rvf_crc32c).toBe('function');
    expect(typeof exportsObj.rvf_witness_verify).toBe('function');
  });

  it('@ruvector/rvf-solver backs AgentDBSolver', async () => {
    if (installedVersion('@ruvector/rvf-solver') === null) return;
    const { AgentDBSolver } = await import('../../src/backends/rvf/RvfSolver.js');
    expect(await AgentDBSolver.isAvailable()).toBe(true);
    expect(await AgentDBSolver.create()).toBeTruthy();
  });

  it('@ruvector/rabitq-wasm node entry backs loadRabitq()', async () => {
    if (installedVersion('@ruvector/rabitq-wasm') === null) return;
    const { loadRabitq } = await import('../../src/backends/rvf/RabitqAccelerator.js');
    const RabitqIndex: any = await loadRabitq();
    expect(typeof RabitqIndex?.build).toBe('function');
  });
});
