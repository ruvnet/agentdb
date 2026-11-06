import { configDefaults, defineConfig } from 'vitest/config';
import baseConfig from './vitest.config';

/**
 * Benchmark and simulation suites (#44).
 *
 * These measure machine speed or run long simulations, so the default
 * `vitest` run excludes them. The `benchmark:*` and `test:bench` scripts run
 * them with `--config vitest.bench.config.ts`. Absolute speed thresholds are
 * skipped when `CI` is set; see tests/benchmarks/helpers/speed-gate.ts.
 */
export default defineConfig({
  ...baseConfig,
  test: {
    ...baseConfig.test,
    include: [
      'benchmarks/**/*.test.ts',
      'simulation/tests/**/*.test.ts',
      'tests/benchmarks/**/*.test.ts',
    ],
    exclude: [...configDefaults.exclude],
    testTimeout: 600_000,
    hookTimeout: 600_000,
  },
});
