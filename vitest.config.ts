import { configDefaults, defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    exclude: [
      ...configDefaults.exclude,
      // Run by `claude plugin test plugins/agentdb` (imports claude-code/testing).
      'plugins/**',
      // Need a real browser (`window`) or are standalone node scripts.
      'tests/browser/attention-browser.test.js',
      'tests/browser/flash-attention-v2-browser.test.js',
      'tests/browser/browser-bundle-v2.test.js',
      // Benchmark and simulation suites assert machine speed or run for
      // minutes (#44). Run them with `npm run test:bench` (vitest.bench.config.ts).
      'benchmarks/**',
      'simulation/**',
      'tests/benchmarks/**',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/**',
        'dist/**',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/test-*.ts',
        'benchmarks/**',
        'scripts/**',
        'coverage/**',
        '**/*.config.ts',
        '**/db-fallback.ts', // Fallback type definitions
      ],
      include: ['src/**/*.ts'],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
    testTimeout: 30000,
    hookTimeout: 30000,
    setupFiles: ['./tests/setup.ts'],
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
