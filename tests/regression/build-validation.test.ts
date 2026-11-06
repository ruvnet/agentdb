/**
 * AgentDB Regression Tests - Build Validation
 * Tests compiled output, imports, and package metadata.
 *
 * The dist/ checks need a prior `npm run build` (tsc emits under dist/src/
 * because rootDir is "."). They are skipped, not failed, when dist/ is absent
 * so the default `vitest run` stays meaningful; paths are read from
 * package.json so the test follows the real published layout.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';

const ROOT = path.join(__dirname, '../..');
const DIST = path.join(ROOT, 'dist');
const packageJson = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf-8'));
const MAIN = path.join(ROOT, packageJson.main);
const OUT = path.dirname(MAIN); // dist/src
const hasDist = fs.existsSync(MAIN);

if (!hasDist) {
  console.warn('[build-validation] dist/ not built; run `npm run build` to enable the dist checks');
}

const importDist = (rel: string) => import(pathToFileURL(path.join(OUT, rel)).href);

describe('Build Validation Tests', () => {
  describe.skipIf(!hasDist)('TypeScript Compilation (requires npm run build)', () => {
    it('should have compiled all TypeScript files', () => {
      const keyFiles = [
        'index.js',
        'index.d.ts',
        'cli/agentdb-cli.js',
        'controllers/ReflexionMemory.js',
        'controllers/SkillLibrary.js',
        'controllers/CausalMemoryGraph.js',
        'controllers/EmbeddingService.js',
        'controllers/CausalRecall.js',
        'controllers/ExplainableRecall.js',
        'controllers/NightlyLearner.js',
        'mcp/agentdb-mcp-server.js',
        'db-fallback.js'
      ];

      keyFiles.forEach(file => {
        expect(fs.existsSync(path.join(OUT, file)), file).toBe(true);
      });
    });

    it('should have generated type definitions', () => {
      expect(fs.existsSync(path.join(ROOT, packageJson.types))).toBe(true);

      const typeFiles = [
        'controllers/ReflexionMemory.d.ts',
        'controllers/SkillLibrary.d.ts',
        'controllers/CausalMemoryGraph.d.ts',
        'controllers/EmbeddingService.d.ts'
      ];

      typeFiles.forEach(file => {
        expect(fs.existsSync(path.join(OUT, file)), file).toBe(true);
      });
    });

    it('should have copied schema files', () => {
      const schemasPath = path.join(DIST, 'schemas');
      ['schema.sql', 'frontier-schema.sql'].forEach(file => {
        expect(fs.existsSync(path.join(schemasPath, file)), file).toBe(true);
      });
    });

    it('should have built browser bundle', () => {
      const browserBundle = path.join(DIST, 'agentdb.min.js');
      expect(fs.existsSync(browserBundle)).toBe(true);

      const stats = fs.statSync(browserBundle);
      expect(stats.size).toBeGreaterThan(1000);
      expect(stats.size).toBeLessThan(10 * 1024 * 1024);
    });
  });

  describe.skipIf(!hasDist)('Import Resolution (requires npm run build)', () => {
    it('should resolve main exports', async () => {
      const mainExports = await import(pathToFileURL(MAIN).href);

      expect(mainExports).toHaveProperty('ReflexionMemory');
      expect(mainExports).toHaveProperty('SkillLibrary');
      expect(mainExports).toHaveProperty('CausalMemoryGraph');
      expect(mainExports).toHaveProperty('EmbeddingService');
      expect(mainExports).toHaveProperty('CausalRecall');
      expect(mainExports).toHaveProperty('ExplainableRecall');
      expect(mainExports).toHaveProperty('NightlyLearner');
    });

    it('should resolve controller imports', async () => {
      expect(await importDist('controllers/ReflexionMemory.js')).toHaveProperty('ReflexionMemory');
      expect(await importDist('controllers/SkillLibrary.js')).toHaveProperty('SkillLibrary');
      expect(await importDist('controllers/CausalMemoryGraph.js')).toHaveProperty('CausalMemoryGraph');
      expect(await importDist('controllers/EmbeddingService.js')).toHaveProperty('EmbeddingService');
    });

    it('should resolve CLI import', async () => {
      expect(await importDist('cli/agentdb-cli.js')).toHaveProperty('AgentDBCLI');
    });

    it('should resolve db-fallback', async () => {
      expect(await importDist('db-fallback.js')).toHaveProperty('createDatabase');
    });

    it('should point bin at a built file', () => {
      expect(fs.existsSync(path.join(ROOT, packageJson.bin.agentdb))).toBe(true);
    });
  });

  describe('Package Structure', () => {
    it('should have correct package.json structure', () => {
      expect(packageJson.name).toBe('agentdb');
      expect(packageJson.version).toMatch(/^\d+\.\d+\.\d+(-[\w.]+)?$/);
      expect(packageJson.type).toBe('module');
      expect(packageJson.main).toBe('dist/src/index.js');
      expect(packageJson.types).toBe('dist/src/index.d.ts');
    });

    it('should have correct bin configuration', () => {
      expect(packageJson.bin).toHaveProperty('agentdb');
      expect(packageJson.bin.agentdb).toBe('dist/src/cli/agentdb-cli.js');
    });

    it('should have correct exports configuration', () => {
      expect(packageJson.exports).toHaveProperty('.');
      expect(packageJson.exports).toHaveProperty('./cli');
      expect(packageJson.exports).toHaveProperty('./controllers');
      expect(packageJson.exports).toHaveProperty('./controllers/ReflexionMemory');
      expect(packageJson.exports).toHaveProperty('./controllers/SkillLibrary');
      expect(packageJson.exports).toHaveProperty('./controllers/CausalMemoryGraph');
      expect(packageJson.exports).toHaveProperty('./controllers/EmbeddingService');
    });

    it('should have required dependencies', () => {
      const runtime = { ...packageJson.dependencies, ...packageJson.optionalDependencies };

      expect(packageJson.dependencies).toHaveProperty('@modelcontextprotocol/sdk');
      expect(packageJson.dependencies).toHaveProperty('sql.js');
      expect(packageJson.optionalDependencies).toHaveProperty('@huggingface/transformers');
      // CLI-only deps are optional so library installs stay lean
      expect(runtime).toHaveProperty('chalk');
      expect(runtime).toHaveProperty('commander');
    });

    it('should have required devDependencies', () => {
      expect(packageJson.devDependencies).toHaveProperty('@types/node');
      expect(packageJson.devDependencies).toHaveProperty('typescript');
      expect(packageJson.devDependencies).toHaveProperty('vitest');
    });
  });

  describe('File Inclusion', () => {
    it('should include required files in package', () => {
      expect(packageJson.files).toContain('dist/src/');
      expect(packageJson.files).toContain('dist/schemas/');
      expect(packageJson.files).toContain('scripts/postinstall.cjs');
      expect(packageJson.files).toContain('README.md');
      expect(packageJson.files).toContain('LICENSE');
    });
  });
});
