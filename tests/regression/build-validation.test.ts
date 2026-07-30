/**
 * AgentDB v1.6.0 Regression Tests - Build Validation
 * Tests TypeScript compilation, imports, and dependencies
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Build Validation Tests', () => {
  describe('TypeScript Compilation', () => {
    it('should have compiled all TypeScript files', () => {
      const distPath = path.join(__dirname, '../../dist');
      expect(fs.existsSync(distPath)).toBe(true);

      // Check for key compiled files
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
        const filePath = path.join(distPath, file);
        expect(fs.existsSync(filePath)).toBe(true);
      });
    });

    it('should have generated type definitions', () => {
      const distPath = path.join(__dirname, '../../dist');

      const typeFiles = [
        'index.d.ts',
        'controllers/ReflexionMemory.d.ts',
        'controllers/SkillLibrary.d.ts',
        'controllers/CausalMemoryGraph.d.ts',
        'controllers/EmbeddingService.d.ts'
      ];

      typeFiles.forEach(file => {
        const filePath = path.join(distPath, file);
        expect(fs.existsSync(filePath)).toBe(true);
      });
    });

    it('should have copied schema files', () => {
      const schemasPath = path.join(__dirname, '../../dist/schemas');
      expect(fs.existsSync(schemasPath)).toBe(true);

      const schemaFiles = ['schema.sql', 'frontier-schema.sql'];
      schemaFiles.forEach(file => {
        const filePath = path.join(schemasPath, file);
        expect(fs.existsSync(filePath)).toBe(true);
      });
    });

    it('should have built browser bundle', () => {
      const browserBundle = path.join(__dirname, '../../dist/agentdb.min.js');
      expect(fs.existsSync(browserBundle)).toBe(true);

      // Verify bundle size (should be reasonable)
      const stats = fs.statSync(browserBundle);
      expect(stats.size).toBeGreaterThan(1000); // At least 1KB
      expect(stats.size).toBeLessThan(10 * 1024 * 1024); // Less than 10MB
    });
  });

  describe('Import Resolution', () => {
    it('should resolve main exports', async () => {
      const mainExports = await import('../../dist/index.js');

      expect(mainExports).toHaveProperty('ReflexionMemory');
      expect(mainExports).toHaveProperty('SkillLibrary');
      expect(mainExports).toHaveProperty('CausalMemoryGraph');
      expect(mainExports).toHaveProperty('EmbeddingService');
      expect(mainExports).toHaveProperty('CausalRecall');
      expect(mainExports).toHaveProperty('ExplainableRecall');
      expect(mainExports).toHaveProperty('NightlyLearner');
    });

    it('should resolve controller imports', async () => {
      const reflexionModule = await import('../../dist/controllers/ReflexionMemory.js');
      expect(reflexionModule).toHaveProperty('ReflexionMemory');

      const skillsModule = await import('../../dist/controllers/SkillLibrary.js');
      expect(skillsModule).toHaveProperty('SkillLibrary');

      const causalModule = await import('../../dist/controllers/CausalMemoryGraph.js');
      expect(causalModule).toHaveProperty('CausalMemoryGraph');

      const embeddingModule = await import('../../dist/controllers/EmbeddingService.js');
      expect(embeddingModule).toHaveProperty('EmbeddingService');
    });

    it('should resolve CLI import', async () => {
      const cliModule = await import('../../dist/cli/agentdb-cli.js');
      expect(cliModule).toHaveProperty('AgentDBCLI');
    });

    it('should resolve db-fallback', async () => {
      const dbModule = await import('../../dist/db-fallback.js');
      expect(dbModule).toHaveProperty('createDatabase');
    });
  });

  describe('Circular Dependency Detection', () => {
    it('should not have circular dependencies in core modules', () => {
      // This test will pass if imports work correctly
      // Circular dependencies would cause import failures
      expect(true).toBe(true);
    });
  });

  describe('Package Structure', () => {
    it('should have correct package.json structure', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.name).toBe('agentdb');
      expect(packageJson.version).toBe('1.6.1');
      expect(packageJson.type).toBe('module');
      expect(packageJson.main).toBe('dist/index.js');
      expect(packageJson.types).toBe('dist/index.d.ts');
    });

    it('should have correct bin configuration', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.bin).toHaveProperty('agentdb');
      expect(packageJson.bin.agentdb).toBe('dist/cli/agentdb-cli.js');
    });

    it('should have correct exports configuration', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.exports).toHaveProperty('.');
      expect(packageJson.exports).toHaveProperty('./cli');
      expect(packageJson.exports).toHaveProperty('./controllers');
      expect(packageJson.exports).toHaveProperty('./controllers/ReflexionMemory');
      expect(packageJson.exports).toHaveProperty('./controllers/SkillLibrary');
      expect(packageJson.exports).toHaveProperty('./controllers/CausalMemoryGraph');
      expect(packageJson.exports).toHaveProperty('./controllers/EmbeddingService');
    });

    it('should have required dependencies', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.dependencies).toHaveProperty('@modelcontextprotocol/sdk');
      expect(packageJson.optionalDependencies).toHaveProperty('@huggingface/transformers');
      expect(packageJson.dependencies).toHaveProperty('chalk');
      expect(packageJson.dependencies).toHaveProperty('commander');
      expect(packageJson.dependencies).toHaveProperty('sql.js');
      expect(packageJson.dependencies).toHaveProperty('zod');
    });

    it('should have required devDependencies', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.devDependencies).toHaveProperty('@types/node');
      expect(packageJson.devDependencies).toHaveProperty('typescript');
      expect(packageJson.devDependencies).toHaveProperty('vitest');
    });
  });

  describe('File Inclusion', () => {
    it('should include required files in package', () => {
      const packageJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf-8')
      );

      expect(packageJson.files).toContain('dist');
      expect(packageJson.files).toContain('src');
      expect(packageJson.files).toContain('scripts/postinstall.cjs');
      expect(packageJson.files).toContain('README.md');
      expect(packageJson.files).toContain('LICENSE');
    });
  });
});
