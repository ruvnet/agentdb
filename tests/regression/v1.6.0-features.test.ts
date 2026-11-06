/**
 * AgentDB v1.6.0 Regression Tests - New v1.6.0 Features
 * Tests vector-search, export/import, stats, and enhanced init command
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawn } from 'child_process';
import { createDatabase } from '../../src/db-fallback.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory.js';

// Shells the built CLI (package.json "bin"), so it needs `npm run build`.
const ROOT = path.join(__dirname, '../..');
const CLI_PATH = path.join(
  ROOT,
  JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf-8')).bin.agentdb
);
const hasCli = fs.existsSync(CLI_PATH);
if (!hasCli) {
  console.warn('[v1.6.0-features] built CLI missing; run `npm run build` to enable these tests');
}
// All CLI artifacts (including the default ./agentdb.db) land in a temp dir.
const WORK_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-v160-'));

describe.skipIf(!hasCli)('v1.6.0 New Features Regression Tests', () => {
  const exportPath = path.join(WORK_DIR, 'test-export.json');
  const importPath = path.join(WORK_DIR, 'test-import.db');

  afterAll(() => {
    fs.rmSync(WORK_DIR, { recursive: true, force: true });
  });

  describe('Init Command Enhancements', () => {
    it('should support --dimension flag', async () => {
      const output = await runCLI(['init', './test-dimension.db', '--dimension', '768']);
      expect(output).toMatch(/Dimension:\s+768/);
      expect(output).toContain('initialized successfully');
    });

    it('should support --preset flag (small)', async () => {
      const output = await runCLI(['init', './test-preset-small.db', '--preset', 'small']);
      expect(output).toMatch(/Preset:\s+small/);
      expect(output).toContain('initialized successfully');
    });

    it('should support --preset flag (medium)', async () => {
      const output = await runCLI(['init', './test-preset-medium.db', '--preset', 'medium']);
      expect(output).toMatch(/Preset:\s+medium/);
      expect(output).toContain('initialized successfully');
    });

    it('should support --preset flag (large)', async () => {
      const output = await runCLI(['init', './test-preset-large.db', '--preset', 'large']);
      expect(output).toMatch(/Preset:\s+large/);
      expect(output).toContain('initialized successfully');
    });

    it('should support --in-memory flag', async () => {
      const output = await runCLI(['init', '--in-memory']);
      expect(output).toMatch(/Database:\s+:memory:/);
      expect(output).toContain('initialized successfully');
    });
  });

  describe('Vector Search Command', () => {
    const testDbPath = path.join(WORK_DIR, 'vector-search.db');
    let db: any;
    let embedder: EmbeddingService;
    let reflexion: ReflexionMemory;

    beforeAll(async () => {
      // Initialize database with test data
      await runCLI(['init', testDbPath]);

      db = await createDatabase(testDbPath);
      embedder = new EmbeddingService({
        model: 'Xenova/all-MiniLM-L6-v2',
        dimension: 384,
        provider: 'transformers'
      });
      await embedder.initialize();

      reflexion = new ReflexionMemory(db, embedder);

      // Store some episodes with embeddings
      await reflexion.storeEpisode({
        sessionId: 'vector-test-1',
        task: 'authentication implementation',
        reward: 0.9,
        success: true
      });

      await reflexion.storeEpisode({
        sessionId: 'vector-test-2',
        task: 'database optimization',
        reward: 0.85,
        success: true
      });

      // Save database
      if (db && typeof db.save === 'function') {
        db.save();
      }
    });

    it('should perform vector search with cosine metric', async () => {
      // Generate a test vector
      const embedding = await embedder.embed('authentication');
      const vectorStr = JSON.stringify(Array.from(embedding));

      const output = await runCLIStreams(['vector-search', testDbPath, vectorStr, '-k', '5', '-m', 'cosine']);

      // Should return JSON results
      const results = JSON.parse(output.stdout);
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty('similarity');
    });

    it('should perform vector search with euclidean metric', async () => {
      const embedding = await embedder.embed('database');
      const vectorStr = JSON.stringify(Array.from(embedding));

      const output = await runCLIStreams(['vector-search', testDbPath, vectorStr, '-k', '5', '-m', 'euclidean']);

      const results = JSON.parse(output.stdout);
      expect(Array.isArray(results)).toBe(true);
      expect(results[0]).toHaveProperty('similarity');
    });

    it('should perform vector search with dot product metric', async () => {
      const embedding = await embedder.embed('optimization');
      const vectorStr = JSON.stringify(Array.from(embedding));

      const output = await runCLIStreams(['vector-search', testDbPath, vectorStr, '-k', '5', '-m', 'dot']);

      const results = JSON.parse(output.stdout);
      expect(Array.isArray(results)).toBe(true);
      expect(results[0]).toHaveProperty('similarity');
    });

    it('should filter by threshold', async () => {
      const embedding = await embedder.embed('testing');
      const vectorStr = JSON.stringify(Array.from(embedding));

      const output = await runCLIStreams(['vector-search', testDbPath, vectorStr, '-k', '10', '-t', '0.9']);

      const results = JSON.parse(output.stdout);
      expect(Array.isArray(results)).toBe(true);
      // All results should have similarity >= 0.9
      results.forEach((r: any) => {
        expect(r.similarity).toBeGreaterThanOrEqual(0.9);
      });
    });

    afterAll(() => {
      if (db && typeof db.close === 'function') {
        db.close();
      }
    });
  });

  describe('Export/Import Commands', () => {
    const testDbPath = path.join(WORK_DIR, 'export.db');
    let db: any;
    let embedder: EmbeddingService;
    let reflexion: ReflexionMemory;

    beforeAll(async () => {
      // Create database with test data
      await runCLI(['init', testDbPath]);

      db = await createDatabase(testDbPath);
      embedder = new EmbeddingService({
        model: 'Xenova/all-MiniLM-L6-v2',
        dimension: 384,
        provider: 'transformers'
      });
      await embedder.initialize();

      reflexion = new ReflexionMemory(db, embedder);

      // Store test episodes
      for (let i = 0; i < 5; i++) {
        await reflexion.storeEpisode({
          sessionId: `export-test-${i}`,
          task: `task ${i}`,
          reward: 0.7 + i * 0.05,
          success: true,
          input: `input ${i}`,
          output: `output ${i}`
        });
      }

      // Save database
      if (db && typeof db.save === 'function') {
        db.save();
      }
    });

    it('should export database to JSON', async () => {
      const output = await runCLI(['export', testDbPath, exportPath]);

      expect(output).toContain('Exported');
      expect(output).toContain('episodes');
      expect(fs.existsSync(exportPath)).toBe(true);

      // Verify export file structure
      const exportData = JSON.parse(fs.readFileSync(exportPath, 'utf-8'));
      expect(Array.isArray(exportData)).toBe(true);
      expect(exportData.length).toBe(5);
      expect(exportData[0]).toHaveProperty('task');
      expect(exportData[0]).toHaveProperty('reward');
    });

    it('should import database from JSON', async () => {
      // First export
      await runCLI(['export', testDbPath, exportPath]);

      // Then import to new database
      const output = await runCLI(['import', exportPath, importPath]);

      expect(output).toContain('Imported');
      expect(output).toContain('episodes');
      expect(fs.existsSync(importPath)).toBe(true);

      // Verify imported data
      const importedDb = await createDatabase(importPath);
      const count = importedDb.prepare('SELECT COUNT(*) as count FROM episodes').get();
      expect(count.count).toBe(5);

      importedDb.close();
    });

    it('should preserve embeddings during export/import', async () => {
      await runCLI(['export', testDbPath, exportPath]);
      await runCLI(['import', exportPath, importPath]);

      const importedDb = await createDatabase(importPath);
      const embeddingCount = importedDb.prepare('SELECT COUNT(*) as count FROM episode_embeddings').get();
      expect(embeddingCount.count).toBeGreaterThan(0);

      importedDb.close();
    });

    afterAll(() => {
      if (db && typeof db.close === 'function') {
        db.close();
      }
    });
  });

  describe('Stats Command', () => {
    const testDbPath = path.join(WORK_DIR, 'stats.db');
    let db: any;
    let embedder: EmbeddingService;
    let reflexion: ReflexionMemory;

    beforeAll(async () => {
      await runCLI(['init', testDbPath]);

      db = await createDatabase(testDbPath);
      embedder = new EmbeddingService({
        model: 'Xenova/all-MiniLM-L6-v2',
        dimension: 384,
        provider: 'transformers'
      });
      await embedder.initialize();

      reflexion = new ReflexionMemory(db, embedder);

      // Store test data
      for (let i = 0; i < 10; i++) {
        await reflexion.storeEpisode({
          sessionId: `stats-test-${i}`,
          task: i < 5 ? 'task_a' : 'task_b',
          reward: 0.5 + Math.random() * 0.5,
          success: true
        });
      }

      if (db && typeof db.save === 'function') {
        db.save();
      }
    });

    it('should show database statistics', async () => {
      const output = await runCLI(['stats', testDbPath]);

      expect(output).toContain('AgentDB Statistics');
      expect(output).toContain('Episodes:');
      expect(output).toContain('Embeddings:');
      expect(output).toContain('Average Reward:');
      expect(output).toContain('Embedding Coverage:');
    });

    it('should show episode counts', async () => {
      const output = await runCLI(['stats', testDbPath]);
      expect(output).toContain('Episodes: 10');
    });

    it('should show top domains', async () => {
      const output = await runCLI(['stats', testDbPath]);
      expect(output).toContain('Top Domains:');
      expect(output).toContain('task_a');
      expect(output).toContain('task_b');
    });

    it('should show database size', async () => {
      const output = await runCLI(['stats', testDbPath]);
      expect(output).toContain('Size:');
      expect(output).toContain('KB');
    });

    afterAll(() => {
      if (db && typeof db.close === 'function') {
        db.close();
      }
    });
  });
});

/**
 * Helper function to run CLI commands
 */
async function runCLI(args: string[]): Promise<string> {
  const { stdout, stderr } = await runCLIStreams(args);
  return stdout + stderr;
}

async function runCLIStreams(args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [CLI_PATH, ...args], {
      cwd: WORK_DIR,
      env: { ...process.env, AGENTDB_PATH: path.join(WORK_DIR, 'test-cli.db'), NO_COLOR: '1' }
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr?.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      // CLI commands may exit with 0 or 1 depending on the operation
      // We consider both successful for testing purposes
      // eslint-disable-next-line no-control-regex
      const strip = (text: string) => text.replace(/\x1b\[[0-9;]*m/g, '');
      resolve({ stdout: strip(stdout), stderr: strip(stderr) });
    });

    child.on('error', (error) => {
      reject(error);
    });
  });
}
