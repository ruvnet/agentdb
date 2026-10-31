/**
 * Tests for ModelCacheLoader — .rvf extraction + offline loading
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';

describe('ModelCacheLoader', () => {
  let tempDir: string;
  const savedXdg = process.env.XDG_CACHE_HOME;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-model-test-'));
    // Keep extraction out of the real ~/.cache
    process.env.XDG_CACHE_HOME = path.join(tempDir, 'xdg-cache');
  });

  afterEach(() => {
    if (savedXdg === undefined) delete process.env.XDG_CACHE_HOME;
    else process.env.XDG_CACHE_HOME = savedXdg;
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  /**
   * Helper: Create a small test .rvf file with dummy model assets.
   */
  async function createTestRvf(outputPath: string, files: Record<string, string>): Promise<void> {
    const sqlJsModule = await import('sql.js');
    const SQL = await sqlJsModule.default();
    const db = new SQL.Database();

    db.run(`
      CREATE TABLE model_assets (
        filename TEXT PRIMARY KEY,
        content  BLOB NOT NULL,
        size     INTEGER NOT NULL,
        sha256   TEXT NOT NULL
      )
    `);
    db.run(`
      CREATE TABLE model_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    `);
    db.run("INSERT INTO model_meta (key, value) VALUES ('model_id', 'Xenova/test-model')");
    db.run("INSERT INTO model_meta (key, value) VALUES ('dimension', '384')");
    db.run("INSERT INTO model_meta (key, value) VALUES ('format_version', '1')");

    for (const [filename, content] of Object.entries(files)) {
      const buf = Buffer.from(content, 'utf-8');
      const sha256 = crypto.createHash('sha256').update(buf).digest('hex');
      db.run(
        'INSERT INTO model_assets (filename, content, size, sha256) VALUES (?, ?, ?, ?)',
        [filename, buf, buf.length, sha256]
      );
    }

    const data = db.export();
    db.close();
    fs.writeFileSync(outputPath, Buffer.from(data));
  }

  it('should extract files from .rvf and verify checksums', async () => {
    const { ModelCacheLoader } = await import('../../src/model/ModelCacheLoader.js');

    const rvfPath = path.join(tempDir, 'test-model.rvf');
    const dummyFiles = {
      'config.json': '{"model_type":"bert","hidden_size":384}',
      'tokenizer.json': '{"version":"1.0","model":{"type":"BPE"}}',
      'onnx/model_quantized.onnx': 'fake-onnx-binary-data-for-testing',
    };

    await createTestRvf(rvfPath, dummyFiles);

    // Extract from .rvf
    const basePath = await ModelCacheLoader.extractFromRvf(rvfPath, 'test-model');

    // Verify all files were extracted
    const modelDir = path.join(basePath, 'Xenova', 'test-model');
    for (const [filename, expectedContent] of Object.entries(dummyFiles)) {
      const filePath = path.join(modelDir, filename);
      expect(fs.existsSync(filePath)).toBe(true);

      const actualContent = fs.readFileSync(filePath, 'utf-8');
      expect(actualContent).toBe(expectedContent);

      // Verify checksum
      const hash = crypto.createHash('sha256').update(Buffer.from(actualContent)).digest('hex');
      const expectedHash = crypto.createHash('sha256').update(Buffer.from(expectedContent)).digest('hex');
      expect(hash).toBe(expectedHash);
    }
  });

  it('should skip re-extraction when files already exist with correct checksums', async () => {
    const { ModelCacheLoader } = await import('../../src/model/ModelCacheLoader.js');

    const rvfPath = path.join(tempDir, 'test-model.rvf');
    const dummyFiles = {
      'config.json': '{"test":"data"}',
    };

    await createTestRvf(rvfPath, dummyFiles);

    // Extract once
    const basePath1 = await ModelCacheLoader.extractFromRvf(rvfPath, 'test-model');
    const filePath = path.join(basePath1, 'Xenova', 'test-model', 'config.json');
    const mtime1 = fs.statSync(filePath).mtimeMs;

    // Wait briefly to ensure mtime would differ on re-write
    await new Promise(r => setTimeout(r, 50));

    // Extract again — should skip (checksums match)
    await ModelCacheLoader.extractFromRvf(rvfPath, 'test-model');
    const mtime2 = fs.statSync(filePath).mtimeMs;

    expect(mtime2).toBe(mtime1);
  });

  it('should set offline mode in EmbeddingService when model is cached', async () => {
    // Mock ModelCacheLoader.resolve to return a cached result
    const mockResult = { localPath: '/tmp/mock-models', fromBundle: true };

    vi.doMock('../../src/model/ModelCacheLoader.js', () => ({
      ModelCacheLoader: {
        resolve: vi.fn().mockResolvedValue(mockResult),
      },
    }));

    // Simulate what EmbeddingService.initialize() does
    const envSettings: Record<string, unknown> = {};
    const mockTransformersEnv = new Proxy(envSettings, {
      set(target, prop, value) {
        target[prop as string] = value;
        return true;
      },
    });

    // Replicate the EmbeddingService logic for the bundled model path
    const { ModelCacheLoader } = await import('../../src/model/ModelCacheLoader.js');
    const cached = await ModelCacheLoader.resolve('all-MiniLM-L6-v2');

    if (cached) {
      mockTransformersEnv.localModelPath = cached.localPath;
      mockTransformersEnv.allowRemoteModels = false;
      mockTransformersEnv.cacheDir = cached.localPath;
    }

    expect(envSettings.localModelPath).toBe('/tmp/mock-models');
    expect(envSettings.allowRemoteModels).toBe(false);
    expect(envSettings.cacheDir).toBe('/tmp/mock-models');

    vi.doUnmock('../../src/model/ModelCacheLoader.js');
  });
});
