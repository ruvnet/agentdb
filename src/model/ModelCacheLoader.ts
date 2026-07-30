/**
 * ModelCacheLoader - Extracts bundled ONNX models from .rvf files
 *
 * Resolution order:
 * 1. AGENTDB_MODEL_PATH env var (user override)
 * 2. Bundled .rvf at <package>/dist/models/<modelId>.rvf
 * 3. Existing Transformers.js cache directory
 * 4. Previously extracted temp dir
 * 5. null (caller falls through to network download)
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';

export interface ModelCacheResult {
  localPath: string;
  fromBundle: boolean;
}

const TEMP_MODEL_DIR = path.join(os.tmpdir(), 'agentdb-models');

// Cache sql.js factory across calls to avoid repeated WASM init (~15ms each)
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- sql.js has no shared TS type
let cachedSqlFactory: any = null;

async function getSqlFactory() {
  if (!cachedSqlFactory) {
    const mod = await import('sql.js');
    cachedSqlFactory = await mod.default();
  }
  return cachedSqlFactory;
}

function validateModelId(modelId: string): void {
  if (modelId.includes('..') || path.isAbsolute(modelId) || /[<>:"|?*]/.test(modelId)) {
    throw new Error(`Invalid model ID: ${modelId}`);
  }
}

export class ModelCacheLoader {
  static async resolve(modelId: string): Promise<ModelCacheResult | null> {
    validateModelId(modelId);

    // 1. Check AGENTDB_MODEL_PATH env var
    const envPath = process.env.AGENTDB_MODEL_PATH;
    if (envPath) {
      const modelDir = path.join(envPath, 'Xenova', modelId);
      if (fs.existsSync(modelDir)) {
        return { localPath: envPath, fromBundle: false };
      }
    }

    // 2. Check for bundled .rvf
    const rvfPath = ModelCacheLoader.findBundledRvf(modelId);
    if (rvfPath) {
      const extractedPath = await ModelCacheLoader.extractFromRvf(rvfPath, modelId);
      return { localPath: extractedPath, fromBundle: true };
    }

    // 3. Check existing transformers.js cache locations
    const cacheDirs = [
      path.join(process.cwd(), 'node_modules', '@huggingface', 'transformers', '.cache'),
      // Preserve already-downloaded v2 models during the runtime migration.
      path.join(process.cwd(), 'node_modules', '@xenova', 'transformers', '.cache'),
      path.join(os.homedir(), '.cache', 'huggingface', 'hub'),
    ];
    for (const cacheDir of cacheDirs) {
      const onnxPath = path.join(cacheDir, 'Xenova', modelId, 'onnx', 'model_quantized.onnx');
      if (fs.existsSync(onnxPath)) {
        return { localPath: cacheDir, fromBundle: false };
      }
    }

    // 4. Check previously extracted temp dir
    const tempOnnx = path.join(TEMP_MODEL_DIR, 'Xenova', modelId, 'onnx', 'model_quantized.onnx');
    if (fs.existsSync(tempOnnx)) {
      return { localPath: TEMP_MODEL_DIR, fromBundle: true };
    }

    return null;
  }

  /**
   * Extract model files from a .rvf bundle to a temp directory.
   * Skips files whose on-disk checksum already matches.
   */
  static async extractFromRvf(rvfPath: string, modelId: string): Promise<string> {
    validateModelId(modelId);
    const targetDir = path.join(TEMP_MODEL_DIR, 'Xenova', modelId);

    const SQL = await getSqlFactory();
    const fileBuffer = fs.readFileSync(rvfPath);
    const db = new SQL.Database(new Uint8Array(fileBuffer));

    try {
      const rows = db.exec('SELECT filename, content, sha256 FROM model_assets');
      if (!rows.length || !rows[0].values.length) {
        throw new Error('No model assets found in .rvf file');
      }

      for (const [filename, content, sha256] of rows[0].values) {
        const name = filename as string;
        const blob = content as Uint8Array;
        const expectedHash = sha256 as string;

        const filePath = path.join(targetDir, name);

        // Skip if file exists with matching checksum
        if (fs.existsSync(filePath)) {
          const existingHash = crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
          if (existingHash === expectedHash) continue;
        }

        // Verify blob checksum in-memory before writing (avoids double disk read)
        const blobBuffer = Buffer.from(blob);
        const blobHash = crypto.createHash('sha256').update(blobBuffer).digest('hex');
        if (blobHash !== expectedHash) {
          throw new Error(`SHA-256 mismatch for ${name}: expected ${expectedHash}, got ${blobHash}`);
        }

        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        fs.writeFileSync(filePath, blobBuffer);
      }
    } finally {
      db.close();
    }

    return TEMP_MODEL_DIR;
  }

  private static findBundledRvf(modelId: string): string | null {
    const dirname = path.dirname(new URL(import.meta.url).pathname);
    const candidates = [
      path.join(dirname, '../../models', `${modelId}.rvf`),
      path.join(dirname, '../../../dist/models', `${modelId}.rvf`),
      path.join(dirname, '../../../../dist/models', `${modelId}.rvf`),
    ];

    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) return candidate;
    }
    return null;
  }
}
