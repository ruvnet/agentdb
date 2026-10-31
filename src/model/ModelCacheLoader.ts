/**
 * ModelCacheLoader - Extracts bundled ONNX models from .rvf files
 *
 * Resolution order:
 * 1. AGENTDB_MODEL_PATH env var (user override)
 * 2. Bundled .rvf at <package>/dist/models/<modelId>.rvf
 * 3. Existing Transformers.js cache directory
 * 4. Previously extracted per-user cache dir
 * 5. null (caller falls through to network download)
 *
 * Extracted models live in a per-user cache directory
 * ($XDG_CACHE_HOME/agentdb-models, default ~/.cache/agentdb-models), not the
 * shared os.tmpdir(): another local user could pre-create or symlink a
 * shared path and plant model files.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as crypto from 'crypto';

export interface ModelCacheResult {
  localPath: string;
  fromBundle: boolean;
}

/** Per-user model cache directory (resolved per call so XDG_CACHE_HOME changes apply). */
export function getModelCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME && path.isAbsolute(process.env.XDG_CACHE_HOME)
    ? process.env.XDG_CACHE_HOME
    : path.join(os.homedir(), '.cache');
  return path.join(base, 'agentdb-models');
}

/**
 * Create (0o700) or check the cache directory: it must be a real directory,
 * not a symlink, owned by the current user. Throws otherwise.
 */
function ensurePrivateDir(dir: string, create: boolean): boolean {
  if (create) fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  let st: fs.Stats;
  try {
    st = fs.lstatSync(dir);
  } catch {
    return false;
  }
  if (st.isSymbolicLink() || !st.isDirectory()) {
    throw new Error(`Model cache directory is not a real directory: ${dir}`);
  }
  if (typeof process.getuid === 'function' && st.uid !== process.getuid()) {
    throw new Error(`Model cache directory is owned by another user: ${dir}`);
  }
  if (process.platform !== 'win32' && (st.mode & 0o077) !== 0) {
    fs.chmodSync(dir, 0o700);
  }
  return true;
}

/**
 * Resolve an asset filename under targetDir. Rejects absolute paths, NUL
 * bytes and anything (e.g. "../") that would leave targetDir.
 */
function resolveAssetPath(targetDir: string, name: string): string {
  if (typeof name !== 'string' || name.length === 0 || name.includes('\0') || path.isAbsolute(name)) {
    throw new Error(`Invalid model asset filename: ${JSON.stringify(name)}`);
  }
  const root = path.resolve(targetDir);
  const resolved = path.resolve(root, name);
  if (!resolved.startsWith(root + path.sep)) {
    throw new Error(`Model asset path escapes the cache directory: ${JSON.stringify(name)}`);
  }
  return resolved;
}

/** sha256 of a regular file; null if missing or not a regular file (e.g. a symlink). */
function regularFileHash(filePath: string): string | null {
  let st: fs.Stats;
  try {
    st = fs.lstatSync(filePath);
  } catch {
    return null;
  }
  if (!st.isFile()) return null;
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

/** Write via a fresh temp file and rename, so an existing symlink at filePath is replaced, never followed. */
function writeFileNoFollow(filePath: string, data: Buffer): void {
  const tmp = `${filePath}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.tmp`;
  const flags = fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL
    | (fs.constants.O_NOFOLLOW ?? 0);
  const fd = fs.openSync(tmp, flags, 0o600);
  try {
    fs.writeSync(fd, data);
  } finally {
    fs.closeSync(fd);
  }
  try {
    fs.renameSync(tmp, filePath);
  } catch (error) {
    fs.rmSync(tmp, { force: true });
    throw error;
  }
}

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

    // 4. Check previously extracted cache dir. No pinned hash manifest ships
    // for these models, so only the directory's ownership is checked here;
    // step 2 re-verifies every asset against the bundle's SHA-256 values.
    const cacheDir = getModelCacheDir();
    const cachedOnnx = path.join(cacheDir, 'Xenova', modelId, 'onnx', 'model_quantized.onnx');
    if (ensurePrivateDir(cacheDir, false) && regularFileHash(cachedOnnx) !== null) {
      return { localPath: cacheDir, fromBundle: true };
    }

    return null;
  }

  /**
   * Extract model files from a .rvf bundle to a temp directory.
   * Skips files whose on-disk checksum already matches.
   */
  static async extractFromRvf(rvfPath: string, modelId: string): Promise<string> {
    validateModelId(modelId);
    const cacheDir = getModelCacheDir();
    ensurePrivateDir(cacheDir, true);
    const targetDir = path.join(cacheDir, 'Xenova', modelId);

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

        const filePath = resolveAssetPath(targetDir, name);

        // Skip if a regular file with a matching checksum is already there
        if (regularFileHash(filePath) === expectedHash) continue;

        // Verify blob checksum in-memory before writing (avoids double disk read)
        const blobBuffer = Buffer.from(blob);
        const blobHash = crypto.createHash('sha256').update(blobBuffer).digest('hex');
        if (blobHash !== expectedHash) {
          throw new Error(`SHA-256 mismatch for ${name}: expected ${expectedHash}, got ${blobHash}`);
        }

        fs.mkdirSync(path.dirname(filePath), { recursive: true, mode: 0o700 });
        writeFileNoFollow(filePath, blobBuffer);
      }
    } finally {
      db.close();
    }

    return cacheDir;
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
