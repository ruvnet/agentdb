/**
 * AgentDB v2 Path Security Utilities
 *
 * Prevents path traversal attacks and ensures safe file operations:
 * - Path validation and canonicalization
 * - Symlink detection and handling
 * - Safe file read/write operations
 * - Temporary file cleanup
 */

import * as path from 'path';
import * as fs from 'fs';
import { randomBytes } from 'crypto';
import { SecurityError } from './limits.js';

/**
 * Validate and sanitize file path
 * Prevents path traversal attacks
 */
export function validatePath(filePath: string, baseDir: string): string {
  if (!filePath || typeof filePath !== 'string') {
    throw new SecurityError(
      'File path must be a non-empty string',
      'INVALID_PATH'
    );
  }

  if (!baseDir || typeof baseDir !== 'string') {
    throw new SecurityError(
      'Base directory must be a non-empty string',
      'INVALID_BASE_DIR'
    );
  }

  // Resolve to absolute paths
  const resolvedBase = path.resolve(baseDir);
  const resolvedPath = path.resolve(baseDir, filePath);

  // Calculate relative path
  const relativePath = path.relative(resolvedBase, resolvedPath);

  // Check for path traversal attempts
  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new SecurityError(
      `Path traversal attempt detected: ${filePath}`,
      'PATH_TRAVERSAL'
    );
  }

  // Additional security checks
  if (filePath.includes('\x00')) {
    throw new SecurityError(
      'Path contains null bytes',
      'NULL_BYTE_IN_PATH'
    );
  }

  return resolvedPath;
}

/**
 * Check if path is a symbolic link
 */
export async function isSymbolicLink(filePath: string): Promise<boolean> {
  try {
    const stats = await fs.promises.lstat(filePath);
    return stats.isSymbolicLink();
  } catch (error) {
    // File doesn't exist
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

/**
 * Secure file write operation
 * Prevents writing to symbolic links and validates paths
 */
export async function secureWrite(
  filePath: string,
  data: Buffer | string,
  baseDir: string,
  options?: { overwrite?: boolean; encoding?: BufferEncoding }
): Promise<void> {
  const safePath = validatePath(filePath, baseDir);

  // Check if file exists and is a symlink
  if (await isSymbolicLink(safePath)) {
    throw new SecurityError(
      'Cannot write to symbolic link',
      'SYMLINK_WRITE_DENIED'
    );
  }

  // Check if file exists and overwrite is not allowed
  if (!options?.overwrite) {
    try {
      await fs.promises.access(safePath, fs.constants.F_OK);
      throw new SecurityError(
        'File already exists and overwrite is not allowed',
        'FILE_EXISTS'
      );
    } catch (error) {
      // File doesn't exist, which is what we want
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error;
      }
    }
  }

  // Ensure directory exists
  const dir = path.dirname(safePath);
  await fs.promises.mkdir(dir, { recursive: true });

  // Write file with atomic operation (write to temp, then rename)
  const tempPath = `${safePath}.tmp.${Date.now()}`;

  try {
    if (options?.encoding && typeof data === 'string') {
      await fs.promises.writeFile(tempPath, data, { encoding: options.encoding });
    } else {
      await fs.promises.writeFile(tempPath, data);
    }

    // Atomic rename
    await fs.promises.rename(tempPath, safePath);
  } catch (error) {
    // Clean up temp file on error
    try {
      await fs.promises.unlink(tempPath);
    } catch {
      // Ignore cleanup errors
    }
    throw error;
  }
}

/**
 * Secure file read operation
 * Validates paths and prevents symlink attacks
 */
export async function secureRead(
  filePath: string,
  baseDir: string,
  options?: { encoding?: BufferEncoding; followSymlinks?: boolean }
): Promise<Buffer | string> {
  const safePath = validatePath(filePath, baseDir);

  // Check for symlinks if not allowed
  if (!options?.followSymlinks && await isSymbolicLink(safePath)) {
    throw new SecurityError(
      'Cannot read symbolic link',
      'SYMLINK_READ_DENIED'
    );
  }

  // Verify file exists and is readable
  try {
    await fs.promises.access(safePath, fs.constants.R_OK);
  } catch (error) {
    throw new SecurityError(
      `File not found or not readable: ${path.basename(filePath)}`,
      'FILE_NOT_READABLE'
    );
  }

  // Read file
  if (options?.encoding) {
    return await fs.promises.readFile(safePath, { encoding: options.encoding });
  } else {
    return await fs.promises.readFile(safePath);
  }
}

/**
 * Secure directory listing
 * Prevents path traversal and filters out sensitive files
 */
export async function secureListDir(
  dirPath: string,
  baseDir: string,
  options?: { recursive?: boolean; includeDotFiles?: boolean }
): Promise<string[]> {
  const safeDir = validatePath(dirPath, baseDir);

  // Verify directory exists
  try {
    const stats = await fs.promises.stat(safeDir);
    if (!stats.isDirectory()) {
      throw new SecurityError(
        'Path is not a directory',
        'NOT_A_DIRECTORY'
      );
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new SecurityError(
        'Directory not found',
        'DIRECTORY_NOT_FOUND'
      );
    }
    throw error;
  }

  const entries = await fs.promises.readdir(safeDir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    // Skip dot files unless explicitly included
    if (!options?.includeDotFiles && entry.name.startsWith('.')) {
      continue;
    }

    const fullPath = path.join(dirPath, entry.name);

    if (entry.isFile()) {
      files.push(fullPath);
    } else if (entry.isDirectory() && options?.recursive) {
      const subFiles = await secureListDir(fullPath, baseDir, options);
      files.push(...subFiles);
    }
  }

  return files;
}

/**
 * Secure file deletion
 * Validates paths and prevents symlink attacks
 */
export async function secureDelete(
  filePath: string,
  baseDir: string,
  options?: { force?: boolean }
): Promise<void> {
  const safePath = validatePath(filePath, baseDir);

  // Check if file is a symlink
  if (await isSymbolicLink(safePath)) {
    if (!options?.force) {
      throw new SecurityError(
        'Cannot delete symbolic link without force option',
        'SYMLINK_DELETE_DENIED'
      );
    }
    // Delete the symlink itself, not the target
    await fs.promises.unlink(safePath);
    return;
  }

  // Delete file
  try {
    await fs.promises.unlink(safePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      // File doesn't exist, which is fine
      return;
    }
    throw error;
  }
}

/**
 * Temporary file manager with automatic cleanup
 */
export class TempFileManager {
  private tempFiles: Set<string> = new Set();
  private tempDir: string;
  private cleanupScheduled: boolean = false;

  constructor(baseDir: string) {
    this.tempDir = path.join(baseDir, '.tmp');
  }

  /**
   * Initialize temp directory
   */
  async init(): Promise<void> {
    await fs.promises.mkdir(this.tempDir, { recursive: true });

    // Schedule cleanup on process exit
    if (!this.cleanupScheduled) {
      process.on('exit', () => this.cleanupSync());
      process.on('SIGINT', () => {
        this.cleanupSync();
        process.exit(0);
      });
      process.on('SIGTERM', () => {
        this.cleanupSync();
        process.exit(0);
      });
      this.cleanupScheduled = true;
    }
  }

  /**
   * Create a temporary file
   */
  async createTempFile(prefix: string = 'agentdb'): Promise<string> {
    await this.init();

    // Use cryptographically secure random bytes to prevent temp-file prediction attacks
    const filename = `${prefix}-${Date.now()}-${randomBytes(8).toString('hex')}`;
    const tempPath = path.join(this.tempDir, filename);

    this.tempFiles.add(tempPath);
    return tempPath;
  }

  /**
   * Write to temporary file
   */
  async writeTempFile(
    data: Buffer | string,
    prefix: string = 'agentdb'
  ): Promise<string> {
    const tempPath = await this.createTempFile(prefix);
    await fs.promises.writeFile(tempPath, data);
    return tempPath;
  }

  /**
   * Delete a specific temp file
   */
  async deleteTempFile(tempPath: string): Promise<void> {
    if (!this.tempFiles.has(tempPath)) {
      throw new SecurityError(
        'File is not managed by this temp file manager',
        'NOT_TEMP_FILE'
      );
    }

    try {
      await fs.promises.unlink(tempPath);
      this.tempFiles.delete(tempPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        throw error;
      }
    }
  }

  /**
   * Clean up all temporary files
   */
  async cleanup(): Promise<void> {
    const deletePromises = Array.from(this.tempFiles).map(async (tempPath) => {
      try {
        await fs.promises.unlink(tempPath);
      } catch (error) {
        // Ignore errors during cleanup
        console.warn(`Failed to delete temp file: ${tempPath}`, error);
      }
    });

    await Promise.all(deletePromises);
    this.tempFiles.clear();

    // Try to remove temp directory if empty
    try {
      await fs.promises.rmdir(this.tempDir);
    } catch {
      // Directory not empty or doesn't exist, which is fine
    }
  }

  /**
   * Synchronous cleanup for process exit
   */
  private cleanupSync(): void {
    for (const tempPath of this.tempFiles) {
      try {
        fs.unlinkSync(tempPath);
      } catch {
        // Ignore errors during cleanup
      }
    }

    try {
      fs.rmdirSync(this.tempDir);
    } catch {
      // Directory not empty or doesn't exist
    }
  }

  /**
   * Get list of managed temp files
   */
  getTempFiles(): string[] {
    return Array.from(this.tempFiles);
  }
}

/**
 * Ensure directory exists with safe permissions
 */
export async function ensureDir(
  dirPath: string,
  baseDir: string
): Promise<string> {
  const safeDir = validatePath(dirPath, baseDir);

  await fs.promises.mkdir(safeDir, {
    recursive: true,
    mode: 0o755, // rwxr-xr-x
  });

  return safeDir;
}

/**
 * Get safe file stats without following symlinks
 */
export async function safeStats(
  filePath: string,
  baseDir: string
): Promise<fs.Stats> {
  const safePath = validatePath(filePath, baseDir);
  return await fs.promises.lstat(safePath);
}

/**
 * Check if path exists within base directory
 */
export async function pathExists(
  filePath: string,
  baseDir: string
): Promise<boolean> {
  try {
    const safePath = validatePath(filePath, baseDir);
    await fs.promises.access(safePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}
