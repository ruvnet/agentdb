/**
 * ModelCacheLoader extracted bundled models into the shared
 * os.tmpdir()/agentdb-models, followed symlinks when writing, and joined
 * asset filenames from the .rvf without checking they stayed in the target
 * directory.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as crypto from 'crypto';
import { ModelCacheLoader } from '../../src/model/ModelCacheLoader.js';

async function writeRvf(file: string, assets: Record<string, string>): Promise<void> {
  const SQL = await (await import('sql.js')).default();
  const db = new SQL.Database();
  db.run('CREATE TABLE model_assets (filename TEXT PRIMARY KEY, content BLOB NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL)');
  for (const [name, text] of Object.entries(assets)) {
    const buf = Buffer.from(text);
    db.run('INSERT INTO model_assets VALUES (?, ?, ?, ?)', [name, buf, buf.length, crypto.createHash('sha256').update(buf).digest('hex')]);
  }
  fs.writeFileSync(file, Buffer.from(db.export()));
  db.close();
}

describe('ModelCacheLoader cache hardening', () => {
  let dir: string;
  const savedXdg = process.env.XDG_CACHE_HOME;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-model-sec-'));
    process.env.XDG_CACHE_HOME = path.join(dir, 'cache');
  });

  afterEach(() => {
    if (savedXdg === undefined) delete process.env.XDG_CACHE_HOME;
    else process.env.XDG_CACHE_HOME = savedXdg;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('extracts into a private per-user directory, not the shared tmpdir', async () => {
    const rvf = path.join(dir, 'm.rvf');
    await writeRvf(rvf, { 'config.json': '{}' });

    const base = await ModelCacheLoader.extractFromRvf(rvf, 'sec-model');

    expect(base).toBe(path.join(dir, 'cache', 'agentdb-models'));
    expect(base.startsWith(os.tmpdir() + path.sep + 'agentdb-models')).toBe(false);
    if (process.platform !== 'win32') {
      expect(fs.statSync(base).mode & 0o777).toBe(0o700);
    }
  });

  it('refuses a cache directory that is a symlink', async () => {
    const elsewhere = path.join(dir, 'elsewhere');
    fs.mkdirSync(elsewhere);
    fs.mkdirSync(path.join(dir, 'cache'));
    fs.symlinkSync(elsewhere, path.join(dir, 'cache', 'agentdb-models'));
    const rvf = path.join(dir, 'm.rvf');
    await writeRvf(rvf, { 'config.json': '{}' });

    await expect(ModelCacheLoader.extractFromRvf(rvf, 'sec-model')).rejects.toThrow(/not a real directory/);
    expect(fs.readdirSync(elsewhere)).toEqual([]);
  });

  it('rejects asset filenames that escape the target directory', async () => {
    // (A NUL in a TEXT filename is truncated by SQLite before it reaches us;
    // the loader still rejects one.)
    for (const name of ['../../escaped.txt', path.join(dir, 'abs.txt')]) {
      const rvf = path.join(dir, 'evil.rvf');
      await writeRvf(rvf, { [name]: 'pwned' });
      await expect(ModelCacheLoader.extractFromRvf(rvf, 'sec-model')).rejects.toThrow(/Invalid model asset filename|escapes/);
    }
    expect(fs.existsSync(path.join(dir, 'cache', 'agentdb-models', 'escaped.txt'))).toBe(false);
    expect(fs.existsSync(path.join(dir, 'abs.txt'))).toBe(false);
  });

  it('replaces a planted symlink instead of writing through it', async () => {
    const victim = path.join(dir, 'victim.txt');
    fs.writeFileSync(victim, 'original');
    const target = path.join(dir, 'cache', 'agentdb-models', 'Xenova', 'sec-model');
    fs.mkdirSync(target, { recursive: true, mode: 0o700 });
    fs.symlinkSync(victim, path.join(target, 'config.json'));
    const rvf = path.join(dir, 'm.rvf');
    await writeRvf(rvf, { 'config.json': '{"real":true}' });

    await ModelCacheLoader.extractFromRvf(rvf, 'sec-model');

    expect(fs.readFileSync(victim, 'utf8')).toBe('original');
    expect(fs.lstatSync(path.join(target, 'config.json')).isSymbolicLink()).toBe(false);
    expect(fs.readFileSync(path.join(target, 'config.json'), 'utf8')).toBe('{"real":true}');
  });
});
