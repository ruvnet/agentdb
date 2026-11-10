/**
 * README "Quick Start" Paths A and B run as written against the built package.
 * Path A is extracted from README.md, so the README cannot drift from the API.
 * Path B also guards the CLI write commands, which exited 1 with
 * "this.db.save is not a function" on native better-sqlite3 (fixed in 3.8.2).
 */
import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = resolve('.');
const cli = join(root, 'dist/src/cli/agentdb-cli.js');
const selfLearning = join(root, 'dist/src/backends/rvf/SelfLearningRvfBackend.js');
const built = existsSync(cli) && existsSync(selfLearning);
const readme = readFileSync('README.md', 'utf8');
const quickStart = readme.slice(readme.indexOf('## Quick Start'), readme.indexOf('### Path C'));

function block(after: string, lang: string): string {
  const start = quickStart.indexOf(after);
  expect(start).toBeGreaterThan(-1);
  const m = new RegExp('```' + lang + '\\n([\\s\\S]*?)```').exec(quickStart.slice(start));
  expect(m).not.toBeNull();
  return m![1];
}

function env() {
  const e = { ...process.env };
  delete e.AGENTDB_FORCE_SQLJS;
  return e;
}

describe.skipIf(!built)('README quick start', () => {
  it('Path A (npm library) runs as written', () => {
    const code = block('### Path A', 'typescript').replace(
      "'agentdb/backends/self-learning'",
      JSON.stringify(pathToFileURL(selfLearning).href),
    );
    expect(code).toContain('SelfLearningRvfBackend.create');
    const dir = mkdtempSync(join(tmpdir(), 'agentdb-qs-a-'));
    try {
      writeFileSync(join(dir, 'qs.mjs'), `${code}\nconsole.log('HIT', hits[0].id);\n`);
      const r = spawnSync(process.execPath, ['qs.mjs'], { cwd: dir, encoding: 'utf8', env: env(), timeout: 120_000 });
      expect(r.stderr).not.toMatch(/SyntaxError|TypeError/);
      expect(r.status).toBe(0);
      expect(r.stdout).toContain('HIT doc1');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 150_000);

  it('Path B (CLI) commands exit 0 and round-trip the episode', () => {
    const lines = block('### Path B', 'bash')
      .split('\n')
      .filter(l => l.startsWith('npx agentdb '));
    expect(lines.length).toBeGreaterThanOrEqual(3);
    const dir = mkdtempSync(join(tmpdir(), 'agentdb-qs-b-'));
    try {
      let last = '';
      for (const line of lines) {
        const args = (line.slice('npx agentdb '.length).match(/"[^"]*"|\S+/g) ?? []).map(a => a.replace(/^"|"$/g, ''));
        const r = spawnSync(process.execPath, [cli, ...args], { cwd: dir, encoding: 'utf8', env: env(), timeout: 120_000 });
        expect(`${r.stdout}${r.stderr}`).not.toContain('is not a function');
        expect(r.status, `${line}\n${r.stderr}`).toBe(0);
        last = r.stdout;
      }
      expect(last).toContain('Used OAuth2 with PKCE');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 300_000);
});
