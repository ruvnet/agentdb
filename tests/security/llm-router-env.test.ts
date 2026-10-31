/**
 * LLMRouter's constructor used to read <cwd>/.env, <cwd>/../../.env and
 * /workspaces/agentic-flow/.env into process.env. Constructing a router in
 * an untrusted directory must not load anything.
 */
import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { LLMRouter } from '../../src/services/LLMRouter.js';

const KEY = 'AGENTDB_TEST_IMPLICIT_ENV_KEY';

describe('LLMRouter env loading', () => {
  const cwd = process.cwd();
  let dir: string | undefined;

  afterEach(() => {
    process.chdir(cwd);
    delete process.env[KEY];
    if (dir) fs.rmSync(dir, { recursive: true, force: true });
  });

  it('does not load <cwd>/.env implicitly', () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-llm-env-'));
    fs.writeFileSync(path.join(dir, '.env'), `${KEY}=planted\n`);
    process.chdir(dir);

    new LLMRouter({ provider: 'onnx' });

    expect(process.env[KEY]).toBeUndefined();
  });

  it('loads an explicit envFile without overriding existing values', () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-llm-env-'));
    const file = path.join(dir, 'keys.env');
    fs.writeFileSync(file, `${KEY}=from-file\n`);

    new LLMRouter({ provider: 'onnx', envFile: file });

    expect(process.env[KEY]).toBe('from-file');
  });
});
