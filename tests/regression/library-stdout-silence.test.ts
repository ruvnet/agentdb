/**
 * AgentDB wrote "[AgentDB] Using RuVector backend (native)" and
 * "[AgentDB] Initialized with ..." to stdout. Hosts that speak JSON-RPC over
 * stdio (ruflo `mcp start`) corrupt their protocol stream when a library
 * writes there. Library diagnostics go to stderr.
 */
import { describe, it, expect } from 'vitest';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

const ROOT = path.resolve(__dirname, '../..');
const TSX = path.join(ROOT, 'node_modules/.bin/tsx');

describe('AgentDB library stdout', () => {
  it('initialize, store and search write nothing to stdout', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-stdout-'));
    const script = path.join(dir, 'child.mts');
    fs.writeFileSync(script, `
      import { AgentDB } from '${ROOT}/src/core/AgentDB.ts';
      const db = new AgentDB({ dbPath: '${path.join(dir, 'agent.db')}' });
      await db.initialize();
      const memory = db.getController('reflexion');
      await memory.storeEpisode({ sessionId: 's', task: 'parse csv files', reward: 1, success: true });
      const hits = await memory.retrieveRelevant({ task: 'parse csv files', k: 3 });
      await db.close();
      process.stderr.write('done ' + hits.length + '\\n');
    `);
    try {
      const child = spawn(TSX, [script], {
        cwd: dir,
        env: { ...process.env, AGENTDB_DISABLE_TRANSFORMERS: '1' },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (d) => { stdout += d; });
      child.stderr.on('data', (d) => { stderr += d; });
      const code = await new Promise<number | null>((resolve) => child.once('exit', resolve));

      expect(stderr).toMatch(/done \d+/);
      expect(code).toBe(0);
      expect(stdout).toBe('');
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
