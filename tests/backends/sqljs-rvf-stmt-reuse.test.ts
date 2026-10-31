/**
 * SqlJsRvfBackend reuses one prepared upsert statement. sql.js frees every
 * prepared statement on export(), so writes after save() must re-prepare.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { SqlJsRvfBackend } from '../../src/backends/rvf/SqlJsRvfBackend.js';

describe('SqlJsRvfBackend upsert statement reuse', () => {
  it('keeps writing after save() and persists every row', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-sqljs-stmt-'));
    const file = path.join(dir, 'store.rvf');
    try {
      const backend = new SqlJsRvfBackend({ dimension: 4, metric: 'cosine', storagePath: file } as any);
      await backend.initialize();
      await backend.insertAsync('a', new Float32Array([1, 0, 0, 0]));
      await backend.save(file);
      await backend.insertAsync('b', new Float32Array([0, 1, 0, 0]));
      await backend.insertAsync('a', new Float32Array([0, 0, 1, 0])); // upsert
      backend.close();

      const reopened = new SqlJsRvfBackend({ dimension: 4, metric: 'cosine', storagePath: file } as any);
      await reopened.initialize();
      expect(reopened.getStats().count).toBe(2);
      const [top] = await reopened.searchAsync(new Float32Array([0, 0, 1, 0]), 1);
      expect(top.id).toBe('a');
      reopened.close();
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
