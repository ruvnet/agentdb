/**
 * On the sql.js path, save() calls db.export(), which frees every prepared
 * statement. The statement cache kept handing out the freed statements, so
 * the next write threw the string "Statement closed" (seen by callers as an
 * error with message `undefined`). Reported from agentic-flow:
 * storeEpisode -> db.save() -> storeEpisode threw.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const embed = async () => new Float32Array(8).fill(0.5);
const embedder = { embedQuery: embed, embedPassage: embed, embed } as any;

describe('sql.js save() in the middle of a session', () => {
  let dir: string;
  const saved = process.env.AGENTDB_FORCE_SQLJS;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentdb-sqljs-save-'));
    process.env.AGENTDB_FORCE_SQLJS = '1';
  });

  afterEach(() => {
    if (saved === undefined) delete process.env.AGENTDB_FORCE_SQLJS;
    else process.env.AGENTDB_FORCE_SQLJS = saved;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('keeps cached statements usable after save()', async () => {
    const { createDatabase } = await import('../../src/db-fallback.js');
    const { SCHEMA_SQL } = await import('../../src/schemas/inline.js');
    const { ReflexionMemory } = await import('../../src/controllers/ReflexionMemory.js');
    const file = path.join(dir, 'agentdb.db');
    const db = await createDatabase(file);
    db.exec(SCHEMA_SQL);
    const memory = new ReflexionMemory(db, embedder);

    await memory.storeEpisode({ sessionId: 's', task: 'first', reward: 1, success: true });
    db.save();
    await memory.storeEpisode({ sessionId: 's', task: 'second', reward: 1, success: true });
    db.save();
    db.close();

    const reopened = await createDatabase(file);
    expect((reopened.prepare('SELECT COUNT(*) AS n FROM episodes').get() as any).n).toBe(2);
    reopened.close();
  });
});
