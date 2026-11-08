/**
 * RvfBackend.flush() must not drop queued writes when an ingest fails, and
 * concurrent flushes must not ingest the same item twice (#46).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

const ingested: string[] = [];
const ingestBatch = vi.fn();

const mockDb = {
  status: vi.fn(),
  ingestBatch: (...args: unknown[]) => ingestBatch(...args),
  close: vi.fn(),
};

vi.mock('@ruvector/rvf', () => ({
  RvfDatabase: {
    create: vi.fn(async () => mockDb),
    open: vi.fn(async () => mockDb),
  },
}));

import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';
import { closeRvfBackendsAfterEach } from './helpers/close-rvf-backends.js';

// Close every store each test opens (they are in-memory, each with a temp dir).
closeRvfBackendsAfterEach();

const vec = (x: number) => new Float32Array([x, 1, 0, 0]);

async function freshBackend(): Promise<RvfBackend> {
  const backend = new RvfBackend({ dimension: 4, storagePath: ':memory:', batchThreshold: 1000 });
  await backend.initialize();
  return backend;
}

describe('RvfBackend flush retry', () => {
  beforeEach(() => {
    ingested.length = 0;
    mockDb.status.mockResolvedValue({ totalVectors: 0 });
    mockDb.close.mockResolvedValue(undefined);
    ingestBatch.mockReset();
    ingestBatch.mockImplementation(async (entries: Array<{ id: string }>) => {
      ingested.push(...entries.map((e) => e.id));
      return { accepted: entries.length, rejected: 0, epoch: 1 };
    });
  });

  it('keeps queued items when ingestBatch rejects, and ingests them on the next flush', async () => {
    const backend = await freshBackend();
    backend.insert('a', vec(1));
    backend.insert('b', vec(2));

    ingestBatch.mockImplementationOnce(async () => {
      throw new Error('disk full');
    });

    await expect(backend.flush()).rejects.toThrow('disk full');
    expect(backend.getStats().count).toBe(2);
    expect(ingested).toEqual([]);

    await backend.flush();
    expect(ingested).toEqual(['a', 'b']);
    expect(backend.getStats().count).toBe(2);
  });

  it('does not ingest an item twice when flushes overlap', async () => {
    const backend = await freshBackend();
    backend.insert('a', vec(1));
    backend.insert('b', vec(2));

    await Promise.all([backend.flush(), backend.flush(), backend.flush()]);

    expect(ingested).toEqual(['a', 'b']);
    expect(ingestBatch).toHaveBeenCalledTimes(1);
    expect(backend.getStats().count).toBe(2);
  });

  it('a failed flush does not block later flushes', async () => {
    const backend = await freshBackend();
    backend.insert('a', vec(1));
    ingestBatch.mockImplementationOnce(async () => {
      throw new Error('transient');
    });

    const first = backend.flush();
    const second = backend.flush();
    await expect(first).rejects.toThrow('transient');
    await second;

    expect(ingested).toEqual(['a']);
  });
});
