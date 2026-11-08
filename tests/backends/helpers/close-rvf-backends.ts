/**
 * Close every RvfBackend a test initialized, after each test.
 *
 * Tests that construct backends inline and never close them leave in-memory
 * stores (and their temp directories) open until the worker exits. This
 * records each initialized instance and closes it in afterEach.
 */
import { afterEach, expect } from 'vitest';
import { RvfBackend, liveRvfTempDirs } from '../../../src/backends/rvf/RvfBackend.js';

export function closeRvfBackendsAfterEach(): void {
  const opened = new Set<RvfBackend>();
  const proto = RvfBackend.prototype as unknown as { initialize(this: RvfBackend): Promise<void> };
  const initialize = proto.initialize;
  proto.initialize = async function (this: RvfBackend) {
    opened.add(this);
    return initialize.call(this);
  };
  afterEach(async () => {
    for (const backend of opened) {
      await backend.closeAsync().catch(() => backend.close());
    }
    opened.clear();
    expect([...liveRvfTempDirs()]).toEqual([]);
  });
}
