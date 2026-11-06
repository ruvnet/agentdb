/**
 * Speed gates for benchmark and simulation suites (#44).
 *
 * Absolute timing thresholds (ops/sec, ms per call, "faster than a fixed
 * cost model") depend on the machine and on its load, so they are skipped
 * under CI or when AGENTDB_SKIP_SPEED_GATES=1. Functional assertions (shape,
 * correctness, finite values) must stay unconditional; only wrap the timing
 * check itself in `if (SPEED_GATES)` or declare the whole test with itSpeed.
 *
 * BENCH_FULL (AGENTDB_BENCH_FULL=1) restores the original problem sizes of
 * the long-running suites; by default they use small sizes so each file
 * finishes in well under two minutes.
 */

import { it } from 'vitest';

export const SPEED_GATES = !process.env.CI && process.env.AGENTDB_SKIP_SPEED_GATES !== '1';

export const itSpeed = SPEED_GATES ? it : it.skip;

export const BENCH_FULL = process.env.AGENTDB_BENCH_FULL === '1';

/** Pick the original value under AGENTDB_BENCH_FULL=1, the small one otherwise. */
export function benchSize<T>(full: T, small: T): T {
  return BENCH_FULL ? full : small;
}
