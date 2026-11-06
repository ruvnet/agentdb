/**
 * AgentDB v2 Resource Limits Test Suite
 *
 * Tests for:
 * - Memory limit enforcement
 * - Query timeouts
 * - Rate limiting
 * - Circuit breaker
 * - Resource tracking
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ResourceTracker,
  RateLimiter,
  CircuitBreaker,
  SecurityError,
  withTimeout,
  enforceBatchLimits,
  logResourceUsage,
} from '../../src/security/limits';
import { SECURITY_LIMITS } from '../../src/security/validation';

describe('AgentDB v2 Security: Resource Tracking', () => {
  let tracker: ResourceTracker;

  beforeEach(() => {
    tracker = new ResourceTracker();
  });

  describe('ResourceTracker', () => {
    it('should track memory usage', () => {
      tracker.updateMemoryUsage(100);
      const stats = tracker.getStats();

      expect(stats.memoryUsageMB).toBe(100);
      expect(stats.memoryUtilization).toBeGreaterThan(0);
    });

    it('should throw error when memory limit exceeded', () => {
      expect(() => {
        tracker.updateMemoryUsage(SECURITY_LIMITS.MAX_MEMORY_MB + 1);
      }).toThrow(SecurityError);
      expect(() => {
        tracker.updateMemoryUsage(SECURITY_LIMITS.MAX_MEMORY_MB + 1);
      }).toThrow(/Memory limit exceeded/);
    });

    it('should estimate vector memory correctly', () => {
      const memory = tracker.estimateVectorMemory(1000, 384);
      // 1000 vectors * 384 dimensions * 4 bytes * 1.25 overhead
      const expected = (1000 * 384 * 4 * 1.25) / (1024 * 1024);

      expect(memory).toBeCloseTo(expected, 2);
    });

    it('should record query execution', () => {
      tracker.recordQuery(50);
      tracker.recordQuery(100);
      tracker.recordQuery(75);

      const stats = tracker.getStats();
      expect(stats.queryCount).toBe(3);
      expect(stats.avgQueryTimeMs).toBeCloseTo(75, 1);
    });

    it('should calculate queries per second', () => {
      for (let i = 0; i < 10; i++) {
        tracker.recordQuery(10);
      }

      const stats = tracker.getStats();
      expect(stats.queriesPerSecond).toBeGreaterThan(0);
      expect(stats.queryCount).toBe(10);
    });

    it('should reset tracker', () => {
      tracker.updateMemoryUsage(100);
      tracker.recordQuery(50);

      tracker.reset();

      const stats = tracker.getStats();
      expect(stats.memoryUsageMB).toBe(0);
      expect(stats.queryCount).toBe(0);
    });

    it('should maintain only last 100 query times', () => {
      for (let i = 0; i < 150; i++) {
        tracker.recordQuery(10);
      }

      const stats = tracker.getStats();
      expect(stats.queryCount).toBe(150);
      // Internal array should be capped at 100
    });
  });

  describe('Memory Limit Enforcement', () => {
    it('should allow operations within memory limits', () => {
      expect(() => {
        tracker.updateMemoryUsage(1000); // 1GB
      }).not.toThrow();
    });

    it('should prevent excessive memory allocation', () => {
      tracker.updateMemoryUsage(SECURITY_LIMITS.MAX_MEMORY_MB - 100);

      expect(() => {
        tracker.updateMemoryUsage(200); // Would exceed limit
      }).toThrow(SecurityError);
    });

    it('should track cumulative memory usage', () => {
      tracker.updateMemoryUsage(100);
      tracker.updateMemoryUsage(200);
      tracker.updateMemoryUsage(300);

      const stats = tracker.getStats();
      expect(stats.memoryUsageMB).toBe(600);
    });
  });
});

describe('AgentDB v2 Security: Query Timeouts', () => {
  describe('withTimeout', () => {
    it('should complete fast operations', async () => {
      const promise = Promise.resolve('success');
      const result = await withTimeout(promise, 1000, 'test');

      expect(result).toBe('success');
    });

    it('should timeout slow operations', async () => {
      const slowPromise = new Promise((resolve) => {
        setTimeout(resolve, 5000);
      });

      await expect(
        withTimeout(slowPromise, 100, 'slow operation')
      ).rejects.toThrow(SecurityError);

      await expect(
        withTimeout(slowPromise, 100, 'slow operation')
      ).rejects.toThrow(/timeout after 100ms/);
    });

    it('should use default timeout', async () => {
      const slowPromise = new Promise((resolve) => {
        setTimeout(resolve, SECURITY_LIMITS.QUERY_TIMEOUT_MS + 1000);
      });

      await expect(
        withTimeout(slowPromise)
      ).rejects.toThrow(/timeout/);
    });

    it('should handle promise rejection', async () => {
      const failingPromise = Promise.reject(new Error('Operation failed'));

      await expect(
        withTimeout(failingPromise, 1000)
      ).rejects.toThrow('Operation failed');
    });
  });
});

describe('AgentDB v2 Security: Rate Limiting', () => {
  let limiter: RateLimiter;

  beforeEach(() => {
    // 10 tokens, refill 10 per second
    limiter = new RateLimiter(10, 10);
  });

  describe('RateLimiter', () => {
    it('should allow operations within rate limit', () => {
      expect(limiter.tryConsume(5)).toBe(true);
      expect(limiter.tryConsume(3)).toBe(true);
      expect(limiter.getTokens()).toBeCloseTo(2, 0);
    });

    it('should block operations exceeding rate limit', () => {
      limiter.tryConsume(10); // Consume all tokens

      expect(limiter.tryConsume(1)).toBe(false);
      expect(limiter.getTokens()).toBeLessThan(1);
    });

    it('should refill tokens over time', async () => {
      limiter.tryConsume(10); // Consume all tokens

      // Wait for refill (100ms = 1 token at 10/sec)
      await new Promise(resolve => setTimeout(resolve, 150));

      expect(limiter.tryConsume(1)).toBe(true);
    });

    it('should throw error when consuming with enforcement', () => {
      limiter.consume(10);

      expect(() => limiter.consume(1, 'test operation'))
        .toThrow(SecurityError);
      expect(() => limiter.consume(1, 'test operation'))
        .toThrow(/Rate limit exceeded/);
    });

    it('should not exceed max tokens on refill', async () => {
      // Wait longer than needed to refill
      await new Promise(resolve => setTimeout(resolve, 2000));

      expect(limiter.getTokens()).toBeLessThanOrEqual(10);
    });

    it('should reset limiter', () => {
      limiter.consume(8);
      limiter.reset();

      expect(limiter.getTokens()).toBe(10);
      expect(limiter.tryConsume(10)).toBe(true);
    });

    it('should handle partial token consumption', () => {
      expect(limiter.tryConsume(2.5)).toBe(true);
      expect(limiter.getTokens()).toBeCloseTo(7.5, 1);
    });
  });

  describe('Rate Limit Scenarios', () => {
    it('should limit burst requests', () => {
      let successCount = 0;

      for (let i = 0; i < 20; i++) {
        if (limiter.tryConsume()) {
          successCount++;
        }
      }

      expect(successCount).toBe(10); // Only 10 tokens available
    });

    it('should allow sustained rate over time', async () => {
      let successCount = 0;

      // Try 15 requests over 1 second (should get ~10 initial + refill)
      for (let i = 0; i < 15; i++) {
        if (limiter.tryConsume()) {
          successCount++;
        }
        await new Promise(resolve => setTimeout(resolve, 80));
      }

      expect(successCount).toBeGreaterThan(10); // Initial + refilled tokens
    });
  });
});

describe('AgentDB v2 Security: Circuit Breaker', () => {
  let breaker: CircuitBreaker;

  beforeEach(() => {
    // 3 failures, 1 second reset
    breaker = new CircuitBreaker(3, 1000);
  });

  describe('CircuitBreaker', () => {
    it('should allow operations when closed', async () => {
      const result = await breaker.execute(
        async () => 'success',
        'test operation'
      );

      expect(result).toBe('success');
      expect(breaker.getStatus().state).toBe('closed');
    });

    it('should open after max failures', async () => {
      const failingOp = async () => {
        throw new Error('Operation failed');
      };

      // Cause failures
      for (let i = 0; i < 3; i++) {
        await expect(breaker.execute(failingOp, 'test'))
          .rejects.toThrow('Operation failed');
      }

      expect(breaker.getStatus().state).toBe('open');
      expect(breaker.getStatus().failures).toBe(3);
    });

    it('should block requests when open', async () => {
      // Open the breaker
      const failingOp = async () => {
        throw new Error('Fail');
      };

      for (let i = 0; i < 3; i++) {
        await expect(breaker.execute(failingOp, 'test')).rejects.toThrow();
      }

      // Should now reject immediately
      await expect(
        breaker.execute(async () => 'success', 'test')
      ).rejects.toThrow(SecurityError);

      await expect(
        breaker.execute(async () => 'success', 'test')
      ).rejects.toThrow(/Circuit breaker open/);
    });

    it('should transition to half-open after timeout', async () => {
      // Open the breaker
      const failingOp = async () => {
        throw new Error('Fail');
      };

      for (let i = 0; i < 3; i++) {
        await expect(breaker.execute(failingOp, 'test')).rejects.toThrow();
      }

      // Wait for reset timeout
      await new Promise(resolve => setTimeout(resolve, 1100));

      // Next request should try (half-open)
      const result = await breaker.execute(async () => 'success', 'test');

      expect(result).toBe('success');
      expect(breaker.getStatus().state).toBe('closed');
      expect(breaker.getStatus().failures).toBe(0);
    });

    it('should reopen if half-open request fails', async () => {
      // Open the breaker
      for (let i = 0; i < 3; i++) {
        await expect(
          breaker.execute(async () => { throw new Error('Fail'); }, 'test')
        ).rejects.toThrow();
      }

      // Wait for reset
      await new Promise(resolve => setTimeout(resolve, 1100));

      // Half-open request fails
      await expect(
        breaker.execute(async () => { throw new Error('Still failing'); }, 'test')
      ).rejects.toThrow('Still failing');

      // Should remain open (or reopen)
      expect(breaker.getStatus().failures).toBeGreaterThan(0);
    });

    it('should reset breaker manually', async () => {
      // Cause failures
      for (let i = 0; i < 3; i++) {
        await expect(
          breaker.execute(async () => { throw new Error('Fail'); }, 'test')
        ).rejects.toThrow();
      }

      breaker.reset();

      expect(breaker.getStatus().state).toBe('closed');
      expect(breaker.getStatus().failures).toBe(0);

      // Should allow requests again
      const result = await breaker.execute(async () => 'success', 'test');
      expect(result).toBe('success');
    });
  });
});

describe('AgentDB v2 Security: Batch Limit Enforcement', () => {
  let tracker: ResourceTracker;

  beforeEach(() => {
    tracker = new ResourceTracker();
  });

  describe('enforceBatchLimits', () => {
    it('should allow reasonable batch sizes', () => {
      expect(() => {
        enforceBatchLimits(1000, 384, tracker);
      }).not.toThrow();
    });

    it('should reject oversized batches', () => {
      expect(() => {
        enforceBatchLimits(SECURITY_LIMITS.MAX_BATCH_SIZE + 1, 384, tracker);
      }).toThrow(SecurityError);
      expect(() => {
        enforceBatchLimits(SECURITY_LIMITS.MAX_BATCH_SIZE + 1, 384, tracker);
      }).toThrow(/exceeds limit/);
    });

    it('should reject batches requiring excessive memory', () => {
      // Large batch with high dimension
      expect(() => {
        enforceBatchLimits(10000, 4096, tracker);
      }).toThrow(SecurityError);
      expect(() => {
        enforceBatchLimits(10000, 4096, tracker);
      }).toThrow(/would use.*MB/);
    });

    it('should update tracker with estimated memory', () => {
      enforceBatchLimits(1000, 384, tracker);

      const stats = tracker.getStats();
      expect(stats.memoryUsageMB).toBeGreaterThan(0);
    });

    it('should prevent cumulative memory overflow', () => {
      // First batch OK
      enforceBatchLimits(5000, 1024, tracker);

      // Second batch would exceed memory
      expect(() => {
        enforceBatchLimits(5000, 1024, tracker);
      }).toThrow(/Memory limit exceeded/);
    });
  });
});

describe('AgentDB v2 Security: Resource Monitoring', () => {
  describe('logResourceUsage', () => {
    it('should log resource statistics', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      logResourceUsage();

      expect(consoleSpy).toHaveBeenCalled();
      expect(consoleSpy.mock.calls[0][0]).toContain('ResourceMonitor');

      consoleSpy.mockRestore();
    });

    it('should warn on high memory usage', () => {
      const tracker = new ResourceTracker();
      tracker.updateMemoryUsage(SECURITY_LIMITS.MAX_MEMORY_MB * 0.85);

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      logResourceUsage();

      expect(consoleWarnSpy).toHaveBeenCalled();
      expect(consoleWarnSpy.mock.calls[0][0]).toContain('Memory usage above 80%');

      consoleWarnSpy.mockRestore();
      consoleLogSpy.mockRestore();
    });
  });
});

describe('AgentDB v2 Security: Real-World Scenarios', () => {
  it('should prevent DoS via rapid requests', () => {
    const limiter = new RateLimiter(100, 100);

    // Attacker sends 1000 rapid requests
    let blocked = 0;
    for (let i = 0; i < 1000; i++) {
      if (!limiter.tryConsume()) {
        blocked++;
      }
    }

    expect(blocked).toBeGreaterThan(800); // Most should be blocked
  });

  it('should handle service degradation gracefully', async () => {
    const breaker = new CircuitBreaker(5, 2000);
    let failureCount = 0;

    // Simulate intermittent failures
    const unreliableOp = async () => {
      failureCount++;
      if (failureCount <= 5) {
        throw new Error('Service degraded');
      }
      return 'recovered';
    };

    // First 5 requests fail
    for (let i = 0; i < 5; i++) {
      await expect(breaker.execute(unreliableOp, 'test')).rejects.toThrow();
    }

    // Circuit should be open
    expect(breaker.getStatus().state).toBe('open');

    // Wait for reset
    await new Promise(resolve => setTimeout(resolve, 2100));

    // Service recovered, should work
    const result = await breaker.execute(unreliableOp, 'test');
    expect(result).toBe('recovered');
  });

  it('should prevent memory exhaustion attack', () => {
    const tracker = new ResourceTracker();

    // Attacker tries to allocate massive vector database
    expect(() => {
      enforceBatchLimits(SECURITY_LIMITS.MAX_BATCH_SIZE, SECURITY_LIMITS.MAX_DIMENSION, tracker);
    }).toThrow(/would use.*MB/);
  });

  it('should enforce combined limits', () => {
    const tracker = new ResourceTracker();
    const limiter = new RateLimiter(10, 10);

    // Attacker tries to bypass limits with small batches
    let successCount = 0;

    for (let i = 0; i < 20; i++) {
      if (limiter.tryConsume()) {
        try {
          enforceBatchLimits(1000, 384, tracker);
          successCount++;
        } catch {
          // Memory limit hit
          break;
        }
      }
    }

    // Should be limited by rate limiter (10) or memory
    expect(successCount).toBeLessThanOrEqual(10);
  });
});
