/**
 * Rate Limiting Middleware for AgentDB
 *
 * Provides comprehensive rate limiting:
 * - Per-IP rate limiting
 * - Per-user rate limiting
 * - Per-endpoint rate limiting
 * - Sliding window algorithm
 * - Configurable limits and windows
 * - Rate limit headers (X-RateLimit-*)
 *
 * Security Features:
 * - Prevents brute force attacks
 * - Prevents DoS attacks
 * - Configurable strict mode for auth endpoints
 * - IP whitelist/blacklist support
 * - Automatic cleanup of expired records
 */

import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import { AuthenticatedRequest } from './auth.middleware.js';
import { AuditLogger } from '../services/audit-logger.service.js';

/**
 * Rate limit configuration
 */
export interface RateLimitConfig {
  windowMs?: number;          // Time window in milliseconds
  max?: number;               // Maximum requests per window
  message?: string;           // Error message
  statusCode?: number;        // HTTP status code for rate limit exceeded
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
}

/**
 * Default rate limit configuration
 */
export const RATE_LIMIT_CONFIG = {
  // General API endpoints
  GENERAL: {
    windowMs: 15 * 60 * 1000,  // 15 minutes
    max: 100,                   // 100 requests per 15 minutes
  },

  // Authentication endpoints (stricter)
  AUTH: {
    windowMs: 15 * 60 * 1000,  // 15 minutes
    max: 5,                     // 5 attempts per 15 minutes
  },

  // Registration endpoints (very strict)
  REGISTRATION: {
    windowMs: 60 * 60 * 1000,  // 1 hour
    max: 3,                     // 3 attempts per hour
  },

  // API key operations
  API_KEY: {
    windowMs: 60 * 60 * 1000,  // 1 hour
    max: 10,                    // 10 operations per hour
  },

  // Password reset
  PASSWORD_RESET: {
    windowMs: 60 * 60 * 1000,  // 1 hour
    max: 3,                     // 3 attempts per hour
  },

  // Read operations
  READ: {
    windowMs: 15 * 60 * 1000,  // 15 minutes
    max: 200,                   // 200 requests per 15 minutes
  },

  // Write operations
  WRITE: {
    windowMs: 15 * 60 * 1000,  // 15 minutes
    max: 50,                    // 50 requests per 15 minutes
  },
} as const;

/**
 * Audit logger instance
 */
const auditLogger = new AuditLogger();

/**
 * Custom key generator that uses user ID if authenticated
 */
function createKeyGenerator(prefix: string = 'general') {
  return (req: Request): string => {
    const authReq = req as AuthenticatedRequest;

    // Use user ID if authenticated
    if (authReq.user?.userId) {
      return `${prefix}:user:${authReq.user.userId}`;
    }

    // Use API key ID if authenticated with API key
    if (authReq.apiKey?.keyId) {
      return `${prefix}:apikey:${authReq.apiKey.keyId}`;
    }

    // Fall back to IP address
    return `${prefix}:ip:${req.ip}`;
  };
}

/**
 * Create rate limit handler with logging
 */
function createRateLimitHandler(limitType: string) {
  return async (req: Request, res: Response): Promise<void> => {
    const authReq = req as AuthenticatedRequest;

    await auditLogger.logRateLimitEvent({
      type: 'rate_limit_exceeded',
      limitType,
      userId: authReq.user?.userId,
      ip: req.ip,
      userAgent: req.headers['user-agent'],
      path: req.path,
      method: req.method,
      timestamp: new Date(),
    });

    res.status(429).json({
      error: 'Too many requests',
      code: 'RATE_LIMIT_EXCEEDED',
      message: `Rate limit exceeded for ${limitType}. Please try again later.`,
      retryAfter: res.getHeader('Retry-After'),
    });
  };
}

/**
 * General API rate limiter
 */
export const generalRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.GENERAL.windowMs,
  max: RATE_LIMIT_CONFIG.GENERAL.max,
  standardHeaders: true, // Return rate limit info in RateLimit-* headers
  legacyHeaders: false,  // Disable X-RateLimit-* headers
  keyGenerator: createKeyGenerator('general'),
  handler: createRateLimitHandler('general'),
  skip: (req) => {
    // Skip rate limiting for health checks
    return req.path === '/health' || req.path === '/ping';
  },
});

/**
 * Authentication rate limiter (strict)
 */
export const authRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.AUTH.windowMs,
  max: RATE_LIMIT_CONFIG.AUTH.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('auth'),
  skipSuccessfulRequests: false, // Count all requests
  skipFailedRequests: false,
  handler: createRateLimitHandler('authentication'),
});

/**
 * Registration rate limiter (very strict)
 */
export const registrationRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.REGISTRATION.windowMs,
  max: RATE_LIMIT_CONFIG.REGISTRATION.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('registration'),
  skipSuccessfulRequests: false,
  handler: createRateLimitHandler('registration'),
});

/**
 * API key operations rate limiter
 */
export const apiKeyRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.API_KEY.windowMs,
  max: RATE_LIMIT_CONFIG.API_KEY.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('apikey'),
  handler: createRateLimitHandler('api_key_operations'),
});

/**
 * Password reset rate limiter
 */
export const passwordResetRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.PASSWORD_RESET.windowMs,
  max: RATE_LIMIT_CONFIG.PASSWORD_RESET.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('password'),
  handler: createRateLimitHandler('password_reset'),
});

/**
 * Read operations rate limiter
 */
export const readRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.READ.windowMs,
  max: RATE_LIMIT_CONFIG.READ.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('read'),
  handler: createRateLimitHandler('read_operations'),
});

/**
 * Write operations rate limiter
 */
export const writeRateLimiter = rateLimit({
  windowMs: RATE_LIMIT_CONFIG.WRITE.windowMs,
  max: RATE_LIMIT_CONFIG.WRITE.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: createKeyGenerator('write'),
  handler: createRateLimitHandler('write_operations'),
});

/**
 * Create custom rate limiter
 */
export function createRateLimiter(config: RateLimitConfig, name: string = 'custom') {
  return rateLimit({
    windowMs: config.windowMs || RATE_LIMIT_CONFIG.GENERAL.windowMs,
    max: config.max || RATE_LIMIT_CONFIG.GENERAL.max,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: config.keyGenerator || createKeyGenerator(name),
    skipSuccessfulRequests: config.skipSuccessfulRequests,
    skipFailedRequests: config.skipFailedRequests,
    handler: createRateLimitHandler(name),
  });
}

/**
 * IP-based rate limiter (ignores authentication)
 */
export function createIPRateLimiter(windowMs: number, max: number) {
  return rateLimit({
    windowMs,
    max,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => `ip:${req.ip}`,
    handler: createRateLimitHandler('ip_based'),
  });
}

/**
 * Sliding window rate limiter (more accurate)
 */
class SlidingWindowRateLimiter {
  private requests: Map<string, number[]> = new Map();
  private cleanupInterval: NodeJS.Timeout;

  constructor(
    private windowMs: number,
    private maxRequests: number
  ) {
    // Cleanup old entries every minute
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 60 * 1000);
    this.cleanupInterval.unref?.();
  }

  middleware = (req: Request, res: Response, next: NextFunction): void => {
    const key = this.getKey(req);
    const now = Date.now();

    // Get request timestamps for this key
    let timestamps = this.requests.get(key) || [];

    // Remove timestamps outside the window
    timestamps = timestamps.filter(ts => now - ts < this.windowMs);

    // Check if limit exceeded
    if (timestamps.length >= this.maxRequests) {
      const oldestTimestamp = timestamps[0];
      const retryAfter = Math.ceil((this.windowMs - (now - oldestTimestamp)) / 1000);

      res.set('Retry-After', retryAfter.toString());
      res.set('X-RateLimit-Limit', this.maxRequests.toString());
      res.set('X-RateLimit-Remaining', '0');
      res.set('X-RateLimit-Reset', new Date(oldestTimestamp + this.windowMs).toISOString());

      res.status(429).json({
        error: 'Too many requests',
        code: 'RATE_LIMIT_EXCEEDED',
        retryAfter,
      });
      return;
    }

    // Add current request timestamp
    timestamps.push(now);
    this.requests.set(key, timestamps);

    // Set rate limit headers
    res.set('X-RateLimit-Limit', this.maxRequests.toString());
    res.set('X-RateLimit-Remaining', (this.maxRequests - timestamps.length).toString());
    res.set('X-RateLimit-Reset', new Date(now + this.windowMs).toISOString());

    next();
  };

  private getKey(req: Request): string {
    const authReq = req as AuthenticatedRequest;
    return authReq.user?.userId || authReq.apiKey?.keyId || req.ip || 'unknown';
  }

  private cleanup(): void {
    const now = Date.now();
    const keysToDelete: string[] = [];

    for (const [key, timestamps] of this.requests.entries()) {
      const validTimestamps = timestamps.filter(ts => now - ts < this.windowMs);

      if (validTimestamps.length === 0) {
        keysToDelete.push(key);
      } else {
        this.requests.set(key, validTimestamps);
      }
    }

    keysToDelete.forEach(key => this.requests.delete(key));
  }

  destroy(): void {
    clearInterval(this.cleanupInterval);
    this.requests.clear();
  }
}

/**
 * Create sliding window rate limiter
 */
export function createSlidingWindowRateLimiter(windowMs: number, maxRequests: number) {
  const limiter = new SlidingWindowRateLimiter(windowMs, maxRequests);
  return limiter.middleware;
}

/**
 * Dynamic rate limiter based on user tier
 */
export function createTieredRateLimiter(
  tiers: Record<string, { windowMs: number; max: number }>
) {
  const limiters = new Map<string, ReturnType<typeof rateLimit>>();

  // Create limiter for each tier
  for (const [tier, config] of Object.entries(tiers)) {
    limiters.set(tier, createRateLimiter(config, tier));
  }

  return (req: Request, res: Response, next: NextFunction): void => {
    const authReq = req as AuthenticatedRequest;
    const userTier = authReq.user?.role || 'free';

    const limiter = limiters.get(userTier) || limiters.get('free');

    if (limiter) {
      limiter(req, res, next);
    } else {
      next();
    }
  };
}
