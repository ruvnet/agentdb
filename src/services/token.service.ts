/**
 * JWT Token Service for AgentDB Authentication
 *
 * Manages JWT access and refresh tokens:
 * - Access tokens: Short-lived (15 minutes), for API requests
 * - Refresh tokens: Long-lived (7 days), for renewing access tokens
 * - Token verification and validation
 * - Token revocation support
 * - Secure secret management
 *
 * Security Features:
 * - RS256 or HS256 signing algorithms
 * - Automatic expiration handling
 * - Token rotation on refresh
 * - Revocation list support
 * - Audience and issuer validation
 */

import jwt from 'jsonwebtoken';
import { generateSecureRandomBase64 } from '../utils/crypto.utils.js';

/**
 * Token configuration
 */
export const TOKEN_CONFIG = {
  ACCESS_TOKEN_EXPIRES_IN: '15m',      // 15 minutes
  REFRESH_TOKEN_EXPIRES_IN: '7d',      // 7 days
  ALGORITHM: 'HS256' as const,         // HMAC SHA-256
  ISSUER: 'agentdb',
  AUDIENCE: 'agentdb-api',
} as const;

/**
 * Token payload interface
 */
export interface TokenPayload {
  userId: string;
  email?: string;
  role?: string;
  permissions?: string[];
  type: 'access' | 'refresh';
  sessionId?: string;
}

/**
 * Token verification result
 */
export interface TokenVerificationResult {
  valid: boolean;
  payload?: TokenPayload;
  error?: string;
  expired?: boolean;
}

/**
 * Token pair (access + refresh)
 */
export interface TokenPair {
  accessToken: string;
  refreshToken: string;
  accessTokenExpiresAt: Date;
  refreshTokenExpiresAt: Date;
}

/**
 * In-memory token revocation list (for production, use Redis)
 */
const revokedTokens = new Set<string>();

/**
 * Session-to-tokens mapping for bulk revocation
 * Maps sessionId -> Set of token JTIs
 */
const sessionTokens = new Map<string, Set<string>>();

/**
 * Token JTI to full token mapping for revocation
 */
const tokenJtiMap = new Map<string, string>();

/**
 * Get JWT secret from environment
 */
function getJWTSecret(): string {
  const secret = process.env.JWT_SECRET;

  if (!secret) {
    throw new Error(
      'JWT_SECRET environment variable is not set. ' +
      'Generate one with: node -e "console.log(require(\'crypto\').randomBytes(64).toString(\'base64\'))"'
    );
  }

  if (secret.length < 32) {
    throw new Error('JWT_SECRET must be at least 32 characters long');
  }

  return secret;
}

/**
 * Get refresh token secret from environment
 */
function getRefreshTokenSecret(): string {
  const secret = process.env.REFRESH_TOKEN_SECRET || getJWTSecret();

  if (secret.length < 32) {
    throw new Error('REFRESH_TOKEN_SECRET must be at least 32 characters long');
  }

  return secret;
}

/**
 * Create JWT access token
 */
export function createAccessToken(payload: Omit<TokenPayload, 'type'>): string {
  const secret = getJWTSecret();

  const tokenPayload: TokenPayload = {
    ...payload,
    type: 'access',
  };

  const token = jwt.sign(tokenPayload, secret, {
    expiresIn: TOKEN_CONFIG.ACCESS_TOKEN_EXPIRES_IN,
    algorithm: TOKEN_CONFIG.ALGORITHM,
    issuer: TOKEN_CONFIG.ISSUER,
    audience: TOKEN_CONFIG.AUDIENCE,
    jwtid: generateSecureRandomBase64(16),
  });

  return token;
}

/**
 * Create JWT refresh token
 */
export function createRefreshToken(payload: Omit<TokenPayload, 'type'>): string {
  const secret = getRefreshTokenSecret();

  const tokenPayload: TokenPayload = {
    ...payload,
    type: 'refresh',
  };

  const token = jwt.sign(tokenPayload, secret, {
    expiresIn: TOKEN_CONFIG.REFRESH_TOKEN_EXPIRES_IN,
    algorithm: TOKEN_CONFIG.ALGORITHM,
    issuer: TOKEN_CONFIG.ISSUER,
    audience: TOKEN_CONFIG.AUDIENCE,
    jwtid: generateSecureRandomBase64(16),
  });

  return token;
}

/**
 * Create both access and refresh tokens
 */
export function createTokenPair(
  payload: Omit<TokenPayload, 'type'>
): TokenPair {
  const accessToken = createAccessToken(payload);
  const refreshToken = createRefreshToken(payload);

  const now = new Date();

  return {
    accessToken,
    refreshToken,
    accessTokenExpiresAt: new Date(now.getTime() + 15 * 60 * 1000), // 15 minutes
    refreshTokenExpiresAt: new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000), // 7 days
  };
}

/**
 * Verify JWT token
 */
export function verifyToken(
  token: string,
  type: 'access' | 'refresh' = 'access'
): TokenVerificationResult {
  if (!token) {
    return {
      valid: false,
      error: 'Token is required',
    };
  }

  // Check revocation list
  if (revokedTokens.has(token)) {
    return {
      valid: false,
      error: 'Token has been revoked',
    };
  }

  const secret = type === 'access' ? getJWTSecret() : getRefreshTokenSecret();

  try {
    const decoded = jwt.verify(token, secret, {
      algorithms: [TOKEN_CONFIG.ALGORITHM],
      issuer: TOKEN_CONFIG.ISSUER,
      audience: TOKEN_CONFIG.AUDIENCE,
    }) as jwt.JwtPayload & TokenPayload;

    // Verify token type matches
    if (decoded.type !== type) {
      return {
        valid: false,
        error: `Invalid token type: expected ${type}, got ${decoded.type}`,
      };
    }

    return {
      valid: true,
      payload: {
        userId: decoded.userId,
        email: decoded.email,
        role: decoded.role,
        permissions: decoded.permissions,
        type: decoded.type,
        sessionId: decoded.sessionId,
      },
    };
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      return {
        valid: false,
        error: 'Token has expired',
        expired: true,
      };
    }

    if (error instanceof jwt.JsonWebTokenError) {
      return {
        valid: false,
        error: `Invalid token: ${error.message}`,
      };
    }

    return {
      valid: false,
      error: 'Token verification failed',
    };
  }
}

/**
 * Verify access token
 */
export function verifyAccessToken(token: string): TokenVerificationResult {
  return verifyToken(token, 'access');
}

/**
 * Verify refresh token
 */
export function verifyRefreshToken(token: string): TokenVerificationResult {
  return verifyToken(token, 'refresh');
}

/**
 * Refresh access token using refresh token
 */
export function refreshAccessToken(refreshToken: string): {
  success: boolean;
  accessToken?: string;
  accessTokenExpiresAt?: Date;
  error?: string;
} {
  const verification = verifyRefreshToken(refreshToken);

  if (!verification.valid || !verification.payload) {
    return {
      success: false,
      error: verification.error || 'Invalid refresh token',
    };
  }

  // Create new access token with same payload
  const { type, ...payload } = verification.payload;
  const accessToken = createAccessToken(payload);

  return {
    success: true,
    accessToken,
    accessTokenExpiresAt: new Date(Date.now() + 15 * 60 * 1000),
  };
}

/**
 * Revoke token (add to revocation list)
 */
export function revokeToken(token: string): void {
  revokedTokens.add(token);

  // Auto-cleanup after expiration
  // In production, use Redis with TTL instead
  setTimeout(() => {
    revokedTokens.delete(token);
  }, 7 * 24 * 60 * 60 * 1000).unref?.(); // 7 days; cleanup must not hold the process open
}

/**
 * Register a token with its session for bulk revocation support
 */
export function registerTokenWithSession(token: string, sessionId: string): void {
  const decoded = decodeToken(token);
  if (!decoded || !decoded.jti) return;

  // Track token JTI -> full token
  tokenJtiMap.set(decoded.jti, token);

  // Track session -> token JTIs
  if (!sessionTokens.has(sessionId)) {
    sessionTokens.set(sessionId, new Set());
  }
  sessionTokens.get(sessionId)!.add(decoded.jti);

  // Auto-cleanup after token expiration
  const expiresIn = decoded.exp ? (decoded.exp * 1000 - Date.now()) : 7 * 24 * 60 * 60 * 1000;
  setTimeout(() => {
    tokenJtiMap.delete(decoded.jti!);
    sessionTokens.get(sessionId)?.delete(decoded.jti!);
    if (sessionTokens.get(sessionId)?.size === 0) {
      sessionTokens.delete(sessionId);
    }
  }, Math.max(expiresIn, 0)).unref?.();
}

/**
 * Revoke all tokens for a user (by session ID)
 */
export function revokeUserSession(sessionId: string): { revokedCount: number } {
  const tokenJtis = sessionTokens.get(sessionId);

  if (!tokenJtis || tokenJtis.size === 0) {
    return { revokedCount: 0 };
  }

  let revokedCount = 0;
  for (const jti of tokenJtis) {
    const token = tokenJtiMap.get(jti);
    if (token) {
      revokedTokens.add(token);
      tokenJtiMap.delete(jti);
      revokedCount++;
    }
  }

  sessionTokens.delete(sessionId);
  return { revokedCount };
}

/**
 * Get active session count
 */
export function getActiveSessionCount(): number {
  return sessionTokens.size;
}

/**
 * Get tokens count for a session
 */
export function getSessionTokenCount(sessionId: string): number {
  return sessionTokens.get(sessionId)?.size ?? 0;
}

/**
 * Decode token without verification (use cautiously)
 */
export function decodeToken(token: string): jwt.JwtPayload | null {
  try {
    return jwt.decode(token) as jwt.JwtPayload;
  } catch (error) {
    return null;
  }
}

/**
 * Get token expiration time
 */
export function getTokenExpiration(token: string): Date | null {
  const decoded = decodeToken(token);

  if (!decoded || !decoded.exp) {
    return null;
  }

  return new Date(decoded.exp * 1000);
}

/**
 * Check if token is expired
 */
export function isTokenExpired(token: string): boolean {
  const expiration = getTokenExpiration(token);

  if (!expiration) {
    return true;
  }

  return expiration.getTime() < Date.now();
}

/**
 * Get time until token expires (in seconds)
 */
export function getTokenTimeRemaining(token: string): number {
  const expiration = getTokenExpiration(token);

  if (!expiration) {
    return 0;
  }

  const remaining = Math.floor((expiration.getTime() - Date.now()) / 1000);
  return Math.max(0, remaining);
}

/**
 * Rotate token pair (create new tokens and revoke old refresh token)
 */
export function rotateTokenPair(
  refreshToken: string
): {
  success: boolean;
  tokens?: TokenPair;
  error?: string;
} {
  const verification = verifyRefreshToken(refreshToken);

  if (!verification.valid || !verification.payload) {
    return {
      success: false,
      error: verification.error || 'Invalid refresh token',
    };
  }

  // Revoke old refresh token
  revokeToken(refreshToken);

  // Create new token pair
  const { type, ...payload } = verification.payload;
  const tokens = createTokenPair(payload);

  return {
    success: true,
    tokens,
  };
}

/**
 * Extract token from Authorization header
 */
export function extractTokenFromHeader(authHeader: string | undefined): string | null {
  if (!authHeader) {
    return null;
  }

  const parts = authHeader.split(' ');

  if (parts.length !== 2 || parts[0] !== 'Bearer') {
    return null;
  }

  return parts[1];
}

/**
 * Create token for service account (long-lived, no expiration)
 */
export function createServiceAccountToken(payload: Omit<TokenPayload, 'type'>): string {
  const secret = getJWTSecret();

  const tokenPayload: TokenPayload = {
    ...payload,
    type: 'access',
  };

  const token = jwt.sign(tokenPayload, secret, {
    algorithm: TOKEN_CONFIG.ALGORITHM,
    issuer: TOKEN_CONFIG.ISSUER,
    audience: TOKEN_CONFIG.AUDIENCE,
    jwtid: generateSecureRandomBase64(16),
    // No expiration for service accounts
  });

  return token;
}
