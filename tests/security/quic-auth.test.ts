/**
 * QUICServer accepted every request when no auth token was configured,
 * compared tokens with ===, bound 0.0.0.0 by default, and returned the
 * token from getStatus().
 */
import { describe, it, expect } from 'vitest';
import Database from 'better-sqlite3';
import { QUICServer } from '../../src/controllers/QUICServer.js';

describe('QUICServer authentication', () => {
  const db = new Database(':memory:');

  it('rejects requests when no token is configured', () => {
    const server = new QUICServer(db as any, {});
    expect((server as any).authenticate('c1', '')).toBe(false);
    expect((server as any).authenticate('c1', 'anything')).toBe(false);
  });

  it('accepts only the exact configured token', () => {
    const server = new QUICServer(db as any, { authToken: 'secret-token' });
    expect((server as any).authenticate('c1', 'secret-token')).toBe(true);
    expect((server as any).authenticate('c1', 'secret-tokeN')).toBe(false);
    expect((server as any).authenticate('c1', 'secret')).toBe(false);
  });

  it('binds to loopback by default and redacts the token in getStatus()', () => {
    const server = new QUICServer(db as any, { authToken: 'secret-token' });
    const status = server.getStatus();
    expect(status.config.host).toBe('127.0.0.1');
    expect(status.config.authToken).not.toBe('secret-token');
    expect(JSON.stringify(status)).not.toContain('secret-token');
  });
});

describe('agentdb sync start-server token generation', () => {
  it('generates 256-bit tokens from the CSPRNG', async () => {
    const { AgentDBCLI } = await import('../../src/cli/agentdb-cli.js');
    const cli = new AgentDBCLI() as any;
    const a: string = cli.generateAuthToken();
    const b: string = cli.generateAuthToken();
    expect(a).toMatch(/^[0-9a-f]{64}$/);
    expect(a).not.toBe(b);
  });
});
