import { describe, expect, it } from 'vitest';
import {
  buildMcpToolPolicy,
  filterMcpTools,
  isMcpToolAllowed,
} from '../src/security/mcp-policy.js';

describe('AgentDB MCP capability policy', () => {
  it('defaults to the learning profile when unconfigured', () => {
    const policy = buildMcpToolPolicy({});
    expect(policy.configured).toBe(false);
    expect(policy.profile).toBe('learning');
    expect(isMcpToolAllowed('reflexion_store', policy)).toBe(true);
    expect(isMcpToolAllowed('agentdb_search', policy)).toBe(true);
    for (const tool of ['agentdb_init', 'agentdb_delete', 'agentdb_delete_batch', 'agentdb_clear_cache']) {
      expect(isMcpToolAllowed(tool, policy)).toBe(false);
    }
  });

  it('grants destructive tools to the admin profile', () => {
    const policy = buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: 'admin' });
    expect(isMcpToolAllowed('agentdb_init', policy)).toBe(true);
    expect(isMcpToolAllowed('agentdb_delete', policy)).toBe(true);
  });

  it('limits readonly clients to inspection and retrieval', () => {
    const policy = buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: 'readonly' });
    expect(isMcpToolAllowed('agentdb_search', policy)).toBe(true);
    expect(isMcpToolAllowed('agentdb_insert', policy)).toBe(false);
    expect(filterMcpTools(
      [{ name: 'agentdb_search' }, { name: 'agentdb_delete' }],
      policy,
    )).toEqual([{ name: 'agentdb_search' }]);
  });

  it('gives deny precedence over profile and explicit allow', () => {
    const policy = buildMcpToolPolicy({
      AGENTDB_MCP_PROFILE: 'learning',
      AGENTDB_MCP_ALLOW: 'agentdb_insert',
      AGENTDB_MCP_DENY: 'learning_train,agentdb_insert',
    });
    expect(isMcpToolAllowed('learning_train', policy)).toBe(false);
    expect(isMcpToolAllowed('agentdb_insert', policy)).toBe(false);
    expect(isMcpToolAllowed('agentdb_search', policy)).toBe(true);
  });

  it('rejects unknown profiles', () => {
    expect(() => buildMcpToolPolicy({ AGENTDB_MCP_PROFILE: 'root' })).toThrow(
      /Unknown AGENTDB_MCP_PROFILE/,
    );
  });
});

