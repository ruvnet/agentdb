export type AgentDbMcpProfile = 'readonly' | 'retrieval' | 'learning' | 'admin';

const READONLY_TOOLS = [
  'agentdb_search',
  'agentdb_pattern_search',
  'agentdb_pattern_stats',
  'agentdb_stats',
  'db_stats',
  'causal_query',
  'causal_traverse',
  'learning_explain',
  'learning_metrics',
  'learning_predict',
  'recall_with_certificate',
  'reflexion_retrieve',
  'skill_search',
] as const;

const LEARNING_TOOLS = [
  ...READONLY_TOOLS,
  'experience_record',
  'learning_end_session',
  'learning_feedback',
  'learning_start_session',
  'learning_train',
  'reward_signal',
] as const;

const PROFILE_TOOLS: Record<Exclude<AgentDbMcpProfile, 'admin'>, readonly string[]> = {
  readonly: READONLY_TOOLS,
  retrieval: READONLY_TOOLS,
  learning: LEARNING_TOOLS,
};

export interface McpToolPolicy {
  profile: AgentDbMcpProfile | null;
  allow: Set<string> | null;
  deny: Set<string>;
  configured: boolean;
}

function parseList(value?: string): string[] {
  return (value ?? '')
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function buildMcpToolPolicy(
  env: Record<string, string | undefined> = process.env,
): McpToolPolicy {
  const rawProfile = env.AGENTDB_MCP_PROFILE?.trim();
  const profile = rawProfile as AgentDbMcpProfile | undefined;
  if (rawProfile && !['readonly', 'retrieval', 'learning', 'admin'].includes(rawProfile)) {
    throw new Error(`Unknown AGENTDB_MCP_PROFILE "${rawProfile}"`);
  }

  const explicitAllow = parseList(env.AGENTDB_MCP_ALLOW);
  const deny = new Set(parseList(env.AGENTDB_MCP_DENY));
  let allow: Set<string> | null = null;
  if (profile && profile !== 'admin') allow = new Set(PROFILE_TOOLS[profile]);
  if (explicitAllow.length > 0) allow = new Set([...(allow ?? []), ...explicitAllow]);

  return {
    profile: profile ?? null,
    allow,
    deny,
    configured: Boolean(profile || explicitAllow.length || deny.size),
  };
}

export function isMcpToolAllowed(name: string, policy: McpToolPolicy): boolean {
  if (policy.deny.has(name)) return false;
  if (policy.allow) return policy.allow.has(name);
  return true;
}

export function filterMcpTools<T extends { name: string }>(
  tools: T[],
  policy: McpToolPolicy,
): T[] {
  return tools.filter((tool) => isMcpToolAllowed(tool.name, policy));
}

