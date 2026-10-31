export type AgentDbMcpProfile = 'readonly' | 'retrieval' | 'learning' | 'admin';

const READONLY_TOOLS = [
  'agentdb_capabilities',
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
  'learning_transfer',
  'learner_discover',
  'reward_signal',
  // Outcome feedback that drives reward-aware ranking (ADR-015)
  'agentdb_pattern_feedback',
  'agentdb_skill_feedback',
  // Memory writes. Destructive tools are in ADMIN_ONLY_TOOLS.
  'agentdb_insert',
  'agentdb_insert_batch',
  'agentdb_pattern_store',
  'agentdb_pattern_store_batch',
  'causal_add_edge',
  'consolidate_now',
  'reflexion_store',
  'reflexion_store_batch',
  'skill_create',
  'skill_create_batch',
] as const;

/** Tools that reset or delete data: only the admin profile (or an explicit allow) grants them. */
export const ADMIN_ONLY_TOOLS: readonly string[] = [
  'agentdb_init',
  'agentdb_delete',
  'agentdb_delete_batch',
  'agentdb_clear_cache',
];

/** Profile used when AGENTDB_MCP_PROFILE is unset */
export const DEFAULT_MCP_PROFILE: AgentDbMcpProfile = 'learning';

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
  // Unset means the learning profile, not allow-all: destructive tools such
  // as agentdb_init (reset) must be opted into with AGENTDB_MCP_PROFILE=admin.
  const profile = (rawProfile || DEFAULT_MCP_PROFILE) as AgentDbMcpProfile;
  if (rawProfile && !['readonly', 'retrieval', 'learning', 'admin'].includes(rawProfile)) {
    throw new Error(`Unknown AGENTDB_MCP_PROFILE "${rawProfile}"`);
  }

  const explicitAllow = parseList(env.AGENTDB_MCP_ALLOW);
  const deny = new Set(parseList(env.AGENTDB_MCP_DENY));
  let allow: Set<string> | null = null;
  if (profile !== 'admin') allow = new Set(PROFILE_TOOLS[profile]);
  if (explicitAllow.length > 0) allow = new Set([...(allow ?? []), ...explicitAllow]);

  return {
    profile,
    allow,
    deny,
    configured: Boolean(rawProfile || explicitAllow.length || deny.size),
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

