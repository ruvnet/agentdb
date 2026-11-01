/**
 * The guard's decision, pure: no `$`. It judges only calls to an MCP server named `agentdb` (this plugin's,
 * `mcp__plugin_agentdb_agentdb__*`, or one registered by hand, `mcp__agentdb__*`). ruflo's server is left to
 * ruflo-agentdb, which guards its own tool names.
 */
import type { GuardMode } from './options'
import { scanInput } from './secrets'

/** `mcp__<server>__<tool>` split in two; tool names never hold a double underscore. */
export function splitTool(name: string): { readonly server: string; readonly tool: string } | undefined {
  if (!name.startsWith('mcp__')) return undefined
  const at = name.lastIndexOf('__')
  return at > 5 ? { server: name.slice(5, at), tool: name.slice(at + 2) } : undefined
}

const AGENTDB_SERVER = /^(?:plugin_[A-Za-z0-9-]+_)?agentdb$/

/** The agentdb tool a call names, or undefined when it goes to another server or is not an MCP tool. */
export function agentdbTool(name: string): string | undefined {
  const parts = splitTool(name)
  return parts && AGENTDB_SERVER.test(parts.server) ? parts.tool : undefined
}

/** The agentdb MCP tools that write text into memory (src/security/mcp-policy.ts, learning profile, agentdb 3.1.0). */
export const WRITERS: ReadonlySet<string> = new Set([
  'agentdb_insert',
  'agentdb_insert_batch',
  'agentdb_pattern_store',
  'agentdb_pattern_store_batch',
  'reflexion_store',
  'reflexion_store_batch',
  'skill_create',
  'skill_create_batch',
  'causal_add_edge',
  'experience_record',
  'learning_feedback',
  // agentdb 3.1.0: outcome feedback for the reward-aware rerank; it can carry a free-text note.
  'agentdb_pattern_feedback',
  'agentdb_skill_feedback',
  // agentdb 3.3.0 (ADR-017): in-place updates write new text and metadata, so they are screened like inserts.
  'agentdb_update',
  'agentdb_pattern_update',
  'skill_update',
])

/** The admin-only tools that delete or reset data (ADMIN_ONLY_TOOLS); agentdb_init only with `reset: true`. */
export const DESTRUCTIVE: ReadonlySet<string> = new Set([
  'agentdb_delete',
  'agentdb_delete_batch',
  'agentdb_clear_cache',
  'agentdb_init',
  // agentdb 3.3.0 (ADR-017)
  'agentdb_pattern_delete',
  'skill_delete',
])

/** Whether the guard looks at a call at all. */
export const isGuarded = (name: string): boolean => {
  const tool = agentdbTool(name)
  return tool !== undefined && (WRITERS.has(tool) || DESTRUCTIVE.has(tool))
}

export type Refusal = { readonly cls: 'secret' | 'oversize' | 'destructive'; readonly reason: string }
export type Policy = { readonly secretGuard: GuardMode; readonly allowDestructive: boolean }

export const SECRET_REFUSAL =
  'agentdb mod: this memory write holds what looks like a secret (a key, token or password). Store a reference to where it lives, not the value.'
export const OVERSIZE_REFUSAL = 'agentdb mod: this memory write is too large to screen for secrets in full. Store it in smaller pieces.'
export const GUARD_FAILED = 'agentdb mod: the guard could not check this call, so it was refused.'
export const destructiveRefusal = (tool: string): string =>
  `agentdb mod: ${tool} deletes or resets AgentDB data and is refused. To allow it, turn on the plugin option allow_destructive (and use the admin MCP profile).`

const resets = (input: unknown): boolean => {
  const reset = (input as { reset?: unknown } | null | undefined)?.reset
  return reset === true || reset === 'true'
}

/** Why a call is refused, or undefined when it may go. `secretGuard` gates only the secret check. Never echoes a matched value. */
export function judge(name: string, input: unknown, policy: Policy): Refusal | undefined {
  const tool = agentdbTool(name)
  if (tool === undefined) return undefined
  if (DESTRUCTIVE.has(tool) && !policy.allowDestructive && (tool !== 'agentdb_init' || resets(input))) {
    return { cls: 'destructive', reason: destructiveRefusal(tool) }
  }
  if (!WRITERS.has(tool) || policy.secretGuard === 'off') return undefined
  const { secrets, oversize } = scanInput(input)
  if (oversize) return { cls: 'oversize', reason: OVERSIZE_REFUSAL }
  return secrets.length > 0 ? { cls: 'secret', reason: SECRET_REFUSAL } : undefined
}

export type FailureKind = 'throw' | 'timeout' | 're-entry'

/**
 * The `.catch` handler's verdict when the guard itself failed. A call whose `next` already ran is left alone.
 * A re-entry (a call raised beneath the guard's own `$` call) is judged once; a throw or overrun refuses
 * any guarded call, so the guard fails closed. A destructive call is refused in every mode; a write only in enforce.
 */
export function fallbackVerdict(policy: Policy, name: string, input: unknown, called: boolean, kind: FailureKind): { readonly deny: string } | undefined {
  if (called || !isGuarded(name)) return undefined
  const tool = agentdbTool(name) ?? ''
  const enforced = DESTRUCTIVE.has(tool) ? !policy.allowDestructive : policy.secretGuard === 'enforce'
  if (!enforced) return undefined
  if (kind !== 're-entry') return { deny: GUARD_FAILED }
  try {
    const refusal = judge(name, input, policy)
    return refusal === undefined ? undefined : { deny: refusal.reason }
  } catch {
    return { deny: GUARD_FAILED }
  }
}

/** The call's arguments without the keys the engine reserves (`tool`, `tool_use_id`, `consent`, `agentId`). */
export function argsOf(e: Readonly<Record<string, unknown>>): Record<string, unknown> {
  const { tool: _tool, tool_use_id: _id, consent: _consent, agentId: _agent, ...rest } = e
  return rest
}
