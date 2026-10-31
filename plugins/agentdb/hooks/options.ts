import type { PluginOptions } from 'claude-code'

/** The secret guard's mode. `enforce` (default): refuse. `notify`: let the write through, toast and count it. `off`: no check. */
export type GuardMode = 'off' | 'notify' | 'enforce'
export const GUARD_MODES: readonly GuardMode[] = ['off', 'notify', 'enforce']

/** The plugin's `userConfig` as the mod reads it; the four MCP options go to the server's env through .mcp.json, not here. */
export type ModOptions = {
  readonly secretGuard: GuardMode
  readonly allowDestructive: boolean
}

/** `on`/`true` read as enforce and `false` as off, so a boolean-style value still means what it says. */
export function asMode(value: unknown): GuardMode | undefined {
  if (value === true || value === 'on' || value === 'true') return 'enforce'
  if (value === false || value === 'false') return 'off'
  return GUARD_MODES.find(m => m === value)
}

/** Only an explicit yes enables the destructive tools; anything else, missing included, keeps them refused. */
const yes = (value: unknown): boolean => value === true || value === 'true' || value === 'on'

/** Validated options: a missing or unknown value is the safe default (guard enforced, destructive tools refused). */
export function readOptions(options: PluginOptions | Readonly<Record<string, unknown>> | undefined): ModOptions {
  const o = (options ?? {}) as Readonly<Record<string, unknown>>
  return { secretGuard: asMode(o.secret_guard) ?? 'enforce', allowDestructive: yes(o.allow_destructive) }
}
