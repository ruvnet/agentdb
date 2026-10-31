/** `/agentdb-guard`, answered locally with no model turn. Pure: register.ts passes in what it read through `$`. */
import type { ModOptions } from './options'
import { scanText } from './secrets'

export const COMMAND = 'agentdb-guard'

/** Session counters. Refusals keep the tool and class only, never the input. */
export type Stats = { calls: number; blocked: number; flagged: number; lastRefused?: string }
export const newStats = (): Stats => ({ calls: 0, blocked: 0, flagged: 0 })

export type CommandDeps = {
  readonly opts: ModOptions
  readonly stats: Stats
  /** The connected agentdb tool names, or undefined when the list could not be read. */
  readonly tools: () => Promise<readonly string[] | undefined>
}

const HELP = ['/agentdb-guard status', '/agentdb-guard scan <text>'].join('\n')

export async function answer(args: string, deps: CommandDeps): Promise<string> {
  const [verb = '', ...rest] = args.trim().split(/\s+/)
  const arg = rest.join(' ')
  const { opts, stats } = deps

  if (verb === '' || verb === 'help') return HELP

  if (verb === 'status') {
    const tools = await deps.tools()
    const server =
      tools === undefined ? 'agentdb MCP server: tool list unavailable' : tools.length === 0 ? 'agentdb MCP server: not connected (no agentdb tools)' : `agentdb MCP server: ${tools.length} tools`
    return [
      `secret guard ${opts.secretGuard} · destructive tools ${opts.allowDestructive ? 'allowed' : 'refused'}`,
      server,
      `guarded calls ${stats.calls} · blocked ${stats.blocked} · flagged ${stats.flagged}${stats.lastRefused ? ` · last refused ${stats.lastRefused}` : ''}`,
    ].join('\n')
  }

  if (verb === 'scan') {
    if (arg === '') return 'usage: /agentdb-guard scan <text>'
    const found = scanText(arg)
    return found.length > 0 ? `The guard would refuse a memory write of that (${found.join(', ')}).` : 'Nothing found: the guard would let that be stored.'
  }

  return `Unknown: ${verb}\n${HELP}`
}
