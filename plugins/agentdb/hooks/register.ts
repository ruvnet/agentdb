import type { Register } from 'claude-code'

import { answer, COMMAND, newStats } from './command'
import { agentdbTool, argsOf, fallbackVerdict, isGuarded, judge } from './guard'
import { readOptions } from './options'

/**
 * agentdb as a mod: a tighten-only guard on the agentdb MCP server's tools (no secrets into memory, no
 * delete/reset unless allow_destructive is on) and `/agentdb-guard`. Every `$` call is in this file; the
 * helpers it imports are pure. No network, no process, no file writes.
 */
export const register: Register = (on, options) => {
  const opts = readOptions(options)
  const policy = { secretGuard: opts.secretGuard, allowDestructive: opts.allowDestructive }
  const stats = newStats()

  on('session.start', async ($, e, next) => {
    const result = await next(e)
    try {
      await $.command.register({ name: COMMAND, description: 'AgentDB mod: status, scan <text>' })
    } catch {
      /* a name taken by another plugin must not stop the guard */
    }
    return result
  })

  // Tighten-only: a deny or the event unchanged; never a rewrite, never an allow over another hook.
  on('tool.call', async ($, e, next) => {
    if (!isGuarded(e.tool)) return next(e)
    stats.calls++
    const refusal = judge(e.tool, argsOf(e as Record<string, unknown>), policy)
    if (refusal === undefined) return next(e)
    stats.lastRefused = `${agentdbTool(e.tool)} (${refusal.cls})`
    if (refusal.cls === 'destructive' || opts.secretGuard === 'enforce') {
      stats.blocked++
      return { deny: refusal.reason }
    }
    stats.flagged++
    try {
      $.ui.toast('agentdb mod (notify): a memory write that looks like it holds a secret went through')
    } catch {
      /* a refused toast changes nothing */
    }
    return next(e)
  }).catch(($, e, next) => {
    // Fail closed: a guard that threw or overran refuses the call; a re-entry is judged once.
    const verdict = fallbackVerdict(policy, e.tool, argsOf(e as Record<string, unknown>), next.called, next.error.kind)
    return verdict ?? next(e)
  })

  on('command.run', { command: COMMAND }, async ($, e) => {
    const args = typeof e.args === 'string' ? e.args : ''
    const text = await answer(args, {
      opts,
      stats,
      tools: async () => {
        try {
          return (await $.tool.list()).map(t => t.name).filter(n => agentdbTool(n) !== undefined)
        } catch {
          return undefined
        }
      },
    })
    return { text }
  })
}
