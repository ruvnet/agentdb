import type { On } from 'claude-code'
import { describe, expect, test, tier } from 'claude-code/testing'

import { fallbackVerdict, GUARD_FAILED } from '../hooks/guard'

tier('user')

const START = { surface: 'terminal', isInteractive: true, cwd: '/work' } as const
const slash = (args: string) => ({ command: 'agentdb-guard', args, origin: { kind: 'composer' }, presentation: { isFullscreen: false, columns: 100 } }) as const
const TOKEN = 'ghp_' + 'Zq8vT3mK9wX2LpR7sNd4Yb6Hc1Jf5Ge0Qa3W'
const PLUGIN = 'mcp__plugin_agentdb_agentdb__'
const MANUAL = 'mcp__agentdb__'

/** The engine beneath the mod: every tool call that reaches core answers `ran`. */
function world(on: On) {
  const ran: string[] = []
  const toasts: string[] = []
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('tool.list', () => ({ value: [{ name: `${PLUGIN}agentdb_stats`, description: '', mcp: true }, { name: 'Bash', description: '', mcp: false }] }))
  on('ui.toast', ($, e) => (toasts.push(JSON.stringify(e)), { value: undefined }))
  on('tool.call', ($, e) => (ran.push(e.tool), { result: 'ran' }))
  return { ran, toasts }
}

const call = async ($: { tool: { call: (i: never) => Promise<unknown> } }, tool: string, args: Record<string, unknown>) =>
  JSON.stringify(await $.tool.call({ tool, tool_use_id: 't1', ...args } as never))

describe('secret write guard (enforce, the default)', () => {
  test('a pattern store holding a token is refused, and the reason never echoes it', async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    const out = await call($, `${PLUGIN}agentdb_pattern_store`, { taskType: 'deploy', approach: `use ${TOKEN} for the registry` })
    expect(out).toContain('looks like a secret')
    expect(out).not.toContain(TOKEN)
    expect(w.ran).toEqual([])
  })

  test('both server spellings are guarded, and every writer', async ($, on) => {
    world(on)
    await $.session.start(START)
    expect(await call($, `${MANUAL}reflexion_store`, { session_id: 's', task: 't', critique: 'password=Zq8vT3mK9wX2' })).toContain('looks like a secret')
    expect(await call($, `${PLUGIN}agentdb_insert`, { text: 'x', metadata: { api_key: 'Zq8vT3mK9wX2LpR7' } })).toContain('looks like a secret')
    expect(await call($, `${PLUGIN}agentdb_skill_feedback`, { success: true, reward: 1, note: `worked with ${TOKEN}` })).toContain('looks like a secret')
    expect(await call($, `${PLUGIN}agentdb_pattern_feedback`, { success: true, reward: 0.8 })).toContain('ran')
    expect(await call($, `${PLUGIN}skill_create_batch`, { skills: [{ name: 'deploy', code: '-----BEGIN RSA PRIVATE KEY-----' }] })).toContain('looks like a secret')
  })

  test('ordinary memory goes through', async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_pattern_store`, { taskType: 'auth', approach: 'rotate the refresh token on every use; keep the key in the vault' })).toContain('ran')
    expect(await call($, `${PLUGIN}agentdb_insert`, { text: 'api_key lives in GCP secret NPM_TOKEN:latest', metadata: { api_key: '${API_KEY}' } })).toContain('ran')
    expect(w.ran).toHaveLength(2)
  })

  test("reads and other servers' tools are not judged", async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_search`, { query: TOKEN })).toContain('ran')
    expect(await call($, 'mcp__plugin_ruflo-core_ruflo__agentdb_pattern-store', { pattern: TOKEN })).toContain('ran')
    expect(w.ran).toHaveLength(2)
  })
})

describe('secret write guard: notify and off', () => {
  test('notify lets the write through and toasts', { options: { secret_guard: 'notify' } }, async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_pattern_store`, { approach: TOKEN })).toContain('ran')
    expect(w.toasts.join(' ')).toContain('agentdb mod (notify)')
  })

  test('off does not screen writes, but still refuses destructive tools', { options: { secret_guard: 'off' } }, async ($, on) => {
    world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_pattern_store`, { approach: TOKEN })).toContain('ran')
    expect(await call($, `${PLUGIN}agentdb_delete`, { ids: [1] })).toContain('allow_destructive')
  })
})

describe('destructive tool guard', () => {
  test('delete, delete_batch, clear_cache and init with reset are refused by default', async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    for (const tool of ['agentdb_delete', 'agentdb_delete_batch', 'agentdb_clear_cache']) {
      expect(await call($, `${PLUGIN}${tool}`, {})).toContain(`${tool} deletes or resets`)
    }
    expect(await call($, `${MANUAL}agentdb_init`, { reset: true })).toContain('allow_destructive')
    expect(w.ran).toEqual([])
  })

  test('init without reset goes through', async ($, on) => {
    world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_init`, {})).toContain('ran')
    expect(await call($, `${PLUGIN}agentdb_init`, { reset: false })).toContain('ran')
  })

  test('allow_destructive lets them through', { options: { allow_destructive: true } }, async ($, on) => {
    const w = world(on)
    await $.session.start(START)
    expect(await call($, `${PLUGIN}agentdb_delete`, { ids: [1] })).toContain('ran')
    expect(await call($, `${PLUGIN}agentdb_init`, { reset: true })).toContain('ran')
    expect(w.ran).toHaveLength(2)
  })
})

describe('the .catch handler (fail closed)', () => {
  const policy = { secretGuard: 'enforce', allowDestructive: false } as const
  const throwing = {
    get approach(): string {
      throw new Error('unreadable input')
    },
  }

  test('a throw or overrun refuses a guarded call outright', () => {
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_pattern_store`, { approach: 'fine' }, false, 'throw')).toEqual({ deny: GUARD_FAILED })
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_delete`, {}, false, 'timeout')).toEqual({ deny: GUARD_FAILED })
  })

  test('a re-entry is judged once; a check that throws refuses', () => {
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_pattern_store`, { approach: TOKEN }, false, 're-entry')?.deny).toContain('looks like a secret')
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_pattern_store`, { approach: 'fine' }, false, 're-entry')).toBeUndefined()
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_pattern_store`, throwing, false, 're-entry')).toEqual({ deny: GUARD_FAILED })
  })

  test('unguarded tools, calls whose next already ran, and notify writes are left alone', () => {
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_search`, {}, false, 'throw')).toBeUndefined()
    expect(fallbackVerdict(policy, `${PLUGIN}agentdb_delete`, {}, true, 'throw')).toBeUndefined()
    expect(fallbackVerdict({ secretGuard: 'notify', allowDestructive: false }, `${PLUGIN}agentdb_pattern_store`, {}, false, 'throw')).toBeUndefined()
    expect(fallbackVerdict({ secretGuard: 'off', allowDestructive: false }, `${PLUGIN}agentdb_delete`, {}, false, 'throw')).toEqual({ deny: GUARD_FAILED })
  })
})

describe('/agentdb-guard', () => {
  test('status reports the modes, the server and the counters', async ($, on) => {
    world(on)
    await $.session.start(START)
    await call($, `${PLUGIN}agentdb_delete`, {})
    const text = (await $.command.run(slash('status'))).text
    expect(text).toContain('secret guard enforce')
    expect(text).toContain('destructive tools refused')
    expect(text).toContain('agentdb MCP server: 1 tools')
    expect(text).toContain('blocked 1')
  })

  test('scan names the rule, never the value; unknown verbs get the help', async ($, on) => {
    world(on)
    await $.session.start(START)
    const scanned = (await $.command.run(slash(`scan deploy with ${TOKEN}`))).text
    expect(scanned).toContain('github token')
    expect(scanned).not.toContain(TOKEN)
    expect((await $.command.run(slash('scan the weather is nice'))).text).toContain('Nothing found')
    expect((await $.command.run(slash('bogus'))).text).toContain('/agentdb-guard status')
  })
})
