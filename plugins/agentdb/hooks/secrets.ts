/**
 * Pure secret screening for AgentDB memory writes. Findings are rule NAMES only: the matched text is never
 * returned, logged or counted by value.
 *
 * The token shapes in TOKEN_RULES are a snapshot of COMMON_SECRETS in ruflo's plugins/ruflo-agentdb/hooks/screen.ts
 * (ruflo's sync-mod-screen.mjs does not reach this repository; refresh by hand when that list grows).
 */

export type Rules = readonly (readonly [string, RegExp])[]

/** Token shapes that are secrets wherever they appear. */
export const TOKEN_RULES: Rules = [
  ['private key', /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ['aws access key', /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/],
  ['github token', /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b/],
  ['slack token', /\bxox[abprs]-[A-Za-z0-9-]{10,}/],
  ['slack webhook', /\bhooks\.slack\.com\/services\/T[A-Z0-9]{6,}\/B[A-Z0-9]{6,}\/[A-Za-z0-9]{16,}/],
  ['google api key', /\bAIza[0-9A-Za-z_-]{35}\b/],
  ['anthropic or openai key', /\bsk-(?:(?:ant|proj|svcacct|admin)-[A-Za-z0-9_-]{20,}|(?=[A-Za-z]{0,40}\d)[A-Za-z0-9]{32,})/],
  ['stripe key', /\b[rs]k_live_[A-Za-z0-9]{16,}/],
  ['npm token', /\bnpm_[A-Za-z0-9]{36}\b/],
  ['huggingface token', /\bhf_[A-Za-z0-9]{30,}\b/],
  ['sendgrid key', /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}/],
  ['jwt', /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/],
  ['bearer token', /\bBearer\s+[A-Za-z0-9._~+/=-]{24,}/],
  ['url with credentials', /\b[a-z][a-z0-9+.-]{1,20}:\/\/[^\s:@/]+:[^\s@/]{3,}@[^\s/]+/i],
]

/** A name that says its value is a credential: api_key, client-secret, DB_PASSWORD, accessToken. */
const SECRET_NAME = /(?:api[_-]?key|apikey|secret|passw(?:or)?d|\bpwd\b|token|private[_-]?key|access[_-]?key|credential)/i
/** `NAME = value` / `NAME: value` / `"NAME": "value"` inside free text. */
const ASSIGNMENT = /([A-Za-z0-9_.-]{2,40})["']?\s*[:=]\s*["']?([^\s"',;]{8,})/g
/** Values that name where a secret lives, or stand in for one, rather than being one. */
const NOT_A_VALUE = /^(?:\$\{?|%|<|\{\{|process\.env|env\.|vault:|op:\/\/|arn:|projects\/[^/]+\/secrets\/)|placeholder|your[-_]|example|changeme|redacted|dummy|x{5,}|\*{3,}|\.{3}/i

function entropy(v: string): number {
  const counts = new Map<string, number>()
  for (const ch of v) counts.set(ch, (counts.get(ch) ?? 0) + 1)
  let h = 0
  for (const n of counts.values()) h -= (n / v.length) * Math.log2(n / v.length)
  return h
}

/** Whether a value looks random enough to be a credential: 8+ characters, letters plus a digit or symbol, 2.5+ bits per character. */
export function looksRandom(value: string): boolean {
  const v = value.trim()
  if (v.length < 8 || v.length > 512 || NOT_A_VALUE.test(v)) return false
  if (!/[A-Za-z]/.test(v) || !/[0-9_\-+/=!@#%^&*]/.test(v)) return false
  return entropy(v) >= 2.5
}

/** A credential-named field with a credential-shaped value. */
export const isSecretPair = (name: string, value: string): boolean => SECRET_NAME.test(name) && looksRandom(value)

/** The rule names a text trips: a token shape, or `name=value` with a credential name and a random value. */
export function scanText(text: string): string[] {
  const found = new Set<string>()
  for (const [name, re] of TOKEN_RULES) if (re.test(text)) found.add(name)
  for (const m of text.matchAll(ASSIGNMENT)) if (isSecretPair(m[1] ?? '', m[2] ?? '')) found.add('credential assignment')
  return [...found]
}

/** Walker budgets: past either the input is not fully read, and the guard refuses rather than pass what it did not see. */
export const NODE_CAP = 20_000
export const CHAR_CAP = 2_000_000

const NAME_FIELD = /^(?:key|name|field|label|variable|env|header|param|property)$/i
const VALUE_FIELD = /^(?:value|val|content|secret|data|text|string)$/i

export type Scan = { readonly secrets: readonly string[]; readonly oversize: boolean }

/**
 * Every secret rule `input` trips, reading every string in it (object keys too), credential-named properties
 * (`{ api_key: '...' }`), name/value field pairs (`{ key: 'api_key', value: '...' }`) and `[name, value]` pairs.
 */
export function scanInput(input: unknown): Scan {
  const found = new Set<string>()
  const queue: unknown[] = [input]
  let chars = 0
  for (let head = 0; head < queue.length; head++) {
    if (queue.length > NODE_CAP) return { secrets: [...found], oversize: true }
    const node = queue[head]
    if (typeof node === 'string') {
      chars += node.length
      if (chars > CHAR_CAP) return { secrets: [...found], oversize: true }
      for (const name of scanText(node)) found.add(name)
    } else if (Array.isArray(node)) {
      if (node.length === 2 && typeof node[0] === 'string' && typeof node[1] === 'string' && isSecretPair(node[0], node[1])) found.add('credential field')
      for (const v of node) queue.push(v)
    } else if (typeof node === 'object' && node !== null) {
      const o = node as Record<string, unknown>
      const keys = Object.keys(o)
      const name = keys.find(k => NAME_FIELD.test(k) && typeof o[k] === 'string')
      const value = keys.find(k => VALUE_FIELD.test(k) && typeof o[k] === 'string')
      if (name && value && isSecretPair(o[name] as string, o[value] as string)) found.add('credential field')
      for (const k of keys) {
        const v = o[k]
        if (typeof v === 'string' && isSecretPair(k, v)) found.add('credential field')
        chars += k.length
        queue.push(k, v)
      }
    }
  }
  return { secrets: [...found], oversize: false }
}
