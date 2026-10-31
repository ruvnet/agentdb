#!/usr/bin/env bash
# Static contract for the agentdb mod (plugins/agentdb/hooks), checked without
# running Claude Code. The loader refuses a whole module on the first broken rule,
# so these mirror its rules:
#   - hooks.json names ./register.ts as its one module
#   - each event has at most one bare on(event, hook); the rest carry a matcher
#   - `$` appears only in register.ts (passing `$` to an imported helper is refused)
#   - no computed $.env.get/set names, no Node/process/network, no dynamic import
#   - the tool.call guard has a .catch (fail closed) and ships with tests
#
# Run: bash plugins/agentdb/scripts/smoke-mod.sh     Exit 0 = contract holds.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS="$PLUGIN_DIR/hooks"

fail=0
pass() { printf '  ok   %s\n' "$1"; }
err()  { printf '  FAIL %s\n' "$1"; fail=1; }

echo "== agentdb mod contract =="

node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"));process.exit(Array.isArray(j.modules)&&j.modules.length===1&&j.modules[0]==="./register.ts"?0:1)' "$HOOKS/hooks.json" 2>/dev/null \
  && pass 'hooks.json modules == ["./register.ts"]' || err 'hooks.json must be valid JSON listing exactly ./register.ts'

for f in "$HOOKS"/*.ts; do
  lines=$(wc -l < "$f")
  [ "$lines" -lt 500 ] && pass "hooks/$(basename "$f") ($lines lines)" || err "hooks/$(basename "$f") is $lines lines (limit 500)"
done
grep -q "export const register: Register" "$HOOKS/register.ts" && pass "register.ts exports register: Register" || err "register.ts must export register: Register"

# One bare on(event, hook) per event.
dups=$(grep -oE "on\('[a-z.]+', *(async *)?\(" "$HOOKS/register.ts" | sed -E "s/on\('([a-z.]+)'.*/\1/" | sort | uniq -d)
[ -z "$dups" ] && pass "at most one bare on(event) per event" || err "more than one bare on() for: $dups"

# `$` stays in register.ts.
leaks=$(grep -lE '\$\.|\(\$[,)]' "$HOOKS"/*.ts | grep -v '/register.ts$' || true)
[ -z "$leaks" ] && pass "\$ is used only in register.ts" || err "\$ used outside register.ts: $leaks"

# Imports: only siblings or 'claude-code' (types only); nothing Node-shaped.
bad=$(grep -hE "^\s*import\b" "$HOOKS"/*.ts | grep -vE "from '(\./[a-z-]+|claude-code)'" || true)
[ -z "$bad" ] && pass "imports are ./<sibling> or 'claude-code'" || err "forbidden import: $bad"
grep -hE "^\s*import\s+\{" "$HOOKS"/*.ts | grep -q "from 'claude-code'" && err "value import from 'claude-code' (use import type)" || pass "'claude-code' imported for types only"
forbidden=$(grep -nE "\brequire\s*\(|\bimport\s*\(|\bprocess\.|from 'node:|\\\$\[|\\\$\.env\.(get|set)\(\s*[^'\" ]" "$HOOKS"/*.ts || true)
[ -z "$forbidden" ] && pass "no require/import()/process/computed \$[x] or \$.env names" || err "sandbox violation: $forbidden"

grep -q "\.catch(" "$HOOKS/register.ts" && pass "tool.call guard has a .catch handler" || err "tool.call guard needs a .catch handler"
for f in tests/mod.test.ts tests/secrets.test.ts; do [ -f "$PLUGIN_DIR/$f" ] && pass "$f exists" || err "$f missing"; done

echo
if [ "$fail" -eq 0 ]; then echo "agentdb mod contract holds"; else echo "agentdb mod contract violated"; fi
exit "$fail"
