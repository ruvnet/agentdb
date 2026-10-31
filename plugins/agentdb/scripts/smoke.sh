#!/usr/bin/env bash
# Structural contract for the agentdb plugin. Runs offline in a scrubbed sandbox
# (temp HOME, no credentials, no network): it reads files only, plus the
# engine's own `claude plugin validate` / `claude plugin test` when `claude` is
# on PATH. No version string is written here; the version is read from plugin.json.
#
# Run: bash plugins/agentdb/scripts/smoke.sh     Exit 0 = contract holds.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PLUGIN_DIR/../.." && pwd)"
PLUGIN_NAME="agentdb"

fail=0
pass() { printf '  ok   %s\n' "$1"; }
err()  { printf '  FAIL %s\n' "$1"; fail=1; }

# Prints the value of a JS expression over the parsed JSON file `$1` (the file is `j`), or fails.
jget() { node -e 'const j=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"));const v=eval(process.argv[2]);if(v===undefined)process.exit(1);process.stdout.write(typeof v==="string"?v:JSON.stringify(v))' "$1" "$2" 2>/dev/null; }

echo "== agentdb plugin contract =="

command -v node >/dev/null 2>&1 || { echo "  FAIL node is required"; exit 1; }

MANIFEST="$PLUGIN_DIR/.claude-plugin/plugin.json"
MCP_JSON="$PLUGIN_DIR/.mcp.json"
MARKETPLACE="$REPO_ROOT/.claude-plugin/marketplace.json"

# 1. Manifests are valid JSON.
for f in "$MANIFEST" "$MCP_JSON" "$MARKETPLACE" "$PLUGIN_DIR/hooks/hooks.json"; do
  jget "$f" 'j' >/dev/null && pass "$(basename "$(dirname "$f")")/$(basename "$f") is valid JSON" || err "$f missing or invalid JSON"
done

# 2. Name and version.
[ "$(jget "$MANIFEST" 'j.name')" = "$PLUGIN_NAME" ] && pass "plugin.json name == $PLUGIN_NAME" || err "plugin.json name != $PLUGIN_NAME"
VERSION="$(jget "$MANIFEST" 'j.version')"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] && pass "plugin.json version $VERSION" || err "plugin.json version missing or not semver"
jget "$MANIFEST" 'j.keywords.includes("mod")&&j.keywords.includes("function-hooks")||undefined' >/dev/null \
  && pass "keywords include mod and function-hooks" || err "keywords must include mod and function-hooks"
jget "$MANIFEST" '("skills" in j)||("commands" in j)||("agents" in j)||undefined' >/dev/null \
  && err "plugin.json declares skills/commands/agents (rely on auto-discovery)" || pass "plugin.json relies on auto-discovery"

# 3. The MCP server: named agentdb, `mcp start`, pinned to the plugin version.
jget "$MCP_JSON" 'j.mcpServers.agentdb' >/dev/null && pass ".mcp.json registers server agentdb" || err ".mcp.json has no server named agentdb"
jget "$MCP_JSON" 'Object.keys(j.mcpServers).length===1||undefined' >/dev/null && pass ".mcp.json has exactly one server" || err ".mcp.json must have exactly one server"
PIN="$(jget "$MCP_JSON" 'j.mcpServers.agentdb.args.find(a=>a.startsWith("agentdb@"))')"
[ "$PIN" = "agentdb@$VERSION" ] && pass ".mcp.json pins $PIN (== plugin.json version)" || err ".mcp.json pin '${PIN:-<none>}' != agentdb@$VERSION"
[ "$(jget "$MCP_JSON" 'j.mcpServers.agentdb.args.slice(-2).join(" ")')" = "mcp start" ] && pass ".mcp.json starts 'mcp start'" || err ".mcp.json must end its args with: mcp start"

# 4. Every ${user_config.X} is declared with a default.
for key in $(grep -oE '\$\{user_config\.[A-Za-z0-9_]+\}' "$MCP_JSON" | sed -E 's/.*\.([A-Za-z0-9_]+)\}/\1/' | sort -u); do
  jget "$MANIFEST" "('default' in (j.userConfig||{})['$key'])||undefined" >/dev/null \
    && pass "userConfig.$key declared with a default" || err "userConfig.$key used by .mcp.json but not declared with a default"
done

# 5. The marketplace lists this plugin at its folder.
jget "$MARKETPLACE" "j.plugins.find(p=>p.name==='$PLUGIN_NAME'&&p.source==='./plugins/$PLUGIN_NAME')" >/dev/null \
  && pass "marketplace.json lists $PLUGIN_NAME at ./plugins/$PLUGIN_NAME" || err "marketplace.json does not list $PLUGIN_NAME at ./plugins/$PLUGIN_NAME"

# 6. Skills: SKILL.md front matter whose name matches the folder, with a description; no wildcard tools.
n=0
for d in "$PLUGIN_DIR"/skills/*/; do
  s="$d/SKILL.md"; dir="$(basename "$d")"; n=$((n + 1))
  [ -f "$s" ] || { err "skill $dir has no SKILL.md"; continue; }
  fm="$(awk 'NR==1&&/^---/{f=1;next} f&&/^---/{exit} f' "$s")"
  name="$(printf '%s\n' "$fm" | sed -nE 's/^name:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/p' | head -1)"
  [ "$name" = "$dir" ] && pass "skill $dir: name matches folder" || err "skill $dir: front-matter name '${name:-<none>}' != folder"
  printf '%s\n' "$fm" | grep -qE '^description:' || err "skill $dir: no description"
  printf '%s\n' "$fm" | grep -qE '\*' && err "skill $dir: wildcard in front matter"
done
[ "$n" -ge 1 ] && pass "skills: $n" || err "no skills"

# 7. Agents and commands carry name + description; agents a model.
for f in "$PLUGIN_DIR"/agents/*.md "$PLUGIN_DIR"/commands/*.md; do
  head -8 "$f" | grep -qE '^name:' && head -8 "$f" | grep -qE '^description:' \
    && pass "$(basename "$(dirname "$f")")/$(basename "$f") front matter" || err "$f missing name/description"
done
for f in "$PLUGIN_DIR"/agents/*.md; do head -8 "$f" | grep -qE '^model:' || err "agent $(basename "$f") has no model"; done

# 8. Docs.
for f in README.md CHANGELOG.md; do [ -s "$PLUGIN_DIR/$f" ] && pass "$f present" || err "$f missing"; done
grep -qF "## [$VERSION]" "$PLUGIN_DIR/CHANGELOG.md" && pass "CHANGELOG has an entry for $VERSION" || err "CHANGELOG has no '## [$VERSION]' entry"

# 9. The mod's own contract.
bash "$SCRIPT_DIR/smoke-mod.sh" >/dev/null 2>&1 && pass "smoke-mod.sh passed" || { err "smoke-mod.sh failed:"; bash "$SCRIPT_DIR/smoke-mod.sh" | grep FAIL; }

# 10. The engine's checks, when Claude Code is installed.
if command -v claude >/dev/null 2>&1; then
  out="$(cd "$REPO_ROOT" && claude plugin validate "plugins/$PLUGIN_NAME" 2>&1)" && pass "claude plugin validate" || { err "claude plugin validate"; printf '%s\n' "$out" | tail -15; }
  out="$(cd "$REPO_ROOT" && claude plugin test "plugins/$PLUGIN_NAME" 2>&1)" && pass "claude plugin test ($(printf '%s\n' "$out" | grep -oE '^ *[0-9]+ pass' | tr -d ' '))" || { err "claude plugin test"; printf '%s\n' "$out" | tail -25; }
else
  pass "claude not on PATH: validate/test skipped"
fi

echo "======================================="
if [ "$fail" -eq 0 ]; then echo "PASS: agentdb plugin contract holds"; exit 0; fi
echo "FAIL: agentdb plugin contract violated"; exit 1
