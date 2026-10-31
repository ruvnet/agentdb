# agentdb (Claude Code plugin)

AgentDB's own MCP server for Claude Code: vector memory, reasoning patterns, reflexion episodes, a skill library, causal edges and RL feedback, from the `agentdb` npm package. The plugin also adds a memory specialist agent, three skills, a status command, and a small mod that keeps secrets out of memory and refuses destructive admin tools unless you turn them on.

## Which plugin do I want?

| Plugin | MCP server | Use it when |
|---|---|---|
| **`agentdb`** (this one, `ruvnet/agentdb` marketplace) | AgentDB's standalone server, `npx agentdb mcp start`, named `agentdb` | You want AgentDB directly, with or without ruflo. |
| `ruflo-agentdb` (`ruvnet/ruflo` marketplace) | ruflo's MCP server (`claude-flow`/`ruflo`), which bridges to AgentDB controllers | You already run ruflo and want AgentDB through its controller bridge, recall-into-prompt and console. |

They can be installed together. The servers have different names (`agentdb` here, ruflo's own there) and different tool names (`agentdb_pattern_store` here, `agentdb_pattern-store` in ruflo), and each mod guards only its own server's tools.

The other plugins in this marketplace (`agentdb-core`, `agentdb-memory`, `agentdb-causal`, `agentdb-learning`, `agentdb-graph`, `agentdb-search`) add commands and skills that call a server named `agentdb`. This plugin supplies that server.

## Install

```
/plugin marketplace add ruvnet/agentdb
/plugin install agentdb@agentdb
```

The install registers the MCP server `plugin:agentdb:agentdb`, which runs `npx -y agentdb@<plugin version> mcp start` (Node 18 or later; the first start downloads the package). Check it with `claude mcp list` or `/agentdb-guard status`. Tools appear as `mcp__plugin_agentdb_agentdb__<tool>`.

## Options (`userConfig`)

Set them with `/plugin configure agentdb@agentdb`, then restart the session so the server picks them up. Every option has a default, so the server starts with none set.

| Option | Sets | Default | Effect |
|---|---|---|---|
| `mcp_profile` | `AGENTDB_MCP_PROFILE` | empty = `learning` | Which tools the server exposes: `readonly`/`retrieval` (search, stats, recall), `learning` (plus memory writes, reflexion, skills, causal edges, RL and pattern/skill feedback, `agentdb_capabilities`), `admin` (everything, including `agentdb_init`, `agentdb_delete`, `agentdb_delete_batch`, `agentdb_clear_cache`). An unknown name stops the server. |
| `db_path` | `AGENTDB_PATH` | empty = `./agentdb.db` | The database file. Relative paths resolve from the directory Claude Code starts the server in. |
| `mcp_allow` | `AGENTDB_MCP_ALLOW` | empty | Tool names, comma or space separated, added to the profile. Naming an admin-only tool exposes it; the mod still refuses it unless `allow_destructive` is on. |
| `mcp_deny` | `AGENTDB_MCP_DENY` | empty | Tool names removed in any profile. |
| `secret_guard` | (mod) | `enforce` | `enforce` refuses memory writes that hold what looks like a key, token, password or private key; `notify` lets them through with a toast; `off` does not check. |
| `allow_destructive` | (mod) | `false` | When off, the mod refuses `agentdb_delete`, `agentdb_delete_batch`, `agentdb_clear_cache` and `agentdb_init` with `reset: true` (drops every table in the open database). Deleting data takes both exposing those tools (`mcp_profile=admin`, or naming them in `mcp_allow`) and this option. |

The four server options are passed as environment variables even when empty; the server treats an empty value as unset (`learning` profile, `./agentdb.db`, no extra allow or deny).

## What's included

| Kind | Name | Purpose |
|---|---|---|
| Agent | `agentdb-memory-specialist` | Stores and recalls memory, patterns, episodes and skills; records feedback. |
| Skill | `agentdb-vector-memory` | `agentdb_insert`/`agentdb_search`, patterns, certified recall. |
| Skill | `agentdb-reflexion-learning` | Reflexion episodes, skills from what worked, pattern/skill outcome feedback, causal edges, RL feedback, consolidation. |
| Skill | `agentdb-setup-diagnostics` | Connection, backend (`agentdb_capabilities`), profiles, database location, why a tool is missing or refused. |
| Command | `/agentdb-status` | Server health and counts. |
| Mod command | `/agentdb-guard status`, `/agentdb-guard scan <text>` | The guard's modes and counters; what the guard would refuse (the rule name, never the value). Answered locally, no model turn. |

## The mod

`hooks/` is a function-hooks module (ADR-445 shape, as `ruflo-agentdb` and `agentic-qe-fleet`). It hooks `tool.call` and judges only calls to an MCP server named `agentdb`:

- **Secret guard.** Writes through `agentdb_insert[_batch]`, `agentdb_pattern_store[_batch]`, `reflexion_store[_batch]`, `skill_create[_batch]`, `causal_add_edge`, `experience_record`, `learning_feedback`, `agentdb_pattern_feedback` and `agentdb_skill_feedback` are screened for token shapes (private keys, cloud and SaaS keys, JWTs, bearer tokens, URLs with credentials) and for credential-named fields with random-looking values. The refusal never repeats the value. Inputs too large to screen in full are refused.
- **Destructive guard.** The admin-only tools are refused unless `allow_destructive` is on. The server's default profile already hides them; the mod is a second layer for the admin profile.
- **Fail closed.** If the guard throws or overruns on a guarded call, the call is refused.

It is tighten-only (a refusal or the call unchanged), uses no network or processes, and writes no files.

## Development

```
bash plugins/agentdb/scripts/smoke.sh      # offline structural contract (+ validate/test when claude is installed)
claude plugin validate plugins/agentdb
claude plugin test plugins/agentdb         # tests/*.test.ts
```

The plugin version, the `.mcp.json` pin and the `agentdb` npm version move together; `smoke.sh` fails if the pin and `plugin.json` disagree.

## License

MIT
