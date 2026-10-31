---
name: agentdb-setup-diagnostics
description: Set up and diagnose the agentdb plugin and its MCP server — which profile and database are in use, whether the server is connected, what it holds, and why a tool is missing or refused. Use when AgentDB tools are missing, a call is refused, the user asks "is agentdb working", or wants to change the profile or database path.
allowed-tools:
  - mcp__plugin_agentdb_agentdb__agentdb_stats
  - mcp__plugin_agentdb_agentdb__db_stats
  - mcp__plugin_agentdb_agentdb__agentdb_pattern_stats
  - mcp__plugin_agentdb_agentdb__learning_metrics
  - Bash(claude mcp list)
  - Bash(claude plugin details agentdb)
---

# AgentDB setup and diagnostics

## 1. Is the server connected?

Run `/agentdb-guard status`: it reports the guard's modes and how many `agentdb` tools are connected. No tools means the server did not start; check `claude mcp list` for `plugin:agentdb:agentdb` and its error. The server runs `npx -y agentdb@<plugin version> mcp start`, so the first start downloads the package and needs network access and Node 18 or later.

## 2. What does it hold?

- `agentdb_stats(detailed: true)`: table counts, storage, performance.
- `db_stats()`: record counts.
- `agentdb_pattern_stats()`: patterns, success rates, top task types.
- `learning_metrics()`: RL session success rates and rewards.

An empty database is normal on first use. Data lives in `AGENTDB_PATH` (the `db_path` option), else `./agentdb.db` in the directory Claude Code started the server from, so a different project directory means a different database.

## 3. Why is a tool missing?

The server only lists the tools its profile allows (`mcp_profile` option, env `AGENTDB_MCP_PROFILE`):

| Profile | Tools |
|---|---|
| `readonly` / `retrieval` | search, stats, recall, predict |
| `learning` (default, also when empty) | the above plus memory writes, reflexion, skills, causal edges, RL feedback |
| `admin` | everything, including `agentdb_init`, `agentdb_delete`, `agentdb_delete_batch`, `agentdb_clear_cache` |

`mcp_allow` adds named tools to a profile; `mcp_deny` removes them in any profile. An unknown profile name stops the server from starting.

## 4. Why was a call refused?

The plugin's mod refuses, before the server sees them:

- memory writes that hold what looks like a secret (`secret_guard`: `enforce` default, `notify`, `off`). `/agentdb-guard scan <text>` names the rule a text trips, never the value.
- the destructive tools above, and `agentdb_init` with `reset: true`, unless `allow_destructive` is on. Turning them on takes both `mcp_profile=admin` and `allow_destructive`.

## 5. Change settings

`/plugin configure agentdb@agentdb` (or `claude plugin configure agentdb`), then restart the session so the server restarts with the new environment.
