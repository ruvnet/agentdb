# Changelog

## [3.1.0]

First release of the `agentdb` Claude Code plugin. Its version follows the `agentdb` npm package it starts.

- `.mcp.json` registers AgentDB's own stdio MCP server as `agentdb`: `npx -y agentdb@3.1.0 mcp start`, pinned to the plugin version.
- `userConfig`: `mcp_profile` (`AGENTDB_MCP_PROFILE`), `db_path` (`AGENTDB_PATH`), `mcp_allow` (`AGENTDB_MCP_ALLOW`), `mcp_deny` (`AGENTDB_MCP_DENY`), all empty by default, which the server reads as its own defaults; plus `secret_guard` and `allow_destructive` for the mod.
- Mod (`hooks/`): a tighten-only `tool.call` guard on the `agentdb` server that refuses memory writes holding what looks like a secret, and refuses `agentdb_delete`, `agentdb_delete_batch`, `agentdb_clear_cache` and `agentdb_init` with `reset` unless `allow_destructive` is on. Fails closed. `/agentdb-guard status|scan`.
- Agent `agentdb-memory-specialist`; skills `agentdb-vector-memory`, `agentdb-reflexion-learning`, `agentdb-setup-diagnostics`; command `/agentdb-status`.
- `scripts/smoke.sh` and `scripts/smoke-mod.sh`: offline structural contract.
