---
name: agentdb-status
description: Show the agentdb MCP server's health — connection, backend, database counts, patterns and learning metrics
---

# AgentDB status

Report on AgentDB's own MCP server (`agentdb`, from this plugin) in a short summary:

1. Run `/agentdb-guard status` for the guard's modes and how many agentdb tools are connected. If none are, say the server is not connected, suggest `claude mcp list`, and stop.
2. Call `mcp__plugin_agentdb_agentdb__agentdb_capabilities` for the vector backend, SQL driver and embedding model (agentdb 3.1.0; skip it quietly if missing).
3. Call `mcp__plugin_agentdb_agentdb__db_stats` and `mcp__plugin_agentdb_agentdb__agentdb_stats` (`detailed: false`) for record counts and storage.
4. Call `mcp__plugin_agentdb_agentdb__agentdb_pattern_stats` for patterns and top task types.
5. Call `mcp__plugin_agentdb_agentdb__learning_metrics` if it is available (it is not under every profile; skip it quietly if missing).

Answer in at most ten lines: connected or not, the backend, the counts that are non-zero, top task types, and anything that looks wrong (a missing tool the user expects, an error). Do not write to AgentDB.
