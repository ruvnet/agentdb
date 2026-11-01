# ADR-017: Full CRUD for episodes, reasoning patterns and skills (3.3.0)

- **Status:** Accepted
- **Date:** 2026-11-01
- **Release:** agentdb 3.3.0

## Context

Before 3.3.0 agentdb could create, search and delete records but not update or fetch one by id:

| | create | read by id | list | update | delete |
|---|---|---|---|---|---|
| episodes (`agentdb_insert` records) | yes | no (search only) | no | no | yes |
| reasoning patterns | yes | library only | no | stats only | library only |
| skills | yes | no | no | stats only | prune only |

The only way to change a record was delete + insert, which re-embeds it and gives it a new id, breaking
every reference (causal edges, skill `created_from_episode`, callers' own ids).

Reviewing the delete paths for this work also found defects that a CRUD release must not ship on top of:

1. `ReflexionMemory.deleteEpisode` called `vectorBackend.delete()`, but `VectorBackend` exposes `remove()`,
   so a deleted episode stayed in the vector index and kept occupying top-k slots.
2. `deleteEpisode` parsed string ids with `parseInt`, so `"12-notes"` deleted episode 12.
3. `ReasoningBank.deletePattern` never removed the pattern's vector-index entry.
4. `pruneEpisodes`, `pruneSkills`, `agentdb_delete` (bulk filters) and `agentdb_delete_batch` left orphaned
   embedding rows (and, for skills, `skill_links`) because SQLite foreign-key cascades are off by default.
5. `agentdb_insert` / `agentdb_insert_batch` did not validate `text`, `session_id`, `tags` or `metadata`.
6. The MCP server's catch-all returned `error.stack` to the client.
7. `agentdb_init` failed on better-sqlite3 (the default driver): it used a double-quoted SQL string
   literal (`type="table"`), which current SQLite builds reject.

## Decision

### API

| | episodes (`ReflexionMemory`) | patterns (`ReasoningBank`) | skills (`SkillLibrary`) |
|---|---|---|---|
| create | `storeEpisode` | `storePattern` | `createSkill` |
| read | `getEpisode(id, {includeEmbedding})` | `getPattern(id)` | `getSkill(id)` |
| list | `listEpisodes({sessionId, success, limit, offset, order})` | `listPatterns({taskType, limit, offset})` | `listSkills({minSuccessRate, limit, offset})` |
| update | `updateEpisode(id, patch, {replaceMetadata})` | `updatePattern(id, patch, …)` | `updateSkill(id, patch, …)` |
| delete | `deleteEpisode(id)` | `deletePattern(id)` | `deleteSkill(id)` |

MCP tools: `agentdb_get`, `agentdb_list`, `agentdb_update`, `agentdb_pattern_get`, `agentdb_pattern_list`,
`agentdb_pattern_update`, `agentdb_pattern_delete`, `skill_get`, `skill_list`, `skill_update`, `skill_delete`.

### Patch semantics (all three record types)

- Absent / `undefined` leaves a field alone; `null` clears a nullable field.
- Ids, creation timestamps, the episode's session and **usage statistics** (`uses`, `successRate`,
  `avgReward`, `avgLatencyMs`) are not writable by an update. Statistics keep their existing writers
  (`recordOutcome`, `updatePatternStats`, `agentdb_*_feedback`) so a content edit can never fake a track record.
- Unknown patch fields are rejected, not ignored, so a typo is never a silent no-op.
- `metadata` merges by default (a `null` value removes that key); `replaceMetadata` replaces it. The merged
  result, not just the patch, is size-checked (64 KB) and scanned at every depth for `__proto__`,
  `constructor` and `prototype`.
- A skill rename onto an existing name is a validation error, not a leaked SQLite message.
- Not found is `null` (get/update) or `false` (delete), never a throw.

### Consistency

An update moves every store that holds the record, or fails before writing:

1. The new embedding is computed **only** when a field it is built from changes (episode task / critique /
   output; pattern taskType / approach; skill name / description / signature). Metadata, tag, reward and
   code edits never call the embedder.
2. The embedding is computed before the transaction (driver transactions are synchronous). Inside the
   transaction the row is re-read; if its embedded text changed meanwhile the update aborts with
   "modified concurrently" rather than storing a vector for text that is no longer there.
3. The SQL row and the embedding row (now an upsert) are written in one transaction
   (`transaction()` on better-sqlite3, `BEGIN/COMMIT/ROLLBACK` on sql.js).
4. The vector-backend entry is replaced (`remove` + `insert`, same id).
5. Query caches are invalidated, and the MCP server's tool caches (`stats`, `patterns`) are cleared.
6. The BM25 keyword index is patched in place (`Bm25Index.replace`), so hybrid search returns the new text
   immediately. Its refresh signature is `(COUNT, MAX(id))`, which an UPDATE does not change; without the
   patch, keyword search would keep matching the old text.

Deletes remove the embedding row, the vector entry and (skills) links in both directions explicitly, since
cascades cannot be assumed.

Under a **graph backend** the id `storeEpisode` / `createSkill` return is derived from the graph node and is
not the SQL row id, so the id-addressed get/list/update/delete methods refuse with a clear error instead of
touching the wrong row.

### Security

- MCP profiles: get/list are `readonly`; updates are in `learning` (they keep ids and cannot touch
  statistics, like inserts); `agentdb_pattern_delete` and `skill_delete` are `admin`-only like
  `agentdb_delete`.
- The Claude Code plugin's mod screens `agentdb_update`, `agentdb_pattern_update` and `skill_update` for
  secrets like any write, and refuses the two new delete tools unless `allow_destructive` is on.
- Ids are strict positive integers (`"12-notes"` is rejected everywhere). Page size is capped at 500.
- Text fields are length-capped and stripped of NUL bytes but not filtered by content: stored text may be
  code, and `validateTaskString`'s `on\w+=` heuristic rejects ordinary text such as `reason=5`.
- `sanitizeMetadata` (which drops any key containing "key", "pin", "auth", …, e.g. `keywords`, `mapping`)
  is deliberately not applied to the new paths; secret screening is the plugin mod's job.
- Client-facing errors carry the message only; stacks go to the server log.

## Measured

`benchmarks/crud-benchmark.mts`, better-sqlite3 in memory, Node 22, single thread. p50 / p99 in ms.

Mock embedder (DB and index work only), 10,000 episodes, 500 ops each:

| operation | p50 | p99 |
|---|---|---|
| episode get | 0.012 | 0.033 |
| episode list, page of 50 (session filter) | 0.120 | 0.384 |
| episode update, metadata only | 0.042 | 0.139 |
| episode update, text (re-embed) | 0.116 | 0.262 |
| delete + insert (the only option before 3.3.0) | 0.086 | 0.193 |
| episode delete | 0.009 | 0.022 |
| hybrid search, no writes | 33.1 | 61.3 |
| hybrid search right after a text update | 33.4 | 52.6 |

Real embedder (`Xenova/all-MiniLM-L6-v2` via transformers.js, CPU), 2,000 episodes, 100 ops each:

| operation | p50 | p99 |
|---|---|---|
| episode update, metadata only | 0.038 | 0.788 |
| delete + insert | 1.129 | 6.593 |
| episode update, text (re-embed) | 2.482 | 11.016 |
| pattern update, approach (re-embed) | 1.470 | 7.854 |
| skill update, description (re-embed) | 2.213 | 12.161 |
| hybrid search, no writes / after update | 12.74 / 12.79 | 29.1 / 31.8 |

Readings:

- A metadata/tag/reward edit is about 30x cheaper than the old delete + insert with a real embedder,
  because it skips the embedder, and it keeps the id.
- A text edit costs about one embedding, as delete + insert does; its benefit is the stable id and the
  consistent indexes, not speed.
- Patching the keyword index in place removed a full BM25 rebuild from the first search after an edit:
  with lazy rebuild-on-invalidate that search was 46.8 ms p50 against 33.1 ms for a search without writes
  (10,000 episodes); with `Bm25Index.replace` it is 33.4 ms.

## Verification

- `tests/unit/controllers/crud-adr017.test.ts` (20 tests). The keyword-index, prune-orphan and
  pattern-vector tests were each confirmed to fail with their fix removed.
- `tests/mcp-crud-e2e.mjs` (`npm run test:mcp-crud`): every tool over stdio against the built server, on
  sql.js and on better-sqlite3, plus a check that the default `learning` profile hides and refuses the
  delete tools. Asserts stdout stays pure JSON-RPC and that errors carry no stack trace.
- Plugin: `claude plugin test plugins/agentdb` (21 pass), `smoke.sh`, `smoke-mod.sh`.

## Consequences

- No breaking change. `deleteEpisode` now returns `false` for a malformed id instead of deleting a
  different episode; callers that relied on `parseInt` coercion were deleting the wrong record.
- `agentdb_insert` now rejects empty text, non-object metadata and malformed tags it used to store.
