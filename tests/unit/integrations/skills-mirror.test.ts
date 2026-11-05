/**
 * Codex reads skills from .agents/skills/<name>/SKILL.md; Claude Code reads
 * them from the plugins. The ecosystem-integration skills (ADR-018) ship in
 * both places and must not drift.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

const root = path.join(__dirname, '../../..');
const codexDir = path.join(root, '.agents/skills');

function pluginSkill(name: string): string | null {
  const plugins = path.join(root, 'plugins');
  for (const plugin of fs.readdirSync(plugins)) {
    const file = path.join(plugins, plugin, 'skills', name, 'SKILL.md');
    if (fs.existsSync(file)) return file;
  }
  return null;
}

const codexSkills = fs.existsSync(codexDir) ? fs.readdirSync(codexDir) : [];

describe('Codex skill mirrors', () => {
  it('ships the ecosystem-integration skills', () => {
    expect(codexSkills).toEqual(
      expect.arrayContaining(['agentdb-graph-partition', 'agentdb-quantized-search', 'agentdb-memory-guard'])
    );
  });

  it.each(codexSkills)('%s matches its Claude Code plugin copy', (name) => {
    const codex = fs.readFileSync(path.join(codexDir, name, 'SKILL.md'), 'utf8');
    const source = pluginSkill(name);
    expect(source, `no plugins/*/skills/${name}/SKILL.md`).not.toBeNull();
    expect(codex).toBe(fs.readFileSync(source!, 'utf8'));
    const front = codex.match(/^---\n([\s\S]*?)\n---\n/);
    expect(front).not.toBeNull();
    expect(front![1]).toMatch(new RegExp(`^name: ${name}$`, 'm'));
    expect(front![1]).toMatch(/^description: .{40,}/m);
  });
});
