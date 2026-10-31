/**
 * ReflexionMemory.buildCypherQuery and GraphDatabaseAdapter spliced
 * minReward / k into Cypher text unchecked. Latent (callers pass numbers),
 * but a string reaching them became query text.
 */
import { describe, it, expect } from 'vitest';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory.js';
import { GraphDatabaseAdapter } from '../../src/backends/graph/GraphDatabaseAdapter.js';

describe('Cypher interpolation', () => {
  it('ReflexionMemory rejects a non-numeric minReward', () => {
    const memory = Object.create(ReflexionMemory.prototype);
    const build = (q: any) => (memory as any).buildCypherQuery(q);

    expect(() => build({ task: 't', minReward: '0 OR true' })).toThrow(/finite number/);
    expect(() => build({ task: 't', k: '1 MATCH (n) DETACH DELETE n' })).toThrow();
    expect(build({ task: 't', minReward: '0.5', k: 2 })).toContain('e.reward >= 0.5');
  });

  it('GraphDatabaseAdapter rejects a non-integer k', async () => {
    const queries: string[] = [];
    const adapter = Object.create(GraphDatabaseAdapter.prototype);
    adapter.query = async (cypher: string) => { queries.push(cypher); return { nodes: [] }; };

    await expect(adapter.searchSkills(new Float32Array(4), '1 MATCH (n) DETACH DELETE n')).rejects.toThrow();
    await expect(adapter.searchSimilarEpisodes(new Float32Array(4), 'x')).rejects.toThrow();
    expect(queries).toEqual([]);

    await adapter.searchSkills(new Float32Array(4), 3);
    expect(queries).toEqual(['MATCH (s:Skill) RETURN s LIMIT 3']);
  });
});
