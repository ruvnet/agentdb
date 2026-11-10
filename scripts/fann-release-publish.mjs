import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
const dir = resolve(process.argv[2]);
const run = (cmd, args, cwd = process.cwd(), env = process.env) => execFileSync(cmd, args, { cwd, env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 20 * 60_000 }).trim();
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const seal = JSON.parse(readFileSync(join(dir, 'candidate-seal.json'), 'utf8'));
if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_WORKFLOW !== 'protected-release' || process.env.GITHUB_REF !== 'refs/heads/main' || process.env.GITHUB_SHA !== seal.sourceSha || process.env.EXPECTED_SHA !== seal.sourceSha || run('git', ['rev-parse', 'HEAD']) !== seal.sourceSha) throw new Error('Protected exact-main release context required');
if (seal.schema !== 'agentdb.neural-release/1' || seal.name !== 'agentdb' || seal.version !== '3.8.2' || seal.filename !== 'agentdb-3.8.2.tgz' || seal.releaseTag !== 'v3.8.2' || seal.qualification?.packedConsumer !== 'passed') throw new Error('Invalid candidate seal');
if (run('gh', ['api', `repos/${process.env.GITHUB_REPOSITORY}/commits/main`, '--jq', '.sha']) !== seal.sourceSha) throw new Error('Main moved after qualification');
if (run('gh', ['issue', 'list', '--label', 'release-blocker', '--state', 'open', '--json', 'number', '--jq', 'length']) !== '0') throw new Error('Open release blockers');
const tests = JSON.parse(readFileSync(join(dir, 'library-tests.json'), 'utf8'));
if (hash(readFileSync(join(dir, 'library-tests.json'))) !== seal.qualification.testReportSha256 || !tests.success || tests.numPassedTests !== seal.qualification.testsPassed || tests.numPassedTests !== tests.numTotalTests || tests.numPassedTests < 1 || tests.numPendingTests || tests.numTodoTests) throw new Error('Candidate test evidence mismatch');
const tarball = join(dir, seal.filename);
if (hash(readFileSync(tarball)) !== seal.sha256) throw new Error('Candidate digest mismatch');
const metadata = JSON.parse(run('tar', ['-xOzf', tarball, 'package/package.json']));
if (metadata.name !== seal.name || metadata.version !== seal.version || metadata.dependencies?.['ruv-fann'] !== '0.1.0') throw new Error('Archive identity mismatch');
const publicURL = `https://registry.npmjs.org/agentdb/-/${seal.filename}`;
async function download() {
  const r = await fetch(publicURL, { redirect: 'error' });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Registry HTTP ${r.status}`);
  const bytes = Buffer.from(await r.arrayBuffer());
  if (hash(bytes) !== seal.sha256) throw new Error('Public version exists with different bytes');
  return bytes;
}
async function waitPublic() {
  const deadline = Date.now() + 16 * 60_000;
  while (Date.now() < deadline) {
    const bytes = await download(); if (bytes) return bytes;
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
  throw new Error('Registry propagation timeout: upload not downloadable after 16 minutes');
}
const consumerEnv = { ...process.env };
for (const key of ['NODE_AUTH_TOKEN', 'NPM_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN']) delete consumerEnv[key];
async function qualify(spec) {
  const temp = mkdtempSync(join(tmpdir(), 'agentdb-public-'));
  try {
    writeFileSync(join(temp, 'package.json'), '{"private":true,"type":"module"}');
    writeFileSync(join(temp, '.npmrc'), 'registry=https://registry.npmjs.org\n');
    const env = { ...consumerEnv, NPM_CONFIG_USERCONFIG: join(temp, '.npmrc'), NPM_CONFIG_CACHE: join(temp, 'npm-cache') };
    run('npm', ['install', '--omit=optional', '--ignore-scripts', '--no-audit', '--no-fund', spec], temp, env);
    const lock = JSON.parse(readFileSync(join(temp, 'package-lock.json'), 'utf8'));
    const expectedIntegrity = 'sha512-' + createHash('sha512').update(readFileSync(tarball)).digest('base64');
    if (lock.packages?.['node_modules/agentdb']?.integrity !== expectedIntegrity) throw new Error('Installed public package differs from sealed bytes');
    console.log(run(process.execPath, [resolve('scripts/fann-packed-consumer.mjs')], temp, env));
  } finally { rmSync(temp, { recursive: true, force: true }); }
}
// Requalify the same packed bytes on Node24 before any irreversible operation.
await qualify(tarball);
if (!await download()) {
  const result = spawnSync('npm', ['publish', tarball, '--ignore-scripts', '--access', 'public', '--provenance', '--registry=https://registry.npmjs.org'], { stdio: 'inherit' });
  if (result.status !== 0) console.error('Publish returned nonzero; checking for an earlier accepted upload before failing.');
}
const bytes = await waitPublic();
writeFileSync(join(dir, 'npm-public.tgz'), bytes);
await qualify('agentdb@3.8.2');
const receipt = { schema: 'agentdb.neural-publication/1', sourceSha: seal.sourceSha, name: seal.name, version: seal.version, sha256: seal.sha256, publicConsumer: 'passed', node: process.version };
const receiptPath = join(dir, 'publication-receipt.json');
writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
const notes = join(dir, 'release-notes.md');
writeFileSync(notes, 'Pins @ruvector/rvf-wasm 0.1.13, @ruvector/rvf-solver 0.1.11 and @ruvector/rabitq-wasm 0.1.2 exactly (were ^0.x ranges), fixes CLI write commands exiting 1 with \"this.db.save is not a function\" on native better-sqlite3, and corrects the README quickstart. See CHANGELOG.md.\n');
function verifyTag(allowMissing = false) {
  const result = spawnSync('gh', ['api', `repos/${process.env.GITHUB_REPOSITORY}/git/ref/tags/${seal.releaseTag}`], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  if (result.status !== 0) {
    if (allowMissing && result.stderr.includes('404')) return;
    throw new Error('Cannot verify release tag');
  }
  let object = JSON.parse(result.stdout).object;
  for (let depth = 0; object.type === 'tag' && depth < 5; depth++) object = JSON.parse(run('gh', ['api', `repos/${process.env.GITHUB_REPOSITORY}/git/tags/${object.sha}`])).object;
  if (object.type !== 'commit' || object.sha !== seal.sourceSha) throw new Error('Release tag does not resolve to sealed source SHA');
}
verifyTag(true);
const existing = spawnSync('gh', ['release', 'view', seal.releaseTag, '--json', 'tagName'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
if (existing.status !== 0) run('gh', ['release', 'create', seal.releaseTag, tarball, join(dir, 'candidate-seal.json'), receiptPath, '--target', seal.sourceSha, '--title', 'AgentDB 3.8.2 — exact WASM pins, CLI writes exit 0 on native SQLite', '--notes-file', notes]);
verifyTag();
const downloaded = mkdtempSync(join(tmpdir(), 'agentdb-release-'));
try {
  run('gh', ['release', 'download', seal.releaseTag, '--dir', downloaded]);
  for (const file of [seal.filename, 'candidate-seal.json', 'publication-receipt.json']) if (hash(readFileSync(join(downloaded, file))) !== hash(readFileSync(join(dir, file)))) throw new Error(`GitHub artifact mismatch: ${file}`);
} finally { rmSync(downloaded, { recursive: true, force: true }); }
writeFileSync(join(dir, 'publication-verified.json'), JSON.stringify({ ...receipt, githubArtifacts: 'byte-equal', status: 'verified' }, null, 2) + '\n', { flag: 'wx' });
console.log('AgentDB npm and GitHub release artifacts verified against the candidate seal.');
