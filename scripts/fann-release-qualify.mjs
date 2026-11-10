import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const workflow = readFileSync('.github/workflows/library.yml', 'utf8');
const line = workflow.split('\n').find(line => line.includes('run: npx vitest run '));
if (!line) throw new Error('Library regression command missing');
execFileSync('npx', [...line.trim().slice('run: npx '.length).split(/\s+/), '--reporter=json', '--outputFile=/tmp/fann-release-tests.json'], { stdio: 'inherit' });
const report = JSON.parse(readFileSync('/tmp/fann-release-tests.json', 'utf8'));
if (!report.success || report.numTotalTests < 1 || report.numPassedTests !== report.numTotalTests || report.numPendingTests || report.numTodoTests) throw new Error('All nonzero release tests must pass without skips');
