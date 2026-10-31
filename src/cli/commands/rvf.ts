/**
 * RVF CLI Command - Manage RVF vector stores and AGI capabilities
 *
 * Provides subcommands for:
 * - status: Show RVF store status and statistics
 * - compact: Reclaim dead space in an RVF store
 * - derive: Create a COW branch from an existing store
 * - segments: List store segments and their metadata
 * - detect: Detect RVF SDK and backend availability
 * - witness: Verify SHAKE-256 witness chain integrity
 * - freeze: Snapshot-freeze store state
 * - index-stats: Show HNSW index statistics
 * - solver train: Train the self-learning solver
 * - solver test: Run A/B/C acceptance test
 */

import { Command } from 'commander';
import * as fs from 'fs';
import * as path from 'path';

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function log(msg: string): void {
  console.log(msg);
}

/**
 * Parse an integer from CLI input with NaN guard
 */
function safeParseInt(value: string, name: string): number {
  const n = parseInt(value, 10);
  if (Number.isNaN(n)) {
    throw new Error(`Invalid value for ${name}: "${value}" (expected integer)`);
  }
  return n;
}

/**
 * Validate a CLI-supplied store path against traversal and dangerous patterns.
 * Resolves to absolute path and checks against denylist.
 */
function validateStorePath(inputPath: string): string {
  if (!inputPath || typeof inputPath !== 'string') {
    throw new Error('Store path must be a non-empty string');
  }
  if (inputPath.includes('\0')) {
    throw new Error('Store path must not contain null bytes');
  }

  const resolved = path.resolve(inputPath);
  const dangerous = ['/etc/', '/proc/', '/sys/', '/dev/', '/boot/', '/root/'];
  for (const prefix of dangerous) {
    if (resolved.startsWith(prefix)) {
      throw new Error(`Store path must not be inside ${prefix}`);
    }
  }
  if (/\.\.[\\/]/.test(inputPath)) {
    throw new Error('Store path must not contain path traversal');
  }
  return resolved;
}

/**
 * Load RVF backend lazily
 */
async function loadRvfBackend(storePath: string, options: { backend?: string } = {}) {
  const validPath = validateStorePath(storePath);
  const { RvfBackend } = await import('../../backends/rvf/RvfBackend.js');
  const backend = new RvfBackend({
    dimension: 384, // Will be overridden on load
    metric: 'cosine',
    storagePath: validPath,
    rvfBackend: (options.backend as 'auto' | 'node' | 'wasm') ?? 'auto',
  });
  // RvfBackend has initialize() but it's not in the VectorBackend interface
  await (backend as unknown as { initialize(): Promise<void> }).initialize();
  return backend;
}

/**
 * Main rvf command
 */
export const rvfCommand = new Command('rvf')
  .description('RVF vector store management')
  .addCommand(
    new Command('status')
      .description('Show RVF store status and statistics')
      .argument('<store>', 'Path to .rvf store file')
      .option('--json', 'Output as JSON')
      .action(async (store: string, opts: { json?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);
          const status = await backend.status();
          const fileId = await backend.fileId();
          const parentId = await backend.parentId();
          const lineageDepth = await backend.lineageDepth();
          const perf = backend.getPerformanceStats();

          if (opts.json) {
            console.log(JSON.stringify({
              file: store,
              fileId,
              parentId,
              lineageDepth,
              totalVectors: status.totalVectors,
              totalSegments: status.totalSegments,
              performance: perf,
            }, null, 2));
          } else {
            log(`\n${colors.bright}${colors.cyan}RVF Store Status${colors.reset}\n`);
            log(`  File:        ${colors.blue}${store}${colors.reset}`);
            log(`  File ID:     ${colors.blue}${fileId}${colors.reset}`);
            log(`  Parent ID:   ${colors.blue}${parentId}${colors.reset}`);
            log(`  Lineage:     ${colors.blue}${lineageDepth}${colors.reset}`);
            log(`  Vectors:     ${colors.green}${status.totalVectors}${colors.reset}`);
            log(`  Segments:    ${colors.green}${status.totalSegments}${colors.reset}`);

            if (fs.existsSync(store)) {
              const stat = fs.statSync(store);
              const sizeMB = (stat.size / 1024 / 1024).toFixed(2);
              log(`  File Size:   ${colors.blue}${sizeMB} MB${colors.reset}`);
            }
            log('');
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('compact')
      .description('Compact an RVF store to reclaim dead space')
      .argument('<store>', 'Path to .rvf store file')
      .option('--dry-run', 'Show what would be compacted without making changes')
      .action(async (store: string, opts: { dryRun?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);

          if (opts.dryRun) {
            const segments = await backend.segments();
            const status = await backend.status();
            log(`\n${colors.bright}${colors.cyan}RVF Compaction Analysis (dry run)${colors.reset}\n`);
            log(`  Vectors:  ${status.totalVectors}`);
            log(`  Segments: ${segments.length}`);
            log(`  Estimated reclaimable space: (run without --dry-run to see)`);
            log('');
          } else {
            log(`\nCompacting ${store}...`);
            const result = await backend.compact();
            log(`${colors.green}Compaction complete${colors.reset}`);
            log(`  Segments compacted: ${result.segmentsCompacted}`);
            log(`  Bytes reclaimed:    ${result.bytesReclaimed}`);
            log('');
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('derive')
      .description('Create a COW branch from an existing RVF store')
      .argument('<parent>', 'Path to parent .rvf store')
      .argument('<child>', 'Path for new child .rvf store')
      .action(async (parent: string, child: string) => {
        try {
          const validChild = validateStorePath(child);
          if (fs.existsSync(validChild)) {
            console.error(`${colors.red}Error: Child path already exists: ${validChild}${colors.reset}`);
            process.exit(1);
          }

          const backend = await loadRvfBackend(parent);
          log(`\nDeriving branch from ${parent}...`);
          const childBackend = await backend.derive(validChild);
          const childId = await childBackend.fileId();
          const parentId = await childBackend.parentId();

          log(`${colors.green}Branch created successfully${colors.reset}`);
          log(`  Parent:      ${parent}`);
          log(`  Child:       ${child}`);
          log(`  Child ID:    ${childId}`);
          log(`  Parent ID:   ${parentId}`);
          log(`  Lineage:     ${await childBackend.lineageDepth()}`);
          log('');

          childBackend.close();
          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('segments')
      .description('List segments in an RVF store')
      .argument('<store>', 'Path to .rvf store file')
      .option('--json', 'Output as JSON')
      .action(async (store: string, opts: { json?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);
          const segments = await backend.segments();

          if (opts.json) {
            console.log(JSON.stringify(segments, null, 2));
          } else {
            log(`\n${colors.bright}${colors.cyan}RVF Store Segments${colors.reset}\n`);
            if (segments.length === 0) {
              log('  No segments found.');
            } else {
              log(`  ${'ID'.padEnd(6)} ${'Type'.padEnd(16)} ${'Size (bytes)'.padStart(14)}`);
              log(`  ${'─'.repeat(6)} ${'─'.repeat(16)} ${'─'.repeat(14)}`);
              for (const seg of segments) {
                log(`  ${String(seg.id).padEnd(6)} ${seg.segType.padEnd(16)} ${String(seg.payloadLength).padStart(14)}`);
              }
            }
            log(`\n  Total: ${segments.length} segments\n`);
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('detect')
      .description('Detect RVF SDK and backend availability')
      .option('--json', 'Output as JSON')
      .action(async (opts: { json?: boolean }) => {
        const result: Record<string, unknown> = {
          sdk: false,
          node: false,
          wasm: false,
          sdkVersion: null,
        };

        try {
          const rvf = await import('@ruvector/rvf');
          result.sdk = true;
          result.sdkVersion = (rvf as Record<string, unknown>).version ?? 'unknown';
        } catch {
          // SDK not available
        }

        try {
          await import('@ruvector/rvf-node');
          result.node = true;
        } catch {
          // N-API not available
        }

        try {
          await import('@ruvector/rvf-wasm');
          result.wasm = true;
        } catch {
          // WASM not available
        }

        result.platform = `${process.platform}-${process.arch}`;

        if (opts.json) {
          console.log(JSON.stringify(result, null, 2));
        } else {
          log(`\n${colors.bright}${colors.cyan}RVF Backend Detection${colors.reset}\n`);
          log(`  SDK:       ${result.sdk ? `${colors.green}Yes${colors.reset} (${result.sdkVersion})` : `${colors.red}No${colors.reset}`}`);
          log(`  N-API:     ${result.node ? `${colors.green}Yes${colors.reset}` : `${colors.yellow}No${colors.reset}`}`);
          log(`  WASM:      ${result.wasm ? `${colors.green}Yes${colors.reset}` : `${colors.yellow}No${colors.reset}`}`);
          log(`  Platform:  ${colors.blue}${result.platform}${colors.reset}`);

          if (!result.sdk) {
            log(`\n  Install: ${colors.cyan}npm install @ruvector/rvf${colors.reset}`);
          }
          if (!result.node && !result.wasm) {
            log(`  Native:  ${colors.cyan}npm install @ruvector/rvf-node${colors.reset}`);
            log(`  WASM:    ${colors.cyan}npm install @ruvector/rvf-wasm${colors.reset}`);
          }
          log('');
        }
      })
  )
  .addCommand(
    new Command('witness')
      .description('Verify SHAKE-256 witness chain integrity')
      .argument('<store>', 'Path to .rvf store file')
      .option('--json', 'Output as JSON')
      .action(async (store: string, opts: { json?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);
          const witness = await backend.verifyWitnessAsync();

          if (opts.json) {
            console.log(JSON.stringify(witness, null, 2));
          } else {
            log(`\n${colors.bright}${colors.cyan}Witness Chain Verification${colors.reset}\n`);
            log(`  Store:     ${colors.blue}${store}${colors.reset}`);
            log(`  Valid:     ${witness.valid ? `${colors.green}Yes${colors.reset}` : `${colors.red}No${colors.reset}`}`);
            log(`  Entries:   ${colors.blue}${witness.entries}${colors.reset}`);
            if (witness.error) {
              log(`  Error:     ${colors.red}${witness.error}${colors.reset}`);
            }
            log('');
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('freeze')
      .description('Snapshot-freeze store state and return epoch')
      .argument('<store>', 'Path to .rvf store file')
      .option('--json', 'Output as JSON')
      .action(async (store: string, opts: { json?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);
          const epoch = await backend.freeze();

          if (opts.json) {
            console.log(JSON.stringify({ store, epoch }, null, 2));
          } else {
            log(`\n${colors.bright}${colors.cyan}State Frozen${colors.reset}\n`);
            log(`  Store:  ${colors.blue}${store}${colors.reset}`);
            log(`  Epoch:  ${colors.green}${epoch}${colors.reset}`);
            log('');
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('index-stats')
      .description('Show HNSW index statistics')
      .argument('<store>', 'Path to .rvf store file')
      .option('--json', 'Output as JSON')
      .action(async (store: string, opts: { json?: boolean }) => {
        try {
          const backend = await loadRvfBackend(store);
          const stats = backend.indexStats();
          const metric = backend.metric();

          if (opts.json) {
            console.log(JSON.stringify({ store, metric, ...stats }, null, 2));
          } else {
            log(`\n${colors.bright}${colors.cyan}HNSW Index Statistics${colors.reset}\n`);
            log(`  Store:            ${colors.blue}${store}${colors.reset}`);
            log(`  Metric:           ${colors.blue}${metric}${colors.reset}`);
            log(`  Indexed Vectors:  ${colors.green}${stats.indexedVectors}${colors.reset}`);
            log(`  Layers:           ${colors.blue}${stats.layers}${colors.reset}`);
            log(`  M:                ${colors.blue}${stats.m}${colors.reset}`);
            log(`  efConstruction:   ${colors.blue}${stats.efConstruction}${colors.reset}`);
            log(`  Needs Rebuild:    ${stats.needsRebuild ? `${colors.yellow}Yes${colors.reset}` : `${colors.green}No${colors.reset}`}`);
            log('');
          }

          backend.close();
        } catch (error) {
          console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
          process.exit(1);
        }
      })
  )
  .addCommand(
    new Command('solver')
      .description('Self-learning temporal solver')
      .addCommand(
        new Command('train')
          .description('Train the solver on generated puzzles')
          .option('-n, --count <n>', 'Number of puzzles', '100')
          .option('--min-difficulty <n>', 'Minimum difficulty (1-10)', '1')
          .option('--max-difficulty <n>', 'Maximum difficulty (1-10)', '10')
          .option('--seed <n>', 'RNG seed for reproducibility')
          .option('--json', 'Output as JSON')
          .action(async (opts: {
            count: string;
            minDifficulty: string;
            maxDifficulty: string;
            seed?: string;
            json?: boolean;
          }) => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let solver: any = null;
            try {
              const { AgentDBSolver } = await import('../../backends/rvf/RvfSolver.js');
              solver = await AgentDBSolver.create();

              const result = solver.train({
                count: safeParseInt(opts.count, 'count'),
                minDifficulty: safeParseInt(opts.minDifficulty, 'min-difficulty'),
                maxDifficulty: safeParseInt(opts.maxDifficulty, 'max-difficulty'),
                seed: opts.seed ? safeParseInt(opts.seed, 'seed') : undefined,
              }) as { trained: number; correct: number; accuracy: number; patternsLearned: number };

              if (opts.json) {
                console.log(JSON.stringify(result, null, 2));
              } else {
                log(`\n${colors.bright}${colors.cyan}Solver Training Results${colors.reset}\n`);
                log(`  Trained:          ${colors.green}${result.trained}${colors.reset} puzzles`);
                log(`  Correct:          ${colors.green}${result.correct}${colors.reset}`);
                log(`  Accuracy:         ${colors.green}${(result.accuracy * 100).toFixed(1)}%${colors.reset}`);
                log(`  Patterns Learned: ${colors.blue}${result.patternsLearned}${colors.reset}`);
                log('');
              }
            } catch (error) {
              console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
              process.exit(1);
            } finally {
              solver?.destroy();
            }
          })
      )
      .addCommand(
        new Command('test')
          .description('Run A/B/C acceptance test')
          .option('--cycles <n>', 'Number of train/test cycles', '5')
          .option('--holdout <n>', 'Holdout puzzles per cycle', '50')
          .option('--training <n>', 'Training puzzles per cycle', '200')
          .option('--seed <n>', 'RNG seed for reproducibility')
          .option('--json', 'Output as JSON')
          .action(async (opts: {
            cycles: string;
            holdout: string;
            training: string;
            seed?: string;
            json?: boolean;
          }) => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let solver: any = null;
            try {
              const { AgentDBSolver } = await import('../../backends/rvf/RvfSolver.js');
              solver = await AgentDBSolver.create();

              const manifest = solver.acceptance({
                cycles: safeParseInt(opts.cycles, 'cycles'),
                holdoutSize: safeParseInt(opts.holdout, 'holdout'),
                trainingPerCycle: safeParseInt(opts.training, 'training'),
                seed: opts.seed ? safeParseInt(opts.seed, 'seed') : undefined,
              }) as {
                modeA: { passed: boolean; finalAccuracy: number; cycles: { cycle: number; accuracy: number; costPerSolve: number }[] };
                modeB: { passed: boolean; finalAccuracy: number; cycles: { cycle: number; accuracy: number; costPerSolve: number }[] };
                modeC: { passed: boolean; finalAccuracy: number; cycles: { cycle: number; accuracy: number; costPerSolve: number }[] };
                allPassed: boolean;
                witnessEntries: number;
                witnessChainBytes: number;
              };

              if (opts.json) {
                console.log(JSON.stringify(manifest, null, 2));
              } else {
                log(`\n${colors.bright}${colors.cyan}A/B/C Acceptance Test${colors.reset}\n`);

                const showMode = (name: string, mode: typeof manifest.modeA) => {
                  const status = mode.passed ? `${colors.green}PASS${colors.reset}` : `${colors.red}FAIL${colors.reset}`;
                  log(`  Mode ${name}: ${status}  (accuracy: ${(mode.finalAccuracy * 100).toFixed(1)}%)`);
                  for (const c of mode.cycles) {
                    log(`    Cycle ${c.cycle}: ${(c.accuracy * 100).toFixed(1)}% accuracy, cost=${c.costPerSolve.toFixed(1)}`);
                  }
                };

                showMode('A (Heuristic)', manifest.modeA);
                showMode('B (Compiler)', manifest.modeB);
                showMode('C (Learned)', manifest.modeC);

                log('');
                log(`  ${colors.bright}Overall:${colors.reset}  ${manifest.allPassed ? `${colors.green}ALL PASSED${colors.reset}` : `${colors.red}SOME FAILED${colors.reset}`}`);
                log(`  Witness:  ${manifest.witnessEntries} entries (${manifest.witnessChainBytes} bytes)`);
                log('');
              }
            } catch (error) {
              console.error(`${colors.red}Error: ${(error as Error).message}${colors.reset}`);
              process.exit(1);
            } finally {
              solver?.destroy();
            }
          })
      )
  );
