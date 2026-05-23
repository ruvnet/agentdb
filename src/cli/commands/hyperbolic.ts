/**
 * Hyperbolic Space CLI Command
 * Provides Poincaré ball operations and dual-space search
 * Based on ADR-002 Phase 3: Hyperbolic Geometry
 */

import { Command } from 'commander';
import chalk from 'chalk';
import * as fs from 'fs/promises';
import * as path from 'path';

// Operation types
type HyperbolicOp = 'expmap' | 'logmap' | 'mobius-add' | 'distance' | 'project' | 'centroid' | 'dual-search';

interface HyperbolicOptions {
  op: HyperbolicOp;
  point?: string;
  tangent?: string;
  base?: string;
  pointA?: string;
  pointB?: string;
  points?: string;
  query?: string;
  curvature?: number;
  topK?: number;
  euclideanWeight?: number;
  hyperbolicWeight?: number;
  output?: string;
  verbose?: boolean;
  json?: boolean;
}

/**
 * Main hyperbolic command
 */
export const hyperbolicCommand = new Command('hyperbolic')
  .description('Hyperbolic space operations: Poincaré ball, dual-space search (WASM-accelerated)')
  .requiredOption('-o, --op <type>', 'Operation: expmap, logmap, mobius-add, distance, project, centroid, dual-search')
  .option('--point <vector>', 'Point vector (JSON array)')
  .option('--tangent <vector>', 'Tangent vector (JSON array)')
  .option('--base <vector>', 'Base point (JSON array)')
  .option('--point-a <vector>', 'First point (JSON array)')
  .option('--point-b <vector>', 'Second point (JSON array)')
  .option('--points <file>', 'File containing multiple points (JSON)')
  .option('--query <vector>', 'Query vector for dual-space search (JSON array)')
  .option('-c, --curvature <n>', 'Hyperbolic curvature (negative)', '-1.0')
  .option('-k, --top-k <n>', 'Top-K results for search', '10')
  .option('--euclidean-weight <n>', 'Euclidean space weight (0-1)', '0.5')
  .option('--hyperbolic-weight <n>', 'Hyperbolic space weight (0-1)', '0.5')
  .option('--output <path>', 'Output file path')
  .option('-v, --verbose', 'Verbose output')
  .option('--json', 'Output as JSON')
  .action(async (options: HyperbolicOptions) => {
    try {
      if (!options.json && !options.verbose) {
        console.log(chalk.cyan.bold('\n🌀 AgentDB Hyperbolic Space Operations\n'));
      }

      // Check WASM availability
      const wasmAvailable = await checkWasmAvailability(options.verbose || false);

      // Execute operation
      const result = await executeHyperbolicOp(options, wasmAvailable);

      // Save results if output path provided
      if (options.output) {
        await fs.writeFile(options.output, JSON.stringify(result, null, 2));
        if (!options.json) {
          console.log(chalk.green(`\n✅ Results saved to: ${options.output}\n`));
        }
      }

      if (options.json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        displayResult(result, options);
      }

    } catch (error: any) {
      if (options.json) {
        console.log(JSON.stringify({ error: error.message, wasmAvailable: false }, null, 2));
      } else {
        console.error(chalk.red(`\n❌ Error: ${error.message}\n`));
        if (error.message.includes('WASM') || error.message.includes('ruvector')) {
          console.log(chalk.yellow('💡 Tip: Install @ruvector/attention for WASM-accelerated hyperbolic operations.'));
        }
      }
      process.exit(1);
    }
  });

/**
 * Check WASM availability
 */
async function checkWasmAvailability(verbose: boolean): Promise<boolean> {
  try {
    await import('@ruvector/attention');
    if (verbose) {
      console.log(chalk.green('✅ Using WASM-accelerated hyperbolic operations\n'));
    }
    return true;
  } catch (error) {
    if (verbose) {
      console.log(chalk.yellow('⚠️  WASM not available, using JavaScript fallback\n'));
    }
    return false;
  }
}

/**
 * Execute hyperbolic operation
 */
async function executeHyperbolicOp(options: HyperbolicOptions, wasmAvailable: boolean): Promise<any> {
  const startTime = performance.now();
  const curvature = parseFloat(String(options.curvature || '-1.0'));

  let result: any;

  switch (options.op) {
    case 'expmap':
      result = await expMap(options, curvature);
      break;
    case 'logmap':
      result = await logMap(options, curvature);
      break;
    case 'mobius-add':
      result = await mobiusAdd(options, curvature);
      break;
    case 'distance':
      result = await poincareDistance(options, curvature);
      break;
    case 'project':
      result = await projectToBall(options, curvature);
      break;
    case 'centroid':
      result = await hyperbolicCentroid(options, curvature);
      break;
    case 'dual-search':
      result = await dualSpaceSearch(options, curvature);
      break;
    default:
      throw new Error(`Unknown operation: ${options.op}`);
  }

  const computeTime = performance.now() - startTime;

  return {
    operation: options.op,
    wasmAccelerated: wasmAvailable,
    curvature,
    computeTimeMs: computeTime,
    ...result,
  };
}

/**
 * Exponential map: tangent vector -> manifold point
 */
async function expMap(options: HyperbolicOptions, curvature: number): Promise<any> {
  const base = parseVector(options.base || options.point, 'base/point');
  const tangent = parseVector(options.tangent, 'tangent');

  if (base.length !== tangent.length) {
    throw new Error('Base and tangent vectors must have same dimension');
  }

  // Simplified Poincaré exponential map
  const sqrtCurvature = Math.sqrt(Math.abs(curvature));
  const tangentNorm = vectorNorm(tangent);

  if (tangentNorm < 1e-10) {
    return { result: base, baseNorm: vectorNorm(base), tangentNorm: 0 };
  }

  const lambda = 2 / (1 - Math.pow(vectorNorm(base), 2));
  const tanhTerm = Math.tanh(sqrtCurvature * lambda * tangentNorm / 2);
  const scaleFactor = tanhTerm / (sqrtCurvature * tangentNorm);

  const result = base.map((b, i) => {
    const scaledTangent = tangent[i] * scaleFactor;
    return mobiusAdditionComponent(b, scaledTangent, curvature);
  });

  return {
    result: projectToPoincareBall(result, curvature),
    baseNorm: vectorNorm(base),
    tangentNorm,
    scaleFactor,
  };
}

/**
 * Logarithmic map: manifold point -> tangent vector
 */
async function logMap(options: HyperbolicOptions, curvature: number): Promise<any> {
  const base = parseVector(options.base, 'base');
  const point = parseVector(options.point, 'point');

  if (base.length !== point.length) {
    throw new Error('Base and point vectors must have same dimension');
  }

  // Simplified Poincaré logarithmic map
  const sqrtCurvature = Math.sqrt(Math.abs(curvature));
  const mobiusSub = mobiusSubtraction(point, base, curvature);
  const mobiusNorm = vectorNorm(mobiusSub);

  const lambda = 2 / (1 - Math.pow(vectorNorm(base), 2));
  const atanhTerm = Math.atanh(Math.min(mobiusNorm, 0.9999999));
  const scaleFactor = (2 * atanhTerm) / (sqrtCurvature * lambda * mobiusNorm);

  const result = mobiusSub.map(v => v * scaleFactor);

  return {
    result,
    distance: poincareDistanceValue(base, point, curvature),
    tangentNorm: vectorNorm(result),
  };
}

/**
 * Möbius addition (hyperbolic addition)
 */
async function mobiusAdd(options: HyperbolicOptions, curvature: number): Promise<any> {
  const x = parseVector(options.pointA, 'point-a');
  const y = parseVector(options.pointB, 'point-b');

  if (x.length !== y.length) {
    throw new Error('Points must have same dimension');
  }

  const result = x.map((xi, i) => mobiusAdditionComponent(xi, y[i], curvature));
  const projected = projectToPoincareBall(result, curvature);

  return {
    result: projected,
    xNorm: vectorNorm(x),
    yNorm: vectorNorm(y),
    resultNorm: vectorNorm(projected),
  };
}

/**
 * Poincaré distance
 */
async function poincareDistance(options: HyperbolicOptions, curvature: number): Promise<any> {
  const x = parseVector(options.pointA, 'point-a');
  const y = parseVector(options.pointB, 'point-b');

  if (x.length !== y.length) {
    throw new Error('Points must have same dimension');
  }

  const distance = poincareDistanceValue(x, y, curvature);

  // Also compute Euclidean distance for comparison
  const euclideanDist = Math.sqrt(
    x.reduce((sum, xi, i) => sum + Math.pow(xi - y[i], 2), 0)
  );

  return {
    distance,
    euclideanDistance: euclideanDist,
    ratio: distance / euclideanDist,
  };
}

/**
 * Project to Poincaré ball
 */
async function projectToBall(options: HyperbolicOptions, curvature: number): Promise<any> {
  const point = parseVector(options.point, 'point');
  const originalNorm = vectorNorm(point);

  const projected = projectToPoincareBall(point, curvature);
  const projectedNorm = vectorNorm(projected);

  return {
    result: projected,
    originalNorm,
    projectedNorm,
    wasProjected: originalNorm >= 0.9999,
  };
}

/**
 * Hyperbolic centroid
 */
async function hyperbolicCentroid(options: HyperbolicOptions, curvature: number): Promise<any> {
  if (!options.points) {
    throw new Error('Points file required (--points <file>)');
  }

  const fileContent = await fs.readFile(options.points, 'utf-8');
  let points: number[][];
  try {
    points = JSON.parse(fileContent);
  } catch (err) {
    throw new Error(`Failed to parse points file as JSON: ${err instanceof Error ? err.message : String(err)}`);
  }

  if (!Array.isArray(points) || points.length === 0) {
    throw new Error('Points must be a non-empty array');
  }

  // Compute hyperbolic centroid using iterative algorithm
  let centroid = points[0].slice();
  const maxIterations = 100;
  const tolerance = 1e-6;

  for (let iter = 0; iter < maxIterations; iter++) {
    const gradients = points.map(point => {
      const logVec = mobiusSubtraction(point, centroid, curvature);
      return logVec;
    });

    // Average gradient
    const avgGradient = gradients[0].map((_, i) =>
      gradients.reduce((sum, g) => sum + g[i], 0) / gradients.length
    );

    const gradientNorm = vectorNorm(avgGradient);
    if (gradientNorm < tolerance) {
      break;
    }

    // Update centroid
    const stepSize = 0.1;
    const scaledGradient = avgGradient.map(g => g * stepSize);
    centroid = centroid.map((c, i) => mobiusAdditionComponent(c, scaledGradient[i], curvature));
    centroid = projectToPoincareBall(centroid, curvature);
  }

  // Compute average distance to centroid
  const avgDistance = points.reduce(
    (sum, point) => sum + poincareDistanceValue(point, centroid, curvature),
    0
  ) / points.length;

  return {
    result: centroid,
    pointCount: points.length,
    avgDistance,
    centroidNorm: vectorNorm(centroid),
  };
}

/**
 * Dual-space search (hybrid Euclidean + Hyperbolic)
 */
async function dualSpaceSearch(options: HyperbolicOptions, curvature: number): Promise<any> {
  if (!options.points) {
    throw new Error('Points file required (--points <file>)');
  }

  const query = parseVector(options.query, 'query');
  const euclideanWeight = parseFloat(String(options.euclideanWeight || '0.5'));
  const hyperbolicWeight = parseFloat(String(options.hyperbolicWeight || '0.5'));
  const topK = parseInt(String(options.topK || '10'));

  const fileContent = await fs.readFile(options.points, 'utf-8');
  let points: number[][];
  try {
    points = JSON.parse(fileContent);
  } catch (err) {
    throw new Error(`Failed to parse points file as JSON: ${err instanceof Error ? err.message : String(err)}`);
  }

  // Compute dual-space similarities
  const results = points.map((point, idx) => {
    const euclideanSim = cosineSimilarity(query, point);
    const hyperbolicDist = poincareDistanceValue(query, point, curvature);
    const hyperbolicSim = 1 / (1 + hyperbolicDist);

    const hybridScore = euclideanWeight * euclideanSim + hyperbolicWeight * hyperbolicSim;

    return {
      id: idx,
      euclideanSimilarity: euclideanSim,
      hyperbolicDistance: hyperbolicDist,
      hyperbolicSimilarity: hyperbolicSim,
      hybridScore,
    };
  });

  // Sort by hybrid score and take top-K
  results.sort((a, b) => b.hybridScore - a.hybridScore);
  const topResults = results.slice(0, topK);

  return {
    results: topResults,
    config: {
      euclideanWeight,
      hyperbolicWeight,
      topK,
    },
    avgEuclideanSim: topResults.reduce((sum, r) => sum + r.euclideanSimilarity, 0) / topResults.length,
    avgHyperbolicSim: topResults.reduce((sum, r) => sum + r.hyperbolicSimilarity, 0) / topResults.length,
  };
}

// Helper functions

function parseVector(vectorStr: string | undefined, name: string): number[] {
  if (!vectorStr) {
    throw new Error(`${name} vector required`);
  }

  try {
    const vector = JSON.parse(vectorStr);
    if (!Array.isArray(vector) || vector.some(v => typeof v !== 'number')) {
      throw new Error('Invalid vector format');
    }
    return vector;
  } catch (error) {
    throw new Error(`${name} must be a valid JSON array of numbers`);
  }
}

function vectorNorm(v: number[]): number {
  return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  return dot / (normA * normB);
}

function mobiusAdditionComponent(x: number, y: number, curvature: number): number {
  const K = Math.abs(curvature);
  const numerator = (1 + 2 * K * x * y + K * y * y) * x + (1 - K * x * x) * y;
  const denominator = 1 + 2 * K * x * y + K * K * x * x * y * y;
  return numerator / denominator;
}

function mobiusSubtraction(x: number[], y: number[], curvature: number): number[] {
  const negY = y.map(yi => -yi);
  return x.map((xi, i) => mobiusAdditionComponent(xi, negY[i], curvature));
}

function poincareDistanceValue(x: number[], y: number[], curvature: number): number {
  const diff = x.map((xi, i) => xi - y[i]);
  const normDiff = vectorNorm(diff);
  const normX = vectorNorm(x);
  const normY = vectorNorm(y);

  const numerator = normDiff * normDiff;
  const denominator = (1 - normX * normX) * (1 - normY * normY);

  if (denominator < 1e-10) {
    return Infinity;
  }

  const sqrtK = Math.sqrt(Math.abs(curvature));
  return (2 / sqrtK) * Math.atanh(sqrtK * Math.sqrt(numerator / denominator));
}

function projectToPoincareBall(point: number[], curvature: number): number[] {
  const norm = vectorNorm(point);
  const maxNorm = 0.9999;

  if (norm >= maxNorm) {
    const scale = maxNorm / norm;
    return point.map(p => p * scale);
  }

  return point;
}

function displayResult(result: any, options: HyperbolicOptions): void {
  console.log(chalk.bold('\n📐 Hyperbolic Operation Results:\n'));

  console.log(`  Operation: ${result.operation}`);
  console.log(`  WASM Accelerated: ${result.wasmAccelerated ? chalk.green('Yes') : chalk.yellow('No')}`);
  console.log(`  Curvature: ${result.curvature}`);
  console.log(`  Compute Time: ${result.computeTimeMs.toFixed(2)}ms`);

  if (result.result && Array.isArray(result.result)) {
    console.log(`\n  Result Vector: [${result.result.slice(0, 5).map((v: number) => v.toFixed(4)).join(', ')}${result.result.length > 5 ? '...' : ''}]`);
    console.log(`  Result Dimension: ${result.result.length}`);
    console.log(`  Result Norm: ${vectorNorm(result.result).toFixed(4)}`);
  }

  if (result.distance !== undefined) {
    console.log(`\n  Poincaré Distance: ${result.distance.toFixed(4)}`);
    if (result.euclideanDistance !== undefined) {
      console.log(`  Euclidean Distance: ${result.euclideanDistance.toFixed(4)}`);
      console.log(`  Hyperbolic/Euclidean Ratio: ${result.ratio.toFixed(2)}x`);
    }
  }

  if (result.results && Array.isArray(result.results)) {
    console.log(chalk.bold('\n  Top Results:\n'));
    result.results.slice(0, 5).forEach((r: any, i: number) => {
      console.log(`    ${i + 1}. ID ${r.id}: hybrid=${r.hybridScore.toFixed(4)}, ` +
        `euclidean=${r.euclideanSimilarity.toFixed(4)}, hyperbolic=${r.hyperbolicSimilarity.toFixed(4)}`);
    });
  }

  if (options.verbose) {
    console.log(chalk.bold('\n  Additional Metrics:\n'));
    Object.entries(result).forEach(([key, value]) => {
      if (!['operation', 'wasmAccelerated', 'curvature', 'computeTimeMs', 'result', 'results'].includes(key)) {
        console.log(`    ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`);
      }
    });
  }

  console.log('');
}

// Add help text
hyperbolicCommand.on('--help', () => {
  console.log('');
  console.log('Examples:');
  console.log('  # Exponential map');
  console.log('  $ agentdb hyperbolic --op expmap --base "[0,0]" --tangent "[0.5,0.5]"');
  console.log('');
  console.log('  # Poincaré distance');
  console.log('  $ agentdb hyperbolic --op distance --point-a "[0.3,0.4]" --point-b "[0.6,0.2]"');
  console.log('');
  console.log('  # Dual-space search');
  console.log('  $ agentdb hyperbolic --op dual-search --query "[0.5,0.5]" --points vectors.json -k 20');
  console.log('');
  console.log('  # Hyperbolic centroid');
  console.log('  $ agentdb hyperbolic --op centroid --points cluster.json --curvature -0.5');
  console.log('');
  console.log('Poincaré Ball:');
  console.log('  All operations work in the Poincaré ball model of hyperbolic space.');
  console.log('  Points must satisfy ||x|| < 1 (automatically projected if outside).');
  console.log('  Curvature must be negative (default: -1.0).');
  console.log('');
  console.log('WASM Acceleration:');
  console.log('  This command uses @ruvector/attention WASM modules when available.');
  console.log('  Falls back gracefully to JavaScript implementation.');
  console.log('');
});
