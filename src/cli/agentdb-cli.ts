#!/usr/bin/env node
/**
 * AgentDB CLI - Command-line interface for frontier memory features
 *
 * Provides commands for:
 * - Causal memory graph operations
 * - Explainable recall with certificates
 * - Nightly learner automation
 * - Database management
 */

import { createDatabase } from '../db-fallback.js';
import { parseJsonStrict } from '../security/input-validation.js';
import { CausalMemoryGraph } from '../controllers/CausalMemoryGraph.js';
import { CausalRecall } from '../controllers/CausalRecall.js';
import { ExplainableRecall } from '../controllers/ExplainableRecall.js';
import { NightlyLearner } from '../controllers/NightlyLearner.js';
import { ReflexionMemory, Episode, ReflexionQuery } from '../controllers/ReflexionMemory.js';
import { SkillLibrary, Skill, SkillQuery } from '../controllers/SkillLibrary.js';
import { EmbeddingService } from '../controllers/EmbeddingService.js';
import { MMRDiversityRanker } from '../controllers/MMRDiversityRanker.js';
import { ContextSynthesizer } from '../controllers/ContextSynthesizer.js';
import { MetadataFilter } from '../controllers/MetadataFilter.js';
import { QUICServer, QUICServerConfig } from '../controllers/QUICServer.js';
import { QUICClient, QUICClientConfig, SyncProgress as ClientSyncProgress } from '../controllers/QUICClient.js';
import { SyncCoordinator, SyncCoordinatorConfig, SyncProgress, SyncReport } from '../controllers/SyncCoordinator.js';
import { initCommand } from './commands/init.js';
import { statusCommand } from './commands/status.js';
import { installEmbeddingsCommand } from './commands/install-embeddings.js';
import { migrateCommand } from './commands/migrate.js';
import { doctorCommand } from './commands/doctor.js';
import { attentionCommand } from './commands/attention.js';
import { learnCommand } from './commands/learn.js';
import { routeCommand } from './commands/route.js';
import { hyperbolicCommand } from './commands/hyperbolic.js';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m'
};

const log = {
  success: (msg: string) => console.log(`${colors.green}✅ ${msg}${colors.reset}`),
  error: (msg: string) => console.error(`${colors.red}❌ ${msg}${colors.reset}`),
  info: (msg: string) => console.log(`${colors.blue}ℹ ${msg}${colors.reset}`),
  warning: (msg: string) => console.log(`${colors.yellow}⚠ ${msg}${colors.reset}`),
  header: (msg: string) => console.log(`${colors.bright}${colors.cyan}${msg}${colors.reset}`)
};

// Spinner utility for progress indication
class Spinner {
  private frames = ['|', '/', '-', '\\'];
  private current = 0;
  private interval: NodeJS.Timeout | null = null;
  private message: string = '';

  start(message: string): void {
    this.message = message;
    this.current = 0;
    process.stdout.write(`\r${this.frames[0]} ${message}`);
    this.interval = setInterval(() => {
      this.current = (this.current + 1) % this.frames.length;
      process.stdout.write(`\r${this.frames[this.current]} ${this.message}`);
    }, 100);
  }

  update(message: string): void {
    this.message = message;
    process.stdout.write(`\r${this.frames[this.current]} ${this.message}`);
  }

  succeed(message: string): void {
    this.stop();
    console.log(`\r${colors.green}v${colors.reset} ${message}`);
  }

  fail(message: string): void {
    this.stop();
    console.log(`\r${colors.red}x${colors.reset} ${message}`);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    process.stdout.write('\r' + ' '.repeat(this.message.length + 10) + '\r');
  }
}

// Progress bar utility
class ProgressBar {
  private width = 40;
  private current = 0;
  private total = 100;
  private message = '';

  update(current: number, total: number, message?: string): void {
    this.current = current;
    this.total = total;
    if (message) this.message = message;

    const percent = Math.round((current / total) * 100);
    const filled = Math.round((current / total) * this.width);
    const empty = this.width - filled;
    const bar = `${colors.green}${'='.repeat(filled)}${colors.reset}${'-'.repeat(empty)}`;

    process.stdout.write(`\r[${bar}] ${percent}% ${this.message}`);
  }

  complete(message: string): void {
    const bar = `${colors.green}${'='.repeat(this.width)}${colors.reset}`;
    console.log(`\r[${bar}] 100% ${message}`);
  }
}

class AgentDBCLI {
  public db?: any; // Database instance from createDatabase (public for init command)
  private causalGraph?: CausalMemoryGraph;
  private causalRecall?: CausalRecall;
  private explainableRecall?: ExplainableRecall;
  private nightlyLearner?: NightlyLearner;
  private reflexion?: ReflexionMemory;
  private skills?: SkillLibrary;
  private embedder?: EmbeddingService;

  // QUIC sync controllers
  private quicServer?: QUICServer;
  private quicClient?: QUICClient;
  private syncCoordinator?: SyncCoordinator;

  async initialize(dbPath: string = './agentdb.db'): Promise<void> {
    // Initialize database
    this.db = await createDatabase(dbPath);

    // Configure for performance
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('cache_size = -64000');

    // Load both schemas from the inlined constants — bundled into the package,
    // so globally-installed CLIs and weird __dirname resolutions just work.
    // (Was a long __dirname/cwd dance that failed for `npm install -g`. #1)
    const { SCHEMA_SQL, FRONTIER_SCHEMA_SQL } = await import('../schemas/inline.js');
    if (SCHEMA_SQL) this.db.exec(SCHEMA_SQL);
    if (FRONTIER_SCHEMA_SQL) this.db.exec(FRONTIER_SCHEMA_SQL);

    // Initialize embedding service
    this.embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await this.embedder.initialize();

    // Initialize controllers
    this.causalGraph = new CausalMemoryGraph(this.db);
    this.explainableRecall = new ExplainableRecall(this.db);
    this.causalRecall = new CausalRecall(this.db, this.embedder, undefined, {
      alpha: 0.7,
      beta: 0.2,
      gamma: 0.1,
      minConfidence: 0.6
    });
    this.nightlyLearner = new NightlyLearner(this.db, this.embedder);

    // ReflexionMemory and SkillLibrary support optional GNN/Graph backends
    // These will be undefined if @ruvector/gnn or @ruvector/graph-node are not installed
    this.reflexion = new ReflexionMemory(
      this.db,
      this.embedder,
      undefined,  // vectorBackend - would be created with detectBackends()
      undefined,  // learningBackend - requires @ruvector/gnn
      undefined   // graphBackend - requires @ruvector/graph-node
    );
    this.skills = new SkillLibrary(this.db, this.embedder);
  }

  // ============================================================================
  // Causal Commands
  // ============================================================================

  async causalAddEdge(params: {
    cause: string;
    effect: string;
    uplift: number;
    confidence?: number;
    sampleSize?: number;
  }): Promise<void> {
    if (!this.causalGraph) throw new Error('Not initialized');

    log.header('\n📊 Adding Causal Edge');
    log.info(`Cause: ${params.cause}`);
    log.info(`Effect: ${params.effect}`);
    log.info(`Uplift: ${params.uplift}`);

    const edgeId = this.causalGraph.addCausalEdge({
      fromMemoryId: 1,
      fromMemoryType: 'episode',
      toMemoryId: 2,
      toMemoryType: 'episode',
      similarity: 0.9,
      uplift: params.uplift,
      confidence: params.confidence || 0.95,
      sampleSize: params.sampleSize || 0,
      mechanism: `${params.cause} → ${params.effect}`,
      evidenceIds: []
    });

    log.success(`Added causal edge #${edgeId}`);
  }

  async causalExperimentCreate(params: {
    name: string;
    cause: string;
    effect: string;
  }): Promise<void> {
    if (!this.causalGraph) throw new Error('Not initialized');

    log.header('\n🧪 Creating A/B Experiment');
    log.info(`Name: ${params.name}`);
    log.info(`Cause: ${params.cause}`);
    log.info(`Effect: ${params.effect}`);

    // Create a dummy episode for treatment reference
    const dummyEpisode = this.db!.prepare(
      'INSERT INTO episodes (session_id, task, reward, success, created_at) VALUES (?, ?, ?, ?, ?)'
    ).run('experiment-placeholder', params.name, 0.0, 0, Math.floor(Date.now() / 1000));

    const treatmentId = Number(dummyEpisode.lastInsertRowid);

    const expId = this.causalGraph.createExperiment({
      name: params.name,
      hypothesis: `Does ${params.cause} causally affect ${params.effect}?`,
      treatmentId: treatmentId,
      treatmentType: params.cause,
      controlId: undefined,
      startTime: Math.floor(Date.now() / 1000),
      sampleSize: 0,
      status: 'running',
      metadata: { effect: params.effect }
    });

    log.success(`Created experiment #${expId}`);
    log.info('Use `agentdb causal experiment add-observation` to record data');

    // Save database to persist experiment
    this.db.save();
  }

  async causalExperimentAddObservation(params: {
    experimentId: number;
    isTreatment: boolean;
    outcome: number;
    context?: string;
  }): Promise<void> {
    if (!this.causalGraph) throw new Error('Not initialized');

    // Create a dummy episode to get an episode ID
    const insertResult = this.db!.prepare('INSERT INTO episodes (session_id, task, reward, success, created_at) VALUES (?, ?, ?, ?, ?)').run('cli-session', 'experiment', params.outcome, 1, Math.floor(Date.now() / 1000));
    if (!insertResult || !insertResult.lastInsertRowid) {
      throw new Error('Failed to create episode');
    }
    const episodeId = Number(insertResult.lastInsertRowid);

    this.causalGraph.recordObservation({
      experimentId: params.experimentId,
      episodeId: episodeId,
      isTreatment: params.isTreatment,
      outcomeValue: params.outcome,
      outcomeType: 'reward',
      // #6 §C.1 — strict parse so a crafted/malformed `--context` JSON
      // surfaces as a ValidationError, not an unhandled SyntaxError leaking
      // a stack trace.
      context: params.context && params.context.trim()
        ? parseJsonStrict<Record<string, unknown>>(params.context, 'context')
        : undefined
    });

    // Save database to persist changes
    this.db.save();

    log.success(`Recorded ${params.isTreatment ? 'treatment' : 'control'} observation: ${params.outcome}`);
  }

  async causalExperimentCalculate(experimentId: number): Promise<void> {
    if (!this.causalGraph) throw new Error('Not initialized');

    log.header('\n📈 Calculating Uplift');

    const result = this.causalGraph.calculateUplift(experimentId);

    // Fetch experiment details (now includes calculated means)
    const experiment = this.db!.prepare('SELECT * FROM causal_experiments WHERE id = ?').get(experimentId) as any;
    if (!experiment) {
      throw new Error(`Experiment ${experimentId} not found`);
    }

    log.info(`Experiment: ${experiment.hypothesis || 'Unknown'}`);
    log.info(`Treatment Mean: ${experiment.treatment_mean?.toFixed(3) || 'N/A'}`);
    log.info(`Control Mean: ${experiment.control_mean?.toFixed(3) || 'N/A'}`);
    log.success(`Uplift: ${result?.uplift?.toFixed(3) || 'N/A'}`);
    if (result?.confidenceInterval && result.confidenceInterval.length === 2) {
      log.info(`95% CI: [${result.confidenceInterval[0]?.toFixed(3) || 'N/A'}, ${result.confidenceInterval[1]?.toFixed(3) || 'N/A'}]`);
    }
    if (result?.pValue !== undefined) {
      log.info(`p-value: ${result.pValue.toFixed(4)}`);
    }

    // Get sample sizes from observations
    const counts = this.db!.prepare('SELECT COUNT(*) as total, SUM(is_treatment) as treatment FROM causal_observations WHERE experiment_id = ?').get(experimentId) as any;
    if (!counts) {
      throw new Error(`Failed to get observation counts for experiment ${experimentId}`);
    }
    log.info(`Sample Sizes: ${counts.treatment || 0} treatment, ${(counts.total || 0) - (counts.treatment || 0)} control`);

    if (result && result.pValue !== undefined && result.pValue < 0.05) {
      log.success('Result is statistically significant (p < 0.05)');
    } else {
      log.warning('Result is not statistically significant');
    }
  }

  async causalQuery(params: {
    cause?: string;
    effect?: string;
    minConfidence?: number;
    minUplift?: number;
    limit?: number;
  }): Promise<void> {
    if (!this.causalGraph) throw new Error('Not initialized');

    log.header('\n🔍 Querying Causal Edges');

    const edges = this.causalGraph.queryCausalEffects({
      interventionMemoryId: 0,
      interventionMemoryType: params.cause || '',
      outcomeMemoryId: params.effect ? 0 : undefined,
      minConfidence: params.minConfidence || 0.7,
      minUplift: params.minUplift || 0.1
    });

    if (edges.length === 0) {
      log.warning('No causal edges found');
      return;
    }

    console.log('\n' + '═'.repeat(80));
    edges.slice(0, params.limit || 10).forEach((edge, i) => {
      console.log(`${colors.bright}#${i + 1}: ${edge.fromMemoryType} → ${edge.toMemoryType}${colors.reset}`);
      console.log(`  Uplift: ${colors.green}${(edge.uplift || 0).toFixed(3)}${colors.reset}`);
      console.log(`  Confidence: ${edge.confidence.toFixed(2)} (n=${edge.sampleSize})`);
      console.log('─'.repeat(80));
    });

    log.success(`Found ${edges.length} causal edges`);
  }

  // ============================================================================
  // Recall Commands
  // ============================================================================

  async recallWithCertificate(params: {
    query: string;
    k?: number;
    alpha?: number;
    beta?: number;
    gamma?: number;
  }): Promise<void> {
    if (!this.causalRecall) throw new Error('Not initialized');

    log.header('\n🔍 Causal Recall with Certificate');
    log.info(`Query: "${params.query}"`);
    log.info(`k: ${params.k || 12}`);

    const startTime = Date.now();

    const result = await this.causalRecall.recall(
      'cli-' + Date.now(),
      params.query,
      params.k || 12,
      undefined,
      'internal'
    );

    const duration = Date.now() - startTime;

    console.log('\n' + '═'.repeat(80));
    console.log(`${colors.bright}Results (${result.candidates.length})${colors.reset}`);
    console.log('═'.repeat(80));

    result.candidates.slice(0, 5).forEach((r, i) => {
      console.log(`\n${colors.bright}#${i + 1}: ${r.type} ${r.id}${colors.reset}`);
      console.log(`  Content: ${r.content.substring(0, 50)}...`);
      console.log(`  Similarity: ${colors.cyan}${r.similarity.toFixed(3)}${colors.reset}`);
      console.log(`  Uplift: ${colors.green}${r.uplift?.toFixed(3) || 'N/A'}${colors.reset}`);
      console.log(`  Utility: ${colors.yellow}${r.utilityScore.toFixed(3)}${colors.reset}`);
    });

    console.log('\n' + '═'.repeat(80));
    log.info(`Certificate ID: ${result.certificate.id}`);
    log.info(`Query: ${result.certificate.queryText}`);
    log.info(`Completeness: ${result.certificate.completenessScore.toFixed(2)}`);
    log.success(`Completed in ${duration}ms`);
  }

  // ============================================================================
  // Learner Commands
  // ============================================================================

  async learnerRun(params: {
    minAttempts?: number;
    minSuccessRate?: number;
    minConfidence?: number;
    dryRun?: boolean;
  }): Promise<void> {
    if (!this.nightlyLearner) throw new Error('Not initialized');

    log.header('\n🌙 Running Nightly Learner');
    log.info(`Min Attempts: ${params.minAttempts || 3}`);
    log.info(`Min Success Rate: ${params.minSuccessRate || 0.6}`);
    log.info(`Min Confidence: ${params.minConfidence || 0.7}`);

    const startTime = Date.now();

    const discovered = await this.nightlyLearner.discover({
      minAttempts: params.minAttempts || 3,
      minSuccessRate: params.minSuccessRate || 0.6,
      minConfidence: params.minConfidence || 0.7,
      dryRun: params.dryRun || false
    });

    const duration = Date.now() - startTime;

    log.success(`Discovered ${discovered.length} causal edges in ${(duration / 1000).toFixed(1)}s`);

    if (discovered.length > 0) {
      console.log('\n' + '═'.repeat(80));
      discovered.slice(0, 10).forEach((edge: any, i: number) => {
        console.log(`${colors.bright}#${i + 1}: ${edge.cause} → ${edge.effect}${colors.reset}`);
        console.log(`  Uplift: ${colors.green}${edge.uplift.toFixed(3)}${colors.reset} (CI: ${edge.confidence.toFixed(2)})`);
        console.log(`  Sample size: ${edge.sampleSize}`);
        console.log('─'.repeat(80));
      });
    }
  }

  async learnerPrune(params: {
    minConfidence?: number;
    minUplift?: number;
    maxAgeDays?: number;
  }): Promise<void> {
    if (!this.nightlyLearner) throw new Error('Not initialized');

    log.header('\n🧹 Pruning Low-Quality Edges');

    // Update config and run pruning
    this.nightlyLearner.updateConfig({
      confidenceThreshold: params.minConfidence || 0.6,
      upliftThreshold: params.minUplift || 0.05,
      edgeMaxAgeDays: params.maxAgeDays || 90
    });

    const report = await this.nightlyLearner.run();

    log.success(`Pruned ${report.edgesPruned} edges`);
  }

  // ============================================================================
  // Reflexion Commands
  // ============================================================================

  async reflexionStoreEpisode(params: {
    sessionId: string;
    task: string;
    input?: string;
    output?: string;
    critique?: string;
    reward: number;
    success: boolean;
    latencyMs?: number;
    tokensUsed?: number;
  }): Promise<void> {
    if (!this.reflexion) throw new Error('Not initialized');

    log.header('\n💭 Storing Episode');
    log.info(`Task: ${params.task}`);
    log.info(`Success: ${params.success ? 'Yes' : 'No'}`);
    log.info(`Reward: ${params.reward.toFixed(2)}`);

    const episodeId = await this.reflexion.storeEpisode(params as Episode);

    // Save database to persist changes
    this.db.save();

    log.success(`Stored episode #${episodeId}`);
    if (params.critique) {
      log.info(`Critique: "${params.critique}"`);
    }
  }

  async reflexionRetrieve(params: {
    task: string;
    k?: number;
    onlyFailures?: boolean;
    onlySuccesses?: boolean;
    minReward?: number;
    synthesizeContext?: boolean;
    filters?: any;
  }): Promise<void> {
    if (!this.reflexion) throw new Error('Not initialized');

    log.header('\n🔍 Retrieving Past Episodes');
    log.info(`Task: "${params.task}"`);
    log.info(`k: ${params.k || 5}`);
    if (params.onlyFailures) log.info('Filter: Failures only');
    if (params.onlySuccesses) log.info('Filter: Successes only');
    if (params.synthesizeContext) log.info('Context synthesis: enabled');

    let episodes = await this.reflexion.retrieveRelevant({
      task: params.task,
      k: params.k || 5,
      onlyFailures: params.onlyFailures,
      onlySuccesses: params.onlySuccesses,
      minReward: params.minReward
    });

    // Apply metadata filters if provided
    if (params.filters && Object.keys(params.filters).length > 0) {
      episodes = MetadataFilter.apply(episodes, params.filters);
      log.info(`Filtered to ${episodes.length} results matching metadata criteria`);
    }

    if (episodes.length === 0) {
      log.warning('No episodes found');
      return;
    }

    console.log('\n' + '═'.repeat(80));
    episodes.forEach((ep, i) => {
      console.log(`${colors.bright}#${i + 1}: Episode ${ep.id}${colors.reset}`);
      console.log(`  Task: ${ep.task}`);
      console.log(`  Reward: ${colors.green}${ep.reward.toFixed(2)}${colors.reset}`);
      console.log(`  Success: ${ep.success ? colors.green + 'Yes' : colors.red + 'No'}${colors.reset}`);
      console.log(`  Similarity: ${colors.cyan}${ep.similarity?.toFixed(3) || 'N/A'}${colors.reset}`);
      if (ep.critique) {
        console.log(`  Critique: "${ep.critique}"`);
      }
      console.log('─'.repeat(80));
    });

    log.success(`Retrieved ${episodes.length} relevant episodes`);

    // Synthesize context if requested
    if (params.synthesizeContext && episodes.length > 0) {
      const context = ContextSynthesizer.synthesize(episodes.map(ep => ({
        task: ep.task,
        reward: ep.reward,
        success: ep.success,
        critique: ep.critique,
        input: ep.input,
        output: ep.output,
        similarity: ep.similarity
      })));

      console.log('\n' + '═'.repeat(80));
      console.log(`${colors.bright}${colors.cyan}SYNTHESIZED CONTEXT${colors.reset}`);
      console.log('═'.repeat(80));
      console.log(`\n${context.summary}\n`);

      if (context.patterns.length > 0) {
        console.log(`${colors.yellow}Common Patterns:${colors.reset}`);
        context.patterns.forEach(p => console.log(`  • ${p}`));
        console.log('');
      }

      if (context.keyInsights.length > 0) {
        console.log(`${colors.cyan}Key Insights:${colors.reset}`);
        context.keyInsights.forEach(i => console.log(`  • ${i}`));
        console.log('');
      }

      if (context.recommendations.length > 0) {
        console.log(`${colors.green}Recommendations:${colors.reset}`);
        context.recommendations.forEach((r, i) => console.log(`  ${i + 1}. ${r}`));
        console.log('');
      }

      console.log('═'.repeat(80));
    }
  }

  async reflexionRetrieveJson(params: {
    task: string;
    k?: number;
    onlyFailures?: boolean;
    onlySuccesses?: boolean;
    minReward?: number;
  }): Promise<any[]> {
    if (!this.reflexion) throw new Error('Not initialized');

    const episodes = await this.reflexion.retrieveRelevant({
      task: params.task,
      k: params.k || 5,
      onlyFailures: params.onlyFailures,
      onlySuccesses: params.onlySuccesses,
      minReward: params.minReward
    });

    return episodes;
  }

  async reflexionGetCritiqueSummary(params: {
    task: string;
    k?: number;
  }): Promise<void> {
    if (!this.reflexion) throw new Error('Not initialized');

    log.header('\n📋 Critique Summary');
    log.info(`Task: "${params.task}"`);

    const summary = await this.reflexion.getCritiqueSummary({
      task: params.task,
      k: params.k
    });

    console.log('\n' + '═'.repeat(80));
    console.log(colors.bright + 'Past Lessons:' + colors.reset);
    console.log(summary);
    console.log('═'.repeat(80));
  }

  async reflexionPrune(params: {
    minReward?: number;
    maxAgeDays?: number;
    keepMinPerTask?: number;
  }): Promise<void> {
    if (!this.reflexion) throw new Error('Not initialized');

    log.header('\n🧹 Pruning Episodes');

    const pruned = this.reflexion.pruneEpisodes({
      minReward: params.minReward || 0.3,
      maxAgeDays: params.maxAgeDays || 30,
      keepMinPerTask: params.keepMinPerTask || 5
    });

    log.success(`Pruned ${pruned} low-quality episodes`);
  }

  // ============================================================================
  // Skill Library Commands
  // ============================================================================

  async skillCreate(params: {
    name: string;
    description: string;
    code?: string;
    successRate?: number;
    episodeId?: number;
  }): Promise<void> {
    if (!this.skills) throw new Error('Not initialized');

    log.header('\n🎯 Creating Skill');
    log.info(`Name: ${params.name}`);
    log.info(`Description: ${params.description}`);

    const skillId = await this.skills.createSkill({
      name: params.name,
      description: params.description,
      signature: { inputs: {}, outputs: {} },
      code: params.code,
      successRate: params.successRate || 0.0,
      uses: 0,
      avgReward: 0.0,
      avgLatencyMs: 0.0,
      createdFromEpisode: params.episodeId
    });

    // Save database to persist changes
    this.db.save();

    log.success(`Created skill #${skillId}`);
  }

  async skillSearch(params: {
    task: string;
    k?: number;
    minSuccessRate?: number;
  }): Promise<void> {
    if (!this.skills) throw new Error('Not initialized');

    log.header('\n🔍 Searching Skills');
    log.info(`Task: "${params.task}"`);
    log.info(`Min Success Rate: ${params.minSuccessRate || 0.0}`);

    const skills = await this.skills.searchSkills({
      task: params.task,
      k: params.k || 10,
      minSuccessRate: params.minSuccessRate || 0.0
    });

    if (skills.length === 0) {
      log.warning('No skills found');
      return;
    }

    console.log('\n' + '═'.repeat(80));
    skills.forEach((skill: any, i: number) => {
      console.log(`${colors.bright}#${i + 1}: ${skill.name}${colors.reset}`);
      console.log(`  Description: ${skill.description}`);
      console.log(`  Success Rate: ${colors.green}${(skill.successRate * 100).toFixed(1)}%${colors.reset}`);
      console.log(`  Uses: ${skill.uses}`);
      console.log(`  Avg Reward: ${skill.avgReward.toFixed(2)}`);
      console.log(`  Avg Latency: ${skill.avgLatencyMs.toFixed(0)}ms`);
      console.log('─'.repeat(80));
    });

    log.success(`Found ${skills.length} matching skills`);
  }

  async skillConsolidate(params: {
    minAttempts?: number;
    minReward?: number;
    timeWindowDays?: number;
    extractPatterns?: boolean;
  }): Promise<void> {
    if (!this.skills) throw new Error('Not initialized');

    log.header('\n🔄 Consolidating Episodes into Skills with Pattern Extraction');
    log.info(`Min Attempts: ${params.minAttempts || 3}`);
    log.info(`Min Reward: ${params.minReward || 0.7}`);
    log.info(`Time Window: ${params.timeWindowDays || 7} days`);
    log.info(`Pattern Extraction: ${params.extractPatterns !== false ? 'Enabled' : 'Disabled'}`);

    const startTime = Date.now();

    const result = await this.skills.consolidateEpisodesIntoSkills({
      minAttempts: params.minAttempts || 3,
      minReward: params.minReward || 0.7,
      timeWindowDays: params.timeWindowDays || 7,
      extractPatterns: params.extractPatterns !== false
    });

    // Save database to persist changes
    this.db.save();

    const duration = Date.now() - startTime;

    log.success(`Created ${result.created} new skills, updated ${result.updated} existing skills in ${duration}ms`);

    // Display extracted patterns if available
    if (result.patterns.length > 0) {
      console.log('\n' + '═'.repeat(80));
      console.log(`${colors.bright}${colors.cyan}Extracted Patterns:${colors.reset}`);
      console.log('═'.repeat(80));

      result.patterns.forEach((pattern, i) => {
        console.log(`\n${colors.bright}#${i + 1}: ${pattern.task}${colors.reset}`);
        console.log(`  Avg Reward: ${colors.green}${pattern.avgReward.toFixed(2)}${colors.reset}`);

        if (pattern.commonPatterns.length > 0) {
          console.log(`  ${colors.cyan}Common Patterns:${colors.reset}`);
          pattern.commonPatterns.forEach(p => console.log(`    • ${p}`));
        }

        if (pattern.successIndicators.length > 0) {
          console.log(`  ${colors.yellow}Success Indicators:${colors.reset}`);
          pattern.successIndicators.forEach(s => console.log(`    • ${s}`));
        }

        console.log('─'.repeat(80));
      });
    }

    if (result.created === 0 && result.updated === 0) {
      log.warning('No episodes met the criteria for skill consolidation');
      log.info('Try lowering minReward or increasing timeWindowDays');
    }
  }

  async skillPrune(params: {
    minUses?: number;
    minSuccessRate?: number;
    maxAgeDays?: number;
  }): Promise<void> {
    if (!this.skills) throw new Error('Not initialized');

    log.header('\n🧹 Pruning Skills');

    const pruned = this.skills.pruneSkills({
      minUses: params.minUses || 3,
      minSuccessRate: params.minSuccessRate || 0.4,
      maxAgeDays: params.maxAgeDays || 60
    });

    log.success(`Pruned ${pruned} underperforming skills`);
  }

  // ============================================================================
  // QUIC Synchronization Commands
  // ============================================================================

  async quicStartServer(params: {
    port?: number;
    cert?: string;
    key?: string;
    authToken?: string;
    maxConnections?: number;
  }): Promise<void> {
    if (!this.db) throw new Error('Not initialized');

    log.header('\n[QUIC] Starting QUIC Sync Server');

    const spinner = new Spinner();
    const port = params.port || 4433;
    const authToken = params.authToken || this.generateAuthToken();

    console.log('\n' + '='.repeat(80));
    console.log(`${colors.bright}Server Configuration${colors.reset}`);
    console.log('='.repeat(80));
    console.log(`  Host: ${colors.cyan}0.0.0.0${colors.reset}`);
    console.log(`  Port: ${colors.cyan}${port}${colors.reset}`);
    console.log(`  Auth Token: ${colors.cyan}${authToken.substring(0, 8)}...${colors.reset}`);
    console.log(`  Max Connections: ${colors.cyan}${params.maxConnections || 100}${colors.reset}`);

    if (params.cert && params.key) {
      console.log(`  TLS Certificate: ${colors.green}${params.cert}${colors.reset}`);
      console.log(`  TLS Key: ${colors.green}${params.key}${colors.reset}`);
    } else {
      console.log(`  TLS: ${colors.yellow}Self-signed (development mode)${colors.reset}`);
    }
    console.log('='.repeat(80));

    try {
      spinner.start('Initializing QUIC server...');

      // Create QUIC server configuration
      const serverConfig: QUICServerConfig = {
        host: '0.0.0.0',
        port,
        maxConnections: params.maxConnections || 100,
        authToken,
        tlsConfig: {
          cert: params.cert,
          key: params.key,
        },
        rateLimit: {
          maxRequestsPerMinute: 60,
          maxBytesPerMinute: 10 * 1024 * 1024, // 10MB
        },
      };

      // Initialize the QUIC server with database
      this.quicServer = new QUICServer(this.db, serverConfig);

      spinner.update('Starting server...');
      await this.quicServer.start();

      spinner.succeed('QUIC server started successfully');

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Server Status${colors.reset}`);
      console.log('-'.repeat(80));

      const status = this.quicServer.getStatus();
      console.log(`  Status: ${colors.green}Running${colors.reset}`);
      console.log(`  Active Connections: ${colors.cyan}${status.activeConnections}${colors.reset}`);
      console.log(`  Rate Limit: ${colors.cyan}${status.config.rateLimit?.maxRequestsPerMinute} req/min${colors.reset}`);
      console.log('-'.repeat(80));

      log.info('\nServer is ready to accept connections.');
      log.info(`Clients can connect using: agentdb sync connect <host> ${port} --auth-token ${authToken}`);
      log.info('\nPress Ctrl+C to stop the server');

      // Handle graceful shutdown
      const shutdown = async () => {
        console.log('\n');
        log.info('Shutting down server...');
        const shutdownSpinner = new Spinner();
        shutdownSpinner.start('Closing connections...');

        try {
          if (this.quicServer) {
            await this.quicServer.stop();
          }
          shutdownSpinner.succeed('Server stopped gracefully');
        } catch (error) {
          shutdownSpinner.fail(`Error during shutdown: ${(error as Error).message}`);
        }
        process.exit(0);
      };

      process.on('SIGINT', shutdown);
      process.on('SIGTERM', shutdown);

      // Monitor server status periodically
      const monitorInterval = setInterval(() => {
        if (this.quicServer) {
          const currentStatus = this.quicServer.getStatus();
          if (currentStatus.activeConnections > 0) {
            process.stdout.write(`\r${colors.cyan}[${new Date().toLocaleTimeString()}]${colors.reset} Active connections: ${currentStatus.activeConnections} | Total requests: ${currentStatus.totalRequests}  `);
          }
        }
      }, 5000);

      // Keep process alive
      await new Promise<void>((resolve) => {
        process.on('SIGINT', () => {
          clearInterval(monitorInterval);
          resolve();
        });
        process.on('SIGTERM', () => {
          clearInterval(monitorInterval);
          resolve();
        });
      });
    } catch (error) {
      spinner.fail(`Failed to start server: ${(error as Error).message}`);
      throw error;
    }
  }

  async quicConnect(params: {
    host: string;
    port: number;
    authToken?: string;
    cert?: string;
    timeout?: number;
  }): Promise<void> {
    log.header('\n[QUIC] Connecting to QUIC Sync Server');

    const spinner = new Spinner();

    console.log('\n' + '='.repeat(80));
    console.log(`${colors.bright}Connection Details${colors.reset}`);
    console.log('='.repeat(80));
    console.log(`  Server: ${colors.cyan}${params.host}:${params.port}${colors.reset}`);
    console.log(`  Authentication: ${params.authToken ? colors.green + 'Enabled' : colors.yellow + 'Disabled'}${colors.reset}`);
    console.log(`  Timeout: ${colors.cyan}${params.timeout || 30000}ms${colors.reset}`);

    if (params.cert) {
      console.log(`  TLS Certificate: ${colors.green}${params.cert}${colors.reset}`);
    } else {
      console.log(`  TLS: ${colors.yellow}Insecure (no certificate provided)${colors.reset}`);
    }
    console.log('='.repeat(80));

    try {
      spinner.start('Initializing QUIC client...');

      // Create QUIC client configuration
      const clientConfig: QUICClientConfig = {
        serverHost: params.host,
        serverPort: params.port,
        authToken: params.authToken,
        maxRetries: 3,
        retryDelayMs: 1000,
        timeoutMs: params.timeout || 30000,
        poolSize: 5,
        tlsConfig: {
          cert: params.cert,
          rejectUnauthorized: !!params.cert,
        },
      };

      // Initialize the QUIC client
      this.quicClient = new QUICClient(clientConfig);

      spinner.update('Establishing connection...');
      await this.quicClient.connect();

      spinner.update('Testing connection (ping)...');
      const pingResult = await this.quicClient.ping();

      if (!pingResult.success) {
        throw new Error(pingResult.error || 'Ping failed');
      }

      spinner.succeed('Connected to remote server successfully');

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Connection Status${colors.reset}`);
      console.log('-'.repeat(80));

      const status = this.quicClient.getStatus();
      console.log(`  Status: ${colors.green}Connected${colors.reset}`);
      console.log(`  Latency: ${colors.cyan}${pingResult.latencyMs}ms${colors.reset}`);
      console.log(`  Pool Size: ${colors.cyan}${status.poolSize}${colors.reset}`);
      console.log(`  Active Connections: ${colors.cyan}${status.activeConnections}${colors.reset}`);
      console.log('-'.repeat(80));

      // Initialize the SyncCoordinator for subsequent push/pull operations
      if (this.db) {
        this.syncCoordinator = new SyncCoordinator({
          db: this.db,
          client: this.quicClient,
        });
        log.info('\nSync coordinator initialized. Ready for push/pull operations.');
      }

      log.success('\nConnection established.');
      log.info('Use "agentdb sync push" or "agentdb sync pull" to sync data.');
    } catch (error) {
      spinner.fail(`Connection failed: ${(error as Error).message}`);

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Troubleshooting${colors.reset}`);
      console.log('-'.repeat(80));
      console.log('  1. Verify the server is running and accessible');
      console.log('  2. Check firewall settings allow traffic on the specified port');
      console.log('  3. Ensure the auth token matches the server configuration');
      console.log('  4. For TLS connections, verify the certificate is valid');
      console.log('-'.repeat(80));

      throw error;
    }
  }

  async quicPush(params: {
    server: string;
    incremental?: boolean;
    filter?: string;
    authToken?: string;
    batchSize?: number;
  }): Promise<void> {
    if (!this.db) throw new Error('Not initialized');

    log.header('\n[QUIC] Pushing Changes to Remote');

    const spinner = new Spinner();
    const progressBar = new ProgressBar();

    // Parse server address
    const [host, portStr] = params.server.split(':');
    const port = parseInt(portStr) || 4433;

    console.log('\n' + '='.repeat(80));
    console.log(`${colors.bright}Push Configuration${colors.reset}`);
    console.log('='.repeat(80));
    console.log(`  Server: ${colors.cyan}${host}:${port}${colors.reset}`);
    console.log(`  Mode: ${colors.cyan}${params.incremental ? 'Incremental' : 'Full Sync'}${colors.reset}`);
    console.log(`  Batch Size: ${colors.cyan}${params.batchSize || 100}${colors.reset}`);
    if (params.filter) {
      console.log(`  Filter: ${colors.cyan}${params.filter}${colors.reset}`);
    }
    console.log('='.repeat(80));

    try {
      // Step 1: Analyze pending changes
      spinner.start('Analyzing local changes...');
      const pendingChanges = this.getPendingChangesDetailed(params.incremental, params.filter);
      spinner.succeed(`Found ${pendingChanges.totalItems} items to push`);

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Pending Changes${colors.reset}`);
      console.log('-'.repeat(80));
      console.log(`  Episodes: ${colors.cyan}${pendingChanges.episodes.length}${colors.reset}`);
      console.log(`  Skills: ${colors.cyan}${pendingChanges.skills.length}${colors.reset}`);
      console.log(`  Edges: ${colors.cyan}${pendingChanges.edges.length}${colors.reset}`);
      console.log(`  Total Size: ${colors.cyan}${(pendingChanges.totalSize / 1024).toFixed(2)} KB${colors.reset}`);
      console.log('-'.repeat(80));

      if (pendingChanges.totalItems === 0) {
        log.info('\nNo changes to push. Database is up to date.');
        return;
      }

      // Step 2: Establish connection if not already connected
      spinner.start('Connecting to server...');

      if (!this.quicClient) {
        const clientConfig: QUICClientConfig = {
          serverHost: host,
          serverPort: port,
          authToken: params.authToken,
          maxRetries: 3,
          retryDelayMs: 1000,
          timeoutMs: 30000,
          poolSize: 5,
        };
        this.quicClient = new QUICClient(clientConfig);
        await this.quicClient.connect();
      }
      spinner.succeed('Connected to server');

      // Step 3: Initialize sync coordinator if not exists
      if (!this.syncCoordinator) {
        this.syncCoordinator = new SyncCoordinator({
          db: this.db,
          client: this.quicClient,
          batchSize: params.batchSize || 100,
        });
      }

      // Step 4: Push changes with progress tracking
      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Push Progress${colors.reset}`);
      console.log('-'.repeat(80));

      let pushedItems = 0;
      const startTime = Date.now();

      // Perform sync with progress callback
      const syncReport = await this.syncCoordinator.sync((progress: SyncProgress) => {
        if (progress.phase === 'pushing') {
          pushedItems = progress.current;
          progressBar.update(progress.current, progress.total, progress.itemType || 'items');
        } else if (progress.phase === 'completed') {
          progressBar.complete('Push completed');
        } else if (progress.phase === 'error') {
          console.log(`\n${colors.red}Error: ${progress.error}${colors.reset}`);
        }
      });

      const duration = Date.now() - startTime;

      console.log('\n' + '='.repeat(80));
      console.log(`${colors.bright}Push Summary${colors.reset}`);
      console.log('='.repeat(80));
      console.log(`  Status: ${syncReport.success ? colors.green + 'Success' : colors.red + 'Failed'}${colors.reset}`);
      console.log(`  Items Pushed: ${colors.cyan}${syncReport.itemsPushed}${colors.reset}`);
      console.log(`  Bytes Transferred: ${colors.cyan}${(syncReport.bytesTransferred / 1024).toFixed(2)} KB${colors.reset}`);
      console.log(`  Duration: ${colors.cyan}${duration}ms${colors.reset}`);
      console.log(`  Throughput: ${colors.cyan}${((syncReport.bytesTransferred / 1024) / (duration / 1000)).toFixed(2)} KB/s${colors.reset}`);

      if (syncReport.errors.length > 0) {
        console.log(`  Errors: ${colors.red}${syncReport.errors.length}${colors.reset}`);
        syncReport.errors.forEach((err, i) => {
          console.log(`    ${i + 1}. ${err}`);
        });
      }
      console.log('='.repeat(80));

      if (syncReport.success) {
        log.success('\nChanges pushed successfully');
      } else {
        log.error('\nPush completed with errors');
      }
    } catch (error) {
      spinner.fail(`Push failed: ${(error as Error).message}`);
      throw error;
    }
  }

  // Helper to get detailed pending changes with actual data
  private getPendingChangesDetailed(incremental?: boolean, filter?: string): {
    episodes: any[];
    skills: any[];
    edges: any[];
    totalItems: number;
    totalSize: number;
  } {
    if (!this.db) {
      return { episodes: [], skills: [], edges: [], totalItems: 0, totalSize: 0 };
    }

    try {
      // Get last sync time
      let lastSyncTime = 0;
      if (incremental) {
        try {
          const syncState = this.db.prepare('SELECT last_sync_at FROM sync_state WHERE id = 1').get();
          if (syncState) {
            lastSyncTime = syncState.last_sync_at || 0;
          }
        } catch {
          // Table might not exist
        }
      }

      // Query episodes
      let episodesQuery = 'SELECT * FROM episodes WHERE 1=1';
      const episodeParams: any[] = [];
      if (incremental && lastSyncTime > 0) {
        episodesQuery += ' AND ts > ?';
        episodeParams.push(lastSyncTime);
      }
      const episodes = this.db.prepare(episodesQuery).all(...episodeParams) || [];

      // Query skills
      let skillsQuery = 'SELECT * FROM skills WHERE 1=1';
      const skillParams: any[] = [];
      if (incremental && lastSyncTime > 0) {
        skillsQuery += ' AND ts > ?';
        skillParams.push(lastSyncTime);
      }
      let skills: any[] = [];
      try {
        skills = this.db.prepare(skillsQuery).all(...skillParams) || [];
      } catch {
        // Skills table might not exist
      }

      // Query edges
      let edgesQuery = 'SELECT * FROM skill_edges WHERE 1=1';
      const edgeParams: any[] = [];
      if (incremental && lastSyncTime > 0) {
        edgesQuery += ' AND ts > ?';
        edgeParams.push(lastSyncTime);
      }
      let edges: any[] = [];
      try {
        edges = this.db.prepare(edgesQuery).all(...edgeParams) || [];
      } catch {
        // Edges table might not exist
      }

      // Calculate total size
      const totalSize =
        JSON.stringify(episodes).length +
        JSON.stringify(skills).length +
        JSON.stringify(edges).length;

      return {
        episodes,
        skills,
        edges,
        totalItems: episodes.length + skills.length + edges.length,
        totalSize,
      };
    } catch (error) {
      log.warning(`Error analyzing changes: ${(error as Error).message}`);
      return { episodes: [], skills: [], edges: [], totalItems: 0, totalSize: 0 };
    }
  }

  async quicPull(params: {
    server: string;
    incremental?: boolean;
    filter?: string;
    authToken?: string;
    batchSize?: number;
    conflictStrategy?: 'local-wins' | 'remote-wins' | 'latest-wins';
  }): Promise<void> {
    if (!this.db) throw new Error('Not initialized');

    log.header('\n[QUIC] Pulling Changes from Remote');

    const spinner = new Spinner();
    const progressBar = new ProgressBar();

    // Parse server address
    const [host, portStr] = params.server.split(':');
    const port = parseInt(portStr) || 4433;

    console.log('\n' + '='.repeat(80));
    console.log(`${colors.bright}Pull Configuration${colors.reset}`);
    console.log('='.repeat(80));
    console.log(`  Server: ${colors.cyan}${host}:${port}${colors.reset}`);
    console.log(`  Mode: ${colors.cyan}${params.incremental ? 'Incremental' : 'Full Sync'}${colors.reset}`);
    console.log(`  Conflict Strategy: ${colors.cyan}${params.conflictStrategy || 'latest-wins'}${colors.reset}`);
    console.log(`  Batch Size: ${colors.cyan}${params.batchSize || 100}${colors.reset}`);
    if (params.filter) {
      console.log(`  Filter: ${colors.cyan}${params.filter}${colors.reset}`);
    }
    console.log('='.repeat(80));

    try {
      // Step 1: Establish connection if not already connected
      spinner.start('Connecting to server...');

      if (!this.quicClient) {
        const clientConfig: QUICClientConfig = {
          serverHost: host,
          serverPort: port,
          authToken: params.authToken,
          maxRetries: 3,
          retryDelayMs: 1000,
          timeoutMs: 30000,
          poolSize: 5,
        };
        this.quicClient = new QUICClient(clientConfig);
        await this.quicClient.connect();
      }
      spinner.succeed('Connected to server');

      // Step 2: Initialize sync coordinator if not exists
      if (!this.syncCoordinator) {
        this.syncCoordinator = new SyncCoordinator({
          db: this.db,
          client: this.quicClient,
          batchSize: params.batchSize || 100,
          conflictStrategy: params.conflictStrategy || 'latest-wins',
        });
      }

      // Step 3: Pull changes with progress tracking
      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Pull Progress${colors.reset}`);
      console.log('-'.repeat(80));

      const startTime = Date.now();
      let currentPhase = '';
      let episodesReceived = 0;
      let skillsReceived = 0;
      let edgesReceived = 0;

      // Perform sync with progress callback
      const syncReport = await this.syncCoordinator.sync((progress: SyncProgress) => {
        if (progress.phase === 'pulling') {
          currentPhase = progress.itemType || 'items';
          progressBar.update(progress.current, progress.total, `Pulling ${currentPhase}...`);

          // Track received items
          if (progress.itemType === 'episodes') episodesReceived = progress.current;
          if (progress.itemType === 'skills') skillsReceived = progress.current;
          if (progress.itemType === 'edges') edgesReceived = progress.current;
        } else if (progress.phase === 'resolving') {
          progressBar.update(progress.current, progress.total, 'Resolving conflicts...');
        } else if (progress.phase === 'applying') {
          progressBar.update(progress.current, progress.total, 'Applying changes...');
        } else if (progress.phase === 'completed') {
          progressBar.complete('Pull completed');
        } else if (progress.phase === 'error') {
          console.log(`\n${colors.red}Error: ${progress.error}${colors.reset}`);
        }
      });

      const duration = Date.now() - startTime;

      console.log('\n' + '='.repeat(80));
      console.log(`${colors.bright}Pull Summary${colors.reset}`);
      console.log('='.repeat(80));
      console.log(`  Status: ${syncReport.success ? colors.green + 'Success' : colors.red + 'Failed'}${colors.reset}`);
      console.log(`  Items Pulled: ${colors.cyan}${syncReport.itemsPulled}${colors.reset}`);
      console.log(`  Conflicts Resolved: ${colors.cyan}${syncReport.conflictsResolved}${colors.reset}`);
      console.log(`  Bytes Transferred: ${colors.cyan}${(syncReport.bytesTransferred / 1024).toFixed(2)} KB${colors.reset}`);
      console.log(`  Duration: ${colors.cyan}${duration}ms${colors.reset}`);
      console.log(`  Throughput: ${colors.cyan}${((syncReport.bytesTransferred / 1024) / (duration / 1000)).toFixed(2)} KB/s${colors.reset}`);

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Received Data${colors.reset}`);
      console.log('-'.repeat(80));
      console.log(`  Episodes: ${colors.cyan}${episodesReceived}${colors.reset}`);
      console.log(`  Skills: ${colors.cyan}${skillsReceived}${colors.reset}`);
      console.log(`  Edges: ${colors.cyan}${edgesReceived}${colors.reset}`);
      console.log('-'.repeat(80));

      if (syncReport.errors.length > 0) {
        console.log(`\n${colors.bright}Errors${colors.reset}`);
        console.log('-'.repeat(80));
        syncReport.errors.forEach((err, i) => {
          console.log(`  ${colors.red}${i + 1}. ${err}${colors.reset}`);
        });
        console.log('-'.repeat(80));
      }
      console.log('='.repeat(80));

      if (syncReport.success) {
        log.success('\nChanges pulled and merged successfully');
      } else {
        log.error('\nPull completed with errors');
      }

      // Display local database status after pull
      spinner.start('Verifying local database...');
      const localStatus = this.getLocalDatabaseStatus();
      spinner.succeed('Local database verified');

      console.log('\n' + '-'.repeat(80));
      console.log(`${colors.bright}Local Database Status${colors.reset}`);
      console.log('-'.repeat(80));
      console.log(`  Total Episodes: ${colors.cyan}${localStatus.episodes}${colors.reset}`);
      console.log(`  Total Skills: ${colors.cyan}${localStatus.skills}${colors.reset}`);
      console.log(`  Total Edges: ${colors.cyan}${localStatus.edges}${colors.reset}`);
      console.log('-'.repeat(80));
    } catch (error) {
      spinner.fail(`Pull failed: ${(error as Error).message}`);
      throw error;
    }
  }

  // Helper to get local database status
  private getLocalDatabaseStatus(): {
    episodes: number;
    skills: number;
    edges: number;
  } {
    if (!this.db) {
      return { episodes: 0, skills: 0, edges: 0 };
    }

    let episodes = 0;
    let skills = 0;
    let edges = 0;

    try {
      const episodeCount = this.db.prepare('SELECT COUNT(*) as count FROM episodes').get();
      episodes = episodeCount?.count || 0;
    } catch {
      // Table might not exist
    }

    try {
      const skillCount = this.db.prepare('SELECT COUNT(*) as count FROM skills').get();
      skills = skillCount?.count || 0;
    } catch {
      // Table might not exist
    }

    try {
      const edgeCount = this.db.prepare('SELECT COUNT(*) as count FROM skill_edges').get();
      edges = edgeCount?.count || 0;
    } catch {
      // Table might not exist
    }

    return { episodes, skills, edges };
  }

  async quicStatus(): Promise<void> {
    if (!this.db) throw new Error('Not initialized');

    log.header('\n[QUIC] Sync Status');

    const spinner = new Spinner();

    spinner.start('Loading sync status...');

    // Get sync metadata from database
    const syncMeta = this.getSyncMetadataFromDb();

    // Get pending changes
    const pendingChanges = this.getPendingChangesDetailed(true);

    spinner.succeed('Status loaded');

    console.log('\n' + '='.repeat(80));
    console.log(`${colors.bright}Sync Overview${colors.reset}`);
    console.log('='.repeat(80));

    // Last sync information
    if (syncMeta.lastSyncAt > 0) {
      const lastSync = new Date(syncMeta.lastSyncAt);
      console.log(`  Last Sync: ${colors.cyan}${lastSync.toLocaleString()}${colors.reset}`);
      console.log(`  Time Ago: ${colors.cyan}${this.timeAgo(Math.floor(syncMeta.lastSyncAt / 1000))}${colors.reset}`);
    } else {
      console.log(`  Last Sync: ${colors.yellow}Never${colors.reset}`);
    }

    console.log(`  Total Syncs: ${colors.cyan}${syncMeta.syncCount}${colors.reset}`);
    console.log(`  Total Items Synced: ${colors.cyan}${syncMeta.totalItemsSynced}${colors.reset}`);
    console.log(`  Total Data Synced: ${colors.cyan}${(syncMeta.totalBytesSynced / 1024).toFixed(2)} KB${colors.reset}`);

    if (syncMeta.lastError) {
      console.log(`  Last Error: ${colors.red}${syncMeta.lastError}${colors.reset}`);
    }

    console.log('\n' + '-'.repeat(80));
    console.log(`${colors.bright}Pending Changes (Unsynced)${colors.reset}`);
    console.log('-'.repeat(80));
    console.log(`  Episodes: ${colors.cyan}${pendingChanges.episodes.length}${colors.reset}`);
    console.log(`  Skills: ${colors.cyan}${pendingChanges.skills.length}${colors.reset}`);
    console.log(`  Edges: ${colors.cyan}${pendingChanges.edges.length}${colors.reset}`);
    console.log(`  Total Size: ${colors.cyan}${(pendingChanges.totalSize / 1024).toFixed(2)} KB${colors.reset}`);

    console.log('\n' + '-'.repeat(80));
    console.log(`${colors.bright}Local Database${colors.reset}`);
    console.log('-'.repeat(80));
    const localStatus = this.getLocalDatabaseStatus();
    console.log(`  Episodes: ${colors.cyan}${localStatus.episodes}${colors.reset}`);
    console.log(`  Skills: ${colors.cyan}${localStatus.skills}${colors.reset}`);
    console.log(`  Edges: ${colors.cyan}${localStatus.edges}${colors.reset}`);

    console.log('\n' + '-'.repeat(80));
    console.log(`${colors.bright}Controller Status${colors.reset}`);
    console.log('-'.repeat(80));

    // Server status
    if (this.quicServer) {
      const serverStatus = this.quicServer.getStatus();
      console.log(`  QUIC Server: ${serverStatus.isRunning ? colors.green + 'Running' : colors.yellow + 'Stopped'}${colors.reset}`);
      if (serverStatus.isRunning) {
        console.log(`    Active Connections: ${colors.cyan}${serverStatus.activeConnections}${colors.reset}`);
        console.log(`    Total Requests: ${colors.cyan}${serverStatus.totalRequests}${colors.reset}`);
      }
    } else {
      console.log(`  QUIC Server: ${colors.yellow}Not initialized${colors.reset}`);
    }

    // Client status
    if (this.quicClient) {
      const clientStatus = this.quicClient.getStatus();
      console.log(`  QUIC Client: ${clientStatus.isConnected ? colors.green + 'Connected' : colors.yellow + 'Disconnected'}${colors.reset}`);
      if (clientStatus.isConnected) {
        console.log(`    Server: ${colors.cyan}${clientStatus.config.serverHost}:${clientStatus.config.serverPort}${colors.reset}`);
        console.log(`    Pool Size: ${colors.cyan}${clientStatus.poolSize}${colors.reset}`);
        console.log(`    Total Requests: ${colors.cyan}${clientStatus.totalRequests}${colors.reset}`);
      }
    } else {
      console.log(`  QUIC Client: ${colors.yellow}Not initialized${colors.reset}`);
    }

    // Sync coordinator status
    if (this.syncCoordinator) {
      const coordStatus = this.syncCoordinator.getStatus();
      console.log(`  Sync Coordinator: ${colors.green}Ready${colors.reset}`);
      console.log(`    Auto-Sync: ${coordStatus.autoSyncEnabled ? colors.green + 'Enabled' : colors.yellow + 'Disabled'}${colors.reset}`);
      console.log(`    Currently Syncing: ${coordStatus.isSyncing ? colors.yellow + 'Yes' : colors.cyan + 'No'}${colors.reset}`);
    } else {
      console.log(`  Sync Coordinator: ${colors.yellow}Not initialized${colors.reset}`);
    }

    console.log('='.repeat(80));

    // Recommendations
    if (pendingChanges.totalItems > 0) {
      log.info(`\nTip: You have ${pendingChanges.totalItems} unsynced items. Use "agentdb sync push" to sync them.`);
    }

    if (!this.quicClient && !this.quicServer) {
      log.info('\nTo start syncing:');
      log.info('  - Start a server: agentdb sync start-server --port 4433');
      log.info('  - Or connect to one: agentdb sync connect <host> <port>');
    }
  }

  // Get sync metadata from database
  private getSyncMetadataFromDb(): {
    lastSyncAt: number;
    lastEpisodeSync: number;
    lastSkillSync: number;
    lastEdgeSync: number;
    totalItemsSynced: number;
    totalBytesSynced: number;
    syncCount: number;
    lastError?: string;
  } {
    if (!this.db) {
      return {
        lastSyncAt: 0,
        lastEpisodeSync: 0,
        lastSkillSync: 0,
        lastEdgeSync: 0,
        totalItemsSynced: 0,
        totalBytesSynced: 0,
        syncCount: 0,
      };
    }

    try {
      const row = this.db
        .prepare('SELECT * FROM sync_state WHERE id = 1')
        .get();

      if (row) {
        return {
          lastSyncAt: row.last_sync_at || 0,
          lastEpisodeSync: row.last_episode_sync || 0,
          lastSkillSync: row.last_skill_sync || 0,
          lastEdgeSync: row.last_edge_sync || 0,
          totalItemsSynced: row.total_items_synced || 0,
          totalBytesSynced: row.total_bytes_synced || 0,
          syncCount: row.sync_count || 0,
          lastError: row.last_error || undefined,
        };
      }
    } catch {
      // Table might not exist
    }

    return {
      lastSyncAt: 0,
      lastEpisodeSync: 0,
      lastSkillSync: 0,
      lastEdgeSync: 0,
      totalItemsSynced: 0,
      totalBytesSynced: 0,
      syncCount: 0,
    };
  }

  private generateAuthToken(): string {
    // Generate a random 32-character token
    return Array.from({ length: 32 }, () =>
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }

  private getPendingChanges(incremental?: boolean, filter?: string): {
    episodes: number;
    skills: number;
    causalEdges: number;
    totalSize: number;
  } {
    // Mock implementation - would query sync metadata table
    return {
      episodes: 10,
      skills: 3,
      causalEdges: 5,
      totalSize: 25600
    };
  }

  private getSyncMetadata(): {
    lastSyncTime: number | null;
    pendingEpisodes: number;
    pendingSkills: number;
    pendingCausalEdges: number;
    servers: Array<{
      host: string;
      port: number;
      connected: boolean;
      lastSeen: number;
    }>;
  } {
    // Mock implementation - would query sync metadata table
    return {
      lastSyncTime: null,
      pendingEpisodes: 10,
      pendingSkills: 3,
      pendingCausalEdges: 5,
      servers: []
    };
  }

  private timeAgo(timestamp: number): string {
    const now = Math.floor(Date.now() / 1000);
    const diff = now - timestamp;

    if (diff < 60) return `${diff} seconds ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
    return `${Math.floor(diff / 86400)} days ago`;
  }

  // ============================================================================
  // Database Commands
  // ============================================================================

  async dbStats(): Promise<void> {
    if (!this.db) throw new Error('Not initialized');

    log.header('\n📊 Database Statistics');

    const tables = ['causal_edges', 'causal_experiments', 'causal_observations',
                    'certificates', 'provenance_lineage', 'episodes'];

    console.log('\n' + '═'.repeat(80));
    tables.forEach(table => {
      try {
        const count = this.db!.prepare(`SELECT COUNT(*) as count FROM ${table}`).get() as { count: number };
        console.log(`${colors.bright}${table}:${colors.reset} ${colors.cyan}${count.count}${colors.reset} records`);
      } catch (e) {
        console.log(`${colors.bright}${table}:${colors.reset} ${colors.yellow}N/A${colors.reset}`);
      }
    });
    console.log('═'.repeat(80));
  }
}

// ============================================================================
// CLI Entry Point
// ============================================================================

/**
 * Helper function to execute Commander-based commands
 */
async function handleCommanderCommand(command: any, args: string[]): Promise<void> {
  // Parse directly using the command instance without wrapping in a parent program
  // This avoids issues with subcommand routing
  await command.parseAsync(['node', command.name(), ...args], { from: 'user' });
}

async function main() {
  const args = process.argv.slice(2);

  // Handle version flag
  if (args[0] === '--version' || args[0] === '-v' || args[0] === 'version') {
    // Try multiple paths to find package.json (handles different execution contexts)
    const possiblePaths = [
      path.join(__dirname, '../../package.json'),  // dist/src/cli/../../package.json (local dev)
      path.join(__dirname, '../../../package.json'), // dist/package.json (published package)
      path.join(process.cwd(), 'package.json'),
      path.join(process.cwd(), 'node_modules/agentdb/package.json')
    ];

    let version = '2.0.0-alpha.2.6'; // Fallback version
    for (const pkgPath of possiblePaths) {
      try {
        if (fs.existsSync(pkgPath)) {
          const packageJson = JSON.parse(fs.readFileSync(pkgPath, 'utf-8'));
          if (packageJson.name === 'agentdb' && packageJson.version) {
            version = packageJson.version;
            break;
          }
        }
      } catch {
        continue;
      }
    }
    console.log(`agentdb v${version}`);
    process.exit(0);
  }

  // Handle help - show help if no args or help flag
  if (args.length === 0 || args[0] === 'help' || args[0] === '--help' || args[0] === '-h') {
    printHelp();
    process.exit(0);
  }

  const command = args[0];

  // Handle MCP server command separately (doesn't need CLI initialization)
  if (command === 'mcp') {
    await handleMcpCommand(args.slice(1));
    return;
  }

  // Handle init command with new v2 implementation
  if (command === 'init') {
    const options: any = { dbPath: './agentdb.db', dimension: 384 };
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--backend' && i + 1 < args.length) {
        options.backend = args[++i];
      } else if (arg === '--dimension' && i + 1 < args.length) {
        options.dimension = parseInt(args[++i]);
      } else if (arg === '--model' && i + 1 < args.length) {
        options.model = args[++i];
      } else if (arg === '--preset' && i + 1 < args.length) {
        options.preset = args[++i];
      } else if (arg === '--in-memory') {
        options.inMemory = true;
      } else if (arg === '--dry-run') {
        options.dryRun = true;
      } else if (arg === '--db' && i + 1 < args.length) {
        options.dbPath = args[++i];
      } else if (!arg.startsWith('--')) {
        options.dbPath = arg;
      }
    }
    await initCommand(options);
    return;
  }

  // Handle status command
  if (command === 'status') {
    const options: any = { dbPath: './agentdb.db', verbose: false };
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--db' && i + 1 < args.length) {
        options.dbPath = args[++i];
      } else if (arg === '--verbose' || arg === '-v') {
        options.verbose = true;
      } else if (!arg.startsWith('--')) {
        options.dbPath = arg;
      }
    }
    await statusCommand(options);
    return;
  }

  // Handle install-embeddings command
  if (command === 'install-embeddings') {
    const options: any = { global: false };
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--global' || arg === '-g') {
        options.global = true;
      }
    }
    await installEmbeddingsCommand(options);
    return;
  }

  // Handle migrate command
  if (command === 'migrate') {
    const options: any = { optimize: true, dryRun: false, verbose: false };
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--source' && i + 1 < args.length) {
        options.sourceDb = args[++i];
      } else if (arg === '--target' && i + 1 < args.length) {
        options.targetDb = args[++i];
      } else if (arg === '--no-optimize') {
        options.optimize = false;
      } else if (arg === '--dry-run') {
        options.dryRun = true;
      } else if (arg === '--verbose' || arg === '-v') {
        options.verbose = true;
      } else if (!arg.startsWith('--') && !options.sourceDb) {
        options.sourceDb = arg;
      }
    }
    if (!options.sourceDb) {
      log.error('Source database path required');
      console.log('Usage: agentdb migrate <source-db> [--target <target-db>] [--no-optimize] [--dry-run] [--verbose]');
      process.exit(1);
    }
    await migrateCommand(options);
    return;
  }

  // Handle doctor command
  if (command === 'doctor') {
    const options: any = { verbose: false };
    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--db' && i + 1 < args.length) {
        options.dbPath = args[++i];
      } else if (arg === '--verbose' || arg === '-v') {
        options.verbose = true;
      } else if (!arg.startsWith('--')) {
        options.dbPath = arg;
      }
    }
    await doctorCommand(options);
    return;
  }

  // Handle vector search commands (no CLI initialization needed)
  if (command === 'vector-search') {
    await handleVectorSearchCommand(args.slice(1));
    return;
  }

  if (command === 'export') {
    await handleExportCommand(args.slice(1));
    return;
  }

  if (command === 'import') {
    await handleImportCommand(args.slice(1));
    return;
  }

  if (command === 'stats') {
    await handleStatsCommand(args.slice(1));
    return;
  }

  // Handle advanced neural commands (WASM-accelerated when available)
  if (command === 'attention') {
    await handleCommanderCommand(attentionCommand, args.slice(1));
    return;
  }

  if (command === 'learn') {
    await handleCommanderCommand(learnCommand, args.slice(1));
    return;
  }

  if (command === 'route') {
    await handleCommanderCommand(routeCommand, args.slice(1));
    return;
  }

  if (command === 'hyperbolic') {
    await handleCommanderCommand(hyperbolicCommand, args.slice(1));
    return;
  }

  // Handle simulate command - run simulation CLI
  if (command === 'simulate') {
    // Use pathToFileURL for proper ESM module resolution
    const { pathToFileURL } = await import('url');

    // Get current directory using import.meta.url
    const currentUrl = import.meta.url;
    const currentPath = currentUrl.replace(/^file:\/\//, '');
    const __dirname = path.dirname(currentPath);

    // Dynamic import with proper file URL for ESM compatibility
    // Note: simulation files are in dist/simulation, not dist/src/simulation
    const runnerPath = path.resolve(__dirname, '../../simulation/runner.js');
    const runnerUrl = pathToFileURL(runnerPath).href;

    try {
      const { runSimulation, listScenarios, initScenario } = await import(runnerUrl);
      const subcommand = args[1];

      if (!subcommand || subcommand === 'list') {
        await listScenarios();
        return;
      }

      if (subcommand === 'init') {
        const scenario = args[2];
        const options: any = { template: 'basic' };
        for (let i = 3; i < args.length; i++) {
          if (args[i] === '-t' || args[i] === '--template') {
            options.template = args[++i];
          }
        }
        await initScenario(scenario, options);
        return;
      }

      if (subcommand === 'run') {
        const scenario = args[2];
        const options: any = {
          config: 'simulation/configs/default.json',
          verbosity: '2',
          iterations: '10',
          swarmSize: '5',
          model: 'anthropic/claude-3.5-sonnet',
          parallel: false,
          output: 'simulation/reports',
          stream: false,
          optimize: false
        };

        for (let i = 3; i < args.length; i++) {
          const arg = args[i];
          if (arg === '-c' || arg === '--config') options.config = args[++i];
          else if (arg === '-v' || arg === '--verbosity') options.verbosity = args[++i];
          else if (arg === '-i' || arg === '--iterations') options.iterations = args[++i];
          else if (arg === '-s' || arg === '--swarm-size') options.swarmSize = args[++i];
          else if (arg === '-m' || arg === '--model') options.model = args[++i];
          else if (arg === '-p' || arg === '--parallel') options.parallel = true;
          else if (arg === '-o' || arg === '--output') options.output = args[++i];
          else if (arg === '--stream') options.stream = true;
          else if (arg === '--optimize') options.optimize = true;
        }

        await runSimulation(scenario, options);
        return;
      }

      log.error(`Unknown simulate subcommand: ${subcommand}`);
      log.info('Available: simulate list, simulate run <scenario>, simulate init <scenario>');
      return;
    } catch (error) {
      log.error(`Failed to load simulation module: ${(error as Error).message}`);
      log.info('Falling back to agentdb-simulate binary...');
      log.info('Usage: npx agentdb-simulate <command>');
      return;
    }
  }

  const cli = new AgentDBCLI();
  const dbPath = process.env.AGENTDB_PATH || './agentdb.db';

  try {
    await cli.initialize(dbPath);

    const subcommand = args[1];

    if (command === 'causal') {
      await handleCausalCommands(cli, subcommand, args.slice(2));
    } else if (command === 'recall') {
      await handleRecallCommands(cli, subcommand, args.slice(2));
    } else if (command === 'learner') {
      await handleLearnerCommands(cli, subcommand, args.slice(2));
    } else if (command === 'reflexion') {
      await handleReflexionCommands(cli, subcommand, args.slice(2));
    } else if (command === 'skill') {
      await handleSkillCommands(cli, subcommand, args.slice(2));
    } else if (command === 'db') {
      await handleDbCommands(cli, subcommand, args.slice(2));
    } else if (command === 'sync') {
      await handleSyncCommands(cli, subcommand, args.slice(2));
    } else if (command === 'query') {
      await handleQueryCommand(cli, args.slice(1));
    } else if (command === 'store-pattern') {
      await handleStorePatternCommand(cli, args.slice(1));
    } else if (command === 'train') {
      await handleTrainCommand(cli, args.slice(1));
    } else if (command === 'optimize-memory') {
      await handleOptimizeMemoryCommand(cli, args.slice(1));
    } else {
      log.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
    }
  } catch (error) {
    log.error((error as Error).message);
    process.exit(1);
  }
}

// Command handlers

// Init command handler
async function handleInitCommand(args: string[]) {
  // Parse arguments
  let dbPath = './agentdb.db';
  let dimension = 1536; // Default OpenAI ada-002
  let preset: 'small' | 'medium' | 'large' | null = null;
  let inMemory = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--dimension' && i + 1 < args.length) {
      dimension = parseInt(args[++i]);
    } else if (arg === '--preset' && i + 1 < args.length) {
      preset = args[++i] as 'small' | 'medium' | 'large';
    } else if (arg === '--in-memory') {
      inMemory = true;
      dbPath = ':memory:';
    } else if (!arg.startsWith('--')) {
      dbPath = arg;
    }
  }

  // Apply preset configurations
  if (preset) {
    if (preset === 'small') {
      log.info('Using SMALL preset (<10K vectors)');
    } else if (preset === 'medium') {
      log.info('Using MEDIUM preset (10K-100K vectors)');
    } else if (preset === 'large') {
      log.info('Using LARGE preset (>100K vectors)');
    }
  }

  log.info(`Initializing AgentDB at: ${dbPath}`);
  log.info(`Embedding dimension: ${dimension}`);
  if (inMemory) {
    log.info('Using in-memory database (data will not persist)');
  }

  // Check if database already exists
  if (!inMemory && fs.existsSync(dbPath)) {
    log.warning(`Database already exists at ${dbPath}`);
    log.info('Use a different path or remove the existing file to reinitialize');
    return;
  }

  // Create parent directories if needed
  if (!inMemory) {
    const parentDir = path.dirname(dbPath);
    if (parentDir !== '.' && !fs.existsSync(parentDir)) {
      log.info(`Creating directory: ${parentDir}`);
      fs.mkdirSync(parentDir, { recursive: true });
    }
  }

  // Create new database with schemas
  const cli = new AgentDBCLI();
  await cli.initialize(dbPath);

  // CRITICAL: Save the database to disk (unless in-memory)
  // sql.js keeps everything in memory until explicitly saved
  if (!inMemory) {
    if (cli.db && typeof cli.db.save === 'function') {
      cli.db.save();
    } else if (cli.db && typeof cli.db.close === 'function') {
      // close() calls save() internally
      cli.db.close();
    }

    // Verify database file was created
    if (!fs.existsSync(dbPath)) {
      log.error(`Failed to create database file at ${dbPath}`);
      log.error('The database may be in memory only');
      process.exit(1);
    }
  }

  // Verify database has tables
  try {
    const db = await createDatabase(dbPath);
    const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all();
    db.close();

    if (tables.length === 0) {
      log.warning('Database file created but no tables found');
      log.warning('Schemas may not have been loaded correctly');
    } else {
      log.success(`Database created with ${tables.length} tables`);
    }
  } catch (error) {
    log.warning(`Could not verify database tables: ${(error as Error).message}`);
  }

  log.success(`✅ AgentDB initialized successfully at ${dbPath}`);
  log.info('Database includes:');
  log.info('  - Core vector tables (episodes, embeddings)');
  log.info('  - Causal memory graph');
  log.info('  - Reflexion memory');
  log.info('  - Skill library');
  log.info('  - Learning system');
  log.info('');
  log.info('Next steps:');
  log.info('  - Use "agentdb mcp start" to start MCP server');
  log.info('  - Use "agentdb causal add" to add causal edges');
  log.info('  - Use "agentdb reflexion add" to store episodes');
  log.info('  - See "agentdb help" for all commands');
}

async function handleMcpCommand(args: string[]) {
  const subcommand = args[0];

  if (subcommand === 'start' || !subcommand) {
    console.error('Starting AgentDB MCP Server...');

    // Spawn the MCP server as a child process (like agentic-flow does)
    // This ensures the server stays running with proper stdio inheritance
    const mcpServerPath = path.join(__dirname, '../mcp/agentdb-mcp-server.js');

    if (!fs.existsSync(mcpServerPath)) {
      log.error('MCP server not found. Please rebuild the package: npm run build');
      process.exit(1);
    }

    // Spawn server process with stdio inheritance
    const { spawn } = await import('child_process');
    const serverProcess = spawn('node', [mcpServerPath], {
      stdio: 'inherit',
      env: process.env
    });

    // Handle server exit
    serverProcess.on('exit', (code) => {
      process.exit(code || 0);
    });

    // Forward termination signals
    process.on('SIGINT', () => {
      serverProcess.kill('SIGINT');
    });

    process.on('SIGTERM', () => {
      serverProcess.kill('SIGTERM');
    });

    // Keep this process alive until server exits
    return new Promise(() => {
      // Never resolve - wait for server process
    });
  } else {
    log.error(`Unknown mcp subcommand: ${subcommand}`);
    log.info('Usage: agentdb mcp start');
    process.exit(1);
  }
}

async function handleCausalCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'add-edge') {
    await cli.causalAddEdge({
      cause: args[0],
      effect: args[1],
      uplift: parseFloat(args[2]),
      confidence: args[3] ? parseFloat(args[3]) : undefined,
      sampleSize: args[4] ? parseInt(args[4]) : undefined
    });
  } else if (subcommand === 'experiment' && args[0] === 'create') {
    await cli.causalExperimentCreate({
      name: args[1],
      cause: args[2],
      effect: args[3]
    });
  } else if (subcommand === 'experiment' && args[0] === 'add-observation') {
    await cli.causalExperimentAddObservation({
      experimentId: parseInt(args[1]),
      isTreatment: args[2] === 'true',
      outcome: parseFloat(args[3]),
      context: args[4]
    });
  } else if (subcommand === 'experiment' && args[0] === 'calculate') {
    await cli.causalExperimentCalculate(parseInt(args[1]));
  } else if (subcommand === 'query') {
    await cli.causalQuery({
      cause: args[0],
      effect: args[1],
      minConfidence: args[2] ? parseFloat(args[2]) : undefined,
      minUplift: args[3] ? parseFloat(args[3]) : undefined,
      limit: args[4] ? parseInt(args[4]) : undefined
    });
  } else {
    log.error(`Unknown causal subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleRecallCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'with-certificate') {
    await cli.recallWithCertificate({
      query: args[0],
      k: args[1] ? parseInt(args[1]) : undefined,
      alpha: args[2] ? parseFloat(args[2]) : undefined,
      beta: args[3] ? parseFloat(args[3]) : undefined,
      gamma: args[4] ? parseFloat(args[4]) : undefined
    });
  } else {
    log.error(`Unknown recall subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleLearnerCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'run') {
    await cli.learnerRun({
      minAttempts: args[0] ? parseInt(args[0]) : undefined,
      minSuccessRate: args[1] ? parseFloat(args[1]) : undefined,
      minConfidence: args[2] ? parseFloat(args[2]) : undefined,
      dryRun: args[3] === 'true'
    });
  } else if (subcommand === 'prune') {
    await cli.learnerPrune({
      minConfidence: args[0] ? parseFloat(args[0]) : undefined,
      minUplift: args[1] ? parseFloat(args[1]) : undefined,
      maxAgeDays: args[2] ? parseInt(args[2]) : undefined
    });
  } else {
    log.error(`Unknown learner subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleReflexionCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'store') {
    await cli.reflexionStoreEpisode({
      sessionId: args[0],
      task: args[1],
      reward: parseFloat(args[2]),
      success: args[3] === 'true',
      critique: args[4],
      input: args[5],
      output: args[6],
      latencyMs: args[7] ? parseInt(args[7]) : undefined,
      tokensUsed: args[8] ? parseInt(args[8]) : undefined
    });
  } else if (subcommand === 'retrieve') {
    // Parse retrieve command with new flags
    let task = args[0];
    let k: number | undefined = undefined;
    let minReward: number | undefined = undefined;
    let onlyFailures: boolean | undefined = undefined;
    let onlySuccesses: boolean | undefined = undefined;
    let synthesizeContext = false;
    let filters: any = {};

    for (let i = 1; i < args.length; i++) {
      const arg = args[i];
      if (arg === '--k' && i + 1 < args.length) {
        k = parseInt(args[++i]);
      } else if (arg === '--min-reward' && i + 1 < args.length) {
        minReward = parseFloat(args[++i]);
      } else if (arg === '--only-failures') {
        onlyFailures = true;
      } else if (arg === '--only-successes') {
        onlySuccesses = true;
      } else if (arg === '--synthesize-context') {
        synthesizeContext = true;
      } else if (arg === '--filters' && i + 1 < args.length) {
        try {
          filters = JSON.parse(args[++i]);
        } catch (error) {
          log.error(`Invalid JSON in --filters parameter: ${(error as Error).message}`);
          process.exit(1);
        }
      } else if (!arg.startsWith('--') && k === undefined) {
        // Legacy positional argument parsing
        k = parseInt(arg);
        if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
          minReward = parseFloat(args[++i]);
        }
        if (i + 1 < args.length && args[i + 1] === 'true') {
          onlyFailures = true;
          i++;
        }
        if (i + 1 < args.length && args[i + 1] === 'true') {
          onlySuccesses = true;
          i++;
        }
      }
    }

    await cli.reflexionRetrieve({
      task,
      k,
      minReward,
      onlyFailures,
      onlySuccesses,
      synthesizeContext,
      filters: Object.keys(filters).length > 0 ? filters : undefined
    });
  } else if (subcommand === 'critique-summary') {
    await cli.reflexionGetCritiqueSummary({
      task: args[0],
      k: args[1] ? parseInt(args[1]) : undefined
    });
  } else if (subcommand === 'prune') {
    await cli.reflexionPrune({
      maxAgeDays: args[0] ? parseInt(args[0]) : undefined,
      minReward: args[1] ? parseFloat(args[1]) : undefined
    });
  } else {
    log.error(`Unknown reflexion subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleSkillCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'create') {
    await cli.skillCreate({
      name: args[0],
      description: args[1],
      code: args[2]
    });
  } else if (subcommand === 'search') {
    await cli.skillSearch({
      task: args[0],
      k: args[1] ? parseInt(args[1]) : undefined
    });
  } else if (subcommand === 'consolidate') {
    await cli.skillConsolidate({
      minAttempts: args[0] ? parseInt(args[0]) : undefined,
      minReward: args[1] ? parseFloat(args[1]) : undefined,
      timeWindowDays: args[2] ? parseInt(args[2]) : undefined,
      extractPatterns: args[3] !== 'false' // Default true, set to false if explicitly passed
    });
  } else if (subcommand === 'prune') {
    await cli.skillPrune({
      minUses: args[0] ? parseInt(args[0]) : undefined,
      minSuccessRate: args[1] ? parseFloat(args[1]) : undefined,
      maxAgeDays: args[2] ? parseInt(args[2]) : undefined
    });
  } else {
    log.error(`Unknown skill subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleDbCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'stats') {
    await cli.dbStats();
  } else {
    log.error(`Unknown db subcommand: ${subcommand}`);
    printHelp();
  }
}

async function handleSyncCommands(cli: AgentDBCLI, subcommand: string, args: string[]) {
  if (subcommand === 'start-server') {
    // Parse options
    let port: number | undefined;
    let cert: string | undefined;
    let key: string | undefined;
    let authToken: string | undefined;
    let maxConnections: number | undefined;

    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--port' && i + 1 < args.length) {
        port = parseInt(args[++i]);
      } else if (args[i] === '--cert' && i + 1 < args.length) {
        cert = args[++i];
      } else if (args[i] === '--key' && i + 1 < args.length) {
        key = args[++i];
      } else if (args[i] === '--auth-token' && i + 1 < args.length) {
        authToken = args[++i];
      } else if (args[i] === '--max-connections' && i + 1 < args.length) {
        maxConnections = parseInt(args[++i]);
      }
    }

    await cli.quicStartServer({ port, cert, key, authToken, maxConnections });
  } else if (subcommand === 'connect') {
    // Parse host and port
    const host = args[0];
    const port = parseInt(args[1]);

    if (!host || !port) {
      log.error('Missing required arguments: host and port');
      log.info('Usage: agentdb sync connect <host> <port> [--auth-token <token>] [--cert <path>] [--timeout <ms>]');
      process.exit(1);
    }

    let authToken: string | undefined;
    let cert: string | undefined;
    let timeout: number | undefined;

    for (let i = 2; i < args.length; i++) {
      if (args[i] === '--auth-token' && i + 1 < args.length) {
        authToken = args[++i];
      } else if (args[i] === '--cert' && i + 1 < args.length) {
        cert = args[++i];
      } else if (args[i] === '--timeout' && i + 1 < args.length) {
        timeout = parseInt(args[++i]);
      }
    }

    await cli.quicConnect({ host, port, authToken, cert, timeout });
  } else if (subcommand === 'push') {
    // Parse options
    let server: string | undefined;
    let incremental = false;
    let filter: string | undefined;
    let authToken: string | undefined;
    let batchSize: number | undefined;

    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--server' && i + 1 < args.length) {
        server = args[++i];
      } else if (args[i] === '--incremental') {
        incremental = true;
      } else if (args[i] === '--filter' && i + 1 < args.length) {
        filter = args[++i];
      } else if (args[i] === '--auth-token' && i + 1 < args.length) {
        authToken = args[++i];
      } else if (args[i] === '--batch-size' && i + 1 < args.length) {
        batchSize = parseInt(args[++i]);
      } else if (!args[i].startsWith('--') && !server) {
        server = args[i];
      }
    }

    if (!server) {
      log.error('Missing required --server parameter');
      log.info('Usage: agentdb sync push --server <host:port> [--incremental] [--filter <pattern>] [--auth-token <token>] [--batch-size <n>]');
      process.exit(1);
    }

    await cli.quicPush({ server, incremental, filter, authToken, batchSize });
  } else if (subcommand === 'pull') {
    // Parse options
    let server: string | undefined;
    let incremental = false;
    let filter: string | undefined;
    let authToken: string | undefined;
    let batchSize: number | undefined;
    let conflictStrategy: 'local-wins' | 'remote-wins' | 'latest-wins' | undefined;

    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--server' && i + 1 < args.length) {
        server = args[++i];
      } else if (args[i] === '--incremental') {
        incremental = true;
      } else if (args[i] === '--filter' && i + 1 < args.length) {
        filter = args[++i];
      } else if (args[i] === '--auth-token' && i + 1 < args.length) {
        authToken = args[++i];
      } else if (args[i] === '--batch-size' && i + 1 < args.length) {
        batchSize = parseInt(args[++i]);
      } else if (args[i] === '--conflict-strategy' && i + 1 < args.length) {
        const strategy = args[++i];
        if (strategy === 'local-wins' || strategy === 'remote-wins' || strategy === 'latest-wins') {
          conflictStrategy = strategy;
        } else {
          log.warning(`Unknown conflict strategy: ${strategy}. Using default: latest-wins`);
        }
      } else if (!args[i].startsWith('--') && !server) {
        server = args[i];
      }
    }

    if (!server) {
      log.error('Missing required --server parameter');
      log.info('Usage: agentdb sync pull --server <host:port> [--incremental] [--filter <pattern>] [--auth-token <token>] [--batch-size <n>] [--conflict-strategy <local-wins|remote-wins|latest-wins>]');
      process.exit(1);
    }

    await cli.quicPull({ server, incremental, filter, authToken, batchSize, conflictStrategy });
  } else if (subcommand === 'status') {
    await cli.quicStatus();
  } else {
    log.error(`Unknown sync subcommand: ${subcommand}`);
    log.info('Available subcommands: start-server, connect, push, pull, status');
    log.info('');
    log.info('Examples:');
    log.info('  agentdb sync start-server --port 4433 --auth-token secret123');
    log.info('  agentdb sync connect localhost 4433 --auth-token secret123');
    log.info('  agentdb sync push --server localhost:4433 --incremental');
    log.info('  agentdb sync pull --server localhost:4433 --conflict-strategy latest-wins');
    log.info('  agentdb sync status');
    printHelp();
  }
}

// Query command - semantic search across reflexion episodes with context synthesis
async function handleQueryCommand(cli: AgentDBCLI, args: string[]) {
  // Parse command-line arguments
  let domain = '';
  let query = '';
  let k = 5;
  let minConfidence = 0.0;
  let format = 'json';
  let synthesizeContext = false;
  let filters: any = {};

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--domain' && i + 1 < args.length) {
      domain = args[++i];
    } else if (args[i] === '--query' && i + 1 < args.length) {
      query = args[++i];
    } else if (args[i] === '--k' && i + 1 < args.length) {
      k = parseInt(args[++i]);
    } else if (args[i] === '--min-confidence' && i + 1 < args.length) {
      minConfidence = parseFloat(args[++i]);
    } else if (args[i] === '--format' && i + 1 < args.length) {
      format = args[++i];
    } else if (args[i] === '--synthesize-context') {
      synthesizeContext = true;
    } else if (args[i] === '--filters' && i + 1 < args.length) {
      try {
        filters = JSON.parse(args[++i]);
      } catch (error) {
        log.error(`Invalid JSON in --filters parameter: ${(error as Error).message}`);
        process.exit(1);
      }
    }
  }

  if (!query) {
    log.error('Missing required --query parameter');
    process.exit(1);
  }

  // Validate filters if provided
  if (Object.keys(filters).length > 0) {
    const validation = MetadataFilter.validate(filters);
    if (!validation.valid) {
      log.error('Invalid filters:');
      validation.errors.forEach(err => log.error(`  - ${err}`));
      process.exit(1);
    }
  }

  // Use reflexionRetrieveJson method for JSON output
  let results = await cli.reflexionRetrieveJson({
    task: query,
    k,
    minReward: minConfidence,
    onlySuccesses: domain === 'successful-edits'
  });

  // Apply metadata filters if provided
  if (Object.keys(filters).length > 0) {
    results = MetadataFilter.apply(results, filters);
    log.info(`Filtered to ${results.length} results matching metadata criteria`);
  }

  // Synthesize context if requested
  if (synthesizeContext && results.length > 0) {
    const context = ContextSynthesizer.synthesize(results.map((r: any) => ({
      task: r.task,
      reward: r.reward,
      success: r.success,
      critique: r.critique,
      input: r.input,
      output: r.output,
      similarity: r.similarity
    })));

    if (format === 'json') {
      console.log(JSON.stringify({ results, synthesizedContext: context }, null, 2));
    } else {
      console.log('\n' + '═'.repeat(80));
      console.log('SYNTHESIZED CONTEXT');
      console.log('═'.repeat(80));
      console.log(`\n${context.summary}\n`);

      if (context.patterns.length > 0) {
        console.log('Common Patterns:');
        context.patterns.forEach(p => console.log(`  • ${p}`));
        console.log('');
      }

      if (context.keyInsights.length > 0) {
        console.log('Key Insights:');
        context.keyInsights.forEach(i => console.log(`  • ${i}`));
        console.log('');
      }

      if (context.recommendations.length > 0) {
        console.log('Recommendations:');
        context.recommendations.forEach((r, i) => console.log(`  ${i + 1}. ${r}`));
        console.log('');
      }

      console.log('═'.repeat(80));
      console.log(`\nMatched ${results.length} memories\n`);
    }
  } else {
    if (format === 'json') {
      console.log(JSON.stringify(results, null, 2));
    } else {
      console.log(results);
    }
  }
}

// Store-pattern command - store learned patterns
async function handleStorePatternCommand(cli: AgentDBCLI, args: string[]) {
  let type = '';
  let domain = '';
  let pattern = '';
  let confidence = 0.5;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--type' && i + 1 < args.length) {
      type = args[++i];
    } else if (args[i] === '--domain' && i + 1 < args.length) {
      domain = args[++i];
    } else if (args[i] === '--pattern' && i + 1 < args.length) {
      pattern = args[++i];
    } else if (args[i] === '--confidence' && i + 1 < args.length) {
      confidence = parseFloat(args[++i]);
    }
  }

  if (!pattern) {
    log.error('Missing required --pattern parameter');
    process.exit(1);
  }

  // Parse pattern JSON
  let patternData;
  try {
    patternData = JSON.parse(pattern);
  } catch (error) {
    log.error('Invalid JSON in --pattern parameter');
    process.exit(1);
  }

  // Store as reflexion episode with pattern metadata
  const sessionId = `pattern-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  await cli.reflexionStoreEpisode({
    sessionId,
    task: `${type}:${domain}`,
    reward: confidence,
    success: true,
    critique: JSON.stringify({
      type,
      domain,
      pattern: patternData,
      storedAt: Date.now()
    })
  });

  console.log(JSON.stringify({ success: true, sessionId }, null, 2));
}

// Train command - trigger pattern learning
async function handleTrainCommand(cli: AgentDBCLI, args: string[]) {
  let domain = '';
  let epochs = 10;
  let batchSize = 32;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--domain' && i + 1 < args.length) {
      domain = args[++i];
    } else if (args[i] === '--epochs' && i + 1 < args.length) {
      epochs = parseInt(args[++i]);
    } else if (args[i] === '--batch-size' && i + 1 < args.length) {
      batchSize = parseInt(args[++i]);
    }
  }

  // Run learner to discover causal patterns
  log.info(`Training on domain: ${domain || 'all'} (${epochs} epochs, batch size: ${batchSize})`);

  await cli.learnerRun({
    minAttempts: 3,
    minSuccessRate: 0.6,
    minConfidence: 0.7,
    dryRun: false
  });

  // Also consolidate skills from successful episodes
  await cli.skillConsolidate({
    minAttempts: 3,
    minReward: 0.7,
    timeWindowDays: 7,
    extractPatterns: true
  });

  console.log(JSON.stringify({ success: true, message: 'Training completed' }, null, 2));
}

// Optimize-memory command - memory consolidation and compression
async function handleOptimizeMemoryCommand(cli: AgentDBCLI, args: string[]) {
  let compress = true;
  let consolidatePatterns = true;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--compress' && i + 1 < args.length) {
      compress = args[++i] === 'true';
    } else if (args[i] === '--consolidate-patterns' && i + 1 < args.length) {
      consolidatePatterns = args[++i] === 'true';
    }
  }

  log.info('Optimizing memory...');

  // Prune old/low-quality episodes
  if (compress) {
    await cli.reflexionPrune({
      maxAgeDays: 90,
      minReward: 0.3
    });
  }

  // Consolidate patterns into skills
  if (consolidatePatterns) {
    await cli.skillConsolidate({
      minAttempts: 3,
      minReward: 0.7,
      timeWindowDays: 7,
      extractPatterns: true
    });
  }

  // Prune underperforming skills
  await cli.skillPrune({
    minUses: 3,
    minSuccessRate: 0.4,
    maxAgeDays: 60
  });

  // Prune low-confidence causal edges
  await cli.learnerPrune({
    minConfidence: 0.5,
    minUplift: 0.05,
    maxAgeDays: 90
  });

  console.log(JSON.stringify({ success: true, message: 'Memory optimization completed' }, null, 2));
}

// Vector-search command - direct vector similarity search with MMR diversity ranking
async function handleVectorSearchCommand(args: string[]) {
  // Parse arguments
  let dbPath = './agentdb.db';
  let vector: number[] = [];
  let k = 10;
  let threshold = 0.0;
  let metric = 'cosine';
  let format = 'json';
  let verbose = false;
  let useMmr = false;
  let mmrLambda = 0.5;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-k' && i + 1 < args.length) {
      k = parseInt(args[++i]);
    } else if (arg === '-t' && i + 1 < args.length) {
      threshold = parseFloat(args[++i]);
    } else if (arg === '-m' && i + 1 < args.length) {
      metric = args[++i];
    } else if (arg === '-f' && i + 1 < args.length) {
      format = args[++i];
    } else if (arg === '-v' || arg === '--verbose') {
      verbose = true;
    } else if (arg === '--mmr') {
      useMmr = true;
      if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
        mmrLambda = parseFloat(args[++i]);
      }
    } else if (!dbPath.endsWith('.db') && !arg.startsWith('[')) {
      dbPath = arg;
    } else if (arg.startsWith('[') || (!isNaN(parseFloat(arg)))) {
      // Parse vector - either JSON array or space-separated numbers
      try {
        if (arg.startsWith('[')) {
          vector = JSON.parse(arg);
        } else {
          // Collect space-separated numbers
          while (i < args.length && !isNaN(parseFloat(args[i]))) {
            vector.push(parseFloat(args[i++]));
          }
          i--; // Back up one since loop will increment
        }
      } catch (e) {
        log.error('Invalid vector format. Use JSON array: "[0.1,0.2,0.3]" or space-separated: "0.1 0.2 0.3"');
        process.exit(1);
      }
    }
  }

  if (vector.length === 0) {
    log.error('Missing required vector parameter');
    log.info('Usage: agentdb vector-search <db-path> <vector> [-k 10] [-t 0.75] [-m cosine] [-f json] [-v] [--mmr [lambda]]');
    process.exit(1);
  }

  if (useMmr) {
    log.info(`Using MMR diversity ranking (λ=${mmrLambda})`);
  }

  // Initialize database
  const cli = new AgentDBCLI();
  await cli.initialize(dbPath);

  // Perform vector search using reflexion's retrieveRelevant (which uses embeddings)
  // We'll need to search episode_embeddings table directly
  const query = `
    SELECT
      e.id,
      e.session_id,
      e.task,
      e.reward,
      e.success,
      ee.embedding
    FROM episodes e
    JOIN episode_embeddings ee ON e.id = ee.episode_id
    LIMIT ?
  `;

  const results = cli.db.prepare(query).all(k * 10); // Get more for filtering

  // Calculate similarities
  let scored = results.map((row: any) => {
    const embedding = new Float32Array(row.embedding);
    const similarity = calculateSimilarity(vector, Array.from(embedding), metric);
    return {
      id: row.id,
      session_id: row.session_id,
      task: row.task,
      reward: row.reward,
      success: row.success,
      similarity,
      embedding: Array.from(embedding)
    };
  }).filter(r => r.similarity >= threshold)
    .sort((a, b) => b.similarity - a.similarity);

  // Apply MMR diversity ranking if requested
  if (useMmr && scored.length > k) {
    scored = MMRDiversityRanker.selectDiverse(scored, vector, {
      lambda: mmrLambda,
      k,
      metric: metric as 'cosine' | 'euclidean' | 'dot'
    });
  } else {
    scored = scored.slice(0, k);
  }

  // Remove embedding from output
  const output = scored.map(({ embedding, ...rest }) => rest);

  if (format === 'json') {
    console.log(JSON.stringify(output, null, 2));
  } else {
    console.log(`Found ${output.length} results:`);
    output.forEach((r, i) => {
      console.log(`${i + 1}. [${r.id}] ${r.task} (similarity: ${r.similarity.toFixed(4)})`);
      if (verbose) {
        console.log(`   Session: ${r.session_id}, Reward: ${r.reward}, Success: ${r.success}`);
      }
    });
  }
}

// Helper function to calculate similarity
function calculateSimilarity(v1: number[], v2: number[], metric: string): number {
  if (v1.length !== v2.length) {
    throw new Error(`Vector dimension mismatch: ${v1.length} vs ${v2.length}`);
  }

  if (metric === 'cosine') {
    let dot = 0, mag1 = 0, mag2 = 0;
    for (let i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
      mag1 += v1[i] * v1[i];
      mag2 += v2[i] * v2[i];
    }
    return dot / (Math.sqrt(mag1) * Math.sqrt(mag2));
  } else if (metric === 'euclidean') {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
      const diff = v1[i] - v2[i];
      sum += diff * diff;
    }
    return 1 / (1 + Math.sqrt(sum)); // Normalize to 0-1 range
  } else if (metric === 'dot') {
    let dot = 0;
    for (let i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
    }
    return dot;
  }

  return 0;
}

// Export command - export vectors to JSON with optional compression
async function handleExportCommand(args: string[]) {
  let dbPath = './agentdb.db';
  let outputPath = './agentdb-export.json';
  let compress = false;

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--compress') {
      compress = true;
    } else if (arg === '--output' && i + 1 < args.length) {
      outputPath = args[++i];
    } else if (!arg.startsWith('--')) {
      if (dbPath === './agentdb.db') {
        dbPath = arg;
      } else if (outputPath === './agentdb-export.json') {
        outputPath = arg;
      }
    }
  }

  // Add .gz extension if compressing and not already present
  if (compress && !outputPath.endsWith('.gz')) {
    outputPath += '.gz';
  }

  log.info(`Exporting vectors from: ${dbPath}`);
  if (compress) {
    log.info('Compression: enabled');
  }

  const cli = new AgentDBCLI();
  await cli.initialize(dbPath);

  // Export all episodes with embeddings
  const query = `
    SELECT
      e.*,
      ee.embedding
    FROM episodes e
    LEFT JOIN episode_embeddings ee ON e.id = ee.episode_id
  `;

  const results = cli.db.prepare(query).all();

  const jsonData = JSON.stringify(results, null, 2);

  try {
    if (compress) {
      // Compress with gzip
      const compressed = zlib.gzipSync(jsonData);
      fs.writeFileSync(outputPath, compressed);
      const originalSize = Buffer.byteLength(jsonData);
      const compressedSize = compressed.length;
      const ratio = ((1 - compressedSize / originalSize) * 100).toFixed(1);
      log.success(`Exported ${results.length} episodes to ${outputPath}`);
      log.info(`Original size: ${(originalSize / 1024).toFixed(2)} KB`);
      log.info(`Compressed size: ${(compressedSize / 1024).toFixed(2)} KB (${ratio}% reduction)`);
    } else {
      fs.writeFileSync(outputPath, jsonData);
      log.success(`Exported ${results.length} episodes to ${outputPath}`);
    }
  } catch (error) {
    log.error(`Failed to export: ${(error as Error).message}`);
    process.exit(1);
  }
}

// Import command - import vectors from JSON with optional decompression
async function handleImportCommand(args: string[]) {
  let inputPath = '';
  let dbPath = './agentdb.db';
  let decompress = false;

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--decompress') {
      decompress = true;
    } else if (arg === '--db' && i + 1 < args.length) {
      dbPath = args[++i];
    } else if (!arg.startsWith('--')) {
      if (!inputPath) {
        inputPath = arg;
      } else if (dbPath === './agentdb.db') {
        dbPath = arg;
      }
    }
  }

  // Auto-detect compression from .gz extension
  if (inputPath.endsWith('.gz')) {
    decompress = true;
  }

  if (!inputPath) {
    log.error('Missing required input file');
    log.info('Usage: agentdb import <input-file.json> [db-path] [--decompress]');
    process.exit(1);
  }

  log.info(`Importing vectors from: ${inputPath}`);
  if (decompress) {
    log.info('Decompression: enabled');
  }

  let data: any[];

  try {
    if (decompress) {
      // Decompress with gunzip
      const compressed = fs.readFileSync(inputPath);
      const decompressed = zlib.gunzipSync(compressed);
      data = JSON.parse(decompressed.toString('utf-8'));
      log.info(`Decompressed ${(compressed.length / 1024).toFixed(2)} KB to ${(decompressed.length / 1024).toFixed(2)} KB`);
    } else {
      data = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));
    }
  } catch (error) {
    log.error(`Failed to read/parse input file: ${(error as Error).message}`);
    process.exit(1);
  }

  const cli = new AgentDBCLI();
  await cli.initialize(dbPath);

  let imported = 0;
  for (const item of data) {
    try {
      // Import episode
      const episodeQuery = `
        INSERT INTO episodes (session_id, task, input, output, critique, reward, success, latency_ms, tokens_used, tags, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `;

      const result = cli.db.prepare(episodeQuery).run(
        item.session_id,
        item.task,
        item.input,
        item.output,
        item.critique,
        item.reward,
        item.success,
        item.latency_ms,
        item.tokens_used,
        item.tags,
        item.metadata
      );

      // Import embedding if exists
      if (item.embedding) {
        const embeddingQuery = `INSERT INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)`;
        cli.db.prepare(embeddingQuery).run(result.lastInsertRowid, item.embedding);
      }

      imported++;
    } catch (error) {
      log.warning(`Failed to import item ${imported + 1}: ${(error as Error).message}`);
    }
  }

  // Save database
  if (cli.db && typeof cli.db.save === 'function') {
    cli.db.save();
  }

  log.success(`Imported ${imported} episodes`);
}

// Stats command - show database statistics
async function handleStatsCommand(args: string[]) {
  const dbPath = args[0] || './agentdb.db';

  log.info(`Getting statistics for: ${dbPath}`);

  const cli = new AgentDBCLI();
  await cli.initialize(dbPath);

  // Get counts (with fallback for missing tables)
  const episodeCount = cli.db.prepare('SELECT COUNT(*) as count FROM episodes').get()?.count || 0;
  const embeddingCount = cli.db.prepare('SELECT COUNT(*) as count FROM episode_embeddings').get()?.count || 0;

  let skillCount = 0;
  try {
    skillCount = cli.db.prepare('SELECT COUNT(*) as count FROM skill_library').get()?.count || 0;
  } catch (e) {
    // skill_library table may not exist in older databases
  }

  let causalEdges = 0;
  try {
    causalEdges = cli.db.prepare('SELECT COUNT(*) as count FROM causal_edges').get()?.count || 0;
  } catch (e) {
    // causal_edges table may not exist in older databases
  }

  // Get database file size
  let dbSize = 0;
  if (dbPath !== ':memory:' && fs.existsSync(dbPath)) {
    dbSize = fs.statSync(dbPath).size;
  }

  // Get average confidence
  const avgConfidence = cli.db.prepare('SELECT AVG(reward) as avg FROM episodes').get().avg || 0;

  // Get domains
  const domains = cli.db.prepare(`
    SELECT task, COUNT(*) as count
    FROM episodes
    GROUP BY task
    ORDER BY count DESC
    LIMIT 5
  `).all();

  console.log(`
📊 AgentDB Statistics

Database: ${dbPath}
Size: ${(dbSize / 1024).toFixed(2)} KB

📈 Counts:
  Episodes: ${episodeCount}
  Embeddings: ${embeddingCount}
  Skills: ${skillCount}
  Causal Edges: ${causalEdges}

📊 Metrics:
  Average Reward: ${avgConfidence.toFixed(3)}
  Embedding Coverage: ${episodeCount > 0 ? ((embeddingCount / episodeCount) * 100).toFixed(1) : 0}%

🏷️  Top Domains:
${domains.map(d => `  • ${d.task}: ${d.count}`).join('\n')}
  `);
}

function printHelp() {
  console.log(`
${colors.bright}${colors.cyan}█▀█ █▀▀ █▀▀ █▄░█ ▀█▀ █▀▄ █▄▄
█▀█ █▄█ ██▄ █░▀█ ░█░ █▄▀ █▄█${colors.reset}

${colors.bright}${colors.cyan}AgentDB v2 CLI - Vector Intelligence with Auto Backend Detection${colors.reset}

${colors.bright}CORE COMMANDS:${colors.reset}
  ${colors.cyan}init${colors.reset} [options]              Initialize database with backend detection
    --backend <type>           Backend: auto (default), ruvector, hnswlib
    --dimension <n>            Vector dimension (default: 384)
    --model <name>             Embedding model (default: Xenova/all-MiniLM-L6-v2)
                               Popular: Xenova/bge-base-en-v1.5 (768d production)
                                        Xenova/bge-small-en-v1.5 (384d best quality)
    --dry-run                  Show detection info without initializing
    --db <path>                Database path (default: ./agentdb.db)

  ${colors.cyan}status${colors.reset} [options]            Show database and backend status
    --db <path>                Database path (default: ./agentdb.db)
    --verbose, -v              Show detailed statistics

  ${colors.cyan}doctor${colors.reset} [options]            System diagnostics and health check
    --db <path>                Database path to check (optional)
    --verbose, -v              Show detailed system information

${colors.bright}USAGE:${colors.reset}
  agentdb <command> <subcommand> [options]

${colors.bright}SETUP COMMANDS:${colors.reset}
  agentdb init [db-path] [--dimension 384] [--model <name>] [--preset small|medium|large] [--in-memory]
    Initialize a new AgentDB database (default: ./agentdb.db)
    Options:
      --dimension <n>     Vector dimension (default: 384 for all-MiniLM, 768 for bge-base)
      --model <name>      Embedding model (default: Xenova/all-MiniLM-L6-v2)
                          Examples:
                            Xenova/bge-small-en-v1.5 (384d) - Best quality at 384-dim
                            Xenova/bge-base-en-v1.5 (768d)  - Production quality
                            Xenova/all-mpnet-base-v2 (768d) - All-around excellence
                          See: docs/EMBEDDING-MODELS-GUIDE.md for full list
      --preset <size>     small (<10K), medium (10K-100K), large (>100K vectors)
      --in-memory         Use temporary in-memory database (:memory:)

  agentdb install-embeddings [--global]
    Install optional embedding dependencies (@huggingface/transformers)
    By default uses mock embeddings - run this for real ML-powered embeddings
    Options:
      --global, -g        Install globally instead of locally
    Note: Requires build tools (python3, make, g++)

  agentdb migrate <source-db> [--target <target-db>] [--no-optimize] [--dry-run] [-v]
    Migrate legacy AgentDB v1 or claude-flow memory databases to v2 format
    Automatically detects source type and optimizes for RuVector GNN
    Options:
      --target <path>     Target database path (default: source-v2.db)
      --no-optimize       Skip GNN optimization analysis
      --dry-run           Analyze migration without making changes
      --verbose, -v       Show detailed migration progress

${colors.bright}VECTOR SEARCH COMMANDS:${colors.reset}
  agentdb vector-search <db-path> <vector> [-k 10] [-t 0.75] [-m cosine] [-f json] [-v] [--mmr [lambda]]
    Direct vector similarity search without text embeddings
    Arguments:
      <db-path>          Database file path (or :memory:)
      <vector>           Vector as JSON array [0.1,0.2,...] or space-separated numbers
    Options:
      -k <n>             Number of results (default: 10)
      -t <threshold>     Minimum similarity threshold (default: 0.0)
      -m <metric>        Similarity metric: cosine|euclidean|dot (default: cosine)
      -f <format>        Output format: json|table (default: json)
      -v                 Verbose mode with similarity scores
      --mmr [lambda]     Enable MMR diversity ranking (lambda: 0-1, default: 0.5)
                         0 = max diversity, 1 = max relevance
    Example: agentdb vector-search ./vectors.db "[0.1,0.2,0.3]" -k 10 -m cosine
    Example: agentdb vector-search ./vectors.db "[0.1,0.2,0.3]" --mmr 0.7

  agentdb export <db-path> [output-file] [--compress]
    Export all vectors and episodes to JSON file
    Options:
      --compress         Compress output with gzip (adds .gz extension)
      --output <file>    Output file path
    Example: agentdb export ./agentdb.db ./backup.json
    Example: agentdb export ./agentdb.db --compress --output backup.json.gz

  agentdb import <input-file> [db-path] [--decompress]
    Import vectors and episodes from JSON file
    Options:
      --decompress       Decompress gzip input (auto-detected for .gz files)
      --db <path>        Database file path
    Example: agentdb import ./backup.json ./new-db.db
    Example: agentdb import ./backup.json.gz --decompress

  agentdb stats [db-path]
    Show detailed database statistics and metrics
    Example: agentdb stats ./agentdb.db

${colors.bright}MCP COMMANDS:${colors.reset}
  agentdb mcp start
    Start the MCP server for Claude Desktop integration

${colors.bright}QUIC SYNC COMMANDS:${colors.reset}
  agentdb sync start-server [--port 4433] [--cert <path>] [--key <path>] [--auth-token <token>]
    Start a QUIC synchronization server for multi-agent coordination
    Options:
      --port <n>           Server port (default: 4433)
      --cert <path>        TLS certificate file path
      --key <path>         TLS key file path
      --auth-token <token> Authentication token (auto-generated if not provided)
    Example: agentdb sync start-server --port 4433 --cert ./cert.pem --key ./key.pem

  agentdb sync connect <host> <port> [--auth-token <token>] [--cert <path>]
    Connect to a remote QUIC sync server
    Arguments:
      <host>              Remote server hostname or IP
      <port>              Remote server port
    Options:
      --auth-token <token> Authentication token
      --cert <path>        TLS certificate for verification
    Example: agentdb sync connect 192.168.1.100 4433 --auth-token abc123

  agentdb sync push --server <host:port> [--incremental] [--filter <pattern>]
    Push local changes to remote server
    Options:
      --server <host:port> Remote server address (e.g., 192.168.1.100:4433)
      --incremental        Only push changes since last sync
      --filter <pattern>   Filter changes by pattern (e.g., "episodes", "skills")
    Example: agentdb sync push --server 192.168.1.100:4433 --incremental
    Example: agentdb sync push --server localhost:4433 --filter "episodes"

  agentdb sync pull --server <host:port> [--incremental] [--filter <pattern>]
    Pull remote changes from server
    Options:
      --server <host:port> Remote server address (e.g., 192.168.1.100:4433)
      --incremental        Only pull changes since last sync
      --filter <pattern>   Filter changes by pattern (e.g., "skills", "causal_edges")
    Example: agentdb sync pull --server 192.168.1.100:4433 --incremental
    Example: agentdb sync pull --server localhost:4433 --filter "skills"

  agentdb sync status
    Show synchronization status, pending changes, and connected servers
    Example: agentdb sync status

${colors.bright}ADVANCED NEURAL COMMANDS (WASM-ACCELERATED):${colors.reset}
  agentdb attention <subcommand> [options]
    Attention mechanism operations: flash, hyperbolic, sparse, MoE
    Subcommands:
      init --mechanism <type>         Initialize attention configuration
      compute --query <text>          Compute attention for query-key-value
      benchmark --all                 Benchmark all mechanisms
      optimize --mechanism <type>     Optimize parameters
    Example: agentdb attention benchmark --all --iterations 100

  agentdb learn --mode <type> --data <file> [options]
    Advanced learning: curriculum, contrastive loss, hard negative mining
    Modes:
      curriculum    Progressive difficulty training with cosine/linear schedules
      contrastive   InfoNCE and local contrastive loss training
      hard-negatives Hard negative mining for contrastive learning
    Options:
      --schedule <type>        Difficulty schedule: linear|cosine|exponential
      --temperature <n>        InfoNCE temperature (default: 0.07)
      --strategy <type>        Mining strategy: hard|semi-hard|distance-based
    Example: agentdb learn --mode curriculum --data train.json --schedule cosine
    Example: agentdb learn --mode contrastive --data pairs.json --epochs 15

  agentdb route --prompt <text> [options]
    LLM routing with FastGRNN model selection (haiku/sonnet/opus)
    Options:
      --prompt <text>          Prompt to route
      --prompt-file <path>     File containing prompt
      --context <json>         Conversation context (JSON)
      --explain                Explain routing decision
    Subcommands:
      feedback --model <type> --outcome <result>    Record feedback for learning
      stats                                          View routing statistics
    Example: agentdb route --prompt "Explain quantum computing" --explain
    Example: agentdb route feedback --model sonnet --outcome success

  agentdb hyperbolic --op <type> [options]
    Hyperbolic space operations: Poincaré ball geometry
    Operations:
      expmap       Exponential map (tangent -> manifold)
      logmap       Logarithmic map (manifold -> tangent)
      mobius-add   Möbius addition (hyperbolic addition)
      distance     Poincaré distance between points
      project      Project point to Poincaré ball
      centroid     Compute hyperbolic centroid
      dual-search  Hybrid Euclidean + Hyperbolic search
    Example: agentdb hyperbolic --op distance --point-a "[0.3,0.4]" --point-b "[0.6,0.2]"
    Example: agentdb hyperbolic --op dual-search --query "[0.5,0.5]" --points vectors.json

${colors.bright}CAUSAL COMMANDS:${colors.reset}
  agentdb causal add-edge <cause> <effect> <uplift> [confidence] [sample-size]
    Add a causal edge manually

  agentdb causal experiment create <name> <cause> <effect>
    Create a new A/B experiment

  agentdb causal experiment add-observation <experiment-id> <is-treatment> <outcome> [context]
    Record an observation (is-treatment: true/false)

  agentdb causal experiment calculate <experiment-id>
    Calculate uplift and statistical significance

  agentdb causal query [cause] [effect] [min-confidence] [min-uplift] [limit]
    Query causal edges with filters

${colors.bright}RECALL COMMANDS:${colors.reset}
  agentdb recall with-certificate <query> [k] [alpha] [beta] [gamma]
    Retrieve episodes with causal utility and provenance certificate
    Defaults: k=12, alpha=0.7, beta=0.2, gamma=0.1

${colors.bright}LEARNER COMMANDS:${colors.reset}
  agentdb learner run [min-attempts] [min-success-rate] [min-confidence] [dry-run]
    Discover causal edges from episode patterns
    Defaults: min-attempts=3, min-success-rate=0.6, min-confidence=0.7

  agentdb learner prune [min-confidence] [min-uplift] [max-age-days]
    Remove low-quality or old causal edges
    Defaults: min-confidence=0.5, min-uplift=0.05, max-age-days=90

${colors.bright}REFLEXION COMMANDS:${colors.reset}
  agentdb reflexion store <session-id> <task> <reward> <success> [critique] [input] [output] [latency-ms] [tokens]
    Store episode with self-critique

  agentdb reflexion retrieve <task> [--k <n>] [--min-reward <r>] [--only-failures] [--only-successes] [--synthesize-context] [--filters <json>]
    Retrieve relevant past episodes
    Options:
      --k <n>                Number of results (default: 5)
      --min-reward <r>       Minimum reward threshold
      --only-failures        Return only failed episodes
      --only-successes       Return only successful episodes
      --synthesize-context   Generate coherent summary with patterns and insights
      --filters <json>       MongoDB-style metadata filters (e.g., '{"metadata.year":{"$gte":2024}}')
    Example: agentdb reflexion retrieve "authentication" --k 10 --synthesize-context
    Example: agentdb reflexion retrieve "bug-fix" --filters '{"success":true,"reward":{"$gte":0.8}}'

  agentdb reflexion critique-summary <task> [only-failures]
    Get aggregated critique lessons

  agentdb reflexion prune [max-age-days] [max-reward]
    Clean up old or low-value episodes

${colors.bright}SKILL COMMANDS:${colors.reset}
  agentdb skill create <name> <description> [code]
    Create a reusable skill

  agentdb skill search <query> [k]
    Find applicable skills by similarity

  agentdb skill consolidate [min-attempts] [min-reward] [time-window-days] [extract-patterns]
    Auto-create skills from successful episodes with ML pattern extraction
    Defaults: min-attempts=3, min-reward=0.7, time-window-days=7, extract-patterns=true
    Analyzes: keyword frequency, critique patterns, reward distribution, metadata, learning curves

  agentdb skill prune [min-uses] [min-success-rate] [max-age-days]
    Remove underperforming skills (defaults: 3, 0.4, 60)

${colors.bright}DATABASE COMMANDS:${colors.reset}
  agentdb db stats
    Show database statistics

${colors.bright}HOOKS INTEGRATION COMMANDS:${colors.reset}
  agentdb query --query <query> [--domain <domain>] [--k <k>] [--min-confidence <conf>] [--format json] [--synthesize-context] [--filters <json>]
    Semantic search across stored episodes and patterns
    Options:
      --query <q>            Query string (required)
      --domain <d>           Domain filter (e.g., "successful-edits")
      --k <n>                Number of results (default: 5)
      --min-confidence <c>   Minimum confidence threshold (default: 0.0)
      --format <f>           Output format: json|text (default: json)
      --synthesize-context   Generate coherent summary with patterns and insights
      --filters <json>       MongoDB-style metadata filters
    Example: agentdb query --query "authentication" --k 5 --min-confidence 0.8 --synthesize-context
    Example: agentdb query --query "bug-fix" --filters '{"metadata.priority":"high"}' --synthesize-context

  agentdb store-pattern --type <type> --domain <domain> --pattern <json> --confidence <conf>
    Store a learned pattern for future retrieval
    Example: agentdb store-pattern --type "experience" --domain "code-edits" --pattern '{"success":true}' --confidence 0.9

  agentdb train --domain <domain> --epochs <n> --batch-size <n>
    Trigger pattern learning and skill consolidation
    Example: agentdb train --domain "code-edits" --epochs 10 --batch-size 32

  agentdb optimize-memory --compress <bool> --consolidate-patterns <bool>
    Memory consolidation, compression, and cleanup
    Example: agentdb optimize-memory --compress true --consolidate-patterns true

${colors.bright}ENVIRONMENT:${colors.reset}
  AGENTDB_PATH    Database file path (default: ./agentdb.db)

${colors.bright}EXAMPLES:${colors.reset}
  # QUIC Sync: Multi-agent coordination
  # On server machine:
  agentdb sync start-server --port 4433 --auth-token secret123

  # On client machines:
  agentdb sync connect 192.168.1.100 4433 --auth-token secret123
  agentdb sync push --server 192.168.1.100:4433 --incremental
  agentdb sync pull --server 192.168.1.100:4433 --incremental
  agentdb sync status

  # Vector Search: Direct similarity queries
  agentdb init ./vectors.db --dimension 768 --preset medium
  agentdb vector-search ./vectors.db "[0.1,0.2,0.3]" -k 10 -m cosine -f json
  agentdb export ./vectors.db ./backup.json
  agentdb import ./backup.json ./new-vectors.db
  agentdb stats ./vectors.db

  # Reflexion: Store and retrieve episodes
  agentdb reflexion store "session-1" "implement_auth" 0.95 true "Used OAuth2"
  agentdb reflexion retrieve "authentication" 10 0.8
  agentdb reflexion critique-summary "bug_fix" true

  # Skills: Create and search
  agentdb skill create "jwt_auth" "Generate JWT tokens" "code here..."
  agentdb skill search "authentication" 5
  agentdb skill consolidate 3 0.7 7 true  # With pattern extraction
  agentdb skill consolidate 5 0.8 14      # Higher thresholds, longer window

  # Causal: Add edges and run experiments
  agentdb causal add-edge "add_tests" "code_quality" 0.25 0.95 100
  agentdb causal experiment create "test-coverage-quality" "test_coverage" "bug_rate"
  agentdb causal experiment add-observation 1 true 0.15
  agentdb causal experiment calculate 1

  # Retrieve with causal utility
  agentdb recall with-certificate "implement authentication" 10

  # Discover patterns automatically
  agentdb learner run 3 0.6 0.7

  # Get database stats
  agentdb db stats
`);
}

// ESM entry point check - run if this is the main module
// Handle both direct execution and npx/symlink scenarios
const isMainModule = import.meta.url === `file://${process.argv[1]}` ||
                     import.meta.url.endsWith('/agentdb-cli.js');

// Run main() when executed as main module (removed argv length check to allow no-args help)
if (isMainModule) {
  main()
    .then(() => {
      // Force immediate exit to avoid onnxruntime cleanup crash
      process.exit(0);
    })
    .catch((error) => {
      console.error('AgentDB CLI Error:', error);
      process.exit(1);
    });
}

export { AgentDBCLI };
