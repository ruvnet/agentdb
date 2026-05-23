/**
 * Report viewer and management
 * View, list, and export simulation reports
 */

import chalk from 'chalk';
import type { SimulationReport } from '../lib/simulation-runner.js';

export interface ReportMetadata {
  id: string;
  scenarioId: string;
  timestamp: string;
  path: string;
  format: string;
  coherenceScore: number;
  successRate: number;
}

export async function viewReport(reportId?: string): Promise<void> {
  if (!reportId) {
    await listReports();
    return;
  }

  try {
    const report = await loadReport(reportId);
    displayReport(report);
  } catch (error: any) {
    console.error(chalk.red(`\n❌ Error loading report: ${error.message}\n`));
  }
}

async function loadReport(reportId: string): Promise<SimulationReport> {
  const fs = await import('fs/promises');
  const path = await import('path');

  // Try to load from reports directory
  const reportPath = path.join('./reports', reportId);

  let content: string;
  try {
    content = await fs.readFile(reportPath, 'utf-8');
  } catch {
    // Try with .json extension
    content = await fs.readFile(`${reportPath}.json`, 'utf-8');
  }

  try {
    return JSON.parse(content);
  } catch (err) {
    throw new Error(`Failed to parse report file as JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
}

async function listReports(): Promise<void> {
  const fs = await import('fs/promises');
  const path = await import('path');

  console.log(chalk.cyan.bold('\n📊 Simulation Reports\n'));

  try {
    const reportsDir = './reports';
    const files = await fs.readdir(reportsDir);

    const reports: ReportMetadata[] = [];

    for (const file of files) {
      if (!file.endsWith('.json')) continue;

      const filePath = path.join(reportsDir, file);
      const content = await fs.readFile(filePath, 'utf-8');
      let report: SimulationReport;
      try {
        report = JSON.parse(content);
      } catch {
        // Skip malformed report files
        continue;
      }

      reports.push({
        id: file.replace('.json', ''),
        scenarioId: report.scenarioId,
        timestamp: report.startTime,
        path: filePath,
        format: 'json',
        coherenceScore: report.coherenceScore,
        successRate: report.summary.successRate,
      });
    }

    if (reports.length === 0) {
      console.log(chalk.gray('No reports found.\n'));
      return;
    }

    // Sort by timestamp (newest first)
    reports.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    // Display table
    console.log(chalk.bold('Available Reports:\n'));
    console.log(
      chalk.gray('ID'.padEnd(30) + 'Scenario'.padEnd(20) + 'Success'.padEnd(12) + 'Coherence'.padEnd(12) + 'Timestamp')
    );
    console.log(chalk.gray('─'.repeat(100)));

    reports.forEach((r) => {
      const success = (r.successRate * 100).toFixed(0) + '%';
      const coherence = (r.coherenceScore * 100).toFixed(0) + '%';
      const timestamp = new Date(r.timestamp).toLocaleString();

      console.log(
        r.id.padEnd(30) +
          chalk.cyan(r.scenarioId.padEnd(20)) +
          success.padEnd(12) +
          coherence.padEnd(12) +
          chalk.gray(timestamp)
      );
    });

    console.log('\n' + chalk.gray('Use: agentdb simulate --report [id] to view details\n'));
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      console.log(chalk.gray('No reports directory found.\n'));
    } else {
      console.error(chalk.red(`Error: ${error.message}\n`));
    }
  }
}

function displayReport(report: SimulationReport): void {
  console.log(chalk.cyan.bold(`\n📊 Simulation Report: ${report.scenarioId}\n`));

  // Meta information
  console.log(chalk.bold('Meta Information:'));
  console.log(`  Generated: ${report.startTime}`);
  console.log(`  Duration: ${(report.totalDuration / 1000).toFixed(2)}s`);
  console.log(`  Iterations: ${report.iterations.length}`);
  console.log('');

  // Configuration
  console.log(chalk.bold('Configuration:'));
  if (report.optimal) {
    console.log(chalk.green('  ✅ Optimal configuration'));
  } else {
    console.log(chalk.yellow('  ⚠️  Non-optimal configuration'));
  }
  console.log('');

  // Summary
  console.log(chalk.bold('Summary:'));
  console.log(`  Success Rate: ${(report.summary.successRate * 100).toFixed(1)}%`);
  console.log(`  Coherence Score: ${(report.coherenceScore * 100).toFixed(1)}%`);
  console.log(`  Avg Latency (p50): ${report.summary.avgLatencyUs.toFixed(2)}μs`);
  console.log(`  Avg Recall@10: ${(report.summary.avgRecall * 100).toFixed(1)}%`);
  console.log(`  Avg QPS: ${report.summary.avgQps.toLocaleString()}`);
  console.log(`  Avg Memory: ${report.summary.avgMemoryMB.toFixed(0)}MB`);
  console.log('');

  // Variance
  console.log(chalk.bold('Variance:'));
  console.log(`  Latency: ${report.varianceMetrics.latencyVariance.toFixed(4)}`);
  console.log(`  Recall: ${report.varianceMetrics.recallVariance.toFixed(6)}`);
  console.log(`  QPS: ${report.varianceMetrics.qpsVariance.toFixed(2)}`);
  console.log('');

  // Iterations
  console.log(chalk.bold('Iteration Results:'));
  report.iterations.forEach((iter) => {
    const status = iter.success ? chalk.green('✅') : chalk.red('❌');
    const duration = (iter.duration / 1000).toFixed(2) + 's';
    console.log(`  ${status} Iteration ${iter.iteration}: ${duration}`);
  });
  console.log('');

  // Warnings
  if (report.warnings.length > 0) {
    console.log(chalk.bold('Warnings:'));
    report.warnings.forEach((w) => console.log(chalk.yellow(`  ⚠️  ${w}`)));
    console.log('');
  }
}
