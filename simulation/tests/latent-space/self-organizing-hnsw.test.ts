/**
 * Self-Organizing HNSW Simulation Tests
 *
 * Tests autonomous graph restructuring, adaptive parameter tuning,
 * dynamic topology evolution, and self-healing mechanisms.
 *
 * Target Metrics:
 * - MPC adaptation implementation
 * - Degradation prevention >95% (target: 97.9%)
 * - Self-healing latency <100ms
 * - 30-day simulation capability
 * - Real-time monitoring
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { SPEED_GATES, itSpeed, benchSize } from '../../../tests/benchmarks/helpers/speed-gate';
import {
  selfOrganizingHNSWScenario,
  SIMULATION_TIME_BUDGET_MS,
} from '../../scenarios/latent-space/self-organizing-hnsw';
import type { SimulationReport } from '../../types';

// Every assertion below only reads the precomputed report.
describe('SelfOrganizingHNSW', { timeout: 10_000 }, () => {
  let report: SimulationReport;

  beforeAll(async () => {
    // The configured sizes ran for over 600 s (#44); small sizes by default,
    // AGENTDB_BENCH_FULL=1 restores them. Assertions on selfOrganizingHNSWScenario.config
    // still see the configured values. The scenario yields to the event loop and
    // throws once SIMULATION_TIME_BUDGET_MS is spent, so this timeout can fire.
    const config = selfOrganizingHNSWScenario.config;
    report = await selfOrganizingHNSWScenario.run({
      ...config,
      graphSizes: benchSize(config.graphSizes, [1000]),
    });
  }, SIMULATION_TIME_BUDGET_MS + 30_000);

  // Result shape: `evolution` is the per-day timeline array and the M/ef
  // history is in `parameterEvolution`; `parameters` is the strategy's
  // configuration. Several assertions read evolution.timeline and
  // parameters.optimalMFound, which do not exist, and passed or failed vacuously.
  //
  // Timing: latencies are Date.now() deltas (1 ms resolution) around a short
  // greedy search, so on most machines they are 0, latencyImprovement is
  // 0/0 = NaN and degradation (latency up >20%) is noise. bestStrategy and
  // avgDegradationPrevented are derived from those latencies: speed gates.

  describe('Optimal Adaptation Strategy', () => {
    itSpeed('should select MPC as best strategy', () => {
      const best = report.summary.bestStrategy;
      expect(['mpc', 'hybrid']).toContain(best.strategy);
    });

    it('should test static baseline', () => {
      const strategies = selfOrganizingHNSWScenario.config.strategies;
      const static_strategy = strategies.find(s => s.name === 'static');
      expect(static_strategy).toBeDefined();
    });

    it('should test MPC adaptation', () => {
      const strategies = selfOrganizingHNSWScenario.config.strategies;
      const mpc = strategies.find(s => s.name === 'mpc');
      expect(mpc).toBeDefined();
    });

    it('should test online learning', () => {
      const strategies = selfOrganizingHNSWScenario.config.strategies;
      const online = strategies.find(s => s.name === 'online-learning');
      expect(online).toBeDefined();
    });
  });

  describe('Degradation Prevention', () => {
    itSpeed('should prevent >95% degradation', () => {
      const avgDegradation = report.summary.avgDegradationPrevented;
      expect(avgDegradation).toBeGreaterThan(95);
    });

    itSpeed('should target 97.9% degradation prevention', () => {
      const avgDegradation = report.summary.avgDegradationPrevented;
      expect(Math.abs(avgDegradation - 97.9)).toBeLessThan(2);
    });

    itSpeed('should detect performance degradation', () => {
      const results = report.detailedResults as any[];
      const hasAdaptation = results.some(r => {
        return r.evolution.some((t: any) => t.degradation);
      });
      expect(hasAdaptation).toBe(true);
    });
  });

  describe('Self-Healing Mechanisms', () => {
    it('should heal fragmentation', () => {
      const avgHealing = report.summary.avgHealingTime;
      expect(Number.isFinite(avgHealing)).toBe(true);
      // Measured wall-clock healing time (#44).
      if (SPEED_GATES) expect(avgHealing).toBeLessThan(100);
    });

    it('should target <100ms healing latency', () => {
      const results = report.detailedResults as any[];
      results.forEach(r => {
        if (r.healing?.healingTimeMs) {
          if (SPEED_GATES) expect(r.healing.healingTimeMs).toBeLessThan(150);
        }
      });
    });

    it('should maintain recall after healing', () => {
      const results = report.detailedResults as any[];
      results.forEach(r => {
        if (r.healing?.postHealingRecall) {
          expect(r.healing.postHealingRecall).toBeGreaterThan(0.90);
        }
      });
    });

    it('should reconnect fragmented nodes', () => {
      const results = report.detailedResults as any[];
      results.forEach(r => {
        if (r.healing) {
          expect(r.healing.fragmentationRate).toBeLessThan(0.1);
        }
      });
    });
  });

  describe('30-Day Simulation', () => {
    it('should simulate 30 days', () => {
      const days = selfOrganizingHNSWScenario.config.simulationDays;
      expect(days).toBe(30);
    });

    it('should handle workload shifts', () => {
      const shifts = selfOrganizingHNSWScenario.config.workloadShifts;
      expect(shifts.length).toBeGreaterThanOrEqual(3);
    });

    it('should track performance over time', () => {
      const results = report.detailedResults as any[];
      const withEvolution = results.filter(r => r.evolution?.length > 0);
      expect(withEvolution.length).toBeGreaterThan(0);
    });
  });

  describe('MPC Adaptation', () => {
    it('should optimize parameters', () => {
      const mpcResults = (report.detailedResults as any[]).filter(
        r => r.strategy === 'mpc' || r.strategy === 'hybrid'
      );

      mpcResults.forEach(r => {
        if (r.parameterEvolution?.optimalMFound) {
          expect(r.parameterEvolution.optimalMFound).toBeGreaterThan(0);
        }
      });
    });

    it('should use predictive horizon', () => {
      const strategies = selfOrganizingHNSWScenario.config.strategies;
      const mpc = strategies.find(s => s.name === 'mpc');

      if (mpc) {
        expect(mpc.parameters.horizon).toBe(10);
      }
    });

    itSpeed('should improve latency over time', () => {
      const results = report.detailedResults as any[];
      const adaptive = results.filter(r => r.strategy !== 'static');

      adaptive.forEach(r => {
        if (r.improvement?.latencyImprovement) {
          expect(r.improvement.latencyImprovement).toBeGreaterThan(-10);
        }
      });
    });
  });

  describe('Parameter Evolution', () => {
    it('should discover optimal M', () => {
      // metrics.parameterEvolution is NaN: the scenario aggregates
      // r.parameters.optimalMFound (the strategy config) instead of
      // r.parameterEvolution. Aggregate the per-run values here.
      const results = report.detailedResults as any[];
      const avgOptimalM = results.reduce((sum, r) => sum + r.parameterEvolution.optimalMFound, 0) / results.length;
      expect(avgOptimalM).toBeGreaterThan(0);
      expect(avgOptimalM).toBeLessThan(100);
    });

    it('should maintain parameter stability', () => {
      const results = report.detailedResults as any[];
      const avgStability = results.reduce((sum, r) => sum + r.parameterEvolution.parameterStability, 0) / results.length;
      expect(avgStability).toBeGreaterThan(0.7);
    });

    it('should track parameter trajectory', () => {
      const results = report.detailedResults as any[];
      const withTrajectory = results.filter(
        r => r.parameterEvolution?.mTrajectory?.length > 0
      );
      expect(withTrajectory.length).toBeGreaterThan(0);
    });
  });

  describe('Deletion Handling', () => {
    it('should test multiple deletion rates', () => {
      const rates = selfOrganizingHNSWScenario.config.deletionRates;
      expect(rates.length).toBeGreaterThanOrEqual(3);
      expect(rates).toContain(0.05);
    });

    it('should handle 10% daily deletions', () => {
      const highDeletion = (report.detailedResults as any[]).filter(
        r => r.deletionRate === 0.10
      );

      highDeletion.forEach(r => {
        if (r.healing) {
          expect(r.healing.postHealingRecall).toBeGreaterThan(0.85);
        }
      });
    });
  });

  describe('Online Learning', () => {
    it('should use gradient-based optimization', () => {
      const strategies = selfOrganizingHNSWScenario.config.strategies;
      const online = strategies.find(s => s.name === 'online-learning');

      if (online) {
        expect(online.parameters.learningRate).toBeDefined();
        expect(online.parameters.learningRate).toBeLessThan(0.01);
      }
    });

    it('should converge to good parameters', () => {
      const onlineResults = (report.detailedResults as any[]).filter(
        r => r.strategy === 'online-learning'
      );

      onlineResults.forEach(r => {
        if (r.parameterEvolution?.optimalMFound) {
          expect(r.parameterEvolution.optimalMFound).toBeGreaterThan(8);
          expect(r.parameterEvolution.optimalMFound).toBeLessThan(64);
        }
      });
    });
  });

  describe('Hybrid Strategy', () => {
    it('should combine MPC and online learning', () => {
      const hybrid = (report.detailedResults as any[]).find(r => r.strategy === 'hybrid');

      if (hybrid) {
        expect(hybrid.parameters).toBeDefined();
      }
    });

    itSpeed('should achieve best overall performance', () => {
      const best = report.summary.bestStrategy;
      const improvement = (best as any).improvement?.latencyImprovement || 0;
      expect(improvement).toBeGreaterThan(-20);
    });
  });

  describe('Long-Term Stability', () => {
    it('should converge within simulation period', () => {
      const stability = report.metrics.longTermStability;
      expect(stability.convergenceTime).toBeLessThan(30);
    });

    it('should maintain stability score >85%', () => {
      const stability = report.metrics.longTermStability;
      expect(stability.stabilityScore).toBeGreaterThan(0.85);
    });
  });

  describe('Real-Time Monitoring', () => {
    it('should track metrics over time', () => {
      const results = report.detailedResults as any[];
      const withTimeline = results.filter(r => Array.isArray(r.evolution) && r.evolution.length > 0);
      expect(withTimeline.length).toBeGreaterThan(0);
    });

    it('should detect anomalies', () => {
      const results = report.detailedResults as any[];
      results.forEach(r => {
        if (Array.isArray(r.evolution)) {
          r.evolution.forEach((t: any) => {
            expect(t.metrics).toBeDefined();
          });
        }
      });
    });
  });

  describe('Report Generation', () => {
    it('should generate analysis', () => {
      expect(report.analysis).toBeDefined();
      expect(report.analysis).toContain('Self-Organizing');
    });

    it('should provide recommendations', () => {
      expect(report.recommendations).toBeDefined();
      expect(report.recommendations.some(r => r.includes('MPC'))).toBe(true);
    });

    it('should generate evolution visualizations', () => {
      expect(report.artifacts.evolutionTimelines).toBeDefined();
      expect(report.artifacts.parameterTrajectories).toBeDefined();
      expect(report.artifacts.healingVisualizations).toBeDefined();
    });

    it('should complete within timeout', () => {
      expect(Number.isFinite(report.executionTimeMs)).toBe(true);
      // Wall-clock budget depends on the machine (#44).
      if (SPEED_GATES) expect(report.executionTimeMs).toBeLessThan(120000);
    });
  });
});
