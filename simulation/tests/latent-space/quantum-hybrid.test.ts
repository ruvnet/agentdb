/**
 * Quantum-Hybrid HNSW Simulation Tests (Theoretical)
 *
 * Tests theoretical quantum-enhanced HNSW approaches including
 * quantum amplitude encoding, Grover search, and quantum walks.
 *
 * **IMPORTANT**: This is theoretical validation only, not actual quantum computing.
 *
 * Target Metrics:
 * - Viability assessment (2025: 12.4%, 2030: 38.2%, 2040: 84.7%)
 * - Theoretical speedup calculations (Grover: 4x)
 * - Hardware requirement progression
 * - Keep as theoretical validation only
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { SPEED_GATES } from '../../../tests/benchmarks/helpers/speed-gate';
import { quantumHybridScenario } from '../../scenarios/latent-space/quantum-hybrid';
import type { SimulationReport } from '../../types';

describe('QuantumHybrid (Theoretical)', () => {
  let report: SimulationReport;

  beforeAll(async () => {
    report = await quantumHybridScenario.run(quantumHybridScenario.config);
  }, 60000); // 60s timeout

  describe('Disclaimer and Scope', () => {
    it('should be marked as theoretical', () => {
      expect(report.scenarioId).toBe('quantum-hybrid');
      expect(report.analysis).toContain('DISCLAIMER');
      expect(report.analysis).toContain('theoretical');
    });

    it('should not claim real quantum computing', () => {
      expect(report.analysis).toContain('research purposes');
    });
  });

  describe('2025 Viability (Current)', () => {
    it('should assess near-term viability ~12.4%', () => {
      const nearTerm = report.summary.nearTermViability;
      expect(nearTerm).toBeLessThan(0.20);
      expect(nearTerm).toBeGreaterThan(0.05);
    });

    it('should target ~12.4% viability', () => {
      const nearTerm = report.summary.nearTermViability;
      expect(nearTerm).toBeCloseTo(0.124, 0.08);
    });

    it('should have limited qubit count', () => {
      const hardware2025 = quantumHybridScenario.config.hardwareProfiles.find(h => h.year === 2025);
      expect(hardware2025?.qubits).toBe(100);
    });

    it('should have high error rates', () => {
      const hardware2025 = quantumHybridScenario.config.hardwareProfiles.find(h => h.year === 2025);
      expect(hardware2025?.errorRate).toBe(0.001);
    });
  });

  describe('2030 Viability (Mid-term)', () => {
    it('should project ~38.2% viability', () => {
      const results = (report.detailedResults as any[]).filter(r => r.hardwareYear === 2030);

      if (results.length > 0) {
        // The 2030 projection is `viability`; `current2025Viability` is the
        // 2025 baseline (0.124) carried on every result, so it never measured 2030.
        const avg = results.reduce((sum, r) => sum + (r.viability?.viability || 0), 0) / results.length;
        expect(avg).toBeGreaterThan(0.20);
        expect(avg).toBeLessThan(0.60);
      }
    });
  });

  describe('2040 Viability (Long-term)', () => {
    it('should project ~84.7% viability', () => {
      const longTerm = report.summary.longTermProjection;
      expect(longTerm).toBeGreaterThan(0.70);
    });

    it('should target 84.7% by 2040', () => {
      const longTerm = report.summary.longTermProjection;
      expect(longTerm).toBeCloseTo(0.847, 0.15);
    });

    it('should have advanced hardware', () => {
      const hardware2040 = quantumHybridScenario.config.hardwareProfiles.find(h => h.year === 2040);
      expect(hardware2040?.qubits).toBe(10000);
      expect(hardware2040?.errorRate).toBe(0.00001);
    });
  });

  describe('Grover Algorithm', () => {
    it('should test Grover search', () => {
      const algorithms = quantumHybridScenario.config.algorithms;
      const grover = algorithms.find(a => a.name === 'grover');
      expect(grover).toBeDefined();
    });

    it('should calculate √M speedup', () => {
      const groverResults = (report.detailedResults as any[]).filter(r => r.algorithm === 'grover');

      groverResults.forEach(r => {
        if (r.speedups?.groverSpeedup) {
          const M = r.parameters?.neighborhoodSize || 16;
          expect(r.speedups.groverSpeedup).toBeCloseTo(Math.sqrt(M), 2);
        }
      });
    });

    it('should target 4x speedup for M=16', () => {
      const grover16 = (report.detailedResults as any[]).find(
        r => r.algorithm === 'grover' && r.parameters?.neighborhoodSize === 16
      );

      if (grover16 && grover16.speedups) {
        expect(grover16.speedups.groverSpeedup).toBeCloseTo(4, 1);
      }
    });
  });

  describe('Quantum Walk', () => {
    it('should test quantum walk algorithm', () => {
      const algorithms = quantumHybridScenario.config.algorithms;
      const qwalk = algorithms.find(a => a.name === 'quantum-walk');
      expect(qwalk).toBeDefined();
    });

    it('should calculate speedup', () => {
      const qwalkResults = (report.detailedResults as any[]).filter(r => r.algorithm === 'quantum-walk');

      qwalkResults.forEach(r => {
        if (r.speedups?.quantumWalkSpeedup) {
          expect(r.speedups.quantumWalkSpeedup).toBeGreaterThan(1);
        }
      });
    });
  });

  describe('Amplitude Encoding', () => {
    it('should test amplitude encoding', () => {
      const algorithms = quantumHybridScenario.config.algorithms;
      const amplitude = algorithms.find(a => a.name === 'amplitude-encoding');
      expect(amplitude).toBeDefined();
    });

    it('should require log(d) qubits', () => {
      const ampResults = (report.detailedResults as any[]).filter(r => r.algorithm === 'amplitude-encoding');

      ampResults.forEach(r => {
        const dim = r.dimension || 128;
        const expectedQubits = Math.ceil(Math.log2(dim));
        if (r.quantumMetrics?.qubitsRequired) {
          expect(r.quantumMetrics.qubitsRequired).toBeCloseTo(expectedQubits, 2);
        }
      });
    });
  });

  describe('Hybrid Approach', () => {
    it('should test hybrid classical-quantum', () => {
      const algorithms = quantumHybridScenario.config.algorithms;
      const hybrid = algorithms.find(a => a.name === 'hybrid');
      expect(hybrid).toBeDefined();
    });

    it('should split workload', () => {
      const hybridResults = (report.detailedResults as any[]).filter(r => r.algorithm === 'hybrid');

      hybridResults.forEach(r => {
        if (r.quantumMetrics) {
          expect(r.quantumMetrics.classicalFraction).toBeGreaterThan(0);
          expect(r.quantumMetrics.quantumFraction).toBeGreaterThan(0);
          expect(r.quantumMetrics.classicalFraction + r.quantumMetrics.quantumFraction).toBeCloseTo(1.0, 0.1);
        }
      });
    });

    it('should respect qubit budget', () => {
      const hybrid = quantumHybridScenario.config.algorithms.find(a => a.name === 'hybrid');
      expect(hybrid?.parameters.quantumBudget).toBe(50);
    });
  });

  describe('Resource Requirements', () => {
    it('should track qubit requirements', () => {
      const metrics = report.metrics.resourceRequirements;
      expect(metrics.avgQubitsRequired).toBeGreaterThan(0);
    });

    it('should track gate depth', () => {
      const metrics = report.metrics.resourceRequirements;
      expect(metrics.maxGateDepth).toBeGreaterThan(0);
    });

    it('should assess feasibility', () => {
      const results = report.detailedResults as any[];
      results.forEach(r => {
        if (r.resources) {
          expect(r.resources.feasible).toBeDefined();
        }
      });
    });
  });

  describe('Hardware Progression', () => {
    it('should test multiple hardware profiles', () => {
      const profiles = quantumHybridScenario.config.hardwareProfiles;
      expect(profiles.length).toBe(3);
      expect(profiles.map(p => p.year)).toEqual([2025, 2030, 2040]);
    });

    it('should show qubit growth', () => {
      const profiles = quantumHybridScenario.config.hardwareProfiles;
      expect(profiles[0].qubits).toBeLessThan(profiles[1].qubits);
      expect(profiles[1].qubits).toBeLessThan(profiles[2].qubits);
    });

    it('should show error rate improvement', () => {
      const profiles = quantumHybridScenario.config.hardwareProfiles;
      expect(profiles[0].errorRate).toBeGreaterThan(profiles[1].errorRate);
      expect(profiles[1].errorRate).toBeGreaterThan(profiles[2].errorRate);
    });
  });

  describe('Bottleneck Analysis', () => {
    it('should identify bottlenecks', () => {
      const results = report.detailedResults as any[];
      const with2025 = results.filter(r => r.hardwareYear === 2025);

      with2025.forEach(r => {
        if (r.viability?.bottleneck) {
          expect(['qubits', 'coherence', 'error-rate']).toContain(r.viability.bottleneck);
        }
      });
    });
  });

  describe('Theoretical Speedup Validation', () => {
    it('should calculate theoretical speedups', () => {
      const speedups = report.metrics.theoreticalSpeedups;
      expect(speedups.maxTheoreticalSpeedup).toBeGreaterThan(1);
    });

    it('should not exceed theoretical limits', () => {
      const results = report.detailedResults as any[];
      // Amplitude encoding models O(1) vs O(d) inner products, so its speedup
      // is the dimension itself (1024 at the largest configured dim). The
      // bound is the largest dimension, not a fixed 1000.
      const maxDim = Math.max(...quantumHybridScenario.config.dimensions);
      results.forEach(r => {
        if (r.speedups?.theoreticalSpeedup) {
          expect(r.speedups.theoreticalSpeedup).toBeLessThanOrEqual(maxDim);
        }
      });
    });
  });

  describe('Classical Baseline', () => {
    it('should test classical baseline', () => {
      const algorithms = quantumHybridScenario.config.algorithms;
      const classical = algorithms.find(a => a.name === 'classical');
      expect(classical).toBeDefined();
    });

    it('should have 1x speedup for classical', () => {
      const classicalResults = (report.detailedResults as any[]).filter(r => r.algorithm === 'classical');

      classicalResults.forEach(r => {
        if (r.speedups) {
          expect(r.speedups.theoreticalSpeedup).toBe(1.0);
        }
      });
    });
  });

  describe('Recommendations', () => {
    it('should warn about current viability', () => {
      expect(report.recommendations.some(r => r.includes('NOT viable'))).toBe(true);
    });

    it('should suggest hybrid approaches', () => {
      expect(report.recommendations.some(r => r.includes('hybrid'))).toBe(true);
    });

    it('should project future timeline', () => {
      expect(report.recommendations.some(r => r.includes('2040'))).toBe(true);
    });
  });

  describe('Report Generation', () => {
    it('should generate theoretical analysis', () => {
      expect(report.analysis).toBeDefined();
      expect(report.analysis).toContain('Quantum');
    });

    it('should provide viability assessment', () => {
      expect(report.analysis).toContain('2025');
      expect(report.analysis).toContain('2045');
    });

    it('should generate visualizations', () => {
      expect(report.artifacts.speedupCharts).toBeDefined();
      expect(report.artifacts.resourceDiagrams).toBeDefined();
      expect(report.artifacts.viabilityTimeline).toBeDefined();
    });

    it('should complete efficiently', () => {
      expect(Number.isFinite(report.executionTimeMs)).toBe(true);
      // Wall-clock budget depends on the machine (#44).
      if (SPEED_GATES) expect(report.executionTimeMs).toBeLessThan(60000);
    });
  });
});
