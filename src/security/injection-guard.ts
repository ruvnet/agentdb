/**
 * Opt-in prompt-injection screening for text written into memory (ADR-018).
 *
 * Stored episodes, reasoning patterns and skills are read back into agent
 * prompts later, so an instruction planted in one is a memory-poisoning
 * vector. When enabled, writes are scanned with aidefence's pattern-pack
 * InjectionDetector (`aidefence/dist/detection`, the side-effect-free entry
 * that does not load its HTTP gateway).
 *
 * Mode, from `setInjectionGuard()` or AGENTDB_INJECTION_GUARD:
 *   off   (default) no scanning, aidefence is never loaded
 *   warn  log detections to stderr, keep the write
 *   block refuse the write with ValidationError code INJECTION_DETECTED;
 *         fails closed if aidefence is not installed
 */

import { ValidationError } from './input-validation.js';
import { optionalImport, pickExport } from '../utils/optional-import.js';

export type InjectionGuardMode = 'off' | 'warn' | 'block';

/** Severity at or above which a detection counts (default 'medium'). */
export type InjectionSeverity = 'low' | 'medium' | 'high' | 'critical';

export const AIDEFENCE_DETECTION_ENTRY = 'aidefence/dist/detection/index.js';

const SEVERITY_RANK: Record<InjectionSeverity, number> = { low: 1, medium: 2, high: 3, critical: 4 };

export interface InjectionFinding {
  field: string;
  severity: InjectionSeverity;
  ids: string[];
}

interface Detector {
  detect(input: string): {
    safe: boolean;
    maxSeverity: InjectionSeverity | null;
    threats: ReadonlyArray<{ id: string; severity: InjectionSeverity }>;
  };
}

let configuredMode: InjectionGuardMode | null = null;
let minSeverity: InjectionSeverity = 'medium';
let detectorPromise: Promise<Detector | null> | null = null;

export function setInjectionGuard(mode: InjectionGuardMode | null, options: { minSeverity?: InjectionSeverity } = {}): void {
  configuredMode = mode;
  if (options.minSeverity) minSeverity = options.minSeverity;
}

export function getInjectionGuardMode(): InjectionGuardMode {
  if (configuredMode) return configuredMode;
  const env = typeof process !== 'undefined' ? process.env?.AGENTDB_INJECTION_GUARD : undefined;
  return env === 'warn' || env === 'block' ? env : 'off';
}

function loadDetector(): Promise<Detector | null> {
  if (!detectorPromise) {
    detectorPromise = optionalImport(AIDEFENCE_DETECTION_ENTRY).then((mod) => {
      const create = pickExport<() => Detector>(mod, 'createInjectionDetector');
      return typeof create === 'function' ? create() : null;
    });
  }
  return detectorPromise;
}

/** Scan named fields; returns findings at or above the configured severity. */
export async function scanForInjection(fields: Record<string, string | null | undefined>): Promise<InjectionFinding[] | null> {
  const detector = await loadDetector();
  if (!detector) return null;
  const findings: InjectionFinding[] = [];
  for (const [field, value] of Object.entries(fields)) {
    if (typeof value !== 'string' || value.length === 0) continue;
    const report = detector.detect(value);
    const hits = report.threats.filter((t) => SEVERITY_RANK[t.severity] >= SEVERITY_RANK[minSeverity]);
    if (hits.length === 0) continue;
    const top = hits.reduce((a, b) => (SEVERITY_RANK[b.severity] > SEVERITY_RANK[a.severity] ? b : a));
    findings.push({ field, severity: top.severity, ids: [...new Set(hits.map((t) => t.id))] });
  }
  return findings;
}

/**
 * Called by the memory controllers before a write. No-op when the guard is off.
 * @param kind record type, used in messages only (e.g. 'episode')
 */
export async function screenStoredText(
  kind: string,
  fields: Record<string, string | null | undefined>
): Promise<void> {
  const mode = getInjectionGuardMode();
  if (mode === 'off') return;
  const findings = await scanForInjection(fields);
  if (findings === null) {
    if (mode === 'block') {
      throw new ValidationError(
        'AGENTDB_INJECTION_GUARD=block requires the aidefence package (npm install aidefence)',
        'INJECTION_GUARD_UNAVAILABLE'
      );
    }
    warnOnce('[agentdb] AGENTDB_INJECTION_GUARD=warn but aidefence is not installed; writes are not screened');
    return;
  }
  if (findings.length === 0) return;
  const summary = findings.map((f) => `${f.field}: ${f.severity} (${f.ids.join(', ')})`).join('; ');
  if (mode === 'block') {
    throw new ValidationError(`Possible prompt injection in ${kind} — ${summary}`, 'INJECTION_DETECTED', findings[0].field);
  }
  console.warn(`[agentdb] possible prompt injection in ${kind} — ${summary}`);
}

let warned = false;
function warnOnce(message: string): void {
  if (warned) return;
  warned = true;
  console.warn(message);
}

/** Test hook. */
export function resetInjectionGuard(): void {
  configuredMode = null;
  minSeverity = 'medium';
  detectorPromise = null;
  warned = false;
}
