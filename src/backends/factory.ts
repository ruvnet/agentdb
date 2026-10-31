/**
 * Backend Factory - Automatic Backend Detection and Selection
 *
 * Detects available vector backends and creates appropriate instances.
 * Priority: RuVector (native/WASM) > RVF (native/WASM) > HNSWLib (Node.js)
 *
 * Features:
 * - Automatic detection of @ruvector and @ruvector/rvf packages
 * - Native vs WASM detection for RuVector and RVF
 * - GNN and Graph capabilities detection
 * - Graceful fallback chain: RuVector -> RVF -> HNSWLib
 * - Clear error messages for missing dependencies
 */

import type { VectorBackend, VectorConfig } from './VectorBackend.js';
import { RuVectorBackend } from './ruvector/RuVectorBackend.js';

// Note: HNSWLibBackend and RvfBackend are lazy-loaded to avoid import failures
// on systems without build tools. The imports happen in helper functions.

export type BackendType = 'auto' | 'ruvector' | 'rvf' | 'hnswlib';

export interface RvfDetection {
  sdk: boolean;
  node: boolean;
  wasm: boolean;
}

export interface BackendDetection {
  available: 'ruvector' | 'rvf' | 'hnswlib' | 'sqljsrvf' | 'none';
  ruvector: {
    core: boolean;
    gnn: boolean;
    graph: boolean;
    native: boolean;
  };
  rvf: RvfDetection;
  hnswlib: boolean;
  sqljsRvf: boolean;
}

/**
 * Detect available vector backends
 */
export async function detectBackends(): Promise<BackendDetection> {
  const result: BackendDetection = {
    available: 'none',
    ruvector: {
      core: false,
      gnn: false,
      graph: false,
      native: false
    },
    rvf: {
      sdk: false,
      node: false,
      wasm: false,
    },
    hnswlib: false,
    sqljsRvf: false,
  };

  // Check RuVector packages (main package or scoped packages)
  try {
    // Try main ruvector package first
    const ruvector = await import('ruvector');
    result.ruvector.core = true;
    result.ruvector.gnn = true; // Main package includes GNN
    result.ruvector.graph = true; // Main package includes Graph
    result.ruvector.native = ruvector.isNative?.() ?? false;
    result.available = 'ruvector';
  } catch {
    // Try scoped packages as fallback
    try {
      const core = await import('@ruvector/core');
      result.ruvector.core = true;
      result.ruvector.native = core.isNative?.() ?? false;
      result.available = 'ruvector';

      // Check optional packages
      try {
        await import('@ruvector/gnn');
        result.ruvector.gnn = true;
      } catch {
        // GNN not installed - this is optional
      }

      try {
        await import('@ruvector/graph-node');
        result.ruvector.graph = true;
      } catch {
        // Graph not installed - this is optional
      }
    } catch {
      // RuVector not installed - will try RVF or HNSWLib fallback
    }
  }

  // Check RVF SDK (@ruvector/rvf with N-API or WASM backend)
  try {
    await import('@ruvector/rvf');
    result.rvf.sdk = true;

    // Check for N-API native backend
    try {
      await import('@ruvector/rvf-node');
      result.rvf.node = true;
    } catch {
      // N-API backend not available
    }

    // Check for WASM backend
    try {
      await import('@ruvector/rvf-wasm');
      result.rvf.wasm = true;
    } catch {
      // WASM backend not available
    }

    if (result.available === 'none') {
      result.available = 'rvf';
    }
  } catch {
    // RVF SDK not installed
  }

  // Check HNSWLib
  try {
    await import('hnswlib-node');
    result.hnswlib = true;

    if (result.available === 'none') {
      result.available = 'hnswlib';
    }
  } catch {
    // HNSWLib not installed
  }

  // Check sql.js (always-available built-in RVF fallback)
  try {
    await import('sql.js');
    result.sqljsRvf = true;
    if (result.available === 'none') {
      result.available = 'sqljsrvf';
    }
  } catch {
    result.sqljsRvf = false;
  }

  return result;
}

/**
 * Lazy-load HNSWLibBackend to avoid import failures on systems without build tools
 */
async function createHNSWLibBackend(config: VectorConfig): Promise<VectorBackend> {
  const { HNSWLibBackend } = await import('./hnswlib/HNSWLibBackend.js');
  return new HNSWLibBackend(config);
}

/**
 * Lazy-load RvfBackend to avoid import failures when @ruvector/rvf is not installed
 */
async function createRvfBackend(config: VectorConfig): Promise<VectorBackend> {
  const { RvfBackend } = await import('./rvf/RvfBackend.js');
  return new RvfBackend(config);
}

/**
 * Lazy-load SqlJsRvfBackend - built-in RVF persistence using sql.js WASM.
 * Always available since sql.js is a hard dependency.
 */
async function createSqlJsRvfBackend(config: VectorConfig): Promise<VectorBackend> {
  const { SqlJsRvfBackend } = await import('./rvf/SqlJsRvfBackend.js');
  return new SqlJsRvfBackend(config);
}

/**
 * Create vector backend with automatic detection
 *
 * @param type - Backend type: 'auto', 'ruvector', 'rvf', or 'hnswlib'
 * @param config - Vector configuration
 * @returns Initialized VectorBackend instance
 */
export async function createBackend(
  type: BackendType,
  config: VectorConfig
): Promise<VectorBackend> {
  const detection = await detectBackends();

  let backend: VectorBackend;

  // Handle explicit backend selection
  if (type === 'ruvector') {
    if (!detection.ruvector.core) {
      throw new Error(
        'RuVector not available.\n' +
        'Install with: npm install @ruvector/core\n' +
        'Optional GNN support: npm install @ruvector/gnn\n' +
        'Optional Graph support: npm install @ruvector/graph-node'
      );
    }
    backend = new RuVectorBackend(config);
  } else if (type === 'rvf') {
    // Try native @ruvector/rvf first, fall back to sql.js-rvf
    if (detection.rvf.sdk) {
      backend = await createRvfBackend(config);
      console.error(
        `[AgentDB] Using RVF backend (${detection.rvf.node ? 'N-API native' : 'WASM'})`
      );
    } else if (detection.sqljsRvf) {
      backend = await createSqlJsRvfBackend(config);
      console.error('[AgentDB] Using sql.js RVF backend (built-in)');
    } else {
      throw new Error(
        'RVF backend not available.\n' +
        'Install with: npm install @ruvector/rvf\n' +
        'Native backend: npm install @ruvector/rvf-node\n' +
        'WASM backend: npm install @ruvector/rvf-wasm'
      );
    }
  } else if (type === 'hnswlib') {
    if (!detection.hnswlib) {
      throw new Error(
        'HNSWLib not available.\n' +
        'Install with: npm install hnswlib-node'
      );
    }
    backend = await createHNSWLibBackend(config);
  } else {
    // Auto-detect best available backend (priority: ruvector > rvf > hnswlib)
    if (detection.ruvector.core) {
      backend = new RuVectorBackend(config);
      console.error(
        `[AgentDB] Using RuVector backend (${detection.ruvector.native ? 'native' : 'WASM'})`
      );

      // Try to initialize RuVector, fallback to RVF then HNSWLib if it fails
      try {
        await (backend as unknown as { initialize(): Promise<void> }).initialize();
        return backend;
      } catch (error) {
        const errorMessage = (error as Error).message;

        // Try RVF as first fallback
        if (detection.rvf.sdk) {
          console.error('[AgentDB] RuVector initialization failed, trying RVF backend');
          console.error(`[AgentDB] Reason: ${errorMessage.split('\n')[0]}`);
          try {
            backend = await createRvfBackend(config);
            await (backend as unknown as { initialize(): Promise<void> }).initialize();
            console.error(`[AgentDB] Using RVF backend (${detection.rvf.node ? 'N-API' : 'WASM'} fallback)`);
            return backend;
          } catch {
            // RVF also failed, try HNSWLib
          }
        }

        // Try HNSWLib as next fallback
        if (detection.hnswlib) {
          console.error('[AgentDB] Falling back to HNSWLib');
          backend = await createHNSWLibBackend(config);
          console.error('[AgentDB] Using HNSWLib backend (fallback)');
        } else if (detection.sqljsRvf) {
          console.error('[AgentDB] Falling back to sql.js RVF backend');
          backend = await createSqlJsRvfBackend(config);
          console.error('[AgentDB] Using sql.js RVF backend (built-in fallback)');
        } else {
          throw error;
        }
      }
    } else if (detection.rvf.sdk) {
      backend = await createRvfBackend(config);
      console.error(`[AgentDB] Using RVF backend (${detection.rvf.node ? 'N-API native' : 'WASM'})`);
    } else if (detection.hnswlib) {
      backend = await createHNSWLibBackend(config);
      console.error('[AgentDB] Using HNSWLib backend (fallback)');
    } else if (detection.sqljsRvf) {
      backend = await createSqlJsRvfBackend(config);
      console.error('[AgentDB] Using sql.js RVF backend (built-in)');
    } else {
      throw new Error(
        'No vector backend available.\n' +
        'Install one of:\n' +
        '  - npm install @ruvector/core (recommended)\n' +
        '  - npm install @ruvector/rvf (single-file format)\n' +
        '  - npm install hnswlib-node (fallback)'
      );
    }
  }

  // Initialize the backend (if not already initialized)
  try {
    await (backend as unknown as { initialize(): Promise<void> }).initialize();
  } catch (error) {
    if (!(error as Error).message.includes('already initialized')) {
      throw error;
    }
  }

  return backend;
}

/**
 * Get recommended backend type based on environment
 */
export async function getRecommendedBackend(): Promise<BackendType> {
  const detection = await detectBackends();

  if (detection.ruvector.core) {
    return 'ruvector';
  } else if (detection.rvf.sdk) {
    return 'rvf';
  } else if (detection.hnswlib) {
    return 'hnswlib';
  } else {
    return 'auto';
  }
}

/**
 * Check if a specific backend is available
 */
export async function isBackendAvailable(backend: 'ruvector' | 'rvf' | 'hnswlib'): Promise<boolean> {
  const detection = await detectBackends();

  if (backend === 'ruvector') {
    return detection.ruvector.core;
  }
  if (backend === 'rvf') {
    return detection.rvf.sdk;
  }

  return detection.hnswlib;
}

/**
 * Get installation instructions for a backend
 */
export function getInstallCommand(backend: 'ruvector' | 'rvf' | 'hnswlib'): string {
  if (backend === 'ruvector') return 'npm install ruvector';
  if (backend === 'rvf') return 'npm install @ruvector/rvf @ruvector/rvf-node';
  return 'npm install hnswlib-node';
}
