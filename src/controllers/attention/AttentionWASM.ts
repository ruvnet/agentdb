/**
 * AttentionWASM - WASM/NAPI module management
 *
 * Handles:
 * - Module loading (WASM/NAPI)
 * - Runtime detection
 * - Warm-up
 * - Fallback handling
 */

/**
 * NAPI Attention Module Interface
 */
export interface NAPIAttentionModule {
  multiHeadAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, mask?: Float32Array): { output: Float32Array; weights?: Float32Array };
  flashAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, mask?: Float32Array): Float32Array;
  flashAttentionV2?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, options: any): { output: Float32Array; speedup?: number; baselineTimeMs?: number };
  linearAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number): Float32Array;
  hyperbolicAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, curvature: number): Float32Array;
  moeAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, numExperts: number, topK: number, mask?: Float32Array): Float32Array;
}

/**
 * WASM Attention Module Interface
 */
export interface WASMAttentionModule {
  default(): Promise<void>;
  dispose?(): Promise<void>;
  multiHeadAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, mask?: Float32Array): { output: Float32Array; weights?: Float32Array };
  flashAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, mask?: Float32Array): Float32Array;
  flashAttentionV2?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, options: any): { output: Float32Array; speedup?: number; baselineTimeMs?: number };
  linearAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number): Float32Array;
  hyperbolicAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, curvature: number): Float32Array;
  moeAttention?(query: Float32Array, key: Float32Array, value: Float32Array, numHeads: number, headDim: number, numExperts: number, topK: number, mask?: Float32Array): Float32Array;
}

/**
 * Global WASM instance cache (shared across all AttentionService instances)
 * Prevents re-initialization overhead (2-5s → <10ms cold start)
 */
const wasmInstanceCache = new Map<string, WASMAttentionModule>();

/**
 * Runtime environment detection
 */
export type RuntimeEnvironment = 'nodejs' | 'browser' | 'unknown';

/**
 * Detect the current runtime environment
 */
function detectRuntime(): RuntimeEnvironment {
  // Check for Node.js
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    return 'nodejs';
  }

  // Check for browser (with proper type guards)
  if (typeof globalThis !== 'undefined') {
    const global = globalThis as any;
    if (typeof global.window !== 'undefined' && typeof global.document !== 'undefined') {
      return 'browser';
    }
  }

  return 'unknown';
}

/**
 * AttentionWASMManager - Manages WASM/NAPI module loading
 */
export class AttentionWASMManager {
  private runtime: RuntimeEnvironment;
  private napiModule: NAPIAttentionModule | null = null;
  private wasmModule: WASMAttentionModule | null = null;

  constructor() {
    this.runtime = detectRuntime();
  }

  /**
   * Initialize and load appropriate modules
   */
  async initialize(): Promise<void> {
    if (this.runtime === 'nodejs') {
      await this.loadNAPIModule();
    } else if (this.runtime === 'browser') {
      await this.loadWASMModule();
    } else {
      console.warn('⚠️  Unknown runtime environment, using fallback implementation');
    }
  }

  /**
   * Load NAPI module for Node.js runtime
   */
  private async loadNAPIModule(): Promise<void> {
    try {
      // Try to import @ruvector/attention (NAPI bindings)
      // @ts-expect-error - Optional dependency
      this.napiModule = await import('@ruvector/attention');
      console.error('✅ Loaded @ruvector/attention NAPI module');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.warn(`⚠️  Failed to load @ruvector/attention: ${errorMessage}`);
      console.warn('   Falling back to JavaScript implementation');
      this.napiModule = null;
    }
  }

  /**
   * Load WASM module for browser runtime with caching
   * Uses global cache to share instances across AttentionService instances
   */
  private async loadWASMModule(): Promise<void> {
    const cacheKey = 'ruvector-attention-wasm';

    // Check cache first (optimization: 2-5s → <10ms)
    if (wasmInstanceCache.has(cacheKey)) {
      this.wasmModule = wasmInstanceCache.get(cacheKey)!;
      console.error('✅ Loaded WASM from cache (<10ms)');
      return;
    }

    try {
      // Try to import ruvector-attention-wasm
      const mod = await import('ruvector-attention-wasm').catch(() => null);

      if (!mod) {
        throw new Error('WASM module not available');
      }

      // Initialize WASM once
      if (typeof mod.default === 'function') {
        await mod.default();
      }

      this.wasmModule = mod as unknown as WASMAttentionModule;
      wasmInstanceCache.set(cacheKey, mod as unknown as WASMAttentionModule);

      console.error('✅ Loaded and cached WASM module');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.warn(`⚠️  Failed to load ruvector-attention-wasm: ${errorMessage}`);
      console.warn('   Falling back to JavaScript implementation');
      this.wasmModule = null;
    }
  }

  /**
   * Dispose WASM module
   */
  async dispose(): Promise<void> {
    if (this.wasmModule && typeof this.wasmModule.dispose === 'function') {
      await this.wasmModule.dispose();
    }
    this.napiModule = null;
    this.wasmModule = null;
  }

  /**
   * Get runtime environment
   */
  getRuntime(): RuntimeEnvironment {
    return this.runtime;
  }

  /**
   * Get NAPI module
   */
  getNAPIModule(): NAPIAttentionModule | null {
    return this.napiModule;
  }

  /**
   * Get WASM module
   */
  getWASMModule(): WASMAttentionModule | null {
    return this.wasmModule;
  }

  /**
   * Check if NAPI is available
   */
  hasNAPI(): boolean {
    return this.napiModule !== null;
  }

  /**
   * Check if WASM is available
   */
  hasWASM(): boolean {
    return this.wasmModule !== null;
  }
}
