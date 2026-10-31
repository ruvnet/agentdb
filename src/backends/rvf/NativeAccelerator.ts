/**
 * NativeAccelerator - ADR-007 Phase 1 Capability Bridge
 *
 * Lazy-loads @ruvector APIs and provides accelerated alternatives to JS
 * implementations. All imports are lazy/optional with JS fallbacks.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { WasmStoreBridge } from './WasmStoreBridge.js';
import * as fb from './SimdFallbacks.js';
export { WasmStoreBridge } from './WasmStoreBridge.js';
export type { WasmStoreQueryResult } from './WasmStoreBridge.js';

export interface AcceleratorStats {
  simdAvailable: boolean;
  simdActivationsAvailable: boolean;
  wasmVerifyAvailable: boolean;
  wasmStoreAvailable: boolean;
  wasmQuantizationAvailable: boolean;
  nativeInfoNceAvailable: boolean;
  nativeAdamWAvailable: boolean;
  nativeTensorCompressAvailable: boolean;
  routerPersistAvailable: boolean;
  sonaExtendedAvailable: boolean;
  capabilities: string[];
}

export interface WitnessVerifyResult { valid: boolean; entryCount: number; }
export interface SegmentVerifyResult { valid: boolean; crc: number; }

const MAX_DIM = 4096;

/**
 * AdaptiveIndexTuner level numbers -> @ruvector/gnn compression configs.
 * Only 'half' (level 1) is delegated: native pq8/pq4/binary train a codebook
 * per vector, which reconstructs far worse than the JS fallbacks
 * (cosine distance ~0.88 vs < 0.1 in tests/backends/adaptive-index-tuner).
 */
const TENSOR_LEVELS: Record<number, Record<string, unknown>> = {
  1: { levelType: 'half', scale: 1.0 },
};

export class NativeAccelerator {
  private _simd: any = null;
  private _witnessVerify: ((chain: Uint8Array) => boolean) | null = null;
  private _witnessCount: ((chain: Uint8Array) => number) | null = null;
  private _verifyHeader: ((data: Uint8Array) => boolean) | null = null;
  private _crc32c: ((data: Uint8Array) => number) | null = null;
  private _infoNceLoss: any = null;
  private _adamWOptimizer: any = null;
  private _tensorCompress: any = null;
  private _routerSave: ((router: any, path: string) => Promise<void>) | null = null;
  private _routerLoad: ((path: string, config: { dimension: number }) => Promise<any>) | null = null;
  private _sonaFlush = false;
  private _sonaContext = false;
  private _sonaBaseLora = false;
  private _graphTx = false;
  private _graphBatchInsert = false;
  private _graphCypher = false;
  private _coreBatchInsert = false;
  private _ewcManager: any = null;
  private _sqLoad: ((params: Uint8Array, dim: number) => boolean) | null = null;
  private _dequantI8: ((src: Uint8Array, dst: Float32Array, count: number) => boolean) | null = null;
  private _pqLoadCodebook: ((codebook: Uint8Array, m: number, k: number) => boolean) | null = null;
  private _pqDistances: ((codes: Uint8Array, count: number) => Float32Array) | null = null;
  private _wasmStore: WasmStoreBridge = new WasmStoreBridge();
  private _initialized = false;

  async initialize(): Promise<AcceleratorStats> {
    if (this._initialized) return this.getStats();
    await Promise.allSettled([
      this.loadSimd(), this.loadWasmVerify(), this.loadWasmQuantization(),
      this.loadWasmStore(), this.loadNativeAttention(), this.loadNativeTensorCompress(),
      this.loadRouterPersistence(), this.loadSonaExtended(),
      this.loadGraphCapabilities(), this.loadCoreBatch(), this.loadEwcManager(),
    ]);
    this._initialized = true;
    return this.getStats();
  }

  // ─── SIMD Vector Math ───

  get simdAvailable(): boolean { return this._simd !== null; }

  cosineSimilarity(a: Float32Array, b: Float32Array): number {
    this.validateDims(a, b);
    if (this._simd) { try { return this._simd.cosineSimilarity(a, b); } catch { /* fallback */ } }
    return fb.jsCosineSimilarity(a, b);
  }

  dotProduct(a: Float32Array, b: Float32Array): number {
    this.validateDims(a, b);
    if (this._simd) { try { return this._simd.dotProduct(a, b); } catch { /* fallback */ } }
    return fb.jsDotProduct(a, b);
  }

  l2Distance(a: Float32Array, b: Float32Array): number {
    this.validateDims(a, b);
    if (this._simd) { try { return this._simd.l2Distance(a, b); } catch { /* fallback */ } }
    return fb.jsL2Distance(a, b);
  }

  hammingDistance(a: Uint8Array, b: Uint8Array): number {
    if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
    if (this._simd) { try { return this._simd.hammingDistance(a, b); } catch { /* fallback */ } }
    return fb.jsHammingDistance(a, b);
  }

  // ─── SIMD Activations & Element-wise Ops ───

  get simdActivationsAvailable(): boolean { return this._simd !== null; }

  matvec(matrix: number[][], vector: number[]): number[] {
    if (this._simd) { try { return this._simd.matvec(matrix, vector); } catch { /* fallback */ } }
    return fb.jsMatvec(matrix, vector);
  }

  softmax(input: number[]): number[] {
    if (this._simd) { try { return this._simd.softmax(input); } catch { /* fallback */ } }
    return fb.jsSoftmax(input);
  }

  relu(input: number[]): number[] {
    if (this._simd) { try { return this._simd.relu(input); } catch { /* fallback */ } }
    return fb.jsRelu(input);
  }

  gelu(input: number[]): number[] {
    if (this._simd) { try { return this._simd.gelu(input); } catch { /* fallback */ } }
    return fb.jsGelu(input);
  }

  sigmoid(input: number[]): number[] {
    if (this._simd) { try { return this._simd.sigmoid(input); } catch { /* fallback */ } }
    return fb.jsSigmoid(input);
  }

  layerNorm(input: number[], eps: number = 1e-5): number[] {
    if (this._simd) { try { return this._simd.layerNorm(input, eps); } catch { /* fallback */ } }
    return fb.jsLayerNorm(input, eps);
  }

  add(a: number[], b: number[]): number[] {
    if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
    if (this._simd) { try { return this._simd.add(a, b); } catch { /* fallback */ } }
    return fb.jsAdd(a, b);
  }

  mul(a: number[], b: number[]): number[] {
    if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
    if (this._simd) { try { return this._simd.mul(a, b); } catch { /* fallback */ } }
    return fb.jsMul(a, b);
  }

  scale(a: number[], scalar: number): number[] {
    if (this._simd) { try { return this._simd.scale(a, scalar); } catch { /* fallback */ } }
    return fb.jsScale(a, scalar);
  }

  normalizeVec(a: number[]): number[] {
    if (this._simd) { try { return this._simd.normalize(a); } catch { /* fallback */ } }
    return fb.jsNormalizeVec(a);
  }

  // ─── WASM Quantization Bridge ───

  get wasmQuantizationAvailable(): boolean { return this._sqLoad !== null; }

  loadSqParams(params: Uint8Array, dim: number): boolean {
    if (this._sqLoad) { try { return this._sqLoad(params, dim); } catch { /* fallback */ } }
    return false;
  }

  dequantI8(src: Uint8Array, dst: Float32Array, count: number): boolean {
    if (this._dequantI8) { try { return this._dequantI8(src, dst, count); } catch { /* fallback */ } }
    return false;
  }

  loadPqCodebook(codebook: Uint8Array, m: number, k: number): boolean {
    if (this._pqLoadCodebook) { try { return this._pqLoadCodebook(codebook, m, k); } catch { /* fallback */ } }
    return false;
  }

  pqDistances(codes: Uint8Array, count: number): Float32Array | null {
    if (this._pqDistances) { try { return this._pqDistances(codes, count); } catch { /* fallback */ } }
    return null;
  }

  // ─── WASM Store Bridge ───

  get wasmStoreAvailable(): boolean { return this._wasmStore.available; }
  get wasmStore(): WasmStoreBridge { return this._wasmStore; }

  wasmStoreCreate(dim: number, metric: number): number | null {
    return this._wasmStore.wasmStoreCreate(dim, metric);
  }
  wasmStoreIngest(handle: number, vecs: Float32Array, ids: number[], count: number): number {
    return this._wasmStore.wasmStoreIngest(handle, vecs, ids, count);
  }
  wasmStoreQuery(handle: number, query: Float32Array, k: number, metric: number): { id: number; distance: number }[] | null {
    return this._wasmStore.wasmStoreQuery(handle, query, k, metric);
  }
  wasmStoreExport(handle: number): Uint8Array | null {
    return this._wasmStore.wasmStoreExport(handle);
  }
  wasmStoreClose(handle: number): boolean {
    return this._wasmStore.wasmStoreClose(handle);
  }

  // ─── WASM Verification ───

  get wasmVerifyAvailable(): boolean { return this._witnessVerify !== null; }

  verifyWitnessChain(chain: Uint8Array): WitnessVerifyResult {
    if (!chain || chain.length === 0) return { valid: false, entryCount: 0 };
    const entryCount = Math.floor(chain.length / 73);
    if (this._witnessVerify && this._witnessCount) {
      try {
        return { valid: this._witnessVerify(chain), entryCount: this._witnessCount(chain) };
      } catch { /* fallback */ }
    }
    return { valid: chain.length > 0 && chain.length % 73 === 0, entryCount };
  }

  verifySegmentHeader(data: Uint8Array): SegmentVerifyResult {
    if (!data || data.length < 4) return { valid: false, crc: 0 };
    if (this._verifyHeader && this._crc32c) {
      try {
        return { valid: this._verifyHeader(data), crc: this._crc32c(data) };
      } catch { /* fallback */ }
    }
    const valid = data[0] === 0x52 && data[1] === 0x56 && data[2] === 0x46;
    return { valid, crc: fb.jsCrc32c(data) };
  }

  // ─── Native InfoNCE Loss ───

  get nativeInfoNceAvailable(): boolean { return this._infoNceLoss !== null; }

  infoNceLoss(anchor: Float32Array, positive: Float32Array, negatives: Float32Array[], temperature: number): number {
    if (this._infoNceLoss) {
      try { return this._infoNceLoss.compute(anchor, positive, negatives, temperature); } catch { /* fallback */ }
    }
    return fb.jsInfoNceLoss(anchor, positive, negatives, temperature);
  }

  // ─── Native AdamW Optimizer ───

  get nativeAdamWAvailable(): boolean { return this._adamWOptimizer !== null; }

  adamWStep(params: Float32Array, grads: Float32Array, m: Float32Array, v: Float32Array, step: number, lr: number, wd: number): void {
    if (this._adamWOptimizer) {
      try { this._adamWOptimizer.step(params, grads, m, v, step, lr, wd); return; } catch { /* fallback */ }
    }
    fb.jsAdamWStep(params, grads, m, v, step, lr, wd);
  }

  // ─── Native Tensor Compression ───

  get nativeTensorCompressAvailable(): boolean { return this._tensorCompress !== null; }

  /**
   * Compress with @ruvector/gnn at a numeric level (see TENSOR_LEVELS).
   * Returns the native compressed JSON as UTF-8 bytes, or null when the
   * level is not delegated or the package is unavailable.
   */
  tensorCompress(vec: Float32Array, level: number): Uint8Array | null {
    const config = TENSOR_LEVELS[level];
    if (this._tensorCompress && config) {
      try {
        const json: string = this._tensorCompress.compressWithLevel(Array.from(vec), config);
        return new TextEncoder().encode(json);
      } catch { /* fallback */ }
    }
    return null;
  }

  tensorDecompress(compressed: Uint8Array, dim: number): Float32Array | null {
    if (this._tensorCompress) {
      try {
        const out = this._tensorCompress.decompress(new TextDecoder().decode(compressed));
        const vec = out instanceof Float32Array ? out : Float32Array.from(out);
        if (vec.length === dim) return vec;
      } catch { /* fallback */ }
    }
    return null;
  }

  /** No native batch API exists; compresses each vector in turn. */
  tensorBatchCompress(vecs: Float32Array[], level: number): Uint8Array[] | null {
    if (!this._tensorCompress) return null;
    const out: Uint8Array[] = [];
    for (const vec of vecs) {
      const compressed = this.tensorCompress(vec, level);
      if (!compressed) return null;
      out.push(compressed);
    }
    return out;
  }

  // ─── Router Persistence ───

  get routerPersistAvailable(): boolean { return this._routerSave !== null; }

  async routerSave(router: any, path: string): Promise<boolean> {
    if (this._routerSave) { try { await this._routerSave(router, path); return true; } catch { /* fallback */ } }
    return false;
  }

  /** Load a saved @ruvector/router SemanticRouter (load() is an instance method). */
  async routerLoad(path: string, config: { dimension: number } = { dimension: 384 }): Promise<any | null> {
    if (this._routerLoad) { try { return await this._routerLoad(path, config); } catch { /* fallback */ } }
    return null;
  }

  // ─── SONA Extended ───

  get sonaExtendedAvailable(): boolean { return this._sonaFlush || this._sonaContext || this._sonaBaseLora; }

  sonaAddContext(engine: any, trajectoryId: number, context: Record<string, unknown>): boolean {
    if (this._sonaContext && engine) {
      try { engine.addTrajectoryContext(trajectoryId, JSON.stringify(context)); return true; } catch { /* skip */ }
    }
    return false;
  }

  sonaFlush(engine: any): boolean {
    if (this._sonaFlush && engine) { try { engine.flush(); return true; } catch { /* skip */ } }
    return false;
  }

  sonaApplyBaseLora(engine: any, loraData: Float32Array): boolean {
    if (this._sonaBaseLora && engine) {
      try { engine.applyBaseLora(Array.from(loraData)); return true; } catch { /* skip */ }
    }
    return false;
  }

  // ─── Graph Transactions (ADR-007 Phase 1) ───

  get graphTxAvailable(): boolean { return this._graphTx; }
  get graphBatchInsertAvailable(): boolean { return this._graphBatchInsert; }
  get graphCypherAvailable(): boolean { return this._graphCypher; }

  /**
   * Run fn inside a @ruvector/graph-node transaction
   * (db.begin() -> txId, db.commit(txId) / db.rollback(txId)).
   */
  async graphTransaction(db: any, fn: (tx: any) => void | Promise<void>): Promise<boolean> {
    if (this._graphTx && typeof db?.begin === 'function') {
      const txId = await db.begin();
      try {
        await fn(db);
        await db.commit(txId);
        return true;
      } catch {
        try { await db.rollback(txId); } catch { /* already failed */ }
        return false;
      }
    }
    try { await fn(db); return true; } catch { return false; }
  }

  /** db.batchInsert() (async in @ruvector/graph-node) */
  async graphBatchInsertNodes(db: any, batch: unknown): Promise<boolean> {
    if (this._graphBatchInsert && typeof db?.batchInsert === 'function') {
      try { await db.batchInsert(batch); return true; } catch { /* fallback */ }
    }
    return false;
  }

  /** Cypher via db.query() (async in @ruvector/graph-node; it takes no params) */
  async graphCypherQuery(db: any, query: string): Promise<any | null> {
    if (this._graphCypher && typeof db?.query === 'function') {
      try { return await db.query(query); } catch { /* fallback */ }
    }
    return null;
  }

  // ─── Core Batch Operations (ADR-007 Phase 1) ───

  get coreBatchInsertAvailable(): boolean { return this._coreBatchInsert; }

  /** VectorDb.insertBatch() (async in @ruvector/core; metadata is a JSON string there) */
  async coreBatchInsert(
    vectorDb: any,
    items: Array<{ id: string; vector: Float32Array; metadata?: Record<string, unknown> }>
  ): Promise<boolean> {
    if (this._coreBatchInsert && typeof vectorDb?.insertBatch === 'function') {
      try {
        await vectorDb.insertBatch(items.map((item) => ({
          id: item.id,
          vector: item.vector,
          metadata: item.metadata ? JSON.stringify(item.metadata) : undefined,
        })));
        return true;
      } catch { /* fallback */ }
    }
    return false;
  }

  // ─── EWC Memory Protection (ADR-007 Phase 1) ───

  get ewcManagerAvailable(): boolean { return this._ewcManager !== null; }

  ewcPenalty(params: Float32Array): number {
    if (this._ewcManager) { try { return this._ewcManager.computePenalty(Array.from(params)); } catch { /* fallback */ } }
    return 0;
  }

  ewcUpdateFisher(params: Float32Array, importance: number): boolean {
    if (this._ewcManager) { try { this._ewcManager.updateFisher(Array.from(params), importance); return true; } catch { /* fallback */ } }
    return false;
  }

  // ─── Stats ───

  getStats(): AcceleratorStats {
    const capabilities: string[] = [];
    if (this._simd) { capabilities.push('simd', 'simd-activations'); }
    if (this._witnessVerify) capabilities.push('wasm-verify');
    if (this._wasmStore.available) capabilities.push('wasm-store');
    if (this._sqLoad) capabilities.push('wasm-quantization');
    if (this._infoNceLoss) capabilities.push('native-infonce');
    if (this._adamWOptimizer) capabilities.push('native-adamw');
    if (this._tensorCompress) capabilities.push('native-tensor-compress');
    if (this._routerSave) capabilities.push('router-persist');
    if (this._sonaFlush || this._sonaContext || this._sonaBaseLora) capabilities.push('sona-extended');
    if (this._graphTx) capabilities.push('graph-tx');
    if (this._graphBatchInsert) capabilities.push('graph-batch');
    if (this._graphCypher) capabilities.push('graph-cypher');
    if (this._coreBatchInsert) capabilities.push('core-batch');
    if (this._ewcManager) capabilities.push('ewc-manager');
    return {
      simdAvailable: this._simd !== null,
      simdActivationsAvailable: this._simd !== null,
      wasmVerifyAvailable: this._witnessVerify !== null,
      wasmStoreAvailable: this._wasmStore.available,
      wasmQuantizationAvailable: this._sqLoad !== null,
      nativeInfoNceAvailable: this._infoNceLoss !== null,
      nativeAdamWAvailable: this._adamWOptimizer !== null,
      nativeTensorCompressAvailable: this._tensorCompress !== null,
      routerPersistAvailable: this._routerSave !== null,
      sonaExtendedAvailable: this._sonaFlush || this._sonaContext || this._sonaBaseLora,
      capabilities,
    };
  }

  /**
   * Which native features actually loaded, by name. Unlike getStats(), this
   * covers every optional package probed in initialize().
   */
  capabilities(): Record<string, boolean> {
    return {
      simd: this._simd !== null,
      wasmVerify: this._witnessVerify !== null,
      wasmStore: this._wasmStore.available,
      wasmQuantization: this._sqLoad !== null,
      nativeInfoNce: this._infoNceLoss !== null,
      nativeAdamW: this._adamWOptimizer !== null,
      nativeTensorCompress: this._tensorCompress !== null,
      routerPersist: this._routerSave !== null && this._routerLoad !== null,
      sonaExtended: this._sonaFlush || this._sonaContext || this._sonaBaseLora,
      graphTx: this._graphTx,
      graphBatchInsert: this._graphBatchInsert,
      graphCypher: this._graphCypher,
      coreBatchInsert: this._coreBatchInsert,
      ewcManager: this._ewcManager !== null,
    };
  }

  // ─── Loaders ───

  private async loadSimd(): Promise<void> {
    try { const { SimdOps } = await import('@ruvector/ruvllm'); if (SimdOps) this._simd = new SimdOps(); } catch { /* not available */ }
  }

  private async loadWasmVerify(): Promise<void> {
    try {
      const wasm: any = await import('@ruvector/rvf-wasm');
      if (wasm.rvf_witness_verify) this._witnessVerify = wasm.rvf_witness_verify;
      if (wasm.rvf_witness_count) this._witnessCount = wasm.rvf_witness_count;
      if (wasm.rvf_verify_header) this._verifyHeader = wasm.rvf_verify_header;
      if (wasm.rvf_crc32c) this._crc32c = wasm.rvf_crc32c;
    } catch { /* not available */ }
  }

  private async loadWasmQuantization(): Promise<void> {
    try {
      const wasm: any = await import('@ruvector/rvf-wasm');
      if (wasm.rvf_sq_load_params) this._sqLoad = wasm.rvf_sq_load_params;
      if (wasm.rvf_dequant_i8) this._dequantI8 = wasm.rvf_dequant_i8;
      if (wasm.rvf_pq_load_codebook) this._pqLoadCodebook = wasm.rvf_pq_load_codebook;
      if (wasm.rvf_pq_distances) this._pqDistances = wasm.rvf_pq_distances;
    } catch { /* not available */ }
  }

  private async loadWasmStore(): Promise<void> { await this._wasmStore.initialize(); }

  private async loadNativeAttention(): Promise<void> {
    try {
      const attn = await import('@ruvector/attention');
      if (attn.InfoNceLoss) this._infoNceLoss = new attn.InfoNceLoss();
      if (attn.AdamWOptimizer) this._adamWOptimizer = new attn.AdamWOptimizer();
    } catch { /* not available */ }
  }

  private async loadNativeTensorCompress(): Promise<void> {
    try {
      const gnnMod: any = await import('@ruvector/gnn');
      // compress/decompress are instance methods. Use the raw binding: the
      // JS TensorCompress wrapper passes a Float32Array where the native
      // method requires a plain array, so every call through it throws.
      const Ctor = gnnMod.NativeTensorCompress ?? gnnMod.default?.NativeTensorCompress;
      if (typeof Ctor?.prototype?.compressWithLevel === 'function') this._tensorCompress = new Ctor();
    } catch { /* not available */ }
  }

  private async loadRouterPersistence(): Promise<void> {
    try {
      const router: any = await import('@ruvector/router');
      const SemanticRouter = router.SemanticRouter ?? router.default?.SemanticRouter;
      // save() and load() are both instance methods
      if (typeof SemanticRouter?.prototype?.save === 'function' && typeof SemanticRouter.prototype.load === 'function') {
        this._routerSave = async (r: any, p: string) => r.save(p);
        this._routerLoad = async (p: string, config: { dimension: number }) => {
          const r = new SemanticRouter(config);
          await r.load(p);
          return r;
        };
      }
    } catch { /* not available */ }
  }

  private async loadSonaExtended(): Promise<void> {
    try {
      const sonaMod: any = await import('@ruvector/sona');
      if (sonaMod.SonaEngine?.prototype?.flush) this._sonaFlush = true;
      if (sonaMod.SonaEngine?.prototype?.addTrajectoryContext) this._sonaContext = true;
      if (sonaMod.SonaEngine?.prototype?.applyBaseLora) this._sonaBaseLora = true;
    } catch { /* not available */ }
  }

  private async loadGraphCapabilities(): Promise<void> {
    try {
      const graphMod: any = await import('@ruvector/graph-node');
      const proto = (graphMod.GraphDatabase ?? graphMod.default?.GraphDatabase)?.prototype;
      if (typeof proto?.begin === 'function' && typeof proto.commit === 'function') this._graphTx = true;
      if (typeof proto?.batchInsert === 'function') this._graphBatchInsert = true;
      if (typeof proto?.query === 'function') this._graphCypher = true;
    } catch { /* not available */ }
  }

  private async loadCoreBatch(): Promise<void> {
    try {
      const coreMod: any = await import('@ruvector/core');
      const VectorDb = coreMod.VectorDb ?? coreMod.default?.VectorDb;
      if (typeof VectorDb?.prototype?.insertBatch === 'function') this._coreBatchInsert = true;
    } catch { /* not available */ }
  }

  private async loadEwcManager(): Promise<void> {
    try { const { EwcManager } = await import('@ruvector/ruvllm'); if (EwcManager) this._ewcManager = new EwcManager(); } catch { /* not available */ }
  }

  private validateDims(a: Float32Array, b: Float32Array): void {
    if (a.length !== b.length) throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
    if (a.length > MAX_DIM) throw new Error(`Dimension exceeds maximum: ${a.length} > ${MAX_DIM}`);
  }
}

let _globalAccelerator: NativeAccelerator | null = null;
let _initPromise: Promise<NativeAccelerator> | null = null;

export async function getAccelerator(): Promise<NativeAccelerator> {
  if (_globalAccelerator) return _globalAccelerator;
  if (!_initPromise) {
    _initPromise = (async () => {
      const accel = new NativeAccelerator();
      await accel.initialize();
      _globalAccelerator = accel;
      return accel;
    })();
  }
  return _initPromise;
}

export function resetAccelerator(): void {
  _globalAccelerator = null;
  _initPromise = null;
}
