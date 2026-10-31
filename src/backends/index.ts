/**
 * AgentDB Backends - Unified Vector Storage Interface
 *
 * Provides automatic backend selection between RuVector and HNSWLib
 * with graceful fallback and clear error messages.
 */

// Core interfaces
export type {
  VectorBackend,
  VectorConfig,
  SearchResult,
  SearchOptions,
  VectorStats,
  VectorBackendAsync
} from './VectorBackend.js';

// Search that works for sync and async-native backends (RuVector, RVF)
export { searchVectors, isAsyncVectorBackend } from './VectorBackend.js';

// Backend implementations
export { RuVectorBackend } from './ruvector/RuVectorBackend.js';
export { RuVectorLearning } from './ruvector/RuVectorLearning.js';
export { HNSWLibBackend } from './hnswlib/HNSWLibBackend.js';

// Factory and detection
export {
  createBackend,
  detectBackends,
  getRecommendedBackend,
  isBackendAvailable,
  getInstallCommand
} from './factory.js';

export type { BackendType, BackendDetection } from './factory.js';
export type { LearningConfig, EnhancementOptions } from './ruvector/RuVectorLearning.js';
