/**
 * Ambient declarations for optional dependencies.
 *
 * Each module below is an optionalDependency (or, for `semver`, a package that
 * only arrives transitively) that agentdb loads behind a runtime guard. An
 * ambient declaration takes precedence over the package's own typings, so the
 * build sees the same `any`-typed surface whether the optional install
 * succeeded, failed, or resolved a different version. That keeps `tsc` from
 * failing with TS2307 when an optional dependency is missing; every call site
 * already checks for the symbols it needs at runtime.
 *
 * Shorthand declarations (`declare module 'x';`) make every import `any`.
 * Modules whose exports are also used as types declare those exports as
 * classes. Keep the list in sync with `optionalDependencies`: CI type-checks
 * after `npm ci --omit=optional`.
 */

declare module 'better-sqlite3' {
  class Database {
    constructor(filename?: string, options?: unknown);
    [key: string]: any;
  }
  export { Database };
  export default Database;
}

declare module 'commander' {
  export class Command {
    constructor(name?: string);
    [key: string]: any;
  }
}

declare module '@ruvector/sona' {
  export class SonaEngine {
    constructor(...args: any[]);
    static [key: string]: any;
    [key: string]: any;
  }
}

declare module '@huggingface/transformers';
declare module '@opentelemetry/resources';
declare module '@opentelemetry/sdk-node';
declare module '@opentelemetry/semantic-conventions';
declare module '@ruvector/attention';
declare module '@ruvector/router';
declare module '@ruvector/ruvllm';
declare module '@ruvector/rvf';
declare module '@ruvector/rvf-node';
declare module '@ruvector/rvf-solver';
declare module '@ruvector/rvf-wasm';
declare module 'chalk';
declare module 'hnswlib-node';
declare module 'inquirer';
declare module 'ruvector';
declare module 'ruvector-attention-wasm';
declare module 'semver';
