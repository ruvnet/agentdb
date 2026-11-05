/**
 * Lazy loader for optional ecosystem packages (ADR-018).
 *
 * The module name goes through a variable so TypeScript and bundlers do not try
 * to resolve it at build time; a missing package resolves to null instead of
 * throwing. Results are memoised per name, so a package is probed once.
 */

const cache = new Map<string, Promise<any | null>>();

export function optionalImport<T = any>(name: string): Promise<T | null> {
  let pending = cache.get(name);
  if (!pending) {
    const specifier = name;
    pending = import(/* @vite-ignore */ specifier).then(
      (mod) => mod,
      () => null
    );
    cache.set(name, pending);
  }
  return pending as Promise<T | null>;
}

/** Named export from an ESM namespace or a CommonJS module loaded through import(). */
export function pickExport<T = any>(mod: any, name: string): T | undefined {
  if (!mod) return undefined;
  return (mod[name] ?? mod.default?.[name]) as T | undefined;
}

/** Test hook: forget memoised probes. */
export function resetOptionalImportCache(): void {
  cache.clear();
}
