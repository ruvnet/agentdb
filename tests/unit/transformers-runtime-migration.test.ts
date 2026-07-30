import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('Transformers.js runtime migration', () => {
  it('uses the maintained runtime without the legacy onnx-proto chain', () => {
    const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
    const packageLock = readFileSync('package-lock.json', 'utf8');

    expect(packageJson.optionalDependencies['@huggingface/transformers']).toBe('^4.2.0');
    expect(packageJson.optionalDependencies['@xenova/transformers']).toBeUndefined();
    expect(packageLock).not.toContain('node_modules/@xenova/transformers');
    expect(packageLock).not.toContain('node_modules/onnx-proto');
  });
});
