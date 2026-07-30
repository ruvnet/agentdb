#!/usr/bin/env node
/**
 * Build script: Pack Xenova/all-MiniLM-L6-v2 model files into a single .rvf file.
 *
 * Usage:
 *   node scripts/build-model-rvf.mjs [--source <dir>] [--output <path>]
 *
 * Defaults:
 *   --source: auto-detected from node_modules or ~/.cache/huggingface
 *   --output: dist/models/all-MiniLM-L6-v2.rvf
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.join(__dirname, '..');

const MODEL_ID = 'all-MiniLM-L6-v2';
const MODEL_FILES = [
  'onnx/model_quantized.onnx',
  'tokenizer.json',
  'config.json',
  'tokenizer_config.json',
];

// Parse CLI args
const args = process.argv.slice(2);
let sourceDir = null;
let outputPath = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--source' && args[i + 1]) sourceDir = args[++i];
  if (args[i] === '--output' && args[i + 1]) outputPath = args[++i];
}

// Auto-detect source directory
function findModelSource() {
  if (sourceDir) return sourceDir;

  const candidates = [
    path.join(packageRoot, 'node_modules', '@huggingface', 'transformers', '.cache', 'Xenova', MODEL_ID),
    path.join(packageRoot, 'node_modules', '@xenova', 'transformers', '.cache', 'Xenova', MODEL_ID),
    path.join(process.env.HOME || '', '.cache', 'huggingface', 'hub', 'Xenova', MODEL_ID),
    path.join(process.env.HOME || '', '.cache', 'huggingface', 'Xenova', MODEL_ID),
    // Transformers.js v3 cache location
    path.join(process.env.HOME || '', '.cache', 'onnx', 'Xenova', MODEL_ID),
  ];

  for (const dir of candidates) {
    const onnxPath = path.join(dir, 'onnx', 'model_quantized.onnx');
    if (fs.existsSync(onnxPath)) {
      console.log(`Found model at: ${dir}`);
      return dir;
    }
  }

  return null;
}

async function main() {
  const source = findModelSource();
  if (!source) {
    console.error('Error: Model source not found.');
    console.error('');
    console.error('Please download the model first:');
    console.error('  1. Run AgentDB with embeddings once (it auto-downloads), or');
    console.error('  2. Specify --source <path-to-model-dir>');
    console.error('');
    console.error('Expected files in source dir:');
    for (const f of MODEL_FILES) {
      console.error(`  ${f}`);
    }
    process.exit(1);
  }

  // Verify all files exist
  const missingFiles = MODEL_FILES.filter(f => !fs.existsSync(path.join(source, f)));
  if (missingFiles.length > 0) {
    console.error('Error: Missing model files in source directory:');
    for (const f of missingFiles) {
      console.error(`  ${f}`);
    }
    process.exit(1);
  }

  const output = outputPath || path.join(packageRoot, 'dist', 'models', `${MODEL_ID}.rvf`);
  const outputDir = path.dirname(output);
  fs.mkdirSync(outputDir, { recursive: true });

  console.log(`Packing model into .rvf: ${output}`);
  console.log(`Source: ${source}`);
  console.log('');

  // Load sql.js
  const sqlJsModule = await import('sql.js');
  const SQL = await sqlJsModule.default();
  const db = new SQL.Database();

  // Create schema
  db.run(`
    CREATE TABLE model_assets (
      filename TEXT PRIMARY KEY,
      content  BLOB NOT NULL,
      size     INTEGER NOT NULL,
      sha256   TEXT NOT NULL
    )
  `);
  db.run(`
    CREATE TABLE model_meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )
  `);

  // Insert metadata
  db.run("INSERT INTO model_meta (key, value) VALUES ('model_id', ?)", [`Xenova/${MODEL_ID}`]);
  db.run("INSERT INTO model_meta (key, value) VALUES ('dimension', '384')");
  db.run("INSERT INTO model_meta (key, value) VALUES ('format_version', '1')");
  db.run("INSERT INTO model_meta (key, value) VALUES ('created_at', ?)", [new Date().toISOString()]);

  // Pack each file
  let totalSize = 0;
  for (const filename of MODEL_FILES) {
    const filePath = path.join(source, filename);
    const content = fs.readFileSync(filePath);
    const sha256 = crypto.createHash('sha256').update(content).digest('hex');

    db.run(
      'INSERT INTO model_assets (filename, content, size, sha256) VALUES (?, ?, ?, ?)',
      [filename, content, content.length, sha256]
    );

    const sizeKB = (content.length / 1024).toFixed(1);
    console.log(`  ${filename.padEnd(40)} ${sizeKB.padStart(10)} KB  sha256:${sha256.slice(0, 12)}...`);
    totalSize += content.length;
  }

  // Export database to file
  const data = db.export();
  const buffer = Buffer.from(data);
  fs.writeFileSync(output, buffer);
  db.close();

  const rvfSizeKB = (buffer.length / 1024).toFixed(1);
  const rvfSizeMB = (buffer.length / (1024 * 1024)).toFixed(1);
  console.log('');
  console.log(`Total source size: ${(totalSize / (1024 * 1024)).toFixed(1)} MB`);
  console.log(`RVF file size:     ${rvfSizeMB} MB (${rvfSizeKB} KB)`);
  console.log(`Output:            ${output}`);
  console.log('');
  console.log('Done! Add dist/models/ to your package.json "files" array.');
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
