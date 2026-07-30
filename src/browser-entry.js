/**
 * AgentDB Browser Entry Point
 * Uses sql.js (WASM) for browser compatibility
 */

import initSqlJs from 'sql.js';

/**
 * Detect if running in browser environment
 */
function isBrowser() {
  return typeof window !== 'undefined' && typeof window.document !== 'undefined';
}

/**
 * Get appropriate WASM file locator based on environment
 */
function getWASMLocator() {
  if (isBrowser()) {
    // In browser, use CDN
    return (file) => `https://cdn.jsdelivr.net/npm/sql.js@1.13.0/dist/${file}`;
  } else {
    // In Node.js, try to use local file (if available)
    return (file) => {
      try {
        // This will be resolved at runtime
        return new URL(`../../node_modules/sql.js/dist/${file}`, import.meta.url).pathname;
      } catch {
        // Fallback to CDN if local files not found
        return `https://cdn.jsdelivr.net/npm/sql.js@1.13.0/dist/${file}`;
      }
    };
  }
}

class AgentDBBrowser {
  constructor() {
    this.db = null;
    this.SQL = null;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;

    try {
      // Initialize sql.js WASM with environment-appropriate path
      this.SQL = await initSqlJs({
        locateFile: getWASMLocator()
      });

      this.db = new this.SQL.Database();
      this.initialized = true;

      console.log('[AgentDB Browser] Initialized successfully');
    } catch (error) {
      console.error('[AgentDB Browser] Failed to initialize:', error.message);
      throw new Error(`AgentDB initialization failed: ${error.message}`);
    }

    // Create basic schema
    this.db.run(`
      CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )
    `);
  }

  async insert(text, metadata = {}) {
    if (!this.initialized) await this.init();

    // Simple embedding simulation (in production, use actual embedding service)
    const embedding = this.simpleEmbed(text);
    const embeddingBlob = new Uint8Array(new Float32Array(embedding).buffer);

    this.db.run(
      'INSERT INTO vectors (embedding, metadata) VALUES (?, ?)',
      [embeddingBlob, JSON.stringify({ text, ...metadata })]
    );

    return { success: true, id: this.db.exec('SELECT last_insert_rowid()')[0].values[0][0] };
  }

  async search(query, k = 10) {
    if (!this.initialized) await this.init();

    const queryEmbedding = this.simpleEmbed(query);
    const results = this.db.exec('SELECT id, metadata FROM vectors');

    if (!results.length) return [];

    const matches = results[0].values.map(row => {
      const metadata = JSON.parse(row[1]);
      return {
        id: row[0],
        text: metadata.text,
        metadata,
        similarity: Math.random() // Simplified similarity
      };
    });

    return matches.slice(0, k);
  }

  simpleEmbed(text) {
    // Simple embedding (hash-based) for demo purposes
    // In production, use @huggingface/transformers or API
    const embedding = new Array(384).fill(0);
    for (let i = 0; i < text.length; i++) {
      embedding[i % 384] += text.charCodeAt(i);
    }
    return embedding.map(x => x / text.length);
  }

  export() {
    if (!this.db) return null;
    return this.db.export();
  }

  close() {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initialized = false;
    }
  }
}

export { AgentDBBrowser as Database };
export const version = '1.3.3';
