/**
 * HybridSearch - Combined Vector + Keyword Search for AgentDB
 *
 * Features:
 * - BM25 keyword search with inverted index
 * - Vector similarity search integration
 * - Reciprocal Rank Fusion (RRF) for result combination
 * - Linear and max fusion methods
 * - Configurable weights and fusion strategies
 * - Metadata filtering support
 *
 * @module search/HybridSearch
 */

import type { VectorBackend, SearchResult, SearchOptions } from '../backends/VectorBackend.js';
import { searchVectors } from '../backends/VectorBackend.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Options for hybrid search operations
 */
export interface HybridSearchOptions {
  /** Maximum number of results to return */
  limit: number;

  /** Weight for vector search results (default: 0.7) */
  vectorWeight?: number;

  /** Weight for keyword search results (default: 0.3) */
  keywordWeight?: number;

  /** Metadata filter to apply to results */
  filter?: Record<string, any>;

  /** Fusion method for combining results */
  fusionMethod?: 'rrf' | 'linear' | 'max';

  /** RRF k parameter (default: 60) - higher values reduce impact of top ranks */
  rrfK?: number;

  /** Minimum score threshold (0-1) */
  threshold?: number;

  /** Override efSearch for vector queries */
  efSearch?: number;
}

/**
 * Result from hybrid search
 */
export interface HybridSearchResult {
  /** Document/vector ID */
  id: string;

  /** Combined score (0-1, higher is better) */
  score: number;

  /** Vector similarity score (if vector search was performed) */
  vectorScore?: number;

  /** Keyword relevance score (if keyword search was performed) */
  keywordScore?: number;

  /** Original metadata from vector store */
  metadata?: Record<string, any>;

  /** Source of the result */
  source: 'vector' | 'keyword' | 'both';
}

/**
 * Query input for hybrid search
 */
export interface HybridQuery {
  /** Text query for keyword search */
  text?: string;

  /** Vector embedding for similarity search */
  vector?: Float32Array;
}

/**
 * Configuration for BM25 scoring
 */
export interface BM25Config {
  /** Term frequency saturation parameter (default: 1.2) */
  k1: number;

  /** Document length normalization (default: 0.75) */
  b: number;
}

/**
 * Document entry in the keyword index
 */
interface DocumentEntry {
  /** Document ID */
  id: string;

  /** Term frequencies in this document */
  termFrequencies: Map<string, number>;

  /** Total number of terms in document */
  length: number;

  /** Original text (optional, for debugging) */
  originalText?: string;

  /** Associated metadata */
  metadata?: Record<string, any>;
}

/**
 * Term entry in the inverted index
 */
interface TermEntry {
  /** Document frequency - number of documents containing this term */
  documentFrequency: number;

  /** Postings list: document IDs that contain this term */
  postings: Set<string>;
}

// ============================================================================
// Stopwords
// ============================================================================

/**
 * Common English stopwords to filter out during tokenization
 */
const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
  'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
  'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
  'his', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
  'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
  'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
  'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
  'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
  'if', 'else', 'because', 'although', 'though', 'while', 'unless',
  'until', 'after', 'before', 'since', 'during', 'about', 'into',
  'through', 'between', 'under', 'over', 'above', 'below', 'up', 'down',
  'out', 'off', 'again', 'further', 'any', 'even', 'still',
]);

// ============================================================================
// KeywordIndex Class - BM25 Inverted Index
// ============================================================================

/**
 * KeywordIndex - Inverted index with BM25 scoring for text search
 *
 * Implements the BM25 ranking function for document retrieval:
 * score(D,Q) = SUM(IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)))
 *
 * Where:
 * - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
 * - f(qi,D) = term frequency of qi in document D
 * - |D| = document length
 * - avgdl = average document length
 * - k1, b = tuning parameters
 */
export class KeywordIndex {
  /** Inverted index: term -> posting list */
  private invertedIndex: Map<string, TermEntry> = new Map();

  /** Document store: id -> document entry */
  private documents: Map<string, DocumentEntry> = new Map();

  /** Total number of documents */
  private documentCount: number = 0;

  /** Sum of all document lengths (for average calculation) */
  private totalDocumentLength: number = 0;

  /** BM25 configuration */
  private config: BM25Config;

  /** Custom stopwords (merged with defaults) */
  private stopwords: Set<string>;

  constructor(config?: Partial<BM25Config>, customStopwords?: string[]) {
    this.config = {
      k1: 1.2,
      b: 0.75,
      ...config,
    };

    this.stopwords = new Set([...STOPWORDS]);
    if (customStopwords) {
      customStopwords.forEach((word) => this.stopwords.add(word.toLowerCase()));
    }
  }

  /**
   * Tokenize text into terms
   * - Lowercases text
   * - Splits on non-alphanumeric characters
   * - Removes stopwords
   * - Filters short tokens (< 2 chars)
   */
  private tokenize(text: string): string[] {
    if (!text || typeof text !== 'string') {
      return [];
    }

    return text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((token) => token.length >= 2 && !this.stopwords.has(token));
  }

  /**
   * Calculate term frequencies for a list of tokens
   */
  private calculateTermFrequencies(tokens: string[]): Map<string, number> {
    const frequencies = new Map<string, number>();

    for (const token of tokens) {
      frequencies.set(token, (frequencies.get(token) || 0) + 1);
    }

    return frequencies;
  }

  /**
   * Add a document to the index
   *
   * @param id - Unique document identifier
   * @param text - Document text content
   * @param metadata - Optional metadata to associate with the document
   */
  add(id: string, text: string, metadata?: Record<string, any>): void {
    // Remove existing document if present (update case)
    if (this.documents.has(id)) {
      this.remove(id);
    }

    const tokens = this.tokenize(text);
    const termFrequencies = this.calculateTermFrequencies(tokens);

    // Create document entry
    const docEntry: DocumentEntry = {
      id,
      termFrequencies,
      length: tokens.length,
      metadata,
    };

    this.documents.set(id, docEntry);
    this.documentCount++;
    this.totalDocumentLength += tokens.length;

    // Update inverted index
    for (const term of termFrequencies.keys()) {
      let termEntry = this.invertedIndex.get(term);

      if (!termEntry) {
        termEntry = {
          documentFrequency: 0,
          postings: new Set(),
        };
        this.invertedIndex.set(term, termEntry);
      }

      if (!termEntry.postings.has(id)) {
        termEntry.documentFrequency++;
        termEntry.postings.add(id);
      }
    }
  }

  /**
   * Remove a document from the index
   *
   * @param id - Document ID to remove
   * @returns true if document was removed, false if not found
   */
  remove(id: string): boolean {
    const docEntry = this.documents.get(id);
    if (!docEntry) {
      return false;
    }

    // Update inverted index
    for (const term of docEntry.termFrequencies.keys()) {
      const termEntry = this.invertedIndex.get(term);
      if (termEntry) {
        termEntry.postings.delete(id);
        termEntry.documentFrequency--;

        // Remove term if no documents contain it
        if (termEntry.documentFrequency === 0) {
          this.invertedIndex.delete(term);
        }
      }
    }

    // Update statistics
    this.totalDocumentLength -= docEntry.length;
    this.documentCount--;

    // Remove document
    this.documents.delete(id);

    return true;
  }

  /**
   * Calculate IDF (Inverse Document Frequency) for a term
   * Using BM25 IDF formula: log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
   */
  private calculateIDF(documentFrequency: number): number {
    const N = this.documentCount;
    const n = documentFrequency;

    // BM25 IDF with smoothing
    return Math.log((N - n + 0.5) / (n + 0.5) + 1);
  }

  /**
   * Calculate BM25 score for a document given query terms
   */
  private calculateBM25Score(
    docEntry: DocumentEntry,
    queryTerms: string[]
  ): number {
    const { k1, b } = this.config;
    const avgdl = this.totalDocumentLength / Math.max(this.documentCount, 1);
    const docLength = docEntry.length;

    let score = 0;

    for (const term of queryTerms) {
      const termEntry = this.invertedIndex.get(term);
      if (!termEntry) continue;

      const tf = docEntry.termFrequencies.get(term) || 0;
      if (tf === 0) continue;

      const idf = this.calculateIDF(termEntry.documentFrequency);

      // BM25 scoring formula
      const numerator = tf * (k1 + 1);
      const denominator = tf + k1 * (1 - b + b * (docLength / avgdl));
      const termScore = idf * (numerator / denominator);

      score += termScore;
    }

    return score;
  }

  /**
   * Search the index for documents matching the query
   *
   * @param query - Search query text
   * @param limit - Maximum number of results
   * @param filter - Optional metadata filter
   * @returns Array of search results sorted by score (descending)
   */
  search(
    query: string,
    limit: number,
    filter?: Record<string, any>
  ): { id: string; score: number; metadata?: Record<string, any> }[] {
    const queryTerms = this.tokenize(query);

    if (queryTerms.length === 0) {
      return [];
    }

    // Find candidate documents (union of postings lists)
    const candidateIds = new Set<string>();
    for (const term of queryTerms) {
      const termEntry = this.invertedIndex.get(term);
      if (termEntry) {
        for (const docId of termEntry.postings) {
          candidateIds.add(docId);
        }
      }
    }

    // Score candidates
    const results: { id: string; score: number; metadata?: Record<string, any> }[] = [];

    for (const docId of candidateIds) {
      const docEntry = this.documents.get(docId);
      if (!docEntry) continue;

      // Apply metadata filter
      if (filter && !this.matchesFilter(docEntry.metadata, filter)) {
        continue;
      }

      const score = this.calculateBM25Score(docEntry, queryTerms);

      results.push({
        id: docId,
        score,
        metadata: docEntry.metadata,
      });
    }

    // Sort by score (descending) and limit
    results.sort((a, b) => b.score - a.score);

    return results.slice(0, limit);
  }

  /**
   * Check if document metadata matches the filter criteria
   */
  private matchesFilter(
    metadata: Record<string, any> | undefined,
    filter: Record<string, any>
  ): boolean {
    if (!metadata) return false;

    for (const [key, value] of Object.entries(filter)) {
      if (metadata[key] !== value) {
        return false;
      }
    }

    return true;
  }

  /**
   * Get index statistics
   */
  getStats(): {
    documentCount: number;
    termCount: number;
    avgDocumentLength: number;
    k1: number;
    b: number;
  } {
    return {
      documentCount: this.documentCount,
      termCount: this.invertedIndex.size,
      avgDocumentLength: this.documentCount > 0
        ? this.totalDocumentLength / this.documentCount
        : 0,
      k1: this.config.k1,
      b: this.config.b,
    };
  }

  /**
   * Update BM25 parameters
   */
  setConfig(config: Partial<BM25Config>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Clear the entire index
   */
  clear(): void {
    this.invertedIndex.clear();
    this.documents.clear();
    this.documentCount = 0;
    this.totalDocumentLength = 0;
  }

  /**
   * Check if a document exists in the index
   */
  has(id: string): boolean {
    return this.documents.has(id);
  }

  /**
   * Get document count
   */
  size(): number {
    return this.documentCount;
  }

  /**
   * Add stopwords dynamically
   */
  addStopwords(words: string[]): void {
    words.forEach((word) => this.stopwords.add(word.toLowerCase()));
  }

  /**
   * Remove stopwords dynamically
   */
  removeStopwords(words: string[]): void {
    words.forEach((word) => this.stopwords.delete(word.toLowerCase()));
  }

  /**
   * Get all indexed terms (for debugging/analysis)
   */
  getTerms(): string[] {
    return Array.from(this.invertedIndex.keys());
  }

  /**
   * Get document frequency for a term
   */
  getDocumentFrequency(term: string): number {
    const termEntry = this.invertedIndex.get(term.toLowerCase());
    return termEntry?.documentFrequency || 0;
  }
}

// ============================================================================
// HybridSearch Class
// ============================================================================

/**
 * HybridSearch - Combines vector similarity and keyword search
 *
 * Supports multiple fusion strategies:
 * - RRF (Reciprocal Rank Fusion): Robust rank-based fusion
 * - Linear: Weighted score combination
 * - Max: Take maximum score from either source
 *
 * Usage:
 * ```typescript
 * const hybrid = new HybridSearch(vectorBackend, keywordIndex);
 *
 * // Combined search
 * const results = await hybrid.search(
 *   { text: "machine learning", vector: embedding },
 *   { limit: 10, vectorWeight: 0.7, keywordWeight: 0.3 }
 * );
 *
 * // Vector-only search
 * const vectorResults = await hybrid.search(
 *   { vector: embedding },
 *   { limit: 10 }
 * );
 *
 * // Keyword-only search
 * const keywordResults = await hybrid.search(
 *   { text: "neural networks" },
 *   { limit: 10 }
 * );
 * ```
 */
export class HybridSearch {
  private vectorBackend: VectorBackend;
  private keywordIndex: KeywordIndex;

  constructor(vectorBackend: VectorBackend, keywordIndex: KeywordIndex) {
    this.vectorBackend = vectorBackend;
    this.keywordIndex = keywordIndex;
  }

  /**
   * Perform hybrid search combining vector and keyword results
   *
   * @param query - Query object containing text and/or vector
   * @param options - Search options including weights and fusion method
   * @returns Combined search results
   */
  async search(
    query: HybridQuery,
    options: HybridSearchOptions
  ): Promise<HybridSearchResult[]> {
    const {
      limit,
      vectorWeight = 0.7,
      keywordWeight = 0.3,
      filter,
      fusionMethod = 'rrf',
      rrfK = 60,
      threshold,
      efSearch,
    } = options;

    // Validate weights
    if (vectorWeight < 0 || keywordWeight < 0) {
      throw new Error('Weights must be non-negative');
    }

    // Normalize weights
    const totalWeight = vectorWeight + keywordWeight;
    const normalizedVectorWeight = vectorWeight / totalWeight;
    const normalizedKeywordWeight = keywordWeight / totalWeight;

    // Collect results from both sources
    let vectorResults: SearchResult[] = [];
    let keywordResults: { id: string; score: number; metadata?: Record<string, any> }[] = [];

    // Perform vector search if vector provided
    if (query.vector) {
      const searchOptions: SearchOptions = {
        filter,
        efSearch,
      };

      // Request more results than limit for better fusion
      vectorResults = await searchVectors(
        this.vectorBackend,
        query.vector,
        Math.min(limit * 3, 1000),
        searchOptions
      );
    }

    // Perform keyword search if text provided
    if (query.text) {
      // Request more results than limit for better fusion
      keywordResults = this.keywordIndex.search(
        query.text,
        Math.min(limit * 3, 1000),
        filter
      );
    }

    // If only one type of search was performed, return those results
    if (!query.vector && query.text) {
      return this.normalizeKeywordResults(keywordResults, limit, threshold);
    }

    if (query.vector && !query.text) {
      return this.normalizeVectorResults(vectorResults, limit, threshold);
    }

    // Combine results using fusion method
    let fusedResults: HybridSearchResult[];

    switch (fusionMethod) {
      case 'rrf':
        fusedResults = this.reciprocalRankFusion(
          vectorResults,
          keywordResults,
          normalizedVectorWeight,
          normalizedKeywordWeight,
          rrfK
        );
        break;

      case 'linear':
        fusedResults = this.linearFusion(
          vectorResults,
          keywordResults,
          normalizedVectorWeight,
          normalizedKeywordWeight
        );
        break;

      case 'max':
        fusedResults = this.maxFusion(vectorResults, keywordResults);
        break;

      default:
        throw new Error(`Unknown fusion method: ${fusionMethod}`);
    }

    // Apply threshold and limit
    if (threshold !== undefined) {
      fusedResults = fusedResults.filter((r) => r.score >= threshold);
    }

    return fusedResults.slice(0, limit);
  }

  /**
   * Reciprocal Rank Fusion (RRF)
   *
   * Combines results based on their ranks rather than scores.
   * RRF(d) = SUM(1 / (k + rank(d)))
   *
   * This is robust to different score distributions between sources.
   */
  private reciprocalRankFusion(
    vectorResults: SearchResult[],
    keywordResults: { id: string; score: number; metadata?: Record<string, any> }[],
    vectorWeight: number,
    keywordWeight: number,
    k: number
  ): HybridSearchResult[] {
    const scoreMap = new Map<string, {
      rrfScore: number;
      vectorScore?: number;
      keywordScore?: number;
      vectorRank?: number;
      keywordRank?: number;
      metadata?: Record<string, any>;
    }>();

    // Process vector results
    vectorResults.forEach((result, index) => {
      const rank = index + 1;
      const rrfContribution = vectorWeight / (k + rank);

      scoreMap.set(result.id, {
        rrfScore: rrfContribution,
        vectorScore: result.similarity,
        vectorRank: rank,
        metadata: result.metadata,
      });
    });

    // Process keyword results
    keywordResults.forEach((result, index) => {
      const rank = index + 1;
      const rrfContribution = keywordWeight / (k + rank);

      const existing = scoreMap.get(result.id);
      if (existing) {
        existing.rrfScore += rrfContribution;
        existing.keywordScore = result.score;
        existing.keywordRank = rank;
        // Merge metadata if not already present
        if (!existing.metadata && result.metadata) {
          existing.metadata = result.metadata;
        }
      } else {
        scoreMap.set(result.id, {
          rrfScore: rrfContribution,
          keywordScore: result.score,
          keywordRank: rank,
          metadata: result.metadata,
        });
      }
    });

    // Convert to results array
    const results: HybridSearchResult[] = [];

    for (const [id, data] of scoreMap) {
      const hasVector = data.vectorScore !== undefined;
      const hasKeyword = data.keywordScore !== undefined;

      results.push({
        id,
        score: data.rrfScore,
        vectorScore: data.vectorScore,
        keywordScore: data.keywordScore,
        metadata: data.metadata,
        source: hasVector && hasKeyword ? 'both' : hasVector ? 'vector' : 'keyword',
      });
    }

    // Sort by RRF score (descending)
    results.sort((a, b) => b.score - a.score);

    // Normalize scores to 0-1 range
    if (results.length > 0) {
      const maxScore = results[0].score;
      for (const result of results) {
        result.score = result.score / maxScore;
      }
    }

    return results;
  }

  /**
   * Linear Fusion
   *
   * Combines normalized scores linearly:
   * score(d) = vectorWeight * vectorScore(d) + keywordWeight * keywordScore(d)
   */
  private linearFusion(
    vectorResults: SearchResult[],
    keywordResults: { id: string; score: number; metadata?: Record<string, any> }[],
    vectorWeight: number,
    keywordWeight: number
  ): HybridSearchResult[] {
    // Normalize keyword scores to 0-1
    const maxKeywordScore = keywordResults.length > 0
      ? Math.max(...keywordResults.map((r) => r.score))
      : 1;

    const normalizedKeywordScores = new Map<string, {
      score: number;
      metadata?: Record<string, any>;
    }>();

    for (const result of keywordResults) {
      normalizedKeywordScores.set(result.id, {
        score: maxKeywordScore > 0 ? result.score / maxKeywordScore : 0,
        metadata: result.metadata,
      });
    }

    // Create vector score map
    const vectorScores = new Map<string, {
      score: number;
      metadata?: Record<string, any>;
    }>();

    for (const result of vectorResults) {
      vectorScores.set(result.id, {
        score: result.similarity,
        metadata: result.metadata,
      });
    }

    // Combine all unique IDs
    const allIds = new Set([
      ...vectorScores.keys(),
      ...normalizedKeywordScores.keys(),
    ]);

    // Calculate combined scores
    const results: HybridSearchResult[] = [];

    for (const id of allIds) {
      const vectorData = vectorScores.get(id);
      const keywordData = normalizedKeywordScores.get(id);

      const vScore = vectorData?.score || 0;
      const kScore = keywordData?.score || 0;

      const combinedScore = vectorWeight * vScore + keywordWeight * kScore;

      const hasVector = vectorData !== undefined;
      const hasKeyword = keywordData !== undefined;

      results.push({
        id,
        score: combinedScore,
        vectorScore: vectorData?.score,
        keywordScore: keywordData?.score,
        metadata: vectorData?.metadata || keywordData?.metadata,
        source: hasVector && hasKeyword ? 'both' : hasVector ? 'vector' : 'keyword',
      });
    }

    // Sort by combined score (descending)
    results.sort((a, b) => b.score - a.score);

    return results;
  }

  /**
   * Max Fusion
   *
   * Takes the maximum normalized score from either source.
   * score(d) = max(vectorScore(d), keywordScore(d))
   */
  private maxFusion(
    vectorResults: SearchResult[],
    keywordResults: { id: string; score: number; metadata?: Record<string, any> }[]
  ): HybridSearchResult[] {
    // Normalize keyword scores to 0-1
    const maxKeywordScore = keywordResults.length > 0
      ? Math.max(...keywordResults.map((r) => r.score))
      : 1;

    const scoreMap = new Map<string, {
      maxScore: number;
      vectorScore?: number;
      keywordScore?: number;
      metadata?: Record<string, any>;
    }>();

    // Process vector results
    for (const result of vectorResults) {
      scoreMap.set(result.id, {
        maxScore: result.similarity,
        vectorScore: result.similarity,
        metadata: result.metadata,
      });
    }

    // Process keyword results
    for (const result of keywordResults) {
      const normalizedScore = maxKeywordScore > 0 ? result.score / maxKeywordScore : 0;
      const existing = scoreMap.get(result.id);

      if (existing) {
        existing.keywordScore = normalizedScore;
        existing.maxScore = Math.max(existing.maxScore, normalizedScore);
        if (!existing.metadata && result.metadata) {
          existing.metadata = result.metadata;
        }
      } else {
        scoreMap.set(result.id, {
          maxScore: normalizedScore,
          keywordScore: normalizedScore,
          metadata: result.metadata,
        });
      }
    }

    // Convert to results array
    const results: HybridSearchResult[] = [];

    for (const [id, data] of scoreMap) {
      const hasVector = data.vectorScore !== undefined;
      const hasKeyword = data.keywordScore !== undefined;

      results.push({
        id,
        score: data.maxScore,
        vectorScore: data.vectorScore,
        keywordScore: data.keywordScore,
        metadata: data.metadata,
        source: hasVector && hasKeyword ? 'both' : hasVector ? 'vector' : 'keyword',
      });
    }

    // Sort by max score (descending)
    results.sort((a, b) => b.score - a.score);

    return results;
  }

  /**
   * Normalize vector-only results to HybridSearchResult format
   */
  private normalizeVectorResults(
    results: SearchResult[],
    limit: number,
    threshold?: number
  ): HybridSearchResult[] {
    let normalized = results.map((r) => ({
      id: r.id,
      score: r.similarity,
      vectorScore: r.similarity,
      metadata: r.metadata,
      source: 'vector' as const,
    }));

    if (threshold !== undefined) {
      normalized = normalized.filter((r) => r.score >= threshold);
    }

    return normalized.slice(0, limit);
  }

  /**
   * Normalize keyword-only results to HybridSearchResult format
   */
  private normalizeKeywordResults(
    results: { id: string; score: number; metadata?: Record<string, any> }[],
    limit: number,
    threshold?: number
  ): HybridSearchResult[] {
    // Normalize scores to 0-1
    const maxScore = results.length > 0
      ? Math.max(...results.map((r) => r.score))
      : 1;

    let normalized = results.map((r) => ({
      id: r.id,
      score: maxScore > 0 ? r.score / maxScore : 0,
      keywordScore: maxScore > 0 ? r.score / maxScore : 0,
      metadata: r.metadata,
      source: 'keyword' as const,
    }));

    if (threshold !== undefined) {
      normalized = normalized.filter((r) => r.score >= threshold);
    }

    return normalized.slice(0, limit);
  }

  /**
   * Add a document to the keyword index
   * (Convenience method - delegates to KeywordIndex)
   */
  addDocument(id: string, text: string, metadata?: Record<string, any>): void {
    this.keywordIndex.add(id, text, metadata);
  }

  /**
   * Remove a document from the keyword index
   * (Convenience method - delegates to KeywordIndex)
   */
  removeDocument(id: string): boolean {
    return this.keywordIndex.remove(id);
  }

  /**
   * Get statistics about the hybrid search system
   */
  getStats(): {
    vector: {
      count: number;
      dimension: number;
      backend: string;
    };
    keyword: {
      documentCount: number;
      termCount: number;
      avgDocumentLength: number;
    };
  } {
    const vectorStats = this.vectorBackend.getStats();
    const keywordStats = this.keywordIndex.getStats();

    return {
      vector: {
        count: vectorStats.count,
        dimension: vectorStats.dimension,
        backend: vectorStats.backend,
      },
      keyword: {
        documentCount: keywordStats.documentCount,
        termCount: keywordStats.termCount,
        avgDocumentLength: keywordStats.avgDocumentLength,
      },
    };
  }

  /**
   * Get the underlying vector backend
   */
  getVectorBackend(): VectorBackend {
    return this.vectorBackend;
  }

  /**
   * Get the underlying keyword index
   */
  getKeywordIndex(): KeywordIndex {
    return this.keywordIndex;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new KeywordIndex with default configuration
 */
export function createKeywordIndex(
  config?: Partial<BM25Config>,
  customStopwords?: string[]
): KeywordIndex {
  return new KeywordIndex(config, customStopwords);
}

/**
 * Create a new HybridSearch instance
 */
export function createHybridSearch(
  vectorBackend: VectorBackend,
  keywordIndex?: KeywordIndex
): HybridSearch {
  const kwIndex = keywordIndex || new KeywordIndex();
  return new HybridSearch(vectorBackend, kwIndex);
}

// ============================================================================
// Re-export types for convenience
// ============================================================================

export type {
  VectorBackend,
  SearchResult,
  SearchOptions,
} from '../backends/VectorBackend.js';
