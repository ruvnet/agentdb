# Getting Started with AgentDB Attention Mechanisms

Welcome to AgentDB's attention mechanisms! This tutorial will guide you through setting up and using hyperbolic memory, Flash consolidation, Graph-RoPE recall, and MoE routing.

## What You'll Learn

- Installing and initializing AgentDB
- Enabling attention mechanisms
- Storing and retrieving memories
- Understanding when to use each mechanism

## Prerequisites

- Node.js 20.9.0+ or modern browser
- Basic understanding of embeddings/vectors
- TypeScript recommended but not required

## Installation

```bash
npm install @agentic/agentdb better-sqlite3
```

For TypeScript:
```bash
npm install -D @types/better-sqlite3
```

## Basic Setup

### Step 1: Initialize Database

```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

// Create an in-memory database (or use a file path)
const db = new Database(':memory:');

// Initialize with all features enabled
const attention = new AttentionService(db, {
  enableHyperbolic: true,
  enableFlash: true,
  enableGraphRoPE: true,
  enableMoE: true,
  vectorDimension: 1536  // Match your embedding model
});

console.log('AgentDB initialized successfully!');
```

### Step 2: Create Your First Embeddings

For this tutorial, we'll use mock embeddings. In production, use a real embedding model like OpenAI's text-embedding-3-small.

```typescript
// Helper to create mock embeddings (replace with real embeddings)
function createEmbedding(text: string): Float32Array {
  const dimension = 1536;
  const vector = new Float32Array(dimension);

  // Simple hash-based mock (use real embeddings in production!)
  for (let i = 0; i < dimension; i++) {
    vector[i] = Math.sin(text.charCodeAt(i % text.length) * i);
  }

  // Normalize
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  for (let i = 0; i < dimension; i++) {
    vector[i] /= magnitude;
  }

  return vector;
}

// Example usage
const embedding = createEmbedding("Hello, world!");
console.log(`Created embedding with dimension ${embedding.length}`);
```

### Step 3: Store Your First Memory

```typescript
// Store a simple memory
const text = "Paris is the capital of France";
const embedding = createEmbedding(text);

const memoryId = await attention.hyperbolic.storeWithHierarchy(
  embedding,
  {
    text,
    type: 'fact',
    category: 'geography'
  },
  0  // Root level (depth 0)
);

console.log(`Stored memory with ID: ${memoryId}`);
```

### Step 4: Search Memories

```typescript
// Search for similar memories
const query = createEmbedding("What is the capital of France?");

const results = await attention.hyperbolic.hierarchicalSearch(
  query,
  5  // Top 5 results
);

results.forEach((result, index) => {
  console.log(`${index + 1}. ${result.metadata.text}`);
  console.log(`   Score: ${result.hyperbolicScore.toFixed(4)}`);
  console.log(`   Depth: ${result.depth}`);
});
```

Expected output:
```
1. Paris is the capital of France
   Score: 0.9876
   Depth: 0
```

## Understanding Each Mechanism

### Hyperbolic Memory: Hierarchical Organization

**When to use:** Organizing knowledge with parent-child relationships

```typescript
// Store hierarchical knowledge
const continentId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("Europe"),
  { name: "Europe", type: "continent" },
  0  // Root level
);

const countryId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("France"),
  { name: "France", type: "country", parent: continentId },
  1  // Child level
);

const cityId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("Paris"),
  { name: "Paris", type: "city", parent: countryId },
  2  // Grandchild level
);

// Search respects hierarchy
const results = await attention.hyperbolic.hierarchicalSearch(
  createEmbedding("European cities"),
  10,
  2  // Search up to depth 2
);

console.log('Hierarchical search results:');
results.forEach(result => {
  console.log(`${'  '.repeat(result.depth)}${result.metadata.name} (depth ${result.depth})`);
});
```

Output:
```
Hierarchical search results:
Europe (depth 0)
  France (depth 1)
    Paris (depth 2)
```

### Flash Consolidation: Efficient Memory Compression

**When to use:** Consolidating large memory sets, reducing storage

```typescript
// Store many related memories
const memories = [
  "The Eiffel Tower is in Paris",
  "Paris has the Louvre Museum",
  "Notre-Dame Cathedral is in Paris",
  "Paris is known for its cuisine",
  "The Seine river flows through Paris"
];

const vectors = memories.map(text => createEmbedding(text));

// Consolidate into efficient representation
const consolidated = await attention.flash.consolidateMemories(
  vectors,
  128  // Window size
);

console.log(`Consolidated ${consolidated.sourceCount} memories`);
console.log(`Compression ratio: ${consolidated.compressionRatio.toFixed(2)}x`);

// Query consolidated memories
const query = createEmbedding("Tell me about Paris landmarks");
const results = await attention.flash.queryConsolidated(query, 3);

results.forEach(result => {
  console.log(`Score: ${result.flashScore.toFixed(4)}`);
  console.log(`Window: tokens ${result.windowInfo.start}-${result.windowInfo.end}`);
});
```

### Graph-RoPE: Contextual Relationships

**When to use:** Finding related memories through connections

```typescript
// Build a knowledge graph
const memories = [
  { text: "Paris is in France", id: 1 },
  { text: "France is in Europe", id: 2 },
  { text: "London is in England", id: 3 },
  { text: "England is in Europe", id: 4 }
];

const graphData = memories.map(mem => ({
  id: mem.id,
  vector: createEmbedding(mem.text),
  metadata: { text: mem.text }
}));

await attention.graphRoPE.buildMemoryGraph(graphData);

// Add explicit relationships
await attention.graphRoPE.addEdge(1, 2, 0.9);  // Paris -> France
await attention.graphRoPE.addEdge(2, 4, 0.8);  // France -> Europe
await attention.graphRoPE.addEdge(3, 4, 0.9);  // London -> England

// Search with graph awareness
const results = await attention.graphRoPE.graphAwareSearch(
  createEmbedding("European cities"),
  5,
  2  // Explore 2 hops
);

results.forEach(result => {
  console.log(`${result.metadata.text}`);
  console.log(`  Path length: ${result.pathLength} hops`);
  console.log(`  Connected to: ${result.connectedIds.length} memories`);
});
```

### MoE Routing: Specialized Retrieval

**When to use:** Different types of queries need different experts

```typescript
// Create specialized experts
const technicalVectors = [
  createEmbedding("JavaScript async/await syntax"),
  createEmbedding("React hooks usage"),
  createEmbedding("TypeScript generics")
].map(vec => vec);

const businessVectors = [
  createEmbedding("Quarterly revenue analysis"),
  createEmbedding("Market segmentation strategy"),
  createEmbedding("Customer acquisition cost")
].map(vec => vec);

// Add experts
await attention.moe.addExpert(
  'Technical Expert',
  'technical_documentation',
  technicalVectors
);

await attention.moe.addExpert(
  'Business Expert',
  'business_analysis',
  businessVectors
);

// Query automatically routes to best expert
const technicalQuery = createEmbedding("How to use React hooks?");
const technicalResults = await attention.moe.routeQuery(technicalQuery, 3);

console.log('Technical query routed to:');
technicalResults.forEach(result => {
  console.log(`  Expert: ${result.expertName}`);
  console.log(`  Confidence: ${result.routingScore.toFixed(4)}`);
});

const businessQuery = createEmbedding("What is our CAC?");
const businessResults = await attention.moe.routeQuery(businessQuery, 3);

console.log('Business query routed to:');
businessResults.forEach(result => {
  console.log(`  Expert: ${result.expertName}`);
  console.log(`  Confidence: ${result.routingScore.toFixed(4)}`);
});
```

## Combining Mechanisms

The real power comes from combining mechanisms:

```typescript
// 1. Store with hierarchy (Hyperbolic)
const parentId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("Programming languages"),
  { category: "Programming" },
  0
);

const childId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("JavaScript features"),
  { category: "JavaScript", parent: parentId },
  1
);

// 2. Consolidate related memories (Flash)
const jsMemories = [
  createEmbedding("JavaScript closures"),
  createEmbedding("JavaScript promises"),
  createEmbedding("JavaScript async/await")
];

await attention.flash.consolidateMemories(jsMemories);

// 3. Build connections (Graph-RoPE)
await attention.graphRoPE.buildMemoryGraph([
  { id: childId, vector: jsMemories[0], metadata: { text: "closures" } },
  { id: childId + 1, vector: jsMemories[1], metadata: { text: "promises" } },
  { id: childId + 2, vector: jsMemories[2], metadata: { text: "async/await" } }
]);

// 4. Create specialized expert (MoE)
await attention.moe.addExpert(
  'JavaScript Expert',
  'javascript_programming',
  jsMemories
);

// Now query with all mechanisms working together!
const query = createEmbedding("How do JavaScript async functions work?");

// Each mechanism contributes its strengths
const hyperbolicResults = await attention.hyperbolic.hierarchicalSearch(query, 5);
const flashResults = await attention.flash.queryConsolidated(query, 5);
const graphResults = await attention.graphRoPE.graphAwareSearch(query, 5);
const moeResults = await attention.moe.routeQuery(query, 5);

console.log('Results from all mechanisms:');
console.log(`Hyperbolic (hierarchy): ${hyperbolicResults.length} results`);
console.log(`Flash (consolidated): ${flashResults.length} results`);
console.log(`Graph-RoPE (connected): ${graphResults.length} results`);
console.log(`MoE (specialized): ${moeResults.length} results`);
```

## Performance Monitoring

Check system status and statistics:

```typescript
// Overall status
const status = attention.getStatus();
console.log('System status:', status);

// Hyperbolic stats
const hyperbolicStats = attention.hyperbolic.getHierarchyStats();
console.log('Hierarchy:', hyperbolicStats);

// Flash stats
const flashStats = attention.flash.getConsolidationStats();
console.log('Consolidation:', flashStats);

// Graph stats
const graphStats = attention.graphRoPE.getGraphStats();
console.log('Graph:', graphStats);

// MoE stats
const expertStats = attention.moe.getExpertStats();
console.log('Experts:', expertStats);
```

## Cleanup

Always clean up when done:

```typescript
// Graceful shutdown
attention.shutdown();
db.close();
console.log('Cleanup complete');
```

## Next Steps

Now that you understand the basics, explore:

- [Hyperbolic Memory Tutorial](02-hyperbolic-memory.md) - Deep dive into hierarchies
- [Flash Consolidation Tutorial](03-flash-consolidation.md) - Optimize memory usage
- [Graph-RoPE Tutorial](04-graph-rope-recall.md) - Build knowledge graphs
- [MoE Routing Tutorial](05-moe-routing.md) - Create expert systems

## Common Patterns

### Pattern 1: Document Storage and Retrieval

```typescript
// Store document chunks with hierarchy
const docId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("Introduction to Machine Learning"),
  { type: 'document', title: 'ML Guide' },
  0
);

const chunkId = await attention.hyperbolic.storeWithHierarchy(
  createEmbedding("Supervised learning uses labeled data"),
  { type: 'chunk', parent: docId, section: 'Chapter 1' },
  1
);

// Search with context
const results = await attention.hyperbolic.hierarchicalSearch(
  createEmbedding("What is supervised learning?"),
  5
);
```

### Pattern 2: Conversational Memory

```typescript
// Consolidate conversation history
const conversation = [
  "User: What's the weather?",
  "Assistant: It's sunny today.",
  "User: Should I bring an umbrella?",
  "Assistant: No need, no rain expected."
];

const vectors = conversation.map(msg => createEmbedding(msg));
await attention.flash.consolidateMemories(vectors);

// Retrieve relevant context
const query = createEmbedding("User: What did we discuss about weather?");
const context = await attention.flash.queryConsolidated(query, 3);
```

### Pattern 3: Multi-Domain Knowledge Base

```typescript
// Create domain experts
const domains = [
  { name: 'Medical', docs: medicalDocuments },
  { name: 'Legal', docs: legalDocuments },
  { name: 'Technical', docs: technicalDocuments }
];

for (const domain of domains) {
  const vectors = domain.docs.map(doc => createEmbedding(doc));
  await attention.moe.addExpert(
    `${domain.name} Expert`,
    domain.name.toLowerCase(),
    vectors
  );
}

// Route queries to appropriate domain
const query = createEmbedding("What are HIPAA requirements?");
const results = await attention.moe.routeQuery(query, 5, 1);
// Automatically routes to Medical Expert
```

## Troubleshooting

### Issue: "Vector dimension mismatch"

**Solution:** Ensure all vectors have the same dimension specified in config:

```typescript
const attention = new AttentionService(db, {
  vectorDimension: 1536  // Must match your embedding model
});
```

### Issue: "Feature not enabled"

**Solution:** Enable the feature in configuration:

```typescript
attention.enableFeatures({
  enableHyperbolic: true,
  enableFlash: true,
  enableGraphRoPE: true,
  enableMoE: true
});
```

### Issue: "Out of memory"

**Solution:** Use smaller window sizes or consolidate less frequently:

```typescript
attention.enableFeatures({
  flashWindowSize: 128  // Reduce from default 256
});
```

## Summary

You've learned:
- ✅ How to initialize AgentDB with attention mechanisms
- ✅ Basic operations for each mechanism
- ✅ When to use each mechanism
- ✅ How to combine mechanisms for powerful retrieval
- ✅ Common patterns and troubleshooting

Happy building with AgentDB! 🚀
