# ─────────────────────────────────────────────────────
# AgentDB Clean-Install Verification
#
# Simulates: npm install agentdb@alpha
# - Uses ONLY hard dependencies (sql.js, ajv, etc.)
# - NO optional native backends (@ruvector/*, hnswlib-node, better-sqlite3)
# - Verifies SqlJsRvfBackend kicks in as default
# - Tests all 10 capability tiers
#
# Build:  docker build -f tests/docker/Dockerfile -t agentdb-test .
# Run:    docker run --rm agentdb-test
# ─────────────────────────────────────────────────────

# ── Stage 1: Node 20.9.0 (minimum API compatibility) ────────────
FROM node:20.9.0-slim AS test-node-minimum
WORKDIR /app
COPY agentdb-3.0.0-alpha.0.tgz ./
RUN npm init -y && npm install --no-optional ./agentdb-3.0.0-alpha.0.tgz 2>&1
COPY tests/docker/test-clean-install.mjs ./test-clean-install.mjs
RUN echo "═══ Testing on Node $(node -v) ($(uname -m)) ═══" && \
    node test-clean-install.mjs && \
    echo "node-minimum:PASS" > /app/result.txt

# ── Stage 2: Node 22 (current LTS) ──────────────────
FROM node:22-slim AS test-node22
WORKDIR /app
COPY agentdb-3.0.0-alpha.0.tgz ./
RUN npm init -y && npm install --no-optional ./agentdb-3.0.0-alpha.0.tgz 2>&1
COPY tests/docker/test-clean-install.mjs ./test-clean-install.mjs
RUN echo "═══ Testing on Node $(node -v) ($(uname -m)) ═══" && \
    node test-clean-install.mjs && \
    echo "node22:PASS" > /app/result.txt

# ── Stage 3: Alpine / musl libc ─────────────────────
FROM node:20-alpine AS test-alpine
WORKDIR /app
COPY agentdb-3.0.0-alpha.0.tgz ./
RUN npm init -y && npm install --no-optional ./agentdb-3.0.0-alpha.0.tgz 2>&1
COPY tests/docker/test-clean-install.mjs ./test-clean-install.mjs
RUN echo "═══ Testing on Node $(node -v) Alpine/musl ($(uname -m)) ═══" && \
    node test-clean-install.mjs && \
    echo "alpine:PASS" > /app/result.txt

# ── Final: aggregate results ─────────────────────────
FROM node:22-slim AS final
WORKDIR /results
COPY --from=test-node-minimum  /app/result.txt ./node-minimum.txt
COPY --from=test-node22  /app/result.txt ./node22.txt
COPY --from=test-alpine  /app/result.txt ./alpine.txt
RUN cat *.txt && echo "" && \
    echo "══════════════════════════════════════════════════" && \
    echo "  AgentDB Clean-Install Tests: ALL STAGES PASSED" && \
    echo "  ✓ Node 20.9.0 (minimum API compatibility)" && \
    echo "  ✓ Node 22 (current LTS)" && \
    echo "  ✓ Node 20 Alpine/musl (alternative libc)" && \
    echo "══════════════════════════════════════════════════" && \
    echo "" && \
    echo "Cross-platform notes:" && \
    echo "  • Linux glibc: Verified (Node 20.9.0 + 22 slim)" && \
    echo "  • Linux musl:  Verified (Alpine)" && \
    echo "  • macOS:  sql.js WASM + pure JS SIMD = no native deps" && \
    echo "  • Windows: sql.js WASM + pure JS SIMD = no native deps" && \
    echo ""
CMD ["cat", "node-minimum.txt", "node22.txt", "alpine.txt"]
