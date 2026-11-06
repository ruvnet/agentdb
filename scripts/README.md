# AgentDB Scripts Directory

This directory contains build, validation, and deployment scripts for AgentDB v2.

## Build Scripts

### Browser Builds

#### `build-browser.js`
**Purpose:** Original browser bundle builder
**Usage:** `node scripts/build-browser.js`
**Output:** `dist/agentdb.browser.js`
**Features:**
- Basic SQL.js WASM integration
- Single-file browser bundle
- CDN-ready output

#### `build-browser-v2.js`
**Purpose:** Enhanced v2 browser bundle with advanced features
**Usage:** `node scripts/build-browser-v2.js`
**Output:** `dist/agentdb.browser.v2.js`
**Features:**
- Multi-backend support (SQL.js/IndexedDB auto-detection)
- GNN optimization integration
- IndexedDB persistence
- Cross-tab synchronization
- 100% v1 API backward compatibility
- Enhanced error handling

**Recommended:** Use this for new projects requiring browser support.

#### `build-browser-advanced.cjs`
**Purpose:** Advanced browser features build
**Usage:** `node scripts/build-browser-advanced.cjs`
**Features:**
- Advanced WASM optimization
- Progressive loading
- Service worker integration
- Memory management enhancements

### Dependencies

#### `postinstall.cjs`
**Purpose:** Post-installation setup and verification
**Auto-runs:** After `npm install`
**Tasks:**
- Validates environment
- Checks native dependencies (better-sqlite3)
- Sets up development environment
- Verifies WASM files

## Validation Scripts

### `comprehensive-review.ts`
**Purpose:** Complete v2 feature validation and performance testing
**Usage:** `tsx scripts/comprehensive-review.ts`
**Tests:**
- @ruvector/core integration
- @ruvector/gnn integration
- ReasoningBank functionality
- All v2 controllers (HNSW, QUIC, etc.)
- Backend performance comparison
- Memory usage analysis
- Optimization opportunities

**Output:** Comprehensive test report with metrics

### `validate-security-fixes.ts`
**Purpose:** Security validation and audit
**Usage:** `tsx scripts/validate-security-fixes.ts`
**Checks:**
- SQL injection prevention
- Input sanitization
- Path traversal protection
- Dependency vulnerabilities
- Code signing verification

### `verify-bundle.js`
**Purpose:** Bundle integrity verification
**Usage:** `node scripts/verify-bundle.js`
**Validates:**
- Bundle size limits
- Export completeness
- API surface consistency
- WASM file integrity

### `verify-core-tools-6-10.sh`
**Purpose:** Core tools validation (tools 6-10)
**Usage:** `bash scripts/verify-core-tools-6-10.sh`
**Tests:**
- Tool 6: Batch insert operations
- Tool 7: Hybrid search
- Tool 8: QUIC synchronization
- Tool 9: Learning plugins
- Tool 10: Performance benchmarks

## Release Scripts

### `npm-release.sh`
**Purpose:** Automated NPM release workflow
**Usage:** `bash scripts/npm-release.sh [version]`
**Process:**
1. Version bump (semver)
2. Changelog generation
3. Build verification
4. Test suite execution
5. Bundle validation
6. NPM publish
7. Git tag creation

**Requirements:**
- NPM authentication
- Git repository
- Clean working tree

### `pre-release-validation.sh`
**Purpose:** Pre-release quality gate
**Usage:** `bash scripts/pre-release-validation.sh`
**Validates:**
- All tests passing
- No TypeScript errors
- Bundle integrity
- Documentation accuracy
- Security audit clean
- Performance benchmarks met

## Testing Scripts

### `docker-test.sh`
**Purpose:** Docker environment testing
**Usage:** `bash scripts/docker-test.sh`
**Tests:**
- Installation in clean environment
- Runtime dependencies
- Cross-platform compatibility
- Network isolation scenarios

### `docker-validation.sh`
**Purpose:** Comprehensive Docker validation suite
**Usage:** `bash scripts/docker-validation.sh`
**Includes:**
- Multi-stage build verification
- Container security scan
- Resource usage monitoring
- Integration test suite

## AgentDB Version

**Current Version:** 1.6.1
**Target:** v2.0.0 with full backward compatibility

All scripts are designed to work with:
- **Node.js:** >=20.9.0
- **TypeScript:** ^5.7.2
- **Better-sqlite3:** ^11.8.1 (optional)
- **@ruvector/core:** ^0.1.15
- **@ruvector/gnn:** ^0.1.15

## Development Workflow

### 1. Local Development
```bash
npm run build          # Full build pipeline
npm run dev           # Development mode with tsx
npm test              # Run test suite
```

### 2. Browser Testing
```bash
npm run build:browser  # Build browser bundle
npm run test:browser   # Test browser bundle
npm run verify:bundle  # Verify bundle integrity
```

### 3. Pre-Release Validation
```bash
bash scripts/pre-release-validation.sh
tsx scripts/comprehensive-review.ts
tsx scripts/validate-security-fixes.ts
```

### 4. Release
```bash
bash scripts/npm-release.sh patch  # Patch release
bash scripts/npm-release.sh minor  # Minor release
bash scripts/npm-release.sh major  # Major release
```

## Script Dependencies

### Required Global Tools
- `node` (>=20.9.0)
- `npm` (>=9.0.0)
- `bash` (>=4.0)
- `tsx` (for TypeScript scripts)
- `docker` (for container tests)

### Package Scripts Integration

Scripts integrate with `package.json` scripts:

```json
{
  "scripts": {
    "build": "npm run build:ts && npm run copy:schemas && npm run build:browser",
    "build:browser": "node scripts/build-browser.js",
    "postinstall": "node scripts/postinstall.cjs || true",
    "verify:bundle": "node scripts/verify-bundle.js",
    "docker:test": "bash scripts/docker-test.sh"
  }
}
```

## Troubleshooting

### Build Failures

**Problem:** Browser bundle build fails
**Solution:**
```bash
# Clear dist and rebuild
rm -rf dist
npm run build
```

**Problem:** WASM files not loading
**Solution:**
```bash
# Re-download dependencies
rm -rf node_modules
npm install
```

### Validation Failures

**Problem:** Security validation fails
**Solution:**
```bash
npm audit fix
tsx scripts/validate-security-fixes.ts
```

**Problem:** Bundle size exceeded
**Solution:**
```bash
# Check bundle analysis
npm run verify:bundle
# Consider code splitting or lazy loading
```

### Release Issues

**Problem:** NPM publish fails
**Solution:**
```bash
# Verify authentication
npm whoami
npm login

# Check version
npm version patch --no-git-tag-version
```

## Best Practices

1. **Always run validation before releases:**
   ```bash
   bash scripts/pre-release-validation.sh
   ```

2. **Test browser builds locally:**
   ```bash
   npm run test:browser
   ```

3. **Keep dependencies updated:**
   ```bash
   npm outdated
   npm update
   ```

4. **Run security audits regularly:**
   ```bash
   tsx scripts/validate-security-fixes.ts
   ```

5. **Verify Docker compatibility:**
   ```bash
   bash scripts/docker-test.sh
   ```

## Contributing

When adding new scripts:

1. Add executable permissions: `chmod +x script-name.sh`
2. Include shebang line: `#!/usr/bin/env node` or `#!/usr/bin/env bash`
3. Add comprehensive error handling
4. Document in this README
5. Add to package.json scripts if appropriate
6. Include validation tests

## Support

For issues with scripts:
- Check logs in `logs/` directory
- Review error messages carefully
- Verify Node.js/npm versions
- Check GitHub Issues: https://github.com/ruvnet/agentic-flow/issues

---

**Last Updated:** 2025-11-29
**AgentDB Version:** 1.6.1 → 2.0.0
