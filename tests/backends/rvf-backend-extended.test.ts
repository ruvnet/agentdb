/**
 * ADR-007 Phase 1: RvfBackend Extended API Test Suite
 *
 * Tests compression profiles, filter expressions, read-only access,
 * kernel/eBPF embed/extract, and lineage introspection.
 *
 * Since @ruvector/rvf is a native module that may not be installed in CI,
 * we mock the RvfDatabase class to validate the wiring logic.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { FilterBuilder, type RvfFilterExpr, type FilterPredicate } from '../../src/backends/rvf/FilterBuilder.js';

// ─── Mock @ruvector/rvf ───

const mockDb = {
  status: vi.fn().mockResolvedValue({ totalVectors: 42, totalSegments: 3 }),
  ingestBatch: vi.fn().mockResolvedValue({ accepted: 1, rejected: 0, epoch: 1 }),
  query: vi.fn().mockResolvedValue([]),
  delete: vi.fn().mockResolvedValue({ deleted: 1, epoch: 2 }),
  deleteByFilter: vi.fn().mockResolvedValue({ deleted: 5, epoch: 3 }),
  compact: vi.fn().mockResolvedValue({ segmentsCompacted: 1, bytesReclaimed: 1024, epoch: 4 }),
  close: vi.fn().mockResolvedValue(undefined),
  fileId: vi.fn().mockResolvedValue('abc123def456'),
  parentId: vi.fn().mockResolvedValue('0000000000000000'),
  lineageDepth: vi.fn().mockResolvedValue(0),
  derive: vi.fn().mockResolvedValue(null),
  segments: vi.fn().mockResolvedValue([]),
  dimension: vi.fn().mockResolvedValue(128),
  metric: vi.fn().mockReturnValue('cosine'),
  indexStats: vi.fn().mockReturnValue({ indexedVectors: 42, layers: 3, m: 16, efConstruction: 200, needsRebuild: false }),
  verifyWitness: vi.fn().mockReturnValue({ valid: true, entries: 10 }),
  freeze: vi.fn().mockReturnValue(1),
  embedKernel: vi.fn().mockResolvedValue(7),
  extractKernel: vi.fn().mockResolvedValue({ header: new Uint8Array([1, 2, 3]), image: new Uint8Array([4, 5, 6]) }),
  embedEbpf: vi.fn().mockResolvedValue(8),
  extractEbpf: vi.fn().mockResolvedValue({ header: new Uint8Array([10, 20]), payload: new Uint8Array([30, 40]) }),
};

const mockReadonlyDb = {
  ...mockDb,
  status: vi.fn().mockResolvedValue({ totalVectors: 100, totalSegments: 5, readOnly: true }),
  dimension: vi.fn().mockResolvedValue(64),
};

const mockCreateFn = vi.fn().mockResolvedValue(mockDb);
const mockOpenFn = vi.fn().mockResolvedValue(mockDb);
const mockOpenReadonlyFn = vi.fn().mockResolvedValue(mockReadonlyDb);

vi.mock('@ruvector/rvf', () => ({
  RvfDatabase: {
    create: (...args: unknown[]) => mockCreateFn(...args),
    open: (...args: unknown[]) => mockOpenFn(...args),
    openReadonly: (...args: unknown[]) => mockOpenReadonlyFn(...args),
  },
}));

// Must import after mock setup
import { RvfBackend } from '../../src/backends/rvf/RvfBackend.js';

describe('ADR-007 Phase 1: RvfBackend Extended APIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Re-establish all mock return values after clearAllMocks
    mockCreateFn.mockResolvedValue(mockDb);
    mockOpenFn.mockResolvedValue(mockDb);
    mockOpenReadonlyFn.mockResolvedValue(mockReadonlyDb);
    // Restore mockDb method implementations
    mockDb.status.mockResolvedValue({ totalVectors: 42, totalSegments: 3 });
    mockDb.ingestBatch.mockResolvedValue({ accepted: 1, rejected: 0, epoch: 1 });
    mockDb.query.mockResolvedValue([]);
    mockDb.delete.mockResolvedValue({ deleted: 1, epoch: 2 });
    mockDb.deleteByFilter.mockResolvedValue({ deleted: 5, epoch: 3 });
    mockDb.compact.mockResolvedValue({ segmentsCompacted: 1, bytesReclaimed: 1024, epoch: 4 });
    mockDb.close.mockResolvedValue(undefined);
    mockDb.fileId.mockResolvedValue('abc123def456');
    mockDb.parentId.mockResolvedValue('0000000000000000');
    mockDb.lineageDepth.mockResolvedValue(0);
    mockDb.derive.mockResolvedValue(null);
    mockDb.segments.mockResolvedValue([]);
    mockDb.dimension.mockResolvedValue(128);
    mockDb.metric.mockReturnValue('cosine');
    mockDb.indexStats.mockReturnValue({ indexedVectors: 42, layers: 3, m: 16, efConstruction: 200, needsRebuild: false });
    mockDb.verifyWitness.mockReturnValue({ valid: true, entries: 10 });
    mockDb.freeze.mockReturnValue(1);
    mockDb.embedKernel.mockResolvedValue(7);
    mockDb.extractKernel.mockResolvedValue({ header: new Uint8Array([1, 2, 3]), image: new Uint8Array([4, 5, 6]) });
    mockDb.embedEbpf.mockResolvedValue(8);
    mockDb.extractEbpf.mockResolvedValue({ header: new Uint8Array([10, 20]), payload: new Uint8Array([30, 40]) });
    // Restore mockReadonlyDb method implementations
    mockReadonlyDb.status.mockResolvedValue({ totalVectors: 100, totalSegments: 5, readOnly: true });
    mockReadonlyDb.dimension.mockResolvedValue(64);
  });

  // ─── Compression Profiles ───

  describe('Compression profiles', () => {
    it('should pass compression=scalar to RvfDatabase.create()', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
        compression: 'scalar',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      expect(mockCreateFn).toHaveBeenCalledTimes(1);
      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.compression).toBe('scalar');
    });

    it('should pass compression=product to RvfDatabase.create()', async () => {
      const backend = new RvfBackend({
        dimension: 256,
        metric: 'l2',
        storagePath: ':memory:',
        compression: 'product',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      expect(mockCreateFn).toHaveBeenCalledTimes(1);
      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.compression).toBe('product');
    });

    it('should not pass compression when set to none', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
        compression: 'none',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.compression).toBeUndefined();
    });

    it('should not pass compression when unset (default)', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.compression).toBeUndefined();
    });
  });

  // ─── Hardware Profiles ───

  describe('Hardware profiles', () => {
    it('should pass hardwareProfile to RvfDatabase.create() as profile', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
        hardwareProfile: 2,
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.profile).toBe(2);
    });

    it('should not pass profile when hardwareProfile is unset', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.profile).toBeUndefined();
    });

    it('should accept profile=0 (Generic)', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
        hardwareProfile: 0,
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.profile).toBe(0);
    });
  });

  // ─── FilterBuilder ───

  describe('FilterBuilder', () => {
    it('should build eq filter from plain value', () => {
      const filter = FilterBuilder.buildFilter({ status: 'active' });
      expect(filter).toEqual({ op: 'eq', field: 'status', value: 'active' });
    });

    it('should build AND of multiple predicates', () => {
      const filter = FilterBuilder.buildFilter({ status: 'active', score: 42 });
      expect(filter).not.toBeNull();
      expect((filter as { op: string }).op).toBe('and');
      const andExpr = filter as { op: 'and'; exprs: RvfFilterExpr[] };
      expect(andExpr.exprs).toHaveLength(2);
      expect(andExpr.exprs[0]).toEqual({ op: 'eq', field: 'status', value: 'active' });
      expect(andExpr.exprs[1]).toEqual({ op: 'eq', field: 'score', value: 42 });
    });

    it('should build $gt predicate', () => {
      const filter = FilterBuilder.buildFilter({ score: { $gt: 0.5 } });
      expect(filter).toEqual({ op: 'gt', field: 'score', value: 0.5 });
    });

    it('should build $lt predicate', () => {
      const filter = FilterBuilder.buildFilter({ age: { $lt: 30 } });
      expect(filter).toEqual({ op: 'lt', field: 'age', value: 30 });
    });

    it('should build $ne predicate', () => {
      const filter = FilterBuilder.buildFilter({ deleted: { $ne: true } });
      expect(filter).toEqual({ op: 'ne', field: 'deleted', value: true });
    });

    it('should build $in predicate', () => {
      const filter = FilterBuilder.buildFilter({ type: { $in: ['a', 'b', 'c'] } });
      expect(filter).toEqual({ op: 'in', field: 'type', values: ['a', 'b', 'c'] });
    });

    it('should build $range predicate', () => {
      const filter = FilterBuilder.buildFilter({ score: { $range: [0, 100] } });
      // Inclusive bounds: rvf's native range excludes high, so ge+le is emitted.
      expect(filter).toEqual({
        op: 'and',
        exprs: [
          { op: 'ge', field: 'score', value: 0 },
          { op: 'le', field: 'score', value: 100 },
        ],
      });
    });

    it('should build $le and $ge predicates', () => {
      const filter = FilterBuilder.buildFilter({ min: { $ge: 10 }, max: { $le: 50 } });
      expect(filter).not.toBeNull();
      const andExpr = filter as { op: 'and'; exprs: RvfFilterExpr[] };
      expect(andExpr.exprs[0]).toEqual({ op: 'ge', field: 'min', value: 10 });
      expect(andExpr.exprs[1]).toEqual({ op: 'le', field: 'max', value: 50 });
    });

    it('should build $eq predicate explicitly', () => {
      const filter = FilterBuilder.buildFilter({ color: { $eq: 'blue' } });
      expect(filter).toEqual({ op: 'eq', field: 'color', value: 'blue' });
    });

    it('should return null for empty predicates', () => {
      expect(FilterBuilder.buildFilter({})).toBeNull();
    });

    it('should return null for null input', () => {
      expect(FilterBuilder.buildFilter(null as unknown as FilterPredicate)).toBeNull();
    });

    it('should skip null/undefined values', () => {
      const filter = FilterBuilder.buildFilter({ a: null as unknown as string, b: 'ok' });
      expect(filter).toEqual({ op: 'eq', field: 'b', value: 'ok' });
    });

    it('should use fluent builder API', () => {
      const builder = new FilterBuilder();
      const filter = builder.eq('name', 'test').gt('score', 5).build();
      expect(filter).not.toBeNull();
      const andExpr = filter as { op: 'and'; exprs: RvfFilterExpr[] };
      expect(andExpr.op).toBe('and');
      expect(andExpr.exprs).toHaveLength(2);
    });

    it('should support not() wrapper', () => {
      const builder = new FilterBuilder();
      const inner: RvfFilterExpr = { op: 'eq', fieldId: 0, value: 'test' };
      const filter = builder.not(inner).build();
      expect(filter).toEqual({ op: 'not', expr: inner });
    });

    it('should support in() operator', () => {
      const builder = new FilterBuilder();
      const filter = builder.in('status', ['a', 'b']).build();
      expect(filter).toEqual({ op: 'in', field: 'status', values: ['a', 'b'] });
    });

    it('should support range() operator', () => {
      const builder = new FilterBuilder();
      const filter = builder.range('temp', 20, 30).build();
      expect(filter).toEqual({
        op: 'and',
        exprs: [
          { op: 'ge', field: 'temp', value: 20 },
          { op: 'le', field: 'temp', value: 30 },
        ],
      });
    });

    it('should assign sequential field IDs', () => {
      const builder = new FilterBuilder();
      expect(builder.fieldId('a')).toBe(0);
      expect(builder.fieldId('b')).toBe(1);
      expect(builder.fieldId('a')).toBe(0); // cached
    });

    it('should expose field map', () => {
      const builder = new FilterBuilder();
      builder.eq('x', 1).eq('y', 2);
      const map = builder.getFieldMap();
      expect(map.get('x')).toBe(0);
      expect(map.get('y')).toBe(1);
    });

    it('should support reset()', () => {
      const builder = new FilterBuilder();
      builder.eq('x', 1);
      builder.reset();
      expect(builder.build()).toBeNull();
    });

    it('should throw when exceeding max filter expressions', () => {
      const predicates: Record<string, string> = {};
      for (let i = 0; i < 65; i++) {
        predicates[`field${i}`] = 'val';
      }
      expect(() => FilterBuilder.buildFilter(predicates)).toThrow('exceeds maximum');
    });
  });

  // ─── deleteByFilter ───

  describe('deleteByFilter', () => {
    it('should delegate to db.deleteByFilter with RvfFilterExpr', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();

      const filter: RvfFilterExpr = { op: 'eq', fieldId: 0, value: 'test' };
      const result = await backend.deleteByFilter(filter);

      expect(mockDb.deleteByFilter).toHaveBeenCalledWith(filter);
      expect(result.deleted).toBe(5);
      expect(result.epoch).toBe(3);
    });

    it('should accept predicate DSL object', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();

      await backend.deleteByFilter({ status: 'expired' });

      expect(mockDb.deleteByFilter).toHaveBeenCalledTimes(1);
      const calledFilter = mockDb.deleteByFilter.mock.calls[0][0];
      expect(calledFilter.op).toBe('eq');
      expect(calledFilter.value).toBe('expired');
    });

    it('should throw on empty predicate object', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();

      await expect(backend.deleteByFilter({})).rejects.toThrow('Cannot build filter');
    });

    it('should throw when not initialized', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await expect(
        backend.deleteByFilter({ op: 'eq', fieldId: 0, value: 'x' }),
      ).rejects.toThrow('not initialized');
    });

    it('should update cachedCount after delete', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();

      const statsBefore = backend.getStats();
      const countBefore = statsBefore.count;

      await backend.deleteByFilter({ status: 'old' });

      // cachedCount should have been decremented by 5 (the mock deleted count)
      const statsAfter = backend.getStats();
      expect(statsAfter.count).toBe(countBefore - 5);
    });
  });

  // ─── Read-Only Access ───

  describe('Read-only access', () => {
    it('should open a read-only store via static openReadonly()', async () => {
      const backend = await RvfBackend.openReadonly('/tmp/test-ro.rvf', {
        metric: 'cosine',
      });

      expect(mockOpenReadonlyFn).toHaveBeenCalledWith('/tmp/test-ro.rvf', 'auto');
      expect(backend.isInitialized()).toBe(true);
    });

    it('should determine dimension from the store', async () => {
      const backend = await RvfBackend.openReadonly('/tmp/test-ro.rvf');

      const dim = await backend.getDimension();
      expect(dim).toBe(64); // mockReadonlyDb.dimension returns 64
    });

    it('should validate path on openReadonly', async () => {
      await expect(
        RvfBackend.openReadonly('../../../etc/passwd'),
      ).rejects.toThrow('forbidden');
    });

    it('should accept :memory: path for openReadonly', async () => {
      const backend = await RvfBackend.openReadonly(':memory:');
      expect(backend.isInitialized()).toBe(true);
    });

    it('should respect rvfBackend option', async () => {
      await RvfBackend.openReadonly('/tmp/test-ro.rvf', { rvfBackend: 'wasm' });
      expect(mockOpenReadonlyFn).toHaveBeenCalledWith('/tmp/test-ro.rvf', 'wasm');
    });
  });

  // ─── Kernel Embed/Extract ───

  describe('Kernel embed/extract', () => {
    let backend: InstanceType<typeof RvfBackend>;

    beforeEach(async () => {
      backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();
    });

    it('should embed a kernel image and return segment ID', async () => {
      const image = new Uint8Array([0xDE, 0xAD, 0xBE, 0xEF]);
      const segId = await backend.embedKernel(1, 2, 0, image, 8080, 'root=/dev/sda1');

      expect(mockDb.embedKernel).toHaveBeenCalledWith(1, 2, 0, image, 8080, 'root=/dev/sda1');
      expect(segId).toBe(7);
    });

    it('should embed without optional cmdline', async () => {
      const image = new Uint8Array([0x01, 0x02]);
      await backend.embedKernel(0, 1, 0, image, 9090);

      expect(mockDb.embedKernel).toHaveBeenCalledWith(0, 1, 0, image, 9090, undefined);
    });

    it('should extract kernel data', async () => {
      const data = await backend.extractKernel();

      expect(data).not.toBeNull();
      expect(data!.header).toEqual(new Uint8Array([1, 2, 3]));
      expect(data!.image).toEqual(new Uint8Array([4, 5, 6]));
    });

    it('should return null when kernel not present', async () => {
      mockDb.extractKernel.mockResolvedValueOnce(null);
      const data = await backend.extractKernel();
      expect(data).toBeNull();
    });

    it('should return null when extractKernel throws', async () => {
      mockDb.extractKernel.mockRejectedValueOnce(new Error('no kernel segment'));
      const data = await backend.extractKernel();
      expect(data).toBeNull();
    });

    it('should throw when embedKernel fails', async () => {
      mockDb.embedKernel.mockRejectedValueOnce(new Error('store closed'));
      await expect(
        backend.embedKernel(0, 0, 0, new Uint8Array(1), 0),
      ).rejects.toThrow('embedKernel failed');
    });

    it('kernel round-trip: embed then extract', async () => {
      const image = new Uint8Array([0xCA, 0xFE]);
      await backend.embedKernel(1, 0, 0, image, 4000);
      const extracted = await backend.extractKernel();
      expect(extracted).not.toBeNull();
    });
  });

  // ─── eBPF Embed/Extract ───

  describe('eBPF embed/extract', () => {
    let backend: InstanceType<typeof RvfBackend>;

    beforeEach(async () => {
      backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();
    });

    it('should embed an eBPF program and return segment ID', async () => {
      const bytecode = new Uint8Array([0xEB, 0xBF, 0x00, 0x01]);
      const segId = await backend.embedEbpf(1, 2, 384, bytecode);

      expect(mockDb.embedEbpf).toHaveBeenCalledWith(1, 2, 384, bytecode, undefined);
      expect(segId).toBe(8);
    });

    it('should embed with BTF metadata', async () => {
      const bytecode = new Uint8Array([0x01]);
      const btf = new Uint8Array([0xBF]);
      await backend.embedEbpf(0, 0, 128, bytecode, btf);

      expect(mockDb.embedEbpf).toHaveBeenCalledWith(0, 0, 128, bytecode, btf);
    });

    it('should extract eBPF data', async () => {
      const data = await backend.extractEbpf();

      expect(data).not.toBeNull();
      expect(data!.header).toEqual(new Uint8Array([10, 20]));
      expect(data!.payload).toEqual(new Uint8Array([30, 40]));
    });

    it('should return null when eBPF not present', async () => {
      mockDb.extractEbpf.mockResolvedValueOnce(null);
      const data = await backend.extractEbpf();
      expect(data).toBeNull();
    });

    it('should return null when extractEbpf throws', async () => {
      mockDb.extractEbpf.mockRejectedValueOnce(new Error('no ebpf segment'));
      const data = await backend.extractEbpf();
      expect(data).toBeNull();
    });

    it('should throw when embedEbpf fails', async () => {
      mockDb.embedEbpf.mockRejectedValueOnce(new Error('invalid bytecode'));
      await expect(
        backend.embedEbpf(0, 0, 128, new Uint8Array(1)),
      ).rejects.toThrow('embedEbpf failed');
    });

    it('eBPF round-trip: embed then extract', async () => {
      const bytecode = new Uint8Array([0x01, 0x02, 0x03]);
      await backend.embedEbpf(1, 0, 256, bytecode);
      const extracted = await backend.extractEbpf();
      expect(extracted).not.toBeNull();
    });
  });

  // ─── Lineage Introspection ───

  describe('Lineage introspection', () => {
    let backend: InstanceType<typeof RvfBackend>;

    beforeEach(async () => {
      backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();
    });

    it('should return fileId from the store', async () => {
      const id = await backend.fileId();
      expect(id).toBe('abc123def456');
      expect(mockDb.fileId).toHaveBeenCalledTimes(1);
    });

    it('should return parentId from the store', async () => {
      const id = await backend.parentId();
      expect(id).toBe('0000000000000000');
      expect(mockDb.parentId).toHaveBeenCalledTimes(1);
    });

    it('should return lineageDepth from the store', async () => {
      const depth = await backend.lineageDepth();
      expect(depth).toBe(0);
      expect(mockDb.lineageDepth).toHaveBeenCalledTimes(1);
    });

    it('fileId should throw when not initialized', async () => {
      const uninit = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await expect(uninit.fileId()).rejects.toThrow('not initialized');
    });

    it('parentId should throw when not initialized', async () => {
      const uninit = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await expect(uninit.parentId()).rejects.toThrow('not initialized');
    });

    it('lineageDepth should throw when not initialized', async () => {
      const uninit = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await expect(uninit.lineageDepth()).rejects.toThrow('not initialized');
    });
  });

  // ─── Filter wiring in searchAsync ───

  describe('Filter expressions in searchAsync', () => {
    let backend: InstanceType<typeof RvfBackend>;

    beforeEach(async () => {
      backend = new RvfBackend({
        dimension: 4,
        metric: 'cosine',
        storagePath: ':memory:',
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);
      await backend.initialize();
    });

    it('should pass filter to db.query when provided in options', async () => {
      const query = new Float32Array([1, 0, 0, 0]);
      await backend.searchAsync(query, 10, {
        filter: { status: 'active' },
      });

      expect(mockDb.query).toHaveBeenCalledTimes(1);
      const opts = mockDb.query.mock.calls[0][2];
      expect(opts).toBeDefined();
      expect(opts.filter).toBeDefined();
      expect(opts.filter.op).toBe('eq');
      expect(opts.filter.value).toBe('active');
    });

    it('should not pass filter when not provided', async () => {
      const query = new Float32Array([1, 0, 0, 0]);
      await backend.searchAsync(query, 10);

      expect(mockDb.query).toHaveBeenCalledTimes(1);
      const opts = mockDb.query.mock.calls[0][2];
      expect(opts).toBeUndefined();
    });
  });

  // ─── Combined compression + hardware profile ───

  describe('Combined compression and hardware profile', () => {
    it('should pass both compression and profile together', async () => {
      const backend = new RvfBackend({
        dimension: 128,
        metric: 'cosine',
        storagePath: ':memory:',
        compression: 'product',
        hardwareProfile: 3,
      } as import('../../src/backends/rvf/RvfBackend.js').RvfConfig);

      await backend.initialize();

      const createOpts = mockCreateFn.mock.calls[0][1];
      expect(createOpts.compression).toBe('product');
      expect(createOpts.profile).toBe(3);
      expect(createOpts.dimensions).toBe(128);
      expect(createOpts.m).toBe(16);
      expect(createOpts.efConstruction).toBe(200);
    });
  });
});
