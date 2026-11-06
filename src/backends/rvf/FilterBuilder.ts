/**
 * FilterBuilder - Constructs RvfFilterExpr objects from a simple predicate DSL.
 *
 * Translates field-name predicates into RvfFilterExpr objects for
 * @ruvector/rvf >= 0.4.7. Leaves name their field by the metadata key used at
 * ingest (`{ op, field: 'status', value }`); rvf resolves the name against the
 * store's own field registry. Numeric ids assigned by this builder are per
 * instance and would not match that registry, so they are never emitted.
 *
 * Usage:
 *   const filter = new FilterBuilder()
 *     .eq('status', 'active')
 *     .gt('score', 0.5)
 *     .build();    // => RvfFilterExpr (AND of all predicates)
 *
 *   // Or build directly:
 *   FilterBuilder.buildFilter({ status: 'active', score: { $gt: 0.5 } });
 */

/** Re-export filter types from @ruvector/rvf for convenience */
export type RvfFilterValue = number | string | boolean;

/**
 * Names the field a leaf tests: `field` is the metadata key used at ingest
 * (what FilterBuilder emits); `fieldId` is the raw rvf field id from the
 * store's registry, accepted for callers that already hold one.
 */
export type RvfFilterField =
  | { field: string; fieldId?: undefined }
  | { fieldId: number; field?: undefined };

export type RvfFilterExpr =
  | ({ op: 'eq'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'ne'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'lt'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'le'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'gt'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'ge'; value: RvfFilterValue } & RvfFilterField)
  | ({ op: 'in'; values: RvfFilterValue[] } & RvfFilterField)
  /** rvf's native range is half-open: low <= value < high. */
  | ({ op: 'range'; low: RvfFilterValue; high: RvfFilterValue } & RvfFilterField)
  | { op: 'and'; exprs: RvfFilterExpr[] }
  | { op: 'or'; exprs: RvfFilterExpr[] }
  | { op: 'not'; expr: RvfFilterExpr };

/**
 * Simple predicate DSL value: either a plain value (eq), or an operator object.
 *
 * Examples:
 *   'active'          => eq('field', 'active')
 *   { $gt: 5 }        => gt('field', 5)
 *   { $in: [1,2,3] }  => in('field', [1,2,3])
 *   { $range: [0,10]} => range('field', 0, 10)   (inclusive: 0 <= v <= 10)
 *   { $ne: false }    => ne('field', false)
 */
export type PredicateValue =
  | RvfFilterValue
  | { $eq?: RvfFilterValue }
  | { $ne?: RvfFilterValue }
  | { $lt?: RvfFilterValue }
  | { $le?: RvfFilterValue }
  | { $gt?: RvfFilterValue }
  | { $ge?: RvfFilterValue }
  | { $in?: RvfFilterValue[] }
  | { $range?: [RvfFilterValue, RvfFilterValue] };

/** Predicate DSL: field name => predicate value */
export type FilterPredicate = Record<string, PredicateValue>;

/** Maximum number of filter expressions allowed (safety bound) */
const MAX_FILTER_EXPRS = 64;

/** Maximum metadata field-name length accepted in a filter leaf */
const MAX_FIELD_NAME_LENGTH = 256;

function checkFieldName(name: string): string {
  if (typeof name !== 'string' || name.length === 0) {
    throw new Error('Filter field name must be a non-empty string');
  }
  if (name.length > MAX_FIELD_NAME_LENGTH) {
    throw new Error(`Filter field name exceeds maximum length of ${MAX_FIELD_NAME_LENGTH}`);
  }
  return name;
}

/** Builds RvfFilterExpr trees whose leaves reference fields by name. */
export class FilterBuilder {
  private fieldMap: Map<string, number> = new Map();
  private nextFieldId = 0;
  private exprs: RvfFilterExpr[] = [];

  /**
   * Get or assign a per-instance numeric id for a field name.
   * @deprecated Ids from this map do not match rvf's stored field registry
   * and are not used in built expressions; leaves carry `field: name`.
   */
  fieldId(name: string): number {
    let id = this.fieldMap.get(name);
    if (id === undefined) {
      id = this.nextFieldId++;
      this.fieldMap.set(name, id);
    }
    return id;
  }

  /** @deprecated See fieldId(). */
  getFieldMap(): ReadonlyMap<string, number> {
    return this.fieldMap;
  }

  /** Validate a field name and record it in the (deprecated) field map */
  private name(field: string): string {
    this.fieldId(checkFieldName(field));
    return field;
  }

  /** Add an equality predicate */
  eq(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'eq', field: this.name(field), value });
    return this;
  }

  /** Add a not-equal predicate */
  ne(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'ne', field: this.name(field), value });
    return this;
  }

  /** Add a less-than predicate */
  lt(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'lt', field: this.name(field), value });
    return this;
  }

  /** Add a less-than-or-equal predicate */
  le(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'le', field: this.name(field), value });
    return this;
  }

  /** Add a greater-than predicate */
  gt(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'gt', field: this.name(field), value });
    return this;
  }

  /** Add a greater-than-or-equal predicate */
  ge(field: string, value: RvfFilterValue): this {
    this.addExpr({ op: 'ge', field: this.name(field), value });
    return this;
  }

  /** Add an IN predicate (value in set) */
  in(field: string, values: RvfFilterValue[]): this {
    this.addExpr({ op: 'in', field: this.name(field), values });
    return this;
  }

  /**
   * Add an inclusive range predicate (low <= value <= high). Emitted as
   * `and(ge, le)` because rvf's native `range` op excludes `high`.
   */
  range(field: string, low: RvfFilterValue, high: RvfFilterValue): this {
    const name = this.name(field);
    this.addExpr({
      op: 'and',
      exprs: [
        { op: 'ge', field: name, value: low },
        { op: 'le', field: name, value: high },
      ],
    });
    return this;
  }

  /** Add a NOT wrapper around an expression */
  not(expr: RvfFilterExpr): this {
    this.addExpr({ op: 'not', expr });
    return this;
  }

  /** Build the final filter expression (AND of all added predicates) */
  build(): RvfFilterExpr | null {
    if (this.exprs.length === 0) return null;
    if (this.exprs.length === 1) return this.exprs[0];
    return { op: 'and', exprs: [...this.exprs] };
  }

  /** Reset the builder state */
  reset(): this {
    this.exprs = [];
    return this;
  }

  /**
   * Build an RvfFilterExpr from a simple predicate DSL object.
   *
   * Example:
   *   buildFilter({ status: 'active', score: { $gt: 0.5 } })
   *   => AND(eq(status, 'active'), gt(score, 0.5))
   */
  static buildFilter(predicates: FilterPredicate): RvfFilterExpr | null {
    if (!predicates || typeof predicates !== 'object') return null;

    const keys = Object.keys(predicates);
    if (keys.length === 0) return null;

    const builder = new FilterBuilder();

    for (const field of keys) {
      if (builder.exprs.length >= MAX_FILTER_EXPRS) {
        throw new Error(`Filter expression count exceeds maximum of ${MAX_FILTER_EXPRS}`);
      }

      const val = predicates[field];

      if (val === null || val === undefined) continue;

      // Plain value => eq
      if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
        builder.eq(field, val);
        continue;
      }

      // Operator object
      if (typeof val === 'object') {
        const obj = val as Record<string, unknown>;
        if ('$eq' in obj) builder.eq(field, obj.$eq as RvfFilterValue);
        else if ('$ne' in obj) builder.ne(field, obj.$ne as RvfFilterValue);
        else if ('$lt' in obj) builder.lt(field, obj.$lt as RvfFilterValue);
        else if ('$le' in obj) builder.le(field, obj.$le as RvfFilterValue);
        else if ('$gt' in obj) builder.gt(field, obj.$gt as RvfFilterValue);
        else if ('$ge' in obj) builder.ge(field, obj.$ge as RvfFilterValue);
        else if ('$in' in obj) builder.in(field, obj.$in as RvfFilterValue[]);
        else if ('$range' in obj) {
          const range = obj.$range as [RvfFilterValue, RvfFilterValue];
          builder.range(field, range[0], range[1]);
        }
      }
    }

    return builder.build();
  }

  private addExpr(expr: RvfFilterExpr): void {
    if (this.exprs.length >= MAX_FILTER_EXPRS) {
      throw new Error(`Filter expression count exceeds maximum of ${MAX_FILTER_EXPRS}`);
    }
    this.exprs.push(expr);
  }
}
