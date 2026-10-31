/**
 * Advanced Metadata Filtering
 *
 * Implements MongoDB-style query operators for filtering
 * episodes and patterns based on metadata fields.
 *
 * Supported operators:
 * - $eq: Equal to
 * - $ne: Not equal to
 * - $gt: Greater than
 * - $gte: Greater than or equal to
 * - $lt: Less than
 * - $lte: Less than or equal to
 * - $in: Value is in array
 * - $nin: Value is not in array
 * - $contains: String/array contains value
 * - $exists: Field exists
 */

const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

function assertIdentifier(name: string, what: string): string {
  if (typeof name !== 'string' || !IDENTIFIER.test(name)) {
    throw new Error(`Invalid ${what} in metadata filter: ${JSON.stringify(name)}`);
  }
  return name;
}

export type FilterOperator = '$eq' | '$ne' | '$gt' | '$gte' | '$lt' | '$lte' | '$in' | '$nin' | '$contains' | '$exists';

export type FilterValue = string | number | boolean | string[] | number[] | { [op in FilterOperator]?: any };

export interface MetadataFilters {
  [field: string]: FilterValue;
}

export interface FilterableItem {
  metadata?: any;
  [key: string]: any;
}

export class MetadataFilter {
  /**
   * Apply filters to a collection of items
   *
   * @param items - Items to filter
   * @param filters - MongoDB-style filter object
   * @returns Filtered items
   */
  static apply<T extends FilterableItem>(items: T[], filters: MetadataFilters): T[] {
    if (!filters || Object.keys(filters).length === 0) {
      return items;
    }

    return items.filter(item => this.matchesFilters(item, filters));
  }

  /**
   * Check if an item matches all filters
   */
  private static matchesFilters(item: FilterableItem, filters: MetadataFilters): boolean {
    for (const [field, filter] of Object.entries(filters)) {
      if (!this.matchesFilter(item, field, filter)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Check if an item matches a single filter
   */
  private static matchesFilter(item: FilterableItem, field: string, filter: FilterValue): boolean {
    // Get field value (supports nested paths like "metadata.year")
    const value = this.getFieldValue(item, field);

    // Handle simple equality
    if (typeof filter !== 'object' || Array.isArray(filter)) {
      return value === filter;
    }

    // Handle operator-based filters
    const operators = filter as { [op in FilterOperator]?: any };

    for (const [operator, operand] of Object.entries(operators)) {
      switch (operator as FilterOperator) {
        case '$eq':
          if (value !== operand) return false;
          break;

        case '$ne':
          if (value === operand) return false;
          break;

        case '$gt':
          if (!(value > operand)) return false;
          break;

        case '$gte':
          if (!(value >= operand)) return false;
          break;

        case '$lt':
          if (!(value < operand)) return false;
          break;

        case '$lte':
          if (!(value <= operand)) return false;
          break;

        case '$in':
          if (!Array.isArray(operand) || !operand.includes(value)) return false;
          break;

        case '$nin':
          if (!Array.isArray(operand) || operand.includes(value)) return false;
          break;

        case '$contains':
          if (typeof value === 'string') {
            if (!value.includes(operand)) return false;
          } else if (Array.isArray(value)) {
            if (!value.includes(operand)) return false;
          } else {
            return false;
          }
          break;

        case '$exists':
          const exists = value !== undefined && value !== null;
          if (exists !== operand) return false;
          break;

        default:
          console.warn(`Unknown operator: ${operator}`);
          return false;
      }
    }

    return true;
  }

  /**
   * Get field value from item (supports nested paths)
   */
  private static getFieldValue(item: FilterableItem, field: string): any {
    const parts = field.split('.');
    let value: any = item;

    for (const part of parts) {
      if (value === null || value === undefined) {
        return undefined;
      }

      // Parse metadata JSON if needed
      if (part === 'metadata' && typeof value.metadata === 'string') {
        try {
          value.metadata = JSON.parse(value.metadata);
        } catch (e) {
          return undefined;
        }
      }

      value = value[part];
    }

    return value;
  }

  /**
   * Build SQL WHERE clause from filters (for database queries)
   *
   * @param filters - Metadata filters
   * @param tableName - Table name for column references
   * @returns SQL WHERE clause and parameters
   */
  static toSQL(filters: MetadataFilters, tableName: string = ''): { where: string; params: any[] } {
    const conditions: string[] = [];
    const params: any[] = [];
    // Table, column and JSON-path segments are identifiers only; the JSON
    // path itself is bound as a parameter rather than spliced into SQL.
    const prefix = tableName ? `${assertIdentifier(tableName, 'table name')}.` : '';

    for (const [field, filter] of Object.entries(filters)) {
      // For metadata fields, use JSON extraction
      const isMetadata = field.startsWith('metadata.');
      let columnRef: string;
      let refParams: any[];
      if (isMetadata) {
        const segments = field.slice(9).split('.');
        segments.forEach((segment) => assertIdentifier(segment, 'metadata path segment'));
        columnRef = `json_extract(${prefix}metadata, ?)`;
        refParams = [`$.${segments.join('.')}`];
      } else {
        columnRef = `${prefix}${assertIdentifier(field, 'field')}`;
        refParams = [];
      }
      const add = (condition: string, ...values: any[]): void => {
        conditions.push(condition);
        params.push(...refParams, ...values);
      };

      if (typeof filter !== 'object' || Array.isArray(filter)) {
        // Simple equality
        add(`${columnRef} = ?`, filter);
      } else {
        // Operator-based filters
        const operators = filter as { [op in FilterOperator]?: any };

        for (const [operator, operand] of Object.entries(operators)) {
          switch (operator as FilterOperator) {
            case '$eq':
              add(`${columnRef} = ?`, operand);
              break;

            case '$ne':
              add(`${columnRef} != ?`, operand);
              break;

            case '$gt':
              add(`${columnRef} > ?`, operand);
              break;

            case '$gte':
              add(`${columnRef} >= ?`, operand);
              break;

            case '$lt':
              add(`${columnRef} < ?`, operand);
              break;

            case '$lte':
              add(`${columnRef} <= ?`, operand);
              break;

            case '$in':
              if (Array.isArray(operand)) {
                const placeholders = operand.map(() => '?').join(', ');
                add(`${columnRef} IN (${placeholders})`, ...operand);
              }
              break;

            case '$nin':
              if (Array.isArray(operand)) {
                const placeholders = operand.map(() => '?').join(', ');
                add(`${columnRef} NOT IN (${placeholders})`, ...operand);
              }
              break;

            case '$contains':
              add(`${columnRef} LIKE ?`, `%${operand}%`);
              break;

            case '$exists':
              if (operand) {
                add(`${columnRef} IS NOT NULL`);
              } else {
                add(`${columnRef} IS NULL`);
              }
              break;
          }
        }
      }
    }

    const where = conditions.length > 0 ? conditions.join(' AND ') : '1=1';
    return { where, params };
  }

  /**
   * Validate filter object
   */
  static validate(filters: MetadataFilters): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    for (const [field, filter] of Object.entries(filters)) {
      if (!field || field.trim() === '') {
        errors.push('Filter field name cannot be empty');
      }

      if (typeof filter === 'object' && !Array.isArray(filter)) {
        const operators = Object.keys(filter);
        for (const op of operators) {
          if (!op.startsWith('$')) {
            errors.push(`Invalid operator: ${op} (must start with $)`);
          }
        }
      }
    }

    return { valid: errors.length === 0, errors };
  }
}
