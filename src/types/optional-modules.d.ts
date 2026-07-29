declare module 'better-sqlite3' {
  class Database {
    constructor(filename?: string, options?: unknown);
    [key: string]: any;
  }
  export { Database };
  export default Database;
}
