/**
 * SyncCoordinator - Orchestrate AgentDB Synchronization
 *
 * Coordinates bidirectional synchronization between local and remote AgentDB instances.
 * Handles change detection, conflict resolution, batching, and progress tracking.
 *
 * Features:
 * - Detect changes since last sync
 * - Bidirectional sync (push and pull)
 * - Conflict resolution strategies
 * - Batch operations for efficiency
 * - Progress tracking and reporting
 * - Comprehensive error handling
 * - Sync state persistence
 */

import chalk from 'chalk';
import { QUICClient, SyncOptions, SyncResult, PushResult } from './QUICClient.js';
import { QUICServer, SyncRequest } from './QUICServer.js';

// Database type from db-fallback
type Database = any;

export interface SyncCoordinatorConfig {
  db: Database;
  client?: QUICClient;
  server?: QUICServer;
  conflictStrategy?: 'local-wins' | 'remote-wins' | 'latest-wins' | 'merge';
  batchSize?: number;
  autoSync?: boolean;
  syncIntervalMs?: number;
}

export interface SyncState {
  lastSyncAt: number;
  lastEpisodeSync: number;
  lastSkillSync: number;
  lastEdgeSync: number;
  totalItemsSynced: number;
  totalBytesSynced: number;
  syncCount: number;
  lastError?: string;
}

export interface SyncProgress {
  phase: 'detecting' | 'pushing' | 'pulling' | 'resolving' | 'applying' | 'completed' | 'error';
  current: number;
  total: number;
  itemType?: string;
  message?: string;
  error?: string;
}

export interface SyncReport {
  success: boolean;
  startTime: number;
  endTime: number;
  durationMs: number;
  itemsPushed: number;
  itemsPulled: number;
  conflictsResolved: number;
  errors: string[];
  bytesTransferred: number;
}

export class SyncCoordinator {
  private db: Database;
  private client?: QUICClient;
  private server?: QUICServer;
  private config: Required<Omit<SyncCoordinatorConfig, 'db' | 'client' | 'server'>>;
  private syncState: SyncState;
  private isSyncing: boolean = false;
  private autoSyncInterval: NodeJS.Timeout | null = null;

  constructor(config: SyncCoordinatorConfig) {
    this.db = config.db;
    this.client = config.client;
    this.server = config.server;
    this.config = {
      conflictStrategy: config.conflictStrategy || 'latest-wins',
      batchSize: config.batchSize || 100,
      autoSync: config.autoSync || false,
      syncIntervalMs: config.syncIntervalMs || 60000, // 1 minute
    };

    // Load sync state
    this.syncState = this.loadSyncState();

    // Start auto-sync if enabled
    if (this.config.autoSync) {
      this.startAutoSync();
    }
  }

  /**
   * Perform bidirectional synchronization
   */
  async sync(onProgress?: (progress: SyncProgress) => void): Promise<SyncReport> {
    if (this.isSyncing) {
      throw new Error('Sync already in progress');
    }

    if (!this.client) {
      throw new Error('QUICClient not configured');
    }

    this.isSyncing = true;
    const startTime = Date.now();
    const errors: string[] = [];
    let itemsPushed = 0;
    let itemsPulled = 0;
    let conflictsResolved = 0;
    let bytesTransferred = 0;

    try {
      console.log(chalk.blue('🔄 Starting synchronization...'));

      // Phase 1: Detect changes
      onProgress?.({ phase: 'detecting', current: 0, total: 100, message: 'Detecting changes...' });
      const changes = await this.detectChanges();
      console.log(chalk.gray(`  Changes detected: ${changes.episodes.length + changes.skills.length + changes.edges.length} items`));

      // Phase 2: Push changes to remote
      if (changes.episodes.length > 0 || changes.skills.length > 0 || changes.edges.length > 0) {
        onProgress?.({ phase: 'pushing', current: 0, total: changes.episodes.length + changes.skills.length + changes.edges.length });
        const pushResult = await this.pushChanges(changes, onProgress);
        itemsPushed = pushResult.itemsPushed;
        bytesTransferred += pushResult.bytesTransferred;
        errors.push(...pushResult.errors);
      }

      // Phase 3: Pull changes from remote
      onProgress?.({ phase: 'pulling', current: 0, total: 100, message: 'Pulling remote changes...' });
      const pullResult = await this.pullChanges(onProgress);
      itemsPulled = pullResult.itemsPulled;
      bytesTransferred += pullResult.bytesTransferred;
      errors.push(...pullResult.errors);

      // Phase 4: Resolve conflicts
      if (pullResult.conflicts && pullResult.conflicts.length > 0) {
        onProgress?.({ phase: 'resolving', current: 0, total: pullResult.conflicts.length, message: 'Resolving conflicts...' });
        conflictsResolved = await this.resolveConflicts(pullResult.conflicts);
      }

      // Phase 5: Apply changes
      onProgress?.({ phase: 'applying', current: 0, total: itemsPulled, message: 'Applying changes...' });
      await this.applyChanges(pullResult.data);

      // Update sync state
      this.syncState.lastSyncAt = Date.now();
      this.syncState.totalItemsSynced += itemsPushed + itemsPulled;
      this.syncState.totalBytesSynced += bytesTransferred;
      this.syncState.syncCount++;
      this.syncState.lastError = errors.length > 0 ? errors[0] : undefined;
      this.saveSyncState();

      const endTime = Date.now();
      const durationMs = endTime - startTime;

      console.log(chalk.green('✓ Synchronization completed'));
      console.log(chalk.gray(`  Items pushed: ${itemsPushed}`));
      console.log(chalk.gray(`  Items pulled: ${itemsPulled}`));
      console.log(chalk.gray(`  Conflicts resolved: ${conflictsResolved}`));
      console.log(chalk.gray(`  Duration: ${durationMs}ms`));

      onProgress?.({ phase: 'completed', current: 100, total: 100, message: 'Sync completed' });

      return {
        success: errors.length === 0,
        startTime,
        endTime,
        durationMs,
        itemsPushed,
        itemsPulled,
        conflictsResolved,
        errors,
        bytesTransferred,
      };
    } catch (error) {
      const err = error as Error;
      const endTime = Date.now();

      console.error(chalk.red('✗ Synchronization failed:'), err.message);
      errors.push(err.message);

      onProgress?.({ phase: 'error', current: 0, total: 0, error: err.message });

      return {
        success: false,
        startTime,
        endTime,
        durationMs: endTime - startTime,
        itemsPushed,
        itemsPulled,
        conflictsResolved,
        errors,
        bytesTransferred,
      };
    } finally {
      this.isSyncing = false;
    }
  }

  /**
   * Detect changes since last sync
   */
  private async detectChanges(): Promise<{
    episodes: any[];
    skills: any[];
    edges: any[];
  }> {
    const { lastEpisodeSync, lastSkillSync, lastEdgeSync } = this.syncState;

    // Detect new/modified episodes
    const episodes = this.db
      .prepare('SELECT * FROM episodes WHERE ts > ?')
      .all(lastEpisodeSync);

    // Detect new/modified skills
    const skills = this.db
      .prepare('SELECT * FROM skills WHERE ts > ?')
      .all(lastSkillSync);

    // Detect new/modified edges
    const edges = this.db
      .prepare('SELECT * FROM skill_edges WHERE ts > ?')
      .all(lastEdgeSync);

    return { episodes, skills, edges };
  }

  /**
   * Push local changes to remote
   */
  private async pushChanges(
    changes: { episodes: any[]; skills: any[]; edges: any[] },
    onProgress?: (progress: SyncProgress) => void
  ): Promise<{ itemsPushed: number; bytesTransferred: number; errors: string[] }> {
    if (!this.client) {
      throw new Error('QUICClient not configured');
    }

    const errors: string[] = [];
    let itemsPushed = 0;
    let bytesTransferred = 0;
    const totalItems = changes.episodes.length + changes.skills.length + changes.edges.length;

    try {
      console.log(chalk.blue('📤 Pushing changes to remote...'));
      console.log(chalk.gray(`  Episodes: ${changes.episodes.length}`));
      console.log(chalk.gray(`  Skills: ${changes.skills.length}`));
      console.log(chalk.gray(`  Edges: ${changes.edges.length}`));

      // Push episodes
      if (changes.episodes.length > 0) {
        onProgress?.({
          phase: 'pushing',
          current: itemsPushed,
          total: totalItems,
          itemType: 'episodes',
          message: `Pushing ${changes.episodes.length} episodes...`,
        });

        const episodesResult = await this.client.push({
          type: 'episodes',
          data: changes.episodes,
          batchSize: this.config.batchSize,
          onProgress: (progress) => {
            onProgress?.({
              phase: 'pushing',
              current: itemsPushed + (progress.itemsPushed || 0),
              total: totalItems,
              itemType: 'episodes',
              message: `Pushing episodes: batch ${progress.currentBatch || 0}/${progress.totalBatches || 0}`,
            });
          },
        });

        if (episodesResult.success) {
          itemsPushed += episodesResult.itemsPushed;
          bytesTransferred += episodesResult.bytesTransferred;
          console.log(chalk.gray(`  Episodes pushed: ${episodesResult.itemsPushed}`));
        } else {
          errors.push(`Episodes push failed: ${episodesResult.error || 'Unknown error'}`);
          console.log(chalk.yellow(`  Episodes push failed: ${episodesResult.error}`));
        }
      }

      // Push skills
      if (changes.skills.length > 0) {
        onProgress?.({
          phase: 'pushing',
          current: itemsPushed,
          total: totalItems,
          itemType: 'skills',
          message: `Pushing ${changes.skills.length} skills...`,
        });

        const skillsResult = await this.client.push({
          type: 'skills',
          data: changes.skills,
          batchSize: this.config.batchSize,
          onProgress: (progress) => {
            onProgress?.({
              phase: 'pushing',
              current: itemsPushed + (progress.itemsPushed || 0),
              total: totalItems,
              itemType: 'skills',
              message: `Pushing skills: batch ${progress.currentBatch || 0}/${progress.totalBatches || 0}`,
            });
          },
        });

        if (skillsResult.success) {
          itemsPushed += skillsResult.itemsPushed;
          bytesTransferred += skillsResult.bytesTransferred;
          console.log(chalk.gray(`  Skills pushed: ${skillsResult.itemsPushed}`));
        } else {
          errors.push(`Skills push failed: ${skillsResult.error || 'Unknown error'}`);
          console.log(chalk.yellow(`  Skills push failed: ${skillsResult.error}`));
        }
      }

      // Push edges
      if (changes.edges.length > 0) {
        onProgress?.({
          phase: 'pushing',
          current: itemsPushed,
          total: totalItems,
          itemType: 'edges',
          message: `Pushing ${changes.edges.length} edges...`,
        });

        const edgesResult = await this.client.push({
          type: 'edges',
          data: changes.edges,
          batchSize: this.config.batchSize,
          onProgress: (progress) => {
            onProgress?.({
              phase: 'pushing',
              current: itemsPushed + (progress.itemsPushed || 0),
              total: totalItems,
              itemType: 'edges',
              message: `Pushing edges: batch ${progress.currentBatch || 0}/${progress.totalBatches || 0}`,
            });
          },
        });

        if (edgesResult.success) {
          itemsPushed += edgesResult.itemsPushed;
          bytesTransferred += edgesResult.bytesTransferred;
          console.log(chalk.gray(`  Edges pushed: ${edgesResult.itemsPushed}`));
        } else {
          errors.push(`Edges push failed: ${edgesResult.error || 'Unknown error'}`);
          console.log(chalk.yellow(`  Edges push failed: ${edgesResult.error}`));
        }
      }

      console.log(chalk.green(`✓ Push completed: ${itemsPushed} items, ${bytesTransferred} bytes`));
    } catch (error) {
      const err = error as Error;
      errors.push(err.message);
      console.error(chalk.red('✗ Push failed:'), err.message);
    }

    return { itemsPushed, bytesTransferred, errors };
  }

  /**
   * Pull changes from remote
   */
  private async pullChanges(
    onProgress?: (progress: SyncProgress) => void
  ): Promise<{
    itemsPulled: number;
    bytesTransferred: number;
    data: any;
    conflicts?: any[];
    errors: string[];
  }> {
    if (!this.client) {
      throw new Error('QUICClient not configured');
    }

    const errors: string[] = [];
    let itemsPulled = 0;
    let bytesTransferred = 0;
    const allData: any = { episodes: [], skills: [], edges: [] };

    try {
      // Pull episodes
      const episodesResult = await this.client.sync({
        type: 'episodes',
        since: this.syncState.lastEpisodeSync,
        batchSize: this.config.batchSize,
        onProgress: (progress) => {
          onProgress?.({
            phase: 'pulling',
            current: progress.itemsSynced || 0,
            total: 100,
            itemType: 'episodes',
          });
        },
      });

      if (episodesResult.success && episodesResult.data) {
        allData.episodes = episodesResult.data;
        itemsPulled += episodesResult.itemsReceived;
        bytesTransferred += episodesResult.bytesTransferred;
        this.syncState.lastEpisodeSync = Date.now();
      } else {
        errors.push(episodesResult.error || 'Failed to sync episodes');
      }

      // Pull skills
      const skillsResult = await this.client.sync({
        type: 'skills',
        since: this.syncState.lastSkillSync,
        batchSize: this.config.batchSize,
        onProgress: (progress) => {
          onProgress?.({
            phase: 'pulling',
            current: progress.itemsSynced || 0,
            total: 100,
            itemType: 'skills',
          });
        },
      });

      if (skillsResult.success && skillsResult.data) {
        allData.skills = skillsResult.data;
        itemsPulled += skillsResult.itemsReceived;
        bytesTransferred += skillsResult.bytesTransferred;
        this.syncState.lastSkillSync = Date.now();
      } else {
        errors.push(skillsResult.error || 'Failed to sync skills');
      }

      // Pull edges
      const edgesResult = await this.client.sync({
        type: 'edges',
        since: this.syncState.lastEdgeSync,
        batchSize: this.config.batchSize,
        onProgress: (progress) => {
          onProgress?.({
            phase: 'pulling',
            current: progress.itemsSynced || 0,
            total: 100,
            itemType: 'edges',
          });
        },
      });

      if (edgesResult.success && edgesResult.data) {
        allData.edges = edgesResult.data;
        itemsPulled += edgesResult.itemsReceived;
        bytesTransferred += edgesResult.bytesTransferred;
        this.syncState.lastEdgeSync = Date.now();
      } else {
        errors.push(edgesResult.error || 'Failed to sync edges');
      }
    } catch (error) {
      const err = error as Error;
      errors.push(err.message);
    }

    return {
      itemsPulled,
      bytesTransferred,
      data: allData,
      errors,
    };
  }

  /**
   * Resolve conflicts between local and remote data
   */
  private async resolveConflicts(conflicts: any[]): Promise<number> {
    let resolved = 0;

    for (const conflict of conflicts) {
      switch (this.config.conflictStrategy) {
        case 'local-wins':
          // Keep local version
          break;
        case 'remote-wins':
          // Keep remote version
          resolved++;
          break;
        case 'latest-wins':
          // Keep version with latest timestamp
          if (conflict.remote.ts > conflict.local.ts) {
            resolved++;
          }
          break;
        case 'merge':
          // Attempt to merge (simplified)
          resolved++;
          break;
      }
    }

    return resolved;
  }

  /**
   * Apply pulled changes to local database
   */
  private async applyChanges(data: any): Promise<void> {
    // Apply episodes
    if (data.episodes && data.episodes.length > 0) {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO episodes (
          id, ts, session_id, task, input, output, critique, reward, success,
          latency_ms, tokens_used, tags, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      for (const episode of data.episodes) {
        stmt.run(
          episode.id,
          episode.ts,
          episode.sessionId,
          episode.task,
          episode.input,
          episode.output,
          episode.critique,
          episode.reward,
          episode.success ? 1 : 0,
          episode.latencyMs,
          episode.tokensUsed,
          JSON.stringify(episode.tags),
          JSON.stringify(episode.metadata)
        );
      }
    }

    // Apply skills
    if (data.skills && data.skills.length > 0) {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO skills (
          id, ts, name, description, code, success_rate, usage_count,
          avg_reward, tags, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      for (const skill of data.skills) {
        stmt.run(
          skill.id,
          skill.ts,
          skill.name,
          skill.description,
          skill.code,
          skill.successRate,
          skill.usageCount,
          skill.avgReward,
          JSON.stringify(skill.tags),
          JSON.stringify(skill.metadata)
        );
      }
    }

    // Apply edges
    if (data.edges && data.edges.length > 0) {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO skill_edges (
          id, ts, from_skill_id, to_skill_id, weight, co_occurrences
        ) VALUES (?, ?, ?, ?, ?, ?)
      `);

      for (const edge of data.edges) {
        stmt.run(
          edge.id,
          edge.ts,
          edge.fromSkillId,
          edge.toSkillId,
          edge.weight,
          edge.coOccurrences
        );
      }
    }
  }

  /**
   * Load sync state from database
   */
  private loadSyncState(): SyncState {
    try {
      const row = this.db
        .prepare('SELECT * FROM sync_state WHERE id = 1')
        .get();

      if (row) {
        return {
          lastSyncAt: row.last_sync_at,
          lastEpisodeSync: row.last_episode_sync,
          lastSkillSync: row.last_skill_sync,
          lastEdgeSync: row.last_edge_sync,
          totalItemsSynced: row.total_items_synced,
          totalBytesSynced: row.total_bytes_synced,
          syncCount: row.sync_count,
          lastError: row.last_error,
        };
      }
    } catch (error) {
      // Table might not exist yet
    }

    return {
      lastSyncAt: 0,
      lastEpisodeSync: 0,
      lastSkillSync: 0,
      lastEdgeSync: 0,
      totalItemsSynced: 0,
      totalBytesSynced: 0,
      syncCount: 0,
    };
  }

  /**
   * Save sync state to database
   */
  private saveSyncState(): void {
    try {
      // Create table if not exists
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS sync_state (
          id INTEGER PRIMARY KEY,
          last_sync_at INTEGER,
          last_episode_sync INTEGER,
          last_skill_sync INTEGER,
          last_edge_sync INTEGER,
          total_items_synced INTEGER,
          total_bytes_synced INTEGER,
          sync_count INTEGER,
          last_error TEXT
        )
      `);

      // Upsert state
      this.db
        .prepare(`
          INSERT OR REPLACE INTO sync_state (
            id, last_sync_at, last_episode_sync, last_skill_sync, last_edge_sync,
            total_items_synced, total_bytes_synced, sync_count, last_error
          ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        `)
        .run(
          this.syncState.lastSyncAt,
          this.syncState.lastEpisodeSync,
          this.syncState.lastSkillSync,
          this.syncState.lastEdgeSync,
          this.syncState.totalItemsSynced,
          this.syncState.totalBytesSynced,
          this.syncState.syncCount,
          this.syncState.lastError || null
        );
    } catch (error) {
      const err = error as Error;
      console.error(chalk.red('✗ Failed to save sync state:'), err.message);
    }
  }

  /**
   * Start automatic synchronization
   */
  private startAutoSync(): void {
    if (this.autoSyncInterval) {
      return;
    }

    console.log(chalk.blue(`🔄 Auto-sync enabled (interval: ${this.config.syncIntervalMs}ms)`));

    this.autoSyncInterval = setInterval(async () => {
      try {
        await this.sync();
      } catch (error) {
        const err = error as Error;
        console.error(chalk.red('✗ Auto-sync failed:'), err.message);
      }
    }, this.config.syncIntervalMs);
    // Background sync must not keep an otherwise finished process alive
    this.autoSyncInterval.unref?.();
  }

  /**
   * Stop automatic synchronization
   */
  stopAutoSync(): void {
    if (this.autoSyncInterval) {
      clearInterval(this.autoSyncInterval);
      this.autoSyncInterval = null;
      console.log(chalk.blue('🔄 Auto-sync disabled'));
    }
  }

  /**
   * Get sync state
   */
  getSyncState(): SyncState {
    return { ...this.syncState };
  }

  /**
   * Get sync status
   */
  getStatus(): {
    isSyncing: boolean;
    autoSyncEnabled: boolean;
    state: SyncState;
  } {
    return {
      isSyncing: this.isSyncing,
      autoSyncEnabled: this.autoSyncInterval !== null,
      state: this.getSyncState(),
    };
  }
}
