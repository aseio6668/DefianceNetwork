//! Storage layer for DefianceNetwork data persistence

use std::path::Path;
use sqlx::{SqlitePool, Row};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::error::{DefianceError, Result};
use crate::user::User;
use crate::content::{Content, ContentMetadata};
use crate::streaming::BroadcastSession;

/// Storage manager for DefianceNetwork
pub struct DefianceStorage {
    pool: SqlitePool,
}

impl DefianceStorage {
    /// Create new storage instance
    pub async fn new(database_path: &str) -> Result<Self> {
        let pool = if database_path == ":memory:" {
            SqlitePool::connect("sqlite::memory:").await
        } else {
            // Ensure directory exists
            if let Some(parent) = Path::new(database_path).parent() {
                tokio::fs::create_dir_all(parent).await
                    .map_err(|e| DefianceError::Storage(format!("Failed to create directory: {}", e)))?;
            }
            
            let db_url = format!("sqlite://{}", database_path);
            SqlitePool::connect(&db_url).await
        }
        .map_err(|e| DefianceError::Storage(format!("Database connection failed: {}", e)))?;
        
        let storage = Self { pool };
        storage.initialize_schema().await?;
        
        Ok(storage)
    }
    
    /// Initialize database schema
    async fn initialize_schema(&self) -> Result<()> {
        // Users table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                opt_in_visibility BOOLEAN NOT NULL DEFAULT FALSE,
                reputation_score REAL NOT NULL DEFAULT 0.0,
                total_broadcasts INTEGER NOT NULL DEFAULT 0,
                total_watch_time INTEGER NOT NULL DEFAULT 0
            )
            "#
        )
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to create users table: {}", e)))?;
        
        // Content metadata table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS content_metadata (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                content_type TEXT NOT NULL,
                creator TEXT NOT NULL,
                creator_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                duration INTEGER,
                size INTEGER NOT NULL DEFAULT 0,
                checksum BLOB,
                available_qualities TEXT,
                tags TEXT,
                language TEXT,
                thumbnail BLOB,
                is_live BOOLEAN NOT NULL DEFAULT FALSE,
                viewer_count INTEGER NOT NULL DEFAULT 0,
                total_views INTEGER NOT NULL DEFAULT 0,
                rating REAL NOT NULL DEFAULT 0.0,
                ratings_count INTEGER NOT NULL DEFAULT 0
            )
            "#
        )
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to create content_metadata table: {}", e)))?;
        
        // Content chunks table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS content_chunks (
                content_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                data BLOB NOT NULL,
                checksum BLOB NOT NULL,
                quality TEXT NOT NULL,
                timestamp INTEGER,
                PRIMARY KEY (content_id, chunk_id, quality)
            )
            "#
        )
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to create content_chunks table: {}", e)))?;
        
        // Broadcasts table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS broadcasts (
                id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                ended_at INTEGER,
                max_viewers INTEGER,
                is_live BOOLEAN NOT NULL DEFAULT TRUE,
                quality_levels TEXT
            )
            "#
        )
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to create broadcasts table: {}", e)))?;
        
        // Settings table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            "#
        )
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to create settings table: {}", e)))?;
        
        tracing::info!("Database schema initialized");
        Ok(())
    }
    
    /// Store user in database
    pub async fn store_user(&mut self, user: &User) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO users 
            (id, username, created_at, last_seen, opt_in_visibility, reputation_score, total_broadcasts, total_watch_time)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#
        )
        .bind(user.id.to_string())
        .bind(&user.username.value)
        .bind(user.created_at)
        .bind(user.last_seen)
        .bind(user.opt_in_visibility)
        .bind(user.reputation_score)
        .bind(user.total_broadcasts as i64)
        .bind(user.total_watch_time as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to store user: {}", e)))?;
        
        Ok(())
    }
    
    /// Update user in database
    pub async fn update_user(&mut self, user: &User) -> Result<()> {
        self.store_user(user).await // Same as store since we use INSERT OR REPLACE
    }
    
    /// Get current user (assumes single user per node)
    pub async fn get_current_user(&self) -> Result<User> {
        let row = sqlx::query(
            "SELECT id, username, created_at, last_seen, opt_in_visibility, reputation_score, total_broadcasts, total_watch_time FROM users LIMIT 1"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to get current user: {}", e)))?;
        
        let user = User {
            id: Uuid::parse_str(row.get::<String, _>("id").as_str())
                .map_err(|e| DefianceError::Storage(format!("Invalid user ID: {}", e)))?,
            username: crate::user::Username {
                value: row.get("username"),
                generated_at: 0, // TODO: Store this separately
            },
            created_at: row.get("created_at"),
            last_seen: row.get("last_seen"),
            opt_in_visibility: row.get("opt_in_visibility"),
            reputation_score: row.get("reputation_score"),
            total_broadcasts: row.get::<i64, _>("total_broadcasts") as u64,
            total_watch_time: row.get::<i64, _>("total_watch_time") as u64,
        };
        
        Ok(user)
    }
    
    /// Store content metadata
    pub async fn store_content_metadata(&mut self, metadata: &ContentMetadata) -> Result<()> {
        let content_type_str = format!("{:?}", metadata.content_type);
        let qualities_str = serde_json::to_string(&metadata.available_qualities)
            .map_err(|e| DefianceError::Storage(format!("Failed to serialize qualities: {}", e)))?;
        let tags_str = serde_json::to_string(&metadata.tags)
            .map_err(|e| DefianceError::Storage(format!("Failed to serialize tags: {}", e)))?;
        
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO content_metadata 
            (id, title, description, content_type, creator, creator_id, created_at, duration, size, 
             available_qualities, tags, language, thumbnail, is_live, viewer_count, total_views, rating, ratings_count)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
            "#
        )
        .bind(metadata.id.to_string())
        .bind(&metadata.title)
        .bind(&metadata.description)
        .bind(content_type_str)
        .bind(&metadata.creator)
        .bind(metadata.creator_id.to_string())
        .bind(metadata.created_at)
        .bind(metadata.duration.map(|d| d as i64))
        .bind(metadata.size as i64)
        .bind(qualities_str)
        .bind(tags_str)
        .bind(&metadata.language)
        .bind(&metadata.thumbnail)
        .bind(metadata.is_live)
        .bind(metadata.viewer_count as i64)
        .bind(metadata.total_views as i64)
        .bind(metadata.rating)
        .bind(metadata.ratings_count as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to store content metadata: {}", e)))?;
        
        Ok(())
    }
    
    /// Store broadcast session
    pub async fn store_broadcast(&mut self, broadcast: &BroadcastSession) -> Result<()> {
        // First store the content metadata
        self.store_content_metadata(&broadcast.content.metadata).await?;
        
        // Then store the broadcast info
        let quality_levels_str = serde_json::to_string(&broadcast.quality_levels)
            .map_err(|e| DefianceError::Storage(format!("Failed to serialize quality levels: {}", e)))?;
        
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO broadcasts 
            (id, content_id, started_at, max_viewers, is_live, quality_levels)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#
        )
        .bind(broadcast.id.to_string())
        .bind(broadcast.content.metadata.id.to_string())
        .bind(broadcast.started_at)
        .bind(broadcast.max_viewers.map(|m| m as i64))
        .bind(broadcast.is_live)
        .bind(quality_levels_str)
        .execute(&self.pool)
        .await
        .map_err(|e| DefianceError::Storage(format!("Failed to store broadcast: {}", e)))?;
        
        Ok(())
    }
    
    /// Get setting value
    pub async fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let result = sqlx::query("SELECT value FROM settings WHERE key = ?1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DefianceError::Storage(format!("Failed to get setting: {}", e)))?;
        
        Ok(result.map(|row| row.get("value")))
    }
    
    /// Set setting value
    pub async fn set_setting(&mut self, key: &str, value: &str) -> Result<()> {
        sqlx::query("INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)")
            .bind(key)
            .bind(value)
            .execute(&self.pool)
            .await
            .map_err(|e| DefianceError::Storage(format!("Failed to set setting: {}", e)))?;
        
        Ok(())
    }
    
    /// Get database pool for advanced operations
    pub fn get_pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user::Username;
    
    #[tokio::test]
    async fn test_storage_initialization() {
        let storage = DefianceStorage::new(":memory:").await;
        assert!(storage.is_ok());
    }
    
    #[tokio::test]
    async fn test_user_storage() {
        let mut storage = DefianceStorage::new(":memory:").await.unwrap();
        
        let user = User {
            id: Uuid::new_v4(),
            username: Username::generate(),
            created_at: chrono::Utc::now().timestamp(),
            last_seen: chrono::Utc::now().timestamp(),
            opt_in_visibility: false,
            reputation_score: 0.0,
            total_broadcasts: 0,
            total_watch_time: 0,
        };
        
        assert!(storage.store_user(&user).await.is_ok());
        
        let retrieved_user = storage.get_current_user().await.unwrap();
        assert_eq!(user.id, retrieved_user.id);
        assert_eq!(user.username.value, retrieved_user.username.value);
    }
    
    #[tokio::test]
    async fn test_settings() {
        let mut storage = DefianceStorage::new(":memory:").await.unwrap();
        
        assert!(storage.set_setting("test_key", "test_value").await.is_ok());
        
        let value = storage.get_setting("test_key").await.unwrap();
        assert_eq!(value, Some("test_value".to_string()));
        
        let missing = storage.get_setting("missing_key").await.unwrap();
        assert_eq!(missing, None);
    }
}