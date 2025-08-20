//! # Defiance Audigy
//! 
//! Audio streaming component for DefianceNetwork supporting educational content,
//! podcasts, audiobooks, and lectures with the .augy file format for content discovery.

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use url::Url;
use anyhow::Result;

pub mod augy;
pub mod player;
pub mod discovery;
pub mod streaming;

// Re-export commonly used types
pub use augy::{AugyFile, AugyEntry, AudioMetadata, AudioGenre, AudioType};
pub use player::{AudioPlayer, PlaybackState, PlayerEvent};
pub use discovery::{NetworkDiscovery, ContentSource};
pub use streaming::{AudioStreamer, StreamingSession};

/// Audigy engine managing audio content and playback
pub struct AudiogyEngine {
    player: AudioPlayer,
    discovery: NetworkDiscovery,
    streamer: AudioStreamer,
    content_cache: HashMap<Uuid, AudioContent>,
    active_sessions: HashMap<Uuid, StreamingSession>,
    config: AudiogyConfig,
}

/// Audio content representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    pub id: Uuid,
    pub metadata: AudioMetadata,
    pub source_urls: Vec<Url>,
    pub local_path: Option<String>,
    pub download_progress: f32,
    pub is_cached: bool,
}

/// Audigy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudiogyConfig {
    pub cache_directory: String,
    pub max_cache_size_gb: u64,
    pub auto_download: bool,
    pub preferred_quality: streaming::StreamingQuality,
    pub network_sources: Vec<String>, // URLs to .augy files or network endpoints
    pub enable_discovery: bool,
}

// Audio quality enum moved to streaming module as StreamingQuality

impl Default for AudiogyConfig {
    fn default() -> Self {
        Self {
            cache_directory: "./audigy_cache".to_string(),
            max_cache_size_gb: 5,
            auto_download: false,
            preferred_quality: streaming::StreamingQuality::High,
            network_sources: vec![
                "https://audigy.defiancenetwork.org/featured.augy".to_string(),
                "https://podcasts.example.org/catalog.augy".to_string(),
            ],
            enable_discovery: true,
        }
    }
}

impl AudiogyEngine {
    /// Create new Audigy engine
    pub async fn new(config: AudiogyConfig) -> Result<Self> {
        tracing::info!("Initializing Audigy engine");

        // Ensure cache directory exists
        tokio::fs::create_dir_all(&config.cache_directory).await?;

        let player = AudioPlayer::new().await?;
        let discovery = NetworkDiscovery::new(config.network_sources.clone()).await?;
        let streamer = AudioStreamer::new().await?;

        Ok(Self {
            player,
            discovery,
            streamer,
            content_cache: HashMap::new(),
            active_sessions: HashMap::new(),
            config,
        })
    }

    /// Start the Audigy engine
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting Audigy engine");

        // Start player
        self.player.start().await?;

        // Start discovery if enabled
        if self.config.enable_discovery {
            self.discovery.start().await?;
            self.refresh_content_sources().await?;
        }

        // Start streamer
        self.streamer.start().await?;

        Ok(())
    }

    /// Stop the Audigy engine
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping Audigy engine");

        // Stop all active sessions
        for (_, session) in self.active_sessions.drain() {
            self.streamer.stop_session(session.id).await?;
        }

        // Stop components
        self.player.stop().await?;
        self.discovery.stop().await?;
        self.streamer.stop().await?;

        Ok(())
    }

    /// Load content from .augy file
    pub async fn load_augy_file(&mut self, path: &Path) -> Result<Vec<AudioContent>> {
        tracing::info!("Loading .augy file: {:?}", path);

        let augy_file = AugyFile::load_from_file(path).await?;
        let mut content_list = Vec::new();

        for entry in augy_file.entries {
            let content = AudioContent {
                id: Uuid::new_v4(),
                metadata: entry.metadata,
                source_urls: entry.uri_points,
                local_path: None,
                download_progress: 0.0,
                is_cached: false,
            };

            self.content_cache.insert(content.id, content.clone());
            content_list.push(content);
        }

        tracing::info!("Loaded {} audio items from .augy file", content_list.len());
        Ok(content_list)
    }

    /// Search for audio content by query
    pub async fn search(&self, query: &str, genre: Option<AudioGenre>) -> Result<Vec<AudioContent>> {
        let mut results = Vec::new();

        // Search in cached content
        for content in self.content_cache.values() {
            let matches_query = content.metadata.title.to_lowercase().contains(&query.to_lowercase())
                || content.metadata.description.to_lowercase().contains(&query.to_lowercase())
                || content.metadata.author.to_lowercase().contains(&query.to_lowercase());

            let matches_genre = if let Some(ref filter_genre) = genre {
                content.metadata.genre == *filter_genre
            } else {
                true
            };

            if matches_query && matches_genre {
                results.push(content.clone());
            }
        }

        // Search across network sources
        let network_results = self.discovery.search(query, genre).await?;
        results.extend(network_results);

        Ok(results)
    }

    /// Play audio content
    pub async fn play(&mut self, content_id: Uuid) -> Result<()> {
        if let Some(content) = self.content_cache.get(&content_id) {
            if content.is_cached && content.local_path.is_some() {
                // Play from local cache
                self.player.play_local(content.local_path.as_ref().unwrap()).await?;
            } else if !content.source_urls.is_empty() {
                // Stream from network
                let session_id = self.streamer.start_streaming(&content.source_urls[0]).await?;
                let session = StreamingSession {
                    id: session_id,
                    content_id,
                    source_url: content.source_urls[0].clone(),
                    started_at: chrono::Utc::now().timestamp(),
                    quality: self.config.preferred_quality.clone(),
                    state: streaming::StreamingState::Initializing,
                    bytes_downloaded: 0,
                    total_bytes: None,
                    download_speed: 0.0,
                    buffer_health: 0.0,
                };
                self.active_sessions.insert(session_id, session);
                self.player.play_stream(session_id).await?;
            } else {
                return Err(anyhow::anyhow!("No source available for content"));
            }
        } else {
            return Err(anyhow::anyhow!("Content not found"));
        }

        Ok(())
    }

    /// Pause playback
    pub async fn pause(&mut self) -> Result<()> {
        self.player.pause().await
    }

    /// Resume playback
    pub async fn resume(&mut self) -> Result<()> {
        self.player.resume().await
    }

    /// Stop playback
    pub async fn stop_playback(&mut self) -> Result<()> {
        self.player.stop().await
    }

    /// Download content for offline use
    pub async fn download(&mut self, content_id: Uuid) -> Result<()> {
        if let Some(content) = self.content_cache.get_mut(&content_id) {
            if content.is_cached {
                return Ok(()); // Already downloaded
            }

            if !content.source_urls.is_empty() {
                let url = &content.source_urls[0];
                let filename = format!("{}.audio", content.id);
                let local_path = Path::new(&self.config.cache_directory).join(filename);

                // Download the file
                let response = reqwest::get(url.clone()).await?;
                let content_bytes = response.bytes().await?;
                
                tokio::fs::write(&local_path, content_bytes).await?;

                // Update content info
                content.local_path = Some(local_path.to_string_lossy().to_string());
                content.is_cached = true;
                content.download_progress = 100.0;

                tracing::info!("Downloaded content {} to cache", content_id);
            }
        }

        Ok(())
    }

    /// Get random content recommendation
    pub async fn get_random_content(&self, genre: Option<AudioGenre>) -> Option<AudioContent> {
        let filtered_content: Vec<_> = if let Some(filter_genre) = genre {
            self.content_cache.values()
                .filter(|content| content.metadata.genre == filter_genre)
                .collect()
        } else {
            self.content_cache.values().collect()
        };

        if filtered_content.is_empty() {
            return None;
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        filtered_content.choose(&mut rng).map(|&content| content.clone())
    }

    /// List available genres
    pub fn get_available_genres(&self) -> Vec<AudioGenre> {
        let mut genres: Vec<_> = self.content_cache.values()
            .map(|content| content.metadata.genre.clone())
            .collect();
        genres.sort();
        genres.dedup();
        genres
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> Result<CacheStats> {
        let cache_dir = Path::new(&self.config.cache_directory);
        let mut total_size = 0u64;
        let mut file_count = 0usize;

        if cache_dir.exists() {
            let mut entries = tokio::fs::read_dir(cache_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                if let Ok(metadata) = entry.metadata().await {
                    total_size += metadata.len();
                    file_count += 1;
                }
            }
        }

        Ok(CacheStats {
            total_size_bytes: total_size,
            file_count,
            cached_content_count: self.content_cache.values()
                .filter(|c| c.is_cached)
                .count(),
            max_size_bytes: self.config.max_cache_size_gb * 1024 * 1024 * 1024,
        })
    }

    /// Refresh content from network sources
    async fn refresh_content_sources(&mut self) -> Result<()> {
        tracing::info!("Refreshing content from network sources");

        for source_url in &self.config.network_sources {
            if let Ok(url) = Url::parse(source_url) {
                if let Ok(augy_content) = self.discovery.fetch_augy_from_url(&url).await {
                    for entry in augy_content.entries {
                        let content = AudioContent {
                            id: Uuid::new_v4(),
                            metadata: entry.metadata,
                            source_urls: entry.uri_points,
                            local_path: None,
                            download_progress: 0.0,
                            is_cached: false,
                        };

                        self.content_cache.insert(content.id, content);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_size_bytes: u64,
    pub file_count: usize,
    pub cached_content_count: usize,
    pub max_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audigy_engine_creation() {
        let config = AudiogyConfig::default();
        let engine = AudiogyEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[test]
    fn test_audio_quality_ordering() {
        assert_eq!(AudioQuality::Low, AudioQuality::Low);
        assert_ne!(AudioQuality::Low, AudioQuality::High);
    }

    #[test]
    fn test_default_config() {
        let config = AudiogyConfig::default();
        assert_eq!(config.max_cache_size_gb, 5);
        assert_eq!(config.preferred_quality, streaming::StreamingQuality::High);
        assert!(!config.auto_download);
        assert!(config.enable_discovery);
    }
}