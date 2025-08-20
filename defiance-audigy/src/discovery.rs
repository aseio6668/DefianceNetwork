//! Network discovery for Audigy content

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;
use anyhow::Result;
use crate::augy::{AugyFile, AudioGenre};
use crate::AudioContent;

/// Network discovery manager for finding audio content across networks
pub struct NetworkDiscovery {
    sources: Vec<ContentSource>,
    cache: HashMap<String, CachedDiscoveryResult>,
    config: DiscoveryConfig,
}

/// Content source definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSource {
    pub id: Uuid,
    pub name: String,
    pub url: Url,
    pub source_type: SourceType,
    pub priority: u8, // 0-255, higher = more priority
    pub last_updated: Option<i64>,
    pub is_available: bool,
    pub content_count: usize,
}

/// Types of content sources
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    /// Direct .augy file URL
    AugyFile,
    /// API endpoint providing .augy format
    ApiEndpoint,
    /// RSS/Podcast feed
    RssFeed,
    /// Decentralized network node
    P2PNode,
    /// IPFS hash/gateway
    IpfsGateway,
    /// Custom protocol
    Custom(String),
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub cache_duration_hours: u64,
    pub max_concurrent_requests: usize,
    pub request_timeout_seconds: u64,
    pub retry_attempts: u8,
    pub enable_p2p_discovery: bool,
    pub enable_ipfs_discovery: bool,
}

/// Cached discovery result
#[derive(Debug, Clone)]
struct CachedDiscoveryResult {
    content: Vec<AudioContent>,
    cached_at: i64,
    expires_at: i64,
}

/// Discovery search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub content: Vec<AudioContent>,
    pub sources: Vec<String>, // Source names that provided results
    pub total_found: usize,
    pub search_time_ms: u64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            cache_duration_hours: 6,
            max_concurrent_requests: 10,
            request_timeout_seconds: 30,
            retry_attempts: 3,
            enable_p2p_discovery: true,
            enable_ipfs_discovery: true,
        }
    }
}

impl NetworkDiscovery {
    /// Create new network discovery instance
    pub async fn new(source_urls: Vec<String>) -> Result<Self> {
        let mut sources = Vec::new();
        
        // Convert URL strings to ContentSource objects
        for url_str in source_urls {
            if let Ok(url) = Url::parse(&url_str) {
                let source = ContentSource {
                    id: Uuid::new_v4(),
                    name: url.host_str().unwrap_or("Unknown").to_string(),
                    url,
                    source_type: SourceType::AugyFile, // Default, could be detected
                    priority: 100,
                    last_updated: None,
                    is_available: true,
                    content_count: 0,
                };
                sources.push(source);
            }
        }
        
        Ok(Self {
            sources,
            cache: HashMap::new(),
            config: DiscoveryConfig::default(),
        })
    }
    
    /// Start discovery service
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting network discovery with {} sources", self.sources.len());
        
        // Test connectivity to all sources
        self.test_source_connectivity().await;
        
        Ok(())
    }
    
    /// Stop discovery service
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping network discovery");
        
        // Clear cache
        self.cache.clear();
        
        Ok(())
    }
    
    /// Add a new content source
    pub fn add_source(&mut self, url: Url, source_type: SourceType, priority: u8) -> Uuid {
        let source = ContentSource {
            id: Uuid::new_v4(),
            name: url.host_str().unwrap_or("Unknown").to_string(),
            url,
            source_type,
            priority,
            last_updated: None,
            is_available: true,
            content_count: 0,
        };
        
        let id = source.id;
        self.sources.push(source);
        
        tracing::info!("Added content source: {} ({})", id, self.sources.last().unwrap().name);
        id
    }
    
    /// Remove a content source
    pub fn remove_source(&mut self, source_id: Uuid) -> bool {
        let initial_len = self.sources.len();
        self.sources.retain(|source| source.id != source_id);
        
        if self.sources.len() < initial_len {
            tracing::info!("Removed content source: {}", source_id);
            true
        } else {
            false
        }
    }
    
    /// Search for audio content across all sources
    pub async fn search(
        &self,
        query: &str,
        genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        let start_time = std::time::Instant::now();
        let mut all_results = Vec::new();
        let mut search_sources = Vec::new();
        
        // Search in available sources
        for source in &self.sources {
            if !source.is_available {
                continue;
            }
            
            match self.search_source(source, query, genre.clone()).await {
                Ok(mut results) => {
                    search_sources.push(source.name.clone());
                    all_results.append(&mut results);
                }
                Err(e) => {
                    tracing::warn!("Failed to search source {}: {}", source.name, e);
                }
            }
        }
        
        // Remove duplicates based on content ID or URL
        all_results.sort_by_key(|content| content.id);
        all_results.dedup_by_key(|content| content.id);
        
        let search_time = start_time.elapsed().as_millis() as u64;
        tracing::info!(
            "Search '{}' found {} results from {} sources in {}ms",
            query,
            all_results.len(),
            search_sources.len(),
            search_time
        );
        
        Ok(all_results)
    }
    
    /// Fetch .augy file from URL
    pub async fn fetch_augy_from_url(&self, url: &Url) -> Result<AugyFile> {
        tracing::debug!("Fetching .augy file from: {}", url);
        
        let client = reqwest::Client::new();
        let response = client
            .get(url.clone())
            .timeout(std::time::Duration::from_secs(self.config.request_timeout_seconds))
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to fetch .augy file: HTTP {}",
                response.status()
            ));
        }
        
        let content = response.text().await?;
        let augy_file = AugyFile::parse_from_string(&content)?;
        
        Ok(augy_file)
    }
    
    /// Discover content from a specific source
    async fn search_source(
        &self,
        source: &ContentSource,
        query: &str,
        genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        // Check cache first
        let cache_key = format!("{}:{}:{:?}", source.id, query, genre);
        if let Some(cached) = self.cache.get(&cache_key) {
            let now = chrono::Utc::now().timestamp();
            if now < cached.expires_at {
                tracing::debug!("Using cached results for source: {}", source.name);
                return Ok(cached.content.clone());
            }
        }
        
        match source.source_type {
            SourceType::AugyFile | SourceType::ApiEndpoint => {
                self.search_augy_source(source, query, genre).await
            }
            SourceType::RssFeed => {
                self.search_rss_source(source, query, genre).await
            }
            SourceType::P2PNode => {
                if self.config.enable_p2p_discovery {
                    self.search_p2p_source(source, query, genre).await
                } else {
                    Ok(Vec::new())
                }
            }
            SourceType::IpfsGateway => {
                if self.config.enable_ipfs_discovery {
                    self.search_ipfs_source(source, query, genre).await
                } else {
                    Ok(Vec::new())
                }
            }
            SourceType::Custom(_) => {
                // TODO: Implement custom protocol handlers
                Ok(Vec::new())
            }
        }
    }
    
    /// Search .augy file source
    async fn search_augy_source(
        &self,
        source: &ContentSource,
        query: &str,
        genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        let augy_file = self.fetch_augy_from_url(&source.url).await?;
        
        let matching_entries = augy_file.search(query);
        let mut results = Vec::new();
        
        for entry in matching_entries {
            // Filter by genre if specified
            if let Some(ref filter_genre) = genre {
                if entry.metadata.genre != *filter_genre {
                    continue;
                }
            }
            
            // Check if content is currently available
            if !entry.is_available_now() {
                continue;
            }
            
            let content = AudioContent {
                id: entry.id,
                metadata: entry.metadata.clone(),
                source_urls: entry.uri_points.clone(),
                local_path: None,
                download_progress: 0.0,
                is_cached: false,
            };
            
            results.push(content);
        }
        
        Ok(results)
    }
    
    /// Search RSS feed source
    async fn search_rss_source(
        &self,
        _source: &ContentSource,
        _query: &str,
        _genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        // TODO: Implement RSS feed parsing and search
        Ok(Vec::new())
    }
    
    /// Search P2P network source
    async fn search_p2p_source(
        &self,
        _source: &ContentSource,
        _query: &str,
        _genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        // TODO: Implement P2P network search
        Ok(Vec::new())
    }
    
    /// Search IPFS source
    async fn search_ipfs_source(
        &self,
        _source: &ContentSource,
        _query: &str,
        _genre: Option<AudioGenre>,
    ) -> Result<Vec<AudioContent>> {
        // TODO: Implement IPFS content search
        Ok(Vec::new())
    }
    
    /// Test connectivity to all sources
    async fn test_source_connectivity(&mut self) {
        // Collect URLs first to avoid borrowing conflicts
        let source_tests: Vec<_> = self.sources.iter()
            .enumerate()
            .map(|(i, source)| (i, source.url.clone(), source.name.clone()))
            .collect();
        
        for (index, url, name) in source_tests {
            let is_available = self.test_source(&url).await;
            self.sources[index].is_available = is_available;
            if !is_available {
                tracing::warn!("Source {} is not available", name);
            }
        }
    }
    
    /// Test if a source URL is accessible
    async fn test_source(&self, url: &Url) -> bool {
        let client = reqwest::Client::new();
        
        match client
            .head(url.clone())
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
    
    /// Get list of available sources
    pub fn get_sources(&self) -> &[ContentSource] {
        &self.sources
    }
    
    /// Get discovery statistics
    pub fn get_stats(&self) -> DiscoveryStats {
        let total_sources = self.sources.len();
        let available_sources = self.sources.iter().filter(|s| s.is_available).count();
        let total_content = self.sources.iter().map(|s| s.content_count).sum();
        
        DiscoveryStats {
            total_sources,
            available_sources,
            total_content,
            cache_entries: self.cache.len(),
        }
    }
}

/// Discovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    pub total_sources: usize,
    pub available_sources: usize,
    pub total_content: usize,
    pub cache_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_network_discovery_creation() {
        let sources = vec![
            "https://example.com/catalog.augy".to_string(),
            "https://podcasts.example.org/feed.augy".to_string(),
        ];
        
        let discovery = NetworkDiscovery::new(sources).await;
        assert!(discovery.is_ok());
        
        let discovery = discovery.unwrap();
        assert_eq!(discovery.sources.len(), 2);
    }
    
    #[tokio::test]
    async fn test_source_management() {
        let mut discovery = NetworkDiscovery::new(vec![]).await.unwrap();
        
        let url = Url::parse("https://example.com/test.augy").unwrap();
        let source_id = discovery.add_source(url, SourceType::AugyFile, 100);
        
        assert_eq!(discovery.sources.len(), 1);
        
        let removed = discovery.remove_source(source_id);
        assert!(removed);
        assert_eq!(discovery.sources.len(), 0);
    }
    
    #[test]
    fn test_source_type_equality() {
        assert_eq!(SourceType::AugyFile, SourceType::AugyFile);
        assert_ne!(SourceType::AugyFile, SourceType::RssFeed);
        assert_eq!(SourceType::Custom("test".to_string()), SourceType::Custom("test".to_string()));
    }
}