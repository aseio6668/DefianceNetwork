//! # Defiance Core
//! 
//! Core P2P streaming and network functionality for DefianceNetwork.
//! This crate provides the fundamental networking, streaming, and data structures
//! needed for the decentralized streaming platform.

use defiance_discovery::PeerDiscovery;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;

pub mod network;
pub mod streaming;
pub mod video;
pub mod storage;
pub mod user;
pub mod content;
pub mod crypto;
pub mod error;
pub mod monitoring;
pub mod broadcast;

// Re-export commonly used types
pub use error::DefianceError;
pub use network::{P2PNetwork, NetworkMessage, PeerInfo};
pub use streaming::{StreamingEngine, BroadcastSession, ViewingSession};
pub use video::{VideoEngine, VideoStream, VideoQuality, VideoResolution};
pub use user::{User, Username, UserManager};
pub use content::{Content, ContentMetadata, ContentType};
pub use crypto::{CryptoManager, EncryptedContent};
pub use monitoring::{NetworkMonitor, NetworkHealth, MonitoringEvent};
pub use broadcast::{BroadcastManager, BroadcastProposal, ViewerSubscription, BroadcastEvent};

/// Core constants for DefianceNetwork
pub const DEFIANCE_VERSION: &str = "0.1.0";
pub const DEFIANCE_PROTOCOL_VERSION: &str = "defiance/1.0.0";
pub const DEFAULT_PORT: u16 = 9080;
pub const MAX_PEERS: usize = 100;
pub const CHUNK_SIZE: usize = 1024 * 64; // 64KB chunks
pub const MAX_CONTENT_SIZE: u64 = 1024 * 1024 * 1024 * 5; // 5GB max content size

/// Node configuration for DefianceNetwork
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub node_id: Uuid,
    pub network_port: u16,
    pub data_dir: String,
    pub enable_streaming: bool,
    pub enable_broadcasting: bool,
    pub max_peers: usize,
    pub max_upload_bandwidth: Option<u64>, // bytes per second
    pub max_download_bandwidth: Option<u64>, // bytes per second
    pub enable_discovery_fallback: bool,
    pub discovery_github_repo: Option<String>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            network_port: DEFAULT_PORT,
            data_dir: "./defiance_data".to_string(),
            enable_streaming: true,
            enable_broadcasting: true,
            max_peers: MAX_PEERS,
            max_upload_bandwidth: None,
            max_download_bandwidth: None,
            enable_discovery_fallback: true,
            discovery_github_repo: Some("DefianceNetwork/seed-nodes".to_string()),
        }
    }
}

/// Core DefianceNetwork node
pub struct DefianceNode {
    pub config: NodeConfig,
    pub network: Arc<RwLock<network::P2PNetwork>>,
    pub streaming: Arc<RwLock<streaming::StreamingEngine>>,
    pub video: Arc<RwLock<video::VideoEngine>>,
    pub storage: Arc<RwLock<storage::DefianceStorage>>,
    pub user_manager: Arc<RwLock<user::UserManager>>,
    pub crypto: Arc<crypto::CryptoManager>,
    pub monitor: Arc<RwLock<monitoring::NetworkMonitor>>,
    pub broadcast_manager: Arc<RwLock<broadcast::BroadcastManager>>,
    pub active_broadcasts: Arc<RwLock<HashMap<Uuid, BroadcastSession>>>,
    pub peer_discovery: Arc<RwLock<PeerDiscovery>>,
    pub bridge: Option<Arc<defiance_bridge::BridgeManager>>,
}

impl DefianceNode {
    /// Create a new DefianceNetwork node
    pub async fn new(config: NodeConfig) -> Result<Self> {
        tracing::info!("Initializing DefianceNetwork node {}", config.node_id);

        // Ensure data directory exists
        tokio::fs::create_dir_all(&config.data_dir).await?;

        // Initialize storage
        let storage = Arc::new(RwLock::new(
            storage::DefianceStorage::new(&config.data_dir).await?
        ));

        // Initialize crypto manager
        let crypto = Arc::new(crypto::CryptoManager::new()?);

        // Initialize user manager
        let user_manager = Arc::new(RwLock::new(
            user::UserManager::new(storage.clone()).await?
        ));

        // Initialize network
        let network = Arc::new(RwLock::new(
            network::P2PNetwork::new(config.node_id, config.network_port).await?
        ));

        // Initialize streaming engine
        let streaming = Arc::new(RwLock::new(
            streaming::StreamingEngine::new(
                network.clone(),
                storage.clone(),
                crypto.clone()
            ).await?
        ));

        // Initialize video engine
        let video = Arc::new(RwLock::new(
            video::VideoEngine::new(streaming.clone(), video::VideoEngineConfig::default()).await?
        ));

        // Initialize network monitor
        let monitor = Arc::new(RwLock::new(
            monitoring::NetworkMonitor::new()
        ));

        // Initialize broadcast manager
        let broadcast_manager = Arc::new(RwLock::new(
            broadcast::BroadcastManager::new(config.node_id, (*crypto).clone())
        ));

        // Initialize peer discovery
        let peer_discovery = Arc::new(RwLock::new(
            PeerDiscovery::new().with_github_repo(config.discovery_github_repo.clone().unwrap_or_default())
        ));

        // Initialize bridge (optional)
        let bridge = if let Ok(bridge_manager) = defiance_bridge::BridgeManager::new().await {
            Some(Arc::new(bridge_manager))
        } else {
            tracing::warn!("Failed to initialize cryptocurrency bridge - continuing without bridge support");
            None
        };

        Ok(Self {
            config,
            network,
            streaming,
            video,
            storage,
            user_manager,
            crypto,
            monitor,
            broadcast_manager,
            active_broadcasts: Arc::new(RwLock::new(HashMap::new())),
            peer_discovery,
            bridge,
        })
    }

    /// Start the DefianceNetwork node
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting DefianceNetwork node");

        // Start network layer
        {
            let mut network = self.network.write().await;
            network.start().await?;
        }

        // Start streaming engine
        {
            let mut streaming = self.streaming.write().await;
            streaming.start().await?;
        }

        // Start video engine
        {
            let mut video = self.video.write().await;
            video.start().await?;
        }

        // Start user manager
        {
            let mut user_manager = self.user_manager.write().await;
            user_manager.start().await?;
        }

        // Start network monitor
        {
            let mut monitor = self.monitor.write().await;
            monitor.start().await?;
        }

        // Start peer discovery
        {
            let mut peer_discovery = self.peer_discovery.write().await;
            peer_discovery.start().await?;
        }

        tracing::info!("DefianceNetwork node started successfully");
        Ok(())
    }

    /// Stop the DefianceNetwork node
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping DefianceNetwork node");

        // Stop streaming engine
        {
            let mut streaming = self.streaming.write().await;
            streaming.stop().await?;
        }

        // Stop network layer
        {
            let mut network = self.network.write().await;
            network.stop().await?;
        }

        tracing::info!("DefianceNetwork node stopped");
        Ok(())
    }

    /// Get the current user
    pub async fn get_current_user(&self) -> Result<Option<User>> {
        let user_manager = self.user_manager.read().await;
        user_manager.get_current_user().await
            .map_err(|e| anyhow::anyhow!("Failed to get current user: {}", e))
    }

    /// Create a new broadcast proposal
    pub async fn submit_broadcast_proposal(&self, proposal: broadcast::BroadcastProposal) -> Result<Uuid> {
        let broadcast_manager = self.broadcast_manager.read().await;
        broadcast_manager.submit_proposal(proposal).await
    }

    /// Start a broadcast session from approved proposal
    pub async fn start_broadcast_from_proposal(&self, proposal_id: Uuid) -> Result<Uuid> {
        let network = self.network.read().await;
        let local_peer_id = libp2p::PeerId::random(); // TODO: Get actual peer ID from network
        
        let broadcast_manager = self.broadcast_manager.read().await;
        broadcast_manager.start_broadcast(proposal_id, local_peer_id).await
    }

    /// Create a new broadcast session (legacy)
    pub async fn start_broadcast(&self, title: String, description: String, content_type: ContentType) -> Result<Uuid> {
        let mut streaming = self.streaming.write().await;
        streaming.start_broadcast(title, description, content_type).await
            .map_err(|e| anyhow::anyhow!("Failed to start broadcast: {}", e))
    }

    /// Join a viewing session
    pub async fn join_viewing_session(&self, content_id: Uuid, viewer_peer_id: libp2p::PeerId) -> Result<Uuid> {
        let mut streaming = self.streaming.write().await;
        streaming.join_viewing_session(content_id, viewer_peer_id).await
            .map_err(|e| anyhow::anyhow!("Failed to join viewing session: {}", e))
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> Result<NetworkStats> {
        let network = self.network.read().await;
        let streaming = self.streaming.read().await;
        let monitor = self.monitor.read().await;
        let health = monitor.get_network_health().await;
        
        Ok(NetworkStats {
            connected_peers: network.get_peer_count(),
            active_broadcasts: streaming.get_active_broadcast_count(),
            active_viewers: streaming.get_active_viewer_count(),
            upload_bandwidth: health.total_bandwidth_out,
            download_bandwidth: health.total_bandwidth_in,
            latency_ms: health.average_latency.as_millis() as u64,
        })
    }

    /// Get broadcast manager
    pub fn get_broadcast_manager(&self) -> Arc<RwLock<broadcast::BroadcastManager>> {
        self.broadcast_manager.clone()
    }

    /// Get network monitor
    pub fn get_network_monitor(&self) -> Arc<RwLock<monitoring::NetworkMonitor>> {
        self.monitor.clone()
    }

    /// Get bridge manager if available
    pub fn get_bridge(&self) -> Option<Arc<defiance_bridge::BridgeManager>> {
        self.bridge.clone()
    }
}

/// Network statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub active_broadcasts: usize,
    pub active_viewers: usize,
    pub upload_bandwidth: u64, // bytes per second
    pub download_bandwidth: u64, // bytes per second
    pub latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_creation() {
        let config = NodeConfig::default();
        let node = DefianceNode::new(config).await;
        assert!(node.is_ok());
    }

    #[test]
    fn test_default_config() {
        let config = NodeConfig::default();
        assert_eq!(config.network_port, DEFAULT_PORT);
        assert_eq!(config.max_peers, MAX_PEERS);
        assert!(config.enable_streaming);
        assert!(config.enable_broadcasting);
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_PORT, 9080);
        assert_eq!(MAX_PEERS, 100);
        assert_eq!(CHUNK_SIZE, 1024 * 64);
    }
}