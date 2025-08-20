//! Broadcast proposal and viewer opt-in system for DefianceNetwork

use std::collections::{HashMap, HashSet};
use std::time::Duration;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use anyhow::Result;
use crate::crypto::CryptoManager;

/// Broadcast proposal and management system
pub struct BroadcastManager {
    node_id: Uuid,
    active_broadcasts: Arc<RwLock<HashMap<Uuid, BroadcastSession>>>,
    pending_proposals: Arc<RwLock<HashMap<Uuid, BroadcastProposal>>>,
    viewer_subscriptions: Arc<RwLock<HashMap<Uuid, ViewerSubscription>>>,
    reputation_system: Arc<RwLock<ReputationTracker>>,
    crypto_manager: CryptoManager,
    event_sender: mpsc::UnboundedSender<BroadcastEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<BroadcastEvent>>,
}

/// Broadcast proposal submitted by content creators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastProposal {
    pub proposal_id: Uuid,
    pub broadcaster_id: Uuid,
    pub title: String,
    pub description: String,
    pub content_type: ContentType,
    pub estimated_duration: Duration,
    pub quality_levels: Vec<QualityLevel>,
    pub required_bandwidth: u64,
    pub max_viewers: Option<u32>,
    pub reward_per_relay: Option<f64>, // Paradigm tokens per relay peer
    pub created_at: i64,
    pub start_time: Option<i64>,
    pub tags: Vec<String>,
    pub language: Option<String>,
    pub age_rating: AgeRating,
    pub monetization: Option<MonetizationModel>,
}

/// Active broadcast session
#[derive(Debug, Clone)]
pub struct BroadcastSession {
    pub broadcast_id: Uuid,
    pub proposal: BroadcastProposal,
    pub broadcaster: PeerId,
    pub relay_peers: HashSet<PeerId>,
    pub viewers: HashSet<PeerId>,
    pub quality_metrics: QualityMetrics,
    pub start_time: i64,
    pub last_activity: i64,
    pub status: BroadcastStatus,
    pub encryption_key: Option<[u8; 32]>,
    pub content_chunks: HashMap<u64, ContentChunk>,
}

/// Viewer subscription and participation
#[derive(Debug, Clone)]
pub struct ViewerSubscription {
    pub viewer_id: Uuid,
    pub peer_id: PeerId,
    pub subscribed_broadcasts: HashSet<Uuid>,
    pub preferred_quality: QualityLevel,
    pub available_bandwidth: u64,
    pub willing_to_relay: bool,
    pub max_relay_streams: u32,
    pub reputation_score: f32,
    pub region: Option<String>,
    pub device_capabilities: DeviceCapabilities,
}

/// Content type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    LiveVideo,
    LiveAudio,
    Podcast,
    Gaming,
    Educational,
    Entertainment,
    News,
    Sports,
    Music,
    Custom(String),
}

/// Video quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QualityLevel {
    Audio, // Audio-only stream
    Low,   // 480p
    Medium, // 720p
    High,   // 1080p
    Ultra,  // 4K
}

/// Age rating system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgeRating {
    AllAges,
    Teen,
    Mature,
    Adult,
}

/// Monetization models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonetizationModel {
    Free,
    PayPerView { price: f64 },
    Subscription { monthly_price: f64 },
    Donation,
    TokenReward { reward_per_minute: f64 },
}

/// Broadcast status
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastStatus {
    Proposed,
    Approved,
    Active,
    Paused,
    Ended,
    Failed,
}

/// Quality metrics for broadcasts
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub average_bitrate: u64,
    pub frame_drops: u64,
    pub latency_ms: u64,
    pub buffer_health: f32, // 0.0 to 1.0
    pub viewer_satisfaction: f32, // Based on feedback
    pub relay_efficiency: f32, // Network distribution efficiency
}

/// Content chunk for streaming
#[derive(Debug, Clone)]
pub struct ContentChunk {
    pub chunk_id: u64,
    pub data: Vec<u8>,
    pub quality: QualityLevel,
    pub timestamp: i64,
    pub checksum: [u8; 32],
    pub encryption_nonce: Option<[u8; 12]>,
}

/// Device capabilities for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub max_resolution: (u32, u32),
    pub supported_codecs: Vec<String>,
    pub hardware_acceleration: bool,
    pub battery_level: Option<f32>,
    pub connection_type: ConnectionType,
}

/// Connection type for bandwidth optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Ethernet,
    WiFi,
    Cellular4G,
    Cellular5G,
    Satellite,
    Unknown,
}

/// Reputation tracking for quality assurance
#[derive(Debug, Default)]
pub struct ReputationTracker {
    broadcaster_reputation: HashMap<Uuid, BroadcasterReputation>,
    viewer_reputation: HashMap<Uuid, ViewerReputation>,
}

#[derive(Debug, Clone)]
pub struct BroadcasterReputation {
    pub broadcaster_id: Uuid,
    pub total_broadcasts: u32,
    pub successful_broadcasts: u32,
    pub average_quality: f32,
    pub viewer_ratings: Vec<f32>,
    pub reliability_score: f32,
    pub last_broadcast: i64,
}

#[derive(Debug, Clone)]
pub struct ViewerReputation {
    pub viewer_id: Uuid,
    pub relay_uptime: f32,
    pub bandwidth_shared: u64,
    pub successful_relays: u32,
    pub reliability_score: f32,
    pub last_active: i64,
}

/// Broadcast events
#[derive(Debug, Clone)]
pub enum BroadcastEvent {
    ProposalSubmitted { proposal: BroadcastProposal },
    ProposalApproved { proposal_id: Uuid },
    ProposalRejected { proposal_id: Uuid, reason: String },
    BroadcastStarted { broadcast_id: Uuid },
    BroadcastEnded { broadcast_id: Uuid },
    ViewerJoined { broadcast_id: Uuid, viewer_id: Uuid },
    ViewerLeft { broadcast_id: Uuid, viewer_id: Uuid },
    RelayPeerAdded { broadcast_id: Uuid, peer_id: PeerId },
    RelayPeerRemoved { broadcast_id: Uuid, peer_id: PeerId },
    QualityAlert { broadcast_id: Uuid, issue: QualityIssue },
    ReputationUpdated { user_id: Uuid, new_score: f32 },
}

/// Quality issues that can occur during broadcasts
#[derive(Debug, Clone)]
pub enum QualityIssue {
    HighLatency(u64),
    LowBandwidth(u64),
    FrameDrops(u64),
    BufferUnderrun,
    RelayPeerFailure(PeerId),
    EncodingError,
}

impl BroadcastManager {
    /// Create new broadcast manager
    pub fn new(node_id: Uuid, crypto_manager: CryptoManager) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Self {
            node_id,
            active_broadcasts: Arc::new(RwLock::new(HashMap::new())),
            pending_proposals: Arc::new(RwLock::new(HashMap::new())),
            viewer_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            reputation_system: Arc::new(RwLock::new(ReputationTracker::default())),
            crypto_manager,
            event_sender,
            event_receiver: Some(event_receiver),
        }
    }
    
    /// Submit a new broadcast proposal
    pub async fn submit_proposal(&self, mut proposal: BroadcastProposal) -> Result<Uuid> {
        proposal.proposal_id = Uuid::new_v4();
        proposal.broadcaster_id = self.node_id;
        proposal.created_at = chrono::Utc::now().timestamp();
        
        tracing::info!(
            "Submitting broadcast proposal: {} - '{}'", 
            proposal.proposal_id, 
            proposal.title
        );
        
        // Validate proposal
        self.validate_proposal(&proposal).await?;
        
        // Store proposal
        {
            let mut proposals = self.pending_proposals.write().await;
            proposals.insert(proposal.proposal_id, proposal.clone());
        }
        
        let proposal_id = proposal.proposal_id;
        
        // Emit event
        let _ = self.event_sender.send(BroadcastEvent::ProposalSubmitted { proposal });
        
        Ok(proposal_id)
    }
    
    /// Validate a broadcast proposal
    async fn validate_proposal(&self, proposal: &BroadcastProposal) -> Result<()> {
        // Check title length
        if proposal.title.is_empty() || proposal.title.len() > 200 {
            return Err(anyhow::anyhow!("Invalid title length"));
        }
        
        // Check description length
        if proposal.description.len() > 2000 {
            return Err(anyhow::anyhow!("Description too long"));
        }
        
        // Validate bandwidth requirements
        if proposal.required_bandwidth == 0 || proposal.required_bandwidth > 100_000_000 { // 100 Mbps max
            return Err(anyhow::anyhow!("Invalid bandwidth requirement"));
        }
        
        // Check broadcaster reputation
        let reputation = self.reputation_system.read().await;
        if let Some(rep) = reputation.broadcaster_reputation.get(&proposal.broadcaster_id) {
            if rep.reliability_score < 0.3 {
                return Err(anyhow::anyhow!("Insufficient broadcaster reputation"));
            }
        }
        
        Ok(())
    }
    
    /// Approve a broadcast proposal
    pub async fn approve_proposal(&self, proposal_id: Uuid) -> Result<()> {
        let proposal = {
            let mut proposals = self.pending_proposals.write().await;
            proposals.remove(&proposal_id)
                .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?
        };
        
        tracing::info!("Approved broadcast proposal: {}", proposal_id);
        
        let _ = self.event_sender.send(BroadcastEvent::ProposalApproved { proposal_id });
        
        Ok(())
    }
    
    /// Start a broadcast session
    pub async fn start_broadcast(&self, proposal_id: Uuid, broadcaster: PeerId) -> Result<Uuid> {
        let proposal = {
            let proposals = self.pending_proposals.read().await;
            proposals.get(&proposal_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Proposal not found"))?
        };
        
        let broadcast_id = Uuid::new_v4();
        
        // Generate encryption key if needed
        let encryption_key = if matches!(proposal.content_type, ContentType::LiveVideo | ContentType::Gaming) {
            Some(CryptoManager::generate_random_bytes(32).try_into().unwrap())
        } else {
            None
        };
        
        let session = BroadcastSession {
            broadcast_id,
            proposal,
            broadcaster,
            relay_peers: HashSet::new(),
            viewers: HashSet::new(),
            quality_metrics: QualityMetrics::default(),
            start_time: chrono::Utc::now().timestamp(),
            last_activity: chrono::Utc::now().timestamp(),
            status: BroadcastStatus::Active,
            encryption_key,
            content_chunks: HashMap::new(),
        };
        
        // Store active broadcast
        {
            let mut broadcasts = self.active_broadcasts.write().await;
            broadcasts.insert(broadcast_id, session);
        }
        
        tracing::info!("Started broadcast session: {}", broadcast_id);
        
        let _ = self.event_sender.send(BroadcastEvent::BroadcastStarted { broadcast_id });
        
        Ok(broadcast_id)
    }
    
    /// Register viewer subscription
    pub async fn register_viewer(&self, subscription: ViewerSubscription) -> Result<()> {
        let viewer_id = subscription.viewer_id;
        
        {
            let mut subscriptions = self.viewer_subscriptions.write().await;
            subscriptions.insert(viewer_id, subscription);
        }
        
        tracing::info!("Registered viewer: {}", viewer_id);
        Ok(())
    }
    
    /// Join a broadcast as a viewer
    pub async fn join_broadcast(&self, viewer_id: Uuid, broadcast_id: Uuid) -> Result<QualityLevel> {
        // Check if broadcast exists and is active
        let optimal_quality = {
            let mut broadcasts = self.active_broadcasts.write().await;
            let broadcast = broadcasts.get_mut(&broadcast_id)
                .ok_or_else(|| anyhow::anyhow!("Broadcast not found"))?;
                
            if broadcast.status != BroadcastStatus::Active {
                return Err(anyhow::anyhow!("Broadcast is not active"));
            }
            
            // Add viewer
            let peer_id = self.get_peer_id_for_viewer(viewer_id).await?;
            broadcast.viewers.insert(peer_id);
            broadcast.last_activity = chrono::Utc::now().timestamp();
            
            // Determine optimal quality based on viewer capabilities
            self.determine_optimal_quality(viewer_id, &broadcast.proposal.quality_levels).await?
        };
        
        tracing::info!("Viewer {} joined broadcast {}", viewer_id, broadcast_id);
        
        let _ = self.event_sender.send(BroadcastEvent::ViewerJoined { broadcast_id, viewer_id });
        
        Ok(optimal_quality)
    }
    
    /// Leave a broadcast
    pub async fn leave_broadcast(&self, viewer_id: Uuid, broadcast_id: Uuid) -> Result<()> {
        let peer_id = self.get_peer_id_for_viewer(viewer_id).await?;
        
        {
            let mut broadcasts = self.active_broadcasts.write().await;
            if let Some(broadcast) = broadcasts.get_mut(&broadcast_id) {
                broadcast.viewers.remove(&peer_id);
                broadcast.relay_peers.remove(&peer_id);
                broadcast.last_activity = chrono::Utc::now().timestamp();
            }
        }
        
        let _ = self.event_sender.send(BroadcastEvent::ViewerLeft { broadcast_id, viewer_id });
        
        Ok(())
    }
    
    /// Opt-in to relay content for a broadcast
    pub async fn opt_in_as_relay(&self, viewer_id: Uuid, broadcast_id: Uuid) -> Result<()> {
        // Check viewer capability and reputation
        let subscription = {
            let subscriptions = self.viewer_subscriptions.read().await;
            subscriptions.get(&viewer_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Viewer not registered"))?
        };
        
        if !subscription.willing_to_relay {
            return Err(anyhow::anyhow!("Viewer not willing to relay"));
        }
        
        // Check reputation threshold
        if subscription.reputation_score < 0.5 {
            return Err(anyhow::anyhow!("Insufficient reputation for relay"));
        }
        
        let peer_id = subscription.peer_id;
        
        // Add as relay peer
        {
            let mut broadcasts = self.active_broadcasts.write().await;
            if let Some(broadcast) = broadcasts.get_mut(&broadcast_id) {
                broadcast.relay_peers.insert(peer_id);
                broadcast.last_activity = chrono::Utc::now().timestamp();
            }
        }
        
        tracing::info!("Viewer {} opted in as relay for broadcast {}", viewer_id, broadcast_id);
        
        let _ = self.event_sender.send(BroadcastEvent::RelayPeerAdded { broadcast_id, peer_id });
        
        Ok(())
    }
    
    /// End a broadcast session
    pub async fn end_broadcast(&self, broadcast_id: Uuid) -> Result<()> {
        let session = {
            let mut broadcasts = self.active_broadcasts.write().await;
            broadcasts.remove(&broadcast_id)
                .ok_or_else(|| anyhow::anyhow!("Broadcast not found"))?
        };
        
        // Update broadcaster reputation
        self.update_broadcaster_reputation(&session).await;
        
        // Reward relay peers
        self.reward_relay_peers(&session).await;
        
        tracing::info!("Ended broadcast session: {}", broadcast_id);
        
        let _ = self.event_sender.send(BroadcastEvent::BroadcastEnded { broadcast_id });
        
        Ok(())
    }
    
    /// Get active broadcasts matching viewer preferences
    pub async fn get_available_broadcasts(&self, viewer_id: Uuid) -> Result<Vec<BroadcastProposal>> {
        let subscription = {
            let subscriptions = self.viewer_subscriptions.read().await;
            subscriptions.get(&viewer_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Viewer not registered"))?
        };
        
        let broadcasts = self.active_broadcasts.read().await;
        let mut available = Vec::new();
        
        for session in broadcasts.values() {
            if session.status == BroadcastStatus::Active {
                // Filter by quality capabilities
                let has_compatible_quality = session.proposal.quality_levels.iter()
                    .any(|q| self.is_quality_compatible(q, &subscription.device_capabilities));
                
                if has_compatible_quality {
                    available.push(session.proposal.clone());
                }
            }
        }
        
        // Sort by relevance (simplified)
        available.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(available)
    }
    
    /// Update quality metrics for a broadcast
    pub async fn update_quality_metrics(&self, broadcast_id: Uuid, metrics: QualityMetrics) -> Result<()> {
        let mut broadcasts = self.active_broadcasts.write().await;
        if let Some(broadcast) = broadcasts.get_mut(&broadcast_id) {
            broadcast.quality_metrics = metrics.clone();
            broadcast.last_activity = chrono::Utc::now().timestamp();
            
            // Check for quality issues
            self.check_quality_issues(broadcast_id, &metrics).await;
        }
        
        Ok(())
    }
    
    /// Check for quality issues and emit alerts
    async fn check_quality_issues(&self, broadcast_id: Uuid, metrics: &QualityMetrics) {
        if metrics.latency_ms > 500 {
            let _ = self.event_sender.send(BroadcastEvent::QualityAlert {
                broadcast_id,
                issue: QualityIssue::HighLatency(metrics.latency_ms),
            });
        }
        
        if metrics.frame_drops > 100 {
            let _ = self.event_sender.send(BroadcastEvent::QualityAlert {
                broadcast_id,
                issue: QualityIssue::FrameDrops(metrics.frame_drops),
            });
        }
        
        if metrics.buffer_health < 0.2 {
            let _ = self.event_sender.send(BroadcastEvent::QualityAlert {
                broadcast_id,
                issue: QualityIssue::BufferUnderrun,
            });
        }
    }
    
    /// Determine optimal quality level for a viewer
    async fn determine_optimal_quality(&self, viewer_id: Uuid, available_qualities: &[QualityLevel]) -> Result<QualityLevel> {
        let subscription = {
            let subscriptions = self.viewer_subscriptions.read().await;
            subscriptions.get(&viewer_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Viewer not registered"))?
        };
        
        // Find the best quality that fits the viewer's bandwidth and device capabilities
        let mut best_quality = QualityLevel::Audio;
        
        for quality in available_qualities {
            if self.is_quality_compatible(quality, &subscription.device_capabilities) {
                let required_bandwidth = self.get_quality_bandwidth_requirement(quality);
                if subscription.available_bandwidth >= required_bandwidth {
                    best_quality = quality.clone();
                }
            }
        }
        
        Ok(best_quality)
    }
    
    /// Check if quality level is compatible with device capabilities
    fn is_quality_compatible(&self, quality: &QualityLevel, capabilities: &DeviceCapabilities) -> bool {
        match quality {
            QualityLevel::Audio => true,
            QualityLevel::Low => capabilities.max_resolution.1 >= 480,
            QualityLevel::Medium => capabilities.max_resolution.1 >= 720,
            QualityLevel::High => capabilities.max_resolution.1 >= 1080,
            QualityLevel::Ultra => capabilities.max_resolution.1 >= 2160,
        }
    }
    
    /// Get bandwidth requirement for quality level
    fn get_quality_bandwidth_requirement(&self, quality: &QualityLevel) -> u64 {
        match quality {
            QualityLevel::Audio => 128_000,      // 128 Kbps
            QualityLevel::Low => 1_000_000,      // 1 Mbps
            QualityLevel::Medium => 3_000_000,   // 3 Mbps
            QualityLevel::High => 6_000_000,     // 6 Mbps
            QualityLevel::Ultra => 25_000_000,   // 25 Mbps
        }
    }
    
    /// Get peer ID for a viewer
    async fn get_peer_id_for_viewer(&self, viewer_id: Uuid) -> Result<PeerId> {
        let subscriptions = self.viewer_subscriptions.read().await;
        let subscription = subscriptions.get(&viewer_id)
            .ok_or_else(|| anyhow::anyhow!("Viewer not found"))?;
        Ok(subscription.peer_id)
    }
    
    /// Update broadcaster reputation after a broadcast
    async fn update_broadcaster_reputation(&self, session: &BroadcastSession) {
        let broadcaster_id = session.proposal.broadcaster_id;
        let duration = chrono::Utc::now().timestamp() - session.start_time;
        let was_successful = duration > 60 && session.quality_metrics.viewer_satisfaction > 0.5;
        
        let mut reputation = self.reputation_system.write().await;
        let rep = reputation.broadcaster_reputation
            .entry(broadcaster_id)
            .or_insert_with(|| BroadcasterReputation {
                broadcaster_id,
                total_broadcasts: 0,
                successful_broadcasts: 0,
                average_quality: 0.0,
                viewer_ratings: Vec::new(),
                reliability_score: 1.0,
                last_broadcast: 0,
            });
        
        rep.total_broadcasts += 1;
        if was_successful {
            rep.successful_broadcasts += 1;
        }
        
        rep.average_quality = (rep.average_quality + session.quality_metrics.viewer_satisfaction) / 2.0;
        rep.reliability_score = rep.successful_broadcasts as f32 / rep.total_broadcasts as f32;
        rep.last_broadcast = chrono::Utc::now().timestamp();
        
        let _ = self.event_sender.send(BroadcastEvent::ReputationUpdated {
            user_id: broadcaster_id,
            new_score: rep.reliability_score,
        });
    }
    
    /// Reward relay peers for participation
    async fn reward_relay_peers(&self, session: &BroadcastSession) {
        if let Some(reward_per_relay) = session.proposal.reward_per_relay {
            let broadcast_duration = chrono::Utc::now().timestamp() - session.start_time;
            let reward_amount = reward_per_relay * (broadcast_duration as f64 / 3600.0); // Per hour
            
            // TODO: Integrate with Paradigm payment system to distribute rewards
            tracing::info!(
                "Rewarding {} relay peers with {} tokens each for broadcast {}",
                session.relay_peers.len(),
                reward_amount,
                session.broadcast_id
            );
        }
    }
    
    /// Get broadcast statistics
    pub async fn get_broadcast_stats(&self, broadcast_id: Uuid) -> Result<BroadcastStats> {
        let broadcasts = self.active_broadcasts.read().await;
        let session = broadcasts.get(&broadcast_id)
            .ok_or_else(|| anyhow::anyhow!("Broadcast not found"))?;
        
        Ok(BroadcastStats {
            broadcast_id,
            title: session.proposal.title.clone(),
            viewer_count: session.viewers.len(),
            relay_count: session.relay_peers.len(),
            duration: chrono::Utc::now().timestamp() - session.start_time,
            quality_metrics: session.quality_metrics.clone(),
            status: session.status.clone(),
        })
    }
    
    /// Take the event receiver for processing broadcast events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<BroadcastEvent>> {
        self.event_receiver.take()
    }
}

/// Broadcast statistics
#[derive(Debug, Clone)]
pub struct BroadcastStats {
    pub broadcast_id: Uuid,
    pub title: String,
    pub viewer_count: usize,
    pub relay_count: usize,
    pub duration: i64,
    pub quality_metrics: QualityMetrics,
    pub status: BroadcastStatus,
}

impl Default for QualityLevel {
    fn default() -> Self {
        QualityLevel::Medium
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_resolution: (1920, 1080),
            supported_codecs: vec!["h264".to_string(), "vp9".to_string()],
            hardware_acceleration: false,
            battery_level: None,
            connection_type: ConnectionType::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::CryptoManager;
    
    #[tokio::test]
    async fn test_broadcast_manager_creation() {
        let crypto = CryptoManager::new().unwrap();
        let manager = BroadcastManager::new(Uuid::new_v4(), crypto);
        
        // Manager should be created successfully
        assert!(manager.active_broadcasts.read().await.is_empty());
    }
    
    #[tokio::test]
    async fn test_proposal_submission() {
        let crypto = CryptoManager::new().unwrap();
        let manager = BroadcastManager::new(Uuid::new_v4(), crypto);
        
        let proposal = BroadcastProposal {
            proposal_id: Uuid::new_v4(), // Will be overwritten
            broadcaster_id: Uuid::new_v4(), // Will be overwritten
            title: "Test Broadcast".to_string(),
            description: "A test broadcast".to_string(),
            content_type: ContentType::LiveVideo,
            estimated_duration: Duration::from_secs(3600),
            quality_levels: vec![QualityLevel::Medium, QualityLevel::High],
            required_bandwidth: 5_000_000,
            max_viewers: Some(100),
            reward_per_relay: Some(0.1),
            created_at: 0, // Will be overwritten
            start_time: None,
            tags: vec!["test".to_string()],
            language: Some("en".to_string()),
            age_rating: AgeRating::AllAges,
            monetization: Some(MonetizationModel::Free),
        };
        
        let proposal_id = manager.submit_proposal(proposal).await.unwrap();
        
        let proposals = manager.pending_proposals.read().await;
        assert!(proposals.contains_key(&proposal_id));
    }
    
    #[tokio::test]
    async fn test_viewer_registration() {
        let crypto = CryptoManager::new().unwrap();
        let manager = BroadcastManager::new(Uuid::new_v4(), crypto);
        
        let subscription = ViewerSubscription {
            viewer_id: Uuid::new_v4(),
            peer_id: libp2p::PeerId::random(),
            subscribed_broadcasts: HashSet::new(),
            preferred_quality: QualityLevel::High,
            available_bandwidth: 10_000_000,
            willing_to_relay: true,
            max_relay_streams: 3,
            reputation_score: 0.8,
            region: Some("us-west".to_string()),
            device_capabilities: DeviceCapabilities::default(),
        };
        
        let viewer_id = subscription.viewer_id;
        manager.register_viewer(subscription).await.unwrap();
        
        let subscriptions = manager.viewer_subscriptions.read().await;
        assert!(subscriptions.contains_key(&viewer_id));
    }
    
    #[test]
    fn test_quality_bandwidth_requirements() {
        let crypto = CryptoManager::new().unwrap();
        let manager = BroadcastManager::new(Uuid::new_v4(), crypto);
        
        assert_eq!(manager.get_quality_bandwidth_requirement(&QualityLevel::Audio), 128_000);
        assert_eq!(manager.get_quality_bandwidth_requirement(&QualityLevel::Low), 1_000_000);
        assert_eq!(manager.get_quality_bandwidth_requirement(&QualityLevel::Medium), 3_000_000);
        assert_eq!(manager.get_quality_bandwidth_requirement(&QualityLevel::High), 6_000_000);
        assert_eq!(manager.get_quality_bandwidth_requirement(&QualityLevel::Ultra), 25_000_000);
    }
    
    #[test]
    fn test_quality_compatibility() {
        let crypto = CryptoManager::new().unwrap();
        let manager = BroadcastManager::new(Uuid::new_v4(), crypto);
        
        let capabilities_720p = DeviceCapabilities {
            max_resolution: (1280, 720),
            ..Default::default()
        };
        
        assert!(manager.is_quality_compatible(&QualityLevel::Audio, &capabilities_720p));
        assert!(manager.is_quality_compatible(&QualityLevel::Low, &capabilities_720p));
        assert!(manager.is_quality_compatible(&QualityLevel::Medium, &capabilities_720p));
        assert!(!manager.is_quality_compatible(&QualityLevel::High, &capabilities_720p));
        assert!(!manager.is_quality_compatible(&QualityLevel::Ultra, &capabilities_720p));
    }
}