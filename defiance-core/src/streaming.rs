//! Streaming engine for broadcast and viewing sessions

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::error::{DefianceError, Result};
use crate::network::{P2PNetwork, NetworkMessage};
use crate::storage::DefianceStorage;
use crate::crypto::CryptoManager;
use crate::content::{Content, ContentType, ContentChunk, Quality};
use libp2p::PeerId;

/// Streaming engine managing broadcasts and viewing sessions
pub struct StreamingEngine {
    network: Arc<RwLock<P2PNetwork>>,
    storage: Arc<RwLock<DefianceStorage>>,
    crypto: Arc<CryptoManager>,
    active_broadcasts: HashMap<Uuid, BroadcastSession>,
    active_viewers: HashMap<Uuid, ViewingSession>,
}

/// Active broadcast session
#[derive(Debug, Clone)]
pub struct BroadcastSession {
    pub id: Uuid,
    pub content: Content,
    pub started_at: i64,
    pub viewers: HashMap<Uuid, PeerId>, // Map user ID to network PeerId
    pub max_viewers: Option<usize>,
    pub is_live: bool,
    pub quality_levels: Vec<Quality>,
    pub current_chunk: u64,
}

/// Active viewing session
#[derive(Debug, Clone)]
pub struct ViewingSession {
    pub id: Uuid,
    pub content_id: Uuid,
    pub user_id: Uuid,
    pub started_at: i64,
    pub current_position: u64, // current chunk or timestamp
    pub quality: Quality,
    pub is_buffering: bool,
    pub download_progress: f32,
}

/// Streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub active_broadcasts: usize,
    pub active_viewers: usize,
    pub total_bandwidth_usage: u64,
    pub average_latency: Option<u64>,
    pub chunk_cache_size: usize,
}

impl StreamingEngine {
    /// Create new streaming engine
    pub async fn new(
        network: Arc<RwLock<P2PNetwork>>,
        storage: Arc<RwLock<DefianceStorage>>,
        crypto: Arc<CryptoManager>,
    ) -> Result<Self> {
        Ok(Self {
            network,
            storage,
            crypto,
            active_broadcasts: HashMap::new(),
            active_viewers: HashMap::new(),
        })
    }
    
    /// Start the streaming engine
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting streaming engine");
        // TODO: Initialize streaming components
        Ok(())
    }
    
    /// Stop the streaming engine
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping streaming engine");
        
        // Stop all active broadcasts - collect first to avoid borrowing issues
        let broadcasts: Vec<_> = self.active_broadcasts.drain().map(|(_, b)| b).collect();
        for mut broadcast in broadcasts {
            self.stop_broadcast_internal(&mut broadcast).await?;
        }
        
        // Stop all viewing sessions - collect first to avoid borrowing issues
        let sessions: Vec<_> = self.active_viewers.drain().map(|(_, s)| s).collect();
        for mut session in sessions {
            self.stop_viewing_session_internal(&mut session).await?;
        }
        
        Ok(())
    }
    
    /// Start a new broadcast
    pub async fn start_broadcast(
        &mut self,
        title: String,
        description: String,
        content_type: ContentType,
    ) -> Result<Uuid> {
        let broadcast_id = Uuid::new_v4();
        let creator_id = Uuid::new_v4(); // TODO: Get from user manager
        
        let content = if content_type == ContentType::LiveStream {
            Content::new_live_stream(title.clone(), description.clone(), "Anonymous".to_string(), creator_id)
        } else {
            Content::new(title.clone(), description.clone(), content_type.clone(), "Anonymous".to_string(), creator_id)
        };
        
        let broadcast = BroadcastSession {
            id: broadcast_id,
            content,
            started_at: chrono::Utc::now().timestamp(),
            viewers: HashMap::new(),
            max_viewers: None,
            is_live: true,
            quality_levels: vec![Quality::Source, Quality::High, Quality::Medium],
            current_chunk: 0,
        };
        
        // Store broadcast in storage
        {
            let mut storage = self.storage.write().await;
            storage.store_broadcast(&broadcast).await?;
        }
        
        // Announce broadcast to network
        let announcement = NetworkMessage::BroadcastAnnouncement {
            broadcast_id,
            title,
            description,
            content_type: format!("{:?}", content_type),
            broadcaster: "Anonymous".to_string(),
        };
        
        {
            let mut network = self.network.write().await;
            network.broadcast(announcement).await?;
        }
        
        self.active_broadcasts.insert(broadcast_id, broadcast);
        
        tracing::info!("Started broadcast session: {}", broadcast_id);
        Ok(broadcast_id)
    }
    
    /// Stop a broadcast
    pub async fn stop_broadcast(&mut self, broadcast_id: Uuid) -> Result<()> {
        if let Some(mut broadcast) = self.active_broadcasts.remove(&broadcast_id) {
            self.stop_broadcast_internal(&mut broadcast).await?;
            tracing::info!("Stopped broadcast session: {}", broadcast_id);
        }
        Ok(())
    }
    
    /// Internal broadcast stopping logic
    async fn stop_broadcast_internal(&self, broadcast: &mut BroadcastSession) -> Result<()> {
        broadcast.is_live = false;
        
        // TODO: Notify viewers that broadcast ended
        // TODO: Save final broadcast metadata
        
        Ok(())
    }
    
    /// Join a viewing session
    pub async fn join_viewing_session(&mut self, content_id: Uuid, viewer_peer_id: PeerId) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4(); // TODO: Get from user manager
        
        let session = ViewingSession {
            id: session_id,
            content_id,
            user_id,
            started_at: chrono::Utc::now().timestamp(),
            current_position: 0,
            quality: Quality::High,
            is_buffering: true,
            download_progress: 0.0,
        };
        
        // Check if this is a live broadcast and add viewer
        if let Some(broadcast) = self.active_broadcasts.get_mut(&content_id) {
            broadcast.viewers.insert(user_id, viewer_peer_id);
            broadcast.content.metadata.increment_viewers();
        }
        
        // TODO: Start downloading/streaming content
        
        self.active_viewers.insert(session_id, session);
        
        tracing::info!("Started viewing session: {} for content: {}", session_id, content_id);
        Ok(session_id)
    }
    
    /// Leave a viewing session
    pub async fn leave_viewing_session(&mut self, session_id: Uuid) -> Result<()> {
        if let Some(mut session) = self.active_viewers.remove(&session_id) {
            self.stop_viewing_session_internal(&mut session).await?;
            tracing::info!("Stopped viewing session: {}", session_id);
        }
        Ok(())
    }
    
    /// Internal viewing session stopping logic
    async fn stop_viewing_session_internal(&mut self, session: &mut ViewingSession) -> Result<()> {
        // Remove viewer from broadcast if it's live
        if let Some(broadcast) = self.active_broadcasts.get_mut(&session.content_id) {
            broadcast.viewers.remove(&session.user_id);
        }
        
        Ok(())
    }
    
    /// Add content chunk to a broadcast and distribute it to viewers.
    pub async fn add_broadcast_chunk(
        &mut self,
        broadcast_id: Uuid,
        chunk: ContentChunk,
    ) -> Result<()> {
        if let Some(broadcast) = self.active_broadcasts.get_mut(&broadcast_id) {
            let chunk_id = broadcast.current_chunk;
            broadcast.content.add_chunk(chunk.clone())?;
            broadcast.current_chunk += 1;

            // Encrypt and prepare the chunk for sending
            let encrypted_data = (*self.crypto).encrypt(&chunk.data)?;
            let checksum = blake3::hash(&encrypted_data).into();

            let message = NetworkMessage::StreamChunk {
                content_id: broadcast.content.metadata.id,
                chunk_id,
                data: encrypted_data,
                checksum,
            };

            // Distribute chunk to all viewers
            let mut network = self.network.write().await;
            for peer_id in broadcast.viewers.values() {
                if let Err(e) = network.send_to_peer(*peer_id, message.clone()).await {
                    tracing::warn!("Failed to send chunk to peer {}: {}", peer_id, e);
                }
            }
            
            Ok(())
        } else {
            Err(DefianceError::Streaming("Broadcast not found".to_string()))
        }
    }

    /// Process an encoded video packet, wrap it in a ContentChunk, and distribute it.
    pub async fn process_video_packet(
        &mut self,
        broadcast_id: Uuid,
        packet_data: Vec<u8>,
        timestamp: Option<u64>,
    ) -> Result<()> {
        if let Some(broadcast) = self.active_broadcasts.get(&broadcast_id) {
            let checksum = blake3::hash(&packet_data).into();
            let chunk = ContentChunk {
                content_id: broadcast.content.metadata.id,
                chunk_id: broadcast.current_chunk,
                data: packet_data,
                checksum,
                quality: Quality::Source, // Assuming source quality for now
                timestamp,
            };
            
            self.add_broadcast_chunk(broadcast_id, chunk).await
        } else {
            Err(DefianceError::Streaming("Broadcast not found".to_string()))
        }
    }
    
    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        StreamingStats {
            active_broadcasts: self.active_broadcasts.len(),
            active_viewers: self.active_viewers.len(),
            total_bandwidth_usage: 0, // TODO: Calculate from network stats
            average_latency: None, // TODO: Get from network
            chunk_cache_size: 0, // TODO: Get from storage
        }
    }
    
    /// Get active broadcast count
    pub fn get_active_broadcast_count(&self) -> usize {
        self.active_broadcasts.len()
    }
    
    /// Get active viewer count
    pub fn get_active_viewer_count(&self) -> usize {
        self.active_viewers.len()
    }
    
    /// Get broadcast by ID
    pub fn get_broadcast(&self, broadcast_id: Uuid) -> Option<&BroadcastSession> {
        self.active_broadcasts.get(&broadcast_id)
    }
    
    /// Get viewing session by ID
    pub fn get_viewing_session(&self, session_id: Uuid) -> Option<&ViewingSession> {
        self.active_viewers.get(&session_id)
    }
    
    /// List all active broadcasts
    pub fn list_active_broadcasts(&self) -> Vec<&BroadcastSession> {
        self.active_broadcasts.values().collect()
    }
    
    /// Change viewing quality
    pub async fn change_viewing_quality(
        &mut self,
        session_id: Uuid,
        quality: Quality,
    ) -> Result<()> {
        if let Some(session) = self.active_viewers.get_mut(&session_id) {
            session.quality = quality.clone();
            session.is_buffering = true; // Need to rebuffer for new quality
            
            // TODO: Request new quality chunks from network
            
            tracing::info!("Changed viewing quality for session {} to {:?}", session_id, quality);
            Ok(())
        } else {
            Err(DefianceError::Streaming("Viewing session not found".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::DefianceStorage;
    use crate::crypto::CryptoManager;
    use crate::network::P2PNetwork;
    
    async fn create_test_engine() -> StreamingEngine {
        let storage = Arc::new(RwLock::new(
            DefianceStorage::new(":memory:").await.unwrap()
        ));
        let crypto = Arc::new(CryptoManager::new().unwrap());
        let network = Arc::new(RwLock::new(
            P2PNetwork::new(Uuid::new_v4(), 9999).await.unwrap()
        ));
        
        StreamingEngine::new(network, storage, crypto).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_broadcast_lifecycle() {
        let mut engine = create_test_engine().await;
        
        let broadcast_id = engine.start_broadcast(
            "Test Stream".to_string(),
            "A test stream".to_string(),
            ContentType::LiveStream,
        ).await.unwrap();
        
        assert_eq!(engine.get_active_broadcast_count(), 1);
        
        engine.stop_broadcast(broadcast_id).await.unwrap();
        
        assert_eq!(engine.get_active_broadcast_count(), 0);
    }
    
    #[tokio::test]
    async fn test_viewing_session() {
        let mut engine = create_test_engine().await;
        
        let content_id = Uuid::new_v4();
        let peer_id = PeerId::random();
        let session_id = engine.join_viewing_session(content_id, peer_id).await.unwrap();
        
        assert_eq!(engine.get_active_viewer_count(), 1);
        
        engine.leave_viewing_session(session_id).await.unwrap();
        
        assert_eq!(engine.get_active_viewer_count(), 0);
    }
}