//! P2P networking layer for DefianceNetwork

use std::collections::HashMap;
use std::time::Duration;
use libp2p::{
    PeerId, Multiaddr, Transport,
    swarm::{NetworkBehaviour, SwarmEvent},
    gossipsub, mdns, kad,
    noise, yamux, tcp,
    request_response::{self, ProtocolSupport},
    identity::Keypair,
    Swarm, SwarmBuilder, StreamProtocol
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use futures::StreamExt;
use anyhow::Result;

/// Network message types for DefianceNetwork P2P communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    /// Content streaming data chunk
    StreamChunk {
        content_id: Uuid,
        chunk_id: u64,
        data: Vec<u8>,
        checksum: [u8; 32],
    },
    /// Broadcast announcement
    BroadcastAnnouncement {
        broadcast_id: Uuid,
        title: String,
        description: String,
        content_type: String,
        broadcaster: String,
    },
    /// Request to join a broadcast
    JoinBroadcast {
        broadcast_id: Uuid,
        viewer_id: Uuid,
    },
    /// Response to join request
    JoinResponse {
        broadcast_id: Uuid,
        accepted: bool,
        stream_endpoints: Vec<Multiaddr>,
    },
    /// Peer discovery and capability announcement
    PeerAnnouncement {
        peer_id: Uuid,
        capabilities: Vec<String>,
        bandwidth: u64,
        location: Option<String>,
    },
    /// Network health check
    Heartbeat {
        peer_id: Uuid,
        timestamp: i64,
        load: f32,
    },
    /// Content metadata request
    ContentRequest {
        content_id: Uuid,
        chunk_range: Option<(u64, u64)>,
    },
    /// Content metadata response
    ContentResponse {
        content_id: Uuid,
        metadata: Vec<u8>,
        available_chunks: Vec<u64>,
    },
}

/// Type alias for the request-response Codec
type DefianceCodec = request_response::cbor::Behaviour<NetworkMessage, NetworkMessage>;

/// Custom network behaviour combining gossipsub, mDNS, and Kademlia DHT
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "DefianceBehaviourEvent")]
pub struct DefianceBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub mdns: mdns::tokio::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub request_response: DefianceCodec,
}

/// Events emitted by the DefianceBehaviour
#[derive(Debug)]
pub enum DefianceBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Mdns(mdns::Event),
    Kademlia(kad::Event),
    RequestResponse(request_response::Event<NetworkMessage, NetworkMessage>),
}

impl From<gossipsub::Event> for DefianceBehaviourEvent {
    fn from(event: gossipsub::Event) -> Self {
        DefianceBehaviourEvent::Gossipsub(event)
    }
}

impl From<mdns::Event> for DefianceBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        DefianceBehaviourEvent::Mdns(event)
    }
}

impl From<kad::Event> for DefianceBehaviourEvent {
    fn from(event: kad::Event) -> Self {
        DefianceBehaviourEvent::Kademlia(event)
    }
}

impl From<request_response::Event<NetworkMessage, NetworkMessage>> for DefianceBehaviourEvent {
    fn from(event: request_response::Event<NetworkMessage, NetworkMessage>) -> Self {
        DefianceBehaviourEvent::RequestResponse(event)
    }
}

/// P2P Network implementation for DefianceNetwork
pub struct P2PNetwork {
    node_id: Uuid,
    swarm: Option<Swarm<DefianceBehaviour>>,
    peers: Arc<RwLock<HashMap<PeerId, PeerInfo>>>,
    bandwidth_tracker: Arc<RwLock<BandwidthTracker>>,
    event_sender: mpsc::UnboundedSender<NetworkEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<NetworkEvent>>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: Uuid,
    pub address: Multiaddr,
    pub capabilities: Vec<String>,
    pub bandwidth: u64,
    pub latency: Option<u64>,
    pub last_seen: i64,
}

#[derive(Debug, Default)]
pub struct BandwidthTracker {
    upload_bytes: u64,
    download_bytes: u64,
    last_reset: i64,
}

/// Network events
#[derive(Debug)]
pub enum NetworkEvent {
    PeerConnected { peer_id: PeerId },
    PeerDisconnected { peer_id: PeerId },
    MessageReceived { from: PeerId, message: NetworkMessage },
    BroadcastReceived { topic: String, message: Vec<u8> },
    RequestResponseReceived { from: PeerId, message: request_response::Message<NetworkMessage, NetworkMessage> },
}

impl P2PNetwork {
    /// Create a new P2P network instance
    pub async fn new(node_id: Uuid, _port: u16) -> Result<Self> {
        tracing::info!("Initializing P2P network");
        
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            node_id,
            swarm: None,
            peers: Arc::new(RwLock::new(HashMap::new())),
            bandwidth_tracker: Arc::new(RwLock::new(BandwidthTracker::default())),
            event_sender,
            event_receiver: Some(event_receiver),
        })
    }

    /// Start the P2P network
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting P2P network for node {}", self.node_id);
        
        // Generate a random keypair for this node
        let local_key = Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        
        tracing::info!("Local peer ID: {}", local_peer_id);
        
        // Create transport using the new libp2p API
        let transport = {
            let tcp = tcp::tokio::Transport::default();
            let noise_config = noise::Config::new(&local_key).unwrap();
            let yamux_config = yamux::Config::default();
            
            tcp.upgrade(libp2p::core::upgrade::Version::V1)
                .authenticate(noise_config)
                .multiplex(yamux_config)
                .timeout(Duration::from_secs(20))
                .boxed()
        };
        
        // Set up gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|msg| anyhow::anyhow!("Wrong configuration: {}", msg))?;
        
        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        ).map_err(|e| anyhow::anyhow!("Failed to create gossipsub: {}", e))?;
        
        // Subscribe to DefianceNetwork topics
        gossipsub.subscribe(&gossipsub::IdentTopic::new("defiance/broadcasts"))?;
        gossipsub.subscribe(&gossipsub::IdentTopic::new("defiance/peers"))?;
        gossipsub.subscribe(&gossipsub::IdentTopic::new("defiance/content"))?;
        
        // Set up mDNS for local peer discovery
        let mdns = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            local_peer_id,
        )?;
        
        // Set up Kademlia DHT
        let store = kad::store::MemoryStore::new(local_peer_id);
        let kademlia = kad::Behaviour::new(local_peer_id, store);
        
        // Set up request-response
        let protocols = vec![(
            StreamProtocol::new("/defiance/req-resp/1.0.0"),
            ProtocolSupport::Full,
        )];
        let request_response = request_response::cbor::Behaviour::new(
            protocols,
            request_response::Config::default(),
        );
        
        // Create the network behaviour
        let behaviour = DefianceBehaviour {
            gossipsub,
            mdns,
            kademlia,
            request_response,
        };
        
        // Create the swarm
        let mut swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_other_transport(|_| transport)?
            .with_behaviour(|_| behaviour)?
            .build();
        
        // Start listening on all interfaces
        swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", 0).parse()?)?;
        
        self.swarm = Some(swarm);
        
        // Start the event loop
        self.start_event_loop().await?;
        
        Ok(())
    }

    /// Stop the P2P network
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping P2P network");
        self.swarm = None;
        Ok(())
    }

    /// Start the event processing loop
    async fn start_event_loop(&mut self) -> Result<()> {
        if let Some(mut swarm) = self.swarm.take() {
            let event_sender = self.event_sender.clone();
            
            tokio::spawn(async move {
                loop {
                    match swarm.select_next_some().await {
                        SwarmEvent::NewListenAddr { address, .. } => {
                            tracing::info!("Listening on {}", address);
                        }
                        SwarmEvent::Behaviour(event) => {
                            Self::handle_behaviour_event(event, &event_sender).await;
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            tracing::info!("Connected to peer: {}", peer_id);
                            let _ = event_sender.send(NetworkEvent::PeerConnected { peer_id });
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            tracing::info!("Disconnected from peer: {}", peer_id);
                            let _ = event_sender.send(NetworkEvent::PeerDisconnected { peer_id });
                        }
                        _ => {}
                    }
                }
            });
        }
        
        Ok(())
    }

    /// Handle behaviour events
    async fn handle_behaviour_event(
        event: DefianceBehaviourEvent,
        event_sender: &mpsc::UnboundedSender<NetworkEvent>,
    ) {
        match event {
            DefianceBehaviourEvent::Gossipsub(gossip_event) => {
                if let gossipsub::Event::Message { message, .. } = gossip_event {
                     let _ = event_sender.send(NetworkEvent::BroadcastReceived { topic: message.topic.into_string(), message: message.data });
                }
            }
            DefianceBehaviourEvent::Mdns(mdns_event) => {
                if let mdns::Event::Discovered(list) = mdns_event {
                    for (peer_id, _multiaddr) in list {
                        tracing::info!("mDNS discovered a new peer: {}", peer_id);
                    }
                }
            }
            DefianceBehaviourEvent::Kademlia(kad_event) => {
                tracing::debug!("Kademlia event: {:?}", kad_event);
            }
            DefianceBehaviourEvent::RequestResponse(rr_event) => {
                if let request_response::Event::Message { peer, message } = rr_event {
                    let _ = event_sender.send(NetworkEvent::RequestResponseReceived { from: peer, message });
                }
            }
        }
    }

    /// Broadcast a message to all connected peers
    pub async fn broadcast(&mut self, message: NetworkMessage) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
            let topic = gossipsub::IdentTopic::new("defiance/broadcasts");
            let serialized = bincode::serialize(&message)?;
            
            swarm.behaviour_mut().gossipsub.publish(topic, serialized)?;
            tracing::debug!("Broadcast message sent to network");
        }
        Ok(())
    }

    /// Send a direct message to a specific peer
    pub async fn send_to_peer(&mut self, peer_id: PeerId, message: NetworkMessage) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
           swarm.behaviour_mut().request_response.send_request(&peer_id, message);
           Ok(())
        } else {
            Err(anyhow::anyhow!("Swarm not initialized"))
        }
    }

    /// Get the number of connected peers
    pub fn get_peer_count(&self) -> usize {
        if let Some(swarm) = &self.swarm {
            swarm.connected_peers().count()
        } else {
            0
        }
    }

    /// Get current upload bandwidth in bytes per second
    pub fn get_upload_bandwidth(&self) -> u64 {
        // TODO: Calculate actual bandwidth from tracker
        0
    }

    /// Get current download bandwidth in bytes per second
    pub fn get_download_bandwidth(&self) -> u64 {
        // TODO: Calculate actual bandwidth from tracker
        0
    }

    /// Get average latency to peers in milliseconds
    pub fn get_average_latency(&self) -> Option<u64> {
        // TODO: Implement latency calculation
        Some(50) // Mock value
    }

    /// Add a new peer to the network
    pub async fn add_peer(&self, peer_id: PeerId, info: PeerInfo) {
        tracing::info!("Adding peer {} to network", peer_id);
        let mut peers = self.peers.write().await;
        peers.insert(peer_id, info);
    }

    /// Remove a peer from the network
    pub async fn remove_peer(&self, peer_id: &PeerId) {
        let mut peers = self.peers.write().await;
        if peers.remove(peer_id).is_some() {
            tracing::info!("Removed peer {} from network", peer_id);
        }
    }

    /// Get information about all connected peers
    pub async fn get_peers(&self) -> HashMap<PeerId, PeerInfo> {
        let peers = self.peers.read().await;
        peers.clone()
    }

    /// Take the event receiver for processing network events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<NetworkEvent>> {
        self.event_receiver.take()
    }

    /// Discover peers on the network
    pub async fn discover_peers(&mut self) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
            // Use Kademlia to bootstrap and find peers
            swarm.behaviour_mut().kademlia.bootstrap()?;
            tracing::info!("Started peer discovery via Kademlia DHT");
        }
        Ok(())
    }

    /// Join a specific topic for content discovery
    pub async fn join_topic(&mut self, topic: &str) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
            let topic = gossipsub::IdentTopic::new(topic);
            swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
            tracing::info!("Subscribed to topic: {}", topic);
        }
        Ok(())
    }

    /// Leave a topic
    pub async fn leave_topic(&mut self, topic: &str) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
            let topic = gossipsub::IdentTopic::new(topic);
            swarm.behaviour_mut().gossipsub.unsubscribe(&topic)?;
            tracing::info!("Unsubscribed from topic: {}", topic);
        }
        Ok(())
    }
}

impl BandwidthTracker {
    pub fn record_upload(&mut self, bytes: u64) {
        self.upload_bytes += bytes;
    }
    
    pub fn record_download(&mut self, bytes: u64) {
        self.download_bytes += bytes;
    }
    
    pub fn get_upload_rate(&self) -> u64 {
        // TODO: Calculate rate based on time window
        self.upload_bytes
    }
    
    pub fn get_download_rate(&self) -> u64 {
        // TODO: Calculate rate based on time window
        self.download_bytes
    }
    
    pub fn reset(&mut self) {
        self.upload_bytes = 0;
        self.download_bytes = 0;
        self.last_reset = chrono::Utc::now().timestamp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_p2p_network_creation() {
        let node_id = Uuid::new_v4();
        let network = P2PNetwork::new(node_id, 9080).await;
        assert!(network.is_ok());
    }

    #[test]
    fn test_bandwidth_tracker() {
        let mut tracker = BandwidthTracker::default();
        tracker.record_upload(1024);
        tracker.record_download(2048);
        
        assert_eq!(tracker.get_upload_rate(), 1024);
        assert_eq!(tracker.get_download_rate(), 2048);
    }

    #[test]
    fn test_network_message_serialization() {
        let message = NetworkMessage::Heartbeat {
            peer_id: Uuid::new_v4(),
            timestamp: chrono::Utc::now().timestamp(),
            load: 0.5,
        };
        
        let serialized = bincode::serialize(&message);
        assert!(serialized.is_ok());
        
        let deserialized: Result<NetworkMessage, _> = bincode::deserialize(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
}