//! Peer discovery and network bootstrapping for DefianceNetwork

use std::collections::HashMap;
use std::time::Duration;
use libp2p::{PeerId, Multiaddr};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use anyhow::Result;

/// Peer discovery manager with multiple fallback strategies
pub struct PeerDiscovery {
    github_repo: Option<String>,
    known_bootstrap_nodes: Vec<BootstrapNode>,
    discovered_peers: Arc<RwLock<HashMap<PeerId, DiscoveredPeer>>>,
    event_sender: mpsc::UnboundedSender<DiscoveryEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<DiscoveryEvent>>,
}

/// Bootstrap node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapNode {
    pub peer_id: String,
    pub multiaddr: String,
    pub region: Option<String>,
    pub capabilities: Vec<String>,
    pub last_seen: i64,
    pub reliability_score: f32,
}

/// Discovered peer information
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub discovery_method: DiscoveryMethod,
    pub first_seen: i64,
    pub last_seen: i64,
    pub connection_attempts: u32,
    pub successful_connections: u32,
}

/// Methods used to discover peers
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryMethod {
    GitHubBootstrap,
    LocalMDNS,
    KademliaDHT,
    ManualSeed,
    PeerExchange,
}

/// Discovery events
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    PeerDiscovered { peer: DiscoveredPeer },
    PeerLost { peer_id: PeerId },
    BootstrapNodeUpdated { node: BootstrapNode },
    GitHubSyncCompleted { nodes_found: usize },
    DiscoveryFailed { method: DiscoveryMethod, error: String },
}

/// GitHub-based peer registry format
#[derive(Debug, Serialize, Deserialize)]
pub struct GitHubPeerRegistry {
    pub version: String,
    pub last_updated: i64,
    pub bootstrap_nodes: Vec<BootstrapNode>,
    pub regions: HashMap<String, Vec<BootstrapNode>>,
    pub network_stats: NetworkStats,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_nodes: u64,
    pub active_nodes: u64,
    pub average_uptime: f32,
    pub network_health: f32,
}

impl PeerDiscovery {
    /// Create new peer discovery manager
    pub fn new() -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Self {
            github_repo: None,
            known_bootstrap_nodes: Vec::new(),
            discovered_peers: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Some(event_receiver),
        }
    }
    
    /// Configure GitHub repository for peer discovery
    pub fn with_github_repo(mut self, repo: String) -> Self {
        self.github_repo = Some(repo);
        self
    }
    
    /// Add manual bootstrap nodes
    pub fn add_bootstrap_nodes(&mut self, nodes: Vec<BootstrapNode>) {
        self.known_bootstrap_nodes.extend(nodes);
        tracing::info!("Added {} bootstrap nodes", self.known_bootstrap_nodes.len());
    }
    
    /// Start peer discovery process
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting peer discovery");
        
        // Start GitHub-based discovery if configured
        if let Some(repo) = &self.github_repo {
            self.start_github_discovery(repo.clone()).await?;
        }
        
        // Start periodic bootstrap node refresh
        self.start_bootstrap_refresh().await?;
        
        Ok(())
    }
    
    /// Start GitHub-based peer discovery
    async fn start_github_discovery(&self, repo: String) -> Result<()> {
        let event_sender = self.event_sender.clone();
        let discovered_peers = Arc::clone(&self.discovered_peers);
        
        tokio::spawn(async move {
            loop {
                match Self::fetch_github_peer_registry(&repo).await {
                    Ok(registry) => {
                        tracing::info!("Fetched {} bootstrap nodes from GitHub", registry.bootstrap_nodes.len());
                        
                        let nodes_count = registry.bootstrap_nodes.len();
                        for node in registry.bootstrap_nodes {
                            if let Ok(peer_id) = node.peer_id.parse::<PeerId>() {
                                if let Ok(multiaddr) = node.multiaddr.parse::<Multiaddr>() {
                                    let discovered_peer = DiscoveredPeer {
                                        peer_id,
                                        addresses: vec![multiaddr],
                                        discovery_method: DiscoveryMethod::GitHubBootstrap,
                                        first_seen: chrono::Utc::now().timestamp(),
                                        last_seen: chrono::Utc::now().timestamp(),
                                        connection_attempts: 0,
                                        successful_connections: 0,
                                    };
                                    
                                    {
                                        let mut peers = discovered_peers.write().await;
                                        peers.insert(peer_id, discovered_peer.clone());
                                    }
                                    
                                    let _ = event_sender.send(DiscoveryEvent::PeerDiscovered { 
                                        peer: discovered_peer 
                                    });
                                }
                            }
                        }
                        
                        let _ = event_sender.send(DiscoveryEvent::GitHubSyncCompleted { 
                            nodes_found: nodes_count 
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to fetch GitHub peer registry: {}", e);
                        let _ = event_sender.send(DiscoveryEvent::DiscoveryFailed {
                            method: DiscoveryMethod::GitHubBootstrap,
                            error: e.to_string(),
                        });
                    }
                }
                
                // Refresh every 5 minutes
                tokio::time::sleep(Duration::from_secs(300)).await;
            }
        });
        
        Ok(())
    }
    
    /// Fetch peer registry from GitHub
    async fn fetch_github_peer_registry(repo: &str) -> Result<GitHubPeerRegistry> {
        let url = format!("https://raw.githubusercontent.com/{}/main/peer-registry.json", repo);
        
        let response = reqwest::get(&url).await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("GitHub API returned status: {}", response.status()));
        }
        
        let registry: GitHubPeerRegistry = response.json().await?;
        Ok(registry)
    }
    
    /// Start periodic bootstrap node refresh
    async fn start_bootstrap_refresh(&self) -> Result<()> {
        let _event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1800)); // 30 minutes
            
            loop {
                interval.tick().await;
                
                // TODO: Implement bootstrap node health checks and updates
                tracing::debug!("Performing bootstrap node refresh");
            }
        });
        
        Ok(())
    }
    
    /// Get all discovered peers
    pub async fn get_discovered_peers(&self) -> HashMap<PeerId, DiscoveredPeer> {
        let peers = self.discovered_peers.read().await;
        peers.clone()
    }
    
    /// Get peers discovered by a specific method
    pub async fn get_peers_by_method(&self, method: DiscoveryMethod) -> Vec<DiscoveredPeer> {
        let peers = self.discovered_peers.read().await;
        peers.values()
            .filter(|peer| peer.discovery_method == method)
            .cloned()
            .collect()
    }
    
    /// Get the best bootstrap nodes for connecting
    pub async fn get_best_bootstrap_nodes(&self, count: usize) -> Vec<BootstrapNode> {
        let mut nodes = self.known_bootstrap_nodes.clone();
        
        // Sort by reliability score (descending)
        nodes.sort_by(|a, b| b.reliability_score.partial_cmp(&a.reliability_score).unwrap());
        
        nodes.into_iter().take(count).collect()
    }
    
    /// Record a successful connection to a peer
    pub async fn record_successful_connection(&self, peer_id: &PeerId) {
        let mut peers = self.discovered_peers.write().await;
        if let Some(peer) = peers.get_mut(peer_id) {
            peer.successful_connections += 1;
            peer.last_seen = chrono::Utc::now().timestamp();
        }
    }
    
    /// Record a failed connection attempt
    pub async fn record_failed_connection(&self, peer_id: &PeerId) {
        let mut peers = self.discovered_peers.write().await;
        if let Some(peer) = peers.get_mut(peer_id) {
            peer.connection_attempts += 1;
        }
    }
    
    /// Remove a peer that's no longer reachable
    pub async fn remove_peer(&self, peer_id: &PeerId) {
        let mut peers = self.discovered_peers.write().await;
        if peers.remove(peer_id).is_some() {
            let _ = self.event_sender.send(DiscoveryEvent::PeerLost { 
                peer_id: *peer_id 
            });
            tracing::info!("Removed unreachable peer: {}", peer_id);
        }
    }
    
    /// Add a manually discovered peer
    pub async fn add_manual_peer(&self, peer_id: PeerId, addresses: Vec<Multiaddr>) {
        let discovered_peer = DiscoveredPeer {
            peer_id,
            addresses,
            discovery_method: DiscoveryMethod::ManualSeed,
            first_seen: chrono::Utc::now().timestamp(),
            last_seen: chrono::Utc::now().timestamp(),
            connection_attempts: 0,
            successful_connections: 0,
        };
        
        {
            let mut peers = self.discovered_peers.write().await;
            peers.insert(peer_id, discovered_peer.clone());
        }
        
        let _ = self.event_sender.send(DiscoveryEvent::PeerDiscovered { 
            peer: discovered_peer 
        });
    }
    
    /// Get discovery statistics
    pub async fn get_discovery_stats(&self) -> DiscoveryStats {
        let peers = self.discovered_peers.read().await;
        
        let mut stats = DiscoveryStats::default();
        stats.total_peers = peers.len();
        
        for peer in peers.values() {
            match peer.discovery_method {
                DiscoveryMethod::GitHubBootstrap => stats.github_peers += 1,
                DiscoveryMethod::LocalMDNS => stats.mdns_peers += 1,
                DiscoveryMethod::KademliaDHT => stats.dht_peers += 1,
                DiscoveryMethod::ManualSeed => stats.manual_peers += 1,
                DiscoveryMethod::PeerExchange => stats.peer_exchange_peers += 1,
            }
            
            if peer.successful_connections > 0 {
                stats.successful_connections += 1;
            }
        }
        
        stats
    }
    
    /// Take the event receiver for processing discovery events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<DiscoveryEvent>> {
        self.event_receiver.take()
    }
    
    /// Generate a peer registry for publishing to GitHub
    pub async fn generate_peer_registry(&self) -> GitHubPeerRegistry {
        let peers = self.discovered_peers.read().await;
        
        let bootstrap_nodes: Vec<BootstrapNode> = peers.values()
            .filter(|peer| peer.successful_connections > 0)
            .map(|peer| BootstrapNode {
                peer_id: peer.peer_id.to_string(),
                multiaddr: peer.addresses.first()
                    .map(|addr| addr.to_string())
                    .unwrap_or_default(),
                region: None, // TODO: Implement region detection
                capabilities: vec!["streaming".to_string(), "content".to_string()],
                last_seen: peer.last_seen,
                reliability_score: if peer.connection_attempts > 0 {
                    peer.successful_connections as f32 / peer.connection_attempts as f32
                } else {
                    0.0
                },
            })
            .collect();
        
        GitHubPeerRegistry {
            version: "1.0.0".to_string(),
            last_updated: chrono::Utc::now().timestamp(),
            bootstrap_nodes,
            regions: HashMap::new(), // TODO: Implement regional grouping
            network_stats: NetworkStats {
                total_nodes: peers.len() as u64,
                active_nodes: peers.values()
                    .filter(|p| p.successful_connections > 0)
                    .count() as u64,
                average_uptime: 0.85, // TODO: Calculate actual uptime
                network_health: 0.90, // TODO: Calculate actual health
            },
        }
    }
}

/// Discovery statistics
#[derive(Debug, Default)]
pub struct DiscoveryStats {
    pub total_peers: usize,
    pub github_peers: usize,
    pub mdns_peers: usize,
    pub dht_peers: usize,
    pub manual_peers: usize,
    pub peer_exchange_peers: usize,
    pub successful_connections: usize,
}

impl Default for PeerDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_peer_discovery_creation() {
        let discovery = PeerDiscovery::new();
        assert!(discovery.github_repo.is_none());
        assert!(discovery.known_bootstrap_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_github_repo_configuration() {
        let discovery = PeerDiscovery::new()
            .with_github_repo("DefianceNetwork/seed-nodes".to_string());
        assert!(discovery.github_repo.is_some());
    }

    #[tokio::test]
    async fn test_manual_peer_addition() {
        let discovery = PeerDiscovery::new();
        let peer_id = PeerId::random();
        let addresses = vec!["/ip4/127.0.0.1/tcp/9080".parse().unwrap()];
        
        discovery.add_manual_peer(peer_id, addresses).await;
        
        let peers = discovery.get_discovered_peers().await;
        assert_eq!(peers.len(), 1);
        assert!(peers.contains_key(&peer_id));
    }

    #[test]
    fn test_bootstrap_node_serialization() {
        let node = BootstrapNode {
            peer_id: "12D3KooWExample".to_string(),
            multiaddr: "/ip4/192.168.1.100/tcp/9080".to_string(),
            region: Some("us-west".to_string()),
            capabilities: vec!["streaming".to_string()],
            last_seen: chrono::Utc::now().timestamp(),
            reliability_score: 0.95,
        };
        
        let serialized = serde_json::to_string(&node);
        assert!(serialized.is_ok());
        
        let deserialized: Result<BootstrapNode, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_peer_registry_generation() {
        let registry = GitHubPeerRegistry {
            version: "1.0.0".to_string(),
            last_updated: chrono::Utc::now().timestamp(),
            bootstrap_nodes: vec![],
            regions: HashMap::new(),
            network_stats: NetworkStats {
                total_nodes: 10,
                active_nodes: 8,
                average_uptime: 0.92,
                network_health: 0.88,
            },
        };
        
        let serialized = serde_json::to_string_pretty(&registry);
        assert!(serialized.is_ok());
    }
}