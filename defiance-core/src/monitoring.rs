//! Network monitoring and health indicators for DefianceNetwork

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use anyhow::Result;

/// Network health monitoring system
pub struct NetworkMonitor {
    peer_metrics: Arc<RwLock<HashMap<PeerId, PeerMetrics>>>,
    network_health: Arc<RwLock<NetworkHealth>>,
    bandwidth_history: Arc<RwLock<VecDeque<BandwidthSample>>>,
    latency_history: Arc<RwLock<VecDeque<LatencySample>>>,
    event_sender: mpsc::UnboundedSender<MonitoringEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<MonitoringEvent>>,
    start_time: Instant,
}

/// Metrics for individual peers
#[derive(Debug, Clone)]
pub struct PeerMetrics {
    pub peer_id: PeerId,
    pub connection_time: Duration,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub average_latency: Duration,
    pub packet_loss_rate: f32,
    pub connection_stability: f32,
    pub last_seen: Instant,
    pub capability_scores: HashMap<String, f32>,
}

/// Overall network health information
#[derive(Debug, Clone)]
pub struct NetworkHealth {
    pub connected_peers: usize,
    pub active_streams: usize,
    pub total_bandwidth_in: u64,
    pub total_bandwidth_out: u64,
    pub average_latency: Duration,
    pub network_stability: f32,
    pub content_availability: f32,
    pub regional_distribution: HashMap<String, usize>,
    pub protocol_health: HashMap<String, f32>,
    pub last_updated: Instant,
}

/// Bandwidth measurement sample
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    pub timestamp: Instant,
    pub upload_bps: u64,
    pub download_bps: u64,
    pub peer_count: usize,
}

/// Latency measurement sample
#[derive(Debug, Clone)]
pub struct LatencySample {
    pub timestamp: Instant,
    pub peer_id: PeerId,
    pub latency: Duration,
    pub region: Option<String>,
}

/// Monitoring events
#[derive(Debug, Clone)]
pub enum MonitoringEvent {
    PeerConnected { peer_id: PeerId },
    PeerDisconnected { peer_id: PeerId },
    LatencyMeasured { peer_id: PeerId, latency: Duration },
    BandwidthMeasured { upload: u64, download: u64 },
    HealthStatusChanged { old_health: f32, new_health: f32 },
    NetworkCongestion { severity: CongestionLevel },
    PeerPerformanceAlert { peer_id: PeerId, issue: PerformanceIssue },
}

/// Network congestion levels
#[derive(Debug, Clone, PartialEq)]
pub enum CongestionLevel {
    Low,
    Moderate,
    High,
    Critical,
}

/// Performance issues detected in peers
#[derive(Debug, Clone)]
pub enum PerformanceIssue {
    HighLatency(Duration),
    PacketLoss(f32),
    LowBandwidth(u64),
    UnstableConnection,
    Timeout,
}

/// Network statistics for external reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub uptime: Duration,
    pub peer_count: usize,
    pub active_connections: usize,
    pub total_data_transferred: u64,
    pub average_latency_ms: u64,
    pub network_health_score: f32,
    pub regional_distribution: HashMap<String, usize>,
    pub content_distribution_efficiency: f32,
    pub bandwidth_utilization: f32,
}

impl NetworkMonitor {
    /// Create a new network monitor
    pub fn new() -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Self {
            peer_metrics: Arc::new(RwLock::new(HashMap::new())),
            network_health: Arc::new(RwLock::new(NetworkHealth::default())),
            bandwidth_history: Arc::new(RwLock::new(VecDeque::new())),
            latency_history: Arc::new(RwLock::new(VecDeque::new())),
            event_sender,
            event_receiver: Some(event_receiver),
            start_time: Instant::now(),
        }
    }
    
    /// Start monitoring services
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting network monitoring");
        
        self.start_metrics_collection().await?;
        self.start_health_calculation().await?;
        self.start_cleanup_task().await?;
        
        Ok(())
    }
    
    /// Start metrics collection loop
    async fn start_metrics_collection(&self) -> Result<()> {
        let peer_metrics = Arc::clone(&self.peer_metrics);
        let bandwidth_history = Arc::clone(&self.bandwidth_history);
        let event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                // Collect bandwidth metrics
                let (upload, download, peer_count) = {
                    let metrics = peer_metrics.read().await;
                    let upload = metrics.values().map(|m| m.bytes_sent).sum::<u64>();
                    let download = metrics.values().map(|m| m.bytes_received).sum::<u64>();
                    let peer_count = metrics.len();
                    (upload, download, peer_count)
                };
                
                // Store bandwidth sample
                {
                    let mut history = bandwidth_history.write().await;
                    history.push_back(BandwidthSample {
                        timestamp: Instant::now(),
                        upload_bps: upload,
                        download_bps: download,
                        peer_count,
                    });
                    
                    // Keep only last 100 samples (about 8 minutes)
                    while history.len() > 100 {
                        history.pop_front();
                    }
                }
                
                let _ = event_sender.send(MonitoringEvent::BandwidthMeasured {
                    upload,
                    download,
                });
            }
        });
        
        Ok(())
    }
    
    /// Start health calculation loop
    async fn start_health_calculation(&self) -> Result<()> {
        let peer_metrics = Arc::clone(&self.peer_metrics);
        let network_health = Arc::clone(&self.network_health);
        let bandwidth_history = Arc::clone(&self.bandwidth_history);
        let latency_history = Arc::clone(&self.latency_history);
        let event_sender = self.event_sender.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let old_health = {
                    let health = network_health.read().await;
                    health.network_stability
                };
                
                // Calculate new health metrics
                let new_health = Self::calculate_network_health(
                    &peer_metrics,
                    &bandwidth_history,
                    &latency_history,
                ).await;
                
                // Update network health
                {
                    let mut health = network_health.write().await;
                    *health = new_health.clone();
                }
                
                // Send health change event if significant
                if (old_health - new_health.network_stability).abs() > 0.1 {
                    let _ = event_sender.send(MonitoringEvent::HealthStatusChanged {
                        old_health,
                        new_health: new_health.network_stability,
                    });
                }
                
                // Check for network congestion
                let congestion = Self::assess_congestion(&new_health);
                if congestion != CongestionLevel::Low {
                    let _ = event_sender.send(MonitoringEvent::NetworkCongestion {
                        severity: congestion,
                    });
                }
            }
        });
        
        Ok(())
    }
    
    /// Start cleanup task for old metrics
    async fn start_cleanup_task(&self) -> Result<()> {
        let peer_metrics = Arc::clone(&self.peer_metrics);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                
                let cutoff_time = Instant::now() - Duration::from_secs(3600); // 1 hour
                
                let mut metrics = peer_metrics.write().await;
                metrics.retain(|_, peer_metric| peer_metric.last_seen > cutoff_time);
            }
        });
        
        Ok(())
    }
    
    /// Calculate overall network health
    async fn calculate_network_health(
        peer_metrics: &Arc<RwLock<HashMap<PeerId, PeerMetrics>>>,
        bandwidth_history: &Arc<RwLock<VecDeque<BandwidthSample>>>,
        latency_history: &Arc<RwLock<VecDeque<LatencySample>>>,
    ) -> NetworkHealth {
        let metrics = peer_metrics.read().await;
        let bandwidth_hist = bandwidth_history.read().await;
        let latency_hist = latency_history.read().await;
        
        let connected_peers = metrics.len();
        
        // Calculate average latency
        let average_latency = if latency_hist.is_empty() {
            Duration::from_millis(50) // Default
        } else {
            let total_latency: Duration = latency_hist.iter()
                .map(|sample| sample.latency)
                .sum();
            total_latency / latency_hist.len() as u32
        };
        
        // Calculate bandwidth totals
        let (total_bandwidth_in, total_bandwidth_out) = if let Some(latest) = bandwidth_hist.back() {
            (latest.download_bps, latest.upload_bps)
        } else {
            (0, 0)
        };
        
        // Calculate network stability (based on peer connection stability)
        let network_stability = if connected_peers > 0 {
            metrics.values()
                .map(|m| m.connection_stability)
                .sum::<f32>() / connected_peers as f32
        } else {
            0.0
        };
        
        // Calculate content availability (mock implementation)
        let content_availability = if connected_peers > 5 {
            0.95
        } else if connected_peers > 2 {
            0.80
        } else {
            0.60
        };
        
        // Regional distribution (simplified)
        let mut regional_distribution = HashMap::new();
        regional_distribution.insert("unknown".to_string(), connected_peers);
        
        // Protocol health (mock)
        let mut protocol_health = HashMap::new();
        protocol_health.insert("gossipsub".to_string(), 0.95);
        protocol_health.insert("kademlia".to_string(), 0.90);
        protocol_health.insert("mdns".to_string(), 0.85);
        
        NetworkHealth {
            connected_peers,
            active_streams: 0, // TODO: Get from streaming engine
            total_bandwidth_in,
            total_bandwidth_out,
            average_latency,
            network_stability,
            content_availability,
            regional_distribution,
            protocol_health,
            last_updated: Instant::now(),
        }
    }
    
    /// Assess network congestion level
    fn assess_congestion(health: &NetworkHealth) -> CongestionLevel {
        let latency_ms = health.average_latency.as_millis() as u64;
        
        if latency_ms > 500 || health.network_stability < 0.5 {
            CongestionLevel::Critical
        } else if latency_ms > 200 || health.network_stability < 0.7 {
            CongestionLevel::High
        } else if latency_ms > 100 || health.network_stability < 0.8 {
            CongestionLevel::Moderate
        } else {
            CongestionLevel::Low
        }
    }
    
    /// Record a peer connection
    pub async fn record_peer_connection(&self, peer_id: PeerId) {
        let mut metrics = self.peer_metrics.write().await;
        metrics.insert(peer_id, PeerMetrics::new(peer_id));
        
        let _ = self.event_sender.send(MonitoringEvent::PeerConnected { peer_id });
    }
    
    /// Record a peer disconnection
    pub async fn record_peer_disconnection(&self, peer_id: &PeerId) {
        let mut metrics = self.peer_metrics.write().await;
        metrics.remove(peer_id);
        
        let _ = self.event_sender.send(MonitoringEvent::PeerDisconnected { peer_id: *peer_id });
    }
    
    /// Record latency measurement for a peer
    pub async fn record_latency(&self, peer_id: PeerId, latency: Duration) {
        // Update peer metrics
        {
            let mut metrics = self.peer_metrics.write().await;
            if let Some(peer_metric) = metrics.get_mut(&peer_id) {
                peer_metric.average_latency = latency;
                peer_metric.last_seen = Instant::now();
            }
        }
        
        // Add to latency history
        {
            let mut history = self.latency_history.write().await;
            history.push_back(LatencySample {
                timestamp: Instant::now(),
                peer_id,
                latency,
                region: None, // TODO: Implement region detection
            });
            
            // Keep only last 1000 samples
            while history.len() > 1000 {
                history.pop_front();
            }
        }
        
        let _ = self.event_sender.send(MonitoringEvent::LatencyMeasured { peer_id, latency });
        
        // Check for performance issues
        if latency > Duration::from_millis(300) {
            let _ = self.event_sender.send(MonitoringEvent::PeerPerformanceAlert {
                peer_id,
                issue: PerformanceIssue::HighLatency(latency),
            });
        }
    }
    
    /// Record data transfer for a peer
    pub async fn record_data_transfer(&self, peer_id: &PeerId, bytes_sent: u64, bytes_received: u64) {
        let mut metrics = self.peer_metrics.write().await;
        if let Some(peer_metric) = metrics.get_mut(peer_id) {
            peer_metric.bytes_sent += bytes_sent;
            peer_metric.bytes_received += bytes_received;
            peer_metric.last_seen = Instant::now();
        }
    }
    
    /// Get current network health
    pub async fn get_network_health(&self) -> NetworkHealth {
        let health = self.network_health.read().await;
        health.clone()
    }
    
    /// Get peer metrics
    pub async fn get_peer_metrics(&self, peer_id: &PeerId) -> Option<PeerMetrics> {
        let metrics = self.peer_metrics.read().await;
        metrics.get(peer_id).cloned()
    }
    
    /// Get all peer metrics
    pub async fn get_all_peer_metrics(&self) -> HashMap<PeerId, PeerMetrics> {
        let metrics = self.peer_metrics.read().await;
        metrics.clone()
    }
    
    /// Get network statistics
    pub async fn get_network_stats(&self) -> NetworkStats {
        let health = self.network_health.read().await;
        let metrics = self.peer_metrics.read().await;
        
        let total_data_transferred = metrics.values()
            .map(|m| m.bytes_sent + m.bytes_received)
            .sum();
        
        NetworkStats {
            uptime: self.start_time.elapsed(),
            peer_count: health.connected_peers,
            active_connections: health.connected_peers, // Simplified
            total_data_transferred,
            average_latency_ms: health.average_latency.as_millis() as u64,
            network_health_score: health.network_stability,
            regional_distribution: health.regional_distribution.clone(),
            content_distribution_efficiency: health.content_availability,
            bandwidth_utilization: 0.75, // Mock value
        }
    }
    
    /// Take the event receiver for processing monitoring events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<MonitoringEvent>> {
        self.event_receiver.take()
    }
}

impl PeerMetrics {
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            connection_time: Duration::from_secs(0),
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            average_latency: Duration::from_millis(50),
            packet_loss_rate: 0.0,
            connection_stability: 1.0,
            last_seen: Instant::now(),
            capability_scores: HashMap::new(),
        }
    }
}

impl Default for NetworkHealth {
    fn default() -> Self {
        Self {
            connected_peers: 0,
            active_streams: 0,
            total_bandwidth_in: 0,
            total_bandwidth_out: 0,
            average_latency: Duration::from_millis(50),
            network_stability: 1.0,
            content_availability: 1.0,
            regional_distribution: HashMap::new(),
            protocol_health: HashMap::new(),
            last_updated: Instant::now(),
        }
    }
}

impl Default for NetworkMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;

    #[tokio::test]
    async fn test_network_monitor_creation() {
        let monitor = NetworkMonitor::new();
        let health = monitor.get_network_health().await;
        assert_eq!(health.connected_peers, 0);
    }

    #[tokio::test]
    async fn test_peer_connection_recording() {
        let monitor = NetworkMonitor::new();
        let peer_id = PeerId::random();
        
        monitor.record_peer_connection(peer_id).await;
        
        let metrics = monitor.get_peer_metrics(&peer_id).await;
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().peer_id, peer_id);
    }

    #[tokio::test]
    async fn test_latency_recording() {
        let monitor = NetworkMonitor::new();
        let peer_id = PeerId::random();
        let latency = Duration::from_millis(100);
        
        monitor.record_peer_connection(peer_id).await;
        monitor.record_latency(peer_id, latency).await;
        
        let metrics = monitor.get_peer_metrics(&peer_id).await;
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().average_latency, latency);
    }

    #[test]
    fn test_congestion_assessment() {
        let mut health = NetworkHealth::default();
        
        // Test low congestion
        health.average_latency = Duration::from_millis(50);
        health.network_stability = 0.9;
        assert_eq!(NetworkMonitor::assess_congestion(&health), CongestionLevel::Low);
        
        // Test high congestion
        health.average_latency = Duration::from_millis(300);
        health.network_stability = 0.6;
        assert_eq!(NetworkMonitor::assess_congestion(&health), CongestionLevel::High);
        
        // Test critical congestion
        health.average_latency = Duration::from_millis(600);
        health.network_stability = 0.4;
        assert_eq!(NetworkMonitor::assess_congestion(&health), CongestionLevel::Critical);
    }

    #[tokio::test]
    async fn test_network_stats_generation() {
        let monitor = NetworkMonitor::new();
        let peer_id = PeerId::random();
        
        monitor.record_peer_connection(peer_id).await;
        monitor.record_data_transfer(&peer_id, 1024, 2048).await;
        
        let stats = monitor.get_network_stats().await;
        assert_eq!(stats.peer_count, 1);
        assert_eq!(stats.total_data_transferred, 3072);
    }
}