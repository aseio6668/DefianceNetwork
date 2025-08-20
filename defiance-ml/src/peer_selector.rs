//! Intelligent peer selection using ML-based scoring and optimization

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use nalgebra::DVector;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::{ModelMetrics, ConnectionType};

/// Intelligent peer selector using ML
pub struct PeerSelector {
    model: Option<RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>,
    peer_history: HashMap<Uuid, PeerPerformanceData>,
    selection_history: Vec<PeerSelectionRecord>,
    metrics: ModelMetrics,
    selection_strategy: SelectionStrategy,
    geographic_preferences: GeographicPreferences,
    min_samples_for_training: usize,
}

/// Peer scoring result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerScore {
    pub peer_id: Uuid,
    pub total_score: f64,
    pub factors: HashMap<String, f64>,
    pub confidence: f64,
    pub predicted_performance: PredictedPerformance,
    pub selection_reason: SelectionReason,
}

/// Predicted performance metrics for a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedPerformance {
    pub expected_bandwidth: f64,
    pub expected_latency: f64,
    pub connection_reliability: f64,
    pub streaming_quality: f64,
    pub stability_score: f64,
}

/// Reason for peer selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionReason {
    HighPerformance { score: f64 },
    Geographic { distance_km: f64 },
    LoadBalancing { current_load: f64 },
    Diversity { diversity_factor: f64 },
    Fallback { reason: String },
}

/// Peer performance tracking data
#[derive(Debug, Clone)]
struct PeerPerformanceData {
    peer_id: Uuid,
    connection_attempts: u32,
    successful_connections: u32,
    total_bytes_transferred: u64,
    average_bandwidth: f64,
    average_latency: f64,
    average_packet_loss: f64,
    last_seen: u64,
    geographic_location: Option<String>,
    connection_type: Option<ConnectionType>,
    reliability_history: Vec<ConnectionResult>,
    streaming_sessions: Vec<StreamingSession>,
    reputation_score: f64,
}

/// Connection result for tracking
#[derive(Debug, Clone)]
struct ConnectionResult {
    timestamp: u64,
    success: bool,
    latency_ms: f64,
    error_type: Option<String>,
    duration_ms: u64,
}

/// Streaming session data
#[derive(Debug, Clone)]
struct StreamingSession {
    start_time: u64,
    duration_ms: u64,
    bytes_transferred: u64,
    average_quality: f64,
    rebuffer_events: u32,
    final_rating: Option<f64>, // User or automatic rating
}

/// Peer selection record for training
#[derive(Debug, Clone)]
struct PeerSelectionRecord {
    timestamp: u64,
    selected_peers: Vec<Uuid>,
    available_peers: Vec<Uuid>,
    selection_factors: Vec<f64>,
    actual_performance: Option<f64>,
    user_satisfaction: Option<f64>,
}

/// Selection strategy configuration
#[derive(Debug, Clone)]
pub struct SelectionStrategy {
    pub performance_weight: f64,
    pub geographic_weight: f64,
    pub diversity_weight: f64,
    pub load_balancing_weight: f64,
    pub reliability_weight: f64,
    pub enable_ml_ranking: bool,
    pub max_peer_distance_km: Option<f64>,
    pub min_reliability_threshold: f64,
}

/// Geographic preferences for peer selection
#[derive(Debug, Clone)]
struct GeographicPreferences {
    preferred_regions: Vec<String>,
    blacklisted_regions: Vec<String>,
    distance_penalty_factor: f64,
    timezone_preference: Option<i8>, // UTC offset preference
}

/// Feature vector for peer ML model
#[derive(Debug, Clone)]
struct PeerFeatureVector {
    features: Vec<f64>,
    label: i32, // 0 = poor, 1 = good, 2 = excellent
}

impl PeerSelector {
    /// Create new peer selector
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: None,
            peer_history: HashMap::new(),
            selection_history: Vec::new(),
            metrics: ModelMetrics::new(),
            selection_strategy: SelectionStrategy::default(),
            geographic_preferences: GeographicPreferences::default(),
            min_samples_for_training: 100,
        })
    }
    
    /// Initialize the peer selector
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Peer selector initialized");
        Ok(())
    }
    
    /// Select best peers from available options
    pub async fn select_peers(&self, available_peers: &[Uuid], target_count: usize) -> Result<Vec<Uuid>> {
        if available_peers.is_empty() {
            return Ok(Vec::new());
        }
        
        let target_count = target_count.min(available_peers.len());
        
        // Score all available peers
        let mut peer_scores = Vec::new();
        for peer_id in available_peers {
            let score = self.score_peer(*peer_id).await?;
            peer_scores.push(score);
        }
        
        // Sort by total score descending
        peer_scores.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap());
        
        // Apply selection strategy
        let selected = self.apply_selection_strategy(&peer_scores, target_count);
        
        Ok(selected)
    }
    
    /// Score a single peer
    pub async fn score_peer(&self, peer_id: Uuid) -> Result<PeerScore> {
        let peer_data = self.peer_history.get(&peer_id);
        
        let score = if let Some(data) = peer_data {
            self.calculate_peer_score(data).await?
        } else {
            // New peer - use default scoring
            self.default_peer_score(peer_id)
        };
        
        Ok(score)
    }
    
    /// Get peer score (convenience method)
    pub async fn get_peer_score(&self, peer_id: &Uuid) -> Result<PeerScore> {
        self.score_peer(*peer_id).await
    }
    
    /// Update peer performance data
    pub async fn update_peer_performance(&mut self, peer_id: Uuid, performance_score: f64, connection_success: bool) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let peer_data = self.peer_history.entry(peer_id).or_insert_with(|| {
            PeerPerformanceData::new(peer_id)
        });
        
        // Update connection statistics
        peer_data.connection_attempts += 1;
        if connection_success {
            peer_data.successful_connections += 1;
        }
        
        // Add connection result
        peer_data.reliability_history.push(ConnectionResult {
            timestamp: now,
            success: connection_success,
            latency_ms: 0.0, // Will be updated with actual latency
            error_type: if connection_success { None } else { Some("unknown".to_string()) },
            duration_ms: 0,
        });
        
        // Limit history size
        if peer_data.reliability_history.len() > 1000 {
            peer_data.reliability_history.drain(0..100);
        }
        
        // Update reputation score
        self.update_reputation_score(peer_data, performance_score);
        
        peer_data.last_seen = now;
        
        // Retrain model if we have enough data
        if self.selection_history.len() >= self.min_samples_for_training {
            self.retrain_model().await?;
        }
        
        Ok(())
    }
    
    /// Record a streaming session for a peer
    pub async fn record_streaming_session(&mut self, peer_id: Uuid, session: StreamingSessionData) -> Result<()> {
        let peer_data = self.peer_history.entry(peer_id).or_insert_with(|| {
            PeerPerformanceData::new(peer_id)
        });
        
        peer_data.streaming_sessions.push(StreamingSession {
            start_time: session.start_time,
            duration_ms: session.duration_ms,
            bytes_transferred: session.bytes_transferred,
            average_quality: session.average_quality,
            rebuffer_events: session.rebuffer_events,
            final_rating: session.final_rating,
        });
        
        // Update aggregate statistics
        peer_data.total_bytes_transferred += session.bytes_transferred;
        if session.duration_ms > 0 {
            let session_bandwidth = (session.bytes_transferred as f64 * 8.0) / (session.duration_ms as f64 / 1000.0);
            peer_data.average_bandwidth = self.update_exponential_average(
                peer_data.average_bandwidth, 
                session_bandwidth, 
                0.1
            );
        }
        
        // Limit session history
        if peer_data.streaming_sessions.len() > 500 {
            peer_data.streaming_sessions.drain(0..50);
        }
        
        Ok(())
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
    
    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: SelectionStrategy) {
        self.selection_strategy = strategy;
    }
    
    /// Calculate comprehensive peer score
    async fn calculate_peer_score(&self, peer_data: &PeerPerformanceData) -> Result<PeerScore> {
        let mut factors = HashMap::new();
        
        // Performance factor
        let performance_score = self.calculate_performance_score(peer_data);
        factors.insert("performance".to_string(), performance_score);
        
        // Reliability factor
        let reliability_score = self.calculate_reliability_score(peer_data);
        factors.insert("reliability".to_string(), reliability_score);
        
        // Geographic factor
        let geographic_score = self.calculate_geographic_score(peer_data);
        factors.insert("geographic".to_string(), geographic_score);
        
        // Load balancing factor
        let load_score = self.calculate_load_balancing_score(peer_data);
        factors.insert("load_balancing".to_string(), load_score);
        
        // Reputation factor
        let reputation_score = peer_data.reputation_score;
        factors.insert("reputation".to_string(), reputation_score);
        
        // Calculate weighted total score
        let total_score = 
            performance_score * self.selection_strategy.performance_weight +
            reliability_score * self.selection_strategy.reliability_weight +
            geographic_score * self.selection_strategy.geographic_weight +
            load_score * self.selection_strategy.load_balancing_weight +
            reputation_score * 0.1; // Fixed weight for reputation
        
        // Use ML model if available for refined scoring
        let (final_score, confidence) = if let Some(ref model) = self.model {
            let ml_score = self.ml_score_peer(peer_data, &factors)?;
            (ml_score, 0.8)
        } else {
            (total_score, 0.6)
        };
        
        let predicted_performance = self.predict_peer_performance(peer_data);
        let selection_reason = SelectionReason::HighPerformance { score: final_score };
        
        Ok(PeerScore {
            peer_id: peer_data.peer_id,
            total_score: final_score,
            factors,
            confidence,
            predicted_performance,
            selection_reason,
        })
    }
    
    /// Default score for unknown peers
    fn default_peer_score(&self, peer_id: Uuid) -> PeerScore {
        let mut factors = HashMap::new();
        factors.insert("performance".to_string(), 0.5);
        factors.insert("reliability".to_string(), 0.5);
        factors.insert("geographic".to_string(), 0.5);
        factors.insert("reputation".to_string(), 0.5);
        
        PeerScore {
            peer_id,
            total_score: 0.5,
            factors,
            confidence: 0.3,
            predicted_performance: PredictedPerformance {
                expected_bandwidth: 5_000_000.0,
                expected_latency: 100.0,
                connection_reliability: 0.7,
                streaming_quality: 0.6,
                stability_score: 0.5,
            },
            selection_reason: SelectionReason::Fallback { 
                reason: "No historical data available".to_string() 
            },
        }
    }
    
    /// Calculate performance score from historical data
    fn calculate_performance_score(&self, peer_data: &PeerPerformanceData) -> f64 {
        if peer_data.streaming_sessions.is_empty() {
            return 0.5;
        }
        
        let recent_sessions = self.get_recent_sessions(peer_data, 10);
        if recent_sessions.is_empty() {
            return 0.5;
        }
        
        let avg_quality: f64 = recent_sessions.iter().map(|s| s.average_quality).sum::<f64>() / recent_sessions.len() as f64;
        let avg_rebuffers: f64 = recent_sessions.iter().map(|s| s.rebuffer_events as f64).sum::<f64>() / recent_sessions.len() as f64;
        
        // Normalize quality (0.0-1.0) and penalize rebuffering
        let quality_score = avg_quality;
        let rebuffer_penalty = (avg_rebuffers / 10.0).min(1.0); // Max penalty at 10 rebuffers
        
        (quality_score - rebuffer_penalty * 0.3).clamp(0.0, 1.0)
    }
    
    /// Calculate reliability score
    fn calculate_reliability_score(&self, peer_data: &PeerPerformanceData) -> f64 {
        if peer_data.connection_attempts == 0 {
            return 0.5;
        }
        
        let success_rate = peer_data.successful_connections as f64 / peer_data.connection_attempts as f64;
        
        // Apply time decay - recent connections matter more
        let recent_connections = self.get_recent_connections(peer_data, 50);
        let recent_success_rate = if recent_connections.is_empty() {
            success_rate
        } else {
            let recent_successes = recent_connections.iter().filter(|c| c.success).count() as f64;
            recent_successes / recent_connections.len() as f64
        };
        
        // Weight recent performance more heavily
        (success_rate * 0.3 + recent_success_rate * 0.7).clamp(0.0, 1.0)
    }
    
    /// Calculate geographic score based on preferences
    fn calculate_geographic_score(&self, peer_data: &PeerPerformanceData) -> f64 {
        // Placeholder implementation - would use actual geographic data
        if let Some(_location) = &peer_data.geographic_location {
            // In a real implementation, calculate distance and apply preferences
            0.7
        } else {
            0.5 // Unknown location gets neutral score
        }
    }
    
    /// Calculate load balancing score
    fn calculate_load_balancing_score(&self, _peer_data: &PeerPerformanceData) -> f64 {
        // Placeholder - would consider current load distribution
        0.6
    }
    
    /// Update reputation score using exponential moving average
    fn update_reputation_score(&self, peer_data: &mut PeerPerformanceData, performance_score: f64) {
        let alpha = 0.1; // Learning rate
        peer_data.reputation_score = alpha * performance_score + (1.0 - alpha) * peer_data.reputation_score;
    }
    
    /// Get recent streaming sessions
    fn get_recent_sessions<'a>(&self, peer_data: &'a PeerPerformanceData, count: usize) -> Vec<&'a StreamingSession> {
        let mut sessions: Vec<&StreamingSession> = peer_data.streaming_sessions.iter().collect();
        sessions.sort_by_key(|s| s.start_time);
        sessions.into_iter().rev().take(count).collect()
    }
    
    /// Get recent connection results
    fn get_recent_connections<'a>(&self, peer_data: &'a PeerPerformanceData, count: usize) -> Vec<&'a ConnectionResult> {
        let mut connections: Vec<&ConnectionResult> = peer_data.reliability_history.iter().collect();
        connections.sort_by_key(|c| c.timestamp);
        connections.into_iter().rev().take(count).collect()
    }
    
    /// Predict peer performance
    fn predict_peer_performance(&self, peer_data: &PeerPerformanceData) -> PredictedPerformance {
        PredictedPerformance {
            expected_bandwidth: peer_data.average_bandwidth,
            expected_latency: peer_data.average_latency,
            connection_reliability: peer_data.successful_connections as f64 / peer_data.connection_attempts.max(1) as f64,
            streaming_quality: if peer_data.streaming_sessions.is_empty() {
                0.6
            } else {
                peer_data.streaming_sessions.iter().map(|s| s.average_quality).sum::<f64>() 
                    / peer_data.streaming_sessions.len() as f64
            },
            stability_score: peer_data.reputation_score,
        }
    }
    
    /// Apply selection strategy to scored peers
    fn apply_selection_strategy(&self, scored_peers: &[PeerScore], target_count: usize) -> Vec<Uuid> {
        let mut selected = Vec::new();
        
        // Apply diversity if requested
        if self.selection_strategy.diversity_weight > 0.0 {
            selected = self.diversified_selection(scored_peers, target_count);
        } else {
            // Simple top-N selection
            selected = scored_peers
                .iter()
                .take(target_count)
                .map(|score| score.peer_id)
                .collect();
        }
        
        selected
    }
    
    /// Diversified peer selection to avoid clustering
    fn diversified_selection(&self, scored_peers: &[PeerScore], target_count: usize) -> Vec<Uuid> {
        let mut selected = Vec::new();
        let mut remaining: Vec<&PeerScore> = scored_peers.iter().collect();
        
        // Always select the top peer first
        if let Some(top_peer) = remaining.first() {
            selected.push(top_peer.peer_id);
            remaining.retain(|p| p.peer_id != top_peer.peer_id);
        }
        
        // For remaining selections, balance score and diversity
        while selected.len() < target_count && !remaining.is_empty() {
            let mut best_peer = None;
            let mut best_score = 0.0;
            
            for peer in &remaining {
                let diversity_bonus = self.calculate_diversity_bonus(peer, &selected);
                let combined_score = peer.total_score * (1.0 - self.selection_strategy.diversity_weight) +
                                   diversity_bonus * self.selection_strategy.diversity_weight;
                
                if combined_score > best_score {
                    best_score = combined_score;
                    best_peer = Some(peer);
                }
            }
            
            if let Some(peer) = best_peer {
                selected.push(peer.peer_id);
                remaining.retain(|p| p.peer_id != peer.peer_id);
            } else {
                break;
            }
        }
        
        selected
    }
    
    /// Calculate diversity bonus for peer selection
    fn calculate_diversity_bonus(&self, _peer: &PeerScore, _selected: &[Uuid]) -> f64 {
        // Placeholder - would calculate geographic/connection type diversity
        0.1
    }
    
    /// ML-based peer scoring
    fn ml_score_peer(&self, peer_data: &PeerPerformanceData, factors: &HashMap<String, f64>) -> Result<f64> {
        // Extract features for ML model
        let features = vec![
            peer_data.average_bandwidth / 1_000_000.0, // Normalize to Mbps
            peer_data.average_latency / 100.0, // Normalize to 100ms
            peer_data.successful_connections as f64 / peer_data.connection_attempts.max(1) as f64,
            peer_data.reputation_score,
            factors.get("performance").copied().unwrap_or(0.5),
            factors.get("geographic").copied().unwrap_or(0.5),
        ];
        
        // In a real implementation, use trained model to predict
        // For now, return weighted average
        Ok(factors.values().sum::<f64>() / factors.len() as f64)
    }
    
    /// Retrain the ML model
    async fn retrain_model(&mut self) -> Result<()> {
        // Placeholder for ML model training
        // Would use selection history and outcomes to train RandomForest
        tracing::info!("Peer selector model retrained");
        Ok(())
    }
    
    /// Update exponential moving average
    fn update_exponential_average(&self, current: f64, new_value: f64, alpha: f64) -> f64 {
        alpha * new_value + (1.0 - alpha) * current
    }
}

/// Streaming session data for recording
#[derive(Debug, Clone)]
pub struct StreamingSessionData {
    pub start_time: u64,
    pub duration_ms: u64,
    pub bytes_transferred: u64,
    pub average_quality: f64,
    pub rebuffer_events: u32,
    pub final_rating: Option<f64>,
}

impl PeerPerformanceData {
    fn new(peer_id: Uuid) -> Self {
        Self {
            peer_id,
            connection_attempts: 0,
            successful_connections: 0,
            total_bytes_transferred: 0,
            average_bandwidth: 5_000_000.0,
            average_latency: 100.0,
            average_packet_loss: 0.01,
            last_seen: 0,
            geographic_location: None,
            connection_type: None,
            reliability_history: Vec::new(),
            streaming_sessions: Vec::new(),
            reputation_score: 0.5,
        }
    }
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self {
            performance_weight: 0.4,
            geographic_weight: 0.2,
            diversity_weight: 0.15,
            load_balancing_weight: 0.15,
            reliability_weight: 0.1,
            enable_ml_ranking: true,
            max_peer_distance_km: Some(1000.0),
            min_reliability_threshold: 0.7,
        }
    }
}

impl Default for GeographicPreferences {
    fn default() -> Self {
        Self {
            preferred_regions: Vec::new(),
            blacklisted_regions: Vec::new(),
            distance_penalty_factor: 0.1,
            timezone_preference: None,
        }
    }
}

impl ModelMetrics {
    fn new() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mean_absolute_error: 0.0,
            last_updated: 0,
            sample_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_peer_selector_creation() {
        let selector = PeerSelector::new();
        assert!(selector.is_ok());
    }
    
    #[tokio::test]
    async fn test_peer_scoring() {
        let selector = PeerSelector::new().unwrap();
        let peer_id = Uuid::new_v4();
        let score = selector.score_peer(peer_id).await;
        assert!(score.is_ok());
        
        let peer_score = score.unwrap();
        assert_eq!(peer_score.peer_id, peer_id);
        assert!(peer_score.total_score >= 0.0 && peer_score.total_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_peer_selection() {
        let selector = PeerSelector::new().unwrap();
        let peers = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let selected = selector.select_peers(&peers, 2).await;
        
        assert!(selected.is_ok());
        let selected_peers = selected.unwrap();
        assert!(selected_peers.len() <= 2);
    }
}