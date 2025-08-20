//! # DefianceML - Network Intelligence & Optimization
//! 
//! Machine learning-powered network optimization for DefianceNetwork streaming platform.
//! Provides adaptive quality selection, latency prediction, bandwidth optimization,
//! and intelligent peer selection using lightweight ML algorithms.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use nalgebra::{DVector, DMatrix};
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

pub mod network_predictor;
pub mod quality_optimizer;
pub mod peer_selector;
pub mod bandwidth_estimator;
pub mod latency_predictor;
pub mod adaptive_streaming;
pub mod anomaly_detector;

// Re-export main components
pub use network_predictor::{NetworkPredictor, NetworkPrediction};
pub use quality_optimizer::{QualityOptimizer, QualityRecommendation};
pub use peer_selector::{PeerSelector, PeerScore};
pub use bandwidth_estimator::{BandwidthEstimator, BandwidthPrediction};
pub use latency_predictor::{LatencyPredictor, LatencyPrediction};
pub use adaptive_streaming::{AdaptiveStreaming, StreamingDecision};
pub use anomaly_detector::{AnomalyDetector, NetworkAnomaly};

/// Main ML optimization coordinator
pub struct DefianceML {
    network_predictor: Arc<RwLock<NetworkPredictor>>,
    quality_optimizer: Arc<RwLock<QualityOptimizer>>,
    peer_selector: Arc<RwLock<PeerSelector>>,
    bandwidth_estimator: Arc<RwLock<BandwidthEstimator>>,
    latency_predictor: Arc<RwLock<LatencyPredictor>>,
    adaptive_streaming: Arc<RwLock<AdaptiveStreaming>>,
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    
    metrics_history: Arc<RwLock<MetricsHistory>>,
    optimization_settings: OptimizationSettings,
    event_sender: mpsc::UnboundedSender<MLEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<MLEvent>>,
}

/// Network metrics for ML training and prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub timestamp: u64,
    pub peer_id: Option<Uuid>,
    pub latency_ms: f64,
    pub bandwidth_bps: f64,
    pub packet_loss: f64,
    pub jitter_ms: f64,
    pub connection_quality: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub battery_level: Option<f64>,
    pub connection_type: ConnectionType,
    pub location_hint: Option<String>,
}

/// Connection type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    Ethernet,
    WiFi,
    Cellular4G,
    Cellular5G,
    Satellite,
    Unknown,
}

/// Quality level for streaming
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QualityLevel {
    AudioOnly,
    Low,      // 480p
    Medium,   // 720p
    High,     // 1080p
    Ultra,    // 4K
}

/// ML optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    pub learning_rate: f64,
    pub prediction_window: Duration,
    pub min_samples_for_prediction: usize,
    pub quality_adaptation_aggressiveness: f64,
    pub peer_selection_diversity: f64,
    pub enable_anomaly_detection: bool,
    pub battery_optimization: bool,
    pub privacy_mode: bool, // Limits data collection
}

/// Historical metrics storage
pub struct MetricsHistory {
    network_metrics: VecDeque<NetworkMetrics>,
    quality_decisions: VecDeque<QualityDecision>,
    peer_performance: HashMap<Uuid, PeerPerformanceHistory>,
    bandwidth_samples: VecDeque<BandwidthSample>,
    max_history_size: usize,
}

/// Quality decision record
#[derive(Debug, Clone)]
pub struct QualityDecision {
    pub timestamp: u64,
    pub recommended_quality: QualityLevel,
    pub actual_quality: QualityLevel,
    pub network_conditions: NetworkMetrics,
    pub user_satisfaction: Option<f64>, // Feedback from user behavior
}

/// Peer performance tracking
#[derive(Debug, Clone)]
pub struct PeerPerformanceHistory {
    pub peer_id: Uuid,
    pub connection_successes: u32,
    pub connection_failures: u32,
    pub average_latency: f64,
    pub average_bandwidth: f64,
    pub reliability_score: f64,
    pub last_seen: u64,
    pub geographic_hint: Option<String>,
}

/// Bandwidth measurement sample
#[derive(Debug, Clone)]
pub struct BandwidthSample {
    pub timestamp: u64,
    pub measured_bps: f64,
    pub predicted_bps: f64,
    pub accuracy: f64,
    pub connection_type: ConnectionType,
}

/// ML events for monitoring and debugging
#[derive(Debug, Clone)]
pub enum MLEvent {
    PredictionMade {
        predictor_type: String,
        prediction: f64,
        confidence: f64,
    },
    QualityRecommendation {
        recommended_quality: QualityLevel,
        current_quality: QualityLevel,
        reasoning: String,
    },
    PeerSelected {
        peer_id: Uuid,
        score: f64,
        factors: HashMap<String, f64>,
    },
    AnomalyDetected {
        anomaly_type: String,
        severity: f64,
        description: String,
    },
    ModelUpdated {
        model_type: String,
        accuracy_improvement: f64,
    },
    OptimizationApplied {
        optimization_type: String,
        expected_improvement: f64,
    },
}

/// Feature vector for ML models
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub features: DVector<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: u64,
}

/// ML model performance metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub mean_absolute_error: f64,
    pub last_updated: u64,
    pub sample_count: usize,
}

impl DefianceML {
    /// Create new ML optimization system
    pub fn new(settings: OptimizationSettings) -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            network_predictor: Arc::new(RwLock::new(NetworkPredictor::new()?)),
            quality_optimizer: Arc::new(RwLock::new(QualityOptimizer::new()?)),
            peer_selector: Arc::new(RwLock::new(PeerSelector::new()?)),
            bandwidth_estimator: Arc::new(RwLock::new(BandwidthEstimator::new()?)),
            latency_predictor: Arc::new(RwLock::new(LatencyPredictor::new()?)),
            adaptive_streaming: Arc::new(RwLock::new(AdaptiveStreaming::new()?)),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new()?)),
            
            metrics_history: Arc::new(RwLock::new(MetricsHistory::new())),
            optimization_settings: settings,
            event_sender,
            event_receiver: Some(event_receiver),
        })
    }
    
    /// Initialize the ML system
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize all ML components
        self.network_predictor.write().await.initialize().await?;
        self.quality_optimizer.write().await.initialize().await?;
        self.peer_selector.write().await.initialize().await?;
        self.bandwidth_estimator.write().await.initialize().await?;
        self.latency_predictor.write().await.initialize().await?;
        self.adaptive_streaming.write().await.initialize().await?;
        self.anomaly_detector.write().await.initialize().await?;
        
        tracing::info!("DefianceML system initialized");
        Ok(())
    }
    
    /// Ingest network metrics for ML training
    pub async fn ingest_metrics(&self, metrics: NetworkMetrics) -> Result<()> {
        // Store metrics in history
        {
            let mut history = self.metrics_history.write().await;
            history.add_network_metrics(metrics.clone());
        }
        
        // Update all ML components with new data
        self.network_predictor.write().await.add_sample(metrics.clone()).await?;
        self.bandwidth_estimator.write().await.add_sample(metrics.clone()).await?;
        self.latency_predictor.write().await.add_sample(metrics.clone()).await?;
        
        if self.optimization_settings.enable_anomaly_detection {
            self.anomaly_detector.write().await.analyze_metrics(metrics).await?;
        }
        
        Ok(())
    }
    
    /// Get quality recommendation based on current conditions
    pub async fn get_quality_recommendation(&self, current_metrics: &NetworkMetrics) -> Result<QualityLevel> {
        let quality_optimizer = self.quality_optimizer.read().await;
        let recommendation = quality_optimizer.recommend_quality(current_metrics).await?;
        
        let _ = self.event_sender.send(MLEvent::QualityRecommendation {
            recommended_quality: recommendation.quality.clone(),
            current_quality: recommendation.current_quality.clone(),
            reasoning: recommendation.reasoning.clone(),
        });
        
        Ok(recommendation.quality)
    }
    
    /// Select best peers for content delivery
    pub async fn select_best_peers(&self, available_peers: &[Uuid], target_count: usize) -> Result<Vec<Uuid>> {
        let peer_selector = self.peer_selector.read().await;
        let selected_peers = peer_selector.select_peers(available_peers, target_count).await?;
        
        for peer_id in &selected_peers {
            let score = peer_selector.get_peer_score(peer_id).await.unwrap_or_default();
            let _ = self.event_sender.send(MLEvent::PeerSelected {
                peer_id: *peer_id,
                score: score.total_score,
                factors: score.factors,
            });
        }
        
        Ok(selected_peers)
    }
    
    /// Predict network conditions for the next time window
    pub async fn predict_network_conditions(&self, prediction_window: Duration) -> Result<NetworkPrediction> {
        let predictor = self.network_predictor.read().await;
        let prediction = predictor.predict_conditions(prediction_window).await?;
        
        let _ = self.event_sender.send(MLEvent::PredictionMade {
            predictor_type: "network_conditions".to_string(),
            prediction: prediction.confidence,
            confidence: prediction.confidence,
        });
        
        Ok(prediction)
    }
    
    /// Get adaptive streaming decision
    pub async fn get_streaming_decision(&self, context: &StreamingContext) -> Result<StreamingDecision> {
        let adaptive_streaming = self.adaptive_streaming.read().await;
        adaptive_streaming.make_decision(context).await
    }
    
    /// Update model with feedback
    pub async fn provide_feedback(&self, feedback: MLFeedback) -> Result<()> {
        match feedback {
            MLFeedback::QualityFeedback { decision_id, user_satisfaction, actual_quality } => {
                let mut quality_optimizer = self.quality_optimizer.write().await;
                quality_optimizer.update_with_feedback(decision_id, user_satisfaction, actual_quality).await?;
            }
            MLFeedback::PeerFeedback { peer_id, performance_score, connection_success } => {
                let mut peer_selector = self.peer_selector.write().await;
                peer_selector.update_peer_performance(peer_id, performance_score, connection_success).await?;
            }
            MLFeedback::BandwidthFeedback { prediction_id, actual_bandwidth } => {
                let mut bandwidth_estimator = self.bandwidth_estimator.write().await;
                bandwidth_estimator.update_with_actual(prediction_id, actual_bandwidth).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get model performance metrics
    pub async fn get_model_metrics(&self) -> Result<HashMap<String, ModelMetrics>> {
        let mut metrics = HashMap::new();
        
        metrics.insert("network_predictor".to_string(), 
                      self.network_predictor.read().await.get_metrics().await);
        metrics.insert("quality_optimizer".to_string(), 
                      self.quality_optimizer.read().await.get_metrics().await);
        metrics.insert("peer_selector".to_string(), 
                      self.peer_selector.read().await.get_metrics().await);
        metrics.insert("bandwidth_estimator".to_string(), 
                      self.bandwidth_estimator.read().await.get_metrics().await);
        metrics.insert("latency_predictor".to_string(), 
                      self.latency_predictor.read().await.get_metrics().await);
        
        Ok(metrics)
    }
    
    /// Enable or disable privacy mode
    pub async fn set_privacy_mode(&mut self, enabled: bool) -> Result<()> {
        self.optimization_settings.privacy_mode = enabled;
        
        if enabled {
            // Clear sensitive data and reduce data collection
            let mut history = self.metrics_history.write().await;
            history.anonymize_data();
        }
        
        Ok(())
    }
    
    /// Optimize for battery life
    pub async fn optimize_for_battery(&self, battery_level: f64) -> Result<()> {
        if !self.optimization_settings.battery_optimization {
            return Ok(());
        }
        
        if battery_level < 0.2 {
            // Aggressive battery optimization
            self.quality_optimizer.write().await.set_battery_mode(true).await;
            self.adaptive_streaming.write().await.set_power_saving_mode(true).await;
        } else if battery_level < 0.5 {
            // Moderate battery optimization
            self.quality_optimizer.write().await.set_battery_mode(false).await;
            self.adaptive_streaming.write().await.set_power_saving_mode(false).await;
        }
        
        Ok(())
    }
    
    /// Take event receiver for processing ML events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<MLEvent>> {
        self.event_receiver.take()
    }
}

/// Feedback for improving ML models
#[derive(Debug, Clone)]
pub enum MLFeedback {
    QualityFeedback {
        decision_id: Uuid,
        user_satisfaction: f64,
        actual_quality: QualityLevel,
    },
    PeerFeedback {
        peer_id: Uuid,
        performance_score: f64,
        connection_success: bool,
    },
    BandwidthFeedback {
        prediction_id: Uuid,
        actual_bandwidth: f64,
    },
}

/// Streaming context for adaptive decisions
#[derive(Debug, Clone)]
pub struct StreamingContext {
    pub current_quality: QualityLevel,
    pub buffer_health: f64, // 0.0 to 1.0
    pub bandwidth_utilization: f64,
    pub cpu_usage: f64,
    pub viewer_count: u32,
    pub content_type: String,
    pub priority: StreamingPriority,
}

/// Streaming priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl MetricsHistory {
    /// Create new metrics history
    pub fn new() -> Self {
        Self {
            network_metrics: VecDeque::new(),
            quality_decisions: VecDeque::new(),
            peer_performance: HashMap::new(),
            bandwidth_samples: VecDeque::new(),
            max_history_size: 10000,
        }
    }
    
    /// Add network metrics to history
    pub fn add_network_metrics(&mut self, metrics: NetworkMetrics) {
        self.network_metrics.push_back(metrics);
        
        if self.network_metrics.len() > self.max_history_size {
            self.network_metrics.pop_front();
        }
    }
    
    /// Add quality decision to history
    pub fn add_quality_decision(&mut self, decision: QualityDecision) {
        self.quality_decisions.push_back(decision);
        
        if self.quality_decisions.len() > self.max_history_size {
            self.quality_decisions.pop_front();
        }
    }
    
    /// Get recent metrics for analysis
    pub fn get_recent_metrics(&self, duration: Duration) -> Vec<NetworkMetrics> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - duration.as_secs();
        
        self.network_metrics
            .iter()
            .filter(|m| m.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
    
    /// Anonymize data for privacy
    pub fn anonymize_data(&mut self) {
        // Remove location hints and peer IDs from historical data
        for metrics in &mut self.network_metrics {
            metrics.peer_id = None;
            metrics.location_hint = None;
        }
        
        // Clear peer performance data
        self.peer_performance.clear();
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            prediction_window: Duration::from_secs(60),
            min_samples_for_prediction: 10,
            quality_adaptation_aggressiveness: 0.7,
            peer_selection_diversity: 0.3,
            enable_anomaly_detection: true,
            battery_optimization: true,
            privacy_mode: false,
        }
    }
}

impl NetworkMetrics {
    /// Create feature vector from metrics
    pub fn to_feature_vector(&self) -> FeatureVector {
        let features = vec![
            self.latency_ms,
            self.bandwidth_bps,
            self.packet_loss,
            self.jitter_ms,
            self.connection_quality,
            self.cpu_usage,
            self.memory_usage,
            self.battery_level.unwrap_or(1.0),
            self.connection_type_numeric(),
        ];
        
        let feature_names = vec![
            "latency_ms".to_string(),
            "bandwidth_bps".to_string(),
            "packet_loss".to_string(),
            "jitter_ms".to_string(),
            "connection_quality".to_string(),
            "cpu_usage".to_string(),
            "memory_usage".to_string(),
            "battery_level".to_string(),
            "connection_type".to_string(),
        ];
        
        FeatureVector {
            features: DVector::from_vec(features),
            feature_names,
            timestamp: self.timestamp,
        }
    }
    
    fn connection_type_numeric(&self) -> f64 {
        match self.connection_type {
            ConnectionType::Ethernet => 5.0,
            ConnectionType::WiFi => 4.0,
            ConnectionType::Cellular5G => 3.0,
            ConnectionType::Cellular4G => 2.0,
            ConnectionType::Satellite => 1.0,
            ConnectionType::Unknown => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_defiance_ml_creation() {
        let settings = OptimizationSettings::default();
        let ml_system = DefianceML::new(settings);
        assert!(ml_system.is_ok());
    }
    
    #[test]
    fn test_network_metrics_feature_vector() {
        let metrics = NetworkMetrics {
            timestamp: 1234567890,
            peer_id: Some(Uuid::new_v4()),
            latency_ms: 50.0,
            bandwidth_bps: 1000000.0,
            packet_loss: 0.01,
            jitter_ms: 5.0,
            connection_quality: 0.8,
            cpu_usage: 0.3,
            memory_usage: 0.5,
            battery_level: Some(0.7),
            connection_type: ConnectionType::WiFi,
            location_hint: None,
        };
        
        let feature_vector = metrics.to_feature_vector();
        assert_eq!(feature_vector.features.len(), 9);
        assert_eq!(feature_vector.feature_names.len(), 9);
    }
    
    #[test]
    fn test_metrics_history() {
        let mut history = MetricsHistory::new();
        
        let metrics = NetworkMetrics {
            timestamp: 1234567890,
            peer_id: None,
            latency_ms: 50.0,
            bandwidth_bps: 1000000.0,
            packet_loss: 0.01,
            jitter_ms: 5.0,
            connection_quality: 0.8,
            cpu_usage: 0.3,
            memory_usage: 0.5,
            battery_level: Some(0.7),
            connection_type: ConnectionType::WiFi,
            location_hint: None,
        };
        
        history.add_network_metrics(metrics);
        assert_eq!(history.network_metrics.len(), 1);
    }
    
    #[test]
    fn test_optimization_settings_default() {
        let settings = OptimizationSettings::default();
        assert_eq!(settings.learning_rate, 0.01);
        assert!(settings.enable_anomaly_detection);
        assert!(settings.battery_optimization);
        assert!(!settings.privacy_mode);
    }
}