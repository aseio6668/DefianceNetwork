//! Latency prediction and network delay optimization using ML

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use nalgebra::DVector;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::linalg::basic::matrix::DenseMatrix;
use statrs::statistics::Statistics;

use crate::{NetworkMetrics, ModelMetrics, ConnectionType};

/// Latency predictor using multiple prediction techniques
pub struct LatencyPredictor {
    model: Option<LinearRegression<f64, DenseMatrix<f64>>>,
    latency_samples: VecDeque<LatencySample>,
    prediction_history: VecDeque<LatencyPredictionRecord>,
    trend_analyzer: TrendAnalyzer,
    pattern_detector: PatternDetector,
    metrics: ModelMetrics,
    min_samples_for_training: usize,
    max_samples: usize,
}

/// Latency prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPrediction {
    pub prediction_id: Uuid,
    pub predicted_latency_ms: f64,
    pub confidence_interval: (f64, f64),
    pub prediction_method: PredictionMethod,
    pub time_horizon: Duration,
    pub contributing_factors: LatencyFactors,
    pub quality_impact: QualityImpact,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Factors contributing to latency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyFactors {
    pub network_congestion: f64,
    pub geographic_distance: f64,
    pub connection_quality: f64,
    pub routing_efficiency: f64,
    pub device_processing_delay: f64,
    pub protocol_overhead: f64,
    pub time_of_day_impact: f64,
}

/// Impact of latency on streaming quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityImpact {
    pub user_experience_score: f64,
    pub buffering_risk: f64,
    pub interaction_delay: f64,
    pub adaptive_streaming_impact: f64,
    pub real_time_suitability: RealTimeSuitability,
}

/// Real-time application suitability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealTimeSuitability {
    Excellent,  // < 20ms
    Good,       // 20-50ms
    Acceptable, // 50-100ms
    Poor,       // 100-200ms
    Unacceptable, // > 200ms
}

/// Optimization suggestions for reducing latency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub description: String,
    pub expected_improvement_ms: f64,
    pub implementation_effort: ImplementationEffort,
    pub priority: Priority,
}

/// Types of optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    PeerSelection,
    RoutingOptimization,
    ProtocolTuning,
    BufferOptimization,
    QualityAdjustment,
    NetworkPathChange,
    LocalCaching,
}

/// Implementation effort required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Immediate,  // Can be applied instantly
    Quick,      // < 1 second
    Moderate,   // 1-10 seconds
    Slow,       // > 10 seconds
}

/// Optimization priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Latency sample for analysis
#[derive(Debug, Clone)]
struct LatencySample {
    timestamp: u64,
    latency_ms: f64,
    peer_id: Option<Uuid>,
    connection_type: ConnectionType,
    packet_size: u32,
    hop_count: Option<u8>,
    jitter_ms: f64,
    packet_loss: f64,
    bandwidth_utilization: f64,
    time_of_day: u8,
    day_of_week: u8,
    geographic_hint: Option<String>,
    measurement_method: MeasurementMethod,
}

/// Method used for latency measurement
#[derive(Debug, Clone, PartialEq)]
enum MeasurementMethod {
    ICMP,         // Ping
    TCP,          // TCP connect time
    UDP,          // UDP round-trip
    ApplicationLayer, // End-to-end application latency
    WebRTC,       // WebRTC connection stats
}

/// Prediction method used
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PredictionMethod {
    MachineLearning,
    TrendAnalysis,
    PatternMatching,
    StatisticalModel,
    HybridModel,
}

/// Trend analysis for latency patterns
#[derive(Debug, Clone)]
struct TrendAnalyzer {
    short_term_trend: f64,
    medium_term_trend: f64,
    long_term_trend: f64,
    trend_confidence: f64,
    trend_samples: VecDeque<f64>,
}

/// Pattern detection for cyclical latency behavior
#[derive(Debug, Clone)]
struct PatternDetector {
    hourly_patterns: Vec<f64>,      // 24 hour pattern
    daily_patterns: Vec<f64>,       // 7 day pattern
    weekly_patterns: Vec<f64>,      // 4 week pattern
    pattern_strength: f64,
    pattern_confidence: f64,
    last_pattern_update: u64,
}

/// Latency prediction record for accuracy tracking
#[derive(Debug, Clone)]
struct LatencyPredictionRecord {
    prediction_id: Uuid,
    timestamp: u64,
    predicted_latency: f64,
    actual_latency: Option<f64>,
    method_used: PredictionMethod,
    accuracy: Option<f64>,
    confidence: f64,
}

impl LatencyPredictor {
    /// Create new latency predictor
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: None,
            latency_samples: VecDeque::new(),
            prediction_history: VecDeque::new(),
            trend_analyzer: TrendAnalyzer::new(),
            pattern_detector: PatternDetector::new(),
            metrics: ModelMetrics::new(),
            min_samples_for_training: 50,
            max_samples: 3000,
        })
    }
    
    /// Initialize the latency predictor
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Latency predictor initialized");
        Ok(())
    }
    
    /// Add a latency measurement sample
    pub async fn add_sample(&mut self, metrics: NetworkMetrics) -> Result<()> {
        let sample = LatencySample {
            timestamp: metrics.timestamp,
            latency_ms: metrics.latency_ms,
            peer_id: metrics.peer_id,
            connection_type: metrics.connection_type.clone(),
            packet_size: 1400, // Default MTU size
            hop_count: None,   // Would be populated by traceroute
            jitter_ms: metrics.jitter_ms,
            packet_loss: metrics.packet_loss,
            bandwidth_utilization: 0.0, // Would be calculated
            time_of_day: self.get_hour_of_day(metrics.timestamp),
            day_of_week: self.get_day_of_week(metrics.timestamp),
            geographic_hint: metrics.location_hint.clone(),
            measurement_method: MeasurementMethod::ApplicationLayer,
        };
        
        self.latency_samples.push_back(sample.clone());
        
        // Limit sample size
        if self.latency_samples.len() > self.max_samples {
            self.latency_samples.pop_front();
        }
        
        // Update analyzers
        self.update_trend_analyzer(&sample);
        self.update_pattern_detector(&sample);
        
        // Retrain ML model if we have enough samples
        if self.latency_samples.len() >= self.min_samples_for_training {
            self.retrain_model().await?;
        }
        
        Ok(())
    }
    
    /// Predict latency for the specified time horizon
    pub async fn predict_latency(&self, time_horizon: Duration) -> Result<LatencyPrediction> {
        let prediction_id = Uuid::new_v4();
        
        // Try different prediction methods
        let predictions = vec![
            self.ml_prediction(time_horizon, prediction_id).await,
            self.trend_prediction(time_horizon, prediction_id),
            self.pattern_prediction(time_horizon, prediction_id),
            self.statistical_prediction(time_horizon, prediction_id),
        ];
        
        // Select best prediction based on confidence and historical accuracy
        let best_prediction = predictions
            .into_iter()
            .filter_map(|p| p.ok())
            .max_by(|a, b| {
                let score_a = self.calculate_prediction_score(a);
                let score_b = self.calculate_prediction_score(b);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap_or_else(|| self.fallback_prediction(time_horizon, prediction_id));
        
        Ok(best_prediction)
    }
    
    /// Update prediction with actual measured latency
    pub async fn update_with_actual(&mut self, prediction_id: Uuid, actual_latency: f64) -> Result<()> {
        if let Some(record) = self.prediction_history.iter_mut().find(|r| r.prediction_id == prediction_id) {
            record.actual_latency = Some(actual_latency);
            record.accuracy = Some(self.calculate_prediction_accuracy(record.predicted_latency, actual_latency));
            
            // Update overall metrics
            self.update_model_metrics().await;
        }
        
        Ok(())
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
    
    /// Machine learning based prediction
    async fn ml_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> Result<LatencyPrediction> {
        if let Some(ref model) = self.model {
            if let Some(latest_sample) = self.latency_samples.back() {
                let features = self.extract_features(latest_sample, time_horizon);
                let x_matrix = DenseMatrix::from_2d_vec(&vec![features]);
                let predictions = model.predict(&x_matrix)?;
                
                let predicted_latency = predictions[0].max(0.0);
                
                return Ok(self.create_prediction_result(
                    prediction_id,
                    predicted_latency,
                    PredictionMethod::MachineLearning,
                    time_horizon,
                    0.8, // ML confidence
                ));
            }
        }
        
        Err(anyhow::anyhow!("ML model not available"))
    }
    
    /// Trend-based prediction
    fn trend_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> Result<LatencyPrediction> {
        if let Some(latest_sample) = self.latency_samples.back() {
            let base_latency = latest_sample.latency_ms;
            let trend_adjustment = self.trend_analyzer.short_term_trend * time_horizon.as_secs_f64() / 60.0;
            let predicted_latency = (base_latency + trend_adjustment).max(0.0);
            
            Ok(self.create_prediction_result(
                prediction_id,
                predicted_latency,
                PredictionMethod::TrendAnalysis,
                time_horizon,
                self.trend_analyzer.trend_confidence,
            ))
        } else {
            Err(anyhow::anyhow!("No samples available for trend prediction"))
        }
    }
    
    /// Pattern-based prediction
    fn pattern_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> Result<LatencyPrediction> {
        if let Some(latest_sample) = self.latency_samples.back() {
            let future_timestamp = latest_sample.timestamp + time_horizon.as_secs();
            let pattern_factor = self.pattern_detector.get_pattern_factor(future_timestamp);
            let base_latency = self.calculate_baseline_latency();
            let predicted_latency = base_latency * pattern_factor;
            
            Ok(self.create_prediction_result(
                prediction_id,
                predicted_latency,
                PredictionMethod::PatternMatching,
                time_horizon,
                self.pattern_detector.pattern_confidence,
            ))
        } else {
            Err(anyhow::anyhow!("No samples available for pattern prediction"))
        }
    }
    
    /// Statistical model prediction
    fn statistical_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> Result<LatencyPrediction> {
        if self.latency_samples.len() < 5 {
            return Err(anyhow::anyhow!("Insufficient samples for statistical prediction"));
        }
        
        let recent_latencies: Vec<f64> = self.latency_samples
            .iter()
            .rev()
            .take(20)
            .map(|s| s.latency_ms)
            .collect();
        
        let mean = recent_latencies.iter().sum::<f64>() / recent_latencies.len() as f64;
        let std_dev = (recent_latencies.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_latencies.len() as f64).sqrt();
        
        // Simple statistical prediction with time decay
        let time_factor = 1.0 + (time_horizon.as_secs_f64() / 3600.0) * 0.1; // Small increase over time
        let predicted_latency = mean * time_factor;
        
        Ok(self.create_prediction_result(
            prediction_id,
            predicted_latency,
            PredictionMethod::StatisticalModel,
            time_horizon,
            0.6, // Statistical confidence
        ))
    }
    
    /// Fallback prediction when other methods fail
    fn fallback_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> LatencyPrediction {
        let default_latency = if let Some(latest) = self.latency_samples.back() {
            latest.latency_ms
        } else {
            50.0 // Default 50ms
        };
        
        self.create_prediction_result(
            prediction_id,
            default_latency,
            PredictionMethod::StatisticalModel,
            time_horizon,
            0.3, // Low confidence for fallback
        )
    }
    
    /// Create prediction result with all details
    fn create_prediction_result(
        &self,
        prediction_id: Uuid,
        predicted_latency: f64,
        method: PredictionMethod,
        time_horizon: Duration,
        confidence: f64,
    ) -> LatencyPrediction {
        let confidence_interval = self.calculate_confidence_interval(predicted_latency, confidence);
        let contributing_factors = self.analyze_latency_factors();
        let quality_impact = self.assess_quality_impact(predicted_latency);
        let optimization_suggestions = self.generate_optimization_suggestions(predicted_latency);
        
        // Record this prediction
        let record = LatencyPredictionRecord {
            prediction_id,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            predicted_latency,
            actual_latency: None,
            method_used: method.clone(),
            accuracy: None,
            confidence,
        };
        
        // Note: In a mutable context, we would add this record to prediction_history
        
        LatencyPrediction {
            prediction_id,
            predicted_latency_ms: predicted_latency,
            confidence_interval,
            prediction_method: method,
            time_horizon,
            contributing_factors,
            quality_impact,
            optimization_suggestions,
        }
    }
    
    /// Extract features for ML model
    fn extract_features(&self, sample: &LatencySample, time_horizon: Duration) -> Vec<f64> {
        vec![
            sample.latency_ms,
            sample.jitter_ms,
            sample.packet_loss,
            sample.bandwidth_utilization,
            sample.connection_type_numeric(),
            sample.time_of_day as f64 / 24.0,
            sample.day_of_week as f64 / 7.0,
            time_horizon.as_secs_f64() / 3600.0,
            self.trend_analyzer.short_term_trend,
            self.pattern_detector.get_current_pattern_factor(),
            sample.hop_count.unwrap_or(10) as f64,
            sample.packet_size as f64 / 1500.0,
        ]
    }
    
    /// Update trend analyzer with new sample
    fn update_trend_analyzer(&mut self, sample: &LatencySample) {
        self.trend_analyzer.add_sample(sample.latency_ms);
    }
    
    /// Update pattern detector with new sample
    fn update_pattern_detector(&mut self, sample: &LatencySample) {
        self.pattern_detector.add_sample(sample.timestamp, sample.latency_ms);
    }
    
    /// Calculate confidence interval for prediction
    fn calculate_confidence_interval(&self, prediction: f64, confidence: f64) -> (f64, f64) {
        let uncertainty = 1.0 - confidence;
        let margin = prediction * uncertainty * 0.3; // 30% margin based on uncertainty
        ((prediction - margin).max(0.0), prediction + margin)
    }
    
    /// Analyze factors contributing to latency
    fn analyze_latency_factors(&self) -> LatencyFactors {
        LatencyFactors {
            network_congestion: self.estimate_network_congestion(),
            geographic_distance: self.estimate_geographic_impact(),
            connection_quality: self.estimate_connection_quality(),
            routing_efficiency: self.estimate_routing_efficiency(),
            device_processing_delay: self.estimate_device_delay(),
            protocol_overhead: self.estimate_protocol_overhead(),
            time_of_day_impact: self.estimate_time_of_day_impact(),
        }
    }
    
    /// Assess impact of latency on quality
    fn assess_quality_impact(&self, latency_ms: f64) -> QualityImpact {
        let user_experience_score = match latency_ms {
            l if l < 20.0 => 1.0,
            l if l < 50.0 => 0.9,
            l if l < 100.0 => 0.7,
            l if l < 200.0 => 0.5,
            _ => 0.2,
        };
        
        let buffering_risk = (latency_ms / 1000.0).min(1.0);
        let interaction_delay = (latency_ms / 100.0).min(1.0);
        let adaptive_streaming_impact = (latency_ms / 500.0).min(1.0);
        
        let real_time_suitability = match latency_ms {
            l if l < 20.0 => RealTimeSuitability::Excellent,
            l if l < 50.0 => RealTimeSuitability::Good,
            l if l < 100.0 => RealTimeSuitability::Acceptable,
            l if l < 200.0 => RealTimeSuitability::Poor,
            _ => RealTimeSuitability::Unacceptable,
        };
        
        QualityImpact {
            user_experience_score,
            buffering_risk,
            interaction_delay,
            adaptive_streaming_impact,
            real_time_suitability,
        }
    }
    
    /// Generate optimization suggestions
    fn generate_optimization_suggestions(&self, predicted_latency: f64) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        
        if predicted_latency > 100.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::PeerSelection,
                description: "Select peers with lower latency paths".to_string(),
                expected_improvement_ms: predicted_latency * 0.3,
                implementation_effort: ImplementationEffort::Quick,
                priority: Priority::High,
            });
        }
        
        if predicted_latency > 50.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::QualityAdjustment,
                description: "Reduce streaming quality to compensate for latency".to_string(),
                expected_improvement_ms: 0.0, // Doesn't reduce latency but improves experience
                implementation_effort: ImplementationEffort::Immediate,
                priority: Priority::Medium,
            });
        }
        
        if self.estimate_network_congestion() > 0.7 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::RoutingOptimization,
                description: "Find alternative network routes to avoid congestion".to_string(),
                expected_improvement_ms: predicted_latency * 0.2,
                implementation_effort: ImplementationEffort::Moderate,
                priority: Priority::High,
            });
        }
        
        suggestions
    }
    
    /// Retrain ML model
    async fn retrain_model(&mut self) -> Result<()> {
        if self.latency_samples.len() < self.min_samples_for_training {
            return Ok(());
        }
        
        let (features, targets) = self.prepare_training_data();
        
        if features.is_empty() || targets.is_empty() {
            return Ok(());
        }
        
        let x_matrix = DenseMatrix::from_2d_vec(&features);
        let y_vector = targets;
        
        let model = LinearRegression::fit(&x_matrix, &y_vector, Default::default())?;
        
        // Evaluate model
        let predictions = model.predict(&x_matrix)?;
        let mae = self.calculate_mean_absolute_error(&y_vector, &predictions);
        
        self.metrics.mean_absolute_error = mae;
        self.metrics.accuracy = 1.0 - (mae / self.calculate_target_mean());
        self.metrics.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.metrics.sample_count = self.latency_samples.len();
        
        self.model = Some(model);
        
        tracing::info!(
            "Latency predictor retrained: accuracy={:.3}, samples={}", 
            self.metrics.accuracy, 
            self.metrics.sample_count
        );
        
        Ok(())
    }
    
    /// Prepare training data for ML model
    fn prepare_training_data(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for (i, sample) in self.latency_samples.iter().enumerate() {
            if i == 0 {
                continue;
            }
            
            let feature_vector = self.extract_features(sample, Duration::from_secs(60));
            features.push(feature_vector);
            targets.push(sample.latency_ms);
        }
        
        (features, targets)
    }
    
    /// Helper methods for estimation
    fn calculate_baseline_latency(&self) -> f64 {
        if self.latency_samples.is_empty() {
            return 50.0;
        }
        
        let recent_samples: Vec<f64> = self.latency_samples
            .iter()
            .rev()
            .take(10)
            .map(|s| s.latency_ms)
            .collect();
        
        recent_samples.iter().sum::<f64>() / recent_samples.len() as f64
    }
    
    fn calculate_prediction_score(&self, prediction: &LatencyPrediction) -> f64 {
        // Combine accuracy score with method confidence
        prediction.quality_impact.user_experience_score * 0.7 + 
        (1.0 - prediction.predicted_latency_ms / 200.0).max(0.0) * 0.3
    }
    
    fn calculate_prediction_accuracy(&self, predicted: f64, actual: f64) -> f64 {
        1.0 - ((predicted - actual).abs() / actual.max(1.0))
    }
    
    fn calculate_mean_absolute_error(&self, actual: &[f64], predicted: &[f64]) -> f64 {
        actual.iter().zip(predicted.iter())
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>() / actual.len() as f64
    }
    
    fn calculate_target_mean(&self) -> f64 {
        if self.latency_samples.is_empty() {
            return 1.0;
        }
        
        let sum: f64 = self.latency_samples.iter().map(|s| s.latency_ms).sum();
        sum / self.latency_samples.len() as f64
    }
    
    // Estimation helper functions
    fn estimate_network_congestion(&self) -> f64 { 0.5 }
    fn estimate_geographic_impact(&self) -> f64 { 0.3 }
    fn estimate_connection_quality(&self) -> f64 { 0.7 }
    fn estimate_routing_efficiency(&self) -> f64 { 0.8 }
    fn estimate_device_delay(&self) -> f64 { 0.1 }
    fn estimate_protocol_overhead(&self) -> f64 { 0.2 }
    fn estimate_time_of_day_impact(&self) -> f64 { 0.1 }
    
    fn get_hour_of_day(&self, timestamp: u64) -> u8 {
        ((timestamp / 3600) % 24) as u8
    }
    
    fn get_day_of_week(&self, timestamp: u64) -> u8 {
        (((timestamp / 86400) + 3) % 7) as u8 // Unix epoch was Thursday
    }
    
    async fn update_model_metrics(&mut self) {
        let recent_predictions: Vec<_> = self.prediction_history
            .iter()
            .filter(|p| p.actual_latency.is_some())
            .collect();
        
        if !recent_predictions.is_empty() {
            let total_accuracy: f64 = recent_predictions
                .iter()
                .map(|p| p.accuracy.unwrap_or(0.0))
                .sum();
            
            self.metrics.accuracy = total_accuracy / recent_predictions.len() as f64;
            self.metrics.sample_count = recent_predictions.len();
        }
    }
}

// Implementation for helper structs
impl TrendAnalyzer {
    fn new() -> Self {
        Self {
            short_term_trend: 0.0,
            medium_term_trend: 0.0,
            long_term_trend: 0.0,
            trend_confidence: 0.0,
            trend_samples: VecDeque::new(),
        }
    }
    
    fn add_sample(&mut self, latency: f64) {
        self.trend_samples.push_back(latency);
        if self.trend_samples.len() > 100 {
            self.trend_samples.pop_front();
        }
        
        self.update_trends();
    }
    
    fn update_trends(&mut self) {
        if self.trend_samples.len() < 5 {
            return;
        }
        
        // Calculate short-term trend (last 5 samples)
        let recent: Vec<f64> = self.trend_samples.iter().rev().take(5).cloned().collect();
        self.short_term_trend = self.calculate_trend(&recent);
        
        // Calculate medium-term trend (last 20 samples)
        if self.trend_samples.len() >= 20 {
            let medium: Vec<f64> = self.trend_samples.iter().rev().take(20).cloned().collect();
            self.medium_term_trend = self.calculate_trend(&medium);
        }
        
        // Update confidence based on trend consistency
        self.trend_confidence = self.calculate_trend_confidence();
    }
    
    fn calculate_trend(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let first_half = &samples[samples.len()/2..];
        let second_half = &samples[..samples.len()/2];
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        second_avg - first_avg
    }
    
    fn calculate_trend_confidence(&self) -> f64 {
        if self.trend_samples.len() < 10 {
            return 0.3;
        }
        
        // Calculate consistency of trend direction
        let consistency = if self.short_term_trend.signum() == self.medium_term_trend.signum() {
            0.8
        } else {
            0.4
        };
        
        consistency
    }
}

impl PatternDetector {
    fn new() -> Self {
        Self {
            hourly_patterns: vec![1.0; 24],
            daily_patterns: vec![1.0; 7],
            weekly_patterns: vec![1.0; 4],
            pattern_strength: 0.1,
            pattern_confidence: 0.5,
            last_pattern_update: 0,
        }
    }
    
    fn add_sample(&mut self, timestamp: u64, latency: f64) {
        let hour = ((timestamp / 3600) % 24) as usize;
        let day = (((timestamp / 86400) + 3) % 7) as usize;
        
        if hour < self.hourly_patterns.len() {
            let alpha = 0.05;
            let baseline = 50.0; // Base latency for normalization
            self.hourly_patterns[hour] = alpha * (latency / baseline) + (1.0 - alpha) * self.hourly_patterns[hour];
        }
        
        if day < self.daily_patterns.len() {
            let alpha = 0.02;
            let baseline = 50.0;
            self.daily_patterns[day] = alpha * (latency / baseline) + (1.0 - alpha) * self.daily_patterns[day];
        }
        
        self.last_pattern_update = timestamp;
        self.update_pattern_confidence();
    }
    
    fn get_pattern_factor(&self, timestamp: u64) -> f64 {
        let hour = ((timestamp / 3600) % 24) as usize;
        let day = (((timestamp / 86400) + 3) % 7) as usize;
        
        let hourly_factor = if hour < self.hourly_patterns.len() {
            self.hourly_patterns[hour]
        } else {
            1.0
        };
        
        let daily_factor = if day < self.daily_patterns.len() {
            self.daily_patterns[day]
        } else {
            1.0
        };
        
        // Weighted combination
        hourly_factor * 0.7 + daily_factor * 0.3
    }
    
    fn get_current_pattern_factor(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.get_pattern_factor(now)
    }
    
    fn update_pattern_confidence(&mut self) {
        // Calculate pattern strength based on variance from baseline
        let variance: f64 = self.hourly_patterns
            .iter()
            .map(|&x| (x - 1.0).powi(2))
            .sum::<f64>() / self.hourly_patterns.len() as f64;
        
        self.pattern_strength = variance.sqrt();
        self.pattern_confidence = (self.pattern_strength * 2.0).min(1.0);
    }
}

impl LatencySample {
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
    async fn test_latency_predictor_creation() {
        let predictor = LatencyPredictor::new();
        assert!(predictor.is_ok());
    }
    
    #[tokio::test]
    async fn test_latency_prediction() {
        let predictor = LatencyPredictor::new().unwrap();
        let prediction = predictor.predict_latency(Duration::from_secs(60)).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.predicted_latency_ms > 0.0);
        assert!(pred.confidence_interval.0 <= pred.predicted_latency_ms);
        assert!(pred.confidence_interval.1 >= pred.predicted_latency_ms);
    }
    
    #[test]
    fn test_trend_analyzer() {
        let mut analyzer = TrendAnalyzer::new();
        analyzer.add_sample(50.0);
        analyzer.add_sample(55.0);
        analyzer.add_sample(60.0);
        analyzer.add_sample(65.0);
        analyzer.add_sample(70.0);
        
        assert!(analyzer.short_term_trend > 0.0); // Should detect upward trend
        assert!(analyzer.trend_confidence > 0.0);
    }
    
    #[test]
    fn test_pattern_detector() {
        let mut detector = PatternDetector::new();
        let base_timestamp = 1640995200; // Jan 1, 2022 00:00:00 UTC
        
        // Add samples for different hours
        for hour in 0..24 {
            let timestamp = base_timestamp + hour * 3600;
            let latency = 50.0 + (hour as f64 * 2.0); // Increasing latency through the day
            detector.add_sample(timestamp, latency);
        }
        
        // Test pattern factor for different times
        let morning_factor = detector.get_pattern_factor(base_timestamp + 8 * 3600); // 8 AM
        let evening_factor = detector.get_pattern_factor(base_timestamp + 20 * 3600); // 8 PM
        
        assert!(evening_factor > morning_factor); // Should detect higher latency in evening
    }
}