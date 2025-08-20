//! Quality optimization using machine learning for adaptive streaming

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use nalgebra::{DVector, DMatrix};
use crate::{NetworkMetrics, QualityLevel, ModelMetrics, FeatureVector};

/// Quality optimizer using ML to recommend optimal streaming quality
pub struct QualityOptimizer {
    model: QualityPredictionModel,
    decision_history: VecDeque<QualityDecisionRecord>,
    performance_metrics: ModelMetrics,
    battery_mode: bool,
    power_saving_threshold: f64,
}

/// ML model for quality prediction
struct QualityPredictionModel {
    weights: DMatrix<f64>,
    bias: DVector<f64>,
    feature_scaler: FeatureScaler,
    model_type: ModelType,
    is_trained: bool,
}

/// Model type for quality prediction
#[derive(Debug, Clone)]
enum ModelType {
    LinearRegression,
    NeuralNetwork,
    DecisionTree,
    Ensemble,
}

/// Feature scaling for normalization
struct FeatureScaler {
    feature_means: DVector<f64>,
    feature_stds: DVector<f64>,
    feature_mins: DVector<f64>,
    feature_maxs: DVector<f64>,
}

/// Quality recommendation result
#[derive(Debug, Clone)]
pub struct QualityRecommendation {
    pub quality: QualityLevel,
    pub current_quality: QualityLevel,
    pub confidence: f64,
    pub reasoning: String,
    pub expected_metrics: ExpectedMetrics,
    pub decision_id: Uuid,
}

/// Expected streaming metrics for a quality level
#[derive(Debug, Clone)]
pub struct ExpectedMetrics {
    pub bandwidth_requirement: f64,
    pub cpu_usage: f64,
    pub buffer_stability: f64,
    pub user_satisfaction_prediction: f64,
}

/// Quality decision record for learning
#[derive(Debug, Clone)]
struct QualityDecisionRecord {
    decision_id: Uuid,
    timestamp: u64,
    network_metrics: NetworkMetrics,
    recommended_quality: QualityLevel,
    actual_quality: QualityLevel,
    user_feedback: Option<f64>,
    streaming_success: bool,
    buffer_health: f64,
    actual_bandwidth_used: f64,
}

/// Quality adaptation strategy
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    Conservative,   // Prioritize stability
    Aggressive,     // Prioritize highest quality
    Balanced,       // Balance quality and stability
    BatteryOptimized, // Prioritize power efficiency
}

impl QualityOptimizer {
    /// Create new quality optimizer
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: QualityPredictionModel::new()?,
            decision_history: VecDeque::new(),
            performance_metrics: ModelMetrics::default(),
            battery_mode: false,
            power_saving_threshold: 0.3,
        })
    }
    
    /// Initialize the optimizer
    pub async fn initialize(&mut self) -> Result<()> {
        // Load pre-trained model if available
        self.load_pretrained_model().await?;
        tracing::info!("Quality optimizer initialized");
        Ok(())
    }
    
    /// Recommend optimal quality based on network metrics
    pub async fn recommend_quality(&self, metrics: &NetworkMetrics) -> Result<QualityRecommendation> {
        let feature_vector = self.extract_features(metrics);
        let quality_scores = self.model.predict(&feature_vector)?;
        
        let recommended_quality = self.select_quality_from_scores(&quality_scores, metrics)?;
        let confidence = self.calculate_confidence(&quality_scores, &recommended_quality);
        let reasoning = self.generate_reasoning(metrics, &recommended_quality, &quality_scores);
        let expected_metrics = self.predict_streaming_metrics(&recommended_quality, metrics);
        
        Ok(QualityRecommendation {
            quality: recommended_quality.clone(),
            current_quality: self.estimate_current_quality(metrics),
            confidence,
            reasoning,
            expected_metrics,
            decision_id: Uuid::new_v4(),
        })
    }
    
    /// Update model with feedback
    pub async fn update_with_feedback(
        &mut self,
        decision_id: Uuid,
        user_satisfaction: f64,
        actual_quality: QualityLevel,
    ) -> Result<()> {
        // Find the decision record
        if let Some(record) = self.decision_history.iter_mut().find(|r| r.decision_id == decision_id) {
            record.user_feedback = Some(user_satisfaction);
            record.actual_quality = actual_quality;
            
            // Retrain model with new feedback
            self.retrain_model().await?;
            
            tracing::debug!("Updated quality optimizer with feedback for decision {}", decision_id);
        }
        
        Ok(())
    }
    
    /// Set battery optimization mode
    pub async fn set_battery_mode(&mut self, enabled: bool) {
        self.battery_mode = enabled;
        
        if enabled {
            // Adjust power saving threshold for more aggressive optimization
            self.power_saving_threshold = 0.5;
        } else {
            self.power_saving_threshold = 0.3;
        }
        
        tracing::info!("Battery mode {}", if enabled { "enabled" } else { "disabled" });
    }
    
    /// Extract features from network metrics
    fn extract_features(&self, metrics: &NetworkMetrics) -> FeatureVector {
        let mut features = metrics.to_feature_vector();
        
        // Add derived features
        let bandwidth_mbps = metrics.bandwidth_bps / 1_000_000.0;
        let latency_score = 1.0 / (1.0 + metrics.latency_ms / 100.0);
        let stability_score = 1.0 - metrics.packet_loss - (metrics.jitter_ms / 100.0);
        let resource_usage = (metrics.cpu_usage + metrics.memory_usage) / 2.0;
        
        // Battery consideration
        let battery_factor = if self.battery_mode {
            metrics.battery_level.unwrap_or(1.0)
        } else {
            1.0
        };
        
        // Connection quality based on type and conditions
        let connection_factor = metrics.connection_quality * 
            self.connection_type_reliability(&metrics.connection_type);
        
        // Add these derived features
        features.features = DVector::from_vec(vec![
            features.features[0], // latency_ms
            features.features[1], // bandwidth_bps
            features.features[2], // packet_loss
            features.features[3], // jitter_ms
            features.features[4], // connection_quality
            features.features[5], // cpu_usage
            features.features[6], // memory_usage
            features.features[7], // battery_level
            features.features[8], // connection_type
            bandwidth_mbps,       // derived: bandwidth in Mbps
            latency_score,        // derived: latency quality score
            stability_score,      // derived: connection stability
            resource_usage,       // derived: average resource usage
            battery_factor,       // derived: battery consideration
            connection_factor,    // derived: connection reliability
        ]);
        
        features.feature_names.extend(vec![
            "bandwidth_mbps".to_string(),
            "latency_score".to_string(),
            "stability_score".to_string(),
            "resource_usage".to_string(),
            "battery_factor".to_string(),
            "connection_factor".to_string(),
        ]);
        
        features
    }
    
    /// Select quality level from prediction scores
    fn select_quality_from_scores(
        &self,
        scores: &HashMap<QualityLevel, f64>,
        metrics: &NetworkMetrics,
    ) -> Result<QualityLevel> {
        let mut best_quality = QualityLevel::Low;
        let mut best_score = -1.0;
        
        // Apply adaptation strategy
        let strategy = self.determine_adaptation_strategy(metrics);
        
        for (quality, score) in scores {
            let adjusted_score = self.apply_strategy_adjustment(*score, quality, &strategy, metrics);
            
            if adjusted_score > best_score {
                best_score = adjusted_score;
                best_quality = quality.clone();
            }
        }
        
        // Safety checks
        if !self.is_quality_feasible(&best_quality, metrics) {
            best_quality = self.find_fallback_quality(metrics);
        }
        
        Ok(best_quality)
    }
    
    /// Determine adaptation strategy based on conditions
    fn determine_adaptation_strategy(&self, metrics: &NetworkMetrics) -> AdaptationStrategy {
        if self.battery_mode && metrics.battery_level.unwrap_or(1.0) < self.power_saving_threshold {
            return AdaptationStrategy::BatteryOptimized;
        }
        
        // Check network stability
        let stability = 1.0 - metrics.packet_loss - (metrics.jitter_ms / 100.0);
        let resource_pressure = (metrics.cpu_usage + metrics.memory_usage) / 2.0;
        
        if stability < 0.7 || resource_pressure > 0.8 {
            AdaptationStrategy::Conservative
        } else if stability > 0.9 && resource_pressure < 0.5 && metrics.bandwidth_bps > 5_000_000.0 {
            AdaptationStrategy::Aggressive
        } else {
            AdaptationStrategy::Balanced
        }
    }
    
    /// Apply strategy-specific adjustments to quality scores
    fn apply_strategy_adjustment(
        &self,
        base_score: f64,
        quality: &QualityLevel,
        strategy: &AdaptationStrategy,
        metrics: &NetworkMetrics,
    ) -> f64 {
        let quality_rank = self.quality_to_rank(quality);
        
        match strategy {
            AdaptationStrategy::Conservative => {
                // Penalize higher qualities more
                base_score - (quality_rank as f64 * 0.2)
            }
            AdaptationStrategy::Aggressive => {
                // Bonus for higher qualities if network can handle it
                let bandwidth_bonus = if metrics.bandwidth_bps > self.quality_bandwidth_requirement(quality) * 1.5 {
                    quality_rank as f64 * 0.3
                } else {
                    0.0
                };
                base_score + bandwidth_bonus
            }
            AdaptationStrategy::Balanced => {
                // Slight preference for middle qualities
                let balance_bonus = match quality {
                    QualityLevel::Medium | QualityLevel::High => 0.1,
                    _ => 0.0,
                };
                base_score + balance_bonus
            }
            AdaptationStrategy::BatteryOptimized => {
                // Strong penalty for high-power qualities
                let power_penalty = match quality {
                    QualityLevel::Ultra => -0.8,
                    QualityLevel::High => -0.4,
                    QualityLevel::Medium => -0.1,
                    _ => 0.1,
                };
                base_score + power_penalty
            }
        }
    }
    
    /// Check if quality level is feasible given current conditions
    fn is_quality_feasible(&self, quality: &QualityLevel, metrics: &NetworkMetrics) -> bool {
        let required_bandwidth = self.quality_bandwidth_requirement(quality);
        let available_bandwidth = metrics.bandwidth_bps * 0.8; // 80% utilization safety margin
        
        if available_bandwidth < required_bandwidth {
            return false;
        }
        
        // Check CPU constraints
        let estimated_cpu_usage = metrics.cpu_usage + self.quality_cpu_overhead(quality);
        if estimated_cpu_usage > 0.9 {
            return false;
        }
        
        // Battery constraints
        if self.battery_mode {
            let battery_level = metrics.battery_level.unwrap_or(1.0);
            let power_consumption = self.quality_power_consumption(quality);
            
            if battery_level < 0.2 && power_consumption > 0.5 {
                return false;
            }
        }
        
        true
    }
    
    /// Find fallback quality when preferred quality is not feasible
    fn find_fallback_quality(&self, metrics: &NetworkMetrics) -> QualityLevel {
        let qualities = vec![
            QualityLevel::AudioOnly,
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ];
        
        // Start from lowest quality and find first feasible option
        for quality in qualities {
            if self.is_quality_feasible(&quality, metrics) {
                return quality;
            }
        }
        
        // Fallback to audio-only if nothing else works
        QualityLevel::AudioOnly
    }
    
    /// Calculate confidence in the recommendation
    fn calculate_confidence(&self, scores: &HashMap<QualityLevel, f64>, recommended: &QualityLevel) -> f64 {
        let recommended_score = scores.get(recommended).unwrap_or(&0.0);
        let max_score = scores.values().fold(0.0, |a, &b| a.max(b));
        let min_score = scores.values().fold(1.0, |a, &b| a.min(b));
        
        if max_score == min_score {
            return 0.5; // Uncertain when all scores are equal
        }
        
        // Normalize confidence based on score spread
        let confidence = (recommended_score - min_score) / (max_score - min_score);
        
        // Adjust based on model training quality
        let model_quality_factor = self.performance_metrics.accuracy;
        
        (confidence * model_quality_factor).clamp(0.0, 1.0)
    }
    
    /// Generate human-readable reasoning for the recommendation
    fn generate_reasoning(
        &self,
        metrics: &NetworkMetrics,
        quality: &QualityLevel,
        scores: &HashMap<QualityLevel, f64>,
    ) -> String {
        let mut reasons = Vec::new();
        
        // Bandwidth analysis
        let required_bw = self.quality_bandwidth_requirement(quality);
        let available_bw = metrics.bandwidth_bps;
        let bw_ratio = available_bw / required_bw;
        
        if bw_ratio > 2.0 {
            reasons.push("Excellent bandwidth available".to_string());
        } else if bw_ratio > 1.5 {
            reasons.push("Good bandwidth for this quality".to_string());
        } else if bw_ratio > 1.1 {
            reasons.push("Adequate bandwidth".to_string());
        } else {
            reasons.push("Limited bandwidth constrains quality".to_string());
        }
        
        // Latency analysis
        if metrics.latency_ms < 50.0 {
            reasons.push("low latency connection".to_string());
        } else if metrics.latency_ms < 100.0 {
            reasons.push("moderate latency".to_string());
        } else {
            reasons.push("high latency may affect experience".to_string());
        }
        
        // Stability analysis
        if metrics.packet_loss < 0.01 && metrics.jitter_ms < 10.0 {
            reasons.push("stable connection".to_string());
        } else if metrics.packet_loss > 0.05 || metrics.jitter_ms > 50.0 {
            reasons.push("unstable connection requires lower quality".to_string());
        }
        
        // Battery consideration
        if self.battery_mode {
            if let Some(battery) = metrics.battery_level {
                if battery < 0.2 {
                    reasons.push("low battery requires power optimization".to_string());
                } else if battery < 0.5 {
                    reasons.push("moderate battery usage optimization".to_string());
                }
            }
        }
        
        // Resource usage
        if metrics.cpu_usage > 0.8 || metrics.memory_usage > 0.8 {
            reasons.push("high system resource usage".to_string());
        }
        
        let base_reason = format!("Recommended {} quality based on", quality_to_string(quality));
        format!("{}: {}", base_reason, reasons.join(", "))
    }
    
    /// Predict streaming metrics for a quality level
    fn predict_streaming_metrics(&self, quality: &QualityLevel, metrics: &NetworkMetrics) -> ExpectedMetrics {
        ExpectedMetrics {
            bandwidth_requirement: self.quality_bandwidth_requirement(quality),
            cpu_usage: metrics.cpu_usage + self.quality_cpu_overhead(quality),
            buffer_stability: self.predict_buffer_stability(quality, metrics),
            user_satisfaction_prediction: self.predict_user_satisfaction(quality, metrics),
        }
    }
    
    /// Get bandwidth requirement for quality level
    fn quality_bandwidth_requirement(&self, quality: &QualityLevel) -> f64 {
        match quality {
            QualityLevel::AudioOnly => 128_000.0,      // 128 Kbps
            QualityLevel::Low => 1_000_000.0,          // 1 Mbps
            QualityLevel::Medium => 3_000_000.0,       // 3 Mbps
            QualityLevel::High => 6_000_000.0,         // 6 Mbps
            QualityLevel::Ultra => 25_000_000.0,       // 25 Mbps
        }
    }
    
    /// Get CPU overhead for quality level
    fn quality_cpu_overhead(&self, quality: &QualityLevel) -> f64 {
        match quality {
            QualityLevel::AudioOnly => 0.05,
            QualityLevel::Low => 0.1,
            QualityLevel::Medium => 0.2,
            QualityLevel::High => 0.35,
            QualityLevel::Ultra => 0.6,
        }
    }
    
    /// Get power consumption estimate for quality level
    fn quality_power_consumption(&self, quality: &QualityLevel) -> f64 {
        match quality {
            QualityLevel::AudioOnly => 0.1,
            QualityLevel::Low => 0.2,
            QualityLevel::Medium => 0.4,
            QualityLevel::High => 0.7,
            QualityLevel::Ultra => 1.0,
        }
    }
    
    /// Convert quality to numeric rank for calculations
    fn quality_to_rank(&self, quality: &QualityLevel) -> u8 {
        match quality {
            QualityLevel::AudioOnly => 0,
            QualityLevel::Low => 1,
            QualityLevel::Medium => 2,
            QualityLevel::High => 3,
            QualityLevel::Ultra => 4,
        }
    }
    
    /// Get connection type reliability factor
    fn connection_type_reliability(&self, connection_type: &crate::ConnectionType) -> f64 {
        match connection_type {
            crate::ConnectionType::Ethernet => 1.0,
            crate::ConnectionType::WiFi => 0.9,
            crate::ConnectionType::Cellular5G => 0.8,
            crate::ConnectionType::Cellular4G => 0.7,
            crate::ConnectionType::Satellite => 0.6,
            crate::ConnectionType::Unknown => 0.5,
        }
    }
    
    /// Estimate current quality from metrics
    fn estimate_current_quality(&self, metrics: &NetworkMetrics) -> QualityLevel {
        // Simple heuristic based on bandwidth usage
        let bandwidth_mbps = metrics.bandwidth_bps / 1_000_000.0;
        
        if bandwidth_mbps < 0.5 {
            QualityLevel::AudioOnly
        } else if bandwidth_mbps < 2.0 {
            QualityLevel::Low
        } else if bandwidth_mbps < 5.0 {
            QualityLevel::Medium
        } else if bandwidth_mbps < 15.0 {
            QualityLevel::High
        } else {
            QualityLevel::Ultra
        }
    }
    
    /// Predict buffer stability for quality/metrics combination
    fn predict_buffer_stability(&self, quality: &QualityLevel, metrics: &NetworkMetrics) -> f64 {
        let required_bw = self.quality_bandwidth_requirement(quality);
        let available_bw = metrics.bandwidth_bps;
        let bw_margin = (available_bw - required_bw) / required_bw;
        
        let stability_base = 1.0 - metrics.packet_loss - (metrics.jitter_ms / 1000.0);
        let stability_adjusted = stability_base * (1.0 + bw_margin * 0.5);
        
        stability_adjusted.clamp(0.0, 1.0)
    }
    
    /// Predict user satisfaction for quality/metrics combination
    fn predict_user_satisfaction(&self, quality: &QualityLevel, metrics: &NetworkMetrics) -> f64 {
        let quality_preference = self.quality_to_rank(quality) as f64 / 4.0; // Normalize to 0-1
        let stability = self.predict_buffer_stability(quality, metrics);
        let responsiveness = 1.0 / (1.0 + metrics.latency_ms / 100.0);
        
        // Weighted combination
        (quality_preference * 0.4 + stability * 0.4 + responsiveness * 0.2).clamp(0.0, 1.0)
    }
    
    /// Retrain the model with accumulated feedback
    async fn retrain_model(&mut self) -> Result<()> {
        if self.decision_history.len() < 10 {
            return Ok(()); // Need minimum samples
        }
        
        // Extract training data from decision history
        let (features, labels) = self.prepare_training_data();
        
        // Retrain the model (simplified implementation)
        self.model.train(&features, &labels)?;
        
        // Update performance metrics
        self.update_performance_metrics(&features, &labels).await;
        
        tracing::info!("Retrained quality optimization model");
        Ok(())
    }
    
    /// Prepare training data from decision history
    fn prepare_training_data(&self) -> (Vec<FeatureVector>, Vec<f64>) {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for record in &self.decision_history {
            if let Some(feedback) = record.user_feedback {
                let feature_vector = self.extract_features(&record.network_metrics);
                features.push(feature_vector);
                labels.push(feedback);
            }
        }
        
        (features, labels)
    }
    
    /// Update model performance metrics
    async fn update_performance_metrics(&mut self, features: &[FeatureVector], labels: &[f64]) {
        // Calculate accuracy, precision, recall, etc.
        // Simplified implementation
        if !features.is_empty() {
            let mut total_error = 0.0;
            let mut correct_predictions = 0;
            
            for (i, feature) in features.iter().enumerate() {
                if let Ok(scores) = self.model.predict(feature) {
                    let predicted_quality = self.select_quality_from_scores(&scores, 
                        &self.decision_history[i].network_metrics).unwrap_or(QualityLevel::Medium);
                    let actual_satisfaction = labels[i];
                    let predicted_satisfaction = self.predict_user_satisfaction(
                        &predicted_quality, &self.decision_history[i].network_metrics);
                    
                    total_error += (predicted_satisfaction - actual_satisfaction).abs();
                    
                    if (predicted_satisfaction - actual_satisfaction).abs() < 0.2 {
                        correct_predictions += 1;
                    }
                }
            }
            
            self.performance_metrics.accuracy = correct_predictions as f64 / features.len() as f64;
            self.performance_metrics.mean_absolute_error = total_error / features.len() as f64;
            self.performance_metrics.sample_count = features.len();
            self.performance_metrics.last_updated = 
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        }
    }
    
    /// Load pre-trained model
    async fn load_pretrained_model(&mut self) -> Result<()> {
        // In a real implementation, this would load from file
        // For now, just initialize with reasonable defaults
        self.model.initialize_defaults()?;
        Ok(())
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.performance_metrics.clone()
    }
}

impl QualityPredictionModel {
    fn new() -> Result<Self> {
        Ok(Self {
            weights: DMatrix::zeros(5, 15), // 5 quality levels, 15 features
            bias: DVector::zeros(5),
            feature_scaler: FeatureScaler::new(),
            model_type: ModelType::LinearRegression,
            is_trained: false,
        })
    }
    
    fn predict(&self, features: &FeatureVector) -> Result<HashMap<QualityLevel, f64>> {
        if !self.is_trained {
            return self.default_prediction();
        }
        
        let scaled_features = self.feature_scaler.scale(&features.features);
        let raw_scores = &self.weights * &scaled_features + &self.bias;
        
        let mut scores = HashMap::new();
        scores.insert(QualityLevel::AudioOnly, raw_scores[0]);
        scores.insert(QualityLevel::Low, raw_scores[1]);
        scores.insert(QualityLevel::Medium, raw_scores[2]);
        scores.insert(QualityLevel::High, raw_scores[3]);
        scores.insert(QualityLevel::Ultra, raw_scores[4]);
        
        // Apply softmax for probability distribution
        let exp_scores: Vec<f64> = raw_scores.iter().map(|&x| x.exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        
        for (_, score) in scores.iter_mut() {
            *score = score.exp() / sum_exp;
        }
        
        Ok(scores)
    }
    
    fn train(&mut self, features: &[FeatureVector], labels: &[f64]) -> Result<()> {
        // Simplified training implementation
        // In practice, would use proper ML algorithms
        if features.is_empty() {
            return Ok(());
        }
        
        // Update feature scaler
        self.feature_scaler.fit(features)?;
        
        // Simple gradient descent (simplified)
        // This is a placeholder - real implementation would use proper ML library
        self.is_trained = true;
        
        Ok(())
    }
    
    fn default_prediction(&self) -> Result<HashMap<QualityLevel, f64>> {
        // Default scores when model isn't trained
        let mut scores = HashMap::new();
        scores.insert(QualityLevel::AudioOnly, 0.1);
        scores.insert(QualityLevel::Low, 0.2);
        scores.insert(QualityLevel::Medium, 0.4);
        scores.insert(QualityLevel::High, 0.2);
        scores.insert(QualityLevel::Ultra, 0.1);
        Ok(scores)
    }
    
    fn initialize_defaults(&mut self) -> Result<()> {
        // Initialize with reasonable default weights
        self.weights = DMatrix::from_fn(5, 15, |i, j| {
            (i as f64 + 1.0) * 0.1 + (j as f64) * 0.01
        });
        
        self.bias = DVector::from_fn(5, |i, _| (i as f64) * 0.1);
        self.is_trained = true;
        
        Ok(())
    }
}

impl FeatureScaler {
    fn new() -> Self {
        Self {
            feature_means: DVector::zeros(15),
            feature_stds: DVector::ones(15),
            feature_mins: DVector::zeros(15),
            feature_maxs: DVector::ones(15),
        }
    }
    
    fn fit(&mut self, features: &[FeatureVector]) -> Result<()> {
        if features.is_empty() {
            return Ok(());
        }
        
        let n_features = features[0].features.len();
        let n_samples = features.len();
        
        // Calculate means
        self.feature_means = DVector::zeros(n_features);
        for feature_vec in features {
            self.feature_means += &feature_vec.features;
        }
        self.feature_means /= n_samples as f64;
        
        // Calculate standard deviations
        self.feature_stds = DVector::zeros(n_features);
        for feature_vec in features {
            let diff = &feature_vec.features - &self.feature_means;
            self.feature_stds += diff.component_mul(&diff);
        }
        self.feature_stds /= n_samples as f64;
        self.feature_stds = self.feature_stds.map(|x| x.sqrt().max(1e-8));
        
        Ok(())
    }
    
    fn scale(&self, features: &DVector<f64>) -> DVector<f64> {
        (features - &self.feature_means).component_div(&self.feature_stds)
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.5,
            precision: 0.5,
            recall: 0.5,
            f1_score: 0.5,
            mean_absolute_error: 0.5,
            last_updated: 0,
            sample_count: 0,
        }
    }
}

fn quality_to_string(quality: &QualityLevel) -> &'static str {
    match quality {
        QualityLevel::AudioOnly => "audio-only",
        QualityLevel::Low => "low (480p)",
        QualityLevel::Medium => "medium (720p)",
        QualityLevel::High => "high (1080p)",
        QualityLevel::Ultra => "ultra (4K)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConnectionType;
    
    #[tokio::test]
    async fn test_quality_optimizer_creation() {
        let optimizer = QualityOptimizer::new();
        assert!(optimizer.is_ok());
    }
    
    #[tokio::test]
    async fn test_quality_recommendation() {
        let mut optimizer = QualityOptimizer::new().unwrap();
        optimizer.initialize().await.unwrap();
        
        let metrics = NetworkMetrics {
            timestamp: 1234567890,
            peer_id: None,
            latency_ms: 50.0,
            bandwidth_bps: 5_000_000.0,
            packet_loss: 0.01,
            jitter_ms: 10.0,
            connection_quality: 0.8,
            cpu_usage: 0.3,
            memory_usage: 0.4,
            battery_level: Some(0.8),
            connection_type: ConnectionType::WiFi,
            location_hint: None,
        };
        
        let recommendation = optimizer.recommend_quality(&metrics).await.unwrap();
        assert!(matches!(recommendation.quality, QualityLevel::Medium | QualityLevel::High));
        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.confidence <= 1.0);
    }
    
    #[test]
    fn test_bandwidth_requirements() {
        let optimizer = QualityOptimizer::new().unwrap();
        
        assert_eq!(optimizer.quality_bandwidth_requirement(&QualityLevel::AudioOnly), 128_000.0);
        assert_eq!(optimizer.quality_bandwidth_requirement(&QualityLevel::Low), 1_000_000.0);
        assert_eq!(optimizer.quality_bandwidth_requirement(&QualityLevel::Medium), 3_000_000.0);
        assert_eq!(optimizer.quality_bandwidth_requirement(&QualityLevel::High), 6_000_000.0);
        assert_eq!(optimizer.quality_bandwidth_requirement(&QualityLevel::Ultra), 25_000_000.0);
    }
    
    #[test]
    fn test_feasibility_check() {
        let optimizer = QualityOptimizer::new().unwrap();
        
        let good_metrics = NetworkMetrics {
            timestamp: 1234567890,
            peer_id: None,
            latency_ms: 30.0,
            bandwidth_bps: 10_000_000.0,
            packet_loss: 0.001,
            jitter_ms: 5.0,
            connection_quality: 0.9,
            cpu_usage: 0.2,
            memory_usage: 0.3,
            battery_level: Some(0.8),
            connection_type: ConnectionType::WiFi,
            location_hint: None,
        };
        
        assert!(optimizer.is_quality_feasible(&QualityLevel::High, &good_metrics));
        
        let poor_metrics = NetworkMetrics {
            bandwidth_bps: 500_000.0, // Low bandwidth
            cpu_usage: 0.95,         // High CPU
            ..good_metrics
        };
        
        assert!(!optimizer.is_quality_feasible(&QualityLevel::Ultra, &poor_metrics));
    }
}