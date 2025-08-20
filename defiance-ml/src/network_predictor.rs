//! Network condition prediction using time series analysis and machine learning

use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use nalgebra::{DVector, DMatrix};
// Note: ML features simplified due to smartcore API changes
// use smartcore::linear::linear_regression::LinearRegression;
// use smartcore::linalg::basic::matrix::DenseMatrix;
// use smartcore::metrics::mean_absolute_error;
use statrs::statistics::{Statistics, Distribution};
use statrs::distribution::Normal;

use crate::{NetworkMetrics, ModelMetrics, FeatureVector, ConnectionType};

/// Network condition predictor using ML
pub struct NetworkPredictor {
    // model: Option<LinearRegression<f64, DenseMatrix<f64>>>, // Simplified for now
    training_data: VecDeque<NetworkMetrics>,
    prediction_history: VecDeque<PredictionResult>,
    moving_averages: MovingAverages,
    seasonal_patterns: SeasonalPatterns,
    metrics: ModelMetrics,
    min_samples_for_training: usize,
    max_training_samples: usize,
}

/// Network prediction with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPrediction {
    pub predicted_bandwidth: f64,
    pub predicted_latency: f64,
    pub predicted_packet_loss: f64,
    pub predicted_quality: f64,
    pub confidence: f64,
    pub prediction_window: Duration,
    pub factors: PredictionFactors,
    pub uncertainty_range: UncertaintyRange,
}

/// Factors contributing to the prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFactors {
    pub historical_trend: f64,
    pub seasonal_impact: f64,
    pub connection_type_impact: f64,
    pub time_of_day_impact: f64,
    pub recent_performance: f64,
}

/// Uncertainty range for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyRange {
    pub bandwidth_range: (f64, f64),
    pub latency_range: (f64, f64),
    pub quality_range: (f64, f64),
}

/// Historical prediction result for accuracy tracking
#[derive(Debug, Clone)]
struct PredictionResult {
    timestamp: u64,
    predicted_values: Vec<f64>,
    actual_values: Option<Vec<f64>>,
    prediction_window: Duration,
    accuracy: Option<f64>,
}

/// Moving averages for trend analysis
#[derive(Debug, Clone)]
struct MovingAverages {
    bandwidth_5min: f64,
    bandwidth_15min: f64,
    bandwidth_1hour: f64,
    latency_5min: f64,
    latency_15min: f64,
    latency_1hour: f64,
    quality_5min: f64,
    quality_15min: f64,
    quality_1hour: f64,
}

/// Seasonal pattern detection
#[derive(Debug, Clone)]
struct SeasonalPatterns {
    hourly_bandwidth_pattern: Vec<f64>,
    hourly_latency_pattern: Vec<f64>,
    daily_pattern_strength: f64,
    weekly_pattern_strength: f64,
    last_pattern_update: u64,
}

/// Time-based features for prediction
#[derive(Debug, Clone)]
struct TimeFeatures {
    hour_of_day: f64,
    day_of_week: f64,
    is_weekend: f64,
    is_peak_hours: f64,
    time_since_last_sample: f64,
}

impl NetworkPredictor {
    /// Create new network predictor
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: None,
            training_data: VecDeque::new(),
            prediction_history: VecDeque::new(),
            moving_averages: MovingAverages::new(),
            seasonal_patterns: SeasonalPatterns::new(),
            metrics: ModelMetrics::new(),
            min_samples_for_training: 50,
            max_training_samples: 5000,
        })
    }
    
    /// Initialize the predictor
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Network predictor initialized");
        Ok(())
    }
    
    /// Add a new network sample for training
    pub async fn add_sample(&mut self, metrics: NetworkMetrics) -> Result<()> {
        self.training_data.push_back(metrics.clone());
        
        // Limit training data size
        if self.training_data.len() > self.max_training_samples {
            self.training_data.pop_front();
        }
        
        // Update moving averages
        self.update_moving_averages(&metrics);
        
        // Update seasonal patterns
        self.update_seasonal_patterns(&metrics);
        
        // Retrain model if we have enough samples
        if self.training_data.len() >= self.min_samples_for_training {
            self.retrain_model().await?;
        }
        
        Ok(())
    }
    
    /// Predict network conditions for the specified time window
    pub async fn predict_conditions(&self, prediction_window: Duration) -> Result<NetworkPrediction> {
        if self.training_data.is_empty() {
            return Ok(self.fallback_prediction(prediction_window));
        }
        
        let latest_metrics = self.training_data.back().unwrap();
        let time_features = self.extract_time_features(prediction_window);
        
        // Use ML model if available, otherwise use statistical methods
        let prediction = if let Some(ref model) = self.model {
            self.ml_prediction(model, latest_metrics, &time_features, prediction_window)?
        } else {
            self.statistical_prediction(latest_metrics, &time_features, prediction_window)
        };
        
        Ok(prediction)
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
    
    /// Retrain the ML model with current data
    async fn retrain_model(&mut self) -> Result<()> {
        if self.training_data.len() < self.min_samples_for_training {
            return Ok(());
        }
        
        let (features, targets) = self.prepare_training_data()?;
        
        if features.is_empty() || targets.is_empty() {
            return Ok(());
        }
        
        // Create training matrices
        let x_matrix = DenseMatrix::from_2d_vec(&features);
        let y_vector = targets.clone();
        
        // Train linear regression model
        let model = LinearRegression::fit(&x_matrix, &y_vector, Default::default())?;
        
        // Evaluate model performance
        let predictions = model.predict(&x_matrix)?;
        let mae = mean_absolute_error(&y_vector, &predictions);
        
        // Update metrics
        self.metrics.mean_absolute_error = mae;
        self.metrics.accuracy = 1.0 - (mae / self.calculate_target_mean());
        self.metrics.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.metrics.sample_count = self.training_data.len();
        
        self.model = Some(model);
        
        tracing::info!(
            "Network predictor retrained: accuracy={:.3}, samples={}", 
            self.metrics.accuracy, 
            self.metrics.sample_count
        );
        
        Ok(())
    }
    
    /// Prepare training data for ML model
    fn prepare_training_data(&self) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for (i, metrics) in self.training_data.iter().enumerate() {
            if i == 0 {
                continue; // Skip first sample as we need previous data
            }
            
            let prev_metrics = &self.training_data[i - 1];
            let feature_vector = self.create_feature_vector(prev_metrics, metrics);
            let target = self.create_target_value(metrics);
            
            features.push(feature_vector);
            targets.push(target);
        }
        
        Ok((features, targets))
    }
    
    /// Create feature vector for ML model
    fn create_feature_vector(&self, prev_metrics: &NetworkMetrics, current_metrics: &NetworkMetrics) -> Vec<f64> {
        let time_diff = (current_metrics.timestamp - prev_metrics.timestamp) as f64;
        
        vec![
            prev_metrics.bandwidth_bps,
            prev_metrics.latency_ms,
            prev_metrics.packet_loss,
            prev_metrics.connection_quality,
            prev_metrics.cpu_usage,
            prev_metrics.memory_usage,
            prev_metrics.connection_type_numeric(),
            time_diff,
            self.moving_averages.bandwidth_5min,
            self.moving_averages.latency_5min,
            self.seasonal_patterns.get_current_hour_factor(),
            if self.is_peak_hours(current_metrics.timestamp) { 1.0 } else { 0.0 },
        ]
    }
    
    /// Create target value for training (composite network quality score)
    fn create_target_value(&self, metrics: &NetworkMetrics) -> f64 {
        // Normalize and combine multiple metrics into a single quality score
        let bandwidth_score = (metrics.bandwidth_bps / 10_000_000.0).min(1.0); // Normalize to 10 Mbps
        let latency_score = (200.0 - metrics.latency_ms).max(0.0) / 200.0; // Lower latency is better
        let loss_score = (1.0 - metrics.packet_loss).max(0.0);
        
        // Weighted combination
        0.4 * bandwidth_score + 0.3 * latency_score + 0.3 * loss_score
    }
    
    /// ML-based prediction using trained model
    fn ml_prediction(
        &self,
        model: &LinearRegression<f64, DenseMatrix<f64>>,
        latest_metrics: &NetworkMetrics,
        time_features: &TimeFeatures,
        prediction_window: Duration,
    ) -> Result<NetworkPrediction> {
        let feature_vector = vec![
            latest_metrics.bandwidth_bps,
            latest_metrics.latency_ms,
            latest_metrics.packet_loss,
            latest_metrics.connection_quality,
            latest_metrics.cpu_usage,
            latest_metrics.memory_usage,
            latest_metrics.connection_type_numeric(),
            prediction_window.as_secs() as f64,
            self.moving_averages.bandwidth_5min,
            self.moving_averages.latency_5min,
            self.seasonal_patterns.get_current_hour_factor(),
            time_features.is_peak_hours,
        ];
        
        let x_matrix = DenseMatrix::from_2d_vec(&vec![feature_vector]);
        let predictions = model.predict(&x_matrix)?;
        let predicted_quality = predictions[0];
        
        // Convert quality score back to individual metrics
        let predicted_bandwidth = latest_metrics.bandwidth_bps * (0.8 + 0.4 * predicted_quality);
        let predicted_latency = latest_metrics.latency_ms * (1.2 - 0.4 * predicted_quality);
        let predicted_packet_loss = latest_metrics.packet_loss * (1.1 - 0.2 * predicted_quality);
        
        let confidence = self.calculate_prediction_confidence(predicted_quality);
        let uncertainty_range = self.calculate_uncertainty_range(
            predicted_bandwidth, 
            predicted_latency, 
            predicted_quality, 
            confidence
        );
        
        Ok(NetworkPrediction {
            predicted_bandwidth,
            predicted_latency,
            predicted_packet_loss,
            predicted_quality,
            confidence,
            prediction_window,
            factors: PredictionFactors {
                historical_trend: 0.3,
                seasonal_impact: self.seasonal_patterns.daily_pattern_strength,
                connection_type_impact: self.connection_type_impact(latest_metrics.connection_type.clone()),
                time_of_day_impact: time_features.is_peak_hours * 0.2,
                recent_performance: predicted_quality,
            },
            uncertainty_range,
        })
    }
    
    /// Statistical prediction using moving averages and trends
    fn statistical_prediction(
        &self,
        latest_metrics: &NetworkMetrics,
        time_features: &TimeFeatures,
        prediction_window: Duration,
    ) -> NetworkPrediction {
        let trend_factor = self.calculate_trend_factor();
        let seasonal_factor = self.seasonal_patterns.get_prediction_factor(time_features.hour_of_day);
        
        let predicted_bandwidth = self.moving_averages.bandwidth_15min * trend_factor * seasonal_factor;
        let predicted_latency = self.moving_averages.latency_15min / trend_factor;
        let predicted_packet_loss = latest_metrics.packet_loss * (2.0 - trend_factor);
        let predicted_quality = (predicted_bandwidth / 5_000_000.0 + 
                               (100.0 - predicted_latency) / 100.0 + 
                               (1.0 - predicted_packet_loss)) / 3.0;
        
        let confidence = 0.6; // Lower confidence for statistical predictions
        let uncertainty_range = self.calculate_uncertainty_range(
            predicted_bandwidth, 
            predicted_latency, 
            predicted_quality, 
            confidence
        );
        
        NetworkPrediction {
            predicted_bandwidth,
            predicted_latency,
            predicted_packet_loss,
            predicted_quality: predicted_quality.clamp(0.0, 1.0),
            confidence,
            prediction_window,
            factors: PredictionFactors {
                historical_trend: trend_factor - 1.0,
                seasonal_impact: seasonal_factor - 1.0,
                connection_type_impact: self.connection_type_impact(latest_metrics.connection_type.clone()),
                time_of_day_impact: time_features.is_peak_hours * 0.15,
                recent_performance: predicted_quality,
            },
            uncertainty_range,
        }
    }
    
    /// Fallback prediction when no data is available
    fn fallback_prediction(&self, prediction_window: Duration) -> NetworkPrediction {
        NetworkPrediction {
            predicted_bandwidth: 5_000_000.0, // 5 Mbps default
            predicted_latency: 50.0,
            predicted_packet_loss: 0.01,
            predicted_quality: 0.7,
            confidence: 0.3,
            prediction_window,
            factors: PredictionFactors {
                historical_trend: 0.0,
                seasonal_impact: 0.0,
                connection_type_impact: 0.0,
                time_of_day_impact: 0.0,
                recent_performance: 0.7,
            },
            uncertainty_range: UncertaintyRange {
                bandwidth_range: (1_000_000.0, 10_000_000.0),
                latency_range: (20.0, 200.0),
                quality_range: (0.3, 0.9),
            },
        }
    }
    
    /// Update moving averages with new sample
    fn update_moving_averages(&mut self, metrics: &NetworkMetrics) {
        // Simple exponential moving average implementation
        let alpha_5min = 0.1;
        let alpha_15min = 0.05;
        let alpha_1hour = 0.02;
        
        self.moving_averages.bandwidth_5min = 
            alpha_5min * metrics.bandwidth_bps + (1.0 - alpha_5min) * self.moving_averages.bandwidth_5min;
        self.moving_averages.bandwidth_15min = 
            alpha_15min * metrics.bandwidth_bps + (1.0 - alpha_15min) * self.moving_averages.bandwidth_15min;
        self.moving_averages.bandwidth_1hour = 
            alpha_1hour * metrics.bandwidth_bps + (1.0 - alpha_1hour) * self.moving_averages.bandwidth_1hour;
        
        self.moving_averages.latency_5min = 
            alpha_5min * metrics.latency_ms + (1.0 - alpha_5min) * self.moving_averages.latency_5min;
        self.moving_averages.latency_15min = 
            alpha_15min * metrics.latency_ms + (1.0 - alpha_15min) * self.moving_averages.latency_15min;
        self.moving_averages.latency_1hour = 
            alpha_1hour * metrics.latency_ms + (1.0 - alpha_1hour) * self.moving_averages.latency_1hour;
        
        let quality = metrics.connection_quality;
        self.moving_averages.quality_5min = 
            alpha_5min * quality + (1.0 - alpha_5min) * self.moving_averages.quality_5min;
        self.moving_averages.quality_15min = 
            alpha_15min * quality + (1.0 - alpha_15min) * self.moving_averages.quality_15min;
        self.moving_averages.quality_1hour = 
            alpha_1hour * quality + (1.0 - alpha_1hour) * self.moving_averages.quality_1hour;
    }
    
    /// Update seasonal patterns
    fn update_seasonal_patterns(&mut self, metrics: &NetworkMetrics) {
        let hour = self.get_hour_of_day(metrics.timestamp);
        
        // Update hourly pattern
        if self.seasonal_patterns.hourly_bandwidth_pattern.len() < 24 {
            self.seasonal_patterns.hourly_bandwidth_pattern.resize(24, 0.0);
            self.seasonal_patterns.hourly_latency_pattern.resize(24, 0.0);
        }
        
        let alpha = 0.05; // Learning rate for pattern updates
        self.seasonal_patterns.hourly_bandwidth_pattern[hour] = 
            alpha * metrics.bandwidth_bps + (1.0 - alpha) * self.seasonal_patterns.hourly_bandwidth_pattern[hour];
        self.seasonal_patterns.hourly_latency_pattern[hour] = 
            alpha * metrics.latency_ms + (1.0 - alpha) * self.seasonal_patterns.hourly_latency_pattern[hour];
        
        self.seasonal_patterns.last_pattern_update = metrics.timestamp;
    }
    
    /// Extract time-based features
    fn extract_time_features(&self, prediction_window: Duration) -> TimeFeatures {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let future_time = now + prediction_window.as_secs();
        let hour = self.get_hour_of_day(future_time);
        let day_of_week = self.get_day_of_week(future_time);
        
        TimeFeatures {
            hour_of_day: hour as f64 / 24.0,
            day_of_week: day_of_week as f64 / 7.0,
            is_weekend: if day_of_week >= 5 { 1.0 } else { 0.0 },
            is_peak_hours: if self.is_peak_hours(future_time) { 1.0 } else { 0.0 },
            time_since_last_sample: prediction_window.as_secs() as f64,
        }
    }
    
    /// Calculate prediction confidence based on model performance and data quality
    fn calculate_prediction_confidence(&self, predicted_quality: f64) -> f64 {
        let model_confidence = if self.metrics.accuracy > 0.0 {
            self.metrics.accuracy
        } else {
            0.5
        };
        
        let data_quality = if self.training_data.len() >= self.min_samples_for_training {
            (self.training_data.len() as f64 / self.max_training_samples as f64).min(1.0)
        } else {
            0.3
        };
        
        let prediction_stability = (1.0 - (predicted_quality - 0.5).abs() * 2.0).max(0.1);
        
        (model_confidence * 0.5 + data_quality * 0.3 + prediction_stability * 0.2).clamp(0.0, 1.0)
    }
    
    /// Calculate uncertainty range for predictions
    fn calculate_uncertainty_range(
        &self,
        bandwidth: f64,
        latency: f64,
        quality: f64,
        confidence: f64,
    ) -> UncertaintyRange {
        let uncertainty_factor = 1.0 - confidence;
        
        UncertaintyRange {
            bandwidth_range: (
                bandwidth * (1.0 - uncertainty_factor * 0.5),
                bandwidth * (1.0 + uncertainty_factor * 0.5),
            ),
            latency_range: (
                latency * (1.0 - uncertainty_factor * 0.3),
                latency * (1.0 + uncertainty_factor * 0.7),
            ),
            quality_range: (
                (quality - uncertainty_factor * 0.3).max(0.0),
                (quality + uncertainty_factor * 0.3).min(1.0),
            ),
        }
    }
    
    /// Calculate trend factor from recent performance
    fn calculate_trend_factor(&self) -> f64 {
        if self.training_data.len() < 5 {
            return 1.0;
        }
        
        let recent_samples = &self.training_data[self.training_data.len() - 5..];
        let mut bandwidth_trend = 0.0;
        
        for i in 1..recent_samples.len() {
            let current = recent_samples[i].bandwidth_bps;
            let previous = recent_samples[i - 1].bandwidth_bps;
            bandwidth_trend += (current - previous) / previous;
        }
        
        bandwidth_trend /= (recent_samples.len() - 1) as f64;
        (1.0 + bandwidth_trend).clamp(0.5, 2.0)
    }
    
    /// Get connection type impact factor
    fn connection_type_impact(&self, connection_type: ConnectionType) -> f64 {
        match connection_type {
            ConnectionType::Ethernet => 0.9,
            ConnectionType::WiFi => 0.7,
            ConnectionType::Cellular5G => 0.6,
            ConnectionType::Cellular4G => 0.4,
            ConnectionType::Satellite => 0.2,
            ConnectionType::Unknown => 0.5,
        }
    }
    
    /// Check if given timestamp is during peak hours
    fn is_peak_hours(&self, timestamp: u64) -> bool {
        let hour = self.get_hour_of_day(timestamp);
        (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 22)
    }
    
    /// Get hour of day from timestamp
    fn get_hour_of_day(&self, timestamp: u64) -> usize {
        ((timestamp / 3600) % 24) as usize
    }
    
    /// Get day of week from timestamp (0 = Monday)
    fn get_day_of_week(&self, timestamp: u64) -> usize {
        (((timestamp / 86400) + 3) % 7) as usize // Unix epoch was Thursday
    }
    
    /// Calculate mean of target values for accuracy calculation
    fn calculate_target_mean(&self) -> f64 {
        if self.training_data.is_empty() {
            return 1.0;
        }
        
        let sum: f64 = self.training_data
            .iter()
            .map(|metrics| self.create_target_value(metrics))
            .sum();
        
        sum / self.training_data.len() as f64
    }
}

impl MovingAverages {
    fn new() -> Self {
        Self {
            bandwidth_5min: 5_000_000.0,
            bandwidth_15min: 5_000_000.0,
            bandwidth_1hour: 5_000_000.0,
            latency_5min: 50.0,
            latency_15min: 50.0,
            latency_1hour: 50.0,
            quality_5min: 0.7,
            quality_15min: 0.7,
            quality_1hour: 0.7,
        }
    }
}

impl SeasonalPatterns {
    fn new() -> Self {
        Self {
            hourly_bandwidth_pattern: vec![5_000_000.0; 24],
            hourly_latency_pattern: vec![50.0; 24],
            daily_pattern_strength: 0.1,
            weekly_pattern_strength: 0.05,
            last_pattern_update: 0,
        }
    }
    
    fn get_current_hour_factor(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let hour = ((now / 3600) % 24) as usize;
        
        if hour < self.hourly_bandwidth_pattern.len() {
            self.hourly_bandwidth_pattern[hour] / 5_000_000.0
        } else {
            1.0
        }
    }
    
    fn get_prediction_factor(&self, hour_of_day: f64) -> f64 {
        let hour = (hour_of_day * 24.0) as usize % 24;
        if hour < self.hourly_bandwidth_pattern.len() {
            self.hourly_bandwidth_pattern[hour] / 5_000_000.0
        } else {
            1.0
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
            mean_absolute_error: f64::INFINITY,
            last_updated: 0,
            sample_count: 0,
        }
    }
}

impl NetworkMetrics {
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
    async fn test_network_predictor_creation() {
        let predictor = NetworkPredictor::new();
        assert!(predictor.is_ok());
    }
    
    #[tokio::test]
    async fn test_fallback_prediction() {
        let predictor = NetworkPredictor::new().unwrap();
        let prediction = predictor.predict_conditions(Duration::from_secs(60)).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.confidence > 0.0);
        assert!(pred.predicted_bandwidth > 0.0);
    }
    
    #[test]
    fn test_time_features_extraction() {
        let predictor = NetworkPredictor::new().unwrap();
        let features = predictor.extract_time_features(Duration::from_secs(300));
        
        assert!(features.hour_of_day >= 0.0 && features.hour_of_day <= 1.0);
        assert!(features.day_of_week >= 0.0 && features.day_of_week <= 1.0);
        assert!(features.is_weekend == 0.0 || features.is_weekend == 1.0);
    }
}