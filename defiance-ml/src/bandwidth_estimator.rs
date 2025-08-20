//! Real-time bandwidth estimation and prediction using ML techniques

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

/// Bandwidth estimator using multiple estimation techniques
pub struct BandwidthEstimator {
    model: Option<LinearRegression<f64, DenseMatrix<f64>>>,
    bandwidth_samples: VecDeque<BandwidthMeasurement>,
    prediction_history: VecDeque<BandwidthPredictionRecord>,
    estimation_methods: EstimationMethods,
    metrics: ModelMetrics,
    calibration_data: CalibrationData,
    min_samples_for_training: usize,
    max_samples: usize,
}

/// Bandwidth prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthPrediction {
    pub prediction_id: Uuid,
    pub predicted_bandwidth_bps: f64,
    pub confidence_interval: (f64, f64),
    pub prediction_method: EstimationMethod,
    pub time_horizon: Duration,
    pub factors: BandwidthFactors,
    pub quality_recommendation: QualityRecommendation,
    pub accuracy_score: f64,
}

/// Factors affecting bandwidth prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthFactors {
    pub historical_trend: f64,
    pub connection_type_impact: f64,
    pub time_of_day_factor: f64,
    pub network_congestion: f64,
    pub device_load_impact: f64,
    pub seasonal_adjustment: f64,
}

/// Quality recommendation based on bandwidth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub recommended_bitrate: f64,
    pub adaptive_strategy: AdaptiveStrategy,
    pub buffer_target: Duration,
    pub quality_level: String,
}

/// Adaptive streaming strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    Conservative,  // Lower bitrates for stability
    Balanced,     // Balance quality and stability
    Aggressive,   // Higher bitrates when possible
    PowerSaving,  // Optimize for battery life
}

/// Bandwidth measurement sample
#[derive(Debug, Clone)]
struct BandwidthMeasurement {
    timestamp: u64,
    measured_bandwidth: f64,
    measurement_method: MeasurementMethod,
    connection_type: ConnectionType,
    latency_ms: f64,
    packet_loss: f64,
    measurement_duration: Duration,
    bytes_transferred: u64,
    concurrent_streams: u32,
    device_cpu_usage: f64,
    network_interface: Option<String>,
}

/// Method used for bandwidth measurement
#[derive(Debug, Clone, PartialEq)]
enum MeasurementMethod {
    ActiveProbing,    // Direct bandwidth test
    PassiveMonitoring, // Monitor actual data transfer
    ApplicationLayer,  // Monitor streaming performance
    PacketPair,       // Packet pair technique
    CapacityEstimation, // Network capacity estimation
}

/// Estimation method used for prediction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EstimationMethod {
    MachineLearning,
    MovingAverage,
    ExponentialSmoothing,
    LinearRegression,
    SeasonalDecomposition,
    Kalman,
    Ensemble,
}

/// Different estimation techniques
#[derive(Debug, Clone)]
struct EstimationMethods {
    moving_average: MovingAverageEstimator,
    exponential_smoothing: ExponentialSmoothingEstimator,
    seasonal_decomposer: SeasonalDecomposer,
    kalman_filter: KalmanFilterEstimator,
}

/// Moving average bandwidth estimator
#[derive(Debug, Clone)]
struct MovingAverageEstimator {
    short_window: VecDeque<f64>,  // 5 samples
    medium_window: VecDeque<f64>, // 20 samples
    long_window: VecDeque<f64>,   // 100 samples
}

/// Exponential smoothing estimator
#[derive(Debug, Clone)]
struct ExponentialSmoothingEstimator {
    alpha: f64,           // Smoothing parameter
    current_estimate: f64,
    trend: f64,
    seasonal_estimates: Vec<f64>,
}

/// Seasonal pattern decomposer
#[derive(Debug, Clone)]
struct SeasonalDecomposer {
    hourly_patterns: Vec<f64>,    // 24 hour pattern
    daily_patterns: Vec<f64>,     // 7 day pattern
    pattern_strength: f64,
    last_update: u64,
}

/// Kalman filter for bandwidth estimation
#[derive(Debug, Clone)]
struct KalmanFilterEstimator {
    state_estimate: f64,
    error_covariance: f64,
    process_noise: f64,
    measurement_noise: f64,
}

/// Bandwidth prediction record for tracking accuracy
#[derive(Debug, Clone)]
struct BandwidthPredictionRecord {
    prediction_id: Uuid,
    timestamp: u64,
    predicted_value: f64,
    actual_value: Option<f64>,
    method_used: EstimationMethod,
    accuracy: Option<f64>,
    confidence: f64,
}

/// Calibration data for improving accuracy
#[derive(Debug, Clone)]
struct CalibrationData {
    method_accuracies: std::collections::HashMap<EstimationMethod, f64>,
    connection_type_biases: std::collections::HashMap<ConnectionType, f64>,
    time_of_day_factors: Vec<f64>, // 24 hour factors
    device_specific_adjustments: f64,
}

impl BandwidthEstimator {
    /// Create new bandwidth estimator
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: None,
            bandwidth_samples: VecDeque::new(),
            prediction_history: VecDeque::new(),
            estimation_methods: EstimationMethods::new(),
            metrics: ModelMetrics::new(),
            calibration_data: CalibrationData::new(),
            min_samples_for_training: 30,
            max_samples: 2000,
        })
    }
    
    /// Initialize the bandwidth estimator
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Bandwidth estimator initialized");
        Ok(())
    }
    
    /// Add a bandwidth measurement sample
    pub async fn add_sample(&mut self, metrics: NetworkMetrics) -> Result<()> {
        let measurement = BandwidthMeasurement {
            timestamp: metrics.timestamp,
            measured_bandwidth: metrics.bandwidth_bps,
            measurement_method: MeasurementMethod::PassiveMonitoring,
            connection_type: metrics.connection_type.clone(),
            latency_ms: metrics.latency_ms,
            packet_loss: metrics.packet_loss,
            measurement_duration: Duration::from_secs(1), // Default
            bytes_transferred: 0, // Would be populated in real implementation
            concurrent_streams: 1,
            device_cpu_usage: metrics.cpu_usage,
            network_interface: None,
        };
        
        self.bandwidth_samples.push_back(measurement.clone());
        
        // Limit sample size
        if self.bandwidth_samples.len() > self.max_samples {
            self.bandwidth_samples.pop_front();
        }
        
        // Update all estimation methods
        self.update_estimators(&measurement).await?;
        
        // Retrain ML model if we have enough samples
        if self.bandwidth_samples.len() >= self.min_samples_for_training {
            self.retrain_model().await?;
        }
        
        Ok(())
    }
    
    /// Predict bandwidth for the specified time horizon
    pub async fn predict_bandwidth(&self, time_horizon: Duration) -> Result<BandwidthPrediction> {
        let prediction_id = Uuid::new_v4();
        
        // Try different estimation methods and use the most accurate one
        let methods = vec![
            EstimationMethod::MachineLearning,
            EstimationMethod::ExponentialSmoothing,
            EstimationMethod::MovingAverage,
            EstimationMethod::Kalman,
        ];
        
        let mut best_prediction = None;
        let mut best_accuracy = 0.0;
        
        for method in methods {
            if let Ok(prediction) = self.predict_with_method(method.clone(), time_horizon, prediction_id).await {
                let accuracy = self.calibration_data.method_accuracies.get(&method).unwrap_or(&0.5);
                if *accuracy > best_accuracy {
                    best_accuracy = *accuracy;
                    best_prediction = Some(prediction);
                }
            }
        }
        
        // Fallback to ensemble if no individual method works
        best_prediction.or_else(|| {
            self.ensemble_prediction(time_horizon, prediction_id).ok()
        }).ok_or_else(|| anyhow::anyhow!("Failed to generate bandwidth prediction"))
    }
    
    /// Update model with actual bandwidth measurement
    pub async fn update_with_actual(&mut self, prediction_id: Uuid, actual_bandwidth: f64) -> Result<()> {
        // Find the prediction record
        if let Some(record) = self.prediction_history.iter_mut().find(|r| r.prediction_id == prediction_id) {
            record.actual_value = Some(actual_bandwidth);
            record.accuracy = Some(self.calculate_accuracy(record.predicted_value, actual_bandwidth));
            
            // Update method accuracy in calibration data
            let method_accuracy = self.calibration_data.method_accuracies
                .entry(record.method_used.clone())
                .or_insert(0.5);
            
            *method_accuracy = 0.9 * *method_accuracy + 0.1 * record.accuracy.unwrap();
            
            // Update overall metrics
            self.update_model_metrics().await;
        }
        
        Ok(())
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }
    
    /// Predict using specific method
    async fn predict_with_method(
        &self,
        method: EstimationMethod,
        time_horizon: Duration,
        prediction_id: Uuid,
    ) -> Result<BandwidthPrediction> {
        let predicted_bandwidth = match method {
            EstimationMethod::MachineLearning => self.ml_prediction(time_horizon)?,
            EstimationMethod::MovingAverage => self.moving_average_prediction(),
            EstimationMethod::ExponentialSmoothing => self.exponential_smoothing_prediction(),
            EstimationMethod::Kalman => self.kalman_prediction(),
            EstimationMethod::SeasonalDecomposition => self.seasonal_prediction(time_horizon),
            _ => return Err(anyhow::anyhow!("Unsupported prediction method")),
        };
        
        let confidence_interval = self.calculate_confidence_interval(predicted_bandwidth, &method);
        let factors = self.calculate_bandwidth_factors(time_horizon);
        let quality_recommendation = self.generate_quality_recommendation(predicted_bandwidth);
        let accuracy_score = self.calibration_data.method_accuracies.get(&method).unwrap_or(&0.5);
        
        Ok(BandwidthPrediction {
            prediction_id,
            predicted_bandwidth_bps: predicted_bandwidth,
            confidence_interval,
            prediction_method: method,
            time_horizon,
            factors,
            quality_recommendation,
            accuracy_score: *accuracy_score,
        })
    }
    
    /// Machine learning based prediction
    fn ml_prediction(&self, _time_horizon: Duration) -> Result<f64> {
        if let Some(ref model) = self.model {
            if self.bandwidth_samples.is_empty() {
                return Ok(5_000_000.0); // 5 Mbps default
            }
            
            let latest_sample = self.bandwidth_samples.back().unwrap();
            let features = self.extract_features(latest_sample);
            let x_matrix = DenseMatrix::from_2d_vec(&vec![features]);
            let predictions = model.predict(&x_matrix)?;
            
            Ok(predictions[0].max(0.0))
        } else {
            Err(anyhow::anyhow!("ML model not trained"))
        }
    }
    
    /// Moving average prediction
    fn moving_average_prediction(&self) -> f64 {
        if self.estimation_methods.moving_average.medium_window.is_empty() {
            return 5_000_000.0;
        }
        
        let sum: f64 = self.estimation_methods.moving_average.medium_window.iter().sum();
        sum / self.estimation_methods.moving_average.medium_window.len() as f64
    }
    
    /// Exponential smoothing prediction
    fn exponential_smoothing_prediction(&self) -> f64 {
        self.estimation_methods.exponential_smoothing.current_estimate
    }
    
    /// Kalman filter prediction
    fn kalman_prediction(&self) -> f64 {
        self.estimation_methods.kalman_filter.state_estimate
    }
    
    /// Seasonal decomposition prediction
    fn seasonal_prediction(&self, time_horizon: Duration) -> f64 {
        let base_prediction = self.exponential_smoothing_prediction();
        let seasonal_factor = self.get_seasonal_factor(time_horizon);
        base_prediction * seasonal_factor
    }
    
    /// Ensemble prediction combining multiple methods
    fn ensemble_prediction(&self, time_horizon: Duration, prediction_id: Uuid) -> Result<BandwidthPrediction> {
        let mut predictions = Vec::new();
        let mut weights = Vec::new();
        
        // Collect predictions from available methods
        if self.model.is_some() {
            if let Ok(pred) = self.ml_prediction(time_horizon) {
                predictions.push(pred);
                weights.push(0.4); // Higher weight for ML
            }
        }
        
        predictions.push(self.exponential_smoothing_prediction());
        weights.push(0.3);
        
        predictions.push(self.moving_average_prediction());
        weights.push(0.2);
        
        predictions.push(self.kalman_prediction());
        weights.push(0.1);
        
        // Weighted average
        let weighted_sum: f64 = predictions.iter().zip(weights.iter()).map(|(p, w)| p * w).sum();
        let weight_sum: f64 = weights.iter().sum();
        let ensemble_prediction = weighted_sum / weight_sum;
        
        let confidence_interval = self.calculate_confidence_interval(ensemble_prediction, &EstimationMethod::Ensemble);
        let factors = self.calculate_bandwidth_factors(time_horizon);
        let quality_recommendation = self.generate_quality_recommendation(ensemble_prediction);
        
        Ok(BandwidthPrediction {
            prediction_id,
            predicted_bandwidth_bps: ensemble_prediction,
            confidence_interval,
            prediction_method: EstimationMethod::Ensemble,
            time_horizon,
            factors,
            quality_recommendation,
            accuracy_score: 0.7, // Ensemble typically more accurate
        })
    }
    
    /// Update all estimation methods with new sample
    async fn update_estimators(&mut self, measurement: &BandwidthMeasurement) -> Result<()> {
        let bandwidth = measurement.measured_bandwidth;
        
        // Update moving averages
        self.estimation_methods.moving_average.update(bandwidth);
        
        // Update exponential smoothing
        self.estimation_methods.exponential_smoothing.update(bandwidth);
        
        // Update Kalman filter
        self.estimation_methods.kalman_filter.update(bandwidth);
        
        // Update seasonal patterns
        self.estimation_methods.seasonal_decomposer.update(measurement.timestamp, bandwidth);
        
        Ok(())
    }
    
    /// Retrain ML model
    async fn retrain_model(&mut self) -> Result<()> {
        if self.bandwidth_samples.len() < self.min_samples_for_training {
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
        self.metrics.sample_count = self.bandwidth_samples.len();
        
        self.model = Some(model);
        
        tracing::info!(
            "Bandwidth estimator retrained: accuracy={:.3}, samples={}", 
            self.metrics.accuracy, 
            self.metrics.sample_count
        );
        
        Ok(())
    }
    
    /// Prepare training data for ML model
    fn prepare_training_data(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut features = Vec::new();
        let mut targets = Vec::new();
        
        for (i, sample) in self.bandwidth_samples.iter().enumerate() {
            if i == 0 {
                continue; // Skip first sample
            }
            
            let feature_vector = self.extract_features(sample);
            features.push(feature_vector);
            targets.push(sample.measured_bandwidth);
        }
        
        (features, targets)
    }
    
    /// Extract features for ML model
    fn extract_features(&self, sample: &BandwidthMeasurement) -> Vec<f64> {
        vec![
            sample.latency_ms,
            sample.packet_loss,
            sample.device_cpu_usage,
            sample.connection_type_numeric(),
            self.get_time_of_day_factor(sample.timestamp),
            sample.concurrent_streams as f64,
            self.estimation_methods.moving_average.get_short_average(),
            self.estimation_methods.exponential_smoothing.current_estimate,
        ]
    }
    
    /// Calculate confidence interval
    fn calculate_confidence_interval(&self, prediction: f64, method: &EstimationMethod) -> (f64, f64) {
        let accuracy = self.calibration_data.method_accuracies.get(method).unwrap_or(&0.5);
        let uncertainty = 1.0 - accuracy;
        let margin = prediction * uncertainty * 0.5;
        
        ((prediction - margin).max(0.0), prediction + margin)
    }
    
    /// Calculate bandwidth factors
    fn calculate_bandwidth_factors(&self, time_horizon: Duration) -> BandwidthFactors {
        BandwidthFactors {
            historical_trend: self.calculate_trend(),
            connection_type_impact: 0.0, // Would be calculated based on current connection
            time_of_day_factor: self.get_time_of_day_factor_for_horizon(time_horizon),
            network_congestion: self.estimate_network_congestion(),
            device_load_impact: self.estimate_device_load_impact(),
            seasonal_adjustment: self.get_seasonal_factor(time_horizon),
        }
    }
    
    /// Generate quality recommendation
    fn generate_quality_recommendation(&self, predicted_bandwidth: f64) -> QualityRecommendation {
        let recommended_bitrate = predicted_bandwidth * 0.8; // Use 80% of predicted bandwidth
        
        let (quality_level, adaptive_strategy) = match predicted_bandwidth as u64 {
            0..=1_000_000 => ("Low", AdaptiveStrategy::Conservative),
            1_000_001..=3_000_000 => ("Medium", AdaptiveStrategy::Balanced),
            3_000_001..=8_000_000 => ("High", AdaptiveStrategy::Balanced),
            _ => ("Ultra", AdaptiveStrategy::Aggressive),
        };
        
        QualityRecommendation {
            recommended_bitrate,
            adaptive_strategy,
            buffer_target: Duration::from_secs(10),
            quality_level: quality_level.to_string(),
        }
    }
    
    /// Helper methods
    fn calculate_accuracy(&self, predicted: f64, actual: f64) -> f64 {
        1.0 - ((predicted - actual).abs() / actual.max(1.0))
    }
    
    fn calculate_mean_absolute_error(&self, actual: &[f64], predicted: &[f64]) -> f64 {
        actual.iter().zip(predicted.iter())
            .map(|(a, p)| (a - p).abs())
            .sum::<f64>() / actual.len() as f64
    }
    
    fn calculate_target_mean(&self) -> f64 {
        if self.bandwidth_samples.is_empty() {
            return 1.0;
        }
        
        let sum: f64 = self.bandwidth_samples.iter().map(|s| s.measured_bandwidth).sum();
        sum / self.bandwidth_samples.len() as f64
    }
    
    fn calculate_trend(&self) -> f64 {
        if self.bandwidth_samples.len() < 2 {
            return 0.0;
        }
        
        let recent_count = (self.bandwidth_samples.len() / 4).max(2);
        let recent_samples: Vec<_> = self.bandwidth_samples.iter().rev().take(recent_count).collect();
        
        if recent_samples.len() < 2 {
            return 0.0;
        }
        
        let first_avg = recent_samples[recent_count - 1].measured_bandwidth;
        let last_avg = recent_samples[0].measured_bandwidth;
        
        (last_avg - first_avg) / first_avg
    }
    
    fn get_time_of_day_factor(&self, timestamp: u64) -> f64 {
        let hour = ((timestamp / 3600) % 24) as usize;
        if hour < self.calibration_data.time_of_day_factors.len() {
            self.calibration_data.time_of_day_factors[hour]
        } else {
            1.0
        }
    }
    
    fn get_time_of_day_factor_for_horizon(&self, time_horizon: Duration) -> f64 {
        let future_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + time_horizon.as_secs();
        
        self.get_time_of_day_factor(future_timestamp)
    }
    
    fn get_seasonal_factor(&self, _time_horizon: Duration) -> f64 {
        1.0 // Placeholder implementation
    }
    
    fn estimate_network_congestion(&self) -> f64 {
        0.5 // Placeholder - would analyze latency and packet loss trends
    }
    
    fn estimate_device_load_impact(&self) -> f64 {
        if let Some(latest) = self.bandwidth_samples.back() {
            latest.device_cpu_usage
        } else {
            0.5
        }
    }
    
    async fn update_model_metrics(&mut self) {
        let recent_predictions: Vec<_> = self.prediction_history
            .iter()
            .filter(|p| p.actual_value.is_some())
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
impl EstimationMethods {
    fn new() -> Self {
        Self {
            moving_average: MovingAverageEstimator::new(),
            exponential_smoothing: ExponentialSmoothingEstimator::new(),
            seasonal_decomposer: SeasonalDecomposer::new(),
            kalman_filter: KalmanFilterEstimator::new(),
        }
    }
}

impl MovingAverageEstimator {
    fn new() -> Self {
        Self {
            short_window: VecDeque::new(),
            medium_window: VecDeque::new(),
            long_window: VecDeque::new(),
        }
    }
    
    fn update(&mut self, value: f64) {
        self.short_window.push_back(value);
        if self.short_window.len() > 5 {
            self.short_window.pop_front();
        }
        
        self.medium_window.push_back(value);
        if self.medium_window.len() > 20 {
            self.medium_window.pop_front();
        }
        
        self.long_window.push_back(value);
        if self.long_window.len() > 100 {
            self.long_window.pop_front();
        }
    }
    
    fn get_short_average(&self) -> f64 {
        if self.short_window.is_empty() {
            return 0.0;
        }
        self.short_window.iter().sum::<f64>() / self.short_window.len() as f64
    }
}

impl ExponentialSmoothingEstimator {
    fn new() -> Self {
        Self {
            alpha: 0.3,
            current_estimate: 5_000_000.0,
            trend: 0.0,
            seasonal_estimates: vec![1.0; 24],
        }
    }
    
    fn update(&mut self, value: f64) {
        let old_estimate = self.current_estimate;
        self.current_estimate = self.alpha * value + (1.0 - self.alpha) * (self.current_estimate + self.trend);
        self.trend = 0.1 * (self.current_estimate - old_estimate) + 0.9 * self.trend;
    }
}

impl SeasonalDecomposer {
    fn new() -> Self {
        Self {
            hourly_patterns: vec![1.0; 24],
            daily_patterns: vec![1.0; 7],
            pattern_strength: 0.1,
            last_update: 0,
        }
    }
    
    fn update(&mut self, timestamp: u64, value: f64) {
        let hour = ((timestamp / 3600) % 24) as usize;
        if hour < self.hourly_patterns.len() {
            let alpha = 0.05;
            self.hourly_patterns[hour] = alpha * (value / 5_000_000.0) + (1.0 - alpha) * self.hourly_patterns[hour];
        }
        self.last_update = timestamp;
    }
}

impl KalmanFilterEstimator {
    fn new() -> Self {
        Self {
            state_estimate: 5_000_000.0,
            error_covariance: 1000000.0,
            process_noise: 100000.0,
            measurement_noise: 500000.0,
        }
    }
    
    fn update(&mut self, measurement: f64) {
        // Prediction step
        let predicted_state = self.state_estimate;
        let predicted_covariance = self.error_covariance + self.process_noise;
        
        // Update step
        let kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise);
        self.state_estimate = predicted_state + kalman_gain * (measurement - predicted_state);
        self.error_covariance = (1.0 - kalman_gain) * predicted_covariance;
    }
}

impl CalibrationData {
    fn new() -> Self {
        let mut method_accuracies = std::collections::HashMap::new();
        method_accuracies.insert(EstimationMethod::MovingAverage, 0.6);
        method_accuracies.insert(EstimationMethod::ExponentialSmoothing, 0.7);
        method_accuracies.insert(EstimationMethod::MachineLearning, 0.8);
        method_accuracies.insert(EstimationMethod::Kalman, 0.75);
        method_accuracies.insert(EstimationMethod::Ensemble, 0.8);
        
        Self {
            method_accuracies,
            connection_type_biases: std::collections::HashMap::new(),
            time_of_day_factors: vec![1.0; 24],
            device_specific_adjustments: 1.0,
        }
    }
}

impl BandwidthMeasurement {
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
    async fn test_bandwidth_estimator_creation() {
        let estimator = BandwidthEstimator::new();
        assert!(estimator.is_ok());
    }
    
    #[tokio::test]
    async fn test_bandwidth_prediction() {
        let estimator = BandwidthEstimator::new().unwrap();
        let prediction = estimator.predict_bandwidth(Duration::from_secs(60)).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.predicted_bandwidth_bps > 0.0);
        assert!(pred.confidence_interval.0 <= pred.predicted_bandwidth_bps);
        assert!(pred.confidence_interval.1 >= pred.predicted_bandwidth_bps);
    }
    
    #[test]
    fn test_moving_average_estimator() {
        let mut estimator = MovingAverageEstimator::new();
        estimator.update(1000000.0);
        estimator.update(2000000.0);
        estimator.update(3000000.0);
        
        let avg = estimator.get_short_average();
        assert!(avg > 0.0);
        assert_eq!(avg, 2000000.0);
    }
}