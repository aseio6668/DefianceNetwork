//! Audio streaming implementation for Audigy

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use url::Url;
use anyhow::Result;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

/// Audio streaming manager
pub struct AudioStreamer {
    active_sessions: Arc<RwLock<HashMap<Uuid, StreamingSession>>>,
    event_sender: mpsc::UnboundedSender<StreamingEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<StreamingEvent>>,
    config: StreamingConfig,
}

/// Active streaming session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSession {
    pub id: Uuid,
    pub content_id: Uuid,
    pub source_url: Url,
    pub started_at: i64,
    pub quality: StreamingQuality,
    pub state: StreamingState,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub download_speed: f64, // bytes per second
    pub buffer_health: f32, // 0.0 to 1.0
}

/// Streaming quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamingQuality {
    Low,    // 64 kbps
    Medium, // 128 kbps
    High,   // 256 kbps
    Ultra,  // 320+ kbps
    Auto,   // Adaptive based on connection
}

/// Streaming session state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StreamingState {
    Initializing,
    Buffering,
    Streaming,
    Paused,
    Completed,
    Error(String),
}

/// Streaming events
#[derive(Debug, Clone)]
pub enum StreamingEvent {
    SessionStarted { session_id: Uuid },
    BufferingProgress { session_id: Uuid, progress: f32 },
    StreamingStarted { session_id: Uuid },
    DataReceived { session_id: Uuid, bytes: u64 },
    QualityChanged { session_id: Uuid, quality: StreamingQuality },
    SessionPaused { session_id: Uuid },
    SessionResumed { session_id: Uuid },
    SessionCompleted { session_id: Uuid },
    SessionError { session_id: Uuid, error: String },
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub buffer_size_kb: usize,
    pub min_buffer_seconds: u64,
    pub max_concurrent_streams: usize,
    pub adaptive_quality: bool,
    pub download_timeout_seconds: u64,
    pub retry_attempts: u8,
    pub preferred_quality: StreamingQuality,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size_kb: 512,
            min_buffer_seconds: 30,
            max_concurrent_streams: 5,
            adaptive_quality: true,
            download_timeout_seconds: 30,
            retry_attempts: 3,
            preferred_quality: StreamingQuality::High,
        }
    }
}

impl AudioStreamer {
    /// Create new audio streamer
    pub async fn new() -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Some(event_receiver),
            config: StreamingConfig::default(),
        })
    }

    /// Start the audio streamer
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting audio streamer");
        // TODO: Initialize streaming backend
        Ok(())
    }

    /// Stop the audio streamer
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping audio streamer");
        
        // Stop all active sessions
        let sessions: Vec<Uuid> = {
            let sessions_lock = self.active_sessions.read().await;
            sessions_lock.keys().cloned().collect()
        };
        
        for session_id in sessions {
            self.stop_session(session_id).await?;
        }
        
        Ok(())
    }

    /// Start streaming from URL
    pub async fn start_streaming(&self, source_url: &Url) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        let content_id = Uuid::new_v4(); // TODO: Get actual content ID

        let session = StreamingSession {
            id: session_id,
            content_id,
            source_url: source_url.clone(),
            started_at: chrono::Utc::now().timestamp(),
            quality: self.config.preferred_quality.clone(),
            state: StreamingState::Initializing,
            bytes_downloaded: 0,
            total_bytes: None,
            download_speed: 0.0,
            buffer_health: 0.0,
        };

        // Add to active sessions
        {
            let mut sessions = self.active_sessions.write().await;
            
            // Check concurrent stream limit
            if sessions.len() >= self.config.max_concurrent_streams {
                return Err(anyhow::anyhow!("Maximum concurrent streams reached"));
            }
            
            sessions.insert(session_id, session);
        }

        // Send event
        self.send_event(StreamingEvent::SessionStarted { session_id }).await;

        // Start streaming task
        let sessions_arc = Arc::clone(&self.active_sessions);
        let event_sender = self.event_sender.clone();
        let source_url = source_url.clone();
        let source_url_for_log = source_url.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::stream_task(sessions_arc, event_sender, session_id, source_url, config).await {
                tracing::error!("Streaming task failed for session {}: {}", session_id, e);
            }
        });

        tracing::info!("Started streaming session {} from {}", session_id, source_url_for_log);
        Ok(session_id)
    }

    /// Stop a streaming session
    pub async fn stop_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(mut session) = sessions.remove(&session_id) {
            session.state = StreamingState::Completed;
            self.send_event(StreamingEvent::SessionCompleted { session_id }).await;
            tracing::info!("Stopped streaming session {}", session_id);
        }
        
        Ok(())
    }

    /// Pause a streaming session
    pub async fn pause_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            if session.state == StreamingState::Streaming {
                session.state = StreamingState::Paused;
                self.send_event(StreamingEvent::SessionPaused { session_id }).await;
                tracing::info!("Paused streaming session {}", session_id);
            }
        }
        
        Ok(())
    }

    /// Resume a streaming session
    pub async fn resume_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            if session.state == StreamingState::Paused {
                session.state = StreamingState::Streaming;
                self.send_event(StreamingEvent::SessionResumed { session_id }).await;
                tracing::info!("Resumed streaming session {}", session_id);
            }
        }
        
        Ok(())
    }

    /// Change streaming quality
    pub async fn change_quality(&self, session_id: Uuid, quality: StreamingQuality) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if let Some(session) = sessions.get_mut(&session_id) {
            session.quality = quality.clone();
            self.send_event(StreamingEvent::QualityChanged { session_id, quality }).await;
            tracing::info!("Changed quality for session {} to {:?}", session_id, session.quality);
        }
        
        Ok(())
    }

    /// Get session information
    pub async fn get_session(&self, session_id: Uuid) -> Option<StreamingSession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    /// Get all active sessions
    pub async fn get_active_sessions(&self) -> Vec<StreamingSession> {
        let sessions = self.active_sessions.read().await;
        sessions.values().cloned().collect()
    }

    /// Take event receiver for listening to streaming events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<StreamingEvent>> {
        self.event_receiver.take()
    }

    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        let sessions = self.active_sessions.read().await;
        
        let total_sessions = sessions.len();
        let active_streams = sessions.values()
            .filter(|s| s.state == StreamingState::Streaming)
            .count();
        let total_downloaded = sessions.values()
            .map(|s| s.bytes_downloaded)
            .sum();
        let average_speed = if total_sessions > 0 {
            sessions.values().map(|s| s.download_speed).sum::<f64>() / total_sessions as f64
        } else {
            0.0
        };

        StreamingStats {
            total_sessions,
            active_streams,
            total_downloaded,
            average_speed,
        }
    }

    /// Internal streaming task
    async fn stream_task(
        sessions: Arc<RwLock<HashMap<Uuid, StreamingSession>>>,
        event_sender: mpsc::UnboundedSender<StreamingEvent>,
        session_id: Uuid,
        source_url: Url,
        config: StreamingConfig,
    ) -> Result<()> {
        // Update session state to buffering
        {
            let mut sessions_lock = sessions.write().await;
            if let Some(session) = sessions_lock.get_mut(&session_id) {
                session.state = StreamingState::Buffering;
            }
        }

        let _ = event_sender.send(StreamingEvent::BufferingProgress { 
            session_id, 
            progress: 0.0 
        });

        // Create HTTP client
        let client = reqwest::Client::new();
        
        // Start streaming request
        let response = client
            .get(source_url.clone())
            .timeout(std::time::Duration::from_secs(config.download_timeout_seconds))
            .send()
            .await?;

        if !response.status().is_success() {
            let error = format!("HTTP error: {}", response.status());
            let _ = event_sender.send(StreamingEvent::SessionError { 
                session_id, 
                error: error.clone() 
            });
            return Err(anyhow::anyhow!(error));
        }

        // Get content length if available
        let total_size = response.content_length();
        {
            let mut sessions_lock = sessions.write().await;
            if let Some(session) = sessions_lock.get_mut(&session_id) {
                session.total_bytes = total_size;
            }
        }

        // Start streaming
        {
            let mut sessions_lock = sessions.write().await;
            if let Some(session) = sessions_lock.get_mut(&session_id) {
                session.state = StreamingState::Streaming;
            }
        }

        let _ = event_sender.send(StreamingEvent::StreamingStarted { session_id });

        // Stream data in chunks
        let mut bytes_stream = response.bytes_stream();
        let mut total_bytes_received = 0u64;
        let mut buffer = Vec::new();
        let start_time = std::time::Instant::now();

        use futures::StreamExt;
        while let Some(chunk_result) = bytes_stream.next().await {
            // Check if session is still active
            let session_active = {
                let sessions_lock = sessions.read().await;
                sessions_lock.contains_key(&session_id)
            };

            if !session_active {
                break;
            }

            match chunk_result {
                Ok(chunk) => {
                    let chunk_size = chunk.len() as u64;
                    total_bytes_received += chunk_size;
                    buffer.extend_from_slice(&chunk);

                    // Calculate download speed
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        total_bytes_received as f64 / elapsed
                    } else {
                        0.0
                    };

                    // Update session
                    {
                        let mut sessions_lock = sessions.write().await;
                        if let Some(session) = sessions_lock.get_mut(&session_id) {
                            session.bytes_downloaded = total_bytes_received;
                            session.download_speed = speed;
                            
                            // Calculate buffer health
                            let buffer_seconds = buffer.len() as f64 / (session.quality.bitrate_bps() as f64 / 8.0);
                            session.buffer_health = (buffer_seconds / config.min_buffer_seconds as f64).min(1.0) as f32;
                        }
                    }

                    let _ = event_sender.send(StreamingEvent::DataReceived { 
                        session_id, 
                        bytes: chunk_size 
                    });

                    // Simulate buffer management
                    if buffer.len() > config.buffer_size_kb * 1024 {
                        buffer.drain(0..buffer.len() / 2); // Remove half the buffer
                    }
                }
                Err(e) => {
                    let error = format!("Streaming error: {}", e);
                    let _ = event_sender.send(StreamingEvent::SessionError { 
                        session_id, 
                        error 
                    });
                    break;
                }
            }
        }

        // Mark session as completed
        {
            let mut sessions_lock = sessions.write().await;
            if let Some(session) = sessions_lock.get_mut(&session_id) {
                session.state = StreamingState::Completed;
            }
        }

        let _ = event_sender.send(StreamingEvent::SessionCompleted { session_id });
        
        Ok(())
    }

    /// Internal helper to send events
    async fn send_event(&self, event: StreamingEvent) {
        if let Err(_) = self.event_sender.send(event) {
            tracing::warn!("Failed to send streaming event - no receivers");
        }
    }
}

impl StreamingQuality {
    /// Get bitrate in bits per second
    pub fn bitrate_bps(&self) -> u32 {
        match self {
            Self::Low => 64_000,
            Self::Medium => 128_000,
            Self::High => 256_000,
            Self::Ultra => 320_000,
            Self::Auto => 256_000, // Default for auto
        }
    }

    /// Get quality as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "Low (64 kbps)",
            Self::Medium => "Medium (128 kbps)",
            Self::High => "High (256 kbps)",
            Self::Ultra => "Ultra (320+ kbps)",
            Self::Auto => "Auto",
        }
    }
}

/// Streaming statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub total_sessions: usize,
    pub active_streams: usize,
    pub total_downloaded: u64,
    pub average_speed: f64, // bytes per second
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audio_streamer_creation() {
        let streamer = AudioStreamer::new().await;
        assert!(streamer.is_ok());
    }

    #[tokio::test]
    async fn test_session_management() {
        let streamer = AudioStreamer::new().await.unwrap();
        
        // Test that we start with no sessions
        let sessions = streamer.get_active_sessions().await;
        assert_eq!(sessions.len(), 0);
        
        // Test stats with no sessions
        let stats = streamer.get_stats().await;
        assert_eq!(stats.total_sessions, 0);
        assert_eq!(stats.active_streams, 0);
    }

    #[test]
    fn test_streaming_quality() {
        assert_eq!(StreamingQuality::Low.bitrate_bps(), 64_000);
        assert_eq!(StreamingQuality::High.bitrate_bps(), 256_000);
        assert_eq!(StreamingQuality::Low.as_str(), "Low (64 kbps)");
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.buffer_size_kb, 512);
        assert_eq!(config.max_concurrent_streams, 5);
        assert!(config.adaptive_quality);
    }

    #[test]
    fn test_streaming_state_equality() {
        assert_eq!(StreamingState::Initializing, StreamingState::Initializing);
        assert_ne!(StreamingState::Streaming, StreamingState::Paused);
        
        let error1 = StreamingState::Error("test".to_string());
        let error2 = StreamingState::Error("test".to_string());
        assert_eq!(error1, error2);
    }
}