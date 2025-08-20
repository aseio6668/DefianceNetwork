//! Chromecast and device casting support for DefianceNetwork

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use url::Url;

pub mod chromecast;
pub mod airplay;
pub mod dlna;

// Re-export main types
pub use chromecast::{ChromecastManager, ChromecastDevice, CastSession, CastCommand, MediaMetadata};
pub use airplay::{AirPlayManager, AirPlayDevice, AirPlaySession};

/// Universal casting manager supporting multiple protocols
pub struct CastingManager {
    chromecast: Arc<RwLock<chromecast::ChromecastManager>>,
    airplay: Arc<RwLock<airplay::AirPlayManager>>,
    event_sender: mpsc::UnboundedSender<CastingEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<CastingEvent>>,
    config: CastingConfig,
}

/// Universal casting device representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastingDevice {
    pub id: String,
    pub name: String,
    pub device_type: CastingProtocol,
    pub status: CastingStatus,
    pub capabilities: Vec<CastingCapability>,
    pub volume: f32,
    pub last_seen: i64,
}

/// Supported casting protocols
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CastingProtocol {
    Chromecast,
    AirPlay,
    DLNA,
    Miracast,
}

/// Device status for casting
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CastingStatus {
    Available,
    Busy,
    Offline,
    Connecting,
}

/// Casting capabilities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CastingCapability {
    Video,
    Audio,
    Screen,
    VolumeControl,
    RemoteControl,
    HDR,
    FourK,
    Subtitles,
}

/// Universal casting session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalCastSession {
    pub id: Uuid,
    pub device_id: String,
    pub protocol: CastingProtocol,
    pub content_url: Url,
    pub title: String,
    pub state: CastingState,
    pub position: f64,
    pub duration: Option<f64>,
    pub user_id: Uuid,
    pub started_at: i64,
}

/// Casting session state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CastingState {
    Loading,
    Playing,
    Paused,
    Buffering,
    Stopped,
    Error(String),
}

/// Casting events
#[derive(Debug, Clone)]
pub enum CastingEvent {
    DeviceDiscovered { device: CastingDevice },
    DeviceLost { device_id: String },
    SessionStarted { session_id: Uuid, protocol: CastingProtocol },
    SessionEnded { session_id: Uuid },
    StateChanged { session_id: Uuid, state: CastingState },
    VolumeChanged { device_id: String, volume: f32 },
    Error { device_id: Option<String>, error: String },
}

/// Casting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastingConfig {
    pub enable_chromecast: bool,
    pub enable_airplay: bool,
    pub enable_dlna: bool,
    pub discovery_timeout: u64,
    pub auto_discovery: bool,
    pub preferred_quality: String,
    pub buffer_size: usize,
}

impl Default for CastingConfig {
    fn default() -> Self {
        Self {
            enable_chromecast: true,
            enable_airplay: true,
            enable_dlna: true,
            discovery_timeout: 30,
            auto_discovery: true,
            preferred_quality: "high".to_string(),
            buffer_size: 8192,
        }
    }
}

impl CastingManager {
    /// Create new universal casting manager
    pub async fn new() -> Result<Self> {
        let config = CastingConfig::default();
        Self::with_config(config).await
    }

    /// Create casting manager with custom config
    pub async fn with_config(config: CastingConfig) -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        let chromecast = if config.enable_chromecast {
            Arc::new(RwLock::new(chromecast::ChromecastManager::new().await?))
        } else {
            Arc::new(RwLock::new(chromecast::ChromecastManager::new().await?)) // Still create for consistency
        };

        let airplay = if config.enable_airplay {
            Arc::new(RwLock::new(airplay::AirPlayManager::new().await?))
        } else {
            Arc::new(RwLock::new(airplay::AirPlayManager::new().await?)) // Still create for consistency
        };

        Ok(Self {
            chromecast,
            airplay,
            event_sender,
            event_receiver: Some(event_receiver),
            config,
        })
    }

    /// Start all casting services
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting universal casting manager");

        if self.config.enable_chromecast {
            let mut chromecast = self.chromecast.write().await;
            chromecast.start().await?;
        }

        if self.config.enable_airplay {
            let mut airplay = self.airplay.write().await;
            airplay.start().await?;
        }

        if self.config.auto_discovery {
            self.start_device_discovery().await?;
        }

        tracing::info!("Universal casting manager started");
        Ok(())
    }

    /// Stop all casting services
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping universal casting manager");

        if self.config.enable_chromecast {
            let mut chromecast = self.chromecast.write().await;
            chromecast.stop().await?;
        }

        if self.config.enable_airplay {
            let mut airplay = self.airplay.write().await;
            airplay.stop().await?;
        }

        tracing::info!("Universal casting manager stopped");
        Ok(())
    }

    /// Start device discovery for all protocols
    async fn start_device_discovery(&self) -> Result<()> {
        tracing::info!("Starting device discovery for all casting protocols");

        // Start periodic discovery updates
        let chromecast_arc = Arc::clone(&self.chromecast);
        let airplay_arc = Arc::clone(&self.airplay);
        let event_sender = self.event_sender.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check for new Chromecast devices
                // TODO: Implement actual discovery polling
                
                // Check for new AirPlay devices
                // TODO: Implement actual discovery polling
            }
        });

        Ok(())
    }

    /// Get all available casting devices
    pub async fn get_all_devices(&self) -> Vec<CastingDevice> {
        let mut devices = Vec::new();

        // Get Chromecast devices
        if self.config.enable_chromecast {
            let chromecast = self.chromecast.read().await;
            let chromecast_devices = chromecast.get_devices().await;
            
            for device in chromecast_devices {
                devices.push(CastingDevice {
                    id: device.id,
                    name: device.name,
                    device_type: CastingProtocol::Chromecast,
                    status: match device.status {
                        chromecast::DeviceStatus::Available => CastingStatus::Available,
                        chromecast::DeviceStatus::Busy => CastingStatus::Busy,
                        chromecast::DeviceStatus::Offline => CastingStatus::Offline,
                        chromecast::DeviceStatus::Unknown => CastingStatus::Offline,
                    },
                    capabilities: device.capabilities.into_iter().map(|cap| match cap {
                        chromecast::DeviceCapability::VideoPlayback => CastingCapability::Video,
                        chromecast::DeviceCapability::AudioPlayback => CastingCapability::Audio,
                        chromecast::DeviceCapability::VolumeControl => CastingCapability::VolumeControl,
                        chromecast::DeviceCapability::HDR => CastingCapability::HDR,
                        chromecast::DeviceCapability::FourK => CastingCapability::FourK,
                        chromecast::DeviceCapability::SubtitleSupport => CastingCapability::Subtitles,
                        _ => CastingCapability::Video, // Default mapping
                    }).collect(),
                    volume: device.volume_level,
                    last_seen: device.last_seen,
                });
            }
        }

        // Get AirPlay devices
        if self.config.enable_airplay {
            let airplay = self.airplay.read().await;
            let airplay_devices = airplay.get_devices().await;
            
            for device in airplay_devices {
                devices.push(CastingDevice {
                    id: device.id,
                    name: device.name,
                    device_type: CastingProtocol::AirPlay,
                    status: match device.status {
                        airplay::AirPlayStatus::Available => CastingStatus::Available,
                        airplay::AirPlayStatus::Playing => CastingStatus::Busy,
                        airplay::AirPlayStatus::Busy => CastingStatus::Busy,
                        airplay::AirPlayStatus::Offline => CastingStatus::Offline,
                    },
                    capabilities: device.features.into_iter().map(|feat| match feat {
                        airplay::AirPlayFeature::Video => CastingCapability::Video,
                        airplay::AirPlayFeature::Audio => CastingCapability::Audio,
                        airplay::AirPlayFeature::VolumeControl => CastingCapability::VolumeControl,
                        airplay::AirPlayFeature::Screen => CastingCapability::Screen,
                        _ => CastingCapability::Video, // Default mapping
                    }).collect(),
                    volume: device.volume,
                    last_seen: device.last_seen,
                });
            }
        }

        devices
    }

    /// Cast content to any supported device
    pub async fn cast_content(
        &mut self,
        device_id: String,
        content_url: Url,
        title: String,
        content_type: String,
        user_id: Uuid,
    ) -> Result<Uuid> {
        let devices = self.get_all_devices().await;
        let device = devices.into_iter()
            .find(|d| d.id == device_id)
            .ok_or_else(|| anyhow::anyhow!("Device not found"))?;

        match device.device_type {
            CastingProtocol::Chromecast => {
                let mut chromecast = self.chromecast.write().await;
                let metadata = MediaMetadata::new_video(title, None)
                    .with_description("DefianceNetwork content".to_string());
                
                let session_id = chromecast.cast_content(
                    device_id,
                    Uuid::new_v4(), // content_id
                    content_url,
                    metadata,
                    user_id,
                ).await?;

                self.send_event(CastingEvent::SessionStarted { 
                    session_id, 
                    protocol: CastingProtocol::Chromecast 
                }).await;

                Ok(session_id)
            }
            CastingProtocol::AirPlay => {
                let mut airplay = self.airplay.write().await;
                let content_type = if content_type.starts_with("video/") {
                    airplay::AirPlayContentType::Video
                } else {
                    airplay::AirPlayContentType::Audio
                };

                let session_id = airplay.start_session(
                    device_id,
                    content_url,
                    content_type,
                    title,
                    user_id,
                ).await?;

                self.send_event(CastingEvent::SessionStarted { 
                    session_id, 
                    protocol: CastingProtocol::AirPlay 
                }).await;

                Ok(session_id)
            }
            _ => {
                Err(anyhow::anyhow!("Protocol not yet implemented"))
            }
        }
    }

    /// Get casting statistics
    pub async fn get_stats(&self) -> CastingStats {
        let devices = self.get_all_devices().await;
        let available_devices = devices.iter().filter(|d| d.status == CastingStatus::Available).count();
        let busy_devices = devices.iter().filter(|d| d.status == CastingStatus::Busy).count();

        CastingStats {
            total_devices: devices.len(),
            available_devices,
            busy_devices,
            protocols_enabled: vec![
                if self.config.enable_chromecast { Some(CastingProtocol::Chromecast) } else { None },
                if self.config.enable_airplay { Some(CastingProtocol::AirPlay) } else { None },
                if self.config.enable_dlna { Some(CastingProtocol::DLNA) } else { None },
            ].into_iter().flatten().collect(),
        }
    }

    /// Take event receiver
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<CastingEvent>> {
        self.event_receiver.take()
    }

    /// Send casting event
    async fn send_event(&self, event: CastingEvent) {
        if let Err(_) = self.event_sender.send(event) {
            tracing::warn!("Failed to send casting event - no receivers");
        }
    }
}

/// Casting statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastingStats {
    pub total_devices: usize,
    pub available_devices: usize,
    pub busy_devices: usize,
    pub protocols_enabled: Vec<CastingProtocol>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_casting_manager_creation() {
        let manager = CastingManager::new().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_casting_with_config() {
        let config = CastingConfig {
            enable_chromecast: true,
            enable_airplay: false,
            enable_dlna: false,
            ..Default::default()
        };

        let manager = CastingManager::with_config(config).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_casting_protocol_equality() {
        assert_eq!(CastingProtocol::Chromecast, CastingProtocol::Chromecast);
        assert_ne!(CastingProtocol::Chromecast, CastingProtocol::AirPlay);
    }
}