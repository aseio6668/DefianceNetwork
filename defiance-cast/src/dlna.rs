//! DLNA (Digital Living Network Alliance) support

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use url::Url;

/// DLNA device manager
pub struct DLNAManager {
    devices: Arc<RwLock<HashMap<String, DLNADevice>>>,
    active_sessions: Arc<RwLock<HashMap<Uuid, DLNASession>>>,
    event_sender: mpsc::UnboundedSender<DLNAEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<DLNAEvent>>,
}

/// DLNA device representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DLNADevice {
    pub id: String,
    pub name: String,
    pub manufacturer: String,
    pub model: String,
    pub ip_address: IpAddr,
    pub port: u16,
    pub device_type: DLNADeviceType,
    pub capabilities: Vec<DLNACapability>,
    pub status: DLNAStatus,
    pub udn: String, // Unique Device Name
    pub last_seen: i64,
}

/// DLNA device types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DLNADeviceType {
    MediaRenderer,
    MediaServer,
    MediaController,
    DigitalMediaPlayer,
    TVDevice,
}

/// DLNA capabilities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DLNACapability {
    VideoPlayback,
    AudioPlayback,
    ImageDisplay,
    VolumeControl,
    PlaylistSupport,
    SubtitleSupport,
    SeekSupport,
}

/// DLNA device status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DLNAStatus {
    Available,
    Playing,
    Paused,
    Stopped,
    Busy,
    Error,
    Offline,
}

/// DLNA session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DLNASession {
    pub id: Uuid,
    pub device_id: String,
    pub content_url: Url,
    pub content_type: DLNAContentType,
    pub title: String,
    pub state: DLNAPlayState,
    pub position: u64, // seconds
    pub duration: Option<u64>, // seconds
    pub volume: u8, // 0-100
    pub started_at: i64,
    pub user_id: Uuid,
}

/// DLNA content types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DLNAContentType {
    Video,
    Audio,
    Image,
}

/// DLNA playback state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DLNAPlayState {
    NoMediaPresent,
    Stopped,
    Paused,
    Playing,
    Transitioning,
    Recording,
}

/// DLNA events
#[derive(Debug, Clone)]
pub enum DLNAEvent {
    DeviceDiscovered { device: DLNADevice },
    DeviceLost { device_id: String },
    SessionStarted { session_id: Uuid },
    SessionEnded { session_id: Uuid },
    PlayStateChanged { session_id: Uuid, state: DLNAPlayState },
    VolumeChanged { device_id: String, volume: u8 },
    PositionChanged { session_id: Uuid, position: u64 },
    Error { device_id: Option<String>, error: String },
}

impl DLNAManager {
    /// Create new DLNA manager
    pub async fn new() -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Some(event_receiver),
        })
    }

    /// Start DLNA discovery and service
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting DLNA manager");
        self.start_upnp_discovery().await?;
        Ok(())
    }

    /// Stop DLNA service
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping DLNA manager");
        
        // End all sessions
        let session_ids: Vec<Uuid> = {
            let sessions = self.active_sessions.read().await;
            sessions.keys().cloned().collect()
        };
        
        for session_id in session_ids {
            self.end_session(session_id).await?;
        }
        
        Ok(())
    }

    /// Start UPnP device discovery
    async fn start_upnp_discovery(&self) -> Result<()> {
        tracing::info!("Starting DLNA/UPnP device discovery");
        
        // TODO: Implement actual UPnP SSDP discovery
        // Look for devices with service types:
        // - urn:schemas-upnp-org:device:MediaRenderer:1
        // - urn:schemas-upnp-org:device:MediaServer:1
        
        // Mock device for testing
        let mock_device = DLNADevice {
            id: "dlna_smart_tv".to_string(),
            name: "Smart TV (DLNA)".to_string(),
            manufacturer: "Samsung".to_string(),
            model: "UN55RU7100".to_string(),
            ip_address: "192.168.1.102".parse().unwrap(),
            port: 1234,
            device_type: DLNADeviceType::MediaRenderer,
            capabilities: vec![
                DLNACapability::VideoPlayback,
                DLNACapability::AudioPlayback,
                DLNACapability::VolumeControl,
                DLNACapability::SeekSupport,
            ],
            status: DLNAStatus::Available,
            udn: "uuid:12345678-1234-1234-1234-123456789012".to_string(),
            last_seen: chrono::Utc::now().timestamp(),
        };

        self.add_device(mock_device).await;
        Ok(())
    }

    /// Add discovered DLNA device
    async fn add_device(&self, device: DLNADevice) {
        let device_id = device.id.clone();
        
        {
            let mut devices = self.devices.write().await;
            devices.insert(device_id.clone(), device.clone());
        }
        
        self.send_event(DLNAEvent::DeviceDiscovered { device }).await;
        tracing::info!("Discovered DLNA device: {}", device_id);
    }

    /// Get available DLNA devices
    pub async fn get_devices(&self) -> Vec<DLNADevice> {
        let devices = self.devices.read().await;
        devices.values()
            .filter(|d| matches!(d.status, DLNAStatus::Available | DLNAStatus::Stopped))
            .cloned()
            .collect()
    }

    /// Start DLNA playback session
    pub async fn start_playback(
        &mut self,
        device_id: String,
        content_url: Url,
        content_type: DLNAContentType,
        title: String,
        user_id: Uuid,
    ) -> Result<Uuid> {
        let device = {
            let devices = self.devices.read().await;
            devices.get(&device_id).cloned()
        };

        let device = device.ok_or_else(|| anyhow::anyhow!("DLNA device not found"))?;
        
        if !matches!(device.status, DLNAStatus::Available | DLNAStatus::Stopped) {
            return Err(anyhow::anyhow!("DLNA device not available"));
        }

        let session_id = Uuid::new_v4();

        let session = DLNASession {
            id: session_id,
            device_id: device_id.clone(),
            content_url: content_url.clone(),
            content_type,
            title,
            state: DLNAPlayState::Transitioning,
            position: 0,
            duration: None,
            volume: 50, // Default volume
            started_at: chrono::Utc::now().timestamp(),
            user_id,
        };

        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id, session);
        }

        // Update device status
        {
            let mut devices = self.devices.write().await;
            if let Some(device) = devices.get_mut(&device_id) {
                device.status = DLNAStatus::Playing;
            }
        }

        // Send UPnP commands to start playback
        self.send_upnp_play_command(&device_id, &content_url).await?;

        self.send_event(DLNAEvent::SessionStarted { session_id }).await;
        tracing::info!("Started DLNA playback session {} on device {}", session_id, device_id);

        Ok(session_id)
    }

    /// Send UPnP play command
    async fn send_upnp_play_command(&self, device_id: &str, content_url: &Url) -> Result<()> {
        tracing::debug!("Sending UPnP play command to device {}: {}", device_id, content_url);
        
        // TODO: Implement actual UPnP AVTransport commands:
        // 1. SetAVTransportURI
        // 2. Play
        
        // Simulate command execution
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        // Update session state to playing
        {
            let mut sessions = self.active_sessions.write().await;
            for session in sessions.values_mut() {
                if session.device_id == device_id {
                    session.state = DLNAPlayState::Playing;
                    self.send_event(DLNAEvent::PlayStateChanged { 
                        session_id: session.id, 
                        state: DLNAPlayState::Playing 
                    }).await;
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// Control DLNA playback
    pub async fn control_playback(
        &self,
        session_id: Uuid,
        action: DLNAAction,
    ) -> Result<()> {
        let session = {
            let sessions = self.active_sessions.read().await;
            sessions.get(&session_id).cloned()
        };

        let session = session.ok_or_else(|| anyhow::anyhow!("DLNA session not found"))?;
        
        self.send_upnp_control_command(&session.device_id, action).await?;
        
        Ok(())
    }

    /// Send UPnP control command
    async fn send_upnp_control_command(&self, device_id: &str, action: DLNAAction) -> Result<()> {
        tracing::debug!("Sending UPnP control command to device {}: {:?}", device_id, action);
        
        // TODO: Implement actual UPnP AVTransport control commands
        
        // Simulate command execution and update session state
        let new_state = match action {
            DLNAAction::Play => DLNAPlayState::Playing,
            DLNAAction::Pause => DLNAPlayState::Paused,
            DLNAAction::Stop => DLNAPlayState::Stopped,
            DLNAAction::Seek(_) => DLNAPlayState::Playing,
            DLNAAction::SetVolume(_) => return Ok(()), // Volume doesn't change play state
        };

        // Update session state
        {
            let mut sessions = self.active_sessions.write().await;
            for session in sessions.values_mut() {
                if session.device_id == device_id {
                    session.state = new_state.clone();
                    self.send_event(DLNAEvent::PlayStateChanged { 
                        session_id: session.id, 
                        state: new_state 
                    }).await;
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// End DLNA session
    pub async fn end_session(&mut self, session_id: Uuid) -> Result<()> {
        let session = {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(&session_id)
        };

        if let Some(session) = session {
            // Send stop command
            self.send_upnp_control_command(&session.device_id, DLNAAction::Stop).await?;

            // Update device status
            {
                let mut devices = self.devices.write().await;
                if let Some(device) = devices.get_mut(&session.device_id) {
                    device.status = DLNAStatus::Available;
                }
            }

            self.send_event(DLNAEvent::SessionEnded { session_id }).await;
            tracing::info!("Ended DLNA session {}", session_id);
        }

        Ok(())
    }

    /// Get DLNA session
    pub async fn get_session(&self, session_id: Uuid) -> Option<DLNASession> {
        let sessions = self.active_sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    /// Take event receiver
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<DLNAEvent>> {
        self.event_receiver.take()
    }

    /// Send DLNA event
    async fn send_event(&self, event: DLNAEvent) {
        if let Err(_) = self.event_sender.send(event) {
            tracing::warn!("Failed to send DLNA event - no receivers");
        }
    }
}

/// DLNA control actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DLNAAction {
    Play,
    Pause,
    Stop,
    Seek(u64), // position in seconds
    SetVolume(u8), // 0-100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dlna_manager_creation() {
        let manager = DLNAManager::new().await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_dlna_device_types() {
        assert_eq!(DLNADeviceType::MediaRenderer, DLNADeviceType::MediaRenderer);
        assert_ne!(DLNADeviceType::MediaRenderer, DLNADeviceType::MediaServer);
    }

    #[test]
    fn test_dlna_capabilities() {
        let capabilities = vec![
            DLNACapability::VideoPlayback,
            DLNACapability::VolumeControl,
            DLNACapability::SeekSupport,
        ];
        
        assert!(capabilities.contains(&DLNACapability::VideoPlayback));
        assert!(capabilities.contains(&DLNACapability::VolumeControl));
    }
}