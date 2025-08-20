//! # Renaissance UI Framework
//! 
//! Eco-friendly, beautiful UI framework for DefianceNetwork with cross-platform support.
//! Inspired by Renaissance art principles: harmony, proportion, and natural beauty.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

pub mod theme;
pub mod components;
pub mod layout;
pub mod config;
pub mod animation;

#[cfg(feature = "desktop")]
pub mod desktop;

#[cfg(feature = "audio")]
pub mod audio;

// Re-export commonly used types
pub use theme::{RenaissanceTheme, ColorPalette, ThemeManager};
pub use components::{Component, ComponentStyle, UIElement};
pub use layout::{LayoutManager, LayoutDirection, Spacing};
pub use config::{UIConfig, UISettings};

/// Core UI framework manager
pub struct RenaissanceUI {
    theme_manager: ThemeManager,
    config: UIConfig,
    components: HashMap<String, Box<dyn Component>>,
    animation_manager: animation::AnimationManager,
}

/// UI framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIFrameworkConfig {
    pub app_name: String,
    pub version: String,
    pub theme: String,
    pub language: String,
    pub accessibility: AccessibilityConfig,
    pub performance: PerformanceConfig,
    pub eco_mode: EcoModeConfig,
}

/// Accessibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityConfig {
    pub high_contrast: bool,
    pub large_text: bool,
    pub reduced_motion: bool,
    pub screen_reader_support: bool,
    pub keyboard_navigation: bool,
    pub color_blind_friendly: bool,
}

/// Performance configuration for eco-friendly operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub target_fps: u32,
    pub vsync: bool,
    pub gpu_acceleration: bool,
    pub battery_optimization: bool,
    pub thermal_throttling: bool,
}

/// Eco-mode configuration to reduce environmental impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcoModeConfig {
    pub enabled: bool,
    pub dark_theme_preference: bool,
    pub reduce_animations: bool,
    pub lower_quality_images: bool,
    pub power_saving_colors: bool,
    pub minimal_refresh_rate: bool,
}

impl Default for UIFrameworkConfig {
    fn default() -> Self {
        Self {
            app_name: "DefianceNetwork".to_string(),
            version: "0.1.0".to_string(),
            theme: "renaissance_natural".to_string(),
            language: "en".to_string(),
            accessibility: AccessibilityConfig {
                high_contrast: false,
                large_text: false,
                reduced_motion: false,
                screen_reader_support: true,
                keyboard_navigation: true,
                color_blind_friendly: true,
            },
            performance: PerformanceConfig {
                target_fps: 60,
                vsync: true,
                gpu_acceleration: true,
                battery_optimization: true,
                thermal_throttling: true,
            },
            eco_mode: EcoModeConfig {
                enabled: true,
                dark_theme_preference: true,
                reduce_animations: false,
                lower_quality_images: false,
                power_saving_colors: true,
                minimal_refresh_rate: false,
            },
        }
    }
}

impl RenaissanceUI {
    /// Create new Renaissance UI framework
    pub fn new(config: UIFrameworkConfig) -> Result<Self> {
        let theme_manager = ThemeManager::new()?;
        let ui_config = UIConfig::from_framework_config(&config)?;
        let animation_manager = animation::AnimationManager::new();
        
        Ok(Self {
            theme_manager,
            config: ui_config,
            components: HashMap::new(),
            animation_manager,
        })
    }
    
    /// Load configuration from file
    pub fn load_config(path: &str) -> Result<UIFrameworkConfig> {
        let content = std::fs::read_to_string(path)?;
        let config: UIFrameworkConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_config(config: &UIFrameworkConfig, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get default configuration directory
    pub fn get_config_dir() -> Result<std::path::PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?
            .join("DefianceNetwork");
        
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;
        }
        
        Ok(config_dir)
    }
    
    /// Initialize the UI framework
    pub fn initialize(&mut self) -> Result<()> {
        // Load theme
        self.theme_manager.load_theme(&self.config.theme_name)?;
        
        // Initialize components
        self.register_default_components()?;
        
        // Setup animation system
        self.animation_manager.initialize()?;
        
        tracing::info!("Renaissance UI framework initialized");
        Ok(())
    }
    
    /// Register default components
    fn register_default_components(&mut self) -> Result<()> {
        // Register built-in components
        self.register_component("button", components::create_button())?;
        self.register_component("text", components::create_text())?;
        self.register_component("input", components::create_input())?;
        self.register_component("video_player", components::create_video_player())?;
        self.register_component("audio_player", components::create_audio_player())?;
        self.register_component("broadcast_list", components::create_broadcast_list())?;
        self.register_component("peer_status", components::create_peer_status())?;
        self.register_component("payment_widget", components::create_payment_widget())?;
        
        Ok(())
    }
    
    /// Register a custom component
    pub fn register_component(&mut self, name: &str, component: Box<dyn Component>) -> Result<()> {
        self.components.insert(name.to_string(), component);
        tracing::debug!("Registered component: {}", name);
        Ok(())
    }
    
    /// Get component by name
    pub fn get_component(&self, name: &str) -> Option<&Box<dyn Component>> {
        self.components.get(name)
    }
    
    /// Get theme manager
    pub fn theme_manager(&self) -> &ThemeManager {
        &self.theme_manager
    }
    
    /// Get mutable theme manager
    pub fn theme_manager_mut(&mut self) -> &mut ThemeManager {
        &mut self.theme_manager
    }
    
    /// Get animation manager
    pub fn animation_manager(&self) -> &animation::AnimationManager {
        &self.animation_manager
    }
    
    /// Get mutable animation manager
    pub fn animation_manager_mut(&mut self) -> &mut animation::AnimationManager {
        &mut self.animation_manager
    }
    
    /// Update UI framework (called each frame)
    pub fn update(&mut self, delta_time: f32) -> Result<()> {
        // Update animations
        self.animation_manager.update(delta_time)?;
        
        // Update components
        for component in self.components.values_mut() {
            component.update(delta_time)?;
        }
        
        Ok(())
    }
    
    /// Apply eco-mode optimizations
    pub fn apply_eco_mode(&mut self, config: &EcoModeConfig) -> Result<()> {
        if config.enabled {
            // Switch to dark theme if preferred
            if config.dark_theme_preference {
                self.theme_manager.switch_theme("renaissance_dark")?;
            }
            
            // Reduce animation complexity
            if config.reduce_animations {
                self.animation_manager.set_reduced_mode(true);
            }
            
            // Apply power-saving colors
            if config.power_saving_colors {
                self.theme_manager.apply_power_saving_palette()?;
            }
            
            tracing::info!("Eco-mode optimizations applied");
        }
        
        Ok(())
    }
    
    /// Check battery level and auto-adjust settings
    pub fn auto_adjust_for_battery(&mut self, battery_level: Option<f32>) -> Result<()> {
        if let Some(level) = battery_level {
            if level < 0.2 { // Below 20%
                // Aggressive power saving
                self.animation_manager.set_reduced_mode(true);
                self.theme_manager.switch_theme("renaissance_minimal")?;
                tracing::info!("Low battery: Applied aggressive power saving");
            } else if level < 0.5 { // Below 50%
                // Moderate power saving
                self.animation_manager.set_performance_mode(false);
                tracing::info!("Medium battery: Applied moderate power saving");
            }
        }
        
        Ok(())
    }
    
    /// Get UI configuration
    pub fn config(&self) -> &UIConfig {
        &self.config
    }
    
    /// Update UI configuration
    pub fn update_config(&mut self, config: UIConfig) -> Result<()> {
        self.config = config;
        
        // Apply configuration changes
        self.theme_manager.load_theme(&self.config.theme_name)?;
        
        Ok(())
    }
}

/// UI Framework builder for easy configuration
pub struct RenaissanceUIBuilder {
    config: UIFrameworkConfig,
}

impl RenaissanceUIBuilder {
    /// Create new builder with default config
    pub fn new() -> Self {
        Self {
            config: UIFrameworkConfig::default(),
        }
    }
    
    /// Set application name
    pub fn app_name(mut self, name: &str) -> Self {
        self.config.app_name = name.to_string();
        self
    }
    
    /// Set theme
    pub fn theme(mut self, theme: &str) -> Self {
        self.config.theme = theme.to_string();
        self
    }
    
    /// Enable eco-mode
    pub fn eco_mode(mut self, enabled: bool) -> Self {
        self.config.eco_mode.enabled = enabled;
        self
    }
    
    /// Set target FPS
    pub fn target_fps(mut self, fps: u32) -> Self {
        self.config.performance.target_fps = fps;
        self
    }
    
    /// Enable accessibility features
    pub fn accessibility(mut self, config: AccessibilityConfig) -> Self {
        self.config.accessibility = config;
        self
    }
    
    /// Build the UI framework
    pub fn build(self) -> Result<RenaissanceUI> {
        RenaissanceUI::new(self.config)
    }
}

impl Default for RenaissanceUIBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ui_framework_creation() {
        let ui = RenaissanceUIBuilder::new()
            .app_name("Test App")
            .theme("renaissance_natural")
            .build();
        
        assert!(ui.is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = UIFrameworkConfig::default();
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());
        
        let deserialized: Result<UIFrameworkConfig, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
    
    #[test]
    fn test_eco_mode_config() {
        let eco_config = EcoModeConfig {
            enabled: true,
            dark_theme_preference: true,
            reduce_animations: true,
            lower_quality_images: false,
            power_saving_colors: true,
            minimal_refresh_rate: false,
        };
        
        assert!(eco_config.enabled);
        assert!(eco_config.dark_theme_preference);
    }
    
    #[test]
    fn test_config_directory() {
        let config_dir = RenaissanceUI::get_config_dir();
        assert!(config_dir.is_ok());
    }
}