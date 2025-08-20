//! Configuration system for Renaissance UI

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use crate::{UIFrameworkConfig, AccessibilityConfig, PerformanceConfig, EcoModeConfig};

/// UI configuration management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    pub theme_name: String,
    pub language: String,
    pub font_scale: f32,
    pub animation_scale: f32,
    pub auto_theme: bool,
    pub settings: UISettings,
}

/// Detailed UI settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UISettings {
    pub accessibility: AccessibilitySettings,
    pub performance: PerformanceSettings,
    pub eco_mode: EcoModeSettings,
    pub appearance: AppearanceSettings,
    pub behavior: BehaviorSettings,
    pub privacy: PrivacySettings,
}

/// Accessibility settings for inclusive design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilitySettings {
    pub high_contrast: bool,
    pub large_text: bool,
    pub reduced_motion: bool,
    pub screen_reader_support: bool,
    pub keyboard_navigation: bool,
    pub color_blind_friendly: bool,
    pub focus_indicators: bool,
    pub touch_target_size: TouchTargetSize,
    pub font_weight_adjustment: f32,
    pub cursor_size_multiplier: f32,
}

/// Touch target size for accessibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TouchTargetSize {
    Small,   // 32px minimum
    Medium,  // 44px minimum (recommended)
    Large,   // 56px minimum
}

/// Performance settings for optimal operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    pub target_fps: u32,
    pub vsync: bool,
    pub gpu_acceleration: bool,
    pub battery_optimization: bool,
    pub thermal_throttling: bool,
    pub memory_management: MemoryManagement,
    pub render_quality: RenderQuality,
    pub background_processing: bool,
}

/// Memory management strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryManagement {
    Conservative, // Minimize memory usage
    Balanced,     // Balance memory and performance
    Performance,  // Prioritize performance
}

/// Render quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderQuality {
    Low,     // Reduce visual quality for performance
    Medium,  // Balanced quality and performance
    High,    // Best visual quality
    Auto,    // Automatically adjust based on hardware
}

/// Eco-mode settings for environmental consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcoModeSettings {
    pub enabled: bool,
    pub auto_enable_on_battery: bool,
    pub battery_threshold: f32, // Auto-enable below this percentage
    pub dark_theme_preference: bool,
    pub reduce_animations: bool,
    pub lower_quality_images: bool,
    pub power_saving_colors: bool,
    pub minimal_refresh_rate: bool,
    pub cpu_throttling: bool,
    pub background_dimming: bool,
}

/// Appearance customization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppearanceSettings {
    pub theme_preference: ThemePreference,
    pub custom_accent_color: Option<String>, // Hex color
    pub window_transparency: f32,
    pub corner_radius: f32,
    pub shadow_intensity: f32,
    pub icon_style: IconStyle,
    pub density: UIDensity,
    pub compact_mode: bool,
}

/// Theme preference options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThemePreference {
    Light,
    Dark,
    Auto, // Follow system preference
    Custom(String), // Custom theme name
}

/// Icon style preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IconStyle {
    Outline,
    Filled,
    Rounded,
    Sharp,
}

/// UI density for information display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UIDensity {
    Comfortable, // More spacing
    Standard,    // Default spacing
    Compact,     // Less spacing
}

/// Behavior and interaction settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorSettings {
    pub double_click_speed: f32,
    pub tooltip_delay: f32,
    pub auto_save_interval: u32, // seconds
    pub undo_history_size: u32,
    pub default_view_mode: ViewMode,
    pub confirm_destructive_actions: bool,
    pub auto_play_videos: bool,
    pub loop_audio: bool,
    pub notification_settings: NotificationSettings,
}

/// Default view mode for content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewMode {
    List,
    Grid,
    Tiles,
    Details,
}

/// Notification preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSettings {
    pub enabled: bool,
    pub sound_enabled: bool,
    pub show_on_screen: bool,
    pub duration: f32, // seconds
    pub position: NotificationPosition,
    pub categories: HashMap<String, bool>, // Category -> enabled
}

/// Notification display position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

/// Privacy and security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    pub analytics_enabled: bool,
    pub crash_reporting: bool,
    pub usage_statistics: bool,
    pub location_sharing: bool,
    pub peer_discovery: bool,
    pub auto_connect: bool,
    pub data_collection_level: DataCollectionLevel,
}

/// Data collection preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataCollectionLevel {
    None,     // No data collection
    Essential, // Only essential data for functionality
    Standard, // Standard analytics and improvement data
    Full,     // All available data for optimization
}

/// Configuration validator
pub struct ConfigValidator;

impl UIConfig {
    /// Create from framework config
    pub fn from_framework_config(framework_config: &UIFrameworkConfig) -> Result<Self> {
        Ok(Self {
            theme_name: framework_config.theme.clone(),
            language: framework_config.language.clone(),
            font_scale: 1.0,
            animation_scale: 1.0,
            auto_theme: true,
            settings: UISettings::from_framework_config(framework_config)?,
        })
    }
    
    /// Load configuration from file
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        ConfigValidator::validate(&config)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        ConfigValidator::validate(self)?;
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Apply eco-mode settings
    pub fn apply_eco_mode(&mut self, enabled: bool) {
        self.settings.eco_mode.enabled = enabled;
        
        if enabled {
            // Automatically adjust other settings for eco-mode
            self.settings.performance.target_fps = self.settings.performance.target_fps.min(30);
            self.settings.performance.render_quality = RenderQuality::Low;
            self.settings.appearance.window_transparency = 0.0;
            self.settings.appearance.shadow_intensity = 0.5;
            self.animation_scale = 0.5;
            
            if self.settings.eco_mode.dark_theme_preference {
                self.theme_name = "renaissance_dark".to_string();
            }
        }
    }
    
    /// Apply accessibility settings
    pub fn apply_accessibility(&mut self, config: &AccessibilityConfig) {
        self.settings.accessibility.high_contrast = config.high_contrast;
        self.settings.accessibility.large_text = config.large_text;
        self.settings.accessibility.reduced_motion = config.reduced_motion;
        self.settings.accessibility.screen_reader_support = config.screen_reader_support;
        self.settings.accessibility.keyboard_navigation = config.keyboard_navigation;
        self.settings.accessibility.color_blind_friendly = config.color_blind_friendly;
        
        // Adjust other settings based on accessibility needs
        if config.large_text {
            self.font_scale = 1.25;
        }
        
        if config.reduced_motion {
            self.animation_scale = 0.2;
        }
        
        if config.high_contrast {
            self.theme_name = "renaissance_high_contrast".to_string();
        }
    }
    
    /// Get effective target FPS (considering eco-mode)
    pub fn effective_target_fps(&self) -> u32 {
        if self.settings.eco_mode.enabled {
            self.settings.performance.target_fps.min(30)
        } else {
            self.settings.performance.target_fps
        }
    }
    
    /// Check if animations should be reduced
    pub fn should_reduce_animations(&self) -> bool {
        self.settings.accessibility.reduced_motion || 
        (self.settings.eco_mode.enabled && self.settings.eco_mode.reduce_animations) ||
        self.animation_scale < 0.5
    }
    
    /// Get effective font scale
    pub fn effective_font_scale(&self) -> f32 {
        let mut scale = self.font_scale;
        
        if self.settings.accessibility.large_text {
            scale *= 1.25;
        }
        
        scale * self.settings.accessibility.font_weight_adjustment
    }
}

impl UISettings {
    /// Create from framework config
    pub fn from_framework_config(framework_config: &UIFrameworkConfig) -> Result<Self> {
        Ok(Self {
            accessibility: AccessibilitySettings::from(&framework_config.accessibility),
            performance: PerformanceSettings::from(&framework_config.performance),
            eco_mode: EcoModeSettings::from(&framework_config.eco_mode),
            appearance: AppearanceSettings::default(),
            behavior: BehaviorSettings::default(),
            privacy: PrivacySettings::default(),
        })
    }
}

impl From<&AccessibilityConfig> for AccessibilitySettings {
    fn from(config: &AccessibilityConfig) -> Self {
        Self {
            high_contrast: config.high_contrast,
            large_text: config.large_text,
            reduced_motion: config.reduced_motion,
            screen_reader_support: config.screen_reader_support,
            keyboard_navigation: config.keyboard_navigation,
            color_blind_friendly: config.color_blind_friendly,
            focus_indicators: true,
            touch_target_size: TouchTargetSize::Medium,
            font_weight_adjustment: 1.0,
            cursor_size_multiplier: 1.0,
        }
    }
}

impl From<&PerformanceConfig> for PerformanceSettings {
    fn from(config: &PerformanceConfig) -> Self {
        Self {
            target_fps: config.target_fps,
            vsync: config.vsync,
            gpu_acceleration: config.gpu_acceleration,
            battery_optimization: config.battery_optimization,
            thermal_throttling: config.thermal_throttling,
            memory_management: MemoryManagement::Balanced,
            render_quality: RenderQuality::Auto,
            background_processing: true,
        }
    }
}

impl From<&EcoModeConfig> for EcoModeSettings {
    fn from(config: &EcoModeConfig) -> Self {
        Self {
            enabled: config.enabled,
            auto_enable_on_battery: true,
            battery_threshold: 0.2,
            dark_theme_preference: config.dark_theme_preference,
            reduce_animations: config.reduce_animations,
            lower_quality_images: config.lower_quality_images,
            power_saving_colors: config.power_saving_colors,
            minimal_refresh_rate: config.minimal_refresh_rate,
            cpu_throttling: true,
            background_dimming: true,
        }
    }
}

impl Default for AppearanceSettings {
    fn default() -> Self {
        Self {
            theme_preference: ThemePreference::Auto,
            custom_accent_color: None,
            window_transparency: 0.0,
            corner_radius: 8.0,
            shadow_intensity: 1.0,
            icon_style: IconStyle::Rounded,
            density: UIDensity::Standard,
            compact_mode: false,
        }
    }
}

impl Default for BehaviorSettings {
    fn default() -> Self {
        Self {
            double_click_speed: 500.0,
            tooltip_delay: 1000.0,
            auto_save_interval: 300, // 5 minutes
            undo_history_size: 50,
            default_view_mode: ViewMode::Grid,
            confirm_destructive_actions: true,
            auto_play_videos: false,
            loop_audio: false,
            notification_settings: NotificationSettings::default(),
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            sound_enabled: true,
            show_on_screen: true,
            duration: 5.0,
            position: NotificationPosition::TopRight,
            categories: HashMap::new(),
        }
    }
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            analytics_enabled: false,
            crash_reporting: true,
            usage_statistics: false,
            location_sharing: false,
            peer_discovery: true,
            auto_connect: false,
            data_collection_level: DataCollectionLevel::Essential,
        }
    }
}

impl ConfigValidator {
    /// Validate configuration
    pub fn validate(config: &UIConfig) -> Result<()> {
        // Validate font scale
        if config.font_scale < 0.5 || config.font_scale > 3.0 {
            return Err(anyhow::anyhow!("Font scale must be between 0.5 and 3.0"));
        }
        
        // Validate animation scale
        if config.animation_scale < 0.0 || config.animation_scale > 2.0 {
            return Err(anyhow::anyhow!("Animation scale must be between 0.0 and 2.0"));
        }
        
        // Validate target FPS
        if config.settings.performance.target_fps < 15 || config.settings.performance.target_fps > 120 {
            return Err(anyhow::anyhow!("Target FPS must be between 15 and 120"));
        }
        
        // Validate battery threshold
        if config.settings.eco_mode.battery_threshold < 0.0 || config.settings.eco_mode.battery_threshold > 1.0 {
            return Err(anyhow::anyhow!("Battery threshold must be between 0.0 and 1.0"));
        }
        
        // Validate transparency
        if config.settings.appearance.window_transparency < 0.0 || config.settings.appearance.window_transparency > 1.0 {
            return Err(anyhow::anyhow!("Window transparency must be between 0.0 and 1.0"));
        }
        
        // Validate timeout values
        if config.settings.behavior.tooltip_delay < 0.0 || config.settings.behavior.tooltip_delay > 5000.0 {
            return Err(anyhow::anyhow!("Tooltip delay must be between 0 and 5000ms"));
        }
        
        Ok(())
    }
    
    /// Sanitize configuration values
    pub fn sanitize(config: &mut UIConfig) {
        // Clamp values to valid ranges
        config.font_scale = config.font_scale.clamp(0.5, 3.0);
        config.animation_scale = config.animation_scale.clamp(0.0, 2.0);
        config.settings.performance.target_fps = config.settings.performance.target_fps.clamp(15, 120);
        config.settings.eco_mode.battery_threshold = config.settings.eco_mode.battery_threshold.clamp(0.0, 1.0);
        config.settings.appearance.window_transparency = config.settings.appearance.window_transparency.clamp(0.0, 1.0);
        config.settings.behavior.tooltip_delay = config.settings.behavior.tooltip_delay.clamp(0.0, 5000.0);
        
        // Ensure consistent eco-mode settings
        if config.settings.eco_mode.enabled {
            config.settings.performance.target_fps = config.settings.performance.target_fps.min(60);
        }
    }
    
    /// Check for deprecated settings and migrate
    pub fn migrate(config: &mut UIConfig) -> Result<()> {
        // Future migration logic would go here
        // For now, just ensure we have all required fields
        
        // Ensure notification categories exist for core features
        let categories = &mut config.settings.behavior.notification_settings.categories;
        categories.entry("broadcasts".to_string()).or_insert(true);
        categories.entry("payments".to_string()).or_insert(true);
        categories.entry("network".to_string()).or_insert(false);
        categories.entry("system".to_string()).or_insert(true);
        
        Ok(())
    }
}

/// Configuration preset manager
pub struct ConfigPresets;

impl ConfigPresets {
    /// Get accessibility-focused preset
    pub fn accessibility_focused() -> UIConfig {
        let mut config = UIConfig::default();
        config.font_scale = 1.5;
        config.animation_scale = 0.2;
        config.settings.accessibility.high_contrast = true;
        config.settings.accessibility.large_text = true;
        config.settings.accessibility.reduced_motion = true;
        config.settings.accessibility.touch_target_size = TouchTargetSize::Large;
        config.settings.appearance.density = UIDensity::Comfortable;
        config
    }
    
    /// Get performance-focused preset
    pub fn performance_focused() -> UIConfig {
        let mut config = UIConfig::default();
        config.settings.performance.target_fps = 60;
        config.settings.performance.render_quality = RenderQuality::Medium;
        config.settings.performance.memory_management = MemoryManagement::Performance;
        config.settings.appearance.window_transparency = 0.0;
        config.settings.appearance.shadow_intensity = 0.5;
        config.animation_scale = 0.8;
        config
    }
    
    /// Get eco-friendly preset
    pub fn eco_friendly() -> UIConfig {
        let mut config = UIConfig::default();
        config.theme_name = "renaissance_dark".to_string();
        config.settings.eco_mode.enabled = true;
        config.settings.eco_mode.reduce_animations = true;
        config.settings.eco_mode.power_saving_colors = true;
        config.settings.performance.target_fps = 30;
        config.settings.performance.render_quality = RenderQuality::Low;
        config.animation_scale = 0.5;
        config
    }
    
    /// Get privacy-focused preset
    pub fn privacy_focused() -> UIConfig {
        let mut config = UIConfig::default();
        config.settings.privacy.analytics_enabled = false;
        config.settings.privacy.crash_reporting = false;
        config.settings.privacy.usage_statistics = false;
        config.settings.privacy.data_collection_level = DataCollectionLevel::None;
        config.settings.behavior.notification_settings.enabled = false;
        config
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme_name: "renaissance_natural".to_string(),
            language: "en".to_string(),
            font_scale: 1.0,
            animation_scale: 1.0,
            auto_theme: true,
            settings: UISettings {
                accessibility: AccessibilitySettings::from(&AccessibilityConfig {
                    high_contrast: false,
                    large_text: false,
                    reduced_motion: false,
                    screen_reader_support: true,
                    keyboard_navigation: true,
                    color_blind_friendly: true,
                }),
                performance: PerformanceSettings::from(&PerformanceConfig {
                    target_fps: 60,
                    vsync: true,
                    gpu_acceleration: true,
                    battery_optimization: true,
                    thermal_throttling: true,
                }),
                eco_mode: EcoModeSettings::from(&EcoModeConfig {
                    enabled: false,
                    dark_theme_preference: true,
                    reduce_animations: false,
                    lower_quality_images: false,
                    power_saving_colors: false,
                    minimal_refresh_rate: false,
                }),
                appearance: AppearanceSettings::default(),
                behavior: BehaviorSettings::default(),
                privacy: PrivacySettings::default(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = UIConfig::default();
        assert_eq!(config.theme_name, "renaissance_natural");
        assert_eq!(config.font_scale, 1.0);
        assert!(config.auto_theme);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = UIConfig::default();
        assert!(ConfigValidator::validate(&config).is_ok());
        
        // Test invalid font scale
        config.font_scale = 5.0;
        assert!(ConfigValidator::validate(&config).is_err());
        
        // Test sanitization
        ConfigValidator::sanitize(&mut config);
        assert_eq!(config.font_scale, 3.0); // Clamped to max
    }
    
    #[test]
    fn test_eco_mode_application() {
        let mut config = UIConfig::default();
        config.apply_eco_mode(true);
        
        assert!(config.settings.eco_mode.enabled);
        assert!(config.settings.performance.target_fps <= 30);
        assert_eq!(config.animation_scale, 0.5);
    }
    
    #[test]
    fn test_accessibility_application() {
        let mut config = UIConfig::default();
        let accessibility_config = AccessibilityConfig {
            high_contrast: true,
            large_text: true,
            reduced_motion: true,
            screen_reader_support: true,
            keyboard_navigation: true,
            color_blind_friendly: true,
        };
        
        config.apply_accessibility(&accessibility_config);
        
        assert!(config.settings.accessibility.high_contrast);
        assert!(config.settings.accessibility.large_text);
        assert_eq!(config.font_scale, 1.25);
        assert_eq!(config.animation_scale, 0.2);
    }
    
    #[test]
    fn test_config_presets() {
        let eco_preset = ConfigPresets::eco_friendly();
        assert!(eco_preset.settings.eco_mode.enabled);
        assert_eq!(eco_preset.theme_name, "renaissance_dark");
        
        let accessibility_preset = ConfigPresets::accessibility_focused();
        assert!(accessibility_preset.settings.accessibility.high_contrast);
        assert_eq!(accessibility_preset.font_scale, 1.5);
        
        let performance_preset = ConfigPresets::performance_focused();
        assert_eq!(performance_preset.settings.performance.target_fps, 60);
        
        let privacy_preset = ConfigPresets::privacy_focused();
        assert!(!privacy_preset.settings.privacy.analytics_enabled);
    }
    
    #[test]
    fn test_effective_values() {
        let mut config = UIConfig::default();
        config.settings.accessibility.large_text = true;
        
        let effective_scale = config.effective_font_scale();
        assert!(effective_scale > 1.0);
        
        config.apply_eco_mode(true);
        let effective_fps = config.effective_target_fps();
        assert!(effective_fps <= 30);
        
        assert!(config.should_reduce_animations());
    }
}