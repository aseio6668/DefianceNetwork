//! Renaissance-inspired theming system with eco-friendly considerations

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Renaissance theme manager
pub struct ThemeManager {
    current_theme: Option<RenaissanceTheme>,
    available_themes: HashMap<String, RenaissanceTheme>,
    theme_cache: HashMap<String, RenaissanceTheme>,
}

/// Renaissance-inspired color palette
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    // Primary colors inspired by Renaissance art
    pub primary: RgbaColor,           // Deep renaissance blue
    pub secondary: RgbaColor,         // Warm terracotta
    pub accent: RgbaColor,            // Golden highlight
    
    // Surface colors
    pub background: RgbaColor,        // Parchment/canvas
    pub surface: RgbaColor,           // Slightly darker surface
    pub surface_variant: RgbaColor,   // Card/panel background
    
    // Text colors
    pub on_primary: RgbaColor,        // Text on primary
    pub on_secondary: RgbaColor,      // Text on secondary
    pub on_background: RgbaColor,     // Main text color
    pub on_surface: RgbaColor,        // Surface text
    
    // State colors
    pub success: RgbaColor,           // Natural green
    pub warning: RgbaColor,           // Amber
    pub error: RgbaColor,             // Muted red
    pub info: RgbaColor,              // Soft blue
    
    // Eco-friendly variants
    pub power_saving: RgbaColor,      // Dark mode background
    pub low_contrast: RgbaColor,      // Reduced eye strain
}

/// RGBA color representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RgbaColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

/// Typography settings inspired by Renaissance manuscripts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Typography {
    // Font families
    pub heading_family: String,       // Serif for headings
    pub body_family: String,          // Sans-serif for readability
    pub monospace_family: String,     // Code/technical text
    
    // Font sizes (in pixels)
    pub display_large: f32,           // 48px - Main headings
    pub display_medium: f32,          // 36px - Section headers
    pub display_small: f32,           // 24px - Subsections
    pub headline_large: f32,          // 20px - Card titles
    pub headline_medium: f32,         // 18px - List headers
    pub headline_small: f32,          // 16px - Small headers
    pub body_large: f32,              // 16px - Main text
    pub body_medium: f32,             // 14px - Secondary text
    pub body_small: f32,              // 12px - Captions
    pub label_large: f32,             // 14px - Button text
    pub label_medium: f32,            // 12px - Input labels
    pub label_small: f32,             // 10px - Small labels
    
    // Line heights
    pub line_height_tight: f32,       // 1.2
    pub line_height_normal: f32,      // 1.5
    pub line_height_relaxed: f32,     // 1.8
    
    // Letter spacing
    pub letter_spacing_tight: f32,    // -0.01em
    pub letter_spacing_normal: f32,   // 0em
    pub letter_spacing_wide: f32,     // 0.05em
}

/// Renaissance-inspired spacing system based on golden ratio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spacing {
    pub xs: f32,        // 4px
    pub sm: f32,        // 8px
    pub md: f32,        // 16px
    pub lg: f32,        // 24px
    pub xl: f32,        // 32px
    pub xxl: f32,       // 48px
    pub xxxl: f32,      // 64px
    
    // Golden ratio based spacing
    pub golden_small: f32,    // 13px (8 * 1.618)
    pub golden_medium: f32,   // 26px (16 * 1.618)
    pub golden_large: f32,    // 42px (26 * 1.618)
}

/// Border radius settings for organic, natural feel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderRadius {
    pub none: f32,      // 0px
    pub small: f32,     // 4px
    pub medium: f32,    // 8px
    pub large: f32,     // 16px
    pub xl: f32,        // 24px
    pub circular: f32,  // 50%
}

/// Shadow settings for depth and hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shadows {
    pub none: ShadowStyle,
    pub small: ShadowStyle,
    pub medium: ShadowStyle,
    pub large: ShadowStyle,
    pub xl: ShadowStyle,
}

/// Individual shadow style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowStyle {
    pub x: f32,
    pub y: f32,
    pub blur: f32,
    pub spread: f32,
    pub color: RgbaColor,
}

/// Complete Renaissance theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenaissanceTheme {
    pub name: String,
    pub description: String,
    pub version: String,
    pub is_dark: bool,
    pub is_eco_friendly: bool,
    
    pub colors: ColorPalette,
    pub typography: Typography,
    pub spacing: Spacing,
    pub border_radius: BorderRadius,
    pub shadows: Shadows,
    
    // Animation timings
    pub animation_duration_fast: f32,    // 150ms
    pub animation_duration_normal: f32,  // 300ms
    pub animation_duration_slow: f32,    // 500ms
    
    // Eco-mode settings
    pub eco_mode_enabled: bool,
    pub power_saving_colors: bool,
    pub reduced_animations: bool,
}

impl RgbaColor {
    /// Create new RGBA color
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    
    /// Create from hex string
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 && hex.len() != 8 {
            return Err(anyhow::anyhow!("Invalid hex color format"));
        }
        
        let r = u8::from_str_radix(&hex[0..2], 16)? as f32 / 255.0;
        let g = u8::from_str_radix(&hex[2..4], 16)? as f32 / 255.0;
        let b = u8::from_str_radix(&hex[4..6], 16)? as f32 / 255.0;
        let a = if hex.len() == 8 {
            u8::from_str_radix(&hex[6..8], 16)? as f32 / 255.0
        } else {
            1.0
        };
        
        Ok(Self::new(r, g, b, a))
    }
    
    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        let r = (self.r * 255.0) as u8;
        let g = (self.g * 255.0) as u8;
        let b = (self.b * 255.0) as u8;
        let a = (self.a * 255.0) as u8;
        
        if a == 255 {
            format!("#{:02x}{:02x}{:02x}", r, g, b)
        } else {
            format!("#{:02x}{:02x}{:02x}{:02x}", r, g, b, a)
        }
    }
    
    /// Create a lighter variant
    pub fn lighten(&self, amount: f32) -> Self {
        Self {
            r: (self.r + amount).min(1.0),
            g: (self.g + amount).min(1.0),
            b: (self.b + amount).min(1.0),
            a: self.a,
        }
    }
    
    /// Create a darker variant
    pub fn darken(&self, amount: f32) -> Self {
        Self {
            r: (self.r - amount).max(0.0),
            g: (self.g - amount).max(0.0),
            b: (self.b - amount).max(0.0),
            a: self.a,
        }
    }
    
    /// Create a more transparent variant
    pub fn with_alpha(&self, alpha: f32) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a: alpha.clamp(0.0, 1.0),
        }
    }
}

impl ColorPalette {
    /// Create natural/light Renaissance palette
    pub fn renaissance_natural() -> Self {
        Self {
            primary: RgbaColor::from_hex("#2E5B8A").unwrap(),       // Renaissance blue
            secondary: RgbaColor::from_hex("#B8860B").unwrap(),     // Dark goldenrod
            accent: RgbaColor::from_hex("#DAA520").unwrap(),        // Goldenrod
            
            background: RgbaColor::from_hex("#FAF7F0").unwrap(),    // Parchment
            surface: RgbaColor::from_hex("#F5F1E8").unwrap(),       // Light canvas
            surface_variant: RgbaColor::from_hex("#E8E2D4").unwrap(), // Card background
            
            on_primary: RgbaColor::from_hex("#FFFFFF").unwrap(),
            on_secondary: RgbaColor::from_hex("#FFFFFF").unwrap(),
            on_background: RgbaColor::from_hex("#1A1A1A").unwrap(),
            on_surface: RgbaColor::from_hex("#2D2D2D").unwrap(),
            
            success: RgbaColor::from_hex("#5D8A3A").unwrap(),       // Natural green
            warning: RgbaColor::from_hex("#D2691E").unwrap(),       // Chocolate/amber
            error: RgbaColor::from_hex("#A0522D").unwrap(),         // Sienna
            info: RgbaColor::from_hex("#4682B4").unwrap(),          // Steel blue
            
            power_saving: RgbaColor::from_hex("#1A1A1A").unwrap(),
            low_contrast: RgbaColor::from_hex("#6B6B6B").unwrap(),
        }
    }
    
    /// Create dark Renaissance palette for eco-mode
    pub fn renaissance_dark() -> Self {
        Self {
            primary: RgbaColor::from_hex("#4A7BA7").unwrap(),       // Lighter blue
            secondary: RgbaColor::from_hex("#CD853F").unwrap(),     // Peru
            accent: RgbaColor::from_hex("#F4A460").unwrap(),        // Sandy brown
            
            background: RgbaColor::from_hex("#0D1117").unwrap(),    // Very dark
            surface: RgbaColor::from_hex("#161B22").unwrap(),       // Dark surface
            surface_variant: RgbaColor::from_hex("#21262D").unwrap(), // Card background
            
            on_primary: RgbaColor::from_hex("#FFFFFF").unwrap(),
            on_secondary: RgbaColor::from_hex("#000000").unwrap(),
            on_background: RgbaColor::from_hex("#F0F6FC").unwrap(),
            on_surface: RgbaColor::from_hex("#E6EDF3").unwrap(),
            
            success: RgbaColor::from_hex("#7FB069").unwrap(),       // Soft green
            warning: RgbaColor::from_hex("#F0C674").unwrap(),       // Soft amber
            error: RgbaColor::from_hex("#F07178").unwrap(),         // Soft red
            info: RgbaColor::from_hex("#82B1FF").unwrap(),          // Light blue
            
            power_saving: RgbaColor::from_hex("#000000").unwrap(),
            low_contrast: RgbaColor::from_hex("#4A4A4A").unwrap(),
        }
    }
    
    /// Create minimal eco-friendly palette
    pub fn renaissance_minimal() -> Self {
        Self {
            primary: RgbaColor::from_hex("#333333").unwrap(),
            secondary: RgbaColor::from_hex("#666666").unwrap(),
            accent: RgbaColor::from_hex("#999999").unwrap(),
            
            background: RgbaColor::from_hex("#000000").unwrap(),
            surface: RgbaColor::from_hex("#0A0A0A").unwrap(),
            surface_variant: RgbaColor::from_hex("#1A1A1A").unwrap(),
            
            on_primary: RgbaColor::from_hex("#FFFFFF").unwrap(),
            on_secondary: RgbaColor::from_hex("#FFFFFF").unwrap(),
            on_background: RgbaColor::from_hex("#CCCCCC").unwrap(),
            on_surface: RgbaColor::from_hex("#AAAAAA").unwrap(),
            
            success: RgbaColor::from_hex("#666666").unwrap(),
            warning: RgbaColor::from_hex("#888888").unwrap(),
            error: RgbaColor::from_hex("#AAAAAA").unwrap(),
            info: RgbaColor::from_hex("#777777").unwrap(),
            
            power_saving: RgbaColor::from_hex("#000000").unwrap(),
            low_contrast: RgbaColor::from_hex("#333333").unwrap(),
        }
    }
}

impl Typography {
    /// Create default Renaissance typography
    pub fn renaissance_default() -> Self {
        Self {
            heading_family: "Playfair Display, serif".to_string(),
            body_family: "Inter, sans-serif".to_string(),
            monospace_family: "JetBrains Mono, monospace".to_string(),
            
            display_large: 48.0,
            display_medium: 36.0,
            display_small: 24.0,
            headline_large: 20.0,
            headline_medium: 18.0,
            headline_small: 16.0,
            body_large: 16.0,
            body_medium: 14.0,
            body_small: 12.0,
            label_large: 14.0,
            label_medium: 12.0,
            label_small: 10.0,
            
            line_height_tight: 1.2,
            line_height_normal: 1.5,
            line_height_relaxed: 1.8,
            
            letter_spacing_tight: -0.01,
            letter_spacing_normal: 0.0,
            letter_spacing_wide: 0.05,
        }
    }
}

impl Spacing {
    /// Create default spacing based on 8px grid and golden ratio
    pub fn renaissance_default() -> Self {
        Self {
            xs: 4.0,
            sm: 8.0,
            md: 16.0,
            lg: 24.0,
            xl: 32.0,
            xxl: 48.0,
            xxxl: 64.0,
            
            golden_small: 13.0,   // 8 * 1.618
            golden_medium: 26.0,  // 16 * 1.618
            golden_large: 42.0,   // 26 * 1.618
        }
    }
}

impl BorderRadius {
    /// Create default border radius values
    pub fn renaissance_default() -> Self {
        Self {
            none: 0.0,
            small: 4.0,
            medium: 8.0,
            large: 16.0,
            xl: 24.0,
            circular: 50.0,
        }
    }
}

impl Shadows {
    /// Create default shadow system
    pub fn renaissance_default() -> Self {
        Self {
            none: ShadowStyle {
                x: 0.0, y: 0.0, blur: 0.0, spread: 0.0,
                color: RgbaColor::new(0.0, 0.0, 0.0, 0.0),
            },
            small: ShadowStyle {
                x: 0.0, y: 1.0, blur: 3.0, spread: 0.0,
                color: RgbaColor::new(0.0, 0.0, 0.0, 0.1),
            },
            medium: ShadowStyle {
                x: 0.0, y: 4.0, blur: 6.0, spread: -1.0,
                color: RgbaColor::new(0.0, 0.0, 0.0, 0.15),
            },
            large: ShadowStyle {
                x: 0.0, y: 10.0, blur: 15.0, spread: -3.0,
                color: RgbaColor::new(0.0, 0.0, 0.0, 0.2),
            },
            xl: ShadowStyle {
                x: 0.0, y: 20.0, blur: 25.0, spread: -5.0,
                color: RgbaColor::new(0.0, 0.0, 0.0, 0.25),
            },
        }
    }
}

impl RenaissanceTheme {
    /// Create natural Renaissance theme
    pub fn renaissance_natural() -> Self {
        Self {
            name: "Renaissance Natural".to_string(),
            description: "Light theme inspired by Renaissance art and natural materials".to_string(),
            version: "1.0.0".to_string(),
            is_dark: false,
            is_eco_friendly: true,
            
            colors: ColorPalette::renaissance_natural(),
            typography: Typography::renaissance_default(),
            spacing: Spacing::renaissance_default(),
            border_radius: BorderRadius::renaissance_default(),
            shadows: Shadows::renaissance_default(),
            
            animation_duration_fast: 150.0,
            animation_duration_normal: 300.0,
            animation_duration_slow: 500.0,
            
            eco_mode_enabled: false,
            power_saving_colors: false,
            reduced_animations: false,
        }
    }
    
    /// Create dark Renaissance theme for eco-mode
    pub fn renaissance_dark() -> Self {
        let mut theme = Self::renaissance_natural();
        theme.name = "Renaissance Dark".to_string();
        theme.description = "Dark theme for reduced eye strain and power consumption".to_string();
        theme.is_dark = true;
        theme.colors = ColorPalette::renaissance_dark();
        theme.eco_mode_enabled = true;
        theme.power_saving_colors = true;
        theme
    }
    
    /// Create minimal eco-friendly theme
    pub fn renaissance_minimal() -> Self {
        let mut theme = Self::renaissance_dark();
        theme.name = "Renaissance Minimal".to_string();
        theme.description = "Ultra-minimal theme for maximum power efficiency".to_string();
        theme.colors = ColorPalette::renaissance_minimal();
        theme.reduced_animations = true;
        theme.animation_duration_fast = 100.0;
        theme.animation_duration_normal = 200.0;
        theme.animation_duration_slow = 300.0;
        theme
    }
}

impl ThemeManager {
    /// Create new theme manager
    pub fn new() -> Result<Self> {
        let mut available_themes = HashMap::new();
        
        // Register built-in themes
        available_themes.insert(
            "renaissance_natural".to_string(),
            RenaissanceTheme::renaissance_natural()
        );
        available_themes.insert(
            "renaissance_dark".to_string(),
            RenaissanceTheme::renaissance_dark()
        );
        available_themes.insert(
            "renaissance_minimal".to_string(),
            RenaissanceTheme::renaissance_minimal()
        );
        
        Ok(Self {
            current_theme: None,
            available_themes,
            theme_cache: HashMap::new(),
        })
    }
    
    /// Load a theme by name
    pub fn load_theme(&mut self, name: &str) -> Result<()> {
        if let Some(theme) = self.available_themes.get(name) {
            self.current_theme = Some(theme.clone());
            tracing::info!("Loaded theme: {}", name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Theme '{}' not found", name))
        }
    }
    
    /// Switch to a different theme
    pub fn switch_theme(&mut self, name: &str) -> Result<()> {
        self.load_theme(name)
    }
    
    /// Get current theme
    pub fn current_theme(&self) -> Option<&RenaissanceTheme> {
        self.current_theme.as_ref()
    }
    
    /// Get available theme names
    pub fn available_themes(&self) -> Vec<String> {
        self.available_themes.keys().cloned().collect()
    }
    
    /// Register a custom theme
    pub fn register_theme(&mut self, name: String, theme: RenaissanceTheme) {
        self.available_themes.insert(name.clone(), theme);
        tracing::info!("Registered custom theme: {}", name);
    }
    
    /// Apply power-saving color palette
    pub fn apply_power_saving_palette(&mut self) -> Result<()> {
        if let Some(ref mut theme) = self.current_theme {
            theme.power_saving_colors = true;
            // Reduce color saturation and brightness for power saving
            theme.colors.primary = theme.colors.primary.darken(0.2);
            theme.colors.secondary = theme.colors.secondary.darken(0.2);
            theme.colors.accent = theme.colors.accent.darken(0.2);
            
            tracing::info!("Applied power-saving color palette");
        }
        Ok(())
    }
    
    /// Auto-detect user's preferred theme based on system
    pub fn detect_preferred_theme(&self) -> String {
        // This would typically check system dark mode preference
        // For now, default to natural theme
        "renaissance_natural".to_string()
    }
    
    /// Save theme to file
    pub fn save_theme(&self, theme: &RenaissanceTheme, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(theme)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load theme from file
    pub fn load_theme_from_file(&mut self, path: &str) -> Result<RenaissanceTheme> {
        let content = std::fs::read_to_string(path)?;
        let theme: RenaissanceTheme = serde_json::from_str(&content)?;
        Ok(theme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rgba_color_creation() {
        let color = RgbaColor::new(1.0, 0.5, 0.0, 1.0);
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.5);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 1.0);
    }
    
    #[test]
    fn test_hex_color_parsing() {
        let color = RgbaColor::from_hex("#FF8000").unwrap();
        assert!((color.r - 1.0).abs() < 0.01);
        assert!((color.g - 0.5).abs() < 0.01);
        assert!((color.b - 0.0).abs() < 0.01);
        assert_eq!(color.a, 1.0);
    }
    
    #[test]
    fn test_color_manipulation() {
        let color = RgbaColor::from_hex("#808080").unwrap();
        let lighter = color.lighten(0.2);
        let darker = color.darken(0.2);
        let transparent = color.with_alpha(0.5);
        
        assert!(lighter.r > color.r);
        assert!(darker.r < color.r);
        assert_eq!(transparent.a, 0.5);
    }
    
    #[test]
    fn test_theme_creation() {
        let theme = RenaissanceTheme::renaissance_natural();
        assert_eq!(theme.name, "Renaissance Natural");
        assert!(!theme.is_dark);
        assert!(theme.is_eco_friendly);
    }
    
    #[test]
    fn test_theme_manager() {
        let mut manager = ThemeManager::new().unwrap();
        assert!(manager.load_theme("renaissance_natural").is_ok());
        assert!(manager.current_theme().is_some());
        
        let themes = manager.available_themes();
        assert!(themes.contains(&"renaissance_natural".to_string()));
        assert!(themes.contains(&"renaissance_dark".to_string()));
    }
    
    #[test]
    fn test_color_palettes() {
        let natural = ColorPalette::renaissance_natural();
        let dark = ColorPalette::renaissance_dark();
        let minimal = ColorPalette::renaissance_minimal();
        
        // Dark theme should have darker background
        assert!(dark.background.r < natural.background.r);
        
        // Minimal theme should be very dark
        assert!(minimal.background.r < dark.background.r);
    }
}