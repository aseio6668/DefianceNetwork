//! UI components for the Renaissance framework

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use crate::theme::{RenaissanceTheme, RgbaColor};

/// Base component trait that all UI elements implement
pub trait Component: Send + Sync {
    /// Update component state
    fn update(&mut self, delta_time: f32) -> Result<()>;
    
    /// Render component (platform-specific implementation)
    fn render(&self, context: &mut dyn RenderContext) -> Result<()>;
    
    /// Handle input events
    fn handle_input(&mut self, event: &InputEvent) -> Result<bool>;
    
    /// Get component bounds
    fn bounds(&self) -> Rect;
    
    /// Set component bounds
    fn set_bounds(&mut self, bounds: Rect);
    
    /// Get component ID
    fn id(&self) -> Uuid;
    
    /// Get component type name
    fn type_name(&self) -> &'static str;
    
    /// Check if component is visible
    fn is_visible(&self) -> bool;
    
    /// Set component visibility
    fn set_visible(&mut self, visible: bool);
    
    /// Apply theme to component
    fn apply_theme(&mut self, theme: &RenaissanceTheme);
}

/// Render context trait for platform abstraction
pub trait RenderContext {
    /// Draw rectangle
    fn draw_rect(&mut self, rect: Rect, color: RgbaColor, border_radius: f32);
    
    /// Draw text
    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle);
    
    /// Draw image
    fn draw_image(&mut self, image_id: &str, rect: Rect, opacity: f32);
    
    /// Set clip region
    fn set_clip(&mut self, rect: Rect);
    
    /// Clear clip region
    fn clear_clip(&mut self);
    
    /// Get text dimensions
    fn text_size(&self, text: &str, style: &TextStyle) -> Size;
}

/// Component styling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStyle {
    pub background_color: Option<RgbaColor>,
    pub border_color: Option<RgbaColor>,
    pub border_width: f32,
    pub border_radius: f32,
    pub padding: Padding,
    pub margin: Margin,
    pub shadow: Option<crate::theme::ShadowStyle>,
    pub opacity: f32,
}

/// Text styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStyle {
    pub font_family: String,
    pub font_size: f32,
    pub font_weight: FontWeight,
    pub color: RgbaColor,
    pub line_height: f32,
    pub letter_spacing: f32,
    pub text_align: TextAlign,
}

/// Font weight options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Thin,
    Light,
    Regular,
    Medium,
    SemiBold,
    Bold,
    ExtraBold,
    Black,
}

/// Text alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

/// Geometric types
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Padding {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Margin {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

/// Input events
#[derive(Debug, Clone)]
pub enum InputEvent {
    MouseMove { position: Point },
    MouseDown { position: Point, button: MouseButton },
    MouseUp { position: Point, button: MouseButton },
    KeyDown { key: KeyCode, modifiers: KeyModifiers },
    KeyUp { key: KeyCode, modifiers: KeyModifiers },
    TextInput { text: String },
    Scroll { delta: Point },
    Touch { touches: Vec<TouchPoint> },
}

#[derive(Debug, Clone)]
pub struct TouchPoint {
    pub id: u64,
    pub position: Point,
    pub pressure: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyCode {
    Space,
    Enter,
    Escape,
    Backspace,
    Delete,
    Tab,
    Up,
    Down,
    Left,
    Right,
    // Add more as needed
    Letter(char),
    Number(u8),
}

#[derive(Debug, Clone, Copy)]
pub struct KeyModifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
}

/// UI element abstraction
pub trait UIElement {
    /// Get element type
    fn element_type(&self) -> &'static str;
    
    /// Get element properties
    fn properties(&self) -> HashMap<String, String>;
    
    /// Set element property
    fn set_property(&mut self, key: &str, value: String);
}

/// Button component
pub struct Button {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    text: String,
    style: ComponentStyle,
    text_style: TextStyle,
    state: ButtonState,
    on_click: Option<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ButtonState {
    Normal,
    Hovered,
    Pressed,
    Disabled,
}

/// Text component
pub struct Text {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    content: String,
    style: ComponentStyle,
    text_style: TextStyle,
    selectable: bool,
}

/// Input field component
pub struct Input {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    value: String,
    placeholder: String,
    style: ComponentStyle,
    text_style: TextStyle,
    focused: bool,
    cursor_position: usize,
    selection_start: Option<usize>,
    selection_end: Option<usize>,
    on_change: Option<Box<dyn Fn(&str) + Send + Sync>>,
}

/// Video player component for DefianceNetwork streams
pub struct VideoPlayer {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    style: ComponentStyle,
    stream_url: Option<String>,
    is_playing: bool,
    volume: f32,
    current_time: f32,
    duration: f32,
    quality: VideoQuality,
    controls_visible: bool,
    on_play: Option<Box<dyn Fn() + Send + Sync>>,
    on_pause: Option<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VideoQuality {
    Audio,
    Low,
    Medium,
    High,
    Ultra,
}

/// Audio player component for Audigy streams
pub struct AudioPlayer {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    style: ComponentStyle,
    track_info: Option<TrackInfo>,
    is_playing: bool,
    volume: f32,
    current_time: f32,
    duration: f32,
    visualization_enabled: bool,
    on_play: Option<Box<dyn Fn() + Send + Sync>>,
    on_pause: Option<Box<dyn Fn() + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub title: String,
    pub artist: String,
    pub album: Option<String>,
    pub duration: f32,
    pub cover_art: Option<String>,
}

/// Broadcast list component
pub struct BroadcastList {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    style: ComponentStyle,
    broadcasts: Vec<BroadcastItem>,
    selected_index: Option<usize>,
    scroll_offset: f32,
    item_height: f32,
    on_select: Option<Box<dyn Fn(&BroadcastItem) + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct BroadcastItem {
    pub id: Uuid,
    pub title: String,
    pub broadcaster: String,
    pub viewer_count: u32,
    pub quality: VideoQuality,
    pub thumbnail: Option<String>,
    pub duration: Option<f32>,
    pub category: String,
}

/// Peer status widget
pub struct PeerStatus {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    style: ComponentStyle,
    connected_peers: u32,
    network_health: f32,
    upload_speed: f32,
    download_speed: f32,
    last_update: f32,
}

/// Payment widget for Paradigm integration
pub struct PaymentWidget {
    id: Uuid,
    bounds: Rect,
    visible: bool,
    style: ComponentStyle,
    balance: f64,
    currency: String,
    recent_transactions: Vec<TransactionItem>,
    on_payment: Option<Box<dyn Fn(f64) + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct TransactionItem {
    pub id: String,
    pub amount: f64,
    pub description: String,
    pub timestamp: i64,
    pub status: String,
}

// Implementation helpers
impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Size {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }
    
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
    
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x && point.x <= self.x + self.width &&
        point.y >= self.y && point.y <= self.y + self.height
    }
    
    pub fn center(&self) -> Point {
        Point::new(
            self.x + self.width / 2.0,
            self.y + self.height / 2.0
        )
    }
}

impl Padding {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
    }
    
    pub fn uniform(value: f32) -> Self {
        Self::new(value, value, value, value)
    }
    
    pub fn zero() -> Self {
        Self::uniform(0.0)
    }
}

impl Margin {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
    }
    
    pub fn uniform(value: f32) -> Self {
        Self::new(value, value, value, value)
    }
    
    pub fn zero() -> Self {
        Self::uniform(0.0)
    }
}

impl Default for ComponentStyle {
    fn default() -> Self {
        Self {
            background_color: None,
            border_color: None,
            border_width: 0.0,
            border_radius: 0.0,
            padding: Padding::zero(),
            margin: Margin::zero(),
            shadow: None,
            opacity: 1.0,
        }
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_family: "Inter, sans-serif".to_string(),
            font_size: 14.0,
            font_weight: FontWeight::Regular,
            color: RgbaColor::new(0.0, 0.0, 0.0, 1.0),
            line_height: 1.5,
            letter_spacing: 0.0,
            text_align: TextAlign::Left,
        }
    }
}

impl KeyModifiers {
    pub fn none() -> Self {
        Self {
            ctrl: false,
            alt: false,
            shift: false,
            meta: false,
        }
    }
}

// Component factory functions
pub fn create_button() -> Box<dyn Component> {
    Box::new(Button {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        text: "Button".to_string(),
        style: ComponentStyle::default(),
        text_style: TextStyle::default(),
        state: ButtonState::Normal,
        on_click: None,
    })
}

pub fn create_text() -> Box<dyn Component> {
    Box::new(Text {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        content: "Text".to_string(),
        style: ComponentStyle::default(),
        text_style: TextStyle::default(),
        selectable: false,
    })
}

pub fn create_input() -> Box<dyn Component> {
    Box::new(Input {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        value: String::new(),
        placeholder: "Enter text...".to_string(),
        style: ComponentStyle::default(),
        text_style: TextStyle::default(),
        focused: false,
        cursor_position: 0,
        selection_start: None,
        selection_end: None,
        on_change: None,
    })
}

pub fn create_video_player() -> Box<dyn Component> {
    Box::new(VideoPlayer {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        style: ComponentStyle::default(),
        stream_url: None,
        is_playing: false,
        volume: 1.0,
        current_time: 0.0,
        duration: 0.0,
        quality: VideoQuality::Medium,
        controls_visible: true,
        on_play: None,
        on_pause: None,
    })
}

pub fn create_audio_player() -> Box<dyn Component> {
    Box::new(AudioPlayer {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        style: ComponentStyle::default(),
        track_info: None,
        is_playing: false,
        volume: 1.0,
        current_time: 0.0,
        duration: 0.0,
        visualization_enabled: true,
        on_play: None,
        on_pause: None,
    })
}

pub fn create_broadcast_list() -> Box<dyn Component> {
    Box::new(BroadcastList {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        style: ComponentStyle::default(),
        broadcasts: Vec::new(),
        selected_index: None,
        scroll_offset: 0.0,
        item_height: 80.0,
        on_select: None,
    })
}

pub fn create_peer_status() -> Box<dyn Component> {
    Box::new(PeerStatus {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        style: ComponentStyle::default(),
        connected_peers: 0,
        network_health: 1.0,
        upload_speed: 0.0,
        download_speed: 0.0,
        last_update: 0.0,
    })
}

pub fn create_payment_widget() -> Box<dyn Component> {
    Box::new(PaymentWidget {
        id: Uuid::new_v4(),
        bounds: Rect::zero(),
        visible: true,
        style: ComponentStyle::default(),
        balance: 0.0,
        currency: "PAR".to_string(),
        recent_transactions: Vec::new(),
        on_payment: None,
    })
}

// Component trait implementations would go here for each component type
// This is a simplified example for Button:

impl Component for Button {
    fn update(&mut self, _delta_time: f32) -> Result<()> {
        // Update button animations, state changes, etc.
        Ok(())
    }
    
    fn render(&self, context: &mut dyn RenderContext) -> Result<()> {
        // Render button background
        let bg_color = match self.state {
            ButtonState::Normal => self.style.background_color.unwrap_or(RgbaColor::new(0.9, 0.9, 0.9, 1.0)),
            ButtonState::Hovered => self.style.background_color.unwrap_or(RgbaColor::new(0.8, 0.8, 0.8, 1.0)),
            ButtonState::Pressed => self.style.background_color.unwrap_or(RgbaColor::new(0.7, 0.7, 0.7, 1.0)),
            ButtonState::Disabled => self.style.background_color.unwrap_or(RgbaColor::new(0.5, 0.5, 0.5, 0.5)),
        };
        
        context.draw_rect(self.bounds, bg_color, self.style.border_radius);
        
        // Render button text
        let text_pos = Point::new(
            self.bounds.x + self.bounds.width / 2.0,
            self.bounds.y + self.bounds.height / 2.0
        );
        context.draw_text(&self.text, text_pos, &self.text_style);
        
        Ok(())
    }
    
    fn handle_input(&mut self, event: &InputEvent) -> Result<bool> {
        match event {
            InputEvent::MouseMove { position } => {
                if self.bounds.contains(*position) {
                    if self.state == ButtonState::Normal {
                        self.state = ButtonState::Hovered;
                    }
                } else if self.state == ButtonState::Hovered {
                    self.state = ButtonState::Normal;
                }
                Ok(true)
            }
            InputEvent::MouseDown { position, button: MouseButton::Left } => {
                if self.bounds.contains(*position) {
                    self.state = ButtonState::Pressed;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            InputEvent::MouseUp { position, button: MouseButton::Left } => {
                if self.state == ButtonState::Pressed && self.bounds.contains(*position) {
                    if let Some(ref callback) = self.on_click {
                        callback();
                    }
                    self.state = ButtonState::Hovered;
                    Ok(true)
                } else {
                    self.state = ButtonState::Normal;
                    Ok(false)
                }
            }
            _ => Ok(false)
        }
    }
    
    fn bounds(&self) -> Rect {
        self.bounds
    }
    
    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
    
    fn id(&self) -> Uuid {
        self.id
    }
    
    fn type_name(&self) -> &'static str {
        "Button"
    }
    
    fn is_visible(&self) -> bool {
        self.visible
    }
    
    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    
    fn apply_theme(&mut self, theme: &RenaissanceTheme) {
        self.style.background_color = Some(theme.colors.primary);
        self.text_style.color = theme.colors.on_primary;
        self.text_style.font_family = theme.typography.body_family.clone();
        self.text_style.font_size = theme.typography.label_large;
        self.style.border_radius = theme.border_radius.medium;
        self.style.padding = Padding::uniform(theme.spacing.md);
    }
}

// Similar implementations would be added for all other component types
// This provides a complete foundation for the UI component system

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_point_creation() {
        let point = Point::new(10.0, 20.0);
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 20.0);
    }
    
    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        assert!(rect.contains(Point::new(50.0, 50.0)));
        assert!(!rect.contains(Point::new(150.0, 50.0)));
    }
    
    #[test]
    fn test_button_creation() {
        let button = create_button();
        assert_eq!(button.type_name(), "Button");
        assert!(button.is_visible());
    }
    
    #[test]
    fn test_component_style_default() {
        let style = ComponentStyle::default();
        assert_eq!(style.opacity, 1.0);
        assert_eq!(style.border_width, 0.0);
    }
}