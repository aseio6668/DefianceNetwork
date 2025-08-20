//! Layout management system for Renaissance UI

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use crate::components::{Rect, Point, Size, Margin, Padding};
use crate::theme::Spacing;

/// Layout manager for organizing UI components
pub struct LayoutManager {
    layouts: HashMap<Uuid, Box<dyn Layout>>,
    root_layout: Option<Uuid>,
    constraints: HashMap<Uuid, LayoutConstraints>,
}

/// Layout trait for different layout algorithms
pub trait Layout: Send + Sync {
    /// Calculate layout for children
    fn layout(&self, container: Rect, children: &mut [LayoutItem]) -> Result<()>;
    
    /// Get layout type name
    fn layout_type(&self) -> &'static str;
    
    /// Get minimum size required
    fn min_size(&self, children: &[LayoutItem]) -> Size;
    
    /// Get preferred size
    fn preferred_size(&self, children: &[LayoutItem]) -> Size;
    
    /// Check if layout is responsive
    fn is_responsive(&self) -> bool;
}

/// Layout item containing a component and its constraints
#[derive(Debug, Clone)]
pub struct LayoutItem {
    pub component_id: Uuid,
    pub bounds: Rect,
    pub constraints: LayoutConstraints,
    pub margin: Margin,
    pub padding: Padding,
    pub visible: bool,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub flex_basis: FlexBasis,
}

/// Layout constraints for responsive design
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConstraints {
    pub min_width: Option<f32>,
    pub max_width: Option<f32>,
    pub min_height: Option<f32>,
    pub max_height: Option<f32>,
    pub aspect_ratio: Option<f32>,
    pub alignment: Alignment,
    pub position: Position,
}

/// Alignment options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Alignment {
    Start,
    Center,
    End,
    Stretch,
    Baseline,
}

/// Positioning options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Position {
    Static,
    Relative { x: f32, y: f32 },
    Absolute { x: f32, y: f32 },
    Fixed { x: f32, y: f32 },
}

/// Layout direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutDirection {
    Row,
    Column,
    RowReverse,
    ColumnReverse,
}

/// Flex basis for flexible layouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlexBasis {
    Auto,
    Content,
    Pixels(f32),
    Percentage(f32),
}

/// Wrap behavior for flex layouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlexWrap {
    NoWrap,
    Wrap,
    WrapReverse,
}

/// Justify content options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JustifyContent {
    FlexStart,
    FlexEnd,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

/// Align items options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignItems {
    FlexStart,
    FlexEnd,
    Center,
    Baseline,
    Stretch,
}

/// Linear layout (row or column)
pub struct LinearLayout {
    direction: LayoutDirection,
    spacing: f32,
    padding: Padding,
    alignment: AlignItems,
    justify_content: JustifyContent,
}

/// Flexible box layout
pub struct FlexLayout {
    direction: LayoutDirection,
    wrap: FlexWrap,
    justify_content: JustifyContent,
    align_items: AlignItems,
    align_content: JustifyContent,
    gap: f32,
    padding: Padding,
}

/// Grid layout
pub struct GridLayout {
    columns: GridTrackList,
    rows: GridTrackList,
    gap: Size,
    padding: Padding,
    auto_flow: GridAutoFlow,
}

/// Grid track definition
#[derive(Debug, Clone)]
pub enum GridTrack {
    Pixels(f32),
    Fraction(f32), // fr units
    Auto,
    MinContent,
    MaxContent,
    FitContent(f32),
    MinMax(Box<GridTrack>, Box<GridTrack>),
}

/// Grid track list
#[derive(Debug, Clone)]
pub struct GridTrackList {
    tracks: Vec<GridTrack>,
    repeat: Option<GridRepeat>,
}

/// Grid repeat pattern
#[derive(Debug, Clone)]
pub struct GridRepeat {
    count: GridRepeatCount,
    tracks: Vec<GridTrack>,
}

#[derive(Debug, Clone)]
pub enum GridRepeatCount {
    Number(u32),
    AutoFit,
    AutoFill,
}

/// Grid auto flow direction
#[derive(Debug, Clone)]
pub enum GridAutoFlow {
    Row,
    Column,
    RowDense,
    ColumnDense,
}

/// Absolute layout (no automatic positioning)
pub struct AbsoluteLayout {
    padding: Padding,
}

/// Stack layout (children stack on top of each other)
pub struct StackLayout {
    padding: Padding,
    alignment: Alignment,
}

/// Responsive breakpoints for adaptive layouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakpointSystem {
    pub mobile: f32,    // 480px
    pub tablet: f32,    // 768px
    pub desktop: f32,   // 1024px
    pub wide: f32,      // 1440px
}

/// Responsive layout configuration
#[derive(Debug, Clone)]
pub struct ResponsiveLayout {
    breakpoints: BreakpointSystem,
    layouts: HashMap<String, Box<dyn Layout>>, // Layout per breakpoint
    current_breakpoint: String,
}

impl LayoutManager {
    /// Create new layout manager
    pub fn new() -> Self {
        Self {
            layouts: HashMap::new(),
            root_layout: None,
            constraints: HashMap::new(),
        }
    }
    
    /// Register a layout
    pub fn register_layout(&mut self, id: Uuid, layout: Box<dyn Layout>) {
        self.layouts.insert(id, layout);
    }
    
    /// Set root layout
    pub fn set_root_layout(&mut self, id: Uuid) {
        self.root_layout = Some(id);
    }
    
    /// Perform layout for container
    pub fn layout_container(&self, container: Rect, items: &mut [LayoutItem]) -> Result<()> {
        if let Some(root_id) = self.root_layout {
            if let Some(layout) = self.layouts.get(&root_id) {
                layout.layout(container, items)?;
            }
        }
        Ok(())
    }
    
    /// Add layout constraints for a component
    pub fn set_constraints(&mut self, component_id: Uuid, constraints: LayoutConstraints) {
        self.constraints.insert(component_id, constraints);
    }
    
    /// Get layout constraints for a component
    pub fn get_constraints(&self, component_id: &Uuid) -> Option<&LayoutConstraints> {
        self.constraints.get(component_id)
    }
    
    /// Calculate responsive breakpoint
    pub fn calculate_breakpoint(&self, container_width: f32) -> String {
        let breakpoints = BreakpointSystem::default();
        
        if container_width < breakpoints.mobile {
            "mobile".to_string()
        } else if container_width < breakpoints.tablet {
            "tablet".to_string()
        } else if container_width < breakpoints.desktop {
            "desktop".to_string()
        } else {
            "wide".to_string()
        }
    }
}

impl LinearLayout {
    /// Create new linear layout
    pub fn new(direction: LayoutDirection) -> Self {
        Self {
            direction,
            spacing: 8.0,
            padding: Padding::zero(),
            alignment: AlignItems::Stretch,
            justify_content: JustifyContent::FlexStart,
        }
    }
    
    /// Set spacing between items
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }
    
    /// Set padding
    pub fn padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }
    
    /// Set alignment
    pub fn align_items(mut self, alignment: AlignItems) -> Self {
        self.alignment = alignment;
        self
    }
    
    /// Set content justification
    pub fn justify_content(mut self, justify: JustifyContent) -> Self {
        self.justify_content = justify;
        self
    }
}

impl Layout for LinearLayout {
    fn layout(&self, container: Rect, children: &mut [LayoutItem]) -> Result<()> {
        if children.is_empty() {
            return Ok(());
        }
        
        let available_width = container.width - self.padding.left - self.padding.right;
        let available_height = container.height - self.padding.top - self.padding.bottom;
        
        let start_x = container.x + self.padding.left;
        let start_y = container.y + self.padding.top;
        
        match self.direction {
            LayoutDirection::Row | LayoutDirection::RowReverse => {
                self.layout_row(start_x, start_y, available_width, available_height, children)?;
            }
            LayoutDirection::Column | LayoutDirection::ColumnReverse => {
                self.layout_column(start_x, start_y, available_width, available_height, children)?;
            }
        }
        
        Ok(())
    }
    
    fn layout_type(&self) -> &'static str {
        "LinearLayout"
    }
    
    fn min_size(&self, children: &[LayoutItem]) -> Size {
        match self.direction {
            LayoutDirection::Row | LayoutDirection::RowReverse => {
                let total_width = children.iter()
                    .map(|item| item.constraints.min_width.unwrap_or(0.0))
                    .sum::<f32>() + (children.len().saturating_sub(1) as f32 * self.spacing);
                let max_height = children.iter()
                    .map(|item| item.constraints.min_height.unwrap_or(0.0))
                    .fold(0.0, f32::max);
                
                Size::new(
                    total_width + self.padding.left + self.padding.right,
                    max_height + self.padding.top + self.padding.bottom
                )
            }
            LayoutDirection::Column | LayoutDirection::ColumnReverse => {
                let max_width = children.iter()
                    .map(|item| item.constraints.min_width.unwrap_or(0.0))
                    .fold(0.0, f32::max);
                let total_height = children.iter()
                    .map(|item| item.constraints.min_height.unwrap_or(0.0))
                    .sum::<f32>() + (children.len().saturating_sub(1) as f32 * self.spacing);
                
                Size::new(
                    max_width + self.padding.left + self.padding.right,
                    total_height + self.padding.top + self.padding.bottom
                )
            }
        }
    }
    
    fn preferred_size(&self, children: &[LayoutItem]) -> Size {
        // Similar to min_size but with preferred dimensions
        self.min_size(children)
    }
    
    fn is_responsive(&self) -> bool {
        true
    }
}

impl LinearLayout {
    fn layout_row(&self, start_x: f32, start_y: f32, available_width: f32, available_height: f32, children: &mut [LayoutItem]) -> Result<()> {
        let total_spacing = (children.len().saturating_sub(1) as f32) * self.spacing;
        let available_content_width = available_width - total_spacing;
        
        // Calculate item widths
        let mut x_offset = start_x;
        
        for (i, item) in children.iter_mut().enumerate() {
            if !item.visible {
                continue;
            }
            
            let item_width = if let Some(min_width) = item.constraints.min_width {
                min_width.min(available_content_width / children.len() as f32)
            } else {
                available_content_width / children.len() as f32
            };
            
            let item_height = if matches!(self.alignment, AlignItems::Stretch) {
                available_height
            } else if let Some(min_height) = item.constraints.min_height {
                min_height.min(available_height)
            } else {
                available_height
            };
            
            let y_position = match self.alignment {
                AlignItems::FlexStart => start_y,
                AlignItems::FlexEnd => start_y + available_height - item_height,
                AlignItems::Center => start_y + (available_height - item_height) / 2.0,
                AlignItems::Stretch | AlignItems::Baseline => start_y,
            };
            
            item.bounds = Rect::new(x_offset, y_position, item_width, item_height);
            
            x_offset += item_width;
            if i < children.len() - 1 {
                x_offset += self.spacing;
            }
        }
        
        Ok(())
    }
    
    fn layout_column(&self, start_x: f32, start_y: f32, available_width: f32, available_height: f32, children: &mut [LayoutItem]) -> Result<()> {
        let total_spacing = (children.len().saturating_sub(1) as f32) * self.spacing;
        let available_content_height = available_height - total_spacing;
        
        // Calculate item heights
        let mut y_offset = start_y;
        
        for (i, item) in children.iter_mut().enumerate() {
            if !item.visible {
                continue;
            }
            
            let item_height = if let Some(min_height) = item.constraints.min_height {
                min_height.min(available_content_height / children.len() as f32)
            } else {
                available_content_height / children.len() as f32
            };
            
            let item_width = if matches!(self.alignment, AlignItems::Stretch) {
                available_width
            } else if let Some(min_width) = item.constraints.min_width {
                min_width.min(available_width)
            } else {
                available_width
            };
            
            let x_position = match self.alignment {
                AlignItems::FlexStart => start_x,
                AlignItems::FlexEnd => start_x + available_width - item_width,
                AlignItems::Center => start_x + (available_width - item_width) / 2.0,
                AlignItems::Stretch | AlignItems::Baseline => start_x,
            };
            
            item.bounds = Rect::new(x_position, y_offset, item_width, item_height);
            
            y_offset += item_height;
            if i < children.len() - 1 {
                y_offset += self.spacing;
            }
        }
        
        Ok(())
    }
}

impl FlexLayout {
    /// Create new flex layout
    pub fn new() -> Self {
        Self {
            direction: LayoutDirection::Row,
            wrap: FlexWrap::NoWrap,
            justify_content: JustifyContent::FlexStart,
            align_items: AlignItems::Stretch,
            align_content: JustifyContent::Stretch,
            gap: 0.0,
            padding: Padding::zero(),
        }
    }
    
    /// Set flex direction
    pub fn direction(mut self, direction: LayoutDirection) -> Self {
        self.direction = direction;
        self
    }
    
    /// Set wrap behavior
    pub fn wrap(mut self, wrap: FlexWrap) -> Self {
        self.wrap = wrap;
        self
    }
    
    /// Set justify content
    pub fn justify_content(mut self, justify: JustifyContent) -> Self {
        self.justify_content = justify;
        self
    }
    
    /// Set align items
    pub fn align_items(mut self, align: AlignItems) -> Self {
        self.align_items = align;
        self
    }
    
    /// Set gap between items
    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }
}

impl Layout for FlexLayout {
    fn layout(&self, container: Rect, children: &mut [LayoutItem]) -> Result<()> {
        // Simplified flex layout implementation
        // In a full implementation, this would handle all flex properties
        let linear = LinearLayout::new(self.direction.clone())
            .spacing(self.gap)
            .padding(self.padding.clone())
            .align_items(self.align_items.clone())
            .justify_content(self.justify_content.clone());
        
        linear.layout(container, children)
    }
    
    fn layout_type(&self) -> &'static str {
        "FlexLayout"
    }
    
    fn min_size(&self, children: &[LayoutItem]) -> Size {
        Size::zero() // Simplified implementation
    }
    
    fn preferred_size(&self, children: &[LayoutItem]) -> Size {
        Size::zero() // Simplified implementation
    }
    
    fn is_responsive(&self) -> bool {
        true
    }
}

impl LayoutItem {
    /// Create new layout item
    pub fn new(component_id: Uuid) -> Self {
        Self {
            component_id,
            bounds: Rect::zero(),
            constraints: LayoutConstraints::default(),
            margin: Margin::zero(),
            padding: Padding::zero(),
            visible: true,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: FlexBasis::Auto,
        }
    }
    
    /// Set flex grow factor
    pub fn flex_grow(mut self, grow: f32) -> Self {
        self.flex_grow = grow;
        self
    }
    
    /// Set flex shrink factor
    pub fn flex_shrink(mut self, shrink: f32) -> Self {
        self.flex_shrink = shrink;
        self
    }
    
    /// Set flex basis
    pub fn flex_basis(mut self, basis: FlexBasis) -> Self {
        self.flex_basis = basis;
        self
    }
    
    /// Set margin
    pub fn margin(mut self, margin: Margin) -> Self {
        self.margin = margin;
        self
    }
    
    /// Set constraints
    pub fn constraints(mut self, constraints: LayoutConstraints) -> Self {
        self.constraints = constraints;
        self
    }
}

impl Default for LayoutConstraints {
    fn default() -> Self {
        Self {
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            aspect_ratio: None,
            alignment: Alignment::Start,
            position: Position::Static,
        }
    }
}

impl Default for BreakpointSystem {
    fn default() -> Self {
        Self {
            mobile: 480.0,
            tablet: 768.0,
            desktop: 1024.0,
            wide: 1440.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layout_manager_creation() {
        let manager = LayoutManager::new();
        assert!(manager.root_layout.is_none());
    }
    
    #[test]
    fn test_linear_layout_creation() {
        let layout = LinearLayout::new(LayoutDirection::Row);
        assert_eq!(layout.layout_type(), "LinearLayout");
        assert!(layout.is_responsive());
    }
    
    #[test]
    fn test_layout_item_creation() {
        let item = LayoutItem::new(Uuid::new_v4());
        assert!(item.visible);
        assert_eq!(item.flex_grow, 0.0);
        assert_eq!(item.flex_shrink, 1.0);
    }
    
    #[test]
    fn test_breakpoint_calculation() {
        let manager = LayoutManager::new();
        assert_eq!(manager.calculate_breakpoint(320.0), "mobile");
        assert_eq!(manager.calculate_breakpoint(800.0), "tablet");
        assert_eq!(manager.calculate_breakpoint(1200.0), "desktop");
        assert_eq!(manager.calculate_breakpoint(1600.0), "wide");
    }
    
    #[test]
    fn test_flex_layout_builder() {
        let layout = FlexLayout::new()
            .direction(LayoutDirection::Column)
            .gap(16.0)
            .justify_content(JustifyContent::Center);
        
        assert_eq!(layout.gap, 16.0);
        assert!(matches!(layout.direction, LayoutDirection::Column));
        assert!(matches!(layout.justify_content, JustifyContent::Center));
    }
}