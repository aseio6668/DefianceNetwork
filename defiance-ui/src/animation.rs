//! Animation system for Renaissance UI framework

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;

/// Animation manager for coordinating UI animations
pub struct AnimationManager {
    active_animations: HashMap<Uuid, Box<dyn Animation>>,
    timeline: Timeline,
    global_speed_multiplier: f32,
    reduced_motion_mode: bool,
    performance_mode: bool,
}

/// Core animation trait
pub trait Animation: Send + Sync {
    /// Update animation with delta time
    fn update(&mut self, delta_time: f32) -> Result<AnimationState>;
    
    /// Get current animation value (0.0 to 1.0)
    fn value(&self) -> f32;
    
    /// Check if animation is complete
    fn is_complete(&self) -> bool;
    
    /// Get animation duration
    fn duration(&self) -> f32;
    
    /// Get animation type
    fn animation_type(&self) -> &'static str;
    
    /// Reset animation to beginning
    fn reset(&mut self);
    
    /// Set animation direction
    fn set_direction(&mut self, direction: AnimationDirection);
    
    /// Set loop mode
    fn set_loop_mode(&mut self, mode: LoopMode);
}

/// Animation state
#[derive(Debug, Clone, PartialEq)]
pub enum AnimationState {
    Playing,
    Paused,
    Complete,
    Cancelled,
}

/// Animation direction
#[derive(Debug, Clone, PartialEq)]
pub enum AnimationDirection {
    Forward,
    Reverse,
    Alternate,
}

/// Loop mode for animations
#[derive(Debug, Clone, PartialEq)]
pub enum LoopMode {
    None,
    Infinite,
    Count(u32),
}

/// Easing functions for smooth animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInBack,
    EaseOutBack,
    EaseInOutBack,
    EaseInBounce,
    EaseOutBounce,
    EaseInOutBounce,
    EaseInElastic,
    EaseOutElastic,
    EaseInOutElastic,
    Custom(Vec<f32>), // Cubic bezier control points
}

/// Timeline for animation sequencing
pub struct Timeline {
    sequences: Vec<AnimationSequence>,
    current_time: f32,
    total_duration: f32,
    playing: bool,
}

/// Animation sequence for chaining multiple animations
#[derive(Debug, Clone)]
pub struct AnimationSequence {
    animations: Vec<AnimationClip>,
    delay: f32,
    repeat_count: u32,
}

/// Individual animation clip
#[derive(Debug, Clone)]
pub struct AnimationClip {
    target_id: Uuid,
    property: AnimatedProperty,
    from_value: f32,
    to_value: f32,
    duration: f32,
    easing: EasingFunction,
    delay: f32,
}

/// Properties that can be animated
#[derive(Debug, Clone)]
pub enum AnimatedProperty {
    Opacity,
    Scale,
    Rotation,
    TranslateX,
    TranslateY,
    Width,
    Height,
    BorderRadius,
    BackgroundColor,
    TextColor,
    Custom(String),
}

/// Tween animation for property interpolation
pub struct TweenAnimation {
    id: Uuid,
    start_value: f32,
    end_value: f32,
    current_value: f32,
    duration: f32,
    elapsed_time: f32,
    easing: EasingFunction,
    direction: AnimationDirection,
    loop_mode: LoopMode,
    loop_count: u32,
    state: AnimationState,
}

/// Spring animation for natural motion
pub struct SpringAnimation {
    id: Uuid,
    target_value: f32,
    current_value: f32,
    velocity: f32,
    stiffness: f32,
    damping: f32,
    mass: f32,
    threshold: f32,
    state: AnimationState,
}

/// Keyframe animation for complex motion
pub struct KeyframeAnimation {
    id: Uuid,
    keyframes: Vec<Keyframe>,
    current_keyframe: usize,
    current_value: f32,
    duration: f32,
    elapsed_time: f32,
    loop_mode: LoopMode,
    state: AnimationState,
}

/// Individual keyframe
#[derive(Debug, Clone)]
pub struct Keyframe {
    time: f32,    // 0.0 to 1.0
    value: f32,
    easing: EasingFunction,
}

/// Particle animation for visual effects
pub struct ParticleAnimation {
    id: Uuid,
    particles: Vec<Particle>,
    emitter: ParticleEmitter,
    duration: f32,
    elapsed_time: f32,
    state: AnimationState,
}

/// Individual particle
#[derive(Debug, Clone)]
pub struct Particle {
    position: (f32, f32),
    velocity: (f32, f32),
    acceleration: (f32, f32),
    color: (f32, f32, f32, f32), // RGBA
    size: f32,
    life: f32,
    max_life: f32,
}

/// Particle emitter configuration
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    position: (f32, f32),
    emission_rate: f32,
    particle_life: f32,
    initial_velocity: (f32, f32),
    velocity_variation: (f32, f32),
    size_range: (f32, f32),
    color_range: ((f32, f32, f32, f32), (f32, f32, f32, f32)),
}

/// Transition effects for UI state changes
#[derive(Debug, Clone)]
pub enum TransitionEffect {
    Fade,
    Slide { direction: SlideDirection },
    Scale { from_scale: f32, to_scale: f32 },
    Flip { axis: FlipAxis },
    Dissolve { grain_size: f32 },
    Zoom { focal_point: (f32, f32) },
    Morph,
}

/// Slide direction for transitions
#[derive(Debug, Clone)]
pub enum SlideDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Flip axis for transitions
#[derive(Debug, Clone)]
pub enum FlipAxis {
    Horizontal,
    Vertical,
}

impl AnimationManager {
    /// Create new animation manager
    pub fn new() -> Self {
        Self {
            active_animations: HashMap::new(),
            timeline: Timeline::new(),
            global_speed_multiplier: 1.0,
            reduced_motion_mode: false,
            performance_mode: false,
        }
    }
    
    /// Initialize animation system
    pub fn initialize(&mut self) -> Result<()> {
        tracing::info!("Animation system initialized");
        Ok(())
    }
    
    /// Update all active animations
    pub fn update(&mut self, delta_time: f32) -> Result<()> {
        let effective_delta = if self.reduced_motion_mode {
            delta_time * 0.2 // Slow down animations for reduced motion
        } else {
            delta_time * self.global_speed_multiplier
        };
        
        // Update timeline
        self.timeline.update(effective_delta)?;
        
        // Update individual animations
        let mut completed_animations = Vec::new();
        
        for (id, animation) in self.active_animations.iter_mut() {
            match animation.update(effective_delta)? {
                AnimationState::Complete | AnimationState::Cancelled => {
                    completed_animations.push(*id);
                }
                _ => {}
            }
        }
        
        // Remove completed animations
        for id in completed_animations {
            self.active_animations.remove(&id);
        }
        
        Ok(())
    }
    
    /// Add animation to manager
    pub fn add_animation(&mut self, animation: Box<dyn Animation>) -> Uuid {
        let id = Uuid::new_v4();
        self.active_animations.insert(id, animation);
        id
    }
    
    /// Remove animation by ID
    pub fn remove_animation(&mut self, id: &Uuid) -> bool {
        self.active_animations.remove(id).is_some()
    }
    
    /// Get animation value by ID
    pub fn get_animation_value(&self, id: &Uuid) -> Option<f32> {
        self.active_animations.get(id).map(|anim| anim.value())
    }
    
    /// Set reduced motion mode
    pub fn set_reduced_mode(&mut self, enabled: bool) {
        self.reduced_motion_mode = enabled;
        
        if enabled {
            // Cancel non-essential animations
            self.active_animations.retain(|_, anim| {
                matches!(anim.animation_type(), "essential" | "accessibility")
            });
        }
    }
    
    /// Set performance mode
    pub fn set_performance_mode(&mut self, enabled: bool) {
        self.performance_mode = enabled;
        
        if enabled {
            self.global_speed_multiplier = 1.5; // Speed up animations
        } else {
            self.global_speed_multiplier = 1.0;
        }
    }
    
    /// Create fade transition
    pub fn create_fade_transition(&mut self, target_id: Uuid, from_opacity: f32, to_opacity: f32, duration: f32) -> Uuid {
        let animation = TweenAnimation::new(from_opacity, to_opacity, duration, EasingFunction::EaseInOut);
        self.add_animation(Box::new(animation))
    }
    
    /// Create slide transition
    pub fn create_slide_transition(&mut self, target_id: Uuid, direction: SlideDirection, distance: f32, duration: f32) -> Uuid {
        let (from_x, to_x) = match direction {
            SlideDirection::Left => (distance, 0.0),
            SlideDirection::Right => (-distance, 0.0),
            _ => (0.0, 0.0),
        };
        
        let animation = TweenAnimation::new(from_x, to_x, duration, EasingFunction::EaseOutBack);
        self.add_animation(Box::new(animation))
    }
    
    /// Create spring animation
    pub fn create_spring_animation(&mut self, target_value: f32, stiffness: f32, damping: f32) -> Uuid {
        let animation = SpringAnimation::new(target_value, stiffness, damping);
        self.add_animation(Box::new(animation))
    }
    
    /// Create particle effect
    pub fn create_particle_effect(&mut self, emitter: ParticleEmitter, duration: f32) -> Uuid {
        let animation = ParticleAnimation::new(emitter, duration);
        self.add_animation(Box::new(animation))
    }
    
    /// Pause all animations
    pub fn pause_all(&mut self) {
        self.timeline.pause();
    }
    
    /// Resume all animations
    pub fn resume_all(&mut self) {
        self.timeline.resume();
    }
    
    /// Clear all animations
    pub fn clear_all(&mut self) {
        self.active_animations.clear();
        self.timeline.clear();
    }
}

impl TweenAnimation {
    /// Create new tween animation
    pub fn new(start_value: f32, end_value: f32, duration: f32, easing: EasingFunction) -> Self {
        Self {
            id: Uuid::new_v4(),
            start_value,
            end_value,
            current_value: start_value,
            duration,
            elapsed_time: 0.0,
            easing,
            direction: AnimationDirection::Forward,
            loop_mode: LoopMode::None,
            loop_count: 0,
            state: AnimationState::Playing,
        }
    }
}

impl Animation for TweenAnimation {
    fn update(&mut self, delta_time: f32) -> Result<AnimationState> {
        if self.state != AnimationState::Playing {
            return Ok(self.state.clone());
        }
        
        self.elapsed_time += delta_time;
        
        if self.elapsed_time >= self.duration {
            match self.loop_mode {
                LoopMode::None => {
                    self.current_value = self.end_value;
                    self.state = AnimationState::Complete;
                }
                LoopMode::Infinite => {
                    self.elapsed_time = 0.0;
                    if self.direction == AnimationDirection::Alternate {
                        std::mem::swap(&mut self.start_value, &mut self.end_value);
                    }
                }
                LoopMode::Count(count) => {
                    self.loop_count += 1;
                    if self.loop_count >= count {
                        self.current_value = self.end_value;
                        self.state = AnimationState::Complete;
                    } else {
                        self.elapsed_time = 0.0;
                        if self.direction == AnimationDirection::Alternate {
                            std::mem::swap(&mut self.start_value, &mut self.end_value);
                        }
                    }
                }
            }
        } else {
            let progress = self.elapsed_time / self.duration;
            let eased_progress = self.apply_easing(progress);
            self.current_value = self.start_value + (self.end_value - self.start_value) * eased_progress;
        }
        
        Ok(self.state.clone())
    }
    
    fn value(&self) -> f32 {
        self.current_value
    }
    
    fn is_complete(&self) -> bool {
        self.state == AnimationState::Complete
    }
    
    fn duration(&self) -> f32 {
        self.duration
    }
    
    fn animation_type(&self) -> &'static str {
        "tween"
    }
    
    fn reset(&mut self) {
        self.elapsed_time = 0.0;
        self.current_value = self.start_value;
        self.state = AnimationState::Playing;
        self.loop_count = 0;
    }
    
    fn set_direction(&mut self, direction: AnimationDirection) {
        self.direction = direction;
    }
    
    fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }
}

impl TweenAnimation {
    fn apply_easing(&self, t: f32) -> f32 {
        match &self.easing {
            EasingFunction::Linear => t,
            EasingFunction::EaseIn => t * t,
            EasingFunction::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            EasingFunction::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            }
            EasingFunction::EaseInBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                c3 * t * t * t - c1 * t * t
            }
            EasingFunction::EaseOutBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
            }
            EasingFunction::EaseInOutBack => {
                let c1 = 1.70158;
                let c2 = c1 * 1.525;
                if t < 0.5 {
                    (2.0 * t).powi(2) * ((c2 + 1.0) * 2.0 * t - c2) / 2.0
                } else {
                    ((2.0 * t - 2.0).powi(2) * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0
                }
            }
            EasingFunction::EaseOutBounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t_adj = t - 1.5 / d1;
                    n1 * t_adj * t_adj + 0.75
                } else if t < 2.5 / d1 {
                    let t_adj = t - 2.25 / d1;
                    n1 * t_adj * t_adj + 0.9375
                } else {
                    let t_adj = t - 2.625 / d1;
                    n1 * t_adj * t_adj + 0.984375
                }
            }
            _ => t, // Simplified for other easing functions
        }
    }
}

impl SpringAnimation {
    /// Create new spring animation
    pub fn new(target_value: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            target_value,
            current_value: 0.0,
            velocity: 0.0,
            stiffness,
            damping,
            mass: 1.0,
            threshold: 0.001,
            state: AnimationState::Playing,
        }
    }
}

impl Animation for SpringAnimation {
    fn update(&mut self, delta_time: f32) -> Result<AnimationState> {
        if self.state != AnimationState::Playing {
            return Ok(self.state.clone());
        }
        
        // Spring physics simulation
        let spring_force = -self.stiffness * (self.current_value - self.target_value);
        let damping_force = -self.damping * self.velocity;
        let acceleration = (spring_force + damping_force) / self.mass;
        
        self.velocity += acceleration * delta_time;
        self.current_value += self.velocity * delta_time;
        
        // Check if spring has settled
        if (self.current_value - self.target_value).abs() < self.threshold && self.velocity.abs() < self.threshold {
            self.current_value = self.target_value;
            self.velocity = 0.0;
            self.state = AnimationState::Complete;
        }
        
        Ok(self.state.clone())
    }
    
    fn value(&self) -> f32 {
        self.current_value
    }
    
    fn is_complete(&self) -> bool {
        self.state == AnimationState::Complete
    }
    
    fn duration(&self) -> f32 {
        f32::INFINITY // Springs don't have fixed duration
    }
    
    fn animation_type(&self) -> &'static str {
        "spring"
    }
    
    fn reset(&mut self) {
        self.current_value = 0.0;
        self.velocity = 0.0;
        self.state = AnimationState::Playing;
    }
    
    fn set_direction(&mut self, _direction: AnimationDirection) {
        // Springs don't have direction in the traditional sense
    }
    
    fn set_loop_mode(&mut self, _mode: LoopMode) {
        // Springs don't loop
    }
}

impl Timeline {
    /// Create new timeline
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            current_time: 0.0,
            total_duration: 0.0,
            playing: false,
        }
    }
    
    /// Add animation sequence to timeline
    pub fn add_sequence(&mut self, sequence: AnimationSequence) {
        self.total_duration += sequence.duration() + sequence.delay;
        self.sequences.push(sequence);
    }
    
    /// Update timeline
    pub fn update(&mut self, delta_time: f32) -> Result<()> {
        if !self.playing {
            return Ok(());
        }
        
        self.current_time += delta_time;
        
        // Update active sequences
        for sequence in &mut self.sequences {
            sequence.update(self.current_time, delta_time)?;
        }
        
        Ok(())
    }
    
    /// Play timeline
    pub fn play(&mut self) {
        self.playing = true;
    }
    
    /// Pause timeline
    pub fn pause(&mut self) {
        self.playing = false;
    }
    
    /// Resume timeline
    pub fn resume(&mut self) {
        self.playing = true;
    }
    
    /// Clear timeline
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.current_time = 0.0;
        self.total_duration = 0.0;
        self.playing = false;
    }
}

impl AnimationSequence {
    /// Get total duration of sequence
    pub fn duration(&self) -> f32 {
        self.animations.iter().map(|clip| clip.duration + clip.delay).fold(0.0, f32::max)
    }
    
    /// Update sequence
    pub fn update(&mut self, current_time: f32, delta_time: f32) -> Result<()> {
        // Update individual clips based on timeline
        for clip in &mut self.animations {
            let clip_start_time = self.delay + clip.delay;
            if current_time >= clip_start_time && current_time <= clip_start_time + clip.duration {
                // Clip is active, update its progress
                let progress = (current_time - clip_start_time) / clip.duration;
                // Apply easing and interpolate value
                // This would be connected to the actual UI component property
            }
        }
        Ok(())
    }
}

impl ParticleAnimation {
    /// Create new particle animation
    pub fn new(emitter: ParticleEmitter, duration: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            particles: Vec::new(),
            emitter,
            duration,
            elapsed_time: 0.0,
            state: AnimationState::Playing,
        }
    }
}

impl Animation for ParticleAnimation {
    fn update(&mut self, delta_time: f32) -> Result<AnimationState> {
        if self.state != AnimationState::Playing {
            return Ok(self.state.clone());
        }
        
        self.elapsed_time += delta_time;
        
        // Emit new particles
        let particles_to_emit = (self.emitter.emission_rate * delta_time) as usize;
        for _ in 0..particles_to_emit {
            if self.particles.len() < 1000 { // Limit particle count
                self.particles.push(self.emitter.emit_particle());
            }
        }
        
        // Update existing particles
        self.particles.retain_mut(|particle| {
            particle.update(delta_time);
            particle.life > 0.0
        });
        
        // Check if animation is complete
        if self.elapsed_time >= self.duration {
            self.state = AnimationState::Complete;
        }
        
        Ok(self.state.clone())
    }
    
    fn value(&self) -> f32 {
        self.elapsed_time / self.duration
    }
    
    fn is_complete(&self) -> bool {
        self.state == AnimationState::Complete && self.particles.is_empty()
    }
    
    fn duration(&self) -> f32 {
        self.duration
    }
    
    fn animation_type(&self) -> &'static str {
        "particle"
    }
    
    fn reset(&mut self) {
        self.elapsed_time = 0.0;
        self.particles.clear();
        self.state = AnimationState::Playing;
    }
    
    fn set_direction(&mut self, _direction: AnimationDirection) {
        // Particles don't have direction in the traditional sense
    }
    
    fn set_loop_mode(&mut self, _mode: LoopMode) {
        // Particle effects typically don't loop
    }
}

impl ParticleEmitter {
    /// Emit a new particle
    pub fn emit_particle(&self) -> Particle {
        use std::f32::consts::PI;
        
        // Random values for variation (simplified)
        let angle = (rand::random::<f32>() * 2.0 * PI);
        let speed = self.initial_velocity.0 + (rand::random::<f32>() - 0.5) * self.velocity_variation.0;
        
        Particle {
            position: self.position,
            velocity: (speed * angle.cos(), speed * angle.sin()),
            acceleration: (0.0, -9.8), // Gravity
            color: self.color_range.0, // Simplified
            size: self.size_range.0 + rand::random::<f32>() * (self.size_range.1 - self.size_range.0),
            life: self.particle_life,
            max_life: self.particle_life,
        }
    }
}

impl Particle {
    /// Update particle physics
    pub fn update(&mut self, delta_time: f32) {
        // Update position based on velocity
        self.position.0 += self.velocity.0 * delta_time;
        self.position.1 += self.velocity.1 * delta_time;
        
        // Update velocity based on acceleration
        self.velocity.0 += self.acceleration.0 * delta_time;
        self.velocity.1 += self.acceleration.1 * delta_time;
        
        // Update life
        self.life -= delta_time;
        
        // Fade alpha as particle dies
        let life_ratio = self.life / self.max_life;
        self.color.3 = life_ratio;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_animation_manager_creation() {
        let manager = AnimationManager::new();
        assert!(!manager.reduced_motion_mode);
        assert!(!manager.performance_mode);
        assert_eq!(manager.global_speed_multiplier, 1.0);
    }
    
    #[test]
    fn test_tween_animation() {
        let mut animation = TweenAnimation::new(0.0, 1.0, 1.0, EasingFunction::Linear);
        
        assert_eq!(animation.value(), 0.0);
        assert!(!animation.is_complete());
        
        // Update halfway
        animation.update(0.5).unwrap();
        assert!((animation.value() - 0.5).abs() < 0.001);
        
        // Complete animation
        animation.update(0.5).unwrap();
        assert!((animation.value() - 1.0).abs() < 0.001);
        assert!(animation.is_complete());
    }
    
    #[test]
    fn test_spring_animation() {
        let mut animation = SpringAnimation::new(1.0, 100.0, 10.0);
        
        assert_eq!(animation.value(), 0.0);
        assert!(!animation.is_complete());
        
        // Update spring - should move towards target
        animation.update(0.016).unwrap(); // ~60fps
        assert!(animation.value() > 0.0);
        assert!(animation.value() < 1.0);
    }
    
    #[test]
    fn test_easing_functions() {
        let animation = TweenAnimation::new(0.0, 1.0, 1.0, EasingFunction::Linear);
        
        assert_eq!(animation.apply_easing(0.0), 0.0);
        assert_eq!(animation.apply_easing(0.5), 0.5);
        assert_eq!(animation.apply_easing(1.0), 1.0);
        
        let ease_in = TweenAnimation::new(0.0, 1.0, 1.0, EasingFunction::EaseIn);
        let halfway = ease_in.apply_easing(0.5);
        assert!(halfway < 0.5); // Should be slower at the beginning
    }
    
    #[test]
    fn test_timeline_creation() {
        let timeline = Timeline::new();
        assert_eq!(timeline.current_time, 0.0);
        assert_eq!(timeline.total_duration, 0.0);
        assert!(!timeline.playing);
    }
    
    #[test]
    fn test_particle_emitter() {
        let emitter = ParticleEmitter {
            position: (0.0, 0.0),
            emission_rate: 10.0,
            particle_life: 2.0,
            initial_velocity: (50.0, 50.0),
            velocity_variation: (10.0, 10.0),
            size_range: (1.0, 5.0),
            color_range: ((1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
        };
        
        let particle = emitter.emit_particle();
        assert_eq!(particle.position, (0.0, 0.0));
        assert_eq!(particle.life, 2.0);
        assert!(particle.size >= 1.0 && particle.size <= 5.0);
    }
}