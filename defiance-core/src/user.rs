//! User management and random username generation

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use names::{Generator, Name};
use rand::Rng;
use crate::error::{DefianceError, Result};
use crate::storage::DefianceStorage;

/// User representation in DefianceNetwork
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: Username,
    pub created_at: i64,
    pub last_seen: i64,
    pub opt_in_visibility: bool,
    pub reputation_score: f32,
    pub total_broadcasts: u64,
    pub total_watch_time: u64, // in seconds
}

/// Immutable username with humorous nature-inspired generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Username {
    pub value: String,
    pub generated_at: i64,
}

impl Username {
    /// Generate a new random username in format: [Nature/Adjective][Noun][Number]
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let generator = Generator::default();
        
        // Nature/adjective words
        let adjectives = [
            "Water", "Golden", "Mystic", "Forest", "Ocean", "Crystal", "Silver", "Emerald",
            "Ruby", "Sapphire", "Cosmic", "Solar", "Lunar", "Arctic", "Tropical", "Wild",
            "Free", "Swift", "Gentle", "Brave", "Wise", "Ancient", "Noble", "Serene",
            "Radiant", "Peaceful", "Thunder", "Lightning", "Mountain", "Valley", "River",
            "Lake", "Island", "Desert", "Prairie", "Meadow", "Garden", "Bloom", "Dawn",
            "Sunset", "Twilight", "Aurora", "Star", "Moon", "Sun", "Wind", "Rain", "Snow"
        ];
        
        // Animal and nature nouns
        let nouns = [
            "Angel", "Fish", "Bird", "Wolf", "Fox", "Bear", "Eagle", "Hawk", "Owl",
            "Deer", "Rabbit", "Squirrel", "Butterfly", "Dragon", "Phoenix", "Tiger",
            "Lion", "Elephant", "Whale", "Dolphin", "Turtle", "Penguin", "Seal",
            "Otter", "Beaver", "Badger", "Hedgehog", "Robin", "Sparrow", "Finch",
            "Cardinal", "Blue jay", "Hummingbird", "Crane", "Swan", "Duck", "Goose",
            "Tree", "Flower", "Rose", "Lily", "Daisy", "Orchid", "Fern", "Moss",
            "Stone", "Crystal", "Gem", "Pearl", "Shell", "Coral", "Wave", "Breeze"
        ];
        
        let adjective = adjectives[rng.gen_range(0..adjectives.len())];
        let noun = nouns[rng.gen_range(0..nouns.len())].replace(" ", "");
        let number = rng.gen_range(10..99);
        
        let username = format!("{}{}{:02}", adjective, noun, number);
        
        Self {
            value: username,
            generated_at: chrono::Utc::now().timestamp(),
        }
    }
    
    /// Validate username format (used for imported usernames)
    pub fn is_valid(username: &str) -> bool {
        // Check length (6-20 characters)
        if username.len() < 6 || username.len() > 20 {
            return false;
        }
        
        // Check that it contains only alphanumeric characters
        username.chars().all(|c| c.is_alphanumeric())
    }
}

/// User manager for handling user operations
pub struct UserManager {
    storage: Arc<RwLock<DefianceStorage>>,
    current_user: Option<User>,
}

impl UserManager {
    /// Create a new user manager
    pub async fn new(storage: Arc<RwLock<DefianceStorage>>) -> Result<Self> {
        let mut manager = Self {
            storage,
            current_user: None,
        };
        
        // Try to load existing user or create new one
        manager.initialize_user().await?;
        
        Ok(manager)
    }
    
    /// Start user manager
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting user manager");
        Ok(())
    }
    
    /// Initialize user (load existing or create new)
    async fn initialize_user(&mut self) -> Result<()> {
        let storage = self.storage.read().await;
        
        // Try to load existing user
        if let Ok(user) = storage.get_current_user().await {
            self.current_user = Some(user);
            tracing::info!("Loaded existing user: {}", self.current_user.as_ref().unwrap().username.value);
        } else {
            // Create new user
            drop(storage);
            let user = self.create_new_user().await?;
            self.current_user = Some(user);
            tracing::info!("Created new user: {}", self.current_user.as_ref().unwrap().username.value);
        }
        
        Ok(())
    }
    
    /// Create a new user with random username
    async fn create_new_user(&self) -> Result<User> {
        let username = Username::generate();
        let now = chrono::Utc::now().timestamp();
        
        let user = User {
            id: Uuid::new_v4(),
            username,
            created_at: now,
            last_seen: now,
            opt_in_visibility: false, // Privacy by default
            reputation_score: 0.0,
            total_broadcasts: 0,
            total_watch_time: 0,
        };
        
        // Store user in database
        let mut storage = self.storage.write().await;
        storage.store_user(&user).await?;
        
        Ok(user)
    }
    
    /// Get the current user
    pub async fn get_current_user(&self) -> Result<Option<User>> {
        Ok(self.current_user.clone())
    }
    
    /// Update user's last seen timestamp
    pub async fn update_last_seen(&mut self) -> Result<()> {
        if let Some(ref mut user) = self.current_user {
            user.last_seen = chrono::Utc::now().timestamp();
            
            let mut storage = self.storage.write().await;
            storage.update_user(user).await?;
        }
        
        Ok(())
    }
    
    /// Toggle visibility opt-in
    pub async fn toggle_visibility(&mut self) -> Result<bool> {
        if let Some(ref mut user) = self.current_user {
            user.opt_in_visibility = !user.opt_in_visibility;
            
            let mut storage = self.storage.write().await;
            storage.update_user(user).await?;
            
            Ok(user.opt_in_visibility)
        } else {
            Err(DefianceError::User("No current user".to_string()))
        }
    }
    
    /// Increment broadcast count
    pub async fn increment_broadcasts(&mut self) -> Result<()> {
        if let Some(ref mut user) = self.current_user {
            user.total_broadcasts += 1;
            
            let mut storage = self.storage.write().await;
            storage.update_user(user).await?;
        }
        
        Ok(())
    }
    
    /// Add watch time
    pub async fn add_watch_time(&mut self, seconds: u64) -> Result<()> {
        if let Some(ref mut user) = self.current_user {
            user.total_watch_time += seconds;
            
            let mut storage = self.storage.write().await;
            storage.update_user(user).await?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_username_generation() {
        let username = Username::generate();
        assert!(username.value.len() >= 6);
        assert!(username.value.len() <= 20);
        assert!(Username::is_valid(&username.value));
    }
    
    #[test]
    fn test_username_validation() {
        assert!(Username::is_valid("WaterAngel03"));
        assert!(Username::is_valid("GoldenFish22"));
        assert!(!Username::is_valid("abc")); // too short
        assert!(!Username::is_valid("a".repeat(25).as_str())); // too long
        assert!(!Username::is_valid("user@domain")); // invalid chars
        assert!(!Username::is_valid("user name")); // space
    }
}