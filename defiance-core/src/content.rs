//! Content management and metadata handling

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use crate::error::{DefianceError, Result};

/// Content types supported by DefianceNetwork
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    LiveStream,
    Video,
    Audio,
    Podcast,
    AudioBook,
    Lecture,
    Music,
    Talk,
    Documentary,
    Movie,
    TVShow,
    Educational,
    Other(String),
}

impl ContentType {
    pub fn is_video(&self) -> bool {
        matches!(self, 
            ContentType::LiveStream | 
            ContentType::Video | 
            ContentType::Documentary | 
            ContentType::Movie | 
            ContentType::TVShow
        )
    }
    
    pub fn is_audio(&self) -> bool {
        matches!(self, 
            ContentType::Audio | 
            ContentType::Podcast | 
            ContentType::AudioBook | 
            ContentType::Lecture | 
            ContentType::Music | 
            ContentType::Talk
        )
    }
}

/// Content quality levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Quality {
    Low,     // 480p / 128kbps
    Medium,  // 720p / 256kbps
    High,    // 1080p / 320kbps
    Ultra,   // 4K / 512kbps
    Source,  // Original quality
}

/// Content metadata structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub content_type: ContentType,
    pub creator: String,
    pub creator_id: Uuid,
    pub created_at: i64,
    pub duration: Option<u64>, // in seconds
    pub size: u64, // in bytes
    pub checksum: [u8; 32],
    pub available_qualities: Vec<Quality>,
    pub tags: Vec<String>,
    pub language: Option<String>,
    pub thumbnail: Option<Vec<u8>>,
    pub is_live: bool,
    pub viewer_count: u64,
    pub total_views: u64,
    pub rating: f32, // 0.0 to 5.0
    pub ratings_count: u64,
}

/// Content chunk for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentChunk {
    pub content_id: Uuid,
    pub chunk_id: u64,
    pub data: Vec<u8>,
    pub checksum: [u8; 32],
    pub quality: Quality,
    pub timestamp: Option<u64>, // for live streams
}

/// Complete content structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub metadata: ContentMetadata,
    pub chunks: HashMap<u64, ContentChunk>,
    pub total_chunks: u64,
}

impl ContentMetadata {
    /// Increment viewer count
    pub fn increment_viewers(&mut self) {
        self.viewer_count += 1;
        self.total_views += 1;
    }
    
    /// Decrement viewer count
    pub fn decrement_viewers(&mut self) {
        if self.viewer_count > 0 {
            self.viewer_count -= 1;
        }
    }
}

impl Content {
    /// Create new content with metadata
    pub fn new(
        title: String,
        description: String,
        content_type: ContentType,
        creator: String,
        creator_id: Uuid,
    ) -> Self {
        let metadata = ContentMetadata {
            id: Uuid::new_v4(),
            title,
            description,
            content_type,
            creator,
            creator_id,
            created_at: chrono::Utc::now().timestamp(),
            duration: None,
            size: 0,
            checksum: [0; 32],
            available_qualities: vec![Quality::Source],
            tags: Vec::new(),
            language: None,
            thumbnail: None,
            is_live: false,
            viewer_count: 0,
            total_views: 0,
            rating: 0.0,
            ratings_count: 0,
        };
        
        Self {
            metadata,
            chunks: HashMap::new(),
            total_chunks: 0,
        }
    }
    
    /// Create live stream content
    pub fn new_live_stream(
        title: String,
        description: String,
        creator: String,
        creator_id: Uuid,
    ) -> Self {
        let mut content = Self::new(title, description, ContentType::LiveStream, creator, creator_id);
        content.metadata.is_live = true;
        content
    }
    
    /// Add a chunk to the content
    pub fn add_chunk(&mut self, chunk: ContentChunk) -> Result<()> {
        if chunk.content_id != self.metadata.id {
            return Err(DefianceError::Content("Chunk content ID mismatch".to_string()));
        }
        
        self.chunks.insert(chunk.chunk_id, chunk);
        self.total_chunks = self.chunks.len() as u64;
        
        // Update content size
        self.metadata.size = self.chunks.values()
            .map(|chunk| chunk.data.len() as u64)
            .sum();
        
        Ok(())
    }
    
    /// Get chunk by ID
    pub fn get_chunk(&self, chunk_id: u64) -> Option<&ContentChunk> {
        self.chunks.get(&chunk_id)
    }
    
    /// Get available chunk IDs
    pub fn get_available_chunks(&self) -> Vec<u64> {
        self.chunks.keys().copied().collect()
    }
    
    /// Calculate content completion percentage
    pub fn get_completion_percentage(&self) -> f32 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        (self.chunks.len() as f32 / self.total_chunks as f32) * 100.0
    }
    
    /// Add tag to content
    pub fn add_tag(&mut self, tag: String) {
        if !self.metadata.tags.contains(&tag) {
            self.metadata.tags.push(tag);
        }
    }
    
    /// Set thumbnail
    pub fn set_thumbnail(&mut self, thumbnail: Vec<u8>) {
        self.metadata.thumbnail = Some(thumbnail);
    }
    
    /// Increment viewer count
    pub fn increment_viewers(&mut self) {
        self.metadata.viewer_count += 1;
        self.metadata.total_views += 1;
    }
    
    /// Decrement viewer count
    pub fn decrement_viewers(&mut self) {
        if self.metadata.viewer_count > 0 {
            self.metadata.viewer_count -= 1;
        }
    }
    
    /// Add rating
    pub fn add_rating(&mut self, rating: f32) -> Result<()> {
        if rating < 0.0 || rating > 5.0 {
            return Err(DefianceError::Content("Rating must be between 0.0 and 5.0".to_string()));
        }
        
        let total_rating = self.metadata.rating * self.metadata.ratings_count as f32;
        self.metadata.ratings_count += 1;
        self.metadata.rating = (total_rating + rating) / self.metadata.ratings_count as f32;
        
        Ok(())
    }
    
    /// Check if content is complete (all chunks available)
    pub fn is_complete(&self) -> bool {
        !self.metadata.is_live && self.total_chunks > 0 && self.chunks.len() as u64 == self.total_chunks
    }
    
    /// Get content in specific quality
    pub fn get_quality_chunks(&self, quality: Quality) -> Vec<&ContentChunk> {
        self.chunks.values()
            .filter(|chunk| chunk.quality == quality)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_content_creation() {
        let creator_id = Uuid::new_v4();
        let content = Content::new(
            "Test Video".to_string(),
            "A test video".to_string(),
            ContentType::Video,
            "TestCreator".to_string(),
            creator_id,
        );
        
        assert_eq!(content.metadata.title, "Test Video");
        assert_eq!(content.metadata.content_type, ContentType::Video);
        assert_eq!(content.metadata.creator_id, creator_id);
        assert!(!content.metadata.is_live);
    }
    
    #[test]
    fn test_live_stream_creation() {
        let creator_id = Uuid::new_v4();
        let content = Content::new_live_stream(
            "Live Stream".to_string(),
            "A live stream".to_string(),
            "Streamer".to_string(),
            creator_id,
        );
        
        assert!(content.metadata.is_live);
        assert_eq!(content.metadata.content_type, ContentType::LiveStream);
    }
    
    #[test]
    fn test_content_type_checks() {
        assert!(ContentType::Video.is_video());
        assert!(ContentType::Audio.is_audio());
        assert!(!ContentType::Audio.is_video());
        assert!(!ContentType::Video.is_audio());
    }
    
    #[test]
    fn test_rating_system() {
        let creator_id = Uuid::new_v4();
        let mut content = Content::new(
            "Test".to_string(),
            "Test".to_string(),
            ContentType::Video,
            "Creator".to_string(),
            creator_id,
        );
        
        assert!(content.add_rating(4.5).is_ok());
        assert!(content.add_rating(3.5).is_ok());
        assert_eq!(content.metadata.ratings_count, 2);
        assert_eq!(content.metadata.rating, 4.0);
        
        assert!(content.add_rating(6.0).is_err()); // Invalid rating
    }
}