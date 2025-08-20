//! # .augy File Format
//! 
//! Specification and parser for the .augy file format used by Audigy.
//! The .augy format provides structured metadata for audio content discovery
//! across decentralized networks.

use std::path::Path;
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc, Datelike, Timelike};

/// Complete .augy file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugyFile {
    /// File format version
    pub version: String,
    /// File metadata
    pub metadata: AugyFileMetadata,
    /// List of audio content entries
    pub entries: Vec<AugyEntry>,
    /// Signature for file verification (optional)
    pub signature: Option<String>,
}

/// Metadata about the .augy file itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugyFileMetadata {
    /// Title/name of this collection
    pub title: String,
    /// Description of this collection
    pub description: String,
    /// Creator of this .augy file
    pub creator: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Language code (e.g., "en", "es", "fr")
    pub language: Option<String>,
    /// Tags for this collection
    pub tags: Vec<String>,
    /// Website or source URL
    pub website: Option<Url>,
    /// Contact information
    pub contact: Option<String>,
}

/// Individual audio content entry in .augy file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugyEntry {
    /// Unique identifier for this entry
    pub id: Uuid,
    /// Audio content metadata
    pub metadata: AudioMetadata,
    /// List of URIs where this content can be accessed
    pub uri_points: Vec<Url>,
    /// Backup/mirror URIs
    pub backup_uris: Vec<Url>,
    /// File checksums for verification
    pub checksums: Vec<FileChecksum>,
    /// Availability schedule (for live content)
    pub schedule: Option<ContentSchedule>,
}

/// Audio content metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Title of the audio content
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Content author/creator
    pub author: String,
    /// Publisher/distributor
    pub publisher: Option<String>,
    /// Duration in seconds
    pub duration: Option<u64>,
    /// File size in bytes
    pub size: Option<u64>,
    /// Audio genre
    pub genre: AudioGenre,
    /// Content type
    pub audio_type: AudioType,
    /// Language code
    pub language: Option<String>,
    /// Publication date
    pub published_at: Option<DateTime<Utc>>,
    /// Episode number (for series content)
    pub episode_number: Option<u32>,
    /// Season number (for series content)
    pub season_number: Option<u32>,
    /// Series/podcast name
    pub series: Option<String>,
    /// Keywords/tags
    pub keywords: Vec<String>,
    /// Content rating (e.g., "G", "PG", "R")
    pub rating: Option<String>,
    /// Transcript availability
    pub has_transcript: bool,
    /// Thumbnail/cover art URL
    pub thumbnail: Option<Url>,
    /// Related content IDs
    pub related_content: Vec<Uuid>,
}

/// Audio content genres
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AudioGenre {
    Podcast,
    AudioBook,
    Lecture,
    Educational,
    Documentary,
    News,
    Interview,
    Discussion,
    Meditation,
    Music,
    Comedy,
    Drama,
    SciFi,
    Fantasy,
    Mystery,
    History,
    Science,
    Technology,
    Philosophy,
    Religion,
    Health,
    Business,
    Politics,
    Sports,
    Arts,
    Literature,
    Language,
    Kids,
    Fiction,
    NonFiction,
    Biography,
    SelfHelp,
    Other(String),
}

/// Audio content types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioType {
    /// Single audio file
    Single,
    /// Part of a series/podcast
    Episode,
    /// Multi-part content (audiobook chapters)
    Chapter,
    /// Live stream
    LiveStream,
    /// Short audio clip
    Clip,
    /// Full album/collection
    Album,
}

/// File checksum for content verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChecksum {
    /// Hash algorithm used (e.g., "sha256", "md5")
    pub algorithm: String,
    /// Hex-encoded hash value
    pub hash: String,
    /// File size in bytes
    pub size: u64,
}

/// Content scheduling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSchedule {
    /// When content becomes available
    pub available_from: Option<DateTime<Utc>>,
    /// When content expires/is removed
    pub available_until: Option<DateTime<Utc>>,
    /// Days of week when available (0=Sunday, 6=Saturday)
    pub available_days: Option<Vec<u8>>,
    /// Time range when available (24-hour format)
    pub available_hours: Option<(u8, u8)>,
    /// Timezone for schedule
    pub timezone: Option<String>,
}

impl AugyFile {
    /// Create a new empty .augy file
    pub fn new(title: String, description: String, creator: String) -> Self {
        let now = Utc::now();
        
        Self {
            version: "1.0".to_string(),
            metadata: AugyFileMetadata {
                title,
                description,
                creator,
                created_at: now,
                updated_at: now,
                language: None,
                tags: Vec::new(),
                website: None,
                contact: None,
            },
            entries: Vec::new(),
            signature: None,
        }
    }

    /// Load .augy file from filesystem
    pub async fn load_from_file(path: &Path) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        Self::parse_from_string(&content)
    }

    /// Parse .augy file from string content
    pub fn parse_from_string(content: &str) -> Result<Self> {
        // Try JSON first
        if let Ok(augy_file) = serde_json::from_str::<AugyFile>(content) {
            return Ok(augy_file);
        }

        // Try YAML
        if let Ok(augy_file) = serde_yaml::from_str::<AugyFile>(content) {
            return Ok(augy_file);
        }

        // Try TOML
        if let Ok(augy_file) = toml::from_str::<AugyFile>(content) {
            return Ok(augy_file);
        }

        Err(anyhow::anyhow!("Failed to parse .augy file - unsupported format"))
    }

    /// Save .augy file to filesystem in JSON format
    pub async fn save_to_file(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Save .augy file in YAML format
    pub async fn save_to_yaml(&self, path: &Path) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Add an audio entry to the file
    pub fn add_entry(&mut self, entry: AugyEntry) {
        self.entries.push(entry);
        self.metadata.updated_at = Utc::now();
    }

    /// Remove an entry by ID
    pub fn remove_entry(&mut self, id: Uuid) -> bool {
        let initial_len = self.entries.len();
        self.entries.retain(|entry| entry.id != id);
        
        if self.entries.len() < initial_len {
            self.metadata.updated_at = Utc::now();
            true
        } else {
            false
        }
    }

    /// Find entry by ID
    pub fn find_entry(&self, id: Uuid) -> Option<&AugyEntry> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// Find entries by genre
    pub fn find_by_genre(&self, genre: AudioGenre) -> Vec<&AugyEntry> {
        self.entries.iter()
            .filter(|entry| entry.metadata.genre == genre)
            .collect()
    }

    /// Search entries by text query
    pub fn search(&self, query: &str) -> Vec<&AugyEntry> {
        let query_lower = query.to_lowercase();
        
        self.entries.iter()
            .filter(|entry| {
                entry.metadata.title.to_lowercase().contains(&query_lower) ||
                entry.metadata.description.to_lowercase().contains(&query_lower) ||
                entry.metadata.author.to_lowercase().contains(&query_lower) ||
                entry.metadata.keywords.iter().any(|k| k.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Validate file integrity
    pub fn validate(&self) -> Result<()> {
        // Check version compatibility
        if self.version != "1.0" {
            return Err(anyhow::anyhow!("Unsupported .augy version: {}", self.version));
        }

        // Validate entries
        for entry in &self.entries {
            entry.validate()?;
        }

        Ok(())
    }

    /// Get all unique genres in this file
    pub fn get_genres(&self) -> Vec<AudioGenre> {
        let mut genres: Vec<_> = self.entries.iter()
            .map(|entry| entry.metadata.genre.clone())
            .collect();
        genres.sort();
        genres.dedup();
        genres
    }

    /// Get total duration of all content
    pub fn get_total_duration(&self) -> u64 {
        self.entries.iter()
            .filter_map(|entry| entry.metadata.duration)
            .sum()
    }

    /// Get total size of all content
    pub fn get_total_size(&self) -> u64 {
        self.entries.iter()
            .filter_map(|entry| entry.metadata.size)
            .sum()
    }
}

impl AugyEntry {
    /// Create a new audio entry
    pub fn new(metadata: AudioMetadata, uri_points: Vec<Url>) -> Self {
        Self {
            id: Uuid::new_v4(),
            metadata,
            uri_points,
            backup_uris: Vec::new(),
            checksums: Vec::new(),
            schedule: None,
        }
    }

    /// Add a backup URI
    pub fn add_backup_uri(&mut self, uri: Url) {
        if !self.backup_uris.contains(&uri) {
            self.backup_uris.push(uri);
        }
    }

    /// Add a file checksum
    pub fn add_checksum(&mut self, algorithm: String, hash: String, size: u64) {
        let checksum = FileChecksum {
            algorithm,
            hash,
            size,
        };
        self.checksums.push(checksum);
    }

    /// Set content schedule
    pub fn set_schedule(&mut self, schedule: ContentSchedule) {
        self.schedule = Some(schedule);
    }

    /// Check if content is currently available based on schedule
    pub fn is_available_now(&self) -> bool {
        if let Some(ref schedule) = self.schedule {
            let now = Utc::now();

            // Check date range
            if let Some(from) = schedule.available_from {
                if now < from {
                    return false;
                }
            }
            
            if let Some(until) = schedule.available_until {
                if now > until {
                    return false;
                }
            }

            // Check day of week
            if let Some(ref days) = schedule.available_days {
                let weekday = now.weekday().num_days_from_sunday() as u8;
                if !days.contains(&weekday) {
                    return false;
                }
            }

            // Check hour range
            if let Some((start_hour, end_hour)) = schedule.available_hours {
                let current_hour = now.hour() as u8;
                if current_hour < start_hour || current_hour >= end_hour {
                    return false;
                }
            }
        }

        true
    }

    /// Validate entry data
    pub fn validate(&self) -> Result<()> {
        if self.metadata.title.trim().is_empty() {
            return Err(anyhow::anyhow!("Entry title cannot be empty"));
        }

        if self.uri_points.is_empty() {
            return Err(anyhow::anyhow!("Entry must have at least one URI point"));
        }

        // Validate URLs
        for uri in &self.uri_points {
            if uri.scheme() != "http" && uri.scheme() != "https" && uri.scheme() != "ipfs" {
                return Err(anyhow::anyhow!("Unsupported URI scheme: {}", uri.scheme()));
            }
        }

        Ok(())
    }
}

impl AudioMetadata {
    /// Create minimal audio metadata
    pub fn new(title: String, author: String, genre: AudioGenre, audio_type: AudioType) -> Self {
        Self {
            title,
            description: String::new(),
            author,
            publisher: None,
            duration: None,
            size: None,
            genre,
            audio_type,
            language: None,
            published_at: None,
            episode_number: None,
            season_number: None,
            series: None,
            keywords: Vec::new(),
            rating: None,
            has_transcript: false,
            thumbnail: None,
            related_content: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: u64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set size
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    /// Add keyword
    pub fn add_keyword(&mut self, keyword: String) {
        if !self.keywords.contains(&keyword) {
            self.keywords.push(keyword);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augy_file_creation() {
        let augy_file = AugyFile::new(
            "Test Podcast Collection".to_string(),
            "A collection of test podcasts".to_string(),
            "Test Creator".to_string(),
        );

        assert_eq!(augy_file.metadata.title, "Test Podcast Collection");
        assert_eq!(augy_file.version, "1.0");
        assert!(augy_file.entries.is_empty());
    }

    #[test]
    fn test_audio_metadata_creation() {
        let metadata = AudioMetadata::new(
            "Test Episode".to_string(),
            "Test Author".to_string(),
            AudioGenre::Podcast,
            AudioType::Episode,
        );

        assert_eq!(metadata.title, "Test Episode");
        assert_eq!(metadata.author, "Test Author");
        assert_eq!(metadata.genre, AudioGenre::Podcast);
        assert_eq!(metadata.audio_type, AudioType::Episode);
    }

    #[test]
    fn test_entry_validation() {
        let metadata = AudioMetadata::new(
            "Test".to_string(),
            "Author".to_string(),
            AudioGenre::Podcast,
            AudioType::Single,
        );

        let uri = Url::parse("https://example.com/audio.mp3").unwrap();
        let entry = AugyEntry::new(metadata, vec![uri]);

        assert!(entry.validate().is_ok());

        // Test empty title
        let mut bad_metadata = entry.metadata.clone();
        bad_metadata.title = "".to_string();
        let bad_entry = AugyEntry::new(bad_metadata, vec![]);
        assert!(bad_entry.validate().is_err());
    }

    #[test]
    fn test_genre_ordering() {
        let mut genres = vec![
            AudioGenre::Podcast,
            AudioGenre::AudioBook,
            AudioGenre::Educational,
        ];
        genres.sort();

        assert_eq!(genres[0], AudioGenre::AudioBook);
        assert_eq!(genres[1], AudioGenre::Educational);
        assert_eq!(genres[2], AudioGenre::Podcast);
    }

    #[tokio::test]
    async fn test_json_serialization() {
        let mut augy_file = AugyFile::new(
            "Test".to_string(),
            "Description".to_string(),
            "Creator".to_string(),
        );

        let metadata = AudioMetadata::new(
            "Episode 1".to_string(),
            "Author".to_string(),
            AudioGenre::Podcast,
            AudioType::Episode,
        );

        let uri = Url::parse("https://example.com/episode1.mp3").unwrap();
        let entry = AugyEntry::new(metadata, vec![uri]);
        augy_file.add_entry(entry);

        let json = serde_json::to_string_pretty(&augy_file).unwrap();
        let parsed = serde_json::from_str::<AugyFile>(&json).unwrap();

        assert_eq!(augy_file.metadata.title, parsed.metadata.title);
        assert_eq!(augy_file.entries.len(), parsed.entries.len());
    }
}