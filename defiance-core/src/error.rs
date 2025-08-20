//! Error types for DefianceNetwork core functionality

use thiserror::Error;

/// Main error type for DefianceNetwork operations
#[derive(Error, Debug)]
pub enum DefianceError {
    #[error("Network error: {0}")]
    Network(#[from] anyhow::Error),
    
    #[error("Streaming error: {0}")]
    Streaming(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Cryptography error: {0}")]
    Crypto(String),
    
    #[error("User management error: {0}")]
    User(String),
    
    #[error("Content error: {0}")]
    Content(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Peer discovery error: {0}")]
    Discovery(String),
    
    #[error("Bridge error: {0}")]
    Bridge(String),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Already exists: {0}")]
    AlreadyExists(String),
}

pub type Result<T> = std::result::Result<T, DefianceError>;