//! Cryptographic utilities for DefianceNetwork

use std::collections::HashMap;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use aes_gcm::{Aes256Gcm, Key, Nonce, KeyInit};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use sha2::{Sha256, Digest};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use crate::error::DefianceError;

/// Cryptographic manager for DefianceNetwork
#[derive(Clone)]
pub struct CryptoManager {
    signing_key: SigningKey,
    content_keys: HashMap<Uuid, ContentKey>,
}

/// Content encryption key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentKey {
    pub content_id: Uuid,
    pub key: [u8; 32],
    pub created_at: i64,
}

/// Encrypted content with signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedContent {
    pub content_id: Uuid,
    pub encrypted_data: Vec<u8>,
    pub nonce: [u8; 12],
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub created_at: i64,
}

impl CryptoManager {
    /// Create new crypto manager with generated keys
    pub fn new() -> Result<Self> {
        let mut csprng = OsRng;
        let mut secret_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        
        Ok(Self {
            signing_key,
            content_keys: HashMap::new(),
        })
    }
    
    /// Get public key for signature verification
    pub fn get_public_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }
    
    /// Generate content encryption key
    pub fn generate_content_key(&self, content_id: Uuid) -> ContentKey {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        
        ContentKey {
            content_id,
            key,
            created_at: chrono::Utc::now().timestamp(),
        }
    }
    
    /// Store content key
    pub fn store_content_key(&mut self, content_key: ContentKey) {
        self.content_keys.insert(content_key.content_id, content_key);
    }
    
    /// Get content key
    pub fn get_content_key(&self, content_id: &Uuid) -> Option<&ContentKey> {
        self.content_keys.get(content_id)
    }

    /// Encrypt data using a default internal key.
    /// This is useful for transient data that doesn't need a specific content key.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let key_bytes = self.signing_key.to_bytes();
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes[..32]);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(b"unique nonce"); // NOTE: In a real application, this MUST be unique for each encryption
        cipher.encrypt(nonce, data).map_err(|e| DefianceError::Crypto(e.to_string()).into())
    }

    /// Decrypt data using the default internal key.
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        let key_bytes = self.signing_key.to_bytes();
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes[..32]);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(b"unique nonce"); // NOTE: This must match the nonce used for encryption
        cipher.decrypt(nonce, encrypted_data).map_err(|e| DefianceError::Crypto(e.to_string()).into())
    }
    
    /// Encrypt content with signature
    pub fn encrypt_content(
        &self,
        content_id: Uuid,
        data: &[u8],
        content_key: &ContentKey,
    ) -> Result<EncryptedContent> {
        // Generate nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // Encrypt data
        let key = Key::<Aes256Gcm>::from_slice(&content_key.key);
        let cipher = Aes256Gcm::new(key);
        let encrypted_data = cipher
            .encrypt(nonce, data)
            .map_err(|e| DefianceError::Crypto(format!("Encryption failed: {}", e)))?;
        
        // Sign encrypted data
        let signature = self.signing_key.sign(&encrypted_data);
        let public_key = self.get_public_key();
        
        Ok(EncryptedContent {
            content_id,
            encrypted_data,
            nonce: nonce_bytes,
            signature: signature.to_bytes().to_vec(),
            public_key: public_key.to_vec(),
            created_at: chrono::Utc::now().timestamp(),
        })
    }
    
    /// Decrypt and verify content
    pub fn decrypt_content(
        &self,
        encrypted_content: &EncryptedContent,
        content_key: &ContentKey,
    ) -> Result<Vec<u8>> {
        // Verify signature
        let public_key = VerifyingKey::try_from(&encrypted_content.public_key[..])
            .map_err(|e| DefianceError::Crypto(format!("Invalid public key: {}", e)))?;
        
        let signature = Signature::try_from(&encrypted_content.signature[..])
            .map_err(|e| DefianceError::Crypto(format!("Invalid signature: {}", e)))?;
        
        public_key
            .verify(&encrypted_content.encrypted_data, &signature)
            .map_err(|e| DefianceError::Crypto(format!("Signature verification failed: {}", e)))?;
        
        // Decrypt data
        let key = Key::<Aes256Gcm>::from_slice(&content_key.key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(&encrypted_content.nonce);
        
        let decrypted_data = cipher
            .decrypt(nonce, encrypted_content.encrypted_data.as_ref())
            .map_err(|e| DefianceError::Crypto(format!("Decryption failed: {}", e)))?;
        
        Ok(decrypted_data)
    }
    
    /// Sign arbitrary data
    pub fn sign_data(&self, data: &[u8]) -> Result<Signature> {
        Ok(self.signing_key.sign(data))
    }
    
    /// Verify signature on data
    pub fn verify_signature(
        &self,
        data: &[u8],
        signature: &Signature,
        public_key: &VerifyingKey,
    ) -> Result<()> {
        Ok(public_key
            .verify(data, signature)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?)
    }
    
    /// Hash data using SHA-256
    pub fn hash_data(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Generate content checksum
    pub fn generate_content_checksum(&self, content_id: Uuid, data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(content_id.as_bytes());
        hasher.update(data);
        hasher.finalize().into()
    }
    
    /// Verify content integrity
    pub fn verify_content_integrity(
        &self,
        content_id: Uuid,
        data: &[u8],
        expected_checksum: &[u8; 32],
    ) -> bool {
        let calculated_checksum = self.generate_content_checksum(content_id, data);
        calculated_checksum == *expected_checksum
    }
    
    /// Generate secure random bytes
    pub fn generate_random_bytes(length: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; length];
        OsRng.fill_bytes(&mut bytes);
        bytes
    }
    
    /// Derive key from password (for user key derivation)
    pub fn derive_key_from_password(password: &str, salt: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(salt);
        
        // Simple key derivation - in production, use a proper KDF like PBKDF2 or Argon2
        let mut result = hasher.finalize();
        
        // Additional rounds for better security
        for _ in 0..10000 {
            let mut hasher = Sha256::new();
            hasher.update(&result);
            hasher.update(salt);
            result = hasher.finalize();
        }
        
        result.into()
    }
}

/// Content encryption utilities
impl EncryptedContent {
    /// Verify the encrypted content's signature
    pub fn verify_signature(&self) -> Result<()> {
        let public_key = VerifyingKey::try_from(&self.public_key[..])
            .map_err(|e| anyhow::anyhow!("Invalid public key: {}", e))?;
        
        let signature = Signature::try_from(&self.signature[..])
            .map_err(|e| anyhow::anyhow!("Invalid signature: {}", e))?;
        
        Ok(public_key
            .verify(&self.encrypted_data, &signature)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?)
    }
    
    /// Get the size of encrypted data
    pub fn encrypted_size(&self) -> usize {
        self.encrypted_data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crypto_manager_creation() {
        let crypto = CryptoManager::new();
        assert!(crypto.is_ok());
    }
    
    #[test]
    fn test_content_encryption_decryption() {
        let crypto = CryptoManager::new().unwrap();
        let content_id = Uuid::new_v4();
        let content_key = crypto.generate_content_key(content_id);
        
        let original_data = b"Hello, DefianceNetwork!";
        
        let encrypted = crypto.encrypt_content(content_id, original_data, &content_key).unwrap();
        assert!(encrypted.verify_signature().is_ok());
        
        let decrypted = crypto.decrypt_content(&encrypted, &content_key).unwrap();
        assert_eq!(original_data, decrypted.as_slice());
    }
    
    #[test]
    fn test_signature_verification() {
        let crypto = CryptoManager::new().unwrap();
        let data = b"test data for signing";
        
        let signature = crypto.sign_data(data).unwrap();
        let public_key = VerifyingKey::from_bytes(&crypto.get_public_key()).unwrap();
        
        assert!(crypto.verify_signature(data, &signature, &public_key).is_ok());
        
        // Test with wrong data
        let wrong_data = b"wrong data";
        assert!(crypto.verify_signature(wrong_data, &signature, &public_key).is_err());
    }
    
    #[test]
    fn test_content_integrity() {
        let crypto = CryptoManager::new().unwrap();
        let content_id = Uuid::new_v4();
        let data = b"test content data";
        
        let checksum = crypto.generate_content_checksum(content_id, data);
        assert!(crypto.verify_content_integrity(content_id, data, &checksum));
        
        // Test with modified data
        let modified_data = b"modified content data";
        assert!(!crypto.verify_content_integrity(content_id, modified_data, &checksum));
    }
    
    #[test]
    fn test_key_derivation() {
        let password = "test_password";
        let salt = b"random_salt_1234";
        
        let key1 = CryptoManager::derive_key_from_password(password, salt);
        let key2 = CryptoManager::derive_key_from_password(password, salt);
        
        assert_eq!(key1, key2); // Same password and salt should produce same key
        
        let key3 = CryptoManager::derive_key_from_password("different_password", salt);
        assert_ne!(key1, key3); // Different password should produce different key
    }
}