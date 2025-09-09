//! # Arceon Blockchain Integration
//! 
//! Complete integration with the Arceon cryptocurrency network.
//! Provides wallet management, transaction processing, and cross-chain bridge functionality.

use async_trait::async_trait;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::bridge_core::{BridgeNetwork, BridgeTransaction, NetworkInfo};
use crate::{CryptoNetwork, CryptoAmount, TransactionStatus};

pub mod rpc_client;
pub mod wallet;
pub mod types;
pub mod bridge;

pub use rpc_client::ArceonRpcClient;
pub use wallet::{ArceonWallet, ArceonAccount};
pub use types::{ArceonBlock, ArceonTransaction, ArceonAddress};
pub use bridge::ArceonBridge;

/// Main Arceon network implementation for DefianceNetwork
pub struct ArceonNetwork {
    rpc_client: Arc<ArceonRpcClient>,
    wallet: Arc<RwLock<ArceonWallet>>,
    config: ArceonConfig,
    transaction_cache: Arc<RwLock<HashMap<String, BridgeTransaction>>>,
}

/// Configuration for Arceon network connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonConfig {
    pub node_urls: Vec<String>,
    pub api_key: Option<String>,
    pub network_type: ArceonNetworkType,
    pub default_fee_rate: u64, // satoshis per byte
    pub confirmation_target: u32,
    pub enable_wallet: bool,
    pub wallet_path: Option<String>,
    pub connection_timeout_seconds: u64,
    pub retry_attempts: u32,
}

/// Arceon network types (mainnet, testnet, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArceonNetworkType {
    Mainnet,
    Testnet,
    Devnet,
}

impl Default for ArceonConfig {
    fn default() -> Self {
        Self {
            node_urls: vec![
                "https://rpc.arceon.network".to_string(),
                "https://backup-rpc.arceon.network".to_string(),
            ],
            api_key: None,
            network_type: ArceonNetworkType::Mainnet,
            default_fee_rate: 1000, // 1000 satoshis per byte
            confirmation_target: 6,
            enable_wallet: true,
            wallet_path: None,
            connection_timeout_seconds: 30,
            retry_attempts: 3,
        }
    }
}

impl ArceonNetwork {
    /// Create new Arceon network instance
    pub async fn new(config: ArceonConfig) -> Result<Self> {
        tracing::info!("Initializing Arceon network with {} nodes", config.node_urls.len());
        
        // Initialize RPC client
        let rpc_client = Arc::new(ArceonRpcClient::new(config.clone()).await?);
        
        // Initialize wallet if enabled
        let wallet = if config.enable_wallet {
            Arc::new(RwLock::new(ArceonWallet::new(config.clone()).await?))
        } else {
            Arc::new(RwLock::new(ArceonWallet::readonly()))
        };
        
        let network = Self {
            rpc_client,
            wallet,
            config,
            transaction_cache: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Test connectivity
        if !network.test_connectivity().await {
            return Err(anyhow!("Failed to connect to Arceon network"));
        }
        
        tracing::info!("Arceon network initialized successfully");
        Ok(network)
    }
    
    /// Test connectivity to Arceon network
    pub async fn test_connectivity(&self) -> bool {
        match self.rpc_client.get_blockchain_info().await {
            Ok(info) => {
                tracing::info!("Connected to Arceon network: {} blocks", info.blocks);
                true
            }
            Err(e) => {
                tracing::error!("Failed to connect to Arceon network: {}", e);
                false
            }
        }
    }
    
    /// Get current Arceon network status
    pub async fn get_status(&self) -> Result<ArceonNetworkStatus> {
        let blockchain_info = self.rpc_client.get_blockchain_info().await?;
        let network_info = self.rpc_client.get_network_info().await?;
        
        Ok(ArceonNetworkStatus {
            is_connected: true,
            current_block: blockchain_info.blocks,
            peer_count: network_info.connections,
            is_synced: blockchain_info.verification_progress >= 0.99,
            chain: blockchain_info.chain,
            difficulty: blockchain_info.difficulty,
            network_hash_rate: network_info.hash_rate,
            mempool_size: self.rpc_client.get_mempool_info().await?.size,
        })
    }
    
    /// Create a new Arceon wallet account
    pub async fn create_account(&self, name: String) -> Result<ArceonAccount> {
        let mut wallet = self.wallet.write().await;
        wallet.create_account(name).await
    }
    
    /// Import existing account from private key
    pub async fn import_account(&self, name: String, private_key: String) -> Result<ArceonAccount> {
        let mut wallet = self.wallet.write().await;
        wallet.import_account(name, private_key).await
    }
    
    /// List all wallet accounts
    pub async fn list_accounts(&self) -> Result<Vec<ArceonAccount>> {
        let wallet = self.wallet.read().await;
        Ok(wallet.list_accounts())
    }
    
    /// Send Arceon transaction with enhanced features
    pub async fn send_transaction_enhanced(
        &self,
        from_account: &str,
        to_address: &str,
        amount: u64,
        options: TransactionOptions,
    ) -> Result<BridgeTransaction> {
        // Validate addresses
        if !self.validate_address(to_address) {
            return Err(anyhow!("Invalid destination address"));
        }
        
        // Get account
        let wallet = self.wallet.read().await;
        let account = wallet.get_account(from_account)?;
        
        // Calculate fee
        let fee = self.calculate_fee(&options).await?;
        
        // Check balance
        let balance = self.get_balance(&account.address).await?;
        let total_needed = amount + fee.amount_smallest_unit() as u64;
        
        if balance.amount_smallest_unit() < total_needed as i64 {
            return Err(anyhow!("Insufficient balance: need {} ARC, have {} ARC", 
                              total_needed as f64 / 100_000_000.0, 
                              balance.amount_smallest_unit() as f64 / 100_000_000.0));
        }
        
        // Create and sign transaction
        let tx_hash = self.rpc_client.send_transaction(
            &account.address,
            to_address,
            amount,
            fee.amount_smallest_unit() as u64,
            &account.private_key,
            &options,
        ).await?;
        
        let bridge_tx = BridgeTransaction {
            hash: tx_hash.clone(),
            network: CryptoNetwork::Arceon,
            chain_id: CryptoNetwork::Arceon.chain_id(),
            from_address: account.address.clone(),
            to_address: to_address.to_string(),
            amount: CryptoAmount::new(CryptoNetwork::Arceon, amount as i64, 8),
            fee,
            status: TransactionStatus::Pending,
            confirmations: 0,
            block_height: None,
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        // Cache transaction
        let mut cache = self.transaction_cache.write().await;
        cache.insert(tx_hash.clone(), bridge_tx.clone());
        
        tracing::info!("Sent Arceon transaction: {} ARC from {} to {} ({})",
                      amount as f64 / 100_000_000.0, account.address, to_address, tx_hash);
        
        Ok(bridge_tx)
    }
    
    /// Calculate transaction fee based on options
    async fn calculate_fee(&self, options: &TransactionOptions) -> Result<CryptoAmount> {
        let fee_rate = options.fee_rate.unwrap_or(self.config.default_fee_rate);
        
        // Estimate transaction size (simplified)
        let estimated_size = 250; // bytes for typical transaction
        let fee_amount = fee_rate * estimated_size;
        
        Ok(CryptoAmount::new(CryptoNetwork::Arceon, fee_amount as i64, 8))
    }
    
    /// Get detailed transaction information
    pub async fn get_transaction_details(&self, tx_hash: &str) -> Result<ArceonTransactionDetails> {
        // Check cache first
        {
            let cache = self.transaction_cache.read().await;
            if let Some(cached_tx) = cache.get(tx_hash) {
                if let Ok(status) = self.rpc_client.get_transaction_status(tx_hash).await {
                    return Ok(ArceonTransactionDetails {
                        hash: tx_hash.to_string(),
                        status,
                        confirmations: cached_tx.confirmations,
                        block_height: cached_tx.block_height,
                        amount: cached_tx.amount.clone(),
                        fee: cached_tx.fee.clone(),
                        from_address: cached_tx.from_address.clone(),
                        to_address: cached_tx.to_address.clone(),
                        timestamp: cached_tx.timestamp,
                        raw_transaction: None,
                    });
                }
            }
        }
        
        // Fetch from network
        let tx_info = self.rpc_client.get_transaction_info(tx_hash).await?;
        
        Ok(ArceonTransactionDetails {
            hash: tx_hash.to_string(),
            status: tx_info.status,
            confirmations: tx_info.confirmations,
            block_height: tx_info.block_height,
            amount: CryptoAmount::new(CryptoNetwork::Arceon, tx_info.amount, 8),
            fee: CryptoAmount::new(CryptoNetwork::Arceon, tx_info.fee, 8),
            from_address: tx_info.from_address,
            to_address: tx_info.to_address,
            timestamp: tx_info.timestamp,
            raw_transaction: Some(tx_info.raw_transaction),
        })
    }
    
    /// Monitor transactions for confirmations
    pub async fn start_transaction_monitor(&self) -> Result<()> {
        let cache = Arc::clone(&self.transaction_cache);
        let rpc_client = Arc::clone(&self.rpc_client);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let tx_hashes: Vec<String> = {
                    let cache_read = cache.read().await;
                    cache_read.keys()
                        .filter(|hash| {
                            if let Some(tx) = cache_read.get(*hash) {
                                tx.status != TransactionStatus::Confirmed && 
                                !matches!(tx.status, TransactionStatus::Failed(_))
                            } else {
                                false
                            }
                        })
                        .cloned()
                        .collect()
                };
                
                for tx_hash in tx_hashes {
                    if let Ok(status) = rpc_client.get_transaction_status(&tx_hash).await {
                        let mut cache_write = cache.write().await;
                        if let Some(tx) = cache_write.get_mut(&tx_hash) {
                            tx.status = status;
                            if let Ok(confirmations) = rpc_client.get_transaction_confirmations(&tx_hash).await {
                                tx.confirmations = confirmations;
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
}

/// Transaction options for enhanced sending
#[derive(Debug, Clone)]
pub struct TransactionOptions {
    pub fee_rate: Option<u64>,
    pub confirmation_target: Option<u32>,
    pub replace_by_fee: bool,
    pub chain_id: Option<u64>, // Chain ID for multi-network support
    pub memo: Option<String>,
}

impl Default for TransactionOptions {
    fn default() -> Self {
        Self {
            fee_rate: None,
            confirmation_target: None,
            replace_by_fee: false,
            chain_id: None,
            memo: None,
        }
    }
}

/// Current Arceon network status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonNetworkStatus {
    pub is_connected: bool,
    pub current_block: u64,
    pub peer_count: u32,
    pub is_synced: bool,
    pub chain: String,
    pub difficulty: f64,
    pub network_hash_rate: Option<f64>,
    pub mempool_size: u32,
}

/// Detailed transaction information
#[derive(Debug, Clone)]
pub struct ArceonTransactionDetails {
    pub hash: String,
    pub status: TransactionStatus,
    pub confirmations: u32,
    pub block_height: Option<u64>,
    pub amount: CryptoAmount,
    pub fee: CryptoAmount,
    pub from_address: String,
    pub to_address: String,
    pub timestamp: i64,
    pub raw_transaction: Option<String>,
}

#[async_trait]
impl BridgeNetwork for ArceonNetwork {
    fn network(&self) -> CryptoNetwork {
        CryptoNetwork::Arceon
    }
    
    async fn is_available(&self) -> bool {
        self.test_connectivity().await
    }
    
    async fn get_balance(&self, address: &str) -> Result<CryptoAmount> {
        let balance = self.rpc_client.get_balance(address).await?;
        Ok(CryptoAmount::new(CryptoNetwork::Arceon, balance, 8))
    }
    
    async fn send_transaction(&self, from: &str, to: &str, amount: CryptoAmount) -> Result<BridgeTransaction> {
        // Find account by address
        let wallet = self.wallet.read().await;
        let account = wallet.get_account_by_address(from)?;
        drop(wallet);
        
        let mut tx_options = TransactionOptions::default();
        tx_options.chain_id = CryptoNetwork::Arceon.chain_id();
        
        self.send_transaction_enhanced(
            &account.name,
            to,
            amount.amount_smallest_unit() as u64,
            tx_options,
        ).await
    }
    
    async fn get_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus> {
        // Check cache first
        {
            let cache = self.transaction_cache.read().await;
            if let Some(tx) = cache.get(tx_hash) {
                return Ok(tx.status.clone());
            }
        }
        
        // Query network
        self.rpc_client.get_transaction_status(tx_hash).await
    }
    
    fn validate_address(&self, address: &str) -> bool {
        // Arceon addresses start with 'ARC' and are 43 characters long
        if !address.starts_with("ARC") || address.len() != 43 {
            return false;
        }
        
        // Additional validation could include checksum verification
        address.chars().all(|c| c.is_alphanumeric())
    }
    
    async fn get_network_info(&self) -> Result<NetworkInfo> {
        let status = self.get_status().await?;
        
        Ok(NetworkInfo {
            network: CryptoNetwork::Arceon,
            name: "Arceon".to_string(),
            symbol: "ARC".to_string(),
            decimals: 8,
            block_height: status.current_block,
            difficulty: Some(status.difficulty),
            hash_rate: status.network_hash_rate,
            peer_count: Some(status.peer_count),
            is_synced: status.is_synced,
        })
    }
    
    async fn generate_address(&self) -> Result<String> {
        let mut wallet = self.wallet.write().await;
        let account = wallet.create_account(format!("auto_generated_{}", Uuid::new_v4())).await?;
        Ok(account.address)
    }
    
    async fn get_transaction_history(&self, address: &str, limit: Option<usize>) -> Result<Vec<BridgeTransaction>> {
        let limit = limit.unwrap_or(100);
        let transactions = self.rpc_client.get_transaction_history(address, limit).await?;
        
        let mut bridge_transactions = Vec::new();
        for tx in transactions {
            bridge_transactions.push(BridgeTransaction {
                hash: tx.hash,
                network: CryptoNetwork::Arceon,
                chain_id: CryptoNetwork::Arceon.chain_id(),
                from_address: tx.from_address,
                to_address: tx.to_address,
                amount: CryptoAmount::new(CryptoNetwork::Arceon, tx.amount, 8),
                fee: CryptoAmount::new(CryptoNetwork::Arceon, tx.fee, 8),
                status: tx.status,
                confirmations: tx.confirmations,
                block_height: tx.block_height,
                timestamp: tx.timestamp,
            });
        }
        
        Ok(bridge_transactions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_arceon_config_default() {
        let config = ArceonConfig::default();
        assert_eq!(config.network_type, ArceonNetworkType::Mainnet);
        assert!(config.enable_wallet);
        assert_eq!(config.confirmation_target, 6);
    }

    #[test]
    fn test_address_validation() {
        // We can't test the full network without a real connection,
        // but we can test address validation logic
        let dummy_config = ArceonConfig::default();
        
        // Create a mock network (this would need modification for actual testing)
        // For now, just test the validation logic independently
        
        assert!(validate_arceon_address("ARC1234567890abcdef1234567890abcdef12345678"));
        assert!(!validate_arceon_address("BTC1234567890abcdef1234567890abcdef12345678"));
        assert!(!validate_arceon_address("ARC123")); // too short
        assert!(!validate_arceon_address("ARC1234567890abcdef1234567890abcdef123456789")); // too long
    }
    
    fn validate_arceon_address(address: &str) -> bool {
        if !address.starts_with("ARC") || address.len() != 43 {
            return false;
        }
        address.chars().all(|c| c.is_alphanumeric())
    }
}