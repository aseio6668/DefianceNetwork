//! Core bridge management functionality

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use crate::{
    CryptoNetwork, CryptoAmount, CrossChainTransaction, ExchangeRate, 
    BridgeConfig, TransactionStatus
};

/// Main bridge manager coordinating all cryptocurrency integrations
pub struct BridgeManager {
    config: BridgeConfig,
    networks: HashMap<CryptoNetwork, Arc<dyn BridgeNetwork>>,
    active_transactions: Arc<RwLock<HashMap<Uuid, CrossChainTransaction>>>,
    exchange_rates: Arc<RwLock<HashMap<(CryptoNetwork, CryptoNetwork), ExchangeRate>>>,
    event_sender: mpsc::UnboundedSender<BridgeEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<BridgeEvent>>,
}

/// Bridge network trait for different cryptocurrency implementations
#[async_trait::async_trait]
pub trait BridgeNetwork: Send + Sync {
    /// Get network identifier
    fn network(&self) -> CryptoNetwork;
    
    /// Check if network is available
    async fn is_available(&self) -> bool;
    
    /// Get wallet balance
    async fn get_balance(&self, address: &str) -> Result<CryptoAmount>;
    
    /// Send transaction
    async fn send_transaction(
        &self, 
        from: &str, 
        to: &str, 
        amount: CryptoAmount
    ) -> Result<BridgeTransaction>;
    
    /// Get transaction status
    async fn get_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus>;
    
    /// Validate address format
    fn validate_address(&self, address: &str) -> bool;
    
    /// Get network info
    async fn get_network_info(&self) -> Result<NetworkInfo>;
    
    /// Generate new address
    async fn generate_address(&self) -> Result<String>;
    
    /// Get transaction history
    async fn get_transaction_history(
        &self, 
        address: &str, 
        limit: Option<usize>
    ) -> Result<Vec<BridgeTransaction>>;
}

/// Bridge transaction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeTransaction {
    pub hash: String,
    pub network: CryptoNetwork,
    pub chain_id: Option<u64>, // Chain ID for EIP-155 compliance and multi-network support
    pub from_address: String,
    pub to_address: String,
    pub amount: CryptoAmount,
    pub fee: CryptoAmount,
    pub status: TransactionStatus,
    pub confirmations: u32,
    pub block_height: Option<u64>,
    pub timestamp: i64,
}

/// Network information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub network: CryptoNetwork,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub block_height: u64,
    pub difficulty: Option<f64>,
    pub hash_rate: Option<f64>,
    pub peer_count: Option<u32>,
    pub is_synced: bool,
}

/// Bridge events
#[derive(Debug, Clone)]
pub enum BridgeEvent {
    NetworkConnected { network: CryptoNetwork },
    NetworkDisconnected { network: CryptoNetwork },
    TransactionCreated { transaction_id: Uuid },
    TransactionConfirmed { transaction_id: Uuid, confirmations: u32 },
    TransactionCompleted { transaction_id: Uuid },
    TransactionFailed { transaction_id: Uuid, reason: String },
    ExchangeRateUpdated { from: CryptoNetwork, to: CryptoNetwork, rate: f64 },
    BalanceUpdated { network: CryptoNetwork, address: String, balance: CryptoAmount },
}

/// Bridge statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStats {
    pub connected_networks: usize,
    pub total_transactions: usize,
    pub pending_transactions: usize,
    pub completed_transactions: usize,
    pub failed_transactions: usize,
    pub total_volume: HashMap<CryptoNetwork, CryptoAmount>,
    pub average_confirmation_time: HashMap<CryptoNetwork, f64>, // in seconds
}

impl BridgeManager {
    /// Create new bridge manager
    pub async fn new() -> Result<Self> {
        let config = BridgeConfig::default();
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            config,
            networks: HashMap::new(),
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            exchange_rates: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Some(event_receiver),
        })
    }

    /// Create bridge manager with custom config
    pub async fn with_config(config: BridgeConfig) -> Result<Self> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        Ok(Self {
            config,
            networks: HashMap::new(),
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            exchange_rates: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            event_receiver: Some(event_receiver),
        })
    }

    /// Register a network implementation
    pub async fn register_network(&mut self, network: Arc<dyn BridgeNetwork>) -> Result<()> {
        let network_type = network.network();
        
        // Test network connectivity
        if !network.is_available().await {
            tracing::warn!("Network {:?} is not available during registration", network_type);
        }
        
        self.networks.insert(network_type.clone(), network);
        self.send_event(BridgeEvent::NetworkConnected { network: network_type.clone() }).await;
        
        tracing::info!("Registered network: {:?}", network_type);
        Ok(())
    }

    /// Remove a network
    pub async fn unregister_network(&mut self, network: CryptoNetwork) -> bool {
        if self.networks.remove(&network).is_some() {
            self.send_event(BridgeEvent::NetworkDisconnected { network: network.clone() }).await;
            tracing::info!("Unregistered network: {:?}", network);
            true
        } else {
            false
        }
    }

    /// Check if network is supported
    pub fn is_network_supported(&self, network: &CryptoNetwork) -> bool {
        self.networks.contains_key(network)
    }

    /// Get available networks
    pub fn get_available_networks(&self) -> Vec<CryptoNetwork> {
        self.networks.keys().cloned().collect()
    }

    /// Create cross-chain bridge transaction
    pub async fn create_bridge_transaction(
        &self,
        from_network: CryptoNetwork,
        to_network: CryptoNetwork,
        from_address: String,
        to_address: String,
        amount: CryptoAmount,
    ) -> Result<Uuid> {
        // Validate networks are supported
        if !self.is_network_supported(&from_network) {
            return Err(anyhow::anyhow!("Source network {:?} not supported", from_network));
        }
        if !self.is_network_supported(&to_network) {
            return Err(anyhow::anyhow!("Destination network {:?} not supported", to_network));
        }

        // Validate addresses
        let from_net = self.networks.get(&from_network).unwrap();
        let to_net = self.networks.get(&to_network).unwrap();
        
        if !from_net.validate_address(&from_address) {
            return Err(anyhow::anyhow!("Invalid source address"));
        }
        if !to_net.validate_address(&to_address) {
            return Err(anyhow::anyhow!("Invalid destination address"));
        }

        // Check minimum/maximum amounts
        if let Some(min_amount) = self.config.min_bridge_amount.get(&from_network) {
            if amount.amount < *min_amount {
                return Err(anyhow::anyhow!("Amount below minimum bridge threshold"));
            }
        }
        if let Some(max_amount) = self.config.max_bridge_amount.get(&from_network) {
            if amount.amount > *max_amount {
                return Err(anyhow::anyhow!("Amount exceeds maximum bridge threshold"));
            }
        }

        // Calculate bridge fee
        let fee = amount.percentage(self.config.bridge_fee_percentage);

        // Create transaction
        let transaction = CrossChainTransaction::new(
            from_network.clone(),
            to_network.clone(),
            from_address,
            to_address,
            amount,
            fee,
        );

        let transaction_id = transaction.id;

        // Store transaction
        {
            let mut transactions = self.active_transactions.write().await;
            transactions.insert(transaction_id, transaction);
        }

        self.send_event(BridgeEvent::TransactionCreated { transaction_id }).await;

        tracing::info!(
            "Created bridge transaction {} from {:?} to {:?}",
            transaction_id,
            from_network,
            to_network
        );

        Ok(transaction_id)
    }

    /// Execute bridge transaction
    pub async fn execute_bridge_transaction(&self, transaction_id: Uuid) -> Result<()> {
        let transaction = {
            let transactions = self.active_transactions.read().await;
            transactions.get(&transaction_id).cloned()
        };

        let mut transaction = transaction.ok_or_else(|| {
            anyhow::anyhow!("Transaction not found: {}", transaction_id)
        })?;

        if transaction.status != TransactionStatus::Pending {
            return Err(anyhow::anyhow!("Transaction is not in pending state"));
        }

        // Get source network
        let from_network = self.networks.get(&transaction.from_network)
            .ok_or_else(|| anyhow::anyhow!("Source network not available"))?;

        // Check balance
        let balance = from_network.get_balance(&transaction.from_address).await?;
        let total_cost = transaction.total_cost()?;
        
        if !balance.greater_than(&total_cost)? {
            transaction.fail("Insufficient balance".to_string());
            self.update_transaction(transaction).await;
            return Err(anyhow::anyhow!("Insufficient balance"));
        }

        // Send transaction on source network
        match from_network.send_transaction(
            &transaction.from_address,
            &self.get_bridge_address(&transaction.from_network)?,
            transaction.amount.clone(),
        ).await {
            Ok(bridge_tx) => {
                transaction.confirm(bridge_tx.hash);
                self.update_transaction(transaction.clone()).await;
                self.send_event(BridgeEvent::TransactionConfirmed { 
                    transaction_id, 
                    confirmations: bridge_tx.confirmations 
                }).await;

                // Start monitoring for confirmations
                self.monitor_transaction(transaction_id).await;
            }
            Err(e) => {
                transaction.fail(format!("Failed to send transaction: {}", e));
                self.update_transaction(transaction).await;
                self.send_event(BridgeEvent::TransactionFailed { 
                    transaction_id, 
                    reason: e.to_string() 
                }).await;
                return Err(e);
            }
        }

        Ok(())
    }

    /// Get bridge address for network
    fn get_bridge_address(&self, network: &CryptoNetwork) -> Result<String> {
        // TODO: Return actual bridge addresses for each network
        match network {
            CryptoNetwork::Paradigm => Ok("PARbridgeAddress1234567890abcdef123456789".to_string()),
            CryptoNetwork::Arceon => Ok("ARCbridgeAddress1234567890abcdef123456789".to_string()),
            _ => Err(anyhow::anyhow!("Bridge address not configured for network {:?}", network)),
        }
    }

    /// Monitor transaction for confirmations
    async fn monitor_transaction(&self, _transaction_id: Uuid) {
        // This would typically run in a background task
        // For now, we'll just mark as completed after a delay
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
            // TODO: Implement actual confirmation monitoring
        });
    }

    /// Update transaction in storage
    async fn update_transaction(&self, transaction: CrossChainTransaction) {
        let mut transactions = self.active_transactions.write().await;
        transactions.insert(transaction.id, transaction);
    }

    /// Get transaction by ID
    pub async fn get_transaction(&self, transaction_id: Uuid) -> Option<CrossChainTransaction> {
        let transactions = self.active_transactions.read().await;
        transactions.get(&transaction_id).cloned()
    }

    /// Get all transactions
    pub async fn get_all_transactions(&self) -> Vec<CrossChainTransaction> {
        let transactions = self.active_transactions.read().await;
        transactions.values().cloned().collect()
    }

    /// Get pending transactions
    pub async fn get_pending_transactions(&self) -> Vec<CrossChainTransaction> {
        let transactions = self.active_transactions.read().await;
        transactions.values()
            .filter(|tx| tx.status == TransactionStatus::Pending)
            .cloned()
            .collect()
    }

    /// Update exchange rate
    pub async fn update_exchange_rate(&self, rate: ExchangeRate) {
        let key = (rate.from_network.clone(), rate.to_network.clone());
        
        {
            let mut rates = self.exchange_rates.write().await;
            rates.insert(key, rate.clone());
        }

        self.send_event(BridgeEvent::ExchangeRateUpdated {
            from: rate.from_network,
            to: rate.to_network,
            rate: rate.rate,
        }).await;
    }

    /// Get exchange rate
    pub async fn get_exchange_rate(
        &self,
        from: &CryptoNetwork,
        to: &CryptoNetwork,
    ) -> Option<ExchangeRate> {
        let rates = self.exchange_rates.read().await;
        rates.get(&(from.clone(), to.clone())).cloned()
    }

    /// Calculate bridge cost including fees
    pub async fn calculate_bridge_cost(
        &self,
        _from_network: &CryptoNetwork,
        _to_network: &CryptoNetwork,
        amount: CryptoAmount,
    ) -> Result<(CryptoAmount, CryptoAmount)> {
        let fee = amount.percentage(self.config.bridge_fee_percentage);
        let total_cost = amount.add(&fee)?;

        Ok((fee, total_cost))
    }

    /// Get bridge statistics
    pub async fn get_stats(&self) -> BridgeStats {
        let transactions = self.active_transactions.read().await;
        
        let total_transactions = transactions.len();
        let pending_transactions = transactions.values()
            .filter(|tx| tx.status == TransactionStatus::Pending)
            .count();
        let completed_transactions = transactions.values()
            .filter(|tx| tx.status == TransactionStatus::Completed)
            .count();
        let failed_transactions = transactions.values()
            .filter(|tx| matches!(tx.status, TransactionStatus::Failed(_)))
            .count();

        // Calculate volume per network
        let mut total_volume: HashMap<CryptoNetwork, CryptoAmount> = HashMap::new();
        for tx in transactions.values() {
            if let Some(existing) = total_volume.get(&tx.amount.network) {
                if let Ok(sum) = existing.add(&tx.amount) {
                    total_volume.insert(tx.amount.network.clone(), sum);
                }
            } else {
                total_volume.insert(tx.amount.network.clone(), tx.amount.clone());
            }
        }

        BridgeStats {
            connected_networks: self.networks.len(),
            total_transactions,
            pending_transactions,
            completed_transactions,
            failed_transactions,
            total_volume,
            average_confirmation_time: HashMap::new(), // TODO: Calculate from actual data
        }
    }

    /// Take event receiver
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<BridgeEvent>> {
        self.event_receiver.take()
    }

    /// Get bridge configuration
    pub fn get_config(&self) -> &BridgeConfig {
        &self.config
    }

    /// Send bridge event
    async fn send_event(&self, event: BridgeEvent) {
        if let Err(_) = self.event_sender.send(event) {
            tracing::warn!("Failed to send bridge event - no receivers");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_manager_creation() {
        let manager = BridgeManager::new().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_network_support_check() {
        let manager = BridgeManager::new().await.unwrap();
        
        assert!(!manager.is_network_supported(&CryptoNetwork::Paradigm));
        assert!(!manager.is_network_supported(&CryptoNetwork::Bitcoin));
    }

    #[test]
    fn test_bridge_stats_creation() {
        let stats = BridgeStats {
            connected_networks: 2,
            total_transactions: 10,
            pending_transactions: 2,
            completed_transactions: 7,
            failed_transactions: 1,
            total_volume: HashMap::new(),
            average_confirmation_time: HashMap::new(),
        };

        assert_eq!(stats.connected_networks, 2);
        assert_eq!(stats.total_transactions, 10);
        assert_eq!(stats.pending_transactions, 2);
    }
}