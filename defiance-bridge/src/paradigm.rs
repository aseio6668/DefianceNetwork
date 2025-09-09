//! Paradigm cryptocurrency integration

use async_trait::async_trait;
use anyhow::Result;
use crate::bridge_core::{BridgeNetwork, BridgeTransaction, NetworkInfo};
use crate::{CryptoNetwork, CryptoAmount, TransactionStatus};
use crate::paradigm_api::{ParadigmApiClient, BroadcastRequest};

/// Paradigm network implementation
pub struct ParadigmNetwork {
    api_client: ParadigmApiClient,
}

impl ParadigmNetwork {
    pub fn new(node_url: String, api_key: Option<String>) -> Self {
        let api_client = ParadigmApiClient::new(node_url, api_key);
        Self { api_client }
    }
}

#[async_trait]
impl BridgeNetwork for ParadigmNetwork {
    fn network(&self) -> CryptoNetwork {
        CryptoNetwork::Paradigm
    }
    
    async fn is_available(&self) -> bool {
        self.api_client.health_check().await.unwrap_or(false)
    }
    
    async fn get_balance(&self, address: &str) -> Result<CryptoAmount> {
        let balance = self.api_client.get_balance(address).await?;
        Ok(CryptoAmount::from_u64(CryptoNetwork::Paradigm, balance.balance, 8))
    }
    
    async fn send_transaction(&self, from: &str, to: &str, amount: CryptoAmount) -> Result<BridgeTransaction> {
        // Estimate fee first
        let fee = self.api_client.estimate_fee(from, to, amount.raw_amount()).await
            .unwrap_or(10000); // Default fee if estimation fails
        
        let broadcast_request = BroadcastRequest {
            from: from.to_string(),
            to: to.to_string(),
            amount: amount.raw_amount(),
            fee,
            chain_id: CryptoNetwork::Paradigm.chain_id(),
            memo: Some(format!("DefianceNetwork transaction at {}", chrono::Utc::now().timestamp())),
            private_key: "encrypted_private_key".to_string(), // Would be properly encrypted in production
        };
        
        let response = self.api_client.broadcast_transaction(broadcast_request).await?;
        
        if !response.success {
            return Err(anyhow::anyhow!("Transaction failed: {:?}", response.error));
        }
        
        let tx_hash = response.transaction_hash.unwrap_or_else(|| "unknown".to_string());
        
        Ok(BridgeTransaction {
            hash: tx_hash,
            network: CryptoNetwork::Paradigm,
            chain_id: CryptoNetwork::Paradigm.chain_id(),
            from_address: from.to_string(),
            to_address: to.to_string(),
            amount,
            fee: CryptoAmount::from_u64(CryptoNetwork::Paradigm, fee, 8),
            status: TransactionStatus::Pending,
            confirmations: 0,
            block_height: None,
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
    
    async fn get_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus> {
        let transaction = self.api_client.get_transaction(tx_hash).await?;
        
        let status = match transaction.status.as_str() {
            "confirmed" => TransactionStatus::Confirmed,
            "pending" => TransactionStatus::Pending,
            "failed" => TransactionStatus::Failed("Transaction failed".to_string()),
            _ => TransactionStatus::Pending,
        };
        
        Ok(status)
    }
    
    fn validate_address(&self, address: &str) -> bool {
        address.starts_with("PAR") && address.len() == 43
    }
    
    async fn get_network_info(&self) -> Result<NetworkInfo> {
        let stats = self.api_client.get_network_stats().await?;
        
        Ok(NetworkInfo {
            network: CryptoNetwork::Paradigm,
            name: "Paradigm".to_string(),
            symbol: "PAR".to_string(),
            decimals: 8,
            block_height: stats.block_height,
            difficulty: Some(stats.difficulty),
            hash_rate: Some(stats.hash_rate as f64),
            peer_count: Some(stats.peer_count),
            is_synced: true, // Assume synced if we can get stats
        })
    }
    
    async fn generate_address(&self) -> Result<String> {
        self.api_client.generate_address().await
    }
    
    async fn get_transaction_history(&self, address: &str, limit: Option<usize>) -> Result<Vec<BridgeTransaction>> {
        let paradigm_txs = self.api_client.get_transaction_history(address, limit).await?;
        
        let mut bridge_txs = Vec::new();
        for tx in paradigm_txs {
            let status = match tx.status.as_str() {
                "confirmed" => TransactionStatus::Confirmed,
                "pending" => TransactionStatus::Pending,
                "failed" => TransactionStatus::Failed("Transaction failed".to_string()),
                _ => TransactionStatus::Pending,
            };
            
            bridge_txs.push(BridgeTransaction {
                hash: tx.hash,
                network: CryptoNetwork::Paradigm,
                chain_id: tx.chain_id.or(CryptoNetwork::Paradigm.chain_id()),
                from_address: tx.from,
                to_address: tx.to,
                amount: CryptoAmount::from_u64(CryptoNetwork::Paradigm, tx.amount, 8),
                fee: CryptoAmount::from_u64(CryptoNetwork::Paradigm, tx.fee, 8),
                status,
                confirmations: tx.confirmations,
                block_height: tx.block_height,
                timestamp: tx.timestamp,
            });
        }
        
        Ok(bridge_txs)
    }
}