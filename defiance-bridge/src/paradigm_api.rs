//! Paradigm API client for blockchain operations

use std::collections::HashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Paradigm API client for blockchain communication
pub struct ParadigmApiClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

/// Paradigm blockchain transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmTransaction {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub amount: u64, // In satoshi-like units
    pub fee: u64,
    pub timestamp: i64,
    pub confirmations: u32,
    pub block_height: Option<u64>,
    pub status: String,
    pub memo: Option<String>,
}

/// Paradigm account balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadigmBalance {
    pub address: String,
    pub balance: u64,
    pub locked_balance: u64,
    pub pending_balance: u64,
    pub last_updated: i64,
}

/// Paradigm address validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressValidation {
    pub address: String,
    pub is_valid: bool,
    pub address_type: Option<String>,
    pub network: String,
}

/// Transaction broadcast request
#[derive(Debug, Clone, Serialize)]
pub struct BroadcastRequest {
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub fee: u64,
    pub memo: Option<String>,
    pub private_key: String, // Would be encrypted in production
}

/// Transaction broadcast response
#[derive(Debug, Clone, Deserialize)]
pub struct BroadcastResponse {
    pub success: bool,
    pub transaction_hash: Option<String>,
    pub error: Option<String>,
    pub estimated_confirmation_time: Option<u64>,
}

/// Network statistics from Paradigm blockchain
#[derive(Debug, Clone, Deserialize)]
pub struct NetworkStats {
    pub block_height: u64,
    pub difficulty: f64,
    pub hash_rate: u64,
    pub total_supply: u64,
    pub circulating_supply: u64,
    pub peer_count: u32,
    pub mempool_size: u32,
    pub average_block_time: f64,
}

/// Staking information for Paradigm network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingInfo {
    pub address: String,
    pub staked_amount: u64,
    pub rewards_earned: u64,
    pub lock_period: u64,
    pub stake_start_time: i64,
    pub auto_compound: bool,
    pub validator: Option<String>,
}

impl ParadigmApiClient {
    /// Create new Paradigm API client
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
            
        Self {
            client,
            base_url,
            api_key,
        }
    }
    
    /// Add authentication headers if API key is present
    fn add_auth_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        request.header("Content-Type", "application/json")
    }
    
    /// Get account balance for a Paradigm address
    pub async fn get_balance(&self, address: &str) -> Result<ParadigmBalance> {
        let url = format!("{}/api/v1/balance/{}", self.base_url, address);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("API request failed: {}", response.status()));
        }
        
        let balance: ParadigmBalance = response.json().await?;
        Ok(balance)
    }
    
    /// Send a transaction to the Paradigm network
    pub async fn broadcast_transaction(&self, request: BroadcastRequest) -> Result<BroadcastResponse> {
        let url = format!("{}/api/v1/broadcast", self.base_url);
        let http_request = self.client.post(&url);
        let http_request = self.add_auth_headers(http_request);
        
        let response = http_request.json(&request).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Broadcast failed: {}", response.status()));
        }
        
        let broadcast_response: BroadcastResponse = response.json().await?;
        Ok(broadcast_response)
    }
    
    /// Get transaction details by hash
    pub async fn get_transaction(&self, tx_hash: &str) -> Result<ParadigmTransaction> {
        let url = format!("{}/api/v1/transaction/{}", self.base_url, tx_hash);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Transaction not found: {}", response.status()));
        }
        
        let transaction: ParadigmTransaction = response.json().await?;
        Ok(transaction)
    }
    
    /// Get transaction history for an address
    pub async fn get_transaction_history(&self, address: &str, limit: Option<usize>) -> Result<Vec<ParadigmTransaction>> {
        let mut url = format!("{}/api/v1/history/{}", self.base_url, address);
        if let Some(limit) = limit {
            url.push_str(&format!("?limit={}", limit));
        }
        
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get history: {}", response.status()));
        }
        
        let transactions: Vec<ParadigmTransaction> = response.json().await?;
        Ok(transactions)
    }
    
    /// Validate a Paradigm address
    pub async fn validate_address(&self, address: &str) -> Result<AddressValidation> {
        let url = format!("{}/api/v1/validate/{}", self.base_url, address);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Validation failed: {}", response.status()));
        }
        
        let validation: AddressValidation = response.json().await?;
        Ok(validation)
    }
    
    /// Generate a new Paradigm address
    pub async fn generate_address(&self) -> Result<String> {
        let url = format!("{}/api/v1/address/generate", self.base_url);
        let request = self.client.post(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Address generation failed: {}", response.status()));
        }
        
        #[derive(Deserialize)]
        struct AddressResponse {
            address: String,
        }
        
        let addr_response: AddressResponse = response.json().await?;
        Ok(addr_response.address)
    }
    
    /// Get current network statistics
    pub async fn get_network_stats(&self) -> Result<NetworkStats> {
        let url = format!("{}/api/v1/network/stats", self.base_url);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Stats request failed: {}", response.status()));
        }
        
        let stats: NetworkStats = response.json().await?;
        Ok(stats)
    }
    
    /// Get staking information for an address
    pub async fn get_staking_info(&self, address: &str) -> Result<StakingInfo> {
        let url = format!("{}/api/v1/staking/{}", self.base_url, address);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Staking info failed: {}", response.status()));
        }
        
        let staking: StakingInfo = response.json().await?;
        Ok(staking)
    }
    
    /// Create a stake transaction
    pub async fn create_stake(&self, address: &str, amount: u64, validator: Option<String>) -> Result<BroadcastResponse> {
        #[derive(Serialize)]
        struct StakeRequest {
            address: String,
            amount: u64,
            validator: Option<String>,
        }
        
        let stake_request = StakeRequest {
            address: address.to_string(),
            amount,
            validator,
        };
        
        let url = format!("{}/api/v1/staking/stake", self.base_url);
        let request = self.client.post(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.json(&stake_request).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Stake creation failed: {}", response.status()));
        }
        
        let stake_response: BroadcastResponse = response.json().await?;
        Ok(stake_response)
    }
    
    /// Unstake tokens
    pub async fn unstake(&self, address: &str, amount: u64) -> Result<BroadcastResponse> {
        #[derive(Serialize)]
        struct UnstakeRequest {
            address: String,
            amount: u64,
        }
        
        let unstake_request = UnstakeRequest {
            address: address.to_string(),
            amount,
        };
        
        let url = format!("{}/api/v1/staking/unstake", self.base_url);
        let request = self.client.post(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.json(&unstake_request).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Unstake failed: {}", response.status()));
        }
        
        let unstake_response: BroadcastResponse = response.json().await?;
        Ok(unstake_response)
    }
    
    /// Estimate transaction fee
    pub async fn estimate_fee(&self, from: &str, to: &str, amount: u64) -> Result<u64> {
        #[derive(Serialize)]
        struct FeeRequest {
            from: String,
            to: String,
            amount: u64,
        }
        
        let fee_request = FeeRequest {
            from: from.to_string(),
            to: to.to_string(),
            amount,
        };
        
        let url = format!("{}/api/v1/fee/estimate", self.base_url);
        let request = self.client.post(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.json(&fee_request).send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Fee estimation failed: {}", response.status()));
        }
        
        #[derive(Deserialize)]
        struct FeeResponse {
            estimated_fee: u64,
        }
        
        let fee_response: FeeResponse = response.json().await?;
        Ok(fee_response.estimated_fee)
    }
    
    /// Get current gas price for Paradigm network
    pub async fn get_gas_price(&self) -> Result<u64> {
        let url = format!("{}/api/v1/gas/price", self.base_url);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Gas price request failed: {}", response.status()));
        }
        
        #[derive(Deserialize)]
        struct GasResponse {
            gas_price: u64,
        }
        
        let gas_response: GasResponse = response.json().await?;
        Ok(gas_response.gas_price)
    }
    
    /// Check if the Paradigm node is healthy and synchronized
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/v1/health", self.base_url);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        Ok(response.status().is_success())
    }
    
    /// Subscribe to real-time transaction notifications (WebSocket)
    pub async fn subscribe_to_address(&self, _address: &str) -> Result<()> {
        // TODO: Implement WebSocket subscription for real-time updates
        tracing::info!("WebSocket subscription not yet implemented");
        Ok(())
    }
    
    /// Get mempool information
    pub async fn get_mempool_info(&self) -> Result<HashMap<String, serde_json::Value>> {
        let url = format!("{}/api/v1/mempool", self.base_url);
        let request = self.client.get(&url);
        let request = self.add_auth_headers(request);
        
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Mempool request failed: {}", response.status()));
        }
        
        let mempool: HashMap<String, serde_json::Value> = response.json().await?;
        Ok(mempool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_paradigm_client_creation() {
        let client = ParadigmApiClient::new(
            "https://api.paradigm.network".to_string(),
            Some("test_api_key".to_string())
        );
        
        assert_eq!(client.base_url, "https://api.paradigm.network");
        assert!(client.api_key.is_some());
    }
    
    #[tokio::test]
    async fn test_health_check_mock() {
        // This would require a mock server for proper testing
        // For now, just test that the method exists and can be called
        let client = ParadigmApiClient::new(
            "https://mock.paradigm.network".to_string(),
            None
        );
        
        // This will fail but shows the method signature is correct
        let result = client.health_check().await;
        assert!(result.is_err()); // Expected since mock server doesn't exist
    }
    
    #[test]
    fn test_transaction_serialization() {
        let tx = ParadigmTransaction {
            hash: "test_hash".to_string(),
            from: "PAR1234".to_string(),
            to: "PAR5678".to_string(),
            amount: 1000000,
            fee: 10000,
            timestamp: 1640995200,
            confirmations: 6,
            block_height: Some(12345),
            status: "confirmed".to_string(),
            memo: Some("Test transaction".to_string()),
        };
        
        let serialized = serde_json::to_string(&tx);
        assert!(serialized.is_ok());
        
        let deserialized: Result<ParadigmTransaction, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
}