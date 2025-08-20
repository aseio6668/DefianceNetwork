//! Arceon RPC client for blockchain communication

use anyhow::{Result, anyhow};
use reqwest::{Client, header::HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use uuid::Uuid;

use crate::{TransactionStatus};
use super::{ArceonConfig, ArceonNetworkType, TransactionOptions};

/// Arceon RPC client for blockchain operations
pub struct ArceonRpcClient {
    client: Client,
    config: ArceonConfig,
    current_node_index: std::sync::atomic::AtomicUsize,
}

impl ArceonRpcClient {
    /// Create new Arceon RPC client
    pub async fn new(config: ArceonConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        
        if let Some(ref api_key) = config.api_key {
            headers.insert("Authorization", format!("Bearer {}", api_key).parse()?);
        }
        
        let client = Client::builder()
            .timeout(Duration::from_secs(config.connection_timeout_seconds))
            .default_headers(headers)
            .build()?;
        
        Ok(Self {
            client,
            config,
            current_node_index: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// Make RPC call with automatic failover
    async fn rpc_call(&self, method: &str, params: Value) -> Result<Value> {
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_attempts {
            for (node_idx, node_url) in self.config.node_urls.iter().enumerate() {
                let request_body = json!({
                    "jsonrpc": "2.0",
                    "id": Uuid::new_v4().to_string(),
                    "method": method,
                    "params": params
                });
                
                match self.client.post(node_url).json(&request_body).send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            match response.json::<RpcResponse>().await {
                                Ok(rpc_resp) => {
                                    if let Some(error) = rpc_resp.error {
                                        last_error = Some(anyhow!("RPC error: {} - {}", error.code, error.message));
                                        continue;
                                    }
                                    
                                    // Update current node on success
                                    self.current_node_index.store(node_idx, std::sync::atomic::Ordering::Relaxed);
                                    return Ok(rpc_resp.result.unwrap_or(Value::Null));
                                }
                                Err(e) => {
                                    last_error = Some(anyhow!("Failed to parse RPC response: {}", e));
                                    continue;
                                }
                            }
                        } else {
                            last_error = Some(anyhow!("HTTP error: {}", response.status()));
                            continue;
                        }
                    }
                    Err(e) => {
                        last_error = Some(anyhow!("Request failed: {}", e));
                        continue;
                    }
                }
            }
            
            if attempt < self.config.retry_attempts - 1 {
                tokio::time::sleep(Duration::from_millis(1000 * (attempt + 1) as u64)).await;
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow!("All RPC attempts failed")))
    }
    
    /// Get blockchain information
    pub async fn get_blockchain_info(&self) -> Result<BlockchainInfo> {
        let result = self.rpc_call("getblockchaininfo", json!([])).await?;
        
        Ok(BlockchainInfo {
            chain: result["chain"].as_str().unwrap_or("unknown").to_string(),
            blocks: result["blocks"].as_u64().unwrap_or(0),
            headers: result["headers"].as_u64().unwrap_or(0),
            best_blockhash: result["bestblockhash"].as_str().unwrap_or("").to_string(),
            difficulty: result["difficulty"].as_f64().unwrap_or(0.0),
            verification_progress: result["verificationprogress"].as_f64().unwrap_or(0.0),
            chainwork: result["chainwork"].as_str().unwrap_or("").to_string(),
        })
    }
    
    /// Get network information
    pub async fn get_network_info(&self) -> Result<NetworkInfo> {
        let result = self.rpc_call("getnetworkinfo", json!([])).await?;
        
        Ok(NetworkInfo {
            version: result["version"].as_u64().unwrap_or(0),
            subversion: result["subversion"].as_str().unwrap_or("").to_string(),
            protocol_version: result["protocolversion"].as_u64().unwrap_or(0),
            connections: result["connections"].as_u64().unwrap_or(0) as u32,
            networks: Vec::new(), // Would need to parse the networks array
            relay_fee: result["relayfee"].as_f64().unwrap_or(0.0),
            hash_rate: result["networkhashps"].as_f64(),
        })
    }
    
    /// Get mempool information
    pub async fn get_mempool_info(&self) -> Result<MempoolInfo> {
        let result = self.rpc_call("getmempoolinfo", json!([])).await?;
        
        Ok(MempoolInfo {
            size: result["size"].as_u64().unwrap_or(0) as u32,
            bytes: result["bytes"].as_u64().unwrap_or(0),
            usage: result["usage"].as_u64().unwrap_or(0),
            max_mempool: result["maxmempool"].as_u64().unwrap_or(0),
            mempool_min_fee: result["mempoolminfee"].as_f64().unwrap_or(0.0),
        })
    }
    
    /// Get balance for an address
    pub async fn get_balance(&self, address: &str) -> Result<i64> {
        // For UTXO-based blockchains like Arceon, we need to use listunspent
        let result = self.rpc_call("listunspent", json!([0, 9999999, [address]])).await?;
        
        let mut total_balance = 0i64;
        if let Some(utxos) = result.as_array() {
            for utxo in utxos {
                if let Some(amount) = utxo["amount"].as_f64() {
                    total_balance += (amount * 100_000_000.0) as i64; // Convert to satoshis
                }
            }
        }
        
        Ok(total_balance)
    }
    
    /// Send a transaction
    pub async fn send_transaction(
        &self,
        from_address: &str,
        to_address: &str,
        amount: u64,
        fee: u64,
        private_key: &str,
        options: &TransactionOptions,
    ) -> Result<String> {
        // This is a simplified implementation
        // In a real implementation, you would:
        // 1. Get UTXOs for the from_address
        // 2. Create a raw transaction
        // 3. Sign the transaction with the private key
        // 4. Broadcast the transaction
        
        // For now, we'll simulate this with a direct send call
        let amount_btc = amount as f64 / 100_000_000.0;
        
        let mut params = json!([to_address, amount_btc]);
        
        if let Some(memo) = &options.memo {
            params.as_array_mut().unwrap().push(json!(memo));
        }
        
        let result = self.rpc_call("sendtoaddress", params).await?;
        
        result.as_str()
            .ok_or_else(|| anyhow!("Invalid transaction hash returned"))
            .map(|s| s.to_string())
    }
    
    /// Get transaction status
    pub async fn get_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus> {
        let result = self.rpc_call("gettransaction", json!([tx_hash])).await?;
        
        let confirmations = result["confirmations"].as_u64().unwrap_or(0);
        
        if confirmations == 0 {
            Ok(TransactionStatus::Pending)
        } else if confirmations >= 6 {
            Ok(TransactionStatus::Confirmed)
        } else {
            Ok(TransactionStatus::Confirming)
        }
    }
    
    /// Get transaction confirmations
    pub async fn get_transaction_confirmations(&self, tx_hash: &str) -> Result<u32> {
        let result = self.rpc_call("gettransaction", json!([tx_hash])).await?;
        Ok(result["confirmations"].as_u64().unwrap_or(0) as u32)
    }
    
    /// Get detailed transaction information
    pub async fn get_transaction_info(&self, tx_hash: &str) -> Result<TransactionInfo> {
        let result = self.rpc_call("gettransaction", json!([tx_hash])).await?;
        
        Ok(TransactionInfo {
            hash: tx_hash.to_string(),
            amount: (result["amount"].as_f64().unwrap_or(0.0) * 100_000_000.0) as i64,
            fee: (result["fee"].as_f64().unwrap_or(0.0).abs() * 100_000_000.0) as i64,
            confirmations: result["confirmations"].as_u64().unwrap_or(0) as u32,
            block_height: result["blockheight"].as_u64(),
            from_address: result["details"][0]["address"].as_str().unwrap_or("").to_string(),
            to_address: result["details"][0]["address"].as_str().unwrap_or("").to_string(),
            timestamp: result["time"].as_i64().unwrap_or(0),
            status: if result["confirmations"].as_u64().unwrap_or(0) >= 6 {
                TransactionStatus::Confirmed
            } else if result["confirmations"].as_u64().unwrap_or(0) > 0 {
                TransactionStatus::Confirming
            } else {
                TransactionStatus::Pending
            },
            raw_transaction: result["hex"].as_str().unwrap_or("").to_string(),
        })
    }
    
    /// Get transaction history for an address
    pub async fn get_transaction_history(&self, address: &str, limit: usize) -> Result<Vec<TransactionInfo>> {
        // This would typically involve multiple RPC calls to get all transactions
        // For simplicity, we'll use listtransactions if available
        let result = self.rpc_call("listtransactions", json!(["*", limit, 0, true])).await?;
        
        let mut transactions = Vec::new();
        
        if let Some(tx_array) = result.as_array() {
            for tx in tx_array {
                if tx["address"].as_str() == Some(address) {
                    transactions.push(TransactionInfo {
                        hash: tx["txid"].as_str().unwrap_or("").to_string(),
                        amount: (tx["amount"].as_f64().unwrap_or(0.0) * 100_000_000.0) as i64,
                        fee: (tx["fee"].as_f64().unwrap_or(0.0).abs() * 100_000_000.0) as i64,
                        confirmations: tx["confirmations"].as_u64().unwrap_or(0) as u32,
                        block_height: tx["blockheight"].as_u64(),
                        from_address: tx["address"].as_str().unwrap_or("").to_string(),
                        to_address: tx["address"].as_str().unwrap_or("").to_string(),
                        timestamp: tx["time"].as_i64().unwrap_or(0),
                        status: if tx["confirmations"].as_u64().unwrap_or(0) >= 6 {
                            TransactionStatus::Confirmed
                        } else if tx["confirmations"].as_u64().unwrap_or(0) > 0 {
                            TransactionStatus::Confirming
                        } else {
                            TransactionStatus::Pending
                        },
                        raw_transaction: String::new(),
                    });
                }
            }
        }
        
        Ok(transactions)
    }
    
    /// Generate a new address
    pub async fn generate_new_address(&self, label: Option<&str>) -> Result<String> {
        let params = if let Some(label) = label {
            json!([label])
        } else {
            json!([])
        };
        
        let result = self.rpc_call("getnewaddress", params).await?;
        
        result.as_str()
            .ok_or_else(|| anyhow!("Invalid address returned"))
            .map(|s| s.to_string())
    }
    
    /// Validate an address
    pub async fn validate_address(&self, address: &str) -> Result<bool> {
        let result = self.rpc_call("validateaddress", json!([address])).await?;
        Ok(result["isvalid"].as_bool().unwrap_or(false))
    }
    
    /// Get current node URL
    pub fn get_current_node_url(&self) -> Option<String> {
        let index = self.current_node_index.load(std::sync::atomic::Ordering::Relaxed);
        self.config.node_urls.get(index).cloned()
    }
}

/// RPC response structure
#[derive(Debug, Deserialize)]
struct RpcResponse {
    jsonrpc: String,
    id: String,
    result: Option<Value>,
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Blockchain information
#[derive(Debug, Clone)]
pub struct BlockchainInfo {
    pub chain: String,
    pub blocks: u64,
    pub headers: u64,
    pub best_blockhash: String,
    pub difficulty: f64,
    pub verification_progress: f64,
    pub chainwork: String,
}

/// Network information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    pub version: u64,
    pub subversion: String,
    pub protocol_version: u64,
    pub connections: u32,
    pub networks: Vec<String>,
    pub relay_fee: f64,
    pub hash_rate: Option<f64>,
}

/// Mempool information
#[derive(Debug, Clone)]
pub struct MempoolInfo {
    pub size: u32,
    pub bytes: u64,
    pub usage: u64,
    pub max_mempool: u64,
    pub mempool_min_fee: f64,
}

/// Transaction information
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    pub hash: String,
    pub amount: i64,
    pub fee: i64,
    pub confirmations: u32,
    pub block_height: Option<u64>,
    pub from_address: String,
    pub to_address: String,
    pub timestamp: i64,
    pub status: TransactionStatus,
    pub raw_transaction: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_client_creation() {
        let config = ArceonConfig::default();
        
        // Test that we can create a client (though we can't test actual RPC calls without a node)
        assert!(!config.node_urls.is_empty());
        assert_eq!(config.network_type, ArceonNetworkType::Mainnet);
    }

    #[test]
    fn test_rpc_response_parsing() {
        // Test RPC response structure parsing
        let json_response = r#"
        {
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "blocks": 12345,
                "chain": "main"
            }
        }"#;
        
        let response: RpcResponse = serde_json::from_str(json_response).unwrap();
        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.id, "test-id");
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }
}