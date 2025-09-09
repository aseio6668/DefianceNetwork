//! # Defiance Bridge
//! 
//! Cryptocurrency bridge for connecting multiple blockchain networks including
//! Paradigm, Arceon, and other cryptocurrencies to the DefianceNetwork ecosystem.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;

pub mod paradigm;
pub mod paradigm_api;
pub mod bridge_core;
pub mod payment;
pub mod payment_gateway;
pub mod staking;

#[path = "arceon/mod.rs"]
pub mod arceon;

// Re-export commonly used types
pub use bridge_core::{BridgeManager, BridgeNetwork, BridgeTransaction};
pub use payment::{PaymentProcessor, PaymentIntent, PaymentStatus};
pub use payment_gateway::{PaymentGateway, PaymentModel, PaymentEvent};
pub use staking::{StakingManager, StakingPool, StakingReward};

/// Supported cryptocurrency networks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CryptoNetwork {
    Paradigm,
    Arceon,
    Bitcoin,
    Ethereum,
    Litecoin,
    Monero,
    Custom(String),
}

impl CryptoNetwork {
    /// Get the symbol for this network
    pub fn symbol(&self) -> &str {
        match self {
            CryptoNetwork::Paradigm => "PAR",
            CryptoNetwork::Arceon => "ARC",
            CryptoNetwork::Bitcoin => "BTC",
            CryptoNetwork::Ethereum => "ETH",
            CryptoNetwork::Litecoin => "LTC",
            CryptoNetwork::Monero => "XMR",
            CryptoNetwork::Custom(name) => name,
        }
    }

    /// Get the chain ID for this network (EIP-155 compliance)
    pub fn chain_id(&self) -> Option<u64> {
        match self {
            CryptoNetwork::Paradigm => Some(9080), // Custom chain ID for Paradigm
            CryptoNetwork::Arceon => Some(9081),   // Custom chain ID for Arceon
            CryptoNetwork::Bitcoin => None,        // Bitcoin doesn't use chain IDs
            CryptoNetwork::Ethereum => Some(1),    // Ethereum mainnet
            CryptoNetwork::Litecoin => None,       // Litecoin doesn't use chain IDs
            CryptoNetwork::Monero => None,         // Monero doesn't use chain IDs
            CryptoNetwork::Custom(_) => Some(9999), // Default for custom networks
        }
    }
}

/// Cryptocurrency amounts with precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoAmount {
    pub network: CryptoNetwork,
    pub amount: u64, // Amount in smallest unit (satoshis, wei, etc.)
    pub decimals: u8,
}

/// Wallet address for different networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletAddress {
    pub network: CryptoNetwork,
    pub address: String,
    pub is_valid: bool,
}

/// Transaction between networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainTransaction {
    pub id: Uuid,
    pub from_network: CryptoNetwork,
    pub to_network: CryptoNetwork,
    pub from_address: String,
    pub to_address: String,
    pub amount: CryptoAmount,
    pub fee: CryptoAmount,
    pub status: TransactionStatus,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub transaction_hash: Option<String>,
    pub bridge_hash: Option<String>,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Confirming,
    Confirmed,
    Bridging,
    Completed,
    Failed(String),
    Cancelled,
}

/// Exchange rate between currencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRate {
    pub from_network: CryptoNetwork,
    pub to_network: CryptoNetwork,
    pub rate: f64,
    pub updated_at: i64,
    pub source: String,
}

/// Bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub supported_networks: Vec<CryptoNetwork>,
    pub paradigm_node_url: Option<String>,
    pub arceon_node_url: Option<String>,
    pub min_bridge_amount: HashMap<CryptoNetwork, u64>,
    pub max_bridge_amount: HashMap<CryptoNetwork, u64>,
    pub bridge_fee_percentage: f64,
    pub confirmation_blocks: HashMap<CryptoNetwork, u32>,
    pub enable_staking: bool,
    pub enable_payments: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        let mut min_amounts = HashMap::new();
        min_amounts.insert(CryptoNetwork::Paradigm, 100_00000000); // 100 PAR
        min_amounts.insert(CryptoNetwork::Arceon, 50_00000000);   // 50 ARC

        let mut max_amounts = HashMap::new();
        max_amounts.insert(CryptoNetwork::Paradigm, 1000000_00000000); // 1M PAR
        max_amounts.insert(CryptoNetwork::Arceon, 500000_00000000);   // 500K ARC

        let mut confirmations = HashMap::new();
        confirmations.insert(CryptoNetwork::Paradigm, 3);
        confirmations.insert(CryptoNetwork::Arceon, 6);
        confirmations.insert(CryptoNetwork::Bitcoin, 6);
        confirmations.insert(CryptoNetwork::Ethereum, 12);

        Self {
            supported_networks: vec![
                CryptoNetwork::Paradigm,
                CryptoNetwork::Arceon,
                CryptoNetwork::Bitcoin,
                CryptoNetwork::Ethereum,
            ],
            paradigm_node_url: Some("http://localhost:8080".to_string()),
            arceon_node_url: Some("http://localhost:8081".to_string()),
            min_bridge_amount: min_amounts,
            max_bridge_amount: max_amounts,
            bridge_fee_percentage: 0.01, // 0.01% fee (~$0.01 on $100 transaction)
            confirmation_blocks: confirmations,
            enable_staking: true,
            enable_payments: true,
        }
    }
}

impl CryptoAmount {
    /// Create new crypto amount
    pub fn new(network: CryptoNetwork, amount: i64, decimals: u8) -> Self {
        Self {
            network,
            amount: amount.max(0) as u64, // Ensure non-negative
            decimals,
        }
    }
    
    /// Create new crypto amount from u64
    pub fn from_u64(network: CryptoNetwork, amount: u64, decimals: u8) -> Self {
        Self {
            network,
            amount,
            decimals,
        }
    }
    
    /// Get amount in smallest unit (compatible with Arceon modules)
    pub fn amount_smallest_unit(&self) -> i64 {
        self.amount as i64
    }
    
    /// Get the network for this amount
    pub fn network(&self) -> CryptoNetwork {
        self.network.clone()
    }
    
    /// Get decimals for this amount
    pub fn decimals(&self) -> u8 {
        self.decimals
    }

    /// Get raw amount for API calls
    pub fn raw_amount(&self) -> u64 {
        self.amount
    }

    /// Create from decimal amount
    pub fn from_decimal(network: CryptoNetwork, amount: f64, decimals: u8) -> Self {
        let multiplier = 10u64.pow(decimals as u32);
        let amount_units = (amount * multiplier as f64) as u64;
        
        Self {
            network,
            amount: amount_units,
            decimals,
        }
    }

    /// Convert to decimal representation
    pub fn to_decimal(&self) -> f64 {
        let divisor = 10u64.pow(self.decimals as u32) as f64;
        self.amount as f64 / divisor
    }

    /// Format as string with symbol
    pub fn format(&self) -> String {
        let symbol = match self.network {
            CryptoNetwork::Paradigm => "PAR",
            CryptoNetwork::Arceon => "ARC",
            CryptoNetwork::Bitcoin => "BTC",
            CryptoNetwork::Ethereum => "ETH",
            CryptoNetwork::Litecoin => "LTC",
            CryptoNetwork::Monero => "XMR",
            CryptoNetwork::Custom(ref name) => name,
        };

        format!("{:.8} {}", self.to_decimal(), symbol)
    }

    /// Add another amount (must be same network)
    pub fn add(&self, other: &CryptoAmount) -> Result<CryptoAmount> {
        if self.network != other.network {
            return Err(anyhow::anyhow!("Cannot add amounts from different networks"));
        }

        Ok(CryptoAmount {
            network: self.network.clone(),
            amount: self.amount + other.amount,
            decimals: self.decimals,
        })
    }

    /// Subtract another amount (must be same network)
    pub fn subtract(&self, other: &CryptoAmount) -> Result<CryptoAmount> {
        if self.network != other.network {
            return Err(anyhow::anyhow!("Cannot subtract amounts from different networks"));
        }

        if self.amount < other.amount {
            return Err(anyhow::anyhow!("Insufficient balance"));
        }

        Ok(CryptoAmount {
            network: self.network.clone(),
            amount: self.amount - other.amount,
            decimals: self.decimals,
        })
    }

    /// Calculate percentage of amount
    pub fn percentage(&self, percentage: f64) -> CryptoAmount {
        let fee_amount = (self.amount as f64 * percentage / 100.0) as u64;
        
        CryptoAmount {
            network: self.network.clone(),
            amount: fee_amount,
            decimals: self.decimals,
        }
    }

    /// Check if amount is zero
    pub fn is_zero(&self) -> bool {
        self.amount == 0
    }

    /// Check if amount is greater than other
    pub fn greater_than(&self, other: &CryptoAmount) -> Result<bool> {
        if self.network != other.network {
            return Err(anyhow::anyhow!("Cannot compare amounts from different networks"));
        }
        Ok(self.amount > other.amount)
    }
}

impl WalletAddress {
    /// Create new wallet address
    pub fn new(network: CryptoNetwork, address: String) -> Self {
        let is_valid = Self::validate_address(&network, &address);
        
        Self {
            network,
            address,
            is_valid,
        }
    }

    /// Validate address format for network
    pub fn validate_address(network: &CryptoNetwork, address: &str) -> bool {
        match network {
            CryptoNetwork::Paradigm => {
                // Paradigm addresses start with "PAR" followed by 40 hex chars
                address.starts_with("PAR") && address.len() == 43
            }
            CryptoNetwork::Arceon => {
                // Arceon addresses start with "ARC" followed by 40 hex chars
                address.starts_with("ARC") && address.len() == 43
            }
            CryptoNetwork::Bitcoin => {
                // Simplified Bitcoin address validation
                (address.starts_with('1') || address.starts_with('3') || address.starts_with("bc1")) 
                    && address.len() >= 26 && address.len() <= 62
            }
            CryptoNetwork::Ethereum => {
                // Ethereum addresses are 42 chars starting with 0x
                address.starts_with("0x") && address.len() == 42
            }
            CryptoNetwork::Litecoin => {
                // Litecoin addresses start with L, M, or ltc1
                (address.starts_with('L') || address.starts_with('M') || address.starts_with("ltc1"))
                    && address.len() >= 26 && address.len() <= 62
            }
            CryptoNetwork::Monero => {
                // Monero addresses are 95 chars starting with 4
                address.starts_with('4') && address.len() == 95
            }
            CryptoNetwork::Custom(_) => {
                // Custom networks - basic length check
                address.len() >= 20 && address.len() <= 100
            }
        }
    }

    /// Get network-specific address format info
    pub fn get_format_info(&self) -> String {
        match self.network {
            CryptoNetwork::Paradigm => "PAR + 40 hex characters".to_string(),
            CryptoNetwork::Arceon => "ARC + 40 hex characters".to_string(),
            CryptoNetwork::Bitcoin => "Legacy (1...), Script (3...), or Bech32 (bc1...)".to_string(),
            CryptoNetwork::Ethereum => "0x + 40 hex characters".to_string(),
            CryptoNetwork::Litecoin => "Legacy (L..., M...) or Bech32 (ltc1...)".to_string(),
            CryptoNetwork::Monero => "95 characters starting with 4".to_string(),
            CryptoNetwork::Custom(ref name) => format!("Custom format for {}", name),
        }
    }
}

impl CrossChainTransaction {
    /// Create new cross-chain transaction
    pub fn new(
        from_network: CryptoNetwork,
        to_network: CryptoNetwork,
        from_address: String,
        to_address: String,
        amount: CryptoAmount,
        fee: CryptoAmount,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_network,
            to_network,
            from_address,
            to_address,
            amount,
            fee,
            status: TransactionStatus::Pending,
            created_at: chrono::Utc::now().timestamp(),
            completed_at: None,
            transaction_hash: None,
            bridge_hash: None,
        }
    }

    /// Mark transaction as confirmed
    pub fn confirm(&mut self, transaction_hash: String) {
        self.status = TransactionStatus::Confirmed;
        self.transaction_hash = Some(transaction_hash);
    }

    /// Mark transaction as bridging
    pub fn start_bridging(&mut self, bridge_hash: String) {
        self.status = TransactionStatus::Bridging;
        self.bridge_hash = Some(bridge_hash);
    }

    /// Complete the transaction
    pub fn complete(&mut self) {
        self.status = TransactionStatus::Completed;
        self.completed_at = Some(chrono::Utc::now().timestamp());
    }

    /// Fail the transaction
    pub fn fail(&mut self, reason: String) {
        self.status = TransactionStatus::Failed(reason);
    }

    /// Check if transaction is final (completed or failed)
    pub fn is_final(&self) -> bool {
        matches!(
            self.status,
            TransactionStatus::Completed | TransactionStatus::Failed(_) | TransactionStatus::Cancelled
        )
    }

    /// Get total cost (amount + fee)
    pub fn total_cost(&self) -> Result<CryptoAmount> {
        self.amount.add(&self.fee)
    }

    /// Get transaction age in seconds
    pub fn age_seconds(&self) -> i64 {
        chrono::Utc::now().timestamp() - self.created_at
    }
}

impl ExchangeRate {
    /// Create new exchange rate
    pub fn new(
        from_network: CryptoNetwork,
        to_network: CryptoNetwork,
        rate: f64,
        source: String,
    ) -> Self {
        Self {
            from_network,
            to_network,
            rate,
            updated_at: chrono::Utc::now().timestamp(),
            source,
        }
    }

    /// Convert amount using this exchange rate
    pub fn convert(&self, amount: &CryptoAmount) -> Result<CryptoAmount> {
        if amount.network != self.from_network {
            return Err(anyhow::anyhow!("Amount network doesn't match exchange rate from_network"));
        }

        let converted_decimal = amount.to_decimal() * self.rate;
        let to_decimals = match self.to_network {
            CryptoNetwork::Paradigm => 8,
            CryptoNetwork::Arceon => 8,
            CryptoNetwork::Bitcoin => 8,
            CryptoNetwork::Ethereum => 18,
            CryptoNetwork::Litecoin => 8,
            CryptoNetwork::Monero => 12,
            CryptoNetwork::Custom(_) => 8,
        };

        Ok(CryptoAmount::from_decimal(
            self.to_network.clone(),
            converted_decimal,
            to_decimals,
        ))
    }

    /// Check if rate is fresh (less than 1 hour old)
    pub fn is_fresh(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        (now - self.updated_at) < 3600 // 1 hour
    }

    /// Get rate age in minutes
    pub fn age_minutes(&self) -> i64 {
        let now = chrono::Utc::now().timestamp();
        (now - self.updated_at) / 60
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crypto_amount_creation() {
        let amount = CryptoAmount::new(CryptoNetwork::Paradigm, 100_00000000, 8);
        assert_eq!(amount.to_decimal(), 100.0);
        assert_eq!(amount.format(), "100.00000000 PAR");
    }

    #[test]
    fn test_crypto_amount_arithmetic() {
        let amount1 = CryptoAmount::new(CryptoNetwork::Paradigm, 100_00000000, 8);
        let amount2 = CryptoAmount::new(CryptoNetwork::Paradigm, 50_00000000, 8);

        let sum = amount1.add(&amount2).unwrap();
        assert_eq!(sum.amount, 150_00000000);

        let diff = amount1.subtract(&amount2).unwrap();
        assert_eq!(diff.amount, 50_00000000);

        let fee = amount1.percentage(0.01);
        assert_eq!(fee.amount, 1000000); // 0.01% of 100 PAR
    }

    #[test]
    fn test_wallet_address_validation() {
        assert!(WalletAddress::validate_address(
            &CryptoNetwork::Paradigm,
            "PAR1234567890abcdef1234567890abcdef12345678"
        ));
        
        assert!(WalletAddress::validate_address(
            &CryptoNetwork::Ethereum,
            "0x1234567890abcdef1234567890abcdef12345678"
        ));
        
        assert!(!WalletAddress::validate_address(
            &CryptoNetwork::Bitcoin,
            "invalid_address"
        ));
    }

    #[test]
    fn test_cross_chain_transaction() {
        let amount = CryptoAmount::new(CryptoNetwork::Paradigm, 100_00000000, 8);
        let fee = CryptoAmount::new(CryptoNetwork::Paradigm, 1_00000000, 8);

        let mut tx = CrossChainTransaction::new(
            CryptoNetwork::Paradigm,
            CryptoNetwork::Arceon,
            "PAR1234567890abcdef1234567890abcdef12345678".to_string(),
            "ARC1234567890abcdef1234567890abcdef12345678".to_string(),
            amount,
            fee,
        );

        assert_eq!(tx.status, TransactionStatus::Pending);
        assert!(!tx.is_final());

        tx.confirm("tx_hash_123".to_string());
        assert_eq!(tx.status, TransactionStatus::Confirmed);

        tx.complete();
        assert_eq!(tx.status, TransactionStatus::Completed);
        assert!(tx.is_final());
    }

    #[test]
    fn test_exchange_rate() {
        let rate = ExchangeRate::new(
            CryptoNetwork::Paradigm,
            CryptoNetwork::Arceon,
            2.0,
            "test_source".to_string(),
        );

        let par_amount = CryptoAmount::new(CryptoNetwork::Paradigm, 100_00000000, 8);
        let arc_amount = rate.convert(&par_amount).unwrap();

        assert_eq!(arc_amount.network, CryptoNetwork::Arceon);
        assert_eq!(arc_amount.to_decimal(), 200.0);
    }

    #[test]
    fn test_network_equality() {
        assert_eq!(CryptoNetwork::Paradigm, CryptoNetwork::Paradigm);
        assert_ne!(CryptoNetwork::Paradigm, CryptoNetwork::Arceon);
        
        let custom1 = CryptoNetwork::Custom("test".to_string());
        let custom2 = CryptoNetwork::Custom("test".to_string());
        assert_eq!(custom1, custom2);
    }

    #[test]
    fn test_chain_id() {
        assert_eq!(CryptoNetwork::Paradigm.chain_id(), Some(9080));
        assert_eq!(CryptoNetwork::Arceon.chain_id(), Some(9081));
        assert_eq!(CryptoNetwork::Ethereum.chain_id(), Some(1));
        assert_eq!(CryptoNetwork::Bitcoin.chain_id(), None);
        assert_eq!(CryptoNetwork::Custom("test".to_string()).chain_id(), Some(9999));
    }
}