//! Arceon blockchain types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{CryptoAmount, CryptoNetwork, TransactionStatus};

/// Arceon block representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonBlock {
    pub hash: String,
    pub height: u64,
    pub previous_hash: String,
    pub timestamp: i64,
    pub merkle_root: String,
    pub transactions: Vec<ArceonTransaction>,
    pub difficulty: f64,
    pub nonce: u64,
    pub size: u64,
    pub weight: u64,
    pub version: u32,
    pub confirmations: u32,
}

/// Arceon transaction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonTransaction {
    pub hash: String,
    pub version: u32,
    pub chain_id: Option<u64>, // Chain ID for multi-network support
    pub inputs: Vec<ArceonInput>,
    pub outputs: Vec<ArceonOutput>,
    pub lock_time: u32,
    pub block_hash: Option<String>,
    pub block_height: Option<u64>,
    pub confirmations: u32,
    pub timestamp: i64,
    pub fee: i64,
    pub size: u32,
    pub weight: u32,
    pub status: TransactionStatus,
}

/// Arceon transaction input (UTXO reference)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonInput {
    pub txid: String,
    pub vout: u32,
    pub script_sig: ArceonScript,
    pub sequence: u32,
    pub witness: Option<Vec<String>>,
    pub previous_output: Option<ArceonOutput>,
}

/// Arceon transaction output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonOutput {
    pub value: i64, // Amount in smallest unit (satoshis)
    pub script_pubkey: ArceonScript,
    pub spent: bool,
    pub spent_by: Option<String>, // Transaction hash that spent this output
}

/// Arceon script representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonScript {
    pub hex: String,
    pub asm: String,
    pub script_type: ArceonScriptType,
    pub addresses: Vec<String>,
    pub required_signatures: Option<u32>,
}

/// Types of scripts in Arceon
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArceonScriptType {
    P2PKH,        // Pay to Public Key Hash
    P2SH,         // Pay to Script Hash
    P2WPKH,       // Pay to Witness Public Key Hash
    P2WSH,        // Pay to Witness Script Hash
    P2TR,         // Pay to Taproot
    Multisig,     // Multi-signature
    OpReturn,     // Data storage
    NonStandard,  // Non-standard script
}

/// Arceon address representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonAddress {
    pub address: String,
    pub script_type: ArceonScriptType,
    pub network: ArceonNetworkType,
    pub is_valid: bool,
    pub is_witness: bool,
}

/// Arceon network types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArceonNetworkType {
    Mainnet,
    Testnet,
    Devnet,
}

/// UTXO (Unspent Transaction Output) for Arceon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonUtxo {
    pub txid: String,
    pub vout: u32,
    pub address: String,
    pub script_pubkey: String,
    pub amount: i64,
    pub confirmations: u32,
    pub spendable: bool,
    pub solvable: bool,
    pub safe: bool,
    pub label: Option<String>,
}

/// Arceon mempool entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonMempoolEntry {
    pub txid: String,
    pub size: u32,
    pub weight: u32,
    pub fee: i64,
    pub modified_fee: i64,
    pub time: i64,
    pub height: u64,
    pub descendant_count: u32,
    pub descendant_size: u32,
    pub descendant_fees: i64,
    pub ancestor_count: u32,
    pub ancestor_size: u32,
    pub ancestor_fees: i64,
    pub wtxid: String,
    pub fees: ArceonFeeDetails,
    pub depends: Vec<String>,
    pub spent_by: Vec<String>,
    pub bip125_replaceable: bool,
}

/// Fee details for transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonFeeDetails {
    pub base: i64,
    pub modified: i64,
    pub ancestor: i64,
    pub descendant: i64,
}

/// Arceon peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonPeer {
    pub id: u32,
    pub addr: String,
    pub addr_local: Option<String>,
    pub services: String,
    pub relaytxes: bool,
    pub last_send: i64,
    pub last_recv: i64,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    pub connection_time: i64,
    pub time_offset: i64,
    pub ping_time: Option<f64>,
    pub min_ping: Option<f64>,
    pub version: u32,
    pub subver: String,
    pub inbound: bool,
    pub addnode: bool,
    pub start_height: i64,
    pub ban_score: u32,
    pub synced_headers: i64,
    pub synced_blocks: i64,
    pub inflight: Vec<u64>,
    pub whitelisted: bool,
}

/// Arceon wallet transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonWalletTransaction {
    pub account: String,
    pub address: String,
    pub category: String, // "send", "receive", "generate", etc.
    pub amount: f64,
    pub label: Option<String>,
    pub vout: u32,
    pub fee: Option<f64>,
    pub confirmations: i32,
    pub generated: Option<bool>,
    pub blockhash: Option<String>,
    pub blockindex: Option<u32>,
    pub blocktime: Option<i64>,
    pub txid: String,
    pub walletconflicts: Vec<String>,
    pub time: i64,
    pub timereceived: i64,
    pub bip125_replaceable: String,
    pub comment: Option<String>,
    pub to: Option<String>,
}

/// Arceon smart contract representation (if supported)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonContract {
    pub address: String,
    pub code: String,
    pub storage: HashMap<String, String>,
    pub balance: i64,
    pub nonce: u64,
    pub created_at: i64,
    pub creator: String,
    pub contract_type: ArceonContractType,
}

/// Types of smart contracts (if Arceon supports them)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArceonContractType {
    Payment,
    Escrow,
    MultiSig,
    TimeLock,
    Custom,
}

/// Arceon mining information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonMiningInfo {
    pub blocks: u64,
    pub difficulty: f64,
    pub networkhashps: f64,
    pub pooledtx: u32,
    pub chain: String,
    pub warnings: String,
}

/// Arceon block template for mining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonBlockTemplate {
    pub version: u32,
    pub rules: Vec<String>,
    pub vb_available: HashMap<String, u32>,
    pub vb_required: u32,
    pub previous_blockhash: String,
    pub transactions: Vec<ArceonTemplateTransaction>,
    pub coinbase_aux: HashMap<String, String>,
    pub coinbase_value: i64,
    pub longpollid: String,
    pub target: String,
    pub min_time: i64,
    pub mutable: Vec<String>,
    pub nonce_range: String,
    pub sigoplimit: u32,
    pub sizelimit: u32,
    pub weightlimit: u32,
    pub cur_time: i64,
    pub bits: String,
    pub height: u64,
}

/// Transaction in block template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonTemplateTransaction {
    pub data: String,
    pub txid: String,
    pub hash: String,
    pub depends: Vec<u32>,
    pub fee: i64,
    pub sigops: u32,
    pub weight: u32,
}

/// Validation state for addresses and transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArceonValidationResult {
    pub is_valid: bool,
    pub address: Option<String>,
    pub script_pubkey: Option<String>,
    pub is_mine: Option<bool>,
    pub is_watch_only: Option<bool>,
    pub is_script: Option<bool>,
    pub is_witness: Option<bool>,
    pub witness_version: Option<u32>,
    pub witness_program: Option<String>,
    pub pubkey: Option<String>,
    pub is_compressed: Option<bool>,
    pub account: Option<String>,
    pub hd_key_path: Option<String>,
    pub hd_seed_id: Option<String>,
}

impl ArceonTransaction {
    /// Convert to bridge transaction format
    pub fn to_bridge_transaction(&self) -> crate::bridge_core::BridgeTransaction {
        let total_output_value: i64 = self.outputs.iter().map(|o| o.value).sum();
        
        crate::bridge_core::BridgeTransaction {
            hash: self.hash.clone(),
            network: CryptoNetwork::Arceon,
            chain_id: self.chain_id.or(CryptoNetwork::Arceon.chain_id()),
            from_address: self.get_input_addresses().join(","),
            to_address: self.get_output_addresses().join(","),
            amount: CryptoAmount::new(CryptoNetwork::Arceon, total_output_value, 8),
            fee: CryptoAmount::new(CryptoNetwork::Arceon, self.fee, 8),
            status: self.status.clone(),
            confirmations: self.confirmations,
            block_height: self.block_height,
            timestamp: self.timestamp,
        }
    }
    
    /// Get all input addresses
    pub fn get_input_addresses(&self) -> Vec<String> {
        self.inputs.iter()
            .filter_map(|input| input.previous_output.as_ref())
            .flat_map(|output| output.script_pubkey.addresses.clone())
            .collect()
    }
    
    /// Get all output addresses
    pub fn get_output_addresses(&self) -> Vec<String> {
        self.outputs.iter()
            .flat_map(|output| output.script_pubkey.addresses.clone())
            .collect()
    }
    
    /// Calculate transaction fee rate (satoshis per byte)
    pub fn fee_rate(&self) -> f64 {
        if self.size == 0 {
            0.0
        } else {
            self.fee as f64 / self.size as f64
        }
    }
    
    /// Check if transaction is confirmed
    pub fn is_confirmed(&self) -> bool {
        self.confirmations >= 6
    }
    
    /// Check if transaction is in mempool
    pub fn is_in_mempool(&self) -> bool {
        self.block_hash.is_none() && self.confirmations == 0
    }
}

impl ArceonBlock {
    /// Calculate block reward (simplified)
    pub fn calculate_reward(&self) -> i64 {
        // Simplified block reward calculation
        // In reality, this would depend on block height and halving schedule
        if self.height < 210000 {
            5000000000 // 50 ARC
        } else if self.height < 420000 {
            2500000000 // 25 ARC
        } else if self.height < 630000 {
            1250000000 // 12.5 ARC
        } else {
            625000000  // 6.25 ARC
        }
    }
    
    /// Get total transaction fees in block
    pub fn total_fees(&self) -> i64 {
        self.transactions.iter().map(|tx| tx.fee).sum()
    }
    
    /// Check if block is confirmed
    pub fn is_confirmed(&self) -> bool {
        self.confirmations >= 6
    }
}

impl ArceonAddress {
    /// Create new address from string
    pub fn from_string(address: &str, network: ArceonNetworkType) -> Self {
        let is_valid = Self::validate_format(address);
        let script_type = Self::detect_script_type(address);
        let is_witness = Self::is_witness_address(address);
        
        Self {
            address: address.to_string(),
            script_type,
            network,
            is_valid,
            is_witness,
        }
    }
    
    /// Validate address format
    fn validate_format(address: &str) -> bool {
        // Basic validation for Arceon addresses
        if address.starts_with("ARC") && address.len() == 43 {
            address[3..].chars().all(|c| c.is_ascii_hexdigit())
        } else {
            false
        }
    }
    
    /// Detect script type from address
    fn detect_script_type(address: &str) -> ArceonScriptType {
        // Simplified script type detection based on address format
        if address.starts_with("ARC") {
            ArceonScriptType::P2PKH
        } else {
            ArceonScriptType::NonStandard
        }
    }
    
    /// Check if address is witness (SegWit)
    fn is_witness_address(_address: &str) -> bool {
        // Simplified - in reality would check for SegWit address formats
        false
    }
}

impl ArceonUtxo {
    /// Convert to CryptoAmount
    pub fn to_crypto_amount(&self) -> CryptoAmount {
        CryptoAmount::new(CryptoNetwork::Arceon, self.amount, 8)
    }
    
    /// Check if UTXO is mature (can be spent)
    pub fn is_mature(&self, required_confirmations: u32) -> bool {
        self.confirmations >= required_confirmations && self.spendable && self.safe
    }
    
    /// Get UTXO identifier
    pub fn get_outpoint(&self) -> String {
        format!("{}:{}", self.txid, self.vout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arceon_address_validation() {
        let valid_address = "ARC1234567890abcdef1234567890abcdef123456";
        let invalid_address = "BTC1234567890abcdef1234567890abcdef123456";
        
        let addr1 = ArceonAddress::from_string(valid_address, ArceonNetworkType::Mainnet);
        assert!(addr1.is_valid);
        assert_eq!(addr1.script_type, ArceonScriptType::P2PKH);
        
        let addr2 = ArceonAddress::from_string(invalid_address, ArceonNetworkType::Mainnet);
        assert!(!addr2.is_valid);
    }

    #[test]
    fn test_arceon_transaction_fee_rate() {
        let tx = ArceonTransaction {
            hash: "test_hash".to_string(),
            version: 1,
            chain_id: Some(9081),
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
            block_hash: None,
            block_height: None,
            confirmations: 0,
            timestamp: 0,
            fee: 1000,
            size: 250,
            weight: 1000,
            status: TransactionStatus::Pending,
        };
        
        assert_eq!(tx.fee_rate(), 4.0); // 1000 / 250 = 4 sat/byte
    }

    #[test]
    fn test_arceon_block_reward() {
        let block = ArceonBlock {
            hash: "test_hash".to_string(),
            height: 100000,
            previous_hash: "prev_hash".to_string(),
            timestamp: 0,
            merkle_root: "merkle".to_string(),
            transactions: vec![],
            difficulty: 1.0,
            nonce: 0,
            size: 1000,
            weight: 4000,
            version: 1,
            confirmations: 10,
        };
        
        assert_eq!(block.calculate_reward(), 5000000000); // 50 ARC for early block
    }

    #[test]
    fn test_utxo_maturity() {
        let utxo = ArceonUtxo {
            txid: "test_txid".to_string(),
            vout: 0,
            address: "ARC1234567890abcdef1234567890abcdef123456".to_string(),
            script_pubkey: "script".to_string(),
            amount: 100000000,
            confirmations: 10,
            spendable: true,
            solvable: true,
            safe: true,
            label: None,
        };
        
        assert!(utxo.is_mature(6));
        assert!(!utxo.is_mature(20));
    }

    #[test]
    fn test_network_types() {
        let mainnet = ArceonNetworkType::Mainnet;
        let testnet = ArceonNetworkType::Testnet;
        let devnet = ArceonNetworkType::Devnet;
        
        assert_eq!(mainnet, ArceonNetworkType::Mainnet);
        assert_ne!(mainnet, testnet);
        assert_ne!(testnet, devnet);
    }
}