//! Payment processing system

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use crate::{CryptoNetwork, CryptoAmount};

/// Payment processor for handling content payments
pub struct PaymentProcessor {
    supported_currencies: Vec<CryptoNetwork>,
    payment_intents: HashMap<Uuid, PaymentIntent>,
}

/// Payment intent for content access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentIntent {
    pub id: Uuid,
    pub content_id: Uuid,
    pub user_id: Uuid,
    pub amount: CryptoAmount,
    pub currency: CryptoNetwork,
    pub status: PaymentStatus,
    pub created_at: i64,
    pub expires_at: i64,
    pub payment_address: Option<String>,
    pub transaction_hash: Option<String>,
}

/// Payment status tracking
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentStatus {
    Created,
    AwaitingPayment,
    Processing,
    Confirmed,
    Completed,
    Failed(String),
    Expired,
    Refunded,
}

impl PaymentProcessor {
    pub fn new(supported_currencies: Vec<CryptoNetwork>) -> Self {
        Self {
            supported_currencies,
            payment_intents: HashMap::new(),
        }
    }
    
    pub async fn create_payment_intent(
        &mut self,
        content_id: Uuid,
        user_id: Uuid,
        amount: CryptoAmount,
    ) -> Result<Uuid> {
        let currency = amount.network.clone();
        let payment_intent = PaymentIntent {
            id: Uuid::new_v4(),
            content_id,
            user_id,
            amount,
            currency,
            status: PaymentStatus::Created,
            created_at: chrono::Utc::now().timestamp(),
            expires_at: chrono::Utc::now().timestamp() + 3600, // 1 hour
            payment_address: None,
            transaction_hash: None,
        };
        
        let id = payment_intent.id;
        self.payment_intents.insert(id, payment_intent);
        
        Ok(id)
    }
    
    pub fn get_payment_intent(&self, id: Uuid) -> Option<&PaymentIntent> {
        self.payment_intents.get(&id)
    }
}