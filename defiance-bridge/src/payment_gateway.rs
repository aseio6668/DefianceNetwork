//! Payment gateway integration for DefianceNetwork monetization

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;
use crate::{CryptoNetwork, CryptoAmount, TransactionStatus};
use crate::paradigm_api::ParadigmApiClient;

/// Payment gateway for handling broadcast monetization
pub struct PaymentGateway {
    paradigm_client: Option<ParadigmApiClient>,
    active_payments: Arc<RwLock<HashMap<Uuid, PaymentSession>>>,
    payment_history: Arc<RwLock<Vec<PaymentRecord>>>,
    event_sender: mpsc::UnboundedSender<PaymentEvent>,
    event_receiver: Option<mpsc::UnboundedReceiver<PaymentEvent>>,
}

/// Payment session for broadcasts
#[derive(Debug, Clone)]
pub struct PaymentSession {
    pub session_id: Uuid,
    pub broadcast_id: Uuid,
    pub broadcaster_address: String,
    pub payment_model: PaymentModel,
    pub total_earnings: CryptoAmount,
    pub viewer_payments: HashMap<Uuid, ViewerPayment>,
    pub relay_rewards: HashMap<Uuid, RelayReward>,
    pub created_at: i64,
    pub last_payout: Option<i64>,
    pub status: PaymentSessionStatus,
}

/// Payment models for broadcasts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaymentModel {
    Free,
    PayPerView {
        price: CryptoAmount,
    },
    PayPerMinute {
        price_per_minute: CryptoAmount,
    },
    Subscription {
        monthly_price: CryptoAmount,
        daily_price: Option<CryptoAmount>,
    },
    DonationBased {
        suggested_amount: Option<CryptoAmount>,
    },
    TokenReward {
        reward_per_minute: CryptoAmount,
        max_reward_per_user: Option<CryptoAmount>,
    },
}

/// Viewer payment information
#[derive(Debug, Clone)]
pub struct ViewerPayment {
    pub viewer_id: Uuid,
    pub payment_id: Uuid,
    pub amount: CryptoAmount,
    pub payment_type: ViewerPaymentType,
    pub transaction_hash: Option<String>,
    pub status: TransactionStatus,
    pub timestamp: i64,
    pub duration_minutes: Option<u64>,
}

/// Types of viewer payments
#[derive(Debug, Clone)]
pub enum ViewerPaymentType {
    OneTime,
    PerMinute,
    Subscription,
    Donation,
}

/// Relay rewards for content distribution
#[derive(Debug, Clone)]
pub struct RelayReward {
    pub relay_peer_id: Uuid,
    pub reward_id: Uuid,
    pub amount: CryptoAmount,
    pub duration_minutes: u64,
    pub data_transferred: u64, // bytes
    pub quality_score: f32, // 0.0 to 1.0
    pub transaction_hash: Option<String>,
    pub status: TransactionStatus,
    pub timestamp: i64,
}

/// Payment session status
#[derive(Debug, Clone, PartialEq)]
pub enum PaymentSessionStatus {
    Active,
    Paused,
    Completed,
    Cancelled,
}

/// Payment record for history tracking
#[derive(Debug, Clone)]
pub struct PaymentRecord {
    pub record_id: Uuid,
    pub session_id: Uuid,
    pub payment_type: PaymentRecordType,
    pub amount: CryptoAmount,
    pub from_address: String,
    pub to_address: String,
    pub transaction_hash: String,
    pub status: TransactionStatus,
    pub timestamp: i64,
    pub metadata: HashMap<String, String>,
}

/// Types of payment records
#[derive(Debug, Clone)]
pub enum PaymentRecordType {
    ViewerPayment,
    RelayReward,
    BroadcasterPayout,
    PlatformFee,
}

/// Payment events
#[derive(Debug, Clone)]
pub enum PaymentEvent {
    PaymentReceived {
        session_id: Uuid,
        viewer_id: Uuid,
        amount: CryptoAmount,
    },
    PaymentFailed {
        session_id: Uuid,
        viewer_id: Uuid,
        error: String,
    },
    RewardDistributed {
        session_id: Uuid,
        relay_peer_id: Uuid,
        amount: CryptoAmount,
    },
    PayoutCompleted {
        session_id: Uuid,
        broadcaster_address: String,
        amount: CryptoAmount,
    },
    SessionCompleted {
        session_id: Uuid,
        total_earnings: CryptoAmount,
    },
}

/// Payment statistics
#[derive(Debug, Clone)]
pub struct PaymentStats {
    pub total_payments_received: CryptoAmount,
    pub total_rewards_distributed: CryptoAmount,
    pub total_platform_fees: CryptoAmount,
    pub active_sessions: usize,
    pub successful_payments: u64,
    pub failed_payments: u64,
    pub average_payment_amount: f64,
    pub top_earning_broadcasters: Vec<(String, CryptoAmount)>,
}

impl PaymentGateway {
    /// Create new payment gateway
    pub fn new(paradigm_api_url: Option<String>, paradigm_api_key: Option<String>) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        
        let paradigm_client = if let Some(url) = paradigm_api_url {
            Some(ParadigmApiClient::new(url, paradigm_api_key))
        } else {
            None
        };
        
        Self {
            paradigm_client,
            active_payments: Arc::new(RwLock::new(HashMap::new())),
            payment_history: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            event_receiver: Some(event_receiver),
        }
    }
    
    /// Start a payment session for a broadcast
    pub async fn start_payment_session(
        &self,
        broadcast_id: Uuid,
        broadcaster_address: String,
        payment_model: PaymentModel,
    ) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        
        let session = PaymentSession {
            session_id,
            broadcast_id,
            broadcaster_address,
            payment_model,
            total_earnings: CryptoAmount::new(CryptoNetwork::Paradigm, 0, 8),
            viewer_payments: HashMap::new(),
            relay_rewards: HashMap::new(),
            created_at: chrono::Utc::now().timestamp(),
            last_payout: None,
            status: PaymentSessionStatus::Active,
        };
        
        {
            let mut sessions = self.active_payments.write().await;
            sessions.insert(session_id, session);
        }
        
        tracing::info!("Started payment session {} for broadcast {}", session_id, broadcast_id);
        Ok(session_id)
    }
    
    /// Process viewer payment
    pub async fn process_viewer_payment(
        &self,
        session_id: Uuid,
        viewer_id: Uuid,
        viewer_address: String,
        payment_type: ViewerPaymentType,
        duration_minutes: Option<u64>,
    ) -> Result<Uuid> {
        let payment_id = Uuid::new_v4();
        
        // Calculate payment amount based on session payment model
        let amount = {
            let sessions = self.active_payments.read().await;
            let session = sessions.get(&session_id)
                .ok_or_else(|| anyhow::anyhow!("Payment session not found"))?;
            
            self.calculate_payment_amount(&session.payment_model, &payment_type, duration_minutes)?
        };
        
        // Process payment through Paradigm network
        if let Some(paradigm_client) = &self.paradigm_client {
            let sessions = self.active_payments.read().await;
            let session = sessions.get(&session_id).unwrap();
            
            // Estimate fee and broadcast transaction
            let fee = paradigm_client.estimate_fee(&viewer_address, &session.broadcaster_address, amount.raw_amount())
                .await.unwrap_or(10000);
            
            let broadcast_request = crate::paradigm_api::BroadcastRequest {
                from: viewer_address.clone(),
                to: session.broadcaster_address.clone(),
                amount: amount.raw_amount(),
                fee,
                chain_id: CryptoNetwork::Paradigm.chain_id(),
                memo: Some(format!("DefianceNetwork payment for broadcast {}", session.broadcast_id)),
                private_key: "encrypted_viewer_key".to_string(), // Would be properly managed in production
            };
            
            match paradigm_client.broadcast_transaction(broadcast_request).await {
                Ok(response) => {
                    if response.success {
                        let payment = ViewerPayment {
                            viewer_id,
                            payment_id,
                            amount: amount.clone(),
                            payment_type,
                            transaction_hash: response.transaction_hash.clone(),
                            status: TransactionStatus::Pending,
                            timestamp: chrono::Utc::now().timestamp(),
                            duration_minutes,
                        };
                        
                        // Update session
                        {
                            let mut sessions = self.active_payments.write().await;
                            if let Some(session) = sessions.get_mut(&session_id) {
                                session.viewer_payments.insert(viewer_id, payment);
                                session.total_earnings = session.total_earnings.add(&amount)?;
                            }
                        }
                        
                        // Record payment
                        self.record_payment(PaymentRecord {
                            record_id: Uuid::new_v4(),
                            session_id,
                            payment_type: PaymentRecordType::ViewerPayment,
                            amount: amount.clone(),
                            from_address: viewer_address,
                            to_address: session.broadcaster_address.clone(),
                            transaction_hash: response.transaction_hash.unwrap_or_default(),
                            status: TransactionStatus::Pending,
                            timestamp: chrono::Utc::now().timestamp(),
                            metadata: HashMap::new(),
                        }).await;
                        
                        let _ = self.event_sender.send(PaymentEvent::PaymentReceived {
                            session_id,
                            viewer_id,
                            amount,
                        });
                        
                        tracing::info!("Viewer payment {} processed for session {}", payment_id, session_id);
                    } else {
                        let error = response.error.unwrap_or("Unknown error".to_string());
                        let _ = self.event_sender.send(PaymentEvent::PaymentFailed {
                            session_id,
                            viewer_id,
                            error: error.clone(),
                        });
                        return Err(anyhow::anyhow!("Payment failed: {}", error));
                    }
                }
                Err(e) => {
                    let _ = self.event_sender.send(PaymentEvent::PaymentFailed {
                        session_id,
                        viewer_id,
                        error: e.to_string(),
                    });
                    return Err(e);
                }
            }
        }
        
        Ok(payment_id)
    }
    
    /// Distribute rewards to relay peers
    pub async fn distribute_relay_rewards(
        &self,
        session_id: Uuid,
        relay_peers: Vec<(Uuid, String, u64, u64, f32)>, // peer_id, address, duration_minutes, data_transferred, quality_score
    ) -> Result<()> {
        let base_reward_per_minute = CryptoAmount::new(CryptoNetwork::Paradigm, 1000000, 8); // 0.01 PAR per minute
        
        for (peer_id, address, duration_minutes, data_transferred, quality_score) in relay_peers {
            let reward_amount = self.calculate_relay_reward(
                &base_reward_per_minute,
                duration_minutes,
                data_transferred,
                quality_score,
            )?;
            
            if let Some(paradigm_client) = &self.paradigm_client {
                // Use platform address as sender for rewards
                let platform_address = "PAR_PLATFORM_ADDRESS_HERE".to_string();
                
                let broadcast_request = crate::paradigm_api::BroadcastRequest {
                    from: platform_address.clone(),
                    to: address.clone(),
                    amount: reward_amount.raw_amount(),
                    fee: 10000, // Fixed fee for rewards
                    chain_id: CryptoNetwork::Paradigm.chain_id(),
                    memo: Some(format!("DefianceNetwork relay reward for session {}", session_id)),
                    private_key: "encrypted_platform_key".to_string(),
                };
                
                match paradigm_client.broadcast_transaction(broadcast_request).await {
                    Ok(response) => {
                        if response.success {
                            let reward = RelayReward {
                                relay_peer_id: peer_id,
                                reward_id: Uuid::new_v4(),
                                amount: reward_amount.clone(),
                                duration_minutes,
                                data_transferred,
                                quality_score,
                                transaction_hash: response.transaction_hash.clone(),
                                status: TransactionStatus::Pending,
                                timestamp: chrono::Utc::now().timestamp(),
                            };
                            
                            // Update session
                            {
                                let mut sessions = self.active_payments.write().await;
                                if let Some(session) = sessions.get_mut(&session_id) {
                                    session.relay_rewards.insert(peer_id, reward);
                                }
                            }
                            
                            // Record reward
                            self.record_payment(PaymentRecord {
                                record_id: Uuid::new_v4(),
                                session_id,
                                payment_type: PaymentRecordType::RelayReward,
                                amount: reward_amount.clone(),
                                from_address: platform_address,
                                to_address: address,
                                transaction_hash: response.transaction_hash.unwrap_or_default(),
                                status: TransactionStatus::Pending,
                                timestamp: chrono::Utc::now().timestamp(),
                                metadata: HashMap::new(),
                            }).await;
                            
                            let _ = self.event_sender.send(PaymentEvent::RewardDistributed {
                                session_id,
                                relay_peer_id: peer_id,
                                amount: reward_amount,
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to distribute reward to relay peer {}: {}", peer_id, e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Complete payment session and process final payout
    pub async fn complete_payment_session(&self, session_id: Uuid) -> Result<()> {
        let session = {
            let mut sessions = self.active_payments.write().await;
            sessions.remove(&session_id)
                .ok_or_else(|| anyhow::anyhow!("Payment session not found"))?
        };
        
        // Calculate platform fee (e.g., 0.5% to keep fees minimal)
        let platform_fee = session.total_earnings.percentage(0.5);
        let broadcaster_payout = session.total_earnings.subtract(&platform_fee)?;
        
        // Process final payout to broadcaster
        if !broadcaster_payout.is_zero() {
            if let Some(paradigm_client) = &self.paradigm_client {
            let platform_address = "PAR_PLATFORM_ADDRESS_HERE".to_string();
            
            let broadcast_request = crate::paradigm_api::BroadcastRequest {
                from: platform_address.clone(),
                to: session.broadcaster_address.clone(),
                amount: broadcaster_payout.raw_amount(),
                fee: 10000,
                chain_id: CryptoNetwork::Paradigm.chain_id(),
                memo: Some(format!("DefianceNetwork broadcaster payout for session {}", session_id)),
                private_key: "encrypted_platform_key".to_string(),
            };
            
            match paradigm_client.broadcast_transaction(broadcast_request).await {
                Ok(response) => {
                    if response.success {
                        // Record payout
                        self.record_payment(PaymentRecord {
                            record_id: Uuid::new_v4(),
                            session_id,
                            payment_type: PaymentRecordType::BroadcasterPayout,
                            amount: broadcaster_payout.clone(),
                            from_address: platform_address,
                            to_address: session.broadcaster_address.clone(),
                            transaction_hash: response.transaction_hash.unwrap_or_default(),
                            status: TransactionStatus::Pending,
                            timestamp: chrono::Utc::now().timestamp(),
                            metadata: HashMap::new(),
                        }).await;
                        
                        let _ = self.event_sender.send(PaymentEvent::PayoutCompleted {
                            session_id,
                            broadcaster_address: session.broadcaster_address,
                            amount: broadcaster_payout,
                        });
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to process broadcaster payout: {}", e);
                }
            }
        }
        }
        
        let _ = self.event_sender.send(PaymentEvent::SessionCompleted {
            session_id,
            total_earnings: session.total_earnings,
        });
        
        tracing::info!("Payment session {} completed", session_id);
        Ok(())
    }
    
    /// Calculate payment amount based on model
    fn calculate_payment_amount(
        &self,
        payment_model: &PaymentModel,
        payment_type: &ViewerPaymentType,
        duration_minutes: Option<u64>,
    ) -> Result<CryptoAmount> {
        match payment_model {
            PaymentModel::Free => Ok(CryptoAmount::new(CryptoNetwork::Paradigm, 0, 8)),
            PaymentModel::PayPerView { price } => Ok(price.clone()),
            PaymentModel::PayPerMinute { price_per_minute } => {
                let minutes = duration_minutes.unwrap_or(1);
                Ok(CryptoAmount::from_u64(
                    price_per_minute.network.clone(),
                    price_per_minute.raw_amount() * minutes,
                    price_per_minute.decimals,
                ))
            }
            PaymentModel::Subscription { daily_price, monthly_price } => {
                match payment_type {
                    ViewerPaymentType::Subscription => {
                        if let Some(daily) = daily_price {
                            Ok(daily.clone())
                        } else {
                            Ok(monthly_price.clone())
                        }
                    }
                    _ => Ok(monthly_price.clone()),
                }
            }
            PaymentModel::DonationBased { suggested_amount } => {
                Ok(suggested_amount.clone().unwrap_or_else(|| {
                    CryptoAmount::new(CryptoNetwork::Paradigm, 100000000, 8) // 1 PAR default
                }))
            }
            PaymentModel::TokenReward { reward_per_minute, max_reward_per_user } => {
                let minutes = duration_minutes.unwrap_or(1);
                let total_reward = CryptoAmount::from_u64(
                    reward_per_minute.network.clone(),
                    reward_per_minute.raw_amount() * minutes,
                    reward_per_minute.decimals,
                );
                
                if let Some(max_reward) = max_reward_per_user {
                    if total_reward.greater_than(max_reward)? {
                        Ok(max_reward.clone())
                    } else {
                        Ok(total_reward)
                    }
                } else {
                    Ok(total_reward)
                }
            }
        }
    }
    
    /// Calculate relay reward based on performance
    fn calculate_relay_reward(
        &self,
        base_reward: &CryptoAmount,
        duration_minutes: u64,
        data_transferred: u64,
        quality_score: f32,
    ) -> Result<CryptoAmount> {
        // Base reward for time
        let time_reward = CryptoAmount::from_u64(
            base_reward.network.clone(),
            base_reward.raw_amount() * duration_minutes,
            base_reward.decimals,
        );
        
        // Bonus for data transferred (small amount per MB)
        let data_bonus_per_mb = 100000; // 0.001 PAR per MB
        let data_mb = data_transferred / (1024 * 1024);
        let data_bonus = CryptoAmount::from_u64(
            base_reward.network.clone(),
            data_bonus_per_mb * data_mb,
            base_reward.decimals,
        );
        
        // Quality multiplier (0.5x to 2.0x based on quality score)
        let quality_multiplier = 0.5 + (quality_score * 1.5);
        let total_before_quality = time_reward.add(&data_bonus)?;
        let final_amount = (total_before_quality.raw_amount() as f32 * quality_multiplier) as u64;
        
        Ok(CryptoAmount::from_u64(
            base_reward.network.clone(),
            final_amount,
            base_reward.decimals,
        ))
    }
    
    /// Record payment in history
    async fn record_payment(&self, record: PaymentRecord) {
        let mut history = self.payment_history.write().await;
        history.push(record);
        
        // Keep only last 10,000 records
        if history.len() > 10000 {
            history.remove(0);
        }
    }
    
    /// Get payment statistics
    pub async fn get_payment_stats(&self) -> PaymentStats {
        let history = self.payment_history.read().await;
        let sessions = self.active_payments.read().await;
        
        let mut total_payments = CryptoAmount::new(CryptoNetwork::Paradigm, 0, 8);
        let mut total_rewards = CryptoAmount::new(CryptoNetwork::Paradigm, 0, 8);
        let mut total_fees = CryptoAmount::new(CryptoNetwork::Paradigm, 0, 8);
        let mut successful_payments = 0u64;
        let mut failed_payments = 0u64;
        let mut broadcaster_earnings: HashMap<String, u64> = HashMap::new();
        
        for record in history.iter() {
            match record.payment_type {
                PaymentRecordType::ViewerPayment => {
                    if record.status == TransactionStatus::Confirmed {
                        total_payments = total_payments.add(&record.amount).unwrap_or(total_payments);
                        successful_payments += 1;
                        *broadcaster_earnings.entry(record.to_address.clone()).or_insert(0) += record.amount.raw_amount();
                    } else if matches!(record.status, TransactionStatus::Failed(_)) {
                        failed_payments += 1;
                    }
                }
                PaymentRecordType::RelayReward => {
                    if record.status == TransactionStatus::Confirmed {
                        total_rewards = total_rewards.add(&record.amount).unwrap_or(total_rewards);
                    }
                }
                PaymentRecordType::PlatformFee => {
                    if record.status == TransactionStatus::Confirmed {
                        total_fees = total_fees.add(&record.amount).unwrap_or(total_fees);
                    }
                }
                _ => {}
            }
        }
        
        let average_payment = if successful_payments > 0 {
            total_payments.to_decimal() / successful_payments as f64
        } else {
            0.0
        };
        
        // Get top earners
        let mut top_earners: Vec<(String, CryptoAmount)> = broadcaster_earnings
            .into_iter()
            .map(|(addr, amount)| (addr, CryptoAmount::from_u64(CryptoNetwork::Paradigm, amount, 8)))
            .collect();
        top_earners.sort_by(|a, b| b.1.raw_amount().cmp(&a.1.raw_amount()));
        top_earners.truncate(10);
        
        PaymentStats {
            total_payments_received: total_payments,
            total_rewards_distributed: total_rewards,
            total_platform_fees: total_fees,
            active_sessions: sessions.len(),
            successful_payments,
            failed_payments,
            average_payment_amount: average_payment,
            top_earning_broadcasters: top_earners,
        }
    }
    
    /// Get payment session by ID
    pub async fn get_payment_session(&self, session_id: Uuid) -> Option<PaymentSession> {
        let sessions = self.active_payments.read().await;
        sessions.get(&session_id).cloned()
    }
    
    /// Take the event receiver for processing payment events
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<PaymentEvent>> {
        self.event_receiver.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_payment_gateway_creation() {
        let gateway = PaymentGateway::new(None, None);
        assert!(gateway.paradigm_client.is_none());
        
        let sessions = gateway.active_payments.read().await;
        assert!(sessions.is_empty());
    }
    
    #[tokio::test]
    async fn test_payment_session_creation() {
        let gateway = PaymentGateway::new(None, None);
        
        let session_id = gateway.start_payment_session(
            Uuid::new_v4(),
            "PAR1234567890abcdef1234567890abcdef12345678".to_string(),
            PaymentModel::PayPerView {
                price: CryptoAmount::new(CryptoNetwork::Paradigm, 500000000, 8), // 5 PAR
            },
        ).await.unwrap();
        
        let session = gateway.get_payment_session(session_id).await;
        assert!(session.is_some());
        assert_eq!(session.unwrap().status, PaymentSessionStatus::Active);
    }
    
    #[test]
    fn test_payment_amount_calculation() {
        let gateway = PaymentGateway::new(None, None);
        
        // Pay per view
        let ppv_model = PaymentModel::PayPerView {
            price: CryptoAmount::new(CryptoNetwork::Paradigm, 500000000, 8), // 5 PAR
        };
        let amount = gateway.calculate_payment_amount(&ppv_model, &ViewerPaymentType::OneTime, None).unwrap();
        assert_eq!(amount.raw_amount(), 500000000);
        
        // Pay per minute
        let ppm_model = PaymentModel::PayPerMinute {
            price_per_minute: CryptoAmount::new(CryptoNetwork::Paradigm, 10000000, 8), // 0.1 PAR per minute
        };
        let amount = gateway.calculate_payment_amount(&ppm_model, &ViewerPaymentType::PerMinute, Some(30)).unwrap();
        assert_eq!(amount.raw_amount(), 300000000); // 30 minutes * 0.1 PAR
    }
    
    #[test]
    fn test_relay_reward_calculation() {
        let gateway = PaymentGateway::new(None, None);
        let base_reward = CryptoAmount::new(CryptoNetwork::Paradigm, 1000000, 8); // 0.01 PAR per minute
        
        let reward = gateway.calculate_relay_reward(
            &base_reward,
            60, // 1 hour
            1024 * 1024 * 100, // 100 MB
            0.9, // High quality score
        ).unwrap();
        
        // Should be base (60 * 0.01) + data bonus (100 * 0.001) + quality multiplier (0.9 -> 1.85x)
        // = 0.6 + 0.1 = 0.7 PAR * 1.85 = 1.295 PAR
        assert!(reward.to_decimal() > 1.0 && reward.to_decimal() < 1.5);
    }
}