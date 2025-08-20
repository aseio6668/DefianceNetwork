//! Staking system for earning rewards

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::{CryptoNetwork, CryptoAmount};

/// Staking manager for crypto rewards
pub struct StakingManager {
    pools: HashMap<CryptoNetwork, StakingPool>,
    user_stakes: HashMap<Uuid, Vec<UserStake>>,
}

/// Staking pool for a specific cryptocurrency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingPool {
    pub network: CryptoNetwork,
    pub total_staked: CryptoAmount,
    pub reward_rate: f64, // Annual percentage rate
    pub min_stake: CryptoAmount,
    pub lock_period: u64, // Seconds
    pub total_rewards_distributed: CryptoAmount,
}

/// User staking position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStake {
    pub id: Uuid,
    pub user_id: Uuid,
    pub network: CryptoNetwork,
    pub amount: CryptoAmount,
    pub staked_at: i64,
    pub unlock_at: i64,
    pub rewards_earned: CryptoAmount,
    pub is_active: bool,
}

/// Staking reward information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingReward {
    pub user_id: Uuid,
    pub network: CryptoNetwork,
    pub amount: CryptoAmount,
    pub earned_at: i64,
    pub stake_id: Uuid,
}

impl StakingManager {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            user_stakes: HashMap::new(),
        }
    }
    
    pub fn add_staking_pool(&mut self, pool: StakingPool) {
        self.pools.insert(pool.network.clone(), pool);
    }
    
    pub fn get_staking_pool(&self, network: &CryptoNetwork) -> Option<&StakingPool> {
        self.pools.get(network)
    }
}