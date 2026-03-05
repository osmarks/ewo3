use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::map::Map;
use crate::worldgen::GeneratedWorld;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameMetrics {
    pub plants_died: u64,
    pub plants_reproduced: u64,
    pub enemies_spawned: u64
}

impl GameMetrics {
    pub fn new() -> Self {
        GameMetrics { plants_died: 0, plants_reproduced: 0, enemies_spawned: 0 }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SavedGame {
    pub ticks: u64,
    pub rng_seed: u64,
    pub map: GeneratedWorld,
    pub dynamic_soil_nutrients: Map<f32>,
    pub dynamic_groundwater: Map<f32>,
    #[serde(with = "serde_bytes")]
    pub world: Vec<u8>,
    pub metrics: GameMetrics
}

impl SavedGame {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::serde::decode_from_slice(bytes, bincode::config::standard())?.0)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(bincode::serde::encode_to_vec(self, bincode::config::standard())?)
    }
}
