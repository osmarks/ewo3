use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::map::Map;
use crate::worldgen::GeneratedWorld;

#[derive(Serialize, Deserialize)]
pub struct SavedGame {
    pub ticks: u64,
    pub rng_seed: u64,
    pub map: GeneratedWorld,
    pub dynamic_soil_nutrients: Map<f32>,
    pub dynamic_groundwater: Map<f32>,
    pub world: Vec<u8>,
}

impl SavedGame {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::serde::decode_from_slice(bytes, bincode::config::standard())?.0)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(bincode::serde::encode_to_vec(self, bincode::config::standard())?)
    }
}
