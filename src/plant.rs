use serde::{Deserialize, Serialize};
use crate::worldgen::TerrainType;
use crate::util::*;
use crate::util::config::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CropType {
    Grass,
    EucalyptusTree,
    BushTomato,
    GoldenWattleTree
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Genome {
    pub crop_type: CropType,
    // polygenic traits; parameterized as N(0,1) (allegedly)
    // groundwater is [0,1] so this is sort of questionable
    // TODO: reparameterize or something
    nutrient_addition_rate: f32,
    optimal_water_level: f32, // no longer absolute level; sigmoided
    optimal_temperature: f32,
    temperature_tolerance: f32,
    water_tolerance: f32,
    salt_tolerance: f32,
    mature_size: f32,
    lifespan_multiplier: f32,
    reproduction_rate: f32,
    initial_size_scale: f32
    // TODO number of seeds produced?
    // TODO color trait
}

impl Genome {
    pub fn terrain_valid(&self, terrain: &TerrainType) -> bool {
        match terrain {
            TerrainType::Empty => true,
            _ => false
        }
    }

    pub fn base_growth_rate(&self, nutrients: f32, water: f32, temperature: f32, salt: f32, terrain: &TerrainType) -> f32 {
        if !self.terrain_valid(terrain) {
            return 0.0;
        }
        let mut water_diff = (water - sigmoid(self.optimal_water_level)).powf(2.0);
        if water_diff >= 0.0 {
            water_diff *= 0.5; // ugly hack for asymmetry
        }
        let temperature_diff = (temperature - sigmoid(self.optimal_temperature)).powf(2.0);
        let base = 1.0
            - self.nutrient_addition_rate() * 0.04 // nutrient enrichment has a growth tradeoff
            - self.water_tolerance * 0.11
            - self.temperature_tolerance * 0.09
            - self.salt_tolerance.max(0.0) * 0.01;
        let base = base.max(0.0);

        let water_tolerance_coefficient = 13.0 * (1.0 + (-self.water_tolerance).exp());
        let temperature_tolerance_coefficient = 3.0 * (1.0 + (-self.temperature_tolerance).exp());
        let salt_tolerance_coefficient = 8.0 * (-self.salt_tolerance).exp();

        let raw = base
            * (2.0 * nutrients - 1.5).min(0.0).exp()
            * (-water_tolerance_coefficient * water_diff).exp()
            * (-temperature_tolerance_coefficient * temperature_diff).exp()
            * (-salt.abs() * salt_tolerance_coefficient).exp();

        raw.min(1.0)
    }

    pub fn mature_size(&self) -> f32 {
        self.mature_size.exp()
    }

    // This is not directly used in computations, as plants mature when they have gained enough size rather than at a specifc time.
    // However, old age death is directly time-based.
    pub fn age_at_maturity(&self) -> f32 {
        // allometric scaling law
        self.mature_size().powf(0.25)
    }

    pub fn lifespan_multiplier(&self) -> f32 {
        self.lifespan_multiplier.exp() + 1.0
    }

    pub fn lifespan(&self) -> f32 {
        self.age_at_maturity() * self.lifespan_multiplier()
    }

    pub fn nutrient_addition_rate(&self) -> f32 {
        self.nutrient_addition_rate.max(0.0)
    }

    pub fn initial_size_scale(&self) -> f32 {
        self.initial_size_scale.exp()
    }

    pub fn reproduction_rate(&self) -> f32 {
        sigmoid(self.reproduction_rate)
    }

    pub fn random(rng: &mut fastrand::Rng) -> Genome {
        let crop_type = match rng.usize(0..4) {
            0 => CropType::Grass,
            1 => CropType::EucalyptusTree,
            2 => CropType::BushTomato,
            3 => CropType::GoldenWattleTree,
            _ => unreachable!()
        };

        // Mature sizes aren't fully realistic for performance reasons: modelling individual blades of grass at EWO3 speeds is unfortunately not currently feasible. The sizes may represent an aggregation of several plants.
        // Size is something like total mass of a 1m^2 collection of this.
        // TODO pick these more precisely.
        let (nutrient_addition_rate, optimal_water_level, optimal_temperature, salt_tolerance, lifespan_multiplier, mature_size, reproduction_rate) = match crop_type {
            CropType::Grass =>            (-10.0,-1.0,-0.5, 0.0, 0.5,  0.0,  0.0), // TODO: tie reproduction rate to something else?
            CropType::EucalyptusTree =>   (-10.0, 1.0, 0.0, 0.6, 5.0,  6.0, -4.0),
            CropType::BushTomato =>       (-10.0, 0.0, 1.0, 1.2, 1.5,  1.0, -3.0),
            CropType::GoldenWattleTree => (  2.0, 0.5, 0.2, 4.0, 3.0,  4.0, -4.0),

        };

        Genome {
            crop_type: crop_type,
            nutrient_addition_rate,
            optimal_water_level,
            optimal_temperature,
            temperature_tolerance: normal(rng),
            water_tolerance: normal(rng),
            salt_tolerance: normal(rng) + salt_tolerance,
            mature_size,
            lifespan_multiplier,
            reproduction_rate,
            initial_size_scale: mature_size * 0.5 // TODO
        }
    }

    // TODO: do this better
    pub fn water_efficiency(&self) -> f32 {
        sigmoid(self.optimal_water_level)
    }

    pub fn hybridize(&self, rng: &mut fastrand::Rng, other: &Genome) -> Option<Genome> {
        if self.crop_type != other.crop_type { return None }
        Some(Genome {
            crop_type: self.crop_type,
            nutrient_addition_rate: (self.nutrient_addition_rate + other.nutrient_addition_rate) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.03),
            optimal_water_level: (self.optimal_water_level + other.optimal_water_level) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.03),
            optimal_temperature: (self.optimal_temperature + other.optimal_temperature) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.03),
            temperature_tolerance: (self.temperature_tolerance + other.temperature_tolerance) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.1),
            water_tolerance: (self.water_tolerance + other.water_tolerance) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.02),
            salt_tolerance: (self.salt_tolerance + other.salt_tolerance) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.02),
            mature_size: (self.mature_size + other.mature_size) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.1),
            lifespan_multiplier: (self.lifespan_multiplier + other.lifespan_multiplier) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.1),
            reproduction_rate: (self.reproduction_rate + other.reproduction_rate) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.05),
            initial_size_scale: (self.initial_size_scale + other.initial_size_scale) / 2.0 + EVOLUTION_RATE * normal_scaled(rng, 0.0, 0.1),
        })
    }
}
