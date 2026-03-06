use serde::{Deserialize, Serialize};
use crate::worldgen::TerrainType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CropType {
    Grass,
    EucalyptusTree,
    BushTomato,
    GoldenWattleTree
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Genome {
    crop_type: CropType,
    // polygenic traits; parameterized as N(0,1) (allegedly)
    // groundwater is [0,1] so this is sort of questionable
    // TODO: reparameterize or something
    growth_rate: f32,
    nutrient_addition_rate: f32,
    optimal_water_level: f32,
    optimal_temperature: f32,
    reproduction_rate: f32,
    reproductive_size_fraction_param: f32,
    temperature_tolerance: f32,
    water_tolerance: f32,
    salt_tolerance: f32,
    max_size: f32
    // TODO number of seeds produced?
    // TODO color trait
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Box-Muller transform
fn normal() -> f32 {
    let u = fastrand::f32();
    let v = fastrand::f32();
    (v * std::f32::consts::TAU).cos() * (-2.0 * u.ln()).sqrt()
}

fn normal_scaled(mu: f32, sigma: f32) -> f32 {
    normal() * sigma + mu
}

impl Genome {
    pub fn reproductive_size_fraction(&self) -> f32 {
        sigmoid(self.reproductive_size_fraction_param) * 0.49 + 0.5
    }

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
        let water_diff = (water - sigmoid(self.optimal_water_level)).powf(2.0);
        let temperature_diff = (temperature - sigmoid(self.optimal_temperature)).powf(2.0);
        let base = 1.0
            - self.reproduction_rate * 0.1 // faster reproduction trades off slightly against growth
            - self.nutrient_addition_rate() * 0.16 // nutrient enrichment has a growth tradeoff
            - self.water_tolerance * 0.08
            - self.temperature_tolerance * 0.07
            - self.salt_tolerance.max(0.0) * 0.15;

        let water_tolerance_coefficient = 15.0 * (1.0 + (-self.water_tolerance).exp());
        let temperature_tolerance_coefficient = 12.0 * (1.0 + (-self.temperature_tolerance).exp());
        let salt_tolerance_coefficient = 8.0 * (-self.salt_tolerance).exp();

        base
            * (2.0 * nutrients - 1.5).min(0.0).exp()
            * (-water_tolerance_coefficient * water_diff).exp()
            * (-temperature_tolerance_coefficient * temperature_diff).exp()
            * (-salt.abs() * salt_tolerance_coefficient).exp()
    }

    pub fn max_size(&self) -> f32 {
        self.max_size.min(0.0).exp().max(self.max_size + 1.0) + 0.5
    }

    pub fn effective_growth_rate(&self, nutrients: f32, water: f32, temperature: f32, salt: f32, terrain: &TerrainType) -> f32 {
        1.5f32.powf(self.growth_rate) * 0.5 * self.base_growth_rate(nutrients, water, temperature, salt, terrain)
    }

    pub fn nutrient_addition_rate(&self) -> f32 {
        self.nutrient_addition_rate.max(0.0)
    }

    pub fn reproduction_rate(&self) -> f32 {
        sigmoid(self.reproduction_rate)
    }

    pub fn random() -> Genome {
        let crop_type = match fastrand::usize(0..4) {
            0 => CropType::Grass,
            1 => CropType::EucalyptusTree,
            2 => CropType::BushTomato,
            3 => CropType::GoldenWattleTree,
            _ => unreachable!()
        };

        let (nutrient_addition_rate, optimal_water_level, optimal_temperature, reproductive_size_fraction_param, salt_tolerance, max_size) = match crop_type {
            CropType::Grass =>            (-10.0, 0.0, 0.0, -1.0, 0.0, 1.0),
            CropType::EucalyptusTree =>   (-10.0, 1.0, 0.7,  1.0, 0.1, 5.0),
            CropType::BushTomato =>       (-10.0, 0.0, 1.0, -0.3, 0.2, 1.5),
            CropType::GoldenWattleTree => (  2.0, 0.5, 0.7,  0.5, 0.4, 3.0),

        };

        Genome {
            crop_type: crop_type,
            growth_rate: normal(),
            nutrient_addition_rate,
            optimal_water_level,
            optimal_temperature,
            reproduction_rate: normal(),
            reproductive_size_fraction_param: normal() + reproductive_size_fraction_param,
            temperature_tolerance: normal(),
            water_tolerance: normal(),
            salt_tolerance: normal() + salt_tolerance,
            max_size
        }
    }

    // TODO: this might be unreasonable
    pub fn water_efficiency(&self) -> f32 {
        sigmoid(self.optimal_water_level * 3.0 - 1.0)
    }

    pub fn hybridize(&self, other: &Genome) -> Option<Genome> {
        if self.crop_type != other.crop_type { return None }
        Some(Genome {
            crop_type: self.crop_type,
            growth_rate: (self.growth_rate + other.growth_rate) / 2.0 + normal_scaled(0.0, 0.1),
            nutrient_addition_rate: (self.nutrient_addition_rate + other.nutrient_addition_rate) / 2.0 + normal_scaled(0.0, 0.03),
            optimal_water_level: (self.optimal_water_level + other.optimal_water_level) / 2.0 + normal_scaled(0.0, 0.03),
            optimal_temperature: (self.optimal_temperature + other.optimal_temperature) / 2.0 + normal_scaled(0.0, 0.03),
            reproduction_rate: (self.reproduction_rate + other.reproduction_rate) / 2.0 + normal_scaled(0.0, 0.5),
            reproductive_size_fraction_param: (self.reproductive_size_fraction_param + other.reproductive_size_fraction_param) / 2.0 + normal_scaled(0.0, 0.2),
            temperature_tolerance: (self.temperature_tolerance + other.temperature_tolerance) / 2.0 + normal_scaled(0.0, 0.2),
            water_tolerance: (self.water_tolerance + other.water_tolerance) / 2.0 + normal_scaled(0.0, 0.2),
            salt_tolerance: (self.salt_tolerance + other.salt_tolerance) / 2.0 + normal_scaled(0.0, 0.2),
            max_size: (self.max_size + other.max_size) / 2.0 + normal_scaled(0.0, 0.3)
        })
    }
}
