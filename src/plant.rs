#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CropType {
    Grass,
    EucalyptusTree,
    BushTomato,
    GoldenWattleTree
}

#[derive(Debug, Clone, Copy)]
pub struct Genome {
    crop_type: CropType,
    // polygenic traits; parameterized as N(0,1)
    growth_rate: f32,
    nitrogen_fixation_rate: f32,
    optimal_water_level: f32,
    optimal_temperature: f32,
    reproduction_rate: f32,
    temperature_tolerance: f32,
    water_tolerance: f32,
    max_size: f32
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
    pub fn effective_growth_rate(&self, water: f32, temperature: f32) -> f32 {
        let water_diff = (water - self.optimal_water_level).abs();
        let temperature_diff = (temperature - self.optimal_temperature).abs();
        1.5f32.powf(self.growth_rate)
            - self.reproduction_rate * 0.1 // faster reproduction trades off slightly against growth
            - self.nitrogen_fixation_rate.max(0.0) * 0.16 // same for nitrogen fixation
            - (water_diff - sigmoid(self.water_tolerance)).max(0.0) // penalize plants when far from optimal environmental range
            - (temperature_diff - sigmoid(self.temperature_tolerance)).max(0.0) // same for temperature
            - self.water_tolerance * 0.2
            - self.temperature_tolerance * 0.2
    }

    pub fn random() -> Genome {
        let crop_type = match fastrand::usize(0..4) {
            0 => CropType::Grass,
            1 => CropType::EucalyptusTree,
            2 => CropType::BushTomato,
            3 => CropType::GoldenWattleTree,
            _ => unreachable!()
        };

        let (nitrogen_fixation_rate, optimal_water_level, optimal_temperature, max_size) = match crop_type {
            CropType::Grass => (-10.0, 0.0, 0.0, 0.0),
            CropType::EucalyptusTree => (-10.0, 2.0, 1.0, 5.0),
            CropType::BushTomato => (-10.0, -1.0, 1.5, 1.0),
            CropType::GoldenWattleTree => (2.0, 1.5, 1.0, 3.0),

        };

        Genome {
            crop_type: crop_type,
            growth_rate: normal(),
            nitrogen_fixation_rate,
            optimal_water_level,
            optimal_temperature,
            reproduction_rate: normal(),
            temperature_tolerance: normal(),
            water_tolerance: normal(),
            max_size
        }
    }

    pub fn hybridize(&self, other: &Genome) -> Option<Genome> {
        if self.crop_type != other.crop_type { return None }
        Some(Genome {
            crop_type: self.crop_type,
            growth_rate: (self.growth_rate + other.growth_rate) / 2.0 + normal_scaled(0.0, 0.1),
            nitrogen_fixation_rate: (self.nitrogen_fixation_rate + other.nitrogen_fixation_rate) / 2.0 + normal_scaled(0.0, 0.03),
            optimal_water_level: (self.optimal_water_level + other.optimal_water_level) / 2.0 + normal_scaled(0.0, 0.03),
            optimal_temperature: (self.optimal_temperature + other.optimal_temperature) / 2.0 + normal_scaled(0.0, 0.03),
            reproduction_rate: (self.reproduction_rate + other.reproduction_rate) / 2.0 + normal_scaled(0.0, 0.5),
            temperature_tolerance: (self.temperature_tolerance + other.temperature_tolerance) / 2.0 + normal_scaled(0.0, 0.2),
            water_tolerance: (self.water_tolerance + other.water_tolerance) / 2.0 + normal_scaled(0.0, 0.2),
            max_size: (self.max_size + other.max_size) / 2.0 + normal_scaled(0.0, 0.3)
        })
    }
}
