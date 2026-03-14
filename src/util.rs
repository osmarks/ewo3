pub fn sigmoid(x: f32)-> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Box-Muller transform
pub fn normal(rng: &mut fastrand::Rng) -> f32 {
    let u = rng.f32();
    let v = rng.f32();
    ((v * std::f32::consts::TAU).cos() * (-2.0 * u.ln()).sqrt()).clamp(-6.0, 6.0)
}

pub fn normal_scaled(rng: &mut fastrand::Rng, mu: f32, sigma: f32) -> f32 {
    normal(rng) * sigma + mu
}

pub mod config {
    // Runtime game logic (plants)
    pub const PLANT_TICK_DELAY: u64 = 128;
    pub const FIELD_DECAY_DELAY: u64 = 100;
    pub const PLANT_GROWTH_SCALE: f32 = 0.01;
    pub const SOIL_NUTRIENT_CONSUMPTION_RATE: f32 = 0.8;
    pub const SOIL_NUTRIENT_FIXATION_RATE: f32 = 0.1;
    pub const WATER_CONSUMPTION_RATE: f32 = 0.05;
    pub const PLANT_IDLE_WATER_CONSUMPTION_OFFSET: f32 = 0.2;
    pub const PLANT_DIEOFF_THRESHOLD: f32 = 0.3;
    pub const PLANT_DIEOFF_RATE: f32 = 0.2;
    pub const AUTOSAVE_INTERVAL_TICKS: u64 = 1024;
    pub const PLANT_POLLINATION_RADIUS: i32 = 12; // TODO: should be directional (wind, insects, etc) and vary by plant.
    pub const PLANT_POLLINATION_SCAN_FRACTION: f32 = 0.1;
    pub const PLANT_SEEDING_RADIUS: i32 = 5; // TODO: as above
    pub const PLANT_SEEDING_ATTEMPTS: usize = 10;
    pub const SOIL_NUTRIENT_DECOMP_RETURN_RATE: f32 = 1.0 / SOIL_NUTRIENT_CONSUMPTION_RATE * 0.6;
    pub const PLANT_LIFESPAN_SCALE: f32 = 1.0 / PLANT_GROWTH_SCALE;
    pub const PLANT_REPRODUCTION_ATTEMPT_COST: f32 = 0.01;
    pub const INITIAL_PLANTS: usize = 131072;
    pub const EVOLUTION_RATE: f32 = 0.1;
    // Runtime game logic (misc)
    pub const VIEW: i32 = 15;
    pub const RANDOM_DESPAWN_INV_RATE: u64 = 4000;
    // Worldgen constants
    pub const WORLD_RADIUS: i32 = 1024;
    pub const NOISE_SCALE: f32 = 0.0005;
    pub const HEIGHT_EXPONENT: f32 = 0.3;
    pub const WATER_SOURCES: usize = 40;
    pub const CONTOUR_INTERVAL: f32 = 0.1;
    pub const SEA_LEVEL: f32 = -0.8;
    pub const EROSION: f32 = 0.09;
    pub const EROSION_EXPONENT: f32 = 1.5;
    pub const SALT_REMOVAL: f32 = 0.13;
    pub const SALT_RANGE: f32 = 0.2;
    pub const BASE_TEMPERATURE: f32 = 30.0; // degrees
    pub const HEIGHT_SCALE: f32 = 1e3; // unrealistic but makes world more interesting; m
    //const SEA_LEVEL_AIR_PRESSURE: f32 = 1013.0; // hPa
    //const PRESSURE_DROP_PER_METER: f32 = 0.001; // hPa m^-1
    pub const AIR_SPECIFIC_HEAT_CAPACITY: f32 = 1012.0; // J kg^-1 K^-1
    pub const EARTH_GRAVITY: f32 = 9.81; // m s^-2
    pub const NUTRIENT_NOISE_SCALE: f32 = 0.0015;
}
