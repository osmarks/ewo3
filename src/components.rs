use std::f32;

use indexmap::IndexMap;
use hecs::Entity;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};
use enum_map::{Enum, EnumMap};

use crate::map::*;
use crate::plant;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Item {
    Dirt,
    Bone,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Enum, Copy)]
pub enum HealthChangeType {
    BluntForce,
    Magic,
    NaturalRegeneration,
    Starvation
}

impl Item {
    pub fn name(&self) -> &'static str {
        match self {
            Item::Dirt => "Dirt",
            Item::Bone => "Bone",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Item::Dirt => "It's from the ground. You're carrying it for some reason.",
            Item::Bone => "Disassembling your enemies for resources is probably ethical.",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerCharacter;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum MapLayer {
    Particles,
    Entities,
    Terrain,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub layer: MapLayer,
    pub coords: SmallVec<[Coord; 2]>,
    pub move_count: u64,
}

pub struct PositionIndex {
    pub particles: Map<Option<Entity>>,
    pub entities: Map<Option<Entity>>,
    pub terrain: Map<Option<Entity>>,
}

impl PositionIndex {
    pub fn new(radius: i32) -> Self {
        Self {
            particles: Map::new(radius, None),
            entities: Map::new(radius, None),
            terrain: Map::new(radius, None),
        }
    }
}

impl Position {
    pub fn head(&self) -> Coord {
        self.coords[0]
    }

    pub fn single_tile(c: Coord, layer: MapLayer) -> Self {
        Self {
            layer,
            coords: smallvec![c],
            move_count: 0,
        }
    }

    pub fn iter_coords(&self) -> impl Iterator<Item = Coord> + '_ {
        self.coords.iter().copied()
    }

    pub fn record_for(&mut self, index: &mut PositionIndex, entity: Option<Entity>) {
        let target_layer = match self.layer {
            MapLayer::Particles => &mut index.particles,
            MapLayer::Entities => &mut index.entities,
            MapLayer::Terrain => &mut index.terrain,
        };
        for coord in self.coords.iter() {
            target_layer[*coord] = entity;
        }
    }

    pub fn record_for_erase(&mut self, index: &mut PositionIndex, entity: Entity) {
        let target_layer = match self.layer {
            MapLayer::Particles => &mut index.particles,
            MapLayer::Entities => &mut index.entities,
            MapLayer::Terrain => &mut index.terrain,
        };
        for coord in self.coords.iter() {
            if target_layer[*coord] == Some(entity) {
                target_layer[*coord] = None;
            }
        }
    }

    // Return value indicates whether any coordinate remains.
    pub fn remove_coord(&mut self, coord: Coord, index: &mut PositionIndex, entity: Entity) -> bool {
        self.record_for(index, None);
        self.coords.retain(|x| *x != coord);
        self.record_for(index, Some(entity));
        !self.coords.is_empty()
    }

    pub fn move_into(&mut self, coord: Coord, index: &mut PositionIndex, entity: Entity) -> Coord {
        self.record_for(index, None);
        let fst = self.coords.remove(0);
        self.coords.push(coord);
        self.record_for(index, Some(entity));
        self.move_count += 1;
        fst
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MovingInto(pub Coord);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthChangeModifier {
    fraction: f32,
    offset: f32
}

impl HealthChangeModifier {
    fn default() -> Self {
        HealthChangeModifier { fraction: 1.0, offset: 0.0 }
    }

    fn invulnerable() -> Self {
        HealthChangeModifier { fraction: 0.0, offset: 0.0 }
    }

    fn adjust(&self, delta: f32) -> f32 {
        if delta > 0.0 {
            (delta + self.offset).max(0.0) * self.fraction
        } else {
            (delta + self.offset).min(0.0) * self.fraction
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub current: f32,
    pub max: f32,
    pub health_change_totals: EnumMap<HealthChangeType, f32>,
    pub health_change_modifiers: EnumMap<HealthChangeType, HealthChangeModifier>
}

impl Health {
    pub fn pct(&self) -> f32 {
        if self.max == 0.0 {
            0.0
        } else {
            self.current / self.max
        }
    }

    pub fn new(current: f32, max: f32) -> Self {
        Self {
            current,
            max,
            health_change_modifiers: EnumMap::from_fn(|_| HealthChangeModifier::default()),
            health_change_totals: EnumMap::from_fn(|_| 0.0)
        }
    }

    pub fn apply(&mut self, ty: HealthChangeType, delta: f32) {
        let original = self.current;
        let adjusted_delta = self.health_change_modifiers[ty].adjust(delta);
        self.current = (self.current + adjusted_delta).min(self.max);
        let real_change = self.current - original;
        self.health_change_totals[ty] += real_change;
    }

    pub fn invulnerable() -> Self {
        Self {
            current: f32::MAX,
            max: f32::MAX,
            health_change_modifiers: EnumMap::from_fn(|_| HealthChangeModifier::invulnerable()),
            health_change_totals: EnumMap::from_fn(|_| 0.0)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkOnDeath;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Render(pub char);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StochasticNumber {
    Constant(f32),
    Triangle { min: f32, max: f32, mode: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attack {
    pub damage: SmallVec<[(HealthChangeType, StochasticNumber); 1]>,
    pub energy: f32,
    pub hits: u64,
    pub kills: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangedAttack {
    pub damage: SmallVec<[(HealthChangeType, StochasticNumber); 1]>,
    pub energy: f32,
    pub range: u64,
    pub firings: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DespawnOnTick(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DespawnRandomly(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyTarget {
    pub spawn_range: std::ops::RangeInclusive<i32>,
    pub spawn_density: f32,
    pub spawn_rate_inv: usize,
    pub aggression_range: i32,
    pub spawn_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Enemy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveCost(pub StochasticNumber);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Velocity(pub CoordVec);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstruction {
    pub entry_multiplier: f32,
    pub exit_multiplier: f32,
    pub obstruction_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Energy {
    pub current: f32,
    pub regeneration_rate: f32,
    pub burst: f32,
    pub total_used: f32,
}

impl Energy {
    pub fn try_consume(&mut self, cost: f32) -> bool {
        if self.current >= -1e-12 {
            self.current -= cost;
            self.total_used += cost;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drops(pub Vec<(Item, StochasticNumber)>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jump(pub i32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DespawnOnImpact;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    pub contents: IndexMap<Item, u64>,
    pub added_items: u64,
    pub additions: u64,
    pub taken_items: u64,
    pub takings: u64,
}

impl Inventory {
    pub fn add(&mut self, item: Item, qty: u64) {
        *self.contents.entry(item).or_default() += qty;
        self.added_items += qty;
        self.additions += 1;
    }

    pub fn extend(&mut self, other: &Inventory) {
        for (item, count) in other.contents.iter() {
            self.add(item.clone(), *count);
        }
    }

    pub fn take(&mut self, item: Item, qty: u64) -> bool {
        match self.contents.entry(item) {
            indexmap::map::Entry::Occupied(mut o) => {
                let current = o.get_mut();
                if *current >= qty {
                    *current -= qty;
                    self.taken_items += qty;
                    self.takings += 1;
                    return true;
                }
                false
            }
            indexmap::map::Entry::Vacant(_) => false,
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.contents.iter().any(|(_, c)| *c > 0)
    }

    pub fn empty() -> Self {
        Self {
            contents: IndexMap::new(),
            added_items: 0,
            additions: 0,
            taken_items: 0,
            takings: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plant {
    pub genome: plant::Genome,
    pub current_size: f32,
    pub nutrients_consumed: f32,
    pub nutrients_added: f32,
    pub water_consumed: f32,
    pub growth_ticks: u64,
    pub children: u64,
    pub ready_for_reproduce_ticks: u64,
}

impl Plant {
    pub fn new(genome: plant::Genome) -> Self {
        Self {
            genome,
            current_size: 0.1,
            nutrients_consumed: 0.0,
            nutrients_added: 0.0,
            water_consumed: 0.0,
            growth_ticks: 0,
            children: 0,
            ready_for_reproduce_ticks: 0,
        }
    }

    pub fn can_reproduce(&self) -> bool {
        self.current_size >= self.genome.max_size() * self.genome.reproductive_size_fraction()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewlyAdded;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatedAt(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocksEnemySpawn(pub u64);

fn triangle_distribution(min: f32, max: f32, mode: f32, rng: &mut fastrand::Rng) -> f32 {
    let sample = rng.f32();
    let threshold = (mode - min) / (max - min);
    if sample < threshold {
        min + (sample * (max - min) * (mode - min)).sqrt()
    } else {
        max - ((1.0 - sample) * (max - min) * (max - mode)).sqrt()
    }
}

impl StochasticNumber {
    pub fn sample(&self, rng: &mut fastrand::Rng) -> f32 {
        match self {
            StochasticNumber::Constant(x) => *x,
            StochasticNumber::Triangle { min, max, mode } => {
                triangle_distribution(*min, *max, *mode, rng)
            }
        }
    }

    pub fn sample_rounded<T: TryFrom<i128>>(&self, rng: &mut fastrand::Rng) -> T {
        T::try_from(self.sample(rng).round() as i128)
            .map_err(|_| "conversion failed")
            .unwrap()
    }

    pub fn triangle_from_min_range(min: f32, range: f32) -> Self {
        StochasticNumber::Triangle {
            min,
            max: min + range,
            mode: (min + range) / 2.0,
        }
    }
}
