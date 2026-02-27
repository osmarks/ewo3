#![feature(test)]
extern crate test;

use hecs::{CommandBuffer, Entity, With, World};
use futures_util::{stream::TryStreamExt, SinkExt, StreamExt};
use indexmap::IndexMap;
use smallvec::{smallvec, SmallVec};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio::sync::{mpsc, Mutex};
use anyhow::{Result, Context, anyhow};
use std::{convert::TryFrom, hash::{Hash, Hasher}, net::SocketAddr, ops::DerefMut, sync::Arc, time::Duration};
use slab::Slab;
use serde::{Serialize, Deserialize};

pub mod worldgen;
pub mod map;
pub mod plant;
pub mod world_serde;

use map::*;

async fn handle_connection(raw_stream: TcpStream, addr: SocketAddr, mut frames_rx: mpsc::Receiver<Frame>, inputs_tx: mpsc::Sender<Input>) -> Result<()> {
    let ws_stream = tokio_tungstenite::accept_async(raw_stream).await.context("websocket handshake failure")?;
    let (mut outgoing, incoming) = ws_stream.split();

    let broadcast_incoming = incoming.map_err(anyhow::Error::from).try_for_each(|msg| {
        let inputs_tx = inputs_tx.clone();
        async move {
            if msg.is_close() { return Err(anyhow!("connection closed")) }

            let input: Input = serde_json::from_str(msg.to_text()?)?;

            inputs_tx.send(input).await?;

            anyhow::Result::<(), anyhow::Error>::Ok(())
        }
    });

    let send_state = async move {
        while let Some(frame) = frames_rx.recv().await {
            outgoing.send(Message::Text(serde_json::to_string(&frame)?)).await?;
            match frame {
                Frame::Dead => return Ok(()),
                _ => ()
            }
        }
        anyhow::Result::<(), anyhow::Error>::Ok(())
    };

    tokio::select! {
        result = broadcast_incoming => {
            println!("{:?}", result)
        },
        result = send_state => {
            println!("{:?}", result)
        }
    };

    println!("{} disconnected", &addr);

    Ok(())
}

#[derive(Serialize, Deserialize, Clone)]
enum Input {
    UpLeft,
    UpRight,
    Left,
    Right,
    DownLeft,
    DownRight,
    Dig
}

#[derive(Serialize, Deserialize, Clone)]
enum Frame {
    Dead,
    Display { nearby: Vec<(i32, i32, char, f32)>, health: f32, inventory: Vec<(String, String, u64)> },
    PlayerCount(usize),
    Message(String)
}

struct Client {
    inputs_rx: mpsc::Receiver<Input>,
    frames_tx: mpsc::Sender<Frame>,
    entity: Entity
}

struct GameState {
    world: World,
    clients: Slab<Client>,
    ticks: u64,
    rng: fastrand::Rng,
    map: worldgen::GeneratedWorld,
    baseline_soil_nutrients: Map<f32>,
    baseline_groundwater: Map<f32>,
    baseline_salt: Map<f32>,
    baseline_temperature: Map<f32>,
    dynamic_soil_nutrients: Map<f32>,
    dynamic_groundwater: Map<f32>,
    positions: PositionIndex
}

impl GameState {
    fn actual_groundwater(&self, pos: Coord) -> f32 {
        self.baseline_groundwater[pos] + self.dynamic_groundwater[pos]
    }

    fn actual_soil_nutrients(&self, pos: Coord) -> f32 {
        self.baseline_soil_nutrients[pos] + self.dynamic_soil_nutrients[pos]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum Item {
    Dirt,
    Bone
}

impl Item {
    fn name(&self) -> &'static str {
        use Item::*;
        match self {
            Dirt => "Dirt",
            Bone => "Bone"
        }
    }

    fn description(&self) -> &'static str {
        use Item::*;
        match self {
            Dirt => "It's from the ground. You're carrying it for some reason.",
            Bone => "Disassembling your enemies for resources is probably ethical."
        }
    }
}

struct PositionIndex {
    particles: Map<Option<Entity>>,
    entities: Map<Option<Entity>>,
    terrain: Map<Option<Entity>>
}

impl PositionIndex {
    fn new(radius: i32) -> Self {
        Self {
            particles: Map::new(radius, None),
            entities: Map::new(radius, None),
            terrain: Map::new(radius, None)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlayerCharacter;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
enum MapLayer {
    Particles,
    Entities,
    Terrain
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Position {
    layer: MapLayer,
    coords: SmallVec<[Coord; 2]>,
    move_count: u64
}

impl Position {
    fn head(&self) -> Coord {
        self.coords[0]
    }

    fn single_tile(c: Coord, layer: MapLayer) -> Self {
        Self {
            layer,
            coords: smallvec![c],
            move_count: 0
        }
    }

    fn iter_coords(&self) -> impl Iterator<Item=Coord> + '_ {
        self.coords.iter().copied()
    }

    fn record_for(&mut self, index: &mut PositionIndex, entity: Option<Entity>) {
        let target_layer = match self.layer {
            MapLayer::Particles => &mut index.particles,
            MapLayer::Entities => &mut index.entities,
            MapLayer::Terrain => &mut index.terrain,
        };
        for coord in self.coords.iter() {
            target_layer[*coord] = entity;
        }
    }

    // return value is whether it is now dead/positionless
    fn remove_coord(&mut self, coord: Coord, index: &mut PositionIndex, entity: Entity) -> bool {
        self.record_for(index, None);
        self.coords.retain(|x| *x != coord);
        self.record_for(index, Some(entity));
        self.coords.len() > 0
    }

    fn move_into(&mut self, coord: Coord, index: &mut PositionIndex, entity: Entity) -> Coord {
        self.record_for(index, None);
        let fst = self.coords.remove(0);
        self.coords.push(coord);
        self.record_for(index, Some(entity));
        self.move_count += 1;
        fst
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct MovingInto(Coord);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Health { current: f32, max: f32, damage_taken: f32, healing_taken: f32 }

impl Health {
    fn pct(&self) -> f32 {
        if self.max == 0.0 { 0.0 }
        else { self.current / self.max }
    }

    fn new(current: f32, max: f32) -> Self {
        Health { current, max, damage_taken: 0.0, healing_taken: 0.0 }
    }

    fn apply(&mut self, delta: f32) {
        let original = self.current;
        self.current = (self.current + delta).min(self.max);
        if delta < 0.0 {
            self.damage_taken += original - self.current;
        } else {
            self.healing_taken += self.current - original;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShrinkOnDeath;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Render(char);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Attack { damage: StochasticNumber, energy: f32, hits: u64, kills: u64 }

// TODO: would be nice to track kills on RangedAttacks
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RangedAttack { damage: StochasticNumber, energy: f32, range: u64, firings: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DespawnOnTick(u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DespawnRandomly(u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EnemyTarget { spawn_range: std::ops::RangeInclusive<i32>, spawn_density: f32, spawn_rate_inv: usize, aggression_range: i32, spawn_count: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Enemy;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoveCost(StochasticNumber);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Velocity(CoordVec);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Obstruction { entry_multiplier: f32, exit_multiplier: f32, obstruction_count: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Energy { current: f32, regeneration_rate: f32, burst: f32, total_used: f32 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Drops(Vec<(Item, StochasticNumber)>);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Jump(i32);

impl Energy {
    fn try_consume(&mut self, cost: f32) -> bool {
        if self.current >= -1e-12 { // numerics
            self.current -= cost;
            self.total_used += cost;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DespawnOnImpact;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Inventory { contents: indexmap::IndexMap<Item, u64>, added_items: u64, additions: u64, taken_items: u64, takings: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Plant {
    genome: plant::Genome,
    current_size: f32,
    nutrients_consumed: f32,
    nutrients_added: f32,
    water_consumed: f32,
    growth_ticks: u64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewlyAdded; // ugly hack to work around ECS deficiencies

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreatedAt(u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlocksEnemySpawn(u64);

impl Inventory {
    fn add(&mut self, item: Item, qty: u64) {
        *self.contents.entry(item).or_default() += qty;
        self.added_items += qty;
        self.additions += 1;
    }

    fn take(&mut self, item: Item, qty: u64) -> bool {
        match self.contents.entry(item) {
            indexmap::map::Entry::Occupied(mut o) => {
                let current = o.get_mut();
                if *current >= qty {
                    *current -= qty;
                    self.taken_items += qty;
                    self.takings += 1;
                    return true;
                }
                return false;
            },
            indexmap::map::Entry::Vacant(_) => return false
        }
    }

    fn extend(&mut self, other: &Inventory) {
        for (item, count) in other.contents.iter() {
            self.add(item.clone(), *count);
        }
    }

    fn is_empty(&self) -> bool {
        !self.contents.iter().any(|(_, c)| *c > 0)
    }

    fn empty() -> Self {
        Self { contents: IndexMap::new(), added_items: 0, additions: 0, taken_items: 0, takings: 0 }
    }
}

const VIEW: i32 = 15;
const RANDOM_DESPAWN_INV_RATE: u64 = 4000;

struct EnemySpec {
    symbol: char,
    min_damage: f32,
    damage_range: f32,
    initial_health: f32,
    move_delay: usize,
    attack_cooldown: u64,
    ranged: bool,
    movement: i32,
    drops: Vec<(Item, StochasticNumber)>
}

impl EnemySpec {
    // Numbers ported from original EWO. Fudge constants added elsewhere.
    fn random(rng: &mut fastrand::Rng) -> EnemySpec {
        match rng.usize(0..650) {
            0..=99 => EnemySpec { symbol: 'I', min_damage: 10.0, damage_range: 5.0, initial_health: 50.0, move_delay: 70, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // IBIS
            100..=199 => EnemySpec { symbol: 'K', min_damage: 5.0, damage_range: 25.0, initial_health: 60.0, move_delay: 30, attack_cooldown: 12, ranged: false, drops: vec![], movement: 2 }, // KANGAROO
            200..=299 => EnemySpec { symbol: 'S', min_damage: 5.0, damage_range: 5.0, initial_health: 20.0, move_delay: 50, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // SNAKE
            300..=399 => EnemySpec { symbol: 'E', min_damage: 10.0, damage_range: 20.0, initial_health: 80.0, move_delay: 80, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // EMU
            400..=499 => EnemySpec { symbol: 'O', min_damage: 8.0, damage_range: 17.0, initial_health: 150.0, move_delay: 100, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // OGRE
            500..=599 => EnemySpec { symbol: 'R', min_damage: 5.0, damage_range: 5.0, initial_health: 15.0, move_delay: 40, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // RAT
            600..=609 => EnemySpec { symbol: 'M' , min_damage: 20.0, damage_range: 10.0, initial_health: 150.0, move_delay: 70, attack_cooldown: 10, ranged: false, drops: vec![], movement: 1 }, // MOA
            610..=649 => EnemySpec { symbol: 'P', min_damage: 10.0, damage_range: 5.0, initial_health: 15.0, move_delay: 20, attack_cooldown: 10, ranged: true, drops: vec![], movement: 1 }, // PLATYPUS
            _ => unreachable!()
        }
    }
}

fn rng_from_hash<H: Hash>(x: H) -> fastrand::Rng {
    let mut h = seahash::SeaHasher::new();
    x.hash(&mut h);
    fastrand::Rng::with_seed(h.finish())
}

fn sample_range_rng(range: i32, rng: &mut fastrand::Rng) -> CoordVec {
    let q = rng.i32(-range..=range);
    let r = rng.i32((-range).max(-q - range)..=range.min(-q + range));
    CoordVec::new(q, r)
}

fn consume_energy_if_available<R: DerefMut<Target=Energy>>(e: &mut Option<R>, cost: f32) -> bool {
    e.is_none() || e.as_mut().unwrap().try_consume(cost)
}

fn triangle_distribution(min: f32, max: f32, mode: f32, rng: &mut fastrand::Rng) -> f32 {
    let sample = rng.f32();
    let threshold = (mode - min) / (max - min);
    if sample < threshold {
        min + (sample * (max - min) * (mode - min)).sqrt()
    } else {
        max - ((1.0 - sample) * (max - min) * (max - mode)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum StochasticNumber {
    Constant(f32),
    Triangle { min: f32, max: f32, mode: f32 }
}

fn rebuild_position_index(world: &World, radius: i32) -> PositionIndex {
    let mut index = PositionIndex::new(radius);
    for (entity, position) in world.query::<&mut Position>().iter() {
        position.record_for(&mut index, Some(entity));
    }
    index
}

#[derive(Serialize, Deserialize)]
struct SavedGame {
    ticks: u64,
    rng_seed: u64,
    map: worldgen::GeneratedWorld,
    dynamic_soil_nutrients: Map<f32>,
    dynamic_groundwater: Map<f32>,
    world: Vec<u8>,
}

impl SavedGame {
    fn from_state(state: &GameState) -> Result<Self> {
        Ok(Self {
            ticks: state.ticks,
            rng_seed: state.rng.get_seed(),
            map: state.map.clone(),
            dynamic_soil_nutrients: state.dynamic_soil_nutrients.clone(),
            dynamic_groundwater: state.dynamic_groundwater.clone(),
            world: world_serde::serialize_world_to_bytes(&state.world)?,
        })
    }

    fn into_state(self) -> Result<GameState> {
        let world = world_serde::deserialize_world_from_bytes(&self.world)?;
        let positions = rebuild_position_index(&world, self.map.radius);
        let baseline_soil_nutrients = self.map.soil_nutrients.clone();
        let baseline_groundwater = self.map.groundwater.clone();
        let baseline_salt = self.map.salt.clone();
        let baseline_temperature = self.map.temperature.clone();
        Ok(GameState {
            world,
            clients: Slab::new(),
            ticks: self.ticks,
            rng: fastrand::Rng::with_seed(self.rng_seed),
            map: self.map,
            baseline_soil_nutrients,
            baseline_groundwater,
            baseline_salt,
            baseline_temperature,
            dynamic_soil_nutrients: self.dynamic_soil_nutrients,
            dynamic_groundwater: self.dynamic_groundwater,
            positions,
        })
    }
}

impl StochasticNumber {
    fn sample(&self, rng: &mut fastrand::Rng) -> f32 {
        match self {
            StochasticNumber::Constant(x) => *x,
            StochasticNumber::Triangle { min, max, mode } => triangle_distribution(*min, *max, *mode, rng)
        }
    }

    fn sample_rounded<T: TryFrom<i128>>(&self, rng: &mut fastrand::Rng) -> T {
        T::try_from(self.sample(rng).round() as i128).map_err(|_| "convert fail").unwrap()
    }

    fn triangle_from_min_range(min: f32, range: f32) -> Self {
        StochasticNumber::Triangle { min: min, max: min + range, mode: (min + range) / 2.0 }
    }
}

const PLANT_TICK_DELAY: u64 = 128;
const FIELD_DECAY_DELAY: u64 = 100;
const PLANT_GROWTH_SCALE: f32 = 0.01;
const SOIL_NUTRIENT_CONSUMPTION_RATE: f32 = 0.04;
const SOIL_NUTRIENT_FIXATION_RATE: f32 = 0.0002;
const WATER_CONSUMPTION_RATE: f32 = 0.02;
const PLANT_IDLE_WATER_CONSUMPTION_OFFSET: f32 = 0.05;
const PLANT_DIEOFF_THRESHOLD: f32 = 0.01;
const PLANT_DIEOFF_RATE: f32 = 0.005;
const SAVE_FILE: &str = "save.bin";
const AUTOSAVE_INTERVAL_TICKS: u64 = 1024;

async fn game_tick(state: &mut GameState) -> Result<()> {
    let mut rng = fastrand::Rng::with_seed(state.rng.get_seed());
    let mut buffer = hecs::CommandBuffer::new();

    for (entity, position) in state.world.query_mut::<With<&mut Position, &NewlyAdded>>() {
        position.record_for(&mut state.positions, Some(entity));
        buffer.remove_one::<NewlyAdded>(entity);
        buffer.insert_one(entity, CreatedAt(state.ticks));
    }

    buffer.run_on(&mut state.world);

    if state.ticks % FIELD_DECAY_DELAY == 0 {
        state.dynamic_soil_nutrients.for_each_mut(|nutrients| *nutrients *= 0.999);
    } else if state.ticks % FIELD_DECAY_DELAY == 1 {
        state.dynamic_groundwater.for_each_mut(|water| *water *= 0.999);
    } else if state.ticks % FIELD_DECAY_DELAY == 2 {
        state.dynamic_soil_nutrients = smooth(&state.dynamic_soil_nutrients, 3);
    } else if state.ticks % FIELD_DECAY_DELAY == 3 {
        state.dynamic_groundwater = smooth(&state.dynamic_groundwater, 3);
    }

    let mut despawn_buffer = Vec::new();

    // This might lead to a duping glitch, which would at least be funny.
    // TODO: Players should drop items on disconnect.
    // The final position argument is in some sense redundant but exists to satisfy dynamic borrow checking.
    let kill = |buffer: &mut CommandBuffer, despawn_buffer: &mut Vec<(Entity, Position)>, state: &GameState, rng: &mut fastrand::Rng, entity: Entity, killer: Option<Entity>| {
        let position = (*state.world.get::<&Position>(entity).unwrap()).clone();
        let position_head = position.head();
        despawn_buffer.push((entity, position));
        buffer.despawn(entity);
        let mut materialized_drops = Inventory::empty();
        if let Ok(drops) = state.world.get::<&Drops>(entity) {
            for (drop, frequency) in drops.0.iter() {
                materialized_drops.add(drop.clone(), frequency.sample_rounded(rng))
            }
        }
        if let Ok(other_inv) = state.world.get::<&Inventory>(entity) {
            materialized_drops.extend(&other_inv);
        }
        let killer_consumed_items = if let Some(killer) = killer {
            if let Ok(mut inv) = state.world.get::<&mut Inventory>(killer) {
                inv.extend(&materialized_drops);
                true
            } else {
                false
            }
        } else { false };
        if !killer_consumed_items && !materialized_drops.is_empty() {
            buffer.spawn((
                Position::single_tile(position_head, MapLayer::Entities),
                Render('☒'),
                materialized_drops,
                NewlyAdded,
                Health::new(10.0, 10.0)
            ));
        }
    };

    // Spawn enemies
    for (_entity, (pos, EnemyTarget { spawn_range, spawn_density, spawn_rate_inv, spawn_count, .. })) in state.world.query::<(&Position, &mut EnemyTarget)>().iter() {
        let pos = pos.head();
        if rng.usize(0..*spawn_rate_inv) == 0 {
            let c = count_hexes(*spawn_range.end());
            let mut newpos = pos + sample_range_rng(*spawn_range.end(), &mut rng);
            let mut occupied = false;
            for _ in 0..(c as f32 / *spawn_density * 0.005).ceil() as usize {
                if let Some(entity) = state.positions.entities[newpos] {
                    if let Ok(mut count) = state.world.get::<&mut BlocksEnemySpawn>(entity) {
                        occupied = true;
                        count.0 += 1;
                        break;
                    }
                }
                newpos = pos + sample_range_rng(*spawn_range.end(), &mut rng);
            }
            if !occupied && state.map.get_terrain(newpos).entry_cost().is_some() && hex_distance(newpos, pos) >= *spawn_range.start() {
                let mut spec = EnemySpec::random(&mut rng);
                spec.drops.push((Item::Bone, StochasticNumber::Triangle { min: 0.7 * spec.initial_health / 40.0, max: 1.3 * spec.initial_health / 40.0, mode: spec.initial_health / 40.0 }));
                if spec.ranged {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health::new(spec.initial_health, spec.initial_health),
                        Enemy,
                        RangedAttack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32, range: 4, firings: 0 },
                        Position::single_tile(newpos, MapLayer::Entities),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0, total_used: 0.0 },
                        Drops(spec.drops),
                        Jump(spec.movement),
                        NewlyAdded,
                        BlocksEnemySpawn(0)
                    ))
                } else {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health::new(spec.initial_health, spec.initial_health),
                        Enemy,
                        Attack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32, hits: 0, kills: 0 },
                        Position::single_tile(newpos, MapLayer::Entities),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0, total_used: 0.0 },
                        Drops(spec.drops),
                        Jump(spec.movement),
                        NewlyAdded,
                        BlocksEnemySpawn(0)
                    ))
                };
                *spawn_count += 1;
            }
        }
    }

    buffer.run_on(&mut state.world);

    // Run plant simulations.
    for (entity, (pos, plant)) in state.world.query::<(&Position, &mut Plant)>().iter() {
        if (entity.id() as u64) % PLANT_TICK_DELAY == state.ticks % PLANT_TICK_DELAY {
            let pos = pos.head();
            let water = state.actual_groundwater(pos);
            let soil_nutrients = state.actual_soil_nutrients(pos);
            let salt = state.baseline_salt[pos];
            let temperature = state.baseline_temperature[pos];
            let base_growth_rate = plant.genome.effective_growth_rate(soil_nutrients, water, temperature, salt);
            if base_growth_rate < PLANT_DIEOFF_THRESHOLD {
                if let Ok(mut health) = state.world.get::<&mut Health>(entity) {
                    health.apply(PLANT_DIEOFF_RATE);
                    if health.current <= 0.0 {
                        // TODO: this is inelegant and should be shared with the other death code
                        // also, it might break the position tracker
                        kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, entity, None);
                    }
                }
            }
            let original_size = plant.current_size;
            plant.current_size += base_growth_rate * PLANT_GROWTH_SCALE * plant.current_size.powf(-0.25); // allometric scaling law
            plant.current_size = plant.current_size.min(plant.genome.max_size);
            let difference = (plant.current_size - original_size).max(0.0);
            let can_reproduce = plant.current_size >= plant.genome.max_size * plant.genome.reproductive_size_fraction();
            if can_reproduce {
                // TODO: Implement reproduction logic
            }
            state.dynamic_soil_nutrients[pos] -= difference * SOIL_NUTRIENT_CONSUMPTION_RATE;
            plant.nutrients_consumed += difference * SOIL_NUTRIENT_CONSUMPTION_RATE;
            state.dynamic_soil_nutrients[pos] += plant.genome.nutrient_addition_rate() * SOIL_NUTRIENT_FIXATION_RATE;
            plant.nutrients_added += plant.genome.nutrient_addition_rate() * SOIL_NUTRIENT_FIXATION_RATE;
            let water_consumed = (difference * WATER_CONSUMPTION_RATE + PLANT_IDLE_WATER_CONSUMPTION_OFFSET) * plant.genome.water_efficiency();
            state.dynamic_groundwater[pos] -= water_consumed;
            plant.water_consumed += water_consumed;

            if difference > 0.0 {
                plant.growth_ticks += 1;
            }
        }
    }

    buffer.run_on(&mut state.world);

    // Process enemy motion and ranged attacks
    for (entity, (pos, ranged, energy, jump)) in state.world.query::<hecs::With<(&Position, Option<&mut RangedAttack>, Option<&mut Energy>, Option<&Jump>), &Enemy>>().iter() {
        let pos = pos.head();

        for direction in DIRECTIONS.iter() {
            if let Some(target) = &state.positions.entities[pos + *direction] {
                if let Ok(_) = state.world.get::<&EnemyTarget>(*target) {
                    buffer.insert_one(entity, MovingInto(pos + *direction));
                    continue;
                }
            }
        }

        let mut closest = None;

        // TODO we maybe need a spatial index for this
        for (_entity, (target_pos, target)) in state.world.query::<(&Position, &EnemyTarget)>().iter() {
            let target_pos = target_pos.head();
            let distance = hex_distance(pos, target_pos);
            if distance < target.aggression_range {
                match closest {
                    Some((_pos, old_distance)) if old_distance < distance => closest = Some((target_pos, distance)),
                    None => closest = Some((target_pos, distance)),
                    _ => ()
                }
            }
        }

        if let Some((target_pos, _)) = closest {
            if let Some(ranged_attack) = ranged {
                // slightly smart behaviour for ranged attacker: try to stay just within range
                let direction = DIRECTIONS.iter().min_by_key(|dir|
                    (hex_distance(pos + **dir, target_pos) - (ranged_attack.range as i32 - 1)).abs()).unwrap();
                buffer.insert_one(entity, MovingInto(pos + *direction));
                // do ranged attack if valid
                let atk_dir = target_pos - pos;
                if on_axis(atk_dir) && (energy.is_none() || energy.unwrap().try_consume(ranged_attack.energy)) {
                    let atk_dir = atk_dir.clamp(-CoordVec::one(), CoordVec::one());
                    // TODO: for future metrics reasons, perhaps track source
                    buffer.spawn((
                        Render('*'),
                        Enemy,
                        Attack { damage: ranged_attack.damage, energy: 0.0, hits: 0, kills: 0 },
                        Velocity(atk_dir),
                        Position::single_tile(pos, MapLayer::Particles),
                        DespawnOnTick(state.ticks.wrapping_add(ranged_attack.range)),
                        DespawnOnImpact,
                        NewlyAdded
                    ));
                    ranged_attack.firings += 1;
                }
            } else {
                let direction = *DIRECTIONS.iter().min_by_key(|dir| hex_distance(pos + **dir, target_pos)).unwrap();
                let max_movement_distance = jump.map(|j| j.0).unwrap_or(1);
                let mut best_scale = 1;
                let mut best_distance = hex_distance(pos + direction, target_pos);
                for i in 1..=max_movement_distance {
                    let new_distance = hex_distance(pos + direction * i, target_pos);
                    if new_distance < best_distance {
                        best_distance = new_distance;
                        best_scale = i;
                    }
                }
                buffer.insert_one(entity, MovingInto(pos + direction * best_scale));
            }
        } else {
            // wander randomly (ethical)
            let direction = DIRECTIONS[rng.usize(0..DIRECTIONS.len())];
            buffer.insert_one(entity, MovingInto(pos + direction));
        }
    }

    // Process velocity
    for (entity, (pos, Velocity(vel))) in state.world.query_mut::<(&Position, &Velocity)>() {
        buffer.insert_one(entity, MovingInto(pos.head() + *vel));
    }

    buffer.run_on(&mut state.world);

    // Process inputs
    for (_id, client) in state.clients.iter_mut() {
        let mut next_movement = CoordVec::zero();
        let position = state.world.get::<&Position>(client.entity).context("read player")?.head();
        let mut energy = state.world.get::<&mut Energy>(client.entity)?;
        let mut inventory = state.world.get::<&mut Inventory>(client.entity)?;
        loop {
            let recv = client.inputs_rx.try_recv();
            match recv {
                Err(e) if e == mpsc::error::TryRecvError::Empty => break,
                Ok(input) => match input {
                    Input::UpLeft => {
                        next_movement = CoordVec::new(0, -1);
                    },
                    Input::UpRight => {
                        next_movement = CoordVec::new(1, -1);
                    },
                    Input::Left => {
                        next_movement = CoordVec::new(-1, 0);
                    },
                    Input::Right => {
                        next_movement = CoordVec::new(1, 0);
                    },
                    Input::DownRight => {
                        next_movement = CoordVec::new(0, 1);
                    },
                    Input::DownLeft => {
                        next_movement = CoordVec::new(-1, 1);
                    },
                    Input::Dig => {
                        // Dig a hole
                        // TODO: work out geology in more detail; maybe you shouldn't be able to dig anything
                        if state.positions.terrain[position].is_none() && energy.try_consume(5.0) {
                            buffer.spawn((
                                Render('_'),
                                Obstruction { entry_multiplier: 5.0, exit_multiplier: 5.0, obstruction_count: 0 },
                                // TODO: do this more cleanly
                                DespawnOnTick(state.ticks.wrapping_add(StochasticNumber::triangle_from_min_range(5000.0, 5000.0).sample(&mut rng).round() as u64)),
                                Position::single_tile(position, MapLayer::Terrain),
                                NewlyAdded
                            ));
                            inventory.add(Item::Dirt, StochasticNumber::triangle_from_min_range(1.0, 3.0).sample_rounded(&mut rng));
                        }
                    }
                },
                Err(e) => return Err(e.into())
            }
        }

        let target = position + next_movement;
        if state.map.get_terrain(target).entry_cost().is_some() && target != position {
            buffer.insert_one(client.entity, MovingInto(target));
        }
    }

    buffer.run_on(&mut state.world);

    let mut about_to_move = Vec::new();
    // Process motion and attacks
    for (entity, (current_pos, MovingInto(target_pos), damage, mut energy, move_cost, despawn_on_impact)) in state.world.query::<(&Position, &MovingInto, Option<&mut Attack>, Option<&mut Energy>, Option<&MoveCost>, Option<&DespawnOnImpact>)>().iter() {
        let mut move_cost = move_cost.map(|x| x.0.sample(&mut rng)).unwrap_or(0.0);

        move_cost *= (hex_distance(*target_pos, current_pos.head()) as f32).powf(0.5);

        for tile in current_pos.iter_coords() {
            // TODO: perhaps large enemies should not be exponentially more vulnerable to environmental hazards
            if let Some(current_terrain) = &state.positions.terrain[tile] {
                if let Ok(obstruction) = state.world.get::<&Obstruction>(*current_terrain) {
                    move_cost *= obstruction.exit_multiplier;
                }
            }
        }

        // TODO: attacks into obstructions are still cheap; is this desirable?
        if let Some(target_terrain) = &state.positions.terrain[*target_pos] {
            if let Ok(mut obstruction) = state.world.get::<&mut Obstruction>(*target_terrain) {
                move_cost *= obstruction.entry_multiplier;
                obstruction.obstruction_count += 1;
            }
        }

        if let Some(entry_cost) = state.map.get_terrain(*target_pos).entry_cost() {
            move_cost += entry_cost as f32;
            let can_move = match &state.positions.entities[*target_pos] {
                Some(target_entity) => {
                    let target_entity = target_entity.clone();
                    if let Ok(mut health) = state.world.get::<&mut Health>(target_entity) {
                        match damage {
                            Some(Attack { damage, energy: energy_cost, hits, .. }) => {
                                if consume_energy_if_available(&mut energy, *energy_cost) {
                                    let sampled_damage = damage.sample(&mut rng);
                                    health.apply(-sampled_damage);
                                    *hits += 1;
                                }
                            },
                            _ => ()
                        }
                        if despawn_on_impact.is_some() {
                            kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, entity, Some(target_entity));
                        }
                        if health.current < 0.0 {
                            // TODO: this may be totally broken
                            if state.world.get::<&ShrinkOnDeath>(target_entity).is_ok() {
                                let mut positions = state.world.get::<&mut Position>(target_entity).unwrap();

                                if positions.remove_coord(*target_pos, &mut state.positions, target_entity) {
                                    std::mem::drop(positions);
                                    kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, target_entity, Some(entity));
                                } else {
                                    health.current = health.max; // reset health
                                }
                            } else {
                                kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, target_entity, Some(entity));
                                if let Some(Attack { kills, .. }) = damage {
                                    *kills += 1;
                                }
                            }
                            true // murdered to death; space is now open
                        } else {
                            false // still alive; cannot move there
                        }
                    } else {
                        false // if no health, cannot be destroyed
                    }
                },
                None => true // empty, can move
            };
            if can_move {
                about_to_move.push((entity, *target_pos, move_cost));
            }
        }
        buffer.remove_one::<MovingInto>(entity);
    }

    for (entity, target_pos, move_cost) in about_to_move.drain(..) {
        // TODO: perhaps this should be applied to attacks too?
        let mut energy = state.world.get::<&mut Energy>(entity).ok();
        let mut current_pos = state.world.get::<&mut Position>(entity).unwrap() ;
        if consume_energy_if_available(&mut energy, move_cost) {
            let tail_pos = current_pos.move_into(target_pos, &mut state.positions, entity);
            current_pos.remove_coord(tail_pos, &mut state.positions, entity);
        }
    }

    buffer.run_on(&mut state.world);

    for (_entity, energy) in state.world.query_mut::<&mut Energy>() {
        energy.current = (energy.current + energy.regeneration_rate).min(energy.burst);
    }

    // Process transient entities
    for (entity, tick) in state.world.query::<&DespawnOnTick>().iter() {
        if state.ticks == tick.0 {
            kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, entity, None);
        }
    }

    for (entity, DespawnRandomly(inv_rate)) in state.world.query::<&DespawnRandomly>().iter() {
        if rng.u64(0..*inv_rate) == 0 {
            kill(&mut buffer, &mut despawn_buffer, &state, &mut rng, entity, None);
        }
    }

    buffer.run_on(&mut state.world);

    // extremely O(n^2), bad, fix at some point
    for (entity, position) in despawn_buffer.drain(..) {
        for coord in position.iter_coords() {
            // TODO: fix
            if state.positions.particles[coord] == Some(entity) {
                state.positions.particles[coord] = None;
            }
            if state.positions.entities[coord] == Some(entity) {
                state.positions.entities[coord] = None;
            }
            if state.positions.terrain[coord] == Some(entity) {
                state.positions.terrain[coord] = None;
            }
        }
    }

    // Send views to clients
    // TODO: terrain layer below others
    for (_id, client) in state.clients.iter() {
        client.frames_tx.send(Frame::PlayerCount(state.clients.len())).await?;
        let mut nearby = vec![];
        if let Ok(pos) = state.world.get::<&Position>(client.entity) {
            let pos = pos.head();
            for offset in hex_circle(VIEW) {
                let pos = pos + offset;
                let mut rng = rng_from_hash(pos);

                if let Some(entity) = &state.positions.particles[pos].or(state.positions.entities[pos]) {
                    let render = state.world.get::<&Render>(*entity)?;
                    let health = if let Ok(h) = state.world.get::<&Health>(*entity) {
                        h.pct()
                    } else { 1.0 };
                    nearby.push((offset.x, offset.y, render.0, health));
                } else if let Some(entity) = &state.positions.terrain[pos] {
                    let render = state.world.get::<&Render>(*entity)?;
                    nearby.push((offset.x, offset.y, render.0, 1.0));
                } else if let Some(terrain) = state.map.get_terrain(pos).symbol() {
                    nearby.push((offset.x, offset.y, terrain, rng.f32() * 0.1 + 0.9));
                } else {
                    let bg = if rng.usize(0..10) == 0 { ',' } else { '.' };
                    nearby.push((offset.x, offset.y, bg, rng.f32() * 0.1 + 0.9))
                }
            }
            let health = state.world.get::<&Health>(client.entity)?.current;
            let inventory = state.world.get::<&Inventory>(client.entity)?.contents
                .iter().map(|(i, q)| (i.name().to_string(), i.description().to_string(), *q)).filter(|(_, _, q)| *q > 0).collect();
            client.frames_tx.send(Frame::Display { nearby, health, inventory }).await?;
        } else {
            client.frames_tx.send(Frame::Dead).await?;
        }
    }

    state.ticks = state.ticks.wrapping_add(1);

    state.rng = rng;
    Ok(())
}

lazy_static::lazy_static! {
    static ref IDENTS: Vec<char> = {
        let mut chars = vec![];
        for range in [
            'א'..='ת',
            'Ⓐ'..='ⓩ',
            '🨀'..='🨅'
        ] {
            chars.extend(range);
        }
        chars.extend("𝔸𝕒𝔹𝕓ℂ𝕔𝔻𝕕ⅅⅆ𝔼𝕖ⅇ𝔽𝕗𝔾𝕘ℍ𝕙𝕀𝕚ⅈ𝕁𝕛ⅉ𝕂𝕜𝕃𝕝𝕄𝕞ℕ𝕟𝕆𝕠ℙ𝕡ℚ𝕢ℝ𝕣𝕊𝕤𝕋𝕥𝕌𝕦𝕍𝕧𝕎𝕨𝕏𝕩𝕐𝕪ℤ𝕫ℾℽℿℼ⅀𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡🩊".chars());
        chars
    };
}

fn random_identifier(rng: &mut fastrand::Rng) -> char {
    IDENTS[rng.usize(0..IDENTS.len())]
}

fn add_new_player(state: &mut GameState) -> Result<Entity> {
    let pos = loop {
        let pos = Coord::origin() + sample_range_rng(state.map.radius() - 10, &mut state.rng);
        if state.map.get_terrain(pos).entry_cost().is_some() {
            break pos;
        }
    };
    Ok(state.world.spawn((
        Position::single_tile(pos, MapLayer::Entities),
        PlayerCharacter,
        Render(random_identifier(&mut state.rng)),
        Attack { damage: StochasticNumber::Triangle { min: 20.0, max: 60.0, mode: 20.0 }, energy: 5.0, kills: 0, hits: 0 },
        Health::new(128.0, 128.0),
        EnemyTarget {
            spawn_density: 0.01,
            spawn_range: 3..=10,
            spawn_rate_inv: 20,
            aggression_range: 5,
            spawn_count: 0
        },
        Energy { current: 0.0, regeneration_rate: 1.0, burst: 5.0, total_used: 0.0 },
        Inventory::empty(),
        NewlyAdded,
        BlocksEnemySpawn
    )))
}

async fn load_world() -> Result<worldgen::GeneratedWorld> {
    let data = tokio::fs::read("world.bin").await?;
    Ok(bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0)
}

async fn load_saved_game() -> Result<Option<SavedGame>> {
    match tokio::fs::read(SAVE_FILE).await {
        Ok(data) => Ok(Some(
            bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

async fn save_game(state: &GameState) -> Result<()> {
    let encoded = bincode::serde::encode_to_vec(SavedGame::from_state(state)?, bincode::config::standard())?;
    tokio::fs::write(SAVE_FILE, encoded).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "0.0.0.0:8011".to_string());
    let mut loaded_save = false;
    let state = if let Some(saved) = load_saved_game().await? {
        println!("Loaded game state from {}", SAVE_FILE);
        loaded_save = true;
        Arc::new(Mutex::new(saved.into_state()?))
    } else {
        let world = match load_world().await {
            Ok(world) => world,
            Err(e) => {
                println!("Failed to load world, generating new one: {:?}", e);
                let world = worldgen::generate_world();
                tokio::fs::write("world.bin", bincode::serde::encode_to_vec(&world, bincode::config::standard())?).await?;
                world
            }
        };

        let baseline_soil_nutrients = world.soil_nutrients.clone();
        let baseline_groundwater = world.groundwater.clone();
        let baseline_salt = world.salt.clone();
        let baseline_temperature = world.temperature.clone();
        let dynamic_soil_nutrients = Map::new(world.radius, 0.0);
        let dynamic_groundwater = Map::new(world.radius, 0.0);

        Arc::new(Mutex::new(GameState {
            world: World::new(),
            clients: Slab::new(),
            ticks: 0,
            rng: fastrand::Rng::with_seed(fastrand::u64(..)),
            positions: PositionIndex::new(world.radius),
            map: world,
            baseline_soil_nutrients,
            baseline_groundwater,
            baseline_salt,
            baseline_temperature,
            dynamic_soil_nutrients,
            dynamic_groundwater
        }))
    };

    if !loaded_save {
        let mut state = state.lock().await;
        let count = count_hexes(state.map.radius() / 5);
        let mut batch = Vec::with_capacity(count as usize);
        for (_distance, offset) in hex_range(state.map.radius() / 5) {
            batch.push((
                Position::single_tile(Coord::origin() + offset * 5, MapLayer::Entities),
                Render('+'),
                Health::new(10.0, 10.0),
                //ShrinkOnDeath,
                Plant { genome: plant::Genome::random(), current_size: 0.1, nutrients_consumed: 0.0, nutrients_added: 0.0, water_consumed: 0.0, growth_ticks: 0 },
                NewlyAdded
            ));
        }
        state.world.spawn_batch(batch);
    }

    let try_socket = TcpListener::bind(&addr).await;
    let listener = try_socket.expect("Failed to bind");
    println!("Listening on: {}", addr);

    let state_ = state.clone();
    tokio::spawn(async move {
        let state = state_.clone();
        let mut interval = tokio::time::interval(Duration::from_millis(56));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let mut state = state.lock().await;
            let time = std::time::Instant::now();
            if let Err(e) = game_tick(&mut state).await {
                println!("Tick failed: {:?}", e);
            }
            if state.ticks % AUTOSAVE_INTERVAL_TICKS == 0 {
                if let Err(e) = save_game(&state).await {
                    println!("Autosave failed: {:?}", e);
                }
            }
            let tick_elapsed = time.elapsed();
            println!("Tick time: {:?}", tick_elapsed);
            interval.tick().await;
        }
    });

    let state_ = state.clone();
    while let Ok((stream, addr)) = listener.accept().await {
        let state_ = state_.clone();
        let (frames_tx, frames_rx) = mpsc::channel(10);
        let (inputs_tx, inputs_rx) = mpsc::channel(10);
        let (id, entity) = {
            let mut state = state_.lock().await;
            let entity = add_new_player(&mut state)?; // TODO
            let client = Client {
                inputs_rx,
                frames_tx,
                entity
            };
            let id = state.clients.insert(client);
            (id, entity)
        };

        tokio::spawn(async move {
            println!("conn result {:?}", handle_connection(stream, addr, frames_rx, inputs_tx).await);
            let mut state = state_.lock().await;
            state.clients.remove(id);
            let mut pos = match state.world.get::<&Position>(entity) {
                Err(_) => return,
                Ok(p) => {
                    (*p).clone()
                }
            };
            pos.record_for(&mut state.positions, None);
            let _ = state.world.despawn(entity);
        });
    }

    Ok(())
}
