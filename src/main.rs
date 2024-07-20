use hecs::{CommandBuffer, Entity, World};
use futures_util::{stream::TryStreamExt, SinkExt, StreamExt};
use indexmap::IndexMap;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio::sync::{mpsc, Mutex};
use anyhow::{Result, Context, anyhow};
use std::{collections::{hash_map::Entry, HashMap, HashSet, VecDeque}, convert::TryFrom, hash::{Hash, Hasher}, net::SocketAddr, sync::Arc, time::Duration};
use slab::Slab;
use serde::{Serialize, Deserialize};

pub mod worldgen;
pub mod map;

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
    Display { nearby: Vec<(i64, i64, char, f32)>, health: f32, inventory: Vec<(String, String, u64)> },
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
    map: worldgen::GeneratedWorld
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
struct PlayerCharacter;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Position(VecDeque<Coord>);

impl Position {
    fn head(&self) -> Coord {
        *self.0.front().unwrap()
    }

    fn single_tile(c: Coord) -> Self {
        Self(VecDeque::from([c]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MovingInto(Coord);

#[derive(Debug, Clone)]
struct Health(f32, f32);

impl Health {
    fn pct(&self) -> f32 {
        if self.1 == 0.0 { 0.0 }
        else { self.0 / self.1 }
    }
}

#[derive(Debug, Clone)]
struct Render(char);

#[derive(Debug, Clone)]
struct Attack { damage: StochasticNumber, energy: f32 }

#[derive(Debug, Clone)]
struct RangedAttack { damage: StochasticNumber, energy: f32, range: u64 }

#[derive(Debug, Clone)]
struct DespawnOnTick(u64);

#[derive(Debug, Clone)]
struct DespawnRandomly(u64);

#[derive(Debug, Clone)]
struct EnemyTarget { spawn_range: std::ops::RangeInclusive<i64>, spawn_density: f32, spawn_rate_inv: usize, aggression_range: i64 }

#[derive(Debug, Clone)]
struct Enemy;

#[derive(Debug, Clone)]
struct MoveCost(StochasticNumber);

#[derive(Debug, Clone)]
struct Collidable;

#[derive(Debug, Clone)]
struct Velocity(CoordVec);

#[derive(Debug, Clone)]
struct Terrain;

#[derive(Debug, Clone)]
struct Obstruction { entry_multiplier: f32, exit_multiplier: f32 }

#[derive(Debug, Clone)]
struct Energy { current: f32, regeneration_rate: f32, burst: f32 }

#[derive(Debug, Clone)]
struct Drops(Vec<(Item, StochasticNumber)>);

#[derive(Debug, Clone)]
struct Jump(i64);

impl Energy {
    fn try_consume(&mut self, cost: f32) -> bool {
        if self.current >= -1e-12 { // numerics
            self.current -= cost;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
struct DespawnOnImpact;

#[derive(Debug, Clone)]
struct Inventory(indexmap::IndexMap<Item, u64>);

impl Inventory {
    fn add(&mut self, item: Item, qty: u64) {
        *self.0.entry(item).or_default() += qty;
    }

    fn take(&mut self, item: Item, qty: u64) -> bool {
        match self.0.entry(item) {
            indexmap::map::Entry::Occupied(mut o) => {
                let current = o.get_mut();
                if *current >= qty {
                    *current -= qty;
                    return true;
                }
                return false;
            },
            indexmap::map::Entry::Vacant(_) => return false
        }
    }

    fn extend(&mut self, other: &Inventory) {
        for (item, count) in other.0.iter() {
            self.add(item.clone(), *count);
        }
    }

    fn is_empty(&self) -> bool {
        self.0.iter().any(|(_, c)| *c > 0)
    }

    fn empty() -> Self {
        Self(IndexMap::new())
    }
}

const VIEW: i64 = 15;
const RANDOM_DESPAWN_INV_RATE: u64 = 4000;

struct EnemySpec {
    symbol: char,
    min_damage: f32,
    damage_range: f32,
    initial_health: f32,
    move_delay: usize,
    attack_cooldown: u64,
    ranged: bool,
    movement: i64,
    drops: Vec<(Item, StochasticNumber)>
}

impl EnemySpec {
    // Numbers ported from original EWO. Fudge constants added elsewhere. 
    fn random() -> EnemySpec {
        match fastrand::usize(0..650) {
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

fn consume_energy_if_available(e: &mut Option<&mut Energy>, cost: f32) -> bool {
    e.is_none() || e.as_mut().unwrap().try_consume(cost)
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

fn triangle_distribution(min: f32, max: f32, mode: f32) -> f32 {
    let sample = fastrand::f32();
    let threshold = (mode - min) / (max - min);
    if sample < threshold {
        min + (sample * (max - min) * (mode - min)).sqrt()
    } else {
        max - ((1.0 - sample) * (max - min) * (max - mode)).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
enum StochasticNumber {
    Constant(f32),
    Triangle { min: f32, max: f32, mode: f32 }
}

impl StochasticNumber {
    fn sample(&self) -> f32 {
        match self {
            StochasticNumber::Constant(x) => *x,
            StochasticNumber::Triangle { min, max, mode } => triangle_distribution(*min, *max, *mode)
        }
    }

    fn sample_rounded<T: TryFrom<i128>>(&self) -> T {
        T::try_from(self.sample().round() as i128).map_err(|_| "convert fail").unwrap()
    }

    fn triangle_from_min_range(min: f32, range: f32) -> Self {
        StochasticNumber::Triangle { min: min, max: min + range, mode: (min + range) / 2.0 }
    }
}

async fn game_tick(state: &mut GameState) -> Result<()> {
    let mut terrain_positions = HashMap::new();
    let mut positions = HashMap::new();

    for (entity, pos) in state.world.query_mut::<hecs::With<&Position, &Collidable>>() {
        for subpos in pos.0.iter() {
            positions.insert(*subpos, entity);
        }
    }

    for (entity, pos) in state.world.query_mut::<hecs::With<&Position, &Terrain>>() {
        for subpos in pos.0.iter() {
            terrain_positions.insert(*subpos, entity);
        }
    }

    let mut buffer = hecs::CommandBuffer::new();

    // Spawn enemies
    for (_entity, (pos, EnemyTarget { spawn_range, spawn_density, spawn_rate_inv, .. })) in state.world.query::<(&Position, &EnemyTarget)>().iter() {
        let pos = pos.head();
        if fastrand::usize(0..*spawn_rate_inv) == 0 {
            let c = count_hexes(*spawn_range.end());
            let mut newpos = pos + sample_range(*spawn_range.end());
            let mut occupied = false;
            for _ in 0..(c as f32 / spawn_density * 0.005).ceil() as usize {
                if positions.contains_key(&newpos) {
                    occupied = true;
                    break;
                }
                newpos = pos + sample_range(*spawn_range.end());
            }
            if !occupied && state.map.get_terrain(newpos).entry_cost().is_some() && hex_distance(newpos, pos) >= *spawn_range.start() {
                let mut spec = EnemySpec::random();
                spec.drops.push((Item::Bone, StochasticNumber::Triangle { min: 0.7 * spec.initial_health / 40.0, max: 1.3 * spec.initial_health / 40.0, mode: spec.initial_health / 40.0 }));
                if spec.ranged {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health(spec.initial_health, spec.initial_health),
                        Enemy,
                        RangedAttack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32, range: 4 },
                        Position::single_tile(newpos),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        Collidable,
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0 },
                        Drops(spec.drops),
                        Jump(spec.movement)
                    ));
                } else {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health(spec.initial_health, spec.initial_health),
                        Enemy,
                        Attack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32 },
                        Position::single_tile(newpos),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        Collidable,
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0 },
                        Drops(spec.drops),
                        Jump(spec.movement)
                    ));
                }
            }
        }
    }

    // Process enemy motion and ranged attacks
    for (entity, (pos, ranged, energy, jump)) in state.world.query::<hecs::With<(&Position, Option<&mut RangedAttack>, Option<&mut Energy>, Option<&Jump>), &Enemy>>().iter() {
        let pos = pos.head();

        for direction in DIRECTIONS.iter() {
            if let Some(target) = positions.get(&(pos + *direction)) {
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
                    (hex_distance(pos + **dir, target_pos) - (ranged_attack.range as i64 - 1)).abs()).unwrap();
                buffer.insert_one(entity, MovingInto(pos + *direction));
                // do ranged attack if valid
                let atk_dir = target_pos - pos;
                if on_axis(atk_dir) && (energy.is_none() || energy.unwrap().try_consume(ranged_attack.energy)) {
                    let atk_dir = atk_dir.clamp(-CoordVec::one(), CoordVec::one());
                    buffer.spawn((
                        Render('*'),
                        Enemy,
                        Attack { damage: ranged_attack.damage, energy: 0.0 },
                        Velocity(atk_dir),
                        Position::single_tile(pos),
                        DespawnOnTick(state.ticks.wrapping_add(ranged_attack.range))
                    ));
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
            buffer.insert_one(entity, MovingInto(pos + *fastrand::choice(DIRECTIONS).unwrap()));
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
        let position = state.world.get::<&Position>(client.entity)?.head();
        let mut energy = state.world.get::<&mut Energy>(client.entity)?;
        let mut inventory = state.world.get::<&mut Inventory>(client.entity)?;
        loop {
            let recv = client.inputs_rx.try_recv();
            match recv {
                Err(e) if e == mpsc::error::TryRecvError::Empty => break,
                Ok(input) => match input {
                    Input::UpLeft => next_movement = CoordVec::new(0, -1),
                    Input::UpRight => next_movement = CoordVec::new(1, -1),
                    Input::Left => next_movement = CoordVec::new(-1, 0),
                    Input::Right => next_movement = CoordVec::new(1, 0),
                    Input::DownRight => next_movement = CoordVec::new(0, 1),
                    Input::DownLeft => next_movement = CoordVec::new(-1, 1),
                    Input::Dig => {
                        if terrain_positions.get(&position).is_none() && energy.try_consume(5.0) {
                            buffer.spawn((
                                Terrain,
                                Render('_'),
                                Obstruction { entry_multiplier: 5.0, exit_multiplier: 5.0 },
                                DespawnOnTick(state.ticks.wrapping_add(StochasticNumber::triangle_from_min_range(5000.0, 5000.0).sample().round() as u64)),
                                Position::single_tile(position)
                            ));
                            inventory.add(Item::Dirt, StochasticNumber::triangle_from_min_range(1.0, 3.0).sample_rounded());
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

    let mut despawn_buffer = HashSet::new();

    // This might lead to a duping glitch, which would at least be funny.
    // TODO: Players should drop items on disconnect.
    let kill = |buffer: &mut CommandBuffer, despawn_buffer: &mut HashSet<Entity>, state: &GameState, entity: Entity, killer: Option<Entity>, position: Option<Coord>| {
        let position = position.unwrap_or_else(|| state.world.get::<&Position>(entity).unwrap().head());
        despawn_buffer.insert(entity);
        buffer.despawn(entity);
        let mut materialized_drops = Inventory::empty();
        if let Ok(drops) = state.world.get::<&Drops>(entity) {
            for (drop, frequency) in drops.0.iter() {
                materialized_drops.add(drop.clone(), frequency.sample_rounded())
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
                Position::single_tile(position),
                Render('☒'),
                materialized_drops
            ));
        }
    };

    // Process motion and attacks
    for (entity, (current_pos, MovingInto(target_pos), damage, mut energy, move_cost, despawn_on_impact)) in state.world.query::<(&mut Position, &MovingInto, Option<&mut Attack>, Option<&mut Energy>, Option<&MoveCost>, Option<&DespawnOnImpact>)>().iter() {
        let mut move_cost = move_cost.map(|x| x.0.sample()).unwrap_or(0.0);

        move_cost *= (hex_distance(*target_pos, current_pos.head()) as f32).powf(0.5);
        
        for tile in current_pos.0.iter() {
            // TODO: perhaps large enemies should not be exponentially more vulnerable to environmental hazards
            if let Some(current_terrain) = terrain_positions.get(tile) {
                if let Ok(obstruction) = state.world.get::<&Obstruction>(*current_terrain) {
                    move_cost *= obstruction.exit_multiplier;
                }
            }
        }

        // TODO: attacks into obstructions are still cheap; is this desirable?
        if let Some(target_terrain) = terrain_positions.get(target_pos) {
            if let Ok(obstruction) = state.world.get::<&Obstruction>(*target_terrain) {
                move_cost *= obstruction.entry_multiplier;
            }
        }

        if let Some(entry_cost) = state.map.get_terrain(*target_pos).entry_cost() {
            move_cost += entry_cost as f32;
            let entry = match positions.entry(*target_pos) {
                Entry::Occupied(o) => {
                    let target_entity = *o.get();
                    if let Ok(mut x) = state.world.get::<&mut Health>(target_entity) {
                        match damage {
                            Some(Attack { damage, energy: energy_cost }) => {
                                if consume_energy_if_available(&mut energy, *energy_cost) {
                                    x.0 -= damage.sample();
                                }
                            },
                            _ => ()
                        }
                        if despawn_on_impact.is_some() {
                            kill(&mut buffer, &mut despawn_buffer, &state, entity, Some(target_entity), Some(*target_pos));
                        }
                        if x.0 < 0.0 {
                            kill(&mut buffer, &mut despawn_buffer, &state, target_entity, Some(entity), Some(*target_pos));
                            Some(Entry::Occupied(o))
                        } else {
                            None
                        }
                    } else {
                        None // no "on pickup" exists; emulated with health 0
                    }
                },
                Entry::Vacant(v) => Some(Entry::Vacant(v))
            };
            if let Some(entry) = entry {
                // TODO: perhaps this should be applied to attacks too?
                if consume_energy_if_available(&mut energy, move_cost) {
                    *entry.or_insert(entity) = entity;
                    positions.remove(&current_pos.0.pop_back().unwrap());
                    current_pos.0.push_front(*target_pos);
                }
            }
        }
        buffer.remove_one::<MovingInto>(entity);
    }

    buffer.run_on(&mut state.world);

    for (_entity, energy) in state.world.query_mut::<&mut Energy>() {
        energy.current = (energy.current + energy.regeneration_rate).min(energy.burst);
    }

    // Process transient entities
    for (entity, tick) in state.world.query::<&DespawnOnTick>().iter() {
        if state.ticks == tick.0 {
            kill(&mut buffer, &mut despawn_buffer, &state, entity, None, None);
        }
    }

    for (entity, DespawnRandomly(inv_rate)) in state.world.query::<&DespawnRandomly>().iter() {
        if fastrand::u64(0..*inv_rate) == 0 {
            kill(&mut buffer, &mut despawn_buffer, &state, entity, None, None);
        }
    }

    buffer.run_on(&mut state.world);

    let mut delete = vec![];
    for (position, entity) in positions.iter() {
        if despawn_buffer.contains(entity) {
            delete.push(*position);
        }
    }
    for position in delete {
        positions.remove(&position);
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

                if let Some(entity) = positions.get(&pos) {
                    let render = state.world.get::<&Render>(*entity)?;
                    let health = if let Ok(h) = state.world.get::<&Health>(*entity) {
                        h.pct()
                    } else { 1.0 };
                    nearby.push((offset.x, offset.y, render.0, health));
                } else if let Some(entity) = terrain_positions.get(&pos) {
                    let render = state.world.get::<&Render>(*entity)?;
                    nearby.push((offset.x, offset.y, render.0, 1.0));
                } else if let Some(terrain) = state.map.get_terrain(pos).symbol() {
                    nearby.push((offset.x, offset.y, terrain, rng.f32() * 0.1 + 0.9));
                } else {
                    let bg = if rng.usize(0..10) == 0 { ',' } else { '.' };
                    nearby.push((offset.x, offset.y, bg, rng.f32() * 0.1 + 0.9))
                }
            }
            let health = state.world.get::<&Health>(client.entity)?.0;
            let inventory = state.world.get::<&Inventory>(client.entity)?.0
                .iter().map(|(i, q)| (i.name().to_string(), i.description().to_string(), *q)).filter(|(_, _, q)| *q > 0).collect();
            client.frames_tx.send(Frame::Display { nearby, health, inventory }).await?;
        } else {
            client.frames_tx.send(Frame::Dead).await?;
        }
    }

    state.ticks = state.ticks.wrapping_add(1);

    Ok(())
}

lazy_static::lazy_static! {
    static ref IDENTS: Vec<char> = {
        let mut chars = vec![];
        for range in [
            'Α'..='ω',
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

fn random_identifier() -> char {
    *fastrand::choice(IDENTS.iter()).unwrap()
}

fn add_new_player(state: &mut GameState) -> Result<Entity> {
    let pos = loop {
        let pos = Coord::origin() + sample_range(state.map.radius() - 10);
        if state.map.get_terrain(pos).entry_cost().is_some() {
            break pos;
        }
    };
    Ok(state.world.spawn((
        Position::single_tile(pos),
        PlayerCharacter,
        Render(random_identifier()),
        Collidable,
        Attack { damage: StochasticNumber::Triangle { min: 20.0, max: 60.0, mode: 20.0 }, energy: 5.0 },
        Health(128.0, 128.0),
        EnemyTarget {
            spawn_density: 0.01,
            spawn_range: 3..=10,
            spawn_rate_inv: 20,
            aggression_range: 5
        },
        Energy { current: 0.0, regeneration_rate: 1.0, burst: 5.0 },
        Inventory::empty()
    )))
}

async fn load_world() -> Result<worldgen::GeneratedWorld> {
    let data = tokio::fs::read("world.bin").await?;
    Ok(bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0)
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "0.0.0.0:8080".to_string());

    let world = match load_world().await {
        Ok(world) => world,
        Err(e) => {
            println!("Failed to load world, generating new one: {:?}", e);
            let world = worldgen::generate_world();
            tokio::fs::write("world.bin", bincode::serde::encode_to_vec(&world, bincode::config::standard())?).await?;
            world
        }
    };

    let state = Arc::new(Mutex::new(GameState {
        world: World::new(),
        clients: Slab::new(),
        ticks: 0,
        map: world
    }));

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
            if let Err(e) = game_tick(&mut state).await {
                println!("{:?}", e);
            }
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
            let _ = state.world.despawn(entity);
        });
    }

    Ok(())
}