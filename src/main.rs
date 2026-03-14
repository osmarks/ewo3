#![feature(test)]
extern crate test;

use hecs::{CommandBuffer, Entity, With, World};
use futures_util::{stream::TryStreamExt, SinkExt, StreamExt};
use lazy_static::lazy_static;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio::sync::{mpsc, Mutex};
use anyhow::{Result, Context, anyhow};
use argh::FromArgs;
use std::{collections::HashSet, hash::{Hash, Hasher}, net::SocketAddr, ops::DerefMut, sync::Arc, time::Duration};
use slab::Slab;
use serde::{Serialize, Deserialize};
use smallvec::smallvec;

use ewo3::components::*;
use ewo3::map::*;
use ewo3::plant;
use ewo3::save::{SavedGame, GameMetrics};
use ewo3::world_serde;
use ewo3::worldgen;
use ewo3::util::config::*;

#[derive(FromArgs)]
/// Run the game server.
struct Args {
    /// websocket listen address
    #[argh(option, default = "String::from(\"0.0.0.0:8011\")")]
    listen_addr: String,
    /// simulation tick interval in milliseconds
    #[argh(option, default = "56")]
    tick_interval_ms: u64,
    /// make players invulnerable
    #[argh(switch)]
    players_invulnerable: bool,
    /// gamestate save file
    #[argh(option, default = "String::from(\"save.bin\")")]
    save_file: String,
    /// halt when tick fails
    #[argh(switch)]
    halt_on_error: bool,
}

lazy_static! {
    static ref ARGS: Args = argh::from_env();
}

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

fn rebuild_position_index(world: &World, radius: i32) -> PositionIndex {
    let mut index = PositionIndex::new(radius);
    for (entity, position) in world.query::<(Entity, &mut Position)>().iter() {
        position.record_for(&mut index, Some(entity));
    }
    index
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
    positions: PositionIndex,
    metrics: GameMetrics,
}

impl GameState {
    fn actual_groundwater(&self, pos: Coord) -> f32 {
        self.baseline_groundwater[pos] + self.dynamic_groundwater[pos]
    }

    fn actual_soil_nutrients(&self, pos: Coord) -> f32 {
        self.baseline_soil_nutrients[pos] + self.dynamic_soil_nutrients[pos]
    }
}

struct Buffer {
    cmd: CommandBuffer,
    despawn: Vec<Entity>,
}

impl Buffer {
    fn kill(&mut self, world: &World, entity: Entity, rng: &mut fastrand::Rng, killer: Option<Entity>) {
        let position = (*world.get::<&Position>(entity).unwrap()).clone();
        let position_head = position.head();

        self.despawn.push(entity);
        self.cmd.despawn(entity);

        let mut materialized_drops = Inventory::empty();
        if let Ok(drops) = world.get::<&Drops>(entity) {
            for (drop, frequency) in drops.0.iter() {
                materialized_drops.add(drop.clone(), frequency.sample_rounded(rng))
            }
        }
        if let Ok(other_inv) = world.get::<&Inventory>(entity) {
            materialized_drops.extend(&other_inv);
        }
        let killer_consumed_items = if let Some(killer) = killer {
            if let Ok(mut inv) = world.get::<&mut Inventory>(killer) {
                inv.extend(&materialized_drops);
                true
            } else {
                false
            }
        } else { false };
        if !killer_consumed_items && !materialized_drops.is_empty() {
            self.cmd.spawn((
                Position::single_tile(position_head, MapLayer::Entities),
                Render('☒'),
                materialized_drops,
                NewlyAdded,
                Health::new(10.0, 10.0)
            ));
        }
    }

    fn apply(&mut self, state: &mut GameState) {
        for entity in self.despawn.drain(..) {
            if let Ok(mut position) = state.world.get::<&mut Position>(entity) {
                position.record_for_erase(&mut state.positions, entity);
            }
        }

        self.cmd.run_on(&mut state.world);
    }

    fn new() -> Self {
        Buffer {
            despawn: Vec::new(),
            cmd: CommandBuffer::new()
        }
    }
}

fn saved_game_from_state(state: &GameState) -> Result<SavedGame> {
    Ok(SavedGame {
        ticks: state.ticks,
        rng_seed: state.rng.get_seed(),
        map: state.map.clone(),
        dynamic_soil_nutrients: state.dynamic_soil_nutrients.clone(),
        dynamic_groundwater: state.dynamic_groundwater.clone(),
        world: world_serde::serialize_world_to_bytes(&state.world)?,
        metrics: state.metrics.clone()
    })
}

fn game_state_from_saved(saved: SavedGame) -> Result<GameState> {
    let world = world_serde::deserialize_world_from_bytes(&saved.world)?;
    let positions = rebuild_position_index(&world, saved.map.radius);
    let baseline_soil_nutrients = saved.map.soil_nutrients.clone();
    let baseline_groundwater = saved.map.groundwater.clone();
    let baseline_salt = saved.map.salt.clone();
    let baseline_temperature = saved.map.temperature.clone();
    Ok(GameState {
        world,
        clients: Slab::new(),
        ticks: saved.ticks,
        rng: fastrand::Rng::with_seed(saved.rng_seed),
        map: saved.map,
        baseline_soil_nutrients,
        baseline_groundwater,
        baseline_salt,
        baseline_temperature,
        dynamic_soil_nutrients: saved.dynamic_soil_nutrients,
        dynamic_groundwater: saved.dynamic_groundwater,
        positions,
        metrics: saved.metrics,
    })
}

async fn game_tick(state: &mut GameState) -> Result<()> {
    let mut buffer = Buffer::new();

    let mut rng = fastrand::Rng::with_seed(state.rng.get_seed());
    for (entity, position) in state.world.query_mut::<With<(Entity, &mut Position), &NewlyAdded>>() {
        position.record_for(&mut state.positions, Some(entity));
        buffer.cmd.remove_one::<NewlyAdded>(entity);
        buffer.cmd.insert_one(entity, CreatedAt(state.ticks));
    }

    buffer.apply(state);

    if state.ticks % FIELD_DECAY_DELAY == 0 {
        state.dynamic_soil_nutrients.for_each_mut(|nutrients| *nutrients *= 0.9999);
    } else if state.ticks % FIELD_DECAY_DELAY == 1 {
        for (pos, water) in state.dynamic_groundwater.iter_mut() {
            *water *= 0.999;
            if state.map.water[pos] > 0.0 {
                *water *= 0.9; // reversion to baseline much faster in active water areas
            }
        }
    } else if state.ticks % FIELD_DECAY_DELAY == 2 {
        state.dynamic_soil_nutrients = smooth(&state.dynamic_soil_nutrients, 3);
    } else if state.ticks % FIELD_DECAY_DELAY == 3 {
        state.dynamic_groundwater = smooth(&state.dynamic_groundwater, 3);
    }

    // Spawn enemies
    for (_entity, pos, EnemyTarget { spawn_range, spawn_density, spawn_rate_inv, spawn_count, .. }) in state.world.query::<(Entity, &Position, &mut EnemyTarget)>().iter() {
        let pos = pos.head();
        if rng.usize(0..*spawn_rate_inv) == 0 {
            let c = count_hexes(*spawn_range.end());
            // TODO: generalize this kind of logic
            let mut newpos = pos + sample_range_rng(*spawn_range.end(), &mut rng);
            let mut occupied = false;
            for _ in 0..(c as f32 / *spawn_density * 0.005).ceil() as usize {
                if !state.positions.entities.in_range(newpos) {
                    continue;
                }
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
                spec.drops.push((Item::Fungible(FungibleItem::Bone), StochasticNumber::Triangle { min: 0.7 * spec.initial_health / 40.0, max: 1.3 * spec.initial_health / 40.0, mode: spec.initial_health / 40.0 }));
                if spec.ranged {
                    buffer.cmd.spawn((
                        Render(spec.symbol),
                        Health::new(spec.initial_health, spec.initial_health),
                        Enemy,
                        RangedAttack { damage: smallvec![(HealthChangeType::Magic, StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range))], energy: spec.attack_cooldown as f32, range: 4, firings: 0 },
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
                    buffer.cmd.spawn((
                        Render(spec.symbol),
                        Health::new(spec.initial_health, spec.initial_health),
                        Enemy,
                        Attack { damage: smallvec![(HealthChangeType::BluntForce, StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range))], energy: spec.attack_cooldown as f32, hits: 0, kills: 0 },
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
                state.metrics.enemies_spawned += 1;
            }
        }
    }

    buffer.apply(state);

    let mut plants_to_reproduce = vec![];

    // Run plant simulations.
    for (entity, pos, plant) in state.world.query::<(Entity, &Position, &mut Plant)>().iter() {
        if (entity.id() as u64) % PLANT_TICK_DELAY == state.ticks % PLANT_TICK_DELAY {
            let pos = pos.head();
            let water = state.actual_groundwater(pos);
            let soil_nutrients = state.actual_soil_nutrients(pos);
            let salt = state.baseline_salt[pos];
            let temperature = state.baseline_temperature[pos];
            let terrain = &state.map.get_terrain(pos);
            //println!("{:?} {} {} {} {} {}", plant.genome, water, soil_nutrients, salt, temperature, base_growth_rate);
            if plant.genome.base_growth_rate(soil_nutrients, water, temperature, salt, terrain) < PLANT_DIEOFF_THRESHOLD {
                if let Ok(mut health) = state.world.get::<&mut Health>(entity) {
                    health.apply(HealthChangeType::Starvation, -PLANT_DIEOFF_RATE);
                    if health.current <= 0.0 {
                        buffer.kill(&state.world, entity, &mut rng, None);
                        state.metrics.plants_died_starvation += 1;
                        // return nutrients to soil upon death; TODO implement decomposition modelling
                        state.dynamic_soil_nutrients[pos] += plant.current_size * SOIL_NUTRIENT_DECOMP_RETURN_RATE;
                    }
                }
            }

            let original_size = plant.current_size;
            plant.current_size += plant.genome.base_growth_rate(soil_nutrients, water, temperature, salt, terrain) * PLANT_GROWTH_SCALE * plant.current_size.max(0.1).powf(-0.25); // allometric scaling law
            let difference = (plant.current_size - original_size).max(0.0);

            if plant.can_reproduce() {
                plant.ready_for_reproduce_ticks += 1;
                if rng.f32() < plant.genome.reproduction_rate() {
                    plants_to_reproduce.push(entity);
                    plant.current_size -= PLANT_REPRODUCTION_ATTEMPT_COST;
                }
            } else {
                plant.ready_for_reproduce_ticks = 0;
            }

            state.dynamic_soil_nutrients[pos] -= difference * SOIL_NUTRIENT_CONSUMPTION_RATE;
            plant.nutrients_consumed += difference * SOIL_NUTRIENT_CONSUMPTION_RATE;
            plant.total_growth += difference;
            if plant.current_size > plant.genome.mature_size() {
                state.dynamic_soil_nutrients[pos] += plant.genome.nutrient_addition_rate() * SOIL_NUTRIENT_FIXATION_RATE;
                plant.nutrients_added += plant.genome.nutrient_addition_rate() * SOIL_NUTRIENT_FIXATION_RATE;
            }
            // TODO: water consumption should depend on atmospheric humidity
            let water_consumed = (difference + PLANT_IDLE_WATER_CONSUMPTION_OFFSET) * WATER_CONSUMPTION_RATE * plant.genome.water_efficiency();
            state.dynamic_groundwater[pos] -= water_consumed;
            plant.water_consumed += water_consumed;

            if difference > 0.0 {
                plant.growth_ticks += 1;
                if let Ok(mut health) = state.world.get::<&mut Health>(entity) {
                    health.max += difference;
                    health.apply(HealthChangeType::NaturalRegeneration, difference * 2.0);
                }
            }

            plant.age += 1;

            if plant.age as f32 > plant.genome.lifespan() * PLANT_LIFESPAN_SCALE {
                // TODO refactor
                if let Ok(mut health) = state.world.get::<&mut Health>(entity) {
                    health.apply(HealthChangeType::Senescence, -PLANT_DIEOFF_RATE);
                    if health.current <= 0.0 {
                        buffer.kill(&state.world, entity, &mut rng, None);
                        state.metrics.plants_died_old_age += 1;
                        state.dynamic_soil_nutrients[pos] += plant.current_size * SOIL_NUTRIENT_DECOMP_RETURN_RATE;
                    }
                }
            }
        }
    }

    // TODO: Since we don't update the positions index immediately, plants might sometimes spawn on top of each other.
    for entity in plants_to_reproduce {
        let pos = state.world.get::<&Position>(entity)?.head();
        let own_genome = state.world.get::<&Plant>(entity)?.genome.clone();

        let count = count_hexes(PLANT_POLLINATION_RADIUS);
        let reproduction_tries = (count as f32 * PLANT_POLLINATION_SCAN_FRACTION).ceil() as usize;
        let mut maybe_other = None;
        for _ in 0..reproduction_tries {
            let newpos = pos + sample_range_rng(PLANT_POLLINATION_RADIUS, &mut rng);
            if !state.map.heightmap.in_range(newpos) || newpos == pos {
                continue;
            }
            if let Some(other) = state.positions.entities[newpos] {
                if let Ok(other_plant) = state.world.get::<&mut Plant>(other) {
                    if other_plant.can_reproduce() {
                        if let Some(_hybrid) = other_plant.genome.hybridize(&mut rng, &own_genome) {
                            maybe_other = Some(other);
                            break;
                        }
                    }
                }
            }
        }

        if let Some(other) = maybe_other {
            // TODO: multiple seeds from one plant with different hybridizations?
            let [plant, other_plant] = state.world.query_disjoint_mut::<&mut Plant, 2>([entity, other]);
            let plant = plant?;
            let other_plant = other_plant?;
            let hybrid_genome = other_plant.genome.hybridize(&mut rng, &own_genome).unwrap();

            for _ in 0..PLANT_SEEDING_ATTEMPTS {
                let newpos = pos + sample_range_rng(PLANT_SEEDING_RADIUS, &mut rng);
                if !state.map.heightmap.in_range(newpos) || state.positions.entities[newpos].is_some() || !hybrid_genome.terrain_valid(&state.map.get_terrain(newpos)) {
                    continue;
                }

                let child_size = (plant.genome.initial_size_scale() + other_plant.genome.initial_size_scale()) * 0.5;

                buffer.cmd.spawn((
                    Position::single_tile(newpos, MapLayer::Entities),
                    Render('+'),
                    // TODO: work out more reasonable health/size parameterization, or at least factor this out
                    Health::new(1.0 + (child_size - 1.0) * 2.0, 1.0 + (child_size - 1.0) * 2.0),
                    Plant::new(hybrid_genome.clone(), child_size),
                    NewlyAdded
                ));
                plant.children += 1;
                other_plant.children += 1;
                // TODO: can this go negative?
                // TODO: gendering?
                plant.current_size -= plant.genome.initial_size_scale() * 0.5;
                other_plant.current_size -= other_plant.genome.initial_size_scale() * 0.5;
                state.metrics.plants_reproduced += 1;
                break;
            }
        }
    }

    buffer.apply(state);

    // Process enemy motion and ranged attacks
    for (entity, pos, ranged, energy, jump) in state.world.query::<hecs::With<(Entity, &Position, Option<&mut RangedAttack>, Option<&mut Energy>, Option<&Jump>), &Enemy>>().iter() {
        let pos = pos.head();

        for direction in DIRECTIONS.iter() {
            if let Some(target) = &state.positions.entities[pos + *direction] {
                if let Ok(_) = state.world.get::<&EnemyTarget>(*target) {
                    buffer.cmd.insert_one(entity, MovingInto(pos + *direction));
                    continue;
                }
            }
        }

        let mut closest = None;

        // TODO we maybe need a spatial index for this
        for (target_pos, target) in state.world.query::<(&Position, &EnemyTarget)>().iter() {
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
                buffer.cmd.insert_one(entity, MovingInto(pos + *direction));
                // do ranged attack if valid
                let atk_dir = target_pos - pos;
                if on_axis(atk_dir) && (energy.is_none() || energy.unwrap().try_consume(ranged_attack.energy)) {
                    let atk_dir = atk_dir.clamp(-CoordVec::one(), CoordVec::one());
                    // TODO: for future metrics reasons, perhaps track source
                    buffer.cmd.spawn((
                        Render('*'),
                        Enemy,
                        Attack { damage: ranged_attack.damage.clone(), energy: 0.0, hits: 0, kills: 0 },
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
                buffer.cmd.insert_one(entity, MovingInto(pos + direction * best_scale));
            }
        } else {
            // wander randomly (ethical)
            let direction = DIRECTIONS[rng.usize(0..DIRECTIONS.len())];
            buffer.cmd.insert_one(entity, MovingInto(pos + direction));
        }
    }

    // Process velocity
    for (entity, pos, Velocity(vel)) in state.world.query_mut::<(Entity, &Position, &Velocity)>() {
        buffer.cmd.insert_one(entity, MovingInto(pos.head() + *vel));
    }

    buffer.apply(state);

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
                            buffer.cmd.spawn((
                                Render('_'),
                                Obstruction { entry_multiplier: 5.0, exit_multiplier: 5.0, obstruction_count: 0 },
                                // TODO: do this more cleanly
                                DespawnOnTick(state.ticks.wrapping_add(StochasticNumber::triangle_from_min_range(5000.0, 5000.0).sample(&mut rng).round() as u64)),
                                Position::single_tile(position, MapLayer::Terrain),
                                NewlyAdded
                            ));
                            inventory.add(Item::Fungible(FungibleItem::Dirt), StochasticNumber::triangle_from_min_range(1.0, 3.0).sample_rounded(&mut rng));
                        }
                    }
                },
                Err(e) => return Err(e.into())
            }
        }

        let target = position + next_movement;
        if state.map.get_terrain(target).entry_cost().is_some() && target != position {
            buffer.cmd.insert_one(client.entity, MovingInto(target));
        }
    }

    buffer.apply(state);

    let mut about_to_move = Vec::new();
    // Process motion and attacks
    for (entity, current_pos, MovingInto(target_pos), damage, mut energy, move_cost, despawn_on_impact) in state.world.query::<(Entity, &Position, &MovingInto, Option<&mut Attack>, Option<&mut Energy>, Option<&MoveCost>, Option<&DespawnOnImpact>)>().iter() {
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
                                    for (ty, cnt) in damage.iter() {
                                        let sampled_damage = cnt.sample(&mut rng);
                                        health.apply(*ty, -sampled_damage);
                                    }
                                    *hits += 1;
                                }
                            },
                            _ => ()
                        }
                        if despawn_on_impact.is_some() {
                            buffer.kill(&state.world, entity, &mut rng, Some(target_entity));
                        }
                        if health.current < 0.0 {
                            // TODO: this may be totally broken
                            if state.world.get::<&ShrinkOnDeath>(target_entity).is_ok() {
                                let mut positions = state.world.get::<&mut Position>(target_entity).unwrap();

                                if positions.remove_coord(*target_pos, &mut state.positions, target_entity) {
                                    std::mem::drop(positions);
                                    buffer.kill(&state.world, target_entity, &mut rng, Some(entity));
                                } else {
                                    health.current = health.max; // reset health
                                }
                            } else {
                                buffer.kill(&state.world, target_entity, &mut rng, Some(entity));
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
        buffer.cmd.remove_one::<MovingInto>(entity);
    }

    for (entity, target_pos, move_cost) in about_to_move {
        // TODO: perhaps this should be applied to attacks too?
        let mut energy = state.world.get::<&mut Energy>(entity).ok();
        let mut current_pos = state.world.get::<&mut Position>(entity).unwrap();
        if consume_energy_if_available(&mut energy, move_cost) {
            let tail_pos = current_pos.move_into(target_pos, &mut state.positions, entity);
            current_pos.remove_coord(tail_pos, &mut state.positions, entity);
        }
    }

    buffer.apply(state);

    for energy in state.world.query_mut::<&mut Energy>() {
        energy.current = (energy.current + energy.regeneration_rate).min(energy.burst);
    }

    // Process transient entities
    for (entity, tick) in state.world.query::<(Entity, &DespawnOnTick)>().iter() {
        if state.ticks == tick.0 {
            buffer.kill(&state.world, entity, &mut rng, None);
        }
    }

    for (entity, DespawnRandomly(inv_rate)) in state.world.query::<(Entity, &DespawnRandomly)>().iter() {
        if rng.u64(0..*inv_rate) == 0 {
            buffer.kill(&state.world, entity, &mut rng, None);
        }
    }

    buffer.apply(state);

    // Send views to clients
    for (_id, client) in state.clients.iter() {
        client.frames_tx.send(Frame::PlayerCount(state.clients.len())).await?;
        let mut nearby = vec![];
        if let Ok(pos) = state.world.get::<&Position>(client.entity) {
            let pos = pos.head();
            for offset in hex_circle(VIEW) {
                let pos = pos + offset;
                let mut rng = rng_from_hash(pos);

                if let Some(entity) = &state.positions.particles[pos].or(state.positions.entities[pos]).or(state.positions.terrain[pos]) {
                    let render = state.world.get::<&Render>(*entity)?;
                    let health = if let Ok(h) = state.world.get::<&Health>(*entity) {
                        h.pct()
                    } else { 1.0 };
                    nearby.push((offset.x, offset.y, render.0, health));
                } else if let Some(terrain) = state.map.get_terrain(pos).symbol() {
                    nearby.push((offset.x, offset.y, terrain, rng.f32() * 0.1 + 0.9));
                } else {
                    let bg = if rng.usize(0..10) == 0 { ',' } else { '.' };
                    nearby.push((offset.x, offset.y, bg, rng.f32() * 0.1 + 0.9))
                }
            }
            let health = state.world.get::<&Health>(client.entity)?.current;
            // TODO do this properly
            let inventory = state.world.get::<&Inventory>(client.entity)?.fungible
                .iter().map(|(i, q)| (Item::Fungible(i.clone()).name().to_string(), Item::Fungible(i.clone()).description().to_string(), *q)).filter(|(_, _, q)| *q > 0).collect();
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
        if state.map.get_terrain(pos).entry_cost().is_some() && !(state.positions.entities[pos].is_some() || state.positions.terrain[pos].is_some()) {
            break pos;
        }
    };
    Ok(state.world.spawn((
        Position::single_tile(pos, MapLayer::Entities),
        PlayerCharacter,
        Render(random_identifier(&mut state.rng)),
        Attack { damage: smallvec![(HealthChangeType::BluntForce, StochasticNumber::Triangle { min: 20.0, max: 60.0, mode: 20.0 })], energy: 5.0, kills: 0, hits: 0 },
        if ARGS.players_invulnerable { Health::invulnerable() } else { Health::new(128.0, 128.0) },
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
    match tokio::fs::read(&ARGS.save_file).await {
        Ok(data) => Ok(Some(SavedGame::decode(&data)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

async fn save_game(state: &GameState) -> Result<()> {
    let encoded = saved_game_from_state(state)?.encode()?;
    tokio::fs::write(&ARGS.save_file, encoded).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut loaded_save = false;
    let state = if let Some(saved) = load_saved_game().await? {
        println!("Loaded game state from {}", ARGS.save_file);
        loaded_save = true;
        Arc::new(Mutex::new(game_state_from_saved(saved)?))
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
            dynamic_groundwater,
            metrics: GameMetrics::new(),
        }))
    };

    if !loaded_save {
        let mut state = state.lock().await;
        let mut batch = Vec::with_capacity(INITIAL_PLANTS);
        let mut used = HashSet::new();
        while batch.len() < INITIAL_PLANTS {
            let genome = plant::Genome::random(&mut state.rng);
            let radius = state.map.radius();
            let pos = Coord::origin() + sample_range(&mut state.rng, radius);
            if genome.base_growth_rate(state.actual_soil_nutrients(pos), state.actual_groundwater(pos), state.baseline_temperature[pos], state.baseline_salt[pos], &state.map.get_terrain(pos)) > 0.2 && !used.contains(&pos) {
                let initial_size = genome.initial_size_scale();
                batch.push((
                    Position::single_tile(pos, MapLayer::Entities),
                    Render('+'),
                    Health::new(1.0 + (initial_size - 1.0) * 2.0, 1.0 + (initial_size - 1.0) * 2.0),
                    //ShrinkOnDeath,
                    Plant::new(genome, initial_size),
                    NewlyAdded
                ));
                used.insert(pos);
            }
        }
        state.world.spawn_batch(batch);
    }

    let try_socket = TcpListener::bind(&ARGS.listen_addr).await;
    let listener = try_socket.expect("Failed to bind");
    println!("Listening on: {}", ARGS.listen_addr);

    let state_ = state.clone();
    tokio::spawn(async move {
        let state = state_.clone();
        let mut interval = tokio::time::interval(Duration::from_millis(ARGS.tick_interval_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let mut state = state.lock().await;
            let time = std::time::Instant::now();
            if let Err(e) = game_tick(&mut state).await {
                println!("Tick failed: {:?}", e);
                if ARGS.halt_on_error {
                    std::process::exit(1);
                }
            }
            if state.ticks % AUTOSAVE_INTERVAL_TICKS == 0 {
                if let Err(e) = save_game(&state).await {
                    println!("Autosave failed: {:?}", e);
                }
            }
            let tick_elapsed = time.elapsed();
            println!("Tick time: {:?}", tick_elapsed);
            println!("{:?}", state.metrics);
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
