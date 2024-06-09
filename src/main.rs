use hecs::{Entity, World};
use euclid::{Point3D, Point2D, Vector2D};
use futures_util::{stream::TryStreamExt, SinkExt, StreamExt};
use noise_functions::Sample3;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio::sync::{mpsc, Mutex};
use anyhow::{Result, Context, anyhow};
use std::{collections::{hash_map::Entry, HashMap}, hash::{Hash, Hasher}, net::SocketAddr, sync::Arc, thread::current, time::Duration};
use slab::Slab;
use serde::{Serialize, Deserialize};

struct AxialWorldSpace;
struct CubicWorldSpace;
type Coord = Point2D<i64, AxialWorldSpace>;
type CubicCoord = Point3D<i64, CubicWorldSpace>;
type CoordVec = Vector2D<i64, AxialWorldSpace>;

fn to_cubic(p0: Coord) -> CubicCoord {
    CubicCoord::new(p0.x, p0.y, -p0.x - p0.y)
}

fn hex_distance(p0: Coord, p1: Coord) -> i64 {
    let ax_dist = p0 - p1;
    (ax_dist.x.abs() + ax_dist.y.abs() + (ax_dist.x + ax_dist.y).abs()) / 2
}

fn on_axis(p: CoordVec) -> bool {
    let p = to_cubic(Coord::origin() + p);
    let mut zero_ax = 0;
    if p.x == 0 { zero_ax += 1 }
    if p.y == 0 { zero_ax += 1 }
    if p.z == 0 { zero_ax += 1 }
    zero_ax >= 1
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
    Display { nearby: Vec<(i64, i64, char, f32)>, health: f32 },
    PlayerCount(usize)
}

struct Client {
    inputs_rx: mpsc::Receiver<Input>,
    frames_tx: mpsc::Sender<Frame>,
    entity: Entity
}

struct GameState {
    world: World,
    clients: Slab<Client>,
    ticks: u64
}

#[derive(Debug, Clone)]
enum Item {

}

#[derive(Debug, Clone)]
struct PlayerCharacter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Position(Coord);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MovingInto(Coord);

#[derive(Debug, Clone)]
struct Health(f32, f32);

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
struct DeferredRandomly<T: Clone + std::fmt::Debug + hecs::Bundle>(u64, T);

#[derive(Debug, Clone)]
struct Terrain;

#[derive(Debug, Clone)]
struct Obstruction { entry_cost: StochasticNumber, exit_cost: StochasticNumber }

#[derive(Debug, Clone)]
struct Energy { current: f32, regeneration_rate: f32, burst: f32 }

#[derive(Debug, Clone)]
struct DespawnOnImpact;

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

const VIEW: i64 = 15;
const WALL: i64 = 128;
const RANDOM_DESPAWN_INV_RATE: u64 = 4000;
const DIRECTIONS: &[CoordVec] = &[CoordVec::new(0, -1), CoordVec::new(1, -1), CoordVec::new(-1, 0), CoordVec::new(1, 0), CoordVec::new(0, 1), CoordVec::new(-1, 1)];

#[derive(Debug, Clone, PartialEq, Eq)]
enum BaseTerrain {
    Empty,
    Occupied,
    VeryOccupied
}

impl BaseTerrain {
    fn can_enter(&self) -> bool {
        *self == BaseTerrain::Empty
    }

    fn symbol(&self) -> Option<char> {
        match *self {
            Self::Empty => None,
            Self::Occupied => Some('#'),
            Self::VeryOccupied => Some('█')
        }
    }
}

const NOISE_SCALE: f32 = 0.05;

fn get_base_terrain(pos: Coord) -> BaseTerrain {
    let distance = hex_distance(pos, Coord::origin());
    if distance >= (WALL + 12) {
        return BaseTerrain::VeryOccupied
    }
    if distance >= WALL {
        return BaseTerrain::Occupied
    }
    let pos = to_cubic(pos);
    let noise = noise_functions::CellDistance.ridged(2, 1.00, 0.20).seed(406).sample3([pos.x as f32 * NOISE_SCALE, pos.y as f32 * NOISE_SCALE, pos.z as f32 * NOISE_SCALE]);
    if noise >= 0.3 {
        return BaseTerrain::VeryOccupied
    }
    if noise >= 0.2 {
        return BaseTerrain::Occupied
    }
    return BaseTerrain::Empty
}

fn sample_range(range: i64) -> CoordVec {
    let q = fastrand::i64(-range..=range);
    let r = fastrand::i64((-range).max(-q-range)..=range.min(-q+range));
    CoordVec::new(q, r)
}

struct EnemySpec {
    symbol: char,
    min_damage: f32,
    damage_range: f32,
    initial_health: f32,
    move_delay: usize,
    attack_cooldown: u64,
    ranged: bool
}

impl EnemySpec {
    // Numbers ported from original EWO. Fudge constants added elsewhere. 
    fn random() -> EnemySpec {
        match fastrand::usize(0..650) {
            0..=99 => EnemySpec { symbol: 'I', min_damage: 10.0, damage_range: 5.0, initial_health: 50.0, move_delay: 70, attack_cooldown: 10, ranged: false }, // IBIS
            100..=199 => EnemySpec { symbol: 'K', min_damage: 5.0, damage_range: 15.0, initial_health: 30.0, move_delay: 40, attack_cooldown: 10, ranged: false }, // KESTREL
            200..=299 => EnemySpec { symbol: 'S', min_damage: 5.0, damage_range: 5.0, initial_health: 20.0, move_delay: 50, attack_cooldown: 10, ranged: false }, // SNAKE
            300..=399 => EnemySpec { symbol: 'E', min_damage: 10.0, damage_range: 20.0, initial_health: 80.0, move_delay: 80, attack_cooldown: 10, ranged: false }, // EMU
            400..=499 => EnemySpec { symbol: 'O', min_damage: 8.0, damage_range: 17.0, initial_health: 150.0, move_delay: 100, attack_cooldown: 10, ranged: false }, // OGRE
            500..=599 => EnemySpec { symbol: 'R', min_damage: 5.0, damage_range: 5.0, initial_health: 15.0, move_delay: 40, attack_cooldown: 10, ranged: false }, // RAT
            600..=609 => EnemySpec { symbol: 'M' , min_damage: 20.0, damage_range: 10.0, initial_health: 150.0, move_delay: 70, attack_cooldown: 10, ranged: false }, // MOA
            610..=649 => EnemySpec { symbol: 'P', min_damage: 10.0, damage_range: 5.0, initial_health: 15.0, move_delay: 20, attack_cooldown: 10, ranged: true }, // PLATYPUS
            _ => unreachable!()
        }
    }
}

fn count_hexes(x: i64) -> i64 {
    x*(x+1)*3+1
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

    fn triangle_from_min_range(min: f32, range: f32) -> Self {
        StochasticNumber::Triangle { min: min, max: min + range, mode: (min + range) / 2.0 }
    }
}

async fn game_tick(state: &mut GameState) -> Result<()> {
    let mut terrain_positions = HashMap::new();
    let mut positions = HashMap::new();

    for (entity, pos) in state.world.query_mut::<hecs::With<&Position, &Collidable>>() {
        positions.insert(pos.0, entity);
    }

    for (entity, pos) in state.world.query_mut::<hecs::With<&Position, &Terrain>>() {
        terrain_positions.insert(pos.0, entity);
    }

    let mut buffer = hecs::CommandBuffer::new();

    // Spawn enemies
    for (_entity, (Position(pos), EnemyTarget { spawn_range, spawn_density, spawn_rate_inv, .. })) in state.world.query::<(&Position, &EnemyTarget)>().iter() {
        if fastrand::usize(0..*spawn_rate_inv) == 0 {
            let c = count_hexes(*spawn_range.end());
            let mut newpos = *pos + sample_range(*spawn_range.end());
            let mut occupied = false;
            for _ in 0..(c as f32 / spawn_density * 0.005).ceil() as usize {
                if positions.contains_key(&newpos) {
                    occupied = true;
                    break;
                }
                newpos = *pos + sample_range(*spawn_range.end());
            }
            if !occupied && get_base_terrain(newpos).can_enter() && hex_distance(newpos, *pos) >= *spawn_range.start() {
                let spec = EnemySpec::random();
                if spec.ranged {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health(spec.initial_health, spec.initial_health),
                        Enemy,
                        RangedAttack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32, range: 4 },
                        Position(newpos),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        Collidable,
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0 }
                    ));
                } else {
                    buffer.spawn((
                        Render(spec.symbol),
                        Health(spec.initial_health, spec.initial_health),
                        Enemy,
                        Attack { damage: StochasticNumber::triangle_from_min_range(spec.min_damage, spec.damage_range), energy: spec.attack_cooldown as f32 },
                        Position(newpos),
                        MoveCost(StochasticNumber::Triangle { min: 0.0, max: 2.0 * spec.move_delay as f32 / 3.0, mode: spec.move_delay as f32 / 3.0 }),
                        Collidable,
                        DespawnRandomly(RANDOM_DESPAWN_INV_RATE),
                        Energy { regeneration_rate: 1.0, current: 0.0, burst: 0.0 }
                    ));
                }
            }
        }
    }

    // Process enemy motion and ranged attacks
    for (entity, (Position(pos), ranged, energy)) in state.world.query::<hecs::With<(&Position, Option<&mut RangedAttack>, Option<&mut Energy>), &Enemy>>().iter() {
        for direction in DIRECTIONS.iter() {
            if let Some(target) = positions.get(&(*pos + *direction)) {
                if let Ok(_) = state.world.get::<&EnemyTarget>(*target) {
                    buffer.insert_one(entity, MovingInto(*pos + *direction));
                    continue;
                }
            }
        }

        let mut closest = None;

        // TODO we maybe need a spatial index for this
        for (_entity, (target_pos, target)) in state.world.query::<(&Position, &EnemyTarget)>().iter() {
            let distance = hex_distance(*pos, target_pos.0);
            if distance < target.aggression_range {
                match closest {
                    Some((_pos, old_distance)) if old_distance < distance => closest = Some((target_pos.0, distance)),
                    None => closest = Some((target_pos.0, distance)),
                    _ => ()
                }
            }
        }

        if let Some((target_pos, _)) = closest {
            if let Some(ranged_attack) = ranged {
                // slightly smart behaviour for ranged attacker: try to stay just within range
                let direction = DIRECTIONS.iter().min_by_key(|dir|
                    (hex_distance(*pos + **dir, target_pos) - (ranged_attack.range as i64 - 1)).abs()).unwrap();
                buffer.insert_one(entity, MovingInto(*pos + *direction));
                // do ranged attack if valid
                let atk_dir = target_pos - *pos;
                if on_axis(atk_dir) && (energy.is_none() || energy.unwrap().try_consume(ranged_attack.energy)) {
                    let atk_dir = atk_dir.clamp(-CoordVec::one(), CoordVec::one());
                    buffer.spawn((
                        Render('*'),
                        Enemy,
                        Attack { damage: ranged_attack.damage, energy: 0.0 },
                        Velocity(atk_dir),
                        Position(*pos),
                        DespawnOnTick(state.ticks.wrapping_add(ranged_attack.range))
                    ));
                }
            } else {
                let direction = DIRECTIONS.iter().min_by_key(|dir| hex_distance(*pos + **dir, target_pos)).unwrap();
                buffer.insert_one(entity, MovingInto(*pos + *direction));
            }
        } else {
            // wander randomly (ethical)
            buffer.insert_one(entity, MovingInto(*pos + *fastrand::choice(DIRECTIONS).unwrap()));
        }
    }

    // Process velocity
    for (entity, (Position(pos), Velocity(vel))) in state.world.query_mut::<(&Position, &Velocity)>() {
        buffer.insert_one(entity, MovingInto(*pos + *vel));
    }

    buffer.run_on(&mut state.world);

    // Process inputs
    for (_id, client) in state.clients.iter_mut() {
        let mut next_movement = CoordVec::zero();
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

                    }
                },
                Err(e) => return Err(e.into())
            }
        }
        let position = state.world.get::<&mut Position>(client.entity)?.0;
        let target = position + next_movement;
        if get_base_terrain(target).can_enter() && target != position {
            state.world.insert_one(client.entity, MovingInto(target)).unwrap();
        }
    }

    // Process motion and attacks
    for (entity, (Position(current_pos), MovingInto(target_pos), damage, mut energy, move_cost, despawn_on_impact)) in state.world.query::<(&mut Position, &MovingInto, Option<&mut Attack>, Option<&mut Energy>, Option<&MoveCost>, Option<&DespawnOnImpact>)>().iter() {
        let mut move_cost = move_cost.map(|x| x.0.sample()).unwrap_or(0.0);
        if let Some(current_terrain) = terrain_positions.get(current_pos) {
            move_cost += 1.0;
        }
        // TODO will break attacks kind of, desirable? Doubtful.
        if let Some(target_terrain) = terrain_positions.get(target_pos) {
            move_cost += 1.0;
        }

        if get_base_terrain(*target_pos).can_enter() {
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
                            buffer.despawn(entity);
                        }
                        if x.0 <= 0.0 {
                            buffer.despawn(target_entity);
                            Some(Entry::Occupied(o))
                        } else {
                            None
                        }
                    } else {
                        None // TODO: on pickup or something
                    }
                },
                Entry::Vacant(v) => Some(Entry::Vacant(v))
            };
            if let Some(entry) = entry {
                // TODO: perhaps this should be applied to attacks too?
                if consume_energy_if_available(&mut energy, move_cost) {
                    *entry.or_insert(entity) = entity;
                    positions.remove(current_pos);
                    *current_pos = *target_pos;
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
    for (entity, tick) in state.world.query_mut::<&DespawnOnTick>() {
        if state.ticks == tick.0 {
            buffer.despawn(entity);
        }
    }

    for (entity, DespawnRandomly(inv_rate)) in state.world.query_mut::<&DespawnRandomly>() {
        if fastrand::u64(0..*inv_rate) == 0 {
            buffer.despawn(entity);
        }
    }

    buffer.run_on(&mut state.world);

    // Send views to clients
    for (_id, client) in state.clients.iter() {
        client.frames_tx.send(Frame::PlayerCount(state.clients.len())).await?;
        let mut nearby = vec![];
        if let Ok(pos) = state.world.get::<&Position>(client.entity) {
            let pos = pos.0;
            for q in -VIEW..=VIEW {
                for r in (-VIEW).max(-q - VIEW)..= VIEW.min(-q+VIEW) {
                    let offset = CoordVec::new(q, r);
                    let pos = pos + offset;
                    if let Some(symbol) = get_base_terrain(pos).symbol() {
                        nearby.push((q, r, symbol, 1.0));
                    } else {
                        if let Some(entity) = positions.get(&pos) {
                            let render = state.world.get::<&Render>(*entity)?;
                            let health = if let Ok(h) = state.world.get::<&Health>(*entity) {
                                h.0 / h.1
                            } else { 1.0 };
                            nearby.push((q, r, render.0, health))
                        } else {
                            let mut rng = rng_from_hash(pos);
                            let bg = if rng.usize(0..10) == 0 { ',' } else { '.' };
                            nearby.push((q, r, bg, rng.f32() * 0.1 + 0.9))
                        }
                    }
                }
            }
            let health = state.world.get::<&Health>(client.entity)?.0;
            client.frames_tx.send(Frame::Display { nearby, health }).await?;
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
        let pos = Coord::origin() + sample_range(WALL - 10);
        if get_base_terrain(pos).can_enter() {
            break pos;
        }
    };
    Ok(state.world.spawn((
        Position(pos),
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
        }
    )))
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "0.0.0.0:8080".to_string());

    let state = Arc::new(Mutex::new(GameState {
        world: World::new(),
        clients: Slab::new(),
        ticks: 0
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