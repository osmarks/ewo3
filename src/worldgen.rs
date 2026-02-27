use std::{cmp::Ordering, collections::{hash_map::Entry, BinaryHeap, HashMap, HashSet, VecDeque}, hash::{Hash, Hasher}};

use noise_functions::Sample3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::map::*;

pub const WORLD_RADIUS: i32 = 1024;
const NOISE_SCALE: f32 = 0.0005;
const HEIGHT_EXPONENT: f32 = 0.3;
const WATER_SOURCES: usize = 40;

pub fn height_baseline_with_radius(pos: Coord, radius: i32) -> f32 {
    let w_frac = (hex_distance(pos, Coord::origin()) as f32 / radius as f32).powf(3.0);
    let pos = to_cubic(pos);
    let noise =
        noise_functions::OpenSimplex2s.ridged(6, 0.7, 1.55).seed(406).sample3([10.0 + pos.x as f32 * NOISE_SCALE, pos.y as f32 * NOISE_SCALE, pos.z as f32 * NOISE_SCALE]);
    let range = 1.0 - 2.0 * (w_frac - 0.5).powf(2.0);
    noise * range - w_frac
}

pub fn height_baseline(pos: Coord) -> f32 {
    height_baseline_with_radius(pos, WORLD_RADIUS)
}

fn percentilize<F: Fn(f32) -> f32>(raw: &mut Map<f32>, postprocess: F) {
    let mut xs: Vec<(Coord, f32)> = raw.iter().map(|(coord, value)| (coord, *value)).collect();
    xs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (j, (i, _x)) in xs.into_iter().enumerate() {
        let percentile = j as f32 / raw.size() as f32;
        raw[i] = postprocess(percentile);
    }
}

fn normalize<F: Fn(f32) -> f32 + Sync>(raw: &mut Map<f32>, postprocess: F) {
    let mut min = raw.iter_data().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let mut max = raw.iter_data().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    if min == max {
        min = 0.0;
        max = 1.0;
    }
    for (_c, x) in raw.iter_mut() {
        *x = postprocess((*x - min) / (max - min));
    }
}

pub fn generate_heights() -> Map<f32> {
    generate_heights_with_radius(WORLD_RADIUS)
}

pub fn generate_heights_with_radius(radius: i32) -> Map<f32> {
    let mut raw = Map::<f32>::from_fn(|pos| height_baseline_with_radius(pos, radius), radius);
    percentilize(&mut raw, |x| {
        let s = 1.0 - (1.0 - x).powf(HEIGHT_EXPONENT);
        s * 2.0 - 1.0
    });
    raw
}

pub fn generate_contours(field: &Map<f32>, interval: f32) -> Vec<(Coord, f32, f32, CoordVec)> {
    // Starting at the origin, we want to detect contour lines in any of the six directions.
    // Go in one of the perpendicular directions to generate base directions then scan until the edge is reached starting from any of those points.
    DIRECTIONS.par_iter()
        .map(|scan_direction| {
            let mut out = vec![];
            // "perpendicular" doesn't really work in axial coordinates, but this apparently does so whatever
            let slit_direction = rotate_60(*scan_direction);
            for x in 0..field.radius {
                let base = Coord::zero() + slit_direction * x;
                let mut last: Option<f32> = None;
                for y in 0..field.radius {
                    let point = base + *scan_direction * y;
                    if hex_distance(point, Coord::zero()) <= field.radius {
                        let sample = field[point] / interval;
                        if let Some(last) = last {
                            if last.trunc() != sample.trunc() {
                                out.push((point, last, sample, *scan_direction));
                            }
                        }
                        last = Some(sample);
                    }
                }
            }
            out
        })
        .reduce(Vec::new, |mut acc, mut chunk| {
            acc.append(&mut chunk);
            acc
        })
}

#[derive(Clone, Copy, Debug)]
struct PointWrapper<T: Hash + Eq + PartialEq>(f32, T);

impl<T: Hash + Eq + PartialEq> PartialEq for PointWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        assert!(self.0.is_finite() && other.0.is_finite());
        self.0 == other.0 && self.1 == other.1
    }
}

impl<T: Hash + Eq + PartialEq> Eq for PointWrapper<T> {}

fn hash_thing<T: Hash>(thing: &T) -> u64 {
    let mut hasher = seahash::SeaHasher::new();
    thing.hash(&mut hasher);
    hasher.finish()
}

// we only need a consistent ordering, even if it's not semantically meaningful
// reverse order so that we can pop the smallest element first
impl<T: Hash + Eq + PartialEq> Ord for PointWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap().then_with(|| hash_thing(&self.1).cmp(&hash_thing(&other.1)))
    }
}

impl<T: Hash + Eq + PartialEq> PartialOrd for PointWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn generate_separated_high_points(n: usize, sep: i32, map: &Map<f32>) -> Vec<Coord> {
    let mut points = vec![];
    let mut priority = BinaryHeap::with_capacity(map.size());
    for (coord, height) in map.iter() {
        priority.push(PointWrapper(-*height, coord));
    }
    while points.len() < n {
        let next = priority.pop().unwrap();
        if points.iter().all(|x| hex_distance(*x, next.1) > sep) {
            points.push(next.1)
        }
    }
    points
}

fn astar<C: PartialEq + Eq + Hash + Copy + std::fmt::Debug, F: FnMut(C) -> f32, G: FnMut(C) -> I, I: Iterator<Item=(C, f32)>, H: FnMut(C) -> bool>(start: C, mut is_end: H, mut heuristic: F, mut get_neighbors: G) -> Vec<C> {
    let mut frontier = BinaryHeap::new();
    frontier.push(PointWrapper(0.0, start));
    let mut came_from: HashMap<C, PointWrapper<C>> = HashMap::new();
    came_from.insert(start, PointWrapper(0.0, start));

    let mut end = None;
    while let Some(PointWrapper(_est_cost, current)) = frontier.pop() {
        if is_end(current) {
            end = Some(current);
            break
        }

        let cost = came_from[&current].0;
        for (neighbour, next_cost) in get_neighbors(current) {
            let new_cost = cost + next_cost;
            match came_from.entry(neighbour.clone()) {
                Entry::Occupied(mut o) => {
                    let PointWrapper(old_cost, _old_parent) = o.get();
                    if new_cost < *old_cost {
                        o.insert(PointWrapper(new_cost, current.clone()));
                        frontier.push(PointWrapper(new_cost + heuristic(neighbour), neighbour));
                    }
                },
                Entry::Vacant(v) => {
                    v.insert(PointWrapper(new_cost, current.clone()));
                    frontier.push(PointWrapper(new_cost + heuristic(neighbour), neighbour));
                }
            }
        }
    }

    let mut end = end.unwrap();
    let mut path = vec![];
    while let Some(next) = came_from.get(&end) {
        path.push(next.1);
        end = came_from.get(&end).unwrap().1;
        if end == start {
            break;
        }
    }
    path.reverse();
    path
}

pub fn distance_map<I: Iterator<Item=Coord>>(radius: i32, sources: I) -> Map<f32> {
    let radius_f = radius as f32;
    let mut distances = Map::<f32>::new(radius, radius_f);
    let mut queue = VecDeque::new();
    for source in sources {
        queue.push_back((0.0, source));
    }
    while let Some((dist, coord)) = queue.pop_front() {
        if distances[coord] < radius_f {
            continue;
        }
        distances[coord] = dist;
        for offset in DIRECTIONS {
            let neighbor = coord + *offset;
            let new_distance = dist + 1.0;
            if distances.in_range(neighbor) && new_distance < distances[neighbor] {
                queue.push_back((new_distance, neighbor));
            }
        }
    }
    distances
}

pub fn compute_groundwater(water: &Map<f32>, rain: &Map<f32>, heightmap: &Map<f32>) -> Map<f32> {
    let mut groundwater = distance_map(
        water.radius,
        water.iter().filter_map(|(c, i)| if *i > 0.0 { Some(c) } else { None }));
    percentilize(&mut groundwater, |x| (1.0 - x).powf(0.3));

    for (coord, gw) in groundwater.iter_mut() {
        *gw -= heightmap[coord] * 0.05;
        *gw += rain[coord] * 0.15;
    }
    percentilize(&mut groundwater, |x| x.powf(0.7));
    groundwater
}

const SEA_LEVEL: f32 = -0.8;
const EROSION: f32 = 0.09;
const EROSION_EXPONENT: f32 = 1.5;

fn floodfill(src: Coord, all: &HashSet<Coord>) -> HashSet<Coord> {
    let mut out = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(src);
    out.insert(src);
    while let Some(coord) = queue.pop_front() {
        for offset in DIRECTIONS {
            let neighbor = coord + *offset;
            if all.contains(&neighbor) && !out.contains(&neighbor) {
                queue.push_back(neighbor);
                out.insert(neighbor);
            }
        }
    }
    out
}

pub fn get_sea(heightmap: &Map<f32>) -> (HashSet<Coord>, HashSet<Coord>) {
    let sinks = heightmap.iter_coords().filter(|c| heightmap[*c] <= SEA_LEVEL).collect::<HashSet<_>>();
    let sea = floodfill(Coord::new(0, heightmap.radius), &sinks);
    (sinks, sea)
}

const SALT_REMOVAL: f32 = 0.13;
const SALT_RANGE: f32 = 0.33;

pub fn simulate_water(heightmap: &mut Map<f32>, rain_map: &Map<f32>, sea: &HashSet<Coord>, sinks: &HashSet<Coord>) -> (Map<f32>, Map<f32>) {
    let mut watermap = Map::<f32>::new(heightmap.radius, 0.0);

    let sources = generate_separated_high_points(WATER_SOURCES, (heightmap.radius / 10).max(1), rain_map);
    let mut remainder = sinks.clone();

    for s in sea.iter() {
        remainder.remove(&s);
    }

    let mut lakes = vec![];
    loop {
        let next = remainder.iter().next();
        match next {
            Some(s) => {
                let lake = floodfill(*s, &remainder);
                for l in lake.iter() {
                    remainder.remove(l);
                }
                lakes.push(lake);
            },
            None => break
        }
    }

    let mut salt = distance_map(heightmap.radius, sea.iter().copied());
    normalize(&mut salt, |x| (SALT_RANGE - x).max(0.0) / SALT_RANGE);
    for (coord, salt) in salt.iter_mut() {
        let rain = rain_map[coord];
        if rain > 0.0 {
            *salt = (*salt - rain * 0.3).max(0.0);
        }
    }

    for sink in sinks.iter() {
        watermap[*sink] = 10.0;
    }

    for source in sources.iter() {
        let heightmap_ = &*heightmap;
        let watermap_ = &watermap;
        let get_neighbours = |c: Coord| {
            DIRECTIONS.iter().flat_map(move |offset| {
                let neighbor = c + *offset;
                if heightmap_.in_range(neighbor) {
                    let factor = if watermap_[neighbor] > 0.0 { 0.1 } else { 1.0 };
                    Some((neighbor, factor * (heightmap_[neighbor] - heightmap_[c] + 0.0001).max(0.0)))
                } else {
                    None
                }
            })
        };
        let heuristic = |c: Coord| {
            (heightmap[c] * 0.00).max(0.0)
        };
        let mut path = astar(*source, |c| sinks.contains(&c), heuristic, get_neighbours);

        let end = path.last().unwrap();
        if !sea.contains(end) {
            // route lake to sea
            path.extend(astar(*end, |c| sea.contains(&c), heuristic, get_neighbours));
        }

        for point in path {
            let water_range_raw = watermap[point] * 1.0 - heightmap[point];
            let water_range = water_range_raw.ceil() as i32;
            for (_, nearby) in hex_range(water_range) {
                if !watermap.in_range(point + nearby) {
                    continue;
                }
                watermap[point + nearby] += 0.5;
                watermap[point + nearby] = watermap[point + nearby].min(3.0);
            }

            let erosion_range_raw = (water_range_raw * 2.0 + 2.0).powf(EROSION_EXPONENT);
            let erosion_range = erosion_range_raw.ceil() as i32;
            for (this_range, nearby) in hex_range(erosion_range) {
                if !watermap.in_range(point + nearby) {
                    continue;
                }
                // Erode ground (down to sea level at most)
                if this_range > 0 {
                    let water_rate = watermap[point] / (this_range as f32) / erosion_range_raw.max(1.0).powf(EROSION_EXPONENT);
                    heightmap[point + nearby] -= EROSION * water_rate;
                    heightmap[point + nearby] = heightmap[point + nearby].max(SEA_LEVEL);
                    salt[point + nearby] -= SALT_REMOVAL * water_rate; // freshwater rivers reduce salt nearby
                    salt[point + nearby] = salt[point + nearby].max(0.0);
                }
            }
        }
    }

    (watermap, salt)
}

const NUTRIENT_NOISE_SCALE: f32 = 0.0015;

// As a handwave, define soil nutrients to be partly randomized and partly based on water.
// This kind of sort of makes sense because nitrogen is partly fixed by plants, which would have grown in water-having areas.
pub fn soil_nutrients(groundwater: &Map<f32>) -> Map<f32> {
    let mut soil_nutrients = Map::<f32>::from_fn(|d| {
        let c = to_cubic(d);
        noise_functions::OpenSimplex2s.seed(406).sample3([
            10.0 + c.x as f32 * NUTRIENT_NOISE_SCALE,
            c.y as f32 * NUTRIENT_NOISE_SCALE,
            c.z as f32 * NUTRIENT_NOISE_SCALE
        ]) + groundwater[d]
    }, groundwater.radius);
    percentilize(&mut soil_nutrients, |x| x.powf(0.4));
    soil_nutrients
}

struct WindSlice {
    coord: Coord,
    humidity: f32, // hPa
    temperature: f32, // relative, offset from base temperature
    last_height: f32
}

// https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology
// temperature in degrees Celsius for some reason, pressure in hPa
// returns approx. saturation vapour pressure of water
fn august_roche_magnus(temperature: f32) -> f32 {
    6.1094 * f32::exp((17.625 * temperature) / (243.04 + temperature))
}

const BASE_TEMPERATURE: f32 = 30.0; // degrees
const HEIGHT_SCALE: f32 = 1e3; // unrealistic but makes world more interesting; m
//const SEA_LEVEL_AIR_PRESSURE: f32 = 1013.0; // hPa
//const PRESSURE_DROP_PER_METER: f32 = 0.001; // hPa m^-1
const AIR_SPECIFIC_HEAT_CAPACITY: f32 = 1012.0; // J kg^-1 K^-1
const EARTH_GRAVITY: f32 = 9.81; // m s^-2

pub fn simulate_air(heightmap: &Map<f32>, sea: &HashSet<Coord>, scan_dir: CoordVec, perpendicular_dir: CoordVec) -> (Map<f32>, Map<f32>, Map<f32>) {
    let radius = heightmap.radius;
    let start_pos = Coord::origin() + -scan_dir * radius;
    let mut rain_map = Map::<f32>::new(heightmap.radius, 0.0);
    let mut temperature_map = Map::<f32>::new(heightmap.radius, 0.0);
    let mut atmo_humidity = Map::<f32>::new(heightmap.radius, 0.0); // relative humidity
    let mut frontier = (-radius..=radius).map(|x| WindSlice {
        coord: start_pos + perpendicular_dir * x,
        humidity: 0.0,
        temperature: 0.0,
        last_height: -1.0
    }).collect::<Vec<_>>();
    loop {
        let mut any_in_range = false;
        // Wind moves across the terrain in some direction.
        // We treat it as a line advancing and gaining/losing water and temperature.
        // Water is lost when the partial pressure of water in the air is greater than the saturation vapour pressure.
        // Temperature changes with height based on a slightly dubious equation I derived.
        // TODO: recalculate this; Wikipedia says temperature change is actually
        // because of air pressure changes and not directly derived from GPE.
        for slice in frontier.iter_mut() {
            if heightmap.in_range(slice.coord) {
                any_in_range = true;
                // okay approximation
                //let air_pressure = SEA_LEVEL_AIR_PRESSURE - PRESSURE_DROP_PER_METER * heightmap[slice.coord] * HEIGHT_SCALE; // hPa
                let max_water = august_roche_magnus(slice.temperature);

                // sea: reset temperature, max humidity
                if sea.contains(&slice.coord) {
                    slice.temperature = BASE_TEMPERATURE;
                    slice.humidity = max_water * 0.9;
                    slice.last_height = SEA_LEVEL;

                } else {
                    let excess = (slice.humidity - max_water).max(0.0);
                    slice.humidity -= excess;
                    let delta_h = (heightmap[slice.coord] - slice.last_height) * HEIGHT_SCALE;
                    // ΔGPE = mgh = hgρV = hgρHS
                    // ΔHE = CρHSΔT
                    // ΔT = hg / C
                    slice.temperature -= EARTH_GRAVITY * delta_h / AIR_SPECIFIC_HEAT_CAPACITY;
                    rain_map[slice.coord] = excess;
                    slice.last_height = heightmap[slice.coord];
                }

                atmo_humidity[slice.coord] = slice.humidity / max_water;
                temperature_map[slice.coord] = slice.temperature;
            }

            slice.coord += scan_dir;
        }

        let mut next_temperature = vec![0.0; frontier.len()];
        let mut next_humidity = vec![0.0; frontier.len()];
        // Smooth out temperature and humidity.
        // TODO at some point: replace with generalized convolution
        let frontier_len = frontier.len();
        next_temperature.par_iter_mut().enumerate().for_each(|(i, out)| {
            if i == 0 || i + 1 >= frontier_len {
                return;
            }
            *out = 1.0/3.0 * (frontier[i-1].temperature + frontier[i+1].temperature + frontier[i].temperature);
        });
        next_humidity.par_iter_mut().enumerate().for_each(|(i, out)| {
            if i == 0 || i + 1 >= frontier_len {
                return;
            }
            *out = 1.0/3.0 * (frontier[i-1].humidity + frontier[i+1].humidity + frontier[i].humidity);
        });

        frontier.par_iter_mut().enumerate().for_each(|(i, slice)| {
            slice.temperature = next_temperature[i];
            slice.humidity = next_humidity[i];
        });

        if !any_in_range { break; }
    }

    normalize(&mut rain_map, |x| x.powf(0.5));
    let rain_map = smooth(&rain_map, 3);

    normalize(&mut temperature_map, |x| x);

    (rain_map, temperature_map, atmo_humidity)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TerrainType {
    Empty,
    Contour,
    ShallowWater,
    DeepWater,
    Wall
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedWorld {
    pub heightmap: Map<f32>,
    pub terrain: Map<TerrainType>,
    pub groundwater: Map<f32>,
    pub salt: Map<f32>,
    pub atmo_humidity: Map<f32>,
    pub temperature: Map<f32>,
    pub soil_nutrients: Map<f32>,
    pub radius: i32
}

pub fn generate_world() -> GeneratedWorld {
    let mut heightmap = generate_heights();
    let mut terrain = Map::<TerrainType>::new(heightmap.radius, TerrainType::Empty);

    let (sinks, sea) = get_sea(&heightmap);

    let (rain, temperature, atmo_humidity) = simulate_air(&heightmap, &sea, CoordVec::new(0, -1), CoordVec::new(1, 0));

    let (water, salt) = simulate_water(&mut heightmap, &rain, &sea, &sinks);

    let contours = generate_contours(&heightmap, 0.1);

    for (point, _, _, _) in contours {
        terrain[point] = TerrainType::Contour;
    }

    for (point, water) in water.iter() {
        if *water > 1.0 {
            terrain[point] = TerrainType::DeepWater;
        } else if *water > 0.0 {
            terrain[point] = TerrainType::ShallowWater;
        }
    }

    let groundwater = compute_groundwater(&water, &rain, &heightmap);

    let soil_nutrients = soil_nutrients(&groundwater);

    GeneratedWorld {
        radius: heightmap.radius,
        heightmap,
        terrain,
        groundwater,
        salt,
        atmo_humidity,
        temperature,
        soil_nutrients
    }
}

impl TerrainType {
    pub fn entry_cost(&self) -> Option<i64> {
        match *self {
            Self::Empty => Some(0),
            Self::Wall => None,
            Self::ShallowWater => Some(10),
            Self::DeepWater => None,
            Self::Contour => Some(1)
        }
    }

    pub fn symbol(&self) -> Option<char> {
        match *self {
            Self::Empty => None,
            Self::Wall => Some('#'),
            Self::ShallowWater => Some('~'),
            Self::DeepWater => Some('≈'),
            Self::Contour => Some('/')
        }
    }
}

impl GeneratedWorld {
    pub fn get_terrain(&self, pos: Coord) -> TerrainType {
        let distance = hex_distance(pos, Coord::origin());

        if distance >= self.heightmap.radius {
            return TerrainType::Wall
        }

        self.terrain[pos].clone()
    }

    pub fn radius(&self) -> i32 {
        self.heightmap.radius
    }
}

#[cfg(test)]
mod test {
    use super::{Map, smooth, WORLD_RADIUS};
    use test::bench::Bencher;

    #[bench]
    fn bench_smooth_map(b: &mut Bencher) {
        use std::hint::black_box;
        b.iter(|| {
            let map = black_box(Map::new(WORLD_RADIUS, 0.0f32));
            smooth(&map, 3);
        });
    }
}
