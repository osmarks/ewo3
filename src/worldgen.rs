use std::{cmp::Ordering, collections::{hash_map::Entry, BinaryHeap, HashMap, HashSet, VecDeque}, hash::{Hash, Hasher}, ops::{Index, IndexMut}};

use noise_functions::Sample3;
use serde::{Deserialize, Serialize};
use crate::map::*;

pub const WORLD_RADIUS: i64 = 1024;
const NOISE_SCALE: f32 = 0.001;
const HEIGHT_EXPONENT: f32 = 0.3;
const WATER_SOURCES: usize = 40;

pub fn height_baseline(pos: Coord) -> f32 {
    let w_frac = (hex_distance(pos, Coord::origin()) as f32 / WORLD_RADIUS as f32).powf(3.0);
    let pos = to_cubic(pos);
    let noise = 
        noise_functions::OpenSimplex2s.ridged(6, 0.7, 1.55).seed(406).sample3([10.0 + pos.x as f32 * NOISE_SCALE, pos.y as f32 * NOISE_SCALE, pos.z as f32 * NOISE_SCALE]);
    let range = 1.0 - 2.0 * (w_frac - 0.5).powf(2.0);
    noise * range - w_frac
}

fn percentilize<F: Fn(f32) -> f32>(raw: &mut Map<f32>, postprocess: F) {
    let mut xs: Vec<(usize, f32)> = raw.data.iter().copied().enumerate().collect();
    xs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (j, (i, _x)) in xs.into_iter().enumerate() {
        let percentile = j as f32 / raw.data.len() as f32;
        raw.data[i] = postprocess(percentile);
    }
}

pub fn generate_heights() -> Map<f32> {
    let mut raw = Map::<f32>::from_fn(height_baseline, WORLD_RADIUS);
    percentilize(&mut raw, |x| {
        let s = 1.0 - (1.0 - x).powf(HEIGHT_EXPONENT);
        s * 2.0 - 1.0
    });
    raw
}

struct CoordsIndexIterator {
    radius: i64,
    index: usize,
    r: i64,
    q: i64,
    max: usize
}

impl Iterator for CoordsIndexIterator {
    type Item = (Coord, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.max {
            return None;
        }
        let result = (Coord::new(self.q, self.r), self.index);
        self.index += 1;
        self.q += 1;
        if self.r < 0 && self.q == self.radius + 1 {
            self.r += 1;
            self.q = -self.radius - self.r;
        }
        if self.r >= 0 && self.q + self.r == self.radius + 1 {
            self.r += 1;
            self.q = -self.radius;
        }
        Some(result)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Map<T> {
    pub data: Vec<T>,
    pub radius: i64
}

impl<T> Map<T> {
    pub fn new(radius: i64, fill: T) -> Map<T> where T: Clone {
        let size = count_hexes(radius) as usize;
        Map {
            data: vec![fill; size],
            radius
        }
    }

    pub fn from_fn<S, F: FnMut(Coord) -> S>(mut f: F, radius: i64) -> Map<S> {
        let size = count_hexes(radius) as usize;
        Map {
            radius,
            data: Vec::from_iter(CoordsIndexIterator {
                radius,
                index: 0,
                max: size,
                r: -radius,
                q: 0
            }.map(|(c, _i)| f(c)))
        }
    }

    pub fn map<S, F: FnMut(&T) -> S>(mut f: F, other: &Self) -> Map<S> {
        Map::<S>::from_fn(|c| f(&other[c]), other.radius)
    }

    fn coord_to_index(&self, c: Coord) -> usize {
        let r = c.y + self.radius;
        let fh = r.min(self.radius);
        let mut coords_above = fh*(self.radius+1) + fh*(fh-1)/2;
        if fh < r {
            let d = r - fh;
            coords_above += d*(2*self.radius+1) - d*(d-1)/2;
        }
        let q_start = if r < self.radius { -r } else { -self.radius };
        (coords_above + (c.x - q_start)) as usize
    }

    fn in_range(&self, coord: Coord) -> bool {
        hex_distance(coord, Coord::origin()) <= self.radius
    }

    fn iter_coords(&self) -> impl Iterator<Item=(Coord, usize)> {
        CoordsIndexIterator {
            radius: self.radius,
            index: 0,
            max: self.data.len(),
            r: -self.radius,
            q: 0
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=(Coord, &T)> {
        self.iter_coords().map(|(c, i)| (c, &self.data[i]))
    }
}

impl<T> Index<Coord> for Map<T> {
    type Output = T;
    fn index(&self, index: Coord) -> &Self::Output {
        //println!("{:?}", index);
        &self.data[self.coord_to_index(index)]
    }
}

impl<T> IndexMut<Coord> for Map<T> {
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        let i = self.coord_to_index(index);
        &mut self.data[i]
    }
}

pub fn generate_contours(field: &Map<f32>, interval: f32) -> Vec<(Coord, f32, f32, CoordVec)> {
    let mut v = vec![];
    // Starting at the origin, we want to detect contour lines in any of the six directions.
    // Go in one of the perpendicular directions to generate base directions then scan until the edge is reached starting from any of those points.
    for scan_direction in DIRECTIONS {
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
                            v.push((point, last, sample, *scan_direction));
                        }
                    }
                    last = Some(sample);
                }
            }
        }
    }
    v
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

pub fn generate_separated_high_points(n: usize, sep: i64, map: &Map<f32>) -> Vec<Coord> {
    let mut points = vec![];
    let mut priority = BinaryHeap::with_capacity(map.data.len());
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

pub fn distance_map<I: Iterator<Item=Coord>>(radius: i64, sources: I) -> Map<f32> {
    let radius_f = radius as f32;
    let mut distances = Map::<f32>::new(radius, radius_f);
    let mut queue = BinaryHeap::new();
    for source in sources {
        queue.push(PointWrapper(0.0, source));
    }
    while let Some(PointWrapper(dist, coord)) = queue.pop() {
        if distances[coord] < radius_f {
            continue;
        }
        if dist < radius_f {
            distances[coord] = dist;
        }
        for offset in DIRECTIONS {
            let neighbor = coord + *offset;
            let new_distance = dist + 1.0;
            if distances.in_range(neighbor) && new_distance < distances[neighbor] {
                queue.push(PointWrapper(new_distance, neighbor));
            }
        }
    }
    distances
}

pub fn compute_humidity(distances: Map<f32>, heightmap: &Map<f32>) -> Map<f32> {
    let mut humidity = distances;
    percentilize(&mut humidity, |x| (1.0 - x).powf(0.3));
    for (coord, h) in heightmap.iter() {
        humidity[coord] -= *h * 0.6;
    }
    percentilize(&mut humidity, |x| x.powf(1.0));
    humidity
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

pub fn simulate_water(heightmap: &mut Map<f32>) -> Map<f32> {
    let mut watermap = Map::<f32>::new(heightmap.radius, 0.0);

    let sources = generate_separated_high_points(WATER_SOURCES, WORLD_RADIUS / 10, &heightmap);
    let sinks = heightmap.iter_coords().filter(|(c, _)| heightmap[*c] <= SEA_LEVEL).map(|(c, _)| c).collect::<HashSet<_>>();
    let mut remainder = sinks.clone();
    let sea = floodfill(Coord::new(0, WORLD_RADIUS), &sinks);

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
            let water_range = water_range_raw.ceil() as i64;
            for (_, nearby) in hex_range(water_range) {
                if !watermap.in_range(point + nearby) {
                    continue;
                }
                watermap[point + nearby] += 0.5;
                watermap[point + nearby] = watermap[point + nearby].min(3.0);
            }

            let erosion_range_raw = (water_range_raw * 2.0 + 2.0).powf(EROSION_EXPONENT);
            let erosion_range = erosion_range_raw.ceil() as i64;
            for (this_range, nearby) in hex_range(erosion_range) {
                if !watermap.in_range(point + nearby) {
                    continue;
                }
                if this_range > 0 {
                    heightmap[point + nearby] -= EROSION * watermap[point] / (this_range as f32) / erosion_range_raw.max(1.0).powf(EROSION_EXPONENT);
                    heightmap[point + nearby] = heightmap[point + nearby].max(SEA_LEVEL);
                }
            }
        }
    }

    watermap
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
    heightmap: Map<f32>,
    terrain: Map<TerrainType>,
    humidity: Map<f32>
}

pub fn generate_world() -> GeneratedWorld {
    let mut heightmap = generate_heights();
    let mut terrain = Map::<TerrainType>::new(heightmap.radius, TerrainType::Empty);

    let water = simulate_water(&mut heightmap);

    let contours = generate_contours(&heightmap, 0.15);

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

    let distances = distance_map(
        WORLD_RADIUS, 
        water.iter().filter_map(|(c, i)| if *i > 0.0 { Some(c) } else { None }));
    let humidity = compute_humidity(distances, &heightmap);

    GeneratedWorld {
        heightmap,
        terrain,
        humidity
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

    pub fn radius(&self) -> i64 {
        self.heightmap.radius
    }
}