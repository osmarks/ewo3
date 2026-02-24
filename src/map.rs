use euclid::{Point3D, Point2D, Vector2D};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

pub struct AxialWorldSpace;
pub struct CubicWorldSpace;
pub type Coord = Point2D<i32, AxialWorldSpace>;
pub type CubicCoord = Point3D<i32, CubicWorldSpace>;
pub type CoordVec = Vector2D<i32, AxialWorldSpace>;

pub fn to_cubic(p0: Coord) -> CubicCoord {
    CubicCoord::new(p0.x, p0.y, -p0.x - p0.y)
}

pub fn vec_length(ax_dist: CoordVec) -> i32 {
    (ax_dist.x.abs() + ax_dist.y.abs() + (ax_dist.x + ax_dist.y).abs()) / 2
}

pub fn hex_distance(p0: Coord, p1: Coord) -> i32 {
    let ax_dist = p0 - p1;
    vec_length(ax_dist)
}

pub fn on_axis(p: CoordVec) -> bool {
    let p = to_cubic(Coord::origin() + p);
    let mut zero_ax = 0;
    if p.x == 0 { zero_ax += 1 }
    if p.y == 0 { zero_ax += 1 }
    if p.z == 0 { zero_ax += 1 }
    zero_ax >= 1
}

pub fn rotate_60(p0: CoordVec) -> CoordVec {
    let s = -p0.x - p0.y;
    CoordVec::new(s, p0.x)
}

pub const DIRECTIONS: &[CoordVec] = &[CoordVec::new(0, -1), CoordVec::new(1, -1), CoordVec::new(-1, 0), CoordVec::new(1, 0), CoordVec::new(0, 1), CoordVec::new(-1, 1)];

pub fn sample_range(range: i32) -> CoordVec {
    let q = fastrand::i32(-range..=range);
    let r = fastrand::i32((-range).max(-q-range)..=range.min(-q+range));
    CoordVec::new(q, r)
}

pub fn hex_circle(range: i32) -> impl Iterator<Item=CoordVec> {
    (-range..=range).flat_map(move |q| {
        ((-range).max(-q - range)..= range.min(-q+range)).map(move |r| {
            CoordVec::new(q, r)
        })
    })
}

pub fn hex_range(range: i32) -> impl Iterator<Item=(i32, CoordVec)> {
    (0..=range).flat_map(|x| hex_circle(x).map(move |c| (x, c)))
}


pub fn count_hexes(x: i32) -> i32 {
    x*(x+1)*3+1
}

struct CoordsIndexIterator {
    radius: i32,
    index: usize,
    r: i32,
    q: i32,
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
    pub radius: i32
}

impl<T> Map<T> {
    pub fn new(radius: i32, fill: T) -> Map<T> where T: Clone {
        let size = count_hexes(radius) as usize;
        Map {
            data: vec![fill; size],
            radius
        }
    }

    pub fn from_fn<S, F: FnMut(Coord) -> S>(mut f: F, radius: i32) -> Map<S> {
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

    pub fn coord_to_index(&self, c: Coord) -> usize {
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

    pub fn in_range(&self, coord: Coord) -> bool {
        hex_distance(coord, Coord::origin()) <= self.radius
    }

    pub fn iter_coords(&self) -> impl Iterator<Item=(Coord, usize)> {
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