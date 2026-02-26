use euclid::{Point3D, Point2D, Vector2D};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;
use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};

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

struct CoordIterator {
    radius: i32,
    count: usize,
    r: i32,
    q: i32,
    max: usize
}

struct CoordIterMut<'a, T> {
    coords: CoordIterator,
    data: *mut T,
    radius: i32,
    side_length: usize,
    _marker: PhantomData<&'a mut T>
}

impl Iterator for CoordIterator {
    type Item = Coord;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count == self.max {
            return None;
        }
        let result = Coord::new(self.q, self.r);
        self.count += 1;
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

// blame OpenAI for this, and also Mozilla
impl<'a, T> Iterator for CoordIterMut<'a, T> {
    type Item = (Coord, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let coord = self.coords.next()?;
        let q = (coord.x + self.radius) as usize;
        let r = (coord.y + self.radius) as usize;
        let i = q + r * self.side_length;
        unsafe {
            // CoordIterator yields each valid map coordinate once, so each backing index is yielded once.
            Some((coord, &mut *self.data.add(i)))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Map<T> {
    data: Vec<T>,
    pub radius: i32,
    side_length: usize
}

impl<T> Map<T> {
    pub fn new(radius: i32, fill: T) -> Map<T> where T: Clone {
        // We represent worlds using axial coordinates (+q is right, +r is down-right).
        // This results in a parallelogram shape.
        // However, the map is a hexagon for the purposes of gameplay, so the top-left and bottom-right corners are invalid.
        let side_length = (radius * 2 + 1) as usize;
        let size = side_length.pow(2);
        Map {
            data: vec![fill; size],
            radius,
            side_length
        }
    }

    pub fn size(&self) -> usize {
        count_hexes(self.radius) as usize
    }

    pub fn from_fn<S: Default + Clone, F: FnMut(Coord) -> S>(mut f: F, radius: i32) -> Map<S> {
        let mut map = Map::new(radius, S::default());
        for coord in map.iter_coords() {
            map[coord] = f(coord);
        }
        map
    }

    pub fn map<S: Default + Clone, F: FnMut(&T) -> S>(mut f: F, other: &Self) -> Map<S> {
        Map::<S>::from_fn(|c| f(&other[c]), other.radius)
    }

    pub fn coord_to_index(&self, c: Coord) -> usize {
        let q = (c.x + self.radius) as usize;
        let r = (c.y + self.radius) as usize;
        q * self.side_length + r
    }

    pub fn in_range(&self, coord: Coord) -> bool {
        hex_distance(coord, Coord::origin()) <= self.radius
    }

    pub fn iter_coords(&self) -> impl Iterator<Item=Coord> {
        CoordIterator {
            radius: self.radius,
            count: 0,
            max: self.size(),
            r: -self.radius,
            q: 0
        }
    }

    pub fn iter(&self) -> impl Iterator<Item=(Coord, &T)> {
        self.iter_coords().map(|c| (c, &self[c]))
    }

    pub fn iter_data(&self) -> impl Iterator<Item=&T> {
        self.iter_coords().map(|c| &self[c])
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=(Coord, &mut T)> {
        CoordIterMut {
            coords: CoordIterator {
                radius: self.radius,
                count: 0,
                max: self.size(),
                r: -self.radius,
                q: 0
            },
            data: self.data.as_mut_ptr(),
            radius: self.radius,
            side_length: self.side_length,
            _marker: PhantomData
        }
    }

    pub fn for_each_mut(&mut self, mut f: impl FnMut(&mut T)) {
        for (_, value) in self.iter_mut() {
            f(value);
        }
    }
}

// 2D hex convolution
pub fn smooth(map: &Map<f32>, radius: i32) -> Map<f32> {
    //let mut map = map.clone();
    //map[Coord::new(1, 0)] = 3.0;
    let data: ArrayBase<ndarray::ViewRepr<&f32>, Dim<[usize; 2]>, f32> = ArrayView2::from_shape((map.side_length, map.side_length), map.data.as_slice()).unwrap();
    //println!("{:?}", data[(map.radius as usize + 1, map.radius as usize)]);
    let mut kernel = Array2::from_elem((radius as usize * 2 + 1, radius as usize * 2 + 1), 0.0);

    for (_, offset) in hex_range(radius) {
        kernel[(offset.x as usize + radius as usize, offset.y as usize + radius as usize)] = 1.0 / count_hexes(radius) as f32;
    }

    // TODO: this is still really slow!
    let result = ConvExt::conv(&data, &kernel, ConvMode::Same, PaddingMode::Replicate).unwrap();

    Map {
        radius: map.radius,
        side_length: map.side_length,
        data: result.into_raw_vec(), // TODO fix
    }
}

impl<T> Index<Coord> for Map<T> {
    type Output = T;
    fn index(&self, index: Coord) -> &Self::Output {
        debug_assert!(self.in_range(index), "invalid coord: {:?}", index);
        &self.data[self.coord_to_index(index)]
    }
}

impl<T> IndexMut<Coord> for Map<T> {
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        debug_assert!(self.in_range(index), "invalid coord: {:?}", index);
        let i = self.coord_to_index(index);
        &mut self.data[i]
    }
}
