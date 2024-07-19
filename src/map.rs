use euclid::{Point3D, Point2D, Vector2D};

pub struct AxialWorldSpace;
pub struct CubicWorldSpace;
pub type Coord = Point2D<i64, AxialWorldSpace>;
pub type CubicCoord = Point3D<i64, CubicWorldSpace>;
pub type CoordVec = Vector2D<i64, AxialWorldSpace>;

pub fn to_cubic(p0: Coord) -> CubicCoord {
    CubicCoord::new(p0.x, p0.y, -p0.x - p0.y)
}

pub fn hex_distance(p0: Coord, p1: Coord) -> i64 {
    let ax_dist = p0 - p1;
    (ax_dist.x.abs() + ax_dist.y.abs() + (ax_dist.x + ax_dist.y).abs()) / 2
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

pub fn sample_range(range: i64) -> CoordVec {
    let q = fastrand::i64(-range..=range);
    let r = fastrand::i64((-range).max(-q-range)..=range.min(-q+range));
    CoordVec::new(q, r)
}

pub fn hex_circle(range: i64) -> impl Iterator<Item=CoordVec> {
    (-range..=range).flat_map(move |q| {
        ((-range).max(-q - range)..= range.min(-q+range)).map(move |r| {
            CoordVec::new(q, r)
        })
    })
}

pub fn hex_range(range: i64) -> impl Iterator<Item=(i64, CoordVec)> {
    (0..=range).flat_map(|x| hex_circle(x).map(move |c| (x, c)))
}


pub fn count_hexes(x: i64) -> i64 {
    x*(x+1)*3+1
}