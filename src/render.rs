use crate::map::Coord;

pub fn hex_to_image_coords(pos: Coord, radius: i32) -> (u32, u32) {
    let col = pos.x + (pos.y - (pos.y & 1)) / 2 + radius;
    let row = pos.y + radius;
    (col as u32, row as u32)
}

pub fn normalize(v: f32, min: f32, max: f32) -> f32 {
    if (max - min).abs() < f32::EPSILON {
        0.5
    } else {
        ((v - min) / (max - min)).clamp(0.0, 1.0)
    }
}

pub fn unit_to_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0) as u8
}
