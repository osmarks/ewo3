use anyhow::Result;
use image::{ImageBuffer, Rgb};
use std::collections::{HashMap, HashSet};

mod worldgen;
mod map;

use worldgen::*;
use map::*;

fn main() -> Result<()> {
    let mut heightmap = generate_heights();
    let (sinks, sea) = get_sea(&heightmap);

    println!("wind...");
    let (rain, temperature, atmo_humidity) = simulate_air(&heightmap, &sea, CoordVec::new(0, -1), CoordVec::new(1, 0));

    println!("hydro...");
    let water = simulate_water(&mut heightmap, &rain, &sea, &sinks);

    println!("contours...");
    let contours = generate_contours(&heightmap, 0.15);
    let mut contour_points = HashMap::new();

    for (point, x1, x2, _) in contours {
        let steepness = x1 - x2;
        let entry = contour_points.entry(point).or_default();
        *entry = std::cmp::max(*entry, (steepness * 4000.0).abs() as u8);
    }

    println!("humidity...");
    let groundwater = compute_groundwater(&water, &rain, &heightmap);

    println!("rendering...");
    let mut image = ImageBuffer::from_pixel((WORLD_RADIUS * 2 + 1) as u32, (WORLD_RADIUS * 2 + 1) as u32, Rgb::from([0u8, 0, 0]));

    for (position, value) in heightmap.iter() {
        let col = position.x + (position.y - (position.y & 1)) / 2 + WORLD_RADIUS;
        let row = position.y + WORLD_RADIUS;
        //let height = ((*value + 1.0) * 127.5) as u8;
        let green_channel = groundwater[position];
        let red_channel = temperature[position];
        let blue_channel = water[position].min(1.0);
        image.put_pixel(col as u32, row as u32, Rgb::from([(red_channel * 255.0) as u8, (green_channel * 255.0) as u8, (blue_channel * 255.0) as u8]));
    }

    image.save("./out.png")?;

    Ok(())
}