use anyhow::Result;
use image::{ImageBuffer, Rgb};
use std::collections::{HashMap, HashSet};

mod worldgen;
mod map;

use worldgen::*;
use map::*;

fn main() -> Result<()> {
    let mut heightmap = generate_heights();

    println!("hydro...");
    let water = simulate_water(&mut heightmap);

    println!("contours...");
    let contours = generate_contours(&heightmap, 0.15);
    let mut contour_points = HashMap::new();

    for (point, x1, x2, _) in contours {
        let steepness = x1 - x2;
        let entry = contour_points.entry(point).or_default();
        *entry = std::cmp::max(*entry, (steepness * 4000.0).abs() as u8);
    }

    println!("humidity...");
    let water_distances = distance_map(
            WORLD_RADIUS, 
            water.iter().filter_map(|(c, i)| if *i > 0.0 { Some(c) } else { None }));
    let humidity = compute_humidity(water_distances, &heightmap);

    println!("rendering...");
    let mut image = ImageBuffer::from_pixel((WORLD_RADIUS * 2 + 1) as u32, (WORLD_RADIUS * 2 + 1) as u32, Rgb::from([0u8, 0, 0]));

    for (position, value) in heightmap.iter() {
        let col = position.x + (position.y - (position.y & 1)) / 2 + WORLD_RADIUS;
        let row = position.y + WORLD_RADIUS;
        //let contour = contour_points.get(&position).copied().unwrap_or_default();
        let contour = (255.0 * humidity[position]) as u8;
        let water = water[position];
        image.put_pixel(col as u32, row as u32, Rgb::from([contour, 0, (water.min(1.0) * 255.0) as u8]));
    }

    image.save("./out.png")?;

    Ok(())
}