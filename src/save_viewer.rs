use anyhow::Result;
use argh::FromArgs;
use ewo3::components::{Plant, Position};
use ewo3::map::{Coord, Map};
use ewo3::render::{hex_to_image_coords, normalize, unit_to_u8};
use ewo3::save::SavedGame;
use ewo3::world_serde;
use image::{ImageBuffer, Rgb};

#[derive(FromArgs)]
/// Render groundwater/nutrients and plant growth status from a save file.
struct Args {
    /// input save path
    #[argh(option, default = "String::from(\"save.bin\")")]
    save: String,
    /// output image path
    #[argh(option, default = "String::from(\"save_view.png\")")]
    output: String,
}

fn total_field(base: &Map<f32>, dynamic: &Map<f32>, c: Coord) -> f32 {
    base[c] + dynamic[c]
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    let data = std::fs::read(&args.save)?;
    let save = SavedGame::decode(&data)?;
    let world = world_serde::deserialize_world_from_bytes(&save.world)?;

    let radius = save.map.radius;
    let side = (radius * 2 + 1) as u32;
    let mut image = ImageBuffer::from_pixel(side, side, Rgb([0u8, 0, 0]));

    let mut gw_min = f32::INFINITY;
    let mut gw_max = f32::NEG_INFINITY;
    let mut gw_base_min = f32::INFINITY;
    let mut gw_base_max = f32::NEG_INFINITY;
    let mut gw_dyn_min = f32::INFINITY;
    let mut gw_dyn_max = f32::NEG_INFINITY;
    let mut soil_min = f32::INFINITY;
    let mut soil_max = f32::NEG_INFINITY;
    let mut soil_base_min = f32::INFINITY;
    let mut soil_base_max = f32::NEG_INFINITY;
    let mut soil_dyn_min = f32::INFINITY;
    let mut soil_dyn_max = f32::NEG_INFINITY;
    for coord in save.map.heightmap.iter_coords() {
        let gw_base = save.map.groundwater[coord];
        let gw_dyn = save.dynamic_groundwater[coord];
        let gw = gw_base + gw_dyn;
        let soil_base = save.map.soil_nutrients[coord];
        let soil_dyn = save.dynamic_soil_nutrients[coord];
        let soil = soil_base + soil_dyn;
        gw_base_min = gw_base_min.min(gw_base);
        gw_base_max = gw_base_max.max(gw_base);
        gw_dyn_min = gw_dyn_min.min(gw_dyn);
        gw_dyn_max = gw_dyn_max.max(gw_dyn);
        gw_min = gw_min.min(gw);
        gw_max = gw_max.max(gw);
        soil_base_min = soil_base_min.min(soil_base);
        soil_base_max = soil_base_max.max(soil_base);
        soil_dyn_min = soil_dyn_min.min(soil_dyn);
        soil_dyn_max = soil_dyn_max.max(soil_dyn);
        soil_min = soil_min.min(soil);
        soil_max = soil_max.max(soil);
    }

    for coord in save.map.heightmap.iter_coords() {
        let gw = total_field(&save.map.groundwater, &save.dynamic_groundwater, coord);
        let soil = total_field(&save.map.soil_nutrients, &save.dynamic_soil_nutrients, coord);
        let gw_n = normalize(gw, gw_min, gw_max);
        let soil_n = normalize(soil, soil_min, soil_max);
        let (x, y) = hex_to_image_coords(coord, radius);
        image.put_pixel(
            x,
            y,
            Rgb([
                unit_to_u8(soil_n),
                unit_to_u8((soil_n + gw_n) * 0.5),
                unit_to_u8(gw_n),
            ]),
        );
    }

    let mut plants = 0usize;
    for (position, plant) in world.query::<(&Position, &Plant)>().iter() {
        let pos = position.head();
        if !save.map.heightmap.in_range(pos) {
            continue;
        }
        let gw = total_field(&save.map.groundwater, &save.dynamic_groundwater, pos);
        let soil = total_field(&save.map.soil_nutrients, &save.dynamic_soil_nutrients, pos);
        let salt = save.map.salt[pos];
        let temp = save.map.temperature[pos];
        let terrain = save.map.get_terrain(pos);
        let g = plant.genome.base_growth_rate(soil, gw, temp, salt, &terrain);
        let color = if g > 0.2 {
            Rgb([20u8, 220, 40])
        } else if g > 0.0 {
            Rgb([240u8, 200, 40])
        } else {
            Rgb([220u8, 40, 40])
        };
        let (x, y) = hex_to_image_coords(pos, radius);
        image.put_pixel(x, y, color);
        plants += 1;
    }

    image.save(&args.output)?;
    println!(
        "saved {} (ticks={}, plants={}, radius={})",
        args.output, save.ticks, plants, radius
    );
    println!(
        "groundwater: base[min={:.4}, max={:.4}] dynamic[min={:.4}, max={:.4}] total[min={:.4}, max={:.4}]",
        gw_base_min, gw_base_max, gw_dyn_min, gw_dyn_max, gw_min, gw_max
    );
    println!(
        "soil_nutrients: base[min={:.4}, max={:.4}] dynamic[min={:.4}, max={:.4}] total[min={:.4}, max={:.4}]",
        soil_base_min, soil_base_max, soil_dyn_min, soil_dyn_max, soil_min, soil_max
    );
    Ok(())
}
