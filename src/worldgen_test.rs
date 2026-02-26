#![feature(test)]
extern crate test;

use anyhow::Result;
use argh::FromArgs;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::time::Instant;

mod worldgen;
mod map;

use map::*;
use worldgen::*;

#[derive(FromArgs)]
/// Render worldgen debug fields to a PNG.
struct Args {
    /// world radius
    #[argh(option, default = "WORLD_RADIUS")]
    radius: i32,
    /// output path
    #[argh(option, default = "String::from(\"./out.png\")")]
    output: String,
    /// first channel field (defaults to 0 if omitted)
    #[argh(option)]
    c1: Option<String>,
    /// second channel field (defaults to 0 if omitted)
    #[argh(option)]
    c2: Option<String>,
    /// third channel field (defaults to 0 if omitted)
    #[argh(option)]
    c3: Option<String>,
    /// color space: rgb or oklab
    #[argh(option, default = "String::from(\"oklab\")")]
    color_space: String,
    /// percentile-normalize each selected channel
    #[argh(switch)]
    normalize: bool,
}

#[derive(Clone, Copy)]
enum Field {
    Height,
    Rain,
    Water,
    Groundwater,
    Salt,
    Temperature,
    Humidity,
    Soil,
    Contour,
    SeaDistance,
}

impl Field {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "height" => Ok(Self::Height),
            "rain" => Ok(Self::Rain),
            "water" => Ok(Self::Water),
            "groundwater" => Ok(Self::Groundwater),
            "salt" => Ok(Self::Salt),
            "temperature" => Ok(Self::Temperature),
            "humidity" => Ok(Self::Humidity),
            "soil" => Ok(Self::Soil),
            "contour" => Ok(Self::Contour),
            "sea_distance" => Ok(Self::SeaDistance),
            _ => anyhow::bail!("unknown field: {s}"),
        }
    }
}

struct RenderData<'a> {
    heightmap: &'a Map<f32>,
    rain: &'a Map<f32>,
    water: &'a Map<f32>,
    groundwater: &'a Map<f32>,
    salt: &'a Map<f32>,
    temperature: &'a Map<f32>,
    humidity: &'a Map<f32>,
    soil: &'a Map<f32>,
    contour_points: &'a HashMap<Coord, u8>,
    sea_distance: &'a Map<f32>,
}

fn sample_field(field: Option<Field>, position: Coord, data: &RenderData) -> f32 {
    match field {
        None => 0.0,
        Some(field) => match field {
        Field::Height => ((data.heightmap[position] + 1.0) * 0.5).clamp(0.0, 1.0),
        Field::Rain => data.rain[position].clamp(0.0, 1.0),
        Field::Water => data.water[position].min(1.0),
        Field::Groundwater => data.groundwater[position].clamp(0.0, 1.0),
        Field::Salt => data.salt[position].clamp(0.0, 1.0),
        Field::Temperature => data.temperature[position].clamp(0.0, 1.0),
        Field::Humidity => data.humidity[position].clamp(0.0, 1.0),
        Field::Soil => data.soil[position].clamp(0.0, 1.0),
        Field::Contour => data.contour_points.get(&position).copied().unwrap_or_default() as f32 / 255.0,
        Field::SeaDistance => data.sea_distance[position].clamp(0.0, 1.0),
        },
    }
}

fn field_range(field: Option<Field>, data: &RenderData) -> (f32, f32) {
    if field.is_none() {
        return (0.0, 1.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (position, _) in data.heightmap.iter() {
        let v = sample_field(field, position, data);
        min = min.min(v);
        max = max.max(v);
    }
    if (max - min).abs() < f32::EPSILON {
        (0.0, 1.0)
    } else {
        (min, max)
    }
}

fn to_rgb(c1: f32, c2: f32, c3: f32, color_space: &str) -> [u8; 3] {
    fn linear_to_srgb(x: f32) -> f32 {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    let (r, g, b) = match color_space {
        "rgb" => (c1, c2, c3),
        "oklab" => {
            let l = c1.clamp(0.0, 1.0);
            let a = c2 * 2.0 - 1.0;
            let b = c3 * 2.0 - 1.0;

            let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
            let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
            let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

            let l3 = l_ * l_ * l_;
            let m3 = m_ * m_ * m_;
            let s3 = s_ * s_ * s_;

            let r_lin = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
            let g_lin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
            let b_lin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3;

            (
                linear_to_srgb(r_lin).clamp(0.0, 1.0),
                linear_to_srgb(g_lin).clamp(0.0, 1.0),
                linear_to_srgb(b_lin).clamp(0.0, 1.0),
            )
        }
        _ => (c1, c2, c3),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    let args: Args = argh::from_env();
    let f1 = args.c1.as_deref().map(Field::parse).transpose()?;
    let f2 = args.c2.as_deref().map(Field::parse).transpose()?;
    let f3 = args.c3.as_deref().map(Field::parse).transpose()?;
    let radius = args.radius.max(1);

    let t = Instant::now();
    let mut heightmap = generate_heights_with_radius(radius);
    println!("heights: {:.3}s", t.elapsed().as_secs_f32());

    println!("hydro...");
    let hydro_start = Instant::now();
    let t = Instant::now();
    let (sinks, sea) = get_sea(&heightmap);
    println!("  sea/sinks: {:.3}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let (rain, temperature, atmo_humidity) =
        simulate_air(&heightmap, &sea, CoordVec::new(0, -1), CoordVec::new(1, 0));
    println!("  air: {:.3}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let (water, salt) = simulate_water(&mut heightmap, &rain, &sea, &sinks);
    println!("  water: {:.3}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let groundwater = compute_groundwater(&water, &rain, &heightmap);
    println!("  groundwater: {:.3}s", t.elapsed().as_secs_f32());
    let t = Instant::now();
    let mut sea_distance = distance_map(heightmap.radius, sea.iter().copied());
    let radius = heightmap.radius as f32;
    for (_, value) in sea_distance.iter_mut() {
        *value = (*value / radius).clamp(0.0, 1.0);
    }
    println!("  sea distance: {:.3}s", t.elapsed().as_secs_f32());
    println!("hydro total: {:.3}s", hydro_start.elapsed().as_secs_f32());

    println!("contours...");
    let t = Instant::now();
    let contours = generate_contours(&heightmap, 0.15);
    println!("contours: {:.3}s", t.elapsed().as_secs_f32());
    let mut contour_points = HashMap::new();

    for (point, x1, x2, _) in contours {
        let steepness = x1 - x2;
        let entry = contour_points.entry(point).or_default();
        *entry = std::cmp::max(*entry, (steepness * 4000.0).abs() as u8);
    }

    println!("soil...");
    let t = Instant::now();
    let soil_nutrients = soil_nutrients(&groundwater);
    println!("soil: {:.3}s", t.elapsed().as_secs_f32());

    println!("rendering...");
    let t = Instant::now();
    let image_radius = heightmap.radius;
    let mut image = ImageBuffer::from_pixel((image_radius * 2 + 1) as u32, (image_radius * 2 + 1) as u32, Rgb::from([0u8, 0, 0]));
    let render_data = RenderData {
        heightmap: &heightmap,
        rain: &rain,
        water: &water,
        groundwater: &groundwater,
        salt: &salt,
        temperature: &temperature,
        humidity: &atmo_humidity,
        soil: &soil_nutrients,
        contour_points: &contour_points,
        sea_distance: &sea_distance,
    };
    let r1 = field_range(f1, &render_data);
    let r2 = field_range(f2, &render_data);
    let r3 = field_range(f3, &render_data);
    println!("ranges: {:?}, {:?}, {:?}", r1, r2, r3);

    for (position, _) in heightmap.iter() {
        let col = position.x + (position.y - (position.y & 1)) / 2 + image_radius;
        let row = position.y + image_radius;
        let mut c1 = sample_field(f1, position, &render_data);
        let mut c2 = sample_field(f2, position, &render_data);
        let mut c3 = sample_field(f3, position, &render_data);
        if args.normalize {
            c1 = ((c1 - r1.0) / (r1.1 - r1.0)).clamp(0.0, 1.0);
            c2 = ((c2 - r2.0) / (r2.1 - r2.0)).clamp(0.0, 1.0);
            c3 = ((c3 - r3.0) / (r3.1 - r3.0)).clamp(0.0, 1.0);
        }
        let rgb = to_rgb(c1, c2, c3, &args.color_space);
        image.put_pixel(col as u32, row as u32, Rgb::from(rgb));
    }
    println!("render: {:.3}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    image.save(args.output)?;
    println!("save: {:.3}s", t.elapsed().as_secs_f32());
    println!("total: {:.3}s", total_start.elapsed().as_secs_f32());

    Ok(())
}
