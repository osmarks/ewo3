#![feature(test)]
extern crate test;

use anyhow::Result;
use argh::FromArgs;
use ewo3::components::{Plant, Position};
use ewo3::map::*;
use ewo3::render::{hex_to_image_coords, normalize};
use ewo3::save::SavedGame;
use ewo3::world_serde;
use ewo3::worldgen::*;
use hecs::World;
use image::{ImageBuffer, Rgb};
use std::collections::HashMap;
use std::time::Instant;
use crate::map::Coord;

#[derive(FromArgs)]
/// Render world/debug fields to a PNG from either generated terrain or a saved game.
struct Args {
    /// world radius (used when --save is not provided)
    #[argh(option, default = "WORLD_RADIUS")]
    radius: i32,
    /// optional save path; when provided, loads world state from this save
    #[argh(option)]
    save: Option<String>,
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
    #[argh(option, default = "String::from(\"rgb\")")]
    color_space: String,
    /// normalize each selected channel
    #[argh(switch)]
    normalize: bool,
    /// field to include in PCA mode (repeat this option multiple times)
    #[argh(option)]
    pca_field: Vec<String>,
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
    Plants,
}

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
            "plants" => Ok(Self::Plants),
            _ => anyhow::bail!("unknown field: {s}"),
        }
    }
}

fn field_name(field: Field) -> &'static str {
    match field {
        Field::Height => "height",
        Field::Rain => "rain",
        Field::Water => "water",
        Field::Groundwater => "groundwater",
        Field::Salt => "salt",
        Field::Temperature => "temperature",
        Field::Humidity => "humidity",
        Field::Soil => "soil",
        Field::Contour => "contour",
        Field::SeaDistance => "sea_distance",
        Field::Plants => "plants",
    }
}

struct RenderData {
    world: GeneratedWorld,
    rain: Map<f32>,
    water: Map<f32>,
    groundwater: Map<f32>,
    soil: Map<f32>,
    contour_points: HashMap<Coord, u8>,
    sea_distance: Map<f32>,
    plants: Map<f32>,
}

fn sample_field(field: Field, position: Coord, data: &RenderData) -> f32 {
    match field {
        Field::Height => ((data.world.heightmap[position] + 1.0) * 0.5).clamp(0.0, 1.0),
        Field::Rain => data.rain[position].clamp(0.0, 1.0),
        Field::Water => data.water[position].min(1.0),
        Field::Groundwater => data.groundwater[position],
        Field::Salt => data.world.salt[position].clamp(0.0, 1.0),
        Field::Temperature => data.world.temperature[position].clamp(0.0, 1.0),
        Field::Humidity => data.world.atmo_humidity[position].clamp(0.0, 1.0),
        Field::Soil => data.soil[position],
        Field::Contour => data
            .contour_points
            .get(&position)
            .copied()
            .unwrap_or_default() as f32
            / 255.0,
        Field::SeaDistance => data.sea_distance[position].clamp(0.0, 1.0),
        Field::Plants => data.plants[position],
    }
}

fn field_range(field: Option<Field>, data: &RenderData) -> (f32, f32) {
    if field.is_none() {
        return (0.0, 1.0);
    }
    let field = field.unwrap();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (position, _) in data.world.heightmap.iter() {
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

fn build_derived_data(
    world: GeneratedWorld,
    ecs_world: &World,
    dynamic_groundwater: &Map<f32>,
    dynamic_soil_nutrients: &Map<f32>,
) -> RenderData {
    let mut heightmap_for_water = world.heightmap.clone();

    let (sinks, sea) = get_sea(&world.heightmap);
    let (rain, _temperature_sim, _atmo_humidity_sim) =
        simulate_air(&world.heightmap, &sea, CoordVec::new(0, -1), CoordVec::new(1, 0));
    let (water, _salt_sim) = simulate_water(&mut heightmap_for_water, &rain, &sea, &sinks);

    let mut sea_distance = distance_map(world.radius, sea.iter().copied());
    let radius_f = world.radius as f32;
    for (_, value) in sea_distance.iter_mut() {
        *value = (*value / radius_f).clamp(0.0, 1.0);
    }

    let contours = generate_contours(&world.heightmap, 0.15);
    let mut contour_points = HashMap::new();
    for (point, x1, x2, _) in contours {
        let steepness = x1 - x2;
        let entry = contour_points.entry(point).or_default();
        *entry = std::cmp::max(*entry, (steepness * 4000.0).abs() as u8);
    }

    let groundwater = Map::<f32>::from_fn(
        |coord| world.groundwater[coord] + dynamic_groundwater[coord],
        world.radius,
    );
    let soil = Map::<f32>::from_fn(
        |coord| world.soil_nutrients[coord] + dynamic_soil_nutrients[coord],
        world.radius,
    );

    let mut plants = Map::new(world.radius, 0.0f32);
    for (position, plant) in ecs_world.query::<(&Position, &Plant)>().iter() {
        let pos = position.head();
        if !plants.in_range(pos) {
            continue;
        }
        let g = plant.genome.base_growth_rate(
            soil[pos],
            groundwater[pos],
            world.temperature[pos],
            world.salt[pos],
            &world.get_terrain(pos),
        );
        if g > plants[pos] {
            plants[pos] = g;
        }
    }

    RenderData {
        world,
        rain,
        water,
        groundwater,
        soil,
        contour_points,
        sea_distance,
        plants,
    }
}

fn load_render_data(args: &Args) -> Result<RenderData> {
    if let Some(save_path) = &args.save {
        let data = std::fs::read(save_path)?;
        let save = SavedGame::decode(&data)?;
        let ecs_world = world_serde::deserialize_world_from_bytes(&save.world)?;
        Ok(build_derived_data(
            save.map,
            &ecs_world,
            &save.dynamic_groundwater,
            &save.dynamic_soil_nutrients,
        ))
    } else {
        let radius = args.radius.max(1);
        let mut heightmap = generate_heights_with_radius(radius);
        let (sinks, sea) = get_sea(&heightmap);
        let (rain, temperature, atmo_humidity) =
            simulate_air(&heightmap, &sea, CoordVec::new(0, -1), CoordVec::new(1, 0));
        let (water, salt) = simulate_water(&mut heightmap, &rain, &sea, &sinks);
        let groundwater = compute_groundwater(&water, &rain, &heightmap);
        let soil_nutrients = soil_nutrients(&groundwater);

        let mut terrain = Map::<TerrainType>::new(heightmap.radius, TerrainType::Empty);
        for (point, _, _, _) in generate_contours(&heightmap, 0.1) {
            terrain[point] = TerrainType::Contour;
        }
        for (point, w) in water.iter() {
            if *w > 1.0 {
                terrain[point] = TerrainType::DeepWater;
            } else if *w > 0.0 {
                terrain[point] = TerrainType::ShallowWater;
            }
        }

        let world = GeneratedWorld {
            radius: heightmap.radius,
            heightmap,
            terrain,
            groundwater,
            salt,
            atmo_humidity,
            temperature,
            soil_nutrients,
        };

        let ecs_world = World::new();
        let zero = Map::new(world.radius, 0.0f32);
        Ok(build_derived_data(world, &ecs_world, &zero, &zero))
    }
}

fn mat_vec_mul(mat: &[f32], n: usize, v: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += mat[i * n + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize_vec(v: &mut [f32]) {
    let n = vec_norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn top_pca_vectors(cov: &[f32], dim: usize, k: usize) -> Vec<Vec<f32>> {
    let mut a = cov.to_vec();
    let mut out = Vec::new();
    let components = k.min(dim);

    for comp in 0..components {
        let mut v = (0..dim)
            .map(|i| 1.0 + (i + comp) as f32 * 0.013)
            .collect::<Vec<_>>();
        normalize_vec(&mut v);

        for _ in 0..64 {
            let mut next = mat_vec_mul(&a, dim, &v);
            normalize_vec(&mut next);
            v = next;
        }

        let av = mat_vec_mul(&a, dim, &v);
        let lambda = dot(&v, &av);
        if !lambda.is_finite() || lambda.abs() < 1e-8 {
            break;
        }

        for i in 0..dim {
            for j in 0..dim {
                a[i * dim + j] -= lambda * v[i] * v[j];
            }
        }

        out.push(v);
    }

    out
}

fn render_with_pca(
    image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    data: &RenderData,
    fields: &[Field],
    color_space: &str,
) -> Result<()> {
    if fields.len() < 2 {
        anyhow::bail!("PCA mode requires at least 2 fields")
    }

    let coords = data.world.heightmap.iter_coords().collect::<Vec<_>>();
    let n = coords.len();
    let m = fields.len();

    let mut samples = vec![0.0f32; n * m];
    for (i, coord) in coords.iter().copied().enumerate() {
        for (j, field) in fields.iter().copied().enumerate() {
            samples[i * m + j] = sample_field(field, coord, data);
        }
    }

    let mut means = vec![0.0f32; m];
    for j in 0..m {
        let mut acc = 0.0;
        for i in 0..n {
            acc += samples[i * m + j];
        }
        means[j] = acc / n as f32;
    }

    let mut stds = vec![0.0f32; m];
    for j in 0..m {
        let mut acc = 0.0;
        for i in 0..n {
            let d = samples[i * m + j] - means[j];
            acc += d * d;
        }
        stds[j] = (acc / (n as f32 - 1.0).max(1.0)).sqrt().max(1e-6);
    }

    // Standardize to z-scores before covariance so high-variance fields do not dominate.
    for i in 0..n {
        for j in 0..m {
            samples[i * m + j] = (samples[i * m + j] - means[j]) / stds[j];
        }
    }

    let mut cov = vec![0.0f32; m * m];
    let denom = (n as f32 - 1.0).max(1.0);
    for a in 0..m {
        for b in 0..m {
            let mut acc = 0.0;
            for i in 0..n {
                acc += samples[i * m + a] * samples[i * m + b];
            }
            cov[a * m + b] = acc / denom;
        }
    }

    let pcs = top_pca_vectors(&cov, m, 3);
    if pcs.is_empty() {
        anyhow::bail!("PCA failed to produce principal components")
    }

    for (k, pc) in pcs.iter().enumerate() {
        println!("pca_channel_{} coefficients:", k + 1);
        for (field, coeff) in fields.iter().zip(pc.iter()) {
            println!("  {:<12} {:>10.6}", field_name(*field), coeff);
        }
    }

    let mut chans = vec![vec![0.0f32; n]; 3];
    for i in 0..n {
        let row = &samples[i * m..(i + 1) * m];
        for k in 0..3 {
            chans[k][i] = if k < pcs.len() { dot(row, &pcs[k]) } else { 0.0 };
        }
    }

    let mut ranges = [(0.0f32, 1.0f32); 3];
    for k in 0..3 {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for v in chans[k].iter().copied() {
            min = min.min(v);
            max = max.max(v);
        }
        ranges[k] = if (max - min).abs() < f32::EPSILON {
            (0.0, 1.0)
        } else {
            (min, max)
        };
    }

    for (i, coord) in coords.iter().copied().enumerate() {
        let (x, y) = hex_to_image_coords(coord, data.world.radius);
        let c1 = normalize(chans[0][i], ranges[0].0, ranges[0].1);
        let c2 = normalize(chans[1][i], ranges[1].0, ranges[1].1);
        let c3 = normalize(chans[2][i], ranges[2].0, ranges[2].1);
        let rgb = to_rgb(c1, c2, c3, color_space);
        image.put_pixel(x, y, Rgb::from(rgb));
    }

    Ok(())
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    let args: Args = argh::from_env();

    let t = Instant::now();
    let data = load_render_data(&args)?;
    println!("load/build: {:.3}s", t.elapsed().as_secs_f32());

    let f1 = args.c1.as_deref().map(Field::parse).transpose()?;
    let f2 = args.c2.as_deref().map(Field::parse).transpose()?;
    let f3 = args.c3.as_deref().map(Field::parse).transpose()?;
    let pca_fields = args
        .pca_field
        .iter()
        .map(|x| Field::parse(x))
        .collect::<Result<Vec<_>>>()?;

    let image_radius = data.world.radius;
    let mut image = ImageBuffer::from_pixel(
        (image_radius * 2 + 1) as u32,
        (image_radius * 2 + 1) as u32,
        Rgb::from([0u8, 0, 0]),
    );

    let t = Instant::now();
    if !pca_fields.is_empty() {
        render_with_pca(&mut image, &data, &pca_fields, &args.color_space)?;
        println!("render mode: pca(fields={})", pca_fields.len());
    } else {
        let r1 = field_range(f1, &data);
        let r2 = field_range(f2, &data);
        let r3 = field_range(f3, &data);
        println!("ranges: {:?}, {:?}, {:?}", r1, r2, r3);

        for (position, _) in data.world.heightmap.iter() {
            let (col, row) = hex_to_image_coords(position, image_radius);
            let mut c1 = f1.map(|f| sample_field(f, position, &data)).unwrap_or(0.0);
            let mut c2 = f2.map(|f| sample_field(f, position, &data)).unwrap_or(0.0);
            let mut c3 = f3.map(|f| sample_field(f, position, &data)).unwrap_or(0.0);
            if args.normalize {
                c1 = normalize(c1, r1.0, r1.1);
                c2 = normalize(c2, r2.0, r2.1);
                c3 = normalize(c3, r3.0, r3.1);
            }
            let rgb = to_rgb(c1, c2, c3, &args.color_space);
            image.put_pixel(col, row, Rgb::from(rgb));
        }
        println!("render mode: channels");
    }
    println!("render: {:.3}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    image.save(&args.output)?;
    println!("save: {:.3}s", t.elapsed().as_secs_f32());
    println!("total: {:.3}s", total_start.elapsed().as_secs_f32());

    Ok(())
}
