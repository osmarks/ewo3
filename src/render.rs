#![feature(test)]
extern crate test;

use anyhow::{anyhow, Context, Result};
use argh::FromArgs;
use ewo3::components::{Health, MapLayer, Plant, Position, PositionIndex, Render};
use ewo3::map::*;
use ewo3::save::SavedGame;
use ewo3::world_serde;
use ewo3::worldgen::*;
use glow::HasContext;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, WindowSurface};
use imgui::Condition;
use imgui_glow_renderer::{AutoRenderer as ImguiRenderer, TextureMap};
use imgui_winit_support::winit::window::WindowBuilder;
use hecs::World;
use image::{ImageBuffer, Rgb};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoopBuilder;
use raw_window_handle::HasRawWindowHandle;

#[derive(FromArgs)]
/// Render world/debug fields from generated terrain or a saved game.
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
    /// open an interactive imgui window instead of writing PNG
    #[argh(switch)]
    window: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

const ALL_FIELDS: [Field; 11] = [
    Field::Height,
    Field::Rain,
    Field::Water,
    Field::Groundwater,
    Field::Salt,
    Field::Temperature,
    Field::Humidity,
    Field::Soil,
    Field::Contour,
    Field::SeaDistance,
    Field::Plants,
];

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

    fn name(self) -> &'static str {
        match self {
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorSpace {
    Rgb,
    Oklab,
}

impl ColorSpace {
    fn parse(s: &str) -> Self {
        match s {
            "oklab" => Self::Oklab,
            _ => Self::Rgb,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ColorSpace::Rgb => "rgb",
            ColorSpace::Oklab => "oklab",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct RenderSettings {
    c1: Option<Field>,
    c2: Option<Field>,
    c3: Option<Field>,
    normalize: bool,
    color_space: ColorSpace,
    pca_fields: Vec<Field>,
}

struct RenderData {
    world: GeneratedWorld,
    ecs_world: World,
    positions: PositionIndex,
    groundwater: Map<f32>,
    soil: Map<f32>,
    contour_points: HashMap<Coord, u8>,
    sea_distance: Map<f32>,
    plants: Map<f32>,
}

struct RenderedImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    pca_coefficients: Vec<Vec<(Field, f32)>>,
}

fn hex_to_image_coords(pos: Coord, radius: i32) -> (u32, u32) {
    let col = pos.x + (pos.y - (pos.y & 1)) / 2 + radius;
    let row = pos.y + radius;
    (col as u32, row as u32)
}

fn normalize(v: f32, min: f32, max: f32) -> f32 {
    if (max - min).abs() < f32::EPSILON {
        0.5
    } else {
        ((v - min) / (max - min)).clamp(0.0, 1.0)
    }
}

fn sample_field(field: Field, position: Coord, data: &RenderData) -> f32 {
    match field {
        Field::Height => ((data.world.heightmap[position] + 1.0) * 0.5).clamp(0.0, 1.0),
        Field::Rain => data.world.rain[position].clamp(0.0, 1.0),
        Field::Water => data.world.water[position].min(1.0),
        Field::Groundwater => data.groundwater[position],
        Field::Salt => data.world.salt[position].clamp(0.0, 1.0),
        Field::Temperature => data.world.temperature[position].clamp(0.0, 1.0),
        Field::Humidity => data.world.atmo_humidity[position].clamp(0.0, 1.0),
        Field::Soil => data.soil[position],
        Field::Contour => data.contour_points.get(&position).copied().unwrap_or_default() as f32 / 255.0,
        Field::SeaDistance => data.sea_distance[position].clamp(0.0, 1.0),
        Field::Plants => data.plants[position],
    }
}

fn field_range(field: Option<Field>, data: &RenderData) -> (f32, f32) {
    let Some(field) = field else { return (0.0, 1.0) };
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

fn to_rgb(c1: f32, c2: f32, c3: f32, color_space: ColorSpace) -> [u8; 3] {
    fn linear_to_srgb(x: f32) -> f32 {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    }

    let (r, g, b) = match color_space {
        ColorSpace::Rgb => (c1, c2, c3),
        ColorSpace::Oklab => {
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
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn build_derived_data(
    world: &GeneratedWorld,
    ecs_world: &World,
    dynamic_groundwater: &Map<f32>,
    dynamic_soil_nutrients: &Map<f32>,
) -> (Map<f32>, Map<f32>, Map<f32>, Map<f32>, HashMap<Coord, u8>) {
    let (sinks, sea) = get_sea(&world.heightmap);

    let mut sea_distance = distance_map(world.radius, sea.iter().copied());
    let radius_f = world.radius as f32;
    for (_, value) in sea_distance.iter_mut() {
        *value = (*value / radius_f).clamp(0.0, 1.0);
    }

    let contours = generate_contours(&world.heightmap);
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

    (
        sea_distance,
        groundwater,
        soil,
        plants,
        contour_points
    )
}

fn rebuild_position_index(world: &World, radius: i32) -> PositionIndex {
    let mut index = PositionIndex::new(radius);
    for (entity, position) in world.query::<(hecs::Entity, &Position)>().iter() {
        let mut pos = position.clone();
        pos.record_for(&mut index, Some(entity));
    }
    index
}

fn load_render_data(args: &Args) -> Result<RenderData> {
    if let Some(save_path) = &args.save {
        let data = std::fs::read(save_path)?;
        let save = SavedGame::decode(&data)?;
        let ecs_world = world_serde::deserialize_world_from_bytes(&save.world)?;
        let positions = rebuild_position_index(&ecs_world, save.map.radius);
        let world = save.map;
        let (sea_distance, groundwater, soil, plants, contour_points) = build_derived_data(
            &world,
            &ecs_world,
            &save.dynamic_groundwater,
            &save.dynamic_soil_nutrients,
        );
        Ok(RenderData {
            world,
            ecs_world,
            positions,
            groundwater,
            soil,
            contour_points,
            sea_distance,
            plants,
        })
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
        for (point, _, _, _) in generate_contours(&heightmap) {
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
            rain,
            water
        };

        let ecs_world = World::new();
        let positions = PositionIndex::new(world.radius);
        let zero = Map::new(world.radius, 0.0f32);
        let (sea_distance, groundwater, soil, plants, contour_points) = build_derived_data(&world, &ecs_world, &zero, &zero);
        Ok(RenderData {
            world,
            ecs_world,
            positions,
            groundwater,
            soil,
            contour_points,
            sea_distance,
            plants,
        })
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

fn render_pca(data: &RenderData, fields: &[Field], color_space: ColorSpace) -> Result<RenderedImage> {
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

    let mut coeffs = Vec::new();
    for pc in pcs.iter() {
        let mut channel_coeffs = Vec::new();
        for (field, coeff) in fields.iter().zip(pc.iter()) {
            channel_coeffs.push((*field, *coeff));
        }
        coeffs.push(channel_coeffs);
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

    let side = (data.world.radius * 2 + 1) as u32;
    let mut pixels = vec![0u8; (side * side * 3) as usize];
    for (i, coord) in coords.iter().copied().enumerate() {
        let (x, y) = hex_to_image_coords(coord, data.world.radius);
        let c1 = normalize(chans[0][i], ranges[0].0, ranges[0].1);
        let c2 = normalize(chans[1][i], ranges[1].0, ranges[1].1);
        let c3 = normalize(chans[2][i], ranges[2].0, ranges[2].1);
        let rgb = to_rgb(c1, c2, c3, color_space);
        let idx = ((y * side + x) * 3) as usize;
        pixels[idx] = rgb[0];
        pixels[idx + 1] = rgb[1];
        pixels[idx + 2] = rgb[2];
    }

    Ok(RenderedImage {
        width: side,
        height: side,
        pixels,
        pca_coefficients: coeffs,
    })
}

fn render_channels(data: &RenderData, settings: &RenderSettings) -> RenderedImage {
    let r1 = field_range(settings.c1, data);
    let r2 = field_range(settings.c2, data);
    let r3 = field_range(settings.c3, data);

    let side = (data.world.radius * 2 + 1) as u32;
    let mut pixels = vec![0u8; (side * side * 3) as usize];

    for (position, _) in data.world.heightmap.iter() {
        let (x, y) = hex_to_image_coords(position, data.world.radius);
        let mut c1 = settings.c1.map(|f| sample_field(f, position, data)).unwrap_or(0.0);
        let mut c2 = settings.c2.map(|f| sample_field(f, position, data)).unwrap_or(0.0);
        let mut c3 = settings.c3.map(|f| sample_field(f, position, data)).unwrap_or(0.0);
        if settings.normalize {
            c1 = normalize(c1, r1.0, r1.1);
            c2 = normalize(c2, r2.0, r2.1);
            c3 = normalize(c3, r3.0, r3.1);
        }
        let rgb = to_rgb(c1, c2, c3, settings.color_space);
        let idx = ((y * side + x) * 3) as usize;
        pixels[idx] = rgb[0];
        pixels[idx + 1] = rgb[1];
        pixels[idx + 2] = rgb[2];
    }

    RenderedImage {
        width: side,
        height: side,
        pixels,
        pca_coefficients: Vec::new(),
    }
}

fn render_image(data: &RenderData, settings: &RenderSettings) -> Result<RenderedImage> {
    if settings.pca_fields.len() >= 2 {
        render_pca(data, &settings.pca_fields, settings.color_space)
    } else {
        Ok(render_channels(data, settings))
    }
}

fn save_rendered_png(image: &RenderedImage, path: &str) -> Result<()> {
    let mut out = ImageBuffer::from_pixel(image.width, image.height, Rgb([0u8, 0, 0]));
    for y in 0..image.height {
        for x in 0..image.width {
            let idx = ((y * image.width + x) * 3) as usize;
            out.put_pixel(x, y, Rgb([image.pixels[idx], image.pixels[idx + 1], image.pixels[idx + 2]]));
        }
    }
    out.save(path)?;
    Ok(())
}

fn print_pca_coeffs(coeffs: &[Vec<(Field, f32)>]) {
    for (k, channel) in coeffs.iter().enumerate() {
        println!("pca_channel_{} coefficients:", k + 1);
        for (field, coeff) in channel.iter() {
            println!("  {:<12} {:>10.6}", field.name(), coeff);
        }
    }
}

fn parse_settings(args: &Args) -> Result<RenderSettings> {
    Ok(RenderSettings {
        c1: args.c1.as_deref().map(Field::parse).transpose()?,
        c2: args.c2.as_deref().map(Field::parse).transpose()?,
        c3: args.c3.as_deref().map(Field::parse).transpose()?,
        normalize: args.normalize,
        color_space: ColorSpace::parse(&args.color_space),
        pca_fields: args
            .pca_field
            .iter()
            .map(|x| Field::parse(x))
            .collect::<Result<Vec<_>>>()?,
    })
}

fn choose_field_combo(ui: &imgui::Ui, label: &str, current: &mut Option<Field>) {
    let mut labels = vec!["<none>"];
    for f in ALL_FIELDS {
        labels.push(f.name());
    }

    let mut idx = match current {
        None => 0,
        Some(f) => 1 + ALL_FIELDS.iter().position(|x| x == f).unwrap_or(0),
    };

    if ui.combo_simple_string(label, &mut idx, &labels) {
        *current = if idx == 0 { None } else { Some(ALL_FIELDS[idx - 1]) };
    }
}

fn inspect_entity(data: &RenderData, entity: hecs::Entity, layer: MapLayer, ui: &imgui::Ui) {
    ui.text(format!("entity {:?} layer={:?}", entity, layer));

    if let Ok(render) = data.ecs_world.get::<&Render>(entity) {
        ui.text(format!("  Render: {}", render.0));
    }
    if let Ok(health) = data.ecs_world.get::<&Health>(entity) {
        ui.text(format!(
            "  Health: current={:.3} max={:.3}",
            health.current, health.max
        ));
        ui.text(format!(
            "  Health totals: {:?}",
            health.health_change_totals
        ));
        ui.text(format!(
            "  Health modifiers: {:?}",
            health.health_change_modifiers
        ));
    }
    if let Ok(plant) = data.ecs_world.get::<&Plant>(entity) {
        ui.text(format!(
            "  Plant: size={:.4} growth_ticks={} children={} ready_ticks={}",
            plant.current_size, plant.growth_ticks, plant.children, plant.ready_for_reproduce_ticks
        ));
        ui.text(format!(
            "  Plant: nutrients_consumed={:.4} nutrients_added={:.4} water_consumed={:.4}",
            plant.nutrients_consumed, plant.nutrients_added, plant.water_consumed
        ));
        ui.text(format!(
            "  Plant genome: {:?}",
            plant.genome
        ));
    }
}

fn create_window(
) -> Result<(
    winit::event_loop::EventLoop<()>,
    winit::window::Window,
    Surface<WindowSurface>,
    PossiblyCurrentContext,
)> {
    let event_loop = EventLoopBuilder::new()
        .build()
        .map_err(|e| anyhow!("event loop error: {e}"))?;

    let window_builder = WindowBuilder::new()
        .with_title("World Render Debug")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 960.0));
    let (window, cfg) = glutin_winit::DisplayBuilder::new()
        .with_window_builder(Some(window_builder))
        .build(&event_loop, ConfigTemplateBuilder::new(), |mut configs| {
            configs.next().expect("no GL config")
        })
        .map_err(|e| anyhow!("failed to create OpenGL window: {e}"))?;
    let window = window.expect("window is required");

    let context_attribs = ContextAttributesBuilder::new().build(Some(window.raw_window_handle()));
    let context = unsafe {
        cfg.display()
            .create_context(&cfg, &context_attribs)
            .map_err(|e| anyhow!("failed to create OpenGL context: {e}"))?
    };

    let size = window.inner_size();
    let surface_attribs = SurfaceAttributesBuilder::<WindowSurface>::new()
        .with_srgb(Some(true))
        .build(
            window.raw_window_handle(),
            NonZeroU32::new(size.width.max(1)).unwrap(),
            NonZeroU32::new(size.height.max(1)).unwrap(),
        );
    let surface = unsafe {
        cfg.display()
            .create_window_surface(&cfg, &surface_attribs)
            .map_err(|e| anyhow!("failed to create OpenGL surface: {e}"))?
    };

    let context = context
        .make_current(&surface)
        .map_err(|e| anyhow!("failed to make OpenGL context current: {e}"))?;

    Ok((event_loop, window, surface, context))
}

fn glow_context(context: &PossiblyCurrentContext) -> glow::Context {
    unsafe {
        glow::Context::from_loader_function_cstr(|s| context.display().get_proc_address(s).cast())
    }
}

fn run_window(data: RenderData, mut settings: RenderSettings) -> Result<()> {
    let (event_loop, window, surface, context) = create_window()?;
    let gl = glow_context(&context);

    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

    let mut renderer = ImguiRenderer::initialize(gl, &mut imgui)
        .map_err(|e| anyhow!("imgui renderer init: {e}"))?;

    let mut last_frame = Instant::now();
    let mut map_texture: Option<glow::Texture> = None;
    let mut map_texture_id: Option<imgui::TextureId> = None;
    let mut last_settings: Option<RenderSettings> = None;
    let mut cached_frame: Option<RenderedImage> = None;
    let mut hovered_coord: Option<Coord> = None;
    let mut selected_coord: Option<Coord> = None;

    event_loop.run(move |event, window_target| {
            match event {
                Event::NewEvents(_) => {
                    imgui.io_mut().update_delta_time(last_frame.elapsed());
                    last_frame = Instant::now();
                }
                Event::AboutToWait => {
                    if let Err(e) = platform.prepare_frame(imgui.io_mut(), &window) {
                        eprintln!("prepare_frame failed: {e}");
                        window_target.exit();
                        return;
                    }
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    window_target.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let ui = imgui.frame();

                    ui.window("Controls")
                        .position([20.0, 20.0], Condition::FirstUseEver)
                        .size([420.0, 420.0], Condition::FirstUseEver)
                        .build(|| {
                            ui.text(format!("radius: {}", data.world.radius));

                            let mut pca_mode = !settings.pca_fields.is_empty();
                            ui.checkbox("PCA mode", &mut pca_mode);

                            let mut normalize = settings.normalize;
                            if ui.checkbox("Normalize channels", &mut normalize) {
                                settings.normalize = normalize;
                            }

                            let mut cs_idx = if settings.color_space == ColorSpace::Rgb { 0 } else { 1 };
                            let cs_labels = ["rgb", "oklab"];
                            if ui.combo_simple_string("Color space", &mut cs_idx, &cs_labels) {
                                settings.color_space = if cs_idx == 0 { ColorSpace::Rgb } else { ColorSpace::Oklab };
                            }

                            if pca_mode {
                                if settings.pca_fields.is_empty() {
                                    settings.pca_fields = vec![Field::Groundwater, Field::Soil, Field::Plants];
                                }
                                ui.text("PCA fields:");
                                for field in ALL_FIELDS {
                                    let mut enabled = settings.pca_fields.contains(&field);
                                    if ui.checkbox(field.name(), &mut enabled) {
                                        if enabled {
                                            settings.pca_fields.push(field);
                                        } else {
                                            settings.pca_fields.retain(|x| *x != field);
                                        }
                                    }
                                }
                                if settings.pca_fields.len() < 2 {
                                    ui.text_colored(
                                        [1.0, 0.6, 0.4, 1.0],
                                        "Select at least 2 PCA fields (falling back to channel mode).",
                                    );
                                }
                            } else {
                                settings.pca_fields.clear();
                                choose_field_combo(ui, "c1", &mut settings.c1);
                                choose_field_combo(ui, "c2", &mut settings.c2);
                                choose_field_combo(ui, "c3", &mut settings.c3);
                            }
                        });

                    let settings_changed = last_settings.as_ref() != Some(&settings);
                    if settings_changed || cached_frame.is_none() || map_texture.is_none() {
                        let frame = match render_image(&data, &settings) {
                            Ok(x) => x,
                            Err(e) => {
                                eprintln!("render_image failed: {e:#}");
                                window_target.exit();
                                return;
                            }
                        };
                        cached_frame = Some(frame);
                        last_settings = Some(settings.clone());

                        if let Some(old_tex) = map_texture.take() {
                            unsafe { renderer.gl_context().delete_texture(old_tex) };
                        }
                        map_texture_id = None;

                        let tex = match unsafe { renderer.gl_context().create_texture() } {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("texture creation failed: {e}");
                                window_target.exit();
                                return;
                            }
                        };

                        let frame_ref = cached_frame.as_ref().expect("cached frame");
                        unsafe {
                            let gl = renderer.gl_context();
                            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                            gl.tex_parameter_i32(
                                glow::TEXTURE_2D,
                                glow::TEXTURE_MIN_FILTER,
                                glow::LINEAR as i32,
                            );
                            gl.tex_parameter_i32(
                                glow::TEXTURE_2D,
                                glow::TEXTURE_MAG_FILTER,
                                glow::LINEAR as i32,
                            );
                            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                            gl.tex_image_2d(
                                glow::TEXTURE_2D,
                                0,
                                glow::RGB as i32,
                                frame_ref.width as i32,
                                frame_ref.height as i32,
                                0,
                                glow::RGB,
                                glow::UNSIGNED_BYTE,
                                Some(&frame_ref.pixels),
                            );
                        }

                        let tid = renderer
                            .texture_map_mut()
                            .register(tex)
                            .expect("texture registration failed");
                        map_texture = Some(tex);
                        map_texture_id = Some(tid);
                    }

                    let frame = cached_frame.as_ref().expect("cached frame");
                    let tid = map_texture_id.expect("texture id");

                    ui.window("Map")
                        .position([460.0, 20.0], Condition::FirstUseEver)
                        .size(
                            [frame.width as f32 + 24.0, frame.height as f32 + 42.0],
                            Condition::FirstUseEver,
                        )
                        .scroll_bar(false)
                        .scrollable(false)
                        .build(|| {
                            imgui::Image::new(tid, [frame.width as f32, frame.height as f32]).build(ui);

                            hovered_coord = None;
                            if ui.is_item_hovered() {
                                let mouse = ui.io().mouse_pos;
                                let min = ui.item_rect_min();
                                let max = ui.item_rect_max();
                                let tx = ((mouse[0] - min[0]) / (max[0] - min[0]) * frame.width as f32)
                                    .floor()
                                    .clamp(0.0, frame.width as f32 - 1.0) as i32;
                                let ty = ((mouse[1] - min[1]) / (max[1] - min[1]) * frame.height as f32)
                                    .floor()
                                    .clamp(0.0, frame.height as f32 - 1.0) as i32;
                                let y = ty - data.world.radius;
                                let x = tx - data.world.radius - (y - (y & 1)) / 2;
                                let coord = Coord::new(x, y);
                                if data.world.heightmap.in_range(coord) {
                                    hovered_coord = Some(coord);
                                    if ui.is_mouse_clicked(imgui::MouseButton::Left) {
                                        selected_coord = Some(coord);
                                    }
                                }
                                if ui.is_mouse_clicked(imgui::MouseButton::Right) {
                                    selected_coord = None;
                                }
                            }
                        });

                    ui.window("Inspector")
                        .position([20.0, 460.0], Condition::FirstUseEver)
                        .size([420.0, 460.0], Condition::FirstUseEver)
                        .build(|| {
                            ui.text("Field ranges:");
                            if settings.pca_fields.len() >= 2 {
                                for f in settings.pca_fields.iter().copied() {
                                    let r = field_range(Some(f), &data);
                                    ui.text(format!("{}: [{:.4}, {:.4}]", f.name(), r.0, r.1));
                                }
                                if !frame.pca_coefficients.is_empty() {
                                    ui.separator();
                                    ui.text("PCA coefficients:");
                                    for (k, coeffs) in frame.pca_coefficients.iter().enumerate() {
                                        ui.text(format!("channel {}", k + 1));
                                        for (field, coeff) in coeffs.iter() {
                                            ui.text(format!("  {:<12} {:>10.6}", field.name(), coeff));
                                        }
                                    }
                                }
                            } else {
                                let r1 = field_range(settings.c1, &data);
                                let r2 = field_range(settings.c2, &data);
                                let r3 = field_range(settings.c3, &data);
                                ui.text(format!(
                                    "c1 {}: [{:.4}, {:.4}]",
                                    settings.c1.map(|f| f.name()).unwrap_or("<none>"),
                                    r1.0,
                                    r1.1
                                ));
                                ui.text(format!(
                                    "c2 {}: [{:.4}, {:.4}]",
                                    settings.c2.map(|f| f.name()).unwrap_or("<none>"),
                                    r2.0,
                                    r2.1
                                ));
                                ui.text(format!(
                                    "c3 {}: [{:.4}, {:.4}]",
                                    settings.c3.map(|f| f.name()).unwrap_or("<none>"),
                                    r3.0,
                                    r3.1
                                ));
                            }

                            ui.separator();
                            let inspect_coord = selected_coord.or(hovered_coord);
                            if let Some(coord) = inspect_coord {
                                if selected_coord.is_some() {
                                    ui.text(format!("selected: {:?}", coord));
                                    ui.text("right-click map to clear");
                                } else {
                                    ui.text(format!("hover: {:?}", coord));
                                    ui.text("left-click map to lock");
                                }
                                for f in ALL_FIELDS {
                                    ui.text(format!(
                                        "{:<12} {:>10.6}",
                                        f.name(),
                                        sample_field(f, coord, &data)
                                    ));
                                }
                                ui.separator();
                                ui.text("Entities at tile:");
                                let mut any = false;
                                if let Some(entity) = data.positions.particles[coord] {
                                    any = true;
                                    inspect_entity(&data, entity, MapLayer::Particles, ui);
                                }
                                if let Some(entity) = data.positions.entities[coord] {
                                    any = true;
                                    inspect_entity(&data, entity, MapLayer::Entities, ui);
                                }
                                if let Some(entity) = data.positions.terrain[coord] {
                                    any = true;
                                    inspect_entity(&data, entity, MapLayer::Terrain, ui);
                                }
                                if !any {
                                    ui.text("  (none)");
                                }
                            } else {
                                ui.text("hover map to inspect values");
                                ui.text("left-click map to lock selection");
                            }
                        });

                    platform.prepare_render(ui, &window);
                    let draw_data = imgui.render();

                    unsafe {
                        renderer.gl_context().clear_color(0.05, 0.05, 0.08, 1.0);
                        renderer.gl_context().clear(glow::COLOR_BUFFER_BIT);
                    }
                    if let Err(e) = renderer.render(draw_data) {
                        eprintln!("imgui render failed: {e}");
                        window_target.exit();
                        return;
                    }
                    if let Err(e) = surface.swap_buffers(&context) {
                        eprintln!("swap failed: {e}");
                        window_target.exit();
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(new_size),
                    ..
                } => {
                    if new_size.width > 0 && new_size.height > 0 {
                        surface.resize(
                            &context,
                            NonZeroU32::new(new_size.width).unwrap(),
                            NonZeroU32::new(new_size.height).unwrap(),
                        );
                    }
                }
                Event::LoopExiting => {
                    if let Some(old_tex) = map_texture.take() {
                        unsafe { renderer.gl_context().delete_texture(old_tex) };
                    }
                }
                _ => (),
            }
            platform.handle_event(imgui.io_mut(), &window, &event);
        })
        .context("event loop run failed")?;
    Ok(())
}

fn main() -> Result<()> {
    let total_start = Instant::now();
    let args: Args = argh::from_env();

    let t = Instant::now();
    let data = load_render_data(&args)?;
    println!("load/build: {:.3}s", t.elapsed().as_secs_f32());

    let settings = parse_settings(&args)?;

    if args.window {
        return run_window(data, settings);
    }

    let t = Instant::now();
    let image = render_image(&data, &settings)?;
    println!("render: {:.3}s", t.elapsed().as_secs_f32());

    if !image.pca_coefficients.is_empty() {
        print_pca_coeffs(&image.pca_coefficients);
    }

    let t = Instant::now();
    save_rendered_png(&image, &args.output)?;
    println!("save: {:.3}s", t.elapsed().as_secs_f32());
    println!("color_space: {}", settings.color_space.name());
    println!("total: {:.3}s", total_start.elapsed().as_secs_f32());

    Ok(())
}
