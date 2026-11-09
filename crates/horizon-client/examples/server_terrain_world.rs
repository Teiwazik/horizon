//! Turns cached server far terrain (`server_terrain.rs` summaries) into plain
//! Anvil chunks: each column's spans filled with their block states and
//! biomes, light left for the game to compute. A dev tool for profiling far
//! terrain on a world that looks like a server's.
//!
//! `cargo run --profile fast -p horizon-client --example server_terrain_world --
//! <resources> <summary level dir> <world region dir>`
extern crate horizon_client_lib as horizon_client;
use horizon_client::{biome::Biomes, chunk::Chunk, save};
use horizon_core::position::ChunkPos;
use horizon_world::{
    compression::Compression,
    nbt::{Limits, Tag},
    region::{ChunkRecord, RegionFile},
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (root, src, dst) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    let registry = horizon_world::level::data::load_registry(root)?;
    let biomes = Biomes::load(root).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::create_dir_all(dst)?;
    let (mut count, mut bounds) = (0u64, [i32::MAX, i32::MAX, i32::MIN, i32::MIN]);
    let mut heights: Vec<(i32, i32, i32)> = Vec::new();
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "mca") {
            continue;
        }
        let mut input = RegionFile::open(&path)?;
        let positions: Vec<ChunkPos> = input.chunks().map(|(p, _)| p).collect();
        let mut output = RegionFile::open_writable(dst.join(path.file_name().unwrap()), true)?;
        for pos in positions {
            let Some(record) = input.read(pos)? else {
                continue;
            };
            let tag = record.nbt(Limits::default())?.tag;
            let Some(Tag::IntArray(ints)) = tag.get("columns") else {
                continue;
            };
            let Some(chunk) = build(pos, ints, registry.len() as u32, &biomes, &mut heights) else {
                eprintln!("bad record {pos:?}");
                continue;
            };
            let named = save::encode(&chunk, &registry, &biomes);
            let out = ChunkRecord::from_nbt(&named, Compression::Zlib, 0, Limits::default())?;
            output.write(pos, &out)?;
            count += 1;
            bounds = [
                bounds[0].min(pos.x),
                bounds[1].min(pos.z),
                bounds[2].max(pos.x),
                bounds[3].max(pos.z),
            ];
        }
        output.sync()?;
    }
    heights.sort_unstable();
    let mid = heights.get(heights.len() / 2).copied().unwrap_or_default();
    println!(
        "chunks {count} bounds chunk x {}..{} z {}..{}",
        bounds[0], bounds[2], bounds[1], bounds[3]
    );
    println!("median chunk (x z top) {mid:?}");
    Ok(())
}

fn build(
    pos: ChunkPos,
    ints: &[i32],
    states: u32,
    _biomes: &Biomes,
    heights: &mut Vec<(i32, i32, i32)>,
) -> Option<Chunk> {
    let mut chunk = Chunk::empty(pos.x, pos.z);
    let (min_y, max_y) = (
        chunk.min_y(),
        chunk.min_y() + chunk.sections.len() as i32 * 16,
    );
    let mut it = ints.iter().copied();
    let mut top = min_y;
    for i in 0..256 {
        let (x, z) = ((i & 15) as i32, (i >> 4) as i32);
        if it.next()? == 1 {
            for _ in 0..6 {
                it.next()?;
            }
        }
        let n = it.next()?;
        for _ in 0..n {
            let range = it.next()?;
            let state = it.next()? as u32;
            let material = it.next()?;
            let (lo, hi) = (i32::from((range >> 16) as i16), i32::from(range as i16));
            let biome = (material >> 16) as u16;
            if state >= states {
                continue;
            }
            for y in lo.max(min_y)..hi.min(max_y) {
                chunk.set(x, y, z, state);
                if x & 3 == 0 && z & 3 == 0 && y & 3 == 0 {
                    chunk.set_biome_cell(x >> 2, (y - min_y) >> 2, z >> 2, biome);
                }
            }
            top = top.max(hi);
        }
    }
    heights.push((pos.x, pos.z, top));
    Some(chunk)
}
