//! Block light for Horizon's shader packs (`[[lights]]`, docs/api/shaders.md): a grid of
//! one-block cells around the camera, which the material stage reads through the prelude's
//! `block_light_color`. Each cell holds the colour of the light sources that reach it,
//! averaged by how near each is, and how much of their light gets there: spread from each
//! source as the crow flies (a step to an edge or corner neighbour costs its length), not
//! through the blocks that stop light, so it falls off round and smooth, without the light
//! engine's diamonds and steps.
//!
//! The grid is rebuilt on a thread when the camera moves on to another 16-block step or the
//! sources or walls change, from what the section meshes list (`SectionMesh::lights`,
//! `SectionMesh::light_walls`).

use crate::fast_hash::FxHashMap;
use glam::{DVec3, Vec3};

use super::SectionKey;

/// Cells along each side of the grid.
pub const SIZE: u32 = 128;
/// The side of a cell, in blocks.
pub const CELL: f32 = 1.0;
/// Blocks past each side of the grid whose sources still reach into it.
const MARGIN: i32 = 15;
/// The colours are averaged on a coarser grid, cells of this many blocks.
const COLOR_CELL: i32 = 2;
/// A light level, in the units the light spreads in: steps cost 16, 23 and 28 of them.
const UNIT: u8 = 16;
/// The grid moves in steps of this many blocks.
const STEP: i32 = 16;
/// The most sources one rebuild spreads, the nearest first (a lava lake lists thousands).
const MAX_SOURCES: usize = 8192;
/// The most whose colours it averages.
const MAX_COLORED: usize = 2048;
/// `[[lights]]` colours go up to 4; the cells keep them over 4.
const COLOR_RANGE: f32 = 4.0;

/// The light a block state gives off: its `[[lights]]` colour and its level, 1 to 15.
pub type LightTable = FxHashMap<u32, ([f32; 3], u8)>;

pub struct BlockLight {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// `block_light_grid`: the grid's corner, camera-relative, and the side of a cell.
    pub grid: wgpu::Buffer,
    table: LightTable,
    /// The world corner and the sources' revision the texture was last asked for.
    built: Option<([i32; 3], u64)>,
    /// The grid being filled on a thread of its own (its corner): with sources all over new
    /// ground it took 20 to 40 ms, a hitch every few frames in flight.
    pending: Option<([i32; 3], std::thread::JoinHandle<Vec<u16>>)>,
    /// The corner of the grid the texture holds.
    shown: Option<[i32; 3]>,
}

impl BlockLight {
    pub fn new(device: &wgpu::Device, table: LightTable) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Native shader pack block light colours"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Native shader pack block light grid"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            view: texture.create_view(&Default::default()),
            texture,
            grid,
            table,
            built: None,
            pending: None,
            shown: None,
        }
    }

    /// The light each block state gives off.
    pub fn table(&self) -> &LightTable {
        &self.table
    }

    /// Places the grid around `camera`, rebuilding it when it moved or `revision` (of
    /// `sections`' sources) changed.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        camera: DVec3,
        sections: &FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
        walls: &FxHashMap<SectionKey, Box<[u16; 256]>>,
        revision: u64,
    ) {
        let half = (SIZE as f32 * CELL / 2.0) as f64;
        let origin = ((camera - half) / f64::from(STEP))
            .floor()
            .as_ivec3()
            .to_array()
            .map(|v| v * STEP);
        // A grid filled since: into the texture, which then stands at its corner.
        if self.pending.as_ref().is_some_and(|(_, job)| job.is_finished())
            && let Some((at, job)) = self.pending.take()
        {
            if let Ok(cells) = job.join() {
                queue.write_texture(
                    self.texture.as_image_copy(),
                    bytemuck::cast_slice(&cells),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(SIZE * 8),
                        rows_per_image: Some(SIZE),
                    },
                    self.texture.size(),
                );
                self.shown = Some(at);
            }
        }
        // The next one, once the last is in: the sources and walls are gathered here (a walk
        // over the sections' lists), their light spread over the cells on a thread.
        if self.built != Some((origin, revision)) && self.pending.is_none() {
            self.built = Some((origin, revision));
            let sources = sources(&self.table, origin, camera, sections);
            let walls = near_walls(origin, walls);
            self.pending = Some((origin, std::thread::spawn(move || build_cells(&sources, &walls))));
        }
        let at = self.shown.unwrap_or(origin);
        let corner = (glam::IVec3::from(at).as_dvec3() - camera).as_vec3();
        queue.write_buffer(
            &self.grid,
            0,
            bytemuck::cast_slice(&[corner.x, corner.y, corner.z, CELL]),
        );
    }
}

/// The cells of the grid with its corner at `origin`.
#[cfg(test)]
fn build(
    table: &LightTable,
    origin: [i32; 3],
    camera: DVec3,
    sections: &FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
    walls: &FxHashMap<SectionKey, Box<[u16; 256]>>,
) -> Vec<u16> {
    build_cells(
        &sources(table, origin, camera, sections),
        &near_walls(origin, walls),
    )
}

/// A source in the grid: its block (from the corner of the grid), its colour and its level.
type Source = (glam::IVec3, [f32; 3], u8);

/// The walls of the sections the grid with its corner at `origin` or its margin touches:
/// each section's corner (from the grid's) and its rows of cells light can't pass.
type Walls = Vec<(glam::IVec3, [u16; 256])>;

/// The sections' walls in reach of the grid with its corner at `origin`.
fn near_walls(origin: [i32; 3], walls: &FxHashMap<SectionKey, Box<[u16; 256]>>) -> Walls {
    let span = SIZE as i32;
    let lo = origin.map(|v| (v - MARGIN).div_euclid(16));
    let hi = origin.map(|v| (v + span + MARGIN - 1).div_euclid(16));
    walls
        .iter()
        .filter_map(|(key, rows)| {
            let key = [key.0, key.1 + crate::chunk::MIN_Y.div_euclid(16), key.2];
            (0..3)
                .all(|i| key[i] >= lo[i] && key[i] <= hi[i])
                .then(|| (glam::IVec3::from(key) * 16 - glam::IVec3::from(origin), **rows))
        })
        .collect()
}

/// The sources the grid with its corner at `origin` or its margin holds, the nearest
/// `MAX_SOURCES`, the nearest first `MAX_COLORED` of them.
fn sources(
    table: &LightTable,
    origin: [i32; 3],
    camera: DVec3,
    sections: &FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
) -> Vec<Source> {
    if table.is_empty() {
        return Vec::new();
    }
    let span = SIZE as i32;
    let lo = origin.map(|v| (v - MARGIN).div_euclid(16));
    let hi = origin.map(|v| (v + span + MARGIN - 1).div_euclid(16));
    let corner = glam::IVec3::from(origin);
    let mut sources: Vec<(Source, f64)> = Vec::new();
    for (key, lights) in sections {
        let key = [key.0, key.1 + crate::chunk::MIN_Y.div_euclid(16), key.2];
        if (0..3).any(|i| key[i] < lo[i] || key[i] > hi[i]) {
            continue;
        }
        for &(pos, state) in lights {
            let Some(&(color, level)) = table.get(&state) else {
                continue;
            };
            let pos = glam::IVec3::from(pos);
            let center = pos.as_dvec3() + 0.5;
            sources.push(((pos - corner, color, level), center.distance_squared(camera)));
        }
    }
    if sources.len() > MAX_SOURCES {
        sources.select_nth_unstable_by(MAX_SOURCES, |a, b| a.1.total_cmp(&b.1));
        sources.truncate(MAX_SOURCES);
    }
    if sources.len() > MAX_COLORED {
        sources.select_nth_unstable_by(MAX_COLORED, |a, b| a.1.total_cmp(&b.1));
    }
    sources.into_iter().map(|(s, _)| s).collect()
}

/// The cells `sources` light, past `walls`: RGBA as f16 bits, x fastest, then y, then z;
/// rgb the colour over `COLOR_RANGE`, a the light reaching the cell over 15.
fn build_cells(sources: &[Source], walls: &Walls) -> Vec<u16> {
    let n = SIZE as usize;
    let mut out = vec![0u16; n * n * n * 4];
    if sources.is_empty() {
        return out;
    }
    let level = flood(sources, walls);
    let colors = colors(&sources[..sources.len().min(MAX_COLORED)]);
    let m = MARGIN as usize;
    let w = n + 2 * m;
    let cn = (SIZE as i32 / COLOR_CELL) as usize;
    let coarse = |x: usize, y: usize, z: usize| colors[x + cn * (y + cn * z)];
    // Where a cell's centre falls among the colour cells' centres, per axis.
    let taps: Vec<(usize, usize, f32)> = (0..n)
        .map(|i| {
            let c = ((i as f32 + 0.5) / COLOR_CELL as f32 - 0.5).clamp(0.0, (cn - 1) as f32);
            let a = (c.floor() as usize).min(cn - 2);
            (a, a + 1, c - a as f32)
        })
        .collect();
    for z in 0..n {
        let (z0, z1, fz) = taps[z];
        for y in 0..n {
            let (y0, y1, fy) = taps[y];
            for x in 0..n {
                let l = level[(x + m) + w * ((y + m) + w * (z + m))];
                if l <= 0.0 {
                    continue;
                }
                let (x0, x1, fx) = taps[x];
                let mut s = [0.0f32; 4];
                for (zi, wz) in [(z0, 1.0 - fz), (z1, fz)] {
                    for (yi, wy) in [(y0, 1.0 - fy), (y1, fy)] {
                        for (xi, wx) in [(x0, 1.0 - fx), (x1, fx)] {
                            let c = coarse(xi, yi, zi);
                            let k = wx * wy * wz;
                            for i in 0..4 {
                                s[i] += c[i] * k;
                            }
                        }
                    }
                }
                let cell = &mut out[(x + n * (y + n * z)) * 4..][..4];
                if s[3] > 0.0 {
                    for i in 0..3 {
                        cell[i] = half::f16::from_f32(s[i] / s[3] / COLOR_RANGE).to_bits();
                    }
                }
                cell[3] = half::f16::from_f32(l).to_bits();
            }
        }
    }
    out
}

/// The light reaching each cell of the grid and its margin (x fastest, then y, then z), 0 to
/// 1: spread from the sources (each from its own block at its level, wall or not) to all 26
/// neighbours, a step costing its length, never into a wall nor past a wall's edge. A wall
/// takes the light of the open cells beside it, so the grid blends smoothly up to it.
fn flood(sources: &[Source], walls: &Walls) -> Vec<f32> {
    let w = (SIZE as i32 + 2 * MARGIN) as usize;
    let at = |p: glam::IVec3| -> Option<usize> {
        let p = p + MARGIN;
        (p.cmpge(glam::IVec3::ZERO).all() && p.cmplt(glam::IVec3::splat(w as i32)).all())
            .then(|| p.x as usize + w * (p.y as usize + w * p.z as usize))
    };
    let mut wall = vec![false; w * w * w];
    for (corner, rows) in walls {
        for (row, &bits) in rows.iter().enumerate() {
            if bits == 0 {
                continue;
            }
            let (y, z) = ((row >> 4) as i32, (row & 15) as i32);
            for x in 0..16 {
                if bits & (1 << x) != 0
                    && let Some(i) = at(*corner + glam::IVec3::new(x, y, z))
                {
                    wall[i] = true;
                }
            }
        }
    }
    // The steps: each neighbour, what the step costs and the face and edge neighbours it
    // passes, which must be open too.
    let mut steps: Vec<(isize, u8, Vec<isize>)> = Vec::new();
    let offset = |d: [i32; 3]| d[0] as isize + w as isize * (d[1] as isize + w as isize * d[2] as isize);
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let d = [dx, dy, dz];
                let axes = d.iter().filter(|&&v| v != 0).count();
                if axes == 0 {
                    continue;
                }
                let cost = [0, UNIT, 23, 28][axes];
                // Every neighbour on the way: d with some of its axes zeroed, not all, not none.
                let mut passed = Vec::new();
                for mask in 1..7u8 {
                    let p = [0, 1, 2].map(|i| if mask & (1 << i) != 0 { d[i] } else { 0 });
                    if p != d && p != [0; 3] && p.iter().zip(&d).all(|(a, b)| a == b || *a == 0) {
                        let o = offset(p);
                        if !passed.contains(&o) {
                            passed.push(o);
                        }
                    }
                }
                steps.push((offset(d), cost, passed));
            }
        }
    }
    // The cells by their light, brightest first (Dial's buckets: a step costs a whole number
    // of units), each spreading once, at the most light that reached it.
    let top = 15 * UNIT as usize;
    let mut level = vec![0u8; w * w * w];
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); top + 1];
    for &(pos, _, l) in sources {
        if let Some(i) = at(pos) {
            let l = l.min(15) * UNIT;
            if l > level[i] {
                level[i] = l;
                buckets[l as usize].push(i as u32);
            }
        }
    }
    for b in (1..=top).rev() {
        while let Some(i) = buckets[b].pop() {
            let i = i as usize;
            if level[i] as usize != b {
                continue;
            }
            let (x, y, z) = (i % w, i / w % w, i / (w * w));
            // At the edge of the margin, the steps out of it are skipped.
            let edge = x == 0 || y == 0 || z == 0 || x == w - 1 || y == w - 1 || z == w - 1;
            for (o, cost, passed) in &steps {
                if usize::from(*cost) >= b {
                    continue;
                }
                let j = i as isize + o;
                if edge {
                    let (jx, jy, jz) = (
                        x as isize + step_axis(*o, w, 0),
                        y as isize + step_axis(*o, w, 1),
                        z as isize + step_axis(*o, w, 2),
                    );
                    if [jx, jy, jz].iter().any(|&v| v < 0 || v >= w as isize) {
                        continue;
                    }
                }
                let j = j as usize;
                let nl = (b - usize::from(*cost)) as u8;
                if nl <= level[j] || wall[j] {
                    continue;
                }
                if passed.iter().any(|p| wall[(i as isize + p) as usize]) {
                    continue;
                }
                level[j] = nl;
                buckets[nl as usize].push(j as u32);
            }
        }
    }
    let top = top as f32;
    let mut out: Vec<f32> = level.iter().map(|&l| f32::from(l) / top).collect();
    for i in 0..w * w * w {
        if !wall[i] || level[i] > 0 {
            continue;
        }
        let (x, y, z) = (i % w, i / w % w, i / (w * w));
        let (mut sum, mut count) = (0.0, 0);
        for (axis, v) in [x, y, z].into_iter().enumerate() {
            let stride = [1, w, w * w][axis];
            if v > 0 && !wall[i - stride] {
                sum += out[i - stride];
                count += 1;
            }
            if v + 1 < w && !wall[i + stride] {
                sum += out[i + stride];
                count += 1;
            }
        }
        if count > 0 {
            out[i] = sum / count as f32;
        }
    }
    out
}

/// The step of a neighbour offset `o` (in a grid `w` wide) along `axis`: -1, 0 or 1.
fn step_axis(o: isize, w: usize, axis: usize) -> isize {
    let w = w as isize;
    // o = dx + w (dy + w dz), each of dx, dy, dz in -1..=1.
    let dx = (o + 1).rem_euclid(w) - 1;
    let rest = (o - dx) / w;
    let dy = (rest + 1).rem_euclid(w) - 1;
    let dz = (rest - dy) / w;
    [dx, dy, dz][axis]
}

/// The colour sums of `sources` on the coarse grid (cells `COLOR_CELL` blocks wide): rgb by
/// weight, and the weight, as block light falls off, strongest at the source.
fn colors(sources: &[Source]) -> Vec<[f32; 4]> {
    let n = (SIZE as i32 / COLOR_CELL) as usize;
    let cell = COLOR_CELL as f32;
    let mut sum = vec![[0.0f32; 4]; n * n * n];
    for &(pos, color, level) in sources {
        let at = pos.as_vec3() + 0.5;
        let reach = f32::from(level);
        let first = ((at - reach) / cell).floor().max(Vec3::ZERO).as_ivec3();
        let last = ((at + reach) / cell).ceil().min(Vec3::splat(n as f32 - 1.0)).as_ivec3();
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    let c = (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * cell;
                    let d = c.distance(at);
                    if d >= reach {
                        continue;
                    }
                    let w = (1.0 - d / reach).powi(2);
                    let s = &mut sum[x as usize + n * (y as usize + n * z as usize)];
                    s[0] += color[0] * w;
                    s[1] += color[1] * w;
                    s[2] += color[2] * w;
                    s[3] += w;
                }
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A torch of level 14 at world (8, 64, 8), its section's key, and the grid's corner.
    fn torch() -> (LightTable, FxHashMap<SectionKey, Vec<([i32; 3], u32)>>, [i32; 3], SectionKey) {
        let mut table = LightTable::default();
        table.insert(7, ([2.0, 1.0, 0.0], 14));
        let mut sections = FxHashMap::default();
        let sy = (64 - crate::chunk::MIN_Y) / 16;
        sections.insert((0, sy, 0), vec![([8, 64, 8], 7)]);
        (table, sections, [-64, 0, -64], (0, sy, 0))
    }

    fn cell(cells: &[u16], origin: [i32; 3], p: [i32; 3]) -> [f32; 4] {
        let n = SIZE as usize;
        let c = [0, 1, 2].map(|i| (p[i] - origin[i]) as usize);
        let i = (c[0] + n * (c[1] + n * c[2])) * 4;
        [0, 1, 2, 3].map(|k| half::f16::from_bits(cells[i + k]).to_f32())
    }

    #[test]
    fn a_source_lights_the_cells_it_reaches_with_its_colour_falling_off_round() {
        let (table, sections, origin, _) = torch();
        let camera = DVec3::new(8.0, 64.0, 8.0);
        let cells = build(&table, origin, camera, &sections, &FxHashMap::default());
        let at = |p| cell(&cells, origin, p);
        let near = at([8, 64, 8]);
        // The colour over 4, whatever the weight: (2, 1, 0) / 4.
        assert!((near[0] - 0.5).abs() < 0.01 && (near[1] - 0.25).abs() < 0.01, "{near:?}");
        assert!((near[3] - 14.0 / 15.0).abs() < 0.01);
        // A step costs its length: four blocks along an axis or diagonally, both near 10.
        assert!((at([12, 64, 8])[3] * 15.0 - 10.0).abs() < 0.1);
        let diagonal = at([11, 67, 8])[3] * 15.0;
        assert!((diagonal - (14.0 - 3.0 * 23.0 / 16.0)).abs() < 0.1, "{diagonal}");
        assert_eq!(at([8, 64, 40]), [0.0; 4], "past its level, nothing");
        assert!(
            build(&LightTable::default(), origin, camera, &sections, &FxHashMap::default())
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    fn walls_stop_the_light_which_goes_round_them() {
        let (table, sections, origin, key) = torch();
        let camera = DVec3::new(8.0, 64.0, 8.0);
        // A wall at x = 10 across the section and those above and below, but for z = 12 to
        // 15: the light goes round it.
        let mut rows = [0u16; 256];
        for y in 0..16 {
            for z in 0..12 {
                rows[y << 4 | z] = 1 << 10;
            }
        }
        let mut walls = FxHashMap::default();
        for dy in -1..=1 {
            walls.insert((key.0, key.1 + dy, key.2), Box::new(rows));
        }
        let cells = build(&table, origin, camera, &sections, &walls);
        let at = |p| cell(&cells, origin, p)[3] * 15.0;
        let open = build(&table, origin, camera, &sections, &FxHashMap::default());
        let open_at = |p| cell(&open, origin, p)[3] * 15.0;
        // Behind the wall, the long way round (out to z = 12 and back): darker than in the open.
        assert!(at([11, 64, 8]) < open_at([11, 64, 8]) - 4.0, "{}", at([11, 64, 8]));
        assert!(at([11, 64, 8]) > 0.0);
        // On the torch's side, as in the open.
        assert_eq!(at([9, 64, 8]), open_at([9, 64, 8]));
        // The wall itself takes the light beside it.
        assert!(at([10, 64, 8]) > 0.0);
    }
}
