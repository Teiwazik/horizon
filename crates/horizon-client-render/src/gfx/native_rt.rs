//! Hardware ray tracing for Horizon's own shader packs (a prototype, behind `HORIZON_RT=1`).
//!
//! Each section near the camera has a structure for its solid terrain and one for its
//! translucent terrain, built from its vertices where they lie and kept until its mesh changes;
//! entities get one structure a frame. The vertices are also copied to a storage buffer, where
//! the shader reads a candidate's texture: a cutout's transparent texels let a ray through, and
//! translucent terrain tints it. The nearest light sources go to a second storage buffer.

use super::SectionKey;
use super::block_light::LightTable;
use glam::DVec3;

/// Sections further than this from the camera, in blocks, are left out.
pub const RADIUS: f32 = 128.0;
/// Sections' structures built per frame, nearest first.
const BUILDS_PER_FRAME: usize = 64;
/// The CPU time a frame spends creating sections' structures, nearest first: each is a call into
/// the driver, and flying over new ground asked for dozens a frame (frames of 30 to 55 ms). What
/// is left waits for the next frames; far sections are in the rays a few frames later.
const BUILD_BUDGET: std::time::Duration = std::time::Duration::from_micros(2500);
/// Top-level instances: two per section at most, and the entities.
const MAX_INSTANCES: usize = 8192;
/// Terrain vertices the shader can read (24 bytes each).
const DATA_VERTICES: u64 = (192 << 20) / 24;
/// Light sources the shader loops over: the strongest here.
pub const MAX_LIGHTS: usize = 2048;
/// The grid of which sources reach where: cells along a side, a cell's blocks, and a cell's
/// words: 16 bits of count, then up to 63 sources by 16 bits each, strongest first (31 drew
/// seams between cells under a crowd of lamps, each cell keeping different ones).
const LIGHT_GRID: u32 = 20;
const LIGHT_CELL: f32 = 8.0;
const LIGHT_CELL_WORDS: usize = 32;
/// `instance_custom_data` of translucent terrain.
const TRANSLUCENT: u32 = 1 << 23;
/// `instance_custom_data` of the entities, whose vertices aren't in the scene's buffer: the
/// vertex buffer never fills to its very last vertex.
const ENTITY: u32 = (1 << 23) - 1;
/// Instance masks: what `ray_light` and `ray_occluded` ask for.
const MASK_SOLID: u8 = 1;
const MASK_TRANSLUCENT: u8 = 2;

/// The usage a buffer needs for structures to be built from it.
pub fn input_usage(device: &wgpu::Device) -> wgpu::BufferUsages {
    if device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
    {
        wgpu::BufferUsages::BLAS_INPUT
    } else {
        wgpu::BufferUsages::empty()
    }
}

/// A section as the scene needs it.
pub struct RaySection<'a> {
    pub key: SectionKey,
    pub origin: [f32; 3],
    pub first_vertex: u64,
    pub solid: u32,
    pub translucent: u32,
    pub vertices: &'a wgpu::Buffer,
}

/// The entities' vertices this frame: quads from the start, then the shadow-only ones.
pub struct RayEntities<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub stride: u64,
    pub quads: u32,
    pub shadow_only: (u32, u32),
}

/// One light for the shader: camera-relative position and reach; linear colour and intensity.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Light {
    position: [f32; 4],
    color: [f32; 4],
}

struct Built {
    solid: Option<wgpu::Blas>,
    translucent: Option<wgpu::Blas>,
    /// Its vertices' place in `data`.
    data: u64,
    count: u64,
    solid_quads: u32,
}

pub struct RayScene {
    pub tlas: wgpu::Tlas,
    /// Sections' vertices, for the shader to read a candidate's texture.
    pub data: wgpu::Buffer,
    /// `lights[0].position.x` the count, the lights after it.
    pub lights: wgpu::Buffer,
    /// Each grid cell's list of `lights`.
    pub light_cells: wgpu::Buffer,
    /// A 64x64 tile of blue noise, two channels of 16 bits in each word (`blue_noise`).
    pub blue_noise: wgpu::Buffer,
    light_grid: Vec<u32>,
    built: crate::fast_hash::FxHashMap<SectionKey, Built>,
    /// Free ranges of `data`, in vertices, in order.
    free: Vec<(u64, u64)>,
    used: usize,
    logged: std::time::Instant,
    builds: usize,
}

fn sizes(quads: u32, opaque: bool) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: quads * 4,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(quads * 6),
        flags: if opaque {
            wgpu::AccelerationStructureGeometryFlags::OPAQUE
        } else {
            wgpu::AccelerationStructureGeometryFlags::empty()
        },
    }
}

fn blas(
    device: &wgpu::Device,
    descriptors: Vec<wgpu::BlasTriangleGeometrySizeDescriptor>,
    flags: wgpu::AccelerationStructureFlags,
) -> wgpu::Blas {
    device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: None,
            flags,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors },
    )
}

/// Blue noise by void and cluster (Ulichney): a 64x64 tile wrapping at its edges, each texel's
/// rank in 0..65536, spread so that texels of close ranks lie far apart. Random numbers drawn from
/// it vary only between distant pixels: a filter over a few neighbours averages them out, where
/// the same count of white noise leaves clumps. `seed` gives another, independent tile.
pub fn blue_noise(seed: u64) -> Vec<u16> {
    const SIZE: usize = 64;
    const N: usize = SIZE * SIZE;
    // How much each point weighs on the others, by wrapped distance.
    let mut kernel = vec![0f32; N];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x.min(SIZE - x) as f32;
            let dy = y.min(SIZE - y) as f32;
            kernel[y * SIZE + x] = (-(dx * dx + dy * dy) / (2.0 * 1.9 * 1.9)).exp();
        }
    }
    let spread = |energy: &mut [f32], at: usize, sign: f32| {
        let (ax, ay) = (at % SIZE, at / SIZE);
        for y in 0..SIZE {
            let ky = (y + SIZE - ay) % SIZE;
            for x in 0..SIZE {
                energy[y * SIZE + x] += sign * kernel[ky * SIZE + (x + SIZE - ax) % SIZE];
            }
        }
    };
    let extreme = |energy: &[f32], on: &[bool], want: bool, most: bool| {
        let mut best = usize::MAX;
        for i in 0..N {
            if on[i] == want
                && (best == usize::MAX
                    || (energy[i] > energy[best]) == most && energy[i] != energy[best])
            {
                best = i;
            }
        }
        best
    };
    // A first scatter, then points moved from the tightest cluster to the largest void until
    // none would move.
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut on = vec![false; N];
    let mut energy = vec![0f32; N];
    let mut placed = 0;
    while placed < N / 10 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let i = (state >> 33) as usize % N;
        if !on[i] {
            on[i] = true;
            spread(&mut energy, i, 1.0);
            placed += 1;
        }
    }
    // Settled long before this many; the cap only guards against two points trading places.
    for _ in 0..N {
        let cluster = extreme(&energy, &on, true, true);
        on[cluster] = false;
        spread(&mut energy, cluster, -1.0);
        let void = extreme(&energy, &on, false, false);
        on[void] = true;
        spread(&mut energy, void, 1.0);
        if void == cluster {
            break;
        }
    }
    let mut rank = vec![0usize; N];
    // The first points' ranks, taking the tightest cluster away each time.
    let (mut first, mut first_energy) = (on.clone(), energy.clone());
    for r in (0..placed).rev() {
        let cluster = extreme(&first_energy, &first, true, true);
        first[cluster] = false;
        spread(&mut first_energy, cluster, -1.0);
        rank[cluster] = r;
    }
    // The rest, filling the largest void each time.
    for r in placed..N {
        let void = extreme(&energy, &on, false, false);
        on[void] = true;
        spread(&mut energy, void, 1.0);
        rank[void] = r;
    }
    rank.iter()
        .map(|&r| ((r * 65536 + 32768) / N) as u16)
        .collect()
}

impl RayScene {
    /// The scene, where the device has ray queries.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        {
            return None;
        }
        Some(Self {
            tlas: device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("Sections near the camera"),
                max_instances: MAX_INSTANCES as u32,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            }),
            data: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ray traced terrain's vertices"),
                size: DATA_VERTICES * 24,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            light_cells: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ray traced light grid"),
                size: u64::from(LIGHT_GRID * LIGHT_GRID * LIGHT_GRID) * LIGHT_CELL_WORDS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blue_noise: {
                let words: Vec<u32> = blue_noise(1)
                    .iter()
                    .zip(blue_noise(2))
                    .map(|(&a, b)| u32::from(a) | u32::from(b) << 16)
                    .collect();
                wgpu::util::DeviceExt::create_buffer_init(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Ray noise"),
                        contents: bytemuck::cast_slice(&words),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                )
            },
            light_grid: vec![0; (LIGHT_GRID * LIGHT_GRID * LIGHT_GRID) as usize * LIGHT_CELL_WORDS],
            lights: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Ray traced lights"),
                size: ((MAX_LIGHTS + 1) * size_of::<Light>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            built: Default::default(),
            free: vec![(0, DATA_VERTICES)],
            used: 0,
            logged: std::time::Instant::now(),
            builds: 0,
        })
    }

    fn alloc(&mut self, count: u64) -> Option<u64> {
        let i = self.free.iter().position(|f| f.1 >= count)?;
        let (start, len) = self.free[i];
        if len == count {
            self.free.remove(i);
        } else {
            self.free[i] = (start + count, len - count);
        }
        Some(start)
    }

    fn release(&mut self, start: u64, count: u64) {
        let i = self.free.partition_point(|f| f.0 < start);
        self.free.insert(i, (start, count));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    /// A section's mesh went away.
    pub fn forget(&mut self, key: SectionKey) {
        if let Some(b) = self.built.remove(&key) {
            self.release(b.data, b.count);
        }
    }

    /// This frame's light sources: the block lights in `sections` and the dynamic ones,
    /// strongest here first.
    pub fn update_lights(
        &mut self,
        queue: &wgpu::Queue,
        camera: DVec3,
        sections: &crate::fast_hash::FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
        table: &LightTable,
        dynamic: &[[f32; 4]],
        dynamic_states: &[Option<u32>],
        colored: &[crate::dynamic_light::Colored],
        particles: &[crate::dynamic_light::ParticleSource],
    ) {
        // Sources by how much they matter to the camera, with a stable order among equals.
        let mut lights: Vec<(f32, [i32; 3], Light)> = Vec::new();
        let mut add =
            |at: [f32; 3], key: [i32; 3], reach: f32, color: [f32; 3], size: f32, bright: f32| {
                let d2 = at[0] * at[0] + at[1] * at[1] + at[2] * at[2];
                if reach > 0.0 && d2 < (reach + 80.0) * (reach + 80.0) {
                    // The brightness is the level's: `[[lights]]` colours go up to 4.
                    let top = color[0].max(color[1]).max(color[2]).max(0.001);
                    lights.push((
                        d2 - reach * reach,
                        key,
                        Light {
                            position: [at[0], at[1], at[2], reach],
                            // w: half the source's size, which softens its shadows.
                            color: [
                                color[0] / top * bright,
                                color[1] / top * bright,
                                color[2] / top * bright,
                                size,
                            ],
                        },
                    ));
                }
            };
        for (at, state) in sections.values().flatten() {
            if let Some((color, level)) = table.get(state) {
                let p = (DVec3::new(f64::from(at[0]), f64::from(at[1]), f64::from(at[2])) + 0.5
                    - camera)
                    .as_vec3();
                add(p.to_array(), *at, f32::from(*level), *color, 0.5, 1.0);
            }
        }
        // A source that moves has no block: after every block on ties.
        for (i, l) in dynamic.iter().enumerate() {
            // The colour of the block it shines as, by the pack's `[[lights]]`.
            let color = dynamic_states
                .get(i)
                .copied()
                .flatten()
                .and_then(|state| table.get(&state))
                .map_or([1.0, 0.85, 0.65], |(color, _)| *color);
            add([l[0], l[1], l[2]], [i32::MAX; 3], l[3], color, 0.5, 1.0);
        }
        for [p, c] in colored {
            add(
                [p[0], p[1], p[2]],
                [i32::MAX; 3],
                p[3],
                [c[0], c[1], c[2]],
                0.5,
                1.0,
            );
        }
        for p in particles {
            let bright = p.color[0].max(p.color[1]).max(p.color[2]);
            add(p.at, [i32::MAX; 3], p.reach, p.color, p.size, bright);
        }
        lights.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        lights.truncate(MAX_LIGHTS);
        // The grid sits on the world's cells, so a list doesn't change as the camera moves
        // within one, and a far pile of sources takes nothing from the ones here.
        let snap = (camera / f64::from(LIGHT_CELL)).floor() * f64::from(LIGHT_CELL);
        let half = (LIGHT_GRID / 2) as f32 * LIGHT_CELL;
        let origin = (snap - camera).as_vec3() - half;
        self.light_grid.fill(0);
        let mut cells =
            vec![Vec::<(f32, u16)>::new(); (LIGHT_GRID * LIGHT_GRID * LIGHT_GRID) as usize];
        for (i, (_, _, light)) in lights.iter().enumerate() {
            let [x, y, z, reach] = light.position;
            let cell = |v: f32, o: f32| ((v - o) / LIGHT_CELL).floor() as i32;
            let span = |v: f32, o: f32| {
                cell(v - reach, o).max(0)..=cell(v + reach, o).min(LIGHT_GRID as i32 - 1)
            };
            for cz in span(z, origin.z) {
                for cy in span(y, origin.y) {
                    for cx in span(x, origin.x) {
                        // How strong it is at the cell's nearest point, by the shader's falloff:
                        // ranked at the middle, a source by the cell's edge lost its place to
                        // farther ones and its light ended on the edge in a straight line.
                        let middle = |c: i32, o: f32| o + (c as f32 + 0.5) * LIGHT_CELL;
                        let off = |v: f32, c: i32, o: f32| {
                            ((v - middle(c, o)).abs() - LIGHT_CELL * 0.5).max(0.0)
                        };
                        let (dx, dy, dz) = (
                            off(x, cx, origin.x),
                            off(y, cy, origin.y),
                            off(z, cz, origin.z),
                        );
                        let d2 = dx * dx + dy * dy + dz * dz;
                        let n = LIGHT_GRID as i32;
                        cells[((cz * n + cy) * n + cx) as usize]
                            .push((reach * reach / (1.0 + d2 * 0.12), i as u16 + 1));
                    }
                }
            }
        }
        for (i, cell) in cells.iter_mut().enumerate() {
            cell.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let words = &mut self.light_grid[i * LIGHT_CELL_WORDS..(i + 1) * LIGHT_CELL_WORDS];
            let count = cell.len().min(LIGHT_CELL_WORDS * 2 - 1);
            words[0] = count as u32;
            for (j, (_, id)) in cell.iter().take(count).enumerate() {
                words[(j + 1) / 2] |= u32::from(*id) << (((j + 1) % 2) * 16);
            }
        }
        let mut out = vec![Light {
            position: [lights.len() as f32, origin.x, origin.y, origin.z],
            color: [0.0; 4],
        }];
        out.extend(lights.into_iter().map(|l| l.2));
        queue.write_buffer(&self.lights, 0, bytemuck::cast_slice(&out));
        queue.write_buffer(&self.light_cells, 0, bytemuck::cast_slice(&self.light_grid));
    }

    /// This frame's top-level structure over `sections` and `entities`, building sections'
    /// structures not built yet, nearest first and `BUILDS_PER_FRAME` at most.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mut sections: Vec<RaySection<'_>>,
        entities: Option<RayEntities<'_>>,
        indices: &wgpu::Buffer,
    ) {
        let near = |o: [f32; 3]| o[0] * o[0] + o[1] * o[1] + o[2] * o[2];
        sections.retain(|s| s.solid + s.translucent > 0 && near(s.origin) <= RADIUS * RADIUS);
        sections.sort_by(|a, b| near(a.origin).total_cmp(&near(b.origin)));
        let mut seen = crate::fast_hash::FxHashMap::default();
        sections.retain(|s| seen.insert(s.key, ()).is_none());
        // What left the rays' reach gives its room back.
        let gone: Vec<SectionKey> = self
            .built
            .keys()
            .filter(|key| !seen.contains_key(*key))
            .copied()
            .collect();
        for key in gone {
            self.forget(key);
        }
        // Sections' structures, and what they are built from.
        struct New {
            section: usize,
            size: wgpu::BlasTriangleGeometrySizeDescriptor,
            blas: wgpu::Blas,
            first_vertex: u64,
        }
        let mut new: Vec<New> = Vec::new();
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;
        let started = std::time::Instant::now();
        for (i, s) in sections.iter().enumerate() {
            if self.built.contains_key(&s.key) {
                continue;
            }
            if new.len() >= BUILDS_PER_FRAME * 2 || (!new.is_empty() && started.elapsed() > BUILD_BUDGET) {
                break;
            }
            let count = u64::from(s.solid + s.translucent) * 4;
            // Full: the farthest sections give way to a nearer one.
            let data = loop {
                if let Some(data) = self.alloc(count) {
                    break Some(data);
                }
                let farther = sections[i + 1..]
                    .iter()
                    .rev()
                    .find(|o| self.built.contains_key(&o.key))
                    .map(|o| o.key);
                match farther {
                    Some(key) => self.forget(key),
                    None => break None,
                }
            };
            let Some(data) = data else {
                continue;
            };
            encoder.copy_buffer_to_buffer(
                s.vertices,
                s.first_vertex * 24,
                &self.data,
                data * 24,
                count * 24,
            );
            let make = |quads: u32| {
                let size = sizes(quads, false);
                (quads > 0).then(|| (blas(device, vec![size.clone()], flags), size))
            };
            let solid = make(s.solid);
            let translucent = make(s.translucent);
            let mut built = Built {
                solid: None,
                translucent: None,
                data,
                count,
                solid_quads: s.solid,
            };
            if let Some((blas, size)) = solid {
                built.solid = Some(blas.clone());
                new.push(New {
                    section: i,
                    size,
                    blas,
                    first_vertex: s.first_vertex,
                });
            }
            if let Some((blas, size)) = translucent {
                built.translucent = Some(blas.clone());
                new.push(New {
                    section: i,
                    size,
                    blas,
                    first_vertex: s.first_vertex + u64::from(s.solid) * 4,
                });
            }
            self.built.insert(s.key, built);
        }
        self.builds += new.len();
        let triangles = |size, vertices, first_vertex, stride| wgpu::BlasTriangleGeometry {
            size,
            vertex_buffer: vertices,
            first_vertex,
            vertex_stride: stride,
            index_buffer: Some(indices),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };
        let mut entries: Vec<wgpu::BlasBuildEntry<'_>> = new
            .iter()
            .map(|n| wgpu::BlasBuildEntry {
                blas: &n.blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![triangles(
                    &n.size,
                    sections[n.section].vertices,
                    n.first_vertex as u32,
                    24,
                )]),
            })
            .collect();
        // The entities, every frame: they move and animate.
        let entity_sizes = entities
            .as_ref()
            .map(|e| [sizes(e.quads, true), sizes(e.shadow_only.1, true)]);
        let entity_blas = entities
            .as_ref()
            .zip(entity_sizes.as_ref())
            .and_then(|(e, s)| {
                let descriptors: Vec<_> =
                    s.iter().filter(|d| d.vertex_count > 0).cloned().collect();
                (!descriptors.is_empty()).then(|| {
                    let _ = e;
                    blas(
                        device,
                        descriptors,
                        wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                    )
                })
            });
        if let (Some(e), Some(s), Some(b)) = (&entities, &entity_sizes, &entity_blas) {
            let mut geometry = Vec::new();
            if e.quads > 0 {
                geometry.push(triangles(&s[0], e.vertices, 0, e.stride));
            }
            if e.shadow_only.1 > 0 {
                geometry.push(triangles(&s[1], e.vertices, e.shadow_only.0, e.stride));
            }
            entries.push(wgpu::BlasBuildEntry {
                blas: b,
                geometry: wgpu::BlasGeometries::TriangleGeometries(geometry),
            });
        }
        // The instances, camera-relative.
        let mut instances: Vec<wgpu::TlasInstance> = Vec::new();
        for s in &sections {
            let Some(b) = self.built.get(&s.key) else {
                continue;
            };
            let [x, y, z] = s.origin;
            let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z];
            if let Some(blas) = &b.solid
                && instances.len() < MAX_INSTANCES - 1
            {
                instances.push(wgpu::TlasInstance::new(
                    blas,
                    transform,
                    b.data as u32,
                    MASK_SOLID,
                ));
            }
            if let Some(blas) = &b.translucent
                && instances.len() < MAX_INSTANCES - 1
            {
                let data = b.data as u32 + b.solid_quads * 4;
                instances.push(wgpu::TlasInstance::new(
                    blas,
                    transform,
                    data | TRANSLUCENT,
                    MASK_TRANSLUCENT,
                ));
            }
        }
        if let Some(b) = &entity_blas {
            let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            instances.push(wgpu::TlasInstance::new(b, identity, ENTITY, MASK_SOLID));
        }
        let count = instances.len();
        let slots = self
            .tlas
            .get_mut_slice(0..MAX_INSTANCES)
            .expect("the structure's own size");
        for (slot, instance) in slots.iter_mut().zip(instances) {
            *slot = Some(instance);
        }
        for slot in &mut slots[count..self.used.max(count)] {
            *slot = None;
        }
        self.used = count;
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
        if self.logged.elapsed().as_secs() >= 5 {
            let free: u64 = self.free.iter().map(|f| f.1).sum();
            log::info!(
                "ray scene: {count} instances, {} sections, {} structures built in 5 s, vertices {} MiB of {} MiB",
                self.built.len(),
                self.builds,
                (DATA_VERTICES - free) * 24 >> 20,
                DATA_VERTICES * 24 >> 20,
            );
            self.logged = std::time::Instant::now();
            self.builds = 0;
        }
    }
}

#[cfg(test)]
mod blue_noise_tests {
    use super::blue_noise;

    #[test]
    fn every_rank_once_and_neighbours_apart() {
        let tile = blue_noise(1);
        let mut seen = tile.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 64 * 64);
        // White noise puts a right-hand neighbour within 1/16 of the range about 1 time in 8;
        // blue noise far less often.
        let close = (0..64 * 64)
            .filter(|&i| {
                let right = (i / 64) * 64 + (i % 64 + 1) % 64;
                (i32::from(tile[i]) - i32::from(tile[right])).abs() < 65536 / 16
            })
            .count();
        assert!(close < 64 * 64 / 32, "{close} close neighbours");
        assert_ne!(blue_noise(1), blue_noise(2));
    }
}
