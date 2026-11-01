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
const RADIUS: f32 = 128.0;
/// Sections' structures built per frame, nearest first.
const BUILDS_PER_FRAME: usize = 64;
/// Top-level instances: two per section at most, and the entities.
const MAX_INSTANCES: usize = 8192;
/// Terrain vertices the shader can read (24 bytes each).
const DATA_VERTICES: u64 = (192 << 20) / 24;
/// Light sources the shader loops over: the strongest here.
pub const MAX_LIGHTS: usize = 48;
/// `instance_custom_data` of translucent terrain.
const TRANSLUCENT: u32 = 1 << 23;
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
        &self,
        queue: &wgpu::Queue,
        camera: DVec3,
        sections: &crate::fast_hash::FxHashMap<SectionKey, Vec<([i32; 3], u32)>>,
        table: &LightTable,
        dynamic: &[[f32; 4]],
        colored: &[crate::dynamic_light::Colored],
    ) {
        let mut lights: Vec<(f32, Light)> = Vec::new();
        let mut add = |at: [f32; 3], reach: f32, color: [f32; 3]| {
            let d2 = at[0] * at[0] + at[1] * at[1] + at[2] * at[2];
            if reach > 0.0 && d2 < (reach + 48.0) * (reach + 48.0) {
                lights.push((
                    reach * reach / (d2 + 1.0),
                    Light {
                        position: [at[0], at[1], at[2], reach],
                        color: [color[0], color[1], color[2], 1.0],
                    },
                ));
            }
        };
        for (at, state) in sections.values().flatten() {
            if let Some((color, level)) = table.get(state) {
                let p = (DVec3::new(f64::from(at[0]), f64::from(at[1]), f64::from(at[2])) + 0.5
                    - camera)
                    .as_vec3();
                add(p.to_array(), f32::from(*level), *color);
            }
        }
        for l in dynamic {
            add([l[0], l[1], l[2]], l[3], [1.0, 0.85, 0.65]);
        }
        for [p, c] in colored {
            add([p[0], p[1], p[2]], p[3], [c[0], c[1], c[2]]);
        }
        lights.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        lights.truncate(MAX_LIGHTS);
        let mut out = vec![Light {
            position: [lights.len() as f32, 0.0, 0.0, 0.0],
            color: [0.0; 4],
        }];
        out.extend(lights.into_iter().map(|l| l.1));
        queue.write_buffer(&self.lights, 0, bytemuck::cast_slice(&out));
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
        // Sections' structures, and what they are built from.
        struct New {
            section: usize,
            size: wgpu::BlasTriangleGeometrySizeDescriptor,
            blas: wgpu::Blas,
            first_vertex: u64,
        }
        let mut new: Vec<New> = Vec::new();
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;
        for (i, s) in sections.iter().enumerate() {
            if self.built.contains_key(&s.key) || new.len() >= BUILDS_PER_FRAME * 2 {
                continue;
            }
            let count = u64::from(s.solid + s.translucent) * 4;
            let Some(data) = self.alloc(count) else {
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
            instances.push(wgpu::TlasInstance::new(b, identity, 0, MASK_SOLID));
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
