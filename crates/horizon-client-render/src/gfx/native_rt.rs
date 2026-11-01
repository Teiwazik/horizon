//! Hardware ray tracing for Horizon's own shader packs (a prototype, behind `HORIZON_RT=1`):
//! one bottom-level structure per section's solid terrain, one top-level structure over those
//! near the camera, rebuilt each frame in camera-relative space, read by the material
//! prelude's `ray_occluded`.

use super::SectionKey;

/// Sections further than this from the camera, in blocks, are left out.
const RADIUS: f32 = 96.0;
/// Sections' structures built per frame, nearest first.
const BUILDS_PER_FRAME: usize = 64;
/// Top-level instances: sections within `RADIUS` at most.
const MAX_INSTANCES: usize = 8192;

/// A section as the scene needs it: its key and origin, its first vertex, its solid quads and
/// the buffer of its vertices.
pub struct RaySection<'a> {
    pub key: SectionKey,
    pub origin: [f32; 3],
    pub first_vertex: u32,
    pub quads: u32,
    pub vertices: &'a wgpu::Buffer,
}

pub struct RayScene {
    pub tlas: wgpu::Tlas,
    blas: crate::fast_hash::FxHashMap<SectionKey, wgpu::Blas>,
    used: usize,
    logged: std::time::Instant,
    built: usize,
}

fn sizes(quads: u32) -> wgpu::BlasTriangleGeometrySizeDescriptor {
    wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: quads * 4,
        index_format: Some(wgpu::IndexFormat::Uint32),
        index_count: Some(quads * 6),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    }
}

impl RayScene {
    /// The scene, where the device has ray queries.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            .then(|| Self {
                tlas: device.create_tlas(&wgpu::CreateTlasDescriptor {
                    label: Some("Sections near the camera"),
                    max_instances: MAX_INSTANCES as u32,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                }),
                blas: Default::default(),
                used: 0,
                logged: std::time::Instant::now(),
                built: 0,
            })
    }

    /// A section's mesh went away.
    pub fn forget(&mut self, key: SectionKey) {
        self.blas.remove(&key);
    }

    /// This frame's top-level structure over `sections`, building those without a structure
    /// yet, nearest first and `BUILDS_PER_FRAME` at most.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mut sections: Vec<RaySection<'_>>,
        indices: &wgpu::Buffer,
    ) {
        let near = |o: [f32; 3]| o[0] * o[0] + o[1] * o[1] + o[2] * o[2];
        sections.retain(|s| s.quads > 0 && near(s.origin) <= RADIUS * RADIUS);
        sections.sort_by(|a, b| near(a.origin).total_cmp(&near(b.origin)));
        sections.dedup_by_key(|s| s.key);
        let mut seen = crate::fast_hash::FxHashMap::default();
        sections.retain(|s| seen.insert(s.key, ()).is_none());
        sections.truncate(MAX_INSTANCES);
        let new: Vec<(usize, wgpu::BlasTriangleGeometrySizeDescriptor, wgpu::Blas)> = sections
            .iter()
            .enumerate()
            .filter(|(_, s)| !self.blas.contains_key(&s.key))
            .take(BUILDS_PER_FRAME)
            .map(|(i, s)| {
                let size = sizes(s.quads);
                let blas = device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: None,
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                );
                (i, size, blas)
            })
            .collect();
        let entries: Vec<wgpu::BlasBuildEntry<'_>> = new
            .iter()
            .map(|(i, size, blas)| {
                let s = &sections[*i];
                wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                        wgpu::BlasTriangleGeometry {
                            size,
                            vertex_buffer: s.vertices,
                            first_vertex: s.first_vertex,
                            vertex_stride: std::mem::size_of::<crate::mesher::TerrainVertex>()
                                as u64,
                            index_buffer: Some(indices),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        },
                    ]),
                }
            })
            .collect();
        self.built += new.len();
        let added: Vec<(SectionKey, wgpu::Blas)> = new
            .iter()
            .map(|(i, _, blas)| (sections[*i].key, blas.clone()))
            .collect();
        self.blas.extend(added);
        let mut count = 0;
        let slots = self
            .tlas
            .get_mut_slice(0..MAX_INSTANCES)
            .expect("the structure's own size");
        for s in &sections {
            if let Some(blas) = self.blas.get(&s.key) {
                let [x, y, z] = s.origin;
                slots[count] = Some(wgpu::TlasInstance::new(
                    blas,
                    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z],
                    0,
                    0xFF,
                ));
                count += 1;
            }
        }
        for slot in &mut slots[count..self.used.max(count)] {
            *slot = None;
        }
        self.used = count;
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
        if self.logged.elapsed().as_secs() >= 5 {
            log::info!(
                "ray scene: {count} instances of {} structures, {} built in 5 s",
                self.blas.len(),
                self.built
            );
            self.logged = std::time::Instant::now();
            self.built = 0;
        }
    }
}
