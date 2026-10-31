//! Shared nearest-sampled sprite array for block and item textures.
use crate::Result;
use horizon_assets::animation::{AnimationClip, AnimationMetadata};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::BufReader,
    path::Path,
    sync::Arc,
};

#[derive(Clone)]
pub struct SpriteAnimation {
    pub layer: u32,
    frames: Arc<Vec<Vec<u8>>>,
    clip: Arc<AnimationClip>,
}
/// What an animated sprite shows: `frame`, mixed `millis` thousandths of the
/// way towards `next` (0 for a single frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameBlend {
    pub frame: usize,
    pub next: usize,
    pub millis: u32,
}
impl SpriteAnimation {
    pub fn frame(&self, tick: u64) -> usize {
        self.clip.sample(tick).frame as usize
    }
    /// `SpriteContents.AnimationState.drawToAtlas`: an animation with
    /// `interpolate` mixes the frame with the next one by
    /// `(int)(subFrame / time * 1000)` thousandths; others show the frame.
    pub fn blend(&self, tick: u64) -> FrameBlend {
        let sample = self.clip.sample(tick);
        let frame = sample.frame as usize;
        let next = sample.next_frame as usize;
        if self.clip.interpolates() && sample.interpolation_millis > 0 && next != frame {
            FrameBlend {
                frame,
                next,
                millis: sample.interpolation_millis,
            }
        } else {
            FrameBlend {
                frame,
                next: frame,
                millis: 0,
            }
        }
    }
    pub fn resized_pixels(&self, frame: usize, width: u32, height: u32) -> Vec<u8> {
        let source = self.pixels(frame);
        let side = (source.len() / 4).isqrt() as u32;
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let offset = ((y * side / height * side + x * side / width) * 4) as usize;
                out.extend_from_slice(&source[offset..offset + 4]);
            }
        }
        out
    }
    pub fn pixels(&self, frame: usize) -> &[u8] {
        &self.frames[frame]
    }
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

#[derive(Default)]
pub struct Sprites {
    /// Edge length of every array layer and the layers' current frames, set
    /// by [`Sprites::finish`].
    pub size: u32,
    /// Smallest native layer before array-size resampling.
    pub minimum_size: u32,
    pub pixels: Vec<u8>,
    pub layers: BTreeMap<String, u32>,
    pub animations: Vec<SpriteAnimation>,
    pub opaque: Vec<bool>,
    pub translucent: Vec<bool>,
    /// Each layer's `MipmapGenerator` settings.
    pub mips: Vec<crate::mipmap::Settings>,
    /// Each layer's frames at the texture's own resolution (square), and the
    /// frame shown first.
    native: Vec<(u32, Arc<Vec<Vec<u8>>>, usize)>,
    /// Each layer's LabPBR maps (`_n` normals, `_s` specular) beside its texture, their first
    /// frame at their own resolution (square).
    pbr: Vec<[Option<(u32, Vec<u8>)>; 2]>,
    /// The layers' normal and specular maps at `pbr_size`, layer after layer, set by
    /// [`Sprites::finish`]: empty when no texture has either.
    pub normals: Vec<u8>,
    pub pbr_size: u32,
    pub specular: Vec<u8>,
    /// Still textures by their pixels, their LabPBR maps and mip settings: mods copy
    /// vanilla's textures under their own names, and one layer serves them all (the GPU's
    /// array may hold as few as 2048).
    still: std::collections::HashMap<(Vec<u8>, u64, u32, [u32; 2], [bool; 4]), u32>,
    /// Images decoded ahead by [`Sprites::prefetch`], until a texture takes its own.
    prefetched: std::collections::HashMap<std::path::PathBuf, Image>,
    /// Decoded textures kept on disk between launches ([`Sprites::with_disk_cache`]).
    disk: Option<crate::sprite_cache::SpriteCache>,
}
/// Edge length of vanilla's block textures: the smallest array layer.
pub const SPRITE_SIZE: u32 = 16;
/// A layer's normal map where its texture has none: flat, unoccluded, at full height.
pub const FLAT_NORMAL: [u8; 4] = [128, 128, 255, 255];
/// A layer's specular map where its texture has none: rough, not metal, not glowing.
pub const NO_SPECULAR: [u8; 4] = [0, 0, 0, 0];
/// The block textures' video memory (all layers of the top mip level, colour and LabPBR
/// maps together) when the settings leave it to Horizon: resource packs with bigger
/// textures are sampled down to fit.
pub const DEFAULT_TEXTURE_BUDGET: u64 = 1 << 30;
/// The budget in bytes the next resource load keeps to (`set_texture_budget`).
static TEXTURE_BUDGET: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(DEFAULT_TEXTURE_BUDGET);
/// The block textures' budget from the `textureBudgetMiB` setting (0 for the default),
/// for resources loaded from now on.
pub fn set_texture_budget(mib: u32) {
    let bytes = if mib == 0 {
        DEFAULT_TEXTURE_BUDGET
    } else {
        u64::from(mib) << 20
    };
    TEXTURE_BUDGET.store(bytes, std::sync::atomic::Ordering::Relaxed);
}
/// What the video memory held when a renderer could not take the block textures: no load
/// of this session asks for more (`lower_texture_cap`).
static TEXTURE_CAP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
/// The bytes the next resource load keeps its block textures to: the setting's budget,
/// under the cap video memory set.
pub fn texture_budget() -> u64 {
    TEXTURE_BUDGET
        .load(std::sync::atomic::Ordering::Relaxed)
        .min(TEXTURE_CAP.load(std::sync::atomic::Ordering::Relaxed))
}
/// The video memory had no room for the block textures at the budget: halves it for the
/// rest of the session. False once it is too small to lower further.
pub fn lower_texture_cap() -> bool {
    let budget = texture_budget();
    let lower = budget / 2;
    if lower < 64 << 20 {
        return false;
    }
    TEXTURE_CAP.store(lower, std::sync::atomic::Ordering::Relaxed);
    true
}
/// Fewer block textures than any game has (vanilla's are about 1300): what a texture is
/// kept at while loading assumes at least these many share the budget.
const LAYERS_AT_LEAST: u64 = 1024;
/// The largest power-of-two side from `start` down (at least `SPRITE_SIZE`) at which
/// `layers` square RGBA layers take at most `bytes`.
fn side_within(start: u32, layers: u64, bytes: u64) -> u32 {
    let mut size = start.max(SPRITE_SIZE).next_power_of_two();
    while size > SPRITE_SIZE && layers * u64::from(size) * u64::from(size) * 4 > bytes {
        size /= 2;
    }
    size
}
/// The colour's and each LabPBR map's share of `budget`: halves, the maps a quarter each.
fn shares(budget: u64, pbr: bool) -> (u64, u64) {
    if pbr {
        (budget / 2, budget / 4)
    } else {
        (budget, 0)
    }
}
/// The mean of `texels` (RGBA). A colour's texels count by their alpha: a cutout's transparent
/// ones often hold black, which would darken what shows around them. A LabPBR map's alpha is
/// data, so its channels (`color` false) are plain means.
fn mean_texel<'a>(texels: impl Iterator<Item = &'a [u8]>, color: bool) -> [u8; 4] {
    let (mut sum, mut weighted, mut n) = ([0u32; 4], [0u64; 3], 0u32);
    for texel in texels {
        for (s, &v) in sum.iter_mut().zip(texel) {
            *s += u32::from(v);
        }
        for (w, &v) in weighted.iter_mut().zip(texel) {
            *w += u64::from(v) * u64::from(texel[3]);
        }
        n += 1;
    }
    let mut out = sum.map(|s| ((s + n / 2) / n) as u8);
    let alpha = u64::from(sum[3]);
    if color && alpha > 0 {
        for (o, w) in out.iter_mut().zip(weighted) {
            *o = ((w + alpha / 2) / alpha) as u8;
        }
    }
    out
}
/// Whether a colour's `pixels` are a cutout: every texel in or out. Averaged down it must stay
/// one, or the partial alphas of the mean would send it to the translucent pass.
fn is_cutout(pixels: &[u8], color: bool) -> bool {
    color && pixels.chunks_exact(4).all(|p| p[3] == 0 || p[3] == 255)
}
/// A cutout's averaged alpha at texel (`x`, `y`) through an ordered dither: in or out, about as
/// many in as the texels it stands for, however sparse they are.
fn cutout_alpha(alpha: u8, x: usize, y: usize) -> u8 {
    const BAYER: [[u16; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    if u16::from(alpha) * 16 > (BAYER[y % 4][x % 4] * 2 + 1) * 255 / 2 {
        255
    } else {
        0
    }
}
/// `frame` (a square RGBA `side`) averaged down by whole powers of two to at most `max`.
fn shrink(frame: Vec<u8>, side: u32, max: u32, color: bool) -> (u32, Vec<u8>) {
    if side <= max || !side.is_power_of_two() || !max.is_power_of_two() {
        return (side, frame);
    }
    let (out_side, f) = (max as usize, (side / max) as usize);
    let side = side as usize;
    let cutout = is_cutout(&frame, color);
    let mut out = Vec::with_capacity(out_side * out_side * 4);
    for y in 0..out_side {
        for x in 0..out_side {
            let texels = (0..f).flat_map(|dy| {
                let row = ((y * f + dy) * side + x * f) * 4;
                frame[row..row + f * 4].chunks_exact(4)
            });
            let mut texel = mean_texel(texels, color);
            if cutout {
                texel[3] = cutout_alpha(texel[3], x, y);
            }
            out.extend(texel);
        }
    }
    (max, out)
}
/// Nearest-resample a square RGBA frame to another square size.
pub(crate) fn resample(frame: &[u8], from: u32, to: u32) -> Vec<u8> {
    if from == to {
        return frame.to_vec();
    }
    let mut out = Vec::with_capacity((to * to * 4) as usize);
    for row in 0..to {
        let source_row = row * from / to;
        for column in 0..to {
            let offset = ((source_row * from + column * from / to) * 4) as usize;
            out.extend_from_slice(&frame[offset..offset + 4]);
        }
    }
    out
}
impl Sprites {
    /// Sprites that reuse textures decoded by an earlier launch, kept in a
    /// cache file in `root`.
    pub fn with_disk_cache(root: &Path) -> Self {
        Self {
            disk: crate::sprite_cache::SpriteCache::open(root),
            ..Self::default()
        }
    }
    /// Write what this run decoded to the cache file.
    pub fn save_disk_cache(&mut self) {
        if let Some(disk) = &mut self.disk {
            disk.save();
        }
    }
    /// Decodes every PNG under `assets/minecraft/textures/<folder>` of `root` on all cores,
    /// ahead of the models asking for their textures one by one: a high-resolution pack's
    /// thousands of images (and LabPBR maps) decode together. Each is averaged down to
    /// what the texture budget keeps of it (animated ones, whose `.mcmeta` may count
    /// frames in texels, aside).
    pub fn prefetch(&mut self, root: &Path, folders: &[&str]) {
        let textures = root.join("assets/minecraft/textures");
        let mut files = Vec::new();
        for folder in folders {
            png_files(&textures.join(folder), &mut files);
        }
        let is_map = |p: &Path| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            stem.strip_suffix("_n")
                .or_else(|| stem.strip_suffix("_s"))
                .is_some_and(|base| p.with_file_name(format!("{base}.png")).is_file())
        };
        // Textures an earlier launch decoded come from the disk cache instead.
        if let Some(disk) = &self.disk {
            files.retain(|p| {
                let name = p
                    .strip_prefix(&textures)
                    .ok()
                    .and_then(|r| r.with_extension("").to_str().map(|r| r.replace('\\', "/")));
                is_map(p) || name.is_none_or(|n| disk.get(&format!("minecraft:{n}")).is_none())
            });
        }
        let maps = files.iter().any(|p| is_map(p));
        let budget = texture_budget();
        let (color_bytes, _) = shares(budget, maps);
        let (_, map_bytes) = shares(budget, true);
        let color_cap = side_within(1 << 14, LAYERS_AT_LEAST, color_bytes);
        let map_cap = side_within(1 << 14, LAYERS_AT_LEAST, map_bytes);
        let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
        let chunk = files.len().div_ceil(threads).max(1);
        let decoded: Vec<(std::path::PathBuf, Image)> = std::thread::scope(|scope| {
            let handles: Vec<_> = files
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .filter_map(|path| {
                                let image = decode(path).ok()?;
                                let image = if path.with_extension("png.mcmeta").exists() {
                                    image
                                } else {
                                    shrink_image(
                                        image,
                                        if is_map(path) { map_cap } else { color_cap },
                                        !is_map(path),
                                    )
                                };
                                Some((path.clone(), image))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        self.prefetched.extend(decoded);
    }
    /// The image at `path`: the prefetched one, or decoded now.
    fn decoded(&mut self, path: &Path) -> Result<Image> {
        match self.prefetched.remove(path) {
            Some(image) => Ok(image),
            None => decode(path),
        }
    }
    /// How many array layers the textures take.
    pub fn count(&self) -> usize {
        self.native.len()
    }
    /// `SpriteContents.computeTransparency(u0, v0, u1, v1)` is NONE: every texel under
    /// the quad's UV box is fully opaque in every frame, so the quad needs no alpha test.
    pub fn region_opaque(&self, layer: u32, uv: &[[f32; 2]; 4]) -> bool {
        if self.opaque[layer as usize] {
            return true;
        }
        let (side, frames, _) = &self.native[layer as usize];
        let side = *side as usize;
        let range = |axis: usize| {
            let lo = uv.iter().map(|p| p[axis]).fold(f32::INFINITY, f32::min);
            let hi = uv.iter().map(|p| p[axis]).fold(f32::NEG_INFINITY, f32::max);
            let clamp = |v: f32| (v.max(0.0) as usize).min(side);
            clamp((lo * side as f32).floor())..clamp((hi * side as f32).ceil())
        };
        let (columns, rows) = (range(0), range(1));
        frames.iter().all(|frame| {
            rows.clone().all(|y| {
                columns
                    .clone()
                    .all(|x| frame[(y * side + x) * 4 + 3] == 255)
            })
        })
    }
    /// Lay every layer out at one size: the largest texture's (resource packs
    /// may ship 32x, 64x... textures, like vanilla's stitched atlas), rounded
    /// up to a power of two and limited to the colour's share of the texture
    /// budget (`set_texture_budget`). Smaller textures are scaled up by whole
    /// pixels, so UVs stay per layer. The LabPBR maps, where there are any, take
    /// one size of their own within their share, at most the colour's.
    pub fn finish(&mut self) {
        self.lay_out(true);
    }
    /// [`Sprites::finish`] for a server, which draws nothing: the layers'
    /// size, but neither their pixels nor the animations' frames.
    pub fn finish_without_pixels(&mut self) {
        self.lay_out(false);
    }
    fn lay_out(&mut self, pixels: bool) {
        self.prefetched = Default::default();
        let layers = self.native.len().max(1) as u64;
        let largest = self.native.iter().map(|n| n.0).max().unwrap_or(SPRITE_SIZE);
        self.minimum_size = self.native.iter().map(|n| n.0).min().unwrap_or(SPRITE_SIZE);
        let has_pbr = self.pbr.iter().flatten().any(Option::is_some);
        let (color_bytes, map_bytes) = shares(texture_budget(), has_pbr);
        let size = side_within(largest, layers, color_bytes);
        let largest_map = self.pbr.iter().flatten().flatten().map(|m| m.0).max();
        let pbr_size =
            largest_map.map_or(SPRITE_SIZE, |l| side_within(l.min(size), layers, map_bytes));
        if !pixels {
            self.pixels = Vec::new();
            self.animations = Vec::new();
            self.pbr = Vec::new();
            self.size = size;
            return;
        }
        log::info!(
            "Horizon: block sprites: {layers} layers at {size} px (largest {largest} px, smallest {}){}",
            self.minimum_size,
            if has_pbr {
                format!(", LabPBR maps at {pbr_size} px")
            } else {
                String::new()
            }
        );
        let mut resized = Vec::with_capacity(self.native.len());
        self.pixels.clear();
        for (from, frames, initial) in &self.native {
            let frames: Arc<Vec<Vec<u8>>> =
                Arc::new(frames.iter().map(|f| resample(f, *from, size)).collect());
            self.pixels.extend_from_slice(&frames[*initial]);
            resized.push(frames);
        }
        for animation in &mut self.animations {
            animation.frames = resized[animation.layer as usize].clone();
        }
        self.size = size;
        self.pbr_size = pbr_size;
        self.normals.clear();
        self.specular.clear();
        if has_pbr {
            let flat = |p: [u8; 4]| p.repeat((pbr_size * pbr_size) as usize);
            for maps in std::mem::take(&mut self.pbr) {
                for (map, (out, default)) in maps.into_iter().zip([
                    (&mut self.normals, FLAT_NORMAL),
                    (&mut self.specular, NO_SPECULAR),
                ]) {
                    match map {
                        Some((from, pixels)) => {
                            let (from, pixels) = shrink(pixels, from, pbr_size, false);
                            out.extend(resample(&pixels, from, pbr_size));
                        }
                        None => out.extend(flat(default)),
                    }
                }
            }
        }
    }
    /// The layer of a texture, added on first use. A texture that is missing
    /// or does not load (a resource pack's filter or a broken file) becomes
    /// vanilla's missing texture, as in a stitched atlas.
    pub fn insert(&mut self, root: &Path, name: &str) -> Result<u32> {
        if let Some(layer) = self.layers.get(name) {
            return Ok(*layer);
        }
        match self.load(root, name) {
            Ok(layer) => Ok(layer),
            Err(e) => {
                // `MissingTextureAtlasSprite.getLocation()`: vanilla models name the
                // missing texture on purpose (block/air.json's particle), #853.
                if name != "minecraft:missingno" {
                    log::warn!("Horizon: using the missing texture for {name}: {e}");
                }
                // MissingTextureAtlasSprite.generateMissingImage: 16x16,
                // magenta and black quarters.
                let frame: Vec<u8> = (0..SPRITE_SIZE * SPRITE_SIZE)
                    .flat_map(|i| {
                        let (x, y) = (i % SPRITE_SIZE, i / SPRITE_SIZE);
                        if (y < SPRITE_SIZE / 2) ^ (x < SPRITE_SIZE / 2) {
                            [248, 0, 248, 255]
                        } else {
                            [0, 0, 0, 255]
                        }
                    })
                    .collect();
                let layer = u32::try_from(self.native.len())?;
                self.opaque.push(true);
                self.translucent.push(false);
                self.pbr.push([None, None]);
                self.mips.push(crate::mipmap::Settings::default());
                self.layers.insert(name.to_owned(), layer);
                self.native.push((SPRITE_SIZE, Arc::new(vec![frame]), 0));
                Ok(layer)
            }
        }
    }
    fn load(&mut self, root: &Path, name: &str) -> Result<u32> {
        let cached = self.disk.as_ref().and_then(|d| d.get(name)).cloned();
        let decoded = match cached {
            Some(decoded) => decoded,
            None => {
                let decoded = self.decode(root, name)?;
                if let Some(disk) = &mut self.disk {
                    disk.insert(name, decoded.clone());
                }
                decoded
            }
        };
        self.register(root, name, decoded)
    }
    /// Read and decode one texture and its `.mcmeta` (the prefetched image when there is one).
    fn decode(&mut self, root: &Path, name: &str) -> Result<crate::sprite_cache::Decoded> {
        let (namespace, path) = name.split_once(':').ok_or("invalid sprite identifier")?;
        let path = root.join(format!("assets/{namespace}/textures/{path}.png"));
        let metadata = match fs::read_to_string(path.with_extension("png.mcmeta")) {
            Ok(text) => Some(serde_json::from_str::<
                BTreeMap<String, Box<serde_json::value::RawValue>>,
            >(&text)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let animation_text = metadata
            .as_ref()
            .and_then(|v| v.get("animation"))
            .map(|v| v.get().to_owned());
        let animation = animation_text
            .as_deref()
            .map(AnimationMetadata::from_json)
            .transpose()?;
        let image = self.decoded(&path)?;
        let info = (image.width, image.height);
        let pixels = image.pixels;
        let (width, height) = if let Some(animation) = &animation {
            let [width, height] =
                animation.frame_size(i32::try_from(info.0)?, i32::try_from(info.1)?);
            (u32::try_from(width)?, u32::try_from(height)?)
        } else {
            (info.0, info.1)
        };
        if width == 0 || height == 0 || info.0 % width != 0 || info.1 % height != 0 {
            return Err(format!("sprite {name} has invalid frame dimensions").into());
        }
        let columns = info.0 / width;
        let count = columns * (info.1 / height);
        // Layers are square; a frame of another shape is nearest-resampled to
        // its longer side so UVs (normalized per layer) are unchanged.
        let side = width.max(height);
        let mut frames = Vec::new();
        for index in 0..count {
            let x = index % columns * width;
            let y = index / columns * height;
            let mut frame = Vec::with_capacity((side * side * 4) as usize);
            for row in 0..side {
                let source_row = y + row * height / side;
                for column in 0..side {
                    let offset = ((source_row * info.0 + x + column * width / side) * 4) as usize;
                    frame.extend_from_slice(&pixels[offset..offset + 4]);
                }
            }
            frames.push(frame);
        }
        let texture = texture_section(metadata.as_ref());
        Ok(crate::sprite_cache::Decoded {
            side,
            frames,
            animation: animation_text,
            strategy: texture
                .as_ref()
                .and_then(|t| t["mipmap_strategy"].as_str())
                .and_then(crate::mipmap::Strategy::parse)
                .unwrap_or_default(),
            alpha_cutoff_bias: texture
                .as_ref()
                .and_then(|t| t["alpha_cutoff_bias"].as_f64())
                .unwrap_or(0.0) as f32,
            has_transparent: pixels.as_chunks::<4>().0.iter().any(|p| p[3] == 0),
            opaque: pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
            translucent: pixels
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[3] != 0 && p[3] != 255),
        })
    }
    /// Give a decoded texture its layer (or the one an identical still has), with the LabPBR
    /// maps beside it in `root`.
    fn register(
        &mut self,
        root: &Path,
        name: &str,
        decoded: crate::sprite_cache::Decoded,
    ) -> Result<u32> {
        let crate::sprite_cache::Decoded {
            side,
            frames,
            animation: animation_text,
            strategy,
            alpha_cutoff_bias,
            has_transparent,
            opaque,
            translucent,
        } = decoded;
        let animation = animation_text
            .as_deref()
            .map(AnimationMetadata::from_json)
            .transpose()?;
        let count = frames.len() as u32;
        let layer = u32::try_from(self.native.len())?;
        // LabPBR's maps beside the texture, laid out as its frames (a column of them): the
        // first frame of each.
        let pbr = match name.split_once(':') {
            Some((namespace, path)) => {
                let path = root.join(format!("assets/{namespace}/textures/{path}.png"));
                ["_n", "_s"].map(|suffix| self.pbr_map(&path, suffix, count))
            }
            None => [None, None],
        };
        // The frames are kept no larger than the colour's share of the budget allows any
        // game's textures (`finish` would sample them down to it).
        let (color_bytes, _) = shares(texture_budget(), pbr.iter().any(Option::is_some));
        let cap = side_within(side, LAYERS_AT_LEAST, color_bytes);
        let (side, frames) = if side > cap && side.is_power_of_two() {
            let frames: Vec<_> = frames
                .into_iter()
                .map(|f| shrink(f, side, cap, true).1)
                .collect();
            (cap, frames)
        } else {
            (side, frames)
        };
        let settings = crate::mipmap::Settings {
            native_size: side,
            strategy,
            alpha_cutoff_bias,
            has_transparent,
            item: name
                .split_once(':')
                .is_some_and(|(_, p)| p.starts_with("item/")),
        };
        let still = (animation.is_none() && frames.len() == 1).then(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            pbr.hash(&mut hasher);
            (
                frames[0].clone(),
                hasher.finish(),
                side,
                [
                    settings.strategy as u32,
                    settings.alpha_cutoff_bias.to_bits(),
                ],
                [settings.has_transparent, settings.item, opaque, translucent],
            )
        });
        if let Some(&same) = still.as_ref().and_then(|k| self.still.get(k)) {
            self.layers.insert(name.to_owned(), same);
            return Ok(same);
        }
        if let Some(key) = still {
            self.still.insert(key, layer);
        }
        let frames = Arc::new(frames);
        let mut initial = 0;
        if let Some(clip) = animation
            .as_ref()
            .map(|m| m.clip(count))
            .transpose()?
            .flatten()
        {
            initial = clip.sample(0).frame as usize;
            self.animations.push(SpriteAnimation {
                layer,
                frames: frames.clone(),
                clip: Arc::new(clip),
            });
        }
        self.mips.push(settings);
        self.opaque.push(opaque);
        self.translucent.push(translucent);
        self.pbr.push(pbr);
        self.layers.insert(name.to_owned(), layer);
        self.native.push((side, frames, initial));
        Ok(layer)
    }
    /// The LabPBR map with `suffix` beside the texture at `path` whose frames lie in a column
    /// of `count`: its first frame, square, no larger than the maps' share of the budget
    /// allows any game's textures (a 512x pack's maps would otherwise fill the memory first).
    fn pbr_map(&mut self, path: &Path, suffix: &str, count: u32) -> Option<(u32, Vec<u8>)> {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let map = path.with_file_name(format!("{stem}{suffix}.png"));
        let Image {
            width: map_width,
            height: map_height,
            pixels,
        } = self.decoded(&map).ok()?;
        let (w, h) = (map_width, map_height / count.max(1));
        if w == 0 || h == 0 {
            return None;
        }
        let side = w.max(h);
        let mut frame = Vec::with_capacity((side * side * 4) as usize);
        for row in 0..side {
            for column in 0..side {
                let offset = (((row * h / side) * map_width + column * w / side) * 4) as usize;
                frame.extend_from_slice(&pixels[offset..offset + 4]);
            }
        }
        let (_, map_bytes) = shares(texture_budget(), true);
        Some(shrink(
            frame,
            side,
            side_within(side, LAYERS_AT_LEAST, map_bytes),
            false,
        ))
    }
}

/// A decoded image: its width, height and RGBA pixels.
pub(crate) struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}
/// A PNG's size and its pixels as RGBA.
fn decode(path: &Path) -> Result<Image> {
    let mut decoder = png::Decoder::new(BufReader::new(File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("sprite exceeds decoder limits")?
    ];
    let info = reader.next_frame(&mut bytes)?;
    let channels = info.color_type.samples();
    let mut pixels = Vec::with_capacity(info.width as usize * info.height as usize * 4);
    for pixel in bytes[..info.buffer_size()].chunks_exact(channels) {
        let rgba = match info.color_type {
            png::ColorType::Rgba => [pixel[0], pixel[1], pixel[2], pixel[3]],
            png::ColorType::Rgb => [pixel[0], pixel[1], pixel[2], 255],
            png::ColorType::Grayscale => [pixel[0], pixel[0], pixel[0], 255],
            png::ColorType::GrayscaleAlpha => [pixel[0], pixel[0], pixel[0], pixel[1]],
            _ => return Err("unsupported expanded sprite format".into()),
        };
        pixels.extend(rgba);
    }
    Ok(Image {
        width: info.width,
        height: info.height,
        pixels,
    })
}

/// `image` averaged down by the largest power of two that keeps its shorter side at least
/// `max` and divides both sides.
fn shrink_image(image: Image, max: u32, color: bool) -> Image {
    let short = image.width.min(image.height);
    let mut f = 1;
    while short / (f * 2) >= max.max(1)
        && short / f > max
        && image.width % (f * 2) == 0
        && image.height % (f * 2) == 0
    {
        f *= 2;
    }
    if f == 1 {
        return image;
    }
    let (w, h, f) = (image.width as usize, image.height as usize, f as usize);
    let (ow, oh) = (w / f, h / f);
    let cutout = is_cutout(&image.pixels, color);
    let mut out = Vec::with_capacity(ow * oh * 4);
    for y in 0..oh {
        for x in 0..ow {
            let texels = (0..f).flat_map(|dy| {
                let row = ((y * f + dy) * w + x * f) * 4;
                image.pixels[row..row + f * 4].chunks_exact(4)
            });
            let mut texel = mean_texel(texels, color);
            if cutout {
                texel[3] = cutout_alpha(texel[3], x, y);
            }
            out.extend(texel);
        }
    }
    Image {
        width: ow as u32,
        height: oh as u32,
        pixels: out,
    }
}

/// Every PNG under `dir`, recursively.
fn png_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            png_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "png") {
            out.push(path);
        }
    }
}

/// SpriteContents: the `.mcmeta` `texture` section (TextureMetadataSection's
/// mipmap strategy and alpha cutoff bias).
fn texture_section(
    metadata: Option<&BTreeMap<String, Box<serde_json::value::RawValue>>>,
) -> Option<serde_json::Value> {
    metadata
        .and_then(|v| v.get("texture"))
        .and_then(|v| serde_json::from_str(v.get()).ok())
}

#[cfg(test)]
pub(crate) fn test_animation(size: u32, layer: u32) -> SpriteAnimation {
    test_clip(size, layer, r#"{"frametime":1,"frames":[0,1]}"#)
}
#[cfg(test)]
pub(crate) fn test_clip(size: u32, layer: u32, metadata: &str) -> SpriteAnimation {
    let frame = |blue: u8| {
        (0..size * size)
            .flat_map(|i| [(i % size) as u8, (i / size) as u8, blue, 255])
            .collect()
    };
    test_frames(layer, vec![frame(0), frame(99)], metadata)
}
#[cfg(test)]
pub(crate) fn test_frames(layer: u32, frames: Vec<Vec<u8>>, metadata: &str) -> SpriteAnimation {
    let metadata = AnimationMetadata::from_json(metadata).unwrap();
    SpriteAnimation {
        layer,
        clip: Arc::new(metadata.clip(frames.len() as u32).unwrap().unwrap()),
        frames: Arc::new(frames),
    }
}

/// How many reduced copies a sprite atlas keeps: the Mipmap Levels option plus the
/// resampling-only levels an upscaled pack needs, within what the atlas holds.
pub(crate) fn sprite_mipmap_limit(requested: u32, size: u32, minimum: u32, available: u32) -> u32 {
    // Only the atlas native minimum determines how many resampling-only mips to skip.
    let upscale = (size / minimum.max(1)).max(1).ilog2();
    requested.saturating_add(upscale).min(available - 1)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_cutouts_black_transparent_texels_do_not_darken_it_when_shrunk() {
        let frame = [
            [200, 180, 100, 255],
            [0, 0, 0, 0],
            [200, 180, 100, 255],
            [0, 0, 0, 0],
        ]
        .concat();
        let (side, color) = super::shrink(frame.clone(), 2, 1, true);
        assert_eq!((side, color), (1, vec![200, 180, 100, 255]));
        // A map's alpha is data: its channels stay plain means.
        assert_eq!(super::shrink(frame, 2, 1, false).1, vec![100, 90, 50, 128]);
    }
    #[test]
    fn a_cutout_stays_one_when_shrunk_and_keeps_its_coverage() {
        // One texel in four solid, as sparse as redstone dust: 64 texels to 16 of them.
        let side = 64usize;
        let mut frame = vec![0u8; side * side * 4];
        for y in 0..side {
            for x in 0..side {
                if x % 2 == 0 && y % 2 == 0 {
                    frame[(y * side + x) * 4..][..4].copy_from_slice(&[120, 20, 20, 255]);
                }
            }
        }
        let (_, small) = super::shrink(frame, 64, 16, true);
        let alphas: Vec<u8> = small.chunks_exact(4).map(|p| p[3]).collect();
        assert!(alphas.iter().all(|&a| a == 0 || a == 255));
        let solid = alphas.iter().filter(|&&a| a == 255).count();
        assert_eq!(solid, alphas.len() / 4);
        // Real partial alpha (glass) is not a cutout and stays partial.
        let glass = [120u8, 120, 120, 100].repeat(4);
        assert_eq!(super::shrink(glass, 2, 1, true).1[3], 100);
    }
    use super::*;
    #[test]
    fn mipmap_option_keeps_real_16px_reductions() {
        for levels in 0..=4 {
            assert_eq!(sprite_mipmap_limit(levels, 32, 16, 6), levels + 1);
            assert_eq!(sprite_mipmap_limit(levels, 16, 16, 5), levels);
        }
    }
    #[test]
    fn identical_still_textures_share_a_layer() {
        let root =
            std::env::temp_dir().join(format!("horizon-sprite-share-{}", std::process::id()));
        let directory = root.join("assets/a/textures/block");
        fs::create_dir_all(&directory).unwrap();
        let write = |name: &str, red: u8| {
            let mut encoder =
                png::Encoder::new(File::create(directory.join(name)).unwrap(), 16, 16);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let pixels: Vec<u8> = (0..256).flat_map(|i| [red, i as u8, 0, 255]).collect();
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&pixels)
                .unwrap();
        };
        write("one.png", 10);
        write("copy.png", 10);
        write("other.png", 20);
        let mut sprites = Sprites::default();
        let one = sprites.insert(&root, "a:block/one").unwrap();
        let copy = sprites.insert(&root, "a:block/copy").unwrap();
        let other = sprites.insert(&root, "a:block/other").unwrap();
        // The same pixels as an item sprite mip differently: a layer of its own.
        fs::create_dir_all(root.join("assets/a/textures/item")).unwrap();
        fs::copy(
            directory.join("one.png"),
            root.join("assets/a/textures/item/one.png"),
        )
        .unwrap();
        let item = sprites.insert(&root, "a:item/one").unwrap();
        fs::remove_dir_all(&root).unwrap();
        assert_eq!((one, copy, other, item), (0, 0, 1, 2));
        assert_eq!((sprites.count(), sprites.layers.len()), (3, 4));
        assert_eq!((sprites.opaque.len(), sprites.mips.len()), (3, 3));
        sprites.finish();
        assert_eq!(sprites.pixels.len(), 3 * 16 * 16 * 4);
    }
    #[test]
    fn a_32px_only_atlas_does_not_add_an_extra_mipmap() {
        let mut sprites = Sprites {
            native: vec![
                (32, Arc::new(vec![vec![255; 32 * 32 * 4]]), 0),
                (32, Arc::new(vec![vec![128; 32 * 32 * 4]]), 0),
            ],
            ..Default::default()
        };
        sprites.finish();
        assert_eq!((sprites.size, sprites.minimum_size), (32, 32));
        for level in 0..=4 {
            assert_eq!(
                sprite_mipmap_limit(level, sprites.size, sprites.minimum_size, 6),
                level
            );
        }
        sprites
            .native
            .push((16, Arc::new(vec![vec![255; 16 * 16 * 4]]), 0));
        sprites.finish();
        assert_eq!((sprites.size, sprites.minimum_size), (32, 16));
        assert_eq!(
            sprite_mipmap_limit(0, sprites.size, sprites.minimum_size, 6),
            1
        );
    }

    #[test]
    fn hud_animation_resampling_uses_the_actual_frame_size() {
        for size in [16, 32, 64] {
            let animation = test_animation(size, 0);
            let resized = animation.resized_pixels(1, 16, 16);
            let last = (15 * size / 16) as u8;
            assert_eq!(&resized[15 * 4..16 * 4], &[last, 0, 99, 255]);
            assert_eq!(&resized[255 * 4..256 * 4], &[last, last, 99, 255]);
        }
    }

    #[test]
    fn native_fire_sprites_register_full_animation_sequences() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.local/horizon-client")
            .join(horizon_core::VANILLA_VERSION);
        if !root
            .join("assets/minecraft/textures/block/fire_0.png")
            .exists()
        {
            return;
        }
        let mut sprites = Sprites::default();
        for name in ["fire_0", "fire_1", "soul_fire_0", "soul_fire_1"] {
            let layer = sprites
                .insert(&root, &format!("minecraft:block/{name}"))
                .unwrap();
            let animation = sprites
                .animations
                .iter()
                .find(|a| a.layer == layer)
                .expect("fire must be animated");
            assert_ne!(
                animation.pixels(animation.frame(0)),
                animation.pixels(animation.frame(1))
            );
            assert_ne!(
                animation.resized_pixels(animation.frame(0), 16, 16),
                animation.resized_pixels(animation.frame(1), 16, 16)
            );
            assert_eq!(
                animation.pixels(animation.frame(0)).len(),
                (SPRITE_SIZE * SPRITE_SIZE * 4) as usize
            );
        }
    }

    #[test]
    fn native_32_pixel_animation_frames_keep_every_texel() {
        let root = std::env::temp_dir().join(format!(
            "horizon-sprite-detail-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let directory = root.join("assets/test/textures");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("flow.png");
        let mut pixels = Vec::new();
        for y in 0..64u8 {
            for x in 0..32u8 {
                pixels.extend([x, y, 0, 255]);
            }
        }
        let mut encoder = png::Encoder::new(File::create(&path).unwrap(), 32, 64);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&pixels)
            .unwrap();
        fs::write(path.with_extension("png.mcmeta"), r#"{"animation":{}}"#).unwrap();
        let mut sprites = Sprites::default();
        sprites.insert(&root, "test:flow").unwrap();
        sprites.finish();
        let size = sprites.size;
        let initial = sprites.pixels.clone();
        let animation = sprites.animations[0].clone();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(
            size, 32,
            "half a flowing-water frame must retain 16 texels, not 8"
        );
        assert_eq!(initial, pixels[..32 * 32 * 4]);
        assert_eq!(animation.pixels(animation.frame(1)), &pixels[32 * 32 * 4..]);
    }

    #[test]
    fn animation_boundaries_and_repeated_frames_do_not_depend_on_render_rate() {
        let animation = SpriteAnimation {
            layer: 0,
            frames: Arc::new(vec![vec![], vec![], vec![]]),
            clip: Arc::new(AnimationMetadata::from_json(r#"{"frames":[{"index":2,"time":8},{"index":1,"time":3},{"index":1,"time":2},0]}"#).unwrap().clip(3).unwrap().unwrap()),
        };
        for (tick, frame) in [
            (0, 2),
            (7, 2),
            (8, 1),
            (10, 1),
            (11, 1),
            (12, 1),
            (13, 0),
            (14, 2),
            (29, 2),
            (u64::MAX, 2),
        ] {
            assert_eq!(animation.frame(tick), frame);
        }
    }

    #[test]
    fn interpolated_animations_blend_towards_the_next_frame() {
        let animation = test_clip(16, 0, r#"{"frametime":4,"interpolate":true}"#);
        let blend = |frame, next, millis| FrameBlend {
            frame,
            next,
            millis,
        };
        for (tick, expected) in [
            (0, blend(0, 0, 0)),
            (1, blend(0, 1, 250)),
            (3, blend(0, 1, 750)),
            (4, blend(1, 1, 0)),
            (5, blend(1, 0, 250)),
            (8, blend(0, 0, 0)),
        ] {
            assert_eq!(animation.blend(tick), expected, "tick {tick}");
        }
        let stepped = test_clip(16, 0, r#"{"frametime":4}"#);
        assert_eq!(stepped.blend(1), blend(0, 0, 0));
        assert_eq!(stepped.blend(5), blend(1, 1, 0));
    }
}
