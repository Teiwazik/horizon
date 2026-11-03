//! Horizon's own shader packs in the game (docs/shaders.md): the pick in `shaderpacks/` (a
//! folder with `horizon-shaders.toml`, or a `.hshader`; `game/shader_packs.rs`) runs once it
//! passes `horizon_shader_api::check`. The pack is looked at twice a second: a saved file (or
//! saved option values) is checked and the pack rebuilt, and a
//! pack that fails is refused with its problems in the chat while the last good one keeps
//! drawing.

use std::{
    hash::{Hash, Hasher},
    path::Path,
    time::{Duration, Instant},
};

use horizon_client::text::Text;
use horizon_shader_api::{Pack, check};

use super::Game;

const LOOK_EVERY: Duration = Duration::from_millis(500);
const RED: u32 = 0xFF5555;

#[derive(Default)]
pub(super) struct NativeShaders {
    /// The pack's name in `shaderpacks/`, of either kind (`game/shader_packs.rs`); empty: none.
    pub picked: String,
    /// The pack's files when it was last read (`fingerprint`).
    stamp: Option<u64>,
    looked: Option<Instant>,
}

impl NativeShaders {
    pub fn picking(name: &str) -> Self {
        Self {
            picked: name.to_owned(),
            ..Self::default()
        }
    }
}

impl Game<'_> {
    /// Builds the pack at `path` when its files changed since it was last read.
    pub(super) fn tick_native_shader_pack(&mut self, path: &Path) {
        let now = Instant::now();
        let state = &mut self.native_shaders;
        if state.looked.is_some_and(|t| now - t < LOOK_EVERY) {
            return;
        }
        state.looked = Some(now);
        let stamp = fingerprint(path);
        if stamp == state.stamp {
            return;
        }
        let reload = state.stamp.is_some();
        state.stamp = stamp;
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        let result = Pack::read(path).map_err(|e| vec![e]).and_then(|pack| {
            let read_ms = now.elapsed().as_millis();
            let report = check(&pack)
                .map_err(|p| p.iter().map(ToString::to_string).collect::<Vec<String>>())?;
            log::info!("shader pack read in {read_ms} ms, checked by {} ms", now.elapsed().as_millis());
            // The player's values (Shader Pack Settings), else the defaults.
            let options = &pack.manifest.options;
            let values = horizon_client::shader_pack_list::load_values(path, options);
            let constants = options.iter().map(|o| o.id.clone()).zip(values).collect();
            // Each sprite layer's `[[materials]]` id, by the layer's texture name.
            let names = &self.pack.render.sprite_layers;
            let layers = names.values().max().map_or(0, |&l| l as usize + 1);
            let mut materials = vec![0u32; layers];
            for (name, &layer) in names.iter() {
                let id = report.manifest.material_of(name);
                if id != 0 {
                    materials[layer as usize] = id;
                }
            }
            // `blocks` lists: the layers their models draw with, where no pattern named them.
            let listed: String = report
                .manifest
                .materials
                .iter()
                .filter(|m| !m.blocks.is_empty())
                .map(|m| format!("block.{}={}\n", m.id, m.blocks.join(" ")))
                .collect();
            if !listed.is_empty() {
                let macros = horizon_shaders::Macros::standard();
                let ids = horizon_shaders::BlockIds::parse(&horizon_shaders::Properties::parse(
                    &listed, &macros,
                ));
                let table = super::shader_packs::block_id_table(&self.pack, &ids);
                for (layer, id) in materials.iter_mut().enumerate() {
                    if *id == 0
                        && let Some(&listed) = table.get(layer * 2)
                        && listed > 0
                    {
                        *id = listed as u32;
                    }
                }
            }
            // Each light-giving block state's `[[lights]]` colour, with its level.
            let mut lights = horizon_client::gfx::block_light::LightTable::default();
            if !report.manifest.lights.is_empty() {
                for (id, state) in self.res.facts().states.iter().enumerate() {
                    if state.emission == 0 {
                        continue;
                    }
                    // A source the pack gives no colour still lights: warm, as block light is.
                    let color = report
                        .manifest
                        .light_of(&state.name)
                        .unwrap_or([1.0, 0.75, 0.5]);
                    lights.insert(id as u32, (color, state.emission));
                }
            }
            // `[[materials]] translucent`: the layers of those ids, for the mesher.
            let translucent_ids: Vec<u32> = report
                .manifest
                .materials
                .iter()
                .filter(|m| m.translucent)
                .map(|m| m.id)
                .collect();
            let translucent: Vec<bool> = materials
                .iter()
                .map(|id| translucent_ids.contains(id))
                .collect();
            let labels = renderer
                .set_native_shader_pack(Some((report, constants)), &materials, lights)
                .map_err(|e| vec![e])?;
            Ok((pack.manifest, labels, translucent))
        });
        // The resources were loaded without a native pack, so without the resource packs'
        // LabPBR maps its material stage reads: load them once, now that one runs.
        if result.as_ref().is_ok_and(|(m, _, _)| m.material.is_some())
            && !horizon_client::sprites::labpbr_maps()
        {
            horizon_client::sprites::set_labpbr_maps(true);
            log::info!("Horizon: reloading resources for the LabPBR maps a native pack reads");
            self.reload_packs();
        }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        match result {
            Ok((m, labels, translucent)) => {
                self.set_translucent_layers(translucent);
                log::info!(
                    "shader pack {} {}: {} in {} ms",
                    m.id,
                    m.version,
                    labels.join(", "),
                    now.elapsed().as_millis()
                );
                if reload {
                    self.system_message(Text::literal(format!("Shader pack {} reloaded", m.name)));
                }
            }
            Err(problems) => {
                log::warn!("shader pack {name} refused: {}", problems.join("; "));
                self.system_message(Text::colored(format!("Shader pack {name} refused:"), RED));
                for p in problems {
                    self.system_message(Text::colored(p, RED));
                }
            }
        }
    }
}

/// A hash of the pack's file names, sizes and modification times, and of its option values
/// (`None` when unreadable).
fn fingerprint(path: &Path) -> Option<u64> {
    fn walk(path: &Path, hasher: &mut std::hash::DefaultHasher) -> Option<()> {
        let meta = std::fs::metadata(path).ok()?;
        path.hash(hasher);
        if meta.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(path)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .collect();
            entries.sort();
            for entry in entries {
                walk(&entry, hasher)?;
            }
        } else {
            meta.len().hash(hasher);
            meta.modified().ok()?.hash(hasher);
        }
        Some(())
    }
    let mut hasher = std::hash::DefaultHasher::new();
    walk(path, &mut hasher)?;
    std::fs::read(horizon_client::shader_pack_list::values_path(path))
        .unwrap_or_default()
        .hash(&mut hasher);
    Some(hasher.finish())
}
