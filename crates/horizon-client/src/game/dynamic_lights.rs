//! Horizon Extras' dynamic lights (`horizon_client::dynamic_light`): the light sources near the
//! camera this frame. Nothing is gathered while the extra is off.
use super::{Game, shield};
use glam::DVec3;
use horizon_client::dynamic_light::{self, BURNING, Luminance, Source};

impl Game<'_> {
    /// Fills `dynamic_lights` for the frame: the player's and mobs' held items and fire, and
    /// glowing dropped items, the nearest [`dynamic_light::MAX`] of them.
    pub(super) fn gather_dynamic_lights(&mut self, camera: DVec3, alpha: f64) {
        let mut out = std::mem::take(&mut self.dynamic_lights);
        out.clear();
        self.dynamic_states.clear();
        if !self.settings.extras.dynamic_lights {
            self.dynamic_lights = out;
            return;
        }
        if self.luminance.is_none() {
            self.luminance = Some(Luminance::new(self.res.facts(), self.pack.assets.registry));
        }
        let lum = self.luminance.as_ref().expect("built above");
        let held = |item: Option<&str>| item.map_or((0, None), |i| (lum.of(i), lum.state_of(i)));
        // The brighter of two hands, with what it holds; fire if nothing held outshines it.
        let brighter = |a: (u8, Option<u32>), b: (u8, Option<u32>), on_fire: bool| {
            let top = if b.0 > a.0 { b } else { a };
            if on_fire && BURNING > top.0 {
                (BURNING, None)
            } else {
                top
            }
        };
        let mut sources = Vec::new();
        if let Some(p) = &self.player
            && !p.spectator
        {
            let feet = self
                .previous_position
                .lerp(p.position, self.clock.camera_fraction());
            let (level, state) = brighter(
                held(self.stack_in(self.selected).map(|s| &*s.item)),
                held(self.stack_in(shield::OFFHAND_SLOT).map(|s| &*s.item)),
                p.fire_ticks > 0,
            );
            // About where the hands are.
            sources.push(Source {
                at: DVec3::new(feet.x, feet.y + 1.2, feet.z),
                level,
                state,
            });
        }
        for m in self.mobs.iter().filter(|m| !m.removed) {
            let (level, state) = brighter(
                held(m.extra.main_hand.as_deref()),
                held(m.extra.off_hand.as_deref()),
                m.fire_ticks > 0,
            );
            if level == 0 {
                continue;
            }
            let at = m.previous.lerp(m.position, alpha);
            sources.push(Source {
                at: DVec3::new(at.x, at.y + f64::from(m.height()) * 0.6, at.z),
                level,
                state,
            });
        }
        for item in &self.dropped {
            let (level, state) = brighter(
                held(Some(&*item.stack.item)),
                (0, None),
                item.fire_ticks > 0,
            );
            if level == 0 {
                continue;
            }
            let at = item.previous.lerp(item.position, alpha);
            sources.push(Source {
                at: DVec3::new(at.x, at.y + 0.25, at.z),
                level,
                state,
            });
        }
        let mut states = std::mem::take(&mut self.dynamic_states);
        dynamic_light::select(&sources, camera, &mut out, &mut states);
        self.dynamic_states = states;
        self.dynamic_lights = out;
    }
}
