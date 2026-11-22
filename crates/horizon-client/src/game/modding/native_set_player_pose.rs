//! The mods' `set_player_pose` hook: once a frame, before the entity pass draws, each player model (this
//! window's own and the other players of the level) is posed as vanilla and its emote pose it, and
//! the mods may set its head, body, arms and legs. What they set is kept on the model for the
//! frame (`Mob::mod_pose`) and laid over by `anim::setup`, which every draw of the model asks:
//! the body, its outer layers, armour, held items, cape, the mods' layers and the GPU-posed skin.
//! With no listener nothing is posed or built.
use super::Game;
use super::native_mob_draw::{PlayerRender, mob_render, render_id};
use horizon_client::mobs::{Mob, anim, render};
use mod_api::{HookId, PlayerBodyPose};
use std::cell::Cell;

thread_local! {
    /// Whether a model may still carry a pose from an earlier frame: set while a mod listens, so
    /// the frame after the last listener left clears them once.
    static POSED: Cell<bool> = const { Cell::new(false) };
}

/// Whether `m` is another player's model (not a mannequin's).
fn is_player(m: &Mob) -> bool {
    &*m.kind.id == "minecraft:player"
}

impl Game<'_> {
    /// Asks the mods to pose each player model at `partial` this frame (`set_player_pose`), before
    /// any model is drawn. With no listener it returns at once, after clearing the poses once
    /// when the last listener left.
    pub(in crate::game) fn pose_players(&mut self, partial: f32) {
        if !mod_api::__private::listened(HookId::SetPlayerPose) {
            if POSED.replace(false) {
                if let Some(m) = self.player_model.as_mut() {
                    m.mod_pose = None;
                }
                for m in self.mobs.iter_mut() {
                    m.mod_pose = None;
                }
            }
            return;
        }
        POSED.set(true);
        // Asked with the models borrowed: what each mod set, applied after.
        let own = self.player_model.as_ref().map(|m| {
            let name = self.own_profile_name();
            let who = PlayerRender {
                name: &name,
                local: true,
            };
            self.player_mod_pose(m, &who, partial)
        });
        let others: Vec<(u32, Option<Box<anim::ModPose>>)> = self
            .mobs
            .iter()
            .filter(|m| is_player(m))
            .map(|m| {
                // As `render_player_pre`: a player is told by its name, and one whose name the
                // client doesn't know is asked nothing.
                let posed = self.remote_player_name(m.id).and_then(|name| {
                    let who = PlayerRender { name, local: false };
                    self.player_mod_pose(m, &who, partial)
                });
                (m.id, posed)
            })
            .collect();
        if let (Some(m), Some(posed)) = (self.player_model.as_mut(), own) {
            m.mod_pose = posed;
        }
        let mut others = others.into_iter().peekable();
        for m in self.mobs.iter_mut().filter(|m| is_player(m)) {
            if let Some((_, posed)) = others.next_if(|(id, _)| *id == m.id) {
                m.mod_pose = posed;
            }
        }
    }

    /// The parts the mods set on `m` this frame (`who`), or None when they set none.
    fn player_mod_pose(
        &self,
        m: &Mob,
        who: &PlayerRender<'_>,
        partial: f32,
    ) -> Option<Box<anim::ModPose>> {
        let (key, _) = render::body(m);
        let model = m.kind.layer(&key).unwrap_or(&m.kind.model);
        let mut poses = anim::Poses::pooled();
        poses.reset(model);
        const IDENTITY: [f32; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let read = |poses: &anim::Poses| {
            anim::BODY_PARTS.map(|part| poses.pose_of(part).unwrap_or(IDENTITY))
        };
        let at_rest = read(&poses);
        let state = anim::state(m, partial);
        anim::setup_unposed(m, &state, &mut poses, &self.res.mob_data().animations);
        let mut pose = PlayerBodyPose::new(at_rest, read(&poses));
        let render = mob_render(m, render_id(m.id, who.local, self.own_entity_id()), partial);
        self.native_set_player_pose(&render, who, &mut pose);
        if pose.is_empty() {
            return None;
        }
        Some(Box::new(mod_api::POSABLE_PARTS.map(|part| {
            pose.is_set(part)
                .then(|| pose.get(part))
                .flatten()
                .map(|p| {
                    [
                        p.x, p.y, p.z, p.x_rot, p.y_rot, p.z_rot, p.x_scale, p.y_scale, p.z_scale,
                    ]
                })
        })))
    }
}
