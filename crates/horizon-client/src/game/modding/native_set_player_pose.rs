//! The mods' `set_player_pose` hook: once a frame, before the HUD and the entity pass draw, each player model (this
//! window's own and the other players of the level) is posed as vanilla and its emote pose it, and
//! the mods may set its head, body, arms and legs, bend its elbows, knees and waist, and move, turn
//! and scale the whole model (a back flip). What they set is kept on the model for the
//! frame (`Mob::mod_pose`) and laid over by `anim::setup`, which every draw of the model asks:
//! the body, its outer layers, armour, held items, cape, the mods' layers and the GPU-posed skin.
//! With no listener nothing is posed or built.
use super::Game;
use super::native_mob_draw::{PlayerRender, mob_render, render_id};
use horizon_client::mobs::{Mob, anim, render};
use mod_api::{HookId, PlayerBodyPose};

/// Whether `m` is another player's model (not a mannequin's).
fn is_player(m: &Mob) -> bool {
    &*m.kind.id == "minecraft:player"
}

impl Game<'_> {
    /// Asks the mods to pose each player model this frame (`set_player_pose`), before any model is
    /// drawn: the others at the level's `partial`, the window's own at the partial tick it is
    /// drawn with (`own_player_alpha`: smooth while the level is frozen, a replay's clock while
    /// one plays). With no listener it returns at once, after clearing the poses once when the
    /// last listener left.
    pub(in crate::game) fn pose_players(&mut self, partial: f32) {
        if !mod_api::__private::listened(HookId::SetPlayerPose) {
            if std::mem::take(&mut self.players_posed) {
                if let Some(m) = self.player_model.as_mut() {
                    m.mod_pose = None;
                }
                for m in self.mobs.iter_mut() {
                    m.mod_pose = None;
                }
            }
            return;
        }
        self.players_posed = true;
        // Asked with the models borrowed: what each mod set, applied after.
        let own = self.player_model.as_ref().map(|m| {
            let name = self.own_profile_name();
            let who = PlayerRender {
                name: &name,
                local: true,
            };
            let own_partial = self.own_player_alpha() as f32;
            self.player_mod_pose(m, &who, self.own_uuid(), own_partial)
        });
        let others: Vec<(u32, Option<Box<anim::ModPose>>)> = self
            .mobs
            .iter()
            .filter(|m| is_player(m))
            .map(|m| {
                // As `render_player_pre`: a player is told by its name and UUID, and one whose
                // name or UUID the client doesn't know is asked nothing.
                let name = self.remote_player_name(m.id);
                let posed = name
                    .zip(self.remote_player_uuid(m.id))
                    .and_then(|(name, uuid)| {
                        let who = PlayerRender { name, local: false };
                        self.player_mod_pose(m, &who, uuid, partial)
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

    /// The parts the mods set on `m` this frame (`who`, `uuid`), or None when they set none.
    fn player_mod_pose(
        &self,
        m: &Mob,
        who: &PlayerRender<'_>,
        uuid: u128,
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
        self.native_set_player_pose(&render, who, uuid, &mut pose);
        if pose.is_empty() {
            return None;
        }
        Some(Box::new(anim::ModPose {
            parts: mod_api::POSABLE_PARTS.map(|part| {
                pose.is_set(part)
                    .then(|| pose.get(part))
                    .flatten()
                    .map(|p| {
                        [
                            p.x, p.y, p.z, p.x_rot, p.y_rot, p.z_rot, p.x_scale, p.y_scale,
                            p.z_scale,
                        ]
                    })
            }),
            bends: mod_api::BENDABLE_PARTS.map(|part| pose.bend(part).map(|(a, b)| [a, b])),
            root: pose.root().map(|p| {
                [
                    p.x, p.y, p.z, p.x_rot, p.y_rot, p.z_rot, p.x_scale, p.y_scale, p.z_scale,
                ]
            }),
        }))
    }
}
