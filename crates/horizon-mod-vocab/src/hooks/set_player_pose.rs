//! The `set_player_pose` hook: declarations the mod API generator reads (see `hooks/mod.rs`).

pub trait Hooks {
    /// Pose a player's body this frame, as an emote or a body tracker does: once a frame for each
    /// other player in the level and for this window's player, before any of them is drawn, and
    /// only while a mod listens. `player` is its render state at the frame's partial tick (its
    /// `partial` and `age` time an animation between ticks), `who` its profile name and whether it
    /// is this window's own player, and `player_uuid` its profile's UUID, as the server knows the
    /// player (`ServerPlayer::uuid`, what a mod's payloads name it by). `pose` holds the head, body, arms and legs at rest and as
    /// vanilla posed them this frame (a Horizon emote included); `pose.set(part, part_pose)` draws
    /// that part as the mod wants it this frame, `pose.set_bend(part, axis, angle)` bends an arm or
    /// leg at the elbow or knee, `pose.reset(part)` draws it as vanilla again. The model's
    /// outer layers follow the part they cover, and so do its armour, held items, cape, the layers
    /// mods added and the parts `render_living_post` reads. Nothing is kept: a part no listener
    /// set next frame draws as vanilla poses it. Visual only, on this client.
    #[hook(side = "client", contract = "listen")]
    fn set_player_pose();
}
