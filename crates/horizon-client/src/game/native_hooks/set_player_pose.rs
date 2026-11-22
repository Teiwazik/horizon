use super::*;

impl Game<'_> {
    /// Asks the mods to pose a player's body this frame (`set_player_pose`): `player` at the frame's
    /// partial tick, `who` its name and whether it is this window's own, `pose` its parts at
    /// rest and as vanilla posed them, which the mods set.
    pub(in crate::game) fn native_set_player_pose(
        &self,
        player: &MobRender<'_>,
        who: &PlayerRender<'_>,
        pose: &mut PlayerBodyPose,
    ) {
        self.fire_set_player_pose(player, who, pose);
    }
}
