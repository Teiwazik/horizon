//! The body of a player model as a mod poses it this frame (`set_player_pose`): the six parts an
//! emote or a body tracker turns and moves, each as vanilla posed it and, once a mod sets it, as
//! the mod wants it drawn. The model's outer layers follow the part they cover (the hat its head,
//! a sleeve its arm), and so do its armour, held items, cape and the layers mods added. The arms and
//! legs also bend at the elbow and knee, and the body at the waist, and the whole model can move,
//! turn and scale as one (a back flip, lying down).

use crate::client::posed_parts::HumanoidPart;

/// The parts [`PlayerBodyPose`] poses, in its order: the parts vanilla's `PlayerModel` animates.
pub const POSABLE_PARTS: [HumanoidPart; 6] = [
    HumanoidPart::Head,
    HumanoidPart::Body,
    HumanoidPart::RightArm,
    HumanoidPart::LeftArm,
    HumanoidPart::RightLeg,
    HumanoidPart::LeftLeg,
];

/// The parts [`PlayerBodyPose`] bends half way down, at the elbow, knee or waist, in its order.
pub const BENDABLE_PARTS: [HumanoidPart; 5] = [
    HumanoidPart::RightArm,
    HumanoidPart::LeftArm,
    HumanoidPart::RightLeg,
    HumanoidPart::LeftLeg,
    HumanoidPart::Body,
];

/// Why [`PlayerBodyPose::set`] took no pose, or [`PlayerBodyPose::set_bend`] no bend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, horizon_mod_api_macros::Wire)]
#[non_exhaustive]
pub enum PoseError {
    /// The part is not one of [`POSABLE_PARTS`]: an outer layer follows the part it covers.
    NotPosable,
    /// A number of the pose is not finite.
    NotFinite,
    /// The part is not one of [`BENDABLE_PARTS`]: only the arms, legs and body bend.
    NotBendable,
}

/// A player model's body this frame, for `set_player_pose`: for each of [`POSABLE_PARTS`] its pose at
/// rest (before any animation), as vanilla posed it this frame (walking, swinging, sneaking, a
/// Horizon emote laid over it), and whether a mod set its own. A part no mod set draws as vanilla
/// posed it. Poses are [`PartPose`](crate::PartPose)s relative to the part's parent, as a baked
/// layer's parts take them: x, y, z in model pixels (y down), rotations in radians.
///
/// Each of [`BENDABLE_PARTS`] also bends half way down, at the elbow, knee or waist, once a mod
/// [`set_bend`](Self::set_bend)s it: a limb's lower half turns about the joint, carrying the held
/// item with a forearm, the body's upper half with the cape and elytra, and a band fills the
/// outer side of the bend.
///
/// The whole model also moves, turns and scales as one once a mod [`set_root`](Self::set_root)s
/// it, with everything drawn with it (outer layers, armour, held items, cape, elytra).
#[derive(Clone, Copy, Debug, PartialEq, horizon_mod_api_macros::Wire)]
#[wire(via = PlayerBodyPoseWire)]
#[non_exhaustive]
pub struct PlayerBodyPose {
    at_rest: [crate::PartPose; 6],
    vanilla: [crate::PartPose; 6],
    posed: [crate::PartPose; 6],
    set: u8,
    bend: [[f32; 2]; 5],
    bent: u8,
    root: Option<crate::PartPose>,
}

fn part_pose(v: [f32; 9]) -> crate::PartPose {
    crate::PartPose {
        x: v[0],
        y: v[1],
        z: v[2],
        x_rot: v[3],
        y_rot: v[4],
        z_rot: v[5],
        x_scale: v[6],
        y_scale: v[7],
        z_scale: v[8],
    }
}

fn finite(p: &crate::PartPose) -> bool {
    [
        p.x, p.y, p.z, p.x_rot, p.y_rot, p.z_rot, p.x_scale, p.y_scale, p.z_scale,
    ]
    .iter()
    .all(|v| v.is_finite())
}

impl PlayerBodyPose {
    /// A body no mod posed yet: each of [`POSABLE_PARTS`] at rest and as vanilla posed it this
    /// frame, as `ModelPart`'s nine numbers (x, y, z, x rot, y rot, z rot, x, y, z scale).
    pub fn new(at_rest: [[f32; 9]; 6], vanilla: [[f32; 9]; 6]) -> Self {
        let vanilla = vanilla.map(part_pose);
        Self {
            at_rest: at_rest.map(part_pose),
            vanilla,
            posed: vanilla,
            set: 0,
            bend: [[0.0; 2]; 5],
            bent: 0,
            root: None,
        }
    }

    /// `part`'s index in [`POSABLE_PARTS`], if it is one.
    pub fn index(part: HumanoidPart) -> Option<usize> {
        POSABLE_PARTS.iter().position(|p| *p == part)
    }

    /// `part` at rest, before any animation: where an emote's offsets start from.
    pub fn at_rest(&self, part: HumanoidPart) -> Option<crate::PartPose> {
        Some(self.at_rest[Self::index(part)?])
    }

    /// `part` as vanilla posed it this frame, whatever a mod set.
    pub fn vanilla(&self, part: HumanoidPart) -> Option<crate::PartPose> {
        Some(self.vanilla[Self::index(part)?])
    }

    /// `part` as it draws: the pose a mod set, else vanilla's.
    pub fn get(&self, part: HumanoidPart) -> Option<crate::PartPose> {
        Some(self.posed[Self::index(part)?])
    }

    /// Draws `part` with `pose` this frame, in place of vanilla's: a later listener may set it
    /// again or [`reset`](Self::reset) it.
    pub fn set(&mut self, part: HumanoidPart, pose: crate::PartPose) -> Result<(), PoseError> {
        let i = Self::index(part).ok_or(PoseError::NotPosable)?;
        if !finite(&pose) {
            return Err(PoseError::NotFinite);
        }
        self.posed[i] = pose;
        self.set |= 1 << i;
        Ok(())
    }

    /// `part`'s index in [`BENDABLE_PARTS`], if it is one.
    pub fn bend_index(part: HumanoidPart) -> Option<usize> {
        BENDABLE_PARTS.iter().position(|p| *p == part)
    }

    /// Bends `part` this frame at the elbow, knee or waist, as bendy-lib does: a limb's lower half
    /// (the body's upper half) turns by `angle` radians about the horizontal axis through the
    /// joint that is the model's x axis turned by `axis` radians about y. At `axis` 0 the part
    /// bends as `x_rot` turns it: a negative angle swings a forearm forwards (a knee bent to kneel
    /// back is a positive one), a positive angle bows the chest forwards; at `axis` π/2 it bends
    /// sideways. The bend is relative to the part's own pose, whoever set it. The head and arms
    /// are parts of their own, not the body's: to carry them with a bowing chest, a mod turns
    /// their poses about the waist too, as PlayerAnimationLibrary does for its `torso` bend.
    pub fn set_bend(&mut self, part: HumanoidPart, axis: f32, angle: f32) -> Result<(), PoseError> {
        let i = Self::bend_index(part).ok_or(PoseError::NotBendable)?;
        if !axis.is_finite() || !angle.is_finite() {
            return Err(PoseError::NotFinite);
        }
        self.bend[i] = [axis, angle];
        self.bent |= 1 << i;
        Ok(())
    }

    /// `part`'s bend this frame as `(axis, angle)`, if a mod bent it.
    pub fn bend(&self, part: HumanoidPart) -> Option<(f32, f32)> {
        let i = Self::bend_index(part)?;
        (self.bent & (1 << i) != 0).then(|| (self.bend[i][0], self.bend[i][1]))
    }

    /// Whether a mod bent `part` this frame.
    pub fn is_bent(&self, part: HumanoidPart) -> bool {
        self.bend(part).is_some()
    }

    /// Moves, turns and scales the whole model this frame, as PlayerAnimationLibrary's `body` bone
    /// does: `pose`'s x, y, z move it in model pixels (y down, so a negative y lifts it), its
    /// rotations (radians, z then y then x as a part's) turn it about the middle of the body, 12
    /// model pixels above the feet, and its scales scale it about that point. Whatever the model
    /// draws turns with it: outer layers, armour, held items, cape and elytra.
    /// [`PartPose::ZERO`](crate::PartPose::ZERO) draws the model where vanilla does. A later
    /// listener may set it again or [`reset_root`](Self::reset_root) it.
    pub fn set_root(&mut self, pose: crate::PartPose) -> Result<(), PoseError> {
        if !finite(&pose) {
            return Err(PoseError::NotFinite);
        }
        self.root = Some(pose);
        Ok(())
    }

    /// The whole model's move, turn and scale this frame, if a mod [`set_root`](Self::set_root) it.
    pub fn root(&self) -> Option<crate::PartPose> {
        self.root
    }

    /// Draws the whole model where vanilla does again.
    pub fn reset_root(&mut self) {
        self.root = None;
    }

    /// Draws `part` as vanilla posed it again, straight.
    pub fn reset(&mut self, part: HumanoidPart) {
        if let Some(i) = Self::index(part) {
            self.posed[i] = self.vanilla[i];
            self.set &= !(1 << i);
        }
        if let Some(i) = Self::bend_index(part) {
            self.bend[i] = [0.0; 2];
            self.bent &= !(1 << i);
        }
    }

    /// Whether a mod set `part`'s pose this frame.
    pub fn is_set(&self, part: HumanoidPart) -> bool {
        Self::index(part).is_some_and(|i| self.set & (1 << i) != 0)
    }

    /// Whether no mod set any part's pose, bent a limb or moved the whole model this frame.
    pub fn is_empty(&self) -> bool {
        self.set == 0 && self.bent == 0 && self.root.is_none()
    }
}

#[derive(horizon_mod_api_macros::Wire)]
struct PlayerBodyPoseWire {
    at_rest: [crate::PartPose; 6],
    vanilla: [crate::PartPose; 6],
    posed: [crate::PartPose; 6],
    set: u8,
    bend: [[f32; 2]; 5],
    bent: u8,
    root: Option<crate::PartPose>,
}
impl From<&PlayerBodyPose> for PlayerBodyPoseWire {
    fn from(v: &PlayerBodyPose) -> Self {
        Self {
            at_rest: v.at_rest,
            vanilla: v.vanilla,
            posed: v.posed,
            set: v.set,
            bend: v.bend,
            bent: v.bent,
            root: v.root,
        }
    }
}
impl TryFrom<PlayerBodyPoseWire> for PlayerBodyPose {
    type Error = crate::wire::WireError;
    fn try_from(v: PlayerBodyPoseWire) -> Result<Self, Self::Error> {
        use crate::wire::WireError;
        if v.set >> 6 != 0 || v.bent >> 5 != 0 {
            return Err(WireError::Invalid("PlayerBodyPose set bits"));
        }
        // A limb no mod bent is straight; a bent one only with finite numbers.
        let mut bend = v.bend;
        for (i, b) in bend.iter_mut().enumerate() {
            if v.bent & (1 << i) == 0 {
                *b = [0.0; 2];
            } else if !b.iter().all(|x| x.is_finite()) {
                return Err(WireError::Invalid("PlayerBodyPose bend not finite"));
            }
        }
        // A part no mod set draws as vanilla posed it; a set one only with finite numbers.
        let mut posed = v.posed;
        for (i, pose) in posed.iter_mut().enumerate() {
            if v.set & (1 << i) == 0 {
                *pose = v.vanilla[i];
            } else if !finite(pose) {
                return Err(WireError::Invalid("PlayerBodyPose pose not finite"));
            }
        }
        if v.root.as_ref().is_some_and(|r| !finite(r)) {
            return Err(WireError::Invalid("PlayerBodyPose root not finite"));
        }
        Ok(Self {
            at_rest: v.at_rest,
            vanilla: v.vanilla,
            posed,
            set: v.set,
            bend,
            bent: v.bent,
            root: v.root,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nine(x: f32) -> [f32; 9] {
        [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    }

    #[test]
    fn a_set_part_draws_as_set_and_the_rest_as_vanilla() {
        let mut body = PlayerBodyPose::new(std::array::from_fn(|i| nine(i as f32)), [nine(9.0); 6]);
        assert!(body.is_empty());
        let arm = HumanoidPart::RightArm;
        assert_eq!(body.at_rest(arm).unwrap().x, 2.0);
        let mut raised = body.vanilla(arm).unwrap();
        raised.x_rot = -3.0;
        assert_eq!(body.set(arm, raised), Ok(()));
        assert!(body.is_set(arm) && !body.is_set(HumanoidPart::Head));
        assert_eq!(body.get(arm).unwrap().x_rot, -3.0);
        assert_eq!(body.vanilla(arm).unwrap().x_rot, 0.0);
        assert_eq!(
            body.set(HumanoidPart::Hat, raised),
            Err(PoseError::NotPosable)
        );
        assert_eq!(body.get(HumanoidPart::RightSleeve), None);
        raised.y = f32::NAN;
        assert_eq!(body.set(arm, raised), Err(PoseError::NotFinite));
        body.reset(arm);
        assert!(body.is_empty());
        assert_eq!(body.get(arm), body.vanilla(arm));
    }

    #[test]
    fn a_limb_bends_until_reset_and_nothing_else_does() {
        let mut body = PlayerBodyPose::new([nine(0.0); 6], [nine(1.0); 6]);
        let knee = HumanoidPart::LeftLeg;
        assert_eq!(body.bend(knee), None);
        assert_eq!(body.set_bend(knee, 0.5, 1.2), Ok(()));
        assert!(!body.is_empty() && body.is_bent(knee) && !body.is_set(knee));
        assert_eq!(body.bend(knee), Some((0.5, 1.2)));
        // The body bends at the waist, on a bit of its own.
        assert_eq!(body.set_bend(HumanoidPart::Body, 0.0, 0.7), Ok(()));
        assert_eq!(body.bend(HumanoidPart::Body), Some((0.0, 0.7)));
        body.reset(HumanoidPart::Body);
        assert!(!body.is_bent(HumanoidPart::Body) && body.is_bent(knee));
        assert_eq!(
            body.set_bend(HumanoidPart::Head, 0.0, 1.0),
            Err(PoseError::NotBendable)
        );
        assert_eq!(
            body.set_bend(HumanoidPart::LeftPants, 0.0, 1.0),
            Err(PoseError::NotBendable)
        );
        assert_eq!(
            body.set_bend(knee, 0.0, f32::NAN),
            Err(PoseError::NotFinite)
        );
        assert_eq!(body.bend(knee), Some((0.5, 1.2)));
        body.reset(knee);
        assert!(body.is_empty() && body.bend(knee).is_none());
    }

    #[test]
    fn the_whole_model_moves_until_reset() {
        let mut body = PlayerBodyPose::new([nine(0.0); 6], [nine(1.0); 6]);
        assert_eq!(body.root(), None);
        // A back flip turns past a full turn; nothing clamps it.
        let flip = part_pose([0.0, -23.0, 0.0, 6.5, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(body.set_root(flip), Ok(()));
        assert!(!body.is_empty() && !body.is_set(HumanoidPart::Body));
        assert_eq!(body.root(), Some(flip));
        assert_eq!(
            body.set_root(part_pose(nine(f32::INFINITY))),
            Err(PoseError::NotFinite)
        );
        assert_eq!(body.root(), Some(flip));
        body.reset(HumanoidPart::Body);
        assert_eq!(body.root(), Some(flip));
        body.reset_root();
        assert!(body.is_empty() && body.root().is_none());
    }

    #[test]
    fn the_wire_keeps_a_set_pose_and_refuses_bad_ones() {
        use crate::wire::{Reader, Writer};
        let mut body = PlayerBodyPose::new([nine(0.0); 6], [nine(1.0); 6]);
        body.set(HumanoidPart::Head, part_pose(nine(5.0))).unwrap();
        body.set_bend(HumanoidPart::RightArm, 0.25, -1.5).unwrap();
        body.set_bend(HumanoidPart::Body, 0.0, 0.5).unwrap();
        body.set_root(part_pose([0.0, -20.0, 0.0, 6.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
        let mut w = Writer::new();
        w.put(&body);
        assert_eq!(
            Reader::new(w.bytes()).take::<PlayerBodyPose>().unwrap(),
            body
        );
        for (set, x, bent, angle, root) in [
            (1 << 6, 0.0, 0, 0.0, None),
            (1, f32::INFINITY, 0, 0.0, None),
            (0, 0.0, 1 << 5, 0.0, None),
            (0, 0.0, 1, f32::NAN, None),
            (0, 0.0, 0, 0.0, Some(part_pose(nine(f32::NAN)))),
        ] {
            let raw = PlayerBodyPoseWire {
                at_rest: [part_pose(nine(0.0)); 6],
                vanilla: [part_pose(nine(0.0)); 6],
                posed: [part_pose(nine(x)); 6],
                set,
                bend: [[0.0, angle]; 5],
                bent,
                root,
            };
            let mut w = Writer::new();
            w.put(&raw);
            assert!(Reader::new(w.bytes()).take::<PlayerBodyPose>().is_err());
        }
    }
}
