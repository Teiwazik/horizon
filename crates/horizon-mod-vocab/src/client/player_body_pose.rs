//! The body of a player model as a mod poses it this frame (`set_player_pose`): the six parts an
//! emote or a body tracker turns and moves, each as vanilla posed it and, once a mod sets it, as
//! the mod wants it drawn. The model's outer layers follow the part they cover (the hat its head,
//! a sleeve its arm), and so do its armour, held items, cape and the layers mods added.

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

/// Why [`PlayerBodyPose::set`] took no pose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, horizon_mod_api_macros::Wire)]
#[non_exhaustive]
pub enum PoseError {
    /// The part is not one of [`POSABLE_PARTS`]: an outer layer follows the part it covers.
    NotPosable,
    /// A number of the pose is not finite.
    NotFinite,
}

/// A player model's body this frame, for `set_player_pose`: for each of [`POSABLE_PARTS`] its pose at
/// rest (before any animation), as vanilla posed it this frame (walking, swinging, sneaking, a
/// Horizon emote laid over it), and whether a mod set its own. A part no mod set draws as vanilla
/// posed it. Poses are [`PartPose`](crate::PartPose)s relative to the part's parent, as a baked
/// layer's parts take them: x, y, z in model pixels (y down), rotations in radians.
#[derive(Clone, Copy, Debug, PartialEq, horizon_mod_api_macros::Wire)]
#[wire(via = PlayerBodyPoseWire)]
#[non_exhaustive]
pub struct PlayerBodyPose {
    at_rest: [crate::PartPose; 6],
    vanilla: [crate::PartPose; 6],
    posed: [crate::PartPose; 6],
    set: u8,
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

    /// Draws `part` as vanilla posed it again.
    pub fn reset(&mut self, part: HumanoidPart) {
        if let Some(i) = Self::index(part) {
            self.posed[i] = self.vanilla[i];
            self.set &= !(1 << i);
        }
    }

    /// Whether a mod set `part`'s pose this frame.
    pub fn is_set(&self, part: HumanoidPart) -> bool {
        Self::index(part).is_some_and(|i| self.set & (1 << i) != 0)
    }

    /// Whether no mod set any part's pose this frame.
    pub fn is_empty(&self) -> bool {
        self.set == 0
    }
}

#[derive(horizon_mod_api_macros::Wire)]
struct PlayerBodyPoseWire {
    at_rest: [crate::PartPose; 6],
    vanilla: [crate::PartPose; 6],
    posed: [crate::PartPose; 6],
    set: u8,
}
impl From<&PlayerBodyPose> for PlayerBodyPoseWire {
    fn from(v: &PlayerBodyPose) -> Self {
        Self {
            at_rest: v.at_rest,
            vanilla: v.vanilla,
            posed: v.posed,
            set: v.set,
        }
    }
}
impl TryFrom<PlayerBodyPoseWire> for PlayerBodyPose {
    type Error = crate::wire::WireError;
    fn try_from(v: PlayerBodyPoseWire) -> Result<Self, Self::Error> {
        use crate::wire::WireError;
        if v.set >> 6 != 0 {
            return Err(WireError::Invalid("PlayerBodyPose set bits"));
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
        Ok(Self {
            at_rest: v.at_rest,
            vanilla: v.vanilla,
            posed,
            set: v.set,
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
    fn the_wire_keeps_a_set_pose_and_refuses_bad_ones() {
        use crate::wire::{Reader, Writer};
        let mut body = PlayerBodyPose::new([nine(0.0); 6], [nine(1.0); 6]);
        body.set(HumanoidPart::Head, part_pose(nine(5.0))).unwrap();
        let mut w = Writer::new();
        w.put(&body);
        assert_eq!(
            Reader::new(w.bytes()).take::<PlayerBodyPose>().unwrap(),
            body
        );
        for (set, x) in [(1 << 6, 0.0), (1, f32::INFINITY)] {
            let raw = PlayerBodyPoseWire {
                at_rest: [part_pose(nine(0.0)); 6],
                vanilla: [part_pose(nine(0.0)); 6],
                posed: [part_pose(nine(x)); 6],
                set,
            };
            let mut w = Writer::new();
            w.put(&raw);
            assert!(Reader::new(w.bytes()).take::<PlayerBodyPose>().is_err());
        }
    }
}
