//! A mob's baked model: the parts of `entity-models.json` as polygons, and the normal stack that packs their
//! normals the way `ModelPart` does.
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct Polygon {
    pub normal: [f32; 3],
    pub vertices: [[f32; 5]; 4],
    /// The model's render type culls back faces (`entityCutoutCull`: the bat,
    /// the baby turtle).
    pub cull_back: bool,
}
impl Polygon {
    pub(super) fn vertex_normal(&self, pose: ModelNormals) -> [i8; 4] {
        let n = pose.matrix * glam::Vec3::from_array(self.normal);
        let n = if pose.trusted {
            n
        } else {
            n.normalize_or_zero()
        };
        let packed = n.to_array().map(|v| (v.clamp(-1.0, 1.0) * 127.0) as i8);
        let w = if self.cull_back {
            crate::mesher::SHADED_CULL_W
        } else {
            -127
        };
        [packed[0], packed[1], packed[2], w]
    }
}

/// ModelPart uses a separate normal stack: uniform scaling keeps normal lengths,
/// while nonuniform scaling requires normalization after the complete part transform.
#[derive(Clone, Copy)]
pub(super) struct ModelNormals {
    pub(super) matrix: glam::Mat3,
    pub(super) trusted: bool,
}
impl ModelNormals {
    // shortcut: infer the root normal pose from its geometry matrix; pass an explicit
    // root normal stack when root-transform rounding is ported as well.
    pub(super) fn new(base: glam::Mat4) -> Self {
        let m = glam::Mat3::from_mat4(base);
        let lengths = glam::Vec3::new(m.x_axis.length(), m.y_axis.length(), m.z_axis.length());
        let uniform =
            lengths.min_element() > 0.0 && lengths.max_element() - lengths.min_element() < 1.0e-6;
        Self {
            matrix: if uniform {
                m / lengths.x
            } else {
                m.inverse().transpose()
            },
            trusted: uniform,
        }
    }
    pub(super) fn rotation(x: f32, y: f32, z: f32) -> glam::Mat3 {
        // JOML derives cosine from the rounded sine, then retains quaternion length.
        // Assuming an exactly unit quaternion changes a packed normal by one byte.
        let sin_cos = |angle: f32| {
            let angle = angle * 0.5;
            let sin = f64::from(angle).sin() as f32;
            let cos = (1.0 - sin * sin).sqrt();
            let a = angle + std::f32::consts::FRAC_PI_2;
            let tau = std::f32::consts::TAU;
            let mut quadrant = a - (a / tau) as i32 as f32 * tau;
            if quadrant < 0.0 {
                quadrant += tau;
            }
            (
                sin,
                if quadrant >= std::f32::consts::PI {
                    -cos
                } else {
                    cos
                },
            )
        };
        let (sx, cx) = sin_cos(x);
        let (sy, cy) = sin_cos(y);
        let (sz, cz) = sin_cos(z);
        let (cycz, sysz, sycz, cysz) = (cy * cz, sy * sz, sy * cz, cy * sz);
        let (w, x, y, z) = (
            cx * cycz + sx * sysz,
            sx * cycz - cx * sysz,
            cx * sycz + sx * cysz,
            cx * cysz - sx * sycz,
        );
        let (ww, xx, yy, zz) = (w * w, x * x, y * y, z * z);
        let (xy, xz, xw, yz, yw, zw) = (x * y, x * z, x * w, y * z, y * w, z * w);
        glam::Mat3::from_cols_array(&[
            ww + xx - zz - yy,
            (xy + xy) + (zw + zw),
            (xz + xz) - (yw + yw),
            (xy + xy) - (zw + zw),
            yy - zz + ww - xx,
            (yz + yz) + (xw + xw),
            (yw + yw) + (xz + xz),
            (yz + yz) - (xw + xw),
            zz - yy - xx + ww,
        ])
    }
    pub(super) fn part(mut self, pose: [f32; 9]) -> Self {
        self.matrix *= Self::rotation(pose[3], pose[4], pose[5]);
        let scale = glam::Vec3::new(pose[6], pose[7], pose[8]);
        if scale.x.abs() == scale.y.abs() && scale.x.abs() == scale.z.abs() {
            if scale.min_element() < 0.0 {
                self.matrix *= glam::Mat3::from_diagonal(scale.signum());
            }
        } else {
            self.matrix *= glam::Mat3::from_diagonal(scale.recip());
            self.trusted = false;
        }
        self
    }
    pub(super) fn part_rotated(mut self, pose: [f32; 9], rotation: glam::Mat3) -> Self {
        self.matrix *= rotation;
        let scale = glam::Vec3::new(pose[6], pose[7], pose[8]);
        if scale.x.abs() == scale.y.abs() && scale.x.abs() == scale.z.abs() {
            if scale.min_element() < 0.0 {
                self.matrix *= glam::Mat3::from_diagonal(scale.signum());
            }
        } else {
            self.matrix *= glam::Mat3::from_diagonal(scale.recip());
            self.trusted = false;
        }
        self
    }
}

#[derive(Clone, Debug)]
pub struct Part {
    pub name: String,
    /// x, y, z, xRot, yRot, zRot, xScale, yScale, zScale
    pub pose: [f32; 9],
    pub polygons: Vec<Polygon>,
    pub children: Vec<Part>,
}
impl Part {
    /// Marks every polygon as drawn by a back-face-culling render type.
    pub(super) fn set_cull_back(&mut self) {
        for poly in &mut self.polygons {
            poly.cull_back = true;
        }
        for child in &mut self.children {
            child.set_cull_back();
        }
    }
    pub(super) fn parse(name: &str, v: &Value) -> Self {
        let pose: Vec<f32> = v["pose"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let mut p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        for (i, x) in pose.iter().enumerate().take(9) {
            p[i] = *x;
        }
        let polygons: Vec<Polygon> = v["polygons"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .filter_map(|(index, poly)| {
                let verts: Vec<[f32; 5]> = poly
                    .as_array()?
                    .iter()
                    .map(|vx| {
                        let a: Vec<f32> = vx
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                            .collect();
                        [a[0], a[1], a[2], a[3], a[4]]
                    })
                    .collect();
                (verts.len() == 4).then(|| Polygon {
                    normal: v["normals"][index].as_array().map_or_else(
                        || {
                            // shortcut: collapsed legacy faces have no recoverable normal; re-prepare resources.
                            let p = verts
                                .iter()
                                .map(|v| glam::Vec3::new(v[0], v[1], v[2]))
                                .collect::<Vec<_>>();
                            (p[1] - p[0])
                                .cross(p[2] - p[0])
                                .normalize_or_zero()
                                .to_array()
                        },
                        |n| std::array::from_fn(|i| n[i].as_f64().unwrap_or(0.0) as f32),
                    ),
                    vertices: [verts[0], verts[1], verts[2], verts[3]],
                    cull_back: false,
                })
            })
            .collect();
        let mut children: Vec<Part> = v["children"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, c)| Part::parse(k, c))
            .collect();
        children.sort_by(|a, b| a.name.cmp(&b.name));
        // A model whose root holds a part also named "root" (allay, vex,
        // frog...): that inner part is what `root.getChild("root")` and the
        // keyframe bones mean, so the outer one gives up the name.
        let name = if name == "root" && children.iter().any(|c| c.name == "root") {
            ""
        } else {
            name
        };
        Self {
            name: name.to_owned(),
            pose: p,
            polygons,
            children,
        }
    }
}

/// A player limb split at its joint so it can bend: the part that keeps the upper half, its outer
/// layer, and the names of the parts that hold the lower halves and fill the bend.
pub struct Limb {
    pub upper: &'static str,
    pub overlay: &'static str,
    /// The lower half, a child of `upper`, posed about the joint.
    pub lower: &'static str,
    /// The overlay's lower half, a child of `overlay`.
    pub lower_overlay: &'static str,
    /// The band about the joint that fills the outer side of a bend, drawn only while it bends.
    pub filler: &'static str,
    pub filler_overlay: &'static str,
}

/// The limbs of the player model that bend, in `BENDABLE_PARTS` order (right arm, left arm, right
/// leg, left leg).
pub const LIMBS: [Limb; 4] = [
    Limb {
        upper: "right_arm",
        overlay: "right_sleeve",
        lower: "right_forearm",
        lower_overlay: "right_forearm_sleeve",
        filler: "right_elbow",
        filler_overlay: "right_elbow_sleeve",
    },
    Limb {
        upper: "left_arm",
        overlay: "left_sleeve",
        lower: "left_forearm",
        lower_overlay: "left_forearm_sleeve",
        filler: "left_elbow",
        filler_overlay: "left_elbow_sleeve",
    },
    Limb {
        upper: "right_leg",
        overlay: "right_pants",
        lower: "right_shin",
        lower_overlay: "right_shin_pants",
        filler: "right_knee",
        filler_overlay: "right_knee_pants",
    },
    Limb {
        upper: "left_leg",
        overlay: "left_pants",
        lower: "left_shin",
        lower_overlay: "left_shin_pants",
        filler: "left_knee",
        filler_overlay: "left_knee_pants",
    },
];

/// The pose that leaves a part where its parent is.
const IDENTITY: [f32; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// A part drawn not at all (a zero scale leaves its subtree out).
const HIDDEN: [f32; 9] = [0.0; 9];

/// Whether the model layer `key` (with or without its namespace) is the player model's body or
/// armour, whose limbs [`Part::split_limbs`] splits at the joint.
pub fn bends_limbs(key: &str) -> bool {
    let key = key.strip_prefix("minecraft:").unwrap_or(key);
    let Some((model, layer)) = key.split_once('#') else {
        return false;
    };
    matches!(model, "player" | "player_slim")
        && matches!(
            layer,
            "main" | "helmet" | "chestplate" | "leggings" | "boots"
        )
}

impl Part {
    /// Splits each limb of a player model (its arms and legs, their sleeves and pants) at the
    /// joint, half way down, so the lower half can turn on its own: the part keeps its top face
    /// and the upper half of its sides, and a new child (`Limb::lower`) the rest, with the
    /// vertices where they were and the identity pose, so unbent it draws as before. A second
    /// child (`Limb::filler`) holds the sides of a band about the joint, at a zero scale until a
    /// bend opens a gap for it to fill.
    pub fn split_limbs(&mut self) {
        for limb in &LIMBS {
            if let Some(upper) = self.find_mut(limb.upper) {
                upper.split_at_joint(limb.lower, limb.filler, true);
                if let Some(overlay) = upper.find_mut(limb.overlay) {
                    overlay.split_at_joint(limb.lower_overlay, limb.filler_overlay, false);
                }
            }
        }
    }

    fn find_mut(&mut self, name: &str) -> Option<&mut Part> {
        if self.name == name {
            return Some(self);
        }
        self.children.iter_mut().find_map(|c| c.find_mut(name))
    }

    /// Moves the lower half of this one-cube part into a child `lower` and adds the band `filler`
    /// (first among the children when `first`, so the halves draw before an outer layer).
    fn split_at_joint(&mut self, lower: &str, filler: &str, first: bool) {
        let sides = |p: &&Polygon| p.normal[1] == 0.0;
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in self.polygons.iter().flat_map(|p| p.vertices) {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        // Only a single box: its four sides and two ends (armour without this limb has none).
        if self.polygons.len() != 6 || self.polygons.iter().filter(sides).count() != 4 {
            return;
        }
        let joint = (lo[1] + hi[1]) * 0.5;
        let band = (hi[0] - lo[0]).max(hi[2] - lo[2]) * 0.25;
        let mut upper = Vec::with_capacity(5);
        let mut lower_half = Vec::with_capacity(5);
        let mut fill = Vec::with_capacity(4);
        for poly in self.polygons.drain(..) {
            if poly.normal[1] < 0.0 {
                upper.push(poly);
            } else if poly.normal[1] > 0.0 {
                lower_half.push(poly);
            } else {
                upper.push(poly.cut(lo[1], joint));
                fill.push(poly.cut(joint - band, joint + band));
                lower_half.push(poly.cut(joint, hi[1]));
            }
        }
        self.polygons = upper;
        let lower = Part {
            name: lower.to_owned(),
            pose: IDENTITY,
            polygons: lower_half,
            children: Vec::new(),
        };
        let filler = Part {
            name: filler.to_owned(),
            pose: HIDDEN,
            polygons: fill,
            children: Vec::new(),
        };
        if first {
            self.children.splice(0..0, [lower, filler]);
        } else {
            self.children.extend([lower, filler]);
        }
    }
}

impl Polygon {
    /// The part of this side face (its edges run along y) between `top` and `bottom`, its texture
    /// cut alike.
    fn cut(&self, top: f32, bottom: f32) -> Self {
        let mut out = self.clone();
        for (i, v) in self.vertices.iter().enumerate() {
            // The corner across the face's other edge along y: same x and z.
            let Some(other) = self
                .vertices
                .iter()
                .find(|o| o[0] == v[0] && o[2] == v[2] && o[1] != v[1])
            else {
                continue;
            };
            let y = if v[1] < other[1] {
                v[1].max(top)
            } else {
                v[1].min(bottom)
            };
            if y == v[1] {
                continue;
            }
            let t = (y - v[1]) / (other[1] - v[1]);
            out.vertices[i] = std::array::from_fn(|k| {
                if k == 1 {
                    y
                } else {
                    v[k] + (other[k] - v[k]) * t
                }
            });
        }
        out
    }
}

#[cfg(test)]
mod limb_tests {
    use super::*;

    /// A part with a 4x12x4 box from (-3, -2, -2), faces as the baked models have them.
    fn arm(name: &str, children: Vec<Part>) -> Part {
        let b = |x: f32, y: f32, z: f32, u: f32, v: f32| [x, y, z, u, v];
        let quad = |normal: [f32; 3], vertices| Polygon {
            normal,
            vertices,
            cull_back: false,
        };
        Part {
            name: name.to_owned(),
            pose: [-5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            polygons: vec![
                quad(
                    [0.0, -1.0, 0.0],
                    [
                        b(1., -2., 2., 0.75, 0.25),
                        b(-3., -2., 2., 0.6875, 0.25),
                        b(-3., -2., -2., 0.6875, 0.3125),
                        b(1., -2., -2., 0.75, 0.3125),
                    ],
                ),
                quad(
                    [0.0, 1.0, 0.0],
                    [
                        b(1., 10., -2., 0.8125, 0.3125),
                        b(-3., 10., -2., 0.75, 0.3125),
                        b(-3., 10., 2., 0.75, 0.25),
                        b(1., 10., 2., 0.8125, 0.25),
                    ],
                ),
                quad(
                    [-1.0, 0.0, 0.0],
                    [
                        b(-3., -2., -2., 0.6875, 0.3125),
                        b(-3., -2., 2., 0.625, 0.3125),
                        b(-3., 10., 2., 0.625, 0.5),
                        b(-3., 10., -2., 0.6875, 0.5),
                    ],
                ),
                quad(
                    [0.0, 0.0, -1.0],
                    [
                        b(1., -2., -2., 0.75, 0.3125),
                        b(-3., -2., -2., 0.6875, 0.3125),
                        b(-3., 10., -2., 0.6875, 0.5),
                        b(1., 10., -2., 0.75, 0.5),
                    ],
                ),
                quad(
                    [1.0, 0.0, 0.0],
                    [
                        b(1., -2., 2., 0.8125, 0.3125),
                        b(1., -2., -2., 0.75, 0.3125),
                        b(1., 10., -2., 0.75, 0.5),
                        b(1., 10., 2., 0.8125, 0.5),
                    ],
                ),
                quad(
                    [0.0, 0.0, 1.0],
                    [
                        b(-3., -2., 2., 0.875, 0.3125),
                        b(1., -2., 2., 0.8125, 0.3125),
                        b(1., 10., 2., 0.8125, 0.5),
                        b(-3., 10., 2., 0.875, 0.5),
                    ],
                ),
            ],
            children,
        }
    }

    #[test]
    fn a_split_limb_keeps_its_faces_and_hides_the_band_until_it_bends() {
        assert!(bends_limbs("minecraft:player_slim#chestplate") && bends_limbs("player#main"));
        assert!(!bends_limbs("minecraft:zombie#main") && !bends_limbs("player#cape"));
        let whole = arm("right_arm", vec![arm("right_sleeve", Vec::new())]);
        let mut root = Part {
            name: "root".into(),
            pose: IDENTITY,
            polygons: Vec::new(),
            children: vec![whole.clone()],
        };
        root.split_limbs();
        let split = &root.children[0];
        let names: Vec<_> = split.children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["right_forearm", "right_elbow", "right_sleeve"]);
        let (lower, band) = (&split.children[0], &split.children[1]);
        assert_eq!((lower.pose, band.pose), (IDENTITY, HIDDEN));
        assert_eq!(
            (
                split.polygons.len(),
                lower.polygons.len(),
                band.polygons.len()
            ),
            (5, 5, 4)
        );
        // Each side face is cut at the joint, y 4, texture and all, into halves that meet there.
        let side = |p: &Part| {
            p.polygons
                .iter()
                .find(|p| p.normal == [-1.0, 0.0, 0.0])
                .cloned()
                .unwrap()
        };
        let (up, down) = (side(split), side(lower));
        assert_eq!(up.vertices[2], [-3.0, 4.0, 2.0, 0.625, 0.40625]);
        assert_eq!(down.vertices[1], [-3.0, 4.0, 2.0, 0.625, 0.40625]);
        assert_eq!(down.vertices[2], whole.polygons[2].vertices[2]);
        assert_eq!(side(band).vertices[0][1], 3.0);
        assert_eq!(side(band).vertices[2][1], 5.0);
        // The ends stay with their half; the sleeve splits alike, its halves after its own faces.
        assert_eq!(split.polygons[0].vertices, whole.polygons[0].vertices);
        assert_eq!(lower.polygons[0].vertices, whole.polygons[1].vertices);
        let sleeve = &split.children[2];
        let names: Vec<_> = sleeve.children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["right_forearm_sleeve", "right_elbow_sleeve"]);
        // A limb without a box (a helmet layer's arm) is left as it is.
        let mut empty = Part {
            name: "root".into(),
            pose: IDENTITY,
            polygons: Vec::new(),
            children: vec![Part {
                polygons: Vec::new(),
                ..whole.clone()
            }],
        };
        empty.split_limbs();
        assert!(
            empty.children[0]
                .children
                .iter()
                .all(|c| c.name == "right_sleeve")
        );
    }
}
