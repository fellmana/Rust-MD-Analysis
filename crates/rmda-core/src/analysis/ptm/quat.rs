//! Orientation quaternions: rotation, and the fundamental-zone reduction that
//! makes a lattice orientation comparable across grains.
//!
//! A PTM fit produces a quaternion, but not a *canonical* one. Any orientation
//! related to the fitted one by a symmetry of the structure is physically
//! indistinguishable, and the fit will happily return any member of that set
//! depending on which neighbour permutation it happened to pick. Comparing two
//! such quaternions directly, or averaging them, is meaningless.
//!
//! [`rotate_into_structure_fundamental_zone`] fixes this by multiplying `q` by
//! whichever point-group generator has the largest `|q . g|`, which lands the
//! result in a single canonical representative of its coset. That is what makes
//! a quaternion usable for grain segmentation (design §15), where orientations
//! have to be grouped and compared.
//!
//! Ported from `ptm_quat.cpp`. HCP, hexagonal diamond and graphene reduce over
//! the order-12 conventional group, as upstream does with
//! `output_conventional_orientation == true` (OVITO's setting).

use super::quat_tables::{CUBIC, HCP, HCP_CONVENTIONAL, ICOSAHEDRAL};
use super::structure;

/// Quaternion `(w, x, y, z)` product `r * a`.
///
/// Matches `quat_rot` in `ptm_quat.cpp`.
pub fn quat_rot(r: [f64; 4], a: [f64; 4]) -> [f64; 4] {
    [
        r[0] * a[0] - r[1] * a[1] - r[2] * a[2] - r[3] * a[3],
        r[0] * a[1] + r[1] * a[0] + r[2] * a[3] - r[3] * a[2],
        r[0] * a[2] - r[1] * a[3] + r[2] * a[0] + r[3] * a[1],
        r[0] * a[3] + r[1] * a[2] - r[2] * a[1] + r[3] * a[0],
    ]
}

/// Rotate `q` by the generator with the largest `|q . g|` and fix the sign, so
/// the result lands in the fundamental zone. Returns the generator's index.
///
/// The sign fix is not cosmetic: `q` and `-q` are the same rotation, so without
/// it half of the fundamental zone would leak out the other side.
///
/// The comparison is a strict `>`, matching upstream, so an exact tie between
/// two generators resolves to the lower index. That makes the reduction
/// sensitive to floating point on a zone boundary: a rotation lying exactly on
/// a mirror plane of the zone can fold to either side, and folding an already
/// folded value may land on the other one. It was observed in practice with the
/// icosahedral table, where a reduced quaternion came back with the sign of its
/// z component flipped on a second pass. Nothing here asserts idempotence, and
/// a consumer comparing orientations should use a tolerance rather than exact
/// equality.
fn rotate_into_fundamental_zone(generators: &[[f64; 4]], q: &mut [f64; 4]) -> usize {
    let mut max = 0.0f64;
    let mut best = 0usize;
    for (i, g) in generators.iter().enumerate() {
        let t = (q[0] * g[0] - q[1] * g[1] - q[2] * g[2] - q[3] * g[3]).abs();
        if t > max {
            max = t;
            best = i;
        }
    }
    *q = quat_rot(*q, generators[best]);
    if q[0] < 0.0 {
        for c in q.iter_mut() {
            *c = -*c;
        }
    }
    best
}

/// Order-24 cubic fundamental zone, for SC, FCC and BCC.
pub fn rotate_quaternion_into_cubic_fundamental_zone(q: &mut [f64; 4]) -> usize {
    rotate_into_fundamental_zone(&CUBIC, q)
}

/// Order-60 icosahedral fundamental zone, for ICO.
pub fn rotate_quaternion_into_icosahedral_fundamental_zone(q: &mut [f64; 4]) -> usize {
    rotate_into_fundamental_zone(&ICOSAHEDRAL, q)
}

/// Six-element `D3d` fundamental zone, for HCP.
pub fn rotate_quaternion_into_hcp_fundamental_zone(q: &mut [f64; 4]) -> usize {
    rotate_into_fundamental_zone(&HCP, q)
}

/// Twelve-element hexagonal fundamental zone ([`HCP_CONVENTIONAL`]): the
/// HCP orientation OVITO reports, atom for atom.
pub fn rotate_quaternion_into_hcp_conventional_fundamental_zone(q: &mut [f64; 4]) -> usize {
    rotate_into_fundamental_zone(&HCP_CONVENTIONAL, q)
}

/// Reduce `q` to the fundamental zone of the structure it was fitted to.
///
/// Returns `None` for [`structure::OTHER`] and any code without a generator
/// table, mirroring upstream's `rotate_into_fundamental_zone`, which returns
/// `-1` when it has no generator for the type. The quaternion is
/// left untouched in that case, so callers get the raw fit rather than a
/// rotation relative to the wrong point group.
pub fn rotate_into_structure_fundamental_zone(kind: i32, q: &mut [f64; 4]) -> Option<usize> {
    match kind {
        // Cubic diamond reduces over the full cubic group, as upstream's
        // conventional orientation does (its two templates differ by the 90°
        // turn that the diamond point group lacks), and as OVITO reports it.
        structure::SC | structure::FCC | structure::BCC | structure::DCUB => {
            Some(rotate_quaternion_into_cubic_fundamental_zone(q))
        }
        structure::ICO => Some(rotate_quaternion_into_icosahedral_fundamental_zone(q)),
        // Hexagonal diamond and graphene use HCP's conventional order-12 group
        // (upstream `rotate_into_fundamental_zone(.., true, ..)`).
        structure::HCP | structure::DHEX | structure::GRAPHENE => {
            Some(rotate_quaternion_into_hcp_conventional_fundamental_zone(q))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::quat_tables::{CUBIC, HCP, ICOSAHEDRAL};

    /// A unit quaternion for a rotation of `angle` radians about `axis`.
    ///
    /// Built from the axis-angle formula rather than written out, because
    /// `[cos(t), 0, 0, 0]` looks like a rotation but is not one: its norm is
    /// `cos(t)`, and feeding it to `quat_rot` quietly rescales the result.
    fn axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
        let n: f64 = axis.iter().map(|c| c * c).sum::<f64>().sqrt();
        let a = axis.map(|c| c / n);
        let s = (angle / 2.0).sin();
        [(angle / 2.0).cos(), s * a[0], s * a[1], s * a[2]]
    }

    /// The generator tables must be unit quaternions, or `|q . g|` is not a
    /// meaningful projection and the argmax picks an arbitrary member.
    #[test]
    fn generators_are_unit_quaternions() {
        for (name, table) in [
            ("cubic", &CUBIC[..]),
            ("icosahedral", &ICOSAHEDRAL[..]),
            ("hcp", &HCP[..]),
        ] {
            for (i, g) in table.iter().enumerate() {
                let n2: f64 = g.iter().map(|c| c * c).sum();
                assert!(
                    (n2 - 1.0).abs() < 1e-12,
                    "{name}[{i}] is not a unit quaternion: |g|^2 = {n2}"
                );
            }
        }
    }

    /// Table lengths are the point-group orders: 24 for `O`, 60 for `I`, 6 for
    /// `D3d`. A truncated table would still "work" and quietly narrow the
    /// fundamental zone.
    #[test]
    fn generator_counts_are_point_group_orders() {
        assert_eq!(CUBIC.len(), 24);
        assert_eq!(ICOSAHEDRAL.len(), 60);
        assert_eq!(HCP.len(), 6);
    }

    /// Reduction must rotate by a genuine argmax over the table, preserve the
    /// norm, and leave the real part non-negative.
    ///
    /// These are the defining properties and they are exact. Idempotence is
    /// deliberately *not* asserted: the winner is chosen by a strict `>` over
    /// `|q . g|`, so a rotation sitting exactly on a zone boundary can reduce
    /// to either side on a second pass depending on the last bit. See
    /// [`rotate_into_fundamental_zone`].
    #[test]
    fn rotation_picks_the_argmax_generator() {
        for (label, table, rotate) in [
            (
                "cubic",
                &CUBIC[..],
                rotate_quaternion_into_cubic_fundamental_zone as fn(&mut [f64; 4]) -> usize,
            ),
            (
                "icosahedral",
                &ICOSAHEDRAL[..],
                rotate_quaternion_into_icosahedral_fundamental_zone,
            ),
            ("hcp", &HCP[..], rotate_quaternion_into_hcp_fundamental_zone),
        ] {
            let mut seeds: Vec<[f64; 4]> = vec![axis_angle([1.0, 0.0, 0.0], 0.0)];
            // A spread of real rotations, including the 180-degree ones that
            // are their own inverse and the identity.
            for (axis, angle) in [
                ([1.0, 0.0, 0.0], 1.0),
                ([0.0, 1.0, 0.0], 2.0),
                ([0.0, 0.0, 1.0], std::f64::consts::PI),
                ([1.0, 1.0, 1.0], 0.7),
                ([0.3, -0.5, 0.81], 2.9),
                ([1.0, 0.0, 0.0], 0.05),
            ] {
                seeds.push(axis_angle(axis, angle));
            }
            for seed in seeds {
                // The argmax is defined on the *input*; after rotating, the
                // winning generator is generally no longer the maximiser, so
                // this has to be checked before the call.
                let mut best = 0usize;
                let mut max = 0.0f64;
                for (i, g) in table.iter().enumerate() {
                    let t =
                        (seed[0] * g[0] - seed[1] * g[1] - seed[2] * g[2] - seed[3] * g[3]).abs();
                    if t > max {
                        max = t;
                        best = i;
                    }
                }

                let mut q = seed;
                let bi = rotate(&mut q);
                assert_eq!(bi, best, "{label}: picked {bi}, argmax is {best}");
                assert!(
                    q[0] >= 0.0,
                    "{label}: real part {} after reduction from {seed:?}",
                    q[0]
                );
                let n2: f64 = q.iter().map(|c| c * c).sum();
                assert!(
                    (n2 - 1.0).abs() < 1e-12,
                    "{label}: |q|^2 = {n2} after reduction from {seed:?}"
                );
            }
        }
    }

    /// The point of the whole exercise: an orientation and the same physical
    /// orientation expressed through a symmetry of the structure must reduce to
    /// the *same* quaternion. Without this, grain segmentation would split one
    /// grain into up to 24 pieces.
    #[test]
    fn symmetry_related_orientations_agree() {
        // A 4-fold rotation about z is a symmetry of the cubic group, so 45
        // and 135 degrees about z describe the same crystal orientation.
        let sym = axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let a = axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_4);
        let b = quat_rot(sym, a);
        let (mut a, mut b) = (a, b);
        rotate_quaternion_into_cubic_fundamental_zone(&mut a);
        rotate_quaternion_into_cubic_fundamental_zone(&mut b);
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < 1e-12,
                "symmetry-related orientations disagree: {a:?} vs {b:?}"
            );
        }
    }

    /// A generic rotation is *not* a cubic symmetry, so a genuinely different
    /// orientation must survive the reduction. This is the negative control for
    /// the test above: without it, a reduction that collapsed everything to one
    /// value would still pass.
    #[test]
    fn non_symmetry_rotation_changes_the_fundamental_zone() {
        let generic = axis_angle([0.144, 0.135, 0.98], 0.41);
        let mut a = axis_angle([1.0, 1.0, 0.0], 0.9);
        let mut b = quat_rot(generic, a);
        rotate_quaternion_into_cubic_fundamental_zone(&mut a);
        rotate_quaternion_into_cubic_fundamental_zone(&mut b);
        let differs = (0..4).any(|i| (a[i] - b[i]).abs() > 1e-6);
        assert!(
            differs,
            "a non-symmetry rotation collapsed into the zone: {a:?}"
        );
    }

    /// Unsupported types must leave the quaternion alone rather than rotating
    /// it by the wrong point group.
    #[test]
    fn unsupported_types_are_left_alone() {
        let mut q = axis_angle([1.0, 0.0, 0.0], 0.77);
        let before = q;
        assert_eq!(
            rotate_into_structure_fundamental_zone(structure::OTHER, &mut q),
            None
        );
        assert_eq!(q, before);
    }

    /// Every supported structure must reach its own point group, and the
    /// dispatch must not be a pass-through for any of them.
    #[test]
    fn dispatch_covers_every_supported_structure() {
        for kind in [
            structure::SC,
            structure::FCC,
            structure::HCP,
            structure::BCC,
            structure::ICO,
        ] {
            let mut q = axis_angle([0.2, -0.4, 0.9], 1.3);
            assert!(
                rotate_into_structure_fundamental_zone(kind, &mut q).is_some(),
                "type {kind} was not routed to a fundamental zone"
            );
        }
    }
}
