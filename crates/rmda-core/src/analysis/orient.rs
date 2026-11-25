//! Orientation algebra shared by segmentation and defect analysis
//! (design §11.3).
//!
//! PTM's quaternion rotates the template frame into the sample frame, and the
//! lattice symmetry acts in the template frame, so `q` and `q·g` are the same
//! orientation: symmetry multiplies **from the right**, everywhere.

use crate::analysis::ptm::{quat::quat_rot, quat_tables, structure};

/// Quaternion product `a·b`, `(w, x, y, z)`.
#[inline]
pub fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    quat_rot(a, b)
}

/// The proper rotations orientations of `st` are compared under: the order-24
/// cubic group for FCC, BCC and SC; for HCP the order-12 hexagonal group. The
/// 60° rotation about c is included because PTM's alternate HCP templates
/// differ by it and a 60° turn plus a c/2 shift maps the HCP
/// crystal onto itself. `None` for a structure without a lattice.
pub fn group(st: i32) -> Option<Vec<[f64; 4]>> {
    match st {
        structure::FCC | structure::BCC | structure::SC => Some(quat_tables::CUBIC.to_vec()),
        // dHCP has the same point group (6/mmm) and template frame as HCP.
        structure::HCP | structure::DHCP => Some(quat_tables::HCP_CONVENTIONAL.to_vec()),
        _ => None,
    }
}

#[inline]
fn dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Disorientation angle in degrees: `2·acos(max_g |qa · (qb·g)|)`.
pub fn disorientation(qa: &[f64; 4], qb: &[f64; 4], group: &[[f64; 4]]) -> f64 {
    crate::analysis::grains::disorientation(qa, qb, group)
}

/// The `±q·g` closest to `reference`.
#[inline]
pub fn align_to(reference: &[f64; 4], q: &[f64; 4], group: &[[f64; 4]]) -> [f64; 4] {
    let mut best = (0.0f64, *q);
    for g in group {
        let c = qmul(*q, *g);
        let d = dot(&c, reference);
        if d.abs() > best.0.abs() {
            best = (d, c);
        }
    }
    if best.0 < 0.0 {
        best.1.map(|v| -v)
    } else {
        best.1
    }
}

/// Running symmetry-aligned sum of orientations.
#[derive(Debug, Clone, Copy)]
pub struct MeanAcc {
    pub reference: [f64; 4],
    pub sum: [f64; 4],
    pub count: f64,
}

impl MeanAcc {
    pub fn new(reference: [f64; 4]) -> Self {
        Self {
            reference,
            sum: [0.0; 4],
            count: 0.0,
        }
    }

    #[inline]
    pub fn add(&mut self, q: &[f64; 4], weight: f64, group: &[[f64; 4]]) {
        if !q[0].is_finite() {
            return;
        }
        let a = align_to(&self.reference, q, group);
        for (s, v) in self.sum.iter_mut().zip(a) {
            *s += weight * v;
        }
        self.count += weight;
    }

    /// Merge another accumulator, aligning its mean to this reference.
    pub fn merge(&mut self, o: &MeanAcc, group: &[[f64; 4]]) {
        if o.count > 0.0 {
            self.add(&o.mean(), o.count, group);
        }
    }

    /// The normalised mean, or the reference when empty.
    pub fn mean(&self) -> [f64; 4] {
        let l = dot(&self.sum, &self.sum).sqrt();
        if l > 0.0 {
            self.sum.map(|v| v / l)
        } else {
            self.reference
        }
    }
}

/// HCP template frame to FCC template frame, for close-packed stacking (design
/// §11.3): the HCP c axis along FCC [111], the in-plane neighbour direction
/// a = [100]_template along FCC [1-10], which puts the HCP template's lower
/// neighbour layer on FCC first-shell sites. An HCP atom on a stacking fault or
/// twin plane of an FCC crystal with orientation `q_F` has `q_H = q_F · m`
/// (up to the HCP group), so `q_F = q_H · m⁻¹`.
pub fn hcp_to_fcc() -> [f64; 4] {
    let (s2, s3, s6) = (2f64.sqrt(), 3f64.sqrt(), 6f64.sqrt());
    // Columns: images of the HCP template's x, y = z × x and z.
    let m = crate::cell::Mat3::from_rows(
        [1.0 / s2, 1.0 / s6, 1.0 / s3],
        [-1.0 / s2, 1.0 / s6, 1.0 / s3],
        [0.0, -2.0 / s6, 1.0 / s3],
    );
    crate::lattice::fit::mat_to_quat(&m)
}

fn conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Whether orientations of `a` and `b` can be compared: the same lattice
/// structure, or FCC and HCP when `close_packed` (stacking faults and twin
/// planes).
pub fn compatible(a: i32, b: i32, close_packed: bool) -> bool {
    a == b && group(a).is_some()
        || close_packed
            && matches!(
                (a, b),
                (structure::FCC, structure::HCP) | (structure::HCP, structure::FCC)
            )
}

/// The template frame `st`'s orientations are in: dHCP's is HCP's (its FCC
/// layers are mapped there by `stacking::relabel_dhcp`).
fn frame(st: i32) -> i32 {
    if st == structure::DHCP {
        structure::HCP
    } else {
        st
    }
}

/// `q` (an orientation of structure `from`) expressed in the frame of `to`,
/// as the symmetry variant nearest `reference` (an orientation of `to`).
/// Same structure: `align_to`. FCC↔HCP through [`hcp_to_fcc`]; the HCP
/// group's 60° turn covers both stacking variants, so an HCP twin-plane atom
/// matches either twin.
pub fn convert(from: i32, q: &[f64; 4], to: i32, reference: &[f64; 4]) -> [f64; 4] {
    let gt = group(to).expect("lattice structure");
    match (frame(from), frame(to)) {
        (structure::HCP, structure::FCC) => {
            let mi = conj(hcp_to_fcc());
            group(structure::HCP)
                .unwrap()
                .iter()
                .map(|&g| align_to(reference, &qmul(qmul(*q, g), mi), &gt))
                .max_by(|a, b| dot(a, reference).abs().total_cmp(&dot(b, reference).abs()))
                .unwrap()
        }
        // Every cubic variant, as the other direction tries every hexagonal
        // one: which {111} becomes c is the variant's choice. Comparisons
        // never come this way (`disorientation_between` works in the FCC
        // frame, where twins stay 60° apart); it only folds an FCC part into
        // a cluster kept in the HCP frame, after the merge was decided.
        (structure::FCC, structure::HCP) => group(structure::FCC)
            .unwrap()
            .iter()
            .map(|&g| align_to(reference, &qmul(qmul(*q, g), hcp_to_fcc()), &gt))
            .max_by(|a, b| dot(a, reference).abs().total_cmp(&dot(b, reference).abs()))
            .unwrap(),
        _ => align_to(reference, q, &gt),
    }
}

/// Disorientation, degrees, between `qa` of structure `a` and `qb` of `b`,
/// compared in `a`'s frame; `NaN` when they are not [`compatible`].
pub fn disorientation_between(a: i32, qa: &[f64; 4], b: i32, qb: &[f64; 4]) -> f64 {
    if !compatible(frame(a), frame(b), true) {
        return f64::NAN;
    }
    // A hexagonal–FCC pair (HCP or dHCP) is compared in the FCC frame,
    // whichever comes first. In the HCP frame the 60° turn about c (in the
    // order-12 group) is the FCC twin operation, so an FCC orientation
    // converted there matches both twins; in the FCC frame the HCP one is
    // tried in all twelve variants and matches the twin it is coherent with.
    if frame(a) == structure::HCP && b == structure::FCC {
        return disorientation_between(b, qb, a, qa);
    }
    let qb = convert(b, qb, a, qa);
    disorientation(qa, &qb, &group(a).unwrap())
}

/// Symmetry-aligned mean of `qs` (two passes: the second aligns to the first
/// pass's mean, so the starting reference does not bias the result).
pub fn mean<'a>(qs: impl Iterator<Item = &'a [f64; 4]> + Clone, group: &[[f64; 4]]) -> [f64; 4] {
    let Some(first) = qs.clone().find(|q| q[0].is_finite()) else {
        return [1.0, 0.0, 0.0, 0.0];
    };
    let mut reference = *first;
    for _ in 0..2 {
        let mut acc = MeanAcc::new(reference);
        for q in qs.clone() {
            acc.add(q, 1.0, group);
        }
        reference = acc.mean();
    }
    reference
}

#[cfg(test)]
mod tests {
    use super::*;
    type Coord = crate::cell::Coord3;

    fn norm(q: [f64; 4]) -> [f64; 4] {
        let l = dot(&q, &q).sqrt();
        q.map(|v| v / l)
    }

    /// The order-12 table is the six D3d generators and their products with
    /// the 60° turn about c, built here from the axis-angle formula.
    #[test]
    fn hexagonal_group_is_d3d_and_its_sixty_degree_coset() {
        let h = std::f64::consts::FRAC_PI_6;
        let r60 = [h.cos(), 0.0, 0.0, h.sin()];
        let g = group(structure::HCP).unwrap();
        assert_eq!(g.len(), 12);
        for (k, &d) in quat_tables::HCP.iter().enumerate() {
            for (got, want) in [(g[k], d), (g[k + 6], qmul(d, r60))] {
                assert!(
                    got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-15),
                    "{k}"
                );
            }
        }
    }

    #[test]
    fn symmetry_equivalents_have_zero_disorientation_and_average_to_one() {
        let q = norm([0.71, 0.31, -0.42, 0.22]);
        for st in [structure::FCC, structure::HCP] {
            let g = group(st).unwrap();
            let eq: Vec<[f64; 4]> = g.iter().map(|&s| qmul(q, s)).collect();
            for e in &eq {
                assert!(disorientation(&q, e, &g) < 1e-6);
            }
            let m = mean(eq.iter(), &g);
            assert!(disorientation(&q, &m, &g) < 1e-6, "st={st}");
        }
    }

    #[test]
    fn an_hcp_plane_in_fcc_maps_onto_the_fcc_orientation_and_its_twin() {
        use crate::analysis::ptm::tables::{FCC_TEMPLATES, HCP_TEMPLATES};
        use crate::lattice::fit::quat_to_mat;
        let qf = norm([0.71, 0.31, -0.42, 0.22]);
        let qh = qmul(qf, hcp_to_fcc());
        let (rf, rh) = (quat_to_mat(qf), quat_to_mat(qh));
        // The HCP template's basal ring and lower layer sit on FCC sites.
        let fcc: Vec<Coord> = FCC_TEMPLATES[0][1..13]
            .iter()
            .map(|v| rf.mul_v(*v))
            .collect();
        for v in &HCP_TEMPLATES[0][1..13] {
            if v[2] > 0.1 {
                continue;
            }
            let p = rh.mul_v(*v);
            let d = fcc
                .iter()
                .map(|f| (0..3).map(|k| (f[k] - p[k]).powi(2)).sum::<f64>())
                .fold(f64::INFINITY, f64::min);
            assert!(d < 1e-12, "{v:?}");
        }
        // Any HCP-symmetric representative maps back to qf (0°) ...
        let gh = group(structure::HCP).unwrap();
        for g in &gh {
            let d = disorientation_between(structure::FCC, &qf, structure::HCP, &qmul(qh, *g));
            assert!(d < 1e-6, "{d}");
        }
        // ... and to the Σ3 twin (60° about [111]) as well.
        let h = std::f64::consts::FRAC_PI_6 * 1.0;
        let s = h.sin() / 3f64.sqrt();
        let twin = qmul(qf, [h.cos(), s, s, s]);
        assert!((disorientation(&qf, &twin, &group(structure::FCC).unwrap()) - 60.0).abs() < 1e-6);
        assert!(disorientation_between(structure::FCC, &twin, structure::HCP, &qh) < 1e-6);
        assert!(disorientation_between(structure::BCC, &qf, structure::HCP, &qh).is_nan());
    }

    #[test]
    fn the_hexagonal_group_is_closed_under_the_sixty_degree_turn() {
        let g = group(structure::HCP).unwrap();
        assert_eq!(g.len(), 12);
        let h = std::f64::consts::FRAC_PI_6;
        let r60 = [h.cos(), 0.0, 0.0, h.sin()];
        assert!(disorientation(&[1.0, 0.0, 0.0, 0.0], &r60, &g) < 1e-6);
    }
}
