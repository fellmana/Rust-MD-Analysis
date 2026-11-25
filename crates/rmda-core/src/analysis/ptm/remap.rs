//! Template remapping, the elastic deformation gradient and alloy ordering
//! types (upstream `ptm_remap_template`, `calculate_deformation_gradient` and
//! `find_alloy_type`; design §14.2).
//!
//! These run only when one of the opt-in outputs ([`super::output`]) is
//! requested; the default PTM path never reaches this module.
//!
//! OVITO calls `ptm_index` with `output_conventional_orientation = true`, so
//! the remap here is the conventional one: HCP, graphene, cubic and hexagonal
//! diamond pick one of their alternative templates (the *template index*)
//! from the fundamental-zone generator, and the point mapping is permuted to
//! that template before the deformation gradient is fitted.

use super::quat_tables::{CUBIC, ICOSAHEDRAL};
use super::refdata::RefData;
use super::remap_tables as t;
use super::{structure, MAX_POINTS};

/// OVITO's `PolyhedralTemplateMatchingModifier.OrderingType` codes, equal to
/// upstream's `PTM_ALLOY_*`.
pub mod ordering {
    /// No ordering recognised (or not tested, or no structure).
    pub const NONE: u8 = 0;
    /// Every atom of the environment has the central atom's type.
    pub const PURE: u8 = 1;
    /// L1₀ (CuAu), FCC.
    pub const L10: u8 = 2;
    /// L1₂ (Cu₃Au), central atom of the majority type (`PTM_ALLOY_L12_CU`).
    pub const L12_A: u8 = 3;
    /// L1₂, central atom of the minority type (`PTM_ALLOY_L12_AU`).
    pub const L12_B: u8 = 4;
    /// B2 (CsCl), BCC.
    pub const B2: u8 = 5;
    /// Zincblende / wurtzite (SiC), cubic or hexagonal diamond.
    pub const ZINCBLENDE_WURTZITE: u8 = 6;
    /// Hexagonal boron nitride, graphene.
    pub const BORON_NITRIDE: u8 = 7;
}

/// Upstream `typedata`: FCC's three binary orderings and their canonical
/// codes.
const FCC_ALLOYS: [(u8, u32); 3] = [
    (ordering::L10, 0x0000_0db6),
    (ordering::L12_A, 0x0000_0492),
    (ordering::L12_B, 0x0000_1ffe),
];

/// The fundamental point mappings of a structure (`ref->mapping`, first
/// `num_mappings` rows); empty for graphene (upstream `num_mappings = -1`).
fn fundamental_mappings(kind: i32) -> &'static [[i8; MAX_POINTS]] {
    match kind {
        structure::SC => &t::MAPPING_SC,
        structure::FCC => &t::MAPPING_FCC,
        structure::HCP => &t::MAPPING_HCP,
        structure::ICO => &t::MAPPING_ICO,
        structure::BCC => &t::MAPPING_BCC,
        structure::DCUB => &t::MAPPING_DCUB,
        structure::DHEX => &t::MAPPING_DHEX,
        _ => &[],
    }
}

fn shell_structure(num_points: usize, mapping: &[i8], numbers: &[u32], num_inner: usize) -> bool {
    let c = numbers[0];
    let binary = |i: usize| numbers[mapping[i] as usize] != c;
    let b0 = binary(0);
    (1..=num_inner).all(|i| binary(i) != b0) && (num_inner + 1..num_points).all(|i| binary(i) == b0)
}

/// Upstream `find_alloy_type`: the chemical ordering of a matched environment.
///
/// `mapping[i]` is the data point matched to template point `i` (the fit's
/// own mapping, before the template remap; the result does not depend on
/// which), and `numbers[j]` the type of data point `j` (`numbers[0]` the
/// central atom).
pub fn find_alloy_type(kind: i32, mapping: &[i8; MAX_POINTS], numbers: &[u32]) -> u8 {
    let Some(rd) = RefData::for_type(kind) else {
        return ordering::NONE;
    };
    let np = rd.num_nbrs + 1;
    let c = numbers[0];
    if numbers[1..np].iter().all(|&n| n == c) {
        return ordering::PURE;
    }
    // Binary: at most one type besides the central atom's.
    let mut other = None;
    for &n in &numbers[1..np] {
        if n != c {
            match other {
                None => other = Some(n),
                Some(o) if o != n => return ordering::NONE,
                _ => {}
            }
        }
    }

    // Canonical code: the minimum over the point group of the bit pattern
    // "differs from the central atom", permuted by each fundamental mapping.
    let mut best = u32::MAX;
    for m in fundamental_mappings(kind) {
        let mut code = 0u32;
        for i in 0..np {
            let bit = (numbers[mapping[i] as usize] != c) as u32;
            code |= bit << m[i];
        }
        best = best.min(code);
    }
    if kind == structure::FCC {
        for &(ord, code) in &FCC_ALLOYS {
            if best == code {
                return ord;
            }
        }
    }
    let inner = match kind {
        structure::BCC => Some((8, ordering::B2)),
        structure::DCUB | structure::DHEX => Some((4, ordering::ZINCBLENDE_WURTZITE)),
        structure::GRAPHENE => Some((3, ordering::BORON_NITRIDE)),
        _ => None,
    };
    if let Some((n_inner, ord)) = inner {
        if shell_structure(np, mapping, numbers, n_inner) {
            return ord;
        }
    }
    ordering::NONE
}

/// Argmax-generator reduction (upstream `rotate_quaternion_into_fundamental_zone`),
/// returning the generator index in the given table's order.
fn generator_index(generators: &[[f64; 4]], q: &[f64; 4]) -> usize {
    let mut max = 0.0f64;
    let mut best = 0usize;
    for (i, g) in generators.iter().enumerate() {
        let v = (q[0] * g[0] - q[1] * g[1] - q[2] * g[2] - q[3] * g[3]).abs();
        if v > max {
            max = v;
            best = i;
        }
    }
    best
}

/// Upstream `ptm_remap_template` with `output_conventional_orientation`, as
/// OVITO calls it: the template index and the point mapping permuted to that
/// template, from the fit's raw quaternion `q` (not the reduced orientation).
///
/// The generator is chosen over upstream's own table order (for HCP-like
/// structures [`t::GENERATOR_HCP_CONVENTIONAL`], which lists the same twelve
/// rotations as [`super::quat_tables::HCP_CONVENTIONAL`] in a different
/// order), so exact ties break as upstream breaks them.
pub fn remap_template(
    kind: i32,
    q: &[f64; 4],
    mapping: &[i8; MAX_POINTS],
) -> (u8, [i8; MAX_POINTS]) {
    let mut out = [-1i8; MAX_POINTS];
    let Some(rd) = RefData::for_type(kind) else {
        return (0, *mapping);
    };
    let np = rd.num_nbrs + 1;
    let (perm, template): (&[i8; MAX_POINTS], u8) = match kind {
        structure::SC | structure::FCC | structure::BCC => {
            (&fundamental_mappings(kind)[generator_index(&CUBIC, q)], 0)
        }
        structure::ICO => (&t::MAPPING_ICO[generator_index(&ICOSAHEDRAL, q)], 0),
        structure::DCUB => {
            let bi = generator_index(&CUBIC, q);
            (
                &t::MAPPING_DCUB_CONVENTIONAL[bi],
                t::TEMPLATE_INDICES_DCUB[bi],
            )
        }
        structure::HCP | structure::DHEX | structure::GRAPHENE => {
            let bi = generator_index(&t::GENERATOR_HCP_CONVENTIONAL, q);
            match kind {
                structure::HCP => (
                    &t::MAPPING_HCP_CONVENTIONAL[bi],
                    t::TEMPLATE_INDICES_HCP[bi],
                ),
                structure::DHEX => (
                    &t::MAPPING_DHEX_CONVENTIONAL[bi],
                    t::TEMPLATE_INDICES_DHEX[bi],
                ),
                _ => (
                    &t::MAPPING_GRAPHENE_CONVENTIONAL[bi],
                    t::TEMPLATE_INDICES_GRAPHENE[bi],
                ),
            }
        }
        _ => return (0, *mapping),
    };
    for i in 0..np {
        out[perm[i] as usize] = mapping[i];
    }
    (template, out)
}

/// The Penrose inverse of `kind`'s template `variant`.
fn penrose(kind: i32, variant: u8) -> &'static [[f64; 3]; MAX_POINTS] {
    let v = variant as usize;
    match kind {
        structure::SC => &t::PENROSE_SC[0],
        structure::FCC => &t::PENROSE_FCC[0],
        structure::HCP => &t::PENROSE_HCP[v],
        structure::ICO => &t::PENROSE_ICO[0],
        structure::BCC => &t::PENROSE_BCC[0],
        structure::DCUB => &t::PENROSE_DCUB[v],
        structure::DHEX => &t::PENROSE_DHEX[v],
        _ => &t::PENROSE_GRAPHENE[v],
    }
}

/// Upstream `calculate_deformation_gradient` as `output_data` calls it: the
/// least-squares `F` (row-major, `F[3 i + j]`) taking template `template` onto
/// the barycentred, `scale`d environment `points` under the remapped
/// `mapping`. Graphene's `F[8]` is set to 1, upstream's pseudo-2D hack.
pub fn deformation_gradient(
    kind: i32,
    template: u8,
    points: &[[f64; 3]],
    mapping: &[i8; MAX_POINTS],
    scale: f64,
) -> [f64; 9] {
    let mut f = [0.0f64; 9];
    let Some(rd) = RefData::for_type(kind) else {
        return f;
    };
    let np = rd.num_nbrs + 1;
    // subtract_barycentre, then scale, as upstream.
    let mut sum = [0.0f64; 3];
    for p in &points[..np] {
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    for s in sum.iter_mut() {
        *s /= np as f64;
    }
    let mut scaled = [[0.0f64; 3]; MAX_POINTS];
    for (o, p) in scaled.iter_mut().zip(&points[..np]) {
        for k in 0..3 {
            o[k] = (p[k] - sum[k]) * scale;
        }
    }
    let pen = penrose(kind, template);
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..np {
                acc += pen[k][j] * scaled[mapping[k] as usize][i];
            }
            f[i * 3 + j] = acc;
        }
    }
    if kind == structure::GRAPHENE {
        f[8] = 1.0;
    }
    f
}

/// The residual of [`deformation_gradient`] (upstream's `F_res`, from the
/// `calculate_deformation_gradient` of the PTM library bundled with LAMMPS;
/// OVITO 3.15 does not output it): per Cartesian axis `i`, the sum over the
/// template points `k` of `((F t_k)_i - s_{mapping[k], i})²`, with `t` the
/// template variant `template` and `s` the barycentred, scaled points. Units
/// of the scaled (template) frame, squared.
pub fn deformation_residual(
    kind: i32,
    template: u8,
    points: &[[f64; 3]],
    mapping: &[i8; MAX_POINTS],
    scale: f64,
    f: &[f64; 9],
) -> [f64; 3] {
    let mut res = [0.0f64; 3];
    let Some(rd) = RefData::for_type(kind) else {
        return res;
    };
    let np = rd.num_nbrs + 1;
    let mut sum = [0.0f64; 3];
    for p in &points[..np] {
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    for s in sum.iter_mut() {
        *s /= np as f64;
    }
    let tpl = rd.points(template as usize);
    for k in 0..np {
        let p = points[mapping[k] as usize];
        for i in 0..3 {
            let mut acc = 0.0f64;
            for j in 0..3 {
                acc += f[i * 3 + j] * tpl[k][j];
            }
            let delta = acc - (p[i] - sum[i]) * scale;
            res[i] += delta * delta;
        }
    }
    res
}

/// Upstream `PTM_MAX_INPUT_POINTS`: the centre and 18 neighbours.
const MAX_INPUT_POINTS: usize = 19;
/// Upstream `MAX_MULTISHELL_NEIGHBOURS`: the two-shell structures draw from
/// the 13 nearest neighbours of each list (`log2(13·12·11·10) < 15`).
const MAX_MULTISHELL_NEIGHBOURS: usize = 13;

/// Upstream `permutation_to_index`: the first `n` entries of a partial
/// permutation of `0..base` as a mixed-radix (Lehmer) code. Entries outside
/// `0..base` or repeated (which upstream asserts never occur) give 0.
fn permutation_to_index(base: usize, n: usize, perm: &[i8]) -> u64 {
    let mut p = [0usize; MAX_INPUT_POINTS];
    let mut q = [0usize; MAX_INPUT_POINTS];
    for i in 0..base {
        p[i] = i;
        q[i] = i;
    }
    let mut code = [0u64; MAX_INPUT_POINTS];
    for i in 0..n {
        let e = perm[i];
        if e < 0 || e as usize >= base {
            return 0;
        }
        let qe = q[e as usize];
        // A repeated entry was already moved below `i`: not a permutation.
        let Some(d) = qe.checked_sub(i) else {
            return 0;
        };
        code[i] = d as u64;
        if qe > i {
            let j = qe;
            let (pi, pj) = (p[i], p[j]);
            q.swap(pi, pj);
            p.swap(i, j);
        }
    }
    let mut encoded = 0u64;
    for it in 0..n {
        encoded = encoded * (1 + base - n + it) as u64 + code[n - it - 1];
    }
    encoded
}

/// Upstream `encode_correspondences`, OVITO's `Correspondences` property
/// (int64; template index in bits 62–63, so DHEX templates 2–3 come out
/// negative as in OVITO). `corr[i]` is the correspondence of template point
/// `i` (`env->correspondences[mapping[i]]` under the remapped mapping): 0 for
/// the centre, else one plus the neighbour's distance rank in its list.
///
/// First-shell structures: the correspondences, completed by the unused
/// ranks in increasing order, as one permutation of 18. Two-shell: the inner
/// atoms (15 bits) and each inner atom's outer atoms (11 bits apiece), each
/// as a partial permutation of 13.
pub fn encode_correspondences(kind: i32, corr: &[u8], template: u8) -> i64 {
    let Some(rd) = RefData::for_type(kind) else {
        return 0;
    };
    let num_nbrs = rd.num_nbrs;
    let mut c = [0i8; MAX_INPUT_POINTS];
    let mut t = [0i8; MAX_INPUT_POINTS];
    let encoded = match kind {
        structure::DCUB | structure::DHEX | structure::GRAPHENE => {
            let (num_inner, num_outer) = if kind == structure::GRAPHENE {
                (3, 2)
            } else {
                (4, 3)
            };
            for k in 0..num_nbrs {
                t[k] = corr[k + 1] as i8 - 1;
            }
            let mut e = permutation_to_index(MAX_MULTISHELL_NEIGHBOURS, num_inner, &t);
            for i in 0..num_inner {
                let part = permutation_to_index(
                    MAX_MULTISHELL_NEIGHBOURS,
                    num_outer,
                    &t[num_inner + i * num_outer..],
                );
                e |= part << (15 + 11 * i);
            }
            e
        }
        _ => {
            // complete_correspondences
            let np = num_nbrs + 1;
            let mut hit = [false; MAX_INPUT_POINTS];
            for i in 0..np {
                c[i] = corr[i] as i8;
                if (corr[i] as usize) < MAX_INPUT_POINTS {
                    hit[corr[i] as usize] = true;
                }
            }
            let mut next = np;
            for (v, &h) in hit.iter().enumerate() {
                if !h && next < MAX_INPUT_POINTS {
                    c[next] = v as i8;
                    next += 1;
                }
            }
            for k in 0..MAX_INPUT_POINTS - 1 {
                t[k] = c[k + 1] - 1;
            }
            permutation_to_index(MAX_INPUT_POINTS - 1, MAX_INPUT_POINTS - 1, &t)
        }
    };
    (encoded | (template as u64) << 62) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ptm::quat_tables::HCP_CONVENTIONAL;

    /// Each Penrose table is the pseudo-inverse of its template variant:
    /// `sum_k P[k][j] T[k][i] = delta_ij`. This pins the pairing of
    /// `PENROSE_*[v]` with `RefData::points(v)` (the alt templates).
    #[test]
    fn penrose_inverts_its_template() {
        for kind in 1..=8 {
            let rd = RefData::for_type(kind).unwrap();
            let np = rd.num_nbrs + 1;
            for v in 0..rd.templates.len() as u8 {
                let tpl = rd.points(v as usize);
                let pen = penrose(kind, v);
                for i in 0..3 {
                    for j in 0..3 {
                        let s: f64 = (0..np).map(|k| pen[k][j] * tpl[k][i]).sum();
                        let want = if i == j { 1.0 } else { 0.0 };
                        // Graphene is planar: its z row is null.
                        if kind == structure::GRAPHENE && (i == 2 || j == 2) {
                            continue;
                        }
                        assert!((s - want).abs() < 1e-12, "kind {kind} v{v} ({i},{j}) = {s}");
                    }
                }
            }
        }
    }

    /// Upstream's twelve hexagonal generators are ours, reordered.
    #[test]
    fn hcp_generators_are_a_permutation() {
        for g in &t::GENERATOR_HCP_CONVENTIONAL {
            assert!(HCP_CONVENTIONAL.iter().any(|h| h == g), "{g:?}");
        }
    }

    /// An ideal template mapped onto itself has F = I; the template remap of a
    /// rotated fit yields the alternative template and still F = I.
    #[test]
    fn identity_fit_has_unit_f() {
        for kind in 1..=8 {
            let rd = RefData::for_type(kind).unwrap();
            let np = rd.num_nbrs + 1;
            let pts = rd.points(0);
            let mut mapping = [-1i8; MAX_POINTS];
            for (i, m) in mapping.iter_mut().enumerate().take(np) {
                *m = i as i8;
            }
            let (ti, m2) = remap_template(kind, &[1.0, 0.0, 0.0, 0.0], &mapping);
            assert_eq!(ti, 0);
            let f = deformation_gradient(kind, ti, &pts[..], &m2, 1.0);
            for i in 0..3 {
                for j in 0..3 {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((f[3 * i + j] - want).abs() < 1e-12, "kind {kind}: {f:?}");
                }
            }
        }
    }

    /// F is the least-squares fit, so its residual is orthogonal to the fit:
    /// `sum F_res = sum |s|² - sum |F t|²` over the barycentred, scaled points
    /// `s`. Checked on every template variant with pseudo-random points, and
    /// an affine image of the template has zero residual.
    #[test]
    fn residual_is_the_least_squares_remainder() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a = [1.03, 0.02, -0.01, 0.0, 0.98, 0.03, 0.01, -0.02, 1.01];
        for kind in 1..=8 {
            let rd = RefData::for_type(kind).unwrap();
            let np = rd.num_nbrs + 1;
            let mut mapping = [-1i8; MAX_POINTS];
            for (i, m) in mapping.iter_mut().enumerate().take(np) {
                *m = ((i + 3) % np) as i8;
            }
            for v in 0..rd.templates.len() as u8 {
                let tpl = rd.points(v as usize);
                let scale = 0.7;
                // Affine image: data point mapping[k] = A t_k / scale.
                let mut pts = [[0.0f64; 3]; MAX_POINTS];
                for k in 0..np {
                    for i in 0..3 {
                        pts[mapping[k] as usize][i] =
                            (0..3).map(|j| a[3 * i + j] * tpl[k][j]).sum::<f64>() / scale;
                    }
                }
                let f = deformation_gradient(kind, v, &pts[..], &mapping, scale);
                let r = deformation_residual(kind, v, &pts[..], &mapping, scale, &f);
                assert!(r.iter().all(|&x| x < 1e-24), "kind {kind} v{v}: {r:?}");
                // Noisy: the orthogonality identity.
                for p in pts.iter_mut().take(np) {
                    for x in p.iter_mut() {
                        *x += 0.1 * rnd();
                    }
                }
                let f = deformation_gradient(kind, v, &pts[..], &mapping, scale);
                let r = deformation_residual(kind, v, &pts[..], &mapping, scale, &f);
                let mut bar = [0.0f64; 3];
                for p in &pts[..np] {
                    for i in 0..3 {
                        bar[i] += p[i] / np as f64;
                    }
                }
                let (mut ss, mut ff) = (0.0, 0.0);
                for k in 0..np {
                    for i in 0..3 {
                        let s = (pts[k][i] - bar[i]) * scale;
                        ss += s * s;
                        let ft: f64 = (0..3).map(|j| f[3 * i + j] * tpl[k][j]).sum();
                        ff += ft * ft;
                    }
                }
                let total: f64 = r.iter().sum();
                assert!(total > 1e-6, "kind {kind} v{v}");
                assert!(
                    (total - (ss - ff)).abs() < 1e-12,
                    "kind {kind} v{v}: {total} vs {}",
                    ss - ff
                );
            }
        }
    }

    /// Hand-checked encodings: the identity of 18 is 0; one transposition of
    /// the first two ranks is the smallest nonzero code; the template index
    /// lands in bits 62–63.
    #[test]
    fn correspondence_encoding() {
        let id: Vec<u8> = (0..13).collect();
        assert_eq!(encode_correspondences(structure::FCC, &id, 0), 0);
        let mut sw = id.clone();
        sw.swap(1, 2);
        assert_eq!(encode_correspondences(structure::FCC, &sw, 0), 1);
        assert_eq!(encode_correspondences(structure::HCP, &id, 1), 1i64 << 62);
        assert_eq!(
            encode_correspondences(
                structure::DHEX,
                &(0..17u8)
                    .map(|i| if i <= 13 { i } else { i - 13 })
                    .collect::<Vec<_>>(),
                3
            ) as u64
                >> 62,
            3
        );
        // A repeated entry is not a permutation: 0, not an overflow.
        assert_eq!(permutation_to_index(13, 3, &[12, 12, 12]), 0);
    }

    #[test]
    fn alloy_types_of_ideal_orderings() {
        // B2: inner 8 of the other type, outer 6 of the central's.
        let mut m = [-1i8; MAX_POINTS];
        for (i, v) in m.iter_mut().enumerate().take(15) {
            *v = i as i8;
        }
        let mut n = [1u32; 15];
        n[1..9].fill(2);
        assert_eq!(find_alloy_type(structure::BCC, &m, &n), ordering::B2);
        assert_eq!(
            find_alloy_type(structure::BCC, &m, &[1; 15]),
            ordering::PURE
        );
        n[9] = 3;
        assert_eq!(find_alloy_type(structure::BCC, &m, &n), ordering::NONE);
        // L1₂ with a minority centre: all 12 neighbours of the other type.
        let mut n = [2u32; 13];
        n[0] = 1;
        assert_eq!(find_alloy_type(structure::FCC, &m, &n), ordering::L12_B);
    }
}
