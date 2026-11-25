//! Structure matching (upstream `ptm_structure_matcher.cpp`).
//!
//! Matching runs in two stages. A cheap topological filter reduces the
//! candidate set to structures whose hull could possibly match: the same number
//! of facets, and no node above the reference's degree bound. The survivors
//! contribute their reference graphs, and the candidate's own canonical hash
//! selects which of those graphs are worth trying. Finally each surviving
//! automorphism is scored by RMSD and the best one wins.
//!
//! The hash is a filter, not a discriminator: distinct reference graphs do
//! collide (FCC has 7 distinct hashes across 8 graphs), so several automorphisms
//! are routinely scored for one candidate.

use crate::analysis::ptm::graph::{canonical_form_coloured, graph_degree, CanonicalForm};
use crate::analysis::ptm::hull::{get_convex_hull, ConvexHull, HullError};
use crate::analysis::ptm::normalize::subtract_barycentre_n;
use crate::analysis::ptm::polar::calc_rmsd;
use crate::analysis::ptm::refdata::{GraphInfo, RefData, Structure};
use crate::analysis::ptm::structure;
use crate::analysis::ptm::tables;
use crate::analysis::ptm::{check, MAX_FACETS, MAX_POINTS};

/// Best match found for one atom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchResult {
    /// Root-mean-square deviation of the fit; `f64::INFINITY` if nothing
    /// matched.
    pub rmsd: f64,
    /// `rmsd` threshold below which a match is accepted.
    pub rmsd_cutoff: f64,
    /// Reciprocal data-to-template scale.
    pub scale: f64,
    /// Rotation as a quaternion `(w, x, y, z)`.
    pub q: [f64; 4],
    /// `mapping[i]` is the index of the data point matched to template point
    /// `i`, or `-1` when nothing matched.
    pub mapping: [i8; MAX_POINTS],
    /// Matched structure, or [`structure::OTHER`].
    pub kind: i32,
    /// Best-fitting structure ignoring `rmsd_cutoff`, or
    /// [`structure::OTHER`] if no graph matched at all.
    ///
    /// This is what `ptm_index` returns: the library applies no cutoff of its
    /// own, so it always answers "this is the closest structure I can find".
    /// Only the host then decides whether to keep it. Keeping both means a
    /// caller can report the accepted label *and* the geometry behind it, which
    /// is what OVITO does - an atom rejected by the cutoff still gets a real
    /// RMSD, interatomic distance and orientation.
    pub best_kind: i32,
    /// Nearest-neighbour distance implied by `scale`, in the caller's units.
    ///
    /// [`interatomic_distance`] of [`best_kind`], or `0.0` when nothing
    /// matched. Computed by [`Ptm::classify`](crate::analysis::ptm::Ptm::classify).
    pub interatomic_distance: f64,
    /// Lattice constant implied by `interatomic_distance`.
    ///
    /// [`lattice_constant`] of [`best_kind`], or `0.0` when nothing matched.
    pub lattice_constant: f64,
    /// Orientation as a unit quaternion `(w, x, y, z)`, reduced into the
    /// fundamental zone of [`best_kind`].
    ///
    /// This is the raw fit rotated by the point-group generator with the
    /// largest `|q . g|`, i.e.
    /// [`super::quat::rotate_into_structure_fundamental_zone`]. Left as the
    /// identity when nothing matched or the type has no generator table.
    ///
    /// Two caveats, both worth knowing before this is used for grain
    /// segmentation (design §15):
    ///
    /// * It is *template-relative*. The quaternion maps the reference template
    ///   onto the data, and the templates are not stored in the same
    ///   orientation as an arbitrary input cell, so an axis-aligned lattice
    ///   does not report the identity. Only relative comparisons between atoms
    ///   are meaningful.
    /// * The reduction is upstream's single argmax fold. It does move
    ///   orientations closer to a canonical representative, but it
    ///   is not a proven canonical form on the symmetry coset: a perfect FCC
    ///   lattice rotated 90 degrees about z was observed to reduce to a
    ///   representative differing from the unrotated one by a 180-degree turn
    ///   about x. Consumers that need unique representatives must compare
    ///   orientations modulo the point group explicitly.
    pub orientation: [f64; 4],
}

/// `interatomic_distance` factor per structure, from upstream
/// `calculate_interatomic_distance` (`ptm_index.cpp`).
///
/// The comment there notes these equal `norm(template[1])`, i.e. the length of
/// the template's first neighbour vector, which is what makes the result a
/// nearest-neighbour distance rather than an arbitrary scale.
///
/// Indexed by [`structure`] code (entries 6-8 are DCUB, DHEX, GRAPHENE). The
/// values are literals rather than expressions because `f64::sqrt` is not
/// `const` on this toolchain and these are read once per atom; the test below
/// re-derives each entry from its formula so a transcription slip cannot hide.
pub fn interatomic_distance_factor(kind: i32) -> f64 {
    const C: [f64; 9] = [
        0.0,
        1.0,                // FCC
        1.0,                // HCP
        0.9378221735089296, // BCC   7 - 3.5 * sqrt(3)
        1.0,                // ICO
        1.0,                // SC
        0.6780833887941489, // DCUB  4*sqrt(3)/(6*sqrt(2)+sqrt(3))
        0.6780833887941489, // DHEX  same
        0.6720277132193877, // GRAPH -3/11 + 6*sqrt(3)/11
    ];
    C[kind as usize]
}

/// `lattice_constant` factor per structure, from upstream
/// `calculate_lattice_constant` (`ptm_index.cpp`), indexed by [`structure`]
/// code. Literals for the same reason as [`interatomic_distance_factor`].
pub fn lattice_constant_factor(kind: i32) -> f64 {
    // `SQRT_2` is the correctly-rounded sqrt(2), so `2.0 / SQRT_2` is
    // bit-identical to the `2./sqrt(2)` written in the C++ while staying
    // const-evaluable and clippy-clean.
    use std::f64::consts::SQRT_2;
    const C: [f64; 9] = [
        0.0,
        2.0 / SQRT_2,       // FCC   2/sqrt(2)
        2.0 / SQRT_2,       // HCP   2/sqrt(2)
        1.1547005383792517, // BCC   2/sqrt(3)
        2.0 / SQRT_2,       // ICO   2/sqrt(2)
        1.0,                // SC
        2.3094010767585034, // DCUB  4/sqrt(3)
        2.3094010767585034, // DHEX  4/sqrt(3)
        1.7320508075688772, // GRAPH sqrt(3)
    ];
    C[kind as usize]
}

/// Nearest-neighbour distance of a match, in the caller's length units.
///
/// `scale` is the factor taking the *normalised data* onto the *template*, so
/// the distance is `factor / scale`, not `factor * scale`. Getting this
/// backwards still yields a perfect RMSD on a self-fit, so it is worth stating
/// out loud.
pub fn interatomic_distance(kind: i32, scale: f64) -> f64 {
    interatomic_distance_factor(kind) / scale
}

/// Lattice constant of a match: `factor * interatomic_distance`.
pub fn lattice_constant(kind: i32, interatomic_distance: f64) -> f64 {
    lattice_constant_factor(kind) * interatomic_distance
}

impl Default for MatchResult {
    /// The failure case: nothing matched, and no match can beat it.
    fn default() -> Self {
        Self {
            rmsd: f64::INFINITY,
            rmsd_cutoff: 4.0,
            scale: 0.0,
            q: [1.0, 0.0, 0.0, 0.0],
            mapping: [-1; MAX_POINTS],
            kind: structure::OTHER,
            best_kind: structure::OTHER,
            interatomic_distance: 0.0,
            lattice_constant: 0.0,
            orientation: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl MatchResult {
    pub fn new(rmsd_cutoff: f64) -> Self {
        Self {
            rmsd_cutoff,
            ..Self::default()
        }
    }

    /// Whether a match was found and accepted.
    pub fn is_match(&self) -> bool {
        self.kind != structure::OTHER && self.rmsd < self.rmsd_cutoff
    }
}

/// Grow the hull of `ch_points` and discard the candidate if the reference
/// could not possibly match.
///
/// Returns `None` if the candidate is rejected, leaving `ch` populated so the
/// caller can reuse it for the next structure in the check set.
fn topological_filter<const N: usize>(
    ref_data: &RefData,
    ch_points: &[[f64; 3]; N],
    points: &[[f64; 3]; N],
    ch: &mut ConvexHull,
) -> Option<(CanonicalForm, [[f64; 3]; N])> {
    let num_points = ref_data.num_nbrs + 1;
    // A reference that wants more neighbours than the caller found cannot
    // match. Upstream reads past the end of its buffer here; refusing is both
    // safe and the same answer, since the hull would be of the wrong size.
    if num_points > N {
        return None;
    }

    let mut facets = [[0i8; 3]; MAX_FACETS];
    let num_facets = match get_convex_hull(&ch_points[..num_points], ch, &mut facets) {
        Ok(()) => ch.num_facets,
        Err(_) => return None,
    };

    // Wrong facet count: a different polyhedron, so a different structure.
    if num_facets != ref_data.num_facets {
        return None;
    }

    let (degree, max_degree) = graph_degree(num_facets, &facets, ref_data.num_nbrs);
    if max_degree > ref_data.max_degree {
        return None;
    }

    // SC is the one structure where the degree sequence is fully determined by
    // the type, so upstream checks it exactly rather than just bounding it.
    if ref_data.structure == Structure::Sc && degree[..ref_data.num_nbrs].iter().any(|d| *d != 4) {
        return None;
    }

    let normalized = subtract_barycentre_n(points, num_points);
    let colours = [0i8; MAX_POINTS];
    let form =
        canonical_form_coloured(num_facets, &facets, ref_data.num_nbrs, &degree, &colours).ok()?;

    Some((form, normalized))
}

/// Score every reference graph of `ref_data` whose hash matches, over all of
/// that graph's automorphisms, keeping the best RMSD.
fn check_graphs<const N: usize>(
    ref_data: &RefData,
    hash: u64,
    canonical_labelling: &[i8; MAX_POINTS],
    normalized: &[[f64; 3]; N],
    res: &mut MatchResult,
) {
    let num_points = ref_data.num_nbrs + 1;
    let template = ref_data.points(0);
    // Upstream indexes the 17-wide template array with a runtime `num_points`;
    // copying into an N-wide one keeps the const-generic fit monomorphic.
    let mut ideal_points = [[0.0f64; 3]; N];
    ideal_points[..num_points].copy_from_slice(&template[..num_points]);

    let mut inverse_labelling = [-1i8; MAX_POINTS];
    for i in 0..num_points {
        inverse_labelling[canonical_labelling[i] as usize] = i as i8;
    }

    // The two squared norms, shared by every automorphism: only the rotation
    // changes between candidates.
    let mut g1 = 0.0f64;
    let mut g2 = 0.0f64;
    for i in 0..num_points {
        for k in 0..3 {
            let a = ideal_points[i][k];
            let b = normalized[i][k];
            g1 += a * a;
            g2 += b * b;
        }
    }

    let mut mapping = [0usize; N];
    for graph in ref_data
        .graphs()
        .iter()
        .filter(|g: &&GraphInfo| g.hash == hash)
    {
        for j in 0..graph.num_automorphisms {
            let auto = &tables::AUTOMORPHISMS[graph.automorphism_index + j];
            // `mapping[i]` is the data point that template point `i` matches.
            // Compose the graph's automorphism (which renames the reference's
            // nodes) with the inverse of the candidate's canonical labelling
            // (which renames the data's nodes into the reference's order).
            for k in 0..num_points {
                mapping[auto[k] as usize] =
                    inverse_labelling[graph.canonical_labelling[k] as usize] as usize;
            }

            let (rmsd, scale, q, _rot) =
                calc_rmsd(&ideal_points, normalized, &mapping, num_points, g1, g2);
            if rmsd < res.rmsd {
                res.rmsd = rmsd;
                res.scale = scale;
                res.q = q;
                for (k, &m) in mapping[..num_points].iter().enumerate() {
                    res.mapping[k] = m as i8;
                }
                res.kind = ref_data.structure.type_code();
                res.best_kind = res.kind;
            }
        }
    }
}

/// Match against a structure with its own facet count and degree bound
/// (upstream `match_general`): SC and BCC.
///
/// Upstream reads only the first `num_nbrs + 1` points, which is meaningful
/// solely because the caller supplies neighbours sorted by distance.
pub fn match_general<const N: usize>(
    ref_data: &RefData,
    ch_points: &[[f64; 3]; N],
    points: &[[f64; 3]; N],
    ch: &mut ConvexHull,
    res: &mut MatchResult,
) -> Result<(), HullError> {
    let Some((form, normalized)) = topological_filter(ref_data, ch_points, points, ch) else {
        return Ok(());
    };

    check_graphs(
        ref_data,
        form.hash,
        &form.canonical_labelling,
        &normalized,
        res,
    );
    Ok(())
}

/// Match against the shared FCC/HCP/ICO topology (upstream
/// `match_fcc_hcp_ico`).
///
/// All three have 12 neighbours, 20 facets and the same degree bound, so the
/// hull is built once and the resulting graph is tested against each requested
/// structure's graph table. That is what makes an FCC/HCP ambiguity a pure RMSD
/// decision between two graph tables rather than two hull computations.
pub fn match_fcc_hcp_ico<const N: usize>(
    ch_points: &[[f64; 3]; N],
    points: &[[f64; 3]; N],
    flags: u32,
    ch: &mut ConvexHull,
    res: &mut MatchResult,
) -> Result<(), HullError> {
    use crate::analysis::ptm::refdata::{FCC, HCP, ICO};

    let num_nbrs = 12usize;
    let num_facets = 20usize;
    let max_degree = 6i8;

    if num_nbrs + 1 > N {
        return Ok(());
    }

    let mut facets = [[0i8; 3]; MAX_FACETS];
    let found = match get_convex_hull(&ch_points[..num_nbrs + 1], ch, &mut facets) {
        Ok(()) => ch.num_facets,
        Err(_) => return Ok(()),
    };
    if found != num_facets {
        return Ok(());
    }

    let (degree, max) = graph_degree(found, &facets, num_nbrs);
    if max > max_degree {
        return Ok(());
    }

    let normalized = subtract_barycentre_n(points, num_nbrs + 1);
    let colours = [0i8; MAX_POINTS];
    let Ok(form) = canonical_form_coloured(found, &facets, num_nbrs, &degree, &colours) else {
        return Ok(());
    };

    for (bit, ref_data) in [(check::FCC, &FCC), (check::HCP, &HCP), (check::ICO, &ICO)] {
        if flags & bit != 0 {
            check_graphs(
                ref_data,
                form.hash,
                &form.canonical_labelling,
                &normalized,
                res,
            );
        }
    }
    Ok(())
}

/// Match the two-shell diamond environment (upstream `match_dcub_dhex`).
///
/// `ch_points` and `points` hold the central atom, its 4 inner neighbours
/// (points 1-4) and, for inner neighbour `k`, its 3 outer neighbours at points
/// `5 + 3k ..= 7 + 3k`, as [`two_shell_environment`] lays them out. The hull of
/// the 16 neighbours is an FCC-like 20-facet polyhedron once every inner atom
/// is reconnected to the triangle of its own outer atoms, which is what the
/// facet surgery below does before the coloured canonical form.
///
/// Reproduced verbatim, including two upstream quirks that decide which
/// environments survive: the first degree check counts only the first 20
/// facets of a hull that may have more, and the facet count is compared with
/// FCC's 20 plus two per inner atom that is itself on the hull.
pub fn match_dcub_dhex(
    ch_points: &[[f64; 3]; MAX_POINTS],
    points: &[[f64; 3]; MAX_POINTS],
    flags: u32,
    ch: &mut ConvexHull,
    res: &mut MatchResult,
) {
    use crate::analysis::ptm::refdata::{DCUB, DHEX, DIAMOND_COLOURS, FCC};
    let num_nbrs = DCUB.num_nbrs;
    let num_facets = FCC.num_facets;
    let max_degree = DCUB.max_degree;

    let mut facets = [[0i8; 3]; MAX_FACETS];
    if get_convex_hull(&ch_points[..num_nbrs + 1], ch, &mut facets).is_err() {
        return;
    }
    let mut nf = ch.num_facets;

    // Facets with more than one inner atom.
    let mut inverted = [false; 4];
    for f in facets.iter().take(nf) {
        let mut n = 0;
        for &v in f {
            if v <= 3 {
                inverted[v as usize] = true;
                n += 1;
            }
        }
        if n > 1 {
            return;
        }
    }
    let num_inverted = inverted.iter().filter(|&&b| b).count();
    if nf != num_facets + 2 * num_inverted {
        return;
    }

    // Upstream passes FCC's `num_facets` (20), not the hull's count, here.
    let (_, md) = graph_degree(num_facets, &facets, num_nbrs);
    if md > max_degree {
        return;
    }

    let mut num_found = 0usize;
    let mut toadd = [[0i8; 3]; 4];
    let mut i = 0usize;
    while i < nf {
        let [a, b, c] = facets[i];
        if a <= 3 || b <= 3 || c <= 3 {
            i += 1;
            continue;
        }
        let (i0, i1, i2) = ((a - 4) / 3, (b - 4) / 3, (c - 4) / 3);
        if i0 == i1 && i0 == i2 {
            if num_found + num_inverted >= 4 {
                return;
            }
            toadd[num_found] = [a, b, c];
            num_found += 1;
            facets[i] = facets[nf - 1];
            nf -= 1;
            // Re-examine slot `i`, which now holds the swapped-in facet.
            continue;
        }
        i += 1;
    }
    if num_found + num_inverted != 4 {
        return;
    }

    for t in toadd.iter().take(num_found) {
        let [a, b, c] = *t;
        let i0 = (a - 4) / 3;
        if nf + 3 > MAX_FACETS {
            return;
        }
        facets[nf] = [i0, b, c];
        facets[nf + 1] = [a, i0, c];
        facets[nf + 2] = [a, b, i0];
        nf += 3;
    }

    let (degree, md) = graph_degree(nf, &facets, num_nbrs);
    if md > max_degree {
        return;
    }

    let normalized = subtract_barycentre_n(points, num_nbrs + 1);
    let Ok(form) = canonical_form_coloured(nf, &facets, num_nbrs, &degree, &DIAMOND_COLOURS) else {
        return;
    };
    if flags & check::DCUB != 0 {
        check_graphs(
            &DCUB,
            form.hash,
            &form.canonical_labelling,
            &normalized,
            res,
        );
    }
    if flags & check::DHEX != 0 {
        check_graphs(
            &DHEX,
            form.hash,
            &form.canonical_labelling,
            &normalized,
            res,
        );
    }
}

/// Match the two-shell graphene environment (upstream `match_graphene`).
///
/// `points` holds the central atom, its 3 inner neighbours (1-3) and two outer
/// neighbours per inner one (`4 + 2k`, `5 + 2k`). There is no hull: the eight
/// orderings of the outer pairs are fitted directly, in upstream's order (the
/// first pair swap happens before the first fit), keeping the strictly lowest
/// RMSD.
pub fn match_graphene(points: &[[f64; 3]; MAX_POINTS], res: &mut MatchResult) {
    use crate::analysis::ptm::refdata::GRAPHENE;
    let num_points = GRAPHENE.num_nbrs + 1;
    let template = GRAPHENE.points(0);
    let mut ideal = [[0.0f64; 3]; MAX_POINTS];
    ideal[..num_points].copy_from_slice(&template[..num_points]);
    let normalized = subtract_barycentre_n(points, num_points);

    let mut g1 = 0.0f64;
    let mut g2 = 0.0f64;
    for i in 0..num_points {
        let (x1, y1, z1) = (ideal[i][0], ideal[i][1], ideal[i][2]);
        let (x2, y2, z2) = (normalized[i][0], normalized[i][1], normalized[i][2]);
        g1 += x1 * x1 + y1 * y1 + z1 * z1;
        g2 += x2 * x2 + y2 * y2 + z2 * z2;
    }

    let mut mapping = [0usize; MAX_POINTS];
    for (k, m) in mapping.iter_mut().enumerate() {
        *m = k;
    }
    for _ in 0..2 {
        mapping.swap(4, 5);
        for _ in 0..2 {
            mapping.swap(6, 7);
            for _ in 0..2 {
                mapping.swap(8, 9);
                let (rmsd, scale, q, _rot) =
                    calc_rmsd(&ideal, &normalized, &mapping, num_points, g1, g2);
                if rmsd < res.rmsd {
                    res.rmsd = rmsd;
                    res.scale = scale;
                    res.q = q;
                    for (k, &m) in mapping[..num_points].iter().enumerate() {
                        res.mapping[k] = m as i8;
                    }
                    res.kind = structure::GRAPHENE;
                    res.best_kind = res.kind;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The literal tables above must agree with the expressions in
    /// `ptm_index.cpp`. A self-fit cannot catch a mis-transcribed factor
    /// (it divides the same number out again), so check the formulas directly.
    #[test]
    fn derived_length_factors_match_upstream_formulas() {
        let s2 = 2.0f64.sqrt();
        let s3 = 3.0f64.sqrt();
        let up_int = [
            0.0,
            1.0,
            1.0,
            7.0 - 3.5 * s3,
            1.0,
            1.0,
            4.0 * s3 / (6.0 * s2 + s3),
            4.0 * s3 / (6.0 * s2 + s3),
            -3.0 / 11.0 + 6.0 * s3 / 11.0,
        ];
        let up_lat = [
            0.0,
            2.0 / s2,
            2.0 / s2,
            2.0 / s3,
            2.0 / s2,
            1.0,
            4.0 / s3,
            4.0 / s3,
            s3,
        ];
        for kind in 1..=8i32 {
            assert_eq!(
                interatomic_distance_factor(kind),
                up_int[kind as usize],
                "interatomic_distance factor for type {kind}"
            );
            assert_eq!(
                lattice_constant_factor(kind),
                up_lat[kind as usize],
                "lattice_constant factor for type {kind}"
            );
        }
        assert_eq!(interatomic_distance_factor(0), 0.0);
        assert_eq!(lattice_constant_factor(0), 0.0);
    }

    /// `scale` is the factor taking normalised data onto the template, so the
    /// distance is `factor / scale`. Both orderings fit a self-fit equally
    /// well, so pin the sign explicitly.
    #[test]
    fn interatomic_distance_divides_by_scale() {
        assert_eq!(interatomic_distance(structure::FCC, 2.0), 0.5);
        assert_eq!(interatomic_distance(structure::SC, 4.0), 0.25);
        // SC's lattice constant is its own nearest-neighbour distance.
        let d = interatomic_distance(structure::SC, 1.5);
        assert_eq!(lattice_constant(structure::SC, d), d);
        // FCC's is a face diagonal: 2/sqrt(2) times the NN distance.
        let d = interatomic_distance(structure::FCC, 1.0);
        let a = lattice_constant(structure::FCC, d);
        assert!(
            (a - d * 2.0f64.sqrt()).abs() < 1e-12,
            "fcc lattice constant {a}"
        );
    }

    use super::*;
    use crate::analysis::ptm::normalize::normalize_vertices_n;
    use crate::analysis::ptm::refdata;
    use crate::analysis::ptm::tables;

    /// The templates are 17-wide and zero-padded past the real points, so trim
    /// to the width a structure actually reads.
    fn trim<const M: usize, const N: usize>(t: &[[f64; 3]; N], n: usize) -> [[f64; 3]; M] {
        let mut out = [[0.0f64; 3]; M];
        out[..n].copy_from_slice(&t[..n]);
        out
    }

    fn sc_template() -> [[f64; 3]; 7] {
        trim::<7, 17>(&tables::SC_TEMPLATES[0], 7)
    }
    fn fcc_template() -> [[f64; 3]; 13] {
        trim::<13, 17>(&tables::FCC_TEMPLATES[0], 13)
    }
    fn hcp_template() -> [[f64; 3]; 13] {
        trim::<13, 17>(&tables::HCP_TEMPLATES[0], 13)
    }
    fn ico_template() -> [[f64; 3]; 13] {
        trim::<13, 17>(&tables::ICO_TEMPLATES[0], 13)
    }
    fn bcc_template() -> [[f64; 3]; 15] {
        trim::<15, 17>(&tables::BCC_TEMPLATES[0], 15)
    }

    /// Feed an ideal template back through the matcher. The point count is the
    /// array width, mirroring how a real environment is normalised once for the
    /// neighbours actually found and then read as a prefix by each reference.
    fn match_ideal<const N: usize>(template: &[[f64; 3]; N], flags: u32) -> MatchResult {
        let (ch_points, _) = normalize_vertices_n(template, N);
        let mut res = MatchResult::new(4.0);
        let mut ch = ConvexHull::default();

        if flags & check::SC != 0 {
            let _ = match_general(&refdata::SC, &ch_points, template, &mut ch, &mut res);
        }
        if flags & check::FCC != 0 || flags & check::HCP != 0 || flags & check::ICO != 0 {
            let _ = match_fcc_hcp_ico(&ch_points, template, flags, &mut ch, &mut res);
        }
        if flags & check::BCC != 0 {
            let _ = match_general(&refdata::BCC, &ch_points, template, &mut ch, &mut res);
        }
        res
    }

    /// The environment width must not leak into the fit.
    ///
    /// `Ptm::run` always hands the matcher a 19-point environment, because that
    /// is the widest any reference needs, and each reference then consumes only
    /// its own prefix: 13 points for FCC/HCP/ICO, 15 for BCC. The fit and its
    /// RMSD must depend on that prefix alone.
    ///
    /// Taking the dot products and the final RMSD over the full array width
    /// would scale the production RMSD by `sqrt(13/19)` - a cutoff of 0.1
    /// silently behaving like 0.082. The other tests here cannot see that,
    /// because they all build environments of exactly the reference's own
    /// width, where the two agree; and a self-fit cannot detect a divisor change
    /// either, because the residual is ~1e-8 and the square root of it barely
    /// moves.
    #[test]
    fn rmsd_is_independent_of_environment_padding() {
        // The 13-point ICO environment scored against the FCC reference. This
        // is the `ico/fcc` cell of the upstream matrix above.
        const EXPECTED: f64 = 0.220739465991453;
        let tight = match_ideal(&ico_template(), check::FCC);
        assert!(
            (tight.rmsd - EXPECTED).abs() < 1e-12,
            "unpadded rmsd {} != {EXPECTED}",
            tight.rmsd
        );

        // Same 13 points, followed by six far-away points. The hull is built
        // from the normalized prefix, so the extra points must not reach it
        // either.
        let mut wide = [[0.0f64; 3]; 19];
        wide[..13].copy_from_slice(&ico_template());
        for (n, p) in wide[13..].iter_mut().enumerate() {
            *p = [40.0 + n as f64, -30.0, 55.0];
        }
        let padded = match_ideal(&wide, check::FCC);

        assert_eq!(padded.kind, tight.kind, "structure changed when padded");
        assert!(
            (padded.rmsd - EXPECTED).abs() < 1e-12,
            "padded rmsd {} != {EXPECTED}",
            padded.rmsd
        );
        assert_eq!(padded.scale, tight.scale, "scale changed when padded");
    }

    /// The reference matrix, captured from compiled upstream
    /// `ptm_structure_matcher.cpp` through a small probe program.
    ///
    /// Two things are worth reading off it. A perfect self-fit is not exactly
    /// zero RMSD upstream: SC and BCC land at 1.6e-8 and 2.2e-8, the numerical
    /// floor of the quaternion fit. And the check set is not a mere veto. The
    /// FCC hull's canonical hash appears in no HCP reference graph, so
    /// restricting the check set to HCP makes an FCC environment report nothing
    /// at all, while the ICO hull's hash appears in the FCC, HCP *and* ICO
    /// tables and is settled purely on RMSD.
    ///
    /// The tolerance is loose on purpose. Structure, mapping, scale and
    /// rotation all agree with upstream exactly, but the RMSD of a
    /// self-fitting environment sits at the QCP iteration's convergence floor,
    /// and a sub-ULP difference in the accumulated inner product is enough to
    /// move which side of that floor the iteration stops on. That shows up as a
    /// difference of order 1e-8 - and it is not even one-sided: upstream
    /// reports 0 for ICO and 2.2e-8 for BCC, this port the other way round. The
    /// decision cutoff is 4.0, six orders of magnitude away, so the residual is
    /// recorded rather than chased.
    #[test]
    fn reproduces_the_upstream_reference_matrix() {
        for (name, res, want_kind, want_rmsd) in [
            (
                "sc",
                match_ideal(&sc_template(), check::SC),
                structure::SC,
                1.59300113864198e-08,
            ),
            // Upstream reports 2.18e-8 here. Both results are self-fits to
            // within the quaternion iteration's convergence floor; see the note
            // on the tolerance below.
            (
                "bcc",
                match_ideal(&bcc_template(), check::BCC),
                structure::BCC,
                2.17645389837093e-08,
            ),
            (
                "fcc/fcc",
                match_ideal(&fcc_template(), check::FCC),
                structure::FCC,
                0.0,
            ),
            (
                "fcc/hcp",
                match_ideal(&fcc_template(), check::HCP),
                structure::OTHER,
                f64::INFINITY,
            ),
            (
                "fcc/ico",
                match_ideal(&fcc_template(), check::ICO),
                structure::OTHER,
                f64::INFINITY,
            ),
            (
                "hcp/fcc",
                match_ideal(&hcp_template(), check::FCC),
                structure::OTHER,
                f64::INFINITY,
            ),
            (
                "hcp/hcp",
                match_ideal(&hcp_template(), check::HCP),
                structure::HCP,
                0.0,
            ),
            (
                "hcp/ico",
                match_ideal(&hcp_template(), check::ICO),
                structure::OTHER,
                f64::INFINITY,
            ),
            (
                "ico/fcc",
                match_ideal(&ico_template(), check::FCC),
                structure::FCC,
                0.220739465991453,
            ),
            (
                "ico/hcp",
                match_ideal(&ico_template(), check::HCP),
                structure::HCP,
                0.243665189523163,
            ),
            // Upstream reports exactly 0 here.
            (
                "ico/ico",
                match_ideal(&ico_template(), check::ICO),
                structure::ICO,
                2.3378865139e-08,
            ),
            (
                "ico/family",
                match_ideal(&ico_template(), check::DEFAULT),
                structure::ICO,
                2.3378865139e-08,
            ),
        ] {
            assert_eq!(res.kind, want_kind, "{name}: wrong structure");
            if want_rmsd.is_finite() {
                assert!(
                    (res.rmsd - want_rmsd).abs() < 1e-6,
                    "{name}: rmsd {} != upstream {want_rmsd}",
                    res.rmsd
                );
            } else {
                assert!(
                    res.rmsd.is_infinite(),
                    "{name}: rmsd {} is finite",
                    res.rmsd
                );
            }
        }
    }

    /// Each ideal template identifies its own structure, up to the numerical
    /// floor upstream itself reports.
    #[test]
    fn each_ideal_template_matches_its_own_structure() {
        for (name, res, want) in [
            ("sc", match_ideal(&sc_template(), check::SC), structure::SC),
            (
                "fcc",
                match_ideal(&fcc_template(), check::DEFAULT),
                structure::FCC,
            ),
            (
                "hcp",
                match_ideal(&hcp_template(), check::DEFAULT),
                structure::HCP,
            ),
            (
                "hcp1",
                match_ideal(
                    &trim::<13, 17>(&tables::HCP_TEMPLATES[1], 13),
                    check::DEFAULT,
                ),
                structure::HCP,
            ),
            (
                "ico",
                match_ideal(&ico_template(), check::DEFAULT),
                structure::ICO,
            ),
        ] {
            assert_eq!(res.kind, want, "{name}: wrong structure");
            assert!(
                res.rmsd < 1e-6,
                "{name}: rmsd {} is not a self-fit",
                res.rmsd
            );
            assert!(res.is_match(), "{name}: self-fit was rejected");
            assert!(
                (res.scale - 1.0).abs() < 1e-9,
                "{name}: scale {}",
                res.scale
            );
        }
    }

    /// The FCC self-fit mapping is a specific permutation, and reproducing it
    /// exactly is the strongest available check that the automorphism walk and
    /// the canonical-labelling inversion agree with upstream.
    #[test]
    fn reproduces_the_upstream_fcc_mapping() {
        let res = match_ideal(&fcc_template(), check::DEFAULT);
        assert_eq!(res.kind, structure::FCC);
        assert_eq!(
            &res.mapping[..13],
            &[0, 7, 11, 6, 10, 8, 3, 1, 5, 12, 4, 2, 9]
        );
    }

    /// The check set genuinely removes structures: asking only for ICO must not
    /// report FCC, even though the ICO environment does fit FCC to 0.22.
    #[test]
    fn the_check_set_excludes_untested_structures() {
        let t = ico_template();
        assert_eq!(match_ideal(&t, check::ICO).kind, structure::ICO);
        assert_eq!(match_ideal(&t, check::FCC).kind, structure::FCC);
        assert_eq!(match_ideal(&t, check::HCP).kind, structure::HCP);
        assert_eq!(match_ideal(&t, check::BCC).kind, structure::OTHER);
        assert_eq!(match_ideal(&t, 0).kind, structure::OTHER);
    }

    /// With everything enabled the best RMSD across the candidate structures
    /// wins, so the icosahedral environment is still called ICO rather than the
    /// FCC it also fits.
    #[test]
    fn the_best_fitting_structure_wins() {
        for (name, res, want) in [
            (
                "fcc",
                match_ideal(&fcc_template(), check::DEFAULT),
                structure::FCC,
            ),
            (
                "hcp",
                match_ideal(&hcp_template(), check::DEFAULT),
                structure::HCP,
            ),
            (
                "ico",
                match_ideal(&ico_template(), check::DEFAULT),
                structure::ICO,
            ),
        ] {
            assert_eq!(res.kind, want, "{name}");
            assert!(res.rmsd < 1e-6, "{name}: rmsd {}", res.rmsd);
        }
    }

    /// A rigid rotation and uniform scaling of an environment must not change
    /// the verdict: the fit absorbs both.
    #[test]
    fn a_rotated_and_scaled_environment_still_matches() {
        let base = fcc_template();
        let (angle, axis) = (37.0f64.to_radians(), [0.3f64, -0.5, 0.81]);
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let (s, c) = (angle / 2.0).sin_cos();
        let q = [c, s * axis[0] / n, s * axis[1] / n, s * axis[2] / n];
        // `quaternion_to_rotation_matrix` is row-major: rot[3 * row + col].
        let rot = crate::analysis::ptm::polar::quaternion_to_rotation_matrix(&q);
        let scale = 2.5;
        let mut moved = base;
        for p in moved.iter_mut() {
            let v = *p;
            for col in 0..3 {
                p[col] = scale * (rot[col] * v[0] + rot[3 + col] * v[1] + rot[6 + col] * v[2]);
            }
        }

        let res = match_ideal(&moved, check::DEFAULT);
        assert_eq!(res.kind, structure::FCC);
        assert!(res.rmsd < 1e-6, "rmsd {}", res.rmsd);
        assert!(
            (res.scale - 1.0 / scale).abs() < 1e-9,
            "scale {}",
            res.scale
        );
    }

    /// Distorting the neighbour radii keeps the icosahedral topology - the hull
    /// is still 20 facets, so the topological filter passes - but wrecks the
    /// fit, and the cutoff is what finally rejects it.
    #[test]
    fn a_distorted_environment_is_rejected_by_the_cutoff() {
        let base = ico_template();
        let mut warped = base;
        for (i, p) in warped.iter_mut().enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let target = 0.45 + 1.9 * ((i % 5) as f64) / 4.0;
            for pk in p.iter_mut() {
                *pk *= target / r;
            }
        }

        let res = match_ideal(&warped, check::DEFAULT);
        let ideal = match_ideal(&base, check::DEFAULT);
        assert!(res.rmsd > ideal.rmsd, "expected a worse fit");
        assert!(
            res.rmsd > res.rmsd_cutoff,
            "rmsd {} should exceed the default cutoff",
            res.rmsd
        );
        assert!(!res.is_match(), "a bad fit was accepted");
        assert!(ideal.is_match(), "the undistorted environment was rejected");
    }

    /// The cutoff is the caller's decision, not the matcher's, so tightening it
    /// must reject a perfect fit without changing the fit that was found.
    #[test]
    fn the_cutoff_is_applied_by_the_caller() {
        let t = ico_template();
        let (ch_points, _) = normalize_vertices_n(&t, 13);
        let mut res = MatchResult::new(0.0);
        let mut ch = ConvexHull::default();
        let _ = match_fcc_hcp_ico(&ch_points, &t, check::DEFAULT, &mut ch, &mut res);
        assert_eq!(res.kind, structure::ICO);
        assert!(!res.is_match(), "a zero cutoff accepted a perfect fit");
    }

    /// SC and BCC read a fixed prefix of the environment: SC takes the nearest
    /// six, BCC the nearest fourteen. A cuboctahedral environment has neither
    /// an 8-facet prefix nor a 14-point one, so neither structure can claim it.
    #[test]
    fn sc_and_bcc_read_only_their_own_prefix() {
        // Seven points: a perfect octahedron, then six far outliers.
        let mut padded = [[0.0f64; 3]; 13];
        padded[..7].copy_from_slice(&sc_template());
        for (i, p) in padded[7..].iter_mut().enumerate() {
            *p = [50.0 + i as f64, 0.0, 0.0];
        }

        let res = match_ideal(&padded, check::SC);
        assert_eq!(res.kind, structure::SC, "SC ignored the outliers");
        assert!(res.rmsd < 1e-6, "rmsd {}", res.rmsd);

        // 13 points cannot supply BCC's 14.
        let res = match_ideal(&fcc_template(), check::BCC);
        assert_eq!(res.kind, structure::OTHER);
        assert_eq!(res.rmsd, f64::INFINITY);
    }

    /// Whatever the structure, the mapping must be a permutation covering every
    /// point, and the untouched tail must stay marked as unmatched.
    #[test]
    fn the_mapping_is_a_permutation_covering_every_point() {
        for t in [fcc_template(), hcp_template(), ico_template()] {
            let res = match_ideal(&t, check::DEFAULT);
            let n = t.len();
            let mut seen: Vec<i8> = res.mapping[..n].to_vec();
            seen.sort_unstable();
            assert_eq!(seen, (0..n as i8).collect::<Vec<i8>>());
            assert!(res.mapping[n..].iter().all(|m| *m == -1));
        }
    }
    /// Permuting tied neighbours must not change the verdict.
    ///
    /// An ideal crystal has whole shells at *identical* distance: FCC and HCP each
    /// put 12 neighbours at exactly `a`, BCC 14, SC 6. So the order within a shell
    /// is a free choice, and whatever the neighbour list happens to produce, the
    /// matcher has to reach the same structure. The neighbour list breaks ties by
    /// atom index, but that is an arbitrary convention - a different host (OVITO,
    /// LAMMPS) will break them differently, and a thermal snapshot has shells that
    /// are near-tied rather than exactly tied.
    ///
    /// This is the empirical form of the claim that the reference graphs are
    /// compared up to their automorphism group: if the automorphisms cover every
    /// reordering, the result is invariant. The templates are used directly as
    /// data, so the only thing varying between runs is the neighbour order.
    #[test]
    fn tied_neighbour_order_does_not_change_the_verdict() {
        /// Deterministic shuffle: no RNG in a correctness test, and a failure has
        /// to be reproducible.
        fn permute(n: usize, seed: u64) -> Vec<usize> {
            let mut v: Vec<usize> = (0..n).collect();
            let mut s = seed | 1;
            for i in (1..n).rev() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                v.swap(i, (s % (i as u64 + 1)) as usize);
            }
            v
        }

        use crate::analysis::ptm::structure;
        // (name, template, shell size, flags, expected type code)
        type Case<'a> = (&'a str, &'a [[f64; 3]], usize, u32, i32);
        let cases: &[Case<'_>] = &[
            ("sc", &sc_template(), 6, check::SC, structure::SC),
            ("fcc", &fcc_template(), 12, check::FCC, structure::FCC),
            ("hcp", &hcp_template(), 12, check::HCP, structure::HCP),
            ("ico", &ico_template(), 12, check::ICO, structure::ICO),
            ("bcc", &bcc_template(), 14, check::BCC, structure::BCC),
        ];

        for (name, tpl, n_shell, flags, want) in cases {
            let num_points = n_shell + 1;
            let mut rng = 0x9E3779B97F4A7C15u64;
            let mut seen = std::collections::HashSet::new();
            for trial in 0..200 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let order = permute(*n_shell, rng ^ trial);
                seen.insert(order.clone());

                // `points[0]` is the central atom and must stay put: only the
                // neighbours are a free order.
                let mut points = [[0.0f64; 3]; 17];
                points[0] = tpl[0];
                for (slot, &src) in points[1..num_points].iter_mut().zip(order.iter()) {
                    *slot = tpl[1 + src];
                }
                let wide: [[f64; 3]; 17] = points;
                let res = match_ideal(&wide, *flags);
                assert_eq!(
                    res.kind, *want,
                    "{name}: trial {trial} order {order:?} gave {} (rmsd {})",
                    res.kind, res.rmsd
                );
                assert!(
                    res.rmsd < 1e-6,
                    "{name}: trial {trial} gave a poor fit, rmsd {}",
                    res.rmsd
                );
            }
            assert!(seen.len() > 150, "{name}: shuffle was not varied enough");
        }
    }

    /// The two-shell templates fit themselves through the diamond hull
    /// surgery and the graphene permutations, and the check set is honoured.
    #[test]
    fn two_shell_templates_match_themselves() {
        let run_dia = |t: &[[f64; 3]; 17], flags: u32| {
            let (ch_points, _) = normalize_vertices_n(t, 17);
            let mut res = MatchResult::new(0.1);
            let mut ch = ConvexHull::default();
            match_dcub_dhex(&ch_points, t, flags, &mut ch, &mut res);
            res
        };
        for (name, t, want) in [
            ("dcub0", &tables::DCUB_TEMPLATES[0], structure::DCUB),
            ("dcub1", &tables::DCUB_TEMPLATES[1], structure::DCUB),
            ("dhex0", &tables::DHEX_TEMPLATES[0], structure::DHEX),
            ("dhex3", &tables::DHEX_TEMPLATES[3], structure::DHEX),
        ] {
            let res = run_dia(t, check::DCUB | check::DHEX);
            assert_eq!(res.kind, want, "{name}");
            assert!(res.rmsd < 1e-6, "{name}: rmsd {}", res.rmsd);
            assert!(
                (res.scale - 1.0).abs() < 1e-9,
                "{name}: scale {}",
                res.scale
            );
        }
        // With only the other diamond checked, nothing acceptable is found
        // (a hash collision may still produce a fit, but a poor one).
        for (t, flags) in [
            (&tables::DCUB_TEMPLATES[0], check::DHEX),
            (&tables::DHEX_TEMPLATES[0], check::DCUB),
        ] {
            let res = run_dia(t, flags);
            assert!(!res.is_match() || res.rmsd > 0.1, "rmsd {}", res.rmsd);
            assert!(res.rmsd > 1e-3, "cross-diamond self-fit: rmsd {}", res.rmsd);
        }
        for t in &tables::GRAPHENE_TEMPLATES {
            let mut res = MatchResult::new(0.1);
            match_graphene(t, &mut res);
            assert_eq!(res.kind, structure::GRAPHENE);
            assert!(res.rmsd < 1e-6, "graphene rmsd {}", res.rmsd);
        }
    }
}
