//! Common-neighbour analysis (design §13).
//!
//! Faithful port of OVITO's `CommonNeighborAnalysisModifier`. For each central
//! atom and each of its bonded neighbours `i`:
//!
//! ```text
//! common = N(i) ∩ N(centre)       (neighbours shared by centre and i)
//! bonds  = |bonds among that common set|
//! chain  = longest bond path inside the common set
//! ```
//!
//! The triple is the `abc` signature. Structure codes are
//! `OTHER=0, FCC=1, HCP=2, BCC=3, ICO=4` (design §13.1).
//!
//! Adaptive mode mirrors `determineStructureAdaptive`: it computes the
//! 12-neighbour signature first (`421`/`422`/`555` for FCC/HCP/ICO) and only
//! falls through to the 14-neighbour signature (`444`/`666` for BCC) when that
//! yields `OTHER`. The fixed and shell modes mirror
//! `analyzeSmallSignature`/`analyzeLargeSignature` under an exact 12- or
//! 14-neighbour gate.
//!
//! Three further modes port the rest of OVITO 3.15's structure-identification
//! CNA family:
//!
//! * [`CnaMode::Interval`] — Larsen's interval CNA (OVITO `IntervalCutoff`,
//!   `determineStructureInterval`): bonds are added shortest first and every
//!   gap in the neighbour-neighbour distances whose bond graph has the right
//!   coordination is tried; the widest interval with a valid signature wins.
//!   Parameter-free.
//! * [`CnaMode::Bond`] — bond-based CNA (OVITO `BondBased` on the bonds of a
//!   `CreateBondsModifier` with a uniform cutoff): one `(common, bonds, chain)`
//!   signature per bond, over every bond an atom has (no 12/14 gate).
//! * [`CnaMode::Diamond`] — OVITO's `IdentifyDiamondModifier`: an FCC/HCP CNA
//!   over the 12 second neighbours reached through the 4 nearest, then the
//!   first- and second-neighbour shells of each diamond atom. It writes
//!   [`diamond`] codes, not the FCC/HCP/BCC/ICO ones.
//!
//! The hot loop keeps the central atom's neighbour set as a `u16` bond bitmask
//! over the ≤14 nearest neighbours, as OVITO's `CNAPairBond` does, so the
//! common-neighbour subgraph search needs no heap and stays L1-resident.

use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::data::property::PropertyStore;
use crate::error::Result;
use crate::neighbor::{KnnIndex, KnnScratch};

/// Structure codes, matching OVITO's `CommonNeighborAnalysisModifier::StructureType`.
pub const OTHER: u8 = 0;
pub const FCC: u8 = 1;
pub const HCP: u8 = 2;
pub const BCC: u8 = 3;
pub const ICO: u8 = 4;

/// Neighbours scanned by the small (FCC/HCP/ICO) signatures.
const N_SMALL: usize = 12;
/// Neighbours scanned by the large (BCC) signature.
const N_LARGE: usize = 14;

/// `(1+√2)/2` — the adaptive small-case scale factor (design §13.1).
const ADAPTIVE_SCALE: f64 = 0.5 * (1.0 + std::f64::consts::SQRT_2);
/// OVITO's `1.207f` — the adaptive BCC-case scale factor (design §13.1).
const LARGE_SCALE: f64 = 1.207;
/// Bitmask of the 12 neighbours the small signatures scan.
const SMALL_KEEP: u16 = (1 << N_SMALL) - 1;
/// Bitmask of the 14 neighbours the large signature scans.
const LARGE_KEEP: u16 = (1 << N_LARGE) - 1;
/// k-NN fetches one beyond the 14 the signatures use, so the fixed/shell
/// gates can detect "15 or more inside the cutoff".
const MAX_FETCH: usize = N_LARGE + 1;

/// Cutoff-mode selection (design §13.1).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CnaMode {
    /// Per-atom cutoff from the nearest neighbours.
    #[default]
    Adaptive,
    /// Single global cutoff.
    Fixed { cutoff: f64 },
    /// Two radii bracketing the nearest neighbours: the neighbours in
    /// `(low, high]`, an exact 12 or 14 of them, bonded at `high`. Not an OVITO
    /// mode; [`CnaMode::Interval`] is Larsen's interval CNA.
    Shell { low: f64, high: f64 },
    /// Larsen's interval CNA (OVITO `IntervalCutoff`).
    Interval,
    /// Bond-based CNA over the bonds within `cutoff` (OVITO `BondBased` after
    /// `CreateBondsModifier(cutoff)`).
    Bond { cutoff: f64 },
    /// OVITO's identify-diamond-structure ([`diamond`] codes).
    Diamond,
}

/// Structure codes written by [`CnaMode::Diamond`], OVITO's
/// `IdentifyDiamondModifier::StructureType` numbering.
pub mod diamond {
    pub const OTHER: u8 = 0;
    pub const CUBIC: u8 = 1;
    pub const CUBIC_FIRST: u8 = 2;
    pub const CUBIC_SECOND: u8 = 3;
    pub const HEX: u8 = 4;
    pub const HEX_FIRST: u8 = 5;
    pub const HEX_SECOND: u8 = 6;
    /// Summary names, indexed by code.
    pub const NAMES: [&str; 7] = [
        "other",
        "cubic_diamond",
        "cubic_diamond_first",
        "cubic_diamond_second",
        "hex_diamond",
        "hex_diamond_first",
        "hex_diamond_second",
    ];
}

/// Per-atom signature counters, emitted when `--cna-full` is given
/// (design §13.2).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SignatureCounts {
    pub n421: u32,
    pub n422: u32,
    pub n555: u32,
    pub n444: u32,
    pub n666: u32,
}

impl SignatureCounts {
    /// Classify from the signature counts alone (design §13.1).
    ///
    /// The order matters only for the tie case: OVITO tests the exact
    /// signatures, so `OTHER` is the residual class.
    ///
    /// Note these thresholds are the *saturated* counts of a full 12-neighbour
    /// (small) or 14-neighbour (large) scan. The small counters are only
    /// meaningful when `analyze_small` scanned all 12 and the large ones when
    /// `analyze_large` scanned all 14; the fixed/shell modes additionally
    /// require the atom to have exactly that many neighbours in range.
    pub fn classify(&self) -> u8 {
        if self.n421 == 12 {
            return FCC;
        }
        if self.n421 == 6 && self.n422 == 6 {
            return HCP;
        }
        if self.n555 == 12 {
            return ICO;
        }
        if self.n444 == 6 && self.n666 == 8 {
            return BCC;
        }
        OTHER
    }

    fn as_array(&self) -> [u32; 5] {
        [self.n421, self.n422, self.n555, self.n444, self.n666]
    }
}

pub struct Cna {
    pub mode: CnaMode,
    /// Emit the per-signature counter properties (design §13.2 `--cna-full`).
    pub full: bool,
}

impl Default for Cna {
    fn default() -> Self {
        Cna {
            mode: CnaMode::Adaptive,
            full: false,
        }
    }
}

impl Cna {
    pub fn new(mode: CnaMode) -> Self {
        Cna { mode, full: false }
    }

    pub fn with_full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }
}

/// The 12/14 nearest neighbours of one central atom, with the bond bitmask
/// between them. Allocated once per thread and reused (no per-atom heap churn).
struct Neighbourhood {
    /// Squared distance from the centre, ascending.
    d2: [f64; N_LARGE],
    /// Image-true displacement from the centre, one array per axis (so the
    /// pair distances are branch-free, vectorisable sweeps), parallel to `d2`.
    ///
    /// Kept so neighbour-neighbour bonds are a direct `|dr[a] - dr[b]|`
    /// comparison. The same atom may appear twice when a periodic image is
    /// also within range; the two entries are distinct bond slots. Re-querying
    /// the cell list by atom id instead would conflate two periodic images of
    /// the same atom and over-bond (measured: 60 pairs instead of 36 for 2x2x2
    /// BCC at the adaptive radius).
    dr: [[f64; N_LARGE]; 3],
    /// Squared neighbour-neighbour distances, computed on first use. Adaptive
    /// mode bonds the same neighbours at two radii (the 12- and 14-neighbour
    /// cutoffs), so each pair's distance is computed once and both bond
    /// matrices are threshold compares on it.
    pair: Option<[[f64; N_LARGE]; N_LARGE]>,
    /// `bond[i]` has bit j set when neighbours i and j are bonded.
    bond: [u16; N_LARGE],
    /// Number of valid entries (≤14).
    len: usize,
}

impl Neighbourhood {
    fn new() -> Self {
        Neighbourhood {
            d2: [f64::MAX; N_LARGE],
            dr: [[0.0; N_LARGE]; 3],
            pair: None,
            bond: [0; N_LARGE],
            len: 0,
        }
    }
}

impl Analysis for Cna {
    fn name(&self) -> &'static str {
        "cna"
    }

    /// None, in every mode: CNA gathers its neighbours with its own exact,
    /// unbounded k-NN index (design §9.2), as OVITO does, and each mode's
    /// cutoff is applied to that distance-ordered list.
    fn cutoffs(&self) -> Vec<f64> {
        Vec::new()
    }

    fn run(
        &self,
        ctx: &AnalysisContext,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        match self.mode {
            CnaMode::Bond { cutoff } => return self.run_bond(ctx, cutoff, out, summary),
            CnaMode::Diamond => return run_diamond(ctx, out, summary),
            _ => {}
        }
        let n = ctx.frame.natoms();
        let positions = &ctx.frame.particles.positions[..n];
        let index = {
            let _t = crate::prof::StageTimer::start("cna.knn_build");
            KnnIndex::build(&ctx.frame.cell, positions, ctx.par)?
        };
        // Adaptive mode never looks past the 14th neighbour, and the query is
        // an exact top-k under a total order, so asking for 14 returns exactly
        // the first 14 of the 15. It is also far cheaper: in BCC the 15th
        // neighbour is in the third shell (a√2), beyond one bin width, so a
        // 15-query usually scans 5³ bins where a 14-query stops at 3³.
        let fetch = match self.mode {
            CnaMode::Adaptive | CnaMode::Interval => N_LARGE,
            _ => MAX_FETCH,
        };
        let classify_atom = |(scratch, entries): &mut (KnnScratch, Vec<_>), i: usize| {
            let mut nb = Neighbourhood::new();
            index.knn(i, positions[i], fetch, scratch, entries);
            let m = entries.len().min(N_LARGE);
            nb.len = m;
            for (t, &(_, dr_v, d2_v)) in entries[..m].iter().enumerate() {
                nb.d2[t] = d2_v;
                nb.dr[0][t] = dr_v[0];
                nb.dr[1][t] = dr_v[1];
                nb.dr[2][t] = dr_v[2];
            }
            // The 15th candidate, if any: used only by the fixed/shell modes
            // to tell an exact 12/14 from an over-coordinated atom.
            let next_d2 = entries.get(N_LARGE).map(|e| e.2);
            self.classify(&mut nb, next_d2)
        };
        let scratch = || (KnnScratch::default(), Vec::with_capacity(MAX_FETCH));

        // Each atom is independent, so the scan parallelises exactly; results
        // are written to a pre-sized slot, keeping it deterministic. Atoms are
        // visited in the index's bin order, not file order, so consecutive
        // queries read the same bins (measured: the scan is memory-bound in
        // file order), and the labels are scattered back to atom order after.
        // The per-atom counters (20 B/atom) are only kept when `full` asks.
        let _t_scan = crate::prof::StageTimer::start("cna.scan");
        let order = index.order();
        let (types, counts): (Vec<u8>, Vec<SignatureCounts>) = if self.full {
            let mut by_slot = vec![(OTHER, SignatureCounts::default()); n];
            ctx.par
                .try_fill_with(&mut by_slot, scratch, |s, slot, out| {
                    *out = classify_atom(s, order[slot] as usize);
                    Ok::<(), std::convert::Infallible>(())
                })
                .unwrap_or_else(|e| match e {});
            let mut types = vec![OTHER; n];
            let mut counts = vec![SignatureCounts::default(); n];
            for (&i, &(t, c)) in order.iter().zip(&by_slot) {
                types[i as usize] = t;
                counts[i as usize] = c;
            }
            (types, counts)
        } else {
            let mut by_slot = vec![OTHER; n];
            ctx.par
                .try_fill_with(&mut by_slot, scratch, |s, slot, out| {
                    *out = classify_atom(s, order[slot] as usize).0;
                    Ok::<(), std::convert::Infallible>(())
                })
                .unwrap_or_else(|e| match e {});
            let mut types = vec![OTHER; n];
            for (&i, &t) in order.iter().zip(&by_slot) {
                types[i as usize] = t;
            }
            (types, Vec::new())
        };
        drop(_t_scan);
        drop(index);

        let mut hist = [0usize; 5];
        for &t in &types {
            hist[t as usize] += 1;
        }
        for code in [OTHER, FCC, HCP, BCC, ICO] {
            summary.merge_from(Summary::scalar(
                &format!("cna_{}", code_name(code)),
                hist[code as usize] as f64,
            ));
        }
        out.register_scalar(crate::data::property::DType::U8, "StructureType", types);
        if self.full {
            const NAMES: [&str; 5] = ["cna_n421", "cna_n422", "cna_n555", "cna_n444", "cna_n666"];
            for (comp, name) in NAMES.iter().enumerate() {
                let v: Vec<i32> = counts.iter().map(|c| c.as_array()[comp] as i32).collect();
                out.register_scalar(crate::data::property::DType::I32, name, v);
                // The whole-frame total of each neighbour count, reported beside
                // the per-atom array it sums so the CLI's `--full` block and the
                // Python binding's stats dict come from one place rather than
                // the printer re-deriving it.
                let total: i64 = counts.iter().map(|c| c.as_array()[comp] as i64).sum();
                summary.merge_from(Summary::scalar(&format!("{name}_total"), total as f64));
            }
        }
        Ok(())
    }
}

impl Cna {
    /// Classify one atom from its gathered neighbourhood.
    ///
    /// `next_d2` is the squared distance to the 15th nearest neighbour, when
    /// one exists; the fixed and shell modes need it to tell an exact 12/14
    /// from an over-coordinated atom (which is `OTHER`).
    fn classify(&self, nb: &mut Neighbourhood, next_d2: Option<f64>) -> (u8, SignatureCounts) {
        match self.mode {
            // OVITO's adaptive order matters: it tries the 12-neighbour
            // FCC/HCP/ICO signature first and only falls through to the
            // 14-neighbour BCC signature if that is `OTHER`. Choosing 12 vs 14
            // up front from the 13th distance would misclassify a thermally
            // perturbed FCC atom.
            //
            // With a full 14-neighbour shell, both bond matrices come from one
            // sweep: every atom that is not FCC/HCP/ICO needs the second (in a
            // BCC sample, nearly every atom).
            CnaMode::Adaptive => {
                if nb.len < N_SMALL {
                    return (OTHER, SignatureCounts::default());
                }
                let large = if nb.len >= N_LARGE {
                    Some(build_both_bonds(nb, small_cutoff(nb), large_cutoff(nb)))
                } else {
                    build_bonds(nb, SMALL_KEEP, small_cutoff(nb));
                    None
                };
                let c = analyze_small(nb, SMALL_KEEP);
                if c.classify() != OTHER {
                    return (c.classify(), c);
                }
                if let Some(large) = large {
                    nb.bond = large;
                    let c = analyze_large(nb, LARGE_KEEP);
                    if c.classify() != OTHER {
                        return (c.classify(), c);
                    }
                }
                (OTHER, SignatureCounts::default())
            }
            // OVITO's fixed mode counts every neighbour inside the cutoff and
            // classifies only an exact 12 (small) or 14 (large).
            CnaMode::Fixed { cutoff } => {
                let c2 = cutoff * cutoff;
                let (keep, count) = count_within(nb, next_d2, |d2| d2 <= c2);
                classify_exact(nb, keep, count, cutoff)
            }
            // A single fixed-width shell.
            CnaMode::Shell { low, high } => {
                let lo2 = low * low;
                let hi2 = high * high;
                let (keep, count) = count_within(nb, next_d2, |d2| d2 > lo2 && d2 <= hi2);
                classify_exact(nb, keep, count, high)
            }
            CnaMode::Interval => classify_interval(nb),
            // Handled by `run` before any per-atom k-NN scan.
            CnaMode::Bond { .. } | CnaMode::Diamond => (OTHER, SignatureCounts::default()),
        }
    }
}

/// Select the neighbours whose squared distance passes `pred`, returning the
/// bond mask over the stored slots and the total count (including the 15th
/// candidate, so `count` may exceed the number of mask bits).
fn count_within(
    nb: &Neighbourhood,
    next_d2: Option<f64>,
    pred: impl Fn(f64) -> bool,
) -> (u16, usize) {
    let mut keep = 0u16;
    let mut count = 0usize;
    for t in 0..nb.len {
        if pred(nb.d2[t]) {
            keep |= 1 << t;
            count += 1;
        }
    }
    if let Some(d2) = next_d2 {
        if pred(d2) {
            count += 1;
        }
    }
    (keep, count)
}

/// Fixed/shell classification: an exact 12 runs the small signature, an
/// exact 14 the large one, and anything else is `OTHER`.
fn classify_exact(
    nb: &mut Neighbourhood,
    keep: u16,
    count: usize,
    radius: f64,
) -> (u8, SignatureCounts) {
    if count == N_SMALL {
        build_bonds(nb, keep, radius);
        let c = analyze_small(nb, keep);
        (c.classify(), c)
    } else if count == N_LARGE {
        build_bonds(nb, keep, radius);
        let c = analyze_large(nb, keep);
        (c.classify(), c)
    } else {
        (OTHER, SignatureCounts::default())
    }
}

/// Fill `nb.bond` with the neighbour-neighbour bonds among the `keep` slots,
/// using `|dr[a] - dr[b]| <= radius`. `dr` are minimum-image displacements from
/// the same centre, so their difference is already the image-to-image
/// separation and needs no further periodic reduction.
///
/// Each row is one fixed-length compare over all 14 slots of the pair-distance
/// matrix, masked afterwards, rather than a branchy upper-triangle walk.
fn build_bonds(nb: &mut Neighbourhood, keep: u16, radius: f64) {
    let mut bond = [0u16; N_LARGE];
    let b2 = radius * radius;
    let keep = keep & ((1u32 << nb.len) - 1) as u16;
    if b2 > 0.0 {
        let pair = nb.pair.get_or_insert_with(|| pair_d2(&nb.dr));
        for a in 0..nb.len {
            if keep & (1 << a) == 0 {
                continue;
            }
            let mut row = 0u16;
            for (b, &d2) in pair[a].iter().enumerate() {
                row |= ((d2 <= b2) as u16) << b;
            }
            bond[a] = row & keep & !(1 << a);
        }
    }
    nb.bond = bond;
}

/// Adaptive mode with a full shell: the small (12-neighbour, `small_r`) bonds
/// into `nb.bond`, and the large (14-neighbour, `large_r`) bonds returned.
///
/// Exactly what [`build_bonds`] gives for each `(keep, radius)` in turn: the
/// same squared distance, as [`pair_d2`] computes it, is compared against
/// both radii in one square sweep, so neither the distances nor a matrix of
/// them is kept. A radius whose square is not positive bonds nothing.
fn build_both_bonds(nb: &mut Neighbourhood, small_r: f64, large_r: f64) -> [u16; N_LARGE] {
    // -1 is below every squared distance, so "bonds nothing" needs no branch.
    let positive = |r2: f64| if r2 > 0.0 { r2 } else { -1.0 };
    let (s2, l2) = (positive(small_r * small_r), positive(large_r * large_r));
    let [x, y, z] = &nb.dr;
    let mut small = [0u16; N_LARGE];
    let mut large = [0u16; N_LARGE];
    for a in 0..N_LARGE {
        let (mut rs, mut rl) = (0u16, 0u16);
        for b in 0..N_LARGE {
            let d = [x[a] - x[b], y[a] - y[b], z[a] - z[b]];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            rs |= ((d2 <= s2) as u16) << b;
            rl |= ((d2 <= l2) as u16) << b;
        }
        if a < N_SMALL {
            small[a] = rs & SMALL_KEEP & !(1 << a);
        }
        large[a] = rl & LARGE_KEEP & !(1 << a);
    }
    nb.bond = small;
    large
}

/// Squared distances between every two neighbour slots, as the full square:
/// fixed-length sweeps the compiler vectorises. Entry `[b][a]` squares
/// `dr[b] - dr[a]`, the exact negation of `[a][b]`'s difference, so the matrix
/// is exactly symmetric. Unused slots hold zero displacement; their entries
/// are masked out by the callers' `keep`.
fn pair_d2(dr: &[[f64; N_LARGE]; 3]) -> [[f64; N_LARGE]; N_LARGE] {
    let [x, y, z] = dr;
    std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let d = [x[a] - x[b], y[a] - y[b], z[a] - z[b]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        })
    })
}

/// 12-neighbour local cutoff: `(1/12)·Σ√d² · (1+√2)/2` over the 12 nearest
/// (OVITO `determineStructureAdaptive`, small case).
fn small_cutoff(nb: &Neighbourhood) -> f64 {
    let mut sum = 0.0;
    for t in 0..N_SMALL {
        sum += nb.d2[t].sqrt();
    }
    sum / N_SMALL as f64 * ADAPTIVE_SCALE
}

/// 14-neighbour local cutoff (OVITO `determineStructureAdaptive`, BCC case).
///
/// The first 8 neighbours are weighted as `√(d²/(3/4))` and the next 6 as
/// `√d²`, then scaled by `1.207/14`. That weighting maps bcc's first shell
/// (`a√3/2`) and second shell (`a`) onto the same value `a`, so the cutoff
/// lands above the first shell but below the second — which is what keeps the
/// 8/6 split the 444/666 signatures need.
fn large_cutoff(nb: &Neighbourhood) -> f64 {
    let mut sum = 0.0;
    for t in 0..8 {
        sum += (nb.d2[t] / 0.75).sqrt();
    }
    for t in 8..N_LARGE {
        sum += nb.d2[t].sqrt();
    }
    sum / N_LARGE as f64 * LARGE_SCALE
}

/// Per-neighbour signature of one bonded neighbour `m`: the common neighbour
/// set shared by the centre and `m`, the number of bonds inside that set, and
/// the longest bond chain in it.
///
/// The classifiers test the same triple a field at a time, so this whole-
/// triple form is what the unit tests use to check hand-built common sets.
#[cfg(test)]
struct PairBond {
    common: u16,
    bonds: u32,
    chain: u32,
}

/// The common set of neighbour `m`: atoms bonded to `m` that are themselves
/// bonded to the centre. `nb.bond[m]` is already restricted to `keep` when the
/// bond matrix was built, but mask again so the invariant is local and obvious.
#[inline]
fn common_of(nb: &Neighbourhood, keep: u16, m: usize) -> u16 {
    nb.bond[m] & keep & !(1 << m)
}

/// Bonds inside a common set, and the set's adjacency indexed by neighbour
/// slot (no dense re-index: every member is a bit of a `u16`). A pair `x < y`
/// is bonded when row `x` has bit `y`, as OVITO reads its bond list.
///
/// With [`max_chain`], this is the reference the classifiers' shortcut is
/// tested against (`fast_signatures_match_the_reference`).
#[cfg(test)]
fn common_bonds(nb: &Neighbourhood, common: u16) -> (u32, [u16; N_LARGE]) {
    let mut adj = [0u16; N_LARGE];
    let mut bonds = 0u32;
    let mut rest = common;
    while rest != 0 {
        let x = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        // `rest` now holds exactly the members above `x`.
        let up = nb.bond[x] & rest;
        bonds += up.count_ones();
        adj[x] |= up;
        let mut u = up;
        while u != 0 {
            let y = u.trailing_zeros() as usize;
            u &= u - 1;
            adj[y] |= 1 << x;
        }
    }
    (bonds, adj)
}

#[cfg(test)]
impl PairBond {
    fn compute(nb: &Neighbourhood, keep: u16, m: usize) -> Self {
        let common = common_of(nb, keep, m);
        if common == 0 {
            return PairBond {
                common,
                bonds: 0,
                chain: 0,
            };
        }
        let (bonds, adj) = common_bonds(nb, common);
        PairBond {
            common,
            bonds,
            chain: max_chain(&adj, common),
        }
    }

    /// `421`: 4 common neighbours holding two *disjoint* bonds, so the longest
    /// chain is a single bond. This is the FCC signature.
    fn is_421(&self) -> bool {
        self.common.count_ones() == 4 && self.bonds == 2 && self.chain == 1
    }

    /// `422`: 4 common neighbours holding two bonds that share an atom, so the
    /// longest chain is a 3-atom path. This is the HCP signature.
    fn is_422(&self) -> bool {
        self.common.count_ones() == 4 && self.bonds == 2 && self.chain == 2
    }

    /// `555`: 5 common neighbours forming one closed 5-ring. The ICO signature.
    fn is_555(&self) -> bool {
        self.common.count_ones() == 5 && self.bonds == 5 && self.chain == 5
    }

    /// `444`: 4 common neighbours forming one closed 4-ring.
    fn is_444(&self) -> bool {
        self.common.count_ones() == 4 && self.bonds == 4 && self.chain == 4
    }

    /// `666`: 6 common neighbours forming one closed 6-ring.
    fn is_666(&self) -> bool {
        self.common.count_ones() == 6 && self.bonds == 6 && self.chain == 6
    }
}

/// Twice the number of bonds inside `set`, counting each bond from both
/// ends: the bond matrix is exactly symmetric (see [`pair_d2`]).
#[inline]
fn bond_ends(nb: &Neighbourhood, set: u16) -> u32 {
    let mut ends = 0;
    let mut rest = set;
    while rest != 0 {
        let x = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        ends += (nb.bond[x] & set).count_ones();
    }
    ends
}

/// Whether every bond inside `common` lies in a single cluster.
///
/// A common set with `n` bonds has a longest chain of `n` exactly when this
/// holds: [`max_chain`] is the largest cluster's bond count, and no cluster can
/// hold more than all of them. So for `555`, `444` and `666`, whose chain must
/// equal their bond count, this replaces the chain search. The cluster of the
/// lowest bonded member is grown a frontier at a time; when it spans the whole
/// set (any ring does) the answer needs no count.
#[inline]
fn bonds_connected(nb: &Neighbourhood, common: u16, ends: u32) -> bool {
    let mut seed = common;
    while seed != 0 && nb.bond[seed.trailing_zeros() as usize] & common == 0 {
        seed &= seed - 1;
    }
    let mut comp = seed & seed.wrapping_neg();
    let mut frontier = comp;
    while frontier != 0 {
        let mut grow = 0u16;
        while frontier != 0 {
            let x = frontier.trailing_zeros() as usize;
            frontier &= frontier - 1;
            grow |= nb.bond[x] & common;
        }
        frontier = grow & !comp;
        comp |= frontier;
    }
    comp == common || bond_ends(nb, comp) == ends
}

/// OVITO `analyzeSmallSignature` (12-neighbour FCC/HCP/ICO).
///
/// Every one of the 12 neighbours must contribute an accepted `(common, bonds,
/// chain)` triple; the first mismatch aborts the whole atom to `OTHER`, exactly
/// as OVITO `break`s out of the loop. The triple is tested a field at a time,
/// so a neighbour that fails on its common count or bond count never pays for
/// the chain; the accepted set is exactly `421 | 422 | 555`.
fn analyze_small(nb: &Neighbourhood, keep: u16) -> SignatureCounts {
    let mut c = SignatureCounts::default();
    for m in 0..N_SMALL {
        if keep & (1 << m) == 0 {
            return SignatureCounts::default();
        }
        match small_signature(nb, keep, m) {
            Some(0) => c.n421 += 1,
            Some(1) => c.n422 += 1,
            Some(_) => c.n555 += 1,
            None => return SignatureCounts::default(),
        }
    }
    c
}

/// Which small signature neighbour `m` has: 0 for `421`, 1 for `422`, 2 for
/// `555`, `None` for anything else. With exactly 2 bonds among 4 the chain is
/// 2 when the bonds share a member (one of degree 2), else 1; both are taken.
#[inline]
fn small_signature(nb: &Neighbourhood, keep: u16, m: usize) -> Option<usize> {
    let common = common_of(nb, keep, m);
    match (common.count_ones(), bond_ends(nb, common)) {
        (4, 4) => {
            let mut rest = common;
            let mut shared = false;
            while rest != 0 {
                let x = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                shared |= (nb.bond[x] & common).count_ones() == 2;
            }
            Some(shared as usize)
        }
        (5, 10) if bonds_connected(nb, common, 10) => Some(2),
        _ => None,
    }
}

/// OVITO `analyzeLargeSignature` (14-neighbour BCC), with the same abort-on-
/// mismatch rule and field-at-a-time test as [`analyze_small`].
fn analyze_large(nb: &Neighbourhood, keep: u16) -> SignatureCounts {
    let mut c = SignatureCounts::default();
    for m in 0..N_LARGE {
        if keep & (1 << m) == 0 {
            return SignatureCounts::default();
        }
        match large_signature(nb, keep, m) {
            Some(3) => c.n444 += 1,
            Some(_) => c.n666 += 1,
            None => return SignatureCounts::default(),
        }
    }
    c
}

/// Which large signature neighbour `m` has: 3 for `444`, 4 for `666`, `None`
/// for anything else. Both need as many bonds, and as long a chain, as common
/// neighbours.
#[inline]
fn large_signature(nb: &Neighbourhood, keep: u16, m: usize) -> Option<usize> {
    let common = common_of(nb, keep, m);
    let nc = common.count_ones();
    if nc != 4 && nc != 6 {
        return None;
    }
    let ends = bond_ends(nb, common);
    if ends != 2 * nc || !bonds_connected(nb, common, ends) {
        return None;
    }
    Some(if nc == 4 { 3 } else { 4 })
}

/// Longest continuous bond chain inside a common set, counted in bonds (a
/// single dimer is 1, a 3-atom path is 2, a closed n-ring is n).
///
/// This mirrors OVITO's `calcMaxChainLength`: group the bonds into connected
/// clusters and take the largest cluster's edge count. `adj` is the slot-indexed
/// adjacency of [`common_bonds`] and `members` the common set; each component
/// is grown a frontier at a time as a bitmask.
#[cfg(test)]
fn max_chain(adj: &[u16; N_LARGE], members: u16) -> u32 {
    let mut best = 0u32;
    let mut rest = members;
    while rest != 0 {
        let s = rest.trailing_zeros();
        let mut comp = 1u16 << s;
        let mut frontier = comp;
        while frontier != 0 {
            let mut grow = 0u16;
            while frontier != 0 {
                let x = frontier.trailing_zeros() as usize;
                frontier &= frontier - 1;
                grow |= adj[x];
            }
            frontier = grow & !comp;
            comp |= frontier;
        }
        rest &= !comp;
        // Count the edges inside the component; `adj` is symmetric, so halve.
        let mut edges = 0u32;
        let mut bits = comp;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            edges += (adj[i] & comp).count_ones();
        }
        best = best.max(edges / 2);
    }
    best
}

/// Per-neighbour signature tally used by the unit tests to exercise `PairBond`
/// on small hand-built common sets (the real classifier is `analyze_small` /
/// `analyze_large`, which additionally enforce OVITO's abort-on-mismatch rule).
#[cfg(test)]
fn signature_scan(nb: &Neighbourhood, keep: u16) -> SignatureCounts {
    let mut c = SignatureCounts::default();
    for m in 0..nb.len {
        if keep & (1 << m) == 0 {
            continue;
        }
        let p = PairBond::compute(nb, keep, m);
        if p.is_421() {
            c.n421 += 1;
        } else if p.is_422() {
            c.n422 += 1;
        } else if p.is_555() {
            c.n555 += 1;
        } else if p.is_444() {
            c.n444 += 1;
        } else if p.is_666() {
            c.n666 += 1;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Interval CNA (OVITO `determineStructureInterval`, Larsen 2020).
// ---------------------------------------------------------------------------

/// OVITO's `const FloatType x = 2.0f / 3.0f`: a *float* division widened to
/// double, so the threshold fraction is not exactly two thirds of the way.
const INTERVAL_X: f64 = (2.0f32 / 3.0f32) as f64;

const EDGE_SHORT: u8 = 1;
const EDGE_LONG: u8 = 2;

#[derive(Clone, Copy)]
struct GraphEdge {
    i: u8,
    j: u8,
    length: f64,
    kind: u8,
}

/// Interval CNA for one atom: OVITO's `EdgeIterator` and the two interval
/// walks, verbatim. `nb` holds the (up to) 14 nearest neighbours.
///
/// Neighbour-neighbour bonds are added shortest first along two paths: the
/// short path (bonds among the 12 nearest shorter than the short threshold,
/// then the shortest bond that is not) and the long path (bonds among all 14).
/// Whenever the bond graph has FCC/HCP/ICO (all 12 of coordination 4, or all
/// of 5) or BCC (6 of 4 and 8 of 6) coordination, the ordinary signature is
/// tested, and the valid structure with the widest gap to the next bond wins.
fn classify_interval(nb: &mut Neighbourhood) -> (u8, SignatureCounts) {
    let analyze_short = nb.len >= N_SMALL;
    let analyze_long = nb.len >= N_LARGE;
    let nn = if analyze_long {
        N_LARGE
    } else if analyze_short {
        N_SMALL
    } else {
        return (OTHER, SignatureCounts::default());
    };

    let mut lengths = [0.0f64; N_LARGE];
    for (t, l) in lengths.iter_mut().enumerate().take(nn) {
        let (x, y, z) = (nb.dr[0][t], nb.dr[1][t], nb.dr[2][t]);
        *l = (x * x + y * y + z * z).sqrt();
    }
    let fraction = (1.0 - INTERVAL_X) * 1.0 + INTERVAL_X * 2f64.sqrt();
    let mut short_threshold = 0.0;
    let mut long_threshold = 0.0;
    if analyze_short {
        let mut s = 0.0;
        for l in &lengths[..N_SMALL] {
            s += l;
        }
        s /= N_SMALL as f64;
        short_threshold = fraction * s;
    }
    if analyze_long {
        let w = 0.75f64.sqrt();
        let mut s = 0.0;
        for l in &lengths[..8] {
            s += l / w;
        }
        for l in &lengths[8..N_LARGE] {
            s += l;
        }
        s /= N_LARGE as f64;
        long_threshold = fraction * s;
    }

    // EdgeIterator
    if nn < N_SMALL {
        short_threshold = 0.0;
    }
    if nn < N_LARGE {
        long_threshold = 0.0;
    }
    let pair = *nb.pair.get_or_insert_with(|| pair_d2(&nb.dr));
    let mut edges = [GraphEdge {
        i: 0,
        j: 0,
        length: 0.0,
        kind: 0,
    }; N_LARGE * (N_LARGE - 1) / 2 + 2];
    let mut n_edges = 0usize;
    let mut short_end = GraphEdge {
        i: u8::MAX,
        j: u8::MAX,
        length: f64::INFINITY,
        kind: EDGE_SHORT,
    };
    let mut long_end = GraphEdge {
        i: u8::MAX,
        j: u8::MAX,
        length: f64::INFINITY,
        kind: EDGE_LONG,
    };
    for (i, row) in pair.iter().enumerate().take(nn) {
        for (j, &d2) in row.iter().enumerate().take(nn).skip(i + 1) {
            let length = d2.sqrt();
            let mut kind = 0u8;
            if i < N_SMALL && j < N_SMALL && length < short_threshold {
                kind |= EDGE_SHORT;
            }
            if length < long_threshold {
                kind |= EDGE_LONG;
            }
            let e = GraphEdge {
                i: i as u8,
                j: j as u8,
                length,
                kind,
            };
            if kind == 0 {
                // Verbatim: a candidate is weighed against the long end first.
                if length < long_end.length {
                    long_end = GraphEdge {
                        kind: EDGE_LONG,
                        ..e
                    };
                } else if length < short_end.length {
                    short_end = GraphEdge {
                        kind: EDGE_SHORT,
                        ..e
                    };
                }
            } else {
                edges[n_edges] = e;
                n_edges += 1;
            }
        }
    }
    // Ties do not matter: a gap of zero never beats the best width, and the
    // edge set at the end of a run of equal lengths is the same in any order.
    edges[..n_edges].sort_by(|a, b| {
        a.length
            .partial_cmp(&b.length)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if short_end.i != u8::MAX {
        edges[n_edges] = short_end;
        n_edges += 1;
    }
    if long_end.i != u8::MAX {
        edges[n_edges] = long_end;
        n_edges += 1;
    }
    let edges = &edges[..n_edges];

    let mut best_width = 0.0f64;
    let mut best = (OTHER, SignatureCounts::default());

    if analyze_short {
        nb.bond = [0; N_LARGE];
        let mut coord = [0u32; N_LARGE];
        let (mut n4, mut n5) = (0usize, 0usize);
        let mut path = edges.iter().filter(|e| e.kind & EDGE_SHORT != 0).peekable();
        while let (Some(edge), Some(next)) = (path.next(), path.peek()) {
            let (i, j) = (edge.i as usize, edge.j as usize);
            coord[i] += 1;
            coord[j] += 1;
            nb.bond[i] |= 1 << j;
            nb.bond[j] |= 1 << i;
            let mut over = false;
            for v in [i, j] {
                if coord[v] == 4 {
                    n4 += 1;
                }
                if coord[v] == 5 {
                    n4 -= 1;
                    n5 += 1;
                }
                if coord[v] > 5 {
                    over = true;
                    break;
                }
            }
            if over {
                break;
            }
            if n4 == N_SMALL || n5 == N_SMALL {
                let c = analyze_small(nb, SMALL_KEEP);
                let t = c.classify();
                if t != OTHER {
                    let width = next.length - edge.length;
                    if width > best_width {
                        best_width = width;
                        best = (t, c);
                    }
                }
            }
        }
    }

    if analyze_long {
        nb.bond = [0; N_LARGE];
        let mut coord = [0u32; N_LARGE];
        let (mut n4, mut n6) = (0usize, 0usize);
        let mut path = edges.iter().filter(|e| e.kind & EDGE_LONG != 0).peekable();
        while let (Some(edge), Some(next)) = (path.next(), path.peek()) {
            let (i, j) = (edge.i as usize, edge.j as usize);
            coord[i] += 1;
            coord[j] += 1;
            nb.bond[i] |= 1 << j;
            nb.bond[j] |= 1 << i;
            let mut over = false;
            for v in [i, j] {
                if coord[v] == 4 {
                    n4 += 1;
                }
                if coord[v] == 5 {
                    n4 -= 1;
                }
                if coord[v] == 6 {
                    n6 += 1;
                }
                if coord[v] > 6 {
                    over = true;
                    break;
                }
            }
            if over {
                break;
            }
            if n4 == 6 && n6 == 8 {
                let c = analyze_large(nb, LARGE_KEEP);
                let t = c.classify();
                if t != OTHER {
                    let width = next.length - edge.length;
                    if width > best_width {
                        best_width = width;
                        best = (t, c);
                    }
                }
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Bond-based CNA (OVITO `BondCNAAlgorithm`).
// ---------------------------------------------------------------------------

/// Full bond list as CSR: `nbr[off[i]..off[i + 1]]` are atom `i`'s bonds,
/// each `(j, image shift of j relative to i)`.
struct BondList {
    off: Vec<usize>,
    nbr: Vec<(u32, [i32; 3])>,
}

impl BondList {
    #[inline]
    fn of(&self, i: usize) -> &[(u32, [i32; 3])] {
        &self.nbr[self.off[i]..self.off[i + 1]]
    }
}

#[inline]
fn add_shift(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Largest connected cluster of bonds, counted in bonds (OVITO
/// `calcMaxChainLength`). `bonds` are pairs of member bits.
fn max_chain_bonds(bonds: &[u64]) -> u32 {
    let mut used = vec![false; bonds.len()];
    let mut best = 0u32;
    for s in 0..bonds.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let mut atoms = bonds[s];
        let mut size = 1u32;
        loop {
            let mut grew = false;
            for (b, u) in bonds.iter().zip(used.iter_mut()) {
                if !*u && b & atoms != 0 {
                    *u = true;
                    atoms |= b;
                    size += 1;
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        best = best.max(size);
    }
    best
}

/// `(common neighbours, bonds among them, longest chain)` of bond `i -> j`
/// at image `s`, with OVITO's limits (32 common neighbours, 64 bonds).
fn bond_signature(bl: &BondList, i: usize, j: usize, s: [i32; 3]) -> Result<(u32, u32, u32)> {
    let mut common = [(0u32, [0i32; 3]); 32];
    let mut nc = 0usize;
    for &(k, s1) in bl.of(i) {
        for &(k2, s2) in bl.of(j) {
            if k2 == k && s1 == add_shift(s, s2) {
                if nc == common.len() {
                    return Err(crate::error::Error::Analysis(
                        "Two of the particles have more than 32 common neighbors, which is the \
                         built-in limit. Cannot perform CNA in this case."
                            .into(),
                    ));
                }
                common[nc] = (k, s1);
                nc += 1;
                break;
            }
        }
    }
    let mut bonds = [0u64; 64];
    let mut nb = 0usize;
    for ni1 in 0..nc {
        let (k1, s1) = common[ni1];
        for &(kk, ss) in bl.of(k1 as usize) {
            for (ni2, &(k2, s2)) in common[..ni1].iter().enumerate() {
                if k2 == kk && add_shift(s1, ss) == s2 {
                    if nb == bonds.len() {
                        return Err(crate::error::Error::Analysis(
                            "There are more than 64 bonds between common neighbors, which is \
                             the built-in limit. Cannot perform CNA in this case."
                                .into(),
                        ));
                    }
                    bonds[nb] = (1u64 << ni1) | (1u64 << ni2);
                    nb += 1;
                    break;
                }
            }
        }
    }
    Ok((nc as u32, nb as u32, max_chain_bonds(&bonds[..nb])))
}

impl Cna {
    /// Bond-based CNA over every bond within `cutoff`.
    fn run_bond(
        &self,
        ctx: &AnalysisContext,
        cutoff: f64,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let n = ctx.frame.natoms();
        let positions = &ctx.frame.particles.positions[..n];
        let bl = {
            let _t = crate::prof::StageTimer::start("cna.bonds");
            let cl =
                crate::neighbor::cell_list::CellList::build(&ctx.frame.cell, positions, cutoff)?;
            let mut lists: Vec<Vec<(u32, [i32; 3])>> = vec![Vec::new(); n];
            ctx.par.try_fill_with(
                &mut lists,
                Vec::new,
                |buf: &mut Vec<(usize, crate::cell::Coord3, [i64; 3])>, i, o| -> Result<()> {
                    cl.neighbor_entries(i, cutoff, buf)?;
                    o.extend(
                        buf.iter()
                            .map(|&(j, _, s)| (j as u32, [s[0] as i32, s[1] as i32, s[2] as i32])),
                    );
                    Ok(())
                },
            )?;
            let mut off = Vec::with_capacity(n + 1);
            off.push(0usize);
            let mut nbr = Vec::with_capacity(lists.iter().map(Vec::len).sum());
            for l in lists {
                nbr.extend_from_slice(&l);
                off.push(nbr.len());
            }
            BondList { off, nbr }
        };
        let _t = crate::prof::StageTimer::start("cna.scan");
        let mut res = vec![(OTHER, SignatureCounts::default()); n];
        ctx.par.try_fill_with(
            &mut res,
            || (),
            |_, i, o| -> Result<()> {
                let mut c = SignatureCounts::default();
                let mut total = 0u32;
                for &(j, s) in bl.of(i) {
                    let sig = bond_signature(&bl, i, j as usize, s)?;
                    match sig {
                        (4, 2, 1) => c.n421 += 1,
                        (4, 2, 2) => c.n422 += 1,
                        (4, 4, 4) => c.n444 += 1,
                        (5, 5, 5) => c.n555 += 1,
                        (6, 6, 6) => c.n666 += 1,
                        (4, _, _) => {}
                        _ => return Ok(()),
                    }
                    total += 1;
                }
                let t = if c.n421 == 12 && total == 12 {
                    FCC
                } else if c.n421 == 6 && c.n422 == 6 && total == 12 {
                    HCP
                } else if c.n444 == 6 && c.n666 == 8 && total == 14 {
                    BCC
                } else if c.n555 == 12 && total == 12 {
                    ICO
                } else {
                    OTHER
                };
                *o = (t, c);
                Ok(())
            },
        )?;
        drop(_t);
        let types: Vec<u8> = res.iter().map(|r| r.0).collect();
        let counts: Vec<SignatureCounts> = if self.full {
            res.iter().map(|r| r.1).collect()
        } else {
            Vec::new()
        };
        summary.merge_from(Summary::scalar("cna_bonds", bl.nbr.len() as f64 / 2.0));
        self.finish_output(types, counts, out, summary);
        Ok(())
    }

    /// Histogram, `StructureType` and the optional counters (shared tail of
    /// the bond-based mode; the k-NN modes write the same in `run`).
    fn finish_output(
        &self,
        types: Vec<u8>,
        counts: Vec<SignatureCounts>,
        out: &mut PropertyStore,
        summary: &mut Summary,
    ) {
        let mut hist = [0usize; 5];
        for &t in &types {
            hist[t as usize] += 1;
        }
        for code in [OTHER, FCC, HCP, BCC, ICO] {
            summary.merge_from(Summary::scalar(
                &format!("cna_{}", code_name(code)),
                hist[code as usize] as f64,
            ));
        }
        out.register_scalar(crate::data::property::DType::U8, "StructureType", types);
        if self.full {
            const NAMES: [&str; 5] = ["cna_n421", "cna_n422", "cna_n555", "cna_n444", "cna_n666"];
            for (comp, name) in NAMES.iter().enumerate() {
                let v: Vec<i32> = counts.iter().map(|c| c.as_array()[comp] as i32).collect();
                out.register_scalar(crate::data::property::DType::I32, name, v);
                let total: i64 = counts.iter().map(|c| c.as_array()[comp] as i64).sum();
                summary.merge_from(Summary::scalar(&format!("{name}_total"), total as f64));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Identify diamond structure (OVITO `IdentifyDiamondModifier`).
// ---------------------------------------------------------------------------

/// The 4 nearest neighbours of one atom: index (`u32::MAX` when missing) and
/// displacement.
type Four = [(u32, [f64; 3]); 4];

/// OVITO `Vector3::isZero(1e-2f)`: every component within the float `1e-2`.
#[inline]
fn is_zero_1e2(v: [f64; 3]) -> bool {
    const TOL: f64 = 1e-2f32 as f64;
    v[0].abs() <= TOL && v[1].abs() <= TOL && v[2].abs() <= TOL
}

/// Stage 2 for one atom: cubic/hexagonal diamond from the FCC/HCP signature
/// of its 12 second neighbours, else `OTHER`.
fn diamond_core(first: &[Four], i: usize) -> u8 {
    let nl = &first[i];
    if nl.iter().any(|e| e.0 == u32::MAX) {
        return diamond::OTHER;
    }
    let mut nb = Neighbourhood::new();
    let mut m = 0usize;
    for (ni, &(j, v0)) in nl.iter().enumerate() {
        for &(_, w) in first[j as usize].iter() {
            // A missing neighbour of a neighbour is a zero vector upstream, so
            // `v0 + 0` is accepted: reproduced.
            let v = [v0[0] + w[0], v0[1] + w[1], v0[2] + w[2]];
            if is_zero_1e2(v) {
                continue;
            }
            if m == N_SMALL {
                return diamond::OTHER;
            }
            nb.dr[0][m] = v[0];
            nb.dr[1][m] = v[1];
            nb.dr[2][m] = v[2];
            m += 1;
        }
        if m != ni * 3 + 3 {
            return diamond::OTHER;
        }
    }
    nb.len = N_SMALL;
    let mut sum = 0.0;
    for t in 0..N_SMALL {
        let (x, y, z) = (nb.dr[0][t], nb.dr[1][t], nb.dr[2][t]);
        sum += (x * x + y * y + z * z).sqrt();
    }
    sum /= N_SMALL as f64;
    // `FloatType(1.2071068)`: sqrt(2) * (1 + sqrt(0.5)) / 2, rounded.
    let cutoff = sum * 1.2071068;
    build_bonds(&mut nb, SMALL_KEEP, cutoff);
    let (mut n421, mut n422) = (0, 0);
    for k in 0..N_SMALL {
        match small_signature(&nb, SMALL_KEEP, k) {
            Some(0) => n421 += 1,
            Some(1) => n422 += 1,
            _ => break,
        }
    }
    if n421 == 12 {
        diamond::CUBIC
    } else if n421 == 6 && n422 == 6 {
        diamond::HEX
    } else {
        diamond::OTHER
    }
}

/// Identify-diamond over a frame: per-atom [`diamond`] codes.
pub fn identify_diamond(
    cell: &crate::cell::UnitCell,
    positions: &[crate::cell::Coord3],
    par: &crate::exec::Par,
) -> Result<Vec<u8>> {
    let n = positions.len();
    let index = {
        let _t = crate::prof::StageTimer::start("cna.knn_build");
        KnnIndex::build(cell, positions, par)?
    };
    let _t = crate::prof::StageTimer::start("cna.scan");
    let mut first: Vec<Four> = vec![[(u32::MAX, [0.0; 3]); 4]; n];
    par.try_fill_with(
        &mut first,
        || (KnnScratch::default(), Vec::with_capacity(4)),
        |(scratch, entries): &mut (KnnScratch, Vec<_>), i, o| {
            index.knn(i, positions[i], 4, scratch, entries);
            for (slot, &(j, dr, _)) in o.iter_mut().zip(entries.iter()) {
                *slot = (j, dr);
            }
            Ok::<(), std::convert::Infallible>(())
        },
    )
    .unwrap_or_else(|e| match e {});
    drop(index);
    let mut types = vec![diamond::OTHER; n];
    par.try_fill_with(
        &mut types,
        || (),
        |_, i, o| {
            *o = diamond_core(&first, i);
            Ok::<(), std::convert::Infallible>(())
        },
    )
    .unwrap_or_else(|e| match e {});
    // First and second neighbours of the diamond atoms, in place and in index
    // order, so a shared neighbour takes the type of the lower-indexed seed.
    for i in 0..n {
        let t = types[i];
        if t == diamond::CUBIC || t == diamond::HEX {
            let mark = if t == diamond::CUBIC {
                diamond::CUBIC_FIRST
            } else {
                diamond::HEX_FIRST
            };
            for &(j, _) in &first[i] {
                if j != u32::MAX && types[j as usize] == diamond::OTHER {
                    types[j as usize] = mark;
                }
            }
        }
    }
    for i in 0..n {
        let t = types[i];
        if t == diamond::CUBIC_FIRST || t == diamond::HEX_FIRST {
            let mark = if t == diamond::CUBIC_FIRST {
                diamond::CUBIC_SECOND
            } else {
                diamond::HEX_SECOND
            };
            for &(j, _) in &first[i] {
                if j != u32::MAX && types[j as usize] == diamond::OTHER {
                    types[j as usize] = mark;
                }
            }
        }
    }
    Ok(types)
}

fn run_diamond(
    ctx: &AnalysisContext,
    out: &mut PropertyStore,
    summary: &mut Summary,
) -> Result<()> {
    let n = ctx.frame.natoms();
    let types = identify_diamond(
        &ctx.frame.cell,
        &ctx.frame.particles.positions[..n],
        ctx.par,
    )?;
    let mut hist = [0usize; 7];
    for &t in &types {
        hist[t as usize] += 1;
    }
    for (code, name) in diamond::NAMES.iter().enumerate() {
        summary.merge_from(Summary::scalar(&format!("cna_{name}"), hist[code] as f64));
    }
    out.register_scalar(crate::data::property::DType::U8, "StructureType", types);
    Ok(())
}

fn code_name(code: u8) -> &'static str {
    match code {
        FCC => "fcc",
        HCP => "hcp",
        BCC => "bcc",
        ICO => "ico",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(n421: u32, n422: u32, n555: u32, n444: u32, n666: u32) -> SignatureCounts {
        SignatureCounts {
            n421,
            n422,
            n555,
            n444,
            n666,
        }
    }

    #[test]
    fn classify_signatures() {
        assert_eq!(c(12, 0, 0, 0, 0).classify(), FCC);
        assert_eq!(c(6, 6, 0, 0, 0).classify(), HCP);
        assert_eq!(c(0, 0, 12, 0, 0).classify(), ICO);
        assert_eq!(c(0, 0, 0, 6, 8).classify(), BCC);
        assert_eq!(c(0, 0, 0, 0, 0).classify(), OTHER);
        // near-misses stay OTHER
        assert_eq!(c(11, 0, 0, 0, 0).classify(), OTHER);
        assert_eq!(c(6, 5, 0, 0, 0).classify(), OTHER);
        assert_eq!(c(0, 0, 0, 6, 7).classify(), OTHER);
    }

    /// End-to-end on hand-built lattices: ideal FCC, HCP, BCC and an SC cube
    /// (which is legitimately OTHER for CNA). This exercises the real
    /// neighbour search, the bond matrix and the cutoff modes together, which
    /// the bitmask tests above deliberately do not.
    pub mod lattices {
        use super::*;
        use crate::analysis::NeighborhoodContext;
        use crate::cell::UnitCell;
        use crate::data::frame::Frame;
        use crate::data::particle::ParticleData;
        use crate::exec::Par;

        /// Build a frame from raw positions, run CNA, return the per-atom types.
        pub fn run(cell: &UnitCell, pos: &[[f64; 3]], mode: CnaMode) -> Vec<u8> {
            let mut p = ParticleData::with_capacity(pos.len());
            for (i, q) in pos.iter().enumerate() {
                p.add_atom(*q, i as i64 + 1, 1);
            }
            let frame = Frame::new(*cell, p);
            // CNA builds its own k-NN index; no shared cell list is needed.
            let nk = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame: &frame,
                nkeys: &nk,
                par: &Par::new(1),
            };
            let mut out = PropertyStore::new();
            let mut summary = Summary::default();
            Cna::new(mode).run(&ctx, &mut out, &mut summary).unwrap();
            out.get("StructureType").unwrap().as_u8().unwrap().clone()
        }

        fn run_file(frame: &Frame, mode: CnaMode) -> Vec<u8> {
            let nkeys = NeighborhoodContext { cell: None };
            let ctx = AnalysisContext {
                frame,
                nkeys: &nkeys,
                par: &Par::new(1),
            };
            let mut out = PropertyStore::new();
            let mut summary = Summary::default();
            Cna::new(mode).run(&ctx, &mut out, &mut summary).unwrap();
            out.get("StructureType").unwrap().as_u8().unwrap().clone()
        }

        /// Conventional-cubic FCC lattice, 2x2x2 cells.
        pub fn fcc(a: f64) -> (UnitCell, Vec<[f64; 3]>) {
            let basis = [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ];
            let mut pos = Vec::new();
            for ix in 0..2 {
                for iy in 0..2 {
                    for iz in 0..2 {
                        for b in &basis {
                            pos.push([
                                (ix as f64 + b[0]) * a,
                                (iy as f64 + b[1]) * a,
                                (iz as f64 + b[2]) * a,
                            ]);
                        }
                    }
                }
            }
            (UnitCell::orthorhombic(2.0 * a, 2.0 * a, 2.0 * a), pos)
        }

        /// Conventional-cubic BCC lattice, 2x2x2 cells.
        pub fn bcc(a: f64) -> (UnitCell, Vec<[f64; 3]>) {
            let mut pos = Vec::new();
            for ix in 0..2 {
                for iy in 0..2 {
                    for iz in 0..2 {
                        for b in [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] {
                            pos.push([
                                (ix as f64 + b[0]) * a,
                                (iy as f64 + b[1]) * a,
                                (iz as f64 + b[2]) * a,
                            ]);
                        }
                    }
                }
            }
            (UnitCell::orthorhombic(2.0 * a, 2.0 * a, 2.0 * a), pos)
        }

        /// Simple cubic, 2x2x2 cells.
        pub fn sc(a: f64) -> (UnitCell, Vec<[f64; 3]>) {
            let mut pos = Vec::new();
            for ix in 0..2 {
                for iy in 0..2 {
                    for iz in 0..2 {
                        pos.push([ix as f64 * a, iy as f64 * a, iz as f64 * a]);
                    }
                }
            }
            (UnitCell::orthorhombic(2.0 * a, 2.0 * a, 2.0 * a), pos)
        }

        #[test]
        fn ideal_fcc_is_fcc_in_both_modes() {
            let (cell, pos) = fcc(3.615);
            for mode in [CnaMode::Adaptive, CnaMode::Fixed { cutoff: 3.5 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == FCC), "mode {mode:?} gave {t:?}");
            }
        }

        #[test]
        fn ideal_bcc_is_bcc_in_both_modes() {
            let (cell, pos) = bcc(2.866);
            for mode in [CnaMode::Adaptive, CnaMode::Fixed { cutoff: 3.5 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == BCC), "mode {mode:?} gave {t:?}");
            }
        }

        /// SC has 6 neighbours, so no 12/14-neighbour signature can be
        /// saturated: every atom must be OTHER. This is a real check that we
        /// are not simply pattern-matching on coordination number.
        #[test]
        fn simple_cubic_is_other() {
            let (cell, pos) = sc(2.98);
            for mode in [CnaMode::Adaptive, CnaMode::Fixed { cutoff: 3.5 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == OTHER), "mode {mode:?} gave {t:?}");
            }
        }

        /// Ideal HCP: 3x3x3 hexagonal cells with ABAB stacking.
        ///
        /// Built through `from_general_triclinic`, the same path the LAMMPS
        /// readers use, so the test also covers triclinic cell arithmetic.
        ///
        /// 3x3x3 is required, not 2x2x2: the primitive hexagonal cell is only
        /// 2 atoms and half its cell height, so a 2x2x2 supercell leaves each
        /// atom with periodic images of itself at distance ~0. The k-NN query then
        /// fills its 14 slots with zero-distance ghosts and the 12/14 gate
        /// rejects the atom. `ideal_hcp_from_fixture_is_hcp_in_both_modes`
        /// covers the OVITO-verified 2x2x2 data independently.
        pub fn hcp(a: f64, c: f64) -> (UnitCell, Vec<[f64; 3]>) {
            let avec = [3.0 * a, 0.0, 0.0];
            let bvec = [-3.0 * a / 2.0, 3.0 * a * 3f64.sqrt() / 2.0, 0.0];
            let cvec = [0.0, 0.0, 3.0 * c];
            let (cell, _rot) =
                UnitCell::from_general_triclinic_rot(avec, bvec, cvec, [0.0; 3]).unwrap();
            // frac_to_cart via the restricted cell, so positions match what the
            // reader would produce. The cell spans three primitive cells per
            // axis, so each integer cell offset maps to a third of the cell.
            let h = cell.h_matrix();
            // two basis atoms: (0,0,0) and the 2/3,1/3,1/2 site
            let sites = [[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]];
            let mut pos = Vec::new();
            for ix in 0..3 {
                for iy in 0..3 {
                    for iz in 0..3 {
                        for s in &sites {
                            let f = [
                                (ix as f64 + s[0]) / 3.0,
                                (iy as f64 + s[1]) / 3.0,
                                (iz as f64 + s[2]) / 3.0,
                            ];
                            pos.push(frac_to_cart(h, f));
                        }
                    }
                }
            }
            (cell, pos)
        }

        /// Fractional to Cartesian. `Mat3` stores cell vectors as *rows*, so
        /// this is `f0*row0 + f1*row1 + f2*row2` — reading columns instead
        /// shears the lattice and silently yields zero HCP atoms.
        pub fn frac_to_cart(h: &crate::cell::Mat3, f: [f64; 3]) -> [f64; 3] {
            [
                f[0] * h.m[0][0] + f[1] * h.m[1][0] + f[2] * h.m[2][0],
                f[0] * h.m[0][1] + f[1] * h.m[1][1] + f[2] * h.m[2][1],
                f[0] * h.m[0][2] + f[1] * h.m[1][2] + f[2] * h.m[2][2],
            ]
        }

        /// HCP from the OVITO-verified fixture, which uses a 2x2x2 primitive
        /// cell. The hand-built `hcp()` lattice is unusable here: at 2x2x2 its
        /// own periodic images land within ~0 of the reference atom, so
        /// the k-NN query fills its 14 slots with zero-distance ghosts.
        #[test]
        fn ideal_hcp_from_fixture_is_hcp_in_both_modes() {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/data/hcp_ortho.xyz");
            let frame = crate::io::xyz::XyzReader::read_first(&path).unwrap();
            for mode in [CnaMode::Adaptive, CnaMode::Fixed { cutoff: 3.5 }] {
                let t = run_file(&frame, mode);
                assert!(t.iter().all(|&x| x == HCP), "mode {mode:?} gave {t:?}");
            }
        }

        #[test]
        fn ideal_hcp_is_hcp_in_both_modes() {
            let (cell, pos) = hcp(3.209, 5.211);
            for mode in [CnaMode::Adaptive, CnaMode::Fixed { cutoff: 3.5 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == HCP), "mode {mode:?} gave {t:?}");
            }
        }

        /// The interval and bond-based modes on the ideal lattices.
        #[test]
        fn ideal_lattices_under_interval_and_bond_modes() {
            let (cell, pos) = fcc(3.615);
            for mode in [CnaMode::Interval, CnaMode::Bond { cutoff: 3.0 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == FCC), "fcc {mode:?} gave {t:?}");
            }
            let (cell, pos) = bcc(2.866);
            for mode in [CnaMode::Interval, CnaMode::Bond { cutoff: 3.2 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == BCC), "bcc {mode:?} gave {t:?}");
            }
            let (cell, pos) = hcp(3.209, 5.211);
            for mode in [CnaMode::Interval, CnaMode::Bond { cutoff: 3.6 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == HCP), "hcp {mode:?} gave {t:?}");
            }
            let (cell, pos) = sc(2.98);
            for mode in [CnaMode::Interval, CnaMode::Bond { cutoff: 3.2 }] {
                let t = run(&cell, &pos, mode);
                assert!(t.iter().all(|&x| x == OTHER), "sc {mode:?} gave {t:?}");
            }
        }

        /// The bond-based mode counts every bond: a cutoff that takes in the
        /// FCC second shell leaves no atom with a 12-bond signature.
        #[test]
        fn bond_mode_has_no_coordination_gate() {
            let (cell, pos) = fcc(3.615);
            let t = run(&cell, &pos, CnaMode::Bond { cutoff: 3.7 });
            assert!(t.iter().all(|&x| x == OTHER), "{t:?}");
        }

        /// Cubic diamond, 2x2x2 conventional cells.
        fn dia(a: f64) -> (UnitCell, Vec<[f64; 3]>) {
            let (cell, fcc_pos) = fcc(a);
            let mut pos = fcc_pos.clone();
            pos.extend(
                fcc_pos
                    .iter()
                    .map(|p| [p[0] + a / 4.0, p[1] + a / 4.0, p[2] + a / 4.0]),
            );
            (cell, pos)
        }

        /// Identify-diamond: ideal cubic diamond is all cubic diamond, a
        /// vacancy turns its four neighbours into first neighbours (and
        /// their neighbours into second neighbours), and FCC is OTHER.
        #[test]
        fn identify_diamond_on_ideal_and_defective_diamond() {
            let (cell, pos) = dia(5.431);
            let t = run(&cell, &pos, CnaMode::Diamond);
            assert!(t.iter().all(|&x| x == diamond::CUBIC), "{t:?}");
            let t = run(&cell, &pos[1..], CnaMode::Diamond);
            let n = |c| t.iter().filter(|&&x| x == c).count();
            assert_eq!(n(diamond::OTHER), 0, "{t:?}");
            assert!(n(diamond::CUBIC_FIRST) >= 4, "{t:?}");
            assert!(n(diamond::CUBIC) > 0);
            let (cell, pos) = fcc(3.615);
            let t = run(&cell, &pos, CnaMode::Diamond);
            assert!(t.iter().all(|&x| x == diamond::OTHER), "{t:?}");
        }

        /// A vacancy in FCC must flip its neighbours away from FCC, which
        /// proves the signature scan is actually local, not a global guess.
        #[test]
        fn fcc_vacancy_breaks_local_symmetry() {
            let (cell, pos) = fcc(3.615);
            let with_hole = &pos[..pos.len() - 1];
            let t = run(&cell, with_hole, CnaMode::Adaptive);
            assert!(
                t.iter().any(|&x| x == OTHER),
                "a vacancy must leave at least one non-FCC atom, got {t:?}"
            );
        }

        /// CNA is deterministic: the same input must give byte-identical
        /// output regardless of thread count (design §10.2).
        #[test]
        fn results_are_thread_count_independent() {
            fn run_threads(cell: &UnitCell, pos: &[[f64; 3]], threads: usize) -> Vec<u8> {
                let mut p = ParticleData::with_capacity(pos.len());
                for (i, q) in pos.iter().enumerate() {
                    p.add_atom(*q, i as i64 + 1, 1);
                }
                let frame = Frame::new(*cell, p);
                let nk = NeighborhoodContext { cell: None };
                let ctx = AnalysisContext {
                    frame: &frame,
                    nkeys: &nk,
                    par: &Par::new(threads),
                };
                let mut out = PropertyStore::new();
                let mut summary = Summary::default();
                Cna::new(CnaMode::Adaptive)
                    .run(&ctx, &mut out, &mut summary)
                    .unwrap();
                out.get("StructureType").unwrap().as_u8().unwrap().clone()
            }
            let (cell, pos) = fcc(3.615);
            let one = run_threads(&cell, &pos, 1);
            for t in [2, 4, 8] {
                assert_eq!(run_threads(&cell, &pos, t), one, "threads={t} differ");
            }
        }
    }

    #[test]
    fn code_names_are_distinct() {
        let names: Vec<_> = [OTHER, FCC, HCP, BCC, ICO]
            .iter()
            .map(|&x| code_name(x))
            .collect();
        let mut s = names.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), names.len());
    }

    /// 421: 4 common neighbours holding 2 *disjoint* bonds, so every one of
    /// them has degree 1. This is the FCC signature.
    #[test]
    fn disjoint_dimers_are_421() {
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        let keep = 0b1111_1111;
        // neighbour 0 shares {1,2,3,4} with the centre
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        nb.bond[0] = common;
        // two separate dimers inside the common set
        nb.bond[1] = 1 << 3;
        nb.bond[3] = 1 << 1;
        nb.bond[2] = 1 << 4;
        nb.bond[4] = 1 << 2;
        let p = PairBond::compute(&nb, keep, 0);
        assert!(
            p.is_421(),
            "common={} bonds={} chain={}",
            p.common.count_ones(),
            p.bonds,
            p.chain
        );
        // `421` and `422` differ only in chain length (exactly 1 vs exactly 2),
        // so they are mutually exclusive.
        let c = signature_scan(&nb, keep);
        assert_eq!(c.n421, 1, "{c:?}");
        assert_eq!(c.n422, 0, "421 must not be double-counted: {c:?}");
    }

    /// 422: the same 4 common neighbours and the same 2 bonds, but they share
    /// an atom, so one has degree 2 and one is isolated. This is the HCP
    /// signature, and it is the *only* thing separating it from FCC.
    #[test]
    fn shared_chain_is_422() {
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        let keep = 0b1111_1111;
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        nb.bond[0] = common;
        // 1-2-3 is a 3-atom path (2 bonds); 4 is isolated
        nb.bond[1] = 1 << 2;
        nb.bond[2] = (1 << 1) | (1 << 3);
        nb.bond[3] = 1 << 2;
        let p = PairBond::compute(&nb, keep, 0);
        assert!(
            p.is_422(),
            "common={} bonds={} chain={}",
            p.common.count_ones(),
            p.bonds,
            p.chain
        );
        assert!(!p.is_421());
        let c = signature_scan(&nb, keep);
        assert_eq!(c.n422, 1, "{c:?}");
    }

    /// 555: 5 common in a pentagon, every atom degree 2. The ICO signature.
    #[test]
    fn pentagon_is_555() {
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        let keep = 0b1111_1111;
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        nb.bond[0] = common;
        // ring 1-2-3-4-5-1
        nb.bond[1] = (1 << 2) | (1 << 5);
        nb.bond[2] = (1 << 1) | (1 << 3);
        nb.bond[3] = (1 << 2) | (1 << 4);
        nb.bond[4] = (1 << 3) | (1 << 5);
        nb.bond[5] = (1 << 4) | (1 << 1);
        let p = PairBond::compute(&nb, keep, 0);
        assert!(
            p.is_555(),
            "common={} bonds={} chain={}",
            p.common.count_ones(),
            p.bonds,
            p.chain
        );
        let c = signature_scan(&nb, keep);
        assert_eq!(c.n555, 1, "{c:?}");
    }

    /// 444 and 666 are the large (14-neighbour) BCC signatures: a square and a
    /// hexagon, both all-degree-2.
    #[test]
    fn square_and_hexagon_are_444_and_666() {
        let keep = 0b1111_1111;
        let mut nb = Neighbourhood::new();
        nb.len = 14;
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        nb.bond[0] = common;
        nb.bond[1] = (1 << 2) | (1 << 4);
        nb.bond[2] = (1 << 1) | (1 << 3);
        nb.bond[3] = (1 << 2) | (1 << 4);
        nb.bond[4] = (1 << 3) | (1 << 1);
        let p = PairBond::compute(&nb, keep, 0);
        assert!(
            p.is_444(),
            "common={} bonds={} chain={}",
            p.common.count_ones(),
            p.bonds,
            p.chain
        );

        let mut nb = Neighbourhood::new();
        nb.len = 14;
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6);
        nb.bond[0] = common;
        nb.bond[1] = (1 << 2) | (1 << 6);
        nb.bond[2] = (1 << 1) | (1 << 3);
        nb.bond[3] = (1 << 2) | (1 << 4);
        nb.bond[4] = (1 << 3) | (1 << 5);
        nb.bond[5] = (1 << 4) | (1 << 6);
        nb.bond[6] = (1 << 5) | (1 << 1);
        let p = PairBond::compute(&nb, keep, 0);
        assert!(
            p.is_666(),
            "common={} bonds={} chain={}",
            p.common.count_ones(),
            p.bonds,
            p.chain
        );
    }

    /// A common set of the right size but the wrong degree pattern must not
    /// match any signature - otherwise the test lattices would pass for the
    /// wrong reason.
    #[test]
    fn wrong_degree_pattern_is_rejected() {
        let keep = 0b1111_1111;
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        // 4 common, but only 1 bond -> neither 421 nor 422
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        nb.bond[0] = common;
        nb.bond[1] = 1 << 2;
        let p = PairBond::compute(&nb, keep, 0);
        assert!(!p.is_421() && !p.is_422() && !p.is_444());
        // 5 common but a star (max degree 4) -> not 555
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        let common = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        nb.bond[0] = common;
        nb.bond[1] = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        let p = PairBond::compute(&nb, keep, 0);
        assert!(!p.is_555(), "chain={}", p.chain);
    }

    /// The classifiers' per-neighbour shortcut must agree exactly with the
    /// whole-triple reference (`PairBond`, via `max_chain`) on every neighbour
    /// of many random symmetric bond graphs, dense and sparse enough to
    /// produce every signature and every near miss.
    #[test]
    fn fast_signatures_match_the_reference() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut hits = [0usize; 5];
        for trial in 0..200_000 {
            let mut nb = Neighbourhood::new();
            nb.len = N_LARGE;
            // Edge probability cycles through 1/8 .. 7/8.
            let p = (trial % 7 + 1) as u64;
            for a in 0..N_LARGE {
                for b in (a + 1)..N_LARGE {
                    if next() % 8 < p {
                        nb.bond[a] |= 1 << b;
                        nb.bond[b] |= 1 << a;
                    }
                }
            }
            for (keep, n) in [(SMALL_KEEP, N_SMALL), (LARGE_KEEP, N_LARGE)] {
                for m in 0..n {
                    let r = PairBond::compute(&nb, keep, m);
                    let small = [r.is_421(), r.is_422(), r.is_555()].iter().position(|&x| x);
                    let large = [r.is_444(), r.is_666()]
                        .iter()
                        .position(|&x| x)
                        .map(|i| i + 3);
                    assert_eq!(small_signature(&nb, keep, m), small, "{trial} {m}");
                    assert_eq!(large_signature(&nb, keep, m), large, "{trial} {m}");
                    for s in small.into_iter().chain(large) {
                        hits[s] += 1;
                    }
                }
            }
        }
        // Every signature must have been exercised, or the test proves little.
        assert!(hits.iter().all(|&h| h > 0), "{hits:?}");
    }

    #[test]
    fn pair_bond_excludes_the_neighbour_itself() {
        let keep = 0b1111_1111;
        let mut nb = Neighbourhood::new();
        nb.len = 8;
        // neighbour 0 is "bonded" to itself by mistake; it must not count
        nb.bond[0] = (1 << 0) | (1 << 1) | (1 << 2);
        let p = PairBond::compute(&nb, keep, 0);
        assert_eq!(p.common.count_ones(), 2);
        assert!(!p.is_421() && !p.is_422());
    }

    #[test]
    fn small_cutoff_scales_by_1207() {
        // 12 neighbours all at r=1 -> local = 1 * 1.2071
        let mut nb = Neighbourhood::new();
        nb.len = 12;
        for t in 0..12 {
            nb.d2[t] = 1.0;
        }
        let lc = small_cutoff(&nb);
        assert!((lc - ADAPTIVE_SCALE).abs() < 1e-12, "got {lc}");
    }

    /// The BCC weighting exists to put both shells on one scale: the first
    /// shell is rescaled by `√(d²/(3/4))`, which for bcc is exactly `a√3/2 /
    /// (√3/2) = a`. So all 14 weighted terms collapse to the lattice constant
    /// and the local cutoff becomes exactly `a·1.207`, independent of which
    /// shell any individual neighbour came from.
    #[test]
    fn large_cutoff_bcc_collapses_both_shells_onto_a() {
        let a = 2.866_f64; // bcc Fe
        let nn = a * 3f64.sqrt() / 2.0; // 8 first-shell neighbours
        let second = a; // 6 second-shell neighbours
        let mut nb = Neighbourhood::new();
        nb.len = 14;
        for t in 0..8 {
            nb.d2[t] = nn * nn;
        }
        for t in 8..14 {
            nb.d2[t] = second * second;
        }
        // both shells rescale to a, so the mean is exactly a
        assert!(
            ((nn * nn / 0.75).sqrt() - a).abs() < 1e-12,
            "first shell must rescale to a"
        );
        let lc = large_cutoff(&nb);
        assert!(
            (lc - a * LARGE_SCALE).abs() < 1e-12,
            "lc={lc} want {}",
            a * LARGE_SCALE
        );
        // The cutoff must sit above the first shell (so 8 bond) and below the
        // second (so 6 do not): that gap is exactly the BCC 8/6 split.
        assert!(lc > nn, "lc={lc} must exceed first shell {nn}");
        assert!(lc < second * LARGE_SCALE + second, "lc={lc}");
    }
}
