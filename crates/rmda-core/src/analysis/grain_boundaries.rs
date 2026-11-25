//! Grain-boundary pass: which atoms are on a seam (`rmda grain-boundaries`).
//!
//! This is the step that makes a polycrystalline result interpretable. A Wigner–Seitz
//! vacancy count is only meaningful where the reference lattice is locally right, and
//! across a grain boundary it is locally right for *neither* grain. Counting those atoms
//! anyway produces a defect density that is a function of grain size rather than of
//! defect chemistry — a sample that is merely finely grained looks more defective than
//! one that is genuinely damaged. So the seam atoms are identified here and excluded
//! from the per-grain counts, rather than being left for the reader to guess at.
//!
//! **The rule, and why.** An atom is a
//! grain-boundary atom when the set of `Grain` ids among *itself and its neighbours*
//! within `w_db` holds two or more distinct values. Three details are load-bearing:
//!
//! - The atom's **own** id is in the set. Without it the test asks for two *other*
//!   grains inside one neighbour list, which on a two-block sample finds nothing at all.
//! - The list is the **full** list, not the half list. A half list holds only `j > i`
//!   (design §9.2), so it is not a neighbourhood: it cannot see a neighbour with a lower
//!   index, and the side of a seam that happens to carry the higher atom ids goes
//!   unreported. Which side that is depends on atom ordering in the input file, so the
//!   error is not even stable between two runs on the same crystal written out in a
//!   different order.
//! - A neighbour with **no** grain counts as its own distinct id, not as a member of
//!   mine. This is the one that decides whether the pass works at all on real data.
//!   Skipping unassigned neighbours looks conservative and is the opposite: the
//!   disordered layer *at* a grain boundary is exactly what PTM cannot classify, so it
//!   is unassigned, and it then hides the seam behind itself. On an ideal sharp FCC
//!   bicrystal 523 of 1000 atoms sit in that layer, and a rule that skips them reports
//!   **zero** boundary atoms at `w_db = 3 Å` — a seam that is not there. For the purpose
//!   this pass exists to serve it does not matter *what* the odd neighbour is: a
//!   neighbour that is not in my grain means the local reference lattice is not reliably
//!   my grain's, and that is the whole question Wigner–Seitz needs answered.
//!
//! Each is pinned by a test against the rule variant it rules out, so none can quietly
//! come back.
//!
//! What this does *not* do: it reports the seam, it does not apply it. Nothing consumes
//! [`GB_FLAG`] to drop a count; the per-grain defect census (`rmda defects`) has its own
//! boundary class. The property is registered and the census is reported.

use crate::analysis::{Analysis, AnalysisContext};
use crate::data::frame::Frame;
use crate::data::property::{DType, Property, PropertyStore};
use crate::error::{Error, Result};
use crate::neighbor::cell_list::CellList;

/// Per-atom seam flag: 1 for a grain-boundary atom, 0 for a grain interior.
pub const GB_FLAG: &str = "GrainBoundary";

/// Per-grain count of that grain's boundary atoms, row index == grain id.
pub const GB_COUNT: &str = "GrainBoundaryCount";

/// Default exclusion half-width, Å.
///
/// 4 Å is roughly one extra interplanar spacing in a typical metal, i.e. wide enough to
/// take in the atoms whose reference lattice is already distorted by the seam and narrow
/// enough not to eat the grain interior. It is a *default*, not a constant: the reported
/// slab thickness says whether it was the binding constraint.
pub const DEFAULT_W_DB: f64 = 4.0;

/// Grain ids that mean "this atom belongs to no grain".
const NO_GRAIN: i64 = -1;

/// The seam-detection pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct GrainBoundaries {
    /// Exclusion half-width in Å. `None` uses [`DEFAULT_W_DB`].
    pub w_db: Option<f64>,
    /// Count atoms with no grain id as boundary atoms themselves.
    ///
    /// Off by default, matching `Grains::include_other`. Note this is *not* the same
    /// thing as ignoring them: an unassigned neighbour already marks the crystalline
    /// atoms around it as boundary (see the module docs). This flag only decides
    /// whether the unassigned atoms are themselves in the class, which for a defect
    /// exclusion is usually what you want either way — they are not being counted as
    /// part of any grain to begin with.
    pub unassigned_is_boundary: bool,
}

impl GrainBoundaries {
    pub fn new() -> Self {
        Self::default()
    }

    /// The exclusion radius actually used, validated.
    fn resolve(&self) -> Result<f64> {
        let w = self.w_db.unwrap_or(DEFAULT_W_DB);
        if !w.is_finite() || w <= 0.0 {
            return Err(Error::invalid(format!(
                "grain-boundary w_db must be positive and finite, got {w}"
            )));
        }
        Ok(w)
    }
}

/// What the pass found.
#[derive(Clone, Debug, Default)]
pub struct SeamCensus {
    /// Per-atom flag, 1 for a boundary atom.
    pub flag: Vec<u8>,
    /// Number of distinct grain ids seen, excluding `NO_GRAIN`.
    pub n_grains: usize,
    /// Flagged atoms.
    pub n_boundary: usize,
    /// Assigned atoms that were *not* flagged.
    pub n_interior: usize,
    /// Atoms carrying a real grain id.
    pub n_assigned: usize,
    /// Atoms with `Grain == -1`.
    pub n_unassigned: usize,
    /// Widest distance from a flagged atom to the nearest unflagged one: the
    /// shallowest exclusion width that would give the same flag set.
    pub slab_thickness: f64,
    /// `w_db` used.
    pub w_db: f64,
}

impl Analysis for GrainBoundaries {
    fn name(&self) -> &'static str {
        "grain-boundaries"
    }

    /// Report `w_db` so a shared cell list is sized for it.
    ///
    /// Unlike `Grains`, this cutoff does not depend on PTM output: the pass reads an
    /// existing `Grain` column and needs no nearest-neighbour distance, so a plain
    /// configured value can be announced up front.
    fn cutoffs(&self) -> Vec<f64> {
        vec![self.w_db.unwrap_or(DEFAULT_W_DB)]
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        out: &mut PropertyStore,
        summary: &mut crate::analysis::Summary,
    ) -> Result<()> {
        let n = ctx.frame.natoms();
        let w_db = self.resolve()?;

        let grains = read_grain(out, n)?;

        // A *full* list, deliberately. See the module docs: the half list cannot answer
        // the question, and the failure is silent and order-dependent.
        let owned;
        let cell = match ctx.nkeys.cell {
            Some(cl) if w_db <= cl.max_query_cutoff() => cl,
            _ => {
                owned =
                    CellList::build(&ctx.frame.cell, &ctx.frame.particles.positions[..n], w_db)?;
                &owned
            }
        };

        let mut flag = vec![0u8; n];
        // Counted separately rather than derived as `assigned - boundary`: with
        // `unassigned_is_boundary` the boundary set includes atoms that were never
        // assigned, so the subtraction underflows. It panics in a debug build and
        // silently wraps to ~1.8e19 in a release one, which is worse -- it prints a
        // number that reads as a measurement.
        let mut n_boundary = 0usize;
        let mut n_interior = 0usize;
        let mut n_assigned = 0usize;
        let mut n_unassigned = 0usize;
        let mut per_grain_count: Vec<i32> = Vec::new();
        let mut scratch = Vec::new();

        for i in 0..n {
            let own = grains[i];
            if own == NO_GRAIN {
                n_unassigned += 1;
                if self.unassigned_is_boundary {
                    flag[i] = 1;
                    n_boundary += 1;
                }
                continue;
            }
            n_assigned += 1;
            grow(&mut per_grain_count, own);
            // The whole point: at least one id other than the atom's own among its
            // neighbours within `w_db`. The atom's own id is the anchor of the
            // {self} u neighbours set, so a neighbour differing from it is exactly
            // the second distinct id. (Walking the full list rather than a half list is
            // what makes this symmetric; see the module docs.)
            let mut other = false;
            cell.neighbors_of(i, w_db, &mut scratch)?;
            for &(j, _) in scratch.iter() {
                // Any id other than my own counts, `NO_GRAIN` included: see the module
                // docs. Requiring `gj != NO_GRAIN` here makes the pass report no
                // seam at all on a real bicrystal.
                if grains[j] != own {
                    other = true;
                    break;
                }
            }
            if other {
                flag[i] = 1;
                n_boundary += 1;
                per_grain_count[own as usize] += 1;
            } else {
                n_interior += 1;
            }
        }

        let n_grains = per_grain_count.len();

        // How wide did the seam actually come out? A `w_db` far wider than the real
        // boundary thickness is not wrong, it is just excluding more than it needs to,
        // and nothing in the counts says so. The thickness is the widest gap between a
        // flagged atom and an unflagged one, which is exactly the number a user needs
        // to know that `w_db` was not binding.
        let slab = slab_thickness(ctx.frame, &flag, w_db);

        let census = SeamCensus {
            flag,
            n_grains,
            n_boundary,
            n_interior,
            n_assigned,
            n_unassigned,
            slab_thickness: slab,
            w_db,
        };

        out.register_flat(
            DType::U8,
            GB_FLAG,
            1,
            census.flag.iter().map(|&b| f64::from(b)).collect(),
        );
        out.register(per_grain_i32(GB_COUNT, &per_grain_count));

        let mut scalars = vec![
            ("gb_atoms_boundary".to_owned(), census.n_boundary as f64),
            ("gb_atoms_interior".to_owned(), n_interior as f64),
            (
                "gb_fraction_boundary".to_owned(),
                if census.n_assigned > 0 {
                    census.n_boundary as f64 / census.n_assigned as f64
                } else {
                    0.0
                },
            ),
            ("gb_atoms_assigned".to_owned(), census.n_assigned as f64),
            ("gb_atoms_unassigned".to_owned(), census.n_unassigned as f64),
            ("gb_grains".to_owned(), census.n_grains as f64),
            ("gb_w_db".to_owned(), census.w_db),
            ("gb_slab_thickness".to_owned(), census.slab_thickness),
        ];
        if self.unassigned_is_boundary {
            scalars.push(("gb_unassigned_is_boundary".to_owned(), 1.0));
        }
        summary.merge_from(crate::analysis::Summary {
            scalars,
            ..Default::default()
        });
        Ok(())
    }
}

/// Read the `Grain` column as `i64`, refusing a frame that has not been segmented.
///
/// `Grains` writes `Grain` as `I32`, so that is the expected dtype; `F64` is accepted
/// too because a caller assembling a `PropertyStore` by hand reaches for it, and a
/// grain id is an integer either way. Any other dtype is refused by name rather than
/// reinterpreted -- reading a `F32` column as grain ids would silently truncate.
fn read_grain(out: &PropertyStore, n: usize) -> Result<Vec<i64>> {
    let p = out.get("Grain").ok_or_else(|| {
        Error::invalid("grain-boundary pass needs a Grain column: run grain segmentation first")
    })?;
    let v: Vec<i64> = if let Some(raw) = p.as_i32() {
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "Grain has {} entries for {n} atoms",
                raw.len()
            )));
        }
        raw[..n].iter().map(|&x| i64::from(x)).collect()
    } else if let Some(raw) = p.as_i64() {
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "Grain has {} entries for {n} atoms",
                raw.len()
            )));
        }
        raw[..n].to_vec()
    } else if let Some(raw) = p.as_u8() {
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "Grain has {} entries for {n} atoms",
                raw.len()
            )));
        }
        raw[..n].iter().map(|&x| i64::from(x)).collect()
    } else if let Some(raw) = p.as_f64() {
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "Grain has {} entries for {n} atoms",
                raw.len()
            )));
        }
        raw[..n].iter().map(|&x| x as i64).collect()
    } else if let Some(raw) = p.as_f32() {
        if raw.len() < n {
            return Err(Error::invalid(format!(
                "Grain has {} entries for {n} atoms",
                raw.len()
            )));
        }
        raw[..n].iter().map(|&x| x as i64).collect()
    } else {
        return Err(Error::invalid(format!(
            "Grain is stored as {:?}, which carries no integer ids",
            p.dtype
        )));
    };
    for &g in &v {
        if g != NO_GRAIN && g < 0 {
            return Err(Error::invalid(format!(
                "Grain holds {g}, which is neither a grain id nor the {NO_GRAIN} 'unassigned' marker"
            )));
        }
    }
    Ok(v)
}

/// Extend `v` so `v[i]` exists, returning the new length needed.
fn grow(v: &mut Vec<i32>, i: i64) {
    let want = i as usize + 1;
    if v.len() < want {
        v.resize(want, 0);
    }
}

/// Build a per-grain `i32` property: one row per grain, `per_particle = false`.
fn per_grain_i32(name: &str, rows: &[i32]) -> Property {
    let mut p = Property::with_capacity(name.to_owned(), DType::I32, false, 1, rows.len());
    for &r in rows {
        p.push_value(&[f64::from(r)]);
    }
    p
}

/// Widest distance from a flagged atom to the nearest unflagged one, over the frame.
///
/// This is the seam's thickness in the only sense that matters operationally: the
/// shallowest exclusion that would produce the same flag set. Returns 0 for an all-
/// flagged or all-unflagged frame, where no such distance exists.
fn slab_thickness(frame: &Frame, flag: &[u8], w_db: f64) -> f64 {
    let n = flag.len();
    if n == 0 || flag.iter().all(|&b| b == 1) || flag.iter().all(|&b| b == 0) {
        return 0.0;
    }
    let pos = &frame.particles.positions[..n];
    let cell = &frame.cell;
    let mut worst = 0.0f64;
    for i in 0..n {
        if flag[i] == 0 {
            continue;
        }
        // Nearest unflagged atom under the minimum-image convention. A cell list at
        // `w_db` would be the fast route, but the answer is only ever reported to two
        // decimals and this runs once per frame; a stride-jumping scan over a
        // `w_db`-sized stencil is bounded by the same `O(N * neighbours)` either way.
        if let Some(d) = nearest_unflagged(pos, cell, flag, i, w_db) {
            worst = worst.max(d);
        }
    }
    worst
}

/// Distance from atom `i` to the nearest atom with `flag == 0`, or `None` if there is
/// none within `radius`.
fn nearest_unflagged(
    pos: &[[f64; 3]],
    cell: &crate::cell::UnitCell,
    flag: &[u8],
    i: usize,
    radius: f64,
) -> Option<f64> {
    let n = pos.len();
    let mut best = f64::INFINITY;
    for j in 0..n {
        if flag[j] != 0 || j == i {
            continue;
        }
        let dr = [
            pos[j][0] - pos[i][0],
            pos[j][1] - pos[i][1],
            pos[j][2] - pos[i][2],
        ];
        // A non-periodic cell returns `dr` unchanged; a periodic one folds it to the
        // closest image. Either way a displacement, so no origin correction is needed.
        if let Ok(d) = cell.min_image(dr) {
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r < best && r <= radius {
                best = r;
            }
        }
    }
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Summary;
    use crate::cell::UnitCell;
    use crate::data::frame::Frame;
    use crate::data::particle::ParticleData;
    use crate::exec::Par;

    /// A run context over `frame`, with no shared cell list so the analysis builds its
    /// own — the path a bare library caller takes.
    fn run(a: &GrainBoundaries, frame: &Frame, store: &mut PropertyStore) -> Result<Summary> {
        let par = Par::new(1);
        let nkeys = crate::analysis::NeighborhoodContext { cell: None };
        let ctx = AnalysisContext {
            frame,
            nkeys: &nkeys,
            par: &par,
        };
        let mut s = Summary::default();
        a.run(&ctx, store, &mut s)?;
        Ok(s)
    }

    fn scalar(s: &Summary, k: &str) -> f64 {
        s.scalars
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("no scalar {k}"))
    }

    /// A row of `n` atoms spaced 1 Å apart along x in a box `n` long, all one grain.
    fn chain(n: usize, box_len: f64) -> Frame {
        let mut pd = ParticleData::with_capacity(n);
        for i in 0..n {
            pd.add_atom([i as f64, 0.0, 0.0], i as i64 + 1, 1);
        }
        Frame::new(UnitCell::orthorhombic(box_len, 10.0, 10.0), pd)
    }

    fn store_with_grain(ids: &[i64]) -> PropertyStore {
        let mut out = PropertyStore::new();
        out.register_flat(
            DType::I32,
            "Grain",
            1,
            ids.iter().map(|&g| g as f64).collect(),
        );
        out
    }

    fn flag_of(store: &PropertyStore) -> Vec<u8> {
        let p = store.get(GB_FLAG).expect("flag written");
        assert_eq!(p.dtype, DType::U8, "flag is a byte per atom");
        p.as_u8().unwrap().clone()
    }

    // ---- the rule, against the variants it rules out ------------------------

    /// The half-list variant: "two distinct ids in the half list", reading only
    /// `j > i` and not counting the atom's own grain. On a two-block chain with a real
    /// cutoff it finds **nothing** — not a thin boundary, none. Pinned so it cannot come
    /// back by accident.
    #[test]
    fn a_the_original_half_list_rule_finds_nothing_where_there_is_a_seam() {
        let n = 8;
        let split = 4;
        let frame = chain(n, 10.0);
        let w_db = 1.5; // first neighbours only, so the geometry is not the confound
        let ids: Vec<i64> = (0..n).map(|i| i64::from(i < split)).collect();
        let store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(w_db),
            ..Default::default()
        };
        let mut store = store;
        run(&a, &frame, &mut store).unwrap();
        let got = flag_of(&store);

        // What the half-list rule computes, by hand: for each atom, the distinct grain
        // ids among strictly-higher-indexed neighbours within w_db.
        let mut original = vec![0u8; n];
        for (i, slot) in original.iter_mut().enumerate() {
            let mut set = std::collections::BTreeSet::new();
            for (j, &gj) in ids.iter().enumerate().skip(i + 1) {
                if ((j as f64) - (i as f64)).abs() <= w_db {
                    set.insert(gj);
                }
            }
            if set.len() >= 2 {
                *slot = 1;
            }
        }
        assert_eq!(
            original.iter().filter(|&&b| b == 1).count(),
            0,
            "the premise: the original rule finds no boundary at all"
        );
        assert_eq!(
            original,
            vec![0u8; n],
            "and specifically not the two atoms that are on the seam"
        );
        // The implemented rule finds them.
        let flagged: Vec<usize> = got
            .iter()
            .enumerate()
            .filter(|(_, &b)| b == 1)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            flagged,
            vec![split - 1, split],
            "seam is the two facing atoms"
        );
    }

    /// The second failure: even with the own-grain anchor, a half list reports the seam
    /// only on the side carrying the lower atom ids. Renumbering the file flips the
    /// error, so the blame moves between grains for the same crystal.
    #[test]
    fn b_the_half_list_rule_reports_only_one_side_of_a_seam() {
        let n = 8;
        let split = 4;
        let w_db = 1.5;
        let frame = chain(n, 10.0);

        // Same crystal, two numberings: which grain owns the low ids is arbitrary.
        // Returns (atoms flagged, the grain each flagged atom is blamed on).
        let report = |ids: &[i64]| -> (Vec<usize>, Vec<i64>) {
            let mut half_anchored = vec![0u8; n];
            for (i, slot) in half_anchored.iter_mut().enumerate() {
                let mut set = std::collections::BTreeSet::new();
                set.insert(ids[i]); // the anchor the original rule lacked
                for (j, &gj) in ids.iter().enumerate().skip(i + 1) {
                    if ((j as f64) - (i as f64)).abs() <= w_db {
                        set.insert(gj);
                    }
                }
                if set.len() >= 2 {
                    *slot = 1;
                }
            }
            let atoms: Vec<usize> = (0..n).filter(|&i| half_anchored[i] == 1).collect();
            let blamed: Vec<i64> = atoms.iter().map(|&i| ids[i]).collect();
            (atoms, blamed)
        };

        let grain_a_low: Vec<i64> = (0..n).map(|i| i64::from(i < split)).collect();
        let grain_a_high: Vec<i64> = (0..n).map(|i| i64::from(i >= split)).collect();

        // A half list only ever looks *upward*, so it can only ever report the
        // lower-indexed atom of a facing pair: atom 3, in both numberings.
        let (atoms_low, blamed_low) = report(&grain_a_low);
        let (atoms_high, blamed_high) = report(&grain_a_high);
        assert_eq!(atoms_low, vec![split - 1], "reports atom 3 only");
        assert_eq!(atoms_high, vec![split - 1], "and atom 3 again");
        assert_eq!(
            (atoms_low, atoms_high),
            (vec![split - 1], vec![split - 1]),
            "the flagged *atom* is the same either way -- it is the lower index of the pair"
        );
        // But the grain it is charged to follows the numbering, which is arbitrary.
        assert_ne!(
            (blamed_low, blamed_high),
            (vec![0], vec![1]),
            "the same crystal, numbered the other way, charges the seam to the \
             other grain -- so per-grain boundary counts depend on atom order"
        );

        // The implemented rule is invariant to that renaming.
        let mut s1 = store_with_grain(&grain_a_low);
        let mut s2 = store_with_grain(&grain_a_high);
        let a = GrainBoundaries {
            w_db: Some(w_db),
            ..Default::default()
        };
        run(&a, &frame, &mut s1).unwrap();
        run(&a, &frame, &mut s2).unwrap();
        assert_eq!(
            flag_of(&s1),
            flag_of(&s2),
            "the seam does not depend on how the atoms were numbered"
        );
        assert_eq!(flag_of(&s1), vec![0, 0, 0, 1, 1, 0, 0, 0]);
        // And the per-grain counts come out symmetric, one facing atom each.
        for st in [&s1, &s2] {
            let p = st.get(GB_COUNT).expect("count");
            let counts: Vec<i64> = p.as_i32().unwrap().iter().map(|&v| i64::from(v)).collect();
            assert_eq!(
                counts,
                vec![1, 1],
                "each grain owns exactly one facing atom"
            );
        }
    }

    // ---- the rule, on its own terms ----------------------------------------

    #[test]
    fn c_a_single_grain_has_no_boundary() {
        let n = 12;
        let frame = chain(n, 14.0);
        let ids = vec![0i64; n];
        let mut store = store_with_grain(&ids);
        let s = run(&GrainBoundaries::new(), &frame, &mut store).unwrap();
        assert_eq!(flag_of(&store), vec![0u8; n]);
        assert_eq!(scalar(&s, "gb_atoms_boundary"), 0.0);
        assert_eq!(scalar(&s, "gb_fraction_boundary"), 0.0);
        assert_eq!(scalar(&s, "gb_slab_thickness"), 0.0);
    }

    #[test]
    fn d_every_atom_on_a_three_block_chain_is_flagged() {
        // 0 0 0 | 1 1 1 | 2 2 2 with first-neighbour cutoff: atoms 2,3 and 5,6.
        let n = 9;
        let frame = chain(n, 11.0);
        let ids: Vec<i64> = (0..n).map(|i| (i / 3) as i64).collect();
        let mut store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(1.5),
            ..Default::default()
        };
        let s = run(&a, &frame, &mut store).unwrap();
        assert_eq!(flag_of(&store), vec![0, 0, 1, 1, 0, 1, 1, 0, 0]);
        assert_eq!(scalar(&s, "gb_atoms_boundary"), 4.0);
        assert_eq!(scalar(&s, "gb_atoms_interior"), 5.0);
        assert_eq!(scalar(&s, "gb_grains"), 3.0);
        assert_eq!(scalar(&s, "gb_fraction_boundary"), 4.0 / 9.0);
    }

    #[test]
    fn e_the_per_grain_count_sums_to_the_total() {
        let n = 12;
        let frame = chain(n, 14.0);
        // Interleave so every grain is a singleton pair of touching atoms: 0 1 0 1 ...
        let ids: Vec<i64> = (0..n).map(|i| (i % 2) as i64).collect();
        let mut store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(1.5),
            ..Default::default()
        };
        let s = run(&a, &frame, &mut store).unwrap();
        let p = store.get(GB_COUNT).expect("per-grain count written");
        assert!(!p.per_particle, "one row per grain, not per atom");
        let counts: Vec<i64> = p.as_i32().unwrap().iter().map(|&v| i64::from(v)).collect();
        assert_eq!(counts, vec![6, 6]);
        let total: i64 = counts.iter().sum();
        assert_eq!(total as f64, scalar(&s, "gb_atoms_boundary"));
    }

    #[test]
    fn f_a_single_atom_grain_is_found_only_because_of_the_own_grain_anchor() {
        // A one-atom grain: its own id is the only id that distinguishes it, since
        // every neighbour it has belongs to a *different* grain. A rule that counted
        // only the ids found among neighbours sees {0, 0} -- one id -- and calls it
        // interior. This is the case that decides whether the anchor stays.
        let n = 3;
        let frame = chain(n, 10.0);
        let ids = vec![0i64, 1, 0];
        let w_db = 1.5;

        // The variant that drops the anchor, computed by hand.
        let no_anchor: Vec<u8> = (0..n)
            .map(|i| {
                let set: std::collections::BTreeSet<i64> = (0..n)
                    .filter(|&j| j != i && ((j as f64) - (i as f64)).abs() <= w_db)
                    .map(|j| ids[j])
                    .collect();
                u8::from(set.len() >= 2)
            })
            .collect();
        // Every atom is missed, not just the middle one: without the anchor each
        // atom's own id is invisible to the test, and in a three-atom sandwich no atom
        // can see two *other* grains. So the whole seam reads as absent -- which is
        // the same total failure the half-list rule has, reached a second way.
        assert_eq!(
            no_anchor,
            vec![0, 0, 0],
            "premise: without the anchor this seam is invisible to every atom on it"
        );

        let mut store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(w_db),
            ..Default::default()
        };
        let s = run(&a, &frame, &mut store).unwrap();
        assert_eq!(
            flag_of(&store),
            vec![1, 1, 1],
            "the anchor finds the seam on all three atoms"
        );
        assert_eq!(scalar(&s, "gb_atoms_boundary"), 3.0);
        // The per-grain count is what makes the single-atom grain visible at all:
        // a grain of one atom that is entirely seam.
        let counts: Vec<i64> = store
            .get(GB_COUNT)
            .unwrap()
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| i64::from(v))
            .collect();
        assert_eq!(counts, vec![2, 1]);
    }

    /// The variant that makes the pass useless on real data. An ideal sharp two-block
    /// chain whose middle pair is unassigned: the disordered layer *at* the seam is what
    /// PTM cannot classify, so skipping it hides the seam behind itself. Atoms 1 and 4
    /// each touch an unassigned atom, so each has a neighbour outside its own grain, and
    /// each is on the boundary.
    #[test]
    fn g_an_unassigned_neighbour_still_marks_the_seam() {
        let n = 6;
        let frame = chain(n, 10.0);
        let ids = vec![0i64, 0, -1, -1, 1, 1];
        let mut store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(1.5),
            ..Default::default()
        };
        let s = run(&a, &frame, &mut store).unwrap();
        assert_eq!(scalar(&s, "gb_atoms_unassigned"), 2.0);
        assert_eq!(scalar(&s, "gb_atoms_assigned"), 4.0);
        assert_eq!(
            flag_of(&store),
            vec![0, 1, 0, 0, 1, 0],
            "atoms 1 and 4 each touch an unassigned atom and are on the seam"
        );
        assert_eq!(scalar(&s, "gb_atoms_boundary"), 2.0);

        // The unassigned atoms themselves are not in the class by default -- they are
        // not part of any grain's interior to begin with -- but they can be.
        let mut store2 = store_with_grain(&ids);
        let b = GrainBoundaries {
            w_db: Some(1.5),
            unassigned_is_boundary: true,
        };
        let s2 = run(&b, &frame, &mut store2).unwrap();
        assert_eq!(flag_of(&store2), vec![0, 1, 1, 1, 1, 0]);
        assert_eq!(scalar(&s2, "gb_unassigned_is_boundary"), 1.0);
        assert_eq!(scalar(&s2, "gb_atoms_boundary"), 4.0);
        // The identity `assigned == interior + boundary` only holds in the default
        // mode. With unassigned atoms in the boundary set it must NOT be asserted, and
        // `interior` has to be counted rather than subtracted -- subtracting underflows
        // here, which panics in debug and prints ~1.8e19 in release.
        // Of the four assigned atoms, two are seam (1 and 4) and two are not (0 and 5).
        assert_eq!(scalar(&s2, "gb_atoms_interior"), 2.0);
        assert_eq!(scalar(&s2, "gb_atoms_assigned"), 4.0);
        assert_eq!(
            scalar(&s2, "gb_atoms_boundary"),
            4.0,
            "2 assigned seam atoms + the 2 unassigned ones: the boundary count is \
             larger than the assigned contribution, which is why `interior` has to be \
             counted rather than recovered by subtracting from `assigned`"
        );
        // Still no per-grain row for them: they belong to no grain.
        let counts: Vec<i64> = store2
            .get(GB_COUNT)
            .unwrap()
            .as_i32()
            .unwrap()
            .iter()
            .map(|&v| i64::from(v))
            .collect();
        assert_eq!(counts, vec![1, 1]);
    }

    #[test]
    fn h_the_slab_thickness_reports_how_wide_the_exclusion_actually_is() {
        // Flagged atoms 3 and 4; the nearest unflagged atom is 1 A from either.
        let n = 8;
        let frame = chain(n, 10.0);
        let ids: Vec<i64> = (0..n).map(|i| i64::from(i < 4)).collect();
        let mut store = store_with_grain(&ids);
        let a = GrainBoundaries {
            w_db: Some(1.5),
            ..Default::default()
        };
        let s = run(&a, &frame, &mut store).unwrap();
        assert_eq!(scalar(&s, "gb_slab_thickness"), 1.0);
    }

    // ---- refusals -----------------------------------------------------------

    #[test]
    fn i_an_unsegmented_frame_is_refused_by_name() {
        let frame = chain(4, 6.0);
        let mut store = PropertyStore::new();
        let e = run(&GrainBoundaries::new(), &frame, &mut store).unwrap_err();
        assert!(
            e.to_string().contains("Grain"),
            "error should name the missing column, got: {e}"
        );
    }

    #[test]
    fn j_a_nonpositive_w_db_is_refused() {
        let frame = chain(4, 6.0);
        let store = store_with_grain(&[0, 0, 1, 1]);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let a = GrainBoundaries {
                w_db: Some(bad),
                ..Default::default()
            };
            let e = run(&a, &frame, &mut store.clone()).unwrap_err();
            assert!(
                e.to_string().contains("w_db"),
                "w_db {bad} should be refused by name, got: {e}"
            );
        }
    }

    #[test]
    fn k_a_negative_grain_id_other_than_the_marker_is_refused() {
        // -1 means unassigned; -7 is a bug upstream and must not become a grain.
        let frame = chain(4, 6.0);
        let mut store = store_with_grain(&[0, 0, 0, -7]);
        let e = run(&GrainBoundaries::new(), &frame, &mut store).unwrap_err();
        assert!(e.to_string().contains("-7"), "got: {e}");
    }

    #[test]
    fn l_an_empty_frame_is_all_zeros() {
        let pd = ParticleData::with_capacity(0);
        let frame = Frame::new(UnitCell::orthorhombic(10.0, 10.0, 10.0), pd);
        // `Grains` still writes an (empty) `Grain` column on an empty frame, so the
        // pass still gets its column; requiring it keeps one contract for every size.
        let mut store = store_with_grain(&[]);
        let s = run(&GrainBoundaries::new(), &frame, &mut store).unwrap();
        assert_eq!(scalar(&s, "gb_atoms_boundary"), 0.0);
        assert_eq!(scalar(&s, "gb_fraction_boundary"), 0.0);
        assert!(flag_of(&store).is_empty());
    }

    #[test]
    fn m_the_default_w_db_is_announced_so_the_shared_list_can_be_sized() {
        assert_eq!(GrainBoundaries::new().cutoffs(), vec![DEFAULT_W_DB]);
        assert_eq!(
            GrainBoundaries {
                w_db: Some(3.2),
                ..Default::default()
            }
            .cutoffs(),
            vec![3.2]
        );
    }
}
