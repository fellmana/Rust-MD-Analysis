//! Radial distribution function (design §18.2).
//!
//! The normalisation here is OVITO's and LAMMPS' `compute rdf`, and it is
//! deliberately *not* the textbook `g(r) = Npair(r) / (4*pi*r^2*dr*rho)` written
//! with a full pair list. Getting those two forms to agree on paper is where the
//! ½ and the `N(N-1)` versus `N²` arguments start, and the factor looks wrong
//! enough to be suspected as a bug. It is not, so the derivation is written
//! out here rather than left to the reader:
//!
//! - Pairs are counted on a **half** list: each unordered pair once, so the total
//!   is `N(N-1)/2`, not `N²`.
//! - The shell-volume normalisation uses the total count over the whole frame,
//!   `c = N1*N2` for a type pair (`N²` for the total), which is a *full*-list
//!   quantity.
//! - Halving the pair count while normalising by the full count therefore needs
//!   a factor 2 — but only for same-type pairs, where the half list really does
//!   undercount by 2×. For a mixed pair `1≠2` each unordered cross pair is still
//!   counted exactly once by `i<j`, so no factor applies. OVITO encodes this as
//!   `normalizeRDF(..., mult = 2 if i == j else 1)`.
//! - Hence `g_k = mult * H_k * V / (N1*N2 * vol_k)`. Equivalently: `g` is the pair
//!   density in the shell over the uniform-fluid pair density,
//!   `g = ((2H/N)/vol_k)/rho`, and with `vol_k ≈ 4*pi*r^2*dr` that is the familiar
//!   `g(r) = Npair(r)/(4*pi*r^2*dr*rho)` for the full-list count `Npair = 2H`.
//!   Consistent, and identical to LAMMPS.
//!
//! **A tempting transcription error.** The phrasing `prefactor = (4/3)*pi*N1*N2/V`
//! with `c = N1*N2` in the denominator collapses to `(4/3)*pi*H/(V*vol)`.
//! That has units of 1/Å⁶ — `g` must be dimensionless — and it is additionally
//! wrong by `(V/N)^2`. On a 3375-atom lattice it puts the first shell at `g = 0.08`
//! where the exact answer is 13.86, and the coordination number at 6 neighbours as
//! 0.43. Loud, but only if something checks against a hand-counted lattice, so
//! everything below is pinned to exact simple-cubic shell counts rather than to a
//! second copy of the formula.
//!
//! So: keep both the half-list counting and the `N1*N2` (`N²`) denominator, and never
//! "fix" the factor. The raw `H_k` is reported alongside `g(r)` so a downstream
//! tool wanting the un-normalised histogram does not have to invert this.
//!
//! Partials are per-type-pair `g_ab(r)` with `N_a*N_b`, same prefactor, same
//! diagonal doubling.

use crate::analysis::occupancy::Occupancy;
use crate::analysis::{Analysis, AnalysisContext, Summary};
use crate::error::{Error, Result};
use crate::neighbor::cell_list::CellList;
use std::collections::BTreeMap;

/// Narrowest Gaussian, Å, the effective volume's density estimate may use
/// (`Occupancy::smoothed_density`).
const MIN_DENSITY_WIDTH: f64 = 2.0;

/// Radial distribution function over one frame.
#[derive(Clone, Debug)]
pub struct Rdf {
    /// Maximum pair distance, Å. The last bin ends exactly here.
    pub rmax: f64,
    /// Bin width, Å.
    pub dr: f64,
    /// Whether to accumulate per-type-pair partials.
    pub partials: bool,
    /// Whether to integrate the coordination number `n(r) = 4*pi*rho*∫g r² dr`.
    pub coordination: bool,
    /// The volume `g(r)` is normalised by (design §18.2).
    pub volume: RdfVolume,
    /// Count pairs only around atoms more than `rmax` from a free surface, so
    /// the shells of the atoms counted are whole and `g(r) → 1` in a film.
    pub edge_correct: bool,
}

/// The volume `V` in `g = mult·H·V/(N1·N2·vol)`: `N/V` must be the material's
/// number density, which the box gives only when the material fills it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RdfVolume {
    /// `Effective` when an axis is open or a periodic axis has a vacuum gap,
    /// else `Box`.
    Auto,
    /// The cell volume (periodic bulk).
    Box,
    /// `N / ρ`, with ρ the interior number density measured on an occupancy
    /// grid (`analysis::occupancy`), so vacuum and surface cells do not count.
    Effective,
    /// A volume in Å³ given explicitly.
    Value(f64),
}

/// The most bins an RDF histogram may have: 80 MB per pair of types at u64.
const MAX_BINS: usize = 10_000_000;

impl std::str::FromStr for RdfVolume {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(RdfVolume::Auto),
            "box" => Ok(RdfVolume::Box),
            "effective" => Ok(RdfVolume::Effective),
            v => match v.parse::<f64>() {
                Ok(x) if x.is_finite() && x > 0.0 => Ok(RdfVolume::Value(x)),
                _ => Err(format!(
                    "rdf volume must be auto, box, effective or a positive volume in A^3, got {v}"
                )),
            },
        }
    }
}

impl Default for Rdf {
    /// Library defaults: `rmax` 10 Å, `dr` 0.02 Å, partials and coordination
    /// numbers on. The CLI and Python make `rmax` explicit.
    fn default() -> Self {
        Rdf {
            rmax: 10.0,
            dr: 0.02,
            partials: true,
            coordination: true,
            volume: RdfVolume::Auto,
            edge_correct: false,
        }
    }
}

impl Rdf {
    pub fn new(rmax: f64, dr: f64) -> Self {
        Rdf {
            rmax,
            dr,
            ..Default::default()
        }
    }

    /// Number of bins. `rmax` need not be a multiple of `dr`; the last bin is
    /// then shorter than the others and its `g` is scaled by its own volume,
    /// which is why the shell volumes are computed per bin rather than once.
    pub fn nbins(&self) -> usize {
        ((self.rmax / self.dr).ceil() as usize).max(1)
    }

    /// Bin edges, `nbins + 1` of them.
    ///
    /// The last edge is `rmax`, not `nbins * dr`. When `rmax` is not a multiple of
    /// `dr` those differ, and reporting an edge past `rmax` would invite a consumer
    /// to integrate over a range no pair was counted in.
    pub fn edges(&self) -> Vec<f64> {
        (0..self.nbins())
            .map(|i| i as f64 * self.dr)
            .chain(std::iter::once(self.rmax))
            .collect()
    }

    /// Shell volume `4*pi/3 * (r2³ - r1³)`, with `r2` clamped to `rmax`.
    ///
    /// The clamp matters for the same reason as in [`Rdf::edges`]: without it the
    /// final bin of a non-integral `rmax` is a whole `dr` wide while holding only the
    /// pairs below `rmax`, so its `g` is scaled down by `dr/(rmax - r1)` -- a step at
    /// the last shell that is not in the data. On `rmax = 1.05, dr = 0.5` that makes
    /// the last bin's volume 11x too large.
    fn shell_volume(&self, bin: usize) -> f64 {
        let r1 = bin as f64 * self.dr;
        let r2 = ((bin as f64 + 1.0) * self.dr).min(self.rmax);
        if r2 <= r1 {
            return 0.0;
        }
        (4.0 / 3.0) * std::f64::consts::PI * (r2 * r2 * r2 - r1 * r1 * r1)
    }

    /// Bin `k` for a pair distance `r`, or `None` past `rmax`.
    ///
    /// `r` exactly on a bin edge goes in the *upper* bin (`floor`), which is what
    /// both OVITO and LAMMPS do: the shell is `(r1, r2]`. The alternative
    /// (`round`) moves every boundary case and is a silent ½-bin shift of the
    /// whole histogram.
    fn bin_of(&self, r: f64) -> Option<usize> {
        // The `r >= rmax` test is explicit rather than left to fall out of
        // `k < nbins`. Without it the boundary depends on whether `rmax` happens to
        // be a multiple of `dr`: at `rmax = 2.0, dr = 0.5` a pair at exactly 2.0 is
        // dropped, and at `rmax = 1.05, dr = 0.5` the same pair is kept. Two
        // neighbouring runs would then disagree about whether the boundary shell
        // exists, which is the kind of difference that looks like physics.
        if !r.is_finite() || r < 0.0 || r >= self.rmax {
            return None;
        }
        let k = (r / self.dr).floor() as usize;
        if k >= self.nbins() {
            None
        } else {
            Some(k)
        }
    }
    /// Check the requested range against the geometry, before any work is done.
    ///
    /// Split out from [`Analysis::run`] so a driver can reject a nonsense range
    /// *before* it builds a neighbour list for it. Otherwise `rmax = 0` fails
    /// inside the cell-list constructor with a message about a "cutoff", which is
    /// true of the pipeline and useless to someone who asked for an RDF.
    pub fn validate(&self, cell: Option<&crate::cell::UnitCell>) -> Result<()> {
        // A non-finite or non-positive `dr` makes `bin_of` return garbage (every
        // distance lands in bin 0, or the division is NaN and the cast is
        // undefined), and a non-positive `rmax` reports an empty histogram that
        // looks like "no pairs at all". Both are refused rather than answered.
        if !self.dr.is_finite() || self.dr <= 0.0 {
            return Err(Error::invalid(format!(
                "rdf bin width must be positive and finite, got {}",
                self.dr
            )));
        }
        if !self.rmax.is_finite() || self.rmax <= 0.0 {
            return Err(Error::invalid(format!(
                "rdf rmax must be positive and finite, got {}",
                self.rmax
            )));
        }
        // The histogram is dense: `rmax / dr` bins. A typo such as
        // `--rmax 1e18` asks for 10^19 of them and would abort the allocation.
        let bins = (self.rmax / self.dr).ceil();
        if bins > MAX_BINS as f64 {
            return Err(Error::invalid(format!(
                "rdf rmax / dr asks for {bins:.3e} bins; at most {MAX_BINS} (raise dr or lower rmax)"
            )));
        }
        let Some(cell) = cell else { return Ok(()) };
        // `rmax` at or beyond half the smallest cell height cannot be measured:
        // the minimum-image convention only considers one image per pair, so a
        // real neighbour at that distance is folded onto a shorter one. The
        // histogram would still be produced, and would be wrong in a way that
        // looks like structure. This is the RDF analogue of refusing a non-orthogonal
        // cell, and unlike the shells it is a hard error because the caller asked
        // for a range the geometry cannot support.
        // Only periodic axes fold pairs; an open axis may be shorter than rmax.
        // The limit is half the smallest perpendicular *height*, not edge: in
        // a tilted cell the image across a face is a height away, and the edge
        // lengths overstate the range (a cell tilted by half a box length would
        // let rmax run ~12 % past where pairs fold).
        let h = cell.heights();
        let half_min = (0..3)
            .filter(|&a| cell.pbc[a])
            .map(|a| h[a])
            .fold(f64::INFINITY, f64::min)
            / 2.0;
        if cell.pbc.iter().any(|&p| p) && self.rmax > half_min {
            return Err(Error::invalid(format!(
                "rdf rmax {} exceeds half the smallest cell height ({half_min:.4} A); \
                 the minimum-image convention cannot resolve pairs that far",
                self.rmax
            )));
        }
        let volume = cell.volume();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(Error::invalid(format!(
                "rdf needs a positive cell volume, got {volume}"
            )));
        }
        Ok(())
    }
}

impl Analysis for Rdf {
    fn name(&self) -> &'static str {
        "rdf"
    }

    fn cutoffs(&self) -> Vec<f64> {
        vec![self.rmax]
    }

    fn run(
        &self,
        ctx: &AnalysisContext<'_>,
        _out: &mut crate::data::property::PropertyStore,
        summary: &mut Summary,
    ) -> Result<()> {
        let frame = ctx.frame;
        let n = frame.natoms();
        let nbins = self.nbins();

        self.validate(Some(&frame.cell))?;
        // An empty frame is reported as an empty histogram, not an error: `rdf`
        // cannot miscount a frame with no atoms, and refusing it would make an
        // `analyze` run over an empty trajectory fail outright.
        if n == 0 {
            summary.merge_from(Summary::scalar("rdf_rmax", self.rmax));
            summary.merge_from(Summary::scalar("rdf_dr", self.dr));
            summary.merge_from(Summary::scalar("rdf_bins", nbins as f64));
            summary.merge_from(Summary::scalar("rdf_atoms", 0.0));
            summary.merge_from(Summary::scalar("rdf_number_density", 0.0));
            summary.merge_from(Summary::scalar("rdf_pairs", 0.0));
            summary.merge_from(Summary::scalar("rdf_pairs_possible", 0.0));
            let zeros = vec![0.0; nbins];
            for name in ["rdf_g_r", "rdf_pair_counts"] {
                summary.histograms.push((name.to_string(), zeros.clone()));
            }
            return Ok(());
        }
        let cell = &frame.cell;
        let positions = &frame.particles.positions[..n];
        // The normalisation volume (design §18.2). The occupancy grid is only
        // built when a mode needs it; a periodic bulk frame under `auto` costs
        // one O(N) pass to rule out a vacuum gap.
        let grid = match (self.volume, self.edge_correct) {
            (RdfVolume::Auto | RdfVolume::Effective, _) | (_, true) => {
                Some(Occupancy::build(cell, positions)?)
            }
            _ => None,
        };
        let effective = match self.volume {
            RdfVolume::Box | RdfVolume::Value(_) => false,
            RdfVolume::Effective => true,
            RdfVolume::Auto => {
                let gap = grid.as_ref().map_or([false; 3], |g| g.periodic_gap());
                (0..3).any(|a| !cell.pbc[a] || gap[a])
            }
        };
        let volume = match self.volume {
            RdfVolume::Value(v) => v,
            _ if effective => {
                // The smoothed neighbour count where some atom is deep enough,
                // else the grid's interior cells (aliasing with the lattice
                // planes costs them a few per cent in small structures).
                let g = grid.as_ref().expect("grid built");
                let rho = match g.smoothed_density(cell, positions, MIN_DENSITY_WIDTH)? {
                    Some(r) => r,
                    None => g.interior_density(),
                };
                if !(rho.is_finite() && rho > 0.0) {
                    return Err(Error::invalid(
                        "rdf: could not measure the material's number density",
                    ));
                }
                n as f64 / rho
            }
            // Already checked by `validate`.
            _ => cell.volume(),
        };
        // Pair centres: every atom, or with `edge_correct` the atoms more than
        // rmax from a free surface, whose shells are whole.
        let centre: Option<Vec<bool>> = if self.edge_correct {
            let g = grid.as_ref().expect("grid built");
            let inv_t = cell.inverse_transposed()?;
            Some(
                positions
                    .iter()
                    .map(|&x| g.depth_of(Occupancy::frac(cell, &inv_t, x)) > self.rmax)
                    .collect(),
            )
        } else {
            None
        };
        let n_centres = centre
            .as_ref()
            .map_or(n, |c| c.iter().filter(|&&b| b).count());
        if self.edge_correct && n_centres == 0 {
            return Err(Error::invalid(format!(
                "rdf: no atom is more than rmax = {} A from a free surface; \
                 lower rmax or drop --rdf-edge-correct",
                self.rmax
            )));
        }

        // Own cell list rather than the pipeline's: `rmax` is the largest cutoff
        // any analysis in the pipeline may declare, and a narrower shared list
        // would truncate the query stencil and drop the far shells this analysis
        // exists to measure. Same reasoning as `ClusterAnalysis`'s cutoff path.
        let cell_list = CellList::build(cell, &frame.particles.positions[..n], self.rmax)?;

        // Raw pair counts. `u64` because at 128M atoms the first shell alone can
        // hold ~1e8 pairs and `f32` would lose integer precision above 2^24 --
        // which is exactly the shell where g(r) is largest and most often read.
        let mut total = vec![0u64; nbins];
        // `BTreeMap` so partials are emitted in a fixed (a <= b) order rather than
        // hash order, keeping the output diffable between runs.
        let mut partials: BTreeMap<(u32, u32), Vec<u64>> = BTreeMap::new();
        let mut type_counts: BTreeMap<u32, u64> = BTreeMap::new();
        for &t in &frame.particles.types[..n] {
            *type_counts.entry(t).or_insert(0) += 1;
        }

        // Pair loop over fixed §10.2 chunks in the analysis's own parallel
        // context, so `--threads` reaches the RDF reduction. Each chunk owns
        // its outer-atom range and keeps the `j > i` half-list rule, so every
        // unordered pair is counted exactly once by exactly one chunk; the
        // per-chunk totals and partials are then merged in ascending chunk
        // ordinal order. Integer adds into zeroed buffers make the merged
        // result bit-identical for any thread count (design §10.2).
        let chunk_hist = ctx.par.try_map_chunks(n, |s, e| -> Result<_> {
            let mut local_total = vec![0u64; nbins];
            let mut local_partials: BTreeMap<(u32, u32), Vec<u64>> = BTreeMap::new();
            let mut scratch = Vec::new();
            for i in s..e {
                if centre.as_ref().map_or(false, |c| !c[i]) {
                    continue;
                }
                cell_list.neighbors_of(i, self.rmax, &mut scratch)?;
                let ti = frame.particles.types[i];
                for &(j, dr_vec) in scratch.iter() {
                    // Half list: `neighbors_of` is a full list, so keep `j > i` to
                    // count each unordered pair exactly once. Around edge-corrected
                    // centres every neighbour counts, once per centre.
                    if if centre.is_some() { j == i } else { j <= i } {
                        continue;
                    }
                    let r = (dr_vec[0] * dr_vec[0] + dr_vec[1] * dr_vec[1] + dr_vec[2] * dr_vec[2])
                        .sqrt();
                    let Some(k) = self.bin_of(r) else { continue };
                    local_total[k] += 1;
                    if self.partials {
                        let tj = frame.particles.types[j];
                        let key = if ti <= tj { (ti, tj) } else { (tj, ti) };
                        local_partials
                            .entry(key)
                            .or_insert_with(|| vec![0u64; nbins])[k] += 1;
                    }
                }
            }
            Ok((local_total, local_partials))
        })?;
        for (chunk_total, chunk_partials) in chunk_hist {
            for (t, c) in total.iter_mut().zip(chunk_total.iter()) {
                *t += *c;
            }
            for (key, hist) in chunk_partials {
                let slot = partials.entry(key).or_insert_with(|| vec![0u64; nbins]);
                for (a, b) in slot.iter_mut().zip(hist.iter()) {
                    *a += *b;
                }
            }
        }

        let rho = n as f64 / volume;
        summary.merge_from(Summary::scalar("rdf_volume", volume));
        summary.merge_from(Summary::scalar(
            "rdf_volume_effective",
            if effective { 1.0 } else { 0.0 },
        ));
        summary.merge_from(Summary::scalar("rdf_centres", n_centres as f64));
        summary.merge_from(Summary::scalar("rdf_rmax", self.rmax));
        summary.merge_from(Summary::scalar("rdf_dr", self.dr));
        summary.merge_from(Summary::scalar("rdf_bins", nbins as f64));
        summary.merge_from(Summary::scalar("rdf_atoms", n as f64));
        summary.merge_from(Summary::scalar("rdf_number_density", rho));
        let pairs: u64 = total.iter().sum();
        summary.merge_from(Summary::scalar("rdf_pairs", pairs as f64));
        // `rdf_pairs` (above) counts the unordered pairs that fell inside
        // `rmax`; this is all of the frame's, `N(N-1)/2`. They agree only when
        // `rmax` reaches the whole box.
        summary.merge_from(Summary::scalar(
            "rdf_pairs_possible",
            (n * (n - 1) / 2) as f64,
        ));

        // g(r) for the whole frame. `mult = 2` cancels the half-list factor for the
        // total, as derived in the module docs.
        // With edge-corrected centres the count is a full list around each of
        // `n_centres` atoms: `g = H / (n_centres·ρ·vol)`, which is the same
        // formula with `mult·V/(N1·N2)` replaced by `1/(n_centres·ρ)`.
        let g = if self.edge_correct {
            normalize(
                &total,
                &self.shell_volumes(),
                1.0,
                n_centres as f64 * rho,
                1.0,
            )
        } else {
            normalize(&total, &self.shell_volumes(), volume, (n * n) as f64, 2.0)
        };
        summary.histograms.push(("rdf_g_r".to_string(), g.clone()));

        // The un-normalised histogram, so nobody has to invert the normalisation
        // above to get at pair counts.
        summary.histograms.push((
            "rdf_pair_counts".to_string(),
            total.iter().map(|&v| v as f64).collect(),
        ));

        if self.coordination {
            // n(r) = 4*pi*rho * integral_0^r g(rho) rho^2 drho, accumulated over
            // the same shells. This is the number of neighbours within r, so it is
            // the integral people actually want, and it must integrate the *same*
            // g(r) that is reported -- a separate approximation here would make the
            // two disagree at the last shell with no visible cause.
            //
            // The `4*pi` is already inside the shell volume
            // (`vol_k = 4/3*pi*(r2^3 - r1^3) ~ 4*pi*s^2*ds`), so the discrete form
            // is `rho * sum(g*vol)` with **no** second factor. Writing
            // `4*pi*rho*sum(g*vol)` is the obvious-looking form and is wrong by
            // exactly `4*pi`; on the lattice in this module's tests it reports 6
            // neighbours as 0.433.
            let mut acc = 0.0;
            let cn: Vec<f64> = g
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    acc += v * self.shell_volume(k) * rho;
                    acc
                })
                .collect();
            let at_rmax = cn.last().copied().unwrap_or(0.0);
            summary
                .histograms
                .push(("rdf_coordination_number".to_string(), cn));
            summary.merge_from(Summary::scalar("rdf_coordination_at_rmax", at_rmax));
        }

        if self.partials {
            for ((a, b), counts) in &partials {
                let na = type_counts.get(a).copied().unwrap_or(0) as f64;
                let nb = type_counts.get(b).copied().unwrap_or(0) as f64;
                let nab = na * nb;
                if nab == 0.0 {
                    continue;
                }
                // The diagonal doubling is per *type pair*: `a == b` means the
                // half list undercounted same-type pairs by 2x and needs the
                // factor. This is the `mult = 2 if i == j else 1` of OVITO's
                // normalizeRDF, and getting it wrong makes every partial read low
                // by 2x -- a uniform factor, so it is invisible in the shape and
                // easy to miss.
                let mult = if a == b { 2.0 } else { 1.0 };
                let name = format!("rdf_g_{a}_{b}");
                let g = if let Some(c) = &centre {
                    // Centres of type a see b at density ρ·x_b, and for a ≠ b the
                    // key also holds b-centred pairs seeing a.
                    let centres_of = |t: u32| {
                        frame.particles.types[..n]
                            .iter()
                            .zip(c)
                            .filter(|&(&ty, &on)| on && ty == t)
                            .count() as f64
                    };
                    let (xa, xb) = (na / n as f64, nb / n as f64);
                    let mut d = centres_of(*a) * rho * xb;
                    if a != b {
                        d += centres_of(*b) * rho * xa;
                    }
                    normalize(counts, &self.shell_volumes(), 1.0, d, 1.0)
                } else {
                    normalize(counts, &self.shell_volumes(), volume, nab, mult)
                };
                summary.histograms.push((name, g));
            }
        }

        Ok(())
    }
}

impl Rdf {
    fn shell_volumes(&self) -> Vec<f64> {
        (0..self.nbins()).map(|k| self.shell_volume(k)).collect()
    }
}

/// `g_k = mult * H_k * V / (N1*N2 * vol_k)` for a pair population of `N1*N2` atoms
/// in volume `V`, with `mult = 2` for a same-type (or total) pair and `1` for a
/// mixed one.
///
/// The dimension check is what pins this form down: `H` is a bare count and
/// `V/vol` is dimensionless, so `H*V/(N1*N2*vol)` is dimensionless and is the only
/// candidate for `g`. The tempting transcription of OVITO's source —
/// `(4/3)*pi*N1*N2/V` in the numerator with `N1*N2` also in the
/// denominator — collapses to `(4/3)*pi*H/(V*vol)`, which carries units of 1/Å⁶
/// and is wrong by `(V/N)^2` on top of that. On the 3375-atom lattice in this
/// module's tests `(V/N)^2` is 729, so the wrong form puts the first shell at
/// `g = 0.08` where the exact answer is 13.86. Loud enough to catch, but only if
/// something checks against a hand-counted lattice.
///
/// Equivalently, and this is the form to remember: `g` is the pair density in the
/// shell divided by the uniform-fluid pair density, `g = ((2H/N)/vol_k)/rho`.
fn normalize(counts: &[u64], vols: &[f64], volume: f64, n_pairs: f64, mult: f64) -> Vec<f64> {
    if n_pairs == 0.0 {
        return vec![0.0; counts.len()];
    }
    let prefactor = mult * volume / n_pairs;
    counts
        .iter()
        .zip(vols.iter())
        .map(|(&h, &vol)| prefactor * (h as f64) / vol)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisContext;
    use crate::cell::UnitCell;
    use crate::data::frame::Frame;
    use crate::data::particle::ParticleData;
    use crate::exec::Par;

    /// Simple-cubic lattice of spacing `a`, `ncell` cells per side.
    ///
    /// SC rather than FCC because its shell structure is countable by hand: shell
    /// `m` has exactly `6m²` vectors, and every distance is `a·√(sum of squares)`,
    /// so a test can state the expected pair count instead of recomputing the
    /// formula under test.
    fn sc(ncell: usize, a: f64, ty: u32) -> Frame {
        let mut pd = ParticleData::with_capacity(ncell.pow(3));
        let mut id = 0i64;
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    pd.add_atom(
                        [i as f64 * a, j as f64 * a, k as f64 * a],
                        {
                            id += 1;
                            id
                        },
                        ty,
                    );
                }
            }
        }
        Frame::new(
            UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
            pd,
        )
    }

    /// Two well-separated SC blocks of the given types.
    fn two_blocks(ncell: usize, a: f64, ty_a: u32, ty_b: u32) -> Frame {
        let mut pd = ParticleData::with_capacity(2 * ncell.pow(3));
        let mut id = 0i64;
        let span = ncell as f64 * a;
        for (base, ty) in [(0.0, ty_a), (10.0 * span, ty_b)] {
            for i in 0..ncell {
                for j in 0..ncell {
                    for k in 0..ncell {
                        pd.add_atom(
                            [base + i as f64 * a, j as f64 * a, k as f64 * a],
                            {
                                id += 1;
                                id
                            },
                            ty,
                        );
                    }
                }
            }
        }
        Frame::new(
            UnitCell::orthorhombic(12.0 * span, 12.0 * span, 12.0 * span),
            pd,
        )
    }

    fn run(f: &Frame, rdf: &Rdf) -> Summary {
        run_with(f, rdf, 1)
    }

    fn run_with(f: &Frame, rdf: &Rdf, threads: usize) -> Summary {
        let nk = crate::analysis::NeighborhoodContext { cell: None };
        let par = Par::new(threads);
        let ctx = AnalysisContext {
            frame: f,
            nkeys: &nk,
            par: &par,
        };
        let mut store = crate::data::property::PropertyStore::new();
        let mut sc = Summary::default();
        rdf.run(&ctx, &mut store, &mut sc).unwrap();
        sc
    }

    /// Uniform two-type SC lattice (checkerboard) filling the whole cell.
    ///
    /// Uniform density matters for runtime, not physics: a clustered frame in
    /// a big box trips the cell list's bin-count cap, which raises the bin
    /// width far above the cutoff and turns each query into thousands of
    /// candidate checks. `ncell = 41` gives 68921 atoms — past `CHUNK_ATOMS`,
    /// so the reduction spans two chunks — while the 123 Å box at binsize 4.5
    /// Å asks for only 27³ bins, well under the atom count, so the cap never
    /// grows the bins.
    fn sc_two_types(ncell: usize, a: f64) -> Frame {
        let mut pd = ParticleData::with_capacity(ncell.pow(3));
        let mut id = 0i64;
        for k in 0..ncell {
            for j in 0..ncell {
                for i in 0..ncell {
                    let ty = if (i + j + k) % 2 == 0 { 1 } else { 2 };
                    pd.add_atom(
                        [i as f64 * a, j as f64 * a, k as f64 * a],
                        {
                            id += 1;
                            id
                        },
                        ty,
                    );
                }
            }
        }
        let box_len = ncell as f64 * a;
        Frame::new(UnitCell::orthorhombic(box_len, box_len, box_len), pd)
    }

    /// Design §10.2: fixed chunk ordinals plus ascending-order integer combines
    /// must give **bit-identical** output for any thread count. 41³ = 68921
    /// atoms crosses `CHUNK_ATOMS` (65536), so the pair loop really does split
    /// across chunks, and the checkerboard's two types cover same- and
    /// cross-type partial folds (1,1), (1,2), (2,2).
    #[test]
    fn rdf_is_bit_identical_across_thread_counts() {
        let f = sc_two_types(41, 3.0);
        let rdf = Rdf {
            rmax: 4.5,
            dr: 0.05,
            partials: true,
            coordination: true,
            ..Default::default()
        };
        let seq = run_with(&f, &rdf, 1);
        for threads in [2usize, 4, 8] {
            let par = run_with(&f, &rdf, threads);
            assert_eq!(
                seq.histograms.len(),
                par.histograms.len(),
                "threads={threads}: histogram count diverged"
            );
            for (name, a) in &seq.histograms {
                let b = par
                    .histograms
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("threads={threads}: missing histogram {name}"));
                assert_eq!(a, &b.1, "threads={threads}: histogram {name} diverged");
            }
            for (name, a) in &seq.scalars {
                let b = par
                    .scalars
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("threads={threads}: missing scalar {name}"));
                assert_eq!(a, &b.1, "threads={threads}: scalar {name} diverged");
            }
        }
    }

    fn hist<'a>(sc: &'a Summary, name: &str) -> &'a Vec<f64> {
        &sc.histograms
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| {
                panic!(
                    "no histogram {name}; have {:?}",
                    sc.histograms.iter().map(|(n, _)| n).collect::<Vec<_>>()
                )
            })
            .1
    }

    fn scalar(sc: &Summary, name: &str) -> f64 {
        sc.scalars
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no scalar {name}"))
            .1
    }

    /// Pairs inside `rmax` summed over bins whose centre is in `(lo, hi)`.
    fn pairs_in(counts: &[f64], dr: f64, lo: f64, hi: f64) -> f64 {
        counts
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                let r = (*k as f64 + 0.5) * dr;
                r > lo && r <= hi
            })
            .map(|(_, &v)| v)
            .sum()
    }

    /// Lattice vectors of simple-cubic spacing `a` within `rmax`, counted exactly.
    ///
    /// This is the reference the tests check against. One atom in a periodic box of
    /// side `L > 2*rmax` has exactly **one** neighbour per lattice vector with
    /// `|v| <= rmax` -- the box holds no second image of the same neighbour inside
    /// the range -- so the per-atom count is a plain vector count and the unordered
    /// pair count is `N/2` times that. Multiplying each vector by `(L/r)^3`
    /// would count images rather than neighbours, off by 729 on these fixtures.
    fn sc_neighbours_per_atom(a: f64, rmax: f64) -> f64 {
        let m = (rmax / a).ceil() as i64 + 1;
        let mut total = 0.0;
        for v in -m..=m {
            for w in -m..=m {
                for x in -m..=m {
                    if (v, w, x) == (0, 0, 0) {
                        continue;
                    }
                    let r = (a * a * (v * v + w * w + x * x) as f64).sqrt();
                    // Strictly below `rmax`, matching `Rdf::bin_of`: a pair sitting
                    // exactly at `rmax` falls outside the last bin.
                    if r < rmax {
                        total += 1.0;
                    }
                }
            }
        }
        total
    }

    /// Exact `g` for a bin holding `shell_vecs` lattice vectors' worth of pairs.
    ///
    /// `g_k = mult * H_k * V / (N^2 * vol_k)` with `H_k = shell_vecs * N / 2` on a
    /// half list, collapsed. Stated once here so every `g` assertion in this module
    /// is the same closed form and none of them can drift into re-deriving it.
    fn exact_g(shell_vecs: f64, n: f64, volume: f64, vol_k: f64) -> f64 {
        2.0 * (shell_vecs * n / 2.0) * volume / (n * n * vol_k)
    }

    /// Lattice vectors at exactly one shell radius (tolerance `eps`).
    fn sc_shell_vectors(a: f64, r: f64) -> f64 {
        sc_shell_count(a, r, 1e-9)
    }

    fn sc_shell_count(a: f64, r: f64, eps: f64) -> f64 {
        let m = (r / a).ceil() as i64 + 1;
        let mut total = 0.0;
        for v in -m..=m {
            for w in -m..=m {
                for x in -m..=m {
                    if (v, w, x) == (0, 0, 0) {
                        continue;
                    }
                    let d = (a * a * (v * v + w * w + x * x) as f64).sqrt();
                    if (d - r).abs() <= eps {
                        total += 1.0;
                    }
                }
            }
        }
        total
    }

    #[test]
    fn a_uniform_lattice_reports_the_exact_pair_count_inside_rmax() {
        // The raw count is the load-bearing one: every normalisation downstream is
        // a function of it, so it is checked against lattice-vector counting rather
        // than against another copy of the same loop over positions.
        let f = sc(15, 3.0, 1);
        let sc = run(&f, &Rdf::new(9.0, 0.1));
        let n = f.natoms() as f64;
        let expected = sc_neighbours_per_atom(3.0, 9.0) * n / 2.0;
        let got = scalar(&sc, "rdf_pairs");
        assert!(
            (got - expected).abs() < 1.0,
            "rdf_pairs={got}, lattice counting says {expected}"
        );
        let total: f64 = hist(&sc, "rdf_pair_counts").iter().sum();
        assert!((total - got).abs() < 1.0, "histogram must sum to rdf_pairs");
        assert!(
            got <= n * (n - 1.0) / 2.0,
            "a half list cannot see more pairs than exist"
        );
        assert_eq!(scalar(&sc, "rdf_pairs_possible"), n * (n - 1.0) / 2.0);
        assert_eq!(scalar(&sc, "rdf_atoms"), n);
        assert_eq!(scalar(&sc, "rdf_bins"), 90.0);
    }

    #[test]
    fn g_r_reads_one_on_a_uniform_lattice_and_the_coordination_number_the_shell_sum() {
        // g(r) = 1 is the statement that the OVITO normalisation and the
        // half-list factor agree. It only holds where the pair count at `r` equals
        // the uniform-fluid expectation `rho·4/3·pi·r³`, which needs `r` small
        // enough that `(L/r)³` is not truncated: at `r = 3√2` in a 45 A box the
        // discrete shells already run 5% below the continuum value. So the
        // assertion is on the *near-first-shell* region, where the exact count is
        // known, and the tolerance follows from the truncation rather than being
        // chosen to pass.
        let ncell = 15;
        let a = 3.0;
        let f = sc(ncell, a, 1);
        let rdf = Rdf::new(9.0, 0.1);
        let sc = run(&f, &rdf);
        let n = f.natoms() as f64;
        let volume = f.cell.volume();
        // Coordination number just past the first shell: exactly the lattice count,
        // independent of any continuum approximation.
        let cn = hist(&sc, "rdf_coordination_number");
        let at = |r: f64| ((r / 0.1).round() as usize).min(cn.len() - 1);
        let shell1 = sc_shell_vectors(a, a);
        let got = cn[at(3.5)];
        assert!(
            (got - shell1).abs() < 0.05,
            "n(3.5)={got}, first shell holds {shell1} per atom"
        );

        // g(r) is *not* 1 on a lattice: every shell is a delta spike sitting above
        // the continuum average, and the spike's height is fixed by the exact shell
        // population. `g(r=a)` must be `6 / (rho * vol_bin)` for the 6 first-shell
        // vectors, which is 13.86 here -- a value no plausible mis-normalisation
        // produces: `mult = 1` would halve it, a doubled pair list would double it,
        // and the transcription error described in the module docs puts it at 0.08.
        let g = hist(&sc, "rdf_g_r");
        let expect = exact_g(
            shell1,
            n,
            volume,
            rdf.shell_volume((a / rdf.dr).floor() as usize),
        );
        let gv = g[at(a)];
        assert!(
            (gv - expect).abs() < 1e-9,
            "g({a})={gv}, exact lattice prediction {expect}"
        );
        assert!(
            gv > 1.0,
            "a lattice shell is a spike above the continuum, got {gv}"
        );

        // Each further shell obeys the same law with its own population.
        for (r, shell) in [
            (
                3.0 * 2.0_f64.sqrt(),
                sc_shell_vectors(3.0, 3.0 * 2.0_f64.sqrt()),
            ),
            (
                3.0 * 3.0_f64.sqrt(),
                sc_shell_vectors(3.0, 3.0 * 3.0_f64.sqrt()),
            ),
        ] {
            let bin = (r / rdf.dr).floor() as usize;
            let e = exact_g(shell, n, volume, rdf.shell_volume(bin));
            let v = g[bin];
            assert!(
                (v - e).abs() < 1e-9,
                "g({r})={v}, exact lattice prediction {e} for {shell} vectors"
            );
        }

        // Averaged over a range holding many shells, g does come back to 1 -- that
        // is the sense in which "g(r) = 1 for a uniform system" is true, and it is
        // the reason the pointwise spikes above are not a bug. Checked as
        // `sum(g*vol)/(rho*vol_of_range)`, i.e. the mean pair density over the
        // region divided by the uniform value.
        let lo = 5usize;
        let hi = 70usize;
        let mass: f64 = (lo..hi).map(|k| g[k] * rdf.shell_volume(k)).sum();
        let vol: f64 = (lo..hi).map(|k| rdf.shell_volume(k)).sum();
        let mean_g = mass / vol;
        assert!(
            (mean_g - 1.0).abs() < 0.25,
            "volume-weighted mean g over many shells should approach 1, got {mean_g}"
        );
    }

    #[test]
    fn a_pair_exactly_on_a_bin_edge_goes_in_the_upper_bin() {
        // Both LAMMPS and OVITO use `(r1, r2]`. Reversing it shifts the whole
        // histogram by one bin, which is invisible in a smooth `g(r)` and visible
        // in every peak position.
        let rdf = Rdf::new(2.0, 0.5);
        assert_eq!(rdf.bin_of(0.5), Some(1));
        assert_eq!(rdf.bin_of(0.4999), Some(0));
        assert_eq!(rdf.bin_of(1.0), Some(2));
        assert_eq!(rdf.bin_of(2.0), None, "rmax is exclusive");
    }

    #[test]
    fn the_same_type_partial_is_doubled_and_mixed_ones_are_not() {
        // OVITO's `mult = 2 if i == j else 1`. Two well-separated blocks of one
        // type each: the (1,1) partial sees both blocks' first shells, the (1,2)
        // partial sees nothing at that range.
        let f = two_blocks(8, 3.0, 1, 2);
        let rdf = Rdf::new(4.0, 0.05);
        let sc = run(&f, &rdf);
        // First shells only, one per block. The blocks are 8 atoms wide inside a
        // 288 A box, so they have free surfaces and the interior's "6 neighbours per
        // atom" does not hold: a finite n³ SC block has `3*(n-1)*n²` bonds. Counting
        // `n³*6/2` instead would be counting images, not bonds.
        let n = 8.0f64;
        let expected = 2.0 * 3.0 * (n - 1.0) * n * n;
        let first = pairs_in(hist(&sc, "rdf_pair_counts"), 0.05, 2.9, 3.1);
        assert!((first - expected).abs() < 1.0, "{first} vs {expected}");
        assert!(
            hist(&sc, "rdf_g_1_1").iter().any(|&v| v > 0.0),
            "self partial must have content"
        );
        // The cross pair between the blocks is 240 A away, so there is no (1,2) pair
        // in range and the partial is **absent** rather than present-and-zero. That
        // convention is deliberate: a flat zero partial reads as "measured, nothing
        // there", which is a different claim from "not measured", and only one of
        // them is true.
        assert!(
            !sc.histograms.iter().any(|(n, _)| n == "rdf_g_1_2"),
            "a type pair with no pairs in range emits no partial: {:?}",
            sc.histograms.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mixed_partials_of_a_substitutional_alloy_match_their_own_pair_counts() {
        // The case that actually exposes a spurious diagonal doubling: in a
        // substitutional alloy every type pair exists, so `g_13` reads 2x its count
        // while `g_11` reads correctly, and the error is a uniform factor that no
        // shape check catches. Each partial is therefore compared against the pair
        // stream's own count for that type pair, at one shell.
        let ncell: usize = 12;
        let a = 3.0;
        let ntypes = 3i32;
        // Deterministic substitutional assignment: type by (i+j+k) mod 3, which is
        // a checkerboard-free pattern that still leaves every type pair abundant.
        let mut pd = ParticleData::with_capacity(ncell.pow(3));
        let mut id = 0i64;
        for i in 0..ncell {
            for j in 0..ncell {
                for k in 0..ncell {
                    let t = ((i + j + k) % ntypes as usize) as u32 + 1;
                    pd.add_atom(
                        [i as f64 * a, j as f64 * a, k as f64 * a],
                        {
                            id += 1;
                            id
                        },
                        t,
                    );
                }
            }
        }
        // Count first-shell pairs per type pair by brute force, so the expectation
        // for each partial is independent of the analysis being tested. Type of the
        // site `(i, j, k)` is `(i+j+k) mod ntypes + 1`.
        let ty = |i: i32, j: i32, k: i32| ((i + j + k).rem_euclid(ntypes)) as u32 + 1;
        let n32 = ncell as i32;
        let mut counts: std::collections::BTreeMap<(u32, u32), f64> = Default::default();
        for i in 0..n32 {
            for j in 0..n32 {
                for k in 0..n32 {
                    let ti = ty(i, j, k);
                    for (di, dj, dk) in [(1i32, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        let (ii, jj, kk) = (i + di, j + dj, k + dk);
                        // Wrap, because the neighbour across the boundary *is* the
                        // one at distance `a`: the frame is periodic, so an atom at
                        // `i = ncell-1` bonds to `i = 0`. Skipping out-of-range
                        // indices instead drops one bond per boundary atom -- 3
                        // faces x 12² = 432 of them here -- and misattributes the
                        // rest, so the expectation would disagree with a correct
                        // analysis by exactly that much.
                        let tj = ty(ii.rem_euclid(n32), jj.rem_euclid(n32), kk.rem_euclid(n32));
                        // The +x/+y/+z directions already give one representative
                        // of each bond, so this is an unordered count.
                        let key = if ti <= tj { (ti, tj) } else { (tj, ti) };
                        *counts.entry(key).or_default() += 1.0;
                    }
                }
            }
        }
        assert_eq!(
            counts.values().sum::<f64>(),
            n32 as f64 * n32 as f64 * n32 as f64 * 3.0
        );

        let f = Frame::new(
            UnitCell::orthorhombic(ncell as f64 * a, ncell as f64 * a, ncell as f64 * a),
            pd,
        );
        let rdf = Rdf::new(3.5, 0.05);
        let sc = run(&f, &rdf);
        let n = f.natoms() as f64;
        let volume = f.cell.volume();
        // Type populations for the `(i+j+k) mod 3` pattern on a 12³ grid: each type
        // has exactly a third of the atoms.
        let per_type = n / 3.0;
        let shell_vol = rdf.shell_volume((a / rdf.dr).round() as usize);
        for ((x, y), &h) in &counts {
            let mult = if x == y { 2.0 } else { 1.0 };
            // Same law as the total, with the pair population being the type pair:
            // `g = mult * H * V / (N1*N2 * vol)`.
            let expect = mult * h * volume / (per_type * per_type * shell_vol);
            let g = hist(&sc, &format!("rdf_g_{x}_{y}"))[(a / rdf.dr).round() as usize];
            assert!(
                (g - expect).abs() < 1e-9,
                "g_{x}_{y} at the first shell = {g}, expected {expect} for {h} pairs"
            );
        }
    }

    #[test]
    fn an_rmax_past_half_the_cell_is_refused_rather_than_folded() {
        // In a periodic box every atom has many images inside a large `rmax`, and
        // the minimum-image convention keeps only one per pair, so the histogram
        // would look fine and be wrong. That is what this refuses.
        let f = sc(6, 3.0, 1);
        let nk = crate::analysis::NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &par,
        };
        let mut store = crate::data::property::PropertyStore::new();
        let mut sc = Summary::default();
        let err = Rdf::new(9.5, 0.1)
            .run(&ctx, &mut store, &mut sc)
            .unwrap_err();
        assert!(
            err.to_string().contains("half the smallest cell height"),
            "{err}"
        );
        // Just inside the limit is allowed, so this is a boundary and not a blanket
        // rejection of large frames.
        let mut sc2 = Summary::default();
        let mut store2 = crate::data::property::PropertyStore::new();
        Rdf::new(9.0, 0.1).run(&ctx, &mut store2, &mut sc2).unwrap();
    }

    #[test]
    fn the_folding_limit_of_a_tilted_cell_is_half_its_height_not_its_edge() {
        // 18 Å box with the z vector tilted by 9 Å along y (`from_h` takes
        // lx, ly, lz, yz, xz, xy): edges 18, 18, 20.1; the y height is
        // V / |c × a| = 16.1 Å. Pairs fold past 8.05 Å, which the edge-based
        // limit (9 Å) would let through.
        let cell = UnitCell::from_h([18.0, 18.0, 18.0, 9.0, 0.0, 0.0]);
        let half = 0.5 * cell.heights().iter().copied().fold(f64::INFINITY, f64::min);
        assert!((half - 8.05).abs() < 0.01, "{half}");
        let err = Rdf::new(8.5, 0.1).validate(Some(&cell)).unwrap_err();
        assert!(
            err.to_string().contains("half the smallest cell height"),
            "{err}"
        );
        Rdf::new(8.0, 0.1).validate(Some(&cell)).unwrap();
    }

    #[test]
    fn invalid_ranges_are_refused() {
        let f = sc(6, 3.0, 1);
        let nk = crate::analysis::NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &par,
        };
        for (rmax, dr, needle) in [
            (3.0, 0.0, "bin width"),
            (3.0, -0.1, "bin width"),
            (3.0, f64::NAN, "bin width"),
            (3.0, f64::INFINITY, "bin width"),
            (0.0, 0.1, "rmax"),
            (-1.0, 0.1, "rmax"),
            (f64::NAN, 0.1, "rmax"),
            (f64::INFINITY, 0.1, "rmax must be positive and finite"),
        ] {
            let mut store = crate::data::property::PropertyStore::new();
            let mut sc = Summary::default();
            let rdf = Rdf::new(rmax, dr);
            let err = rdf.run(&ctx, &mut store, &mut sc).unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "rmax={rmax} dr={dr}: {err}"
            );
        }
    }

    #[test]
    fn partials_are_emitted_in_a_fixed_pair_order() {
        // Hash order would make the histogram names differ between runs while every
        // value stayed correct, which is the kind of change a test suite passes over.
        let mut pd = ParticleData::with_capacity(3);
        pd.add_atom([0.0, 0.0, 0.0], 1, 3);
        pd.add_atom([1.0, 0.0, 0.0], 2, 1);
        pd.add_atom([0.0, 1.0, 0.0], 3, 2);
        let f = Frame::new(UnitCell::orthorhombic(6.0, 6.0, 6.0), pd);
        let sc = run(&f, &Rdf::new(1.5, 0.1));
        // All three pairs are inside 1.5 (1.0 and sqrt(2)), so all three partials
        // are emitted -- a partial with no pairs in range is simply absent rather
        // than present-and-zero, which is why this asserts names, not values.
        let names: Vec<&str> = sc
            .histograms
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| n.starts_with("rdf_g_") && !n.ends_with("_r"))
            .collect();
        assert_eq!(names, vec!["rdf_g_1_2", "rdf_g_1_3", "rdf_g_2_3"]);
    }

    #[test]
    fn a_non_periodic_frame_has_no_image_folding_and_needs_no_rmax_check() {
        // The half-cell limit is a *periodic* statement: with pbc off there is one
        // image per pair and any `rmax` is representable. Refusing a large `rmax`
        // here would be wrong, and would be wrong for the void-analysis frames
        // that have free surfaces.
        let mut pd = ParticleData::with_capacity(2);
        pd.add_atom([0.0, 0.0, 0.0], 1, 1);
        pd.add_atom([20.0, 0.0, 0.0], 2, 1);
        let cell = UnitCell::orthorhombic(20.0, 20.0, 20.0).with_pbc([false, false, false]);
        let f = Frame::new(cell, pd);
        let nk = crate::analysis::NeighborhoodContext { cell: None };
        let par = Par::new(1);
        let ctx = AnalysisContext {
            frame: &f,
            nkeys: &nk,
            par: &par,
        };
        let mut store = crate::data::property::PropertyStore::new();
        let mut sc = Summary::default();
        Rdf::new(25.0, 0.5).run(&ctx, &mut store, &mut sc).unwrap();
        let counts = hist(&sc, "rdf_pair_counts");
        assert_eq!(pairs_in(counts, 0.5, 19.5, 20.5), 1.0);
    }

    /// An `n`×`n`×`layers` simple-cubic slab of spacing `a`, periodic in x
    /// and y, in a box twice its height along z (vacuum above it), with z
    /// periodic or open.
    fn sc_slab(n: usize, layers: usize, a: f64, z_periodic: bool) -> Frame {
        let mut pd = ParticleData::with_capacity(n * n * layers);
        let mut id = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..layers {
                    id += 1;
                    pd.add_atom([i as f64 * a, j as f64 * a, (k as f64 + 0.5) * a], id, 1);
                }
            }
        }
        let l = n as f64 * a;
        let cell = UnitCell::orthorhombic(l, l, 2.0 * layers as f64 * a)
            .with_pbc([true, true, z_periodic]);
        Frame::new(cell, pd)
    }

    #[test]
    fn periodic_bulk_under_auto_is_the_box_normalisation() {
        let f = sc(12, 2.5, 1);
        let mut boxed = Rdf::new(7.0, 0.1);
        boxed.volume = RdfVolume::Box;
        let auto = Rdf::new(7.0, 0.1);
        let (b, a) = (run(&f, &boxed), run(&f, &auto));
        assert_eq!(hist(&b, "rdf_g_r"), hist(&a, "rdf_g_r"));
        assert_eq!(scalar(&a, "rdf_volume_effective"), 0.0);
    }

    #[test]
    fn a_slab_in_vacuum_reads_as_bulk_with_the_effective_volume_and_edge_correction() {
        let a = 2.5;
        let bulk = run(&sc(16, a, 1), &Rdf::new(7.0, 0.1));
        for z_periodic in [true, false] {
            let f = sc_slab(16, 16, a, z_periodic);
            // The box volume halves the density and doubles g.
            let mut boxed = Rdf::new(7.0, 0.1);
            boxed.volume = RdfVolume::Box;
            let b = run(&f, &boxed);
            assert!((scalar(&b, "rdf_number_density") * a.powi(3) - 0.5).abs() < 1e-12);
            // Effective: the material's density, chosen without being asked.
            let e = run(&f, &Rdf::new(7.0, 0.1));
            assert_eq!(scalar(&e, "rdf_volume_effective"), 1.0, "{z_periodic}");
            let rho = scalar(&e, "rdf_number_density") * a.powi(3);
            assert!((rho - 1.0).abs() < 1e-3, "{z_periodic}: density {rho}");
            // Edge-corrected: every counted shell is whole, so g and n(r)
            // are bulk's, to the accuracy of the density.
            let mut edge = Rdf::new(7.0, 0.1);
            edge.edge_correct = true;
            let c = run(&f, &edge);
            assert!(scalar(&c, "rdf_centres") > 0.0);
            for (k, (x, y)) in hist(&c, "rdf_g_r")
                .iter()
                .zip(hist(&bulk, "rdf_g_r"))
                .enumerate()
            {
                assert!(
                    (x - y).abs() <= 1e-3 * y.max(1.0),
                    "{z_periodic} bin {k}: {x} vs {y}"
                );
            }
            let (nc, nb) = (
                scalar(&c, "rdf_coordination_at_rmax"),
                scalar(&bulk, "rdf_coordination_at_rmax"),
            );
            assert!((nc - nb).abs() < 1e-9, "{z_periodic}: n(rmax) {nc} vs {nb}");
        }
    }

    #[test]
    fn the_bin_count_and_edges_follow_a_non_integral_rmax() {
        // `rmax / dr` need not be a whole number, and the last bin is then short.
        // Its `g` must be scaled by its own volume, so the histogram stays
        // normalised -- a fixed-volume last bin would put a step in `g` at `rmax`.
        let rdf = Rdf::new(1.05, 0.5);
        assert_eq!(rdf.nbins(), 3);
        // The final edge is `rmax`, not `nbins * dr = 1.5`: reporting an edge past
        // `rmax` would invite a consumer to integrate over an empty range.
        assert_eq!(rdf.edges(), vec![0.0, 0.5, 1.0, 1.05]);
        // Bin 1 is [0.5, 1.0): volume is 4/3 pi (1 - 0.125), *not* 4/3 pi.
        assert_eq!(
            rdf.shell_volume(1),
            4.0 / 3.0 * std::f64::consts::PI * (1.0 - 0.125)
        );
        // The last bin is [1.0, 1.05], a sliver: volume must reflect that, or its `g`
        // is scaled down by the ratio of the nominal width to the real one. The
        // exact value is the point -- it is 1/10th the *width* but only ~1/5.5th
        // the volume, since shell volume goes as r².
        assert_eq!(
            rdf.shell_volume(2),
            4.0 / 3.0 * std::f64::consts::PI * (1.05f64.powi(3) - 1.0)
        );
        assert!(
            rdf.shell_volume(2) < rdf.shell_volume(1),
            "the sliver must be smaller than a full bin"
        );
        // A bin past the last is empty rather than negative.
        assert_eq!(rdf.shell_volume(3), 0.0);
        assert_eq!(rdf.bin_of(1.04), Some(2));
        assert_eq!(rdf.bin_of(1.5), None);
        assert_eq!(rdf.bin_of(1.05), None, "rmax is exclusive");
    }
}
