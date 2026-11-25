//! Linked-cell (bin) list — the spatial index workhorse (design §9.2).
//!
//! Ports the LAMMPS binned-neighbour scheme: dense `binhead` + `bins` chains.
//! Bins tile fractional space, sized relative to the cutoff; the stencil
//! reach is `⌈cutoff / min binwidth⌉` bins (at least 1).

use crate::cell::{Coord3, UnitCell};
use crate::error::{Error, Result};

/// Local per-atom index: the chain link type of the dense bin lists.
pub type MdaLocal = u32;

/// Chain terminator for the `u32` bin lists: `-1` reinterpreted.
const NONE: MdaLocal = MdaLocal::MAX;

/// Decode a bin-list entry back to the signed form, restoring the `-1`
/// terminator (`u32::MAX` does *not* wrap to -1 through a widening cast).
#[inline]
fn as_i64(v: MdaLocal) -> i64 {
    if v == NONE {
        -1
    } else {
        v as i64
    }
}

/// A cell list over periodic wrapped coordinates.
pub struct CellList<'a> {
    cell: UnitCell,
    /// Positions as given, borrowed from the frame being indexed. The list
    /// never copies them — at 128M atoms a private copy is 3.07 GB — and
    /// nothing in the crate mutates positions while a list is alive, so the
    /// borrow is sound (the list costs only the 8 B/atom of its bin chains;
    /// memory budget, design §24.2).
    positions: &'a [Coord3],
    /// `(H^-1)^T`, hoisted so `bin_of` does not invert the cell per point.
    inv_t: crate::cell::Mat3,
    /// Effective per-axis bin width measured along that axis' perpendicular
    /// direction; each is >= the requested binsize, which is what makes the
    /// +/-1 stencil sufficient for any cell shape.
    binwidth: Coord3,
    /// Bin counts per dim.
    nx: usize,
    ny: usize,
    nz: usize,
    /// Stencil reach in bins per axis (includes ±r window).
    reach: i64,
    /// binhead[lin] first atom in bin; bins[i] next. Both are per-atom /
    /// per-bin `u32` chains (LAMMPS `binhead`/`bins`), which is what keeps the
    /// structure at ~8 B/atom; `-1` is stored as `u32::MAX` because an `i64`
    /// sentinel would double the footprint.
    binhead: Vec<MdaLocal>,
    bins: Vec<MdaLocal>,
    /// Per-linear-bin atom count, only materialised on request: it is a dense
    /// per-bin array, so at 128M atoms it would add ~0.5 GB to a build that
    /// only wants it for diagnostics.
    counts: Option<Vec<u32>>,
}

impl<'a> CellList<'a> {
    /// Build from wrapped positions, bins sized to the cutoff (design §9.2),
    /// giving a 3³ stencil.
    pub fn build(cell: &UnitCell, positions: &'a [Coord3], cutoff: f64) -> Result<CellList<'a>> {
        Self::build_with_binsize(cell, positions, cutoff, cutoff)
    }

    /// Build with a requested minimum bin size.
    ///
    /// Bins tile the cell's **fractional space**, one lattice period per axis,
    /// as LAMMPS does for triclinic cells: with a perpendicular bin width >=
    /// cutoff, two atoms within `cutoff` differ by at most one bin index per
    /// axis. The actual width per axis is `height / floor(height / binsize)`,
    /// i.e. rounded *up* from the request, which preserves that guarantee while
    /// tiling the cell exactly.
    pub fn build_with_binsize(
        cell: &UnitCell,
        positions: &'a [Coord3],
        cutoff: f64,
        binsize: f64,
    ) -> Result<CellList<'a>> {
        Self::build_inner(cell, positions, cutoff, binsize, false)
    }

    /// As [`CellList::build_with_binsize`], but also materialises the
    /// per-bin occupancy counts used by diagnostics and tests.
    pub fn build_with_counts(
        cell: &UnitCell,
        positions: &'a [Coord3],
        cutoff: f64,
        binsize: f64,
    ) -> Result<CellList<'a>> {
        Self::build_inner(cell, positions, cutoff, binsize, true)
    }

    fn build_inner(
        cell: &UnitCell,
        positions: &'a [Coord3],
        cutoff: f64,
        binsize: f64,
        want_counts: bool,
    ) -> Result<CellList<'a>> {
        if cutoff <= 0.0 {
            return Err(Error::invalid("cutoff must be > 0"));
        }
        if binsize <= 0.0 {
            return Err(Error::invalid("binsize must be > 0"));
        }
        // Bins tile *fractional* space, one unit per lattice period along each axis.
        //
        // Tiling the Cartesian bounding box instead, on the reasoning that a bin width >= cutoff
        // means two atoms within `cutoff` differ by at most one bin index per axis, holds only if
        // the bin axes are the metric axes, and they are not: the box of a rotated cell is a
        // Cartesian box whose edges are not lattice vectors, so bin adjacency is not metric
        // adjacency. In a 16 Å FCC block turned by 0.35 rad, two of twelve first-shell neighbours
        // at 2.828 Å sit one bin away and the other ten sit two to four bins away — the box's
        // diagonal direction has no bin boundary anywhere near it. A ±1 stencil finds the five
        // whose wrapped coordinates happen to be nearby and silently misses the rest.
        //
        // Fractional space has no such gap. A bin edge is a lattice plane, so `rem_euclid` on a bin
        // index is exactly a lattice image of the whole bin, and two atoms within `cutoff` differ by
        // at most one bin per axis whenever the bin's perpendicular width is >= cutoff — which
        // `floor(height / binsize)` guarantees by rounding the width up. This is what LAMMPS does for
        // triclinic cells.
        let heights = cell.heights();
        let mut span = [0.0f64; 3];
        let mut n = [1usize; 3];
        for d in 0..3 {
            span[d] = heights[d];
            if !span[d].is_finite() || span[d] <= 0.0 {
                // degenerate axis (e.g. a 2d slab): a single bin
                span[d] = binsize;
                n[d] = 1;
            } else {
                n[d] = ((span[d] / binsize).floor() as usize).max(1);
            }
        }

        // Cap the bin count at ~1 bin per atom. `binhead`/`counts` are dense
        // arrays, so a huge box with a small cutoff would otherwise allocate
        // (span/binsize)^3 entries for a handful of atoms — a 4000 Å box at
        // cutoff 3 Å asks for 2.4e9 bins (~19 GB) for 10k atoms. LAMMPS makes
        // the same trade in `Neighbor::setup_bins`: the bin size is *raised*
        // above the cutoff when the box is far larger than the atom count, and
        // the stencil reach grows with it, so correctness is preserved.
        //
        // The ratio is taken in floating point, because the count it is taken
        // from does not have to fit in an integer: a 1e30 Å box at cutoff 3 Å
        // asks for 3.3e29 bins on an axis, and the product of three of those
        // overflows a `u64` long before it overflows an `f64`.
        let wanted = (n[0] as f64) * (n[1] as f64) * (n[2] as f64);
        let cap = (positions.len() as f64).max(1.0);
        if wanted > cap {
            // raise the binsize by the cube root of the excess
            let factor = (wanted / cap).cbrt();
            let grown = binsize * factor * (1.0 + 1e-9);
            for d in 0..3 {
                if span[d] > 0.0 {
                    n[d] = ((span[d] / grown).floor() as usize).max(1);
                }
            }
        }
        // **And the count has to be a number a `usize` holds**, whatever the
        // arithmetic above did. The growth brings the count down to about one
        // bin per atom, but it does so through a bin *size*, and the size
        // cannot be raised past the box: an axis shorter than the request is a
        // single bin of that shorter height, and a box of 1e30 Å with a cutoff
        // of 3 Å is asking for a single bin of 3.3e29 Å. So the axes are
        // scaled down here until the product fits, and the bin width — and
        // with it the stencil reach — follows. Without this the count
        // overflows the `usize` it is about to be used as an array length.
        let cap_bins = (positions.len() as u128).max(1);
        loop {
            let total = (n[0] as u128) * (n[1] as u128) * (n[2] as u128);
            if total <= cap_bins {
                break;
            }
            // Halve the largest axis, which is where the excess is.
            let mut d = 0usize;
            for k in 1..3 {
                if n[k] > n[d] {
                    d = k;
                }
            }
            if n[d] <= 1 {
                break;
            }
            n[d] = (n[d] / 2).max(1);
        }
        let binwidth = [
            span[0] / n[0] as f64,
            span[1] / n[1] as f64,
            span[2] / n[2] as f64,
        ];
        // Every width is >= binsize by the floor above — unless the axis is shorter than the
        // request, in which case it is a single bin of that shorter height. Keep the guarantee
        // explicit rather than inherited, because the stencil size is derived from it.
        debug_assert!(
            (0..3).all(|d| binwidth[d] >= binsize.min(span[d]) - 1e-9),
            "bin width {binwidth:?} below the request for spans {span:?}"
        );

        // With binwidth >= cutoff, +/-1 suffices; a bin narrower than the
        // cutoff (a binsize below it, or an axis shorter than the request)
        // widens the reach to match.
        let reach = (cutoff / binwidth[0].min(binwidth[1]).min(binwidth[2])).ceil() as i64;
        let reach = reach.max(1);

        let (nx, ny, nz) = (n[0], n[1], n[2]);
        let mut cl = CellList {
            cell: *cell,
            positions,
            inv_t: cell.inverse_transposed()?,
            binwidth,
            nx,
            ny,
            nz,
            reach,
            binhead: vec![NONE; nx * ny * nz],
            bins: vec![NONE; positions.len()],
            counts: if want_counts {
                Some(vec![0u32; nx * ny * nz])
            } else {
                None
            },
        };
        cl.fill();
        Ok(cl)
    }

    /// Bin index of a point, in fractional space.
    ///
    /// Periodic axes wrap, which is what makes an index step a lattice image; non-periodic axes
    /// clamp, so an atom outside the box (a free surface, unwrapped input) shares an edge bin with
    /// the atoms it can actually reach and is filtered out by the cutoff rather than lost.
    #[inline]
    fn bin_of(&self, p: Coord3) -> (i64, i64, i64) {
        let lam = self.cell.x2lamda_pre(p, &self.inv_t);
        let idx = |d: usize, n: usize| -> i64 {
            let n = n as i64;
            if self.cell.pbc[d] {
                ((lam[d] * n as f64).floor() as i64).rem_euclid(n)
            } else {
                ((lam[d] * n as f64).floor() as i64).clamp(0, n - 1)
            }
        };
        (idx(0, self.nx), idx(1, self.ny), idx(2, self.nz))
    }

    /// Wrap a bin index into the periodic grid (rem_euclid in periodic dims);
    /// returns an error if a non-periodic dim asks for an out-of-range bin.
    pub fn wrap_bin(&self, ix: i64, iy: i64, iz: i64) -> Result<(i64, i64, i64)> {
        let (nx, ny, nz) = (self.nx as i64, self.ny as i64, self.nz as i64);
        let pbc = self.cell.pbc;
        let wx = if pbc[0] {
            ix.rem_euclid(nx)
        } else if ix < 0 || ix >= nx {
            return Err(Error::invalid("bin out of range in non-periodic x"));
        } else {
            ix
        };
        let wy = if pbc[1] {
            iy.rem_euclid(ny)
        } else if iy < 0 || iy >= ny {
            return Err(Error::invalid("bin out of range in non-periodic y"));
        } else {
            iy
        };
        let wz = if pbc[2] {
            iz.rem_euclid(nz)
        } else if iz < 0 || iz >= nz {
            return Err(Error::invalid("bin out of range in non-periodic z"));
        } else {
            iz
        };
        Ok((wx, wy, wz))
    }

    /// Linear bin index for wrapped in-range indices.
    pub fn lin(&self, ix: i64, iy: i64, iz: i64) -> Option<usize> {
        let (nx, ny, nz) = (self.nx as i64, self.ny as i64, self.nz as i64);
        if ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz {
            return None;
        }
        Some(((ix * ny + iy) * nz + iz) as usize)
    }

    /// Distinct bin indices along one axis covered by a stencil of half-width
    /// `r` around `home`.
    ///
    /// Wrapping makes `home ± k` alias when the grid has fewer than `2r+1`
    /// bins; visiting the raw stencil would then report the same bin (and
    /// hence the same atoms) several times. Deduplicating per axis before the
    /// triple loop keeps every candidate atom visited exactly once.
    pub fn stencil_axis(&self, axis: usize, home: i64, r: i64) -> Vec<i64> {
        let n = match axis {
            0 => self.nx as i64,
            1 => self.ny as i64,
            _ => self.nz as i64,
        };
        let periodic = self.cell.pbc[axis];
        let mut out: Vec<i64> = Vec::with_capacity((2 * r + 1) as usize);
        for k in -r..=r {
            let raw = home + k;
            let v = if periodic {
                raw.rem_euclid(n)
            } else if raw < 0 || raw >= n {
                continue;
            } else {
                raw
            };
            if !out.contains(&v) {
                out.push(v);
            }
            if out.len() as i64 == n {
                break; // whole axis covered
            }
        }
        out
    }

    /// Head of bin chain at linear index.
    #[inline]
    /// First atom in a bin, or -1 when empty.
    pub fn bin_head(&self, lin: usize) -> i64 {
        as_i64(self.binhead[lin])
    }

    /// Next atom in bin chain.
    #[inline]
    /// Next atom in the chain, or -1 at the end.
    pub fn bin_next(&self, atom: usize) -> i64 {
        as_i64(self.bins[atom])
    }

    /// Minimum-image helper forwarding to the cell.
    #[inline]
    pub fn min_image(&self, dr: Coord3) -> Result<Coord3> {
        self.cell.min_image(dr)
    }

    fn fill(&mut self) {
        let mut head = std::mem::take(&mut self.binhead);
        let mut bins = std::mem::take(&mut self.bins);
        let mut counts = self.counts.take();
        for (i, &p) in self.positions.iter().enumerate() {
            let (ix, iy, iz) = self.bin_of(p);
            let lin = self.lin(ix, iy, iz).expect("in-range bin");
            bins[i] = head[lin];
            head[lin] = i as MdaLocal;
            if let Some(c) = counts.as_mut() {
                c[lin] += 1;
            }
        }
        self.binhead = head;
        self.bins = bins;
        self.counts = counts;
    }

    pub fn nx(&self) -> usize {
        self.nx
    }
    pub fn ny(&self) -> usize {
        self.ny
    }
    pub fn nz(&self) -> usize {
        self.nz
    }
    pub fn nbins(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    pub fn reach(&self) -> i64 {
        self.reach
    }
    /// Largest query cutoff this list answers correctly.
    ///
    /// The candidate stencil is sized once, at build time, from the *effective*
    /// bin width -- which is not the requested cutoff: the bin count is capped at
    /// roughly one bin per atom, and the width is raised above the request when
    /// that cap bites. The stencil therefore covers `reach * min(binwidth)`, and
    /// querying wider than that returns a truncated neighbour set silently rather
    /// than as an error.
    ///
    /// Callers that want to reuse a shared list instead of building their own
    /// should check their cutoff against this before querying.
    pub fn max_query_cutoff(&self) -> f64 {
        let w = self.binwidth[0].min(self.binwidth[1]).min(self.binwidth[2]);
        self.reach as f64 * w
    }

    pub fn binsize(&self) -> f64 {
        self.binwidth[0].min(self.binwidth[1]).min(self.binwidth[2])
    }

    /// Occupancy of a linear bin, or 0 when counts were not materialised
    /// (see [`CellList::build_with_counts`]).
    pub fn bin_count(&self, lin: usize) -> u32 {
        self.counts
            .as_ref()
            .and_then(|c| c.get(lin))
            .copied()
            .unwrap_or(0)
    }

    /// Whether the per-bin counts were materialised.
    pub fn counts_built(&self) -> bool {
        self.counts.is_some()
    }

    pub fn positions(&self) -> &'a [Coord3] {
        self.positions
    }

    /// Number of extra periodic images to scan per axis for a cutoff radius.
    ///
    /// Two images of the same atom can only both lie inside the cutoff if the
    /// cell step is at most `2·cutoff` long, so per axis we need
    /// `|m| ≤ 2·cutoff/height_d` extra steps around the pair's minimum image.
    /// This yields the cheap single-image case (`0`) for the normal
    /// `cutoff < half cell` regime and the LAMMPS-style repeated neighbour
    /// entries for small cells.
    pub fn image_shift_range(&self, cutoff: f64) -> [i64; 3] {
        let heights = self.cell.heights();
        let mut out = [0i64; 3];
        for d in 0..3 {
            if !self.cell.pbc[d] || heights[d] <= 0.0 {
                continue;
            }
            out[d] = (2.0 * cutoff / heights[d]).floor() as i64;
        }
        out
    }

    /// All neighbours of atom `i` within `cutoff`, as `(j, dr, shift)` triples.
    ///
    /// `shift` is the periodic image of `j` relative to `i` (zero for the
    /// common single-image case), so the same neighbour atom can legitimately
    /// appear more than once when the cutoff exceeds half the cell width —
    /// exactly what LAMMPS emits, with the image recoverable from the shift.
    pub fn neighbor_entries(
        &self,
        i: usize,
        cutoff: f64,
        out: &mut Vec<(usize, Coord3, [i64; 3])>,
    ) -> Result<()> {
        out.clear();
        self.scan_within(self.positions[i], cutoff, Some(i), |j, dr, shift| {
            out.push((j, dr, shift))
        })
    }

    /// Core candidate scan, pushing each accepted neighbour straight into the
    /// caller's sink. A closure sink instead of an intermediate `Vec` because
    /// this runs once per *atom* in every hot analysis loop: a fresh triple
    /// buffer per query (grown to the neighbour count and dropped), re-mapped
    /// into the caller's buffer, would cost two copies and one allocation per
    /// atom, where the caller's `out` already retains its capacity across
    /// queries.
    ///
    /// The stencil and per-axis dedup are computed once per query; `visit`
    /// fires only for pairs inside `cutoff`, so the indirect call costs ~1 per
    /// accepted neighbour, not per candidate.
    fn scan_within<F>(
        &self,
        p: Coord3,
        cutoff: f64,
        skip: Option<usize>,
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(usize, Coord3, [i64; 3]),
    {
        let (ix, iy, iz) = self.bin_of(p);
        let r = self.reach;
        let cut2 = cutoff * cutoff;

        // dedupe the stencil per axis so each candidate atom is visited once
        let ax = self.stencil_axis(0, ix, r);
        let ay = self.stencil_axis(1, iy, r);
        let az = self.stencil_axis(2, iz, r);
        let sh = self.image_shift_range(cutoff);
        // The cell matrix is loop-invariant, and the candidate loops call this
        // per candidate: hoisting the inversion drops ~7% of executed
        // instructions, though that turned out not to move wall clock, since
        // this loop is stalled on `positions` rather than short of ALU work.
        let inv_t = self.cell.inverse_transposed()?;

        for &wx in &ax {
            for &wy in &ay {
                for &wz in &az {
                    let Some(lin) = self.lin(wx, wy, wz) else {
                        continue;
                    };
                    let mut j = as_i64(self.binhead[lin]);
                    while j >= 0 {
                        let jj = j as usize;
                        if Some(jj) != skip {
                            let pj = self.positions[jj];
                            let raw = [pj[0] - p[0], pj[1] - p[1], pj[2] - p[2]];
                            // Start from the minimum image, then walk the
                            // images that can also fall inside the cutoff.
                            //
                            // `min_image_with_shift_pre` rounds only the
                            // periodic λ components, so a free-surface axis
                            // keeps the raw displacement without any patch-up
                            // here.
                            let (base, n0) = self.cell.min_image_with_shift_pre(raw, &inv_t);
                            let rx = (n0[0] - sh[0])..=(n0[0] + sh[0]);
                            let ry = (n0[1] - sh[1])..=(n0[1] + sh[1]);
                            let rz = (n0[2] - sh[2])..=(n0[2] + sh[2]);
                            for tx in rx {
                                for ty in ry.clone() {
                                    for tz in rz.clone() {
                                        let d = self.cell.shift_vector([
                                            tx - n0[0],
                                            ty - n0[1],
                                            tz - n0[2],
                                        ]);
                                        let dr = [base[0] + d[0], base[1] + d[1], base[2] + d[2]];
                                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                                        if d2 <= cut2 {
                                            visit(jj, dr, [tx, ty, tz]);
                                        }
                                    }
                                }
                            }
                        }
                        j = as_i64(self.bins[jj]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Neighbours of `i` within `cutoff` using the single nearest image per
    /// neighbour atom. Equivalent to [`CellList::neighbor_entries`] whenever the
    /// cutoff is smaller than every cell width, which is the normal case.
    ///
    /// `out` is cleared but its capacity kept, so a loop over atoms allocates
    /// it once and then reuses it — no per-query buffer.
    pub fn neighbors_of(
        &self,
        i: usize,
        cutoff: f64,
        out: &mut Vec<(usize, Coord3)>,
    ) -> Result<()> {
        out.clear();
        self.scan_within(self.positions[i], cutoff, Some(i), |j, dr, _| {
            out.push((j, dr))
        })
    }

    /// Atoms within `cutoff` of an arbitrary query point `p`, as `(j, dr)` pairs.
    ///
    /// [`CellList::neighbors_of`] can only ask about an atom the list was built
    /// over, which excludes every query that is not one of its own positions —
    /// the points on a Wigner-Seitz cell boundary, a scan line, a grid sample.
    ///
    /// `cutoff` must not exceed the cutoff the list was built at. The candidate
    /// stencil is sized once, at build time, so a wider query returns a truncated
    /// neighbour set rather than an error, and `dr` is relative to `p` rather than
    /// to the query atom, so the caller gets a direction rather than an index.
    pub fn neighbors_within_point(
        &self,
        p: Coord3,
        cutoff: f64,
        out: &mut Vec<(usize, Coord3)>,
    ) -> Result<()> {
        out.clear();
        self.scan_within(p, cutoff, None, |j, dr, _| out.push((j, dr)))
    }

    /// Count of neighbour pairs with centre `i` within `cutoff` (useful for the
    /// neighbour census; allocates one scratch `Vec` per call).
    pub fn degree(&self, i: usize, cutoff: f64) -> Result<usize> {
        let mut scratch = Vec::new();
        self.neighbors_of(i, cutoff, &mut scratch)?;
        Ok(scratch.len())
    }

    /// Bin index accessor (public, for tests).
    pub fn bin_of_pub(&self, p: Coord3) -> (i64, i64, i64) {
        self.bin_of(p)
    }

    pub fn cell(&self) -> &UnitCell {
        &self.cell
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Mat3;

    fn dot(a: Coord3, b: Coord3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn small_fcc_positions() -> Vec<Coord3> {
        // 2x2x2 conventional fcc unit cells in a 10x10x10 box
        let mut v = Vec::new();
        for ix in 0..2 {
            for iy in 0..2 {
                for iz in 0..2 {
                    v.push([ix as f64 * 5.0, iy as f64 * 5.0, iz as f64 * 5.0]);
                }
            }
        }
        v
    }

    /// Norm of a displacement vector. Local to the tests so the module's private helpers are
    /// not pulled into the production surface just to check an assertion.
    fn norm(v: Coord3) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn sub3(a: Coord3, b: Coord3) -> Coord3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn add3(a: Coord3, b: Coord3) -> Coord3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    #[test]
    fn bins_cover_positions() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build_with_counts(&cell, &pos, 3.0, 3.0).unwrap();
        // every atom in exactly one bin
        let total: u32 = (0..cl.nbins()).map(|b| cl.bin_count(b)).sum();
        assert_eq!(total as usize, pos.len());
    }

    #[test]
    fn finds_nn() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 3.0).unwrap();
        let mut scratch = Vec::new();
        cl.neighbors_of(0, 3.0, &mut scratch).unwrap();
        // atom 0 at (0,0,0); its min-image neighbours: (5,0,0),(0,5,0),(0,0,5),
        // and the periodic images (5,5,0),(5,0,5),(0,5,5),(5,5,5) are all >3? no:
        // nearest = 5 > 3 => zero neighbours within 3.0
        assert_eq!(scratch.len(), 0);
    }

    /// A box too large to bin does not overflow the bin count.
    ///
    /// The bin count is a dense array's length, and the count a 1e30 Å box asks
    /// for at a 3 Å cutoff does not fit in a `usize`. The guard that keeps the
    /// count near one bin per atom cannot get there through the bin *size*,
    /// because the size cannot be raised past the box: an axis shorter than the
    /// request is a single bin of that shorter height. So the axes are scaled
    /// down until the product fits -- and the neighbours are still found.
    #[test]
    fn a_box_too_large_to_bin_still_builds() {
        let cell = UnitCell::orthorhombic(1e30, 1e30, 1e30);
        let pos = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let cl = CellList::build(&cell, &pos, 3.0).unwrap();
        let mut scratch = Vec::new();
        cl.neighbors_of(0, 3.0, &mut scratch).unwrap();
        assert_eq!(scratch.len(), 2, "both neighbours are within the cutoff");
    }

    /// Binning in fractional space, not in a Cartesian bounding box.
    ///
    /// Tiling the Cartesian box, on the argument that a bin width `>= cutoff` means two atoms
    /// within `cutoff` differ by at most one bin index per axis, is right only when the bin axes
    /// are the metric axes. In a rotated cell they are not, and the failure is not subtle at the
    /// edges — it opens a hole in the middle of the structure. Turn a 16 Å FCC block by 0.35 rad
    /// and the first shell at 2.828 Å sits one bin away along the lattice axes but two to four
    /// bins away along the box diagonal, where the lattice direction happens to lie halfway
    /// between two bin planes and so has no boundary near it at all. Five of the twelve
    /// neighbours come back; seven are silently absent.
    ///
    /// Nothing about the query is wrong, the positions are right, and every reported distance is
    /// correct — so such a defect shows up three layers away, as a perfect crystal measuring
    /// 1392 A^3 of void. The check is completeness against brute force, on every atom, because a
    /// defect that drops some neighbours still returns plausible ones.
    #[test]
    fn a_rotated_cell_finds_every_first_shell_neighbour() {
        let a = 4.0_f64;
        let cells = 4_i64;
        let ang = 0.35_f64;
        let rot = Mat3::from_rows(
            [
                a * cells as f64 * ang.cos(),
                a * cells as f64 * ang.sin(),
                0.0,
            ],
            [
                -a * cells as f64 * ang.sin(),
                a * cells as f64 * ang.cos(),
                0.0,
            ],
            [0.0, 0.0, a * cells as f64],
        );
        let base = a * cells as f64;
        let cell = UnitCell::from_matrix(rot, [true; 3], [0.0; 3]);
        // `(a/2)(i,j,k)` with `i+j+k` even: the conventional FCC lattice.
        let n = 2 * cells;
        let sites: Vec<Coord3> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .flat_map(|(i, j)| (0..n).map(move |k| (i, j, k)))
            .filter(|(i, j, k)| (i + j + k) % 2 == 0)
            .map(|(i, j, k)| {
                // Two conversions, and both are silent when wrong.
                //
                // The coordinates are *fractional*, not absolute: the box edge is `a * cells`, so
                // `(a/2)(i,j,k)` sits at `(a/2)(i,j,k)/(a*cells)` fractionally. Read as absolute the
                // lattice comes out `cells` times too dense and stops being commensurate with the
                // box that is supposed to contain it.
                //
                // And they go through `lamda2x`, not `Mat3::mul_v`. The cell reads rows as lattice
                // vectors so `lamda2x` sums over rows; `mul_v` sums over columns. The two are
                // transposes, equal only for a symmetric matrix, so a block built with one and
                // tiled with the other is not a lattice of the declared cell at all. It fails
                // quietly — distinct sites coincide, the first shell lands at 1.22 A instead of
                // 2.83 A, and the neighbour search takes the blame for it.
                cell.lamda2x([
                    a / 2.0 * i as f64 / base,
                    a / 2.0 * j as f64 / base,
                    a / 2.0 * k as f64 / base,
                ])
            })
            .collect();
        assert_eq!(sites.len(), 4 * cells.pow(3) as usize);

        let first_shell = a / 2.0 * 2f64.sqrt(); // a / sqrt(2)
        let cutoff = first_shell * 1.05;
        let cl = CellList::build(&cell, &sites, cutoff).unwrap();
        let positions = cl.positions();

        for i in 0..positions.len() {
            // Brute force over every image, via the minimum image of each pair.
            let mut expected = 0usize;
            for (j, q) in positions.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dr = [
                    q[0] - positions[i][0],
                    q[1] - positions[i][1],
                    q[2] - positions[i][2],
                ];
                let mi = cell.min_image(dr).unwrap();
                if dot(mi, mi).sqrt() < first_shell * 1.005 {
                    expected += 1;
                }
            }
            assert_eq!(expected, 12, "fixture must be a perfect FCC lattice");

            let mut pairs: Vec<(usize, Coord3)> = Vec::new();
            cl.neighbors_of(i, cutoff, &mut pairs).unwrap();
            let got: Vec<usize> = pairs.iter().map(|p| p.0).collect();
            assert_eq!(
                got.len(),
                12,
                "atom {i} found {} of its 12 first-shell neighbours",
                got.len()
            );
            for j in got {
                let dr = [
                    positions[j][0] - positions[i][0],
                    positions[j][1] - positions[i][1],
                    positions[j][2] - positions[i][2],
                ];
                let mi = cell.min_image(dr).unwrap();
                let d = dot(mi, mi).sqrt();
                assert!(
                    (d - first_shell).abs() < 1e-9,
                    "atom {i} neighbour {j} at {d}, expected {first_shell}"
                );
            }
        }
    }

    #[test]
    fn finds_bonded_with_larger_cut() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 5.1).unwrap();
        let mut scratch = Vec::new();
        cl.neighbors_of(0, 5.1, &mut scratch).unwrap();
        // corner atom at (0,0,0) in a 10 A box, cutoff 5.1 > half the box: the
        // three lattice neighbours at distance 5 each appear through *two*
        // images (+a and -a are both 5 away), which is what LAMMPS reports.
        // Face diagonals sit at sqrt(50) > 5.1 and are excluded.
        assert_eq!(scratch.len(), 6);
        for (_, dr) in &scratch {
            let d = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
            assert!((d - 5.0).abs() < 1e-9, "unexpected distance {d}");
        }
    }

    #[test]
    fn images_carry_distinct_shifts() {
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 5.1).unwrap();
        let mut entries = Vec::new();
        cl.neighbor_entries(0, 5.1, &mut entries).unwrap();
        // atom 1 sits at (0,0,5); the two qualifying images are the atom
        // itself (shift 0) and its -z image (shift [0,0,-1]).
        let mut shifts_for_1: Vec<[i64; 3]> = entries
            .iter()
            .filter(|(j, _, _)| *j == 1)
            .map(|(_, _, s)| *s)
            .collect();
        shifts_for_1.sort();
        assert_eq!(shifts_for_1, vec![[0, 0, -1], [0, 0, 0]]);
    }

    #[test]
    fn single_image_regime_is_unaffected() {
        // cutoff well below half the cell: one image per pair, no repeats
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 5.1).unwrap();
        assert_eq!(cl.image_shift_range(4.0), [0, 0, 0]);
        let mut entries = Vec::new();
        cl.neighbor_entries(0, 4.0, &mut entries).unwrap();
        assert!(entries.is_empty());
        // 2r >= h triggers the multi-image scan on every axis
        assert_eq!(cl.image_shift_range(5.1), [1, 1, 1]);
    }

    // ---------------------------------------------------------------------
    // Arbitrary-point queries (`neighbors_within_point`).
    //
    // This entry point is what `analysis::void_b` samples the phi field through, and it is the
    // one query that can be asked about a point the list was never built around. Its contract
    // has two halves that fail differently: the *candidate set* must not miss an atom inside
    // the cutoff (a miss raises the minimum and invents void), and the reported `dr` must be
    // the displacement from the query point rather than from some other atom — a caller using
    // the direction gets silently wrong geometry if it is off by the query atom's position.
    // ---------------------------------------------------------------------

    /// Every `(j, |dr|)` a correct query at `p` must return, by brute force over all atoms
    /// and every image the cutoff can reach.
    fn brute_within_point(
        cell: &UnitCell,
        pos: &[Coord3],
        p: Coord3,
        cutoff: f64,
    ) -> Vec<(usize, f64)> {
        // The image range is the same rule the list uses, obtained through a throwaway list so
        // the brute force stays independent of the code under test where it matters (the
        // candidate walk) while still sharing the one physical rule (how far images reach).
        let shift = CellList::build(cell, pos, cutoff.max(1e-9))
            .unwrap()
            .image_shift_range(cutoff);
        let mut want: Vec<(usize, f64)> = Vec::new();
        for (j, &pj) in pos.iter().enumerate() {
            let raw = sub3(pj, p);
            let (base, n0) = cell.min_image_with_shift(raw).unwrap();
            for tx in (n0[0] - shift[0])..=(n0[0] + shift[0]) {
                for ty in (n0[1] - shift[1])..=(n0[1] + shift[1]) {
                    for tz in (n0[2] - shift[2])..=(n0[2] + shift[2]) {
                        let d = cell.shift_vector([tx - n0[0], ty - n0[1], tz - n0[2]]);
                        let dr = add3(base, d);
                        let r = norm(dr);
                        if r <= cutoff {
                            want.push((j, r));
                        }
                    }
                }
            }
        }
        want.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        want
    }

    fn sorted_hits(out: &[(usize, Coord3)]) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = out.iter().map(|&(j, dr)| (j, norm(dr))).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        v
    }

    #[test]
    fn a_point_query_agrees_with_brute_force_including_images() {
        // The load-bearing check. Points are deliberately spread over the whole box *and* over
        // the region outside it, because a query point off the box is exactly the case where a
        // stencil built from a clamped bin index silently drops candidates — and a dropped
        // candidate here becomes invented void in `void_b`, not a wrong number in a test.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cutoff = 5.1;
        let cl = CellList::build(&cell, &pos, cutoff).unwrap();

        // A deterministic spread of probes: interior, on faces, and outside the box on every
        // side, including corners. Any periodicity bug shows up as a count mismatch here.
        let probes: Vec<Coord3> = vec![
            [5.0, 5.0, 5.0],    // dead centre
            [0.0, 0.0, 0.0],    // corner, on three faces
            [10.0, 5.0, 5.0],   // far face
            [2.5, 7.5, 1.25],   // interior
            [-3.0, 5.0, 5.0],   // outside -x
            [13.0, 5.0, 5.0],   // outside +x
            [5.0, -4.0, 5.0],   // outside -y
            [5.0, 5.0, 12.5],   // outside +z
            [-2.5, -2.5, -2.5], // outside a corner
            [11.5, 11.5, 11.5], // outside the far corner
            [0.0, 5.0, 5.0],    // exactly on a face
            [5.0, 5.0, 10.0],   // exactly on the far face
        ];

        for &p in probes.iter() {
            let mut got = Vec::new();
            cl.neighbors_within_point(p, cutoff, &mut got).unwrap();
            assert_eq!(
                sorted_hits(&got),
                brute_within_point(&cell, &pos, p, cutoff),
                "point query at {p:?} disagrees with brute force"
            );
        }
    }

    #[test]
    fn a_point_query_reports_displacements_from_the_query_point() {
        // `dr` must be `p_j - p`, not `p_j - p_i` for some anchor atom. Probing exactly on top
        // of an atom is the sharpest version of this: the answer must be the zero vector, and a
        // displacement measured from the wrong origin would instead be a full bond length.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cutoff = 5.1;
        let cl = CellList::build(&cell, &pos, cutoff).unwrap();

        let mut got = Vec::new();
        cl.neighbors_within_point(pos[3], cutoff, &mut got).unwrap();
        let self_hits: Vec<(usize, Coord3)> =
            got.iter().copied().filter(|&(j, _)| j == 3).collect();
        assert!(!self_hits.is_empty(), "querying on an atom must find it");
        for &(_, dr) in self_hits.iter() {
            // The zero image of the atom itself sits exactly on the query point; further
            // images of it are legitimate periodic neighbours and are not zero.
            let r = norm(dr);
            assert!(
                r < 1e-12 || (r - 5.0).abs() < 1e-9,
                "atom 3 reported at distance {r} from its own position"
            );
        }
        // And every direction must point from the query point towards its atom, which is what
        // `void_b` assumes when it negates `dr` to form `x - x_j`.
        for &(j, dr) in got.iter() {
            let from_query = sub3(pos[j], pos[3]);
            let norm_dr = norm(dr);
            let norm_raw = norm(from_query);
            assert!(
                (norm_dr - norm_raw).abs() < 1e-9,
                "atom {j}: |dr| {norm_dr} against |p_j - p| {norm_raw}"
            );
        }
    }

    #[test]
    fn a_point_query_honours_its_own_smaller_cutoff() {
        // The cutoff is a parameter, not the build radius. Shrinking it must drop the far
        // candidates and keep the near ones — the behaviour `void_b` relies on when it builds
        // one field and queries it at several thresholds.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 5.1).unwrap();
        let p = [5.0, 5.0, 5.0];
        let mut wide = Vec::new();
        cl.neighbors_within_point(p, 5.1, &mut wide).unwrap();
        let mut narrow = Vec::new();
        cl.neighbors_within_point(p, 1.0, &mut narrow).unwrap();
        assert!(
            narrow.len() < wide.len(),
            "a smaller cutoff returned {} against {} at a wider one",
            narrow.len(),
            wide.len()
        );
        for &(_, dr) in narrow.iter() {
            assert!(norm(dr) <= 1.0 + 1e-12, "cutoff not respected");
        }
        assert_eq!(
            sorted_hits(&narrow),
            brute_within_point(&cell, &pos, p, 1.0),
            "narrow query disagrees with brute force"
        );
    }

    #[test]
    fn a_point_query_clears_the_caller_scratch_each_time() {
        // `out` is reused across calls in the estimator's hot loop. If a stale entry survived,
        // it would be read as a real neighbour at a position from the previous query — which in
        // `void_b` means evaluating phi against an atom from somewhere else entirely.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cl = CellList::build(&cell, &pos, 5.1).unwrap();
        let mut scratch = Vec::new();
        cl.neighbors_within_point([5.0, 5.0, 5.0], 5.1, &mut scratch)
            .unwrap();
        let first = scratch.len();
        assert!(first > 0, "the probe should find the surrounding shell");
        // A point in the empty gap between the lattice sites finds strictly fewer, so a
        // leftover from the previous query would be visible as a count that does not shrink.
        cl.neighbors_within_point([1.25, 1.25, 1.25], 1.2, &mut scratch)
            .unwrap();
        assert!(
            scratch.len() < first,
            "scratch kept {} entries from a previous query",
            scratch.len()
        );
        assert_eq!(
            sorted_hits(&scratch),
            brute_within_point(&cell, &pos, [1.25, 1.25, 1.25], 1.2)
        );
        // And a query with no hits at all must leave it empty rather than unchanged.
        cl.neighbors_within_point([5.0, 5.0, 2.5], 0.2, &mut scratch)
            .unwrap();
        assert!(
            scratch.is_empty(),
            "empty result left {} entries",
            scratch.len()
        );
    }

    #[test]
    fn a_point_query_agrees_with_the_atom_query_at_an_atom_position() {
        // The two entry points must not disagree about the same geometry. Probing at an atom's
        // own position reproduces `neighbors_of` exactly, which ties the arbitrary-point path
        // to the already-validated per-atom one instead of leaving it to a separate argument.
        let cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        let pos = small_fcc_positions();
        let cutoff = 5.1;
        let cl = CellList::build(&cell, &pos, cutoff).unwrap();
        for (i, &site) in pos.iter().enumerate() {
            let mut by_point = Vec::new();
            cl.neighbors_within_point(site, cutoff, &mut by_point)
                .unwrap();
            let mut by_atom = Vec::new();
            cl.neighbors_of(i, cutoff, &mut by_atom).unwrap();
            // One documented difference, not a discrepancy: the per-atom query excludes the
            // atom itself, so its neighbour list is the neighbours *of* i, while a point query
            // at i's position legitimately finds i at distance zero. The atom that owns the
            // question is not a neighbour of itself.
            assert_eq!(
                by_point.len(),
                by_atom.len() + 1,
                "atom {i}: point query found {} against {} neighbours plus self",
                by_point.len(),
                by_atom.len()
            );
            let point_set = sorted_hits(&by_point);
            let atom_set = sorted_hits(&by_atom);
            // Same multiset once the zero-distance self hit is removed from the point query.
            let without_self: Vec<(usize, f64)> = point_set
                .iter()
                .copied()
                .filter(|&(j, r)| !(j == i && r < 1e-12))
                .collect();
            assert_eq!(
                without_self, atom_set,
                "atom {i}: the two queries disagree beyond the self hit"
            );
        }
    }

    #[test]
    fn a_point_query_honours_a_non_periodic_axis() {
        // A surface is the case where periodicity must *not* be invented: a query outside a
        // non-periodic face has no image to reach back, so the far side of the box must not
        // appear. This is the same reason method A cannot see a crater open to a free surface.
        let mut cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        cell.pbc = [true, true, false];
        let pos = small_fcc_positions();
        let cutoff = 5.1;
        let cl = CellList::build(&cell, &pos, cutoff).unwrap();
        // The block occupies z = 0 and z = 5 with nothing above. A query at z = 9 is above the
        // whole block, so in a non-periodic cell it can only reach the block downwards — every
        // returned displacement must have a negative z, i.e. point from the query towards an atom
        // that is physically below it. This is the meaningful statement about a surface: not
        // "nothing below" (the atoms are below, that is where the material is) but "no image
        // above", which would be periodicity leaking across a free surface.
        let mut above = Vec::new();
        cl.neighbors_within_point([5.0, 5.0, 9.0], cutoff, &mut above)
            .unwrap();
        assert!(
            !above.is_empty(),
            "the block below should still be reachable"
        );
        for &(_, dr) in above.iter() {
            assert!(
                dr[2] < 0.0,
                "a non-periodic query above the block reached an atom above it: {dr:?}"
            );
            assert!(
                dr[2] >= -10.0,
                "displacement {dr:?} is larger than the box, so an image was invented"
            );
        }
        // And the count is strictly smaller than the fully periodic case on the same query,
        // because the periodic version also reaches the block *through* the wrap from above.
        // That difference is the free surface: the same numbers as method A's inability to see
        // a crater open to a vacuum (§17.1).
        let mut periodic_cell = UnitCell::orthorhombic(10.0, 10.0, 10.0);
        periodic_cell.pbc = [true, true, true];
        let cl_periodic = CellList::build(&periodic_cell, &pos, cutoff).unwrap();
        let mut wrapped = Vec::new();
        cl_periodic
            .neighbors_within_point([5.0, 5.0, 9.0], cutoff, &mut wrapped)
            .unwrap();
        assert!(
            above.len() < wrapped.len(),
            "non-periodic {} not fewer than periodic {} on the same query",
            above.len(),
            wrapped.len()
        );
    }
}
