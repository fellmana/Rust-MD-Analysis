//! A coarse occupancy grid over the atoms, for analyses that need the volume
//! the material actually fills rather than the box (design §18.2): a thin film
//! in a box with vacuum, a free cluster, a slab with open boundaries.
//!
//! The grid's cells are about [`TARGET_PER_CELL`] atoms' worth of the material
//! on a side, measured on the material itself (the cell edge is refined from
//! the occupied volume, so vacuum does not inflate it). A cell is *interior*
//! when it and its six face neighbours hold atoms; the interior number density
//! is the atoms in interior cells over their volume, so the partly filled cells
//! of a free surface do not dilute it. Each cell also knows its Chebyshev
//! distance, in cells, to the nearest empty cell or open box face, which bounds
//! an atom's distance to a free surface from below.

use crate::cell::{Coord3, UnitCell};
use crate::error::Result;

/// Atoms per cell the edge is chosen for: enough that a cell inside the
/// material is never empty by chance, few enough to resolve a film a few nm
/// thick.
const TARGET_PER_CELL: f64 = 8.0;
/// Cells per atom the grid may have at most. The target is one cell per
/// [`TARGET_PER_CELL`] atoms in the material, so this only binds on vacuum and
/// on degenerate (flat) atom sets.
const MAX_CELLS_PER_ATOM: usize = 64;
/// The cell cap for frames of very few atoms.
const MIN_CELL_CAP: usize = 4096;
/// Gaussian width of [`Occupancy::smoothed_density`], Å, at most.
const DENSITY_WIDTH: f64 = 4.0;
/// Centres [`Occupancy::smoothed_density`] averages over, at most.
const DENSITY_CENTRES: usize = 4000;

/// The occupancy grid of one frame.
pub struct Occupancy {
    /// Cells per axis.
    n: [usize; 3],
    /// Fractional origin and extent of the grid per axis: the box on periodic
    /// axes, the atoms' range on open ones.
    lo: [f64; 3],
    span: [f64; 3],
    pbc: [bool; 3],
    /// Atoms per cell.
    count: Vec<u32>,
    /// Chebyshev distance in cells to the nearest empty cell or open face.
    depth: Vec<u32>,
    /// One cell's volume and its perpendicular thickness per axis, Å.
    cell_volume: f64,
    cell_height: [f64; 3],
}

impl Occupancy {
    /// Build the grid for `positions` in `cell`.
    pub fn build(cell: &UnitCell, positions: &[Coord3]) -> Result<Self> {
        let inv_t = cell.inverse_transposed()?;
        let frac: Vec<Coord3> = positions
            .iter()
            .map(|&x| Self::frac(cell, &inv_t, x))
            .collect();
        let mut lo = [0.0; 3];
        let mut span = [1.0; 3];
        for a in 0..3 {
            if !cell.pbc[a] && !frac.is_empty() {
                let (mn, mx) = frac
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), l| {
                        (mn.min(l[a]), mx.max(l[a]))
                    });
                lo[a] = mn;
                span[a] = (mx - mn).max(1e-9);
            }
        }
        let heights = cell.heights();
        let full = cell.volume() * span[0] * span[1] * span[2];
        let atoms = positions.len().max(1) as f64;
        // Refine the edge on the occupied volume, starting from the grid's.
        let mut edge = (TARGET_PER_CELL * full / atoms).cbrt();
        let mut grid = Self::with_edge(&frac, lo, span, cell, heights, edge);
        for _ in 0..3 {
            let occupied = grid.count.iter().filter(|&&c| c > 0).count() as f64;
            let next = (TARGET_PER_CELL * occupied * grid.cell_volume / atoms).cbrt();
            if (next - edge).abs() < 0.02 * edge {
                break;
            }
            edge = next;
            grid = Self::with_edge(&frac, lo, span, cell, heights, edge);
        }
        grid.depth = grid.distance_to_empty();
        Ok(grid)
    }

    fn with_edge(
        frac: &[Coord3],
        lo: [f64; 3],
        span: [f64; 3],
        cell: &UnitCell,
        heights: Coord3,
        edge: f64,
    ) -> Self {
        let mut n: [usize; 3] =
            std::array::from_fn(|a| ((heights[a] * span[a] / edge).floor() as usize).max(1));
        // At most MAX_CELLS_PER_ATOM cells per atom (and a floor for tiny
        // frames), coarsening the longest axis: the edge comes from the
        // occupied volume, which is near zero when the atoms are flat on an
        // open axis (a monolayer, a row, two atoms), and the cells per axis would
        // run to millions: uncapped, two atoms on a line 10 Å apart take 11 s and
        // 3.8 GB in `rmda rdf`.
        let cap = (MAX_CELLS_PER_ATOM * frac.len()).max(MIN_CELL_CAP) as f64;
        while (n[0] as f64) * (n[1] as f64) * (n[2] as f64) > cap {
            let a = (0..3).max_by_key(|&a| n[a]).unwrap();
            if n[a] == 1 {
                break;
            }
            n[a] = (n[a] / 2).max(1);
        }
        let mut count = vec![0u32; n[0] * n[1] * n[2]];
        for l in frac {
            let c: [usize; 3] = std::array::from_fn(|a| {
                let f = ((l[a] - lo[a]) / span[a] * n[a] as f64).floor();
                (f.max(0.0) as usize).min(n[a] - 1)
            });
            count[(c[0] * n[1] + c[1]) * n[2] + c[2]] += 1;
        }
        let cells = (n[0] * n[1] * n[2]) as f64;
        Occupancy {
            n,
            lo,
            span,
            pbc: cell.pbc,
            count,
            depth: Vec::new(),
            cell_volume: cell.volume() * span[0] * span[1] * span[2] / cells,
            cell_height: std::array::from_fn(|a| heights[a] * span[a] / n[a] as f64),
        }
    }

    fn index(&self, c: [usize; 3]) -> usize {
        (c[0] * self.n[1] + c[1]) * self.n[2] + c[2]
    }

    /// The cell at `c + d`, wrapping on periodic axes; `None` past an open face.
    fn step(&self, c: [usize; 3], d: [i64; 3]) -> Option<[usize; 3]> {
        let mut r = [0usize; 3];
        for a in 0..3 {
            let v = c[a] as i64 + d[a];
            let na = self.n[a] as i64;
            r[a] = if (0..na).contains(&v) {
                v as usize
            } else if self.pbc[a] {
                v.rem_euclid(na) as usize
            } else {
                return None;
            };
        }
        Some(r)
    }

    /// Multi-source breadth-first Chebyshev distance from the empty cells and
    /// the open faces (a cell on an open face is at distance 1 from outside).
    fn distance_to_empty(&self) -> Vec<u32> {
        let total = self.count.len();
        let mut depth = vec![u32::MAX; total];
        let mut queue = std::collections::VecDeque::new();
        for x in 0..self.n[0] {
            for y in 0..self.n[1] {
                for z in 0..self.n[2] {
                    let c = [x, y, z];
                    let i = self.index(c);
                    let on_open_face =
                        (0..3).any(|a| !self.pbc[a] && (c[a] == 0 || c[a] + 1 == self.n[a]));
                    if self.count[i] == 0 {
                        depth[i] = 0;
                        queue.push_back(c);
                    } else if on_open_face {
                        depth[i] = 1;
                        queue.push_back(c);
                    }
                }
            }
        }
        while let Some(c) = queue.pop_front() {
            let d = depth[self.index(c)];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(t) = self.step(c, [dx, dy, dz]) {
                            let j = self.index(t);
                            if depth[j] > d + 1 {
                                depth[j] = d + 1;
                                queue.push_back(t);
                            }
                        }
                    }
                }
            }
        }
        // A grid with no empty cell and no open face (bulk): unbounded.
        depth
    }

    /// The cell holding fractional position `l` (already wrapped on periodic
    /// axes).
    fn cell_of_frac(&self, l: Coord3) -> [usize; 3] {
        std::array::from_fn(|a| {
            let f = ((l[a] - self.lo[a]) / self.span[a] * self.n[a] as f64).floor();
            (f.max(0.0) as usize).min(self.n[a] - 1)
        })
    }

    /// Interior cells: occupied, with all six face neighbours occupied.
    fn interior(&self, c: [usize; 3]) -> bool {
        if self.count[self.index(c)] == 0 {
            return false;
        }
        for a in 0..3 {
            for s in [-1i64, 1] {
                let mut d = [0i64; 3];
                d[a] = s;
                match self.step(c, d) {
                    Some(t) if self.count[self.index(t)] > 0 => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Number density of the material: atoms in interior cells over their
    /// volume, or over every occupied cell when none is interior (a structure
    /// too thin for one).
    pub fn interior_density(&self) -> f64 {
        let (mut atoms, mut cells) = (0u64, 0u64);
        let (mut all_atoms, mut all_cells) = (0u64, 0u64);
        for x in 0..self.n[0] {
            for y in 0..self.n[1] {
                for z in 0..self.n[2] {
                    let c = [x, y, z];
                    let k = self.count[self.index(c)] as u64;
                    if k > 0 {
                        all_atoms += k;
                        all_cells += 1;
                        if self.interior(c) {
                            atoms += k;
                            cells += 1;
                        }
                    }
                }
            }
        }
        if cells > 0 {
            atoms as f64 / (cells as f64 * self.cell_volume)
        } else {
            all_atoms as f64 / (all_cells.max(1) as f64 * self.cell_volume)
        }
    }

    /// Number density of the material from a Gaussian-weighted neighbour
    /// count, `ρ = Σ_j exp(−r²/2s²) / (2πs²)^{3/2}` (self included) averaged
    /// over up to [`DENSITY_CENTRES`] atoms at least `4s` from any free
    /// surface. Exact for a uniform medium; for a crystal the shell structure
    /// enters as `exp(−(G·s)²/2)` of its shortest reciprocal vector G (2.8
    /// Å⁻¹ in W, 3.0 in Cu): about 1e-7 at s = 2 Å, so unlike
    /// [`Occupancy::interior_density`] it does not alias with the lattice
    /// planes. The kernel is cut at `4s` (weight e⁻⁸ there) and normalised by
    /// its own integral to that radius, not the full Gaussian's: a cut at 3s
    /// leaves out 2.9 % of the weight. `s` is [`DENSITY_WIDTH`] or a quarter of
    /// the deepest atom's depth,
    /// whichever is smaller; `None` when no atom is deep enough for
    /// `s ≥ min_width` (a structure too thin) — use `interior_density`.
    pub fn smoothed_density(
        &self,
        cell: &UnitCell,
        positions: &[Coord3],
        min_width: f64,
    ) -> Result<Option<f64>> {
        let inv_t = cell.inverse_transposed()?;
        let depth: Vec<f64> = positions
            .iter()
            .map(|&x| self.depth_of(Self::frac(cell, &inv_t, x)))
            .collect();
        let deepest = depth.iter().copied().fold(0.0f64, f64::max);
        let s = DENSITY_WIDTH.min(deepest / 4.0);
        if s.is_nan() || s < min_width {
            return Ok(None);
        }
        let reach = 4.0 * s;
        let deep: Vec<usize> = (0..positions.len())
            .filter(|&i| depth[i] >= reach)
            .collect();
        // Centres chosen and summed by position, so the estimate does not
        // depend on the order of atoms in the file.
        let centres =
            crate::analysis::defects::place_sample(cell, &deep, |i| positions[i], DENSITY_CENTRES);
        let list = crate::neighbor::cell_list::CellList::build(cell, positions, reach)?;
        let mut scratch = Vec::new();
        let mut total = 0.0;
        for &i in &centres {
            list.neighbors_of(i, reach, &mut scratch)?;
            let mut w = 1.0;
            for &(_, d) in scratch.iter() {
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                w += (-r2 / (2.0 * s * s)).exp();
            }
            total += w;
        }
        // ∫₀^reach 4πr² exp(−r²/2s²) dr, by Simpson's rule.
        let steps = 2000;
        let h = reach / steps as f64;
        let f = |r: f64| 4.0 * std::f64::consts::PI * r * r * (-r * r / (2.0 * s * s)).exp();
        let norm = (0..=steps)
            .map(|k| {
                let c = if k == 0 || k == steps {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                c * f(k as f64 * h)
            })
            .sum::<f64>()
            * h
            / 3.0;
        Ok(Some(total / (centres.len() as f64 * norm)))
    }

    /// Whether some layer of cells across a periodic axis is empty: a vacuum
    /// gap, so the box volume is not the material's.
    pub fn periodic_gap(&self) -> [bool; 3] {
        std::array::from_fn(|a| {
            self.pbc[a]
                && (0..self.n[a]).any(|k| {
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    (0..self.n[b]).all(|u| {
                        (0..self.n[c]).all(|v| {
                            let mut idx = [0usize; 3];
                            idx[a] = k;
                            idx[b] = u;
                            idx[c] = v;
                            self.count[self.index(idx)] == 0
                        })
                    })
                })
        })
    }

    /// A lower bound on the distance from the atom at fractional `l` to a free
    /// surface (an empty cell or an open face), Å; infinite in bulk.
    pub fn depth_of(&self, l: Coord3) -> f64 {
        let d = self.depth[self.index(self.cell_of_frac(l))];
        if d == u32::MAX {
            return f64::INFINITY;
        }
        // The atom may sit at the near edge of its cell: one cell less.
        let h = self.cell_height[0]
            .min(self.cell_height[1])
            .min(self.cell_height[2]);
        (d.saturating_sub(1)) as f64 * h
    }

    /// The fractional coordinates `build` uses, for [`Occupancy::depth_of`].
    pub fn frac(cell: &UnitCell, inv_t: &crate::cell::Mat3, x: Coord3) -> Coord3 {
        let mut l = cell.x2lamda_pre(x, inv_t);
        for (v, &p) in l.iter_mut().zip(&cell.pbc) {
            if p {
                *v -= v.floor();
            }
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Atoms flat on open axes (a row, a monolayer) have an occupied volume
    /// near zero, which uncapped gives a cell edge of ~1e-5 Å and millions of
    /// cells: two atoms 10 Å apart take 3.8 GB. The grid stays within its cap.
    #[test]
    fn flat_atom_sets_in_open_boxes_get_a_bounded_grid() {
        let open = UnitCell::orthorhombic(20.0, 20.0, 20.0).with_pbc([false; 3]);
        let row = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let g = Occupancy::build(&open, &row).unwrap();
        assert!(g.count.len() <= MIN_CELL_CAP, "{} cells", g.count.len());
        assert_eq!(g.count.iter().map(|&c| c as usize).sum::<usize>(), 2);

        let mut layer = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                layer.push([i as f64 * 2.7, j as f64 * 2.7, 5.0]);
            }
        }
        let g = Occupancy::build(&open.clone(), &layer).unwrap();
        let cap = (MAX_CELLS_PER_ATOM * layer.len()).max(MIN_CELL_CAP);
        assert!(
            g.count.len() <= cap,
            "{} cells for {} atoms",
            g.count.len(),
            layer.len()
        );
        assert_eq!(
            g.count.iter().map(|&c| c as usize).sum::<usize>(),
            layer.len()
        );
    }
}
