//! Minkowski lattice reduction — a direct port of ASE's
//! `ase/geometry/minkowski_reduction.py` (Nguyen & Stehlé, *Low-dimensional
//! Lattice Basis Reduction Revisited*, ACM Trans. Algorithms 5(4), 2009).
//!
//! This is the engine behind the general minimum-image path of
//! [`UnitCell::min_image`](super::UnitCell::min_image) (design §6.3): a
//! displacement cannot be reduced correctly by rounding its fractional
//! coordinates unless the basis is sufficiently reduced, so severely skewed
//! boxes first get a Minkowski-reduced basis and then the nearest lattice
//! vector is found by testing the Voronoi-relevant images `{-1,0,1}^d`
//! (ASE `general_find_mic`).
//!
//! The port is deliberately literal: `find_mic` results are meant to match
//! ASE's, so tie-breaks, iteration order and tolerances follow ASE
//! rather than being re-derived.

use crate::cell::Coord3;

/// Iteration cap, matching ASE's `MAX_IT`.
const MAX_IT: usize = 100_000;
/// Comparison tolerance, matching ASE's `TOL`.
const TOL: f64 = 1e-12;

/// Unimodular integer transform: `rcell[i] = Σ_j op[i][j] * h[j]`.
pub(super) type Op = [[i64; 3]; 3];

/// Rows are the lattice vectors.
type Basis = [[f64; 3]; 3];

#[inline]
fn row_norm(row: [f64; 3]) -> f64 {
    (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt()
}

#[inline]
fn dot3v(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn crossv(a: Coord3, b: Coord3) -> Coord3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `rcell = op @ h` — every reduced row is an integer combination of `h`'s.
fn apply_op(op: &Op, h: &Basis) -> Basis {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for (k, &o) in op[i].iter().enumerate() {
                s += o as f64 * h[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// `int_row @ basis`.
fn int_row_times(row: [i64; 3], h: &Basis) -> Coord3 {
    [
        row[0] as f64 * h[0][0] + row[1] as f64 * h[1][0] + row[2] as f64 * h[2][0],
        row[0] as f64 * h[0][1] + row[1] as f64 * h[1][1] + row[2] as f64 * h[2][1],
        row[0] as f64 * h[0][2] + row[1] as f64 * h[1][2] + row[2] as f64 * h[2][2],
    ]
}

/// Python's `round(float)` — banker's rounding (ties to even), as used by
/// `int(round(...))` in ASE's `reduction_gauss` and `closest_vector`.
#[inline]
fn round_half_even(x: f64) -> i64 {
    let fl = x.floor();
    let d = x - fl;
    // Python's round semantics: down on d < 0.5, up on d > 0.5, tie to even.
    let r = if d < 0.5 || (d == 0.5 && (fl as i64) % 2 == 0) {
        fl
    } else {
        fl + 1.0
    };
    r as i64
}

/// Guard against cycling between equivalent basis descriptions, exactly as
/// ASE's `CycleChecker`: membership over a fixed window of previously seen
/// sites (initialised to zero rows, like ASE's zero-filled array).
struct CycleChecker {
    seen: Vec<Vec<i64>>,
}

impl CycleChecker {
    fn new(d: usize) -> Self {
        // max_cycle_length = prod(n - i for i in 0..d) * np.prod(d), where the
        // second factor is numpy's product of a *scalar* d, i.e. d itself.
        let n = if d == 2 { 6 } else { 12 };
        let mut cap = 1usize;
        for i in 0..d {
            cap *= n - i;
        }
        cap *= d;
        CycleChecker {
            seen: vec![vec![0; 3 * d]; cap],
        }
    }

    /// Insert `site` at the top and report whether it was already present.
    fn add_site(&mut self, site: &[i64]) -> bool {
        debug_assert_eq!(site.len(), self.seen[0].len());
        let found = self.seen.iter().any(|s| s == site);
        self.seen.rotate_right(1);
        self.seen[0].copy_from_slice(site);
        found
    }
}

/// The nine `{-1,0,1}^2` combinations in `itertools.product` order.
fn product_2d() -> [[i64; 2]; 9] {
    let mut out = [[0i64; 2]; 9];
    let mut k = 0;
    for a in [-1i64, 0, 1] {
        for b in [-1i64, 0, 1] {
            out[k] = [a, b];
            k += 1;
        }
    }
    out
}

/// ASE's `relevant_vectors_2D`: the nine combinations sorted by length,
/// keeping the seven shortest (stable, so ties keep product order).
fn relevant_vectors_2d(u: [f64; 2], v: [f64; 2]) -> (Vec<[f64; 2]>, Vec<[i64; 2]>) {
    let cs = product_2d();
    let mut pairs: Vec<([i64; 2], [f64; 2])> = cs
        .iter()
        .map(|&c| {
            (
                c,
                [
                    c[0] as f64 * u[0] + c[1] as f64 * v[0],
                    c[0] as f64 * u[1] + c[1] as f64 * v[1],
                ],
            )
        })
        .collect();
    pairs.sort_by(|a, b| {
        let na = a.1[0] * a.1[0] + a.1[1] * a.1[1];
        let nb = b.1[0] * b.1[0] + b.1[1] * b.1[1];
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(7);
    (
        pairs.iter().map(|p| p.1).collect(),
        pairs.iter().map(|p| p.0).collect(),
    )
}

/// ASE's `closest_vector`: integers `a` minimising `|t0 + a0·u + a1·v|` in the
/// projection plane, found by a pruned walk over the Voronoi-relevant vectors.
fn closest_vector(t0: [f64; 2], u: [f64; 2], v: [f64; 2]) -> Result<[i64; 2], String> {
    let (rs, cs) = relevant_vectors_2d(u, v);
    let mut t = t0;
    let mut a = [0i64, 0i64];
    let mut dprev = f64::INFINITY;
    for _ in 0..MAX_IT {
        let mut best = 0usize;
        let mut bestd = f64::INFINITY;
        for (i, r) in rs.iter().enumerate() {
            let dx = r[0] + t[0];
            let dy = r[1] + t[1];
            let dn = (dx * dx + dy * dy).sqrt();
            if dn < bestd {
                bestd = dn;
                best = i;
            }
        }
        // index 0 is the zero vector, so it means "no improving move".
        if best == 0 || bestd >= dprev {
            return Ok(a);
        }
        dprev = bestd;
        let r = rs[best];
        let c = cs[best];
        let rr = r[0] * r[0] + r[1] * r[1];
        let kopt = round_half_even(-(t[0] * r[0] + t[1] * r[1]) / rr);
        a[0] += kopt * c[0];
        a[1] += kopt * c[1];
        t = [
            t0[0] + a[0] as f64 * u[0] + a[1] as f64 * v[0],
            t0[1] + a[0] as f64 * u[1] + a[1] as f64 * v[1],
        ];
    }
    Err("closest vector not found within MAX_IT".into())
}

/// ASE's `reduction_gauss`: Gauss-reduce the two given basis rows (integer
/// combinations of `h`'s rows), returning the pair in ASE's return order.
fn reduction_gauss(
    h: &Basis,
    hu0: [i64; 3],
    hv0: [i64; 3],
) -> Result<([i64; 3], [i64; 3]), String> {
    let mut hu = hu0;
    let mut hv = hv0;
    let mut u = int_row_times(hu, h);
    let mut v = int_row_times(hv, h);
    let mut cyc = CycleChecker::new(2);
    for _ in 0..MAX_IT {
        let uu = dot3v(u, u);
        if uu <= 0.0 {
            return Err("degenerate basis in Gauss reduction".into());
        }
        let x = round_half_even(dot3v(u, v) / uu);
        let next_hu = [hv[0] - x * hu[0], hv[1] - x * hu[1], hv[2] - x * hu[2]];
        let next_hv = hu;
        hu = next_hu;
        hv = next_hv;
        u = int_row_times(hu, h);
        v = int_row_times(hv, h);
        if dot3v(u, u) >= dot3v(v, v) || cyc.add_site(&[hu[0], hu[1], hu[2], hv[0], hv[1], hv[2]]) {
            return Ok((hv, hu));
        }
    }
    Err("Gaussian basis not found within MAX_IT".into())
}

/// ASE's `reduction_full`: the 3-D Nguyen–Stehlé reduction. Returns
/// `(R, H)` where `R = H @ h` is the reduced basis and `H` its integer
/// transform.
fn reduction_full(h: &Basis) -> Result<(Basis, Op), String> {
    let mut op: Op = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let mut norms = [row_norm(h[0]), row_norm(h[1]), row_norm(h[2])];
    let mut cyc = CycleChecker::new(3);
    for _ in 0..MAX_IT {
        // Stable argsort of the transform rows by current norms.
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| {
            norms[a]
                .partial_cmp(&norms[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        op = [op[order[0]], op[order[1]], op[order[2]]];

        let hw = op[2];
        let (hu, hv) = reduction_gauss(h, op[0], op[1])?;
        op = [hu, hv, hw];
        let r = apply_op(&op, h);

        // Gram–Schmidt the first two reduced vectors; the third is projected
        // onto that plane and corrected by `closest_vector`.
        let u = r[0];
        let v = r[1];
        let un = row_norm(u);
        if un <= 0.0 {
            return Err("degenerate basis in reduction_full".into());
        }
        let x = [u[0] / un, u[1] / un, u[2] / un];
        let vd = dot3v(v, x);
        let y = [v[0] - x[0] * vd, v[1] - x[1] * vd, v[2] - x[2] * vd];
        let yn = row_norm(y);
        if yn <= 0.0 {
            return Err("degenerate basis in reduction_full".into());
        }
        let y = [y[0] / yn, y[1] / yn, y[2] / yn];
        let pu = [dot3v(r[0], x), dot3v(r[0], y)];
        let pv = [dot3v(r[1], x), dot3v(r[1], y)];
        let pw = [dot3v(r[2], x), dot3v(r[2], y)];
        let nb = closest_vector(pw, pu, pv)?;

        op[2] = [
            nb[0] * op[0][0] + nb[1] * op[1][0] + op[2][0],
            nb[0] * op[0][1] + nb[1] * op[1][1] + op[2][1],
            nb[0] * op[0][2] + nb[1] * op[1][2] + op[2][2],
        ];
        let r = apply_op(&op, h);
        norms = [row_norm(r[0]), row_norm(r[1]), row_norm(r[2])];
        if norms[2] >= norms[1]
            || cyc.add_site(&[
                op[0][0], op[0][1], op[0][2], op[1][0], op[1][1], op[1][2], op[2][0], op[2][1],
                op[2][2],
            ])
        {
            return Ok((r, op));
        }
    }
    Err("reduced basis not found within MAX_IT".into())
}

/// ASE's `is_minkowski_reduced`: Minkowski's norm conditions on the basis.
pub(super) fn is_minkowski_reduced(h: &Basis, pbc: &[bool; 3]) -> bool {
    let dim = pbc.iter().filter(|&&p| p).count();
    if dim <= 1 {
        return true;
    }
    if dim == 2 {
        // Reorder to [shortest, longest, aperiodic] with the aperiodic row
        // zeroed first so it sorts to the front, like ASE.
        let aperiodic = pbc.iter().position(|&p| !p).unwrap_or(2);
        let mut c = *h;
        c[aperiodic] = [0.0; 3];
        let mut norms = [row_norm(c[0]), row_norm(c[1]), row_norm(c[2])];
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| {
            norms[a]
                .partial_cmp(&norms[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sel = [order[1], order[2], order[0]];
        let cc: Basis = [c[sel[0]], c[sel[1]], c[sel[2]]];
        norms = [row_norm(cc[0]), row_norm(cc[1]), row_norm(cc[2])];
        // A = [[0,1,0], [1,-1,0], [1,1,0]] over the first two rows.
        let lhs = [
            cc[1],
            [
                cc[0][0] - cc[1][0],
                cc[0][1] - cc[1][1],
                cc[0][2] - cc[1][2],
            ],
            [
                cc[0][0] + cc[1][0],
                cc[0][1] + cc[1][1],
                cc[0][2] + cc[1][2],
            ],
        ];
        let rhs = [norms[0], norms[1], norms[1]];
        (0..3).all(|k| row_norm(lhs[k]) >= rhs[k] - TOL)
    } else {
        cc_row_norms_ok(h)
    }
}

/// The 3-D Minkowski conditions (the `A` table in ASE's docstring).
fn cc_row_norms_ok(h: &Basis) -> bool {
    const A: [[i32; 3]; 12] = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [1, -1, -1],
    ];
    const RHS: [usize; 12] = [0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2];
    let norms = [row_norm(h[0]), row_norm(h[1]), row_norm(h[2])];
    for k in 0..12 {
        let mut v = [0.0f64; 3];
        for (d, vd) in v.iter_mut().enumerate() {
            for (j, &a) in A[k].iter().enumerate() {
                *vd += a as f64 * h[j][d];
            }
        }
        if row_norm(v) < norms[RHS[k]] - TOL {
            return false;
        }
    }
    true
}

fn det_sign(h: &Basis) -> i32 {
    let d = crate::cell::Mat3::from_rows(h[0], h[1], h[2]).det();
    if d >= 0.0 {
        1
    } else {
        -1
    }
}

/// ASE's `minkowski_reduce`: a Minkowski-reduced basis of the periodic
/// lattice vectors, plus the unimodular transform `rcell = op @ h`.
///
/// Only vectors with `pbc[i]` take part, so a slab keeps its free axis out of
/// the reduction (reducing the full 3-D lattice would smuggle a translation
/// along the non-periodic axis into the periodic shifts).
pub(super) fn reduce(h: &Basis, pbc: &[bool; 3]) -> Result<(Basis, Op), String> {
    let dim = pbc.iter().filter(|&&p| p).count();
    let mut op: Op = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    if is_minkowski_reduced(h, pbc) {
        return Ok((*h, op));
    }
    if dim == 2 {
        // Move the periodic vectors to the first two rows (and permute the
        // Cartesian axes with them, as ASE's `cell[perm][:, perm]` does).
        let mut perm = [0usize, 1, 2];
        perm.sort_by_key(|&i| pbc[i] as u8);
        perm.reverse();
        let mut pcell: Basis = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                pcell[i][j] = h[perm[i]][perm[j]];
            }
        }
        let mut norms = [row_norm(pcell[0]), row_norm(pcell[1]), row_norm(pcell[2])];
        norms[2] = f64::INFINITY;
        let mut indices = [0usize, 1, 2];
        indices.sort_by(|&a, &b| {
            norms[a]
                .partial_cmp(&norms[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        op = [op[indices[0]], op[indices[1]], op[indices[2]]];
        let (hu, hv) = reduction_gauss(&pcell, op[0], op[1])?;
        op[0] = hu;
        op[1] = hv;
        // Undo the row/column permutation: op = op[invperm][:, invperm].
        let mut inv = [0usize; 3];
        for i in 0..3 {
            inv[perm[i]] = i;
        }
        let mut op2: Op = [[0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                op2[i][j] = op[inv[i]][inv[j]];
            }
        }
        op = op2;
        // Preserve handedness: compare orientation of the original lattice
        // against the reduced one with the free axis replaced by a common
        // normal (ASE's dim==2 branch).
        let free = pbc.iter().position(|&p| !p).unwrap_or(2);
        let normal = crossv(h[(free + 1) % 3], h[(free + 2) % 3]);
        let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if nl > 0.0 {
            let normal = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
            let mut cell2 = *h;
            cell2[free] = normal;
            let mut rcell = apply_op(&op, h);
            rcell[free] = normal;
            if det_sign(&cell2) != det_sign(&rcell) {
                // `op[index - 1] *= -1` with Python's negative indexing.
                let row = (free + 2) % 3;
                for j in 0..3 {
                    op[row][j] = -op[row][j];
                }
            }
        }
    } else if dim == 3 {
        let (_r, op_full) = reduction_full(h)?;
        op = op_full;
        let rcell = apply_op(&op, h);
        if det_sign(h) != det_sign(&rcell) {
            for row in op.iter_mut() {
                for v in row.iter_mut() {
                    *v = -*v;
                }
            }
        }
    } else {
        // dim <= 1 is trivially reduced; `is_minkowski_reduced` returns true
        // above, so this arm is unreachable.
        return Ok((*h, op));
    }

    let rcell = apply_op(&op, h);
    let mut n1 = [row_norm(h[0]), row_norm(h[1]), row_norm(h[2])];
    let mut n2 = [row_norm(rcell[0]), row_norm(rcell[1]), row_norm(rcell[2])];
    n1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    n2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for k in 0..3 {
        if n2[k] > n1[k] + TOL {
            return Err(format!(
                "Minkowski reduction failed: sorted norm {} grew from {} to {}",
                k, n1[k], n2[k]
            ));
        }
    }
    Ok((rcell, op))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows for a strongly sheared, otherwise ordinary cell.
    fn skewed() -> Basis {
        [[10.0, 0.0, 0.0], [5.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    }

    #[test]
    fn cube_is_already_reduced() {
        let h = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        assert!(is_minkowski_reduced(&h, &[true; 3]));
        let (rcell, op) = reduce(&h, &[true; 3]).unwrap();
        assert_eq!(rcell, h, "an already-reduced cell is returned unchanged");
        assert_eq!(op, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    }

    #[test]
    fn skewed_cell_reduces_and_preserves_the_lattice() {
        let h = skewed();
        assert!(!is_minkowski_reduced(&h, &[true; 3]));
        let (rcell, op) = reduce(&h, &[true; 3]).unwrap();
        // op must be unimodular (integer matrix with |det| == 1), or the
        // "reduced" basis would generate a different lattice.
        let det = {
            let (a, b, c) = (op[0], op[1], op[2]);
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        };
        assert!(det == 1 || det == -1, "op must be unimodular, det = {det}");
        // and the reduced rows really are op @ h
        let expect = apply_op(&op, &h);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (rcell[i][j] - expect[i][j]).abs() < 1e-12,
                    "rcell[{i}][{j}]"
                );
            }
        }
        // sorted norms must not grow, and the result must be reduced
        let mut n0 = [row_norm(h[0]), row_norm(h[1]), row_norm(h[2])];
        let mut n1 = [row_norm(rcell[0]), row_norm(rcell[1]), row_norm(rcell[2])];
        n0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        n1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..3 {
            assert!(
                n1[k] <= n0[k] + 1e-9,
                "norm {k} grew: {} > {}",
                n1[k],
                n0[k]
            );
        }
        assert!(
            is_minkowski_reduced(&rcell, &[true; 3]),
            "output is reduced"
        );
    }

    #[test]
    fn slab_reduction_keeps_the_free_axis_out() {
        // pbc = [t, t, f]: the reduction may only combine rows a and b, so no
        // periodic shift may ever contain a c component.
        let h = [[10.0, 0.0, 1.0], [5.0, 10.0, 2.0], [0.0, 0.0, 50.0]];
        let pbc = [true, true, false];
        assert!(
            !is_minkowski_reduced(&h, &pbc),
            "this cell is deliberately skewed in-plane"
        );
        let (_rcell, op) = reduce(&h, &pbc).unwrap();
        for (i, &pi) in pbc.iter().enumerate() {
            if pi {
                for (j, &pj) in pbc.iter().enumerate() {
                    assert!(
                        op[i][j] == 0 || pj,
                        "periodic reduced row {i} must not combine the free vector (op[{i}][{j}] = {})",
                        op[i][j]
                    );
                }
            }
        }
        assert_ne!(op, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "reduction ran");
    }

    #[test]
    fn round_half_even_matches_python_round() {
        // Python's round(): ties to even.
        for (x, want) in [
            (0.5, 0i64),
            (1.5, 2),
            (2.5, 2),
            (3.5, 4),
            (-0.5, 0),
            (-1.5, -2),
            (-2.5, -2),
            (0.4, 0),
            (0.6, 1),
            (-0.6, -1),
            (2.0, 2),
        ] {
            assert_eq!(round_half_even(x), want, "round({x})");
        }
    }
}
