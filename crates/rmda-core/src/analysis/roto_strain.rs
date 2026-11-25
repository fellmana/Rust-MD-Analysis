//! The local roto-strain field `T(x)` (Mason et al., PRM 5, 095403 (2021)).
//!
//! Method B's rigid φ field ([`super::void_b`]) holds the face normals at their ideal
//! values while taking the cell *centres* from the observed positions. That measures how far
//! atoms sit from their ideal sites, so on a relaxed crystal `{φ > 1}` is a shell around every
//! displaced atom rather than a cavity. The field removes that by letting each cell deform with the
//! crystal: fit the local affine map `T(x)` that carries the ideal neighbour vectors onto the
//! observed ones, and read the face normals off the *deformed* cell.
//!
//! # The affine map, and why it is a 3×3 solve rather than a 12×12
//!
//! Pair each ideal neighbour vector `a_p` with the observed neighbour vector `b_p` and minimise
//!
//! ```text
//! f(T) = Σ_p w_p |b_p − T a_p|²
//! ```
//!
//! which is linear in `T`. Differentiating gives, per component,
//!
//! ```text
//! (T A)[m][n] = C[m][n],   A[j][n] = Σ_p w_p a_p[j] a_p[n],   C[m][n] = Σ_p w_p b_p[m] a_p[n]
//! ```
//!
//! so `T = C A⁻¹` — and the 9+3 = 12 parameters of the fit collapse, because the design matrix `A`
//! does not depend on the component of `b` being fitted. One 3×3 solve recovers all nine entries.
//! The trade is honest and worth naming: because `A` is shared across the whole mesh when the
//! template and the acceptance set are, `A` is the *same* matrix at every node for an ideal
//! crystal, so the field degenerates to a plain weighted average of per-atom gradients — which is
//! precisely what a least-squares fit should do, and why the formulation needs no matrix library.
//! Where the acceptance set differs between atoms the `A` differ and the solve is genuinely per
//! node, but it is still 3×3.
//!
//! # Pairing without a template match
//!
//! Mason's fit uses PTM's per-atom ideal neighbour directions. No API in this crate exposes those
//! ([`crate::analysis::ptm`] classifies and returns an orientation, not an oriented neighbour
//! set), so the ideal directions are taken instead from the primitive cell's own Delaunay
//! vectors — the same set [`super::void_b::face_normals`] builds the field from, inverted through
//! `v̂ = 2n̂/|n̂|²`. That is not a weaker substitute for this purpose: the W–S faces *are* the
//! bisectors of those vectors, so a map that preserves them preserves the cell.
//!
//! Pairing then needs no search over correspondences, only a sign choice. For each ideal vector
//! `t_k` take the observed neighbour with `o·t_k > 0` that maximises `o·t_k`. Requiring the
//! positive side is what makes the fit correct rather than merely well-conditioned: the Delaunay
//! set is symmetric under `t → −t`, so `T t_k = o_k` and `T(−t_k) = −o_k` are the same statement,
//! and matching both halves of a class to the same observed neighbour would ask `T` to be
//! antisymmetric about an identity it is supposed to be close to.
//!
//! # What this does and does not buy
//!
//! A *uniform* affine deformation — dilation, rotation, shear, or any mix — is exactly representable,
//! and the deformed cells still tile space, so φ stays `≤ 1` and a defect-free crystal reads as
//! defect-free. `rotated_lattice_has_no_void` and `dilated_lattice_has_no_void` pin both. That is a
//! correctness gain over the rigid field, which reports a rotated perfect lattice as largely void.
//!
//! A *non-uniform* deformation is not exactly representable, and the deformed cells stop tiling.
//! Thermal displacement therefore still leaves a residual shell, smaller than the rigid field's but
//! not zero — the mesh spacing sets the scale below which strain is averaged away. So this narrows
//! the rigid field's error rather than removing it, and method B's gate (design §17.2) is not
//! waived by it.

use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::lattice::PrimitiveCell;
use crate::neighbor::cell_list::CellList;
use crate::neighbor::nnearest::nearest_k_excluding_self;

/// Inverse of `n̂ = 2v̂/|v̂|²`: the ideal Delaunay neighbour vectors behind a normal set.
///
/// Exact inversion of [`super::void_b::face_normals`]'s output, so the template and the field's
/// normals are guaranteed to describe the same lattice rather than two independently derived
/// descriptions of it that happen to agree.
pub fn template_from_normals(normals: &[Coord3]) -> Result<Vec<Coord3>> {
    if normals.is_empty() {
        return Err(Error::invalid(
            "cannot build a template from an empty normal set",
        ));
    }
    let mut out = Vec::with_capacity(normals.len());
    for &n in normals.iter() {
        let n2 = dot(n, n);
        if !n2.is_finite() || n2 <= 0.0 {
            return Err(Error::invalid(format!(
                "face normal {n:?} has |n|² = {n2}, which has no inverse vector"
            )));
        }
        out.push(scale(n, 2.0 / n2));
    }
    Ok(out)
}

/// `(ideal, observed)` pairs for one atom, matched one-to-one by direction.
///
/// `observed` should be the atom's `k` nearest displacement vectors, `k` at least
/// `template.len()` — [`RotoStrainField::fit`] passes exactly that.
///
/// Each ideal vector needs *its own* observed neighbour, and enforcing that is the whole
/// difficulty. Matching each ideal vector independently to whichever observed vector best agrees
/// with it lets two ideal vectors claim the same neighbour, and then the pair set is not a
/// function of direction at all: the least-squares fit still returns a matrix, but one fitted to
/// an inconsistent correspondence. That is not a small error. On a rotated 4×4×4 FCC block it
/// produced strains of up to 63 % on 16 of 256 atoms, and a mesh node averaging those in reported
/// 35 % of a perfect crystal as void. `every_atom_of_a_rotated_fcc_block_recovers_the_rotation`
/// pins the behaviour.
///
/// So candidates are collected, sorted by projection descending, and claimed exclusively. Sorting
/// is what makes greedy safe here: for a correct neighbour the projection is the maximum possible
/// (`|t|`), and any competing neighbour sits at least a `30°` angle away for FCC, so all the true
/// pairs are claimed before a wrong one can win a slot. Ties break on index, so the result does
/// not depend on the sort's stability.
///
/// Pairs longer apart than `max_deviation` are dropped when that is `Some`, which is Mason's
/// rejection test. It cuts the influence of a neighbour that is nowhere near where the ideal
/// lattice says it should be — a substitution or a first-shell vacancy — at the cost of biasing
/// the fit *towards* the ideal lattice, which is a real trade and not only a safeguard.
/// `rejection_bites_at_a_defect_and_biases_toward_the_ideal_lattice` measures it.
pub fn match_template(
    template: &[Coord3],
    observed: &[Coord3],
    max_deviation: Option<f64>,
) -> Vec<(Coord3, Coord3)> {
    let mut cand: Vec<(f64, usize, usize)> = Vec::with_capacity(template.len() * observed.len());
    for (ti, &t) in template.iter().enumerate() {
        let t2 = dot(t, t);
        if t2 <= 0.0 {
            continue;
        }
        for (oi, &o) in observed.iter().enumerate() {
            let proj = dot(o, t) / t2;
            // `proj > 0` is the side test: a partner must lie beyond the bisector plane, and only
            // the neighbour on that side can define the face.
            if proj > 0.0 && !(max_deviation.is_some_and(|lim| dist2(o, t) > lim * lim)) {
                cand.push((proj, ti, oi));
            }
        }
    }
    // Descending projection, ties by template then observed index so the assignment is
    // reproducible run to run.
    cand.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    let mut taken_t = vec![false; template.len()];
    let mut taken_o = vec![false; observed.len()];
    let mut pairs = Vec::with_capacity(template.len());
    for (_, ti, oi) in cand {
        if taken_t[ti] || taken_o[oi] {
            continue;
        }
        taken_t[ti] = true;
        taken_o[oi] = true;
        pairs.push((template[ti], observed[oi]));
    }
    // Back to template order, so a caller's pair list is reproducible position by position
    // regardless of how the greedy order fell out.
    pairs.sort_by(|a, b| {
        template
            .iter()
            .position(|v| *v == a.0)
            .unwrap_or(usize::MAX)
            .cmp(
                &template
                    .iter()
                    .position(|v| *v == b.0)
                    .unwrap_or(usize::MAX),
            )
    });
    pairs
}

/// Per-atom normal-equation moments, so the mesh pass never revisits a neighbour list.
///
/// `a` is the symmetric `Σ w a aᵀ` and `c` the `Σ w b aᵀ` of one atom's pairs; summing these over
/// a neighbourhood and solving once gives the same `T = C A⁻¹` as fitting the pairs directly,
/// because the sum of sums is the sum. Storing them (18 floats) instead of the pairs (24 floats
/// each, 12 pairs for FCC) is what keeps a 256k-atom fit inside a sane memory budget, and it is
/// the reason the field is built once and then reused.
#[derive(Debug, Clone, Copy)]
pub struct Moments {
    /// `A[j][n] = Σ_p w_p a_p[j] a_p[n]`, symmetric.
    pub a: Mat3,
    /// `C[m][n] = Σ_p w_p b_p[m] a_p[n]`.
    pub c: Mat3,
    /// Pairs that survived matching. Zero means this atom contributed nothing.
    pub pairs: u32,
}

impl Default for Moments {
    /// Zeroed, *not* derived.
    ///
    /// `#[derive(Default)]` here would call [`Mat3::default`], which is hand-written as the
    /// identity (`cell.rs`), because identity is the useful default for a lattice matrix. For an
    /// accumulator that is wrong in a way no assertion catches at the point of the mistake: every
    /// node's normal equations would start from `I`, so `T` would be solved against a design
    /// matrix inflated by one spurious unit and a strain of a few percent would read as a few
    /// percent of *identity* rather than of the crystal. Written out so the intent is visible.
    fn default() -> Self {
        Self {
            a: Mat3 { m: [[0.0; 3]; 3] },
            c: Mat3 { m: [[0.0; 3]; 3] },
            pairs: 0,
        }
    }
}

impl Moments {
    /// Accumulate one pair of weight `w`.
    pub fn add(&mut self, ideal: Coord3, observed: Coord3, w: f64) {
        for m in 0..3 {
            for n in 0..3 {
                self.a.m[m][n] += w * ideal[m] * ideal[n];
                self.c.m[m][n] += w * observed[m] * ideal[n];
            }
        }
        self.pairs += 1;
    }

    /// `T = C A⁻¹`, or `None` if `A` cannot be inverted at this precision.
    ///
    /// `None` rather than a silent identity: a node or atom with no usable pairs has no affine
    /// map, and reporting "undeformed" for it would make a region with no data read as a region
    /// with no strain.
    pub fn solve(&self) -> Option<Mat3> {
        if self.pairs == 0 {
            return None;
        }
        // `T = C A⁻¹`, so column j of `T` is `C` times column j of `A⁻¹` — obtained by solving
        // `A y_j = e_j`. Solving `A⁻¹` against the columns of `C` instead would give `A⁻¹C`,
        // which is `Tᵀ` and is wrong by construction; the two coincide only when `A` is a
        // multiple of the identity, which is exactly the case that hides the mistake.
        let mut t = Mat3::identity();
        for j in 0..3 {
            let mut e = [0.0; 3];
            e[j] = 1.0;
            let y = self.a.solve(e).ok()?;
            for m in 0..3 {
                t.m[m][j] = self.c.m[m][0] * y[0] + self.c.m[m][1] * y[1] + self.c.m[m][2] * y[2];
            }
        }
        Some(t)
    }

    /// `A` and `C` summed over a weighted neighbourhood, weights already folded in.
    pub fn accumulate(&mut self, other: &Moments, w: f64) {
        if w <= 0.0 {
            return;
        }
        for m in 0..3 {
            for n in 0..3 {
                self.a.m[m][n] += w * other.a.m[m][n];
                self.c.m[m][n] += w * other.c.m[m][n];
            }
        }
        self.pairs += other.pairs;
    }
}

/// How to build a [`RotoStrainField`].
#[derive(Debug, Clone, Copy)]
pub struct RotoStrainOptions {
    /// Mesh spacing in Ångström, at the finest pass.
    pub node_spacing: f64,
    /// Gaussian width in Ångström, at the finest pass. Mason's schedule ends near `a₀`.
    pub sigma: f64,
    /// Coarse-to-fine passes. Each halves `sigma` and halves the mesh spacing, so the first pass
    /// sees `sigma · 2^(passes − 1)` Ångström of the crystal and the last sees `sigma`.
    pub refine_passes: usize,
    /// Rejection threshold in Ångström, Mason's `a₀/4`. `None` disables rejection.
    pub max_deviation: Option<f64>,
    /// Fill a node no neighbourhood could solve from the nearest solved node. On by default,
    /// because a real void leaves some nodes with too few usable pairs and refusing to build would
    /// make the field unusable exactly where it is most needed. The count is exposed as
    /// [`RotoStrainField::unsolved`] rather than hidden.
    pub fill_unsolved: bool,
}

impl Default for RotoStrainOptions {
    fn default() -> Self {
        Self {
            node_spacing: 2.0,
            sigma: 4.0,
            refine_passes: 1,
            max_deviation: None,
            fill_unsolved: true,
        }
    }
}

/// A periodic trilinear mesh of local affine maps, one per node.
#[derive(Debug, Clone)]
pub struct RotoStrainField {
    /// `T` at each node, node `i + dims[0]*(j + dims[1]*k)`; identity where unsolved.
    nodes: Vec<Mat3>,
    dims: [usize; 3],
    /// Node step in *fractional* coordinates, i.e. `1 / dims` per axis.
    ///
    /// Fractional, not Cartesian, and that is load-bearing: a mesh laid out on Cartesian axes
    /// covers an orthorhombic box exactly and a rotated or triclinic box not at all — its nodes
    /// walk off the crystal, so the fit returns whatever neighbourhood happened to contain them.
    /// Every lattice test here is orthorhombic and would not have caught it;
    /// `rotated_perfect_lattice_is_not_void_once_strained` is the test that does.
    spacing: Coord3,
    /// The cell the mesh is built on, needed to map a query point to fractional coordinates.
    cell: UnitCell,
    solved: Vec<bool>,
    /// How many nodes were unsolved *before* filling. Kept separately because `fill` overwrites
    /// `solved`, so counting the flag afterwards would always report a clean field.
    n_unsolved: usize,
    n_atoms: usize,
    n_pairs: usize,
    refine_passes: usize,
}

impl RotoStrainField {
    /// Fit the field over `atoms` for the lattice described by `pc`.
    ///
    /// `pc` supplies the template and the rejection scale, so it must be the lattice the atoms
    /// were built to be compared against — the same assumption [`super::void_b::face_normals`]
    /// makes, and the reason this reports a strain rather than a mis-assignment.
    pub fn fit(
        cell: &UnitCell,
        atoms: &[Coord3],
        pc: &PrimitiveCell,
        opts: &RotoStrainOptions,
    ) -> Result<Self> {
        let normals = super::void_b::face_normals(pc)?;
        let template = template_from_normals(&normals)?;
        Self::fit_with_template(cell, atoms, &template, opts)
    }

    /// As [`RotoStrainField::fit`], with the template supplied rather than derived.
    ///
    /// Exposed because PTM, once it exposes oriented neighbour vectors, is the better source and
    /// this is the seam it would arrive through.
    pub fn fit_with_template(
        cell: &UnitCell,
        atoms: &[Coord3],
        template: &[Coord3],
        opts: &RotoStrainOptions,
    ) -> Result<Self> {
        if atoms.is_empty() {
            return Err(Error::invalid("roto-strain fit needs at least one atom"));
        }
        if template.is_empty() {
            return Err(Error::invalid("roto-strain fit needs a non-empty template"));
        }
        if !opts.sigma.is_finite() || opts.sigma <= 0.0 {
            return Err(Error::invalid(format!(
                "roto-strain sigma must be positive, got {}",
                opts.sigma
            )));
        }
        if !opts.node_spacing.is_finite() || opts.node_spacing <= 0.0 {
            return Err(Error::invalid(format!(
                "roto-strain node spacing must be positive, got {}",
                opts.node_spacing
            )));
        }

        // One neighbour query per atom, done once: the per-atom moments do not depend on where a
        // node is, only the Gaussian weights applied to them do. The list here is sized to the
        // template's longest vector rather than to any mesh pass's sigma, because that is all
        // `nearest_k` can reach and the mesh passes get their own lists below.
        let reach = template
            .iter()
            .map(|v| dot(*v, *v).sqrt())
            .fold(0.0f64, f64::max);
        let pair_cut = reach * 1.2;
        let list = CellList::build(cell, atoms, pair_cut)?;
        let mut moments: Vec<Moments> = Vec::with_capacity(atoms.len());
        let mut scratch: Vec<(u32, Coord3, f64)> = Vec::with_capacity(template.len() + 1);
        let mut n_pairs = 0usize;
        for i in 0..atoms.len() {
            nearest_k_excluding_self(&list, i, template.len(), &mut scratch)?;
            let observed: Vec<Coord3> = scratch.iter().map(|s| s.1).collect();
            let mut mm = Moments::default();
            for (t, o) in match_template(template, &observed, opts.max_deviation) {
                mm.add(t, o, 1.0);
            }
            n_pairs += mm.pairs as usize;
            moments.push(mm);
        }

        // Coarse to fine. Each pass halves both the width and the spacing, so the first pass
        // averages over the most crystal and the last is the sharpest field the spacing allows.
        let mut best: Option<RotoStrainField> = None;
        let passes = opts.refine_passes.max(1);
        for p in 0..passes {
            let scale = 2f64.powi(passes as i32 - 1 - p as i32);
            // The requested spacing is a Cartesian length, so the node count per axis has to be
            // derived through the cell's own perpendicular widths — see `pass`.
            let spacing = [
                opts.node_spacing * scale,
                opts.node_spacing * scale,
                opts.node_spacing * scale,
            ];
            let sigma = opts.sigma * scale;
            // A list per pass, sized to that pass's sigma. Sharing the coarsest pass's list would
            // work, but the stencil is sized at build time, so every finer query would scan the
            // coarse stencil — a `(2*cutoff/binsize + 1)³` cost per node that no amount of tuning
            // the query radius can recover.
            let cl = CellList::build(cell, atoms, sigma)?;
            best = Some(pass(cell, &cl, &moments, &spacing, sigma)?);
        }

        let mut field = best.expect("at least one pass");
        field.n_atoms = atoms.len();
        field.n_pairs = n_pairs;
        field.refine_passes = passes;
        if field.unsolved() > 0 && !opts.fill_unsolved {
            return Err(Error::invalid(format!(
                "{} of {} roto-strain mesh nodes could not be solved from their neighbourhood; \
                 widen sigma, coarsen node_spacing, or set fill_unsolved",
                field.unsolved(),
                field.len_nodes()
            )));
        }
        Ok(field)
    }

    /// `T` at the node with index `(i, j, k)`, or `None` if out of range.
    pub fn node(&self, i: usize, j: usize, k: usize) -> Option<Mat3> {
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return None;
        }
        Some(self.nodes[self.lin(i, j, k)])
    }

    /// `T(x)`, trilinearly interpolated and periodic in every axis of the cell.
    ///
    /// Trilinear rather than nearest-node so that the normals the φ field reads vary continuously
    /// with `x`: a nearest-node field would put a step in `φ` at every node plane, which is a
    /// surface that is not in the crystal.
    pub fn at(&self, x: Coord3) -> Mat3 {
        // Fractional coordinates first, so the mesh is the cell's own lattice rather than the
        // Cartesian axes. Taking the integer part of a fractional coordinate is also what makes
        // the wrap at 1.0 land on node 0 rather than on an extrapolated node.
        // A degenerate cell cannot be mapped into fractional coordinates; there is no mesh to
        // interpolate on, so identity is the only honest answer and `unsolved` reports the
        // condition that produced it.
        let lam = match self.cell.x2lamda(x) {
            Ok(l) => l,
            Err(_) => return Mat3::identity(),
        };
        let mut base = [0usize; 3];
        let mut frac = [0.0f64; 3];
        for d in 0..3 {
            let u = lam[d] * self.dims[d] as f64;
            let f = u.floor();
            frac[d] = u - f;
            base[d] = (f as i64).rem_euclid(self.dims[d] as i64) as usize;
        }
        // Zero, not `Mat3::default()`: `Default` for `Mat3` is the *identity* (cell.rs), so an
        // accumulator seeded from it returns `I + interpolation` and reports strain where there is
        // none. Every entry below is assigned, so the trap only shows up as a wrong number.
        let mut acc = Mat3 { m: [[0.0; 3]; 3] };
        let weights = [
            (1.0 - frac[0]) * (1.0 - frac[1]) * (1.0 - frac[2]),
            frac[0] * (1.0 - frac[1]) * (1.0 - frac[2]),
            (1.0 - frac[0]) * frac[1] * (1.0 - frac[2]),
            frac[0] * frac[1] * (1.0 - frac[2]),
            (1.0 - frac[0]) * (1.0 - frac[1]) * frac[2],
            frac[0] * (1.0 - frac[1]) * frac[2],
            (1.0 - frac[0]) * frac[1] * frac[2],
            frac[0] * frac[1] * frac[2],
        ];
        let mut c = 0;
        for dk in 0..2usize {
            let k = (base[2] + dk) % self.dims[2];
            for dj in 0..2usize {
                let j = (base[1] + dj) % self.dims[1];
                for di in 0..2usize {
                    let i = (base[0] + di) % self.dims[0];
                    let w = weights[c];
                    c += 1;
                    if w == 0.0 {
                        continue;
                    }
                    let n = self.nodes[self.lin(i, j, k)];
                    for r in 0..3 {
                        for col in 0..3 {
                            acc.m[r][col] += w * n.m[r][col];
                        }
                    }
                }
            }
        }
        acc
    }

    /// Mesh dimensions per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Node step per axis in fractional coordinates, i.e. `1 / dims`.
    pub fn spacing(&self) -> Coord3 {
        self.spacing
    }

    /// Total node count.
    pub fn len_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Nodes no neighbourhood could solve, before filling. Non-zero means part of the field is
    /// borrowed from elsewhere rather than measured; on a real void cluster that is expected, and
    /// the number says how much of it.
    pub fn unsolved(&self) -> usize {
        self.n_unsolved
    }

    /// Atoms the fit consumed.
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Neighbour pairs that survived matching across all atoms.
    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    /// Coarse-to-fine passes actually run.
    pub fn refine_passes(&self) -> usize {
        self.refine_passes
    }

    /// How much of the field deviates from the identity, as `max_ij |T_ij − δ_ij|`.
    ///
    /// The one number that says whether this fit found anything: a field that is identity
    /// everywhere has measured no deformation, which for a nominal crystal is a failure to detect
    /// rather than a clean result.
    pub fn max_deviation_from_identity(&self) -> f64 {
        let mut worst = 0.0f64;
        for n in self.nodes.iter() {
            for r in 0..3 {
                for c in 0..3 {
                    let target = if r == c { 1.0 } else { 0.0 };
                    worst = worst.max((n.m[r][c] - target).abs());
                }
            }
        }
        worst
    }

    fn lin(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }
}

fn pass(
    cell: &UnitCell,
    list: &CellList,
    moments: &[Moments],
    spacing: &Coord3,
    sigma: f64,
) -> Result<RotoStrainField> {
    let heights = cell.heights();
    let mut dims = [0usize; 3];
    for d in 0..3 {
        if heights[d] <= 0.0 {
            return Err(Error::invalid(format!(
                "cell height on axis {d} is {}, which cannot hold a mesh",
                heights[d]
            )));
        }
        // `heights` rather than `lengths`: the number of nodes that fit along an axis is set by the
        // *perpendicular* width of the cell, so a sheared box is not divided by its slanted edge
        // length and left under-resolved along its short axis.
        // Two nodes minimum: a single-node axis would make the interpolation a constant, which
        // is a field with no variation along that direction and no way to say so.
        let n = (heights[d] / spacing[d]).round().max(2.0) as usize;
        dims[d] = n;
    }
    let spacing = [
        1.0 / dims[0] as f64,
        1.0 / dims[1] as f64,
        1.0 / dims[2] as f64,
    ];
    let sigma2 = 2.0 * sigma * sigma;
    let n_nodes = dims[0] * dims[1] * dims[2];
    let mut nodes = vec![Mat3::identity(); n_nodes];
    let mut solved = vec![false; n_nodes];
    let mut hits: Vec<(usize, Coord3)> = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let x = cell.lamda2x([
                    i as f64 * spacing[0],
                    j as f64 * spacing[1],
                    k as f64 * spacing[2],
                ]);
                list.neighbors_within_point(x, sigma, &mut hits)?;
                let mut acc = Moments::default();
                for &(idx, dr) in hits.iter() {
                    let w = (-dist2(dr, [0.0, 0.0, 0.0]) / sigma2).exp();
                    if w <= 0.0 {
                        continue;
                    }
                    acc.accumulate(&moments[idx], w);
                }
                let slot = i + dims[0] * (j + dims[1] * k);
                if let Some(t) = acc.solve() {
                    nodes[slot] = t;
                    solved[slot] = true;
                }
            }
        }
    }

    let mut field = RotoStrainField {
        nodes,
        dims,
        spacing,
        cell: *cell,
        solved,
        n_unsolved: 0,
        n_atoms: 0,
        n_pairs: 0,
        refine_passes: 0,
    };
    field.n_unsolved = field.solved.iter().filter(|&&s| !s).count();
    if field.n_unsolved > 0 {
        fill(&mut field);
    }
    Ok(field)
}

/// Replace unsolved nodes with a copy of the nearest solved one.
fn fill(field: &mut RotoStrainField) {
    let snapshot = field.nodes.clone();
    let solved: Vec<bool> = field.solved.clone();
    // Breadth-first flood over the 6-connected node lattice: every unsolved node takes the value
    // of the nearest solved one, and an unsolved component with no solved neighbour anywhere keeps
    // the identity rather than inventing a strain for it.
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for (n, done) in solved.iter().enumerate() {
        if !*done {
            queue.push_back(n);
        }
    }
    let dims = field.dims;
    let idx_of = |n: usize| -> [usize; 3] {
        let k = n / (dims[0] * dims[1]);
        let rem = n % (dims[0] * dims[1]);
        [rem % dims[0], rem / dims[0], k]
    };
    while let Some(n) = queue.pop_front() {
        let [i, j, k] = idx_of(n);
        let mut neigh = Vec::with_capacity(6);
        for dk in -1i64..=1 {
            for dj in -1i64..=1 {
                for di in -1i64..=1 {
                    if di.abs() + dj.abs() + dk.abs() != 1 {
                        continue;
                    }
                    let ii = (i as i64 + di).rem_euclid(dims[0] as i64) as usize;
                    let jj = (j as i64 + dj).rem_euclid(dims[1] as i64) as usize;
                    let kk = (k as i64 + dk).rem_euclid(dims[2] as i64) as usize;
                    neigh.push(ii + dims[0] * (jj + dims[1] * kk));
                }
            }
        }
        for m in neigh {
            if !field.solved[m] {
                field.nodes[m] = snapshot[n];
                field.solved[m] = true;
                queue.push_back(m);
            }
        }
    }
}

fn dot(a: Coord3, b: Coord3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: Coord3, s: f64) -> Coord3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dist2(a: Coord3, b: Coord3) -> f64 {
    dot(sub(a, b), sub(a, b))
}

fn sub(a: Coord3, b: Coord3) -> Coord3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[cfg(test)]
mod fixtures {
    //! Blocks of ideal crystal, shared by the unit tests and the `φ`-integration tests.
    //!
    //! They live out here because both sets need them, and two copies of a fixture that has to be
    //! *exactly* commensurate is one more thing that can drift apart unnoticed.

    use super::{Coord3, Mat3, PrimitiveCell, UnitCell};
    use crate::lattice::fcc::Fcc;
    use crate::lattice::Lattice;

    /// Lattice constant for every fixture here, in Angstrom.
    pub const A: f64 = 4.0;

    /// `cells` conventional FCC cells in an orthorhombic box, edge to edge.
    pub fn fcc_block(cells: usize) -> (UnitCell, Vec<Coord3>, PrimitiveCell) {
        let pc = Fcc { a: A }.primitive();
        let cell = UnitCell::orthorhombic(A * cells as f64, A * cells as f64, A * cells as f64);
        (cell, fcc_sites(cells, Mat3::identity()), pc)
    }

    /// `cells` conventional FCC cells, filled edge to edge.
    ///
    /// FCC is the set `(a/2)·(i, j, k)` with `i + j + k` even, and that is how this is generated.
    /// The obvious alternatives both produce a fixture that is not a crystal:
    ///
    /// * `pc.h.mul_v([i + b, ...])` over `i in 0..cells` with `Fcc::primitive()`'s four-atom basis —
    ///   `Fcc::primitive()` is the *conventional* cell, `h = a·I`, so the axis index advances a whole
    ///   lattice constant while the repeat along an axis is `a / 2`. The last `a / 2` of every axis
    ///   comes up empty and periodicity drops a gap through the middle of the box instead of closing
    ///   it. Atoms bordering the gap fall back on second-shell neighbours and the fit absorbs the
    ///   error symmetrically, so a perfect crystal goes on measuring tens of percent void.
    /// * the same expression over `i in 0..2 * cells` — that fixes the gap and creates eight copies of
    ///   every site instead, because `i = 0, b = 0.5` and `i = 1, b = 0` are the same atom. Coincident
    ///   duplicates then sit at zero distance in the neighbour list and consume the `nearest_k` budget.
    ///
    /// Neither is caught by a mean or an extremum: duplicated and missing sites are symmetric, and the
    /// node values are Gaussian-weighted averages with somewhere to hide.
    /// `fcc_block_fills_the_box_it_declares` counts them,
    /// `every_atom_of_a_rotated_fcc_block_recovers_the_rotation` checks each atom separately, and
    /// `rotated_perfect_lattice_is_not_void_once_strained` asks the question that started all this.
    pub fn fcc_sites(cells: usize, t: Mat3) -> Vec<Coord3> {
        let cell = block_cell(cells, t);
        let n = 2 * cells;
        let half = A / 2.0;
        let mut sites = Vec::with_capacity(4 * cells.pow(3));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if (i + j + k) % 2 != 0 {
                        continue;
                    }
                    sites.push(cell.lamda2x(frac(
                        cells,
                        [half * i as f64, half * j as f64, half * k as f64],
                    )));
                }
            }
        }
        sites
    }

    /// Unrotated Cartesian coordinates of a `cells³` block → fractional coordinates of its box.
    ///
    /// The two differ by the box edge, and reading one as the other is silent: the lattice keeps
    /// its shape, it is just `1/cells` as dense and no longer commensurate with the box, so the
    /// block stops being a crystal of the declared cell. `lamda2x` wants the fractional form.
    pub fn frac(cells: usize, v: Coord3) -> Coord3 {
        let base = A * cells as f64;
        [v[0] / base, v[1] / base, v[2] / base]
    }

    /// The box of a `cells³` block after an affine map `t`, built the way `UnitCell` reads a matrix.
    ///
    /// `from_matrix` takes the **rows** as lattice vectors, so the `i`th edge of the mapped box is
    /// `Σ_d t[i][d]·(L·e_d)` — a scaled row of `t`, not a scaled column. `Mat3::mul_v` is the other
    /// convention (`M·v`, summing over columns), and the two differ by a transpose: for a rotation
    /// by `θ` one gives `+θ` and the other `−θ`.
    ///
    /// Mixing them is what broke this fixture, and it breaks quietly. The box came out rotated one
    /// way and the sites the other, so the block was never commensurate with the cell that claims
    /// to contain it: a perfect FCC lattice's 2.828 Å first shell measured 1.221 Å, three sites
    /// landed exactly on top of the origin, and the cell list — blamed for reporting five of twelve
    /// neighbours — was faithfully reporting on the positions it had been handed. Every quantity
    /// downstream was consistent and all of them were wrong.
    pub fn block_cell(cells: usize, t: Mat3) -> UnitCell {
        let base = A * cells as f64;
        // `lamda2x` sums over the matrix's *rows*: `x_d = Σ_i h[i][d]·λ_i`. For that to reproduce
        // the map it was handed (`x_d = Σ_d t[d][i]·v_i`) the cell's rows have to be `t`'s columns,
        // i.e. `tᵀ` read as a matrix of row vectors. Passing `t` unchanged applies `tᵀ`, which is
        // indistinguishable from `t` for the diagonal maps these fixtures use for dilation and is
        // exactly wrong for a rotation — the recovered rotation came back mirrored.
        let tr = t.transposed();
        let scaled = Mat3 {
            m: [
                [tr.m[0][0] * base, tr.m[0][1] * base, tr.m[0][2] * base],
                [tr.m[1][0] * base, tr.m[1][1] * base, tr.m[1][2] * base],
                [tr.m[2][0] * base, tr.m[2][1] * base, tr.m[2][2] * base],
            ],
        };
        UnitCell::from_matrix(scaled, [true; 3], [0.0; 3])
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{block_cell, fcc_block, fcc_sites, frac};
    use super::*;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::sc::Sc;
    use crate::lattice::Lattice;

    const A: f64 = 4.0;

    /// A block of `cells³` cells with one site each, so the sites are the simple-cubic lattice
    /// and the periodic images are the rest of it.
    fn sc_block(cells: usize) -> (UnitCell, Vec<Coord3>) {
        let cell = UnitCell::orthorhombic(A * cells as f64, A * cells as f64, A * cells as f64);
        let mut sites = Vec::new();
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    sites.push([i as f64 * A, j as f64 * A, k as f64 * A]);
                }
            }
        }
        (cell, sites)
    }

    /// The same block, pushed through an affine map `x ↦ T x`, periodic in the *deformed* cell.
    ///
    /// The cell is rebuilt as `T·[0, L)`, so the deformed structure tiles its own box rather
    /// than being cut by the undeformed one — otherwise the deformation would show up as a
    /// surface and the test would measure the wrong thing.
    fn affine_block(cells: usize, t: Mat3) -> (UnitCell, Vec<Coord3>) {
        // `from_matrix` takes *rows as lattice vectors* (cell.rs), so the box of the deformed
        // block has to be built by mapping each axis vector, not by passing `t` itself. Passing
        // the map gives a box of edge `t·e_a` instead of `t·(L·e_a)` — a box 12× too small, whose
        // only symptom is a mesh squeezed to two nodes per axis and no measured strain.
        // The sites then have to be placed by the same convention as the box, or the two describe
        // different deformations; see `block_cell`.
        let cell = block_cell(cells, t);
        let mut sites = Vec::new();
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    sites.push(
                        cell.lamda2x(frac(cells, [i as f64 * A, j as f64 * A, k as f64 * A])),
                    );
                }
            }
        }
        (cell, sites)
    }

    fn sc_pc() -> PrimitiveCell {
        Sc { a: A }.primitive()
    }

    fn opts() -> RotoStrainOptions {
        RotoStrainOptions {
            node_spacing: A,
            sigma: 3.0 * A,
            refine_passes: 0,
            max_deviation: None,
            fill_unsolved: true,
        }
    }

    /// The entry that every theory claim needs: `C A⁻¹` on an exactly-known pair set.
    #[test]
    fn fit_recovers_a_known_affine_map() {
        let t = Mat3::from_rows([1.1, 0.03, -0.02], [0.0, 0.97, 0.05], [0.01, 0.0, 1.05]);
        let template: Vec<Coord3> = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        .to_vec();
        let mut mm = Moments::default();
        for a in template.iter() {
            let b = t.mul_v(*a);
            mm.add(*a, b, 1.0);
        }
        let got = mm.solve().expect("6 independent pairs must be solvable");
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got.m[r][c] - t.m[r][c]).abs() < 1e-10,
                    "T[{r}][{c}] = {} not {}",
                    got.m[r][c],
                    t.m[r][c]
                );
            }
        }
    }

    /// Weights must not bias the answer: a pair at weight 1e-3 contributes 1e-3 as much, so the
    /// solve has to stay exact on the pairs that carry the weight.
    #[test]
    fn weights_scale_the_moments_not_the_answer() {
        let t = Mat3::from_rows([1.2, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.9]);
        let dirs: Vec<Coord3> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].to_vec();
        let mut mm = Moments::default();
        for (i, a) in dirs.iter().enumerate() {
            mm.add(*a, t.mul_v(*a), 10f64.powi(i as i32));
        }
        let got = mm.solve().expect("3 independent directions are solvable");
        for r in 0..3 {
            for c in 0..3 {
                assert!((got.m[r][c] - t.m[r][c]).abs() < 1e-10);
            }
        }
    }

    /// The sign rule is the whole pairing argument, so pin it: a Delaunay template is symmetric,
    /// and matching both halves of one direction class to the same observed vector would fit an
    /// antisymmetry the crystal does not have.
    #[test]
    fn template_matching_takes_the_positive_side() {
        let template = [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]];
        let observed = vec![[2.4, 0.0, 0.0], [-2.4, 0.0, 0.0]];
        let pairs = match_template(&template, &observed, None);
        assert_eq!(pairs.len(), 2);
        // +template must pair with the +observed vector, -template with the -observed one.
        assert!(dot(pairs[0].1, template[0]) > 0.0);
        assert!(dot(pairs[1].1, template[1]) > 0.0);
        // ...which is the only pairing under which one T explains both. A collinear template
        // cannot pin an affine map at all, and saying so beats returning a plausible-looking
        // matrix fitted to whatever subspace it happened to span.
        let mut mm = Moments::default();
        for (a, b) in pairs.iter() {
            mm.add(*a, *b, 1.0);
        }
        assert!(mm.solve().is_none(), "collinear pairs must not solve");
    }

    /// The positive-side rule inside a template that *can* determine `T`: scaling one axis must
    /// come back as a diagonal strain of exactly that factor.
    #[test]
    fn positive_side_pairing_recovers_axis_scaling() {
        let template: Vec<Coord3> = [
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, -4.0],
        ]
        .to_vec();
        let observed = vec![
            [4.8, 0.0, 0.0],
            [-4.8, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, -4.0],
        ];
        let mut mm = Moments::default();
        for (a, b) in match_template(&template, &observed, None) {
            mm.add(a, b, 1.0);
        }
        let t = mm.solve().expect("3 independent directions are solvable");
        assert!((t.m[0][0] - 1.2).abs() < 1e-12, "T_xx = {}", t.m[0][0]);
        assert!((t.m[1][1] - 1.0).abs() < 1e-12);
        assert!((t.m[2][2] - 1.0).abs() < 1e-12);
    }

    /// Rejection is a bias, not just a filter, so it has to be optional and visible.
    #[test]
    fn rejection_drops_distant_pairs() {
        let template = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let good = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        // Rejection is a *deviation* limit, so an exact match passes it however tight it is. A
        // limit below the observed deviation is what drops a pair.
        assert_eq!(match_template(&template, &good, None).len(), 3);
        assert_eq!(match_template(&template, &good, Some(0.1)).len(), 3);
        // The test is `deviation > limit`, so a limit of exactly 0 still admits a pair with zero
        // deviation. Near-exact is what it rejects.
        let near = vec![[1.001, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_eq!(match_template(&template, &near, Some(0.0)).len(), 2);
        let noisy = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        assert_eq!(match_template(&template, &noisy, Some(0.5)).len(), 2);
        assert_eq!(match_template(&template, &noisy, Some(2.5)).len(), 3);
    }

    /// `template_from_normals` must invert `face_normals` exactly, since both describe the same
    /// lattice and any drift between them would show up as a strain that is not there.
    #[test]
    fn template_inverts_the_normal_set() {
        for pc in [Fcc { a: A }.primitive(), Sc { a: A }.primitive()] {
            let normals = super::super::void_b::face_normals(&pc).unwrap();
            let template = template_from_normals(&normals).unwrap();
            assert_eq!(template.len(), normals.len());
            for &t in template.iter() {
                let t2 = dot(t, t);
                let n = scale(t, 2.0 / t2);
                // The recovered normal must be one of the ones we started from.
                assert!(
                    normals.iter().any(|m| dist2(*m, n) < 1e-18),
                    "normal {n:?} not in the original set"
                );
            }
        }
    }

    /// The identity case, which is also the case that must not be *trivially* true: if the fit
    /// returned identity by construction, this would pass while measuring nothing.
    #[test]
    fn perfect_crystal_gives_the_identity() {
        let (cell, sites) = sc_block(3);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        assert!(
            f.max_deviation_from_identity() < 1e-9,
            "undeformed crystal gave a field deviating by {}",
            f.max_deviation_from_identity()
        );
        assert_eq!(f.unsolved(), 0);
        // Identity to machine precision also at interpolated points, not just nodes.
        for x in [[1.3, 0.7, 2.1], [5.5, 5.5, 5.5]] {
            let t = f.at(x);
            for r in 0..3 {
                for c in 0..3 {
                    let target = if r == c { 1.0 } else { 0.0 };
                    assert!((t.m[r][c] - target).abs() < 1e-9);
                }
            }
        }
    }

    /// The load-bearing test. A uniform dilation is exactly representable, so the field must
    /// report the strain and *not* the identity — a fit that returned identity here would be
    /// measuring nothing while looking clean.
    #[test]
    fn uniform_dilation_is_recovered() {
        let s = 1.06;
        let t = Mat3::from_rows([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]);
        let (cell, sites) = affine_block(3, t);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        let got = f.at([2.0, 2.0, 2.0]);
        for d in 0..3 {
            assert!(
                (got.m[d][d] - s).abs() < 0.02,
                "T_{d}{d} = {} not {s}",
                got.m[d][d]
            );
        }
        // And it must not be identity: that is the difference between measuring and not.
        assert!(f.max_deviation_from_identity() > 0.03);
    }

    /// A rotation is the case the rigid field gets most wrong, because it keeps the normals
    /// fixed in the frame while the atoms rotate out of it.
    #[test]
    fn rotation_is_recovered() {
        let ang: f64 = 0.4;
        let (c, s) = (ang.cos(), ang.sin());
        let t = Mat3::from_rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]);
        let (cell, sites) = affine_block(3, t);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        let got = f.at([2.0, 2.0, 2.0]);
        for r in 0..3 {
            for c2 in 0..3 {
                assert!(
                    (got.m[r][c2] - t.m[r][c2]).abs() < 0.02,
                    "T[{r}][{c2}] = {} not {}",
                    got.m[r][c2],
                    t.m[r][c2]
                );
            }
        }
    }

    /// Shear is the off-diagonal case, so it is the one a diagonal-only or symmetric-only fit
    /// would silently miss.
    #[test]
    fn shear_is_recovered() {
        let g = 0.12;
        let t = Mat3::from_rows([1.0, g, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let (cell, sites) = affine_block(3, t);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        let got = f.at([2.0, 2.0, 2.0]);
        assert!(
            (got.m[0][1] - g).abs() < 0.02,
            "T_xy = {} not {g}",
            got.m[0][1]
        );
    }

    /// Trilinear interpolation is continuous, which the φ field depends on: a step would put a
    /// surface in the crystal that is not in the material.
    #[test]
    fn interpolation_is_continuous_across_node_planes() {
        let t = Mat3::from_rows([1.0, 0.1, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 0.97]);
        let (cell, sites) = affine_block(3, t);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        let eps = 1e-7;
        for x in [[1.5, 2.0, 2.5], [3.0, 1.25, 0.75]] {
            let a = f.at(x);
            for d in 0..3 {
                let mut y = x;
                y[d] += eps;
                let b = f.at(y);
                let jump = (0..3)
                    .flat_map(|r| (0..3).map(move |c| (b.m[r][c] - a.m[r][c]).abs()))
                    .fold(0.0, f64::max);
                assert!(jump < 1e-5, "jump {jump} across axis {d} at {x:?}");
            }
        }
    }

    /// Periodicity is what lets the φ field wrap, and it is not automatic: a mesh that merely
    /// covers the box would return the wrong `T` for a query point past the last node.
    #[test]
    fn field_is_periodic_across_the_boundary() {
        let (cell, sites) = sc_block(3);
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &opts()).unwrap();
        let l = cell.lengths();
        for d in 0..3 {
            let mut x = [1.7, 2.3, 0.9];
            let mut y = x;
            x[d] = 0.4;
            y[d] = 0.4 + l[d];
            let a = f.at(x);
            let b = f.at(y);
            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        (a.m[r][c] - b.m[r][c]).abs() < 1e-12,
                        "axis {d} not periodic: {} vs {}",
                        a.m[r][c],
                        b.m[r][c]
                    );
                }
            }
        }
    }

    /// Coarse-to-fine must actually end finer, and must not change what the field *means* — the
    /// schedule is an optimisation, so a perfect crystal stays identity at every resolution.
    #[test]
    fn refinement_preserves_a_perfect_crystal_and_increases_resolution() {
        let (cell, sites) = sc_block(3);
        let mut o = opts();
        o.refine_passes = 2;
        o.node_spacing = A / 2.0;
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &o).unwrap();
        assert!(f.max_deviation_from_identity() < 1e-9);
        assert_eq!(f.refine_passes(), 2);
        // spacing == lengths/dims, so more passes must mean a strictly finer mesh.
        assert!(f.spacing()[0] < A);
        assert!(f.len_nodes() > 27);
    }

    /// FCC is the real target structure, and the one where a broken template or a mis-sized `k`
    /// shows up: 12 neighbours, and the template has to pair all of them.
    #[test]
    fn fcc_block_fits_all_twelve_neighbours() {
        let pc = Fcc { a: A }.primitive();
        let cells = 3;
        let cell = UnitCell::orthorhombic(A * cells as f64, A * cells as f64, A * cells as f64);
        let sites = fcc_sites(cells, Mat3::identity());
        let normals = super::super::void_b::face_normals(&pc).unwrap();
        let mut o = opts();
        o.node_spacing = A / std::f64::consts::SQRT_2;
        let f = RotoStrainField::fit(&cell, &sites, &pc, &o).unwrap();
        assert_eq!(normals.len(), 12);
        assert_eq!(f.n_atoms(), sites.len());
        assert_eq!(f.n_pairs(), sites.len() * 12, "every atom must pair all 12");
        assert!(f.max_deviation_from_identity() < 1e-9);
    }

    /// The block must be a crystal, all the way to its declared edges.
    ///
    /// Cheapest possible statement of "no gap", and the reason `fcc_sites` indexes `0..2 * cells`:
    /// counting sites is enough. A gap shows up immediately as a missing fraction of the atoms a
    /// commensurate block must have, so this fails loudly instead of quietly feeding the fit a
    /// lattice with a missing plane in it.
    #[test]
    fn fcc_block_fills_the_box_it_declares() {
        let (cell, sites, pc) = fcc_block(3);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for s in sites.iter() {
            for d in 0..3 {
                lo[d] = lo[d].min(s[d]);
                hi[d] = hi[d].max(s[d]);
            }
        }
        for d in 0..3 {
            // Sites sit on multiples of `a / 2`, so the outermost plane is half a lattice constant
            // short of the edge — but only one such step, not `a / 2` of nothing.
            assert!(
                lo[d] < A / 4.0,
                "axis {d} starts at {}, leaving a gap at the low edge",
                lo[d]
            );
            assert!(
                hi[d] > cell.lengths()[d] - A,
                "axis {d} stops at {}, leaving a gap at the high edge (box {})",
                hi[d],
                cell.lengths()[d]
            );
        }
        // FCC is four sites per `a^3`.
        let cells = (cell.lengths()[0] / A).round() as usize;
        let expected = 4 * cells.pow(3);
        assert_eq!(sites.len(), expected, "4 sites per conventional cell");
        assert_eq!(
            sites.len(),
            4 * 3usize.pow(3),
            "and no coincident duplicates"
        );
        assert_eq!(pc.basis.len(), 4);
    }

    /// Every atom of a rotated FCC block must recover the rotation, not the field on average.
    ///
    /// This is the assertion that found the real defect. Node values are Gaussian-weighted means,
    /// so a neighbourhood containing a few badly-fitted atoms still averages close to the truth —
    /// `fcc_block_fits_all_twelve_neighbours` passed against a block that was missing a plane in
    /// every direction, for exactly that reason. Checking each atom separately has no such escape:
    /// a wrong pairing shows up as a wrong matrix on that atom and nowhere else.
    ///
    #[test]
    fn every_atom_of_a_rotated_fcc_block_recovers_the_rotation() {
        let ang: f64 = 0.35;
        let (c, s) = (ang.cos(), ang.sin());
        let rot = Mat3::from_rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]);
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let cell = block_cell(cells, rot);
        let sites = fcc_sites(cells, rot);

        let normals = super::super::void_b::face_normals(&pc).unwrap();
        let tmpl = template_from_normals(&normals).unwrap();
        assert_eq!(tmpl.len(), 12);
        let reach = tmpl.iter().map(|v| dot(*v, *v).sqrt()).fold(0.0, f64::max);
        let list = CellList::build(&cell, &sites, reach * 1.2).unwrap();
        let mut scr = Vec::new();
        let mut worst = 0.0f64;
        for i in 0..sites.len() {
            nearest_k_excluding_self(&list, i, tmpl.len(), &mut scr).unwrap();
            let observed: Vec<Coord3> = scr.iter().map(|e| e.1).collect();
            let pairs = match_template(&tmpl, &observed, None);
            assert_eq!(pairs.len(), 12, "atom {i} must pair all twelve");
            let mut mm = Moments::default();
            for (a, b) in pairs.iter() {
                mm.add(*a, *b, 1.0);
            }
            let got = mm.solve().unwrap();
            let e: f64 = (0..3)
                .map(|a| {
                    (0..3)
                        .map(|b| (got.m[a][b] - rot.m[a][b]).abs())
                        .sum::<f64>()
                })
                .sum();
            worst = worst.max(e);
        }
        assert!(worst < 1e-9, "worst per-atom error {worst} exceeds 1e-9");
    }

    /// A strained FCC block must report the strain, and by the right amount — this is the
    /// composition that the HEA fixture is made of.
    #[test]
    fn fcc_block_reports_its_dilation() {
        let pc = Fcc { a: A }.primitive();
        let s = 1.05;
        let t = Mat3::from_rows([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]);
        let cells = 3;
        let (cell, _) = affine_block(cells, t);
        let sites = fcc_sites(cells, t);
        let mut o = opts();
        o.node_spacing = A / std::f64::consts::SQRT_2;
        let f = RotoStrainField::fit(&cell, &sites, &pc, &o).unwrap();
        let got = f.at(sites[sites.len() / 2]);
        for d in 0..3 {
            assert!(
                (got.m[d][d] - s).abs() < 0.02,
                "T_{d}{d} = {} not {s}",
                got.m[d][d]
            );
        }
    }

    /// Bad arguments must be refused rather than absorbed. A NaN sigma builds a mesh whose every
    /// weight is NaN, which solves to nothing and then reads as "no strain".
    #[test]
    fn invalid_options_are_refused() {
        let (cell, sites) = sc_block(2);
        let pc = sc_pc();
        for bad in [
            RotoStrainOptions {
                sigma: 0.0,
                ..opts()
            },
            RotoStrainOptions {
                sigma: f64::NAN,
                ..opts()
            },
            RotoStrainOptions {
                node_spacing: -1.0,
                ..opts()
            },
            RotoStrainOptions {
                node_spacing: f64::INFINITY,
                ..opts()
            },
        ] {
            assert!(RotoStrainField::fit(&cell, &sites, &pc, &bad).is_err());
        }
        assert!(RotoStrainField::fit(&cell, &[], &pc, &opts()).is_err());
        assert!(RotoStrainField::fit_with_template(&cell, &sites, &[], &opts()).is_err());
    }

    /// A node with no solvable neighbourhood is recorded, not silently replaced, unless filling
    /// is explicitly allowed — the count is what tells a caller how much of the field is real.
    #[test]
    fn unsolved_nodes_are_counted() {
        let (cell, sites) = sc_block(2);
        let mut o = opts();
        // sigma far below the neighbour distance, and a node spacing that does not divide the box
        // so the nodes miss the sites: with a commensurate spacing every node sits *on* an atom and
        // solves perfectly well, which would make this test measure nothing.
        o.sigma = 0.1;
        o.node_spacing = 3.0;
        let f = RotoStrainField::fit(&cell, &sites, &sc_pc(), &o).unwrap();
        assert!(f.unsolved() > 0, "tiny sigma must leave unsolved nodes");
        // The field still answers, rather than panicking, but says how much it does not know.
        let t = f.at([1.0, 1.0, 1.0]);
        assert_eq!(t.m.len(), 3);
        let strict = RotoStrainOptions {
            fill_unsolved: false,
            ..o
        };
        assert!(RotoStrainField::fit(&cell, &sites, &sc_pc(), &strict).is_err());
    }
}

#[cfg(test)]
mod integration {
    //! Does the roto-strain field actually change what φ measures?
    //!
    //! These are the tests that justify wiring the field in at all. Everything else in this module
    //! checks that the arithmetic is right; these check that the arithmetic is aimed at something.

    use super::fixtures::{block_cell, fcc_block, fcc_sites};
    use super::*;
    use crate::analysis::void_b::estimate_void_volume;
    use crate::analysis::void_b::PhiField;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::sc::Sc;
    use crate::lattice::Lattice;

    const A: f64 = 4.0;

    /// A threshold just above the cell boundary. `estimate_void_volume` refuses `ε = 0` — a
    /// threshold *at* φ = 1 is not above it, and accepting it would make the void set measure
    /// zero by definition — so the "must not report void" tests use the smallest value the API
    /// allows rather than the one that reads most naturally.
    const TINY: f64 = 1e-9;

    fn opts() -> RotoStrainOptions {
        RotoStrainOptions {
            node_spacing: A,
            sigma: 3.0 * A,
            refine_passes: 1,
            max_deviation: None,
            fill_unsolved: true,
        }
    }

    /// Deterministic thermal jitter of `amp` (as a fraction of the FCC nearest-neighbour
    /// distance) — SplitMix64 rather than `rand` so a reported number is reproducible, which is the
    /// whole point of seeding the MC estimator in the first place.
    fn jitter(sites: &[Coord3], amp: f64) -> Vec<Coord3> {
        let mut rng: u64 = 0x5eed_1234;
        let amp = amp * A / std::f64::consts::SQRT_2;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            amp * ((rng >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5)
        };
        sites
            .iter()
            .map(|p| [p[0] + next(), p[1] + next(), p[2] + next()])
            .collect()
    }

    /// A rotated crystal is the clearest case where the rigid field is wrong: the face normals stay
    /// fixed in the frame while every atom rotates out of it. The strained field has to measure the
    /// result as perfectly packed. This is a correctness test, not an accuracy improvement.
    #[test]
    fn rotated_perfect_lattice_is_not_void_once_strained() {
        let ang: f64 = 0.35;
        let (c, s) = (ang.cos(), ang.sin());
        let rot = Mat3::from_rows([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]);
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let base = A * cells as f64;
        let cell = block_cell(cells, rot);
        let sites = fcc_sites(cells, rot);
        let region = [(0.0, base), (0.0, base), (0.0, base)];
        let rigid = PhiField::for_threshold(&cell, &sites, &pc, TINY).unwrap();
        let strain = RotoStrainField::fit(&cell, &sites, &pc, &opts()).unwrap();
        let deformed =
            PhiField::for_threshold_strained(&cell, &sites, &pc, TINY, Some(strain)).unwrap();

        let v_rigid = estimate_void_volume(&rigid, &region, TINY, 20_000, 7).unwrap();
        let v_deformed = estimate_void_volume(&deformed, &region, TINY, 20_000, 7).unwrap();

        assert!(
            v_deformed.volume < 1e-6,
            "a rotated perfect lattice must not measure void; got {} A^3",
            v_deformed.volume
        );
        // The control, at the magnitude it actually has.
        //
        // This test was written expecting the rigid field to read a 20°-rotated crystal as *mostly*
        // void. It does not: it reads 244 A^3 of 4096, about 6 %. That is the honest size of the
        // effect, and it is worth being precise about why, because the earlier number was an
        // artefact rather than a measurement. The FCC interstitial polyhedron is the rhombic
        // dodecahedron and its twelve faces are already normal to <110>, so rotating the crystal
        // moves a polyhedron that is nearly congruent with the one the template describes — the
        // disagreement is the sliver between the two, not the volume of either.
        //
        // The claim the method needs is therefore comparative, not absolute: the rotation has to
        // register as *some* void for the test to mean anything, and strain has to remove all of
        // it rather than most of it. Measured: 244.5 A^3 rigid, below 1e-6 strained.
        assert!(
            v_rigid.volume > 0.03 * base * base * base,
            "the rigid field was expected to register the rotation, got only {} A^3",
            v_rigid.volume
        );
        assert!(
            v_deformed.volume < 1e-3 * v_rigid.volume,
            "strain should remove the rotation's void, not merely shrink it: {} vs {} A^3",
            v_deformed.volume,
            v_rigid.volume
        );
    }

    /// The claim the whole method rests on: thermal expansion is a *uniform dilation*, which is
    /// exactly representable, so a lattice dilated by 2 % must measure as defect-free. The rigid
    /// field cannot do this — it has no way to know the atoms moved coherently.
    #[test]
    fn uniformly_dilated_lattice_is_not_void_once_strained() {
        let s = 1.02;
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let (_cell, sites, _) = fcc_block(cells);
        // Scale the whole thing — cell and sites together — so it stays commensurate.
        let dil = Mat3::from_rows([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]);
        let base = A * cells as f64;
        let dcell = block_cell(cells, dil);
        let dsites: Vec<Coord3> = sites.iter().map(|p| dil.mul_v(*p)).collect();
        let region = [(0.0, base * s), (0.0, base * s), (0.0, base * s)];

        let rigid = PhiField::for_threshold(&dcell, &dsites, &pc, TINY).unwrap();
        let strain = RotoStrainField::fit(&dcell, &dsites, &pc, &opts()).unwrap();
        let deformed =
            PhiField::for_threshold_strained(&dcell, &dsites, &pc, TINY, Some(strain)).unwrap();
        let v_rigid = estimate_void_volume(&rigid, &region, TINY, 20_000, 7).unwrap();
        let v_deformed = estimate_void_volume(&deformed, &region, TINY, 20_000, 7).unwrap();
        assert!(
            v_deformed.volume < 1e-6,
            "2 % dilation must not read as void; got {} A^3",
            v_deformed.volume
        );
        assert!(
            v_rigid.volume > 0.05 * (base * s).powi(3),
            "the rigid field was expected to fail here, got {} A^3",
            v_rigid.volume
        );
    }

    /// And the converse, which is what protects against a field that "fixes" void by absorbing
    /// it: a real cavity carved out of a dilated lattice must still measure as void, at close to
    /// its true size. If strain made voids invisible, this would collapse to zero.
    #[test]
    fn a_strained_lattice_still_measures_a_real_void() {
        let s = 1.02;
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let (_, sites, _) = fcc_block(cells);
        let base = A * cells as f64;
        let mid = base / 2.0;

        // Carve a sphere of ~2 Å radius about the box centre, then dilate everything.
        // FCC first shell is a/sqrt(2) = 2.83 A, so 2.4 removed a single site. 3.4 takes the
        // central site plus its 12, which is a cavity with real volume to measure.
        let carve_r = 3.4;
        let kept: Vec<Coord3> = sites
            .iter()
            .filter(|p| {
                let d = [p[0] - mid, p[1] - mid, p[2] - mid];
                dot(d, d).sqrt() > carve_r
            })
            .copied()
            .collect();
        let removed = sites.len() - kept.len();
        assert!(removed >= 8, "test needs a cavity with real volume");

        let dil = Mat3::from_rows([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]);
        let dcell = block_cell(cells, dil);
        let kept: Vec<Coord3> = kept.iter().map(|p| dil.mul_v(*p)).collect();
        // Region around the cavity, kept well inside the box so periodic images are irrelevant.
        let r0 = mid - 4.0;
        let r1 = mid + 4.0;
        let region = [(r0, r1), (r0, r1), (r0, r1)];

        let strain = RotoStrainField::fit(&dcell, &kept, &pc, &opts()).unwrap();
        let deformed =
            PhiField::for_threshold_strained(&dcell, &kept, &pc, 1e-3, Some(strain)).unwrap();
        let v = estimate_void_volume(&deformed, &region, TINY, 200_000, 11).unwrap();
        let carved = 4.0 / 3.0 * std::f64::consts::PI * (carve_r * s).powi(3);
        assert!(
            v.volume > 0.4 * carved,
            "cavity measured {} A^3, expected a substantial fraction of {carved} A^3",
            v.volume
        );
        assert!(
            v.volume < 2.0 * carved,
            "cavity measured {} A^3, implausibly larger than {carved} A^3",
            v.volume
        );
    }

    /// A coherent strain plus thermal noise: the case the field is actually for.
    ///
    /// The two parts play different roles and the test needs both. A uniform dilation opens a gap
    /// between every pair of neighbouring atoms that no amount of smoothing can close, and `T`
    /// closes it exactly — that is the whole mechanism. Jitter on top is what a real relaxed crystal
    /// has, and it is there to check the dilation is still recovered through the noise rather than
    /// only in the noiseless limit.
    #[test]
    fn strain_reduces_the_shell_of_a_dilated_jittered_crystal() {
        let pc = Fcc { a: A }.primitive();
        let cells = 5;
        let base = A * cells as f64;
        let (cell, sites, _) = fcc_block(cells);
        let s = 1.05;
        let t = Mat3::from_rows([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]);
        // Jitter after the strain, so it is not scaled along with it — a real crystal is dilated and
        // then relaxes.
        let noisy = jitter(&sites, 0.04);
        let stretched: Vec<Coord3> = noisy.iter().map(|p| t.mul_v(*p)).collect();
        let dcell = UnitCell::orthorhombic(
            cell.lengths()[0] * s,
            cell.lengths()[1] * s,
            cell.lengths()[2] * s,
        );
        let region = [(2.0, base - 2.0), (2.0, base - 2.0), (2.0, base - 2.0)];

        let rigid = PhiField::for_threshold(&dcell, &stretched, &pc, TINY).unwrap();
        let v_rigid = estimate_void_volume(&rigid, &region, TINY, 200_000, 3).unwrap();

        let strain = RotoStrainField::fit(&dcell, &stretched, &pc, &opts()).unwrap();
        let deformed =
            PhiField::for_threshold_strained(&dcell, &stretched, &pc, TINY, Some(strain)).unwrap();
        let v_deformed = estimate_void_volume(&deformed, &region, TINY, 200_000, 3).unwrap();

        let vol = (base - 4.0).powi(3);
        assert!(
            v_rigid.volume > 0.05 * vol,
            "a 5 % dilation should tear the rigid field open; got {} of {vol} A^3",
            v_rigid.volume
        );
        assert!(
            v_deformed.volume < 0.25 * v_rigid.volume,
            "strain should close most of it: {} vs {} A^3",
            v_deformed.volume,
            v_rigid.volume
        );
    }

    /// Pure thermal jitter: the field cannot help, and the honest thing is to say so in a test.
    ///
    /// Random displacements have no coherent component, so the least-squares affine fit at every
    /// node converges on the identity and `T(x) = I` — the strained field is then the rigid field
    /// and the shell is exactly as large. A test asserting a reduction here would be asserting that
    /// an affine map can absorb non-affine noise, and it passed at 2 % jitter only because the two
    /// numbers were close enough for the Monte Carlo error to decide the sign. At 4 % the
    /// difference is not a reduction at all: 79.4 A^3 strained against 78.9 A^3 rigid.
    ///
    /// So the residual shell is a property of the φ definition with ideal face normals, not a
    /// missing correction, and pinning that here is what stops the number being rediscovered later
    /// as a bug in `RotoStrainField`.
    #[test]
    fn strain_cannot_remove_a_non_affine_shell() {
        let pc = Fcc { a: A }.primitive();
        let cells = 5;
        let base = A * cells as f64;
        let (cell, sites, _) = fcc_block(cells);
        let jittered = jitter(&sites, 0.04);
        let region = [(2.0, base - 2.0), (2.0, base - 2.0), (2.0, base - 2.0)];

        let rigid = PhiField::for_threshold(&cell, &jittered, &pc, TINY).unwrap();
        let v_rigid = estimate_void_volume(&rigid, &region, TINY, 200_000, 3).unwrap();

        let mut o = opts();
        o.sigma = 2.0 * A;
        let strain = RotoStrainField::fit(&cell, &jittered, &pc, &o).unwrap();
        let deformed =
            PhiField::for_threshold_strained(&cell, &jittered, &pc, TINY, Some(strain.clone()))
                .unwrap();
        let v_deformed = estimate_void_volume(&deformed, &region, TINY, 200_000, 3).unwrap();

        let vol = (base - 4.0).powi(3);
        assert!(
            v_rigid.volume > 0.01 * vol,
            "the rigid shell should be measurable, got {} of {vol} A^3",
            v_rigid.volume
        );
        // No coherent strain to find, so the field is the identity to within fitting noise. The
        // slack is generous because both numbers carry Monte Carlo error, but it is far tighter
        // than any reduction would need to be.
        assert!(
            strain.max_deviation_from_identity() < 0.01,
            "pure jitter should leave no coherent strain, got {}",
            strain.max_deviation_from_identity()
        );
        assert!(
            (v_deformed.volume - v_rigid.volume).abs() < 0.15 * v_rigid.volume,
            "strain should not change a non-affine shell: {} vs {} A^3",
            v_deformed.volume,
            v_rigid.volume
        );
    }

    /// The rejection test has to be measured, not assumed: it biases the fit toward the ideal
    /// lattice, so what it costs has to be shown rather than argued.
    ///
    /// Two claims, and the first one is a *negative* result worth pinning: at Mason's `a₀/4`,
    /// a crystal with ordinary thermal jitter rejects nothing at all. `a₀/4` is a defect test, not
    /// a noise filter, and a test that quietly assumed otherwise would have set σ from a limit that
    /// never bites.
    #[test]
    fn rejection_at_a0_over_4_ignores_thermal_jitter() {
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let base = A * cells as f64;
        let cell = UnitCell::orthorhombic(base, base, base);
        let (_, sites, _) = fcc_block(cells);
        let jittered = jitter(&sites, 0.02);

        let mut with = opts();
        with.max_deviation = Some(A / 4.0);
        let mut without = opts();
        without.max_deviation = None;
        let f_with = RotoStrainField::fit(&cell, &jittered, &pc, &with).unwrap();
        let f_without = RotoStrainField::fit(&cell, &jittered, &pc, &without).unwrap();
        assert_eq!(
            f_with.n_pairs(),
            f_without.n_pairs(),
            "a0/4 must not reject thermal noise"
        );
        assert_eq!(f_with.n_pairs(), sites.len() * 12);
    }

    /// ...and where it *does* bite, it bites by discarding the evidence near the defect, which is
    /// the intended effect but is a bias and not free. Measured both ways.
    #[test]
    fn rejection_bites_at_a_defect_and_biases_toward_the_ideal_lattice() {
        let pc = Fcc { a: A }.primitive();
        let cells = 4;
        let base = A * cells as f64;
        let cell = UnitCell::orthorhombic(base, base, base);
        let (_, sites, _) = fcc_block(cells);
        let mid = base / 2.0;
        let kept: Vec<Coord3> = sites
            .iter()
            .filter(|p| dist2(**p, [mid; 3]).sqrt() > 3.4)
            .copied()
            .collect();
        assert!(
            sites.len() - kept.len() >= 8,
            "need a cavity for this to bite"
        );

        let mut with = opts();
        with.max_deviation = Some(A / 4.0);
        let mut without = opts();
        without.max_deviation = None;
        let f_with = RotoStrainField::fit(&cell, &kept, &pc, &with).unwrap();
        let f_without = RotoStrainField::fit(&cell, &kept, &pc, &without).unwrap();

        assert!(
            f_with.n_pairs() < f_without.n_pairs(),
            "rejection dropped nothing: {} vs {} pairs",
            f_with.n_pairs(),
            f_without.n_pairs()
        );
        assert!(
            f_with.max_deviation_from_identity() < f_without.max_deviation_from_identity(),
            "rejection should bias the field toward the ideal lattice: {} vs {}",
            f_with.max_deviation_from_identity(),
            f_without.max_deviation_from_identity()
        );
    }

    /// An SC block is a weaker template (6 neighbours, all axis-aligned) and the degenerate
    /// neighbour shell is worth one check that the integration does not depend on FCC.
    #[test]
    fn simple_cubic_void_survives_strain() {
        let pc = Sc { a: A }.primitive();
        let cells = 4;
        let base = A * cells as f64;
        let mut sites = Vec::new();
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    sites.push([i as f64 * A, j as f64 * A, k as f64 * A]);
                }
            }
        }
        let mid = base / 2.0;
        let carve_r = 3.2;
        let kept: Vec<Coord3> = sites
            .iter()
            .filter(|p| dist2(**p, [mid; 3]).sqrt() > carve_r)
            .copied()
            .collect();
        let cell = UnitCell::orthorhombic(base, base, base);
        let region = [(mid - 4.0, mid + 4.0); 3];
        let strain = RotoStrainField::fit(&cell, &kept, &pc, &opts()).unwrap();
        let f = PhiField::for_threshold_strained(&cell, &kept, &pc, 1e-3, Some(strain)).unwrap();
        let v = estimate_void_volume(&f, &region, 1e-3, 200_000, 5).unwrap();
        let exact = 4.0 / 3.0 * std::f64::consts::PI * carve_r.powi(3);
        assert!(
            v.volume > 0.4 * exact && v.volume < 2.0 * exact,
            "SC cavity measured {} A^3 against {exact} A^3",
            v.volume
        );
    }
}
