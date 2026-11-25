//! Void method B (design §17.2): the local scalar field φ, and void volume as a superlevel set.
//!
//! Method A ([`super::void`]) counts clusters of *empty W–S sites*, so it can only see a
//! void that emptied whole sites. This method instead evaluates a field over space and takes
//! the volume where it exceeds 1, which sees a void of any size and, in a distorted lattice,
//! one whose cells never emptied at all.
//!
//! # The field
//!
//! The Wigner–Seitz cell around site `x_i` is the intersection of the half-spaces that
//! bisect `x_i` from its neighbours. A neighbour at `v̂` puts its face on the plane
//! `v̂·(x − x_i) = |v̂|²/2`, so with the **plane normal**
//!
//! ```text
//! n̂ = 2 v̂ / |v̂|²
//! ```
//!
//! that face reads `n̂·(x − x_i) = 1`, and the cell is `max_k |n̂_k·(x − x_i)| ≤ 1` over the
//! face normals `n̂_k`. Writing the per-atom max as `ψ_i(x)` and taking the minimum over atoms,
//!
//! ```text
//! φ(x) = min_i ψ_i(x)
//! ```
//!
//! makes `φ ≤ 1` exactly the union of the W–S cells, so **`φ > 1` is inside no crystal cell**
//! — the void region. That identity is what makes this testable rather than plausible: on a
//! perfect lattice it holds exactly, and `phi_is_one_on_every_cell_vertex_of_a_perfect_lattice`
//! asserts it.
//!
//! Note what φ is *not*: a distance. It is dimensionless, it is a minimum of maxima, and it is
//! not linear in space, so interpolating it or differencing it means nothing. Only its sign
//! against 1 carries meaning, which is why the estimator below thresholds it and never
//! differentiates it.
//!
//! # Scope: this measures displacement, not voids
//!
//! The default field is *rigid*: its face normals are the ideal cell's. Method B can also take
//! them from a local roto-strain field `T(x)` ([`super::roto_strain`], `--strain`), so that the
//! cells deform with the crystal rather than staying ideal; that narrows the shell below but
//! does not remove it.
//!
//! **Without it, φ does not measure voids, and this module's volume estimator must not be
//! reported as a void volume on a real crystal.** Holding the normals ideal while the centres
//! are the observed positions makes φ a measure of *how far the atoms are from their ideal
//! sites*, so `{φ > 1}` is a shell around every displaced atom rather than a cavity. Measured
//! on the relaxed 256k-atom HEA fixture, which method A reports as having **zero** defects:
//!
//! ```text
//! ε      V(ε) [Å³]    share of the 40×40 Å sample
//! 0.002    1900.8        2.97 %
//! 0.010    1339.7        2.09 %
//! 0.020     801.1        1.25 %
//! 0.050     114.1        0.18 %
//! 0.100       2.1        0.00 %
//! ```
//!
//! At ε = 0.002 that is ~3 % of the sample, or 0.33 Å³ per site — 3 % of a W–S cell — while
//! the seven *planted* voids in that volume would be 76.7 Å³. Over 95 % of what this reports
//! as void in a defect-free crystal is thermal shell. The same code on the *ideal* lattice at
//! the same thresholds returns exactly 0, which is what isolates the cause to displacement
//! rather than to the field.
//!
//! The consequence for the estimator is not a tuning problem. Extrapolating `V(ε)` back to
//! ε = 0 does not remove the shell, because the shell really is there at φ = 1: a linear fit
//! through the two smallest thresholds returns ~1800 Å³ and a quadratic through three returns
//! ~1450 Å³, both of which are extrapolations of a curve whose value at 0 is genuinely
//! ~1900 Å³. See [`extrapolate_to_phi_one`] on why a line through ≥2 ε values is not usable,
//! and [`extrapolate_shell_free`] for the window above the shell.
//!
//! So what this module *is* good for, and what the tests below pin down, is:
//!
//! - The **field itself**, which satisfies `φ = 1` on every cell boundary exactly and is
//!   correct as a function. That is what the synthetic tests verify, and it is the part the
//!   roto-strain version reuses.
//! - **Exactly-commensurate inputs**, where atoms sit on their ideal sites: an ideal lattice,
//!   or a defect pattern in an unrelaxed block. There φ > 1 really is void, and a single
//!   vacancy really does measure exactly one cell.
//! - **Comparison against a reference** — a measured region against the same region of the
//!   ideal lattice — which cancels the shell instead of trying to extrapolate it away.
//!
//! For a void volume with thermal displacements, the shell-free window
//! ([`extrapolate_shell_free`], `--thermal-window`) is the only method-B route that passes the
//! gate of design §17.2. Method B stays behind `--experimental`: method A is the production
//! answer.
//!
//! # Sampling
//!
//! [`PhiField::at_cell_vertices`] evaluates φ at [`ws_vertices`] — the corners of each site's
//! W–S cell — rather than on a cubical grid, so the sample count is O(atoms) and independent of
//! resolution. The volume is a Monte Carlo integral of `1[φ > 1 + ε]` over a stated region, with
//! a seeded generator so a reported volume and its error bar are reproducible.

use crate::analysis::roto_strain::RotoStrainField;
use crate::cell::{Coord3, Mat3, UnitCell};
use crate::error::{Error, Result};
use crate::lattice::PrimitiveCell;
use crate::neighbor::cell_list::CellList;

/// Cell translations enumerated when searching for the shortest lattice vectors.
///
/// Two is a safety margin, not a guess about any particular structure: the shortest vectors of
/// every lattice in the registry are within one cell of the origin, so ±1 already suffices and
/// ±2 only guards against a primitive cell that is not primitive in the direction searched.
const SEARCH_RANGE: i64 = 2;

/// Plane normals of the Wigner–Seitz faces around one site, as `n̂ = 2v̂/|v̂|²`.
///
/// One normal per **Voronoi-relevant** vector `v` from site 0 to another site: `v` bounds
/// the cell when the midpoint `v/2` lies strictly inside every other bisecting half-space
/// (`u·v < |u|²` for every other vector `u`). For FCC and SC these are exactly the first
/// shell (12, 6); for BCC they are the first *and second* shells (8 hexagonal faces and 6
/// square ones of the truncated octahedron). Taking only the shortest vectors would make the
/// BCC cell the octahedron `|x|+|y|+|z| ≤ 3a/4` (volume 0.5625 a³ against the true 0.5 a³),
/// whose overlapping cells cover part of a vacancy: every BCC void would read 5–20 % small
/// (design §17.2). The set is symmetric
/// under `v → −v`.
pub fn face_normals(pc: &PrimitiveCell) -> Result<Vec<Coord3>> {
    if pc.basis.is_empty() {
        return Err(Error::invalid("primitive cell has no basis atoms"));
    }
    let mut vecs: Vec<Coord3> = Vec::new();
    for m in -SEARCH_RANGE..=SEARCH_RANGE {
        for n in -SEARCH_RANGE..=SEARCH_RANGE {
            for p in -SEARCH_RANGE..=SEARCH_RANGE {
                let cell = [m as f64, n as f64, p as f64];
                for (j, &bj) in pc.basis.iter().enumerate() {
                    if j == 0 && m == 0 && n == 0 && p == 0 {
                        continue;
                    }
                    // Fractional offset from basis 0 to basis j in cell (m, n, p), then
                    // Cartesian. `basis` is fractional and `h` is the conversion; mixing
                    // the two is the mistake `nn_distance` already documents.
                    let d = [
                        cell[0] + bj[0] - pc.basis[0][0],
                        cell[1] + bj[1] - pc.basis[0][1],
                        cell[2] + bj[2] - pc.basis[0][2],
                    ];
                    let v = pc.h.mul_v(d);
                    if dot(v, v) > 0.0 {
                        vecs.push(v);
                    }
                }
            }
        }
    }
    let best = vecs
        .iter()
        .map(|&v| dot(v, v))
        .fold(f64::INFINITY, f64::min);
    if !best.is_finite() {
        return Err(Error::invalid("primitive cell has no lattice vectors"));
    }
    // A face's vector is shorter than twice the shortest (its midpoint must beat the
    // shortest vector's bisector); the rest cannot bound the cell.
    vecs.retain(|&v| dot(v, v) < 4.0 * best);
    let mut faces: Vec<Coord3> = Vec::new();
    for &v in &vecs {
        let relevant = vecs
            .iter()
            .all(|&u| dist2(u, v) < 1e-18 * best || dot(u, v) < dot(u, u) * (1.0 - 1e-9));
        if relevant && !faces.iter().any(|&w| dist2(w, v) < 1e-18 * best) {
            faces.push(v);
        }
    }
    Ok(faces.iter().map(|&v| scale(v, 2.0 / dot(v, v))).collect())
}

/// The corners of the Wigner–Seitz cell whose faces the given normals describe.
///
/// Each corner is where three face planes meet, so it is found by solving one 3×3 system per
/// triple and keeping the solutions that satisfy every other face. The normal set is symmetric,
/// so enumerating triples with all three right-hand sides at `+1` still finds corners that need
/// a left-hand side at `−1` — it does so through the `−n̂` member of the pair.
///
/// Returns the vertices *relative to the site at the origin*; the same corners translated to
/// each site are the sampling points for [`PhiField`].
pub fn ws_vertices(normals: &[Coord3]) -> Vec<Coord3> {
    let mut out: Vec<Coord3> = Vec::new();
    let m = normals.len();
    for i in 0..m {
        for j in i + 1..m {
            for k in j + 1..m {
                let rows = Mat3::from_rows(normals[i], normals[j], normals[k]);
                let Ok(v) = rows.solve([1.0, 1.0, 1.0]) else {
                    continue;
                };
                // Inside every half-space means inside the cell, and only then is this
                // triple's intersection a corner rather than a point beyond a face.
                if normals.iter().all(|&n| dot(n, v).abs() <= 1.0 + 1e-9)
                    && !out.iter().any(|u| dist2(*u, v) < 1e-12)
                {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Radius of the sphere containing the Wigner–Seitz cell.
///
/// This is the distance beyond which a point can no longer be inside the cell, and it is the
/// radius φ's neighbour query needs: a point whose nearest atom is further than this is outside
/// that atom's cell, and no further atom's cell can reach it either.
pub fn circumradius(normals: &[Coord3]) -> Result<f64> {
    let vs = ws_vertices(normals);
    if vs.is_empty() {
        return Err(Error::invalid(
            "Wigner-Seitz cell has no corners: the face normals do not bound a cell",
        ));
    }
    Ok(vs.iter().map(|v| dot(*v, *v).sqrt()).fold(0.0, f64::max))
}

/// The local scalar field φ, built on the *observed* atom positions and the *ideal*
/// cell's face normals.
///
/// Holding the normals ideal and the centres observed is deliberate: the gap between them is
/// the signal. An atom that has moved off its site no longer fills its own cell, φ rises above
/// 1 around it, and the region is counted as void. That is the same physics method A sees
/// discretely — an emptied site — expressed continuously, and it is why this module is a
/// refinement of A rather than an independent measurement.
///
/// # Radius
///
/// `radius` must reach every atom that could hold a query point *below a threshold*: because
/// φ is a minimum over atoms, leaving one out can only raise φ and invent void where there is
/// none. The exact requirement is not "as far as the minimising atom" — it is that
/// `{ψ_i(x) ≤ c}`, being a W–S cell scaled by `c`, lies inside the ball of radius `c·R` about
/// `x_i` for circumradius `R`. So searching `R(1+ε)` is *sufficient and minimal* for the
/// threshold `φ > 1+ε` that the volume estimator actually uses, and no larger.
///
/// [`PhiField::for_threshold`] derives that radius from `ε` instead of taking one, because a
/// radius that is merely safe is easy to get wrong in the expensive direction.
pub struct PhiField<'a> {
    normals: Vec<Coord3>,
    list: CellList<'a>,
    radius: f64,
    /// The largest ε whose `φ > 1 + ε` decision the radius is exact for:
    /// `radius / (R · inflate) − 1`, `R` the circumradius and `inflate` the strain's `1 + max|T − I|`.
    max_epsilon: f64,
    /// The local affine map (roto-strain field), or `None` for the rigid field.
    ///
    /// Held by value rather than by reference: it is a mesh of 3×3 maps (tens of kB to a few MB),
    /// it is built once and then queried per atom per sample point, and a lifetime on `PhiField`
    /// would ripple a `'a` through every `&PhiField` in the crate for no benefit.
    strain: Option<RotoStrainField>,
    scratch: std::cell::RefCell<Vec<(usize, Coord3)>>,
}

impl<'a> PhiField<'a> {
    /// Build over `atoms`, searching `radius` around each query point.
    ///
    /// The radius must satisfy the contract on the struct: large enough to contain every atom
    /// that could put a query point below the threshold of interest, and no larger than needed,
    /// since the cell list's cost grows with its cube. Prefer [`PhiField::for_threshold`].
    pub fn new(
        cell: &UnitCell,
        atoms: &'a [Coord3],
        pc: &PrimitiveCell,
        radius: f64,
    ) -> Result<Self> {
        Self::new_strained(cell, atoms, pc, radius, None)
    }

    /// As [`PhiField::new`], with the roto-strain field that deforms the cells.
    ///
    /// `strain` of `None` gives the rigid field this module has always computed. Passing a field
    /// replaces each atom's ideal cell with its locally-deformed one, which is the difference
    /// between measuring displacement and measuring void: see the module docs and
    /// [`crate::analysis::roto_strain`].
    pub fn new_strained(
        cell: &UnitCell,
        atoms: &'a [Coord3],
        pc: &PrimitiveCell,
        radius: f64,
        strain: Option<RotoStrainField>,
    ) -> Result<Self> {
        // NaN is rejected as well as the non-positive values: a NaN radius would sail
        // through a `> 0` check in some formulations and silently build an empty list,
        // which then reads as void everywhere.
        if radius <= 0.0 || radius.is_nan() {
            return Err(Error::invalid(format!(
                "phi search radius must be positive, got {radius}"
            )));
        }
        let normals = face_normals(pc)?;
        let inflate = strain
            .as_ref()
            .map_or(1.0, |s| 1.0 + s.max_deviation_from_identity());
        let max_epsilon = radius / (circumradius(&normals)? * inflate) - 1.0;
        Ok(Self {
            normals,
            list: CellList::build(cell, atoms, radius)?,
            radius,
            max_epsilon,
            strain,
            scratch: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// Build with the minimal radius that is exact for the threshold `φ > 1 + epsilon`.
    ///
    /// `{ψ_i(x) ≤ c}` is the W–S cell scaled by `c` about `x_i`, and a scaled cell of
    /// circumradius `R` lies within `c·R` of `x_i`. Any atom farther than `R(1+ε)` than the
    /// query point therefore *cannot* hold it below `1+ε`, so leaving those out cannot change
    /// the `φ > 1+ε` decision — only the returned φ value, which is already documented as an
    /// upper bound outside the searched set.
    ///
    /// This is much smaller than the region's own span, and that matters: the radius sets the
    /// cell list's stencil, so a radius of the box size would put every atom of a large sample
    /// in every query and turn the estimator quadratic in the atom count.
    pub fn for_threshold(
        cell: &UnitCell,
        atoms: &'a [Coord3],
        pc: &PrimitiveCell,
        epsilon: f64,
    ) -> Result<Self> {
        Self::for_threshold_strained(cell, atoms, pc, epsilon, None)
    }

    /// As [`PhiField::for_threshold`], with the roto-strain field.
    ///
    /// The radius is inflated by `1 + max|T − I|` because a deformed cell can be *larger* than the
    /// ideal one, and the `R(1+ε)` argument only bounds the ideal cell. Without that, a strained
    /// field silently drops the atoms that hold down the outermost points of the enlarged cells —
    /// the field's `is_void` decision then turns on atoms that were never searched.
    pub fn for_threshold_strained(
        cell: &UnitCell,
        atoms: &'a [Coord3],
        pc: &PrimitiveCell,
        epsilon: f64,
        strain: Option<RotoStrainField>,
    ) -> Result<Self> {
        let normals = face_normals(pc)?;
        let circum = circumradius(&normals)?;
        let inflate = strain
            .as_ref()
            .map_or(1.0, |s| 1.0 + s.max_deviation_from_identity());
        // A little slack above R(1+ε) so that a point just outside the threshold cannot be
        // starved by floating-point disagreement between the two computations of the same
        // distance. One part in 10^6 is far below any physical tolerance and far above the
        // rounding error here.
        Self::new_strained(
            cell,
            atoms,
            pc,
            circum * (1.0 + epsilon) * inflate * (1.0 + 1e-6),
            strain,
        )
    }

    /// The roto-strain field in use, if any.
    pub fn strain(&self) -> Option<&RotoStrainField> {
        self.strain.as_ref()
    }

    /// The search radius in use, i.e. what [`PhiField::new`] was built with.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The largest ε for which [`PhiField::is_void`] decides `φ > 1 + ε` exactly. Beyond it an
    /// atom outside the search radius could hold a point below the threshold, and the point
    /// would read as void: [`estimate_void_volume`] refuses such an ε.
    pub fn max_epsilon(&self) -> f64 {
        self.max_epsilon
    }

    /// ψ_i(x) for one atom: the largest `|n̂·(x − x_i)|` over the face normals.
    ///
    /// This is 1 exactly on that atom's cell boundary and less inside it, so it is `≥ 1`
    /// everywhere outside the cell, growing with distance — which is what makes the minimum
    /// over atoms meaningful.
    pub fn psi(&self, x: Coord3, pj: Coord3) -> f64 {
        self.psi_inner(x, pj, None)
    }

    /// ψ_i(x) with the cell of site `j` deformed by `T_j`, i.e. the roto-strain form.
    ///
    /// Deforming the cell is *not* the same as rotating the normals, and the difference is easy to
    /// get backwards. With `n̂' = T⁻ᵀn̂`,
    ///
    /// ```text
    /// ψ'_i(x) = max_k |(T⁻ᵀn̂_k)·(x − x_i)| = max_k |n̂_k·(T⁻¹(x − x_i))|
    /// ```
    ///
    /// so the transformed object is the *displacement*, and the ideal normal set is reused
    /// untouched. That also makes it cheap: one 3×3 solve per atom instead of a transform per
    /// normal, and the ideal `ws_vertices` stay valid for sampling.
    ///
    /// The check that this is right is that a dilated lattice gives *larger* cells: `T = cI` scales
    /// ψ by `1/c`, so `{ψ ≤ 1}` grows by `c`, which is what thermal expansion must do. Rotating the
    /// normals instead would scale by `c` the wrong way.
    fn psi_inner(&self, x: Coord3, pj: Coord3, t: Option<Mat3>) -> f64 {
        let mut d = sub(x, pj);
        if let Some(t) = t {
            // A singular T is the degenerate case the field should never contain, but silently
            // skipping the solve would turn a broken cell into an infinitely large one.
            match t.solve(d) {
                Ok(v) => d = v,
                Err(_) => return f64::INFINITY,
            }
        }
        self.normals
            .iter()
            .map(|&n| dot(n, d).abs())
            .fold(0.0, f64::max)
    }

    /// `T` at site `j`, or `None` for the rigid field.
    #[inline]
    fn t_of(&self, j: usize) -> Option<Mat3> {
        self.strain.as_ref().map(|s| s.at(self.list.positions()[j]))
    }

    /// φ(x) = min_i ψ_i(x), the smallest "outside-ness" over the atoms in range.
    ///
    /// This is the *quantitative* field, and it is only meaningful where an atom is in range.
    /// Where none is, it returns `f64::INFINITY` — a sentinel, not a number, and deep in a
    /// cavity that is not a rare corner case but the common one. The void *decision* is still
    /// correct there (see [`PhiField::is_void`]); only the value is undefined.
    ///
    /// Use [`PhiField::is_void`] to threshold, not `at(...) > 1.0`: the sentinel would compare
    /// as void by accident rather than by argument, and a field built with too small a radius
    /// would then report its own edge as void instead of failing.
    pub fn at(&self, x: Coord3) -> f64 {
        let mut hits = self.scratch.borrow_mut();
        if self
            .list
            .neighbors_within_point(x, self.radius, &mut hits)
            .is_err()
        {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        for &(j, dr) in hits.iter() {
            // `dr` is the vector from the query point to the atom, so `x − pj = −dr`. The strain
            // acts on that displacement once per atom — not once per normal — which is both the
            // cheap form and the one `psi_inner` documents.
            let mut d = [-dr[0], -dr[1], -dr[2]];
            if let Some(t) = self.t_of(j) {
                match t.solve(d) {
                    Ok(v) => d = v,
                    Err(_) => continue,
                }
            }
            let mut acc = 0.0f64;
            for &n in self.normals.iter() {
                acc = acc.max(dot(n, d).abs());
                if acc >= best {
                    break;
                }
            }
            if acc < best {
                best = acc;
            }
            if best <= 1.0 {
                // No other atom can lower it: ψ ≥ 0 and this one is already inside
                // its own cell.
                break;
            }
        }
        best
    }

    /// Whether `x` lies in a void at the threshold `φ > 1 + epsilon`, in the sense of
    /// *no atom can hold it below the threshold* rather than in the sense of a numeric φ.
    ///
    /// This is the decision the volume estimator actually needs, and it stays well-defined
    /// where [`PhiField::at`] returns its sentinel: a point with no atom within `R(1+ε)` is
    /// *provably* void at that threshold, because `{ψ ≤ 1+ε}` lies within `R(1+ε)` of its
    /// site. Treating that as void is a consequence of the definition, not a failure to find
    /// data — which is what makes the small radius of [`PhiField::for_threshold`] sound.
    pub fn is_void(&self, x: Coord3, epsilon: f64) -> bool {
        let cut = 1.0 + epsilon;
        let mut hits = self.scratch.borrow_mut();
        if self
            .list
            .neighbors_within_point(x, self.radius, &mut hits)
            .is_err()
        {
            return true;
        }
        // Void unless some atom in range holds the point below the threshold. The early exit
        // is the common case: a point inside the crystal is decided by its own site.
        for &(j, dr) in hits.iter() {
            let mut d = [-dr[0], -dr[1], -dr[2]];
            if let Some(t) = self.t_of(j) {
                match t.solve(d) {
                    Ok(v) => d = v,
                    Err(_) => continue,
                }
            }
            let mut acc = 0.0f64;
            for &n in self.normals.iter() {
                acc = acc.max(dot(n, d).abs());
                if acc > cut {
                    break;
                }
            }
            if acc <= cut {
                return false;
            }
        }
        true
    }

    /// The same field sampled at every Wigner–Seitz cell corner of every site in the list.
    ///
    /// These are the only points where φ can be near 1 for
    /// a structural reason: a corner is by construction where several cells meet, so it is where
    /// the crystal's surface is, and where a void opens. O(atoms × corners), so it is a
    /// resolution-independent sample of the surface rather than a grid.
    pub fn at_cell_vertices(&self) -> Vec<(Coord3, f64)> {
        let corners = ws_vertices(&self.normals);
        let mut out = Vec::with_capacity(self.list.positions().len() * corners.len());
        for (j, &site) in self.list.positions().iter().enumerate() {
            let t = self.t_of(j);
            for &v in corners.iter() {
                // Under strain the cell is the affine image of the ideal one, so its corners move
                // to `site + T·v` — the same `T` the query path inverts, so the samples sit on
                // the surface the field is thresholded at. Sampling the *ideal* corners would put
                // every sample a fraction of a cell off it.
                let x = match t {
                    Some(t) => add(site, t.mul_v(v)),
                    None => add(site, v),
                };
                out.push((x, self.at(x)));
            }
        }
        out
    }
}

/// A void-volume estimate from the φ field, with the uncertainty that goes with it.
#[derive(Debug, Clone, Copy)]
pub struct VoidEstimate {
    /// Estimated volume of `{φ > 1}` within the region sampled.
    pub volume: f64,
    /// Standard error of `volume`, from the sample count and the hit rate.
    pub standard_error: f64,
    /// Number of sample points drawn.
    pub samples: usize,
    /// Number of samples that landed in a void.
    pub hits: usize,
    /// Volume of the region sampled.
    pub region_volume: f64,
}

impl VoidEstimate {
    /// The estimate as `1 ± se`, which is how a sampled volume should be quoted: a void
    /// volume from Monte Carlo is not a number to 5 significant figures.
    pub fn summary(&self) -> String {
        if self.samples == 0 {
            return "no samples".to_owned();
        }
        let rel = if self.volume > 0.0 {
            100.0 * self.standard_error / self.volume
        } else {
            0.0
        };
        format!(
            "{:.4} +/- {:.4} A^3 ({:.2}% of region, {} of {} samples)",
            self.volume, self.standard_error, rel, self.hits, self.samples
        )
    }
}

/// Estimate the volume of `{φ > 1 + epsilon}` in a region by seeded Monte Carlo.
///
/// The threshold is `1 + epsilon` rather than `1` because thermal displacement puts a thin
/// layer of every atom's own cell above 1 in any real crystal: measuring at exactly 1 would
/// count that layer as void and report a void volume for a defect-free crystal. Design §17.2
/// extrapolates the measured volumes back to `ε = 0`; this returns the volume at one `epsilon`
/// so that extrapolation can be done by the caller from as many values as it has.
///
/// `seed` fixes the sample set, so a reported volume is reproducible — otherwise the error bar
/// would be the only reproducible part of the answer, and two runs of the same command could
/// not be compared.
pub fn estimate_void_volume(
    field: &PhiField,
    region: &[(f64, f64); 3],
    epsilon: f64,
    samples: usize,
    seed: u64,
) -> Result<VoidEstimate> {
    if samples == 0 {
        return Err(Error::invalid(
            "void volume estimate needs at least one sample",
        ));
    }
    let cut = 1.0 + epsilon;
    if cut <= 1.0 || cut.is_nan() {
        return Err(Error::invalid(format!(
            "epsilon {epsilon} gives a threshold of {cut}, which is not above the \
             Wigner-Seitz boundary phi = 1"
        )));
    }
    // A field searched for a smaller threshold misses atoms that hold points below this one, and
    // counts those points as void: on a W vacancy V(eps) then stopped falling at eps ~ 0.3
    // instead of shrinking as (1 - eps)^3. Build the field for the largest epsilon measured.
    if epsilon > field.max_epsilon() {
        return Err(Error::invalid(format!(
            "epsilon {epsilon} exceeds the {:.6} the phi field's search radius is exact for; \
             build the field for the largest epsilon measured (PhiField::for_threshold)",
            field.max_epsilon()
        )));
    }
    let dv: f64 = (0..3).map(|d| (region[d].1 - region[d].0).abs()).product();
    if dv <= 0.0 || dv.is_nan() {
        return Err(Error::invalid("void volume region has zero extent"));
    }
    let mut rng = Rng(seed);
    let mut hits = 0usize;
    for _ in 0..samples {
        let mut x = [0.0f64; 3];
        for d in 0..3 {
            x[d] = region[d].0 + rng.unit() * (region[d].1 - region[d].0);
        }
        // The decision goes through `is_void` rather than `at(x) > cut`, so that a point with
        // no atom in range counts as void by the argument rather than by an infinity
        // comparison that would also fire for a field built with too small a radius.
        if field.is_void(x, epsilon) {
            hits += 1;
        }
    }
    let p = hits as f64 / samples as f64;
    let volume = p * dv;
    // Binomial standard error, scaled to volume. With no hits the rule-of-three estimate
    // is the honest one: the naive sqrt(0) would claim an exact zero.
    let se = if hits == 0 {
        dv * (3.0 / samples as f64).sqrt()
    } else {
        dv * (p * (1.0 - p) / samples as f64).sqrt()
    };
    Ok(VoidEstimate {
        volume,
        standard_error: se,
        samples,
        hits,
        region_volume: dv,
    })
}

/// A void volume extrapolated from several thresholds back to `φ = 1`.
///
/// Design §17.2 measures at `φ = 1+ε` and extrapolates to `φ = 1`, because at `ε = 0` the
/// thermal shell around every atom contributes and a defect-free crystal would not measure zero.
///
/// # The extrapolation is quadratic, and it has to be
///
/// The obvious prescription is to "fit a line through ≥2 ε values". Measured on the relaxed
/// 256k-atom HEA fixture, that does not work: `V(ε)` falls 1283 → 768 → 112 → 0 Å³ at
/// ε = 0.01, 0.02, 0.05, 0.10, so the implied slope changes by a factor of 20 across the
/// range. A line fitted through the two smallest values extrapolates to **1798 Å³ of void in a
/// crystal that has none** — the thermal shell is strongly curved near φ = 1, and a straight
/// line through it amplifies the curvature into the answer.
///
/// The shape is quadratic because it is a surface integral: `{φ > 1+ε}` erodes a shell of
/// roughly uniform thickness off the void boundary as ε rises, so `V` falls like `(1+ε)³`
/// locally, which is quadratic to second order in ε about 0. Fitting `V = a + bε + cε²` and
/// taking the constant term therefore extrapolates a *curve* with a curve, and on the fixture
/// above it returns a small negative number rather than 1798 Å³.
///
/// Three or more distinct `epsilons` are required for a quadratic. With two there is no
/// curvature to fit and the answer silently degenerates to the linear prescription, with the
/// overstatement above.
#[derive(Debug, Clone)]
pub struct ExtrapolatedVoid {
    /// The individual `V(ε)` measurements the fit was made through.
    pub measurements: Vec<VoidEstimate>,
    /// The extrapolated volume at `ε = 0`, i.e. at `φ = 1`.
    pub volume_at_one: f64,
    /// Standard error of `volume_at_one`, propagated from the measurements.
    pub standard_error: f64,
    /// The fitted constant, linear and quadratic coefficients `(a, b, c)` of
    /// `V(ε) = a + bε + cε²`. Exposed because `b` and `c` are the diagnostics: a large `c`
    /// means the measurements do not span the curvature and the extrapolation is being asked
    /// for more than the data supports.
    pub coefficients: (f64, f64, f64),
    /// RMS residual of the fit in Å³. Large relative to `volume_at_one` means the quadratic
    /// does not describe these measurements, so the extrapolated value should not be trusted
    /// even though a number came back.
    pub residual: f64,
}

impl ExtrapolatedVoid {
    /// The estimate as `1 ± se` in Å³.
    pub fn summary(&self) -> String {
        format!(
            "{:.4} +/- {:.4} A^3 (quadratic extrapolation to phi = 1 from {} thresholds,              residual {:.4} A^3)",
            self.volume_at_one,
            self.standard_error,
            self.measurements.len(),
            self.residual
        )
    }

    /// Whether the fitted curve is falling in ε, as a void region's volume must be.
    ///
    /// A rising or flat `V(ε)` means the measurements are not a void eroding away, and the
    /// extrapolation is then fitting noise: worth a named predicate rather than a bare
    /// comparison in a test, because it is a physical requirement and not a numerical one.
    pub fn slope_negative(&self) -> bool {
        self.coefficients.1 < 0.0
    }

    /// Whether the fit is good enough for the extrapolated value to mean anything.
    ///
    /// The check that a residual is reported without this would be missing: a fit can produce
    /// a confident-looking number and a large residual at the same time, and the number is the
    /// one a caller will print.
    pub fn is_trustworthy(&self) -> bool {
        // A residual above a tenth of the answer, or a negative answer, both mean the
        // measurements do not describe a smooth curve that flattens at phi = 1.
        self.residual < 0.1 * self.volume_at_one.abs() && self.volume_at_one >= 0.0
    }
}

/// Measure at several `ε` and extrapolate the volume back to `φ = 1` with a quadratic fit.
///
/// Fitting the curve rather than reporting one threshold is what makes the number comparable
/// with method A: `V(ε)` is the `ε`-eroded void, so a single measurement is biased by however
/// much the crystal is thermally broadened, and that bias is not a constant offset — it is a
/// curvature that a straight line cannot follow (see [`ExtrapolatedVoid`]).
///
/// # Errors
///
/// At least three distinct positive `epsilons` are required. Fewer cannot support a quadratic
/// and would silently reduce this to the linear extrapolation that overstates a defect-free
/// crystal by two orders of magnitude. At least three *positive* ones are required too: `ε = 0`
/// is the point being extrapolated *to*, and including it would make the fit trivially agree
/// with itself while measuring nothing.
///
/// The error bar is propagated from each measurement's own standard error through the
/// least-squares weights. For a fit that extrapolates, those weights exceed 1, and that
/// amplification is real: it is the fit admitting it is reaching outside the measured range.
pub fn extrapolate_to_phi_one(
    field: &PhiField,
    region: &[(f64, f64); 3],
    epsilons: &[f64],
    samples: usize,
    seed: u64,
) -> Result<ExtrapolatedVoid> {
    if epsilons.len() < 3 {
        return Err(Error::invalid(format!(
            "the quadratic extrapolation to phi = 1 needs at least 3 epsilons, got {}; with \
             fewer it degenerates to the straight-line fit, which overstates a defect-free \
             crystal by orders of magnitude",
            epsilons.len()
        )));
    }
    // Distinct values, not just a count: three copies of one epsilon give three rows of the
    // same design matrix, so the fit is underdetermined in a way that a singular-matrix check
    // may or may not catch depending on rounding. Rejecting duplicates states the actual
    // requirement rather than hoping the arithmetic notices.
    for (i, &e) in epsilons.iter().enumerate() {
        if epsilons[..i].contains(&e) {
            return Err(Error::invalid(format!(
                "epsilons {epsilons:?} repeat {e}: the quadratic fit needs distinct values, \
                 or it has fewer independent rows than coefficients"
            )));
        }
    }
    if epsilons.iter().any(|&e| e <= 0.0 || e.is_nan()) {
        return Err(Error::invalid(format!(
            "epsilons {epsilons:?} must all be positive: epsilon = 0 is the point being \
             extrapolated to, and measuring there instead of extrapolating would defeat the \
             correction"
        )));
    }
    let mut measurements = Vec::with_capacity(epsilons.len());
    for &e in epsilons {
        measurements.push(estimate_void_volume(field, region, e, samples, seed)?);
    }

    // Least squares on the Vandermonde [1, ε, ε²], accumulated as 3×3 normal equations so
    // the fit needs no matrix dependency beyond the one the crate already has.
    let coef = quadratic_coefficients(
        epsilons,
        &measurements.iter().map(|m| m.volume).collect::<Vec<_>>(),
    )?;

    let n = epsilons.len() as f64;
    // Residuals: how far the fitted curve is from each measurement it was fitted to. A large
    // RMS means a quadratic is not the right shape for this data, whatever it extrapolated to.
    let mut ss = 0.0;
    for (&e, m) in epsilons.iter().zip(measurements.iter()) {
        let fit = coef[0] + coef[1] * e + coef[2] * e * e;
        ss += (m.volume - fit).powi(2);
    }
    let residual = (ss / n).sqrt();

    // The constant term's standard error, from the measurement errors through the inverse
    // normal matrix's first row. Built from the same 3x3 solve path rather than a general
    // inverter: the weights for the constant term come from solving the transposed system.
    let weights = variance_weights(epsilons);
    let var: f64 = weights
        .iter()
        .zip(measurements.iter())
        .map(|(w, m)| (w * m.standard_error).powi(2))
        .sum();

    Ok(ExtrapolatedVoid {
        measurements,
        volume_at_one: coef[0],
        standard_error: var.sqrt(),
        coefficients: (coef[0], coef[1], coef[2]),
        residual,
    })
}

/// A void volume measured as the *excess* over an ideal reference lattice.
///
/// This is the only way the rigid-lattice field measures something defect-like on a relaxed
/// crystal, and it is a subtraction rather than a correction. `{φ > 1+ε}` on a real sample is
/// the union of genuine voids and the thermal shell around every displaced atom; the same
/// threshold on the corresponding *ideal* lattice is the shell's absence, so
///
/// ```text
/// excess(ε) = V_sample(ε) − V_ideal(ε)
/// ```
///
/// cancels the shell instead of extrapolating it away. On the relaxed HEA fixture the sample
/// measures 1339.7 Å³ at ε = 0.01 while the ideal lattice measures 0, so the excess is the full
/// shell — which is the honest answer, and a reminder that the subtraction removes only what is
/// common to both. It becomes a void volume only when the two lattices are the same shape, i.e.
/// when the sample's atoms are close to where the reference puts them.
#[derive(Debug, Clone)]
pub struct ExcessVoid {
    /// Threshold-by-threshold excess volumes, with their error bars.
    pub measurements: Vec<VoidEstimate>,
    /// Excess volume at `ε = 0` (three or more thresholds), or the single measurement itself
    /// when one threshold was given.
    ///
    /// Read this as a void volume **only** if the reference is the sample's own locally-relaxed
    /// structure. See [`excess_void_volume`] for why an ideal reference does not qualify.
    pub volume_at_one: f64,
    /// Standard error of `volume_at_one`, from the two independent sample counts.
    pub standard_error: f64,
    /// ε values the measurements were taken at, in order.
    pub epsilons: Vec<f64>,
    /// Fitted `(a, b, c)` of the excess curve `V = a + bε + cε²`.
    pub coefficients: (f64, f64, f64),
}

impl ExcessVoid {
    /// The estimate as `1 ± se` in Å³, labelled as an excess rather than a void volume.
    ///
    /// The wording is deliberate: this number is only a void volume when the reference shares the
    /// sample's local relaxation, and a summary string is what usually ends up in output nobody
    /// re-reads the API docs for.
    pub fn summary(&self) -> String {
        let how = match self.epsilons.as_slice() {
            [e] => format!("at epsilon {e:.4}, not extrapolated"),
            eps => format!("extrapolated to phi = 1 from {} thresholds", eps.len()),
        };
        format!(
            "{:.4} +/- {:.4} A^3 (excess over the reference {how}; a void volume only if the \
             reference is locally relaxed)",
            self.volume_at_one, self.standard_error,
        )
    }
}

/// Measure the excess of `sample` over `reference` at the same thresholds.
///
/// Both fields are evaluated on the *same* sample points, so the difference is paired: the
/// thermal shell is present in both and cancels point by point rather than in expectation,
/// which removes most of the common-mode noise. The two error bars are then independent (each
/// point's verdicts are separate draws) and add in quadrature.
///
/// # What the result means depends on the reference, not on this code
///
/// The subtraction removes exactly what the two configurations have *in common*. It becomes a
/// void volume when the reference is the sample's own locally-relaxed structure — the case
/// `void_b_reference` pins with a frame measured against itself, where every threshold cancels to
/// zero. It becomes a displacement diagnostic when the reference is an ideal lattice, because then
/// the two share no displacement at all: on the relaxed HEA fixture the sample reads 1339.7 Å³ at
/// ε = 0.01 while the ideal lattice reads 0, so the "excess" is the whole shell.
///
/// The second case is not a defect in the routine, it is the physics: without a roto-strain
/// field there is no configuration that isolates void from displacement, and picking the ideal
/// reference is a choice to measure displacement. Callers wanting a void volume need `T(x)`; callers
/// wanting a displacement diagnostic want the ideal reference. Nothing here can tell them apart,
/// which is why the API doc and [`ExcessVoid::summary`] both avoid the word "void".
///
/// A negative excess is possible and is not clamped: it means the sample happens to fill its cells
/// more completely than the reference, and clamping to zero would hide that.
pub fn excess_void_volume(
    sample: &PhiField,
    reference: &PhiField,
    region: &[(f64, f64); 3],
    epsilons: &[f64],
    samples: usize,
    seed: u64,
) -> Result<ExcessVoid> {
    if epsilons.is_empty() {
        return Err(Error::invalid(
            "excess void volume needs at least one epsilon",
        ));
    }
    // One threshold is a measurement, three or more an extrapolation. Two would be the
    // straight-line fit `extrapolate_to_phi_one` refuses; anything below three cannot feed the
    // quadratic, whose 3x3 normal matrix is then rank-deficient but not exactly singular in
    // floating point, so it "solved" to a headline unrelated to the row it was fitted to
    // (3603.5 A^3 against a measured 1393.8 A^3 on a 89-site W void).
    if epsilons.len() == 2 {
        return Err(Error::invalid(format!(
            "excess void volume needs one epsilon (a single measurement) or at least 3 (a \
             quadratic extrapolation to phi = 1), got {epsilons:?}"
        )));
    }
    let mut measurements = Vec::with_capacity(epsilons.len());
    for &e in epsilons {
        let a = estimate_void_volume(sample, region, e, samples, seed)?;
        let b = estimate_void_volume(reference, region, e, samples, seed)?;
        measurements.push(VoidEstimate {
            volume: a.volume - b.volume,
            // Paired difference of two independent binomials: variances add, they do not cancel,
            // because the sample points that differ between the two fields are exactly the
            // ones carrying the signal.
            standard_error: (a.standard_error.powi(2) + b.standard_error.powi(2)).sqrt(),
            samples: a.samples,
            hits: a.hits.saturating_sub(b.hits),
            region_volume: a.region_volume,
        });
    }
    if measurements.len() == 1 {
        // Not extrapolated: the headline is the measurement itself.
        let m = &measurements[0];
        return Ok(ExcessVoid {
            volume_at_one: m.volume,
            standard_error: m.standard_error,
            epsilons: epsilons.to_vec(),
            coefficients: (m.volume, 0.0, 0.0),
            measurements,
        });
    }
    // Fit the *excess* curve, so the shell is removed before the extrapolation rather than
    // amplified by it. This is the whole reason to subtract rather than to extrapolate: the
    // shell is a large common term whose slope near φ = 1 is what wrecked the raw fit.
    let coef = quadratic_coefficients(
        epsilons,
        &measurements.iter().map(|m| m.volume).collect::<Vec<_>>(),
    )?;
    let weights = variance_weights(epsilons);
    let var: f64 = weights
        .iter()
        .zip(measurements.iter())
        .map(|(w, m)| (w * m.standard_error).powi(2))
        .sum();
    Ok(ExcessVoid {
        measurements,
        volume_at_one: coef[0],
        standard_error: var.sqrt(),
        epsilons: epsilons.to_vec(),
        coefficients: (coef[0], coef[1], coef[2]),
    })
}

/// Width of the shell-free window in ε (design §17.2).
pub const WINDOW_WIDTH: f64 = 0.3;
/// Thresholds measured across the window, evenly spaced.
pub const WINDOW_POINTS: usize = 7;
/// The window never starts below this ε, so a perfect frame is still measured off `φ = 1`.
pub const WINDOW_FLOOR: f64 = 0.01;

/// The largest *face gap* of a frame (design §17.2): the thermal shell's reach in ε.
///
/// Two atoms whose separation `r` matches the face vector `v_k` of the ideal cell (`n̂_k =
/// 2v_k/|v_k|²`) share the face `n̂_k·(x − x_i) = 1`. Displaced, their cells part along it,
/// and the gap between them is void up to `φ = n̂_k·r / 2 = v_k·r / |v_k|²`. The face gap of
/// the pair is that minus one, and the shell of the whole frame is closed at every threshold
/// above the largest one. Measured, not modelled: no amplitude, distribution or temperature
/// enters. Bonds are matched to the nearest face vector within a quarter of the shortest; a
/// pair matching none (beyond the face shells, or across a void) does not count.
#[derive(Debug, Clone, Copy)]
pub struct FaceGap {
    /// The largest gap, ≥ 0 (pairs pressed together give negative gaps, which close nothing).
    pub max: f64,
    /// Matched pairs, each counted from both ends.
    pub bonds: usize,
}

/// [`FaceGap`] of `atoms` in `cell` against the face vectors of `pc`.
pub fn max_face_gap(cell: &UnitCell, atoms: &[Coord3], pc: &PrimitiveCell) -> Result<FaceGap> {
    let vecs: Vec<Coord3> = face_normals(pc)?
        .iter()
        .map(|&n| scale(n, 2.0 / dot(n, n)))
        .collect();
    let shortest = vecs
        .iter()
        .map(|&v| dot(v, v).sqrt())
        .fold(f64::INFINITY, f64::min);
    let longest = vecs.iter().map(|&v| dot(v, v).sqrt()).fold(0.0, f64::max);
    let tol2 = (0.25 * shortest).powi(2);
    let reach = longest + 0.25 * shortest;
    let list = CellList::build(cell, atoms, reach)?;
    let mut nb = Vec::new();
    let mut max = 0.0f64;
    let mut bonds = 0usize;
    for i in 0..atoms.len() {
        list.neighbor_entries(i, reach, &mut nb)?;
        for &(_, dr, _) in nb.iter() {
            let (best, d2) =
                vecs.iter()
                    .map(|&v| (v, dist2(v, dr)))
                    .fold(
                        ([0.0; 3], f64::INFINITY),
                        |a, b| if b.1 < a.1 { b } else { a },
                    );
            if d2 <= tol2 {
                bonds += 1;
                max = max.max(dot(best, dr) / dot(best, best) - 1.0);
            }
        }
    }
    Ok(FaceGap { max, bonds })
}

/// The thresholds of the shell-free window above a face gap: [`WINDOW_POINTS`] evenly over
/// `[lo, lo + WINDOW_WIDTH]`, `lo = max(gap, WINDOW_FLOOR)`.
///
/// # Errors
///
/// When the window would reach `ε = 0.9`: a single vacancy's cell is gone at `ε = 1` (it
/// shrinks as `(1 − ε)³`), so a frame whose atoms part that far has no window in which a void
/// is both shell-free and still there.
pub fn window_epsilons(gap: f64) -> Result<Vec<f64>> {
    let lo = gap.max(WINDOW_FLOOR);
    if !lo.is_finite() || lo + WINDOW_WIDTH > 0.9 {
        return Err(Error::invalid(format!(
            "the largest face gap is {gap:.4}: the thermal shell reaches beyond epsilon {:.2}, \
             where small voids have shrunk away, so there is no shell-free window to measure in \
             (atoms far off their lattice: a disordered region, a wrong lattice or nn distance)",
            0.9 - WINDOW_WIDTH
        )));
    }
    Ok((0..WINDOW_POINTS)
        .map(|i| lo + WINDOW_WIDTH * i as f64 / (WINDOW_POINTS - 1) as f64)
        .collect())
}

/// A void volume extrapolated to `φ = 1` from the shell-free window (design §17.2).
#[derive(Debug, Clone)]
pub struct WindowVoid {
    /// The frame's largest face gap, where the window starts (at least [`WINDOW_FLOOR`]).
    pub face_gap: FaceGap,
    /// The thresholds measured.
    pub epsilons: Vec<f64>,
    /// `V(ε)` at each.
    pub measurements: Vec<VoidEstimate>,
    /// `α³`, the extrapolation of `V^(1/3) = α + βε` to `ε = 0`.
    pub volume_at_one: f64,
    /// Propagated from the measurements' standard errors.
    pub standard_error: f64,
    /// `(α, β)` in Å.
    pub coefficients: (f64, f64),
    /// RMS residual of the fit, in Å of `V^(1/3)`.
    pub residual: f64,
}

impl WindowVoid {
    /// The estimate as `1 ± se` in Å³.
    pub fn summary(&self) -> String {
        format!(
            "{:.4} +/- {:.4} A^3 (V^(1/3) linear in epsilon over the shell-free window \
             [{:.4}, {:.4}], {} thresholds, residual {:.4} A)",
            self.volume_at_one,
            self.standard_error,
            self.epsilons[0],
            self.epsilons[self.epsilons.len() - 1],
            self.epsilons.len(),
            self.residual
        )
    }
}

/// Measure `V(ε)` across the shell-free window above `face_gap` and extrapolate it to `φ = 1`
/// through `V^(1/3) = α + βε` (design §17.2).
///
/// Below the largest face gap `V(ε)` holds thermal shell; above it, only void (and whatever
/// part of a cavity's own boundary the threshold erodes). The erosion moves each face of the
/// void inward by `ε` times its distance from the site behind it, so a vacancy's cell shrinks
/// homothetically, `V = V₀(1 − ε)³`, and a compact cluster nearly so: `V^(1/3)` is linear in
/// ε, exactly for one vacancy. The field must be searched for the window's top
/// ([`PhiField::max_epsilon`]).
pub fn extrapolate_shell_free(
    field: &PhiField,
    region: &[(f64, f64); 3],
    face_gap: FaceGap,
    samples: usize,
    seed: u64,
) -> Result<WindowVoid> {
    let epsilons = window_epsilons(face_gap.max)?;
    let mut measurements = Vec::with_capacity(epsilons.len());
    for &e in &epsilons {
        measurements.push(estimate_void_volume(field, region, e, samples, seed)?);
    }
    let y: Vec<f64> = measurements
        .iter()
        .map(|m| m.volume.max(0.0).cbrt())
        .collect();
    let n = epsilons.len() as f64;
    let mx = epsilons.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxx: f64 = epsilons.iter().map(|e| (e - mx).powi(2)).sum();
    let beta = epsilons
        .iter()
        .zip(&y)
        .map(|(e, y)| (e - mx) * (y - my))
        .sum::<f64>()
        / sxx;
    let alpha = my - beta * mx;
    let residual = (epsilons
        .iter()
        .zip(&y)
        .map(|(e, y)| (y - alpha - beta * e).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    // se(α) from the intercept's weights `1/n − x̄(xᵢ − x̄)/Sxx`, each y's error from its V's
    // by d(V^(1/3))/dV = 1/(3 V^(2/3)), with V floored at its own error bar so an empty
    // measurement does not divide by zero.
    let var_alpha: f64 = epsilons
        .iter()
        .zip(&measurements)
        .map(|(e, m)| {
            let w = 1.0 / n - mx * (e - mx) / sxx;
            let v = m.volume.max(m.standard_error);
            let se_y = m.standard_error / (3.0 * v.powf(2.0 / 3.0));
            (w * se_y).powi(2)
        })
        .sum();
    let volume_at_one = alpha.powi(3);
    Ok(WindowVoid {
        face_gap,
        epsilons,
        measurements,
        volume_at_one,
        standard_error: 3.0 * alpha * alpha * var_alpha.sqrt(),
        coefficients: (alpha, beta),
        residual,
    })
}

/// The least-squares weights of the constant term of the quadratic fit, i.e. the first row of
/// `(XᵀX)⁻¹Xᵀ`.
///
/// Computed by solving `XᵀX w = Xᵀ 1`, which is the same 3×3 solve the fit uses, so the error
/// propagation cannot drift away from the fit it is describing.
fn variance_weights(epsilons: &[f64]) -> Vec<f64> {
    // The same normal matrix the fit used; only the right-hand side differs, being the design
    // matrix applied to a vector of ones — i.e. the column sums of Xᵀ. Sharing the matrix
    // means the error propagation cannot drift away from the fit it describes.
    let (mat, _) = normal_equations(epsilons);
    let n = epsilons.len() as f64;
    let rhs = [
        n,
        epsilons.iter().sum::<f64>(),
        epsilons.iter().map(|&e| e * e).sum::<f64>(),
    ];
    mat.solve(rhs).unwrap_or([0.0; 3]).to_vec()
}

/// Least-squares quadratic coefficients `(a, b, c)` of `V = a + bε + cε²` at the given
/// `(ε, V)` pairs.
///
/// Shared by [`extrapolate_to_phi_one`] and [`excess_void_volume`] so the raw and the
/// reference-subtracted estimate cannot drift apart in how they fit, only in what they fit.
fn quadratic_coefficients(epsilons: &[f64], values: &[f64]) -> Result<Coord3> {
    if epsilons.len() != values.len() {
        return Err(Error::invalid("quadratic fit needs one value per epsilon"));
    }
    let (mat, _) = normal_equations(epsilons);
    let mut rhs = [0.0f64; 3];
    for (&e, &v) in epsilons.iter().zip(values.iter()) {
        rhs[0] += v;
        rhs[1] += e * v;
        rhs[2] += e * e * v;
    }
    mat.solve([rhs[0], rhs[1], rhs[2]])
        .map_err(|_| Error::invalid("the epsilons are degenerate: the quadratic fit is singular"))
}

/// Accumulate the 3×3 normal equations of the quadratic basis `[1, ε, ε²]` at `epsilons`.
///
/// Returns the matrix and a right-hand side of `Σ εᵏ`. The fit substitutes the measured
/// volumes in place of that vector and [`variance_weights`] substitutes ones, so both use
/// one implementation and cannot disagree about the design matrix.
fn normal_equations(epsilons: &[f64]) -> (Mat3, Coord3) {
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for &e in epsilons {
        let x = [1.0, e, e * e];
        for (i, &xi) in x.iter().enumerate() {
            rhs[i] += xi;
            for (j, &xj) in x.iter().enumerate() {
                m[i][j] += xi * xj;
            }
        }
    }
    (Mat3::from_rows(m[0], m[1], m[2]), [rhs[0], rhs[1], rhs[2]])
}

/// SplitMix64: a small, well-distributed, fully specified generator.
///
/// The crate has no RNG dependency and this needs to be reproducible forever — a void volume
/// whose error bar came from an unversioned generator is not a regression test — so the
/// generator is pinned here rather than inherited.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, from the top 53 bits so the result is exactly representable.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

fn dot(a: Coord3, b: Coord3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist2(a: Coord3, b: Coord3) -> f64 {
    dot(sub(a, b), sub(a, b))
}

fn add(a: Coord3, b: Coord3) -> Coord3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Coord3, b: Coord3) -> Coord3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Coord3, s: f64) -> Coord3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::sc::Sc;
    use crate::lattice::Lattice;

    /// Every site of a perfect block `cells` cells per axis, on a cubic cell of edge
    /// `cells * a`, so the sites tile the box exactly and the block has no surface of its own
    /// to be mistaken for void.
    fn block(pc: &PrimitiveCell, cells: usize, a: f64) -> (UnitCell, Vec<Coord3>) {
        let cell = cell_of(a, cells);
        let mut sites = Vec::new();
        for i in 0..cells {
            for j in 0..cells {
                for k in 0..cells {
                    for &b in pc.basis.iter() {
                        sites.push(pc.h.mul_v([i as f64 + b[0], j as f64 + b[1], k as f64 + b[2]]));
                    }
                }
            }
        }
        (cell, sites)
    }

    fn cell_of(a: f64, cells: usize) -> UnitCell {
        UnitCell::orthorhombic(cells as f64 * a, cells as f64 * a, cells as f64 * a)
    }

    fn fcc_pc(a: f64) -> PrimitiveCell {
        Fcc { a }.primitive()
    }

    fn sc_pc(a: f64) -> PrimitiveCell {
        Sc { a }.primitive()
    }

    #[test]
    fn the_face_normals_are_the_expected_counts_per_structure() {
        // A wrong count is a wrong cell, and a wrong cell still produces a plausible field,
        // so these are the counts rather than a check on the field.
        assert_eq!(face_normals(&fcc_pc(3.5)).unwrap().len(), 12, "fcc");
        assert_eq!(face_normals(&sc_pc(3.5)).unwrap().len(), 6, "sc");
        // BCC's cell is the truncated octahedron: 8 first-shell faces and 6 second-shell.
        let bcc = crate::lattice::bcc::Bcc { a: 3.165 }.primitive();
        assert_eq!(face_normals(&bcc).unwrap().len(), 14, "bcc");
    }

    #[test]
    fn each_face_plane_bisects_its_bond() {
        // The defining property of n̂ = 2v̂/|v̂|²: the plane it names, n̂·x = 1, passes
        // through v̂/2 — the perpendicular bisector of the bond it was built from. If this
        // is off by a factor the whole field is rescaled, and then φ = 1 stops being the
        // cell boundary while every other test still sees a plausible field.
        for (pc, bond) in [(fcc_pc(3.5), 3.5 / 2f64.sqrt()), (sc_pc(3.5), 3.5)] {
            let normals = face_normals(&pc).unwrap();
            for &n in normals.iter() {
                // Recover the lattice vector the normal came from: v = 2n/|n|².
                let v = scale(n, 2.0 / dot(n, n));
                assert!(
                    (dot(v, v).sqrt() - bond).abs() < 1e-9,
                    "recovered bond {} against {bond}",
                    dot(v, v).sqrt()
                );
                assert!(
                    (dot(n, scale(v, 0.5)) - 1.0).abs() < 1e-9,
                    "plane does not bisect the bond: n.v/2 = {}",
                    dot(n, scale(v, 0.5))
                );
            }
        }
    }

    #[test]
    fn the_wigner_seitz_cell_has_the_right_shape() {
        // FCC's Wigner-Seitz cell is a rhombic dodecahedron: 14 corners. SC's is a cube: 8.
        // These are exact combinatorial facts, so they catch a vertex search that is missing
        // corners — which would sample the surface sparsely and undercount void volume.
        assert_eq!(
            ws_vertices(&face_normals(&fcc_pc(3.5)).unwrap()).len(),
            14,
            "fcc"
        );
        assert_eq!(
            ws_vertices(&face_normals(&sc_pc(3.5)).unwrap()).len(),
            8,
            "sc"
        );
    }

    #[test]
    fn the_circumradius_matches_the_largest_corner() {
        let normals = face_normals(&fcc_pc(3.5)).unwrap();
        let r = circumradius(&normals).unwrap();
        let want = ws_vertices(&normals)
            .iter()
            .map(|v| dot(*v, *v).sqrt())
            .fold(0.0, f64::max);
        assert!((r - want).abs() < 1e-12, "{r} against {want}");
        // FCC: the largest corner is at a/2 along an axis, so the circumradius is a/2.
        assert!(
            (r - 1.75).abs() < 1e-9,
            "circumradius {r} against a/2 = 1.75"
        );
    }

    #[test]
    fn phi_is_one_on_every_cell_vertex_of_a_perfect_lattice() {
        // The identity that makes this method checkable: φ = 1 exactly where the Wigner-Seitz
        // cells meet. If φ is systematically above or below 1 at a corner, the void volume is
        // biased in a direction that no other test here would catch.
        for pc in [fcc_pc(3.5), sc_pc(3.5)] {
            let normals = face_normals(&pc).unwrap();
            let corners = ws_vertices(&normals);
            // One site only: with a whole block of them, a corner of site A is interior to
            // site B's cell in a periodic lattice, and the test would be about the block
            // rather than about φ's definition. Check a single site against itself.
            let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0);
            let atoms = vec![[30.0, 30.0, 30.0]];
            let field = PhiField::new(&cell, &atoms, &pc, 6.0).unwrap();
            for &v in corners.iter() {
                let x = add([30.0, 30.0, 30.0], v);
                let phi = field.at(x);
                assert!(
                    (phi - 1.0).abs() < 1e-9,
                    "phi {phi} at corner {v:?} of {}",
                    pc.basis.len()
                );
            }
            // The same claim through the sampling entry point, so it is not a helper that only
            // works when a test drives it one corner at a time.
            let sampled = field.at_cell_vertices();
            assert_eq!(
                sampled.len(),
                corners.len(),
                "one site, so one cell of corners"
            );
            for (x, phi) in sampled {
                assert!((phi - 1.0).abs() < 1e-9, "phi {phi} at sampled {x:?}");
            }
        }
    }

    #[test]
    fn phi_is_below_one_inside_a_cell_and_above_one_outside() {
        let pc = fcc_pc(3.5);
        let cell = UnitCell::orthorhombic(60.0, 60.0, 60.0);
        let atoms = vec![[30.0, 30.0, 30.0]];
        let field = PhiField::new(&cell, &atoms, &pc, 8.0).unwrap();
        let c = [30.0, 30.0, 30.0];
        // Dead centre is well inside the cell.
        assert!(field.at(c) < 0.2, "centre {}", field.at(c));
        // Just outside the largest corner is outside every cell.
        let r = circumradius(&face_normals(&pc).unwrap()).unwrap();
        let outside = add(c, [r * 1.2, 0.0, 0.0]);
        assert!(field.at(outside) > 1.0, "outside {}", field.at(outside));
    }

    #[test]
    fn a_perfect_crystal_has_no_void_volume() {
        // The failure this guards: thermal displacement lifts φ above 1 in a thin shell
        // around every atom, so measuring at exactly 1 would report a void in a perfect
        // crystal. Measuring at 1 + ε with a sane ε must return nothing.
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        assert_eq!(sites.len(), 4 * cells * cells * cells);
        let region = [
            (1.0, cells as f64 * a - 1.0),
            (1.0, cells as f64 * a - 1.0),
            (1.0, cells as f64 * a - 1.0),
        ];
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();
        let est = estimate_void_volume(&field, &region, 0.05, 4000, 12345).unwrap();
        // Exact sites: the only points above 1 are the cells' shared corners, which have
        // measure zero. A handful of samples may land near one, so allow a hair.
        assert!(
            est.volume < 0.01 * est.region_volume,
            "perfect crystal reported {} of {} void",
            est.volume,
            est.region_volume
        );
    }

    #[test]
    fn a_vacancy_leaves_exactly_its_own_cell_empty() {
        // The sharpest ground truth available, and it is exact rather than approximate. With
        // every atom on its ideal site the Wigner-Seitz cells tile space, so a point is in
        // some cell unless it is in the missing one: the void region is *exactly* the
        // vacancy's cell. That cell has volume det(h)/n_basis = a³/4 for FCC, the same
        // per-site volume method A charges a void — so method B measures the same quantity
        // continuously and has to land on the same number.
        //
        // This asserts the region's extent pointwise instead of integrating it, which is both
        // exact and free: φ above 1 at the cell's centre and at every point inside it, back to
        // 1 on its corners, and below 1 just outside any corner.
        let a = 3.5;
        let cells = 6;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let victim = sites[3 * sites.len() / 4];
        let kept: Vec<Coord3> = sites.iter().copied().filter(|&s| s != victim).collect();
        let field = PhiField::for_threshold(&cell, &kept, &pc, 0.05).unwrap();
        assert!(
            field.at(victim) > 1.0,
            "a vacancy's own site must read as void, got {}",
            field.at(victim)
        );
        // Halfway to each corner is still inside the cell, so still void.
        let corners = ws_vertices(&face_normals(&pc).unwrap());
        assert_eq!(corners.len(), 14, "fcc cell corners");
        for &v in corners.iter() {
            let inside = add(victim, scale(v, 0.5));
            assert!(field.at(inside) > 1.0, "{inside:?} should be void");
            let on = add(victim, v);
            assert!(
                (field.at(on) - 1.0).abs() < 1e-9,
                "cell corner {v:?} should sit on phi = 1, got {}",
                field.at(on)
            );
        }
        // The cell's faces are exactly as sharp as its corners: φ = 1 on the midpoint of
        // every face, i.e. half a bond out along each of the 12 nearest-neighbour vectors,
        // and back below 1 at those neighbours themselves. That last part is what makes the
        // void bounded at one cell instead of a ball that keeps growing.
        //
        // The cells *do* tile. The W–S cell is the Voronoi cell, so it tiles space exactly and
        // φ ≤ 1 holds everywhere in a perfect lattice — checked numerically rather than
        // assumed, since the tetrahedral and octahedral interstices of FCC are real and it is
        // tempting to mistake them for gaps between cells. They are not: an octahedral hole
        // sits deep inside a neighbour's cell, so φ is small there. What the tiling buys is
        // that the void is bounded, which the neighbour assertion below is what actually needs.
        for &n in face_normals(&pc).unwrap().iter() {
            let v = scale(n, 2.0 / dot(n, n));
            let on_face = add(victim, scale(v, 0.5));
            assert!(
                (field.at(on_face) - 1.0).abs() < 1e-9,
                "face midpoint {on_face:?} should sit on phi = 1, got {}",
                field.at(on_face)
            );
            let neighbour = add(victim, v);
            assert!(
                field.at(neighbour) < 1.0,
                "neighbouring site {neighbour:?} should be crystal, got {}",
                field.at(neighbour)
            );
        }
        // And the count that goes with it, to pin the volume the assertion above implies.
        assert!(
            (a * a * a / 4.0 - 10.718_75).abs() < 1e-9,
            "fcc cell volume"
        );
    }

    #[test]
    fn void_volume_grows_with_the_number_of_removed_sites() {
        // Carving more atoms must carve more void, and never more than the removed sites' own
        // cells hold, because the surviving atoms' cells reach back into the cavity. That
        // upper bound is exact and is the ceiling method A would also respect; between them
        // the two methods have to agree on the size of a defect.
        let a = 3.5;
        let cells = 5;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let per_cell = a * a * a / 4.0;
        let centre = [
            cells as f64 * a / 2.0,
            cells as f64 * a / 2.0,
            cells as f64 * a / 2.0,
        ];
        let mut prev = 0.0;
        // All three radii must clear the nearest-neighbour shell at a/√2, or the first one
        // removes nothing and the growth check below passes on two empty voids.
        for radius in [2.2f64, 3.2, 4.2] {
            let kept: Vec<Coord3> = sites
                .iter()
                .copied()
                .filter(|&s| dist2(s, centre) > radius * radius)
                .collect();
            let removed = sites.len() - kept.len();
            assert!(removed > 0, "radius {radius} carved no atoms out at all");
            // Only the void has to be inside the region: it stops about one cell
            // circumradius past the carved sphere, so a reach of that much is enough and a
            // tighter box means a smaller neighbour stencil and a faster test.
            let reach = radius + 2.0 * circumradius(&face_normals(&pc).unwrap()).unwrap();
            let region = [
                (centre[0] - reach, centre[0] + reach),
                (centre[1] - reach, centre[1] + reach),
                (centre[2] - reach, centre[2] + reach),
            ];
            let field = PhiField::for_threshold(&cell, &kept, &pc, 0.05).unwrap();
            // Few samples: this test claims an ordering and an upper bound, both of which
            // are far from the sampling noise, so paying for a tight error bar here would
            // only slow the suite down.
            let est = estimate_void_volume(&field, &region, 0.05, 1_200, 31).unwrap();
            // The void cannot be bigger than the removed sites' own cells hold. In practice
            // the bound is nearly attained, because the surviving atoms' cells only nibble a
            // thin shell where the cavity meets the crystal — so this is a statistical claim
            // and has to be tested against the error bar. A strict `<` would fail on noise,
            // which is how a real ceiling gets quietly loosened.
            let upper = removed as f64 * per_cell;
            assert!(
                est.volume <= upper + 3.0 * est.standard_error,
                "radius {radius}: {} exceeds the {removed} removed cells ({upper}) by more \
                 than 3 sigma (+/- {})",
                est.volume,
                est.standard_error
            );
            assert!(
                est.volume > prev - 3.0 * est.standard_error,
                "radius {radius}: {} did not grow past {prev} beyond noise (+/- {})",
                est.volume,
                est.standard_error
            );
            prev = est.volume;
        }
    }

    #[test]
    fn the_sample_set_is_reproducible() {
        // A void volume whose error bar is the only reproducible part cannot gate a
        // regression, and a test that flakes is worse than no test.
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let region = [
            (0.5, cells as f64 * a - 0.5),
            (0.5, cells as f64 * a - 0.5),
            (0.5, cells as f64 * a - 0.5),
        ];
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();
        let one = estimate_void_volume(&field, &region, 0.05, 2000, 99).unwrap();
        let two = estimate_void_volume(&field, &region, 0.05, 2000, 99).unwrap();
        assert_eq!(one.hits, two.hits);
        assert_eq!(one.volume, two.volume);
        // And a different seed samples somewhere else.
        let other = estimate_void_volume(&field, &region, 0.05, 2000, 100).unwrap();
        assert!(one.volume != other.volume || one.hits == 0 || other.hits == 0);
    }

    #[test]
    fn the_threshold_radius_is_minimal_and_exact() {
        // The contract for_threshold makes, and it has to be both halves at once. Minimal:
        // a radius much larger than R(1+ε) would put every atom of a large sample into
        // every query. Exact: any atom further than R(1+ε) from a query point cannot hold
        // it below the threshold, so it cannot change the φ > 1+ε decision.
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let circum = circumradius(&face_normals(&pc).unwrap()).unwrap();

        for eps in [0.0, 0.05, 0.2] {
            let field = PhiField::for_threshold(&cell, &sites, &pc, eps).unwrap();
            let want = circum * (1.0 + eps);
            assert!(
                field.radius() >= want && field.radius() <= want * 1.000_002,
                "radius {} against R(1+{eps}) = {want}",
                field.radius()
            );
            // Far below the box: if this crept up to the span, a 256k-atom sample would put
            // every atom into every query.
            assert!(
                field.radius() < a,
                "radius {} should be a fraction of a bond, not of the box",
                field.radius()
            );
        }
    }

    #[test]
    fn the_small_radius_gives_the_same_verdicts_as_the_whole_box() {
        // The real correctness claim behind for_threshold, and the one worth testing directly:
        // a radius of R(1+ε) decides `φ > 1+ε` exactly as a radius covering every atom in the
        // box would. If this drifts, every measured void volume is suspect, because the whole
        // method rests on the small radius being sufficient rather than merely convenient.
        let a = 3.5;
        let cells = 6;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let len = cells as f64 * a;
        let centre = [len / 2.0, len / 2.0, len / 2.0];

        // Carve a cavity several bonds across, so part of it lies beyond R(1+ε) from every
        // atom — the case where "no atoms in range" has to be reasoned about rather than
        // measured.
        let cavity = 1.5 * a;
        let kept: Vec<Coord3> = sites
            .iter()
            .copied()
            .filter(|&s| dist2(s, centre) > cavity * cavity)
            .collect();
        assert!(
            sites.len() - kept.len() > 10,
            "the cavity should have removed a real shell of atoms"
        );

        let small = PhiField::for_threshold(&cell, &kept, &pc, 0.05).unwrap();
        // A radius that cannot miss anything: the box diagonal covers every atom under any
        // image, so this field's verdicts are the reference.
        // A radius that cannot miss anything: longer than the box diagonal, so every atom
        // is within reach of every point under every image. This is the reference the small
        // radius is checked against, and it is deliberately extravagant -- at len*2 every
        // query visits every atom in the block, which is the cost this test is paying.
        let big = PhiField::new(&cell, &kept, &pc, len * 2.0).unwrap();
        assert!(
            small.radius() * 3.0 < big.radius(),
            "the two radii should differ enough for this to mean anything"
        );

        let mut rng = Rng(20240502);
        let mut voids = 0;
        // 400 samples rather than 4000: this compares two fields over a fixed sample set, so
        // it is a consistency check with no statistical error to shrink, and every extra
        // sample costs a full extra pass over every atom in the block.
        for _ in 0..400 {
            let mut x = [0.0f64; 3];
            for v in x.iter_mut() {
                *v = rng.unit() * len;
            }
            let a_ = small.is_void(x, 0.05);
            let b_ = big.is_void(x, 0.05);
            assert_eq!(
                a_,
                b_,
                "radius {} and radius {} disagree at {x:?}",
                small.radius(),
                big.radius()
            );
            if a_ {
                voids += 1;
            }
        }
        assert!(
            small.at(centre).is_infinite(),
            "the cavity centre should be out of range of the small radius, or this test \
             never exercises the case it is about"
        );
        assert!(voids > 0, "no sampled point was void");
    }

    #[test]
    fn a_cavity_reads_as_void_and_the_crystal_does_not() {
        // The end-to-end behaviour the radius reasoning is in service of, stated so a
        // regression cannot pass by making everything void.
        let a = 3.5;
        let cells = 8;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let len = cells as f64 * a;
        let centre = [len / 2.0, len / 2.0, len / 2.0];
        let cavity = 1.5 * a;
        let kept: Vec<Coord3> = sites
            .iter()
            .copied()
            .filter(|&s| dist2(s, centre) > cavity * cavity)
            .collect();
        let field = PhiField::for_threshold(&cell, &kept, &pc, 0.05).unwrap();

        // Deep in the cavity, no atom within R(1+ε): `at` has no value there, but the void
        // decision follows from the definition rather than from finding data.
        assert!(field.is_void(centre, 0.05), "a cavity centre is void");
        // Every surviving atom is crystal, including the ones on the cavity wall.
        for &p in kept.iter() {
            assert!(!field.is_void(p, 0.05), "atom at {p:?} read as void");
        }
    }

    #[test]
    fn a_larger_threshold_finds_fewer_atoms_than_a_smaller_one() {
        // The mechanism behind the radius contract, checked directly: widening ε must widen
        // the searched set, since more atoms become capable of holding a point below the
        // threshold. If this were flat, the radius would not track ε and the extrapolation
        // would be measuring something other than what it claims.
        let a = 3.5;
        let cells = 5;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let mut prev = 0.0;
        for eps in [0.0, 0.1, 0.5, 1.0] {
            let r = PhiField::for_threshold(&cell, &sites, &pc, eps)
                .unwrap()
                .radius();
            assert!(r > prev, "radius did not grow with epsilon at {eps}");
            prev = r;
        }
    }

    #[test]
    fn the_extrapolation_recovers_a_known_void() {
        // Carve a cavity, then require the quadratic extrapolation back to phi = 1 to land on
        // the volume the cells say it should have. The reference is the removed sites' own
        // cell volume, which is exact and is what method A charges.
        let a = 3.5;
        let cells = 5;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let per_cell = a * a * a / 4.0;
        let centre = [
            cells as f64 * a / 2.0,
            cells as f64 * a / 2.0,
            cells as f64 * a / 2.0,
        ];
        let radius = 3.2;
        let kept: Vec<Coord3> = sites
            .iter()
            .copied()
            .filter(|&s| dist2(s, centre) > radius * radius)
            .collect();
        let removed = sites.len() - kept.len();
        assert!(removed > 0, "the sphere should have removed atoms");
        let reach = radius + 2.0 * circumradius(&face_normals(&pc).unwrap()).unwrap();
        let region = [
            (centre[0] - reach, centre[0] + reach),
            (centre[1] - reach, centre[1] + reach),
            (centre[2] - reach, centre[2] + reach),
        ];
        let field = PhiField::for_threshold(&cell, &kept, &pc, 0.10).unwrap();
        let ex = extrapolate_to_phi_one(&field, &region, &[0.02, 0.05, 0.10], 20_000, 5).unwrap();

        // {φ > 1+ε} is a subset of {φ > 1}, so the extrapolation lands *above* every
        // measurement: it puts back the shell that any threshold below 1 shaves off.
        for m in ex.measurements.iter() {
            assert!(
                m.volume > 0.0,
                "a carved sphere must have volume at every epsilon"
            );
            assert!(
                ex.volume_at_one > m.volume,
                "extrapolated {} should exceed the measurement {}",
                ex.volume_at_one,
                m.volume
            );
        }
        // Monotone: a stricter threshold can only shrink the superlevel set.
        assert!(
            ex.measurements[0].volume > ex.measurements[1].volume
                && ex.measurements[1].volume > ex.measurements[2].volume,
            "volumes should fall as epsilon rises"
        );
        assert!(
            ex.slope_negative(),
            "the fitted curve should be falling in epsilon"
        );
        // The real check: against the exact per-site ceiling of the removed cells.
        let upper = removed as f64 * per_cell;
        assert!(
            ex.volume_at_one < upper && ex.volume_at_one > 0.5 * upper,
            "extrapolated {} against a ceiling of {upper} ({} removed cells)",
            ex.volume_at_one,
            removed
        );
        assert!(
            ex.is_trustworthy(),
            "the fit should be good here: {}",
            ex.summary()
        );
    }

    #[test]
    fn the_extrapolation_refuses_what_it_cannot_fit() {
        // Two epsilons cannot support a quadratic, and epsilons of zero would make the fit
        // agree with itself at the point it is supposed to be predicting. Both are refused
        // rather than quietly producing an uncorrected number.
        let a = 3.5;
        let cells = 3;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let region = [
            (0.0, cells as f64 * a),
            (0.0, cells as f64 * a),
            (0.0, cells as f64 * a),
        ];
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();

        let one = extrapolate_to_phi_one(&field, &region, &[0.05], 10, 1).unwrap_err();
        assert!(one.to_string().contains("at least 3 epsilons"), "{one}");
        let two = extrapolate_to_phi_one(&field, &region, &[0.05, 0.10], 10, 1).unwrap_err();
        assert!(two.to_string().contains("at least 3 epsilons"), "{two}");
        let with_zero =
            extrapolate_to_phi_one(&field, &region, &[0.0, 0.05, 0.10], 10, 1).unwrap_err();
        assert!(
            with_zero.to_string().contains("must all be positive"),
            "{with_zero}"
        );
        // Same epsilon twice is degenerate: a singular normal matrix, not a fit.
        // Repeated epsilons give fewer independent rows than the fit has coefficients, which
        // a singular-matrix check would only catch by luck of the rounding.
        let same = extrapolate_to_phi_one(&field, &region, &[0.05, 0.05, 0.10], 10, 1).unwrap_err();
        assert!(same.to_string().contains("repeat"), "{same}");
    }

    #[test]
    fn a_defect_free_crystal_extrapolates_to_nothing() {
        // The failure that motivated replacing the straight-line fit. A perfect block has no
        // void, so the extrapolation must say so rather than reporting the thermal shell it
        // measured — and it must get there by following the curvature, because the shell's
        // volume is strongly nonlinear in epsilon near phi = 1.
        //
        // The block is built on ideal sites here, so this isolates the fit's behaviour from the
        // displacement of a relaxed crystal; the integration test covers the real fixture,
        // where the shell is large enough that a straight line extrapolates to thousands of
        // cubic Angstroms of void in a crystal that has none.
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let region = [
            (1.0, cells as f64 * a - 1.0),
            (1.0, cells as f64 * a - 1.0),
            (1.0, cells as f64 * a - 1.0),
        ];
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.10).unwrap();
        let ex = extrapolate_to_phi_one(&field, &region, &[0.02, 0.05, 0.10], 20_000, 99).unwrap();
        let region_volume = (region[0].1 - region[0].0).powi(3);
        assert!(
            ex.volume_at_one < 0.02 * region_volume,
            "a perfect crystal extrapolated to {} of void in a region of {region_volume}; {}",
            ex.volume_at_one,
            ex.summary()
        );
    }

    #[test]
    fn a_degenerate_epsilon_is_refused() {
        // epsilon <= 0 thresholds at phi = 1, where a perfect crystal has measure-zero
        // hits but a real one has a thermal shell of them — the exact reason for measuring
        // above 1. Refusing the request is better than returning a meaningless volume.
        let a = 3.5;
        let cells = 3;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let region = [
            (0.0, cells as f64 * a),
            (0.0, cells as f64 * a),
            (0.0, cells as f64 * a),
        ];
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();
        let e = estimate_void_volume(&field, &region, 0.0, 10, 1).unwrap_err();
        assert!(e.to_string().contains("epsilon"), "{e}");
    }

    #[test]
    fn an_empty_region_is_refused() {
        let a = 3.5;
        let cell = cell_of(a, 3);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, 3, a);
        let field = PhiField::new(&cell, &sites, &pc, 4.0).unwrap();
        let e = estimate_void_volume(&field, &[(1.0, 1.0), (1.0, 2.0), (1.0, 2.0)], 0.05, 10, 1)
            .unwrap_err();
        assert!(e.to_string().contains("zero extent"), "{e}");
    }

    #[test]
    fn a_single_threshold_excess_reports_its_own_measurement() {
        // With one epsilon, a quadratic fit through one point (a rank-1 normal matrix that is
        // not exactly singular in floating point) reads 3603.5 A^3 against its own row's
        // 1393.8 A^3. One threshold is a measurement.
        let a = 3.5;
        let cells = 6;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let centre = [cells as f64 * a / 2.0 + 0.01; 3];
        let kept: Vec<Coord3> = sites
            .iter()
            .copied()
            .filter(|&s| dist2(s, centre) > (1.1 * a).powi(2))
            .collect();
        let n_removed = (sites.len() - kept.len()) as f64;
        let sample = PhiField::for_threshold(&cell, &kept, &pc, 0.05).unwrap();
        let reference = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();
        let region = [(a, 5.0 * a), (a, 5.0 * a), (a, 5.0 * a)];
        for eps in [0.01, 0.05] {
            let x = excess_void_volume(&sample, &reference, &region, &[eps], 50_000, 7).unwrap();
            assert_eq!(x.measurements.len(), 1);
            assert_eq!(x.volume_at_one, x.measurements[0].volume, "eps {eps}");
            assert_eq!(x.standard_error, x.measurements[0].standard_error);
            assert!(x.summary().contains("not extrapolated"), "{}", x.summary());
            // And it is the void: the removed cells, eroded a little by eps.
            let want = n_removed * a * a * a / 4.0;
            assert!(
                x.volume_at_one > 0.6 * want && x.volume_at_one < 1.1 * want,
                "eps {eps}: {} against {want}",
                x.volume_at_one
            );
        }
        // Two thresholds would be the straight-line fit the raw path refuses.
        let e =
            excess_void_volume(&sample, &reference, &region, &[0.01, 0.02], 1000, 7).unwrap_err();
        assert!(e.to_string().contains("at least 3"), "{e}");
        // Three or more extrapolate, and the summary says so.
        let x = excess_void_volume(&sample, &reference, &region, &[0.01, 0.02, 0.04], 50_000, 7)
            .unwrap();
        assert!(x.summary().contains("extrapolated"), "{}", x.summary());
    }

    #[test]
    fn a_threshold_beyond_the_fields_search_radius_is_refused() {
        // A field built for eps = 0.01 and measured at larger eps misses atoms beyond its
        // radius and counts the points they hold as void (V(eps) of a vacancy stops falling
        // at ~0.53 of its cell instead of shrinking as (1 - eps)^3).
        let a = 3.5;
        let cell = cell_of(a, 3);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, 3, a);
        let field = PhiField::for_threshold(&cell, &sites, &pc, 0.05).unwrap();
        assert!(field.max_epsilon() >= 0.05 && field.max_epsilon() < 0.05 + 1e-5);
        let region = [(0.0, 3.0 * a), (0.0, 3.0 * a), (0.0, 3.0 * a)];
        assert!(estimate_void_volume(&field, &region, 0.05, 10, 1).is_ok());
        let e = estimate_void_volume(&field, &region, 0.06, 10, 1).unwrap_err();
        assert!(e.to_string().contains("search radius"), "{e}");
    }

    #[test]
    fn a_vacancy_shrinks_as_the_cube_of_one_minus_epsilon() {
        // The law the shell-free window rests on: each face of an empty cell moves inward by
        // eps times its own distance, so the cell shrinks homothetically. Pointwise and exact:
        // the vertex scaled by (1 - eps) is on the threshold.
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let victim = sites[sites.len() / 2];
        let kept: Vec<Coord3> = sites.iter().copied().filter(|&s| s != victim).collect();
        let field = PhiField::for_threshold(&cell, &kept, &pc, 0.6).unwrap();
        for eps in [0.1, 0.3, 0.5] {
            for &v in ws_vertices(&face_normals(&pc).unwrap()).iter() {
                let inside = add(victim, scale(v, (1.0 - eps) * 0.999));
                let outside = add(victim, scale(v, (1.0 - eps) * 1.001));
                assert!(field.is_void(inside, eps), "eps {eps}: {inside:?}");
                assert!(!field.is_void(outside, eps), "eps {eps}: {outside:?}");
            }
        }
    }

    #[test]
    fn the_face_gap_is_zero_on_a_lattice_and_measures_a_displacement() {
        let a = 3.5;
        let cells = 4;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, mut sites) = block(&pc, cells, a);
        let g = max_face_gap(&cell, &sites, &pc).unwrap();
        assert!(g.max.abs() < 1e-12, "{}", g.max);
        assert_eq!(
            g.bonds,
            sites.len() * 12,
            "every fcc atom has 12 face neighbours"
        );
        // Moving one atom by t along a face vector v parts it from the neighbour behind it by
        // t: the gap is t / |v|.
        let nn = a / 2f64.sqrt();
        let t = 0.07;
        let u = [t / 2f64.sqrt(), t / 2f64.sqrt(), 0.0];
        sites[5] = add(sites[5], u);
        let g = max_face_gap(&cell, &sites, &pc).unwrap();
        assert!((g.max - t / nn).abs() < 1e-12, "{} vs {}", g.max, t / nn);
        // The window starts there and spans WINDOW_WIDTH; a frame parted too far has none.
        let w = window_epsilons(g.max).unwrap();
        assert_eq!(w.len(), WINDOW_POINTS);
        assert!((w[0] - t / nn).abs() < 1e-12 && (w[6] - w[0] - WINDOW_WIDTH).abs() < 1e-12);
        assert_eq!(window_epsilons(0.0).unwrap()[0], WINDOW_FLOOR);
        assert!(window_epsilons(0.7).is_err());
    }

    #[test]
    fn the_shell_free_window_recovers_a_void_under_thermal_noise() {
        // An 13-site FCC cluster (a site and its 12 neighbours) with 0.05 A Gaussian noise on
        // every atom: V(0.01) is mostly thermal shell; the window recovers the cluster.
        let a = 3.5;
        let cells = 6;
        let cell = cell_of(a, cells);
        let pc = fcc_pc(a);
        let (_, sites) = block(&pc, cells, a);
        let centre = [cells as f64 * a / 2.0; 3];
        let mut rng = Rng(3);
        let mut gauss = || {
            let (u, v) = (rng.unit().max(1e-300), rng.unit());
            0.05 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let kept: Vec<Coord3> = sites
            .iter()
            .copied()
            .filter(|&s| dist2(s, centre) > 2.6 * 2.6)
            .map(|s| [s[0] + gauss(), s[1] + gauss(), s[2] + gauss()])
            .collect();
        let n_removed = (sites.len() - kept.len()) as f64;
        assert_eq!(n_removed, 13.0);
        let want = n_removed * a * a * a / 4.0;
        let gap = max_face_gap(&cell, &kept, &pc).unwrap();
        let top = *window_epsilons(gap.max).unwrap().last().unwrap();
        let field = PhiField::for_threshold(&cell, &kept, &pc, top).unwrap();
        let region = [(a, 5.0 * a), (a, 5.0 * a), (a, 5.0 * a)];
        let raw = estimate_void_volume(&field, &region, 0.01, 100_000, 5).unwrap();
        assert!(
            raw.volume > 1.2 * want,
            "the shell: {} vs {want}",
            raw.volume
        );
        let x = extrapolate_shell_free(&field, &region, gap, 100_000, 5).unwrap();
        assert!(
            (x.volume_at_one / want - 1.0).abs() < 0.10,
            "{} vs {want} ({})",
            x.volume_at_one,
            x.summary()
        );
    }
}
