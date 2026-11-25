//! `Frame` = cell + particles + metadata (design §7.1).

use crate::cell::UnitCell;
use crate::data::particle::ParticleData;

/// Frame-level metadata (timestep, time, source id, labels).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub timestep: i64,
    pub time: Option<f64>,
    /// LAMMPS unit style from a dump's `ITEM: UNITS` (`metal`, `real`, ...).
    pub units: Option<String>,
    pub source_id: Option<String>,
    pub labels: Vec<String>,
    /// The cell was inferred from the atoms rather than read from the file.
    ///
    /// A cell-less file is read in the box its atoms occupy, so every number
    /// derived from the cell -- the density above all -- is the atoms' own
    /// rather than a stand-in's. This flag says the box was not in the file,
    /// so the output can say which numbers came from where.
    pub cell_inferred: bool,
}

/// One realized snapshot in memory.
#[derive(Debug, Clone)]
pub struct Frame {
    pub cell: UnitCell,
    pub particles: ParticleData,
    pub meta: Metadata,
    /// Optional user-provided labels for atom types: index by type-1.
    pub type_labels: Vec<String>,
    /// Per-type masses from a data file's `Masses` section: index by type-1,
    /// `NaN` for a type the file gave no mass. Empty when the source has none.
    pub type_masses: Vec<f64>,
}

/// The longest box edge over the smallest box height a frame may have.
const MAX_SKEW: f64 = 1e6;

impl Frame {
    pub fn new(cell: UnitCell, particles: ParticleData) -> Self {
        Frame {
            cell,
            particles,
            meta: Metadata::default(),
            type_labels: Vec::new(),
            type_masses: Vec::new(),
        }
    }

    pub fn natoms(&self) -> usize {
        self.particles.natoms
    }

    /// Whether a frame a reader produced is one the analyses can take: as
    /// many positions, ids and types as the atom count says, every coordinate
    /// and every box component finite, a box of nonzero volume. Every frame
    /// the readers return passes through this, so a truncated section or a
    /// `nan`/`inf`/`1e400` in a file is an error naming the atom, not a
    /// panic or a silently wrong result further on.
    pub fn check_input(&self) -> crate::error::Result<()> {
        use crate::error::Error;
        let p = &self.particles;
        let n = p.natoms;
        for (what, len) in [
            ("positions", p.positions.len()),
            ("ids", p.ids.len()),
            ("types", p.types.len()),
        ] {
            if len != n {
                return Err(Error::Input(format!(
                    "the frame declares {n} atoms but holds {len} {what} (a truncated or \
                     missing atoms section?)"
                )));
            }
        }
        let h = self.cell.h_matrix();
        let o = self.cell.origin;
        if !(h.m.iter().flatten().all(|v| v.is_finite()) && o.iter().all(|v| v.is_finite())) {
            return Err(Error::Input(format!(
                "the box is not finite: vectors {:?}, origin {o:?}",
                h.m
            )));
        }
        let v = self.cell.volume();
        if !(v > 0.0 && v.is_finite()) {
            return Err(Error::Input(format!(
                "the box has no volume ({v} Å³): vectors {:?}",
                h.m
            )));
        }
        // A box so skewed that one of its heights is a millionth of its
        // longest edge is no simulation box (LAMMPS keeps tilts within half a
        // box length): image searches step through ~edge/height cells, and a
        // tilt of 1e20 Å on a 7 Å box would hang the neighbour search.
        let (len, ht) = (self.cell.lengths(), self.cell.heights());
        let longest = len.iter().copied().fold(0.0, f64::max);
        let thinnest = ht.iter().copied().fold(f64::INFINITY, f64::min);
        if longest > MAX_SKEW * thinnest {
            return Err(Error::Input(format!(
                "the box is degenerate: its longest edge ({longest:.4e} Å) is more than {MAX_SKEW:e} \
                 times its smallest height ({thinnest:.4e} Å); reduce the tilt (rmda convert \
                 --reduce-tilt) or check the box"
            )));
        }
        if let Some(i) = p
            .positions
            .iter()
            .position(|x| !x.iter().all(|c| c.is_finite()))
        {
            return Err(Error::Input(format!(
                "atom id {} (row {} of the frame) has a non-finite coordinate {:?}",
                p.ids[i],
                i + 1,
                p.positions[i]
            )));
        }
        Ok(())
    }

    pub fn ntypes(&self) -> usize {
        if self.particles.types.is_empty() {
            0
        } else {
            *self.particles.types.iter().max().unwrap_or(&0) as usize
        }
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Reduce the cell's tilts LAMMPS-style ([`UnitCell::reduce_tilts`]) and
    /// remap the atoms into the reduced cell, as LAMMPS does after a flip
    /// (`Domain::image_flip` + `remap`).
    ///
    /// Positions are wrapped along the periodic axes in fractional space; an
    /// atom already inside the new cell keeps its bits. Image flags, when the
    /// frame has them, are carried so that every unwrapped position
    /// `x + n·h` is unchanged: `n' = n·M⁻¹ + s`, with `s` the wrap shift.
    /// Returns whether anything changed.
    pub fn reduce_tilts(&mut self) -> crate::error::Result<bool> {
        use crate::cell::{image_ipart, image_jpart, image_kpart, pack_image};
        let Some((cell, m)) = self.cell.reduce_tilts() else {
            return Ok(false);
        };
        // M is unit lower-triangular: [[1,0,0],[k,1,0],[q,p,1]], whose
        // inverse is [[1,0,0],[-k,1,0],[kp-q,-p,1]].
        let (k, q, p) = (m[1][0], m[2][0], m[2][1]);
        let minv = [[1, 0, 0], [-k, 1, 0], [k * p - q, -p, 1]];
        let inv_t = cell.inverse_transposed()?;
        let pbc = cell.pbc;
        let n = self.particles.positions.len();
        let mut images = self.particles.images.take().filter(|im| im.len() == n);
        for i in 0..n {
            let x = self.particles.positions[i];
            let mut lam = cell.x2lamda_pre(x, &inv_t);
            let mut s = [0i64; 3];
            let mut moved = false;
            for d in 0..3 {
                if pbc[d] {
                    let fl = lam[d].floor();
                    let mut f = lam[d] - fl;
                    let mut sd = fl as i64;
                    if f >= 1.0 {
                        f = 0.0;
                        sd += 1;
                    }
                    if sd != 0 {
                        lam[d] = f;
                        s[d] = sd;
                        moved = true;
                    }
                }
            }
            if moved {
                self.particles.positions[i] = cell.lamda2x(lam);
            }
            if let Some(im) = images.as_mut() {
                let old = [image_ipart(im[i]), image_jpart(im[i]), image_kpart(im[i])];
                let mut new = s;
                for (c, nc) in new.iter_mut().enumerate() {
                    *nc += (0..3).map(|r| old[r] * minv[r][c]).sum::<i64>();
                }
                im[i] = pack_image(new[0], new[1], new[2]);
            }
        }
        self.particles.images = images;
        self.cell = cell;
        Ok(true)
    }
}
