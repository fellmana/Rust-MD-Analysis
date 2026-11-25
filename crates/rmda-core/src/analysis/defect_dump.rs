//! Writing Wigner–Seitz defects out as separate atom files (`rmda ws --defects`).
//!
//! # Why this is not just `write_frame_with`
//!
//! Every other analysis publishes its results as columns on the input atoms, and
//! that is the right shape for them. Defects are not that shape. A vacancy is an
//! *empty lattice site*: there is no atom there, so no per-atom column can hold
//! one, and a file with the atoms still in it cannot show the user where the holes
//! are. Interstitials are the mirror case — they are atoms, but they arrive as a
//! *count* (`occupancy - 1` summed over crowded sites) with no record of which
//! atoms those were, because the assignment makes no such choice.
//!
//! So the defect results exist only as scalars and a per-*site* occupancy array
//! that is deliberately kept out of the atom stream
//! (`per_particle = false`, `ws.rs`). `ws_occupancy` has one entry per site and the
//! atom table has one row per atom; on a block with three vacancies those are 864
//! against 861, and writing it as a column misaligns the table rather than failing.
//! Hence three files instead of three columns.
//!
//! # What "the interstitial atoms" means here, precisely
//!
//! When `k > 1` atoms claim one site, the tally counts `k - 1` interstitial atoms.
//! Which `k - 1` of the `k` is *not* a question the algorithm answers: every one of
//! them is at its nearest lattice point, and picking a "real" one would invent a
//! distinction the assignment never made. So `interstitials.xyz` contains **every
//! atom seated on an over-occupied site**, carries the site's occupancy as a
//! column, and the file is as wide as the crowding rather than one atom per counted
//! interstitial. Use the occupancy column, or `sum(occupancy - 1)`, for the count.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::analysis::void::VoidCounts;
use crate::cell::{Coord3, UnitCell};
use crate::data::frame::{Frame, Metadata};
use crate::data::particle::{MdaTag, ParticleData};
use crate::data::property::{DType, Property, PropertyStore};
use crate::error::{Error, Result};
use crate::io::{XyzOrigin, XyzWriter};
use crate::lattice::fit::InRegionSites;
use crate::lattice::sites::SiteLattice;

/// Species label for a vacancy *site*, which has no atom and so no element.
///
/// A pseudo-element rather than an empty frame: OVITO's structure type and
/// selection machinery both key off species, so a file of vacancies has to be
/// selectable in the same breath as a file of atoms. Kept short and upper-case to
/// read unambiguously in a rendered scene.
pub const VACANCY_SPECIES: &str = "VAC";

/// Species label for an over-occupied lattice site.
pub const INTERSTITIAL_SITE_SPECIES: &str = "INT";

/// Site positions, kept lazy so a run that only wants counts pays nothing.
#[derive(Debug, Clone)]
pub enum SiteGrid {
    /// A declared reference lattice that tiles the box.
    Lattice(SiteLattice),
    /// A region fitted by the self-built engine.
    Region(InRegionSites),
    /// Sites given directly as positions, which is what a reference frame supplies:
    /// its atoms *are* the sites, so there is no lattice to reconstruct and no index
    /// arithmetic to get wrong.
    Positions(Vec<Coord3>),
}

impl SiteGrid {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SiteGrid::Lattice(l) => l.len(),
            SiteGrid::Region(r) => r.len(),
            SiteGrid::Positions(p) => p.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn position(&self, idx: usize) -> Coord3 {
        match self {
            SiteGrid::Lattice(l) => l.get(idx),
            SiteGrid::Region(r) => r.position(idx),
            SiteGrid::Positions(p) => p[idx],
        }
    }
}

/// Everything the writer needs, from either Wigner–Seitz engine.
///
/// Deliberately a plain bundle of the arrays both engines already produce, rather
/// than a trait each implements: the two engines differ in *where the reference
/// came from*, not in what a vacancy is, and the only thing that genuinely differs
/// is how a site position is computed — which is what [`SiteGrid`] absorbs.
#[derive(Debug, Clone)]
pub struct DefectDump {
    /// Cell the site grid lives in.
    ///
    /// The cell the sites were *assigned in*: the input cell normally, or the
    /// reference cell under `--map to-reference`.
    ///
    /// Not always the input frame's cell, and that is not a detail: under
    /// `--map to-reference` the sites belong to the reference frame while the atoms
    /// still belong to the input one. The site files therefore carry this cell and
    /// `interstitials.xyz` carries the input cell, because writing reference-frame
    /// positions under the input cell's lattice would be a file that looks fine and
    /// plots in the wrong place.
    pub cell: UnitCell,
    /// Occupancy per site, parallel to [`DefectDump::sites`].
    pub occupancy: Vec<u32>,
    /// The site grid `occupancy` indexes.
    pub sites: SiteGrid,
    /// Site index per input atom. Negative entries are atoms with no site within
    /// tolerance, or one outside the region; they appear in no output file.
    pub site_of_atom: Vec<i64>,
    /// Void cluster id per site, `-1` where the site is not part of a reported
    /// void. `None` when void detection did not run.
    pub void_of_site: Option<Vec<i32>>,
}

impl DefectDump {
    /// Check the arrays agree with each other.
    ///
    /// Worth doing explicitly: a length mismatch here would otherwise be caught
    /// by an out-of-bounds index or, worse, by reading a neighbouring site's
    /// occupancy and writing a vacancy at a perfectly good position.
    pub fn validate(&self) -> Result<()> {
        let n = self.occupancy.len();
        if self.sites.len() != n {
            return Err(Error::invalid(format!(
                "defect dump: {} occupancy entries but {} sites",
                n,
                self.sites.len()
            )));
        }
        if let Some(v) = &self.void_of_site {
            if v.len() != n {
                return Err(Error::invalid(format!(
                    "defect dump: {} void ids but {} sites",
                    v.len(),
                    n
                )));
            }
        }
        Ok(())
    }

    /// Atom indices seated on an over-occupied site.
    ///
    /// See the module docs on why this is every atom on such a site rather than
    /// one atom per counted interstitial.
    pub fn interstitial_atoms(&self) -> Vec<usize> {
        self.site_of_atom
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= 0 && self.occupancy[s as usize] > 1)
            .map(|(i, _)| i)
            .collect()
    }

    /// Build the per-site void ids from a void clustering result.
    ///
    /// Ids number the clusters in `VoidCounts` order, so `voids[i].sites` are
    /// exactly the entries carrying id `i`.
    pub fn with_voids(mut self, counts: &VoidCounts) -> Self {
        let mut ids = vec![-1i32; self.occupancy.len()];
        for (id, cluster) in counts.voids.iter().enumerate() {
            for &s in &cluster.sites {
                if let Some(slot) = ids.get_mut(s) {
                    *slot = id as i32;
                }
            }
        }
        self.void_of_site = Some(ids);
        self
    }
}

/// Where and whether to write the defect files.
#[derive(Debug, Clone)]
pub struct DefectDumpOptions {
    /// Directory to write into. Created if absent.
    pub dir: PathBuf,
    /// Write a file even when it would hold zero atoms.
    ///
    /// On by default: a fixed set of three filenames is easier to script against
    /// than one whose contents decide which files exist, and a 0-atom extended XYZ
    /// is valid. Turn it off to keep a defects directory free of empty files.
    pub write_empty: bool,
    /// Where each written cell's origin is placed.
    ///
    /// Defaults to [`XyzOrigin::Native`], so the defect files describe the same
    /// box the input frame had and can be overlaid on it directly.
    pub origin: XyzOrigin,
}

impl Default for DefectDumpOptions {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("defects"),
            write_empty: true,
            origin: XyzOrigin::Native,
        }
    }
}

/// One file that was written.
#[derive(Debug, Clone)]
pub struct DumpFile {
    pub name: &'static str,
    pub path: PathBuf,
    /// Rows written, i.e. atoms or sites in the file.
    pub rows: usize,
    /// Rows deliberately skipped because they were empty.
    pub skipped_empty: bool,
}

/// What a dump produced, for the CLI to report.
#[derive(Debug, Clone, Default)]
pub struct DefectDumpReport {
    pub vacancies: usize,
    pub interstitial_sites: usize,
    /// Atoms on over-occupied sites. Greater than the excess-atom count whenever a
    /// site holds three or more atoms; see the module docs.
    pub interstitial_atoms: usize,
    /// `sum(max(occupancy - 1, 0))`, the tally's interstitial count.
    pub excess_atoms: usize,
    pub files: Vec<DumpFile>,
}

impl DefectDumpReport {
    /// One line per file plus the counts that distinguish them.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} vacancy site(s), {} interstitial site(s) holding {} atom(s) ({} excess)",
            self.vacancies, self.interstitial_sites, self.interstitial_atoms, self.excess_atoms
        );
        push_file_lines(&self.files, &mut s);
        s
    }
}

/// The per-file lines both dump reports append to their summary.
fn push_file_lines(files: &[DumpFile], s: &mut String) {
    for f in files {
        if f.skipped_empty {
            s.push_str(&format!("\n  {} (empty, not written)", f.name));
        } else {
            s.push_str(&format!("\n  {}: {} row(s)", f.name, f.rows));
        }
    }
}

/// Write `vacancies.xyz`, `interstitials.xyz` and `interstitials_sites.xyz`.
///
/// `frame` supplies the cell, the timestep, and — for the interstitial atoms —
/// the species. Site files are written in the *reference* frame, which is the
/// frame the site grid was built in; the comment line records that, because a
/// position means nothing without knowing which cell it is relative to.
pub fn write_defect_files(
    frame: &Frame,
    dump: &DefectDump,
    opts: &DefectDumpOptions,
) -> Result<DefectDumpReport> {
    dump.validate()?;
    std::fs::create_dir_all(&opts.dir).map_err(|e| {
        Error::Io(format!(
            "cannot create defect directory {}: {e}",
            opts.dir.display()
        ))
    })?;

    let mut report = DefectDumpReport::default();

    // --- vacancies: the empty sites themselves -----------------------------
    let vac_sites: Vec<usize> = dump
        .occupancy
        .iter()
        .enumerate()
        .filter(|(_, &o)| o == 0)
        .map(|(i, _)| i)
        .collect();
    report.vacancies = vac_sites.len();
    let mut vac_store = PropertyStore::new();
    push_i64(&mut vac_store, "site_index", &vac_sites);
    if let Some(v) = &dump.void_of_site {
        push_i32_sites(&mut vac_store, "void_id", v, &vac_sites);
    }
    let vac_frame = site_frame(frame, &vac_sites, &dump.sites, VACANCY_SPECIES, &dump.cell);
    report.files.push(write_one(
        &opts.dir,
        "vacancies.xyz",
        &vac_frame,
        &vac_store,
        vac_sites.len(),
        opts.write_empty,
        opts.origin,
    )?);

    // --- interstitial atoms: the atoms on crowded sites --------------------
    let inter_atoms = dump.interstitial_atoms();
    report.interstitial_atoms = inter_atoms.len();
    let mut inter_store = PropertyStore::new();
    push_i64_atoms(
        &mut inter_store,
        "site_index",
        &dump.site_of_atom,
        &inter_atoms,
    );
    push_i32_atoms(
        &mut inter_store,
        "site_occupancy",
        &dump.occupancy,
        &dump.site_of_atom,
        &inter_atoms,
    );
    let inter_frame = subset_frame(frame, &inter_atoms);
    report.files.push(write_one(
        &opts.dir,
        "interstitials.xyz",
        &inter_frame,
        &inter_store,
        inter_atoms.len(),
        opts.write_empty,
        opts.origin,
    )?);

    // --- interstitial sites: the crowded lattice points --------------------
    let inter_sites: Vec<usize> = dump
        .occupancy
        .iter()
        .enumerate()
        .filter(|(_, &o)| o > 1)
        .map(|(i, _)| i)
        .collect();
    report.interstitial_sites = inter_sites.len();
    report.excess_atoms = dump
        .occupancy
        .iter()
        .map(|&o| o.saturating_sub(1) as usize)
        .sum();
    let mut isite_store = PropertyStore::new();
    push_i64(&mut isite_store, "site_index", &inter_sites);
    let occ_of: Vec<u32> = inter_sites.iter().map(|&s| dump.occupancy[s]).collect();
    let mut occ_prop = Property::with_capacity(
        "site_occupancy".to_owned(),
        DType::I32,
        true,
        1,
        occ_of.len(),
    );
    for o in &occ_of {
        occ_prop.push(*o as f64);
    }
    isite_store.register(occ_prop);
    let isite_frame = site_frame(
        frame,
        &inter_sites,
        &dump.sites,
        INTERSTITIAL_SITE_SPECIES,
        &dump.cell,
    );
    report.files.push(write_one(
        &opts.dir,
        "interstitials_sites.xyz",
        &isite_frame,
        &isite_store,
        inter_sites.len(),
        opts.write_empty,
        opts.origin,
    )?);

    Ok(report)
}

/// Write one file, or record that it was skipped for being empty.
fn write_one(
    dir: &Path,
    name: &'static str,
    frame: &Frame,
    store: &PropertyStore,
    rows: usize,
    write_empty: bool,
    origin: XyzOrigin,
) -> Result<DumpFile> {
    let path = dir.join(name);
    if rows == 0 && !write_empty {
        return Ok(DumpFile {
            name,
            path,
            rows: 0,
            skipped_empty: true,
        });
    }
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&path)
            .map_err(|e| Error::Io(format!("cannot create {}: {e}", path.display())))?,
    );
    XyzWriter::write_frame_with_origin(&mut f, frame, store, origin)
        .map_err(|e| Error::Io(format!("cannot write {}: {e}", path.display())))?;
    f.flush()
        .map_err(|e| Error::Io(format!("cannot flush {}: {e}", path.display())))?;
    Ok(DumpFile {
        name,
        path,
        rows,
        skipped_empty: false,
    })
}

/// A frame whose "atoms" are lattice sites, all of one pseudo-species.
///
/// The cell is the site grid's own, not the input frame's: under
/// `--map to-reference` those differ.
fn site_frame(
    base: &Frame,
    sites: &[usize],
    grid: &SiteGrid,
    species: &str,
    cell: &UnitCell,
) -> Frame {
    let mut particles = ParticleData::new(sites.len());
    for &s in sites {
        // Wrapped into the cell that goes in the header, always.
        //
        // A site grid need not live in the same box the header describes. Under
        // `--self` the region is indexed modulo the box, so site index 0 can sit
        // at the *far* edge of the fitted reference's origin: on the 256k-atom
        // relaxed fixture the fitted origin lands near 74 A while the input box
        // is [2.3, 143.4], and unwrapped site positions came out at 205-212 A --
        // 60 A outside a 141 A box, in a file whose header claims that box. The
        // occupancy and the void ids are unaffected; only the printed coordinate
        // was wrong, and a vacancy rendered outside its own cell is worse than
        // useless because it still looks like a plausible point defect.
        //
        // Wrapping into `cell` is exactly the convention the rest of the writer
        // uses for input positions, so the site files and the interstitial-atom
        // file end up in one frame of reference.
        particles.positions.push(cell.wrap(grid.position(s)));
        particles.types.push(1);
        particles.ids.push(s as MdaTag + 1);
    }
    let mut frame = Frame::new(*cell, particles);
    frame.type_labels = vec![species.to_owned()];
    frame.meta = Metadata {
        timestep: base.meta.timestep,
        ..Metadata::default()
    };
    frame
}

/// A frame holding only the given input atoms, keeping their species and cell.
fn subset_frame(base: &Frame, keep: &[usize]) -> Frame {
    let mut particles = ParticleData::new(keep.len());
    for &i in keep {
        particles.positions.push(base.particles.positions[i]);
        particles.types.push(base.particles.types[i]);
        particles.ids.push(
            base.particles
                .ids
                .get(i)
                .copied()
                .unwrap_or(i as MdaTag + 1),
        );
    }
    let mut frame = Frame::new(base.cell, particles);
    frame.type_labels = base.type_labels.clone();
    frame.meta = Metadata {
        timestep: base.meta.timestep,
        ..Metadata::default()
    };
    frame
}

fn push_i64(store: &mut PropertyStore, name: &str, values: &[usize]) {
    let mut p = Property::with_capacity(name.to_owned(), DType::I64, true, 1, values.len());
    for &v in values {
        p.push(v as f64);
    }
    store.register(p);
}

fn push_i32_sites(store: &mut PropertyStore, name: &str, all: &[i32], which: &[usize]) {
    let mut p = Property::with_capacity(name.to_owned(), DType::I32, true, 1, which.len());
    for &s in which {
        p.push(all[s] as f64);
    }
    store.register(p);
}

fn push_i64_atoms(store: &mut PropertyStore, name: &str, all: &[i64], which: &[usize]) {
    let mut p = Property::with_capacity(name.to_owned(), DType::I64, true, 1, which.len());
    for &i in which {
        p.push(all[i] as f64);
    }
    store.register(p);
}

fn push_i32_atoms(
    store: &mut PropertyStore,
    name: &str,
    occupancy: &[u32],
    site_of_atom: &[i64],
    which: &[usize],
) {
    let mut p = Property::with_capacity(name.to_owned(), DType::I32, true, 1, which.len());
    for &i in which {
        let s = site_of_atom[i];
        p.push(if s >= 0 {
            occupancy[s as usize] as f64
        } else {
            0.0
        });
    }
    store.register(p);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::fcc::Fcc;
    use crate::lattice::Lattice;

    fn frame_with(n: usize, a: f64) -> Frame {
        let cell = crate::cell::UnitCell::orthorhombic(n as f64 * a, n as f64 * a, n as f64 * a);
        let lattice = SiteLattice::build(&cell, &Fcc { a }.primitive(), [0.0; 3]).unwrap();
        let mut particles = ParticleData::new(lattice.len());
        for p in lattice.iter() {
            particles.positions.push(p);
            particles.types.push(1);
            particles.ids.push(particles.ids.len() as MdaTag + 1);
        }
        let mut f = Frame::new(cell, particles);
        f.type_labels = vec!["Al".to_owned()];
        f
    }

    /// Site positions are wrapped into the cell written in the header.
    ///
    /// This is a regression test for a real one: on the 256k-atom relaxed HEA
    /// fixture the fitted region's origin lands near 74 A while the input box is
    /// [2.3, 143.4], and unwrapped site positions came out at 205-212 A. The
    /// file's header still declared the 141 A input box, so all seven vacancies
    /// were 60 A outside the cell they were reported in. The counts were right and
    /// the coordinates were not, which is the worst way for this to break.
    #[test]
    fn site_positions_are_wrapped_into_the_written_cell() {
        // A cell whose origin is nowhere near zero, like the HEA fixture's.
        let mut cell = crate::cell::UnitCell::orthorhombic(10.0, 10.0, 10.0);
        cell.origin = [2.321_323_446_986_792_5; 3];
        // Two atoms, because `site_of_atom` below seats two of them on site 1.
        let mut particles = ParticleData::new(2);
        particles.positions.push([5.0, 5.0, 5.0]);
        particles.positions.push([5.1, 5.0, 5.0]);
        particles.types.push(1);
        particles.types.push(1);
        particles.ids.push(1);
        particles.ids.push(2);
        let base = Frame::new(cell, particles);

        // Sites at indices well outside the box, as an unanchored fitted region
        // produces: roughly one and a half box lengths up.
        let sites: Vec<Coord3> = [0usize, 1, 2]
            .iter()
            .map(|k| [2.321_323_446_986_792_5 + 15.0 + 3.0 * *k as f64; 3])
            .collect();
        let dump = DefectDump {
            cell,
            // One empty site, and a second that is over-occupied, so both the
            // vacancy and the interstitial-site path go through the wrap.
            occupancy: vec![0, 2, 1],
            sites: SiteGrid::Positions(sites),
            site_of_atom: vec![1, 1],
            void_of_site: None,
        };
        let dir = tempdir("wrap");
        let opts = DefectDumpOptions {
            dir: dir.clone(),
            write_empty: true,
            ..Default::default()
        };
        let _ = write_defect_files(&base, &dump, &opts).unwrap();

        let read = |name: &str| -> Vec<Vec<f64>> {
            let text = std::fs::read_to_string(dir.join(name)).unwrap();
            text.lines()
                .skip(2)
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    // Skip the species label; everything after it is numeric.
                    l.split_whitespace()
                        .skip(1)
                        .map(|t| t.parse::<f64>().unwrap())
                        .collect()
                })
                .collect()
        };

        let lo = 2.321_323_446_986_792_5;
        let hi = lo + 10.0;
        for name in ["vacancies.xyz", "interstitials_sites.xyz"] {
            let rows = read(name);
            assert!(!rows.is_empty(), "{name} lost its rows");
            for r in &rows {
                for (d, &v) in r.iter().take(3).enumerate() {
                    assert!(
                        v >= lo - 1e-9 && v < hi + 1e-9,
                        "{name}: coordinate {d} = {v} is outside [{lo}, {hi})"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An assignment with `holes` empty and one site holding `crowd_occ` atoms.
    ///
    /// `sum(occupancy) == natoms` is maintained, which is the invariant that makes
    /// this a *defect* state rather than a different atom count: the atoms pushed
    /// out of the holes are the ones that crowd onto the site. A fixture that
    /// invented crowding without paying for it in vacancies would not exercise the
    /// `k - 1` arithmetic the dump reports.
    fn defects(frame: &Frame, holes: &[usize], crowded: usize, crowd_occ: u32) -> DefectDump {
        let n = frame.natoms();
        let mut occ: Vec<u32> = vec![1; n];
        for &h in holes {
            occ[h] = 0;
        }
        occ[crowded] = crowd_occ;
        let total: usize = occ.iter().map(|&o| o as usize).sum();
        assert_eq!(
            total,
            n,
            "fixture must keep sum(occupancy) == natoms; {} holes, crowd_occ {crowd_occ}",
            holes.len()
        );

        let nonhole: Vec<usize> = (0..n).filter(|&i| occ[i] > 0).collect();
        let mut site_of_atom = vec![-1i64; n];
        let mut a = 0usize;
        for (s, &want) in occ.iter().enumerate() {
            for _ in 0..want {
                if a >= nonhole.len() {
                    break;
                }
                site_of_atom[nonhole[a]] = s as i64;
                a += 1;
            }
        }
        DefectDump {
            cell: frame.cell,
            occupancy: occ,
            sites: SiteGrid::Lattice(lattice_for(frame)),
            site_of_atom,
            void_of_site: None,
        }
    }

    fn lattice_for(frame: &Frame) -> SiteLattice {
        SiteLattice::build(&frame.cell, &Fcc { a: 3.5 }.primitive(), [0.0; 3]).unwrap()
    }

    #[test]
    fn writes_three_files_with_the_right_rows() {
        let frame = frame_with(4, 3.5);
        // 3 sites emptied, their 3 atoms crowding onto site 5 (occupancy 4).
        let d = defects(&frame, &[3, 17, 42], 5, 4);
        let dir = tempdir("three");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(report.vacancies, 3);
        assert_eq!(report.interstitial_sites, 1);
        assert_eq!(
            report.interstitial_atoms, 4,
            "every atom on the crowded site is listed, not one per excess"
        );
        assert_eq!(report.excess_atoms, 3, "the tally counts occupancy - 1");
        assert_eq!(report.files.len(), 3);
        for f in &report.files {
            assert!(f.path.exists(), "{} missing", f.path.display());
            assert!(!f.skipped_empty);
        }
        let names: Vec<&str> = report.files.iter().map(|f| f.name).collect();
        assert_eq!(
            names,
            vec![
                "vacancies.xyz",
                "interstitials.xyz",
                "interstitials_sites.xyz"
            ]
        );
        cleanup(&report.files);
    }

    /// The vacancy file holds *sites*, so its row count is the vacancy count and
    /// its species is the pseudo-element rather than any real one.
    #[test]
    fn vacancy_rows_are_sites_with_the_pseudo_species() {
        let frame = frame_with(4, 3.5);
        let holes = [3usize, 17, 42, 55];
        let d = defects(&frame, &holes, 5, 5);
        let dir = tempdir("vacrows");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();
        let vac = &report.files[0];
        let text = std::fs::read_to_string(&vac.path).unwrap();
        assert_eq!(text.lines().next().unwrap().trim(), "4");
        assert!(text.contains(&format!("{VACANCY_SPECIES} ")), "{text}");
        // Every vacancy row must sit on a real lattice point.
        for line in text.lines().skip(2) {
            let f: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(f[0], VACANCY_SPECIES);
            let p = [
                f[1].parse::<f64>().unwrap(),
                f[2].parse::<f64>().unwrap(),
                f[3].parse::<f64>().unwrap(),
            ];
            assert!(p.iter().all(|v| v.is_finite()));
        }
        cleanup(&report.files);
    }

    /// The interstitial-atom file must preserve the input species, or it cannot be
    /// overlaid on the original scene.
    #[test]
    fn interstitial_atoms_keep_their_input_species() {
        let mut frame = frame_with(4, 3.5);
        frame.particles.types[7] = 2;
        frame.type_labels = vec!["Al".to_owned(), "W".to_owned()];
        // Site 7 is crowded, so atoms 7 and 8 both land on it.
        let d = defects(&frame, &[3], 7, 2);
        let dir = tempdir("interspecies");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&report.files[1].path).unwrap();
        assert!(
            text.contains("W "),
            "crowded site's atom lost its species:\n{text}"
        );
        assert_eq!(text.lines().next().unwrap().trim(), "2");
        cleanup(&report.files);
    }

    /// Void ids must partition the vacancy sites the same way the clusters do.
    #[test]
    fn void_ids_follow_the_clustering() {
        let frame = frame_with(4, 3.5);
        let holes = [3usize, 17, 42, 55];
        let d = defects(&frame, &holes, 5, 5);
        let mut counts = VoidCounts {
            min_sites: 2,
            cutoff: 2.5,
            ..Default::default()
        };
        counts.voids = vec![crate::analysis::void::VoidCluster {
            n_sites: 3,
            sites: vec![3, 17, 42],
            centroid: [0.0; 3],
            equivalent_radius: 1.0,
        }];
        let d = d.with_voids(&counts);
        let ids = d.void_of_site.as_ref().unwrap();
        assert_eq!((ids[3], ids[17], ids[42]), (0, 0, 0));
        assert_eq!(
            ids[55], -1,
            "a vacancy outside every reported void stays -1"
        );

        let dir = tempdir("voidids");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&report.files[0].path).unwrap();
        assert!(
            text.contains("void_id:I:1"),
            "no void_id column:\n{}",
            text.lines().nth(1).unwrap()
        );
        cleanup(&report.files);
    }

    /// Without void detection there is no void_id column, rather than a column of
    /// -1s that a viewer would group as one big void.
    #[test]
    fn no_void_column_when_void_detection_did_not_run() {
        let frame = frame_with(4, 3.5);
        let d = defects(&frame, &[3, 17], 5, 3);
        let dir = tempdir("novoidcol");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();
        let text = std::fs::read_to_string(&report.files[0].path).unwrap();
        assert!(!text.contains("void_id"), "unexpected column:\n{text}");
        cleanup(&report.files);
    }

    /// A mismatch between occupancy and the site count must be refused, not read
    /// past: the neighbouring site's occupancy would place a vacancy on a site that
    /// is perfectly occupied.
    #[test]
    fn a_length_mismatch_is_refused() {
        let frame = frame_with(3, 3.5);
        let mut d = defects(&frame, &[3], 7, 2);
        d.occupancy.pop();
        let dir = tempdir("mismatch");
        let err = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir: dir.clone(),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("occupancy entries"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_files_can_be_skipped() {
        let frame = frame_with(3, 3.5);
        let d = defects(&frame, &[], 0, 1);
        let dir = tempdir("skipempty");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir: dir.clone(),
                write_empty: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(report.vacancies, 0);
        assert!(report.files.iter().all(|f| f.skipped_empty));
        assert!(!report.files.iter().any(|f| f.path.exists()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_zero_defect_run_still_writes_readable_files() {
        let frame = frame_with(3, 3.5);
        let mut occ = vec![1u32; frame.natoms()];
        occ[0] = 2;
        occ[1] = 2;
        let d = DefectDump {
            cell: frame.cell,
            occupancy: occ,
            sites: SiteGrid::Lattice(
                SiteLattice::build(&frame.cell, &Fcc { a: 3.5 }.primitive(), [0.0; 3]).unwrap(),
            ),
            site_of_atom: (0..frame.natoms() as i64).collect(),
            void_of_site: None,
        };
        let dir = tempdir("zerodefect");
        let report = write_defect_files(
            &frame,
            &d,
            &DefectDumpOptions {
                dir,
                ..Default::default()
            },
        )
        .unwrap();
        // 0 vacancies must still produce a parseable file, not a missing one.
        let text = std::fs::read_to_string(&report.files[0].path).unwrap();
        assert_eq!(text.lines().next().unwrap().trim(), "0");
        assert!(text
            .lines()
            .nth(1)
            .unwrap()
            .contains("Properties=species:S:1:pos:R:3"));
        cleanup(&report.files);
    }

    #[test]
    fn the_site_frame_carries_the_reference_cell() {
        let frame = frame_with(3, 3.5);
        let d = defects(&frame, &[3], 7, 2);
        let f = site_frame(&frame, &[3], &d.sites, VACANCY_SPECIES, &d.cell);
        assert_eq!(f.cell.h_matrix(), d.cell.h_matrix());
        assert_eq!(f.natoms(), 1);
        assert_eq!(f.type_labels, vec![VACANCY_SPECIES.to_owned()]);
        assert_eq!(f.particles.types, vec![1]);
    }

    #[test]
    fn interstitial_atoms_ignores_atoms_with_no_site() {
        let frame = frame_with(3, 3.5);
        let mut d = defects(&frame, &[], 0, 1);
        d.site_of_atom[4] = -1;
        d.site_of_atom[5] = -2;
        assert!(d.interstitial_atoms().is_empty());
    }

    // --- small filesystem helpers ----------------------------------------

    fn tempdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rmda-defect-dump-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn cleanup(files: &[DumpFile]) {
        if let Some(dir) = files.first().and_then(|f| f.path.parent()) {
            std::fs::remove_dir_all(dir).ok();
        }
    }
}
