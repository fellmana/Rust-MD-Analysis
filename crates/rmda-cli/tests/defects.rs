//! End-to-end tests for `rmda ws --defects`.
//!
//! The three defect files are the only place a vacancy can exist, so these check
//! the thing a summary scalar cannot: that the *positions* are the real empty
//! sites. A dump that reported the right count at the wrong coordinates would
//! pass every scalar assertion in `ws.rs` and be useless.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;
use std::path::{Path, PathBuf};
use std::process::Command;

const A: f64 = 3.5;

/// One parsed extended-XYZ file: the comment line's columns and the rows.
struct Xyz {
    columns: Vec<String>,
    lattice: Vec<f64>,
    /// The `Origin=` key: the cell origin travels beside a nine-number
    /// `Lattice=`, which is the only form ASE accepts.
    origin: Vec<f64>,
    rows: Vec<Vec<String>>,
}

impl Xyz {
    fn read(path: &Path) -> Self {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut lines = text.lines();
        let n: usize = lines
            .next()
            .expect("empty xyz")
            .trim()
            .parse()
            .expect("natoms line");
        let comment = lines.next().expect("no comment line").to_owned();
        let rows: Vec<Vec<String>> = lines
            .map(|l| l.split_whitespace().map(str::to_owned).collect())
            .collect();
        assert_eq!(rows.len(), n, "natoms line disagrees with rows in {path:?}");

        let lattice = comment
            .split("Lattice=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("no Lattice")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let origin = comment
            .split("Origin=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .expect("no Origin")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();

        // `Properties=species:S:1:pos:R:3:ws_site:I:1` is (name, kind, count)
        // triples. Flatten to one name per component so a lookup can say "pos:R:2".
        let spec = comment
            .split("Properties=")
            .nth(1)
            .expect("no Properties")
            .split_whitespace()
            .next()
            .expect("empty Properties");
        let parts: Vec<&str> = spec.split(':').collect();
        let mut columns = Vec::new();
        let mut i = 0;
        while i + 3 <= parts.len() {
            let (name, kind, count) = (
                parts[i],
                parts[i + 1],
                parts[i + 2].parse::<usize>().unwrap(),
            );
            for c in 0..count {
                if c == 0 {
                    columns.push(format!("{name}:{kind}"));
                } else {
                    columns.push(format!("{name}:{kind}:{c}"));
                }
            }
            i += 3;
        }
        Self {
            columns,
            lattice,
            origin,
            rows,
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn has_column(&self, name: &str) -> bool {
        self.columns
            .iter()
            .any(|c| c == name || c.starts_with(&format!("{name}:")))
    }

    /// Value of column `col` on row `i`.
    ///
    /// `col` is a base name (`ws_site`) or a component (`pos:R:2`); a scalar
    /// column resolves to its single component either way.
    fn col(&self, i: usize, col: &str) -> f64 {
        let at = self
            .columns
            .iter()
            .position(|c| c == col || c.starts_with(&format!("{col}:")))
            .unwrap_or_else(|| panic!("no column {col} in {:?}", self.columns));
        self.rows[i][at].parse().unwrap()
    }

    fn position(&self, i: usize) -> [f64; 3] {
        [
            self.col(i, "pos:R"),
            self.col(i, "pos:R:1"),
            self.col(i, "pos:R:2"),
        ]
    }

    fn species(&self, i: usize) -> &str {
        &self.rows[i][0]
    }

    /// Shortest periodic distance from row `i` to any row in `other`.
    ///
    /// Through fractional coordinates, not by subtracting whole box lengths per
    /// axis: the lattice is a 3x3 matrix here and its rows are the cell edges, so
    /// `lattice[3]` is a *component* of the second edge and not a box width at all.
    /// Taking it as one divides by zero and every distance comes back `inf`.
    /// Shortest periodic distance from row `i` to any row in `other`.
    ///
    /// Through fractional coordinates rather than by subtracting a whole box
    /// length per axis: `lattice` is a 3x3 matrix whose rows are the cell edges,
    /// so `lattice[3]` is a *component* of the second edge, not a box width.
    /// Treating it as one divides by zero and every distance comes back `inf`.
    fn min_distance_to(&self, i: usize, other: &Xyz) -> f64 {
        let p = self.position(i);
        let h = &other.lattice;
        let inv = inverse_transposed(h);
        other
            .rows
            .iter()
            .map(|r| {
                let q = [
                    r[1].parse::<f64>().unwrap(),
                    r[2].parse::<f64>().unwrap(),
                    r[3].parse::<f64>().unwrap(),
                ];
                // Fractional separation, wrapped to the minimum image.
                let mut f = [0.0f64; 3];
                for (k, slot) in f.iter_mut().enumerate() {
                    let v = (0..3).map(|m| inv[m][k] * (p[m] - q[m])).sum::<f64>();
                    *slot = v - v.round();
                }
                // Back to Cartesian through the cell.
                let d: Vec<f64> = (0..3)
                    .map(|m| (0..3).map(|k| h[m * 3 + k] * f[k]).sum::<f64>())
                    .collect();
                d.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// `(h^-1)^T` for a row-major 3x3, i.e. fractional = this * cartesian.
///
/// Matches the library convention: `lamda2x` is `h^T`, so the inverse applied to
/// a Cartesian vector is the transpose of `h^-1`.
fn inverse_transposed(h: &[f64]) -> [[f64; 3]; 3] {
    let det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
        + h[2] * (h[3] * h[7] - h[4] * h[6]);
    assert!(det.abs() > 1e-12, "degenerate lattice {h:?}");
    [
        [
            (h[4] * h[8] - h[5] * h[7]) / det,
            (h[5] * h[6] - h[3] * h[8]) / det,
            (h[3] * h[7] - h[4] * h[6]) / det,
        ],
        [
            (h[2] * h[7] - h[1] * h[8]) / det,
            (h[0] * h[8] - h[2] * h[6]) / det,
            (h[1] * h[6] - h[0] * h[7]) / det,
        ],
        [
            (h[1] * h[5] - h[2] * h[4]) / det,
            (h[2] * h[3] - h[0] * h[5]) / det,
            (h[0] * h[4] - h[1] * h[3]) / det,
        ],
    ]
}

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rmda-cli-defects-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A perfect FCC block, optionally with sites removed and/or a crowded site.
fn fixture(name: &str, cells: usize, victims: &[usize], crowd: Option<(usize, f64)>) -> PathBuf {
    fixture_at_origin(name, cells, victims, crowd, [0.0, 0.0, 0.0])
}

/// [`fixture`] with the box placed away from the origin, the way a LAMMPS
/// `data` file reports `xlo`.
fn fixture_at_origin(
    name: &str,
    cells: usize,
    victims: &[usize],
    crowd: Option<(usize, f64)>,
    origin: [f64; 3],
) -> PathBuf {
    let len = cells as f64 * A;
    let mut cell = UnitCell::orthorhombic(len, len, len);
    cell.origin = origin;
    let sites: Vec<[f64; 3]> = SiteLattice::build(&cell, &Fcc { a: A }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .map(|p| [p[0] + origin[0], p[1] + origin[1], p[2] + origin[2]])
        .collect();

    // `natoms` must equal what is actually pushed, or the writer walks off the end.
    let n_atoms = sites.len() - victims.len() + usize::from(crowd.is_some());
    let mut pd = ParticleData::new(n_atoms);
    for (i, p) in sites.iter().enumerate() {
        if victims.contains(&i) {
            continue;
        }
        match crowd {
            // Split one site's atom into a symmetric pair, so both are nearest to
            // that same site and the site ends up holding two.
            Some((c, d)) if i == c => {
                pd.positions.push([p[0] + d, p[1], p[2]]);
                pd.ids.push(pd.ids.len() as i64 + 1);
                pd.types.push(1);
                pd.positions.push([p[0] - d, p[1], p[2]]);
                pd.ids.push(pd.ids.len() as i64 + 1);
                pd.types.push(2);
            }
            _ => {
                pd.positions.push(*p);
                pd.ids.push(i as i64 + 1);
                pd.types.push(1);
            }
        }
    }
    let mut frame = Frame::new(cell, pd);
    frame.type_labels = vec!["Al".to_owned(), "W".to_owned()];

    let path = std::env::temp_dir()
        .join("rmda-cli-defect-fixtures")
        .join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

fn rmda(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

fn scalar(stdout: &str, name: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no {name} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn the_three_files_appear_and_hold_the_defects() {
    let src = fixture("mixed.extxyz", 8, &[5, 61, 130], Some((400, 0.3)));
    let dir = tempdir("three-files");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--void-min-sites",
        "2",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "ws --defects failed: {err}");

    for name in [
        "vacancies.xyz",
        "interstitials.xyz",
        "interstitials_sites.xyz",
    ] {
        assert!(dir.join(name).exists(), "{name} not written");
    }

    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    let inter = Xyz::read(&dir.join("interstitials.xyz"));
    let isites = Xyz::read(&dir.join("interstitials_sites.xyz"));

    assert_eq!(vac.len(), scalar(&out, "ws_vacancies") as usize);
    assert_eq!(inter.len(), 2, "both atoms on the crowded site are listed");
    assert_eq!(isites.len(), scalar(&out, "ws_interstitial_sites") as usize);

    // The pseudo-species is what makes these selectable in a viewer.
    for i in 0..vac.len() {
        assert_eq!(vac.species(i), "VAC", "vacancy row {i} species");
    }
    for i in 0..isites.len() {
        assert_eq!(
            isites.species(i),
            "INT",
            "interstitial site row {i} species"
        );
    }

    // The interstitial atoms must keep their own species, or the file cannot be
    // overlaid on the original scene.
    assert_eq!(inter.col(0, "site_occupancy"), 2.0);
    assert_eq!(inter.col(1, "site_occupancy"), 2.0);
    let species: Vec<&str> = (0..inter.len()).map(|i| inter.species(i)).collect();
    assert!(
        species.contains(&"W"),
        "partner atom lost its species: {species:?}"
    );

    // One crowded site means one excess atom, reported as such.
    assert_eq!(scalar(&out, "ws_interstitials"), 1.0, "{out}");
    assert!(
        out.contains("1 interstitial site(s) holding 2 atom(s) (1 excess)"),
        "the summary must distinguish listed atoms from excess ones:\n{out}"
    );
}

#[test]
fn vacancy_positions_are_the_real_empty_lattice_sites() {
    let victims = [5usize, 61, 130, 400, 777];
    let src = fixture("holes.extxyz", 8, &victims, None);
    let dir = tempdir("vac-positions");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    assert_eq!(scalar(&out, "ws_vacancies"), victims.len() as f64, "{out}");

    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    let atoms = Xyz::read(&src);
    assert_eq!(vac.len(), victims.len());

    // A vacancy is a lattice point whose whole first shell is still occupied, so
    // every vacancy must have nearest-neighbour distances equal to a/sqrt(2).
    let nn = A / 2f64.sqrt();
    for i in 0..vac.len() {
        let d = vac.min_distance_to(i, &atoms);
        assert!(
            (d - nn).abs() < 1e-3,
            "vacancy {i} is {d:.4} A from the nearest atom, not the {nn:.4} A \
             first shell of an intact lattice site"
        );
    }
}

#[test]
fn vacancy_rows_are_distinct_sites() {
    let src = fixture("distinct.extxyz", 6, &[1, 2, 3, 40, 41], None);
    let dir = tempdir("vac-distinct");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    assert_eq!(vac.len(), 5, "{out}");

    // Duplicate positions would mean the site index and the site position came
    // from different enumerations -- the exact bug the position/ displacement
    // cross-check guards against in the library.
    let mut seen = std::collections::HashSet::new();
    for i in 0..vac.len() {
        let p = vac.position(i);
        let key: Vec<i64> = p.iter().map(|v| (v * 1e6).round() as i64).collect();
        assert!(seen.insert(key), "duplicate vacancy at {p:?}");
    }
}

#[test]
fn void_ids_group_the_vacancies_into_craters() {
    // Site indices are cell-major, basis-minor, with the x cell edge varying
    // fastest, so the four sites of one cell are consecutive and one cell step
    // is four indices. [0, 1, 2] and [8, 9, 10] are therefore two groups of
    // three genuinely adjacent empties; arbitrary indices like [5, 61, 130]
    // are tens of angstroms apart and cluster as six singletons.
    let src = fixture("crater.extxyz", 8, &[0, 1, 2, 8, 9, 10], None);
    let dir = tempdir("void-ids");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--void-min-sites",
        "2",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");

    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    assert!(
        vac.has_column("void_id"),
        "no void_id column:\n{:?}",
        vac.columns
    );
    let ids: Vec<f64> = (0..vac.len()).map(|i| vac.col(i, "void_id")).collect();
    // One crater of three and one of three, so two ids, each seen three times.
    let mut counts = std::collections::BTreeMap::new();
    for id in &ids {
        *counts.entry(*id as i64).or_insert(0usize) += 1;
    }
    assert_eq!(counts.len(), 2, "expected two craters, got {counts:?}");
    for (id, n) in &counts {
        assert_eq!(*n, 3, "crater {id} has {n} sites");
    }
    assert_eq!(scalar(&out, "void_count"), 2.0, "{out}");
}

#[test]
fn there_is_no_void_id_column_without_void_detection() {
    // `--self` always runs void clustering (default --void-min-sites 1), so the
    // reference engine is the one that must leave `void_id` out: it has no
    // void pass to attribute craters to.
    let src = fixture("novoidcol.extxyz", 6, &[5, 61], None);
    let reference = fixture("novoidcol-ref.extxyz", 6, &[], None);
    let dir = tempdir("no-void-col");
    let (out, err, ok) = rmda(&[
        "ws",
        "--reference",
        reference.to_str().unwrap(),
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    // A column of -1s would invite a viewer to group every vacancy as one void.
    assert!(
        !vac.has_column("void_id"),
        "unexpected column {:?}",
        vac.columns
    );
}

#[test]
fn a_perfect_run_writes_readable_empty_files() {
    let src = fixture("perfect-defects.extxyz", 6, &[], None);
    let dir = tempdir("perfect");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    assert_eq!(scalar(&out, "ws_vacancies"), 0.0, "{out}");
    // A fixed set of three filenames is easier to script against than one whose
    // contents decide which files exist, so all three are written at 0 rows.
    for name in [
        "vacancies.xyz",
        "interstitials.xyz",
        "interstitials_sites.xyz",
    ] {
        let f = Xyz::read(&dir.join(name));
        assert_eq!(f.len(), 0, "{name} should be empty");
        assert!(f.has_column("pos:R"), "{name} has no usable header");
    }
}

#[test]
fn the_reference_engine_writes_the_same_three_files() {
    let refp = fixture("ref-clean.extxyz", 8, &[], None);
    let src = fixture("ref-defects.extxyz", 8, &[5, 61], Some((400, 0.3)));
    let dir = tempdir("reference-engine");
    let (out, err, ok) = rmda(&[
        "ws",
        "--reference",
        refp.to_str().unwrap(),
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");

    let vac = Xyz::read(&dir.join("vacancies.xyz"));
    assert_eq!(vac.len(), 2, "{out}");
    // The reference frame's atoms *are* the sites, so a vacancy sits exactly on a
    // reference atom position with no fit in between.
    let reference = Xyz::read(&refp);
    for i in 0..vac.len() {
        let d = vac.min_distance_to(i, &reference);
        assert!(
            d < 1e-6,
            "vacancy {i} is {d} A from the nearest reference site"
        );
    }
    let inter = Xyz::read(&dir.join("interstitials.xyz"));
    assert_eq!(inter.len(), 2, "{out}");
    assert_eq!(Xyz::read(&dir.join("interstitials_sites.xyz")).len(), 1);
}

#[test]
fn the_per_atom_flag_marks_exactly_the_interstitial_atoms() {
    let src = fixture("flag.extxyz", 8, &[], Some((400, 0.3)));
    let dir = tempdir("flag");
    let tagged = dir.join("tagged.extxyz");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "-o",
        tagged.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    assert_eq!(scalar(&out, "ws_interstitials"), 1.0, "{out}");

    let all = Xyz::read(&tagged);
    assert!(all.has_column("ws_interstitial"), "{:?}", all.columns);
    // ws_site must survive alongside the new flag: it is what tells a viewer which
    // lattice site each atom claimed.
    assert!(all.has_column("ws_site"), "{:?}", all.columns);
    let flagged: Vec<usize> = (0..all.len())
        .filter(|&i| all.col(i, "ws_interstitial") == 1.0)
        .collect();
    assert_eq!(
        flagged.len(),
        2,
        "expected the two atoms on the crowded site"
    );

    // The two flagged atoms must be exactly the pair straddling one site, i.e.
    // they share a `ws_site`, which is what makes them a crowd rather than two
    // unrelated off-lattice atoms.
    let sites: Vec<f64> = flagged.iter().map(|&i| all.col(i, "ws_site")).collect();
    assert_eq!(
        sites[0], sites[1],
        "flagged atoms sit on different sites: {sites:?}"
    );
}

#[test]
fn interstitial_atoms_and_sites_agree_with_the_counts() {
    let src = fixture("agree.extxyz", 8, &[5], Some((400, 0.3)));
    let dir = tempdir("agree");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    let inter = Xyz::read(&dir.join("interstitials.xyz"));
    let isites = Xyz::read(&dir.join("interstitials_sites.xyz"));

    // Every listed atom must name a site that the site file lists as crowded, and
    // the two occupancies must be the same number.
    let site_index_at: Vec<f64> = (0..inter.len())
        .map(|i| inter.col(i, "site_index"))
        .collect();
    let occ_at: Vec<f64> = (0..inter.len())
        .map(|i| inter.col(i, "site_occupancy"))
        .collect();
    let site_index_st: Vec<f64> = (0..isites.len())
        .map(|i| isites.col(i, "site_index"))
        .collect();
    let occ_st: Vec<f64> = (0..isites.len())
        .map(|i| isites.col(i, "site_occupancy"))
        .collect();

    // One crowded site can be named by several atoms, so compare the *sets*.
    let mut at_set: Vec<f64> = site_index_at.clone();
    at_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    at_set.dedup();
    let mut st_set = site_index_st.clone();
    st_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    st_set.dedup();
    assert_eq!(at_set, st_set, "the two files name different sites");
    assert_eq!(at_set.len(), 1, "expected exactly one crowded site");
    for &o in &occ_at {
        assert_eq!(
            o, 2.0,
            "every listed atom should report a site occupancy of 2"
        );
    }
    for &o in &occ_st {
        assert_eq!(o, 2.0, "the crowded site should hold two atoms");
    }
    assert_eq!(
        at_set.len() * 2,
        occ_at.len(),
        "the two counts must line up"
    );

    // The interstitial site must sit between the two atoms it holds.
    let sp = isites.position(0);
    for i in 0..inter.len() {
        let p = inter.position(i);
        let d = (0..3).map(|k| p[k] - sp[k]).collect::<Vec<f64>>();
        let d = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(d < 0.31, "atom {i} is {d:.4} A from its own site");
    }
}

#[test]
fn a_strained_frame_keeps_atoms_and_sites_in_their_own_cells() {
    // Under --map to-reference the sites live in the reference cell and the atoms
    // in the input one. Writing reference-frame positions under the input cell
    // would be a file that looks fine and plots in the wrong place.
    let refp = fixture("strain-ref.extxyz", 6, &[], None);
    let clean = Xyz::read(&refp);
    let s = 1.01;
    let src = fixture("strained.extxyz", 6, &[], Some((20, 0.3)));
    // Rewrite the strained fixture with a scaled cell and positions.
    {
        let text = std::fs::read_to_string(&src).unwrap();
        let mut out = Vec::new();
        for line in text.lines() {
            if line.starts_with("Lattice=") {
                let v: Vec<f64> = line
                    .split("Lattice=\"")
                    .nth(1)
                    .unwrap()
                    .split('"')
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse::<f64>().unwrap() * s)
                    .collect();
                out.push(format!(
                    "Lattice=\"{}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3",
                    v.iter()
                        .map(|x| format!("{x:.10}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            } else if line.split_whitespace().count() == 4 {
                let f: Vec<&str> = line.split_whitespace().collect();
                out.push(format!(
                    "{} {:.10} {:.10} {:.10}",
                    f[0],
                    f[1].parse::<f64>().unwrap() * s,
                    f[2].parse::<f64>().unwrap() * s,
                    f[3].parse::<f64>().unwrap() * s
                ));
            } else {
                out.push(line.to_owned());
            }
        }
        std::fs::write(&src, out.join("\n") + "\n").unwrap();
    }

    let dir = tempdir("strained-cells");
    let (out, err, ok) = rmda(&[
        "ws",
        "--reference",
        refp.to_str().unwrap(),
        "--map",
        "to-reference",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    let inter = Xyz::read(&dir.join("interstitials.xyz"));
    let isites = Xyz::read(&dir.join("interstitials_sites.xyz"));

    let atom_len = inter.lattice[0];
    let site_len = isites.lattice[0];
    assert!(
        (atom_len - site_len).abs() > 0.1,
        "the two files should carry different cells, got {atom_len} and {site_len}"
    );
    assert!(
        (site_len - clean.lattice[0]).abs() < 1e-6,
        "the site file must carry the reference cell, got {site_len} not {}",
        clean.lattice[0]
    );
    assert!(
        (atom_len - clean.lattice[0] * s).abs() < 1e-3,
        "the atom file must carry the input cell, got {atom_len}"
    );
}

#[test]
fn a_missing_directory_is_created() {
    let src = fixture("mkdir.extxyz", 6, &[5], None);
    let dir = std::env::temp_dir().join("rmda-cli-defects-nested/deeper/still");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("rmda-cli-defects-nested"));
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    assert!(dir.join("vacancies.xyz").exists(), "{out}");
    std::fs::remove_dir_all(std::env::temp_dir().join("rmda-cli-defects-nested")).ok();
}

#[test]
fn the_defect_files_carry_the_analysis_cell() {
    let src = fixture("cell.extxyz", 6, &[5], None);
    let dir = tempdir("cell");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let src_x = Xyz::read(&src);
    for name in [
        "vacancies.xyz",
        "interstitials.xyz",
        "interstitials_sites.xyz",
    ] {
        let f = Xyz::read(&dir.join(name));
        assert_eq!(
            f.lattice[0], src_x.lattice[0],
            "{name} carries the wrong cell, so it will not overlay on the input"
        );
    }
}

/// The default has to reproduce the input's box, origin included.
#[test]
fn the_defect_files_reproduce_an_offset_input_box_by_default() {
    let origin = [-0.25, 1.5, -3.0];
    let src = fixture_at_origin("offset.extxyz", 6, &[5], None, origin);
    let dir = tempdir("offset");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    for name in [
        "vacancies.xyz",
        "interstitials.xyz",
        "interstitials_sites.xyz",
    ] {
        let f = Xyz::read(&dir.join(name));
        assert_eq!(
            &f.origin[..],
            &origin,
            "{name} lost the input box origin, so it will not overlay on the input"
        );
    }
}

/// `--xyz-origin zero` moves the box to the origin, and must move the atoms with
/// it: the atom's offset inside its own cell has to be identical either way.
#[test]
fn the_zero_origin_flag_moves_the_box_and_the_atoms_together() {
    let origin = [-0.25, 1.5, -3.0];
    let src = fixture_at_origin("zero.extxyz", 6, &[5], None, origin);
    let dir = tempdir("zero");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--xyz-origin",
        "zero",
        "--defects",
        dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    let f = Xyz::read(&dir.join("vacancies.xyz"));
    assert_eq!(
        &f.origin[..],
        &[0.0, 0.0, 0.0],
        "zero mode still declares an origin"
    );

    // Same cell, and every atom at the same place relative to it.
    let native_dir = tempdir("zeronative");
    let (out, err, ok) = rmda(&[
        "ws",
        "--self",
        "--defects",
        native_dir.to_str().unwrap(),
        src.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let nat = Xyz::read(&native_dir.join("vacancies.xyz"));
    assert_eq!(
        nat.lattice[0..9],
        f.lattice[0..9],
        "zero mode changed the cell"
    );

    assert_eq!(nat.rows.len(), f.rows.len(), "row count changed");
    for (a, b) in nat.rows.iter().zip(&f.rows) {
        // species + pos(3) + type, then the per-row properties.
        for k in 1..4 {
            let shift = b[k].parse::<f64>().unwrap() - a[k].parse::<f64>().unwrap();
            assert!(
                (shift + origin[k - 1]).abs() < 1e-6,
                "coordinate {k} shifted by {shift}, expected {}",
                -origin[k - 1]
            );
        }
        assert_eq!(a[0], b[0], "species changed");
    }
}
