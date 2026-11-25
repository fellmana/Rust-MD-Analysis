//! End-to-end tests for the `rmda cluster` subcommand.
//!
//! The library tests in `analysis::cluster` cover labelling, `min_size` and both
//! bond modes against frames whose answer is known by hand. What only this level
//! can check is the wiring: that `--cutoff` and `--knear` reach the analysis as
//! configured, that neither is silently defaulted, that the per-cluster property
//! stays out of the per-atom table, and that the reported scalars agree with the
//! labels actually written to disk.
//!
//! The fixtures are built here rather than stored, for the same reason the grain
//! fixtures are: the assertion is about a *known* answer. A real input could only
//! be checked against this implementation's own output, which proves nothing.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use std::path::PathBuf;
use std::process::Command;

fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join("rmda-cli-cluster-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

fn scalar(stdout: &str, key: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{key} is not a number: {e}"))
}

fn write_fixture(name: &str, cell: UnitCell, pos: &[[f64; 3]]) -> PathBuf {
    let mut pd = ParticleData::with_capacity(pos.len());
    for (i, p) in pos.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, 1);
    }
    let frame = Frame::new(cell, pd);
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

/// Three dimers along x with gaps of 9 A, 14 A and 24 A (round the boundary).
fn dimers(name: &str) -> PathBuf {
    write_fixture(
        name,
        UnitCell::orthorhombic(50.0, 50.0, 50.0),
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [25.0, 0.0, 0.0],
            [26.0, 0.0, 0.0],
        ],
    )
}

/// A 4x4x4 simple-cubic block at spacing 2.5 A: one cluster, no boundary.
fn sc_block(name: &str) -> PathBuf {
    let mut pos = Vec::new();
    for iz in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                pos.push([ix as f64 * 2.5, iy as f64 * 2.5, iz as f64 * 2.5]);
            }
        }
    }
    write_fixture(name, UnitCell::orthorhombic(10.0, 10.0, 10.0), &pos)
}

/// Column index of `name` in a written `.xyz`, from the `Properties=` header.
///
/// Written XYZ is: atom count, then a header comment carrying
/// `Properties=name:type:count:name:type:count:...`, then one line per atom.
///
/// The property list is **colon**-delimited, not whitespace-delimited, so
/// splitting it on spaces yields the entire list as one token and a whitespace
/// search silently finds nothing. This walks the `name/type/count` triples and
/// accumulates the widths instead, since `pos` is three columns wide and the
/// others are one.
fn column_index(text: &str, name: &str) -> usize {
    let header = text
        .lines()
        .nth(1)
        .expect("no header line in the written xyz");
    let props = header
        .split("Properties=")
        .nth(1)
        .expect("no Properties= in the header")
        .trim();
    let fields: Vec<&str> = props.split(':').collect();
    assert_eq!(
        fields.len() % 3,
        0,
        "Properties= is not a whole number of name/type/count triples: {props}"
    );
    for (t, triple) in fields.chunks(3).enumerate() {
        if triple[0] == name {
            return fields[..t * 3]
                .chunks(3)
                .map(|c| c[2].parse::<usize>().unwrap_or(1))
                .sum();
        }
    }
    panic!("no {name} column in: {props}");
}

/// Column of `name` for every atom line in a written `.xyz`.
fn column(path: &PathBuf, name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    let idx = column_index(&text, name);
    text.lines()
        .skip(2)
        .filter_map(|l| l.split_whitespace().nth(idx))
        .map(str::to_owned)
        .collect()
}

/// The `Cluster` column of a written `.xyz`, as integers.
fn cluster_ids(path: &PathBuf) -> Vec<i32> {
    column(path, "Cluster")
        .iter()
        .map(|v| v.parse().expect("Cluster is not an integer"))
        .collect()
}

#[test]
fn a_cutoff_that_only_reaches_each_dimer_gives_three_clusters() {
    let path = dimers("three.extxyz");
    let (out, err, ok) = rmda(&["cluster", "--cutoff", "2.0", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 3.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest"), 2.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest_fraction"), 2.0 / 6.0, "{out}");
    assert_eq!(scalar(&out, "cluster_mean_size"), 2.0, "{out}");
    assert!(err.contains("mode=cutoff(2)"), "{err}");
}

#[test]
fn a_wider_cutoff_joins_the_first_gap_only() {
    // 9.5 A clears the 9 A gap and neither of the others, so the answer is two
    // clusters and not one and not three.
    let path = dimers("joined.extxyz");
    let (out, err, ok) = rmda(&["cluster", "--cutoff", "9.5", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 2.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest"), 4.0, "{out}");
}

#[test]
fn the_written_labels_agree_with_the_reported_counts() {
    let path = dimers("agree.extxyz");
    let out_path = tempdir().join("agree.out.xyz");
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "9.5",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");

    let labels = cluster_ids(&out_path);
    assert_eq!(labels.len(), 6, "one label per atom");
    // OVITO's ids: contiguous from 0, the largest is count - 1.
    let mut ids: Vec<i32> = labels.clone();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids, vec![0, 1], "{labels:?}");
    assert_eq!(ids.len() as f64, scalar(&out, "cluster_count"));

    // Sizes recomputed from the labels must match what was reported.
    let mut sizes = vec![0usize; ids.len()];
    for &l in &labels {
        sizes[l as usize] += 1;
    }
    assert_eq!(
        sizes.iter().max().copied().unwrap() as f64,
        scalar(&out, "cluster_largest")
    );
    assert_eq!(sizes.iter().sum::<usize>() as f64, 6.0);
}

#[test]
fn ids_follow_atom_order_not_size_or_position() {
    // The 4-atom cluster comes first in the file even though it is the larger one,
    // so a size-ordered labelling would give the opposite answer.
    let path = write_fixture(
        "order.extxyz",
        UnitCell::orthorhombic(50.0, 50.0, 50.0),
        &[
            // The *smaller* cluster comes first in the file: a size-ordered
            // labelling would call this one cluster 1.
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [31.0, 0.0, 0.0],
            [32.0, 0.0, 0.0],
        ],
    );
    let out_path = tempdir().join("order.out.xyz");
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "2.0",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 2.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest"), 3.0, "{out}");
    let labels = cluster_ids(&out_path);
    assert_eq!(labels, vec![0, 0, 1, 1, 1], "{labels:?}");
    // Cluster 0 is the smaller one, which is the whole assertion: largest-first
    // would give [1, 1, 0, 0, 0].
    assert_eq!(labels[0], 0);
}

#[test]
fn a_uniform_block_is_one_cluster_in_both_modes() {
    // The cross-check that `KNear` is the same connectivity rule read differently:
    // on a lattice where the k nearest are exactly the ones inside a cutoff just
    // past the k-th shell, both must give one cluster of 64.
    let path = sc_block("block.extxyz");
    let (cut, cerr, cok) = rmda(&["cluster", "--cutoff", "2.6", path.to_str().unwrap()]);
    assert!(cok, "{cut}{cerr}");
    let (kn, knerr, knok) = rmda(&["cluster", "--knear", "6", path.to_str().unwrap()]);
    assert!(knok, "{kn}{knerr}");
    assert_eq!(scalar(&cut, "cluster_count"), 1.0, "{cut}");
    assert_eq!(scalar(&kn, "cluster_count"), 1.0, "{kn}");
    assert_eq!(scalar(&cut, "cluster_largest"), 64.0);
    assert_eq!(scalar(&kn, "cluster_largest"), 64.0);
    assert!(cerr.contains("mode=cutoff(2.6)"), "{cerr}");
    assert!(knerr.contains("mode=knear(6)"), "{knerr}");
}

#[test]
fn knear_reports_no_cutoff_rather_than_a_zero_one() {
    // `cluster_mode_cutoff` is NaN in this mode, not 0.0: a zero would read as a
    // configured radius of zero and invite a downstream division by it.
    let path = dimers("knear_nan.extxyz");
    let (out, err, ok) = rmda(&["cluster", "--knear", "1", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert!(
        scalar(&out, "cluster_mode_cutoff").is_nan(),
        "expected NaN:\n{out}"
    );
}

#[test]
fn min_size_filters_particle_cluster_but_not_cluster() {
    // Three clusters of two, min-size 3, so everything is filtered. The point is
    // that `cluster_count` still reports 3: the map is unchanged and only
    // `ParticleCluster` is filtered.
    let path = dimers("min_size.extxyz");
    let out_path = tempdir().join("min_size.out.xyz");
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "2.0",
        "--min-size",
        "3",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 3.0, "{out}");
    assert_eq!(scalar(&out, "cluster_min_size"), 3.0, "{out}");
    assert_eq!(scalar(&out, "cluster_count_retained"), 0.0, "{out}");
    assert_eq!(scalar(&out, "cluster_atoms_retained"), 0.0, "{out}");

    // Every atom is in too small a cluster, so the filtered view is -1.
    let vals: Vec<i32> = column(&out_path, "ParticleCluster")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(vals, vec![-1; 6]);
}

#[test]
fn min_size_keeps_the_clusters_that_qualify() {
    // A 5-atom chain plus two dimers: the filter has to keep something and drop
    // something, and `cluster_atoms_retained` has to count the *kept* atoms
    // rather than the number of non-zero ids (cluster 0 is a valid id, so that
    // count would come out zero whenever the largest cluster is id 0).
    let path = write_fixture(
        "min_size_keep.extxyz",
        UnitCell::orthorhombic(50.0, 50.0, 50.0),
        &[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 4.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [31.0, 0.0, 0.0],
        ],
    );
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "1.5",
        "--min-size",
        "3",
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 3.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest"), 5.0, "{out}");
    assert_eq!(scalar(&out, "cluster_count_retained"), 1.0, "{out}");
    assert_eq!(scalar(&out, "cluster_atoms_retained"), 5.0, "{out}");
}

#[test]
fn neither_mode_is_defaulted() {
    // A cutoff that is too large percolates the frame into one cluster and one
    // that is too small splits it into one per atom. Both look like answers, so
    // the command refuses rather than picking one.
    let path = dimers("no_mode.extxyz");
    let (_, err, ok) = rmda(&["cluster", path.to_str().unwrap()]);
    assert!(!ok, "cluster ran with no bond mode");
    assert!(err.contains("--cutoff") && err.contains("--knear"), "{err}");
}

#[test]
fn cutoff_and_knear_cannot_both_be_given() {
    // Contradictory requests must not resolve silently to one of them.
    let path = dimers("both.extxyz");
    let (_, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "2.0",
        "--knear",
        "4",
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "both modes were accepted");
    assert!(err.contains("cannot") || err.contains("conflict"), "{err}");
}

#[test]
fn an_unusable_cutoff_is_refused() {
    // Checked at the boundary: a zero, negative or NaN cutoff makes the bond
    // comparison answer "never" or "always" while looking like a radius.
    let path = dimers("bad_cutoff.extxyz");
    // `=` form: clap reads a bare `-1` as an unknown flag and never reaches the
    // check, so the argument would be refused for the wrong reason.
    for bad in ["0", "-1", "nan", "inf"] {
        let (_, err, ok) = rmda(&[
            "cluster",
            path.to_str().unwrap(),
            &format!("--cutoff={bad}"),
        ]);
        assert!(!ok, "--cutoff={bad} was accepted");
        assert!(err.contains("cutoff"), "{err}");
    }
}

#[test]
fn an_unusable_knear_is_refused() {
    let path = dimers("bad_knear.extxyz");
    for bad in ["0", "6", "99"] {
        let (_, err, ok) = rmda(&["cluster", path.to_str().unwrap(), &format!("--knear={bad}")]);
        assert!(!ok, "--knear={bad} was accepted on a 6-atom frame");
        assert!(err.contains("knear"), "{err}");
    }
}

#[test]
fn the_size_histogram_accounts_for_every_cluster() {
    let path = dimers("hist.extxyz");
    let (out, err, ok) = rmda(&["cluster", "--cutoff", "9.5", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    let body = out
        .split_once("cluster_size_histogram:\n")
        .expect("no histogram")
        .1;
    let total: f64 = body
        .lines()
        .filter_map(|l| l.strip_prefix("  size="))
        .filter_map(|l| l.split_whitespace().nth(1))
        .filter_map(|l| l.strip_prefix("clusters="))
        .filter_map(|v| v.parse::<f64>().ok())
        .sum();
    assert_eq!(total, 2.0, "two clusters:\n{body}");
    assert!(body.contains("size=4 clusters=1"), "{body}");
    assert!(body.contains("size=2 clusters=1"), "{body}");
}

#[test]
fn a_cluster_wraps_the_periodic_boundary() {
    // Two atoms 1 A apart through the boundary and 29 A apart directly. Under
    // minimum image they are bonded; a bond test that ignored the boundary would
    // report two clusters.
    let path = write_fixture(
        "wrap.extxyz",
        UnitCell::orthorhombic(30.0, 30.0, 30.0),
        &[[0.5, 15.0, 15.0], [29.5, 15.0, 15.0]],
    );
    let (out, err, ok) = rmda(&["cluster", "--cutoff", "2.0", path.to_str().unwrap()]);
    assert!(ok, "{out}{err}");
    assert_eq!(scalar(&out, "cluster_count"), 1.0, "{out}");
    assert_eq!(scalar(&out, "cluster_largest"), 2.0, "{out}");
}

#[test]
fn the_run_is_reproducible() {
    // Same input twice must give byte-identical scalars: cluster ids are claimed
    // to depend on atom order alone, and a run that varied would break every
    // stored comparison downstream.
    let path = dimers("repro.extxyz");
    let first = rmda(&["cluster", "--cutoff", "9.5", path.to_str().unwrap()]);
    let second = rmda(&["cluster", "--cutoff", "9.5", path.to_str().unwrap()]);
    assert!(first.2 && second.2);
    assert_eq!(first.0, second.0);
}

#[test]
fn cluster_size_is_not_written_per_atom() {
    // `ClusterSize` has one entry per cluster. If it reached the atom table it
    // would be silently wrong for every frame with more than one cluster, and it
    // is not detectable by reading a single row.
    let path = dimers("size_prop.extxyz");
    let out_path = tempdir().join("size_prop.out.xyz");
    let (_, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "2.0",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    let text = std::fs::read_to_string(&out_path).unwrap();
    let header = text.lines().nth(1).expect("no header line");
    assert!(header.contains("Cluster:I:1"), "{header}");
    assert!(
        !header.contains("ClusterSize"),
        "ClusterSize is per cluster and must be filtered from the atom table:\n{header}"
    );
}

/// Three atoms bonded through the x boundary of a 10 A box (types 1, 2, 1)
/// plus a dimer of type 1, so `--select` has something to choose by.
fn boundary_trio(name: &str) -> PathBuf {
    let mut pd = ParticleData::with_capacity(5);
    for (i, (p, t)) in [
        ([0.5, 5.0, 5.0], 1),
        ([9.5, 5.0, 5.0], 2),
        ([9.0, 5.6, 5.0], 1),
        ([3.0, 3.0, 3.0], 1),
        ([3.5, 3.0, 3.0], 1),
    ]
    .into_iter()
    .enumerate()
    {
        pd.add_atom(p, i as i64 + 1, t);
    }
    let frame = Frame::new(UnitCell::orthorhombic(10.0, 10.0, 10.0), pd);
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

#[test]
fn the_table_unwrap_and_sort_reach_the_outputs() {
    let path = boundary_trio("table.extxyz");
    let table = tempdir().join("table.csv");
    let out_path = tempdir().join("table.out.xyz");
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "1.5",
        "--sort-by-size",
        "--unwrap",
        "--table",
        table.to_str().unwrap(),
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    let text = std::fs::read_to_string(&table).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().map(|l| l.split(',').collect()).collect();
    assert_eq!(
        rows[0].join(","),
        "cluster,size,mass,com_x,com_y,com_z,rg,g_xx,g_yy,g_zz,g_xy,g_xz,g_yz,percolates"
    );
    assert_eq!(rows.len(), 3, "{text}");
    // Largest first; unit weights (an xyz carries no masses): the trio
    // unwraps to x = 0.5, -0.5, -1.0 around its first atom.
    assert_eq!(rows[1][..3], ["0", "3", "3"]);
    let com_x: f64 = rows[1][3].parse().unwrap();
    assert!((com_x - (-1.0 / 3.0)).abs() < 1e-12, "{text}");
    assert_eq!(rows[2][..3], ["1", "2", "2"]);
    // Positions written unwrapped, labels sorted, no duplicate column.
    assert_eq!(cluster_ids(&out_path), vec![0, 0, 0, 1, 1]);
    let written = std::fs::read_to_string(&out_path).unwrap();
    let x: Vec<f64> = written
        .lines()
        .skip(2)
        .map(|l| l.split_whitespace().nth(1).unwrap().parse().unwrap())
        .collect();
    assert_eq!(x[..3], [0.5, -0.5, -1.0]);
    assert!(!written.contains("UnwrappedPosition"));
}

#[test]
fn select_clusters_only_the_matching_atoms() {
    let path = boundary_trio("select.extxyz");
    let out_path = tempdir().join("select.out.xyz");
    let (out, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "1.5",
        "--select",
        "type == 1",
        "-o",
        out_path.to_str().unwrap(),
        path.to_str().unwrap(),
    ]);
    assert!(ok, "{out}{err}");
    // Without the type-2 atom the trio's ends (1.5 A in x, 0.6 A in y apart
    // through the boundary) are beyond the cutoff: three clusters.
    assert_eq!(cluster_ids(&out_path), vec![0, -1, 1, 2, 2]);
    assert_eq!(scalar(&out, "cluster_atoms_selected"), 4.0);
    assert_eq!(scalar(&out, "cluster_count"), 3.0);

    let (_, err, ok) = rmda(&[
        "cluster",
        "--cutoff",
        "1.5",
        "--only-selected",
        path.to_str().unwrap(),
    ]);
    assert!(!ok, "a frame without a Selection column must be refused");
    assert!(err.contains("Selection"), "{err}");
}

/// Six atoms in three well-separated pairs, typed Cu (1) and Zr (2) and named
/// in the extxyz `species`: a Cu-Cu pair 2.0 A apart, a Cu-Zr pair 2.5 A apart
/// and a Zr-Zr pair 2.5 A apart.
fn cu_zr_pairs(name: &str) -> PathBuf {
    let atoms = [
        ([0.0, 0.0, 0.0], 1),
        ([2.0, 0.0, 0.0], 1),
        ([10.0, 0.0, 0.0], 1),
        ([12.5, 0.0, 0.0], 2),
        ([20.0, 0.0, 0.0], 2),
        ([22.5, 0.0, 0.0], 2),
    ];
    let mut pd = ParticleData::with_capacity(atoms.len());
    for (i, (p, t)) in atoms.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, *t);
    }
    let mut frame = Frame::new(UnitCell::orthorhombic(40.0, 40.0, 40.0), pd);
    frame.type_labels = vec!["Cu".into(), "Zr".into()];
    let path = tempdir().join(name);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &frame).unwrap();
    path
}

#[test]
fn pair_cutoffs_bond_each_type_pair_at_its_own_cutoff() {
    // Cu-Cu at 2.1 and Cu-Zr at 2.6 bond those pairs; Zr-Zr is not listed, so
    // its pair stays apart although it is as close as the Cu-Zr one. Types by
    // name and by number give the same answer.
    let path = cu_zr_pairs("pairs.extxyz");
    let dir = tempdir();
    for (k, spec) in [["Cu-Cu=2.1", "Zr-Cu=2.6"], ["1-1=2.1", "1-2=2.6"]]
        .iter()
        .enumerate()
    {
        let o = dir.join(format!("pairs_out{k}.extxyz"));
        let (out, err, ok) = rmda(&[
            "cluster",
            path.to_str().unwrap(),
            "--pair-cutoff",
            spec[0],
            "--pair-cutoff",
            spec[1],
            "-o",
            o.to_str().unwrap(),
        ]);
        assert!(ok, "{out}{err}");
        assert_eq!(scalar(&out, "cluster_count"), 4.0, "{out}");
        assert_eq!(scalar(&out, "cluster_mode_cutoff"), 2.6, "{out}");
        assert!(err.contains("mode=pairs("), "{err}");
        assert_eq!(cluster_ids(&o), vec![0, 0, 1, 1, 2, 3]);
    }
}

#[test]
fn pair_cutoffs_refuse_other_modes_and_unknown_types() {
    let path = cu_zr_pairs("pairs_bad.extxyz");
    let p = path.to_str().unwrap();
    let (_, err, ok) = rmda(&["cluster", p, "--cutoff", "2", "--pair-cutoff", "1-1=2"]);
    assert!(
        !ok && (err.contains("cannot") || err.contains("conflict")),
        "{err}"
    );
    let (_, err, ok) = rmda(&["cluster", p, "--pair-cutoff", "Cu-Ni=2"]);
    assert!(!ok && err.contains("'Ni'"), "{err}");
    let (_, err, ok) = rmda(&["cluster", p, "--pair-cutoff", "1-2"]);
    assert!(!ok && err.contains("TYPE-TYPE=CUTOFF"), "{err}");
}
