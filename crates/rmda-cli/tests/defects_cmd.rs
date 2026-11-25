//! `rmda defects` end to end: regions in, table and defect files out.

use std::path::PathBuf;
use std::process::Command;

/// Run `rmda`; `ok` when it ran and wrote its results: exit 0, or 5, which a
/// single-frame `rmda defects` returns when no grain was measured (design
/// §21; pinned in tests/malformed_input.rs). These tests check the counts and
/// the refusals themselves.
fn rmda(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        matches!(out.status.code(), Some(0) | Some(5)),
    )
}

fn scalar(stdout: &str, name: &str) -> i64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no {name} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap()
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_defects_cmd_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A BCC slab of `n`×`n`×`nz` cells in a box with vacuum above, as extxyz,
/// with the atoms of `skip` (cell, basis) removed.
fn slab(n: usize, nz: usize, a: f64, skip: &[(usize, usize, usize, usize)]) -> String {
    let mut rows = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..nz {
                for (b, off) in [0.0, 0.5].iter().enumerate() {
                    if skip.contains(&(i, j, k, b)) {
                        continue;
                    }
                    let p = [
                        (i as f64 + off) * a,
                        (j as f64 + off) * a,
                        (k as f64 + off) * a,
                    ];
                    rows.push(format!("W {} {} {}", p[0], p[1], p[2] + 5.0));
                }
            }
        }
    }
    let l = n as f64 * a;
    let lz = 3.0 * nz as f64 * a;
    format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {lz}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

#[test]
fn two_regions_are_measured_separately_and_conserve() {
    let a = 3.165;
    let dir = tempdir("regions");
    let input = dir.join("slab.xyz");
    // One vacancy low in the slab, two high.
    std::fs::write(
        &input,
        slab(10, 16, a, &[(5, 5, 3, 0), (2, 7, 12, 1), (6, 3, 13, 0)]),
    )
    .unwrap();
    let mid = format!("z <= {}", 5.0 + 8.0 * a);
    let hi = format!("z > {}", 5.0 + 8.0 * a);
    let table = dir.join("grains.csv");
    let out_dir = dir.join("defects");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--region",
        &mid,
        "--region",
        &hi,
        "--table",
        table.to_str().unwrap(),
        "--defects",
        out_dir.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 3, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_balance_residual"), 0);

    let csv = std::fs::read_to_string(&table).unwrap();
    let rows: Vec<&str> = csv.lines().collect();
    assert_eq!(rows.len(), 3, "{csv}");
    assert!(
        rows[1].contains(",measured,") && rows[2].contains(",measured,"),
        "{csv}"
    );

    let vac = std::fs::read_to_string(out_dir.join("vacancies.xyz")).unwrap();
    assert!(vac.starts_with("3\n"), "{vac}");
    assert!(vac.lines().nth(1).unwrap().contains("Origin="));
    let int = std::fs::read_to_string(out_dir.join("interstitials.xyz")).unwrap();
    assert!(int.starts_with("0\n"), "{int}");
}

#[test]
fn a_bad_region_is_a_usage_error_not_a_crash() {
    let dir = tempdir("bad");
    let input = dir.join("slab.xyz");
    std::fs::write(&input, slab(4, 4, 3.165, &[])).unwrap();
    let (_, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--region", "q < 1"]);
    assert!(!ok);
    assert!(err.contains("unknown field"), "{err}");
}

#[test]
fn segment_finds_one_grain_and_defects_can_use_it() {
    let dir = tempdir("segment");
    let input = dir.join("slab.xyz");
    std::fs::write(&input, slab(10, 12, 3.165, &[(4, 4, 6, 0)])).unwrap();
    let table = dir.join("seg.csv");
    let (out, err, ok) = rmda(&[
        "segment",
        input.to_str().unwrap(),
        "--table",
        table.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "segment_grains"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "segment_atoms_unassigned"), 0, "{out}\n{err}");
    assert!(dir.join("seg.pairs.csv").exists());

    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--segment"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
}

#[test]
fn patch_reference_matches_global_on_a_clean_slab() {
    let dir = tempdir("patch");
    let input = dir.join("slab.xyz");
    std::fs::write(&input, slab(12, 12, 3.165, &[(5, 5, 5, 0), (7, 2, 8, 1)])).unwrap();
    let table = dir.join("grains.csv");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--reference",
        "patch",
        "--patch-size",
        "12",
        "--table",
        table.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
    let csv = std::fs::read_to_string(&table).unwrap();
    assert!(
        csv.lines().nth(1).unwrap().contains(",measured,patch,"),
        "{csv}"
    );
}

#[test]
fn a_series_writes_long_tables_resumes_and_refuses_changed_settings() {
    let dir = tempdir("series");
    // Indices out of lexical order (10 < 9 as strings) and different defects.
    let frames = [
        ("frame-9.xyz", vec![(5, 5, 5, 0)]),
        ("frame-10.xyz", vec![(5, 5, 5, 0), (2, 2, 7, 1)]),
        (
            "frame-11.xyz",
            vec![(5, 5, 5, 0), (5, 5, 5, 1), (2, 2, 7, 1)],
        ),
    ];
    let mut paths = Vec::new();
    for (name, skip) in &frames {
        let p = dir.join(name);
        std::fs::write(&p, slab(10, 12, 3.165, skip)).unwrap();
        paths.push(p.to_str().unwrap().to_owned());
    }
    let out = dir.join("out");
    let mut args = vec!["defects".to_owned()];
    args.extend(paths.iter().rev().cloned());
    args.extend(["--out".to_owned(), out.to_str().unwrap().to_owned()]);
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let (o, e, ok) = rmda(&argv);
    assert!(ok, "{o}\n{e}");

    let summary = std::fs::read_to_string(out.join("summary.csv")).unwrap();
    let rows: Vec<Vec<&str>> = summary
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let idx: Vec<&str> = rows.iter().map(|r| r[0]).collect();
    assert_eq!(idx, ["9", "10", "11"], "{summary}");
    let col = |name: &str| {
        summary
            .lines()
            .next()
            .unwrap()
            .split(',')
            .position(|c| c == name)
            .unwrap()
    };
    let vac: Vec<&str> = rows.iter().map(|r| r[col("vacancies")]).collect();
    assert_eq!(vac, ["1", "2", "3"], "{summary}");
    // The divacancy in frame 11 is one cluster of two.
    assert_eq!(rows[2][col("largest_cluster")], "2", "{summary}");
    let grains = std::fs::read_to_string(out.join("grains.csv")).unwrap();
    assert_eq!(grains.lines().count(), 4, "{grains}");
    let clusters = std::fs::read_to_string(out.join("clusters.csv")).unwrap();
    assert_eq!(clusters.lines().count(), 1 + 1 + 2 + 2, "{clusters}");

    // A rerun skips everything.
    let (o, e, ok) = rmda(&argv);
    assert!(ok, "{o}\n{e}");
    assert_eq!(e.matches("already done").count(), 3, "{e}");
    assert_eq!(
        std::fs::read_to_string(out.join("summary.csv")).unwrap(),
        summary
    );

    // Different settings into the same directory are refused...
    let mut changed = argv.clone();
    changed.extend(["--min-atoms", "50"]);
    let (_, e, ok) = rmda(&changed);
    assert!(!ok);
    assert!(e.contains("different settings"), "{e}");
    // ...unless asked to overwrite.
    changed.push("--overwrite");
    let (o, e, ok) = rmda(&changed);
    assert!(ok, "{o}\n{e}");
    assert_eq!(
        std::fs::read_to_string(out.join("summary.csv"))
            .unwrap()
            .lines()
            .count(),
        4
    );
}

#[test]
fn an_unnumbered_baseline_comes_first_in_a_series() {
    let dir = tempdir("baseline");
    let base = dir.join("data.firstframe.xyz");
    let later = dir.join("data.lastframe-7.xyz");
    std::fs::write(&base, slab(8, 8, 3.165, &[])).unwrap();
    std::fs::write(&later, slab(8, 8, 3.165, &[(3, 3, 3, 0)])).unwrap();
    let out = dir.join("out");
    let (o, e, ok) = rmda(&[
        "defects",
        later.to_str().unwrap(),
        base.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "{o}\n{e}");
    let summary = std::fs::read_to_string(out.join("summary.csv")).unwrap();
    let idx: Vec<&str> = summary
        .lines()
        .skip(1)
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert_eq!(idx, ["-1", "7"], "{summary}");
}

/// FCC as close-packed layers along z with stacking `seq`, periodic in x and
/// y, vacuum in z, as extxyz; `skip` lists (layer, atom-in-layer) to leave out.
fn stacked(seq: &str, nx: usize, ny: usize, d: f64, skip: &[(usize, usize)]) -> String {
    let (lx, ly) = (nx as f64 * d, ny as f64 * d * 3f64.sqrt());
    let dz = d * (2.0f64 / 3.0).sqrt();
    let mut rows = Vec::new();
    for (k, c) in seq.chars().enumerate() {
        let s = match c {
            'A' => 0.0,
            'B' => 1.0,
            _ => 2.0,
        };
        let mut n = 0;
        for i in 0..nx {
            for j in 0..ny {
                for (fx, fy) in [(0.0, 0.0), (0.5, 0.5)] {
                    n += 1;
                    if skip.contains(&(k, n - 1)) {
                        continue;
                    }
                    let x = ((i as f64 + fx) * d + s * 0.5 * d).rem_euclid(lx);
                    let y = ((j as f64 + fy) * d * 3f64.sqrt() + s * d / (2.0 * 3f64.sqrt()))
                        .rem_euclid(ly);
                    rows.push(format!("Cu {x} {y} {}", 10.0 + k as f64 * dz));
                }
            }
        }
    }
    let lz = 20.0 + seq.len() as f64 * dz;
    format!(
        "{}\nLattice=\"{lx} 0 0 0 {ly} 0 0 0 {lz}\" pbc=\"T T F\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

#[test]
fn a_stacking_fault_across_a_grain_is_gated_globally_counted_locally_and_separated() {
    let dir = tempdir("isf");
    let input = dir.join("isf.xyz");
    // ABCABC|BCA...: an intrinsic fault across the slab; one vacancy deep on
    // each side.
    let seq = format!("{}{}", "ABC".repeat(4), "BCA".repeat(4));
    std::fs::write(&input, stacked(&seq, 12, 7, 2.556, &[(5, 40), (18, 77)])).unwrap();
    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--segment"]);
    assert!(ok, "{out}\n{err}");
    // One FCC grain; its two sides are a partial Burgers vector apart, so one
    // lattice cannot count them: global and patch refuse it, and auto counts it
    // locally, where each side is on its own lattice.
    assert_eq!(scalar(&out, "defects_grains"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{out}\n{err}");
    assert!(err.contains("counting it locally"), "{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--segment",
        "--reference",
        "global",
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 0, "{out}\n{err}");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--segment",
        "--seg-stacking-faults",
        "separate",
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains"), 3, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 3, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_balance_residual"), 0);
}

#[test]
fn per_atom_output_is_refused_with_the_options_that_do_write() {
    let dir = tempdir("dash_o");
    let input = dir.join("slab.xyz");
    std::fs::write(&input, slab(4, 4, 3.165, &[])).unwrap();
    let out_file = dir.join("never.xyz");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "-o",
        out_file.to_str().unwrap(),
    ]);
    assert!(!ok, "-o must not be accepted silently:\n{out}\n{err}");
    for hint in ["--table", "--defects", "--out"] {
        assert!(err.contains(hint), "error must point at {hint}:\n{err}");
    }
    assert!(!out_file.exists());
}

#[test]
fn non_lattice_atoms_shell_every_interior_vacancy() {
    let a = 3.165;
    let dir = tempdir("non_lattice");
    let input = dir.join("slab.xyz");
    // Two vacancies deep in a BCC slab with vacuum above: the slab's free
    // surface is non-lattice too, the vacancies' first shells must be.
    std::fs::write(&input, slab(8, 8, a, &[(3, 3, 3, 0), (5, 4, 4, 1)])).unwrap();
    let out_dir = dir.join("defects");
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--defects",
        out_dir.to_str().unwrap(),
        "--non-lattice-atoms",
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 2, "{out}\n{err}");

    let vac = std::fs::read_to_string(out_dir.join("vacancies.xyz")).unwrap();
    assert!(
        vac.lines()
            .nth(1)
            .unwrap()
            .contains(":non_lattice_shell:I:1"),
        "{vac}"
    );
    for row in vac.lines().skip(2).filter(|l| !l.is_empty()) {
        let shell: u32 = row.split_whitespace().last().unwrap().parse().unwrap();
        assert_eq!(
            shell, 8,
            "a BCC vacancy has 8 unmatched first neighbours: {row}"
        );
    }
    let nl = std::fs::read_to_string(out_dir.join("non_lattice.xyz")).unwrap();
    let header = nl.lines().nth(1).unwrap();
    assert!(header.contains("StructureType:I:1:grain:I:1"), "{header}");
    let n: usize = nl.lines().next().unwrap().parse().unwrap();
    assert_eq!(n, nl.lines().skip(2).filter(|l| !l.is_empty()).count());
    // Every listed atom is off the BCC lattice by PTM.
    for row in nl.lines().skip(2).filter(|l| !l.is_empty()) {
        let st: i32 = row.split_whitespace().nth(6).unwrap().parse().unwrap();
        assert_ne!(st, 3, "a BCC atom of a BCC grain is not non-lattice: {row}");
    }

    // Without a place to write it, the flag is an error, not a no-op.
    let (_, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--non-lattice-atoms"]);
    assert!(!ok && err.contains("--defects"), "{err}");
}

/// A free-standing BCC block of `n`³ cells, bent about the y axis by
/// `kappa` radians per Å along x, with the atoms of `skip` removed and
/// `extra` atoms added (Cartesian, before bending).
fn bent_block(
    n: usize,
    a: f64,
    kappa: f64,
    skip: &[(usize, usize, usize, usize)],
    extra: &[[f64; 3]],
) -> String {
    let mut pts = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for (b, off) in [0.0, 0.5].iter().enumerate() {
                    if !skip.contains(&(i, j, k, b)) {
                        pts.push([
                            (i as f64 + off) * a,
                            (j as f64 + off) * a,
                            (k as f64 + off) * a,
                        ]);
                    }
                }
            }
        }
    }
    pts.extend_from_slice(extra);
    let l = n as f64 * a;
    let c = l / 2.0;
    let rows: Vec<String> = pts
        .iter()
        .map(|p| {
            // Rotate each atom about y through the block centre by an angle
            // that grows along x: the lattice turns by kappa·l across it.
            let th = kappa * (p[0] - c);
            let (s, co) = th.sin_cos();
            let (x, z) = (p[0] - c, p[2] - c);
            format!(
                "W {} {} {}",
                co * x + s * z + c + 10.0,
                p[1] + 10.0,
                -s * x + co * z + c + 10.0
            )
        })
        .collect();
    let lb = l + 20.0;
    format!(
        "{}\nLattice=\"{lb} 0 0 0 {lb} 0 0 0 {lb}\" pbc=\"F F F\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

#[test]
fn local_reference_counts_vacancies_a_divacancy_and_a_dumbbell_exactly() {
    let a = 3.165;
    let dir = tempdir("local");
    let input = dir.join("slab.xyz");
    // A vacancy, a first-neighbour divacancy, and a <110> dumbbell made by
    // replacing one atom with two.
    let s = slab(
        12,
        12,
        a,
        &[(5, 5, 5, 0), (7, 2, 8, 0), (7, 2, 8, 1), (3, 3, 6, 0)],
    );
    let d = 0.25 * a;
    let c = [3.0 * a, 3.0 * a, 6.0 * a + 5.0];
    let mut lines: Vec<String> = s.lines().map(str::to_owned).collect();
    let n: usize = lines[0].parse().unwrap();
    lines[0] = (n + 2).to_string();
    lines.push(format!("W {} {} {}", c[0] + d, c[1] + d, c[2]));
    lines.push(format!("W {} {} {}", c[0] - d, c[1] - d, c[2]));
    std::fs::write(&input, lines.join("\n") + "\n").unwrap();
    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", "local"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 3, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_balance_residual"), 0);
}

#[test]
fn local_reference_counts_a_bent_crystal_no_single_lattice_fits() {
    let a = 3.165;
    let n = 16;
    // 12 degrees across the block: 2.6 A off at its ends for one lattice.
    let kappa = 12f64.to_radians() / (n as f64 * a);
    let dir = tempdir("bent");
    let input = dir.join("bent.xyz");
    let skip = [(3, 8, 8, 0), (8, 8, 8, 1), (12, 5, 9, 0), (8, 12, 4, 1)];
    std::fs::write(&input, bent_block(n, a, kappa, &skip, &[])).unwrap();
    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", "global"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 0, "{out}\n{err}");
    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", "local"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 4, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_balance_residual"), 0);
}

#[test]
fn a_series_lays_defect_files_out_by_dirs_files_or_one_appended_file() {
    let dir = tempdir("layout");
    let frames = [
        ("frame-9.xyz", vec![(5, 5, 5, 0)]),
        ("frame-10.xyz", vec![(5, 5, 5, 0), (2, 2, 7, 1)]),
        (
            "frame-11.xyz",
            vec![(5, 5, 5, 0), (6, 2, 7, 1), (2, 2, 7, 1)],
        ),
    ];
    let mut paths = Vec::new();
    for (name, skip) in &frames {
        let p = dir.join(name);
        std::fs::write(&p, slab(10, 12, 3.165, skip)).unwrap();
        paths.push(p.to_str().unwrap().to_owned());
    }
    let run = |layout: &str, out: &std::path::Path, files: &[String]| {
        let mut args = vec!["defects".to_owned()];
        args.extend(files.iter().cloned());
        args.extend(
            [
                "--out",
                out.to_str().unwrap(),
                "--defects-layout",
                layout,
                "--non-lattice-atoms",
            ]
            .map(str::to_owned),
        );
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let (o, e, ok) = rmda(&argv);
        assert!(ok, "{o}\n{e}");
    };

    let out = dir.join("dirs");
    run("dirs", &out, &paths);
    for k in ["9", "10", "11"] {
        for n in ["vacancies", "interstitials", "non_lattice"] {
            assert!(out
                .join("defects")
                .join(k)
                .join(format!("{n}.xyz"))
                .exists());
        }
    }

    let out = dir.join("files");
    run("files", &out, &paths);
    let vac = std::fs::read_to_string(out.join("defects/vacancies-10.xyz")).unwrap();
    assert!(vac.starts_with("2\nindex=10 "), "{vac}");
    assert!(out.join("defects/non_lattice-11.xyz").exists());
    assert!(!out.join("defects/10").exists());

    // Appended: one file, a frame per index in index order, each frame's
    // comment carrying its index.
    let frames_of = |p: &std::path::Path| -> Vec<(String, usize)> {
        let t = std::fs::read_to_string(p).unwrap();
        let lines: Vec<&str> = t.lines().collect();
        let mut v = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let n: usize = lines[i].parse().unwrap();
            let idx = lines[i + 1]
                .split_whitespace()
                .find_map(|w| w.strip_prefix("index="))
                .unwrap()
                .to_owned();
            v.push((idx, n));
            i += n + 2;
        }
        v
    };
    let out = dir.join("append");
    run("append", &out, &paths[..2]);
    // A frame interrupted after appending but before its summary row: its
    // frames are dropped on resume, not duplicated.
    let vac = out.join("defects/vacancies.xyz");
    let extra = std::fs::read_to_string(&vac).unwrap();
    let one = extra
        .split_inclusive('\n')
        .take(3)
        .collect::<String>()
        .replace("index=9 ", "index=11 ");
    std::fs::write(&vac, extra + &one).unwrap();
    run("append", &out, &paths);
    let want: Vec<(String, usize)> = [("9", 1), ("10", 2), ("11", 3)]
        .iter()
        .map(|&(k, n)| (k.to_owned(), n))
        .collect();
    assert_eq!(frames_of(&vac), want);
    let ints = frames_of(&out.join("defects/interstitials.xyz"));
    assert_eq!(
        ints.iter().map(|f| f.0.as_str()).collect::<Vec<_>>(),
        ["9", "10", "11"]
    );
    assert_eq!(frames_of(&out.join("defects/non_lattice.xyz")).len(), 3);
    assert!(!out.join("defects/9").exists());
}

#[test]
fn defect_files_name_types_by_elements_given_else_by_mass() {
    // A free BCC block in a LAMMPS data file with masses and no labels;
    // types alternate 1 (V) and 2 (W). Its surface atoms are non-lattice.
    let a = 3.1;
    let n = 6;
    let mut rows = Vec::new();
    let mut id = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for off in [0.0, 0.5] {
                    id += 1;
                    let t = 1 + id % 2;
                    let p = [
                        (i as f64 + off) * a + 10.0,
                        (j as f64 + off) * a + 10.0,
                        (k as f64 + off) * a + 10.0,
                    ];
                    rows.push(format!("{id} {t} {} {} {}", p[0], p[1], p[2]));
                }
            }
        }
    }
    let l = n as f64 * a + 20.0;
    let data = format!(
        "test\n\n{id} atoms\n2 atom types\n\n0 {l} xlo xhi\n0 {l} ylo yhi\n0 {l} zlo zhi\n\n\
         Masses\n\n1 50.9415\n2 183.84\n\nAtoms # atomic\n\n{}\n",
        rows.join("\n")
    );
    let dir = tempdir("elements");
    let input = dir.join("block.data");
    std::fs::write(&input, data).unwrap();
    let species = |extra: &[&str]| -> std::collections::BTreeSet<String> {
        let out = dir.join("d");
        let mut args = vec![
            "defects",
            input.to_str().unwrap(),
            "--defects",
            out.to_str().unwrap(),
            "--non-lattice-atoms",
        ];
        args.extend_from_slice(extra);
        let (o, e, ok) = rmda(&args);
        assert!(ok, "{o}\n{e}");
        let t = std::fs::read_to_string(out.join("non_lattice.xyz")).unwrap();
        t.lines()
            .skip(2)
            .map(|l| l.split_whitespace().next().unwrap().to_owned())
            .collect()
    };
    assert_eq!(species(&[]), ["V", "W"].map(str::to_owned).into());
    assert_eq!(
        species(&["--elements", "Nb", "Ta"]),
        ["Nb", "Ta"].map(str::to_owned).into()
    );
}

#[test]
fn tables_report_net_vacancies_and_local_figures_only_where_they_apply() {
    let a = 3.165;
    let dir = tempdir("netcols");
    // Three vacancies and one interstitial: V − I = 2.
    let s = slab(
        12,
        12,
        a,
        &[(5, 5, 5, 0), (7, 2, 8, 0), (7, 2, 8, 1), (3, 3, 6, 0)],
    );
    let d = 0.25 * a;
    let c = [3.0 * a, 3.0 * a, 6.0 * a + 5.0];
    let mut lines: Vec<String> = s.lines().map(str::to_owned).collect();
    let n: usize = lines[0].parse().unwrap();
    lines[0] = (n + 2).to_string();
    lines.push(format!("W {} {} {}", c[0] + d, c[1] + d, c[2]));
    lines.push(format!("W {} {} {}", c[0] - d, c[1] - d, c[2]));
    let input = dir.join("slab-1.xyz");
    std::fs::write(&input, lines.join("\n") + "\n").unwrap();

    for reference in ["local", "global"] {
        let out_dir = dir.join(reference);
        let (out, err, ok) = rmda(&[
            "defects",
            input.to_str().unwrap(),
            "--reference",
            reference,
            "--out",
            out_dir.to_str().unwrap(),
        ]);
        assert!(ok, "{out}\n{err}");
        let table = |name: &str| -> Vec<std::collections::HashMap<String, String>> {
            let t = std::fs::read_to_string(out_dir.join(name)).unwrap();
            let mut it = t.lines();
            let head: Vec<String> = it.next().unwrap().split(',').map(str::to_owned).collect();
            it.map(|l| {
                head.iter()
                    .cloned()
                    .zip(l.split(',').map(str::to_owned))
                    .collect()
            })
            .collect()
        };
        let sum = &table("summary.csv")[0];
        assert_eq!(sum["vacancies"], "3", "{reference}");
        assert_eq!(sum["interstitials"], "1", "{reference}");
        assert_eq!(sum["net_vacancies"], "2", "{reference}");
        assert_eq!(sum["gb_net_vacancies"], "0", "{reference}");
        let g = &table("grains.csv")[0];
        assert_eq!(g["reference"], reference);
        assert_eq!(g["net_vacancies"], "2", "{reference}");
        let local_cols = [
            "local_chunks",
            "local_defect_atoms",
            "local_unresolved",
            "local_regions",
            "local_region_atoms",
            "local_extended_atoms",
            "local_fit_rms",
        ];
        let patch_cols = ["n_patches", "n_patches_fitted", "phase_agreement"];
        if reference == "local" {
            for k in patch_cols {
                assert_eq!(g[k], "", "{k} for a local grain");
            }
            for k in local_cols {
                assert_ne!(g[k], "", "{k} for a local grain");
            }
            assert_eq!(g["local_unresolved"], "0");
            assert!(g["local_defect_atoms"].parse::<u64>().unwrap() > 0);
        } else {
            assert_eq!(g["n_patches"], "1");
            assert_eq!(g["phase_agreement"], "1.000000");
            for k in local_cols {
                assert_eq!(g[k], "", "{k} for a global grain");
            }
        }
    }

    // A table written with other columns is not appended to.
    let stale = dir.join("stale");
    std::fs::create_dir_all(&stale).unwrap();
    std::fs::write(stale.join("grains.csv"), "file_index,file,timestep,grain\n").unwrap();
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--out",
        stale.to_str().unwrap(),
    ]);
    assert!(!ok, "{out}\n{err}");
    assert!(err.contains("other columns"), "{err}");
}

/// A periodic BCC W block of `n`³ cells with seeded noise of ±`amp`/2 Å,
/// four vacancies, a ⟨110⟩ dumbbell, and two discs of extra atoms of radius
/// `r` Å between (001) planes, the second offset by `off` cells along x and
/// z: interstitial platelets whose clusters the local census counts whole.
/// Holds V = 4 and I = 1 + both discs' atoms.
fn platelet_block(n: usize, r: f64, amp: f64, off: [f64; 2]) -> (String, usize) {
    let a = 3.165;
    let mut seed: u64 = 12345;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let skip = [(2, 2, 2, 0), (9, 3, 4, 1), (9, 3, 5, 0), (4, 10, 10, 0)];
    let mut pts = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for (b, o) in [0.0, 0.5].iter().enumerate() {
                    if !skip.contains(&(i, j, k, b)) {
                        pts.push([(i as f64 + o) * a, (j as f64 + o) * a, (k as f64 + o) * a]);
                    }
                }
            }
        }
    }
    let c = [3.5 * a, 11.5 * a, 3.5 * a];
    pts.retain(|p| (0..3).map(|x| (p[x] - c[x]).powi(2)).sum::<f64>() > 0.01);
    let d = 0.25 * a;
    pts.push([c[0] + d, c[1] + d, c[2]]);
    pts.push([c[0] - d, c[1] - d, c[2]]);
    let mid = (n / 2) as f64 + 0.25;
    let mut disc = 0;
    for (ox, oz) in [(0.0, 0.0), (off[0], off[1])] {
        for i in 0..n {
            for j in 0..n {
                let (x, y) = ((i as f64 + 0.25) * a, (j as f64 + 0.25) * a);
                if (x - (mid + ox) * a).powi(2) + (y - mid * a).powi(2) < r * r {
                    pts.push([x, y, (mid + oz) * a]);
                    disc += 1;
                }
            }
        }
    }
    let l = n as f64 * a;
    let rows: Vec<String> = pts
        .iter()
        .map(|p| {
            let q: Vec<f64> = p
                .iter()
                .map(|&v| (v + amp * (rnd() - 0.5)).rem_euclid(l))
                .collect();
            format!("W {} {} {}", q[0], q[1], q[2])
        })
        .collect();
    let text = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    );
    (text, 1 + disc)
}

/// `--local-merge partial` counts clusters level by level in parallel; the
/// result must not depend on the thread count, and on the platelets whose
/// union is no region it must count what the default does.
#[test]
fn partial_local_merge_is_exact_and_independent_of_threads() {
    let dir = tempdir("partial_merge");
    for (tag, off) in [("merge", [4.0, 3.0]), ("extended", [5.0, 2.0])] {
        let (text, interstitials) = platelet_block(16, 5.0, 0.3, off);
        let input = dir.join(format!("{tag}.xyz"));
        std::fs::write(&input, text).unwrap();
        let mut runs = Vec::new();
        for threads in ["1", "4"] {
            let d = dir.join(format!("{tag}-{threads}"));
            std::fs::create_dir_all(&d).unwrap();
            let (out, err, ok) = rmda(&[
                "defects",
                input.to_str().unwrap(),
                "--reference",
                "local",
                "--local-merge",
                "partial",
                "--threads",
                threads,
                "--defects",
                d.to_str().unwrap(),
            ]);
            assert!(ok, "{out}\n{err}");
            assert_eq!(scalar(&out, "defects_vacancies"), 4, "{tag}\n{err}");
            assert_eq!(
                scalar(&out, "defects_interstitials"),
                interstitials as i64,
                "{tag}\n{err}"
            );
            let file = |n: &str| std::fs::read(d.join(n)).unwrap();
            runs.push((out, file("vacancies.xyz"), file("interstitials.xyz")));
        }
        assert!(
            runs[0] == runs[1],
            "{tag}: outputs differ between 1 and 4 threads"
        );
    }
}

#[test]
fn every_reference_counts_the_same_at_any_thread_count() {
    let dir = tempdir("threads");
    // Two platelets whose regions overlap and merge, and two whose union is
    // no region: the second is counted as a partial region (before, its 54
    // atoms went to the atom-by-atom count).
    for (tag, off, local_check) in [
        ("merge", [4.0, 3.0], "1 clusters counted as regions"),
        (
            "extended",
            [5.0, 2.0],
            "2 clusters counted as regions (602 atoms), 0 defect atoms in extended defects",
        ),
    ] {
        let (text, interstitials) = platelet_block(16, 5.0, 0.3, off);
        let input = dir.join(format!("{tag}.xyz"));
        std::fs::write(&input, text).unwrap();
        for reference in ["global", "patch", "local"] {
            let mut runs = Vec::new();
            for threads in ["1", "4"] {
                let d = dir.join(format!("{tag}-{reference}-{threads}"));
                std::fs::create_dir_all(&d).unwrap();
                let table = d.join("grains.csv");
                let (out, err, ok) = rmda(&[
                    "defects",
                    input.to_str().unwrap(),
                    "--reference",
                    reference,
                    "--threads",
                    threads,
                    "--table",
                    table.to_str().unwrap(),
                    "--defects",
                    d.to_str().unwrap(),
                ]);
                assert!(ok, "{out}\n{err}");
                assert_eq!(
                    scalar(&out, "defects_grains_measured"),
                    1,
                    "{tag} {reference}\n{err}"
                );
                assert_eq!(
                    scalar(&out, "defects_vacancies"),
                    4,
                    "{tag} {reference}\n{err}"
                );
                assert_eq!(
                    scalar(&out, "defects_interstitials"),
                    interstitials as i64,
                    "{tag} {reference}\n{err}"
                );
                if reference == "local" {
                    assert!(err.contains(local_check), "{tag}: {err}");
                }
                let file = |n: &str| std::fs::read(d.join(n)).unwrap();
                runs.push((
                    out,
                    file("grains.csv"),
                    file("vacancies.xyz"),
                    file("interstitials.xyz"),
                ));
            }
            assert!(
                runs[0] == runs[1],
                "{tag} {reference}: 1 and 4 threads differ"
            );
        }
    }
}

/// A free-standing BCC W block of `n`³ cells holding an interstitial
/// dislocation loop: a disc of radius `r` cells of one extra unit cell (two
/// (002) layers, b = a[001]) at mid-height, with the crystal around it
/// displaced by the loop's solid-angle field `u = b·Ω/4π`, so the loop's
/// inside is perfect crystal and only its rim is a dislocation core. The
/// atoms of `vac` (cell, basis) are removed, and every atom gets Gaussian
/// noise of `sigma` Å per component. Returns the extxyz text and the atoms
/// the loop holds.
fn loop_block(
    n: usize,
    r: f64,
    sigma: f64,
    vac: &[(usize, usize, usize, usize)],
) -> (String, usize) {
    dislocation_loop(n, r, sigma, vac, false)
}

/// As [`loop_block`], or with `vacancy` a collapsed vacancy loop: the unit
/// cell of the stack inside the disc (the layer at mid-height and the one
/// above it) removed and the crystal closed over it by the opposite field.
/// Returns the text and the atoms the loop holds or lacks.
fn dislocation_loop(
    n: usize,
    r: f64,
    sigma: f64,
    vac: &[(usize, usize, usize, usize)],
    vacancy: bool,
) -> (String, usize) {
    let a = 3.165;
    let k0 = n / 2;
    let c = [
        n as f64 / 2.0 * a,
        n as f64 / 2.0 * a,
        (k0 as f64 + 0.25) * a,
    ];
    let rr = r * a;
    let nv = 96;
    let verts: Vec<[f64; 3]> = (0..nv)
        .map(|t| {
            let th = std::f64::consts::TAU * t as f64 / nv as f64;
            [c[0] + rr * th.cos(), c[1] + rr * th.sin(), c[2]]
        })
        .collect();
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    // Solid angle of the loop seen from p, as a fan of triangles from its
    // centre (Van Oosterom and Strackee).
    let omega = |p: [f64; 3]| -> f64 {
        let x = sub(c, p);
        let lx = dot(x, x).sqrt();
        (0..nv)
            .map(|t| {
                let (y, z) = (sub(verts[t], p), sub(verts[(t + 1) % nv], p));
                let (ly, lz) = (dot(y, y).sqrt(), dot(z, z).sqrt());
                let num = dot(x, cross(y, z));
                let den = lx * ly * lz + dot(x, y) * lz + dot(x, z) * ly + dot(y, z) * lx;
                2.0 * num.atan2(den)
            })
            .sum()
    };
    // Above the disc the crystal moves up by b/2, below it down; for a
    // vacancy loop the other way, closing the gap.
    let flip = if vacancy { -1.0 } else { 1.0 };
    let sign = flip * omega([c[0], c[1], c[2] + 0.1]).signum();
    let in_disc = |x: f64, y: f64| (x - c[0]).powi(2) + (y - c[1]).powi(2) < rr * rr;
    let mut held = 0;
    let mut pts = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for (b, o) in [0.0, 0.5].iter().enumerate() {
                    if vac.contains(&(i, j, k, b)) {
                        continue;
                    }
                    let p = [(i as f64 + o) * a, (j as f64 + o) * a, (k as f64 + o) * a];
                    if vacancy && k == k0 && in_disc(p[0], p[1]) {
                        held += 1;
                        continue;
                    }
                    let uz = a * sign * omega(p) / (4.0 * std::f64::consts::PI);
                    pts.push([p[0], p[1], p[2] + uz]);
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            for (ox, oz) in [(0.5, 0.0), (0.0, 0.5)] {
                let (x, y) = ((i as f64 + ox) * a, (j as f64 + ox) * a);
                if !vacancy && in_disc(x, y) {
                    pts.push([x, y, (k0 as f64 + oz) * a]);
                    held += 1;
                }
            }
        }
    }
    let mut seed: u64 = 777;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = || {
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let rows: Vec<String> = pts
        .iter()
        .map(|p| {
            format!(
                "W {} {} {}",
                p[0] + sigma * gauss() + 10.0,
                p[1] + sigma * gauss() + 10.0,
                p[2] + sigma * gauss() + 10.0
            )
        })
        .collect();
    let l = n as f64 * a + 20.0;
    let text = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"F F F\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    );
    (text, held)
}

#[test]
fn vacancies_beside_an_interstitial_loop_and_the_loop_itself_are_counted() {
    let dir = tempdir("loop");
    // On the rim, one cell outside it, above and below the disc inside the
    // loop, and far away.
    let vac = [
        (14, 10, 10, 0),
        (15, 10, 10, 0),
        (10, 15, 11, 0),
        (10, 10, 12, 0),
        (12, 8, 9, 1),
        (13, 13, 10, 1),
        (3, 3, 3, 0),
        (16, 4, 15, 1),
    ];
    let (text, held) = loop_block(20, 4.0, 0.05, &vac);
    assert_eq!(held, 99);
    let input = dir.join("loop.xyz");
    std::fs::write(&input, text).unwrap();
    let run = |reference: &str| -> (i64, i64) {
        let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(
            scalar(&out, "defects_grains_measured"),
            1,
            "{reference}\n{err}"
        );
        assert_eq!(scalar(&out, "defects_balance_residual"), 0);
        (
            scalar(&out, "defects_vacancies"),
            scalar(&out, "defects_interstitials"),
        )
    };
    // Inside the loop the crystal is displaced by up to b/2 from the
    // lattice fitted around it; patches follow the field, and so does a
    // region's census, which continues an ambiguous atom's site from its
    // neighbours'. Both exact (design §16.8: 32 of 32 and 30 of 32 seeds and
    // noise levels).
    assert_eq!(run("patch"), (8, 99));
    assert_eq!(run("local"), (8, 99));
}

/// A large loop whose own region fails, because its shell (a bond outside
/// the rim) lies in the loop's strongest strain, is counted with a larger
/// sphere: the strain falls off as 1/d^2. Before, its 223 atoms went to the
/// atom-by-atom count, which found 4 vacancies and 212 interstitials.
#[test]
fn a_loop_whose_tight_shell_is_strained_is_counted_with_a_larger_sphere() {
    let dir = tempdir("strained_loop");
    let (text, held) = loop_block(24, 6.0, 0.05, &[]);
    assert_eq!(held, 223);
    let input = dir.join("loop.xyz");
    std::fs::write(&input, text).unwrap();
    let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", "local"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(
        (
            scalar(&out, "defects_vacancies"),
            scalar(&out, "defects_interstitials")
        ),
        (0, 223),
        "{err}"
    );
}

#[test]
fn hot_perfect_crystal_has_no_defects() {
    let dir = tempdir("thermal");
    // σ = 0.12 Å per component, about twice W's at room temperature; no
    // reference found a false positive up to 0.15 Å over 8 seeds (§16.8).
    let (text, held) = loop_block(16, 0.0, 0.12, &[]);
    assert_eq!(held, 0);
    let input = dir.join("hot.xyz");
    std::fs::write(&input, text).unwrap();
    for reference in ["local", "patch"] {
        let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(
            scalar(&out, "defects_grains_measured"),
            1,
            "{reference}\n{err}"
        );
        assert_eq!(scalar(&out, "defects_vacancies"), 0, "{reference}\n{err}");
        assert_eq!(
            scalar(&out, "defects_interstitials"),
            0,
            "{reference}\n{err}"
        );
    }
}

/// BCC W sites of an `n`×`n`×`nz` block, less those within `r` Å of each
/// of `holes`, with seeded Gaussian noise of 0.05 Å, shifted up by `dz`, as
/// periodic extxyz in a box `gap` Å taller than the block. Returns the text
/// and the sites removed.
fn holed_block(
    n: usize,
    nz: usize,
    holes: &[([f64; 3], f64)],
    gap: f64,
    dz: f64,
) -> (String, usize) {
    let a = 3.165;
    let mut seed: u64 = 99;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = || {
        let (u, v) = (uniform(), uniform());
        0.05 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let (mut rows, mut removed) = (Vec::new(), 0);
    let (l, lz) = (n as f64 * a, nz as f64 * a + gap);
    for i in 0..n {
        for j in 0..n {
            for k in 0..nz {
                for o in [0.0, 0.5] {
                    let p = [(i as f64 + o) * a, (j as f64 + o) * a, (k as f64 + o) * a];
                    let inside = holes
                        .iter()
                        .any(|(c, r)| (0..3).map(|x| (p[x] - c[x]).powi(2)).sum::<f64>() < r * r);
                    if inside {
                        removed += 1;
                        continue;
                    }
                    let q = [p[0] + gauss(), p[1] + gauss(), p[2] + dz + gauss()];
                    rows.push(format!(
                        "W {} {} {}",
                        q[0].rem_euclid(l),
                        q[1].rem_euclid(l),
                        q[2].rem_euclid(lz)
                    ));
                }
            }
        }
    }
    let text = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {lz}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    );
    (text, removed)
}

#[test]
fn every_reference_counts_a_void_whole() {
    let dir = tempdir("voids");
    let a = 3.165;
    // Voids of 15, 55, 89 and 169 vacancies: no site of one has the five held
    // first neighbours the domain rule needs, so only the cavity test sees it.
    for (r, planted) in [(4.0, 15), (5.5, 55), (7.0, 89), (8.5, 169)] {
        let c = [10.0 * a + 0.1, 10.0 * a + 0.1, 10.0 * a + 0.1];
        let (text, removed) = holed_block(20, 20, &[(c, r)], 0.0, 0.0);
        assert_eq!(removed, planted);
        let input = dir.join(format!("void{planted}.xyz"));
        std::fs::write(&input, text).unwrap();
        for reference in ["local", "patch", "global"] {
            let (out, err, ok) =
                rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
            assert!(ok, "{out}\n{err}");
            assert_eq!(
                scalar(&out, "defects_vacancies"),
                planted as i64,
                "{planted} {reference}\n{err}"
            );
            assert_eq!(
                scalar(&out, "defects_interstitials"),
                0,
                "{planted} {reference}"
            );
        }
    }
}

#[test]
fn vacuum_gaps_and_open_pits_are_not_voids() {
    let dir = tempdir("gaps");
    let a = 3.165;
    // Three single vacancies in a periodic slab under a vacuum gap through
    // the whole box, however thin: the gap wraps round the box, so it is
    // two surfaces, not a cavity.
    let vac = [
        ([5.0 * a, 5.0 * a, 5.0 * a], 0.1),
        ([2.5 * a, 7.5 * a, 8.5 * a], 0.1),
        ([9.0 * a, 3.0 * a, 2.0 * a], 0.1),
    ];
    for gap in [4.0, 10.0, 40.0] {
        let (text, removed) = holed_block(12, 12, &vac, gap, gap / 2.0);
        assert_eq!(removed, 3);
        let input = dir.join(format!("gap{gap}.xyz"));
        std::fs::write(&input, text).unwrap();
        for reference in ["local", "patch", "global"] {
            let (out, err, ok) =
                rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
            assert!(ok, "{out}\n{err}");
            assert_eq!(
                scalar(&out, "defects_vacancies"),
                3,
                "gap {gap} {reference}\n{err}"
            );
        }
    }
    // A pit open to the top surface beside a void of 15: the void is counted
    // and nothing deep in the pit is.
    let pit = [6.0 * a + 0.1, 6.0 * a + 0.1, 11.5 * a + 0.1];
    let void = [3.0 * a + 0.1, 3.0 * a + 0.1, 4.0 * a + 0.1];
    let (text, _) = holed_block(12, 12, &[(pit, 5.0), (void, 4.0)], 8.0, 4.0);
    let input = dir.join("pit.xyz");
    std::fs::write(&input, text).unwrap();
    for reference in ["local", "patch", "global"] {
        let d = dir.join(format!("pit-{reference}"));
        let (out, err, ok) = rmda(&[
            "defects",
            input.to_str().unwrap(),
            "--reference",
            reference,
            "--defects",
            d.to_str().unwrap(),
        ]);
        assert!(ok, "{out}\n{err}");
        let text = std::fs::read_to_string(d.join("vacancies.xyz")).unwrap();
        let near = |c: [f64; 3], r: f64| {
            text.lines()
                .skip(2)
                .filter(|l| {
                    let p: Vec<f64> = l
                        .split_whitespace()
                        .skip(1)
                        .take(3)
                        .map(|x| x.parse().unwrap())
                        .collect();
                    let dz = [c[0], c[1], c[2] + 4.0];
                    (0..3).map(|x| (p[x] - dz[x]).powi(2)).sum::<f64>() < r * r
                })
                .count()
        };
        assert_eq!(near(void, 4.5), 15, "{reference}: void");
        assert_eq!(near(pit, 3.5), 0, "{reference}: inside the open pit");
    }
}

#[test]
fn a_collapsed_vacancy_loop_is_counted_by_every_reference() {
    let dir = tempdir("vloop");
    // 99 vacancies collapsed into an a[001] loop, and three single vacancies
    // away from it. Its inside is perfect crystal again, so only a lattice
    // from outside the loop's field sees its content: global, patches larger
    // than the loop, and a region round its rim.
    let vac = [(3, 3, 3, 0), (16, 4, 15, 1), (15, 10, 10, 0)];
    let (text, lacks) = dislocation_loop(20, 4.0, 0.05, &vac, true);
    assert_eq!(lacks, 99);
    let input = dir.join("vloop.xyz");
    std::fs::write(&input, text).unwrap();
    for reference in ["local", "patch", "global"] {
        let (out, err, ok) = rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(
            scalar(&out, "defects_grains_measured"),
            1,
            "{reference}\n{err}"
        );
        assert_eq!(scalar(&out, "defects_vacancies"), 102, "{reference}\n{err}");
        assert_eq!(
            scalar(&out, "defects_interstitials"),
            0,
            "{reference}\n{err}"
        );
    }
}

#[test]
fn patches_smaller_than_a_loop_are_refused_not_miscounted() {
    let dir = tempdir("smallpatch");
    // With 20 Å patches a 99-vacancy loop spans several: placed from one
    // another every patch agrees, but two faces off the placement tree meet
    // a Burgers vector apart, and counting them anyway gives 47 V / 6 I.
    let vac = [(3, 3, 3, 0), (16, 4, 15, 1), (15, 10, 10, 0)];
    let (text, _) = dislocation_loop(20, 4.0, 0.05, &vac, true);
    let input = dir.join("vloop.xyz");
    std::fs::write(&input, text).unwrap();
    let args = |reference: &str| {
        let (out, err, ok) = rmda(&[
            "defects",
            input.to_str().unwrap(),
            "--reference",
            reference,
            "--patch-size",
            "20",
        ]);
        assert!(ok, "{out}\n{err}");
        (out, err)
    };
    let (out, err) = args("patch");
    assert_eq!(scalar(&out, "defects_grains_measured"), 0, "{err}");
    assert!(err.contains("gate_"), "{err}");
    // Auto keeps the global reference, which counts it: patches are only
    // tried when global fails its gates.
    let (out, err) = args("auto");
    assert_eq!(scalar(&out, "defects_grains_measured"), 1, "{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 102, "{err}");
    assert_eq!(scalar(&out, "defects_interstitials"), 0, "{err}");
}

#[test]
fn each_grain_clusters_its_defects_with_its_own_cutoff() {
    let dir = tempdir("cutoffs");
    // A BCC W block (the larger grain) and an FCC Pb block beside it, each
    // with a first-neighbour divacancy: 2.74 Å in W, 3.50 Å in Pb. W's
    // cutoff (2.95 Å) would split Pb's pair; Pb's own (4.22 Å) joins it.
    let mut rows = Vec::new();
    let w = 3.165;
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                for (b, o) in [[0.0; 3], [0.5; 3]].iter().enumerate() {
                    if (i, j, k, b) == (5, 5, 5, 0) || (i, j, k, b) == (5, 5, 5, 1) {
                        continue;
                    }
                    let p = [
                        (i as f64 + o[0]) * w,
                        (j as f64 + o[1]) * w,
                        (k as f64 + o[2]) * w,
                    ];
                    rows.push(format!("W {} {} {}", p[0] + 10.0, p[1] + 10.0, p[2] + 10.0));
                }
            }
        }
    }
    let pb = 4.95;
    let fcc = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                for (b, o) in fcc.iter().enumerate() {
                    if (i, j, k, b) == (3, 3, 3, 0) || (i, j, k, b) == (3, 3, 3, 3) {
                        continue;
                    }
                    let p = [
                        (i as f64 + o[0]) * pb,
                        (j as f64 + o[1]) * pb,
                        (k as f64 + o[2]) * pb,
                    ];
                    rows.push(format!(
                        "Pb {} {} {}",
                        p[0] + 60.0,
                        p[1] + 10.0,
                        p[2] + 10.0
                    ));
                }
            }
        }
    }
    let input = dir.join("two.xyz");
    std::fs::write(
        &input,
        format!(
            "{}\nLattice=\"110 0 0 0 60 0 0 0 60\" pbc=\"F F F\" Properties=species:S:1:pos:R:3\n{}\n",
            rows.len(),
            rows.join("\n")
        ),
    )
    .unwrap();
    let file = input.to_str().unwrap();
    let regions = ["--region", "x < 50", "--region", "x > 50"];
    let (out, err, ok) = rmda(&[&["defects", file][..], &regions].concat());
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_grains_measured"), 2, "{err}");
    assert_eq!(scalar(&out, "defects_vacancies"), 4, "{err}");
    assert_eq!(scalar(&out, "defects_clusters"), 2, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_largest_cluster"), 2, "{out}");
    // One cutoff for both, as `--cluster-cutoff` gives: W's splits Pb's pair.
    let (out, err, ok) =
        rmda(&[&["defects", file, "--cluster-cutoff", "2.95"][..], &regions].concat());
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "defects_clusters"), 3, "{out}");
}

/// The extxyz `text` with its atom rows in the order `perm` gives.
fn permuted(text: &str, perm: &[usize]) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let rows = &lines[2..];
    assert_eq!(rows.len(), perm.len());
    let mut out = format!("{}\n{}\n", lines[0], lines[1]);
    for &p in perm {
        out.push_str(rows[p]);
        out.push('\n');
    }
    out
}

#[test]
fn every_reference_counts_the_same_whatever_the_order_of_atoms_in_the_file() {
    // Which region is accepted first, which atoms sample a fit and where a
    // chunk's lattice starts all follow positions, not the file: the loop
    // and its vacancies count the same written in lattice order, reversed,
    // or shuffled. (Before, local lost the loop in about half of shuffles.)
    let dir = tempdir("order");
    let vac = [
        (14, 10, 10, 0),
        (15, 10, 10, 0),
        (10, 15, 11, 0),
        (10, 10, 12, 0),
        (12, 8, 9, 1),
        (13, 13, 10, 1),
        (3, 3, 3, 0),
        (16, 4, 15, 1),
    ];
    let (text, held) = loop_block(20, 4.0, 0.08, &vac);
    let n: usize = text.lines().next().unwrap().trim().parse().unwrap();
    let mut shuffled: Vec<usize> = (0..n).collect();
    let mut seed: u64 = 99;
    for i in (1..n).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        shuffled.swap(i, (seed >> 33) as usize % (i + 1));
    }
    let orders = [
        ("lattice", (0..n).collect::<Vec<_>>()),
        ("reversed", (0..n).rev().collect()),
        ("shuffled", shuffled),
    ];
    for reference in ["global", "patch", "local"] {
        let mut seen: Option<String> = None;
        for (tag, perm) in &orders {
            let input = dir.join(format!("{tag}.xyz"));
            std::fs::write(&input, permuted(&text, perm)).unwrap();
            let (out, err, ok) =
                rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
            assert!(ok, "{out}\n{err}");
            if reference != "global" {
                assert_eq!(
                    (
                        scalar(&out, "defects_vacancies"),
                        scalar(&out, "defects_interstitials")
                    ),
                    (8, held as i64),
                    "{reference} {tag}\n{err}"
                );
            }
            match &seen {
                None => seen = Some(out),
                Some(first) => assert_eq!(first, &out, "{reference}: {tag} differs"),
            }
        }
    }
}

/// FCC in its rhombohedral primitive cell (60° between the cell vectors), a
/// strongly triclinic periodic box, with one vacancy and with a vacancy, an
/// interstitial and noise: every reference counts them exactly. The local
/// census bins its empty sites in fractional coordinates; binned in
/// Cartesian coordinates, a site on a cell face was counted once per periodic
/// image (one vacancy read as three).
#[test]
fn every_reference_counts_exactly_in_a_rhombohedral_cell() {
    let a = 3.615;
    let n = 10;
    let h = a / 2.0;
    let cellv = [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]];
    let mut sites = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                sites.push(std::array::from_fn::<f64, 3, _>(|d| {
                    i as f64 * cellv[0][d] + j as f64 * cellv[1][d] + k as f64 * cellv[2][d]
                }));
            }
        }
    }
    let mut seed: u64 = 17;
    let mut noise = |s: f64| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * s
    };
    let dir = tempdir("rhombohedral");
    let lat = |v: f64| format!("{v}");
    let header = (0..3)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| lat(cellv[r][c] * n as f64))
        .collect::<Vec<_>>()
        .join(" ");
    for (case, removed, extra, sigma) in [
        ("one", vec![5usize], vec![], 0.0),
        (
            "mixed",
            vec![5, 600],
            vec![[0.5 * h, 0.31 * h, 0.17 * h]],
            0.04,
        ),
    ] {
        let mut rows: Vec<[f64; 3]> = sites
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(_, p)| *p)
            .collect();
        for e in &extra {
            let host = sites[300];
            rows.push([host[0] + e[0], host[1] + e[1], host[2] + e[2]]);
        }
        let mut text = format!(
            "{}\nLattice=\"{header}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
            rows.len()
        );
        for p in &rows {
            text.push_str(&format!(
                "Cu {} {} {}\n",
                p[0] + noise(sigma),
                p[1] + noise(sigma),
                p[2] + noise(sigma)
            ));
        }
        let input = dir.join(format!("{case}.xyz"));
        std::fs::write(&input, text).unwrap();
        for reference in ["auto", "global", "patch", "local", "periodic"] {
            let (out, err, ok) =
                rmda(&["defects", input.to_str().unwrap(), "--reference", reference]);
            assert!(ok, "{case} {reference}\n{out}\n{err}");
            let tag = format!("{case} {reference}\n{out}\n{err}");
            assert_eq!(
                scalar(&out, "defects_vacancies"),
                removed.len() as i64,
                "{tag}"
            );
            assert_eq!(
                scalar(&out, "defects_interstitials"),
                extra.len() as i64,
                "{tag}"
            );
        }
    }
}
