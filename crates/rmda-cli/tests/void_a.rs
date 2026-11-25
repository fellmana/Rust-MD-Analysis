//! `rmda void` method A (design §17): voids as clusters of the census's
//! vacancy sites. Synthetic BCC W with spherical vacancy clusters of known
//! size: counts and volumes exact, in bulk, under a free surface, in a
//! bicrystal, and whatever the order of atoms in the file.

use std::path::PathBuf;
use std::process::Command;

const A: f64 = 3.165;

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

fn value(stdout: &str, name: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no {name} in:\n{stdout}"))
        .trim()
        .parse()
        .unwrap()
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_void_a_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic Gaussian noise of width `sigma`.
struct Noise(u64, f64);
impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        if self.1 == 0.0 {
            return 0.0;
        }
        let (u, v) = (self.uniform(), self.uniform());
        self.1 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Atoms of a BCC block `n`×`n`×`nz` cells (rotated about z by `rot`, a
/// lattice-to-box map that must tile the box), lifted by `dz`, with the
/// sites inside any hole removed; returns rows and the removed count per hole.
#[allow(clippy::too_many_arguments)]
fn bcc_rows(
    rot: Option<([i32; 2], [i32; 2])>,
    l: f64,
    z_range: (f64, f64),
    holes: &[([f64; 3], f64)],
    dz: f64,
    noise: &mut Noise,
    rows: &mut Vec<[f64; 3]>,
) -> Vec<usize> {
    let mut removed = vec![0; holes.len()];
    // Lattice vectors in the plane (in units of a) spanning the box sides.
    let (u, v) = rot.unwrap_or(([1, 0], [0, 1]));
    let su = ((u[0] * u[0] + u[1] * u[1]) as f64).sqrt();
    let (c, s) = (u[0] as f64 / su, u[1] as f64 / su);
    let _ = v;
    let m = (l / A * 2.0).ceil() as i32 + 2;
    let mut seen = std::collections::HashSet::new();
    let kz0 = (z_range.0 / A).floor() as i32 - 1;
    let kz1 = (z_range.1 / A).ceil() as i32 + 1;
    for i in -m..m {
        for j in -m..m {
            for k in kz0..kz1 {
                for o in [0.0, 0.5] {
                    let f = [i as f64 + o, j as f64 + o, k as f64 + o];
                    let p = [
                        (c * f[0] - s * f[1]) * A,
                        (s * f[0] + c * f[1]) * A,
                        f[2] * A,
                    ];
                    if p[2] < z_range.0 - 1e-6 || p[2] >= z_range.1 - 1e-6 {
                        continue;
                    }
                    let w = [p[0].rem_euclid(l), p[1].rem_euclid(l), p[2]];
                    let w = [
                        if (w[0] - l).abs() < 1e-6 { 0.0 } else { w[0] },
                        if (w[1] - l).abs() < 1e-6 { 0.0 } else { w[1] },
                        w[2],
                    ];
                    let key = [
                        (w[0] * 1e4).round() as i64,
                        (w[1] * 1e4).round() as i64,
                        (w[2] * 1e4).round() as i64,
                    ];
                    if !seen.insert(key) {
                        continue;
                    }
                    let mut hit = false;
                    for (h, (cen, r)) in holes.iter().enumerate() {
                        let d2: f64 = (0..3)
                            .map(|x| {
                                let mut d = w[x] - cen[x];
                                if x < 2 {
                                    d -= l * (d / l).round();
                                }
                                d * d
                            })
                            .sum();
                        if d2 < r * r {
                            removed[h] += 1;
                            hit = true;
                        }
                    }
                    if !hit {
                        rows.push([
                            w[0] + noise.gauss(),
                            w[1] + noise.gauss(),
                            w[2] + dz + noise.gauss(),
                        ]);
                    }
                }
            }
        }
    }
    removed
}

fn extxyz(rows: &[[f64; 3]], l: f64, lz: f64) -> String {
    let mut s = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {lz}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n",
        rows.len()
    );
    for p in rows {
        s.push_str(&format!(
            "W {} {} {}\n",
            p[0].rem_euclid(l),
            p[1].rem_euclid(l),
            p[2].rem_euclid(lz)
        ));
    }
    s
}

/// Per-void rows of a `--table` CSV: (sites, volume, class).
fn table(path: &std::path::Path) -> Vec<(u32, f64, String)> {
    let t = std::fs::read_to_string(path).unwrap();
    let mut lines = t.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |n: &str| head.iter().position(|c| *c == n).unwrap();
    lines
        .map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            (
                c[col("sites")].parse().unwrap(),
                c[col("volume")].parse().unwrap(),
                c[col("class")].to_owned(),
            )
        })
        .collect()
}

/// Four voids of 15, 55, 89 and 169 sites, far apart in a 24-cell block.
fn four_voids(sigma: f64) -> (String, Vec<usize>) {
    let n = 24;
    let l = n as f64 * A;
    let holes = [
        ([6.0 * A + 0.1, 6.0 * A + 0.1, 6.0 * A + 0.1], 4.0),
        ([18.0 * A + 0.1, 6.0 * A + 0.1, 6.0 * A + 0.1], 5.5),
        ([6.0 * A + 0.1, 18.0 * A + 0.1, 18.0 * A + 0.1], 7.0),
        ([18.0 * A + 0.1, 18.0 * A + 0.1, 17.0 * A + 0.1], 8.5),
    ];
    let mut rows = Vec::new();
    let removed = bcc_rows(
        None,
        l,
        (0.0, l),
        &holes,
        0.0,
        &mut Noise(7, sigma),
        &mut rows,
    );
    (extxyz(&rows, l, l), removed)
}

#[test]
fn planted_voids_are_counted_whole_with_exact_volumes_by_every_reference() {
    let dir = tempdir("bulk");
    let omega = A * A * A / 2.0;
    for sigma in [0.0, 0.05] {
        let (text, removed) = four_voids(sigma);
        assert_eq!(removed, [15, 55, 89, 169]);
        let input = dir.join(format!("four-{sigma}.xyz"));
        std::fs::write(&input, &text).unwrap();
        for reference in ["auto", "global", "patch", "local"] {
            let t = dir.join(format!("t-{sigma}-{reference}.csv"));
            let (out, err, ok) = rmda(&[
                "void",
                input.to_str().unwrap(),
                "--site-reference",
                reference,
                "--table",
                t.to_str().unwrap(),
            ]);
            assert!(ok, "{out}\n{err}");
            let tag = format!("sigma {sigma} {reference}\n{out}\n{err}");
            assert_eq!(value(&out, "voids"), 4.0, "{tag}");
            assert_eq!(value(&out, "voids_bulk"), 4.0, "{tag}");
            assert_eq!(value(&out, "void_sites"), 328.0, "{tag}");
            assert_eq!(value(&out, "void_isolated_vacancies"), 0.0, "{tag}");
            let rows = table(&t);
            let sizes: Vec<u32> = rows.iter().map(|r| r.0).collect();
            assert_eq!(sizes, [169, 89, 55, 15], "{tag}");
            // det T / 2 of the fitted cell: exact on the perfect lattice for
            // global and patch; the local reference's cell is its grain-wide
            // prefit, fitted to the lattice atoms around the defects.
            let tol = match (sigma == 0.0, reference) {
                (true, "local") => 1e-3,
                (true, _) => 1e-6,
                _ => 5e-3,
            };
            for (n, vol, class) in &rows {
                let want = *n as f64 * omega;
                assert!(
                    (vol - want).abs() <= tol * want,
                    "{tag}: {n} sites {vol} vs {want}"
                );
                assert_eq!(class, "bulk", "{tag}");
            }
        }
    }
    // The explicit reference (the perfect block's sites) counts the same.
    let (perfect, _) = {
        let l = 24.0 * A;
        let mut rows = Vec::new();
        bcc_rows(None, l, (0.0, l), &[], 0.0, &mut Noise(1, 0.0), &mut rows);
        (extxyz(&rows, l, l), ())
    };
    let r = dir.join("perfect.xyz");
    std::fs::write(&r, perfect).unwrap();
    let input = dir.join("four-0.xyz");
    let t = dir.join("t-ws.csv");
    let (out, err, ok) = rmda(&[
        "void",
        input.to_str().unwrap(),
        "--ws-reference",
        r.to_str().unwrap(),
        "--table",
        t.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(value(&out, "voids"), 4.0, "{out}");
    assert_eq!(value(&out, "voids_bulk"), 4.0, "{out}");
    for (n, vol, _) in table(&t) {
        let want = n as f64 * omega;
        assert!((vol - want).abs() <= 1e-3 * want, "ws: {n} {vol} {want}");
    }
}

#[test]
fn min_sites_and_its_alias_decide_what_is_a_void() {
    let dir = tempdir("minsites");
    let l = 12.0 * A;
    let mut rows = Vec::new();
    // A single vacancy, a divacancy (first neighbours) and a 15-site void.
    let holes = [
        ([2.0 * A, 2.0 * A, 2.0 * A], 0.1),
        ([8.0 * A, 2.0 * A, 2.0 * A], 0.1),
        ([8.5 * A, 2.5 * A, 2.5 * A], 0.1),
        ([6.0 * A + 0.1, 8.0 * A + 0.1, 8.0 * A + 0.1], 4.0),
    ];
    let removed = bcc_rows(
        None,
        l,
        (0.0, l),
        &holes,
        0.0,
        &mut Noise(1, 0.0),
        &mut rows,
    );
    assert_eq!(removed, [1, 1, 1, 15]);
    let input = dir.join("m.xyz");
    std::fs::write(&input, extxyz(&rows, l, l)).unwrap();
    let f = input.to_str().unwrap();
    let (out, err, ok) = rmda(&["void", f]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(value(&out, "void_vacancies"), 18.0, "{out}");
    assert_eq!(value(&out, "voids"), 2.0, "{out}");
    assert_eq!(value(&out, "void_isolated_vacancies"), 1.0, "{out}");
    for (flag, min, voids) in [("--min-sites", "1", 3.0), ("--void-min-sites", "16", 0.0)] {
        let (out, err, ok) = rmda(&["void", f, flag, min]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(value(&out, "voids"), voids, "{flag} {min}\n{out}");
    }
    // Method B's options need --experimental, method A's refuse it.
    for args in [
        vec!["void", f, "--samples", "100"],
        vec!["void", f, "--lattice", "bcc"],
        vec!["void", f, "--experimental", "--min-sites", "3"],
        vec!["void", f, "--experimental", "--site-reference", "global"],
    ] {
        let (out, err, ok) = rmda(&args);
        assert!(!ok, "{args:?} accepted:\n{out}");
        assert!(err.contains("experimental"), "{args:?}: {err}");
    }
    let (_, err, ok) = rmda(&["void", f, "-o", "x.xyz"]);
    assert!(!ok && err.contains("--table"), "{err}");
}

#[test]
fn a_free_surface_opens_a_pit_but_not_a_closed_void_under_it() {
    let dir = tempdir("surface");
    let (n, nz, gap) = (14usize, 12usize, 12.0);
    let l = n as f64 * A;
    let top = (nz as f64 - 0.5) * A; // top atomic layer
    let holes = [
        // Closed void deep in the slab.
        ([4.0 * A + 0.1, 4.0 * A + 0.1, 4.0 * A + 0.1], 5.5),
        // A hemispherical pit cut into the top surface.
        ([10.0 * A + 0.1, 10.0 * A + 0.1, top + 0.1], 7.0),
        // A closed void whose top site is two atomic layers below the surface.
        ([4.5 * A + 0.1, 10.5 * A + 0.1, top - 2.0 * A + 0.1], 4.0),
    ];
    let mut rows = Vec::new();
    let removed = bcc_rows(
        None,
        l,
        (0.0, nz as f64 * A),
        &holes,
        gap / 2.0,
        &mut Noise(3, 0.0),
        &mut rows,
    );
    assert_eq!((removed[0], removed[2]), (55, 15));
    let input = dir.join("slab.xyz");
    std::fs::write(&input, extxyz(&rows, l, nz as f64 * A + gap)).unwrap();
    let omega = A * A * A / 2.0;
    for reference in ["global", "patch", "local"] {
        let t = dir.join(format!("{reference}.csv"));
        let s = dir.join(format!("{reference}.xyz"));
        let (out, err, ok) = rmda(&[
            "void",
            input.to_str().unwrap(),
            "--site-reference",
            reference,
            "--table",
            t.to_str().unwrap(),
            "--sites",
            s.to_str().unwrap(),
            "--min-sites",
            "1",
        ]);
        assert!(ok, "{out}\n{err}");
        let rows = table(&t);
        let bulk: Vec<u32> = rows.iter().filter(|r| r.2 == "bulk").map(|r| r.0).collect();
        assert_eq!(bulk, [55, 15], "{reference}\n{out}\n{rows:?}");
        for (n, vol, class) in &rows {
            if class == "bulk" {
                let want = *n as f64 * omega;
                assert!((vol - want).abs() <= 1e-3 * want, "{reference}: {vol}");
            } else {
                assert_eq!(class, "surface", "{reference}: {rows:?}");
            }
        }
        let sites = std::fs::read_to_string(&s).unwrap();
        let count: usize = sites.lines().next().unwrap().parse().unwrap();
        assert_eq!(
            count as u32,
            rows.iter().map(|r| r.0).sum::<u32>(),
            "{reference}"
        );
        assert!(sites.lines().nth(1).unwrap().contains("void_id:I:1"));
        // A fitted reference counts nothing in the open pit (§16.3): it is
        // outside the counted domain.
        assert!(rows.iter().all(|r| r.2 == "bulk"), "{reference}: {rows:?}");
    }
    // The explicit reference counts every one of its sites, so the pit is a
    // void, with its full volume, and open: a surface void.
    let mut perfect = Vec::new();
    bcc_rows(
        None,
        l,
        (0.0, nz as f64 * A),
        &[],
        gap / 2.0,
        &mut Noise(3, 0.0),
        &mut perfect,
    );
    let r = dir.join("perfect.xyz");
    std::fs::write(&r, extxyz(&perfect, l, nz as f64 * A + gap)).unwrap();
    let t = dir.join("ws.csv");
    let (out, err, ok) = rmda(&[
        "void",
        input.to_str().unwrap(),
        "--ws-reference",
        r.to_str().unwrap(),
        "--table",
        t.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    let rows = table(&t);
    let mut want = vec![
        (removed[1] as u32, "surface"),
        (removed[0] as u32, "bulk"),
        (removed[2] as u32, "bulk"),
    ];
    want.sort_by_key(|e| std::cmp::Reverse(e.0));
    let got: Vec<(u32, &str)> = rows.iter().map(|r| (r.0, r.2.as_str())).collect();
    assert_eq!(got, want, "{out}");
    assert_eq!(value(&out, "voids_surface"), 1.0);
    for (n, vol, _) in &rows {
        let w = *n as f64 * omega;
        assert!((vol - w).abs() <= 1e-3 * w, "ws: {vol} {w}");
    }
}

/// A Σ5 twist bicrystal: two BCC grains rotated ±26.57° about z (lattice
/// vectors (2,1) and (2,−1) along x), stacked along z, with voids in each.
fn bicrystal(sigma: f64) -> (String, f64, Vec<usize>) {
    let m = 4; // box side = m·a·√5
    let l = m as f64 * A * 5f64.sqrt();
    let lz = 20.0 * A;
    let half = lz / 2.0;
    let mut rows = Vec::new();
    let mut noise = Noise(11, sigma);
    let mut removed = bcc_rows(
        Some(([2, 1], [-1, 2])),
        l,
        (0.0, half),
        &[([l / 2.0 + 0.1, l / 2.0 + 0.1, half / 2.0 + 0.1], 5.5)],
        0.0,
        &mut noise,
        &mut rows,
    );
    removed.extend(bcc_rows(
        Some(([2, -1], [1, 2])),
        l,
        (half, lz),
        &[([l / 4.0 + 0.1, l / 4.0 + 0.1, 1.5 * half + 0.1], 7.0)],
        0.0,
        &mut noise,
        &mut rows,
    ));
    (extxyz(&rows, l, lz), half, removed)
}

#[test]
fn each_grain_of_a_bicrystal_counts_its_voids() {
    let dir = tempdir("bicrystal");
    let omega = A * A * A / 2.0;
    for sigma in [0.0, 0.05] {
        let (text, half, removed) = bicrystal(sigma);
        let input = dir.join(format!("bi-{sigma}.xyz"));
        std::fs::write(&input, text).unwrap();
        let lo = format!("z < {half}");
        let hi = format!("z >= {half}");
        let t = dir.join(format!("bi-{sigma}.csv"));
        let (out, err, ok) = rmda(&[
            "void",
            input.to_str().unwrap(),
            "--grain-region",
            &lo,
            "--grain-region",
            &hi,
            "--table",
            t.to_str().unwrap(),
        ]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(value(&out, "void_grains_measured"), 2.0, "{out}\n{err}");
        let rows = table(&t);
        let mut want = removed.clone();
        want.sort_unstable_by(|a, b| b.cmp(a));
        let sizes: Vec<usize> = rows.iter().map(|r| r.0 as usize).collect();
        assert_eq!(sizes, want, "sigma {sigma}\n{out}\n{err}");
        for (n, vol, class) in &rows {
            assert_eq!(class, "bulk");
            let w = *n as f64 * omega;
            assert!((vol - w).abs() <= 5e-3 * w, "{vol} {w}");
        }
        // Segmentation finds the same grains.
        let (out2, err2, ok) = rmda(&["void", input.to_str().unwrap(), "--segment"]);
        assert!(ok, "{out2}\n{err2}");
        assert_eq!(value(&out2, "voids"), 2.0, "{out2}\n{err2}");
        assert_eq!(value(&out2, "void_sites"), (removed[0] + removed[1]) as f64);
    }
}

/// The extxyz `text` with its atom rows in the order `perm` gives.
fn permuted(text: &str, perm: &[usize]) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let rows = &lines[2..];
    let mut out = format!("{}\n{}\n", lines[0], lines[1]);
    for &p in perm {
        out.push_str(rows[p]);
        out.push('\n');
    }
    out
}

#[test]
fn voids_are_the_same_whatever_the_order_of_atoms_in_the_file() {
    let dir = tempdir("order");
    let (text, _) = four_voids(0.05);
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
        let mut seen: Option<(String, String)> = None;
        for (tag, perm) in &orders {
            let input = dir.join(format!("{tag}.xyz"));
            std::fs::write(&input, permuted(&text, perm)).unwrap();
            let t = dir.join(format!("{tag}-{reference}.csv"));
            let (out, err, ok) = rmda(&[
                "void",
                input.to_str().unwrap(),
                "--site-reference",
                reference,
                "--table",
                t.to_str().unwrap(),
            ]);
            assert!(ok, "{out}\n{err}");
            assert_eq!(value(&out, "void_sites"), 328.0, "{reference} {tag}");
            let csv = std::fs::read_to_string(&t).unwrap();
            match &seen {
                None => seen = Some((out, csv)),
                Some((o, c)) => {
                    assert_eq!(o, &out, "{reference}: {tag} differs");
                    assert_eq!(c, &csv, "{reference}: {tag} table differs");
                }
            }
        }
    }
}

#[test]
fn a_series_counts_each_frame_as_a_single_run_resumes_and_refuses_other_settings() {
    let dir = tempdir("series");
    let frames = dir.join("frames");
    std::fs::create_dir_all(&frames).unwrap();
    // Three frames: the four voids perfect and noisy, and a perfect block
    // (`data.firstframe` has no index and comes first, as -1).
    let (clean, _) = four_voids(0.0);
    let (noisy, _) = four_voids(0.05);
    let perfect = {
        let l = 24.0 * A;
        let mut rows = Vec::new();
        bcc_rows(None, l, (0.0, l), &[], 0.0, &mut Noise(1, 0.0), &mut rows);
        extxyz(&rows, l, l)
    };
    let files = [
        (frames.join("data.10"), noisy),
        (frames.join("data.2"), clean),
        (frames.join("data.firstframe"), perfect),
    ];
    for (p, t) in &files {
        std::fs::write(p, t).unwrap();
    }
    let out_dir = dir.join("out");
    let _ = std::fs::remove_dir_all(&out_dir);
    let series = |extra: &[&str]| {
        let mut args = vec!["void", "--out", out_dir.to_str().unwrap()];
        args.extend(files.iter().map(|(p, _)| p.to_str().unwrap()));
        args.extend_from_slice(extra);
        rmda(&args)
    };
    let (out, err, ok) = series(&[]);
    assert!(ok, "{out}\n{err}");

    let summary = std::fs::read_to_string(out_dir.join("summary.csv")).unwrap();
    let rows: Vec<Vec<&str>> = summary
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    let head: Vec<&str> = summary.lines().next().unwrap().split(',').collect();
    let col = |n: &str| head.iter().position(|c| *c == n).unwrap();
    let index: Vec<&str> = rows.iter().map(|r| r[col("file_index")]).collect();
    assert_eq!(index, ["-1", "2", "10"], "ordered by file index\n{summary}");
    let voids: Vec<&str> = rows.iter().map(|r| r[col("voids")]).collect();
    assert_eq!(voids, ["0", "4", "4"], "{summary}");
    let sites: Vec<&str> = rows.iter().map(|r| r[col("void_sites")]).collect();
    assert_eq!(sites, ["0", "328", "328"], "{summary}");

    // Each frame's rows of voids.csv are its single-file table, byte for
    // byte, behind the series key; its summary values are the single run's.
    let voids_csv = std::fs::read_to_string(out_dir.join("voids.csv")).unwrap();
    for (p, _) in &files[..2] {
        let t = dir.join("single.csv");
        let (single, err, ok) =
            rmda(&["void", p.to_str().unwrap(), "--table", t.to_str().unwrap()]);
        assert!(ok, "{single}\n{err}");
        let k = p.extension().unwrap().to_str().unwrap();
        let name = p.file_name().unwrap().to_str().unwrap();
        // Untracked, `stable_grain` (after `grain`) is the label.
        let mine: Vec<String> = voids_csv
            .lines()
            .filter(|l| l.starts_with(&format!("{k},{name},")))
            .map(|l| {
                let mut f: Vec<&str> = l.split(',').skip(3).collect();
                assert_eq!(f[7], f[8], "{l}");
                f.remove(8);
                f.join(",")
            })
            .collect();
        let table: Vec<String> = std::fs::read_to_string(&t)
            .unwrap()
            .lines()
            .skip(1)
            .map(str::to_owned)
            .collect();
        assert_eq!(mine, table, "{name}");
        let row = rows.iter().find(|r| r[col("file_index")] == k).unwrap();
        for (c, key) in [
            ("volume", "void_volume"),
            ("vacancies", "void_vacancies"),
            ("largest_radius", "void_largest_radius"),
            ("cutoff", "void_cutoff"),
        ] {
            assert_eq!(
                row[col(c)].parse::<f64>().unwrap(),
                value(&single, key),
                "{name}: {c}"
            );
        }
    }

    // A rerun skips every frame and leaves the tables as they were.
    let (_, err, ok) = series(&[]);
    assert!(ok, "{err}");
    assert_eq!(err.matches("already done, skipping").count(), 3, "{err}");
    assert_eq!(
        std::fs::read_to_string(out_dir.join("voids.csv")).unwrap(),
        voids_csv
    );
    let after = std::fs::read_to_string(out_dir.join("summary.csv")).unwrap();
    assert_eq!(after, summary);

    // Resume: drop the last frame's rows (as if interrupted) and rerun.
    let cut: String = summary
        .lines()
        .filter(|l| !l.starts_with("10,"))
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(out_dir.join("summary.csv"), cut).unwrap();
    let (_, err, ok) = series(&[]);
    assert!(ok, "{err}");
    assert_eq!(err.matches("already done, skipping").count(), 2, "{err}");
    let resumed = std::fs::read_to_string(out_dir.join("summary.csv")).unwrap();
    assert_eq!(resumed.lines().count(), 4, "{resumed}");

    // Other settings are refused, not mixed; --overwrite starts again.
    let (_, err, ok) = series(&["--min-sites", "20"]);
    assert!(!ok && err.contains("different settings"), "{err}");
    let (_, err, ok) = series(&["--min-sites", "20", "--overwrite"]);
    assert!(ok, "{err}");
    let s = std::fs::read_to_string(out_dir.join("summary.csv")).unwrap();
    let v: Vec<&str> = s
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(col("voids")).unwrap())
        .collect();
    assert_eq!(v, ["0", "3", "3"], "{s}");

    // A defects series directory is not appended to.
    let d = dir.join("defects_out");
    let _ = std::fs::remove_dir_all(&d);
    let (_, err, ok) = rmda(&[
        "defects",
        files[1].0.to_str().unwrap(),
        "--out",
        d.to_str().unwrap(),
    ]);
    assert!(ok, "{err}");
    let (_, err, ok) = rmda(&[
        "void",
        files[1].0.to_str().unwrap(),
        "--out",
        d.to_str().unwrap(),
    ]);
    assert!(!ok && err.contains("different settings"), "{err}");

    // Several files need --out; --out is method A's and takes no --table.
    let (_, _, ok) = rmda(&[
        "void",
        files[0].0.to_str().unwrap(),
        files[1].0.to_str().unwrap(),
    ]);
    assert!(!ok);
    let (_, err, ok) = series(&["--table", "x.csv"]);
    assert!(!ok && err.contains("cannot be used with"), "{err}");
    let (_, err, ok) = series(&["--experimental"]);
    assert!(!ok && err.contains("cannot be used with"), "{err}");
}

/// A free BCC slab bent ±8° about y (16° of misorientation across the grain)
/// with three closed voids of 59 sites, under the local reference. Voids are
/// classified against the grain's single fitted cell; its 0.45-bond tolerance
/// on first-shell positions has to absorb the bend, so every void is found
/// whole and closed.
#[test]
fn closed_voids_in_a_bent_grain_stay_closed_under_the_local_reference() {
    let (nx, ny, nz) = (40, 12, 8);
    let (lx, ly, lz) = (nx as f64 * A, ny as f64 * A, nz as f64 * A);
    let holes: Vec<([f64; 3], f64)> = [0.25, 0.5, 0.75]
        .iter()
        .map(|&f| ([lx * f, ly / 2.0, lz / 2.0], 5.5))
        .collect();
    let radius = (lx / 2.0) / 8f64.to_radians();
    let mut noise = Noise(11, 0.04);
    let mut rows = Vec::new();
    let mut removed = vec![0usize; holes.len()];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                for o in [0.0, 0.5] {
                    let p = [(i as f64 + o) * A, (j as f64 + o) * A, (k as f64 + o) * A];
                    let hit = holes.iter().position(|(c, r)| {
                        (0..3).map(|d| (p[d] - c[d]).powi(2)).sum::<f64>() < r * r
                    });
                    if let Some(h) = hit {
                        removed[h] += 1;
                        continue;
                    }
                    // Bend about the slab's mid-plane, centred along x.
                    let (x, z) = (p[0] - lx / 2.0, p[2] - lz / 2.0);
                    let th = x / radius;
                    rows.push([
                        (radius + z) * th.sin() + lx / 2.0 + 30.0 + noise.gauss(),
                        p[1] + noise.gauss(),
                        (radius + z) * th.cos() - radius + lz / 2.0 + 40.0 + noise.gauss(),
                    ]);
                }
            }
        }
    }
    assert_eq!(removed, [59, 59, 59]);
    let mut text = format!(
        "{}\nLattice=\"{} 0 0 0 {ly} 0 0 0 {}\" pbc=\"F T F\" Properties=species:S:1:pos:R:3\n",
        rows.len(),
        lx + 60.0,
        lz + 80.0
    );
    for p in &rows {
        text.push_str(&format!("W {} {} {}\n", p[0], p[1].rem_euclid(ly), p[2]));
    }
    let dir = tempdir("bent");
    let input = dir.join("bent.xyz");
    std::fs::write(&input, text).unwrap();
    let t = dir.join("bent.csv");
    let (out, err, ok) = rmda(&[
        "void",
        input.to_str().unwrap(),
        "--site-reference",
        "local",
        "--table",
        t.to_str().unwrap(),
    ]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(value(&out, "voids"), 3.0, "{out}\n{err}");
    assert_eq!(value(&out, "voids_bulk"), 3.0, "{out}\n{err}");
    assert_eq!(value(&out, "void_sites"), 177.0, "{out}\n{err}");
    assert_eq!(value(&out, "void_isolated_vacancies"), 0.0, "{out}\n{err}");
}
