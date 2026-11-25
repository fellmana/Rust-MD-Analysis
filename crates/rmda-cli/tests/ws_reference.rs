//! `rmda defects --ws-reference`: explicit-reference Wigner–Seitz on the census
//! outputs (design §16.3a, §16.8).
//!
//! 1. Synthetic BCC with planted vacancies, ⟨110⟩ dumbbells and a 15-site
//!    void, thermal noise, and a homogeneously strained current box: exact
//!    counts and exact vacant/interstitial site ids, with each affine mapping.
//! 2. OVITO parity: a deterministic fixture (thermal reference, a heavily
//!    displaced current frame, removed and randomly inserted atoms; same box,
//!    strained triclinic box, open z) whose OVITO 3.15
//!    `WignerSeitzAnalysisModifier` result is pinned site for site in
//!    `tests/data/ws_reference_ovito.csv`. `RMDA_WS_FIXTURE_DIR=dir` keeps the
//!    frames there; regenerate the expected lines with
//!    `python3 tools/ws_reference_ovito.py CUR REF --counts [off|to-reference]`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

type V3 = [f64; 3];
/// (id, position) per atom.
type Atoms = Vec<(i64, V3)>;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        let (u, v) = (self.f().max(1e-300), self.f());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Perfect BCC, `n³` cells: sites in (cell, basis) order, ids 1-based.
fn bcc(n: usize, a: f64) -> Vec<V3> {
    let mut p = Vec::with_capacity(2 * n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for b in [0.0, 0.5] {
                    p.push([(i as f64 + b) * a, (j as f64 + b) * a, (k as f64 + b) * a]);
                }
            }
        }
    }
    p
}

/// A box `F · diag(l)` (F upper triangular: only xz tilt allowed here).
#[derive(Clone, Copy)]
struct Box3 {
    l: V3,
    f: [[f64; 3]; 3],
    pbc_z: bool,
}

impl Box3 {
    fn ortho(l: f64) -> Self {
        Box3 {
            l: [l; 3],
            f: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc_z: true,
        }
    }
    fn map(&self, x: V3) -> V3 {
        let f = &self.f;
        [
            f[0][0] * x[0] + f[0][1] * x[1] + f[0][2] * x[2],
            f[1][0] * x[0] + f[1][1] * x[1] + f[1][2] * x[2],
            f[2][0] * x[0] + f[2][1] * x[1] + f[2][2] * x[2],
        ]
    }
}

/// LAMMPS dump, ids 1..n (or as given), type 1; triclinic when tilted.
fn write_dump(path: &Path, b: &Box3, atoms: &[(i64, V3)]) {
    let lx = b.f[0][0] * b.l[0];
    let ly = b.f[1][1] * b.l[1];
    let lz = b.f[2][2] * b.l[2];
    let xz = b.f[0][2] * b.l[2];
    assert!(b.f[0][1] == 0.0 && b.f[1][2] == 0.0 && b.f[1][0] == 0.0);
    let zf = if b.pbc_z { "pp" } else { "ff" };
    let mut s = format!(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{}\n",
        atoms.len()
    );
    if xz != 0.0 {
        writeln!(
            s,
            "ITEM: BOX BOUNDS xy xz yz pp pp {zf}\n{} {} 0\n0 {ly} {xz}\n0 {lz} 0",
            xz.min(0.0),
            lx + xz.max(0.0)
        )
        .unwrap();
    } else {
        writeln!(s, "ITEM: BOX BOUNDS pp pp {zf}\n0 {lx}\n0 {ly}\n0 {lz}").unwrap();
    }
    s.push_str("ITEM: ATOMS id type x y z\n");
    for (id, p) in atoms {
        writeln!(s, "{id} 1 {:.6} {:.6} {:.6}", p[0], p[1], p[2]).unwrap();
    }
    std::fs::write(path, s).unwrap();
}

fn rmda(args: &[&str]) -> (String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run rmda");
    let (o, e) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(out.status.success(), "rmda {args:?} failed:\n{o}\n{e}");
    (o, e)
}

fn scalar(stdout: &str, key: &str) -> i64 {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key} in\n{stdout}"))
        .trim()
        .parse()
        .unwrap()
}

/// A named integer column of an extxyz file the defects command wrote.
fn xyz_column(path: &Path, name: &str) -> Vec<i64> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let head = lines.next().unwrap();
    let props: Vec<&str> = head
        .split("Properties=")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .split(':')
        .collect();
    let mut at = 0;
    let mut col = None;
    for c in props.chunks(3) {
        if c[0] == name {
            col = Some(at);
        }
        at += c[2].parse::<usize>().unwrap();
    }
    let col = col.unwrap_or_else(|| panic!("{} has no {name}", path.display()));
    lines
        .take(n)
        .map(|l| l.split_whitespace().nth(col).unwrap().parse().unwrap())
        .collect()
}

fn work_dir(test: &str) -> (PathBuf, bool) {
    match std::env::var_os("RMDA_WS_FIXTURE_DIR") {
        Some(d) => (PathBuf::from(d).join(test), true),
        None => (
            std::env::temp_dir().join(format!("rmda_ws_ref_{test}_{}", std::process::id())),
            false,
        ),
    }
}

const A: f64 = 3.165;

/// Planted BCC case: (current atoms, vacant site ids, dumbbell site ids).
fn planted(n: usize, sigma: f64, seed: u64) -> (Vec<V3>, Atoms, Vec<i64>, Vec<i64>) {
    let sites = bcc(n, A);
    let id = |i: usize, j: usize, k: usize, b: usize| (((i * n + j) * n + k) * 2 + b) as i64 + 1;
    // Isolated vacancies.
    let mut vac: Vec<i64> = vec![
        id(1, 1, 1, 0),
        id(4, 9, 2, 1),
        id(12, 3, 7, 0),
        id(7, 13, 13, 1),
        id(14, 14, 3, 0),
        id(2, 8, 12, 1),
    ];
    // A 15-site void: a site, its 8 first and 6 second neighbours.
    let (ci, cj, ck) = (9usize, 9usize, 9usize);
    vac.push(id(ci, cj, ck, 0));
    for (di, dj, dk) in [
        (0, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
        (-1, -1, -1),
    ] {
        let w = |c: usize, d: i64| (c as i64 + d) as usize;
        vac.push(id(w(ci, di), w(cj, dj), w(ck, dk), 1));
    }
    for (di, dj, dk) in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ] {
        let w = |c: usize, d: i64| (c as i64 + d) as usize;
        vac.push(id(w(ci, di), w(cj, dj), w(ck, dk), 0));
    }
    // ⟨110⟩ dumbbells: the site's atom split ±(a/4)(1,1,0) (and permutations).
    let dumb: Vec<i64> = vec![id(3, 5, 10, 0), id(11, 11, 1, 1), id(5, 2, 5, 0)];
    let dirs = [[1.0, 1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, 1.0]];
    let mut r = Rng(seed);
    let mut atoms: Vec<(i64, V3)> = Vec::new();
    let mut next = sites.len() as i64 + 1;
    for (s, &p) in sites.iter().enumerate() {
        let sid = s as i64 + 1;
        if vac.contains(&sid) {
            continue;
        }
        if let Some(k) = dumb.iter().position(|&d| d == sid) {
            let h = 0.25 * A;
            let d = dirs[k];
            atoms.push((sid, [p[0] + h * d[0], p[1] + h * d[1], p[2] + h * d[2]]));
            atoms.push((next, [p[0] - h * d[0], p[1] - h * d[1], p[2] - h * d[2]]));
            next += 1;
        } else {
            atoms.push((sid, p));
        }
    }
    for a in atoms.iter_mut() {
        for d in 0..3 {
            a.1[d] += sigma * r.g();
        }
    }
    vac.sort_unstable();
    let mut dumb = dumb;
    dumb.sort_unstable();
    (sites, atoms, vac, dumb)
}

#[test]
fn planted_bcc_exact_with_noise_and_strained_box() {
    let (dir, keep) = work_dir("planted");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let n = 16;
    let l = n as f64 * A;
    let ortho = Box3::ortho(l);
    let mut strained = ortho;
    strained.f = [[1.035, 0.0, 0.03], [0.0, 0.98, 0.0], [0.0, 0.0, 1.015]];

    for (case, sigma, bx, mappings) in [
        ("cold", 0.05, ortho, vec!["off"]),
        (
            "hot",
            0.10,
            ortho,
            vec!["off", "to-reference", "to-current"],
        ),
        (
            "strained",
            0.10,
            strained,
            vec!["to-reference", "to-current"],
        ),
    ] {
        let (sites, atoms, vac, dumb) = planted(n, sigma, 17);
        let reference: Vec<(i64, V3)> = sites
            .iter()
            .enumerate()
            .map(|(s, &p)| (s as i64 + 1, p))
            .collect();
        let rp = dir.join(format!("{case}.ref.dump"));
        let cp = dir.join(format!("{case}.cur.dump"));
        write_dump(&rp, &ortho, &reference);
        let cur: Vec<(i64, V3)> = atoms.iter().map(|&(i, p)| (i, bx.map(p))).collect();
        write_dump(&cp, &bx, &cur);
        assert_eq!((vac.len(), dumb.len()), (21, 3));
        for m in mappings {
            let d = dir.join(format!("{case}.{m}"));
            let (o, e) = rmda(&[
                "defects",
                cp.to_str().unwrap(),
                "--ws-reference",
                rp.to_str().unwrap(),
                "--ws-affine",
                m,
                "--defects",
                d.to_str().unwrap(),
            ]);
            let tag = format!("{case} {m}\n{o}\n{e}");
            assert_eq!(scalar(&o, "defects_vacancies"), 21, "{tag}");
            assert_eq!(scalar(&o, "defects_interstitials"), 3, "{tag}");
            assert_eq!(scalar(&o, "defects_boundary_sites"), 0, "{tag}");
            assert_eq!(scalar(&o, "defects_balance_residual"), 0, "{tag}");
            assert_eq!(scalar(&o, "defects_sites"), sites.len() as i64, "{tag}");
            assert_eq!(scalar(&o, "defects_atoms"), atoms.len() as i64, "{tag}");
            // The void is one cluster of 15; six single vacancies, three dumbbells.
            assert_eq!(scalar(&o, "defects_largest_cluster"), 15, "{tag}");
            assert_eq!(scalar(&o, "defects_clusters"), 10, "{tag}");
            let mut v = xyz_column(&d.join("vacancies.xyz"), "site_id");
            v.sort_unstable();
            assert_eq!(v, vac, "{tag}");
            let mut i = xyz_column(&d.join("interstitials.xyz"), "site_id");
            i.sort_unstable();
            assert_eq!(i, dumb, "{tag}");
            assert!(!e.contains("warning"), "{tag}");
        }
        if case == "strained" {
            // Without a mapping the 3.5 % strain is a sea of Frenkel pairs,
            // still conserving, and the cell mismatch is warned about.
            let (o, e) = rmda(&[
                "defects",
                cp.to_str().unwrap(),
                "--ws-reference",
                rp.to_str().unwrap(),
            ]);
            assert!(scalar(&o, "defects_vacancies") > 100, "{o}");
            assert_eq!(scalar(&o, "defects_balance_residual"), 0, "{o}");
            assert!(e.contains("--ws-affine is off"), "{e}");
        }
    }

    // Grains from regions evaluated on the reference: two slabs, a boundary
    // class on each side of both interfaces, everything conserving, same total.
    let (sites, atoms, _, _) = planted(n, 0.08, 23);
    let rp = dir.join("slabs.ref.dump");
    let cp = dir.join("slabs.cur.dump");
    let reference: Vec<(i64, V3)> = sites
        .iter()
        .enumerate()
        .map(|(s, &p)| (s as i64 + 1, p))
        .collect();
    write_dump(&rp, &ortho, &reference);
    write_dump(&cp, &ortho, &atoms);
    let table = dir.join("slabs.csv");
    let half = format!("x < {}", l / 2.0 - 0.1);
    let other = format!("x >= {}", l / 2.0 - 0.1);
    let (o, _) = rmda(&[
        "defects",
        cp.to_str().unwrap(),
        "--ws-reference",
        rp.to_str().unwrap(),
        "--region",
        &half,
        "--region",
        &other,
        "--table",
        table.to_str().unwrap(),
    ]);
    assert_eq!(scalar(&o, "defects_grains"), 2, "{o}");
    assert_eq!(scalar(&o, "defects_balance_residual"), 0, "{o}");
    assert_eq!(
        scalar(&o, "defects_vacancies") + scalar(&o, "defects_boundary_vacancies"),
        21,
        "{o}"
    );
    assert_eq!(
        scalar(&o, "defects_interstitials") + scalar(&o, "defects_boundary_interstitials"),
        3,
        "{o}"
    );
    assert!(scalar(&o, "defects_boundary_sites") > 0, "{o}");
    assert_eq!(
        scalar(&o, "defects_sites") + scalar(&o, "defects_boundary_sites"),
        sites.len() as i64
    );
    let t = std::fs::read_to_string(&table).unwrap();
    assert_eq!(
        t.lines().filter(|l| l.contains(",explicit,")).count(),
        2,
        "{t}"
    );

    // A rerun of a series with other settings is refused (the signature
    // carries the reference).
    let out = dir.join("series");
    rmda(&[
        "defects",
        cp.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    let bad = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args([
            "defects",
            cp.to_str().unwrap(),
            "--ws-reference",
            rp.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    // --ws-affine needs --ws-reference; --reference conflicts with it.
    for args in [
        vec![
            "defects",
            cp.to_str().unwrap(),
            "--ws-affine",
            "to-reference",
        ],
        vec![
            "defects",
            cp.to_str().unwrap(),
            "--ws-reference",
            rp.to_str().unwrap(),
            "--reference",
            "global",
        ],
    ] {
        let r = Command::new(env!("CARGO_BIN_EXE_rmda"))
            .args(&args)
            .output()
            .unwrap();
        assert!(!r.status.success(), "{args:?} should be refused");
    }
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The OVITO fixture: a thermal BCC reference (σ 0.05 Å, 10³ cells) and a
/// current frame displaced by σ 0.40 Å (so dozens of atoms sit past a
/// Voronoi facet), with 4 atoms removed and 3 inserted at random points.
fn ovito_fixture() -> (Atoms, Atoms, f64) {
    let n = 10;
    let l = n as f64 * A;
    let mut r = Rng(0x005e_ed0f_0717);
    let reference: Vec<(i64, V3)> = bcc(n, A)
        .into_iter()
        .enumerate()
        .map(|(s, p)| {
            let q = [
                p[0] + 0.05 * r.g(),
                p[1] + 0.05 * r.g(),
                p[2] + 0.05 * r.g(),
            ];
            (s as i64 + 1, q)
        })
        .collect();
    let removed = [17i64, 400, 1001, 1999];
    let mut cur: Vec<(i64, V3)> = reference
        .iter()
        .filter(|(id, _)| !removed.contains(id))
        .map(|&(id, p)| {
            (
                id,
                [p[0] + 0.4 * r.g(), p[1] + 0.4 * r.g(), p[2] + 0.4 * r.g()],
            )
        })
        .collect();
    for k in 0..3 {
        cur.push((5000 + k, [r.f() * l, r.f() * l, r.f() * l]));
    }
    (reference, cur, l)
}

#[test]
fn matches_ovito_site_for_site_on_pinned_fixture() {
    let (dir, keep) = work_dir("ovito");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (reference, cur, l) = ovito_fixture();
    let ortho = Box3::ortho(l);
    let mut strained = ortho;
    strained.f = [[1.03, 0.0, 0.025], [0.0, 0.985, 0.0], [0.0, 0.0, 1.01]];
    let mut slab = ortho;
    slab.pbc_z = false;

    // Every fixture is written first, so `RMDA_WS_FIXTURE_DIR` keeps them all
    // for OVITO even when a comparison fails.
    for (case, bx, rbox) in [
        ("same_box", ortho, ortho),
        ("strained", strained, ortho),
        ("open_z", slab, slab),
    ] {
        write_dump(&dir.join(format!("{case}.ref.dump")), &rbox, &reference);
        let mapped: Vec<(i64, V3)> = cur.iter().map(|&(i, p)| (i, bx.map(p))).collect();
        write_dump(&dir.join(format!("{case}.cur.dump")), &bx, &mapped);
    }

    let expected = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ws_reference_ovito.csv"),
    )
    .unwrap();
    let mut checked = 0;
    for line in expected
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split(',').collect();
        let (case, mapping) = (f[0], f[1]);
        let (nv, ni): (i64, i64) = (f[2].parse().unwrap(), f[3].parse().unwrap());
        let ids = |s: &str| -> Vec<i64> {
            s.split(';')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        };
        let (want_v, want_i) = (ids(f[4]), ids(f[5]));
        let rp = dir.join(format!("{case}.ref.dump"));
        let cp = dir.join(format!("{case}.cur.dump"));
        let d = dir.join(format!("{case}.{mapping}"));
        let (o, e) = rmda(&[
            "defects",
            cp.to_str().unwrap(),
            "--ws-reference",
            rp.to_str().unwrap(),
            "--ws-affine",
            mapping,
            "--defects",
            d.to_str().unwrap(),
        ]);
        let tag = format!("{case} {mapping}\n{o}\n{e}");
        assert_eq!(scalar(&o, "defects_vacancies"), nv, "{tag}");
        assert_eq!(scalar(&o, "defects_interstitials"), ni, "{tag}");
        assert_eq!(scalar(&o, "defects_balance_residual"), 0, "{tag}");
        let mut v = xyz_column(&d.join("vacancies.xyz"), "site_id");
        v.sort_unstable();
        assert_eq!(v, want_v, "{tag}");
        let mut i = xyz_column(&d.join("interstitials.xyz"), "site_id");
        i.sort_unstable();
        assert_eq!(i, want_i, "{tag}");
        checked += 1;
    }
    assert_eq!(checked, 4);
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
