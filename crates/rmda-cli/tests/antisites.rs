//! `rmda defects --antisites`: antisites in ordered multi-species crystals
//! (design §16.1a).
//!
//! 1. Explicit reference (`--ws-reference`) against OVITO 3.15
//!    `WignerSeitzAnalysisModifier(per_type_occupancies=True)`, site for site:
//!    a B2 and an L1₂ fixture (thermal reference; current frame with planted
//!    vacancies, dumbbells with the wrong species, swapped pairs, single
//!    substitutions, random insertions and heavy thermal noise), pinned in
//!    `tests/data/antisites_ovito.csv`. `RMDA_ANTISITE_FIXTURE_DIR=dir` keeps
//!    the frames there; regenerate a line with
//!    `python3 tools/antisites_ovito.py CUR REF --counts`.
//! 2. Planted counts, exactly, under the explicit reference and the fitted
//!    global and patch references (sublattice species by majority per motif
//!    basis site) and the local census (a vote per region), and the same V/I
//!    and defect files with and without `--antisites`.
//! 3. The local census against the explicit reference, atom for atom, on
//!    larger noisier boxes cut into several vote regions, whatever the order
//!    of atoms in the file.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

type V3 = [f64; 3];

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
    fn below(&mut self, n: usize) -> usize {
        ((self.f() * n as f64) as usize).min(n - 1)
    }
}

/// An ordered crystal: cubic cell `a`, `n³` cells, basis with a type each.
struct Crystal {
    a: f64,
    n: usize,
    basis: Vec<(V3, i32)>,
    /// A first neighbour of basis site 0 on another sublattice, as
    /// (cell offset, basis index).
    partner: ([usize; 3], usize),
    /// Dumbbell half-split vector (Å).
    split: V3,
}

impl Crystal {
    fn b2(n: usize) -> Self {
        let a = 2.87;
        Crystal {
            a,
            n,
            basis: vec![([0.0; 3], 1), ([0.5; 3], 2)],
            partner: ([0, 0, 0], 1),
            split: [0.25 * a, 0.25 * a, 0.0],
        }
    }
    fn l12(n: usize) -> Self {
        let a = 3.57;
        Crystal {
            a,
            n,
            // Al (2) on the corners, Ni (1) on the faces.
            basis: vec![
                ([0.0, 0.0, 0.0], 2),
                ([0.0, 0.5, 0.5], 1),
                ([0.5, 0.0, 0.5], 1),
                ([0.5, 0.5, 0.0], 1),
            ],
            partner: ([0, 0, 0], 1),
            split: [0.3 * a, 0.0, 0.0],
        }
    }
    fn len(&self) -> f64 {
        self.n as f64 * self.a
    }
    fn nb(&self) -> usize {
        self.basis.len()
    }
    /// Site index of (cell, basis); ids are index + 1.
    fn site(&self, c: [usize; 3], k: usize) -> usize {
        let n = self.n;
        (((c[0] % n) * n + c[1] % n) * n + c[2] % n) * self.nb() + k
    }
    fn cell_of(&self, s: usize) -> [usize; 3] {
        let c = s / self.nb();
        let n = self.n;
        [c / (n * n), (c / n) % n, c % n]
    }
    /// Ideal sites: (position, type).
    fn sites(&self) -> Vec<(V3, i32)> {
        let mut v = Vec::new();
        for i in 0..self.n {
            for j in 0..self.n {
                for k in 0..self.n {
                    for &(b, t) in &self.basis {
                        let c = [i as f64 + b[0], j as f64 + b[1], k as f64 + b[2]];
                        v.push(([c[0] * self.a, c[1] * self.a, c[2] * self.a], t));
                    }
                }
            }
        }
        v
    }
}

/// (id, type, position).
type Atoms = Vec<(i64, i32, V3)>;

/// What was planted.
#[derive(Default, Debug)]
struct Truth {
    vacancies: usize,
    interstitials: usize,
    /// (site type, occupant type) → count.
    pairs: BTreeMap<(i32, i32), u64>,
}

impl Truth {
    fn antisites(&self) -> u64 {
        self.pairs.values().sum()
    }
}

fn other(t: i32) -> i32 {
    3 - t
}

/// A current frame: `vac` vacancies, `dumb` dumbbells (extra atom of the
/// other species), `swaps` swapped first-neighbour pairs, `flips` single
/// substitutions, each > 2 cells from the others, then `inserts` atoms at
/// random points and Gaussian noise `sigma` on every atom.
#[allow(clippy::too_many_arguments)]
fn plant(
    x: &Crystal,
    sites: &[(V3, i32)],
    vac: usize,
    dumb: usize,
    swaps: usize,
    flips: usize,
    inserts: usize,
    sigma: f64,
    seed: u64,
) -> (Atoms, Truth) {
    let mut r = Rng(seed);
    let n = x.n as i64;
    let mut used: Vec<[usize; 3]> = Vec::new();
    let mut pick = |r: &mut Rng| -> [usize; 3] {
        loop {
            let c = x.cell_of(r.below(sites.len()));
            let far = used.iter().all(|u| {
                (0..3).any(|d| {
                    let dd = (c[d] as i64 - u[d] as i64).rem_euclid(n);
                    dd.min(n - dd) > 1
                })
            });
            if far {
                used.push(c);
                return c;
            }
        }
    };
    let mut kind: BTreeMap<usize, char> = BTreeMap::new();
    let mut truth = Truth::default();
    let mut types: Vec<i32> = sites.iter().map(|s| s.1).collect();
    for _ in 0..vac {
        let c = pick(&mut r);
        let s = x.site(c, r.below(x.nb()));
        kind.insert(s, 'v');
        truth.vacancies += 1;
    }
    for _ in 0..dumb {
        let c = pick(&mut r);
        let s = x.site(c, r.below(x.nb()));
        kind.insert(s, 'd');
        truth.interstitials += 1;
    }
    for _ in 0..swaps {
        let c = pick(&mut r);
        let (off, k) = x.partner;
        let s0 = x.site(c, 0);
        let s1 = x.site([c[0] + off[0], c[1] + off[1], c[2] + off[2]], k);
        let (t0, t1) = (types[s0], types[s1]);
        assert_ne!(t0, t1);
        types.swap(s0, s1);
        *truth.pairs.entry((t0, t1)).or_insert(0) += 1;
        *truth.pairs.entry((t1, t0)).or_insert(0) += 1;
    }
    for _ in 0..flips {
        let c = pick(&mut r);
        let s = x.site(c, r.below(x.nb()));
        let t = sites[s].1;
        types[s] = other(t);
        *truth.pairs.entry((t, other(t))).or_insert(0) += 1;
    }
    let mut atoms: Atoms = Vec::new();
    let mut next = sites.len() as i64 + 1;
    let h = x.split;
    for (s, &(p, _)) in sites.iter().enumerate() {
        let id = s as i64 + 1;
        match kind.get(&s) {
            Some('v') => {}
            Some('d') => {
                atoms.push((id, types[s], [p[0] + h[0], p[1] + h[1], p[2] + h[2]]));
                atoms.push((
                    next,
                    other(types[s]),
                    [p[0] - h[0], p[1] - h[1], p[2] - h[2]],
                ));
                next += 1;
            }
            _ => atoms.push((id, types[s], p)),
        }
    }
    let l = x.len();
    for k in 0..inserts {
        atoms.push((next, 1 + (k % 2) as i32, [r.f() * l, r.f() * l, r.f() * l]));
        next += 1;
    }
    for a in atoms.iter_mut() {
        for d in 0..3 {
            a.2[d] += sigma * r.g();
        }
    }
    (atoms, truth)
}

/// The reference frame: ideal sites with noise `sigma`.
fn reference(sites: &[(V3, i32)], sigma: f64, seed: u64) -> Atoms {
    let mut r = Rng(seed);
    sites
        .iter()
        .enumerate()
        .map(|(s, &(p, t))| {
            (
                s as i64 + 1,
                t,
                [
                    p[0] + sigma * r.g(),
                    p[1] + sigma * r.g(),
                    p[2] + sigma * r.g(),
                ],
            )
        })
        .collect()
}

/// Periodic cubic LAMMPS dump with types.
fn write_dump(path: &Path, l: f64, atoms: &Atoms) {
    let mut s = format!(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n0 {l}\n0 {l}\n0 {l}\n\
         ITEM: ATOMS id type x y z\n",
        atoms.len()
    );
    for (id, t, p) in atoms {
        writeln!(s, "{id} {t} {:.6} {:.6} {:.6}", p[0], p[1], p[2]).unwrap();
    }
    std::fs::write(path, s).unwrap();
}

/// Periodic cubic extended XYZ with species names `names[type - 1]`.
fn write_xyz(path: &Path, l: f64, atoms: &Atoms, names: [&str; 2]) {
    let mut s = format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" Properties=species:S:1:pos:R:3:id:I:1 pbc=\"T T T\"\n",
        atoms.len()
    );
    for (id, t, p) in atoms {
        writeln!(
            s,
            "{} {:.6} {:.6} {:.6} {id}",
            names[*t as usize - 1],
            p[0],
            p[1],
            p[2]
        )
        .unwrap();
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

/// `defects_antisites_<occ>_on_<site>` lines as (site, occupant) → count.
fn pairs(stdout: &str) -> BTreeMap<(String, String), u64> {
    stdout
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("defects_antisites_")?;
            let (key, v) = rest.split_once(": ")?;
            let (occ, site) = key.split_once("_on_")?;
            Some(((site.to_owned(), occ.to_owned()), v.trim().parse().ok()?))
        })
        .collect()
}

fn truth_pairs(t: &Truth) -> BTreeMap<(String, String), u64> {
    t.pairs
        .iter()
        .map(|(&(s, o), &c)| ((s.to_string(), o.to_string()), c))
        .collect()
}

/// A named column of an extxyz file the defects command wrote.
fn xyz_column(path: &Path, name: &str) -> Vec<String> {
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
        .map(|l| l.split_whitespace().nth(col).unwrap().to_owned())
        .collect()
}

/// A column of a CSV table by header name.
fn csv_column(path: &Path, name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let k = head
        .iter()
        .position(|h| *h == name)
        .unwrap_or_else(|| panic!("{} has no {name}", path.display()));
    lines
        .map(|l| l.split(',').nth(k).unwrap_or("").to_owned())
        .collect()
}

fn work_dir(test: &str) -> (PathBuf, bool) {
    match std::env::var_os("RMDA_ANTISITE_FIXTURE_DIR") {
        Some(d) => (PathBuf::from(d).join(test), true),
        None => (
            std::env::temp_dir().join(format!("rmda_antisites_{test}_{}", std::process::id())),
            false,
        ),
    }
}

/// The OVITO fixtures: (name, crystal, reference, current).
fn ovito_fixtures() -> Vec<(&'static str, Crystal, Atoms, Atoms)> {
    let mut v = Vec::new();
    for (name, x, seed) in [
        ("b2", Crystal::b2(10), 0xb2b2_0001u64),
        ("l12", Crystal::l12(9), 0x1112_0002u64),
    ] {
        let sites = x.sites();
        let rf = reference(&sites, 0.05, seed);
        // Heavy noise (σ 0.3 Å) so that some atoms cross a Voronoi facet.
        let (cur, _) = plant(&x, &sites, 6, 4, 8, 5, 3, 0.3, seed ^ 0x55);
        v.push((name, x, rf, cur));
    }
    v
}

#[test]
fn explicit_reference_matches_ovito_per_type_occupancy() {
    let (dir, keep) = work_dir("ovito");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = ovito_fixtures();
    for (name, x, rf, cur) in &fixtures {
        write_dump(&dir.join(format!("{name}.ref.dump")), x.len(), rf);
        write_dump(&dir.join(format!("{name}.cur.dump")), x.len(), cur);
    }
    let expected = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/antisites_ovito.csv"),
    )
    .unwrap();
    let mut checked = 0;
    for line in expected
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split(',').collect();
        let name = f[0];
        let (nv, ni, na): (i64, i64, i64) = (
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        );
        let want_pairs: BTreeMap<(String, String), u64> = f[4]
            .split(';')
            .filter(|t| !t.is_empty())
            .map(|t| {
                let (k, c) = t.split_once('=').unwrap();
                let (s, o) = k.split_once('>').unwrap();
                ((s.to_owned(), o.to_owned()), c.parse().unwrap())
            })
            .collect();
        let want_sites: Vec<i64> = f[5]
            .split(';')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().unwrap())
            .collect();
        let rp = dir.join(format!("{name}.ref.dump"));
        let cp = dir.join(format!("{name}.cur.dump"));
        let d = dir.join(format!("{name}.out"));
        let (o, e) = rmda(&[
            "defects",
            cp.to_str().unwrap(),
            "--ws-reference",
            rp.to_str().unwrap(),
            "--antisites",
            "--defects",
            d.to_str().unwrap(),
        ]);
        let tag = format!("{name}\n{o}\n{e}");
        assert_eq!(scalar(&o, "defects_vacancies"), nv, "{tag}");
        assert_eq!(scalar(&o, "defects_interstitials"), ni, "{tag}");
        assert_eq!(scalar(&o, "defects_antisites"), na, "{tag}");
        assert_eq!(scalar(&o, "defects_boundary_antisites"), 0, "{tag}");
        assert_eq!(pairs(&o), want_pairs, "{tag}");
        let mut s: Vec<i64> = xyz_column(&d.join("antisites.xyz"), "site_id")
            .iter()
            .map(|v| v.parse().unwrap())
            .collect();
        s.sort_unstable();
        assert_eq!(s, want_sites, "{tag}");
        checked += 1;
    }
    assert_eq!(checked, fixtures.len());
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// `rmda defects` with `extra` args, with and without `--antisites`: the V/I
/// scalars and defect files must agree; returns the `--antisites` stdout.
fn run_both(dir: &Path, tag: &str, cur: &Path, extra: &[&str]) -> String {
    let plain = dir.join(format!("{tag}.plain"));
    let with = dir.join(format!("{tag}.with"));
    let mut a: Vec<&str> = vec!["defects", cur.to_str().unwrap()];
    a.extend_from_slice(extra);
    let pt = plain.join("t.csv");
    let wt = with.join("t.csv");
    std::fs::create_dir_all(&plain).unwrap();
    std::fs::create_dir_all(&with).unwrap();
    let mut ap = a.clone();
    ap.extend([
        "--defects",
        plain.to_str().unwrap(),
        "--table",
        pt.to_str().unwrap(),
    ]);
    let mut aw = a.clone();
    aw.extend([
        "--defects",
        with.to_str().unwrap(),
        "--table",
        wt.to_str().unwrap(),
        "--antisites",
    ]);
    let (op, _) = rmda(&ap);
    let (ow, ew) = rmda(&aw);
    // The plain run's stdout is a prefix of the antisite run's.
    assert!(ow.starts_with(&op), "{tag}: stdout\n{op}\n---\n{ow}\n{ew}");
    for f in ["vacancies.xyz", "interstitials.xyz"] {
        assert_eq!(
            std::fs::read(plain.join(f)).unwrap(),
            std::fs::read(with.join(f)).unwrap(),
            "{tag}: {f}"
        );
    }
    // The table only gains columns at the end.
    let tp = std::fs::read_to_string(&pt).unwrap();
    let tw = std::fs::read_to_string(&wt).unwrap();
    for (lp, lw) in tp.lines().zip(tw.lines()) {
        assert!(lw.starts_with(lp), "{tag}: table\n{lp}\n{lw}");
    }
    assert!(!plain.join("antisites.xyz").exists());
    ow
}

#[test]
fn planted_antisites_exact_under_every_reference() {
    let (dir, keep) = work_dir("planted");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, x, seed, patch) in [
        ("b2", Crystal::b2(14), 0x0b2u64, Some("13")),
        ("l12", Crystal::l12(9), 0x112u64, None),
    ] {
        let sites = x.sites();
        let rf = reference(&sites, 0.03, seed);
        let (cur, truth) = plant(&x, &sites, 6, 4, 8, 5, 0, 0.08, seed ^ 0x77);
        let rp = dir.join(format!("{name}.ref.dump"));
        let cp = dir.join(format!("{name}.cur.dump"));
        write_dump(&rp, x.len(), &rf);
        write_dump(&cp, x.len(), &cur);
        let want = truth_pairs(&truth);

        let mut cases: Vec<(String, Vec<&str>)> = vec![
            (
                "explicit".into(),
                vec!["--ws-reference", rp.to_str().unwrap()],
            ),
            ("global".into(), vec!["--reference", "global"]),
            ("local".into(), vec!["--reference", "local"]),
        ];
        if let Some(p) = patch {
            cases.push((
                "patch".into(),
                vec!["--reference", "patch", "--patch-size", p],
            ));
        }
        let mut explicit_ids: Vec<String> = Vec::new();
        for (case, extra) in &cases {
            let tag = format!("{name}.{case}");
            let o = run_both(&dir, &tag, &cp, extra);
            let msg = format!("{tag}\n{o}");
            assert_eq!(
                scalar(&o, "defects_vacancies") as usize,
                truth.vacancies,
                "{msg}"
            );
            assert_eq!(
                scalar(&o, "defects_interstitials") as usize,
                truth.interstitials,
                "{msg}"
            );
            assert_eq!(
                scalar(&o, "defects_antisites") as u64,
                truth.antisites(),
                "{msg}"
            );
            assert_eq!(pairs(&o), want, "{msg}");
            let with = dir.join(format!("{tag}.with"));
            let reference = csv_column(&with.join("t.csv"), "reference");
            assert_eq!(reference, vec![case.as_str()], "{msg}");
            // The species of each motif basis site, as planted.
            let sub = csv_column(&with.join("t.csv"), "sublattice_types");
            if case == "explicit" {
                assert_eq!(sub, vec![""], "{msg}");
            } else {
                let mut got: Vec<String> = sub[0].split(' ').map(str::to_owned).collect();
                got.sort();
                let mut want_sub: Vec<String> = x.basis.iter().map(|b| b.1.to_string()).collect();
                want_sub.sort();
                assert_eq!(got, want_sub, "{msg}");
            }
            let n = xyz_column(&with.join("antisites.xyz"), "type").len() as u64;
            assert_eq!(n, truth.antisites(), "{msg}");
            // Every listed antisite holds another type than its site's.
            let t = xyz_column(&with.join("antisites.xyz"), "type");
            let st = xyz_column(&with.join("antisites.xyz"), "site_type");
            assert!(t.iter().zip(&st).all(|(a, b)| a != b), "{msg}");
            // The same atoms as the explicit reference's, site for site.
            let mut ids = xyz_column(&with.join("antisites.xyz"), "id");
            ids.sort();
            if case == "explicit" {
                explicit_ids = ids;
            } else {
                assert_eq!(ids, explicit_ids, "{msg}");
            }
            // Clusters: every planted antisite is > 2 cells from any V/I
            // defect, so none joins a V/I cluster; each swapped pair is one
            // antisite-only cluster and each substitution another, numbered
            // after the V/I clusters.
            assert_eq!(scalar(&o, "defects_clustered_antisites"), 0, "{msg}");
            assert_eq!(scalar(&o, "defects_antisite_clusters"), 8 + 5, "{msg}");
            assert_eq!(scalar(&o, "defects_largest_antisite_cluster"), 2, "{msg}");
            let nvi = scalar(&o, "defects_clusters");
            let cl: Vec<i64> = xyz_column(&with.join("antisites.xyz"), "cluster")
                .iter()
                .map(|v| v.parse().unwrap())
                .collect();
            assert!(cl.iter().all(|&c| c >= nvi && c < nvi + 13), "{msg}");
        }
    }
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn species_are_matched_by_name_across_frames() {
    let (dir, keep) = work_dir("names");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let x = Crystal::b2(8);
    let sites = x.sites();
    let rf = reference(&sites, 0.03, 9);
    let (mut cur, truth) = plant(&x, &sites, 2, 1, 3, 2, 0, 0.05, 10);
    // The current frame lists an Al atom first, so a reader numbering species
    // by first appearance numbers them the other way round.
    let k = cur.iter().position(|a| a.1 == 2).unwrap();
    cur.swap(0, k);
    let rp = dir.join("ref.xyz");
    let cp = dir.join("cur.xyz");
    write_xyz(&rp, x.len(), &rf, ["Fe", "Al"]);
    write_xyz(&cp, x.len(), &cur, ["Fe", "Al"]);
    let (o, e) = rmda(&[
        "defects",
        cp.to_str().unwrap(),
        "--ws-reference",
        rp.to_str().unwrap(),
        "--antisites",
    ]);
    let msg = format!("{o}\n{e}");
    assert_eq!(
        scalar(&o, "defects_antisites") as u64,
        truth.antisites(),
        "{msg}"
    );
    let name = |t: i32| ["Fe", "Al"][t as usize - 1].to_owned();
    let want: BTreeMap<(String, String), u64> = truth
        .pairs
        .iter()
        .map(|(&(s, oc), &c)| ((name(s), name(oc)), c))
        .collect();
    assert_eq!(pairs(&o), want, "{msg}");
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn series_tables_with_antisites() {
    let (dir, keep) = work_dir("series");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let x = Crystal::b2(8);
    let sites = x.sites();
    let (cur, truth) = plant(&x, &sites, 2, 1, 3, 2, 0, 0.05, 21);
    let cp = dir.join("frame.1.dump");
    write_dump(&cp, x.len(), &cur);
    let out = dir.join("out");
    rmda(&[
        "defects",
        cp.to_str().unwrap(),
        "--reference",
        "global",
        "--antisites",
        "--out",
        out.to_str().unwrap(),
        "--defects-xyz",
    ]);
    let s = out.join("summary.csv");
    assert_eq!(
        csv_column(&s, "antisites"),
        vec![truth.antisites().to_string()]
    );
    assert_eq!(csv_column(&s, "gb_antisites"), vec!["0"]);
    let head = std::fs::read_to_string(&s).unwrap();
    assert!(head
        .lines()
        .next()
        .unwrap()
        .ends_with("net_vacancies,gb_net_vacancies,antisites,gb_antisites"));
    let a = out.join("antisites.csv");
    let total: u64 = csv_column(&a, "antisites")
        .iter()
        .map(|v| v.parse::<u64>().unwrap())
        .sum();
    assert_eq!(total, truth.antisites());
    assert!(out.join("defects/1/antisites.xyz").exists());
    // clusters.csv: an `antisites` column; the V/I clusters as without
    // --antisites, then one antisite-only row per swapped pair and per
    // substitution (size 0: size counts V and I).
    let c = out.join("clusters.csv");
    let text = std::fs::read_to_string(&c).unwrap();
    assert!(text.lines().next().unwrap().ends_with(",cz,antisites"));
    let size = csv_column(&c, "size");
    let na = csv_column(&c, "antisites");
    let only: Vec<&String> = size
        .iter()
        .zip(&na)
        .filter(|(s, _)| *s == "0")
        .map(|(_, a)| a)
        .collect();
    assert_eq!(only.len(), 3 + 2, "{text}");
    let total: u64 = na.iter().map(|v| v.parse::<u64>().unwrap()).sum();
    assert_eq!(total, truth.antisites(), "{text}");
    let plain = dir.join("plain");
    rmda(&[
        "defects",
        cp.to_str().unwrap(),
        "--reference",
        "global",
        "--out",
        plain.to_str().unwrap(),
    ]);
    let p = std::fs::read_to_string(plain.join("clusters.csv")).unwrap();
    for (lp, lw) in p.lines().skip(1).zip(text.lines().skip(1)) {
        assert!(lw.starts_with(lp), "{lp}\n{lw}");
    }
    assert_eq!(text.lines().count(), p.lines().count() + 5);
    // A rerun without --antisites is other settings: refused, not mixed.
    let bad = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args([
            "defects",
            cp.to_str().unwrap(),
            "--reference",
            "global",
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// `atoms` in a shuffled order.
fn shuffled(atoms: &Atoms, seed: u64) -> Atoms {
    let mut v = atoms.clone();
    let mut r = Rng(seed);
    for i in (1..v.len()).rev() {
        v.swap(i, r.below(i + 1));
    }
    v
}

/// (V, I, antisites, pairs, sorted antisite ids) of one `rmda defects` run.
type Counts = (i64, i64, i64, BTreeMap<(String, String), u64>, Vec<i64>);

fn counts(cur: &Path, out: &Path, extra: &[&str]) -> (Counts, String) {
    let mut a = vec![
        "defects",
        cur.to_str().unwrap(),
        "--antisites",
        "--defects",
        out.to_str().unwrap(),
    ];
    a.extend_from_slice(extra);
    let (o, e) = rmda(&a);
    let mut ids: Vec<i64> = xyz_column(&out.join("antisites.xyz"), "id")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    ids.sort_unstable();
    (
        (
            scalar(&o, "defects_vacancies"),
            scalar(&o, "defects_interstitials"),
            scalar(&o, "defects_antisites"),
            pairs(&o),
            ids,
        ),
        format!("{o}\n{e}"),
    )
}

#[test]
fn local_census_matches_explicit_reference_atom_for_atom() {
    // Boxes cut into 2, 3 and 4 vote regions per axis (the last with
    // margins), thermal noise up to σ 0.15 Å, more defects than above. The
    // local census must find the explicit reference's antisites atom for
    // atom, with the same V and I, in the file's order and shuffled.
    let (dir, keep) = work_dir("local");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (name, x, sigma, seed) in [
        ("l12", Crystal::l12(12), 0.15, 0x1112_0003u64),
        ("b2_3", Crystal::b2(28), 0.15, 0xb2b2_0003u64),
        ("b2_4", Crystal::b2(36), 0.1, 0xb2b2_0004u64),
    ] {
        let sites = x.sites();
        let rf = reference(&sites, 0.03, seed);
        let (cur, truth) = plant(&x, &sites, 12, 8, 16, 10, 0, sigma, seed ^ 0x99);
        let rp = dir.join(format!("{name}.ref.dump"));
        let cp = dir.join(format!("{name}.cur.dump"));
        let sp = dir.join(format!("{name}.shuffled.dump"));
        write_dump(&rp, x.len(), &rf);
        write_dump(&cp, x.len(), &cur);
        write_dump(&sp, x.len(), &shuffled(&cur, seed));
        let (want, we) = counts(
            &cp,
            &dir.join(format!("{name}.explicit")),
            &["--ws-reference", rp.to_str().unwrap()],
        );
        // The explicit reference sees what was planted.
        assert_eq!(want.0 as usize, truth.vacancies, "{name}\n{we}");
        assert_eq!(want.1 as usize, truth.interstitials, "{name}\n{we}");
        assert_eq!(want.3, truth_pairs(&truth), "{name}\n{we}");
        for (tag, file) in [("local", &cp), ("local_shuffled", &sp)] {
            let (got, ge) = counts(
                file,
                &dir.join(format!("{name}.{tag}")),
                &["--reference", "local"],
            );
            assert!(ge.contains("ref=local"), "{name}.{tag}\n{ge}");
            assert_eq!(got, want, "{name}.{tag}\n{ge}\n--- explicit\n{we}");
        }
    }
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
