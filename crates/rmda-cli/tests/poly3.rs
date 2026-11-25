//! The end goal on a synthetic fixture (design §15.5, §16.8): a
//! periodic FCC/BCC/HCP polycrystal with planted point defects and small
//! clusters at known depths, over three frames in which one grain grows and
//! another shrinks, through `rmda segment` and `rmda defects FILES... --segment
//! --out DIR`.
//!
//! The generator is here; `RMDA_POLY3_DIR=dir` keeps its frames and truth
//! tables there (the series in `dir/series`, the few-grain fixtures in
//! `dir/few/few<seed>`) for `tools/poly3_ovito.py`, which runs OVITO on the
//! series frames.

use rmda_core::analysis::orient;
use rmda_core::analysis::ptm::structure;
use rmda_core::cell::{Coord3, Mat3, UnitCell};
use rmda_core::exec::Par;
use rmda_core::lattice::fit::mat_to_quat;
use rmda_core::neighbor::{KnnIndex, KnnScratch};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn rotation(axis: Coord3, deg: f64) -> Mat3 {
    let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / l, axis[1] / l, axis[2] / l);
    let (s, c) = deg.to_radians().sin_cos();
    let t = 1.0 - c;
    Mat3::from_rows(
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    )
}

fn dist2(a: Coord3, b: Coord3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// One grain of the construction.
struct Grain {
    st: i32,
    /// Lattice constant `a` (HCP: ideal c/a).
    a: f64,
    seed: Coord3,
    rot: Mat3,
}

impl Grain {
    fn nn(&self) -> f64 {
        match self.st {
            structure::FCC => self.a / 2f64.sqrt(),
            structure::BCC => self.a * 3f64.sqrt() / 2.0,
            _ => self.a,
        }
    }

    /// Conventional cell edges and basis (fractions).
    fn cell(&self) -> ([f64; 3], Vec<Coord3>) {
        let a = self.a;
        match self.st {
            structure::FCC => (
                [a; 3],
                vec![
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ],
            ),
            structure::BCC => ([a; 3], vec![[0.0; 3], [0.5; 3]]),
            _ => (
                [a, a * 3f64.sqrt(), a * (8.0f64 / 3.0).sqrt()],
                vec![
                    [0.0; 3],
                    [0.5, 0.5, 0.0],
                    [0.0, 1.0 / 3.0, 0.5],
                    [0.5, 5.0 / 6.0, 0.5],
                ],
            ),
        }
    }

    /// Unit vector (crystal frame) a planted dumbbell is split along:
    /// BCC ⟨110⟩, FCC ⟨100⟩, HCP ⟨0001⟩.
    fn dumbbell_axis(&self) -> Coord3 {
        let s = 0.5f64.sqrt();
        let d = match self.st {
            structure::BCC => [s, s, 0.0],
            structure::FCC => [1.0, 0.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
        self.rot.mul_v(d)
    }
}

/// A lattice site of a grain: key for stable ids, position (wrapped).
#[derive(Clone, Copy)]
struct Site {
    key: (usize, [i32; 3], u8),
    p: Coord3,
    depth: f64,
}

/// What the generator planted in one frame.
#[derive(Clone, Copy, Debug)]
struct Planted {
    grain: usize,
    vacancy: bool,
    p: Coord3,
    depth: f64,
}

struct Frame {
    /// (id, type = grain structure slot, position), sorted by id.
    atoms: Vec<(i64, usize, Coord3)>,
    /// The defect-free sites (after boundary collisions), unjittered: the
    /// Wigner–Seitz reference `tools/poly3_ovito.py` gives OVITO.
    reference: Vec<(i64, usize, Coord3)>,
    /// Per grain: the smallest distance of its sites from a facet with its
    /// own periodic image (small: the grain meets itself across a seam).
    own_gap: Vec<f64>,
    planted: Vec<Planted>,
}

/// Grain an id was built in (`id = (grain + 1)·10⁶ + n`).
fn grain_of_id(id: i64) -> usize {
    (id / 1_000_000 - 1) as usize
}

/// Depth below which a planted defect counts as interior, and above which
/// it counts as near the boundary, Å (the default boundary width is about 4 Å).
const DEEP: f64 = 9.0;

struct Poly {
    l: f64,
    grains: Vec<Grain>,
    /// No grain may meet its own periodic image (asserted per frame).
    isolated: bool,
}

impl Poly {
    /// Eight grains on a jittered 2×2×2 arrangement, lattices cycling FCC,
    /// BCC, HCP, same-lattice pairs > 15° apart. The regular arrangement
    /// keeps every grain away from its own periodic image (checked in
    /// [`Poly::frame`]).
    fn new(l: f64, seed: u64) -> Self {
        let mut r = Rng(seed);
        let lattices = [
            (structure::FCC, 3.615),
            (structure::BCC, 3.165),
            (structure::HCP, 2.95),
        ];
        let mut grains: Vec<Grain> = Vec::new();
        while grains.len() < 8 {
            let k = grains.len();
            let (st, a) = lattices[k % 3];
            let c = [k & 1, (k >> 1) & 1, (k >> 2) & 1];
            let seed: Coord3 =
                std::array::from_fn(|d| ((c[d] as f64 + 0.5) * 0.5 + 0.05 * r.g()) * l);
            let rot = rotation([r.g(), r.g(), r.g()], 180.0 * r.f());
            let g = orient::group(st).unwrap();
            let distinct = grains.iter().filter(|o| o.st == st).all(|o| {
                orient::disorientation(&mat_to_quat(&o.rot), &mat_to_quat(&rot), &g) > 15.0
            });
            if distinct {
                grains.push(Grain { st, a, seed, rot });
            }
        }
        Self {
            l,
            grains,
            isolated: true,
        }
    }

    /// `k` BCC (W) grains at random, well-spread seeds in a small box, so
    /// that grains meet their own periodic images across self-seams, as in
    /// an atomsk polycrystal with few grains. Orientations > 15° apart.
    fn few(l: f64, seed: u64, k: usize) -> Self {
        let mut r = Rng(seed);
        let g = orient::group(structure::BCC).unwrap();
        let mic = |d: f64| d - l * (d / l).round();
        let sep = |a: &Coord3, b: &Coord3| -> f64 {
            (0..3).map(|i| mic(a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
        };
        let mut grains: Vec<Grain> = Vec::new();
        while grains.len() < k {
            let seed = (0..200)
                .map(|_| [r.f() * l, r.f() * l, r.f() * l])
                .max_by(|a, b| {
                    let d = |p: &Coord3| grains.iter().map(|o| sep(p, &o.seed)).fold(l, f64::min);
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            let rot = rotation([r.g(), r.g(), r.g()], 180.0 * r.f());
            if grains.iter().all(|o| {
                orient::disorientation(&mat_to_quat(&o.rot), &mat_to_quat(&rot), &g) > 15.0
            }) {
                grains.push(Grain {
                    st: structure::BCC,
                    a: 3.165,
                    seed,
                    rot,
                });
            }
        }
        Self {
            l,
            grains,
            isolated: false,
        }
    }

    /// Power-diagram owner of `p` and its distance to the nearest facet.
    fn owner(&self, p: Coord3, w: &[f64]) -> (usize, f64) {
        let (k, d, _) = self.owner_self(p, w);
        (k, d)
    }

    /// [`Poly::owner`], plus the distance to the nearest facet with the
    /// owner's own periodic image.
    fn owner_self(&self, p: Coord3, w: &[f64]) -> (usize, f64, f64) {
        let l = self.l;
        // Every grain's 27 images around p.
        let mut cand: Vec<(f64, usize, Coord3)> = Vec::with_capacity(27 * self.grains.len());
        for (k, g) in self.grains.iter().enumerate() {
            let base: Coord3 = std::array::from_fn(|d| {
                p[d] + (g.seed[d] - p[d]) - l * ((g.seed[d] - p[d]) / l).round()
            });
            for i in -1..=1 {
                for j in -1..=1 {
                    for m in -1..=1 {
                        let s = [
                            base[0] + i as f64 * l,
                            base[1] + j as f64 * l,
                            base[2] + m as f64 * l,
                        ];
                        cand.push((dist2(p, s) - w[k], k, s));
                    }
                }
            }
        }
        let best = cand
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0))
            .unwrap()
            .0;
        let (sc, k, s) = cand[best];
        let facet = |own: bool| {
            cand.iter()
                .enumerate()
                .filter(|(i, c)| *i != best && (c.1 == k) == own)
                .map(|(_, &(sj, _, s2))| (sj - sc) / (2.0 * dist2(s, s2).sqrt()))
                .fold(f64::INFINITY, f64::min)
        };
        let own = facet(true);
        (k, facet(false).min(own), own)
    }

    /// Power-diagram weights of frame `f`, Å²: grain 0 grows, grain 1 shrinks.
    fn weights(&self, f: usize) -> Vec<f64> {
        let mut w = vec![0.0; self.grains.len()];
        w[0] = 150.0 * f as f64;
        w[1] = -150.0 * f as f64;
        w
    }

    /// Frame `f`: grain 0 grows and grain 1 shrinks with `f`.
    fn frame(&self, f: usize, ids: &mut HashMap<(usize, [i32; 3], u8), i64>) -> Frame {
        let l = self.l;
        let w = self.weights(f);
        // Sites of every grain inside its own power cell.
        let mut sites: Vec<Site> = Vec::new();
        let mut own_gap = vec![f64::INFINITY; self.grains.len()];
        for (k, g) in self.grains.iter().enumerate() {
            let (c, basis) = g.cell();
            let reach = 0.5 * l * 3f64.sqrt();
            let n: Vec<i32> = c.iter().map(|c| (reach / c).ceil() as i32 + 1).collect();
            for i in -n[0]..=n[0] {
                for j in -n[1]..=n[1] {
                    for m in -n[2]..=n[2] {
                        for (b, f) in basis.iter().enumerate() {
                            let x = [
                                (i as f64 + f[0]) * c[0],
                                (j as f64 + f[1]) * c[1],
                                (m as f64 + f[2]) * c[2],
                            ];
                            let q = g.rot.mul_v(x);
                            if q.iter().any(|v| *v < -0.5 * l || *v >= 0.5 * l) {
                                continue;
                            }
                            let p: Coord3 =
                                std::array::from_fn(|d| (g.seed[d] + q[d]).rem_euclid(l));
                            let (owner, depth, own) = self.owner_self(p, &w);
                            // A grain touching its own image would be a
                            // translation fault, not a polycrystal.
                            assert!(
                                !self.isolated || owner != k || own > 3.0,
                                "grain {k} meets its own image"
                            );
                            if owner == k {
                                own_gap[k] = own_gap[k].min(own);
                            }
                            if owner == k {
                                sites.push(Site {
                                    key: (k, [i, j, m], b as u8),
                                    p,
                                    depth,
                                });
                            }
                        }
                    }
                }
            }
        }
        // Atoms closer than 0.8 of the smaller bond across a boundary: the
        // shallower one goes.
        let cell = UnitCell::orthorhombic(l, l, l);
        let pos: Vec<Coord3> = sites.iter().map(|s| s.p).collect();
        let idx = KnnIndex::build(&cell, &pos, &Par::new(4)).unwrap();
        let mut sc = KnnScratch::default();
        let mut nb = Vec::new();
        let mut keep = vec![true; sites.len()];
        for i in 0..sites.len() {
            idx.knn(i, pos[i], 4, &mut sc, &mut nb);
            let (gi, di) = (sites[i].key.0, sites[i].depth);
            for &(j, _, d2) in nb.iter() {
                let s = &sites[j as usize];
                let lim = 0.8 * self.grains[gi].nn().min(self.grains[s.key.0].nn());
                if s.key.0 != gi && d2 < lim * lim && (s.depth, j as usize) > (di, i) {
                    keep[i] = false;
                }
            }
        }
        let sites: Vec<Site> = sites
            .into_iter()
            .zip(keep)
            .filter(|x| x.1)
            .map(|x| x.0)
            .collect();

        // Plant defects per grain: a deep set (interior class) and a shallow
        // set (boundary class), at least 8 Å apart.
        let mut r = Rng(1000 + f as u64);
        let mut planted: Vec<Planted> = Vec::new();
        let mut removed = vec![false; sites.len()];
        let mut extra: Vec<(usize, Coord3)> = Vec::new(); // (site, other half)
        let mut moved: HashMap<usize, Coord3> = HashMap::new();
        let mic = |d: f64| d - l * (d / l).round();
        let sep2 = |a: Coord3, b: Coord3| (0..3).map(|k| mic(a[k] - b[k]).powi(2)).sum::<f64>();
        for (k, g) in self.grains.iter().enumerate() {
            let nn = g.nn();
            let mine: Vec<usize> = (0..sites.len()).filter(|&i| sites[i].key.0 == k).collect();
            let nearest_site = |i: usize| -> Option<usize> {
                mine.iter()
                    .copied()
                    .filter(|&j| j != i && sep2(sites[i].p, sites[j].p) < (1.05 * nn).powi(2))
                    .min()
            };
            // (vacancies, sites per defect, deep): 2 + (k + f) % 2 single
            // vacancies, a divacancy, a dumbbell and a di-interstitial deep;
            // one vacancy and one dumbbell shallow.
            let singles = 2 + (k + f) % 2;
            let mut plan: Vec<(bool, usize, bool)> = vec![(true, 1, true); singles];
            plan.extend([
                (true, 2, true),
                (false, 1, true),
                (false, 2, true),
                (true, 1, false),
                (false, 1, false),
            ]);
            for (vac, count, deep) in plan {
                let ok_depth = |d: f64| {
                    if deep {
                        d > DEEP
                    } else {
                        (0.5..2.5).contains(&d)
                    }
                };
                let mut tries = 0;
                loop {
                    tries += 1;
                    assert!(
                        tries < 100_000,
                        "no room for a defect in grain {k}: {} sites, max depth {:.1}, {} deeper than {DEEP}",
                        mine.len(),
                        mine.iter().map(|&i| sites[i].depth).fold(0.0, f64::max),
                        mine.iter().filter(|&&i| sites[i].depth > DEEP).count()
                    );
                    let i = mine[(r.f() * mine.len() as f64) as usize];
                    if !ok_depth(sites[i].depth) || removed[i] || moved.contains_key(&i) {
                        continue;
                    }
                    let pair = if count == 2 {
                        match nearest_site(i) {
                            Some(j) if ok_depth(sites[j].depth) => Some(j),
                            _ => continue,
                        }
                    } else {
                        None
                    };
                    if planted.iter().any(|p| sep2(p.p, sites[i].p) < 64.0) {
                        continue;
                    }
                    for s in std::iter::once(i).chain(pair) {
                        let site = &sites[s];
                        planted.push(Planted {
                            grain: k,
                            vacancy: vac,
                            p: site.p,
                            depth: site.depth,
                        });
                        if vac {
                            removed[s] = true;
                        } else {
                            let d = g.dumbbell_axis().map(|v| 0.3 * nn * v);
                            let p = site.p;
                            moved.insert(s, [p[0] - d[0], p[1] - d[1], p[2] - d[2]]);
                            extra.push((s, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]));
                        }
                    }
                    break;
                }
            }
        }

        // Atoms: stable ids by site key, new ids for interstitial halves.
        let mut atoms: Vec<(i64, usize, Coord3)> = Vec::new();
        let mut next: Vec<i64> = vec![0; self.grains.len()];
        for v in ids.values() {
            let k = grain_of_id(*v);
            next[k] = next[k].max(*v % 1_000_000 + 1);
        }
        for (i, s) in sites.iter().enumerate() {
            if removed[i] {
                continue;
            }
            let k = s.key.0;
            let id = *ids.entry(s.key).or_insert_with(|| {
                next[k] += 1;
                (k as i64 + 1) * 1_000_000 + next[k] - 1
            });
            atoms.push((id, k, moved.get(&i).copied().unwrap_or(s.p)));
        }
        for (n, (s, p)) in extra.iter().enumerate() {
            let k = sites[*s].key.0;
            atoms.push(((k as i64 + 1) * 1_000_000 + 900_000 + n as i64, k, *p));
        }
        let reference: Vec<(i64, usize, Coord3)> = sites
            .iter()
            .map(|st| {
                let k = st.key.0;
                let id = *ids.entry(st.key).or_insert_with(|| {
                    next[k] += 1;
                    (k as i64 + 1) * 1_000_000 + next[k] - 1
                });
                (id, k, st.p)
            })
            .collect();
        let mut r = Rng(77 + f as u64);
        for a in atoms.iter_mut() {
            for d in 0..3 {
                a.2[d] = (a.2[d] + 0.07 * r.g()).rem_euclid(l);
            }
        }
        atoms.sort_by_key(|a| a.0);
        Frame {
            atoms,
            reference,
            planted,
            own_gap,
        }
    }

    fn write_dump(&self, path: &Path, f: usize, atoms: &[(i64, usize, Coord3)]) {
        let mut s = String::new();
        let l = self.l;
        writeln!(
            s,
            "ITEM: TIMESTEP\n{f}\nITEM: NUMBER OF ATOMS\n{}\nITEM: BOX BOUNDS pp pp pp\n0 {l}\n0 {l}\n0 {l}\nITEM: ATOMS id type x y z",
            atoms.len()
        )
        .unwrap();
        for (id, k, p) in atoms {
            // Type = lattice (1 FCC, 2 BCC, 3 HCP), as an element would be.
            writeln!(s, "{id} {} {:.6} {:.6} {:.6}", k % 3 + 1, p[0], p[1], p[2]).unwrap();
        }
        std::fs::write(path, s).unwrap();
    }
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

/// A CSV as header-indexed rows.
struct Csv {
    head: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Csv {
    fn read(p: &Path) -> Self {
        let t = std::fs::read_to_string(p).unwrap();
        let mut l = t.lines();
        let head = l.next().unwrap().split(',').map(String::from).collect();
        let rows = l
            .map(|r| r.split(',').map(String::from).collect())
            .collect();
        Self { head, rows }
    }
    fn col(&self, name: &str) -> usize {
        self.head
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    }
    fn get<T: std::str::FromStr>(&self, row: &[String], name: &str) -> T
    where
        T::Err: std::fmt::Debug,
    {
        row[self.col(name)].parse().unwrap()
    }
}

/// `Grain` column of an extxyz written by `rmda segment -o`.
fn grain_column(p: &Path) -> Vec<i32> {
    let t = std::fs::read_to_string(p).unwrap();
    let mut l = t.lines();
    let n: usize = l.next().unwrap().trim().parse().unwrap();
    let head = l.next().unwrap();
    let props = head.split("Properties=").nth(1).unwrap();
    let props = props.split_whitespace().next().unwrap();
    let f: Vec<&str> = props.split(':').collect();
    let mut at = 0;
    let mut col = None;
    for c in f.chunks(3) {
        if c[0] == "Grain" {
            col = Some(at);
        }
        at += c[2].parse::<usize>().unwrap();
    }
    let col = col.expect("Grain column");
    l.take(n)
        .map(|r| r.split_whitespace().nth(col).unwrap().parse().unwrap())
        .collect()
}

/// This test's own directory: the tests run in parallel in one process, and
/// each clears its directory, so a shared one let the series test delete the
/// few-grains test's files mid-run.
fn work_dir(test: &str) -> (PathBuf, bool) {
    match std::env::var_os("RMDA_POLY3_DIR") {
        Some(d) => (PathBuf::from(d).join(test), true),
        None => (
            std::env::temp_dir().join(format!("rmda_poly3_{}_{test}", std::process::id())),
            false,
        ),
    }
}

/// `rmda segment` on frame `f`: as many grains as the construction, of the
/// right structures, each matching one construction grain on ≥ 99 % of its
/// interior atoms (> 4 Å from a facet). Returns segmentation label →
/// construction grain.
fn segment_and_map(
    poly: &Poly,
    dir: &Path,
    f: usize,
    p: &Path,
    fr: &Frame,
    report: &mut String,
) -> Vec<usize> {
    let n = poly.grains.len();
    let out = dir.join(format!("seg.{f}.xyz"));
    let table = dir.join(format!("seg.{f}.csv"));
    let (o, _) = rmda(&[
        "segment",
        p.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
        "--table",
        table.to_str().unwrap(),
    ]);
    let label = grain_column(&out);
    assert_eq!(label.len(), fr.atoms.len());
    let t = Csv::read(&table);
    assert_eq!(t.rows.len(), n, "frame {f}: {o}");
    let w = poly.weights(f);
    let mut votes = vec![vec![0usize; n]; n];
    for (a, &g) in fr.atoms.iter().zip(&label) {
        let (_, depth) = poly.owner(a.2, &w);
        if depth > 4.0 && g >= 0 {
            votes[g as usize][grain_of_id(a.0)] += 1;
        }
    }
    let mut map = Vec::new();
    let (mut wrong, mut total) = (0, 0);
    for (g, v) in votes.iter().enumerate() {
        let best = (0..n).max_by_key(|&k| v[k]).unwrap();
        wrong += v.iter().sum::<usize>() - v[best];
        total += v.iter().sum::<usize>();
        let st: i32 = t.get(&t.rows[g], "structure");
        assert_eq!(st, poly.grains[best].st, "frame {f} grain {g}");
        map.push(best);
    }
    let mut sorted = map.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "frame {f}: {map:?}");
    writeln!(
        report,
        "frame {f}: {} atoms, interior mislabelled {wrong}/{total}",
        fr.atoms.len()
    )
    .unwrap();
    assert!(
        (wrong as f64) < 0.01 * total as f64,
        "frame {f}: {wrong} of {total} interior atoms mislabelled"
    );
    map
}

#[test]
#[ignore = "expensive: run with --release --include-ignored"]
fn few_grains_meeting_their_own_images_are_cut_at_the_seam_and_counted() {
    // Three W grains in a 64 Å periodic box: grains meet their own images
    // across self-seams, as in an atomsk polycrystal with few grains. Seed 11
    // has seams with small offsets (0.36 and 0.47 Å, about 3.5 thermal
    // sigmas); 1 and 3 have large ones.
    for seed in [11, 1, 3] {
        few_grains_case(seed);
    }
}

/// One fixture of [`few_grains_meeting_their_own_images_are_cut_at_the_seam_and_counted`].
fn few_grains_case(seed: u64) {
    let (base, keep) = work_dir("few");
    let dir = base.join(format!("few{seed}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let poly = Poly::few(64.0, seed, 3);
    let fr = poly.frame(0, &mut HashMap::new());
    assert!(
        fr.own_gap.iter().any(|&g| g < 1.5),
        "the fixture must have a self-seam: {:?}",
        fr.own_gap
    );
    let path = dir.join("few.dump.0");
    poly.write_dump(&path, 0, &fr.atoms);
    if keep {
        let mut t = String::from("grain,sx,sy,sz,kind,x,y,z,depth\n");
        for (k, g) in poly.grains.iter().enumerate() {
            writeln!(
                t,
                "{k},{},{},{},seed,0,0,0,0",
                g.seed[0], g.seed[1], g.seed[2]
            )
            .unwrap();
        }
        for d in &fr.planted {
            let kind = if d.vacancy { "V" } else { "I" };
            writeln!(
                t,
                "{},0,0,0,{kind},{},{},{},{}",
                d.grain, d.p[0], d.p[1], d.p[2], d.depth
            )
            .unwrap();
        }
        std::fs::write(dir.join("truth.csv"), t).unwrap();
    }
    let mut report = String::new();
    segment_and_map(&poly, &dir, 0, &path, &fr, &mut report);

    let table = dir.join("grains.csv");
    let ddir = dir.join("defects");
    let (o, e) = rmda(&[
        "defects",
        path.to_str().unwrap(),
        "--segment",
        "--table",
        table.to_str().unwrap(),
        "--defects",
        ddir.to_str().unwrap(),
    ]);
    writeln!(report, "own-image gaps {:?}\n{o}\n{e}", fr.own_gap).unwrap();
    assert!(e.contains("meets its own periodic image"), "{report}");
    let scalar = |name: &str| -> i64 {
        o.lines()
            .find_map(|l| l.strip_prefix(&format!("{name}: ")))
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    };
    assert_eq!(scalar("defects_balance_residual"), 0, "{report}");
    let t = Csv::read(&table);
    assert_eq!(t.rows.len(), 3, "{report}");
    for r in &t.rows {
        assert_eq!(&r[t.col("status")], "measured", "{report}");
    }
    // Interior defects by position: each one reported is a planted one, and
    // every deep planted one is reported. (A planted defect 0.5–2.5 Å from a
    // facet may fall either side of the boundary width.)
    let l = poly.l;
    let near = |a: Coord3, b: Coord3, r: f64| {
        (0..3)
            .map(|k| {
                let d = a[k] - b[k];
                (d - l * (d / l).round()).powi(2)
            })
            .sum::<f64>()
            < r * r
    };
    let interior = |file: &str, boundary_col: usize| -> Vec<Coord3> {
        let txt = std::fs::read_to_string(ddir.join(file)).unwrap();
        txt.lines()
            .skip(2)
            .map(|r| r.split_whitespace().collect::<Vec<_>>())
            .filter(|f| f.len() > boundary_col && f[boundary_col] == "0")
            .map(|f| std::array::from_fn(|k| f[1 + k].parse().unwrap()))
            .collect()
    };
    for (file, col, vac, radius) in [
        ("vacancies.xyz", 5, true, 1.0),
        ("interstitials.xyz", 7, false, 1.3),
    ] {
        let got = interior(file, col);
        let planted: Vec<&Planted> = fr.planted.iter().filter(|p| p.vacancy == vac).collect();
        for g in &got {
            assert!(
                planted.iter().any(|p| near(p.p, *g, radius)),
                "{file}: interior defect at {g:?} was not planted\n{report}"
            );
        }
        for p in planted.iter().filter(|p| p.depth > DEEP) {
            assert!(
                got.iter().any(|g| near(p.p, *g, radius)),
                "{file}: planted interior defect at {:?} not reported\n{report}",
                p.p
            );
        }
        writeln!(report, "{file}: {} interior reported", got.len()).unwrap();
    }
    eprintln!("{report}");
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
#[ignore = "expensive: run with --release --include-ignored"]
fn three_lattice_polycrystal_series_per_grain_defects() {
    let (dir, keep) = work_dir("series");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let poly = Poly::new(90.0, 2026);
    let n_frames = 3;
    let mut ids = HashMap::new();
    let mut frames = Vec::new();
    let mut paths = Vec::new();
    let mut truth = String::from("frame,grain,structure,kind,x,y,z,depth\n");
    for f in 0..n_frames {
        let fr = poly.frame(f, &mut ids);
        let p = dir.join(format!("poly3.dump.{f}"));
        poly.write_dump(&p, f, &fr.atoms);
        if keep {
            poly.write_dump(&dir.join(format!("reference.{f}.dump")), f, &fr.reference);
        }
        for d in &fr.planted {
            writeln!(
                truth,
                "{f},{},{},{},{:.4},{:.4},{:.4},{:.3}",
                d.grain,
                poly.grains[d.grain].st,
                if d.vacancy { "V" } else { "I" },
                d.p[0],
                d.p[1],
                d.p[2],
                d.depth
            )
            .unwrap();
        }
        paths.push(p);
        frames.push(fr);
    }
    std::fs::write(dir.join("truth_defects.csv"), &truth).unwrap();
    let mut gt = String::from(
        "frame,grain,structure,a,sx,sy,sz,weight,r00,r01,r02,r10,r11,r12,r20,r21,r22\n",
    );
    for f in 0..n_frames {
        let w = poly.weights(f);
        for (k, g) in poly.grains.iter().enumerate() {
            let m = g.rot.m;
            writeln!(
                gt,
                "{f},{k},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                g.st,
                g.a,
                g.seed[0],
                g.seed[1],
                g.seed[2],
                w[k],
                m[0][0],
                m[0][1],
                m[0][2],
                m[1][0],
                m[1][1],
                m[1][2],
                m[2][0],
                m[2][1],
                m[2][2]
            )
            .unwrap();
        }
    }
    std::fs::write(dir.join("truth_grains.csv"), gt).unwrap();
    let mut report = String::new();

    // 1. Segmentation per frame: eight grains of the right structures, each
    // matching one construction grain on ≥ 99 % of interior atoms.
    let seg_to_truth: Vec<Vec<usize>> = paths
        .iter()
        .zip(&frames)
        .enumerate()
        .map(|(f, (p, fr))| segment_and_map(&poly, &dir, f, p, fr, &mut report))
        .collect();

    // 2. The series.
    let out = dir.join("series");
    let mut args: Vec<String> = vec!["defects".into()];
    args.extend(paths.iter().map(|p| p.to_str().unwrap().to_owned()));
    args.extend(
        ["--segment", "--defects-xyz", "--out", out.to_str().unwrap()]
            .iter()
            .map(|s| s.to_string()),
    );
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let (o, e) = rmda(&argv);
    writeln!(report, "{o}\n{e}").unwrap();

    let summary = Csv::read(&out.join("summary.csv"));
    assert_eq!(summary.rows.len(), n_frames);
    for r in &summary.rows {
        assert_eq!(summary.get::<i64>(r, "balance_residual"), 0, "{report}");
        assert_eq!(summary.get::<usize>(r, "grains"), 8, "{report}");
        assert_eq!(summary.get::<usize>(r, "grains_measured"), 8, "{report}");
    }

    // Stable ids: frame 0 has eight new grains, later frames continue them all.
    let ev = Csv::read(&out.join("grain_events.csv"));
    for r in &ev.rows {
        let f: usize = ev.get(r, "file_index");
        let kind = &r[ev.col("event")];
        assert_eq!(kind, if f == 0 { "new" } else { "continued" }, "{r:?}");
    }
    assert_eq!(ev.rows.len(), 8 * n_frames);

    // Per grain: exact interior counts, the same stable id for the same
    // construction grain in every frame, sensible boundary counts.
    let grains = Csv::read(&out.join("grains.csv"));
    let mut stable_of_truth: HashMap<usize, i64> = HashMap::new();
    let mut gb_totals: std::collections::BTreeMap<usize, (i64, i64)> = Default::default();
    for r in &grains.rows {
        let f: usize = grains.get(r, "file_index");
        let g: usize = grains.get(r, "grain");
        let k = seg_to_truth[f][g];
        let stable: i64 = grains.get(r, "stable_grain");
        assert_eq!(
            *stable_of_truth.entry(k).or_insert(stable),
            stable,
            "construction grain {k} changed stable id"
        );
        assert_eq!(&r[grains.col("status")], "measured", "{r:?}\n{report}");
        let deep = |vac: bool| {
            frames[f]
                .planted
                .iter()
                .filter(|p| p.grain == k && p.vacancy == vac && p.depth > DEEP)
                .count() as i64
        };
        let (v, i): (i64, i64) = (grains.get(r, "vacancies"), grains.get(r, "interstitials"));
        let (gv, gi, gs): (i64, i64, i64) = (
            grains.get(r, "gb_vacancies"),
            grains.get(r, "gb_interstitials"),
            grains.get(r, "gb_sites"),
        );
        writeln!(
            report,
            "frame {f} grain {g} (truth {k}, stable {stable}, st {}): V {v}/{} I {i}/{} gb V {gv} I {gi} of {gs} sites",
            poly.grains[k].st,
            deep(true),
            deep(false)
        )
        .unwrap();
        assert_eq!((v, i), (deep(true), deep(false)), "{report}");
        // Boundary class: the shallow defects plus whatever the mismatched
        // boundary itself holds; a small fraction of the boundary sites.
        assert!((gv + gi) * 10 < gs, "{report}");
        let t = gb_totals.entry(f).or_insert((0, 0));
        t.0 += gv;
        t.1 += gi;
    }
    // A dumbbell planted 0.5–2.5 Å from a boundary may land in the next
    // grain's count or fill a boundary vacancy, so only frame totals are
    // checked: every shallow vacancy is seen, and the boundary class does
    // not invent interstitials.
    for (f, (gv, gi)) in gb_totals {
        let shallow = |vac: bool| {
            frames[f]
                .planted
                .iter()
                .filter(|p| p.vacancy == vac && p.depth < DEEP)
                .count() as i64
        };
        writeln!(
            report,
            "frame {f}: boundary V {gv} (planted {}), I {gi} (planted {})",
            shallow(true),
            shallow(false)
        )
        .unwrap();
        assert!(gv >= shallow(true), "{report}");
        assert!(
            gi + 2 >= shallow(false) && gi <= 2 * shallow(false),
            "{report}"
        );
    }
    // Grain 0 grows, grain 1 shrinks.
    let size = |f: usize, k: usize| -> i64 {
        grains
            .rows
            .iter()
            .find(|r| {
                grains.get::<usize>(r, "file_index") == f
                    && seg_to_truth[f][grains.get::<usize>(r, "grain")] == k
            })
            .map(|r| grains.get(r, "n_atoms"))
            .unwrap()
    };
    assert!(size(0, 0) < size(1, 0) && size(1, 0) < size(2, 0));
    assert!(size(0, 1) > size(1, 1) && size(1, 1) > size(2, 1));
    eprintln!("{report}");
    if !keep {
        let _ = std::fs::remove_dir_all(&dir);
    }
}
