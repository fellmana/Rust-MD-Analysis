//! `rmda bench`: per-stage throughput on a synthetic crystal (design
//! §23.4), as a baseline to compare machines, thread counts and builds.
//!
//! The crystal is built in memory (no I/O in the numbers): a periodic BCC or
//! FCC block of about `--atoms` atoms with Gaussian thermal noise and 0.1 %
//! of its sites left empty. Each stage runs on it in turn and reports its
//! wall time, atoms per second and the peak anonymous memory while it ran.
//! The defect census must find exactly the planted vacancies; a run where it
//! does not says so, because a fast wrong answer is not a benchmark.

use crate::prelude::*;
use rmda_core::analysis::defects::{self, DefectsConfig, ReferenceMode};
use rmda_core::analysis::grain_seg::{self, SegConfig};

/// One stage's measurement.
struct Stage {
    name: &'static str,
    wall: f64,
    peak_anon: u64,
    note: String,
}

/// xorshift64*: reproducible noise without a dependency.
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

/// The synthetic frame and how many vacancies it holds.
fn crystal(atoms: usize, bcc: bool, sigma: f64) -> (Frame, usize) {
    let (a, basis): (f64, &[[f64; 3]]) = if bcc {
        (3.165, &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    } else {
        (
            3.615,
            &[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
        )
    };
    let n = ((atoms as f64 / basis.len() as f64).cbrt().round() as usize).max(4);
    let l = n as f64 * a;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let sites = n * n * n * basis.len();
    // Every 1000th site empty, spread through the block.
    let vacant = |s: usize| s % 1000 == 499;
    let mut p = rmda_core::data::particle::ParticleData::with_capacity(sites);
    let mut id = 0i64;
    let mut s = 0usize;
    let mut holes = 0usize;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for b in basis {
                    s += 1;
                    if vacant(s) {
                        holes += 1;
                        continue;
                    }
                    id += 1;
                    let x = [
                        (i as f64 + b[0]) * a + sigma * rng.g(),
                        (j as f64 + b[1]) * a + sigma * rng.g(),
                        (k as f64 + b[2]) * a + sigma * rng.g(),
                    ];
                    p.add_atom(x, id, 1);
                }
            }
        }
    }
    let cell = rmda_core::cell::UnitCell::orthorhombic(l, l, l);
    (Frame::new(cell, p), holes)
}

/// Run `f`, timing it and recording the peak anonymous memory while it ran.
fn stage<T>(name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<(T, Stage)> {
    rmda_core::prof::reset_peak_anon();
    let t = std::time::Instant::now();
    let out = f()?;
    let wall = t.elapsed().as_secs_f64();
    Ok((
        out,
        Stage {
            name,
            wall,
            peak_anon: rmda_core::prof::peak_anon_bytes(),
            note: String::new(),
        },
    ))
}

/// `rmda bench` (dispatched in main: it reads no input file).
pub(crate) fn command(cli: &Cli, par: Par) -> Result<()> {
    let Cmd::Bench {
        atoms,
        structure,
        sigma,
        json,
    } = &cli.cmd
    else {
        unreachable!("dispatched in main");
    };
    let bcc = structure == "bcc";
    if !bcc && structure != "fcc" {
        return Err(rmda_core::error::Error::usage(format!(
            "--structure {structure}: expected bcc or fcc"
        )));
    }
    if !(sigma.is_finite() && *sigma >= 0.0 && *sigma < 0.3) {
        return Err(rmda_core::error::Error::usage(format!(
            "--sigma {sigma}: expected 0 <= sigma < 0.3 Å"
        )));
    }
    // The memory sampler gives each stage its peak.
    rmda_core::prof::Profile::set_enabled(true);
    let (frame, holes) = crystal(*atoms, bcc, *sigma);
    let n = frame.natoms();
    eprintln!(
        "bench: {} {n} atoms ({holes} vacancies), sigma {sigma} Å, {} thread(s)",
        structure, par.threads
    );
    let (cell, pos) = (&frame.cell, &frame.particles.positions);
    let mut stages = Vec::new();

    let (ptm, mut s) = stage("ptm", || {
        Ptm {
            rmsd_cutoff: 0.1,
            flags: check::OVITO_DEFAULT,
        }
        .compute(cell, pos, &par)
    })?;
    let lattice = ptm
        .iter()
        .filter(|a| a.structure() != structure::OTHER)
        .count();
    s.note = format!("{:.1} % classified", 100.0 * lattice as f64 / n as f64);
    stages.push(s);

    let (_, mut s) = stage("cna", || {
        let cna = Cna::default();
        let radius = cna.cutoffs_for(&frame).into_iter().fold(0.0_f64, f64::max);
        let cl = shared_cell_list(&frame, radius)?;
        let nk = NeighborhoodContext { cell: cl.as_ref() };
        let actx = AnalysisContext {
            frame: &frame,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut sum = Summary::default();
        cna.run(&actx, &mut out, &mut sum)
    })?;
    s.note = String::new();
    stages.push(s);

    let (seg, mut s) = stage("segment", || {
        grain_seg::segment(cell, pos, &ptm, &SegConfig::default(), &par)
    })?;
    s.note = format!("{} grain(s)", seg.grains.len());
    stages.push(s);

    let one_grain = vec![0i32; n];
    for (name, reference) in [
        ("defects.global", ReferenceMode::Global),
        ("defects.local", ReferenceMode::Local),
    ] {
        let cfg = DefectsConfig {
            reference,
            ..Default::default()
        };
        let (r, mut s) = stage(name, || {
            defects::run(cell, pos, &one_grain, &ptm, &cfg, &par)
        })?;
        let (i, b) = r.totals();
        let v = i.vacancies + b.vacancies;
        let int = i.interstitials + b.interstitials;
        s.note = if v as usize == holes && int == 0 {
            format!("{v} V / 0 I, as planted")
        } else {
            format!("{v} V / {int} I, PLANTED {holes} V / 0 I")
        };
        stages.push(s);
    }

    let (_, s) = stage("rdf", || {
        let rdf = Rdf::new(6.0, 0.05);
        rdf.validate(Some(cell))?;
        let nk = NeighborhoodContext { cell: None };
        let actx = AnalysisContext {
            frame: &frame,
            nkeys: &nk,
            par: &par,
        };
        let mut out = PropertyStore::new();
        let mut sum = Summary::default();
        rdf.run(&actx, &mut out, &mut sum)
    })?;
    stages.push(s);

    println!(
        "{:<16} {:>9} {:>14} {:>11}  note",
        "stage", "wall_s", "atoms/s", "peak_MB"
    );
    for s in &stages {
        println!(
            "{:<16} {:>9.3} {:>14.0} {:>11.1}  {}",
            s.name,
            s.wall,
            n as f64 / s.wall.max(1e-9),
            s.peak_anon as f64 / 1048576.0,
            s.note
        );
    }
    if let Some(path) = json {
        use serde_json::json;
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_owned())
            .unwrap_or_default();
        let v = json!({
            "rmda_version": env!("CARGO_PKG_VERSION"),
            "host": host,
            "threads": par.threads,
            "structure": structure,
            "atoms": n,
            "vacancies": holes,
            "sigma": sigma,
            "stages": stages.iter().map(|s| json!({
                "name": s.name,
                "wall_s": s.wall,
                "atoms_per_s": n as f64 / s.wall.max(1e-9),
                "peak_anon_bytes": s.peak_anon,
                "note": s.note,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            path,
            serde_json::to_string_pretty(&v).unwrap_or_default() + "\n",
        )?;
        eprintln!("bench: written to {}", path.display());
    }
    // A census that missed the planted defects makes the run fail: the
    // numbers above would be timing a wrong answer.
    if stages.iter().any(|s| s.note.contains("PLANTED")) {
        return Err(rmda_core::error::Error::Analysis(
            "bench: a defect census did not find the planted defects".into(),
        ));
    }
    Ok(())
}
