//! PTM validation on large, physically realistic MD snapshots.
//!
//! Same fixtures as `cna_reference.rs`, but exercising PTM's full path
//! (gather -> normalise -> convex hull -> graph lookup -> QCP fit) rather than
//! CNA's signature counts:
//!
//! | file | structure | atoms | why it is a good test |
//! |---|---|---|---|
//! | `data.final-lastframe-Al-2000` | FCC | 500 000 | few HCP stacking faults among ~94 % FCC |
//! | `data.final-lastframe-W-2000` | BCC | 500 094 | BCC at high T, huge OTHER fraction at the melt-free box surface |
//! | `data.final-lastframe-Zr-2000` | HCP | 492 800 | HCP is the hardest case for the graph tie-break |
//! | `lastframe-HEA-poly.data` | FCC HEA polycrystal | 42 432 | 5 species, grain boundaries, shifted box origin |
//!
//! These snapshots are not committed (~240 MB), so the test skips itself when
//! they are absent.
//!
//! The assertions are physical rather than a restatement of another tool:
//! 2000 K bulk Al, W and Zr are overwhelmingly their own lattice. The HEA case
//! additionally cross-checks PTM against CNA, whose histogram here *is*
//! OVITO-verified. Index-for-index agreement with OVITO itself is a separate
//! concern, checked in `ptm_oracle.rs` against the cached oracles.

use rmda_core::analysis::cna::{Cna, CnaMode};
use rmda_core::analysis::ptm::{check, structure, Ptm};
use rmda_core::analysis::{Analysis, AnalysisContext, NeighborhoodContext, Summary};
use rmda_core::data::frame::Frame;
use rmda_core::data::property::PropertyStore;
use rmda_core::exec::Par;
use rmda_core::io::read_frames;
use std::path::{Path, PathBuf};

mod common;

const PTM_NAMES: [&str; 6] = ["OTHER", "FCC", "HCP", "BCC", "ICO", "SC"];
const CNA_NAMES: [&str; 5] = ["OTHER", "FCC", "HCP", "BCC", "ICO"];

fn reference_dir() -> PathBuf {
    common::reference_dir()
}

struct Result_ {
    ptm: Vec<i32>,
    cna: Vec<u8>,
    /// Every design §14.2 output, for the thread-determinism gate.
    all: Vec<(&'static str, Vec<f64>)>,
}

fn analyse(frame: &Frame) -> Result_ {
    analyse_at(frame, None)
}

/// Run the full PTM+CNA pipeline. `threads` of `None` uses the default pool,
/// `Some(1)` forces the sequential path, and any other value builds a pool of
/// exactly that size.
fn analyse_at(frame: &Frame, threads: Option<usize>) -> Result_ {
    let cna = Cna::new(CnaMode::Adaptive);
    // Both build their own k-NN index.
    let nkeys = NeighborhoodContext { cell: None };
    let par = match threads {
        Some(t) => Par::new(t),
        None => Par::default(),
    };
    let ctx = AnalysisContext {
        frame,
        nkeys: &nkeys,
        par: &par,
    };

    let mut out = PropertyStore::new();
    let mut summary = Summary::default();
    // Run at the shipped defaults, so this test exercises the cutoff the user
    // actually gets rather than a permissive stand-in. That means the OVITO
    // check set (upstream's own default omits SC) and the 0.1 RMSD cutoff.
    let ptm = Ptm {
        rmsd_cutoff: 0.1,
        flags: check::OVITO_DEFAULT,
    };
    ptm.run(&ctx, &mut out, &mut summary).unwrap();

    // Design 13.3 asks for six per-atom outputs. Check all six are actually
    // published, with the right arity, before anything downstream depends on
    // the names.
    for (name, components) in [
        ("StructureType", 1),
        ("StructureRmsd", 1),
        ("StructureScale", 1),
        ("StructureInteratomicDistance", 1),
        ("StructureLatticeConstant", 1),
        ("StructureOrientation", 4),
    ] {
        let p = out
            .get(name)
            .unwrap_or_else(|| panic!("missing PTM output `{name}`"));
        assert_eq!(
            p.len(),
            ctx.frame.natoms(),
            "`{name}` has {} entries, expected one per atom",
            p.len()
        );
        assert_eq!(
            p.components, components,
            "`{name}` should have {components} component(s)"
        );
    }
    // The orientation must be a real quaternion, not a placeholder.
    let q = out.get("StructureOrientation").unwrap().as_f64().unwrap();
    for (i, chunk) in q.chunks_exact(4).enumerate() {
        let n2: f64 = chunk.iter().map(|c| c * c).sum();
        assert!(
            (n2 - 1.0).abs() < 1e-9,
            "atom {i}: |q|^2 = {n2}, orientation is not a unit quaternion"
        );
    }

    let mut out2 = PropertyStore::new();
    let mut summary2 = Summary::default();
    cna.run(&ctx, &mut out2, &mut summary2).unwrap();

    Result_ {
        ptm: out
            .get("StructureType")
            .unwrap()
            .as_u8()
            .unwrap()
            .iter()
            .map(|&v| v as i32)
            .collect(),
        cna: out2.get("StructureType").unwrap().as_u8().unwrap().clone(),
        all: vec![
            (
                "StructureRmsd",
                out.get("StructureRmsd").unwrap().as_f64().unwrap().clone(),
            ),
            (
                "StructureScale",
                out.get("StructureScale").unwrap().as_f64().unwrap().clone(),
            ),
            (
                "StructureInteratomicDistance",
                out.get("StructureInteratomicDistance")
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    .clone(),
            ),
            (
                "StructureLatticeConstant",
                out.get("StructureLatticeConstant")
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    .clone(),
            ),
            (
                "StructureOrientation",
                out.get("StructureOrientation")
                    .unwrap()
                    .as_f64()
                    .unwrap()
                    .clone(),
            ),
        ],
    }
}

fn load(path: &Path) -> Option<Frame> {
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return None;
    }
    Some(read_frames(path).unwrap().into_iter().next().unwrap())
}

fn histogram_ptm(codes: &[i32]) -> [usize; 6] {
    let mut h = [0usize; 6];
    for &c in codes {
        h[c as usize] += 1;
    }
    h
}

fn histogram_cna(codes: &[u8]) -> [usize; 5] {
    let mut h = [0usize; 5];
    for &c in codes {
        h[c as usize] += 1;
    }
    h
}

fn pct(part: usize, total: usize) -> f64 {
    100.0 * part as f64 / total as f64
}

fn render_ptm(h: &[usize; 6]) -> String {
    (0..6)
        .filter(|&i| h[i] > 0)
        .map(|i| format!("{}={:.2}%", PTM_NAMES[i], pct(h[i], h.iter().sum())))
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_cna(h: &[usize; 5]) -> String {
    (0..5)
        .filter(|&i| h[i] > 0)
        .map(|i| format!("{}={:.2}%", CNA_NAMES[i], pct(h[i], h.iter().sum())))
        .collect::<Vec<_>>()
        .join(" ")
}

/// For each snapshot: the structure that must dominate, the smallest share we
/// accept for it, and the same expectation for CNA.
///
/// These are physical assertions rather than a restatement of another tool:
/// 2000 K bulk Al, W and Zr are overwhelmingly their own lattice. The CNA
/// column doubles as a cross-check, because the two algorithms are independent
/// and must broadly agree on which atoms sit in a well-formed lattice.
///
/// They are expected to *disagree* on the defect population. PTM's fit-based
/// cutoff is stricter, so on the alloy polycrystal - where CNA's 12/14
/// signature is weakest - PTM must report no more OTHER atoms than CNA does.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn ptm_classifies_the_reference_structures() {
    let dir = reference_dir();
    if !dir.is_dir() {
        eprintln!("skipping: {} not present", dir.display());
        return;
    }
    // (file, expected PTM structure, minimum % for PTM, expected CNA code)
    let cases: &[(&str, i32, f64, u8)] = &[
        ("data.final-lastframe-Al-2000", structure::FCC, 90.0, 1),
        ("data.final-lastframe-W-2000", structure::BCC, 93.0, 3),
        ("data.final-lastframe-Zr-2000", structure::HCP, 95.0, 2),
        ("lastframe-HEA-poly.data", structure::FCC, 59.0, 1),
    ];
    for (name, want_ptm, min_pct, want_cna) in cases {
        let Some(frame) = load(&dir.join(name)) else {
            continue;
        };
        let r = analyse(&frame);
        let hp = histogram_ptm(&r.ptm);
        let hc = histogram_cna(&r.cna);
        let n = hp.iter().sum::<usize>();
        assert_eq!(n, r.ptm.len(), "{name}: histogram misses atoms");
        assert_eq!(n, r.cna.len(), "{name}: differing atom counts");
        println!(
            "{name} (n={n})\n  PTM: {}\n  CNA: {}",
            render_ptm(&hp),
            render_cna(&hc)
        );

        assert!(
            pct(hp[*want_ptm as usize], n) >= *min_pct,
            "{name}: expected {} at >= {min_pct}%, got {}",
            PTM_NAMES[*want_ptm as usize],
            render_ptm(&hp)
        );

        // Both algorithms must pick the same dominant lattice.
        let top_ptm = (1..6).max_by_key(|&i| hp[i]).unwrap();
        let top_cna = (1..5).max_by_key(|&i| hc[i]).unwrap();
        assert_eq!(
            top_ptm as i32, *want_ptm,
            "{name}: PTM dominant {}",
            PTM_NAMES[top_ptm]
        );
        assert_eq!(
            top_cna, *want_cna as usize,
            "{name}: CNA dominant {}",
            CNA_NAMES[top_cna]
        );

        assert!(
            hp[structure::OTHER as usize] <= hc[0],
            "{name}: PTM OTHER {:.2}% exceeds CNA OTHER {:.2}%",
            pct(hp[0], n),
            pct(hc[0], n)
        );
    }
}

/// Thread-count independence (design §25.1 determinism gate).
///
/// PTM's `map_collect` writes each atom's result into a slot indexed by atom
/// number and drains the slots in order, so scheduling cannot change the
/// result. This asserts that on real data rather than trusting the argument,
/// and it checks the floating-point outputs too -- those are where a genuine
/// race (two atoms sharing scratch state, say) shows up as a last-bit
/// difference that a histogram comparison would hide.
///
/// The fixture is the 42k-atom HEA polycrystal rather than a 500k snapshot to
/// keep this in the default test run; `ptm_determinism_at_scale` covers the
/// large case separately.
#[test]
#[ignore = "reference-data oracle: set RMDA_REFERENCE_DIR, run with --include-ignored"]
fn ptm_is_independent_of_thread_count() {
    let dir = reference_dir();
    let Some(frame) = load(&dir.join("lastframe-HEA-poly.data")) else {
        eprintln!("skipping: reference snapshot not present");
        return;
    };
    let seq = analyse_at(&frame, Some(1));
    for threads in [3usize, 12] {
        let par = analyse_at(&frame, Some(threads));
        assert_eq!(
            seq.ptm, par.ptm,
            "StructureType differs between 1 and {threads} threads"
        );
        assert_eq!(
            seq.cna, par.cna,
            "CNA StructureType differs between 1 and {threads} threads"
        );
        for ((name, a), (name2, b)) in seq.all.iter().zip(par.all.iter()) {
            assert_eq!(name, name2);
            assert_eq!(
                a.len(),
                b.len(),
                "{name}: length differs between 1 and {threads} threads"
            );
            if a != b {
                let first = a
                    .iter()
                    .zip(b.iter())
                    .position(|(x, y)| x != y)
                    .expect("lengths match");
                panic!(
                    "{name} differs between 1 and {threads} threads at index \
                     {first}: {} vs {}",
                    a[first], b[first]
                );
            }
        }
    }
}

/// Same gate on a 500k snapshot, where the cell list and neighbour search are
/// large enough to hide a data race the small fixture misses. Ignored by
/// default because a single-threaded PTM run takes ~14 s.
#[test]
#[ignore = "500k-atom determinism check; run in release"]
fn ptm_determinism_at_scale() {
    let dir = reference_dir();
    let Some(frame) = load(&dir.join("data.final-lastframe-Al-2000")) else {
        eprintln!("skipping: reference snapshot not present");
        return;
    };
    let seq = analyse_at(&frame, Some(1));
    let par = analyse_at(&frame, Some(12));
    assert_eq!(seq.ptm, par.ptm, "StructureType differs at 500k atoms");
    for ((name, a), (_, b)) in seq.all.iter().zip(par.all.iter()) {
        assert_eq!(
            a, b,
            "{name} differs at 500k atoms between 1 and 12 threads"
        );
    }
}
