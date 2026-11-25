//! The census against defects known by construction (design §16.8a):
//! `tools/gen_ground_truth.py` builds BCC (W), FCC (Ni) and HCP (Zr) blocks
//! holding prismatic loops of both kinds and both of the structure's Burgers
//! vector families, voids, vacancies, divacancies and dumbbells, placed
//! close so their clusters meet, and `tools/eval_ground_truth.py` assigns
//! every vacancy and interstitial found to the feature it belongs to. The
//! limits below are what the census achieves; a change that makes any
//! of them worse fails here.
//!
//! Needs python3 with numpy (as the Python binding tests do): skipped
//! without it. Each test generates and counts blocks of 10^5-10^6 atoms, so
//! they are `#[ignore]`d: `cargo test --release -- --include-ignored` runs
//! them (minutes each in a debug build).

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_ground_truth_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Whether the test can run here: python3 with numpy.
fn runnable() -> bool {
    let ok = Command::new("python3")
        .args(["-c", "import numpy"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED ground truth: python3 with numpy not found");
    }
    ok
}

/// Generates a block with `gen` arguments and scores each configuration;
/// returns the evaluator's JSON.
fn score(tag: &str, gen: &[&str], configs: &[(&str, &str)]) -> serde_json::Value {
    let dir = tempdir(tag);
    let frame = dir.join("block.xyz");
    let tools = repo_root().join("tools");
    let out = Command::new("python3")
        .arg(tools.join("gen_ground_truth.py"))
        .arg(&frame)
        .args(gen)
        .output()
        .expect("python3");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = dir.join("score.json");
    let mut cmd = Command::new("python3");
    cmd.arg(tools.join("eval_ground_truth.py"))
        .arg(&frame)
        .arg("--rmda")
        .arg(env!("CARGO_BIN_EXE_rmda"))
        .arg("--keep")
        .arg(dir.join("out"))
        .arg("--json")
        .arg(&json);
    for (name, args) in configs {
        // A leading space: argparse reads a value starting with `--` as an
        // option.
        cmd.arg("--config").arg(name).arg(format!(" {args}"));
    }
    let out = cmd.output().expect("python3");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("{text}");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    v
}

/// A configuration's error for a feature kind: (sum |dV|, sum |dI|).
fn err(v: &serde_json::Value, config: &str, kind: &str) -> (i64, i64) {
    let e = &v["scores"][config]["by_kind"][kind];
    (
        e["absV"].as_i64().unwrap_or(0),
        e["absI"].as_i64().unwrap_or(0),
    )
}

fn spurious(v: &serde_json::Value, config: &str) -> (i64, i64) {
    let s = &v["scores"][config];
    (
        s["spurious_V"].as_i64().unwrap(),
        s["spurious_I"].as_i64().unwrap(),
    )
}

/// Every kind under every configuration: |dV| and |dI| at most the limit
/// `limits` gives it (config, kind, max |dV|, max |dI|), 0 otherwise; no
/// spurious vacancy or interstitial.
fn check(v: &serde_json::Value, configs: &[&str], limits: &[(&str, &str, i64, i64)]) {
    for &config in configs {
        for kind in KINDS {
            let (mv, mi) = limits
                .iter()
                .find(|l| l.0 == config && l.1 == kind)
                .map_or((0, 0), |l| (l.2, l.3));
            let (dv, di) = err(v, config, kind);
            assert!(
                dv <= mv && di <= mi,
                "{config} {kind}: |dV| {dv} (limit {mv}), |dI| {di} (limit {mi})"
            );
        }
        let (sv, si) = spurious(v, config);
        assert!(sv == 0 && si == 0, "{config}: spurious V {sv}, I {si}");
    }
}

/// A periodic block (no free surface, as the 128M frame): 10 loops of both
/// families and kinds, voids and point defects, many of them 4-8 A from a
/// loop. Without linking the fragments again, local loses 33 of the 240
/// vacancies in its a/2<111> vacancy loops.
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn a_periodic_block_of_crowded_loops_is_counted_by_feature() {
    if !runnable() {
        return;
    }
    let v = score(
        "periodic",
        &[
            "--periodic",
            "--cells",
            "60",
            "--loops",
            "10",
            "--loop-radius",
            "2",
            "4",
            "--voids",
            "3",
            "--vacancies",
            "20",
            "--divacancies",
            "6",
            "--dumbbells",
            "10",
            "--close",
            "10",
            "--gap",
            "4",
            "--sigma",
            "0.045",
            "--seed",
            "3",
        ],
        &[
            ("global", "--reference global"),
            ("global_ra", "--reference global --resolve-ambiguous"),
            ("periodic", "--reference periodic"),
            ("local", "--reference local"),
            ("partial", "--reference local --local-merge partial"),
        ],
    );
    // Global (Wigner-Seitz) puts one Frenkel pair inside one interstitial
    // loop, where the crystal is displaced by nearly b/2 from the lattice;
    // with --resolve-ambiguous (and periodic, which is that past the
    // displacement gates), local and partial are exact.
    check(
        &v,
        &["global", "global_ra", "periodic", "local", "partial"],
        &[("global", "interstitial_loop", 1, 1)],
    );
}

/// A free-standing block with a<001> and a/2<111> loops near voids and
/// point defects (the surface layers of a free block, unlike a periodic
/// one, are defect atoms).
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn a_free_block_of_loops_voids_and_point_defects_is_counted_by_feature() {
    if !runnable() {
        return;
    }
    let v = score(
        "free",
        &[
            "--cells",
            "50",
            "--loops",
            "8",
            "--voids",
            "3",
            "--vacancies",
            "20",
            "--divacancies",
            "6",
            "--dumbbells",
            "10",
            "--close",
            "12",
            "--sigma",
            "0.05",
            "--seed",
            "3",
        ],
        &[
            ("global", "--reference global"),
            ("local", "--reference local"),
            ("partial", "--reference local --local-merge partial"),
        ],
    );
    check(
        &v,
        &["global", "local", "partial"],
        &[
            ("global", "interstitial_loop", 2, 2),
            ("local", "vacancy_loop", 2, 2),
            ("partial", "vacancy_loop", 1, 1),
        ],
    );
}

/// FCC (Ni): perfect prismatic loops b = a/2<110> on {110} and b = a<001>,
/// voids, vacancies, divacancies and <100> dumbbells, in a free block.
/// Without linking the fragments again, local loses 140 of 485 vacancies
/// in loops and 13 of 387 interstitials.
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn an_fcc_block_of_crowded_loops_is_counted_by_feature() {
    if !runnable() {
        return;
    }
    let v = score(
        "fcc",
        &[
            "--structure",
            "fcc",
            "--cells",
            "40",
            "--loops",
            "8",
            "--loop-radius",
            "2",
            "3.5",
            "--voids",
            "3",
            "--vacancies",
            "20",
            "--divacancies",
            "6",
            "--dumbbells",
            "10",
            "--close",
            "12",
            "--gap",
            "4",
            "--sigma",
            "0.045",
            "--seed",
            "3",
        ],
        &[
            ("global", "--reference global"),
            ("local", "--reference local"),
            ("partial", "--reference local --local-merge partial"),
        ],
    );
    check(&v, &["global", "local", "partial"], &LIMITS_FCC);
}

/// FCC faulted Frank loops (b = a/3<111>: one (111) layer inserted in its
/// third stacking position, or removed), in a periodic box of 211 A, which
/// is not a multiple of the local census's region grid (~154 A). Before the
/// grid's keys wrapped as its lookup did, a region centred past the last
/// whole cell went unfound by the atom-by-atom count, which counted its
/// sites again: this block's Frank vacancy loops counted +92 of 217.
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn fcc_frank_loops_are_counted_by_feature() {
    if !runnable() {
        return;
    }
    let v = score(
        "frank",
        &[
            "--structure",
            "fcc",
            "--periodic",
            "--cells",
            "60",
            "--loops",
            "6",
            "--loop-types",
            "frank",
            "--loop-radius",
            "2",
            "4",
            "--voids",
            "0",
            "--vacancies",
            "0",
            "--divacancies",
            "0",
            "--dumbbells",
            "0",
            "--close",
            "0",
            "--gap",
            "15",
            "--sigma",
            "0.045",
            "--seed",
            "2",
        ],
        &[
            ("global", "--reference global"),
            ("patch", "--reference patch"),
            ("periodic", "--reference periodic"),
            ("local", "--reference local"),
        ],
    );
    // Periodic (global with ambiguous atoms resolved) puts one Frenkel pair
    // in one Frank vacancy loop's faulted layer.
    check(
        &v,
        &["global", "patch", "periodic", "local"],
        &[("periodic", "vacancy_loop", 1, 1)],
    );
}

/// An HEA: CoCrFeMnNi as a random equiatomic FCC alloy (a = 3.59 A) with
/// 0.06 A of static distortion on top of 0.06 A of thermal noise, all three
/// FCC loop families, voids and point defects, in a free block (the film's
/// top is a surface). The counts do not depend on the species; and with no
/// chemical order antisites are not defined, so `--antisites` reports none
/// (it reported 80 % of the atoms as antisites before).
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn an_hea_block_is_counted_by_feature_and_has_no_antisites() {
    if !runnable() {
        return;
    }
    let v = score(
        "hea",
        &[
            "--structure",
            "fcc",
            "--lattice-a",
            "3.59",
            "--species",
            "5",
            "--distortion",
            "0.06",
            "--cells",
            "40",
            "--loops",
            "8",
            "--loop-types",
            "110,001,frank",
            "--loop-radius",
            "2",
            "3.5",
            "--voids",
            "3",
            "--vacancies",
            "20",
            "--divacancies",
            "6",
            "--dumbbells",
            "10",
            "--close",
            "12",
            "--gap",
            "4",
            "--sigma",
            "0.06",
            "--seed",
            "4",
        ],
        &[
            ("global", "--reference global"),
            ("global_ra", "--reference global --resolve-ambiguous"),
            ("local", "--reference local"),
            ("antisites", "--antisites"),
        ],
    );
    // Global (Wigner-Seitz) and auto (here global) put 10 Frenkel pairs in
    // the interstitial loops; with --resolve-ambiguous, local: exact.
    check(
        &v,
        &["global", "global_ra", "local", "antisites"],
        &[
            ("global", "interstitial_loop", 10, 10),
            ("antisites", "interstitial_loop", 10, 10),
        ],
    );
    // No chemical order: no antisite count, and the reason on stderr.
    let a = &v["scores"]["antisites"];
    assert!(
        a["stdout"]
            .as_str()
            .is_some_and(|o| o.contains("defects_antisites: nan")),
        "{a}"
    );
    assert!(
        a["stderr"]
            .as_str()
            .is_some_and(|e| e.contains("no chemical order")),
        "{a}"
    );
}

/// HCP (Zr, c/a 1.593): c-loops b = c[0001] (basal) and a-loops
/// b = a/3<11-20> (prismatic), voids and point defects, in a periodic block.
/// Before, local lost 36 of 229 vacancies in loops and 5 of 294
/// interstitials.
#[test]
#[ignore = "expensive ground truth: run with --release --include-ignored"]
fn an_hcp_block_of_crowded_loops_is_counted_by_feature() {
    if !runnable() {
        return;
    }
    let v = score(
        "hcp",
        &[
            "--structure",
            "hcp",
            "--periodic",
            "--cells",
            "60",
            "--loops",
            "10",
            "--loop-radius",
            "2",
            "4",
            "--voids",
            "3",
            "--vacancies",
            "20",
            "--divacancies",
            "6",
            "--dumbbells",
            "10",
            "--close",
            "12",
            "--gap",
            "4",
            "--sigma",
            "0.045",
            "--seed",
            "3",
        ],
        &[
            ("global", "--reference global"),
            ("global_ra", "--reference global --resolve-ambiguous"),
            ("patch_ra", "--reference patch --resolve-ambiguous"),
            ("periodic", "--reference periodic"),
            ("local", "--reference local"),
            ("partial", "--reference local --local-merge partial"),
        ],
    );
    // With --resolve-ambiguous (or periodic), global's 30 Frenkel pairs are
    // gone.
    check(
        &v,
        &[
            "global",
            "global_ra",
            "patch_ra",
            "periodic",
            "local",
            "partial",
        ],
        &LIMITS_HCP,
    );
}

/// FCC: global exact; local and partial 5 vacancies over in one vacancy loop.
const LIMITS_FCC: [(&str, &str, i64, i64); 2] = [
    ("local", "vacancy_loop", 5, 0),
    ("partial", "vacancy_loop", 5, 0),
];
/// HCP: global adds 30 Frenkel pairs inside the interstitial loops (the
/// crystal displaced by nearly b/2); local and partial 2 vacancies short in
/// one vacancy loop.
const LIMITS_HCP: [(&str, &str, i64, i64); 3] = [
    ("global", "interstitial_loop", 30, 30),
    ("local", "vacancy_loop", 2, 0),
    ("partial", "vacancy_loop", 2, 0),
];

const KINDS: [&str; 6] = [
    "interstitial_loop",
    "vacancy_loop",
    "void",
    "vacancy",
    "divacancy",
    "dumbbell",
];

/// Large loops crowded together (R 38-54 A, 3 A apart, point defects and
/// voids 4-8 A from their rims; 3.5M atoms): their rims percolate into one
/// cluster across the box, counted atom by atom, and each loop's own region
/// has a shell its neighbours strain past one affine lattice (60-89 % of it
/// on the fitted lattice). Such a shell is labelled by walking its bonds and
/// counted on a smooth lattice fitted to the labels: the vacancy loops' error
/// fell from 610 to 455 (one loop -219 -> -64). Loops whose shells another
/// loop's rim crosses stay uncounted (design §16.8a). About 5 minutes in a
/// release build, most of it generating the block; far longer in debug, where
/// it skips.
#[test]
#[ignore = "expensive (3.5M atoms): run with --release --include-ignored"]
fn large_crowded_loops_are_counted_on_strained_shells() {
    if cfg!(debug_assertions) || !runnable() {
        eprintln!("skipping: a release build only");
        return;
    }
    let v = score(
        "large_crowded",
        &[
            "--cells",
            "120",
            "--loops",
            "8",
            "--loop-radius",
            "12",
            "17",
            "--loop-types",
            "111",
            "--voids",
            "4",
            "--void-radius",
            "1.5",
            "3",
            "--vacancies",
            "30",
            "--divacancies",
            "10",
            "--dumbbells",
            "20",
            "--close",
            "30",
            "--gap",
            "3",
            "--sigma",
            "0.045",
            "--seed",
            "3",
        ],
        &[("local", "--reference local")],
    );
    check(
        &v,
        &["local"],
        &[
            ("local", "interstitial_loop", 0, 120),
            ("local", "vacancy_loop", 450, 330),
        ],
    );
}

/// Massively overlapping cascades (research data, not distributed;
/// `RMDA_CASCADE_DIR`: one LAMMPS data file `data.lastframe-*` per metal,
/// potential and run, each a periodic single crystal of whole cells that kept
/// its atoms): `--reference periodic`
/// counts each against its one lattice, so V - I is exactly 0. Several hold
/// a box-spanning dislocation, round which loops round the box disagree by a
/// plane; the lattice that holds the atoms is the one they were cut from,
/// and the majority's is a plane off (V - I = one plane, thousands). A frame
/// no lattice through the loops conserves (its atoms are not one crystal's)
/// falls back to auto, saying so. The local census is no check here: it
/// misses defects stored as whole planes. Skipped without the directory.
#[test]
#[ignore = "cascade-data oracle: set RMDA_CASCADE_DIR, run with --include-ignored"]
fn cascade_frames_count_as_periodic_single_crystals() {
    let Some(dir) = std::env::var_os("RMDA_CASCADE_DIR") else {
        eprintln!("SKIPPED: RMDA_CASCADE_DIR not set");
        return;
    };
    let mut frames: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("data.lastframe-"))
        })
        .collect();
    frames.sort();
    assert!(!frames.is_empty(), "no data.lastframe-* in {dir:?}");
    let run = |f: &Path, reference: &str| -> (i64, i64, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
            .args(["defects", f.to_str().unwrap(), "--reference", reference])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let get = |k: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(&format!("{k}: ")))
                .and_then(|v| v.trim().parse::<i64>().ok())
                .unwrap_or(-1)
        };
        (
            get("defects_vacancies"),
            get("defects_interstitials"),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let mut fell_back = 0;
    for f in &frames {
        let (v, i, err) = run(f, "periodic");
        if err.contains("not a periodic single crystal") {
            assert!(
                err.contains("as many sites as atoms"),
                "{f:?}: fell back for another reason: {err}"
            );
            fell_back += 1;
            continue;
        }
        assert_eq!(
            v, i,
            "{f:?}: V - I must be 0 in a crystal that kept its sites"
        );
    }
    assert!(
        fell_back * 10 <= frames.len(),
        "{fell_back} of {} frames fell back to auto",
        frames.len()
    );
}
