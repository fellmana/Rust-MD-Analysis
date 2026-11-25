//! Live parity of the MIC dispatch against ASE's `find_mic` and an
//! independent brute-force oracle (design §6.3).
//!
//! `python3` with ASE installed is not a build dependency, so a missing
//! interpreter or module *skips* (with the reason on stderr) rather than
//! failing — the same contract `python_bindings.rs` uses. The Python side
//! (`mic_parity.py`) prints, per random case, the cell, `pbc`, the
//! displacement, ASE's squared length, and the squared length of the true
//! minimum found by bounded enumeration.
//!
//! The Rust side then asserts:
//!
//!   * `UnitCell::min_image_with_shift` matches the brute-force oracle on
//!     every case, and returns it as `raw + shift_vector(index)`;
//!   * where ASE itself is optimal, our answer matches ASE's too.
//!
//! ASE is *not* assumed correct: on partially periodic cells its
//! `general_find_mic` wraps only the periodic λ coordinates and can miss
//! the true image (see `UnitCell::general_min_image`), and its
//! `0.5 * min(lengths)` guard can over-accept when a row combination is
//! shorter than every row. Cases where ASE disagrees with the oracle are
//! counted and tolerated — but never silently: too few agreeable cases
//! would make the comparison vacuous, which the final assertions reject.

use rmda_core::cell::{Mat3, UnitCell};
use std::path::Path;
use std::process::Command;

fn python3() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn skip(why: &str) {
    eprintln!("SKIPPED mic parity: {why}");
}

#[test]
fn mic_matches_ase_where_ase_is_correct() {
    if !python3() {
        skip("`python3` is not on PATH");
        return;
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/mic_parity.py");
    let out = Command::new("python3")
        .arg(&script)
        .output()
        .expect("failed to run python3");
    if out.status.code() == Some(3) {
        skip("ASE or numpy is not importable");
        return;
    }
    assert!(
        out.status.success(),
        "mic_parity.py failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut checked = 0usize;
    let mut ase_agrees = 0usize;
    let mut ase_off = 0usize;
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(v.len(), 18, "expected 18 fields, got: {line}");
        let pbc = [v[1] == "1", v[2] == "1", v[3] == "1"];
        let nums: Vec<f64> = v[4..]
            .iter()
            .map(|s| {
                s.parse::<f64>()
                    .unwrap_or_else(|_| panic!("bad float {s} in {line}"))
            })
            .collect();
        let h = Mat3::from_rows(
            [nums[0], nums[1], nums[2]],
            [nums[3], nums[4], nums[5]],
            [nums[6], nums[7], nums[8]],
        );
        let dr = [nums[9], nums[10], nums[11]];
        let ase_len2 = nums[12];
        let brute_len2 = nums[13];

        let cell = UnitCell::from_matrix(h, pbc, [0.0; 3]);
        let (mic, index) = cell
            .min_image_with_shift(dr)
            .expect("the generated cells are nonsingular");
        let ours2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
        let tol = |a: f64, b: f64| (a - b).abs() <= 1e-8 * (1.0 + a.abs() + b.abs());
        assert!(
            tol(ours2, brute_len2),
            "case {}: min_image length² {ours2} != brute force {brute_len2}\n\
             pbc={pbc:?} h={:?} dr={dr:?}",
            v[0],
            h.m
        );
        let s = cell.shift_vector(index);
        for d in 0..3 {
            assert!(
                (mic[d] - (dr[d] + s[d])).abs() <= 1e-9 * (1.0 + dr[d].abs()),
                "case {}: mic {mic:?} != raw {dr:?} + shift {s:?} (index {index:?})",
                v[0]
            );
        }
        if tol(ase_len2, brute_len2) {
            assert!(
                tol(ours2, ase_len2),
                "case {}: ours {ours2} and ASE {ase_len2} both claim the \
                 minimum but differ\npbc={pbc:?} h={:?} dr={dr:?}",
                v[0],
                h.m
            );
            ase_agrees += 1;
        } else {
            ase_off += 1;
            eprintln!(
                "note: case {}: ASE is suboptimal (ase {ase_len2}, brute {brute_len2}); \
                 ours {ours2}",
                v[0]
            );
        }
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} usable cases");
    assert!(
        ase_agrees >= 30,
        "only {ase_agrees}/{checked} cases had ASE optimal — the comparison is vacuous"
    );
    eprintln!(
        "mic parity: {checked} cases, {ase_agrees} agreed with ASE, ASE suboptimal on {ase_off}"
    );
}
