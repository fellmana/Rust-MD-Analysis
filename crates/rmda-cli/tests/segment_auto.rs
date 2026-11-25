//! `rmda segment --mode auto` end to end: OVITO's graph clustering with the
//! automatic threshold, PTM in OVITO's neighbour order, the threshold on
//! stdout. The HEA parity itself is pinned in `rmda-core`'s `grain_seg_ovito`.

use rmda_core::cell::UnitCell;
use rmda_core::data::frame::Frame;
use rmda_core::data::particle::ParticleData;
use rmda_core::io::XyzWriter;
use rmda_core::lattice::fcc::Fcc;
use rmda_core::lattice::sites::SiteLattice;
use rmda_core::lattice::Lattice;
use std::path::PathBuf;
use std::process::Command;

const A: f64 = 3.615;

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
        .unwrap()
}

/// A free-standing FCC bicrystal: the right half turned `tilt`° about z,
/// every atom jittered by up to ±0.1 Å.
fn bicrystal(tilt: f64) -> PathBuf {
    let len = 10.0 * A;
    let big = UnitCell::orthorhombic(2.0 * len, 2.0 * len, len);
    let sites: Vec<[f64; 3]> = SiteLattice::build(&big, &Fcc { a: A }.primitive(), [0.0; 3])
        .unwrap()
        .iter()
        .collect();
    let (s, c) = tilt.to_radians().sin_cos();
    let mut pos = Vec::new();
    for p in &sites {
        // Left grain: the lattice as built.
        if p[0] < len && p[1] < len {
            pos.push(*p);
        }
        // Right grain: the lattice turned about the box's centre line.
        let (x, y) = (p[0] - len, p[1] - len);
        let q = [len + c * x - s * y, len + s * x + c * y, p[2]];
        if q[0] >= len + 0.5 && q[0] < 2.0 * len && q[1] >= 0.0 && q[1] < len {
            pos.push(q);
        }
    }
    let mut seed = 0x2545f4914f6cdd1du64;
    for p in pos.iter_mut() {
        for v in p.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v += 0.2 * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) + 5.0;
        }
    }
    let cell =
        UnitCell::orthorhombic(2.0 * len + 10.0, len + 10.0, len + 10.0).with_pbc([false; 3]);
    let mut pd = ParticleData::with_capacity(pos.len());
    for (i, p) in pos.iter().enumerate() {
        pd.add_atom(*p, i as i64 + 1, 1);
    }
    let dir = std::env::temp_dir().join("rmda-cli-segment-auto-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("bicrystal_{tilt}.xyz"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    XyzWriter::write_frame(&mut f, &Frame::new(cell, pd)).unwrap();
    path
}

#[test]
fn auto_mode_splits_a_bicrystal_and_reports_its_threshold() {
    let path = bicrystal(20.0);
    let p = path.to_str().unwrap();
    for extra in [&[][..], &["--ptm-order", "distance"][..]] {
        let mut args = vec!["segment", p, "--mode", "auto"];
        args.extend_from_slice(extra);
        let (out, err, ok) = rmda(&args);
        assert!(ok, "{err}");
        assert_eq!(scalar(&out, "segment_grains"), 2.0, "{out}\n{err}");
        assert_eq!(scalar(&out, "segment_atoms_unassigned"), 0.0);
        assert!(scalar(&out, "segment_merge_threshold").is_finite());
    }
    // The default mode prints no threshold and finds the same two grains.
    let (out, err, ok) = rmda(&["segment", p]);
    assert!(ok, "{err}");
    assert_eq!(scalar(&out, "segment_grains"), 2.0, "{out}");
    assert!(!out.contains("segment_merge_threshold"));
}
