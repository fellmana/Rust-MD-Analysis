//! `rmda ptm --outputs`: the opt-in ordering, deformation-gradient and template
//! columns (design §14.2). Values are checked against OVITO in
//! `rmda-core/tests/ptm_extras_ovito.rs`; this covers the CLI plumbing and that
//! the default run is untouched.

use std::path::PathBuf;
use std::process::Command;

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

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/ptm_extras")
        .join(format!("{name}.xyz"))
        .to_string_lossy()
        .into_owned()
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rmda_ptm_outputs_{tag}_{}.xyz", std::process::id()))
}

#[test]
fn outputs_add_columns_and_ordering_counts() {
    let b2 = fixture("b2_t10");
    let plain = tmp("plain");
    let extra = tmp("extra");
    let (out0, _, ok) = rmda(&[
        "-o",
        plain.to_str().unwrap(),
        "ptm",
        &b2,
        "--ptm-order",
        "voronoi",
    ]);
    assert!(ok);
    assert!(!out0.contains("ptm_ordering_"), "{out0}");
    let (out1, err1, ok) = rmda(&[
        "-o",
        extra.to_str().unwrap(),
        "ptm",
        &b2,
        "--ptm-order",
        "voronoi",
        "--outputs",
        "ordering,F,template",
    ]);
    assert!(ok, "{err1}");
    // The default report is a prefix of the extended one.
    for line in out0.lines() {
        assert!(out1.contains(line), "missing {line:?} in\n{out1}");
    }
    // OVITO on this fixture: 132 B2, 118 none (antisites break the shells).
    assert!(out1.contains("ptm_ordering_b2: 132"), "{out1}");
    assert!(out1.contains("ptm_ordering_none: 118"), "{out1}");
    let head0 = std::fs::read_to_string(&plain).unwrap();
    let head1 = std::fs::read_to_string(&extra).unwrap();
    let props = |t: &str| t.lines().nth(1).unwrap().to_string();
    assert!(!props(&head0).contains("StructureOrdering"));
    for col in [
        "StructureOrdering",
        "StructureDeformationGradient",
        "StructureTemplate",
    ] {
        assert!(
            props(&head1).contains(col),
            "{col} not in {}",
            props(&head1)
        );
    }
    let _ = std::fs::remove_file(plain);
    let _ = std::fs::remove_file(extra);
}

#[test]
fn unknown_output_is_rejected() {
    let (_, err, ok) = rmda(&["ptm", &fixture("b2_t10"), "--outputs", "strain"]);
    assert!(!ok);
    assert!(err.contains("unknown PTM output"), "{err}");
}
