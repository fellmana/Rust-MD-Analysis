//! Double HCP (`--dhcp`): Nd, and the stackings that must not be called dHCP.

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

fn scalar(text: &str, name: &str) -> i64 {
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{name}: ")))
        .unwrap_or_else(|| panic!("no {name} in:\n{text}"))
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_dhcp_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Close-packed layers in the order of `seq` (letters A, B, C), each a
/// `nx`×`ny` orthohexagonal patch of in-plane spacing `a`, `d` apart, in a
/// periodic box, with atoms `skip` (layer, cell x, cell y, which of the
/// cell's two) removed, `extra` added, and seeded Gaussian noise of `sigma`.
#[allow(clippy::too_many_arguments)]
fn stack(
    seq: &str,
    nx: usize,
    ny: usize,
    a: f64,
    d: f64,
    sigma: f64,
    skip: &[(usize, usize, usize, usize)],
    extra: &[[f64; 3]],
) -> String {
    let s3 = 3f64.sqrt();
    let mut seed: u64 = 2024;
    let mut uniform = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = || {
        let (u, v) = (uniform(), uniform());
        sigma * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut pts = Vec::new();
    for (k, l) in seq.chars().enumerate() {
        let off: [(f64, f64); 2] = match l {
            'A' => [(0.0, 0.0), (0.5, 0.5)],
            'B' => [(0.0, 1.0 / 3.0), (0.5, 5.0 / 6.0)],
            _ => [(0.0, 2.0 / 3.0), (0.5, 1.0 / 6.0)],
        };
        for i in 0..nx {
            for j in 0..ny {
                for (m, (u, v)) in off.iter().enumerate() {
                    if !skip.contains(&(k, i, j, m)) {
                        pts.push([(i as f64 + u) * a, (j as f64 + v) * a * s3, k as f64 * d]);
                    }
                }
            }
        }
    }
    pts.extend_from_slice(extra);
    let l = [nx as f64 * a, ny as f64 * a * s3, seq.len() as f64 * d];
    let rows: Vec<String> = pts
        .iter()
        .map(|p| {
            let q: Vec<f64> = (0..3).map(|x| (p[x] + gauss()).rem_euclid(l[x])).collect();
            format!("Nd {} {} {}", q[0], q[1], q[2])
        })
        .collect();
    format!(
        "{}\nLattice=\"{} 0 0 0 {} 0 0 0 {}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        l[0],
        l[1],
        l[2],
        rows.join("\n")
    )
}

/// Nd: a = 3.658 Å, c = 11.799 Å over four layers (c/a 3.226, below the
/// ideal 3.266).
const ND_A: f64 = 3.658;
const ND_D: f64 = 11.799 / 4.0;

#[test]
fn nd_is_one_dhcp_grain_and_its_defects_are_counted_exactly() {
    let dir = tempdir("nd");
    // Three vacancies in different layers and one atom in an octahedral
    // hole between an A layer and the B layer above it.
    let skip = [(10, 2, 2, 0), (5, 5, 3, 1), (14, 4, 5, 0)];
    let oct = [
        3.0 * ND_A,
        (3.0 + 2.0 / 3.0) * ND_A * 3f64.sqrt(),
        8.5 * ND_D,
    ];
    let text = stack(&"ABAC".repeat(5), 10, 6, ND_A, ND_D, 0.05, &skip, &[oct]);
    let input = dir.join("nd.xyz");
    std::fs::write(&input, text).unwrap();
    let file = input.to_str().unwrap();

    let (out, err, ok) = rmda(&["segment", file, "--dhcp"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "segment_grains"), 1, "{out}");
    assert!(err.contains("structure=32"), "{err}");
    for reference in ["global", "patch", "local"] {
        let (out, err, ok) = rmda(&[
            "defects",
            file,
            "--segment",
            "--dhcp",
            "--reference",
            reference,
        ]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(
            scalar(&out, "defects_grains_measured"),
            1,
            "{reference}\n{err}"
        );
        assert_eq!(scalar(&out, "defects_vacancies"), 3, "{reference}\n{err}");
        assert_eq!(
            scalar(&out, "defects_interstitials"),
            1,
            "{reference}\n{err}"
        );
        assert!(err.contains("structure=32"), "{reference}\n{err}");
    }
    // Without the flag stacking-fault merging joins the alternating FCC and
    // HCP layers into one grain, as OVITO's segmentation does, but not a
    // dHCP one: its defects cannot be counted against one lattice.
    let (out, err, ok) = rmda(&["segment", file]);
    assert!(ok);
    assert_eq!(scalar(&out, "segment_grains"), 1, "{out}");
    assert!(!err.contains("structure=32"), "{err}");
}

#[test]
fn fcc_hcp_twins_and_faults_are_not_dhcp() {
    let dir = tempdir("notdhcp");
    // Cu-like spacing; dense random faults (30 % of layers) included.
    let (a, d) = (2.556, 2.556 * (2.0f64 / 3.0).sqrt());
    for (name, seq) in [
        ("fcc", "ABC".repeat(12)),
        ("hcp", "AB".repeat(18)),
        ("twin", "ABC".repeat(6) + &"ACB".repeat(6)),
        ("intrinsic", "ABC".repeat(6) + "AB" + &"ABC".repeat(5) + "A"),
        ("faults", "ABCABCABABABCBCACABCACBCBCABCB".to_owned()),
    ] {
        let input = dir.join(format!("{name}.xyz"));
        std::fs::write(&input, stack(&seq, 10, 6, a, d, 0.05, &[], &[])).unwrap();
        let (out, err, ok) = rmda(&["segment", input.to_str().unwrap(), "--dhcp"]);
        assert!(ok, "{out}\n{err}");
        assert!(err.contains("dhcp: 0 close-packed atoms"), "{name}: {err}");
    }
}

#[test]
fn an_fcc_dhcp_interface_is_two_grains() {
    let dir = tempdir("iface");
    let (a, d) = (2.556, 2.556 * (2.0f64 / 3.0).sqrt());
    let seq = "ABC".repeat(6) + &"ABAC".repeat(5);
    let input = dir.join("iface.xyz");
    std::fs::write(&input, stack(&seq, 10, 6, a, d, 0.05, &[], &[])).unwrap();
    let (out, err, ok) = rmda(&["segment", input.to_str().unwrap(), "--dhcp"]);
    assert!(ok, "{out}\n{err}");
    assert_eq!(scalar(&out, "segment_grains"), 2, "{out}\n{err}");
    // 19 layers each: the h layer where the stackings meet (between two c
    // layers) alternates with the ABAC run, so it is dHCP.
    assert!(err.contains("size=2280 structure=32"), "{err}");
    assert!(err.contains("size=2280 structure=1"), "{err}");
}

#[test]
fn dhcp_faults_join_their_grain_only_when_asked_and_never_another_phase() {
    let dir = tempdir("faults");
    let grains = |seq: &str, flags: &[&str]| -> (i64, String) {
        let input = dir.join("s.xyz");
        std::fs::write(&input, stack(seq, 10, 6, ND_A, ND_D, 0.05, &[], &[])).unwrap();
        let mut args = vec!["segment", input.to_str().unwrap(), "--dhcp"];
        args.extend_from_slice(flags);
        let (out, err, ok) = rmda(&args);
        assert!(ok, "{out}\n{err}");
        (scalar(&out, "segment_grains"), err)
    };
    // One intrinsic-like (hcp) and one cubic fault in dHCP.
    let hcp_fault = "ABAC".repeat(4) + "AB" + &"ABAC".repeat(4) + "AC";
    let cubic_fault = "ABAC".repeat(4) + "ABCB" + &"ABAC".repeat(4);
    for seq in [&hcp_fault, &cubic_fault] {
        assert!(
            grains(seq, &[]).0 > 1,
            "{seq}: faults split dHCP by default"
        );
        let (n, err) = grains(seq, &["--dhcp-merge-faults"]);
        assert_eq!(n, 1, "{seq}\n{err}");
        assert!(err.contains("size=4320 structure=32"), "{err}");
        assert_eq!(
            grains(
                seq,
                &["--dhcp-merge-faults", "--stacking-faults", "separate"]
            )
            .0,
            grains(seq, &["--stacking-faults", "separate"]).0
        );
    }
    // Another phase is not a fault: FCC on dHCP, and seven FCC layers
    // between dHCP, keep their own grains, and no FCC atom joins dHCP.
    let (n, err) = grains(
        &("ABC".repeat(6) + &"ABAC".repeat(5)),
        &["--dhcp-merge-faults"],
    );
    assert_eq!(n, 2, "{err}");
    let sandwich = "ABAC".repeat(4) + "BCABCAB" + "CABAC" + &"ABAC".repeat(3);
    let (_, err) = grains(&sandwich, &["--dhcp-merge-faults"]);
    let (_, plain) = grains(&sandwich, &[]);
    let dhcp_grain = |e: &str| {
        e.lines()
            .find(|l| l.contains("structure=32"))
            .and_then(|l| l.split("size=").nth(1))
            .and_then(|v| v.split_whitespace().next())
            .map(str::to_owned)
    };
    assert_eq!(dhcp_grain(&err), dhcp_grain(&plain), "{err}");
    // The defect census keeps the split grains, which count exactly.
    let input = dir.join("v.xyz");
    let skip = [(1, 2, 3, 0), (3, 4, 1, 1), (6, 7, 2, 0)];
    std::fs::write(
        &input,
        stack(&hcp_fault, 10, 6, ND_A, ND_D, 0.05, &skip, &[]),
    )
    .unwrap();
    let (out, err, ok) = rmda(&[
        "defects",
        input.to_str().unwrap(),
        "--segment",
        "--dhcp",
        "--reference",
        "local",
    ]);
    assert!(ok, "{out}\n{err}");
    let total = |k: &str| {
        scalar(&out, &format!("defects_{k}")) + scalar(&out, &format!("defects_boundary_{k}"))
    };
    assert_eq!(
        (total("vacancies"), total("interstitials")),
        (3, 0),
        "{err}"
    );
}
