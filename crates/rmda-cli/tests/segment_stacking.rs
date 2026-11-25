//! FCC/HCP merging under stacking-fault merging: decided in the frame of the
//! larger cluster, so twins stay apart, faults join, and the order of atoms
//! in the file does not matter (design §15.2).

use std::process::Command;

fn segment(path: &std::path::Path) -> (Vec<i64>, String) {
    let out_xyz = path.with_extension("seg.xyz");
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .args([
            "segment",
            path.to_str().unwrap(),
            "-o",
            out_xyz.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&out_xyz).unwrap();
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap().trim().parse().unwrap();
    let header = lines.next().unwrap();
    let props: Vec<&str> = header
        .split("Properties=")
        .nth(1)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .split(':')
        .collect();
    let (mut col, mut at) = (None, 0);
    for p in props.chunks(3) {
        if p[0] == "Grain" {
            col = Some(at);
        }
        at += p[2].parse::<usize>().unwrap();
    }
    let col = col.unwrap();
    let grain = lines
        .take(n)
        .map(|l| l.split_whitespace().nth(col).unwrap().parse().unwrap())
        .collect();
    (grain, String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Close-packed Cu layers in the order of `steps` (+1 the FCC sense, −1 the
/// other), each 12×7 orthohexagonal cells, with seeded noise; rows written
/// in the order `perm` gives (identity when empty).
fn stack(steps: &[i32], perm_seed: Option<u64>) -> String {
    assert_eq!(steps.iter().sum::<i32>().rem_euclid(3), 0);
    let (a, nx, ny) = (2.556f64, 12usize, 7usize);
    let d = a * (2.0f64 / 3.0).sqrt();
    let s3 = 3f64.sqrt();
    let mut layer = 0i32;
    let mut pts = Vec::new();
    for (k, st) in steps.iter().enumerate() {
        let off: [(f64, f64); 2] = match layer.rem_euclid(3) {
            0 => [(0.0, 0.0), (0.5, 0.5)],
            1 => [(0.0, 1.0 / 3.0), (0.5, 5.0 / 6.0)],
            _ => [(0.0, 2.0 / 3.0), (0.5, 1.0 / 6.0)],
        };
        for i in 0..nx {
            for j in 0..ny {
                for (u, v) in off {
                    pts.push([(i as f64 + u) * a, (j as f64 + v) * a * s3, k as f64 * d]);
                }
            }
        }
        layer += st;
    }
    let mut seed: u64 = 11;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let mut gauss = || {
        let u = ((next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let v = ((next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        0.06 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let l = [nx as f64 * a, ny as f64 * a * s3, steps.len() as f64 * d];
    let mut rows: Vec<String> = pts
        .iter()
        .map(|p| {
            let q: Vec<f64> = (0..3).map(|x| (p[x] + gauss()).rem_euclid(l[x])).collect();
            format!("Cu {} {} {}", q[0], q[1], q[2])
        })
        .collect();
    if let Some(mut s) = perm_seed {
        for i in (1..rows.len()).rev() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rows.swap(i, (s >> 33) as usize % (i + 1));
        }
    }
    format!(
        "{}\nLattice=\"{} 0 0 0 {} 0 0 0 {}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        l[0],
        l[1],
        l[2],
        rows.join("\n")
    )
}

fn sizes(g: &[i64]) -> Vec<usize> {
    let mut m = std::collections::BTreeMap::new();
    for &x in g {
        *m.entry(x).or_insert(0usize) += 1;
    }
    let mut v: Vec<usize> = m.into_values().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v
}

#[test]
fn twins_stay_apart_faults_join_and_atom_order_does_not_matter() {
    let dir = std::env::temp_dir().join(format!("rmda_segstack_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = |n: usize| vec![1; n];
    let r = |n: usize| vec![-1; n];
    let h = |n: usize| [1, -1].repeat(n / 2);
    let cases: Vec<(&str, Vec<i32>, usize)> = vec![
        // Two twins: the twin plane (one HCP layer) joins one of them, also
        // when it comes first in the file.
        ("twins", [f(24), r(24)].concat(), 2),
        // An HCP lamella inside FCC joins it (OVITO: one grain).
        ("lamella", [f(24), h(10), f(15)].concat(), 1),
        // A lamella between two twins: two grains, as OVITO.
        ("lamella_twin", [f(18), h(8), r(18)].concat(), 2),
    ];
    for (name, steps, grains) in cases {
        let mut seen = None;
        for perm in [None, Some(7u64), Some(99)] {
            let p = dir.join(format!("{name}_{}.xyz", perm.unwrap_or(0)));
            std::fs::write(&p, stack(&steps, perm)).unwrap();
            let (g, out) = segment(&p);
            let s = sizes(&g);
            assert_eq!(s.len(), grains, "{name} {perm:?}: {s:?}\n{out}");
            // Same partition whatever the order (grain sizes, multiset).
            match &seen {
                None => seen = Some(s),
                Some(first) => assert_eq!(first, &s, "{name}: order {perm:?} differs"),
            }
        }
    }
}
