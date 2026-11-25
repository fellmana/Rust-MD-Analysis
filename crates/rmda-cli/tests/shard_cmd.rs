//! Sharded series (`rmda defects --out DIR --shard I/N`, then `rmda merge DIR`)
//! and resumed series must write exactly what one uninterrupted run writes,
//! grain tracking included (design §20).

use std::path::{Path, PathBuf};
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

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_shard_cmd_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A periodic BCC W block of `n`×`n`×`nz` cells (atom ids in lattice order),
/// with each column (centre, radius Å, angle °) of the lattice, through the
/// thin z axis, rotated about its axis: a misoriented grain whose atoms keep
/// their ids. Columns rather than spheres keep the frame small (1.6k atoms),
/// so the many debug-build runs of these tests stay quick. Atoms
/// closer than 1.4 Å to a lower-id atom are dropped.
fn frame(n: usize, nz: usize, spheres: &[([f64; 3], f64, f64)]) -> String {
    let a = 3.165;
    let l = n as f64 * a;
    let lz = nz as f64 * a;
    let len = [l, l, lz];
    let mut atoms: Vec<(usize, [f64; 3])> = Vec::new();
    let mut id = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..nz {
                for off in [0.0, 0.5] {
                    id += 1;
                    let mut p = [
                        (i as f64 + off) * a,
                        (j as f64 + off) * a,
                        (k as f64 + off) * a,
                    ];
                    for &(c, r, deg) in spheres {
                        // A column through the (thin, periodic) z axis.
                        let d = [p[0] - c[0], p[1] - c[1], 0.0];
                        if (d[0] * d[0] + d[1] * d[1]).sqrt() < r {
                            let (s, co) = deg.to_radians().sin_cos();
                            p = [
                                c[0] + co * d[0] - s * d[1],
                                c[1] + s * d[0] + co * d[1],
                                p[2],
                            ];
                        }
                    }
                    atoms.push((id, p));
                }
            }
        }
    }
    let mut keep: Vec<(usize, [f64; 3])> = Vec::new();
    for (id, p) in atoms {
        let close = keep.iter().any(|(_, q)| {
            let mut d2 = 0.0;
            for a in 0..3 {
                let mut x = p[a] - q[a];
                x -= len[a] * (x / len[a]).round();
                d2 += x * x;
            }
            d2 < 1.4 * 1.4
        });
        if !close {
            keep.push((id, p));
        }
    }
    let rows: Vec<String> = keep
        .iter()
        .map(|(id, p)| format!("W {:.6} {:.6} {:.6} {id}", p[0], p[1], p[2]))
        .collect();
    format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {lz}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:id:I:1\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

/// Five frames in `dir`: sphere A appears, B appears, A grows, A rotates back
/// into the matrix (merged), only B left (label 1, stable id 2).
fn series(dir: &Path) -> Vec<String> {
    let ca = [12.0, 12.0, 22.0];
    let cb = [32.0, 32.0, 22.0];
    let frames: [Vec<([f64; 3], f64, f64)>; 5] = [
        vec![(ca, 11.0, 25.0)],
        vec![(ca, 11.0, 25.0), (cb, 11.0, -20.0)],
        vec![(ca, 12.0, 25.0), (cb, 11.0, -20.0)],
        vec![(cb, 11.0, -20.0)],
        vec![(cb, 11.5, -20.0)],
    ];
    frames
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let p = dir.join(format!("frame.{}.xyz", (i + 1) * 100));
            std::fs::write(&p, frame(14, 4, s)).unwrap();
            p.to_string_lossy().into_owned()
        })
        .collect()
}

/// A series' outputs as compared: every table except the timing and memory
/// columns of summary.csv, and every defect file.
fn outputs(dir: &Path) -> Vec<(String, String)> {
    let mut v = Vec::new();
    for t in [
        "summary.csv",
        "grains.csv",
        "clusters.csv",
        "grain_events.csv",
    ] {
        let text =
            std::fs::read_to_string(dir.join(t)).unwrap_or_else(|_| panic!("{t} in {dir:?}"));
        let text = if t == "summary.csv" {
            let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
            let keep: Vec<usize> = (0..header.len())
                .filter(|&i| !["wall_s", "rss_mb", "peak_rss_mb"].contains(&header[i]))
                .collect();
            text.lines()
                .map(|l| {
                    let f: Vec<&str> = l.split(',').collect();
                    keep.iter().map(|&i| f[i]).collect::<Vec<_>>().join(",")
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            text
        };
        v.push((t.to_owned(), text));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![dir.join("defects")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            } else {
                files.push(e.path());
            }
        }
    }
    files.sort();
    for f in files {
        let rel = f.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
        v.push((rel, std::fs::read_to_string(&f).unwrap()));
    }
    v
}

fn run(files: &[String], extra: &[&str]) {
    let mut args = vec!["defects", "--segment", "--reference", "global"];
    args.extend(files.iter().map(String::as_str));
    args.extend_from_slice(extra);
    let (out, err, ok) = rmda(&args);
    assert!(ok, "{args:?}\n{out}\n{err}");
}

#[test]
fn sharded_and_merged_series_equal_the_sequential_one() {
    let dir = tempdir("merge");
    let files = series(&dir);
    let seq = dir.join("seq");
    run(
        &files,
        &[
            "--out",
            seq.to_str().unwrap(),
            "--defects-xyz",
            "--threads",
            "2",
        ],
    );
    let want = outputs(&seq);
    // The fixture exercises tracking: a new grain, a split from the matrix
    // and a merge back into it.
    let events = &want
        .iter()
        .find(|(t, _)| t == "grain_events.csv")
        .unwrap()
        .1;
    for kind in [",new,", ",continued,", ",split,", ",merged,"] {
        assert!(events.contains(kind), "no {kind} event:\n{events}");
    }
    // And a grain whose stable id is not its label (B outlives A, so in the
    // last frames it is label 1 with id 2): what renumbering at merge, and
    // seeding on resume, must get right.
    let grains = &want.iter().find(|(t, _)| t == "grains.csv").unwrap().1;
    assert!(
        grains.lines().skip(1).any(|l| {
            let f: Vec<&str> = l.split(',').collect();
            f[3] != f[4]
        }),
        "no grain with stable id != label:\n{grains}"
    );

    for (n, layout) in [(2usize, "dirs"), (3, "append")] {
        let sh = dir.join(format!("sharded-{n}"));
        let seq_l = dir.join(format!("seq-{layout}"));
        run(
            &files,
            &[
                "--out",
                seq_l.to_str().unwrap(),
                "--defects-layout",
                layout,
                "--threads",
                "2",
            ],
        );
        for i in 0..n {
            // Each shard on its own thread count: results must not depend on it.
            let threads = (i + 1).to_string();
            run(
                &files,
                &[
                    "--out",
                    sh.to_str().unwrap(),
                    "--shard",
                    &format!("{i}/{n}"),
                    "--defects-layout",
                    layout,
                    "--threads",
                    &threads,
                ],
            );
        }
        let (out, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
        assert!(ok, "{out}\n{err}");
        let got = outputs(&sh);
        let want = outputs(&seq_l);
        assert_eq!(got.len(), want.len(), "{n} shards: files differ");
        for ((tg, g), (tw, w)) in got.iter().zip(&want) {
            assert_eq!(tg, tw);
            assert_eq!(g, w, "{n} shards ({layout}): {tg} differs");
        }
        // Merging twice needs --overwrite.
        let (_, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
        assert!(!ok && err.contains("--overwrite"), "{err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_refuses_missing_and_unfinished_shards() {
    let dir = tempdir("refuse");
    let files = series(&dir);
    let sh = dir.join("sh");
    run(&files, &["--out", sh.to_str().unwrap(), "--shard", "0/3"]);
    let (_, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
    assert!(
        !ok && err.contains("shard(s) [1, 2] of 3 have not run"),
        "{err}"
    );
    run(&files, &["--out", sh.to_str().unwrap(), "--shard", "1/3"]);
    run(&files, &["--out", sh.to_str().unwrap(), "--shard", "2/3"]);
    // Shard 2 interrupted before its last frame's summary row.
    let summary = sh.join("shard-2-of-3/summary.csv");
    let text = std::fs::read_to_string(&summary).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    lines.pop();
    std::fs::write(&summary, lines.join("\n") + "\n").unwrap();
    let (_, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
    assert!(!ok && err.contains("is not finished"), "{err}");
    // Rerunning the shard resumes it; then the merge goes through.
    run(&files, &["--out", sh.to_str().unwrap(), "--shard", "2/3"]);
    let (out, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
    assert!(ok, "{out}\n{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_resumed_series_equals_an_uninterrupted_one() {
    let dir = tempdir("resume");
    let files = series(&dir);
    let whole = dir.join("whole");
    run(&files, &["--out", whole.to_str().unwrap(), "--defects-xyz"]);
    // Four frames, then rows of an interrupted fifth (no summary row), then
    // the whole series again: it resumes. After frame 400 grain B is label 1
    // with stable id 2; replaying that frame through a fresh tracker, which
    // numbers from 0, would write B as 1 in frame 500.
    let part = dir.join("part");
    run(
        &files[..4],
        &["--out", part.to_str().unwrap(), "--defects-xyz"],
    );
    for t in ["grains.csv", "clusters.csv", "grain_events.csv"] {
        let p = part.join(t);
        let mut text = std::fs::read_to_string(&p).unwrap();
        text.push_str("500,frame.500.xyz,0,interrupted,row\n");
        std::fs::write(&p, text).unwrap();
    }
    run(&files, &["--out", part.to_str().unwrap(), "--defects-xyz"]);
    let (got, want) = (outputs(&part), outputs(&whole));
    assert_eq!(got.len(), want.len());
    for ((t, g), (_, w)) in got.iter().zip(&want) {
        assert_eq!(g, w, "{t} differs after resuming");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A periodic BCC W block of `n`³ cells with the atoms within `r` Å of its
/// centre removed (a void), as extended XYZ with ids.
fn void_frame(n: usize, r: f64) -> String {
    let a = 3.165;
    let l = n as f64 * a;
    let c = l / 2.0 + 0.3;
    let mut rows = Vec::new();
    let mut id = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for off in [0.0, 0.5] {
                    id += 1;
                    let p = [
                        (i as f64 + off) * a,
                        (j as f64 + off) * a,
                        (k as f64 + off) * a,
                    ];
                    let d2: f64 = p.iter().map(|x| (x - c) * (x - c)).sum();
                    if d2 < r * r {
                        continue;
                    }
                    rows.push(format!("W {:.6} {:.6} {:.6} {id}", p[0], p[1], p[2]));
                }
            }
        }
    }
    format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {l}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:id:I:1\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

#[test]
fn a_sharded_void_series_merges_to_the_sequential_one() {
    let dir = tempdir("void");
    let files: Vec<String> = [0.0, 4.0, 5.0, 6.0]
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let p = dir.join(format!("frame.{}.xyz", (i + 1) * 10));
            std::fs::write(&p, void_frame(10, r)).unwrap();
            p.to_string_lossy().into_owned()
        })
        .collect();
    let void = |extra: &[&str]| {
        let mut args = vec!["void"];
        args.extend(files.iter().map(String::as_str));
        args.extend_from_slice(extra);
        let (out, err, ok) = rmda(&args);
        assert!(ok, "{args:?}\n{out}\n{err}");
    };
    let seq = dir.join("seq");
    void(&["--out", seq.to_str().unwrap()]);
    let table = |d: &Path, t: &str| -> String {
        let text = std::fs::read_to_string(d.join(t)).unwrap();
        let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        let keep: Vec<usize> = (0..header.len())
            .filter(|&i| header[i] != "wall_s")
            .collect();
        text.lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                keep.iter().map(|&i| f[i]).collect::<Vec<_>>().join(",")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let voids = table(&seq, "voids.csv");
    assert!(
        voids.lines().count() > 3,
        "the growing void is found:\n{voids}"
    );
    for n in [2usize, 3] {
        let sh = dir.join(format!("sharded-{n}"));
        for i in 0..n {
            void(&[
                "--out",
                sh.to_str().unwrap(),
                "--shard",
                &format!("{i}/{n}"),
            ]);
        }
        let (out, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
        assert!(ok, "{out}\n{err}");
        for t in ["summary.csv", "voids.csv"] {
            assert_eq!(table(&sh, t), table(&seq, t), "{n} shards: {t}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A void series with `--segment` tracks its grains as `rmda defects` does:
/// the same stable ids and events on the tracking fixture (a new grain, a
/// split, a merge, an id that is not its label), sharded and merged equal to
/// sequential (tables and `sites/` files), and resumed equal to whole.
#[test]
fn a_tracked_void_series_keeps_the_defects_grain_ids_and_merges() {
    let dir = tempdir("void_tracked");
    let files = series(&dir);
    let def = dir.join("def");
    run(&files, &["--out", def.to_str().unwrap(), "--threads", "2"]);
    let void = |extra: &[&str]| {
        let mut args = vec!["void", "--segment", "--site-reference", "global"];
        args.extend(files.iter().map(String::as_str));
        args.extend_from_slice(extra);
        let (out, err, ok) = rmda(&args);
        assert!(ok, "{args:?}\n{out}\n{err}");
    };
    let seq = dir.join("seq");
    void(&[
        "--out",
        seq.to_str().unwrap(),
        "--sites-xyz",
        "--threads",
        "2",
    ]);
    // (file_index, grain) -> stable_grain, from a grains.csv.
    let ids = |d: &Path| -> Vec<(String, String, String)> {
        let text = std::fs::read_to_string(d.join("grains.csv")).unwrap();
        let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        let c = |n: &str| header.iter().position(|&h| h == n).unwrap();
        let (k, g, s) = (c("file_index"), c("grain"), c("stable_grain"));
        text.lines()
            .skip(1)
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                (f[k].to_owned(), f[g].to_owned(), f[s].to_owned())
            })
            .collect()
    };
    let got = ids(&seq);
    assert_eq!(
        got,
        ids(&def),
        "void and defects series disagree on grain ids"
    );
    assert!(got.iter().any(|(_, g, s)| g != s), "{got:?}");
    let events = |d: &Path| std::fs::read_to_string(d.join("grain_events.csv")).unwrap();
    assert_eq!(events(&seq), events(&def));
    for kind in [",new,", ",split,", ",merged,"] {
        assert!(events(&seq).contains(kind), "no {kind}:\n{}", events(&seq));
    }
    let outputs = |d: &Path| -> Vec<(String, String)> {
        let mut v = Vec::new();
        for t in ["summary.csv", "voids.csv", "grains.csv", "grain_events.csv"] {
            let text = std::fs::read_to_string(d.join(t)).unwrap();
            let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
            let keep: Vec<usize> = (0..header.len())
                .filter(|&i| header[i] != "wall_s")
                .collect();
            let text = text
                .lines()
                .map(|l| {
                    let f: Vec<&str> = l.split(',').collect();
                    keep.iter().map(|&i| f[i]).collect::<Vec<_>>().join(",")
                })
                .collect::<Vec<_>>()
                .join("\n");
            v.push((t.to_owned(), text));
        }
        let mut sites: Vec<PathBuf> = std::fs::read_dir(d.join("sites"))
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        sites.sort();
        for p in sites {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            v.push((name, std::fs::read_to_string(&p).unwrap()));
        }
        v
    };
    let want = outputs(&seq);
    assert_eq!(want.len(), 4 + files.len(), "one sites file per frame");
    for n in [2usize, 3] {
        let sh = dir.join(format!("sharded-{n}"));
        for i in 0..n {
            let threads = (i + 1).to_string();
            void(&[
                "--out",
                sh.to_str().unwrap(),
                "--sites-xyz",
                "--shard",
                &format!("{i}/{n}"),
                "--threads",
                &threads,
            ]);
        }
        let (out, err, ok) = rmda(&["merge", sh.to_str().unwrap()]);
        assert!(ok, "{out}\n{err}");
        assert_eq!(outputs(&sh), want, "{n} shards");
    }
    // Resume after the third frame: the tracker is seeded from it.
    let summary = std::fs::read_to_string(seq.join("summary.csv")).unwrap();
    let cut: String = summary
        .lines()
        .filter(|l| !l.starts_with("400,") && !l.starts_with("500,"))
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(seq.join("summary.csv"), cut).unwrap();
    void(&[
        "--out",
        seq.to_str().unwrap(),
        "--sites-xyz",
        "--threads",
        "2",
    ]);
    assert_eq!(outputs(&seq), want, "resumed");
    let _ = std::fs::remove_dir_all(&dir);
}
