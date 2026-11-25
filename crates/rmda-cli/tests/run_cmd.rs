//! `rmda run CONFIG.toml` (design §20): a config is a command line written
//! down. The run must equal the same command given directly, write a run
//! report, and refuse a config the CLI would not take with a message that
//! names the key.

use std::path::{Path, PathBuf};
use std::process::Command;

fn rmda_in(dir: &Path, args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .current_dir(dir)
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
    let d = std::env::temp_dir().join(format!("rmda_run_cmd_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A BCC W block (periodic in x and y, vacuum above) with the sites in
/// `skip` left empty, as extended XYZ.
fn slab(n: usize, nz: usize, skip: &[(usize, usize, usize, usize)]) -> String {
    let a = 3.165;
    let mut rows = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..nz {
                for (b, off) in [0.0, 0.5].iter().enumerate() {
                    if skip.contains(&(i, j, k, b)) {
                        continue;
                    }
                    rows.push(format!(
                        "W {} {} {}",
                        (i as f64 + off) * a,
                        (j as f64 + off) * a,
                        (k as f64 + off) * a + 5.0
                    ));
                }
            }
        }
    }
    let l = n as f64 * a;
    let lz = 3.0 * nz as f64 * a;
    format!(
        "{}\nLattice=\"{l} 0 0 0 {l} 0 0 0 {lz}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n{}\n",
        rows.len(),
        rows.join("\n")
    )
}

/// Two frames named for the series order (`frame.<index>.xyz`), with 1 and
/// 3 vacancies, in `dir/data`.
fn frames(dir: &Path) {
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(data.join("frame.10.xyz"), slab(8, 6, &[(4, 4, 3, 0)])).unwrap();
    std::fs::write(
        data.join("frame.20.xyz"),
        slab(8, 6, &[(4, 4, 3, 0), (2, 5, 2, 1), (6, 1, 4, 0)]),
    )
    .unwrap();
}

#[test]
fn a_config_runs_the_series_the_cli_runs_and_writes_a_report() {
    let dir = tempdir("series");
    frames(&dir);
    // Relative paths resolve against the config's directory, whatever the
    // working directory is.
    std::fs::write(
        dir.join("series.toml"),
        r#"
# Two-frame series, as `rmda defects data/frame.*.xyz --out results ...`.
[input]
files = ["data/frame.*.xyz"]

[run]
threads = 2

[defects]
reference = "global"
region = ["z > 0"]
out = "results"
defects_xyz = true
"#,
    )
    .unwrap();
    let elsewhere = std::env::temp_dir();
    let cfg = dir.join("series.toml");
    let (out, err, ok) = rmda_in(&elsewhere, &["run", cfg.to_str().unwrap()]);
    assert!(ok, "{out}\n{err}");
    assert!(err.contains("report written to"), "{err}");

    let (out, err, ok) = rmda_in(
        &dir,
        &[
            "--threads",
            "2",
            "defects",
            "data/frame.10.xyz",
            "data/frame.20.xyz",
            "--reference",
            "global",
            "--region",
            "z > 0",
            "--out",
            "direct",
            "--defects-xyz",
        ],
    );
    assert!(ok, "{out}\n{err}");
    // The same tables, timing and memory columns aside.
    let cmp_summary = |d: &str| -> Vec<String> {
        let text = std::fs::read_to_string(dir.join(d).join("summary.csv")).unwrap();
        let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
        let keep: Vec<usize> = (0..header.len())
            .filter(|&i| !["wall_s", "rss_mb", "peak_rss_mb"].contains(&header[i]))
            .collect();
        text.lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                keep.iter().map(|&i| f[i]).collect::<Vec<_>>().join(",")
            })
            .collect()
    };
    let summary = cmp_summary("results");
    assert_eq!(summary, cmp_summary("direct"));
    assert_eq!(summary.len(), 3, "header and two frames: {summary:?}");
    for f in ["grains.csv", "clusters.csv"] {
        assert_eq!(
            std::fs::read(dir.join("results").join(f)).unwrap(),
            std::fs::read(dir.join("direct").join(f)).unwrap(),
            "{f}"
        );
    }
    for k in ["10", "20"] {
        for f in ["vacancies.xyz", "interstitials.xyz"] {
            let p = |d: &str| dir.join(d).join("defects").join(k).join(f);
            assert_eq!(
                std::fs::read(p("results")).unwrap(),
                std::fs::read(p("direct")).unwrap()
            );
        }
    }

    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("results/report.json")).unwrap())
            .unwrap();
    assert_eq!(report["status"], "ok");
    assert_eq!(report["analysis"], "defects");
    assert_eq!(report["rmda_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(report["threads"], 2);
    assert_eq!(report["config"]["defects"]["reference"], "global");
    let frames = report["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0]["file_index"], 10);
    assert_eq!(frames[0]["vacancies"], 1);
    assert_eq!(frames[1]["vacancies"], 3);
    assert!(report["stages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["name"] == "defects.grain"));
    assert!(report["wall_s"].as_f64().unwrap() > 0.0);
    // Linux: CPU time over all threads, and the efficiency it gives.
    if cfg!(target_os = "linux") {
        assert!(report["cpu_s"].as_f64().unwrap() >= 0.0);
        assert!(report["parallel_efficiency"].as_f64().is_some());
    }
    let command = report["command"].as_str().unwrap();
    assert!(
        command.starts_with("rmda --threads=2 defects") && command.contains("'--region=z > 0'"),
        "{command}"
    );

    // A rerun with the same settings resumes (nothing to do); `--set` that
    // changes a result is refused without overwrite, like the CLI.
    let (out, err, ok) = rmda_in(&dir, &["run", "series.toml"]);
    assert!(ok, "{out}\n{err}");
    let (_, err, ok) = rmda_in(
        &dir,
        &["run", "series.toml", "--set", "defects.min_atoms=50"],
    );
    assert!(
        !ok,
        "a changed setting must not resume into the same tables"
    );
    assert!(err.contains("overwrite"), "{err}");
    let (out, err, ok) = rmda_in(
        &dir,
        &[
            "run",
            "series.toml",
            "--set",
            "defects.min_atoms=50",
            "--set",
            "defects.overwrite=true",
        ],
    );
    assert!(ok, "{out}\n{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dry_run_prints_the_resolved_command_with_overrides() {
    let dir = tempdir("dry");
    frames(&dir);
    std::fs::write(
        dir.join("c.toml"),
        "[input]\nfiles = [\"data/frame.*.xyz\"]\n[run]\nthreads = 4\n\
         [defects]\nsegment = true\nantisites = false\ncell = [10, 10, 30]\nout = \"r\"\n",
    )
    .unwrap();
    // `cell` is a global option: misplaced here.
    let (_, err, ok) = rmda_in(&dir, &["run", "c.toml", "--dry-run"]);
    assert!(!ok);
    assert!(
        err.contains("defects.cell is a global option: put it under [run]"),
        "{err}"
    );

    std::fs::write(
        dir.join("c.toml"),
        "[input]\nfiles = [\"data/frame.*.xyz\"]\n[run]\nthreads = 4\ncell = [25.32, 25.32, 60]\n\
         [defects]\nsegment = true\nantisites = false\nout = \"r\"\n",
    )
    .unwrap();
    let (out, err, ok) = rmda_in(
        &dir,
        &[
            "run",
            "c.toml",
            "--dry-run",
            "--set",
            "defects.reference=local",
            "--threads",
            "3",
        ],
    );
    assert!(ok, "{err}");
    let line = out.trim();
    // `--threads` given to `rmda run` wins; a false switch is left off; an
    // array is one occurrence of a multi-value option; files come last.
    assert!(
        line.starts_with("rmda --cell 25.32 25.32 60 --threads=3 defects"),
        "{line}"
    );
    assert!(
        line.contains("--reference=local") && line.contains("--segment"),
        "{line}"
    );
    assert!(!line.contains("antisites"), "{line}");
    assert!(
        line.contains(" -- ") && line.ends_with("frame.20.xyz"),
        "{line}"
    );
    assert!(!dir.join("r").exists(), "a dry run runs nothing");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_config_the_cli_would_not_take_is_refused_by_name() {
    let dir = tempdir("refused");
    frames(&dir);
    let cases: [(&str, &str); 8] = [
        (
            "[defects]\nrmsd_cutof = 0.1\n",
            "unknown key defects.rmsd_cutof: rmda defects has no option --rmsd-cutof",
        ),
        (
            "[defects]\nthreads = 4\n",
            "defects.threads is a global option: put it under [run]",
        ),
        (
            "[run]\nreference = \"local\"\n[defects]\nout = \"r\"\n",
            "run.reference is a command option: put it under [defects]",
        ),
        (
            "[defects]\nsegment = \"yes\"\n",
            "defects.segment is a switch: write segment = true",
        ),
        (
            "[defects]\nout = \"a\"\n[void]\nout = \"b\"\n",
            "names 2 commands",
        ),
        ("[input]\nfiles = [\"x\"]\n", "names no command"),
        (
            "[input]\nfiles = [\"data/none-*.xyz\"]\n[defects]\nout = \"r\"\n",
            "matches nothing",
        ),
        (
            "[input]\nfiles = [\"data/frame.10.xyz\"]\n[defects]\nreference = \"best\"\n",
            "invalid value 'best' for '--reference",
        ),
    ];
    for (text, want) in cases {
        std::fs::write(dir.join("bad.toml"), text).unwrap();
        let (_, err, ok) = rmda_in(&dir, &["run", "bad.toml"]);
        assert!(!ok, "accepted:\n{text}");
        assert!(err.contains(want), "for\n{text}\nwant {want:?} in\n{err}");
        assert!(!dir.join("r").exists(), "nothing runs:\n{text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_run_still_writes_its_report() {
    let dir = tempdir("failed");
    frames(&dir);
    std::fs::write(dir.join("data/frame.30.xyz"), "not an xyz file\n").unwrap();
    std::fs::write(
        dir.join("f.toml"),
        "[input]\nfiles = [\"data/frame.*.xyz\"]\n[defects]\nout = \"r\"\nregion = [\"z > 0\"]\n",
    )
    .unwrap();
    let (_, err, ok) = rmda_in(&dir, &["run", "f.toml"]);
    assert!(!ok);
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("r/report.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "error", "{err}");
    assert!(report["error"].as_str().is_some_and(|e| !e.is_empty()));
    // The frames before the bad one are done and reported.
    assert_eq!(report["frames"].as_array().unwrap().len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_config_runs_as_shards_and_merges() {
    let dir = tempdir("shards");
    frames(&dir);
    std::fs::write(
        dir.join("s.toml"),
        "[input]\nfiles = [\"data/frame.*.xyz\"]\n[defects]\nreference = \"global\"\n\
         region = [\"z > 0\"]\nout = \"results\"\n",
    )
    .unwrap();
    let (out, err, ok) = rmda_in(&dir, &["run", "s.toml", "--dry-run", "--shard", "1/2"]);
    assert!(ok && out.contains("--shard=1/2"), "{out}\n{err}");
    for i in 0..2 {
        let (out, err, ok) = rmda_in(&dir, &["run", "s.toml", "--shard", &format!("{i}/2")]);
        assert!(ok, "{out}\n{err}");
        assert!(dir
            .join(format!("results/shard-{i}-of-2/report.json"))
            .exists());
    }
    let (out, err, ok) = rmda_in(&dir, &["merge", "results"]);
    assert!(ok, "{out}\n{err}");
    let (out, err, ok) = rmda_in(&dir, &["run", "s.toml", "--set", "defects.out=\"whole\""]);
    assert!(ok, "{out}\n{err}");
    for t in ["grains.csv", "clusters.csv"] {
        assert_eq!(
            std::fs::read(dir.join("results").join(t)).unwrap(),
            std::fs::read(dir.join("whole").join(t)).unwrap(),
            "{t}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
