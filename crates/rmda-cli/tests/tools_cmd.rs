//! `rmda bench`, `rmda explain` and `--mem-limit`: tools for sizing and
//! comparing runs (design §21, §23.4).

use std::path::{Path, PathBuf};
use std::process::Command;

fn rmda_in(dir: &Path, args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_rmda"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to run rmda");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rmda_tools_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(name)
}

#[test]
fn bench_times_every_stage_and_checks_the_census() {
    let dir = tempdir("bench");
    let json = dir.join("b.json");
    let (out, err, code) = rmda_in(
        &dir,
        &[
            "--threads",
            "2",
            "bench",
            "--atoms",
            "3000",
            "--json",
            json.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "{out}\n{err}");
    for stage in [
        "ptm",
        "cna",
        "segment",
        "defects.global",
        "defects.local",
        "rdf",
    ] {
        assert!(out.lines().any(|l| l.starts_with(stage)), "{stage}:\n{out}");
    }
    // Both censuses find exactly the planted vacancies.
    assert_eq!(out.matches("as planted").count(), 2, "{out}");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(v["threads"], 2);
    assert_eq!(v["stages"].as_array().unwrap().len(), 6);
    assert!(v["stages"][0]["atoms_per_s"].as_f64().unwrap() > 0.0);
    let (_, err, code) = rmda_in(&dir, &["bench", "--structure", "hcp"]);
    assert_eq!(code, 2, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_reads_the_inputs_heads_and_runs_nothing() {
    let dir = tempdir("explain");
    // A gzipped input: the atom count comes from its inflated head.
    let gz = dir.join("frame.20.dump.gz");
    {
        use std::io::Write;
        let text = std::fs::read(data("fcc_ortho.dump")).unwrap();
        let out = std::process::Command::new("gzip")
            .arg("-c")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn();
        if let Ok(mut child) = out {
            child.stdin.take().unwrap().write_all(&text).unwrap();
            let o = child.wait_with_output().unwrap();
            std::fs::write(gz, o.stdout).unwrap();
        }
    }
    std::fs::copy(data("fcc_ortho.data"), dir.join("frame.10.data")).unwrap();
    std::fs::copy(data("fcc_ortho.xyz"), dir.join("frame.30.xyz")).unwrap();
    std::fs::write(
        dir.join("e.toml"),
        "[input]\nfiles = [\"frame.*\"]\n[defects]\nout = \"r\"\nshard = \"1/2\"\n",
    )
    .unwrap();
    let (out, err, code) = rmda_in(&dir, &["explain", "e.toml"]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert!(
        out.contains("command: rmda defects --out=r --shard=1/2"),
        "{out}"
    );
    assert!(
        out.contains("3 file(s), shard 1 of 2 takes 2 of them"),
        "{out}"
    );
    // Each format's declared count: data, gzipped dump, xyz.
    let files: Vec<&str> = out
        .lines()
        .filter(|l| l.starts_with("  ") && l.contains("/frame."))
        .collect();
    assert_eq!(files.len(), 3, "{out}");
    assert!(files.iter().all(|l| l.contains(" 32 atoms ")), "{out}");
    assert!(out.contains("memory:"), "{out}");
    assert!(!dir.join("r").exists(), "explain runs nothing");
    let (_, err, code) = rmda_in(&dir, &["explain", "e.toml", "--set", "defects.bogus=1"]);
    assert_eq!(code, 2, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_over_its_memory_limit_stops_with_exit_4() {
    let dir = tempdir("memlimit");
    let f = data("fcc_ortho.dump");
    let (_, err, code) = rmda_in(&dir, &["--mem-limit", "1K", "info", f.to_str().unwrap()]);
    assert_eq!(code, 4, "{err}");
    assert!(err.contains("over --mem-limit"), "{err}");
    let (_, err, code) = rmda_in(&dir, &["--mem-limit", "64G", "info", f.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    let (_, err, code) = rmda_in(&dir, &["--mem-limit", "12X", "info", f.to_str().unwrap()]);
    assert_eq!(code, 2, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
