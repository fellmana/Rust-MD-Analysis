//! Install/import and CLI-vs-Python parity tests for the `rmda` bindings.
//!
//! The bindings live in `rmda-py` as a `cdylib` with `test = false`, because
//! pyo3's `extension-module` feature leaves Rust test binaries with unresolved
//! Python symbols — so `rmda-py` cannot host tests of its own, and `cargo test
//! -p rmda-cli` does not build another crate's artifact either. These tests
//! therefore look for what a `cargo build --release -p rmda-py` produced and
//! *skip*, with the command to run, when the module is absent or older than the
//! binding source. Failing because somebody else's build is stale would say
//! nothing about the code.
//!
//! What is checked here is what a Rust-only test cannot reach:
//!
//!   * the module imports under `python3` and reports the CLI's version;
//!   * the frame Python hands back is the frame `rmda_core::io::read_frame`
//!     gives — cell, positions, types and ids, field for field;
//!   * every analysis returns exactly what the CLI printed for the same input
//!     (the detailed comparisons live in `python_parity.py`, which is runnable
//!     on its own);
//!   * frames built from arrays (`Frame.from_arrays`) get the checks a file
//!     read gets, and columns set with `f[name] = ...` feed the analyses;
//!   * an ASE `Atoms` in, an ASE `Atoms` out: `read`/`iread`/the analyses
//!     accept one wherever they accept a frame, and `from_ase`/`to_ase`
//!     reproduce the file reader's frames exactly (skipped when ASE itself
//!     is not importable);
//!   * an OVITO `DataCollection` in and out (`rmda.ovito`): the frame OVITO
//!     reads is the frame `rmda.read` gives, results land as properties,
//!     attributes and tables, and the conversion runs inside an OVITO
//!     pipeline (skipped when the `ovito` module is not importable).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tempdir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `python3` is not a build dependency, so its absence skips rather than fails.
fn python3() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn skip(what: &str, why: &str) {
    eprintln!("SKIPPED {what}: {why}");
}

/// Newest `rmda-py` artifact under `target/`, or `None` when there is none or
/// the newest one predates the binding source.
fn built_bindings() -> Option<PathBuf> {
    let root = repo_root();
    let mut newest_source = std::fs::metadata(root.join("crates/rmda-py/src/lib.rs"))
        .and_then(|m| m.modified())
        .ok()?;
    for extra in ["crates/rmda-py/Cargo.toml"] {
        if let Ok(m) = std::fs::metadata(root.join(extra)) {
            if let Ok(t) = m.modified() {
                newest_source = newest_source.max(t);
            }
        }
    }

    let target = root.join("target");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for profile in ["release", "debug"] {
        for name in [
            "librmda.so",
            "librmda.dylib",
            "librmda.dll",
            "rmda.dll",
            "rmda.pyd",
        ] {
            let p = target.join(profile).join(name);
            let Ok(md) = std::fs::metadata(&p) else {
                continue;
            };
            let Ok(m) = md.modified() else { continue };
            if best.as_ref().map_or(true, |(t, _)| m > *t) {
                best = Some((m, p));
            }
        }
    }
    let (mtime, path) = best?;
    if mtime < newest_source {
        return None;
    }
    Some(path)
}

/// Stage the built module as the `rmda` package Python imports: the
/// artifact as `rmda/_rmda.so` (`librmda.so` -> `_rmda.so`) beside the
/// package's own `rmda/__init__.py`, in a temp directory that is returned.
///
/// Staged once per test process: the tests run in parallel, and when every
/// one of them copied over the same file, one test's Python could import a
/// module another was still writing (`import rmda` failing with no message,
/// seen when a debug build replaced a release one). Each copy goes to a
/// temporary name and is renamed into place, so a reader in another process
/// sees the old file or the new one, never part of one.
fn staged_module() -> Option<PathBuf> {
    static STAGED: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    STAGED.get_or_init(stage_module).clone()
}

fn stage_module() -> Option<PathBuf> {
    let src = built_bindings()?;
    let dir = tempdir("rmda-bindings-tests");
    let pkg = dir.join("rmda");
    std::fs::create_dir_all(&pkg).ok()?;
    let ext = match src.extension().and_then(|e| e.to_str())? {
        "dylib" | "so" => "_rmda.so",
        _ => "_rmda.pyd",
    };
    let python_dir = repo_root().join("crates/rmda-py/python/rmda");
    let mut files: Vec<(PathBuf, String)> = vec![(src.clone(), ext.to_string())];
    for entry in std::fs::read_dir(&python_dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("py") {
            let name = path.file_name()?.to_str()?.to_string();
            files.push((path, name));
        }
    }
    for (from, name) in &files {
        let (from, name) = (from.as_path(), name.as_str());
        let dst = pkg.join(name);
        let fresh = match (
            std::fs::metadata(&dst).and_then(|m| m.modified()),
            std::fs::metadata(from).and_then(|m| m.modified()),
        ) {
            (Ok(d), Ok(s)) => d >= s,
            _ => false,
        };
        if !fresh {
            let tmp = pkg.join(format!(".staging-{}-{name}", std::process::id()));
            std::fs::copy(from, &tmp).ok()?;
            std::fs::rename(&tmp, &dst).ok()?;
        }
    }
    Some(dir)
}

fn bindings_or_skip(what: &str) -> Option<PathBuf> {
    if !python3() {
        skip(what, "`python3` is not on PATH");
        return None;
    }
    match staged_module() {
        Some(dir) => Some(dir),
        None => {
            skip(
                what,
                "no fresh `rmda-py` artifact; run `cargo build --release -p rmda-py` first",
            );
            None
        }
    }
}

fn run_script(module_dir: &Path, args: &[&str]) -> (String, String, bool) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_parity.py");
    let out = Command::new("python3")
        .arg(&script)
        .arg("--module-dir")
        .arg(module_dir)
        .arg("--rmda")
        .arg(env!("CARGO_BIN_EXE_rmda"))
        .args(args)
        .output()
        .expect("failed to run python3");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// `key: value` / `key: v1 v2 ...` lines, as both the CLI and the frame dump
/// in `python_parity.py` write them.
fn parse_keyed(text: &str) -> HashMap<String, Vec<f64>> {
    text.lines()
        .filter_map(|line| {
            let (k, rest) = line.split_once(": ")?;
            let vals: Option<Vec<f64>> = rest.split_whitespace().map(|v| v.parse().ok()).collect();
            let vals = vals?;
            if vals.is_empty() {
                None
            } else {
                Some((k.to_string(), vals))
            }
        })
        .collect()
}

fn series(map: &HashMap<String, Vec<f64>>, key: &str) -> Vec<f64> {
    map.get(key).cloned().unwrap_or_else(|| {
        panic!(
            "no `{key}` in the frame dump: {:?}",
            map.keys().collect::<Vec<_>>()
        )
    })
}

/// The module imports, and the version it reports is the CLI's version: a
/// binding built from a different revision silently disagreeing with the binary
/// beside it is exactly the drift these tests exist to catch.
#[test]
fn python_module_imports_and_reports_the_cli_version() {
    let Some(module_dir) = bindings_or_skip("import") else {
        return;
    };
    let out = Command::new("python3")
        .arg("-c")
        .arg("import rmda; print(rmda.__version__)")
        .env("PYTHONPATH", &module_dir)
        .output()
        .expect("failed to run python3");
    assert!(
        out.status.success(),
        "`import rmda` failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        version,
        env!("CARGO_PKG_VERSION"),
        "rmda.__version__ does not match the crate the CLI was built from"
    );
}

/// The frame Python reads is the frame the core reads: same cell, positions,
/// types and ids, compared field for field rather than through a summary that
/// could agree while the data underneath differs.
#[test]
fn python_frame_matches_the_rust_reader() {
    let Some(module_dir) = bindings_or_skip("frame getters") else {
        return;
    };
    let fixture = repo_root().join("tests/data/triclinic_two_species.xyz");
    let (stdout, stderr, ok) = run_script(
        &module_dir,
        &[
            "--mode",
            "frame",
            "--fixture",
            fixture.to_str().expect("fixture path is utf-8"),
        ],
    );
    assert!(ok, "frame dump failed:\n{stdout}\n{stderr}");

    let got = parse_keyed(&stdout);
    let frame = rmda_core::io::read_frame(&fixture).expect("the core must read the fixture");

    assert_eq!(series(&got, "natoms"), vec![frame.natoms() as f64]);
    assert_eq!(series(&got, "ntypes"), vec![frame.ntypes() as f64]);

    let cell: Vec<f64> = frame.cell.h_matrix().m.iter().flatten().copied().collect();
    assert_eq!(series(&got, "cell"), cell, "the 3x3 cell differs");

    let pos: Vec<f64> = frame
        .particles
        .positions
        .iter()
        .flatten()
        .copied()
        .collect();
    assert_eq!(series(&got, "pos"), pos, "positions differ");

    let types: Vec<f64> = frame.particles.types.iter().map(|&t| t as f64).collect();
    assert_eq!(series(&got, "types"), types, "types differ");

    let ids: Vec<f64> = frame.particles.ids.iter().map(|&x| x as f64).collect();
    assert_eq!(series(&got, "ids"), ids, "ids differ");
}

/// Every CLI subcommand the bindings expose, checked against its Python twin:
/// scalars, histogram bins, bin edges and the per-atom labels written to disk.
#[test]
fn cli_and_python_analyses_agree() {
    let Some(module_dir) = bindings_or_skip("parity") else {
        return;
    };
    let (stdout, stderr, ok) = run_script(&module_dir, &["--mode", "parity"]);
    assert!(ok, "CLI and Python bindings disagree:\n{stderr}\n{stdout}");
    assert!(
        stdout.contains("parity ok:"),
        "the parity script did not report success:\n{stdout}\n{stderr}"
    );
}

/// Newest `target/wheels/rmda-*-abi3-*.whl` (what `maturin build` writes from
/// the root `pyproject.toml`), or `None` when there is none or it predates the
/// binding source.
fn built_wheel() -> Option<PathBuf> {
    let root = repo_root();
    let mut newest_source = std::fs::metadata(root.join("crates/rmda-py/src/lib.rs"))
        .and_then(|m| m.modified())
        .ok()?;
    for extra in [
        "crates/rmda-py/Cargo.toml",
        "crates/rmda-py/python/rmda/__init__.py",
        "pyproject.toml",
    ] {
        if let Ok(t) = std::fs::metadata(root.join(extra)).and_then(|m| m.modified()) {
            newest_source = newest_source.max(t);
        }
    }
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for e in std::fs::read_dir(root.join("target/wheels"))
        .ok()?
        .flatten()
    {
        let p = e.path();
        let name = p.file_name()?.to_string_lossy().into_owned();
        if !(name.starts_with("rmda-") && name.contains("-abi3-") && name.ends_with(".whl")) {
            continue;
        }
        let Ok(m) = e.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if best.as_ref().map_or(true, |(t, _)| m > *t) {
            best = Some((m, p));
        }
    }
    let (mtime, path) = best?;
    (mtime >= newest_source).then_some(path)
}

/// The abi3 wheel installs into a fresh virtual environment (system site
/// packages visible, for numpy and ASE; nothing downloaded), imports as `rmda`
/// with the CLI's version, and passes the whole parity script as an installed
/// package, and installs the `rmda` console script, which runs the command
/// line. Skips unless `maturin build --release` (in the repository root) has
/// produced a wheel newer than the binding source.
#[test]
fn abi3_wheel_installs_and_passes_parity() {
    if !python3() {
        return skip("wheel", "`python3` is not on PATH");
    }
    let Some(wheel) = built_wheel() else {
        return skip(
            "wheel",
            "no fresh abi3 wheel; run `maturin build --release` in the repository root",
        );
    };
    let venv = tempdir("rmda-wheel-venv");
    let _ = std::fs::remove_dir_all(&venv);
    let ok = Command::new("python3")
        .args(["-m", "venv", "--system-site-packages"])
        .arg(&venv)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return skip("wheel", "`python3 -m venv` is unavailable");
    }
    let py = venv.join("bin/python");
    let out = Command::new(&py)
        .args([
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--force-reinstall",
            "-q",
        ])
        .arg(&wheel)
        .output()
        .expect("failed to run pip");
    assert!(
        out.status.success(),
        "pip install of {} failed:\n{}",
        wheel.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    // Import from outside the repository, so nothing local shadows the
    // installed package.
    let out = Command::new(&py)
        .args([
            "-c",
            "import rmda; print(rmda.__version__); print(rmda.__file__)",
        ])
        .current_dir(&venv)
        .output()
        .expect("failed to run the venv python");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "`import rmda` from the wheel failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(env!("CARGO_PKG_VERSION")));
    assert!(
        lines
            .next()
            .is_some_and(|f| f.starts_with(venv.to_str().unwrap())),
        "imported a module outside the venv: {text}"
    );
    // The console script: the same command line as the binary.
    let out = Command::new(venv.join("bin/rmda"))
        .arg("--version")
        .current_dir(&venv)
        .output()
        .expect("failed to run the installed `rmda` script");
    assert!(
        out.status.success()
            && String::from_utf8_lossy(&out.stdout).trim()
                == format!("rmda {}", env!("CARGO_PKG_VERSION")),
        "`rmda --version` from the wheel: {:?}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(venv.join("bin/rmda"))
        .args(["info"])
        .arg(repo_root().join("tests/data/fcc_ortho.xyz"))
        .output()
        .expect("failed to run the installed `rmda` script");
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stderr).contains("read 1 frame"),
        "`rmda info` from the wheel: {:?}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    // The OVITO modifiers are registered for OVITO Pro's modifier list; when
    // the venv can import ovito they also resolve to the classes.
    let entry_points = r#"
from importlib.metadata import entry_points
eps = sorted((e.name, e.value) for e in entry_points(group="OVITO.Modifier"))
assert eps == [
    ("RMDA defect census", "rmda.ovito:DefectsModifier"),
    ("RMDA grain segmentation", "rmda.ovito:SegmentModifier"),
    ("RMDA voids", "rmda.ovito:VoidModifier"),
], eps
try:
    import ovito  # noqa: F401
except ImportError:
    pass
else:
    from ovito.pipeline import ModifierInterface
    for e in entry_points(group="OVITO.Modifier"):
        assert issubclass(e.load(), ModifierInterface), e
print("ok")
"#;
    let out = Command::new(&py)
        .args(["-W", "ignore", "-c", entry_points])
        .current_dir(&venv)
        .output()
        .expect("failed to run the venv python");
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "ok",
        "OVITO entry points of the wheel:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_parity.py");
    let out = Command::new(&py)
        .arg(&script)
        .arg("--installed")
        .arg("--rmda")
        .arg(env!("CARGO_BIN_EXE_rmda"))
        .output()
        .expect("failed to run the parity script");
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        out.status.success() && stdout.contains("parity ok:"),
        "the installed wheel disagrees with the CLI:\n{stderr}\n{stdout}"
    );
}

/// Design §22 surfaces beyond the per-case parity: `f.run` with kwargs,
/// `f.write` round-tripping attached props, `rmda.segment` after a PTM step, `rmda.ws` against a reference frame, and a `Pipeline`
/// over `iread`. Each branch asserts its own numbers, so a regression in any of
/// them fails here rather than in the aggregate parity run.
#[test]
fn run_write_and_pipeline_surfaces_work() {
    let Some(module_dir) = bindings_or_skip("run/write/pipeline") else {
        return;
    };
    let fixture = repo_root()
        .join("tests/data/m2_fcc_thermal.xyz")
        .to_str()
        .expect("fixture path is utf-8")
        .to_string();
    let script = r#"
import rmda, tempfile, os

fixture = os.environ["fixture"]
f = rmda.read(fixture)
rmda.ptm(f)
# The fixture is a 32-atom cell. The segmentation's `min_size` default of 100
# would dissolve it, so the threshold is lowered where this script is
# exercising bindings rather than the default.
g = rmda.segment(f, min_size=0)
assert g["segment_grains"] == 1, g
assert len(f["Grain"]) == f.natoms

# f.run forwards kwargs; the same analysis spelled through the module agrees.
s = f.run("wc", shells="0,12")
assert f.run("wc", shells="0,12")["wc_m1_nneigh"] == 12.0, s
try:
    f.run("rdf")
    raise SystemExit("f.run('rdf') without rmax must be refused")
except ValueError:
    pass
assert f.run("segment", min_size=0).get("segment_grains") == 1

# f.write emits attached props as extended-XYZ columns that re-read atom for atom.
p = tempfile.mktemp(suffix=".xyz")
f.run("ptm", rmsd_cutoff=0.1)
f.run("segment", min_size=0)
f.write(p)
spec = open(p).read().splitlines()[1].split("Properties=")[1].split(" ")[0]
parts = spec.split(":")
offset = 0
span = {}
for i in range(0, len(parts), 3):
    span[parts[i]] = (offset, offset + int(parts[i + 2]))
    offset += int(parts[i + 2])
rows = [ln.split() for ln in open(p).read().splitlines()[2:]]
lo, hi = span["StructureType"]
assert [int(r[lo:hi][0]) for r in rows] == [int(x) for x in f["StructureType"]], "StructureType column"
lo, hi = span["Grain"]
assert [int(r[lo:hi][0]) for r in rows] == [int(x) for x in f["Grain"]], "Grain column"

# Pipeline over iread: ptm then segment reproduce the single-frame call.
pipe = rmda.Pipeline(rmda.iread(fixture))
pipe.add("ptm")
pipe.add("segment", {"min_size": 0})
frames = list(pipe.run())
assert len(frames) == 1 and frames[0].run("segment", min_size=0)["segment_grains"] == 1

# Reference-mode WS: a frame against itself is defect-free on both paths.
a, n = 3.5, 6
box = n * a
sites = []
for i in range(n):
    for j in range(n):
        for k in range(n):
            sites.append((i*a, j*a, k*a))
            sites.append((i*a+a/2, j*a+a/2, k*a))
            sites.append((i*a+a/2, j*a, k*a+a/2))
            sites.append((i*a, j*a+a/2, k*a+a/2))
ref = tempfile.mktemp(suffix=".extxyz")
with open(ref, "w") as fo:
    fo.write(f"{len(sites)}\n")
    fo.write(f'Lattice="{box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0" '
             'pbc="T T T" Properties=species:S:1:pos:R:3:type:I:1 Timestep=0\n')
    for (x, y, z) in sites:
        fo.write(f"X {x:.6f} {y:.6f} {z:.6f} 1\n")
w = rmda.ws(rmda.read(ref), reference=ref)
assert w["ws_vacancies"] == 0.0 and w["ws_interstitials"] == 0.0, w

# The design §22 surface: nbins == rmax/dr, shells as int or list, and the lattice
# and self-fit WS engines the CLI's --lattice/--self drive.
r = rmda.rdf(f, 3.0, dr=0.125)["histograms"]
assert rmda.rdf(f, 3.0, nbins=24)["histograms"] == r, "rdf nbins"
assert rmda.warren_cowley(f, 12) == rmda.warren_cowley(f), "wc shells int"
assert rmda.warren_cowley(f, [0, 12]) == rmda.warren_cowley(f), "wc shells list"

nn = a / 2**0.5
cur = rmda.read(ref)
out = rmda.ws(cur, lattice="fcc", nn_distance=nn)
assert out["ws_vacancies"] == 0.0 and out["ws_interstitials"] == 0.0, out
rows = open(ref).read().splitlines()
header = rows[1]
positions = [ln.split()[1:4] for ln in rows[2:]]
with open(ref, "w") as fo:
    fo.write(f"{len(positions)-1}\n{header}\n")
    for row in positions[:-1]:
        fo.write("X " + " ".join(row) + " 1\n")
found = rmda.ws(rmda.read(ref), fit_self=True, lattice="fcc", nn_distance=nn)
assert found["ws_vacancies"] == 1.0, found

# Design §22: per-atom data is numpy. Frame getters are read-only,
# zero-copy views that outlive the Python name of their frame; analysis
# outputs are typed arrays, multi-component ones (N, c).
import numpy as np
g = rmda.read(fixture)
pos, types, ids = g.positions, g.types, g.ids
assert isinstance(pos, np.ndarray) and pos.dtype == np.float64 and pos.shape == (g.natoms, 3)
assert types.dtype == np.uint32 and ids.dtype == np.int64 and g.cell.shape == (3, 3)
assert not pos.flags.writeable and not types.flags.writeable
try:
    pos[0, 0] = 1.0
    raise SystemExit("a frame's positions must be read-only")
except ValueError:
    pass
first = pos[0].tolist()
del g
assert pos[0].tolist() == first, "a view must keep its frame alive"
g = rmda.read(fixture)
rmda.ptm(g)
assert g["StructureType"].dtype == np.uint8
assert g["StructureOrientation"].shape == (g.natoms, 4)
# from_ase takes numpy arrays without a Python loop per atom.
class Fake:
    def get_positions(self): return np.asarray(g.positions, dtype=np.float32)
    def get_cell(self): return np.asarray(g.cell)
    def get_pbc(self): return np.array([True, True, True])
    def get_chemical_symbols(self): return ["Cu"] * g.natoms
h = rmda.from_ase(Fake())
assert np.allclose(h.positions, g.positions, atol=1e-5) and h.natoms == g.natoms

# set_threads reaches the core's thread setting.
rmda.set_threads(2)
assert rmda.get_threads() == 2
rmda.set_threads(0)
assert rmda.get_threads() >= 1

# The GIL is released while an analysis runs: two threads on separate
# frames both finish, and agree with a sequential call.
import threading
res = [None, None]
def work(i):
    res[i] = rmda.ptm(rmda.read(fixture))
ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]
[t.start() for t in ts]
[t.join() for t in ts]
assert res[0] == res[1] == rmda.ptm(rmda.read(fixture)), res

# defects and segment are in the step table too.
p2 = rmda.Pipeline(rmda.iread(fixture))
p2.add("segment", {"min_size": 0})
p2.add("defects", {"grain_column": "Grain", "min_atoms": 10})
out = list(p2.run())
assert "Grain" in out[0] and out[0]["Grain"].dtype == np.int32
print("ok")
"#;
    let module = module_dir.to_str().expect("module dir is utf-8");
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .env("PYTHONPATH", module)
        .env("fixture", &fixture)
        .output()
        .expect("failed to run python3");
    assert!(
        out.status.success(),
        "design §22 binding surfaces failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
}

/// ASE `Atoms` in, ASE `Atoms` out (design §22): `rmda.read`/`rmda.iread`
/// and every analysis accept one wherever they accept a frame, `from_ase`
/// and `to_ase` convert explicitly, and both directions reproduce the file
/// reader field for field — same cell, positions, types and ids, and a
/// byte-identical write. Skipped when ASE itself is not importable (the
/// script exits 3).
#[test]
fn ase_atoms_interop() {
    const WHAT: &str = "ase interop";
    let Some(module_dir) = bindings_or_skip(WHAT) else {
        return;
    };
    let fixture = repo_root().join("tests/data/triclinic_two_species.xyz");
    let nonper = repo_root().join("tests/data/cluster_nonperiodic.xyz");
    let script = r#"
import os, tempfile
try:
    import ase, ase.io
except ImportError:
    raise SystemExit(3)
import rmda

fixture, nonper = os.environ["fixture"], os.environ["nonper"]


def same_frame(a, b, what):
    assert a.natoms == b.natoms, (what, "natoms", a.natoms, b.natoms)
    assert a.ntypes == b.ntypes, (what, "ntypes", a.ntypes, b.ntypes)
    assert a.cell.tolist() == b.cell.tolist(), (what, "cell", a.cell, b.cell)
    assert a.positions.tolist() == b.positions.tolist(), (what, "positions")
    assert a.types.tolist() == b.types.tolist(), (what, "types", a.types, b.types)
    assert a.ids.tolist() == b.ids.tolist(), (what, "ids", a.ids, b.ids)


def outcome(call):
    try:
        return ("ok", repr(call()))
    except Exception as e:
        return ("err", f"{type(e).__name__}: {e}")


def check(path):
    at = ase.io.read(path)
    f = rmda.read(path)
    # read(atoms) and from_ase(atoms) both reproduce the file reader.
    same_frame(f, rmda.read(at), path)
    same_frame(f, rmda.from_ase(at), path)
    # ...and through the writer: cell origin, pbc and type labels are all
    # header data, so a byte-identical write is the strongest parity
    # statement reachable from Python.
    written = []
    for other in (f, rmda.read(at), rmda.from_ase(at)):
        p = tempfile.mktemp(suffix=".xyz")
        other.write(p)
        written.append(open(p).read())
    assert written[0] == written[1] == written[2], (path, "write round trip")

    # Every analysis answers the same thing for the Atoms as for the frame.
    # The outcome tuple keeps a shared failure (both sides raising the same
    # validation error) equal, while want_ok still refuses a case where the
    # Atoms is rejected outright.
    def same(make, what, want_ok=True):
        lhs, rhs = outcome(lambda: make(at)), outcome(lambda: make(rmda.read(path)))
        assert lhs == rhs, (path, what, lhs, rhs)
        if want_ok:
            assert lhs[0] == "ok", (path, what, lhs)

    same(lambda x: rmda.cna(x), "cna")
    same(lambda x: rmda.ptm(x), "ptm")
    same(lambda x: rmda.rdf(x, 2.0, dr=0.1), "rdf")
    same(lambda x: rmda.cluster(x, cutoff=3.5), "cluster")
    same(lambda x: rmda.warren_cowley(x, shells="0,4"), "warren_cowley")
    same(lambda x: rmda.ws(x, reference=path), "ws")
    # segment may refuse a fixture (no crystalline grain); the Atoms must then
    # be refused in exactly the same way.
    same(lambda x: rmda.segment(x), "segment", want_ok=False)


check(fixture)
check(nonper)

# A cell-less Atoms gets the box the reader gives a cell-less file: the atoms'
# own bounding box, cut to them and non-periodic. Not a 1e30 cube -- that is a
# way of saying "none" whose volume (1e90) turns every density estimate in the
# toolkit into nonsense.
h2 = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
plain = tempfile.mktemp(suffix=".xyz")
open(plain, "w").write("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
same_frame(rmda.read(plain), rmda.from_ase(h2), "cell-less")
assert rmda.from_ase(h2).cell.tolist() == [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
], "cell-less box"

# A declared type column wins on both paths; the labels stay the symbols.
typed = tempfile.mktemp(suffix=".xyz")
open(typed, "w").write(
    '2\nProperties=species:S:1:pos:R:3:type:I:1 pbc="T T T"\n'
    "Cu 0.0 0.0 0.0 7\nZn 1.0 1.0 1.0 2\n"
)
ta = ase.io.read(typed)
t_file, t_ase = rmda.read(typed), rmda.read(ta)
same_frame(t_file, t_ase, "type column")
same_frame(t_file, rmda.from_ase(ta), "type column")
assert t_file.types.tolist() == [7, 2] and t_file.ntypes == 7, (t_file.types, t_file.ntypes)
tw = []
for other in (t_file, t_ase):
    p = tempfile.mktemp(suffix=".xyz")
    other.write(p)
    tw.append(open(p).read())
assert tw[0] == tw[1], "type column write round trip"

# iread: an Atoms is one frame; a generator or a list of them is a stream.
n = rmda.read(fixture).natoms
assert [fr.natoms for fr in rmda.iread(fixture)] == [n], "iread path"
assert [fr.natoms for fr in rmda.iread(ase.io.read(fixture))] == [n], "iread Atoms"
assert [fr.natoms for fr in rmda.iread([ase.io.read(fixture)])] == [n], "iread list"
assert [fr.natoms for fr in rmda.iread(ase.io.iread(fixture))] == [n], "iread generator"
same_frame(
    rmda.read(fixture),
    next(iter(rmda.iread(ase.io.read(fixture)))),
    "iread(Atoms) frame",
)

# A Pipeline over an Atoms-backed stream runs like the file one.
pipe = rmda.Pipeline(rmda.iread(ase.io.read(fixture)))
pipe.add("ptm")
frames = list(pipe.run())
assert len(frames) == 1 and frames[0].natoms == n, frames

# to_ase hands the frame back as the Atoms it came from...
at0 = ase.io.read(fixture)
back_at = rmda.to_ase(rmda.read(fixture))
assert back_at.get_chemical_symbols() == at0.get_chemical_symbols()
assert back_at.get_positions().tolist() == at0.get_positions().tolist()
assert back_at.get_atomic_numbers().tolist() == at0.get_atomic_numbers().tolist()
assert back_at.get_cell().tolist() == at0.get_cell().tolist()
assert back_at.get_pbc().tolist() == at0.get_pbc().tolist()
same_frame(rmda.read(fixture), rmda.from_ase(back_at), "to_ase/from_ase")

# ...and attached per-atom properties arrive as arrays under their own names.
fp = rmda.read(fixture)
rmda.ptm(fp)
st = rmda.to_ase(fp).arrays["StructureType"]
assert st.tolist() == [int(v) for v in fp["StructureType"]], st.tolist()

# Anything that is neither a path nor an Atoms is refused, in both directions.
def refused(call, what):
    try:
        call()
    except TypeError:
        return
    raise SystemExit(f"{what} must raise TypeError")


refused(lambda: rmda.read(42), "read(42)")
refused(lambda: rmda.iread(42), "iread(42)")
refused(lambda: rmda.from_ase(42), "from_ase(42)")
refused(lambda: rmda.rdf(42, rmax=2.0), "rdf(42)")

# Ids and extra arrays come in from the Atoms and go back out to one.
import numpy as np
at = ase.io.read(fixture)
nat = len(at)
at.new_array("id", (np.arange(nat) * 7 + 3).astype(np.int64))
at.new_array("Grain", np.ones(nat, dtype=np.int32))
at.new_array("c_pe", np.linspace(-1.0, 1.0, nat))
fa = rmda.from_ase(at)
assert fa.ids.tolist() == at.arrays["id"].tolist(), fa.ids
assert fa.columns == ["Grain", "c_pe"], fa.columns
assert fa["Grain"].dtype == np.int32 and fa["c_pe"].tolist() == at.arrays["c_pe"].tolist()
back = rmda.to_ase(fa)
assert back.arrays["id"].tolist() == at.arrays["id"].tolist()
assert back.arrays["c_pe"].tolist() == at.arrays["c_pe"].tolist()
# ...and 1..N ids are not written out at all.
assert "id" not in rmda.to_ase(rmda.read(fixture)).arrays
# Duplicate ids (a repeated cell keeps them) are refused, not renumbered.
dup = at.repeat((2, 1, 1))
try:
    rmda.from_ase(dup)
    raise SystemExit("duplicate ids must be refused")
except ValueError as e:
    assert "not unique" in str(e), e

# Type names that are no element survive to_ase -> from_ase.
named = rmda.Frame.from_arrays(
    rmda.read(fixture).positions, rmda.read(fixture).cell,
    types=[1 + i % 2 for i in range(nat)], type_names={1: "W1", 2: "W2"},
)
na = rmda.to_ase(named)
assert na.info["type_names"] == {1: "W1", 2: "W2"}, na.info
nb = rmda.from_ase(na)
assert nb.type_names == ["W1", "W2"] and nb.types.tolist() == named.types.tolist()
# Element names need none of that.
assert "type" not in rmda.to_ase(rmda.read(fixture)).arrays

# ws takes the reference as a path, a Frame or an Atoms, with one answer.
by_path = rmda.ws(rmda.read(fixture), reference=fixture)
assert by_path == rmda.ws(rmda.read(fixture), reference=rmda.read(fixture))
assert by_path == rmda.ws(rmda.read(fixture), reference=ase.io.read(fixture))

print("ok")
"#;
    let out = Command::new("python3")
        .arg("-c")
        .arg(script)
        .env(
            "PYTHONPATH",
            module_dir.to_str().expect("module dir is utf-8"),
        )
        .env("fixture", fixture.to_str().expect("fixture path is utf-8"))
        .env("nonper", nonper.to_str().expect("fixture path is utf-8"))
        .output()
        .expect("failed to run python3");
    if out.status.code() == Some(3) {
        skip(WHAT, "ase is not importable");
        return;
    }
    assert!(
        out.status.success(),
        "ASE interop failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
}

/// Run `script` under `python3` with the staged module; `envs` are extra
/// environment variables. Exit code 3 is the script's "a dependency is not
/// importable" and skips.
fn run_inline(what: &str, module_dir: &Path, script: &str, envs: &[(&str, PathBuf)]) -> bool {
    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(script).env(
        "PYTHONPATH",
        module_dir.to_str().expect("module dir is utf-8"),
    );
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("failed to run python3");
    if out.status.code() == Some(3) {
        skip(what, &String::from_utf8_lossy(&out.stderr));
        return false;
    }
    assert!(
        out.status.success(),
        "{what} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
    true
}

/// `Frame.from_arrays` builds the frame a file read gives -- field for field
/// and byte for byte through the writer -- and refuses what a reader refuses
/// (duplicate ids, a type below 1, a non-finite coordinate, a wrong shape).
/// Input columns from `properties=` and columns set with `f[name] = ...` feed
/// `grain_column=` the same way.
#[test]
fn frames_from_arrays_and_columns() {
    const WHAT: &str = "from_arrays";
    let Some(module_dir) = bindings_or_skip(WHAT) else {
        return;
    };
    let script = r#"
import os, tempfile
import numpy as np
import rmda

fixture = os.environ["fixture"]
f = rmda.read(fixture)
g = rmda.Frame.from_arrays(
    f.positions, f.cell, pbc=f.pbc, origin=f.origin,
    types=f.types, type_names=f.type_names, ids=f.ids,
)
for k in ("positions", "types", "ids", "cell"):
    assert np.array_equal(getattr(f, k), getattr(g, k)), k
assert (f.pbc, f.origin, f.type_names) == (g.pbc, g.origin, g.type_names)
written = []
for fr in (f, g):
    p = tempfile.mktemp(suffix=".xyz")
    fr.write(p)
    written.append(open(p).read())
assert written[0] == written[1], "write round trip"

# Names instead of ids: coded 1.. in sorted order, as the xyz reader does.
names = [f.type_names[t - 1] for t in f.types]
h = rmda.Frame.from_arrays(f.positions, f.cell, pbc=f.pbc, origin=f.origin, types=names)
assert h.types.tolist() == f.types.tolist() and h.type_names == f.type_names

# No cell: the inferred, non-periodic box of a cell-less file.
c = rmda.Frame.from_arrays(f.positions)
assert c.pbc == [False, False, False] and c.types.tolist() == [1] * f.natoms

def refused(kind, text, **kw):
    args = dict(positions=f.positions, cell=f.cell)
    args.update(kw)
    pos = args.pop("positions")
    try:
        rmda.Frame.from_arrays(pos, **args)
    except kind as e:
        assert text in str(e), (text, str(e))
        return
    raise SystemExit(f"from_arrays({sorted(kw)}) must raise {kind.__name__}")

n = f.natoms
refused(ValueError, "not unique", ids=[1] * n)
refused(ValueError, "not a positive integer", types=[0] * n)
refused(ValueError, "not an integer", ids=np.arange(n) + 0.5)
bad = np.array(f.positions)
bad[1, 2] = np.nan
refused(ValueError, "non-finite", positions=bad)
refused(ValueError, "expected shape", positions=np.zeros((n, 2)))
refused(ValueError, "need a cell", cell=None, pbc=(True, True, True))
refused(ValueError, "expected shape", properties={"g": np.zeros(n + 1)})
refused(TypeError, "must be numeric", properties={"s": np.array(["x"] * n)})

# Input columns: listed, typed, and read by grain_column=.
k = rmda.Frame.from_arrays(
    f.positions, f.cell, pbc=f.pbc, origin=f.origin, types=f.types,
    properties={"G": np.ones(n, dtype=np.int32), "v": np.zeros((n, 3), dtype=np.float32)},
)
assert k.columns == ["G", "v"] and "G" in k and k.properties() == []
assert k["G"].dtype == np.int32 and k["v"].shape == (n, 3) and k["v"].dtype == np.float32
from_column = rmda.defects(k, grain_column="G")
# A column set on the frame is an attached property, used the same way.
k["H"] = np.ones(n, dtype=np.int32)
assert k.properties() == ["H"] and k["H"].tolist() == [1] * n
assert rmda.defects(k, grain_column="H")["defects_grains"] == from_column["defects_grains"] == 1
try:
    k["bad"] = np.zeros(n + 1)
    raise SystemExit("a wrong-length column must be refused")
except ValueError:
    pass

print("ok")
"#;
    run_inline(
        WHAT,
        &module_dir,
        script,
        &[("fixture", repo_root().join("tests/data/fcc_ortho.xyz"))],
    );
}

/// `rmda.ovito`: the frame OVITO's reader gives is the frame `rmda.read`
/// gives (positions, cell, pbc, origin, ids, types and extra columns) on the
/// fixtures, a frame survives frame -> OVITO -> frame bit for bit, results
/// land as particle properties, attributes and `rmda.*` tables, the
/// conversion runs inside an OVITO pipeline, and the census agrees with
/// OVITO's own Wigner-Seitz modifier on planted vacancies. Skipped when the
/// `ovito` module is not importable.
#[test]
fn ovito_interop() {
    const WHAT: &str = "ovito interop";
    let Some(module_dir) = bindings_or_skip(WHAT) else {
        return;
    };
    let script = r#"
import os, sys, warnings
warnings.filterwarnings("ignore")
try:
    import ovito
    from ovito.io import import_file
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import WignerSeitzAnalysisModifier
except ImportError:
    sys.stderr.write("the ovito module is not importable")
    raise SystemExit(3)
import numpy as np
import rmda
import rmda.ovito as ro

data = os.environ["data"]

def same(a, b, what, rotated=False, tol=0.0):
    assert a.natoms == b.natoms, (what, a.natoms, b.natoms)
    assert np.array_equal(a.ids, b.ids), (what, "ids")
    assert np.array_equal(a.types, b.types), (what, "types", a.types, b.types)
    assert a.pbc == b.pbc, (what, a.pbc, b.pbc)
    if rotated:
        # A general triclinic LAMMPS box: rmda rotates it into restricted
        # form, OVITO keeps it. Same cell up to a rotation, so same metric.
        assert np.allclose(a.cell @ a.cell.T, b.cell @ b.cell.T, atol=1e-9), what
    else:
        assert np.array_equal(a.cell, b.cell), (what, a.cell, b.cell)
        assert a.origin == b.origin, (what, a.origin, b.origin)
        assert np.allclose(a.positions, b.positions, atol=tol, rtol=0), (what, "positions")

cases = [
    ("fcc_ortho.xyz", "names", False), ("hcp_ortho.xyz", "names", False),
    ("triclinic_two_species.xyz", "names", False), ("cluster_nonperiodic.xyz", "names", False),
    ("m2_fcc_thermal.xyz", "names", False), ("fcc_ortho.dump", "ids", False),
    ("triclinic_tilted.dump", "ids", False), ("binary2txt_oracle.dump", "ids", False),
    ("fcc_ortho.data", "ids", False), ("triclinic.dump", "ids", True),
]
for name, mode, rotated in cases:
    path = os.path.join(data, name)
    a = rmda.read(path)
    b = ro.from_ovito(import_file(path).compute(), types=mode)
    # A tilted dump's positions agree to the last bit of a wrap.
    same(a, b, name, rotated, tol=1e-12 if name == "triclinic_tilted.dump" else 0.0)
    # OVITO keeps per-type masses as a Mass property; RMDA as type masses.
    assert [c for c in b.columns if c != "Mass"] == a.columns, (name, a.columns, b.columns)

# frame -> OVITO -> frame, bit for bit, with an origin, scrambled ids and a column.
rng = np.random.default_rng(7)
cube = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
grid = np.array([[i, j, k] for i in range(6) for j in range(6) for k in range(6)])
lat = (grid[:, None, :] + cube[None, :, :]).reshape(-1, 3) * 3.615
keep = np.ones(len(lat), bool)
keep[[5, 100, 501]] = False
pos = lat[keep] + rng.normal(0, 0.03, (keep.sum(), 3)) + 5.0
cell = np.eye(3) * 6 * 3.615
f = rmda.Frame.from_arrays(pos, cell, origin=(5, 5, 5), types=["Cu"] * keep.sum(),
                           ids=rng.permutation(keep.sum()) + 100,
                           properties={"c_pe": rng.random(keep.sum())})
d = ro.to_ovito(f)
g = ro.from_ovito(d, types="names")
same(f, g, "round trip")
assert g.type_names == f.type_names and g.columns == f.columns
assert np.array_equal(g["c_pe"], f["c_pe"])

# Results: attached properties, scalars as attributes, tables as rmda.* tables.
seg = rmda.segment(f)
res = rmda.defects(f)
out = ro.to_ovito(f, ro.to_ovito(f), result=res)
assert out.attributes["defects_vacancies"] == res["defects_vacancies"] == 3
for name in f.properties():
    assert np.array_equal(np.asarray(out.particles[name]), f[name]), name
grains = out.tables["rmda.defects.grains"]
assert grains.count == 1 and int(np.asarray(grains["vacancies"])[0]) == 3
assert [t.name for t in grains["status"].types] == ["measured"]
assert out.tables["rmda.defects.vacancies"].count == 3

# Inside a pipeline, as a plain modifier function.
def rmda_defects(frame, data):
    fr = ro.from_ovito(data, types="names")
    ro.to_ovito(fr, data, result=rmda.defects(fr))

pipe = Pipeline(source=StaticSource(data=ro.to_ovito(f)))
pipe.modifiers.append(rmda_defects)
assert pipe.compute().attributes["defects_vacancies"] == 3

# OVITO's own Wigner-Seitz on the same frame against the perfect lattice.
perfect = rmda.Frame.from_arrays(lat + 5.0, cell, origin=(5, 5, 5), types=["Cu"] * len(lat))
pipe = Pipeline(source=StaticSource(data=ro.to_ovito(f)))
ws = WignerSeitzAnalysisModifier()
ws.reference = StaticSource(data=ro.to_ovito(perfect))
pipe.modifiers.append(ws)
o = pipe.compute()
assert (o.attributes["WignerSeitz.vacancy_count"], o.attributes["WignerSeitz.interstitial_count"]) == (3, 0)

# Results onto other atoms are refused.
try:
    ro.to_ovito(rmda.Frame.from_arrays(pos[:10], cell), d)
    raise SystemExit("results onto another particle count must be refused")
except ValueError:
    pass

print("ok")
"#;
    run_inline(
        WHAT,
        &module_dir,
        script,
        &[("data", repo_root().join("tests/data"))],
    );
}

/// The OVITO modifiers in a pipeline (the free `ovito` module runs them as
/// OVITO Pro's GUI does): on a BCC block with a 9-vacancy void, 2 single
/// vacancies and 2 interstitials, `DefectsModifier` gives the direct
/// `rmda.defects` census as attributes, tables and per-atom properties, and
/// with `output="defects"` turns the particles into the 13 defects; a
/// `SegmentModifier` upstream feeds it through the `Grain` property; the
/// reference-file slot gives the explicit-reference census; `VoidModifier`
/// finds the void. Skipped when `ovito` is not importable.
#[test]
fn ovito_modifiers() {
    const WHAT: &str = "ovito modifiers";
    let Some(module_dir) = bindings_or_skip(WHAT) else {
        return;
    };
    let script = r#"
import os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
try:
    from ovito.pipeline import Pipeline, StaticSource, ModifierInterface
except ImportError:
    sys.stderr.write("the ovito module is not importable")
    raise SystemExit(3)
import numpy as np
import rmda
import rmda.ovito as ro

assert "rmda._ovito_modifiers" not in sys.modules, "rmda.ovito must load the modifiers lazily"
from rmda.ovito import DefectsModifier, SegmentModifier, VoidModifier
for cls in (DefectsModifier, SegmentModifier, VoidModifier):
    assert issubclass(cls, ModifierInterface), cls

a = 3.165
g = np.array([[i, j, k] for i in range(12) for j in range(12) for k in range(12)], float)
lat = np.vstack([g, g + 0.5]) * a
centre = np.argsort(np.linalg.norm(lat - lat.mean(0), axis=1))[:9]
keep = np.ones(len(lat), bool)
keep[centre] = False
keep[[3, 1500]] = False
rng = np.random.default_rng(3)
pos = lat[keep] + rng.normal(0, 0.02, (keep.sum(), 3))
pos = np.vstack([pos, np.array([[0.5, 0, 0], [6.5, 6, 3]]) * a + [0.01, 0.02, 0.0]])
cell = np.eye(3) * 12 * a
f = rmda.Frame.from_arrays(pos, cell, types=["W"] * len(pos))
base = ro.to_ovito(f)
direct = rmda.defects(f, segment=True)
assert (direct["defects_vacancies"], direct["defects_interstitials"]) == (11, 2), direct

def run(*mods):
    p = Pipeline(source=StaticSource(data=base))
    for m in mods:
        p.modifiers.append(m)
    return p

m = DefectsModifier()
p = run(m)
o = p.compute()
for k, v in direct.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        assert o.attributes[k] == v, (k, o.attributes[k], v)
assert int(np.asarray(o.particles["Interstitial"]).sum()) == 2
assert np.array_equal(np.asarray(o.particles["atom_grain"]), direct["atom_grain"])
assert o.tables["rmda.defects.vacancies"].count == 11

m.output = "defects"
o = p.compute()
kinds = np.asarray(o.particles["Defect Type"])
assert o.particles.count == 13 and np.bincount(kinds).tolist() == [0, 11, 2], kinds
assert [t.name for t in o.particles["Defect Type"].types] == ["Vacancy", "Interstitial"]
assert o.cell is not None

o = run(SegmentModifier(min_size=50), DefectsModifier(segment=False, grain_column="Grain")).compute()
assert (o.attributes["defects_vacancies"], o.attributes["defects_interstitials"]) == (11, 2)
assert o.attributes["segment_grains"] == 1 and "Grain" in o.particles

ref = os.path.join(tempfile.mkdtemp(), "perfect.xyz")
rmda.Frame.from_arrays(lat, cell, types=["W"] * len(lat)).write(ref)
m = DefectsModifier(segment=False, type_coding="names")
m.ws_reference.load(ref)
o = run(m).compute()
assert (o.attributes["defects_vacancies"], o.attributes["defects_interstitials"]) == (11, 2)

o = run(VoidModifier()).compute()
assert o.attributes["voids"] == 1 and o.attributes["void_largest_sites"] == 9, dict(o.attributes)
assert o.attributes["void_isolated_vacancies"] == 2 and o.tables["rmda.voids"].count == 1

# A bad combination is an error the pipeline reports, not a wrong answer.
try:
    run(DefectsModifier(ws_affine="to-reference")).compute()
    raise SystemExit("ws_affine without a reference must be refused")
except Exception as e:
    assert "ws_reference" in str(e), e

print("ok")
"#;
    run_inline(WHAT, &module_dir, script, &[]);
}
