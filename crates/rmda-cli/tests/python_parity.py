#!/usr/bin/env python3
"""CLI-vs-Python parity for the `rmda` module.

The bindings are a thin wrapper over `rmda-core`, so the two surfaces must
produce the same numbers on the same frame; this script is the parity half of
that claim: for every analysis surface it runs the `rmda` subcommand, parses the
`key: value` stdout, calls the matching Python entry point on the same frame,
and compares

  * every scalar,
  * every histogram bin (one-line form for RDF, the sparse `  size=5
    clusters=3` listing for cluster),
  * the RDF bin edges, and
  * every per-atom label written to disk.

Per-atom labels matter: the aggregate counts a scalar comparison sees are just a
histogram of those labels, so labels permuted between the two surfaces would
leave every scalar identical while telling users opposite things about the same
atom.

It also checks that both surfaces refuse the same input with the same wording,
and that the module version matches the CLI's.

Exit status is non-zero on the first mismatch, which is what the Rust harness
(`crates/rmda-cli/tests/python_bindings.rs`) asserts on. A human can run it:

    cargo build --release -p rmda-py
    python3 crates/rmda-cli/tests/python_parity.py \
        --module-dir target/release --rmda target/release/rmda

or, against the abi3 wheel installed in the running interpreter:

    maturin build --release -m crates/rmda-py/Cargo.toml
    pip install target/wheels/rmda-*-abi3-*.whl
    python3 crates/rmda-cli/tests/python_parity.py --installed --rmda target/release/rmda
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "tests" / "data"

# `  size=5 clusters=3` under a bare `cluster_size_histogram:` header.
SPARSE_ENTRY = re.compile(r"^\s+size=(\d+) clusters=(\d+)$")


class Parity(Exception):
    """A failed CLI-vs-binding comparison; the message is multi-line."""


def load_module(module_dir: Path) -> None:
    """Put an importable `rmda` package on `sys.path`.

    `module_dir` is either a directory holding the staged `rmda/` package (as
    `python_bindings.rs` stages it) or a cargo target directory holding the
    built `librmda.so`, which is staged here as `rmda/_rmda.so` beside the
    package's `__init__.py` in a temp directory (rather than renaming the
    artifact under `target/`, where a later build would not refresh it).
    """
    module_dir = module_dir.resolve()
    if (module_dir / "rmda" / "__init__.py").is_file():
        sys.path.insert(0, str(module_dir))
        return
    staged = {
        "librmda.so": "_rmda.so",
        "librmda.dylib": "_rmda.so",
        "librmda.dll": "_rmda.pyd",
        "rmda.dll": "_rmda.pyd",
    }
    for src, dst in staged.items():
        if (module_dir / src).is_file():
            out = Path(tempfile.mkdtemp(prefix="rmda-parity-"))
            pkg = out / "rmda"
            pkg.mkdir()
            shutil.copy2(module_dir / src, pkg / dst)
            shutil.copy2(ROOT / "crates/rmda-py/python/rmda/__init__.py", pkg / "__init__.py")
            sys.path.insert(0, str(out))
            return
    raise SystemExit(
        f"no rmda extension module in {module_dir}; build it first:\n"
        "  cargo build --release -p rmda-py"
    )


def parse_stdout(text: str) -> tuple[dict, dict]:
    """CLI stdout -> (`dense` key -> values, `sparse` key -> {bin: value}).

    Lines whose values are not all numbers are skipped on purpose: stdout also
    carries human summaries such as `ptm_matched: 32/32 (100.00%) ...` and
    `ptm_lattice_constant: mean=3.615 min=3.615 ...`, which are prose about the
    numbers rather than the numbers themselves.
    """
    dense: dict[str, list[float]] = {}
    sparse: dict[str, dict[int, float]] = {}
    sparse_key: str | None = None
    for line in text.splitlines():
        if sparse_key is not None:
            m = SPARSE_ENTRY.match(line)
            if m:
                sparse[sparse_key][int(m.group(1))] = float(m.group(2))
                continue
            sparse_key = None
        if line.startswith("# r="):
            dense["edges"] = [float(v) for v in line[4:].split()]
            continue
        key, sep, rest = line.partition(": ")
        if not sep:
            # `cluster_size_histogram:` with nothing after the colon opens a
            # sparse block; every following indented `size=.. clusters=..` line
            # belongs to it until a line does not match.
            if line.endswith(":") and not line.startswith((" ", "#")):
                sparse_key = line[:-1]
                sparse.setdefault(sparse_key, {})
            continue
        try:
            vals = [float(tok) for tok in rest.split()]
        except ValueError:
            continue
        if vals:
            dense[key] = vals
    # A key that carries no values is dropped on both sides, so an empty
    # histogram is not reported as "the CLI printed it, Python did not".
    return {k: v for k, v in dense.items() if v}, sparse


def run_cli(rmda_bin: Path, args: list[str]) -> tuple[dict, dict]:
    proc = subprocess.run(
        [str(rmda_bin), *args], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise Parity(
            f"`rmda {' '.join(args)}` exited {proc.returncode}:\n{proc.stderr.strip()}"
        )
    return parse_stdout(proc.stdout)


def flatten(result: dict) -> dict:
    """The bindings return scalars alongside `histograms`/`edges`; the CLI
    prints everything flat, so flatten to the CLI's shape."""
    dense: dict[str, list[float]] = {}
    for key, val in result.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                dense[k2] = [float(x) for x in v2]
        elif isinstance(val, (list, tuple)):
            dense[key] = [float(x) for x in val]
        elif isinstance(val, (int, float)):
            dense[key] = [float(val)]
        else:
            raise Parity(f"binding returned an unsupported value for {key!r}: {val!r}")
    return {k: v for k, v in dense.items() if v}


def _close(x: float, y: float, kind: str) -> bool:
    # `cluster --knear` reports `cluster_mode_cutoff: nan`, and the binding
    # reports the same nan: two identical "not applicable" sentinels agree.
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if kind == "scalar":
        # Both sides print/hold the same f64, so this only absorbs formatting
        # differences; anything looser would hide a real disagreement.
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
    if kind == "edges":
        # The `# r=` line is written with five decimals.
        return math.isclose(x, y, rel_tol=0.0, abs_tol=2e-5)
    # Histograms are written with six decimals.
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6)


def compare(
    label: str, cli: tuple[dict, dict], py: dict, columns: tuple[str, ...] = ()
) -> None:
    """Raise `Parity` unless the CLI and the bindings agree in full."""
    cli_dense, cli_sparse = cli
    py = dict(py)
    problems: list[str] = []

    for key, bins in sorted(cli_sparse.items()):
        series = py.pop(key, None)
        if series is None:
            problems.append(f"{key}: the CLI printed it but the binding did not return it")
            continue
        for i, want in sorted(bins.items()):
            if i >= len(series):
                problems.append(
                    f"{key}[{i}]: CLI {want:g} but the binding has only {len(series)} bins"
                )
            elif series[i] != want:
                problems.append(f"{key}[{i}]: CLI {want:g} vs binding {series[i]:g}")
        for i, got in enumerate(series):
            if i not in bins and got != 0.0:
                problems.append(
                    f"{key}[{i}]: binding reports {got:g} where the CLI printed nothing"
                )

    for key in sorted(set(cli_dense) - set(py)):
        problems.append(f"{key}: the CLI printed it but the binding did not return it")
    for key in sorted(set(py) - set(cli_dense)):
        problems.append(f"{key}: the binding returned it but the CLI did not print it")

    for key in sorted(set(cli_dense) & set(py)):
        want, got = cli_dense[key], py[key]
        if len(want) != len(got):
            problems.append(f"{key}: {len(want)} values from the CLI, {len(got)} from Python")
            continue
        kind = (
            "edges"
            if key == "edges"
            else ("scalar" if len(want) == 1 and key not in cli_sparse else "series")
        )
        for i, (x, y) in enumerate(zip(want, got)):
            if not _close(x, y, kind):
                problems.append(f"{key}[{i}]: CLI {x!r} vs binding {y!r}")

    if problems:
        raise Parity(label + "\n  " + "\n  ".join(problems))


def read_xyz_column(path: Path, name: str) -> list[list[str]]:
    """One column of an extended-XYZ frame the CLI wrote, located through the
    `Properties=` spec so the right token is chosen whatever else was written
    next to it (`structure --full` adds six columns, `ptm` adds five)."""
    lines = path.read_text().splitlines()
    if len(lines) < 3 or "Properties=" not in lines[1]:
        raise Parity(f"{path.name}: not an extended-XYZ frame with a Properties spec")
    spec = lines[1].split("Properties=", 1)[1].split(" ")[0]
    parts = spec.split(":")
    if len(parts) % 3:
        raise Parity(f"{path.name}: malformed Properties={spec}")
    offset = 0
    span = None
    for i in range(0, len(parts), 3):
        count = int(parts[i + 2])
        if parts[i] == name:
            span = (offset, offset + count)
            break
        offset += count
    if span is None:
        raise Parity(f"{path.name}: no `{name}` column in Properties={spec}")
    lo, hi = span
    rows = []
    for row in lines[2:]:
        if not row.strip():
            continue
        toks = row.split()
        if len(toks) < hi:
            raise Parity(f"{path.name}: atom row has {len(toks)} fields, need {hi}")
        rows.append(toks[lo:hi])
    return rows


def compare_column(label: str, cli_vals: list[list[str]], py_vals) -> None:
    if len(cli_vals) != len(py_vals):
        raise Parity(
            f"{label}: {len(cli_vals)} atoms written by the CLI, {len(py_vals)} from the binding"
        )
    for i, (want, got) in enumerate(zip(cli_vals, py_vals)):
        if len(want) != 1:
            raise Parity(f"{label}: expected a single-column property, got {want}")
        x, y = float(want[0]), float(got)
        if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9):
            raise Parity(f"{label}: atom {i} differs (CLI {want[0]}, binding {got})")


def check_case(
    rmda,
    rmda_bin: Path,
    label: str,
    fixture: str,
    cli_args: list[str],
    call,
    columns: tuple[str, ...] = (),
) -> None:
    """Run one subcommand and its binding twin on the same frame."""
    rel = f"tests/data/{fixture}"
    args = list(cli_args)
    out: Path | None = None
    if columns:
        out = Path(tempfile.mkdtemp(prefix="rmda-parity-")) / "frame.xyz"
        args += ["-o", str(out)]
    cli = run_cli(rmda_bin, [*args, rel])

    f = rmda.read(str(DATA / fixture))
    py = flatten(call(rmda, f))
    compare(f"{label} ({rel})", cli, py)

    for col in columns:
        if out is None or not out.is_file():
            raise Parity(f"{label}: the CLI wrote no output file to compare {col} against")
        if col not in f:
            raise Parity(f"{label}: the binding did not attach a `{col}` property")
        compare_column(
            f"{label}: {col} ({rel})", read_xyz_column(out, col), f[col]
        )


# label, fixture, CLI args, binding call, per-atom properties to compare
CASES = [
    ("cna/adaptive", "fcc_ortho.xyz", ["structure"],
     lambda m, f: m.cna(f), ("StructureType",)),
    ("cna/fixed", "fcc_ortho.xyz", ["structure", "--mode", "fixed", "--cutoff", "3.5"],
     lambda m, f: m.cna(f, mode="fixed", cutoff=3.5), ("StructureType",)),
    ("cna/shell", "fcc_ortho.xyz",
     ["structure", "--mode", "shell", "--low", "2.0", "--cutoff", "3.0"],
     lambda m, f: m.cna(f, mode="shell", low=2.0, cutoff=3.0), ("StructureType",)),
    ("cna/interval", "m2_fcc_thermal.xyz", ["structure", "--mode", "interval"],
     lambda m, f: m.cna(f, mode="interval"), ("StructureType",)),
    ("cna/bond", "m2_fcc_thermal.xyz", ["structure", "--mode", "bond", "--cutoff", "3.0"],
     lambda m, f: m.cna(f, mode="bond", cutoff=3.0), ("StructureType",)),
    ("cna/diamond", "m2_fcc_thermal.xyz", ["structure", "--mode", "diamond"],
     lambda m, f: m.cna(f, mode="diamond"), ("StructureType",)),
    ("cna/full", "fcc_ortho.xyz", ["structure", "--full"],
     lambda m, f: m.cna(f, full=True), ("StructureType", "cna_n421")),
    ("cna/lammps-data", "triclinic.data", ["structure"],
     lambda m, f: m.cna(f), ("StructureType",)),
    ("cna/dump", "triclinic.dump", ["structure"],
     lambda m, f: m.cna(f), ("StructureType",)),
    ("cna/two-species", "triclinic_two_species.xyz", ["structure"],
     lambda m, f: m.cna(f), ("StructureType",)),
    # A thermally perturbed FCC cell: 31 atoms sit just under the default rmsd
    # cutoff and one falls over it, so a binding that quietly changed the
    # default would move a count rather than leave it untouched the way a
    # perfect lattice would.
    ("ptm/default", "m2_fcc_thermal.xyz", ["ptm"],
     lambda m, f: m.ptm(f), ("StructureType",)),
    ("ptm/rmsd", "m2_fcc_thermal.xyz", ["ptm", "--rmsd-cutoff", "0.3"],
     lambda m, f: m.ptm(f, rmsd_cutoff=0.3), ("StructureType",)),
    # Narrowing the tested structures changes both the counts and the labels
    # (an atom only OTHER when none of the selected templates fit it), so a
    # binding that silently ignored `types` would move them.
    ("ptm/types-subset", "m2_fcc_thermal.xyz", ["ptm", "--types", "fcc"],
     lambda m, f: m.ptm(f, types=["fcc"]), ("StructureType",)),
    # The two-shell structures and the Voronoi neighbour order.
    ("ptm/two-shell-voronoi", "m2_fcc_thermal.xyz",
     ["ptm", "--types", "fcc,hcp,bcc,dcub,dhex,graphene", "--ptm-order", "voronoi"],
     lambda m, f: m.ptm(f, types=["fcc", "hcp", "bcc", "dcub", "dhex", "graphene"],
                        order="voronoi"), ("StructureType", "StructureRmsd")),
    # Every opt-in output, correspondences and the upstream-only ones included.
    ("ptm/outputs-all", "m2_fcc_thermal.xyz",
     ["ptm", "--ptm-order", "voronoi", "--outputs", "all"],
     lambda m, f: m.ptm(f, order="voronoi", outputs=["all"]),
     ("StructureType", "StructureTemplate", "Correspondences")),
    ("rdf/partials", "triclinic_two_species.xyz", ["rdf", "--rmax", "2", "--dr", "0.1"],
     lambda m, f: m.rdf(f, rmax=2.0, dr=0.1), ()),
    ("rdf/no-partials", "fcc_ortho.xyz",
     ["rdf", "--rmax", "3", "--dr", "0.05", "--no-partials"],
     lambda m, f: m.rdf(f, rmax=3.0, dr=0.05, partials=False), ()),
    ("rdf/no-coordination", "fcc_ortho.xyz",
     ["rdf", "--rmax", "3", "--dr", "0.05", "--no-coordination"],
     lambda m, f: m.rdf(f, rmax=3.0, dr=0.05, coordination=False), ()),
    ("cluster/cutoff", "fcc_ortho.xyz", ["cluster", "--cutoff", "3.5"],
     lambda m, f: m.cluster(f, cutoff=3.5), ("Cluster", "ParticleCluster")),
    ("cluster/knear", "fcc_ortho.xyz", ["cluster", "--knear", "12"],
     lambda m, f: m.cluster(f, knear=12), ("Cluster", "ParticleCluster")),
    ("cluster/pairs", "fcc_ortho.xyz", ["cluster", "--pair-cutoff", "1-1=3.0"],
     lambda m, f: m.cluster(f, pair_cutoffs={(1, 1): 3.0}), ("Cluster", "ParticleCluster")),
    # The selection drops atom 4 (-1) and sorting moves the only dimer to id 0,
    # so a binding that ignored `select` or `sort_by_size` would move labels.
    ("cluster/select-sorted", "cluster_nonperiodic.xyz",
     ["cluster", "--cutoff", "1.5", "--select", "x > 0.8", "--sort-by-size"],
     lambda m, f: m.cluster(f, cutoff=1.5, select="x > 0.8", sort_by_size=True),
     ("Cluster", "ParticleCluster")),
    ("warren-cowley/one-species", "fcc_ortho.xyz", ["wc", "--wc-shells", "0,12"],
     lambda m, f: m.warren_cowley(f, shells="0,12"), ()),
    ("warren-cowley/two-species", "triclinic_two_species.xyz", ["wc", "--wc-shells", "0,4"],
     lambda m, f: m.warren_cowley(f, shells="0,4"), ()),
    # Periodic bulk excludes nothing, but the flag adds wc_atoms_surface and
    # wc_atoms_measured, so a binding that dropped `edge_correct` would differ.
    ("warren-cowley/edge-correct", "triclinic_two_species.xyz",
     ["wc", "--wc-shells", "0,4", "--edge-correct"],
     lambda m, f: m.warren_cowley(f, shells="0,4", edge_correct=True), ()),
]


def check_version(rmda, rmda_bin: Path) -> None:
    """`rmda.__version__` must be the version the CLI reports."""
    proc = subprocess.run(
        [str(rmda_bin), "--version"], cwd=ROOT, capture_output=True, text=True
    )
    cli_version = proc.stdout.split()[-1].strip()
    if rmda.__version__ != cli_version:
        raise Parity(
            f"the CLI reports {cli_version} but rmda.__version__ is {rmda.__version__}"
        )


def check_error_parity(rmda, rmda_bin: Path) -> None:
    """An rmax the CLI refuses must be refused by the binding, in the same
    words: both surfaces run `Rdf::validate`, so a divergence here means one of
    them stopped going through it."""
    needle = "exceeds half the smallest cell height"
    rel = "tests/data/fcc_ortho.xyz"
    proc = subprocess.run(
        [str(rmda_bin), "rdf", "--rmax", "50", rel], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode == 0:
        raise Parity("the CLI accepted an rmax beyond half the smallest cell height")
    try:
        rmda.rdf(rmda.read(str(DATA / "fcc_ortho.xyz")), rmax=50.0)
    except Exception as exc:  # noqa: BLE001 - any exception is fair game here
        if needle not in proc.stderr or needle not in str(exc):
            raise Parity(
                "the two surfaces refused the input for different reasons:\n"
                f"  CLI: {proc.stderr.strip()}\n  binding: {exc}"
            ) from exc
        return
    raise Parity("the binding accepted an rmax the CLI refused")


def dump_frame(rmda, fixture: Path) -> None:
    """Print the frame's fields as `key: value` lines for the Rust harness to
    compare against `rmda_core::io::read_frame`."""
    f = rmda.read(str(fixture))
    print(f"natoms: {f.natoms}")
    print(f"ntypes: {f.ntypes}")
    # `.tolist()`: the getters are numpy arrays, and numpy 2 spells a scalar's
    # repr `np.float64(...)`; Python floats print the shortest round-trip form.
    print("cell: " + " ".join(repr(v) for row in f.cell.tolist() for v in row))
    print("pos: " + " ".join(repr(v) for atom in f.positions.tolist() for v in atom))
    print("types: " + " ".join(str(v) for v in f.types.tolist()))
    print("ids: " + " ".join(str(v) for v in f.ids.tolist()))


def _fcc_block(a: float, n: int, tmp: Path, tag: str) -> Path:
    """A perfect conventional-cubic FCC block as extended-XYZ, as both the CLI
    `--reference` path and the `rmda.ws(f, reference=...)` binding read it."""
    half, box = a / 2.0, n * a
    sites = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                sites.append((i * a, j * a, k * a))
                sites.append((i * a + half, j * a + half, k * a))
                sites.append((i * a + half, j * a, k * a + half))
                sites.append((i * a, j * a + half, k * a + half))
    lines = [
        str(len(sites)),
        f'Lattice="{box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0 {box} 0.0 0.0 0.0" '
        'pbc="T T T" Properties=species:S:1:pos:R:3:type:I:1 Timestep=0',
    ]
    lines += [f"X {x:.6f} {y:.6f} {z:.6f} 1" for (x, y, z) in sites]
    p = tmp / tag
    p.write_text("\n".join(lines) + "\n")
    return p


def check_ws(rmda, rmda_bin: Path) -> None:
    """Reference-mode Wigner-Seitz: a frame against a copy of itself must
    report no vacancies/interstitials on both surfaces, with the same scalar
    set and values."""
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-"))
    ref = _fcc_block(3.5, 6, tmp, "ref.extxyz")
    cur = _fcc_block(3.5, 6, tmp, "cur.extxyz")
    cli = run_cli(rmda_bin, ["ws", "--reference", str(ref), str(cur)])
    f = rmda.read(str(cur))
    py = flatten(rmda.ws(f, reference=str(ref)))
    compare("ws self-vs-self", cli, py)


def _fcc_vacancy(a: float, n: int, tmp: Path, tag: str) -> Path:
    """`_fcc_block` with its last atom dropped: one vacant lattice site."""
    lines = _fcc_block(a, n, tmp, "perfect.extxyz").read_text().splitlines()
    lines = lines[:-1]
    lines[0] = str(len(lines) - 2)
    p = tmp / tag
    p.write_text("\n".join(lines) + "\n")
    return p


def check_ws_lattice_self(rmda, rmda_bin: Path) -> None:
    """"`rmda.ws(f, lattice=...)` and `rmda.ws(f, fit_self=True, ...)` drive the
    same lattice engines as `rmda ws --lattice`/`--self`. A perfect block has no
    defects; the same block minus one atom has exactly one vacancy on both
    surfaces."""
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-"))
    a = 3.5
    nn = a / 2.0**0.5
    perfect = _fcc_block(a, 6, tmp, "perfect.extxyz")
    vacant = _fcc_vacancy(a, 6, tmp, "vacant.extxyz")

    for label, args, pycall_f, vacant_sites in (
        (
            "ws lattice exact",
            ["ws", "--lattice", "fcc", "--nn-distance", f"{nn:.9f}", str(perfect)],
            lambda f: rmda.ws(f, lattice="fcc", nn_distance=nn),
            0,
        ),
        (
            "ws self-fit",
            ["ws", "--self", "--lattice", "fcc", "--nn-distance", f"{nn:.9f}", str(vacant)],
            lambda f: rmda.ws(f, fit_self=True, lattice="fcc", nn_distance=nn),
            1,
        ),
    ):
        cli = run_cli(rmda_bin, args)
        f = rmda.read(str(vacant if "vacant" in " ".join(args) else perfect))
        py = flatten(pycall_f(f))
        # Each engine's cluster assignment can differ in floating-point order, so
        # the per-site displacement is ~1e-9 on one side and ~1e-15 on the other
        # for an atom that sits on its site; both are "on site". Compare the
        # defect census strictly, the displacements loosely.
        cli_dense = dict(cli[0])
        for key in ("ws_max_displacement", "ws_mean_displacement"):
            py.pop(key, None)
            cli_dense.pop(key, None)
        compare(label, (cli_dense, cli[1]), py)
        if any(abs(float(v)) > 1e-6 for v in py.get("ws_displacements", [])):
            raise Parity(f"{label}: an atom sits more than 1e-6 A from its site")
        got = py.get("ws_vacancies", [0.0])
        if abs(float(got[0]) - vacant_sites) > 1e-12:
            raise Parity(f"{label}: {got} vacancies, expected {vacant_sites}")


def check_rdf_nbins(rmda, rmda_bin: Path) -> None:
    """`nbins` is the Python spelling of `rmax / dr`: the two spellings agree and
    both match the CLI's `--dr` invocation."""
    fixture = "m2_fcc_thermal.xyz"
    f = rmda.read(str(DATA / fixture))
    rmax = 3.0
    nbins = 24.0
    by_dr = rmda.rdf(f, rmax, dr=rmax / nbins)
    by_bins = rmda.rdf(f, rmax, nbins=nbins)
    if dict(by_dr) != dict(by_bins):
        raise Parity("rdf: nbins and rmax/dr disagree")

    rel = f"tests/data/{fixture}"
    cli = run_cli(rmda_bin, ["rdf", "--rmax", str(rmax), "--dr", str(rmax / nbins), rel])
    compare("rdf nbins", cli, flatten(by_bins))


def check_wc_shells(rmda, rmda_bin: Path) -> None:
    """`rmda.warren_cowley(f, shells=[0,12])` and `shells=12` render to the same
    comma text as the default and the CLI's `--wc-shells`."""
    fixture = "m2_fcc_thermal.xyz"
    f = rmda.read(str(DATA / fixture))
    cli = run_cli(rmda_bin, ["wc", "--wc-shells", "0,12", f"tests/data/{fixture}"])
    default = flatten(rmda.warren_cowley(f))
    single = flatten(rmda.warren_cowley(f, 12))
    listed = flatten(rmda.warren_cowley(f, [0, 12]))
    if default != single:
        raise Parity("warren_cowley: shells=12 differs from the default")
    if default != listed:
        raise Parity("warren_cowley: shells=[0,12] differs from the default")
    compare("warren_cowley shells", cli, listed)


def check_run_write_pipeline(rmda, rmda_bin: Path) -> None:
    """Design §22's ergonomics: `f.run("segment")` attaches `Grain`,
    `f.write` emits the attached props as extended-XYZ columns, and a
    `Pipeline` of `segment` over `iread` reproduces `rmda segment`."""
    fixture = "m2_fcc_thermal.xyz"
    rel = f"tests/data/{fixture}"
    f = rmda.read(str(DATA / fixture))
    f.run("segment")
    if "Grain" not in f:
        raise Parity("f.run('segment') attached no `Grain` property")

    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-")) / "out.xyz"
    f.write(str(tmp))
    compare_column(
        "f.write: StructureType", read_xyz_column(tmp, "StructureType"), f["StructureType"]
    )
    compare_column("f.write: Grain", read_xyz_column(tmp, "Grain"), f["Grain"])

    pipe = rmda.Pipeline(rmda.iread(str(DATA / fixture)))
    pipe.add("segment")
    frames = list(pipe.run())
    if len(frames) != 1:
        raise Parity(f"pipeline ran {len(frames)} frames, expected 1")
    cli = run_cli(rmda_bin, ["segment", rel])
    compare_scalars("pipeline segment", cli[0], frames[0].run("segment"))


def bcc_slab(
    n: int,
    nz: int,
    a: float,
    skip: tuple = (),
    extra: tuple = (),
    tilt: float = 0.0,
) -> str:
    """A BCC slab of `n`x`n`x`nz` cells under vacuum, as extxyz: the `slab`
    fixture of `tests/defects_cmd.rs`, plus `extra` atoms (Å, before the 5 Å
    lift) and, with `tilt`, its right half (x >= L/2) re-filled with the
    lattice rotated `tilt` degrees about z -- a bicrystal to segment."""
    rows = []
    half = n * a / 2.0
    c, s_ = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    for i in range(n):
        for j in range(n):
            for k in range(nz):
                for b, off in enumerate((0.0, 0.5)):
                    if (i, j, k, b) in skip:
                        continue
                    p = ((i + off) * a, (j + off) * a, (k + off) * a)
                    if tilt and p[0] >= half:
                        continue
                    rows.append(p)
    if tilt:
        # Rotated lattice points that land in the right half of the box.
        m = 2 * n
        for i in range(-m, m):
            for j in range(-m, m):
                for k in range(nz):
                    for off in (0.0, 0.5):
                        u, v = (i + off) * a, (j + off) * a
                        x, y = c * u - s_ * v + half, s_ * u + c * v
                        if half + 0.3 * a <= x < n * a - 0.3 * a and 0.0 <= y < n * a:
                            rows.append((x, y, (k + off) * a))
    rows.extend(extra)
    length, lz = n * a, 3.0 * nz * a
    body = "\n".join(f"W {x!r} {y!r} {z + 5.0!r}" for x, y, z in rows)
    return (
        f"{len(rows)}\n"
        f'Lattice="{length} 0 0 0 {length} 0 0 0 {lz}" pbc="T T T" '
        f"Properties=species:S:1:pos:R:3\n{body}\n"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, row.split(","))) for row in lines[1:]]


def same_value(want: str, got, decimals: int | None = None) -> bool:
    """One CLI table cell against the binding's value: strings exactly,
    numbers to the precision the CLI printed them with."""
    if isinstance(got, str):
        return want == got
    got = float(got)
    if want == "":
        # Not applicable: the CLI leaves the cell empty, the binding gives
        # -1 (integers) or nan (floats).
        return math.isnan(got) or got == -1
    try:
        x = float(want)
    except ValueError:
        return False
    if math.isnan(x) or math.isnan(got):
        return math.isnan(x) and math.isnan(got)
    tol = 0.5 * 10.0 ** -decimals + 1e-12 if decimals is not None else 1e-12
    return math.isclose(x, got, rel_tol=1e-12, abs_tol=tol)


def compare_rows(label: str, rows: list[dict[str, str]], cols: dict, decimals: dict) -> None:
    """CLI CSV rows against binding columns (name -> sequence)."""
    problems = []
    for name, series in cols.items():
        if len(series) != len(rows):
            problems.append(f"{name}: {len(rows)} CLI rows, {len(series)} binding values")
            continue
        for i, (row, got) in enumerate(zip(rows, series)):
            if not same_value(row[name], got, decimals.get(name)):
                problems.append(f"{name}[{i}]: CLI {row[name]} vs binding {got!r}")
    if problems:
        raise Parity(label + "\n  " + "\n  ".join(problems[:40]))


def compare_scalars(label: str, cli: dict, py: dict, extra: tuple = ()) -> None:
    """Every scalar the CLI printed, and no scalar the CLI did not print
    (save `extra`), at equal value."""
    mine = {k: v for k, v in py.items() if not isinstance(v, dict) and not hasattr(v, "shape")}
    for k in extra:
        mine.pop(k, None)
    compare(label, (cli, {}), {k: [float(v)] for k, v in mine.items()})


# The `--table` columns of `rmda defects`, and how many decimals each is printed with.
DEFECT_FLOATS = ("phase_agreement", "bond", "pair_fraction", "pair_rms",
                 "rms_displacement", "offlattice_fraction")


def defect_columns(t: dict) -> tuple[dict, dict]:
    cols = {k: v for k, v in t.items() if k not in ("snapped", "t", "origin", "status_detail")}
    cols["status"] = list(t["status"])
    cols["reference"] = list(t["reference"])
    for d, axis in enumerate("xyz"):
        cols[f"snapped_{axis}"] = [int(v) for v in t["snapped"][:, d]]
    cols["domain_converged"] = [int(v) for v in t["domain_converged"]]
    for r, a in enumerate("xyz"):
        for c, b in enumerate("xyz"):
            cols[f"t_{a}{b}"] = t["t"][:, r, c]
    decimals = {k: 6 for k in DEFECT_FLOATS}
    decimals.update({k: 6 for k in cols if k.startswith("t_")})
    return cols, decimals


def read_defect_xyz(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines()[2:] if ln.strip()]


def check_defect_files(label: str, out_dir: Path, res: dict) -> None:
    vac = read_defect_xyz(out_dir / "vacancies.xyz")
    v = res["vacancies"]
    compare_rows(
        f"{label}: vacancies.xyz",
        [dict(zip(("x", "y", "z", "grain", "boundary", "cluster"), r[1:])) for r in vac],
        {
            "x": v["position"][:, 0], "y": v["position"][:, 1], "z": v["position"][:, 2],
            "grain": v["grain"], "boundary": [int(b) for b in v["boundary"]],
            "cluster": v["cluster"],
        },
        {},
    )
    ins = read_defect_xyz(out_dir / "interstitials.xyz")
    it = res["interstitials"]
    compare_rows(
        f"{label}: interstitials.xyz",
        [dict(zip(("x", "y", "z", "id", "type", "grain", "boundary", "cluster"), r[1:]))
         for r in ins],
        {
            "x": it["position"][:, 0], "y": it["position"][:, 1], "z": it["position"][:, 2],
            "id": it["id"], "type": it["type"], "grain": it["grain"],
            "boundary": [int(b) for b in it["boundary"]], "cluster": it["cluster"],
        },
        {},
    )


def check_defects(rmda, rmda_bin: Path) -> None:
    """`rmda.defects` against `rmda defects` on the `tests/defects_cmd.rs`
    slabs: every stdout scalar, every `--table` cell, and every row of the
    `--defects` vacancy and interstitial files -- through regions, segmentation,
    the patch reference, an interstitial, and a grain column."""
    a = 3.165
    mid = f"z <= {5.0 + 8.0 * a}"
    hi = f"z > {5.0 + 8.0 * a}"
    cases = [
        ("regions", bcc_slab(10, 16, a, ((5, 5, 3, 0), (2, 7, 12, 1), (6, 3, 13, 0))),
         ["--region", mid, "--region", hi], {"regions": [mid, hi]}),
        ("segment", bcc_slab(10, 12, a, ((4, 4, 6, 0),)), ["--segment"], {"segment": True}),
        ("patch", bcc_slab(12, 12, a, ((5, 5, 5, 0), (7, 2, 8, 1))),
         ["--reference", "patch", "--patch-size", "12"],
         {"reference": "patch", "patch_size": 12.0}),
        ("interstitial", bcc_slab(10, 12, a, (), ((5.5 * a, 5.0 * a, 6.0 * a),)),
         ["--cluster-cutoff", "4.0", "--min-atoms", "50"],
         {"cluster_cutoff": 4.0, "min_atoms": 50}),
    ]
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-defects-"))
    for name, text, cli_args, kwargs in cases:
        label = f"defects/{name}"
        src = tmp / f"{name}.xyz"
        src.write_text(text)
        table, files = tmp / f"{name}.csv", tmp / f"{name}-defects"
        cli = run_cli(rmda_bin, ["defects", str(src), *cli_args,
                                "--table", str(table), "--defects", str(files)])
        f = rmda.read(str(src))
        res = rmda.defects(f, **kwargs)
        compare_scalars(label, cli[0], res, extra=("cluster_cutoff",))
        cols, decimals = defect_columns(res["grains"])
        rows = read_csv(table)
        cols["timestep"] = [0] * len(rows)
        compare_rows(f"{label}: --table", rows, cols, decimals)
        check_defect_files(label, files, res)
        # The same step through f.run / a Pipeline agrees with the call.
        again = f.run("defects", **kwargs)
        if {k: v for k, v in again.items() if isinstance(v, int)} != {
            k: v for k, v in res.items() if isinstance(v, int)
        }:
            raise Parity(f"{label}: f.run('defects') disagrees with rmda.defects")
        if len(res["atom_grain"]) != f.natoms or len(res["displacement"]) != f.natoms:
            raise Parity(f"{label}: per-atom arrays are not natoms long")
    if not any(
        rmda.defects(rmda.read(str(tmp / "interstitial.xyz")), **cases[3][3])[k] > 0
        for k in ("defects_interstitials", "defects_boundary_interstitials")
    ):
        raise Parity("defects/interstitial: the planted interstitial was not counted")

    # Explicit-reference WS: scalars, table and both defect files (with the
    # reference site ids), plain and with regions evaluated on the reference.
    ref = tmp / "ws-ref.xyz"
    ref.write_text(bcc_slab(10, 12, a))
    cur = tmp / "ws-cur.xyz"
    cur.write_text(bcc_slab(10, 12, a, ((5, 5, 3, 0), (3, 7, 4, 1)), ((5.5 * a + 0.2, 5.0 * a, 6.0 * a),)))
    for name, cli_args, kwargs in (
        ("ws", [], {}),
        ("ws-regions", ["--region", mid, "--region", hi], {"regions": [mid, hi]}),
    ):
        label = f"defects/{name}"
        table, files = tmp / f"{name}.csv", tmp / f"{name}-defects"
        cli = run_cli(rmda_bin, ["defects", str(cur), "--ws-reference", str(ref), *cli_args,
                                "--table", str(table), "--defects", str(files)])
        res = rmda.defects(rmda.read(str(cur)), ws_reference=str(ref), **kwargs)
        compare_scalars(label, cli[0], res, extra=("cluster_cutoff",))
        if (res["defects_vacancies"] + res["defects_boundary_vacancies"],
                res["defects_interstitials"] + res["defects_boundary_interstitials"]) != (2, 1):
            raise Parity(f"{label}: expected the 2 planted vacancies and 1 interstitial")
        cols, decimals = defect_columns(res["grains"])
        rows = read_csv(table)
        cols["timestep"] = [0] * len(rows)
        compare_rows(f"{label}: --table", rows, cols, decimals)
        check_defect_files(label, files, res)
        for kind in ("vacancies", "interstitials"):
            lines = (files / f"{kind}.xyz").read_text().splitlines()
            got = [int(l.split()[-1]) for l in lines[2:]]
            if got != [int(s) for s in res[kind]["site_id"]]:
                raise Parity(f"{label}: {kind} site_id differs: CLI {got}, binding {list(res[kind]['site_id'])}")
        again = rmda.read(str(cur)).run("defects", ws_reference=str(ref), **kwargs)
        if again["defects_vacancies"] != res["defects_vacancies"]:
            raise Parity(f"{label}: f.run('defects') disagrees with rmda.defects")

    # Antisites (`--antisites` / `antisites=True`): the W reference above and
    # the current frame with every 9th atom made Mo, under the explicit
    # reference (species matched by name across the frames), a global
    # fitted one (species per motif basis site by majority) and the local
    # census (a vote per region); each antisite's cluster, and the clusters
    # table's antisite counts.
    lines = cur.read_text().splitlines()
    body = [("Mo" + l[1:]) if i % 9 == 0 else l for i, l in enumerate(lines[2:])]
    mixed = tmp / "ws-mixed.xyz"
    mixed.write_text("\n".join(lines[:2] + body) + "\n")
    for name, cli_args, kwargs in (
        ("antisites-ws", ["--ws-reference", str(ref)], {"ws_reference": str(ref)}),
        ("antisites-global", ["--reference", "global"], {"reference": "global"}),
        ("antisites-local", ["--reference", "local"], {"reference": "local"}),
    ):
        label = f"defects/{name}"
        table, files = tmp / f"{name}.csv", tmp / f"{name}-defects"
        cli = run_cli(rmda_bin, ["defects", str(mixed), *cli_args, "--antisites",
                                "--table", str(table), "--defects", str(files)])
        res = rmda.defects(rmda.read(str(mixed)), antisites=True, **kwargs)
        compare_scalars(label, cli[0], res, extra=("cluster_cutoff",))
        if res["defects_antisites"] < 10 or res["defects_antisites_Mo_on_W"] != res["defects_antisites"]:
            raise Parity(f"{label}: expected the planted Mo antisites, got {res['defects_antisites']}")
        cols, decimals = defect_columns(res["grains"])
        rows = read_csv(table)
        cols["timestep"] = [0] * len(rows)
        compare_rows(f"{label}: --table", rows, cols, decimals)
        check_defect_files(label, files, res)
        an = (files / "antisites.xyz").read_text().splitlines()
        got = [(int(l.split()[4]), int(l.split()[5]), int(l.split()[6])) for l in an[2:]]
        want = list(zip((int(v) for v in res["antisites"]["id"]),
                        (int(v) for v in res["antisites"]["type"]),
                        (int(v) for v in res["antisites"]["site_type"])))
        if got != want:
            raise Parity(f"{label}: antisites.xyz differs: CLI {got[:5]}..., binding {want[:5]}...")
        got = [int(l.split()[-1]) for l in an[2:]]
        if got != [int(c) for c in res["antisites"]["cluster"]]:
            raise Parity(f"{label}: antisite clusters differ: CLI {got[:5]}..., binding "
                         f"{list(res['antisites']['cluster'][:5])}...")
        cl = res["clusters"]
        if (sum(int(a) for a in cl["antisites"]) != res["defects_antisites"]
                or sum(1 for s in cl["size"] if s == 0) != res["defects_antisite_clusters"]):
            raise Parity(f"{label}: clusters table antisites do not add up")

    # Grain labels from a column: the CLI reads `segment -o`'s Grain column
    # from the file, the binding the same file's column, and also the labels
    # `rmda.segment` attaches to a frame read from the original input.
    src = tmp / "segment.xyz"
    seg_out = tmp / "segment-out.xyz"
    run_cli(rmda_bin, ["segment", str(src), "-o", str(seg_out)])
    cli = run_cli(rmda_bin, ["defects", str(seg_out), "--grain-column", "Grain"])
    from_file = rmda.defects(rmda.read(str(seg_out)), grain_column="Grain")
    compare_scalars("defects/grain-column", cli[0], from_file, extra=("cluster_cutoff",))
    f = rmda.read(str(src))
    rmda.segment(f)
    attached = rmda.defects(f, grain_column="Grain")
    compare_scalars(
        "defects/grain-column (attached)",
        {k: [float(v)] for k, v in from_file.items() if isinstance(v, int)},
        attached,
        extra=("cluster_cutoff",),
    )

    # Both surfaces refuse a bad region, for the same reason.
    proc = subprocess.run(
        [str(rmda_bin), "defects", str(src), "--region", "q < 1"],
        capture_output=True, text=True,
    )
    try:
        rmda.defects(rmda.read(str(src)), regions=["q < 1"])
    except ValueError as exc:
        if proc.returncode == 0 or "unknown field" not in proc.stderr or "unknown field" not in str(exc):
            raise Parity(f"bad region refused differently:\n  CLI: {proc.stderr}\n  binding: {exc}")
    else:
        raise Parity("the binding accepted a region the CLI refused")


def check_segment(rmda, rmda_bin: Path) -> None:
    """`rmda.segment` against `rmda segment` on a tilted BCC bicrystal: the
    stdout scalars, every row of `--table` and its pair table, and the per-atom
    `Grain` column `-o` writes."""
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-segment-"))
    src = tmp / "bicrystal.xyz"
    src.write_text(bcc_slab(12, 8, 3.165, tilt=25.0))
    table, out = tmp / "seg.csv", tmp / "out.xyz"
    for label, cli_args, kwargs in (
        ("segment", [], {}),
        ("segment/no-adopt", ["--no-adopt", "--threshold", "3", "--min-size", "50"],
         {"adopt": False, "threshold": 3.0, "min_size": 50}),
        ("segment/auto", ["--mode", "auto"], {"mode": "auto"}),
    ):
        cli = run_cli(rmda_bin, ["segment", str(src), *cli_args,
                                "--table", str(table), "-o", str(out)])
        f = rmda.read(str(src))
        res = rmda.segment(f, **kwargs)
        compare_scalars(label, cli[0], res)
        if res["segment_grains"] < 2:
            raise Parity(f"{label}: a bicrystal segmented into {res['segment_grains']} grain(s)")
        g = res["grains"]
        q = g["orientation"]
        compare_rows(
            f"{label}: --table",
            read_csv(table),
            {"timestep": [0] * len(g["grain"]), "grain": g["grain"], "size": g["size"],
             "structure": g["structure"], "crystallites": g["crystallites"],
             "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3]},
            {"qw": 8, "qx": 8, "qy": 8, "qz": 8},
        )
        p = res["pairs"]
        compare_rows(
            f"{label}: pairs table",
            read_csv(table.with_suffix(".pairs.csv")),
            {"timestep": [0] * len(p["grain_a"]), "grain_a": p["grain_a"],
             "grain_b": p["grain_b"], "bonds": p["bonds"],
             "disorientation_deg": p["disorientation_deg"]},
            {"disorientation_deg": 4},
        )
        compare_column(f"{label}: Grain", read_xyz_column(out, "Grain"), f["Grain"])
        compare_column(
            f"{label}: StructureType", read_xyz_column(out, "StructureType"), f["StructureType"]
        )


def check_grain_boundaries(rmda, rmda_bin: Path) -> None:
    """`rmda.grain_boundaries` against `rmda grain-boundaries` on a tilted BCC
    bicrystal: every `gb_*` scalar, the per-atom `GrainBoundary` and `Grain`
    columns `-o` writes, and the `--grain-table` sizes and seam counts against
    the returned (and `f.tables`) `grains` table."""
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-gb-"))
    src = tmp / "bicrystal.xyz"
    src.write_text(bcc_slab(12, 8, 3.165, tilt=25.0))
    for label, cli_args, kwargs in (
        ("grain-boundaries", [], {}),
        ("grain-boundaries/options",
         ["--w-db", "6", "--unassigned-is-boundary", "--no-adopt-orphans",
          "--min-grain-size", "50"],
         {"w_db": 6.0, "unassigned_is_boundary": True, "no_adopt_orphans": True,
          "min_grain_size": 50}),
    ):
        out, table = tmp / "out.xyz", tmp / "grains.csv"
        cli = run_cli(rmda_bin, ["grain-boundaries", str(src), *cli_args,
                                "-o", str(out), "--grain-table", str(table)])
        f = rmda.read(str(src))
        res = rmda.grain_boundaries(f, **kwargs)
        compare_scalars(label, cli[0], res)
        if res["gb_atoms_boundary"] <= 0:
            raise Parity(f"{label}: a bicrystal has no boundary atoms")
        for col in ("GrainBoundary", "Grain"):
            compare_column(f"{label}: {col}", read_xyz_column(out, col), f[col])
        rows = [r for r in read_csv_commented(table)]
        g = res["grains"]
        compare_rows(f"{label}: --grain-table", rows,
                     {"grain": list(range(len(g["GrainSize"]))), "size": g["GrainSize"],
                      "boundary_atoms": g["GrainBoundaryCount"]}, {})
        tg = f.tables["grains"]
        if list(tg["GrainBoundaryCount"]) != list(g["GrainBoundaryCount"]):
            raise Parity(f"{label}: f.tables['grains'] differs from the returned table")
        if "GrainSize" in f:
            raise Parity(f"{label}: a per-grain property was attached as a per-atom column")
    try:
        rmda.grain_boundaries(rmda.read(str(src)), w_db=-1.0)
    except ValueError as exc:
        if "w_db must be positive and finite" not in str(exc):
            raise Parity(f"grain-boundaries: bad w_db refused for another reason: {exc}")
    else:
        raise Parity("grain-boundaries: the binding accepted w_db=-1")


def read_csv_commented(path: Path) -> list[dict[str, str]]:
    """`read_csv` for a table with `#` comment lines above its header."""
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    head = lines[0].split(",")
    return [dict(zip(head, row.split(","))) for row in lines[1:]]


VOID_PM = re.compile(r"^(void volume|extrapolated to phi = 1|excess over reference): "
                     r"(-?[0-9.]+) \+/- ([0-9.]+) A\^3")


def check_void(rmda, rmda_bin: Path) -> None:
    """`rmda.void` against `rmda void --experimental` (method B): the field
    figures and every volume +/- error line the CLI prints (to the decimals it
    prints), plain, extrapolated, strained, and as an excess over a reference
    frame; and without `experimental`, method A against `rmda void`."""
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-void-"))
    a = 3.5
    nn = a / 2.0**0.5
    perfect = _fcc_block(a, 5, tmp, "ref.extxyz")
    # The atom nearest the box centre removed: a vacancy inside the default
    # region (the box inset by one bond), where the integral sees it.
    lines = perfect.read_text().splitlines()
    centre = 2.5 * a
    drop = min(range(2, len(lines)), key=lambda i: sum(
        (float(v) - centre) ** 2 for v in lines[i].split()[1:4]))
    lines = lines[:drop] + lines[drop + 1:]
    lines[0] = str(len(lines) - 2)
    vacant = tmp / "vacant.extxyz"
    vacant.write_text("\n".join(lines) + "\n")
    nn_s = f"{nn:.9f}"
    for label, cli_args, kwargs in (
        ("void", [], {}),
        ("void/epsilons", ["--epsilons", "0.01,0.02,0.04", "--samples", "20000"],
         {"epsilons": [0.01, 0.02, 0.04], "samples": 20000}),
        ("void/region-seed", ["--region", "1,15,1,15,1,15", "--seed", "3", "--epsilon", "0.02"],
         {"region": "1,15,1,15,1,15", "seed": 3, "epsilon": 0.02}),
        ("void/strain", ["--strain", "--samples", "5000"], {"strain": True, "samples": 5000}),
        ("void/thermal-window", ["--thermal-window", "--samples", "20000"],
         {"thermal_window": True, "samples": 20000}),
        ("void/excess", ["--reference", str(perfect), "--excess", "--epsilons", "0.01,0.02,0.04"],
         {"reference": str(perfect), "excess": True, "epsilons": "0.01,0.02,0.04"}),
    ):
        proc = subprocess.run(
            [str(rmda_bin), "void", "--experimental", "--nn-distance", nn_s, *cli_args, str(vacant)],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise Parity(f"{label}: the CLI failed:\n{proc.stderr}")
        res = rmda.void(rmda.read(str(vacant)), experimental=True, nn_distance=nn, **kwargs)
        problems = []
        seen = 0
        for line in proc.stdout.splitlines():
            m = VOID_PM.match(line)
            if m:
                seen += 1
                key = {"void volume": "void",
                       "extrapolated to phi = 1":
                           "window" if kwargs.get("thermal_window") else "extrapolated",
                       "excess over reference": "excess"}[m.group(1)]
                for want, got in ((m.group(2), res[f"{key}_volume"]),
                                  (m.group(3), res[f"{key}_standard_error"])):
                    if f"{got:.4f}" != want:
                        problems.append(f"{key}: CLI {want} vs binding {got!r} ({line})")
            elif line.startswith("phi field:"):
                n, r = re.search(r"(\d+) face normal\(s\), circumradius ([0-9.]+) A", line).groups()
                if int(n) != res["face_normals"] or f"{res['circumradius']:.4f}" != r:
                    problems.append(f"field: {line} vs {res['face_normals']} {res['circumradius']}")
            elif line.startswith("observed nn:"):
                want = line.split()[2]
                if f"{res['observed_nn']:.4f}" != want:
                    problems.append(f"observed nn: CLI {want} vs binding {res['observed_nn']}")
            elif line.startswith("  epsilon "):
                pass
        want_lines = 1 + (("epsilons" in kwargs and not kwargs.get("excess"))
                          or bool(kwargs.get("thermal_window")))
        if seen != want_lines:
            problems.append(f"{seen} volume lines from the CLI, expected {want_lines}")
        if "epsilons" in kwargs or kwargs.get("thermal_window"):
            pre = ("excess" if kwargs.get("excess")
                   else "window" if kwargs.get("thermal_window") else "epsilon")
            rows = [l for l in proc.stdout.splitlines() if l.startswith("  epsilon ")]
            for row, v, se in zip(rows, res[f"{pre}_volumes"], res[f"{pre}_standard_errors"]):
                toks = row.split()
                if toks[2] != f"{v:.2f}" or toks[4] != f"{se:.2f}":
                    problems.append(f"per-epsilon: CLI {row!r} vs binding {v} +/- {se}")
        if problems:
            raise Parity(label + "\n  " + "\n  ".join(problems))
    if res["excess_volume"] <= 0 and rmda.void(rmda.read(str(vacant)), experimental=True,
                                              nn_distance=nn)["void_volume"] <= 0:
        raise Parity("void: a vacancy left no void volume at all")
    # Without experimental both are method A (design 17): the census's
    # vacancy clusters, with the same scalars and table.
    proc = subprocess.run([str(rmda_bin), "void", str(vacant), "--min-sites", "1"], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise Parity(f"void (method A): the CLI failed:\n{proc.stderr}")
    res = rmda.void(rmda.read(str(vacant)), min_sites=1)
    problems = []
    for line in proc.stdout.splitlines():
        key, _, val = line.partition(": ")
        if key not in res:
            problems.append(f"method A: {key} missing from the binding")
        elif isinstance(res[key], float):
            if f"{res[key]:.4f}" != val:
                problems.append(f"method A: {key} CLI {val} vs binding {res[key]!r}")
        elif str(res[key]) != val:
            problems.append(f"method A: {key} CLI {val} vs binding {res[key]!r}")
    if res["voids"] != 1 or list(res["voids_table"]["sites"]) != [1]:
        problems.append(f"method A: expected one 1-site void, got {res['voids_table']}")
    try:
        rmda.void(rmda.read(str(vacant)), samples=100)
    except ValueError as exc:
        if "experimental" not in str(exc):
            problems.append(f"method A: method B keyword refused oddly: {exc}")
    else:
        problems.append("method A: a method B keyword ran without experimental=True")
    if problems:
        raise Parity("\n  ".join(problems))


def check_unmeasured_warning(rmda, rmda_bin: Path) -> None:
    """A grain the census cannot measure is a `UserWarning` from `rmda.defects`
    and `rmda.void`, as it is a line on the CLI's stderr: with no grain measured
    the zero counts are not a defect-free crystal. A measured crystal is silent."""
    import warnings

    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-warn-"))
    small = rmda.read(str(_fcc_block(3.5, 2, tmp, "small.extxyz")))  # 32 atoms: too small
    big = rmda.read(str(_fcc_block(3.5, 5, tmp, "big.extxyz")))
    problems = []
    for name in ("defects", "void"):
        for frame, want in ((small, True), (big, False)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                getattr(rmda, name)(frame)
            got = any(issubclass(w.category, UserWarning) and "not measured" in str(w.message)
                      for w in caught)
            if got != want:
                problems.append(f"{name} on {frame.natoms} atoms: warned {got}, expected {want}")
    if problems:
        raise Parity("\n  ".join(problems))


def check_tracker(rmda, rmda_bin: Path) -> None:
    """`rmda.GrainTracker` passed to `rmda.defects`/`rmda.void` frame by frame
    gives the stable grain ids and events of `rmda defects/void FILES --out
    --segment` (a crystal that splits into a bicrystal and stays one)."""
    a = 3.165
    tmp = Path(tempfile.mkdtemp(prefix="rmda-parity-tracker-"))
    files = []
    for k, (tilt, skip) in enumerate([(0.0, ()), (20.0, ()), (20.0, ((2, 3, 4, 0),))]):
        p = tmp / f"frame.{(k + 1) * 10}.xyz"
        p.write_text(bcc_slab(12, 6, a, skip, tilt=tilt))
        files.append(p)

    def stable_rows(out: Path) -> list[tuple[str, str, str]]:
        return [(r["file_index"], r["grain"], r["stable_grain"])
                for r in read_csv(out / "grains.csv")]

    def event_rows(out: Path) -> list[tuple[str, str, str, str]]:
        return [(r["file_index"], r["event"], r["stable"], r["other"])
                for r in read_csv(out / "grain_events.csv")]

    for cmd, call in (("defects", rmda.defects), ("void", rmda.void)):
        out = tmp / f"{cmd}-series"
        extra = ["--site-reference", "global"] if cmd == "void" else ["--reference", "global"]
        run_cli(rmda_bin, [cmd, *map(str, files), "--segment", *extra, "--out", str(out)])
        kw = {"site_reference": "global"} if cmd == "void" else {"reference": "global"}
        tracker = rmda.GrainTracker()
        got_ids, got_events = [], []
        for p in files:
            k = p.stem.split(".")[-1]
            res = call(rmda.read(str(p)), segment=True, tracker=tracker, **kw)
            if cmd == "defects":
                g = res["grains"]
                got_ids += [(k, str(x), str(y)) for x, y in zip(g["grain"], g["stable_grain"])]
            ev = res["grain_events"]
            got_events += [(k, e, str(s_), str(o))
                           for e, s_, o in zip(ev["event"], ev["stable"], ev["other"])]
        want_events = event_rows(out)
        if got_events != want_events:
            raise Parity(f"{cmd} tracker: events {got_events} != CLI {want_events}")
        if not any(e[1] == "split" for e in want_events):
            raise Parity(f"{cmd} tracker: the fixture has no split: {want_events}")
        if cmd == "defects":
            want_ids = stable_rows(out)
            if got_ids != want_ids:
                raise Parity(f"defects tracker: ids {got_ids} != CLI {want_ids}")


def check_tables_and_generator(rmda, rmda_bin: Path) -> None:
    """`f.tables` carries the per-group tables of the calls run on a frame
    (copies, equal to what the calls returned), and `Pipeline.run` is a
    generator that runs the steps on each frame as it is read."""
    fixture = DATA / "m2_fcc_thermal.xyz"
    f = rmda.read(str(fixture))
    if f.tables != {}:
        raise Parity(f"a fresh frame has tables: {sorted(f.tables)}")
    rmda.segment(f, min_size=0)
    t = f.tables
    if sorted(t) != ["segment.grains", "segment.pairs"] or len(t["segment.grains"]["size"]) < 1:
        raise Parity(f"segment: f.tables is {sorted(t)}")
    if "size" in f.properties() or "orientation" in f.properties():
        raise Parity("segment: a per-grain column is attached as a per-atom one")
    res = rmda.defects(f, min_atoms=10)
    t = f.tables
    for name in ("grains", "vacancies", "interstitials", "clusters"):
        mine, theirs = t[f"defects.{name}"], res[name]
        if sorted(mine) != sorted(theirs):
            raise Parity(f"defects.{name}: columns differ")
        for col in theirs:
            if list(map(str, mine[col])) != list(map(str, theirs[col])):
                raise Parity(f"defects.{name}.{col}: f.tables differs from the result")
    t["defects.grains"]["n_atoms"][0] = -7
    if f.tables["defects.grains"]["n_atoms"][0] == -7:
        raise Parity("f.tables hands out the frame's own arrays")

    pipe = rmda.Pipeline(rmda.iread(str(fixture)))
    pipe.add("ptm")
    pipe.add("segment", {"min_size": 0})
    run = pipe.run()
    if isinstance(run, list) or iter(run) is not run:
        raise Parity(f"Pipeline.run returned {type(run).__name__}, not a generator")
    first = next(run)
    if "Grain" not in first or "segment.grains" not in first.tables:
        raise Parity("Pipeline.run yielded a frame without its steps' outputs")
    if list(run):
        raise Parity("Pipeline.run yielded more frames than the file holds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rmda", type=Path, required=True, help="path to the rmda binary")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--module-dir",
        type=Path,
        help="directory holding the built rmda extension (or librmda.so)",
    )
    where.add_argument(
        "--installed",
        action="store_true",
        help="test the `rmda` installed in this interpreter (e.g. from the maturin wheel)",
    )
    ap.add_argument(
        "--mode", choices=("parity", "frame"), default="parity",
        help="parity: run every CLI/binding comparison; frame: print one frame's fields",
    )
    ap.add_argument(
        "--fixture", type=Path,
        help="input file for --mode frame (relative paths are under the repo root)",
    )
    args = ap.parse_args(argv)

    if args.module_dir is not None:
        load_module(args.module_dir)
    import rmda  # noqa: E402 - must follow load_module

    if args.mode == "frame":
        if args.fixture is None:
            ap.error("--mode frame needs --fixture")
        fixture = args.fixture if args.fixture.is_absolute() else ROOT / args.fixture
        dump_frame(rmda, fixture)
        return 0

    failures: list[str] = []
    checks = 0

    for label, fixture, cli_args, call, columns in CASES:
        checks += 1
        try:
            check_case(rmda, args.rmda, label, fixture, cli_args, call, columns)
        except Parity as exc:
            failures.append(str(exc))
            print(f"FAIL {label}", file=sys.stderr)

    for name, fn in (
        ("version", check_version),
        ("error-parity", check_error_parity),
        ("ws", check_ws),
        ("ws lattice/self", check_ws_lattice_self),
        ("rdf nbins", check_rdf_nbins),
        ("wc shells", check_wc_shells),
        ("run/write/pipeline", check_run_write_pipeline),
        ("defects", check_defects),
        ("segment", check_segment),
        ("grain-boundaries", check_grain_boundaries),
        ("void", check_void),
        ("tables/generator", check_tables_and_generator),
        ("tracker", check_tracker),
        ("unmeasured-warning", check_unmeasured_warning),
    ):
        checks += 1
        try:
            fn(rmda, args.rmda)
        except Parity as exc:
            failures.append(str(exc))
            print(f"FAIL {name}", file=sys.stderr)

    if failures:
        print("\n\n".join(failures), file=sys.stderr)
        print(f"\n{len(failures)} of {checks} parity checks failed", file=sys.stderr)
        return 1
    print(f"parity ok: {checks} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
