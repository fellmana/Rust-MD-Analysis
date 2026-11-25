# RMDA — Rust MD Analysis

Structural analysis of atomistic molecular-dynamics frames, written in Rust: structure
identification (CNA, PTM), grain segmentation and tracking, per-grain vacancy and interstitial
counts (Wigner–Seitz against a fitted lattice), voids, clusters, RDF and Warren–Cowley order. It
comes as a multithreaded command line, `rmda`, and a Python module, `rmda`, and is built for frames
of 10⁶–10⁸ atoms on one node, from a laptop to an HPC compute node.

- **Manual**: [`documentation.md`](documentation.md): every command, its options and outputs, and
  the Python API.
- **Design**: [`design.md`](design.md): how each analysis works, the requirements it is held to,
  and the measurements that validate it.

## Install

From PyPI (Linux, CPython ≥ 3.9; one wheel serves every version):

```console
pip install rmda
rmda --version
python -c "import rmda; print(rmda.__version__)"
```

The wheel holds both the Python module and the `rmda` command line; it needs only numpy.

From source (Rust ≥ 1.72; no system libraries):

```console
git clone https://github.com/fellmana/Rust-MD-Analysis.git
cd Rust-MD-Analysis
cargo build --release              # the CLI: target/release/rmda
pip install .                      # the Python package and CLI (needs maturin, fetched by pip)
```

`rmda` uses rayon threads, not OpenMP: `--threads N`, where 0 (the default) means the job's CPUs
under SLURM (`SLURM_CPUS_PER_TASK`), else all cores. On HPE Cray systems `cc` is the Cray compiler
wrapper, which breaks Rust's link step: build with
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc` (or `module load PrgEnv-gnu`).

## Quick start

```console
rmda info frame.data                                   # what is in a file
rmda ptm frame.data -o typed.xyz                       # per-atom structure and orientation
rmda segment frame.data -o grains.xyz                  # grains
rmda defects frame.data --segment --table grains.csv --defects defects/
rmda defects frame-*.data --segment --out results/     # a time series, grains tracked across frames
```

```python
import rmda

f = rmda.read("frame.data")
rmda.set_threads(16)
res = rmda.defects(f, segment=True)
res["defects_vacancies"], res["grains"]["vacancies"]
```

Frames also come from arrays (`rmda.Frame.from_arrays`), ASE (`rmda.read(atoms)`,
`rmda.to_ase(f)`) and OVITO (`rmda.ovito.from_ovito(data)`, `rmda.ovito.to_ovito(f, data)`, usable
inside an OVITO pipeline). The package also registers OVITO modifiers (RMDA grain segmentation,
defect census, voids) that OVITO Pro lists once `rmda` is installed in its interpreter; see the user
guide, §5.

## Commands

| command | what it does |
|---|---|
| `info`, `index`, `convert` | summarise a file; write a frame index for fast seeking; convert formats |
| `structure`, `ptm` | per-atom structure type by CNA or PTM |
| `segment` | grain segmentation, with stacking-fault and twin merging, or OVITO's algorithm |
| `grain-boundaries` | mark grain-boundary atoms |
| `defects` | per-grain vacancies, interstitials and antisites, single files or time series |
| `void` | voids from the defect census |
| `ws` | Wigner–Seitz against a reference frame or lattice |
| `cluster`, `rdf`, `wc`, `census` | cluster analysis, RDF, Warren–Cowley order, coordination |
| `analyze` | several analyses in one pass over the frames |
| `run`, `merge`, `explain`, `bench` | a whole run from a TOML config; join the shards of a sharded run; what a config would do and the memory it needs; per-stage throughput |

Input: LAMMPS dump (text and binary), LAMMPS data, extended XYZ, any of them gzip-compressed. Every
frame is checked before analysis: a truncated or corrupted file is an error naming the atom, not a
wrong result.

## Per-grain defects

`rmda defects` fits a lattice per grain and counts vacancies and interstitials by nearest-site
occupancy; every count satisfies `vacancies − interstitials = sites − atoms`. Grains come from
regions (`--region 'z <= 500'`), an input column, or segmentation (`--segment`). The reference
lattice is chosen per grain (`--reference`):

| frame | run |
|---|---|
| single crystal in a periodic box that kept its atoms (cascade simulations) | `--reference periodic`: one lattice for the box, which also sees defects stored as whole atomic planes |
| polycrystal | `--segment` (`auto`: one lattice per grain, patches for bent grains, a local lattice per defect for grains damaged past both) |
| film on a substrate | `--region 'z <= Z' --region 'z > Z'` |
| a defect-free reference frame exists | `--ws-reference REF` (OVITO's Wigner–Seitz) |

`documentation.md` §3.8 explains each reference and its known limits.

Large series run as SLURM array jobs: `rmda run config.toml --shard I/N` per task, then
`rmda merge results` (`examples/` has configs).

## Tests

```console
cargo test --workspace                              # the fast suite: a few minutes
cargo test --workspace --release -- --include-ignored   # also the expensive validation (≈ 1 h)
```

The expensive tests (multi-million-atom ground-truth blocks, OVITO reference comparisons) are
`#[ignore]`d by default. A few validate against research datasets that are not distributed
(`RMDA_REFERENCE_DIR`, `RMDA_CASCADE_DIR`); they skip with a message when those are unset. The
Python parity tests run when the extension module is built (`cargo build -p rmda-py`).

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Parts derive from PTM and follow OVITO's algorithms; see
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
