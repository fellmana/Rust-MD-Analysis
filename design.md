# RMDA (Rust MD Analysis) — Design Document

**RMDA (Rust MD Analysis)** is a self-contained Rust code for analysing atomistic molecular-dynamics output,
primarily from LAMMPS. It is a command-line tool (`rmda`) and a Python module (`rmda`). It runs on one core
or many, on laptops and on HPC clusters. It is built for systems of 10⁸–10⁹ atoms. The
flagship case is a 128M-atom polycrystal with grains of different lattices, and thin films with free
surfaces, analysed as a time series over many files.

### How to read this document

- **This is the single design document.** It says what the code does and must keep doing, and why.
  Where the code does not yet do what a section specifies, the gap is listed in that section as an
  **Open:** item.
- **Measured evidence** (wall times, memory, OVITO agreement, validation counts) sits in the section it
  validates. Unless stated otherwise, measurements are on a 12-thread i5-1235U laptop against OVITO 3.15,
  and "the thin film" is the 5,759,994-atom BCC W substrate + BCC HEA film (box 192 × 192 × 2660 Å,
  792 MB per frame; `data.firstframe` is its frame before the cascades, `data.lastframe-999` its last).
  "The 128M frame" is a 128,000,000-atom, three-grain W polycrystal after cascades. These datasets, the
  cascade frames and the `reference_structures/` snapshots cited below are research data, not
  distributed with the code; the tests that use them skip without them (§25).
- "Must" is a requirement with a test (§25). "Should" is a default that may be overridden with an ADR.

---

## Table of contents

1. Goals, requirements, acceptance targets
2. Principles
3. Workspace layout
4. Dependencies and toolchain
5. Types and conventions
6. Cell and boundaries
7. Frame and data model
8. I/O
9. Neighbour subsystem
10. Execution model and memory discipline
11. Structure registry and orientation mathematics
12. Analysis framework and pipeline
13. CNA
14. PTM
15. Grain segmentation and tracking
16. Defect subsystem (Wigner–Seitz family)
17. Void detection
18. Warren–Cowley, RDF, cluster analysis
19. Workflows: polycrystal and thin film
20. Run model and time series
21. CLI
22. Python bindings
23. Profiling and resource reporting
24. Performance and memory budgets
25. Testing and validation
26. HPC deployment
27. Status
28. Future work
29. Risks and open questions
30. Appendices
31. References

---

## 1. Goals, requirements, acceptance targets

### 1.1 Requirements

| # | Requirement | Design response | Acceptance (tested, §25) |
|---|---|---|---|
| R1 | Single- and multi-core | One rayon pool; every O(N) pass is parallel, including parsing, writing, grain edges and clustering (§10). | Parallel efficiency ≥ 70 % at 16 threads for CNA, PTM and read. Results bit-identical for 1…N threads. |
| R2 | Easy install, sparse dependencies | Pure Rust, 3 runtime crates in the core, no system libraries, MSRV pinned (§4). | `cargo build --release --offline` from a vendored tree on a cluster without network access. |
| R3 | Self-contained | No runtime use of LAMMPS, OVITO, ASE or Python by the CLI. | — |
| R4 | Fast, complete LAMMPS and xyz I/O | Parallel column-plan parser; all LAMMPS position kinds, box forms and boundary flags; binary dump; streaming gzip (§8). | Read ≥ 1.5 GB/s on 8 threads. Round-trip parity with LAMMPS, ASE and OVITO, including triclinic with negative tilts and `pp pp ss`. |
| R5 | Intuitive CLI + Python | One pipeline model behind `rmda run`, presets and Python; one option namespace; numpy zero-copy (§20–§22). | Python == CLI on every analysis. No 128M-size Python objects. |
| R6 | Walltime and memory monitored and reported | `RunReport` per run, stage and frame: wall, CPU, anonymous RSS, file RSS. Pre-run memory estimate (§23). | Report present in text and JSON. Estimate within ±25 % of measured. |
| R7 | Clean, extensible | `StructureRegistry`, a typed pipeline (requires/provides), unified WS core, no module > 1500 lines (§11, §12, §16). | Adding a structure or analysis touches only its own files plus a registration line. |
| R8 | Profilable | Stage spans (never per-atom), Chrome-trace output, a `profiling` cargo profile with symbols, `rmda bench`. | — |
| R9 | Tests per module | Unit, property, oracle (OVITO, ASE), synthetic-defect and scale tests; CI performance gates (§25). | Every module has tests. Every claimed parity has an oracle test. |

### 1.2 Analysis scope

| Group | Analyses |
|---|---|
| Structure identification | CNA (adaptive, interval, fixed, bond; diamond) §13; PTM (all OVITO structures, orientations, ordering, deformation gradient) §14 |
| Defects | WS with an explicit reference (OVITO) §16.3a; WS with a lattice reference fitted to the frame (Nordlund & Averback; Mason) §16.3b; per-grain patch-local WS (RMDA, new) §16.3c; voids (Mason) §17 |
| Lattices | FCC, BCC, HCP; dHCP (§28.1), diamond and others through the registry; mixed-lattice polycrystals (§11, §15, §16) |
| Misc | Warren–Cowley, cluster analysis, RDF (§18); grain segmentation and tracking (§15) |
| Formats | LAMMPS dump (text, binary, gz), LAMMPS data, XYZ, extended XYZ (ASE- and OVITO-compatible) |

### 1.3 Non-goals

- Force fields, minimisation, dynamics.
- Rendering or GUI.
- DXA (dislocation analysis); the framework must allow it later.
- GPU (see §28.2).
- Multi-node MPI. The design keeps every per-atom analysis spatially local so that domain decomposition
  can be added later (§26.4).

---

## 2. Principles

1. **Correct fundamentals first.** Cell conventions, boundaries, neighbour rules, orientation algebra and
   occupancy semantics are each defined once, in one module, with property tests. Wrong results
   come from a fundamental defined twice or defined loosely.
2. **Invariants, not heuristics.** Every analysis states its invariants and the test asserts them.
   Examples:
   - WS: `V − I = N_sites − N_atoms` per region class.
   - Grains: a single crystal at any orientation is one grain.
   - All analyses: atom-order permutation invariance and thread-count determinism.

   A heuristic that cannot be stated as an invariant needs an ADR.
3. **Refuse, don't fabricate.** When a quality gate fails, the output is "unmeasured" with a reason code.
   It is never a number. Quality metrics are always written next to results, so a disabled gate is
   visible in the output.
4. **Stream and tile.** At most one frame is resident. Per-atom work runs in spatial tiles with
   thread-local scratch, so no transient scales as N × (bytes per neighbour). Per-atom outputs are
   written into preallocated columns.
5. **One neighbour kernel, fused consumers.** Analyses that need local neighbourhoods (CNA, PTM, WC
   shells) consume the same k-NN gather in one fused pass.
6. **Registries for what grows.** Structures, formats and analyses are data registered once. Analysis
   code never matches on structure codes.
7. **Cost is explicit.** Every pipeline prints a pre-run memory estimate and refuses to start past
   `--mem-limit`. Every run ends with a `RunReport`.
8. **Specification lives here; history does not.** Module doc comments state invariants and cite this
   document's section numbers. Narrative goes to ADRs.

---

## 3. Workspace layout

```
Rust-MD-Analysis/
├── Cargo.toml                 # workspace; rust-version 1.72; default-members = core + cli
├── pyproject.toml             # the `rmda` Python package (maturin), §22
├── crates/
│   ├── rmda-core/src/
│   │   ├── lib.rs  error.rs  units.rs  graph.rs  prof.rs
│   │   ├── cell.rs, cell/minkowski.rs
│   │   ├── data/              # frame.rs, particle.rs (SoA), property.rs, select.rs (region expressions)
│   │   ├── io/                # engine.rs (mmap, scan, parallel rows, numparse), lammps.rs, binary.rs,
│   │   │                      # xyz.rs, stream.rs + gzpar.rs (gzip), index.rs (.rmdaidx), format.rs,
│   │   │                      # writer.rs, legacy.rs (test-only serial reference reader)
│   │   ├── neighbor/          # knn.rs (KnnIndex, §9.1); cell_list.rs + nnearest.rs (within-r queries)
│   │   ├── exec/              # pool.rs (Par, fill/try_fill_with, set_threads), checkpoints.rs
│   │   ├── lattice/           # site lattices (fcc bcc hcp dhcp sc dia), sites.rs, fit.rs (rigid fit, `ws --self`)
│   │   └── analysis/
│   │       ├── cna.rs  ptm.rs  ptm/          # PTM kernel, generated tables, voronoi.rs (Voronoi neighbour order)
│   │       ├── orient.rs                     # orientation algebra (§11.3)
│   │       ├── grain_seg.rs  grain_auto.rs  grain_track.rs  stacking.rs   # segmentation (§15, §28.1)
│   │       ├── defects/       # mod.rs, fit.rs, lattice.rs (Motif, AffineLattice), patch.rs, local.rs,
│   │       │                  # census.rs, clusters.rs, antisite.rs, voids.rs, frame.rs,
│   │       │                  # explicit.rs, explicit_ws.rs (§16)
│   │       ├── void_b.rs roto_strain.rs rdf.rs warren_cowley.rs cluster.rs neighbor_census.rs occupancy.rs
│   │       ├── ws.rs void.rs defect_dump.rs  # `rmda ws` (single-lattice WS and its voids)
│   │       └── grains.rs grain_boundaries.rs grain_table.rs   # `rmda grain-boundaries`
│   ├── rmda-cli/src/           # lib.rs (entry point), cli.rs (arguments), cmd_*.rs (one per command)
│   └── rmda-py/               # src/lib.rs (bindings), python/rmda/__init__.py (package)
├── tools/                     # fixture generators (ASE/OVITO/LAMMPS), OVITO cross-checks, PTM table generators
└── tests/data/
```

Rules:

- `rmda-core` has no knowledge of the CLI or Python.
- No source file exceeds about 1500 lines, excluding generated tables.
- Generated PTM tables stay generated (`tools/gen_ptm_*`) and are checked by a test against the generator
  output.

**Open:**

- Target layout: `structure/` (registry, descriptors, `orient.rs`), `pipeline/` (keys, trait, plan,
  config, report), `analysis/{cna,ptm,grains,defects,void}/` split by stage, `prof/` (§12, §21).
- Files over the limit: `rmda-py/src/lib.rs` (4078 lines), `defects/local.rs` (3244), `io/lammps.rs`
  (3124), `grains.rs` (2823), `lattice/fit.rs` (2465), `defects/mod.rs` (2134), `void_b.rs` (2074),
  `cna.rs` (2072), and a handful between 1500 and 2000.
- Two Wigner–Seitz paths exist side by side: the per-grain census of §16 (`rmda defects`, `rmda void`)
  and the single-lattice `ws.rs` + `lattice/fit.rs` self-reference behind `rmda ws --self`
  (with `void.rs` for its `--void-min-sites`). `rmda ws --reference` already runs on
  `defects/explicit.rs` (§16.3a). Likewise `grains.rs` (bond-graph labelling) survives only inside
  `rmda grain-boundaries`.
- `docs/decisions/` (ADRs) does not exist yet.

---

## 4. Dependencies and toolchain

| Scope | Crates | Rationale |
|---|---|---|
| core (always) | `rayon`, `memmap2`, `flate2` (rust backend) | Parallelism, mmap fast path, gzip. All pure Rust. |
| cli | `clap` (derive), pinned `=4.4.18` | The UX is worth one crate; the pin keeps MSRV 1.72. |
| py (`-p rmda-py`) | `pyo3 0.25` (Bound API), `numpy 0.25` (rust-numpy) | Non-deprecated API, zero-copy arrays. Builds and imports under Python 3.10 (numpy 1.26) and 3.12–3.14 (numpy 2.5). |
| dev only | none required; `proptest` allowed | Dev dependencies do not affect user installs. |
| explicitly avoided | `serde`, `nalgebra`/`ndarray`, `regex`, `tokio` | A hand-written JSON writer, a TOML-subset config reader, own 3×3 and quaternion math. |

- `rust-version = "1.72"` is pinned in `[workspace.package]`, and CI builds, lints and tests with 1.72.0.
  MSRV is raised deliberately, not by accident.
- pyo3 0.25 is the newest release whose MSRV fits 1.72 (0.26 needs 1.74). `Cargo.lock` holds back two
  transitive crates for the same reason: `matrixmultiply 0.3.9` (0.3.11 needs 1.75) and
  `rustc-hash 2.0.0` (2.1 needs 1.77). pyo3 0.25 needs Python ≥ 3.7.
- `default-members = [rmda-core, rmda-cli]`: a plain `cargo build` makes the `rmda` binary without Python.
  The bindings build with `-p rmda-py` or `--workspace`.
- Release profile: `lto = true`, `codegen-units = 1`, `opt-level = 3`.

**Open:** vendoring (`cargo vendor` plus `.cargo/config.toml`) for offline HPC builds is not yet
documented or tested.

---

## 5. Types and conventions

| Concept | Rust | Notes |
|---|---|---|
| atom index (local) | `u32` (`AtomIdx`) | Checked at frame construction: N < 2³² − 1. |
| atom id (LAMMPS `tagint`) | `i64` | Always 64-bit (`-DLAMMPS_BIGBIG` safe). |
| type | `u32` (`MdaType`, 1-based) | Plus type labels and masses per type. |
| image flags | `i64` (`MdaImage`) | Packed three ways, as LAMMPS' `imageint`; unpacked `[i32; 3]` is open (§27.2). |
| counts, timesteps | `u64` / `i64` | — |
| positions, cell | `f64` | — |
| derived per-atom floats | `f32` by default | RMSD, scale, quaternions, displacements. `f64` only where an ADR justifies it. |
| structure id | `StructureId(u8)` | From the registry (§11). |
| grain / cluster ids | `u32`, `NONE = u32::MAX` | Exported as `i64` with −1 for none. One representation internally. |

**Units.** Internally lengths are in the file's length unit (Å for LAMMPS `metal`, `real`). An `ITEM:
UNITS` header (read into `meta.units`) records the style (a `--units` flag is open, §21). Velocities and forces are
opaque properties. RMDA never converts silently.

**Quaternions** are `[w, x, y, z]`, unit norm, `w ≥ 0` after folding (§11.3).

**Open:**

- Image flags are still `MdaImage = i64` packed 3 × 10 bits (`cell.rs`), so values beyond ±512 are
  silently masked, and images implied by `xu` columns are discarded.
- `Property::push` routes every element through f64, so i64 values above 2⁵³ lose precision;
  `Frame::ntypes()` is O(N) (max over types); `natoms` is duplicated state.

---

## 6. Cell and boundaries

### 6.1 `Cell`

```rust
pub enum Boundary { Periodic, Fixed, Shrink, ShrinkMin }   // LAMMPS p, f, s, m
pub struct Cell {
    h: Mat3,              // rows a, b, c (ASE/LAMMPS H-matrix convention)
    origin: [f64; 3],     // LAMMPS boxlo
    boundary: [Boundary; 3],
}
impl Cell {
    fn is_periodic(&self, d: usize) -> bool;   // boundary[d] == Periodic
    fn frac(&self, x) -> [f64;3]; fn cart(&self, s) -> [f64;3];  // cached inverse
    fn wrap(&self, x) -> ([f64;3], [i32;3]);    // LAMMPS Domain::remap semantics
    fn unwrap(&self, x, img) -> [f64;3];         // full h, all 9 terms
    fn bbox(&self) -> ([f64;3],[f64;3]);         // LAMMPS bboxlo/hi with min/max of tilts
    fn mic(&self) -> Mic;                        // prebuilt minimum-image engine (§6.3)
}
```

- The inverse, the face heights (`volume / |b×c|` etc.), the orthogonality flag and Smith's MIC bound are
  computed once at construction and cached. They are never recomputed per query.
- `Cell` is immutable. Mutation creates a new `Cell`.

**Open:** the code's `UnitCell` holds `pbc: [bool; 3]`, not the per-axis `Boundary` kind, so `f`, `s` and
`m` all read as non-periodic and are written back as `ff`. `unwrap()` ignores the upper triangle.
`UnitCell::bounds()` is the bounding box of the cell's eight corners (`UnitCell::corners()`), which for a
restricted cell is LAMMPS' `boxlo_bound`/`boxhi_bound` and for an orthorhombic one the plain box
(not `(origin, origin + a + b + c)`, which is wrong for a tilted cell). Its consumers: `rmda void` method B's default region and the fit's atom span. The site lattice
(`SiteLattice::build`) maps the cell's own corners, not the bounding box's, into lattice coordinates.

### 6.2 `cell::lammps_box` — the only place LAMMPS box forms are converted

| Form | Read | Write |
|---|---|---|
| data file `xlo xhi / ylo yhi / zlo zhi / xy xz yz` | `a=(xhi−xlo,0,0)`, `b=(xy, yhi−ylo, 0)`, `c=(xz, yz, zhi−zlo)`, `origin=(xlo,ylo,zlo)` | Inverse. Requires a restricted (lower-triangular) cell, otherwise rotate first (below). |
| dump `ITEM: BOX BOUNDS xy xz yz` (rows `xlo_bound xhi_bound xy` …) | `xlo = xlo_b − min(0,xy,xz,xy+xz)`, `xhi = xhi_b − max(0,xy,xz,xy+xz)`, `ylo = ylo_b − min(0,yz)`, `yhi = yhi_b − max(0,yz)`, then as data | Inverse (`dump_custom.cpp` `header_item_triclinic`). |
| dump `ITEM: BOX BOUNDS abc origin` / data `avec bvec cvec abc origin` / binary `triclinic 2` (general triclinic) | Rotate to restricted with LAMMPS `general_to_restricted_rotation` about the origin; rotate positions and the `vx vy vz`, `fx fy fz`, `mux muy muz`, `omega*`, `angmom*` triples with it (`lammps::rotate_vector_props`). The rotation is not kept. | A cell that is not restricted (`UnitCell::is_restricted`: upper triangle zero, positive diagonal; e.g. a rotated extxyz `Lattice`) is written in the general form, atoms unrotated (`UnitCell::lammps_general`); a left-handed cell is an error. A restricted cell is written in the restricted form. |
| boundary tokens `pp ff ss mm fs sm …` | Per dimension: periodic iff the token is `pp`. A token that mixes `p` with a non-periodic letter, or a count other than 3, is an `Err`. `ss` and `mm` are **non-periodic**. | Written from `boundary`. |

- The dump bounding-box rows are never passed straight to the data-file constructor as lo/hi; doing so
  inflates `lx`/`ly` and shifts the origin of every tilted dump, and a mirrored writer error hides it in
  a self round trip. The test fixture `tests/data/triclinic_tilted.dump` comes from
  `tools/gen_tilted_dump.py` (LAMMPS formulas, checked with ASE).
- **Tilt reduction** (`UnitCell::reduce_tilts`, `Frame::reduce_tilts`, `convert --reduce-tilt`): the
  `fix deform flip` rule (`fix_deform.cpp`) -- `c ∓ b` when `yz·(1/ly)` is beyond ±0.5 (moving `xz` by
  `∓xy`), then `c ∓ a` for `xz` and `b ∓ a` for `xy` -- repeated until every tilt is in range, along
  periodic axes only. The new rows are `M·h` with `M` unit lower-triangular; atoms are wrapped into the
  new cell in fractional space (an atom already inside keeps its bits) and image flags become
  `n·M⁻¹ + s`, so `x + n·h` does not move. Opt-in, also for data files (LAMMPS since 2023 accepts large
  tilts with a warning); `convert` to a data file prints a note when a tilt is over half a period.
- Tests are against real LAMMPS output: negative tilts, all three tilts nonzero, `pp pp ss`, `fs`, and
  general triclinic. ASE `lammpsrun` and OVITO must read RMDA-written files identically.

Checked: a rotated cell written by `convert` as dump and data reads in OVITO 3.15 with the identical cell,
origin, fractional coordinates (3e-16) and velocities; RMDA reads all three LAMMPS forms back as the
restricted rotation (`a_general_cell_is_written_in_the_general_triclinic_form`).

- **`--keep-general`** (`ReadOptions::keep_general`): a general-triclinic box is read as given
  (`UnitCell::general_as_given`: rows `a b c`, the file's origin, validated like the rotated form), with
  no rotation of positions or vector columns (`DumpHeader::apply_keep_general` for text and binary dumps;
  the data reader directly). Such a cell is not restricted, so every writer emits the general form again;
  at full precision a second pass is byte-identical, positions and velocities bit-identical
  (`keep_general_reads_the_general_form_as_written`). Restricted input is unaffected. The rotation is
  never stored, so the option acts at read time; a frame already rotated cannot be un-rotated.

**Open:** `tools/gen_lammps_fixtures.in` (real LAMMPS fixtures for all forms, generated by LAMMPS itself).

### 6.3 Minimum image

- Orthogonal cells: per-axis rounding.
- General cells: fractional reduction, then Smith's criterion. If that fails, a Minkowski-reduced
  general path (`cell/minkowski.rs`).
- Non-periodic axes are never folded.
- The `Mic` object holds all precomputed data and is `Copy`.

---

## 7. Frame and data model

### 7.1 `Frame`

```rust
pub struct Frame {
    pub cell: Cell,
    pub atoms: Atoms,            // mandatory core arrays, SoA
    pub props: PropertyStore,    // per-atom columns (inputs + analysis outputs)
    pub tables: TableStore,      // per-group tables (grains, clusters, defects, partial RDFs)
    pub scalars: ScalarStore,    // per-frame scalars (counts, fit quality, timings)
    pub meta: FrameMeta,         // source path, file index, frame index, timestep, time, units, rotation
}
pub struct Atoms {
    pos: Vec<[f64; 3]>,          // wrapped into the cell on periodic axes
    id: Vec<i64>,
    ty: Vec<u32>,
    image: Option<Vec<[i32; 3]>>,
    types: TypeTable,            // name/element/mass per type
    id_index: OnceCell<IdIndex>, // id → local index, built on demand (dense Vec if ids are compact, else sorted pairs)
}
```

Resident core cost: 34 B/atom, or 46 B/atom with images.

**Current code.** `Frame { cell: UnitCell, particles: ParticleData, meta: Metadata, type_labels,
type_masses }`. `ParticleData` holds positions, ids, types, optional images, an optional selection mask
and the `PropertyStore`. Data-file
`Masses` go to `type_masses` (NaN for a type without a mass), `Atom Type Labels` and dump `element` to
`type_labels`. **Open:** `TableStore`/`ScalarStore` and `id_index`; per-group results are returned by
each analysis as its own structs instead.

### 7.2 Properties (per-atom columns)

```rust
pub struct Key(&'static str);       // namespaced: "ptm.structure", "cna.structure", "grain.id", "ws.occupancy"
pub enum Buffer { U8(Vec<u8>), U16(..), U32(..), I32(..), I64(..), F32(..), F64(..) }
pub struct Column { key: Key, comps: u8, buf: Buffer, unit: Option<&'static str> }
```

- **Zero-copy construction.** `Column::from_vec::<T>(key, comps, Vec<T>)` takes ownership; there is no
  element-wise copy. Typed access is `col.as_slice::<f32>()`. There is no f64 detour, so i64 ids above
  2⁵³ are exact.
- **Namespacing is mandatory.** CNA and PTM write `cna.structure` and `ptm.structure`. Consumers declare
  which key they need. There is no bare `StructureType`.
- **Built-in input keys:** `vel`, `force`, `charge`, `mol`, `mass` (per atom when present), plus any LAMMPS
  column (`c_*`, `f_*`, `v_*`, `d_*`, `i_*`) under its own name.
- Per-group data are **never** stored as per-atom columns. They go to `TableStore` (§7.3). A writer must
  never pad a per-grain array to N rows.

**Open:** CNA and PTM both still write the bare `StructureType` column, so running both makes the second
overwrite the first; consumers assume PTM codes. Input columns are stored under their LAMMPS names
(`vx`, not `vel`).

### 7.3 Tables and scalars

A `Table` has a schema (`Vec<(name, dtype)>`) and columnar storage (`grains`, `grain_pairs`, `clusters`,
`defects.vacancies`, `rdf`, `wc`, …), written as CSV or JSON and concatenated across frames in long format
(§20). Scalars are `name → (value, unit)`; the JSON report and Python dicts are generated from them.

### 7.4 Selections and regions

- **Selection** is a bitset (1 bit/atom). It comes from an expression, from a column, or from analysis
  output (e.g. `ptm.structure != OTHER`).
- **Expression grammar** (`data/select.rs`, hand-written): comparisons on `x y z id type` or any column,
  `&& || !`, parentheses, element names, `within(<selection>, r)`. Implemented today: clauses
  `FIELD OP NUMBER` joined by `&&`, with `FIELD` in `x y z` (Å), `sx sy sz` (fractional), `type`, `id`;
  used by `rmda defects --region EXPR` (repeatable, one region per expression).
- **Region labels.** `--label-region 'film: z > 500; substrate: z <= 500'` writes `region.id` (u8) and a
  name table.

**Open:** `|| !`, parentheses, columns, element names and `within` in the grammar; `--label-region` and a
general `--select`.

### 7.5 Ordering

- The reader preserves file order. `Atoms::id_index` provides id lookups.
- **Spatial reorder** (optional, on by default above 1M atoms): the grid's bin order becomes the storage
  order, with a permutation `perm: Vec<u32>` (4 B/atom). Writers emit the original order, or `--sort-id`.
  All per-atom outputs are permutation-invariant (tested). This roughly doubles the cache efficiency of
  neighbour queries at 10⁸ atoms.

**Open:** spatial reorder and `--sort-id` are not implemented.

---

## 8. I/O

### 8.1 Architecture

```
ByteSource  ──►  FrameScanner  ──►  ColumnPlan  ──►  parallel RowParser  ──►  Frame (preallocated)
 (mmap | pread blocks | gzip stream)   (header, frame byte range)   (resolved once per header)
```

**`ByteSource`** yields a frame's bytes as one contiguous slice:

- **Mmap** (`io/engine.rs`): zero-copy. After each frame it should `madvise(MADV_SEQUENTIAL)`, then
  `MADV_DONTNEED` consumed ranges, so page cache does not accumulate as RSS.
- **Pread**: 64 MB blocks into a reusable buffer, intended as the default on Lustre, where mmap page
  faults are slow (`--io=pread|mmap|auto`; `auto` detects Lustre via `statfs`).
- **Gzip** (`io/stream.rs`): the `MultiGzDecoder` byte stream cut into a per-frame buffer. Concatenated
  members (LAMMPS `dump *.gz`) work, even cut mid-frame; the frame header alone decides where a frame
  ends, and one decompressed frame is resident, not the file.
- **Parallel inflate** (`io/gzpar.rs`, `gzpar::open`). The reader yields exactly `MultiGzDecoder`'s
  bytes — the same `miniz_oxide` inflate, every member's CRC-32 and ISIZE checked — so parsing is
  bit-identical by construction; it only changes who inflates. The compressed file is mapped, and
  consumed pages are `MADV_DONTNEED`ed every 64 MB so it does not accumulate in RSS.
  - *Indexed*: `rmda index FILE.gz` (one serial pass, ~220 MB/s of text, checking every member) writes
    `FILE.gz.rmdaidx` (binary, magic `MDAGZIX1`, the frame index's size+mtime stamp): zran-style access
    points at the first deflate block boundary after every 8 MB of output — input byte, the 0–7
    unconsumed bits of the last byte (`miniz_oxide` `block-boundary` feature,
    `DecompressorOxide::from_block_boundary_state`), output offset, and the member's last ≤ 32 KiB of
    output (stored deflated; the index is ~0.4 % of the `.gz`). A span between two points is inflated on
    its own worker into a buffer prefixed with the window (non-wrapping output, so back-references reach
    it), continuing across member ends; it stops at the next point's input byte with exactly its output
    length, else the index does not match and the read fails. Points sit only inside members, never after
    a final block, so each span owns every trailer before its end; per-span CRCs are combined
    (`Crc::combine`) and checked against each trailer in order. Any gzip file, one member or many.
  - *Members* (no index): after the first member, inflated by flate2, is ≤ 64 MB (`MEMBER_CAP`), every
    gzip header signature ahead of the current member is inflated speculatively as a whole member on the
    pool (`2 × threads` in flight, ≤ 64). A result is used only when its offset is exactly where the
    previous member ended, so a signature inside compressed data costs work, never bytes. This is
    `GzWriter`'s and bgzip's layout.
  - *Serial*: the first member, a member over the cap, and anything the parallel paths reject (header,
    CRC, trailing bytes) go through flate2's own member decoder, so its errors are flate2's. `--threads 1`,
    a reader inside a rayon worker, and an unmappable file use `MultiGzDecoder` unchanged.
  - `Cursor::load` of a `.gz` (the whole-file readers) goes through the same reader, so it reads every
    member of a multi-member file such as `GzWriter`'s.

**`FrameScanner`** parses dump `ITEM:` blocks in any order up to `ATOMS` (`UNITS` and `TIME` fill
`meta.units`/`meta.time`, as written by `dump_modify units/time yes`; duplicate or unknown items are
errors), data-file header keywords with trailing `# comments`, and extxyz count and comment lines. It
returns the body byte range and N. A frame outside the requested window is **stepped over by counting
rows**, never parsed, so a broken row in a skipped frame does not fail. `FrameIter` stops after an error.

**`ColumnPlan`** (`DumpPlan`, extxyz `XCol`) maps header columns to sinks once per header:

```rust
enum Sink { Id, Type, PosX(Kind), .., Image(d), Prop{col, comp, dtype}, Skip }
enum Kind { Wrapped, Unwrapped, Scaled, ScaledUnwrapped }   // x / xu / xs / xsu
```

- Position source priority `x` → `xu` → `xs` → `xsu` (as `reader_native.cpp`); scaled coordinates go
  through `cart()`.
- Every other column becomes a property under its own name: i64 for `mol`, `proc`, `procp1`, `i_*`,
  `i2_*`, f64 otherwise; `element` becomes `type_labels`. A partial set of image flags (`ix` without
  `iy`) is an `Err` naming the columns, never a panic.
- **`Skip`** (`io::ReadOptions`): a skipped column is stepped over and never allocated, in every input
  format. Analysis commands read with `ReadOptions::lean()` (skip `vx vy vz ix iy iz`, which no analysis
  reads); `info`, `convert` and `--reference` files keep everything; `--skip-columns LIST` overrides both
  (`''` keeps all).

**`RowParser`** (one path for dump, data `Atoms`/`Velocities` and extxyz): count newline-aligned blocks
once, in bounded windows so a multi-frame file is not scanned past the frame; prefix-sum the counts; each
thread parses its rows straight into the preallocated SoA slices. No allocation or hash lookup per row,
field splitting is a byte scan, numbers are parsed in one scan of their bytes (§8.2), and row r goes to
slot r. `--threads` also sizes rayon's global pool, on which readers and writers run.

**Bit identity.** A serial reader is kept as a test-only reference (`io/legacy.rs`).
Randomised dumps (four position kinds, shuffled columns, three box kinds, several frames), data files and
extxyz files must give the same position bits, ids, types and cell. On a real file:
`RMDA_EQUIV_FILE=<file> cargo test --release -p rmda-core --lib real_file -- --ignored`.

**Measured** (thin film `data.lastframe-999`, `rmda info --profile`):

| | wall | throughput | peak heap |
|---|---|---|---|
| read, 12 threads, all columns (Atoms + images + Velocities) | 0.44–0.52 s | 1.8 GB/s | 415–419 MB (73 B/atom) |
| read, 12 threads, lean | 0.34–0.36 s | — | 198.8 MB (34.5 B/atom) |
| read, 8 threads / 1 thread, all columns | 0.52–0.53 s / 2.10–2.14 s | 1.5 / 0.38 GB/s | — |
| serial reference reader (`io/legacy.rs`, Atoms only) | 3.5 s | 0.22 GB/s | 199 MB |
| `convert --format dump` (read + write 348 MB); re-read of that dump | 1.14 s; 0.21 s | —; 1.6 GB/s | — |
| `info`, 3-frame gzip (472 MB `.gz`, 1.05 GB text) | 7.5 s | inflate-bound | 0.58 GB |

Gzip, `rmda info`, 12 threads (machine shared with other jobs; `info` stdout identical to the serial
reader's in every case):

| input | serial | members, no index | indexed |
|---|---|---|---|
| thin film ×4 dump, 1.39 GB text, one member (`pigz -6`, 565 MB) | 9.6–12.1 s | (serial) | 3.0–3.5 s |
| the same written by `convert -o x.dump.gz` (4 MB members, 565 MB) | 9.6 s | 3.7–4.4 s | 3.5 s |
| thin film data, 792 MB text, one member (`pigz -6`, 350 MB) | 5.6–6.4 s | (serial) | 1.4–2.7 s |
| `convert -o x.data.gz` of it (170 MB) | 3.1–3.2 s | 0.5–1.3 s | — |
| `rmda index` of the 565 MB / 350 MB files | — | — | 6.1–6.4 s / 3.5 s |

The plain 1.39 GB dump reads in 1.7 s; what is left is the consumer copying 4 MB pieces into the frame
buffer and scanning lines while no frame is parsed, plus the frame parses themselves. Peak RSS on the
dump is ~1.0 GB against 0.6 GB serial: up to `2 × threads` 8 MB spans (or members) in flight or waiting.

**Open:** Pread, `--io`, Lustre detection and `madvise` for mapped text; storing input property columns
at reduced precision (they are always f64; `convert --prop-precision` rounds on output only, §8.6);
gzip: seeking to a frame through the gzip index (`--frame k` still inflates the frames before it), and
overlapping inflate with the parse of the previous frame.

### 8.2 Number parsing

- Integers: `parse_i64`, hand-written, with overflow checks.
- Floats (`parse_f64`): **Eisel–Lemire** for every mantissa of ≤ 19 significant digits, with a 128-bit
  5^q table for q ∈ [−64, 64] (fast_float recipe), else `str::parse`. 8-digit SWAR digit parsing. No
  separate Clinger path: Eisel–Lemire is exact on every such mantissa, and real coordinates (15–17 digits;
  `write_data` prints 17) would miss Clinger's `m ≤ 2⁵³` test about half the time.
- Bit-identical to std, including rejections (`parse_f64_is_bit_identical_to_std`: ≈ 1.5M random
  decimal, shortest, scientific and garbage spellings plus edge cases). 28 ns/float against 45 ns for
  `str::parse` on the thin film's tokens.

### 8.3 LAMMPS dump

- **Text.** Everything in §8.1; multi-frame files; the atom count may change between frames.
- **Binary** (`io/binary.rs`; `.bin`/`.lammpsbin`, or sniffed by the `DUMP…` magic), read and write
  (`Format::LammpsDumpBinary`): old and revision-2 headers (`ntimestep`, `natoms`, `triclinic` or the
  magic/revision header, `boundary[3][2]`, 6 or 9 box doubles, general triclinic, `size_one`, units, time,
  column string, `nchunk`); body `nchunk × (n int, n doubles)`, parsed in parallel 64k-row pieces;
  pre-magic files read as `dump atom` layouts. Checked field by field against
  `dump_atom.cpp`/`dump_custom.cpp`, and LAMMPS `tools/binary2txt.cpp` decodes RMDA's binary into
  `tests/data/binary2txt_oracle.dump`; both read to the same frames.
- **Frame index.** `rmda index FILE` writes `FILE.rmdaidx` (byte offset, line, timestep, N per frame) for
  text dumps, xyz and binary. Readers seek by it while the file's size and mtime match and ignore a stale
  one; error line numbers survive a seek. For a `.gz` file the same command writes the gzip index
  (§8.1) instead.

### 8.4 LAMMPS data file

- **Header:** counts (with trailing comments), box (§6.2), `atom types`.
- **Sections:**
  - `Masses` → `type_masses`, written back. A frame without masses gets **no** `Masses` section; the
    writer never invents 1.0.
  - `Atom Type Labels` → type names; labels are accepted in `Masses` and `Atoms` rows.
  - `Atoms # style`: `atomic`, `charge`, `full`, `molecular`, `bond`, `angle`, `sphere`; `mol`, `q`,
    `diameter`, `density` become properties. Without a style comment the style is `--atom-style`
    (`ReadOptions::atom_style`), else inferred from the column count (5/8 atomic, 7/10 full); 6/9 are
    ambiguous and are an error asking for the style. A comment naming another style than
    `--atom-style` is an error.
  - Trailing image flags in `Atoms` rows → images. `Velocities` → `vx`/`vy`/`vz`, matched by id
    (in-order fast path, sort-join otherwise). Inline `# comments`. Topology sections skipped by count.
- `.lmp`, `.data`, `.lammps` are sniffed (`ITEM:` → dump, an `atoms` header → data); atomsk writes data
  files as `.lmp`.

General-triclinic `avec/bvec/cvec` headers are read and written (§6.2).

### 8.5 XYZ / extended XYZ

- The comment-line grammar is ASE's `key_val_str_to_dict` (`xyz::kv_pairs`): `key=value` with blanks
  allowed around `=`, double-quoted keys and values with `\"`/`\\` escapes, `[...]`/`{...}` arrays
  (commas read as blanks), and bare boolean flags, which are stepped over without ending the line.
  `Timestep=` (an integer, or a real holding one) and `Time=` go to `meta.timestep`/`meta.time`; other
  flags and unknown keys are recognised and dropped (no frame attribute store); a bare structural word
  (`Lattice constant 3.6` as a plain-XYZ title) is never taken for a key. The writer emits `Time=` only
  for a frame that has a time.
- **`Lattice`** is 9 numbers, ASE vector-major by default (`--extxyz-lattice atomsk` for component-major).
  **`Origin="ox oy oz"`** (OVITO's key) is read and written. The writer never emits a 12-number `Lattice`
  (ASE rejects it); the reader still accepts it. ASE and OVITO read RMDA-written files with the right
  cell, origin and pbc.
- **`pbc="T T F"`** maps to boundaries. **`Properties`** maps into a `ColumnPlan`: `species:S:1` through
  a small species→type table (no per-row `String`), an `id` column becomes the atom id (an integer, or
  an integral real; anything else is an error), `L` columns u8. A `species="A B"` key lists the type
  labels in type order and wins over the symbols seen. Otherwise, with a `type` column, type `t` is
  labelled with the symbol its atoms carry when all of them carry the same one and no other type carries
  it (`xyz::labels_from_type_symbols`; a label names one type, as LAMMPS labels and OVITO type names do);
  other types get an empty label, and a frame with none gets no labels, so RMDA's `X` placeholder for
  unlabelled types reads back as no labels. OVITO 3.15 names the types of such a file the same way
  (Ni = 1, Cu = 2 for `Ni … 1 / Cu … 2`), though it numbers types from the symbols in first-seen order
  and ignores the `type` column. Without a `type` column the symbols are the types, numbered in sorted
  order.
- Plain xyz: species + positions; a box from `--cell`, otherwise the atoms' bounding box with
  non-periodic boundaries (`meta.cell_inferred`).

**Open:** `--extxyz-lattice`; a frame attribute store for other keys and flags.

### 8.6 Writers

- Dump, data and xyz writers format ~64k-row chunks on worker threads and write them in order through a
  bounded batch; the bytes equal the serial loop's (`parallel_rows_equal_a_serial_loop`).
- Formats: dump text (`dump custom` header), dump binary, data (Masses when known), xyz/extxyz; all
  four from `rmda convert` (`--format xyz|dump|data|bin`, or the `-o` suffix).
- **`WriteOptions`** (`io/writer.rs`, `Writer::write_with`). `WriteOptions::default()` is byte-identical
  to the plain writers, so `convert` without these flags writes the plain writers' bytes (checked by
  sha256 on the thin film, dump and xyz):
  - `precision: Option<FloatFmt>` (`--precision N|full`), positions and box. Default: 10 decimals in
    dump and xyz, std shortest `{:e}` in data. `Fixed(N)` is `{:.N}`, the semantics of the existing
    10-decimal positions; `Shortest` (`full`) is std's shortest round-trip spelling (exponent form
    below 1e-5 and from 1e16), which reads back bit for bit (`shortest_spelling_round_trips_every_double`,
    200k random doubles through `parse_f64`).
  - `prop_precision` (`--prop-precision N|full`, requires `--props`), float property columns only;
    default `full`. Integers are always written as integers; an `f32` is printed as the `f64` it widens
    to. Analysis property files (`-o` of the analysis commands) keep their 10 decimals.
  - `props: PropSelect` (`--props all|none|a,b,c`): the frame's per-atom properties. Under `all` what
    a format cannot carry is skipped and named on stderr; a named one that cannot be carried, is missing
    or is not one value per atom is an error.
  - `extended` (any `--props`, even `none`): type labels (dump `element` column when every type has a
    bare-token label; data `Atom Type Labels`; extxyz `species=` key in type order), extxyz `id:I:1`
    and `Selection:I:1` columns, data image flags.
- **Per format.** Dump text/binary: a `k`-component property is `name[1] .. name[k]` (LAMMPS' spelling);
  names the reader routes to the atoms (`id type x y z ix iy iz element`) and floats under integer
  names (`mol`, `i_*`) are refused. Dump columns carry no type, so integer properties read back as f64
  holding the same value unless the name is a LAMMPS integer one. Binary: doubles (exact; integers to
  2⁵³), no labels; a frame over one chunk's `int` count is split over several chunks. Extxyz:
  `name:I|R:k`, so dtype class and components survive (u8/i32 come back as i64). Data: `q`, `mol`,
  `diameter`+`density` pick the `Atoms` style (`full`, `charge`, `molecular`, `sphere`, else `atomic`);
  `vx vy vz` go together into `Velocities`; the file declares at least as many types as the atoms use.
- **Gzip output** by the `.gz` suffix (`GzWriter`): 4 MB blocks compressed in parallel as concatenated
  gzip members, written in order; the layout does not depend on the thread count. `.bin.gz` is refused
  (gzipped binary input is not supported). Thin film dump: 348 MB → 141 MB in 7.2 s on 12 threads
  (885 % CPU), against 0.7 s uncompressed — deflate-bound.
- **Round trips** (`io/roundtrip_tests.rs`, `tests/convert_out.rs`): read(write(frame)) at full
  precision gives the same position bits, ids, types, labels, images, selection and property values
  in every format, plain and gzipped; at reduced precision exactly the printed decimals.
- Writers never reorder silently: input order, or `--sort-id`.

- **`WriteOptions::atom_style`** (`convert --out-atom-style STYLE`, data output only; `--atom-style`
  stays the input style): the `Atoms` layout is the named style (`atomic charge full molecular bond
  angle sphere`) instead of the one the selected properties imply. Its `mol`/`q`/`diameter`/`density`
  columns come from the frame's properties of those names, selected or not, and must exist (nothing is
  invented, as with `Masses`); a selected one the style lacks is an error when named and skipped under
  `all`; velocities go to `Velocities` as for any style (`lammps::forced_style_plan`). `atomic` without `--props` is the default
  bytes.

**Open:** data output of more than one frame (refused); unpacked images (`MdaImage` is LAMMPS' packed
`imageint` throughout); the gzip level (default 6) is fixed; dump labels of types absent from the frame
are not written (a dump has no per-type table).

### 8.7 Series (multi-file input)

- `rmda defects FILES...` (§20) orders files by the trailing integer of each name; files without one
  (e.g. `data.firstframe`) come first as indices −n..−1. The file index is carried into every output
  row. Multi-frame files are read one window per file.

**Open:** `Series::from(glob | list | dir + regex)` with a capture-group index, explicit baseline files in
a config, and series input for every command.

---

## 9. Neighbour subsystem

### 9.1 `KnnIndex` (CSR bins)

`neighbor::KnnIndex` (`neighbor/knn.rs`) serves CNA, PTM, segmentation and defects.

- **Bins in fractional space** (triclinic-correct), stored as CSR, built in parallel and deterministically
  (tested equal to the serial build). Fractional coordinates are recomputed per pass, not stored.
- **Bin size comes from the occupied volume**, not V/N of the box. Non-periodic axes are binned over the
  atoms' own extent, so vacuum costs nothing; a declared-periodic slab costs only empty bins. Periodic
  images are explicit, so a cell smaller than the search shell works.

### 9.2 Queries

- **`knn(i, k)`** is exact: a Chebyshev-shell search over bins that stops once the k-th distance is ≤ the
  minimum possible distance of the next shell. The radius is **unbounded**, independent of V/N and of any
  build cutoff, so it is vacuum-independent (a radius sized from V/N makes PTM's gather in a slab crop take
  milliseconds per atom). It also stops once every bin has been visited without a new candidate, so a lone point
  in a periodic box (all images excluded) cannot loop forever. Ties break deterministically; scratch is
  per worker, with no allocation per query. Tested against brute force: orthogonal, triclinic with
  origin, slab in vacuum (periodic and open), a cell smaller than the shell.
- **`within(i, r)`** is served by the `cell_list.rs` linked list (RDF, WC, cluster, `ws`, `grain-boundaries`).

**Open:** a `within` on the CSR grid (precomputed stencil, per-axis reach; `cell_list.rs` recomputes the inverse
cell per query, allocates 3 `Vec`s, builds serially, takes a scalar reach from the thinnest axis and has
no `u32` overflow guard); retiring `nnearest.rs`, whose `nearest_k` silently misses neighbours beyond the
build stencil.

### 9.3 The fused local pass

Per-atom analyses implement `LocalKernel` (§12.1) and one `knn(i, k_max)` per atom feeds all of them
(CNA 14, PTM 18, WC shells 12+6). The k-NN table is **never** materialised for the frame
(18 × 16 B × 128M = 37 GB); it lives for one tile of ~64k atoms per thread, and grains and WS re-query
`knn`, which costs about one PTM gather.

**Measured:** against a radius-bounded neighbour query, `KnnIndex` halves thin-film CNA (20.0 s → 10.0 s)
and cuts PTM from 67 s to 21.3 s, with identical classification: with the radius-bounded query, 70 % of
PTM time is the neighbour query (586 thread-s against 191 matching), not the hull.

**Open:** fusion; CNA and PTM each run their own scan.

---

## 10. Execution model and memory discipline

### 10.1 Threads

- `--threads N` (0 = all cores) calls `exec::set_threads(N)`, read by every `Par::default()`, and
  `exec::init_global_pool(N)`, which bounds rayon's global pool used by readers and writers. There is no
  second pool. Python's `rmda.set_threads` calls the same function; `rmda.get_threads()` reads it back.
- `--threads 1` runs the same code sequentially, with identical results.
- RMDA uses rayon, not OpenMP: `OMP_NUM_THREADS` and `OMP_PLACES` have no effect.

- `--threads 0` (the default) means `SLURM_CPUS_PER_TASK` when set (under SLURM every core is the
  node's, not the job's), else every logical core (`exec::default_threads`).

**Open:** an `RMDA_NUM_THREADS` variable between `--threads` and `SLURM_CPUS_PER_TASK`. `ParPool` is unused and should go.

### 10.2 Parallel primitives (`exec/pool.rs`)

- `Par::fill` and `Par::try_fill_with(out, init, f)` write into **preallocated** slices with per-worker
  scratch: no `Vec<Result<R>>` per item, no second copy. `map_collect` collects per chunk. `map_chunks`,
  `for_each_chunk`, `fold_chunks`, and `for_each` for fine-grained pieces (e.g. 256-atom BFS levels, which
  65 536-atom chunks would serialise).
- **Errors:** the lowest-index error is kept, deterministically (tested across chunks for 1–4 threads).
- **Determinism:** integer reductions merge in chunk order; floating-point reductions are summed per fixed
  chunk (boundaries independent of thread count), then merged in order; union-find roots are component
  minima (§15.2).

### 10.3 Tiles

Per-atom passes iterate over **tiles** of 16k–64k reordered (≈ spatially blocked) atoms, with
thread-local scratch (k-NN buffers, PTM hull workspace, histograms) sized per tile and reused. No
transient may scale as N × (per-neighbour bytes). **Open:** without §7.5 reorder, chunks are contiguous in
file order, not spatial.

### 10.4 Memory estimate and limits

Every analysis implements `fn memory(&self, n, plan) -> MemEstimate { resident, transient }`; the
pipeline prints the estimated peak before running and, past `--mem-limit` (default 90 % of the cgroup or
`MemAvailable` limit), refuses with a message naming the dominant term and the flag that reduces it.

**Current code.** `--mem-limit SIZE` is a run-time watchdog: the run stops with exit code 4 when its
anonymous RSS exceeds SIZE (checked every 100 ms; a series keeps the frames it finished). **Open:** the
per-analysis estimate, the pre-run refusal and the default limit.

---

## 11. Structure registry and orientation mathematics

### 11.1 `StructureRegistry`

```rust
pub struct StructureId(pub u8);   // OVITO-compatible codes for built-ins: OTHER 0, FCC 1, HCP 2, BCC 3, ICO 4,
                                  // SC 5, CUBIC_DIAMOND 6, HEX_DIAMOND 7, GRAPHENE 8; derived ≥ 32 (DHCP 32, …)
pub struct StructureDescriptor {
    pub id: StructureId,
    pub name: &'static str,                       // "fcc"
    pub lattice: Option<LatticeSpec>,             // conventional cell, basis, free params (a, c/a), nn factor
    pub point_group: &'static [Quat],             // PROPER rotations orientations are compared under
    pub ptm: Option<PtmTemplateSet>,              // templates (incl. alternates), conventional-frame offsets, factors
    pub cna: Option<CnaSignature>,                // n neighbours, signature counts, cutoff rule
    pub compatible: &'static [(StructureId, Quat)], // e.g. (FCC, q_HCP→cubic) for stacking-fault/coherent interfaces
    pub derived: Option<DerivedRule>,             // dHCP, 4H, 9R: classify from layer sequence of FCC/HCP neighbours
    pub ws: WsSpec,                               // first-shell distance factor, site motif, per-cell site count
}
```

- **Rule:** analysis code iterates descriptors or looks one up by id; it never matches on codes (a CI grep
  test fails on `match` arms over structure constants outside `structure/`). Adding a structure means one
  descriptor file, optional PTM templates and a registration line.
- **HCP:** `c/a` is a fitted free parameter seeded at the ideal √(8/3) ≈ 1.633 (not √(2/√3) ≈ 1.075), defined exactly once.

**Current code.** Three places must agree: `ptm::structure` (OVITO codes OTHER 0 … SC 5), `orient::group`
(symmetry groups for FCC/BCC/SC/HCP) and `defects::lattice::Motif::for_structure` (cell, basis,
translations and nn for FCC, BCC, SC and HCP; the HCP motif is orthohexagonal at ideal c/a, the same basis
as `lattice::hcp`, and the fit frees c/a). The `lattice/` site lattices (fcc, bcc, hcp, dhcp, sc, dia) serve
only `rmda ws`. **Open:** the registry itself; structure codes are hard-coded at ~85 sites in 11
files, with CNA (`u8`) and PTM (`i32`) codes as two type systems.

### 11.2 Derived structures (dHCP and stacking polytypes)

A `DerivedRule` runs after PTM over close-packed atoms (FCC ∪ HCP): identify close-packed layers per grain
from the ⟨111⟩/c axis, assign an A/B/C letter per layer, and match the period (ABAB → HCP, ABCABC → FCC,
ABAC → dHCP, ABCB → 4H, …). The output is a separate key `structure.derived`, so `ptm.structure` stays
OVITO-comparable; WS uses the derived lattice for a dHCP grain.

**Open:** dHCP is identified per atom by `analysis/stacking.rs` rather than by this layer rule (§28.1);
no other polytype (4H, …) is identified.

### 11.3 Orientation algebra (`analysis/orient.rs`) — one convention, everywhere

- **Convention.** PTM's `q` rotates the template (crystal) frame into the sample frame,
  `x_sample = R(q) · x_template`. Symmetry `g ∈ G` acts in the template frame, so `q` and `q·g` are the
  same orientation: **symmetry acts by right multiplication, everywhere.** Left multiplication (`g·q`)
  gives 25–47° between equivalent orientations, and a unit test re-stating the same formula would
  pin that bug. Tests must compare against an independent construction.
- **Groups** (`group(st)`): order-24 cubic for FCC, BCC, SC; order-**12** for HCP (the hexagonal group
  plus its product with the 60° rotation about c). PTM's alternate HCP templates differ by that 60°, and a
  60° turn plus a c/2 shift maps HCP onto itself; under the order-6 group one HCP crystal splits into ~12
  grains.
- **Functions:** `disorientation(qa, qb, G)` = `2·acos(max_g |qa · (qb·g)|)` in degrees;
  `align_to(r, q, G)`, the `±q·g` closest to r; `mean(qs, G)` / `MeanAcc` (align every q to a reference,
  then the largest eigenvector of `Σ q qᵀ`, Markley). Every per-grain and per-crystallite orientation goes
  through `mean`; a raw quaternion sum is wrong.
- **Cross-structure comparison** (`compatible`, `convert`, `disorientation_between`). `hcp_to_fcc()`
  maps the HCP frame into the FCC frame (c ∥ [111], a ∥ [1-10]); FCC/HCP pairs are compared after
  converting the HCP orientation, in the FCC group, symmetrically in argument order. A test checks that
  the mapped HCP template's basal ring and lower layer land on FCC sites, and that both twins match.
- **Matrix ↔ quaternion** conversions are tested by a matrix round trip in every branch of Shepperd's
  method (an inverse rotation in one branch is invisible to a quaternion-to-quaternion test).
- **Tests:** `disorientation(q, q·g) = 0` (cubic and hexagonal), equivalents average to one orientation,
  closure of the hexagonal group under the 60° turn, the FCC/HCP map. Target: 10⁴ random cases each for
  symmetry, triangle inequality, fold idempotence and `mean` of jittered rotations, and an OVITO
  quaternion fixture at random orientations including near fundamental-zone boundaries.

**HCP output orientation = OVITO's.** OVITO 3.15's PTM has no conventional-orientation switch: after a
match it remaps the quaternion to the template variant nearest the identity (`ptm_remap_template(type, 0,
identity, ..)`), which for HCP is the representative with the largest `|w|` over the order-12 group.
Every structure reduces over its `orient::group` (`HCP_CONVENTIONAL` for HCP); a reduction over the six
D3d generators alone leaves about half of the HCP atoms (those fitted to the alternate template) 60°
about c from OVITO. With OVITO's neighbour order every matched atom's quaternion equals OVITO's to f32 on
the four reference snapshots, sign included (`ptm_oracle.rs`,
`voronoi_order_matches_ovito_orientations_exactly`; the oracle's orientations come from
`tools/capture_ptm.py`). **Open:** `fold`, `misorientation_axis_angle` (Σ/twin identification, GB character) and the
10⁴-case suite.

---

## 12. Analysis framework and pipeline

### 12.1 Trait

```rust
pub trait Analysis: Send + Sync {
    fn id(&self) -> &'static str;                         // "ptm"
    fn requires(&self) -> &[Key];                          // e.g. grains: [ptm.structure, ptm.orientation]
    fn provides(&self) -> &[Key];                          // columns, tables, scalars it writes
    fn neighbors(&self) -> NeighborRequest;                // None | Within(r) | Knn(k) | Local(kernel)
    fn memory(&self, n: usize, plan: &Plan) -> MemEstimate;
    fn run(&self, ctx: &mut Ctx, frame: &mut Frame) -> Result<()>;   // writes into frame; records stage spans
}
pub trait LocalKernel: Send + Sync {          // fused per-atom analyses (CNA, PTM, WC shells, census)
    fn k(&self) -> usize;
    fn columns(&self) -> &[ColumnSpec];
    fn eval(&self, i: AtomIdx, nbrs: &[Nbr], frame: &FrameView, out: &mut RowWriter, scratch: &mut Scratch);
}
```

`Ctx` holds the pool, the frame's grid (built once at the plan's maximum build cutoff), the registry,
the profiler, the memory tracker, the config and the run report.

### 12.2 Plan

- `Pipeline::plan(steps, frame_header)`:
  1. Topologically sort by `requires`/`provides` keys. A missing provider is an error that names it.
  2. Fuse adjacent `LocalKernel`s into one pass.
  3. Choose the grid build cutoff.
  4. Sum the memory estimates.
  5. Print the plan with `--explain`.
- The `Registry` maps `id → constructor(config)`. The CLI, the TOML config and Python all build pipelines
  through it. It is the only dispatch table.

### 12.3 Configuration

- Every analysis has `XxxConfig` with `Default` defined **once** in core (as `SegConfig` and
  `DefectsConfig` already are).
- The CLI flags are derived from it with clap `#[command(flatten)]`. The TOML section `[ptm]` maps
  field by field. Python kwargs map field by field.
- One name per option in all three. Option names are kebab-case in the CLI (`--rmsd-cutoff`) and
  snake_case in TOML and Python (`rmsd_cutoff`).

### 12.4 Results

- Analyses write `Column`s, `Table`s and `Scalar`s into the frame.
- The run model (§20) selects what is written. The `RunReport` (§23) collects per-stage cost.
- No analysis prints to stdout.

**Open (the whole section).** The framework is nominal: `analysis/mod.rs` has an `Analysis` trait with
stringly-typed property dependencies and a WS-specific `take_defects` escape hatch; `registry.rs` has no
users; `analyze` hard-codes order and set; nothing shares neighbours between PTM and grains.
`segment` and `defects` are plain functions (`grain_seg::segment`, `defects::run`); their per-frame glue
(`defects::frame`) is shared by `cmd_defects.rs` and `rmda-py` (§22). Options differ between commands (`segment --threshold`/`defects --seg-threshold`, `--rmsd-cutoff`/`analyze --ptm-rmsd`)
and some defaults differ between core, CLI and Python (e.g. `min_grain_size` 0 vs 100).

---

## 13. CNA

### 13.1 Modes

| Mode | Neighbours | Definition | Status |
|---|---|---|---|
| `adaptive` (default) | k-NN 12 (FCC/HCP/ICO), then 14 (BCC) | OVITO a-CNA: local cutoff from mean of the 12 nearest (×(1+√2)/2) or the 8+6 nearest (×(1+√2)/2 on the weighted mean). | implemented; matches OVITO bit-for-bit on all reference atoms |
| `interval` | k-NN 14 | Larsen 2020 interval CNA (OVITO `IntervalCutoff`, `determineStructureInterval`): neighbour-neighbour bonds added shortest first along a short (12-neighbour) and a long (14-neighbour) path; at each step with FCC/HCP/ICO or BCC coordination the ordinary signature is tested, and the valid type with the widest gap to the next bond wins. Parameter-free and more robust at high T. | implemented; OVITO atom for atom |
| `fixed` | k-NN 15 | Global cutoff; an exact 12 or 14 inside it. | implemented; OVITO atom for atom |
| `bond` | within(r), CSR with image shifts | OVITO `BondBased` on `CreateBondsModifier(r)`: per-bond `(common, bonds, chain)` with image-consistent common neighbours (shift arithmetic, as OVITO's `pbcShift`), per-atom type from the counts over **all** its bonds. 32-common / 64-bond limits are errors, as in OVITO. `--full` counts bonds. | implemented; OVITO atom for atom |
| `diamond` | k-NN 4 → second shell 12 | OVITO `IdentifyDiamondModifier`: 12 second neighbours `v0 + w` (skipping `isZero(1e-2f)`, exactly 3 per first neighbour), cutoff `mean·1.2071068`, FCC/HCP CNA → cubic/hexagonal diamond; then first and second neighbours marked in index order, in place. Codes 0–6 (`cna::diamond`). | implemented; OVITO atom for atom |
| `shell` | k-NN 15 | Not OVITO: the neighbours in `(low, high]`, an exact 12 or 14, bonded at `high`. | implemented |

### 13.2 Implementation

- CNA uses `KnnIndex` and `Par::try_fill_with`. The bitmask common-neighbour and chain-length code is
  kept. No per-atom allocation.
- Signature `counts` are allocated only with `--cna.signatures`.
- Output: `cna.structure` (u8). Optional `cna.signature` table.

**Measured:** thin film, adaptive, 12 threads, interleaved with OVITO 3.15: `cna.scan` 2.4–3.6 s against
OVITO 5.4–5.6 s (0.43–0.65×), 391 MB. The speed comes from
k-NN bins scanned nearest first and pruned by their distance bound, queries run in bin order, one
sweep for the 12- and 14-neighbour bond matrices, and a cheap reject in the top-k insert.

**Validation of the variants** (OVITO 3.15.0, `tools/gen_structure_variants.py`,
`tests/structure_variants_ovito.rs`): interval, fixed, bond-based and identify-diamond equal OVITO's
labels on every atom of the four reference snapshots (Al 500 000, W 500 094, Zr 492 800, HEA 42 432;
fixed/bond at 3.5/3.9/3.9/3.1 Å) and of 39 synthetic fixtures (FCC/BCC/HCP σ = 0–0.20 Å, a Mackay
icosahedron, Si/Ge cubic diamond σ = 0–0.20 Å with vacancies and a free surface, lonsdaleite, graphene,
graphite): 195 of 195 comparisons at 100 %. The interval threshold fraction uses OVITO's
`2.0f/3.0f` (a float division) widened to double. `--mode interval` is Larsen's interval CNA; the two-radius shell is `--mode shell`.

**Open:**

- Per-bond signature output for `bond` (OVITO's `CNA Indices` bond property); only per-atom counts.
- CNA always allocates `counts` (20 B/atom) and clones `types`, and writes the bare `StructureType`
  (§7.2).

---

## 14. PTM

### 14.1 Per-atom algorithm

1. `knn(i, 18)` from `KnnIndex`, in distance order (two-shell neighbourhoods, 4 + 12 or 3 + 6, for the
   diamond and graphene templates: `ptm::two_shell_environment`, upstream
   `calculate_two_shell_neighbour_ordering`. The inner atoms are the first 4 (3) of the central list;
   each inner atom's own 18-list is fetched in the same order, and the outer candidates are pooled and
   taken by rank, skipping atoms (by index) already used, 3 (2) per inner atom. Upstream sorts the pool
   with the unstable `std::sort`; here it is stable, which differs only when one atom is a candidate of
   two inner atoms at the same rank.)
2. For each enabled structure, in decreasing neighbour count: convex hull → graph canonical form →
   template lookup → optimal rotation (polar decomposition) → RMSD over the scaled points (OVITO-exact
   numerics).
3. Pick the lowest RMSD; if `rmsd > cutoff` the atom is OTHER (cutoff 0 = none).
4. Outputs for the winning template: orientation (folded, §11.3), interatomic distance and lattice
   constant, scale; optionally the template index, correspondences (`u8 × k`), the **elastic deformation
   gradient** `F = (Σ x yᵀ)(Σ y yᵀ)⁻¹` over the mapped points (`[f32; 9]`), and the **ordering type**
   (L1₀, L1₂-A/B, B2, zincblende, wurtzite, BN) from the correspondence-mapped species.

**Neighbour order.** OVITO 3.15 (`PTMAlgorithm::Kernel::cacheNeighbors`) runs
`ptm_preorder_neighbours` over the 18 nearest neighbours and hands them to `ptm_index` in that order:
decreasing Voronoi face solid angle, ties by distance (stable sort). The templates read a prefix (12 for
FCC/HCP/ICO, 14 for BCC), so OVITO matches the 12 largest Voronoi faces, not the 12 nearest.
`Ptm::compute_ordered(.., NeighbourOrder::Voronoi)` (`--ptm-order voronoi`) does the same and agrees with
OVITO **atom for atom** on all four reference snapshots: types 100.0000 % (Al, W, Zr, HEA; distance
order: 99.9928, 99.9952, 99.9302, 99.663 %) and every matched RMSD
(`ptm_oracle::voronoi_order_matches_ovito_exactly`). `ptm/voronoi.rs` sums each face's solid angle as a
fan from the first vertex; summing over consecutive vertex triples undercounts faces of five or more
vertices. **Default:** `Ptm::compute` and every command use distance order (`rmda segment --mode auto`
and `rmda defects --seg-mode auto` default to Voronoi), because the order changes every PTM-dependent
count: thin film 999, `defects --segment --reference local`, V/I 14497/14106 with distance order,
14542/14107 with Voronoi order (one grain either way). Cost: one Voronoi cell per atom, `ptm.scan` 56 s
against ~21 s on the film. Voronoi order is therefore opt-in.

### 14.2 Configuration and outputs

- Defaults match OVITO 3.15: structures **FCC, HCP, BCC**; ICO, SC, cubic diamond, hexagonal diamond and
  graphene are opt-in (`--types`), as OVITO ships them disabled (checked in 3.15.0:
  `PolyhedralTemplateMatchingModifier().structures`). RMSD cutoff 0.1; orientations on. Core, CLI and
  Python share them. Summary keys `ptm_dcub`/`ptm_dhex`/`ptm_graphene` appear only when tested, so the
  default output does not carry them.
- **Two-shell structures** (`check::DCUB | DHEX | GRAPHENE`): `matcher::match_dcub_dhex` (hull of the 16
  neighbours, FCC's 20 facets plus two per inner atom on the hull, the facet surgery that reconnects each
  inner atom to its outer triangle, coloured canonical form with the inner atoms coloured 1; upstream's
  first degree check over only 20 facets reproduced) and `matcher::match_graphene` (8 permutations of the
  outer pairs, no hull). Templates and the 36 DCUB/DHEX hull graphs are generated from upstream
  (`tools/gen_ptm_tables.py`); their derived hashes and labellings are pinned against the compiled C++
  (`tests/ptm_refdata.rs`, 280 graphs). Orientation: DCUB reduces over the cubic group, DHEX and graphene
  over HCP's order-12 conventional group, as OVITO's `ptm_remap_template` to the identity.
- **Measured against OVITO 3.15** (Voronoi order, all six or eight structures enabled): types, RMSD
  (to output precision, 5·10⁻¹¹) and orientations (|Δq| ≤ 3·10⁻⁸, output precision) equal on every
  atom of Si/Ge cubic diamond (σ = 0–0.20 Å, with vacancies and a surface), lonsdaleite (σ ≤ 0.15 Å),
  graphene and AB graphite, FCC/BCC/HCP (σ ≤ 0.20 Å), and the four reference snapshots with all eight
  structures (no atom changes type there when the two-shell structures are added; OVITO agrees). Distance
  order differs from OVITO on 0.3–0.7 % of atoms at σ ≥ 0.15 Å, as for the first-shell structures. A
  perfect Mackay icosahedron is the one case with differing orientations (14–16 atoms): exactly
  degenerate fits, broken differently.
- **Cost** (Al 500k, 12 threads): default 1.2 s; with the two-shell structures 5.4 s (four
  extra k-NN lists per atom, recomputed rather than cached); with them and Voronoi order 20 s (a Voronoi
  cell for the centre and each inner atom). Opt-in, so not optimised: a per-frame cache of the ordered
  18-lists would cut both.
- **Two output forms**, one scan: `Ptm::compute` returns a compact `PtmAtom` (u8 structure, f32 RMSD, f32
  bond, f32 quaternion; 28 B), which segmentation and defects consume directly; `Analysis::run` writes
  f64 property columns, so the OVITO RMSD oracle (10⁻⁹) is unaffected.
- **Opt-in outputs** (`rmda ptm --outputs ordering,F,template,correspondences,residual,polar`, Python
  `outputs=`, `Ptm::run_extended`
  with `ptm::output` flags; module `ptm/remap.rs`, tables `ptm/remap_tables.rs` generated by
  `tools/gen_ptm_remap_tables.py` from a compiled probe of the upstream headers): `StructureOrdering`
  (u8, OVITO's `OrderingType` = upstream `PTM_ALLOY_*`: none, pure, L1₀, L1₂ A/B, B2,
  zincblende/wurtzite, BN; `find_alloy_type` on the fit's mapping and the particle types of the matched
  environment, first-shell or two-shell), `StructureDeformationGradient` (9 × f64, row-major) and
  `StructureTemplate` (u8, `best_template_index`). F follows `output_data` with OVITO's
  `output_conventional_orientation = true`: the raw fit quaternion is reduced over upstream's own
  generator table (`GENERATOR_HCP_CONVENTIONAL` is our `HCP_CONVENTIONAL` in upstream's order, so exact
  ties break as upstream), the mapping permuted by `mapping_*_conventional` (HCP, graphene, DCUB, DHEX)
  or the fundamental mapping, then `calculate_deformation_gradient` against the matching alt template
  and Penrose table. All three describe the best fit whether or not the cutoff accepts it, and are zero
  when nothing matched, as OVITO writes them. OVITO's raw F array is ours; its component labels
  (`XX YX ZX …`) read column-major, i.e. the matrix it labels is Fᵀ.
- **Correspondences** (`Correspondences`, int64): what OVITO 3.15 writes with `output_orientation`
  (its `PTMNeighborFinder` decodes it), upstream `ptm_encode_correspondences(type, …,
  output_env->correspondences, template_index)` with `output_env->correspondences[i] =
  env->correspondences[mapping[i]]` under the *remapped* mapping. A point's correspondence is 1 + its
  distance rank in the k-NN list it came from (the centre's for first-shell points and inner atoms,
  the inner atom's for outer ones; `gather` records the rank alongside the Voronoi reorder, a
  per-atom `Vec<u8>` that the default path fills and never reads). First-shell types: completed with
  the unused ranks ascending, one Lehmer index of a permutation of 18; two-shell: inner atoms
  (base 13, 15 bits) and each inner atom's outer atoms (base 13, 11 bits each). Template index in bits
  62–63 (DHEX 2–3 negative, as OVITO's int64). Written for the best fit, rejected atoms included, 0
  where nothing matched (`remap::encode_correspondences`).
- **Upstream-only outputs** (not in OVITO 3.15, which exposes none of them; present in the PTM library
  as LAMMPS bundles it): `StructureDeformationResidual` (3 × f64, `F_res` of
  `calculate_deformation_gradient`: per-axis Σ_k ((F t_k)_i − s_k,i)² over the alt template and the
  barycentred, scaled points) and `StructurePolarU` / `StructurePolarP` (9 × f64, row-major,
  `polar_decomposition_3x3(F, right_sided = false)`: F = P U, U = rotation, P = F Uᵀ;
  `polar::polar_decomposition_3x3`, same arithmetic order as upstream).
- **Measured against OVITO 3.15** (Voronoi order): ordering equal on every atom, F to 1.3·10⁻¹⁵ and
  interatomic distance to 1.2·10⁻¹⁵ (relative) on eight fixtures (`tests/data/ptm_extras/`,
  `tools/gen_ptm_extras.py`: B2, L1₂, L1₀ with σ = 0.08–0.10 Å and 3–4 % antisite swaps, zincblende,
  wurtzite and h-BN with swaps, strained and rotated FCC and HCP; every template variant occurs); on
  the four reference snapshots ordering equal on all atoms (HEA: 42 409 none, 16 L1₀, 1 L1₂ A, 6 L1₂ B;
  pure metals pure or none), F to 1.6·10⁻¹⁴, interatomic distance to 7·10⁻¹⁵, rejected atoms
  included. `PtmAtom` stays 28 B, and the default `run_ordered` path builds no extra per-atom record.
- **Correspondences measured against OVITO 3.15** (Voronoi order): equal on every atom of the eight
  fixtures (1 942 atoms; FCC, BCC, HCP both templates, DCUB, DHEX all four templates, graphene) and
  of the four reference snapshots (1 535 326 atoms, rejected ones included). U and P are bitwise
  equal to upstream's `polar_decomposition_3x3` (LAMMPS copy, compiled `-O0 -ffp-contract=off`) fed
  our F on all fixture atoms; F_res is pinned by the least-squares identity Σ F_res = Σ|s|² − Σ|F t|²
  on every template variant (unit test). The oracle file is `tests/data/ptm_extras_expected.txt.gz`
  (13 significant digits, gzipped, 153 KB), compared to 10⁻¹² relative (measured 5·10⁻¹³, the
  storage rounding). Al 500k, all outputs with Voronoi order: 7.0 s (F alone 6.7 s; default PTM 2.7 s
  in the same session).
- Target columns (all optional except `ptm.structure`; default set 29 B/atom; `--ptm.outputs` selects;
  as shipped: `StructureType`, `StructureRmsd`, `StructureScale`, `StructureInteratomicDistance`,
  `StructureLatticeConstant`, `StructureOrientation` always (f64), `StructureOrdering`,
  `StructureDeformationGradient` (f64), `StructureTemplate`, `Correspondences` (i64),
  `StructureDeformationResidual`, `StructurePolarU`, `StructurePolarP` (f64) with `--outputs`):

  | column | type | B/atom | | column | type | B/atom |
  |---|---|---|---|---|---|---|
  | `ptm.structure` | u8 | 1 | | `ptm.orientation` | [f32;4] | 16 |
  | `ptm.rmsd` | f32 | 4 | | `ptm.ordering` | u8 | 1 |
  | `ptm.scale` | f32 | 4 | | `ptm.F` | [f32;9] | 36 |
  | `ptm.ia_dist` | f32 | 4 | | `ptm.template` | u8 | 1 |

**Measured:** thin film, FCC/HCP/BCC, 12 threads: 21.3 s against OVITO 31.3 s (0.68×).

**Open:** the f32 storage of the target table (columns are f64 today; the namespaced `ptm.*`
names wait on the framework). `is_match` uses `<` where OVITO
uses `>` for rejection, and a doc comment says "mean squared" where the code uses mean distance.

### 14.3 Oracle

`crates/rmda-core/tests/ptm_oracle.rs` (ignored; with `RMDA_REFERENCE_DIR`) compares against OVITO on four
`reference_structures/` snapshots: per-atom structure (RMDA must not be worse), RMSD (10⁻⁹), orientation and
interatomic distance. **Open:** perfect and thermal (σ up to 0.25 Å) FCC, BCC, HCP and diamond at random
orientations, the HEA polycrystal, a thin film with vacuum and a nanoparticle, with orientations compared
by `disorientation` < 10⁻⁴°.

---

## 15. Grain segmentation and tracking

### 15.1 Input and edges

- **Requires** PTM structure and orientation (the compact `PtmAtom`, §14.2). Only crystalline atoms
  (structure ≠ OTHER) are nodes.
- **Edges** link each node to its 12 nearest neighbours from `knn`; there is no global distance cutoff
  (a cutoff from the maximum interatomic distance is outlier-sensitive). An edge is kept for same-structure pairs, or
  FCC/HCP pairs with stacking-fault merging on. **Weight** θ = `disorientation` after the frame map
  (§11.3).

### 15.2 Algorithm: crystallites, then a merge graph (`analysis/grain_seg.rs`)

Chains of sub-threshold bonds must not bridge a boundary (a plain flood fill lets them), the algorithm must
scale to 10⁸ atoms, and it stays close to OVITO's semantics.

1. **Crystallites.** Edges with θ < θ_core (default 2°) are joined in parallel in a lock-free union-find
   whose roots are component minima, so the result is deterministic. This reduces 10⁸ nodes to about
   10³–10⁵.
2. **Crystallite graph.** Per crystallite: size, structure, symmetry-aligned **mean orientation**. Per
   pair: the bond count.
3. **Merge** in order of the disorientation between the clusters' *current mean* orientations (not the
   chain), below `--threshold` (default 5°) and with at least `--min-bonds` shared bonds (default 4).
4. **Stacking faults and twins** (`--stacking-faults merge|separate`, default merge, as in OVITO). FCC and
   HCP crystallites are compared through `hcp_to_fcc`; a merged cluster keeps the structure and frame of
   its larger part, so HCP fault layers join the surrounding FCC grain (per-atom PTM still says HCP). An
   intrinsic stacking fault gives one FCC grain (two or more without the map); a **Σ3 twin** gives two FCC
   grains at 60.0°, the twin plane joining one of them. `separate` exists for defect counting: the two
   sides of a fault are a partial Burgers vector apart, which no single lattice fits, so `rmda defects`
   withholds a merged faulted grain (`gate_rms_displacement`) but counts the three separate grains
   exactly. A fault ending inside a grain, at partials, cannot be counted against one lattice.
5. **Minimum size and adoption.** Grains below `min_size` (default 100) dissolve. With `adopt` (default),
   each unassigned atom (including OTHER) takes the grain reached by the **shortest path through
   unassigned atoms** (multi-source Dijkstra over 12-NN bonds), reported as `segment_atoms_adopted`.
   Starting from OVITO's own pre-adoption labels on the HEA fixture this matches OVITO's final labels on
   99.4 % of atoms (majority vote: 97.9 %; layered nearest neighbour: 98.4 %).
6. **Relabel** by decreasing size; the pair table comes from the final labels.

**Outputs** (`rmda segment FILE [-o out.xyz] [--table grains.csv]`, `rmda.segment`): per-atom `Grain` (−1 for
none), the grains table (id, size, structure, mean orientation, crystallites) and `*.pairs.csv` (pair,
shared bonds, disorientation; NaN between structures).

**FCC–HCP merging is decided in the larger cluster's frame.** Converting b into a's frame in
`disorientation_between(a, b)` fails: FCC→HCP puts the FCC fundamental-zone representative's own {111} on c, so
whether an FCC/HCP pair merges would depend on which crystallite has the lower index (the order of atoms
in the file), and an HCP-majority cluster's mean would be polluted by its FCC atoms. The rule must respect that
the 60° turn about c, in the order-12 hexagonal group, is the FCC twin operation: compared in the HCP
frame, an FCC orientation matches both twins (searching every cubic variant naively merges the Al frame's
host and twins into one grain). So (`grain_seg.rs`, step 3) a close-packed pair is compared in the frame
of the **larger** cluster: a hexagonal host in the HCP frame, where the 60° turn is a symmetry (an HCP
crystal's FCC-labelled faults join it whichever variant they are); an FCC host in the FCC frame, using the
smaller cluster's own FCC parts when it has any (each cluster keeps their mean separately), so a twin
travelling with hexagonal layers still reads as 60° away. Folding a part into a cluster after the decision
uses every cubic variant (`orient::convert`). Grain structure and reported orientation are still the
larger part's. Against OVITO 3.15's default segmentation (GraphClusteringAuto):

| case | grains | OVITO grains | ARI vs OVITO |
|---|---|---|---|
| FCC twins (synthetic Cu) | 2 | 2 | 0.922 |
| twins + intrinsic faults | 2 | 2 | 0.917 |
| HCP lamella in FCC | 1 | 1 | — |
| HCP lamella between twins | 2 | 2 | 0.677 |
| Al reference frame (FCC, twins, HCP faults) | 6 | 6 | 0.735 |

On the Al frame `defects --segment` gives V/I 3074/71 (interior); counted on OVITO's own grains
(`--grain-column`) it is 3072/58. A segmentation that merges twins and HCP faults into the host counts
them on the host lattice and inflates the interstitials five-fold (3125/338). The partition does not
depend on atom order (`tests/segment_stacking.rs`). A dHCP stack without `--dhcp` is one grain, as in
OVITO, not FCC/HCP slabs.

**Memory:** `parent` 4 B/atom (transient) and the grain id 4 B/atom; edges are never stored. **Measured:**
thin film, segmentation 8.7 s and ~1 GB (33 s with PTM).

**Open:** `--twins merge` with a Σ flag and axis in the pair table; `grain.crystallite`; grain centre,
lattice constant and volume columns.

### 15.2a OVITO's algorithm: `--mode auto` (`analysis/grain_auto.rs`)

`SegMode::Auto` (`rmda segment --mode auto`, `rmda defects --segment --seg-mode auto`, opt-in) ports OVITO
3.15's `GrainSegmentationModifier` with its default algorithm `GraphClusteringAuto` (Larsen's
`GrainSegmentationEngine`, MIT) step for step:

1. **Bonds** from PTM neighbour lists: 12 for FCC/HCP, 14 BCC, 6 SC, the 8 nearest for a non-crystalline
   atom; a pair `i < j` is bonded when `j` is in `i`'s list (the lower index decides).
2. **Coherent interfaces** (`--stacking-faults merge`): minority close-packed atoms (HCP if FCC is the
   majority) bonded to the parent phase within 4° are relabelled parent phase with the mapped orientation,
   smallest disorientation first, spreading.
3. **Graph**: same-structure bonds with θ < 4°, weight `exp(−θ²/3)`.
4. **Node pair sampling** (nearest-neighbour chain, distance `w(a)w(b)/w(a,b)`) → dendrogram sorted by
   distance.
5. **Automatic threshold**: per merge `x = ln(harmonic mean of the sizes)`, `y = ln d`; IRLS
   least-absolute-deviations fit weighted by merge size; threshold = the largest `y` whose residual is
   below 1.5 × the median absolute residual (the knee). If the fit is degenerate (no slope) every merge
   is applied (OVITO would merge nothing).
6. Cut at the threshold, dissolve clusters below `--min-size`, then **orphan adoption**: multi-source
   Dijkstra over the step-1 bonds through unassigned atoms, path length = summed bond lengths.

`--threshold`, `--core-angle` and `--min-bonds` do not apply; `segment_merge_threshold` reports the
threshold (OVITO's `GrainSegmentation.auto_merge_threshold`, a log distance, not an angle). Grains report
`crystallites = 1`; `segment_crystallites` is the cluster count after the cut. Serial from step 2 on, as
OVITO; deterministic for any thread count. Memory: bonds 16 B and graph adjacency ~24 B per crystalline
bond plus a list header per atom. **Measured** on thin film 999 (`defects --segment --seg-mode auto
--reference local`, loaded machine): node pair sampling 19.4 s, bonds + edges + adoption 8.4 s, PTM
(Voronoi order) 56 s; 3:42 wall and 4.1 GB peak RSS against 2:27 and 1.2 GB for the default.

### 15.3 Pre-assigned grains

`rmda defects --grain-column COL` or `--region` expressions skip segmentation (`--segment` splits each
region by it instead). The per-grain orientation is `orient::mean` over the grain's crystalline atoms,
stored per grain, never as a per-atom array read per grain.

### 15.4 Tracking across frames (`analysis/grain_track.rs`)

The previous frame is kept as a sorted (atom id, grain) vector (12 B/atom). Grains of frame t are matched
greedily to grains of t−1 whose atom-id overlap exceeds 50 % of both; otherwise the event is **split**,
**merged** or **new**, and matched grains are **continued**. Stable ids persist through all events
(`stable_grain` in `grains.csv`, `grain_events.csv`, §20). Growth beyond 2× per frame makes a new grain by
design.

**Open:** a centroid + orientation fallback (Hungarian over pairs with disorientation < 5° and centroid
distance < grain radius) for re-numbered atom ids.

### 15.5 Validation

| case | result |
|---|---|
| FCC, BCC (44.6° about z, straddling the cubic zone boundary) and HCP single crystals, thermal | 1 grain, 0 unassigned |
| BCC bicrystals at 8°, 15° and 40° | 2 grains; disorientation within 1° |
| periodic 5-grain BCC Voronoi polycrystal (pairwise > 15°) | 5 grains; < 1 % of interior atoms mislabelled |
| FCC with an intrinsic stacking fault; Σ3 twin | 1 grain; 2 grains at 60.0° |
| `poly3` (§16.8), 3 frames | 8/8/8 grains, correct structures; 0 interior atoms (> 4 Å from a facet) outside their grain |
| `poly3` vs OVITO GrainSegmentation (auto threshold 12.9–13.5°) | ARI vs construction: rmda 0.980–0.981, OVITO 0.976–0.978; rmda vs OVITO 0.991–0.992 |
| thin film `data.lastframe-999` | one grain: the film is epitaxial (11 450 crystallites merge at 5°) |
| thin film, `--mode auto` | 4 grains (1.81 M, 1.52 M, 1.46 M, 0.97 M atoms) 0.2–1.3° apart: the automatic threshold (ln d = 21.44) splits the epitaxial film along low-angle tilt; `defects --seg-mode auto --reference local` V/I 14 017 + 389 / 13 499 + 384 (interior + boundary). Not compared with OVITO (memory) |
| 1 vs 6 threads | identical labels |

**HEA polycrystal** (`reference_structures/lastframe-HEA-poly.data`, 42 432 atoms, 39 % non-crystalline):
both codes find 4 FCC grains; OVITO's automatic threshold is 13.2°.

| rmda settings | ARI, all atoms | ARI, OVITO-crystalline atoms |
|---|---|---|
| defaults (5°, min-bonds 4) | 0.946 | 0.9999 |
| `--threshold 8 --min-bonds 1` / `--threshold 13 --min-bonds 1` | 0.966 / 0.965 | 0.9996 / 0.9997 |
| ceiling: OVITO's pre-adoption labels + rmda's adoption | 0.984 | — |
| `--ptm-order voronoi` (crystallite mode) | 0.947 | 0.9999 |
| `--mode auto --ptm-order distance` | 0.966 | 0.975 |
| **`--mode auto`** (implies `--ptm-order voronoi`) | **0.9999** | **1.0000** |
| `--mode auto` fed OVITO's own PTM columns | 0.9999 | — |

Targets (must): a single crystal at 1000 random orientations (incl. near zone boundaries), σ = 0.1 Å, FCC,
BCC and HCP, is one grain; the 3-lattice polycrystal matches construction at ≥ 99 % of interior atoms
(met); HEA parity ARI ≥ 0.98.

**HEA parity met with `--mode auto`: ARI 0.99993** against OVITO 3.15 GrainSegmentationModifier at its
defaults (GraphClusteringAuto, orphan adoption, min size 100, stacking faults handled): 4 grains of
15 276 / 9 534 / 9 358 / 8 264 atoms (OVITO 15 277 / 9 534 / 9 357 / 8 264), automatic threshold 13.17701
(OVITO 13.17711), 17 050 orphans adopted (OVITO: 17 050 unassigned before adoption). Pinned by
`grain_seg_ovito.rs` against the checked-in OVITO labels `tests/data/hea_poly_ovito_grains.txt`. Both
parts are needed: the algorithm alone, on distance-ordered PTM, gives 0.966 (143 atoms typed OTHER
that OVITO types crystalline split off a 593-atom piece just above the knee); OVITO's PTM neighbour
order alone, in crystallite mode, gives 0.947. The default mode reaches 0.946: its mean-orientation
merge leaves ~1 200 more atoms to adoption, and OVITO's adoption rule alone lifts it only to 0.951.
**Open:** the 1000-orientation sweep.

---

## 16. Defect subsystem (Wigner–Seitz family)

### 16.1 Concepts

- A **reference** is a set of sites. The census runs on a `Reference` trait (global affine lattice or
  patch lattice): nearest site of a position, site position, site enumeration. Sites are **addressed** by
  integer lattice indices (n, basis k), never stored.
- **Occupancy (pure WS).** Every atom of a counted grain is assigned to its nearest site. There is **no
  tolerance and no exclusion of PTM-OTHER atoms**: a split dumbbell is PTM-OTHER, and excluding such atoms
  turns every Frenkel pair into 1 V / 0 I.
  - **Vacancy:** a domain site with no atom. **Interstitial count:** `Σ max(occ − 1, 0)`.
    **Interstitial atoms:** for each site with occ > 1, all but the atom closest to the site.
  - **Invariant, asserted per region class and in total:** `V − I = N_sites − N_atoms`, reported as
    `balance_residual` (must be 0). Defining V and I by different tests makes conservation impossible.
- **Off-lattice diagnostic:** the fraction of atoms more than 0.35 bonds from their site is a quality
  metric, never used to create defects.

### 16.1a Antisites (`defects/antisite.rs`, opt-in `--antisites`)

- **Definition** (OVITO's `per_type_occupancies`): an antisite is a **singly occupied** site whose atom
  is not of the site's species (total occupancy 1, own-type occupancy 0). Vacant and over-occupied sites
  are vacancies and interstitials, never antisites, whatever their occupants' types; a dumbbell of a
  wrong-species atom on a site is one interstitial and no antisite. Counted per class and per pair
  (site type → occupant type), read off the census's own assignment: **V and I cannot change**, and
  without `--antisites` nothing is computed (no type is read, outputs byte-identical).
- **Site species, explicit reference:** the type of the reference atom that is the site. When both
  frames name their types and the names differ in order, current types are matched to the reference's
  by name; otherwise type numbers are compared.
- **Site species, fitted reference (global, patch):** per grain, each motif basis site `k` takes the
  majority type of the occupants of its singly occupied sites (ties to the lower type). This is sound
  because the site key `(n, k)` is one index space over the whole grain: the global lattice addresses
  every site by it, and every patch labels with the global indices, its origin placed so that its
  labels — `k` included — agree with its neighbour's (`patch::place` compares `Site`s, not positions
  modulo centring translations). So every site of one basis index lies on one Bravais sublattice of the
  grain, and a chemical sublattice is recovered whenever it is a union of the motif's basis
  sublattices: B2 on BCC's two (cube corner, body centre), L1₂ and L1₀ on FCC's four. The fit's choice
  of origin or cube axes only relabels `k`, which the vote absorbs.
- **Where it does not hold** (documented, not detected): a chemical cell larger than the motif's
  conventional cell (D0₃/L2₁ on BCC, D0₁₉ on HCP, B1 rocksalt, which PTM sees as SC with one basis
  site) mixes species on one `k`; an antiphase boundary inside a grain puts the minority domain's
  sites on the "wrong" species; a disordered (A2/A1) solid solution has no sublattices. The first two show as a
  large antisite count with the species per basis site (`sublattice_types`) reported beside it. A
  grain without chemical order (Cramér's V of its site–type occupancy below `antisite::ORDER_MIN` and
  no host species, §16.8a) gets no antisite count. A box
  vector snapped to a centring translation (§16.3b) also relabels `k` across the periodic face, which
  for an ordered crystal is exactly an antiphase boundary at the face.
- **Site species, local census** (`defects/local/antisites.rs`): there is no index space over the
  grain (a lattice atom is its own site, a defect atom's site comes from a lattice fitted to its
  neighbours), so `k` is assigned and voted on **per vote region**. The box is cut into cells of
  about 10 bonds (at least two per periodic axis); a cell's region is the cell plus half a cell on
  each side, capped at half the box on a periodic axis so that every member's displacement from the
  region centre is the minimum image whatever the grain's shape. Per region: `T` refined from the
  first shells of up to 48 of its lattice atoms (as a census chunk), the origin fitted to up to 256
  of them nearest the centre first (the census's `fit_frame`), every singly occupied site of the
  region labelled with the basis index of its nearest site of that lattice, and each `k` takes the
  majority occupant type over the region (ties to the lower type). A site is decided by the one
  region whose cell holds it; the margin only adds votes. The singly occupied sites are the
  census's own: a lattice atom on no defect atom's site, a defect atom alone on its site, a region's
  (§16.3d) site with one atom. Unresolved defect atoms are on no lattice site and never
  antisites; a region with too few lattice atoms for a fit decides none of its sites. The fit's
  origin only relabels `k`, so regions need not agree on labels; the limits are those of the fitted
  references, but local: an antiphase boundary mislabels at most the regions it crosses, not the
  grain. An antisite is reported at its region lattice's site; `sublattice_types` is the **sorted**
  species list (one per basis site) of the regions holding most sites, since labels are per region.
  Everything is ordered by position (`place_key`, `place_sample`), never by the file's atom order.
- **Cost:** off, none. On, one more pass over the grain's atoms in the census; the explicit reference
  one pass over the assignment; the local census one fit per vote region and a label per site and
  region it is in (0.05 s for 93k atoms).

Outputs: stdout `defects_antisites`, `defects_boundary_antisites` and, per pair,
`defects_antisites_<occupant>_on_<site>` / `defects_boundary_antisites_…` (type names when every
type has a distinct known name, else numbers); grain table and `grains.csv` columns `antisites`,
`gb_antisites`, `sublattice_types` (−1 withheld for an unmeasured grain);
`summary.csv` `antisites`, `gb_antisites`; series `antisites.csv` (one row per grain and pair);
`antisites.xyz` beside the defect files (occupant species, position, id, type, `site_type`,
`site_species`, `site_pos`, grain, boundary, `site_id` with an explicit reference, and `cluster`);
defect clusters (§16.5) gain antisites: stdout `defects_clustered_antisites`,
`defects_antisite_clusters`, `defects_largest_antisite_cluster`, and series `clusters.csv` an
`antisites` column and the antisite-only clusters as rows after the V/I ones. All columns are
appended at the end and only with `--antisites`; the run signature gains `antisites=true`, so a
series is not resumed across the switch.

### 16.2 The census (`defects/census.rs`)

- **Two-pass assignment:** a bounding-box pass, then (site-grid index, displacement) per atom at 8 B/atom.
- **Counted domain:** the occupied sites plus enclosed empty sites: an empty site joins when at least half
  its first shell + 1 is in the domain, iterated to a fixed point (at most 12 iterations). Sites occupied
  by another grain's atoms are excluded. No site of a compact void has that many held neighbours, so
  then a connected set of empty sites that does not reach past the grid, does not wrap round a periodic
  axis (a gap through the box is two surfaces), stays under 20 000 sites and has no site claimed by or
  beside another grain is a cavity and joins whole. The local census does the same atom by atom (walked
  through each site's local lattice, cap 4000) and within its regions.
- **Site keys on periodic faces** use a face tolerance, so a site exactly on a face cannot wrap to 0 from
  one side and 1 − ε from the other (without it a 12-cell slab shows 231 phantom vacancies).
- **Determinism:** 1 and 6 threads give bit-identical T, origin and counts.

- **Site-index limit:** the site-grid index is `u32`; a grain box of more than `u32::MAX` sites is refused
  (split it with `--region`) rather than wrapped (`check_site_count`).

Per-type (antisite) occupancy: §16.1a.

### 16.3 References

**(a) Explicit reference (OVITO WS)** (`defects/explicit.rs`; `rmda defects --ws-reference REF
[--ws-affine off|to-reference|to-current]`, Python `defects(ws_reference=, ws_affine=)`). The
reference frame's atoms are the sites, in their own k-NN index (`KnnIndex`, built once per run).

- **Assignment** is pure WS, as §16.1: every current atom to its nearest site, no tolerance and no
  cutoff (seating atoms only within a tolerance is not OVITO's rule and breaks conservation); ties go to the lower site index, and the occupant of an over-occupied site is its
  nearest atom (lower atom index on a tie). The census is a direct count over the finite site set; the
  lattice census's grid, domain filling and cavity rules do not apply (every site is in the domain).
- **Affine mapping**, OVITO's `affine_mapping`, default `off` as in OVITO. `to_reference` maps current
  positions into the reference cell through fractional coordinates (OVITO's `ToReference`);
  `to_current` maps the sites into the current cell and builds their index per frame. OVITO refuses
  `ToCurrent` for WS ("not supported by the Wigner-Seitz analysis routine"); here it is allowed,
  and differs from `to_reference` only in which cell's metric decides "nearest". Vacancy positions
  are reported in the current cell when a mapping is on, else at the reference positions. With `off`
  and cells differing by more than 1e-6 relative, the CLI warns.
- **Periodicity** is the reference cell's (`off` and `to_reference`) or the current cell's
  (`to_current`); the two frames must have the same pbc flags (refused otherwise).
- **Grains and classes are properties of the sites, decided once on the reference**: one grain by
  default (every site interior: totals are OVITO's), else `--region` / `--grain-column` / `--segment`
  evaluated on the reference frame (it is defect-free, so its segmentation is the clean one). A site
  is `boundary` within the boundary width of a site of another grain or of none (default width 1.5 ×
  the reference's mean PTM bond). An atom counts in the grain and class of its site, so each class
  conserves with `atoms` = atoms assigned to its sites; atoms on sites of no grain are not counted,
  like atoms of no grain in §16.2. Labelling the current atoms instead (by region or segmentation)
  would let an atom and its site fall in different grains and break per-class conservation.
- **No gates:** nothing is fitted, so every grain is `measured`; `min_atoms` does not apply.
  Structure and bond per grain (cluster cutoff, off-lattice fraction) come from the reference's PTM;
  fit columns are NaN. A series needs no grain tracking (the reference's labels are the stable ids).
- **Outputs are the shared ones** (§16.6) through `DefectsResult`; the defect files add `site_id`
  (the site's reference atom id).
- **Antisites** (§16.1a, `--antisites`, `explicit::run_typed`): a count over the assignment with the
  reference's types, without changing it; `antisites.xyz` carries `site_id`.

**`rmda ws --reference`.** `rmda ws --reference` / Python `ws(reference=)` run on this core through
`defects/explicit_ws.rs` (`ExplicitWs`, a pipeline `Analysis` with one grain), with the `ws` scalar and
property names: `ws_vacancies`, `ws_interstitials`, `ws_interstitial_sites`, `ws_sites`,
`ws_config_mean_displacement`/`_max_displacement` (over every atom), `ws_config_tolerance`
(`--tolerance`, else 1.05 × the median first-neighbour distance of 4096 sampled sites) and
`ws_config_unmatched`, the atoms farther than the tolerance from their site, which are still in the
census. Per atom: `ws_site`, `ws_displacement` (3 per atom), `ws_interstitial`, `ws_occupancy`, and the
`--defects` files. `--cutoff` is accepted, validated and ignored with a warning; the two frames must share
pbc flags. Tests: `tests/ws.rs`, `ws_self_reference.rs` (7 V / 7 I on the 256k HEA pair, 0/0 self).

**(b) Global affine lattice per grain** (`defects/fit.rs`, `defects/lattice.rs`): a reference fitted to
the frame itself, as in Nordlund & Averback and Mason (§31), one per grain. Sites
`x = o + T · (n + b_k)`, n ∈ ℤ³, b_k from the structure's `Motif`, T a full 3×3 (rotation, strain, shear,
c/a). A rigid lattice (rotation, isotropic scale, origin: 7 DOF) cannot fit shear, tetragonality or
c/a; on the thin film's 2.6 %-sheared substrate it produces a moiré of fake Frenkel pairs. `nearest` is
closed-form: fractional coordinates, rounded over the basis candidates.

- **Fit:** least squares on first-shell neighbour vectors, seeded from the symmetry-aligned mean PTM
  orientation and the PTM first-neighbour distance; candidate template frames are scored, not assumed.
  At most 200k crystalline atoms are sampled; neighbour vectors are computed on demand
  (`NeighbourSource`), never stored. The origin comes from the fitted lattice's residuals.
- **Neighbours for the fit: 2z, not z** (`fit::fit_neighbours`; also the patches', the local census's
  chunk `T` and its antisite vote). The `z` nearest of a noisy crystal are its shortest bonds: long
  first-shell bonds lose their rank to short second-shell ones, which the 0.25-bond shell match then
  rejects, so a fit on `z` neighbours comes out short. Free BCC W 24³ block, 40 planted V, true bond
  2.7410 Å: with `z` neighbours the fitted bond is 2.7322 Å at σ 0.09 Å, 2.7143 at 0.12
  (`gate_rms_displacement`), 2.6939 at 0.15 and 2.6847 at 0.18, and a periodic box calls the perfect
  crystal at 0.18 a dislocation (`fit_failed`, 1.85 Å closure offset). With 2z: 2.7407–2.7408 to
  σ 0.15, 2.7441 at 0.18, and global, periodic and local all exactly 40 V / 0 I at σ 0.03–0.18. A
  second-shell vector cannot match: two neighbour vectors of one atom differ by an interatomic
  distance (≥ 1 bond), four times the tolerance. `pair_fraction` keeps its meaning: matched pairs over
  the same-grain atoms among each atom's `z` nearest (a stacking fault's two HCP planes have 9).
  Pinned by `a_noisy_free_block_fits_an_unbiased_lattice`. The local census's chunk `T` uses the same
  rule; with it, a B2 28³ block at σ 0.15 counts equal to the explicit reference atom for atom.
- **Unwrapping and commensurability.** Each periodic axis is either *tiled* by the grain's lattice or
  unwrapped:
  - Covering an axis is not spanning it. A preliminary (unsnapped) fit gives, per covered axis, the
    **closure offset** `r = A − T·nint(T⁻¹A)`: how far the box vector is from a lattice translation. The
    axis is tiled without further work when `|r|` is at most a noise floor, `max(0.1 bond,
    3.5 σ)`, where σ is the per-component RMS of the fitted bond residuals (`pair_rms/√3`). (A large
    Voronoi grain covers an axis without being periodic; snapping it strains the lattice by up to 2 %
    and leaves a plane of phantom vacancies.)
  - Otherwise the grain is **unwrapped by a walk over lattice-coherent bonds**: a level-synchronous
    breadth-first walk over the 14 nearest same-grain neighbours. An axis is spanned only if a walked
    bond closes a loop through a periodic image. The walk runs in 256-atom pieces through
    `Par::for_each`, with a per-frame atom → grain-slot table (4 B/atom). Axes with a gap are cut across
    the largest gap. The walk does not run on the first thin-film frame. On the last frame it runs on
    the damaged substrate (closure offset 0.32 Å against a 0.28 Å floor) and finds no seam; it costs
    2.7 s (`defects.grain` 17.8 s against 15.6 s without the walk).
  - **Self-seam versus fault or dislocation.** With few grains in a periodic box (atomsk), a grain
    meets its own image. Its lattice is not commensurate with the box, so the meeting is a translation
    boundary of the grain with itself, offset by exactly the closure offset `r` (reduced modulo lattice
    translations, with signs and pairwise sums for seams across two box faces). Physically it is a
    grain boundary, and it is treated as one:
    - **Bond classes.** A bond's deviation from its nearest first-shell vector is *seam-like* when it is
      nearer one of the offsets than zero (across their perpendicular bisector) and within 0.35 bonds of
      it. It is *coherent* when it is not seam-like and within thermal noise of the shell vector:
      `max(0.1 bond, 5σ)`, σ the per-component RMS of the fitted bond residuals. Offsets no longer than
      the noise floor `max(0.1 bond, 3.5σ)` are not used.
    - **Why coherent is tight.** A bond that is neither (off the lattice by more than noise, toward no
      closure offset) is not walked. atomsk seams leave atoms 0.4–0.5 Å out of place in such
      directions; with a 0.35-bond coherence tolerance the walk crosses the seam through them, the
      loop closes through the box, and the grain is refused. The leak needs one such bond on a seam,
      so it grows with seam area: on unrelaxed 4-grain W samples such a tolerance refuses no grain at
      100–300 Å, one in four at 500 Å, and all three grains of the 1270 Å, 128M-atom sample.
    - **Seam atoms.** An atom with ≥ 2 seam-like bonds is a seam atom.
    - **The cut.** The walk enters no seam atom and crosses no seam-like bond, so the grain is cut open
      at the physical seam, not wherever the walk's fronts meet. Seam atoms are then attached, layer by
      layer, to the side they bond to by their shortest coherent bond.
    - **After the cut.** The cut axis is fitted and counted as non-periodic, without a snap. The grain's
      own atoms that lie across the seam (not where the layout puts them) count as another grain's for
      the boundary class and for claiming empty sites (§16.4).
    - **Relaxed self-seams.** After MD the seam is gone: the grain strains unevenly (0.1–1 %) so that
      its lattice runs coherently round the box, and the offset spreads over the whole grain (lattice
      phases of a 0 K-minimised 4-grain Fe sample spread continuously over ~1 Å, where the unrelaxed
      one has a few sharp values). No bond deviates by the offset, so there is nothing to cut, and the
      *averaged* `T` misses the box vector by up to half a bond, where rounding to the nearest
      translation can pick the wrong one.
    - **The loop's own translation.** The walk therefore carries each atom's lattice coordinate (the
      sum of the shell steps of the walked bonds) and runs over lattice atoms only (PTM matched the
      grain's structure). A loop that closes once round the box along one axis gives that box
      vector's exact lattice translation; the fit snaps to it, within the 2 % relative tolerance alone
      (`FitInput::translations`), and the patches snap to the same translations so that their labels
      share one index space. The remaining uneven strain is for the gates and, in `auto`, the patch
      reference.
    - **A dislocation** shows as loops on its two sides disagreeing by its Burgers vector (in a relaxed
      300 K Fe sample: 222 loops against 22, differing by a/2[1 -1 1]). The translation most loops give
      is used and the disagreement is reported on stderr; the displacement and phase-agreement gates then
      decide whether one reference can count the grain. A grain the walk gives no loop translation for,
      whose averaged `T` is not within the snap tolerance of the box, is `fit_failed`, with the offset.
    - **Example.** On `poly-small` (4 W grains in a 100 Å box), all four grains meet their own images,
      with offsets 0.91–1.54 Å, and all four are counted. The `few` fixture of
      `crates/rmda-cli/tests/poly3.rs` (3 grains, 64 Å) passes on 15 of 15 seeds, including seams of
      0.36 Å (about 3.5 σ).
  - A spanned axis is **snapped** (`n = nint(T⁻¹ A)`, `T = A n⁻¹`, Mason) only when the snap moves the
    lattice by ≤ 2 % relative **and** ≤ 0.25 bonds absolute.

**(c) Patch-local reference** (`defects/patch.rs`; RMDA's extension of Mason) for large or inhomogeneous
grains (films, bent crystals, gradients):

1. **One index space.** The grain is tiled into cubic patches (`--patch-size`, default 12 bonds ≈ 33 Å).
   Each fits its own `T_p` on ≤ 400 of its own crystalline atoms (≥ 60), snapped like (b) with the 2 %
   rule only. All patches label sites with the **global** lattice's integer indices, so a site has one
   key from any patch: seams cannot create or lose sites, and periodic wrapping stays index arithmetic.
2. **Phase continuity.** Patches are placed breadth-first from the best-populated one; each takes the
   origin that makes its labels agree with its placed neighbour's on the atoms nearest that neighbour.
   The minimum agreement is `phase_agreement`, **gated** at ≥ 0.95: a dislocation breaks the shared index
   space, and the grain is refused rather than miscounted. Placement only compares a patch with its
   parent, so round a dislocation loop wider than a patch the tree's two sides meet a Burgers vector apart
   on a face it never checks, and the loop's content lands there: a 99-vacancy loop with 20 Å patches
   reads 47 V / 6 I at `phase_agreement` 0.96. Every pair of face-adjacent fitted patches is therefore compared
   too, on the atoms of both within a quarter patch of their face; the minimum is `face_agreement`,
   **gated** at ≥ 0.75. A Burgers-vector face agrees near 0; ambiguity next to a loop's core inside the
   patches costs a few percent (0.94–0.99). The **periodic box faces** the lattice tiles are compared
   too (`periodic_face_agreement`, folded into `face_agreement`): the first and last patch along a
   periodic axis meet only through the box, so a lattice-vector drift across it (a tilt or shear
   varying along another axis, every patch snapped to one translation) would land there unchecked and
   read as a plane of vacancies beside a plane of interstitials. Each atom within a quarter patch of the
   upper face is labelled by its patch and, at its image, by the patch across the face; the labels are
   compared as the census keys them (wrapped by `census::canonical_site`). Clean periodic crystals,
   the 200-patch seam test included, read 1.
3. **Modes** (`--reference`): `auto` (default) uses (b) and switches to patches only when (b) fails the
   displacement gates on a grain at least three patches across; `global`, `patch` force a mode. Columns
   `reference`, `n_patches`, `n_patches_fitted`, `phase_agreement`, `face_agreement`.

**Quality gates** (per grain, `DefectsConfig`):

| metric | gate (default) | on failure |
|---|---|---|
| grain size / crystalline fraction | ≥ 100 atoms / ≥ 0.3 | `too_small` / `not_crystalline` |
| fit | succeeds (reason recorded) | `fit_failed` |
| neighbour pairs matched by the lattice | ≥ 0.8 | `gate_pair_fraction` |
| rms displacement | ≤ 0.15 bond | `gate_rms_displacement` |
| off-lattice fraction (> 0.35 bond) | ≤ 2 % | `gate_offlattice` |
| patch phase agreement | ≥ 0.95 | `gate_phase_agreement` |

A gated grain's counts are −1. Every metric (a, T, rms/mean displacement, off-lattice and pair fractions,
phase agreement, sites, atoms, status) is written **whether or not it passes**.

**Open:** Mason's origin-grid validation by defect count; adaptive patch size; a smoothed roto-strain
field (`roto_strain.rs` is unused by the census); `--ws.gates off`. Patches keep the 2 % relative snap rule:
adding the 0.25-bond limit changes the thin film's last frame by 2 V / 2 I, with no ground truth to
choose between them; revisit with a bent-crystal fixture of known counts in a large box.

#### 16.3d Local census (`defects/local.rs`)

The 128M-atom W polycrystal after 1146 cascades has no lattice per grain. Its
grains bend by about 2° round the box; loops round the box give 72 different
translations along one axis and miss the box vector by 44 Å along another. The
patch reference's phase agreement is 0.54. Every grain-wide reference refuses
it, and the initial frame cannot be used as a reference after that much damage.
The local census counts it without a lattice for the grain:

1. **Sites.** A lattice atom's site is where it is. Each off-lattice atom gets
   `T` and an origin fitted to the lattice atoms among its 40 nearest
   neighbours. The origin is seeded from the nearest four on each basis site,
   keeping the seed most atoms agree with; then `T` and the origin are
   least-squares fitted together, seeded by a per-chunk `T` (about 10 bonds on a
   side). Atoms whose sites coincide (within 0.35 bonds) share one.
2. **Empty sites.** Empty sites are looked for in the first two shells of each
   off-lattice atom. They need no atom within 0.7 bonds: next to a dislocation,
   the far side's lattice puts sites between atoms. Empty sites within 0.6
   bonds merge, and they join under the §16.2 domain rule.
3. **Regions.** Inside an interstitial loop the extra plane is perfect crystal
   locally, and only the rim is off the lattice. On the 999-cascade thin film,
   1375 of the global reference's 2051 interstitials in one grain are in
   clusters of 10 or more, and steps 1–2 alone find 421. So clusters of at least
   20 bulk-coordinated off-lattice atoms (12 or more neighbours within 1.2
   bonds, which keeps free surfaces out) are counted whole, within radius `R`:
   - a Wigner–Seitz census of every atom within `R` + 1 bond, against one
     lattice fitted to the lattice atoms between `R` + 1 and `R` + 3 bonds;
   - only if at least 90 % of that shell is on the fitted lattice;
   - regions that would share an atom merge transitively, up to 30 bonds and 6
     rounds;
   - a cluster whose union with the regions it meets is no region (too large,
     or a dislocation between them makes the union's shell incoherent) is a
     **partial region**: its own census, counting only its sites outside every
     accepted region's sphere that hold no accepted region's atom, and only
     atoms no region counted. Sites held by another region's atom are crystal
     for its domain rule and cavity walk; sites merely inside another sphere
     are not (past a free surface they are vacuum, and counting them as
     crystal let the fill creep outward: +1459 vacancies on the cut's faces);
   - anything else (a cluster too large or too incoherent on its own) is
     counted atom by atom and reported as extended defects.
4. **Conservation.** Every grain atom is on exactly one counted site, so both
   classes conserve, and the check is asserted. Unlike the global and patch
   census this holds **by construction** (sites are only made where atoms or
   accepted empty sites are), so `balance_residual = 0` here catches
   bookkeeping bugs, not a wrong count; a vacancy lost at a region's edge, or
   an interstitial loop's content left to `net_vacancies`, conserves too.

A grain `auto` will fall back is not tried at full cost: its setup (structure, orientation, the
neighbour sample) is computed once; the global census is skipped when a 200k-atom sample of its
displacement metrics is clearly over a gate (rms > 1.5×, off-lattice > 2× the limit); the patch census
when the patches already fail phase or face agreement. The counts equal those of running every
reference in full; thin film as one grain: `defects.grain` 30.4 s against 46.9 s.

`--reference auto` falls back to it for any grain the global and patch
references refuse. Limits: the interstitial content of a dislocation network (its
extra planes) cannot be assigned locally. Only the rim of a complex larger than a
region counts, so the interstitial count is a lower bound where extended defects
are reported. The vacancy count is not affected. `net_vacancies` (V − I) reports
the rest: in a closed crystal damaged by Frenkel pairs, the interstitials of
loops and networks the census cannot see, less the vacancies of collapsed
vacancy loops it cannot see. Vacancy loops are hidden the same way: when a
loop's rim joins a cluster too large or too incoherent to be a region (a
dislocation network, or a free surface), its content is counted by neither
class, so the vacancy count is a lower bound there too.

**Performance.** The counts are byte-identical across thread counts. Region and domain loops run in fine-grained pieces
(in `Par`'s 65 536-item chunks the ~10k regions of a grain, and each
domain round, would be one chunk on one thread); region sites are indexed by
slot in a box of lattice cells, not hash maps; the domain rule rechecks only the
last round's neighbours; and another grain's claim is asked only of sites the
rule would take (a query in the vacuum past a free surface is slow).

Region merging dominates what is left, and it is an ordered chain: in damaged
W one set of overlapping clusters spans the grain, and each merge usually needs
the union the previous one made. On the 128M frame (3 grains, all local) the
default `union` run takes 49 min on 80 threads, `--local-merge partial` 7.4 min
(§16.8).

**Speculating the union censuses.** A union census,
`region(sphere, T, None)`, reads none of the merge's state; the state only
decides which spheres are asked for. So the merge walks the next clusters
in order against the current state, accepts provisionally those that meet
no region (exactly what the merge will do), and works out the chain of
union censuses of up to `min(2·threads, 24)` that do meet one, in parallel.
The in-order merge takes a census from this cache when it asks
for the same sphere (bit for bit) and `T`, and counts it itself otherwise,
so the result is the in-order one at any thread count. The first speculated
cluster of a batch is always right. Thin film, local, two regions: 2536
union censuses, 2314 of them discarded because the union meets another
region; one census is 27.6 ms on 1 thread and 21 ms on 12 (or 80),
so the in-order merge is serial however many cores there are. The cache
serves 52 % of the censuses at batch 24 (63 % at 4, 34 % at 160: the
misses meet regions made earlier in the batch). On 12 threads speculation
does not shorten the merge; on 80 it runs on otherwise idle cores and
should halve the union time (unmeasured). Starting a new batch at every
miss is exact too but 2–4× slower on 12 threads (each batch wastes most of
its guesses).

Alternatives measured and not used:

- Resolving sets of overlapping regions in parallel and re-merging their unions
  in a second pass **changes the counts**: the second pass re-merges
  unions and, when the larger union fails, sends all their clusters to the
  atom-by-atom count (thin film 999: regions 1426 → 723, V 14 482 → 13 615;
  128M: V +0.4 %, I +2.9 %).
- Planning batches in parallel against a frozen state and committing in order,
  redoing any plan whose looked-up atoms changed owner: identical counts, no
  gain (30–45 % of plans redone; batch 1 fastest at 42 s vs 46–51 s).
- Parallelising inside each region census (residuals, claims, output) without
  reordering any sum: identical counts, no gain. Each census is ~15 ms of
  memory-bound passes.
- Joining overlapping clusters before counting (one census per union): merge
  time 7.0 → 3.6 s on the cut, but V 3528 → 3274 and I 172 → 128: it
  changes which clusters are regions.

Merging much faster than that needs a different merge rule, which is a
change of method to be validated, not an optimisation: `--local-merge
partial` is such a rule (§16.8).

### 16.4 Region classes

| class | atoms | sites |
|---|---|---|
| `Interior` | atoms of the grain farther than the boundary width (default 1.5 first-neighbour distances) from any atom of another grain or of none | domain sites with the same rule |
| `Boundary` | atoms within the boundary width of another grain or of unassigned atoms | domain sites in that shell |
| `Surface` (target) | atoms within w_surf of the free surface | sites in that shell |
| `Excluded` (target) | user selection (e.g. a substrate) | — |

- **A self-seam is a boundary.** In a grain cut open at a self-seam (§16.3 b), its own atoms across the
  seam count as another grain's: for atom flags, for the empty-site class and for claims.
- **An empty site is classified by its own distance to the boundary**, with the rule that flags atoms.
  Taking the class of its nearest atom puts 2–4 phantom interior vacancies per frame on mismatched-lattice
  boundaries, where that atom can lie just outside the width while the site lies inside it.
- PTM-OTHER atoms are not excluded and do not by themselves create a boundary, so a point defect never
  turns its own neighbourhood into a boundary.
- **The boundary class is bookkeeping, not a physical defect count.** It is counted against the grain's
  own lattice and conserves, but on a boundary between mismatched lattices it holds sites no atom fills
  (up to 81 V per FCC grain in `poly3`, against 1 planted). Interior counts are the physical result.

**Open:** the `Surface` class (adatoms are ordinary occupied sites, surface vacancies fall outside the
domain); `Excluded`; a boundary census against the neighbouring grains' references; dislocation cores as
their own class (needs DXA-like detection).

### 16.5 Defect clusters (`defects/clusters.rs`)

- Vacancies and interstitials are joined within r_c = (1st + 2nd shell)/2 of the bond of each one's
  grain (a pair of two grains within the smaller), or `--cluster-cutoff` for all. Per cluster: V, I, net, centre, grain. Centres are **unwrapped
  along the cluster's own links**, so a cluster across a periodic face centres at the face.
- **Antisites** (`--antisites`, `clusters::attach_antisites`) are added **after** the V/I clusters
  and never change them: same members, ids, centres, V, I, `size` and `net` with and without the
  flag. Each antisite, at its site, joins the cluster of its nearest vacancy or interstitial within
  the cutoff (the smaller of the two grains', ties to the lower defect) and counts in that
  cluster's `antisites`; an antisite never bridges two V/I clusters, and antisites do not chain one
  another onto a V/I cluster. Antisites near no V/I defect are clustered among themselves by the
  same rule into **antisite-only clusters** (V = I = 0, `size` 0, `antisites` their size, centre
  unwrapped along their links), numbered after the V/I clusters. So a swapped pair is one
  antisite-only cluster of 2, and an antisite beside a vacancy is a V cluster with one antisite (in
  B2, the vacancy–antisite complex). Joining antisites into the V/I union-find instead would let one
  antisite merge two V/I clusters and change V/I cluster statistics with the flag.

**Open:** a per-grain cutoff (in a multi-lattice frame the largest grain's bond is wrong for the others);
Rg and class per cluster; a vicinity table (V–I pairs within r_c, di- and tri-vacancies,
interstitial complexes) and a net-defect column.

### 16.6 Outputs

`rmda defects` / `rmda.defects`, per frame: the per-grain table (`--table`: status, reference, structure,
atoms, sites, interior and boundary V and I, `balance_residual`, a, T, displacement metrics, patch
columns); scalars on stdout; `--defects DIR` writes `vacancies.xyz` and `interstitials.xyz` (extxyz with
`Lattice`, `Origin`, `pbc`); time-series tables in §20. With `--antisites`, the antisite outputs of
§16.1a.

### 16.7 Thin films

- Axes are non-periodic where the file says so. Film and substrate come from `--region`, pre-assigned
  grains or segmentation, and each gets its own reference.
- On the thin film the substrate stays global and the film goes to patches in `auto` mode: the film is
  epitaxial (one grain) but carries a lattice rotation gradient about y (−0.8° in the substrate, −1.7° at
  z ≈ 700 Å, −0.1° at the top) that one lattice cannot follow.

**Open:** detecting a declared-periodic axis with a vacuum gap (warn, `--film-axis`); the `Surface` class.

### 16.8 Validation

Tests: `rmda-core/tests/defects_synthetic.rs`, `rmda-cli/tests/defects_cmd.rs`, `rmda-cli/tests/poly3.rs`,
`rmda-cli/tests/ws_reference.rs`. All
counts exact, every case conserves.

| case | result |
|---|---|
| thermal BCC, 5 vacancies + 2 ⟨110⟩ dumbbells | V = 5, I = 2 |
| BCC slab in vacuum, 2.6 % shear + 0.8 % tetragonal | V = 3, I = 0 (a rigid fit fails) |
| free FCC cluster rotated 37°; rotated free HCP cluster | V = 4, I = 0; V = 2, I = 0 |
| BCC bicrystal (21° about z) | interior V = 1 in each grain |
| free BCC block bent by 6° | global gated; patch: all 6 V, 0 I, phase agreement > 0.99 |
| BCC block with a 2 % lattice-parameter gradient | exact |
| **seam test:** thermal 24³ periodic BCC forced into ~200 patches | the 5 planted V, 0 I; same on 1 and 4 threads |
| FCC grain crossed by a stacking fault, `--seg-stacking-faults separate` | three grains, exact |
| synthetic BCC, 16M atoms, one grain | exactly the 1000 planted V |

**3-lattice polycrystal `poly3`** (~20 s in release; `#[ignore]`d). Eight
grains in a periodic 90 Å box (power diagram on a jittered 2×2×2 seed set): 3 FCC (a = 3.615), 3 BCC
(a = 3.165), 2 HCP (a = 2.95, ideal c/a), random orientations with same-lattice pairs > 15° apart,
boundary atoms closer than 0.8 bond thinned, σ = 0.07 Å, 47.9k–49.2k atoms. Each grain has 2–3
vacancies, a divacancy, a dumbbell (BCC ⟨110⟩, FCC ⟨100⟩, HCP ⟨0001⟩) and a di-interstitial > 9 Å from
any facet, plus one vacancy and one dumbbell 0.5–2.5 Å from a facet. Three frames: grain 0 grows and grain
1 shrinks (power weights ±150 Å² per frame), ids stable per site; no grain may touch its own periodic image
(asserted). `RMDA_POLY3_DIR=dir` keeps frames, truth tables and defect-free references in `dir/series`
for `tools/poly3_ovito.py`. With `rmda defects FILES... --segment --out DIR`:

- 8 grains measured per frame; interior V and I equal the planted counts in all 24 grain-frames;
  residual 0; `grain_events.csv` 8 × new then 8 × continued twice; growth monotone in `grains.csv`.
- **OVITO WignerSeitzAnalysis against the true defect-free reference** (which rmda never sees): interior
  V/I equal rmda's in every grain of every frame. In the boundary class OVITO sees exactly the planted
  1 V / 1 I per grain; rmda's boundary class holds 3–81 V (§16.4).
- **Explicit reference (§16.3a) against OVITO, site for site** (`tools/ws_reference_ovito.py`: the
  same vacant site ids, the same extra occupants per site, and rmda's kept occupant is the nearest
  atom): on all three frames against their true references 44 V / 32 I each; frame 0 in a strained
  triclinic box (3 %, −1.5 %, 1 %, xz tilt 2.5 %) with `to-reference` 44 / 32, and with `off` the
  4919 V / 4907 I OVITO also gives; `to-current` 44 / 32. With `--segment` on the reference: 8 grains,
  interior 4–5 V / 3 I each (the deep planted defects), totals 44 / 32, residual 0.
- Pinned in `rmda-cli/tests/ws_reference.rs`: a fixture generated in the test (thermal BCC reference,
  current frame displaced σ = 0.4 Å with 4 atoms removed and 3 inserted at random points; same box,
  open z, strained box with `to-reference` and with `off`) against the OVITO 3.15 site ids in
  `tests/data/ws_reference_ovito.csv` (12 V / 11 I; 227 / 226 for strained `off`); and a planted 16³
  BCC (6 vacancies, a 15-site void, 3 ⟨110⟩ dumbbells, σ 0.05–0.1 Å, a 3.5 %/3 %-shear strained box
  under both mappings): exactly 21 V / 3 I with the planted site ids, one 15-cluster, 10 clusters;
  regions on the reference conserve per class with the same totals.

**Antisites (§16.1a)** (`rmda-cli/tests/antisites.rs`, fixtures generated in the test,
`RMDA_ANTISITE_FIXTURE_DIR=dir` keeps them; `tools/antisites_ovito.py` compares any pair):

- *OVITO 3.15 `WignerSeitzAnalysisModifier(per_type_occupancies=True)`, site for site* (same antisite
  site ids, same occupant type per site, same V/I): a B2 (a = 2.87 Å, 10³ cells, 2000 sites) and an L1₂
  (a = 3.57 Å, 9³ cells, 2916 sites) reference with σ 0.05 Å, against a current frame with 6 vacancies,
  4 dumbbells whose extra atom is the other species, 8 swapped first-neighbour pairs, 5 single
  substitutions, 3 atoms inserted at random points and σ 0.3 Å noise. B2: 6 V / 7 I / 23 antisites
  (12 type 2 on type-1 sites, 11 the reverse; 21 planted, the noise moves two more onto foreign
  sites); L1₂: 7 V / 8 I / 21 (12 / 9). Pinned in `tests/data/antisites_ovito.csv`.
- *Planted, exact, σ 0.08 Å:* B2 14³ under the explicit, global and patch (`--patch-size 13`)
  references, L1₂ 9³ under explicit and global: exactly 6 V / 4 I / 21 antisites with the planted
  pairs, the fitted references recovering the species of every basis site (`sublattice_types`); with
  and without `--antisites` the same stdout prefix, table prefix and byte-identical
  `vacancies.xyz`/`interstitials.xyz`. Species matched by name across extxyz frames that number them
  differently; series `summary.csv`/`antisites.csv` columns; a rerun without the flag refused.
- *Local census:* the planted fixtures above under `--reference local` (2 vote regions per axis):
  exactly 6 V / 4 I / 21 with the planted pairs and the same antisite atoms (ids) as the explicit
  reference, and `sublattice_types` `1 2` / `1 1 1 2`. Against the explicit reference atom for atom
  (`local_census_matches_explicit_reference_atom_for_atom`), 12 V / 8 I / 16 swaps / 10
  substitutions planted: L1₂ 12³ σ 0.15 Å (2 regions per axis), B2 28³ σ 0.15 Å (3), B2 36³
  σ 0.1 Å (4, with margins): 12 V / 8 I / 42 antisites each, the same pairs and ids, in the file's
  order and shuffled. The σ 0.3 Å OVITO fixtures are outside the local census's range (PTM calls
  the B2 not crystalline at the default RMSD cutoff and the L1₂ fails the pair gate).
- *Clusters:* in every planted case above no antisite is within the cutoff of a V/I defect (they are
  > 2 cells apart): 0 clustered, 13 antisite-only clusters (8 swapped pairs, 5 single), largest 2,
  and `antisites.xyz` `cluster` ids after the V/I ones; series `clusters.csv` V/I rows are the plain
  run's rows plus the `antisites` column. Unit test: one antisite beside a divacancy joins it, a
  pair and a lone one far away make two antisite-only clusters, the V/I clusters unchanged.

**Thin film** (`--region 'z <= 500' --region 'z > 500'`, auto reference):

| | `data.firstframe` | `data.lastframe-999` |
|---|---|---|
| substrate (global) | 0 V / 0 I, rms 0.085 Å | 2388 V / 2052 I (1 V in the boundary class); shear `t_xz` = −0.081 |
| film (patch, ~2250 patches) | 0 V / 0 I, rms 0.123 Å (global: 0.52 Å, gated) | 12 963 V / 13 277 I, rms 0.176 Å, phase agreement 0.971, face agreement 0.973 |
| `defects.grain` stage | 14.1 s | 15.5 s |

- The clean first frame is the control: ~2200 patch seams add no defects, and its 2232 patches agree
  on every face (1.000).
- Independent checks of the last frame: five separately fitted 100 Å substrate slabs give 2328 V / 1960 I,
  22 hand-made 100 Å film slabs ≈ 12.6k V / 13.1k I (both within ~3 %). The neighbour-pair residual is the
  same in both frames (0.10 Å), so the larger last-frame rms is inhomogeneous deformation, not temperature.
- Forced patches on the substrate give phase agreement 0.61, consistent with dislocation loops (the
  largest clusters are substrate interstitial clusters of 99, 92 and 87); auto stays global there.
- OVITO WS against `data.firstframe` is not a valid comparison for this pair: the ~2.5 % shear between the
  frames is not a box deformation, so its affine mapping cannot remove it. `rmda defects
  data.lastframe-999 --ws-reference data.firstframe` (identical cells, so no mapping applies) gives
  250 065 V / 250 059 I, residual 0: 34 s and 1.18 GB peak RSS on 12 threads, both frames read.
- Vacancies concentrate at the y = 0 box face (194 within 0.5 Å of the y face, ~12 expected);
  they are genuine empty sites (no atom within 0.8 Å), a property of the input data, not of the census.
  Shift test: with x and y shifted by half a box, the excess stays with the atoms (392 within 0.5 Å of
  the original y-face position, against 399 unshifted; 109 at the new box face, against 105 at that
  mid-box position unshifted), and the totals move by 2–3 per grain.

**Local census: accuracy** (`defects_cmd.rs` tests; statistics from scripts over 8 noise seeds,
Gaussian σ per component).

- *Thermal false positives:* a perfect free 20³ W block has 0 V / 0 I under local, patch and global at
  σ = 0.05–0.15 Å in all seeds (global is gated at 0.15, patch once).
- *Vacancies without a loop* (8 planted): exact for all three at σ = 0.03–0.10 Å.
- *Next to an interstitial loop* (a[001] loop of 99 atoms in a free 20³ block, built from the loop's
  solid-angle field `u = b·Ω/4π`, so its inside is perfect crystal; 8 vacancies on the rim, inside the
  loop above and below the disc, just outside and far away):

  | σ (Å) | patch | local: mean pairs off, exact seeds | global |
  |---|---|---|---|
  | 0.03 | exact, 8/8 | 0, 8/8 | 0.38, 5/8 |
  | 0.05 | exact, 8/8 | 0, 8/8 | 1.12, 2/8 |
  | 0.08 | exact, 8/8 | 0, 8/8 | gated 7/8 |
  | 0.10 | exact, 8/8 | 0.12, 7/8 | gated 8/8 |

  Global's errors are whole Frenkel pairs (|ΔV| = |ΔI|; V − I stays exact) inside the loop next to
  the disc, where the crystal is displaced by up to b/2 from the grain's lattice, so two sites are
  nearly equally near. Patches follow the field. So does a local region: an atom more than 0.3 bonds
  from its nearest site takes the site its neighbours' sites and bonds agree on, resolved outward from
  the unambiguous atoms (`--resolve-ambiguous` does the same for global and patch, §16.8a).
- *Order of atoms in the file.* Every reference counts a permuted file the same
  (`every_reference_counts_the_same_whatever_the_order_of_atoms_in_the_file`). Defect atoms are taken
  in position order (chunk, then coordinates), chunks are fitted on ~48 lattice atoms spread over
  them, and fit samples are chosen by a hash of the position and summed in position order
  (`place_sample`). Following the file instead (regions accepted in the order of their lowest atom
  index, chunks fitted on their first 48 lattice atoms, every k-th atom sampled) makes the local count
  on the 0.9M-atom cut of the 128M frame depend on the order (V/I 3597/204, 3880/423 and 3753/278 for
  its file order, a Morton order and a shuffle) and loses whole loops in shuffled files (57 of 128
  loop runs exact). Seeding chunks from the 48 lattice atoms nearest their centre is worse too: a loop
  at a chunk's centre biases its fit (R = 3 vacancy loops 53–58 of 62).
- *Against the validated references.* On the thin film, counted locally, the substrate is 2387 V /
  2040 I against global's 2388 / 2052, and the whole frame as one grain 15 347 / 15 334 against
  `auto`'s 15 351 / 15 329. On grain 0 of the 884 751-atom cut of the 128M frame (`--segment`), local
  counts 4388 V / 474 I and global 4300 / 382; there global's own accuracy is untested.
- *Partial regions.* Most clusters local cannot count as regions are clusters that are regions on
  their own but overlap an accepted region whose union with them is not one (`merge_failed`). They
  count as partial regions (§16.3d); left to the atom-by-atom count, they lose most of the film's
  and the substrate's interstitial content in loops (substrate I about half of global's).
- *`--local-merge partial`* (`LocalMerge::Partial`, `local::accept_partial`) skips the union and
  makes an overlapping cluster a partial region at once. A cluster's outcome then depends only on
  the earlier clusters within reach (radius + 5 bonds each), so clusters are sorted into levels and
  each level is counted in parallel against the levels before; the result is the sequential one at
  any thread count. It loses loops larger than one cluster: thin-film substrate I 2004 against
  `union`'s 2040 (global 2052), at 48 s against 151 s for the frame on 12 threads.
- *Sensitivity to the constants* (the whole thin-film frame as one grain, and the 0.9M-atom cut of the
  128M frame, local reference; defaults in bold). The figures come from a version of the local census
  without partial regions or the joins of §16.8a; they show the spread over the constants, not
  current counts:

  | variant | thin film V / I | cut V / I |
  |---|---|---|
  | **defaults** | **14 482 / 14 104** | **3528 / 172** |
  | `REGION_MAX` 20, 40 bonds (**30**) | 14 693 / 14 075; 14 477 / 14 104 | 3378 / 171; 3623 / 258 |
  | `REGION_MIN` 10, 40 atoms (**20**) | 13 873 / 14 138; 14 643 / 14 258 | 3600 / 201; 3593 / 257 |
  | `COORD_MIN` 11, 13 (**12**) | 14 356 / 14 115; 13 061 / 14 011 | 3239 / 202; 3455 / 145 |
  | `SHELL_COHERENT` 0.80, 0.95 (**0.90**) | 14 482 / 14 104; 14 486 / 14 092 | unchanged |
  | `MERGE_ROUNDS` 2, 12 (**6**) | 14 492 / 14 075; unchanged | unchanged |
  | no regions | 12 413 / 13 525 | 3175 / 109 |

  The interstitial count of the thin film moves by ≤ 1.1 % over the variants; the vacancy count by
  −4 to +1.5 %, and −10 % at `COORD_MIN` 13. On the cut, V moves by −8 to +3 % and I, a small count
  held mostly by the network, by −16 to +50 %. Nothing marks a default as wrong.

- *Vacancy clusters (voids).* A compact void of 15 vacancies (a site, its first and second shells) in a
  periodic 20³ W block, and voids of 55, 89 and 169, are exact under all three references at σ = 0.05
  and 0.10 Å with the cavity test (§16.2). The domain rule alone gives 0, 15, 12 and 0: it keeps the
  vacuum past a free surface out, and no site of a compact void has 5 held first neighbours, so the
  fill never starts. Periodic slabs under vacuum gaps of 4–40 Å keep their 3 planted vacancies; a
  surface pit open to the vacuum beside a void of 15 counts the void and nothing deep in the pit (its
  rim sites with 5 held neighbours are counted).
- *The film's vacancy clusters are collapsed vacancy loops, counted consistently.* The film holds
  dozens of clusters of ~30 vacancies with no interstitials. For four of the largest, a 60 Å cube round
  the cluster was cut out (free surfaces, ~13k atoms) and counted within 12 Å of the centre:

  | cube | global | patch | local | OVITO WS, perfect reference | sites − atoms | DXA |
  |---|---|---|---|---|---|---|
  | 0 | 33 V / 0 I | 32 / 0 | 32 / 0 | 33 / 0 | 13 | closed loop, b = ½[111], 57 Å |
  | 1 | 32 / 1 | 32 / 0 | 32 / 0 | 32 / 0 | 13 | closed loop, ½[1̄11]-type, 51 Å |
  | 2 | 34 / 0 | 34 / 0 | 34 / 0 | 34 / 0 | 14 | closed loop, ½⟨111⟩, 53 Å |
  | 3 | 29 / 0 | 29 / 0 | 29 / 0 | 29 / 0 | 12 | closed loop, ½⟨111⟩, 65 Å |

  The OVITO reference is a perfect BCC lattice with the cube's global `T` and an origin fitted
  separately on atoms > 18 Å from the centre (`WignerSeitzAnalysisModifier`); at 18 and 22 Å all four
  give the same 34–35 V / 1–4 I. So on one lattice every reference and OVITO count the same sites. A
  closed ½⟨111⟩ loop of 51–65 Å circumference holds A·(b·n)/Ω ≈ 35–45 vacancies (r ≈ 9 Å,
  Ω = 16.4 Å³, {110}–{111} habit), as WS says. The density deficit (sites − atoms, 10–19 within
  12–22 Å) is not a vacancy count: a collapsed loop's negative relaxation volume pulls the surrounding
  crystal in, so fewer atoms are missing from any sphere than the loop holds. The first frame gives
  no reference for these clusters: matched by atom id, the shell's atoms moved by a median 4 Å over
  999 cascades, far from affine.
- *Patch: exact or refused.* Placement compares a patch only with its parent, so a loop wider than a
  patch can land its content on an unchecked face (§16.3c); the face gate catches it. Over vacancy
  loops of 59, 99 and 151 and the 99-atom interstitial loop, patch sizes 12–33 Å and 4 seeds each
  (80 runs), patch is exact in 24 and refuses the grain in 56, and is never wrong. With the default
  33 Å patches the 99-vacancy loop is exact (102 V / 0 I) in all seeds.
- *A hot loop's rim.* At σ = 0.08 Å the rim of a collapsed 99-vacancy loop falls apart into clusters
  under `REGION_MIN`: an atom is bulk-like with 12 neighbours within `LINK` = 1.2 bonds (3.29 Å), and
  W's second shell is 3.165 Å, so thermal motion moves second neighbours across it. The fragment
  linking of §16.8a recovers it. A thermally robust density test instead (link and density at
  1.4 bonds) makes the isolated loop exact to σ 0.10, but in cascade damage more atoms are bulk-like,
  more damage joins network-size clusters that cannot be regions, and the atoms counted one by one in
  the film rise from 36 771 to 65 033, moving local away from both validated references.
- *Carving regions out of the left-over defects* (not used). Spheres seeded on 6-bond cells of the
  defect atoms left after the normal regions, at radii of 4–18 bonds, counted as regions only if
  their shell closes (every bond between two shell lattice atoms joining sites that differ by the bond
  rounded to a lattice vector: a Burgers circuit), give 94, 93 and 103 V on three hot 99-vacancy loops
  (truth 102): no gain, and one false vacancy. Of 1436 spheres reaching the test every one reads 0
  inconsistent bonds: spheres cut through the rim already fail on coherence, and the accepted ones are
  mostly empty spheres by the free surface. No seed lies at a loop's centre (its inside is perfect
  crystal, without defect atoms), and a sphere large enough to hold the loop from a seed on its rim
  reaches the block's surface. Counting loops inside networks needs the loops found first, as closed
  rings of dislocation core (dislocation analysis), with regions centred on them.

**What this means for quoting local counts.** Where global or patch pass their gates, use them
(`--reference auto` does). On the thin film a local count is within 0.1 % (V) and 0.6 % (I) of the
validated references; the interstitial count is a lower bound where a loop's rim joins a cluster too
large or incoherent even on its own (`extended` in the output), and `net_vacancies` holds the rest.

**The 128M frame** (80 threads, `--segment`, auto: all three grains are refused by global and patch
and counted locally):

| local merge | V | I | wall |
|---|---|---|---|
| `union` (default) | 582 331 | 66 541 | 49 min |
| `partial` | 581 117 | 66 667 | 7.4 min |

The two rules agree to 0.2 % here. Peak memory is about 210–225 B/atom (27–29 GB). The in-order
union merge (§16.3d) dominates `union`'s wall time; the serial stages dominate both at 80 threads.

**HEA polycrystal** (§15.5): under the global and patch references no grain is measured. Grains 0
and 2 are cut open at a self-seam (grain 0 along x and y, grain 2 along z); every grain fails the
displacement gates (grain 2: rms 0.52 Å, off-lattice 7.3 %; grain 3: 0.49 Å, 6.9 %). This
39 %-disordered sample is beyond a single affine lattice; the grains (50–80 Å) are too small for
patches, and no defect-free reference exists for an OVITO WS comparison.

**Atomsk W polycrystal with few grains** (`poly-small.xyz`, 67k atoms, `--cell 100 100 100 --pbc true
--segment`; every grain meets its own image, offsets 0.95–1.39 Å; refusing the self-seam instead
leaves three grains `fit_failed`):

| | result |
|---|---|
| clean | all 4 measured, interior 0 V / 0 I; boundary 33 V / 175 I; seams of 100–654 atoms |
| 20 random vacancies (`poly-small-20-vac`) | all 4 measured, interior 13 V / 0 I; boundary 39 V / 178 I (6 V above the clean run); 19 of 20 vacancies seen |
| wall, `--profile` | 2.0–2.1 s (`defects.grain` 0.82–0.86 s) |

**Atomsk W polycrystals, 4 grains, unrelaxed, periodic** (a = 3.165 Å):

| box | atoms | result |
|---|---|---|
| 100–300 Å | 0.06–1.7M | 4 of 4 measured, 0 / 0 |
| 500 Å | 7.9M | 4 of 4, cut along xy, interior 0 V / 0 I; 107 s, 1.2 GB (a 0.35-bond coherence tolerance leaks across grain 0's y seam, 0.91 Å off) |
| 500 Å, 100 random vacancies | | all 100 found: 95 interior (each with a full shell of 8 non-lattice atoms), 5 boundary class, 0 missed, 0 false interior |

**Relaxed 4-grain BCC Fe** (atomsk, 150 Å, Mendelev EAM; overlaps deleted, CG-minimised, then 5 ps NPT
at 300 K). Without loop translations (§16.3b) two grains of each relaxed frame are `fit_failed`
(closure offsets 1.18, 1.24 Å):

| frame | result |
|---|---|
| unrelaxed | 4 measured, 0 / 0 |
| 0 K minimised | 2 measured; 2 snapped, withheld by `gate_phase_agreement` (patch, 0.935 / 0.943) |
| 300 K | 3 measured (one by patches, phase 0.990); 1 withheld (phase 0.949, a dislocation from the relaxed seam) |
| 300 K, 100 random vacancies | in the 3 measured grains 60 of 67 planted are interior vacancies at the planted site, the rest boundary class; 0 interior false positives |

The 128M-atom W frame (MD, timestep 19500, 283 K; 3 grains, closure offsets 0.74–1.53 Å, no seam
cut) is this relaxed case.

Targets (must): exact V and I on planted defects, ≤ 1 spurious defect per 10⁶ atoms on thermal cases,
conservation, and global and patch modes agreeing exactly on perfect and planted crystals.

**Open:**

- Dislocated grains (the HEA) need a reference that carries lattice discontinuities.
- Thermal false-positive statistics: 0 interior false positives over ~88k interior-atom-frames at
  σ = 0.07 Å is not enough for "≤ 1 per 10⁶".
- Missing cases: crowdions, 4–64-site vacancy clusters, SIA loops; HCP Zr with fitted c/a; slabs with
  adatoms and surface vacancies; bicrystals with defects at 0, 1, 2 and 5 a from the GB; a bent crystal
  with known counts in a large box; a thin-film differential against OVITO WS on a frame pair without
  shear.

#### 16.8a Ground truth, net content and plane balance

Three checks that need no reference lattice and no other code's answer, in `tools/`:

- **`gen_ground_truth.py` / `eval_ground_truth.py`.** A free-standing BCC W block holding defects
  whose content is known by construction: prismatic loops (b = a<001> on {001}, b = a/2<111> on
  {111}, interstitial and vacancy; one period of lattice planes inserted or removed in a disc, the
  crystal displaced by the loop's solid-angle field, as `dislocation_loop` in the tests), voids,
  vacancies, divacancies and <110> dumbbells, features placed close (4 Å) so their clusters meet,
  an optional bend (no single lattice fits) and thermal noise. The evaluator assigns every V and I
  found to the nearest feature and reports the error per kind and the spurious defects.
- **`net_content.py`.** V − I inside a closed band of crystal from integer lattice labels walked
  over its bonds: lattice sites within the band's middle (the labels' fitted map to positions) −
  atoms on them. The walk starts on the nearest valid lattice site (a start point off the lattice,
  stepped by first-shell vectors, keeps the wrong parity, so no enclosed site is blocked and the
  flood fill leaks), and each atom is counted with its own site (counting atoms by position but
  sites by label biases the net by 10–20 in 30 Å shells). Exact whatever the inside; a band a
  dislocation crosses (the labels do not close) is refused. Equal to the truth on 37 of 37 usable
  shells round the features of a 250k-atom block, and 23 of 23 on BCC and FCC blocks. Blind to
  Frenkel pairs (they do not change V − I).
- **`plane_balance.py`.** Between two frames of one crystal whose lower part has not moved, the
  change of the atom count below the k-th gap between lattice planes is −Δ(V − I) below it. Atoms
  near a cut are split arbitrarily; their number bounds the comparison.

**Results** (error against the truth; FP = Frenkel pairs; `tests/ground_truth.rs` runs both
generators with fixed seeds and fails if any kind's error under global, local or partial grows):

| case | global | local |
|---|---|---|
| isolated a/2<111> interstitial loops, 251–358 I | +13 to +41 FP (exact with `--resolve-ambiguous`) | exact |
| isolated a/2<111> vacancy loops of 54 | — | exact |
| 250k block, 6 loops, voids, points (a<001>) | +4 FP in one loop | exact |
| 432k periodic block (the test) | — | exact, 49 of 49 features |
| 2M free block, 40 loops 4 Å apart, σ 0.08: interstitial loops (2010 I) / vacancy loops (2619 V) / voids | +67 FP / +1 / — | −12 / +2 / 0 |
| same, σ 0.045 | — | 0 / +5 / 0 |
| 2M periodic block, σ 0.045 (no surface) | +22 FP (+7 with `--resolve-ambiguous`) | 0 / +6 / 0 (`partial`: −17 V) |
| 2M free block bent 8°, σ 0.045: interstitial / vacancy loops / voids | refused | −17 / −36 / −254, 103 spurious V (below) |
| thin film 999, substrate, plane balance (net V − I) | matches (median 0, 312 of 312 gaps) | within the ambiguity at 267 of 312 gaps (median +11) |

**How the local census handles crowded loops.** Found on the ground truth and checked on the
references:

1. *A loop whose rim is split into clusters that each fail alone* (its sphere holds an arc; its
   shell crosses the loop's disc). Left to the atom-by-atom count, which sees only the core, such a
   loop loses its content (a 216-atom loop counts none of it). So a cluster with no region alone is
   **joined** to the accepted regions and failed clusters whose spheres come within 2 bonds of its
   own, to a fixed point as a union is; only what is left goes atom by atom. Clusters still pending
   after the merge are joined among themselves (near ones grouped), and a group that is a whole
   region is counted as a partial region round the regions accepted.
2. *A large loop whose own region fails* because its shell, a bond outside the rim, lies in its
   strongest strain: its sphere is **grown** by 1–3 bonds (the strain falls off as 1/d²).
3. *A collapsed vacancy loop's rim falls apart.* Its atoms have 7–13 neighbours within the link
   distance; links need 12 at both ends (`COORD_MIN`, which keeps free surfaces out), so the rim
   breaks into clusters below `REGION_MIN` and the loop's content is lost (an isolated a/2<111>
   loop of 54 vacancies would count 14). So the atoms of such **fragments** are linked again among
   themselves at 11 neighbours (`COORD_LOW`; perfect BCC surfaces have 9, 10 and 7/10 on {001},
   {110}, {111}; FCC and HCP free surfaces at most 9: {111} 9, {100} 8, {110} 7, HCP basal 9);
   clusters already large enough are left as they are (linking every atom at 11 joins a loop of
   the thin film's substrate to its neighbours into one cluster too large, and costs 70
   interstitials there). A cluster linked from fragments, like a join, must have a full shell: a
   free face cut at a random angle has atoms with 11–13 neighbours too.
4. *A join reaching past a free surface* would count the vacuum: a shell half in vacuum is coherent
   over the atoms it has (one such join at the face of the 128M cut gives +786 V / +604 I). A join,
   and a cluster linked from fragments, needs a **full shell**: as many grain atoms as its sites
   predict.
5. *A region holding part of a void.* Fragments of a void's surface can make regions whose sites
   inside the void reach the region's edge and go uncounted (HCP free block: −73 V). A region reports
   the empty sites it leaves uncounted (`open_sites`), and a join or a cluster linked from fragments
   may leave at most 2.

The `partial` rule joins too, before its levels (failed clusters near each other as one group,
grown when alone), so it stays exact at any thread count; it cannot join a failed cluster to a
region.

**Already-counted sites on periodic axes.** The atom-by-atom count asks a grid of the accepted
regions' spheres whether a site is already counted. On a periodic axis the grid's keys are wrapped
exactly as its lookup wraps cell indices (modulo ⌊L / cell⌋, cells ~150 Å); otherwise a region
centred past the last whole cell is never found and its sites are counted twice (Frank vacancy
loops +40 to +92 of 217). `tests/ground_truth.rs` has the Frank block that pins it.

**FCC and HCP.** `gen_ground_truth.py --structure fcc|hcp` builds Ni (a = 3.52 Å) and Zr
(a = 3.232, c = 5.147 Å) blocks with each structure's perfect prismatic loops — FCC b = a/2<110> on
{110} and b = a<001>; HCP b = c[0001] (basal) and b = a/3<11-20> (prismatic) — by the same recipe
(one period of lattice planes normal to b inserted or removed in a disc, the solid-angle field), and
<100> (FCC) or [2-1-10]/[0001] (HCP) dumbbells. In isolation every reference counts every feature
exactly (HCP global: +7 Frenkel pairs in one interstitial loop). Crowded (10 loops 4 Å apart,
features 4–8 Å from loops, σ 0.045 Å):

| block | local | global |
|---|---|---|
| FCC periodic, 863k atoms (loops 485 V / 387 I) | exact, 49/49 features (`partial`: −1 V) | +3 FP |
| FCC free | exact, 49/49 | +3 FP |
| HCP periodic, 319k atoms (loops 229 V / 294 I) | −2 V | +30 FP |
| HCP free | −1 V | +29 FP |

`--resolve-ambiguous` makes global exact on the FCC and HCP periodic blocks (patch too).

**Frank loops.** `--loop-types frank` (FCC) builds faulted Frank loops: b = a/3<111>, one (111) layer
removed in a disc (intrinsic fault) or inserted in the third stacking position (extrinsic fault),
the solid-angle field; the truth is the atoms removed or inserted, so any Frenkel pair inside the
fault is a counting error. Global and patch count them exactly; local is within 1 (0, +1, +1 on three
seeds). Not covered: stacking-fault tetrahedra in the generator (see the Al frame below).

**`--resolve-ambiguous`.** Global and patch are Wigner–Seitz by default (every atom to its nearest
site, OVITO's count), which puts spurious Frenkel pairs inside large loops where the crystal sits
nearly b/2 off the lattice. With `--resolve-ambiguous` (`DefectsConfig::resolve_ambiguous`, Python
`resolve_ambiguous=True`) an atom more than 0.3 bonds from its nearest site takes the site its
neighbours point to (each neighbour on a site votes for the site its bond points to from there; most
votes, then the shorter bonds, then the lower key; resolved outward from the unambiguous atoms), as
the local census's regions do. The ground-truth effect is in the tables above. On real data the
effect is small and its truth unknown: the thin film's substrate 2402 V / 2066 I against 2388 / 2052
(the same net), the 128M cut under `--segment` 4294 / 370 against 4300 / 382.

**The HEA.** `--species 5 --distortion σ --lattice-a 3.59` makes the block a random equiatomic
CoCrFeMnNi with a static displacement per atom (the size mismatch) on top of the thermal noise. A
free and a periodic 500k-atom block (σ 0.06 + 0.06 Å, all three FCC loop families, voids, point
defects, crowded): local exact or −1 V; global and auto +10/+11 Frenkel pairs in interstitial loops,
+2 with `--resolve-ambiguous`; patch +2 (free) or refused by its phase gate (periodic, agreement
0.886, where auto takes global). The counts do not depend on the species. With no chemical order
each motif site's "species" is a chance majority (counting antisites regardless gives 398 310 on
this block, 80 % of the atoms), so a grain's antisites need chemical order — Cramér's V of its
(motif site, type) occupancy at least 0.3 (`antisite::ORDER_MIN`; random ~0.003, B2 with 10 %
antisites ~0.8) — or a host species on every motif site (its majority type holding at least 75 %,
`SHARE_MIN`: a dilute solid solution, W with 11 % Mo in the parity test, keeps its antisites); the
local census averages both over its vote regions. Otherwise they are not counted, a line on stderr
says why, and `defects_antisites` is `nan` when no grain is counted. `tests/ground_truth.rs` has the
HEA block.

**Massively overlapping cascades: defects stored as planes.** 75 frames (Al, Co, Cr, Cu, Fe, Hf,
Mo, Ta, Ti, V, W, Zr; BCC, FCC, HCP; several potentials; ~500k atoms, each a periodic single crystal
of whole cells that kept its atoms, so V − I is exactly 0). `auto` counts 21 globally (V − I = 0, by
construction) and 54 locally, where V − I is off by up to ~2500 (Σ|V − I| 64 073 against
Σ V 101 252): BCC and FCC lacking interstitials (Fe-Zhou-1: 2333 V, 51 I), HCP Ti/Zr lacking
vacancies. In coherent crystal the local census is right — 37 random 20 Å spheres of the Al and Fe
frames, Wigner–Seitz on the exact local lattice (fixed by the crystalline atoms around, labelled by
walking bonds): 438 V / 57 I against local's 434 / 55, Fe 388 / 0 against 387 / 0. The missing
content is stored as whole planes: in Fe-Zhou-1, ~1200–1500 of global's 2255 interstitials lie within
a few Å of one plane spanning the box (181 × 112 Å), an extra plane between two dislocations that
run across the box — interstitial loops grown and joined. Inside such a strip the crystal is
locally perfect, so no local lattice sees it; only the crystal's one lattice does.

`--reference periodic` counts a grain whose lattice is commensurate with the box on every axis
against that lattice, with `--resolve-ambiguous`, past the displacement gates (the pair-fraction
gate still applies): Fe-Zhou-1 2255 V / 2255 I, Al 3220 / 3220, Ti-tabgap-1 2454 / 2454. Its
vacancies are within 10 % of local's in 63 of the 74 frames it counts; HCP Ti/Zr are the
exceptions (+36 to +268 %), the collapsed planes local misses. Other grains are counted as `auto`
counts them.

A box-spanning dislocation makes the loops the walk closes round the box disagree by its Burgers
vector, and in six frames by a whole plane: snapping to the translation most loops give builds a
lattice one plane larger or smaller than the crystal, and V − I comes out exactly one plane
(Cu-tabgap-3 +5000 = 2·50², Fe-Zhou-2 +3969 = 63², Ti/Zr −6160 or −7040). So for a grain that is
the whole periodic box, the candidates are chosen so that the lattice holds as many sites as there
are atoms (|det N| · basis = atoms; ties to the most loops): the crystal the atoms were cut from.
Five of the six frames then count V = I (Cu-tabgap-3 2155 / 2155 against local's 2152 V). In
Ti-tabgap-2 every x loop gives 69 cells where the atoms fill 70, and no candidate conserves them;
`periodic` refuses it ("no lattice through it has as many sites as atoms") and counts it as `auto`
does. All 75 frames: 74 periodic with V − I = 0, one fallback; 6.2 s a frame. `auto` does not switch
to `periodic` itself, but a periodic single grain it counts locally with |V − I| > 50 and > 20 % of V
or I gets a warning that names `periodic`. In coherent spheres periodic's interstitials are not all
local ones (Fe: 583 against 0): they are the extra plane's atoms, which the crystal's lattice counts
and a local lattice cannot. For a polycrystal (the 128M frame) no grain is commensurate with the box;
the local census's V ≫ I there may hold the same blind spot, which needs Burgers-vector analysis of
the dislocations bounding such planes to count.

**Stacking-fault tetrahedra.** The Al reference frame (the cascade set's Al-tabgap-1) holds 28
clusters of ≥ 12 HCP-coordinated atoms: 14 planar (faults, Frank loops), 6 compact (stacking-fault
tetrahedra) and 8 between. `net_content.py` gives an exact count that does not walk through the
faults. On the Al frame (30 Å shells, 20 of 30 usable, the rest crossed by dislocations): local
equals the exact net in 16 of 16 scored shells, periodic in 10 of 10 (the misses are within a bond
of the boundary, or, for periodic, a shell an extra-plane strip crosses). Round the clusters a band
clears (6 of 14; the others sit on dislocations that cross every band), at 3, 6 and 10 Å beyond
each: the two tetrahedra, local within 1 of the exact net in all 6 shells (21–53); the mixed
clusters, local within 1 in 2 of 6 (cluster 23: 139 exact, local 140), −3 in one (cluster 42: 65,
local 62) and +5 to +9 over 25–73 in cluster 35's three. Periodic is exact round the tetrahedra
where no strip passes and off by a plane's content where one does (cluster 23: −118; cluster 35:
−33 to −101). Local counts the tetrahedra right; cluster 35 is the one open miss. The frame's excess
under local (V − I ≈ +1800) is not in these: the shells that can be checked hold none of it,
consistent with its being stored in planes.

**Large loops crowded together: strained shells.** OVITO's DXA on the 128M W frame (its lattice
dislocations only: 89 % of its line length lies in the 8 % of the volume at grain boundaries, which
DXA handles poorly) gives 525 closed loops inside grains. Over each, the census's V − I in a 4 Å slab
against the content a prismatic loop of its b and area stores: 80–115 % for loops of radius
10–40 Å, ~20 % for the five over 40 Å (~5000 atoms). An isolated loop that size is counted exactly
(W block, R 41–54 Å). Crowded ones are not: in a block of 8 loops of R 38–54 Å 3 Å apart, with point
defects and voids 4–8 Å from their rims (3.5M atoms), two things defeat a region: the loops' rims
percolate into one cluster spanning the box (84k defect atoms), counted atom by atom; and each
loop's own cluster fails as a region because its neighbours strain its shell past one affine
lattice (60–89 % of the shell on it, against `SHELL_COHERENT` 0.9). Lowering the gate to 0.6 is far
worse (affine sites in a 70 Å strained sphere are wrong; loops come out inverted), and so is
labelling the whole region by a bond walk (a front crossing a collapsed loop's disc labels the far
side a Burgers vector off). So a shell with 50–90 % on the affine lattice is labelled by walking its
own bonds only (`walk_labels`; the walk stays in the shell, and must close: ≤ 1 % of its bonds
disagree, ≥ 95 % of its lattice atoms reached), a quadratic correction to the lattice is fitted to
the labels (`fit_curvature`), and the region is counted on that, by position, when it puts ≥ 90 %
of the shell on its labelled sites. The crowded block's vacancy loops then lose 251 net (5–10 % of
the loops' content). The rest are loops whose shells another
loop's rim crosses (the walk's circuits disagree by 0.5–2 %), which only a region holding both could
count, larger than `REGION_MAX`. On real frames the strained-shell path leaves the Al frame's 16
exact shells exact. Pinned by `large_crowded_loops_are_counted_on_strained_shells` (ignored; release
only).

**Regions as unions of their parts.** A loop can lie between regions accepted before it: on the thin
film's substrate, two 59 and 49 Å spheres of chained features 50–70 Å away each hold part of a
270-atom loop, and one sphere round all three is 113 Å, past `REGION_MAX`, so with spheres only the
loop is split between three censuses (substrate I 2005 against global's 2052). Re-ordering the merge
(small spheres first; the compact one wins) is worse (1900, 1708 I).

So a merged region's shape is the union of its parts' shapes (`Sphere::balls`) when one sphere
round them all would pass `max_r`: the merging cluster's sphere and the shapes of the regions and
failed clusters it takes in, each kept whole (a loop whose rim is several clusters is one sphere
round them, its plane inside; splitting a union back into its raw clusters leaves such planes
between the balls and loses 90 I on the substrate). Its sites lie within SHELL_IN bonds of a part,
its shell between SHELL_IN and SHELL_OUT of the parts; each part is at most `max_r`, the whole at
most 60 bonds about its first part (`SHAPE_MAX`), and along each periodic axis it spans less than
the box with its shell (it may reach past half the box from its centre: its atoms are gathered
part by part, and an atom two parts place at different images refuses it). Such a region must
also be whole, as a join: its shell full (counted, not reckoned from a sphere's volume, and first
checked on a grid against the atoms present, which refuses shells reaching into vacuum before the
census) and no void cut. A union that fits one sphere of `max_r` is counted on that sphere; single
clusters on their own sphere. Cost: the merge takes 128 s on the thin film and 23 s on the 128M cut
(the composite censuses that fail).

**The bent block.** Its errors (the lost void vacancies and ~100 spurious vacancies, table above) all lie 30–60 Å from a free
face: the generator's bend rotates each atom by an angle proportional to x about the block's centre
line, a shear strain of z·dθ/dx, ~6.6 % at the faces for 8° across 316 Å, which a real crystal does
not hold elastically (MD bending strains are below 1 %). At 2° (~1.6 % at the faces) every void is
exact and nothing is spurious; auto refuses global and falls back to local, which counts the loops
within −40 I / +11 V of 2010 / 2619.

**Open.** The thin film's substrate under local lacks ~12 interstitials against global and the
plane balance. `partial` on the 2M periodic block: −17 V (`union` +6). Global's Frenkel pairs inside
large a/2<111> loops (13–41 per loop) without `--resolve-ambiguous`.

---

## 17. Void detection

### 17.1 Method A: vacancy clusters of the defect census (`defects/voids.rs`, `rmda void`)

- **Input** is the §16 census exactly as `rmda defects` runs it (`cmd_defects::census`), with any
  of its references: `--site-reference auto|global|patch|local` (`rmda defects --reference`) or
  `--ws-reference FILE` (explicit), grains by `--grain-region`, `--grain-column` or `--segment`.
  Vacancies of measured grains only; interstitials take no part.
- **Clusters.** Vacancies join when within the §16.5 cutoff (default per grain, halfway between its
  first and second shells, so exactly first-neighbour adjacency; `--cutoff` overrides), reusing
  `clusters::cluster`. A cluster of at least `--min-sites` (default 2; `--void-min-sites` is an
  alias) is a void; smaller clusters are `void_isolated_vacancies`.
- **Volume** `V = Σ Ω_site` over the members, `Ω_site = |det T| / n_basis` of the member's grain.
  The local reference uses its grain-wide prefit; the explicit reference, which fits no `T`, uses the
  ideal cell of its structure at its measured bond. **Equivalent radius** `(3V/4π)^(1/3)`. Because the census fills enclosed
  cavities whole (§16.2), a closed void's interior is counted and its volume is exact on a lattice.
- **Classes.** `surface` when the void is *open*: some member has a first-shell lattice position
  holding neither an atom nor a vacancy (within 0.45 bonds), i.e. empty space the census did not
  count. With a fitted `T` (every reference but the explicit one; the local reference's grain-wide
  prefit) the positions are the grain's shell vectors (for HCP-type motifs, the basis site whose shell
  is held best); the 0.45-bond tolerance absorbs ~25° of local misorientation, and a ±8° bent grain
  keeps its closed voids closed (`void_a.rs`). Without one (explicit) it is a count of held points
  within the shell cutoff against the coordination; that count is fragile for the local census, whose
  atom-by-atom vacancy sites sit slightly off the ideal positions. Otherwise `boundary` if a member is in the
  boundary class, else `bulk`. A closed void under the surface is `bulk`: two atomic layers close it
  in the tests, and one is enough by construction.
  - The occupancy grid (`occupancy.rs`) is not used for this: its cells are ~2 a on a
    side, so a closed void larger than a cell empties grid cells and looks like a free surface, and a
    pit narrower than a cell is invisible. The lattice test needs no length scale.
  - **What a fitted reference sees of an open void:** nothing, mostly. Its empty sites reach vacuum,
    so they are outside the counted domain (§16.2; `defects_cmd.rs` checks that nothing deep in an
    open pit is a vacancy), and only sites the half-shell rule filled at its rim can show, as a
    `surface` void with a lower-bound volume. The **explicit reference** counts every reference site,
    so a pit into a reference slab is one `surface` void with its full volume.
- **Order.** Voids are sorted by size, then by `place_key` of their centre; the result does not
  depend on the order of atoms in the file.
- **Validation** (`crates/rmda-cli/tests/void_a.rs`, `defects/voids.rs` unit tests), BCC W with
  spherical vacancy clusters of 15, 55, 89 and 169 sites:
  - bulk, perfect and with 0.05 Å Gaussian noise, `auto|global|patch|local`: 4 voids, 328 sites,
    sizes exact, every void `bulk`; volume `n·a³/2` within 1e-6 (fitted, perfect), 1e-3 (local,
    perfect), 5e-3 (noise). Explicit reference: the same counts, volumes within 1e-3.
  - free surface (slab under vacuum): a 55-site void deep and a 15-site void two layers under the
    surface are `bulk` and exact under `global|patch|local`; a hemispherical pit gives no vacancy
    (fitted references) and one `surface` void of all its removed sites (explicit reference).
  - Σ5 twist bicrystal (grains ±26.57° about z, `--grain-region` and `--segment`), perfect and
    0.05 Å noise: one void per grain, sizes exact, `bulk`, volumes within 5e-3.
  - atom order: lattice, reversed and shuffled files give byte-identical stdout and tables under
    `global|patch|local`.
- **Series mode** (`rmda void FILES... --out DIR [--overwrite]`, `cmd_void::run_series`). It uses
  the `rmda defects` series machinery of `cmd_defects` (`series_items`: order by trailing file index, unnumbered files first as −n..−1, duplicate
  indices refused; `open_series`: `run.txt` signature, refusal of other settings, `--overwrite`;
  `open_table`: append, header check; `series_prefix`). Tables: `summary.csv` (one row per frame:
  counts, classes, sites, isolated vacancies, volume, largest void, cutoff, wall time) and
  `voids.csv` (one row per void per frame: the `--table` columns behind `file_index,file,timestep`).
  The signature is `void min_sites=… cutoff=…` plus the census's, so a defects directory or other
  settings are refused rather than mixed. Each frame is counted exactly as a single-file
  `rmda void` counts it; an explicit reference is prepared once. `--table`/`--sites` and
  `--experimental` are refused with `--out`. With `--segment` the grains are tracked as in
  `rmda defects --out` (`cmd_defects::census_tracked`, the same tracker, seeding on
  resume from `grains.csv`, `track.csv` in a shard for `rmda merge` to replay): `voids.csv` gains
  `stable_grain`, `grains.csv` holds one row per grain per frame, `grain_events.csv` the events,
  and `--sites-xyz` writes `sites/<index>.xyz` (merged like defect files, grain column
  remapped). `tests/shard_cmd.rs` checks the ids and events equal a defects series', and
  sharded/resumed runs equal the sequential one. Python: `rmda.GrainTracker` passed as
  `tracker=` to `rmda.defects`/`rmda.void` frame by frame gives the same ids and events
  (`python_parity.py` `check_tracker`).
  `tests/void_a.rs` checks order, per-frame rows byte-identical to single-file tables, skip on
  rerun, resume after an interrupted frame, refusal of other settings and of a defects directory.
  Python has no series mode for
  either command (loop over files).
- **Single-lattice voids:** `rmda ws --self --void-min-sites` (`analysis/void.rs`, `WsSelf`) counts
  voids on `ws`'s self-fitted lattice; `rmda void` is the per-grain command.

### 17.2 Method B: Mason φ-field (experimental, `void_b.rs`, `rmda void --experimental`)

- `φ(x) = minᵢ maxₖ |n̂ₖ(x)·(x − xᵢ)| / dₖ`, with face normals from the ideal cell (or the local T(x)
  with `--strain`). Volume of `{φ > 1 + ε}` by Monte Carlo over a region, extrapolated ε → 0 as in
  Mason. Its options need `--experimental`; method A's refuse it.
- **Gate:** pure W with relaxed vacancy clusters of known size, within 10 % of the analytic volume.
- **Wigner–Seitz cell faces.** `face_normals` keeps every Voronoi-relevant lattice vector: FCC 12,
  SC 6, BCC 14 (the truncated octahedron's 6 square second-shell faces included). The shortest
  vectors alone give BCC an 8-face octahedron (0.5625 a³ against 0.5 a³) that overlaps into every
  vacancy, and BCC voids read 5–20 % small.
- **Gate test on pure W** (`tools/void_b_gate.py`; 10×10×10-cell box, one spherical cluster; relaxed
  at 0 K with Finnis–Sinclair W, `tools/fs_w.py`, in place of LAMMPS: a₀ 3.1652 Å,
  E_coh 8.900 eV, E_v^f 3.63 eV, as published). Extrapolated volume against `n·a³/2`, 400k samples:

  | sites | unrelaxed | relaxed (0 K) | noise 0.03 Å | relaxed + 0.03 Å | noise 0.06 Å |
  |---|---|---|---|---|---|
  | 1 | −2.6 % | −7.1 % | +1554 % | +1605 % | +3816 % |
  | 9 | −1.4 % | −3.6 % | +173 % | +175 % | +421 % |
  | 15 | −0.8 % | −1.2 % | +108 % | +97 % | +258 % |
  | 59 | −0.4 % | −1.5 % | +26 % | +24 % | +59 % |
  | 89 | −0.1 % | −0.9 % | +17 % | +16 % | +36 % |

  **The quadratic passes at 0 K**
  and fails with any thermal displacement: 0.03 Å of noise (below W's room-temperature amplitude)
  adds a shell round every atom that outweighs a 15-site void several times and an 89-site one by
  16 %. `tests/void.rs` pins both (`method_b_passes_the_relaxed_w_gate_at_zero_kelvin` on the
  committed relaxed 15- and 89-site files, `method_b_fails_the_gate_with_thermal_displacements`).
- **Field radius.** `PhiField::for_threshold(ε)` searches `R(1+ε)`, exact only for thresholds up
  to ε. The field records `max_epsilon`, `estimate_void_volume` refuses a larger ε, and the CLI and
  Python build the field for the largest ε measured. A field built for a smaller ε misses atoms
  beyond its radius and counts the points they hold as void (V(ε) of a W vacancy stops falling at
  ε ≈ 0.3, at 0.53 of its cell, instead of shrinking as `(1 − ε)³`); the error is within Monte Carlo
  noise at 0.01–0.04 and decides everything above ε ≈ 0.1.
- **`--excess` with one ε.** One ε reports the measurement, marked `not extrapolated`; two are
  refused like the raw path; three or more extrapolate. A quadratic "fitted" through one point (a
  rank-1 normal matrix, not exactly singular in floating point) gives a value that depends on
  rounding (3603.5 Å³ against the row's 1393.8 Å³ on the unrelaxed 89-site file). Tests in
  `void_b.rs` and `tests/void.rs`. The single-ε excess over a defect-free frame with the same noise is −8 to +1 % for 9–89 sites at
  0.03 Å, but −31 to −6 % at 0.06 Å and −65 to +11 % for one vacancy: the excess at ε = 0.01 is still
  the difference of two shells, not a void volume.
- **Shell-free window** (`--thermal-window`, `max_face_gap`, `extrapolate_shell_free`). Two
  measured facts make it work:
  1. *The shell has compact support in ε, and the frame says where it ends.* Two atoms whose
     separation `r` matches face vector `v_k` share face `n̂_k·(x − xᵢ) = 1`; displaced, their cells
     part along it and the gap between them is void up to `φ = v_k·r/|v_k|²`. The **largest face
     gap** `g = max (v_k·r/|v_k|² − 1)` over matched pairs (each bond matched to the nearest face
     vector within a quarter of the shortest) bounds the shell: measured on V(ε) of every gate
     frame, the shell is gone at ε ≈ g (0.03 Å: g ≈ 0.06; 0.06 Å: g ≈ 0.12–0.14; a perfect box with
     noise reads exactly 0 above g). No amplitude, distribution or temperature is assumed.
  2. *Above it, a void erodes homothetically.* Each face of an empty cell moves inward by ε times
     its distance from the site behind it, so a vacancy's cell is the cell scaled by `(1 − ε)`
     about its site — `V = V₀(1 − ε)³`, exactly (unit test, pointwise) — and any set of vacancies
     likewise; a compact cluster nearly so. So `V^(1/3)` is linear in ε.
  The window is `[max(g, 0.01), +0.3]` in 7 thresholds; `V^(1/3) = α + βε` is fitted by least
  squares and `α³` reported, its error propagated from the binomial errors. A frame with
  `g + 0.3 > 0.9` is refused (small voids are gone by then). Results with the window width 0.2 or
  0.4 instead of 0.3 move by at most a few per cent (1 site) and under 1.5 % (15+ sites); a cubic in
  V instead of the cube root fits as well for clusters but is noisier.
  **Gate** (same fixtures and seed as the table above, window column, 400k samples over the whole
  box; the face gap in brackets for the noisy frames):

  | sites | unrelaxed | relaxed (0 K) | noise 0.03 Å | relaxed + 0.03 Å | noise 0.06 Å | relaxed + 0.06 Å |
  |---|---|---|---|---|---|---|
  | 1 | −4.8 % | −8.9 % | −6.3 % (0.057) | −10.2 % | +1.1 % (0.119) | −11.2 % |
  | 9 | +0.7 % | −2.8 % | +2.5 % (0.062) | −0.2 % | +4.7 % (0.110) | +3.0 % |
  | 15 | +0.3 % | −0.4 % | +1.3 % (0.067) | −0.3 % | +3.9 % (0.104) | +1.0 % |
  | 59 | −0.3 % | −1.2 % | −0.4 % (0.060) | −1.2 % | +0.3 % (0.130) | −1.2 % |
  | 89 | +0.1 % | −0.9 % | +0.6 % (0.060) | −0.7 % | +1.1 % (0.120) | +0.0 % |
  | 0 | 0 | — | 0.0 Å³ | — | 0.0 Å³ | — |

  One site is 16 Å³ in an 18,000 Å³ box (≈5 % Monte Carlo error before the extrapolation); in a
  4a box around it at 2M samples the six one-site cells read +1.0, −3.3, 0.0, −3.7, +5.3 and
  −7.8 % (the relaxed vacancy is genuinely smaller: the 0 K quadratic agrees at −3.3 %), and the
  9-site ones −2.8 to +3.9 %. **Every cluster from 1 to 89 sites is within 10 % at 0.03 and
  0.06 Å.** On the real relaxed 256k-atom HEA (chemical-disorder displacements, g = 0.165 and
  0.207): the defect-free frame reads 0.0 Å³ (raw 53,753, quadratic 84,377), and the frame with
  planted defects 85 ± 14 Å³ at 4M samples, against method A's 7 vacancies = 76.7 Å³.
  `tests/void.rs` pins the 15- and 89-site files at 0, 0.03 and 0.06 Å within 10 % and the perfect
  box with noise at < 1 Å³ (`the_thermal_window_passes_the_w_gate_with_thermal_displacements`);
  `void_b.rs` unit tests pin the face gap (0 on a lattice, `t/|v|` for one atom moved by `t`), the
  `(1 − ε)³` law and a noisy 13-site FCC cluster.
- **Decision:** method B stays behind `--experimental`; `--thermal-window` is opt-in inside it and
  is the only method-B route that passes the gate with thermal displacements. Before it can be
  the default: (1) the gate's noise is independent Gaussian per atom, not correlated phonons, and
  no thermal MD frame of W with a known void has been tested — the HEA agreement with
  method A is the only real-data check; (2) `g` is an extreme statistic, so one far-displaced atom
  (a dislocation core, a boundary, an interstitial) starts the window late for the whole region —
  correct but costly in precision, and refused past 0.6; a per-region gap is the obvious extension;
  (3) the integrator is serial (4M samples × 7 thresholds on 256k atoms: ~2 min). **Open:**
  a re-test on LAMMPS-relaxed W with Mason's potential; a per-grain extension is new research.

---

## 18. Warren–Cowley, RDF, cluster analysis

### 18.1 Warren–Cowley

- Definition: `α_ij^m = 1 − p_ij^m / c_j`, where `p_ij^m` is the fraction of j among the m-th-shell
  neighbours of i-type atoms and `c_j` the concentration of j in the selected region. For i ≠ j, α < 0
  means i–j ordering and α > 0 avoidance (like-atom clustering); for i = j, α > 0 means like-atom
  clustering.
- Shells: k-NN rank shells from the descriptor by default (FCC 12, 6, 24; BCC 8, 6, 12), or distance
  shells `--wc.shells r1,r2,…`.
- **Selection and region aware:** surfaces and GBs excluded by default via the region class or
  `--select`, with the concentration over the same selection.
- Per-atom shell histograms as flat u16 rows (`1 + n_shells × ntypes` per atom), filled in parallel and
  reduced in atom order (bit-identical at any thread count; see below for why not per-chunk u64
  accumulators); per-atom α computed into the output only with `--per-particle`. **Open:** run as a `LocalKernel`, output table `wc` (shell, i, j, α, counts), per grain when grain ids are present.

**Memory and parallelism.** The k-NN pass runs in parallel over fixed 1024-atom tasks, each atom writing its own row of
`1 + n_shells × ntypes` u16 (a status word, then the type histogram per shell); the α averages are then
summed serially over the rows in atom order. Chosen over per-chunk `[n_shells × ntypes × ntypes] u64`
accumulators because α is the reference's mean of per-atom ratios, and only the atom-ordered f64 sum
reproduces the serial result bit for bit: pooled counts give the same α mathematically but move the last
digit (and per-chunk f64 sums would make it depend on the chunking). Per-atom α is not stored at all;
`--per-particle` computes it from the rows into the output column. The `only_selected` reach set is an
`AtomicBool` mask (an OR, order-free). `nearest_k_excluding_self` selects the k nearest
(`select_nth_unstable` on a strict total key: d², j, then the full image shift) rather than sorting all
~450 candidates; the key reproduces a stable sort exactly, which a unit test checks bit for bit,
including tiny cells with image ties. `rmda wc` stdout and `--per-particle` files are byte-identical at
1, 4 and 12 threads. On the 256k-atom HEA frame (shells 0,12,18): 5.4 s / 16 MB serial, 1.2 s on
12 threads; `--per-particle` 1.8 s / 156 MB (66 MB of that is the output column). 42k-atom frame:
0.97 s / 3.2 MB serial, 0.16 s on 12 threads. The search goes through `CellList::neighbor_entries`,
whose minimum image rounds only periodic axes; the only cell-wide quantity is the density estimate that
sizes the cell list (`wc_probe_radius`, box volume including any vacuum), which affects speed, not
results.

**Depth-restricted centres (`--edge-correct`).** An atom is measured only when
`Occupancy::depth_of` (the grid `rdf --edge-correct` uses) exceeds the distance to its own k-th
neighbour, so no reported shell is cut by a free surface; concentrations stay over every atom or the
selection; `wc_atoms_surface` / `wc_atoms_measured` are printed only with the flag. On an ordered L1₂ film
open along z this recovers the periodic bulk's matrix to 1e-12, where the uncorrected film is off by
> 1e-3; periodic bulk excludes nothing. The depth is a lower bound (one occupancy cell, ~2 atomic
spacings, is given up), so the measured set is conservative.

### 18.2 RDF

- Partials in a dense `ntypes² × nbins` u64 array per chunk; the half-pair loop is `j > i` natively in
  `within`. Normalisation uses `N²` (`N1·N2` per type pair) with a factor 2 for same-type pairs, as OVITO and LAMMPS.
- **Normalisation volume** (`rdf --volume`): `box` (periodic bulk); `effective` (default when an axis is
  non-periodic or a vacuum gap is detected), `V = N_sel / ρ` with ρ the interior number density measured
  over the occupied bin volume or from the PTM lattice constant; or a user value. `--edge-correct`
  restricts centres to atoms more than r_max from the free surface. Validity checks use only periodic
  axes.

**Effective volume and edge correction.** `analysis/occupancy.rs` builds a coarse occupancy grid
(cells of ~8 atoms, edge refined on the occupied volume) that detects vacuum gaps on periodic axes and
gives each atom a lower bound on its distance to a free surface (Chebyshev distance to an empty cell or
open face). ρ for `effective` is a Gaussian-weighted neighbour count, `Σ_j exp(−r²/2s²)` over atoms at
least 4s deep, normalised by the kernel's integral to 4s (s ≤ 4 Å, ≥ 2 Å; centres chosen and summed by
position, so the file's atom order does not matter). The grid's own interior-cell density is not used: its
cells alias with the lattice planes, −11 % on a 38 Å W cluster, −1.3 % on an open slab.
On thermal W (σ = 0.08 Å) the smoothed ρ is within 1e-4 of the true density for bulk and slabs (periodic
with a gap, and open), 0.13 % for a 12³-cell cluster. Box-normalised, the slab read g(8–12 Å) = 1.61;
effective 0.90 (the surface atoms' missing neighbours); effective + edge correction 0.995 against bulk's
0.997, with n(12 Å) = 460 as in bulk. Periodic bulk under `auto` is normalised by the box. The r_max
check uses periodic axes only. Warren–Cowley needs no volume (α is a ratio of type counts; the box
density only sizes its cell list), but its k-nearest shells of surface atoms reach second neighbours;
`rmda wc --edge-correct` restricts its centres by depth as here (§18.1). **Open:** a `BTreeMap` per pair
and a full list filtered to `j > i`.

### 18.3 Cluster analysis (OVITO parity)

Cutoff or type-pair cutoffs, `--select`/`--only-selected` (mainly to cluster defects or a phase).
Outputs: `cluster.id`; optional unwrapped coordinates (BFS spanning tree with image accumulation); table
`clusters` with size, centre of mass (mass-weighted when masses exist), Rg and gyration tensor, sorted by
size. Parallel bond generation plus the concurrent union-find (§15.2).

`--select EXPR` (the `data::select` expression language) and `--only-selected` (the input's
`Selection` column, OVITO's `only_selected`) cluster a compacted subset with its own cell list: unselected
atoms bond to nothing, get `Cluster = -1`, `ParticleCluster = -1`; fractions are over the selection.
Ids are OVITO's (3.17 and later): 0-based in first-atom order, -1 for an atom in no cluster (OVITO before
3.17: 1-based, 0); `ParticleCluster` is also -1 below `--min-size`; per-cluster row `k` is id `k`
(`ClusterIdentifier`).
`--sort-by-size` renumbers largest first, ties in first-atom order (OVITO's tie order differs, so ids
agree up to relabelling). `--unwrap` rewrites `-o` positions; `--table FILE` writes size, mass, COM, Rg,
gyration tensor (`xx yy zz xy xz yz`, normalised by mass, `Rg = √tr G`) and `percolates`; the bindings
attach the per-cluster columns. Masses: a per-atom `mass`/`Mass` column, else the data file's per-type
masses when every present type has one, else unit weights (OVITO additionally takes element masses from
type names). Unwrapping uses a weighted union-find (each node's offset to its parent), not a BFS: no
adjacency list and no second neighbour pass, and every spanning tree gives the same coordinates to
rounding for a cluster that does not wrap onto itself. One that does (a loop whose minimum-image vectors
do not sum to zero) is flagged `percolates` / `cluster_percolating`, which OVITO does not do. Bonds are
generated in parallel in 2048-atom tasks, batched (2 tasks per thread, a few MB of buffered bonds), and
unioned serially in atom order: the union-find, labels and float offsets are those of the serial loop at
any thread count (unit test, 1 vs 4 threads, bit for bit, both modes). 256k HEA: `--cutoff 2.9` 1.17 s
serial, 0.31 s on 12 threads; `--knear 4` 0.56 s on 12 threads.

Validation against OVITO 3.15 (`ClusterAnalysisModifier`, compute_com, compute_gyration): a synthetic
536-atom frame of blobs crossing the periodic faces (periodic with per-type masses, open, `only_selected`;
pinned without OVITO by `tests/data/cluster_ovito_expected.txt` from `tools/cluster_ovito_expected.py`,
ids identical, COM/Rg/G to 1e-9 asserted), and end to end through data files: the 42k HEA frame with
`type == 1` (3880 clusters) and `z < 30`, the 256k HEA frame with `type == 2` (3091 clusters) — sizes
exact, COM, Rg and tensor within 2e-14 relative, except the one 57,853-atom cluster of the 256k case,
which percolates and is flagged.

**Type-pair cutoffs.** `--pair-cutoff A-B=CUT` (repeatable; Python `cluster(pair_cutoffs={(a, b):
r})`), `ClusterMode::PairCutoffs`: types by number or by name (`type_labels`), a symmetric table of squared
cutoffs by type number, a pair with no entry never bonds. The cell list is built at the largest pair cutoff
and each bond kept when `d² ≤ c²` of its pair (the list's own test, so a table with one cutoff for every
pair gives the `--cutoff` labels exactly; unit test). Unknown types, a repeated pair and non-positive
cutoffs are refused where OVITO skips them silently. OVITO 3.15's `ClusterAnalysisModifier` has no
`ToPairCutoffs` mode (its `neighbor_mode` is `CutoffRange` or `Bonding`); the OVITO reference is
`CreateBondsModifier(mode=Pairwise)` + `ClusterAnalysisModifier(neighbor_mode=Bonding)`. Validated on the
two-species synthetic frame (types 1/2 named Al/Ni): four cases (ids; names with Ni–Ni absent; open
boundaries; `only_selected`), ids identical and COM/Rg/G to 1e-9, pinned by
`tests/data/cluster_pair_ovito_expected.txt` from `tools/cluster_pair_ovito_expected.py`.

---

## 19. Workflows: polycrystal and thin film

**Current form:**

```
rmda defects dumps/dump.* --segment --out results [--defects-xyz]                        # polycrystal
rmda defects reference.data frame-*.data --region 'z <= 500' --region 'z > 500' --out results   # thin film
```

**Target config form** (`rmda run`, §20):

```toml
# poly.toml
[input]
files = "dumps/dump.*.gz"
file_index = "dump\\.(\\d+)"
[pipeline]
steps = ["ptm", "segment", "defects"]
[ptm]
structures = ["fcc", "bcc", "hcp"]
[segment]
mode = "auto"
min_size = 1000
stacking_faults = "merge"
track = true
[defects]
reference = "auto"           # "auto" | "global" | "patch" | "local" | "periodic"
[output]
dir = "results"
tables = ["grains", "clusters", "grain_events"]   # long format across files
defects_xyz = true
report = "json"
```

A thin film adds `[regions] substrate = "z <= 500"`, `film = "z > 500"`, `film_axis = "z"`, an optional
`reference = "reference.data"` baseline for explicit WS on the substrate, and `wc`/`rdf` steps restricted
to the interior class.

Data flow per frame: read → k-NN index → PTM → grain edges → crystallites → merge → adopt → track → per
grain: fit reference → assignment → regions → census → clusters → tables and xyz → RunReport.

**Open:** the pipeline config form (`[pipeline] steps` and per-analysis sections, §12); `rmda run`
today runs one command per config (§20).

---

## 20. Run model and time series

**Current code: `rmda defects FILES... --out DIR`** (`cmd_defects.rs`):

- **Streaming.** One frame at a time, in the §8.7 file order. Grain tracking (with `--segment`) keeps only
  the previous frame's sorted (id, grain) vector.
- **Long-format tables**, appended and flushed per frame, every row prefixed by the file index:
  - `summary.csv`: per-frame totals, cluster counts, wall time, current and peak memory;
  - `grains.csv`: every per-grain column plus `stable_grain`, `clusters`, `largest_cluster`;
  - `clusters.csv`: per cluster V, I, net, centre and grain;
  - `grain_events.csv` with `--segment`.
- `--defects-xyz` writes each frame's `vacancies`/`interstitials` (and `non_lattice`) `.xyz` with a cluster column. `--defects-layout` puts them in `defects/<index>/` (`dirs`), as `defects/vacancies-<index>.xyz` (`files`), or appended to one file per kind (`append`), with `index=<index>` in every frame's comment line. An appended file is pruned on resume to the frames the summary records, so an interrupted frame is not duplicated.
- **Resume.** Frames already in `summary.csv` are skipped. The summary row is written last, so an
  interrupted frame is redone. `run.txt` holds the settings, and a rerun with different settings is
  refused unless `--overwrite`. A tracked run replays the last finished frame to restore tracking.

**Measured:** thin film, two frames: 69–70 s per frame with tables and clusters. Frame −1
(`data.firstframe`): 0 V, 0 I, 0 clusters. Frame 999: 15 349 V, 15 328 I, 21 990 clusters, largest 100.

**Target (generalising the above to every pipeline):**

- `rmda run CONFIG [overrides…]`: CLI flags override config fields (`--grains.min-size 500`).
- Tables carry `file_index, frame_index, timestep` and a schema version; `results/manifest.json`
  records completed (file, frame) pairs with a config hash.
- **File-parallel mode.** `--files-in-parallel k` splits the thread pool into k sub-pools for many small
  frames. The memory estimate is multiplied by k.

**`rmda run CONFIG.toml`** (`cmd_run.rs`). A config is a command line written down:
`[input] files` (globs), `[run]` global options and one command section whose keys are that
command's long options, resolved through clap's own definition (unknown or misplaced keys refused by
name) and parsed by the CLI's parser, so one option namespace and one set of defaults hold by
construction. `--set section.key=value` overrides, `--dry-run` prints the resolved command.
Every run writes `report.json` (§23.3) in its `out` directory, also on failure. The thin-film series
through `rmda run` equals the same `rmda defects --out` run (tables and defect files byte-identical,
timing columns aside). Not implemented: the registry/planner of §12, presets as one-step pipelines,
`manifest.json` with a config hash (the series' `run.txt` signature plays that role), file-parallel
mode.

**Sharding.** `--shard I/N` takes contiguous blocks (not every n-th frame), so
tracking needs only the frame before a block. Shards do not track: they write labels where stable
ids go and record each frame's tracker input (`track.csv`: size per label, overlap counts with the
previous frame's labels), the first frame against the lead-in frame (segmented, not censused).
`rmda merge` replays `Tracker::step` over all frames in order, which is the code `Tracker::update`
runs, and rewrites the stable-id columns of grains/clusters/antisites and the defect files' `grain`
column; merged output equals the sequential run byte for byte (tests/shard_cmd.rs, 2 and 3 shards,
mixed thread counts). Round-robin shards with renumbering at merge would not: the tracker's
tie-breaks and event order depend on the id values. A resumed tracked series seeds its tracker from
`grains.csv` (`Tracker::seed`) and segments, without censusing, the last finished frame; a fresh
tracker would renumber from 0 and shift later ids once a grain has ended. Rows an interrupted frame
wrote before its summary row are pruned on resume.

**Open:** file-parallel mode, a config hash in the tables; frame indices within multi-frame files;
time series for commands other than `defects` and `void` (both shard).

---

## 21. CLI

**Current commands:** `info`, `index`, `convert` (xyz, dump, binary dump, data), `census` (neighbour census), `structure`
(CNA), `ptm`, `segment`, `defects`, `merge`; `cluster`, `rdf`, `wc`, `void`, `ws` (explicit or
self-fitted reference); `grain-boundaries`, `analyze`, `run`, `explain`, `bench`. Global options: `--threads`, `--input`, `--output`, `--profile`, `--memreport`,
`--mem-limit`, `--skip-columns`, `--atom-style`, `--keep-general`, `--cell`, `--pbc`,
`--xyz-origin native|zero`, `--frame/--start/--end/--stride`.

Target:

```
rmda info FILE...                          # header, cell, counts, columns (no full parse with --quick)
rmda index FILE                            # frame index
rmda convert IN OUT [--columns ...] [--sort-id] [--format ...]
rmda run CONFIG [overrides]                # the pipeline model (§20)
rmda <preset> FILE... [options]            # one-step pipelines: cna ptm grains defects ws void rdf wc cluster
rmda bench [--atoms N] [--structure fcc]   # synthetic benchmark of every stage, writes a RunReport
rmda explain CONFIG                        # plan, memory estimate, keys, without running
```

- Further global options: `--io {auto,mmap,pread}`, `--report {text,json,both}`,
  `--profile [chrome]`, `--select EXPR`, `--label-region SPEC`, `--units`, `--extxyz-lattice`.
- Presets are thin: each builds a one-analysis pipeline (plus required providers, e.g. `grains` adds
  `ptm`) from the same config structs, so `rmda ptm --rmsd-cutoff 0.15` is `[ptm] rmsd_cutoff` and Python
  `rmsd_cutoff`.
- Human text summary on stderr; machine output to files or `--report json` on stdout. Exit codes: 0 ok;
  2 usage; 3 input/parse; 4 memory limit refused; 5 quality gates failed on all grains (results still
  written); 1 other.

**Exit codes:** `Error::exit_code`: 3 for `Parse` and `Input` (a file that cannot be opened or
recognised, anything a reader refuses, `Frame::check_input`), 2 for `Usage` (a bad `rmda run` config,
`void --epsilons nan`) and clap's own errors, 4 when `--mem-limit` stops a run, 1 otherwise; `rmda
defects` on one frame exits 5 when no grain was measured (a series exits 0, the tables record it). Python: `Input`/`Usage`/`Parse` raise `ValueError`, a missing file
`FileNotFoundError`, I/O `IOError`.

**Layout:** `cli.rs` (argument definitions), one `cmd_<name>.rs` per subcommand (`command(ctx)`),
`util.rs` and `prelude.rs`; `main.rs` calls `lib.rs`. `rmda explain`, `rmda bench` and `--mem-limit`
(exit 4) exist.

**Open:** presets over one option namespace, `--report`.

---

## 22. Python bindings

```python
import rmda
rmda.set_threads(32)                       # reaches rmda_core::exec; 0 = all cores; rmda.get_threads()

f = rmda.read("dump.100.gz")              # Frame; rmda.iread(path) iterates frames
f.positions                               # numpy (N,3) float64, zero-copy, read-only; f.types uint32, f.ids int64
f["StructureOrientation"]                 # typed copy, (N,c) for multi-component columns
f.cell                                    # (3,3) array

seg = rmda.segment(f, threshold=5.0, core_angle=2.0, min_size=100, adopt=True, rmsd_cutoff=0.1)
res = rmda.defects(f, regions=["z <= 500", "z > 500"], reference="auto")   # or grain_column=..., segment=True
res["grains"], res["vacancies"], res["interstitials"], res["clusters"]   # dicts of numpy arrays; t is (G,3,3)

atoms = rmda.to_ase(f); f2 = rmda.from_ase(atoms)                  # numpy-based, no per-atom Python loops
g = rmda.Frame.from_arrays(pos, cell, pbc=..., types=..., ids=..., properties={...})
g["Grain"] = labels                                                  # an attached column, for grain_column=
import rmda.ovito; f3 = rmda.ovito.from_ovito(data); rmda.ovito.to_ovito(f3, data, result=res)
```

- pyo3 0.25 Bound API and rust-numpy 0.25 (§4). `Frame` is `frozen`; per-atom arrays are read-only views
  owned by it that safely outlive it; `f[name]` keeps I32/F32 dtypes. `from_ase` goes through numpy
  (only distinct elements are looked up in `ase.data`).
- **One way in.** `Frame.from_arrays`, `from_ase` and `rmda.ovito.from_ovito` all end in one builder
  (`build_frame` in `rmda-py`), which applies the checks a reader applies (`Frame::check_input`: finite
  coordinates, a box with volume; plus unique ids and types ≥ 1). Extra per-atom arrays become input
  columns (`ParticleData::props`), the same place a file's extra columns go, so `grain_column=` and region
  expressions read them; `f[name] = array` attaches a column instead. Type codes follow the readers:
  integer ids as given, names coded 1.. in sorted order (the extended-XYZ rule).
- **ASE.** `to_ase`/`from_ase` round-trip ids (`arrays["id"]` when not 1..N), columns, and type names that
  are no element (`arrays["type"]` + `info["type_names"]`, written only then, so an edited `numbers` array
  is never overridden by a stale type column).
- **OVITO** (`rmda/ovito.py`, pure Python; imports `ovito` only when imported). `from_ovito` reads
  positions, the 3×4 cell (vectors as columns, origin last), pbc, identifiers, typed ids or names, and
  numeric properties; `to_ovito` writes attached properties, `f.tables` as `rmda.<name>` `DataTable`s
  (string columns as integer columns with named types) and a result's scalars as attributes. Equal to
  `rmda.read` field for field on the fixtures, except general-triclinic LAMMPS boxes, which `rmda.read`
  rotates into restricted form (§6.2) and OVITO does not. Tested against OVITO's own reader and
  Wigner–Seitz modifier (`python_bindings.rs::ovito_interop`, skipped without `ovito`).
- **OVITO modifiers** (`rmda/_ovito_modifiers.py`, loaded lazily through `rmda.ovito`):
  `SegmentModifier`, `DefectsModifier`, `VoidModifier`, each a `ModifierInterface` whose Traits mirror the
  Python call's keywords (0 for "automatic"), `grain_column` a `PropertyReference`, the explicit reference
  an extra `FileSource` input slot. Registered under the `OVITO.Modifier` entry point in `pyproject.toml`.
  `DefectsModifier(output="defects")` replaces the particles by vacancy sites and interstitial atoms, as
  OVITO's own Wigner–Seitz modifier does. Tested in a pipeline (`python_bindings.rs::ovito_modifiers`).
- **Every analysis call releases the GIL** (`py.allow_threads`), including index builds, file reads and
  `f.write`; two Python threads can run analyses concurrently.
- Bound: `cna`, `ptm`, `rdf`, `cluster`, `warren_cowley`, `grain_boundaries`, `void`, `ws`, `segment` (attaches
  `Grain`/`StructureType`; CLI scalars plus `grains`/`pairs`) and `defects` (every `rmda defects` option;
  scalars, `grains` with every `--table` column, `vacancies`, `interstitials`, `clusters`, per-atom
  `atom_grain`/`displacement`). `segment` and `defects` are also `f.run`/`Pipeline` steps.
- `grain_boundaries` and `void` (method B, `experimental=True` required) are bound with the CLI's flags as
  keywords and are `f.run`/`Pipeline` steps. `f.tables` holds per-group tables as `{name: {column: array}}`
  (copies): per-grain/per-cluster store properties (`per_particle == false`) as `grains`/`clusters`, and the
  tables `segment` and `defects` return as `segment.*`/`defects.*`; a call replaces its own tables whole.
  `Pipeline.run` is a generator.
- **Parity.** The per-frame defects glue is shared: `rmda_core::analysis::defects::frame` (`analyse`,
  `prepare_reference`, `run_ptm`, with a `FrameObserver` for the CLI's progress lines) runs both
  `rmda defects` and `rmda.defects`/`rmda.segment`. `python_parity.py` checks them: for `defects` (regions,
  `--segment`, patch reference, a planted interstitial, `grain_column` from a file and from
  `rmda.segment`, a bad region) and `segment` (tilted BCC bicrystal, with and without adoption) it compares
  every stdout scalar, table cell, defect-xyz row and per-atom column against the CLI.
  `python_bindings.rs` checks dtypes, view lifetimes, numpy `from_ase`, `set_threads`, concurrency and
  pipeline steps.

**Packaging:** the root `pyproject.toml` (maturin, `python-source = crates/rmda-py/python`,
`module-name = rmda._rmda`) with the crate's `abi3` feature (`pyo3/abi3-py39`): `maturin build --release`
writes one `cp39-abi3` wheel holding the `rmda` package (`__init__.py`, `ovito.py`), the extension and the
`rmda` console script (numpy a dependency, `[ase]` extra). `cargo build -p rmda-py` still builds the
version-specific module the staged-module tests use. `python_bindings.rs::abi3_wheel_installs_and_passes_parity`
installs a fresh wheel into a venv and runs `python_parity.py --installed` against it; it skips without a
wheel newer than the binding source.

**Open:** no `read_series` or `.to_pandas()`; no binding layer generated from a registry; macOS and Windows
wheels are built by the release workflow but not yet verified on those systems (the core and CLI
type-check for both targets); no aarch64 Linux wheel; the OVITO modifiers are tested with the `ovito`
module, not yet in OVITO Pro's GUI; a running analysis cannot be cancelled from OVITO.

---

## 23. Profiling and resource reporting

### 23.1 Spans

Stage spans, **never** per atom (per-atom `Instant::now` calls, millions per stage, distort the
profile). Nested spans form a tree, e.g. `defects.grain` with `.prefit`, `.walk`, `.census`. Each records
wall time and its peak anonymous RSS; target additions are CPU time (`getrusage` deltas), RSS at entry
and exit, file RSS and atom count.

### 23.2 Memory measurement

With `--profile` (or `--memreport`), a sampler thread reads `/proc/self/status` every 20 ms, and every
stage boundary also samples. Each stage reports the peak **anonymous** RSS (heap) while it ran; file-backed
pages are reported separately in the footer, because they are cache the kernel can drop (`VmHWM` counts
the mmapped input: ~140 B/atom of the thin film is its 790 MB data file). Memory and CPU figures come
from `/proc` and read 0 or absent on other systems (**Open:** a `getrusage` fallback).

### 23.3 `RunReport`

```
run: 3 frames, 128,000,000 atoms/frame, 128 threads (SLURM_CPUS_PER_TASK), host nid001234
estimate: peak 11.8 GB  (resident 4.4 GB atoms, 3.7 GB ptm, 1.0 GB grains, 2.7 GB defects transient)
frame 0 (file_index 100): wall 312.4 s  cpu 38,911 s  peak_anon 11.2 GB
  io.read 21.3 s (6.1 GB/s, anon +4.4 GB)   ptm 96.0 s (anon +3.7 GB)   grains 41.2 s   defects 140.1 s
total: wall 935 s, cpu 116,200 s (97 % parallel efficiency), peak_anon 11.4 GB (estimate +3.5 %)
```

The same data goes to `report.json` and Python `result.report`.

### 23.4 Deeper profiling

A `[profile.profiling]` cargo profile (release + `debug = 1`) for `perf`, VTune and `samply`;
`--profile chrome` writes a per-thread `trace.json`; `rmda bench` generates synthetic crystals in memory
and reports per-stage throughput, the basis of CI performance gates.

`rmda run` writes the JSON report (`report.json`: status, version, command, config, host, threads,
wall and CPU time, parallel efficiency, peak RSS, the stage table, per-frame summary rows).

**Open:** per-stage CPU time, the estimate line, Chrome trace, the `profiling` cargo profile.

---

## 24. Performance and memory budgets

### 24.1 Throughput targets (same machine, same threads)

| Stage | Target | Measured (thin film, 12 threads, i5-1235U) |
|---|---|---|
| read text dump/data | ≥ 1.5 GB/s at 8 threads; ≥ 0.4 GB/s at 1 thread | 1.8 GB/s at 12, 1.5 at 8; 0.38 GB/s at 1 (**accepted as met**) |
| read binary dump | ≥ 3 GB/s (disk bound) | not measured |
| read `.gz` | inflate-bound | indexed or multi-member: 1.39 GB of text in 3.0–4.4 s; one member without an index: ~140 MB/s, serial (§8.1) |
| write xyz/dump | ≥ 0.8 GB/s at 8 threads | `convert` read + write 348 MB: 1.14 s |
| CNA adaptive | ≤ 0.66× OVITO wall time | 2.4–3.6 s vs 5.4–5.6 s = 0.43–0.65× (**met**) |
| PTM (FCC/HCP/BCC) | ≤ 0.8× OVITO wall time, robustness ≥ OVITO | 21.3 s vs 31.3 s = 0.68× (met); robustness §14.1 |
| grains | ≤ 1.0× OVITO grain segmentation | 8.7 s (OVITO not timed) |
| defects | ≤ 2× PTM time | `defects.grain` 14–15.5 s (met) |
| thin film 5.76M atoms, full `defects` frame | ≤ 2 min per frame | 58 s; 69–70 s with time-series tables and clusters (met) |
| flagship pipeline, 128M atoms, one 128-core node | ≤ 10 min per frame | the 128M frame, `defects --segment`, 80 threads, every grain local: 7.4 min with `--local-merge partial`, 49 min with `union` (§16.8) |

### 24.2 Memory budget (bytes/atom at peak; flagship pipeline, PTM default outputs)

| Item | B/atom | Live during |
|---|---|---|
| atoms (pos 24, id 8, type 4) | 36 | always |
| images (if present, packed) | 8 | always |
| reorder permutation | 4 | always (if on) |
| grid (`order` + bins) | 4.5 | always |
| `ptm.*` default columns | 29 | after PTM |
| grain id | 4 | after grains |
| union-find parent + crystallite id | 8 | grains (transient) |
| site index (transient) + occupancy (per site ≈ per atom) | 9 | defects |
| region class (atoms + sites) | 2 | defects |
| tile scratch | < 0.5 | per pass |
| **Peak (flagship)** | **≈ 97** (105 with images) | — |
| cell list (`within` users: RDF, WC, cluster, `ws`): `binhead` + bins, `u32` chains | ≤ 8 | that analysis |

At 128M atoms that is **≈ 12.4 GB** (13.4 GB with images), which fits a 256 GB compute node with room
for `--files-in-parallel`; on a 15 GB laptop 64M atoms is the comfortable limit. The input file is not
counted; gzip needs one decompressed frame (~50–60 B/atom of text) during the read only.

**Measured** (peak anonymous heap, §23.2):

| case | peak heap |
|---|---|
| read, lean (velocities and images skipped) | 34.5 B/atom (thin film 198.8 MB) |
| read, all columns | 73 B/atom (velocities 24 + images 8 + slack) |
| thin film `rmda defects`, PTM included | 1.03 GB (179 B/atom); 1.0–1.1 GB per frame in a time series |
| synthetic BCC, 16M atoms, one grain | 2.46 GB (154 B/atom) |

Where 154 B/atom goes: **per frame ~102 B/atom** (frame 36, compact PTM 28, k-NN index 29, grain
membership and flags 5, displacement 4) and **per grain ~50 B per atom of the grain being processed**
(unwrapped positions 24, census 8 + site grid ~6, crystalline lists 8, patch membership 4). The 16M case is one grain spanning the frame, the worst case; in a polycrystal the per-grain term is
bounded by the largest grain. At 128M that projects to ≈ 13 GB + 50 B × (largest grain), ≈ 20 GB in the
single-grain worst case.

**Open (toward the 97 B/atom budget, in order of size):**

- borrow positions instead of copying when a grain needs no unwrapping (−24 B per grain atom);
- f32 relative positions inside the k-NN index (29 → 17 B/atom; needs a precision check against the PTM
  oracle);
- streaming the PTM output per grain instead of per frame;
- `tests/memory_budget.rs` asserts the rows built from fixed-size types (atoms 36, cell list ≤ 8, the
  44 B/atom floor of the two, and that the cell list borrows positions); extend it to the PTM, grain and
  census rows, and assert a 16M measured peak within ±25 % of the estimate.

---

## 25. Testing and validation

### 25.1 Layers

1. **Unit tests** per module, next to the code.
2. **Property tests** (`proptest` dev-dep, or the hand-rolled xorshift): cell round trips, bbox,
   wrap/unwrap; MIC and `knn` against brute force on random cells including triclinic and slab; number
   parsing against std (§8.2); reader bit identity against the legacy reader (§8.1); orientation algebra
   (§11.3).
3. **Format parity:** LAMMPS-generated fixtures compared field by field (`binary2txt_oracle.dump`,
   `triclinic_tilted.dump`); ASE and OVITO read RMDA-written files (when importable) and must agree; atomsk
   `.lmp` and component-major xyz.
4. **OVITO oracles:** CNA (adaptive, exact on all reference atoms), PTM (§14.3), and the `poly3` grain and
   WS cross-check (`tools/poly3_ovito.py`). Target: CNA all modes, PTM orientations, WS explicit, cluster,
   RDF and grains (ARI) on thermal single crystals at random orientations, the HEA polycrystal, a thin film
   and a nanoparticle, from a `tools/gen_ovito_oracles.py`.
5. **Synthetic suites:** defects (§16.8) and grains (§15.5).
6. **Invariants** for every analysis on a mixed fixture: thread-count determinism (1, 3, 8 threads,
   bit-identical), atom-order permutation invariance, origin translation invariance, WS conservation in
   every region class.
7. **Scale and performance:** `rmda bench` at 2M atoms in CI (release) against a stored baseline, failing
   on > 20 % regression; a 16M-atom smoke test weekly; 128M on an HPC node before each release, archived in
   `docs/perf/`.

### 25.2 Gating

- `cargo test --workspace` passes. CI runs fmt, clippy
  `-D warnings` on every crate and target, and the workspace tests with Python 3.12 + numpy (the
  extension is built first, so binding tests run), all on toolchain 1.72.0.
- Locally use `--no-fail-fast`: a plain `cargo test` stops after the first failing test binary and hides
  every later result.
- Expensive tests are `#[ignore]`d with their reason; CI's `slow` job (weekly and by hand) runs them
  with `cargo test --release -- --include-ignored`. The reference-data oracles read
  `RMDA_REFERENCE_DIR` and `RMDA_CASCADE_DIR` and print a skip line where the data is absent. The dev
  profile optimises `rmda-core` (debug assertions on), so the default suite takes minutes.
- Target CI jobs beyond these: MSRV build, Python matrix (3.9 / 3.11 / 3.13 against the maturin
  wheel), bench gate; CI counting skips.

**Open:** release/scale and performance jobs; the Python wheel matrix;
permutation, translation and 1/3/8-thread invariance tests beyond the cases in §15.5 and §16.8.

---

## 26. HPC deployment

### 26.1 Build

- `cargo build --release` builds `rmda-core` and `rmda-cli` only (workspace `default-members`), so no
  Python is needed. The binary is `target/release/rmda`.
- Optional `RUSTFLAGS="-C target-cpu=native"` on the build node of the target partition (an AMD EPYC
  7763 node is Zen 3: `-C target-cpu=znver3`); check `rustc --print target-cpus`.
- **Cray systems.** Under the HPE Cray programming environment `cc` is the Cray compiler wrapper, which
  breaks Rust's link step. Link with gcc: `export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc`, or
  `module load PrgEnv-gnu`.
- **Python bindings** (`cargo build --release -p rmda-py`) need Python ≥ 3.7 (pyo3 0.25). Where the system
  `python3` is older (3.6 on some Cray systems), load a newer one (`module load cray-python`) and point pyo3 at it with
  `export PYO3_PYTHON=$(which python3)`. numpy is a runtime dependency.
- A portable default build with a runtime SIMD dispatch for the newline scan only.

**Open:** offline builds (`cargo vendor` + `.cargo/config.toml`) and a static musl build, documented in a
`docs/hpc.md`.

### 26.2 Threads and affinity

- `rmda` uses rayon, not OpenMP. `OMP_NUM_THREADS` has no effect; by default the thread count is
  `SLURM_CPUS_PER_TASK` under SLURM, else every core (§10.1).
- Recommended: `srun --cpus-per-task=128 --hint=nomultithread`. First-touch allocation happens in the
  parallel parse, so pages land NUMA-local to the threads that use them.
- Thread scaling is measured stage by stage only up to 12 threads (§24.1); the 128M frame runs on 80
  (§16.8).

### 26.3 Filesystem (Lustre)

- **Gzip input** inflates in parallel when the file is indexed (`rmda index FILE.gz`) or written as
  many members (§8.1). A single-member `.gz` without an index inflates serially (~140 MB/s of text):
  index it, or `gunzip` it first so the parallel text reader (~1.8 GB/s at 12 threads) does the work.
- `--io pread` by default on Lustre: 64 MB blocks, with one read-ahead block in flight (**Open**, §8.1;
  today input is mmapped).
- Output tables are small. Frame dumps are written chunk-parallel to one file. Advise
  `lfs setstripe -c 8` for output directories with large xyz.
- Binary dumps are preferred for 10⁸-atom trajectories: smaller, and parsed without number conversion.

### 26.4 Multi-node (not implemented)

- Every per-atom analysis is local with a halo of r_halo = max(k-NN reach, WS patch halo).
- Global analyses (grains, clusters, WS census totals) reduce via a union-find merge across subdomain
  faces and small tables.
- The design keeps all global state as tables or union-find over boundary atoms, so an MPI layer can be
  added behind a feature without touching analysis kernels.
- The MPI layer is a hand-written `extern "C"` binding to the system MPI (no crate dependency).

### 26.5 Other clusters

- The same procedure applies; only the `target-cpu` and the linker setup may differ.
- Python: `pip install rmda` where wheels fit the system (manylinux2014, glibc ≥ 2.17), or build on
  the login node (`maturin build --release -m crates/rmda-py/Cargo.toml`, or
  `cargo build --release -p rmda-py` for an unpackaged module) and install into a `venv` on the
  project filesystem.

---

## 27. Status

### 27.1 Implemented

- **Fundamentals** — disorientation by right multiplication; HCP c/a = √(8/3); dump triclinic bounds read
  and written as the LAMMPS bounding box; boundary flags periodic only for `pp`; 9-number `Lattice` +
  `Origin`; PTM default set = OVITO's.
- **Neighbours and I/O** — exact unbounded `KnnIndex`; preallocated parallel fills; parallel readers and
  writers, Eisel–Lemire floats, every dump column, scaled coordinates, `ITEM:` blocks in any order,
  binary dump read/write, streaming and parallel gzip, frame index and fast skip, data-file completeness,
  `.lmp` sniffing, lean column skipping; MSRV 1.72.
- **Defects** — affine per-grain reference, conserving pure-WS census with interior/boundary classes,
  quality gates, patch-local reference sharing one index space, defect clusters, antisites (§16).
- **Grains** — crystallites + mean-orientation merge, order-12 HCP group, FCC/HCP stacking-fault and twin
  handling, OVITO-like orphan adoption, `--min-bonds`, OVITO's algorithm as `--mode auto`, grain
  tracking (§15); dHCP (§28.1).
- **Time series** — `rmda defects FILES... --out` with long-format tables, resume, sharding and tracking (§20).
- **Memory** — heap-aware profiling, compact PTM records, no stored neighbour tables, two-pass census.
- **Python** — the `rmda` package (pyo3 + numpy, abi3 wheel with the `rmda` command), zero-copy views,
  GIL release, CLI parity tests (§22).
- **Validation** — 3-lattice polycrystal time series exact against construction and OVITO WS; ground
  truth by construction, net content and plane balance (§16.8).
- **Local census** (§16.3d) — grains no single lattice describes, counted from local fits with loops and
  clusters counted whole as regions; `--reference auto` falls back to it. Byte-identical results across
  thread counts (tested).

### 27.2 Remaining

Not pursued: the memory budget of §24.2 (the 128M frame peaks at 210–225 B/atom, 27–29 GB, well within a
node).

- **Accuracy:** region counting inside loops where the displacement nears b/2 (§16.8).
- **Scale:** thread scaling beyond 12 threads is measured only on the 128M frame (§16.3d).
- **Framework:** `StructureRegistry`, typed pipeline, namespaced keys; grain tracking in a
  Python `Pipeline` (a `GrainTracker` per call works today).
- **I/O:** unpacked `[i32; 3]` images; an extxyz attribute store beyond `Timestep=`/`Time=`.
- **Structure ID:** the residual hot-HCP disagreement (§14.1).
- **Defects:** antisites in ordered cells larger than the motif (D0₃, L2₁, D0₁₉, B1).
- **Other analyses:** void method B under thermal displacement (`--thermal-window` passes the gate,
  still `--experimental`, §17.2).

---

## 28. Future work

### 28.1 Double-HCP (dHCP, ABAC stacking)

**Implemented** (`--dhcp`, `analysis/stacking.rs`, structure code 32, `Motif` dhcp). dHCP is the ground
state of La, Pr, Nd and Am, and appears as a stacking polytype in stacking-fault-rich alloys (low-SFE
HEAs, Co-based alloys), where FCC, HCP and dHCP layers alternate. PTM has no dHCP template (in ideal
dHCP every atom is locally FCC or HCP), so dHCP is derived from the PTM labels:

- *Identification.* Per atom, not per layer (no projection, so no wrapping problem in periodic boxes): a
  close-packed atom passes when, among its close-packed neighbours within 1.2 bonds and with the layer
  normal from the HCP atoms among them, all but one in-plane share its label and all but one out of
  plane carry the other (at least 4 and 3 present). The passing set is then opened along the stacking
  (eroded 3 times, an atom staying while 2/3 of its out-of-plane neighbours do, and grown back 3 times
  within the passing set), so only runs of ≥ 7 alternating layers survive, whole. Non-close-packed
  neighbours are absent, not against, so the layers by a free basal surface pass. FCC atoms take the
  cubic variant whose mapped c axis lies along the normal, then the HCP frame: the A layers' two twin
  variants are 60° apart, inside the HCP group.
- *Orientation and segmentation.* dHCP uses the HCP group and frame; it shares a grain only with dHCP
  (no merge rule with FCC or HCP). An FCC/dHCP stack segments as two grains of 19 layers each: the h layer
  where the stackings meet alternates with the ABAC run and goes with it.
- *Defect reference.* Orthohexagonal cell (1, √3, 2·√(8/3)) with 8 basis sites, ABAC; every site 12
  first neighbours; the fit frees c/a and seeds like HCP.
- *Validation (Nd, a = 3.658, c = 11.799 Å, c/a below ideal):* a periodic block with 4 vacancies is one
  grain, V = 4 under global, patch and local at σ = 0–0.08 Å; with an octahedral interstitial V/I exact
  under global and local; free clusters at three random orientations, one grain each, V = 4. PTM labels
  equal OVITO's on every atom of the thermal block, and every OVITO FCC/HCP atom is dHCP. No atom is
  relabelled in FCC, HCP, an FCC twin, an intrinsic fault, FCC with 10 % and 30 % of layers faulted, or
  the HEA polycrystal fixture.
- *Stacking faults of dHCP* (`rmda segment --dhcp --dhcp-merge-faults`, Python `dhcp_merge_faults=True`;
  opt-in, segmentation only). Without it a single fault splits a dHCP crystal into 3–5 grains, even the
  dHCP itself in two. `stacking::dhcp_faults` finds FCC/HCP atoms in a run of at most `MAX_FAULT_LAYERS` = 6 non-dHCP
  layers (one short of the 7-layer minimum dHCP run) with dHCP on both sides: the layers to dHCP along
  each sense of the layer normal are a shortest path over close-packed out-of-plane bonds, normals from
  the hexagonal atoms spread to FCC atoms with none. Segmentation then treats those atoms as dHCP atoms
  in the dHCP frame (FCC ones by the cubic variant whose c lies along the normal, as `relabel_dhcp` does),
  so they join the crystallite around them by the core angle; the PTM labels are untouched. On Nd stacks
  an hcp-type and a cubic fault each give one grain of all 4320 atoms; an FCC|dHCP interface and a
  7-layer FCC slab between dHCP keep their own grains, and the dHCP grain gains no FCC atom. It is not
  offered by `rmda defects`: a faulted dHCP grain with 3 planted vacancies gives local V = 3 but I = 77 (hcp-type
  fault) and 21 (cubic) — the fault plane spans the box, so it is counted atom by atom against local
  lattices straddling two stacking variants — while the split grains count V = 3, I = 0 (global and patch
  gate the merged grain). Pair disorientations between dHCP and FCC grains are reported in the FCC
  frame.
- *Open:* a real (MD) La or Nd sample.

**Limits.** "dHCP grain" versus "FCC with a dense fault array" is a definition, not a measurement: here
it is a run of at least 7 alternating layers. Real c/a (La ≈ 3.23 rather than 2 × 1.633) is absorbed by
the fit, not by the identification.

### 28.2 GPU support

**Where it pays.** Per-atom independent stages with small, fixed work: k-NN queries, PTM template
matching (the largest stage in thread time), CNA, and the Wigner–Seitz nearest-site assignment.

**Where it does not.** I/O is parse-bound and already ~1.8 GB/s on 12 CPU threads. Union-find
segmentation, crystallite merging, adoption, the domain fixed point and other graph steps are irregular
and a small share of runtime.

**Targets.** Large HPC GPU partitions are both AMD (MI250X/MI300, ROCm/HIP) and NVIDIA (CUDA), so a
CUDA-only path would not run on many of them; each target must be checked, not assumed.

**Options in Rust:** kernels in HIP/CUDA C++ behind FFI, built by a feature-gated `rmda-gpu` crate (HIP can
target both vendors from one source); `cudarc` (NVIDIA only); portable compute (wgpu/Vulkan), whose
driver availability on HPC compute nodes is doubtful.

**Data sizes.** 128M atoms of f64 positions ≈ 3 GB, which fits one MI250X GCD (64 GB). Neighbour lists
need not be stored: k-NN runs on the fly, as on the CPU (§9.3).

**Requirements.** The CPU path stays the reference. GPU results match CPU classifications exactly, or
within a stated tolerance where float summation order differs, tested against the CPU on the existing
oracles (PTM, CNA, §16.8). Opt-in at build time: the default build has no GPU dependencies.

**Risks.** PTM's divergent per-atom work (hull and graph canonicalisation per structure) maps poorly to
SIMT; transfers can eat the gain if one stage is ported alone; two code paths double the testing
surface; ROCm toolchains on HPC systems move under the build.

**Plan.** Profile the 128M CPU run first (§27.2) to know where time goes at scale. Then port PTM matching
+ k-NN as a prototype and measure end to end, including host↔device transfer, before deciding on CNA and
the WS assignment.

---

## 29. Risks and open questions

1. **Patch-local WS is new research** (Mason leaves multi-grain to future work). Validated on bent,
   gradient and seam fixtures and on the thin film's control frame (§16.8). Remaining risks: phase
   continuity in heavily dislocated grains (the gate refuses them, e.g. the thin-film substrate at 0.61),
   and ambiguity near dislocation cores.
   - Open question: should dislocation cores become their own region class once DXA exists?
2. **What counts as a defect at a GB or surface is model-dependent.** RMDA reports those classes
   separately with conservation, rather than pretending to a precision it does not have. On
   mismatched-lattice boundaries the boundary class is large by construction (§16.4). The boundary width
   is echoed in every output.
3. **The automatic grain threshold** (OVITO's regression knee) can be unstable on few-grain samples.
   Fixed-threshold mode stays available and should be the default when there are fewer than 5
   crystallites above the minimum size.
4. **Float formatting speed** without a dependency. Std formatting in parallel chunks is the current
   path; a hand-written shortest round-trip formatter is about 400 lines of careful code.
5. **The extxyz Lattice convention** (ASE vs atomsk) cannot be auto-detected (`det H = det Hᵀ`). ASE is
   the default, with an explicit flag for atomsk and a warning heuristic for orthogonal boxes.
6. **The 128M laptop target** needs lean outputs (§24.2). The design does not promise more; HPC is the
   home of 128M runs.
7. **pyo3 and numpy MSRV drift.** pyo3 0.25 is the newest release on MSRV 1.72; a Python release it
   does not support will force a newer pyo3 and a higher MSRV.

---

## 30. Appendices

### 30.1 Glossary

| term | meaning |
|---|---|
| a-CNA | adaptive common neighbour analysis |
| PTM | polyhedral template matching |
| FZ | fundamental zone of orientations under a point group |
| WS | Wigner–Seitz cell occupancy defect analysis |
| reference | an object that generates addressed reference sites (§16.3) |
| crystallite | a low-angle-coherent atom cluster, the node of the grain merge graph (§15.2) |
| region class | Interior / Boundary / Surface / Excluded (§16.4) |
| ΔN | N_sites − N_atoms in a region; equals V − I |
| ARI | adjusted Rand index between two labellings |
| tile | a contiguous block of spatially ordered atoms processed by one thread (§10.3) |

### 30.2 LAMMPS source index

| topic | file |
|---|---|
| integer types | `src/lmptype.h` |
| box, remap, triclinic, general triclinic | `src/domain.h`, `src/domain.cpp` (`remap`, `x2lamda`, `general_to_restricted_rotation`) |
| data read/write | `src/read_data.cpp`, `src/write_data.cpp` |
| dump headers and bounds | `src/dump_atom.cpp` (`header_item_triclinic`, binary header), `src/dump_custom.cpp` |
| dump read | `src/read_dump.cpp`, `src/reader_native.cpp` |
| binary dump decoding | `tools/binary2txt.cpp` |
| binning | `src/nbin_standard.cpp`, `src/npair_bin.cpp` |

### 30.3 Decision records to write

- ADR-0001: symmetry convention (right multiplication).
- ADR-0002: pure WS occupancy and conservation.
- ADR-0003: patch-local reference sharing one index space; the phase-agreement gate.
- ADR-0004: grains via crystallites + merge; shortest-path orphan adoption.
- ADR-0005: namespaced keys and tables vs per-atom columns.
- ADR-0006: pread vs mmap on Lustre.
- ADR-0007: dependency policy (pyo3/numpy, no serde).
- ADR-0008: snap tolerances (global 2 % and 0.25 bond; patches 2 %) and coherent-bond unwrapping.
- ADR-0009: PTM neighbour order (distance, not Voronoi preorder) and the evidence.

---

## 31. References

RMDA is not a port of any of these. Wherever RMDA departs from a reference, the reason is written down;
wherever it claims parity, a test proves it.

**Methods**

1. P. M. Larsen, S. Schmidt, J. Schiøtz, "Robust structural identification via polyhedral template
   matching", *Modelling Simul. Mater. Sci. Eng.* **24**, 055007 (2016). — PTM (§14); the transcribed
   tables are under the PTM licence (`THIRD-PARTY-NOTICES.md`).
2. P. M. Larsen, "Revisiting the common neighbour analysis and the centrosymmetry parameter",
   arXiv:2003.08879 (2020). — interval CNA (§13).
3. K. Nordlund, R. S. Averback, "Defect production in collision cascades in elemental semiconductors and
   fcc metals", *Phys. Rev. B* **57**, 7556 (1998). — Wigner–Seitz defect analysis against a lattice
   fitted to the frame itself (`rmda ws --self`, §16.3).
4. D. R. Mason, F. Granberg, M. Boleininger, T. Schwarz-Selinger, K. Nordlund, S. L. Dudarev,
   "Parameter-free quantitative simulation of high-dose microstructure and hydrogen retention in
   ion-irradiated tungsten", *Phys. Rev. Materials* **5**, 095403 (2021). — self-referenced WS fit
   (T₀, origin), roto-strain field, void φ-field (§16.3, §17.2). It fits one global 9-DOF affine
   lattice snapped to the box (`n = nint(b⁻¹A)`), picks the origin on a 6³ grid by defect count, states
   that V ≠ I in general for the self-referenced method, and leaves multiple grains to future work. RMDA's
   per-grain and patch-local references are therefore new methods, validated as such (§16.8).

**Software consulted for conventions and used as test oracles**

5. LAMMPS (A. P. Thompson *et al.*, *Comput. Phys. Commun.* **271**, 108171 (2022)) — data model, box
   conventions, file formats, binning; `tools/binary2txt` as the binary-dump oracle. Source files in §30.2.
6. OVITO 3.15 (A. Stukowski, *Modelling Simul. Mater. Sci. Eng.* **18**, 015012 (2010)), documentation
   and Python module — CNA, PTM, Wigner–Seitz, grain segmentation, cluster and RDF semantics; parity
   oracle throughout.
7. ASE 3.22 (A. H. Larsen *et al.*, *J. Phys.: Condens. Matter* **29**, 273002 (2017)) — extended-XYZ
   grammar, cell conventions, minimum image (`find_mic`), parity tests.
8. atomsk b0.13.1 (P. Hirel, *Comput. Phys. Commun.* **197**, 212 (2015)) — format quirks (`.lmp` as a
   data file, component-major `Lattice`).
9. WarrenCowleyParameters, an OVITO Python extension by K. Sheriff
   (github.com/killiansheriff/WarrenCowleyParameters) — Warren–Cowley short-range order (§18.1).
