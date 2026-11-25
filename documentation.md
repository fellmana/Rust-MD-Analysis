# `rmda` — RMDA (Rust MD Analysis) user guide

An atomistic-analysis toolkit in Rust: structure identification (CNA, PTM),
neighbour statistics (RDF, Warren–Cowley), defect analysis (Wigner–Seitz and
voids, including a per-grain census for polycrystals and thin films with a
time-series mode), grain segmentation and grain-boundary detection, cluster
analysis, and a multi-analysis pipeline over the frame window. Command-line
first, with a library underneath and Python bindings.

- **Design document:** [`design.md`](design.md) — the specification, section by
  section, with the reasoning behind each choice. This manual refers to its
  sections as "design §N" or by title (*Grain segmentation and tracking*,
  *Defect subsystem*, …).
  Measured evidence (wall times, memory, OVITO agreement, validation counts)
  sits in design.md in the section it validates, and each section lists what the
  code does not do yet as **Open:** items.

> **Which command for a polycrystal?** `rmda defects --segment` segments the frame
> and counts vacancies and interstitials per grain (§3.7); `rmda segment` (§3.6)
> runs the same segmentation on its own and writes the `Grain` column for reuse.

---

## Contents

1. [Installing](#1-installing)
2. [How the analyses work](#2-how-the-analyses-work)
3. [Command-line usage](#3-command-line-usage)
4. [Performance](#4-performance)
5. [Python bindings](#5-python-bindings)
6. [Testing](#6-testing)

---

## 1. Installing

### Requirements

| | |
|---|---|
| Rust | 1.72 or newer (`cargo --version`) |
| Disk | ~2 GB for a debug build, ~400 MB release |
| Network | Only for the first build, to fetch crates |
| Python | Only for the optional bindings: CPython ≥ 3.9 with numpy (ASE optional) |

No MPI and no BLAS/LAPACK, and no Python either unless you build the optional
`rmda-py` bindings (§5) — the default build never looks for an interpreter. The
build is a plain Cargo workspace; every numerical routine is hand-written against
fixed-size matrices so there is nothing to link beyond libc.

### Build

```console
$ cargo build --release
```

This builds the `rmda` binary only, as `target/release/rmda`: the workspace's
default members are `rmda-core` and `rmda-cli`, so a plain build never touches
the Python crate. To put it on your `PATH`:

```console
$ cargo install --path crates/rmda-cli
```

The Python package, which also installs the `rmda` command, is built with pip
(§5):

```console
$ pip install .
```

### Check it

```console
$ rmda --version
rmda 0.1.0
$ rmda -h          # abridged
RMDA (Rust MD Analysis): analysis of atomistic molecular-dynamics frames

Usage: rmda [OPTIONS] <COMMAND>

Commands:
  run               Run a whole analysis from a TOML config ...
  explain           What `rmda run CONFIG` would do, without running it ...
  bench             Per-stage throughput on a synthetic crystal ...
  merge             Join the shards of a sharded series ...
  info              Print summary statistics of the input frames
  index             Write `FILE.rmdaidx`, a frame index ...
  census            Compute the neighbour census (per-atom coordination + histogram)
  cluster           Cluster analysis: connected components under a distance criterion
  rdf               Radial distribution function (g(r), coordination number, partials)
  wc                Warren-Cowley short-range-order parameters (design §17)
  structure         Common-neighbour analysis (per-atom structure type + histogram)
  ptm               Polyhedral template matching (per-atom structure type + RMSD cutoff)
  grain-boundaries  Mark the atoms sitting on a grain boundary (design §19 step 4)
  defects           Per-grain Wigner-Seitz defect census (design §16)
  segment           Grain segmentation (design §15) ...
  ws                Wigner-Seitz defect analysis
  void              Void census (design 17)
  analyze           Multi-analysis pipeline over the frame window (design 21, 25.2 subset)
  convert           Convert the input to extended XYZ, a LAMMPS text or binary dump, or a LAMMPS data file ...
  help              Print this message or the help of the given subcommand(s)
```

Every subcommand takes `--help` with its own flags:
`rmda ws --help`, `rmda void --help`, and so on. `-h` gives the one-line
summary, `--help` the full text.

### Input formats

Detected from the file extension; no flag needed. A name the table does not
cover is identified by its first bytes instead (an `ITEM:` line is a dump, an
`atoms` header a data file, a bare atom count an XYZ file, the `DUMP…` magic a
binary dump).

| Extension | Format |
|---|---|
| `.xyz` | Extended XYZ (one or more frames); `.extxyz` and other names are recognised by content |
| `.dump`, `.lammpstrj`, `.d` | LAMMPS text dump (one or more frames) |
| `.bin`, `.lammpsbin` | LAMMPS binary dump (`dump atom`/`dump custom` written to a `.bin` file) |
| `.data` | LAMMPS data file (single frame) |
| `.lmp`, `.lammps` | sniffed: these names mean a dump in some tools and a data file in others |

What each reader understands:

- **Text dumps** keep every column. `id`, `type` and the position triple (the
  first of `x y z`, `xu yu zu`, `xs ys zs`, `xsu ysu zsu` present) are routed
  to the atoms; `ix iy iz` become image flags; `element` becomes the type
  labels; every other column becomes a per-atom property under its own name.
  `ITEM:` blocks may come in any order before `ATOMS`, and `UNITS`/`TIME` are
  read.
- **Binary dumps** (LAMMPS's own binary layout, little-endian): old pre-magic
  files (read as `dump atom`, `id type xs ys zs [ix iy iz]`) and revision-2
  files with column names, units and time; orthogonal, restricted triclinic
  and general triclinic boxes (see below). The reader is checked against LAMMPS's
  `tools/binary2txt` on `tests/data/binary2txt_oracle.{bin,dump}`. `rmda convert`
  writes them (`-o out.bin` or `--format bin`, §3.15).
- **Data files**: atom styles `atomic`, `charge`, `full`, `molecular`, `bond`,
  `angle` and `sphere`, from the `Atoms # style` comment; without the comment
  the style is inferred from the column count (5 or 8 columns are `atomic`,
  7 or 10 are `full`; 6 or 9 are ambiguous — `charge` or `molecular` — and are
  refused unless `--atom-style charge|molecular` names the style; a comment
  naming a different style than `--atom-style` is an error). An `Atom Type Labels`
  section is read, and the labels may be used in place of numeric types in the
  `Masses` and `Atoms` rows; `rmda info` then prints them (`types: Cu Ni`).
  Trailing image flags, `Velocities` (as `vx vy vz` properties), `N atoms #
  comment` headers and inline `#` comments are handled; `mol`, `q`, `diameter`
  and `density` are kept as properties.
- **Extended XYZ**: see §3.17 for the header this tool writes. On read,
  `Lattice=` may carry nine numbers (with the origin in a separate `Origin=`
  key, OVITO's spelling, or zero) or twelve (nine plus the origin). The comment line follows ASE's grammar: quoted
  values with escapes, `[...]`/`{...}` arrays, and bare flags (`is_relaxed`),
  which are skipped without ending the line. An `id` column becomes the atom
  ids, and a `species="Cu Ni"` key names the types in order. Without that key,
  a file with a `type` column labels each type with the symbol its atoms carry
  (`Ni … 1`, `Cu … 2` gives type 1 = Ni, type 2 = Cu, as OVITO reads it); a
  type whose atoms carry different symbols, or whose symbol another type also
  carries (the `X` this tool writes for unlabelled types), gets no label.
  Without a `type` column the symbols are the types, numbered in sorted order.
  `Timestep=` and `Time=` are read into the frame.
- **General triclinic boxes** (LAMMPS data `avec bvec cvec abc origin`, dump
  `BOX BOUNDS abc origin`, binary `triclinic 2`) are rotated into LAMMPS'
  restricted frame on read, as LAMMPS does: the cell, the positions, and the
  `vx vy vz`, `fx fy fz`, `mux muy muz`, `omega*`, `angmom*` columns.

**Skipping columns.** The analysis commands skip velocities and image flags by
default (`vx,vy,vz,ix,iy,iz`), which no analysis reads, so they are stepped
over in every row and never stored; `info` and `convert` keep everything.
`--skip-columns LIST` replaces that default with your own list, in every
format (`--skip-columns vx,vy,vz,c_pe`), and `--skip-columns ''` keeps every
column everywhere. On a large frame this is the difference between roughly 35
and 73 bytes per atom while reading.

Any input may be **gzip-compressed** by appending `.gz`
(`traj.dump.gz`, `reference.data.gz`): the suffix is stripped before the table
above is consulted, and the sniffed header is gunzipped too, so a compressed
file resolves to exactly what its uncompressed name would. A gzipped
trajectory is streamed: one decompressed frame is resident at a time, and
concatenated gzip members work. Decompression runs on all threads when the
file allows it: a file made of many small members (what `rmda convert -o
x.gz` and bgzip write) is inflated member by member in parallel as it is; any
other `.gz` (one member, as LAMMPS `dump *.gz`, `gzip` and `pigz` write) is
inflated serially until `rmda index FILE.gz` has written a gzip index next to
it (§3.16), after which it too is inflated in parallel. Either way the bytes,
and so every parsed value, are exactly those of the serial decoder; `--threads
1` uses the serial decoder itself. On output, only `rmda convert` compresses (an `-o`
ending in `.gz`, §3.15); every other output is written uncompressed. A gzipped file *without* the `.gz` suffix is
not detected (say `cannot detect format`), because detection and loading key
off the same suffix and must agree.

Which frames are read is explicit: by default an analysis reads **frame 0** of a
multi-frame file, and the global `--frame N`, `--start N`, `--end N`, `--stride N`
pick a different frame or window; `rmda info`, `rmda convert` and `rmda analyze`
process every frame the window selects, and `rmda defects FILES... --out DIR`
reads one frame from each file (§3.7). `-o` writes the per-atom properties of the
frame that was analysed. This is stated rather than silently assumed because
attaching one frame's classifications to another frame's atoms is a mistake that
produces a file that looks right. Frames outside the window are stepped over by
counting rows, never parsed.

**Seeking in a long trajectory** (`rmda index`, §3.16). Reaching `--frame 5000`
normally means scanning the 5000 frames before it. `rmda index traj.dump` writes
`traj.dump.rmdaidx`, and every later read of `traj.dump` seeks by it, as long as
the trajectory's size and modification time still match the ones recorded — a
rewritten file is never read through a stale index. On a `.gz` file the same
command writes a gzip index instead, which makes reading it parallel (it still
cannot seek to a frame).

### A file with no box

An extended-XYZ file with no `Lattice=` line carries neither a box nor a
periodicity. It is read in **the box its own atoms occupy** — the atoms are
where they are, the box is as long as they are, and the density that comes out
of it is the density they actually have, so density-derived radii and the
neighbour search behave as they do for any other frame.

A file that does not say it is periodic is not treated as periodic, because a box cut to the atoms and treated as periodic
would put each atom's own image next to it. A periodic box for such a file is
stated with `--cell A B C` (see
[Command-line usage](#3-command-line-usage)).

---

## 2. How the analyses work

Each subsection gives the theory in the terms the code uses, then where to find it.
Every claim here has a test behind it; the test paths are listed so you can check
the claim rather than trust it.

### 2.1 Shared geometry — `cell.rs`, `neighbor/cell_list.rs`

Everything else depends on getting coordinates and periodicity right, so this is
where the subtle bugs live and where the tests are densest.

**`UnitCell` matrix convention.** A cell is a 3×3 matrix `h` whose **rows** are the
lattice vectors. Fractional → Cartesian is therefore `lamda2x(λ) = hᵀλ`, and
Cartesian → fractional is `x2lamda(x) = (h⁻¹)ᵀx`.

> **Trap worth knowing:** `Mat3::mul_v` sums over *columns*. It is the transpose of
> `lamda2x` and the two agree only for a symmetric matrix. Building a fixture with one
> and tiling with the other silently produces a lattice that is not commensurate with
> its own cell — distinct sites coincide and the first shell lands at the wrong
> distance. All fixture code in this repository goes through `lamda2x` for this reason.

**Minimum image.** `min_image(dr)` reduces a displacement vector to the shortest
lattice image. It goes through the inverse directly rather than through
`x2lamda → round → lamda2x`, because that route subtracts the cell origin first —
harmless at origin zero, but it picks the wrong lattice image anywhere else
(a *valid* image of the pair, just a distant one).

**`wrap`.** Folds a position into the cell. Folds *any* number of box lengths, in
fractional space for triclinic cells and Cartesian space for orthorhombic ones. A
single-fold implementation is correct only for coordinates already within one box of
it, which is not the state of an unwrapped LAMMPS dump.

**Neighbour cell list.** Bins atoms **fractionally**, through the cell's own
inverse: `n_d = ⌊height_d / binsize⌋` bins across axis `d`, periodic axes wrap with
`rem_euclid`, non-periodic axes clamp. Cartesian binning with a ±1
Cartesian stencil would miss neighbours in a rotated cell, because the query box has to
be wider than the true shell by a factor set by the cell's aspect ratio — it fails
by *silently under-reporting*, which is the worst way to fail.

Tests: `crates/rmda-core/src/cell.rs` (18 tests),
`crates/rmda-core/src/neighbor/cell_list.rs` (12 tests, including a rotated FCC
block that must yield exactly 12 first-shell neighbours per site).

### 2.2 Neighbour census — `analysis/neighbor_census.rs`

Counts neighbours within a cutoff and reports the mean coordination and total pair
count. The base layer everything else indexes through: a shared cell list is built
once per run at the largest cutoff any analysis asks for, so adding an analysis does
not add a neighbour search.

CLI: `rmda census --cutoff 3.5 file`

### 2.3 Common-neighbour analysis — `analysis/cna.rs`

**Theory.** For each atom, collect its neighbours within a cutoff and classify them
by how they bond to each other. The signature of a bonded pair is the sorted triple
of common neighbours shared by both. The distinct signatures are

| Signature | Structure |
|---|---|
| 12 × 4-2-1 | FCC |
| 6 × 4-2-2 | HCP |
| 8 × 4-4-4 | BCC |
| 12 × 5-5-5 | Icosahedral |
| 6 × 6-6-6 | Simple cubic |

**Adaptive cutoff.** The default is `--mode adaptive`, which sets the radius from the
local number density rather than from a fixed number. A fixed radius silently
under-coordinates atoms in a hot or expanded lattice, which shows up as a wrong
structure type rather than as an error.

**The other modes** are OVITO 3.15's `CommonNeighborAnalysisModifier` modes and its
`IdentifyDiamondModifier`:

| `--mode` | OVITO | What it does |
|---|---|---|
| `adaptive` (default) | `AdaptiveCutoff` | per-atom cutoff from the 12 (FCC/HCP/ICO) or 14 (BCC) nearest |
| `interval` | `IntervalCutoff` | Larsen's interval CNA: neighbour-neighbour bonds are added shortest first and every gap whose bond graph has the right coordination is tried; the valid structure with the widest gap wins. Parameter-free and more tolerant of heat (FCC at σ = 0.15 Å: 250 of 256 FCC against adaptive's 219) |
| `fixed --cutoff R` | `FixedCutoff` | one global cutoff; exactly 12 or 14 neighbours inside it, else OTHER |
| `bond --cutoff R` | `BondBased` after `CreateBondsModifier(R)` | one `(common, bonds, chain)` signature per bond over every bond an atom has (no 12/14 gate) |
| `diamond` | `IdentifyDiamondModifier` | FCC/HCP CNA over the 12 second neighbours reached through the 4 nearest; writes **its own codes**: 0 other, 1 cubic diamond, 2/3 its first/second neighbours, 4 hexagonal diamond, 5/6 its first/second neighbours |
| `shell --low A --cutoff B` | — | the neighbours in (A, B], exactly 12 or 14 of them, bonded at B. Not an OVITO mode |

`--full` emits the per-signature counters
(`cna_n421`, `cna_n422`, `cna_n444`, `cna_n555`, `cna_n666`; for `bond` they count
bonds).

**Validation.** Adaptive matches OVITO's `CommonNeighborAnalysisModifier` exactly on all
1,535,326 atoms of the reference fixtures, and results are bit-identical between 1
and 4 threads. Interval, fixed, bond-based and identify-diamond agree with OVITO 3.15
atom for atom on the four reference snapshots (Al, W, Zr, HEA) and on synthetic
thermal FCC, BCC, HCP, icosahedral, cubic/hexagonal diamond (Si, Ge) and graphite
crystals. Tests: `crates/rmda-core/tests/cna_golden.rs`, `crates/rmda-core/tests/cna_reference.rs`,
`crates/rmda-core/tests/structure_variants_ovito.rs`.

### 2.4 Polyhedral template matching — `analysis/ptm.rs`

**Theory.** Fit the neighbourhood of each atom to a set of convex polyhedron
templates (FCC, HCP, BCC, icosahedral, simple cubic, …) by minimising the RMSD
between observed and ideal neighbour vectors over rotations *and* uniform scale.
Atoms whose best RMSD exceeds `--rmsd-cutoff` (in units of mean neighbour length,
OVITO's default 0.1) are reported as `OTHER`.

This is rotation- and size-invariant where CNA is not, and it is the classifier
that supplies per-atom orientation quaternions — which is what grain segmentation
(§2.5) consumes.

`--types` selects templates. The default is OVITO's set, `fcc,hcp,bcc`; upstream
`lib_ptm`'s C++ default also tests `ico`, and `ico`, `sc`, cubic diamond (`dcub`),
hexagonal diamond (`dhex`) and `graphene` are available on request, as they are in
OVITO (which ships all five disabled). The diamonds and graphene are matched on a
two-shell environment: the 4 (3) nearest neighbours and 3 (2) neighbours of each of
those, picked by rank in each neighbour's own list (upstream
`calculate_two_shell_neighbour_ordering`). Enabling a structure can change other
atoms' labels, since the lowest RMSD wins; on the four reference snapshots adding
the diamonds and graphene changes nothing (OVITO agrees). `rmda segment` and `rmda
defects` always use the `fcc,hcp,bcc` set.

**Neighbour order.** OVITO hands PTM the 18 nearest neighbours sorted by
decreasing Voronoi face solid angle, so its FCC/HCP templates see the 12 largest
faces rather than the 12 nearest. RMDA uses distance order by default;
`--ptm-order voronoi` on `rmda ptm`, `rmda segment` and `rmda defects` (Rust:
`Ptm::compute_ordered(.., NeighbourOrder::Voronoi)`, `Ptm::run_ordered`) reproduces
OVITO's types and RMSDs atom for atom on all four fixtures, at about 2.5× the PTM
time. With the two-shell structures enabled, Voronoi order also reproduces OVITO's
types, RMSDs and orientations atom for atom on synthetic thermal diamond, lonsdaleite,
graphene and graphite; in distance order 0.3–0.7 % of atoms differ at σ ≥ 0.15 Å.

**Opt-in outputs.** `--outputs ordering,F,template` (Python `outputs=[...]`, Rust
`Ptm::run_extended` with `ptm::output` flags) adds three columns, ported from upstream's
`find_alloy_type`, `ptm_remap_template` and `calculate_deformation_gradient`:

- `StructureOrdering` (u8): OVITO's *Ordering Type* from the particle types —
  0 none, 1 pure, 2 L1₀, 3 L1₂ (A: majority-type centre), 4 L1₂ (B: minority-type
  centre), 5 B2, 6 zincblende/wurtzite, 7 boron nitride. FCC environments are
  canonicalised over the cubic group and compared with the three FCC orderings; B2,
  zincblende/wurtzite and BN require every inner neighbour of the other type and
  every outer one of the centre's. A ternary environment is `none`.
- `StructureDeformationGradient` (9 × f64, row-major `F_ij` at `3i+j`): OVITO's
  *Elastic Deformation Gradient*, the least-squares map from the (conventional)
  template onto the scale-normalised environment. Up to OVITO 3.16 its raw component
  array is the same nine numbers, and its labels (`XX YX ZX …`) read column-major, so
  the matrix OVITO *labels* is Fᵀ; OVITO 3.17 stores the transpose under the same
  labels. Graphene has `F_zz = 1`.
- `StructureTemplate` (u8): which of the structure's alternative templates the
  conventional orientation and F refer to (HCP, cubic diamond, graphene 0–1,
  hexagonal diamond 0–3, others 0).

Three more, by name:

- `correspondences` → `Correspondences` (int64): OVITO's *Correspondences* property
  (written there whenever orientations are), upstream's `ptm_encode_correspondences`
  of which neighbour sits on which template site. For FCC, HCP, BCC, ICO and SC it is
  the permutation of the 18 nearest neighbours (by distance rank; template order,
  unused ranks appended in increasing order) as one mixed-radix index; for the
  diamonds and graphene, the inner atoms (15 bits) and each inner atom's outer atoms
  (11 bits each), each as a partial permutation of 13. Bits 62–63 hold the template
  index, so hexagonal-diamond templates 2–3 come out negative, as in OVITO. Decode
  with OVITO's `PTMNeighborFinder` or upstream's `ptm_decode_correspondences`.
- `residual` → `StructureDeformationResidual` (3 × f64): the F fit's squared
  residual per Cartesian axis, `Σ_k ((F t_k)_i − s_k,i)²` over the template points
  `t` and the barycentred, scale-normalised environment `s` (upstream PTM's `F_res`,
  as in the library LAMMPS bundles; OVITO 3.15 does not output it).
- `polar` → `StructurePolarU`, `StructurePolarP` (9 × f64 each, row-major): upstream's
  polar decomposition `F = P U`, U the rotation and P = F Uᵀ the symmetric left
  stretch (again an upstream output OVITO lacks).

As in OVITO, all of them describe the best fit even where the RMSD cutoff rejects it
(zero where nothing matched). With `--ptm-order voronoi` they equal OVITO 3.15 atom
for atom: ordering and correspondences exactly, F to 10⁻¹⁴, on B2, L1₂, L1₀,
zincblende, wurtzite and h-BN fixtures with noise and antisites, strained FCC/HCP and
the four reference snapshots (`crates/rmda-core/tests/ptm_extras_ovito.rs`, `crates/rmda-core/tests/ptm_oracle.rs`); U
and P are bitwise upstream's `polar_decomposition_3x3`. Default outputs and the
compact record used by segmentation and defects are unchanged.

**Validation.** Passes the OVITO oracle on four fixtures; all six PTM properties and
the quaternion norms are validated at the same scale. Tests: `crates/rmda-core/tests/ptm_oracle.rs`,
`crates/rmda-core/tests/ptm_reference.rs`, `crates/rmda-core/tests/ptm_refdata.rs`, `crates/rmda-core/tests/ptm_extras_ovito.rs`.

### 2.5 Bond-graph grain labelling — `analysis/grains.rs`

*The grain labelling inside `rmda grain-boundaries` (§3.5). It has no command of
its own: `rmda segment` (§2.8) is the grain segmentation, and the one `rmda
defects` uses.*

**Theory.** Every other analysis here fits one reference lattice to the whole
frame, which is exactly the assumption a grain boundary invalidates. This module
supplies the missing first step: which atoms belong to the same grain.

Two atoms share a grain when all three hold — they are neighbours within the
cutoff, their structure types are compatible, and the **disorientation** between
their orientations is small. Structure type alone is not enough (a stacking fault
is a boundary in the structure-type view and not a grain boundary in the
material); orientation alone is not enough either. So type is a cheap pre-filter
and orientation is the actual test.

**The disorientation** is `2·acos(max|dot(qa, g·qb)|)` maximised over the
structure's point group. Maximising `|dot|` minimises the angle, and taking it
over the group is what makes this a *dis*orientation: two orientations differing by
a symmetry rotation of the lattice describe the same physical orientation. The
group is applied to the *second* quaternion deliberately — the alternative
`dot(qa·g, qb)` is equal only when `g` commutes with `qa`, which is false in
general, so the swapped form quietly measures the angle to the wrong coset.

Grain ids follow the **lowest member index**, not discovery order and not size.
Discovery order depends on the cell list's bin layout and so on the frame's
origin; size order renumbers a grain when an unrelated grain grows. Both produce
output that looks stable and is not.

**Cleanup** has the two forms OVITO distinguishes:

- *Reject.* A cluster below `--min-size` atoms (default 1) is discarded: its
  atoms become `-1` and stay there (`grain_rejected_atoms`). This is the "not a
  grain and not to be counted" knob.
- *Dissolve.* A cluster of at least `--min-size` but below `--min-grain-size`
  atoms (default 100, OVITO's `min_grain_size`) is a dislocation core, a patch
  of disordered boundary or the shell around a vacancy rather than a grain. Its
  atoms (`grain_dissolved_atoms`) are handed to the neighbouring grains.
- *Adopt orphans.* Atoms PTM could not classify are handed to the nearest
  adjacent grain as well, as OVITO's `orphan_adoption` does; the scalar
  `grain_adopted_atoms` counts every atom adopted. The reach is 1.75 bond
  lengths per pass, enough to cross the unclassified shell around a vacancy.
  `--no-adopt-orphans` leaves the unclassified atoms at `-1`. Dissolved atoms
  are handed on either way, as OVITO's `min_grain_size` does.

Survivors are renumbered contiguously. Atoms that were never candidates —
`OTHER` type without `--include-other` — are counted in `grain_other_atoms`,
separately from rejected ones, because the two have different causes and a
downstream grain-local measurement needs to tell them apart.

**Merging.** `--merge-threshold DEG` adds a pass one level up, on the *grain*
graph: each pair of grains is joined by an edge weighted by the **minimum**
disorientation over every bond between them, and grains are unioned while that
weight is at or below the threshold. Off by default.

The minimum is what makes it work. A boundary that is clean everywhere except
one noisy bond pair is a real boundary that measurement split; its *mean*
disorientation stays high at any threshold, so the two halves could never be
reunited. One union-find pass suffices because merging two grains can only lower
the minimum distance to any third grain, so no pair crosses the threshold during
the pass that had not already crossed it before.

Grains separated by incompatible structure types are never joined, at any
threshold: a disorientation is undefined across a type mismatch, and `NaN` fails
the comparison rather than passing it.

Merging fixes over-splitting. It cannot undo a bridge — labelling has already
fused the pair by the time a grain graph exists, and no threshold on that graph
can separate them.

**Not implemented.** OVITO's Node-Pair-Sampling partitioning, and OVITO's
automatic `ThresholdSelection` for the merge threshold (`--merge-threshold` is
explicit instead). Connected-components labelling is what partitions, and a
single bridge of sub-threshold bonds fuses two grains. The segmentation of §2.8 (`rmda segment`) is built to
avoid exactly that cost and is the one `rmda defects` uses.

### 2.6 Wigner–Seitz defect detection — `analysis/ws.rs`, `analysis/defects/explicit_ws.rs`

**Theory.** Build a reference lattice, generate its sites, assign each atom to at
most one site, and read the defects off the occupancy: empty sites are vacancies,
over-occupied sites are interstitials. This is the model of OVITO's
`WignerSeitzAnalysisModifier` and of the self-fitted Wigner–Seitz analysis of
Nordlund & Averback (Phys. Rev. B 57, 7556 (1998)), which is the point — it makes
the counts checkable against something else.

Two design points that change the numbers:

- **The reference is anchored at the box corner, not at absolute coordinate zero.**
  Every LAMMPS `write_data` file has nonzero bounds (the HEA fixtures sit at
  xlo = 2.3213), so anchoring at zero shifts the lattice by −xlo relative to the
  atoms — 2.32 Å on a 2.49 Å nearest-neighbour spacing, enough to wreck the
  assignment and report tens of thousands of vacancies in a *perfect* crystal.
- **Each atom claims at most one site**, so the assignment is deterministic and two
  atoms taking the same nearest site is the interstitial tally rather than a conflict
  to resolve arbitrarily.

**Three ways to get the reference**, in increasing order of what they assume:

| Mode | Assumes | When to use |
|---|---|---|
| *(default)* `--lattice fcc --nn-distance 2.49` | the lattice tiles the box exactly | you know the lattice and the box fits it |
| `--reference FILE` | nothing about commensurability | you have a reference configuration; survives cell strain and a defect count |
| `--self` | only that the structure is periodic | no reference exists; fits scale and orientation from the atoms |

`--reference` runs the explicit-reference census (`rmda defects --ws-reference`,
design §16.3a): every atom goes to its nearest reference site, with no
tolerance and no search radius, as OVITO's `WignerSeitzAnalysisModifier`.
`ws_config_tolerance` (`--tolerance`, default 1.05 × the reference's
first-neighbour distance) only sets the diagnostic `ws_config_unmatched`, the
atoms farther than that from their site — which are still counted.

`--self` reports `ws_fit_a`, `ws_fit_tolerance` and `ws_self_seeded` so a fit can be
*checked* rather than trusted. An unseeded fit starts from the identity and cannot
find a grain turned away from the box axes; pass `--orientation w x y z` (a PTM
quaternion) when that is possible.

**Void counting on the `ws` lattice.** `rmda void` (§2.6b, §3.14) is the void
command, on the per-grain census of `rmda defects`; `rmda ws --void-min-sites N`
is the same idea on `ws`'s single lattice. It groups empty sites into craters
by a **distance criterion** — two empty sites are one void when their lattice points
are within a bond of each other, not when their cell indices are neighbours, which
would join sites a whole lattice constant apart. `N = 1` counts every empty site,
which is the vacancy count under another name; `N ≥ 2` asks for craters. Per-site
volume is the exact analytic Fourier-trap `a³/4`, so a void's volume is its site
count times that, with no sampling.

Two limitations stated rather than hidden: the region is the cells the *atoms*
occupy, so a crater open to a free surface shrinks the region instead of appearing in
it; and a vacancy with no atom on the far side has no interior to measure against.

**Validation.** Counts are asserted as literals, not recomputed from the lattice
parameter — a region that quietly changes size with the gauge still balances atoms
against sites, it just reports hundreds of vacancies that are not there. On the
256k-atom relaxed fixtures: 0 defects in the clean reference, the planted count in
the defective one. 17 CLI end-to-end tests in `crates/rmda-cli/tests/ws.rs`.

### 2.6b Voids on the defect census — `analysis/defects/voids.rs`

**Theory** (design §17.1). A void is a cluster of **vacancy sites** of the
`rmda defects` census (§3.7): vacancies join when they are first neighbours (the
cutoff of design §16.5, halfway between the first and second shells of their grain), and a
cluster of at least `--min-sites` (default 2) is a void. Interstitials take no
part. Every reference of `rmda defects` works: global, patch, local, explicit.

- **Volume** `V = Σ Ω_site`, `Ω_site = |det T| / n_basis` of each member's grain
  (its fitted conventional cell over its basis count, for the local reference
  the grain-wide prefit; the ideal cell at the measured bond under the explicit
  reference, which fits no `T`).
  **Equivalent radius** `(3V/4π)^(1/3)`. The census fills enclosed cavities whole,
  so a closed void's interior is counted and its volume is exact on a lattice.
- **Class.** `surface` when the void is open: a member has a first-shell lattice
  position holding neither an atom nor a vacancy — empty space the census did not
  count, such as vacuum. Else `boundary` (a member in a grain-boundary zone) or
  `bulk`. One atomic layer over a void closes it.
- **Open voids under a fitted reference.** The census keeps space connected to
  vacuum out of its domain (§3.7), so a pit or a void broken open to the surface is
  mostly not counted: its `surface` volume is a lower bound, often zero. The
  explicit reference (`--ws-reference`) counts every reference site, so there a
  pit is one `surface` void with its whole volume.

**Validation** (`crates/rmda-cli/tests/void_a.rs`): BCC W with spherical clusters
of 15, 55, 89 and 169 sites are counted whole with volume `n·a³/2` (to 1e-6
fitted, 1e-3 local and explicit, 5e-3 with 0.05 Å noise) under every reference;
the same under a free surface and in a Σ5 twist bicrystal; byte-identical output
for lattice-ordered, reversed and shuffled files.

### 2.7 Void volume by the φ field — `analysis/void_b.rs`, `analysis/roto_strain.rs`

> **Experimental, and gated on `--experimental`.** Read §2.7.1 before quoting a
> number from this.

**Theory.** For a point `x`, the crystal is defined by a set of ideal neighbour
directions around each atom. Define

```
ψ_i(x) = max_k | n̂_k · (x − x_i) |        n̂_k = 2 v̂_k / |v̂_k|²
φ(x)   = min_i ψ_i(x)
```

`φ = 1` is exactly the Wigner–Seitz cell boundary of a perfect lattice: a point is
outside every atom's cell when `φ > 1`. So **`{φ > 1}` is the void region**, and a
void volume is an integral over a superlevel set — which sees a void of any size,
and a distortion that never empties a site. Those are the two limits method A has.

Note the two halves of that definition, because they are the whole design:

- the **face normals** `n̂_k` come from the *ideal* configuration;
- the **cell centres** are the *observed* atom positions.

The gap between them is the signal. An atom off its site no longer fills its own
cell, φ rises above 1 around it, and the region is counted as void. This is method
A's question asked continuously, which is why the two methods must agree on a single
vacancy's volume (they do: `a³/4`) and why this is a refinement of method A rather
than an independent measurement.

**Volume by Monte Carlo.** `estimate_void_volume` samples the region and reports
`volume ± standard_error`, with the hit rate, because a Monte Carlo estimate is a
sample mean and asserting three significant figures would be asserting the
generator's luck. The same seed makes two fields comparable.

**ε → 0 extrapolation.** Thermal broadening makes `V(ε)` fall with ε; the design
(*Void detection*) asks for `V(1+ε)` extrapolated to `φ = 1` by fitting a quadratic. `extrapolate_to_phi_one`
requires **at least 3 distinct** ε values (three copies of one give three rows of the
same design matrix) and returns the fitted coefficients, the residual, and two
predicates — `slope_negative` (a rising curve means the measurements are not a void
eroding) and `is_trustworthy` (a large residual relative to the answer means the
quadratic does not describe them).

**Roto-strain T(x)** (Mason *et al.* 2021). `--strain` fits a per-node affine map
`T(x)·x + δ` on a fractional mesh, with Gaussian weighting by distance to the node,
rejection of pairs with `|Δv| > a₀/4`, and coarse-to-fine passes halving both σ and
the mesh spacing. Ideal directions come from the declared lattice, not from a PTM
template. It recovers known dilations, rotations and shears to ~1e-6.

#### 2.7.1 What the measurements say — and why the gate is closed

On the two 256,000-atom relaxed HEA fixtures (`crates/rmda-core/tests/roto_strain_hea.rs`,
40,000 samples, seed 11):

| fixture | rigid | strained | reduction | max \|T−I\| |
|---|---|---|---|---|
| relaxed, no defects | 2.861 % of box | 2.720 % | 4.9 % | 0.036 |
| relaxed, planted defects | 2.909 % of box | 2.722 % | 6.4 % | 0.090 |

A relaxed solid has no porosity, yet the rigid field reports 2.86 % — 0.31 Å³ per
atom against an atomic volume of 10.95 Å³, which is a shell of thermal thickness.
Strain removes ~5 % of it and leaves 2.72 %. The roto-strain premise is that the
displacement field can be absorbed into a per-node affine map; on a real alloy most
of that field is non-affine thermal noise, and a strain has no representation for
that. The controlled version is the unit test `strain_cannot_remove_a_non_affine_shell`.

So **use method A** (`rmda void`, §3.14) for a void number you intend
to trust. Method B is here to be measured and debugged.

**The pure-W gate** (design §17.2, `tools/void_b_gate.py`). On BCC W with one
spherical cluster of 1–89 vacancies, extrapolated to `φ = 1`: unrelaxed −2.6 to
−0.1 %, relaxed at 0 K with a Finnis–Sinclair potential −7.1 to −0.9 % — **inside
the 10 % gate** — but with 0.03 Å of Gaussian thermal noise +1554 % (1 site) to
+17 % (89 sites). The quadratic passes only at 0 K. (BCC's cell uses both the 8
first-shell and the 6 second-shell faces; the first shell alone reads every BCC
void 5–20 % small.)

**The shell-free window** (`--thermal-window`, design §17.2) passes the gate
with thermal noise. Two neighbours displaced apart open a gap between their
cells that is void up to `φ = v·r/|v|²`; the largest of these over the frame,
the **largest face gap** `g`, is where the thermal shell ends (0.03 Å of noise:
g ≈ 0.06; 0.06 Å: g ≈ 0.12). Above `g` only void is left, and it erodes
homothetically: a vacancy's cell shrinks as `(1 − ε)³`, so `V^(1/3)` is linear
in ε. The window measures 7 thresholds over `[max(g, 0.01), +0.3]` and
extrapolates `V^(1/3)` to `φ = 1`. Pure W, 1–89 vacancies, relaxed or not,
0.03 and 0.06 Å of noise: within 10 % everywhere and within 4 % from 9 sites
up; a perfect box with noise reads 0 (the quadratic: 115 and 405 Å³). On the
relaxed HEA fixture with planted defects it reads 85 ± 14 Å³ against method
A's 7 vacancies (76.7 Å³), and 0 on the defect-free one. It stays behind
`--experimental` until it has been checked on a thermal MD frame (the gate's
noise is uncorrelated Gaussian); one far-displaced atom anywhere (a dislocation
core, a boundary) raises `g` for the whole region, which costs precision, and a
frame with `g > 0.6` is refused.

**Search radius and single-ε excess.** The φ field is searched to the radius
exact for one threshold, so it is built for the largest ε asked for and a larger
one is refused: measured at a larger ε, points held only by unsearched atoms would
read as void. `--excess` with a single ε prints the measurement, marked `not
extrapolated`, since a quadratic through one point is no extrapolation.

What strain *does* do is asserted rather than assumed: it never inflates the void,
and it does not remove more from the defect-bearing fixture than from the clean one
(1263 Å³ of excess removal against 4277 Å³ of sampling noise).

A third measurement bounds the method's resolution: a 40,000-sample volume integral
resolves ~2.1 × 10³ Å³ on these fixtures while the two files differ by 1323 Å³, so
**a volume integral cannot resolve planted point defects at all.** That is method
A's job, and it is why this command is not a defect detector.

**Validation where it is exact.** `crates/rmda-core/tests/void_b_benchmark.rs`: carved cubic clusters
against exact `n·a³/4` (measured 0.95–0.97 ×), spherical voids double-bracketed by
the carved sphere and the vacated-cell total, a slab-vs-ball pair with identical
site populations (the measurement is shape-sensitive, not a cell count in disguise),
extrapolation end-to-end (1269.42 Å³ against the exact 1271.18 Å³), a 92-site cavity
cross-checked against method A, and an uncarved-block negative control.

### 2.8 Grain segmentation — `analysis/grain_seg.rs`, `analysis/grain_track.rs`

The segmentation `rmda segment` and `rmda defects --segment` use. It answers the
same question as §2.5, and is built so that a chain of small, individually
acceptable misorientations cannot bridge two real grains:

1. **Crystallites.** Each crystalline atom is bonded to its PTM neighbours of the
   same structure (the first 12 for FCC/HCP, 14 for BCC) when their
   disorientation is below `--core-angle` (default 2°). Union-find over those
   bonds gives orientation-coherent patches.
2. **Crystallite graph.** Bonds between different crystallites are counted per
   pair, and every crystallite gets a symmetry-aligned mean orientation.
3. **Merge.** Pairs are merged in order of the disorientation between their
   *current mean* orientations, while it is below `--threshold` (default 5°)
   and the pair shares at least `--min-bonds` bonds (default 4). Comparing means
   rather than following atom-to-atom chains is what stops bridging. With
   `--stacking-faults merge` (the default, as in OVITO), FCC and HCP
   crystallites are compared through the close-packed frame map, so the HCP
   layers of a stacking fault or twin plane join the FCC grain around them
   while the two twins (60° apart) stay separate grains; a merged cluster keeps
   the structure of its larger part. `separate` never lets FCC and HCP share a
   grain.
4. **Cleanup.** Grains below `--min-size` (default 100) dissolve. Unassigned
   atoms, non-crystalline ones included, then take the grain nearest to them
   along a path through unassigned atoms (shortest path over 12-nearest-neighbour
   bonds), which reproduces OVITO's orphan adoption on 99.4 % of the atoms of
   the HEA polycrystal fixture. `--no-adopt` leaves them at `-1`. Grains are
   numbered by decreasing size, ties by lowest atom index.

Every pass is deterministic for any thread count.

**OVITO's algorithm (`--mode auto`).** `rmda segment --mode auto` (and
`rmda defects --segment --seg-mode auto`, `rmda.segment(f, mode="auto")`) runs
OVITO 3.15's GrainSegmentationModifier algorithm instead (`analysis/grain_auto.rs`):
node pair sampling over PTM-neighbour bonds below 4°, the merge threshold picked
automatically at the knee of the dendrogram, and OVITO's orphan adoption. It
ignores `--threshold`, `--core-angle` and `--min-bonds`, prints
`segment_merge_threshold` (a log distance, OVITO's `auto_merge_threshold`), and
by default runs PTM in OVITO's neighbour order (`--ptm-order voronoi`). On the
HEA polycrystal it matches OVITO at an adjusted Rand index of 0.9999 (the
default mode: 0.946). It is serial and takes about 3× the memory of the default
mode (4.1 GB peak on the 5.8 M-atom film, against 1.2 GB). Use it when the
result must agree with OVITO; the default merges on mean orientations, which
stops gradual chains from bridging boundaries.

**Tracking across frames.** In a time series (`rmda defects --segment --out`)
each grain gets a *stable* id: a grain matches its predecessor when they share
more than half of each other's atoms (by atom id); otherwise it is recorded as
`new`, `split` or `merged` in `grain_events.csv` (§3.7). Growth by more than a
factor two between frames reads as a new grain by construction.

In the default mode the merge threshold is explicit (`--threshold`); OVITO's
automatic (dendrogram) threshold is `--mode auto` above. Not implemented: a
centroid-based fallback when atom ids are renumbered between frames. See design.md, *Grain segmentation and tracking*.

Tests: `crates/rmda-core/tests/grain_seg_validation.rs` (single crystals near the
fundamental-zone boundary, bicrystals, a Voronoi polycrystal, a stacking fault, a
Σ3 twin, thread invariance), `crates/rmda-cli/tests/poly3.rs` (a three-lattice
FCC/BCC/HCP polycrystal over three frames).

### 2.9 Per-grain Wigner–Seitz census — `analysis/defects/`

The defect count `rmda defects` reports.

**Reference.** For each grain (from regions, an input column, a segmentation, or
the whole frame as one grain):

1. The majority crystalline PTM structure, the symmetry-aligned mean orientation
   and the mean PTM first-neighbour distance seed a reference.
2. A full affine lattice `x = o + T·(n + b_k)` — rotation, strain and shear, nine
   components — is fitted by least squares to the grain's first-shell neighbour
   vectors.
3. On a periodic axis the grain spans, `T` is snapped so the box vector is a
   lattice translation (the `snapped_x/y/z` columns). On the other axes the
   grain is unwrapped and cut across its largest gap, so a grain that straddles
   a periodic face is fitted in one piece. A grain whose lattice continues
   across the periodic boundary but does not close on a lattice translation —
   it meets its own image across a fault, or holds a dislocation — cannot be
   described by one lattice and is reported `fit_failed` with that reason.

With `--reference patch` (or `auto`, see below) the grain is tiled into cubic
patches of `--patch-size` (default 12 first-neighbour distances), each with its
own affine fit, seeded from its neighbour so all patches share one lattice index
space. That is what a bent or strain-graded grain needs; `phase_agreement` is
the lowest fraction of atoms on which neighbouring patches agree about site
labels.

**Occupancy.** Pure Wigner–Seitz: every atom of the grain is assigned to its
nearest site, with no tolerance and no exclusion of PTM-`OTHER` atoms, so a
split dumbbell is an interstitial like any other. A site with no atom is a
vacancy; a site with *n* > 1 atoms contributes *n* − 1 interstitials. The counted
domain is the occupied sites plus the empty sites enclosed by them, so a vacancy
cluster's interior is counted and vacuum beyond a free surface is not. An empty
site joins when most of its first neighbours are in the domain; a compact void,
none of whose sites qualifies, joins whole when its empty sites are enclosed:
they do not reach the edge of the grain's sites, do not run round the periodic
box (a vacuum gap through the box is two surfaces), and do not touch another
grain. Sites that another grain's atoms occupy are excluded.

**Two classes, each conserving atoms.** An atom within `--boundary-width` (default
1.5 first-neighbour distances) of an atom of another grain, or of an atom in no
grain, is in the *boundary* class; so is an empty site with such an atom within
that width of the site itself. Everything else is *interior*. For each class
separately, and for every grain, `vacancies − interstitials = sites − atoms`;
`balance_residual` reports the sum of the two residuals and is 0.

> **Read the interior class as the defect count, and the boundary class as
> bookkeeping.** On a boundary between grains of *different* lattices the zone
> holds sites of a grain's lattice that no atom fills, so the boundary class
> carries tens of "vacancies" per grain where an explicit-reference
> Wigner–Seitz analysis finds the one that was planted (design.md, *Defect
> subsystem*). The same happens on an artificial cut: two `--region`s through
> one perfect crystal give a boundary class with vacancies and a clean interior.
> The class exists so that the totals conserve, not as a boundary defect
> count.

**Quality gates.** Every grain's fit metrics are written whether or not the
grain passes, and a grain is counted only when all of them pass:

| status | meaning | counts |
|---|---|---|
| `measured` | fitted, every gate passed | reported |
| `too_small` | fewer than `--min-atoms` atoms (default 100) | `-1` |
| `not_crystalline` | the majority lattice structure (FCC, HCP, BCC) covers less than 30 % of the grain's atoms | `-1` |
| `fit_failed` | the fit could not be made; the reason is printed on stderr (`grain N (stable N): fit_failed (...)`) — typically a lattice that wraps the box but closes off a lattice translation | `-1` |
| `gate_pair_fraction` | the fitted lattice matches fewer than 80 % of the neighbour pairs | `-1` |
| `gate_rms_displacement` | RMS atom-to-site distance above 0.15 of the fitted bond | `-1` |
| `gate_offlattice` | more than 2 % of atoms more than 0.35 bonds from their site | `-1` |
| `gate_phase_agreement` | (patch reference) neighbouring patches agree on fewer than 95 % of atoms | `-1` |
| `gate_face_agreement` | (patch reference) two adjacent patches, whichever was placed from which, agree on fewer than 75 % of the atoms by their shared face: a dislocation loop wider than a patch, which would otherwise be miscounted | `-1` |

Under `--reference local` the `reference` column reads `local`; the patch columns are empty and the `local_*` columns carry the census's own figures (the per-grain table, §3.7).

A gated grain is **refused, not miscounted.** A grain that contains a
dislocation, a stacking fault crossing it (segmented with
`--seg-stacking-faults merge`), or a large non-affine distortion is not
described by one lattice; it lands in `gate_rms_displacement`,
`gate_offlattice` or `fit_failed` instead of reporting hundreds of spurious
defects. Totals (`defects_vacancies` etc.) are over measured grains only, so a
frame where nothing was measured reads 0 with `defects_grains_measured: 0` —
read that scalar before any count.

`--reference auto` (the default) uses the global affine lattice and falls back to
patches when the global fit fails the RMS-displacement or off-lattice gate on a
grain at least three patches across, and to the local census below when neither
counts the grain (stderr: `grain N: ... under global/patch; counting it
locally`). `global` never uses patches, `patch` always does, `local` always
counts locally.

**Local census (`--reference local`).** For grains no single lattice describes:
grains that bend, hold a dislocation network, or meet their own periodic image
across a relaxed seam, as after heavy cascade damage. Defects are local, so no
lattice for the whole grain is needed:

- A lattice atom (PTM structure the grain's) is on its own site.
- Each atom off the lattice gets a site from a lattice (`T` and origin) fitted
  to the lattice atoms among its 40 nearest neighbours. Atoms whose sites
  coincide share one: a dumbbell is one interstitial.
- Empty sites are looked for in the first two shells of those atoms. They need
  no atom within 0.7 bonds and join under the same domain rule as the census.
- Clusters of off-lattice atoms (at least 20 bulk-coordinated ones, up to
  30 bonds across) are counted whole. All atoms inside, including the
  perfect-looking extra plane inside a dislocation loop, are counted against
  one lattice fitted to the coherent shell around the cluster. Overlapping
  clusters merge when their union can be counted the same way; otherwise the
  later one is a *partial region*, counting only the sites and atoms no other
  region counts.
- `--local-merge union|partial` chooses how an overlapping cluster is counted.
  `union` (default, most accurate) tries the union first; `partial` makes it a
  partial region at once, in parallel. On a 128M-atom W frame on 80 threads: 49 min with `union` and 7.4 min with `partial`, totals within
  0.2 % of each other (V 582 331 / 581 117, I 66 541 / 66 667); the thin film
  (12 threads) 151 s and 48 s, substrate I 2040 and 2004 against global's 2052.
  `union` is the more exact where loops meet; `partial` is the choice when run
  time decides (a long series of 100M-atom frames). Each mode's results are
  identical at any thread count.

Counts conserve per class exactly as above. The stderr line `local census over
N atoms off the lattice (U unresolved); R clusters counted as regions (A atoms),
E defect atoms in extended defects` says how much went each way. *Extended
defects* are clusters too large or too incoherent to be one region: dislocation
lines and networks. Their interstitial content (the extra planes) cannot be
assigned locally, so only their rim atoms count.

| check | result |
|---|---|
| single-crystal Fe, 300 K, 50 vacancies / 50 dumbbells | 50 / 50 exactly |
| relaxed 4-grain Fe at 300 K with 100 vacancies | 96 found, 83 as interior (auto: 71 and 60), 3 interior not planted; the grain with a dislocation is counted |
| bent free-standing W block (12°), 4 vacancies | 4; the global reference refuses the grain |
| W thin film after 999 cascades, substrate / film, vs the global / patch reference | vacancies 2408 / 12 965 vs 2388 / 12 963; interstitials 2005 / 13 263 vs 2052 / 13 277 |
| 884 751-atom cut of a 128M-atom W frame, `--segment`, grain 0, vs global | 4283 V / 352 I vs 4286 / 382 |

**Defect clusters.** Vacancies and interstitials of measured grains are grouped
by union-find under `--cluster-cutoff` (default: each grain's own, halfway
between its first and second neighbour shells times its fitted bond:
`(1 + √2)/2` bonds for FCC/HCP/dHCP, `(1 + 2/√3)/2` for BCC; two defects of
different grains join within the smaller of their grains' cutoffs). A cluster
may cross a grain boundary (`multi_grain = 1`). The `cluster_cutoff` column
reports the largest measured grain's; `--cluster-cutoff` sets one distance for
every grain.

Tests: `crates/rmda-core/tests/defects_synthetic.rs` (thermal BCC with planted vacancies
and dumbbells, a sheared and tetragonal slab, a rotated free cluster, a
bicrystal, HCP, a bent crystal and a graded lattice parameter under patches,
patch seams, thread invariance), `crates/rmda-cli/tests/defects_cmd.rs`
(regions, segmentation, patch vs global, series mode with resume and refusal,
a stacking fault gated under `merge` and counted under `separate`),
`crates/rmda-cli/tests/poly3.rs` (interior counts equal the planted ones in every
grain of every frame, and agree with OVITO's explicit-reference Wigner–Seitz
analysis). Design: design.md, *Defect subsystem (Wigner–Seitz family)*.

---

## 3. Command-line usage

Global options, accepted before or after the subcommand: `--threads N`
(0 = `SLURM_CPUS_PER_TASK` inside a SLURM job, else all cores), `--input FILE`, `--output FILE`, `--profile`, `--memreport`,
`--xyz-origin native|zero`, `--skip-columns LIST`, `--atom-style STYLE`,
`--keep-general` (§3.15), `--cell A B C`, `--pbc`,
`--mem-limit SIZE`, and the frame window `--frame N` / `--start N` / `--end N` /
`--stride N` introduced in [Input formats](#input-formats).

`--mem-limit SIZE` (bytes, or `500M`, `64G`, `1T`) stops a run with exit code 4
when the memory it allocated exceeds SIZE, naming the stages that were running.
It counts anonymous memory only: the mapped input file is cache the kernel can
drop. Set it a little under the job's memory, so a run that needs more says where
instead of being killed without a word; a series keeps its finished frames and
resumes, and `rmda run` still writes its report.

`--output` writes **extended XYZ with the per-atom properties attached**, which is
how you get results into OVITO, VMD, or a script — whatever the extension of the
name you give it. Without it, results appear only as the scalars and histograms on
stdout. Progress lines (`read 1 frame(s), …`, banners, warnings, per-grain detail)
go to stderr, so `> results.txt` keeps only the results.

`--threads N` sets one process-wide pool, which the readers and writers use too;
results are bit-identical for every thread count. `--profile` prints, on exit,
the wall time and peak heap of every stage (`io.read_frames`, `ptm`, `segment`,
`defects.grain`, …) to stderr; `--memreport` prints only the peak-RSS line:

```console
$ rmda --memreport census cu-defects.xyz
read 1 frame(s), 2048 atoms in frame 0
mean_cn: 12.015625
total_pairs: 12304
memreport: peak RSS 5.53 MB
```

`--cell A B C` states the box explicitly, in ångström. A file with a `Lattice=`
line carries its own box; a file without one is read as the isolated system it
says it is, in the box its own atoms occupy — see
[Input formats](#input-formats). `--cell` is how a periodic box is stated for a
file that has none: the edges are taken as given, the origin is the one the file
already had, so stating a box does not move the atoms, and the box is periodic
unless `--pbc false` says otherwise.

```console
$ rmda --cell 100 100 100 info poly-small.xyz
read 1 frame(s), 67157 atoms in frame 0
frame 0: t=0 natoms=67157 ntypes=1 cell=orthorhombic pbc=[true, true, true] volume=1000000.000 Å^3
  lengths: 100.0000 100.0000 100.0000
```

`--pbc` on its own overrides the periodicity a file carries, so a file with a box
can be read as the free system it may also be.

**Exit codes**, for batch scripts (the message on stderr says why):

| code | meaning |
|---|---|
| 0 | done |
| 1 | the analysis or writing an output failed |
| 2 | invalid request: an unknown option, a bad value, an invalid `rmda run` config |
| 3 | unusable input: a missing or unreadable file, a format not recognised, a parse error, or a frame that fails the input checks (atom counts that disagree with the header, a non-finite coordinate or box, a degenerate box) |
| 4 | `--mem-limit` exceeded (the message names the stages running) |
| 5 | `rmda defects` on one frame: results written, but no grain was measured (every grain refused by a gate, too small, or not crystalline); a series records this per frame in its tables and exits 0 |

A panic (101) is a bug; please report it with the input.

### 3.1 `rmda info` — what's in this file

```console
$ rmda info hea-reference.data
read 1 frame(s), 256000 atoms in frame 0
frame 0: t=0 natoms=256000 ntypes=4 cell=orthorhombic pbc=[true, true, true] volume=2805449.442 Å^3
  lengths: 141.0374 141.0374 141.0374
```

One `frame` line per frame in the window, and a `types:` line when the file
carries type labels (a data file's `Atom Type Labels`, a dump's `element` column,
XYZ species). The whole report goes to stderr.

### 3.2 `rmda census` — coordination

```console
$ rmda census perfect-fcc.extxyz
read 1 frame(s), 864 atoms in frame 0
mean_cn: 12
total_pairs: 5184
```

(mean coordination is exactly 12 on a perfect FCC block; a relaxed alloy reads higher at the same cutoff, e.g. 14.25 on the 256k HEA fixture.)

### 3.3 `rmda structure` — CNA

```console
$ rmda structure tests/data/fcc_ortho.dump
read 1 frame(s), 32 atoms in frame 0
cna_other: 0
cna_fcc: 32
cna_hcp: 0
cna_bcc: 0
cna_ico: 0
```

| Flag | Meaning |
|---|---|
| `--mode adaptive\|interval\|fixed\|bond\|diamond\|shell` | CNA variant (§2.3); `adaptive` is the default |
| `--cutoff <R>` | radius in Å, `fixed`, `bond` and `shell` modes; default 3.5 |
| `--low <R>` | inner radius in Å, `shell` mode; default 0 |
| `--full` | also emit the per-signature counters: `cna_n421_total` … `cna_n666_total` on stdout, and per-atom `cna_n421` … `cna_n666` columns on `-o` |

```console
$ rmda structure --mode fixed --cutoff 3.0 --full -o cna.xyz tests/data/fcc_ortho.dump
```

`-o` writes `StructureType` (plus the five counters under `--full`). CNA's
structure codes are CNA's own, not PTM's, which is why `rmda analyze` offers only
PTM (§3.13); `--mode diamond` writes identify-diamond's codes 0–6 and prints
`cna_cubic_diamond`, `cna_cubic_diamond_first`, … instead of `cna_fcc` ….

### 3.4 `rmda ptm` — template matching

```console
$ rmda ptm tests/data/fcc_ortho.dump
read 1 frame(s), 32 atoms in frame 0
ptm: rmsd_cutoff=0.1 types=fcc,hcp,bcc
ptm_other: 0
ptm_fcc: 32
ptm_hcp: 0
ptm_bcc: 0
ptm_ico: 0
ptm_sc: 0
ptm_matched: 32/32 (100.00%) mean_rmsd=0.000000 max_rmsd=0.000000
ptm_lattice_constant: mean=3.615000 min=3.615000 max=3.615000 (n=32)
ptm_interatomic_distance: mean=2.556191 min=2.556191 max=2.556191 (n=32)
```

| Flag | Meaning |
|---|---|
| `--rmsd-cutoff <R>` | in units of mean neighbour length; default 0.1, OVITO's |
| `--types <list>` | templates to test, from `fcc,hcp,bcc,ico,sc,dcub,dhex,graphene`; default `fcc,hcp,bcc` (OVITO's) |
| `--ptm-order distance\|voronoi` | neighbour order; default `distance`, `voronoi` is OVITO's (§2.4) |
| `--outputs <list>` | opt-in columns from `ordering,F,template,correspondences,residual,polar` (or `all`); none by default (§2.4) |

`ptm_dcub`, `ptm_dhex` and `ptm_graphene` are printed only when tested;
`ptm_ordering_<none|pure|l10|l12_a|l12_b|b2|zincblende_wurtzite|boron_nitride>`
counts only with `--outputs ordering`.

`-o` writes `StructureType`, `StructureRmsd`, `StructureScale`,
`StructureInteratomicDistance`, `StructureLatticeConstant` and
`StructureOrientation` (the `w x y z` quaternion), plus `StructureOrdering`,
`StructureDeformationGradient`, `StructureTemplate`, `Correspondences`,
`StructureDeformationResidual`, `StructurePolarU` and `StructurePolarP` when selected
with `--outputs`.

### 3.5 `rmda grain-boundaries` — which atoms are on a seam

A Wigner–Seitz vacancy count is only meaningful where the reference lattice is
locally right, and across a grain boundary it is locally right for *neither* grain.
Count those atoms anyway and the defect density becomes a function of grain size: a
finely grained sample then reads as more defective than one that is genuinely damaged.
This pass marks the seam so the per-grain counts can leave it out.

```console
$ rmda grain-boundaries bicrystal.xyz
read 1 frame(s), 6060 atoms in frame 0
grain_boundaries: w_db=4 unassigned_is_boundary=false
gb_atoms_boundary: 449
gb_atoms_interior: 5611
gb_fraction_boundary: 0.0740924092409241
gb_atoms_assigned: 6060
gb_atoms_unassigned: 0
gb_grains: 2
gb_slab_thickness: 2.6131407115010115
gb_w_db: 4
grain_boundary_histogram:
  boundary_atoms=212 grains=1
  boundary_atoms=237 grains=1
```

(`bicrystal.xyz` here is two FCC Cu blocks tilted ±8° about z, periodic along z
only, 6060 atoms.) The pass runs PTM and the bond-graph labelling of §2.5 first, with
the same `--min-grain-size` (default 100) and orphan adoption (on by default), so
every atom has a grain id. With `--no-adopt-orphans` the 1206 atoms PTM cannot
classify — most of them the free surfaces and the layer at the seam — stay at
`-1`, and the boundary class grows accordingly:

```console
$ rmda grain-boundaries bicrystal.xyz --no-adopt-orphans 2>/dev/null | grep -E 'atoms_boundary|unassigned'
gb_atoms_boundary: 1854
gb_atoms_unassigned: 1206
```

An atom is on the boundary when the grain ids among **itself and its neighbours**
within `--w-db` hold two or more distinct values. Two details in that sentence are
load-bearing:

- **The atom's own id is in the set.** Without it the test asks for two *other* grains
  inside one neighbour list, which on a two-block sample finds nothing at all.
- **An unassigned neighbour counts as its own id**, not as a member of yours. This is
  the one that decides whether the pass works.

**Why the unassigned clause is not a detail.** The disordered layer *at* a grain
boundary is precisely what PTM cannot classify, so without adoption it comes back with
no grain id — and it would then hide the seam behind itself. On an ideal sharp
1000-atom FCC bicrystal with 523 unassigned atoms, a rule that skips unassigned
neighbours reports **0** boundary atoms at `--w-db 3` and 20 at `4`, where the true seam
holds hundreds. That is the worst failure mode available: it finds nothing,
confidently. Counting an unassigned neighbour as a distinct id takes the same fixture to
380 atoms at `--w-db 4`, which is what the geometry says is there.

For the purpose this pass serves, *what* the odd neighbour is does not matter. A
neighbour outside my grain means the local reference lattice is not reliably my grain's,
and that is the question Wigner–Seitz needs answered.

**`gb_slab_thickness` is how you tell whether `--w-db` was the binding constraint.** It
is the widest distance from a flagged atom to the nearest unflagged one — the shallowest
exclusion that would give the same answer. If it sits at `--w-db`, the width decided the
result and a narrower one would report fewer atoms; if it is well below, the boundary is
genuinely that thin and the width is not doing the work.

**`--unassigned-is-boundary` is a separate question.** By default the unassigned atoms
are not themselves in the class — they are not part of any grain's interior to begin
with — but the crystalline atoms around them still are. The flag puts the unassigned
atoms in as well, which for a defect exclusion is usually what you want either way.

**A perfect single crystal reports zero**, and so does a relaxed one that PTM calls a
single grain. If you expected a seam and got `gb_atoms_boundary: 0`, read
`gb_atoms_unassigned` before anything else: a frame with a large unassigned count and no
boundary usually means the boundary was never resolved into grains in the first place.

**The seam counts are per grain, so they reach the CSV table, not the atom table.**

| Flag | Meaning |
|---|---|
| `--w-db` | exclusion half-width in Å; default 4 |
| `--unassigned-is-boundary` | put the unassigned atoms in the class too |
| `--min-grain-size <n>` | crystallites below this many atoms dissolve into their neighbours (OVITO's `min_grain_size`); default 100 |
| `--no-adopt-orphans` | leave the atoms of dissolved crystallites unassigned instead of giving them to the nearest grain |
| `--grain-table FILE` | write the per-grain table, with a `boundary_atoms` column |

`-o` writes the per-atom `Grain` and `GrainBoundary` (1 on the seam) columns
along with PTM's.

`boundary_atoms` appears in that table **only when this pass ran**. A column that is
absent says the quantity was not measured; a column present and reading zero would say the
grain has no boundary atoms, which is a different and wrong claim.

### 3.6 `rmda segment` — grain segmentation

Runs PTM (`fcc,hcp,bcc`), then the crystallite-and-merge segmentation of §2.8.
This is the segmentation `rmda defects --segment` runs, and the one to use for a
polycrystal.

```console
$ rmda segment bicrystal.xyz --table seg.csv -o seg.xyz
read 1 frame(s), 6060 atoms in frame 0
segment_grains: 2
segment_crystallites: 2
segment_atoms_assigned: 6060
segment_atoms_unassigned: 0
segment_atoms_adopted: 1206
grain 0: size=3078 structure=1 crystallites=1 q=[0.9976, -0.0, 0.0, 0.0698]
grain 1: size=2982 structure=1 crystallites=1 q=[0.9976, 0.0, -0.0, -0.0698]
segment: grain table seg.csv and pair table seg.pairs.csv
wrote 2 properties to seg.xyz
  Grain, StructureType
```

The five `segment_*` lines are stdout; the per-grain lines (the first 20 grains;
the rest are in the table) and the file notes are stderr. `segment_atoms_adopted`
counts the unassigned atoms — non-crystalline ones included — handed to a grain
by orphan adoption, so on a real polycrystal it is mostly the boundary layer.

| Flag | Meaning |
|---|---|
| `--threshold <deg>` | crystallites whose *mean* orientations differ by less than this merge; default 5 |
| `--core-angle <deg>` | bonds below this disorientation form crystallites; default 2 |
| `--min-bonds <n>` | crystallites merge only when they share at least this many bonds; default 4 |
| `--min-size <n>` | grains below this many atoms dissolve into their neighbours; default 100 |
| `--stacking-faults merge\|separate` | HCP fault and twin planes join the FCC grain around them (`merge`, default, as OVITO), or FCC and HCP never share a grain (`separate`) |
| `--no-adopt` | leave unassigned atoms at `-1` instead of adopting them |
| `--mode crystallite\|auto` | `crystallite` (default) as above; `auto` is OVITO's algorithm with its automatic threshold (§2.8) |
| `--ptm-order distance\|voronoi` | PTM matches the nearest neighbours (`distance`) or, as OVITO, the neighbours with the largest Voronoi faces (`voronoi`); default `voronoi` with `--mode auto`, else `distance` |
| `--rmsd-cutoff <r>` | PTM RMSD cutoff; default 0.1 |
| `--table <file>` | write the per-grain table here, and the grain-pair table next to it |

**Tables.** `--table seg.csv` writes `seg.csv`, one row per grain,

```csv
timestep,grain,size,structure,crystallites,qw,qx,qy,qz
0,0,3078,1,1,0.99756404,-0.00000159,0.00000090,0.06975663
0,1,2982,1,1,0.99756309,0.00001002,-0.00003310,-0.06977024
```

and `seg.pairs.csv` (the extension replaced by `.pairs.csv`), one row per pair
of grains that share bonds, with the disorientation between their mean
orientations — here the 16° the two blocks were built with:

```csv
timestep,grain_a,grain_b,bonds,disorientation_deg
0,0,1,541,16.0016
```

`structure` is the PTM code (1 FCC, 2 HCP, 3 BCC); `crystallites` is how many
crystallites were merged into the grain; `(qw,qx,qy,qz)` is the symmetry-aligned
mean orientation of its crystalline atoms. HCP orientations are in PTM's template
frame, not OVITO's conventional one.

**`-o`** writes the per-atom `Grain` (`-1` = no grain) and `StructureType`
columns. That file can be fed back to `rmda defects --grain-column Grain`, or to
any other tool, so a segmentation is computed once and reused.

What to tune: `--threshold` is the boundary criterion (the default mode has no
automatic threshold, `--mode auto` has OVITO's; on the HEA polycrystal fixture OVITO picks about
13°, and `--min-bonds 1` with `--threshold` 8–13 agrees with OVITO's labels
better than the defaults do — design.md, *Grain segmentation and tracking*).
`--min-bonds` suppresses noise bridges through a handful of bonds. Use
`--stacking-faults separate` when the grains will be counted with
`rmda defects` and faults cross them (§3.7).

**Double HCP (`--dhcp`, on `segment` and `defects`).** dHCP (ABAC stacking:
La, Pr, Nd, Am) has no PTM template; every one of its atoms is locally FCC or
HCP, and in alternating layers (the A layers FCC, the B and C layers HCP). By
default stacking-fault merging joins those layers into one grain, as OVITO's
segmentation does, but an FCC or HCP one, against whose lattice the dHCP's
defects cannot be counted. With `--dhcp`, close-packed atoms whose
in-plane neighbours share their label and whose neighbours above and below
carry the other one, in a run of at least seven such layers, are relabelled
structure `32` (dHCP) with an orientation in the HCP frame, so the crystal is
one grain, counted against its own ABAC lattice (c/a free). The run length
keeps a twin plane or a short alternating run in heavily faulted FCC from
being called dHCP; on FCC, HCP, twinned and faulted stacks (up to 30 % of
layers faulted) and on the HEA polycrystal fixture it relabels no atom. It is
off by default all the same, because a long enough alternating run in a
faulted FCC crystal *is* dHCP stacking, and would become a grain of its own.
A dHCP crystal's stacking faults (FCC or HCP runs of up to six layers between
dHCP layers) split it into several grains; `rmda segment --dhcp
--dhcp-merge-faults` (Python `dhcp_merge_faults=True`) puts them into the dHCP
grain, while a thicker FCC region next to dHCP keeps its own. It is not
available in `rmda defects`, which needs the split grains to count defects
correctly (a merged faulted grain gives dozens of false interstitials).
`rmda ptm` always writes PTM's own labels. stderr reports how many atoms were
relabelled (`dhcp: N close-packed atoms in ABAC stacking`). In Python,
`rmda.segment(f, dhcp=True)` and `rmda.defects(f, ..., dhcp=True)`. Cost: a frame
without both FCC and HCP atoms (BCC, for instance) is skipped at once; on the
42k-atom HEA polycrystal the pass takes 0.13 s, about 12 % of PTM and 3 % of
a `defects --segment` run.

### 3.7 `rmda defects` — per-grain vacancies and interstitials

The per-grain Wigner–Seitz census of §2.9: a full affine lattice fitted to each
grain, pure nearest-site occupancy, interior and boundary classes that each
conserve atoms, and quality gates that refuse a grain one lattice cannot
describe. For one file it prints a summary; for many it builds a time series
(below).

```console
$ rmda defects cu-defects.xyz --table grains.csv --defects def/
read 1 frame(s), 2048 atoms in frame 0
defects_grains: 1
defects_grains_measured: 1
defects_sites: 2048
defects_atoms: 2048
defects_vacancies: 1
defects_interstitials: 1
defects_boundary_sites: 0
defects_boundary_atoms: 0
defects_boundary_vacancies: 0
defects_boundary_interstitials: 0
defects_net_vacancies: 0
defects_boundary_net_vacancies: 0
defects_balance_residual: 0
defects_clusters: 2
defects_largest_cluster: 1
grain 0 (stable 0): measured ref=global patches=1/1 phase=1.000 atoms=2048 structure=1 bond=2.5562 pairs=1.000 rms_disp=0.065 offlattice=0.0005 snapped=[true, true, true] V=1 I=1 gb_V=0 gb_I=0 net_V=0
defects: per-grain table written to grains.csv
defects: vacancies.xyz and interstitials.xyz written to def/
```

(`cu-defects.xyz`: a periodic 8×8×8-cell FCC Cu block, 0.03 Å thermal noise,
one atom removed and one put on an octahedral site.) With no `--region`,
`--grain-column` or `--segment` the whole frame is one grain, which is the right
call for a single crystal.

**Which reference for which frame.** `--reference auto` (the default) is right
for most frames; the cases where something else counts more, and what each
cannot see:

| frame | run | notes |
|---|---|---|
| single crystal in a periodic box that kept its atoms (cascade simulations) | `--reference periodic` | counts against the crystal's one lattice, including defects stored as whole planes (loops grown across the box, collapsed planes), which no local reference sees; V − I is then exactly the sites the box gained or lost. A frame whose lattice is not commensurate with the box, or whose atoms no lattice through it holds, is counted as `auto` counts it, with the reason on stderr |
| polycrystal | `--segment` (`auto`) | each grain global when one lattice fits it, patch when a few do, local otherwise |
| thin film on a substrate | `--region 'z <= Z' --region 'z > Z'` (`auto`) | substrate and film each against their own lattice |
| comparison with a reference frame (OVITO's Wigner–Seitz) | `--ws-reference REF` | pure nearest-site occupancy against REF's sites |

Known limits, all of the local reference (counted locally: `ref=local` on the
grain's line, `reference` in `grains.csv`):

- defects stored as whole planes across a periodic box are not seen (use
  `periodic` for a single crystal; `auto` warns when a periodic single grain
  counted locally has V − I far from zero);
- a loop whose shell another loop's rim crosses (loops of 40+ Å packed a few
  Å apart) is counted atom by atom and loses part of its plane: on a
  synthetic block of such loops, 5–10 % of their content;
- `--local-merge partial` is 3–7× faster than the default `union` (a
  128M-atom frame: 7.4 against 49 min) and agrees with it within 0.2 % there, but
  is less exact where loops meet (thin-film substrate I −2.3 % against −0.6 %).

Where a validated reference exists, local agrees with it closely: on the thin
film, counted locally, the substrate is 2387 V / 2040 I against global's
2388 / 2052 and the whole film 15 347 / 15 334 against `auto`'s 15 351 / 15 329.

Global and patch are pure Wigner–Seitz, as OVITO's: inside a large loop a few
atoms sit equally near two sites and come out as Frenkel pairs (net unchanged);
`--resolve-ambiguous` gives them the site their neighbours agree on.

**Where the grains come from** — one of:

- `--region EXPR`, repeatable: each expression defines one grain, and an atom
  takes the first region it matches; atoms in none are in no grain. An
  expression is clauses `FIELD OP NUMBER` joined by `&&`, with `FIELD` one of
  `x y z` (Å), `sx sy sz` (fractional), `type`, `id`, and `OP` one of
  `< <= > >= == !=`: `--region 'z <= 500' --region 'z > 500 && type != 3'`.
  This is the thin-film case: substrate and film counted against their own
  lattices.
- `--grain-column NAME`: labels from an input column, e.g. the `Grain` column
  `rmda segment -o` wrote. Negative labels mean no grain. Cannot be combined
  with `--region`.
- neither: the whole frame is one grain.

`--segment` then splits each of those (each region, or the whole frame) by the
grain segmentation (§2.8), with `--seg-threshold` (default 5°), `--seg-min-size`
(default 100) and `--seg-stacking-faults merge|separate` (default `merge`);
`--seg-mode auto` uses OVITO's algorithm instead (§2.8), and `--ptm-order
distance|voronoi` picks PTM's neighbour order (default `voronoi` with
`--seg-mode auto`, else `distance`):

```console
$ rmda defects bicrystal.xyz --segment
...
defects: 2 grain(s) after splitting 1 region(s) by segmentation
defects_grains: 2
defects_grains_measured: 2
defects_sites: 5641
defects_atoms: 5641
defects_vacancies: 0
defects_interstitials: 0
defects_boundary_sites: 419
defects_boundary_atoms: 419
...
```

(`...` stands for the `segment_*` summary of §3.6 and the per-grain lines.)

**What the scalars mean.** `defects_sites/atoms/vacancies/interstitials` are the
interior class, summed over *measured* grains; `defects_boundary_*` the
boundary class (§2.9). `defects_net_vacancies` (and
`defects_boundary_net_vacancies`) is `V − I`, equal to `sites − atoms`: the sites
the class has beyond its atoms. In a closed crystal damaged by Frenkel pairs it
is what the counts do not hold: interstitials in dislocation loops and the
dislocation network whose extra planes are perfect crystal locally and that the
census could not count, *less* vacancies in collapsed vacancy loops it could not
count, plus atoms moved to a surface or boundary. It is therefore a net figure,
not a count of either kind: a positive value means at least that many hidden
interstitials, not exactly that many. `defects_balance_residual` is
`(V − I) − (sites − atoms)` summed over both classes and is always 0 — a nonzero
value would be a bug. `defects_clusters` and `defects_largest_cluster` describe
the defect clusters (§2.9). Read `defects_grains_measured` first: totals over
zero measured grains are zeros, not a clean sample.

**The per-grain line** (stderr) and **the per-grain table** (`--table FILE`, CSV,
one row per grain) carry the same fields:

| column | meaning |
|---|---|
| `timestep`, `grain`, `stable_grain` | frame timestep; grain label in this frame; id tracked across a series (equal to `grain` outside one) |
| `status` | `measured`, `too_small`, `not_crystalline`, `fit_failed` or a `gate_*` name — see §2.9 for each |
| `reference`, `n_patches`, `n_patches_fitted`, `phase_agreement` | `global`, `patch` or `local`; patch count (fitted rather than inherited from a neighbour); the lowest patch-to-patch agreement (1 for global). Empty for `local` |
| `n_atoms`, `n_crystalline`, `structure` | grain atoms; atoms of its majority structure; that structure's PTM code (1 FCC, 2 HCP, 3 BCC; 0 if none was fitted) |
| `bond` | fitted first-neighbour distance, Å |
| `pair_fraction`, `pair_rms` | fraction of neighbour pairs the fitted lattice matches, and their RMS residual |
| `rms_displacement`, `offlattice_fraction` | RMS atom-to-site distance (Å), and the fraction of atoms more than 0.35 bonds from their site |
| `snapped_x/y/z` | 1 where the lattice was made commensurate with a periodic box vector the grain spans |
| `domain_converged` | 1 when the enclosed-site filling converged (a cavity wider than the iteration cap leaves 0) |
| `sites`, `atoms`, `vacancies`, `interstitials` | interior class |
| `gb_sites`, `gb_atoms`, `gb_vacancies`, `gb_interstitials` | boundary class |
| `clusters`, `largest_cluster` | defect clusters of this grain |
| `t_xx` … `t_zz` | the fitted 3×3 lattice matrix `T` of `x = o + T·(n + b_k)`, whose columns are the conventional cell vectors in Å; `t_xy` is row x, column y. For an unstrained cubic grain aligned with the box it is `a` times the identity; an off-diagonal term is a shear or a rotation |
| `net_vacancies`, `gb_net_vacancies` | `V − I` of the interior and boundary classes (the scalars above) |
| `local_chunks`, `local_defect_atoms`, `local_unresolved` | local census only (empty otherwise): chunks whose `T` was refined; atoms off the lattice, placed by a local fit; of those, the ones with too few lattice atoms around them, each kept on a site of its own |
| `face_agreement` | lowest agreement between any two adjacent patches on the atoms by their face (1 for global; empty for local) |
| `local_regions`, `local_region_atoms`, `local_extended_atoms`, `local_fit_rms` | clusters counted whole as regions and the atoms counted that way; defect atoms in extended defects counted atom by atom; RMS residual of the lattice atoms the origins were fitted to, Å |

Every metric is written whatever the status, so a refused grain shows *why*.
The counts of a grain that is not `measured` are `-1` in the table: withheld,
not zero. (The stderr line prints whatever the census produced for a gated
grain; only the table and the totals withhold it.)

**Marker files** (`--defects DIR`): `vacancies.xyz` (species `Va`, columns
`grain`, `boundary` — 1 for the boundary class — and `cluster`) and
`interstitials.xyz` (the interstitial atoms with their own species, columns `id`,
`type`, `grain`, `boundary`, `cluster`). Only measured grains contribute. The
header carries the input cell as nine `Lattice=` numbers plus `Origin=`, `pbc=`
and `Timestep=`, and vacancy positions are wrapped into it, so the files overlay
the input frame in OVITO. Of the atoms crowding a site, the one closest to the
site is the regular atom and the others are the interstitials, so
`interstitials.xyz` has one row per interstitial (unlike `rmda ws`, §3.12).

**Checking the vacancies against PTM** (`--non-lattice-atoms`): also writes
`non_lattice.xyz`, every atom whose PTM structure is not its grain's lattice (PTM
`OTHER`, or FCC/HCP atoms inside a BCC grain), plus atoms in no grain, with columns
`id`, `type`, `StructureType` (OVITO's codes: 0 other, 1 FCC, 2 HCP, 3 BCC) and
`grain`. A real vacancy sits inside a shell of such atoms, so overlaying
`vacancies.xyz` on `non_lattice.xyz` in OVITO shows each one in its shell without
loading the whole frame. `vacancies.xyz` then gains a `non_lattice_shell` column:
how many non-lattice atoms lie within the grain's first shell of the vacancy (BCC
8, FCC/HCP 12 when the shell is complete).

- An interior vacancy with a full or nearly full shell agrees with PTM. One with a
  count near 0 is worth a look.
- Boundary-class vacancies always have high counts, because boundary atoms are
  non-lattice anyway, so the count does not separate real vacancies from boundary
  misfit there.
- On a 4-grain atomsk W polycrystal with 20 random vacancies, all 13 interior
  vacancies found had a shell of 8. The file lists about 23 % of that sample's atoms
  (its unrelaxed boundaries are rough). Large grains have a much smaller boundary
  fraction, but expect a large file at 10⁸ atoms.

**Regions cut through a crystal make a boundary class.** The boundary class is
bookkeeping (§2.9), and a region boundary is a boundary. Cutting the
single crystal above into two regions leaves the interior clean but puts
vacancies in the boundary class, where the two fitted lattices meet:

```console
$ rmda defects cu-defects.xyz --region 'x < 14.46' --region 'x >= 14.46' 2>/dev/null | grep -E 'measured|_vacancies'
defects_grains_measured: 2
defects_vacancies: 0
defects_boundary_vacancies: 36
```

(The planted vacancy sits at x ≈ 0, next to the periodic face, so it is now in the
boundary zone too.) Place region cuts where there is a real interface, or where
the defects you care about are not. On a boundary between different lattices the
boundary class is large for the same reason, and should not be read as a count of
boundary defects.

**A refused grain.** The HEA polycrystal fixture is heavily distorted; none of its
grains can be described by one lattice, and `rmda defects` says so rather than
reporting thousands of vacancies:

```console
$ rmda defects hea-poly.data --segment 2>&1 | grep -E '^grain [0-9] \(|measured:'
defects_grains_measured: 0
grain 0 (stable 0): fit_failed (invalid data: the lattice continues across the periodic boundary along x but closes 0.97 A off a lattice translation (self-contact across a fault, or a dislocation)) ref=global ...
grain 1 (stable 1): fit_failed (... along x but closes 1.38 A off a lattice translation ...) ...
grain 2 (stable 2): fit_failed (... along z but closes 1.24 A off a lattice translation ...) ...
grain 3 (stable 3): gate_rms_displacement ref=global patches=1/1 phase=1.000 atoms=8277 structure=1 bond=2.5701 pairs=0.996 rms_disp=0.492 offlattice=0.0687 ...
```

(lines shortened). 0.97–1.38 Å is about a partial Burgers vector: these grains
contain faults or dislocations. They are too small for patches (`auto` falls back
to patches only on a grain at least three `--patch-size` across). A grain crossed
by a stacking fault, segmented with `--seg-stacking-faults merge`, is refused the
same way (`gate_rms_displacement`); with `separate` its two sides and the fault
layers become three grains and are each counted.

| Flag | Meaning |
|---|---|
| `--region <expr>` | a region defining one grain; repeatable (above) |
| `--grain-column <name>` | grain labels from an input column |
| `--segment` | split the regions (or the frame) into segmented grains |
| `--seg-threshold <deg>`, `--seg-min-size <n>`, `--seg-stacking-faults merge\|separate` | segmentation settings, as `rmda segment`'s `--threshold`, `--min-size`, `--stacking-faults`; defaults 5, 100, `merge` |
| `--seg-mode crystallite\|auto` | segmentation algorithm, as `rmda segment --mode`; default `crystallite`, `auto` is OVITO's (§2.8) |
| `--ptm-order distance\|voronoi` | PTM neighbour order; default `voronoi` with `--segment --seg-mode auto`, else `distance` |
| `--dhcp` | recognise double HCP as its own structure (§3.6) |
| `--reference auto\|global\|patch\|local\|periodic` | reference per grain (§2.9); default `auto`. `periodic`: for a periodic single crystal whose lattice is commensurate with the box (a cascade simulation that kept its sites), Wigner–Seitz against that one lattice with ambiguous atoms resolved, past the displacement gates — the only reference that counts defects stored as whole atomic planes added or removed (interstitial loops grown into strips across the box, collapsed vacancy loops); V − I is then exactly the sites the box gained or lost (0 when no atom left). Where a dislocation spans the box, the lattice is the one that holds the box's atoms. Grains not commensurate with the box, or whose atoms no lattice through the box holds, are counted as `auto` would, with a note on stderr. Under `auto`/`local`, a periodic single grain counted locally whose V − I is far from zero gets a warning pointing here |
| `--patch-size <Å>` | patch edge; default 12 first-neighbour distances |
| `--local-merge union\|partial` | local census: count an overlapping cluster with its union first (`union`, default, most accurate) or as a partial region at once (`partial`, faster; above) |
| `--resolve-ambiguous` | global and patch references: an atom more than 0.3 bonds from its nearest site takes the site its neighbours point to, as the local census does. Off by default: every atom goes to its nearest site (Wigner–Seitz, as OVITO counts it), which reports spurious Frenkel pairs inside large dislocation loops where the crystal sits nearly b/2 off the lattice (13–41 per a/2<111> loop of W, 30 in an HCP test block); on, those are gone (design §16.8a). Python `resolve_ambiguous=True`, `rmda run` `resolve_ambiguous = true` |
| `--ws-reference <file>` | explicit-reference Wigner–Seitz: the sites are this frame's atoms, not a fitted lattice (below); conflicts with `--reference` and `--patch-size` |
| `--ws-affine off\|to-reference\|to-current` | with `--ws-reference`: affine mapping between the reference and current cells; default `off` |
| `--antisites` | also count antisites in multi-species crystals (below); adds columns at the end of the tables, `antisites.csv` (series) and `antisites.xyz` |
| `--boundary-width <Å>` | boundary-zone width; default 1.5 × the frame's mean PTM first-neighbour distance |
| `--min-atoms <n>` | grains with fewer atoms are `too_small`; default 100 |
| `--rmsd-cutoff <r>` | PTM RMSD cutoff for the structure and orientation seed; default 0.1 |
| `--cluster-cutoff <Å>` | defect-cluster cutoff for every grain; default each grain's own, halfway between its first and second neighbour shells |
| `--table <file>` | the per-grain table (single-file mode) |
| `--defects <dir>` | `vacancies.xyz` and `interstitials.xyz` (single-file mode) |
| `--out <dir>` | time-series mode (below) |
| `--shard I/N` | with `--out`: run shard I of N (from 0) of the ordered frames, e.g. one SLURM array task each; `rmda merge` joins them (§3.18) |
| `--defects-xyz` | with `--out`: also write each file's `vacancies.xyz` and `interstitials.xyz` under `defects/`, laid out by `--defects-layout` |
| `--defects-layout dirs\|files\|append` | with `--out`, implies `--defects-xyz`. `dirs` (default): `defects/<index>/vacancies.xyz`. `files`: `defects/vacancies-<index>.xyz`. `append`: one `defects/vacancies.xyz` with every frame in turn, which OVITO reads as a trajectory. Each frame's comment line carries `index=<index>` |
| `--elements E1 E2 …` | element of atom types 1, 2, … for the species column of `interstitials.xyz` and `non_lattice.xyz`. Types not named take the file's own labels, then the element whose standard atomic weight matches the type's mass in a data file's `Masses` section (`183.84` → W, `50.9415` → V), else `X` |
| `--non-lattice-atoms` | also write `non_lattice.xyz` beside the defect files, and a `non_lattice_shell` column in `vacancies.xyz`; needs `--defects`, or `--out` with `--defects-xyz` |
| `--overwrite` | with `--out`: discard the existing tables instead of resuming |

`-o` is refused with an error naming `--table`, `--defects` and `--out`: the per-atom
results are in the table and the marker files.

**Explicit reference: `--ws-reference REF`.** When you have the defect-free
configuration (a relaxed perfect crystal, the first frame of a run), count
against it instead of a fitted lattice. This is OVITO's
`WignerSeitzAnalysisModifier`: every atom of the current frame goes to its
nearest reference atom (its *site*), with no tolerance; a site with no atom is a
vacancy, and every occupant beyond the first is an interstitial (the one nearest
the site stays the occupant). The results go into the usual summary, table,
`vacancies.xyz`/`interstitials.xyz`, clusters and series tables; the table's
`reference` column reads `explicit`, and both defect files gain a `site_id`
column, the reference atom id of the site.

```console
$ rmda defects current.dump --ws-reference reference.dump --ws-affine to-reference --defects def/
ws reference: 8192 atom(s) from reference.dump
read 1 frame(s), 8174 atoms in frame 0
ws reference: 8192 site(s), affine mapping to-reference
defects_grains: 1
defects_grains_measured: 1
defects_sites: 8192
defects_atoms: 8174
defects_vacancies: 21
defects_interstitials: 3
...
defects_balance_residual: 0
defects_clusters: 10
defects_largest_cluster: 15
grain 0 (stable 0): measured ref=explicit ... atoms=8174 structure=3 bond=2.7410 ... rms_disp=0.175 ... V=21 I=3 gb_V=0 gb_I=0 net_V=18
```

(A BCC W block, 16³ cells, with 6 vacancies, a 15-site void and 3 ⟨110⟩
dumbbells, 0.1 Å thermal noise, in a box strained 3.5 % with a 3 % xz shear.)

- **Affine mapping** (`--ws-affine`, OVITO's `affine_mapping`): `off` (the
  default, as in OVITO) takes positions as they are, with periodic images of the
  reference cell; `to-reference` maps current positions into the reference cell
  first (OVITO's `ToReference`); `to-current` maps the sites into the current
  cell instead (OVITO refuses this one; it differs from `to-reference` only in
  which cell's metric decides "nearest"). Use a mapping when the box was
  deformed between the frames: without one the strained example above reports
  hundreds of Frenkel pairs, and `rmda` warns when the cells differ and the
  mapping is off. A mapping cannot remove a deformation that is not a box
  deformation (the thin-film pair's ~2.5 % shear gives 250 065 V / 250 059 I
  against `reference.data`). Vacancy positions are written in the current cell
  when a mapping is on, and are the reference positions with `off`.
- **Periodicity** is the reference cell's; the two frames must have the same
  periodic flags.
- **Grains and classes are decided on the reference, once.** With no
  `--region`, `--grain-column` or `--segment`, every site is one grain and the
  totals are OVITO's. Otherwise those options are evaluated on the *reference*
  frame (its positions, types, ids and columns; `--segment` segments the
  reference), and a site's class is `boundary` within `--boundary-width` of a
  site of another grain or of none. An atom counts in the grain and class of the
  site it was assigned to, so every class conserves (`balance_residual` is 0);
  atoms on sites in no grain are not counted. There is no fit, so there are no
  gates: every grain is `measured`, `--min-atoms` does not apply, and the
  fit columns (`pair_fraction`, `t_*`, …) are NaN. A series needs no grain
  tracking (stable ids are the reference's grains, `grain_events.csv` stays
  empty).
- The reference is read, PTM-analysed (for each grain's structure and bond,
  which set the cluster cutoff) and indexed once per run, also for a series; the
  current frames need no PTM unless `--non-lattice-atoms` is given.
- Checked site for site against OVITO 3.15 on the `poly3` frames against their
  true references (44 V / 32 I per frame, also through a strained triclinic box
  with `to-reference`, and the 4919 V / 4907 I a strained box gives with `off`),
  pinned by `crates/rmda-cli/tests/ws_reference.rs`; `tools/ws_reference_ovito.py`
  repeats the comparison on any pair. Thin film (5.76M atoms): 34 s, 1.2 GB
  peak, on 12 threads.

Python: `rmda.defects(f, ws_reference="ref.dump", ws_affine="to-reference")`
(`ws_reference` may also be a frame).

**Antisites: `--antisites`.** In an ordered alloy (B2 NiAl/FeAl, L1₂ Ni₃Al,
L1₀ CuAu) an antisite is a site of one species holding an atom of another. As
in OVITO's `WignerSeitzAnalysisModifier(per_type_occupancies=True)`, only
**singly occupied** sites count: an empty site is a vacancy and an
over-occupied one an interstitial site, whatever the species (a dumbbell with a
wrong-species atom is one interstitial, no antisite). The vacancy and
interstitial counts are the same with and without the flag. Where a site's
species comes from:

- `--ws-reference`: the type of the reference atom that is the site. If both
  frames name their species (extxyz, data files with labels) in a different
  order, they are matched by name; otherwise type numbers are compared.
- `--reference global` / `patch` (and `auto` while it uses them): per grain,
  each site of the fitted motif (BCC's cube corner and body centre, FCC's four
  cube sites) takes the species most of its singly occupied sites hold. That
  recovers B2, L1₂ and L1₀ exactly. It cannot represent an ordering whose
  chemical cell is larger than the cubic cell (D0₃, L2₁ Heusler, D0₁₉, B1
  rocksalt) or an antiphase boundary inside a grain; those show as a large
  antisite count, with the species found per motif site in the
  `sublattice_types` column. Check it before trusting the count.
- **A random solid solution (an HEA) has no antisites.** With the same
  composition on every motif site, "the species of a site" is meaningless, and
  any atom not of its site's chance majority would count (80 % of a random
  CoCrFeMnNi block). Each grain's chemical order is measured as Cramér's V of
  its (motif site, type) occupancy (0 for a random alloy, 1 for one type per
  site, ~0.8 for B2 with 10 % antisites), and each motif site's host share
  (the share its majority type holds). Antisites are counted when the alloy
  is ordered (V ≥ 0.3) or every motif site has a host species (share ≥ 0.75:
  a dilute solid solution such as W with 5 % Re, whose solutes on host sites
  are antisites); otherwise not, stderr says so (`grain 0: no chemical order
  (Cramér's V 0.003 < 0.3) and no host species (majority share 0.20 <
  0.75)`), and with no grain counted `defects_antisites` is `nan` (Python
  NaN), as are the summary's columns. Under `--reference local` both are
  means over its vote regions. The explicit reference (`--ws-reference`) has its
  sites' types and counts antisites in any alloy.
- `--reference local` (and grains `auto` counts locally): the same majority
  vote, but per region of about ten bonds (plus a margin) instead of per
  grain, each region with its own fitted lattice, since the local census has
  no lattice over the whole grain. An antiphase boundary then spoils only the
  regions it crosses. `sublattice_types` is the sorted list of species (e.g.
  `1 1 1 2` for L1₂), as motif-site labels differ between regions. It agrees
  atom for atom with `--ws-reference` on B2 and L1₂ fixtures up to σ 0.15 Å
  of thermal noise. Defect atoms with no local lattice (unresolved) are never
  antisites.

```console
$ rmda defects b2.cur.dump --ws-reference b2.ref.dump --antisites --defects def/
...
defects_vacancies: 6
defects_interstitials: 7
...
defects_antisites: 23
defects_boundary_antisites: 0
defects_antisites_2_on_1: 12
defects_boundary_antisites_2_on_1: 0
defects_antisites_1_on_2: 11
defects_boundary_antisites_1_on_2: 0
```

The new stdout lines come after the usual ones: totals, then per pair
`defects_antisites_<occupant>_on_<site>` (species names when each type has a
distinct known name, as from `--elements` or the file, else type numbers). The
table and `grains.csv` gain `antisites`, `gb_antisites`, `sublattice_types` at
the end (−1 for an unmeasured grain); `summary.csv` gains `antisites`,
`gb_antisites`; a series writes `antisites.csv` (one row per grain and pair:
`site_type`, `occupant_type`, their species, interior and boundary counts); the
defect files gain `antisites.xyz` (occupant species and position, `id`, `type`,
`site_type`, `site_species`, `site_pos`, `grain`, `boundary`, and `site_id`
with `--ws-reference`, and `cluster`).

Antisites also join the defect clusters, without changing them: each antisite
joins the cluster of the nearest vacancy or interstitial within the cluster
cutoff, and the antisites near none are clustered among themselves into
antisite-only clusters (a swapped pair is one of size 2), numbered after the
vacancy/interstitial clusters. An antisite never joins two vacancy or
interstitial clusters into one, so their ids, members and counts are those of
a run without `--antisites`. Stdout gains `defects_clustered_antisites`
(antisites on vacancy/interstitial clusters), `defects_antisite_clusters` and
`defects_largest_antisite_cluster`; series `clusters.csv` gains an `antisites`
column and the antisite-only clusters as rows after the others (`size`, which
counts vacancies and interstitials, is 0 there). A series run with
`--antisites` is not resumed without it (or the other way round).
Checked site for site against OVITO 3.15 on B2 and L1₂ fixtures
(`tools/antisites_ovito.py`, `crates/rmda-cli/tests/antisites.rs`). Python:
`rmda.defects(f, ..., antisites=True)`.

#### 3.7.1 Time series: `rmda defects FILES... --out DIR`

Several input files need `--out DIR` (one file with `--out` is a series of one).
Each file is read (one frame: the first, or the one `--frame` names), analysed and
dropped, so memory is one frame's worth however long the series is, and the
results are appended to long-format CSV tables in `DIR`:

```console
$ rmda defects series/cu.*.xyz --out res --defects-xyz
series [1/3] 0: series/cu.0.xyz
0: grains=1 measured=1 V=0 I=0 gb_V=0 gb_I=0 clusters=0 wall=0.2s
series [2/3] 1: series/cu.1.xyz
1: grains=1 measured=1 V=1 I=0 gb_V=0 gb_I=0 clusters=1 wall=0.2s
series [3/3] 2: series/cu.2.xyz
2: grains=1 measured=1 V=2 I=0 gb_V=0 gb_I=0 clusters=2 wall=0.2s
series: tables in res
```

**Order and index.** Files are ordered by the *trailing integer in their name*
(`frame-999.data` → 999, `cu.2.xyz` → 2), which becomes the `file_index`
column; the shell's glob order does not matter. Files with no number in the name
(a baseline such as `reference.data`) come first, numbered −n … −1 in the order
given. Two files with the same index are refused.

| file | one row per | columns |
|---|---|---|
| `summary.csv` | file | `file_index, file, timestep, natoms, grains, grains_measured, sites, atoms, vacancies, interstitials, gb_vacancies, gb_interstitials, balance_residual, clusters, largest_cluster, cluster_cutoff, wall_s, rss_mb, peak_rss_mb, net_vacancies, gb_net_vacancies` |
| `grains.csv` | grain per file | `file_index, file`, then the per-grain table's columns |
| `clusters.csv` | defect cluster per file | `file_index, file, timestep, cluster, grain, multi_grain, size, vacancies, interstitials, net, cx, cy, cz` (`net` = I − V; centre wrapped into the box); with `--antisites` also `antisites`, and antisite-only clusters as extra rows |
| `grain_events.csv` | grain event per file (with `--segment`) | `file_index, file, timestep, event, stable, other` |
| `run.txt` | — | the settings that change results, one line |
| `defects/` | file (with `--defects-xyz`) | `vacancies`, `interstitials`, and `non_lattice` with `--non-lattice-atoms`, as `<index>/vacancies.xyz`, `vacancies-<index>.xyz` or one appended `vacancies.xyz` (see `--defects-layout`). On resume an appended file drops any frame whose summary row is missing; `--overwrite` deletes it |

**Grain tracking** (with `--segment`): `stable_grain` in `grains.csv` and
`grain` in `clusters.csv` and the marker files are stable ids, carried across
files by atom-id overlap (§2.8). `grain_events.csv` records `new` and
`continued` grains, `split` (`other` = the grain it came from) and `merged`
(`other` = the grain it went into):

```csv
file_index,file,timestep,event,stable,other
100,bicrystal.100.xyz,0,new,0,-1
100,bicrystal.100.xyz,0,new,1,-1
200,bicrystal.200.xyz,0,continued,0,-1
200,bicrystal.200.xyz,0,continued,1,-1
```

Tracking needs atom ids that persist between files; a renumbered `write_data`
breaks it.

**Resume.** A rerun with the same `--out` skips every file already in
`summary.csv` and continues with the rest, so an interrupted job is restarted
with the same command. The summary row is written last, so a file interrupted
half-way is redone. With `--segment`, the last finished file is re-read first to
restore the tracker's grain ids. A rerun with *different* settings is refused
rather than mixing results:

```console
$ rmda defects series/cu.*.xyz --out res --min-atoms 50
error: res holds results from different settings; use --overwrite or another --out
  before: regions=[] grain_column=None boundary_width=None min_atoms=100 rmsd_cutoff=0.1 reference=Auto patch_size=None segment=None cluster_cutoff=None
  now:    regions=[] grain_column=None boundary_width=None min_atoms=50 rmsd_cutoff=0.1 reference=Auto patch_size=None segment=None cluster_cutoff=None
```

`--overwrite` deletes the tables (`summary.csv`, `grains.csv`, `clusters.csv`,
`grain_events.csv`, `antisites.csv`) and starts again (an existing `defects/`
directory is left in place and overwritten file by file).

Read the tables with `pandas.read_csv("res/grains.csv")`; a measured grain's
vacancies over time are
`g[g.status == "measured"].pivot(index="file_index", columns="stable_grain", values="vacancies")`.

### 3.8 `rmda cluster` — connected components

Groups atoms into connected components under a neighbour criterion, OVITO's
`ClusterAnalysisModifier` and LAMMPS' `compute cluster/atom`. Bonds are
minimum-image, so a component may wrap the box; there is no seam.

```console
$ rmda cluster --cutoff 2.6 fcc-thermal.xyz
read 1 frame(s), 32 atoms in frame 0
cluster: mode=cutoff(2.6) min_size=1
cluster_count: 1
cluster_largest: 32
cluster_largest_fraction: 1
cluster_mean_size: 32
cluster_atoms_retained: 32
cluster_count_retained: 1
cluster_min_size: 1
cluster_mode_cutoff: 2.6
cluster_size_histogram:
  size=32 clusters=1
```

| Flag | Meaning |
|---|---|
| `--cutoff <r>` | bond distance in Å; bonds every pair within `r` |
| `--knear <k>` | bond each atom to its `k` nearest neighbours instead |
| `--pair-cutoff <A-B=r>` | per type-pair cutoff, repeatable; types by number or name (`1-2=2.9`, `Cu-Zr=3.1`); a pair not listed never bonds |
| `--min-size <n>` | clusters below `n` are reported but flagged out; default 1 |
| `--select <expr>` | cluster only atoms matching the expression (`type == 2`, `z > 500 && type != 3`; fields `x y z sx sy sz type id`) |
| `--only-selected` | cluster only atoms flagged by the input's `Selection` column (OVITO's `only_selected`); with `--select`, both must hold |
| `--sort-by-size` | number clusters largest first, ties in first-atom order |
| `--unwrap` | write unwrapped coordinates to `-o` (OVITO's `unwrap_particles`) |
| `--table <file>` | write the per-cluster table (CSV): size, mass, centre of mass, Rg, gyration tensor, percolation flag |

**Exactly one of `--cutoff`, `--knear` and `--pair-cutoff` is required**, and
passing two is an error. This is a deliberate deviation from OVITO, whose `NeighborMode` defaults
to the k-NN criterion: a default here would mean silently picking a radius from
the data, and every cluster count in the output would then depend on an
undisclosed parameter. Refusing to guess costs one flag and makes the report
reproducible from the command line alone.

**`--knear` needs a radius to search in and does not know one.** It estimates a
search radius from the number density, builds a cell list at that radius, and
widens it until every atom really has `k` neighbours inside; a frame too thin to
supply `k` neighbours anywhere is refused rather than silently producing
clusters from a partial neighbour list. The k-NN relation is asymmetric — `i` may
count `j` among its `k` nearest while `j` does not count `i` — so **every**
returned pair is unioned rather than only the `i < j` half. Dropping the other
half would split components that are genuinely connected and quietly change the
answer, so the asymmetry costs one redundant union per pair.

**What it writes.** The per-atom `Cluster` label and `ParticleCluster`; the
`ClusterSize` per-cluster column is kept *out* of the atom table
(`per_particle = false`), indexed by cluster id. `-o` writes both per-atom
columns:

```console
$ rmda cluster --cutoff 5.0 -o clusters.xyz two-blobs.xyz
wrote 2 properties to clusters.xyz
  Cluster, ParticleCluster
```

**Cluster ids are OVITO's (3.17 and later): 0-based, `-1` for an atom in no
cluster.** (OVITO before 3.17 numbered from 1, with 0 for no cluster.) They
follow atom order, not size or position: the component containing the
lowest-numbered atom gets id 0, the next lowest 1, and so on (`--sort-by-size`
renumbers largest first). Size-ordered ids by default would change an id when a
cluster grows, so a stored id would silently refer to a different cluster in the
next frame.

**`ParticleCluster` is the filtered view.** It is the cluster id for atoms in a
cluster at or above `--min-size` and `-1` for atoms in a smaller cluster or in
none; `Cluster` keeps every id. Per-cluster columns (`ClusterSize`, the `--table`
rows) have one row per id, row `k` for id `k`, with the id in the `cluster`
column (`ClusterIdentifier` in Python's `f.tables["clusters"]`).

**`cluster_mode_cutoff` is `NaN` in k-NN mode** because there is no cutoff —
the scalar reports the radius actually used, and `--knear` has none. With
`--pair-cutoff` it is the largest pair cutoff.

**Type-pair cutoffs.** `--pair-cutoff 1-1=2.9 --pair-cutoff 1-2=3.1` bonds two
atoms when their distance is at or below the cutoff of their type pair (either
order); a pair with no entry never bonds. A type is a number, or a name from the
input's type labels (extxyz `species`, data-file type labels). An unknown type, a
pair given twice, or a cutoff that is not positive and finite is refused (OVITO
silently skips an unknown type or a zero cutoff, leaving the pair unbonded).
OVITO 3.15's `ClusterAnalysisModifier` has no pair-cutoff mode of its own; the
equivalent pipeline is `CreateBondsModifier(mode=Pairwise)` then
`ClusterAnalysisModifier(neighbor_mode=Bonding)`, and the result equals it
(ids, sizes, COM, Rg and gyration tensor, with types by id and by name, a
missing pair, open boundaries and `only_selected`; design §18.3).

**Selections cluster defects or one phase.** `--select 'type == 1'` or
`--only-selected` restrict the analysis to the chosen atoms: unselected atoms take
part in no bond (a k-nearest bond is always to another selected atom), get
`Cluster = -1` and `ParticleCluster = -1`, and the fractions and means are over the
selected atoms (`cluster_atoms_selected` says how many).

**`--table` is OVITO's cluster table.** One row per cluster id:

```console
$ rmda cluster --cutoff 2.9 --select 'type == 1' --sort-by-size --table al.csv hea-poly.data
$ head -2 al.csv
cluster,size,mass,com_x,com_y,com_z,rg,g_xx,g_yy,g_zz,g_xy,g_xz,g_yz,percolates
0,28,755.4830779999995,-6.1533633905522525,18.900383312210515,38.48338987098069,7.397002674995341,15.01133578691456,34.156044421253256,5.548268365720419,-15.668643506926113,4.034613053772484,-8.357305465254163,0
```

The centre of mass, radius of gyration and gyration tensor
(`G_ab = Σ m (u − com)_a (u − com)_b / M`, `Rg = √tr G`) are computed as OVITO
computes them: over coordinates **unwrapped** along the cluster's bonds from its
first atom, so a cluster crossing a periodic face is whole; **mass-weighted** when
the input has masses (a per-atom `mass`/`Mass` column, else a data file's
`Masses` section), unit weights otherwise (`cluster_mass_weighted` says which;
OVITO also takes element masses from type names, rmda does not); the centre is not
wrapped back into the box. Against OVITO 3.15 on periodic, open and selected
frames, including the 42k- and 256k-atom HEA frames, ids agree up to relabelling,
sizes exactly, and centres, Rg and tensors to 1e-14 relative.

**A cluster bonded to its own periodic image has no unwrapping**, in rmda or in
OVITO: its coordinates then depend on which spanning tree was followed, and its
centre and Rg mean nothing. rmda detects this (a bond closing a loop whose
minimum-image vectors do not sum to zero) and reports it as `percolates = 1` and
`cluster_percolating`; OVITO reports nothing and gives different numbers.

**Parallel, and identical at any thread count.** Bonds are generated in parallel
and unioned serially in atom order, so labels, unwrapped coordinates and every
float are bit-identical at `-t 1` and `-t 12`. On the 256k-atom HEA frame,
`--cutoff 2.9` takes 1.2 s serial and 0.3 s on 12 threads, `--knear 4` 2.8 s and
0.6 s (reading included).

**Histogram convention.** `cluster_size_histogram` prints one line per occupied
size as `size=<n> clusters=<count>`; `size=1` is every atom with no bond inside
the criterion, so a run whose cutoff is too small produces a huge `size=1` line
rather than an error. On a two-component 6000-atom fixture at `--cutoff 5.0`:

```console
cluster_size_histogram:
  size=1 clusters=2
  size=2 clusters=1
  size=2997 clusters=1
  size=2999 clusters=1
```

### 3.9 `rmda rdf` — radial distribution function

Total `g(r)`, coordination number `n(r)`, per-type-pair partials `g_ab(r)`, and the raw
pair counts `H_k`. Normalisation follows design.md, *Warren–Cowley, RDF, cluster analysis* (OVITO `RDFModifier` and LAMMPS
`compute rdf` agree with it).

```console
$ rmda rdf --rmax 3.5 --dr 0.25 m2_fcc_thermal.xyz
read 1 frame(s), 32 atoms in frame 0
rdf: rmax=3.5 dr=0.25 partials=true coordination=true
# r=0.00000 0.25000 0.50000 0.75000 1.00000 1.25000 1.50000 1.75000 2.00000 ...
rdf_g_r: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.519729 2.205682 3.305065 0.823842 ...
rdf_pair_counts: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 10.000000 53.000000 97.000000 ...
rdf_coordination_number: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.625000 3.937500 10.000000 ...
rdf_g_1_1: 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.519729 2.205682 3.305065 0.823842 ...
rdf_volume: 377.93306700000005
rdf_volume_effective: 0
rdf_centres: 32
rdf_rmax: 3.5
rdf_dr: 0.25
rdf_bins: 14
rdf_atoms: 32
rdf_number_density: 0.08467107748473354
rdf_pairs: 214
rdf_pairs_possible: 496
rdf_coordination_at_rmax: 13.375
```

`rmax` is 3.5 and not 5.0 because this fixture's cell is 7.23 A across: half of
that is 3.615, and a longer `rmax` would be refused (see below). `rdf_pairs` is
214 of 496 possible, i.e. 43% of all pairs fall inside the range.

| Flag | Meaning |
|---|---|
| `--rmax <r>` | largest pair distance in Å; **required** |
| `--dr <d>` | bin width in Å; default 0.02 |
| `--no-partials` | skip the per-type-pair `g_ab(r)` |
| `--no-coordination` | skip the `n(r)` integral |
| `--volume <mode>` | normalisation volume: `auto` (default), `box`, `effective`, or a volume in Å³ |
| `--edge-correct` | count pairs only around atoms more than `rmax` from a free surface |

**Thin films, slabs and clusters: `--volume` and `--edge-correct`.** `g(r)` is
normalised by the number density `N/V`. The box volume is right only when the
material fills the box; for a film in a box with vacuum it is too large, and
`g(r)` tends to `V_box/V_film` instead of 1 (1.6 on a test slab). `auto` uses
the **effective** volume `N/ρ` whenever an axis is open or a periodic axis has an
empty layer (a vacuum gap), and the box otherwise, so periodic bulk output is
unchanged. ρ is the material's number density, measured as a Gaussian-weighted
neighbour count (width up to 4 Å) around atoms at least four widths from any
surface. It is exact for a uniform medium and insensitive to the lattice planes:
on BCC W slabs and a free cluster it matches the true density to 0.01–0.1 %.
`rdf_volume`, `rdf_volume_effective` and `rdf_number_density` report what was
used.

The effective volume fixes the density, but the atoms within `rmax` of a surface
still miss neighbours, so `g(r)` of a film sags below 1 at large `r` (0.90 at
8–12 Å on a 50 Å slab). `--edge-correct` counts pairs only around atoms more
than `rmax` from any surface (`rdf_centres` of them), so every counted shell is
whole: on the test slab `g(r)` then equals bulk's to the density's accuracy and
`n(12 Å)` is bulk's 460. Use it whenever the film is thicker than `2·rmax`; it
is refused when no atom is that deep.

**`--rmax` is required.** A default would put a number nobody chose into the
output, and every peak position would depend on it.

**The `# r=` line carries the bin edges.** Each histogram has `bins + 1` values
below it. Without the abscissa a histogram is a list of numbers and the consumer
has to know `dr` to use it; with it, the last edge is exactly the `--rmax` you
typed, so a range that is not a multiple of `--dr` still integrates over the
range the pairs were counted in.

**`rmax` is exclusive and that is uniform.** A pair at exactly `rmax` is dropped
whenever `rmax` is a multiple of `dr`, and also when it is not — the check is on
`rmax` itself, not on a bin index. When `rmax` is *not* a multiple of `dr` the
last bin is narrower than the rest and its shell volume is the sliver volume, so
its `g` is not comparable to a full bin's `g` at the same height. Pick an `rmax`
that is a multiple of `dr` unless you specifically want the boundary shell.

**`rmax` past half the smallest cell height is refused.** The minimum-image
convention considers one image per pair, so beyond that limit a real neighbour is
folded onto a shorter one and the histogram would be wrong in a way that looks
like structure. The limit is the perpendicular height between opposite faces,
which in a tilted cell is shorter than the edge:

```console
$ rmda rdf --rmax 16.0 sc.xyz
error: invalid data: rdf rmax 16 exceeds half the smallest cell height (15.0000 A); \
       the minimum-image convention cannot resolve pairs that far
```

Exactly half the smallest height is allowed. A non-periodic frame is exempt. `rmax / dr`
may ask for at most 10 million bins.

**`n(r)` is the integral of `g(r)`, not an independent number.** It is computed
that way, and `rdf_coordination_at_rmax` is the last bin of
`rdf_coordination_number` — the two always agree. The `4*pi` angular factor is
inside the shell volume, which is why the integral has no `4*pi` in front of it.

**Partials are emitted only for pairs that exist inside `rmax`.** A partial with
no pairs in range is absent, not zero: absent means "not measured", zero would
claim the measurement found nothing. Order is fixed at `a <= b` ascending, so
output is byte-identical between runs.

**Partials are normalised by their own pair population.** `g_ab` divides by
`N_a * N_b`, not by `N^2`; a same-type pair is doubled to compensate for the
half pair list, a mixed pair is not. Both are easy to get wrong in a way that a
single-element test cannot see, so the substitutional-alloy test compares every
partial against an independent brute-force count of its own type pair.

**`-o` is refused, not ignored.** `g(r)` is a frame-level histogram, not a
per-atom property, so there is nothing to put in an atom file. `-o` is a global
flag, so passing it means a file was expected; silently writing none is how
someone concludes a frame had no pairs in range when in fact nothing was measured
per atom. Use a shell redirect to keep the table.

**It builds its own neighbour list at `rmax`.** A narrower shared list would
truncate the query stencil and drop the far shells, and `rmax` is the largest
cutoff an analysis may declare. The cost is one extra cell build when `rdf` runs
alongside narrower analyses.

**Bin width barely costs anything.** On the 256,000-atom relaxed FCC fixture of §4,
`--rmax 5.0` takes 3.7 s at `--dr 0.05` and 3.8 s at `--dr 0.02` — a 12× change
in bin count moves the wall time by 3%. The cost is the neighbour search, not the
histogram, so a fine `dr` is close to free and there is no reason to coarsen it
to save memory.

### 3.10 `rmda wc` — Warren–Cowley short-range order

`αᵢⱼᵐ = 1 − pᵢⱼᵐ/cⱼ` per species pair and per shell, the same quantity the
`WarrenCowleyParameters` OVITO modifier computes (design.md, *Warren–Cowley, RDF, cluster analysis*). Shells are given as
cumulative k-nearest ranks, not as radii.

On an ordered L1₂ (Cu₃Au) lattice — corners of type 1, face centres of type 2 — the
twelve nearest neighbours are a textbook census, and the parameters come out as the
textbook says:

```console
$ rmda wc --wc-shells 12 l12.xyz
read 1 frame(s), 16384 atoms in frame 0
wc: ranks=0,12 types=1,2 central=16384 skipped=0 probe=4.732 A
wc_shells: 1
wc_ntypes: 2
wc_atoms_central: 16384
wc_atoms_skipped: 0
wc_probe_radius: 4.7319353058274665
wc_conc_1: 0.25
wc_conc_2: 0.75
wc_m1_nneigh: 12
wc_asymmetry_max: 0.00000000000006822320486321587
wc_m1_1_1: 1
wc_m1_1_2: -0.3333333333333257
wc_m1_2_1: -0.33333333333339393
wc_m1_2_2: 0.11111111111109111
```

A corner atom's twelve nearest are all face centres; a face centre's twelve are four
corners and eight face centres. With `c_1 = 1/4` and `c_2 = 3/4` that gives
`α₁₁ = 1`, `α₁₂ = α₂₁ = −1/3`, `α₂₂ = 1/9`. The `6.8e-14` in `wc_asymmetry_max` is
not a calculation artefact: this fixture's coordinates are written to six decimals,
so the lattice is slightly strained and the two directions disagree in the
seventeenth digit. On an exactly symmetric input the number is 0.

| Flag | Meaning |
|---|---|
| `--wc-shells "12,18"` | cumulative neighbour counts ending each shell; default `0,12` |
| `--only-selected` | measure only atoms flagged by a `Selection` column |
| `--only-same-type` | report only the diagonal `αᵢᵢ` |
| `--per-particle` | also write the per-atom matrix (`-o` needs this) |
| `--edge-correct` | measure only atoms deeper below a free surface than their own k-th neighbour (thin films, clusters) |

**Read the sign before the number.** Negative α means the pair likes each other's
company (short-range order); positive means it avoids itself. `α = −1` is the
extreme: every neighbour is the other species where that species is a minority.
`α = 0` is a random alloy; a single-species frame returns all zeros. Note `α₂₂ = +1/9` above: the face centres are *less* ordered
with themselves than a random alloy would be, because they see themselves 8 times
out of 12 against a global 9 out of 12.

**The scalar names carry type *ids*, so the ids are on stderr.** `wc_m1_7_2` is
species 7 around species 2; without the `wc: ... types=` line that is not
interpretable. Sparse ids are compact rather than padded: a frame whose types are
`{1, 7}` gets a 2×2 matrix, matching the reference's `np.unique`.

**A shell is a rank range, so `--wc-shells 6,12` is two shells of six**, not one
shell of twelve and not a shell of twelve plus another of six. `wc_m1_nneigh` and
`wc_m2_nneigh` report the widths so the reading is checkable from the output itself.
A leading zero is optional — `"12"` and `"0,12"` are the same run — and a list that
is not strictly increasing is refused by name.

**Concentrations are over what you measured, not always the whole box.** Under
`--only-selected` they are taken over the selected atoms plus every atom their
outermost shell reaches, as in the reference. A type absent from that set gets
`wc_conc_j = 0` and therefore `NaN`, and the run says which entries are unmeasured
on stderr — a type with no *central* atom leaves its whole row `NaN` too, which is
the reference's `nanmean` of an empty axis.

**`--only-selected` needs the mask to come from somewhere.** It is read from a
`Selection` column in the input file, 0/1; anything else in that column is refused
rather than coerced, because `0.5` is truthy and a per-atom weight wearing that
name would quietly select everything. A frame that *carries* a selection is still
measured in full without the flag — reading a mask off the file is not the same as
asking for a subset, and honouring it silently would make one command measure
different things on two files.

**`wc_asymmetry_max` is 0 on a perfect lattice, not in general.** `pᵢⱼ/cⱼ` and
`pⱼᵢ/cᵢ` divide bond counts by the same two populations, but a k-nearest bond is
not symmetric — `j` can be among the 12 nearest of `i` without `i` being among the
12 nearest of `j` — so on a thermal or relaxed frame the two counts differ (0.036
on the 42k-atom HEA frame). `--only-selected` breaks the tie further, because
there the central populations stop matching the concentration set. It is omitted
entirely under `--only-same-type`, where there is no off-diagonal left to compare.

**`--edge-correct` for films and free clusters.** The k nearest of a surface atom
reach past its missing neighbours into the second shell, which biases α. With
the flag an atom is measured only when no free surface (an empty region of the
occupancy grid `rdf --edge-correct` uses, or an open face) is nearer than its own
k-th neighbour; concentrations stay over every atom or the selection. The
excluded and measured counts are printed as `wc_atoms_surface` and
`wc_atoms_measured`. On an ordered L1₂ film this recovers the bulk matrix to
1e-12; periodic bulk excludes nothing.

**`--per-particle` writes one multi-component column, `WarrenCowleyShell{m}`.** It
holds *this atom's* row of the matrix, flattened `i*ntypes + j`; atoms that are not
central are `NaN` across all components, because zero is a real value meaning "this
neighbour type does not occur here". Useful as a spatial map of where the order is.

**`-o` without `--per-particle` is refused, not ignored.** The global numbers are
not per-atom properties, so there is nothing to write; `-o` is a global flag, so
passing it means a file was expected and silently writing none is how someone
concludes the alloy had no order.

**It builds its own cell list, sized but not bounded.** Shells are ranks, so there
is no cutoff to choose and nothing for a shared narrower list to truncate.
`wc_probe_radius` reports the density estimate used to size it.

**Cost is the k-NN search; it runs in parallel and is identical at any thread
count.** Each atom's shell histogram is a row of `u16` (2 B per shell and type),
the rows are filled in parallel, and the averages are summed over them in atom
order, so every printed digit is the same at `-t 1` and `-t 12`. The k nearest
are selected, not fully sorted, from the ~450 candidates. On the 256,000-atom HEA
frame, `--wc-shells 12,18` takes 5.4 s at 16 MB serial and 1.2 s on 12 threads
(reading included); with `--per-particle`, 1.8 s and 156 MB (the output column
itself is 66 MB).

### 3.11 `rmda ws` — defects and method-A voids

```console
$ rmda ws --self --lattice fcc --nn-distance 2.49 --void-min-sites 2 file.dump
ws_vacancies: 3
ws_interstitials: 0
ws_interstitial_sites: 0
ws_sites: 864
ws_mean_displacement: 0.00013
ws_max_displacement: 0.00015
ws_self_seeded: 0
ws_fit_a: 3.525904
ws_fit_tolerance: 1.246594
void_count: 1
void_sites: 3
void_isolated_sites: 0
void_max_equivalent_radius: 1.987290
void_min_sites: 2
void_cutoff: 2.617850
```

Read `ws_vacancies` and `void_count` as different quantities: three empty cells
adjacent to each other are **one** void and **three** vacancies, and
`void_count` is the one that answers "how many craters".

**Which reference mode?** If you have a reference configuration, use
`--reference` — it assumes nothing and is 40× faster on a 256k-atom frame (1.1 s
versus 41 s for `--self`):

```console
$ rmda ws --reference perfect.data dump.lammpstrj
```

If the cell has strained between the two frames, add `--map to-reference`, which
carries positions through fractional coordinates first. Without it, every atom
reads as displaced.

| Flag | Applies to | Meaning |
|---|---|---|
| `--lattice`, `--nn-distance` | all | reference lattice; defaults `fcc` and 2.86 Å; under `--self` the nn is only the search's starting guess, and the fitted scale comes from the atoms |
| `--self` | — | fit the lattice to the frame |
| `--reference FILE` | — | use a reference frame's sites; mutually exclusive with `--self` |
| `--map none\|to-reference` | `--reference` | how to carry positions into the reference frame |
| `--tolerance <R>` | `--self`, `--reference` | `--self`: how far an atom may sit from a site, default half the nearest-neighbour distance; `--reference`: only the `ws_config_unmatched` threshold, default 1.05 × the reference's nearest-neighbour distance |
| `--cutoff <R>` | `--reference` | ignored with a warning: pure WS has no site search radius |
| `--orientation w x y z` | `--self` | orientation seed, normally a PTM quaternion |
| `--search-sample N` | `--self` | atoms used for the fit search (default 100000); the winner is re-counted over every atom |
| `--void-min-sites N` | `--self` | smallest empty-site cluster to report as a void; default 1 (every empty site) |
| `--defects DIR` | all | write the defect sites to three XYZ files in `DIR` |

**Use `--orientation` when you can.** An unseeded `--self` fit starts from the
identity and cannot find a grain turned away from the box axes. `ws_self_seeded: 0`
in the output is the flag that a displacement number should not be read without
thinking.

#### 3.11.1 Fitting the reference to the frame (`--self`)

`rmda ws --self` is Wigner–Seitz analysis against a lattice fitted to the frame
itself, after Nordlund & Averback (Phys. Rev. B 57, 7556 (1998)): the reference
is built from the atoms being analysed, so no clean reference configuration is
needed. A typical run:

```console
$ rmda ws --self --lattice fcc --nn-distance 2.49 --void-min-sites 2 --defects out/ sample.data
```

and, when the sample is rotated off the box axes, run PTM first so the fit gets
a seed:

```console
$ rmda ptm sample.data -o ptm.out.xyz
```

`rmda ptm -o` writes the per-atom `StructureOrientation` quaternion
(`w x y z`) as a column; in a single crystal the quaternion is the same for
every atom, so take one row's four values and pass them as
`--orientation w x y z`. The Python binding does this in one pass, because
the attached property is on the frame itself:

```python
import rmda
f = rmda.read("sample.data")
rmda.ptm(f)                    # attaches StructureOrientation
q = f["StructureOrientation"][0]   # w, x, y, z
r = rmda.ws(f, fit_self=True, lattice="fcc", nn_distance=2.49,
           orientation=list(q), void_min_sites=2)
```

What the fit does:

- **Lattice type** is `--lattice {fcc,bcc,hcp,sc,dia,dhcp}`; the site basis
  comes from `Lattice::primitive()`, validated against ASE's lattice generators.
- **Scale and cell geometry** are derived from the atom count and site density;
  `--nn-distance` is only the search's *starting* spacing.
- **Origin and orientation** are fitted to the atoms. The orientation search
  starts from the identity, so pass `--orientation` when the crystal is not
  aligned with the box.
- **Assignment** is a bounded nearest-site search: an empty Wigner–Seitz cell
  is a vacancy, a site with *n > 1* atoms contributes *n − 1* interstitials.
- **Outputs**: `ws_mean_displacement`/`ws_max_displacement` on stdout and the
  per-atom `ws_displacement` property on `-o`; `--defects DIR` writes
  `vacancies.xyz`, `interstitials.xyz` and `interstitials_sites.xyz` (§3.12).
- **Voids**: `--void-min-sites N` groups empty sites into clusters of ≥ *N*
  sites; read `void_count` (how many craters) separately from `ws_vacancies`
  (how many empty cells).

### 3.12 Writing the defects as files

`--defects DIR` writes three XYZ files, one row per line, so a viewer or another
tool can read them directly:

```console
$ rmda ws --self --void-min-sites 2 --defects out/crater sample.data
```

| File | One row per | Species | Extra columns |
|---|---|---|---|
| `out/crater/vacancies.xyz` | empty lattice site | `VAC` | `site_index`, and `void_id` under `--self` |
| `out/crater/interstitials.xyz` | input atom on an over-occupied site | the atom's own | `site_index`, `site_occupancy` |
| `out/crater/interstitials_sites.xyz` | over-occupied site | `INT` | `site_index`, `site_occupancy` |

All three are always written, even when empty, so a downstream step can rely on
the filenames existing. `DIR` is created if missing.

**`rmda defects --defects DIR` (§3.7) writes its own pair of files**,
`vacancies.xyz` and `interstitials.xyz`, with `grain`, `boundary` and `cluster`
columns and one row per interstitial (not per crowded-site atom).

**A vacancy is a site, not an atom.** That is the whole reason these are separate
files and not extra columns on `-o`: `ws_occupancy` holds one entry per *site*,
while every other WS property holds one per *atom*, so the two cannot share a
table without one of them being silently misaligned. On `-o` you get
`ws_site`, `ws_displacement` and the per-atom `ws_interstitial` flag, which is
enough to select interstitials but cannot express a vacancy.

**`interstitials.xyz` lists every atom on a crowded site, not one "extra" atom.**
A site claimed by two atoms has occupancy 2, and which of the two is the
interstitial is a question about history that the geometry does not answer — the
Wigner-Seitz construction only sees the total. So the file lists both, and
`ws_interstitials` in the summary counts the *excess*:

| | crowded site of occupancy 2 |
|---|---|
| `ws_interstitials` (stdout) | 1 |
| rows in `interstitials.xyz` | 2 |
| rows in `interstitials_sites.xyz` | 1 |

Read `interstitials.xyz` as "atoms involved in interstitials" and the stdout
counter as "how many interstitial atoms there are". For a crowd of *n*, the file
has *n* rows and the counter has *n-1*.

**Cells differ between the files on purpose.** The site files describe the
reference lattice, so they carry the analysis cell — the fitted cell under
`--self`, or the reference frame's cell under `--reference` (including after
`--map to-reference`). `interstitials.xyz` holds real input atoms, so it keeps the
input frame's cell and its species. Under a strained input these two cells differ
and reading the site file as if it were in the input frame will mislead you.

**Every position in all three files is inside the cell its header declares.** A
site grid need not live in the box the header describes: under `--self` the region
is indexed modulo the box, so site index 0 can sit at the far edge of the fitted
reference's origin, and unwrapped site positions land tens of angstroms outside a
box of 141 Å. Positions are therefore wrapped into the written cell, so a viewer
places every defect inside the box it draws. The wrapping changes coordinates
only, never counts.

**`void_id` groups a crater, so you can select one.** Under `--self`,
`--void-min-sites N` decides what counts as a crater, and each vacancy row carries
the id of the cluster it belongs to (`-1` if it is in none). `--reference` omits
the column entirely rather than writing `-1` everywhere, since that engine runs
no void pass.

### 3.13 `rmda analyze` — several analyses over the frame window

Reads the window **once** and runs every requested analysis on each frame, so one
pass over the file answers several questions instead of re-reading it per command.
At least one analysis is required:

```console
$ rmda analyze file.dump
error: invalid data: analyze needs at least one analysis: --ptm, --rdf, --wc, --cluster
```

| Flag | Runs |
|---|---|
| `--ptm` | PTM classification; `--ptm-rmsd`, `--ptm-structures` as `rmda ptm` |
| `--rdf --rdf-rmax <r>` | g(r); `--rdf-rmax` is required with `--rdf`; `--rdf-dr`, `--rdf-no-partials`, `--rdf-no-coordination`, `--rdf-volume`, `--rdf-edge-correct` as `rmda rdf` |
| `--wc` | Warren–Cowley; `--wc-shells` as `rmda wc` |
| `--cluster` | cluster census; exactly one of `--cluster-cutoff` / `--cluster-knear`, plus `--cluster-min-size` |

Unlike the single-analysis commands, which read one frame, `rmda analyze` processes
**every frame the window selects** — the whole file by default; `--frame`,
`--start`, `--end`, `--stride` narrow it as everywhere else. Grain segmentation
and the defect census are not offered here; a defect time series over many files
is `rmda defects FILES... --out DIR` (§3.7.1).

```console
$ rmda analyze --ptm --cluster --cluster-cutoff 2.6 m2_fcc_thermal.xyz
analyze: ptm(rmsd=0.1, types=fcc,hcp,bcc) + cluster(cutoff=2.6)
frame 0:
  cluster_atoms_retained: 32
  cluster_count: 1
  cluster_count_retained: 1
  cluster_largest: 32
  cluster_largest_fraction: 1
  cluster_mean_size: 32
  cluster_min_size: 1
  cluster_mode_cutoff: 2.6
  ptm_bcc: 0
  ptm_fcc: 31
  ptm_hcp: 0
  ptm_ico: 0
  ptm_other: 1
  ptm_sc: 0
  cluster_size_histogram: 0.000000 0.000000 ... 0.000000 1.000000
analyze: read 1 frame(s), 32 atoms in frame 0
```

(The `analyze:` banner and summary go to stderr; the `frame N:` block is stdout.)
Scalars print as `name: value`, sorted by name; array-valued results follow as
space-separated series, with `rdf_*` arrays below their `# r=` bin-edge line.

**`-o` is a prefix, not a file.** Per frame it writes:

- `PREFIX.<frame>.xyz` — the frame with the per-atom properties attached
  (`--ptm` and `--cluster` produce them; `--wc` and `--rdf` are
  frame-level and have nothing to attach), naming the columns it wrote;
- `PREFIX.rdf.<frame>.dat` — the g(r) table, when `--rdf` ran;

`<frame>` being the index within the window. The directory part of the prefix
must already exist. A run whose analyses would produce neither is refused rather
than writing nothing:

```console
$ rmda analyze --wc -o out/frame file.dump
error: invalid data: rmda analyze cannot write out/frame: none of the requested analyses produced per-atom properties (--wc and --rdf are frame-level; per-atom columns come from --ptm or --cluster)
```

### 3.14 `rmda void` — voids (method A; method B with `--experimental`)

**Method A** (default; §2.6b, design §17.1) runs the `rmda defects` census and
clusters its vacancies. A BCC W block with a single vacancy, a divacancy and a
15-site void:

```console
$ rmda void m.xyz --table voids.csv
void_grains: 1
void_grains_measured: 1
void_vacancies: 18
void_min_sites: 2
void_cutoff: 2.9530
voids: 2
voids_bulk: 2
voids_boundary: 0
voids_surface: 0
void_sites: 17
void_isolated_vacancies: 1
void_volume: 269.4884
void_largest_sites: 15
void_largest_radius: 3.8432
$ cat voids.csv
void,sites,volume,radius,x,y,z,grain,multi_grain,class
0,15,237.783878,3.843244,18.990000,25.320000,25.320000,0,0,bulk
1,2,31.704517,1.963409,26.111250,7.121250,7.121250,0,0,bulk
```

| Flag (method A) | Meaning |
|---|---|
| `--min-sites N` (alias `--void-min-sites`) | smallest vacancy cluster that is a void; default 2 (1 makes every vacancy a void) |
| `--cutoff Å` | joining distance; default each grain's halfway between first and second shells |
| `--table FILE` | per-void CSV: `void,sites,volume,radius,x,y,z,grain,multi_grain,class` |
| `--sites FILE` | the void vacancy sites as extxyz (species `VAC`, `void_id`, `grain`, `class`) |
| `--site-reference auto\|global\|patch\|local\|periodic` | the reference, as `rmda defects --reference` |
| `--patch-size`, `--ws-reference FILE`, `--ws-affine` | as in `rmda defects` |
| `--grain-region EXPR` (repeatable), `--grain-column NAME`, `--segment` | grains, as `rmda defects --region`, `--grain-column`, `--segment` (default settings) |
| `--boundary-width`, `--min-atoms`, `--rmsd-cutoff`, `--dhcp` | as in `rmda defects` |
| `--out DIR` | time series over every FILE (see below); `--overwrite` discards existing tables |
| `--shard I/N`, `--sites-xyz` | with `--out`: run shard I of N, as `rmda defects --shard` (§3.18); write each frame's void sites to `sites/<index>.xyz` |

**Series** (`--out DIR`), the same machinery as `rmda defects --out` (§3.7):
files are ordered by the trailing integer of their name (files without one
first), each frame is counted exactly as a single-file run, and three tables are
appended (and `grain_events.csv` with `--segment`) — `summary.csv`, one row per frame; `voids.csv`, one row per void
per frame with the `--table` columns behind `file_index,file,timestep` and a
`stable_grain` column after `grain`; and `grains.csv`, one row per grain per
frame (`grain,stable_grain,status,reference,n_atoms,vacancies,voids,void_sites`).
A rerun skips frames already in `summary.csv`; other settings (or a `rmda
defects` directory) are refused unless `--overwrite`. `--table`, `--sites` and
`--experimental` do not combine with `--out`; `--sites-xyz` writes each
frame's void sites (the `--sites` file) to `sites/<index>.xyz`. `--shard I/N`
and `rmda merge` work as for `rmda defects` (§3.18).

```console
$ rmda void reference.data data.2 data.10 --out voids/
-1: vacancies=0 voids=0 void_sites=0 volume=0.0000 wall=1.3s
2: vacancies=328 voids=4 void_sites=328 volume=5199.5408 wall=1.3s
10: vacancies=328 voids=4 void_sites=328 volume=5199.5408 wall=1.6s
$ head -2 voids/summary.csv
file_index,file,timestep,natoms,grains,grains_measured,vacancies,min_sites,cutoff,voids,voids_bulk,voids_boundary,voids_surface,void_sites,isolated_vacancies,volume,largest_sites,largest_radius,wall_s
-1,reference.data,0,27648,1,1,0,2,2.9530,0,0,0,0,0,0,0.0000,0,0.0000,1.28
```

With `--segment`, grains are tracked across frames exactly as `rmda defects
--out --segment` tracks them (atoms matched by id): `stable_grain` keeps a
grain's id from frame to frame, `grain_events.csv` records new, continued,
split and merged grains, and a void series and a defects series of the same
files report the same ids. Without `--segment` there is nothing to track and
`stable_grain` is the frame's label.

Grains that are not measured are named on stderr; their vacancies are not
counted. `-o` is refused (nothing per atom). Python: `rmda.void(f, min_sites=2,
site_reference="auto", regions=[...], ...)` returns the same scalars, a
`voids_table` dict of columns and `grain_events`; pass one
`tracker=rmda.GrainTracker()` frame after frame to track grains as the series
does.

**Options.** Method B's options (`--lattice`, `--nn-distance`,
`--reference FILE`, `--map`, `--region`, `--samples`, `--seed`, `--epsilon(s)`,
`--strain*`, `--excess`) need `--experimental`, and method A's refuse it.
Method A's grain regions are `--grain-region` because `--region` is method B's
integration box; its reference mode is `--site-reference` because `--reference`
is method B's reference frame. `rmda ws --self --void-min-sites` (§3.11) counts
voids on `ws`'s single lattice.

**Method B** (`--experimental`, §2.7): the φ-field volume. The quadratic
extrapolation passes the pure-W gate only at 0 K; `--thermal-window` passes it
with 0.03–0.06 Å of thermal noise (§2.7.1). The flag stays a real gate.

A worked example on a synthetic FCC block with a three-cell crater:

```console
$ rmda void --experimental --nn-distance 2.4934 --samples 40000 crater.extxyz
phi field: lattice Fcc, nn 2.4934 A, 12 face normal(s), circumradius 1.7631 A
observed nn: 2.4932 A (-0.0 % vs --nn-distance)
atoms: 861 in the sample, 12 ideal face normal(s), epsilon 1.000e-2
region: [4227 A^3]
void volume: 28.2143 +/- 1.7209 A^3 (6.10% of region, 267 of 40000 samples)
         28.21 A^3 of         4227 A^3 = 0.6675 % of the region
```

The exact answer is `3a³/4 = 32.88 Å³`; method B measures 0.86 of it on this
geometry, consistent with the 0.95–0.97 the benchmarks report for carved clusters.

**`observed nn` is not decoration.** A declared `--nn-distance` that disagrees with
the sample does not error — it returns a field built from cells of the wrong size,
and *which way* it is wrong decides what you see. Too large and the ideal cells
swallow the sample, so both a perfect crystal and a crater read as zero void. Too
small and every interstice reads as void. Both look like results, so the command
measures the sample's nearest-neighbour distance and warns past 10 % drift.

| Flag | Meaning |
|---|---|
| `--experimental` | selects method B; acknowledges it is unvalidated on thermal material |
| `--lattice`, `--nn-distance` | the ideal configuration and its face normals |
| `--reference FILE` | take the ideal configuration from a reference frame (fitted to its atoms) |
| `--map none\|to-reference` | carry the sample's positions into the reference cell |
| `--region XLO,XHI,YLO,YHI,ZLO,ZHI` | region to integrate, in Å; default is the box inset by one nn |
| `--samples N` | Monte Carlo samples; default 40000 |
| `--seed N` | sampler seed; the same seed makes two fields comparable |
| `--epsilon E` | count `φ > 1 + E`; default 0.01 |
| `--epsilons E1,E2,E3,…` | measure at each and extrapolate to `φ = 1`; needs ≥ 3 distinct values |
| `--strain` | fit the roto-strain field `T(x)` (§2.7) and evaluate φ through it |
| `--strain-spacing`, `--strain-sigma` | mesh and Gaussian width for `--strain` |
| `--excess` | report the excess over the reference frame's field (needs `--reference`); with one ε (no `--epsilons`) the measurement itself |
| `--thermal-window` | measure 7 thresholds above the frame's largest face gap and extrapolate `V^(1/3)` to `φ = 1` (§2.7.1); not with `--epsilons`, `--excess`, `--strain` |

The region inset is not cosmetic: the field is periodic through the cell, so a
region touching a face integrates across the wrap, where a point's neighbours are
the periodic images' rather than the crystal's.

Extrapolation, with its own diagnostics:

```console
$ rmda void --experimental --nn-distance 2.4934 --samples 60000 \
      --epsilons 0.01,0.02,0.05 crater.extxyz
void volume: 29.7993 +/- 1.4438 A^3 (4.84% of region, 423 of 60000 samples)
extrapolated to phi = 1: 30.6799 +/- 1.4438 A^3 (quadratic extrapolation to phi = 1 from 3 thresholds, residual 0.0000 A^3)
  falling in epsilon: true, fit trustworthy: true
  epsilon 0.0100:        29.80 +/-     1.44 A^3
  epsilon 0.0200:        28.88 +/-     1.42 A^3
  epsilon 0.0500:        25.92 +/-     1.35 A^3
```

An excess over a reference frame is a **void volume only if that reference shares
the sample's local relaxation**. Against an ideal lattice it is a displacement
measurement, and the summary line says so.

Argument mistakes are refused before any field is built — a run that computes a
256k-atom field and then discovers `--map` was meaningless would waste a minute
and report nothing.

### 3.15 `rmda convert`

Rewrites the frame window in another format: `xyz` (extended XYZ, §3.17),
`dump` (LAMMPS text dump, `id type x y z` plus image flags when the input has
them), `bin` (LAMMPS binary dump) or `data` (LAMMPS data file, one frame).
The format is `--format`, or else the suffix of `-o` (`.xyz`, `.dump`,
`.lammpstrj`, `.data`, `.bin`). An `-o` ending in `.gz` is gzip-compressed
(`out.dump.gz`, `frame.data.gz`; not `.bin.gz`, which nothing reads back).
Without `-o` the output is `out.xyz`, `out.dump`, `out.data` or `out.bin` in
the current directory.

```console
$ rmda convert --format xyz -o frames.xyz tests/data/binary2txt_oracle.bin
read 2 frame(s), 8 atoms in frame 0
wrote frames.xyz
$ rmda convert -o frame.dump.gz --frame 1 traj.dump.gz
$ rmda convert --props all --precision full -o lossless.xyz frame.data
writing 3 properties: vx, vy, vz
```

Every frame the window selects is converted, one at a time, so a long
trajectory never sits in memory. `convert` reads every column (it does not apply
the analysis commands' default `--skip-columns`). **Without `--props` it writes
only** positions, types, ids (not in XYZ) and image flags, and type labels only
as the XYZ species. The options:

- `--props all | none | a,b,c` writes the frame's per-atom properties
  (velocities, `c_pe`, or the `ptm.structure`, `ptm.orientation`, … columns of
  an `-o` file from an analysis). `all` writes what the format can carry and
  names the rest on stderr; a name that is missing or cannot be carried is an
  error. Given at all — even `--props none` — it also writes type labels
  (dump `element` column, data `Atom Type Labels`, an XYZ `species="…"` key in
  type order), the XYZ `id` and `Selection` columns, and data-file image flags.
  - dump / bin: a 4-component property becomes `name[1] .. name[4]`, LAMMPS'
    spelling. A dump column has no type, so integer properties read back as
    real numbers with the same value (unless named like LAMMPS integers:
    `mol`, `i_*`). Binary dumps store doubles and carry no labels.
  - xyz: `name:R:k` / `name:I:k`, so components and integer-ness survive.
  - data: `q` and `mol` choose the atom style (`full`, `charge`,
    `molecular`), `diameter` with `density` gives `sphere` (or name the style
    with `--out-atom-style`, below), and `vx vy vz` become the `Velocities`
    section. Nothing else has a place in a data file.
- `--precision N | full` sets the positions and box: `N` decimals (the dump
  and XYZ default is 10), or `full`, the shortest decimal that reads back to
  the identical double. Data files default to exact already.
- `--prop-precision N | full` (needs `--props`) does the same for real-valued
  property columns; the default is `full`. Integers are never rounded.
- `--reduce-tilt` flips an over-tilted box the way LAMMPS does (`fix deform
  … flip yes`) until `|xy|, |xz| ≤ lx/2` and `|yz| ≤ ly/2` — along periodic
  axes only — and wraps the atoms into the flipped box, carrying image flags
  so unwrapped positions do not move. LAMMPS before 2023 refuses a data file
  with a larger tilt; `convert` to a data file says so when it sees one.
- `--out-atom-style STYLE` (data output only) fixes the `Atoms # STYLE`
  layout instead of letting the written properties pick it: `atomic`,
  `charge`, `full`, `molecular`, `bond`, `angle` or `sphere` (`bond` and
  `angle` rows are `molecular` rows; the file has no bonds). The style's `mol`,
  `q`, `diameter` and `density` columns are taken from the frame's properties
  of those names whether or not `--props` lists them, and a missing one is an
  error — nothing is invented. A listed property the style has no column for
  is an error (under `--props all` it is skipped and named); velocities go to
  `Velocities` as usual. So `rmda convert in.data -o out.data
  --out-atom-style molecular` drops the charges of a `full` file, and
  `--out-atom-style atomic` writes the default bytes. (`--atom-style` is the
  style of a data-file *input*.)

A box that is not LAMMPS-restricted (`a` along x, `b` in the xy plane — e.g. a
rotated extended-XYZ `Lattice`) is written to dump, binary and data files in
LAMMPS' general triclinic form, with the atoms as they are; LAMMPS (7Feb2024
and later), OVITO and this tool read it as the same cell. A left-handed cell
has no LAMMPS form and is refused.

Reading a general triclinic file (data `avec bvec cvec`, dump `BOX BOUNDS abc
origin`, binary) normally rotates it into LAMMPS' restricted frame, as LAMMPS
itself does: the cell, positions and velocity-like columns turn together, and
the rotation is not kept, so `convert` writes the restricted box.
**`--keep-general`** reads the box as written instead — cell rows `a b c` and
origin as in the file, positions and velocities in the file's frame — so
`rmda convert --keep-general in.data -o out.data` writes the same general box
back (at `--precision full`, the same bytes after a second pass), and any
analysis runs in the file's own frame. It changes nothing for a restricted box.

With `--props … --precision full` every format reads back to the same atoms:
identical position bits, ids, types and property values (within what the
format can carry, above); with `N` decimals, to exactly the printed numbers.
A data file holds one frame, so a window of more than one frame is refused at
the second frame. Compression runs on all threads (4 MB gzip members,
concatenated, which every gzip reader accepts) but is still the slow part:
about 7 s for a 350 MB dump on 12 cores, against under a second uncompressed.

### 3.16 `rmda index` — seek in a long trajectory

```console
$ rmda index traj.dump
indexed 2 frame(s), 16 atom rows -> traj.dump.rmdaidx
$ rmda info --frame 1 traj.dump
```

`rmda index FILE` scans the file once, reading frame headers only, and writes
`FILE.rmdaidx` next to it: one line per frame with its byte offset, line number,
timestep and atom count, under a header line recording the file's format, size
and modification time. Every later read then uses it automatically
to seek to `--frame k` / `--start k` instead of stepping over the k frames before
it, and parse errors still report the right line numbers. The index is ignored —
silently, falling back to a scan — as soon as the file's size or modification
time differs from what it recorded, so a rewritten trajectory is never read
through a stale index; rerun `rmda index` after rewriting a file.

Text dumps, XYZ files and binary dumps can be indexed; a data file has only
one frame. The directory holding the trajectory must be writable.

A gzipped file cannot seek to a frame, so for `FILE.gz` the index is a
different thing — a **gzip index** (binary, same name `FILE.gz.rmdaidx`, same
size-and-mtime rule):

```console
$ rmda index traj.dump.gz
indexed gzip: 167 access point(s), 1393370492 bytes decompressed -> traj.dump.gz.rmdaidx
```

It records an access point about every 8 MB of decompressed text (where the
compressed stream can be restarted, with the 32 KB of text before it), so later
reads inflate the spans between points on all threads instead of one. Building
it is one serial pass over the file that also checks every member's CRC
(about 6 s for a 565 MB `.gz` of 1.4 GB text); the index is about 0.4 % of the
compressed size. It pays off for single-member files (LAMMPS, `gzip`, `pigz`)
read more than once; a multi-member file (`rmda convert -o x.gz`, bgzip) is
read in parallel without one. Measured on the 5.76M-atom thin film, 12
threads (`rmda info`, wall time):

| input | serial (`--threads 1`) | parallel, no index | parallel, indexed |
|---|---|---|---|
| 4-frame dump, 1.39 GB text, one member (`pigz`), 565 MB | 9.6–12.1 s | serial | 3.0–3.5 s |
| same, written by `rmda convert -o x.dump.gz` (4 MB members) | 9.6 s | 3.7–4.4 s | 3.5 s |
| data file, 792 MB text, one member (`pigz`), 350 MB | 5.6–6.4 s | serial | 1.4–2.7 s |
| data file written by `rmda convert -o x.data.gz`, 170 MB | 3.1 s | 0.5–1.3 s | — |

(The same dump uncompressed reads in 1.7 s.)

### 3.17 The extended-XYZ output format

Every `.xyz` this tool writes — `-o` output, `rmda convert`, `rmda analyze`'s
per-frame files, and the defect marker files — uses the same header:

```
Lattice="ax ay az bx by bz cx cy cz" Origin="Ox Oy Oz" pbc="T T T" Properties=species:S:1:pos:R:3:type:I:1:... Timestep=0
X 0.6939335720 1.7873087691 0.6716875274 2
```

Details in that line worth knowing, because they exist to avoid losing
information on the way out:

- **The box origin is in `Origin=`.** `Lattice=` carries the nine cell-vector
  components, the standard form ASE accepts; the origin goes in OVITO's
  `Origin=` key, which OVITO reads as the cell origin and ASE keeps as an
  unknown key. This matters for overlay: a LAMMPS `data` box is routinely offset
  from the origin (`xlo` is wherever the simulation put it), and dropping that
  offset puts the written cell in the wrong place relative to the atom
  coordinates even though every atom is individually correct. On read, a
  twelve-number `Lattice=` (nine plus the origin) is also accepted, and
  `Origin=` wins if a file carries both.
- **`Timestep=`** carries the frame's timestep, so a frame written out of a
  trajectory can be placed back in it; it is read back, so `rmda defects` on
  an XYZ series reports the files' timesteps. A frame with a time (a dump's
  `ITEM: TIME`) also gets `Time=`.
- **There is a `type` column.** When an input carries no element labels — a
  multi-component LAMMPS `data` file, for instance — every atom's species reads
  `X`, and the symbol alone cannot say which component an atom is. The `type`
  column carries that identity, and reading a file back prefers it over inferring
  a type from the symbol. A file that declares its columns in a different order
  still reads.
- **A `Selection` column is read, if present.** Values must be 0 or 1; anything
  else is refused by name rather than coerced, since `0.5` is truthy and a
  per-atom weight wearing that name would quietly select everything. Absence
  stays absent rather than becoming an all-zero mask, so "no selection in the
  file" and "every atom deselected" remain distinguishable — which is what lets
  `rmda wc --only-selected` refuse a file that has no mask instead of analysing
  everything under a flag that says otherwise. `selection` in lower case is
  accepted too, because writers disagree about the spelling.

Round-tripping is exact: converting a file to `.xyz` and back reproduces it
byte for byte.

#### Choosing where the box starts

`--xyz-origin` picks which of the two you get:

| | `Lattice=` origin | atom coordinates |
|---|---|---|
| `--xyz-origin native` (default) | the input's | unchanged |
| `--xyz-origin zero` | `0 0 0` | translated by `-origin` |

```console
$ rmda ws --self --defects out/ --xyz-origin zero frame.data
```

The translation is not optional. Zeroing the origin while leaving absolute
coordinates alone would leave every atom that far from where its own cell says
it is — the same overlay error, in the other direction. In both modes an atom
sits at exactly the same offset inside its own cell; only the box's placement
changes. Use `native` unless something downstream insists a box begins at zero.

---

### 3.18 `rmda run` — a whole run from a config file

A config file is a command line written down, so that a run is a file you keep with its results
and rerun on a cluster or a workstation:

```toml
# results/ beside this file; relative paths are relative to the config
[input]
files = ["reference.data", "frame-*.data"]   # paths or * / ? patterns in the file name

[run]                       # global options (rmda --help): threads, atom_style, start, cell, pbc, ...
threads = 0

[defects]                   # one command section: that command's long options
region = ["z <= 500", "z > 500"]
reference = "auto"
out = "results"
defects_xyz = true
```

```console
rmda run thinfilm.toml
rmda run thinfilm.toml --dry-run                        # print the resolved rmda command, run nothing
rmda run thinfilm.toml --set defects.reference=local    # override a value (read as TOML, else a string)
srun rmda run polycrystal.toml --threads $SLURM_CPUS_PER_TASK
```

- **Keys** are the long options of the command (`rmda defects --help`), with `_` or `-`:
  `rmsd_cutoff = 0.15` is `--rmsd-cutoff 0.15`. The resolved command goes through the CLI's own
  parser, so defaults, checks and meanings are exactly the CLI's. A key the command has no option
  for is refused by name (`unknown key defects.rmsd_cutof`), as is a global option outside `[run]`
  and a command option inside it.
- **Values:** a string or number is the option's value; `true` sets a switch (`false` leaves it
  off); an array is one value per occurrence for a repeatable option (`region`) and the values of
  one occurrence for a multi-value one (`cell = [100, 100, 100]`).
- **Paths** in the config (input files, `out`, `ws_reference`, ...) are relative to the config
  file. Options given to `rmda run` itself win: `--threads`, `--profile`.
- **Series** (`out = "..."`) behave as `rmda defects --out` / `rmda void --out`: resume, the settings
  check (`run.txt`) and `overwrite`.
- **Report.** Every run writes `report.json` in its `out` directory (else `<config>.report.json`
  beside the config), also when it fails: `status` and `error`, `rmda_version`, the resolved
  `command`, the `config`, `host`, `threads`, `started_unix`, `wall_s`, `cpu_s` (all threads, Linux)
  and `parallel_efficiency` (`cpu_s / (wall_s × threads)`), `peak_rss_bytes`, the stage
  profile (`stages`: name, `thread_s` (seconds summed over threads, so a parallel stage exceeds
  `wall_s`), `calls`, `peak_anon_bytes`) and, for a series, the per-frame
  `summary.csv` rows (`frames`).

`examples/thinfilm.toml` and `examples/polycrystal.toml` are starting points for the two main
workflows.

**Sharding a series over a SLURM array.** `--shard I/N` (on `rmda defects ... --out DIR` and
`rmda void ... --out DIR`, or on
`rmda run`) cuts the ordered frames into N contiguous blocks and runs block I into
`DIR/shard-I-of-N/`; `rmda merge DIR` then joins the shards into `DIR`'s own tables and defect
files, exactly as one unsharded run writes them:

```bash
#SBATCH --array=0-7
srun rmda run poly.toml --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT --threads $SLURM_CPUS_PER_TASK
# after the array has finished:
rmda merge results
```

- With `--segment` the grains are tracked across the whole series, not per shard: each shard
  records, per frame, its grains' sizes and their atom overlaps with the previous frame's grains
  (`track.csv`; for its first frame it segments the frame before its block, PTM and segmentation
  only), and writes grain labels where stable ids go. `rmda merge` replays the tracker's decisions
  over all frames in order and writes the stable ids and `grain_events.csv` the sequential run
  would. A test checks the merged tables and defect files against an unsharded run byte for byte
  (2 and 3 shards, each on its own thread count).
- A shard resumes like any series. `rmda merge` refuses shards that are missing, unfinished
  (frames without a summary row), run with other settings, or not consecutive, and existing
  merged tables without `--overwrite`.
- Each shard's `report.json` stays in its directory.

**Resuming** a series (rerunning it into the same `--out`) skips the finished frames, drops the
rows an interrupted frame wrote before its summary row, and continues grain tracking with the
stable ids already written (the last finished frame is segmented again, and the tracker is seeded
from `grains.csv`).

### 3.19 `rmda explain` — size a run before submitting it

`rmda explain CONFIG [--set ...]` reads a run config as `rmda run` would and runs
nothing. It prints the resolved command, each input file with its size and atom
count (read from the file's first megabyte, inflated for `.gz`, so an 8 GB frame
costs a few milliseconds), the files a shard takes (marked `*`; the config's
`shard = "I/N"`, or `--set defects.shard=I/N`), and a memory
estimate per frame for `defects` and `void`, from measured runs (190–250 bytes per
atom on the 5.8M- and 128M-atom frames, plus about 0.3 GB): an upper figure for
sizing a job, not a prediction. A config the CLI would not take exits 2, as for
`rmda run`.

```console
$ rmda explain poly.toml --set defects.reference=local
command: rmda defects --out=results --reference=local --segment -- frame-1146.gz
inputs:  1 file(s)
  *        0    7963.4 MB 128000000 atoms  frame-1146.gz
memory:  ~24.6-32.3 GB peak per frame for 128000000 atoms (...)
```

### 3.20 `rmda bench` — per-stage throughput

`rmda bench [--atoms N] [--structure bcc|fcc] [--sigma Å] [--json FILE]` builds a
periodic crystal of about N atoms (default 1 000 000; `bcc` W by default, or `fcc`
Cu) in memory, with Gaussian thermal noise (`--sigma`, default 0.05 Å per component) and 0.1 % of its sites empty, and runs PTM, CNA, segmentation, the
defect census with the global and with the local reference, and the RDF on it,
printing wall time, atoms per second and peak memory per stage. Both censuses must
find exactly the planted vacancies, or the run fails (a fast wrong answer is not a
benchmark). Use it to compare machines, thread counts and builds; `--json` keeps
the numbers with the host and thread count. Work is split into chunks of 65 536
atoms, so a crystal smaller than that runs on one thread whatever `--threads` says.

## 4. Performance

Release build, 12 threads (i5-1235U laptop), 256,000-atom relaxed FCC alloy
frame (141.04 Å cube, `a = 3.5259 Å`, a 23 MB data file); best of three runs,
file read included.

| Command | Wall | Peak RSS |
|---|---|---|
| `rmda info` | 0.03 s | 57 MB |
| `rmda structure` | 0.47 s | 46 MB |
| `rmda ptm` | 1.6 s | 49 MB |
| `rmda census --cutoff 3.5` | 0.56 s | 46 MB |
| `rmda ws --reference ref ref` | 0.29 s | 65 MB |
| `rmda void --experimental --samples 40000` | 0.74 s | 47 MB |
| `rmda ws --self` | 38 s | 47 MB |
| `rmda cluster --cutoff 3.5` | 0.32 s | 46 MB |
| `rmda cluster --knear 12` | 0.90 s | 46 MB |
| `rmda rdf --rmax 5.0 --dr 0.05` | 1.3 s | 46 MB |
| `rmda wc --wc-shells 12` | 0.67 s | 47 MB |
| `rmda wc --wc-shells 12,18` | 0.90 s | 47 MB |
| `rmda grain-boundaries` | 5.8 s | 85 MB |

Two things to take from this. Method B's field is *cheap* — evaluating `{φ > 1+ε}`
indexes the same neighbour cell list as everything else, so a volume integral over
256k atoms costs less than a PTM pass; the expensive part of method B, if it were
ever to work, would be the strain fit. And `--self` is over 100× slower than
`--reference` because it runs an orientation descent over the sample to find a
lattice you may already have.

**`rmda segment` and `rmda defects`** were measured on larger frames (design.md, *Defect subsystem* and *Performance and
memory budgets*):
on the 5.76M-atom thin film, `rmda defects --region 'z <= 500' --region 'z > 500'`
takes 49 s on 12 threads, PTM included, at a peak RSS of 1.06 GB (≈ 184 bytes
per atom), and 178 s with `--reference local`; on the 42,432-atom HEA polycrystal
`rmda segment` takes 1.5 s and `rmda defects --segment` 5.7 s. `--profile` breaks a
run into `ptm`, `segment`, `defects.knn_build`, `defects.boundary`,
`defects.grain` (with `.prefit`, `.walk`, `.census`), `defects.patch` and
`defects.clusters`.

Results are bit-identical between thread counts; the analyses are deterministic by
construction (design.md, *Execution model and memory discipline*) and this is gated
by tests rather than assumed.

---

## 5. Python bindings

> **Status: built, tested** — `crates/rmda-py`, pyo3 0.25 with rust-numpy.
> `crates/rmda-cli/tests/python_parity.py` and
> `crates/rmda-cli/tests/python_bindings.rs` compare every function against the
> CLI and the core reader, field for field, including an ASE `Atoms` round trip.
> This section documents what runs.

Python wraps the library rather than reimplementing it, so a number computed in
Python is the number the CLI computes.

### Installing and importing

```console
$ pip install rmda                 # a wheel; or `pip install .` in a checkout
$ python -c 'import rmda; print(rmda.__version__)'
0.1.0
$ rmda --version
rmda 0.1.0
```

The package is the `rmda` Python package around the compiled extension
`rmda._rmda`, plus the `rmda` console script, which is the full command line of
§3. `pip install .` builds from source and needs a Rust toolchain (≥ 1.72). To
build a wheel to copy elsewhere:

```console
$ pip install maturin
$ maturin build --release
📦 Built wheel for abi3 Python ≥ 3.9 to target/wheels/rmda-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl
$ pip install target/wheels/rmda-0.1.0-cp39-abi3-*.whl
```

The wheel uses CPython's stable ABI (the root `pyproject.toml` turns on the
crate's `abi3` feature), so one file installs on every CPython ≥ 3.9; it depends on
numpy (`pip install 'rmda[ase]'` adds ASE). numpy is required at import; ASE only
for `from_ase`/`to_ase` and the `Atoms` inputs.

### Usage

```python
import rmda

f = rmda.read("run.xyz")              # format autodetected -> Frame (first frame)
f.natoms, f.ntypes, f.cell           # cell: (3, 3) float64, rows are the lattice vectors
f.positions, f.types, f.ids          # read-only zero-copy numpy views

f.run("ptm")                         # == rmda.ptm(f); attaches StructureType, ...
f["StructureType"]                   # attached per-atom property: a numpy copy
f.properties()                       # names of the attached properties
rmda.cna(f)                           # every analysis returns the CLI's scalars as a dict
rmda.rdf(f, rmax=8.0, nbins=400)      # + "histograms" and "edges"
rmda.warren_cowley(f, shells=[12, 18])
rmda.ws(f, reference="run.xyz")
f.write("out.xyz")                   # extended XYZ with the attached properties

# Per-grain defects, as `rmda segment` + `rmda defects`.
p = rmda.read("poly.xyz")
seg = rmda.segment(p)                 # attaches p["Grain"]; seg["grains"], seg["pairs"]
res = rmda.defects(p, grain_column="Grain")
res["defects_vacancies"], res["defects_grains_measured"]
g = res["grains"]                    # dict of numpy columns, one entry per grain
g["status"], g["vacancies"], g["t"].shape
res["vacancies"]["position"]         # (n, 3), wrapped into the box
rmda.defects(p, regions=["z <= 10", "z > 10"], segment=True, reference="global")
sorted(p.tables)                     # per-group tables of the calls run on p:
                                     # defects.grains, defects.vacancies, ..., segment.grains
gb = rmda.grain_boundaries(p)         # == rmda grain-boundaries; attaches GrainBoundary
p.tables["grains"]["GrainBoundaryCount"]   # per-grain seam counts, one row per grain id
# A series in Python: one tracker across frames keeps grain ids, as
# `rmda defects FILES --out --segment` does.
import glob
t = rmda.GrainTracker()
for path in sorted(glob.glob("run.*.dump")):
    fr = rmda.read(path)
    r = rmda.defects(fr, segment=True, tracker=t)
    r["grains"]["stable_grain"], r["grain_events"]["event"]
v = rmda.void(f, segment=True, tracker=rmda.GrainTracker())  # voids_table["stable_grain"]
rmda.void(f, experimental=True, nn_distance=2.553)   # == rmda void --experimental (method B)
rmda.ptm(f, order="voronoi", outputs=["F", "correspondences", "polar"])   # == --outputs

# A frame from arrays, with the checks a file read gets.
import numpy as np
q = rmda.Frame.from_arrays(f.positions, f.cell, pbc=f.pbc, origin=f.origin,
                           types=["Cu"] * f.natoms,          # names, or integer type ids
                           properties={"label": np.ones(f.natoms, dtype=np.int32)})
q.columns, q["label"]                 # input columns, as a file's extra columns
q["Grain"] = np.ones(q.natoms, dtype=np.int32)   # attach a column of your own...
rmda.defects(q, grain_column="Grain") # ...and count defects in those grains

# ASE Atoms in and out; nothing imports ASE unless to_ase is called.
import ase.io
at = ase.io.read("run.xyz")
same = rmda.read(at)                  # == rmda.from_ase(at)
rmda.rdf(at, rmax=6.0)                # every analysis takes an Atoms directly
rmda.ws(f, reference=at)              # a reference may be a path, a Frame or an Atoms
for frame in rmda.iread(ase.io.iread("traj.xyz")): pass
back = rmda.to_ase(f)                 # columns and attached props become atoms.arrays

pipe = rmda.Pipeline(rmda.iread("traj.xyz"))
pipe.add("ptm")
pipe.add("segment", {"min_size": 10}) # keyword arguments as a dict
for fr in pipe.run():                # a generator: one frame read and analysed per step
    fr["Grain"], fr.tables["segment.grains"]

rmda.set_threads(4); rmda.get_threads()
```

(This script runs as written; `run.xyz` and `traj.xyz` are FCC Cu frames,
`poly.xyz` the bicrystal of §3.5.)

### OVITO

`rmda.ovito` converts between frames and OVITO data collections. It needs the
`ovito` Python module (`pip install ovito`, free), or OVITO Pro's embedded
interpreter (`ovitos -m pip install rmda`); `import rmda` alone never imports
it.

```python
from ovito.io import import_file
import rmda, rmda.ovito

pipeline = import_file("run.xyz")
data = pipeline.compute()
f = rmda.ovito.from_ovito(data, types="names")   # an rmda.Frame
res = rmda.defects(f)
rmda.ovito.to_ovito(f, data, result=res)        # results into the same collection
data.attributes["defects_vacancies"]
data.tables["rmda.defects.grains"]["vacancies"]

# As a modifier in an OVITO pipeline (a plain function works in the free module):
def rmda_defects(frame, data):
    f = rmda.ovito.from_ovito(data, types="names")
    rmda.ovito.to_ovito(f, data, result=rmda.defects(f, segment=True))

pipeline.modifiers.append(rmda_defects)
pipeline.compute().attributes["defects_vacancies"]
```

- **`from_ovito(data, *, types="ids")`** carries positions, the cell (vectors,
  origin, periodicity), `Particle Identifier` and every other numeric
  particle property, as input columns (so a `Grain` property computed in
  OVITO feeds `grain_column="Grain"`). `types="ids"` keeps OVITO's numeric
  type ids and their names, which is what `rmda.read` gives for LAMMPS files;
  `types="names"` codes the named types 1.. in sorted order, which is what
  `rmda.read` gives for extended XYZ (OVITO numbers those in order of
  appearance). On the test fixtures the frame is field for field the one
  `rmda.read` gives. One exception: a LAMMPS file with a general triclinic box
  (`abc origin`, `avec/bvec/cvec`) keeps its orientation in OVITO, while
  `rmda.read` rotates it into restricted form as LAMMPS does. The structure is
  the same, so counts agree, but orientations differ by that rotation.
  OVITO's `Periodic Image` is not carried over.
- **`to_ovito(frame, data=None, *, result=None)`** writes into `data` (the
  same atoms in the same order, else `ValueError`): attached per-atom
  properties as particle properties, each `frame.tables` entry as a
  `DataTable` named `rmda.<table>` (a string column such as `status` as an
  integer column with named types; a `(G, 3, 3)` column as `(G, 9)`), and the
  numeric scalars of `result` as global attributes. Without `data` it builds a
  new collection holding the frame itself too: cell, positions, named types,
  identifiers and input columns.
- Each conversion copies the arrays once. Analysing a frame inside OVITO
  therefore holds it twice; for very large frames the command line is the
  better tool.

**Modifiers.** `rmda.ovito.SegmentModifier`, `DefectsModifier` and
`VoidModifier` are OVITO pipeline modifiers (OVITO 3.11 or newer) running
`rmda.segment`, `rmda.defects` and `rmda.void`, with the same options as
settings. In OVITO Pro, install the package into OVITO's interpreter (`ovitos
-m pip install rmda`, or *Install additional package* in the Python settings)
and they appear in the modifier list as *RMDA grain segmentation*, *RMDA
defect census* and *RMDA voids*. With the free `ovito` module they work from a
script:

```python
from ovito.io import import_file
from rmda.ovito import SegmentModifier, DefectsModifier

pipeline = import_file("frame.data")
pipeline.modifiers.append(SegmentModifier(min_size=50))
pipeline.modifiers.append(DefectsModifier(segment=False, grain_column="Grain"))
data = pipeline.compute()
data.attributes["defects_vacancies"], data.tables["rmda.defects.grains"]
```

| Modifier | Adds |
|---|---|
| `SegmentModifier` | particle properties `Grain`, `StructureType`; tables `rmda.segment.grains`, `rmda.segment.pairs`; `segment_*` attributes |
| `DefectsModifier` | tables `rmda.defects.grains`, `.vacancies`, `.interstitials`, `.clusters` (`.antisites`); `defects_*` attributes; with `output="atoms"` (default) the particle properties `Interstitial` (1 on interstitial atoms), `atom_grain`, `displacement`; with `output="defects"` the particles are replaced by the defects, as OVITO's Wigner–Seitz modifier does in its sites mode: vacancy sites and interstitial atoms, with `Defect Type` (Vacancy, Interstitial), `Grain`, `Defect Cluster` (the census's 0-based defect-cluster id), `Boundary` and `Atom Id` |
| `VoidModifier` | table `rmda.voids`; `void_*` attributes |

`DefectsModifier` takes its grains from `grain_column` (a particle property,
picked from a list in the GUI) or `regions` (`;`-separated expressions), and
segments them further when `segment` is on (the default). A file loaded into
`ws_reference` (`modifier.ws_reference.load(path)`) switches to OVITO's
explicit-reference Wigner–Seitz rule against that frame. `0` stands for
"automatic" in `boundary_width`, `cluster_cutoff` and `patch_size`, and
`threads=0` uses all cores. `type_coding` is `from_ovito`'s `types`. A
modifier recomputes when a setting changes, like any OVITO modifier; the
analysis itself runs to completion once started, so OVITO's cancel button
acts only between frames.

### Reference

| Function | CLI equivalent | Notes |
|---|---|---|
| `read(path_or_atoms)` | — | the first frame of a file, or an ASE `Atoms`; no frame window, no `--cell`/`--skip-columns` (every column is read) |
| `iread(path_or_atoms_or_iterable)` | — | a streaming iterator: over a file's frames, over an iterable of `Atoms` (`ase.io.iread(...)`), or the one frame of an `Atoms` |
| `from_ase(atoms)`, `to_ase(frame)` | — | explicit conversion. `from_ase` takes ids from `atoms.arrays["id"]` (else 1..N; duplicates are refused) and every other numeric array as an input column. `to_ase` maps type labels to atomic numbers where they are elements, keeps a non-zero origin in `atoms.info["cell_origin"]`, writes ids other than 1..N to `arrays["id"]`, puts every input column and attached property in `atoms.arrays`, and, when a type name is no element, stores the type ids in `arrays["type"]` and the names in `info["type_names"]`, which `from_ase` reads back |
| `Frame.from_arrays(positions, cell=None, *, pbc=None, origin=None, types=None, type_names=None, ids=None, properties=None)` | — | a frame from arrays, checked as a file read is checked (finite coordinates, a box with volume, unique ids, types ≥ 1). `cell` has one vector per row; `None` is the inferred, non-periodic box of a cell-less file; `pbc` defaults to all periodic. `types` are integer type ids (named by `type_names`, a list or a `{type: name}` dict) or names (coded 1.. in sorted order, as the extended-XYZ reader does); `None` makes every atom type 1. `properties` are input columns `{name: (N,) or (N, c) array}` |
| `Frame.columns`, `Frame.pbc`, `Frame.origin`, `Frame.type_names` | — | the input columns (a file's extra columns, `properties=`, an `Atoms`' extra arrays: what `grain_column=` and region expressions read), periodicity, cell origin, and `type_names[t - 1]`, the name of type `t` |
| `f[name]`, `f[name] = array`, `name in f` | — | `f[name]` is an attached property, else the input column of that name, as a typed copy. Assigning attaches a numeric `(N,)` or `(N, c)` column (its kind kept: bool, integer, float32, float64), which `grain_column=`, `f.write` and `to_ase` then see |
| `ptm(f, rmsd_cutoff=0.1, types=[], order="distance", outputs=[])` | `rmda ptm` | `types=[]` means the default `fcc,hcp,bcc`; `order` is `--ptm-order` (`"distance"` or `"voronoi"`), `outputs` is `--outputs` |
| `cna(f, mode="adaptive", cutoff=3.5, low=0.0, full=False)` | `rmda structure` | |
| `rdf(f, rmax, dr=None, nbins=None, partials=True, coordination=True, volume="auto", edge_correct=False)` | `rmda rdf` | give at most one of `dr` / `nbins` (`dr = rmax/nbins`); default `dr = 0.02`. The curves are in `["histograms"]`, the bin edges in `["edges"]` |
| `cluster(f, cutoff=None, knear=None, min_size=1, select=None, only_selected=False, sort_by_size=False, unwrap=False, properties=False, pair_cutoffs=None)` (alias `clusters`) | `rmda cluster` | exactly one of `cutoff` / `knear` / `pair_cutoffs` (`--pair-cutoff` as a dict `{(a, b): r}`, types by number or name; not available through `f.run`); `properties=True` is `--table`: the per-cluster `ClusterSize`, `ClusterMass`, `ClusterCenterOfMass`, `ClusterRadiusOfGyration`, `ClusterGyrationTensor`, `ClusterPercolates` go to `f.tables["clusters"]` (one row per cluster), `unwrap=True` attaches `UnwrappedPosition` |
| `warren_cowley(f, shells=None, edge_correct=False)` | `rmda wc` | `shells` as a list (`[12, 18]`) or a string (`"0,12"`); default `"0,12"` |
| `grain_boundaries(f, w_db=None, unassigned_is_boundary=False, min_grain_size=100, no_adopt_orphans=False)` | `rmda grain-boundaries` | PTM and grains into a fresh store as the CLI runs them; attaches `GrainBoundary` (and `Grain`, PTM's columns); returns the `gb_*` scalars plus the `grains` table (the `--grain-table` content by property name, with `GrainBoundaryCount`), also in `f.tables["grains"]` |
| `void(f, site_reference="auto", min_sites=2, cutoff=None, regions=[], grain_column=None, segment=False, tracker=None, experimental=False, lattice="fcc", nn_distance=2.86, reference=None, map="none", region=None, samples=40000, seed=11, epsilon=0.01, epsilons=None, strain=False, strain_spacing=None, strain_sigma=None, excess=False, thermal_window=False)` | `rmda void` | method A by default (§3.14): the void census on the `defects` census, returning the CLI's scalars, a `voids_table` dict of columns and `grain_events` (an unmeasured grain is a `UserWarning`, as on the CLI's stderr); `tracker=rmda.GrainTracker()`, passed frame after frame, keeps grain ids as a series does. With `experimental=True`, method B, whose arguments (`lattice` … `thermal_window`) are refused without it, as on the CLI; `region`/`epsilons` as lists or the CLI's comma strings. Method B returns as numbers what the CLI prints: `face_normals`, `circumradius`, `observed_nn`, `nn_drift`, `region`, `region_volume`, `void_volume` ± `void_standard_error` (`void_hits`, `void_samples`, `void_fraction`), with `epsilons` the `extrapolated_*` fit and per-epsilon lists, with `excess` the `excess_*` figures, with `strain` the `strain_*` mesh figures, with `thermal_window` (the CLI's `--thermal-window`) `face_gap`, `face_gap_bonds`, `window_volume` ± `window_standard_error`, `window_coefficients`, `window_residual` and per-epsilon lists. An nn mismatch over 10 % is a Python warning |
| `ws(f, reference=None, lattice="fcc", nn_distance=2.86, fit_self=False, affine_map="none", tolerance=None, cutoff=None, orientation=None, search_sample=100000, void_min_sites=1)` | `rmda ws` | `reference` is a path, a `Frame` or an `Atoms`; `fit_self` is `--self`; `affine_map` is `--map` (`"none"`, or `"reference"` for `to-reference`); with `reference`, the pure-WS census: `tolerance` only sets `ws_config_unmatched`, `cutoff` is ignored |
| `segment(f, threshold=5.0, core_angle=2.0, min_size=100, adopt=True, rmsd_cutoff=0.1, dhcp=False, mode="crystallite", ptm_order=None, dhcp_merge_faults=False)` | `rmda segment` | attaches `Grain` and `StructureType`; returns the `segment_*` scalars plus `grains` and `pairs` tables (dicts of numpy columns). `mode="auto"` is OVITO's algorithm (§2.8), `dhcp`/`dhcp_merge_faults` are `--dhcp`/`--dhcp-merge-faults`. Stacking faults are always merged and `min_bonds` is 4: `--stacking-faults` and `--min-bonds` have no Python spelling yet |
| `defects(f, regions=[], grain_column=None, reference="auto", segment=False, seg_threshold=5.0, seg_min_size=100, boundary_width=None, min_atoms=100, rmsd_cutoff=0.1, patch_size=None, cluster_cutoff=None, dhcp=False, seg_mode="crystallite", ptm_order=None, ws_reference=None, ws_affine="off", antisites=False, local_merge="union", tracker=None, resolve_ambiguous=False)` | `rmda defects` (one frame) | `grain_column` may name an attached property (the `Grain` that `segment` just attached) or an input column; `regions` and `grain_column` together are an error. Returns the `defects_*` scalars, `cluster_cutoff`, the `grains`, `vacancies`, `interstitials` and `clusters` tables, and per-atom `atom_grain` and `displacement` (NaN outside measured grains). With `segment=True` the `segment_*` scalars too. No `--seg-stacking-faults` (always `merge`). `ws_reference` (a path or a frame) and `ws_affine` are `--ws-reference` and `--ws-affine`; grains are then labelled on the reference, and the vacancy and interstitial tables carry `site_id`. `local_merge` and `resolve_ambiguous` are `--local-merge` and `--resolve-ambiguous`; `reference` takes every `--reference` value, `"periodic"` included. `tracker=rmda.GrainTracker()`, passed frame after frame with `segment=True`, keeps grain ids and events as a series does. `antisites=True` is `--antisites` (any reference, `local` included): the `defects_antisites*` scalars and the three antisite-cluster ones, grain columns `antisites`, `gb_antisites`, `sublattice_types`, an `antisites` table (`atom`, `id`, `type`, `site_type`, `position`, `site_position`, `grain`, `boundary`, `site_id`, `cluster`), and in `clusters` an `antisites` column with the antisite-only clusters as extra rows. Grains not measured (too small, not crystalline) raise a `UserWarning`, as the CLI warns on stderr: with none measured the zero counts are not a defect-free crystal |
| `set_threads(n)`, `get_threads()` | `--threads` | process-wide; `0` means `SLURM_CPUS_PER_TASK` under SLURM, else all cores |
| `Frame.run(name, **kwargs)` | — | any analysis by name: `ptm`, `cna`, `rdf`, `cluster`/`clusters`, `warren_cowley`/`wc`, `ws`, `segment`, `defects`, `grain_boundaries`, `void` |
| `Frame.tables` | `--table`, `--grain-table` | the per-group tables of the analyses run on the frame, as `{name: {column: numpy array}}` (fresh copies on each access): `grains` (per-grain properties of `grain_boundaries`), `clusters` (per-cluster properties of `cluster(..., properties=True)`), `segment.grains`/`segment.pairs`, `defects.grains`/`vacancies`/`interstitials`/`clusters`/`antisites`. A later call replaces its own tables whole. Per-group data are never per-atom columns, so they are not in `f.properties()` or `f.write` |
| `Frame.write(path, format="extxyz", props=None)` | `-o` | `props` selects attached properties by name; only `extxyz` is supported |
| `Pipeline(iread(...))`, `.add(name, kwargs=None)`, `.run()` | `rmda analyze` | steps run in the order added, each seeing the previous steps' properties; `add` takes its keyword arguments as one dict and returns `None`, so calls do not chain; `run` returns a generator that reads, analyses and yields one frame at a time (`list(pipe.run())` to collect) |

The `defects` `grains` table has the columns of the CLI's per-grain table
(§3.7), with `status_detail` (the `fit_failed` reason), `snapped` as an `(n, 3)`
bool array, `t` as `(n, 3, 3)` and `origin` as `(n, 3)`; counts of grains that
are not `measured` are `-1`, as in the CSV. A series in Python is a loop over
frames calling `defects` (or `void`) on each with one `tracker=rmda.GrainTracker()`,
which keeps grain ids and events as `rmda defects FILES --out --segment` does.

**Arrays.** `f.positions` (`(N, 3)` float64), `f.types` (uint32) and `f.ids`
(int64) are read-only views of the frame's own memory: free to hand out at any
atom count, and `np.array(f.positions)` makes a writable copy. Attached
properties come back as typed arrays (`StructureType` uint8, `Grain` int32,
multi-component properties as `(N, c)`), copied once per access.

**Threads and the GIL.** Every analysis releases the GIL while it runs, so other
Python threads keep running; the analysis itself runs on the pool
`set_threads` sizes. Errors surface as Python exceptions carrying the library's own
message, so a refusal reads the same in Python as on the command line.

**Open.** Wheels: `.github/workflows/release.yml` builds manylinux2014 abi3 wheels
(README, *Release artifacts*); none are published to PyPI. Python-side mutation of a frame's columns (`f.set`) is not offered; analyses run on the frame they are handed —
`iread` and `Pipeline` are how you walk a trajectory. See design.md, *Python
bindings*, for the intended surface.

---

## 6. Testing

```console
$ cargo test --workspace                                        # the fast suite
$ cargo test --workspace --release -- --include-ignored         # everything (about 1 h)
```

**The default suite** (`rmda-core`'s unit tests plus the integration and CLI
end-to-end binaries of both crates) runs in a few minutes and is what CI gates
every push on (`.github/workflows/ci.yml`). The dev profile optimises the numeric
core (`opt-level = 2` in the workspace `Cargo.toml`, debug assertions on), so the
end-to-end tests that count blocks of 10⁴–10⁵ atoms stay quick. `cargo test
--workspace` does not build the Python module (`rmda-py` has no Rust test
harness); the `python_bindings` tests skip, with the command to run, unless
`cargo build -p rmda-py` has produced a module newer than the binding source and
numpy is importable. CI builds it first.

**Expensive tests are `#[ignore]`d**, each with its reason, and run with
`--include-ignored` (in a release build: several take minutes even then). CI runs
them weekly and by hand (the `slow` job). They are:

| Ignored tests | What they confirm |
|---|---|
| `crates/rmda-cli/tests/ground_truth.rs` | the defect census against synthetic BCC, FCC, HCP and HEA blocks of 10⁵–10⁶ atoms with known loops, voids and point defects (needs python3 with numpy); the 3.5M-atom crowded-loop block |
| `crates/rmda-cli/tests/poly3.rs` | the three-lattice polycrystal series and self-image seams |
| `cna_2m_is_thread_count_independent`, `cna_scale_2m`, `cna_scale_16m` | CNA determinism and scale at 2M and 16M atoms |
| `emit_100k_xyz`, `emit_10m_xyz` | XYZ writer at 100k and 10M atoms (fixtures for OVITO captures and benchmarks) |
| `real_file_is_bit_identical_to_the_serial_reader` (`rmda-core` lib, `io::legacy`) | the parallel reader against the serial reference reader on a file named by `RMDA_EQUIV_FILE` |
| `write_oracle_file` (`rmda-core` lib, `io::binary`) | regenerates `tests/data/binary2txt_oracle.bin` into `RMDA_BIN_OUT` |
| the reference-data oracles (below) | OVITO agreement and defect counts on real MD snapshots |

Run one with, e.g., `cargo test --release -p rmda-core --test cna_scale -- --ignored cna_scale_2m`.

### Where the tests live

| Path | Covers |
|---|---|
| `crates/rmda-core/src/**` | unit tests, colocated with the code |
| `crates/rmda-core/src/analysis/grains.rs` | 58 grain-segmentation tests: disorientation, labelling, graph merging, cleanup, dissolution and orphan adoption |
| `crates/rmda-core/src/analysis/cluster.rs` | 24 connected-component tests: both modes, labelling, min-size, stats, selection, sorting, unwrapping, percolation, thread invariance, and OVITO 3.15 parity on synthetic frames (`tests/data/cluster_ovito_expected.txt`, regenerated by `tools/cluster_ovito_expected.py`) |
| `crates/rmda-core/src/graph.rs` | 8 shared union–find and canonical-labelling tests |
| `crates/rmda-core/src/analysis/grain_table.rs` | 15 per-grain CSV table tests: columns, `nan`, ragged rows, empty frames |
| `crates/rmda-cli/tests/cluster.rs` | `rmda cluster` end-to-end, all three modes, selection, table, unwrap, 20 tests |
| `crates/rmda-core/src/analysis/rdf.rs` | 11 RDF tests: exact lattice shell counts, `g = 1` on a uniform lattice, partial doubling, bin edges |
| `crates/rmda-cli/tests/rdf.rs` | `rmda rdf` end-to-end, 12 tests |
| `crates/rmda-core/src/analysis/warren_cowley.rs` | 24 Warren–Cowley tests: the textbook L1₂ matrix, hand-counted simple-cubic shells, rank semantics, selection, asymmetry, thread invariance, edge correction |
| `crates/rmda-cli/tests/wc.rs` | `rmda wc` end-to-end, 13 tests |
| `crates/rmda-core/src/analysis/grain_boundaries.rs` | 13 seam-detection tests: two rule variants that miss the seam (as negative controls), a one-atom grain, unassigned neighbours, slab thickness, refusals |
| `crates/rmda-core/tests/grain_boundaries_bicrystal.rs` | the seam on real 3D bicrystals: planarity, no holes, order-invariance, 7 tests |
| `crates/rmda-cli/tests/grain_boundaries.rs` | `rmda grain-boundaries` end-to-end, 9 tests |
| `crates/rmda-core/tests/grain_seg_validation.rs` | `rmda segment`'s algorithm (§2.8): single crystals near the fundamental-zone boundary, bicrystals, a Voronoi polycrystal, a stacking fault, a Σ3 twin, thread invariance, 6 tests |
| `crates/rmda-core/tests/defects_synthetic.rs` | the census (§2.9): planted vacancies and dumbbells, strained slabs, free clusters, a bicrystal, HCP, bent and graded crystals under patches, patch seams, thread invariance, 9 tests |
| `crates/rmda-cli/tests/defects_cmd.rs` | `rmda defects` end-to-end: regions, segmentation, patch vs global, series mode with resume and refusal, unnumbered baselines, stacking faults, 7 tests |
| `crates/rmda-cli/tests/poly3.rs` | the three-lattice (FCC/BCC/HCP) polycrystal over three frames: `segment` grain counts and labels, exact interior V/I per grain, `grain_events.csv` (ignored: expensive) |
| `crates/rmda-core/src/io/**` | readers and writers: binary dumps against LAMMPS `binary2txt`, data-file styles, type labels, velocities, column skipping, gzip streaming, the frame index, parallel vs serial parse bit-identity |
| `crates/rmda-core/tests/ws_self_reference.rs` | 256k-atom HEA defect counts |
| `crates/rmda-core/tests/void_b_benchmark.rs` | method B against exact void geometry |
| `crates/rmda-core/tests/void_b_reference.rs` | method B on the reference fixtures |
| `crates/rmda-core/tests/roto_strain_hea.rs` | the roto-strain gate of §2.7.1 on real relaxed alloys |
| `crates/rmda-core/tests/cna_golden.rs`, `ptm_oracle.rs` | OVITO agreement |
| `crates/rmda-core/tests/memory_budget.rs` | memory bounds at scale |
| `crates/rmda-cli/tests/ws.rs` | `rmda ws` end-to-end, 17 tests |
| `crates/rmda-cli/tests/void.rs` | `rmda void` end-to-end, 9 tests |
| `crates/rmda-cli/tests/analyze.rs` | `rmda analyze` end-to-end: multi-analysis output, the per-frame `-o` prefix files, refusals, 8 tests |
| `crates/rmda-cli/tests/defects.rs` | the `--defects` marker files: row semantics, cells, `void_id`, 14 tests |
| `crates/rmda-cli/tests/streaming.rs` | frame-at-a-time input: a trajectory is parsed one frame at a time, 6 tests |
| `crates/rmda-cli/tests/mic_parity.rs`, `mic_parity.py` | minimum-image and cell geometry against NumPy and a brute-force enumeration, skipping without ASE/numpy, 1 test |
| `crates/rmda-cli/tests/python_bindings.rs`, `python_parity.py` | the `rmda` Python module (§5): the Rust harness's surface checks plus the script's scalar/histogram/label parity against the CLI, 5 tests |

### Reference data

Some claims are checked against research data that is not distributed with the
code. The tests that use it are `#[ignore]`d and look for it in a directory named
by an environment variable; a test whose file is missing prints a skip line and
passes, so a checkout without the data stays green.

`RMDA_REFERENCE_DIR` (default: `reference_structures/` in the checkout), used by
`cna_reference`, `ptm_oracle`, `ptm_reference`, `grain_seg_ovito`,
`void_b_reference`, `roto_strain_hea` and `ws_self_reference`:

| File | What it is |
|---|---|
| `data.relaxed-40-HEA-reference` | clean relaxed FCC alloy, 256,000 atoms, no defects |
| `data.relaxed-40-HEA-lastframe` | the same alloy with planted point defects |
| `data.final-lastframe-{Al,W,Zr}-2000` | 2000 K elemental MD frames after cascades |
| `lastframe-HEA-poly.data` | HEA polycrystal, 42,432 atoms, 4 FCC grains, 39 % of atoms non-crystalline; the OVITO grain-segmentation parity fixture |

`RMDA_CASCADE_DIR`, used by `cascade_frames_count_as_periodic_single_crystals`
(`crates/rmda-cli/tests/ground_truth.rs`): a directory of `data.lastframe-*` LAMMPS
data files, each a periodic single crystal after massively overlapping cascades,
which `--reference periodic` must count with V − I exactly 0.

The OVITO per-atom expectations for the snapshots are captured with
`tools/capture_cna.py`, `tools/capture_ptm.py` and `tools/gen_ptm_extras.py`;
those for the small fixtures in `tests/data/` are committed.
