#!/usr/bin/env python3
"""Generate the golden fixtures with ASE and OVITO.

Run once; the outputs under ``tests/data`` are committed so that
``cargo test`` needs neither ASE nor OVITO installed.  Re-run this script when
the fixture set changes and diff the results in review.

Usage:  python3 tools/gen_fixtures.py

Note: this covers the small fixtures only.  The 108k-atom CNA
differential block is deliberately *not* committed; it is rebuilt by the Rust
test at run time and only the OVITO per-atom codes are stored.  See
``tools/capture_100k.py``.
"""

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
os.makedirs(DATA, exist_ok=True)

from ase import Atoms
from ase.build import bulk


def ovito_version():
    """The running OVITO's version, for the generated files' headers."""
    import ovito

    return ".".join(map(str, ovito.version))

CUTOFF = 3.5

# Structure codes shared with OVITO (CommonNeighborAnalysisModifier.StructureType
# and PolyhedralTemplateMatchingModifier.StructureType).  Kept in one place so
# the CNA and PTM golden files cannot drift apart.
ST_OTHER, ST_FCC, ST_HCP, ST_BCC, ST_ICO, ST_SC = 0, 1, 2, 3, 4, 5

M2_FIXTURES = (
    "fcc_ortho.xyz",
    "hcp_ortho.xyz",
    "m2_bcc.xyz",
    "m2_ico.xyz",
    "m2_sc.xyz",
    "m2_fcc_thermal.xyz",
    "triclinic_two_species.xyz",
    "cluster_nonperiodic.xyz",
)


def _icosahedron():
    """A 12-atom icosahedron plus its centre, in a large vacuum cell.

    Only the *centre* atom is a complete icosahedral environment; the 12 shell
    atoms are surface atoms whose first shell is truncated, so OVITO correctly
    calls them OTHER.  That is exactly what we want to pin down: bulk ICO
    cannot be tested in a periodic cell at all, because the icosahedral phase
    is quasiperiodic (no finite unit cell), so a finite cluster is the only
    option and it exercises the 555 signature and the OTHER class together.
    """
    phi = (1.0 + 5.0**0.5) / 2.0
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s2 * phi, 0, s1]]
    radius = 2.7
    pos = [[v[0] * radius, v[1] * radius, v[2] * radius] for v in verts]
    pos.append([0.0, 0.0, 0.0])
    cell = np.eye(3) * 22.0
    return Atoms("Al13", positions=pos, cell=cell, pbc=True)


def _thermalise(kind, sigma, seed):
    """Perfect bulk + Gaussian displacement noise, to leave the ideal class."""
    rng = np.random.RandomState(seed)
    if kind == "fcc":
        at = bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2))
    else:
        raise ValueError(kind)
    at.positions += rng.normal(0.0, sigma, at.positions.shape)
    return at


def save(name, atoms):
    p = os.path.join(DATA, name)
    # Newer ASE gives bulk("Fe") initial magnetic moments; the fixtures are
    # species and positions only.
    atoms.arrays.pop("initial_magmoms", None)
    atoms.write(p, format="extxyz")
    print("wrote", name, len(atoms), "atoms")


def main():
    rng = np.random.RandomState(20240101)

    # 1. orthorhombic FCC copper, 2x2x2 conventional cells
    save("fcc_ortho.xyz", bulk("Cu", "fcc", a=3.615, cubic=True).repeat((2, 2, 2)))

    # 2. hexagonal HCP magnesium, 2x2x2 (non-orthogonal a/b)
    save("hcp_ortho.xyz", bulk("Mg", "hcp", a=3.209, c=5.211).repeat((2, 2, 2)))

    # 3. general triclinic cell with two species and a vacancy
    n = 12
    cell = np.array([[4.0, 1.2, 0.3], [0.0, 4.2, 0.5], [0.0, 0.0, 5.0]])
    frac = rng.rand(n, 3)
    pos = frac @ cell
    syms = ["Cu", "Al"]
    at = Atoms(symbols=[syms[i % 2] for i in range(n)], positions=pos, cell=cell, pbc=True)
    del at[3]  # one vacancy so neighbour counts are not uniform
    save("triclinic_two_species.xyz", at)

    # 4. non-periodic cluster (PBC off) - exercises the non-periodic stencil
    cl = Atoms("Cu8", positions=rng.rand(8, 3) * 6.0, cell=[12.0, 12.0, 12.0], pbc=False)
    save("cluster_nonperiodic.xyz", cl)

    # ---------------------------------------------------------------------
    # Structure-ID fixtures. One ideal bulk of every lattice CNA must
    # distinguish, plus two perturbed configurations so the adaptive cutoff
    # and the OTHER class are actually exercised.
    # ---------------------------------------------------------------------
    save("m2_bcc.xyz", bulk("Fe", "bcc", a=2.866, cubic=True).repeat((2, 2, 2)))
    save("m2_ico.xyz", _icosahedron())
    save("m2_sc.xyz", bulk("Po", "sc", a=2.98, cubic=True).repeat((2, 2, 2)))
    save("m2_fcc_thermal.xyz", _thermalise("fcc", 0.15, 42))

    # OVITO-side expectations: per-atom coordination at CUTOFF, and for the
    # triclinic case also the cell matrix as OVITO understands it.
    from ovito.io import import_file, export_file
    from ovito.modifiers import CoordinationNumberModifier

    expected = {}
    for name in (
        "fcc_ortho.xyz",
        "hcp_ortho.xyz",
        "triclinic_two_species.xyz",
        "cluster_nonperiodic.xyz",
    ):
        path = os.path.join(DATA, name)
        pipe = import_file(path)
        # OVITO's coordination number is the full PBC-aware neighbour count at
        # the given cutoff - exactly what rmda's full neighbour list must match.
        pipe.modifiers.append(CoordinationNumberModifier(cutoff=CUTOFF))
        data = pipe.compute()
        coord = np.array(data.particles["Coordination"], dtype=np.int64)
        # OVITO stores the cell with vectors as COLUMNS; rmda (like ASE) uses
        # rows, so transpose here.
        cellmat = np.array(data.cell.matrix, dtype=float)[:, :3].T
        pbc = np.array(
            [data.cell.pbc_x, data.cell.pbc_y, data.cell.pbc_z], dtype=bool
        )
        expected[name] = {
            "cutoff": CUTOFF,
            "natoms": int(data.particles.count),
            "coordination": coord.tolist(),
            "cell_rows": cellmat.tolist(),
            "pbc": pbc.tolist(),
        }
        print(f"{name}: natoms={data.particles.count} coord[:8]={coord[:8].tolist()}")

        # Round-trip the same structure through OVITO's LAMMPS writers so the
        # dump/data readers are validated against real-world files.
        dump_columns = [
            "Particle Identifier",
            "Particle Type",
            "Position.X",
            "Position.Y",
            "Position.Z",
        ]
        if name == "fcc_ortho.xyz":
            export_file(
                pipe,
                os.path.join(DATA, "fcc_ortho.dump"),
                "lammps/dump",
                columns=dump_columns,
            )
            export_file(pipe, os.path.join(DATA, "fcc_ortho.data"), "lammps/data")
            print("wrote fcc_ortho.dump, fcc_ortho.data (via OVITO)")
        if name == "triclinic_two_species.xyz":
            export_file(
                pipe,
                os.path.join(DATA, "triclinic.dump"),
                "lammps/dump",
                columns=dump_columns,
            )
            export_file(pipe, os.path.join(DATA, "triclinic.data"), "lammps/data")
            print("wrote triclinic.dump, triclinic.data (via OVITO)")

    # OVITO's own read-back of the LAMMPS files it just wrote: the rmda readers
    # must agree with these numbers.
    for name in ("fcc_ortho.dump", "fcc_ortho.data", "triclinic.dump", "triclinic.data"):
        path = os.path.join(DATA, name)
        pipe = import_file(path)
        pipe.modifiers.append(CoordinationNumberModifier(cutoff=CUTOFF))
        data = pipe.compute()
        expected[name] = {
            "cutoff": CUTOFF,
            "natoms": int(data.particles.count),
            "coordination": np.array(
                data.particles["Coordination"], dtype=np.int64
            ).tolist(),
            "cell_rows": np.array(data.cell.matrix, dtype=float)[:, :3].T.tolist(),
            "pbc": [
                bool(data.cell.pbc_x),
                bool(data.cell.pbc_y),
                bool(data.cell.pbc_z),
            ],
        }
        print(f"{name}: natoms={data.particles.count} (OVITO read-back)")

    # Flat, line-oriented expectation file so the Rust test needs no JSON
    # dependency to read it.
    out = []
    for name, e in sorted(expected.items()):
        out.append(f"fixture {name}")
        out.append(f"cutoff {e['cutoff']!r}")
        out.append(f"natoms {e['natoms']}")
        out.append("pbc " + " ".join("T" if p else "F" for p in e["pbc"]))
        out.append("cell")
        for row in e["cell_rows"]:
            out.append("  " + " ".join(repr(float(v)) for v in row))
        out.append("coord " + " ".join(str(c) for c in e["coordination"]))
        out.append("")
    with open(os.path.join(DATA, "ovito_expected.txt"), "w") as fh:
        fh.write(f"# generated by tools/gen_fixtures.py (OVITO {ovito_version()})\n")
        fh.write("\n".join(out))
    print("wrote ovito_expected.txt")

    capture_structure_ids()


def capture_structure_ids():
    """OVITO ground truth for the structure-ID engines (design §13, §14).

    CNA is run in the two modes rmda implements natively (adaptive and fixed);
    PTM with its default structure set.  Per-atom type codes are written for
    every fixture so a difference is localised to the offending atom rather
    than to a histogram.
    """
    from ovito.io import import_file
    from ovito.modifiers import (
        CommonNeighborAnalysisModifier as CNA,
        PolyhedralTemplateMatchingModifier as PTM,
    )

    fixed_cutoff = 3.5
    out = [
        f"# generated by tools/gen_fixtures.py (OVITO {ovito_version()})",
        "# codes: 0=OTHER 1=FCC 2=HCP 3=BCC 4=ICO 5=SC 6=DCUB 7=DHEX 8=GRAPHENE",
    ]
    for name in M2_FIXTURES:
        path = os.path.join(DATA, name)
        block = [f"fixture {name}"]
        hist = {}

        for label, mode in (("adaptive", CNA.Mode.AdaptiveCutoff), ("fixed", CNA.Mode.FixedCutoff)):
            pipe = import_file(path)
            mod = CNA(mode=mode)
            if mode == CNA.Mode.FixedCutoff:
                mod.cutoff = fixed_cutoff
            pipe.modifiers.append(mod)
            st = np.array(pipe.compute().particles["Structure Type"], dtype=np.int64)
            hist[label] = st
            cut = fixed_cutoff if mode == CNA.Mode.FixedCutoff else 0.0
            block.append(f"cna_{label}_cutoff {cut!r}")
            block.append("cna_" + label + " " + " ".join(str(v) for v in st.tolist()))

        pipe = import_file(path)
        pipe.modifiers.append(PTM(output_rmsd=True, output_interatomic_distance=True))
        data = pipe.compute()
        ptm = np.array(data.particles["Structure Type"], dtype=np.int64)
        rmsd = np.array(data.particles["RMSD"], dtype=float)
        dist = np.array(data.particles["Interatomic Distance"], dtype=float)
        block.append("ptm " + " ".join(str(v) for v in ptm.tolist()))
        block.append("ptm_rmsd " + " ".join(f"{v:.6f}" for v in rmsd.tolist()))
        block.append("ptm_distance " + " ".join(f"{v:.6f}" for v in dist.tolist()))
        out.extend(block)
        out.append("")

        def summarise(v):
            u, c = np.unique(v, return_counts=True)
            return ",".join(f"{int(a)}x{int(b)}" for a, b in zip(u, c))

        print(
            f"{name:26s} cna={summarise(hist['adaptive']):20s}"
            f" cna_fixed={summarise(hist['fixed']):20s} ptm={summarise(ptm)}"
        )

    with open(os.path.join(DATA, "structure_expected.txt"), "w") as fh:
        fh.write("\n".join(out))
    print("wrote structure_expected.txt")


if __name__ == "__main__":
    main()
