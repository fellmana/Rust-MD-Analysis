#!/usr/bin/env python3
"""Fixtures and OVITO (3.15 or later) oracle for PTM's opt-in outputs (design §14.2):
alloy ordering type, elastic deformation gradient F, interatomic distance
and the encoded template-to-neighbour correspondences.

Writes small binary-alloy and strained crystals to ``tests/data/ptm_extras/``
(deterministic: fixed seeds; noise plus a few percent of swapped A/B pairs)
and OVITO's per-atom ``Structure Type``, ``Ordering Type``,
``Elastic Deformation Gradient``, ``Interatomic Distance`` and
``Correspondences`` into
``tests/data/ptm_extras_expected.txt.gz`` (gzipped text, mtime 0 so the
output is reproducible), which
``crates/rmda-core/tests/ptm_extras_ovito.rs`` compares atom for atom (with
``NeighbourOrder::Voronoi``, OVITO's neighbour order).

With ``--snapshots`` it instead captures the same columns for the four
``reference_structures/`` snapshots into ``target/refcheck/<name>.ptmx``
(binary; not committed), read by ``ptm_oracle.rs``.

F and the distance are written to 13 significant digits (the test compares
them to 1e-12 relative; OVITO and rmda agree to ~1e-15), the correspondences
as 16 hex digits of the int64 bit pattern.

F is written as upstream's row-major ``F[3 i + j]``, which is rmda's
``StructureDeformationGradient``. OVITO up to 3.16 stores exactly those nine
values (labelled ``XX YX ZX ...``, a column-major reading: the matrix it labels
is F^T); 3.17 stores their transpose under the same labels, which this script
undoes.

Usage:  python3 tools/gen_ptm_extras.py [--snapshots]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_structure_variants import BCC, DIA, FCC, hex_ortho, lattice, noisy  # noqa: E402

ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "tests", "data", "ptm_extras")
EXPECTED = os.path.join(ROOT, "tests", "data", "ptm_extras_expected.txt.gz")

DEFAULT = (1, 2, 3)
TWO_SHELL = (1, 2, 3, 6, 7, 8)


def write(name, cell, pos, species):
    path = os.path.join(OUT, name + ".xyz")
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        lat = " ".join(f"{v:.10f}" for v in np.asarray(cell).reshape(-1))
        f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for s, x in zip(species, pos):
            f.write(f"{s} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}\n")
    return path


def swap(species, frac, seed):
    """Swap `frac` of the A atoms with as many B atoms (antisites)."""
    sp = np.array(species)
    rng = np.random.default_rng(seed)
    a = np.flatnonzero(sp == "A")
    b = np.flatnonzero(sp == "B")
    k = max(1, int(frac * min(len(a), len(b))))
    ia = rng.choice(a, k, replace=False)
    ib = rng.choice(b, k, replace=False)
    sp[ia], sp[ib] = "B", "A"
    return list(sp)


def labelled(cell, basis, reps, labels):
    cell_out, pos = lattice(cell, basis, reps)
    n = len(basis)
    return cell_out, pos, [labels[i % n] for i in range(len(pos))]


def strain(cell, pos, F):
    F = np.asarray(F, float)
    return cell @ F.T, pos @ F.T


def fixtures():
    """(name, cell, positions, species, enabled structures)."""
    out = []
    # B2 (CsCl).
    cell, pos, sp = labelled(np.eye(3) * 2.87, BCC, (5, 5, 5), "AB")
    out.append(("b2_t10", cell, noisy(pos, 0.10, 21), swap(sp, 0.04, 31), DEFAULT))
    # L1_2 (Cu3Au): B on the corners, A on the faces.
    cell, pos, sp = labelled(np.eye(3) * 3.75, FCC, (4, 4, 4), "BAAA")
    out.append(("l12_t10", cell, noisy(pos, 0.10, 22), swap(sp, 0.04, 32), DEFAULT))
    # L1_0 (CuAu): alternating (001) layers, slightly tetragonal.
    cell, pos, sp = labelled(np.diag([3.9, 3.9, 3.7]), FCC, (4, 4, 4), "AABB")
    out.append(("l10_t08", cell, noisy(pos, 0.08, 23), swap(sp, 0.04, 33), DEFAULT))
    # Strained, rotated pure FCC and HCP: F away from the identity, and HCP's
    # alternative template.
    Fs = [[1.03, 0.02, -0.01], [0.00, 0.98, 0.03], [0.01, -0.02, 1.01]]
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]])
    cell, pos = lattice(np.eye(3) * 3.615, FCC, (4, 4, 4))
    cell, pos = strain(cell, pos, R @ np.array(Fs))
    out.append(("fcc_strained_t05", cell, noisy(pos, 0.05, 24), ["A"] * len(pos), DEFAULT))
    a = 3.2
    hc, hf = hex_ortho(a, a * np.sqrt(8 / 3), [[0, 0, 0], [1 / 3, 2 / 3, 0.5]])
    cell, pos = lattice(hc, hf, (5, 3, 3))
    cell, pos = strain(cell, pos, R @ np.array(Fs))
    out.append(("hcp_strained_t08", cell, noisy(pos, 0.08, 25), ["A"] * len(pos), DEFAULT))
    # Zincblende (SiC), wurtzite and hexagonal boron nitride.
    cell, pos, sp = labelled(np.eye(3) * 4.36, DIA, (3, 3, 3), "AAAABBBB")
    out.append(("zincblende_t08", cell, noisy(pos, 0.08, 26), swap(sp, 0.03, 36), TWO_SHELL))
    a = 4.36 / np.sqrt(2)
    hc, hf = hex_ortho(a, a * np.sqrt(8 / 3),
                       [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.375], [2 / 3, 1 / 3, 0.875]])
    cell, pos = lattice(hc, hf, (4, 3, 3))
    # hex_ortho lays out each hexagonal basis atom twice (two sublattice shifts).
    sp = ["A", "A", "A", "A", "B", "B", "B", "B"] * (len(pos) // 8)
    out.append(("wurtzite_t08", cell, noisy(pos, 0.08, 27), swap(sp, 0.03, 37), TWO_SHELL))
    a = 1.45 * np.sqrt(3)
    gc, gf = hex_ortho(a, 6.66, [[0, 0, 0.25], [1 / 3, 2 / 3, 0.25], [0, 0, 0.75], [2 / 3, 1 / 3, 0.75]])
    cell, pos = lattice(gc, gf, (5, 3, 2))
    sp = ["A", "A", "B", "B", "B", "B", "A", "A"] * (len(pos) // 8)
    out.append(("hbn_t05", cell, noisy(pos, 0.05, 28), swap(sp, 0.03, 38), TWO_SHELL))
    return out


def ovito_columns(path, enabled):
    from ovito.io import import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as P
    pipe = import_file(path)
    m = P()
    for s in m.structures:
        if s.id != 0:
            s.enabled = s.id in enabled
    m.output_ordering = True
    m.output_deformation_gradient = True
    m.output_interatomic_distance = True
    m.output_rmsd = True
    m.output_orientation = True  # OVITO writes Correspondences with orientations
    pipe.modifiers.append(m)
    p = pipe.compute().particles
    return (np.array(p["Structure Type"], int), np.array(p["Ordering Type"], int),
            deformation_gradient(p),
            np.array(p["Interatomic Distance"], float),
            np.array(p["Correspondences"], np.int64))


def deformation_gradient(particles):
    """Upstream's row-major F[3 i + j] for every atom. OVITO up to 3.16 stores
    exactly that; 3.17 stores its transpose (under the same component labels)."""
    import ovito

    f = np.array(particles["Elastic Deformation Gradient"], float)
    if ovito.version >= (3, 17, 0):
        f = f.reshape(-1, 3, 3).transpose(0, 2, 1).reshape(-1, 9)
    return f


def main_fixtures():
    os.makedirs(OUT, exist_ok=True)
    import ovito
    lines = [
        "# generated by tools/gen_ptm_extras.py (OVITO %s)" % ".".join(map(str, ovito.version)),
        "# <fixture> structures <ids>",
        "# <fixture> types|ordering <one digit per atom>",
        "# <fixture> F <9 per atom, row-major F_ij>  |  ia <1 per atom>  (%.13g)",
        "# <fixture> corr <1 per atom: Correspondences int64 as 16 hex digits>",
    ]
    for name, cell, pos, sp, enabled in fixtures():
        path = write(name, cell, pos, sp)
        st, od, F, ia, corr = ovito_columns(path, enabled)
        lines.append(f"{name} structures " + ",".join(map(str, enabled)))
        lines.append(f"{name} types " + "".join(map(str, st)))
        lines.append(f"{name} ordering " + "".join(map(str, od)))
        lines.append(f"{name} F " + " ".join(f"{v:.13g}" for v in F.reshape(-1)))
        lines.append(f"{name} ia " + " ".join(f"{v:.13g}" for v in ia))
        lines.append(f"{name} corr " + " ".join(f"{int(v) & (2**64 - 1):016x}" for v in corr))
        print(name, len(pos), "types", np.bincount(st).tolist(), "ordering", np.bincount(od).tolist())
    import gzip
    with open(EXPECTED, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as f:
        f.write(("\n".join(lines) + "\n").encode())
    print("wrote", EXPECTED)


def main_snapshots():
    names = ["data.final-lastframe-Al-2000", "data.final-lastframe-W-2000",
             "data.final-lastframe-Zr-2000", "lastframe-HEA-poly.data"]
    dst = os.path.join(ROOT, "target", "refcheck")
    os.makedirs(dst, exist_ok=True)
    for name in names:
        src = os.path.join(ROOT, "reference_structures", name)
        if not os.path.exists(src):
            print("skip", name)
            continue
        st, od, F, ia, corr = ovito_columns(src, DEFAULT)
        # u8 types, u8 ordering, then f64 F (9 per atom), f64 ia and i64
        # Correspondences, little endian.
        with open(os.path.join(dst, name + ".ptmx"), "wb") as f:
            np.uint64(len(st)).tofile(f)
            st.astype("<u1").tofile(f)
            od.astype("<u1").tofile(f)
            F.astype("<f8").tofile(f)
            ia.astype("<f8").tofile(f)
            corr.astype("<i8").tofile(f)
        print(name, len(st), "ordering", np.bincount(od).tolist())


if __name__ == "__main__":
    if "--snapshots" in sys.argv:
        main_snapshots()
    else:
        main_fixtures()
