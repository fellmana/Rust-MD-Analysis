#!/usr/bin/env python3
"""Fixtures and OVITO 3.15 oracle for the PTM two-shell structures and the CNA
variants (design §14, §13).

Writes small thermal crystals to ``tests/data/variants/*.xyz`` (deterministic:
fixed seeds) and, per fixture, OVITO's per-atom labels for

* ``ptm6``      PTM with FCC, HCP, BCC, cubic/hexagonal diamond and graphene
                enabled (OVITO orders neighbours by Voronoi solid angle, so rmda
                is compared with ``--ptm-order voronoi``);
* ``ptm8``      PTM with all eight structures;
* ``interval``  CNA, ``IntervalCutoff``;
* ``fixed``     CNA, ``FixedCutoff`` at the fixture's cutoff;
* ``bond``      CNA, ``BondBased`` on ``CreateBondsModifier(cutoff)``;
* ``diamond``   ``IdentifyDiamondModifier``.

into ``tests/data/structure_variants_expected.txt``, which
``crates/rmda-core/tests/structure_variants_ovito.rs`` compares atom for atom.

Usage:  python3 tools/gen_structure_variants.py
"""
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "tests", "data", "variants")
EXPECTED = os.path.join(ROOT, "tests", "data", "structure_variants_expected.txt")

DIA = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
       [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]]
FCC = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
BCC = [[0, 0, 0], [.5, .5, .5]]


def write(name, cell, pos, pbc=(True, True, True)):
    path = os.path.join(OUT, name + ".xyz")
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        lat = " ".join(f"{v:.10f}" for v in np.asarray(cell).reshape(-1))
        p = " ".join("T" if b else "F" for b in pbc)
        f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="{p}"\n')
        for x in pos:
            f.write(f"X {x[0]:.10f} {x[1]:.10f} {x[2]:.10f}\n")
    return path


def lattice(cell, basis, reps):
    cell = np.asarray(cell, float)
    pts = [(np.array([i, j, k]) + b) @ cell
           for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])
           for b in np.asarray(basis, float)]
    return cell * np.array(reps)[:, None], np.array(pts)


def hex_ortho(a, c, basis_hex):
    """Orthorhombic cell (a, a*sqrt3, c) holding two hexagonal cells."""
    cell = np.diag([a, a * np.sqrt(3), c])
    pts = []
    for b in basis_hex:
        for sx, sy in [(0, 0), (0.5, 0.5)]:
            x = b[0] * a - 0.5 * b[1] * a + sx * a
            y = b[1] * a * np.sqrt(3) / 2 + sy * a * np.sqrt(3)
            pts.append([x, y, b[2] * c])
    return cell, (np.array(pts) @ np.linalg.inv(cell)) % 1.0


def noisy(pos, sigma, seed):
    return pos + np.random.default_rng(seed).normal(0.0, sigma, pos.shape)


def fixtures():
    """(name, cell, positions, pbc, cna cutoff)."""
    out = []
    a_si = 5.431
    cell, pos = lattice(np.eye(3) * a_si, DIA, (3, 3, 3))
    out.append(("dcub_si_t20", cell, noisy(pos, 0.20, 11), (True,) * 3, 2.8))
    out.append(("dcub_si_t10", cell, noisy(pos, 0.10, 12), (True,) * 3, 2.8))
    keep = np.ones(len(pos), bool)
    keep[[0, 37, 101]] = False
    cell2 = cell.copy()
    cell2[2, 2] += 12.0
    out.append(("dcub_si_vac_surface", cell2, noisy(pos[keep], 0.05, 13), (True,) * 3, 2.8))
    a = a_si / np.sqrt(2)
    hc, hf = hex_ortho(a, a * np.sqrt(8 / 3),
                       [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.375], [2 / 3, 1 / 3, 0.875]])
    cell, pos = lattice(hc, hf, (4, 3, 3))
    out.append(("dhex_si_t15", cell, noisy(pos, 0.15, 14), (True,) * 3, 2.8))
    a = 1.42 * np.sqrt(3)
    gc, gf = hex_ortho(a, 6.70, [[0, 0, 0.25], [1 / 3, 2 / 3, 0.25], [0, 0, 0.75], [2 / 3, 1 / 3, 0.75]])
    cell, pos = lattice(gc, gf, (5, 3, 2))
    out.append(("graphite_t08", cell, noisy(pos, 0.08, 15), (True,) * 3, 1.8))
    cell, pos = lattice(np.eye(3) * 3.615, FCC, (4, 4, 4))
    out.append(("fcc_t15", cell, noisy(pos, 0.15, 16), (True,) * 3, 3.0))
    cell, pos = lattice(np.eye(3) * 2.866, BCC, (5, 5, 5))
    out.append(("bcc_t15", cell, noisy(pos, 0.15, 17), (True,) * 3, 3.4))
    a = 3.2
    hc, hf = hex_ortho(a, a * np.sqrt(8 / 3), [[0, 0, 0], [1 / 3, 2 / 3, 0.5]])
    cell, pos = lattice(hc, hf, (5, 3, 3))
    out.append(("hcp_t15", cell, noisy(pos, 0.15, 18), (True,) * 3, 3.8))
    from ase.cluster import Icosahedron
    p = Icosahedron("Cu", noshells=4).get_positions()
    p = p - p.mean(axis=0) + 20.0
    out.append(("ico_t05", np.eye(3) * 40.0, noisy(p, 0.05, 19), (False,) * 3, 3.0))
    return out


def ovito_labels(path, kind, cutoff):
    from ovito.io import import_file
    from ovito.modifiers import (CommonNeighborAnalysisModifier as C, CreateBondsModifier as B,
                                 IdentifyDiamondModifier as D,
                                 PolyhedralTemplateMatchingModifier as P)
    pipe = import_file(path)
    if kind in ("ptm6", "ptm8"):
        m = P()
        want = {1, 2, 3, 6, 7, 8} if kind == "ptm6" else set(range(1, 9))
        for s in m.structures:
            if s.id != 0:
                s.enabled = s.id in want
        pipe.modifiers.append(m)
    elif kind == "interval":
        pipe.modifiers.append(C(mode=C.Mode.IntervalCutoff))
    elif kind == "fixed":
        pipe.modifiers.append(C(mode=C.Mode.FixedCutoff, cutoff=cutoff))
    elif kind == "bond":
        pipe.modifiers.append(B(cutoff=cutoff))
        pipe.modifiers.append(C(mode=C.Mode.BondBased))
    elif kind == "diamond":
        pipe.modifiers.append(D())
    data = pipe.compute()
    return [int(t) for t in data.particles["Structure Type"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    import ovito
    lines = [
        "# generated by tools/gen_structure_variants.py (OVITO %s)" % ".".join(map(str, ovito.version)),
        "# <fixture> <cna cutoff> <analysis> <per-atom labels, one digit each>",
    ]
    for name, cell, pos, pbc, cutoff in fixtures():
        path = write(name, cell, pos, pbc)
        for kind in ("ptm6", "ptm8", "interval", "fixed", "bond", "diamond"):
            labels = ovito_labels(path, kind, cutoff)
            assert all(0 <= t <= 9 for t in labels)
            lines.append(f"{name} {cutoff} {kind} " + "".join(map(str, labels)))
            print(name, kind, np.bincount(labels).tolist())
    with open(EXPECTED, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", EXPECTED)


if __name__ == "__main__":
    main()
