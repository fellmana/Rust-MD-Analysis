#!/usr/bin/env python3
"""Write tests/data/triclinic_tilted.dump: a restricted-triclinic LAMMPS dump
with negative tilts, a non-zero origin and mixed boundary flags.

The `ITEM: BOX BOUNDS xy xz yz` rows hold xlo_bound/xhi_bound, not xlo/xhi.
The bounds are written with LAMMPS's own formulas (domain.cpp, `boxlo_bound`)
and read back with ASE's lammpsrun reader, an independent implementation, so
the fixture does not depend on the code under test.
"""
import os
import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "data", "triclinic_tilted.dump")

lo = np.array([1.0, -2.0, 0.5])
lx, ly, lz = 10.0, 9.0, 8.0
xy, xz, yz = -2.0, 1.5, -1.0
cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
hi = lo + [lx, ly, lz]

# domain.cpp: boxlo_bound / boxhi_bound
xlo_b = min(lo[0], lo[0] + xy)
xlo_b = min(xlo_b, xlo_b + xz)
ylo_b = min(lo[1], lo[1] + yz)
xhi_b = max(hi[0], hi[0] + xy)
xhi_b = max(xhi_b, xhi_b + xz)
yhi_b = max(hi[1], hi[1] + yz)

frac = np.random.default_rng(7).random((8, 3))
pos = lo + frac @ cell
# float(): numpy 2 reprs a numpy scalar as `np.float64(...)`.

with open(OUT, "w") as f:
    f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n8\n")
    # `ss` (shrink-wrapped) is non-periodic in LAMMPS; `fm` mixes two kinds.
    f.write("ITEM: BOX BOUNDS xy xz yz pp fm ss\n")
    f.write(f"{float(xlo_b)!r} {float(xhi_b)!r} {xy!r}\n")
    f.write(f"{float(ylo_b)!r} {float(yhi_b)!r} {xz!r}\n")
    f.write(f"{float(lo[2])!r} {float(hi[2])!r} {yz!r}\n")
    f.write("ITEM: ATOMS id type x y z\n")
    for i, p in enumerate(pos):
        f.write(f"{i + 1} 1 {float(p[0])!r} {float(p[1])!r} {float(p[2])!r}\n")

back = read(OUT, format="lammps-dump-text")
assert np.allclose(back.cell[:], cell), back.cell[:]
assert np.allclose(back.positions, pos)
print("wrote", os.path.normpath(OUT))
