#!/usr/bin/env python3
"""Regenerate tests/data/cluster_ovito_expected.txt with OVITO (3.15 or later).

The frames are synthetic and generated identically here and in the Rust unit
test `analysis::cluster::tests::matches_ovito_on_synthetic_frames` (same LCG,
same arithmetic), so the test needs no OVITO and no stored frame, only this
file's output.

Each case records OVITO's `ClusterAnalysisModifier` result (cutoff mode,
compute_com, compute_gyration, unwrap_particles, per-type masses), translated
to rmda's conventions: 0-based ids in first-atom order, -1 for an unselected
atom.

    python3 tools/cluster_ovito_expected.py > tests/data/cluster_ovito_expected.txt
"""
import math
import sys

import numpy as np
import ovito
from ovito.data import DataCollection, ParticleType, Particles, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

M64 = (1 << 64) - 1
L = 30.0
CUTOFF = 1.9
MASSES = {1: 26.9815385, 2: 58.6934}


class Lcg:
    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & M64
        return (self.s >> 11) / float(1 << 53)


def synthetic(seed=7, nblobs=40):
    """Blobs of 3..22 atoms in a 30 A box, wrapped into it: some blobs cross
    the periodic boundary. Types 1/2 (70/30) and a 75 % selection."""
    r = Lcg(seed)
    pos, types, sel = [], [], []
    for _ in range(nblobs):
        c = [r.next() * L for _ in range(3)]
        m = 3 + int(math.floor(r.next() * 20.0))
        for _ in range(m):
            p = []
            for d in range(3):
                x = c[d] + (r.next() * 4.0 - 2.0)
                x = x - L * math.floor(x / L)
                p.append(x)
            pos.append(p)
            types.append(1 if r.next() < 0.7 else 2)
            sel.append(1 if r.next() < 0.75 else 0)
    return np.array(pos), types, sel


def run(pos, types, sel, pbc, only_selected):
    dc = DataCollection()
    cell = SimulationCell(pbc=(pbc, pbc, pbc))
    cell[...] = [[L, 0, 0, 0], [0, L, 0, 0], [0, 0, L, 0]]
    dc.objects.append(cell)
    p = Particles()
    dc.objects.append(p)
    p.create_property('Position', data=pos)
    t = p.create_property('Particle Type', data=types)
    for tid, m in MASSES.items():
        t.types.append(ParticleType(id=tid, mass=m))
    p.create_property('Selection', data=sel)
    pl = Pipeline(source=StaticSource(data=dc))
    pl.modifiers.append(ClusterAnalysisModifier(
        cutoff=CUTOFF, compute_com=True, compute_gyration=True,
        unwrap_particles=True, only_selected=only_selected, sort_by_size=False,
        # 3.17 moved the cluster list to a particle system; ask for the table.
        **({"output_table": True} if ovito.version >= (3, 17, 0) else {})))
    d = pl.compute()
    ids = np.asarray(d.particles['Cluster'][...])
    # OVITO 3.17 numbers clusters from 0 (-1: none); earlier versions from 1 (0: none).
    if ovito.version < (3, 17, 0):
        ids = ids - 1
    tab = d.tables['clusters']
    return ids, tab


def main():
    pos, types, sel = synthetic()
    out = sys.stdout
    version = ".".join(map(str, ovito.version))
    out.write(f"# OVITO {version} ClusterAnalysisModifier, cutoff {CUTOFF}, natoms {len(pos)}\n")
    for name, pbc, only in (("periodic", True, False),
                            ("open", False, False),
                            ("selected", True, True)):
        ids, tab = run(pos, types, sel, pbc, only)
        out.write(f"case {name}\n")
        out.write("ids " + " ".join(str(int(i)) for i in ids) + "\n")
        size = tab['Cluster Size'][...]
        com = tab['Center of Mass'][...]
        rg = tab['Radius of Gyration'][...]
        g = tab['Gyration Tensor'][...]
        for k in range(len(size)):
            vals = [com[k][0], com[k][1], com[k][2], rg[k], *g[k]]
            out.write(f"row {k} {int(size[k])} " + " ".join(repr(float(v)) for v in vals) + "\n")


if __name__ == "__main__":
    main()
