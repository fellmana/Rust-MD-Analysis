#!/usr/bin/env python3
"""Regenerate tests/data/cluster_pair_ovito_expected.txt with OVITO (3.15 or later).

Type-pair cutoffs for `rmda cluster --pair-cutoff`. OVITO 3.15's
ClusterAnalysisModifier has no pair-cutoff neighbour mode of its own (its
`neighbor_mode` is CutoffRange or Bonding); the OVITO route is
CreateBondsModifier(mode=Pairwise, set_pairwise_cutoff by type name or id)
followed by ClusterAnalysisModifier(neighbor_mode=Bonding). That is what this
records.

The frame is the two-species synthetic frame of tools/cluster_ovito_expected.py
(types 1/2 named Al/Ni, 70/30, 75 % selected), built identically in the Rust
unit test `analysis::cluster::tests::pair_cutoffs_match_ovito`.

    python3 tools/cluster_pair_ovito_expected.py > tests/data/cluster_pair_ovito_expected.txt
"""
import sys

import numpy as np
import ovito
from ovito.data import DataCollection, ParticleType, Particles, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CreateBondsModifier
from ovito.pipeline import Pipeline, StaticSource

from cluster_ovito_expected import L, MASSES, synthetic

NAMES = {1: "Al", 2: "Ni"}

# name, pbc, only_selected, pair cutoffs (keys as handed to OVITO and rmda)
CASES = (
    ("ids", True, False, {(1, 1): 1.9, (1, 2): 1.5, (2, 2): 2.3}),
    ("names", True, False, {("Al", "Al"): 1.9, ("Ni", "Al"): 2.2}),
    ("open", False, False, {(1, 1): 1.6, (1, 2): 2.0, (2, 2): 1.4}),
    ("selected", True, True, {("Al", "Ni"): 2.1, (2, 2): 1.8}),
)


def run(pos, types, sel, pbc, only_selected, pairs):
    dc = DataCollection()
    cell = SimulationCell(pbc=(pbc, pbc, pbc))
    cell[...] = [[L, 0, 0, 0], [0, L, 0, 0], [0, 0, L, 0]]
    dc.objects.append(cell)
    p = Particles()
    dc.objects.append(p)
    p.create_property('Position', data=pos)
    t = p.create_property('Particle Type', data=types)
    for tid, m in MASSES.items():
        t.types.append(ParticleType(id=tid, name=NAMES[tid], mass=m))
    p.create_property('Selection', data=sel)
    pl = Pipeline(source=StaticSource(data=dc))
    cb = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    for (a, b), c in pairs.items():
        cb.set_pairwise_cutoff(a, b, c)
    pl.modifiers.append(cb)
    pl.modifiers.append(ClusterAnalysisModifier(
        neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding,
        compute_com=True, compute_gyration=True, unwrap_particles=True,
        only_selected=only_selected, sort_by_size=False,
        # 3.17 moved the cluster list to a particle system; ask for the table.
        **({"output_table": True} if ovito.version >= (3, 17, 0) else {})))
    d = pl.compute()
    ids = np.asarray(d.particles['Cluster'][...])
    # OVITO 3.17 numbers clusters from 0 (-1: none); earlier versions from 1 (0: none).
    if ovito.version < (3, 17, 0):
        ids = ids - 1
    return ids, d.tables['clusters']


def main():
    pos, types, sel = synthetic()
    out = sys.stdout
    version = ".".join(map(str, ovito.version))
    out.write(f"# OVITO {version} CreateBondsModifier(Pairwise) + ClusterAnalysisModifier(Bonding), "
              f"natoms {len(pos)}\n")
    for name, pbc, only, pairs in CASES:
        ids, tab = run(pos, types, sel, pbc, only, pairs)
        spec = " ".join(f"{a}-{b}={c}" for (a, b), c in pairs.items())
        out.write(f"case {name} {int(pbc)} {int(only)} {spec}\n")
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
