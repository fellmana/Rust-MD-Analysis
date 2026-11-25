#!/usr/bin/env python3
"""Capture OVITO PTM per-atom structure types, RMSDs and orientations for a structure file.

Used to validate ``rmda`` against OVITO on the large ``reference_structures`` MD
snapshots (FCC Al, BCC W, HCP Zr, and a polycrystalline HEA).

OVITO's PTM modifier is configured with its own defaults, deliberately: in
OVITO 3.15 those check FCC, HCP and BCC only (ICO, simple cubic, both diamonds
and graphene are listed in ``modifier.structures`` but disabled), with
``rmsd_cutoff = 0.1``. That is the configuration ``rmda ptm`` has to match by
default. (The "structures checked" line below lists every structure OVITO
knows, enabled or not.) The two-shell structures are compared separately by
``tools/gen_structure_variants.py``.

Usage:  python3 tools/capture_ptm.py <structure file> <output file>
"""
import sys
from collections import Counter

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM

src, dst = sys.argv[1], sys.argv[2]

pipeline = import_file(src)
modifier = PTM()
# Turn on the residual, so the oracle records *why* an atom was rejected and not
# just that it was. Without this, an OTHER and a mis-fitted FCC are
# indistinguishable in the output file.
modifier.output_rmsd = True
# And the orientation, so per-atom quaternions can be compared too (written
# w x y z, PTM's own order; OVITO stores x y z w).
modifier.output_orientation = True
pipeline.modifiers.append(modifier)
data = pipeline.compute()

types = [int(t) for t in data.particles["Structure Type"]]
rmsd = [float(r) for r in data.particles["RMSD"]]
orient = [(float(q[3]), float(q[0]), float(q[1]), float(q[2])) for q in data.particles["Orientation"]]

hist = Counter(types)
summary = " ".join(f"{k}x{v}" for k, v in sorted(hist.items()))
print(f"{src}: natoms={len(types)} ptm={summary} cutoff={modifier.rmsd_cutoff}")
print(f"  structures checked: {[s.name for s in modifier.structures]}")

with open(dst, "w") as fh:
    fh.write(f"ptm {summary} cutoff={modifier.rmsd_cutoff}\n")
    fh.write(" ".join(map(str, types)))
    fh.write("\n")
    fh.write(" ".join(f"{r:.17g}" for r in rmsd))
    fh.write("\n")
    fh.write(" ".join(f"{v:.17g}" for q in orient for v in q))
    fh.write("\n")
print(f"wrote {dst}")
