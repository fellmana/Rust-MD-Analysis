#!/usr/bin/env python3
"""Capture OVITO CNA (adaptive) per-atom codes for a structure file.

Used to validate ``rmda`` against OVITO on the large ``reference_structures``
MD snapshots (FCC Al, BCC W, HCP Zr, and a polycrystalline HEA).

Usage:  python3 tools/capture_cna.py <structure file> <output file>
"""
import sys
from collections import Counter

from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier as CNA

src, dst = sys.argv[1], sys.argv[2]

pipeline = import_file(src)
pipeline.modifiers.append(CNA(mode=CNA.Mode.AdaptiveCutoff))
data = pipeline.compute()
types = [int(t) for t in data.particles["Structure Type"]]

hist = Counter(types)
summary = " ".join(f"{k}x{v}" for k, v in sorted(hist.items()))
print(f"{src}: natoms={len(types)} cna={summary}")

with open(dst, "w") as fh:
    fh.write(f"cna_adaptive {summary}\n")
    fh.write(" ".join(map(str, types)))
    fh.write("\n")
print(f"wrote {dst}")
