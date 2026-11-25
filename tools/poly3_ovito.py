#!/usr/bin/env python3
"""OVITO cross-check of the 3-lattice polycrystal fixture (crates/rmda-cli/tests/poly3.rs).

Generate the frames, keeping them, then run this on the directory:

    RMDA_POLY3_DIR=/tmp/poly3 cargo test --release -p rmda-cli --test poly3 -- --include-ignored
    python3 tools/poly3_ovito.py /tmp/poly3/series

Per frame it reports
  * grains: OVITO GrainSegmentationModifier (defaults) against the construction
    labels (atom id // 10**6 - 1) and against `rmda segment` (`seg.<f>.xyz`), as
    adjusted Rand indices over all atoms;
  * defects: OVITO WignerSeitzAnalysisModifier against the defect-free reference
    the generator wrote (`reference.<f>.dump`), per construction grain, next to
    `rmda defects --segment` (`series/grains.csv`), which has no reference.

Needs OVITO >= 3.10 and numpy.
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import (
    GrainSegmentationModifier,
    PolyhedralTemplateMatchingModifier,
    WignerSeitzAnalysisModifier,
)
from ovito.pipeline import FileSource

DEEP = 9.0  # Å, as in poly3.rs


def ari(a, b):
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    m = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(m, (a, b), 1)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    s, sa, sb, t = c2(m), c2(m.sum(1)), c2(m.sum(0)), len(a) * (len(a) - 1) / 2
    e = sa * sb / t
    return (s - e) / (0.5 * (sa + sb) - e)


def grain_column(path):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    props = lines[1].split("Properties=")[1].split()[0].split(":")
    at = 0
    for name, _, width in zip(props[0::3], props[1::3], props[2::3]):
        if name == "Grain":
            col = at
        at += int(width)
    return np.array([int(l.split()[col]) for l in lines[2 : 2 + n]])


class Truth:
    def __init__(self, d):
        rows = list(csv.DictReader(open(d / "truth_grains.csv")))
        self.rows = rows
        self.l = None

    def depth(self, p, f, l):
        """Power-diagram owner and distance to its nearest facet."""
        cand = []
        for r in self.rows:
            if int(r["frame"]) != f:
                continue
            s = np.array([float(r["sx"]), float(r["sy"]), float(r["sz"])])
            base = p + (s - p) - l * np.round((s - p) / l)
            for im in itertools.product((-1, 0, 1), repeat=3):
                si = base + l * np.array(im)
                cand.append((((p - si) ** 2).sum() - float(r["weight"]), int(r["grain"]), si))
        cand.sort(key=lambda c: c[0])
        sc, k, s = cand[0]
        return k, min((c[0] - sc) / (2 * np.linalg.norm(s - c[2])) for c in cand[1:])


def main(d):
    d = Path(d)
    truth = Truth(d)
    grains_csv = list(csv.DictReader(open(d / "series" / "grains.csv")))
    frames = sorted(int(p.name.split(".")[-1]) for p in d.glob("poly3.dump.*"))
    for f in frames:
        pipe = import_file(str(d / f"poly3.dump.{f}"))
        pipe.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True))
        pipe.modifiers.append(GrainSegmentationModifier())
        data = pipe.compute()
        ids = np.asarray(data.particles["Particle Identifier"])
        order = np.argsort(ids)
        ids = ids[order]
        construction = ids // 1_000_000 - 1
        ovito = np.asarray(data.particles["Grain"])[order]
        rmda = grain_column(d / f"seg.{f}.xyz")  # written in id order
        print(
            f"frame {f}: grains OVITO {data.attributes['GrainSegmentation.grain_count']} "
            f"(auto threshold {data.attributes['GrainSegmentation.auto_merge_threshold']:.1f} deg); "
            f"ARI OVITO-construction {ari(ovito, construction):.4f}, "
            f"rmda-construction {ari(rmda, construction):.4f}, OVITO-rmda {ari(ovito, rmda):.4f}"
        )

        # Wigner-Seitz against the defect-free reference, per construction grain.
        pipe = import_file(str(d / f"poly3.dump.{f}"))
        ws = WignerSeitzAnalysisModifier(output_displaced=False)
        ws.reference = FileSource()
        ws.reference.load(str(d / f"reference.{f}.dump"))
        pipe.modifiers.append(ws)
        data = pipe.compute()
        occ = np.asarray(data.particles["Occupancy"])
        ref = ws.reference.data
        rid = np.asarray(ref.particles["Particle Identifier"])
        rpos = np.asarray(ref.particles.positions)
        l = float(np.asarray(data.cell)[0, 0])
        out = {}
        for i in range(len(occ)):
            if occ[i] == 1:
                continue
            k = rid[i] // 1_000_000 - 1
            _, depth = truth.depth(rpos[i], f, l)
            key = (k, depth > DEEP)
            v, s = out.get(key, (0, 0))
            out[key] = (v + (occ[i] == 0), s + max(occ[i] - 1, 0))
        # rmda rows of this frame, mapped to construction grains through the
        # rmda segmentation labels (grain label = segmentation label).
        votes = {}
        for g, c in zip(rmda, construction):
            votes.setdefault(g, np.zeros(16, int))[c] += 1
        to_truth = {g: int(v.argmax()) for g, v in votes.items()}
        rows = [r for r in grains_csv if int(r["file_index"]) == f]
        print("  grain  OVITO-WS(ref) interior V/I  boundary V/I | rmda interior V/I  boundary V/I")
        for r in sorted(rows, key=lambda r: to_truth[int(r["grain"])]):
            k = to_truth[int(r["grain"])]
            iv, ii = out.get((k, True), (0, 0))
            bv, bi = out.get((k, False), (0, 0))
            print(
                f"  {k:5d}  {iv:>14d}/{ii:<3d}  {bv:>9d}/{bi:<3d} | "
                f"{r['vacancies']:>13}/{r['interstitials']:<3} "
                f"{r['gb_vacancies']:>9}/{r['gb_interstitials']:<3}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
