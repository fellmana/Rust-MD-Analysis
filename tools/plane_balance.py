#!/usr/bin/env python3
"""Plane-by-plane atom balance between two frames of one crystal, against
the census's defects (design §16.8a).

Below a gap between two lattice planes normal to an axis (z by default),
the two frames hold the same lattice sites when the crystal there has not
moved as a whole (a substrate under an irradiated film, a fixed bottom).
So, counting from the bottom plane up, the change in the number of atoms
below the k-th gap is exactly minus the change of V - I below it:

    N_before(k) - N_after(k) = (V - I)_after(k) - (V - I)_before(k)

with no reference lattice fitted. The census's (V - I) below the same gap
(vacancies by their site's position, interstitials by their atom's, as the
atoms are cut) is compared with it, per frame.

Planes are found in each frame from a histogram of the coordinate: a plane
is a run of bins holding atoms between two near-empty gaps. The k-th gaps of
the two frames are matched from the bottom up; the comparison stops where
the two frames no longer show the same plane sequence (a damaged region
where interstitials fill the gaps), or at --max.

Usage:
  tools/plane_balance.py BEFORE AFTER --max 480 \\
      --after-defects auto=out999/auto --after-defects local=out999/local \\
      [--before-defects auto=out0/auto]
Needs the `rmda` Python package (`pip install rmda`, or `maturin develop --release` in a checkout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import rmda


def coords(path: Path, axis: int) -> np.ndarray:
    f = rmda.read(str(path))
    return np.asarray(f.positions)[:, axis].copy()


def gaps(x: np.ndarray, lo: float, hi: float, bin_w: float, min_plane: int) -> np.ndarray:
    """Gap positions between consecutive planes in [lo, hi): the emptiest bin
    between two plane peaks; a plane holds at least `min_plane` atoms."""
    edges = np.arange(lo, hi + bin_w, bin_w)
    h, _ = np.histogram(x, bins=edges)
    occupied = h > 0.02 * h.max()
    runs = []
    i = 0
    while i < len(h):
        if occupied[i]:
            j = i
            while j < len(h) and occupied[j]:
                j += 1
            if h[i:j].sum() >= min_plane:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for (a0, a1), (b0, _) in zip(runs, runs[1:]):
        # The middle of the emptiest stretch between the two planes.
        seg = h[a1:b0]
        if len(seg):
            low = np.nonzero(seg == seg.min())[0]
            k = a1 + int(round(0.5 * (low[0] + low[-1])))
        else:
            k = a1
        out.append(edges[k] + 0.5 * bin_w)
    return np.array(out)


def read_defects(path: Path, axis: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(0)
    lines = path.read_text().splitlines()
    if not lines:
        return np.zeros(0)
    n = int(lines[0])
    props = lines[1].split("Properties=")[1].split()[0].split(":")
    col, at = 0, 0
    for name, _, width in zip(props[0::3], props[1::3], props[2::3]):
        if name == "pos":
            col = at
            break
        at += int(width)
    return np.array([float(l.split()[col + axis]) for l in lines[2:2 + n]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--axis", choices="xyz", default="z")
    ap.add_argument("--min", type=float, default=None, help="lowest coordinate (default the lowest atom)")
    ap.add_argument("--max", type=float, required=True, help="stop at this coordinate")
    ap.add_argument("--bin", type=float, default=0.05)
    ap.add_argument("--min-plane", type=int, default=500, help="atoms a plane holds at least")
    ap.add_argument("--after-defects", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--before-defects", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--every", type=int, default=10, help="print every n-th gap")
    ap.add_argument("--clean", type=float, default=0.3,
                    help="atoms within this many A of a cut (either frame) are split arbitrarily: "
                         "their number bounds the comparison there")
    args = ap.parse_args(argv)
    axis = "xyz".index(args.axis)
    x0, x1 = coords(args.before, axis), coords(args.after, axis)
    lo = args.min if args.min is not None else min(x0.min(), x1.min()) - 0.5
    g0 = gaps(x0, lo, args.max, args.bin, args.min_plane)
    g1 = gaps(x1, lo, args.max, args.bin, args.min_plane)
    n = min(len(g0), len(g1))
    s0, s1 = np.sort(x0), np.sort(x1)
    below0 = np.searchsorted(s0, g0[:n])
    below1 = np.searchsorted(s1, g1[:n])
    # Planes per gap: the counts between consecutive gaps must be alike in
    # both frames; a plane-sized jump means the sequences no longer match.
    per0 = np.diff(np.r_[0, below0])
    per1 = np.diff(np.r_[0, below1])
    typical = np.median(per0[1:]) if n > 2 else 1
    bad = np.nonzero(np.abs(per0 - per1) > 0.5 * typical)[0]
    stop = int(bad[0]) if len(bad) else n
    if stop < n:
        print(f"planes no longer match after gap {stop} (z {g0[stop]:.2f} / {g1[stop]:.2f})")

    def census(specs, gaps_at):
        out = []
        for spec in specs:
            name, d = spec.split("=", 1)
            v = np.sort(read_defects(Path(d) / "vacancies.xyz", axis))
            i = np.sort(read_defects(Path(d) / "interstitials.xyz", axis))
            out.append((name, np.searchsorted(v, gaps_at) - np.searchsorted(i, gaps_at)))
        return out

    def atoms_near(sorted_x, at, h):
        return np.searchsorted(sorted_x, at + h) - np.searchsorted(sorted_x, at - h)

    amb = atoms_near(s0, g0[:stop], args.clean) + atoms_near(s1, g1[:stop], args.clean)
    clean = np.ones(stop, dtype=bool)
    after = census(args.after_defects, g1[:stop])
    before = dict(census(args.before_defects, g0[:stop]))
    exact = (below0 - below1)[:stop]
    names = [name for name, _ in after]
    print(f"{'gap':>5} {'z before':>9} {'z after':>9} {'N0-N1':>7} " + " ".join(f"{n_:>9}" for n_ in names))
    within = {name: 0 for name in names}
    diffs = {name: [] for name in names}
    print(f"{'':>45} (± atoms at the cut)")
    for k in range(stop):
        row = f"{k:>5} {g0[k]:>9.2f} {g1[k]:>9.2f} {exact[k]:>7} "
        for name, net in after:
            c = int(net[k] - before[name][k]) if name in before else int(net[k])
            ok = abs(c - int(exact[k])) <= amb[k]
            within[name] += ok
            diffs[name].append(c - int(exact[k]))
            row += f"{c:>8}{' ' if ok else '*'} "
        if k % args.every == 0 or k == stop - 1:
            print(row + f"  ±{amb[k]}")
    print(f"\n{stop} gaps up to z {g1[stop - 1]:.1f}; median ambiguity ±{int(np.median(amb))}; "
          f"census - exact (* outside the ambiguity):")
    for name in names:
        d = np.array(diffs[name])
        print(f"  {name}: within at {within[name]} of {stop}; mean {d.mean():+.1f}, median "
              f"{np.median(d):+.0f}, range {d.min():+d}..{d.max():+d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
