#!/usr/bin/env python3
"""Exact net defect content (V - I) inside closed shells of good crystal,
from lattice labels alone (design §16.8a).

Inside a closed surface through good crystal, sites - atoms is the exact
V - I within, whatever the inside holds (loops, voids, dislocation tangles
that do not cross the surface), and no reference lattice is fitted. Here the
surface is a spherical band of atoms, R <= |x - c| < R + w with w of at least
two bonds:

1. every band atom must be crystalline (PTM BCC or FCC, the grain's
   structure);
2. band atoms are labelled with integer lattice coordinates by walking
   first-neighbour bonds (each bond is the nearest first-shell lattice
   vector, in the orientation PTM gives, refined by least squares);
3. every band bond must agree with the labels: a dislocation piercing the
   band breaks that (its Burgers circuit does not close), and the shell is
   then refused;
4. labels must be unique (one atom per site);
5. the labels' affine map to positions is fitted over the labelled atoms;
   the region is the lattice points it puts within the band's middle,
   Rm = R + w/2.

Net V - I = region sites - atoms on them: every atom within R, labelled
atoms whose site is in the region, and unlabelled band atoms (an
interstitial, a core atom) within Rm. An atom near Rm is counted with its
own site, whichever side of Rm thermal motion puts it, so the count is exact
whenever the labels are; a vacancy or interstitial in the band is counted
too. (A flood fill over label space from outside the band would count band
atoms by position but their sites by label: atoms just inside R + w on sites
just outside it make the net 10-20 too low in 30 A shells.)
The census's V - I is taken within Rm, so defects within a bond of Rm may
fall either side (marked ?, not scored). A shell crossing a free surface
or a dislocation is refused. With `--defects NAME=DIR`, the
census's own V - I in the same spheres (its vacancy and interstitial files)
is printed beside it; with a `gen_ground_truth.py` frame, the truth's.

Usage:
  tools/net_content.py FRAME --radius 30 --random 20 --seed 1 \\
      --defects local=out/local --defects global=out/global
  tools/net_content.py FRAME --radius 25 --centre 400,400,300 --centre ...
Needs the `rmda` Python package (`pip install rmda`, or `maturin develop --release` in a checkout).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import rmda

BCC, FCC = 3, 1
SHELLS = {
    # First-shell lattice vectors in units of a/2 (doubled coordinates).
    BCC: np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]),
    FCC: np.array(
        [v for v in
         [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 1], [1, 0, -1], [-1, 0, 1],
          [-1, 0, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]]]),
}


def quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def valid_point(p: np.ndarray, structure: int) -> bool:
    if structure == BCC:
        return bool(np.all(p % 2 == 0) or np.all(p % 2 == 1))
    return bool(np.all(np.abs(p) % 1 == 0) and int(p.sum()) % 2 == 0)


class Frame:
    def __init__(self, path: Path):
        f = rmda.read(str(path))
        rmda.ptm(f)
        self.pos = np.asarray(f.positions, dtype=float)
        self.st = np.asarray(f["StructureType"])
        self.quat = np.asarray(f["StructureOrientation"])
        self.alat = np.asarray(f["StructureLatticeConstant"])
        cell = np.asarray(f.cell, dtype=float)
        if np.any(np.abs(cell - np.diag(np.diag(cell))) > 1e-9):
            raise SystemExit("net_content: only orthorhombic cells")
        self.lengths = np.diag(cell).copy()
        # Periodic along every axis the frame says; a free axis is padded so
        # that the tree's wrap never joins its two ends.
        pbc = getattr(f, "pbc", None)
        self.periodic = np.array(pbc if pbc is not None else [True, True, True], dtype=bool)
        lo = self.pos.min(axis=0)
        self.shift = np.where(self.periodic, 0.0, 1.0 - lo)
        extent = self.pos.max(axis=0) - lo
        self.box = np.where(self.periodic, self.lengths, extent + 400.0)
        p = self.pos + self.shift
        self.p = np.mod(p, self.box)
        self.tree = cKDTree(self.p, boxsize=self.box)

    def delta(self, a: np.ndarray, c: np.ndarray) -> np.ndarray:
        d = a - c
        return np.where(self.periodic, d - self.box * np.round(d / self.box), d)


def shell_net(fr: Frame, centre: np.ndarray, r: float, w: float, structure: int) -> dict:
    c = np.where(fr.periodic, np.mod(centre + fr.shift, fr.box), centre + fr.shift)
    alat0 = float(np.median(fr.alat[fr.st == structure][:100000]))
    buf = 1.5 * alat0
    near = np.array(fr.tree.query_ball_point(c, r + w + buf), dtype=int)
    if len(near) == 0:
        return {"ok": False, "why": "no atoms"}
    d = np.linalg.norm(fr.delta(fr.p[near], c), axis=1)
    n_within = int((d < r + w).sum())
    # The band, and a buffer outside it: both are labelled, so that an atom
    # just outside R + w that sits on a site the band encloses (a loop's
    # field shifts atoms by up to b/2) is counted with the enclosed atoms.
    walked = near[d >= r]
    in_band = d[d >= r] < r + w
    if in_band.sum() < 50:
        return {"ok": False, "why": "empty band"}
    cryst = fr.st[walked] == structure
    if cryst[in_band].mean() < 0.5:
        return {"ok": False, "why": "band mostly non-crystalline"}
    alat = float(np.median(fr.alat[walked][cryst]))
    vecs = SHELLS[structure]
    nn = alat / 2 * np.linalg.norm(vecs[0])
    cut = 0.5 * (nn + alat)  # between the first and second shells
    # Orientation: PTM's at a crystalline band atom, refined below.
    dw = d[d >= r]
    k0 = walked[np.argmin(np.where(cryst & in_band, np.abs(dw - (r + w / 2)), np.inf))]
    t = quat_matrix(fr.quat[k0]) * alat / 2  # doubled lattice coords -> A
    bp = fr.delta(fr.p[walked], c)
    pairs = cKDTree(bp).query_pairs(cut, output_type="ndarray")
    if len(pairs) == 0:
        return {"ok": False, "why": "no bonds"}
    for _ in range(2):
        ti = np.linalg.inv(t)
        v = bp[pairs[:, 1]] - bp[pairs[:, 0]]
        f = v @ ti.T
        dl = np.rint(f).astype(int)
        okb = (np.abs(dl).sum(axis=1) == np.abs(vecs[0]).sum()) & \
            (np.linalg.norm(f - dl, axis=1) < 0.35)
        if okb.sum() < 10:
            return {"ok": False, "why": "bonds match no lattice vector"}
        sol, *_ = np.linalg.lstsq(dl[okb].astype(float), v[okb], rcond=None)
        t = sol.T
    ti = np.linalg.inv(t)
    v = bp[pairs[:, 1]] - bp[pairs[:, 0]]
    f = v @ ti.T
    dl = np.rint(f).astype(int)
    shell_like = (np.abs(dl)[:, None, :] == np.abs(vecs)[None, :, :]).all(axis=2).any(axis=1) & \
        (np.abs(dl).sum(axis=1) == np.abs(vecs[0]).sum())
    good = shell_like & (np.linalg.norm(f - dl, axis=1) < 0.35)
    adj = collections.defaultdict(list)
    for (a, b), gd, dd in zip(pairs, good, dl):
        if gd:
            adj[a].append((b, dd))
            adj[b].append((a, -dd))
    n = len(walked)
    lab = np.zeros((n, 3), dtype=np.int64)
    start = int(np.nonzero(cryst & in_band)[0][0])
    # The start's site: the valid lattice point nearest it. (Stepping an
    # invalid point by a first-shell vector keeps it invalid: (1, 1, 0)
    # keeps an FCC point's parity, (1, 1, 1) a mixed BCC point mixed; every
    # label was then off the lattice and the flood fill leaked.)
    u = bp[start] @ ti.T
    cand = [np.rint(u).astype(int) + np.array(o) for o in np.ndindex(3, 3, 3)]
    cand = [q - 1 for q in cand if valid_point(q - 1, structure)]
    q0 = min(cand, key=lambda q: float(np.linalg.norm((q - u) @ t.T)))
    lab[start] = q0
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    queue = collections.deque([start])
    conflicts = 0
    while queue:
        a = queue.popleft()
        for b, dd in adj[a]:
            want = lab[a] + dd
            if not seen[b]:
                seen[b] = True
                lab[b] = want
                queue.append(b)
            elif not (lab[b] == want).all():
                conflicts += 1
    # Atoms the walk does not reach (interstitials, core atoms) are not
    # labelled; in the band they are counted as atoms only.
    if seen[in_band].sum() < 0.95 * in_band.sum():
        return {"ok": False, "why": f"band not connected ({int((~seen[in_band]).sum())} unreached)"}
    if conflicts:
        return {"ok": False, "why": f"{conflicts // 2} bonds disagree with the labels "
                                     "(a dislocation crosses the band)"}
    labelled = seen
    lab_all = lab
    lab = lab_all[labelled & in_band]
    n = int(in_band.sum())
    labels = {tuple(x) for x in lab_all[labelled].tolist()}
    if len(labels) != int(labelled.sum()):
        return {"ok": False, "why": "two atoms on one site"}
    # Geometric count: the labels' affine map to positions, fitted over the
    # labelled atoms; the region is the lattice points it puts within the
    # band's middle, Rm = R + w/2. Atoms inside: every atom within R (none
    # is labelled), labelled atoms whose site is in the region, unlabelled
    # walked atoms within Rm. Exact when the labels are consistent: a site
    # near Rm is counted with the atom on it, whichever side it sits.
    L = lab_all[labelled].astype(float)
    X = bp[labelled]
    A = np.hstack([L, np.ones((len(L), 1))])
    sol, *_ = np.linalg.lstsq(A, X, rcond=None)
    T, o = sol[:3].T, sol[3]
    fit_rms = float(np.sqrt(((A @ sol - X) ** 2).sum(axis=1).mean()))
    rm = r + w / 2
    Ti = np.linalg.inv(T)
    span = np.abs((np.eye(3) * rm) @ Ti.T).max() + 3
    c0 = np.rint(-o @ Ti.T).astype(int)
    rng_ = np.arange(-int(span), int(span) + 1)
    g = np.stack(np.meshgrid(rng_, rng_, rng_, indexing="ij"), -1).reshape(-1, 3) + c0
    if structure == BCC:
        g = g[np.all(g % 2 == 0, axis=1) | np.all(g % 2 == 1, axis=1)]
    else:
        g = g[g.sum(axis=1) % 2 == 0]
    gp = g @ T.T + o
    region = np.linalg.norm(gp, axis=1) < rm
    sites = int(region.sum())
    rset = {tuple(x) for x in g[region].tolist()}
    on_sites = sum(tuple(x) in rset for x in lab_all[labelled].tolist())
    dwalk = np.linalg.norm(bp, axis=1)
    loose = int(((~labelled) & (dwalk < rm)).sum())
    inner = int((d < r).sum())
    atoms = inner + on_sites + loose
    # Unlabelled walked atoms near Rm could sit on either side.
    edge = int(((~labelled) & (np.abs(dwalk - rm) < 1.0)).sum())
    return {"ok": True, "sites": sites, "atoms": atoms, "net": sites - atoms, "band_atoms": n,
            "pulled_in": on_sites, "loose": loose, "edge": edge, "fit_rms": round(fit_rms, 3)}


def read_defects(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    lines = path.read_text().splitlines()
    if not lines:
        return np.zeros((0, 3))
    n = int(lines[0])
    props = lines[1].split("Properties=")[1].split()[0].split(":")
    col, at = 0, 0
    for name, _, width in zip(props[0::3], props[1::3], props[2::3]):
        if name == "pos":
            col = at
            break
        at += int(width)
    return np.array([[float(x) for x in l.split()[col:col + 3]] for l in lines[2:2 + n]]).reshape(-1, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", type=Path)
    ap.add_argument("--radius", type=float, required=True, help="R, A")
    ap.add_argument("--width", type=float, default=None, help="band width w, A (default 2.5 bonds)")
    ap.add_argument("--centre", action="append", default=[], help="x,y,z (repeatable)")
    ap.add_argument("--random", type=int, default=0, help="this many random centres in the material")
    ap.add_argument("--at-features", action="store_true",
                    help="(a gen_ground_truth.py frame) one shell per feature, R = its radius + --radius")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--structure", choices=("bcc", "fcc"), default="bcc")
    ap.add_argument("--defects", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    structure = BCC if args.structure == "bcc" else FCC
    fr = Frame(args.frame)
    w = args.width
    if w is None:
        alat = float(np.median(fr.alat[fr.st == structure]))
        nn = alat * (3**0.5 / 2 if structure == BCC else 2**-0.5)
        w = 2.5 * nn
    centres = [np.array([float(x) for x in c.split(",")]) for c in args.centre]
    rng = np.random.default_rng(args.seed)
    crystal = np.nonzero(fr.st == structure)[0]
    for _ in range(args.random):
        centres.append(fr.pos[crystal[rng.integers(len(crystal))]].copy())
    truth = None
    tj = args.frame.with_suffix(".truth.json")
    if tj.exists():
        truth = json.loads(tj.read_text())
    radii = [args.radius] * len(centres)
    if args.at_features:
        if truth is None:
            raise SystemExit("net_content: --at-features needs FRAME's .truth.json")
        for ft in truth["features"]:
            centres.append(np.array(ft["centre"]))
            radii.append(ft["radius"] + args.radius)
    sets = []
    for spec in args.defects:
        name, d = spec.split("=", 1)
        sets.append((name, read_defects(Path(d) / "vacancies.xyz"), read_defects(Path(d) / "interstitials.xyz")))

    def within(pts: np.ndarray, c: np.ndarray, r: float) -> int:
        if len(pts) == 0:
            return 0
        dd = (pts + fr.shift) - np.where(fr.periodic, np.mod(c + fr.shift, fr.box), c + fr.shift)
        dd = np.where(fr.periodic, dd - fr.box * np.round(dd / fr.box), dd)
        return int((np.linalg.norm(dd, axis=1) < r).sum())

    rows = []
    agree = {name: [0, 0] for name, _, _ in sets}
    if truth:
        agree["truth"] = [0, 0]
    head = f"{'centre':>26} {'net':>6} " + " ".join(f"{n:>9}" for n, _, _ in sets)
    print(f"{args.frame.name}: R={args.radius} w={w:.2f}")
    print(head + ("    truth" if truth else ""))
    for c, rad in zip(centres, radii):
        res = shell_net(fr, c, rad, w, structure)
        cs = ",".join(f"{x:.1f}" for x in c)
        if not res["ok"]:
            print(f"{cs:>26} refused: {res['why']}")
            rows.append({"centre": c.tolist(), **res})
            continue
        line = f"{cs:>26} {res['net']:>6} (loose {res['loose']} edge {res['edge']} rms {res['fit_rms']}) "
        res["census"] = {}
        nn_ = w / 2.5
        for name, v, i in sets:
            net = within(v, c, rad + w / 2) - within(i, c, rad + w / 2)
            # Defects within a bond of R + w are counted inside or out by
            # position, not by site: a miss there is marked ?, not *.
            edge = (within(v, c, rad + w / 2 + nn_) - within(v, c, rad + w / 2 - nn_)
                    + within(i, c, rad + w / 2 + nn_) - within(i, c, rad + w / 2 - nn_))
            res["census"][name] = net
            if net == res["net"]:
                agree[name][0] += 1
                agree[name][1] += 1
                line += f"{net:>9} "
            elif edge:
                line += f"{net:>8}? "
            else:
                agree[name][0] += 1
                line += f"{net:>8}* "
        if truth:
            tv = sum(f["vacancies"] - f["interstitials"] for f in truth["features"]
                     if np.linalg.norm(np.array(f["centre"]) - c) + f["radius"] < rad + w / 2)
            cut = [f for f in truth["features"]
                   if abs(np.linalg.norm(np.array(f["centre"]) - c) - rad - w / 2) <= f["radius"]]
            if cut:
                line += "   (feature cut)"
            else:
                res["truth"] = tv
                agree["truth"][0] += 1
                agree["truth"][1] += tv == res["net"]
                line += f"{tv:>8}" + ("" if tv == res["net"] else "*")
        print(line)
        rows.append({"centre": c.tolist(), **res})
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n{ok} of {len(rows)} shells usable; agreement with the label count (* a miss, ? a miss "
          f"with defects at the boundary, not scored):")
    for name, (n, a) in agree.items():
        print(f"  {name}: {a} of {n}")
    if args.json:
        args.json.write_text(json.dumps({"radius": args.radius, "width": w, "shells": rows,
                                         "agreement": agree}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
