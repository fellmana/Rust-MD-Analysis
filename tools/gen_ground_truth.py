#!/usr/bin/env python3
"""Ground truth for the defect census (design §16.8a).

A free-standing BCC W block of N^3 cells holding defects whose content is
known by construction, written as extended XYZ with a JSON truth file:

- prismatic dislocation loops, b = a<001> on {001} or b = a/2<111> on {111}
  (the loops of irradiated W), and in FCC faulted Frank loops (b = a/3<111>,
  one (111) layer inserted in its third stacking position, or removed): an interstitial loop is one extra period of
  lattice planes normal to b inside a disc (two (002) layers, or three (222)
  ones), a vacancy loop one period inside a disc removed; the crystal around each
  is displaced by the loop's solid-angle field u = b Omega / 4 pi (the
  construction of `dislocation_loop` in tests/defects_cmd.rs, which every
  reference counts exactly), so a loop's inside is perfect crystal shifted
  by b/2 either side and only its rim is a dislocation core;
- voids (every site within a radius removed), single vacancies,
  divacancies (first neighbours) and <110> dumbbell self-interstitials;
- some features placed close together (`--close`), so that their clusters
  overlap and the local census must merge them;
- optionally a smooth bend about y (`--bend` degrees across the block), so
  that no single lattice fits and the census falls back to local regions as
  on the 128M frame; it adds no defect;
- Gaussian thermal noise.

Fields superpose (each loop's field displaces every atom but its own disc).
Each loop's field is tapered to zero between 5 and 7 loop radii (the far
field there is below 0.03 A; the taper moves no atom off its site).

Truth: per feature its kind, centre, radius, and the vacancies and
interstitials it holds (a loop of n atoms is n interstitials or n
vacancies, as its Wigner-Seitz count against the lattice around it), and
the totals. Atoms carry ids (lattice order), so the file can be permuted.

Usage:
  tools/gen_ground_truth.py OUT.xyz --cells 60 --loops 12 --voids 4 \\
      --vacancies 40 --divacancies 10 --dumbbells 20 --close 4 --bend 0 \\
      --sigma 0.05 --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

A = 3.165  # the lattice parameter along x, A (W's; set per structure in main)
# Per structure: element label, a, c, the conventional (orthogonal) cell in
# units of (a, a, c) or (a, a sqrt3, c), its basis (fractional), the
# nearest-neighbour distance, loop families, dumbbell axes.
STRUCTURES = {
    "bcc": {"element": "W", "a": 3.165, "cell": lambda a, c: (a, a, a),
            "basis": [(0, 0, 0), (0.5, 0.5, 0.5)], "nn": lambda a, c: a * math.sqrt(3) / 2,
            "loops": "001,111"},
    "fcc": {"element": "Ni", "a": 3.52, "cell": lambda a, c: (a, a, a),
            "basis": [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
            "nn": lambda a, c: a / math.sqrt(2), "loops": "110,001"},
    "hcp": {"element": "Zr", "a": 3.232, "c": 5.147,
            "cell": lambda a, c: (a, a * math.sqrt(3), c),
            "basis": [(0, 0, 0), (0.5, 0.5, 0), (0.5, 1 / 6, 0.5), (0, 2 / 3, 0.5)],
            "nn": lambda a, c: min(a, math.sqrt(a * a / 3 + c * c / 4)), "loops": "c,a"},
}
NV = 64  # loop polygon vertices
TAPER = (5.0, 7.0)  # field taper, loop radii


def solid_angle(p: np.ndarray, c: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Solid angle of the polygon `verts` (closed, around centre `c`) seen
    from each point of `p`, as a fan of triangles from the centre (Van
    Oosterom and Strackee), signed as in tests/defects_cmd.rs."""
    x = c[None, :] - p
    lx = np.linalg.norm(x, axis=1)
    total = np.zeros(len(p))
    for t in range(len(verts)):
        y = verts[t][None, :] - p
        z = verts[(t + 1) % len(verts)][None, :] - p
        ly = np.linalg.norm(y, axis=1)
        lz = np.linalg.norm(z, axis=1)
        num = np.einsum("ij,ij->i", x, np.cross(y, z))
        den = (
            lx * ly * lz
            + np.einsum("ij,ij->i", x, y) * lz
            + np.einsum("ij,ij->i", x, z) * ly
            + np.einsum("ij,ij->i", y, z) * lx
        )
        total += 2.0 * np.arctan2(num, den)
    return total


class Block:
    def __init__(self, counts, cellvec, basis):
        self.counts = np.array(counts)
        self.lengths = self.counts * np.array(cellvec)
        nx, ny, nz = counts
        i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ijk = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
        nb = len(basis)
        # Site index = nb * cell + basis; cell in (i, j, k) order.
        self.cell = np.repeat(ijk, nb, axis=0)
        self.basis = np.tile(np.arange(nb), len(ijk))
        frac = np.array(basis, dtype=float)
        self.pos = (self.cell + frac[self.basis]) * np.array(cellvec)
        self.keep = np.ones(len(self.pos), dtype=bool)
        # Atoms added (positions), with the loop they belong to (-1: none).
        self.extra: list[np.ndarray] = []
        self.extra_owner: list[int] = []
        self.taken = np.zeros(len(self.pos), dtype=bool)  # sites a feature uses


def place(rng, lengths, r: float, placed: list, gap: float, near=None):
    """A centre for a feature of radius `r` (A): `gap` A clear of every placed
    one and 25 A inside the block; with `near` (centre, radius), at `gap`
    from that one's edge."""
    lengths = np.asarray(lengths, dtype=float)
    lo, hi = 25.0 + r, lengths - 25.0 - r
    if np.any(hi <= lo):
        return None
    if np.all(hi == hi[0]):
        hi = float(hi[0])
    for _ in range(4000):
        if near is None:
            c = rng.uniform(lo, hi, 3)
        else:
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            c = near[0] + d * (near[1] + r + gap)
            if np.any(c < lo) or np.any(c > hi):
                continue
        if all(np.linalg.norm(c - q) >= rr + r + gap for q, rr in placed):
            return c
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--structure", choices=sorted(STRUCTURES), default="bcc",
                    help="bcc (W), fcc (Ni) or hcp (Zr, c/a 1.593)")
    ap.add_argument("--cells", type=int, default=60,
                    help="cells along x; along y and z as many as make the box about cubic")
    ap.add_argument("--loops", type=int, default=12, help="loops, half interstitial, half vacancy")
    ap.add_argument("--loop-radius", type=float, nargs=2, default=(2.0, 6.0), help="cells")
    ap.add_argument("--loop-types", default=None,
                    help="Burgers vector families, used in turn (prismatic, perfect): bcc 001 "
                         "(b = a<001>) and 111 (b = a/2<111>, as in irradiated W); fcc 110 "
                         "(b = a/2<110>) and 001; hcp c (b = c[0001], basal) and a "
                         "(b = a/3<11-20>, prismatic). Default: both of the structure's")
    ap.add_argument("--voids", type=int, default=4)
    ap.add_argument("--void-radius", type=float, nargs=2, default=(4.0, 9.0), help="A")
    ap.add_argument("--vacancies", type=int, default=40)
    ap.add_argument("--divacancies", type=int, default=10)
    ap.add_argument("--dumbbells", type=int, default=20)
    ap.add_argument("--close", type=int, default=4,
                    help="features placed 4-8 A from a loop's edge (point defects and voids)")
    ap.add_argument("--gap", type=float, default=12.0, help="A between features otherwise")
    ap.add_argument("--bend", type=float, default=0.0, help="degrees of bend about y across x")
    ap.add_argument("--sigma", type=float, default=0.05, help="thermal noise, A")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lattice-a", type=float, default=None,
                    help="lattice parameter a, A (default the structure's: W 3.165, Ni 3.52, "
                         "Zr 3.232; CoCrFeMnNi ~3.59)")
    ap.add_argument("--species", type=int, default=1,
                    help="random equiatomic alloy of this many species on the lattice (an HEA: "
                         "5 gives Co Cr Fe Mn Ni); drawn after everything else, so the positions "
                         "are those of the one-species block")
    ap.add_argument("--distortion", type=float, default=0.0,
                    help="static random displacement per atom, A (an HEA's size mismatch), on top "
                         "of --sigma")
    ap.add_argument("--periodic", action="store_true",
                    help="a periodic box of exactly N cells (no free surfaces, as in a bulk MD "
                         "frame): each loop's field is taken at its nearest image (its taper, "
                         "7 radii, must stay under half the box)")
    args = ap.parse_args(argv)
    if args.periodic and args.bend:
        raise SystemExit("--bend needs free surfaces: drop --periodic")

    global A
    spec = STRUCTURES[args.structure]
    A = args.lattice_a if args.lattice_a else spec["a"]
    c_len = spec.get("c", A)
    cellvec = spec["cell"](A, c_len)
    nn = spec["nn"](A, c_len)
    n = args.cells
    counts = (n, max(1, round(n * A / cellvec[1])), max(1, round(n * A / cellvec[2])))
    rng = np.random.default_rng(args.seed)
    blk = Block(counts, cellvec, spec["basis"])
    lengths = blk.lengths
    placed: list[tuple[np.ndarray, float]] = []
    truth: list[dict] = []
    loops: list[dict] = []

    # Loops: a disc of radius r normal to n (b parallel to n, pure prismatic)
    # whose plane lies midway between two lattice planes normal to n.
    fams = (args.loop_types or spec["loops"]).split(",")
    for li in range(args.loops):
        kind = "interstitial_loop" if li % 2 == 0 else "vacancy_loop"
        fam = fams[(li // 2) % len(fams)]
        if fam == "001" and args.structure in ("bcc", "fcc"):
            nvec = np.eye(3)[int(rng.integers(3))]
            bmag, spacing = A, A / 2
        elif fam == "111" and args.structure == "bcc":
            sg = rng.choice([-1.0, 1.0], size=3)
            sg[0] = 1.0
            nvec = sg / math.sqrt(3.0)
            bmag, spacing = A * math.sqrt(3.0) / 2, A / (2 * math.sqrt(3.0))
        elif fam == "110" and args.structure == "fcc":
            dirs = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
            nvec = np.array(dirs[int(rng.integers(6))], dtype=float) / math.sqrt(2.0)
            bmag, spacing = A / math.sqrt(2.0), A / (2 * math.sqrt(2.0))
        elif fam == "frank" and args.structure == "fcc":
            # b = a/3<111>, normal to the {111} habit plane: one (111) layer,
            # not a lattice period, so the stacking inside is faulted.
            sg = rng.choice([-1.0, 1.0], size=3)
            sg[0] = 1.0
            nvec = sg / math.sqrt(3.0)
            bmag, spacing = A / math.sqrt(3.0), A / math.sqrt(3.0)
        elif fam == "c" and args.structure == "hcp":
            nvec = np.array([0.0, 0.0, 1.0])
            bmag, spacing = c_len, c_len / 2
        elif fam == "a" and args.structure == "hcp":
            nvec = np.array([1.0, 0.0, 0.0])
            bmag, spacing = A, A / 2
        else:
            raise SystemExit(f"unknown loop type {fam} for {args.structure}")
        r = float(rng.uniform(*args.loop_radius)) * A
        c = place(rng, lengths, r + 2 * A, placed, args.gap)
        if c is None:
            print(f"loop {li}: no room", file=sys.stderr)
            continue
        h = (math.floor(c @ nvec / spacing) + 0.5) * spacing
        c = c + (h - c @ nvec) * nvec
        placed.append((c, r + 2 * A))
        loops.append({"kind": kind, "n": nvec, "b": bmag, "h": h, "centre": c, "r": r, "id": len(truth),
                      "frank": fam == "frank"})
        burgers = {"001": "a<001>", "111": "a/2<111>", "110": "a/2<110>", "c": "c[0001]",
                   "a": "a/3<11-20>", "frank": "a/3<111> (Frank)"}[fam]
        truth.append({"kind": kind, "burgers": burgers, "normal": nvec.round(4).tolist(),
                      "centre": c.tolist(), "radius": r, "vacancies": 0, "interstitials": 0})

    def in_disc(lp: dict, p: np.ndarray) -> np.ndarray:
        d = p - lp["centre"]
        d = d - np.outer(d @ lp["n"], lp["n"])
        return np.einsum("ij,ij->i", d, d) < lp["r"] ** 2

    for lp in loops:
        height = blk.pos @ lp["n"]
        disc = in_disc(lp, blk.pos)
        if lp["frank"]:
            # The disc plane is midway between (111) layers k0 and k0 + 1
            # (heights k d). A vacancy loop removes layer k0 in the disc; the
            # layers either side close over it (k0 - 1, k0 + 1: an intrinsic
            # fault). An interstitial loop puts layer k0 + 2's positions on
            # the disc plane (the third stacking position, so no two adjacent
            # layers repeat: an extrinsic fault).
            d = lp["b"]
            if lp["kind"] == "vacancy_loop":
                sel = disc & (np.abs(height - (lp["h"] - 0.5 * d)) < 0.25 * d) & blk.keep
                blk.keep[sel] = False
                blk.taken[sel] = True
                truth[lp["id"]]["vacancies"] = int(sel.sum())
            else:
                sel = disc & (np.abs(height - (lp["h"] + 1.5 * d)) < 0.25 * d)
                for p in blk.pos[sel] - 1.5 * d * lp["n"]:
                    blk.extra.append(p)
                    blk.extra_owner.append(lp["id"])
                truth[lp["id"]]["interstitials"] = int(sel.sum())
        elif lp["kind"] == "vacancy_loop":
            # One period of lattice planes centred on the disc plane, removed.
            sel = disc & (height >= lp["h"] - lp["b"] / 2) & (height < lp["h"] + lp["b"] / 2) & blk.keep
            blk.keep[sel] = False
            blk.taken[sel] = True
            truth[lp["id"]]["vacancies"] = int(sel.sum())
        else:
            # One extra period: the sites of the period above the disc plane,
            # moved down by b/2 (centred on the plane; the field then opens
            # the gap b). For b = a[001] this is tests/defects_cmd.rs's disc.
            sel = disc & (height >= lp["h"]) & (height < lp["h"] + lp["b"])
            for p in blk.pos[sel] - 0.5 * lp["b"] * lp["n"]:
                blk.extra.append(p)
                blk.extra_owner.append(lp["id"])
            truth[lp["id"]]["interstitials"] = int(sel.sum())
        # Sites near the disc are taken: no point defect lands in the core.
        near = np.linalg.norm(blk.pos - lp["centre"], axis=1) < lp["r"] + 2 * A
        blk.taken |= near

    # Loops as placed (radius + 2 cells of core margin), for --close.
    loop_spheres = [(lp["centre"], lp["r"] + 2 * A) for lp in loops]

    def free_site(c: np.ndarray, rad: float):
        """The free site nearest `c` with its first shell free."""
        d = np.linalg.norm(blk.pos - c, axis=1)
        for s in np.argsort(d)[:200]:
            if blk.taken[s] or not blk.keep[s]:
                continue
            shell = np.linalg.norm(blk.pos - blk.pos[s], axis=1) < 1.2 * A
            if blk.taken[shell].any():
                continue
            return int(s)
        return None

    def point_centre(r: float, i_feature: int):
        near = None
        if i_feature < args.close and loop_spheres:
            q, rr = loop_spheres[int(rng.integers(len(loop_spheres)))]
            near = (q, rr)
            return place(rng, lengths, r, placed, float(rng.uniform(4.0, 8.0)), near)
        return place(rng, lengths, r, placed, args.gap)

    feature = 0
    # Voids.
    for _ in range(args.voids):
        r = float(rng.uniform(*args.void_radius))
        c = point_centre(r, feature)
        feature += 1
        if c is None:
            continue
        sel = (np.linalg.norm(blk.pos - c, axis=1) < r) & blk.keep & ~blk.taken
        if sel.sum() < 2:
            continue
        blk.keep[sel] = False
        blk.taken |= np.linalg.norm(blk.pos - c, axis=1) < r + 1.5 * A
        placed.append((c, r + A))
        truth.append({"kind": "void", "centre": c.tolist(), "radius": r,
                      "vacancies": int(sel.sum()), "interstitials": 0})
    # Single vacancies, divacancies, dumbbells.
    # First-neighbour vectors (one of each +- pair) and dumbbell axes.
    if args.structure == "bcc":
        nn1 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]]) * A
        dumbbell_axes = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1],
                                  [0, 1, -1]]) / math.sqrt(2.0)
    else:
        mid = blk.pos[len(blk.pos) // 2]
        dv = blk.pos - mid
        dd = np.linalg.norm(dv, axis=1)
        first = dv[(dd > 0.1) & (dd < 1.05 * nn)]
        nn1 = np.array([v for v in first if (v[0], v[1], v[2]) > (0.0, 0.0, 0.0)])
        dumbbell_axes = (np.eye(3) if args.structure == "fcc"
                         else np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    for kind, count in (("vacancy", args.vacancies), ("divacancy", args.divacancies),
                        ("dumbbell", args.dumbbells)):
        for _ in range(count):
            c = point_centre(2.0 * A, feature)
            feature += 1
            if c is None:
                continue
            s = free_site(c, A)
            if s is None:
                continue
            p = blk.pos[s].copy()
            if kind == "vacancy":
                blk.keep[s] = False
                v, i_ = 1, 0
            elif kind == "divacancy":
                q = p + nn1[int(rng.integers(len(nn1)))]
                d = np.linalg.norm(blk.pos - q, axis=1)
                t = int(np.argmin(d))
                if d[t] > 0.1 or blk.taken[t] or not blk.keep[t]:
                    continue
                blk.keep[s] = False
                blk.keep[t] = False
                v, i_ = 2, 0
            else:
                # A dumbbell on the site: its atom replaced by two at
                # +-0.2 a along a random axis (BCC <110>, FCC <100>, HCP
                # [2-1-10] or [0001]).
                d = dumbbell_axes[int(rng.integers(len(dumbbell_axes)))]
                blk.keep[s] = False
                blk.extra.append(p + 0.2 * A * d)
                blk.extra_owner.append(-1)
                blk.extra.append(p - 0.2 * A * d)
                blk.extra_owner.append(-1)
                v, i_ = 0, 1
            blk.taken |= np.linalg.norm(blk.pos - p, axis=1) < 2.5 * A
            placed.append((p, 2.0 * A))
            truth.append({"kind": kind, "centre": p.tolist(), "radius": 1.5 * A,
                          "vacancies": v, "interstitials": i_})

    # Atoms: kept lattice atoms (owner -1) then the extra ones.
    pos = np.concatenate([blk.pos[blk.keep], np.array(blk.extra).reshape(-1, 3)])
    owner = np.concatenate([np.full(int(blk.keep.sum()), -1), np.array(blk.extra_owner, dtype=int)])
    ids = np.concatenate([np.nonzero(blk.keep)[0] + 1,
                          len(blk.pos) + 1 + np.arange(len(blk.extra))])
    ref = pos.copy()
    # Loop fields, superposed; an atom of a loop's own disc is not moved by it.
    for lp in loops:
        nv = lp["n"]
        c = lp["centre"]
        reach = TAPER[1] * lp["r"]
        rel = ref - c
        if args.periodic:
            rel -= lengths * np.round(rel / lengths)
            if reach > 0.5 * lengths.min():
                raise SystemExit(f"loop radius {lp['r']:.1f} A: its field reaches past half the box")
        sel = np.nonzero((np.linalg.norm(rel, axis=1) < reach) & (owner != lp["id"]))[0]
        e1 = np.cross(nv, [1.0, 0.0, 0.0] if abs(nv[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nv, e1)
        th = 2 * np.pi * np.arange(NV) / NV
        verts = c[None, :] + lp["r"] * (np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :])
        flip = -1.0 if lp["kind"] == "vacancy_loop" else 1.0
        sign = flip * np.sign(solid_angle((c + 0.1 * nv)[None, :], c, verts)[0])
        p = c + rel[sel]
        u = lp["b"] * sign * solid_angle(p, c, verts) / (4 * np.pi)
        dist = np.linalg.norm(p - c, axis=1) / lp["r"]
        w = np.clip((TAPER[1] - dist) / (TAPER[1] - TAPER[0]), 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        pos[sel] += (u * w)[:, None] * nv[None, :]
    # Bend about y: x-dependent rotation, angle `bend` across the block.
    if args.bend:
        length = lengths[0]
        cx, cz = lengths[0] / 2, lengths[2] / 2
        ang = np.radians(args.bend) * (pos[:, 0] - cx) / length
        x, z = pos[:, 0] - cx, pos[:, 2] - cz
        pos[:, 0] = cx + np.cos(ang) * x + np.sin(ang) * z
        pos[:, 2] = cz - np.sin(ang) * x + np.cos(ang) * z
    pos += rng.normal(scale=args.sigma, size=pos.shape)
    # An alloy: species and static distortion drawn last (positions above
    # are the one-species block's, for the same seed).
    names = [spec["element"]]
    species = np.zeros(len(pos), dtype=int)
    if args.species > 1:
        heas = ["Co", "Cr", "Fe", "Mn", "Ni", "Al", "Cu", "Ti"]
        names = heas[: args.species] if args.species <= len(heas) else [f"X{k}" for k in range(args.species)]
        species = rng.integers(args.species, size=len(pos))
    if args.distortion:
        pos += rng.normal(scale=args.distortion, size=pos.shape)
    offset = 0.0 if args.periodic else 15.0
    pos += offset
    if args.periodic:
        pos = np.mod(pos, lengths)

    box = lengths if args.periodic else lengths + 30.0
    pbc = "T T T" if args.periodic else "F F F"
    order = np.argsort(ids)
    rows = "\n".join(
        f"{names[sp]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(i)}"
        for p, i, sp in zip(pos[order], ids[order], species[order])
    )
    args.out.write_text(
        f"{len(pos)}\nLattice=\"{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}\" pbc=\"{pbc}\" "
        f"Properties=species:S:1:pos:R:3:id:I:1\n{rows}\n"
    )
    for t in truth:
        t["centre"] = [x + offset for x in t["centre"]]
    totals = {
        "vacancies": sum(t["vacancies"] for t in truth),
        "interstitials": sum(t["interstitials"] for t in truth),
    }
    meta = {
        "structure": args.structure, "nn": nn, "species": names, "distortion": args.distortion, "c": c_len, "lengths": lengths.tolist(),
        "cells": n, "atoms": int(len(pos)), "a": A, "sigma": args.sigma, "bend": args.bend,
        "seed": args.seed, "offset": offset, "periodic": args.periodic, "totals": totals, "features": truth,
    }
    args.out.with_suffix(".truth.json").write_text(json.dumps(meta, indent=1))
    kinds: dict[str, int] = {}
    for t in truth:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"{args.out}: {len(pos)} atoms, V={totals['vacancies']} I={totals['interstitials']}, "
          f"features {kinds}")
    # Features are kept 25 A inside the block and `--gap` A apart; those that
    # found no room were skipped, and the truth file lists only those placed.
    got = dict(kinds, loop=kinds.get("interstitial_loop", 0) + kinds.get("vacancy_loop", 0))
    wanted = {"loop": args.loops, "void": args.voids, "vacancy": args.vacancies,
              "divacancy": args.divacancies, "dumbbell": args.dumbbells}
    short = {k: f"{got.get(k, 0)}/{n}" for k, n in wanted.items() if got.get(k, 0) < n}
    if short:
        print(f"warning: no room for every feature (placed/requested {short}); "
              "use more --cells or a smaller --gap", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
