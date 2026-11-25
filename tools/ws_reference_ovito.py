#!/usr/bin/env python3
"""OVITO cross-check of `rmda defects --ws-reference` (design §16.3a, §16.8).

    rmda defects CURRENT --ws-reference REF [--ws-affine to-reference] --defects DIR
    python3 tools/ws_reference_ovito.py CURRENT REF DIR [off|to-reference]

Runs OVITO's WignerSeitzAnalysisModifier on the same pair and compares, site
for site (by reference atom id): the vacant sites against rmda's
`vacancies.xyz` `site_id` column, and each over-occupied site's extra
occupants against the `site_id`s of rmda's `interstitials.xyz`. Also checks
that rmda leaves the atom nearest each over-occupied site off its interstitial
list (OVITO lists every occupant of such a site; which one is "the" occupant
is rmda's choice). Exits non-zero on any difference.

With `--counts` it prints only `vacancies,interstitials,vacancy site ids,
interstitial site ids` from OVITO, the line format of
`crates/rmda-cli/tests/data/ws_reference_ovito.csv`.

Needs OVITO >= 3.10 and numpy.
"""
import collections
import sys
from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.pipeline import FileSource, ReferenceConfigurationModifier


def particle_ids(particles):
    """Atom ids, or 1..N in file order when the file has none (as rmda numbers them)."""
    if "Particle Identifier" in particles:
        return np.asarray(particles["Particle Identifier"])
    return np.arange(1, particles.count + 1)


def xyz_column(path, name):
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    props = lines[1].split("Properties=")[1].split()[0].split(":")
    at = 0
    col = None
    for pname, _, width in zip(props[0::3], props[1::3], props[2::3]):
        if pname == name:
            col = at
        at += int(width)
    if col is None:
        sys.exit(f"{path} has no {name} column")
    return [int(l.split()[col]) for l in lines[2 : 2 + n]]


def ovito_ws(current, reference, mapping):
    m = {
        "off": ReferenceConfigurationModifier.AffineMapping.Off,
        "to-reference": ReferenceConfigurationModifier.AffineMapping.ToReference,
    }[mapping]
    out = {}
    for displaced in (False, True):
        pipe = import_file(str(current))
        ws = WignerSeitzAnalysisModifier(output_displaced=displaced, affine_mapping=m)
        ws.reference = FileSource()
        ws.reference.load(str(reference))
        pipe.modifiers.append(ws)
        out[displaced] = pipe.compute()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    counts_only = "--counts" in sys.argv
    current, reference = args[0], args[1]
    mapping = args[-1] if args[-1] in ("off", "to-reference") else "off"
    d = ovito_ws(current, reference, mapping)
    sites = d[False]
    occ = np.asarray(sites.particles["Occupancy"])
    sid = particle_ids(sites.particles)
    vac = sorted(int(i) for i in sid[occ == 0])
    inter = sorted(int(i) for i, o in zip(sid, occ) for _ in range(max(int(o) - 1, 0)))
    nv = sites.attributes["WignerSeitz.vacancy_count"]
    ni = sites.attributes["WignerSeitz.interstitial_count"]
    assert nv == len(vac) and ni == len(inter)
    if counts_only:
        print(f"{nv},{ni},{';'.join(map(str, vac))},{';'.join(map(str, inter))}")
        return

    mdir = Path(args[2])
    mvac = sorted(xyz_column(mdir / "vacancies.xyz", "site_id"))
    minter = sorted(xyz_column(mdir / "interstitials.xyz", "site_id"))
    mids = set(xyz_column(mdir / "interstitials.xyz", "id"))
    ok = True
    print(f"OVITO: V={nv} I={ni}; rmda: V={len(mvac)} I={len(minter)}")
    if mvac != vac:
        ok = False
        a, b = set(vac), set(mvac)
        print(f"vacant sites differ: OVITO only {sorted(a - b)[:20]}, rmda only {sorted(b - a)[:20]}")
    if minter != inter:
        ok = False
        a, b = collections.Counter(inter), collections.Counter(minter)
        print(f"interstitial sites differ: OVITO only {list((a - b).items())[:20]}, "
              f"rmda only {list((b - a).items())[:20]}")

    # The kept occupant of each over-occupied site is its nearest atom.
    atoms = d[True]
    aid = particle_ids(atoms.particles)
    # OVITO names sites by reference id when the reference has ids, else by index.
    if "Site Identifier" in atoms.particles:
        site_of = np.asarray(atoms.particles["Site Identifier"])
    else:
        site_of = np.asarray(atoms.particles["Site Index"]) + 1
    aocc = np.asarray(atoms.particles["Occupancy"])
    pos = np.asarray(atoms.particles.positions)
    rpos = {int(i): p for i, p in zip(sid, np.asarray(sites.particles.positions))}
    # Off: nearest is measured with periodic images of the reference cell.
    cell = np.asarray(sites.cell)[:, :3]
    inv = np.linalg.inv(cell)
    pbc = np.asarray(sites.cell.pbc)
    by_site = collections.defaultdict(list)
    for i in np.nonzero(aocc > 1)[0]:
        dr = pos[i] - rpos[int(site_of[i])]
        if mapping == "off":
            f = inv @ dr
            f[pbc] -= np.round(f[pbc])
            dr = cell @ f
        by_site[int(site_of[i])].append((float(dr @ dr), int(aid[i])))
    kept_wrong = 0
    if mapping == "off":
        for s, v in by_site.items():
            keep = min(v)[1]
            if keep in mids or any(a not in mids for _, a in v if a != keep):
                kept_wrong += 1
        print(f"over-occupied sites whose nearest atom rmda does not keep: {kept_wrong}")
    ok = ok and kept_wrong == 0
    print("site-for-site: " + ("EQUAL" if ok else "DIFFERENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
