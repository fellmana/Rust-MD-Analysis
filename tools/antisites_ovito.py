#!/usr/bin/env python3
"""OVITO cross-check of `rmda defects --ws-reference REF --antisites`
(design §16.1a, §16.8).

    rmda defects CURRENT --ws-reference REF --antisites --defects DIR
    python3 tools/antisites_ovito.py CURRENT REF DIR

Runs OVITO's WignerSeitzAnalysisModifier(per_type_occupancies=True) on the
same pair. An antisite is a site whose total occupancy is 1 and whose
occupancy by its own (reference) type is 0. Compares, site for site (by
reference atom id), OVITO's antisite sites with the `site_id` column of rmda's
`antisites.xyz`, and each one's occupant type with rmda's `type`; also the
vacancy and interstitial counts. Exits non-zero on any difference.

With `--counts` it prints only `vacancies,interstitials,antisites,
pairs,antisite site ids` from OVITO (pairs as `site>occupant=count;...`), the
line format of `crates/rmda-cli/tests/data/antisites_ovito.csv` (prefix the
case name).

Needs OVITO >= 3.10 and numpy.
"""
import sys
from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.pipeline import FileSource


def particle_ids(particles):
    """Atom ids, or 1..N in file order when the file has none (as rmda numbers them)."""
    if "Particle Identifier" in particles:
        return np.asarray(particles["Particle Identifier"])
    return np.arange(1, particles.count + 1)


def xyz_columns(path, names):
    """Columns of an extended-XYZ file: integers, except the string ones."""
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    props = lines[1].split("Properties=")[1].split()[0].split(":")
    at, cols, kinds = 0, {}, {}
    for pname, kind, width in zip(props[0::3], props[1::3], props[2::3]):
        cols[pname], kinds[pname] = at, kind
        at += int(width)
    rows = [l.split() for l in lines[2 : 2 + n]]
    return {k: [r[cols[k]] if kinds[k] == "S" else int(r[cols[k]]) for r in rows]
            for k in names if k in cols}


def ovito_antisites(current, reference):
    pipe = import_file(str(current))
    ws = WignerSeitzAnalysisModifier(per_type_occupancies=True)
    ws.reference = FileSource()
    ws.reference.load(str(reference))
    pipe.modifiers.append(ws)
    d = pipe.compute()
    occ = np.asarray(d.particles["Occupancy"])
    if occ.ndim == 1:
        occ = occ[:, None]
    # Occupancy components follow the current frame's particle types. OVITO
    # numbers named types per file in order of appearance, so with element
    # names the two files' ids need not agree: match by name then.
    cur = import_file(str(current)).compute()
    ref = import_file(str(reference)).compute()
    comp_ids = [t.id for t in cur.particles.particle_types.types]
    cur_name = {t.id: t.name for t in cur.particles.particle_types.types}
    ref_name = {t.id: t.name for t in ref.particles.particle_types.types}
    named = all(cur_name.values()) and all(ref_name.values())
    site_type = np.asarray(d.particles["Particle Type"])
    sid = particle_ids(d.particles)
    total = occ.sum(axis=1)
    out = {}
    for s, t, row, tot in zip(sid, site_type, occ, total):
        if tot != 1:
            continue
        if named:
            want = [k for k, i in enumerate(comp_ids) if cur_name[i] == ref_name[int(t)]]
            own = want[0] if want else None
        else:
            own = comp_ids.index(int(t)) if int(t) in comp_ids else None
        if own is not None and row[own] == 1:
            continue
        occupant = comp_ids[int(np.argmax(row))]
        out[int(s)] = ((ref_name[int(t)], cur_name[occupant]) if named
                       else (int(t), occupant))
    nv = d.attributes["WignerSeitz.vacancy_count"]
    ni = d.attributes["WignerSeitz.interstitial_count"]
    return nv, ni, out, named


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    current, reference = args[0], args[1]
    nv, ni, an, named = ovito_antisites(current, reference)
    pairs = {}
    for t, o in an.values():
        pairs[(t, o)] = pairs.get((t, o), 0) + 1
    if "--counts" in sys.argv:
        p = ";".join(f"{t}>{o}={c}" for (t, o), c in sorted(pairs.items()))
        print(f"{nv},{ni},{len(an)},{p},{';'.join(map(str, sorted(an)))}")
        return
    mdir = Path(args[2])
    m = xyz_columns(mdir / "antisites.xyz",
                    ["site_id", "type", "site_type", "species", "site_species"])
    if named:  # element names on both sides
        mine = {s: (st, t) for s, t, st in zip(m["site_id"], m["species"], m["site_species"])}
    else:
        mine = {s: (st, t) for s, t, st in zip(m["site_id"], m["type"], m["site_type"])}
    print(f"OVITO: V={nv} I={ni} antisites={len(an)} pairs={pairs}")
    print(f"rmda:   antisites={len(mine)}")
    bad = False
    for s in sorted(set(an) | set(mine)):
        if an.get(s) != mine.get(s):
            print(f"site {s}: OVITO {an.get(s)} rmda {mine.get(s)}")
            bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
