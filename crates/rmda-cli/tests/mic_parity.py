"""Random-cell MIC cases for the live ASE parity test (`mic_parity.rs`).

Prints one case per line:

    <n> <pbc0> <pbc1> <pbc2> <h00>..<h22> <dr0> <dr1> <dr2> <ase_len2> <brute_len2>

`ase_len2` is |find_mic(dr)|²; `brute_len2` is the exact minimum over a
bounded enumeration, independent of both ASE and the Rust port. The Rust
test asserts against `brute_len2` always, against `ase_len2` only where
ASE itself is optimal, so a defect on either side is attributable.

Cells come in four flavours: orthorhombic, LAMMPS-restricted triclinic,
and random cells Minkowski-reduced first (partial-pbc cases can otherwise
be pathological for the brute-force bound). Exit code 3 means ASE or
numpy is unavailable — the Rust side turns that into a skip.
"""

import itertools
import sys

try:
    import numpy as np
    from ase.geometry import Cell, find_mic, minkowski_reduce
except ImportError:
    sys.exit(3)


def brute_len2(h, pbc, dr):
    det = abs(np.linalg.det(h))
    if det < 1e-6:
        return None
    heights = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = np.linalg.norm(np.cross(h[j], h[k]))
        heights.append(det / area)
    dl = float(np.linalg.norm(dr))
    ranges = []
    for i in range(3):
        if not pbc[i]:
            ranges.append((0,))
        else:
            r = int(np.ceil(2 * dl / heights[i])) + 1
            if r > 40:
                return None
            ranges.append(range(-r, r + 1))
    count = 1
    for rg in ranges:
        count *= len(rg)
    if count > 400000:
        return None
    best = np.inf
    for n in itertools.product(*ranges):
        v = dr + np.array(n) @ h
        l2 = float(v @ v)
        if l2 < best:
            best = l2
    return best


def main():
    rng = np.random.default_rng(20261006)
    patterns = [(1, 1, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1), (1, 1, 1)]
    kept = 0
    n = 0
    while kept < 60 and n < 6000:
        n += 1
        kind = n % 4
        if kind == 0:
            h = np.diag(rng.uniform(4.0, 12.0, 3))
        elif kind == 1:
            L = rng.uniform(6.0, 14.0)
            h = np.array(
                [
                    [L, 0.0, 0.0],
                    [rng.uniform(-0.5, 0.5) * L, L, 0.0],
                    [rng.uniform(-0.5, 0.5) * L, rng.uniform(-0.5, 0.5) * L, L],
                ]
            )
        else:
            h = rng.uniform(-3.0, 3.0, (3, 3))
        pbc = patterns[n % len(patterns)]
        if abs(np.linalg.det(h)) < 1.0:
            continue
        try:
            h, _ = minkowski_reduce(h, pbc=np.array(pbc, dtype=bool))
            dr = rng.uniform(-6.0, 6.0, 3)
            brute = brute_len2(h, pbc, dr)
            if brute is None:
                continue
            _, lengths = find_mic(
                dr.reshape(1, 3), Cell(h), pbc=np.array(pbc, dtype=bool)
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        ase_len2 = float(lengths[0] ** 2)
        cells = " ".join(repr(float(x)) for x in np.asarray(h).ravel())
        drs = " ".join(repr(float(x)) for x in dr)
        print(
            f"{kept} {pbc[0]} {pbc[1]} {pbc[2]} {cells} {drs} "
            f"{repr(ase_len2)} {repr(brute)}"
        )
        kept += 1
    if kept < 40:
        print(f"mic_parity.py: only {kept} usable cases", file=sys.stderr)
        sys.exit(4)


main()
