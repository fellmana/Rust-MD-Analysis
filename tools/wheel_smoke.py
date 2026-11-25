"""Smoke test for an installed `rmda` wheel (run by .github/workflows/release.yml).

Imports the module from site-packages (run it from outside the source tree's
`crates/rmda-py`), writes a thermal BCC block with three vacancies, and checks
that PTM finds a BCC crystal and `rmda.defects` finds exactly the three
vacancies under the global and local references. Exits non-zero on failure.
"""

import os
import random
import sys
import tempfile

import rmda


def bcc_block(n, a, skip, sigma, seed):
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for b, off in ((0, 0.0), (1, 0.5)):
                    if (i, j, k, b) in skip:
                        continue
                    x, y, z = ((i + off) * a, (j + off) * a, (k + off) * a)
                    lines.append(
                        "W %.6f %.6f %.6f"
                        % tuple(v + rng.gauss(0.0, sigma) for v in (x, y, z))
                    )
    box = n * a
    head = (
        f'Lattice="{box} 0 0 0 {box} 0 0 0 {box}" pbc="T T T" '
        "Properties=species:S:1:pos:R:3"
    )
    return f"{len(lines)}\n{head}\n" + "\n".join(lines) + "\n"


def main():
    print(f"rmda {rmda.__version__} from {os.path.dirname(rmda.__file__)}")
    skip = {(3, 3, 3, 0), (5, 2, 6, 1), (8, 8, 1, 0)}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bcc.xyz")
        with open(path, "w") as fh:
            fh.write(bcc_block(10, 3.165, skip, 0.05, 1))
        f = rmda.read(path)
        for reference in ("global", "local"):
            res = rmda.defects(f, reference=reference)
            got = (res["defects_vacancies"], res["defects_interstitials"])
            print(f"defects --reference {reference}: V={got[0]} I={got[1]}")
            if got != (3, 0):
                sys.exit(f"expected 3 vacancies and 0 interstitials, got {got}")
    print("ok")


if __name__ == "__main__":
    main()
