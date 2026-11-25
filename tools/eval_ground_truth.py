#!/usr/bin/env python3
"""Score `rmda defects` against a `gen_ground_truth.py` truth file.

Runs `rmda defects FRAME --defects DIR [ARGS...]` for each configuration
given (or reads existing defect directories), assigns every vacancy and
interstitial found to the nearest feature whose sphere (radius + 1.5 bonds)
holds it, else counts it as spurious, and prints per configuration the
totals against the truth, the error per feature kind, and the spurious
defects (Frenkel pairs where there is none).

Usage:
  tools/eval_ground_truth.py --rmda target/release/rmda FRAME.xyz \\
      --config global "--reference global" --config local "--reference local"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

BOND = 3.165 * 3**0.5 / 2


def read_positions(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    if not lines:
        return np.zeros((0, 3))
    n = int(lines[0])
    head = lines[1]
    props = head.split("Properties=")[1].split()[0].split(":")
    col, at = None, 0
    for name, _, width in zip(props[0::3], props[1::3], props[2::3]):
        if name == "pos":
            col = at
            break
        at += int(width)
    rows = [l.split() for l in lines[2 : 2 + n]]
    return np.array([[float(r[col + k]) for k in range(3)] for r in rows]).reshape(-1, 3)


def assign(points: np.ndarray, centres: np.ndarray, radii: np.ndarray, bond: float = BOND) -> np.ndarray:
    """Feature index of each point (nearest feature whose sphere, grown by
    1.5 bonds, holds it), -1 when none does."""
    if len(points) == 0:
        return np.zeros(0, dtype=int)
    d = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2) - (radii + 1.5 * bond)[None, :]
    best = np.argmin(d, axis=1)
    return np.where(d[np.arange(len(points)), best] <= 0.0, best, -1)


def score(truth: dict, ddir: Path) -> dict:
    feats = truth["features"]
    centres = np.array([f["centre"] for f in feats])
    radii = np.array([f["radius"] for f in feats])
    v = read_positions(ddir / "vacancies.xyz")
    i = read_positions(ddir / "interstitials.xyz")
    bond = truth.get("nn", BOND)
    av, ai = assign(v, centres, radii, bond), assign(i, centres, radii, bond)
    by_kind: dict[str, dict] = {}
    exact = 0
    for k, f in enumerate(feats):
        fv, fi = int((av == k).sum()), int((ai == k).sum())
        e = by_kind.setdefault(f["kind"], {"n": 0, "exact": 0, "dV": 0, "dI": 0, "absV": 0, "absI": 0,
                                           "V": 0, "I": 0})
        e["n"] += 1
        e["V"] += f["vacancies"]
        e["I"] += f["interstitials"]
        e["dV"] += fv - f["vacancies"]
        e["dI"] += fi - f["interstitials"]
        e["absV"] += abs(fv - f["vacancies"])
        e["absI"] += abs(fi - f["interstitials"])
        ok = fv == f["vacancies"] and fi == f["interstitials"]
        e["exact"] += ok
        exact += ok
    return {
        "V": len(v), "I": len(i),
        "spurious_V": int((av < 0).sum()), "spurious_I": int((ai < 0).sum()),
        "features_exact": exact, "features": len(feats), "by_kind": by_kind,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", type=Path)
    ap.add_argument("--rmda", type=Path, default=Path("target/release/rmda"))
    ap.add_argument("--config", nargs=2, action="append", metavar=("NAME", "ARGS"), default=[])
    ap.add_argument("--keep", type=Path, help="keep defect directories under this one")
    ap.add_argument("--json", type=Path, help="write the scores here")
    args = ap.parse_args(argv)
    truth = json.loads(args.frame.with_suffix(".truth.json").read_text())
    tv, ti = truth["totals"]["vacancies"], truth["totals"]["interstitials"]
    base = args.keep or Path(tempfile.mkdtemp(prefix="rmda-gt-"))
    print(f"{args.frame.name}: {truth.get('structure', 'bcc')}, {truth['atoms']} atoms, truth V={tv} I={ti}, "
          f"{len(truth['features'])} features, sigma {truth['sigma']}, bend {truth['bend']}")
    out = {}
    for name, extra in args.config:
        ddir = base / name
        cmd = [str(args.rmda), "defects", str(args.frame), "--defects", str(ddir), *shlex.split(extra)]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if p.returncode != 0:
            print(f"{name}: exit {p.returncode}\n{p.stderr}", file=sys.stderr)
            continue
        measured = [l for l in p.stdout.splitlines() if l.startswith("defects_grains_measured:")]
        s = score(truth, ddir)
        s["wall_s"] = round(wall, 1)
        s["grains_measured"] = measured[0].split()[-1] if measured else "?"
        s["reference_line"] = [l for l in p.stderr.splitlines() if "local census" in l or "counting it" in l]
        s["stdout"] = p.stdout
        s["stderr"] = p.stderr
        out[name] = s
        print(f"\n{name}: V={s['V']} ({s['V'] - tv:+d}) I={s['I']} ({s['I'] - ti:+d}) "
              f"spurious V={s['spurious_V']} I={s['spurious_I']}  features exact "
              f"{s['features_exact']}/{s['features']}  measured {s['grains_measured']}  {wall:.1f} s")
        for k, e in sorted(s["by_kind"].items()):
            print(f"  {k:18s} n={e['n']:3d} exact={e['exact']:3d}  V {e['V']:5d} {e['dV']:+5d} "
                  f"(|{e['absV']}|)  I {e['I']:5d} {e['dI']:+5d} (|{e['absI']}|)")
    if args.json:
        args.json.write_text(json.dumps({"truth": truth["totals"], "scores": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
