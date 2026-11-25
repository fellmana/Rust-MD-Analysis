"""Method B gate on pure W: vacancy clusters of known size, unrelaxed,
thermal-noised, and relaxed with Finnis-Sinclair W (tools/fs_w.py), design 17.

Usage: python3 tools/void_b_gate.py target/release/rmda OUTDIR [SAMPLES]
(numpy, scipy). Writes the fixtures to OUTDIR (relaxed clusters are cached
there) and prints, per cluster and treatment, the analytic n*a^3/2 against
V(eps=0.01), the quadratic extrapolation to phi = 1 from eps 0.01, 0.02,
0.04, the excess over a defect-free frame with the same noise, and the
shell-free window (--thermal-window). The relaxed 15- and 89-site files are
crates/rmda-cli/tests/data/w_fs_relaxed_void{15,89}.xyz."""
import os, re, subprocess, sys
import numpy as np
import fs_w as F

RMDA = sys.argv[1]
OUT = sys.argv[2]
SAMPLES = sys.argv[3] if len(sys.argv) > 3 else "400000"
os.makedirs(OUT, exist_ok=True)
a = 3.1652
n = 10
L = np.array([n * a] * 3)
nn = a * np.sqrt(3) / 2
omega = a ** 3 / 2
ijk = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float)
perfect = np.concatenate([ijk, ijk + 0.5]) * a
centre = np.array([5 * a + 0.01] * 3)
rng = np.random.default_rng(5)


def run(args):
    r = subprocess.run([RMDA] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(1)
    return r.stdout


def vol(stdout, key):
    for l in stdout.splitlines():
        if l.startswith(key):
            m = re.search(r"([-0-9.]+) \+/- ([0-9.]+)", l)
            return float(m.group(1)), float(m.group(2))
    raise KeyError(key + "\n" + stdout)


common = ["void", "--experimental", "--lattice", "bcc", "--nn-distance", f"{nn}",
          "--samples", SAMPLES]
print("sites treatment          analytic   V(.01)        quadratic     excess        window (gap)")
for r in [0.1, 2.8, 4.0, 5.5, 7.0, None]:
    keep = np.linalg.norm(perfect - centre, axis=1) >= r if r else np.ones(len(perfect), bool)
    nv = int((~keep).sum())
    base = perfect[keep]
    variants = {"unrelaxed": base}
    for s in [0.03, 0.06]:
        variants[f"noise{s}"] = base + rng.normal(0, s, base.shape)
    if r:
        cache = os.path.join(OUT, f"relaxed{nv}.npy")
        if os.path.exists(cache):
            xr = np.load(cache)
        else:
            xr, fm, nit = F.relax(base.copy(), L)
            np.save(cache, xr)
        variants["relaxed"] = xr
        variants["relaxed+noise0.03"] = xr + rng.normal(0, 0.03, base.shape)
        variants["relaxed+noise0.06"] = xr + rng.normal(0, 0.06, base.shape)
    for tag, x in variants.items():
        p = os.path.join(OUT, f"v{nv}-{tag}.xyz")
        F.write_xyz(p, x, L)
        o = run(common + [p, "--epsilons", "0.01,0.02,0.04"])
        v1, _ = vol(o, "void volume:")
        vx, _ = vol(o, "extrapolated to phi = 1:")
        ow = run(common + [p, "--thermal-window"])
        vw, _ = vol(ow, "extrapolated to phi = 1:")
        gap = float(re.search(r"largest face gap: ([0-9.]+)", ow).group(1))
        # Excess over a defect-free crystal with the same treatment.
        sig = 0.06 if "0.06" in tag else 0.03 if "0.03" in tag else 0.0
        ref = perfect + rng.normal(0, sig, perfect.shape) if sig else perfect
        rp = os.path.join(OUT, f"ref-{tag}.xyz")
        F.write_xyz(rp, ref, L)
        o2 = run(common + [p, "--reference", rp, "--excess"])
        vxs, _ = vol(o2, "excess over reference:")
        want = nv * omega
        rel = lambda v: f"{100 * (v / want - 1):+7.1f}%" if nv else f"{v:7.1f}A3"
        print(f"{nv:5d} {tag:18s} {want:8.1f}  {v1:7.1f} {rel(v1)}  {rel(vx)}  {rel(vxs)}  "
              f"{vw:8.1f} {rel(vw)} ({gap:.3f})", flush=True)
