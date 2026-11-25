"""Finnis-Sinclair (1984) tungsten, numpy/scipy: energy, forces, relaxation.

Usage: python3 tools/fs_w.py IN.xyz OUT.xyz  (relax at fixed box), or `check`.

Used only to make relaxed vacancy-cluster fixtures for the method-B gate
(no LAMMPS available). Parameters: Finnis & Sinclair, Phil. Mag. A 50 (1984) 45.
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

d, A_, c = 4.400224, 1.896373, 3.25
c0, c1, c2 = 47.1346499, -33.7665655, 6.2541999


def pairs(x, L):
    m = np.mod(x, L)
    m[m >= L] = 0.0
    t = cKDTree(m, boxsize=L)
    p = t.query_pairs(d, output_type="ndarray")
    dr = x[p[:, 1]] - x[p[:, 0]]
    dr -= L * np.round(dr / L)
    r = np.linalg.norm(dr, axis=1)
    return p, dr, r


def energy_forces(x, L):
    n = len(x)
    p, dr, r = pairs(x, L)
    phi = np.where(r < d, (r - d) ** 2, 0.0)
    dphi = np.where(r < d, 2 * (r - d), 0.0)
    V = np.where(r < c, (r - c) ** 2 * (c0 + c1 * r + c2 * r * r), 0.0)
    dV = np.where(r < c, 2 * (r - c) * (c0 + c1 * r + c2 * r * r) + (r - c) ** 2 * (c1 + 2 * c2 * r), 0.0)
    rho = np.zeros(n)
    np.add.at(rho, p[:, 0], phi)
    np.add.at(rho, p[:, 1], phi)
    sr = np.sqrt(np.maximum(rho, 1e-30))
    E = V.sum() - A_ * sr.sum()
    g = dV - 0.5 * A_ * (1 / sr[p[:, 0]] + 1 / sr[p[:, 1]]) * dphi
    fvec = (g / r)[:, None] * dr  # dE/dx_j along dr; dE/dx_i = -that
    grad = np.zeros_like(x)
    np.add.at(grad, p[:, 1], fvec)
    np.add.at(grad, p[:, 0], -fvec)
    return E, grad


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    lat = lines[1].split('Lattice="')[1].split('"')[0].split()
    L = np.array([float(lat[0]), float(lat[4]), float(lat[8])])
    x = np.array([[float(v) for v in l.split()[1:4]] for l in lines[2:2 + n]])
    return x, L


def write_xyz(path, x, L):
    with open(path, "w") as f:
        f.write(f"{len(x)}\nLattice=\"{L[0]} 0 0 0 {L[1]} 0 0 0 {L[2]}\" pbc=\"T T T\" Properties=species:S:1:pos:R:3\n")
        for p in np.mod(x, L):
            f.write(f"W {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def relax(x, L, fmax=1e-3):
    def fun(v):
        E, g = energy_forces(v.reshape(-1, 3), L)
        return E, g.ravel()
    r = minimize(fun, x.ravel(), jac=True, method="L-BFGS-B",
                 options={"maxiter": 5000, "gtol": fmax, "maxcor": 30})
    xr = r.x.reshape(-1, 3)
    _, g = energy_forces(xr, L)
    return xr, np.abs(g).max(), r.nit


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for a in [3.15, 3.16, 3.1652, 3.17, 3.18]:
            n = 6
            ijk = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)], float)
            x = np.concatenate([ijk, ijk + 0.5]) * a
            E, g = energy_forces(x, np.array([n * a] * 3))
            print(a, E / len(x), np.abs(g).max())
    else:
        x, L = read_xyz(sys.argv[1])
        xr, fm, nit = relax(x, L)
        print(f"relaxed {len(x)} atoms in {nit} iterations, max |F| {fm:.2e} eV/A", file=sys.stderr)
        write_xyz(sys.argv[2], xr, L)
