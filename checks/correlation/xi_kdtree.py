"""Simplest two-point correlation function of a periodic box, with scipy's KD-tree.

Usage:
    python xi_kdtree.py positions.npy --boxsize 500 --rmin 0.1 --rmax 30 --nbins 20

positions.npy holds an (N, 3) array (an .npz with a "pos" array also works).

Steps:
  1. DD: count the ordered pairs (i, j) with separation in each radial bin.
     cKDTree.count_neighbors(tree, r) returns, for every radius r, the number of ordered
     pairs with |x_i - x_j| <= r (nearest periodic image, self pairs i == j included).
     Differences between consecutive radii give the pairs per bin; the N self pairs at
     distance 0 cancel in these differences.
  2. RR: in a periodic box, uniformly random points have exactly
     RR_k = N (N - 1) * V_shell,k / L^3 ordered pairs per bin, so no random catalogue is needed.
  3. xi = DD / RR - 1 (natural estimator).
"""
import argparse
import time
import numpy as np
from scipy.spatial import cKDTree


def correlation_function(pos, r_edges, boxsize):
    n = len(pos)
    tree = cKDTree(pos, boxsize=boxsize)            # periodic tree
    cumulative = tree.count_neighbors(tree, r_edges)  # pairs with r <= r_edges[k]
    dd = np.diff(cumulative).astype(np.float64)

    shell_volume = 4. / 3. * np.pi * np.diff(r_edges**3)
    rr = n * (n - 1) * shell_volume / boxsize**3

    return dd / rr - 1., dd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("positions", help=".npy file with (N, 3) positions, or .npz with 'pos'")
    parser.add_argument("--boxsize", type=float, required=True)
    parser.add_argument("--rmin", type=float, default=0.1)
    parser.add_argument("--rmax", type=float, default=30.)
    parser.add_argument("--nbins", type=int, default=20)
    args = parser.parse_args()

    pos = np.load(args.positions)
    if isinstance(pos, np.lib.npyio.NpzFile):
        pos = pos["pos"]
    # The periodic tree needs positions in [0, boxsize)
    pos = np.mod(np.asarray(pos, dtype=np.float64), args.boxsize)
    pos[pos >= args.boxsize] = 0.

    r_edges = np.geomspace(args.rmin, args.rmax, args.nbins + 1)
    t0 = time.perf_counter()
    xi, dd = correlation_function(pos, r_edges, args.boxsize)
    print(f"N = {len(pos)}, {dd.sum():.4g} pairs, {time.perf_counter() - t0:.2f} s")
    print(f"{'r_lo':>9} {'r_hi':>9} {'DD':>14} {'xi':>11}")
    for lo, hi, d, x in zip(r_edges[:-1], r_edges[1:], dd, xi):
        print(f"{lo:9.3f} {hi:9.3f} {d:14.0f} {x:11.4f}")
