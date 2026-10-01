"""Benchmarks periodic pair counts of jzfmm (GPU), Corrfunc (CPU) and scipy's cKDTree (CPU).

Usage: python bench_three_way.py box.npz --subsample 8 --rmax 10 30 --threads 1 10
"""
import argparse
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # exact float64 counts
import jax.numpy as jnp

from Corrfunc.theory.DD import DD
from jzfmm.correlation import pair_counts
from xi_kdtree import correlation_function as xi_kdtree

parser = argparse.ArgumentParser()
parser.add_argument("box", type=str, help="npz file with pos and boxsize")
parser.add_argument("--subsample", type=int, default=8, help="Use a random 1/n of the particles")
parser.add_argument("--rmax", type=float, nargs="+", default=[10., 30.])
parser.add_argument("--nbins", type=int, default=20)
parser.add_argument("--threads", type=int, nargs="+", default=[1, 10], help="Corrfunc threads")
args = parser.parse_args()

data = np.load(args.box)
boxsize = float(data["boxsize"])
pos_np = data["pos"]
pos_np = pos_np[np.random.default_rng(0).choice(len(pos_np), len(pos_np) // args.subsample, replace=False)]
pos_np = np.ascontiguousarray(pos_np % boxsize, dtype=np.float32)
pos_np[pos_np >= boxsize] = 0.
pos = jnp.asarray(pos_np)
print(f"N = {len(pos_np)}, boxsize = {boxsize}")

def timed(f, repeat=1):
    ts, res = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = f()
        ts.append(time.perf_counter() - t0)
    return min(ts), res

for rmax in args.rmax:
    edges = np.geomspace(0.1, rmax, args.nbins + 1)
    rows = []

    f_gpu = lambda: np.asarray(pair_counts(pos, edges, boxsize=boxsize))
    f_gpu()  # compile
    rows.append(("jzfmm, GPU", *timed(f_gpu, 5)))
    for nt in args.threads:
        f_cf = lambda: DD(1, nt, edges, *pos_np.T, periodic=True, boxsize=boxsize)["npairs"]
        rows.append((f"Corrfunc, {nt} thread{'s' * (nt > 1)}", *timed(f_cf, 2 if nt > 1 else 1)))
    pos64 = pos_np.astype(np.float64)
    rows.append(("scipy cKDTree, 1 thread", *timed(lambda: xi_kdtree(pos64, edges, boxsize)[1])))

    dd_ref = rows[0][2]
    print(f"\nrmax = {rmax}: {dd_ref.sum():.3g} pairs in bins")
    print(f"{'method':<26} {'time [s]':>9} {'vs jzfmm':>9} {'max rel. DD diff to jzfmm':>26}")
    for name, t, dd in rows:
        rel = np.max(np.abs(dd - dd_ref) / np.maximum(dd_ref, 1))
        print(f"{name:<26} {t:9.3f} {t / rows[0][1]:9.1f} {rel:26.1e}", flush=True)
