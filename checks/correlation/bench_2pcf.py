"""Times tree-based pair counts on a periodic box and compares with Corrfunc."""
import argparse
import time
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp

from jzfmm.correlation import PairCountConfig, pair_counts, correlation_function

parser = argparse.ArgumentParser()
parser.add_argument("box", type=str, help="npz file with pos and boxsize")
parser.add_argument("--rmax", type=float, nargs="+", default=[10., 30.])
parser.add_argument("--nbins", type=int, default=20)
parser.add_argument("--subsample", type=int, default=1, help="Use a random 1/n of the particles")
parser.add_argument("--leaf", type=int, nargs="+", default=[64])
parser.add_argument("--alloc", type=float, default=None, help="alloc_fac_ilist, None for automatic")
parser.add_argument("--corrfunc", action="store_true", help="Compare with Corrfunc")
parser.add_argument("--nthreads", type=int, default=16)
parser.add_argument("--repeat", type=int, default=3)
args = parser.parse_args()

data = np.load(args.box)
boxsize = float(data["boxsize"])
# A random subsample, since every n-th particle in Lagrangian order is a regular sub-lattice
pos_np = data["pos"]
pos_np = pos_np[np.random.default_rng(0).choice(len(pos_np), len(pos_np) // args.subsample, replace=False)]
pos_np = np.ascontiguousarray(pos_np % boxsize, dtype=np.float32)
pos = jnp.asarray(pos_np)
print(f"N = {len(pos)}, boxsize = {boxsize}, mean separation = {boxsize / len(pos)**(1/3):.3f}")

def timeit(f):
    jax.block_until_ready(f())  # compile and warm up
    ts = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        res = jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return min(ts), res

for rmax in args.rmax:
    edges = np.geomspace(0.1, rmax, args.nbins + 1)
    for leaf in args.leaf:
        cfg = PairCountConfig(alloc_fac_ilist=args.alloc)
        cfg = replace(cfg, tree=replace(cfg.tree, max_leaf_size=leaf))
        t, dd = timeit(lambda: pair_counts(pos, edges, boxsize=boxsize, cfg=cfg))
        dd = np.asarray(dd)
        print(f"rmax={rmax:6.1f} leaf={leaf:4d}: {t*1e3:9.1f} ms, {dd.sum():.4g} pairs, "
              f"{dd.sum() / t / 1e9:.2f} Gpairs/s (in bins)")

    if args.corrfunc:
        from Corrfunc.theory.DD import DD
        t0 = time.perf_counter()
        res = DD(1, args.nthreads, edges, *pos_np.T, periodic=True, boxsize=boxsize)
        tc = time.perf_counter() - t0
        dd_cf = res["npairs"].astype(np.float64)
        rel = np.max(np.abs(dd - dd_cf) / np.maximum(dd_cf, 1))
        print(f"  Corrfunc ({args.nthreads} threads): {tc*1e3:9.1f} ms, max rel. diff {rel:.2e}, "
              f"max abs. diff {np.max(np.abs(dd - dd_cf)):.0f}")

xi = correlation_function(pos, edges, boxsize=boxsize)
r = np.sqrt(edges[1:] * edges[:-1])
print("r [Mpc/h] :", np.array2string(r, precision=2, max_line_width=200))
print("xi(r)     :", np.array2string(np.asarray(xi), precision=3, max_line_width=200))
