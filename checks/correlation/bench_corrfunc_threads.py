"""Compares tree-based GPU pair counts with Corrfunc for several CPU thread counts."""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

from Corrfunc.theory.DD import DD
from jzfmm.correlation import PairCountConfig, pair_counts

parser = argparse.ArgumentParser()
parser.add_argument("box", type=str, help="npz file with pos and boxsize")
parser.add_argument("--rmax", type=float, nargs="+", default=[10., 30.])
parser.add_argument("--nbins", type=int, default=20)
parser.add_argument("--subsample", type=int, default=8)
parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 11])
parser.add_argument("--leaf", type=int, default=64)
parser.add_argument("--alloc", type=float, default=None, help="alloc_fac_ilist, None for automatic")
args = parser.parse_args()

data = np.load(args.box)
boxsize = float(data["boxsize"])
# A random subsample, since every n-th particle in Lagrangian order is a regular sub-lattice
pos_np = data["pos"]
pos_np = pos_np[np.random.default_rng(0).choice(len(pos_np), len(pos_np) // args.subsample, replace=False)]
pos_np = np.ascontiguousarray(pos_np % boxsize, dtype=np.float32)
pos = jnp.asarray(pos_np)
cfg = PairCountConfig(alloc_fac_ilist=args.alloc)
cfg = replace(cfg, tree=replace(cfg.tree, max_leaf_size=args.leaf))
print(f"N = {len(pos)}, boxsize = {boxsize}")

def best_time(f, repeat):
    ts, res = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = f()
        ts.append(time.perf_counter() - t0)
    return min(ts), res

for rmax in args.rmax:
    edges = np.geomspace(0.1, rmax, args.nbins + 1)
    f_gpu = lambda: np.asarray(pair_counts(pos, edges, boxsize=boxsize, cfg=cfg))
    f_gpu()
    t_gpu, dd = best_time(f_gpu, 5)
    print(f"\nrmax = {rmax}: GPU {t_gpu*1e3:.1f} ms ({dd.sum():.3g} pairs)")
    print(f"{'threads':>8} {'Corrfunc [ms]':>14} {'speedup vs 1':>13} {'GPU speedup':>12} {'max |diff|':>11}")
    t1 = None
    for nt in args.threads:
        f_cf = lambda: DD(1, nt, edges, *pos_np.T, periodic=True, boxsize=boxsize)["npairs"]
        t_cf, dd_cf = best_time(f_cf, 1 if nt == 1 else 2)
        t1 = t1 or t_cf
        print(f"{nt:8d} {t_cf*1e3:14.1f} {t1/t_cf:13.2f} {t_cf/t_gpu:12.1f} "
              f"{np.max(np.abs(dd - dd_cf)):11.0f}", flush=True)
