"""Benchmark of Wendland C2 kernel density estimates of N-body particles at galaxy positions.

Compares the two methods of evaluate_at_positions:
  two_type   sources and queries are separate particle types of one tree
  zero_mass  queries are zero-mass particles of a single-type tree (query_mask)

The N-body particles follow a simple halo model in a periodic unit box: Hernquist halos with a
power-law mass function plus a uniform background. Galaxies follow an HOD-like model: centrals
at the centres of the most massive halos and satellites drawn from the particles of those
halos, so they sit in the densest regions of the particle distribution.

Every configuration runs in a fresh subprocess, so that an out-of-memory error or a CUDA error
only affects that configuration.

Usage:
    python bench_kde_galaxies.py                     # default sweep, prints a table
    python bench_kde_galaxies.py --out results.json  # also stores the results
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


# ------------------------------------------------------------------------------------------------ #
#                                        Initial conditions                                        #
# ------------------------------------------------------------------------------------------------ #

def halo_model(nbody, seed=0, f_halo=0.6, nmin=20, nmax_frac=3e-3, slope=1.9, conc=5.):
    """Clustered particles in the unit box. Returns positions, halo id (-1 = background) and
    halo centres / particle counts sorted by decreasing mass.

    Halo particle numbers follow dn/dN ~ N^-slope between nmin and nmax_frac * nbody, until
    a fraction f_halo of the particles is in halos. Halos have Hernquist profiles truncated at
    r_200 (200 times the mean density) with concentration r_200 / a = conc.
    """
    rng = np.random.default_rng(seed)
    nmax = max(nmin + 1, nmax_frac * nbody)
    a = 1. - slope
    nh = []
    while sum(nh) < f_halo * nbody:
        u = rng.uniform(size=4096)
        nh.extend((nmin**a + u * (nmax**a - nmin**a))**(1. / a))
    nh = np.sort(np.floor(np.asarray(nh)).astype(np.int64))[::-1]
    nh = nh[np.cumsum(nh) <= f_halo * nbody]
    nhalo, nin = len(nh), int(nh.sum())
    cen = rng.uniform(size=(nhalo, 3))
    r200 = (3. * nh / (4. * np.pi * 200. * nbody))**(1. / 3.)
    scale = r200 / conc
    hid = np.repeat(np.arange(nhalo), nh)
    # Hernquist M(<r) ~ r^2/(r+a)^2, truncated at r_200
    uu = rng.uniform(size=nin) * (conc / (1. + conc))**2
    s = np.sqrt(uu)
    r = scale[hid] * s / (1. - s)
    d = rng.normal(size=(nin, 3))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    pos_halo = cen[hid] + r[:, None] * d
    pos_bg = rng.uniform(size=(nbody - nin, 3))
    pos = np.concatenate([pos_halo, pos_bg]) % 1.
    hid = np.concatenate([hid, np.full(nbody - nin, -1)])
    return pos.astype(np.float32), hid, cen, nh

def galaxies(pos, hid, cen, nh, ngal, seed=1, f_sat=0.3):
    """Centrals at the centres of the most massive halos, satellites at particles of halos."""
    rng = np.random.default_rng(seed)
    ncen = min(int(round((1. - f_sat) * ngal)), len(cen))
    nsat = ngal - ncen
    gal_cen = cen[:ncen]
    # Satellites: particles of the halos hosting centrals, number ~ halo mass
    in_hosts = np.nonzero((hid >= 0) & (hid < ncen))[0]
    isat = rng.choice(in_hosts, size=nsat, replace=nsat > len(in_hosts))
    gal = np.concatenate([gal_cen, pos[isat]]) % 1.
    return gal.astype(np.float32)


# ------------------------------------------------------------------------------------------------ #
#                                         Single benchmark                                         #
# ------------------------------------------------------------------------------------------------ #

def run_single(nbody, ngal, ngb, methods, rounds, alloc_fac_ilist, alloc_fac_nodes, grad, seed=0):
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from jzfmm.config import FMMConfig, OpeningBySupport, WendlandC2Kernel
    from jzfmm.data import PosMass
    from jzfmm.fmm import evaluate_at_positions

    pos, hid, cen, nh = halo_model(nbody, seed=seed)
    gal = galaxies(pos, hid, cen, nh, ngal, seed=seed + 1)
    # Support with ngb neighbours at the mean density
    support = float((ngb / (4. / 3. * np.pi * nbody))**(1. / 3.))
    kernel = WendlandC2Kernel(support=support, dim=3, boxsize=1.)
    cfg = FMMConfig(kernel=kernel, p=1, opening=OpeningBySupport(support=support, boxsize=1.),
                    remove_self_interaction=False, alloc_fac_ilist=alloc_fac_ilist)
    # Clustered particles need more tree nodes than the default allocation
    cfg = replace(cfg, tree=replace(cfg.tree, alloc_fac_nodes=alloc_fac_nodes))

    pos = jnp.asarray(pos)
    gal = jnp.asarray(gal)
    mass = jnp.full(nbody, 1. / nbody, dtype=jnp.float32)

    out = dict(nbody=nbody, ngal=ngal, ngb=ngb, support=support, results={})
    ref = None
    for method in methods:
        def density(pos, mass, gal):
            return evaluate_at_positions(PosMass(pos=pos, mass=mass), gal, cfg, method=method).potential()

        fns = {"fwd": jax.jit(density)}
        if grad:
            # Gradient of a loss of the densities w.r.t. the particle and galaxy positions
            fns["grad"] = jax.jit(jax.grad(lambda p, m, g: jnp.sum(density(p, m, g)**2), argnums=(0, 2)))

        res = {}
        for name, fn in fns.items():
            try:
                t0 = time.perf_counter()
                y = jax.block_until_ready(fn(pos, mass, gal))
                t_first = time.perf_counter() - t0
                ts = []
                for _ in range(rounds):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(pos, mass, gal))
                    ts.append(time.perf_counter() - t0)
                res[name] = dict(median=float(np.median(ts)), min=float(np.min(ts)), first=t_first)
                if name == "fwd":
                    rho = np.asarray(y)
                    res["rho_mean_over_mean"] = float(rho.mean())  # mean density is 1
                    res["finite"] = bool(np.isfinite(rho).all())
                    if ref is None:
                        ref = rho
                    else:
                        res["max_rel_diff_to_first"] = float(
                            np.max(np.abs(rho - ref)) / np.max(np.abs(ref))
                        )
            except Exception as e:  # e.g. out of memory or ilist too small
                msg = str(e).splitlines()
                key = [l for l in msg if "allocation too small" in l or "out of memory" in l.lower()]
                res[name] = dict(error=f"{type(e).__name__}: {(key or msg)[0][:200]}")
        out["results"][method] = res
    return out


# ------------------------------------------------------------------------------------------------ #
#                                              Driver                                              #
# ------------------------------------------------------------------------------------------------ #

def fmt_time(r, key):
    if key not in r:
        return "-"
    if "error" in r[key]:
        return "ERR"
    return f"{1e3 * r[key]['median']:.1f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbody", type=int, nargs="+", default=[2**21])
    ap.add_argument("--ngal", type=int, nargs="+", default=[10**3, 10**4, 10**5, 10**6])
    ap.add_argument("--ngb", type=int, nargs="+", default=[32, 256])
    ap.add_argument("--methods", nargs="+", default=["two_type", "zero_mass"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--alloc-fac-ilist", type=float, default=256.)
    ap.add_argument("--alloc-fac-nodes", type=float, default=2.5)
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single is not None:
        kw = json.loads(args.single)
        print("RESULT " + json.dumps(run_single(**kw)), flush=True)
        return

    all_res = []
    hdr = (f"{'nbody':>9} {'ngal':>9} {'ngb':>5} | " + " | ".join(
        f"{m + ' fwd':>15} {m + ' grad':>16}" for m in args.methods) + " | speedup fwd / grad")
    print(hdr)
    print("-" * len(hdr))
    for nbody in args.nbody:
        for ngb in args.ngb:
            for ngal in args.ngal:
                kw = dict(nbody=nbody, ngal=ngal, ngb=ngb, methods=args.methods, rounds=args.rounds,
                          alloc_fac_ilist=args.alloc_fac_ilist, alloc_fac_nodes=args.alloc_fac_nodes,
                          grad=not args.no_grad)
                p = subprocess.run([sys.executable, __file__, "--single", json.dumps(kw)],
                                   capture_output=True, text=True, env=os.environ)
                lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                if not lines:
                    err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
                    print(f"{nbody:>9} {ngal:>9} {ngb:>5} | subprocess failed: {err}")
                    all_res.append(dict(kw, error=err))
                    continue
                r = json.loads(lines[0][len("RESULT "):])
                all_res.append(r)
                rr = r["results"]
                cols = " | ".join(f"{fmt_time(rr[m], 'fwd'):>15} {fmt_time(rr[m], 'grad'):>16}"
                                  for m in args.methods)
                speed = ""
                if len(args.methods) == 2:
                    a, b = (rr[m] for m in args.methods)
                    sp = []
                    for k in ("fwd", "grad"):
                        try:
                            sp.append(f"{b[k]['median'] / a[k]['median']:.2f}x")
                        except (KeyError, TypeError):
                            sp.append("-")
                    speed = " / ".join(sp)
                print(f"{nbody:>9} {ngal:>9} {ngb:>5} | {cols} | {speed}", flush=True)
                for m in args.methods:
                    if "max_rel_diff_to_first" in rr[m]:
                        d = rr[m]["max_rel_diff_to_first"]
                        if not d < 1e-4:
                            print(f"    WARNING: {m} differs from {args.methods[0]} by {d:.2e}")
                    for k in ("fwd", "grad"):
                        if k in rr[m] and "error" in rr[m][k]:
                            print(f"    {m} {k}: {rr[m][k]['error']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_res, f, indent=1)

if __name__ == "__main__":
    main()
