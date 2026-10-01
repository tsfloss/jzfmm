"""Measures xi(r) of a periodic box with jzfmm and Corrfunc and plots both, incl. the BAO bump.

Usage: python plot_2pcf.py box.npz --subsample 4 --nthreads 11 --out xi.png
The box file is created with make_discodj_box.py.
"""
import argparse
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # exact float64 counts
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jzfmm.correlation import correlation_function

parser = argparse.ArgumentParser()
parser.add_argument("box", type=str)
parser.add_argument("--subsample", type=int, default=4, help="Use a random 1/n of the particles")
parser.add_argument("--rmax", type=float, default=200.)
parser.add_argument("--nthreads", type=int, default=11, help="Corrfunc OpenMP threads")
parser.add_argument("--out", type=str, default="xi.png")
args = parser.parse_args()

data = np.load(args.box)
boxsize = float(data["boxsize"])
# A random subsample, since every n-th particle in Lagrangian order is a regular sub-lattice
pos_np = data["pos"]
pos_np = pos_np[np.random.default_rng(0).choice(len(pos_np), len(pos_np) // args.subsample, replace=False)]
pos_np = np.ascontiguousarray(pos_np % boxsize, dtype=np.float32)
n = len(pos_np)
print(f"N = {n}, boxsize = {boxsize}")

# Logarithmic bins on small scales, linear bins around the BAO
edges = np.unique(np.concatenate([np.geomspace(0.5, 40., 17), np.arange(40., args.rmax + 1e-6, 4.)]))
r = 0.5 * (edges[1:] + edges[:-1])

# --- jzfmm (GPU) ---
pos = jnp.asarray(pos_np)
jax.block_until_ready(correlation_function(pos, edges, boxsize))
t0 = time.perf_counter()
xi_jz, dd_jz, _ = correlation_function(pos, edges, boxsize, return_counts=True)
xi_jz, dd_jz = np.asarray(xi_jz, np.float64), np.asarray(dd_jz, np.float64)
t_jz = time.perf_counter() - t0

# --- Corrfunc (CPU), its own xi with analytic randoms ---
from Corrfunc.theory.xi import xi as corrfunc_xi
t0 = time.perf_counter()
res = corrfunc_xi(boxsize, args.nthreads, edges, *pos_np.T)
t_cf = time.perf_counter() - t0
xi_cf, dd_cf = res["xi"], res["npairs"].astype(np.float64)
print(f"jzfmm: {t_jz:.2f} s, Corrfunc ({args.nthreads} threads): {t_cf:.2f} s, "
      f"{dd_jz.sum():.3g} pairs, max |DD diff| = {np.max(np.abs(dd_jz - dd_cf)):.0f}, "
      f"max |xi diff| = {np.max(np.abs(xi_jz - xi_cf)):.2e}")

# --- Linear theory, xi(r) = 1/(2 pi^2) int k^2 P(k) j0(kr) dk, damped for convergence ---
k, pk = data["k_lin"], data["pk_lin"]
kk = np.geomspace(1e-4, 10., 20000)
pkk = np.exp(np.interp(np.log(kk), np.log(k), np.log(pk))) * np.exp(-kk**2)
rl = np.linspace(0.5, args.rmax, 800)
xi_lin = np.trapezoid(kk**3 * pkk * np.sinc(np.outer(rl, kk) / np.pi), np.log(kk), axis=1) / (2 * np.pi**2)

np.savez(args.out.rsplit(".", 1)[0] + ".npz", edges=edges, r=r, xi_jzfmm=xi_jz, xi_corrfunc=xi_cf,
         dd_jzfmm=dd_jz, dd_corrfunc=dd_cf, t_jzfmm=t_jz, t_corrfunc=t_cf, n=n, boxsize=boxsize)

# --- Plot ---
ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
c_jz, c_cf = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink, "xtick.color": muted,
    "ytick.color": muted, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex="col", height_ratios=(3, 1),
                         layout="constrained")

def series(ax, x, y_jz, y_cf, xl, yl):
    ax.plot(xl, yl, color=muted, lw=1.2, ls="--", label="linear theory (z=0)", zorder=1)
    ax.plot(x, y_jz, color=c_jz, lw=2, label=f"jzfmm, GPU ({t_jz:.1f} s)", zorder=2)
    ax.plot(x, y_cf, ls="none", marker="o", ms=5, mfc="none", mec=c_cf, mew=1.3,
            label=f"Corrfunc, {args.nthreads} CPU threads ({t_cf:.1f} s)", zorder=3)

# Left: small scales, log-log
small = r < 60
ax = axes[0, 0]
series(ax, r[small], xi_jz[small], xi_cf[small], rl[rl < 60], xi_lin[rl < 60])
ax.set(xscale="log", yscale="log", ylabel=r"$\xi(r)$", title="Small scales")
ax.legend(frameon=False, loc="lower left")

# Right: BAO, r^2 xi
bao = r >= 40
ax = axes[0, 1]
series(ax, r[bao], r[bao]**2 * xi_jz[bao], r[bao]**2 * xi_cf[bao], rl[rl >= 40], rl[rl >= 40]**2 * xi_lin[rl >= 40])
ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
ax.set(ylabel=r"$r^2\,\xi(r)$ [$h^{-2}\,\mathrm{Mpc}^2$]", title="BAO scale")

# Bottom: relative difference of the pair counts
for ax, m in ((axes[1, 0], small), (axes[1, 1], bao)):
    rel = (dd_jz[m] - dd_cf[m]) / dd_cf[m]
    ax.plot(r[m], rel, color=c_jz, lw=0, marker="o", ms=3.5)
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
    lim = max(1e-8, 1.3 * np.max(np.abs(rel)))
    ax.set(ylim=(-lim, lim), xlabel=r"$r$ [$h^{-1}\,\mathrm{Mpc}$]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axes[1, 0].set_ylabel(r"$DD_\mathrm{jzfmm}/DD_\mathrm{Corrfunc}-1$")

fig.suptitle(f"Two-point correlation function, DISCO-DJ box: {boxsize:.0f} $h^{{-1}}$Mpc, "
             f"{n/1e6:.1f}M particles, z=0", color=ink)
fig.savefig(args.out, dpi=150)
print("saved", args.out)
