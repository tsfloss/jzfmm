"""Simulates a periodic cosmological box with DISCO-DJ.

Saves the z=0 positions together with the linear power spectrum at z=0.
"""
import argparse
import time
import numpy as np
import jax

from discodj import DiscoDJ

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=256, help="Particles per dimension")
parser.add_argument("--boxsize", type=float, default=500., help="Box size in Mpc/h")
parser.add_argument("--nsteps", type=int, default=10, help="Number of PM steps")
parser.add_argument("--out", type=str, default="discodj_box.npz")
args = parser.parse_args()

@jax.jit
def simulate():
    dj = DiscoDJ(dim=3, res=args.res, boxsize=args.boxsize).with_timetables()
    pk_state = dj.with_linear_ps()
    lpt = dj.with_lpt(dj.with_ics(pk_state), n_order=2)
    part = dj.run_lpt(lpt, a=0.05)
    part = dj.run_nbody(part, a_end=1., n_steps=args.nsteps, res_pm=2 * args.res,
                        stepper="bullfrog", adjoint_method=False)
    return part.pos, pk_state.k, dj.evaluate_linear_ps(pk_state, 1., pk_state.k)

t0 = time.perf_counter()
pos, k, pk = jax.block_until_ready(simulate())
pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
print(f"Simulated {len(pos)} particles in {time.perf_counter() - t0:.1f}s, "
      f"range [{pos.min():.3f}, {pos.max():.3f}]")
np.savez(args.out, pos=pos, boxsize=args.boxsize, k_lin=np.asarray(k), pk_lin=np.asarray(pk))
