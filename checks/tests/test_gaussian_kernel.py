"""Kernel density estimates with the Gaussian kernel and OpeningByGaussianError."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jzfmm.config import DirectSummationConfig, FMMConfig, GaussianKernel, OpeningByAngle
from jzfmm.config import OpeningByGaussianError, OpeningBySupport, PlummerKernel
from jzfmm.data import PosMass
from jzfmm.fmm import direct_summation, evaluate_at_positions, fast_multipole_method

# ------------------------------------------------------------------------------------------------ #
#                                             Helpers                                              #
# ------------------------------------------------------------------------------------------------ #

def _density_ref(x, pos, mass, sigma, dim, boxsize=None):
    dx = x[:, None, :] - pos[None, :, :]
    if boxsize is not None:
        dx = dx - boxsize * jnp.round(dx / boxsize)
    r2 = jnp.sum(dx**2, axis=-1)
    norm = (2. * math.pi * sigma**2)**(-dim / 2)
    return norm * jnp.sum(mass[None, :] * jnp.exp(-r2 / (2. * sigma**2)), axis=-1)

def _clustered(nbody, dim, seed=0, dtype=jnp.float32):
    """Clustered particles in [0,1]^dim with unit total mass, i.e. unit mean density."""
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    ncl = 16
    centers = jax.random.uniform(k1, (ncl, dim), dtype=dtype)
    which = jax.random.randint(k2, (nbody // 2,), 0, ncl)
    clustered = centers[which] + 0.03 * jax.random.normal(k3, (nbody // 2, dim), dtype=dtype)
    uniform = jax.random.uniform(k4, (nbody - nbody // 2, dim), dtype=dtype)
    pos = jnp.concatenate([clustered, uniform]) % 1.
    return PosMass(pos=pos, mass=jnp.full(nbody, 1. / nbody, dtype=dtype))

def _queries(n, dim, part, seed=1):
    """Half near particles (clustered), half uniform (randoms)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    near = part.pos[:n // 2] + 0.002 * jax.random.normal(k1, (n // 2, dim), dtype=part.pos.dtype)
    uniform = jax.random.uniform(k2, (n - n // 2, dim), dtype=part.pos.dtype)
    return jnp.concatenate([near, uniform]) % 1.

# ------------------------------------------------------------------------------------------------ #
#                                               Tests                                              #
# ------------------------------------------------------------------------------------------------ #

@pytest.mark.parametrize("dim", (2, 3))
def test_direct_vs_reference(dim):
    sigma = 0.05
    part = _clustered(1024, dim)
    cfg = DirectSummationConfig(kernel=GaussianKernel(sigma, dim), remove_self_interaction=False)
    loc = direct_summation.jit(part, cfg_direct=cfg)
    rho = lambda x: _density_ref(x, part.pos, part.mass, sigma, dim)
    rho_ref = rho(part.pos)
    grad_ref = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(part.pos)
    np.testing.assert_allclose(loc.potential(), rho_ref, rtol=1e-4)
    np.testing.assert_allclose(loc.force(), -grad_ref, rtol=1e-3,
                               atol=1e-5 * float(jnp.max(jnp.abs(grad_ref))))

@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("p", (2, 4, 6))
@pytest.mark.parametrize("dim", (2, 3))
def test_error_criterion_within_tolerance(dim, p, periodic):
    """Errors stay below tol, relative to the mean density 1, also at low-density queries."""
    sigma = 0.02
    boxsize = 1. if periodic else None
    part = _clustered(2**15, dim)
    q = _queries(2**11, dim, part)
    rho_ref = _density_ref(q, part.pos, part.mass, sigma, dim, boxsize)
    for tol in (1e-2, 1e-4):
        cfg = FMMConfig(kernel=GaussianKernel(sigma, dim, boxsize=boxsize),
                        opening=OpeningByGaussianError(tol), p=p)
        rho = evaluate_at_positions.jit(part, q, cfg_fmm=cfg).potential()
        err = np.abs(np.asarray(rho) - np.asarray(rho_ref))
        # float32 accumulation limits the accuracy at the tighter tolerance
        assert err.max() < max(tol, 2e-5 * float(rho_ref.max())), (tol, err.max())
        assert np.all(np.asarray(rho) > 0)

@pytest.mark.parametrize("dim", (2, 3))
def test_error_criterion_converges(dim):
    with jax.enable_x64(True):
        sigma = 0.02
        part = _clustered(2**14, dim, dtype=jnp.float64)
        q = _queries(2**10, dim, part)
        rho_ref = _density_ref(q, part.pos, part.mass, sigma, dim)
        errs = []
        for tol in (1e-3, 1e-6, 1e-9):
            cfg = FMMConfig(kernel=GaussianKernel(sigma, dim), opening=OpeningByGaussianError(tol), p=4)
            rho = evaluate_at_positions.jit(part, q, cfg_fmm=cfg).potential()
            errs.append(float(jnp.max(jnp.abs(rho - rho_ref))))
            assert errs[-1] < tol
        assert errs[2] < 1e-3 * errs[0]

@pytest.mark.parametrize("periodic", (False, True))
def test_gradients_wrt_nbody_and_queries(periodic):
    """Gradients w.r.t. N-body positions and masses and query positions match the reference."""
    with jax.enable_x64(True):
        dim, sigma = 3, 0.03
        boxsize = 1. if periodic else None
        part = _clustered(4096, dim, dtype=jnp.float64)
        q = _queries(256, dim, part)
        w = jax.random.normal(jax.random.PRNGKey(5), (q.shape[0],))
        cfg = FMMConfig(kernel=GaussianKernel(sigma, dim, boxsize=boxsize),
                        opening=OpeningByGaussianError(1e-9), p=4)

        def loss(part, q):
            return jnp.sum(w * jnp.log(evaluate_at_positions(part, q, cfg_fmm=cfg).potential()))

        def loss_ref(pos, mass, q):
            return jnp.sum(w * jnp.log(_density_ref(q, pos, mass, sigma, dim, boxsize)))

        gp, gq = jax.jit(jax.grad(loss, argnums=(0, 1)))(part, q)
        gpos_ref, gmass_ref, gq_ref = jax.jit(jax.grad(loss_ref, argnums=(0, 1, 2)))(part.pos, part.mass, q)
        for name, g, ref in (("pos", gp.pos, gpos_ref), ("mass", gp.mass, gmass_ref), ("query", gq, gq_ref)):
            np.testing.assert_allclose(g, ref, rtol=1e-6, atol=1e-7 * float(jnp.max(jnp.abs(ref))),
                                       err_msg=name)

def test_self_interaction_removed():
    dim, sigma = 3, 0.05
    part = _clustered(2**13, dim)
    kernel = GaussianKernel(sigma, dim)
    ref = direct_summation.jit(part, cfg_direct=DirectSummationConfig(kernel=kernel)).potential()
    cfg = FMMConfig(kernel=kernel, opening=OpeningByGaussianError(1e-5), p=4)
    rho = fast_multipole_method.jit(part, cfg_fmm=cfg).potential()
    np.testing.assert_allclose(rho, ref, rtol=1e-4, atol=2e-5)

def test_truncated_by_support():
    dim, sigma = 3, 0.02
    part = _clustered(2**14, dim)
    q = _queries(2**10, dim, part)
    cfg = FMMConfig(kernel=GaussianKernel(sigma, dim, boxsize=1.),
                    opening=OpeningBySupport(5 * sigma, boxsize=1.), p=1)
    rho = evaluate_at_positions.jit(part, q, cfg_fmm=cfg).potential()
    rho_ref = _density_ref(q, part.pos, part.mass, sigma, dim, 1.)
    # Neglects at most exp(-12.5) of the peak value per particle
    np.testing.assert_allclose(rho, rho_ref, rtol=1e-4, atol=1e-4)

def test_config_validation():
    with pytest.raises(ValueError):
        GaussianKernel(sigma=0.)
    with pytest.raises(ValueError):
        GaussianKernel(sigma=0.1, boxsize=1.)  # boxsize must exceed 12 sigma
    with pytest.raises(ValueError):
        OpeningByGaussianError(tol=0.)
    part = _clustered(256, 3)
    q = part.pos[:8]
    bad = [
        FMMConfig(kernel=PlummerKernel(), opening=OpeningByGaussianError(1e-3)),
        FMMConfig(kernel=GaussianKernel(0.05, dim=2), opening=OpeningByGaussianError(1e-3)),
        FMMConfig(kernel=GaussianKernel(0.05, boxsize=1.), opening=OpeningByAngle()),
        FMMConfig(kernel=GaussianKernel(0.05, boxsize=1.), opening=OpeningBySupport(0.2)),
    ]
    for cfg in bad:
        with pytest.raises(ValueError):
            evaluate_at_positions(part, q, cfg_fmm=cfg)
