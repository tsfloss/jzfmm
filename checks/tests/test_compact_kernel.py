"""Kernel density estimates with the compact Wendland C2 kernel and OpeningBySupport."""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import math
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from jztree.tree import _dense_interaction_list

from jzfmm.config import DirectSummationConfig, FMMConfig, OpeningByAngle, OpeningBySupport
from jzfmm.config import WendlandC2Kernel
from jzfmm.data import PosMass
from jzfmm.fmm import _leaf_leaf_summation, direct_summation, fast_multipole_method

# ------------------------------------------------------------------------------------------------ #
#                                             Helpers                                              #
# ------------------------------------------------------------------------------------------------ #

def _wendland_ref(r2, support, dim):
    """Pure JAX Wendland C2 kernel, differentiable also at r = 0."""
    u = r2 / support**2
    usafe = jnp.where(u > 0, u, 1.)
    q = jnp.where(u > 0, jnp.sqrt(usafe), 0.)
    w = (1 - q)**4 * (1 + 4*q)
    norm = 7. / (math.pi * support**2) if dim == 2 else 21. / (2. * math.pi * support**3)
    return jnp.where(u < 1, norm * w, 0.)

def _density_ref(x, pos, mass, support, dim):
    r2 = jnp.sum((x[:, None, :] - pos[None, :, :])**2, axis=-1)
    return jnp.sum(mass[None, :] * _wendland_ref(r2, support, dim), axis=-1)

def _support_for_neighbours(nbody, dim, nngb=50.):
    """Support radius with about nngb neighbours for nbody uniform particles in [0,1]^dim."""
    vol_unit_ball = math.pi if dim == 2 else 4. / 3. * math.pi
    return (nngb / (nbody * vol_unit_ball))**(1. / dim)

def _particles(nbody, ntracer, dim, seed=0, ic="uniform", dtype=jnp.float32):
    """N-body particles with random positive masses followed by zero-mass tracers."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    if ic == "uniform":
        pos_body = jax.random.uniform(k1, (nbody, dim), dtype=dtype)
    else:
        pos_body = 0.5 + 0.15 * jax.random.normal(k1, (nbody, dim), dtype=dtype)
    pos_tracer = jax.random.uniform(k2, (ntracer, dim), dtype=dtype)
    mass_body = jax.random.uniform(k3, (nbody,), dtype=dtype, minval=0.5, maxval=1.5) / nbody
    return PosMass(
        pos=jnp.concatenate([pos_body, pos_tracer]),
        mass=jnp.concatenate([mass_body, jnp.zeros(ntracer, dtype=dtype)]),
    )

def _configs(support, dim):
    kernel = WendlandC2Kernel(support=support, dim=dim)
    cfg_fmm = FMMConfig(
        kernel=kernel, p=1, opening=OpeningBySupport(support=support),
        remove_self_interaction=False,
    )
    cfg_direct = DirectSummationConfig(kernel=kernel, remove_self_interaction=False)
    return cfg_fmm, cfg_direct

def _dense_leaf_leaf(part, cfg_fmm, leaf_size=32):
    n = part.pos.shape[0]
    assert n % leaf_size == 0
    ispl = jnp.arange(n // leaf_size + 1, dtype=jnp.int32) * leaf_size
    nleaf = len(ispl) - 1
    ilist = _dense_interaction_list(nleaf, nleaf, nleaf**2)
    return lambda p: _leaf_leaf_summation(p, ispl, ilist, cfg_fmm=cfg_fmm)

# ------------------------------------------------------------------------------------------------ #
#                                               Tests                                              #
# ------------------------------------------------------------------------------------------------ #

@pytest.mark.parametrize("dim", (2, 3))
def test_normalization(dim):
    support = 0.7
    ngrid = 400 if dim == 2 else 120
    x = (np.arange(ngrid) + 0.5) / ngrid * 2 * support - support
    grid = np.stack(np.meshgrid(*([x] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    w = _wendland_ref(jnp.asarray(np.sum(grid**2, axis=-1)), support, dim)
    integral = float(jnp.sum(w)) * (2 * support / ngrid)**dim
    assert integral == pytest.approx(1., rel=2e-3)

@pytest.mark.parametrize("dtype", (jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.skip_in_quick)))
@pytest.mark.parametrize("dim", (2, 3))
def test_direct_vs_reference(dim, dtype):
    with jax.enable_x64(dtype == jnp.float64):
        support = 0.125
        part = _particles(512, 64, dim, dtype=dtype)
        # Pairs exactly at, just inside, and just outside the support
        offsets = jnp.asarray([1., 0.999, 1.001], dtype=dtype) * support
        pos = part.pos.at[-3:].set(part.pos[0] + offsets[:, None] * jnp.eye(dim, dtype=dtype)[0])
        part = PosMass(pos=pos, mass=part.mass)

        _, cfg_direct = _configs(support, dim)
        loc = direct_summation.jit(part, cfg_direct=cfg_direct)

        rho = lambda x: _density_ref(x, part.pos, part.mass, support, dim)
        rho_ref = rho(part.pos)
        grad_ref = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(part.pos)

        rtol = 1e-4 if dtype == jnp.float32 else 1e-10
        scale = float(jnp.max(rho_ref))
        np.testing.assert_allclose(loc.potential(), rho_ref, rtol=rtol, atol=rtol * scale)
        np.testing.assert_allclose(
            loc.force(), -grad_ref, rtol=rtol, atol=rtol * float(jnp.max(jnp.abs(grad_ref)))
        )

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("ic", ("uniform", "gaussian"))
@pytest.mark.parametrize("dim", (2, 3))
def test_fmm_vs_direct(dim, ic):
    nbody, ntracer = 2**14, 2**12
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim, ic=ic)
    cfg_fmm, cfg_direct = _configs(support, dim)

    ref = direct_summation.jit(part, cfg_direct=cfg_direct).values
    fmm = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values

    assert np.isfinite(fmm).all()
    for i in range(dim + 1):
        atol = 2e-5 * float(jnp.max(jnp.abs(ref[:, i])))
        np.testing.assert_allclose(fmm[:, i], ref[:, i], rtol=1e-4, atol=atol)

@pytest.mark.parametrize("dim", (2, 3))
def test_tracers(dim):
    nbody, ntracer = 2**13, 2**11
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim)
    # Some tracers far away from all particles
    pos = part.pos.at[-16:].add(10.)
    part = PosMass(pos=pos, mass=part.mass)
    cfg_fmm, _ = _configs(support, dim)

    rho = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).potential()
    rho_ref = _density_ref(part.pos[nbody:], part.pos[:nbody], part.mass[:nbody], support, dim)
    np.testing.assert_allclose(
        rho[nbody:], rho_ref, rtol=1e-4, atol=1e-5 * float(jnp.max(rho_ref))
    )
    assert jnp.all(rho[-16:] == 0.)

    # N-body densities do not depend on the presence of tracers
    part_body = PosMass(pos=part.pos[:nbody], mass=part.mass[:nbody])
    rho_body = fast_multipole_method.jit(part_body, cfg_fmm=cfg_fmm).potential()
    np.testing.assert_allclose(
        rho[:nbody], rho_body, rtol=1e-5, atol=1e-6 * float(jnp.max(rho_body))
    )

@pytest.mark.parametrize("with_force", (False, True))
@pytest.mark.parametrize("dim", (2, 3))
def test_gradients_wrt_nbody(dim, with_force):
    nbody, ntracer = 2048, 512
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim, seed=3)
    cfg_fmm, cfg_direct = _configs(support, dim)
    kw, kv = jax.random.split(jax.random.PRNGKey(7))
    w = jax.random.normal(kw, (ntracer,))
    v = jax.random.normal(kv, (ntracer, dim)) * support

    def loss_from_values(values):
        rho, grad_rho = values[nbody:, 0], values[nbody:, 1:]
        loss = jnp.sum(w * rho)
        if with_force:
            loss = loss + jnp.sum(v * grad_rho)
        return loss

    leaf_leaf = _dense_leaf_leaf(part, cfg_fmm)
    losses = {
        "fmm": lambda p: loss_from_values(fast_multipole_method(p, cfg_fmm=cfg_fmm).values),
        "direct": lambda p: loss_from_values(direct_summation(p, cfg_direct=cfg_direct).values),
        "leaf_leaf": lambda p: loss_from_values(leaf_leaf(p)),
    }

    def loss_ref(pos_body, mass_body):
        rho = lambda x: _density_ref(x, pos_body, mass_body, support, dim)
        loss = jnp.sum(w * rho(part.pos[nbody:]))
        if with_force:
            grad_rho = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(part.pos[nbody:])
            loss = loss + jnp.sum(v * grad_rho)
        return loss

    gpos_ref, gmass_ref = jax.jit(jax.grad(loss_ref, argnums=(0, 1)))(
        part.pos[:nbody], part.mass[:nbody]
    )
    atol_pos = 2e-4 * float(jnp.max(jnp.abs(gpos_ref)))
    atol_mass = 2e-4 * float(jnp.max(jnp.abs(gmass_ref)))

    for name, loss in losses.items():
        g = jax.jit(jax.grad(loss))(part)
        np.testing.assert_allclose(
            g.pos[:nbody], gpos_ref, rtol=2e-3, atol=atol_pos, err_msg=f"{name}: pos"
        )
        np.testing.assert_allclose(
            g.mass[:nbody], gmass_ref, rtol=2e-3, atol=atol_mass, err_msg=f"{name}: mass"
        )

    if not with_force:
        # d loss / d m_j = sum_t w_t W(|x_t - x_j|)
        r2 = jnp.sum((part.pos[nbody:, None] - part.pos[None, :nbody])**2, axis=-1)
        gmass_analytic = jnp.sum(w[:, None] * _wendland_ref(r2, support, dim), axis=0)
        np.testing.assert_allclose(gmass_ref, gmass_analytic, rtol=1e-4, atol=atol_mass)

@pytest.mark.parametrize("dim", (2, 3))
def test_gradients_coincident_pair_finite(dim):
    nbody, ntracer = 1024, 256
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim, seed=5)
    # A tracer exactly on top of an N-body particle, and two coincident N-body particles
    pos = part.pos.at[nbody].set(part.pos[0]).at[1].set(part.pos[2])
    part = PosMass(pos=pos, mass=part.mass)
    cfg_fmm, cfg_direct = _configs(support, dim)

    def loss(p, direct):
        values = (direct_summation(p, cfg_direct=cfg_direct) if direct
                  else fast_multipole_method(p, cfg_fmm=cfg_fmm)).values
        return jnp.sum(values[:, 0]**2) + jnp.sum(values[:, 1:]**2) * support**2

    g_ref = jax.jit(jax.grad(lambda p: loss(p, True)))(part)
    g = jax.jit(jax.grad(lambda p: loss(p, False)))(part)
    for a, b in ((g.pos, g_ref.pos), (g.mass, g_ref.mass)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-4 * float(jnp.max(jnp.abs(b))))

@pytest.mark.skip_in_quick
def test_check_grads_float64():
    dim, nbody, ntracer = 3, 256, 64
    with jax.enable_x64():
        support = _support_for_neighbours(nbody, dim)
        part = _particles(nbody, ntracer, dim, seed=11, dtype=jnp.float64)
        cfg_fmm, _ = _configs(support, dim)
        w = jax.random.normal(jax.random.PRNGKey(1), (ntracer,), dtype=jnp.float64)

        def loss(pos_body, mass_body):
            p = PosMass(
                pos=jnp.concatenate([pos_body, part.pos[nbody:]]),
                mass=jnp.concatenate([mass_body, part.mass[nbody:]]),
            )
            values = fast_multipole_method(p, cfg_fmm=cfg_fmm).values
            return jnp.sum(w * values[nbody:, 0]) + jnp.sum(values[nbody:, 1:]**2) * 1e-4

        check_grads(
            loss, (part.pos[:nbody], part.mass[:nbody]), order=1, modes=["rev"],
            eps=1e-6, atol=1e-5, rtol=1e-5,
        )

@pytest.mark.parametrize("scale", (1e-3, 1e3))
def test_scale_invariance(scale):
    dim, nbody, ntracer = 3, 2**12, 2**10
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim)
    part_scaled = PosMass(pos=part.pos * scale, mass=part.mass)

    cfg_fmm, _ = _configs(support, dim)
    cfg_fmm_scaled, _ = _configs(support * scale, dim)

    rho = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values
    rho_scaled = fast_multipole_method.jit(part_scaled, cfg_fmm=cfg_fmm_scaled).values
    assert np.isfinite(rho_scaled).all()

    np.testing.assert_allclose(
        rho_scaled[:, 0] * scale**dim, rho[:, 0], rtol=1e-4, atol=1e-5 * float(jnp.max(rho[:, 0]))
    )
    np.testing.assert_allclose(
        rho_scaled[:, 1:] * scale**(dim + 1), rho[:, 1:], rtol=1e-3,
        atol=1e-4 * float(jnp.max(jnp.abs(rho[:, 1:])))
    )

    gloss = lambda p, cfg: jax.grad(lambda p: jnp.sum(fast_multipole_method(p, cfg_fmm=cfg).potential()**2))(p)
    g = jax.jit(gloss, static_argnums=1)(part, cfg_fmm)
    g_scaled = jax.jit(gloss, static_argnums=1)(part_scaled, cfg_fmm_scaled)
    assert np.isfinite(g_scaled.pos).all() and np.isfinite(g_scaled.mass).all()
    np.testing.assert_allclose(
        g_scaled.mass * scale**(2 * dim), g.mass, rtol=1e-3, atol=1e-4 * float(jnp.max(jnp.abs(g.mass)))
    )

def test_config_validation():
    dim, support = 3, 0.1
    part = _particles(256, 0, dim)
    kernel = WendlandC2Kernel(support=support, dim=dim)

    with pytest.raises(ValueError, match="OpeningBySupport"):
        fast_multipole_method(part, cfg_fmm=FMMConfig(kernel=kernel, opening=OpeningByAngle()))
    with pytest.raises(ValueError, match="OpeningBySupport"):
        fast_multipole_method(
            part, cfg_fmm=FMMConfig(kernel=kernel, opening=OpeningBySupport(support=0.5 * support))
        )
    with pytest.raises(ValueError, match="dim"):
        direct_summation(
            part, cfg_direct=DirectSummationConfig(kernel=WendlandC2Kernel(support=support, dim=2))
        )
    with pytest.raises(ValueError):
        WendlandC2Kernel(support=support, dim=4)
