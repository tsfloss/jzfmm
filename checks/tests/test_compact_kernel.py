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

from jzfmm.config import DirectSummationConfig, FMMConfig, OpeningByAngle, OpeningBySupport, PlummerKernel
from jzfmm.config import WendlandC2Kernel
from jzfmm.data import PosMass
from jzfmm.fmm import _leaf_leaf_summation, direct_summation, evaluate_at_positions, fast_multipole_method

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

# ------------------------------------------------------------------------------------------------ #
#                                   Query-restricted evaluation                                    #
# ------------------------------------------------------------------------------------------------ #

def _clustered_positions(key, n, dim):
    """Clumps of very different sizes plus a uniform background."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    nc = 32
    cen = jax.random.uniform(k1, (nc, dim))
    rs = 10**jax.random.uniform(k2, (nc,), minval=-2.2, maxval=-1.)
    ic = jax.random.randint(k3, (n,), 0, nc)
    x = cen[ic] + rs[ic, None] * jax.random.normal(k4, (n, dim))
    return jnp.where(jnp.arange(n)[:, None] < 0.7 * n, x, jax.random.uniform(k1, (n, dim)))

@pytest.mark.shrink_in_quick(keep_index=1)
@pytest.mark.parametrize("ic", ("uniform", "clustered"))
@pytest.mark.parametrize("dim", (2, 3))
def test_evaluate_at_positions(dim, ic):
    nbody, ntracer = 2**14, 500
    support = _support_for_neighbours(nbody, dim)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(4), 3)
    if ic == "uniform":
        pos = jax.random.uniform(k1, (nbody, dim))
        tr = jax.random.uniform(k2, (ntracer, dim))
    else:
        pos = _clustered_positions(k1, nbody, dim)
        tr = pos[:ntracer] + 0.3 * support * jax.random.normal(k2, (ntracer, dim))
    mass = jax.random.uniform(k3, (nbody,), minval=0.5, maxval=1.5) / nbody
    cfg_fmm, _ = _configs(support, dim)
    cfg_fmm.alloc_fac_ilist = 256.

    loc = evaluate_at_positions.jit(PosMass(pos=pos, mass=mass), tr, cfg_fmm=cfg_fmm)
    rho = lambda x: _density_ref(x, pos, mass, support, dim)
    rho_ref = rho(tr)
    grad_ref = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(tr)

    assert loc.values.shape == (ntracer, dim + 1)
    np.testing.assert_allclose(loc.potential(), rho_ref, rtol=1e-4, atol=1e-5 * float(jnp.max(rho_ref)))
    np.testing.assert_allclose(
        loc.values[:, 1:], grad_ref, rtol=1e-3, atol=1e-5 * float(jnp.max(jnp.abs(grad_ref)))
    )

@pytest.mark.parametrize("with_force", (False, True))
@pytest.mark.parametrize("dim", (2, 3))
def test_evaluate_at_positions_gradients(dim, with_force):
    nbody, ntracer = 4096, 300
    support = _support_for_neighbours(nbody, dim)
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(9), 4)
    pos = _clustered_positions(k1, nbody, dim)
    tr = pos[:ntracer] + 0.3 * support * jax.random.normal(k2, (ntracer, dim))
    mass = jax.random.uniform(k3, (nbody,), minval=0.5, maxval=1.5) / nbody
    w = jax.random.normal(k4, (ntracer,))
    v = jax.random.normal(k4, (ntracer, dim)) * support
    cfg_fmm, cfg_direct = _configs(support, dim)
    cfg_fmm.alloc_fac_ilist = 256.

    def loss_from_values(values):
        loss = jnp.sum(w * values[:, 0])
        if with_force:
            loss = loss + jnp.sum(v * values[:, 1:])
        return loss

    def loss_fmm(pos, mass, tr):
        return loss_from_values(evaluate_at_positions(PosMass(pos=pos, mass=mass), tr, cfg_fmm).values)

    def loss_direct(pos, mass, tr):
        part = PosMass(pos=jnp.concatenate([pos, tr]),
                       mass=jnp.concatenate([mass, jnp.zeros(ntracer)]))
        return loss_from_values(direct_summation(part, cfg_direct=cfg_direct).values[nbody:])

    g = jax.jit(jax.grad(loss_fmm, argnums=(0, 1, 2)))(pos, mass, tr)
    g_ref = jax.jit(jax.grad(loss_direct, argnums=(0, 1, 2)))(pos, mass, tr)
    for name, a, b in zip(("pos", "mass", "tracer pos"), g, g_ref):
        assert np.isfinite(a).all(), name
        np.testing.assert_allclose(
            a, b, rtol=2e-3, atol=2e-4 * float(jnp.max(jnp.abs(b))), err_msg=name
        )

def test_query_mask_marks_other_particles_nan():
    dim, nbody, ntracer = 3, 4096, 256
    support = _support_for_neighbours(nbody, dim)
    part = _particles(nbody, ntracer, dim)
    cfg_fmm, _ = _configs(support, dim)
    mask = jnp.arange(nbody + ntracer) >= nbody

    full = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values
    restricted = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm, query_mask=mask).values

    assert jnp.all(jnp.isnan(restricted[:nbody]))
    np.testing.assert_allclose(restricted[nbody:], full[nbody:], rtol=1e-5,
                               atol=1e-6 * float(jnp.max(jnp.abs(full[nbody:]))))
    with pytest.raises(ValueError, match="query_mask"):
        fast_multipole_method(part, cfg_fmm=cfg_fmm, query_mask=mask[:-1])

def test_query_mask_far_field_kernel():
    """Query restriction also prunes far-field (M2L) interactions consistently."""
    dim, n = 3, 8192
    part = PosMass(pos=_clustered_positions(jax.random.PRNGKey(2), n, dim),
                   mass=jnp.full(n, 1. / n))
    mask = jax.random.uniform(jax.random.PRNGKey(3), (n,)) < 0.02
    cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=0.01), p=4, opening=OpeningByAngle(theta=0.5),
                        alloc_fac_ilist=256.)

    def loss(part, query_mask):
        values = fast_multipole_method(part, cfg_fmm=cfg_fmm, query_mask=query_mask).values
        return jnp.sum(jnp.where(mask[:, None], values, 0.))

    full = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values
    restricted = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm, query_mask=mask).values
    np.testing.assert_allclose(restricted[mask], full[mask], rtol=1e-5,
                               atol=1e-6 * float(jnp.max(jnp.abs(full[mask]))))

    g_full = jax.jit(jax.grad(lambda p: loss(p, None)))(part)
    g_restricted = jax.jit(jax.grad(lambda p: loss(p, mask)))(part)
    for a, b in ((g_restricted.pos, g_full.pos), (g_restricted.mass, g_full.mass)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * float(jnp.max(jnp.abs(b))))

# ------------------------------------------------------------------------------------------------ #
#                                       Periodic boundaries                                        #
# ------------------------------------------------------------------------------------------------ #

def _density_ref_periodic(x, pos, mass, support, dim, boxsize):
    dx = x[:, None, :] - pos[None, :, :]
    dx = dx - boxsize * jnp.round(dx / boxsize)
    return jnp.sum(mass[None, :] * _wendland_ref(jnp.sum(dx**2, axis=-1), support, dim), axis=-1)

def _periodic_configs(support, dim, boxsize):
    kernel = WendlandC2Kernel(support=support, dim=dim, boxsize=boxsize)
    cfg_fmm = FMMConfig(
        kernel=kernel, p=1, opening=OpeningBySupport(support=support, boxsize=boxsize),
        remove_self_interaction=False, alloc_fac_ilist=256.,
    )
    cfg_direct = DirectSummationConfig(kernel=kernel, remove_self_interaction=False)
    return cfg_fmm, cfg_direct

def _periodic_particles(nbody, ntracer, dim, boxsize, seed=0):
    """Clustered particles in a periodic box, with clumps and tracers spanning its faces."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    pos = _clustered_positions(k1, nbody, dim) * boxsize
    # Clumps centred on a corner and on a face, cut by the box boundary
    ncl = nbody // 8
    pos = pos.at[:ncl].set(0.03 * boxsize * jax.random.normal(k2, (ncl, dim)))
    pos = pos.at[ncl:2*ncl, 0].set(0.02 * boxsize * jax.random.normal(k3, (ncl,)))
    pos = pos % boxsize
    tr = jnp.concatenate([pos[:ntracer // 2], jax.random.uniform(k2, (ntracer // 2, dim)) * boxsize])
    tr = (tr + 0.01 * boxsize * jax.random.normal(k3, tr.shape)) % boxsize
    mass = jax.random.uniform(k3, (nbody,), minval=0.5, maxval=1.5) / nbody
    return pos, mass, tr

@pytest.mark.parametrize("dim", (2, 3))
def test_periodic_vs_reference(dim):
    boxsize, nbody, ntracer = 3., 2**13, 512
    support = 0.08 * boxsize
    pos, mass, tr = _periodic_particles(nbody, ntracer, dim, boxsize)
    cfg_fmm, cfg_direct = _periodic_configs(support, dim, boxsize)

    rho = lambda x: _density_ref_periodic(x, pos, mass, support, dim, boxsize)
    rho_ref = rho(tr)
    grad_ref = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(tr)
    # Non-periodic densities differ at the faces, so this test is sensitive to the wrap
    rho_open = _density_ref(tr, pos, mass, support, dim)
    assert float(jnp.max(jnp.abs(rho_open - rho_ref))) > 0.1 * float(jnp.max(rho_ref))

    part = PosMass(pos=jnp.concatenate([pos, tr]), mass=jnp.concatenate([mass, jnp.zeros(ntracer)]))
    loc_direct = direct_summation.jit(part, cfg_direct=cfg_direct).values[nbody:]
    loc_fmm = fast_multipole_method.jit(part, cfg_fmm=cfg_fmm).values[nbody:]
    loc_eval = evaluate_at_positions.jit(PosMass(pos=pos, mass=mass), tr, cfg_fmm=cfg_fmm).values

    for name, loc in (("direct", loc_direct), ("fmm", loc_fmm), ("evaluate_at_positions", loc_eval)):
        np.testing.assert_allclose(
            loc[:, 0], rho_ref, rtol=1e-4, atol=1e-5 * float(jnp.max(rho_ref)), err_msg=name
        )
        np.testing.assert_allclose(
            loc[:, 1:], grad_ref, rtol=1e-3, atol=1e-5 * float(jnp.max(jnp.abs(grad_ref))), err_msg=name
        )

@pytest.mark.parametrize("dim", (2, 3))
def test_periodic_translation_invariance(dim):
    """Shifting everything by a constant, with or without wrapping into the box, changes nothing."""
    boxsize, nbody, ntracer = 1., 2**14, 1024
    support = _support_for_neighbours(nbody, dim) * boxsize
    pos, mass, tr = _periodic_particles(nbody, ntracer, dim, boxsize, seed=3)
    cfg_fmm, _ = _periodic_configs(support, dim, boxsize)
    evaluate = lambda p, t: evaluate_at_positions.jit(PosMass(pos=p, mass=mass), t, cfg_fmm=cfg_fmm).values

    ref = evaluate(pos, tr)
    shift = jnp.asarray([0.37, 0.81, 0.55][:dim]) * boxsize
    wrapped = evaluate((pos + shift) % boxsize, (tr + shift) % boxsize)
    # Positions outside [0, boxsize), e.g. unwrapped N-body output
    unwrapped = evaluate(pos + shift, tr + shift - boxsize)

    for name, loc in (("wrapped", wrapped), ("unwrapped", unwrapped)):
        np.testing.assert_allclose(
            loc, ref, rtol=1e-4, atol=1e-5 * float(jnp.max(jnp.abs(ref))), err_msg=name
        )

@pytest.mark.parametrize("dim", (2, 3))
def test_periodic_gradients(dim):
    boxsize, nbody, ntracer = 2., 4096, 400
    support = 0.08 * boxsize
    pos, mass, tr = _periodic_particles(nbody, ntracer, dim, boxsize, seed=5)
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    w = jax.random.normal(k1, (ntracer,))
    v = jax.random.normal(k2, (ntracer, dim)) * support
    cfg_fmm, _ = _periodic_configs(support, dim, boxsize)

    def loss_fmm(pos, mass, tr):
        loc = evaluate_at_positions(PosMass(pos=pos, mass=mass), tr, cfg_fmm)
        return jnp.sum(w * loc.values[:, 0]) + jnp.sum(v * loc.values[:, 1:])

    def loss_ref(pos, mass, tr):
        rho = lambda x: _density_ref_periodic(x, pos, mass, support, dim, boxsize)
        grad = jax.vmap(jax.grad(lambda xi: rho(xi[None])[0]))(tr)
        return jnp.sum(w * rho(tr)) + jnp.sum(v * grad)

    g = jax.jit(jax.grad(loss_fmm, argnums=(0, 1, 2)))(pos, mass, tr)
    g_ref = jax.jit(jax.grad(loss_ref, argnums=(0, 1, 2)))(pos, mass, tr)
    for name, a, b in zip(("pos", "mass", "tracer pos"), g, g_ref):
        assert np.isfinite(a).all(), name
        np.testing.assert_allclose(
            a, b, rtol=2e-3, atol=2e-4 * float(jnp.max(jnp.abs(b))), err_msg=name
        )

def test_periodic_config_validation():
    dim, support, boxsize = 3, 0.1, 1.
    part = _particles(256, 0, dim)
    kernel = WendlandC2Kernel(support=support, dim=dim, boxsize=boxsize)

    with pytest.raises(ValueError, match="boxsize"):
        fast_multipole_method(part, cfg_fmm=FMMConfig(
            kernel=kernel, opening=OpeningBySupport(support=support)))
    with pytest.raises(ValueError, match="boxsize"):
        fast_multipole_method(part, cfg_fmm=FMMConfig(
            kernel=WendlandC2Kernel(support=support, dim=dim),
            opening=OpeningBySupport(support=support, boxsize=boxsize)))
    with pytest.raises(ValueError, match="only supported for WendlandC2Kernel"):
        fast_multipole_method(part, cfg_fmm=FMMConfig(
            kernel=PlummerKernel(), opening=OpeningBySupport(support=support, boxsize=boxsize)))
    with pytest.raises(ValueError, match="2\\*support"):
        WendlandC2Kernel(support=0.6, dim=dim, boxsize=boxsize)
    with pytest.raises(ValueError, match="2\\*support"):
        OpeningBySupport(support=0.6, boxsize=boxsize)
