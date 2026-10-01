"""Pair counts and two-point correlation functions on the tree.

Pairs are found through the leaf interaction list of the dual tree walk with
:class:`jzfmm.config.OpeningBySupport`, using the largest bin edge as the support. Every
leaf pair whose boxes come closer than that radius is then histogrammed by direct
summation over its particle pairs on the GPU.
"""
from dataclasses import dataclass, field, replace
import math
import warnings
import numpy as np
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig
from jztree.data import PosMass, get_pos_mass
from jztree.tree import zsort_and_tree
from jztree.comm import in_shard_map_context

from .config import FMMConfig, OpeningBySupport, WendlandC2Kernel
from .multipoles import build_multipole_hierarchy
from .fmm import _fmm_dual_walk

import jzfmm_cuda.ffi_pair_counting as ffi_pair_counting
jax.ffi.register_ffi_target("LeafLeafPairCount", ffi_pair_counting.LeafLeafPairCount(), platform="CUDA")

_BLOCK_SIZE = 128
_NLUT = 1024

@dataclass(unsafe_hash=True, slots=True)
class PairCountConfig:
    """Configures tree-based pair counting.

    Args:
        tree: Tree construction configuration. ``max_leaf_size`` may be at most 128.
        alloc_fac_ilist: Interaction-list entries allocated per leaf node. Each leaf
            interacts with all leaves within the largest bin edge, so this has to grow
            with the cube of the largest bin edge in units of the leaf size. ``None``
            estimates it from the mean leaf size and doubles it on overflow, which
            requires concrete positions or a ``boxsize``.
        remove_self_pairs: Whether to exclude the pair of each particle with itself.
    """

    tree : TreeConfig = field(
        default_factory=lambda: TreeConfig(
            max_leaf_size=64, mass_centered=False, alloc_fac_nodes=1.5, regularization=None,
            coarse_fac=4.0
        )
    )
    alloc_fac_ilist : float | None = None
    remove_self_pairs : bool = True

def _fmm_config(cfg: PairCountConfig, rmax: float, boxsize: float | None) -> FMMConfig:
    # The kernel is never evaluated, it only has to be consistent with the opening
    return FMMConfig(
        tree=cfg.tree, p=1,
        kernel=WendlandC2Kernel(support=rmax, boxsize=boxsize),
        opening=OpeningBySupport(support=rmax, boxsize=boxsize),
        alloc_fac_ilist=cfg.alloc_fac_ilist,
    )

def _estimate_alloc_fac_ilist(cfg: PairCountConfig, rmax: float, n: int, extent: tuple) -> float:
    """Leaf interactions per leaf for uniformly distributed particles, with a safety margin.

    Leaves hold about 0.6*max_leaf_size particles on average. In cosmological boxes,
    clustering raised the number of interactions over this estimate by up to a factor 2.8
    for rmax ~ l, but only by ~1.3 for rmax >> l, so the margin decreases with rmax/l.
    A leaf can interact at most with all leaves.
    """
    dim = len(extent)
    nleaf = max(n, 1) / (0.6 * cfg.tree.max_leaf_size)
    l = (float(np.prod(extent)) / nleaf) ** (1. / dim)
    unit = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    margin = 1.6 + 2. * l / (rmax + l)
    est = min(margin * unit * ((rmax + l) / l) ** dim, 1.5 * nleaf)
    # Rounding limits recompilation for slightly different inputs
    return float(256 * math.ceil(max(est, 64.) / 256))

def _bin_lut(r2_edges: np.ndarray):
    """Table from float bits of r2 to bins, see find_bin in pair_counting.cuh."""
    keys = lambda x: int(np.asarray(x, dtype=np.float32).view(np.int32))
    kmax = keys(r2_edges[-1])
    kmin = keys(r2_edges[0]) if r2_edges[0] > 0 else 0
    # Finer cells than the bins down to a factor 2^-40 below the largest edge
    kmin = max(kmin, kmax - 40 * 2**23)
    shift = 0
    while (kmax - kmin) >> shift >= _NLUT:
        shift += 1
    nlut = ((kmax - kmin) >> shift) + 1
    cell_lo = (kmin + (np.arange(nlut, dtype=np.int64) << shift)).astype(np.int32).view(np.float32)
    nbins = len(r2_edges) - 1
    lut = np.clip(np.searchsorted(r2_edges, cell_lo, side="right") - 1, 0, nbins - 1)
    return lut.astype(np.uint16), np.asarray([kmin, shift, nlut], dtype=np.int32)

def _u64_to_float(x: jax.Array) -> jax.Array:
    """Interprets (n, 2) uint32 words as little-endian uint64 and converts to float."""
    if jax.config.jax_enable_x64:
        return jax.lax.bitcast_convert_type(x, jnp.uint64).astype(jnp.float64)
    return x[:, 0].astype(jnp.float32) + x[:, 1].astype(jnp.float32) * 2.**32

def _f64_to_float(x: jax.Array) -> jax.Array:
    """Interprets (n, 2) uint32 words as little-endian float64 values."""
    if jax.config.jax_enable_x64:
        return jax.lax.bitcast_convert_type(x, jnp.float64)
    lo, hi = x[:, 0], x[:, 1]
    sign = jnp.where(hi >> 31 == 1, -1., 1.).astype(jnp.float32)
    expo = ((hi >> 20) & 0x7FF).astype(jnp.int32)
    mant = 1. + (hi & 0xFFFFF).astype(jnp.float32) * 2.**-20 + lo.astype(jnp.float32) * 2.**-52
    # Accumulated weights are finite, so only zero needs special treatment
    return jnp.where(expo == 0, 0., sign * jnp.ldexp(mant, expo - 1023))

def _pair_counts(
        pos: jax.Array, weights: jax.Array | None, r_edges: tuple, boxsize: float | None,
        cfg: PairCountConfig
    ):
    dim = pos.shape[-1]
    if dim not in (2, 3):
        raise ValueError(f"Pair counting supports dim=2 or dim=3, got dim={dim}")
    if in_shard_map_context():
        raise NotImplementedError("Pair counting is not yet implemented for multiple devices")
    assert cfg.tree.max_leaf_size <= _BLOCK_SIZE
    weighted = weights is not None
    assert cfg.alloc_fac_ilist is not None
    cfg_fmm = _fmm_config(cfg, float(r_edges[-1]), boxsize)

    mass = jnp.ones(pos.shape[:1], pos.dtype) if weights is None else jnp.asarray(weights, pos.dtype)
    partz, th = zsort_and_tree(PosMass(pos=pos, mass=mass), cfg_fmm.tree)

    # Only the opening decisions of the walk matter, the multipoles are never read
    mph = build_multipole_hierarchy(th, partz.pos, partz.mass[:, None], cfg_fmm=cfg_fmm)
    _, ilist = _fmm_dual_walk(th, mph, cfg_fmm=cfg_fmm)

    spl = jnp.asarray(th.splits_leaf_to_part(), dtype=jnp.int32)
    posm = get_pos_mass(partz)
    r2_edges = (np.asarray(r_edges, dtype=np.float64)**2).astype(pos.dtype)
    lut, lut_params = _bin_lut(r2_edges)
    params = jnp.asarray([boxsize or 0.], dtype=pos.dtype)
    nbins = len(r_edges) - 1

    out = jax.ShapeDtypeStruct((nbins, 2), jnp.uint32)
    counts, wcounts = jax.ffi.ffi_call("LeafLeafPairCount", (out, out))(
        jnp.array([0, spl.size - 1], dtype=jnp.int32), spl, spl,
        jnp.asarray(ilist.ispl, dtype=jnp.int32), jnp.asarray(ilist.isrc, dtype=jnp.int32),
        posm, posm, jnp.asarray(r2_edges), jnp.asarray(lut), jnp.asarray(lut_params), params,
        weighted=weighted, remove_self_pairs=bool(cfg.remove_self_pairs),
        block_size=np.uint64(_BLOCK_SIZE),
    )
    counts = _u64_to_float(counts)
    if weighted:
        return counts, _f64_to_float(wcounts)
    return counts
_pair_counts.jit = jax.jit(_pair_counts, static_argnames=("r_edges", "boxsize", "cfg"))

def _check_edges(r_edges, boxsize) -> tuple:
    r_edges = np.asarray(r_edges, dtype=np.float64)
    if r_edges.ndim != 1 or len(r_edges) < 2:
        raise ValueError("r_edges must be a 1D array with at least two entries")
    if not np.all(np.diff(r_edges) > 0) or r_edges[0] < 0:
        raise ValueError("r_edges must be non-negative and strictly increasing")
    if len(r_edges) - 1 > np.iinfo(np.uint16).max:
        raise ValueError(f"At most {np.iinfo(np.uint16).max} bins are supported")
    if boxsize is not None and not boxsize > 2 * r_edges[-1]:
        raise ValueError(f"Periodic pair counts require boxsize > 2*r_edges[-1], got "
                         f"boxsize={boxsize} and r_edges[-1]={r_edges[-1]}")
    return tuple(float(r) for r in r_edges)

def pair_counts(
        pos: jax.Array, r_edges, boxsize: float | None = None, weights: jax.Array | None = None,
        cfg: PairCountConfig = PairCountConfig()
    ):
    """Histograms the separations of all particle pairs.

    **Compatibility:** :compat-jit:`JIT` (:paramref:`r_edges` must be concrete)

    Every ordered pair :math:`(i,j)` with :math:`i\\neq j` and
    ``r_edges[k] <= |x_i - x_j| < r_edges[k+1]`` is counted in bin ``k``, so each unordered
    pair is counted twice. This is the same convention as e.g. ``Corrfunc.theory.DD``
    with ``autocorr=1``.

    Args:
        pos: Positions of shape ``(N, dim)`` with ``dim`` 2 or 3.
        r_edges: Concrete, strictly increasing bin edges of length ``nbins+1``. Only pairs
            closer than ``r_edges[-1]`` are found, and the cost grows with its cube.
        boxsize: Side length of a periodic box, so that separations use the nearest
            periodic image. Requires ``boxsize > 2*r_edges[-1]``. ``None`` for open
            boundaries.
        weights: Optional per-particle weights of shape ``(N,)``.
        cfg: Pair counting configuration.

    Returns:
        The pair counts of shape ``(nbins,)``. With :paramref:`weights`, a tuple of the
        counts and the weighted counts :math:`\\sum w_i w_j`. Counts are float64 if
        ``jax_enable_x64`` is set, and float32 otherwise.
    """
    r_edges = _check_edges(r_edges, boxsize)
    boxsize = None if boxsize is None else float(boxsize)
    pos = jnp.asarray(pos)
    if cfg.alloc_fac_ilist is not None:
        return _pair_counts.jit(pos, weights, r_edges, boxsize, cfg)

    concrete = not isinstance(pos, jax.core.Tracer)
    dim = pos.shape[-1]
    if boxsize is not None:
        extent = (boxsize,) * dim
    elif concrete:
        ext = jnp.max(pos, axis=0) - jnp.min(pos, axis=0)
        extent = tuple(float(e) for e in np.maximum(np.asarray(ext), r_edges[-1]))
    else:
        raise ValueError("PairCountConfig.alloc_fac_ilist=None requires a boxsize or "
                         "concrete positions. Please set alloc_fac_ilist explicitly.")
    cfg = replace(cfg, alloc_fac_ilist=_estimate_alloc_fac_ilist(
        cfg, r_edges[-1], pos.shape[0], extent
    ))
    if not concrete:
        return _pair_counts.jit(pos, weights, r_edges, boxsize, cfg)
    for attempt in range(4):
        try:
            res = jax.block_until_ready(_pair_counts.jit(pos, weights, r_edges, boxsize, cfg))
            # Allocation checks raise from callbacks that the result does not wait for
            jax.effects_barrier()
            return res
        except Exception as e:
            if attempt == 3:
                raise
            if "Interaction list allocation too small" in str(e):
                cfg = replace(cfg, alloc_fac_ilist=2 * cfg.alloc_fac_ilist)
                msg = f"alloc_fac_ilist={cfg.alloc_fac_ilist}"
            elif "Tree-Leaf allocation too small" in str(e):
                cfg = replace(cfg, tree=replace(cfg.tree, alloc_fac_nodes=2 * cfg.tree.alloc_fac_nodes))
                msg = f"tree.alloc_fac_nodes={cfg.tree.alloc_fac_nodes}"
            else:
                raise
            warnings.warn(f"Allocation too small, retrying with {msg}", RuntimeWarning, stacklevel=2)

def shell_volumes(r_edges, dim: int = 3) -> np.ndarray:
    """Volumes of the spherical shells between consecutive bin edges."""
    r_edges = np.asarray(r_edges, dtype=np.float64)
    unit = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    return unit * np.diff(r_edges**dim)

def correlation_function(
        pos: jax.Array, r_edges, boxsize: float, weights: jax.Array | None = None,
        cfg: PairCountConfig = PairCountConfig(), return_counts: bool = False
    ):
    """Two-point correlation function in a periodic box.

    Uses the natural estimator :math:`\\xi = DD/RR - 1` with the analytic random pair
    counts of the periodic box, :math:`RR_k = N(N-1)\\,V_k/L^d` for shells of volume
    :math:`V_k`, or :math:`((\\sum w)^2-\\sum w^2)\\,V_k/L^d` with weights.

    Args:
        pos: Positions of shape ``(N, dim)``.
        r_edges: Concrete, strictly increasing bin edges, see :func:`pair_counts`.
        boxsize: Side length :math:`L` of the periodic box.
        weights: Optional per-particle weights of shape ``(N,)``.
        cfg: Pair counting configuration.
        return_counts: Whether to also return the (weighted) pair counts DD and the
            analytic RR.

    Returns:
        :math:`\\xi` per bin, or ``(xi, DD, RR)`` if :paramref:`return_counts`.
    """
    pos = jnp.asarray(pos)
    dim = pos.shape[-1]
    res = pair_counts(pos, r_edges, boxsize=boxsize, weights=weights, cfg=cfg)
    vol = jnp.asarray(shell_volumes(r_edges, dim) / float(boxsize)**dim, dtype=res[0].dtype if weights is not None else res.dtype)
    if weights is None:
        dd = res
        n = pos.shape[0]
        rr = (n * (n - 1.)) * vol
    else:
        dd = res[1]
        w = jnp.asarray(weights, dtype=dd.dtype)
        rr = (jnp.sum(w)**2 - jnp.sum(w**2)) * vol
    xi = dd / rr - 1.
    return (xi, dd, rr) if return_counts else xi
