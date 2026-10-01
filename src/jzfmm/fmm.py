from typing import Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import warnings
from dataclasses import dataclass, replace


from jztree.data import PosMass, InteractionList, get_pos_mass, TreeHierarchy, PosLvl, PackedArray, RankIdx, get_num
from jztree.tree import grouped_dense_interaction_list, zsort_and_tree
from jztree.tree import distr_grouped_dense_interaction_list, simplify_interaction_list
from jztree.jax_ext import empty_like, pcast_like, raise_if, shard_map_constructor
from jztree.tools import masked_inverse
from jztree.comm import all_to_all_request_children, all_to_all_with_irank, get_rank_info, in_shard_map_context
from jztree.stats import AllocStats, stats_callback
from jax.sharding import PartitionSpec as P

from .config import (DirectSummationConfig, FMMConfig, GaussianKernel, OpeningByAngle,
                     OpeningByGaussianError, OpeningBySupport, WendlandC2Kernel)
from .data import LocalExpansion
from .multipoles import _fmm_node_to_child, _shift_local_to_children_vjp_x, build_multipole_hierarchy, num_multi, p_of_num_multi

import jzfmm_cuda.ffi_fmm as ffi_fmm
import jzfmm_cuda.ffi_pair_summation as ffi_pair_summation
jax.ffi.register_ffi_target("CountInteractionsAndM2L", ffi_fmm.CountInteractionsAndM2L(), platform="CUDA")
jax.ffi.register_ffi_target("InsertInteractions", ffi_fmm.InsertInteractions(), platform="CUDA")
jax.ffi.register_ffi_target("LeafLeafPairSummation", ffi_pair_summation.LeafLeafPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("BwdLeafLeafPairSummation", ffi_pair_summation.BwdLeafLeafPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("DirectPairSummation", ffi_pair_summation.DirectPairSummation(), platform="CUDA")
jax.ffi.register_ffi_target("BwdDirectPairSummation", ffi_pair_summation.BwdDirectPairSummation(), platform="CUDA")

def _check_kernel_config(kernel, dim: int, opening=None):
    """Rejects configurations that would silently produce wrong results."""
    if isinstance(kernel, WendlandC2Kernel):
        if kernel.dim != dim:
            raise ValueError(
                f"WendlandC2Kernel is normalized for dim={kernel.dim}, but the particles "
                f"have dim={dim}."
            )
        if opening is not None and not (
            isinstance(opening, OpeningBySupport) and opening.support >= kernel.support
        ):
            raise ValueError(
                "WendlandC2Kernel requires opening=OpeningBySupport(support) with "
                f"support >= {kernel.support}, got {opening}."
            )
        if opening is not None and opening.boxsize != kernel.boxsize:
            raise ValueError(
                f"WendlandC2Kernel has boxsize={kernel.boxsize}, but OpeningBySupport has "
                f"boxsize={opening.boxsize}. Both must use the same periodic box."
            )
    elif isinstance(kernel, GaussianKernel):
        if kernel.dim != dim:
            raise ValueError(
                f"GaussianKernel is normalized for dim={kernel.dim}, but the particles "
                f"have dim={dim}."
            )
        if isinstance(opening, OpeningBySupport) and opening.boxsize != kernel.boxsize:
            raise ValueError(
                f"GaussianKernel has boxsize={kernel.boxsize}, but OpeningBySupport has "
                f"boxsize={opening.boxsize}. Both must use the same periodic box."
            )
        if kernel.boxsize is not None and isinstance(opening, OpeningByAngle):
            raise ValueError(
                "Periodic boundaries with GaussianKernel require OpeningByGaussianError or "
                "OpeningBySupport, since OpeningByAngle does not ensure unique nearest images."
            )
    elif isinstance(opening, OpeningBySupport) and opening.boxsize is not None:
        raise ValueError(
            "Periodic boundaries (OpeningBySupport.boxsize) are only supported for "
            f"WendlandC2Kernel and GaussianKernel, got {kernel}."
        )
    if isinstance(opening, OpeningByGaussianError) and not isinstance(kernel, GaussianKernel):
        raise ValueError(f"OpeningByGaussianError requires a GaussianKernel, got {kernel}.")

# ------------------------------------------------------------------------------------------------ #
#                                          M2L Evaluation                                          #
# ------------------------------------------------------------------------------------------------ #

@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _FMMChildData:
    poslvl: PosLvl
    mp: jax.Array
    # Per node (number of query particles, mass, number of source particles). Only pairs
    # with queries on one side and sources on the other are kept. The mass is only used
    # by mass-dependent opening criteria.
    weights: jax.Array

def _comm_buffer_size(base_size: int, factor: float) -> int:
    return max(base_size, int(np.ceil(base_size * factor)))

def _fmm_node_to_node(
        node_range: jax.Array,
        node_ilist: InteractionList,
        spl_recv: jax.Array,
        child_recv: _FMMChildData,
        cfg_fmm: FMMConfig,
        single_thread_per_receiver: jax.Array | None = None,
        loc_in: jax.Array | None = None,
        spl_src: jax.Array | None = None,
        child_src: _FMMChildData | None = None
    ) -> Tuple[jax.Array, InteractionList]:
    """
    Evaluates M2L for a tree plane and generates child interaction list.

    single_thread_per_receiver: Use this to make results exactly independent of grouping
       of nodes. Only useful to use at top-level to achieve GPU count independence.

    Outputs:
    - loc: multipole local expansion, shape (Nchild, M)
    - new_ilist: new interaction list for children
    """
    if spl_src is None:
        spl_src = spl_recv
    if child_src is None:
        child_src = child_recv
    if single_thread_per_receiver is None:
        single_thread_per_receiver = jnp.zeros(1, dtype=jnp.int32)
    else:
        single_thread_per_receiver = jnp.asarray(single_thread_per_receiver, dtype=jnp.int32)
    node_range = jnp.asarray(node_range, dtype=jnp.int32)
    spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
    spl_src = jnp.asarray(spl_src, dtype=jnp.int32)
    ilist_ispl = jnp.asarray(node_ilist.ispl, dtype=jnp.int32)
    ilist_isrc = jnp.asarray(node_ilist.isrc, dtype=jnp.int32)

    size = len(child_recv.poslvl.pos)
    ilist_alloc_size = node_ilist.size()
    kernel = cfg_fmm.kernel
    dim = child_recv.poslvl.pos.shape[-1]

    if child_src.mp.dtype == jnp.float64 and cfg_fmm.p >= 6:
        warnings.warn(
            "For double precision, M2L order 6 and above currently spills registers "
            "and can be substantially slower than order 5.",
            RuntimeWarning,
            stacklevel=2,
        )

    assert ilist_alloc_size < 2**31, (
        f"So far only int32 is supported ({ilist_alloc_size / 2**31:.3g} * 2**31 entries requested)"
    )
    assert len(spl_recv) == len(node_ilist.ispl)

    # children = jnp.concatenate((plane.center(), plane.lvl.view(jnp.float32)[...,None]), axis=-1)
    children_recv = child_recv.poslvl.pos_lvl()
    children_src = child_src.poslvl.pos_lvl()
    
    # Determine output shapes
    dtype = child_src.mp.dtype
    kernel_params = kernel.params(dtype=dtype)
    opening = cfg_fmm.opening
    opening_params = opening.params_for(cfg_fmm, dtype=dtype)
    out_loc = jax.ShapeDtypeStruct((size, num_multi(cfg_fmm.p, dim=dim)), dtype)
    if loc_in is None:
        loc_in = jnp.zeros(out_loc.shape, dtype=out_loc.dtype)
        loc_in = pcast_like(loc_in, spl_recv)
    out_interaction_count = jax.ShapeDtypeStruct((size,), jnp.int32)
    
    # Count opened interactions and evaluate M2L
    loc, interaction_counts = jax.ffi.ffi_call(
        "CountInteractionsAndM2L",
        (out_loc, out_interaction_count,),
        input_output_aliases={13: 0},
    )(
        node_range, spl_recv, spl_src, ilist_ispl, ilist_isrc,
        children_recv, children_src, child_recv.weights, child_src.weights,
        child_src.mp, kernel_params, opening_params,
        single_thread_per_receiver, loc_in,
        p=np.int32(cfg_fmm.p),
        radial_kernel_kind=np.int32(kernel.kind_id()),
        opening_criterion_kind=np.int32(opening.kind_id()),
    )
    loc, interaction_counts = jax.tree.map(
        lambda x: pcast_like(x, spl_recv), (loc, interaction_counts)
    )
    # valid_child = jnp.arange(size, dtype=spl_recv.dtype) < spl_recv[-1]
    # interaction_counts = jnp.where(valid_child, interaction_counts, 0)

    # Insert interactions
    ispl_child = jnp.pad(jnp.cumsum(interaction_counts), (1, 0)).astype(jnp.int32)
    out_child_ilist = jax.ShapeDtypeStruct((ilist_alloc_size,), jnp.int32)

    child_ilist = jax.ffi.ffi_call(
        "InsertInteractions",
        (out_child_ilist,)
    )(
        node_range, spl_recv, spl_src, ilist_ispl, ilist_isrc,
        children_recv, children_src, child_recv.weights, child_src.weights,
        ispl_child, opening_params,
        opening_criterion_kind=np.int32(opening.kind_id()),
    )[0]
    child_ilist = pcast_like(child_ilist, spl_recv)

    # Create interaction list from outputs
    new_ilist = InteractionList(
        ispl=ispl_child,
        isrc=child_ilist,
        has_separate_query_and_source_indices=(
            node_ilist.has_separate_query_and_source_indices
        ),
    )

    new_ilist.ispl = new_ilist.ispl + raise_if(
        new_ilist.nfilled() > new_ilist.size(), 
        "Interaction list allocation too small ({nfil}/{size})\nHint: Increase alloc_fac_ilist", 
        nfil=new_ilist.nfilled(), size=new_ilist.size()
    )
    stats_callback(
        "allocation", AllocStats.record_filled_interactions,
        new_ilist.nfilled(), new_ilist.size()
    )
    
    return loc, new_ilist
_fmm_node_to_node.jit = jax.jit(_fmm_node_to_node, static_argnames=['cfg_fmm'])

def _fmm_dual_walk(
        th: TreeHierarchy, mph: PackedArray, cfg_fmm: FMMConfig, qh: PackedArray | None = None,
        mh: PackedArray | None = None, ptypes: Tuple[int, int] | None = None
    ):
    """Dual tree walk.

    qh and mh optionally hold the number of query particles and the mass per node. Both must
    not depend on mph, since the backward pass repeats the walk with cotangent multipoles and
    has to make the same opening decisions. For a tree with separate source and query particle
    types, ptypes = (source type, query type) instead defines the queries and sources per node.
    Otherwise all particles are sources.
    """
    in_smap = in_shard_map_context()
    if in_smap:
        rank, ndev, axis_name = get_rank_info()

    # define root level:
    size = th.size()
    comm_size_nodes = _comm_buffer_size(size, cfg_fmm.alloc_fac_comm_nodes)
    center0 = th.center().get(0, size)
    dim = center0.shape[-1]
    if in_smap:
        spl, ilist, nsup = distr_grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size,
            size_ilist=int(th.size_leaves*cfg_fmm.alloc_fac_ilist),
            size_ids=comm_size_nodes,
            separate_query_and_source_indices=True,
        )
    else:
        spl, ilist, nsup = grouped_dense_interaction_list(
            th.num(th.num_planes()-1), size_ilist=int(th.size_leaves*cfg_fmm.alloc_fac_ilist),
            ngroup=32, size_super=size
        )

    loc = jnp.zeros((size, num_multi(cfg_fmm.p, dim=dim)), dtype=mph.data.dtype)
    if in_smap:
        loc = pcast_like(loc, mph.data)

    # Add super node data into tree information
    spl_n2n = th.ispl_n2n.append(spl, nsup+1, fill_value=spl[-1], resize=True)
    cent = th.center().append(jnp.zeros((size, dim), dtype=center0.dtype), nsup, fill_value=0., resize=True)
    top_level = th.num_planes() - 1
    super_level = jnp.max(th.lvl.get(
        top_level, size, fill_value=jnp.iinfo(jnp.int32).min
    ))
    lvl = th.lvl.append(
        jnp.full(size, super_level, dtype=jnp.int32), nsup,
        fill_value=super_level, resize=True,
    )

    def handle_level(i, carry):
        level = th.num_planes() - 1 - i

        parent_loc, parent_ilist = carry
        parent_spl_recv = spl_n2n.get(level+1, size+1)
        parent_cent = cent.get(level+1, size)
        single_thread_per_receiver = jnp.asarray(i == 0, dtype=jnp.int32)[None]

        dtype = mph.data.dtype
        if ptypes is not None:
            valid = jnp.arange(size) < th.num(level)
            nsrc = jnp.where(valid, th.npart(level, ptypes[0], size), 0).astype(dtype)
            nquery = jnp.where(valid, th.npart(level, ptypes[1], size), 0).astype(dtype)
        else:
            nsrc = jnp.ones(size, dtype=dtype)
            if qh is None:
                nquery = jnp.ones(size, dtype=dtype)
            else:
                nquery = qh.get(level, size=size, fill_value=0.)[:, 0]
        if mh is None:
            mass = jnp.zeros(size, dtype=dtype)
        else:
            mass = mh.get(level, size=size, fill_value=0.)[:, 0]
        child_recv = _FMMChildData(
            poslvl=th.poslvl(level, size),
            mp=mph.get(level, size=size),
            weights=pcast_like(jnp.stack([nquery, mass, nsrc], axis=-1), mph.data),
        )

        if in_smap:
            # Request the remote source node data that local receivers interact with.
            # This also runs in size-one shard maps as a syntax/shape test path.
            (child_src, ids), parent_spl_src, dev_spl = all_to_all_request_children(
                parent_ilist.dev_spl, parent_ilist.ids, parent_spl_recv,
                (child_recv, jnp.arange(size, dtype=jnp.int32)),
                output=empty_like((child_recv, jnp.arange(size, dtype=jnp.int32)), new_size=comm_size_nodes),
                axis_name=axis_name, err_hint_child="\nHint: increase alloc_fac_comm_nodes",
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes"
            )
            stats_callback(
                "allocation", AllocStats.record_filled_nodes_interaction,
                dev_spl[-1], size
            )
            parent_range = jnp.array([0, parent_ilist.ispl.size-1], dtype=jnp.int32)
        else:
            child_src, parent_spl_src = child_recv, parent_spl_recv
            parent_range = jnp.array([0, spl_n2n.num(level+1)-1], dtype=jnp.int32)

        loc_ch = _fmm_node_to_child(
            parent_spl_recv, parent_loc,
            PosLvl(pos=parent_cent, lvl=lvl.get(level+1, size)),
            child_recv.poslvl, cfg_fmm=cfg_fmm
        )

        loc_ch, ilist = _fmm_node_to_node(
            parent_range, parent_ilist, parent_spl_recv, child_recv,
            cfg_fmm=cfg_fmm, single_thread_per_receiver=single_thread_per_receiver,
            loc_in=loc_ch,
            spl_src=parent_spl_src, child_src=child_src
        )

        if in_smap:
            ilist = replace(ilist, ids=ids, dev_spl=dev_spl)
            ilist = simplify_interaction_list(ilist)

        return loc_ch, ilist

    loc, ilist = jax.lax.fori_loop(
        0, th.num_planes(), handle_level, (loc, ilist), unroll=True
    )

    return loc, ilist
_fmm_dual_walk.jit = jax.jit(_fmm_dual_walk, static_argnames=['cfg_fmm'])

# ------------------------------------------------------------------------------------------------ #
#                           Leaf-Leaf (Particle to Particle) Interactions                          #
# ------------------------------------------------------------------------------------------------ #

def _sum_to_input_shape(x: jax.Array, target: jax.Array) -> jax.Array:
    if x.shape == jnp.shape(target):
        return x
    return jnp.reshape(jnp.sum(x), jnp.shape(target))

def _leaf_leaf_summation(
        particles: PosMass,
        ispl: jax.Array,
        ilist: InteractionList,
        cfg_fmm: FMMConfig = None,
        loc_in: jax.Array | None = None,
    ) -> jax.Array:
    particles_recv = particles
    spl_recv = jnp.asarray(ispl, dtype=jnp.int32)
    comm_size_particles = _comm_buffer_size(particles_recv.pos.shape[0], cfg_fmm.alloc_fac_comm_particles)
    in_smap = in_shard_map_context()
    if in_smap:
        rank, _, axis_name = get_rank_info()

    block_size = 128
    assert cfg_fmm.tree.max_leaf_size <= block_size
    dim = particles_recv.pos.shape[-1]

    node_range = jnp.array([0, spl_recv.size-1], dtype=jnp.int32)
    posm_recv = get_pos_mass(particles_recv)
    out_type = jax.ShapeDtypeStruct((particles_recv.pos.shape[0], dim + 1), posm_recv.dtype)
    if loc_in is None:
        loc_in = jnp.zeros(out_type.shape, dtype=out_type.dtype)
        loc_in = pcast_like(loc_in, spl_recv)
    kernel = cfg_fmm.kernel
    kernel_params = kernel.params(dtype=posm_recv.dtype)

    def eval_fwd(particles_recv, spl_recv, ilist, loc_in):
        spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
        ilist = replace(
            ilist,
            ispl=jnp.asarray(ilist.ispl, dtype=jnp.int32),
            isrc=jnp.asarray(ilist.isrc, dtype=jnp.int32),
        )
        if in_smap:
            particles_src, spl_src, _dev_spl_src = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, particles_recv,
                output=empty_like(particles_recv, new_size=comm_size_particles),
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes.",
                err_hint_child="\nHint: increase alloc_fac_comm_particles."
            )
            stats_callback(
                "allocation", AllocStats.record_filled_part_interactions,
                _dev_spl_src[-1], particles_src.pos.shape[0]
            )
        else:
            particles_src, spl_src = particles_recv, spl_recv
        spl_src = jnp.asarray(spl_src, dtype=jnp.int32)

        loc = jax.ffi.ffi_call(
            "LeafLeafPairSummation", (out_type,),
            input_output_aliases={8: 0},
        )(
            node_range, spl_recv, spl_src, ilist.ispl, ilist.isrc,
            get_pos_mass(particles_recv), get_pos_mass(particles_src), kernel_params, loc_in,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg_fmm.kahan_summation),
            remove_self_interaction=bool(cfg_fmm.remove_self_interaction),
        )[0]
        loc = pcast_like(loc, spl_recv)
        num = getattr(particles_recv, "num", None)
        if num is not None:
            valid = jnp.arange(loc.shape[0], dtype=jnp.int32) < num
            loc = jnp.where(valid[:, None], loc, jnp.nan)
        return loc, (particles_recv, spl_recv, ilist, particles_src, spl_src)

    @jax.custom_vjp
    def eval(particles_recv, spl_recv, ilist, loc_in):
        return eval_fwd(particles_recv, spl_recv, ilist, loc_in)[0]
    
    def eval_bwd(res, gloc):
        particles_recv, spl_recv, ilist, particles_src, spl_src = res
        spl_recv = jnp.asarray(spl_recv, dtype=jnp.int32)
        spl_src = jnp.asarray(spl_src, dtype=jnp.int32)
        ilist = replace(
            ilist,
            ispl=jnp.asarray(ilist.ispl, dtype=jnp.int32),
            isrc=jnp.asarray(ilist.isrc, dtype=jnp.int32),
        )
        if in_smap:
            gloc_src, _, _ = all_to_all_request_children(
                ilist.dev_spl, ilist.ids, spl_recv, gloc,
                output=empty_like(gloc, float_val=0., new_size=comm_size_particles),
                axis_name=axis_name,
                err_hint_parent="\nHint: increase alloc_fac_comm_nodes",
                err_hint_child="\nHint: increase alloc_fac_comm_particles."
            )
        else:
            gloc_src = gloc

        # The globally symmetric leaf interaction list lets each owner compute its particle
        # gradients in one pass by requesting source particles and source cotangents.
        gposm = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_type,))(
            node_range, spl_recv, spl_src, ilist.ispl, ilist.isrc,
            get_pos_mass(particles_recv), get_pos_mass(particles_src),
            kernel_params, gloc, gloc_src,
            radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
            kahan=bool(cfg_fmm.kahan_summation),
            remove_self_interaction=bool(cfg_fmm.remove_self_interaction),
        )[0]
        gpos = pcast_like(gposm[:,:dim], particles_recv.pos)
        gmass = pcast_like(
            _sum_to_input_shape(gposm[:,dim], particles_recv.mass),
            particles_recv.mass,
        )
        gnum = None
        if particles_recv.num is not None:
            gnum = jnp.zeros_like(particles_recv.num, dtype=jax.dtypes.float0)

        gposm = PosMass(
            pos=gpos, mass=gmass, num=gnum, num_total=particles_recv.num_total
        )
        return gposm, None, None, gloc
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(particles_recv, spl_recv, ilist, loc_in)
_leaf_leaf_summation.jit = jax.jit(_leaf_leaf_summation, static_argnames=['cfg_fmm'])

def _leaf_leaf_summation_queries(
        particles: PosMass,
        ispl: jax.Array,
        ilist: InteractionList,
        query_mask: jax.Array,
        cfg_fmm: FMMConfig,
        loc_in: jax.Array | None = None,
    ) -> jax.Array:
    """Leaf-leaf summation that only evaluates results at query particles.

    The forward pass uses the query particles as receivers and all particles as
    sources. Since only queries carry cotangents, the backward pass splits into
    queries receiving from all particles and all particles receiving from queries.
    Results at the remaining particles are NaN. Single device only.
    """
    n = particles.pos.shape[0]
    dim = particles.pos.shape[-1]
    block_size = 128
    assert cfg_fmm.tree.max_leaf_size <= block_size

    spl = jnp.asarray(ispl, dtype=jnp.int32)
    ilist_ispl = jnp.asarray(ilist.ispl, dtype=jnp.int32)
    ilist_isrc = jnp.asarray(ilist.isrc, dtype=jnp.int32)
    node_range = jnp.array([0, spl.size-1], dtype=jnp.int32)

    # Compact the query particles, which keeps them grouped by leaf
    q = jnp.asarray(query_mask, dtype=bool)
    qcum = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(q, dtype=jnp.int32)])
    nq = qcum[-1]
    spl_q = qcum[spl]
    arange = jnp.arange(n, dtype=jnp.int32)
    idx = jnp.zeros(n, jnp.int32).at[jnp.where(q, qcum[:-1], n)].set(arange, mode="drop")
    valid_q = arange < nq
    scatter_idx = jnp.where(valid_q, idx, n)

    posm = get_pos_mass(particles)
    out_type = jax.ShapeDtypeStruct((n, dim + 1), posm.dtype)
    if loc_in is None:
        loc_in = jnp.zeros(out_type.shape, dtype=out_type.dtype)
    kernel = cfg_fmm.kernel
    kernel_params = kernel.params(dtype=posm.dtype)
    attrs = dict(
        radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
        kahan=bool(cfg_fmm.kahan_summation),
        remove_self_interaction=bool(cfg_fmm.remove_self_interaction),
    )

    def to_full(x_q, fill):
        return jnp.full(out_type.shape, fill, dtype=x_q.dtype).at[scatter_idx].set(x_q, mode="drop")

    @jax.custom_vjp
    def eval(posm, loc_in):
        posm_q = posm[idx]
        loc_q = jax.ffi.ffi_call("LeafLeafPairSummation", (out_type,), input_output_aliases={8: 0})(
            node_range, spl_q, spl, ilist_ispl, ilist_isrc, posm_q, posm, kernel_params,
            loc_in[idx], **attrs
        )[0]
        return to_full(loc_q, jnp.nan)

    def eval_fwd(posm, loc_in):
        return eval(posm, loc_in), posm

    def eval_bwd(posm, gloc):
        gloc = jnp.where(q[:, None], gloc, 0.)
        posm_q = posm[idx]
        gloc_q = jnp.where(valid_q[:, None], gloc[idx], 0.)
        # Queries receive from all particles
        gposm_q = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_type,))(
            node_range, spl_q, spl, ilist_ispl, ilist_isrc, posm_q, posm, kernel_params,
            gloc_q, gloc, **attrs
        )[0]
        # All particles receive from queries. Only kept for non-queries, since the
        # contributions to queries are already complete above.
        gposm_all = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_type,))(
            node_range, spl, spl_q, ilist_ispl, ilist_isrc, posm, posm_q, kernel_params,
            gloc, gloc_q, **attrs
        )[0]
        gposm = jnp.where(q[:, None], to_full(gposm_q, 0.), gposm_all)
        return gposm, gloc

    eval.defvjp(eval_fwd, eval_bwd)

    return eval(posm, loc_in)


# ------------------------------------------------------------------------------------------------ #
#                                      Direct Summation Forces                                     #
# ------------------------------------------------------------------------------------------------ #

def direct_summation(
        part: PosMass,
        cfg_direct: DirectSummationConfig,
        G: float | jax.Array = 1,
    ) -> LocalExpansion:
    """Evaluates all pair interactions by direct summation.

    **Compatibility:** :compat-jit:`JIT` :compat-shard-local:`Local only`
    :compat-autodiff:`Autodiff`

    **Helpers:** :helper-jit:`.jit`

    This function does not implement multi-device communication and will run
    only on local shards when called inside of a ``shard_map``.

    Args:
        part: Particle positions and masses.
        cfg_direct: Direct-summation configuration.
        G: Gravitational constant or multiplicative interaction strength.
    Returns:
        Local expansion containing the potential and force at each particle.
    """
    block_size = 64
    posm = get_pos_mass(part)
    out_type = jax.ShapeDtypeStruct(posm.shape, posm.dtype)
    kernel = cfg_direct.kernel
    _check_kernel_config(kernel, dim=part.pos.shape[-1])
    kernel_params = kernel.params(dtype=posm.dtype)
    
    @jax.custom_vjp
    def eval(xm):
        loc = jax.ffi.ffi_call("DirectPairSummation", (out_type,))(
        xm, kernel_params, block_size=np.uint64(block_size),
        radial_kernel_kind=np.int32(kernel.kind_id()), kahan=cfg_direct.kahan_summation,
        remove_self_interaction=bool(cfg_direct.remove_self_interaction))[0]
        return loc
    def eval_fwd(xm):
        return eval(xm), xm
    def eval_bwd(xm, gloc):
        gxm = jax.ffi.ffi_call("BwdDirectPairSummation", (out_type,))(
            gloc, xm, kernel_params, block_size=np.uint64(block_size),
            radial_kernel_kind=np.int32(kernel.kind_id()), kahan=cfg_direct.kahan_summation,
            remove_self_interaction=bool(cfg_direct.remove_self_interaction)
        )[0]
        return gxm,
    
    eval.defvjp(eval_fwd, eval_bwd)

    return LocalExpansion(eval(posm) * G, dim=part.pos.shape[-1])
direct_summation.jit = jax.jit(direct_summation, static_argnames=['cfg_direct'])

def _direct_potential_jax(x, m=1., softening=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(rinv * jnp.broadcast_to(m, x.shape[:-1])[None,:], axis=1)
_direct_potential_jax.jit = jax.jit(_direct_potential_jax)

def _direct_force_jax(x, m=1., softening=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(dx * rinv**3 * jnp.broadcast_to(m, x.shape[:-1])[None,:,None], axis=1)
_direct_force_jax.jit = jax.jit(_direct_force_jax)

def _direct_potential_scan_jax(x, m=1., n2lim=1e8, eps=1e-5):
    N = x.shape[0]

    nmax = int(np.ceil(n2lim / len(x)))
    nev = int(np.ceil(x.shape[0] / nmax))

    def potential_over_range(i1, i2):
        xi = x[jnp.arange(nmax, dtype=jnp.int32) + i1]
        # Compute vector distances to all other particles
        r_ij2 = jnp.sum((x - xi[:,None])**2, axis=-1)
        distinv = jnp.where(r_ij2 < 1e-30, 0., 1./jnp.sqrt(r_ij2 + eps**2)) # avoid self-interaction

        return - jnp.sum(m * distinv, axis=1)

    def handle_interval(_, i):
        return None, potential_over_range(nmax * i, nmax * (i + 1))

    _, phis = jax.lax.scan(handle_interval, None, jnp.arange(nev, dtype=jnp.int32))

    return jnp.concatenate(phis)[0:N]
_direct_potential_scan_jax.jit = jax.jit(_direct_potential_scan_jax, static_argnames=("n2lim",))

# ------------------------------------------------------------------------------------------------ #
#                                         Master Functions                                         #
# ------------------------------------------------------------------------------------------------ #


def _query_hierarchy(th: TreeHierarchy, pos: jax.Array, query_mask: jax.Array, cfg_fmm: FMMConfig
                     ) -> PackedArray:
    """Number of query particles per node, as the monopoles of the query indicator."""
    weights = jnp.asarray(query_mask, dtype=pos.dtype)
    return _weight_hierarchy(th, pos, weights, cfg_fmm)

def _weight_hierarchy(th: TreeHierarchy, pos: jax.Array, weights: jax.Array, cfg_fmm: FMMConfig,
                      ptype: int = 0) -> PackedArray:
    """Sum of the particle weights per node, without gradients. Only the monopole is used."""
    weights = jnp.broadcast_to(jnp.asarray(weights, dtype=pos.dtype), pos.shape[:1])
    return build_multipole_hierarchy(
        th, jax.lax.stop_gradient(pos), jax.lax.stop_gradient(weights),
        cfg_fmm=replace(cfg_fmm, p=1), ptype=ptype
    )

def _evaluate_node_node_fmm(
        partz: PosMass, th: TreeHierarchy, *, cfg_fmm: FMMConfig, qh: PackedArray | None = None,
        mh: PackedArray | None = None
    ) -> Tuple[jax.Array, InteractionList]:
    
    def eval_fwd(pos, mp, pout=1):
        mph = build_multipole_hierarchy(th, pos, mp, cfg_fmm=cfg_fmm)
        loc_node, ilist = _fmm_dual_walk(th, mph, cfg_fmm=cfg_fmm, qh=qh, mh=mh)
        ispl = th.splits_leaf_to_part()
        node = th.poslvl(0)
        # Particle scale H=1 makes the L2P output use physical derivative units.
        particle = PosLvl(pos=pos, lvl=jnp.zeros(pos.shape[0], dtype=jnp.int32))
        loc_part = _fmm_node_to_child(ispl, loc_node, node, particle, pout=pout, cfg_fmm=cfg_fmm)
        return (loc_part, ilist), (pos, mp, ispl, node, loc_node)
    
    def eval_bwd(pout, res, grads):
        pos, mp, ispl, node, loc_node = res
        gloc = grads[0]
        particle = PosLvl(pos=pos, lvl=jnp.zeros(pos.shape[0], dtype=jnp.int32))

        # Backwards pass = FMM with gloc as multipole weights
        # plus the position derivatives of the shifting operators

        gx1 = _shift_local_to_children_vjp_x(ispl, loc_node, node, particle, gloc)
        
        dim = pos.shape[-1]
        (gmp, _), (_, _, _, _, gmp_node) = eval_fwd(pos, gloc, pout=p_of_num_multi(mp.shape[-1], dim=dim))

        gx2 = _shift_local_to_children_vjp_x(ispl, gmp_node, node, particle, mp)
        
        return gx1 + gx2, _sum_to_input_shape(gmp, mp)
    
    @partial(jax.custom_vjp, nondiff_argnames=['pout'])
    def eval(pos, mp, pout=1):
        return eval_fwd(pos, mp, pout=pout)[0]
    
    eval.defvjp(eval_fwd, eval_bwd)

    return eval(partz.pos, jnp.reshape(partz.mass, jnp.shape(partz.mass) + (1,)))
_evaluate_node_node_fmm.jit = jax.jit(_evaluate_node_node_fmm, static_argnames=['cfg_fmm', ])

def _fast_multipole_method_z(
        partz: PosMass, th: TreeHierarchy, *, cfg_fmm: FMMConfig, G=1., pout: int = 1,
        query_mask: jax.Array | None = None
    ) -> LocalExpansion:
    assert pout == 1, "Only pout=1 (potential only) is supported currently."
    dim = partz.pos.shape[-1]

    qh = None if query_mask is None else _query_hierarchy(th, partz.pos, query_mask, cfg_fmm)
    # Error bounds scale with the absolute masses, which do not cancel for mixed signs
    mh = _weight_hierarchy(th, partz.pos, jnp.abs(partz.mass), cfg_fmm) if cfg_fmm.opening.uses_node_mass() else None

    if cfg_fmm.opening.evaluates_far_field():
        loc_node_node, ilist = _evaluate_node_node_fmm(partz, th, cfg_fmm=cfg_fmm, qh=qh, mh=mh)
    else:
        # Without far-field interactions the node-node part only determines the leaf
        # interaction list. It contributes nothing to the result or to its gradients.
        # Multipoles are only read for M2L, so any hierarchy of the right shape will do.
        if qh is not None:
            mph = qh
        else:
            mp = jnp.broadcast_to(jnp.reshape(partz.mass, jnp.shape(partz.mass) + (1,)),
                                  partz.pos.shape[:1] + (1,))
            mph = build_multipole_hierarchy(
                th, jax.lax.stop_gradient(partz.pos), jax.lax.stop_gradient(mp), cfg_fmm=cfg_fmm
            )
        _, ilist = _fmm_dual_walk(th, mph, cfg_fmm=cfg_fmm, qh=qh, mh=mh)
        loc_node_node = jnp.zeros((partz.pos.shape[0], num_multi(pout, dim=dim)), dtype=mph.data.dtype)
        loc_node_node = pcast_like(loc_node_node, partz.pos)
    spl = th.splits_leaf_to_part()

    ilist = jax.lax.stop_gradient(ilist)
    if query_mask is None:
        loc = _leaf_leaf_summation(partz, spl, ilist, cfg_fmm=cfg_fmm, loc_in=loc_node_node)
    elif in_shard_map_context():
        # Query-restricted receivers are not implemented for distributed interaction lists,
        # but node pairs without queries are still skipped.
        loc = _leaf_leaf_summation(partz, spl, ilist, cfg_fmm=cfg_fmm, loc_in=loc_node_node)
        loc = jnp.where(jnp.asarray(query_mask, dtype=bool)[:, None], loc, jnp.nan)
    else:
        loc = _leaf_leaf_summation_queries(
            partz, spl, ilist, query_mask, cfg_fmm=cfg_fmm, loc_in=loc_node_node
        )

    return LocalExpansion(loc * G, dim=dim)
_fast_multipole_method_z.jit = jax.jit(_fast_multipole_method_z, static_argnames=("cfg_fmm", "G", "pout"))

def _parse_fmm_result(result: str) -> Tuple[str, ...]:
    keys = tuple(result.split("_"))
    valid = {"loc", "locz", "partz", "tree"}
    unknown = set(keys) - valid
    if unknown:
        raise ValueError(f"Unknown FMM result key(s): {sorted(unknown)}. Valid keys are {sorted(valid)}.")
    return keys

def _as_posmass(part) -> PosMass:
    return PosMass(
        pos=part.pos, mass=part.mass,
        num=getattr(part, "num", None),
        num_total=getattr(part, "num_total", None)
    )

def fast_multipole_method(
        part: PosMass, cfg_fmm: FMMConfig, th: TreeHierarchy | None = None,
        result: str = "loc", G: float = 1., pout: int = 1,
        query_mask: jax.Array | None = None
    ) -> LocalExpansion:
    """Evaluates particle interactions with the fast multipole method.

    **Compatibility:** :compat-jit:`JIT` :compat-shard:`Shard map`
    :compat-autodiff:`Autodiff`

    **Helpers:** :helper-jit:`.jit` :helper-smap:`.smap`

    Args:
        part: Particle data following the :class:`jztree.data.PosMass`
            interface.
        cfg_fmm: Fast-multipole configuration.
        th: Existing tree hierarchy. If provided, :paramref:`part` must already
            be in z-order and ``"loc"`` cannot be requested.
        result: Underscore-separated selection of ``"loc"`` (input-order local
            expansions), ``"locz"`` (z-order local expansions), ``"partz"``
            (z-order particles), and ``"tree"`` (tree hierarchy).
        G: Gravitational constant or multiplicative interaction strength.
        pout: Output expansion order. Currently only ``1`` is supported,
            returning the potential and force.
        query_mask: Optional boolean array of shape ``(size,)`` marking the
            particles at which results are required, in the same order as
            :paramref:`part`. Node pairs that contain no query particle are
            skipped, which can save most of the work when queries are sparse.
            Results at the remaining particles are ``NaN``. Gradients with
            respect to all particles remain exact. See also
            :func:`evaluate_at_positions`.
    Returns:
        The requested result, or a tuple when multiple results are requested.
    """
    assert pout == 1, "Only pout=1 (potential only) is supported currently."
    keys = _parse_fmm_result(result)
    _check_kernel_config(cfg_fmm.kernel, dim=part.pos.shape[-1], opening=cfg_fmm.opening)
    in_smap = in_shard_map_context()

    if th is None:
        size = part.pos.shape[0]
        idx = jnp.arange(size, dtype=jnp.int32)
        if in_smap:
            rank, ndev, axis_name = get_rank_info()
            origin = RankIdx(rank=jnp.full(size, rank, dtype=jnp.int32), idx=idx)
        else:
            ndev = 1
            origin = RankIdx(rank=None, idx=idx)
        num_origin = get_num(part, default_to_length=(ndev == 1))
        if query_mask is None:
            partz, origin_z, th = zsort_and_tree(part, cfg_fmm.tree, data=origin)
            query_mask_z = None
        else:
            query_mask = jnp.asarray(query_mask, dtype=bool)
            if query_mask.shape != part.pos.shape[:1]:
                raise ValueError(
                    f"query_mask must have shape {part.pos.shape[:1]}, got {query_mask.shape}"
                )
            partz, (origin_z, query_mask_z), th = zsort_and_tree(
                part, cfg_fmm.tree, data=(origin, query_mask.astype(jnp.int32))
            )
    elif "loc" in keys:
        raise ValueError(
            "result='loc' is only available when fast_multipole_method builds the tree. "
            "Use result='locz' when providing a tree."
        )
    else:
        partz = part
        origin_z = None
        num_origin = None
        query_mask_z = query_mask

    locz = None
    if ("loc" in keys) or ("locz" in keys):
        locz = _fast_multipole_method_z(
            _as_posmass(partz), th, cfg_fmm=cfg_fmm, G=G, pout=pout, query_mask=query_mask_z
        )

    loc = None
    if "loc" in keys:
        if in_smap:
            (loc_values, idx), _dev_spl = all_to_all_with_irank(
                origin_z.rank, (locz.values, origin_z.idx), num=partz.num,
                axis_name=axis_name, err_hint="\nThis should never fail..."
            )
        else:
            loc_values, idx = locz.values, origin_z.idx
        inv_sort = masked_inverse(idx, mask=jnp.arange(len(idx), dtype=idx.dtype) < num_origin)
        loc = LocalExpansion(loc_values[inv_sort], dim=part.pos.shape[-1])

    out = {
        "loc": loc,
        "locz": locz,
        "partz": partz,
        "tree": th,
    }
    res = tuple(out[key] for key in keys)
    return res[0] if len(res) == 1 else res
fast_multipole_method.jit = jax.jit(fast_multipole_method, static_argnames=("cfg_fmm", "result", "G", "pout"))
fast_multipole_method.smap = shard_map_constructor(
    fast_multipole_method,
    in_specs=(P(-1), None, P(-1), None, None, None, P(-1)),
    static_argnames=("cfg_fmm", "result", "G", "pout"),
)

# ------------------------------------------------------------------------------------------------ #
#                                 Two-Sided Evaluation at Queries                                  #
# ------------------------------------------------------------------------------------------------ #

# Particle types of the two-sided tree
_SRC, _QRY = 0, 1

def _evaluate_node_node_two_sided(
        src: PosMass, pos_q: jax.Array, th: TreeHierarchy, *, cfg_fmm: FMMConfig,
        mh: PackedArray | None = None
    ) -> Tuple[jax.Array, InteractionList]:
    """Far-field (M2L) contribution of the sources at the queries of a two-type tree.

    Mirrors _evaluate_node_node_fmm with separate particle sets: multipoles come from the
    sources and local expansions are evaluated at the queries. The backward pass is the
    same walk with the query cotangents as multipoles, evaluated at the sources.
    """
    node = th.poslvl(0)
    dim = pos_q.shape[-1]

    def walk(pos, mp, ptype):
        mph = build_multipole_hierarchy(th, pos, mp, cfg_fmm=cfg_fmm, ptype=ptype)
        return _fmm_dual_walk(th, mph, cfg_fmm=cfg_fmm, mh=mh, ptypes=(_SRC, _QRY))

    def particles(pos):
        # Particle scale H=1 makes the L2P output use physical derivative units.
        return PosLvl(pos=pos, lvl=jnp.zeros(pos.shape[0], dtype=jnp.int32))

    def eval_fwd(pos_src, mp, pos_q):
        loc_node, ilist = walk(pos_src, mp, _SRC)
        loc_q = _fmm_node_to_child(
            th.splits_leaf_to_part(_QRY), loc_node, node, particles(pos_q), pout=1, cfg_fmm=cfg_fmm
        )
        return (loc_q, ilist), (pos_src, mp, pos_q, loc_node)

    def eval_bwd(res, grads):
        pos_src, mp, pos_q, loc_node = res
        gloc = grads[0]
        spl_src, spl_q = th.splits_leaf_to_part(_SRC), th.splits_leaf_to_part(_QRY)
        # Position derivatives of the shift from the nodes to the queries
        gx_q = _shift_local_to_children_vjp_x(spl_q, loc_node, node, particles(pos_q), gloc)
        # The adjoint FMM uses the query cotangents as multipoles and evaluates at the sources
        gloc_node, _ = walk(pos_q, gloc, _QRY)
        gmp = _fmm_node_to_child(
            spl_src, gloc_node, node, particles(pos_src),
            pout=p_of_num_multi(mp.shape[-1], dim=dim), cfg_fmm=cfg_fmm
        )
        # Position derivatives of the multipoles of the sources
        gx_src = _shift_local_to_children_vjp_x(spl_src, gloc_node, node, particles(pos_src), mp)
        return gx_src, _sum_to_input_shape(gmp, mp), gx_q

    @jax.custom_vjp
    def eval(pos_src, mp, pos_q):
        return eval_fwd(pos_src, mp, pos_q)[0]

    eval.defvjp(eval_fwd, eval_bwd)

    mass = jnp.broadcast_to(src.mass, src.pos.shape[:1])
    return eval(src.pos, mass[:, None], pos_q)

def _leaf_leaf_summation_two_sided(
        src: PosMass, pos_q: jax.Array, th: TreeHierarchy, ilist: InteractionList,
        cfg_fmm: FMMConfig, loc_in: jax.Array
    ) -> jax.Array:
    """Leaf-leaf summation from the sources onto the queries of a two-type tree."""
    dim = pos_q.shape[-1]
    block_size = 128
    assert cfg_fmm.tree.max_leaf_size <= block_size

    spl_src = jnp.asarray(th.splits_leaf_to_part(_SRC), dtype=jnp.int32)
    spl_q = jnp.asarray(th.splits_leaf_to_part(_QRY), dtype=jnp.int32)
    ilist_ispl = jnp.asarray(ilist.ispl, dtype=jnp.int32)
    ilist_isrc = jnp.asarray(ilist.isrc, dtype=jnp.int32)
    node_range = jnp.array([0, spl_q.size-1], dtype=jnp.int32)

    posm_src = get_pos_mass(src)
    out_q = jax.ShapeDtypeStruct((pos_q.shape[0], dim + 1), posm_src.dtype)
    out_src = jax.ShapeDtypeStruct(posm_src.shape, posm_src.dtype)
    kernel = cfg_fmm.kernel
    kernel_params = kernel.params(dtype=posm_src.dtype)
    # Queries and sources are distinct particles, so there are no self interactions
    attrs = dict(
        radial_kernel_kind=np.int32(kernel.kind_id()), block_size=np.uint64(block_size),
        kahan=bool(cfg_fmm.kahan_summation), remove_self_interaction=False,
    )

    def with_zero_mass(pos):
        return jnp.concatenate([pos, jnp.zeros_like(pos[:, :1])], axis=-1)

    @jax.custom_vjp
    def eval(posm_src, pos_q, loc_in):
        return jax.ffi.ffi_call("LeafLeafPairSummation", (out_q,), input_output_aliases={8: 0})(
            node_range, spl_q, spl_src, ilist_ispl, ilist_isrc, with_zero_mass(pos_q), posm_src,
            kernel_params, loc_in, **attrs
        )[0]

    def eval_fwd(posm_src, pos_q, loc_in):
        return eval(posm_src, pos_q, loc_in), (posm_src, pos_q)

    def eval_bwd(res, gloc):
        posm_src, pos_q = res
        posm_q = with_zero_mass(pos_q)
        gloc_src = jnp.zeros(out_src.shape, dtype=gloc.dtype)
        # Queries receive from the sources
        gposm_q = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_q,))(
            node_range, spl_q, spl_src, ilist_ispl, ilist_isrc, posm_q, posm_src, kernel_params,
            gloc, gloc_src, **attrs
        )[0]
        # Sources receive from the queries. The interaction list is symmetric.
        gposm_src = jax.ffi.ffi_call("BwdLeafLeafPairSummation", (out_src,))(
            node_range, spl_src, spl_q, ilist_ispl, ilist_isrc, posm_src, posm_q, kernel_params,
            gloc_src, gloc, **attrs
        )[0]
        return gposm_src, gposm_q[:, :dim], gloc

    eval.defvjp(eval_fwd, eval_bwd)

    return eval(posm_src, pos_q, loc_in)

def _zsort_and_tree_two_types(src: PosMass, pos_q: jax.Array, cfg_fmm: FMMConfig):
    """Same as jztree's zsort_and_tree_multi_type for unpadded sources and queries.

    The per-type leaf splits index the z-sorted particles of each type in order, i.e. the
    stable partition of the z-sorted particles by type. This computes it with prefix sums
    instead of an argsort, which dominated the tree construction for many queries.
    """
    nsrc, nquery = src.pos.shape[0], pos_q.shape[0]
    n = nsrc + nquery
    combined = PosMass(
        pos=jnp.concatenate([src.pos, pos_q]),
        mass=jnp.concatenate([src.mass, jnp.zeros(nquery, dtype=src.mass.dtype)]),
    )
    ptype = (jnp.arange(n, dtype=jnp.int32) >= nsrc).astype(jnp.int32)
    partz, originz, ptypez, th = zsort_and_tree(
        combined, cfg_fmm.tree, data=jnp.arange(n, dtype=jnp.int32), ptype=ptype, num_types=2,
        shrink=False
    )
    arange = jnp.arange(n, dtype=jnp.int32)

    def gather_indices(mask, size):
        dest = jnp.cumsum(mask, dtype=jnp.int32) - mask
        return jnp.zeros(size, jnp.int32).at[jnp.where(mask, dest, size)].set(arange, mode="drop")

    isrc = gather_indices(ptypez == _SRC, nsrc)
    iqry = gather_indices(ptypez == _QRY, nquery)
    srcz = PosMass(pos=partz.pos[isrc], mass=partz.mass[isrc])
    return srcz, partz.pos[iqry], originz[iqry] - nsrc, th

def _evaluate_at_positions_two_sided(
        part: PosMass, pos_query: jax.Array, cfg_fmm: FMMConfig, G: float = 1.
    ) -> LocalExpansion:
    """Builds one tree with the sources and the queries as separate particle types.

    Leaves are split by the number of particles of each type, so that many queries do not
    fragment the source leaves. Only node pairs with sources on one side and queries on the
    other are walked, and all interactions go from the sources onto the queries.
    """
    nsrc, nquery = part.pos.shape[0], pos_query.shape[0]
    dim = part.pos.shape[-1]
    mass = jnp.broadcast_to(jnp.asarray(part.mass, dtype=part.pos.dtype), (nsrc,))
    src = PosMass(pos=part.pos, mass=mass)
    srcz, pos_qz, origin_q, th = _zsort_and_tree_two_types(
        src, jnp.asarray(pos_query, dtype=part.pos.dtype), cfg_fmm
    )

    mh = None
    if cfg_fmm.opening.uses_node_mass() or not cfg_fmm.opening.evaluates_far_field():
        # Error bounds scale with the absolute masses, which do not cancel for mixed signs
        mh = _weight_hierarchy(th, srcz.pos, jnp.abs(srcz.mass), cfg_fmm, ptype=_SRC)

    if cfg_fmm.opening.evaluates_far_field():
        loc_q, ilist = _evaluate_node_node_two_sided(
            srcz, pos_qz, th, cfg_fmm=cfg_fmm, mh=mh if cfg_fmm.opening.uses_node_mass() else None
        )
    else:
        # Without far-field interactions the walk only determines the leaf interaction list.
        # Multipoles are only read for M2L, so any hierarchy will do.
        _, ilist = _fmm_dual_walk(th, mh, cfg_fmm=cfg_fmm, ptypes=(_SRC, _QRY))
        loc_q = jnp.zeros((nquery, num_multi(1, dim=dim)), dtype=srcz.pos.dtype)

    loc_q = _leaf_leaf_summation_two_sided(
        srcz, pos_qz, th, jax.lax.stop_gradient(ilist), cfg_fmm=cfg_fmm, loc_in=loc_q
    )
    # Back to the input order of the queries
    loc = jnp.zeros_like(loc_q).at[origin_q].set(loc_q)
    return LocalExpansion(loc * G, dim=dim)


def evaluate_at_positions(
        part: PosMass, pos_query: jax.Array, cfg_fmm: FMMConfig, G: float = 1.
    ) -> LocalExpansion:
    """Evaluates the field of the particles at arbitrary query positions.

    **Compatibility:** :compat-jit:`JIT` :compat-shard-local:`Local only`
    :compat-autodiff:`Autodiff`

    **Helpers:** :helper-jit:`.jit`

    The sources and the queries are sorted into one tree as separate particle types, whose
    leaves are split by the number of particles of each type. Only interactions from the
    sources onto the queries are evaluated, so the cost grows with the number of source
    neighbours per query, also for many more queries than sources.
    For example, with :class:`jzfmm.config.WendlandC2Kernel` and
    :class:`jzfmm.config.OpeningBySupport`, the potential of the result is the
    kernel density estimate at the query positions. Results are differentiable
    with respect to the particle positions and masses and the query positions.

    Args:
        part: Source particles with positions of shape ``(N, dim)`` and scalar
            or ``(N,)`` masses. Padded particle data is not supported.
        pos_query: Query positions of shape ``(M, dim)``.
        cfg_fmm: Fast-multipole configuration.
        G: Multiplicative interaction strength.
    Returns:
        Local expansion at the ``M`` query positions.
    """
    if in_shard_map_context():
        raise NotImplementedError("evaluate_at_positions does not support shard_map yet.")
    if getattr(part, "num", None) is not None:
        raise ValueError("evaluate_at_positions does not support padded particle data.")
    _check_kernel_config(cfg_fmm.kernel, dim=part.pos.shape[-1], opening=cfg_fmm.opening)
    return _evaluate_at_positions_two_sided(part, pos_query, cfg_fmm, G=G)
evaluate_at_positions.jit = jax.jit(evaluate_at_positions, static_argnames=("cfg_fmm", "G"))
