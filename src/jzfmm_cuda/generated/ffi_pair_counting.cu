// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../common/math.cuh"
#include "../pair_counting.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

using DT = ffi::DataType;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: LeafLeafPairCount                         */
/* ---------------------------------------------------------------------------------------------- */


ffi::Error LeafLeafPairCountFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer node_range,
    ffi::AnyBuffer spl_recv,
    ffi::AnyBuffer spl_src,
    ffi::AnyBuffer spl_ilist,
    ffi::AnyBuffer ilist_isrc,
    ffi::AnyBuffer posm_recv,
    ffi::AnyBuffer posm_src,
    ffi::AnyBuffer r2_edges,
    ffi::AnyBuffer bin_lut,
    ffi::AnyBuffer lut_params,
    ffi::AnyBuffer params,
    ffi::Result<ffi::AnyBuffer> counts,
    ffi::Result<ffi::AnyBuffer> wcounts,
    bool remove_self_pairs,
    bool weighted,
    size_t block_size
) {
    int nbins = r2_edges.element_count() - 1;
    int dim = posm_recv.dimensions()[1] - 1;
    DT tvec = posm_recv.element_type();
    dim3 blockDim(block_size);
    dim3 gridDim(spl_recv.element_count() - 1);
    size_t smem = (nbins + 1 + blockDim.x * (dim + 1)) * (posm_src.element_type() == DT::F64 ? sizeof(double) : sizeof(float)) + div_ceil(blockDim.x, 32) * nbins * 2 * sizeof(float) + blockDim.x * sizeof(int) + bin_lut.element_count() * sizeof(unsigned short);

    // Initialize output buffers
    cudaMemsetAsync(counts->untyped_data(), 0, counts->size_bytes(), stream);
cudaMemsetAsync(wcounts->untyped_data(), 0, wcounts->size_bytes(), stream);

    // Build a bundled argument list for cudaLaunchKernel
    void* node_range_arg = node_range.untyped_data();
    void* spl_recv_arg = spl_recv.untyped_data();
    void* spl_src_arg = spl_src.untyped_data();
    void* spl_ilist_arg = spl_ilist.untyped_data();
    void* ilist_isrc_arg = ilist_isrc.untyped_data();
    void* posm_recv_arg = posm_recv.untyped_data();
    void* posm_src_arg = posm_src.untyped_data();
    void* r2_edges_arg = r2_edges.untyped_data();
    void* bin_lut_arg = bin_lut.untyped_data();
    void* lut_params_arg = lut_params.untyped_data();
    void* params_arg = params.untyped_data();
    void* counts_arg = counts->untyped_data();
    void* wcounts_arg = wcounts->untyped_data();
    void* args[] = {
        &node_range_arg,
        &spl_recv_arg,
        &spl_src_arg,
        &spl_ilist_arg,
        &ilist_isrc_arg,
        &posm_recv_arg,
        &posm_src_arg,
        &r2_edges_arg,
        &bin_lut_arg,
        &lut_params_arg,
        &params_arg,
        &counts_arg,
        &wcounts_arg,
        &nbins,
        &remove_self_pairs
    };


    // We have template parameters, so we need to instantiate all valid templates.
    // We select a function pointer through a map with a stable, type-erased signature.
    using TTuple = std::tuple<bool, int, DT>;
    using TFunc = const void*;

    static const std::map<TTuple, TFunc> instance_map = {
        { {true, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairCount<true, 2, float>) },
        { {true, 2, DT::F64}, reinterpret_cast<TFunc>(&LeafLeafPairCount<true, 2, double>) },
        { {true, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairCount<true, 3, float>) },
        { {true, 3, DT::F64}, reinterpret_cast<TFunc>(&LeafLeafPairCount<true, 3, double>) },
        { {false, 2, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairCount<false, 2, float>) },
        { {false, 2, DT::F64}, reinterpret_cast<TFunc>(&LeafLeafPairCount<false, 2, double>) },
        { {false, 3, DT::F32}, reinterpret_cast<TFunc>(&LeafLeafPairCount<false, 3, float>) },
        { {false, 3, DT::F64}, reinterpret_cast<TFunc>(&LeafLeafPairCount<false, 3, double>) }
    };

    const TTuple key = TTuple(weighted, dim, tvec);

    const auto it = instance_map.find(key);
    if (it == instance_map.end()) {
        return ffi::Error::Internal(
            "\nUnsupported template parameter combination for (weighted, dim, tvec)"\
            " in LeafLeafPairCountFFIHost -- Only supporting:\n"\
            "(true, 2, float), (true, 2, double), (true, 3, float), (true, 3, double), (false, 2, float), (false, 2, double), (false, 3, float), (false, 3, double)"
        );
    }
    const void* instance = it->second;

    cudaLaunchKernel(
        instance,
        gridDim,
        blockDim,
        args,
        smem,
        stream
    );

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LeafLeafPairCountFFI, LeafLeafPairCountFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // node_range
        .Arg<ffi::AnyBuffer>() // spl_recv
        .Arg<ffi::AnyBuffer>() // spl_src
        .Arg<ffi::AnyBuffer>() // spl_ilist
        .Arg<ffi::AnyBuffer>() // ilist_isrc
        .Arg<ffi::AnyBuffer>() // posm_recv
        .Arg<ffi::AnyBuffer>() // posm_src
        .Arg<ffi::AnyBuffer>() // r2_edges
        .Arg<ffi::AnyBuffer>() // bin_lut
        .Arg<ffi::AnyBuffer>() // lut_params
        .Arg<ffi::AnyBuffer>() // params
        .Ret<ffi::AnyBuffer>() // counts
        .Ret<ffi::AnyBuffer>() // wcounts
        .Attr<bool>("remove_self_pairs")
        .Attr<bool>("weighted")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_pair_counting, m) {
    m.def("LeafLeafPairCount", []() { return EncapsulateFfiCall(&LeafLeafPairCountFFI); });
}