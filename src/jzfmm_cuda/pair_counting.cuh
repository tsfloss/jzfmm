#ifndef PAIR_COUNTING_CUH
#define PAIR_COUNTING_CUH

#include "common/data.cuh"
#include "common/math.cuh"
#include "common/iterators.cuh"
#include "radial_kernels.cuh"

/* ---------------------------------------------------------------------------------------------- */
/*                                 Leaf-Leaf Pair Count Histogram                                 */
/* ---------------------------------------------------------------------------------------------- */

// Index of the bin with r2_edges[k] <= r2 < r2_edges[k+1], or -1 outside of all bins.
// r2_edges must be ascending and have nbins+1 entries.
//
// The bit pattern of a positive float increases monotonically with its value and is a
// piecewise linear approximation of its logarithm. A table over (bits - key0) >> shift
// therefore resolves logarithmic and linear bins alike. It stores the bin of the lowest
// value of each cell, which the comparisons with the edges then correct exactly. (Moving
// down is only needed if rounding r2 to float crosses a cell boundary.)
template<typename tvec>
__forceinline__ __device__ int find_bin(
    tvec r2, const tvec* r2_edges, int nbins, const unsigned short* lut, int nlut, int key0,
    int shift
) {
    if(!(r2 >= r2_edges[0]) || !(r2 < r2_edges[nbins]))
        return -1;
    int cell = (__float_as_int(float(r2)) - key0) >> shift;
    int bin = lut[min(max(cell, 0), nlut - 1)];
    while(bin > 0 && r2 < r2_edges[bin])
        bin--;
    while(r2 >= r2_edges[bin + 1])
        bin++;
    return bin;
}

template <bool weighted, int dim, typename tvec>
__global__ void LeafLeafPairCount(
    // inputs:
    const int2* node_range,
    const int* spl_recv,
    const int* spl_src,
    const int* spl_ilist,
    const int* ilist_isrc,
    const PosMass<dim,tvec>* posm_recv,
    const PosMass<dim,tvec>* posm_src,
    const tvec* r2_edges,
    const unsigned short* bin_lut,
    const int* lut_params,
    const tvec* params,
    // outputs (zero initialized, accumulated with atomics):
    unsigned long long* counts,
    double* wcounts,
    // attributes:
    int nbins,
    const bool remove_self_pairs
) {
    // Histograms the separations of all particle pairs between each receiver leaf and the
    // source leaves of its interaction list. Every ordered pair (i,j) is counted once, so
    // with a symmetric interaction list an unordered pair is counted twice. If weighted,
    // wcounts additionally accumulates m_i*m_j.
    const tvec boxsize = params[0];

    int2 nrange = node_range[0];
    int nodeid = nrange.x + blockIdx.x;
    if (nodeid >= nrange.y)
        return;

    int2 prange = {spl_recv[nodeid], spl_recv[nodeid + 1]};
    int num = prange.y-prange.x;
    if (num <= 0)
        return;

    // Same layout as LeafLeafPairSummation: blockDim.x -> (num, n_write) + residuals
    int n_write = blockDim.x / num;
    int a_write = threadIdx.x % num;
    int read_b_offset = threadIdx.x / num;
    bool valid = threadIdx.x < num * n_write;

    int ia = prange.x + a_write;
    PosMass<dim,tvec> xa = posm_recv[ia];

    // Shared memory: source particles | r2 edges | per-warp histograms | source ids | weights
    //                | bin lookup table
    // (The particles come first, since they may require a larger alignment)
    extern __shared__ unsigned char pair_count_smem[];
    const int nwarps = div_ceil(blockDim.x, 32);
    const int warp = threadIdx.x / 32;
    PosMass<dim,tvec>* xm_b = reinterpret_cast<PosMass<dim,tvec>*>(pair_count_smem);
    tvec* r2e = reinterpret_cast<tvec*>(xm_b + blockDim.x);
    unsigned int* hist = reinterpret_cast<unsigned int*>(r2e + (nbins + 1));
    int* id_b = reinterpret_cast<int*>(hist + nwarps * nbins);
    float* whist = reinterpret_cast<float*>(id_b + blockDim.x);
    unsigned short* lut = reinterpret_cast<unsigned short*>(whist + nwarps * nbins);
    const int key0 = lut_params[0], shift = lut_params[1], nlut = lut_params[2];

    for(int i = threadIdx.x; i < nbins + 1; i += blockDim.x)
        r2e[i] = r2_edges[i];
    for(int i = threadIdx.x; i < nlut; i += blockDim.x)
        lut[i] = bin_lut[i];
    for(int i = threadIdx.x; i < nwarps * nbins; i += blockDim.x) {
        hist[i] = 0;
        if constexpr (weighted)
            whist[i] = 0.f;
    }
    // (SegmentManager synchronizes before first use of shared memory)

    __shared__ int2 segments[32];
    SegmentManager seg_mgr(
        ilist_isrc,
        spl_src,
        segments,
        spl_ilist[nodeid],
        spl_ilist[nodeid + 1],
        32
    );

    unsigned int* my_hist = hist + warp * nbins;
    float* my_whist = whist + warp * nbins;

    while(!seg_mgr.finished()) {
        int id = seg_mgr.next();

        if(id >= 0) {
            xm_b[threadIdx.x] = posm_src[id];
            id_b[threadIdx.x] = id;
        }
        __syncthreads();

        if(valid) {
            for(int ib=read_b_offset; ib < seg_mgr.num_loaded; ib += n_write) {
                Vec<dim,tvec> dx = periodic_wrap<dim,tvec>(xm_b[ib].pos - xa.pos, boxsize);
                int bin = find_bin<tvec>(dx.norm2(), r2e, nbins, lut, nlut, key0, shift);
                if(remove_self_pairs && (id_b[ib] == ia))
                    bin = -1;
                if(bin >= 0) {
                    atomicAdd(&my_hist[bin], 1u);
                    if constexpr (weighted)
                        atomicAdd(&my_whist[bin], float(xa.mass * xm_b[ib].mass));
                }
            }
        }
        __syncthreads();
    }

    // Reduce the per-warp histograms and add them to the global result
    for(int k = threadIdx.x; k < nbins; k += blockDim.x) {
        unsigned long long c = 0;
        double w = 0.;
        for(int iw = 0; iw < nwarps; iw++) {
            c += hist[iw * nbins + k];
            if constexpr (weighted)
                w += whist[iw * nbins + k];
        }
        if(c > 0) {
            atomicAdd(&counts[k], c);
            if constexpr (weighted)
                atomicAdd(&wcounts[k], w);
        }
    }
}

#endif
