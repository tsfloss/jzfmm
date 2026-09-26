#ifndef OPENING_H
#define OPENING_H

#include "common/data.cuh"
#include "common/math.cuh"
#include "radial_kernels.cuh"

static constexpr int OPENING_BY_ANGLE = 0;
static constexpr int OPENING_BY_SUPPORT = 1;

template<int opening_criterion_kind>
struct OpeningCriterion;

template<>
struct OpeningCriterion<OPENING_BY_ANGLE> {
    // Node pairs that are not opened interact through M2L
    static constexpr bool evaluates_far_field = true;

    template<typename tvec>
    struct Params {
        tvec theta;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int dim, typename tvec>
    __device__ __forceinline__ static bool should_open(
        Node<dim,tvec> nodeA,
        Node<dim,tvec> nodeB,
        Params<tvec> params
    ) {
        // Opening Criterion
        // 2**levels is the per dimension extent
        // We use a scaling strategy to avoid floating point overflows in the squares
        const Vec<dim,int32_t> levelsA = lvl_vec<dim>(nodeA.level);
        const Vec<dim,int32_t> levelsB = lvl_vec<dim>(nodeB.level);
        const Vec<dim,tvec> dx = nodeA.center - nodeB.center;

        int scale_exp = max(levelsA[dim - 1], levelsB[dim - 1]);
        const tvec dx_max = absmax(dx);
        if(dx_max != tvec(0))
            scale_exp = max(scale_exp, normal_ilogb(dx_max));

        const Vec<dim,tvec> dx_scaled = mulpow2(dx, -scale_exp);
        const Vec<dim,tvec> extent_scaled =
            nonpositive_exp2<tvec>(levelsA - scale_exp)
            + nonpositive_exp2<tvec>(levelsB - scale_exp);

        // These cells have no finite multipole expansion. Use a predicate,
        // rather than an early return, to avoid another traversal branch.
        const bool unbounded = unbounded_node<dim,tvec>(nodeA.level)
            | unbounded_node<dim,tvec>(nodeB.level);
        return unbounded | (tvec(0.25) * extent_scaled.norm2()
            >= params.theta * params.theta * dx_scaled.norm2());
    }
};

template<>
struct OpeningCriterion<OPENING_BY_SUPPORT> {
    // For compactly supported kernels: node pairs that are not opened do not interact
    // and are discarded without M2L. All interactions are evaluated between leaves.
    static constexpr bool evaluates_far_field = false;

    template<typename tvec>
    struct Params {
        tvec support;
        tvec boxsize; // periodic box size, <= 0 if not periodic
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0], params[1]};
    }

    template<int dim, typename tvec>
    __device__ __forceinline__ static bool should_open(
        Node<dim,tvec> nodeA,
        Node<dim,tvec> nodeB,
        Params<tvec> params
    ) {
        // Open if the minimum distance between the two node boxes is within the support.
        // This is symmetric in A and B, so leaf-leaf interaction lists remain symmetric.
        // We use the same scaling strategy as OPENING_BY_ANGLE to avoid overflows.
        // With periodic boundaries the center offset to the nearest image of B also
        // gives the smallest gap, since the gap grows with |dx| on every axis.
        const Vec<dim,int32_t> levelsA = lvl_vec<dim>(nodeA.level);
        const Vec<dim,int32_t> levelsB = lvl_vec<dim>(nodeB.level);
        const Vec<dim,tvec> dx = periodic_wrap<dim,tvec>(nodeA.center - nodeB.center, params.boxsize);

        int scale_exp = max(levelsA[dim - 1], levelsB[dim - 1]);
        const tvec dx_max = absmax(dx);
        if(dx_max != tvec(0))
            scale_exp = max(scale_exp, normal_ilogb(dx_max));

        const Vec<dim,tvec> dx_scaled = mulpow2(dx, -scale_exp);
        const Vec<dim,tvec> extent_scaled =
            nonpositive_exp2<tvec>(levelsA - scale_exp)
            + nonpositive_exp2<tvec>(levelsB - scale_exp);

        Vec<dim,tvec> gap;
        #pragma unroll
        for(int i = 0; i < dim; i++)
            gap[i] = max(tvec(0), abs(dx_scaled[i]) - tvec(0.5) * extent_scaled[i]);

        // Underflows to zero for tiny supports. "<=" still opens overlapping boxes then.
        const tvec support_scaled = mulpow2(params.support, -scale_exp);

        const bool unbounded = unbounded_node<dim,tvec>(nodeA.level)
            | unbounded_node<dim,tvec>(nodeB.level);
        return unbounded | (gap.norm2() <= support_scaled * support_scaled);
    }
};

#endif // OPENING_H
