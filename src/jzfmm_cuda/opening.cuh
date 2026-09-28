#ifndef OPENING_H
#define OPENING_H

#include "common/data.cuh"
#include "common/math.cuh"
#include "radial_kernels.cuh"

static constexpr int OPENING_BY_ANGLE = 0;
static constexpr int OPENING_BY_SUPPORT = 1;
static constexpr int OPENING_BY_GAUSSIAN_ERROR = 2;

// What to do with a pair of nodes. Every criterion must be symmetric in A and B, so that
// the interaction lists stay symmetric: the backward pass relies on this.
static constexpr int INTERACTION_DISCARD = 0;     // negligible, skip entirely
static constexpr int INTERACTION_APPROXIMATE = 1; // interact through M2L
static constexpr int INTERACTION_OPEN = 2;        // descend to the children

// Geometry of a node pair, divided by 2^scale_exp to avoid floating point overflows
template<int dim, typename tvec>
struct ScaledPairGeometry {
    int scale_exp;
    Vec<dim,tvec> dx;     // center offset
    Vec<dim,tvec> extent; // sum of both node extents per axis
    bool unbounded;       // at least one node has no finite multipole expansion

    __device__ __forceinline__ ScaledPairGeometry(
        Node<dim,tvec> nodeA, Node<dim,tvec> nodeB, tvec boxsize = tvec(0)
    ) {
        // 2**levels is the per dimension extent
        const Vec<dim,int32_t> levelsA = lvl_vec<dim>(nodeA.level);
        const Vec<dim,int32_t> levelsB = lvl_vec<dim>(nodeB.level);
        // With periodic boundaries the center offset to the nearest image of B also
        // gives the smallest gap, since the gap grows with |dx| on every axis.
        const Vec<dim,tvec> dx_full = periodic_wrap<dim,tvec>(nodeA.center - nodeB.center, boxsize);

        scale_exp = max(levelsA[dim - 1], levelsB[dim - 1]);
        const tvec dx_max = absmax(dx_full);
        if(dx_max != tvec(0))
            scale_exp = max(scale_exp, normal_ilogb(dx_max));

        dx = mulpow2(dx_full, -scale_exp);
        extent = nonpositive_exp2<tvec>(levelsA - scale_exp) + nonpositive_exp2<tvec>(levelsB - scale_exp);
        unbounded = unbounded_node<dim,tvec>(nodeA.level) | unbounded_node<dim,tvec>(nodeB.level);
    }

    // Squared minimum distance between the two node boxes
    __device__ __forceinline__ tvec gap2() const {
        Vec<dim,tvec> gap;
        #pragma unroll
        for(int i = 0; i < dim; i++)
            gap[i] = max(tvec(0), abs(dx[i]) - tvec(0.5) * extent[i]);
        return gap.norm2();
    }
};

template<int opening_criterion_kind>
struct OpeningCriterion;

template<>
struct OpeningCriterion<OPENING_BY_ANGLE> {
    template<typename tvec>
    struct Params {
        tvec theta;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    // mass is max(M_A, M_B). It is unused here.
    template<int dim, typename tvec>
    __device__ __forceinline__ static int action(
        Node<dim,tvec> nodeA, Node<dim,tvec> nodeB, tvec mass, Params<tvec> params
    ) {
        const ScaledPairGeometry<dim,tvec> geo(nodeA, nodeB);
        // Unbounded cells have no finite multipole expansion. Use a predicate,
        // rather than an early return, to avoid another traversal branch.
        const bool open = geo.unbounded | (tvec(0.25) * geo.extent.norm2()
            >= params.theta * params.theta * geo.dx.norm2());
        return open ? INTERACTION_OPEN : INTERACTION_APPROXIMATE;
    }
};

template<>
struct OpeningCriterion<OPENING_BY_SUPPORT> {
    // For compactly supported kernels: node pairs that are not opened do not interact
    // and are discarded without M2L. All interactions are evaluated between leaves.
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
    __device__ __forceinline__ static int action(
        Node<dim,tvec> nodeA, Node<dim,tvec> nodeB, tvec mass, Params<tvec> params
    ) {
        // Open if the minimum distance between the two node boxes is within the support.
        const ScaledPairGeometry<dim,tvec> geo(nodeA, nodeB, params.boxsize);
        // Underflows to zero for tiny supports. "<=" still opens overlapping boxes then.
        const tvec support_scaled = mulpow2(params.support, -geo.scale_exp);
        const bool open = geo.unbounded | (geo.gap2() <= support_scaled * support_scaled);
        return open ? INTERACTION_OPEN : INTERACTION_DISCARD;
    }
};

template<>
struct OpeningCriterion<OPENING_BY_GAUSSIAN_ERROR> {
    // Error-controlled criterion for the Gaussian kernel K = N exp(-r^2 / (2 sigma^2)).
    //
    // A pair (A,B) interacts through M2L with a Taylor expansion of total order p in the
    // displacement delta = (x - z_A) - (y - z_B), |delta| <= rho sigma, where rho sigma is half
    // the diagonal of the summed node extents. Along any direction u, the n-th derivative of
    // the Gaussian at x is N sigma^-n e^{-b^2/2} He_n(a) e^{-a^2/2} (a = u.x / sigma,
    // a^2 + b^2 = |x|^2 / sigma^2). With t the minimum box distance in units of sigma, the
    // Lagrange remainder is bounded per unit source mass by
    //     N rho^n / n! g_n(t),  n = p + 1,
    //     g_n(t) = min(C sqrt(n!) e^{-t^2/4}, (t^2 + n)^{n/2} e^{-t^2/2}),
    // using Cramer's inequality (C = 1.0865) and |He_n(a)| <= (a^2 + n)^{n/2}. Both terms
    // decrease with t. The bound is within a factor of about 1.5 of the exact supremum.
    //
    // With M = max(M_A, M_B), which keeps the criterion symmetric, a pair is
    //   - discarded if M N e^{-t^2/2} <= tol, i.e. it cannot contribute more than tol,
    //   - approximated if M N rho^n / n! g_n(t) <= tol,
    //   - opened otherwise.
    // tol is an absolute tolerance per node pair in units of the kernel's density.
    template<typename tvec>
    struct Params {
        tvec sigma;
        tvec log_tol;       // log(tol / N)
        tvec boxsize;       // periodic box size, <= 0 if not periodic
        tvec order;         // n = p + 1
        tvec log_inv_nfact; // log(1 / n!)
        tvec log_cramer;    // log(C sqrt(n!))
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0], params[1], params[2], params[3], params[4], params[5]};
    }

    template<int dim, typename tvec>
    __device__ __forceinline__ static int action(
        Node<dim,tvec> nodeA, Node<dim,tvec> nodeB, tvec mass, Params<tvec> params
    ) {
        const ScaledPairGeometry<dim,tvec> geo(nodeA, nodeB, params.boxsize);
        if(geo.unbounded)
            return INTERACTION_OPEN;

        // Everything in units of sigma. A vanishing sigma_scaled yields t2 = inf (discard)
        // or rho2 = inf (open), except for touching boxes, where t2 = 0.
        const tvec sigma_scaled = mulpow2(params.sigma, -geo.scale_exp);
        const tvec sinv2 = tvec(1) / (sigma_scaled * sigma_scaled);
        const tvec gap2 = geo.gap2();
        const tvec t2 = (gap2 > tvec(0)) ? gap2 * sinv2 : tvec(0);
        const tvec log_mass = log(mass);

        if(log_mass - tvec(0.5) * t2 <= params.log_tol)
            return INTERACTION_DISCARD;

        // Self pairs are opened down to the leaves, which handle self interactions.
        if(nodeA.level == nodeB.level && absmax(geo.dx) == tvec(0))
            return INTERACTION_OPEN;

        // With periodic boundaries, the nearest image must be the same for all point pairs
        if(params.boxsize > tvec(0)) {
            const tvec half_box = tvec(0.5) * mulpow2(params.boxsize, -geo.scale_exp);
            #pragma unroll
            for(int i = 0; i < dim; i++) {
                if(abs(geo.dx[i]) + tvec(0.5) * geo.extent[i] >= half_box)
                    return INTERACTION_OPEN;
            }
        }

        const tvec n = params.order;
        const tvec rho2 = tvec(0.25) * geo.extent.norm2() * sinv2;
        const tvec log_g = min(
            params.log_cramer - tvec(0.25) * t2,
            tvec(0.5) * n * log(t2 + n) - tvec(0.5) * t2
        );
        const tvec log_err = log_mass + tvec(0.5) * n * log(rho2) + params.log_inv_nfact + log_g;
        return (log_err <= params.log_tol) ? INTERACTION_APPROXIMATE : INTERACTION_OPEN;
    }
};

#endif // OPENING_H
