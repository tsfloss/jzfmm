#ifndef RADIAL_KERNELS_H
#define RADIAL_KERNELS_H

#include "common/math.cuh"

static constexpr int RADIAL_KERNEL_PLUMMER = 0;
static constexpr int RADIAL_KERNEL_PLUMMER_2D = 1;
static constexpr int RADIAL_KERNEL_SOFTENED_DISTANCE = 2;
static constexpr int RADIAL_KERNEL_WENDLAND_C2 = 3;

// Minimum-image convention: wraps each component of a displacement into
// [-boxsize/2, boxsize/2]. boxsize <= 0 disables wrapping.
template<int dim, typename tvec>
__device__ __forceinline__ Vec<dim,tvec> periodic_wrap(Vec<dim,tvec> dx, tvec boxsize) {
    if(boxsize > tvec(0)) {
        // Multiplying by the inverse avoids a division per pair. It can round differently
        // only at |dx| = boxsize/2, where both images are equally far and beyond the support.
        const tvec inv_boxsize = tvec(1) / boxsize;
        #pragma unroll
        for(int i = 0; i < dim; i++)
            dx[i] -= boxsize * rint(dx[i] * inv_boxsize);
    }
    return dx;
}

// scale_exp is zero for direct interactions, or bounded_expansion_exponent
// for M2L: both 2^scale_exp and its reciprocal are normal floating-point values.
template<int radial_kernel_kind>
struct RadialKernel;

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        tvec rinv = rsqrt(scaled_r2 + softening * softening);
        tvec rinv2 = rinv * rinv;
        coeffs[0] = -rinv;

        #pragma unroll
        for(int n = 1; n <= p; n++) {
            coeffs[n] = -tvec(2*n - 1) * coeffs[n-1] * rinv2;
        }

        coeffs = coeffs * normal_pow2<tvec>(-scale_exp);
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_PLUMMER_2D> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        const tvec rsoft2 = scaled_r2 + softening * softening;
        const tvec sinv = tvec(1) / rsoft2;
        coeffs[0] = tvec(0.5) * log(rsoft2)
            + tvec(scale_exp) * tvec(0.6931471805599453094);

        if constexpr (p >= 1) {
            coeffs[1] = sinv;

            #pragma unroll
            for(int n = 2; n <= p; n++) {
                coeffs[n] = -tvec(2 * (n - 1)) * coeffs[n - 1] * sinv;
            }
        }
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE> {
    template<typename tvec>
    struct Params {
        tvec softening;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        const tvec softening = params.softening * normal_pow2<tvec>(-scale_exp);
        const tvec rsoft = sqrt(scaled_r2 + softening * softening);
        coeffs[0] = rsoft;

        if constexpr (p >= 1) {
            const tvec rinv = tvec(1) / rsoft;
            const tvec rinv2 = rinv * rinv;
            coeffs[1] = rinv;

            #pragma unroll
            for(int n = 2; n <= p; n++) {
                coeffs[n] = -tvec(2*n - 3) * coeffs[n - 1] * rinv2;
            }
        }

        coeffs = coeffs * normal_pow2<tvec>(scale_exp);
    }
};

template<>
struct RadialKernel<RADIAL_KERNEL_WENDLAND_C2> {
    template<typename tvec>
    struct Params {
        tvec support;
        tvec norm;
        tvec boxsize; // periodic box size, <= 0 if not periodic
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec> make_params(const tvec* params) {
        return Params<tvec>{params[0], params[1], params[2]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p+1,tvec>& coeffs,
        int scale_exp = 0
    ) {
        // s = r^2 / R^2, R = 2^scale_exp
        // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
        // With u = q^2 = s / H'^2, H' = H / R:
        // K / norm = (1-q)^4 (1+4q) = 1 - 10u - 15u^2 + u^{3/2} (20 + 4u)  for u < 1
        const tvec support = params.support * normal_pow2<tvec>(-scale_exp);
        const tvec hinv2 = tvec(1) / (support * support);
        const tvec u = scaled_r2 * hinv2;
        // Also false for NaN, e.g. if the scaled support underflows
        const bool inside = u < tvec(1);
        const tvec q = sqrt(u);
        // The half-integer powers are singular at u = 0 for n >= 2. Setting uinv = 0 there
        // yields zero instead. It is only multiplied with dx = 0 in the pair VJP.
        const tvec uinv = (u > tvec(0)) ? tvec(1) / u : tvec(0);
        const tvec fac = tvec(2) * hinv2;

        tvec pw3 = u * q; // u^{3/2 - n}
        tvec pw5 = u * u * q; // u^{5/2 - n}
        tvec c3 = tvec(20);
        tvec c5 = tvec(4);
        tvec scale = params.norm;

        #pragma unroll
        for(int n = 0; n <= p; n++) {
            tvec poly;
            if(n == 0)
                poly = tvec(1) - u * (tvec(10) + tvec(15) * u);
            else if(n == 1)
                poly = tvec(-10) - tvec(30) * u;
            else if(n == 2)
                poly = tvec(-30);
            else
                poly = tvec(0);

            tvec value = scale * (poly + c3 * pw3 + c5 * pw5);
            if(n == 0) {
                // The expanded form cancels near q = 1 and can become slightly negative
                const tvec omq = tvec(1) - q;
                value = scale * (omq * omq) * (omq * omq) * (tvec(1) + tvec(4) * q);
            }
            coeffs[n] = inside ? value : tvec(0);

            c3 *= tvec(1.5) - tvec(n);
            c5 *= tvec(2.5) - tvec(n);
            pw3 *= uinv;
            pw5 *= uinv;
            scale *= fac;
        }
    }
};

// Periodic box size used for pair displacements, or 0 for open boundaries.
// Only compactly supported kernels can be periodic, since they have no far field.
template<int radial_kernel_kind, typename tvec>
__device__ __forceinline__ tvec radial_kernel_boxsize(
    const typename RadialKernel<radial_kernel_kind>::template Params<tvec>& params
) {
    if constexpr (radial_kernel_kind == RADIAL_KERNEL_WENDLAND_C2)
        return params.boxsize;
    else
        return tvec(0);
}

template<int p, typename tvec>
__device__ __forceinline__ void evaluate_radial_kernel_derivatives(
    int radial_kernel_kind,
    tvec scaled_r2,
    const tvec* params,
    int scale_exp,
    Vec<p+1,tvec>& coeffs
) {
    // s = scaled_r2 = r^2 / R^2, R = 2^scale_exp
    // coeffs[n] = 2^n d^n/ds^n K(R sqrt(s)).
    switch(radial_kernel_kind) {
        case RADIAL_KERNEL_WENDLAND_C2: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_WENDLAND_C2>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_WENDLAND_C2>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
        case RADIAL_KERNEL_SOFTENED_DISTANCE: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_SOFTENED_DISTANCE>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
        case RADIAL_KERNEL_PLUMMER_2D: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_PLUMMER_2D>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_PLUMMER_2D>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
        case RADIAL_KERNEL_PLUMMER:
        default: {
            auto kernel_params = RadialKernel<RADIAL_KERNEL_PLUMMER>::template make_params<tvec>(params);
            RadialKernel<RADIAL_KERNEL_PLUMMER>::template r2_derivative_coeffs<p,tvec>(
                scaled_r2, kernel_params, coeffs, scale_exp
            );
            break;
        }
    }
}

template<int radial_kernel_kind, typename tvec>
__device__ __forceinline__ tvec radial_kernel_value(
    tvec r2,
    typename RadialKernel<radial_kernel_kind>::template Params<tvec> params
) {
    Vec<1,tvec> coeffs;
    RadialKernel<radial_kernel_kind>::template r2_derivative_coeffs<0,tvec>(
        r2, params, coeffs
    );
    return coeffs[0];
}

#endif // RADIAL_KERNELS_H
