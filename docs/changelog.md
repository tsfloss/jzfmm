# Changelog

## Unreleased

- Added the compactly supported `WendlandC2Kernel` and the `OpeningBySupport` criterion for kernel density estimates. Node pairs within the support radius are opened down to leaf-leaf direct summation, while all other node pairs are discarded without M2L evaluations. Densities at tracer positions are obtained by adding zero-mass particles, and are differentiable with respect to particle positions and masses. This requires the updated CUDA backend.
- Added `query_mask` to `fast_multipole_method` and the helper `evaluate_at_positions`. Node pairs without any query particle are skipped, which saves most of the work when results are only needed at a small subset of particles or at tracer positions. Gradients with respect to all particles remain exact.
- Added periodic boundaries for kernel density estimates via `boxsize` on `WendlandC2Kernel` and `OpeningBySupport`. Pair distances and the node-pair test use the nearest periodic image, and positions do not need to be wrapped into the box. Periodic boundaries are not supported for kernels with a far field.

## 1.0.1

- Fixed invalid expansion scaling for tree cells spanning positive and negative coordinates and for tiny or coincident-particle cells. These cases could produce NaNs in FMM results and gradients, including MMD losses used in differentiable simulations. Updating is recommended, especially for differentiable workloads; the fix requires the updated CUDA backend as well as the Python package.
- Added regression tests for forces, gradients, and reversible integration, with a compact subset retained in `--quick` mode.
- Improved the `hello_world.py` dependency hints and refreshed the getting-started guide and example outputs.

## 1.0.0

First public release.
