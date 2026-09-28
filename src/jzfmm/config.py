from __future__ import annotations

from dataclasses import dataclass, field
import math
import jax
import jax.numpy as jnp

from jztree.config import TreeConfig, LoggingConfig

# ------------------------------------------------------------------------------------------------ #
#                                              Kernels                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class KernelConfig:
    """Base class for radial interaction-kernel configurations."""

    def kind_id(self) -> int:
        """Returns the backend identifier of the kernel."""
        raise NotImplementedError
    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Returns kernel parameters in the requested dtype."""
        raise NotImplementedError

@dataclass(unsafe_hash=True, slots=True)
class PlummerKernel(KernelConfig):
    r"""Plummer-softened inverse-distance kernel.

    Implements :math:`K(r)=-1/\sqrt{r^2+\epsilon^2}`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 0

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class Plummer2DKernel(KernelConfig):
    r"""Two-dimensional Plummer-softened logarithmic kernel.

    Implements :math:`K(r)=\frac{1}{2}\log(r^2+\epsilon^2)`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 1

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class SoftenedDistanceKernel(KernelConfig):
    r"""Softened distance kernel.

    Implements :math:`K(r)=\sqrt{r^2+\epsilon^2}`.

    Args:
        softening: Softening length :math:`\epsilon`.
    """

    softening : float = 1e-3

    def kind_id(self) -> int:
        return 2

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.softening], dtype=dtype)

def _check_boxsize(boxsize, support, name):
    if boxsize is None:
        return
    # Only the nearest periodic image is considered
    if not boxsize > 2 * support:
        raise ValueError(f"{name} requires boxsize > 2*support, got boxsize={boxsize} "
                         f"and support={support}")

@dataclass(unsafe_hash=True, slots=True)
class WendlandC2Kernel(KernelConfig):
    r"""Compactly supported Wendland C2 kernel for kernel density estimates.

    Implements :math:`K(r)=N_d\,(1-q)^4(1+4q)` for :math:`q=r/H<1` and
    :math:`K(r)=0` otherwise, normalized to unit integral with
    :math:`N_2=7/(\pi H^2)` and :math:`N_3=21/(2\pi H^3)`. The returned
    potential is therefore the density :math:`\sum_j m_j K(|x-x_j|)` and the
    returned force is its negative gradient. Evaluate densities at tracer
    positions by adding them as zero-mass particles.

    Use it together with :class:`OpeningBySupport` in the FMM, which evaluates
    all interactions by leaf-leaf direct summation.

    Args:
        support: Support radius :math:`H` beyond which the kernel vanishes.
        dim: Spatial dimension used for the normalization. Must be 2 or 3.
        boxsize: Side length of a periodic box. Pair distances then use the
            nearest periodic image. Must match :paramref:`OpeningBySupport.boxsize`
            and exceed twice the support. ``None`` for open boundaries.
    """

    support : float = 0.1
    dim : int = 3
    boxsize : float | None = None

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError(f"WendlandC2Kernel supports dim=2 or dim=3, got dim={self.dim}")
        if not self.support > 0:
            raise ValueError(f"WendlandC2Kernel requires support > 0, got {self.support}")
        _check_boxsize(self.boxsize, self.support, "WendlandC2Kernel")

    def norm(self) -> float:
        """Returns the normalization constant :math:`N_d`."""
        if self.dim == 2:
            return 7. / (math.pi * self.support**2)
        return 21. / (2. * math.pi * self.support**3)

    def kind_id(self) -> int:
        return 3

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.support, self.norm(), self.boxsize or 0.], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class GaussianKernel(KernelConfig):
    r"""Gaussian kernel for kernel density estimates.

    Implements :math:`K(r)=(2\pi\sigma^2)^{-d/2}\exp(-r^2/(2\sigma^2))`, normalized to unit
    integral. As for :class:`WendlandC2Kernel`, the returned potential is the density
    :math:`\sum_j m_j K(|x-x_j|)` and the returned force is its negative gradient.

    Unlike compact kernels, the Gaussian is non-zero everywhere, so densities stay positive
    and smooth also far from all particles. Use it with :class:`OpeningByGaussianError`,
    which approximates distant node pairs through multipoles with a controlled absolute
    error, or with :class:`OpeningBySupport` to truncate it at a fixed radius.

    Args:
        sigma: Standard deviation :math:`\sigma` of the Gaussian.
        dim: Spatial dimension used for the normalization. Must be 2 or 3.
        boxsize: Side length of a periodic box. Pair distances then use the nearest periodic
            image, which requires ``boxsize > 12*sigma``. Must match the ``boxsize`` of the
            opening criterion. ``None`` for open boundaries.
    """

    sigma : float = 0.1
    dim : int = 3
    boxsize : float | None = None

    def __post_init__(self):
        if self.dim not in (2, 3):
            raise ValueError(f"GaussianKernel supports dim=2 or dim=3, got dim={self.dim}")
        if not self.sigma > 0:
            raise ValueError(f"GaussianKernel requires sigma > 0, got {self.sigma}")
        # Beyond 6 sigma, the neglected images contribute less than 1e-8 of the peak
        _check_boxsize(self.boxsize, 6. * self.sigma, "GaussianKernel")

    def norm(self) -> float:
        """Returns the normalization constant :math:`(2\\pi\\sigma^2)^{-d/2}`."""
        return (2. * math.pi * self.sigma**2)**(-self.dim / 2)

    def kind_id(self) -> int:
        return 4

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.sigma, self.norm(), self.boxsize or 0.], dtype=dtype)

# ------------------------------------------------------------------------------------------------ #
#                                              Opening                                             #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class OpeningCriterionConfig:
    """Base class for FMM opening-criterion configurations."""

    def kind_id(self) -> int:
        """Returns the backend identifier of the opening criterion."""
        raise NotImplementedError
    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Returns criterion parameters in the requested dtype."""
        raise NotImplementedError
    def params_for(self, cfg_fmm: FMMConfig, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        """Returns criterion parameters, which may depend on the kernel and expansion order."""
        return self.params(dtype=dtype)
    def evaluates_far_field(self) -> bool:
        """Whether node pairs that are not opened interact through multipoles."""
        return True
    def uses_node_mass(self) -> bool:
        """Whether the criterion depends on the node masses."""
        return False

@dataclass(unsafe_hash=True, slots=True)
class OpeningByAngle(OpeningCriterionConfig):
    """Geometric opening criterion based on an opening angle.

    Args:
        theta: Maximum opening angle. Smaller values increase accuracy and
            computational cost.
    """

    theta : float = 0.8

    def kind_id(self) -> int:
        return 0

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.theta], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class OpeningBySupport(OpeningCriterionConfig):
    """Opening criterion for compactly supported kernels.

    Opens every node pair whose boxes are closer than the support radius, so
    that all non-vanishing interactions are evaluated by leaf-leaf direct
    summation. Node pairs further apart are discarded without evaluating any
    multipole interactions. The result is therefore exact rather than an
    approximation, and a low multipole order such as ``p=1`` is sufficient.

    Args:
        support: Interaction radius. Must be at least the support of the
            kernel, e.g. :paramref:`WendlandC2Kernel.support`.
        boxsize: Side length of a periodic box, so that node pairs are
            compared through their nearest periodic image. Must match the
            kernel's ``boxsize``. ``None`` for open boundaries.
    """

    support : float = 0.1
    boxsize : float | None = None

    def __post_init__(self):
        _check_boxsize(self.boxsize, self.support, "OpeningBySupport")

    def kind_id(self) -> int:
        return 1

    def evaluates_far_field(self) -> bool:
        return False

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.support, self.boxsize or 0.], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class OpeningByGaussianError(OpeningCriterionConfig):
    r"""Error-controlled opening criterion for :class:`GaussianKernel`.

    The multipole error of the Gaussian does not depend on the opening angle, but on the
    node size relative to :math:`\sigma` and on the distance in units of :math:`\sigma`.
    For each node pair, with :math:`M` the larger of the two node masses, :math:`\rho\sigma`
    half the diagonal of the summed node extents and :math:`t\sigma` the minimum distance
    between the node boxes, the pair is

    - discarded if :math:`M N e^{-t^2/2}\le` ``tol``, since it cannot contribute more,
    - approximated through multipoles if the bound on the Taylor remainder of order
      :math:`n=p+1`, :math:`M N \rho^n/n!\, g_n(t)\le` ``tol``, with
      :math:`g_n(t)=\min(1.0865\sqrt{n!}\,e^{-t^2/4},\,(t^2+n)^{n/2}e^{-t^2/2})`,
    - opened otherwise.

    Here :math:`N` is the kernel normalization. The per-pair bound is rigorous for the
    density and within a factor of about 1.5 of the worst case. The error at a query
    accumulates over all its node pairs, but the individual errors have varying signs, so
    the total error is typically of order ``tol`` (see ``checks/accuracy_checks/kde_gaussian.py``).
    Discarded pairs always bias the density low, by at most ``tol`` per pair.

    ``tol`` is an absolute tolerance in units of the density, e.g. a small fraction of the
    mean density. The kernel parameters are taken from :attr:`FMMConfig.kernel`, which must
    be a :class:`GaussianKernel`. Periodic boundaries follow its ``boxsize``.

    Args:
        tol: Absolute density tolerance per node pair.
    """

    tol : float = 1e-3

    def __post_init__(self):
        if not self.tol > 0:
            raise ValueError(f"OpeningByGaussianError requires tol > 0, got {self.tol}")

    def kind_id(self) -> int:
        return 2

    def uses_node_mass(self) -> bool:
        return True

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        raise TypeError("OpeningByGaussianError depends on the kernel, use params_for(cfg_fmm)")

    def params_for(self, cfg_fmm: FMMConfig, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        kernel = cfg_fmm.kernel
        if not isinstance(kernel, GaussianKernel):
            raise TypeError(f"OpeningByGaussianError requires a GaussianKernel, got {kernel}")
        n = cfg_fmm.p + 1
        log_nfact = math.lgamma(n + 1)
        return jnp.asarray([
            kernel.sigma, math.log(self.tol / kernel.norm()), kernel.boxsize or 0., n,
            -log_nfact, math.log(1.0865) + 0.5 * log_nfact,
        ], dtype=dtype)

@dataclass(unsafe_hash=True, slots=True)
class PotentialField:
    """Base class for external potential fields.

    Assign an instance to :paramref:`SimConfig.external_potential` to add its
    acceleration during time integration. Subclasses normally implement
    :meth:`potential`; :meth:`acceleration` obtains its negative gradient with
    autodiff. Built-in fields are provided by :mod:`jzfmm.external_potential`.

    The external contribution is applied during integration and is not included
    in the :class:`jzfmm.data.LocalExpansion` returned for particle
    self-interactions.
    """

    def potential(
        self, x: jax.Array, t: float | jax.Array = 0., cfg: SimConfig | None = None
    ) -> jax.Array:
        """Evaluates the potential at positions ``x``."""
        raise NotImplementedError
    def acceleration(
        self, x: jax.Array, t: float | jax.Array = 0., cfg: SimConfig | None = None
    ) -> jax.Array:
        """Evaluates acceleration as the negative potential gradient."""
        return -jax.grad(lambda x: jnp.sum(self.potential(x, t=t, cfg=cfg)))(x)

# ------------------------------------------------------------------------------------------------ #
#                                               Units                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class UnitConfig:
    """Defines simulation units relative to common astrophysical units.

    Args:
        pos_in_kpc: Length represented by one simulation position unit in kpc.
        vel_in_kmps: Speed represented by one simulation velocity unit in km/s.
        mass_in_msol: Mass represented by one simulation mass unit in solar
            masses.
    """

    pos_in_kpc: float = 1.
    vel_in_kmps: float = 1.
    mass_in_msol: float = 1.

    def G(self) -> float:
        """Returns the gravitational constant in simulation units."""
        return 4.30071057317063e-06 * self.mass_in_msol / self.pos_in_kpc / self.vel_in_kmps**2

@dataclass(unsafe_hash=True, slots=True)
class IntegratorConfig:
    """Base class for time-integrator configurations."""

@dataclass(unsafe_hash=True, slots=True)
class DKDConfig(IntegratorConfig):
    """Drift-kick-drift leapfrog integrator configuration."""

@dataclass(unsafe_hash=True, slots=True)
class KDKConfig(IntegratorConfig):
    """Kick-drift-kick leapfrog integrator configuration."""

@dataclass(unsafe_hash=True, slots=True)
class DKDLatticeConfig(IntegratorConfig):
    """Drift-kick-drift integrator using integer phase-space coordinates.

    Args:
        dx: Position lattice spacing.
        dv: Velocity lattice spacing.
        int_dtype: Integer dtype used for lattice coordinates. Both
            :class:`jax.numpy.int32` and :class:`jax.numpy.int64` are supported,
            independently of the floating-point format used elsewhere.
    """

    dx: float = 1e-4
    dv: float = 1e-4
    int_dtype: type = jnp.int32

# ------------------------------------------------------------------------------------------------ #
#                                               Force                                              #
# ------------------------------------------------------------------------------------------------ #

@dataclass(unsafe_hash=True, slots=True)
class DirectSummationConfig:
    """Configures direct pair summation.

    Args:
        kernel: Radial interaction kernel.
        kahan_summation: Whether to use compensated summation.
        remove_self_interaction: Whether to exclude each particle's interaction
            with itself.
    """

    kernel : KernelConfig = field(default_factory=PlummerKernel)
    kahan_summation : bool = True
    remove_self_interaction : bool = True

@dataclass(unsafe_hash=True, slots=True)
class FMMConfig:
    """Configures fast-multipole force evaluation.

    Args:
        tree: Tree construction configuration.
        kernel: Radial interaction kernel.
        p: Multipole expansion order.
        opening: Node-opening criterion.
        alloc_fac_ilist: Interaction-list entries allocated per leaf-node
            buffer entry; the total capacity is approximately this factor
            times the allocated number of leaf nodes.
        alloc_fac_comm_nodes: Node communication-buffer capacity as a multiple
            of the local node-buffer size.
        alloc_fac_comm_particles: Particle communication-buffer capacity as a
            multiple of the local particle-buffer size.
        kahan_summation: Whether to use compensated summation where available.
        remove_self_interaction: Whether to exclude each particle's interaction
            with itself.
    """

    # Tree
    tree : TreeConfig = field(
        default_factory=lambda: TreeConfig(
            mass_centered=False, alloc_fac_nodes=1.2, regularization=None, coarse_fac=4.0
        )
    )

    # Kernel
    kernel : KernelConfig = field(default_factory=PlummerKernel)

    # Multipole order:
    p : int = 5

    # Opening criterion
    opening : OpeningCriterionConfig = field(default_factory=OpeningByAngle)

    # Memory
    alloc_fac_ilist : float = 64.
    alloc_fac_comm_nodes : float = 1.5
    alloc_fac_comm_particles : float = 1.5

    # Other
    kahan_summation : bool = False
    remove_self_interaction : bool = True

@dataclass(unsafe_hash=True, slots=True)
class SimConfig:
    """Collects force, unit, logging, and integration configuration.

    Args:
        force: Force configuration, or ``None`` to disable self-gravity.
        units: Simulation unit configuration.
        logging: Logging configuration.
        external_potential: Optional external potential field.
        integrator: Time-integrator configuration.
    """

    # Sub cfg objects
    force : FMMConfig | DirectSummationConfig | None = field(default_factory=FMMConfig)
    units : UnitConfig = field(default_factory=UnitConfig)
    logging : LoggingConfig = field(default_factory=LoggingConfig)

    # flexible objects
    external_potential : PotentialField | None = None

    # Time integration
    integrator: IntegratorConfig = field(default_factory=DKDConfig)
