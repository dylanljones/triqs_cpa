# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-06-13

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from triqs.gf import Gf, MeshImFreq, MeshReFreq
from triqs.gf.descriptors import Base
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import SumkDiscreteFromLattice
from triqs.utility import mpi

Args = Optional[Tuple[Any, ...]]
Kwargs = Optional[Dict[str, Any]]
Func = Callable[[Gf], Gf]

Onsite = Union[float, Sequence[float], Gf]


def _signed_sqrt(z: np.ndarray) -> np.ndarray:
    """Square root with correct sign for triangular lattice."""
    sign = np.where((z.real < 0) & (z.imag < 0), -1, 1)
    return sign * np.lib.scimath.sqrt(z)


def _u_ellipk(z: np.ndarray) -> np.ndarray:
    """Complete elliptic integral of first kind `scipy.special.ellip` for complex arguments.

    Wraps the `mpmath` implementation `mpmath.fp.ellipf` using `numpy.frompyfunc`.
    """
    from mpmath import fp

    _ellipk_z = np.frompyfunc(partial(fp.ellipf, np.pi / 2), 1, 1)
    ellipk = _ellipk_z(np.asarray(z, dtype=complex))
    try:
        ellipk = ellipk.astype(complex)
    except AttributeError:  # complex not np.ndarray
        pass
    return ellipk


def gf_z_bethe(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function of Bethe lattice for infinite coordination number.

    .. math:: G(z) = 2(z - s\sqrt{z^2 - D^2})/D^2

    where :math:`D` is the half bandwidth and :math:`s=sgn[ℑ{ξ}]`. See
    [georges1996]_.

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the Bethe lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the Bethe Green's function.

    References
    ----------
    .. [georges1996] Georges et al., Rev. Mod. Phys. 68, 13 (1996)
       https://doi.org/10.1103/RevModPhys.68.13

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gf_z_bethe(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    z_rel = np.array(z / half_bandwidth, dtype=np.complex256)
    g = 2.0 / half_bandwidth * z_rel * (1 - np.sqrt(1 - z_rel**-2))
    return g.astype(dtype=complex, copy=False)


def gf_z_onedim(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function of the 1D lattice.

    .. math:: G(z) = \frac{1}{2 π} ∫_{-π}^{π}\frac{dϕ}{z - D\cos(ϕ)}

    where :math:`D` is the half bandwidth. The integral can be evaluated in the
    complex plane along the unit circle. See [economou2006]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 1D lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the square lattice Green's function.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gf_z_onedim(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.ylabel("G*D")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    z_rel_inv = half_bandwidth / z
    return 1.0 / half_bandwidth * z_rel_inv / np.lib.scimath.sqrt(1 - z_rel_inv**2)


def gf_z_square(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function of the 2D square lattice.

    .. math::
        G(z) = \frac{2}{πz} ∫^{π/2}_{0} \frac{dϕ}{\sqrt{1 - (D/z)^2 \cos^2ϕ}}

    where :math:`D` is the half bandwidth and the integral is the complete
    elliptic integral of first kind. See [economou2006]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the square lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/4`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the square lattice Green's function.

    References
    ----------
    .. [economou2006] Economou, E. N. Green's Functions in Quantum Physics.
       Springer, 2006.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gf_z_square(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    z_rel_inv = half_bandwidth / z
    elliptic = _u_ellipk(z_rel_inv**2)
    gf_z = 2.0 / np.pi / half_bandwidth * z_rel_inv * elliptic
    return gf_z


def gf_z_box(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function corresponding to a box DOS.

    .. math:: G(z) = \ln(\frac{z + D}{z - D}) / 2D

    Parameters
    ----------
    z : complex array_like or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the box DOS.

    Returns
    -------
    complex np.ndarray or complex
        Value of the Green's function corresponding to a box DOS.

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500)
    >>> gf_ww = gf_z_box(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(ww, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.xlim(left=ww.min(), right=ww.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    z_rel = z / half_bandwidth
    return 0.5 / half_bandwidth * np.emath.log((z_rel + 1) / (z_rel - 1))


def gf_z_triangular(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function of the 2D triangular lattice.

    Note, that the spectrum is asymmetric and in :math:`[-2D/3, 4D/3]`,
    where :math:`D` is the half-bandwidth.
    The Green's function is evaluated as complete elliptic integral of first
    kind, see [horiguchi1972]_.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the triangular lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=4D/9`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the triangular lattice Green's function.

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=500, dtype=complex) + 1e-64j
    >>> gf_ww = gf_z_triangular(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.axvline(-2/3, color='black', linewidth=0.8)
    >>> _ = plt.axvline(+4/3, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    D = half_bandwidth * 4 / 9
    z = 1.0 / D * np.asarray(z)
    shape = z.shape
    z = z.reshape(-1)
    advanced = z.imag < 0
    z = np.where(advanced, np.conj(z), z)  # calculate retarded only, and use symmetry
    singular = D * z == -1  # try singularity which needs to be avoided
    z[singular] = 0  # mock value to avoid errors
    rr = _signed_sqrt(2 * z + 3)
    gg = 4.0 / (_signed_sqrt(rr - 1) ** 3 * _signed_sqrt(rr + 3))  # eq (2.9)
    kk = _signed_sqrt(rr) * gg  # eq (2.11)
    mm = kk**2
    K = np.asarray(_u_ellipk(mm))
    # eqs (2.22) and eq (2.18), fix correct plane
    K[kk.imag > 0] += 2j * _u_ellipk(1 - mm[kk.imag > 0])
    gf_z = 1 / np.pi / D * gg * K  # eq (2.6)
    gf_z[singular] = 0 - 1j * np.inf
    return np.where(advanced, np.conj(gf_z), gf_z).reshape(shape)  # return to advanced by symmetry


def gf_z_honeycomb(z: np.ndarray, half_bandwidth: float) -> np.ndarray:
    r"""Local Green's function of the 2D honeycomb lattice.

    The Green's function of the 2D honeycomb lattice can be expressed in terms
    of the 2D triangular lattice `gftool.lattice.triangular.gf_z`,
    see [horiguchi1972]_.

    The Green's function has singularities at `z=±half_bandwidth/3`.

    Parameters
    ----------
    z : complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the honeycomb lattice.
        The `half_bandwidth` corresponds to the nearest neighbor hopping
        :math:`t=2D/3`.

    Returns
    -------
    complex np.ndarray or complex
        Value of the honeycomb lattice Green's function.

    See Also
    --------
    gftool.lattice.triangular.gf_z

    References
    ----------
    .. [horiguchi1972] Horiguchi, T., 1972. Lattice Green’s Functions for the
       Triangular and Honeycomb Lattices. Journal of Mathematical Physics 13,
       1411–1419. https://doi.org/10.1063/1.1666155

    Examples
    --------
    >>> ww = np.linspace(-1.5, 1.5, num=501, dtype=complex) + 1e-64j
    >>> gf_ww = gf_z_honeycomb(ww, half_bandwidth=1)

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.axhline(0, color='black', linewidth=0.8)
    >>> _ = plt.plot(ww.real, gf_ww.real, label=r"$\Re G$")
    >>> _ = plt.plot(ww.real, gf_ww.imag, '--', label=r"$\Im G$")
    >>> _ = plt.ylabel(r"$G*D$")
    >>> _ = plt.xlabel(r"$\omega/D$")
    >>> _ = plt.xlim(left=ww.real.min(), right=ww.real.max())
    >>> _ = plt.legend()
    >>> plt.show()
    """
    D = half_bandwidth / 1.5
    z_rel = z / D
    return 2 / D * z_rel * gf_z_triangular(2 * z_rel**2 - 1.5, half_bandwidth=9 / 4)


GF_FUNCS = {
    "bethe": gf_z_bethe,
    "onedim": gf_z_onedim,
    "square": gf_z_square,
    "box": gf_z_box,
    "triangular": gf_z_triangular,
    "honeycomb": gf_z_honeycomb,
}


# noinspection PyPep8Naming
class PartialFunction(Base):
    """Partial function implementation of TRIQS Gf Function descriptor."""

    def __init__(self, function: Func, *args, **kwargs):
        if not callable(function):
            raise RuntimeError("function must be callable!")
        super().__init__(function=function, args=args, kwargs=kwargs)

    # noinspection PyUnresolvedReferences
    def __call__(self, G: Gf) -> Gf:
        """Evaluate the function on the mesh and store the result in the Gf."""
        return self.function(G, *self.args, **self.kwargs)


# noinspection PyPep8Naming
class Partial(ABC):
    """Partial function hanlder for TRIQS Gf Function descriptors.

    This class is used to create partial functions that can be applied to Gfs.
    It acts like a proxy for the actual function in order to be able to pass arguments
    to `triqs.gf.descriptor_base.Function` objects.
    """

    @abstractmethod
    def function(self, G: Gf, **kwargs) -> Gf:
        """Function definition. Must be implemented by the user."""
        pass

    def __call__(self, **kwargs) -> PartialFunction:
        """Return a partial function that can be applied to Gfs."""
        return PartialFunction(self.function, **kwargs)


# noinspection PyPep8Naming
class Ht(Partial, ABC):
    """Base class for partial lattice Hilbert transforms using TRIQS Functions.

    Supports an on-site energy, chemical potewntial and a (diagonal) self energy.

    Parameters
    ----------
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    mu : float, optional
        Baseline chemical potential, by default 0.0. Is added to the value passed to the
        transform call
    """

    def __init__(self, eps: Onsite = 0.0, mu: float = 0.0):
        self.eps = eps
        self.mu = mu

    @abstractmethod
    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        """Transform the Green's function G.

        Has to be implemented by the deriving class.

        Parameters
        ----------
        G : Gf
            The Green's function to transform.
        Sigma : Gf
            The diagonal self energy to use in the transformation.
        eps : float, optional
            The on-site energy. Shifts the Green's function.
        mu : float, optional
            Chemical potential. Sets the Fermi level.
        eta : float, optional
            Imaginary broadening.

        Returns
        -------
        Gf
            The transformed Green's function.
        """
        pass

    def function(
        self, G: Gf, Sigma: Gf = None, eps: Onsite = None, mu: float = None, eta: float = 0.0
    ) -> Gf:
        """Function definition for PartialFunction, see `transform`."""
        if Sigma is None:
            Sigma = G.copy()
            Sigma.zero()
        eps = self.eps if eps is None else eps
        mu = self.mu + mu if mu is not None else self.mu
        return self.transform(G, Sigma, eps, mu, eta)

    def __call__(
        self, Sigma: Gf = None, eps: Onsite = None, mu: float = None, eta: float = 0.0
    ) -> PartialFunction:
        """Return a partial function of `tansform` that can be applied to Gfs.

        Sigma : Gf, optional
            Diagonal self energy to use in the transformation. By default, no
            self energy is used in the transformation.
        eps : float, optional
            On-site energy. By default, the default on-site energy `self.eps` is used.
        mu : float, optional
            Additional chemical potential. If passed, the value is added to the
            base-line chemical potential `self.eps`.
        eta : float, optional
            Imaginary broadening, by default 0.
        """
        return super().__call__(Sigma=Sigma, eps=eps, mu=mu, eta=eta)


# noinspection PyPep8Naming
class SemiCircularHt(Ht):
    r"""Hilbert transform of a semicircular density of states with self energy, i.e.

     .. math::
        g(z - ε - Σ) = \int \frac{A(\omega)}{z - ε - Σ - \omega} d\omega

    where :math:`A(\omega) = \theta( D - |\omega|) 2 \sqrt{ D^2 - \omega^2}/(\pi D^2)`.

    (Only works in combination with frequency Green's functions.)

    Parameters
    ----------
    half_bandwidth : float, optional
        Half bandwidth of the lattice, by default 1.
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    mu : float, optional
        Baseline chemical potential, by default 0.0. Is added to the value passed to the
        transform call
    """

    def __init__(self, half_bandwidth: float = 1.0, eps: Onsite = 0.0, mu: float = 0.0):
        self.half_bandwidth = half_bandwidth
        super().__init__(eps=eps, mu=mu)

    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        ndim = 0 if len(G.target_shape) == 0 else G.target_shape[0]
        eye = complex(1, 0) if ndim == 0 else np.identity(ndim, np.complex128)
        x = np.array(list(G.mesh.values()))
        if ndim > 0:
            x = x[:, None, None]

        om = x + mu - eps - Sigma.data
        D = self.half_bandwidth
        D2 = D**2

        if isinstance(G.mesh, MeshImFreq):
            sign = np.copysign(1, om.imag)
            G.data[...] = (om - 1j * sign * np.sqrt(D2 - om**2)) / D2 * 2 * eye

        elif isinstance(G.mesh, MeshReFreq):
            band = (-D < om.real) & (om.real < D)
            f = 2 / D2
            sign = np.copysign(1, om[~band].real)
            G.data[band, ...] = f * (om[band] - 1j * np.sqrt(D2 - om[band] ** 2))
            G.data[~band, ...] = f * (om[~band] - sign * np.sqrt(om[~band] ** 2 - D2))

        else:
            raise TypeError("This HilbertTransform is only correct in frequency")

        return G


# noinspection PyPep8Naming
class HilbertTransform(Ht):
    """General Hilbert transform Function descriptor using exact lattice Green's functions.

    Parameters
    ----------
    name : {bethe, onedim, square, box, triangular, honeycomb} str
        Name of the lattice Green's function to use.
    half_bandwidth : float, optional
        Half bandwidth of the lattice, by default 2.
    mu : float, optional
        Baseline chemical potential, by default 0.0.
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    """

    def __init__(self, name: str, half_bandwidth: float = 2.0, mu: float = 0.0, eps: Onsite = 0.0):
        try:
            self.func = GF_FUNCS[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown lattice Green's function: {name}")
        self.half_bandwidth = half_bandwidth
        super().__init__(eps, mu)

    @property
    def partial_func(self) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
        return partial(self.func, half_bandwidth=self.half_bandwidth)

    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        if isinstance(eps, Gf):
            eps = eps.data
        ndim = 0 if len(G.target_shape) == 0 else G.target_shape[0]
        eye = complex(1, 0) if ndim == 0 else np.identity(ndim, np.complex128)
        x = np.array(list(G.mesh.values())) + 1j * eta
        if ndim > 0:
            x = x[:, None, None]
        om = x + mu - eps - Sigma.data

        if isinstance(G.mesh, MeshImFreq):
            data = self.func(om, half_bandwidth=self.half_bandwidth)
            G.data[...] = data * eye
        elif isinstance(G.mesh, MeshReFreq):
            data = self.func(om, half_bandwidth=self.half_bandwidth)
            if eta == 0.0:
                data.imag *= np.copysign(1, om.real)
            G.data[...] = data
        else:
            raise TypeError("This HilbertTransform is only correct in frequency")

        return G


# noinspection PyPep8Naming
class HilbertTransformSumK(Ht):
    """General Hilbert transform Function descriptor using the TRIQS `SumkDiscreteFromLattice`.

    Parameters
    ----------
    tb : TBLattice
        The TRIQS `TBLattice` object to use for the lattice Green's function in the
        `SumkDiscreteFromLattice`.
    nk : int, optional
        Number of k-points to use in the BZ sampling, by default 64.
    mu : float, optional
        Baseline chemical potential, by default 0.0.
    eps : float, optional
        Default on-site energy. Can be overwritten in the transform call.
    """

    def __init__(self, tb: TBLattice, nk: int = 64, mu: float = 0.0, eps: Onsite = 0.0):
        self.sumk = SumkDiscreteFromLattice(lattice=tb, n_points=nk)
        super().__init__(eps, mu)

    def transform(self, G: Gf, Sigma: Gf, eps: Onsite, mu: float, eta: float) -> Gf:
        if isinstance(eps, Gf):
            eps = eps.data

        G << 0.0  # noqa
        mpi.barrier()

        n_orb = G.target_shape[0]
        z_mat = np.array([z.value * np.eye(n_orb) for z in G.mesh]) + 1j * eta
        mu_mat = mu * np.eye(n_orb)
        # Loop on k points...
        for wk, eps_k in zip(
            *[mpi.slice_array(A) for A in [self.sumk.bz_weights, self.sumk.hopping]]
        ):
            # these are now all python numpy arrays, here you can replace by any array like object
            # shape [nw, norb,norb]
            # numpy vectorizes the inversion automatically of shape [nw,orb,orb] over nw
            # speed up factor 30, comparable to C++!
            G.data[:, :, :] += wk * np.linalg.inv(
                z_mat[:] + mu_mat - eps_k - eps - Sigma.data[:, :, :]
            )

        G << mpi.all_reduce(G, comm=mpi.world, op=lambda x, y: x + y)
        mpi.barrier()

        return G
