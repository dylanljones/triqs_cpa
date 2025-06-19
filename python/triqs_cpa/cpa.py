# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-06-13

from functools import partial
from typing import Sequence, Union

import numpy as np
from scipy import optimize
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshImTime, MeshReFreq, inverse
from triqs.sumk import SumkDiscreteFromLattice
from triqs.utility import mpi

from .gf import G_coherent, GfLike, Onsite, _validate
from .hilbert import Ht
from .utility import fill_gf, toarray

__all__ = ["solve_vca", "solve_ata", "solve_cpa"]


def _apply_mixing(old: GfLike, new: GfLike, mixing: float = 1.0) -> GfLike:
    """Apply mixing to a Green's function object.

    Parameters
    ----------
    old : Gf or BlockGf
        The old value of the quantity.
    new : Gf or BlockGf
        The new value of the quantity. Will be overwriten with the result!
    mixing: float
        The mixing value. If `mixing=1` no mixing is applied.

    Returns
    -------
    Gf or BlockGf
        The mixed quantity. Same as `new` after calling the method.
    """
    if mixing == 1.0:
        return new

    if isinstance(old, Gf) and isinstance(new, Gf):
        new << new * mixing + old * (1 - mixing)
    elif isinstance(old, BlockGf) and isinstance(new, BlockGf):
        for name in old.indices:
            new[name] << new[name] * mixing + old[name] * (1 - mixing)
    else:
        raise ValueError("Both `new` and `old` must be either Gf or BlockGf objects.")
    return new


def _max_difference(
    old: GfLike, new: GfLike, norm_temp: bool = True, relative: bool = False
) -> float:
    """Calculate the maximum difference between two Green's functions using the Frobenius norm.

    The Frobenius norm is calculated over the grid points of the Green's function. The result is
    divided by `sqrt(beta)` for Matsubara frequencies, `sqrt(N)` for real frequencies and
    `sqrt(N/beta)` for imaginary time frequencies.

    Parameters
    ----------
    old : GfLike
        The old quantity (Gf or self energy).
    new : GfLike
        The new quantity (Gf or self energy).
    norm_temp : bool, optional
        If `True` the norm is divided by an additional `sqrt(beta)` to account for temperature
        scaling. . Only used if `relative=False`. The default is `True`.
    relative : bool, optional
        If `True` the error is normalized by the old value. The default is `False`.

    Returns
    -------
    float
        The maximum difference between the two Green's functions. If the inputs are BlockGfs the
        highest error of the blocks is returned.
    """
    if isinstance(old, BlockGf):
        return max([_max_difference(old[name], new[name], norm_temp, relative) for name, g in old])

    assert old.mesh == new.mesh, "Meshes of inputs do not match."
    assert old.target_shape == new.target_shape, "Target shapes of inputs do not match."

    if relative:
        # Use a simple relative error instead
        norm_grid = np.linalg.norm(old.data - new.data, axis=tuple(range(1, old.data.ndim)))
        norm_old = np.linalg.norm(old.data, axis=tuple(range(1, old.data.ndim)))
        return np.linalg.norm(norm_grid, axis=0) / np.linalg.norm(norm_old, axis=0)

    mesh = old.mesh
    n_points = len(mesh)

    # subtract the largest real value to make sure that G1-G2 falls off to 0
    if isinstance(mesh, MeshImFreq):
        offset = np.diag(np.diag(old.data[-1, :, :].real - new.data[-1, :, :].real))
    else:
        offset = 0.0

    # calculate norm over all axis but the first one which are frequencies
    norm_grid = np.linalg.norm(old.data - new.data - offset, axis=tuple(range(1, old.data.ndim)))

    # now calculate Frobenius norm over grid points
    if isinstance(mesh, MeshImFreq):
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(mesh.beta)
    elif isinstance(mesh, MeshImTime):
        norm = np.linalg.norm(norm_grid, axis=0) * np.sqrt(mesh.beta / n_points)
    elif isinstance(mesh, MeshReFreq):
        norm = np.linalg.norm(norm_grid, axis=0) / np.sqrt(n_points)
    else:
        raise NotImplementedError(f"Mesh type {type(mesh)} not supported.")

    if isinstance(mesh, (MeshImFreq, MeshImTime)) and norm_temp:
        norm = norm / np.sqrt(mesh.beta)

    return float(norm)


# -- VCA Solve methods -----------------------------------------------------------------------------


def solve_vca(
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    name: str = "Σ_vca",
) -> GfLike:
    """Solve the CPA equations using the VCA approximation.

    The virtual crystal approximation (VCA) is the simplest form of the CPA.
    The self-energy is given by the average of the site self-energies weighted by the concentration.

    Parameters
    ----------
    sigma : Gf or BlockGf
        Starting guess for VCA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the VCA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) array_like or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.

    Returns
    -------
    Gf or BlockGf
        The self-consistent VCA self energy `Σ_vca`. Same as the input self energy after
        calling the method.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)
    if is_block:
        for i, (name, sig) in enumerate(sigma):
            sig.data[:] = np.sum(eps[:, i] * conc)
    else:
        sigma.data[:] = np.sum(eps * conc)

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


# -- ATA solve methods -----------------------------------------------------------------------------


def solve_ata(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    mu: float = 0.0,
    eta: float = 0.0,
    name: str = "Σ_ata",
) -> GfLike:
    """Solve the CPA equations using the ATA approximation.

    The average T-matrix approximation (ATA) is a more advanced form of the CPA.

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        Starting guess for ATA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the ATA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) array_like or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.

    Returns
    -------
     Gf or BlockGf
        The self-consistent ATA self energy `Σ_cpa`. Same as thew input self energy after
        calling the method.

    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    sigma_vca = sigma.copy()
    sigma_vca.zero()

    g0 = sigma.copy()
    g0.zero()

    # Unperturbated Green's function (uses VCA)
    if is_block:
        for i, (name, sig) in enumerate(sigma_vca):
            sig.data[:] = np.sum(eps[:, i] * conc)
    else:
        sigma_vca.data[:] = np.sum(eps * conc)

    g0 << G_coherent(ht, sigma_vca, mu=mu, eta=eta)

    # Average T-matrix
    def _tavrg(_g0: Gf, _eps: np.ndarray, _sigma: Gf) -> Gf:
        _tmat = _g0.copy()
        _cmpts = [(_e - _sigma.data) / (1 - (_e - _sigma.data) * _g0.data) for _e in _eps]
        _tmat.data[:] = np.sum([_c * _g.data for _c, _g in zip(conc, _cmpts)], axis=0)
        return _tmat

    tavrg = sigma.copy()
    if is_block:
        for s, (name, t) in enumerate(tavrg):
            t << _tavrg(g0[name], eps[:, s], sigma_vca[name])
    else:
        tavrg << _tavrg(g0, eps, sigma_vca)

    sigma << tavrg * inverse(1 + g0 * tavrg) + sigma_vca

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


# -- CPA solve methods -----------------------------------------------------------------------------


def solve_iter(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    mu: float = 0.0,
    eta: float = 0.0,
    name: str = "Σ_cpa",
    tol: float = 1e-6,
    mixing: float = 0.5,
    maxiter: int = 1000,
    verbosity: int = 1,
) -> GfLike:
    """Determine the CPA self-energy by an iterative solution of the CPA equations.

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        Starting guess for CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) array_like or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.
    tol : float, optional
        The tolerance for the convergence of the CPA self-energy.
        The iteration stops when the norm between the old and new self-energy
        .math:`|Σ_new - Σ_old|` is smaller than `tol`.
    mixing : float, optional
        The mixing parameter for the self-energy update. The new self-energy is
        computed as `Σ_new = (1 - mixing) * Σ_old + mixing * Σ_new`.
        If `mixing=1` no mixing is applied. The default is `0.5`.
    maxiter : int, optional
        The maximum number of iterations, by default 1000.
    verbosity : {0, 1, 2} int, optional
        The verboisity level.

    Returns
    -------
     Gf or BlockGf
        The self-consistent CPA self energy `Σ_cpa`. Same as thew input self energy after
        calling the method.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    # Skip trivial solution
    if len(conc) == 1:
        if mpi.is_master_node():
            mpi.report("Single component, skipping CPA!")
        if is_block:
            for i, (name, sig) in enumerate(sigma):
                sig.data[:] = eps[0, i]
        else:
            sigma.data[:] = eps[0]
        return sigma

    if verbosity > 0:
        if mpi.is_master_node():
            mpi.report(f"Solving CPA problem for {len(conc)} components iteratively...")
    # Initial coherent Green's function
    gc = sigma.copy()
    gc.name = "Gc"
    gc.zero()

    # Old self energy for convergence check
    sigma_old = sigma.copy()

    # Define avrg G method
    def _g_avrg(_g0_inv: Gf, _eps: np.ndarray) -> Gf:
        """Compute component and average Green's function `<G>(z) = ∑ᵢ cᵢ Gᵢ(z)`."""
        _g_i = _g0_inv.copy()
        _cmpts = [1 / (_g0_inv.data - _e) for _e in _eps]
        _g_i.data[:] = np.sum([_g.data * _c for _c, _g in zip(conc, _cmpts)], axis=0)
        return _g_i

    # Begin CPA iterations
    for it in range(maxiter):
        # Compute average GF via the self-energy:
        # <G> = G_0(E - Σ) = 1 / (E - H_0 - Σ)
        gc << G_coherent(ht, sigma, mu=mu, eta=eta)
        mpi.barrier()
        # Compute non-interacting GF via Dyson equation
        g0_inv = sigma + inverse(gc)

        # g_i = [inverse(sigma - eps[i] + inverse(gc)) for i, c in enumerate(conc)]
        # Compute new coherent GF: <G> = c_A G_A + c_B G_B + ...
        if is_block:
            for s, (spin, g) in enumerate(gc):
                g << _g_avrg(g0_inv[spin], eps[:, s])
        else:
            gc << _g_avrg(g0_inv, eps)

        mpi.barrier()
        # Update self energy via Dyson: Σ = G_0^{-1} - <G>^{-1}
        sigma << g0_inv - inverse(gc)

        # Apply mixing
        _apply_mixing(sigma_old, sigma, mixing)

        # Check for convergence
        diff = _max_difference(sigma_old, sigma, norm_temp=True, relative=False)
        if verbosity > 1:
            if mpi.is_master_node():
                mpi.report(f"CPA iteration {it + 1}: Error={diff:.10f}")

        if diff <= tol:
            if verbosity > 0:
                if mpi.is_master_node():
                    mpi.report(f"CPA converged in {it + 1} iterations (Error: {diff:.10f})")
            break
        sigma_old = sigma.copy()
        mpi.barrier()
    else:
        if verbosity > 0:
            if mpi.is_master_node():
                mpi.report(f"CPA did not converge after {maxiter} iterations")
    if mpi.is_master_node():
        mpi.report("")

    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


def _sigma_root(
    x: np.ndarray,
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: np.ndarray,
    eps: np.ndarray,
    mu: float,
    eta: float = 0.0,
) -> np.ndarray:
    """Self-energy root equation r(Σ) for CPA.

    The root equation is given by:
    .. math::
        r(Σ, z) = T(z) / (1 + T(z) * H(z-Σ)),

    where .math`H` is the Hilbert transform `hilbert`.

    Parameters
    ----------
    x : np.ndarray
        The self-energy to evaluate the root equation at.
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        The CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
        Only used to determine the shape of the output!
    conc : (..., N_cmpt) float array_like
        Concentration of the different components used for the average.
    eps : (..., N_cmpt) float or complex np.ndarray
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.

    Returns
    -------
    root : (...) complex np.ndarray
        The result of r(Σ). If `r(Σ)=0`, `Σ` is the correct CPA self-energy.
    """
    sigma_x = sigma.copy()
    fill_gf(sigma_x, x)

    gf_0 = G_coherent(ht, sigma_x, mu=mu, eta=eta)  # Coherent Green's function
    gf_0_arr = toarray(gf_0)

    if isinstance(sigma, BlockGf):
        items = list()
        for i in range(len(conc)):
            eps_eff_i = eps[:, i] - x[i, ..., np.newaxis]
            items.append(eps_eff_i)
        eps_eff = np.array(items)
    else:
        eps_eff = eps - x[..., np.newaxis]

    ti = eps_eff / (1 - eps_eff * gf_0_arr[..., np.newaxis])  # T-matrix elements
    tmat = np.sum(conc * ti, axis=-1)  # Average T-matrix
    root = tmat / (1 + tmat * gf_0_arr)  # Self energy root
    return root


def _sigma_root_restricted(
    x: np.ndarray,
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: np.ndarray,
    eps: np.ndarray,
    mu: float,
    eta: float = 0.0,
) -> np.ndarray:
    """Restricted self-energy root equation r(Σ) for CPA, where `Im Σ > 0`.

    See Also
    --------
    _sigma_root: Self-energy root equation r(Σ) for CPA.
    """
    # Mask of unphysical roots Im(Σ) > 0
    sigma_arr = toarray(sigma)
    unphysical = sigma_arr.imag > 0
    if np.all(~unphysical):  # All sigmas valid
        return _sigma_root(x, ht, sigma, conc, eps, mu, eta)
    # Store offset to valid solution and remove invalid
    offset = sigma_arr.imag[unphysical].copy()
    sigma_arr.imag[unphysical] = 0
    # Compute root equation and enlarge residues
    root = np.asarray(_sigma_root(x, ht, sigma, conc, eps, mu, eta))
    root[unphysical] *= 1 + offset
    # Remove unphysical roots
    root.real[unphysical] += 1e-3 * offset * np.where(root.real[unphysical] >= 0, 1, -1)
    root.real[unphysical] += 1e-3 * offset * np.where(root.real[unphysical] >= 0, 1, -1)
    return root


def solve_cpa_root(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    mu: float = 0.0,
    eta: float = 0.0,
    name: str = "Σ_cpa",
    tol: float = 1e-6,
    maxiter: int = 1000,
    verbosity: int = 1,
    restricted: bool = False,
    **root_kwargs,
) -> GfLike:
    """Determine the CPA self-energy by solving the root problem.

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        Starting guess for CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) array_like or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.
    tol : float, optional
        The tolerance for the convergence of the CPA self-energy.
        The iteration stops when the norm between the old and new self-energy
        .math:`|Σ_new - Σ_old|` is smaller than `tol`.
    maxiter : int, optional
        The maximum number of iterations, by default 1000.
    restricted : bool, optional
        Whether the self-energy is restricted to physical values. (default: True)
        Note, that even if `restricted=True`, the imaginary part can get negative
        within tolerance. This should be removed by hand if necessary.
    verbosity : {0, 1, 2} int, optional
        The verboisity level.

    Returns
    -------
     Gf or BlockGf
        The self-consistent CPA self energy `Σ_cpa`. Same as thew input self energy after
        calling the method.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    # Skip trivial solution
    if len(conc) == 1:
        if mpi.is_master_node():
            mpi.report("Single component, skipping CPA!")
        if is_block:
            for i, (name, sig) in enumerate(sigma):
                sig.data[:] = eps[0, i]
        else:
            sigma.data[:] = eps[0]
        return sigma

    if verbosity > 0:
        if mpi.is_master_node():
            mpi.report(f"Solving CPA problem for {len(conc)} components via root equation...")
    # Initial coherent Green's function
    gc = sigma.copy()
    gc.name = "Gc"
    gc.zero()

    # Setup arguments
    func = _sigma_root_restricted if restricted else _sigma_root
    root_eq = partial(func, ht=ht, sigma=sigma, conc=conc, eps=eps, mu=mu, eta=eta)
    root_kwargs["method"] = "anderson" if restricted else "broyden2"
    root_kwargs["tol"] = tol
    root_kwargs["options"] = {
        "maxiter": maxiter,
    }

    # Optimize root
    sigma0 = toarray(sigma)
    sol = optimize.root(root_eq, x0=sigma0, **root_kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)
    if verbosity > 0:
        if mpi.is_master_node():
            niter = sol.nit if hasattr(sol, "nit") else sol.nfev
            diff = np.max(np.abs(sol.fun))
            mpi.report(
                f"CPA root solver converged in {niter} function evaluations (Error: {diff:.10f})"
            )
    if verbosity > 1:
        if mpi.is_master_node():
            mpi.report(sol)

    fill_gf(sigma, sol.x)
    sigma_out = sigma.copy()
    sigma_out.name = name
    return sigma_out


def solve_cpa(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    mu: float = 0.0,
    eta: float = 0.0,
    name: str = "Σ_cpa",
    method: str = "iter",
    **kwds,
) -> GfLike:
    """Determine the CPA self-energy of the CPA equations.

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        Starting guess for CPA self-energy. Can be a single or spin resolved Gf.
        The self energy will be overwritten with the result for the CPA self-energy.
    conc : (N_cmpt, ) float array_like
        Concentration of the different components used for the average.
    eps : (N_cmpt, [N_spin], ...) complex np.ndarray or BlockGf
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object returned as self-energy.
    method : {"iter", "root"} str, optional
        The method to use for solving the CPA root equation. Can be either 'iter' for
        the iterative algorythm or 'root' for the optimization algorythm.
    **kwds
        Additional keyword arguments passed to the specif solve method.

    Returns
    -------
     Gf or BlockGf
        The self-consistent CPA self energy `Σ_cpa`. Same as the input self energy after
        calling the method.
    """
    supported = {"iter": solve_iter, "root": solve_cpa_root}

    kwds.update(ht=ht, sigma=sigma, eps=eps, conc=conc, mu=mu, eta=eta, name=name)
    try:
        func = supported[method.lower()]
        return func(**kwds)
    except KeyError:
        raise ValueError(f"Invalid method: {method}. Use {list(supported.keys())}!")
