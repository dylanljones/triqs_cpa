# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-06-13

import string
from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from triqs.gf import BlockGf, Gf
from triqs.sumk import SumkDiscreteFromLattice
from triqs.utility import mpi

from .hilbert import Ht
from .utility import GfLike, blockgf, toarray

Onsite = Union[Sequence[Union[float, Sequence[float]]], BlockGf]


def generate_cmpt_names(n_cmpt: int) -> List[str]:
    """Generate component names based on the number of components.

    The names are generated as uppercase letters starting from 'A'.

    Parameters
    ----------
    n_cmpt : int
        The number of components for which to generate names.

    Returns
    -------
    List[str]
        A list of component names, e.g., ['A', 'B', 'C', ...] for n_cmpt = 3.

    Examples
    --------
    >>> generate_cmpt_names(3)
    ['A', 'B', 'C']
    """
    return list(string.ascii_uppercase[:n_cmpt])


def initialize_G_cmpt(
    sigma_cpa: GfLike, cmpts: Union[int, Sequence[str]], name: str = "G_cmpt"
) -> BlockGf:
    """Initialize a BlockGf for components based on the given self energy Σ.

    Parameters
    ----------
    sigma_cpa : GfLike
        The self energy from which the mesh and structure are taken. Can be a Gf or BlockGf.
    cmpts : int or Sequence[str]
        The number of components or a list of component names. If an integer is given, it will
        generate names like ['A', 'B', 'C', ...].
    name : str, optional
        The name of the resulting `BlockGf`. Default is "G_cmpt".

    Returns
    -------
    BlockGf
        A `BlockGf` object representing the component Green's functions G_cmpt.
    """
    if isinstance(cmpts, int):
        cmpts = generate_cmpt_names(cmpts)
    return blockgf(sigma_cpa.mesh, names=list(cmpts), blocks=sigma_cpa, name=name, copy=True)


def initalize_onsite_energy(
    sigma_cpa: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    cmpt_names: Sequence[str] = None,
    name: str = "eps",
) -> BlockGf:
    """Initialize the onsite energy for the component Green's functions G_cmpt.

    Parameters
    ----------
    sigma_cpa : GfLike
        The self energy from which the mesh and structure are taken. Can be a Gf or BlockGf.
    conc : Sequence[float]
        The concentration of each component as a sequence of floats. The length of this sequence
        determines the number of components. The concentration values should be positive and
        add up to 1.
    eps : Onsite
        The onsite energy for each component. This can be a single float, a sequence of floats
        corresponding to the components or a `BlockGf` with indices matching the components used
        for frequency dependent onsite energies. If a single float is given, it is used for all
        components.
    cmpt_names : Sequence[str], optional
        The names of the components. If not provided, they will be generated based on the number
        of components.
    name : str, optional
        The name of the resulting `BlockGf`. Default is "eps".
    """
    if isinstance(eps, BlockGf):
        return eps

    # Remove values with c=0
    conc, eps = zip(*[(c, e) for c, e in zip(conc, eps) if c > 0])
    # Broadcast shapes
    is_block = isinstance(sigma_cpa, BlockGf)
    n_blocks = len(sigma_cpa) if is_block else 0
    n_cmpt = len(conc)
    eps = np.asarray(eps)
    if is_block and len(eps.shape) == 1:
        eps = np.array([eps.copy() for _ in range(n_blocks)]).swapaxes(0, 1)

    names = generate_cmpt_names(n_cmpt) if cmpt_names is None else list(cmpt_names)
    blocks = [sigma_cpa] * n_cmpt
    eps_eff = blockgf(sigma_cpa.mesh, names=names, blocks=blocks, name=name, copy=True)
    for i, (name, eps_i) in enumerate(eps_eff):
        if is_block:
            ei = [eps[i]] * n_cmpt if not hasattr(eps[i], "__len__") else eps[i]
            for s, (spin, eps_is) in enumerate(eps_i):
                eps_is.data[:] = ei[s]
        else:
            eps_i << eps[i]

    return eps_eff


def _validate(
    sigma: GfLike, conc: Sequence[float], eps: Onsite
) -> Tuple[bool, np.ndarray, Union[BlockGf, np.ndarray]]:
    """Validate the inputs for component Green's functions and onsite energies.

    Parameters
    ----------
    sigma : GfLike
        The self energy from which the mesh and structure are taken. Can be a Gf or BlockGf.
    conc : Sequence[float]
        The concentration of each component as a sequence of floats. The length of this sequence
        determines the number of components. The concentration values should be positive and
        add up to 1.
    eps : Onsite
        The onsite energy for each component. This can be a single float, a sequence of floats
        corresponding to the components or a `BlockGf` with indices matching the components used
        for frequency dependent onsite energies. If a single float is given, it is used for all
        components.

    Returns
    -------
    is_block : bool
        True if `sigma` is a BlockGf, False otherwise.
    conc : np.ndarray
        A numpy array of concentrations after removing components with zero concentration.
    eps : Union[BlockGf, np.ndarray]
        A numpy array or BlockGf of onsite energies after removing components with zero
        concentration.
    """
    if isinstance(eps, BlockGf):
        # Convert back to numpy array if eps is a Block Gf
        # ToDo: Implement CPA methods to accept BlockGf
        eps = toarray(eps)

    # Remove values with c=0
    conc, eps = zip(*[(c, e) for c, e in zip(conc, eps) if c > 0])
    # Broadcast shapes
    is_block = isinstance(sigma, BlockGf)
    n_blocks = len(sigma) if is_block else 0
    n_cmpt = len(conc)
    conc = np.asarray(conc)
    eps = np.asarray(eps)
    if is_block and len(eps.shape) == 1:
        eps = np.array([eps.copy() for _ in range(n_blocks)]).swapaxes(0, 1)
    # Check if arguments are valid
    if sum(conc) != 1.0:
        raise ValueError(f"Sum of concentrations {list(conc)} does not add up to 1!")
    if eps.shape[0] != n_cmpt:
        raise ValueError(f"Shape mismatch of eps {eps.shape} and number of components {n_cmpt}!")
    if is_block and eps.shape[1] != n_blocks:
        raise ValueError(f"Shape mismatch of eps {eps.shape} and number of blocks {n_blocks}!")
    return is_block, conc, eps


def sumk(
    sk: SumkDiscreteFromLattice, sigma: Gf, eps: Onsite = 0.0, mu: float = 0.0, eta: float = 0.0
) -> Gf:
    """Calc Gloc with mpi parallelism."""
    gloc = sigma.copy()

    if isinstance(eps, Gf):
        eps = eps.data

    gloc << 0.0  # noqa
    mpi.barrier()

    n_orb = gloc.target_shape[0]
    z_mat = np.array([z.value * np.eye(n_orb) for z in gloc.mesh]) + 1j * eta
    mu_mat = mu * np.eye(n_orb)

    # Loop on k points...
    for wk, eps_k in zip(*[mpi.slice_array(A) for A in [sk.bz_weights, sk.hopping]]):
        # numpy vectorizes the inversion automatically of shape [nw,orb,orb] over nw
        # speed up factor 30, comparable to C++!
        gloc.data[:] += wk * np.linalg.inv(z_mat[:] + mu_mat - eps_k - eps - sigma.data[:])

    gloc << mpi.all_reduce(gloc, comm=mpi.world, op=lambda x, y: x + y)
    mpi.barrier()

    return gloc


def G_coherent(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    mu: float = 0.0,
    eta: float = 0.0,
    name: str = "G_coh",
) -> GfLike:
    """Compute the coherent (total) Green's functions `G_h(z)`.

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        The SSA self-energy as TRIQS gf.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    name : str, optional
        The name of the resulting Gf object.

    Returns
    -------
    Gf or BlockGf
        The coherent Green's function embedded in `sigma`.
    """
    gc = sigma.copy()
    gc.name = name

    if isinstance(sigma, Gf):
        if isinstance(ht, Ht):
            gc << ht(sigma, mu=mu, eta=eta)
        else:
            gc << sumk(ht, sigma, mu=mu, eta=eta)

    elif isinstance(sigma, BlockGf):
        if isinstance(ht, Ht):
            for name, g in gc:
                g << ht(sigma[name], mu=mu, eta=eta)
        else:
            for name, g in gc:
                g << sumk(ht, sigma[name], mu=mu, eta=eta)
    else:
        raise ValueError(f"Invalid type of `sigma`: {type(sigma)}.")
    return gc


def G_component(
    ht: Union[Ht, SumkDiscreteFromLattice],
    sigma: GfLike,
    conc: Sequence[float],
    eps: Onsite,
    mu: float = 0.0,
    eta: float = 0.0,
    scale: bool = False,
    cmpt_names: list = None,
    name: str = "G_cmpt",
) -> BlockGf:
    """Compute the component Green's functions `G_i(z)`.

    .. math::
        G_i(z) = c_i G_i(z) / (1 - (ε_i - Σ(z)) G_0(z - Σ(z)) )

    Parameters
    ----------
    ht : Ht or SumkDiscreteFromLattice
        Lattice Hilbert transformation or discrete k-sum used to calculate the coherent Green's
        function.
    sigma : Gf or BlockGf
        The SSA self-energy as TRIQS gf.
    conc : (..., N_cmpt) float array_like, optional
        Concentration of the different components used for the average.
        If not provided, the component GFs are returned unweighted.
    eps : (N_cmpt, [N_spin], ...) array_like
        On-site energy of the components. This can also include a local frequency
        dependent self-energy of the component sites.
    mu : float, optional
        The chemical potential, defaults to 0.0.
    eta : float, optional
        Complex broadening, should be only used for real frequency Greens functions.
    scale : bool, optional
        If True, the component Gfs are multiplied by the concentrations. In this case
        the sum of the component Gfs is equal to the coherent Green's function.
        The default is `False`.
    cmpt_names : list, optional
        List of names for the components. If not provided, the names are upper case
        ASCII characters `A`, `B`, `C`, ...
    name : str, optional
        The name of the resulting Gf object.

    Returns
    -------
    BlockGf
        The Green's function of the components embedded in `sigma_cpa`.
    """
    is_block, conc, eps = _validate(sigma, conc, eps)

    n_cmpt = len(conc)
    if cmpt_names is None:
        cmpt_names = list(eps.indices) if isinstance(eps, BlockGf) else generate_cmpt_names(n_cmpt)

    g_cmpt = blockgf(sigma.mesh, cmpt_names, blocks=[sigma] * n_cmpt, name=name, copy=True)
    cc = conc if scale else np.ones_like(conc)

    def _g_cmpt(_sig: Gf, _eps_i: ArrayLike, _conc_i: ArrayLike) -> Gf:
        """Compute the component Green's functions for a single Gf."""
        _g = _sig.copy()
        if isinstance(ht, Ht):
            _g << ht(_sig, mu=mu, eta=eta)
        else:
            _g << sumk(ht, _sig, mu=mu, eta=eta)
        _g.data[:] = _conc_i * _g.data / (1 - _g.data * (_eps_i - _sig.data))
        return _g

    if isinstance(sigma, Gf):
        for i, (name, g_i) in enumerate(g_cmpt):
            g_i << _g_cmpt(sigma, eps[i], cc[i])

    elif isinstance(sigma, BlockGf):
        for i, (name, g_i) in enumerate(g_cmpt):
            for s, (spin, g_s) in enumerate(g_i):
                g_s << _g_cmpt(sigma[spin], eps[i, s], cc[i])

    else:
        raise ValueError(f"Invalid type of `sigma`: {type(sigma)}.")

    return g_cmpt
