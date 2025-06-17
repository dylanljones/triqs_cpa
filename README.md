[![build](https://github.com/dylanljones/triqs_cpa/workflows/build/badge.svg)](https://github.com/TRIQS/triqs_cpa/actions?query=workflow%3Abuild)

# triqs-cpa

> ⚠️ **This package is currently in development!.**

The triqs_cpa package includes the follwoing algorithms:

- VCA: Virtual Crystal Approximation
- ATA: Average T-matrix Approximation
- CPA: Coherent Potential Approximation

## Installation

### Installing via pip

> Coming soon!

The package is available on PyPI and can be installed with pip:

```bash
pip install triqs_cpa
```

### Installing from source

The installation instructions for this package are the same as for any TRIQS application:

1. Clone the latest stable version of the ``triqs_cpa`` repository:
   ```bash
   git clone git@github.com:dylanljones/triqs_cpa.git triqs_cpa.src
   ```

2. Create and move to a new directory where you will compile the code:
   ```bash
   mkdir triqs_cpa.build && cd triqs_cpa.build
   ```

3. Ensure that your shell contains the TRIQS environment variables by sourcing the ``triqsvars.sh`` file from your TRIQS installation:
   ```bash
   source path_to_triqs/share/triqs/triqsvars.sh
   ```
   If you are using TRIQS from Anaconda, you can use the ``CONDA_PREFIX`` environment variable:
   ```bash
   source $CONDA_PREFIX/share/triqs/triqsvars.sh
   ```

4. In the build directory call cmake, including any additional custom CMake options, see below:
   ```bash
   cmake ../triqs_cpa.src
   ```

5. Finally, compile the code and install the application:
   ```bash
   make install
   ```


## Examples

Simple example of a two-component semi-circular DOS:

```python
from triqs.gf import Gf, MeshReFreq
from triqs.plot.mpl_interface import oplot, plt

from triqs_cpa import SemiCircularHt, G_component, G_coherent, solve_cpa

# Parameters
mesh = MeshReFreq(-2, +2, 2001)        # Frequency mesh
eta = 1e-2                             # Broadening for the Green's function
conc = [0.2, 0.8]                      # Concentrations of the two components
eps = [-0.4, +0.4]                     # On-size energies of the two components
ht = SemiCircularHt(half_bandwidth=1)  # Semi-circular Hilbert transform

# Set up Gf and self energy
gf = Gf(mesh=mesh, target_shape=[1, 1])
sigma = gf.copy()
# Solve the CPA equations
solve_cpa(ht, sigma, conc, eps, eta=eta)

# Compute the coherent and component Green's functions
g_coh = G_coherent(ht, sigma, eta=eta)
g_cmpt = G_component(ht, sigma, conc, eps, eta=eta, scale=True)

# Plot the results
oplot(-g_coh.imag, color="k", label="$G$")
oplot(-g_cmpt["A"].imag, label="$G_A$")
oplot(-g_cmpt["B"].imag, label="$G_B$")
plt.show()

```

BlockGf example:
```python
from triqs.gf import MeshReFreq
from triqs.plot.mpl_interface import oplot, plt

from triqs_cpa import SemiCircularHt, G_coherent, solve_cpa, blockgf
# Parameters
mesh = MeshReFreq(-2, +2, 2001)        # Frequency mesh
gf_struct = [("up", 1), ("dn", 1)]     # Structure of the Green's function
eta = 1e-2                             # Broadening for the Green's function
conc = [0.2, 0.8]                      # Concentrations of the two components
eps = [-0.4, +0.4]                     # On-size energies of the two components
ht = SemiCircularHt(half_bandwidth=1)  # Semi-circular Hilbert transform

# Set up Gf and self energy
gf = blockgf(mesh, gf_struct=gf_struct)
sigma = gf.copy()
# Solve the CPA equations
solve_cpa(ht, sigma, conc, eps, eta=eta)

# Compute the coherent Green's functions
g_coh = G_coherent(ht, sigma, eta=eta)

# Plot the results
oplot(-g_coh["up"].imag)
oplot(g_coh["dn"].imag)
plt.show()
```

k-summation example:
```python
from triqs.gf import MeshReFreq
from triqs.lattice.tight_binding import TBLattice
from triqs.sumk import SumkDiscreteFromLattice
from triqs.utility import mpi
from triqs.plot.mpl_interface import oplot, plt

from triqs_cpa import G_coherent, solve_cpa, blockgf

# Parameters
mesh = MeshReFreq(-3, +3, 2001)  # Frequency mesh
gf_struct = [("up", 1), ("dn", 1)]    # Structure of the Green's function
eta = 1e-2  # Broadening for the Green's function
conc = [0.2, 0.8]  # Concentrations of the two components
eps = [-0.4, +0.4]  # On-size energies of the two components
t = 0.5  # Hopping parameter
tb = TBLattice(
    units=[
        (1, 0, 0),  # basis vector in the x-direction
        (0, 1, 0),  # basis vector in the y-direction
    ],
    hoppings={
        (+1, 0): [[-t]],  # hopping in the +x direction
        (-1, 0): [[-t]],  # hopping in the -x direction
        (0, +1): [[-t]],  # hopping in the +y direction
        (0, -1): [[-t]],  # hopping in the -y direction
    })
sk = SumkDiscreteFromLattice(lattice=tb, n_points=256)

# Set up Gf and self energy
gf = blockgf(mesh, gf_struct=gf_struct)
sigma = gf.copy()
# Solve the CPA equations
solve_cpa(sk, sigma, conc, eps, eta=eta, verbosity=2)

# Compute the coherent Green's functions
g_coh = G_coherent(sk, sigma, eta=eta)

# Plot the results
if mpi.is_master_node():
    oplot(-g_coh["up"].imag)
    oplot(g_coh["dn"].imag)
    plt.show()
```
