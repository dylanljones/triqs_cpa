[![build](https://github.com/TRIQS/triqs_cpa/workflows/build/badge.svg)](https://github.com/TRIQS/triqs_cpa/actions?query=workflow%3Abuild)

# triqs_cpa

> Single site disorder solvers for TRIQS

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

Initial Setup
-------------

To adapt this skeleton for a new TRIQS application, the following steps are necessary:

* Create a repository, e.g. https://github.com/username/appname

* Run the following commands in order after replacing **appname** accordingly

```bash
git clone https://github.com/triqs/triqs_cpa --branch python_only appname
cd appname
./share/squash_history.sh
./share/replace_and_rename.py appname
git add -A && git commit -m "Adjust triqs_cpa skeleton for appname"
```

You can now add your github repository and push to it

```bash
git remote add origin https://github.com/username/appname
git remote update
git push origin unstable
```

If you prefer to use the [SSH interface](https://help.github.com/en/articles/connecting-to-github-with-ssh)
to the remote repository, replace the http link with e.g. `git@github.com:username/appname`.

### Merging triqs_cpa skeleton updates ###

You can merge future changes to the triqs_cpa skeleton into your project with the following commands

```bash
git remote update
git merge triqs_cpa_remote/python_only -X ours -m "Merge latest triqs_cpa skeleton changes"
```

If you should encounter any conflicts resolve them and `git commit`.
Finally we repeat the replace and rename command from the initial setup.

```bash
./share/replace_and_rename.py appname
git commit --amend
```

Now you can compare against the previous commit with:
```bash
git diff prev_git_hash
````

Getting Started
---------------

After setting up your application as described above you should customize the following files and directories
according to your needs (replace triqs_cpa in the following by the name of your application)

* Adjust or remove the `README.md` and `doc/ChangeLog.md` file
* In the `python/triqs_cpa` subdirectory add your Python source files.
* In the `test/python` subdirectory adjust the example test `Basic.py` or add your own tests.
* Adjust any documentation examples given as `*.rst` files in the doc directory.
* Adjust the sphinx configuration in `doc/conf.py.in` as necessary.
* The build and install process is identical to the one outline [here](https://triqs.github.io/triqs_cpa/unstable/install.html).

### Optional ###
----------------

* Add your email address to the bottom section of `Jenkinsfile` for Jenkins CI notification emails
```
End of build log:
\${BUILD_LOG,maxLines=60}
    """,
    to: 'user@domain.org',
```
