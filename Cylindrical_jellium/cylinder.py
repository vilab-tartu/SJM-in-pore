# test.py

import numpy as np
from math import pi
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW, PoissonSolver, Mixer
from JelliumCylinder import JelliumCylinder

# Parameters
rs = 2.07 * Bohr         # Wigner-Seitz radius in Bohr
h = 0.2                  # Grid spacing in Angstroms
a = 10.0                 # Cell size in x and y (Angstroms)
L = 40.0                 # Cell size in z (Angstroms)
k = 2                    # Number of k-points in x, y, and z

# Cylinder properties
radius = 5.0             # Radius of the cylinder in Angstroms

# Electron density (electrons per volume)
n_electron_density = 3 / (4 * np.pi * (rs / Bohr)**3)  # Valence 3 for a metal like Al

# Volume of the cylinder within the simulation cell
volume_cylinder = pi * radius**2 * L

# Total number of electrons (charge)
ne = n_electron_density * volume_cylinder

# Create the jellium cylinder
jellium = JelliumCylinder(
    charge=ne,           # Total charge per unit cell
    radius=radius        # Radius in Angstroms
)

# Create an empty Atoms object with periodic boundary conditions
atoms = Atoms(pbc=(True, True, True),
              cell=(a, a, L))

# Set up the Poisson solver without the 'eps' parameter
poissonsolver = PoissonSolver()

# Set up the GPAW calculator
calc = GPAW(
    mode='fd',                # Finite difference mode
    background_charge=jellium,
    xc='LDA',                 # Exchange-correlation functional
    eigensolver='dav',
    kpts=(k, k, k),           # k-points in x, y, and z
    h=h,                      # Grid spacing in Angstroms
    convergence={'density': 1e-4},
    mixer=Mixer(0.1, 5, weight=50.0),
    nbands=int(ne / 2) + 20,  # Number of bands
    poissonsolver=poissonsolver,
    txt='cylinder.txt'
)

# Attach the calculator to the atoms object
atoms.calc = calc

# Run the calculation
e = atoms.get_potential_energy()

# Save the calculation to a .gpw file
calc.write('cylinder.gpw', mode='all')
