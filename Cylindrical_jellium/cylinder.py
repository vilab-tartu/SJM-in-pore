import numpy as np
from math import pi
from ase.io import read, write
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW, PoissonSolver, Mixer
from gpaw.jellium import Jellium
import gpaw.jellium
import matplotlib.pyplot as plt

class JelliumCylinder(Jellium):
    """Jellium cylinder infinite along z-axis."""

    def __init__(self, charge, radius):
        """Initialize the infinite Jellium cylinder.

        Parameters:
        - charge: Total background charge per unit cell.
        - radius: Radius of the cylinder in Angstroms.
        """
        super().__init__(charge)
        self.x0 = None
        self.y0 = None
        self.radius = radius / Bohr  # Convert radius to Bohr units

    def todict(self):
        dct = super().todict()
        dct.update(radius=self.radius * Bohr)
        return dct

    def set_grid_descriptor(self, gd):
        """Set the grid descriptor and initialize parameters."""
        # Set x0 and y0 to the center of the cell in Bohr units
        self.x0 = (gd.cell_cv[0, 0] / 2.0)
        self.y0 = (gd.cell_cv[1, 1] / 2.0)
        super().set_grid_descriptor(gd)

    def get_mask(self):
        """Create a mask for the infinite cylindrical region along z."""
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        x = r_gv[..., 0]
        y = r_gv[..., 1]

        # Compute squared distance from the center in the xy-plane
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2

        # Mask for points inside the infinite cylinder
        inside_cylinder = (r2 <= self.radius ** 2)
        return inside_cylinder.astype(float)

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
    convergence={'energy': 0.0005,
                 'density': 1.0e-3,
                 'eigenstates': 1.0e-6,
                 'bands': 'occupied'},
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
#calc.write('cylinder.gpw', mode='all')

# Get the electron density
density = calc.get_pseudo_density()

# Get grid and cell information
cell = calc.atoms.get_cell()
nx, ny, nz = density.shape
x = np.linspace(0, cell[0, 0], nx, endpoint=False)
y = np.linspace(0, cell[1, 1], ny, endpoint=False)
z = np.linspace(0, cell[2, 2], nz, endpoint=False)

# Choose a z-index to slice
z_index = nz // 2  # Middle of the cell

# Extract the 2D density slice at the chosen z-index
density_xy = density[:, :, z_index]

# Create meshgrid for x and y
X, Y = np.meshgrid(x, y, indexing='ij')

# Plot the 2D electron density contour map
plt.figure(figsize=(8.25/2.54, 8.25/2.54))
contour = plt.contourf(X, Y, density_xy.T, levels=100, cmap='viridis')
plt.colorbar(contour, label='Electron Density (e/Å³)')
plt.xlabel('x (Å)')
plt.ylabel('y (Å)')
plt.title(f'Electron Density Contour Map at z = {z[z_index]:.2f} Å')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('density_map.png')

# Proceed to get the electron density
density = calc.get_pseudo_density()

# Get grid and cell information
cell = calc.atoms.get_cell()
nx, ny, nz = density.shape
x = np.linspace(0, cell[0, 0], nx, endpoint=False)
y = np.linspace(0, cell[1, 1], ny, endpoint=False)
z = np.linspace(0, cell[2, 2], nz, endpoint=False)

# Calculate average density profiles along x, y, and z
density_x = density.mean(axis=(1, 2))  # Average over y and z
density_y = density.mean(axis=(0, 2))  # Average over x and z
density_z = density.mean(axis=(0, 1))  # Average over x and y

# Plot density profile along x
plt.figure(figsize=(8.25/2.54, 8.25/2.54))
plt.plot(x, density_x)
plt.xlabel('x (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along x')
plt.grid(True)
plt.savefig('density_profile_x.png')

# Plot density profile along y
plt.figure(figsize=(8.25/2.54, 8.25/2.54))
plt.plot(y, density_y)
plt.xlabel('y (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along y')
plt.grid(True)
plt.savefig('density_profile_y.png')

# Plot density profile along z
plt.figure(figsize=(8.25/2.54, 8.25/2.54))
plt.plot(z, density_z)
plt.xlabel('z (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along z')
plt.grid(True)
plt.savefig('density_profile_z.png')
