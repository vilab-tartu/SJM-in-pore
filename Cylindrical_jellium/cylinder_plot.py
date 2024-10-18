# plot.py

import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
import gpaw.jellium
from my_jellium import JelliumCylinder

# Override the create_background_charge function
def create_background_charge(**kwargs):
    if 'radius' in kwargs:
        return JelliumCylinder(**kwargs)
    else:
        return gpaw.jellium.Jellium(**kwargs)

gpaw.jellium.create_background_charge = create_background_charge

# Load the GPAW calculator
calc = GPAW('cylinder.gpw', txt=None)

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
plt.figure(figsize=(8, 6))
plt.plot(x, density_x)
plt.xlabel('x (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along x')
plt.grid(True)
plt.savefig('density_profile_x.png')

# Plot density profile along y
plt.figure(figsize=(8, 6))
plt.plot(y, density_y)
plt.xlabel('y (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along y')
plt.grid(True)
plt.savefig('density_profile_y.png')

# Plot density profile along z
plt.figure(figsize=(8, 6))
plt.plot(z, density_z)
plt.xlabel('z (Å)')
plt.ylabel('Electron Density (e/Å³)')
plt.title('Electron Density Profile along z')
plt.grid(True)
plt.savefig('density_profile_z.png')

plt.show()
