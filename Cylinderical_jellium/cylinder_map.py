# map.py

import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
import gpaw.jellium
from my_jellium import JelliumCylinder

# Override the create_background_charge function
def create_background_charge(**kwargs):
    if 'charge' in kwargs and 'x0' not in kwargs:
        return JelliumCylinder(**kwargs)
    else:
        return gpaw.jellium.Jellium(**kwargs)

gpaw.jellium.create_background_charge = create_background_charge

# Load the GPAW calculator
calc = GPAW('cylinder.gpw', txt=None)

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
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, density_xy.T, levels=100, cmap='viridis')
plt.colorbar(contour, label='Electron Density (e/Å³)')
plt.xlabel('x (Å)')
plt.ylabel('y (Å)')
plt.title(f'Electron Density Contour Map at z = {z[z_index]:.2f} Å')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('density_map.png')
plt.show()
