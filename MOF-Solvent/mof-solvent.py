from ase.io import read, write
from ase.io.cube import write_cube
from ase.units import Pascal, m, Bohr
from ase.data.vdw import vdw_radii
from gpaw import GPAW, PoissonSolver, MixerDif
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
)
from io import StringIO
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the xyz file
mof_coord = '''54
Lattice="13.266613 0.0 0.0 -6.633417567293187 11.492202306930997 0.0 0.0 0.0 6.740573"  pbc="T T T"
Zn      -3.31150826       5.74538864       0.00000000        0
Zn       0.00879763      11.47506743       0.00000000        1
Zn       0.00000000       0.00000000       0.00000000        2
Zn       9.65468630       6.01657013       3.37028650        3
Zn       6.35684754       0.27389366       3.37028650        4
Zn      12.99981204       0.28368501       3.37028650        5
O        8.63687484       4.32340098       0.00000000        6
O        7.20860204       1.83828119       0.00000000        7
O       -2.00867324       4.31410379       0.00000000        8
O        1.90180127      11.07212784       0.00000000        9
O       -0.58737065       1.85183050       0.00000000       10
O        4.73662434      11.07826467       0.00000000       11
O       -0.58801632       9.63253413       0.00000000       12
O       -2.00919843       7.18027054       0.00000000       13
O        4.75017973       0.41616712       0.00000000       14
O        8.64064154       7.17509905       0.00000000       15
O        1.89148326       0.42332676       0.00000000       16
O        7.21454301       9.64384246       0.00000000       17
O        8.35593197       4.57619496       3.37028650       18
O        6.94657045       2.11919658       3.37028650       19
O       -2.29339735       4.59367460       3.37028650       20
O        1.61722318      11.34098791       3.37028650       21
O       -0.85564415       2.12907988       3.37028650       22
O        4.47567411      11.34565374       3.37028650       23
O       -0.84953712       9.91097870       3.37028650       24
O       -2.28795233       7.43414478       3.37028650       25
O        4.46637775       0.68411931       3.37028650       26
O        8.35805999       7.45291155       3.37028650       27
O        1.63101063       0.69044002       3.37028650       28
O        6.94617664       9.92263179       3.37028650       29
C        9.21471993       3.17267528       0.00000000       30
C        8.49374469       1.91912883       0.00000000       31
C       10.67078378       3.17145710       0.00000000       32
C        9.22588152       0.66180145       0.00000000       33
C       11.39523917       1.92072625       0.00000000       34
C       10.67107335       0.66349081       0.00000000       35
C       -1.86881437       9.57360212       0.00000000       36
C       -2.59384920       8.32194039       0.00000000       37
C       -2.59376735      10.83392896       0.00000000       38
C       -4.04554835       8.32439973       0.00000000       39
C       -4.03879340      10.83703185       0.00000000       40
C       -4.76786783       9.57641771       0.00000000       41
C        8.94695732       3.43701891       3.37028650       42
C        8.22765368       2.18326561       3.37028650       43
C       10.39891510       3.44208697       3.37028650       44
C        8.95723005       0.92591525       3.37028650       45
C       11.12715810       2.19272369       3.37028650       46
C       10.40238871       0.92927097       3.37028650       47
C       -2.13574116       9.83638281       3.37028650       48
C       -2.86225525       8.58742176       3.37028650       49
C       -2.86232594      11.09760605       3.37028650       50
C       -4.31793442       8.59316786       3.37028650       51
C       -4.30775000      11.10192712       3.37028650       52
C       -5.03727557       9.84841515       3.37028650       53
'''

mof = read(StringIO(mof_coord), format='xyz')
mof.cell = [[13.266613, 0.0, 0.0], [-6.633417567293187, 11.492202306930997, 0.0], [0.0, 0.0, 6.740573]]
mof.wrap()
mof.pbc = [True, True, True]

# Set up the GPAW calculator without solvation or jellium, using your mixer
calc = GPAW(
    mode='fd',
    xc='LDA',
    h = 0.2,
    kpts=(2, 2, 2),
    parallel={'augment_grids': True, 'sl_auto': True},
    txt='mof.txt'
)

# Assign the calculator to the MOF
mof.calc = calc

# Run the calculation
energy = mof.get_potential_energy()

# Retrieve the calculated density
density = calc.density

# Define a function to get atomic radii for the MOF atoms, increasing O and Cu radii by 0.1 Å
def atomic_radii(atoms):
    radii = np.array([vdw_radii[number] for number in atoms.numbers])
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == 'O':
            radii[i] += 0.5
        elif symbol == 'Cu':
            radii[i] += 0.5
        elif symbol == 'C':
            radii[i] += 0.2
    return radii

# Set up the solvation parameters with the calculated density
effective_potential = Power12Potential(atomic_radii=atomic_radii, u0=0.180)

# Initialize atomic radii explicitly
effective_potential.atomic_radii_output = atomic_radii(mof)

cavity = EffectivePotentialCavity(
    effective_potential=effective_potential,
    temperature=298.15,
    surface_calculator=GradientSurface()
)

dielectric = LinearDielectric(epsinf=78.36)
interactions = [SurfaceInteraction(surface_tension=18.4e-3 * Pascal * m)]

# Initialize the cavity with the grid descriptor from the calculation
cavity.set_grid_descriptor(calc.density.gd)
cavity.allocate()

# Now update the cavity using the calculated density
cavity.update(mof, density=density)

# Obtain the cavity grid data directly
cavity_data = cavity.g_g

# Create an Atoms object with atomic information to use for the cube file output
dummy_atoms = mof.copy()

# Write the cavity data to a cube file with atomic positions and labels
with open('mof-cavity.cube', 'w') as cube_file:
    write_cube(cube_file, dummy_atoms, cavity_data, comment='Cavity Density with Atoms Cube File')

print("Cavity density with atoms saved to 'mof-cavity.cube'.")

# Set up the SolvationGPAW calculator with the previously defined cavity and run the calculation again
calc_solvation = SolvationGPAW(
    mode='fd',
    xc='LDA',
    h=0.2,
    kpts=(2, 2, 2),
    parallel={'augment_grids': True, 'sl_auto': True},
    ###poissonsolver=PoissonSolver(),
    cavity=cavity,  # Use the pre-defined cavity
    dielectric=dielectric,
    interactions=interactions,
    txt='mof-solvent.txt'
)

# Assign the new calculator to the MOF
mof.calc = calc_solvation

# Run the calculation with solvation
energy_solvation = mof.get_potential_energy()

# create electron density cube file
density = mof.calc.get_all_electron_density(gridrefinement=4)
write('mof-density.cube', mof, data=density * Bohr**3)

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
