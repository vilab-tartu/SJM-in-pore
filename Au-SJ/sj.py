import os
import numpy as np
from ase.build import fcc111
from ase.units import Pascal, m
from gpaw import GPAW
from gpaw.jellium import JelliumSlab
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
    Power12Potential,
)
from gpaw.solvation.poisson import WeightedFDPoissonSolver

def flatten_potential(potential, z, z_end, thickness=3.0):
    """Flatten the potential by subtracting a linear fit in the last `thickness` Å of the slab."""
    z_start = z_end - thickness
    indices = np.where((z >= z_start) & (z <= z_end))[0]
    if len(indices) < 2:
        print("Not enough points in the selected range to fit. Check z_start and z_end values.")
        return potential

    # Perform a linear fit in the specified region
    slope, intercept = np.polyfit(z[indices], potential[indices], 1)

    # Correct the potential by subtracting the fitted line
    corrected_potential = potential - (slope * z + intercept)
    
    return corrected_potential

def ensure_same_length(array1, array2):
    """Truncate or pad arrays to the same length."""
    min_length = min(len(array1), len(array2))
    return array1[:min_length], array2[:min_length]

def write_flattened_potential(calc, path='sj_traces', thickness=3.0):
    """Write the flattened potential to file using the last `thickness` Å of the slab."""
    if not os.path.exists(path):
        os.makedirs(path)

    # Extract the electrostatic potential and average over x and y
    potential = calc.get_electrostatic_potential()
    averaged_potential = potential.mean(axis=(0, 1))

    # Prepare z coordinates
    z = np.linspace(0, calc.atoms.cell[2, 2], len(averaged_potential), endpoint=False)
    z_end = calc.atoms.cell[2, 2]  # Use the full height of the cell as the end point for fitting

    # Apply the flattening correction
    flattened_potential = flatten_potential(averaged_potential, z, z_end, thickness)

    # Save the corrected potential to a text file
    np.savetxt(os.path.join(path, 'potential.txt'), np.column_stack((z, flattened_potential)))

    # Trim z to match cavity and background charge arrays
    z_trimmed = z[:len(flattened_potential)]

    # Optionally handle cavity and background charge traces if available
    if hasattr(calc.hamiltonian, 'cavity') and hasattr(calc.hamiltonian.cavity, 'g_g'):
        cavity = calc.hamiltonian.cavity.g_g
        cavity_z = cavity.mean(axis=(0, 1))
        z_trimmed, cavity_z = ensure_same_length(z_trimmed, cavity_z)  # Ensure matching lengths
        np.savetxt(os.path.join(path, 'cavity.txt'), np.column_stack((z_trimmed, cavity_z)))
    else:
        print("Cavity data not available in this calculation.")

    if hasattr(calc.density, 'background_charge') and hasattr(calc.density.background_charge, 'mask_g'):
        background_charge = calc.density.background_charge.mask_g
        background_charge_z = background_charge.mean(axis=(0, 1))
        z_trimmed, background_charge_z = ensure_same_length(z_trimmed, background_charge_z)  # Ensure matching lengths
        np.savetxt(os.path.join(path, 'background_charge.txt'), np.column_stack((z_trimmed, background_charge_z)))
    else:
        print("Background charge data not available in this calculation.")

# Build a simple Au(111) slab
atoms = fcc111('Au', size=(2, 2, 2), vacuum=6)
atoms.translate([0, 0, -3])
atoms.pbc = [True, True, False]  # Non-periodic in z

# Set up the JelliumSlab background charge
jellium = JelliumSlab(
    charge=0.2,
    z1=atoms.cell[2, 2] - 6,
    z2=atoms.cell[2, 2] - 3
)

# Solvation parameters
cavity = EffectivePotentialCavity(
    effective_potential=Power12Potential(),
    temperature=298.15,
    surface_calculator=GradientSurface()
)
dielectric = LinearDielectric(epsinf=78.36)
interactions = [SurfaceInteraction(surface_tension=18.4e-3 * Pascal * m)]

# Set up the GPAW calculator with implicit solvent using a compatible Poisson solver
poissonsolver = WeightedFDPoissonSolver()  # Use a solver compatible with dielectric corrections

calc = SolvationGPAW(
    mode='fd',
    xc='PBE',
    kpts=(4, 4, 1),
    background_charge=jellium,
    convergence={'density': 1e-4},
    poissonsolver=poissonsolver,
    cavity=cavity,
    dielectric=dielectric,
    interactions=interactions,
    txt='Au_sj.txt'
)

atoms.calc = calc

# Run the calculation
energy = atoms.get_potential_energy()

# Save the calculation results
calc.write('Au_sj.gpw', mode='all')

# Write potential and other traces with flattening correction applied
write_flattened_potential(calc, path='sj_traces', thickness=3.0)
