import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from gpaw import GPAW, PoissonSolver, Mixer
from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
    Power12Potential,
)
from ase.units import Pascal, m, Bohr

# Build a simple Au(111) slab
atoms = fcc111('Au', size=(2, 2, 2), vacuum=6)
atoms.translate([0, 0, -3])
atoms.pbc = [True,True,False]

#   Solvated jellium parameters.
sj = {'excess_electrons': 0.2,
      'grand_output':     False,
      'jelliumregion':    {'top': atoms.cell[2, 2]-3, 'bottom': atoms.cell[2, 2]-6},
     }

# Solvation parameters
cavity = EffectivePotentialCavity(
    effective_potential=Power12Potential(),
    temperature=298.15,
    surface_calculator=GradientSurface()
)
dielectric = LinearDielectric(epsinf=78.36)
interactions = [SurfaceInteraction(surface_tension=18.4e-3 * Pascal * m)]

# Set up the GPAW calculator with implicit solvent
calc = SJM(
    mode='fd',
    xc='PBE',
    kpts=(4,4,1),
    sj=sj,
    convergence={'density': 1e-4},
    cavity=cavity,
    dielectric=dielectric,
    interactions=interactions,
    txt='Au_sjm.txt'
)

atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('Au_sjm.gpw', mode='all')
calc.write_sjm_traces(path='sjm_traces')
