# my_jellium.py

from math import pi
import numpy as np
from ase.units import Bohr
from gpaw.jellium import Jellium

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
