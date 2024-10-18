import os
from ase.io import read, write
from ase.visualize import view
from gpaw import restart

# Load the GPAW file (mof.gpw)
atoms, calc = restart('mof.gpw')

# Define projection settings for POV-Ray
generic_projection_settings = {
    'rotation': '0x,0y,90z',  # Adjust rotation as needed
    'radii': 0.9,               # Atom radius in the visualization
    'colors': None,             # Use default atom colors
    'show_unit_cell': 2,        # Show all of the unit cell
}

# Define POV-Ray settings
povray_settings = {
    'display': False,                  # Do not display while rendering
    'transparent': False,              # Opaque background
    'camera_type': 'orthographic',     # Use orthographic camera for full cell view
    'camera_dist': 5000,              # Adjust camera distance to see the full cell
    'canvas_width': 1024,              # Output image width in pixels
    'canvas_height': None,             # Output image height (None = auto)
    'image_plane': None,               # Distance from front atom to image plane
    'depth_cueing': False,             # No fading with distance
    'point_lights': [],                # No specific point lights
    'area_light': [(2, 3, 40), 'White', 20, 20, 5, 5],  # Soft lighting settings
    'celllinewidth': 0.05,             # Line thickness for unit cell
}

# Define output filenames
pov_file = 'mof_view.pov'
png_file = 'mof_view.png'

# Write the .pov file for the structure
write(pov_file, atoms, format='pov', povray_settings=povray_settings, **generic_projection_settings)

# Run POV-Ray to generate the PNG image
os.system(f'povray +I{pov_file} +O{png_file} +W2048 +H2048 +A +AM2 +UA +Q9')

print(f'POV-Ray figure saved as {png_file}')
