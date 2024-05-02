"""Example script to create visibility maps."""

from fdsvismap import VisMap
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
bg_img = project_root / "misc" / "floorplan.png"  # Create instance of VisMap class

# Set path for FDS simulation directory and background image
sim_dir = str(project_root / "fds_data")

# Create instance of VisMap class
vis = VisMap()
vis.quantity = 'SOOT EXTINCTION COEFFICIENT'

# Read data from FDS simulation directory
vis.read_fds_data(sim_dir)

# Add background image
vis.add_background_image(bg_img)

# Set starpoint and waypoints along escape route
vis.set_start_point(1, 1)
vis.set_waypoint(8, 5, 3, 180)
vis.set_waypoint(9.8, 6, 3, 270)
vis.set_waypoint(17, 0, 3, 0)

# Set times when the simulation should be evaluated
times = range(0, 500, 50)
vis.set_time_points(times)

vis.add_visual_obstruction(8, 8.8, 8.6, 8.8)
# Do the required calculations to create the Vismap
vis.compute_all()

# # Plot ASET map based on Vismaps and save as pdf
fig, ax = vis.create_aset_map_plot(plot_obstructions=True)
ax.set_xlim(0, 20)
ax.set_ylim(10, 0)
plt.savefig('aset_map.pdf', dpi=300)
plt.close()

# # Plot time and waypoint aggregated  Vismap
fig, ax = vis.create_time_agg_wp_agg_vismap()
ax.set_xlim(0, 20)
ax.set_ylim(10, 0)
plt.savefig('time_agg_wp_agg_vismap.pdf', dpi=300)
plt.close()

# Check if waypoint is visible from given location at given time
print(vis.wp_is_visible(50, 12.5, 0.6, 2))

# Get distance from waypoint to given location
print(vis.get_distance_to_wp(17, 5, 2))

# Calculate local visibility at given location and time, considering a specific c factor
print(vis.get_local_visibility(100, 5, 6, 3))
