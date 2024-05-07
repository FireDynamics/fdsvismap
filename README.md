# FDSVisMap
Tool for waypoint-based verification of visibility in the scope of performance-based fire safety assessment 

# Installation
Still to come ...

# Usage Example

```python
"""Example script to create visibility maps."""

from fdsvismap import VisMap
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
bg_img = project_root / "misc" / "floorplan.png"

# Set path for FDS simulation directory and background image.
sim_dir = str(project_root / "fds_data")

# Create instance of VisMap class
vis = VisMap()
vis.quantity = 'SOOT EXTINCTION COEFFICIENT'

# Read data from FDS simulation directory.
vis.read_fds_data(sim_dir)

# Add background image.
vis.add_background_image(bg_img)

# Set starpoint and waypoints along escape route.
vis.set_start_point(1, 9)
vis.set_waypoint(8.4, 4.8, 3, 0)
vis.set_waypoint(9.8, 4, 3, 270)
vis.set_waypoint(17, 10, 3, 180)

# Set times when the simulation should be evaluated.
times = range(0, 500, 50)
vis.set_time_points(times)

# Add a visual obstruction that affects visibility calculations.
vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

# Do the required calculations to create the Vismap.
vis.compute_all()

# # Plot ASET map based on Vismaps and save as pdf.
fig, ax = vis.create_aset_map_plot(plot_obstructions=True)
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
plt.savefig('aset_map.pdf', dpi=300)
plt.close()

# # Plot time and waypoint aggregated Vismap and save as pdf.
fig, ax = vis.create_time_agg_wp_agg_vismap_plot()
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
plt.savefig('time_agg_wp_agg_vismap.pdf', dpi=300)
plt.close()

# Set parameters for local evaluations.
time = 500
x = 2
y = 4
waypoint_id = 1

print("\n")

# Check if waypoint is visible from given location at given time.
wp_is_visible = vis.wp_is_visible(500, 2, 4, 1)
print(f"Is waypoint {waypoint_id} visible at {time} s at coordinates X/Y = ({x},{y})?: {wp_is_visible}")

# Get distance from waypoint to given location.
distance_top_wp = vis.get_distance_to_wp(17, 5, 2)
print(f"The distance from waypoint {waypoint_id} to location X/Y = ({x},{y}) is {distance_top_wp} m.")

# Calculate local visibility at given location and time, considering a specific c factor.
local_visibility = vis.get_local_visibility(400, 2, 4, 3)
print(f"The local visibility at location X/Y = ({x},{y}) is {local_visibility:.2f} m.")

```
