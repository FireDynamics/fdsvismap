"""Example script to create visibility maps."""

import time
from pathlib import Path

import matplotlib.pyplot as plt

from fdsvismap import VisMap

# All paths refer to the directory of this script, so the example runs from any working directory.
example_dir = Path(__file__).parent
sim_dir = example_dir / "fds_data"
bg_img = example_dir / "misc" / "floorplan.png"

# Create instance of VisMap class.
vis = VisMap()

# Read data from FDS simulation directory.
vis.read_fds_data(str(sim_dir), fds_slc_height=2)

# Add background image.
vis.add_background_image(str(bg_img))

# Set start point and waypoints along escape route.
vis.set_start_point(1, 9)
vis.set_waypoint(1, 8.4, 4.8, 3, 0)
vis.set_waypoint(2, 9.8, 4, 3, 270)
vis.set_waypoint(3, 17, 10, 3, 180)

# Set times when the simulation should be evaluated.
times = range(0, 500, 50)
vis.set_time_points(times)

# Add a visual obstruction that affects visibility calculations.
vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

# Do the required calculations to create the Vismap, progress=True shows progress bars.
print("Starting computation...")
start_time = time.perf_counter()
vis.compute_all(progress=True)
print(f"Computation completed in {time.perf_counter() - start_time:.2f} seconds.")

# Plot ASET map based on Vismaps and save it as pdf next to this script.
fig, ax = vis.create_aset_map_plot(plot_obstructions=True)
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
aset_map_file = example_dir / "aset_map.pdf"
fig.savefig(aset_map_file, dpi=300)
plt.close(fig)
print(f"ASET map saved as '{aset_map_file}'.")

# Plot time and waypoint aggregated Vismap and save it as pdf next to this script.
fig, ax = vis.create_time_agg_wp_agg_vismap_plot()
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
vismap_file = example_dir / "time_agg_wp_agg_vismap.pdf"
fig.savefig(vismap_file, dpi=300)
plt.close(fig)
print(f"Time and waypoint aggregated Vismap saved as '{vismap_file}'.")

# Set parameters for local evaluations.
simulation_time = 450
x = 2
y = 4
c = 3
waypoint_id = 2

print()

# Check if waypoint is visible from given location at given time.
wp_is_visible = vis.wp_is_visible(simulation_time, x, y, waypoint_id)
print(
    f"Is waypoint {waypoint_id} visible at {simulation_time} s at coordinates X/Y = ({x},{y})?: {wp_is_visible}"
)

# Get distance from waypoint to given location.
distance_to_wp = vis.get_distance_to_wp(x, y, waypoint_id)
print(
    f"The distance from waypoint {waypoint_id} to location X/Y = ({x},{y}) is {distance_to_wp} m."
)

# Calculate local visibility at given location and time, considering a specific c factor.
local_visibility = vis.get_local_visibility(simulation_time, x, y, c)
print(
    f"The local visibility at time {simulation_time} s and location X/Y = ({x},{y}) is {local_visibility:.2f} m."
)

# Calculate visibility at given location and time relative to a waypoint, considering a specific c factor.
visibility = vis.get_visibility_to_wp(simulation_time, x, y, waypoint_id)
print(
    f"The visibility at time {simulation_time} s and location X/Y = ({x},{y}) relative to waypoint {waypoint_id} is {visibility:.2f} m."
)
