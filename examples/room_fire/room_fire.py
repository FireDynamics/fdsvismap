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

# Add background image, extent is the position of its edges as (x_min, x_max, y_min, y_max) in FDS coordinates.
# The image may also extend beyond the simulation domain.
vis.add_background_image(str(bg_img), extent=(0, 20, 0, 10))

# Add the safety signs with their contrast factor c and their viewing direction alpha, measured clockwise from the
# positive y-axis. Use alpha="omni" for a sign that is visible from all directions.
vis.add_sign(1, 8.4, 4.8, 3, 0)
vis.add_sign(2, 9.8, 4, 3, 270)
vis.add_sign(3, 17, 10, 3, 180)

# Add the route of egress along its waypoints, starting at the first one. The route does not have to pass the
# signs, it is enough to see them.
vis.add_route(
    "exit route",
    [(1, 9), (4, 7), (7, 5.5), (9.5, 4.2), (11, 4.2), (15, 6), (17, 9.5)],
    signs=[1, 2, 3],
)

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

# Plot ASET map of the route and save it as pdf next to this script.
fig, ax = vis.plot_aset_map(route_id="exit route", plot_obstructions=True)
aset_map_file = example_dir / "aset_map.pdf"
fig.savefig(aset_map_file, dpi=300)
plt.close(fig)
print(f"ASET map saved as '{aset_map_file}'.")

# Plot the time aggregated Vismap of the route and save it as pdf next to this script.
fig, ax = vis.plot_time_agg_vismap(route_id="exit route")
time_agg_vismap_file = example_dir / "time_agg_vismap.pdf"
fig.savefig(time_agg_vismap_file, dpi=300)
plt.close(fig)
print(f"Time aggregated Vismap saved as '{time_agg_vismap_file}'.")

# Plot the Vismaps at a single time point side by side, for the whole route and for one sign only. On the map of
# the route, its sections are colored by whether one of its signs is visible from them.
vismap_time = 300
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="compressed")
vis.plot_route_vismap("exit route", vismap_time, ax=axes[0])
axes[0].set_title(f"Route at {vismap_time} s")
vis.plot_sign_vismap(2, vismap_time, ax=axes[1])
axes[1].set_title(f"Sign 2 at {vismap_time} s")
vismap_file = example_dir / f"vismap_{vismap_time}s.pdf"
fig.savefig(vismap_file, dpi=300)
plt.close(fig)
print(f"Vismaps at {vismap_time} s saved as '{vismap_file}'.")

# Set parameters for local evaluations.
simulation_time = 450
x = 2
y = 4
c = 3
sign_id = 2

print()

# Check if a sign is visible from given location at given time.
sign_is_visible = vis.sign_is_visible(simulation_time, x, y, sign_id)
print(
    f"Is sign {sign_id} visible at {simulation_time} s at coordinates X/Y = ({x},{y})?: {sign_is_visible}"
)

# Get distance from a sign to given location.
distance_to_sign = vis.get_distance_to_sign(x, y, sign_id)
print(
    f"The distance from sign {sign_id} to location X/Y = ({x},{y}) is {distance_to_sign} m."
)

# Calculate local visibility at given location and time, considering a specific c factor.
local_visibility = vis.get_local_visibility(simulation_time, x, y, c)
print(
    f"The local visibility at time {simulation_time} s and location X/Y = ({x},{y}) is {local_visibility:.2f} m."
)

# Calculate visibility at given location and time relative to a sign, considering a specific c factor.
visibility = vis.get_visibility_to_sign(simulation_time, x, y, sign_id)
print(
    f"The visibility at time {simulation_time} s and location X/Y = ({x},{y}) relative to sign {sign_id} is {visibility:.2f} m."
)

# Evaluate the route as a whole: the first time at which each of its sections is without a visible sign.
route_aset = vis.get_route_aset("exit route")
print(
    f"The first section of the route loses its sign after {route_aset.min():.0f} s, "
    f"the last one after {route_aset.max():.0f} s."
)
