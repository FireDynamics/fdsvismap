# FDSVisMap

[![PyPI version](https://img.shields.io/pypi/v/fdsvismap.svg)](https://pypi.org/project/fdsvismap/)
[![Code Quality](https://github.com/FireDynamics/fdsvismap/actions/workflows/code_quality.yml/badge.svg)](https://github.com/FireDynamics/fdsvismap/actions/workflows/code_quality.yml)
[![License](https://img.shields.io/github/license/FireDynamics/fdsvismap.svg)](https://github.com/FireDynamics/fdsvismap/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fdsvismap.svg)](https://pypi.org/project/fdsvismap/)

---

**FDSVisMap** is a Python tool for **waypoint-based assessment of visibility** in the context of **performance-based fire safety design**.  
It provides methods for analyzing and visualizing **visibility maps (Vismaps)** derived from FDS (Fire Dynamics Simulator) output data.

---

## Installation


```bash
pip install fdsvismap
```

## Developement

To run code quality checks with Ruff, you can use the following command locally:

```bash
ruff check .
ruff format --check .
```

## Citation 

To cite this work refer to 

```
@article{BORGER2024104269,
title = {A waypoint based approach to visibility in performance based fire safety design},
author = {Kristian Börger and Alexander Belt and Lukas Arnold},
journal = {Fire Safety Journal},
volume = {150},
pages = {104269},
year = {2024},
issn = {0379-7112},
doi = {https://doi.org/10.1016/j.firesaf.2024.104269},
url = {https://www.sciencedirect.com/science/article/pii/S0379711224001826},
}
```

## Usage Example

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

# Read data from FDS simulation directory.
vis.read_fds_data(sim_dir, fds_slc_height=2)

# Add background image.
vis.add_background_image(bg_img)

# Set starpoint and waypoints along escape route.
vis.set_start_point(1, 9)
vis.set_waypoint(1, 8.4, 4.8, 3, 0)
vis.set_waypoint(2, 9.8, 4, 3, 270)
vis.set_waypoint(3, 17, 10, 3, 180)

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
c = 3
waypoint_id = 2

print("\n")

# Check if waypoint is visible from given location at given time.
wp_is_visible = vis.wp_is_visible(time, x, y, waypoint_id)
print(f"Is waypoint {waypoint_id} visible at {time} s at coordinates X/Y = ({x},{y})?: {wp_is_visible}")

# Get distance from waypoint to given location.
distance_top_wp = vis.get_distance_to_wp(x, y, waypoint_id)
print(f"The distance from waypoint {waypoint_id} to location X/Y = ({x},{y}) is {distance_top_wp} m.")

# Calculate local visibility at given location and time, considering a specific c factor.
local_visibility = vis.get_local_visibility(time, x, y, c)
print(f"The local visibility at time {time} s and location X/Y = ({x},{y}) is {local_visibility:.2f} m.")

# Calculate visibility at given location and time relative to a waypoint, considering a specific c factor.
visibility = vis.get_visibility_to_wp(time, x, y, waypoint_id)
print(f"The visibility at time {time} s and location X/Y = ({x},{y}) relative to waypoint {waypoint_id} is {visibility:.2f} m.")

```
