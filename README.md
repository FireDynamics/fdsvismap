# FDSVisMap

[![PyPI version](https://img.shields.io/pypi/v/fdsvismap.svg)](https://pypi.org/project/fdsvismap/)
[![Code Quality](https://github.com/FireDynamics/fdsvismap/actions/workflows/code_quality.yml/badge.svg)](https://github.com/FireDynamics/fdsvismap/actions/workflows/code_quality.yml)
[![License](https://img.shields.io/github/license/FireDynamics/fdsvismap.svg)](https://github.com/FireDynamics/fdsvismap/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/fdsvismap.svg)](https://pypi.org/project/fdsvismap/)
![type checked with mypy](https://img.shields.io/badge/type%20checked-mypy-039dfc)

---

**FDSVisMap** is a Python tool for **waypoint-based assessment of visibility** in the context of **performance-based fire safety design**.

It provides methods for analyzing and visualizing **visibility maps (Vismaps)** derived from FDS (Fire Dynamics Simulator) output data.

---

## Installation & Setup

```bash
pip install fdsvismap
```

### Installation with uv

```bash
# Install fdsvismap in editable mode with its dependencies and the dev dependency group
uv sync

# Additionally install the dependencies for building the documentation
uv sync --extra docs
```

### Running Tests Locally

To run all quality checks (linting, formatting, type checking) as well as the tests, use the following commands:

```bash
uv run pre-commit run --all-files
uv run pytest
```

On Linux and macOS, or with Git Bash on Windows, `./scripts/ci.sh` runs both steps at once.

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

## FDS Slice File (SLCF) Requirements

FDSVisMap requires specific slice file data from your FDS simulation. The tool uses **soot extinction coefficient** or **soot optical density** to calculate visibility.

### Required FDS Input

Add the following to your FDS input file (`.fds`):

```fds
&SLCF QUANTITY='EXTINCTION COEFFICIENT', CELL_CENTERED=T, PBZ=2.0 /
```

Or for optical density:

```fds
&SLCF QUANTITY='OPTICAL DENSITY', CELL_CENTERED=T, PBZ=2.0 /
```

- `PBZ=2.0` sets the z-coordinate (position) of the slice plane (adjust as needed).
- The slice plane height (z-coordinate, corresponding to `PBZ` in FDS) is selected in Python via `fds_slc_height`.
- `CELL_CENTERED=T` writes the values at the cell centres, as in the examples of this repository.
- In the FDS output, these quantities are named `SOOT EXTINCTION COEFFICIENT` and `SOOT OPTICAL DENSITY`. FDS does not accept these names in the input file.
- If smoke is defined as a separate species (`SPEC_ID`), the quantity is named after the species, e.g. `MY SMOKE EXTINCTION COEFFICIENT`. Select such a slice by its ID with `fds_slc_id`.

### Supported Quantities

| Python `quantity` value | Quantity in the FDS output |
|------------------------|--------------|
| `ext_coef_C0.9H0.1` (default) | `SOOT EXTINCTION COEFFICIENT` |
| `ext_coef_C` | `SOOT EXTINCTION COEFFICIENT` |
| `OD_C` | `SOOT OPTICAL DENSITY` |
| `OD_C0.9H0.1` | `SOOT OPTICAL DENSITY` |

You can set `vis.quantity` either to these Python-side names (recommended) or to the FDS quantity names (for example, `'EXTINCTION COEFFICIENT'`, `'SOOT EXTINCTION COEFFICIENT'`, `'OPTICAL DENSITY'` or `'SOOT OPTICAL DENSITY'`); all of them are accepted as aliases.

```python
vis = VisMap()

# Default: uses 'SOOT EXTINCTION COEFFICIENT' (Python-side name: "ext_coef_C0.9H0.1")
vis.read_fds_data(sim_dir, fds_slc_height=2.0)

# Or explicitly set the quantity using the Python-side name
vis.quantity = "ext_coef_C"
vis.read_fds_data(sim_dir, fds_slc_height=2.0)

# Or use optical density (Python-side name: "OD_C")
vis.quantity = "OD_C"
vis.read_fds_data(sim_dir, fds_slc_height=2.0)
```

### Integration with fdsreader

FDSVisMap uses [fdsreader](https://github.com/firemodels/fdsreader) internally to read FDS output files. The `read_fds_data()` method automatically handles:
- Loading the simulation directory via `fds.Simulation(sim_dir)`
- Finding the appropriate slice file by quantity and height
- Extracting grid coordinates and time points

No manual fdsreader usage is required.

## Usage Example

The following script is part of the repository as `examples/room_fire/room_fire.py`, together with the FDS output of the example in `examples/room_fire/fds_data`. The FDS output is stored with [Git LFS](https://git-lfs.com), which has to be installed to get the data when cloning the repository. All paths refer to the directory of the script, so it runs from any working directory on Linux, macOS and Windows and saves the plots next to itself:

```bash
python examples/room_fire/room_fire.py
```

```python
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
```
