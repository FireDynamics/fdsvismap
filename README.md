# FDSVisMap
Tool for waypoint-based verification of visibility in the scope of performance-based fire safety assessment 

# Installation
Still to come ...

# Usage Example
```python
# Import fdsvismap module
from fdsvismap import VisMap

# Set path for FDS simulation directory and background image
sim_dir = 'fds_data'
bg_img = 'misc/floorplan.png'

# Create instance of VisMap class
vis = VisMap(sim_dir, max_vis=30, min_vis=10)

# Add background image
vis.add_background_image(bg_img) #sad

# Set starpoint and waypoints along escape route
vis.set_start_point(1, 1)
vis.set_waypoint(8, 5.4, 3, -2)
vis.set_waypoint(10, 6, 3, -1)
vis.set_waypoint(17, 0, 3, 2)

# Compute vismap for different timesteps
vis.get_abs_bool_vismap(0)
vis.get_abs_bool_vismap(60)
vis.get_abs_bool_vismap(120)

# Compute time agglomerated absolute boolean vismap
vis.get_time_aggl_abs_bool_vismap()

# Plot time agglomerated absolute boolean vismap
vis.plot_time_aggl_abs_bool_vismap()
```
