# Import fdsvismap module
from fdsvismap import VisMap
import matplotlib.pyplot as plt
# Set path for FDS simulation directory and background image
sim_dir = 'fds_data'
bg_img = 'misc/floorplan.png' # Create instance of VisMap class
vis = VisMap(sim_dir, max_vis=30, min_vis=0)
vis.num_edge_cells =1

# Add background image
vis.add_background_image(bg_img) #sad

# Set starpoint and waypoints along escape route
vis.set_start_point(1, 1)
vis.set_waypoint(8, 5, 8, 180)
vis.set_waypoint(9.8, 6, 8, 270)
vis.set_waypoint(17, 0, 8, 0)

# Set times when the simulation should be evaluated
times = range(0, 500, 50)
vis.set_times(times)

# Do the required calculations to create the Vismap
vis.compute_all(view_angle=True)
aset_map = vis.get_aset_map()

# # Plot time agglomerated absolute boolean vismap
fig, ax = vis.create_time_aggl_abs_bool_vismap()
# fix, ax = vis.create_aset_map_plot()
plt.show()
