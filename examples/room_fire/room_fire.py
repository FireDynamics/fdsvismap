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
vis.set_waypoint(8, 5, 3, 180)
vis.set_waypoint(9.8, 6, 3, 270)
vis.set_waypoint(17, 0, 3, 0)

# Set times when the simulation should be evaluated
vis.set_times([0, 60, 120])

# Do the required calculations to create the Vismap
vis.compute_all()

# Plot time agglomerated absolute boolean vismap
vis.plot_time_aggl_abs_bool_vismap()
