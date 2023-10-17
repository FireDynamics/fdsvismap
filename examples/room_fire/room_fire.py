# Import fdsvismap module
from fdsvismap import VisMap

# Set path for FDS simulation directory and background image
sim_dir = 'fds_data'
bg_img = 'misc/floorplan.png'
# Create instance of VisMap class
vis = VisMap(sim_dir, min_vis=10, max_vis=30, eval_height=2, debug=False)

# Add background image
vis.add_background_image(bg_img)

# Set starpoint and waypoints along escape route
vis.set_start_point(1, 1) # startpoint
# waypoints shape: X,Y,C,IOR with = dimensionless constant for ligth emitting (8) and light reflecting (3) signs
vis.set_waypoint(8, 5.5, 3, -2)# waypoints 0, IOR=neg. y axis
vis.set_waypoint(10, 6, 3, -1)# waypoints 1, IOR=neg. x axis
vis.set_waypoint(17, 0, 8, 2)# waypoints 2, IOR=pos. x axis

# Set times when the simulation should be evaluated
vis.set_times([0, 60, 180])

# Do the required calculations to create the Vismap
vis.compute_all()

# Plot time agglomerated absolute boolean vismap -bbox_pad = size of the boxes
vis.plot_time_aggl_abs_bool_vismap(size=[15,10],bbox_pad=1,fontsize=10,display_colorbar=True)
