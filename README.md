# FDSVisMap
Tool for waypoint-based verification of visibility in the scope of performance-based fire safety assessment.

Rescue pathways within a building are imperative for an individual's self-rescue. Crucial factors, such as exposure to hazardous smoke and thermal elements, significantly decrease the possibility of escaping during a fire incident. Visibility, impacted by smoke, influences speed and orientation and is therefore a big factor for a successful evacuation [1] [2]. Visibility can be empirically assessed based on smoke density. With the help of the computational fluid dynamics (CFD) model "FDS", this parameter can be calculated for a defined fire scenario. The tool FDSVisMap proposes an innovative methodology for evaluating the viability of evacuation routes in context of fire-induced smoke production. The approach encompasses a location-based computation of visibility, utilizing FDS-calculated smoke density, with the quality of the building's emergency signs serving as computational reference points. The objective is to generate straightforward schematic representations that can be utilized in an engineering context.
***
[1] Fridolf, K., Nilsson, D., Frantzich, H., Ronchi, E., & Arias, S. (2017). WALKING SPEED IN SMOKE: REPRESENTATION IN LIFE SAFETY VERIFICATIONS. 

[2] Chen, J., Wang, J., Wang, B., Liu, R., & Wang, Q. (2018). An experimental study of visibility effect on evacuation speed on stairs. Fire Safety Journal, 96, 189–202. https://doi.org/10.1016/j.firesaf.2017.11.010


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
```
# Preview Notebook
[View the example notebook](https://github.com/Haukiy/fdsvismap/blob/main/demo_fds_vismap_nb.ipynb)
