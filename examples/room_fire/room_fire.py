from fdsvismap import VisMap

sim_dir = 'fds_data'
bg_img = 'misc/floorplan.png'
vis = VisMap(sim_dir, max_vis=30, min_vis=10)
vis.set_start_point(1, 1)
vis.set_way_point(8, 5.4, 3, -2)
vis.set_way_point(10, 6, 3, -1)
vis.set_way_point(17, 0, 3, 2)
vis.get_absolute_boolean_vismap(0)
vis.get_absolute_boolean_vismap(60)
vis.get_absolute_boolean_vismap(120)

vis.add_background_image(bg_img) #sad
vis.get_time_agglomerated_absolute_boolean_vismap()

vis.plot_time_agglomerated_absolute_boolean_vismap()
