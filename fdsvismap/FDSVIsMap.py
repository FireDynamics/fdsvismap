import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
import matplotlib.colors

import fdsreader as fds
from .helper_functions import find_closest_point
from .Waypoint import Waypoint

class VisMap:
    """
    A class to represent a Visibility Map (VisMap) based on Fire Dynamics Simulator (FDS) data.

    :ivar sim_dir: The directory containing the simulation data.
    :vartype sim_dir: str
    :ivar slc: Slice object for visibility calculations. Initialized as None.
    :vartype slc: object
    :ivar start_point: The starting point for the path. Initialized as None.
    :vartype start_point: tuple
    :ivar way_points_list: List of waypoints for the path. Initialized as an empty list.
    :vartype way_points_list: list
    :ivar mean_extco_array_list: List of mean extinction coefficient arrays. Initialized as an empty list.
    :vartype mean_extco_array_list: list
    :ivar view_array_list: List of view arrays. Initialized as an empty list.
    :vartype view_array_list: list
    :ivar distance_array_list: List of distance arrays. Initialized as an empty list.
    :vartype distance_array_list: list
    :ivar vis_map_array: The array representing the visibility map. Initialized as None.
    :vartype vis_map_array: np.ndarray
    """
    def __init__(self, sim_dir, min_vis=0, max_vis=30, eval_height=2):
        """
        Initialize the VisMap object.

        :param sim_dir: The directory containing the simulation data.
        :type sim_dir: str
        :param min_vis: The minimum visibility value. Default is 0.
        :type min_vis: float, optional
        :param max_vis: The maximum visibility value. Default is 30.
        :type max_vis: float, optional
        :param eval_height: The height at which to evaluate visibility. Default is 2.
        :type eval_height: float, optional
        """
        self.sim_dir = sim_dir
        self.slc = None
        self.start_point = None
        self.way_points_list = []
        self.mean_extco_array_list = []
        self.view_array_list = []
        self.distance_array_list = []
        self.vismap_list = []
        self.colission_array_list = []
        self.delta_array_list = []
        self.min_vis = min_vis # minimum visibility to be required #Todo: set individual for each waypoint
        self.max_vis = max_vis # maximum visibility to be considered #Todo: set individual for each waypoint
        self.background_image = None
        self.view_angle = True
        self.absolute_boolean_vismap_dict = {}
        self.time_agglomerated_absolute_boolean_vismap = None
        self._read_fds_data()

    def _get_waypoint(self, waypoint_id):
        """
        Retrieve the parameters for a specific waypoint by its ID.

        :param waypoint_id: The ID of the waypoint.
        :type waypoint_id: int
        :return: Waypoint object.
        :rtype: Waypoint
        """
        return self.way_points_list[waypoint_id]

    def set_start_point(self, x, y):
        """
        Set the starting point for the path.

        :param x: X coordinate of the starting point.
        :type x: float
        :param y: Y coordinate of the starting point.
        :type y: float
        """
        self.start_point = (x, y)

    def set_waypoint(self, x, y, c=3, ior=None):
        """
        Add a waypoint to the list.

        :param x: X coordinate of the waypoint referring to global FDS coordinates.
        :type x: float
        :param y: Y coordinate of the waypoint referring to global FDS coordinates.
        :type y: float
        :param c: Contrast factor for exit sign according to JIN.
        :type c: int, optional
        :param ior: Orientation of the exit sign according to FDS orientations.
        :type ior: int or None, optional
        """
        self.way_points_list.append(Waypoint(x, y, c, ior))

    def _read_fds_data(self, quantity='OD_C0.9H0.1', slice=None): #: Todo: specify slice closest to given height
        """
        Read FDS data and store relevant slices and obstructions.

        :param quantity: Quantity of FDS slice file to be evaluated.
        :type quantity: str
        :param slice: Index of FDS slice file to be evaluated.
        :type slice: int
        """
        quantity = 'ext_coef_C0.9H0.1'
        # quantity = 'VIS_C0.9H0.1'
        sim = fds.Simulation(self.sim_dir)
        print(sim.slices)
        self.slc = sim.slices.filter_by_quantity(quantity)[0]
        self.obstructions = sim.obstructions
        self.all_x_coords = self.slc.coordinates["x"]
        self.all_y_coords = self.slc.coordinates["y"]

    def _get_extco_array(self, time):
        """
        Get the array of extinction coefficients at a given time.

        :param time: Timestep to evaluate.
        :type time: float
        :return: Array of extinction coefficients at the specified time.
        :rtype: np.ndarray
        """
        time_index = self.slc.get_nearest_timestep(time)
        data = self.slc.to_global_nonuniform()[time_index]
        extco_array = data
        return extco_array

    def _get_mean_extco_array(self, waypoint_id, time):
        """
        Get the array of mean extinction coefficients between the waypoint and all cells.

        :param waypoint_id: Waypoint ID of the exit sign.
        :type waypoint_id: int
        :param time: Timestep to evaluate.
        :type time: float
        :return: Array of mean extinction coefficients.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        extco_array = self._get_extco_array(time)
        i_ref = find_closest_point(self.all_x_coords, wp.x)
        j_ref = find_closest_point(self.all_y_coords, wp.y)
        mean_extco_array = np.zeros_like(extco_array)
        for i, x in enumerate(self.all_x_coords):
            for j, y in enumerate(self.all_y_coords):
                img = np.zeros_like(extco_array)
                rr, cc = line(i_ref, j_ref, i, j)
                img[rr, cc] = 1
                n_cells = np.count_nonzero(img)
                mean_extco = np.sum(extco_array * img) / n_cells
                mean_extco_array[i, j] = mean_extco
        return mean_extco_array

    def _get_dist_array(self, waypoint_id):
        """
        Get the array containing distances between the waypoint and all cells.

        :param waypoint_id: Waypoint ID.
        :type waypoint_id: int
        :return: Array of distances.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        self.xv, self.yv = np.meshgrid(self.all_x_coords, self.all_y_coords)
        distance_array = np.sqrt((self.xv - x)**2 + (self.yv - y)**2)
        self.distance_array_list.append(distance_array)
        return distance_array

    def _get_view_array(self, waypoint_id):
        """
        Get the view array considering view angles.

        :param waypoint_id: Waypoint ID.
        :type waypoint_id: int
        :return: View array.
        :rtype: np.ndarray
        """
        distance_array = self._get_dist_array(waypoint_id)
        wp = self._get_waypoint(waypoint_id)

        # calculate cosinus for every cell from total distance and x / y distance
        if self.view_angle == True and wp.ior != None:
            if wp.ior == 1 or wp.ior == -1:
                view_angle_array = abs((self.xv - wp.x) / distance_array)
            elif wp.ior == 2 or wp.ior == -2:
                view_angle_array = abs((self.yv - wp.y) / distance_array)
        else:
            view_angle_array = np.ones_like(distance_array)

        #  set visibility to zero on all cells that are behind the waypoint and against view direction
        if wp.ior == -1:
            view_array = np.where(self.xv < wp.x, view_angle_array, 0)
        elif wp.ior == 1:
            view_array = np.where(self.xv > wp.x, view_angle_array, 0)
        elif wp.ior == -2:
            view_array = np.where(self.yv < wp.y, view_angle_array, 0)
        elif wp.ior == 2:
            view_array = np.where(self.yv > wp.y, view_angle_array, 0)
        else:
            view_array = view_angle_array
        self.view_array_list.append(view_array)
        return view_array

    def _get_collision_array(self, waypoint_id):
        """
        Calculate the collision array indicating which cells obstruct visibility to a waypoint.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :param z: Height where collision between waypoint and all cells is evaluated.
        :type z: float
        :return: Boolean array describing if the waypoint is visible from a certain cell.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        i_ref = find_closest_point(self.all_x_coords, wp.x)
        j_ref = find_closest_point(self.all_y_coords, wp.y)
        extco_array = self._get_extco_array(0)
        obst_array = np.zeros_like(extco_array)
        for obst in self.obstructions:
            for sub_obst in obst:
                _, x_extend, y_extend, z_extend = sub_obst.extent
                if z_extend[0] <= z <= z_extend[1]:
                    x_i_min = (np.abs(self.all_x_coords - x_extend[0])).argmin()
                    x_i_max = (np.abs(self.all_x_coords - x_extend[1])).argmin()
                    y_i_min = (np.abs(self.all_y_coords - y_extend[0])).argmin()
                    y_i_max = (np.abs(self.all_y_coords - y_extend[1])).argmin()
                    obst_array[x_i_min:x_i_max, y_i_min:y_i_max] = True
        obst_array = np.flip(obst_array, axis=1)
        final = np.zeros_like(obst_array)
        b = final.copy()
        edges = np.ones_like(obst_array)
        edges[1:-1, 1:-1] = False
        edge_x, edge_y = np.where(edges == True)
        for i, j in zip(edge_x, edge_y):
            b_x, b_y = line(i_ref, j_ref, i, j)
            b[b_x, b_y] = True
            cuts = np.where((obst_array == True) & (b == True))
            if cuts[0].size != 0:
                x_cut_coord = np.in1d(b_x, cuts[0])
                x_cut_index = np.where(x_cut_coord == True)[0][0]
                final[b_x[:x_cut_index], b_y[:x_cut_index]] = True
            else:
                final[b_x, b_y] = True
            b = final.copy()
        self.colission_array_list.append(final.T)
        return final.T

    def _get_vismap(self, waypoint_id, timestep):
        """
        Get the visibility map for a specific waypoint.

        :param waypoint_id: Waypoint ID.
        :type waypoint_id: int
        :param timestep: Timestep to evaluate.
        :type timestep: float
        :return: Visibility map.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        mean_extco_array = self._get_mean_extco_array(waypoint_id, timestep)
        vis_array = wp.c / mean_extco_array.T
        vismap = np.where(vis_array > self.max_vis, self.max_vis, vis_array).astype(float)
        return vismap

    def get_bool_vismap(self, waypoint_id, timestep, extinction=True, viewangle=True, colission=True):#TODO: make z value changable
        """
        Generate a boolean visibility map for a specific waypoint.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :param timestep: Timestep for which to calculate the visibility map.
        :type timestep: float
        :param extinction: Flag indicating whether to consider extinction coefficients. Default is True.
        :type extinction: bool, optional
        :param viewangle: Flag indicating whether to consider view angles. Default is True.
        :type viewangle: bool, optional
        :param colission: Flag indicating whether to consider obstructions. Default is True.
        :type colission: bool, optional
        :return: Boolean visibility map for the given waypoint.
        :rtype: np.ndarray
        """
        if viewangle == True:
            view_array = self._get_view_array(waypoint_id)
        else:
            view_array = 1
        if extinction == True:
            vismap = self._get_vismap(waypoint_id, timestep)
        else:
            vismap = self.max_vis
        distance_array = self._get_dist_array(waypoint_id)
        if colission == True:
            colission_array = self._get_col_array(waypoint_id, z)
        else:
            colission_array = 1
        vismap_total = view_array * vismap * colission_array
        delta_map = np.where(vismap_total >= distance_array, True, False)
        delta_map = np.where(vismap_total < self.min_vis, False, delta_map)
        return delta_map

    def get_abs_bool_vismap(self, timestep, extinction=True, viewangle=True):
        """
        Generate an absolute boolean visibility map for all waypoints.

        :param timestep: Timestep for which to calculate the visibility map.
        :type timestep: float
        :param extinction: Flag indicating whether to consider extinction coefficients. Default is True.
        :type extinction: bool, optional
        :param viewangle: Flag indicating whether to consider view angles. Default is True.
        :type viewangle: bool, optional
        :return: Absolute boolean visibility map.
        :rtype: np.ndarray
        """
        boolean_vismap_list = []
        for waypoint_id, waypoint in enumerate(self.way_points_list):
            boolean_vismap = self.get_bool_vismap(waypoint_id, timestep, extinction=extinction, viewangle=viewangle)
            boolean_vismap_list.append(boolean_vismap)
            absolute_boolean_vismap = np.logical_or.reduce(boolean_vismap_list)
            self.absolute_boolean_vismap_dict[timestep] = absolute_boolean_vismap
        return absolute_boolean_vismap

    def get_time_aggl_abs_bool_vismap(self):
        """
        Calculate the time-agglomerated absolute boolean visibility map.

        :return: Time-agglomerated absolute boolean visibility map.
        :rtype: np.ndarray
        """
        self.time_agglomerated_absolute_boolean_vismap = np.logical_and.reduce(list(self.absolute_boolean_vismap_dict.values()))
        return self.time_agglomerated_absolute_boolean_vismap

    def plot_abs_bool_vismap(self): # Todo: is duplicate of plot_time_agglomerated_absolute_boolean_vismap
        """
        Plot the absolute boolean visibility map.
        """

        # if self.time_agglomerated_absolute_boolean_vismap == None:
        #     self.get_time_agglomerated_absolute_boolean_vismap()
        extent = (self.all_x_coords[0], self.all_x_coords[-1], self.all_y_coords[-1], self.all_y_coords[0])
        if self.background_image is not None:
            plt.imshow(self.background_image, extent=extent)
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])

        plt.imshow(self.absolute_boolean_vismap_dict, cmap=cmap, extent=extent, alpha=0.3)
        x, y, _, _ = zip(*self.way_points_list)
        plt.plot((self.start_point[0], *x), (self.start_point[1], *y), color='darkgreen', linestyle='--')
        plt.scatter((self.start_point[0], *x), (self.start_point[1], *y), color='darkgreen')
        plt.xlabel("X / m")
        plt.ylabel("Y / m")

    def add_background_image(self, file):
        """
        Add a background image to the plot.

        :param file: Path to the background image file.
        :type file: str
        """
        self.background_image = plt.imread(file)

    def plot_time_aggl_abs_bool_vismap(self):
        """
        Plot the time-agglomerated absolute boolean visibility map.
        """

        # if self.time_agglomerated_absolute_boolean_vismap == None:
        #     self.get_time_agglomerated_absolute_boolean_vismap()
        extent = (self.all_x_coords[0], self.all_x_coords[-1], self.all_y_coords[-1], self.all_y_coords[0])
        if self.background_image is not None:
            plt.imshow(self.background_image, extent=extent)
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])

        plt.imshow(self.time_agglomerated_absolute_boolean_vismap, cmap=cmap, extent=extent, alpha=0.5)
        x_values = [wp.x for wp in self.way_points_list]
        y_values = [wp.y for wp in self.way_points_list]

        plt.plot((self.start_point[0], *x_values),(self.start_point[1], *y_values), color='darkgreen', linestyle='--')
        plt.scatter((self.start_point[0], *x_values),(self.start_point[1], *y_values), color='darkgreen')
        for wp_id, wp in enumerate(self.way_points_list):
            plt.annotate(f"WP : {wp_id:>}\nC : {wp.c:>}\nIOR : {wp.ior}", xy=(wp.x+0.3, wp.y+1.5),  bbox=dict(boxstyle="round", fc="w"), fontsize=6)
        plt.xlabel("X / m")
        plt.ylabel("Y / m")
        plt.show()
