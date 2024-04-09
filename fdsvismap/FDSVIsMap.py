import numpy as np
from skimage.draw import line, line_aa
import matplotlib.pyplot as plt
import matplotlib.colors

import fdsreader as fds
from fdsvismap.helper_functions import find_closest_point
from fdsvismap.Waypoint import Waypoint


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
    :vartype way_points_list: list[Waypoint]
    :ivar mean_extco_array_list: List of mean extinction coefficient arrays. Initialized as an empty list.
    :vartype mean_extco_array_list: list
    :ivar view_array_list: List of view arrays. Initialized as an empty list.
    :vartype view_array_list: list
    :ivar distance_array_list: List of distance arrays. Initialized as an empty list.
    :vartype distance_array_list: list
    :ivar vis_map_array: The array representing the visibility map. Initialized as None.
    :vartype vis_map_array: np.ndarray
    :ivar quantity: Quantity of FDS slice file to be evaluated.
    :vartype quantity: str
    :ivar num_edge_cells: Number thickness of edge cells that are considered for collision detection.
    :vartype num_edge_cells: int
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
        self.times = None
        self.quantity = 'ext_coef_C0.9H0.1'
        self.sim_dir = sim_dir
        self.slc = None
        self.start_point = None
        self.way_points_list = []
        self.mean_extco_array_list = []
        self.view_array_list = []
        self.distance_array_list = []
        self.vismap_list = []
        self.non_concealed_cells_array_list = []
        self.non_concealed_cells_xy_idx_dict = {}
        self.delta_array_list = []
        self.min_vis = min_vis  # minimum visibility to be required #Todo: set individual for each waypoint
        self.max_vis = max_vis  # maximum visibility to be considered #Todo: set individual for each waypoint
        self.eval_height = eval_height  # height z where the FDS slice is evaluated and collision is checked
        self.background_image = None
        self.view_angle = True
        self.absolute_boolean_vismap_dict = {}
        self.time_agglomerated_absolute_boolean_vismap = None
        self.num_edge_cells = 1

    def set_times(self, times):
        """
        Set the times on which the simulation should be evaluated

        :param times: List of times in the Simulation.
        :type times: list
        """
        self.times = times

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

    def set_waypoint(self, x, y, c=3, alpha=None):
        """
        Add a waypoint to the list.

        :param x: X coordinate of the waypoint referring to global FDS coordinates.
        :type x: float
        :param y: Y coordinate of the waypoint referring to global FDS coordinates.
        :type y: float
        :param c: Contrast factor for exit sign according to JIN.
        :type c: int, optional
        :param alpha: Orientation of the exit sign according to FDS orientations.
        :type alpha: int or None, optional
        """
        self.way_points_list.append(Waypoint(x, y, c, alpha))

    def _read_fds_data(self, slice_id=0):  #: Todo: specify slice closest to given height
        """
        Read FDS data and store relevant slices and obstructions.

        :param slice_id: Index of FDS slice file to be evaluated.
        :type slice_id: int
        """
        sim = fds.Simulation(self.sim_dir)
        self.slc = sim.slices.filter_by_quantity(self.quantity)[slice_id]
        self.obstructions = sim.obstructions
        self.all_x_coords = self.slc.get_coordinates()['x']
        self.all_y_coords = self.slc.get_coordinates()['y']
        self.grid_shape = (len(self.all_x_coords), (len(self.all_y_coords)))

    def _get_extco_array(self, time):
        """
        Get the array of extinction coefficients from the relevant slice file at a given time.

        :param time: Timestep to evaluate in seconds.
        :type time: float
        :return: Array of extinction coefficients at the specified time.
        :rtype: np.ndarray
        """
        time_index = self.slc.get_nearest_timestep(time)
        data = np.flip(self.slc.to_global()[time_index], axis=1)
        extco_array = data
        return extco_array

    def _get_non_concealed_cells_idx(self, waypoint_id):
        """
        Get the array of mean extinction coefficients between the waypoint and all cells.

        :param waypoint_id: Waypoint ID of the exit sign.
        :type waypoint_id: int
        """

        try:
            x_idx = self.non_concealed_cells_xy_idx_dict[waypoint_id][1]
            y_idx = self.non_concealed_cells_xy_idx_dict[waypoint_id][0]
        except:
            x, y = np.meshgrid(range(self.grid_shape[1]), range(self.grid_shape[0]), indexing='ij')
            x_idx = x.flatten().tolist()
            y_idx = y.flatten().tolist()
        return x_idx, y_idx

    def _get_mean_extco_array(self, waypoint_id, time):
        """
        Get the array of mean extinction coefficients between the waypoint and all non concealed cells.

        :param waypoint_id: Waypoint ID of the exit sign.
        :type waypoint_id: int
        :param time: Timestep to evaluate.
        :type time: float
        :return: Array of mean extinction coefficients.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        extco_array = self._get_extco_array(time)
        ref_x_id = find_closest_point(self.all_x_coords, wp.x)
        ref_y_id = find_closest_point(self.all_y_coords, wp.y)
        mean_extco_array = np.zeros_like(extco_array)

        non_concealed_x_idx, non_concealed_y_idx = self._get_non_concealed_cells_idx(waypoint_id)

        for x_id, y_id in zip(non_concealed_x_idx, non_concealed_y_idx):
            img = np.zeros_like(extco_array)
            x_lp_idx, y_lp_idx = line(ref_x_id, ref_y_id, x_id, y_id)
            img[x_lp_idx, y_lp_idx] = 1
            n_cells = len(x_lp_idx)
            mean_extco = np.sum(extco_array * img) / n_cells
            mean_extco_array[x_id, y_id] = mean_extco
        return mean_extco_array.T

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
        distance_array = np.linalg.norm(np.array([self.xv - wp.x, self.yv - wp.y]), axis=0)

        self.distance_array_list.append(distance_array)
        return distance_array


    def _get_view_angle_array(self, waypoint_id):
        """
        Get the view array considering view angles.

        :param waypoint_id: Waypoint ID.
        :type waypoint_id: int
        :return: Array of cosinus values of view angles
        :rtype: np.ndarray
        """
        distance_array = self._get_dist_array(waypoint_id)
        wp = self._get_waypoint(waypoint_id)
        if self.view_angle == True and wp.alpha != None:
            view_angle_array = (np.sin(np.deg2rad(wp.alpha)) * (self.xv - wp.x) + np.cos(np.deg2rad(wp.alpha)) * (self.yv - wp.y)) / distance_array
        else:
            view_angle_array = np.ones_like(distance_array)
        return view_angle_array

    def build_obstructions_array(self):
        # Initialize arrays for external collisions and cell obstructions
        meshgrid = self._get_extco_array(0)
        obstruction_array = np.zeros_like(meshgrid)

        # Update the obstruction_matrix based on defined obstructions and their height ranges
        for obstruction in self.obstructions:
            for sub_obstruction in obstruction:
                _, x_range, y_range, z_range = sub_obstruction.extent
                if z_range[0] <= self.eval_height <= z_range[1]:
                    x_min_id = (np.abs(self.all_x_coords - x_range[0])).argmin()
                    x_max_id = (np.abs(self.all_x_coords - x_range[1])).argmin()
                    y_min_id = (np.abs(self.all_y_coords - y_range[0])).argmin()
                    y_max_id = (np.abs(self.all_y_coords - y_range[1])).argmin()
                    self.obstruction_array[x_min_id:x_max_id, y_min_id:y_max_id] = True

        # Mirror the obstruction_matrix horizontally
        self.obstruction_array = np.flip(obstruction_array.T, axis=0)

    def _get_non_concealed_cells_array(self, waypoint_id, aa=True):
        """
        Compute the visibility matrix indicating obstructed cells when observing a given waypoint.

        :param waypoint_id: The ID of the waypoint under observation.
        :type waypoint_id: int
        :param evaluation_height: The height at which to evaluate obstructions.
        :type evaluation_height: float
        :param aa: Flag indicating whether to anti-aliasing should be used. Default is True.
        :type aa: bool, optional
        :return: A 2D boolean array; True indicates obstructed visibility from the cell to the waypoint.
        :rtype: np.ndarray
        """
        # Retrieve the coordinates for the target waypoint
        wp = self._get_waypoint(waypoint_id)

        # Find the closest grid coordinates to the target waypoint
        closest_x_id = find_closest_point(self.all_x_coords, wp.x)
        closest_y_id = find_closest_point(self.all_y_coords, wp.y)


        # Initialize arrays for the final visibility matrix, buffer matrix, and edge cell identification
        non_concealed_cells_array = np.zeros_like(self.obstruction_array)
        buffer_array = non_concealed_cells_array.copy()
        edge_cells = np.ones_like(self.obstruction_array)
        edge_cells[self.num_edge_cells:-self.num_edge_cells, self.num_edge_cells:-self.num_edge_cells] = False
        edge_x_idx, edge_y_idx = np.where(edge_cells == True)

        # Choose the appropriate line function based on the aa flag
        line_func = line_aa if aa else line

        # Iterate through edge cells to update visibility based on obstructions
        for x_id, y_id in zip(edge_x_idx, edge_y_idx):
            line_x_idx, line_y_idx = line_func(closest_x_id, closest_y_id, x_id, y_id)[:2]

            buffer_array[line_x_idx, line_y_idx] = True
            obstructed_cells = np.where((self.obstruction_array == True) & (buffer_array == True))

            # If line intersects obstructions, mark only the segment before the first obstruction as visible
            if obstructed_cells[0].size != 0:
                num_non_concealed_cells = _count_cells_to_obstruction(line_x_idx, line_y_idx, obstructed_cells)
                non_concealed_cells_array[
                    line_x_idx[:num_non_concealed_cells], line_y_idx[:num_non_concealed_cells]] = True
            # If the line doesn't intersect any obstructions, mark the entire line as visible
            else:
                non_concealed_cells_array[line_x_idx, line_y_idx] = True

            # Reset the buffer matrix for the next iteration
            buffer_array.fill(0)

        # Add the finalized visibility matrix to the list and return its transpose
        non_concealed_cells_array = non_concealed_cells_array.T
        self.non_concealed_cells_array_list.append(non_concealed_cells_array)
        self.non_concealed_cells_xy_idx_dict[waypoint_id] = np.where(non_concealed_cells_array == True)
        return non_concealed_cells_array

    def _get_vismap(self, waypoint_id, time):
        """
        Get the visibility map for a specific waypoint.

        :param waypoint_id: Waypoint ID.
        :type waypoint_id: int
        :param time: Timestep to evaluate.
        :type time: float
        :return: Visibility map.
        :rtype: np.ndarray
        """
        wp = self._get_waypoint(waypoint_id)
        mean_extco_array = self._get_mean_extco_array(waypoint_id, time)
        vis_array = wp.c / mean_extco_array
        vismap = np.where(vis_array > self.max_vis, self.max_vis, vis_array).astype(float)
        return vismap

    def get_bool_vismap(self, waypoint_id, time, extinction, view_angle, collision, aa):
        """
        Generate a boolean visibility map for a specific waypoint.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :param time: Timestep for which to calculate the visibility map.
        :type time: float
        :param extinction: Flag indicating whether to consider extinction coefficients. Default is True.
        :type extinction: bool, optional
        :param view_angle: Flag indicating whether to consider view angles. Default is True.
        :type view_angle: bool, optional
        :param collision: Flag indicating whether to consider obstructions. Default is True.
        :type collision: bool, optional
        :param aa: Flag indicating whether to anti-aliasing should be used. Default is True.
        :type aa: bool, optional
        :return: Boolean visibility map for the given waypoint.
        :rtype: np.ndarray

        """
        if collision:
            non_concealed_cells_array = self._get_non_concealed_cells_array(waypoint_id, aa)
        else:
            non_concealed_cells_array = 1
        if view_angle:
            view_angle_array = self._get_view_angle_array(waypoint_id)
        else:
            view_angle_array = 1
        if extinction:
            vismap = self._get_vismap(waypoint_id, time)
        else:
            vismap = self.max_vis
        distance_array = self._get_dist_array(waypoint_id)

        vismap_total = view_angle_array * vismap * non_concealed_cells_array
        bool_vismap = np.where(vismap_total >= distance_array, True, False)
        bool_vismap = np.where(vismap_total < self.min_vis, False, bool_vismap)
        return bool_vismap

    def get_abs_bool_vismap(self, time, extinction, view_angle, collision, aa):
        """
        Generate an absolute boolean visibility map for all waypoints.

        :param time: Timestep for which to calculate the visibility map.
        :type time: float
        :param extinction: Flag indicating whether to consider extinction coefficients
        :type extinction: bool, optional
        :param view_angle: Flag indicating whether to consider view angles.
        :type view_angle: bool, optional
        :param collision: Flag indicating whether to consider collision.
        :type collision: bool, optional
        :param aa: Flag indicating whether to anti-aliasing should be used. Default is True.
        :type aa: bool, optional
        :return: Absolute boolean visibility map.
        :rtype: np.ndarray
        """
        boolean_vismap_list = []
        for waypoint_id, waypoint in enumerate(self.way_points_list):
            boolean_vismap = self.get_bool_vismap(waypoint_id, time, extinction, view_angle, collision, aa)
            boolean_vismap_list.append(boolean_vismap)
            absolute_boolean_vismap = np.logical_or.reduce(boolean_vismap_list)
        return absolute_boolean_vismap

    def get_time_aggl_abs_bool_vismap(self):
        """
        Calculate the time-agglomerated absolute boolean visibility map.

        :return: Time-agglomerated absolute boolean visibility map.
        :rtype: np.ndarray
        """
        return self.time_agglomerated_absolute_boolean_vismap

    def get_aset_map(self, max_time=None):
        """
        Generate a map indicating the earliest time at which each point becomes non-visible.

        :param max_time: The maximum time to consider. If not specified, the last time in `self.times` is used. :type
        max_time: int, optional :return: A 2D array where each cell represents the earliest time of non-visibility
        for the corresponding point. Cells for points that never become non-visible are set to `max_time`.
        :rtype: np.ndarray
        """
        if not max_time:
            max_time = self.times[-1]
        aset_map = np.full((self.grid_shape[1], self.grid_shape[0]), max_time, dtype=int)
        for time, abs_bool_vismap in zip(self.times, self.absolute_boolean_vismap_dict.values()):
            mask = (abs_bool_vismap == False) & (aset_map == max_time)
            aset_map[mask] = time
        return aset_map

    def plot_abs_bool_vismap(self):  # Todo: is duplicate of plot_time_agglomerated_absolute_boolean_vismap
        """
        Plot the absolute boolean visibility map.
        """

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

    def compute_all(self, extinction=True, view_angle=True, collision=True, aa=True):
        self._read_fds_data()
        self.build_obstructions_array()
        for time in self.times:
            absolute_boolean_vismap = self.get_abs_bool_vismap(time, extinction, view_angle, collision, aa)
            self.absolute_boolean_vismap_dict[time] = absolute_boolean_vismap
        self.time_agglomerated_absolute_boolean_vismap = np.logical_and.reduce(
            list(self.absolute_boolean_vismap_dict.values()))



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

        plt.plot((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen', linestyle='--')
        plt.scatter((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen')
        for wp_id, wp in enumerate(self.way_points_list):
            plt.annotate(f"WP : {wp_id:>}\nC : {wp.c:>}\n$\\alpha$ : {wp.alpha}$^\circ$", xy=(wp.x - 0.2, wp.y + 1.5),
                         bbox=dict(boxstyle="round", fc="w"), fontsize=6)
        plt.xlabel("X / m")
        plt.ylabel("Y / m")


def _count_cells_to_obstruction(line_x, line_y, obstruction):
    """
     Calculate the number of cells until the line intersects with an obstruction.

     :param line_x: 1D array of x coordinates of the line.
     :type line_x: np.ndarray
     :param line_y: 1D array of y coordinates of the line.
     :type line_y: np.ndarray
     :param obstruction: 2D array representing the obstruction. Shape (n, 2) where n is the number of obstruction cells, each row containing [x, y] coordinates.
     :type obstruction: np.ndarray
     :return: The number of cells until the line intersects with the obstruction, or -1 if there's no intersection.
     :rtype: int
     """

    line_x = np.array(line_x)
    line_y = np.array(line_y)
    obstruction = np.array(obstruction).T

    # Create a 2D array from line_x and line_y
    line = np.stack((line_x, line_y), axis=1)

    # Use broadcasting to find differences
    diff = line[:, np.newaxis, :] - obstruction[np.newaxis, :, :]

    # Check for any zero differences (indicating a hit)
    zero_rows = np.all(diff == 0, axis=2)

    # Find if any row in `zero_rows` contains a True (indicating that the line hits the obstruction at that point)
    hits = np.any(zero_rows, axis=1)

    # Find the first occurrence of True in `hits`
    hit_indices = np.where(hits)[0]
    return hit_indices[0] if hit_indices.size > 0 else -1
