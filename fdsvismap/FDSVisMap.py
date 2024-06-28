import fdsreader as fds
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from skimage.draw import line, line_aa

from fdsvismap.Waypoint import Waypoint
from fdsvismap.helper_functions import get_id_of_closest_value, count_cells_to_obstruction


class VisMap:
    """
    A class to build visibility maps (VisMap) based on Fire Dynamics Simulator (FDS) data.

    :ivar obstructions_array: Array indicating obstructed cells in the FDS simulation. Initialized as None.
    :vartype obstruction_array: np.ndarray or None
    :ivar fds_grid_shape: Shape of the FDS grid. Initialized as None.
    :vartype fds_grid_shape: tuple[int, int] or None
    :ivar all_y_coords: y-coordinates of the FDS grid. Initialized as None.
    :vartype all_y_coords: np.ndarray or None
    :ivar all_x_coords: x-coordinates of the FDS grid. Initialized as None.
    :vartype all_x_coords: np.ndarray or None
    :ivar obstructions_collection: Collection of obstruction data from FDS simulation. Initialized as None.
    :vartype obstructions_collection: list or None # TODO: check
    :ivar vismap_time_points: Time points for which the visibility maps are created. Initialized as None.
    :vartype vismap_time_points: np.ndarray or None
    :ivar fds_time_points: Time points available in the FDS simulation data. Initialized as None.
    :vartype fds_time_points: np.ndarray or None
    :ivar quantity: Quantity of FDS slice file to be evaluated, specific to the type of data being visualized. Initialized to 'ext_coef_C0.9H0.1'.
    :vartype quantity: str
    :ivar slc: Slice object for visibility calculations. Initialized as None.
    :vartype slc: fds.Simulation.Slice or None
    :ivar start_point: The starting point coordinates (x, y) for the route of egress. Initialized as None.
    :vartype start_point: tuple[float, float] or None
    :ivar all_wp_dict: Dictionary of waypoints for the path. Initialized as an empty list.
    :vartype all_wp_dict: dict[Waypoint]
    :ivar all_wp_distance_array_dict: Dictionary of distance arrays between each waypoint and all cells. Initialized as an empty list.
    :vartype all_wp_distance_array_list: dicts[np.ndarray]
    :ivar all_wp_non_concealed_cells_array_dict: Dictionary of arrays indicating non-concealed cells for each waypoint. Initialized as an empty list.
    :vartype all_wp_non_concealed_cells_array_dict: dict[np.ndarray]
    :ivar all_wp_angle_array_dict: Dictionary of arrays representing the cosine of the angle of view for each waypoint. Initialized as an empty list.
    :vartype all_wp_angle_array_dict: dict[np.ndarray]
    :ivar all_time_all_wp_vismap_array_list: List of visibility maps for all waypoints at all times. Initialized as an empty list.
    :vartype all_time_all_wp_vismap_array_list: list[list[np.ndarray]]
    :ivar all_wp_non_concealed_cells_xy_idx_dict: Dictionary of indices of non-concealed cells for each waypoint. Initialized as an empty list.
    :vartype all_wp_non_concealed_cells_xy_idx_dict: dict[tuple[np.ndarray, np.ndarray]]
    :ivar min_vis: Minimum local visibility threshold to meet performance criteria. Initialized to 0.
    :vartype min_vis: float
    :ivar max_vis: Maximum visibility threshold. Initialized to 30.
    :vartype max_vis: float
    :ivar fds_slc_height: Height at which the FDS slice is evaluated. Initialized as None.
    :vartype fds_slc_height: float or None
    :ivar background_image: Background image for the plot. Initialized as None.
    :vartype background_image: ndarray or None # TODO: Type?
    :ivar all_time_wp_agg_vismap_list: List of waypoint-aggregated visibility maps for all time steps. Initialized as an empty list.
    :vartype all_time_wp_agg_vismap_list: list[np.ndarray]
    :ivar time_agg_wp_agg_vismap: Time-aggregated, waypoint aggregated visibility map for all waypoints and time points. Initialized as None.
    :vartype time_agg_wp_agg_vismap: np.ndarray or None
    :ivar num_edge_cells: Number of edge cells considered for collision detection. Initialized to 1.
    :vartype num_edge_cells: int
    """

    def __init__(self):
        """
        Initialize the VisMap object.

        """
        self.obstructions_array = None
        self.fds_grid_shape = None
        self.all_y_coords = None
        self.all_x_coords = None
        self.obstructions_collection = None
        self.vismap_time_points = None
        self.fds_time_points = None
        self.quantity = 'ext_coef_C0.9H0.1'
        self.slc = None
        self.start_point = None
        self.all_wp_dict = {}
        self.all_wp_distance_array_dict = {}
        self.all_wp_non_concealed_cells_array_dict = {}
        self.all_wp_angle_array_dict = {}
        self.all_time_all_wp_vismap_array_list = []
        self.all_wp_non_concealed_cells_xy_idx_dict = {}
        self.min_vis = 0
        self.max_vis = 30
        self.fds_slc_height = None
        self.background_image = None
        self.all_time_wp_agg_vismap_list = []
        self.time_agg_wp_agg_vismap = None
        self.num_edge_cells = 1
        self.cell_size = None
        self.extent = None


    def set_time_points(self, time_points):
        """
        Set the times on which the simulation should be evaluated

        :param time_points: List of time points in the simulation.
        :type time_points: list
        """
        self.vismap_time_points = np.array(time_points)

    def set_visibility_bounds(self, min_vis, max_vis):
        """
        Set a lower and upper bound for visibility as a performance criterion. The lower bound is considered as a local minimum value.
        :param min_vis: float
        :type min_vis: Lower limit for local visibility to meet the performance criterion
        :param max_vis: float
        :type max_vis: Upper limit for local visibility to be considered
        """
        self.min_vis = min_vis
        self.max_vis = max_vis

    def set_start_point(self, x, y):
        """
        Set the starting point for the route of egress.

        :param x: x-coordinate of the starting point referring to global FDS coordinates.
        :type x: float
        :param y: y-coordinate of the starting point referring to global FDS coordinates.
        :type y: float
        """
        self.start_point = (x, y)

    def set_waypoint(self, waypoint_id, x, y, c=3, alpha=None):
        """
        Add a waypoint along the route of egress.
        :param waypoint_id: ID of the waypoint to add to the route.
        :type waypoint_id: int
        :param x: x-coordinate of the waypoint referring to global FDS coordinates.
        :type x: float
        :param y: y-coordinate of the waypoint referring to global FDS coordinates.
        :type y: float
        :param c: Contrast factor for exit sign according to Jin.
        :type c: int, optional
        :param alpha: Orientation angle of the exit sign according to global FDS coordinates.
        :type alpha: int or None, optional
        """
        self.all_wp_dict[waypoint_id] = Waypoint(x, y, c, alpha)

    def read_fds_data(self, sim_dir, slice_id=0, fds_slc_height=2):  #: Todo: specify slice closest to given height
        """
        Read FDS data and store relevant coordinates, shape of the meshgrid, slices and obstructions.

        :param sim_dir: Directory where FDS simulation data is stored
        :type sim_dir: object
        :param slice_id: Index of FDS slice file to be evaluated. Default is 0.
        :type slice_id: int
        :param fds_slc_height: The height at which to evaluate visibility. Default is 2.
        :type fds_slc_height: float, optional
        """
        sim = fds.Simulation(sim_dir)
        self.slc = sim.slices.filter_by_quantity(self.quantity)[slice_id]
        self.extent = np.array(self.slc.extent._extents)
        self.all_x_coords = self.slc.get_coordinates()['x']
        self.all_y_coords = self.slc.get_coordinates()['y']
        self.fds_grid_shape = (len(self.all_x_coords), (len(self.all_y_coords)))
        self.cell_size = ((self.extent[0,1]-self.extent[0,0])/self.fds_grid_shape[0], (self.extent[1,1]-self.extent[1,0])/self.fds_grid_shape[1])
        self.fds_time_points = self.slc.times
        self.obstructions_collection = sim.obstructions
        self.fds_slc_height = fds_slc_height
        self.build_obstructions_array()

    def _get_extco_array_at_time(self, time):
        """
        Get the array of extinction coefficients from the relevant slice file closest to the given time.

        :param time: Time point to be evaluated in seconds.
        :type time: float
        :return: Array of extinction coefficients at the specified time.
        :rtype: np.ndarray
        """
        time_index = self.slc.get_nearest_timestep(time)
        extco_array = self.slc.to_global()[time_index]
        return extco_array

    def _get_non_concealed_cells_idx(self, waypoint_id):
        """
        Retrieve the X and Y indices of non-concealed cells for a specific waypoint.

        :param waypoint_id: Index of the waypoint for which to retrieve non-concealed cell indices.
        :type waypoint_id: int
        :return: Tuple of arrays (x_indices, y_indices) representing the X and Y indices of non-concealed cells.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        x_idx = self.all_wp_non_concealed_cells_xy_idx_dict[waypoint_id][1]
        y_idx = self.all_wp_non_concealed_cells_xy_idx_dict[waypoint_id][0]
        return x_idx, y_idx

    def _get_mean_extco_array_at_time(self, waypoint_id, time):
        """
        Get the array of mean extinction coefficients between the waypoint and all non-concealed cells.

        :param waypoint_id: Index of the waypoint for which to calculate the mean extinction coefficients.
        :type waypoint_id: int
        :param time: Time at which the extinction coefficients should be calculated.
        :type time: float
        :return: A 2D numpy array with the mean extinction coefficients at the specified time, transposed for correct orientation.
        :rtype: np.ndarray
        """
        wp = self.all_wp_dict[waypoint_id]
        extco_array = self._get_extco_array_at_time(time)
        ref_x_id = get_id_of_closest_value(self.all_x_coords, wp.x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, wp.y)
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

        :param waypoint_id: The index of the waypoint from which distances are to be calculated.
        :type waypoint_id: int
        :return: A 2D numpy array where each element represents the distance from the specified waypoint to that cell.
        :rtype: np.ndarray
        """
        wp = self.all_wp_dict[waypoint_id]
        self.xv, self.yv = np.meshgrid(self.all_x_coords, self.all_y_coords)
        distance_array = np.linalg.norm(np.array([self.xv - wp.x, self.yv - wp.y]), axis=0)
        return distance_array

    def _get_view_angle_array(self, waypoint_id):
        """
        Get the view array considering view angles.

        :param waypoint_id: The index of the waypoint for which view angles are to be calculated.
        :type waypoint_id: int
        :return: A 2D numpy array with the cosine values of the view angles from the waypoint to each cell.
        :rtype: np.ndarray
        """
        distance_array = self._get_dist_array(waypoint_id)
        wp = self.all_wp_dict[waypoint_id]
        if wp.alpha is not None:
            view_angle_array = np.clip((np.sin(np.deg2rad(wp.alpha)) * (self.xv - wp.x) + np.cos(np.deg2rad(wp.alpha))
                                        * (self.yv - wp.y)) / distance_array, 0, 1)
        else:
            view_angle_array = np.ones_like(distance_array)
        return view_angle_array

    def build_obstructions_array(self):
        """
        Construct an obstruction array based on FDS simulation data. Marks cells in the grid as obstructed based on the
        obstruction objects defined within the FDS simulation. It takes into account the height of the slice
        (fds_slc_height) to determine if an obstruction at a given location blocks visibility.
        """
        # Initialize arrays for external collisions and cell obstructions
        meshgrid = self._get_extco_array_at_time(0)
        obstruction_array = np.zeros_like(meshgrid).T

        # Update the obstruction_matrix based on defined obstructions and their height ranges
        for obstruction in self.obstructions_collection:
            for sub_obstruction in obstruction:
                _, x_range, y_range, z_range = sub_obstruction.extent
                if z_range[0] <= self.fds_slc_height <= z_range[1]:
                    obstruction_array = self._add_visual_object(x_range[0], x_range[1], y_range[0], y_range[1], obstruction_array, True)
        self.obstructions_array = obstruction_array

    def build_help_arrays(self, obstructions, view_angle, aa):
        """
        Construct auxiliary arrays used for the comprehensive creation of visibility maps.

        Note: If 'collision' is False, all cells are considered as non-concealed; if 'view_angle' is False, the angle is not factored into visibility calculations.

        :param obstructions: Flag indicating whether to consider cells being concealed by obstructions.
        :type obstructions: bool
        :param view_angle: Flag indicating whether to consider view angles from each waypoint.
        :type view_angle: bool
        :param aa: Flag indicating whether antialiasing should be used in the calculation of line-of-sight paths, affecting the smoothness of boundaries.
        :type aa: bool, optional
        """
        for waypoint_id in self.all_wp_dict.keys():
            if obstructions:
                non_concealed_cells_array = self._get_non_concealed_cells_array(waypoint_id, aa)
                self.all_wp_non_concealed_cells_array_dict[waypoint_id] = non_concealed_cells_array
                self.all_wp_non_concealed_cells_xy_idx_dict[waypoint_id] = np.where(non_concealed_cells_array == True)
            else:
                self.all_wp_non_concealed_cells_array_dict[waypoint_id] = 1
            if view_angle:
                self.all_wp_angle_array_dict[waypoint_id] = self._get_view_angle_array(waypoint_id)
            else:
                self.all_wp_angle_array_dict[waypoint_id] = 1

            self.all_wp_distance_array_dict[waypoint_id] = self._get_dist_array(waypoint_id)

    def _get_non_concealed_cells_array(self, waypoint_id, aa=True):
        """
        Compute the non_concealed_cells array indicating obstructed cells relative to a certain waypoint.

        :param waypoint_id: The index of the waypoint from where concealed and unconcealed cells are determined.
        :type waypoint_id: int
        :param aa: Flag indicating whether antialiasing should be used in the line drawing process. Antialiasing can improve
                   the visual quality of the line by smoothing jagged edges but may affect performance. Default is True.
        :type aa: bool, optional
        :return: A 2D boolean array where True indicates that the cell is visible (non-obstructed) from the waypoint.
        :rtype: np.ndarray
        """
        # Retrieve the coordinates for the target waypoint
        wp = self.all_wp_dict[waypoint_id]

        # Find the closest grid coordinates to the target waypoint
        closest_y_id = get_id_of_closest_value(self.all_x_coords, wp.x)  # TODO: fix x / y coordinates switch
        closest_x_id = get_id_of_closest_value(self.all_y_coords, wp.y)

        # Initialize arrays for the final visibility matrix, buffer matrix, and edge cell identification
        non_concealed_cells_array = np.zeros_like(self.obstructions_array)
        buffer_array = non_concealed_cells_array.copy()
        edge_cells = np.ones_like(self.obstructions_array)
        edge_cells[self.num_edge_cells:-self.num_edge_cells, self.num_edge_cells:-self.num_edge_cells] = False
        edge_x_idx, edge_y_idx = np.where(edge_cells == True)

        # Choose the appropriate line function based on the aa flag
        line_func = line_aa if aa else line

        # Iterate through edge cells to update visibility based on obstructions
        for x_id, y_id in zip(edge_x_idx, edge_y_idx):
            line_x_idx, line_y_idx = line_func(closest_x_id, closest_y_id, x_id, y_id)[:2]

            buffer_array[line_x_idx, line_y_idx] = True
            obstructed_cells = np.where((self.obstructions_array == True) & (buffer_array == True))

            # If line intersects obstructions, mark only the segment before the first obstruction as visible
            if obstructed_cells[0].size != 0:
                num_non_concealed_cells = count_cells_to_obstruction(line_x_idx, line_y_idx, obstructed_cells)
                non_concealed_cells_array[
                    line_x_idx[:num_non_concealed_cells], line_y_idx[:num_non_concealed_cells]] = True
            # If the line doesn't intersect any obstructions, mark the entire line as visible
            else:
                non_concealed_cells_array[line_x_idx, line_y_idx] = True

            # Reset the buffer matrix for the next iteration
            buffer_array.fill(0)
        # # Add the finalized visibility matrix to the list and return its transpose
        # non_concealed_cells_array = non_concealed_cells_array.T
        return non_concealed_cells_array

    def _get_visibility_array(self, waypoint_id, time):
        """
        Calculate the visibility array for a specific waypoint at a given time.

        :param waypoint_id: The index of the waypoint for which the visibility map is to be calculated.
        :type waypoint_id: int
        :param time: The simulation time at which to evaluate visibility.
        :type time: float
        :return: A 2D numpy array representing the visibility (m) from the waypoint along the line of sight relative to each cell.
        :rtype: np.ndarray
        """
        wp = self.all_wp_dict[waypoint_id]
        mean_extco_array = self._get_mean_extco_array_at_time(waypoint_id, time)
        vis_array = np.divide(wp.c, mean_extco_array, out=np.full_like(mean_extco_array, self.max_vis),
                              where=mean_extco_array != 0)
        vismap = np.where(vis_array > self.max_vis, self.max_vis, vis_array).astype(float)
        return vismap

    def get_vismap(self, waypoint_id, time):
        """
        Generate a boolean  vismap for a specific waypoint at a given time.

        :param waypoint_id: The index of the waypoint for which the visibility map is to be calculated.
        :type waypoint_id: int
        :param time: The simulation time at which to evaluate visibility.
        :type time: float
        :return: Boolean vismap indicating whether the waypoint can be seen (True) from a specific  cell or not (False).
        :rtype: np.ndarray

        """
        non_concealed_cells_array = self.all_wp_non_concealed_cells_array_dict[waypoint_id]
        view_angle_array = self.all_wp_angle_array_dict[waypoint_id]
        visibility_array = self._get_visibility_array(waypoint_id, time)
        distance_array = self._get_dist_array(waypoint_id)

        visibility_array_total = view_angle_array * visibility_array * non_concealed_cells_array
        vismap = np.where(visibility_array_total >= distance_array, True, False)
        vismap = np.where(visibility_array_total < self.min_vis, False, vismap)
        return vismap

    def get_wp_agg_vismap(self, time):
        """
        Get a waypoint aggregated bool type visibility map for a specific point in time.

        :param time: Timestep for which to calculate the visibility map.
        :type time: float
        :return: Waypoint aggregated bool type visibility map.
        :rtype: np.ndarray
        """
        time_id = get_id_of_closest_value(self.vismap_time_points, time)
        return self.all_time_wp_agg_vismap_list[time_id]

    def get_time_agg_wp_agg_vismap(self):
        """
        Get a time-aggregated and waypoint-aggregated boolean visibility map.

        :return: Time-aggregated and waypoint-aggregated boolean visibility map.
        :rtype: np.ndarray
        """
        return self.time_agg_wp_agg_vismap

    def get_aset_map(self, max_time=None):
        """
        Generate a map indicating the earliest time at which each point becomes non-visible.

        :param max_time: The maximum time to consider. If not specified, the last time in `self.times` is used.
        :type max_time: int, optional
        :return: A 2D array where each cell represents the earliest time of non-visibility
        for the corresponding point. Cells for points that never become non-visible are set to `max_time`.
        :rtype: np.ndarray
        """
        if not max_time:
            max_time = self.vismap_time_points[-1]
        aset_map = np.full((self.fds_grid_shape[1], self.fds_grid_shape[0]), max_time, dtype=int)
        for time, wp_agg_vismap in zip(self.vismap_time_points, self.all_time_wp_agg_vismap_list):
            mask = (wp_agg_vismap == False) & (aset_map == max_time)
            aset_map[mask] = time
        return aset_map

    def _create_map_plot(self, map_array, cmap, plot_obstructions, flip_y_axis,  **cbar_kwargs):
        """
        Create a labeled matplotlib plot of a given map array using a specified colormap and colorbar settings.

        :param map_array: A 2D numpy array representing the data to be plotted.
        :type map_array: np.ndarray
        :param cmap: Colormap used for visualizing the data.
        :type cmap: str or matplotlib.colors.Colormap
        :param cbar_kwargs: Keyword arguments for configuring the colorbar (e.g., label, orientation).
                            These are passed directly to `fig.colorbar()`.
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis:  bool, optional
        :type cbar_kwargs: dict
        :return: A tuple containing the matplotlib figure and axes objects.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)

        """
        origin = "lower" if flip_y_axis else "upper"
        if flip_y_axis:
            extent = (self.all_x_coords[0], self.all_x_coords[-1], self.all_y_coords[0], self.all_y_coords[-1])
        else:
            extent = (self.all_x_coords[0], self.all_x_coords[-1], self.all_y_coords[-1], self.all_y_coords[0])
        fig, ax = plt.subplots()
        if self.background_image is not None:
            ax.imshow(self.background_image, extent=extent, origin=origin)
        if plot_obstructions:
            ax.imshow(self.obstructions_array, extent=extent, cmap='Grays', alpha=0.5, origin=origin)
        im = ax.imshow(map_array, cmap=cmap, alpha=0.7, extent=extent, origin=origin)

        fig.colorbar(mappable=im, ax=ax, orientation='horizontal', pad=0.15, **cbar_kwargs)
        ax.set_xlabel("$X$ / m")
        ax.set_ylabel("$Y$ / m")
        return fig, ax

    def create_aset_map_plot(self, max_time=None, plot_obstructions=False, flip_y_axis=True):
        """
        Create a plot visualizing the ASET map (Available Safe Egress Time) map indicating for each cell the first time any waypoint is not visible.

        :param max_time: The maximum time value to consider for the ASET calculations. If None, it defaults to the last time in the visibility data.
        :type max_time: int, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis:  bool, Default is True.
        :return: A tuple containing the matplotlib figure and axes objects that display the ASET map.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
        """
        aset_map_array = self.get_aset_map(max_time)
        cbar_kwargs = {'label': 'Time / s'}
        fig, ax = self._create_map_plot(map_array=aset_map_array, cmap='jet_r', plot_obstructions=plot_obstructions,
                                        flip_y_axis=flip_y_axis, **cbar_kwargs)
        return fig, ax

    def create_time_agg_wp_agg_vismap_plot(self, plot_obstructions=False, flip_y_axis=True):
        """
        Create a plot visualizing the time-aggregated visibility map for all waypoints. The map uses a custom color
        map to distinguish whether any waypoint is visible (green) or not (red) from each cell. The plot also
        features the trajectory of movement from the start point through all waypoints, highlighted with annotations
        for each waypoint.

        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis:  bool, Default is True.
        :return: A tuple containing the matplotlib figure and axes objects that display the aggregated visibility map.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
        """
        cmap = matplotlib.colors.ListedColormap(['red', 'lime'])
        cbar_kwargs = {'label': None, 'ticks': [0, 1], 'format': mticker.FixedFormatter(['not visible', 'visible'])}
        fig, ax = self._create_map_plot(map_array=self.time_agg_wp_agg_vismap, cmap=cmap, plot_obstructions=plot_obstructions,
                                        flip_y_axis=flip_y_axis, **cbar_kwargs)
        x_values = [wp.x for wp in self.all_wp_dict.values()]
        y_values = [wp.y for wp in self.all_wp_dict.values()]

        ax.plot((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen', linestyle='--')
        ax.scatter((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen')
        for wp_id, wp in self.all_wp_dict.items():
            ax.annotate(f"$W_{{{wp_id}}}$\nC : {wp.c:>}\n$\\alpha$ : {wp.alpha}$^\\circ$", xy=(wp.x - 1 , wp.y - 2),
                            bbox=dict(boxstyle="round", fc="w"), fontsize=6)

        return fig, ax

    def add_background_image(self, file):
        """
        Load and set a background image for future plots created within this visualization class.

        :param file: Path to the image file that will be used as the background.
        :type file: str
        """
        self.background_image = np.flip(plt.imread(file), axis=0)

    def compute_all(self, view_angle=True, obstructions=True, aa=True):
        """
        Execute all required computations to generate aggregated visibility maps over all waypoints and time points.

        :param view_angle: Determines if view angles should be considered in the visibility calculations,
                          affecting how visibility is computed relative to the waypoint orientations. Default is True.
        :type view_angle: bool
        :param obstructions: Determines if collisions (obstructions) should be considered, impacting whether
                         certain paths are considered visible based on physical barriers. Default is True.
        :type obstructions: bool
        :param aa: Determines if antialiasing should be applied when computing visibility lines, which can
                  smooth the appearance of the visibility boundaries but might affect computational performance. Default is True.
        :type aa: bool
        """
        self.build_help_arrays(view_angle=view_angle, obstructions=obstructions, aa=aa)
        for time in self.vismap_time_points:
            print(f"Simulation time {time} s of {self.vismap_time_points[-1]} s")
            all_wp_vismap_array_list = []
            for waypoint_id in self.all_wp_dict.keys():
                print(f"Waypoint {waypoint_id}", end=" ")
                vismap = self.get_vismap(waypoint_id, time)
                all_wp_vismap_array_list.append(vismap)
            self.all_time_all_wp_vismap_array_list.append(all_wp_vismap_array_list)
            wp_agg_vismap = np.logical_or.reduce(all_wp_vismap_array_list)
            self.all_time_wp_agg_vismap_list.append(wp_agg_vismap)
            print("")
        self.time_agg_wp_agg_vismap = np.logical_and.reduce(self.all_time_wp_agg_vismap_list)

    def get_local_visibility(self, time, x, y, c):
        """
        Calculate the local visibility at a specific cell closest to the given x, y coordinates at a certain time
        based on local extinction coefficient values.

        :param time: The simulation time at which to calculate the visibility.
        :type time: float
        :param x: The x-coordinate in the simulation grid where visibility is to be calculated.
        :type x: float
        :param y: The y-coordinate in the simulation grid where visibility is to be calculated.
        :type y: float
        :param c: Contrast factor for exit sign according to Jin
        :type c: float
        :return: The computed local visibility value at the given location and time.
        :rtype: float
        """
        ref_x_id = get_id_of_closest_value(self.all_x_coords, x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, y)
        extco_array = self._get_extco_array_at_time(time)
        local_extco = extco_array[ref_x_id, ref_y_id]  # TODO: Why are coordinates switched for extco array?
        if local_extco == 0:
            return self.max_vis
        else:
            local_visibility = c / local_extco
            return min(local_visibility, self.max_vis)

    def wp_is_visible(self, time, x, y, waypoint_id):
        """
        Determine if a waypoint is visible from a specific cell closest to the given x, y coordinates at a certain time.

        :param time: The simulation time for which visibility is checked.
        :type time: float
        :param x: The x-coordinate of the location from which visibility is checked.
        :type x: float
        :param y: The y-coordinate of the location from which visibility is checked.
        :type y: float
        :param waypoint_id: The ID of the waypoint to check visibility for.
        :type waypoint_id: int
        :return: A boolean value indicating whether the specified waypoint is visible from the given location and time.
        :rtype: bool
        """
        time_id = get_id_of_closest_value(self.vismap_time_points, time)
        ref_x_id = get_id_of_closest_value(self.all_x_coords, x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, y)
        vismap_array = self.all_time_all_wp_vismap_array_list[time_id][waypoint_id]
        is_visible = vismap_array[ref_y_id, ref_x_id]
        return is_visible

    def get_distance_to_wp(self, x, y, waypoint_id):
        """
        Calculate the distance from a specific cell closest to the given x, y coordinates to a designated waypoint.

        :param x: The x-coordinate of the location from which to measure distance.
        :type x: float
        :param y: The y-coordinate of the location from which to measure distance.
        :type y: float
        :param waypoint_id: The ID of the waypoint to which distance is measured.
        :type waypoint_id: int
        :return: The distance to the waypoint from the specified location.
        :rtype: float

        The distance value is useful for navigation, proximity checks, or any scenario where the physical distance
        to a waypoint is relevant for decision-making or analysis.
        """
        wp = self.all_wp_dict[waypoint_id]
        distance_to_wp = np.linalg.norm(np.array([x - wp.x, y - wp.y]), axis=0)
        return distance_to_wp

    def _add_visual_object(self, x1, x2, y1, y2, obstructions_array, status):
        """
        Add or remove obstructions from a specified rectangular area within the simulation grid. This is valid for
        everything affected by the ray tracing algorithms.

        :param x1: The x-coordinate of the first corner of the rectangle.
        :type x1: float
        :param x2: The x-coordinate of the opposite corner of the rectangle.
        :type x2: float
        :param y1: The y-coordinate of the first corner of the rectangle.
        :type y1: float
        :param y2: The y-coordinate of the opposite corner of the rectangle.
        :type y2: float
        :param obstructions_array: The array representing obstructions in the simulation area.
        :type obstructions_array: np.ndarray
        :param status: The boolean status to apply within the specified rectangle (True for obstructed, False for clear).
        :type status: bool
        :return: The modified obstructions array with the newly added or removed object.
        :rtype: np.ndarray

        """
        ref_x1_id = get_id_of_closest_value(self.all_x_coords, x1 + self.cell_size[0]/2)
        ref_x2_id = get_id_of_closest_value(self.all_x_coords, x2 - self.cell_size[0]/2) + 1
        ref_y1_id = get_id_of_closest_value(self.all_y_coords, y1 + self.cell_size[0]/2)
        ref_y2_id = get_id_of_closest_value(self.all_y_coords, y2 - self.cell_size[0]/2) + 1
        obstructions_array[ref_y1_id:ref_y2_id, ref_x1_id:ref_x2_id] = status
        return obstructions_array

    def add_visual_hole(self, x1, x2, y1, y2):
        """
        Remove obstructions from a specified rectangular area within the simulation grid. This is valid for
        everything affected by the ray tracing algorithms.

        :param x1: The x-coordinate of the first corner of the rectangle.
        :type x1: float
        :param x2: The x-coordinate of the opposite corner of the rectangle.
        :type x2: float
        :param y1: The y-coordinate of the first corner of the rectangle.
        :type y1: float
        :param y2: The y-coordinate of the opposite corner of the rectangle.
        :type y2: float

        """
        self._add_visual_object(x1, x2, y1, y2, self.obstructions_array, False)


    def add_visual_obstruction(self, x1, x2, y1, y2):
        """
        Add obstructions from a specified rectangular area to the simulation grid. This is valid for
        everything affected by the ray tracing algorithms.

        :param x1: The x-coordinate of the first corner of the rectangle.
        :type x1: float
        :param x2: The x-coordinate of the opposite corner of the rectangle.
        :type x2: float
        :param y1: The y-coordinate of the first corner of the rectangle.
        :type y1: float
        :param y2: The y-coordinate of the opposite corner of the rectangle.
        :type y2: float

        """
        self._add_visual_object(x1, x2, y1, y2, self.obstructions_array, True)
