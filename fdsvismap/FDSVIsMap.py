import numpy as np
from skimage.draw import line, line_aa
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
    :vartype way_points_list: list[Waypoint]
    :ivar mean_extco_array_list: List of mean extinction coefficient arrays. Initialized as an empty list.
    :vartype mean_extco_array_list: list
    :ivar view_array_list: List of view arrays. Initialized as an empty list.
    :vartype view_array_list: list
    :ivar distance_array_list: List of distance arrays. Initialized as an empty list.
    :vartype distance_array_list: list
    :ivar vis_map_array: The array representing the visibility map. Initialized as None.
    :vartype vis_map_array: np.ndarray
    :param ivar: Quantity of FDS slice file to be evaluated.
    :type vartype: str
    """

    def __init__(self, sim_dir, min_vis=0, max_vis=30, eval_height=2, debug=False):
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
        :debug: bool, optional. A flag that enables or disables debugging messages.
                        Default is False (debugging messages disabled).
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
        self.debug = debug

    def debug_message(self, message, output=None):
        """
        Prints debugging information. Useful for identifying issues related to the fdsReader version change.

        Parameters:
        - message (str): A message to be printed, typically naming the output being debugged.
        - output (optional): The variable or object to be examined. It could be of any type, but special
                             information is printed for lists, tuples, and objects with a 'shape' attribute.

        If debugging is enabled, this function will:
        - Always print the message.
        - Print additional information about the 'output' parameter, if provided. The additional information
          varies depending on the type of 'output':
          - If 'output' has a 'shape' attribute, its shape is printed.
          - If 'output' is a list or tuple, its length is printed.
          - Otherwise, 'output' is printed directly.
          (helped to find fdsreader syntax "switch")
        """

        if self.debug:  # Check if debugging is enabled
            print(f"DEBUG - {message}")  # Always print the message

            if output is not None:  # If an output is provided, print additional information
                if hasattr(output, 'shape'):  # Check for 'shape' attribute, typically present in numpy arrays
                    print(f"Output Shape: {output.shape}")  # Print the shape of 'output'
                elif isinstance(output, (list, tuple)):  # Check if 'output' is a list or tuple
                    print(f"Output Length: {len(output)}")  # Print the length of 'output'
                else:  # For other types of 'output', print them directly
                    print(f"Output: {output}")

    def set_times(self, times):
        """
        Set the times on which the simulation should be evaluated

        :param times: List of times in the Simulation.
        :type times: list
        """

        self.times = times
        self.debug_message("set_times:", self.times)


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

    #: Todo: specify slice closest to given height, done:171023, slice_id, can be deleted?
    def _read_fds_data(self, slice_id=0):
        """
        Read FDS data and store relevant slices and obstructions.

        Parameters:
        - slice_id (int): Index of FDS slice file to be evaluated. Default is 0.
        """

        # Load the simulation data
        sim = fds.Simulation(self.sim_dir)

        # Get the relevant slice and obstructions
        self.slc = sim.slices.filter_by_quantity(self.quantity).get_nearest(z=self.eval_height)
        self.slc_alldata, self.coordinates = self.slc.to_global(masked=False, return_coordinates=True)
        self.obstructions = sim.obstructions

        # Store x and y coordinates
        self.all_x_coords = self.coordinates["x"]
        self.all_y_coords = self.coordinates["y"]

        # Verify that the slice is at the expected height
        slc_z_height = self.coordinates["z"][0]
        threshold_z = 0.10 # the slc of the example is at 2.1
        if not (self.eval_height - threshold_z <= slc_z_height <= self.eval_height + threshold_z):
            print("\n!!! ATTENTION !!!")
            print("The slice is NOT at the expected height.")
            print(f"Expected Height: {self.eval_height} ± {threshold_z}")
            print(f"Actual Height: {slc_z_height}")
            print("Please adjust the slice height to match the expected height.\n")

        # Store the shape of the grid
        self.grid_shape = (len(self.all_x_coords), len(self.all_y_coords))

    def _get_extco_array(self, time):
        """
        Get the array of extinction coefficients from the relevant slice file at a given time.

        :param time: Timestep to evaluate in seconds.
        :type time: float
        :return: Array of extinction coefficients at the specified time.
        :rtype: np.ndarray
        """
        time_index = self.slc.get_nearest_timestep(time)
        data = self.slc_alldata[time_index]
        extco_array = data
        self.debug_message("_get_extco_array:",extco_array)
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

        self.debug_message("_get_non_concealed_cells_idx:", x_idx)
        self.debug_message("_get_non_concealed_cells_idx:", y_idx)
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

        self.debug_message("_get_mean_extco_array:", mean_extco_array)
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
        distance_array = np.linalg.norm(np.array([self.xv - wp.x, self.yv - wp.y]), axis=0)

        self.distance_array_list.append(distance_array)

        self.debug_message("_get_dist_array:", distance_array.shape)
        return distance_array

    def _get_view_angle_array(self, waypoint_id):
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

        self.debug_message("_get_view_angle_array:", view_array)
        return view_array

    def build_obstructions_array(self):
        # Initialize arrays for external collisions and cell obstructions
        meshgrid = self._get_extco_array(0)
        self.obstruction_array = np.zeros_like(meshgrid)

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
        self.obstruction_array = np.flip(self.obstruction_array, axis=1)

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
        # edge_cells[1:-1, 1:-1] = False
        # edge_cells[30:-30, 30:-30] = False
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
                num_vis_cells = np.where(np.in1d(line_x_idx, obstructed_cells[0]) == True)[0][0]
                non_concealed_cells_array[line_x_idx[:num_vis_cells], line_y_idx[:num_vis_cells]] = True
            # If the line doesn't intersect any obstructions, mark the entire line as visible
            else:
                non_concealed_cells_array[line_x_idx, line_y_idx] = True

            # Reset the buffer matrix for the next iteration
            buffer_array.fill(0)

        # Add the finalized visibility matrix to the list and return its transpose
        non_concealed_cells_array = non_concealed_cells_array.T
        self.non_concealed_cells_array_list.append(non_concealed_cells_array)
        self.non_concealed_cells_xy_idx_dict[waypoint_id] = np.where(non_concealed_cells_array == True)

        self.debug_message("_get_non_concealed_cells_array:", non_concealed_cells_array)
        return non_concealed_cells_array

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

        self.debug_message("_get_vismap:", vismap)
        return vismap

    def get_bool_vismap(self, waypoint_id, timestep, extinction, view_angle, collision, aa):
        """
        Generate a boolean visibility map for a specific waypoint.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :param timestep: Timestep for which to calculate the visibility map.
        :type timestep: float
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
            view_array = self._get_view_angle_array(waypoint_id)
        else:
            view_array = 1
        if extinction:
            vismap = self._get_vismap(waypoint_id, timestep)
        else:
            vismap = self.max_vis
        distance_array = self._get_dist_array(waypoint_id)

        vismap_total = view_array * vismap * non_concealed_cells_array
        bool_vismap = np.where(vismap_total >= distance_array, True, False)
        bool_vismap = np.where(vismap_total < self.min_vis, False, bool_vismap)

        self.debug_message("get_bool_vismap:", bool_vismap)
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

        self.debug_message("get_abs_bool_vismap:", absolute_boolean_vismap)
        return absolute_boolean_vismap

    def get_time_aggl_abs_bool_vismap(self):
        """
        Calculate the time-agglomerated absolute boolean visibility map.

        :return: Time-agglomerated absolute boolean visibility map.
        :rtype: np.ndarray
        """
        return self.time_agglomerated_absolute_boolean_vismap

    def plot_abs_bool_vismap(self):  # Todo: is duplicate of plot_time_agglomerated_absolute_boolean_vismap
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

    def compute_all(self, extinction=True, view_angle=True, collision=True, aa=True):
        self._read_fds_data()
        self.build_obstructions_array()
        for time in self.times:
            absolute_boolean_vismap = self.get_abs_bool_vismap(time, extinction, view_angle, collision, aa)
            self.absolute_boolean_vismap_dict[time] = absolute_boolean_vismap
        self.time_agglomerated_absolute_boolean_vismap = np.logical_and.reduce(
            list(self.absolute_boolean_vismap_dict.values()))

    def plot_time_aggl_abs_bool_vismap(self, size=(20, 10), bbox_pad=1, fontsize=10, display_colorbar=True):
        """
        Plot the time-agglomerated absolute boolean visibility map.

        Parameters:
        - size (tuple): Size of the figure. Default is (20, 10).
        - bbox_pad (int): Padding for the annotation box. Default is 1.
        - fontsize (int): Font size for the annotation. Default is 10.
        - display_colorbar (bool): Flag to display the colorbar. Default is True.
        """

        # Create a new figure
        plt.figure(figsize=size)

        # Define the extent of the plot
        extent = (self.all_x_coords[0], self.all_x_coords[-1], self.all_y_coords[-1], self.all_y_coords[0])

        # Display background image if available
        if self.background_image is not None:
            plt.imshow(self.background_image, extent=extent)

        # Define the colormap
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])

        # Display the visibility map
        img = plt.imshow(self.time_agglomerated_absolute_boolean_vismap, cmap=cmap, extent=extent, alpha=0.5)

        # Extract and plot x and y values of waypoints
        x_values = [wp.x for wp in self.way_points_list]
        y_values = [wp.y for wp in self.way_points_list]
        plt.plot((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen', linestyle='--')
        plt.scatter((self.start_point[0], *x_values), (self.start_point[1], *y_values), color='darkgreen')

        # Annotate waypoints
        for wp_id, wp in enumerate(self.way_points_list):
            plt.annotate(f"WP : {wp_id:>}\nC : {wp.c:>}\nIOR : {wp.ior}",
                         xy=(wp.x + 0.3, wp.y + 1.5),
                         bbox=dict(boxstyle="round", fc="w", pad=bbox_pad), fontsize=fontsize)

        # Set labels
        plt.xlabel("X / m")
        plt.ylabel("Y / m")

        # Display colorbar if enabled
        if display_colorbar:
            cbar = plt.colorbar(img, orientation='horizontal', pad=0.07, aspect=32, fraction=0.055)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Not Visible', 'Visible'])
            cbar.set_label('Visibility of the sign')

        # Display the plot
        plt.show()
