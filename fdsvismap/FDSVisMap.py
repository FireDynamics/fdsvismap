"""Module for creating visibility maps (VisMap) based on Fire Dynamics Simulator (FDS) data."""

import logging
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import fdsreader as fds  # type: ignore[import-untyped]
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray
from skimage.draw import line, line_aa

from fdsvismap.helper_functions import (
    count_cells_to_obstruction,
    get_id_of_closest_value,
    progress_bar,
)
from fdsvismap.MapStyle import MapStyle
from fdsvismap.Waypoint import Waypoint

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]  # platform-index-sized int
Int32Array = NDArray[np.int32]
ExtCoArray = FloatArray  # extinction coefficient is float
FigureAxes = Tuple[Figure, Axes]


class _TextHandler(HandlerBase):
    """
    Legend handler that draws a text handle, so that a legend entry looks exactly like the text on the map.

    :param text_kwargs: Properties of the text, the same ones the map is annotated with.
    :type text_kwargs: dict
    """

    def __init__(self, text_kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.text_kwargs = text_kwargs

    def create_artists(
        self,
        legend: Any,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Any,
    ) -> List[Artist]:
        """Create the text of a legend entry, centered in the area reserved for the handle."""
        text = Text(
            width / 2 - xdescent,
            height / 2 - ydescent,
            orig_handle.get_text(),
            ha="center",
            va="center",
            **self.text_kwargs,
        )
        text.set_transform(trans)
        return [text]


class RayCastingCache(TypedDict):
    # Flat (nx, ny) indices of the cells along all rays, concatenated
    ray_cells_flat_idx: Int32Array
    # Position of the first cell of each ray in ray_cells_flat_idx
    ray_start_idx: IntArray
    ray_cell_counts: IntArray
    non_concealed_x_idx: IntArray
    non_concealed_y_idx: IntArray


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
    :ivar all_wp_ray_casting_cache_dict: Dictionary storing pre-computed ray casting data (flat indices of the cells along all rays, start index and cell count of each ray) for each waypoint. Initialized as an empty dict.
    :vartype all_wp_ray_casting_cache_dict: dict[int, RayCastingCache]
    :ivar min_vis: Minimum local visibility threshold to meet performance criteria. Initialized to 0.
    :vartype min_vis: float
    :ivar max_vis: Maximum visibility threshold. Initialized to 30.
    :vartype max_vis: float
    :ivar fds_slc_height: Height at which the FDS slice is evaluated. Initialized as None.
    :vartype fds_slc_height: float or None
    :ivar background_image: Background image for the plot. Initialized as None.
    :vartype background_image: ndarray or None # TODO: Type?
    :ivar background_extent: Position of the background image as (x_min, x_max, y_min, y_max). If None, the image covers the simulation domain.
    :vartype background_extent: tuple[float, float, float, float] or None
    :ivar style: Colors and opacities of all plots of this instance.
    :vartype style: MapStyle
    :ivar all_time_wp_agg_vismap_list: List of waypoint-aggregated visibility maps for all time steps. Initialized as an empty list.
    :vartype all_time_wp_agg_vismap_list: list[np.ndarray]
    :ivar num_edge_cells: Number of edge cells considered for collision detection. Initialized to 1.
    :vartype num_edge_cells: int
    """

    def __init__(self) -> None:
        """Initialize the VisMap object."""
        self.obstructions_array: BoolArray = np.array([], dtype=bool)
        self.vismap_time_points: FloatArray = np.array([], dtype=float)
        self.quantity: str = "ext_coef_C0.9H0.1"
        self.start_point: Tuple[float, float] = (0.0, 0.0)
        self.all_wp_dict: Dict[int, Waypoint] = {}
        self.all_wp_distance_array_dict: Dict[int, FloatArray] = {}
        self.all_wp_non_concealed_cells_array_dict: Dict[
            int, Union[BoolArray, int]
        ] = {}
        self.all_wp_angle_array_dict: Dict[int, Union[FloatArray, int]] = {}
        self.all_time_all_wp_vismap_array_list: List[List[BoolArray]] = []
        self.all_wp_non_concealed_cells_xy_idx_dict: Dict[
            int, Tuple[IntArray, IntArray]
        ] = {}
        self.all_wp_ray_casting_cache_dict: Dict[int, RayCastingCache] = {}
        self.min_vis: float = 0.0
        self.max_vis: float = 30
        self.background_image: np.ndarray = np.array([])
        self.background_extent: Optional[Tuple[float, float, float, float]] = None
        self.style: MapStyle = MapStyle()
        self.all_time_wp_agg_vismap_list: List[BoolArray] = []
        self.num_edge_cells: int = 1
        self._t_max_computed: Optional[float] = None

        # FDS parameterPrivate - will be set later in read_fds_data()
        # ----------------------------------------------------
        self.fds_grid_shape: Optional[Tuple[int, int]] = None
        self.slc: Optional[fds.Simulation.Slice] = None
        self.extent: np.ndarray = np.array([])
        self.all_y_coords: FloatArray = np.array([], dtype=float)
        self.all_x_coords: FloatArray = np.array([], dtype=float)
        self.cell_size: Tuple[float, float] = (1.0, 1.0)
        self.fds_time_points: FloatArray = np.array([], dtype=float)
        self.obstructions_collection: Sequence[Any] = []
        self.fds_slc_height: float = 2.0
        # Slice data per FDS time step index, loaded on first access
        self._slice_frames: Dict[int, Float32Array] = {}
        # ----------------------------------------------------

    def set_time_points(self, time_points: Sequence[float]) -> None:
        """
        Set the times on which the simulation should be evaluated.

        Only the slice data of the FDS time steps closest to these time points is kept in memory.

        :param time_points: List of time points in the simulation.
        :type time_points: list
        """
        self.vismap_time_points = np.array(time_points)
        self._release_slice_frames()

    def set_visibility_bounds(self, min_vis: float, max_vis: float) -> None:
        """
        Set a lower and upper bound for visibility as a performance criterion.

        The lower bound is considered as a local minimum value.
        :param min_vis: float
        :type min_vis: Lower limit for local visibility to meet the performance criterion
        :param max_vis: float
        :type max_vis: Upper limit for local visibility to be considered
        """
        self.min_vis = min_vis
        self.max_vis = max_vis

    def set_start_point(self, x: float, y: float) -> None:
        """
        Set the starting point for the route of egress.

        :param x: x-coordinate of the starting point referring to global FDS coordinates.
        :type x: float
        :param y: y-coordinate of the starting point referring to global FDS coordinates.
        :type y: float
        """
        self.start_point = (x, y)

    def set_waypoint(
        self,
        waypoint_id: int,
        x: float,
        y: float,
        c: int,
        alpha: int,
    ) -> None:
        """
        Add a waypoint along the route of egress.

        :param waypoint_id: ID of the waypoint to add to the route.
        :type waypoint_id: int
        :param x: x-coordinate of the waypoint referring to global FDS coordinates.
        :type x: float
        :param y: y-coordinate of the waypoint referring to global FDS coordinates.
        :type y: float
        :param c: Contrast factor for exit sign according to Jin.
        :type c: int
        :param alpha: Orientation angle of the exit sign according to global FDS coordinates.
        :type alpha: int
        """
        self.all_wp_dict[waypoint_id] = Waypoint(x, y, c, alpha)

    def read_fds_data(
        self,
        sim_dir: str,
        fds_slc_height: float = 2.0,
        fds_slc_id: Optional[str] = None,
    ) -> None:
        """
        Read FDS data and store relevant coordinates, shape of the meshgrid, slices and obstructions.

        If defined, the relevant slice file is read by ID, otherwise by quantity and closest to given height.
        The slice data itself is read on first access, see :meth:`get_extco_array_at_time`.

        :param sim_dir: Directory where FDS simulation data is stored
        :type sim_dir: object
        :param fds_slc_id: ID (name) of FDS slice file to be evaluated. Default is None.
        :type fds_slc_id: str
        :param fds_slc_height: The height at which to evaluate visibility. Default is 2.
        :type fds_slc_height: float, optional
        :raises ValueError: If no matching slice is found. The message lists the available slices.
        """
        sim = fds.Simulation(sim_dir)
        if fds_slc_id:
            self.slc = sim.slices.get_by_id(fds_slc_id)
            searched_slice = f"with ID {fds_slc_id!r}"
        else:
            if self.quantity in [
                "ext_coef_C",
                "ext_coef_C0.9H0.1",
                "SOOT EXTINCTION COEFFICIENT",
                "EXTINCTION COEFFICIENT",
            ]:
                fds_quantity = "SOOT EXTINCTION COEFFICIENT"
            elif self.quantity in [
                "OD_C",
                "OD_C0.9H0.1",
                "SOOT OPTICAL DENSITY",
                "OPTICAL DENSITY",
            ]:
                fds_quantity = "SOOT OPTICAL DENSITY"
            else:
                raise ValueError(f"Unsupported quantity: {self.quantity}")
            self.slc = sim.slices.filter_by_quantity(fds_quantity).get_nearest(
                0, 0, fds_slc_height
            )
            searched_slice = f"with quantity {fds_quantity!r}"
        if self.slc is None:
            raise ValueError(
                f"No slice {searched_slice} found in {sim_dir}. Select one of the available slices "
                f"with fds_slc_id:\n{self._describe_slices(sim.slices)}"
            )
        if fds_slc_id:
            logger.info(
                "Slice with ID %s was selected, its quantity is not checked and treated as %s.",
                fds_slc_id,
                self.quantity,
            )
        self.extent = np.array(self.slc.extent._extents)
        self.all_x_coords = self.slc.get_coordinates()["x"]
        self.all_y_coords = self.slc.get_coordinates()["y"]
        self.fds_grid_shape = (len(self.all_x_coords), (len(self.all_y_coords)))
        self.cell_size = (
            (self.extent[0, 1] - self.extent[0, 0]) / self.fds_grid_shape[0],
            (self.extent[1, 1] - self.extent[1, 0]) / self.fds_grid_shape[1],
        )
        self.fds_time_points = self.slc.times
        self.obstructions_collection = sim.obstructions
        self.fds_slc_height = fds_slc_height
        self._slice_frames = {}
        self.build_obstructions_array()

    @staticmethod
    def _describe_slices(slices: Iterable[Any]) -> str:
        """
        Describe FDS slices by ID, quantity and position, one slice per line.

        :param slices: Slices of an FDS simulation.
        :type slices: fdsreader.slcf.SliceCollection
        :return: Description of the slices.
        :rtype: str
        """
        lines = []
        for slc in slices:
            if slc.orientation == 0:
                position = "3D"
            else:
                axis = ("x", "y", "z")[slc.orientation - 1]
                position = f"{axis} = {slc.extent[axis][0]:.2f} m"
            lines.append(f"  {slc.id or '(no ID)'}: {slc.quantity.name}, {position}")
        return "\n".join(lines) if lines else "  (none)"

    def _get_required_time_indices(self) -> Set[int]:
        """
        Get the indices of the FDS time steps closest to the time points set by :meth:`set_time_points`.

        :return: Indices of the FDS time steps.
        :rtype: set[int]
        """
        if self.slc is None:
            return set()
        return {
            int(self.slc.get_nearest_timestep(time)) for time in self.vismap_time_points
        }

    def _release_slice_frames(self) -> None:
        """Remove the slice data of all FDS time steps that are not required for the time points from memory."""
        required_time_indices = self._get_required_time_indices()
        self._slice_frames = {
            time_index: frame
            for time_index, frame in self._slice_frames.items()
            if time_index in required_time_indices
        }

    def _get_slice_frame(self, time_index: int) -> Float32Array:
        """
        Get the slice data of an FDS time step and read it from the FDS output if it is not in memory.

        fdsreader can only assemble the slice for all time steps at once (as float64) and keeps the data of all meshes
        in its cache afterwards. Therefore, all missing time steps required for the time points are read in one go
        together with the requested one and stored as float32, then the fdsreader cache is cleared. Time steps read
        before that are not required for the time points are removed from memory.

        :param time_index: Index of the FDS time step.
        :type time_index: int
        :return: Slice data at the FDS time step, shape (nx, ny).
        :rtype: np.ndarray
        """
        if time_index not in self._slice_frames:
            if self.slc is None:
                raise RuntimeError("FDS data not loaded. Call read_fds_data() first.")
            self._release_slice_frames()
            missing_time_indices = (
                self._get_required_time_indices() | {time_index}
            ) - self._slice_frames.keys()
            logger.info(
                "Reading slice data from the FDS output (missing time steps: %d).",
                len(missing_time_indices),
            )
            slice_data = self.slc.to_global()
            self.slc.clear_cache()
            for index in missing_time_indices:
                self._slice_frames[index] = slice_data[index].astype(np.float32)
        return self._slice_frames[time_index]

    def get_extco_array_at_time(self, time: float) -> ExtCoArray:
        """
        Get the array of extinction coefficients from the relevant slice file closest to the given time.

        The slice data of the FDS time steps closest to the time points (see :meth:`set_time_points`) is read once and
        kept in memory. Other time steps are read from the FDS output on request, which takes longer for large
        simulations.

        :param time: Time point to be evaluated in seconds.
        :type time: float
        :return: Array of extinction coefficients at the specified time.
        :rtype: np.ndarray
        """
        if self.slc is None:
            raise RuntimeError("FDS data not loaded. Call read_fds_data() first.")

        time_index = int(self.slc.get_nearest_timestep(time))
        extco_data = self._get_slice_frame(time_index).astype(np.float64)
        if self.quantity in [
            "OD_C",
            "OD_C0.9H0.1",
            "SOOT OPTICAL DENSITY",
            "OPTICAL DENSITY",
        ]:
            extco_array: ExtCoArray = extco_data * np.log(10)
        else:
            extco_array = extco_data

        return extco_array

    def _get_non_concealed_cells_idx(
        self, waypoint_id: int
    ) -> Tuple[IntArray, IntArray]:
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

    def _get_mean_extco_array_at_time(
        self, waypoint_id: int, time: float
    ) -> FloatArray:
        """
        Get the array of mean extinction coefficients between the waypoint and all non-concealed cells.

        :param waypoint_id: Index of the waypoint for which to calculate the mean extinction coefficients.
        :type waypoint_id: int
        :param time: Time at which the extinction coefficients should be calculated.
        :type time: float
        :return: A 2D numpy array with the mean extinction coefficients at the specified time, transposed for correct orientation.
        :rtype: np.ndarray
        """
        extco_array = self.get_extco_array_at_time(time)
        mean_extco_array = np.zeros_like(extco_array)

        cache = self.all_wp_ray_casting_cache_dict[waypoint_id]
        # Sum up the extinction coefficients along all rays at once, each ray is a segment of the flat index array
        ray_extco_sums = np.add.reduceat(
            extco_array.ravel()[cache["ray_cells_flat_idx"]], cache["ray_start_idx"]
        )
        mean_extco_array[cache["non_concealed_x_idx"], cache["non_concealed_y_idx"]] = (
            ray_extco_sums / cache["ray_cell_counts"]
        )
        return mean_extco_array.T

    def _get_dist_array(self, waypoint_id: int) -> FloatArray:
        """
        Get the array containing distances between the waypoint and all cells.

        :param waypoint_id: The index of the waypoint from which distances are to be calculated.
        :type waypoint_id: int
        :return: A 2D numpy array where each element represents the distance from the specified waypoint to that cell.
        :rtype: np.ndarray
        """
        wp = self.all_wp_dict[waypoint_id]
        self.xv, self.yv = np.meshgrid(self.all_x_coords, self.all_y_coords)
        distance_array: FloatArray = cast(
            FloatArray,
            np.linalg.norm(np.array([self.xv - wp.x, self.yv - wp.y]), axis=0),
        )
        return distance_array

    def _get_view_angle_array(self, waypoint_id: int) -> FloatArray:
        """
        Get the view array considering view angles.

        :param waypoint_id: The index of the waypoint for which view angles are to be calculated.
        :type waypoint_id: int
        :return: A 2D numpy array with the cosine values of the view angles from the waypoint to each cell.
                 A cell at the position of the waypoint itself (distance 0) gets the value 1.
        :rtype: np.ndarray
        """
        distance_array = self._get_dist_array(waypoint_id)
        wp = self.all_wp_dict[waypoint_id]
        view_angle_array: FloatArray
        if wp.alpha is not None:
            view_angle_array = cast(
                FloatArray,
                np.clip(
                    np.divide(
                        np.sin(np.deg2rad(wp.alpha)) * (self.xv - wp.x)
                        + np.cos(np.deg2rad(wp.alpha)) * (self.yv - wp.y),
                        distance_array,
                        out=np.ones_like(distance_array, dtype=np.float64),
                        where=distance_array > 0,
                    ),
                    0,
                    1,
                ),
            )
        else:
            view_angle_array = np.ones_like(distance_array).astype(np.float64)
        return view_angle_array

    def build_obstructions_array(self) -> None:
        """
        Construct an obstruction array based on FDS simulation data.

        Marks cells in the grid as obstructed based on the
        obstruction objects defined within the FDS simulation. It takes into account the height of the slice
        (fds_slc_height) to determine if an obstruction at a given location blocks visibility.
        """
        if self.fds_grid_shape is None:
            raise RuntimeError("FDS data not loaded. Call read_fds_data() first.")
        # Initialize arrays for external collisions and cell obstructions
        obstruction_array: BoolArray = np.zeros(
            (self.fds_grid_shape[1], self.fds_grid_shape[0]), dtype=bool
        )

        # Update the obstruction_matrix based on defined obstructions and their height ranges
        for obstruction in self.obstructions_collection:
            for sub_obstruction in obstruction:
                _, x_range, y_range, z_range = sub_obstruction.extent
                if z_range[0] <= self.fds_slc_height <= z_range[1]:
                    obstruction_array = self._add_visual_object(
                        x_range[0],
                        x_range[1],
                        y_range[0],
                        y_range[1],
                        obstruction_array,
                        True,
                    )
        self.obstructions_array = obstruction_array

    def build_help_arrays(
        self, obstructions: bool, view_angle: bool, aa: bool, progress: bool = False
    ) -> None:
        """
        Construct auxiliary arrays used for the comprehensive creation of visibility maps.

        Note: If 'collision' is False, all cells are considered as non-concealed; if 'view_angle' is False, the angle is not factored into visibility calculations.

        :param obstructions: Flag indicating whether to consider cells being concealed by obstructions.
        :type obstructions: bool
        :param view_angle: Flag indicating whether to consider view angles from each waypoint.
        :type view_angle: bool
        :param aa: Flag indicating whether antialiasing should be used in the calculation of line-of-sight paths, affecting the smoothness of boundaries.
        :type aa: bool, optional
        :param progress: Flag indicating whether a progress bar over the waypoints is shown. Default is False.
        :type progress: bool, optional
        """
        for waypoint_id in progress_bar(
            self.all_wp_dict.keys(), progress, "Preparing waypoints"
        ):
            logger.debug("Preparing waypoint %s", waypoint_id)
            if obstructions:
                non_concealed_cells_array = self._get_non_concealed_cells_array(
                    waypoint_id, aa
                )
                self.all_wp_non_concealed_cells_array_dict[waypoint_id] = (
                    non_concealed_cells_array
                )
                self.all_wp_non_concealed_cells_xy_idx_dict[waypoint_id] = cast(
                    Tuple[IntArray, IntArray], np.where(non_concealed_cells_array)
                )
            else:
                self.all_wp_non_concealed_cells_array_dict[waypoint_id] = 1
                self.all_wp_non_concealed_cells_xy_idx_dict[waypoint_id] = cast(
                    Tuple[IntArray, IntArray],
                    np.where(np.ones_like(self.obstructions_array, dtype=bool)),
                )
            if view_angle:
                self.all_wp_angle_array_dict[waypoint_id] = self._get_view_angle_array(
                    waypoint_id
                )
            else:
                self.all_wp_angle_array_dict[waypoint_id] = 1

            self.all_wp_distance_array_dict[waypoint_id] = self._get_dist_array(
                waypoint_id
            )
            self._build_ray_casting_cache(waypoint_id)

    def _build_ray_casting_cache(self, waypoint_id: int) -> None:
        """
        Pre-compute and cache ray casting data for a waypoint.

        Stores the cells along the rays to all non-concealed cells relative to a waypoint as flat indices of the
        (nx, ny) extinction coefficient array in one int32 array, together with the start index and cell count of
        each ray. This avoids recalculating ray paths at every timestep.

        :param waypoint_id: The index of the waypoint for which to build the cache.
        :type waypoint_id: int
        """
        wp = self.all_wp_dict[waypoint_id]
        ref_x_id = get_id_of_closest_value(self.all_x_coords, wp.x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, wp.y)
        n_y = len(self.all_y_coords)

        non_concealed_x_idx, non_concealed_y_idx = self._get_non_concealed_cells_idx(
            waypoint_id
        )

        ray_paths: List[Int32Array] = []
        for x_id, y_id in zip(non_concealed_x_idx, non_concealed_y_idx):
            x_lp_idx, y_lp_idx = line(ref_x_id, ref_y_id, x_id, y_id)
            ray_paths.append((x_lp_idx * n_y + y_lp_idx).astype(np.int32))

        ray_cells_flat_idx = (
            np.concatenate(ray_paths) if ray_paths else np.array([], dtype=np.int32)
        )
        ray_cell_counts = np.array([len(ray) for ray in ray_paths], dtype=np.intp)
        ray_start_idx = np.zeros_like(ray_cell_counts)
        ray_start_idx[1:] = np.cumsum(ray_cell_counts)[:-1]

        self.all_wp_ray_casting_cache_dict[waypoint_id] = {
            "ray_cells_flat_idx": ray_cells_flat_idx,
            "ray_start_idx": ray_start_idx,
            "ray_cell_counts": ray_cell_counts,
            "non_concealed_x_idx": non_concealed_x_idx,
            "non_concealed_y_idx": non_concealed_y_idx,
        }

    def _get_non_concealed_cells_array(
        self, waypoint_id: int, aa: bool = True
    ) -> BoolArray:
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
        closest_y_id = get_id_of_closest_value(
            self.all_x_coords, wp.x
        )  # TODO: fix x / y coordinates switch
        closest_x_id = get_id_of_closest_value(self.all_y_coords, wp.y)

        # Initialize arrays for the final visibility matrix, buffer matrix, and edge cell identification
        non_concealed_cells_array = np.zeros_like(self.obstructions_array)
        buffer_array = non_concealed_cells_array.copy()
        edge_cells = np.ones_like(self.obstructions_array)
        edge_cells[
            self.num_edge_cells : -self.num_edge_cells,
            self.num_edge_cells : -self.num_edge_cells,
        ] = False
        edge_x_idx, edge_y_idx = np.where(edge_cells)

        # Choose the appropriate line function based on the aa flag
        line_func = line_aa if aa else line

        # Iterate through edge cells to update visibility based on obstructions
        for x_id, y_id in zip(edge_x_idx, edge_y_idx):
            line_x_idx, line_y_idx = line_func(closest_x_id, closest_y_id, x_id, y_id)[  # type: ignore
                :2
            ]

            buffer_array[line_x_idx, line_y_idx] = True
            obstructed_cells = np.where(
                (self.obstructions_array == True) & (buffer_array == True)  # noqa: E712
            )

            # If line intersects obstructions, mark only the segment before the first obstruction as visible
            if obstructed_cells[0].size != 0:
                # TODO: third argument is expected to be an array of floats (coordinates). But obstructed cells is an array of bools.
                num_non_concealed_cells = count_cells_to_obstruction(
                    line_x_idx,
                    line_y_idx,
                    obstructed_cells,  #  type: ignore
                )
                non_concealed_cells_array[
                    line_x_idx[:num_non_concealed_cells],
                    line_y_idx[:num_non_concealed_cells],
                ] = True
            # If the line doesn't intersect any obstructions, mark the entire line as visible
            else:
                non_concealed_cells_array[line_x_idx, line_y_idx] = True

            # Reset the buffer matrix for the next iteration
            buffer_array.fill(0)
        # # Add the finalized visibility matrix to the list and return its transpose
        # non_concealed_cells_array = non_concealed_cells_array.T
        return non_concealed_cells_array

    def _get_visibility_array(self, waypoint_id: int, time: float) -> FloatArray:
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
        vis_array = np.divide(
            wp.c,
            mean_extco_array,
            out=np.full_like(mean_extco_array, self.max_vis),
            where=mean_extco_array != 0,
        )
        vismap = np.where(vis_array > self.max_vis, self.max_vis, vis_array).astype(
            float
        )
        return vismap

    def get_vismap(self, waypoint_id: int, time: float) -> BoolArray:
        """
        Generate a boolean  vismap for a specific waypoint at a given time.

        :param waypoint_id: The index of the waypoint for which the visibility map is to be calculated.
        :type waypoint_id: int
        :param time: The simulation time at which to evaluate visibility.
        :type time: float
        :return: Boolean vismap indicating whether the waypoint can be seen (True) from a specific  cell or not (False).
        :rtype: np.ndarray

        """
        non_concealed_cells_array = self.all_wp_non_concealed_cells_array_dict[
            waypoint_id
        ]
        view_angle_array = self.all_wp_angle_array_dict[waypoint_id]
        visibility_array = self._get_visibility_array(waypoint_id, time)
        distance_array = self.all_wp_distance_array_dict[waypoint_id]

        visibility_array_total = (
            view_angle_array * visibility_array * non_concealed_cells_array
        )
        vismap = np.where(visibility_array_total >= distance_array, True, False)
        vismap = np.where(visibility_array_total < self.min_vis, False, vismap)
        return vismap

    def _check_time_in_computed_range(self, time: float) -> None:
        """Raise a ValueError if ``time`` exceeds the maximum time computed by :meth:`compute_all`."""
        if self._t_max_computed is not None and time > self._t_max_computed:
            raise ValueError(
                f"time={time} exceeds the maximum computed time ({self._t_max_computed}). "
                f"Re-run compute_all() with a higher t_max."
            )

    def get_wp_agg_vismap(self, time: float) -> BoolArray:
        """
        Get a waypoint aggregated bool type visibility map for a specific point in time.

        :param time: Timestep for which to calculate the visibility map.
        :type time: float
        :raises ValueError: If ``time`` exceeds the maximum time computed by :meth:`compute_all`.
        :return: Waypoint aggregated bool type visibility map.
        :rtype: np.ndarray
        """
        self._check_time_in_computed_range(time)
        time_id = get_id_of_closest_value(self.vismap_time_points, time)
        return self.all_time_wp_agg_vismap_list[time_id]

    def get_time_agg_wp_agg_vismap(self, t_max: Optional[float] = None) -> BoolArray:
        """
        Get a time-aggregated and waypoint-aggregated boolean visibility map.

        :param t_max: The maximum time to consider. If not specified, all computed time points are used.
                      Must not exceed the value of ``t_max`` passed to :meth:`compute_all`.
        :type t_max: float, optional
        :raises ValueError: If ``t_max`` exceeds the maximum time computed by :meth:`compute_all`.
        :return: Time-aggregated and waypoint-aggregated boolean visibility map.
        :rtype: BoolArray
        """
        if t_max is not None:
            self._check_time_in_computed_range(t_max)
        maps = [
            wp_agg_vismap
            for time, wp_agg_vismap in zip(
                self.vismap_time_points, self.all_time_wp_agg_vismap_list
            )
            if t_max is None or time <= t_max
        ]
        return np.logical_and.reduce(maps)

    def get_aset_map(self, max_time: Optional[float] = None) -> IntArray:
        """
        Generate a map indicating the earliest time at which each point becomes non-visible.

        :param max_time: The maximum time to consider. If None, the maximum time computed by :meth:`compute_all` is used.
        :type max_time: float, optional
        :return: A 2D array where each cell represents the earliest time of non-visibility
        for the corresponding point. Cells for points that never become non-visible are set to `max_time`.
        :rtype: np.ndarray
        """
        max_time = self._get_max_time(max_time)

        if self.fds_grid_shape is None:
            raise RuntimeError("FDS data not loaded. Call read_fds_data() first.")
        aset_map = np.full(
            (self.fds_grid_shape[1], self.fds_grid_shape[0]), max_time, dtype=int
        )
        for time, wp_agg_vismap in zip(
            self.vismap_time_points, self.all_time_wp_agg_vismap_list
        ):
            if time > max_time:
                break
            mask = ~wp_agg_vismap & (aset_map == max_time)
            aset_map[mask] = time
        return aset_map

    def _get_max_time(self, max_time: Optional[float]) -> float:
        """
        Get the maximum time of an evaluation.

        :param max_time: Requested maximum time. If None, the maximum time computed by :meth:`compute_all` is used.
        :type max_time: float, optional
        :raises ValueError: If ``max_time`` exceeds the maximum time computed by :meth:`compute_all`.
        :return: The maximum time.
        :rtype: float
        """
        if max_time is None:
            return (
                self._t_max_computed
                if self._t_max_computed is not None
                else self.vismap_time_points[-1]
            )
        self._check_time_in_computed_range(max_time)
        return max_time

    def _get_waypoint_position(self, waypoint_id: int) -> int:
        """
        Get the position of a waypoint in the lists of computed vismaps.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :raises ValueError: If there is no waypoint with this ID.
        :return: Position of the waypoint in the order in which the waypoints were set.
        :rtype: int
        """
        if waypoint_id not in self.all_wp_dict:
            raise ValueError(
                f"No waypoint with ID {waypoint_id}. Available IDs: {list(self.all_wp_dict)}"
            )
        return list(self.all_wp_dict).index(waypoint_id)

    def _get_domain_extent(self) -> Tuple[float, float, float, float]:
        """
        Get the extent of the cells of the simulation domain.

        The edges lie half a cell outside the outermost cell centres, so that each pixel of a map covers its cell.

        :return: Extent as (x_min, x_max, y_min, y_max).
        :rtype: tuple[float, float, float, float]
        """
        return (
            float(self.all_x_coords[0] - self.cell_size[0] / 2),
            float(self.all_x_coords[-1] + self.cell_size[0] / 2),
            float(self.all_y_coords[0] - self.cell_size[1] / 2),
            float(self.all_y_coords[-1] + self.cell_size[1] / 2),
        )

    def plot_map(
        self,
        map_array: np.ndarray,
        cmap: Union[str, mcolors.Colormap] = "viridis",
        ax: Optional[Axes] = None,
        plot_obstructions: bool = False,
        flip_y_axis: bool = True,
        alpha: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        cbar_kwargs: Optional[Dict[str, Any]] = None,
    ) -> FigureAxes:
        """
        Plot a map of the simulation domain over the background image.

        The axes show the simulation domain. A background image that extends beyond the domain is cut off, use
        ``ax.set_xlim`` and ``ax.set_ylim`` to show more of it. A new figure uses the compressed layout of
        matplotlib, so that legends next to the map fit into it.

        :param map_array: Array of the shape (ny, nx), e.g. from :meth:`get_aset_map` or :meth:`get_wp_agg_vismap`.
                          Arrays of the shape (nx, ny) such as from :meth:`get_extco_array_at_time` have to be
                          transposed with ``.T``.
        :type map_array: np.ndarray
        :param cmap: Colormap of the map. Default is 'viridis'.
        :type cmap: str or matplotlib.colors.Colormap, optional
        :param ax: Axes to plot into, e.g. a subplot. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis: bool, optional
        :param alpha: Opacity of the map over the background image. If None, ``style.map_alpha`` is used.
        :type alpha: float, optional
        :param vmin: Lower limit of the color scale. If None, the minimum of the map is used.
        :type vmin: float, optional
        :param vmax: Upper limit of the color scale. If None, the maximum of the map is used.
        :type vmax: float, optional
        :param colorbar: Flag indicating whether a colorbar is added. Default is True.
        :type colorbar: bool, optional
        :param cbar_kwargs: Keyword arguments for ``Figure.colorbar``, e.g. ``label`` or ``pad``. They override the
                            defaults ``orientation="horizontal"`` and ``pad=0.15``.
        :type cbar_kwargs: dict, optional
        :raises RuntimeError: If no FDS data has been read.
        :raises ValueError: If the shape of the map does not match the grid.
        :return: The figure and the axes of the plot.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if self.fds_grid_shape is None:
            raise RuntimeError("FDS data not loaded. Call read_fds_data() first.")
        grid_shape = (self.fds_grid_shape[1], self.fds_grid_shape[0])
        if map_array.shape != grid_shape:
            raise ValueError(
                f"The map has the shape {map_array.shape}, expected (ny, nx) = {grid_shape}. "
                "Arrays of the shape (nx, ny) have to be transposed with .T."
            )
        if ax is None:
            fig, ax = plt.subplots(layout="compressed")
        else:
            fig = cast(Figure, ax.figure)

        origin: Literal["upper", "lower"] = "lower" if flip_y_axis else "upper"
        x_min, x_max, y_min, y_max = self._get_domain_extent()
        extent = (
            (x_min, x_max, y_min, y_max)
            if flip_y_axis
            else (x_min, x_max, y_max, y_min)
        )
        # Without add_background_image() the background image is an empty array
        if self.background_image is not None and self.background_image.size:
            bg_x_min, bg_x_max, bg_y_min, bg_y_max = self.background_extent or (
                x_min,
                x_max,
                y_min,
                y_max,
            )
            bg_extent = (
                (bg_x_min, bg_x_max, bg_y_min, bg_y_max)
                if flip_y_axis
                else (bg_x_min, bg_x_max, bg_y_max, bg_y_min)
            )
            ax.imshow(self.background_image, extent=bg_extent, origin=origin)
        im = ax.imshow(
            map_array,
            cmap=cmap,
            alpha=self.style.map_alpha if alpha is None else alpha,
            extent=extent,
            origin=origin,
            vmin=vmin,
            vmax=vmax,
        )
        if plot_obstructions:
            # Only the obstructed cells are drawn, on top of the map
            ax.imshow(
                np.ma.masked_array(
                    self.obstructions_array, mask=~self.obstructions_array
                ),
                extent=extent,
                cmap=mcolors.ListedColormap([self.style.obstruction]),
                alpha=self.style.obstruction_alpha,
                origin=origin,
                vmin=0,
                vmax=1,
            )
        # Fixed limits, otherwise later artists like markers rescale the axes to a larger background image
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if colorbar:
            fig.colorbar(
                mappable=im,
                ax=ax,
                **{"orientation": "horizontal", "pad": 0.15, **(cbar_kwargs or {})},
            )
        ax.set_xlabel("$X$ / m")
        ax.set_ylabel("$Y$ / m")
        return fig, ax

    def _plot_boolean_map(
        self,
        map_array: BoolArray,
        ax: Optional[Axes],
        plot_obstructions: bool,
        flip_y_axis: bool,
        colorbar: bool,
    ) -> FigureAxes:
        """
        Plot a boolean vismap with one color for cells from which no waypoint is visible and one for the others.

        The color scale is fixed to 0 and 1, so that a map without any visible cell or with only visible cells keeps
        its colors.

        :param map_array: Boolean vismap of the shape (ny, nx).
        :type map_array: np.ndarray
        :param ax: Axes to plot into. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis: bool
        :param colorbar: Flag indicating whether a colorbar is added.
        :type colorbar: bool
        :return: The figure and the axes of the plot.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        return self.plot_map(
            map_array,
            cmap=mcolors.ListedColormap([self.style.not_visible, self.style.visible]),
            ax=ax,
            plot_obstructions=plot_obstructions,
            flip_y_axis=flip_y_axis,
            vmin=0,
            vmax=1,
            colorbar=colorbar,
            cbar_kwargs={
                "label": None,
                # Labels in the middle of the two colors
                "ticks": [0.25, 0.75],
                "format": mticker.FixedFormatter(["not visible", "visible"]),
            },
        )

    def _waypoint_id_style(self) -> Dict[str, Any]:
        """
        Get the text properties of the ID of a waypoint, used on the map as well as in the legend.

        :return: Keyword arguments for a text drawing the ID.
        :rtype: dict
        """
        return dict(
            color=self.style.sign,
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=self.style.sign),
            fontsize=7,
        )

    def _add_legend(
        self, ax: Axes, handles: Sequence[Artist], title: Optional[str] = None
    ) -> None:
        """
        Add a legend to the right of the map, outside of the plotted area.

        A text handle is drawn as that text, so that the IDs of the waypoints look the same on the map and in the
        legend.

        :param ax: Axes of the map.
        :type ax: matplotlib.axes.Axes
        :param handles: Legend entries.
        :type handles: list[matplotlib.artist.Artist]
        :param title: Title of the legend.
        :type title: str, optional
        """
        ax.legend(
            handles=list(handles),
            handler_map={Text: _TextHandler(self._waypoint_id_style())},
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=8,
            # The IDs in their circles are higher and wider than the text of the labels
            handleheight=1.6,
            handlelength=1.6,
            title=title,
            title_fontsize=8,
        )

    def _plot_waypoints(
        self, ax: Axes, waypoint_ids: Sequence[int], plot_route: bool
    ) -> List[Artist]:
        """
        Plot the waypoints with their IDs, optionally with the route from the start point through them.

        A waypoint is drawn as a short bar across its viewing direction with an arrow in the viewing direction and
        its ID beyond the arrow, so that the ID stays in the room the sign faces. Bar, arrow and ID have fixed sizes
        in points, independent of the size of the domain. The contrast factor and the viewing angle are described by
        the returned legend entries.

        :param ax: Axes to plot into.
        :type ax: matplotlib.axes.Axes
        :param waypoint_ids: IDs of the waypoints to plot.
        :type waypoint_ids: list[int]
        :param plot_route: Flag indicating whether the route from the start point through the waypoints is plotted.
        :type plot_route: bool
        :return: Legend entries for the waypoints and, with a route, for the start point.
        :rtype: list[matplotlib.artist.Artist]
        """
        waypoints = [self.all_wp_dict[waypoint_id] for waypoint_id in waypoint_ids]
        if plot_route:
            ax.plot(
                [self.start_point[0], *(wp.x for wp in waypoints)],
                [self.start_point[1], *(wp.y for wp in waypoints)],
                color=self.style.sign,
                linestyle="--",
                linewidth=1,
            )
            ax.scatter(
                [self.start_point[0]],
                [self.start_point[1]],
                facecolor=self.style.start_point_face,
                edgecolor=self.style.start_point_edge,
                zorder=3,
            )
        # Directions on the screen: without flip_y_axis the y-axis points downwards
        x_sign = -1 if ax.xaxis_inverted() else 1
        y_sign = -1 if ax.yaxis_inverted() else 1
        for waypoint_id, wp in zip(waypoint_ids, waypoints):
            if wp.alpha is None:
                ax.scatter([wp.x], [wp.y], color=self.style.sign, zorder=3)
                label_x, label_y, label_distance = 0.0, -1.0, 8.0
            else:
                # Viewing direction on the screen, alpha is measured clockwise from the positive y-axis
                view_x = float(np.sin(np.deg2rad(wp.alpha))) * x_sign
                view_y = float(np.cos(np.deg2rad(wp.alpha))) * y_sign
                view_angle = float(np.rad2deg(np.arctan2(view_y, view_x)))
                # The line marker is vertical, rotated by the viewing angle it lies across the viewing direction
                ax.plot(
                    wp.x,
                    wp.y,
                    marker=(2, 2, view_angle),
                    markersize=12,
                    markeredgewidth=3,
                    color=self.style.sign,
                    zorder=3,
                )
                ax.annotate(
                    "",
                    xy=(wp.x, wp.y),
                    xytext=(12 * view_x, 12 * view_y),
                    textcoords="offset points",
                    arrowprops=dict(
                        arrowstyle="<|-",
                        color=self.style.sign,
                        linewidth=1.5,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=3,
                )
                label_x, label_y, label_distance = view_x, view_y, 16.0
            ax.annotate(
                str(waypoint_id),
                xy=(wp.x, wp.y),
                xytext=(label_distance * label_x, label_distance * label_y),
                textcoords="offset points",
                # Aligned so that the ID extends away from the sign
                ha=("right", "center", "left")[round(label_x) + 1],
                va=("top", "center", "bottom")[round(label_y) + 1],
                **self._waypoint_id_style(),
            )
        handles: List[Artist] = [
            self._waypoint_handle(waypoint_id, wp)
            for waypoint_id, wp in zip(waypoint_ids, waypoints)
        ]
        if plot_route:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=self.style.start_point_face,
                    markeredgecolor=self.style.start_point_edge,
                    label="start point",
                )
            )
        return handles

    @staticmethod
    def _waypoint_handle(waypoint_id: int, wp: Waypoint) -> Text:
        """
        Create the legend entry of a waypoint, its ID as handle and its parameters as label.

        The handle is drawn by :class:`_TextHandler`, which gives it the same appearance as the ID on the map.

        :param waypoint_id: ID of the waypoint.
        :type waypoint_id: int
        :param wp: The waypoint.
        :type wp: Waypoint
        :return: Legend entry of the waypoint.
        :rtype: matplotlib.text.Text
        """
        label = f"C = {wp.c}"
        if wp.alpha is not None:
            label += f", $\\alpha$ = {wp.alpha}$^\\circ$"
        return Text(text=str(waypoint_id), label=label)

    def create_aset_map_plot(
        self,
        max_time: Optional[float] = None,
        plot_obstructions: bool = False,
        flip_y_axis: bool = True,
        ax: Optional[Axes] = None,
    ) -> FigureAxes:
        """
        Create a plot visualizing the ASET map (Available Safe Egress Time) map indicating for each cell the first time any waypoint is not visible.

        The color scale ``style.aset_cmap`` runs from 0 to the maximum time. Cells from which no waypoint is visible
        at any time point up to the maximum time are drawn in ``style.never_visible``.

        :param max_time: The maximum time value to consider for the ASET calculations. If None, it defaults to the last time in the visibility data.
        :type max_time: float, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis:  bool, Default is True.
        :param ax: Axes to plot into, e.g. a subplot. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :return: A tuple containing the matplotlib figure and axes objects that display the ASET map.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        max_time = self._get_max_time(max_time)
        aset_map = self.get_aset_map(max_time)
        ever_visible = np.zeros_like(aset_map, dtype=bool)
        for time, wp_agg_vismap in zip(
            self.vismap_time_points, self.all_time_wp_agg_vismap_list
        ):
            if time <= max_time:
                ever_visible |= wp_agg_vismap
        never_visible = ~ever_visible

        # Masked cells are drawn in the "bad" color of the colormap
        cmap = plt.get_cmap(self.style.aset_cmap).with_extremes(
            bad=self.style.never_visible
        )
        fig, ax = self.plot_map(
            np.ma.masked_array(aset_map, mask=never_visible),
            cmap=cmap,
            ax=ax,
            plot_obstructions=plot_obstructions,
            flip_y_axis=flip_y_axis,
            vmin=0,
            vmax=max_time,
            cbar_kwargs={"label": "Time / s"},
        )
        return fig, ax

    def create_time_agg_wp_agg_vismap_plot(
        self,
        t_max: Optional[float] = None,
        plot_obstructions: bool = False,
        flip_y_axis: bool = True,
        ax: Optional[Axes] = None,
        legend: bool = True,
    ) -> FigureAxes:
        """
        Create a plot visualizing the time-aggregated visibility map for all waypoints.

        The map uses the colors ``style.visible`` and ``style.not_visible`` to distinguish whether any waypoint is
        visible or not from each cell. The plot also features the trajectory of movement from the start point through
        all waypoints, marked with their IDs. The contrast factor and the viewing angle of each waypoint are given in
        a legend to the right of the map.

        :param t_max: The maximum time to consider. If not specified, all computed time points are used.
        :type t_max: float, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis:  bool, Default is True.
        :param ax: Axes to plot into, e.g. a subplot. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :param legend: Flag indicating whether a legend is added. Default is True.
        :type legend: bool, optional
        :return: A tuple containing the matplotlib figure and axes objects that display the aggregated visibility map.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        fig, ax = self._plot_boolean_map(
            self.get_time_agg_wp_agg_vismap(t_max),
            ax=ax,
            plot_obstructions=plot_obstructions,
            flip_y_axis=flip_y_axis,
            colorbar=True,
        )
        handles = self._plot_waypoints(ax, list(self.all_wp_dict), plot_route=True)
        if legend and handles:
            self._add_legend(ax, handles, title="Waypoints")
        return fig, ax

    def plot_vismap(
        self,
        time: float,
        waypoint_id: Optional[int] = None,
        ax: Optional[Axes] = None,
        plot_obstructions: bool = False,
        flip_y_axis: bool = True,
        colorbar: bool = True,
        legend: bool = True,
    ) -> FigureAxes:
        """
        Plot the boolean vismap at a time point, either of one waypoint or aggregated over all waypoints.

        The time is rounded to the closest time point computed by :meth:`compute_all`. The plot shows the waypoints
        with their IDs, their contrast factor and viewing angle are given in a legend to the right of the map.

        :param time: Time point in seconds.
        :type time: float
        :param waypoint_id: ID of the waypoint. If None, the vismap aggregated over all waypoints is plotted.
        :type waypoint_id: int, optional
        :param ax: Axes to plot into, e.g. a subplot. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes, optional
        :param plot_obstructions: Flag indicating whether obstruction at the evaluation height should be plotted or not.
        :type plot_obstructions: bool, optional
        :param flip_y_axis: Flag indicating whether y-axis should be flipped or not to have the origin at bottom left.
        :type flip_y_axis: bool, optional
        :param colorbar: Flag indicating whether a colorbar is added. Default is True.
        :type colorbar: bool, optional
        :param legend: Flag indicating whether a legend of the waypoints is added. Default is True.
        :type legend: bool, optional
        :raises RuntimeError: If :meth:`compute_all` has not been called.
        :raises ValueError: If ``time`` exceeds the maximum computed time or there is no waypoint with this ID.
        :return: The figure and the axes of the plot.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if not self.all_time_wp_agg_vismap_list:
            raise RuntimeError("No vismaps computed. Call compute_all() first.")
        if waypoint_id is None:
            vismap = self.get_wp_agg_vismap(time)
            waypoint_ids = list(self.all_wp_dict)
        else:
            position = self._get_waypoint_position(waypoint_id)
            self._check_time_in_computed_range(time)
            time_id = get_id_of_closest_value(self.vismap_time_points, time)
            vismap = self.all_time_all_wp_vismap_array_list[time_id][position]
            waypoint_ids = [waypoint_id]
        fig, ax = self._plot_boolean_map(
            vismap,
            ax=ax,
            plot_obstructions=plot_obstructions,
            flip_y_axis=flip_y_axis,
            colorbar=colorbar,
        )
        handles = self._plot_waypoints(ax, waypoint_ids, plot_route=False)
        if legend and handles:
            self._add_legend(ax, handles, title="Waypoints")
        return fig, ax

    def add_background_image(
        self, file: str, extent: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Load and set a background image for future plots created within this visualization class.

        :param file: Path to the image file that will be used as the background.
        :type file: str
        :param extent: Position of the image edges in global FDS coordinates as (x_min, x_max, y_min, y_max). The
                       image may extend beyond the simulation domain. If None, the image covers exactly the
                       simulation domain.
        :type extent: tuple[float, float, float, float], optional
        :raises ValueError: If the extent is not ordered as (x_min, x_max, y_min, y_max).
        """
        if extent is not None and not (extent[0] < extent[1] and extent[2] < extent[3]):
            raise ValueError(
                f"The extent {extent} has to be ordered as (x_min, x_max, y_min, y_max)."
            )
        image = plt.imread(file)
        # PNG files are read as float32 in [0, 1], uint8 needs a quarter of the memory
        if np.issubdtype(image.dtype, np.floating):
            image = np.round(image * 255).astype(np.uint8)
        self.background_image = np.flip(image, axis=0)
        self.background_extent = extent

    def compute_all(
        self,
        t_max: Optional[float] = None,
        view_angle: bool = True,
        obstructions: bool = True,
        aa: bool = True,
        progress: bool = False,
    ) -> None:
        """
        Execute all required computations to generate aggregated visibility maps over all waypoints and time points.

        The results of previous calls are replaced. Messages about the progress are sent to the logger
        ``fdsvismap.FDSVisMap`` (level INFO per time point, DEBUG per waypoint), e.g. shown by
        ``logging.basicConfig(level=logging.INFO)``.

        :param t_max: The maximum simulation time to compute up to. If not specified, all available time points are computed.
        :type t_max: float, optional
        :param view_angle: Determines if view angles should be considered in the visibility calculations,
                          affecting how visibility is computed relative to the waypoint orientations. Default is True.
        :type view_angle: bool
        :param obstructions: Determines if collisions (obstructions) should be considered, impacting whether
                         certain paths are considered visible based on physical barriers. Default is True.
        :type obstructions: bool
        :param aa: Determines if antialiasing should be applied when computing visibility lines, which can
                  smooth the appearance of the visibility boundaries but might affect computational performance. Default is True.
        :type aa: bool
        :param progress: Determines if progress bars are shown while the waypoints are prepared and the vismaps are
                         computed. Default is False.
        :type progress: bool
        """
        time_points = (
            self.vismap_time_points[self.vismap_time_points <= t_max]
            if t_max is not None
            else self.vismap_time_points
        )
        self._t_max_computed = float(time_points[-1])
        self.all_time_all_wp_vismap_array_list = []
        self.all_time_wp_agg_vismap_list = []
        self.build_help_arrays(
            view_angle=view_angle, obstructions=obstructions, aa=aa, progress=progress
        )
        for time in progress_bar(time_points, progress, "Computing vismaps"):
            logger.info("Simulation time %s s of %s s", time, self._t_max_computed)
            all_wp_vismap_array_list = []
            for waypoint_id in self.all_wp_dict.keys():
                logger.debug("Waypoint %s at simulation time %s s", waypoint_id, time)
                vismap = self.get_vismap(waypoint_id, time)
                all_wp_vismap_array_list.append(vismap)
            self.all_time_all_wp_vismap_array_list.append(all_wp_vismap_array_list)
            wp_agg_vismap = np.logical_or.reduce(all_wp_vismap_array_list)
            self.all_time_wp_agg_vismap_list.append(wp_agg_vismap)

    def get_local_visibility(self, time: float, x: float, y: float, c: float) -> float:
        """
        Calculate the local visibility at a specific cell closest to the given x, y.

        coordinates at a certain time based on local extinction coefficient values.

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
        extco_array = self.get_extco_array_at_time(time)
        local_extco = extco_array[
            ref_x_id, ref_y_id
        ]  # TODO: Why are coordinates switched for extco array?
        if local_extco == 0:
            return self.max_vis

        visibility: float = min(c / local_extco, self.max_vis)
        return visibility

    def get_visibility_to_wp(
        self, time: float, x: float, y: float, waypoint_id: int
    ) -> float:
        """
        Calculate the visibility at a specific cell closest to the given x, y.

        coordinates at a certain time relative to a specific waypoint.

        :param time: The simulation time at which to calculate the visibility.
        :type time: float
        :param x: The x-coordinate in the simulation grid where visibility is to be calculated.
        :type x: float
        :param y: The y-coordinate in the simulation grid where visibility is to be calculated.
        :type y: float
        :param waypoint_id: The ID of the waypoint to check visibility for.
        :type waypoint_id: int
        :return: The computed visibility value at the given location and time relative to a specific waypoint..
        :rtype: float
        """
        ref_x_id = get_id_of_closest_value(self.all_x_coords, x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, y)
        visibility_array = self._get_visibility_array(waypoint_id, time)
        non_concealed_cells_array = self.all_wp_non_concealed_cells_array_dict[
            waypoint_id
        ]
        masked_visibility_array = visibility_array * non_concealed_cells_array
        visibility = float(masked_visibility_array[ref_y_id, ref_x_id])
        return visibility

    def wp_is_visible(self, time: float, x: float, y: float, waypoint_id: int) -> bool:
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
        :raises ValueError: If ``time`` exceeds the maximum time computed by :meth:`compute_all`.
        :return: A boolean value indicating whether the specified waypoint is visible from the given location and time.
        :rtype: bool
        """
        self._check_time_in_computed_range(time)
        time_id = get_id_of_closest_value(self.vismap_time_points, time)
        ref_x_id = get_id_of_closest_value(self.all_x_coords, x)
        ref_y_id = get_id_of_closest_value(self.all_y_coords, y)
        vismap_array = self.all_time_all_wp_vismap_array_list[time_id][waypoint_id]
        is_visible = bool(vismap_array[ref_y_id, ref_x_id])
        return is_visible

    def get_distance_to_wp(self, x: float, y: float, waypoint_id: int) -> float:
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
        """
        wp = self.all_wp_dict[waypoint_id]
        distance_to_wp = float(np.linalg.norm(np.array([x - wp.x, y - wp.y]), axis=0))
        return distance_to_wp

    def _add_visual_object(
        self,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        obstructions_array: BoolArray,
        status: bool,
    ) -> BoolArray:
        """
        Add or remove obstructions from a specified rectangular area within the simulation grid.

        This is valid for everything affected by the ray tracing algorithms.

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
        ref_x1_id = get_id_of_closest_value(
            self.all_x_coords, x1 + self.cell_size[0] / 2
        )

        ref_x2_id = (
            get_id_of_closest_value(self.all_x_coords, x2 - self.cell_size[0] / 2) + 1
        )

        ref_y1_id = get_id_of_closest_value(
            self.all_y_coords, y1 + self.cell_size[1] / 2
        )
        ref_y2_id = (
            get_id_of_closest_value(self.all_y_coords, y2 - self.cell_size[1] / 2) + 1
        )

        obstructions_array[ref_y1_id:ref_y2_id, ref_x1_id:ref_x2_id] = status
        return obstructions_array

    def add_visual_hole(self, x1: float, x2: float, y1: float, y2: float) -> None:
        """
        Remove obstructions from a specified rectangular area within the simulation grid.

        This is valid for everything affected by the ray tracing algorithms.

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

    def add_visual_obstruction(
        self, x1: float, x2: float, y1: float, y2: float
    ) -> None:
        """
        Add obstructions from a specified rectangular area to the simulation grid.

        This is valid for everything affected by the ray tracing algorithms.

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
