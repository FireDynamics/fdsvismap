"""Module defining the Route data structure for fdsvismap."""

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

SignId = Union[int, str]


@dataclass(eq=False)
class Route:
    """
    A route of egress as a polyline of waypoints, together with the signs that guide along it.

    The route does not have to pass the signs, it is enough to see them. Which signs belong to a route follows from
    the way the route runs, not from the order in which the signs were added.

    :param waypoints: Waypoints of the route as (x, y) pairs in global FDS coordinates, the first one is the
                      starting point. They are stored as an array of the shape (n, 2).
    :type waypoints: array_like
    :param signs: IDs of the signs that belong to the route.
    :type signs: list[int or str]
    :raises ValueError: If the waypoints are not pairs of coordinates or the route has less than two of them.
    """

    waypoints: NDArray[np.float64]
    signs: List[SignId] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Store the waypoints as an array of coordinates and check that they describe a polyline."""
        waypoints: ArrayLike = self.waypoints
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2:
            raise ValueError(
                f"The waypoints of a route have to be (x, y) pairs, their shape is {self.waypoints.shape}."
            )
        if len(self.waypoints) < 2:
            raise ValueError(
                f"A route needs at least two waypoints, {len(self.waypoints)} were given."
            )

    @property
    def length(self) -> float:
        """
        Get the length of the route along its polyline.

        :return: Length of the route in meters.
        :rtype: float
        """
        return float(np.sum(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)))

    def sample(self, step: float) -> NDArray[np.float64]:
        """
        Sample the polyline of the route at equidistant points.

        The first sampling point is the starting point, the last one the end of the route. The actual distance
        between the points is at most ``step``.

        :param step: Maximum distance between two sampling points in meters.
        :type step: float
        :raises ValueError: If the step is not positive.
        :return: Coordinates of the sampling points of the shape (m, 2).
        :rtype: np.ndarray
        """
        if step <= 0:
            raise ValueError(f"The step has to be positive, not {step}.")
        segment_lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        # Waypoints repeated at the same position would make the interpolation ambiguous
        keep = segment_lengths > 0
        waypoints = np.vstack([self.waypoints[:1], self.waypoints[1:][keep]])
        segment_lengths = segment_lengths[keep]
        if not len(segment_lengths):
            return waypoints[:1]
        positions = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        num_points = int(np.ceil(positions[-1] / step)) + 1
        sample_positions = np.linspace(0.0, positions[-1], num_points)
        return np.column_stack(
            [
                np.interp(sample_positions, positions, waypoints[:, 0]),
                np.interp(sample_positions, positions, waypoints[:, 1]),
            ]
        )
