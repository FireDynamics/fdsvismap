from dataclasses import dataclass


@dataclass
class Waypoint:
    """
    A waypoint on the visibility map.

    :param x: X coordinate of the waypoint referring to global FDS coordinates.
    :type x: float
    :param x: X coordinate of the waypoint referring to global FDS coordinates.
    :type x: float
    :param y: Y coordinate of the waypoint referring to global FDS coordinates.
    :type y: float
    :param c: Contrast factor for exit sign according to JIN.
    :type c: int
    :param alpha: Orientation angle of the exit sign in the global coordinate system.
    :type alpha: float
    """
    x: float
    y: float
    c: float
    alpha: float
