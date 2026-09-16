"""Module defining the Sign data structure for fdsvismap."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Sign:
    """
    A safety sign, the object whose visibility is evaluated.

    A sign has exactly one viewing direction, given in the global coordinate system. It does not depend on the routes
    the sign belongs to.

    :param x: X coordinate of the sign referring to global FDS coordinates.
    :type x: float
    :param y: Y coordinate of the sign referring to global FDS coordinates.
    :type y: float
    :param c: Contrast factor of the sign according to Jin.
    :type c: float
    :param alpha: Orientation angle of the sign according to global FDS coordinates, measured clockwise from the
                  positive y-axis. None means that the sign is visible from all directions.
    :type alpha: float or None
    """

    x: float
    y: float
    c: float
    alpha: Optional[float]
