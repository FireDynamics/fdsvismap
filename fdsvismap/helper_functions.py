import numpy as np

def find_closest_point(coord_array, value):
    """
    Find the closest point in a given coordinate array to a given value.

    :param coord_array: The FDS meshgrid containing coordinates.
    :type coord_array: numpy.ndarray
    :param value: The coordinate of the waypoint.
    :type value: float
    :return: The index of the closest coordinate in the FDS meshgrid to the given value.
    :rtype: int
    """
    return (np.abs(coord_array - value)).argmin()