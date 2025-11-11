import numpy as np
from numpy.typing import NDArray


def get_id_of_closest_value(values_array: NDArray[np.floating], value: float) -> int:
    """
    Find the closest point in a given coordinate array to a given value.

    :param values_array: An array of values to which the closest value should be found.
    :type values_array: numpy.ndarray
    :param value: The value to find the closest value to.
    :type value: float
    :return: The index of the closest value in an array to a given value.
    :rtype: int
    """
    return int((np.abs(values_array - value)).argmin())


def count_cells_to_obstruction(
    line_x: NDArray[np.float64],
    line_y: NDArray[np.float64],
    obstruction: NDArray[np.float64],
) -> int:
    """
    Calculate the number of cells until the line intersects with an obstruction.

    :param line_x: 1D array of x-coordinates of the line.
    :type line_x: np.ndarray
    :param line_y: 1D array of y-coordinates of the line.
    :type line_y: np.ndarray
    :param obstruction: 2D array representing the obstruction.
    Shape (n, 2) where n is the number of obstruction cells, each row containing [x, y] coordinates.
    :type obstruction: np.ndarray
    :return: The number of cells until the line intersects with the obstruction,
    or -1 if there's no intersection.
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
