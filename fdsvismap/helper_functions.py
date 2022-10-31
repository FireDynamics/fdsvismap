import numpy as np

def find_closest_point(coord_array, value):
    '''
    :param coord_array: FDS meshgrid with coordinates
    :param value: coordinate of waypoint
    :return: closest coordinate in FDS meshgrid to given value
    '''
    return (np.abs(coord_array - value)).argmin()