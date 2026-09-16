from dataclasses import dataclass


@dataclass
class MapStyle:
    """
    Colors and opacities of the map plots of a :class:`~fdsvismap.FDSVisMap.VisMap`.

    Each ``VisMap`` holds its own instance as ``vis.style``. Change single values, e.g.
    ``vis.style.visible = "#2e7d32"``, or assign a new ``MapStyle`` to change all plots of this ``VisMap``.
    Colors are given in any format matplotlib accepts.

    The default colors distinguish cells from which a sign is visible (green) from the others (light, neutral
    gray) by lightness and saturation, so they remain distinguishable with red-green color vision deficiency.

    :param not_visible: Color of cells from which no sign is visible.
    :type not_visible: str
    :param visible: Color of cells from which a sign is visible.
    :type visible: str
    :param sign: Color of the signs, drawn as a short bar across the viewing direction with an arrow in the
                 viewing direction, and of the route through them.
    :type sign: str
    :param start_point_face: Fill color of the start point.
    :type start_point_face: str
    :param start_point_edge: Edge color of the start point.
    :type start_point_edge: str
    :param route_covered: Color of the sections of a route from which a sign of the route is visible. Darker than
                          ``visible``, so that the route stays visible on the map.
    :type route_covered: str
    :param route_uncovered: Color of the sections of a route from which no sign of the route is visible. Darker
                            than ``not_visible``, so that the route stays visible on the map.
    :type route_uncovered: str
    :param aset_cmap: Colormap of the ASET map, scaled from 0 to the maximum time.
    :type aset_cmap: str
    :param never_visible: Color of cells in the ASET map from which no sign is visible at any time point. The
                          default is separated from all colors of viridis, also with color vision deficiency.
    :type never_visible: str
    :param obstruction: Color of obstructions.
    :type obstruction: str
    :param obstruction_alpha: Opacity of obstructions.
    :type obstruction_alpha: float
    :param map_alpha: Opacity of the maps over the background image.
    :type map_alpha: float
    """

    not_visible: str = "#dcd9d2"
    visible: str = "#3f9e5a"
    sign: str = "#0a5f28"
    start_point_face: str = "white"
    start_point_edge: str = "black"
    route_covered: str = "#1b7a3c"
    route_uncovered: str = "#8f8a83"
    aset_cmap: str = "viridis"
    never_visible: str = "#c6c9de"
    obstruction: str = "#5a5a5a"
    obstruction_alpha: float = 0.5
    map_alpha: float = 0.7
