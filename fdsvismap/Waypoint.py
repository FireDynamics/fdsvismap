"""Deprecated module of the 0.2 API, :class:`Waypoint` was replaced by :class:`fdsvismap.Sign.Sign` in 0.3."""

from dataclasses import dataclass

from fdsvismap._deprecation import warn_deprecated
from fdsvismap.Sign import Sign


@dataclass
class Waypoint(Sign):
    """
    A waypoint of fdsvismap 0.2, the former name of a :class:`~fdsvismap.Sign.Sign`.

    .. deprecated:: 0.3.0
        Use :class:`~fdsvismap.Sign.Sign`. ``Waypoint`` is a subclass of ``Sign`` with the same fields ``x``,
        ``y``, ``c`` and ``alpha``; creating an instance warns. It will be removed in 1.0.
    """

    def __post_init__(self) -> None:
        """Warn about the deprecated class, attributed to the line that created the instance."""
        # Frames: warn_deprecated, __post_init__, the generated __init__, the caller
        warn_deprecated("fdsvismap.Waypoint.Waypoint", "fdsvismap.Sign", stacklevel=4)
