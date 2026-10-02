"""Helpers for the deprecated 0.2 API, whose waypoints were replaced by signs and routes in fdsvismap 0.3."""

import warnings
from typing import Any

# Version in which the aliases of the 0.2 API are removed, named in every warning.
REMOVAL_VERSION = "1.0"


def warn_deprecated(old: str, replacement: str, stacklevel: int = 3) -> None:
    """
    Warn that a name of the 0.2 API is used.

    :param old: The deprecated name as the user wrote it, e.g. ``"set_waypoint()"`` or
                ``"The attribute all_wp_dict"``.
    :type old: str
    :param replacement: What to use instead, e.g. ``"add_sign()"``.
    :type replacement: str
    :param stacklevel: Frame the warning is attributed to. The default 3 points at the caller of a method
                       whose body calls this function directly; add one for every frame in between, e.g. the
                       generated ``__init__`` of a dataclass.
    :type stacklevel: int
    """
    warnings.warn(
        f"{old} was replaced by {replacement} in fdsvismap 0.3 and will be removed in "
        f"{REMOVAL_VERSION}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_attribute(old_name: str, new_name: str) -> Any:
    """
    Create a property that forwards an attribute of the 0.2 API to its new name and warns on every access.

    :param old_name: Name of the attribute in 0.2, which becomes the name of the property.
    :type old_name: str
    :param new_name: Name of the attribute that holds the value now.
    :type new_name: str
    :return: Property that reads and writes ``new_name``. Returned as ``Any``, so that mypy treats the
             deprecated attribute as untyped instead of as a ``property`` object.
    :rtype: property
    """

    def getter(self: Any) -> Any:
        warn_deprecated(f"The attribute {old_name}", new_name)
        return getattr(self, new_name)

    def setter(self: Any, value: Any) -> None:
        warn_deprecated(f"The attribute {old_name}", new_name)
        setattr(self, new_name, value)

    return property(
        getter,
        setter,
        doc=f"Deprecated alias of ``{new_name}``.\n\n.. deprecated:: 0.3.0\n    Use ``{new_name}``.",
    )
