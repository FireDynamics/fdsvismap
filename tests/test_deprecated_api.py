"""Tests for the deprecated 0.2 API: every old name warns and forwards to its replacement."""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fdsvismap import Sign, VisMap
from fdsvismap._deprecation import deprecated_attribute, warn_deprecated

matplotlib.use("Agg")


class TestDeprecationHelpers:
    """The helpers that every alias is built on."""

    def test_the_message_names_old_and_new(self):
        with pytest.warns(
            DeprecationWarning,
            match=r"^old\(\) was replaced by new\(\) in fdsvismap 0\.3 and will be removed in 1\.0\.$",
        ):
            warn_deprecated("old()", "new()", stacklevel=2)

    def test_the_warning_points_at_the_caller(self):
        def caller():
            warn_deprecated("old()", "new()", stacklevel=2)

        with pytest.warns(DeprecationWarning) as record:
            caller()
        assert record[0].filename == __file__
        assert record[0].lineno == caller.__code__.co_firstlineno + 1

    def test_attribute_alias_reads_and_writes_the_new_name(self):
        class Holder:
            def __init__(self):
                self.new = 1

            old = deprecated_attribute("old", "new")

        holder = Holder()
        with pytest.warns(
            DeprecationWarning, match="The attribute old was replaced by new"
        ) as record:
            assert holder.old == 1
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning):
            holder.old = 2
        assert holder.new == 2
        assert "old" not in holder.__dict__


class TestWaypoint:
    """fdsvismap.Waypoint.Waypoint is a Sign that warns when it is created."""

    def test_waypoint_is_a_sign_with_the_same_fields(self):
        from fdsvismap.Waypoint import Waypoint

        with pytest.warns(
            DeprecationWarning,
            match=r"fdsvismap\.Waypoint\.Waypoint was replaced by fdsvismap\.Sign",
        ) as record:
            waypoint = Waypoint(8.4, 4.8, 3, 0)
        assert isinstance(waypoint, Sign)
        assert (waypoint.x, waypoint.y, waypoint.c, waypoint.alpha) == (8.4, 4.8, 3, 0)
        assert record[0].filename == __file__

    def test_importing_the_module_and_creating_a_sign_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            import fdsvismap.Waypoint  # noqa: F401

            Sign(8.4, 4.8, 3, 0)


X = np.arange(0.25, 20.0, 0.5)
Y = np.arange(0.25, 10.0, 0.5)


@pytest.fixture
def legacy_scene() -> VisMap:
    """The room fire example of the 0.2 README as a synthetic scene: three waypoints, uniform smoke, one wall."""
    vis = VisMap()
    vis.set_grid(X, Y)
    vis.set_uniform_extco(0.1, time_points=[0.0, 50.0])
    vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)
    with pytest.warns(DeprecationWarning):
        vis.set_waypoint(1, 8.4, 4.8, 3, 0)
        vis.set_waypoint(2, 9.8, 4, 3, 270)
        vis.set_waypoint(3, 17, 10, 3, 180)
    vis.compute_all()
    return vis


class TestSignAliases:
    """Every renamed method warns once and returns what the new name returns."""

    def test_set_waypoint_adds_a_sign(self, legacy_scene):
        assert legacy_scene.all_sign_dict[2] == Sign(9.8, 4, 3, 270)
        assert list(legacy_scene.all_sign_dict) == [1, 2, 3]

    def test_set_waypoint_accepts_an_omnidirectional_waypoint(self):
        vis = VisMap()
        with pytest.warns(
            DeprecationWarning, match=r"set_waypoint\(\) was replaced by add_sign\(\)"
        ):
            vis.set_waypoint("omni", 1.0, 1.0, 3, None)
        assert vis.all_sign_dict["omni"].alpha is None

    @pytest.mark.parametrize(
        "old, new, args",
        [
            ("get_vismap", "get_sign_vismap", (1, 50.0)),
            ("get_wp_agg_vismap", "get_agg_vismap", (50.0,)),
            ("get_time_agg_wp_agg_vismap", "get_time_agg_vismap", ()),
            ("get_time_agg_wp_agg_vismap", "get_time_agg_vismap", (0.0,)),
            ("get_visibility_to_wp", "get_visibility_to_sign", (50.0, 4.0, 4.0, 1)),
            ("wp_is_visible", "sign_is_visible", (50.0, 4.0, 4.0, 1)),
            ("get_distance_to_wp", "get_distance_to_sign", (4.0, 4.0, 1)),
        ],
    )
    def test_old_name_warns_and_returns_the_result_of_the_new_name(
        self, legacy_scene, old, new, args
    ):
        with pytest.warns(
            DeprecationWarning,
            match=rf"{old}\(\) was replaced by {new}\(\) in fdsvismap 0\.3",
        ) as record:
            result = getattr(legacy_scene, old)(*args)
        assert len(record) == 1
        assert record[0].filename == __file__
        np.testing.assert_array_equal(result, getattr(legacy_scene, new)(*args))

    def test_the_new_api_does_not_warn(self):
        """No alias may be used inside the package, a 0.3 script stays silent."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message=r".*was replaced by .* in fdsvismap 0\.3"
            )
            vis = VisMap()
            vis.set_grid(X, Y)
            vis.set_uniform_extco(0.0)
            vis.add_sign(1, 10.0, 5.0, 3, None)
            vis.add_route("route", [(1, 1), (10, 4)])
            vis.compute_all()
            vis.get_agg_vismap(0.0, "route")
            vis.get_time_agg_vismap()
            vis.get_aset_map()
            vis.sign_is_visible(0.0, 5.0, 5.0, 1)
            vis.get_visibility_to_sign(0.0, 5.0, 5.0, 1)
            vis.get_distance_to_sign(5.0, 5.0, 1)
            fig, _ = vis.plot_time_agg_vismap(route_id="route")
            plt.close(fig)
            fig, _ = vis.plot_aset_map(route_id="route")
            plt.close(fig)
