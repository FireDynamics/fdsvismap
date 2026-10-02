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


class TestStartPointAndPlots:
    """set_start_point() is stored for the deprecated plot, the plot aliases forward to the new plots."""

    def test_set_start_point_is_stored_and_read_back(self):
        vis = VisMap()
        with pytest.warns(DeprecationWarning, match="The attribute start_point"):
            assert vis.start_point is None
        with pytest.warns(
            DeprecationWarning,
            match=r"set_start_point\(\) was replaced by the first waypoint of add_route\(\)",
        ) as record:
            vis.set_start_point(1, 9)
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning):
            assert vis.start_point == (1, 9)
        with pytest.warns(DeprecationWarning):
            vis.start_point = (2, 8)
        assert vis._legacy_start_point == (2, 8)

    def test_aset_plot_forwards_to_plot_aset_map(self, legacy_scene):
        with pytest.warns(
            DeprecationWarning,
            match=r"create_aset_map_plot\(\) was replaced by plot_aset_map\(\)",
        ) as record:
            fig, ax = legacy_scene.create_aset_map_plot(plot_obstructions=True)
        # Count only our warnings, matplotlib may add unrelated ones while drawing
        assert sum(issubclass(w.category, DeprecationWarning) for w in record) == 1
        assert isinstance(fig, Figure) and isinstance(ax, Axes)
        fig_new, ax_new = legacy_scene.plot_aset_map(plot_obstructions=True)
        np.testing.assert_array_equal(
            ax.images[0].get_array(), ax_new.images[0].get_array()
        )
        plt.close(fig)
        plt.close(fig_new)

    def test_time_agg_plot_draws_the_trajectory_from_the_start_point(
        self, legacy_scene
    ):
        with pytest.warns(DeprecationWarning):
            legacy_scene.set_start_point(1, 9)
        with pytest.warns(
            DeprecationWarning,
            match=r"create_time_agg_wp_agg_vismap_plot\(\) was replaced by plot_time_agg_vismap\(\)",
        ) as record:
            fig, ax = legacy_scene.create_time_agg_wp_agg_vismap_plot()
        assert sum(issubclass(w.category, DeprecationWarning) for w in record) == 1
        dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed) == 1
        np.testing.assert_array_equal(
            dashed[0].get_xydata(), [(1, 9), (8.4, 4.8), (9.8, 4), (17, 10)]
        )
        # The map itself is the one of the new plot
        fig_new, ax_new = legacy_scene.plot_time_agg_vismap()
        np.testing.assert_array_equal(
            ax.images[0].get_array(), ax_new.images[0].get_array()
        )
        plt.close(fig)
        plt.close(fig_new)

    def test_time_agg_plot_without_a_start_point_draws_no_trajectory(
        self, legacy_scene
    ):
        with pytest.warns(DeprecationWarning):
            fig, ax = legacy_scene.create_time_agg_wp_agg_vismap_plot(
                plot_obstructions=True, flip_y_axis=False
            )
        assert not [line for line in ax.lines if line.get_linestyle() == "--"]
        assert ax.yaxis_inverted()
        plt.close(fig)


ATTRIBUTE_ALIASES = [
    ("all_wp_dict", "all_sign_dict"),
    ("all_wp_distance_array_dict", "all_sign_distance_array_dict"),
    (
        "all_wp_non_concealed_cells_array_dict",
        "all_sign_non_concealed_cells_array_dict",
    ),
    ("all_wp_angle_array_dict", "all_sign_angle_array_dict"),
    ("all_time_all_wp_vismap_array_list", "all_time_all_sign_vismap_list"),
    (
        "all_wp_non_concealed_cells_xy_idx_dict",
        "all_sign_non_concealed_cells_xy_idx_dict",
    ),
    ("all_wp_ray_casting_cache_dict", "all_sign_ray_casting_cache_dict"),
    ("all_time_wp_agg_vismap_list", "all_time_sign_agg_vismap_list"),
]


class TestAttributeAliases:
    """The renamed attributes are reachable under their 0.2 names, reading and writing."""

    @pytest.mark.parametrize("old, new", ATTRIBUTE_ALIASES)
    def test_reading_the_old_name_returns_the_new_attribute(
        self, legacy_scene, old, new
    ):
        with pytest.warns(
            DeprecationWarning, match=f"The attribute {old} was replaced by {new}"
        ) as record:
            value = getattr(legacy_scene, old)
        assert len(record) == 1
        assert record[0].filename == __file__
        assert value is getattr(legacy_scene, new)
        assert len(value) == 3 or len(value) == 2  # 3 signs, or 2 time points

    @pytest.mark.parametrize("old, new", ATTRIBUTE_ALIASES)
    def test_writing_the_old_name_sets_the_new_attribute(self, old, new):
        vis = VisMap()
        marker = [] if old.endswith("list") else {}
        with pytest.warns(DeprecationWarning):
            setattr(vis, old, marker)
        assert getattr(vis, new) is marker
        assert old not in vis.__dict__

    def test_the_cache_of_a_sign_has_the_0_3_layout(self, legacy_scene):
        """The dict is aliased, the per-sign layout of 0.3 (flat ray indices) is kept."""
        with pytest.warns(DeprecationWarning):
            cache = legacy_scene.all_wp_ray_casting_cache_dict[1]
        assert set(cache) == {
            "ray_cells_flat_idx",
            "ray_start_idx",
            "ray_cell_counts",
            "non_concealed_x_idx",
            "non_concealed_y_idx",
        }
