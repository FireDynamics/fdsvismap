"""
Time points after the end of the FDS run (#71) and cells without a loss of visibility in the ASET (#72).

The room_fire example ends with its last slice frame at 500.0 s, ``tests/data/slice_order`` at 20.0 s. The
expected counts of the room_fire cases were derived from the time-aggregated maps, which are computed
independently of the ASET map: a cell is NaN exactly where a sign is visible at every time point up to the
horizon.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fdsvismap import MapStyle, VisMap

matplotlib.use("Agg")

ROOM_FIRE = str(Path(__file__).parent.parent / "examples" / "room_fire" / "fds_data")
SLICE_ORDER = str(Path(__file__).parent / "data" / "slice_order")
README_ROUTE = [(1, 9), (4, 7), (7, 5.5), (9.5, 4.2), (11, 4.2), (15, 6), (17, 9.5)]


def room_fire(times, signs=(1, 2, 3), obstruction=False, route=False, compute=True):
    """Set up the signs of the room_fire example at the given time points."""
    vis = VisMap()
    vis.read_fds_data(ROOM_FIRE, fds_slc_height=2)
    all_signs = {1: (8.4, 4.8, 3, 0), 2: (9.8, 4, 3, 270), 3: (17, 10, 3, 180)}
    for sign_id in signs:
        vis.add_sign(sign_id, *all_signs[sign_id])
    if route:
        vis.add_route("exit route", README_ROUTE, signs=[1, 2, 3])
    vis.set_time_points(times)
    if obstruction:
        vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)
    if compute:
        vis.compute_all()
    return vis


def value_counts(aset):
    """Count the cells per finite ASET value."""
    values, counts = np.unique(aset[np.isfinite(aset)], return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def assert_nan_where_visible_throughout(vis, max_time=None, route_id=None):
    """NaN in the ASET map is exactly where a sign is visible at every time point up to the horizon."""
    np.testing.assert_array_equal(
        np.isnan(vis.get_aset_map(max_time, route_id)),
        vis.get_time_agg_vismap(max_time, route_id),
    )


class TestTimesAfterTheEnd:
    """#71: time points after the last frame of the FDS slice raise instead of reusing that frame."""

    def test_set_time_points_rejects_them(self):
        vis = room_fire([0, 250, 500], compute=False)
        with pytest.raises(ValueError) as excinfo:
            vis.set_time_points([0, 250, 500, 600, 1000])
        message = str(excinfo.value)
        assert "600.0" in message and "1000.0" in message and "500.0" in message
        # A rejected call leaves the time points as they were
        np.testing.assert_array_equal(vis.vismap_time_points, [0.0, 250.0, 500.0])

    def test_a_rejected_call_keeps_the_maps(self):
        vis = room_fire([0, 250, 500], signs=(1, 2))
        aset_map = vis.get_aset_map()
        with pytest.raises(ValueError):
            vis.set_time_points([0, 600])
        np.testing.assert_array_equal(vis.get_aset_map(), aset_map)

    def test_compute_all_rejects_points_set_before_the_data(self):
        vis = VisMap()
        vis.set_time_points([0, 250, 500, 600, 1000])
        vis.read_fds_data(ROOM_FIRE, fds_slc_height=2)
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        with pytest.raises(ValueError) as excinfo:
            vis.compute_all()
        message = str(excinfo.value)
        assert "600.0" in message and "1000.0" in message and "500.0" in message
        # Rejected before any ray casting
        assert not vis.all_sign_ray_casting_cache_dict
        assert not vis.all_time_sign_agg_vismap_list
        # Points excluded by t_max are not evaluated, so they are not rejected
        vis.compute_all(t_max=500)
        assert len(vis.all_time_sign_agg_vismap_list) == 3

    def test_scalar_queries_reject_them(self):
        vis = room_fire(range(0, 500, 50), signs=(1, 2))
        for call in (
            lambda: vis.get_local_visibility(600, 2, 4, 3),
            lambda: vis.get_visibility_to_sign(600, 2, 4, 2),
            lambda: vis.get_extco_array_at_time(600),
        ):
            with pytest.raises(ValueError) as excinfo:
                call()
            assert "600" in str(excinfo.value) and "500.0" in str(excinfo.value)

    def test_the_end_time_is_accepted_within_float_noise(self):
        vis = room_fire(range(0, 500, 50), signs=(1, 2))
        local = vis.get_local_visibility(500, 2, 4, 3)
        to_sign = vis.get_visibility_to_sign(500, 2, 4, 2)
        extco = vis.get_extco_array_at_time(500)
        # Tolerance 1e-6 * 500 s = 5e-4 s
        assert vis.get_local_visibility(500.0004, 2, 4, 3) == local
        assert vis.get_visibility_to_sign(500.0004, 2, 4, 2) == to_sign
        np.testing.assert_array_equal(vis.get_extco_array_at_time(500.0004), extco)
        with pytest.raises(ValueError):
            vis.get_extco_array_at_time(500.001)

    def test_a_short_simulation(self):
        vis = VisMap()
        vis.read_fds_data(SLICE_ORDER, fds_slc_height=2.0)
        vis.set_time_points([0, 20])
        with pytest.raises(ValueError) as excinfo:
            vis.set_time_points([0, 25])
        assert "25.0" in str(excinfo.value) and "20.0" in str(excinfo.value)

    def test_a_uniform_scene_is_exempt(self):
        vis = VisMap()
        vis.set_grid(np.arange(0.25, 20.0, 0.5), np.arange(0.25, 10.0, 0.5))
        vis.set_uniform_extco(0.0, time_points=[0, 1e6])
        vis.add_sign(0, 10.0, 5.0, c=3, alpha=None)
        vis.compute_all()
        assert vis.sign_is_visible(3600, 12.0, 5.0, 0)
        assert vis.get_visibility_to_sign(3600, 12.0, 5.0, 0) == 30


class TestNotLostUpToTheHorizon:
    """#72: no loss of visibility up to the horizon is NaN, not the maximum time."""

    @pytest.fixture
    def synthetic(self):
        """
        Clear air at 0 s and K = 1 /m at 10 s around an omnidirectional sign with c = 3.

        At 10 s the visibility is c / K = 3 m, so the sign stays visible from the 113 cells with a distance of
        at most 3 m, i.e. 0.5 m * (i, j) with i^2 + j^2 <= 36. Everywhere else it is lost at 10 s.
        """
        vis = VisMap()
        vis.set_grid(np.arange(0.25, 20.0, 0.5), np.arange(0.25, 10.0, 0.5))
        vis.set_uniform_extco(0.0, time_points=[0.0, 10.0])
        vis.get_extco_array_at_time = lambda t: np.full(  # type: ignore[method-assign]
            (40, 20), 0.0 if t < 10 else 1.0
        )
        vis.add_sign("S", 5.25, 5.25, c=3, alpha=None)
        vis.add_route("r", [(15.25, 5.25), (7.25, 5.25)], signs=["S"])
        vis.compute_all()
        return vis

    def test_synthetic_map(self, synthetic):
        aset = synthetic.get_aset_map()
        assert int(np.isnan(aset).sum()) == 113
        assert value_counts(aset) == {10.0: 687}
        # 10 m from the sign: lost at 10 s; 2 m from it: never lost
        assert aset[10, 30] == 10.0
        assert np.isnan(aset[10, 14])
        assert_nan_where_visible_throughout(synthetic)

    def test_synthetic_route(self, synthetic):
        aset = synthetic.get_route_aset("r")
        assert len(aset) == 17
        np.testing.assert_array_equal(aset[:14], 10.0)
        assert np.isnan(aset[14:]).all()

    def test_readme_scene(self):
        vis = room_fire(range(0, 500, 50), obstruction=True, route=True)
        aset = vis.get_aset_map()
        assert int(np.isnan(aset).sum()) == 2135
        assert value_counts(aset) == {
            0.0: 1147,
            200.0: 94,
            250.0: 99,
            300.0: 213,
            350.0: 256,
            400.0: 373,
            450.0: 683,
        }
        assert_nan_where_visible_throughout(vis)

        route_aset = vis.get_route_aset("exit route")
        assert len(route_aset) == 100
        assert int(np.isnan(route_aset).sum()) == 57
        assert int((route_aset == 450).sum()) == 12
        coverage = np.array(
            [vis.get_route_coverage("exit route", t) for t in vis.vismap_time_points]
        )
        np.testing.assert_array_equal(np.isnan(route_aset), coverage.all(axis=0))

    def test_up_to_the_end_of_the_simulation(self):
        vis = room_fire([0, 250, 500])
        aset = vis.get_aset_map()
        assert int(np.isnan(aset).sum()) == 1623
        assert value_counts(aset) == {0.0: 1143, 250.0: 193, 500.0: 2041}
        assert_nan_where_visible_throughout(vis)

    def test_max_time_between_time_points(self):
        """The horizon is the last time point up to max_time; max_time itself never appears."""
        vis = room_fire([0, 150, 300, 450], signs=(1, 2))
        aset = vis.get_aset_map()
        assert int(np.isnan(aset).sum()) == 684
        assert value_counts(aset) == {0.0: 3546, 300.0: 388, 450.0: 382}
        assert_nan_where_visible_throughout(vis)

        aset = vis.get_aset_map(400)
        assert int(np.isnan(aset).sum()) == 1066
        assert value_counts(aset) == {0.0: 3546, 300.0: 388}
        assert_nan_where_visible_throughout(vis, 400)

    def test_fractional_time_points(self):
        vis = room_fire([0, 112.5, 225, 337.5, 450], signs=(1, 2))
        aset = vis.get_aset_map()
        assert int(np.isnan(aset).sum()) == 684
        assert value_counts(aset) == {0.0: 3546, 225.0: 133, 337.5: 408, 450.0: 229}
        assert_nan_where_visible_throughout(vis)

        aset = vis.get_aset_map(112.5)
        assert int(np.isnan(aset).sum()) == 1454
        assert value_counts(aset) == {0.0: 3546}
        assert_nan_where_visible_throughout(vis, 112.5)

    @pytest.mark.parametrize("plot_obstructions", [False, True])
    def test_plot_separates_the_three_classes(self, plot_obstructions):
        vis = room_fire(range(0, 500, 50), obstruction=True, route=True)
        style = MapStyle()
        aset = vis.get_aset_map()
        never_visible = ~np.logical_or.reduce(vis.all_time_sign_agg_vismap_list)
        not_lost = np.isnan(aset)
        assert int(never_visible.sum()) == 1147

        fig, ax = vis.plot_aset_map(plot_obstructions=plot_obstructions)
        images = ax.get_images()
        # Map of the times, the two other classes, and the obstructions on top of both
        assert len(images) == (3 if plot_obstructions else 2)
        times_image, classes_image = images[0], images[1]

        assert (times_image.norm.vmin, times_image.norm.vmax) == (0, 450)
        times_rgba = times_image.to_rgba(times_image.get_array())
        # Lost cells show their time, the other cells are transparent in this image
        lost = ~never_visible & ~not_lost
        np.testing.assert_array_equal(
            np.ma.getdata(times_image.get_array())[lost], aset[lost]
        )
        assert (times_rgba[~lost][:, 3] == 0).all()

        classes_rgba = classes_image.to_rgba(classes_image.get_array())
        np.testing.assert_allclose(
            classes_rgba[never_visible], [mcolors.to_rgba(style.never_visible)] * 1147
        )
        np.testing.assert_allclose(
            classes_rgba[not_lost],
            [mcolors.to_rgba(style.visible_until_horizon)] * 2135,
        )
        assert (classes_rgba[lost][:, 3] == 0).all()
        # Not drawn in the color of the latest time either
        assert mcolors.to_hex(style.visible_until_horizon) != mcolors.to_hex(
            times_image.cmap(1.0)
        )

        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["visible until horizon"]
        plt.close(fig)
