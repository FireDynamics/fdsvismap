"""
Time points after the end of the FDS run (#71).

The room_fire example ends with its last slice frame at 500.0 s, ``tests/data/slice_order`` at 20.0 s.
"""

from pathlib import Path

import numpy as np
import pytest

from fdsvismap import VisMap

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
