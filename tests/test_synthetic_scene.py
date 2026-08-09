"""Visibility for a scene that has geometry but no FDS simulation.

``read_fds_data`` couples three things that are separable: the sampling grid,
the extinction field, and the obstructions. Only the field has to come from
FDS. These tests drive the other route -- ``set_grid`` plus
``set_uniform_extco`` -- so that a clear-air scene is evaluated by the same ray
casting, view-angle and ``max_vis`` handling as a fire scene, instead of being
approximated by the caller.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fdsvismap import VisMap

X = np.arange(0.25, 20.0, 0.5)
Y = np.arange(0.25, 10.0, 0.5)
SIGN = (10.0, 5.0)


def scene(extco: float = 0.0, alpha: float | None = None, c: float = 3.0) -> VisMap:
    """A 20 x 10 m room with one sign at the centre and no fire."""
    vis = VisMap()
    vis.set_grid(X, Y)
    vis.set_uniform_extco(extco)
    vis.set_time_points([0.0])
    vis.set_waypoint(0, SIGN[0], SIGN[1], c=c, alpha=alpha)
    return vis


def computed(vis: VisMap) -> VisMap:
    vis.compute_all(view_angle=True, obstructions=True, aa=True)
    return vis


class TestGridSetup:
    def test_grid_matches_the_shape_read_fds_data_would_build(self):
        vis = scene()
        assert vis.fds_grid_shape == (X.size, Y.size)
        # Cell size is derived from the extent exactly as in read_fds_data.
        assert vis.cell_size[0] == pytest.approx(0.5)
        assert vis.cell_size[1] == pytest.approx(0.5)
        # extent is the outer envelope of the cells, not the cell centres.
        assert vis.extent[0, 0] == pytest.approx(X[0] - 0.25)
        assert vis.extent[0, 1] == pytest.approx(X[-1] + 0.25)

    def test_obstructions_can_be_added_straight_after_set_grid(self):
        """read_fds_data allocates the obstruction array; set_grid must too.

        Otherwise the first add_visual_obstruction indexes an empty array.
        """
        vis = scene()
        vis.add_visual_obstruction(9.5, 10.5, 0.0, 4.0)
        assert vis.obstructions_array.shape == (Y.size, X.size)
        assert vis.obstructions_array.any()

    def test_grid_needs_two_coordinates_per_axis(self):
        vis = VisMap()
        with pytest.raises(ValueError, match="at least two coordinates"):
            vis.set_grid([1.0], [1.0, 2.0])

    def test_grid_rejects_descending_coordinates(self):
        """A descending axis would give a negative cell size and silently
        place obstructions on the wrong cells; refuse it instead."""
        with pytest.raises(ValueError, match="ascending"):
            VisMap().set_grid([10.0, 9.0, 8.0], [0.0, 1.0])

    def test_grid_rejects_non_uniform_spacing(self):
        with pytest.raises(ValueError, match="uniformly spaced"):
            VisMap().set_grid([0.0, 1.0, 5.0, 6.0], [0.0, 1.0])

    def test_negative_extinction_is_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            VisMap().set_uniform_extco(-1.0)

    def test_missing_data_names_both_routes(self):
        with pytest.raises(RuntimeError, match="set_grid"):
            VisMap().get_extco_array_at_time(0.0)

    def test_one_call_is_enough_to_set_the_evaluation_times(self):
        """compute_all reads vismap_time_points, not fds_time_points.

        Without this a caller who set only the extinction field would meet an
        empty-time-points failure inside compute_all.
        """
        vis = VisMap()
        vis.set_grid(X, Y)
        vis.set_uniform_extco(0.0, time_points=[0.0, 10.0])
        assert list(vis.vismap_time_points) == [0.0, 10.0]
        vis.set_waypoint(0, *SIGN, c=3.0, alpha=None)
        vis.compute_all(view_angle=True, obstructions=True, aa=True)
        assert vis.get_visibility_to_wp(10.0, 12.0, 5.0, 0) > 0.0

    def test_a_synthetic_field_supersedes_a_loaded_slice(self):
        vis = scene(extco=0.5)
        vis.slc = object()  # stand-in for a previously read simulation
        vis.set_uniform_extco(0.25)
        assert vis.slc is None
        assert vis.get_extco_array_at_time(0.0).max() == pytest.approx(0.25)

    def test_reading_a_simulation_supersedes_a_synthetic_field(self):
        """The other direction: a real slice wins over a uniform value.

        Without this, set_uniform_extco() followed by read_fds_data() keeps
        returning the uniform value and silently ignores the simulation that
        was just read.
        """
        sim_dir = Path(__file__).parent.parent / "examples" / "room_fire" / "fds_data"
        vis = VisMap()
        vis.set_uniform_extco(0.5)
        vis.read_fds_data(str(sim_dir), fds_slc_height=2)
        assert vis._uniform_extco is None
        # The room fire is not uniformly 0.5 anywhere by the end of the run.
        extco = vis.get_extco_array_at_time(float(vis.fds_time_points[-1]))
        assert extco.min() != pytest.approx(extco.max())


class TestClearAir:
    def test_unobstructed_sight_line_gives_max_vis(self):
        vis = computed(scene(extco=0.0))
        assert vis.get_visibility_to_wp(0.0, 12.0, 5.0, 0) == pytest.approx(
            vis.max_vis, rel=1e-3
        )

    def test_obstruction_blocks_the_sight_line(self):
        """A wall from y = 0 to y = 4 blocks what passes through it, not over it.

        The sign sits east of the wall so both sight lines below actually reach
        it; a centred sign would be reached over the wall's top from either
        side and the test would pass without the ray casting doing anything.
        """
        vis = VisMap()
        vis.set_grid(X, Y)
        vis.set_uniform_extco(0.0)
        vis.set_time_points([0.0])
        vis.set_waypoint(0, 18.0, 5.0, c=3.0, alpha=None)
        vis.add_visual_obstruction(9.5, 10.5, 0.0, 4.0)
        computed(vis)
        # (5, 1) -> (18, 5) crosses the wall at y ~ 2.4, below its top.
        assert vis.get_visibility_to_wp(0.0, 5.0, 1.0, 0) == 0.0
        # (5, 6) -> (18, 5) passes over it at y ~ 5.7.
        assert vis.get_visibility_to_wp(0.0, 5.0, 6.0, 0) > 0.0

    @pytest.mark.parametrize("extco", [0.1, 0.3, 1.0, 3.0])
    def test_uniform_smoke_follows_the_jin_relation(self, extco):
        """S = C / K, clamped at max_vis."""
        vis = computed(scene(extco=extco))
        expected = min(vis.max_vis, 3.0 / extco)
        assert vis.get_visibility_to_wp(0.0, 12.0, 5.0, 0) == pytest.approx(
            expected, rel=1e-3
        )

    def test_contrast_constant_scales_visibility(self):
        """A light-emitting sign (C = 8) is legible further than a reflecting one."""
        reflecting = computed(scene(extco=1.0, c=3.0))
        emitting = computed(scene(extco=1.0, c=8.0))
        assert emitting.get_visibility_to_wp(
            0.0, 12.0, 5.0, 0
        ) > reflecting.get_visibility_to_wp(0.0, 12.0, 5.0, 0)


class TestViewAngleConsistency:
    """get_visibility_to_wp and wp_is_visible must apply the same factors.

    get_visibility_to_wp previously omitted the view-angle term that
    get_vismap applies, so a viewer standing behind a directional sign was
    told the sign was fully legible while wp_is_visible said it was not.
    """

    def test_directional_sign_is_dark_from_behind(self):
        vis = computed(scene(extco=0.0, alpha=90))  # readable from the east
        assert vis.get_visibility_to_wp(0.0, 15.0, 5.0, 0) > 0.0
        assert vis.get_visibility_to_wp(0.0, 5.0, 5.0, 0) == 0.0

    @pytest.mark.parametrize("x", [5.0, 8.0, 12.0, 15.0])
    def test_the_two_accessors_agree(self, x):
        vis = computed(scene(extco=0.0, alpha=90))
        visibility = vis.get_visibility_to_wp(0.0, x, 5.0, 0)
        distance = math.dist((x, 5.0), SIGN)
        assert (visibility >= distance) == bool(vis.wp_is_visible(0.0, x, 5.0, 0))

    def test_any_query_time_is_valid_on_a_static_field(self):
        """A uniform field is time-invariant, so t past the computed range must
        resolve to the computed point instead of raising -- otherwise every
        caller of the synthetic route ends up clamping times itself."""
        vis = computed(scene(extco=0.0))
        assert vis.wp_is_visible(1.0, 12.0, 5.0, 0)
        assert vis.get_visibility_to_wp(3600.0, 12.0, 5.0, 0) == pytest.approx(
            vis.get_visibility_to_wp(0.0, 12.0, 5.0, 0)
        )

    def test_omnidirectional_sign_is_readable_from_both_sides(self):
        vis = computed(scene(extco=0.0, alpha=None))
        assert vis.get_visibility_to_wp(0.0, 15.0, 5.0, 0) > 0.0
        assert vis.get_visibility_to_wp(0.0, 5.0, 5.0, 0) > 0.0
