"""Slice cells outside every mesh, ``read_fds_data(outside_mesh_extco=...)`` (fdsvismap#87).

A cell-centred slice of 10 x 4 cells at 0.75 m over three meshes, assembled by
fdsreader's own ``Slice.to_global()``::

    y = 2.625   B B B B . . C C C C       A: x 0..7.5,   y 0..1.5
    y = 1.875   S B B B . . C C C T       B: x 0..3,     y 1.5..3
    y = 1.125   A A A A A A A A A A       C: x 4.5..7.5, y 1.5..3
    y = 0.375   A A A A A A A A A A       . : outside every mesh (NaN)

K = 0.5 /m in every mesh. The omnidirectional sign S (c = 3) is at cell (0, 2),
the target T at cell (9, 2), d = 6.75 m. The ray covers 10 cells, two of them
outside: with K = 0 there the mean is 8 x 0.5 / 10 = 0.4 /m, the visibility
3 / 0.4 = 7.5 m >= 6.75 m, visible. Today, one NaN cell makes the ray NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_fds import FakeSlice, patch_simulation

from fdsvismap import VisMap

MESHES = [
    ("A", 0.0, 7.5, 0.0, 1.5),
    ("B", 0.0, 3.0, 1.5, 3.0),
    ("C", 4.5, 7.5, 1.5, 3.0),
]
SIGN = (0.375, 1.875)
TARGET = (7.125, 1.875)
OUTSIDE_CELL = (3.375, 1.875)
OUTSIDE = np.zeros((10, 4), dtype=bool)
OUTSIDE[4:6, 2:] = True


def uniform(x, y, t):
    return np.full(x.shape, 0.5)


def read(monkeypatch, field=uniform, **kwargs) -> VisMap:
    patch_simulation(
        monkeypatch, FakeSlice(MESHES, field, step=0.75, cell_centered=True)
    )
    vis = VisMap()
    vis.read_fds_data("three_meshes", fds_slc_index=0, **kwargs)
    vis.add_sign(0, *SIGN, c=3, alpha=None)
    vis.set_time_points([0.0])
    vis.compute_all()
    return vis


def test_outside_cells_are_nan_by_default(monkeypatch):
    """Without the option the result is unchanged: NaN, and the sign is not visible."""
    vis = read(monkeypatch)
    assert vis.fds_grid_shape == (10, 4)
    np.testing.assert_array_equal(np.isnan(vis.get_extco_array_at_time(0.0)), OUTSIDE)
    assert np.isnan(vis.get_visibility_to_sign(0.0, *TARGET, 0))
    assert not vis.sign_is_visible(0.0, *TARGET, 0)
    assert np.isnan(vis.get_local_visibility(0.0, *OUTSIDE_CELL, 3))


def test_clear_outside_cells_dilute_the_ray_mean(monkeypatch):
    vis = read(monkeypatch, outside_mesh_extco=0.0)
    assert vis.get_visibility_to_sign(0.0, *TARGET, 0) == pytest.approx(7.5, rel=1e-12)
    assert vis.sign_is_visible(0.0, *TARGET, 0)
    assert vis.get_sign_vismap(0, 0.0)[2, 9]


def test_local_visibility_outside_is_max_vis(monkeypatch):
    vis = read(monkeypatch, outside_mesh_extco=0.0)
    assert vis.get_local_visibility(0.0, *OUTSIDE_CELL, 3) == vis.max_vis


def test_only_outside_cells_are_filled(monkeypatch):
    on = read(monkeypatch, outside_mesh_extco=0.25).get_extco_array_at_time(0.0)
    off = read(monkeypatch).get_extco_array_at_time(0.0)
    np.testing.assert_array_equal(on[OUTSIDE], 0.25)
    np.testing.assert_array_equal(on[~OUTSIDE], off[~OUTSIDE])


def test_nan_inside_a_mesh_is_not_filled(monkeypatch):
    """Missing data within a mesh must not become clear air."""

    def with_hole(x, y, t):
        k = uniform(x, y, t)
        k[np.isclose(x, 1.125) & np.isclose(y, 2.625)] = np.nan
        return k

    extco = read(
        monkeypatch, field=with_hole, outside_mesh_extco=0.0
    ).get_extco_array_at_time(0.0)
    assert np.isnan(extco[1, 3])
    assert np.isnan(extco).sum() == 1


def test_fill_value_is_an_extinction_coefficient(monkeypatch):
    """For an optical-density slice the fill is applied after the conversion to 1/m."""
    patch_simulation(
        monkeypatch, FakeSlice(MESHES, uniform, step=0.75, cell_centered=True)
    )
    vis = VisMap()
    vis.quantity = "SOOT OPTICAL DENSITY"
    vis.read_fds_data("three_meshes", fds_slc_index=0, outside_mesh_extco=0.25)
    extco = vis.get_extco_array_at_time(0.0)
    np.testing.assert_array_equal(extco[OUTSIDE], 0.25)
    np.testing.assert_allclose(extco[~OUTSIDE], 0.5 * np.log(10))


def test_exterior_is_not_an_obstruction(monkeypatch):
    """With the option and no walls, sight passes through the exterior; a wall there stops it."""
    vis = read(monkeypatch, outside_mesh_extco=0.0)
    assert vis.sign_is_visible(0.0, *TARGET, 0)
    vis.add_visual_obstruction(3.0, 4.5, 1.5, 3.0)
    vis.compute_all()
    assert not vis.sign_is_visible(0.0, *TARGET, 0)


def test_reading_again_without_the_option_restores_nan(monkeypatch):
    vis = read(monkeypatch, outside_mesh_extco=0.0)
    vis.read_fds_data("three_meshes", fds_slc_index=0)
    np.testing.assert_array_equal(np.isnan(vis.get_extco_array_at_time(0.0)), OUTSIDE)


def test_uniform_scene_after_the_option_is_uniform(monkeypatch):
    vis = read(monkeypatch, outside_mesh_extco=0.0)
    vis.set_uniform_extco(0.5)
    np.testing.assert_array_equal(vis.get_extco_array_at_time(0.0), 0.5)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_fill_value_is_rejected(monkeypatch, value):
    patch_simulation(
        monkeypatch, FakeSlice(MESHES, uniform, step=0.75, cell_centered=True)
    )
    with pytest.raises(ValueError, match="outside_mesh_extco"):
        VisMap().read_fds_data(
            "three_meshes", fds_slc_index=0, outside_mesh_extco=value
        )


@pytest.mark.parametrize("cell_centered", [True, False])
def test_gap_across_the_whole_slice_is_an_error(monkeypatch, cell_centered):
    """fdsreader gives the gap no coordinates, so its cells cannot be located."""
    meshes = [("A", 0.0, 3.0, 0.0, 1.5), ("B", 4.5, 7.5, 0.0, 1.5)]
    patch_simulation(
        monkeypatch,
        FakeSlice(meshes, uniform, step=0.75, cell_centered=cell_centered),
    )
    with pytest.raises(ValueError, match="every x and y column of the slice covered"):
        VisMap().read_fds_data("two_meshes", fds_slc_index=0, outside_mesh_extco=0.0)


def node_l_slice() -> FakeSlice:
    """Node-based L at 0.5 m: A x 0..2, y 0..1 and B x 0..1, y 1..2; K = x + 10 y."""
    return FakeSlice(
        [("A", 0.0, 2.0, 0.0, 1.0), ("B", 0.0, 1.0, 1.0, 2.0)],
        lambda x, y, t: x + 10 * y,
        step=0.5,
        cell_centered=False,
    )


def read_node(monkeypatch, slc: FakeSlice) -> np.ndarray:
    patch_simulation(monkeypatch, slc)
    vis = VisMap()
    vis.read_fds_data("sim", fds_slc_index=0, outside_mesh_extco=0.0)
    return vis.get_extco_array_at_time(0.0)


def test_node_based_l_has_no_nan(monkeypatch):
    """The four nodes x, y > 1 are outside and clear; the dropped faces of A and B keep their FDS values."""
    extco = read_node(monkeypatch, node_l_slice())
    assert not np.isnan(extco).any()
    np.testing.assert_array_equal(extco[3:, 3:], 0.0)
    assert extco[3, 2] == 11.5 and extco[2, 4] == 21.0


def test_nan_written_at_a_shared_face_is_not_filled(monkeypatch):
    """B writes the shared face x = 1; its NaN there is in a mesh and stays, with the option as without."""
    slc = FakeSlice(
        [("A", 0.0, 1.0, 0.0, 1.0), ("B", 1.0, 2.0, 0.0, 1.0)],
        lambda x, y, t: x + 10 * y,
        step=0.5,
        cell_centered=False,
    )
    slc._subslices["B"]._data[:, 0, 1] = np.nan  # B's node (1, 0.5)
    raw = slc.to_global()[0]
    extco = read_node(monkeypatch, slc)
    assert np.isnan(extco[2, 1])
    # No cell is outside a mesh, so nothing changes; NaN included
    np.testing.assert_array_equal(extco, raw)


def test_stretched_mesh_is_an_error(monkeypatch):
    """A's node x = 1.7 is listed, but fdsreader assembles the data at x = 1.5."""
    slc = node_l_slice()
    slc.move_nodes("A", "x", [0.0, 0.5, 1.0, 1.7, 2.0])
    with pytest.raises(ValueError, match="uniformly spaced nodes on one common grid"):
        read_node(monkeypatch, slc)


def test_domain_without_outside_cells_is_unchanged():
    """room_fire: eight meshes fill the slice, so the option changes no value and no map."""
    sim_dir = str(Path(__file__).parents[1] / "examples" / "room_fire" / "fds_data")
    results = []
    for option in (None, 0.0):
        vis = VisMap()
        vis.read_fds_data(sim_dir, fds_slc_height=2, outside_mesh_extco=option)
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(3, 17, 10, 3, 180)
        vis.set_time_points([0, 250, 450])
        vis.compute_all()
        results.append(vis)
    off, on = results
    for time in (0, 250, 450):
        k_off = off.get_extco_array_at_time(time)
        assert not np.isnan(k_off).any()
        np.testing.assert_array_equal(on.get_extco_array_at_time(time), k_off)
        for sign_id in (1, 3):
            np.testing.assert_array_equal(
                on.get_sign_vismap(sign_id, time), off.get_sign_vismap(sign_id, time)
            )
