"""Nodes of a node-based slice that fdsreader drops at mesh upper faces (fdsvismap#88).

fdsreader's ``Slice.to_global()`` drops the upper x and y node row of each mesh
whose upper face is not at the global maximum, expecting a neighbouring mesh to
supply it (fdsreader#121). On an L-shaped domain no mesh does. The slice below
is assembled by fdsreader itself from two meshes at 0.5 m::

    y = 2.0   B B B . .        A: x 0..2, y 0..1
    y = 1.5   B B B . .        B: x 0..1, y 1..2
    y = 1.0   A A A A A        . : outside both meshes
    y = 0.5   A A A A A
    y = 0.0   A A A A A
           x= 0 .5 1 1.5 2

K = x + 10 y + t, so a value at a transposed index differs from the expected one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_fds import FakeSlice, patch_simulation

from fdsvismap import VisMap

L_MESHES = [("A", 0.0, 2.0, 0.0, 1.0), ("B", 0.0, 1.0, 1.0, 2.0)]
L_MESHES_BY_ID = {mesh_id: box for mesh_id, *box in L_MESHES}
TIMES = (0.0, 10.0)
COORDS = np.arange(0.0, 2.01, 0.5)
# Upper face y = 1 of A beyond B, and upper face x = 1 of B, as (x, y)
DROPPED = [(1.5, 1.0), (2.0, 1.0), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]
OUTSIDE = [(1.5, 1.5), (2.0, 1.5), (1.5, 2.0), (2.0, 2.0)]


def field(x, y, t):
    return x + 10 * y + t


def node(x, y):
    return int(round(x / 0.5)), int(round(y / 0.5))


def l_slice() -> FakeSlice:
    return FakeSlice(L_MESHES, field, step=0.5, cell_centered=False, times=TIMES)


def read_l(monkeypatch) -> VisMap:
    patch_simulation(monkeypatch, l_slice())
    vis = VisMap()
    vis.read_fds_data("l_corridor", fds_slc_index=0)
    return vis


def test_fdsreader_still_drops_the_upper_face_nodes():
    """The defect the workaround exists for.

    If this fails, investigate before removing anything: a partial fix or another
    change in to_global() also fails it. See ``VisMap._recover_dropped_nodes``
    for the removal steps.
    """
    grid = l_slice().to_global()
    nan_nodes = {tuple(COORDS[[i, j]]) for i, j in np.argwhere(np.isnan(grid[0]))}
    assert nan_nodes == set(DROPPED) | set(OUTSIDE)


@pytest.mark.parametrize("time", TIMES)
def test_dropped_nodes_get_their_mesh_values(monkeypatch, time):
    extco = read_l(monkeypatch).get_extco_array_at_time(time)
    for x, y in DROPPED:
        assert extco[node(x, y)] == x + 10 * y + time


def test_nodes_outside_every_mesh_stay_nan(monkeypatch):
    extco = read_l(monkeypatch).get_extco_array_at_time(0.0)
    for x, y in OUTSIDE:
        assert np.isnan(extco[node(x, y)])


def test_assembled_nodes_are_unchanged(monkeypatch):
    raw = l_slice().to_global()[1]
    extco = read_l(monkeypatch).get_extco_array_at_time(10.0)
    assembled = ~np.isnan(raw)
    np.testing.assert_array_equal(extco[assembled], raw[assembled])


def test_rectangular_domain_is_unchanged(monkeypatch):
    """Side-by-side meshes supply each other's faces: nothing is NaN, nothing changes."""
    slc = FakeSlice(
        [("A", 0.0, 1.0, 0.0, 1.0), ("B", 1.0, 2.0, 0.0, 1.0)],
        field,
        step=0.5,
        cell_centered=False,
    )
    raw = slc.to_global()[0]
    patch_simulation(monkeypatch, slc)
    vis = VisMap()
    vis.read_fds_data("rectangle", fds_slc_index=0)
    assert not np.isnan(raw).any()
    np.testing.assert_array_equal(vis.get_extco_array_at_time(0.0), raw)


def test_cell_centred_real_slice_is_unchanged():
    """room_fire: eight meshes, cell-centred slice, as fdsreader assembles it."""
    sim_dir = Path(__file__).parents[1] / "examples" / "room_fire" / "fds_data"
    vis = VisMap()
    vis.read_fds_data(str(sim_dir), fds_slc_height=2.0)
    raw = vis.slc.to_global()[int(vis.slc.get_nearest_timestep(60.0))]
    np.testing.assert_array_equal(
        vis.get_extco_array_at_time(60.0), raw.astype(np.float32)
    )


def read(monkeypatch, slc: FakeSlice) -> np.ndarray:
    patch_simulation(monkeypatch, slc)
    vis = VisMap()
    vis.read_fds_data("sim", fds_slc_index=0)
    return vis.get_extco_array_at_time(0.0)


def grid_id(coords, value):
    return int(np.flatnonzero(np.isclose(coords, value))[0])


def test_nan_written_at_a_shared_face_is_kept(monkeypatch):
    """B writes the shared face x = 1; its NaN there is data and stays, although A has a value."""
    slc = FakeSlice(
        [("A", 0.0, 1.0, 0.0, 1.0), ("B", 1.0, 2.0, 0.0, 1.0)],
        field,
        step=0.5,
        cell_centered=False,
    )
    slc._subslices["B"]._data[:, 0, 1] = np.nan  # B's node (1, 0.5)
    raw = slc.to_global()[0]
    extco = read(monkeypatch, slc)
    assert np.isnan(raw[node(1.0, 0.5)])
    assert np.isnan(extco[node(1.0, 0.5)])
    np.testing.assert_array_equal(extco, raw)


def test_coarse_mesh_face_is_recovered_on_the_grid(monkeypatch):
    """A (1 m) is coarser than the grid (0.5 m); its dropped face y = 1 is sampled as to_global() samples A.

    to_global() repeats a coarse node over the finer grid nodes up to the next one, so the grid node
    (1.5, 1) gets A's node (1, 1) = 11 and (2, 1) gets A's node (2, 1) = 12.
    """
    slc = FakeSlice(
        [
            ("A", 0.0, 2.0, 0.0, 1.0, 1.0),
            ("B", 0.0, 1.0, 1.0, 2.0),
            ("C", 1.0, 2.0, -1.0, 0.0),
            ("D", -1.0, 0.0, 0.0, 2.0),
        ],
        field,
        step=0.5,
        cell_centered=False,
    )
    raw, coords = slc.to_global(return_coordinates=True)
    x, y = coords["x"], coords["y"]
    extco = read(monkeypatch, slc)
    assert extco.shape == raw.shape[1:] == (7, 7)
    assert np.isnan(raw[0, grid_id(x, 1.5), grid_id(y, 1.0)])
    assert extco[grid_id(x, 1.5), grid_id(y, 1.0)] == 11
    assert extco[grid_id(x, 2.0), grid_id(y, 1.0)] == 12
    gx, gy = np.meshgrid(x, y, indexing="ij")
    outside = (gx > 1.0) & (gy > 1.0) | (gx < 1.0) & (gy < 0.0)
    assert not np.isnan(extco[~outside]).any()
    assert np.isnan(extco[outside]).all()


def test_coarse_mesh_with_fewer_slice_coordinates(monkeypatch):
    """fdsreader lists 4 x 4 slice coordinates but assembles 5 x 5 nodes; recovery uses the assembled grid."""
    slc = FakeSlice(
        [("A", 0.0, 2.0, 0.0, 1.0, 1.0), ("B", 0.0, 1.0, 1.0, 2.0)],
        field,
        step=0.5,
        cell_centered=False,
    )
    assert len(slc.get_coordinates()["x"]) == 4
    extco = read(monkeypatch, slc)
    assert extco.shape == (5, 5)
    # Face y = 1 of A sampled from A's nodes (1, 1) and (2, 1), face x = 1 of B from B's own nodes
    assert extco[2, 2] == 11 and extco[3, 2] == 11 and extco[4, 2] == 12
    assert extco[2, 3] == 16 and extco[2, 4] == 21
    assert np.isnan(extco[3:, 3:]).all()


def test_stretched_mesh_is_skipped_with_a_warning(monkeypatch, caplog):
    """A's node x = 1.7 is not on the 0.5 m grid to_global() puts it on: A is not used, its nodes stay NaN.

    (1, 1) is also within B, which is on the grid, so it is recovered from B.
    """
    slc = l_slice()
    slc.move_nodes("A", "x", [0.0, 0.5, 1.0, 1.7, 2.0])
    raw = slc.to_global()[0]
    with caplog.at_level("WARNING", logger="fdsvismap.FDSVisMap"):
        extco = read(monkeypatch, slc)
    assert "Slice fake_slice" in caplog.text
    assert (
        "mesh A" in caplog.text
        and "its x nodes [0.0, 0.5, 1.0, 1.7, 2.0]" in caplog.text
    )
    assert np.isnan(extco[node(1.5, 1.0)]) and np.isnan(extco[node(2.0, 1.0)])
    for x, y in [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]:
        assert extco[node(x, y)] == x + 10 * y
    others = np.ones(raw.shape, dtype=bool)
    for x, y in [(1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]:
        others[node(x, y)] = False
    np.testing.assert_array_equal(extco[others], raw[others])


def test_warning_is_logged_once_per_read(monkeypatch, caplog):
    """Each frame read without time points assembles the slice again; the warning comes once per read_fds_data()."""
    slc = FakeSlice(L_MESHES, field, step=0.5, cell_centered=False, times=TIMES)
    slc.move_nodes("A", "x", [0.0, 0.5, 1.0, 1.7, 2.0])
    patch_simulation(monkeypatch, slc)
    vis = VisMap()

    def count():
        return sum("mesh A" in record.getMessage() for record in caplog.records)

    with caplog.at_level("WARNING", logger="fdsvismap.FDSVisMap"):
        vis.read_fds_data("l_corridor", fds_slc_index=0)
        for time in (0.0, 10.0, 0.0):
            vis.get_extco_array_at_time(time)
        assert count() == 1
        vis.read_fds_data("l_corridor", fds_slc_index=0)
        vis.get_extco_array_at_time(10.0)
        vis.get_extco_array_at_time(0.0)
        assert count() == 2


def test_grid_of_unexpected_shape_is_skipped_with_a_warning(monkeypatch, caplog):
    slc = l_slice()
    padded = np.pad(slc.to_global(), ((0, 0), (0, 1), (0, 0)), constant_values=np.nan)
    slc.to_global = lambda: padded.copy()
    with caplog.at_level("WARNING", logger="fdsvismap.FDSVisMap"):
        extco = read(monkeypatch, slc)
    assert "Slice fake_slice" in caplog.text and "(6, 5)" in caplog.text
    np.testing.assert_array_equal(extco, padded[0])


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_node_in_two_meshes_takes_the_mesh_with_fewest_faces(monkeypatch, order):
    """(1, 1) is on one face of A (y = 1) and on two of B (x = 1, y = 1), a corner of B.

    B's data is offset by 100, so A's value 11 + t and B's 111 + t differ; A's is taken in both orders.
    """
    boxes = dict(L_MESHES_BY_ID)
    slc = FakeSlice(
        [(mesh_id, *boxes[mesh_id]) for mesh_id in order],
        field,
        step=0.5,
        cell_centered=False,
        times=TIMES,
        offsets={"B": 100.0},
    )
    patch_simulation(monkeypatch, slc)
    vis = VisMap()
    vis.read_fds_data("l_corridor", fds_slc_index=0)
    for time in TIMES:
        assert vis.get_extco_array_at_time(time)[node(1.0, 1.0)] == 11 + time


@pytest.mark.parametrize("order, expected", [("ABC", 12.0), ("CBA", 212.0)])
def test_tie_goes_to_the_first_subslice(monkeypatch, order, expected):
    """(2, 1) is on two faces of A (x = 2, y = 1) and two of C (x = 2, y = 1): the first subslice wins.

    A: x 0..2, y 0..1; B: x 0..1, y 1..2 (offset 100); C: x 2..3, y 0..1 (offset 200).
    """
    boxes = dict(L_MESHES_BY_ID, C=(2.0, 3.0, 0.0, 1.0))
    slc = FakeSlice(
        [(mesh_id, *boxes[mesh_id]) for mesh_id in order],
        field,
        step=0.5,
        cell_centered=False,
        offsets={"B": 100.0, "C": 200.0},
    )
    assert np.isnan(slc.to_global()[0, 4, 2])
    assert read(monkeypatch, slc)[4, 2] == expected


def test_corner_touching_meshes(monkeypatch):
    """A's faces x = 1 and y = 1 are recovered, except the corner (1, 1), which B writes."""
    slc = FakeSlice(
        [("A", 0.0, 1.0, 0.0, 1.0), ("B", 1.0, 2.0, 1.0, 2.0)],
        field,
        step=0.5,
        cell_centered=False,
    )
    raw = slc.to_global()[0]
    extco = read(monkeypatch, slc)
    recovered = [(1.0, 0.0), (1.0, 0.5), (0.0, 1.0), (0.5, 1.0)]
    assert all(np.isnan(raw[node(x, y)]) for x, y in recovered)
    for x, y in recovered:
        assert extco[node(x, y)] == x + 10 * y
    written = ~np.isnan(raw)
    np.testing.assert_array_equal(extco[written], raw[written])
    gx, gy = np.meshgrid(COORDS, COORDS, indexing="ij")
    outside = (gx > 1.0) & (gy < 1.0) | (gx < 1.0) & (gy > 1.0)
    assert outside.sum() == 8
    assert np.isnan(extco[outside]).all()
    assert np.isnan(extco).sum() == 8
