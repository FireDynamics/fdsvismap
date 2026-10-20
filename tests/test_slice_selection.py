"""
Selection of the FDS slice by height, index and ID, and the checks of the selected slice.

``tests/data/slice_order`` holds unnamed slices of one quantity: a vertical
slice through x = y = 0 first, then horizontal slices at z = 2.75, 1.75 and
0.75 m, from top to bottom. The input is ``slice_order.fds`` in that folder.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from fdsvismap import VisMap

SIM_DIR = str(Path(__file__).parent / "data" / "slice_order")


def read(**kwargs):
    vis = VisMap()
    vis.read_fds_data(SIM_DIR, **kwargs)
    return vis


@pytest.mark.parametrize(
    "height, expected_z",
    [(0.5, 0.75), (1.5, 1.75), (2.5, 2.75), (2.0, 1.75), (10.0, 2.75)],
)
def test_height_selects_the_nearest_horizontal_slice(height, expected_z):
    vis = read(fds_slc_height=height)
    assert vis.slc.orientation == 3
    assert vis.slc.extent.z_start == pytest.approx(expected_z)


def test_extinction_follows_the_selected_height():
    """The smoke layer makes K grow with height."""
    means = [
        np.mean(read(fds_slc_height=h).get_extco_array_at_time(20.0))
        for h in (0.5, 1.5, 2.5)
    ]
    assert means[0] < means[1] < means[2]


def test_tie_goes_to_the_slice_declared_first():
    """1.25 m lies halfway between 0.75 m and 1.75 m; 1.75 m is declared first."""
    assert read(fds_slc_height=1.25).slc.extent.z_start == pytest.approx(1.75)


def test_index_selects_an_unnamed_slice():
    vis = read(fds_slc_index=3)
    assert vis.slc.orientation == 3
    assert vis.slc.extent.z_start == pytest.approx(0.75)
    assert vis.get_extco_array_at_time(20.0).shape == (16, 16)


def test_index_takes_precedence_over_height():
    assert read(fds_slc_index=1, fds_slc_height=0.5).slc.extent.z_start == (
        pytest.approx(2.75)
    )


@pytest.mark.parametrize("index", [4, -1])
def test_invalid_index_lists_the_slices_with_their_index(index):
    with pytest.raises(ValueError, match=r"\[3\] \(no ID\).*z = 0\.75 m"):
        read(fds_slc_index=index)


def test_a_vertical_slice_is_rejected():
    with pytest.raises(
        ValueError, match=r"(?s)not horizontal.*\[0\] \(no ID\).*y = 0\.00 m"
    ):
        read(fds_slc_index=0)


def test_a_rejected_slice_keeps_the_slice_read_before():
    vis = read(fds_slc_index=3)
    with pytest.raises(ValueError):
        vis.read_fds_data(SIM_DIR, fds_slc_index=0)
    assert vis.slc.extent.z_start == pytest.approx(0.75)


@pytest.mark.parametrize(
    "slc_id, quantity",
    [("slcf_temperature", "TEMPERATURE"), ("slcf_visibility", "SOOT VISIBILITY")],
)
def test_a_slice_of_another_quantity_is_rejected(slc_id, quantity):
    room_fire = Path(__file__).parents[1] / "examples" / "room_fire" / "fds_data"
    with pytest.raises(ValueError, match=f"holds '{quantity}', which is neither"):
        VisMap().read_fds_data(str(room_fire), fds_slc_id=slc_id)


def relabel(tmp_path, name, short_name):
    """Copy the fixture and give its slices another quantity, with the same values."""
    sim_dir = tmp_path / "slice_order"
    shutil.copytree(SIM_DIR, sim_dir, ignore=shutil.ignore_patterns("*.pickle"))
    smv = sim_dir / "slice_order.smv"
    smv.write_text(
        smv.read_text().replace(
            " SOOT EXTINCTION COEFFICIENT\n ext_coef_C\n", f" {name}\n {short_name}\n"
        )
    )
    return str(sim_dir)


def test_optical_density_is_converted_however_the_slice_is_selected(tmp_path):
    sim_dir = relabel(tmp_path, "SOOT OPTICAL DENSITY", "OD_C")
    by_height = VisMap()
    by_height.quantity = "OD_C"
    by_height.read_fds_data(sim_dir, fds_slc_height=0.5)
    by_index = VisMap()
    by_index.read_fds_data(sim_dir, fds_slc_index=3)
    extinction = read(fds_slc_index=3).get_extco_array_at_time(20.0)

    assert by_height.slc.extent.z_start == by_index.slc.extent.z_start
    for vis in (by_height, by_index):
        np.testing.assert_allclose(
            vis.get_extco_array_at_time(20.0), np.log(10) * extinction
        )


def test_a_smoke_species_is_read_as_extinction_coefficient(tmp_path):
    sim_dir = relabel(tmp_path, "MY SMOKE EXTINCTION COEFFICIENT", "ext_coef_MY_SMOKE")
    vis = VisMap()
    vis.read_fds_data(sim_dir, fds_slc_index=3)
    np.testing.assert_allclose(
        vis.get_extco_array_at_time(20.0),
        read(fds_slc_index=3).get_extco_array_at_time(20.0),
    )
