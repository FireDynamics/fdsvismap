"""
Selection of the FDS slice by height and by index.

``tests/data/slice_order`` holds unnamed slices of one quantity: a vertical
slice through x = y = 0 first, then horizontal slices at z = 2.75, 1.75 and
0.75 m, from top to bottom. The input is ``slice_order.fds`` in that folder.
"""

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
