"""Basic tests for fdsvismap example script."""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fdsvismap import MapStyle, VisMap

matplotlib.use("Agg")


warnings.filterwarnings("ignore", message="no explicit representation of timezones")


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def vis_map(project_root):
    """Create a VisMap instance with test data."""

    sim_dir = project_root / "examples" / "room_fire" / "fds_data"
    bg_img = project_root / "examples" / "room_fire" / "misc" / "floorplan.png"

    vis = VisMap()
    vis.read_fds_data(str(sim_dir), fds_slc_height=2)

    if bg_img is not None:
        vis.add_background_image(bg_img)

    # Set up waypoints
    vis.set_start_point(1, 9)
    vis.set_waypoint(1, 8.4, 4.8, 3, 0)
    vis.set_waypoint(2, 9.8, 4, 3, 270)
    vis.set_waypoint(3, 17, 10, 3, 180)

    # Set time points
    times = range(0, 500, 50)
    vis.set_time_points(times)

    # Add visual obstruction
    vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

    # Compute
    vis.compute_all()

    return vis


class TestVisMapBasics:
    """Basic functionality tests."""

    def test_vismap_creation(self):
        """Test that VisMap instance can be created."""
        vis = VisMap()
        assert vis is not None

    def test_fds_data_reading(self, project_root):
        """Test that FDS data can be read."""
        # Try to find fds_data in common locations
        possible_paths = [
            project_root / "examples" / "room_fire" / "fds_data",
            project_root / "fds_data",
        ]

        sim_dir = None
        for path in possible_paths:
            if path.exists():
                sim_dir = str(path)
                break

        if sim_dir is None:
            pytest.skip("FDS data directory not found")

        vis = VisMap()
        vis.read_fds_data(str(sim_dir), fds_slc_height=2)
        assert vis is not None

    def test_background_image_loading(self, project_root):
        """Test that background image can be loaded."""
        bg_img = project_root / "examples" / "room_fire" / "misc" / "floorplan.png"
        if bg_img.exists():
            vis = VisMap()
            vis.add_background_image(bg_img)
            assert vis is not None
        else:
            pytest.skip("Background image not found")

    def test_background_image_extent(self, project_root):
        """Test that the extent of the background image is stored and its order is checked."""
        bg_img = project_root / "examples" / "room_fire" / "misc" / "floorplan.png"
        vis = VisMap()
        vis.add_background_image(bg_img, extent=(-2, 22, -1, 11))
        assert vis.background_extent == (-2, 22, -1, 11)

        # x_min and x_max swapped
        with pytest.raises(ValueError):
            vis.add_background_image(bg_img, extent=(22, -2, -1, 11))


class TestWaypoints:
    """Tests for waypoint functionality."""

    def test_set_start_point(self):
        """Test setting start point."""
        vis = VisMap()
        vis.set_start_point(1, 9)
        # If no error is raised, test passes
        assert True

    def test_set_waypoint(self):
        """Test setting waypoints."""
        vis = VisMap()
        vis.set_waypoint(1, 8.4, 4.8, 3, 0)
        vis.set_waypoint(2, 9.8, 4, 3, 270)
        vis.set_waypoint(3, 17, 10, 3, 180)
        # If no error is raised, test passes
        assert True


class TestVisibilityCalculations:
    """Tests for visibility calculations."""

    def test_waypoint_visibility(self, vis_map):
        """Test waypoint visibility check."""
        time = 450
        x, y = 2, 4
        waypoint_id = 2
        # TODO: Change this function later to return bool
        result = vis_map.wp_is_visible(time, x, y, waypoint_id)
        # Based on the example output, waypoint 2 is NOT visible at these coordinates
        assert isinstance(result, (bool, np.bool_))

    @pytest.mark.parametrize(
        "time,x,y,waypoint_id,expected",
        [
            (450, 2, 4, 2, False),  # From example: not visible
            # TODO: Add more test cases as you discover expected values:
            # (100, 5, 5, 1, True),  # Example: visible case
            # (300, 10, 8, 3, False),  # Example: not visible
        ],
    )
    def test_waypoint_visibility_scenarios(
        self, vis_map, time, x, y, waypoint_id, expected
    ):
        """Test waypoint visibility under various scenarios."""
        result = vis_map.wp_is_visible(time, x, y, waypoint_id)
        result = bool(result)
        assert result is expected, (
            f"Expected waypoint {waypoint_id} visibility to be {expected} at t={time}, ({x},{y})"
        )

    def test_distance_to_waypoint(self, vis_map):
        """Test distance calculation to waypoint."""
        x, y = 2, 4
        waypoint_id = 2

        distance = vis_map.get_distance_to_wp(x, y, waypoint_id)
        assert isinstance(distance, (int, float))
        assert distance >= 0
        # Waypoint 2 is at (9.8, 4), so distance from (2, 4) should be ~7.8m
        assert 7.5 < distance < 8.0, f"Expected distance ~7.8m, got {distance}m"

    def test_local_visibility(self, vis_map):
        """Test local visibility calculation."""
        time = 500
        x, y = 2, 4
        c = 3

        visibility = vis_map.get_local_visibility(time, x, y, c)
        assert isinstance(visibility, (int, float))
        assert visibility >= 0, "Visibility should be non-negative"
        # Visibility should be reasonable (not infinite, typically < 100m)
        assert visibility < 100, f"Visibility {visibility}m seems unreasonably high"

    def test_visibility_to_waypoint(self, vis_map):
        """Test visibility to waypoint calculation."""
        time = 500
        x, y = 2, 4
        waypoint_id = 2

        visibility = vis_map.get_visibility_to_wp(time, x, y, waypoint_id)
        assert isinstance(visibility, (int, float))
        assert visibility >= 0, "Visibility should be non-negative"
        # Visibility should be reasonable
        assert visibility < 100, f"Visibility {visibility}m seems unreasonably high"


class TestPlotGeneration:
    """Tests for plot generation."""

    def test_aset_map_plot_creation(self, vis_map, tmp_path):
        """Test ASET map plot creation."""
        fig, ax = vis_map.create_aset_map_plot(plot_obstructions=True)
        assert fig is not None
        assert ax is not None

        # Test saving
        output_file = tmp_path / "test_aset_map.pdf"
        plt.savefig(output_file, dpi=300)
        plt.close()

        assert output_file.exists()

    def test_time_agg_vismap_plot_creation(self, vis_map, tmp_path):
        """Test time aggregated vismap plot creation."""
        fig, ax = vis_map.create_time_agg_wp_agg_vismap_plot()
        assert fig is not None
        assert ax is not None

        # Test saving
        output_file = tmp_path / "test_time_agg_vismap.pdf"
        plt.savefig(output_file, dpi=300)
        plt.close()

        assert output_file.exists()

    def test_vismap_plot_creation(self, vis_map, tmp_path):
        """Test vismap plots at a single time point in subplots."""
        fig, axes = plt.subplots(1, 2)
        returned_fig, ax = vis_map.plot_vismap(300, ax=axes[0])
        assert returned_fig is fig
        assert ax is axes[0]
        vis_map.plot_vismap(300, waypoint_id=2, ax=axes[1], colorbar=False)

        # The plotted maps are the computed vismaps, also for waypoint IDs starting at 1
        aggregated_map = np.asarray(axes[0].get_images()[-1].get_array())
        waypoint_map = np.asarray(axes[1].get_images()[-1].get_array())
        np.testing.assert_array_equal(
            aggregated_map.astype(bool), vis_map.get_wp_agg_vismap(300)
        )
        np.testing.assert_array_equal(
            waypoint_map.astype(bool), vis_map.get_vismap(2, 300)
        )

        # Two maps and one colorbar
        assert len(fig.axes) == 3

        output_file = tmp_path / "test_vismap.pdf"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)

        assert output_file.exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time": 300, "waypoint_id": 7},  # unknown waypoint ID
            {"time": 600},  # after the last computed time point
        ],
    )
    def test_vismap_plot_invalid_input(self, vis_map, kwargs):
        """Test that invalid input for vismap plots raises a ValueError."""
        with pytest.raises(ValueError):
            vis_map.plot_vismap(**kwargs)
        plt.close("all")

    def test_map_plot_options(self, vis_map):
        """Test the base plot function with colorbar settings and arrays of the wrong shape."""
        fig, ax = vis_map.plot_map(
            vis_map.get_aset_map(),
            cmap="jet_r",
            cbar_kwargs={"label": "Time / s", "pad": 0.05},
        )
        # Map and colorbar
        assert len(fig.axes) == 2
        plt.close(fig)

        # The extinction coefficients are stored as (nx, ny) and have to be transposed
        extco_array = vis_map.get_extco_array_at_time(300)
        fig, ax = vis_map.plot_map(extco_array.T, colorbar=False)
        assert len(fig.axes) == 1
        plt.close(fig)
        with pytest.raises(ValueError):
            vis_map.plot_map(extco_array)
        plt.close("all")

    def test_background_image_extent_in_plot(self, vis_map, project_root):
        """Test that the background image is placed at its extent while the axes show the simulation domain."""
        bg_img = project_root / "examples" / "room_fire" / "misc" / "floorplan.png"
        vis_map.add_background_image(bg_img, extent=(-2, 22, -1, 11))
        fig, ax = vis_map.plot_vismap(300)

        assert tuple(ax.get_images()[0].get_extent()) == (-2, 22, -1, 11)
        assert ax.get_xlim() == pytest.approx((0, 20), abs=1e-6)
        assert ax.get_ylim() == pytest.approx((0, 10), abs=1e-6)
        plt.close(fig)

    def test_style_colors(self, vis_map):
        """Test that the plots use the colors of the style and that changes of the style are applied."""
        default = MapStyle()
        fig, ax = vis_map.plot_vismap(300)
        colors = [mcolors.to_hex(c) for c in ax.get_images()[-1].cmap.colors]
        assert colors == [default.not_visible, default.visible]
        plt.close(fig)

        vis_map.style.visible = "#2e7d32"
        fig, ax = vis_map.plot_vismap(300)
        assert mcolors.to_hex(ax.get_images()[-1].cmap.colors[1]) == "#2e7d32"
        plt.close(fig)

        # ASET map scaled from 0 to the maximum time, never visible cells in their own color
        fig, ax = vis_map.create_aset_map_plot()
        image = ax.get_images()[-1]
        never_visible = ~np.logical_or.reduce(vis_map.all_time_wp_agg_vismap_list)
        assert (image.norm.vmin, image.norm.vmax) == (0, 450)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(image.get_array()), never_visible
        )
        assert mcolors.to_hex(image.cmap.get_bad()) == default.never_visible
        plt.close(fig)


class TestFullExample:
    """Test the full example workflow."""

    def test_complete_example_workflow(self, project_root, tmp_path):
        """Test that the complete example script workflow runs without errors."""

        sim_dir = project_root / "examples" / "room_fire" / "fds_data"

        bg_img = project_root / "examples" / "room_fire" / "misc" / "floorplan.png"

        # Create instance
        vis = VisMap()

        # Read data
        vis.read_fds_data(str(sim_dir), fds_slc_height=2)

        # Add background if available
        if bg_img.exists():
            vis.add_background_image(bg_img, extent=(0, 20, 0, 10))

        # Set waypoints
        vis.set_start_point(1, 9)
        vis.set_waypoint(1, 8.4, 4.8, 3, 0)
        vis.set_waypoint(2, 9.8, 4, 3, 270)
        vis.set_waypoint(3, 17, 10, 3, 180)

        # Set time points
        times = range(0, 500, 50)
        vis.set_time_points(times)

        # Add obstruction
        vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

        # Compute
        vis.compute_all()

        # Create plots
        fig1, ax1 = vis.create_aset_map_plot(plot_obstructions=True)
        plt.savefig(tmp_path / "test_aset_map.pdf", dpi=300)
        plt.close()

        fig2, ax2 = vis.create_time_agg_wp_agg_vismap_plot()
        plt.savefig(tmp_path / "test_time_agg_vismap.pdf", dpi=300)
        plt.close()

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        vis.plot_vismap(300, ax=axes[0])
        vis.plot_vismap(300, waypoint_id=2, ax=axes[1])
        fig3.savefig(tmp_path / "test_vismap_300s.pdf", dpi=300)
        plt.close(fig3)

        # Test local evaluations
        time = 450
        x, y = 2, 4
        waypoint_id = 2

        wp_is_visible = vis.wp_is_visible(time, x, y, waypoint_id)
        wp_is_visible = bool(wp_is_visible)
        distance = vis.get_distance_to_wp(x, y, waypoint_id)
        local_visibility = vis.get_local_visibility(time, x, y, 3)
        visibility = vis.get_visibility_to_wp(time, x, y, waypoint_id)

        # Basic assertions
        assert isinstance(wp_is_visible, bool)
        assert distance >= 0
        assert local_visibility >= 0
        assert visibility >= 0

        # Check output files exist
        assert (tmp_path / "test_aset_map.pdf").exists()
        assert (tmp_path / "test_time_agg_vismap.pdf").exists()
        assert (tmp_path / "test_vismap_300s.pdf").exists()
