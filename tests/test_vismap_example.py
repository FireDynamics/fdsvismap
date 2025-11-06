"""Basic tests for fdsvismap example script."""

import pytest
from pathlib import Path
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from fdsvismap import VisMap

import warnings

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
        print(f"{sim_dir = }")
        print(f"sim_dir exists: {Path(sim_dir).exists()}")
        print("Contents of sim_dir:")
        if Path(sim_dir).exists():
            for item in sorted(Path(sim_dir).iterdir()):
                print(f"  {item.name}")

            # Look for .svm file
            svm_files = list(Path(sim_dir).glob("*.svm"))
            print(f"SVM files found: {len(svm_files)}")
            if svm_files:
                svm_file = svm_files[0]
                print(f"Reading first 10 lines of {svm_file.name}:")
                with open(svm_file, "r") as f:
                    for i, line in enumerate(f):
                        if i < 10:
                            print(f"  {line.rstrip()}")
                        else:
                            break
        else:
            print(f"  Directory does not exist!")

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
        time = 500
        x, y = 2, 4
        waypoint_id = 2
        # TODO: Change this function later to return bool
        result = vis_map.wp_is_visible(time, x, y, waypoint_id)
        # Based on the example output, waypoint 2 is NOT visible at these coordinates
        assert isinstance(result, (bool, np.bool_))

    @pytest.mark.parametrize(
        "time,x,y,waypoint_id,expected",
        [
            (500, 2, 4, 2, False),  # From example: not visible
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
            vis.add_background_image(bg_img)

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

        # Test local evaluations
        time = 500
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
