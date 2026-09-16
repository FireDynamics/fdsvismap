"""Basic tests for fdsvismap example script."""

import itertools
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

    # Set up the signs and the route they guide along
    vis.add_sign(1, 8.4, 4.8, 3, 0)
    vis.add_sign(2, 9.8, 4, 3, 270)
    vis.add_sign(3, 17, 10, 3, 180)
    vis.add_route("exit route", [(1, 9), (7, 5.5), (9.5, 4.2), (17, 9.5)])

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


class TestTimePoints:
    """Tests for the time points the maps are computed at."""

    @pytest.fixture
    def sim_dir(self, project_root):
        """Get the directory of the FDS output of the example."""
        return str(project_root / "examples" / "room_fire" / "fds_data")

    def computed(self, sim_dir, times):
        """Compute the maps of two signs at the given time points."""
        vis = VisMap()
        vis.read_fds_data(sim_dir, fds_slc_height=2)
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(2, 9.8, 4, 3, 270)
        vis.set_time_points(times)
        vis.compute_all()
        return vis

    def test_time_points_are_sorted(self, sim_dir):
        """Test that unsorted time points give the same maps as sorted ones."""
        ordered = self.computed(sim_dir, [0, 150, 300, 450])
        shuffled = self.computed(sim_dir, [300, 0, 450, 150])

        np.testing.assert_array_equal(
            shuffled.vismap_time_points, [0.0, 150.0, 300.0, 450.0]
        )
        np.testing.assert_array_equal(ordered.get_aset_map(), shuffled.get_aset_map())
        for time in (0, 150, 300, 450):
            np.testing.assert_array_equal(
                ordered.get_agg_vismap(time), shuffled.get_agg_vismap(time)
            )

        # The maximum time is the latest one, not the last one that was passed
        assert shuffled.get_aset_map().max() == 450
        with pytest.raises(ValueError):
            shuffled.get_agg_vismap(500)

    def test_duplicate_time_points_are_dropped(self, sim_dir):
        """Test that a time point given twice is computed once."""
        vis = self.computed(sim_dir, [0, 300, 300, 150])
        np.testing.assert_array_equal(vis.vismap_time_points, [0.0, 150.0, 300.0])
        assert len(vis.all_time_sign_agg_vismap_list) == 3


class TestVisibilityCalculations:
    """Tests for visibility calculations."""

    def test_sign_visibility(self, vis_map):
        """Test the visibility check of a sign."""
        time = 450
        x, y = 2, 4
        sign_id = 2
        # TODO: Change this function later to return bool
        result = vis_map.sign_is_visible(time, x, y, sign_id)
        # Based on the example output, sign 2 is NOT visible at these coordinates
        assert isinstance(result, (bool, np.bool_))

    @pytest.mark.parametrize(
        "time,x,y,sign_id,expected",
        [
            (450, 2, 4, 2, False),  # From example: not visible
            # TODO: Add more test cases as you discover expected values:
            # (100, 5, 5, 1, True),  # Example: visible case
            # (300, 10, 8, 3, False),  # Example: not visible
        ],
    )
    def test_sign_visibility_scenarios(self, vis_map, time, x, y, sign_id, expected):
        """Test the visibility of a sign under various scenarios."""
        result = vis_map.sign_is_visible(time, x, y, sign_id)
        result = bool(result)
        assert result is expected, (
            f"Expected sign {sign_id} visibility to be {expected} at t={time}, ({x},{y})"
        )

    def test_distance_to_sign(self, vis_map):
        """Test the distance calculation to a sign."""
        x, y = 2, 4
        sign_id = 2

        distance = vis_map.get_distance_to_sign(x, y, sign_id)
        assert isinstance(distance, (int, float))
        assert distance >= 0
        # Sign 2 is at (9.8, 4), so distance from (2, 4) should be ~7.8m
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

    def test_visibility_to_sign(self, vis_map):
        """Test the visibility calculation to a sign."""
        time = 500
        x, y = 2, 4
        sign_id = 2

        visibility = vis_map.get_visibility_to_sign(time, x, y, sign_id)
        assert isinstance(visibility, (int, float))
        assert visibility >= 0, "Visibility should be non-negative"
        # Visibility should be reasonable
        assert visibility < 100, f"Visibility {visibility}m seems unreasonably high"


class TestPartialComputation:
    """Tests for evaluations after compute_all was limited with t_max."""

    @pytest.fixture
    def partial_map(self, project_root):
        """Compute the maps of two signs up to 200 s, although later time points are set."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(2, 9.8, 4, 3, 270)
        vis.set_time_points(range(0, 500, 50))
        vis.compute_all(t_max=200)
        return vis

    def test_time_aggregation_uses_the_computed_time_points(self, partial_map):
        """Test that the aggregation over time is limited to the computed time points."""
        computed = [0, 50, 100, 150, 200]
        expected = np.logical_and.reduce(
            [partial_map.get_agg_vismap(time) for time in computed]
        )
        np.testing.assert_array_equal(partial_map.get_time_agg_vismap(), expected)

        # A shorter period uses its time points only, a longer one is rejected
        np.testing.assert_array_equal(
            partial_map.get_time_agg_vismap(100),
            np.logical_and.reduce(
                [partial_map.get_agg_vismap(time) for time in (0, 50, 100)]
            ),
        )
        with pytest.raises(ValueError):
            partial_map.get_time_agg_vismap(300)

    def test_time_aggregated_plot(self, partial_map):
        """Test that the plot of the time aggregated map works without a time of its own."""
        fig, ax = partial_map.plot_time_agg_vismap()
        np.testing.assert_array_equal(
            np.asarray(ax.get_images()[-1].get_array()).astype(bool),
            partial_map.get_time_agg_vismap(),
        )
        plt.close(fig)


class TestRoutesPlot:
    """Tests for the plot of the routes and signs without a visibility map."""

    @pytest.fixture
    def layout(self, project_root):
        """Create signs and two routes that share a sign, without computing anything."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign("A", 8.4, 4.8, 3, 0)
        vis.add_sign("B", 9.8, 4, 3, 270)
        vis.add_sign("C", 17, 10, 3, 180)
        vis.add_route("Office", [(1, 9), (7, 5.5), (9.5, 4.2)], signs=["A", "B"])
        vis.add_route("Hall", [(12, 2), (15, 4), (17, 9.5)], signs=["C", "B"])
        return vis

    def test_all_routes_without_computation(self, layout):
        """Test that the routes are drawn in the color of the signs, with their names and the signs."""
        style = MapStyle()
        fig, ax = layout.plot_routes()

        # One dashed line per route, all in the same color
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        assert len(lines) == 2
        assert {mcolors.to_hex(line.get_color()) for line in lines} == {style.sign}
        np.testing.assert_array_equal(
            np.column_stack(lines[0].get_data()),
            layout.all_route_dict["Office"].waypoints,
        )

        # The name of each route is written on the map
        texts = [text.get_text() for text in ax.texts]
        assert "Office" in texts and "Hall" in texts

        # The legend names the routes a sign belongs to, the start points share one entry
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == [
            "C = 3, $\\alpha$ = 0$^\\circ$ (Office)",
            "C = 3, $\\alpha$ = 270$^\\circ$ (Office, Hall)",
            "C = 3, $\\alpha$ = 180$^\\circ$ (Hall)",
            "start point",
        ]
        plt.close(fig)

    def test_a_single_route_with_its_signs(self, layout):
        """Test that only the given route and the signs that belong to it are plotted."""
        fig, ax = layout.plot_routes(route_ids=["Hall"])
        assert (
            len([line for line in ax.get_lines() if line.get_linestyle() == "--"]) == 1
        )
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == [
            "C = 3, $\\alpha$ = 180$^\\circ$ (Hall)",
            "C = 3, $\\alpha$ = 270$^\\circ$ (Hall)",
            "start point",
        ]
        plt.close(fig)

        with pytest.raises(ValueError):
            layout.plot_routes(route_ids=["nowhere"])
        plt.close("all")

    def test_route_names_do_not_overlap(self, layout):
        """Test that the names of routes running along each other are placed apart."""
        # A second route on exactly the same line as Office
        layout.add_route("Twin", [(1, 9), (7, 5.5), (9.5, 4.2)], signs=["A"])
        fig, ax = layout.plot_routes()

        renderer = fig.canvas.get_renderer()
        names = {"Office", "Hall", "Twin"}
        name_boxes = [
            text.get_window_extent(renderer)
            for text in ax.texts
            if text.get_text() in names
        ]
        assert len(name_boxes) == 3
        for first, second in itertools.combinations(name_boxes, 2):
            assert not first.overlaps(second)

        # The names also keep clear of the IDs of the signs
        id_boxes = [
            text.get_window_extent(renderer)
            for text in ax.texts
            if text.get_text() not in names
        ]
        for name_box in name_boxes:
            assert not any(name_box.overlaps(id_box) for id_box in id_boxes)
        plt.close(fig)

    def test_route_name_avoids_the_id_of_a_sign(self, project_root):
        """Test that the name of a route moves when a sign is drawn where it would be written."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        # The sign sits in the middle of the longest section, where the name goes first
        vis.add_sign("M", 4.0, 7.25, 3, 0)
        vis.add_route("Office", [(1, 9), (7, 5.5), (9.5, 4.2)], signs=["M"])

        fig, ax = vis.plot_routes()
        renderer = fig.canvas.get_renderer()
        name = next(text for text in ax.texts if text.get_text() == "Office")
        assert name.get_position() != (4.0, 7.25)
        assert not any(
            name.get_window_extent(renderer).overlaps(text.get_window_extent(renderer))
            for text in ax.texts
            if text.get_text() == "M"
        )
        plt.close(fig)

    def test_no_map_is_drawn(self, layout, project_root):
        """Test that the plot shows the background image and an empty map, without a colorbar."""
        layout.add_background_image(
            project_root / "examples" / "room_fire" / "misc" / "floorplan.png",
            extent=(0, 20, 0, 10),
        )
        fig, ax = layout.plot_routes()

        background, empty_map = ax.get_images()
        assert tuple(background.get_extent()) == (0, 20, 0, 10)
        assert np.ma.getmaskarray(empty_map.get_array()).all()

        # No colorbar as a second axes
        assert len(fig.axes) == 1
        plt.close(fig)


class TestStaleResults:
    """Tests that results are discarded when their input changes."""

    @pytest.fixture
    def computed(self, project_root):
        """Compute the maps of two signs at four time points."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(2, 9.8, 4, 3, 270)
        vis.set_time_points([0, 100, 200, 300])
        vis.compute_all()
        return vis

    def test_new_time_points_discard_the_maps(self, computed):
        """Test that a map of an old time point is not returned for a new one."""
        computed.set_time_points([0, 50])
        with pytest.raises(RuntimeError):
            computed.get_agg_vismap(50)
        with pytest.raises(RuntimeError):
            computed.get_aset_map()

    def test_a_moved_sign_discards_its_arrays(self, computed):
        """Test that a sign that is added again with another position does not keep its old arrays."""
        computed.add_sign(1, 1.0, 1.0, 3, 0)
        with pytest.raises(RuntimeError):
            computed.get_sign_vismap(1, 100)

    def test_a_new_sign_discards_the_maps(self, computed):
        """Test that a sign added after the computation does not raise an IndexError."""
        computed.add_sign(3, 17, 10, 3, 180)
        with pytest.raises(RuntimeError):
            computed.sign_is_visible(100, 5, 5, 3)

    def test_a_new_obstruction_discards_the_maps(self, computed):
        """Test that an obstruction added after the computation invalidates the maps."""
        computed.add_visual_obstruction(8, 8.8, 4.6, 4.8)
        with pytest.raises(RuntimeError):
            computed.get_agg_vismap(100)

        # After computing again the maps are available, with the obstruction
        computed.compute_all()
        assert computed.get_agg_vismap(100).shape == computed.obstructions_array.shape

    def test_getters_before_compute_all(self, project_root):
        """Test that the getters name the missing computation instead of raising a KeyError."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.set_time_points([0, 100])
        for call in (
            lambda: vis.get_sign_vismap(1, 0),
            lambda: vis.get_agg_vismap(0),
            lambda: vis.get_visibility_to_sign(0, 2, 4, 1),
            lambda: vis.sign_is_visible(0, 2, 4, 1),
        ):
            with pytest.raises(RuntimeError):
                call()

        # An unknown ID is still a ValueError, not a missing computation
        with pytest.raises(ValueError):
            vis.get_sign_vismap("nowhere", 0)


class TestAsetMap:
    """Tests for the map of the first time without a visible sign."""

    @pytest.fixture
    def fractional_map(self, project_root):
        """Compute the maps at time points with decimals."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(2, 9.8, 4, 3, 270)
        vis.set_time_points([0, 112.5, 225, 337.5, 450])
        vis.compute_all()
        return vis

    def test_aset_map_keeps_the_time_points(self, fractional_map):
        """Test that the ASET map holds the first time without a sign, also with decimals."""
        times = np.asarray(fractional_map.vismap_time_points)
        not_visible = np.array([~fractional_map.get_agg_vismap(time) for time in times])
        # The first time point without a visible sign, or the maximum time if one stays visible
        expected = np.where(
            not_visible.any(axis=0), times[np.argmax(not_visible, axis=0)], times[-1]
        )

        aset_map = fractional_map.get_aset_map()
        assert aset_map.dtype == np.float64
        np.testing.assert_array_equal(aset_map, expected)

        # Times with decimals occur and are not truncated to full seconds
        assert set(np.unique(aset_map)) <= set(times)
        assert (aset_map % 1 != 0).any()

    def test_aset_map_of_a_part_of_the_time(self, fractional_map):
        """Test that a maximum time with decimals limits the map to the time points below it."""
        aset_map = fractional_map.get_aset_map(112.5)
        assert set(np.unique(aset_map)) <= {0.0, 112.5}
        np.testing.assert_array_equal(aset_map == 0, ~fractional_map.get_agg_vismap(0))


class TestPlotGeneration:
    """Tests for plot generation."""

    def test_aset_map_plot_creation(self, vis_map, tmp_path):
        """Test ASET map plot creation."""
        fig, ax = vis_map.plot_aset_map(plot_obstructions=True)
        assert fig is not None
        assert ax is not None

        # Test saving
        output_file = tmp_path / "test_aset_map.pdf"
        plt.savefig(output_file, dpi=300)
        plt.close()

        assert output_file.exists()

    def test_time_agg_vismap_plot_creation(self, vis_map, tmp_path):
        """Test time aggregated vismap plot creation."""
        fig, ax = vis_map.plot_time_agg_vismap()
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
        vis_map.plot_vismap(300, sign_id=2, ax=axes[1])

        # The plotted maps are the computed vismaps, also for sign IDs starting at 1
        aggregated_map = np.asarray(axes[0].get_images()[-1].get_array())
        sign_map = np.asarray(axes[1].get_images()[-1].get_array())
        np.testing.assert_array_equal(
            aggregated_map.astype(bool), vis_map.get_agg_vismap(300)
        )
        np.testing.assert_array_equal(
            sign_map.astype(bool), vis_map.get_sign_vismap(2, 300)
        )

        # Two maps, the colors of the map are in the legend instead of a colorbar
        assert len(fig.axes) == 2

        output_file = tmp_path / "test_vismap.pdf"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)

        assert output_file.exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time": 300, "sign_id": 7},  # unknown sign ID
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
        fig, ax = vis_map.plot_aset_map()
        image = ax.get_images()[-1]
        never_visible = ~np.logical_or.reduce(vis_map.all_time_sign_agg_vismap_list)
        assert (image.norm.vmin, image.norm.vmax) == (0, 450)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(image.get_array()), never_visible
        )
        assert mcolors.to_hex(image.cmap.get_bad()) == default.never_visible
        plt.close(fig)


class TestSignsAndRoutes:
    """Tests for the data model of signs and routes."""

    @pytest.fixture
    def route_map(self, project_root):
        """Create a VisMap with two named signs and two routes that share one of them."""
        vis = VisMap()
        vis.read_fds_data(
            str(project_root / "examples" / "room_fire" / "fds_data"), fds_slc_height=2
        )
        vis.add_sign("door", 8.4, 4.8, 3, 0)
        vis.add_sign("hall", 9.8, 4, 3, 270)
        vis.add_sign("exit", 17, 10, 3, 180)
        vis.add_route("west", [(1, 9), (8.4, 6), (9.8, 4.5)], signs=["door", "hall"])
        vis.add_route("east", [(15, 2), (17, 5), (17, 9.5)], signs=["exit", "hall"])
        vis.set_time_points(range(0, 500, 100))
        vis.compute_all()
        return vis

    def test_add_sign_and_route(self, route_map):
        """Test that signs and routes are stored with their IDs and parameters."""
        assert list(route_map.all_sign_dict) == ["door", "hall", "exit"]
        assert route_map.all_sign_dict["hall"].alpha == 270
        assert route_map.all_route_dict["west"].signs == ["door", "hall"]
        np.testing.assert_array_equal(
            route_map.all_route_dict["west"].waypoints,
            [[1, 9], [8.4, 6], [9.8, 4.5]],
        )
        assert route_map.all_route_dict["west"].length == pytest.approx(
            np.hypot(7.4, 3) + np.hypot(1.4, 1.5)
        )

    def test_omnidirectional_sign(self):
        """Test that a sign without a viewing direction is set explicitly."""
        vis = VisMap()
        vis.add_sign(1, 1, 1, 3, "omni")
        vis.add_sign(2, 2, 2, 3, None)
        assert vis.all_sign_dict[1].alpha is None
        assert vis.all_sign_dict[2].alpha is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"waypoints": [(1, 1)], "signs": ["door"]},  # only one waypoint, no line
            {"waypoints": [1, 2, 3], "signs": ["door"]},  # no pairs of coordinates
            {"waypoints": [(1, 1), (2, 2)], "signs": ["nowhere"]},  # unknown sign
        ],
    )
    def test_add_route_invalid_input(self, route_map, kwargs):
        """Test that an invalid route raises a ValueError."""
        with pytest.raises(ValueError):
            route_map.add_route("invalid", **kwargs)

    def test_route_vismap_aggregates_its_signs(self, route_map):
        """Test that the vismap of a route is the aggregation of the vismaps of its signs only."""
        route_vismap = route_map.get_agg_vismap(200, route_id="west")
        expected = route_map.get_sign_vismap("door", 200) | route_map.get_sign_vismap(
            "hall", 200
        )
        np.testing.assert_array_equal(route_vismap, expected)

        # The sign of the other route is not included, so the maps differ
        assert not np.array_equal(route_vismap, route_map.get_agg_vismap(200))

    def test_sign_is_visible_uses_the_id_not_the_position(self, route_map):
        """Test that a sign is looked up by its ID, also for IDs that are no list positions."""
        x, y = 8, 6
        for sign_id in route_map.all_sign_dict:
            expected = route_map.get_sign_vismap(sign_id, 200)[
                np.abs(route_map.all_y_coords - y).argmin(),
                np.abs(route_map.all_x_coords - x).argmin(),
            ]
            assert route_map.sign_is_visible(200, x, y, sign_id) == bool(expected)
        with pytest.raises(ValueError):
            route_map.sign_is_visible(200, x, y, "nowhere")

    def test_route_coverage(self, route_map):
        """Test that the coverage is sampled along the route and matches the vismap of the route."""
        points = route_map.get_route_points("west")
        coverage = route_map.get_route_coverage("west", 200)
        assert len(coverage) == len(points)

        # The points are sampled in the resolution of the grid
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert distances.max() <= min(route_map.cell_size) + 1e-9

        # Each point carries the value of the route vismap at its cell
        vismap = route_map.get_agg_vismap(200, route_id="west")
        for point, covered in zip(points, coverage):
            x_id = np.abs(route_map.all_x_coords - point[0]).argmin()
            y_id = np.abs(route_map.all_y_coords - point[1]).argmin()
            assert bool(vismap[y_id, x_id]) == bool(covered)

    def test_route_aset(self, route_map):
        """Test that the ASET along a route is the first time point without coverage per point."""
        aset = route_map.get_route_aset("west")
        assert len(aset) == len(route_map.get_route_points("west"))

        times = np.asarray(route_map.vismap_time_points)
        coverage = np.array(
            [route_map.get_route_coverage("west", time) for time in times]
        )
        # The first time point without coverage, or the maximum time if a sign stays visible
        expected = np.where(
            coverage.all(axis=0), times[-1], times[np.argmin(coverage, axis=0)]
        )
        np.testing.assert_array_equal(aset, expected)

        # Points that lose coverage before the end exist in this scenario
        assert (aset < times[-1]).any()

    def test_route_plots(self, route_map):
        """Test that the route plots draw the route in the colors of covered and uncovered sections."""
        style = MapStyle()
        fig, ax = route_map.plot_route_vismap("west", 200)
        np.testing.assert_array_equal(
            np.asarray(ax.get_images()[-1].get_array()).astype(bool),
            route_map.get_agg_vismap(200, route_id="west"),
        )
        collections = [
            c
            for c in ax.collections
            if isinstance(c, matplotlib.collections.LineCollection)
        ]
        assert len(collections) == 1
        colors = [mcolors.to_hex(c) for c in collections[0].get_colors()]

        # A section is covered if a sign is visible from both of its ends
        coverage = route_map.get_route_coverage("west", 200)
        expected = np.where(
            coverage[:-1] & coverage[1:], style.route_covered, style.route_uncovered
        )
        assert colors == list(expected)

        # The name of the route as title, the colors of the map and its signs as entries
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Route: west"
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == [
            "not visible",
            "visible",
            "C = 3, $\\alpha$ = 0$^\\circ$",
            "C = 3, $\\alpha$ = 270$^\\circ$",
            "start point",
        ]
        plt.close(fig)

        fig, ax = route_map.plot_aset_map(route_id="east")
        np.testing.assert_array_equal(
            np.asarray(ax.get_images()[-1].get_array()),
            route_map.get_aset_map(route_id="east"),
        )
        plt.close(fig)

    def test_plot_vismap_invalid_combination(self, route_map):
        """Test that a sign and a route at the same time raise a ValueError."""
        with pytest.raises(ValueError):
            route_map.plot_vismap(200, sign_id="door", route_id="west")
        plt.close("all")

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

        # Set the signs and the route
        vis.add_sign(1, 8.4, 4.8, 3, 0)
        vis.add_sign(2, 9.8, 4, 3, 270)
        vis.add_sign(3, 17, 10, 3, 180)
        vis.add_route("exit route", [(1, 9), (7, 5.5), (9.5, 4.2), (17, 9.5)])

        # Set time points
        times = range(0, 500, 50)
        vis.set_time_points(times)

        # Add obstruction
        vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

        # Plot the input before any computation
        fig0, ax0 = vis.plot_routes(plot_obstructions=True)
        plt.savefig(tmp_path / "test_routes.pdf", dpi=300)
        plt.close()

        # Compute
        vis.compute_all()

        # Create plots
        fig1, ax1 = vis.plot_aset_map(route_id="exit route", plot_obstructions=True)
        plt.savefig(tmp_path / "test_aset_map.pdf", dpi=300)
        plt.close()

        fig2, ax2 = vis.plot_time_agg_vismap(route_id="exit route")
        plt.savefig(tmp_path / "test_time_agg_vismap.pdf", dpi=300)
        plt.close()

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        vis.plot_route_vismap("exit route", 300, ax=axes[0])
        vis.plot_sign_vismap(2, 300, ax=axes[1])
        fig3.savefig(tmp_path / "test_vismap_300s.pdf", dpi=300)
        plt.close(fig3)

        # Test local evaluations
        time = 450
        x, y = 2, 4
        sign_id = 2

        sign_is_visible = bool(vis.sign_is_visible(time, x, y, sign_id))
        distance = vis.get_distance_to_sign(x, y, sign_id)
        local_visibility = vis.get_local_visibility(time, x, y, 3)
        visibility = vis.get_visibility_to_sign(time, x, y, sign_id)

        # Basic assertions
        assert isinstance(sign_is_visible, bool)
        assert distance >= 0
        assert local_visibility >= 0
        assert visibility >= 0

        # Check output files exist
        assert (tmp_path / "test_routes.pdf").exists()
        assert (tmp_path / "test_aset_map.pdf").exists()
        assert (tmp_path / "test_time_agg_vismap.pdf").exists()
        assert (tmp_path / "test_vismap_300s.pdf").exists()
