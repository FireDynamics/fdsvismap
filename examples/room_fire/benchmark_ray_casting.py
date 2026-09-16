"""Benchmark old vs new ray casting using the room_fire FDS example data."""

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from skimage.draw import line

from fdsvismap import VisMap
from fdsvismap.helper_functions import get_id_of_closest_value

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Int32Array = NDArray[np.int32]


def old_ray_casting(
    extco_array: FloatArray,
    ref_x_id: int,
    ref_y_id: int,
    non_concealed_x_idx: IntArray,
    non_concealed_y_idx: IntArray,
) -> FloatArray:
    mean_extco_array = np.zeros_like(extco_array)
    for x_id, y_id in zip(non_concealed_x_idx, non_concealed_y_idx):
        img = np.zeros_like(extco_array)
        x_lp_idx, y_lp_idx = line(ref_x_id, ref_y_id, x_id, y_id)
        img[x_lp_idx, y_lp_idx] = 1
        n_cells = len(x_lp_idx)
        mean_extco = np.sum(extco_array * img) / n_cells
        mean_extco_array[x_id, y_id] = mean_extco
    return mean_extco_array


def new_ray_casting(
    extco_array: FloatArray,
    ray_cells_flat_idx: Int32Array,
    ray_start_idx: IntArray,
    ray_cell_counts: IntArray,
    non_concealed_x_idx: IntArray,
    non_concealed_y_idx: IntArray,
) -> FloatArray:
    mean_extco_array = np.zeros_like(extco_array)
    ray_extco_sums = np.add.reduceat(
        extco_array.ravel()[ray_cells_flat_idx], ray_start_idx
    )
    mean_extco_array[non_concealed_x_idx, non_concealed_y_idx] = (
        ray_extco_sums / ray_cell_counts
    )
    return mean_extco_array


def benchmark() -> None:
    project_root = Path(__file__).parent
    sim_dir = str(project_root / "fds_data")

    # Set up VisMap exactly as in room_fire.py
    vis = VisMap()
    vis.read_fds_data(sim_dir, fds_slc_height=2)
    vis.add_sign(1, 8.4, 4.8, 3, 0)
    vis.add_sign(2, 9.8, 4, 3, 270)
    vis.add_sign(3, 17, 10, 3, 180)
    vis.add_visual_obstruction(8, 8.8, 4.6, 4.8)

    times = list(range(0, 500, 50))
    vis.set_time_points(times)

    # Build help arrays (populates caches and non-concealed indices)
    vis.build_help_arrays(obstructions=True, view_angle=True, aa=True)

    n_timesteps = len(times)

    for sign_id, sign in vis.all_sign_dict.items():
        ref_x_id = get_id_of_closest_value(vis.all_x_coords, sign.x)
        ref_y_id = get_id_of_closest_value(vis.all_y_coords, sign.y)

        cache = vis.all_sign_ray_casting_cache_dict[sign_id]
        ray_cells_flat_idx = cache["ray_cells_flat_idx"]
        ray_start_idx = cache["ray_start_idx"]
        ray_cell_counts = cache["ray_cell_counts"]
        non_concealed_x_idx = cache["non_concealed_x_idx"]
        non_concealed_y_idx = cache["non_concealed_y_idx"]

        n_visible = len(non_concealed_x_idx)
        grid_shape = (len(vis.all_x_coords), len(vis.all_y_coords))

        # Correctness check with first timestep
        extco_array = vis.get_extco_array_at_time(times[0])
        old_result = old_ray_casting(
            extco_array, ref_x_id, ref_y_id, non_concealed_x_idx, non_concealed_y_idx
        )
        new_result = new_ray_casting(
            extco_array,
            ray_cells_flat_idx,
            ray_start_idx,
            ray_cell_counts,
            non_concealed_x_idx,
            non_concealed_y_idx,
        )
        assert np.allclose(old_result, new_result), "Results differ!"

        print(
            f"Sign {sign_id}: grid {grid_shape[0]}x{grid_shape[1]}, "
            f"visible cells: {n_visible}, timesteps: {n_timesteps}"
        )

        start = time.perf_counter()
        for t in times:
            extco_array = vis.get_extco_array_at_time(t)
            old_ray_casting(
                extco_array,
                ref_x_id,
                ref_y_id,
                non_concealed_x_idx,
                non_concealed_y_idx,
            )
        old_time = time.perf_counter() - start

        start = time.perf_counter()
        for t in times:
            extco_array = vis.get_extco_array_at_time(t)
            new_ray_casting(
                extco_array,
                ray_cells_flat_idx,
                ray_start_idx,
                ray_cell_counts,
                non_concealed_x_idx,
                non_concealed_y_idx,
            )
        new_time = time.perf_counter() - start

        speedup = old_time / new_time
        print(f"  Old (no cache): {old_time:.3f}s")
        print(f"  New (cached):   {new_time:.3f}s")
        print(f"  Speedup:       {speedup:.1f}x")
        print()


if __name__ == "__main__":
    benchmark()
