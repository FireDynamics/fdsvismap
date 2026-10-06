"""A horizontal FDS slice built from given meshes and a field, without FDS output.

The subslices, meshes and extents are fdsreader's own classes, and the slice is
assembled by fdsreader's own ``Slice.to_global()`` and ``get_coordinates()``.
Only reading the ``.sf`` files is replaced: each subslice holds the field
sampled at its nodes (node-based slice) or cell centres (cell-centred slice).
``patch_simulation`` makes ``VisMap.read_fds_data`` read such a slice.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from fdsreader.fds_classes.mesh import Mesh
from fdsreader.slcf.slice import Slice, SubSlice
from fdsreader.utils.extent import Extent

import fdsvismap.FDSVisMap

# (mesh ID, x_start, x_end, y_start, y_end), optionally followed by its own cell size
MeshBox = Tuple[Union[str, float], ...]
# K(x, y, t) in 1/m
Field = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class FakeSlice:
    """Horizontal slice at height *z* over *meshes*, with cell size *step* unless a mesh gives its own."""

    to_global = Slice.to_global
    get_coordinates = Slice.get_coordinates
    subslices = Slice.subslices
    orientation = 3
    id = "fake_slice"
    quantity = SimpleNamespace(name="SOOT EXTINCTION COEFFICIENT")

    def __init__(
        self,
        meshes: Sequence[MeshBox],
        field: Field,
        step: float,
        cell_centered: bool,
        times: Sequence[float] = (0.0,),
        z: float = 2.0,
        offsets: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cell_centered = cell_centered
        self.times = np.asarray(times, dtype=float)
        self.n_t = self.times.size
        self._field = field
        # Added to the field per mesh ID, so that meshes disagree at shared nodes
        self._offsets = offsets or {}
        self._subslices = {}
        for mesh_id, x0, x1, y0, y1, *own_step in meshes:
            mesh_step = own_step[0] if own_step else step
            nodes = {
                "x": np.linspace(x0, x1, round((x1 - x0) / mesh_step) + 1),
                "y": np.linspace(y0, y1, round((y1 - y0) / mesh_step) + 1),
                "z": np.array([0.0, z, 2 * z]),
            }
            mesh = Mesh(
                nodes, {"x": (x0, x1), "y": (y0, y1), "z": (0.0, 2 * z)}, mesh_id
            )
            subslice = SubSlice(self, "", None, Extent(x0, x1, y0, y1, z, z), mesh)
            self._subslices[mesh_id] = subslice
            self._sample(subslice)
        xs = [m[1] for m in meshes] + [m[2] for m in meshes]
        ys = [m[3] for m in meshes] + [m[4] for m in meshes]
        self.extent = Extent(min(xs), max(xs), min(ys), max(ys), z, z)

    def _sample(self, subslice: SubSlice) -> None:
        coords = subslice.get_coordinates()
        x, y = np.meshgrid(coords["x"], coords["y"], indexing="ij")
        offset = self._offsets.get(subslice.mesh.id, 0.0)
        subslice._data = np.stack(
            [self._field(x, y, t) + offset for t in self.times]
        ).astype(np.float32)

    def move_nodes(self, mesh_id: str, dim: str, nodes: np.ndarray) -> None:
        """Give a mesh other node coordinates along *dim*, e.g. stretched, and sample the field there."""
        subslice = self._subslices[mesh_id]
        subslice.mesh.coordinates[dim] = np.asarray(nodes, dtype=float)
        self._sample(subslice)

    def get_nearest_timestep(self, time: float) -> int:
        return int(np.argmin(np.abs(self.times - time)))

    def clear_cache(self) -> None:
        pass


def patch_simulation(monkeypatch, slc: FakeSlice) -> None:
    """Make ``fdsreader.Simulation`` in fdsvismap return a simulation with *slc* as slice 0."""
    sim = SimpleNamespace(
        slices=[slc],
        obstructions=[],
        meshes=[s.mesh for s in slc.subslices],
    )
    monkeypatch.setattr(fdsvismap.FDSVisMap.fds, "Simulation", lambda sim_dir: sim)
