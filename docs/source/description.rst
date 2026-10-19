
Description
===========

In performance-based fire safety design, ensuring safe egress, e.g. by visibility of safety signs, is a crucial safety
goal. Compliance with the building requirements is often demonstrated by simulations of smoke spread. Numerical models
like the Fire Dynamics Simulator generally compute visibility as a local quantity using the light extinction
coefficient, without the consideration of the actual light path to a safety sign.

Visibility maps provide an approach for post-processing fire simulation data. They indicate safe areas along egress
routes, with respect to visibility. At each location, the available visibility to a safety sign is calculated using
Jin's law, with the mean extinction coefficient along the line of sight to the sign. The required visibility results
from the distance between those points. Additional parameters like view angle or visual obstructions are considered. A
location is safe if at least one sign is visible from it. The method allows for temporal visibility assessment, e.g.
in an ASET-RSET analysis.

FDSVisMap implements the method of the original paper [Boerger2024]_. The sections below describe the computation.
The equation numbers refer to the preprint of the paper.

Notation
--------

The computation works on the cells of the horizontal FDS slice that is selected with ``fds_slc_height`` in
``read_fds_data``.

- :math:`\sigma`: extinction coefficient of a cell, read from the slice (``quantity``). An optical density is
  converted with :math:`\sigma = \ln(10) \cdot \mathrm{OD}`.
- :math:`(x_s, y_s)`: position of the sign (``x``, ``y`` of ``add_sign``).
- :math:`C`: contrast factor of the sign according to Jin, e.g. 3 for reflecting and 8 for light-emitting signs (``c``
  of ``add_sign``).
- :math:`\alpha`: orientation of the sign, clockwise from the positive y-axis (``alpha`` of ``add_sign``).
- :math:`(x, y)`: centre of the cell of the observer.
- :math:`V_\mathrm{min}`, :math:`V_\mathrm{max}`: lower and upper limit of the visibility, 0 m and 30 m by default
  (``set_visibility_bounds``).

Visibility of a sign
--------------------

For each sign, time point and cell, FDSVisMap decides whether the sign can be seen from the cell.

**Distance.** The distance between the sign and the cell is the horizontal distance (Eq. 6):

.. math::

   L = \sqrt{(x - x_s)^2 + (y - y_s)^2}

**Mean extinction coefficient.** The line of sight from the cell closest to the sign to the cell of the observer is
rasterised with Bresenham's algorithm. :math:`P` is the set of cells on this line, the cell of the sign and the cell of the
observer included. The mean extinction coefficient is the arithmetic mean over these cells (Eq. 9):

.. math::

   \bar{\sigma} = \frac{1}{|P|} \sum_{p \in P} \sigma_p

Each cell counts with the same weight, including the two end cells. This approximates the mean along the line of sight
:math:`\frac{1}{L}\int_0^L \sigma(l)\,\mathrm{d}l` (Eq. 8), assuming the same path length in each cell.

**View angle.** A sign is seen best from the front. The view angle factor is the cosine of the angle between the
orientation of the sign and the line of sight, and zero behind the sign (Eq. 7):

.. math::

   A = \max\left(0,\; \frac{\sin\alpha\,(x - x_s) + \cos\alpha\,(y - y_s)}{L}\right)

For a sign that is visible from all directions (``alpha="omni"``), or with ``compute_all(view_angle=False)``,
:math:`A = 1`.

**Visual obstructions.** Rays are cast from the cell of the sign to the cells at the edge of the domain. With
``compute_all(aa=True)``, the default, they are drawn with anti-aliasing (Wu's algorithm), so that they do not pass
through gaps between obstructed cells. The obstruction factor :math:`U` is 1 for the cells before the first obstructed
cell of a ray and 0 for the cells behind it. Obstructed cells are the FDS obstructions at the height of the slice and
the rectangles of ``add_visual_obstruction``, without those of ``add_visual_hole``. With
``compute_all(obstructions=False)``, :math:`U = 1`.

**Available visibility.** Jin's law gives the available visibility with the mean extinction coefficient:

.. math::

   V = \min\left(\frac{C}{\bar{\sigma}},\; V_\mathrm{max}\right) \cdot A \cdot U

In clear air, :math:`\bar{\sigma} = 0`, the visibility is :math:`V_\mathrm{max} \cdot A \cdot U`.

**Visibility map of a sign.** The sign :math:`k` is visible from a cell if the available visibility reaches the sign
(Eq. 2):

.. math::

   M_k = \begin{cases} 1 & \text{if } V_k \geq L_k \\ 0 & \text{otherwise} \end{cases}

In addition, a cell with :math:`V_k < V_\mathrm{min}` is not visible. ``get_sign_vismap`` returns this map and
``sign_is_visible`` evaluates it at a single location. ``get_visibility_to_sign`` returns :math:`V_k` at a single
location.

Aggregation
-----------

**Over signs.** A cell is visible if at least one sign is visible from it (Eq. 3):

.. math::

   M = \bigvee_k M_k

``get_agg_vismap`` aggregates all signs, or the signs of a route with ``route_id``. :math:`M^t` denotes this map at
the time point :math:`t`.

**Over time.** A cell of the time-aggregated map is visible if it is visible at all evaluated time points
:math:`t \in T` (Eq. 4):

.. math::

   M_\mathrm{agg} = \bigwedge_{t \in T} M^t

``get_time_agg_vismap`` returns this map.

**ASET.** The ASET of a cell is the first time point at which it is no longer visible (Eq. 5):

.. math::

   \mathrm{ASET} = \min\{\, t \in T \mid M^t = 0 \,\}

``get_aset_map`` returns the ASET of all cells, ``get_route_aset`` that of the points of a route. Cells that stay
visible get ``max_time``, by default the last computed time point.

Reference
---------

.. [Boerger2024] K. Börger, A. Belt, L. Arnold: *A waypoint based approach to visibility in performance based fire
   safety design*. Fire Safety Journal 150 (2024) 104269.
   `doi:10.1016/j.firesaf.2024.104269 <https://doi.org/10.1016/j.firesaf.2024.104269>`_.
   Preprint: `arXiv:2404.11439 <https://arxiv.org/abs/2404.11439>`_.

To cite this work, use:

.. code-block:: bibtex

   @article{BORGER2024104269,
     title = {A waypoint based approach to visibility in performance based fire safety design},
     author = {Kristian Börger and Alexander Belt and Lukas Arnold},
     journal = {Fire Safety Journal},
     volume = {150},
     pages = {104269},
     year = {2024},
     issn = {0379-7112},
     doi = {10.1016/j.firesaf.2024.104269},
     url = {https://www.sciencedirect.com/science/article/pii/S0379711224001826},
   }
