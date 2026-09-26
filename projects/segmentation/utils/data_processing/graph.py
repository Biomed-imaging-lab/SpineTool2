"""Portable adapter for the legacy neck path API.

The original graph extension is absent from this repository. Reuse the
26-neighbour intensity-weighted A* implementation used by paired necks.
"""
from projects.segmentation.utils.data_processing.paired_necks import (
    shortest_path_between_points,
)


def find_path(image, start, target, intensity_factor):
    # Legacy callers operate in zoomed voxel coordinates and rescale afterwards.
    return shortest_path_between_points(
        image, start, target, (1.0, 1.0, 1.0), intensity_factor=intensity_factor
    )
