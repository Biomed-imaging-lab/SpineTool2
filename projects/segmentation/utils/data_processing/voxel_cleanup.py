import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects, generate_binary_structure, label

from projects.segmentation.utils.constants import FILL_HOLES_MAX_COMPONENT_RATIO


def fill_small_enclosed_holes(
    data: np.ndarray,
    max_hole_to_component_ratio: float = FILL_HOLES_MAX_COMPONENT_RATIO,
) -> np.ndarray:
    """Fill small 3-D holes independently inside every labeled component."""
    source = np.asarray(data)
    result = source.copy()
    structure = generate_binary_structure(3, 1)
    ratio = max(0.0, float(max_hole_to_component_ratio))

    for value in np.unique(source):
        if value == 0:
            continue
        components, count = label(source == value, structure=structure)
        component_slices = find_objects(components)
        for component_id, component_slice in enumerate(component_slices, start=1):
            if component_slice is None:
                continue
            component = components[component_slice] == component_id
            component_size = int(np.count_nonzero(component))
            if component_size == 0:
                continue
            source_view = source[component_slice]
            enclosed = binary_fill_holes(component) & ~component & (source_view == 0)
            holes, hole_count = label(enclosed, structure=structure)
            max_hole_size = component_size * ratio
            for hole_id in range(1, hole_count + 1):
                hole = holes == hole_id
                if int(np.count_nonzero(hole)) < max_hole_size:
                    result_view = result[component_slice]
                    result_view[hole] = value
    return result
