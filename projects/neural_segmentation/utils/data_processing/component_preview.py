import numpy as np
from scipy.ndimage import generate_binary_structure, label

from projects.segmentation.utils.data_processing.final_segmentation import (
    build_final_segmentation,
)
from projects.segmentation.utils.data_processing.result import Result
from projects.segmentation.utils.project_info import FinalSegmentationData


def build_component_preview(data, component_id, **kwargs):
    try:
        labels, count = label(data > 0, structure=generate_binary_structure(3, 3))
        if component_id < 1 or component_id > count:
            raise ValueError("Connected component is no longer available")
        component = np.zeros_like(data)
        mask = labels == component_id
        component[mask] = data[mask]
        build_final_segmentation(data=component, min_component_size_rate=None, **kwargs)
    except Exception as error:
        kwargs["queue_out"].put(
            Result(
                FinalSegmentationData(), kwargs["metadata"], error=str(error)
            )
        )
