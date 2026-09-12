from typing import Dict

from projects.project_base import ProjectBase
from projects.neural_segmentation.project_neural_segmentation import (
    NeuralSegmentationProject,
)
from projects.neural_segmentation.utils.constants import TYPE as NEURAL_SEGMENTATION_TYPE
from projects.segmentation.project_segmentation import SegmentationProject
from projects.segmentation.utils.constants import TYPE as SEGMENTATION_TYPE

TYPES: Dict[str, ProjectBase] = {
    SEGMENTATION_TYPE: SegmentationProject,
    NEURAL_SEGMENTATION_TYPE: NeuralSegmentationProject,
}
