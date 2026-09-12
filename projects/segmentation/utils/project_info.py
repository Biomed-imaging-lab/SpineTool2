import os
from json import dump, load
from typing import Dict, List

from projects.segmentation.utils.constants import (
    EMPTY_LAYER_DESCRIPTION,
    EMPTY_LAYER_PARAMETERS,
    EMPTY_PROJECT,
    EMPTY_PROJECT_DATA,
    IMAGE,
    TYPE,
)
from projects.segmentation.utils.constants import LayerDescriptionKeys as LDK
from projects.segmentation.utils.constants import LayerParametersKeys as LPK
from projects.segmentation.utils.constants import (
    NonEditableLayerDescriptionKeys as NELDK,
)
from projects.segmentation.utils.constants import ProjectDataDescriptionkKeys as PDDK
from projects.segmentation.utils.constants import ProjectDescriptionkKeys as PDK
from projects.segmentation.utils.constants import Types

WORKFLOW_MODE_KEY = "workflow_mode"
NEURAL_WORKFLOW_MODE = "neural_cascade"


class ProjectInfo:
    DESCRIPTION_FILENAME = "project_description.json"

    def __init__(self, info=None, folder=None):
        if not info:
            info = EMPTY_PROJECT.copy()
        self.name: str = info[PDK.NAME.value]
        self.type: str = TYPE
        self.subtype: str = info.get(PDK.SUBTYPE.value, IMAGE)
        self.folder: str = folder if folder else os.path.expanduser("~")
        self.original_image: str = info[PDK.ORIGINAL_IMAGE.value]
        self.shape: tuple = tuple(info[PDK.SHAPE.value])
        self.original_shape: tuple = tuple(
            info.get(PDK.ORIGINAL_SHAPE.value, self.shape)
        )
        self.data: ProjectData = ProjectData(info[PDK.DATA.value])
        self.layers_parameters: Dict[int, LayerParameters] = {}
        for layer_parameters in info[PDK.LAYERS_PARAMETERS.value]:
            self.layers_parameters[layer_parameters[LPK.LAYER_ID.value]] = (
                LayerParameters(layer_parameters)
            )
        self.displayed_scale: list = info[PDK.DISPLAYED_SCALE.value]
        self.real_scale: list = info[PDK.REAL_SCALE.value]
        self._scale: list = info[PDK.SCALE.value]
        self.min_coordinates: list = info.get(PDK.MIN_COORDINATES.value, [0.0, 0.0, 0.0])
        self.active_layers: list = info[PDK.ACTIVE_LAYERS.value]
        self.non_editable_layers: List[NonEditableLayerInfo] = []
        for non_editable_layer_info in info[PDK.NON_EDITABLE_LAYERS.value]:
            self.non_editable_layers.append(
                NonEditableLayerInfo(non_editable_layer_info)
            )
        self.background_images: list = info.get(PDK.BACKGROUND_IMAGES.value, [])
        self._device: list = info[PDK.DEVICE.value]

    @classmethod
    def check_file(cls, path: str) -> bool:
        if not path.endswith(ProjectInfo.DESCRIPTION_FILENAME):
            return False
        file = None
        try:
            file = open(path)
            project_info = load(file)
            file.close()
        except:
            return False
        for s in PDK:
            if (
                s.value not in project_info
                and s != PDK.ORIGINAL_SHAPE
                and s != PDK.BACKGROUND_IMAGES
                and s != PDK.SUBTYPE
                and s != PDK.MIN_COORDINATES
            ):
                return False
        if project_info[PDK.TYPE.value] != TYPE:
            return False
        layers_parameters = project_info.get(PDK.LAYERS_PARAMETERS.value, [])
        for layer_parameters in layers_parameters:
            if layer_parameters.get(LPK.LAYER_ID.value) == 0:
                metadata = layer_parameters.get(LPK.METADATA.value, {})
                if (
                    isinstance(metadata, dict)
                    and metadata.get(WORKFLOW_MODE_KEY) == NEURAL_WORKFLOW_MODE
                ):
                    return False
                break
        return True

    @classmethod
    def load(cls, path) -> "ProjectInfo":
        file = open(path, "r")
        project_info = load(file)
        file.close()
        folder, _ = os.path.split(path)
        return ProjectInfo(project_info, folder)

    @property
    def scale(self) -> list:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self.displayed_scale = value
        min_scale = min(self.real_scale)
        self._scale = [s / min_scale * m for s, m in zip(self.real_scale, value[::-1])]

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value) -> None:
        self._device = value

    @property
    def filename(self) -> str:
        return self.folder + "/" + ProjectInfo.DESCRIPTION_FILENAME

    def as_dict(self) -> dict:
        non_editable_layers = []
        for non_editable_layer in self.non_editable_layers:
            non_editable_layers.append(non_editable_layer.as_dict())
        layers_parameters = []
        for layer_parameters in self.layers_parameters.values():
            layers_parameters.append(layer_parameters.as_dict())
        return {
            PDK.NAME.value: self.name,
            PDK.TYPE.value: TYPE,
            PDK.SUBTYPE.value: self.subtype,
            PDK.ORIGINAL_IMAGE.value: self.original_image,
            PDK.DISPLAYED_SCALE.value: self.displayed_scale,
            PDK.REAL_SCALE.value: self.real_scale,
            PDK.ORIGINAL_SHAPE.value: self.original_shape,
            PDK.SHAPE.value: self.shape,
            PDK.SCALE.value: self._scale,
            PDK.MIN_COORDINATES.value: self.min_coordinates,
            PDK.DATA.value: self.data.as_dict(),
            PDK.LAYERS_PARAMETERS.value: layers_parameters,
            PDK.ACTIVE_LAYERS.value: self.active_layers,
            PDK.NON_EDITABLE_LAYERS.value: non_editable_layers,
            PDK.BACKGROUND_IMAGES.value: self.background_images,
            PDK.DEVICE.value: self.device,
        }

    def save(self) -> None:
        filename = self.folder + "/" + ProjectInfo.DESCRIPTION_FILENAME
        file = open(filename, "w")
        dump(self.as_dict(), file)
        file.close()


class ProjectData:
    def __init__(self, info=None):
        if not info:
            info = EMPTY_PROJECT_DATA.copy()
        self.last_layer_id: int = info[PDDK.LAST_LAYER_ID.value]
        self.layers: Dict[int, LayerInfo] = {}
        for layer_info in info[PDDK.LAYERS.value]:
            self.layers[layer_info[LDK.ID.value]] = LayerInfo(layer_info)

    def as_dict(self) -> dict:
        layers = []
        for layer in self.layers.values():
            layers.append(layer.as_dict())
        return {
            PDDK.LAYERS.value: layers,
            PDDK.LAST_LAYER_ID.value: self.last_layer_id,
        }


class LayerInfo:
    def __init__(self, info=None):
        if not info:
            info = EMPTY_LAYER_DESCRIPTION.copy()
        self.id: int | None = info[LDK.ID.value]
        self.type: Types = Types(info[LDK.TYPE.value])
        self.parent_id: int | None = info[LDK.PARENT_ID.value]
        self.child_layers: list = info[LDK.CHILD_LAYERS.value].copy()

    def as_dict(self) -> dict:
        info = {
            LDK.ID.value: self.id,
            LDK.TYPE.value: self.type.value,
            LDK.PARENT_ID.value: self.parent_id,
            LDK.CHILD_LAYERS.value: self.child_layers,
        }
        return info


class LayerParameters:
    def __init__(self, info=None):
        if not info:
            info = EMPTY_LAYER_PARAMETERS.copy()
        self.layer_id: int | None = info[LPK.LAYER_ID.value]
        self.last_change_time = None
        self.name: str = info[LPK.NAME.value]
        self.file: str = info[LPK.FILE.value]
        self.tmp_file: str = self.file
        self.mesh_file: str = info[LPK.MESH_FILE.value]
        self.tmp_mesh_file: str = self.mesh_file
        self.mesh_source_tif_file: str = info[LPK.MESH_SOURCE_TIF_FILE.value]
        self.tmp_mesh_source_tif_file: str = self.mesh_source_tif_file
        self.spines_files: List[str] = info[LPK.SPINES_FILES.value]
        self.tmp_spines_files: List[str] = self.spines_files.copy()
        self.adjusted_spines_files: List[str] = info[LPK.ADJUSTED_SPINES_FILES.value]
        self.tmp_adjusted_spines_files: List[str] = self.adjusted_spines_files.copy()
        self.preview: bool = info[LPK.PREVIEW.value]
        self.tmp_preview: bool = self.preview
        self.preview_need_update: bool = info[LPK.PREVIEW_NEED_UPDATE.value]
        self.tmp_preview_need_update: bool = self.preview_need_update
        self.preview_partially_fixed: bool = info[LPK.PREVIEW_PARTIALLY_FIXED.value]
        self.tmp_preview_partially_fixed: bool = self.preview_partially_fixed
        self.parameters: dict = info[LPK.VISUALISATION_PARAMETERS.value].copy()
        self.metadata: dict = info[LPK.METADATA.value].copy()
        self.tmp_metadata: dict = self.metadata.copy()
        self.deleted_spines: set = set(info[LPK.DELETED_SPINES.value])
        self.tmp_deleted_spines: set = self.deleted_spines.copy()
        self.additional_files: Dict[str, str] = info[LPK.ADDITIONAL_FILES.value].copy()
        self.tmp_additional_files: Dict[str, str] = self.additional_files.copy()

    def as_dict(self) -> dict:
        info = {
            LPK.LAYER_ID.value: self.layer_id,
            LPK.NAME.value: self.name,
            LPK.FILE.value: self.file,
            LPK.MESH_FILE.value: self.mesh_file,
            LPK.MESH_SOURCE_TIF_FILE.value: self.mesh_source_tif_file,
            LPK.SPINES_FILES.value: self.spines_files,
            LPK.ADJUSTED_SPINES_FILES.value: self.adjusted_spines_files,
            LPK.PREVIEW.value: self.preview,
            LPK.PREVIEW_NEED_UPDATE.value: self.preview_need_update,
            LPK.PREVIEW_PARTIALLY_FIXED.value: self.preview_partially_fixed,
            LPK.VISUALISATION_PARAMETERS.value: self.parameters,
            LPK.METADATA.value: self.metadata,
            LPK.DELETED_SPINES.value: list(self.deleted_spines),
            LPK.ADDITIONAL_FILES.value: self.additional_files,
        }
        return info


class NonEditableLayerInfo:
    def __init__(self, info=None):
        if info:
            self.layer_id: int = info[NELDK.LAYER_ID.value]
            self.parameters: dict = info[NELDK.PARAMETERS.value]
        else:
            self.layer_id: int = -1
            self.parameters: dict = {}

    def as_dict(self) -> dict:
        return {
            NELDK.LAYER_ID.value: self.layer_id,
            NELDK.PARAMETERS.value: self.parameters,
        }


class SurfaceData:
    def __init__(self, mesh_file: str = "", mesh_v_f: tuple = None, tif_file: str = ""):
        self.mesh_file = mesh_file
        self.mesh_v_f = mesh_v_f
        self.tif_file = tif_file


class SegmentationData:
    def __init__(
        self,
        spines_files: list = [],
        adjusted_spines_files: list = [],
        mesh_v_f: tuple = None,
        deleted_spines: set = set(),
    ):
        self.spines_files = spines_files
        self.adjusted_spines_files = adjusted_spines_files
        self.mesh_v_f = mesh_v_f
        self.deleted_spines = deleted_spines


class FinalSegmentationData:
    def __init__(
        self,
        mesh_file: str = "",
        mesh_v_f_vv: tuple = None,
        spines_files: list = [],
    ):
        self.mesh_file = mesh_file
        self.mesh_v_f_vv = mesh_v_f_vv
        self.spines_files = spines_files
