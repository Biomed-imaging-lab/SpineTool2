from enum import auto

from utils.misc import StringEnum

TYPE = "segmentation"
IMAGE = "image"
POLYGON_MESH = "polygon_mesh"
BINARY_VOXEL_CORRECTION = "binary_voxel_correction"
LAYERS_PATH = "/layers"
COLORMAPS_PATH = "/colormaps"
ADDITIONAL_COLORMAPS_PATH = "/additional_colormaps"
AUXILIARY_PATH = "/auxiliary"
TMP = "/tmp"
TMP_LAYERS_PATH = TMP + LAYERS_PATH
TMP_AUXILIARY_PATH = TMP + AUXILIARY_PATH

# Developer-tunable voxel-to-mesh profiles. UI exposes only the profile names.
# ``marching_cubes_step_size`` controls complexity while the surface is built,
# avoiding an expensive full-resolution build followed by decimation.
DEFAULT_MESH_COMPLEXITY = "high"
MESH_COMPLEXITY_PROFILES = {
    "high": {"marching_cubes_step_size": 1},
    "medium": {"marching_cubes_step_size": 2},
    "low": {"marching_cubes_step_size": 4},
}

# An enclosed background region is filled only when it is smaller than this
# fraction of the connected component that surrounds it.
FILL_HOLES_MAX_COMPONENT_RATIO = 0.25


NDISPLAY_SHORTCUT_TEXT = "D"
ROLL_SHORTCUT_TEXT = "R"
TRANSPOSE_SHORTCUT_TEXT = "T"
RESET_VIEW_SHORTCUT_TEXT = "H"
RESTORE_SHORTCUT_TEXT = "Ctrl+R"
DELETE_SHORTCUT_TEXT = "Ctrl+⌦"
BACKSPACE_SHORTCUT_TEXT = "Ctrl+⌫"
IMAGE_STAGE_SHORTCUT_TEXT = "I"
ADDITIONAL_LAYER_SHORTCUT_TEXT = "Shift+A"
SELECTED_STAGE_SHORTCUT_TEXT = "E"
PREVIOUS_STAGE_SHORTCUT_TEXT = "B"

RESET_SCROLL_SHORTCUT_TEXT = "Shift+S"
INCREMENT_SCROLL_SHORTCUT_TEXT = "Ctrl+→"
DECREMENT_SCROLL_SHORTCUT_TEXT = "Ctrl+←"
TOGGLE_VISIBILITY_SHORTCUT_TEXT = "V"
PAN_ZOOM_TMP_SHORTCUT_TEXT = "X"

MAX_INTENSITY_SHORTCUT_TEXT = "M"

UNDO_SHORTCUT_TEXT = "Ctrl+Z"
REDO_SHORTCUT_TEXT = "Ctrl+Y"

SHUFFLE_COLORS_SHORTCUT_TEXT = "Shift+C"
PAN_ZOOM_SHORTCUT_TEXT = "1"
BRUSH_SHORTCUT_TEXT = "2"
ERASE_SHORTCUT_TEXT = "3"
FILL_SHORTCUT_TEXT = "4"

NEXT_LABEL_SHORTCUT_TEXT = "+"
PREV_LABEL_SHORTCUT_TEXT = "-"

FILL_3D_SHORTCUT_TEXT = "F"
PRESERVE_BACKGROUND_SHORTCUT_TEXT = "L"
SHOW_SELECTED_SHORTCUT_TEXT = "S"

ADD_SHORTCUT_TEXT = "2"
SELECT_SHORTCUT_TEXT = "3"
SELECT_ALL_SHORTCUT_TEXT = "Ctrl+A"
DELETE_POINT_SHORTCUT_TEXT = "⌦"
BACKSPACE_POINT_SHORTCUT_TEXT = "⌫"
OUT_OF_SLICE_SHORTCUT_TEXT = "O"


class ProjectDescriptionkKeys(StringEnum):
    NAME = auto()
    TYPE = auto()
    SUBTYPE = auto()
    ORIGINAL_IMAGE = auto()
    ORIGINAL_SHAPE = auto()
    SHAPE = auto()
    DISPLAYED_SCALE = auto()
    REAL_SCALE = auto()
    SCALE = auto()
    MIN_COORDINATES = auto()
    ACTIVE_LAYERS = auto()
    NON_EDITABLE_LAYERS = auto()
    DATA = auto()
    LAYERS_PARAMETERS = auto()
    BACKGROUND_IMAGES = auto()
    DEVICE = auto()


class ProjectDataDescriptionkKeys(StringEnum):
    LAYERS = auto()
    LAST_LAYER_ID = auto()


class LayerDescriptionKeys(StringEnum):
    ID = auto()
    TYPE = auto()
    PARENT_ID = auto()
    CHILD_LAYERS = auto()


class LayerParametersKeys(StringEnum):
    LAYER_ID = auto()
    NAME = auto()
    FILE = auto()
    MESH_FILE = auto()
    MESH_SOURCE_TIF_FILE = auto()
    SPINES_FILES = auto()
    ADJUSTED_SPINES_FILES = auto()
    PREVIEW = auto()
    PREVIEW_NEED_UPDATE = auto()
    PREVIEW_PARTIALLY_FIXED = auto()
    VISUALISATION_PARAMETERS = auto()
    METADATA = auto()
    DELETED_SPINES = auto()
    ADDITIONAL_FILES = auto()


class NonEditableLayerDescriptionKeys(StringEnum):
    LAYER_ID = auto()
    PARAMETERS = auto()


class Types(StringEnum):
    BACKGROUND_IMAGE = auto()
    IMAGE = auto()
    MASK = auto()
    BINARIZATION = auto()
    POINTS = auto()
    NECKS = auto()
    POLYGON_MESH = auto()
    POLYGON_MESH_SEGMENTATION = auto()
    VOXEL_MESH_SEGMENTATION = auto()
    FINAL_SEGMENTATION = auto()


EMPTY_PROJECT = {
    ProjectDescriptionkKeys.NAME.value: "",
    ProjectDescriptionkKeys.TYPE.value: TYPE,
    ProjectDescriptionkKeys.SUBTYPE.value: IMAGE,
    ProjectDescriptionkKeys.ORIGINAL_IMAGE.value: "",
    ProjectDescriptionkKeys.ORIGINAL_SHAPE.value: (1, 1, 1),
    ProjectDescriptionkKeys.SHAPE.value: (1, 1, 1),
    ProjectDescriptionkKeys.DISPLAYED_SCALE.value: [1.0, 1.0, 1.0],
    ProjectDescriptionkKeys.REAL_SCALE.value: [1.0, 1.0, 1.0],
    ProjectDescriptionkKeys.SCALE.value: [1.0, 1.0, 1.0],
    ProjectDescriptionkKeys.MIN_COORDINATES.value: [0.0, 0.0, 0.0],
    ProjectDescriptionkKeys.DATA.value: None,
    ProjectDescriptionkKeys.ACTIVE_LAYERS.value: [],
    ProjectDescriptionkKeys.NON_EDITABLE_LAYERS.value: [],
    ProjectDescriptionkKeys.BACKGROUND_IMAGES.value: [],
    ProjectDescriptionkKeys.LAYERS_PARAMETERS.value: {},
    ProjectDescriptionkKeys.DEVICE.value: "cpu"
}


EMPTY_PROJECT_DATA = {
    ProjectDataDescriptionkKeys.LAYERS.value: [],
    ProjectDataDescriptionkKeys.LAST_LAYER_ID.value: 0,
}


EMPTY_LAYER_DESCRIPTION = {
    LayerDescriptionKeys.ID.value: None,
    LayerDescriptionKeys.TYPE.value: Types.IMAGE.value,
    LayerDescriptionKeys.PARENT_ID.value: None,
    LayerDescriptionKeys.CHILD_LAYERS.value: [],
}


EMPTY_LAYER_PARAMETERS = {
    LayerParametersKeys.LAYER_ID.value: None,
    LayerParametersKeys.NAME.value: "",
    LayerParametersKeys.FILE.value: "",
    LayerParametersKeys.MESH_FILE.value: "",
    LayerParametersKeys.MESH_SOURCE_TIF_FILE.value: "",
    LayerParametersKeys.SPINES_FILES.value: [],
    LayerParametersKeys.ADJUSTED_SPINES_FILES.value: [],
    LayerParametersKeys.PREVIEW.value: False,
    LayerParametersKeys.PREVIEW_PARTIALLY_FIXED.value: False,
    LayerParametersKeys.PREVIEW_NEED_UPDATE.value: False,
    LayerParametersKeys.VISUALISATION_PARAMETERS.value: {},
    LayerParametersKeys.METADATA.value: {},
    LayerParametersKeys.DELETED_SPINES.value: [],
    LayerParametersKeys.ADDITIONAL_FILES.value: {},
}
