import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from tifffile import imread, imwrite

from projects.neural_segmentation.settings import NeuralSegmentationSettings
from projects.neural_segmentation.utils.constants import (
    ARTIFACTS_PATH,
    MODEL_RUNS_PATH,
    PLUGIN_RUNTIME_PATH,
    TYPE,
)
from projects.neural_segmentation.utils.data_processing.plugin_inference import (
    run_ai_spines_inference,
)
from projects.neural_segmentation.widgets.qt_voxel_preview_options import (
    QtNeuralVoxelPreviewOptions,
)
from projects.neural_segmentation.widgets.qt_necks_preview_options import (
    QtNeuralNecksPreviewOptions,
)
from projects.neural_segmentation.widgets.qt_stage4_run_options import (
    QtStage4RunOptions,
)
from projects.project_base import ProjectBase
from projects.segmentation.project_segmentation import (
    DataProcessingTask,
    SegmentationProject,
    Types,
)
from projects.segmentation.settings import SegmentationSettings
from projects.segmentation.utils import constants as segmentation_constants
from projects.segmentation.utils.history import (
    get_open_history_files,
    update_open_history_files,
)
from projects.segmentation.utils.project_info import ProjectInfo
from projects.segmentation.utils.data_processing.final_segmentation import (
    build_final_segmentation,
)
from projects.segmentation.utils.data_processing.voxel_cleanup import (
    fill_small_enclosed_holes,
)
from utils.notifications import show_error, show_warning
from utils.progress import progress
from utils.qt_translater import Translater

if TYPE_CHECKING:
    from application.widgets.qt_main_window import Window


WORKFLOW_MODE_KEY = "workflow_mode"
WORKFLOW_MODE_VALUE = "neural_cascade"
NEURAL_STAGE_KEY = "neural_stage"
NEURAL_MODEL_PATH_KEY = "neural_model_path"
NEURAL_PROBABILITY_PATH_KEY = "neural_probability_path"
NEURAL_IMPORTED_FROM_KEY = "neural_imported_from"
NEURAL_PLUGIN_RESPONSE_PATH_KEY = "neural_plugin_response_path"
NEURAL_PLUGIN_STDOUT_KEY = "neural_plugin_stdout"
NEURAL_PLUGIN_STDERR_KEY = "neural_plugin_stderr"
NEURAL_OUTPUT_SHAPE_KEY = "neural_output_shape"
NEURAL_LAYER_SCALE_KEY = "neural_layer_scale"
NEURAL_PLUGIN_ENTRY_SCRIPT = "run_inference.py"
USER_CORRECTION_POSITIVE_PROBABILITY = 0.999
USER_CORRECTION_NEGATIVE_PROBABILITY = 0.001
NEURAL_PLUGIN_STAGE4_SCRIPT = str(
    (Path(__file__).parent / "plugins" / "ai_spines_stage4_inference_cli.py").resolve()
)
MODEL_WEIGHT_EXTENSIONS = (".ckpt", ".pth", ".pt", ".onnx")
NEURAL_STAGE_LABELS = {
    1: "binarization",
    2: "neck restoration",
    3: "scale restoration",
    4: "stem/spines segmentation",
}


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _write_json(path: str, payload: dict) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        return True
    except Exception:
        return False


def _find_root_metadata(payload: dict) -> dict:
    layers_parameters = payload.get("layers_parameters", [])
    for entry in layers_parameters:
        if entry.get("layer_id") == 0:
            metadata = entry.get("metadata")
            if isinstance(metadata, dict):
                return metadata
    return {}


class NeuralSegmentationProject(SegmentationProject):
    @property
    def type(self) -> str:
        return TYPE

    @classmethod
    def check_file(cls, path: str) -> bool:
        if not path.endswith(ProjectInfo.DESCRIPTION_FILENAME):
            return False
        payload = _read_json(path)
        if payload is None:
            return False
        root_metadata = _find_root_metadata(payload)
        return root_metadata.get(WORKFLOW_MODE_KEY) == WORKFLOW_MODE_VALUE

    @classmethod
    def create(cls, window: "Window") -> Optional[str]:
        path = SegmentationProject.create(
            window,
            show_workflow_mode=False,
            default_workflow_mode=segmentation_constants.IMAGE,
        )
        if not path:
            return None

        payload = _read_json(path)
        if payload is None:
            show_error("failed open project description")
            return None

        layers_parameters = payload.get("layers_parameters", [])
        for entry in layers_parameters:
            if entry.get("layer_id") == 0:
                metadata = entry.setdefault("metadata", {})
                metadata[WORKFLOW_MODE_KEY] = WORKFLOW_MODE_VALUE
                break
        else:
            show_error("failed set neural workflow marker")
            return None

        if not _write_json(path, payload):
            show_error("failed save project description")
            return None

        folder, _ = os.path.split(path)
        cls._ensure_neural_storage(folder)
        return path

    @classmethod
    def open(cls, window: "Window", path: str) -> Optional[ProjectBase]:
        try:
            if re.fullmatch(r"[a-zA-Z0-9_\-:\\/\.]+", path):
                project = ProjectInfo.load(path)
                cls._ensure_neural_storage(project.folder)
                return NeuralSegmentationProject(
                    window, project, SegmentationSettings.instance()
                )
            show_error("Invalid characters in path. Move project in other folder")
            return None
        except Exception as err:
            show_error(str(err))
            return None

    @staticmethod
    def _ensure_neural_storage(folder: str) -> None:
        os.makedirs(folder + ARTIFACTS_PATH, exist_ok=True)
        os.makedirs(folder + MODEL_RUNS_PATH, exist_ok=True)
        os.makedirs(folder + PLUGIN_RUNTIME_PATH, exist_ok=True)

    def __init__(
        self,
        window: "Window",
        project_info: ProjectInfo,
        settings: Optional[SegmentationSettings] = None,
    ):
        self._stage4_run_options = None
        self._neural_painted_masks: dict[int, np.ndarray] = {}
        self._neural_paint_original_values: dict[int, np.ndarray] = {}
        self._neural_settings = NeuralSegmentationSettings.instance()
        self._neural_settings.load()
        self._neural_settings.ensure_defaults()
        self._neural_settings.updated.connect(self._on_neural_settings_update)
        super().__init__(window, project_info, settings or SegmentationSettings.instance())
        self._ensure_neural_storage(self._project_info.folder)
        self._ensure_models_storage()
        self._project_info_widget.toggle_cuda_checkbox(True)
        self._stage4_run_options = QtStage4RunOptions(
            self._neural_settings.state,
            self._qt_next_stage_list,
        )
        if self._current_stage_id() >= 4:
            self._qt_next_stage_list.set_run_options_widget(self._mesh_build_options)
            self._qt_next_stage_list.show_run_options(True)
        else:
            self._qt_next_stage_list.set_run_options_widget(self._stage4_run_options)
            self._qt_next_stage_list.show_run_options(self._current_stage_id() == 3)

    @property
    def additional_settings(self) -> dict:
        return {"neural segmentation": self._neural_settings}

    def _on_neural_settings_update(self, state: dict) -> None:
        self._neural_settings.save()

    def _ensure_models_storage(self) -> None:
        models_folder = self._neural_settings.state.get("models_folder", "")
        if models_folder == "":
            return
        plugin_repo_root = self._neural_settings.state.get("plugin_repo_root", "")
        if (
            plugin_repo_root
            and os.path.abspath(models_folder).startswith(os.path.abspath(plugin_repo_root))
            and not os.path.isdir(plugin_repo_root)
        ):
            return
        for stage_id in NEURAL_STAGE_LABELS:
            os.makedirs(self._stage_models_folder(stage_id), exist_ok=True)

    def _current_stage_id(self) -> int:
        if len(self._active_layers_list) == 0:
            return 0
        layer = self._active_layers_list[-1]
        if layer.metadata["type"] == Types.IMAGE:
            return 0
        if layer.metadata["type"] == Types.MASK:
            return 1
        metadata = self._project_info.layers_parameters[layer._id].tmp_metadata
        return int(metadata.get(NEURAL_STAGE_KEY, 0))

    def _set_create_new_stage_button(self, stage, blocked=False) -> None:
        if stage == Types.IMAGE:
            super()._set_create_new_stage_button(stage, blocked)
            return

        try:
            self._qt_next_stage_list.create_new_button.clicked.disconnect()
        except Exception:
            pass

        stage4_options_visible = (
            stage == Types.VOXEL_MESH_SEGMENTATION
            and self._current_stage_id() == 3
        )
        if self._stage4_run_options is not None:
            options_widget = (
                self._stage4_run_options
                if stage4_options_visible
                else self._mesh_build_options
            )
            self._qt_next_stage_list.set_run_options_widget(options_widget)
            self._qt_next_stage_list.show_run_options(
                stage4_options_visible or self._current_stage_id() >= 4
            )

        if stage == Types.MASK:
            self._qt_next_stage_list.create_new_button.setText(
                "run model: binarization"
            )
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_binarization
            )
        elif stage == Types.BINARIZATION:
            self._qt_next_stage_list.create_new_button.setText(
                "run model: neck restoration"
            )
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_necks
            )
        elif stage == Types.NECKS:
            self._qt_next_stage_list.create_new_button.setText(
                "run model: scale restoration"
            )
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_voxel_segmentation
            )
        elif stage == Types.VOXEL_MESH_SEGMENTATION:
            if self._current_stage_id() >= 4:
                self._qt_next_stage_list.create_new_button.setText(
                    "create final segmentation"
                )
                self._qt_next_stage_list.create_new_button.clicked.connect(
                    self._create_new_neural_final_segmentation
                )
            else:
                self._qt_next_stage_list.create_new_button.setText(
                    "run model: stem/spines segmentation"
                )
                self._qt_next_stage_list.create_new_button.clicked.connect(
                    self._create_new_stem_spines_segmentation
                )
        else:
            blocked = True
            self._qt_next_stage_list.create_new_button.setText(
                "stage is not used in neural workflow"
            )

        self._qt_next_stage_list.create_new_button.setEnabled(not blocked)

    def _create_new_neural_final_segmentation(self) -> None:
        if not self._active_layers_list:
            return
        source_layer = self._active_layers_list[-1]
        if self._current_stage_id() < 4:
            return
        mesh_options = self._mesh_build_options.get_params()
        metadata = {
            "name": Translater.instance().get_translation("final segmentation"),
            "parent_id": source_layer._id,
            **mesh_options,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.FINAL_SEGMENTATION,
                build_final_segmentation,
                {
                    "data": source_layer._data,
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "min_coord": self._project_info.min_coordinates,
                    "folder": self._project_info.folder,
                    "metadata": metadata,
                    **mesh_options,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "final_segmentation in queue",
                source_layer._id,
            )
        )

    def _choose_file(self, caption: str, filter: str) -> str:
        dlg = QFileDialog(self._window._qt_window)
        files = get_open_history_files()
        hist = []
        for file in files:
            if os.path.isdir(file):
                hist.append(file)
            else:
                hist.append(os.path.dirname(file))
        hist = [item for item in hist if item]
        if not hist:
            hist = [str(Path.home())]
        dlg.setHistory(hist)
        path, _ = dlg.getOpenFileName(
            caption=Translater.instance().get_translation(caption),
            directory=hist[0],
            filter=filter,
        )
        if path and path != "":
            update_open_history_files(path)
        return path

    def _choose_folder(self, caption: str, initial: str = "") -> str:
        dlg = QFileDialog(self._window._qt_window)
        folder = dlg.getExistingDirectory(
            caption=caption,
            directory=initial if initial else str(Path.home()),
        )
        return folder or ""

    def _choose_python_executable(self, initial: str = "") -> str:
        dlg = QFileDialog(self._window._qt_window)
        start = initial if initial else str(Path.home())
        path, _ = dlg.getOpenFileName(
            caption="Select python executable for neural plugin",
            directory=start,
            filter="Python executable (python.exe python *.exe);;All files (*)",
        )
        return path or ""

    def _relative_project_path(self, path: str) -> str:
        rel = os.path.relpath(path, self._project_info.folder)
        return "/" + rel.replace("\\", "/")

    def _stage_folder_abs(self, stage_id: int) -> str:
        return (
            self._project_info.folder
            + MODEL_RUNS_PATH
            + "/stage_"
            + str(stage_id)
        )

    def _next_stage_probability_paths(self, stage_id: int) -> tuple[str, str]:
        folder = self._stage_folder_abs(stage_id)
        os.makedirs(folder, exist_ok=True)
        filename = (
            "probability_"
            + str(datetime.now())
            .replace(":", "_")
            .replace(" ", "_")
            .replace(".", "_")
            + ".tif"
        )
        output_abs = folder + "/" + filename
        return output_abs, self._relative_project_path(output_abs)

    def _resolve_plugin_runtime(self) -> Optional[tuple[str, str, str]]:
        repo_root = self._neural_settings.state.get("plugin_repo_root", "")
        python_executable = self._neural_settings.state.get(
            "plugin_python_executable", ""
        )

        if not repo_root or not os.path.isdir(repo_root):
            selected = self._choose_folder(
                "Select ai_spines_segmentation plugin repository",
                repo_root,
            )
            if selected == "":
                return None
            repo_root = selected
            self._neural_settings.update({"plugin_repo_root": repo_root})
            self._neural_settings.save()

        if python_executable == "":
            python_executable = sys.executable
        if not os.path.isfile(python_executable):
            selected = self._choose_python_executable(python_executable)
            if selected == "":
                return None
            python_executable = selected
            self._neural_settings.update(
                {"plugin_python_executable": python_executable}
            )
            self._neural_settings.save()

        plugin_script_path = os.path.join(repo_root, NEURAL_PLUGIN_ENTRY_SCRIPT)
        if not os.path.isfile(plugin_script_path):
            show_error(
                "Neural plugin entry script not found: "
                + plugin_script_path
                + ". Expected "
                + NEURAL_PLUGIN_ENTRY_SCRIPT
            )
            return None

        return repo_root, python_executable, plugin_script_path

    def _resolve_stage4_preprocess_runtime(self) -> Optional[dict]:
        values = {
            "vsot_matlab_dir": str(
                self._neural_settings.state.get("vsot_matlab_dir", "")
            ).strip(),
            "vsot_root": str(
                self._neural_settings.state.get("vsot_root", "")
            ).strip(),
            "octave_executable": str(
                self._neural_settings.state.get("octave_executable", "")
            ).strip(),
        }
        labels = {
            "vsot_matlab_dir": "VSOT matlab dir",
            "vsot_root": "VSOT root",
            "octave_executable": "Octave executable",
        }
        missing = [labels[key] for key, value in values.items() if value == ""]
        if missing:
            show_warning(
                "Fill in the neural segmentation settings before running stage 4: "
                + ", ".join(missing) + "."
            )
            return None

        invalid = []
        if not os.path.isdir(values["vsot_matlab_dir"]):
            invalid.append(labels["vsot_matlab_dir"])
        if not os.path.isdir(values["vsot_root"]):
            invalid.append(labels["vsot_root"])
        if not os.path.isfile(values["octave_executable"]):
            invalid.append(labels["octave_executable"])
        if invalid:
            show_warning(
                "Check these paths in the neural segmentation settings: "
                + ", ".join(invalid) + "."
            )
            return None

        return values

    def _stage_models_folder(self, stage_id: int) -> str:
        models_folder = self._neural_settings.state.get(
            "models_folder", NeuralSegmentationSettings.default_models_folder()
        )
        return os.path.join(models_folder, "stage_" + str(stage_id))

    def _project_path(self, relative_or_abs: str) -> str:
        if re.fullmatch(r"[a-zA-Z]:[\\/].*", relative_or_abs):
            return relative_or_abs
        return self._project_info.folder + relative_or_abs

    def _read_probability(
        self, file: str, expected_shape: Optional[tuple] = None
    ) -> Optional[np.ndarray]:
        try:
            data = imread(file)
        except Exception as err:
            show_error(str(err))
            return None
        if (
            expected_shape is not None
            and data.ndim == 3
            and data.shape[0] == 2 * int(expected_shape[0])
            and tuple(data.shape[1:]) == tuple(expected_shape[1:])
        ):
            data = data.reshape((2, *tuple(expected_shape)))
        if data.ndim not in (3, 4):
            show_error("Model output must be 3D tif volume or 4D channel volume")
            return None
        spatial_shape = self._probability_spatial_shape(data)
        if expected_shape is not None and spatial_shape != tuple(expected_shape):
            show_error(
                "Model output shape mismatch. Expected: "
                + str(tuple(expected_shape))
                + ", got: "
                + str(tuple(data.shape))
            )
            return None
        arr = data.astype(np.float32)
        arr_min = float(np.min(arr))
        arr_max = float(np.max(arr))
        if arr_max <= 1.0 and arr_min >= 0:
            return arr
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr.fill(0)
        return arr

    @staticmethod
    def _probability_to_binary(probability: np.ndarray, threshold_0_1: float) -> np.ndarray:
        return (probability >= threshold_0_1).astype(np.uint8)

    @staticmethod
    def _probability_to_stem_spines(
        probability: np.ndarray, trunk_threshold_0_1: float, spine_threshold_0_1: float,
        threshold_mode: str = "trunk", base_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if probability.ndim == 4 and probability.shape[0] == 2:
            trunk_probability = probability[0]
            spine_probability = probability[1]
        elif probability.ndim == 4 and probability.shape[1] == 2:
            trunk_probability = probability[:, 0]
            spine_probability = probability[:, 1]
        else:
            trunk_probability = probability
            spine_probability = probability

        mask = np.ones(trunk_probability.shape, dtype=bool)
        if base_mask is not None:
            mask = np.asarray(base_mask, dtype=bool)
        labels = np.zeros(trunk_probability.shape, dtype=np.uint8)
        if threshold_mode == "spine":
            labels[mask] = 1
            labels[mask & (spine_probability >= spine_threshold_0_1)] = 2
        else:
            labels[mask] = 2
            labels[mask & (trunk_probability >= trunk_threshold_0_1)] = 1
        return labels

    @staticmethod
    def _probability_spatial_shape(probability: np.ndarray) -> tuple:
        if probability.ndim == 4 and probability.shape[0] in (1, 2):
            return tuple(probability.shape[1:])
        if probability.ndim == 4 and probability.shape[1] in (1, 2):
            return (probability.shape[0], probability.shape[2], probability.shape[3])
        return tuple(probability.shape)

    def _layer_for_id(self, layer_id: int):
        for layers in (
            self._active_layers_list,
            self._next_stage_list,
            self._additional_layers_list,
        ):
            for layer in layers:
                if layer._id == layer_id:
                    return layer
        return None

    def _neural_layer_scale(self, output_shape: tuple) -> list:
        project_shape = tuple(self._project_info.shape)
        if len(output_shape) != len(project_shape):
            return list(self._project_info.scale)

        scale = []
        for base_scale, base_size, output_size in zip(
            self._project_info.scale, project_shape, output_shape
        ):
            if output_size <= 0:
                return list(self._project_info.scale)
            scale.append(float(base_scale) * float(base_size) / float(output_size))
        return scale

    def _apply_neural_layer_scale(
        self, layer_id: int, output_shape: Optional[tuple] = None
    ) -> None:
        layer_params = self._project_info.layers_parameters.get(layer_id)
        if layer_params is None:
            return

        stage_id = int(layer_params.tmp_metadata.get(NEURAL_STAGE_KEY, 0))
        if stage_id not in (1, 2):
            return

        layer = self._layer_for_id(layer_id)
        if output_shape is None:
            if layer is None:
                return
            output_shape = tuple(layer.data.shape)

        if tuple(output_shape) == tuple(self._project_info.shape):
            return

        layer_scale = self._neural_layer_scale(tuple(output_shape))
        layer_params.parameters["scale"] = layer_scale
        layer_params.tmp_metadata[NEURAL_OUTPUT_SHAPE_KEY] = list(output_shape)
        layer_params.tmp_metadata[NEURAL_LAYER_SCALE_KEY] = layer_scale
        layer_params.metadata[NEURAL_OUTPUT_SHAPE_KEY] = list(output_shape)
        layer_params.metadata[NEURAL_LAYER_SCALE_KEY] = layer_scale

        if layer is not None:
            layer.scale = layer_scale

    def _probability_file_for_layer(self, layer_id: int) -> Optional[str]:
        probability_path = self._project_info.layers_parameters[layer_id].tmp_metadata.get(
            NEURAL_PROBABILITY_PATH_KEY, ""
        )
        if probability_path == "":
            return None
        probability_file = self._project_path(probability_path)
        if not os.path.isfile(probability_file):
            return None
        return probability_file

    @staticmethod
    def _apply_binary_label_corrections(
        probability: np.ndarray,
        labels: np.ndarray,
        threshold_0_1: float,
        painted_mask: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Apply high/low probabilities only at explicitly painted voxels."""
        labels = np.asarray(labels)
        probability_view = probability
        if probability.ndim == labels.ndim + 1:
            if probability.shape[0] == 1:
                probability_view = probability[0]
            elif probability.shape[1] == 1:
                probability_view = probability[:, 0]

        if probability_view.shape != labels.shape or painted_mask.shape != labels.shape:
            raise ValueError(
                "Probability, label and painted-mask shapes do not match: "
                + str(tuple(probability.shape))
                + " vs "
                + str(tuple(labels.shape))
                + " vs "
                + str(tuple(painted_mask.shape))
            )

        threshold = min(max(float(threshold_0_1), 0.0), 1.0)
        corrected_positive = painted_mask & (labels > 0)
        corrected_negative = painted_mask & (labels == 0)
        correction_count = int(
            np.count_nonzero(corrected_positive)
            + np.count_nonzero(corrected_negative)
        )
        if correction_count == 0:
            return probability, 0

        result = probability.astype(np.float32, copy=True)
        if probability_view is probability:
            result_view = result
        elif probability.shape[0] == 1:
            result_view = result[0]
        else:
            result_view = result[:, 0]
        positive_probability = max(
            USER_CORRECTION_POSITIVE_PROBABILITY,
            float(np.nextafter(np.float32(threshold), np.float32(1.0))),
        )
        negative_probability = min(
            USER_CORRECTION_NEGATIVE_PROBABILITY,
            float(np.nextafter(np.float32(threshold), np.float32(0.0))),
        )
        result_view[corrected_positive] = min(positive_probability, 1.0)
        result_view[corrected_negative] = max(negative_probability, 0.0)
        return result, correction_count

    def _track_neural_paint(self, layer_id: int, history_item) -> None:
        layer = self._layer_for_id(layer_id)
        if layer is None:
            return
        painted_mask = self._neural_painted_masks.get(layer_id)
        if painted_mask is None or painted_mask.shape != tuple(layer.data.shape):
            painted_mask = np.zeros(layer.data.shape, dtype=bool)
            self._neural_painted_masks[layer_id] = painted_mask
            self._neural_paint_original_values[layer_id] = np.zeros_like(layer.data)
        original_values = self._neural_paint_original_values[layer_id]
        for change in history_item:
            if not isinstance(change, tuple) or len(change) < 2:
                continue
            indices = change[0]
            previous_values = change[1]
            newly_painted = ~painted_mask[indices]
            if np.any(newly_painted):
                new_indices = tuple(axis[newly_painted] for axis in indices)
                if isinstance(previous_values, np.ndarray):
                    original_values[new_indices] = previous_values[newly_painted]
                else:
                    original_values[new_indices] = previous_values
            painted_mask[indices] = True

    def _track_neural_layer_painting(self, layer_id: int) -> None:
        layer = self._layer_for_id(layer_id)
        if layer is None or getattr(layer, "_neural_paint_tracking", False):
            return
        stage_id = int(
            self._project_info.layers_parameters[layer_id].tmp_metadata.get(
                NEURAL_STAGE_KEY, 0
            )
        )
        if stage_id not in (1, 2, 3):
            return
        layer.paint_.connect(
            lambda history_item, tracked_layer_id=layer_id: self._track_neural_paint(
                tracked_layer_id, history_item
            )
        )
        layer._neural_paint_tracking = True

    def _sync_probability_with_layer(self, layer_id: int) -> Optional[str]:
        """Persist user label corrections into the stage probability artifact."""
        probability_file = self._probability_file_for_layer(layer_id)
        if probability_file is None:
            return None

        layer = self._layer_for_id(layer_id)
        if layer is None:
            return probability_file
        if layer._changed:
            self._layer_loader.save_layer_data(layer_id, layer.data)
            self._saved = False

        metadata = self._project_info.layers_parameters[layer_id].tmp_metadata
        stage_id = int(metadata.get(NEURAL_STAGE_KEY, 0))
        if stage_id not in (1, 2, 3):
            return probability_file
        if not bool(metadata.get("apply_user_threshold", False)):
            return probability_file

        painted_mask = self._neural_painted_masks.get(layer_id)
        if painted_mask is None or not np.any(painted_mask):
            return probability_file
        original_values = self._neural_paint_original_values.get(layer_id)
        if original_values is None:
            return probability_file
        effective_painted_mask = painted_mask & (layer.data != original_values)
        if not np.any(effective_painted_mask):
            return probability_file

        if stage_id in (1, 2):
            threshold = float(metadata.get("neural_threshold", 0.5))
        else:
            threshold = float(metadata.get("threshold", 50)) / 100.0

        probability = self._read_probability(
            probability_file, expected_shape=tuple(layer.data.shape)
        )
        if probability is None:
            return None

        try:
            corrected_probability, correction_count = (
                self._apply_binary_label_corrections(
                    probability, layer.data, threshold, effective_painted_mask
                )
            )
        except ValueError as err:
            show_error(str(err))
            return None

        if correction_count > 0:
            try:
                imwrite(probability_file, corrected_probability)
            except Exception as err:
                show_error("Failed to save probability corrections: " + str(err))
                return None
            self._neural_painted_masks.pop(layer_id, None)
            self._neural_paint_original_values.pop(layer_id, None)
        return probability_file

    def _runtime_json_paths(self, stage_id: int) -> tuple[str, str]:
        runtime_folder = self._project_info.folder + PLUGIN_RUNTIME_PATH
        os.makedirs(runtime_folder, exist_ok=True)
        request_path = runtime_folder + "/stage_" + str(stage_id) + "_request.json"
        response_path = runtime_folder + "/stage_" + str(stage_id) + "_response.json"
        return request_path, response_path

    def _enqueue_neural_inference_task(
        self,
        *,
        layer_type: Types,
        parent_id: int,
        layer_name: str,
        plugin_runtime: tuple[str, str, str],
        stage_id: int,
        request: dict,
        output_probability_abs: str,
        metadata: dict,
        plugin_script_path_override: Optional[str] = None,
        label_mode: str = "binary",
        threshold_0_1: float = 0.5,
        trunk_threshold_0_1: float = 0.5,
        spine_threshold_0_1: float = 0.8,
        threshold_mode: str = "trunk",
        expected_shape: Optional[tuple] = None,
        process_environment: Optional[dict[str, str]] = None,
    ) -> None:
        _, python_executable, default_plugin_script_path = plugin_runtime
        plugin_script_path = (
            plugin_script_path_override
            if plugin_script_path_override is not None
            else default_plugin_script_path
        )
        if not os.path.isfile(plugin_script_path):
            show_error("Neural plugin script not found: " + plugin_script_path)
            return
        _, response_path = self._runtime_json_paths(stage_id)
        metadata[NEURAL_PLUGIN_RESPONSE_PATH_KEY] = self._relative_project_path(
            response_path
        )
        metadata[NEURAL_PLUGIN_STDOUT_KEY] = ""
        metadata[NEURAL_PLUGIN_STDERR_KEY] = ""

        request["stage"] = stage_id
        request["output_probability_path"] = output_probability_abs

        self._data_processing_task_queue.append(
            DataProcessingTask(
                layer_type,
                run_ai_spines_inference,
                {
                    "python_executable": python_executable,
                    "plugin_script_path": plugin_script_path,
                    "plugin_request": request,
                    "plugin_response_path": response_path,
                    "output_probability_path": output_probability_abs,
                    "metadata": {
                        "name": layer_name,
                        "parent_id": parent_id,
                        "params": metadata,
                    },
                    "expected_shape": expected_shape,
                    "label_mode": label_mode,
                    "threshold_0_1": threshold_0_1,
                    "trunk_threshold_0_1": trunk_threshold_0_1,
                    "spine_threshold_0_1": spine_threshold_0_1,
                    "threshold_mode": threshold_mode,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                    "process_environment": process_environment,
                },
                "stage" + str(stage_id) + " neural inference in queue",
                parent_id,
            )
        )

    def _prepare_neural_metadata(
        self,
        *,
        stage_id: int,
        model_path: str,
        output_source_path: str,
        probability_path: str,
        params: dict,
    ) -> dict:
        metadata = params.copy()
        metadata.update(
            {
                WORKFLOW_MODE_KEY: WORKFLOW_MODE_VALUE,
                NEURAL_STAGE_KEY: stage_id,
                NEURAL_MODEL_PATH_KEY: model_path,
                NEURAL_PROBABILITY_PATH_KEY: probability_path,
                NEURAL_IMPORTED_FROM_KEY: output_source_path,
            }
        )
        return metadata

    def _select_model(self, stage_id: int) -> str:
        label = NEURAL_STAGE_LABELS[stage_id]
        stage_folder = self._stage_models_folder(stage_id)
        os.makedirs(stage_folder, exist_ok=True)
        candidates = []
        for filename in os.listdir(stage_folder):
            path = os.path.join(stage_folder, filename)
            if os.path.isfile(path) and filename.lower().endswith(
                MODEL_WEIGHT_EXTENSIONS
            ):
                candidates.append(path)
        candidates.sort()

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            show_warning(
                "No model weights found for "
                + label
                + ". Put .ckpt/.pth/.pt/.onnx file into: "
                + stage_folder
            )
            return ""

        dlg = QFileDialog(self._window._qt_window)
        path, _ = dlg.getOpenFileName(
            caption="select model for " + label,
            directory=stage_folder,
            filter="Weights (*.ckpt *.pth *.pt *.onnx);;All files (*)",
        )
        if path and path != "":
            update_open_history_files(path)
        return path or ""

    def _create_new_binarization(self) -> None:
        if len(self._active_layers_list) < 2:
            return

        if self._active_layers_list[0]._changed:
            self._layer_loader.save_layer_data(
                self._active_layers_list[0]._id, self._active_layers_list[0].data
            )
            self._saved = False
        if self._active_layers_list[1]._changed:
            self._layer_loader.save_layer_data(
                self._active_layers_list[1]._id, self._active_layers_list[1].data
            )
            self._saved = False

        selected_model = self._select_model(1)
        if selected_model == "":
            return

        plugin_runtime = self._resolve_plugin_runtime()
        if plugin_runtime is None:
            return

        params = {
            "base_threshold": 127,
            "apply_user_threshold": False,
        }
        threshold = float(params["base_threshold"]) / 255.0
        output_probability_abs, output_probability_rel = self._next_stage_probability_paths(
            1
        )
        mask_layer_id = self._active_layers_list[1]._id

        metadata = self._prepare_neural_metadata(
            stage_id=1,
            model_path=selected_model,
            output_source_path="plugin:ai_spines:stage1",
            probability_path=output_probability_rel,
            params=params,
        )
        metadata["neural_threshold"] = threshold

        request = {
            "ckpt_path": selected_model,
            "input_image_path": self._project_path(
                self._project_info.layers_parameters[0].tmp_file
            ),
            "area_path": self._project_path(
                self._project_info.layers_parameters[mask_layer_id].tmp_file
            ),
            "current_scale": list(self._project_info.real_scale),
            "device": self._project_info.device,
        }
        self._enqueue_neural_inference_task(
            layer_type=Types.BINARIZATION,
            parent_id=mask_layer_id,
            layer_name=Translater.instance().get_translation("binarization"),
            plugin_runtime=plugin_runtime,
            stage_id=1,
            request=request,
            output_probability_abs=output_probability_abs,
            metadata=metadata,
            label_mode="binary",
            threshold_0_1=threshold,
            expected_shape=None,
        )

    def _create_new_necks(self) -> None:
        if len(self._active_layers_list) < 3:
            return
        selected_model = self._select_model(2)
        if selected_model == "":
            return

        plugin_runtime = self._resolve_plugin_runtime()
        if plugin_runtime is None:
            return

        stage1_layer_id = self._active_layers_list[2]._id
        stage1_probability = self._sync_probability_with_layer(stage1_layer_id)
        if stage1_probability is None:
            show_warning("Stage 1 probability was not found. Run stage 1 first.")
            return

        params = {
            "threshold": 50,
            "apply_user_threshold": False,
        }
        threshold = float(params["threshold"]) / 100.0
        output_probability_abs, output_probability_rel = self._next_stage_probability_paths(
            2
        )
        metadata = self._prepare_neural_metadata(
            stage_id=2,
            model_path=selected_model,
            output_source_path="plugin:ai_spines:stage2",
            probability_path=output_probability_rel,
            params=params,
        )
        metadata["neural_threshold"] = threshold
        request = {
            "ckpt_path": selected_model,
            "input_image_path": self._project_path(
                self._project_info.layers_parameters[0].tmp_file
            ),
            "stage1_probability_path": stage1_probability,
            "area_path": self._project_path(
                self._project_info.layers_parameters[self._active_layers_list[1]._id].tmp_file
            ),
            "current_scale": list(self._project_info.real_scale),
            "device": self._project_info.device,
        }
        self._enqueue_neural_inference_task(
            layer_type=Types.NECKS,
            parent_id=stage1_layer_id,
            layer_name=Translater.instance().get_translation("necks"),
            plugin_runtime=plugin_runtime,
            stage_id=2,
            request=request,
            output_probability_abs=output_probability_abs,
            metadata=metadata,
            label_mode="binary",
            threshold_0_1=threshold,
            expected_shape=tuple(self._active_layers_list[2].data.shape),
        )

    def _create_new_voxel_segmentation(self) -> None:
        if len(self._active_layers_list) < 4:
            return
        selected_model = self._select_model(3)
        if selected_model == "":
            return

        plugin_runtime = self._resolve_plugin_runtime()
        if plugin_runtime is None:
            return

        stage1_layer_id = self._active_layers_list[2]._id
        stage2_layer_id = self._active_layers_list[3]._id
        stage1_probability = self._sync_probability_with_layer(stage1_layer_id)
        stage2_probability = self._sync_probability_with_layer(stage2_layer_id)
        if stage1_probability is None or stage2_probability is None:
            show_warning("Stage 1/2 probabilities were not found. Run previous stages first.")
            return

        params = {"threshold": 50, "apply_user_threshold": False}
        threshold = float(params["threshold"]) / 100.0
        output_probability_abs, output_probability_rel = self._next_stage_probability_paths(
            3
        )
        metadata = self._prepare_neural_metadata(
            stage_id=3,
            model_path=selected_model,
            output_source_path="plugin:ai_spines:stage3",
            probability_path=output_probability_rel,
            params=params,
        )
        metadata["neural_threshold"] = threshold
        request = {
            "ckpt_path": selected_model,
            "input_image_path": self._project_path(
                self._project_info.layers_parameters[0].tmp_file
            ),
            "stage1_probability_path": stage1_probability,
            "stage2_probability_path": stage2_probability,
            "area_path": self._project_path(
                self._project_info.layers_parameters[self._active_layers_list[1]._id].tmp_file
            ),
            "current_scale": list(self._project_info.real_scale),
            "device": self._project_info.device,
        }
        self._enqueue_neural_inference_task(
            layer_type=Types.VOXEL_MESH_SEGMENTATION,
            parent_id=stage2_layer_id,
            layer_name="scale restoration",
            plugin_runtime=plugin_runtime,
            stage_id=3,
            request=request,
            output_probability_abs=output_probability_abs,
            metadata=metadata,
            label_mode="binary",
            threshold_0_1=threshold,
            expected_shape=tuple(self._project_info.shape),
        )

    def _create_new_stem_spines_segmentation(self) -> None:
        if len(self._active_layers_list) < 5:
            return
        selected_model = self._select_model(4)
        if selected_model == "":
            return

        plugin_runtime = self._resolve_plugin_runtime()
        if plugin_runtime is None:
            return
        stage4_runtime = self._resolve_stage4_preprocess_runtime()
        if stage4_runtime is None:
            return

        stage3_layer_id = self._active_layers_list[4]._id
        stage3_layer = self._active_layers_list[4]
        if stage3_layer._changed:
            self._layer_loader.save_layer_data(stage3_layer_id, stage3_layer.data)
            self._saved = False
        stage3_mask_path = self._project_path(
            self._project_info.layers_parameters[stage3_layer_id].tmp_file
        )

        params = self._stage4_run_options.get_params()
        if bool(params.get("fill_holes_before_stage4", True)):
            filled_stage3 = fill_small_enclosed_holes(stage3_layer.data)
            self._layer_loader.update_layer(stage3_layer_id, filled_stage3, stage3_layer)
            self._layer_loader.save_layer_data(stage3_layer_id, filled_stage3)
            stage3_params = self._project_info.layers_parameters[stage3_layer_id]
            stage3_params.metadata["holes_filled_before_stage4"] = True
            stage3_params.tmp_metadata["holes_filled_before_stage4"] = True
            self._saved = False
        overlap_percent = int(params["overlap_percent"])
        if overlap_percent not in (75, 50, 20):
            overlap_percent = 75
        patch_size = [64, 128, 128]
        overlap = [round(size * overlap_percent / 100) for size in patch_size]
        mixed_precision = bool(params["mixed_precision"])
        trunk_threshold = float(params["trunk_threshold"]) / 100.0
        spine_threshold = float(params["spine_threshold"]) / 100.0
        output_probability_abs, output_probability_rel = self._next_stage_probability_paths(
            4
        )
        metadata = self._prepare_neural_metadata(
            stage_id=4,
            model_path=selected_model,
            output_source_path="plugin:ai_spines:stage4",
            probability_path=output_probability_rel,
            params=params,
        )
        request = {
            "repo_root": plugin_runtime[0],
            "ckpt_path": selected_model,
            "input_image_path": self._project_path(
                self._project_info.layers_parameters[0].tmp_file
            ),
            "stage3_mask_path": stage3_mask_path,
            "area_path": self._project_path(
                self._project_info.layers_parameters[self._active_layers_list[1]._id].tmp_file
            ),
            "current_scale": list(self._project_info.real_scale),
            "device": self._project_info.device,
            "patch_size": patch_size,
            "overlap": overlap,
            "mixed_precision": mixed_precision,
            "skeleton_prune_length_nm": params["skeleton_prune_length_nm"],
            "stage4_prepare_dir": self._stage_folder_abs(4) + "/preprocess",
            "vsot_matlab_dir": stage4_runtime["vsot_matlab_dir"],
            "vsot_root": stage4_runtime["vsot_root"],
            "octave_executable": stage4_runtime["octave_executable"],
        }
        self._enqueue_neural_inference_task(
            layer_type=Types.VOXEL_MESH_SEGMENTATION,
            parent_id=stage3_layer_id,
            layer_name="stem and spines",
            plugin_runtime=plugin_runtime,
            stage_id=4,
            request=request,
            output_probability_abs=output_probability_abs,
            metadata=metadata,
            plugin_script_path_override=NEURAL_PLUGIN_STAGE4_SCRIPT,
            label_mode="stem_spines",
            trunk_threshold_0_1=trunk_threshold,
            spine_threshold_0_1=spine_threshold,
            threshold_mode=params["threshold_mode"],
            expected_shape=tuple(self._project_info.shape),
            process_environment={
                "OCTAVE_EXECUTABLE": stage4_runtime["octave_executable"]
            },
        )

    def _set_preview_options(self) -> None:
        layer = self._viewer_model.active_layer
        if layer and layer.metadata["type"] == Types.NECKS:
            layer_params = self._project_info.layers_parameters[layer._id]
            if int(layer_params.tmp_metadata.get(NEURAL_STAGE_KEY, 0)) == 2:
                delete = (
                    self._current_preview_options is not self._empty_options
                    and self._current_preview_options is not self._no_parameters
                )
                self._current_preview_options = QtNeuralNecksPreviewOptions(
                    layer_params.tmp_metadata,
                    not layer_params.tmp_preview,
                )
                self._current_preview_options.parameters_changed_.connect(
                    self._update_necks
                )
                self._current_preview_options.parameters_fixed_.connect(
                    self._fix_layer_parameters
                )
                self._preview_options_widget.replace_widget(
                    self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
                )
                self._current_preview_options.ndisplay = self._viewer_model.dims.ndisplay
                return
        if not layer or layer.metadata["type"] != Types.VOXEL_MESH_SEGMENTATION:
            super()._set_preview_options()
            return

        stage_id = int(
            self._project_info.layers_parameters[layer._id].tmp_metadata.get(
                NEURAL_STAGE_KEY, 0
            )
        )
        if stage_id not in (3, 4):
            super()._set_preview_options()
            return

        delete = (
            self._current_preview_options is not self._empty_options
            and self._current_preview_options is not self._no_parameters
        )
        self._current_preview_options = QtNeuralVoxelPreviewOptions(
            stage_id,
            self._project_info.layers_parameters[layer._id].tmp_metadata,
            not self._project_info.layers_parameters[layer._id].tmp_preview,
        )
        self._current_preview_options.parameters_changed_.connect(
            self._update_neural_voxel
        )
        self._current_preview_options.parameters_fixed_.connect(self._fix_layer_parameters)
        self._preview_options_widget.replace_widget(
            self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
        )
        self._current_preview_options.ndisplay = self._viewer_model.dims.ndisplay

    def _refresh_labels_from_probability(
        self,
        *,
        layer_id: int,
        stage_id: int,
        params: dict,
    ) -> None:
        layer = None
        for item in self._active_layers_list:
            if item._id == layer_id:
                layer = item
                break
        if layer is None:
            for item in self._next_stage_list:
                if item._id == layer_id:
                    layer = item
                    break
        if layer is None:
            return

        metadata = self._project_info.layers_parameters[layer_id].tmp_metadata
        probability_path = metadata.get(NEURAL_PROBABILITY_PATH_KEY, "")
        if probability_path == "":
            return
        probability_file = self._project_path(probability_path)
        if not os.path.isfile(probability_file):
            show_warning("Probability volume file was not found")
            return
        probability = self._read_probability(
            probability_file, expected_shape=tuple(layer.data.shape)
        )
        if probability is None:
            return

        if stage_id in (1, 2):
            threshold = float(params["neural_threshold"])
            labels = self._probability_to_binary(probability, threshold)
        elif stage_id == 3:
            threshold = float(params["threshold"]) / 100.0
            metadata["neural_threshold"] = threshold
            labels = self._probability_to_binary(probability, threshold)
        else:
            trunk = float(params["trunk_threshold"]) / 100.0
            spine = float(params["spine_threshold"]) / 100.0
            base_mask = self._active_layers_list[4].data > 0
            labels = self._probability_to_stem_spines(
                probability,
                trunk,
                spine,
                params.get("threshold_mode", "trunk"),
                base_mask,
            )

        painted_mask = self._neural_painted_masks.get(layer_id)
        if painted_mask is not None and painted_mask.shape == labels.shape:
            original_values = self._neural_paint_original_values.get(layer_id)
            if original_values is not None:
                effective_painted_mask = painted_mask & (
                    layer.data != original_values
                )
                labels[effective_painted_mask] = layer.data[effective_painted_mask]

        self._layer_loader.update_layer(layer_id, labels, layer)
        self._layer_loader.save_layer_data(layer_id, labels)
        self._saved = False

    def _update_binarization(self) -> None:
        layer = self._viewer_model.active_layer
        if (
            not layer
            or layer.metadata["type"] != Types.BINARIZATION
            or int(
                self._project_info.layers_parameters[layer._id].tmp_metadata.get(
                    NEURAL_STAGE_KEY, 0
                )
            )
            != 1
        ):
            super()._update_binarization()
            return

        params = self._current_preview_options.get_params()
        threshold = float(params["base_threshold"]) / 255.0
        params["neural_threshold"] = threshold
        self._project_info.layers_parameters[layer._id].tmp_metadata.update(params)
        self._refresh_labels_from_probability(
            layer_id=layer._id,
            stage_id=1,
            params=params,
        )

    def _update_necks(self) -> None:
        layer = self._viewer_model.active_layer
        if (
            not layer
            or layer.metadata["type"] != Types.NECKS
            or int(
                self._project_info.layers_parameters[layer._id].tmp_metadata.get(
                    NEURAL_STAGE_KEY, 0
                )
            )
            != 2
        ):
            super()._update_necks()
            return

        params = self._current_preview_options.get_params()
        threshold = float(params["threshold"]) / 100.0
        params["neural_threshold"] = min(max(threshold, 0.0), 1.0)
        self._project_info.layers_parameters[layer._id].tmp_metadata.update(params)
        self._refresh_labels_from_probability(
            layer_id=layer._id,
            stage_id=2,
            params=params,
        )

    def _update_neural_voxel(self) -> None:
        layer = self._viewer_model.active_layer
        if not layer or layer.metadata["type"] != Types.VOXEL_MESH_SEGMENTATION:
            return
        stage_id = int(
            self._project_info.layers_parameters[layer._id].tmp_metadata.get(
                NEURAL_STAGE_KEY, 0
            )
        )
        if stage_id not in (3, 4):
            return

        params = self._current_preview_options.get_params()
        self._project_info.layers_parameters[layer._id].tmp_metadata.update(params)
        self._refresh_labels_from_probability(
            layer_id=layer._id, stage_id=stage_id, params=params
        )

    def _on_layer_loaded(self, layer_info, metadata) -> None:
        info = layer_info[0]
        layer_id = getattr(info, "id", None)
        if layer_id is None:
            layer_id = getattr(info, "layer_id", None)
        data = layer_info[2]

        super()._on_layer_loaded(layer_info, metadata)

        if layer_id is not None and hasattr(data, "shape"):
            self._apply_neural_layer_scale(layer_id, tuple(data.shape))
            self._track_neural_layer_painting(layer_id)

    def _create(self) -> None:
        params = self._active_layer_loader_task.params
        metadata = params.get("metadata", {})
        stage_id = int(metadata.get(NEURAL_STAGE_KEY, 0))
        is_neural_voxel = (
            params.get("layer_type") == Types.VOXEL_MESH_SEGMENTATION
            and stage_id in (3, 4)
        )

        super()._create()

        if stage_id in (1, 2, 3):
            self._track_neural_layer_painting(
                self._project_info.data.last_layer_id
            )

        if stage_id in (1, 2):
            layer_id = self._project_info.data.last_layer_id
            data = params.get("data")
            output_shape = tuple(data.shape) if hasattr(data, "shape") else None
            self._apply_neural_layer_scale(layer_id, output_shape)

        if not is_neural_voxel:
            return
        layer_id = self._project_info.data.last_layer_id
        layer_params = self._project_info.layers_parameters[layer_id]
        layer_params.preview = True
        layer_params.tmp_preview = True
        layer_params.parameters["labels"] = (
            ["background", "restored_mask"]
            if int(layer_params.tmp_metadata.get(NEURAL_STAGE_KEY, 0)) == 3
            else ["background", "trunk", "spines"]
        )
        for layer in self._next_stage_list:
            if layer._id == layer_id:
                layer.preview = True
                break
