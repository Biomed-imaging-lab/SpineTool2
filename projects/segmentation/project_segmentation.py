import os
import re
import subprocess
from datetime import datetime
from enum import auto
from functools import partial
from json import dump
from multiprocessing import Process, Queue
from pathlib import Path
from shutil import copy, rmtree
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from tifffile import imread, imwrite

import projects.segmentation.utils.constants as constants
from application.settings import ApplicationSettings
from application.utils.history import update_open_history
from application.widgets.qt_confirm_save_dialog import QtConfirmSaveDialog
import trimesh
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from projects.project_base import ProjectBase
from projects.segmentation.settings import SegmentationSettings
from projects.segmentation.utils.constants import (
    ADDITIONAL_COLORMAPS_PATH,
    AUXILIARY_PATH,
    COLORMAPS_PATH,
    LAYERS_PATH,
    TMP,
    TMP_AUXILIARY_PATH,
    TMP_LAYERS_PATH,
    Types,
)
from projects.segmentation.utils.data_processing.binarization import binarize
from projects.segmentation.utils.data_processing.binary_voxel import (
    build_voxel_from_binary,
)
from projects.segmentation.utils.data_processing.final_segmentation import (
    build_final_segmentation,
)
from projects.segmentation.utils.data_processing.mesh import (
    _mesh_to_v_f,
    build_mesh,
    create_mesh_descriptions,
)
from projects.segmentation.utils.data_processing.mesh_segmentation import (
    correct_spine,
    fix_segmentation,
    segment_spines,
)
from projects.segmentation.utils.data_processing.necks import necks_reconnection
from projects.segmentation.utils.data_processing.points import find_points
from projects.segmentation.utils.data_processing.result import ProgressUpdate, Result
from projects.segmentation.utils.data_processing.voxel_segmentation import (
    _voxelize_dendrite,
    voxelize,
)
from projects.segmentation.utils.data_processing.voxel_cleanup import (
    fill_small_enclosed_holes,
)
from projects.segmentation.utils.history import (
    get_open_history_files,
    get_open_history_folders,
    update_open_history_files,
    update_open_history_folders,
)
from projects.segmentation.utils.layer_loader import (
    LayerLoader,
    TaskResult,
    delete_layer,
    export_layer,
    fix_project_state,
    load_layers,
    restore_layer,
)
from projects.segmentation.utils.project_info import (
    FinalSegmentationData,
    LayerInfo,
    LayerParameters,
    NonEditableLayerInfo,
    ProjectInfo,
    SegmentationData,
    SurfaceData,
)
from projects.segmentation.widgets.preview_options.qt_binarization_preview_options import (
    QtBinarizationPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_final_segmentation_preview_options import (
    QtFinalSegmentationPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_image_preview_options import (
    QtImagePreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_mesh_segmentation_preview_options import (
    QtMeshSegmentationPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_necks_preview_options import (
    QtNecksPreviewOptions,
)
from projects.segmentation.widgets.preview_options.qt_points_preview_options import (
    QtPointsPreviewOptions,
)
from projects.segmentation.widgets.qt_mesh_build_options import QtMeshBuildOptions
from projects.segmentation.widgets.qt_control_buttons import (
    QtButtonsContainer,
    QtControlButtons,
)
from projects.segmentation.widgets.qt_list_widgets import (
    QtNextStageListWidget,
    QtStageListWidget,
)
from projects.segmentation.widgets.qt_project_info import QtProjectInfo
from projects.segmentation.widgets.qt_scrollable_container import QtScrollableContainer
from utils.misc import StringEnum
from utils.notifications import show_error
from utils.progress import progress
from utils.qt_translater import Translater
from utils.settings import Settings
from utils.shortcuts import Shortcut
from utils.themes import Style
from viewer.components.layer_list.layerlist import LayerList
from viewer.components.viewer_model import ViewerModel
from viewer.layers.image.image import Image
from viewer.layers.labels.labels import Labels
from viewer.layers.points.points import Points
from viewer.layers.points._points_constants import Mode as PointsMode
from viewer.utils.constants import STYLES_PATH, SVGS_PATH
from viewer.widgets.layer_controls.create_layer_controls import create_qt_layer_controls
from viewer.widgets.qt_viewer import QtViewer
from viewer.widgets.qt_viewer_buttons import QtViewerButtons
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from application.widgets.qt_main_window import Window


class LayerType:
    ACTIVE = 0
    ADDITIONAL = 1
    NEXT_STAGE = 2


class TaskType(StringEnum):
    LOAD = auto()
    CREATE = auto()
    DELETE = auto()
    RESTORE = auto()
    EXPORT = auto()
    SAVE = auto()


class LayerLoaderTask:
    def __init__(self, task_type: TaskType, pbar: progress, params: dict = {}):
        self.type = task_type
        self.params = params
        self.pbar = pbar
        self.process: Process | None = None


class DataProcessingTask:
    def __init__(
        self,
        layer_type: Types,
        processor: Callable,
        args: dict,
        pbar_desc: str,
        parent_id: int,
        layer_id: int | None = None,
        layer_params_partially_fixed: bool = False,
    ):
        self.layer_id = layer_id
        self.parent_id = parent_id
        self.type = layer_type
        self.args = args
        self.processor = processor
        self.pbar_desc = pbar_desc
        self.pbar = None
        self.process: Process | None = None
        self.creation_time = str(datetime.now())
        self.layer_params_partially_fixed = layer_params_partially_fixed

    def __del__(self):
        if self.pbar:
            self.pbar.close()


def layers_sorter(layer_lists: List[LayerList]) -> LayerList:
    layers_by_types = {}
    for layer_type in Types:
        layers_by_types[layer_type] = []
    for layer_list in layer_lists:
        for layer in layer_list:
            layers_by_types[layer.metadata["type"]].append(layer)
    layers = []
    for layer_type in Types:
        layers.extend(layers_by_types[layer_type])
    return LayerList(layers)


class QtInputEventFilter(QObject):
    def eventFilter(self, target, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.MouseButtonDblClick
            or event.type() == QEvent.Type.MouseButtonPress
            or event.type() == QEvent.Type.MouseButtonRelease
            or event.type() == QEvent.Type.MouseMove
            or event.type() == QEvent.Type.KeyPress
            or event.type() == QEvent.Type.KeyRelease
        ):
            return False
        return super().eventFilter(target, event)


class SegmentationProject(ProjectBase):
    CHECK_QUEUE_TIME_INTERVAL = 200

    _style = Style(
        [str(x) for x in Path(STYLES_PATH).resolve().iterdir() if x.suffix == ".qss"]
        + [
            str(x)
            for x in Path(SegmentationSettings.STYLES_PATH).resolve().iterdir()
            if x.suffix == ".qss"
        ],
        [
            str(x)
            for x in Path(SegmentationSettings.SVGS_PATH).resolve().iterdir()
            if x.suffix == ".svg"
        ]
        + [str(x) for x in Path(SVGS_PATH).resolve().iterdir() if x.suffix == ".svg"],
    )

    def __init__(
        self,
        window: "Window",
        project_info: ProjectInfo,
        settings: Optional[Settings] = None
    ):
        super().__init__(window, settings)

        self._project_info = project_info

        self._loaded = False
        self._saved = True
        self._closing = False
        self._saving = False
        self._input_filter = QtInputEventFilter()
        self._neck_points_phase_switch = False
        self._neck_points_data_callbacks = {}

        self._active_layer_loader_task: LayerLoaderTask | None = None
        self._layer_loader_task_queue: List[LayerLoaderTask] = []
        self._layer_loader_communication_queue_in = Queue()
        self._layer_loader_communication_queue_out = Queue()
        self._layer_loader_tasks_timer = QTimer()
        self._layer_loader_tasks_timer.setInterval(self.CHECK_QUEUE_TIME_INTERVAL)
        self._layer_loader_tasks_timer.setSingleShot(False)
        self._layer_loader_tasks_timer.timeout.connect(
            self._check_layer_loader_communication_queue
        )
        self._layer_loader_tasks_timer.start()
        self._exported_layers = set()

        self._active_data_processing_task: DataProcessingTask | None = None
        self._data_processing_task_queue: List[DataProcessingTask] = []
        self._data_processing_communication_queue_in = Queue()
        self._data_processing_communication_queue_out = Queue()
        self._data_processing_tasks_timer = QTimer()
        self._data_processing_tasks_timer.setInterval(self.CHECK_QUEUE_TIME_INTERVAL)
        self._data_processing_tasks_timer.setSingleShot(False)
        self._data_processing_tasks_timer.timeout.connect(
            self._check_data_processing_communication_queue
        )
        self._data_processing_tasks_timer.timeout.connect(
            self._next_data_processing_task
        )
        self._data_processing_tasks_timer.start()

        self._layer_loader = LayerLoader(self._project_info)
        self._next_stage_list = LayerList()
        self._additional_layers_list = LayerList()
        self._active_layers_list = LayerList()

        if self._project_info.subtype == constants.POLYGON_MESH:
            self._layer_offset_index = 0
        elif self._project_info.subtype == constants.IMAGE:
            self._layer_offset_index = 5
        elif self._project_info.subtype == constants.BINARY_VOXEL_CORRECTION:
            self._layer_offset_index = 5
        else:
            self._layer_offset_index = 5

        self._project_info_widget = QtProjectInfo(
            self._project_info, self._window._qt_window
        )
        self._project_info_widget.hide()
        self._project_info_widget.renamed_.connect(self._window.set_title)

        self._empty_options = QtLabel("select layer", {"en", "ru"})
        self._empty_options.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_options.setMinimumWidth(200)
        self._no_parameters = QtLabel("no parameters")
        self._current_preview_options = self._empty_options
        self._preview_options_widget = QtScrollableContainer(
            self._empty_options, max_height=350
        )
        self._preview_options_widget.hide()

        self._qt_next_stage_list = QtNextStageListWidget(self._next_stage_list)
        self._mesh_build_options = QtMeshBuildOptions(parent=self._qt_next_stage_list)
        self._qt_next_stage_list.set_run_options_widget(self._mesh_build_options)
        self._qt_next_stage_list.show_run_options(False)
        self._qt_next_stage_list.hide()
        self._qt_next_stage_list.next_stage_list.selectionModel().selectionChanged.connect(
            self._on_next_stage_selection_changed
        )

        self._empty_controls = QtLabel("select layer", {"en", "ru"})
        self._empty_controls.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_controls.setMinimumWidth(200)
        self._current_layer_controls = self._empty_controls
        self._layer_controls_widget = QtScrollableContainer(self._empty_controls)
        self._layer_controls_widget.hide()

        self._qt_additional_layer_list = QtStageListWidget(
            self._additional_layers_list, True, False
        )
        self._qt_additional_layer_list.hide()
        self._qt_additional_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_additional_layers_selection_changed
        )

        self._qt_active_layer_list = QtStageListWidget(
            self._active_layers_list, False, True
        )
        self._qt_active_layer_list.hide()
        self._qt_active_layer_list.setMaximumHeight(375)
        self._qt_active_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_active_layers_selection_changed
        )

        layer_lists = [None, None, None]
        layer_lists[LayerType.ACTIVE] = self._active_layers_list
        layer_lists[LayerType.ADDITIONAL] = self._additional_layers_list
        layer_lists[LayerType.NEXT_STAGE] = self._next_stage_list
        self._viewer_model = ViewerModel(
            layer_lists,
            2,
            (0, 1, 2),
            ("z", "y", "x"),
        )
        self._viewer = QtViewer(self._viewer_model, self._window._qt_window)
        self._viewer.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._project_info_widget.scale_changed_.connect(self._viewer_model.set_scale)

        self._viewer_buttons = QtButtonsContainer(
            QtViewerButtons(self._viewer_model), Qt.AlignmentFlag.AlignLeft
        )
        self._viewer_buttons.hide()
        self._viewer_buttons.buttons.ndisplay_toggled.connect(self._ndisplay_change)

        self._control_buttons = QtButtonsContainer(
            QtControlButtons(self), Qt.AlignmentFlag.AlignRight
        )
        self._control_buttons.hide()

        self._window.set_central_widget(self._viewer)
        self._window.set_menus()
        self._set_shortcuts()
        self._load_initial_state()

        self._project_info_widget.toggle_cuda_checkbox(False)

    def _check_layer_loader_communication_queue(self) -> None:
        if self._layer_loader_communication_queue_in.empty():
            return
        if self._active_layer_loader_task is None:
            while not self._layer_loader_communication_queue_in.empty():
                self._layer_loader_communication_queue_in.get()
            return

        next_task = True
        result: TaskResult = self._layer_loader_communication_queue_in.get()
        if result.project_data:
            self._project_info.data = result.project_data
        if result.layer_loader_state:
            self._layer_loader.state = result.layer_loader_state
        result = result.result
        if self._active_layer_loader_task.type == TaskType.LOAD:
            next_task = False
            while True:
                if isinstance(result, bool):
                    if self._loaded:
                        self._on_loading_finished(result)
                    else:
                        self._active_layer_loader_task.process.join()
                        self._active_layer_loader_task.pbar.close()
                        self._active_layer_loader_task = None
                        self._on_initial_state_loaded(result)
                    next_task = True
                else:
                    self._on_layer_loaded(*result)
                if self._layer_loader_communication_queue_in.empty():
                    break
                else:
                    result = self._layer_loader_communication_queue_in.get()
                    if result.project_data:
                        self._project_info.data = result.project_data
                    if result.layer_loader_state:
                        self._layer_loader.state = result.layer_loader_state
                    result = result.result
        elif self._active_layer_loader_task.type == TaskType.DELETE:
            self._on_deleted(result)
        elif self._active_layer_loader_task.type == TaskType.RESTORE:
            self._on_restored(*result)
        elif self._active_layer_loader_task.type == TaskType.EXPORT:
            self._on_exported()
        elif self._active_layer_loader_task.type == TaskType.SAVE:
            self._on_saved(result)

        if next_task:
            if self._active_layer_loader_task:
                self._active_layer_loader_task.process.join()
                self._active_layer_loader_task.pbar.close()
                self._active_layer_loader_task = None
            self._next_layer_loader_task()

    def _check_data_processing_communication_queue(
        self,
    ) -> None:
        if self._data_processing_communication_queue_in.empty() or (
            self._saving and not self._closing
        ):
            return
        if self._active_data_processing_task is None:
            while not self._data_processing_communication_queue_in.empty():
                self._data_processing_communication_queue_in.get()
            return
        if self._active_data_processing_task.layer_id in self._exported_layers:
            return

        result = self._data_processing_communication_queue_in.get()
        while isinstance(result, ProgressUpdate):
            self._apply_data_processing_progress(result)
            if self._data_processing_communication_queue_in.empty():
                return
            result = self._data_processing_communication_queue_in.get()

        self._active_data_processing_task.process.join()
        self._active_data_processing_task.pbar.close()

        if self._closing:
            self._active_data_processing_task = None
            self._next_layer_loader_task()
            return

        if result.stopped or result.error != "":
            if len(result.additional_files) > 0:
                layer_type = self._active_data_processing_task.type
                if layer_type == Types.POINTS:
                    processed = False
                    parent_id = self._project_info.data.layers[
                        self._active_data_processing_task.layer_id
                    ].parent_id
                    if not self._project_info.layers_parameters[
                        parent_id
                    ].last_change_time or (
                        self._active_data_processing_task.creation_time
                        > self._project_info.layers_parameters[
                            parent_id
                        ].last_change_time
                    ):
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update = False
                        self._layer_loader.replace_additional_files(
                            parent_id, result.additional_files
                        )
                        processed = True
                    if not processed:
                        for desc, filename in result.additional_files.items():
                            if filename != self._project_info.layers_parameters[
                                parent_id
                            ].tmp_additional_files.get(desc, ""):
                                os.remove(self._project_info.folder + filename)
                elif layer_type == Types.NECKS:
                    processed = False
                    parent_id = self._project_info.data.layers[
                        self._active_data_processing_task.layer_id
                    ].parent_id
                    parent_id = self._project_info.data.layers[parent_id].parent_id
                    if not self._project_info.layers_parameters[
                        parent_id
                    ].last_change_time or (
                        self._active_data_processing_task.creation_time
                        > self._project_info.layers_parameters[
                            parent_id
                        ].last_change_time
                    ):
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update = False
                        self._layer_loader.replace_additional_files(
                            parent_id, result.additional_files
                        )
                        processed = True
                    if not processed:
                        for desc, filename in result.additional_files.items():
                            if filename != self._project_info.layers_parameters[
                                parent_id
                            ].tmp_additional_files.get(desc, ""):
                                os.remove(self._project_info.folder + filename)
                else:
                    for desc, filename in result.additional_files.items():
                        if filename != self._project_info.layers_parameters[
                            self._active_data_processing_task.layer_id
                        ].tmp_additional_files.get(desc, ""):
                            os.remove(self._project_info.folder + filename)
            if result.error != "":
                if not (
                    self._active_data_processing_task.layer_id
                    and self._active_data_processing_task.layer_id
                    in self._layer_loader.state.deleted_layers
                ):
                    show_error(result.error)
            self._active_data_processing_task = None
            return

        if self._active_data_processing_task.layer_id:
            if (
                self._active_data_processing_task.layer_id
                in self._layer_loader.state.deleted_layers
                or not self._project_info.layers_parameters[
                    self._active_data_processing_task.layer_id
                ].tmp_preview
                or (
                    isinstance(result.data, SegmentationData)
                    and self._project_info.layers_parameters[
                        self._active_data_processing_task.layer_id
                    ].tmp_preview_partially_fixed
                    and not self._active_data_processing_task.layer_params_partially_fixed
                )
            ):
                parameters = self._project_info.layers_parameters[
                    self._active_data_processing_task.layer_id
                ]
                if isinstance(result.data, SurfaceData):
                    os.remove(self._project_info.folder + result.data.mesh_file)
                    os.remove(self._project_info.folder + result.data.tif_file)
                elif isinstance(result.data, SegmentationData):
                    deleted_files = (
                        set(result.data.spines_files).union(
                            result.data.adjusted_spines_files
                        )
                        - set(parameters.tmp_spines_files)
                        - set(parameters.tmp_adjusted_spines_files)
                    )
                    for file in deleted_files:
                        os.remove(self._project_info.folder + file)
                elif isinstance(result.data, FinalSegmentationData):
                    if parameters.tmp_mesh_file != result.data.mesh_file:
                        os.remove(self._project_info.folder + result.data.mesh_file)
                    deleted_files = (
                        set(result.data.spines_files).union(
                            result.data.adjusted_spines_files
                        )
                        - set(parameters.tmp_spines_files)
                        - set(parameters.tmp_adjusted_spines_files)
                    )
                    for file in deleted_files:
                        os.remove(self._project_info.folder + file)
                for desc, filename in result.additional_files.items():
                    if filename != parameters.tmp_additional_files.get(desc, ""):
                        os.remove(self._project_info.folder + filename)
                self._active_data_processing_task = None
                return

            layer_type = self._active_data_processing_task.type
            if len(result.additional_files) > 0:
                if layer_type == Types.POINTS:
                    parent_id = self._project_info.data.layers[
                        self._active_data_processing_task.layer_id
                    ].parent_id
                    if not self._project_info.layers_parameters[
                        parent_id
                    ].last_change_time or (
                        self._active_data_processing_task.creation_time
                        > self._project_info.layers_parameters[
                            parent_id
                        ].last_change_time
                    ):
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update = False
                        self._layer_loader.replace_additional_files(
                            parent_id, result.additional_files
                        )
                elif layer_type == Types.NECKS:
                    parent_id = self._project_info.data.layers[
                        self._active_data_processing_task.layer_id
                    ].parent_id
                    parent_id = self._project_info.data.layers[parent_id].parent_id
                    if not self._project_info.layers_parameters[
                        parent_id
                    ].last_change_time or (
                        self._active_data_processing_task.creation_time
                        > self._project_info.layers_parameters[
                            parent_id
                        ].last_change_time
                    ):
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update = False
                        self._layer_loader.replace_additional_files(
                            parent_id, result.additional_files
                        )
                else:
                    self._layer_loader.replace_additional_files(
                        self._active_data_processing_task.layer_id,
                        result.additional_files,
                    )

            if (
                layer_type == Types.POLYGON_MESH
                or layer_type == Types.POLYGON_MESH_SEGMENTATION
            ):
                self._layer_loader.replace_mesh_files(
                    self._active_data_processing_task.layer_id, result.data
                )

            layer_to_update = None
            for layer in self._next_stage_list:
                if self._active_data_processing_task.layer_id == layer._id:
                    layer_to_update = layer

            if layer_type == Types.POLYGON_MESH_SEGMENTATION:
                if (
                    self._project_info.layers_parameters[
                        self._active_data_processing_task.layer_id
                    ].tmp_metadata.get("spine_blocked", None)
                    is not None
                ):
                    if (
                        self._project_info.layers_parameters[
                            self._active_data_processing_task.layer_id
                        ].tmp_metadata["spine_blocked"]
                        <= self._active_data_processing_task.creation_time
                    ):
                        del self._project_info.layers_parameters[
                            self._active_data_processing_task.layer_id
                        ].tmp_metadata["spine_blocked"]
                self._project_info.layers_parameters[
                    self._active_data_processing_task.layer_id
                ].tmp_deleted_spines = result.data.deleted_spines

            layer_metadata = self._extract_result_layer_metadata(result.metadata)

            self._layer_loader.update_layer(
                self._active_data_processing_task.layer_id,
                result.data,
                layer_to_update,
                layer_metadata,
            )

            if layer_type == Types.POINTS and isinstance(layer_to_update, Points):
                self._sync_points_pair_metadata(layer_to_update, result.metadata)

            self._layer_loader.save_layer_data(
                self._active_data_processing_task.layer_id, result.data
            )

            self._saved = False
            if layer_to_update and layer_to_update == self._viewer_model.active_layer:
                if (
                    self._current_preview_options is not self._empty_options
                    and self._current_preview_options is not self._no_parameters
                ):
                    self._current_preview_options.block_dtor_effects()
                self._set_preview_options()
        else:
            layer_type = self._active_data_processing_task.type

            layer_metadata = self._extract_result_layer_metadata(result.metadata)

            params = {
                "layer_type": layer_type,
                "name": result.metadata["name"],
                "parent_id": result.metadata["parent_id"],
                "data": result.data,
                "metadata": layer_metadata,
                "additional_files": result.additional_files,
            }

            self._layer_loader_task_queue.append(
                LayerLoaderTask(
                    TaskType.CREATE,
                    progress(total=0, desc="create layer in queue"),
                    params,
                )
            )
            self._next_layer_loader_task()

        self._active_data_processing_task = None

    def _apply_data_processing_progress(self, progress_update: ProgressUpdate) -> None:
        if self._active_data_processing_task is None:
            return
        pbar = self._active_data_processing_task.pbar
        if pbar is None:
            return
        if progress_update.total is not None:
            pbar.total = int(progress_update.total)
        if progress_update.description:
            pbar.set_description(progress_update.description)
        pbar.set_value(progress_update.value)

    def _next_data_processing_task(self) -> None:
        for task in self._data_processing_task_queue:
            if not task.pbar:
                task.pbar = progress(desc=task.pbar_desc)
            self._connect_progress_cancel(task.pbar)

        if self._closing:
            return
        if self._active_data_processing_task is not None:
            return
        if len(self._data_processing_task_queue) == 0:
            return

        while not self._data_processing_communication_queue_out.empty():
            self._data_processing_communication_queue_out.get()

        self._active_data_processing_task = self._data_processing_task_queue.pop(0)
        if not self._active_data_processing_task.pbar:
            self._active_data_processing_task.pbar = progress()
        self._connect_progress_cancel(self._active_data_processing_task.pbar)

        if self._active_data_processing_task.layer_id:
            self._active_data_processing_task.pbar.set_description(
                self._active_data_processing_task.type.value + " update in progress"
            )
        else:
            self._active_data_processing_task.pbar.set_description(
                self._active_data_processing_task.type.value + " in progress"
            )

        if self._active_data_processing_task.type == Types.POINTS:
            parent_id = self._active_data_processing_task.parent_id
            self._active_data_processing_task.args.update(
                {
                    "additional_files": (
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_additional_files
                        if self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update
                        == False
                        else {}
                    )
                }
            )
        elif self._active_data_processing_task.type == Types.POLYGON_MESH_SEGMENTATION:
            if "spine_id" not in self._active_data_processing_task.args:
                parent_id = self._active_data_processing_task.parent_id
                self._active_data_processing_task.args.update(
                    {
                        "additional_files": self._project_info.layers_parameters[
                            parent_id
                        ].tmp_additional_files
                    }
                )
            else:
                self._active_data_processing_task.args.update(
                    {
                        "additional_files": self._project_info.layers_parameters[
                            self._active_data_processing_task.layer_id
                        ].tmp_additional_files
                    }
                )
        elif self._active_data_processing_task.type == Types.NECKS:
            parent_id = self._active_data_processing_task.parent_id
            parent_id = self._project_info.data.layers[parent_id].parent_id
            self._active_data_processing_task.args.update(
                {
                    "additional_files": (
                        self._project_info.layers_parameters[
                            parent_id
                        ].tmp_additional_files
                        if self._project_info.layers_parameters[
                            parent_id
                        ].tmp_preview_need_update
                        == False
                        else {}
                    )
                }
            )

        self._active_data_processing_task.process = Process(
            target=self._active_data_processing_task.processor,
            kwargs=self._active_data_processing_task.args,
        )
        self._active_data_processing_task.process.start()

    def _next_layer_loader_task(self) -> None:
        if self._active_layer_loader_task is not None:
            return
        if len(self._layer_loader_task_queue) == 0:
            return
        if (
            self._closing
            and self._active_data_processing_task is not None
            and self._layer_loader_task_queue[0].type == TaskType.SAVE
        ):
            return

        while not self._layer_loader_communication_queue_out.empty():
            self._layer_loader_communication_queue_out.get()

        self._active_layer_loader_task = self._layer_loader_task_queue.pop(0)
        self._connect_progress_cancel(self._active_layer_loader_task.pbar)

        if self._active_layer_loader_task.type == TaskType.CREATE:
            self._create()
        elif self._active_layer_loader_task.type == TaskType.DELETE:
            self._delete()
        elif self._active_layer_loader_task.type == TaskType.RESTORE:
            self._restore()
        elif self._active_layer_loader_task.type == TaskType.SAVE:
            self._save()
        elif self._active_layer_loader_task.type == TaskType.EXPORT:
            self._export()
        else:
            self._load()

    def _task_in_queue(self, task_type) -> bool:
        for task in self._layer_loader_task_queue:
            if task.type == task_type:
                return True
        return False

    def _connect_progress_cancel(self, pbar: progress) -> None:
        if pbar is None or getattr(pbar, "_cancel_connected", False):
            return
        pbar._cancel_connected = True
        pbar.cancel_requested.connect(partial(self._kill_background_task, pbar))

    @staticmethod
    def _terminate_process_tree(process: Process | None) -> None:
        if process is None or process.pid is None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif process.is_alive():
            process.terminate()
        process.join(timeout=2)

    def _kill_background_task(self, pbar: progress) -> None:
        if (
            self._active_data_processing_task is not None
            and self._active_data_processing_task.pbar is pbar
        ):
            task = self._active_data_processing_task
            self._terminate_process_tree(task.process)
            task.pbar = None
            self._active_data_processing_task = None
            while not self._data_processing_communication_queue_in.empty():
                self._data_processing_communication_queue_in.get()
            pbar.close()
            return

        if (
            self._active_layer_loader_task is not None
            and self._active_layer_loader_task.pbar is pbar
        ):
            self._terminate_process_tree(self._active_layer_loader_task.process)
            self._active_layer_loader_task = None
            while not self._layer_loader_communication_queue_in.empty():
                self._layer_loader_communication_queue_in.get()
            pbar.close()
            self._next_layer_loader_task()
            return

        for queue in (self._data_processing_task_queue, self._layer_loader_task_queue):
            for index, task in enumerate(queue):
                if task.pbar is pbar:
                    task.pbar = None
                    del queue[index]
                    pbar.close()
                    return

    def _is_layer_deleting(self) -> bool:
        return (
            self._active_layer_loader_task
            and self._active_layer_loader_task.type == TaskType.DELETE
        ) or self._task_in_queue(TaskType.DELETE)

    def _load_initial_state(self) -> None:
        try:
            os.mkdir(self._project_info.folder + TMP)
            os.mkdir(self._project_info.folder + TMP_LAYERS_PATH)
            os.mkdir(self._project_info.folder + TMP_AUXILIARY_PATH)
        except:
            pass

        infos = []
        if len(self._project_info.active_layers) == 0:
            infos.append((0, LayerType.ACTIVE))
            for child_id in self._project_info.data.layers[0].child_layers:
                infos.append((child_id, LayerType.NEXT_STAGE))
        else:
            for id in self._project_info.active_layers:
                infos.append((id, LayerType.ACTIVE))
            for child_id in self._project_info.data.layers[id].child_layers:
                infos.append((child_id, LayerType.NEXT_STAGE))

        for background_image_id in self._project_info.background_images:
            infos.append((background_image_id, LayerType.ADDITIONAL))

        for additional_layer_info in self._project_info.non_editable_layers:
            infos.append((additional_layer_info, LayerType.ADDITIONAL))

        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.LOAD,
                progress(total=0, desc="project loading"),
                {"infos": infos},
            )
        )
        self._next_layer_loader_task()

    def _on_initial_state_loaded(self, loaded) -> None:
        if not loaded or self._closing:
            self._window.set_central_widget()
            self._window.set_menus()
            self._window.block_project_specific_actions()
            self._close()
            return

        self._project_info.active_layers = []
        self._project_info.background_images = []
        self._project_info.non_editable_layers = []

        for layer in self._additional_layers_list:
            if layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                continue
            linked = False
            for active_layer in self._active_layers_list:
                if active_layer._id == layer._id:
                    self._layer_loader.parameters_tracker.link_layers(
                        layer, active_layer
                    )
                    linked = True
                    break
            if not linked:
                for next_stage_layer in self._next_stage_list:
                    if next_stage_layer._id == layer._id:
                        self._layer_loader.parameters_tracker.link_layers(
                            layer, next_stage_layer
                        )
                        break

        self._window.set_title(self._project_info.name)
        self._window.add_dock_widget(
            self._project_info_widget,
            "project information",
            "right",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._preview_options_widget,
            "preview options",
            "right",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._layer_controls_widget,
            "layer controls",
            "right",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._viewer_buttons,
            "viewer buttons",
            "right",
            ["right"],
        )
        self._window.add_dock_widget(
            self._qt_active_layer_list,
            "active stages",
            "left",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._qt_next_stage_list,
            "next stage selection",
            "left",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._qt_additional_layer_list,
            "additional layers",
            "left",
            ["right", "left"],
        )
        self._window.add_dock_widget(
            self._control_buttons,
            "control buttons",
            "left",
            ["left"],
        )

        self._control_buttons.buttons.deleteButton.setEnabled(False)
        self._control_buttons.buttons.restoreButton.setEnabled(False)
        self._control_buttons.buttons.selectedStageButton.setEnabled(False)
        self._control_buttons.buttons.exportButton.setEnabled(False)
        self._control_buttons.buttons.addLayerToNonEditableButton.setEnabled(False)
        enable = len(self._active_layers_list) > 1
        self._control_buttons.buttons.previousStageButton.setEnabled(enable)
        self._control_buttons.buttons.imageStageButton.setEnabled(enable)

        self._viewer_model.layers_sorter = layers_sorter
        self._viewer_model.active_layer = None
        self._viewer_model.status_.connect(self._window.on_status_changed)
        self._set_create_new_stage_button(
            self._active_layers_list[-1].metadata["type"],
            self._active_layers_list[-1].metadata["type"] == Types.IMAGE
            and self._project_info.layers_parameters[
                self._active_layers_list[-1]._id
            ].preview,
        )
        self._viewer.reset_when_resize = True

        self._loaded = True
        self.active_shortcuts = ["general"]

    def _on_active_layers_selection_changed(self) -> None:
        self._viewer.reset_when_resize = False
        self._qt_additional_layer_list.list.selectionModel().selectionChanged.disconnect(
            self._on_additional_layers_selection_changed
        )
        self._qt_additional_layer_list.list.selectionModel().clearSelection()
        self._qt_additional_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_additional_layers_selection_changed
        )
        self._qt_next_stage_list.next_stage_list.selectionModel().selectionChanged.disconnect(
            self._on_next_stage_selection_changed
        )
        self._qt_next_stage_list.next_stage_list.selectionModel().clearSelection()
        self._qt_next_stage_list.next_stage_list.selectionModel().selectionChanged.connect(
            self._on_next_stage_selection_changed
        )
        self._viewer_model.active_layer = self._qt_active_layer_list.list.activeItem
        self._set_layer_controls()
        self._set_preview_options()
        self._control_buttons.buttons.deleteButton.setEnabled(False)
        self._control_buttons.buttons.selectedStageButton.setEnabled(
            self._qt_active_layer_list.list.activeItem is not None
            and self._active_layers_list[-1]
            is not self._qt_active_layer_list.list.activeItem
        )
        self._control_buttons.buttons.exportButton.setEnabled(
            self._qt_active_layer_list.list.activeItem is not None
        )
        self._control_buttons.buttons.addLayerToNonEditableButton.setEnabled(
            self._qt_active_layer_list.list.activeItem is not None
            and not (
                self._qt_active_layer_list.list.activeItem.metadata["type"]
                == Types.IMAGE
                and self._project_info.layers_parameters[
                    self._qt_active_layer_list.list.activeItem._id
                ].preview
            )
        )

    def _on_additional_layers_selection_changed(self) -> None:
        self._viewer.reset_when_resize = False
        self._qt_active_layer_list.list.selectionModel().selectionChanged.disconnect(
            self._on_active_layers_selection_changed
        )
        self._qt_active_layer_list.list.selectionModel().clearSelection()
        self._qt_active_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_active_layers_selection_changed
        )
        self._qt_next_stage_list.next_stage_list.selectionModel().selectionChanged.disconnect(
            self._on_next_stage_selection_changed
        )
        self._qt_next_stage_list.next_stage_list.selectionModel().clearSelection()
        self._qt_next_stage_list.next_stage_list.selectionModel().selectionChanged.connect(
            self._on_next_stage_selection_changed
        )
        self._viewer_model.active_layer = self._qt_additional_layer_list.list.activeItem
        self._set_layer_controls()
        self._set_preview_options()
        self._control_buttons.buttons.deleteButton.setEnabled(
            not self._is_layer_deleting()
            and self._qt_additional_layer_list.list.activeItem is not None
        )
        self._control_buttons.buttons.selectedStageButton.setEnabled(False)
        self._control_buttons.buttons.exportButton.setEnabled(
            self._qt_additional_layer_list.list.activeItem is not None
        )
        self._control_buttons.buttons.addLayerToNonEditableButton.setEnabled(False)

    def _on_next_stage_selection_changed(self) -> None:
        self._viewer.reset_when_resize = False
        self._qt_active_layer_list.list.selectionModel().selectionChanged.disconnect(
            self._on_active_layers_selection_changed
        )
        self._qt_active_layer_list.list.selectionModel().clearSelection()
        self._qt_active_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_active_layers_selection_changed
        )
        self._qt_additional_layer_list.list.selectionModel().selectionChanged.disconnect(
            self._on_additional_layers_selection_changed
        )
        self._qt_additional_layer_list.list.selectionModel().clearSelection()
        self._qt_additional_layer_list.list.selectionModel().selectionChanged.connect(
            self._on_additional_layers_selection_changed
        )
        self._viewer_model.active_layer = (
            self._qt_next_stage_list.next_stage_list.activeItem
        )
        self._set_layer_controls()
        self._set_preview_options()
        self._control_buttons.buttons.deleteButton.setEnabled(
            not self._is_layer_deleting()
            and self._qt_next_stage_list.next_stage_list.activeItem is not None
        )
        is_preview = False
        if self._viewer_model.active_layer:
            is_preview = self._project_info.layers_parameters[
                self._viewer_model.active_layer._id
            ].tmp_preview
        self._control_buttons.buttons.selectedStageButton.setEnabled(
            not is_preview
            and self._qt_next_stage_list.next_stage_list.activeItem is not None
            and len(self._active_layers_list) < 3 + self._layer_offset_index
        )
        self._control_buttons.buttons.exportButton.setEnabled(
            self._qt_next_stage_list.next_stage_list.activeItem is not None
        )
        self._control_buttons.buttons.addLayerToNonEditableButton.setEnabled(
            not is_preview
            and self._qt_next_stage_list.next_stage_list.activeItem is not None
        )

    def _set_layer_controls(self) -> None:
        layer = self._viewer_model.active_layer

        delete = self._current_layer_controls is not self._empty_controls
        if not layer:
            self._current_layer_controls = self._empty_controls
            self._layer_controls_widget.replace_widget(self._empty_controls, delete)
            self.active_shortcuts = ["general"]
            return

        self._current_layer_controls = create_qt_layer_controls(
            layer, self._window._qt_window
        )
        self._layer_controls_widget.replace_widget(
            self._current_layer_controls,
            delete,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        self._current_layer_controls.ndisplay = self._viewer_model.dims.ndisplay
        if isinstance(layer, Image):
            self.active_shortcuts = [
                "general",
                "image",
            ]
        elif isinstance(layer, Labels):
            self.active_shortcuts = [
                "general",
                "labels",
            ]
        elif isinstance(layer, Points):
            self.active_shortcuts = [
                "general",
                "points",
            ]

    def _is_binary_voxel_correction_mode(self) -> bool:
        return self._project_info.subtype == constants.BINARY_VOXEL_CORRECTION

    def _set_create_new_stage_button(self, stage, blocked=False) -> None:
        self._qt_next_stage_list.show_run_options(
            stage in (Types.NECKS, Types.VOXEL_MESH_SEGMENTATION)
            and not self._is_binary_voxel_correction_mode()
        )
        try:
            self._qt_next_stage_list.create_new_button.clicked.disconnect()
        except:
            pass
        if stage == Types.IMAGE:
            self._qt_next_stage_list.create_new_button.setText("create new mask")
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_mask
            )
        elif stage == Types.MASK:
            if self._is_binary_voxel_correction_mode():
                self._qt_next_stage_list.create_new_button.setText(
                    "create new voxel segmentation"
                )
                self._qt_next_stage_list.create_new_button.clicked.connect(
                    self._create_new_voxel_segmentation
                )
            else:
                self._qt_next_stage_list.create_new_button.setText(
                    "create new binarization"
                )
                self._qt_next_stage_list.create_new_button.clicked.connect(
                    self._create_new_binarization
                )
        elif stage == Types.BINARIZATION:
            self._qt_next_stage_list.create_new_button.setText("create new points")
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_points
            )
        elif stage == Types.POINTS:
            self._qt_next_stage_list.create_new_button.setText("create new necks")
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_necks
            )
        elif stage == Types.NECKS:
            self._qt_next_stage_list.create_new_button.setText("create new mesh")
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_mesh
            )
        elif stage == Types.POLYGON_MESH:
            self._qt_next_stage_list.create_new_button.setText(
                "create new mesh segmentation"
            )
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_mesh_segmentation
            )
        elif stage == Types.POLYGON_MESH_SEGMENTATION:
            self._qt_next_stage_list.create_new_button.setText(
                "create new voxel segmentation"
            )
            self._qt_next_stage_list.create_new_button.clicked.connect(
                self._create_new_voxel_segmentation
            )
        elif stage == Types.VOXEL_MESH_SEGMENTATION:
            if self._is_binary_voxel_correction_mode():
                self._qt_next_stage_list.create_new_button.setText(
                    "voxel correction is final stage"
                )
                blocked = True
            else:
                self._qt_next_stage_list.create_new_button.setText(
                    "create new final segmentation"
                )
                self._qt_next_stage_list.create_new_button.clicked.connect(
                    self._create_new_final_segmentation
                )
        self._qt_next_stage_list.create_new_button.setEnabled(not blocked)

    def _set_preview_options(self) -> None:
        layer = self._viewer_model.active_layer

        delete = (
            self._current_preview_options is not self._empty_options
            and self._current_preview_options is not self._no_parameters
        )
        if not layer:
            self._current_preview_options = self._empty_options
            self._preview_options_widget.replace_widget(self._empty_options, delete)
            return

        if layer.metadata["type"] == Types.IMAGE:
            self._current_preview_options = QtImagePreviewOptions(
                self._project_info.original_shape,
                self._project_info.layers_parameters[layer._id].tmp_metadata,
                not self._project_info.layers_parameters[layer._id].tmp_preview,
            )
            self._current_preview_options.parameters_fixed_.connect(
                self._fix_layer_parameters
            )
            self._current_preview_options.parameters_changed_.connect(
                self._update_image_parameters
            )
            self._preview_options_widget.replace_widget(
                self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
            )
        elif layer.metadata["type"] == Types.BINARIZATION:
            self._current_preview_options = QtBinarizationPreviewOptions(
                self._project_info.layers_parameters[layer._id].tmp_metadata,
                not self._project_info.layers_parameters[layer._id].tmp_preview,
            )
            self._current_preview_options.parameters_fixed_.connect(
                self._fix_layer_parameters
            )
            self._current_preview_options.parameters_changed_.connect(
                self._update_binarization
            )
            self._preview_options_widget.replace_widget(
                self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
            )
        elif layer.metadata["type"] == Types.POINTS:
            layer_params = self._project_info.layers_parameters[layer._id]
            if layer_params.tmp_preview:
                layer.metadata.setdefault("neck_edit_phase", "spine")
                layer.metadata["allow_preview_point_editing"] = True
                layer.editable_.emit()

                if layer._id not in self._neck_points_data_callbacks:
                    callback = partial(self._on_neck_edit_points_changed, layer)
                    layer.data_.connect(callback)
                    self._neck_points_data_callbacks[layer._id] = callback

            points_params = self._get_points_pair_ui_params(layer)

            self._current_preview_options = QtPointsPreviewOptions(
                points_params,
                not self._project_info.layers_parameters[layer._id].tmp_preview,
            )

            self._current_preview_options.parameters_fixed_.connect(
                self._fix_layer_parameters
            )
            self._current_preview_options.parameters_changed_.connect(
                self._update_points
            )

            self._current_preview_options.shaft_point_changed_.connect(
                self._update_manual_shaft_point
            )
            self._current_preview_options.pair_active_changed_.connect(
                self._update_neck_pair_active
            )
            self._current_preview_options.shaft_point_reset_.connect(
                self._reset_manual_shaft_point
            )
            self._current_preview_options.selected_pair_changed_.connect(
                self._highlight_selected_neck_pair
            )
            self._current_preview_options.spine_points_fixed_.connect(
                self._start_shaft_points_editing
            )

            self._preview_options_widget.replace_widget(
                self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
            )

            self._highlight_selected_neck_pair(0)
        elif layer.metadata["type"] == Types.NECKS:
            self._current_preview_options = QtNecksPreviewOptions(
                self._project_info.layers_parameters[layer._id].tmp_metadata,
                not self._project_info.layers_parameters[layer._id].tmp_preview,
            )
            self._current_preview_options.parameters_fixed_.connect(
                self._fix_layer_parameters
            )
            self._current_preview_options.parameters_changed_.connect(
                self._update_necks
            )
            self._preview_options_widget.replace_widget(
                self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
            )
        elif layer.metadata["type"] == Types.POLYGON_MESH_SEGMENTATION:
            self._current_preview_options = QtMeshSegmentationPreviewOptions(
                layer,
                self._viewer_model.camera,
                self._project_info.folder,
                self._project_info.layers_parameters[layer._id].tmp_additional_files[
                    "spines"
                ],
                self._project_info.layers_parameters[layer._id].tmp_deleted_spines,
                self._project_info.layers_parameters[layer._id].tmp_metadata,
                not self._project_info.layers_parameters[layer._id].tmp_preview,
                self._project_info.layers_parameters[
                    layer._id
                ].tmp_preview_partially_fixed,
            )
            self._current_preview_options.parameters_fixed_.connect(
                self._fix_layer_parameters
            )
            self._current_preview_options.parameters_changed_.connect(
                self._update_mesh_segmentation
            )
            self._current_preview_options.spine_parameters_changed_.connect(
                self._update_spine
            )
            self._current_preview_options.spine_deleted_.connect(self._on_spine_deleted)
            self._current_preview_options.spine_restored_.connect(
                self._on_spine_restored
            )
            self._current_preview_options.camera_policy_changed_.connect(
                self._on_camera_policy_changed
            )
            self._current_preview_options.current_spine_changed_.connect(
                self._on_current_spine_changed
            )
            self._current_preview_options.parameters_fixed_partially_.connect(
                self._fix_partially_layer_parameters
            )
            self._preview_options_widget.replace_widget(
                self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
            )
        elif layer.metadata["type"] == Types.FINAL_SEGMENTATION:
            self._current_preview_options = QtFinalSegmentationPreviewOptions(
                layer,
                self._viewer_model.camera,
                self._project_info.folder,
                self._project_info.layers_parameters[layer._id].tmp_additional_files[
                    "spines"
                ],
                self._project_info.layers_parameters[layer._id].tmp_metadata,
            )
            if len(self._current_preview_options._spines["spines"]) == 0:
                self._current_preview_options.deleteLater()
                self._current_preview_options = self._no_parameters
                self._preview_options_widget.replace_widget(self._no_parameters, delete)
            else:
                self._current_preview_options.camera_policy_changed_.connect(
                    self._on_camera_policy_changed
                )
                self._current_preview_options.current_spine_changed_.connect(
                    self._on_current_spine_changed
                )
                self._preview_options_widget.replace_widget(
                    self._current_preview_options, delete, Qt.AlignmentFlag.AlignTop
                )
        else:
            self._current_preview_options = self._no_parameters
            self._preview_options_widget.replace_widget(self._no_parameters, delete)

        if (
            self._current_preview_options is not self._no_parameters
            and self._current_preview_options is not self._empty_options
        ):
            self._current_preview_options.ndisplay = self._viewer_model.dims.ndisplay

    def _fix_layer_parameters(self) -> None:
        layer = self._viewer_model.active_layer
        if not layer:
            return

        if (
            isinstance(layer, Points)
            and layer.metadata.get("neck_edit_phase") == "shaft"
        ):
            shaft_points = np.asarray(layer.data, dtype=np.uint16).reshape((-1, 3))
            spine_points = np.asarray(
                layer.metadata.get("spine_points", []), dtype=np.uint16
            ).reshape((-1, 3))
            if shaft_points.shape != spine_points.shape:
                show_error(
                    "Cannot fix points: spine points and shaft points are not synchronized."
                )
                return

            pair_source = self._normalise_pair_source_list(
                layer.metadata.get("pair_source", None), len(spine_points), "manual"
            )
            self._save_points_pair_metadata(
                layer,
                {
                    "spine_points": spine_points.tolist(),
                    "shaft_points": shaft_points.tolist(),
                    "pair_source": pair_source,
                    "pair_edit_state": self._pair_edit_state_from_sources(pair_source),
                },
            )

            style = layer.metadata.get("neck_spine_style", {})
            self._neck_points_phase_switch = True
            try:
                layer.data = spine_points
                layer.symbol = style.get("symbol", "disc")
                layer.face_color = style.get("face_color", "white")
                layer.edge_color = style.get("edge_color", "white")
            finally:
                self._neck_points_phase_switch = False

            layer.metadata.pop("neck_edit_phase", None)
            layer.metadata.pop("allow_preview_point_editing", None)
            layer.metadata.pop("neck_spine_style", None)
            layer_params = self._project_info.layers_parameters[layer._id]
            for transient_key in (
                "neck_edit_phase",
                "allow_preview_point_editing",
                "neck_spine_style",
            ):
                layer_params.metadata.pop(transient_key, None)
                layer_params.tmp_metadata.pop(transient_key, None)
            self._layer_loader.save_layer_data(layer._id, spine_points)
            self._saved = False

        self._stop_updating()
        if layer.metadata["type"] == Types.BINARIZATION:
            layer_params = self._project_info.layers_parameters[layer._id]
            if isinstance(self._current_preview_options, QtBinarizationPreviewOptions):
                current_params = self._current_preview_options.get_params()
                layer_params.tmp_metadata.update(current_params)
                layer_params.metadata.update(current_params)
            fill_holes = bool(
                layer_params.tmp_metadata.get("fill_holes_on_fix", False)
            )
            if fill_holes:
                filled = fill_small_enclosed_holes(layer.data)
                self._layer_loader.update_layer(layer._id, filled, layer)
                self._layer_loader.save_layer_data(layer._id, filled)
                layer_params.tmp_metadata["holes_filled_on_fix"] = True
                layer_params.metadata["holes_filled_on_fix"] = True
                self._saved = False
        if layer.metadata["type"] == Types.IMAGE:
            x_range = self._project_info.layers_parameters[0].tmp_metadata.get(
                "x_range", [0, self._project_info.shape[2] - 1]
            )
            y_range = self._project_info.layers_parameters[0].tmp_metadata.get(
                "y_range", [0, self._project_info.shape[2] - 1]
            )
            z_range = self._project_info.layers_parameters[0].tmp_metadata.get(
                "z_range", [0, self._project_info.shape[2] - 1]
            )
            data = layer.data[
                z_range[0] : z_range[1] + 1,
                y_range[0] : y_range[1] + 1,
                x_range[0] : x_range[1] + 1,
            ]
            self._layer_loader.update_layer(
                layer._id,
                data,
                layer,
            )
            self._layer_loader.save_layer_data(layer._id, data)
            for additional_layer in self._additional_layers_list:
                if additional_layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                    background_image_data = additional_layer.data[
                        z_range[0] : z_range[1] + 1,
                        y_range[0] : y_range[1] + 1,
                        x_range[0] : x_range[1] + 1,
                    ]
                    self._layer_loader.update_layer(
                        additional_layer._id,
                        background_image_data,
                        additional_layer,
                    )
                    self._layer_loader.save_layer_data(
                        additional_layer._id, background_image_data
                    )
            self._saved = False
            self._project_info.layers_parameters[layer._id].tmp_preview = False
            self._project_info.shape = data.shape
            self._set_create_new_stage_button(Types.IMAGE)
            self._set_preview_options()
            self._create_new_mask()
            self._viewer_model.reset_view()
        elif layer.metadata["type"] == Types.POLYGON_MESH_SEGMENTATION:
            data, additional_files = fix_segmentation(
                self._project_info.layers_parameters[layer._id].tmp_spines_files,
                self._project_info.layers_parameters[
                    layer._id
                ].tmp_adjusted_spines_files,
                self._project_info.shape,
                self._project_info.real_scale,
                self._project_info.min_coordinates,
                self._project_info.folder,
                self._project_info.layers_parameters[layer._id].tmp_additional_files,
                self._project_info.layers_parameters[layer._id].tmp_deleted_spines,
            )
            self._project_info.layers_parameters[layer._id].tmp_deleted_spines = (
                data.deleted_spines
            )
            self._layer_loader.replace_additional_files(layer._id, additional_files)
            self._layer_loader.replace_mesh_files(layer._id, data)
            if (
                self._project_info.layers_parameters[layer._id].tmp_metadata.get(
                    "spine_blocked", None
                )
                is not None
            ):
                del self._project_info.layers_parameters[layer._id].tmp_metadata[
                    "spine_blocked"
                ]
            self._layer_loader.update_layer(
                layer._id,
                data,
                layer,
            )
            self._layer_loader.save_layer_data(layer._id, data)
            self._saved = False
            self._project_info.layers_parameters[layer._id].tmp_preview = False
            if (
                self._current_preview_options is not self._empty_options
                and self._current_preview_options is not self._no_parameters
            ):
                self._current_preview_options.block_dtor_effects()
            self._set_preview_options()
        else:
            self._project_info.layers_parameters[layer._id].tmp_preview = False
        layer.preview = False
        self._control_buttons.buttons.selectedStageButton.setEnabled(
            len(self._active_layers_list) < 3 + self._layer_offset_index
            and len(self._active_layers_list) > 0
            and layer.metadata["type"] != Types.IMAGE
        )
        self._control_buttons.buttons.addLayerToNonEditableButton.setEnabled(True)

    def _stop_updating(self) -> None:
        layer = self._viewer_model.active_layer
        if (
            self._active_data_processing_task
            and self._active_data_processing_task.layer_id == layer._id
        ):
            self._data_processing_communication_queue_out.put("stop")
        indices = []
        for i, task in enumerate(self._data_processing_task_queue):
            if task.layer_id == layer._id:
                indices.append(i)
        indices.reverse()
        for i in indices:
            del self._data_processing_task_queue[i]

    def _update_image_parameters(self) -> None:
        self._project_info.layers_parameters[0].tmp_metadata.update(
            self._current_preview_options.get_params()
        )

    def _create_new_mask(self) -> None:
        self._saved = False
        params = {
            "layer_type": Types.MASK,
            "name": Translater.instance().get_translation("mask"),
            "data": np.zeros(self._project_info.shape, dtype=np.uint8),
            "parent_id": 0,
        }
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.CREATE,
                progress(total=0, desc="create layer in queue"),
                params,
            )
        )
        self._next_layer_loader_task()

    def _create_new_binarization(self) -> None:
        metadata = {
            "name": Translater.instance().get_translation("binarization"),
            "parent_id": self._active_layers_list[1]._id,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.BINARIZATION,
                binarize,
                {
                    "data": self._active_layers_list[0].data,
                    "area": self._active_layers_list[1].data,
                    "parameters": {},
                    "metadata": metadata,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "binarization in queue",
                self._active_layers_list[1]._id,
            )
        )

    def _update_binarization(self) -> None:
        self._stop_updating()
        params = self._current_preview_options.get_params()
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.BINARIZATION,
                binarize,
                {
                    "data": self._active_layers_list[0].data,
                    "area": self._active_layers_list[1].data,
                    "parameters": params,
                    "metadata": {},
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "binarization update in queue",
                self._active_layers_list[1]._id,
                self._viewer_model.active_layer._id,
            )
        )

    def _create_new_points(self) -> None:
        metadata = {
            "name": Translater.instance().get_translation("points"),
            "parent_id": self._active_layers_list[2]._id,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POINTS,
                find_points,
                {
                    "data": self._active_layers_list[2]._data,
                    "scale": self._project_info.real_scale,
                    "parameters": {},
                    "folder": self._project_info.folder,
                    "metadata": metadata,
                    "additional_files": {},
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "points in queue",
                self._active_layers_list[2]._id,
            )
        )

    def _update_points(self) -> None:
        self._stop_updating()
        params = self._current_preview_options.get_params()
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POINTS,
                find_points,
                {
                    "data": self._active_layers_list[2]._data,
                    "scale": self._project_info.real_scale,
                    "parameters": params,
                    "folder": self._project_info.folder,
                    "metadata": {},
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "points update in queue",
                self._active_layers_list[2]._id,
                self._viewer_model.active_layer._id,
            )
        )

    def _extract_result_layer_metadata(self, result_metadata: dict | None) -> dict:
        result_metadata = result_metadata or {}
        layer_metadata = {}

        params_metadata = result_metadata.get("params", {})
        if isinstance(params_metadata, dict):
            layer_metadata.update(params_metadata)

        for key, value in result_metadata.items():
            if key == "params":
                continue
            if key in ("name", "parent_id"):
                continue
            layer_metadata[key] = value

        return layer_metadata

    def _sync_points_pair_metadata(self, points_layer, result_metadata: dict | None) -> None:
        if result_metadata is None:
            return

        pair_keys = (
            "neck_restoration_pairs_version",
            "coordinate_order",
            "spine_points",
            "shaft_points",
            "auto_shaft_points",
            "pair_active",
            "pair_source",
            "pair_edit_state",
            "spine_component_ids",
            "shaft_component_ids",
        )

        pair_metadata = {}

        for key in pair_keys:
            if key in result_metadata:
                pair_metadata[key] = result_metadata[key]

        params_metadata = result_metadata.get("params", {})
        if isinstance(params_metadata, dict):
            for key in pair_keys:
                if key in params_metadata:
                    pair_metadata[key] = params_metadata[key]

        if not pair_metadata:
            return

        points_layer.metadata.update(pair_metadata)

        layer_params = self._project_info.layers_parameters.get(points_layer._id, None)
        if layer_params is not None:
            layer_params.metadata.update(pair_metadata)
            layer_params.tmp_metadata.update(pair_metadata)

    def _get_neck_pair_parameters(self) -> dict:
        points_layer = self._active_layers_list[3]
        layer_params = self._project_info.layers_parameters.get(points_layer._id, None)

        metadata = {}
        metadata.update(points_layer.metadata or {})

        if layer_params is not None:
            metadata.update(getattr(layer_params, "metadata", {}) or {})
            metadata.update(getattr(layer_params, "tmp_metadata", {}) or {})

        params = {}
        for key in (
            "shaft_points",
            "pair_active",
            "pair_source",
            "spine_component_ids",
            "shaft_component_ids",
        ):
            if key in metadata:
                params[key] = metadata[key]

        return params

    def _get_points_pair_metadata(self, points_layer) -> tuple[dict, object | None]:
        layer_params = self._project_info.layers_parameters.get(points_layer._id, None)

        metadata = {}
        metadata.update(points_layer.metadata or {})

        if layer_params is not None:
            metadata.update(getattr(layer_params, "metadata", {}) or {})
            metadata.update(getattr(layer_params, "tmp_metadata", {}) or {})

        # During step 2 the visible layer data contains shaft points.  Pair
        # reconciliation is meaningful only while layer data contains spines.
        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            metadata.update(points_layer.metadata or {})
            metadata["shaft_points"] = np.asarray(points_layer.data).tolist()
            return metadata, layer_params

        if "neck_restoration_pairs_version" not in metadata:
            return metadata, layer_params

        try:
            current_points = np.asarray(points_layer.data, dtype=np.uint16).reshape((-1, 3))
        except Exception:
            return metadata, layer_params

        shaft_points = metadata.get("shaft_points", None)
        if shaft_points is None:
            return metadata, layer_params

        try:
            shaft_arr = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))
        except Exception:
            return metadata, layer_params

        spine_points = metadata.get("spine_points", None)

        if spine_points is None:
            if len(shaft_arr) == len(current_points):
                pair_metadata = {
                    "neck_restoration_pairs_version": 1,
                    "coordinate_order": metadata.get("coordinate_order", "zyx"),
                    "spine_points": current_points.tolist(),
                }
                self._save_points_pair_metadata(points_layer, pair_metadata)
                metadata.update(pair_metadata)
            return metadata, layer_params

        try:
            old_spine_arr = np.asarray(spine_points, dtype=np.uint16).reshape((-1, 3))
        except Exception:
            return metadata, layer_params

        if shaft_arr.shape == current_points.shape and np.array_equal(
            old_spine_arr, current_points
        ):
            return metadata, layer_params

        if len(old_spine_arr) == 0:
            return metadata, layer_params

        source_indices = []
        used_old_indices = set()

        for point in current_points:
            matches = np.where(np.all(old_spine_arr == point, axis=1))[0]
            source_index = None

            for match in matches:
                match = int(match)
                if match not in used_old_indices:
                    source_index = match
                    break

            source_indices.append(source_index)

            if source_index is not None:
                used_old_indices.add(source_index)

        auto_shaft_points = metadata.get("auto_shaft_points", None)
        if auto_shaft_points is None:
            auto_shaft_arr = shaft_arr.copy()
        else:
            try:
                auto_shaft_arr = np.asarray(auto_shaft_points, dtype=np.uint16).reshape((-1, 3))
            except Exception:
                auto_shaft_arr = shaft_arr.copy()

        old_size = len(old_spine_arr)

        if len(shaft_arr) < old_size:
            padded = np.zeros((old_size, 3), dtype=np.uint16)
            padded[: len(shaft_arr)] = shaft_arr
            shaft_arr = padded
        else:
            shaft_arr = shaft_arr[:old_size]

        if len(auto_shaft_arr) < old_size:
            padded = np.zeros((old_size, 3), dtype=np.uint16)
            padded[: len(auto_shaft_arr)] = auto_shaft_arr
            auto_shaft_arr = padded
        else:
            auto_shaft_arr = auto_shaft_arr[:old_size]

        old_pair_active = self._normalise_pair_bool_list(
            metadata.get("pair_active", None),
            old_size,
            True,
        )
        old_pair_source = self._normalise_pair_source_list(
            metadata.get("pair_source", None),
            old_size,
            "auto_old_medial",
        )

        old_spine_component_ids = list(metadata.get("spine_component_ids", []))
        if len(old_spine_component_ids) < old_size:
            old_spine_component_ids.extend([None] * (old_size - len(old_spine_component_ids)))
        old_spine_component_ids = old_spine_component_ids[:old_size]

        old_shaft_component_ids = list(metadata.get("shaft_component_ids", []))
        if len(old_shaft_component_ids) < old_size:
            old_shaft_component_ids.extend([None] * (old_size - len(old_shaft_component_ids)))
        old_shaft_component_ids = old_shaft_component_ids[:old_size]

        new_shaft_points = []
        new_auto_shaft_points = []
        new_pair_active = []
        new_pair_source = []
        new_spine_component_ids = []
        new_shaft_component_ids = []

        for current_point, source_index in zip(current_points, source_indices):
            if source_index is None:
                new_shaft_points.append(current_point.astype(np.uint16).tolist())
                new_auto_shaft_points.append(current_point.astype(np.uint16).tolist())
                new_pair_active.append(False)
                new_pair_source.append("manual_needs_shaft")
                new_spine_component_ids.append(None)
                new_shaft_component_ids.append(None)
            else:
                new_shaft_points.append(shaft_arr[source_index].astype(np.uint16).tolist())
                new_auto_shaft_points.append(
                    auto_shaft_arr[source_index].astype(np.uint16).tolist()
                )
                new_pair_active.append(bool(old_pair_active[source_index]))
                new_pair_source.append(str(old_pair_source[source_index]))
                new_spine_component_ids.append(old_spine_component_ids[source_index])
                new_shaft_component_ids.append(old_shaft_component_ids[source_index])

        pair_metadata = {
            "neck_restoration_pairs_version": 1,
            "coordinate_order": metadata.get("coordinate_order", "zyx"),
            "spine_points": current_points.tolist(),
            "shaft_points": new_shaft_points,
            "auto_shaft_points": new_auto_shaft_points,
            "pair_active": new_pair_active,
            "pair_source": new_pair_source,
            "pair_edit_state": self._pair_edit_state_from_sources(new_pair_source),
            "spine_component_ids": new_spine_component_ids,
            "shaft_component_ids": new_shaft_component_ids,
        }

        self._save_points_pair_metadata(points_layer, pair_metadata)
        metadata.update(pair_metadata)

        return metadata, layer_params

    def _get_points_pair_ui_params(self, points_layer) -> dict:
        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            metadata = dict(points_layer.metadata or {})
            points_arr = np.asarray(metadata.get("spine_points", []))
            metadata["shaft_points"] = np.asarray(points_layer.data).tolist()
        else:
            metadata, _ = self._get_points_pair_metadata(points_layer)
            points_arr = np.asarray(points_layer.data)

        params = {}
        params.update(metadata)
        params["spine_points"] = points_arr.tolist()
        params["pairs_count"] = int(len(points_arr))

        try:
            params["image_shape"] = tuple(int(v) for v in self._active_layers_list[0].data.shape)
        except Exception:
            params["image_shape"] = tuple(int(v) for v in self._project_info.original_shape)

        return params

    def _start_shaft_points_editing(self) -> None:
        """Freeze spine points and turn the preview layer into a shaft editor."""
        layer = self._viewer_model.active_layer
        if not layer or not isinstance(layer, Points):
            return
        if layer.metadata.get("neck_edit_phase", "spine") != "spine":
            return

        spine_points = np.asarray(layer.data, dtype=np.int64).reshape((-1, 3))
        metadata, _ = self._get_points_pair_metadata(layer)
        old_shaft = np.asarray(
            metadata.get("shaft_points", spine_points), dtype=np.int64
        ).reshape((-1, 3))
        pair_source = self._normalise_pair_source_list(
            metadata.get("pair_source", None), len(spine_points), "manual_needs_shaft"
        )
        pair_active = self._normalise_pair_bool_list(
            metadata.get("pair_active", None), len(spine_points), True
        )

        if old_shaft.shape != spine_points.shape:
            old_shaft = spine_points.copy()

        shaft_points = old_shaft.copy()
        failed = []
        from projects.segmentation.utils.data_processing.paired_necks import (
            snap_to_valid_shaft_point,
        )

        binarization = self._active_layers_list[2]._data
        for index, spine_point in enumerate(spine_points):
            if pair_source[index] != "manual_needs_shaft":
                continue
            try:
                suggested, _ = snap_to_valid_shaft_point(
                    spine_point,
                    binarization,
                    self._project_info.real_scale,
                    max_distance_um=None,
                )
                shaft_points[index] = suggested
                pair_source[index] = "auto_shaft_snap"
                pair_active[index] = True
            except Exception:
                shaft_points[index] = spine_point
                pair_active[index] = False
                failed.append(index + 1)

        pair_metadata = {
            "neck_restoration_pairs_version": 1,
            "coordinate_order": "zyx",
            "spine_points": spine_points.astype(np.uint16).tolist(),
            "shaft_points": shaft_points.astype(np.uint16).tolist(),
            "auto_shaft_points": shaft_points.astype(np.uint16).tolist(),
            "pair_active": pair_active,
            "pair_source": pair_source,
            "pair_edit_state": self._pair_edit_state_from_sources(pair_source),
            "neck_edit_phase": "shaft",
            "allow_preview_point_editing": True,
            "neck_spine_style": {
                "symbol": layer.symbol.value,
                "face_color": layer.face_color.tolist(),
                "edge_color": layer.edge_color.tolist(),
            },
        }
        self._save_points_pair_metadata(layer, pair_metadata)

        self._neck_points_phase_switch = True
        try:
            layer.data = shaft_points
            layer.symbol = "diamond"
            layer.face_color = "orange"
            layer.edge_color = "white"
            layer.selected_data = {0} if len(shaft_points) else set()
            layer.mode = PointsMode.SELECT
            layer.changed = True
        finally:
            self._neck_points_phase_switch = False

        self._set_preview_options()
        if failed:
            show_error(
                "Could not suggest a shaft point for pairs: "
                + ", ".join(str(v) for v in failed)
                + ". These pairs were disabled."
            )

    def _on_neck_edit_points_changed(self, layer: Points, data) -> None:
        if self._neck_points_phase_switch:
            return
        if layer.metadata.get("neck_edit_phase") != "shaft":
            return

        shaft_points = np.asarray(data, dtype=np.int64).reshape((-1, 3))
        previous = np.asarray(
            layer.metadata.get("shaft_points", shaft_points), dtype=np.int64
        ).reshape((-1, 3))
        pair_source = self._normalise_pair_source_list(
            layer.metadata.get("pair_source", None), len(shaft_points), "auto_shaft_snap"
        )
        if previous.shape == shaft_points.shape:
            changed = np.where(np.any(previous != shaft_points, axis=1))[0]
            for index in changed:
                pair_source[int(index)] = "manual"

        self._save_points_pair_metadata(
            layer,
            {
                "shaft_points": shaft_points.astype(np.uint16).tolist(),
                "pair_source": pair_source,
                "pair_edit_state": self._pair_edit_state_from_sources(pair_source),
            },
        )
        if isinstance(self._current_preview_options, QtPointsPreviewOptions):
            self._current_preview_options.update_shaft_points(shaft_points)

    def _highlight_selected_neck_pair(self, pair_index: int) -> None:
        points_layer = self._viewer_model.active_layer

        if not points_layer or not isinstance(points_layer, Points):
            return

        try:
            points_arr = np.asarray(points_layer.data)
        except Exception:
            return

        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            return

        if pair_index < 0 or pair_index >= len(points_arr):
            return

        try:
            points_layer.selected_data = {int(pair_index)}
            points_layer._set_highlight()
        except Exception:
            return

    def _save_points_pair_metadata(self, points_layer, pair_metadata: dict) -> None:
        points_layer.metadata.update(pair_metadata)

        layer_params = self._project_info.layers_parameters.get(points_layer._id, None)
        if layer_params is not None:
            layer_params.metadata.update(pair_metadata)
            layer_params.tmp_metadata.update(pair_metadata)

        points_layer._changed = True
        self._saved = False

    def _normalise_pair_list(
        self,
        value,
        size: int,
        default,
    ) -> list:
        if value is None:
            return [default] * size

        out = list(value)
        if len(out) < size:
            out.extend([default] * (size - len(out)))

        return out[:size]

    def _normalise_pair_bool_list(
        self,
        value,
        size: int,
        default: bool = True,
    ) -> list[bool]:
        if value is None:
            return [default] * size

        out = [bool(v) for v in list(value)]
        if len(out) < size:
            out.extend([default] * (size - len(out)))
        return out[:size]

    def _normalise_pair_source_list(
        self,
        value,
        size: int,
        default: str = "auto_old_medial",
    ) -> list[str]:
        if value is None:
            return [default] * size

        out = [str(v) for v in list(value)]
        if len(out) < size:
            out.extend([default] * (size - len(out)))
        return out[:size]

    def _pair_edit_state_from_sources(self, pair_source: list[str]) -> str:
        return "manual" if any(source == "manual" for source in pair_source) else "auto"

    def _update_manual_shaft_point(
        self,
        pair_index: int,
        shaft_point: list[int],
        snap_to_shaft: bool = False,
    ) -> None:
        points_layer = self._viewer_model.active_layer

        if not points_layer or not isinstance(points_layer, Points):
            return

        metadata, _ = self._get_points_pair_metadata(points_layer)
        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            points_arr = np.asarray(metadata.get("spine_points", []))
        else:
            points_arr = np.asarray(points_layer.data)
        shaft_points = metadata.get("shaft_points", None)

        if shaft_points is None:
            show_error(
                "Cannot edit shaft point: shaft_points metadata is missing. "
                "Please regenerate restoration points."
            )
            return

        try:
            shaft_arr = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))
        except Exception:
            show_error(
                "Cannot edit shaft point: shaft_points metadata has invalid shape. "
                "Please regenerate restoration points."
            )
            return

        if points_arr.shape != shaft_arr.shape:
            show_error(
                "Cannot edit shaft point: points and shaft_points are not synchronized. "
                f"points shape = {points_arr.shape}, shaft_points shape = {shaft_arr.shape}. "
                "Please regenerate restoration points."
            )
            return

        if pair_index < 0 or pair_index >= len(shaft_arr):
            show_error(f"Cannot edit shaft point: invalid pair index {pair_index}.")
            return

        try:
            new_point = np.asarray(shaft_point, dtype=np.int64)
            if new_point.shape != (3,):
                raise ValueError
        except Exception:
            show_error("Cannot edit shaft point: point must have Z/Y/X coordinates.")
            return

        binarization = self._active_layers_list[2]._data
        shape = np.asarray(binarization.shape, dtype=np.int64)

        if np.any(new_point < 0) or np.any(new_point >= shape):
            show_error(
                "Cannot edit shaft point: point is outside image bounds. "
                f"point = {new_point.tolist()}, image shape = {tuple(shape)}."
            )
            return

        if snap_to_shaft:
            try:
                from projects.segmentation.utils.data_processing.paired_necks import (
                    snap_to_valid_shaft_point,
                )

                new_point, _ = snap_to_valid_shaft_point(
                    new_point,
                    binarization,
                    self._project_info.real_scale,
                    max_distance_um=None,
                )
            except Exception as e:
                show_error(f"Cannot snap shaft point to shaft: {e}")
                return

        shaft_arr[pair_index] = new_point.astype(np.uint16)

        auto_shaft_points = metadata.get("auto_shaft_points", None)
        if auto_shaft_points is None:
            auto_shaft_arr = shaft_arr.copy()
        else:
            try:
                auto_shaft_arr = np.asarray(auto_shaft_points, dtype=np.uint16).reshape((-1, 3))
                if auto_shaft_arr.shape != shaft_arr.shape:
                    auto_shaft_arr = shaft_arr.copy()
            except Exception:
                auto_shaft_arr = shaft_arr.copy()

        pair_active = self._normalise_pair_bool_list(
            metadata.get("pair_active", None),
            len(points_arr),
            True,
        )
        pair_source = self._normalise_pair_source_list(
            metadata.get("pair_source", None),
            len(points_arr),
            "auto_old_medial",
        )

        pair_source[pair_index] = "manual"

        pair_metadata = {
            "neck_restoration_pairs_version": 1,
            "coordinate_order": "zyx",
            "spine_points": points_arr.astype(np.uint16).tolist(),
            "shaft_points": shaft_arr.tolist(),
            "auto_shaft_points": auto_shaft_arr.tolist(),
            "pair_active": pair_active,
            "pair_source": pair_source,
            "pair_edit_state": self._pair_edit_state_from_sources(pair_source),
        }

        self._save_points_pair_metadata(points_layer, pair_metadata)

        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            self._neck_points_phase_switch = True
            try:
                points_layer.data = shaft_arr
                points_layer.changed = True
            finally:
                self._neck_points_phase_switch = False

        if snap_to_shaft:
            self._set_preview_options()
        self._highlight_selected_neck_pair(pair_index)

    def _update_neck_pair_active(self, pair_index: int, active: bool) -> None:
        points_layer = self._viewer_model.active_layer

        if not points_layer or not isinstance(points_layer, Points):
            return

        metadata, _ = self._get_points_pair_metadata(points_layer)
        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            points_arr = np.asarray(metadata.get("spine_points", []))
        else:
            points_arr = np.asarray(points_layer.data)

        if pair_index < 0 or pair_index >= len(points_arr):
            show_error(f"Cannot update pair state: invalid pair index {pair_index}.")
            return

        pair_active = self._normalise_pair_bool_list(
            metadata.get("pair_active", None),
            len(points_arr),
            True,
        )
        pair_active[pair_index] = bool(active)

        self._save_points_pair_metadata(
            points_layer,
            {
                "pair_active": pair_active,
            },
        )

    def _reset_manual_shaft_point(self, pair_index: int) -> None:
        points_layer = self._viewer_model.active_layer

        if not points_layer or not isinstance(points_layer, Points):
            return

        metadata, _ = self._get_points_pair_metadata(points_layer)

        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            points_arr = np.asarray(metadata.get("spine_points", []))
        else:
            points_arr = np.asarray(points_layer.data)
        shaft_points = metadata.get("shaft_points", None)
        auto_shaft_points = metadata.get("auto_shaft_points", None)

        if shaft_points is None or auto_shaft_points is None:
            show_error(
                "Cannot reset shaft point: automatic shaft points are missing. "
                "Please regenerate restoration points."
            )
            return

        try:
            shaft_arr = np.asarray(shaft_points, dtype=np.uint16).reshape((-1, 3))
            auto_shaft_arr = np.asarray(auto_shaft_points, dtype=np.uint16).reshape((-1, 3))
        except Exception:
            show_error(
                "Cannot reset shaft point: shaft_points metadata has invalid shape. "
                "Please regenerate restoration points."
            )
            return

        if points_arr.shape != shaft_arr.shape or shaft_arr.shape != auto_shaft_arr.shape:
            show_error(
                "Cannot reset shaft point: points, shaft_points and auto_shaft_points "
                "are not synchronized. Please regenerate restoration points."
            )
            return

        if pair_index < 0 or pair_index >= len(shaft_arr):
            show_error(f"Cannot reset shaft point: invalid pair index {pair_index}.")
            return

        shaft_arr[pair_index] = auto_shaft_arr[pair_index]

        pair_active = self._normalise_pair_bool_list(
            metadata.get("pair_active", None),
            len(points_arr),
            True,
        )
        pair_source = self._normalise_pair_source_list(
            metadata.get("pair_source", None),
            len(points_arr),
            "auto_old_medial",
        )

        pair_source[pair_index] = "auto_old_medial"

        pair_metadata = {
            "neck_restoration_pairs_version": 1,
            "coordinate_order": "zyx",
            "spine_points": points_arr.astype(np.uint16).tolist(),
            "shaft_points": shaft_arr.tolist(),
            "auto_shaft_points": auto_shaft_arr.tolist(),
            "pair_active": pair_active,
            "pair_source": pair_source,
            "pair_edit_state": self._pair_edit_state_from_sources(pair_source),
        }

        self._save_points_pair_metadata(points_layer, pair_metadata)

        if points_layer.metadata.get("neck_edit_phase") == "shaft":
            self._neck_points_phase_switch = True
            try:
                points_layer.data = shaft_arr
                points_layer.changed = True
            finally:
                self._neck_points_phase_switch = False
        self._set_preview_options()
        self._highlight_selected_neck_pair(pair_index)

    def _validate_neck_pair_parameters(self, points_layer, pair_params: dict) -> bool:
        shaft_points = pair_params.get("shaft_points", None)

        if shaft_points is None:
            return True

        points_arr = np.asarray(points_layer.data)
        pair_active = pair_params.get("pair_active", None)
        shaft_arr = np.asarray(shaft_points)

        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            show_error(
                f"Cannot restore necks: points must have shape (N, 3), got {points_arr.shape}."
            )
            return False

        if shaft_arr.ndim != 2 or shaft_arr.shape[1] != 3:
            show_error(
                f"Cannot restore necks: shaft_points must have shape (N, 3), got {shaft_arr.shape}."
            )
            return False

        if points_arr.shape != shaft_arr.shape:
            show_error(
                "Cannot restore necks: points and shaft_points are not synchronized. "
                f"points shape = {points_arr.shape}, shaft_points shape = {shaft_arr.shape}. "
                "Please regenerate restoration points."
            )
            return False

        if pair_active is not None and len(pair_active) != len(points_arr):
            show_error(
                "Cannot restore necks: pair_active length does not match points length. "
                f"pair_active length = {len(pair_active)}, points length = {len(points_arr)}."
            )
            return False

        return True

    def _create_new_necks(self) -> None:
        points_layer = self._active_layers_list[3]
        pair_params = self._get_neck_pair_parameters()

        if not self._validate_neck_pair_parameters(points_layer, pair_params):
            return

        metadata = {
            "name": Translater.instance().get_translation("necks"),
            "parent_id": points_layer._id,
            "coordinate_order": "zyx",
        }

        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.NECKS,
                necks_reconnection,
                {
                    "image": self._active_layers_list[0].data,
                    "binarization": self._active_layers_list[2]._data,
                    "points": points_layer.data,
                    "scale": self._project_info.real_scale,
                    "parameters": pair_params,
                    "folder": self._project_info.folder,
                    "metadata": metadata,
                    "additional_files": {},
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "necks in queue",
                points_layer._id,
            )
        )

    def _update_necks(self) -> None:
        self._stop_updating()
        params = self._current_preview_options.get_params()
        params.update(self._get_neck_pair_parameters())

        if not self._validate_neck_pair_parameters(self._active_layers_list[3], params):
            return

        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.NECKS,
                necks_reconnection,
                {
                    "image": self._active_layers_list[0].data,
                    "binarization": self._active_layers_list[2]._data,
                    "points": self._active_layers_list[3].data,
                    "scale": self._project_info.real_scale,
                    "parameters": params,
                    "folder": self._project_info.folder,
                    "metadata": {},
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "necks update in queue",
                self._active_layers_list[3]._id,
                self._viewer_model.active_layer._id,
            )
        )

    def _create_new_mesh(self) -> None:
        mesh_options = self._mesh_build_options.get_params()
        metadata = {
            "name": Translater.instance().get_translation("polygon mesh"),
            "parent_id": self._active_layers_list[4]._id,
            **mesh_options,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POLYGON_MESH,
                build_mesh,
                {
                    "data": self._active_layers_list[4]._data,
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "folder": self._project_info.folder,
                    "metadata": metadata,
                    **mesh_options,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "polygon_mesh in queue",
                self._active_layers_list[4]._id,
            )
        )

    def _create_new_mesh_segmentation(self) -> None:
        metadata = {
            "name": Translater.instance().get_translation("polygon mesh segmentation"),
            "parent_id": self._active_layers_list[self._layer_offset_index]._id,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POLYGON_MESH_SEGMENTATION,
                segment_spines,
                {
                    "mesh_file": self._project_info.layers_parameters[
                        self._active_layers_list[self._layer_offset_index]._id
                    ].tmp_mesh_file,
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "min_coord": self._project_info.min_coordinates,
                    "parameters": {},
                    "metadata": metadata,
                    "additional_files": {},
                    "folder": self._project_info.folder,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "polygon_mesh_segmentation in queue",
                self._active_layers_list[self._layer_offset_index]._id,
            )
        )

    def _update_mesh_segmentation(self) -> None:
        self._stop_updating()
        params = self._current_preview_options.get_params()
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POLYGON_MESH_SEGMENTATION,
                segment_spines,
                {
                    "mesh_file": self._project_info.layers_parameters[
                        self._active_layers_list[self._layer_offset_index]._id
                    ].tmp_mesh_file,
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "min_coord": self._project_info.min_coordinates,
                    "parameters": params,
                    "metadata": {},
                    "folder": self._project_info.folder,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "polygon_mesh_segmentation update in queue",
                self._active_layers_list[self._layer_offset_index]._id,
                self._viewer_model.active_layer._id,
                self._project_info.layers_parameters[
                    self._viewer_model.active_layer._id
                ].tmp_preview_partially_fixed,
            )
        )

    def _update_spine(self) -> None:
        self._stop_updating()
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_metadata["spine_blocked"] = str(datetime.now())
        layer = self._viewer_model.active_layer
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.POLYGON_MESH_SEGMENTATION,
                correct_spine,
                {
                    "mesh_file": self._project_info.layers_parameters[
                        self._active_layers_list[self._layer_offset_index]._id
                    ].tmp_mesh_file,
                    "spines_files": self._project_info.layers_parameters[
                        layer._id
                    ].tmp_spines_files,
                    "adjusted_spines_files": self._project_info.layers_parameters[
                        layer._id
                    ].tmp_adjusted_spines_files,
                    "spine_id": self._current_preview_options.selection_spin_box.value(),
                    "deleted_spines": self._project_info.layers_parameters[
                        layer._id
                    ].tmp_deleted_spines,
                    "correction": self._current_preview_options.spine_correction_slider.value(),
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "min_coord": self._project_info.min_coordinates,
                    "metadata": {},
                    "folder": self._project_info.folder,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "polygon_mesh_segmentation update in queue",
                self._active_layers_list[self._layer_offset_index]._id,
                self._viewer_model.active_layer._id,
                True,
            )
        )

    def _on_spine_deleted(self) -> None:
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_deleted_spines = self._current_preview_options._deleted_spines

    def _on_spine_restored(self) -> None:
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_deleted_spines = self._current_preview_options._deleted_spines

    def _on_camera_policy_changed(self) -> None:
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_metadata[
            "move_camera"
        ] = self._current_preview_options.move_camera_check_box.isChecked()

    def _on_current_spine_changed(self) -> None:
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_metadata[
            "current_spine"
        ] = self._current_preview_options.selection_spin_box.value()

    def _fix_partially_layer_parameters(self) -> None:
        self._project_info.layers_parameters[
            self._viewer_model.active_layer._id
        ].tmp_preview_partially_fixed = True

    def _create_new_voxel_segmentation(self) -> None:
        if self._is_binary_voxel_correction_mode():
            if len(self._active_layers_list) < 2:
                return
            metadata = {
                "name": Translater.instance().get_translation("voxel mesh segmentation"),
                "parent_id": self._active_layers_list[1]._id,
            }
            self._data_processing_task_queue.append(
                DataProcessingTask(
                    Types.VOXEL_MESH_SEGMENTATION,
                    build_voxel_from_binary,
                    {
                        "data": self._active_layers_list[0].data,
                        "area": self._active_layers_list[1].data,
                        "metadata": metadata,
                        "queue_in": self._data_processing_communication_queue_out,
                        "queue_out": self._data_processing_communication_queue_in,
                    },
                    "voxel_mesh_segmentation in queue",
                    self._active_layers_list[1]._id,
                )
            )
            return

        metadata = {
            "name": Translater.instance().get_translation("voxel mesh segmentation"),
            "parent_id": self._active_layers_list[1 + self._layer_offset_index]._id,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.VOXEL_MESH_SEGMENTATION,
                voxelize,
                {
                    "binary_filename": self._project_info.layers_parameters[
                        self._active_layers_list[self._layer_offset_index]._id
                    ].tmp_mesh_source_tif_file,
                    "spines_files": self._project_info.layers_parameters[
                        self._active_layers_list[1 + self._layer_offset_index]._id
                    ].tmp_adjusted_spines_files,
                    "folder": self._project_info.folder,
                    "shape": self._project_info.shape,
                    "scale": self._project_info.real_scale,
                    "min_coord": self._project_info.min_coordinates,
                    "metadata": metadata,
                    "queue_in": self._data_processing_communication_queue_out,
                    "queue_out": self._data_processing_communication_queue_in,
                },
                "voxel_mesh_segmentation in queue",
                self._active_layers_list[1 + self._layer_offset_index]._id,
            )
        )

    def _create_new_final_segmentation(self) -> None:
        mesh_options = self._mesh_build_options.get_params()
        metadata = {
            "name": Translater.instance().get_translation("final segmentation"),
            "parent_id": self._active_layers_list[2 + self._layer_offset_index]._id,
            **mesh_options,
        }
        self._data_processing_task_queue.append(
            DataProcessingTask(
                Types.FINAL_SEGMENTATION,
                build_final_segmentation,
                {
                    "data": self._active_layers_list[
                        2 + self._layer_offset_index
                    ]._data,
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
                self._active_layers_list[2 + self._layer_offset_index]._id,
            )
        )

    def _previous_stage(self) -> None:
        if len(self._active_layers_list) < 2:
            return

        ids = set()
        ids.add(self._active_layers_list[-1]._id)
        if self._active_layers_list[-1]._changed:
            self._layer_loader.save_layer_data(
                self._active_layers_list[-1]._id, self._active_layers_list[-1].data
            )
            self._saved = False
        for layer in self._next_stage_list:
            ids.add(layer._id)
            if layer._changed:
                self._layer_loader.save_layer_data(layer._id, layer.data)
                self._saved = False

        for layer in self._additional_layers_list:
            if layer._id in ids:
                self._layer_loader.parameters_tracker.unlink_layers(layer)

        self._next_stage_list.clear()
        del self._active_layers_list[-1]
        self._set_create_new_stage_button(self._active_layers_list[-1].metadata["type"])
        enable = len(self._active_layers_list) > 1
        self._control_buttons.buttons.previousStageButton.setEnabled(enable)
        self._control_buttons.buttons.imageStageButton.setEnabled(enable)

        if (
            self._active_layer_loader_task
            and TaskType.LOAD == self._active_layer_loader_task.type
        ):
            self._layer_loader_communication_queue_out.put("stop")
        for task in self._layer_loader_task_queue:
            if task.type == TaskType.LOAD:
                task.pbar.close()
                del task
                break

        infos = [
            (child_id, LayerType.NEXT_STAGE)
            for child_id in self._project_info.data.layers[
                self._active_layers_list[-1]._id
            ].child_layers
        ]
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.LOAD, progress(total=0, desc="load in queue"), {"infos": infos}
            )
        )
        self._next_layer_loader_task()

    def _image_stage(self) -> None:
        if len(self._active_layers_list) < 2:
            return

        ids = set()
        for layer in self._active_layers_list[1:]:
            ids.add(layer._id)
            if layer._changed:
                self._layer_loader.save_layer_data(layer._id, layer.data)
                self._saved = False
        for layer in self._next_stage_list:
            ids.add(layer._id)
            if layer._changed:
                self._layer_loader.save_layer_data(layer._id, layer.data)
                self._saved = False

        for layer in self._additional_layers_list:
            if layer._id in ids:
                self._layer_loader.parameters_tracker.unlink_layers(layer)

        self._next_stage_list.clear()
        del self._active_layers_list[1:]
        self._set_create_new_stage_button(Types.IMAGE)
        self._control_buttons.buttons.previousStageButton.setEnabled(False)
        self._control_buttons.buttons.imageStageButton.setEnabled(False)

        if (
            self._active_layer_loader_task
            and TaskType.LOAD == self._active_layer_loader_task.type
        ):
            self._layer_loader_communication_queue_out.put("stop")
        for task in self._layer_loader_task_queue:
            if task.type == TaskType.LOAD:
                task.pbar.close()
                del task
                break

        infos = [
            (child_id, LayerType.NEXT_STAGE)
            for child_id in self._project_info.data.layers[
                self._active_layers_list[0]._id
            ].child_layers
        ]
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.LOAD, progress(total=0, desc="load in queue"), {"infos": infos}
            )
        )
        self._next_layer_loader_task()

    def _add_background_image(self) -> None:
        dlg = QFileDialog(self._window._qt_window)
        files = get_open_history_files()
        hist = []
        for file in files:
            hist.append(os.path.dirname(file))
        dlg.setHistory(hist)
        path, _ = dlg.getOpenFileName(
            caption=Translater.instance().get_translation("select image"),
            directory=hist[0],
            filter="Tiff (*.tiff *.tif)",
        )
        if not path or path == "":
            return

        update_open_history_files(path)

        data = imread(path)
        if data.shape != self._project_info.original_shape:
            show_error("Invalid background image shape")
            return

        max_value = np.max(data)
        min_value = np.min(data)
        if min_value != 0 or max_value != 255:
            data -= min_value
            alpha = 255 / (max_value - min_value)
            data = data.astype("float") * alpha
            data = np.round(data).astype("uint8")
        if self._project_info.layers_parameters[0].tmp_metadata.get("x_range", None):
            x_range = self._project_info.layers_parameters[0].tmp_metadata["x_range"]
            y_range = self._project_info.layers_parameters[0].tmp_metadata["y_range"]
            z_range = self._project_info.layers_parameters[0].tmp_metadata["z_range"]
            data = data[
                z_range[0] : z_range[1] + 1,
                y_range[0] : y_range[1] + 1,
                x_range[0] : x_range[1] + 1,
            ]

        self._saved = False
        params = {
            "layer_type": Types.BACKGROUND_IMAGE,
            "name": Translater.instance().get_translation("background_image"),
            "data": data,
            "parent_id": None,
        }
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.CREATE,
                progress(total=0, desc="create layer in queue"),
                params,
            )
        )
        self._next_layer_loader_task()

    def _add_layer_to_non_editable(self, layer=None) -> None:
        if not layer:
            layer = (
                self._qt_active_layer_list.list.activeItem
                or self._qt_next_stage_list.next_stage_list.activeItem
            )
        if not layer or self._project_info.layers_parameters[layer._id].tmp_preview:
            return

        info = NonEditableLayerInfo()
        info.layer_id = layer._id
        info.parameters = self._layer_loader.parameters_tracker.get_parameters(layer)
        additional_layer = self._layer_loader.create_layer(info, None)
        if additional_layer.metadata["type"] == Types.POINTS:
            additional_layer.highlight_thickness = self._settings.state[
                "highlight_thickness"
            ]
        self._layer_loader.parameters_tracker.link_layers(additional_layer, layer)

        self._viewer_model.add_layer(additional_layer, LayerType.ADDITIONAL)

    def _selected_stage(self) -> None:
        layer = self._qt_active_layer_list.list.activeItem
        if layer:
            if layer == self._active_layers_list[-1]:
                return

            index = self._active_layers_list.index(layer)
            ids = set()
            for _layer in self._active_layers_list[index + 1 :]:
                ids.add(_layer._id)
                if _layer._changed:
                    self._layer_loader.save_layer_data(_layer._id, _layer.data)
                    self._saved = False
            for _layer in self._next_stage_list:
                ids.add(_layer._id)
                if _layer._changed:
                    self._layer_loader.save_layer_data(_layer._id, _layer.data)
                    self._saved = False

            for _layer in self._additional_layers_list:
                if _layer._id in ids:
                    self._layer_loader.parameters_tracker.unlink_layers(_layer)

            self._next_stage_list.clear()
            del self._active_layers_list[index + 1 :]
            self._set_create_new_stage_button(layer.metadata["type"])
        elif len(self._active_layers_list) < 3 + self._layer_offset_index:
            layer = self._qt_next_stage_list.next_stage_list.activeItem
            if layer:
                ids = set()
                for _layer in self._next_stage_list:
                    ids.add(_layer._id)
                    if _layer._changed:
                        self._layer_loader.save_layer_data(_layer._id, _layer.data)
                        self._saved = False
                ids.remove(layer._id)

                for _layer in self._additional_layers_list:
                    if _layer._id in ids:
                        self._layer_loader.parameters_tracker.unlink_layers(_layer)

                self._next_stage_list.clear()
                self._active_layers_list.append(layer)
                self._set_create_new_stage_button(layer.metadata["type"])
            else:
                return

        enable = len(self._active_layers_list) > 1
        self._control_buttons.buttons.previousStageButton.setEnabled(enable)
        self._control_buttons.buttons.imageStageButton.setEnabled(enable)

        if (
            self._active_layer_loader_task
            and TaskType.LOAD == self._active_layer_loader_task.type
        ):
            self._layer_loader_communication_queue_out.put("stop")
        for task in self._layer_loader_task_queue:
            if task.type == TaskType.LOAD:
                task.pbar.close()
                del task
                break

        infos = [
            (child_id, LayerType.NEXT_STAGE)
            for child_id in self._project_info.data.layers[
                self._active_layers_list[-1]._id
            ].child_layers
        ]
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.LOAD, progress(total=0, desc="load in queue"), {"infos": infos}
            )
        )
        self._next_layer_loader_task()

    def _export_layer(self) -> None:
        layer = self._viewer_model.active_layer
        if not layer:
            return

        dlg = QFileDialog(self._window._qt_window)
        folders = get_open_history_folders()
        dlg.setHistory(folders)
        folder = dlg.getExistingDirectory(
            caption=Translater.instance().get_translation("select folder"),
            directory=folders[0],
        )
        if not folder or folder == "":
            return
        update_open_history_folders(folder)

        self._exported_layers.add(layer._id)
        if (
            layer.metadata["type"] == Types.POLYGON_MESH
            or layer.metadata["type"] == Types.POLYGON_MESH_SEGMENTATION
            or layer.metadata["type"] == Types.FINAL_SEGMENTATION
        ):
            data = None
        else:
            data = layer.data.copy()
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.EXPORT,
                progress(total=0, desc="export in queue"),
                {"layer_id": layer._id, "data": data, "folder": folder},
            )
        )
        self._next_layer_loader_task()

    def _delete_layer(self) -> None:
        if (
            self._active_layer_loader_task
            and TaskType.DELETE == self._active_layer_loader_task.type
            or self._task_in_queue(TaskType.DELETE)
        ):
            return

        additional = True
        layer = self._qt_additional_layer_list.list.activeItem
        if layer and layer.metadata["type"] == Types.BACKGROUND_IMAGE:
            additional = False
        if not layer:
            layer = self._qt_next_stage_list.next_stage_list.activeItem
            additional = False
        if not layer:
            return

        self._saved = False
        if additional:
            info = NonEditableLayerInfo()
            info.layer_id = layer._id
            info.parameters = self._layer_loader.parameters_tracker.get_parameters(
                layer
            )
            self._layer_loader.delete_layer(info)
            self._layer_loader.parameters_tracker.unlink_layers(layer)
            self._viewer_model.remove_layer(layer, LayerType.ADDITIONAL)
            del layer
            self._control_buttons.buttons.restoreButton.setEnabled(True)
        else:
            self._stop_updating()
            self._control_buttons.buttons.deleteButton.setEnabled(False)
            if layer._changed:
                self._layer_loader.save_layer_data(layer._id, layer.data)
            self._layer_loader._state.block_history = True
            if layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                self._viewer_model.remove_layer(layer, LayerType.ADDITIONAL)
            else:
                self._viewer_model.remove_layer(layer, LayerType.NEXT_STAGE)
            self._layer_loader_task_queue.append(
                LayerLoaderTask(
                    TaskType.DELETE,
                    progress(total=0, desc="delete in queue"),
                    {"info": layer._id},
                )
            )
            del layer
            self._next_layer_loader_task()

    def _restore_last(self) -> None:
        if (
            len(self._layer_loader._state.history) == 0
            or self._active_layer_loader_task
            and TaskType.RESTORE == self._active_layer_loader_task.type
            or self._task_in_queue(TaskType.RESTORE)
        ):
            return
        self._saved = False
        self._control_buttons.buttons.restoreButton.setEnabled(False)
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.RESTORE,
                progress(total=0, desc="restore in queue"),
            )
        )
        self._next_layer_loader_task()

    def _create(self) -> None:
        self._active_layer_loader_task.pbar.set_description("creation in progress")

        info = LayerInfo()
        info.type = self._active_layer_loader_task.params["layer_type"]
        info.parent_id = self._active_layer_loader_task.params["parent_id"]
        parameters = LayerParameters()
        parameters.name = self._active_layer_loader_task.params["name"]
        if (
            info.type == Types.BINARIZATION
            or info.type == Types.POINTS
            or info.type == Types.NECKS
            or info.type == Types.POLYGON_MESH_SEGMENTATION
        ):
            parameters.preview = True
            parameters.tmp_preview = True
        if info.type == Types.NECKS:
            parameters.parameters.update(
                {"labels": ["background", "general_label", "restored_necks"]}
            )
        elif info.type == Types.VOXEL_MESH_SEGMENTATION:
            parameters.parameters.update(
                {"labels": ["background", "general_label", "spines"]}
            )
        if (
            "metadata" in self._active_layer_loader_task.params
            and self._active_layer_loader_task.params["metadata"]
        ):
            parameters.metadata.update(
                self._active_layer_loader_task.params["metadata"]
            )
            parameters.tmp_metadata.update(
                self._active_layer_loader_task.params["metadata"]
            )
        if "additional_files" in self._active_layer_loader_task.params:
            if info.type == Types.POINTS:
                self._layer_loader.replace_additional_files(
                    info.parent_id,
                    self._active_layer_loader_task.params["additional_files"],
                )
            elif info.type == Types.NECKS:
                self._layer_loader.replace_additional_files(
                    self._project_info.data.layers[info.parent_id].parent_id,
                    self._active_layer_loader_task.params["additional_files"],
                )
            else:
                parameters.tmp_additional_files.update(
                    self._active_layer_loader_task.params["additional_files"]
                )
        data = self._active_layer_loader_task.params["data"]
        if isinstance(data, SurfaceData):
            parameters.mesh_file = ""
            parameters.tmp_mesh_file = data.mesh_file
            parameters.mesh_source_tif_file = ""
            parameters.tmp_mesh_source_tif_file = data.tif_file
        elif isinstance(data, SegmentationData):
            parameters.spines_files = data.spines_files.copy()
            parameters.tmp_spines_files = data.spines_files.copy()
            parameters.adjusted_spines_files = data.adjusted_spines_files.copy()
            parameters.tmp_adjusted_spines_files = data.adjusted_spines_files.copy()
        elif isinstance(data, FinalSegmentationData):
            parameters.mesh_file = ""
            parameters.tmp_mesh_file = data.mesh_file
            parameters.spines_files = data.spines_files.copy()
            parameters.tmp_spines_files = data.spines_files.copy()
            parameters.adjusted_spines_files = data.spines_files.copy()
            parameters.tmp_adjusted_spines_files = data.spines_files.copy()
        layer = self._layer_loader.create_layer(info, parameters, data)
        layer.preview = self._project_info.layers_parameters[layer._id].tmp_preview
        self._layer_loader.save_layer_data(layer._id, layer.data)
        layer.changed_.connect(partial(self._on_layer_data_changed, layer))
        if info.parent_id == self._active_layers_list[-1]._id:
            if info.type == Types.POINTS:
                layer.highlight_thickness = self._settings.state["highlight_thickness"]
            self._viewer_model.add_layer(layer, LayerType.NEXT_STAGE)
            self._layer_loader.parameters_tracker.track_parameters(layer)
        elif info.type == Types.BACKGROUND_IMAGE:
            self._viewer_model.add_layer(layer, LayerType.ADDITIONAL)
            self._layer_loader.parameters_tracker.track_parameters(layer)
        self._saved = False

        self._active_layer_loader_task.pbar.close()
        self._active_layer_loader_task = None
        if len(self._layer_loader_task_queue) > 0:
            self._next_layer_loader_task()

    def _on_layer_data_changed(self, layer: Labels, changed) -> None:
        if changed:
            self._project_info.layers_parameters[layer._id].last_change_time = str(
                datetime.now()
            )
            self._project_info.layers_parameters[layer._id].tmp_preview_need_update = (
                True
            )

    def _delete(self) -> None:
        self._active_layer_loader_task.pbar.set_description("delete in progress")
        self._active_layer_loader_task.process = Process(
            target=delete_layer,
            args=(
                self._layer_loader,
                self._active_layer_loader_task.params["info"],
                self._layer_loader_communication_queue_in,
            ),
        )
        self._active_layer_loader_task.process.start()

    def _on_deleted(self, ids) -> None:
        layers = self._additional_layers_list.copy()
        for layer in layers:
            if layer._id in ids:
                self._viewer_model.remove_layer(layer, LayerType.ADDITIONAL)
                info = NonEditableLayerInfo()
                info.layer_id = layer._id
                info.parameters = self._layer_loader.parameters_tracker.get_parameters(
                    layer
                )
                self._layer_loader.delete_layer(info)
                self._layer_loader.parameters_tracker.unlink_layers(layer)
                del layer
        del layers

        self._layer_loader._state.block_history = False
        if self._layer_loader._state.staged_history:
            self._layer_loader._state.history.append(
                self._layer_loader._state.staged_history
            )
            self._layer_loader._state.staged_history = []
        self._control_buttons.buttons.deleteButton.setEnabled(
            bool(
                self._qt_next_stage_list.next_stage_list.activeItem
                or self._qt_additional_layer_list.list.activeItem
            )
        )
        self._control_buttons.buttons.restoreButton.setEnabled(True)
        deleted_tasks = set()
        if (
            self._active_data_processing_task
            and self._active_data_processing_task.layer_id
            and self._active_data_processing_task.layer_id in ids
        ):
            self._data_processing_communication_queue_out.put("stop")
        for i, task in enumerate(self._data_processing_task_queue):
            if task.layer_id and task.layer_id in ids:
                deleted_tasks.add(i)
        for i in sorted(deleted_tasks, reverse=True):
            del self._data_processing_task_queue[i]

    def _restore(self) -> None:
        self._active_layer_loader_task.pbar.set_description("restore in progress")
        self._active_layer_loader_task.process = Process(
            target=restore_layer,
            args=(
                self._layer_loader,
                self._layer_loader_communication_queue_in,
            ),
        )
        self._active_layer_loader_task.process.start()

    def _on_restored(self, layers_info, non_editable_layers_info) -> None:
        if len(layers_info) > 0:
            layer = self._layer_loader.create_layer(
                layers_info[0][0], layers_info[0][1], layers_info[0][2]
            )
            if layer.metadata["parent_id"] == self._active_layers_list[-1]._id:
                if layer.metadata["type"] == Types.POINTS:
                    layer.highlight_thickness = self._settings.state[
                        "highlight_thickness"
                    ]
                self._viewer_model.add_layer(layer, LayerType.NEXT_STAGE)
                self._layer_loader.parameters_tracker.track_parameters(layer)
            elif layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                self._viewer_model.add_layer(layer, LayerType.ADDITIONAL)
                self._layer_loader.parameters_tracker.track_parameters(layer)
        for layer_info in non_editable_layers_info:
            layer = self._layer_loader.create_layer(
                layer_info[0], layer_info[1], layer_info[2]
            )
            linked = False
            for active_layer in self._active_layers_list:
                if active_layer._id == layer._id:
                    self._layer_loader.parameters_tracker.link_layers(
                        layer, active_layer
                    )
                    linked = True
                    break
            if not linked:
                for next_stage_layer in self._next_stage_list:
                    if next_stage_layer._id == layer._id:
                        self._layer_loader.parameters_tracker.link_layers(
                            layer, next_stage_layer
                        )
                        break
            if layer.metadata["type"] == Types.POINTS:
                layer.highlight_thickness = self._settings.state["highlight_thickness"]
            self._viewer_model.add_layer(layer, 1)
        self._control_buttons.buttons.restoreButton.setEnabled(
            len(self._layer_loader._state.history) > 0
        )

    def _load(self) -> None:
        self._active_layer_loader_task.pbar.set_description("load in progress")
        if self._closing:
            self._on_loading_finished(False)
            return

        self._active_layer_loader_task.process = Process(
            target=load_layers,
            args=(
                self._layer_loader,
                self._active_layer_loader_task.params.get("infos", []),
                self._layer_loader_communication_queue_out,
                self._layer_loader_communication_queue_in,
            ),
        )
        self._active_layer_loader_task.process.start()

    def _on_layer_loaded(self, layer_info, metadata) -> None:
        if self._closing:
            return

        layer = self._layer_loader.create_layer(
            layer_info[0], layer_info[1], layer_info[2]
        )
        if (
            metadata == LayerType.NEXT_STAGE
            and self._active_layers_list[-1]._id != layer.metadata["parent_id"]
        ):
            return

        layer.preview = self._project_info.layers_parameters[layer._id].tmp_preview
        if layer.metadata["type"] == Types.POINTS:
            layer.highlight_thickness = self._settings.state["highlight_thickness"]
        self._viewer_model.add_layer(layer, metadata)
        if (
            metadata == LayerType.ADDITIONAL
            and layer.metadata["type"] != Types.BACKGROUND_IMAGE
        ):
            return
        self._layer_loader.parameters_tracker.track_parameters(layer)
        layer.changed_.connect(partial(self._on_layer_data_changed, layer))

    def _on_loading_finished(self, loaded) -> None:
        if loaded:
            for layer in self._additional_layers_list:
                if layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                    continue
                linked = False
                for active_layer in self._active_layers_list:
                    if active_layer._id == layer._id:
                        self._layer_loader.parameters_tracker.link_layers(
                            layer, active_layer
                        )
                        linked = True
                        break
                if not linked:
                    for next_stage_layer in self._next_stage_list:
                        if next_stage_layer._id == layer._id:
                            self._layer_loader.parameters_tracker.link_layers(
                                layer, next_stage_layer
                            )
                            break

    def _export(self) -> None:
        self._active_layer_loader_task.pbar.set_description("export in progress")
        self._active_layer_loader_task.process = Process(
            target=export_layer,
            args=(
                self._layer_loader,
                self._active_layer_loader_task.params["layer_id"],
                self._active_layer_loader_task.params["data"],
                self._active_layer_loader_task.params["folder"],
                self._layer_loader_communication_queue_in,
            ),
        )
        self._active_layer_loader_task.process.start()

    def _on_exported(self) -> None:
        self._exported_layers.remove(self._active_layer_loader_task.params["layer_id"])

    def _save(self) -> None:
        self._active_layer_loader_task.pbar.set_description("save in progress")
        self._active_layer_loader_task.process = Process(
            target=fix_project_state,
            args=(
                self._layer_loader,
                self._active_layer_loader_task.params["layers_data"],
                self._active_layer_loader_task.params["save"],
                self._layer_loader_communication_queue_in,
            ),
        )
        self._active_layer_loader_task.process.start()

    def _on_saved(self, project_info: ProjectInfo) -> None:
        self._project_info = project_info
        self._layer_loader._project_info = self._project_info
        self._layer_loader.parameters_tracker._project_info = self._project_info
        if self._closing and len(self._layer_loader_task_queue) == 0:
            rmtree(self._project_info.folder + ADDITIONAL_COLORMAPS_PATH, True)
            background_images = []
            for layer in self._additional_layers_list:
                if layer.metadata["type"] == Types.BACKGROUND_IMAGE:
                    if layer._id in self._project_info.layers_parameters:
                        background_images.append(layer._id)
            self._project_info.background_images = background_images
            if self.settings.state["save_active_stage"]:
                active = []
                for layer in self._active_layers_list:
                    if layer._id in self._project_info.layers_parameters:
                        active.append(layer._id)
                self._project_info.active_layers = active
            if self.settings.state["save_non_editable"]:
                non_editable = []
                for layer in self._additional_layers_list:
                    if layer._id in self._project_info.layers_parameters:
                        info = NonEditableLayerInfo()
                        info.layer_id = layer._id
                        info.parameters = (
                            self._layer_loader.parameters_tracker.get_parameters(layer)
                        )
                        non_editable.append(info)
                self._project_info.non_editable_layers = non_editable
            old_folder: str = self._project_info.folder
            new_folder = (
                "/".join(old_folder.split("/")[:-1]) + "/" + self._project_info.name
            )
            filename = new_folder + "/" + ProjectInfo.DESCRIPTION_FILENAME
            self._project_info.folder = new_folder
            if os.path.normcase(os.path.abspath(old_folder)) != os.path.normcase(
                os.path.abspath(new_folder)
            ):
                os.replace(old_folder, new_folder)
            update_open_history(filename)
        self._project_info.save()
        self._saving = False
        QApplication.instance().removeEventFilter(self._input_filter)
        if self._toggle_activity_dialog:
            self._window.on_toggle_activity_dock()
        self._saved = True
        if len(self._layer_loader_task_queue) == 0 and self._closing:
            self._active_layer_loader_task.process.join()
            self._active_layer_loader_task.pbar.close()
            self._active_layer_loader_task = None
            self._close()

    def _close(self) -> None:
        if (
            self._active_data_processing_task is None
            and self._active_layer_loader_task is None
            and len(self._layer_loader_task_queue) == 0
            and len(self._data_processing_task_queue) == 0
        ):
            self._layer_loader_tasks_timer.stop()
            self._data_processing_tasks_timer.stop()
            self._layer_loader_communication_queue_in.close()
            self._layer_loader_communication_queue_in.join_thread()
            self._data_processing_communication_queue_in.close()
            self._data_processing_communication_queue_in.join_thread()
            rmtree(self._project_info.folder + TMP)
            self.closed.emit()
            self.deleteLater()

    def _set_shortcuts(self):
        general_shortcuts = [
            Shortcut(
                constants.NDISPLAY_SHORTCUT_TEXT,
                self._ndisplay_change,
                description="Toggle 2D/3D view",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.ROLL_SHORTCUT_TEXT,
                self._viewer_model.dims._roll,
                description="Change order of the visible axes, e.g.\u00a0[0,\u00a01,\u00a02]\u00a0\u2011>\u00a0[2,\u00a00,\u00a01]",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.TRANSPOSE_SHORTCUT_TEXT,
                self._viewer_model.dims.transpose,
                description="Transpose order of the last two visible axes, e.g.\u00a0[0,\u00a01]\u00a0\u2011>\u00a0[1,\u00a00]",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.RESET_VIEW_SHORTCUT_TEXT,
                self._viewer_model.reset_view,
                description="Reset view to original state",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.RESTORE_SHORTCUT_TEXT,
                self._restore_last,
                description="Undo the last layer deletion",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.DELETE_SHORTCUT_TEXT,
                self._delete_layer,
                description="Delete selected layer",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.BACKSPACE_SHORTCUT_TEXT,
                self._delete_layer,
                description="Delete selected layer",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.IMAGE_STAGE_SHORTCUT_TEXT,
                self._image_stage,
                description="Go to the image stage",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.ADDITIONAL_LAYER_SHORTCUT_TEXT,
                self._add_layer_to_non_editable,
                description="Add a copy of the selected layer to the non-editable layers",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.SELECTED_STAGE_SHORTCUT_TEXT,
                self._selected_stage,
                description="Go to the selected stage",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.PREVIOUS_STAGE_SHORTCUT_TEXT,
                self._previous_stage,
                description="Return to the previous stage",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.RESET_SCROLL_SHORTCUT_TEXT,
                self._reset_scroll_progress,
                description="Reset scroll",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.INCREMENT_SCROLL_SHORTCUT_TEXT,
                self._viewer_model.dims._increment_dims_right,
                description="Increment dimensions slider to the right",
            ),
            Shortcut(
                constants.DECREMENT_SCROLL_SHORTCUT_TEXT,
                self._viewer_model.dims._increment_dims_left,
                description="Increment dimensions slider to the left",
            ),
            Shortcut(
                constants.TOGGLE_VISIBILITY_SHORTCUT_TEXT,
                self._toggle_layer_visibility,
                description="Toggle visibility of the selected layer",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.PAN_ZOOM_TMP_SHORTCUT_TEXT,
                self._change_mode_to_pan_zoom,
                self._return_previous_mode,
                description="Press and hold for pan/zoom mode",
                ignore_auto_repeat=True,
            ),
        ]
        self.register_shortcuts("general", general_shortcuts)

        image_shortcuts = [
            Shortcut(
                constants.MAX_INTENSITY_SHORTCUT_TEXT,
                self._toggle_image_max_intensity,
                description="Toggle image max intensity view",
                ignore_auto_repeat=True,
            )
        ]
        self.register_shortcuts("image", image_shortcuts)

        labels_shortcuts = [
            Shortcut(
                constants.UNDO_SHORTCUT_TEXT,
                self._undo,
                description="Undo",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.REDO_SHORTCUT_TEXT,
                self._redo,
                description="Redo",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.SHUFFLE_COLORS_SHORTCUT_TEXT,
                self._shuffle_colors,
                description="Shuffle label colors",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.PAN_ZOOM_SHORTCUT_TEXT,
                self._pan_zoom_mode,
                description="Pan/zoom mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.BRUSH_SHORTCUT_TEXT,
                self._paint_mode,
                description="Paint mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.FILL_SHORTCUT_TEXT,
                self._fill_mode,
                description="Fill mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.ERASE_SHORTCUT_TEXT,
                self._erase_mode,
                description="Erase mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.NEXT_LABEL_SHORTCUT_TEXT,
                self._next_label,
                description="Next label",
            ),
            Shortcut(
                constants.PREV_LABEL_SHORTCUT_TEXT,
                self._prev_label,
                description="Previous label",
            ),
            Shortcut(
                constants.FILL_3D_SHORTCUT_TEXT,
                self._toggle_fill_3d,
                description="Toggle fill 3d",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.PRESERVE_BACKGROUND_SHORTCUT_TEXT,
                self._toggle_preserve_background,
                description="Toggle preserve background",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.SHOW_SELECTED_SHORTCUT_TEXT,
                self._toggle_show_selected,
                description="Toggle show selected label",
                ignore_auto_repeat=True,
            ),
        ]
        self.register_shortcuts("labels", labels_shortcuts)

        points_shortcuts = [
            Shortcut(
                constants.UNDO_SHORTCUT_TEXT,
                self._undo,
                description="Undo",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.REDO_SHORTCUT_TEXT,
                self._redo,
                description="Redo",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.PAN_ZOOM_SHORTCUT_TEXT,
                self._pan_zoom_mode,
                description="Pan/zoom mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.ADD_SHORTCUT_TEXT,
                self._add_mode,
                description="Add mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.SELECT_SHORTCUT_TEXT,
                self._select_mode,
                description="Selection mode",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.SELECT_ALL_SHORTCUT_TEXT,
                self._select_all,
                description="Select all visible points",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.DELETE_POINT_SHORTCUT_TEXT,
                self._remove_selected,
                description="Remove selected",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.BACKSPACE_POINT_SHORTCUT_TEXT,
                self._remove_selected,
                description="Remove selected",
                ignore_auto_repeat=True,
            ),
            Shortcut(
                constants.OUT_OF_SLICE_SHORTCUT_TEXT,
                self._toggle_out_of_slice,
                description="Toggle out of slice",
                ignore_auto_repeat=True,
            ),
        ]
        self.register_shortcuts("points", points_shortcuts)

    def _toggle_image_max_intensity(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Image):
            return
        layer.max_projection = not layer.max_projection

    def _undo(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not (isinstance(layer, Labels) or isinstance(layer, Points)):
            return
        if layer.editable and layer.visible:
            layer.undo()

    def _redo(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not (isinstance(layer, Labels) or isinstance(layer, Points)):
            return
        if layer.editable and layer.visible:
            layer.redo()

    def _shuffle_colors(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        layer.new_connected_components_colormap()

    def _pan_zoom_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer:
            return
        layer.mode = layer._modeclass.PAN_ZOOM

    def _paint_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        if layer.editable and layer.visible:
            layer.mode = layer._modeclass.PAINT

    def _fill_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        if layer.editable and layer.visible:
            layer.mode = layer._modeclass.FILL

    def _erase_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        if layer.editable and layer.visible:
            layer.mode = layer._modeclass.ERASE

    def _next_label(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        if layer.selected_label == len(layer.labels) - 1:
            return
        layer.selected_label += 1

    def _prev_label(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        if layer.selected_label == 0:
            return
        layer.selected_label -= 1

    def _toggle_fill_3d(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Labels):
            return
        layer.fill_3d = not layer.fill_3d

    def _toggle_preserve_background(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Labels) or len(layer.labels) <= 2:
            return
        layer.preserve_background = not layer.preserve_background

    def _toggle_show_selected(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Labels) or len(layer.labels) <= 2:
            return
        layer.show_selected_label = not layer.show_selected_label

    def _toggle_out_of_slice(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer or not isinstance(layer, Points):
            return
        layer.out_of_slice_display = not layer.out_of_slice_display

    def _add_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Points):
            return
        if layer.editable and layer.visible:
            layer.mode = layer._modeclass.ADD

    def _select_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Points):
            return
        if layer.editable and layer.visible:
            layer.mode = layer._modeclass.SELECT

    def _select_all(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer or not isinstance(layer, Points):
            return
        if not layer.editable:
            return

        new_selected = set(layer._indices_view[: len(layer._view_data)])
        if new_selected & layer.selected_data == new_selected:
            layer.selected_data = layer.selected_data - new_selected
        else:
            layer.selected_data = layer.selected_data | new_selected
        layer._set_highlight()

    def _remove_selected(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )

        if not layer or not isinstance(layer, Points):
            return

        if not (layer.editable and layer.visible):
            return

        selected_indices = set()
        try:
            selected_indices = {
                int(i)
                for i in layer.selected_data
                if 0 <= int(i) < len(layer.data)
            }
        except Exception:
            selected_indices = set()

        pair_metadata = {}
        if selected_indices:
            pair_metadata = self._build_pair_metadata_after_point_deletion(
                layer,
                selected_indices,
            )

        layer.remove_selected()

        if pair_metadata:
            self._save_points_pair_metadata(layer, pair_metadata)

        try:
            layer.selected_data = set()
            layer._set_highlight()
        except Exception:
            pass

        self._saved = False

        if self._viewer_model.active_layer == layer:
            self._set_preview_options()

    def _change_mode_to_pan_zoom(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer:
            return
        self._previous_layer_mode = layer._modeclass(layer.mode)
        layer.mode = layer._modeclass.PAN_ZOOM

    def _return_previous_mode(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
        )
        if not layer:
            return
        if layer.editable and layer.visible:
            layer.mode = self._previous_layer_mode

    def _toggle_layer_visibility(self) -> None:
        layer = (
            self._qt_active_layer_list.list.activeItem
            or self._qt_next_stage_list.next_stage_list.activeItem
            or self._qt_additional_layer_list.list.activeItem
        )
        if not layer:
            return
        layer.visible = not layer.visible

    def _reset_scroll_progress(self) -> None:
        self._viewer_model.dims._scroll_progress = 0
        self._viewer_model.dims.current_step = (0, 0, 0)

    def _ndisplay_change(self) -> None:
        self._viewer_model.dims.ndisplay = 2 + (self._viewer_model.dims.ndisplay == 2)
        if self._layer_controls_widget.current_widget is not self._empty_controls:
            self._layer_controls_widget.current_widget.ndisplay = (
                self._viewer_model.dims.ndisplay
            )
        if (
            self._preview_options_widget.current_widget is not self._no_parameters
            and self._preview_options_widget.current_widget is not self._empty_options
        ):
            self._preview_options_widget.current_widget.ndisplay = (
                self._viewer_model.dims.ndisplay
            )

    @property
    def type(self) -> str:
        return self._project_info.type

    @property
    def saved(self) -> bool:
        return self._saved and len(self._viewer_model.unsaved_layers()) == 0

    @classmethod
    def check_file(cls, path: str) -> bool:
        return ProjectInfo.check_file(path)

    @classmethod
    def create_from_image(
        cls,
        folder: str,
        name: str,
        image_path: str,
        original_image: str,
        z_scale: float,
        y_scale: float,
        x_scale: float,
        subtype: str = constants.IMAGE,
    ) -> str:
        new_image_path = LAYERS_PATH + "/0.tif"
        data = imread(image_path)
        max_value = np.max(data)
        min_value = np.min(data)
        if min_value != 0 or max_value != 255:
            data -= min_value
            alpha = 255 / (max_value - min_value)
            data = data.astype("float") * alpha
            data = np.round(data).astype("uint8")
        imwrite(folder + new_image_path, data=data)

        project = ProjectInfo(folder=folder)
        project.name = name
        project.subtype = subtype
        project.original_image = original_image
        project.shape = data.shape
        project.original_shape = data.shape
        project.real_scale = [z_scale, y_scale, x_scale]
        project.scale = [1.0, 1.0, 1.0]

        translater = Translater.instance()
        image_layer_desc = LayerInfo()
        image_layer_params = LayerParameters()
        image_layer_desc.id = 0
        image_layer_desc.type = Types.IMAGE
        image_layer_params.layer_id = 0
        image_layer_params.name = translater.get_translation("image")
        image_layer_params.file = new_image_path
        image_layer_params.preview = True

        project.layers_parameters = {0: image_layer_params}
        project.data.layers = {0: image_layer_desc}
        project.data.last_layer_id = 0

        project.save()

        return project.filename
    @classmethod
    def create_from_mesh(
            cls,
            folder: str,
            name: str,
            mesh_path: str,
            original_mesh: str,
    ) -> str | None:
        # фиксируем желаемое разрешение (можно вынести как параметр)
        n=256
        shape = (n, n, n)  # (z, y, x)

        new_mesh_path = LAYERS_PATH + "/0.off"
        copy(mesh_path, folder + new_mesh_path)

        surface_poly = Polyhedron_3(folder + new_mesh_path)
        descriptions, error = create_mesh_descriptions(surface_poly, folder + AUXILIARY_PATH)
        if error != "":
            return None
        descriptions_file = AUXILIARY_PATH + descriptions

        #Вычисление bounding box и scale
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
            if mesh.is_empty or len(mesh.vertices) == 0:
                raise RuntimeError("Меш пустой или не загрузился")
        except Exception as ex:
            print("Ошибка при загрузке меша через trimesh:", ex)
            return None

        bounds_min, bounds_max = mesh.bounds  # (min_xyz), (max_xyz)
        real_size = bounds_max - bounds_min  # (size_x, size_y, size_z)

        shape_z, shape_y, shape_x = shape   # shape в формате (z, y, x)

        eps = 1e-8  # чтобы не делить на 0
        sx, sy, sz = real_size.tolist()
        if sx == 0:
            sx = eps
        if sy == 0:
            sy = eps
        if sz == 0:
            sz = eps

        # вычисляем физический размер одного вокселя по каждой оси
        z_scale = sz / float(shape_z)
        y_scale = sy / float(shape_y)
        x_scale = sx / float(shape_x)

        shape_for_project = shape  # (z,y,x)
        project = ProjectInfo(folder=folder)
        project.name = name
        project.subtype = constants.POLYGON_MESH
        project.original_image = original_mesh
        project.shape = shape_for_project
        project.original_shape = shape_for_project
        project.real_scale = [x_scale, y_scale, z_scale]  # physical voxel size
        project.scale = [1.0, 1.0, 1.0]
        project.min_coordinates = [bounds_min[0], bounds_min[1], bounds_min[2]]

        translater = Translater.instance()
        mesh_layer_desc = LayerInfo()
        mesh_layer_params = LayerParameters()

        mesh_layer_desc.id = 0
        mesh_layer_desc.type = Types.POLYGON_MESH

        mesh_layer_params.layer_id = 0
        mesh_layer_params.name = translater.get_translation("polygon mesh")
        mesh_layer_params.preview = False
        mesh_layer_params.mesh_file = new_mesh_path
        mesh_layer_params.tmp_mesh_file = ""

        try:
            mesh_layer_params.mesh_source_tif_file = _voxelize_dendrite(
                [new_mesh_path, folder, shape_for_project, [x_scale, y_scale, z_scale], bounds_min]
            )
            mesh_layer_params.tmp_mesh_source_tif_file = mesh_layer_params.mesh_source_tif_file
        except Exception as ex:
            print("Ошибка при вызове _voxelize_dendrite:", ex)
            return None

        mesh_v_f = _mesh_to_v_f(surface_poly, shape_for_project, [x_scale, y_scale, z_scale], bounds_min)
        data = (
            mesh_v_f[0],
            mesh_v_f[1],
            np.ones(len(mesh_v_f[0])),
        )
        data = {
            "vertices": data[0].tolist(),
            "faces": data[1].tolist(),
            "vertex_values": data[2].tolist(),
        }
        file = (
                f"/layer_data_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".json"
        )
        f = open(folder + LAYERS_PATH + file, "w")
        dump(data, f)
        f.close()

        mesh_layer_params.file = LAYERS_PATH + file
        mesh_layer_params.additional_files = {"descriptions": descriptions_file}
        project.layers_parameters = {0: mesh_layer_params}
        project.data.layers = {0: mesh_layer_desc}
        project.data.last_layer_id = 0

        project.save()

        return project.filename

    @classmethod
    def create(
        cls,
        window: "Window",
        show_workflow_mode: bool = True,
        default_workflow_mode: str | None = None,
    ) -> Optional[str]:
        from projects.segmentation.widgets.qt_create_dialog import (
            CreateSegmentationDialog,
        )

        dlg = CreateSegmentationDialog(
            window._qt_window,
            show_workflow_mode=show_workflow_mode,
            default_workflow_mode=default_workflow_mode,
        )
        if dlg.exec() == QDialog.Accepted:
            name = dlg.project_name_line.text()
            folder = dlg.project_folder_line.text() + "/" + name
            original_file_path = dlg.project_image_line.text()
            _, original_file = os.path.split(original_file_path)
            mode = dlg.project_mode_combo.currentData()
            z_scale = dlg.z_scale.value()
            y_scale = dlg.y_scale.value()
            x_scale = dlg.x_scale.value()

            os.mkdir(folder)
            os.mkdir(folder + COLORMAPS_PATH)
            os.mkdir(folder + ADDITIONAL_COLORMAPS_PATH)
            os.mkdir(folder + LAYERS_PATH)
            os.mkdir(folder + AUXILIARY_PATH)

            if original_file.lower().endswith((".tif", ".tiff")):
                return cls.create_from_image(
                    folder,
                    name,
                    original_file_path,
                    original_file,
                    z_scale,
                    y_scale,
                    x_scale,
                    mode,
                )
            elif original_file.lower().endswith(".off"):
                os.mkdir(folder + TMP)
                os.mkdir(folder + TMP_AUXILIARY_PATH)
                os.mkdir(folder + TMP_LAYERS_PATH)
                return cls.create_from_mesh(
                    folder,
                    name,
                    original_file_path,
                    original_file,
                )
        return None

    @classmethod
    def open(cls, window: "Window", path: str) -> Optional["ProjectBase"]:
        try:
            if re.fullmatch(r"[a-zA-Z0-9_\-:\\/\.]+", path):
                return SegmentationProject(
                    window, ProjectInfo.load(path), SegmentationSettings.instance()
                )
            else:
                show_error("Invalid characters in path. Move project in other folder")
                return None
        except:
            return None

    def change_theme(self, theme_id) -> None:
        self._viewer.set_theme(theme_id)

    def save(self, save: bool = True) -> None:
        if not self._loaded or (not self._closing and self.saved):
            return

        if (
            len(self._layer_loader_task_queue) > 0
            and self._layer_loader_task_queue[-1].type == TaskType.SAVE
        ) or (
            self._active_layer_loader_task
            and self._active_layer_loader_task.type == TaskType.SAVE
        ):
            return

        self._saving = True
        QApplication.instance().installEventFilter(self._input_filter)
        if (
            not self._window._qt_window._status_bar._activity_item._activityBtn.isChecked()
        ):
            self._toggle_activity_dialog = True
            self._window.on_toggle_activity_dock()
        else:
            self._toggle_activity_dialog = False

        layers_data = []
        self._control_buttons.buttons.restoreButton.setEnabled(False)
        if save:
            for layer in self._viewer_model.unsaved_layers():
                if (
                    layer.metadata["type"] == Types.POLYGON_MESH
                    or layer.metadata["type"] == Types.POLYGON_MESH_SEGMENTATION
                    or layer.metadata["type"] == Types.FINAL_SEGMENTATION
                ):
                    layers_data.append((layer._id, layer.data))
                else:
                    layers_data.append((layer._id, layer.data.copy()))
                layer._changed = False
        params = {"save": save, "layers_data": layers_data}
        self._layer_loader_task_queue.append(
            LayerLoaderTask(
                TaskType.SAVE,
                progress(total=0, desc="save in queue"),
                params,
            )
        )
        self._next_layer_loader_task()

    def close(self) -> None:
        self._closing = True
        self._data_processing_task_queue.clear()
        self._data_processing_communication_queue_out.put("stop")
        self._layer_loader_communication_queue_out.put("stop")
        if self._loaded:
            self._window.set_title()
            self._window.remove_dock_widget(self._project_info_widget)
            self._window.remove_dock_widget(self._preview_options_widget)
            self._window.remove_dock_widget(self._qt_next_stage_list)
            self._window.remove_dock_widget(self._layer_controls_widget)
            self._window.remove_dock_widget(self._qt_additional_layer_list)
            self._window.remove_dock_widget(self._qt_active_layer_list)
            self._window.remove_dock_widget(self._viewer_buttons)
            self._window.remove_dock_widget(self._control_buttons)

        save = True
        if (
            not self.saved
            and not ApplicationSettings.instance().state["auto_saving_when_closing"]
        ):
            dialog = QtConfirmSaveDialog(self._window._qt_window)
            if dialog.exec() != QDialog.Accepted:
                save = False
        self.save(save)
        self._window.set_central_widget()
        self._window.set_menus()
        self._window.block_project_specific_actions()
        self._close()

    def on_settings_update(self, state: Dict[str, Any]) -> None:
        if "highlight_thickness" in state:
            self._viewer_model.set_highlight_thickness(state["highlight_thickness"])
        self._settings.save()
