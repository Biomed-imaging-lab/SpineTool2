import os
import shutil
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from json import dump, load
from multiprocessing import Queue as MPQueue
from queue import Queue
from typing import Any, Dict

import numpy as np
from tifffile import imread, imwrite

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from projects.segmentation.utils.constants import (
    AUXILIARY_PATH,
    LAYERS_PATH,
    TMP_AUXILIARY_PATH,
    TMP_LAYERS_PATH,
    Types,
)
from projects.segmentation.utils.layer_parameters_tracker import LayerParametersTracker
from projects.segmentation.utils.project_info import (
    FinalSegmentationData,
    LayerInfo,
    LayerParameters,
    NonEditableLayerInfo,
    ProjectData,
    ProjectInfo,
    SegmentationData,
    SurfaceData,
)
from utils.colormaps.colormap_utils import AVAILABLE_COLORMAPS_NAMES_SHORT_LIST
from viewer.layers.base.base import Layer
from viewer.layers.image.image import Image
from viewer.layers.labels.labels import Labels
from viewer.layers.points._points_constants import Symbol
from viewer.layers.points.points import Points
from viewer.layers.surface.surface import Surface


class LayerLoaderState:
    def __init__(
        self,
        deleted_layers: set = set(),
        new_layers: set = set(),
        history: deque = deque(),
        staged_history: list = [],
        block_history: bool = False,
    ):
        self.deleted_layers = deleted_layers
        self.new_layers = new_layers
        self.history = history
        self.staged_history = staged_history
        self.block_history = block_history


class TaskResult:
    def __init__(
        self,
        result: Any,
        layer_loader_state: LayerLoaderState | None = None,
        project_data: ProjectData | None = None,
    ):
        self.layer_loader_state = layer_loader_state
        self.project_data = project_data
        self.result = result


def _remove_file(path) -> None:
    try:
        os.remove(path)
    except:
        pass


class LayerLoader:
    EMPTY_IMAGE = np.array([[[0]]], dtype=np.uint8)
    EMPTY_LABELS = np.array([[[0]]], dtype=np.uint8)

    def __init__(self, project_info: ProjectInfo):
        self.parameters_tracker = LayerParametersTracker(project_info)
        self._project_info = project_info
        self._state = LayerLoaderState()

    @property
    def state(self) -> LayerLoaderState:
        return self._state

    @state.setter
    def state(self, new_state: LayerLoaderState) -> None:
        self._state = new_state

    def replace_additional_files(
        self, layer_id: int, new_files: Dict[str, str]
    ) -> None:
        parameters = self._project_info.layers_parameters[layer_id]
        for desc, filename in parameters.tmp_additional_files.items():
            if filename != new_files.get(
                desc, ""
            ) and filename != parameters.additional_files.get(desc, ""):
                _remove_file(self._project_info.folder + filename)
        parameters.tmp_additional_files = new_files.copy()

    def replace_mesh_files(
        self,
        layer_id: int,
        data: SurfaceData | FinalSegmentationData | SegmentationData,
    ) -> None:
        parameters = self._project_info.layers_parameters[layer_id]
        if isinstance(data, SurfaceData):
            if parameters.mesh_file != parameters.tmp_mesh_file:
                _remove_file(self._project_info.folder + parameters.tmp_mesh_file)
            parameters.tmp_mesh_file = data.mesh_file
            if parameters.mesh_source_tif_file != parameters.tmp_mesh_source_tif_file:
                _remove_file(
                    self._project_info.folder + parameters.tmp_mesh_source_tif_file
                )
            parameters.tmp_mesh_source_tif_file = data.tif_file
        elif isinstance(data, SegmentationData):
            tmp_spines_files = (
                set(parameters.tmp_spines_files)
                - set(parameters.spines_files)
                - set(data.spines_files)
            )
            if "" in tmp_spines_files:
                tmp_spines_files.remove("")
            for tmp_spine_file in tmp_spines_files:
                _remove_file(self._project_info.folder + tmp_spine_file)
            parameters.tmp_spines_files = data.spines_files
            tmp_adjusted_spines_files = (
                set(parameters.tmp_adjusted_spines_files)
                - set(parameters.adjusted_spines_files)
                - set(data.adjusted_spines_files)
                - set(parameters.spines_files)
                - set(data.spines_files)
            )
            if "" in tmp_adjusted_spines_files:
                tmp_adjusted_spines_files.remove("")
            for tmp_adjusted_spine_file in tmp_adjusted_spines_files:
                try:
                    _remove_file(self._project_info.folder + tmp_adjusted_spine_file)
                except:
                    pass
            parameters.tmp_adjusted_spines_files = data.adjusted_spines_files
        elif isinstance(data, FinalSegmentationData):
            if parameters.tmp_mesh_file != data.mesh_file:
                if parameters.mesh_file != parameters.tmp_mesh_file:
                    _remove_file(self._project_info.folder + parameters.tmp_mesh_file)
                parameters.tmp_mesh_file = data.mesh_file
            tmp_spines_files = (
                set(parameters.tmp_spines_files)
                - set(parameters.spines_files)
                - set(data.spines_files)
            )
            for tmp_spine_file in tmp_spines_files:
                _remove_file(self._project_info.folder + tmp_spine_file)
            parameters.tmp_spines_files = data.spines_files
            parameters.tmp_adjusted_spines_files = data.spines_files

    def _register_layer(self, info: LayerInfo, parameters: LayerParameters) -> None:
        if info.id is None:
            self._project_info.data.last_layer_id += 1
            info.id = self._project_info.data.last_layer_id
            self._state.new_layers.add(info.id)
            self._project_info.data.layers[info.id] = info
            self._project_info.layers_parameters[info.id] = parameters
            self._project_info.layers_parameters[info.id].layer_id = info.id
            if info.parent_id is not None:
                self._project_info.data.layers[info.parent_id].child_layers.append(
                    info.id
                )

    def load_layer(self, info: LayerInfo | NonEditableLayerInfo | int) -> tuple:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, NonEditableLayerInfo):
            layer_type = self._project_info.data.layers[info.layer_id].type
        else:
            layer_type = info.type

        if layer_type == Types.IMAGE or layer_type == Types.BACKGROUND_IMAGE:
            return self.load_image(info)
        if (
            layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
        ):
            return self.load_labels(info)
        if layer_type == Types.POINTS:
            return self.load_points(info)
        if (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ):
            return self.load_surface(info)

    def create_layer(
        self,
        info: LayerInfo | NonEditableLayerInfo,
        parameters: LayerParameters | None,
        data: (
            np.ndarray | SurfaceData | FinalSegmentationData | SegmentationData | None
        ) = None,
    ) -> Layer:
        if isinstance(info, NonEditableLayerInfo):
            layer_type = self._project_info.data.layers[info.layer_id].type
        else:
            layer_type = info.type

        if layer_type == Types.IMAGE or layer_type == Types.BACKGROUND_IMAGE:
            return self.create_image(
                info, parameters, data if data is not None else LayerLoader.EMPTY_IMAGE
            )
        if (
            layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
        ):
            return self.create_labels(
                info, parameters, data if data is not None else LayerLoader.EMPTY_LABELS
            )
        if layer_type == Types.POINTS:
            return self.create_points(info, parameters, data)
        if (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ):
            return self.create_surface(info, parameters, data)

    def update_layer(
        self,
        layer_id: int,
        data: np.ndarray | SurfaceData | FinalSegmentationData | SegmentationData,
        layer: Layer | None = None,
        metadata: dict = None,
    ) -> None:
        if metadata:
            self._project_info.layers_parameters[layer_id].tmp_metadata.update(metadata)

        if layer is None:
            return

        layer_type = self._project_info.data.layers[layer_id].type
        if (
            layer_type == Types.IMAGE
            or layer_type == Types.BACKGROUND_IMAGE
            or layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
            or layer_type == Types.POINTS
        ):
            layer.data = data
        elif (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ):
            if isinstance(data, SurfaceData) or isinstance(data, SegmentationData):
                layer.data = data.mesh_v_f
            elif isinstance(data, FinalSegmentationData):
                layer.data = data.mesh_v_f_vv

    def export_layer(self, layer_id: int, data: np.ndarray | None, folder: str) -> None:
        layer_type = self._project_info.data.layers[layer_id].type
        if (
            layer_type == Types.IMAGE
            or layer_type == Types.BACKGROUND_IMAGE
            or layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
        ):
            imwrite(
                folder + f"/{self._project_info.layers_parameters[layer_id].name}.tif",
                data=data,
            )
        elif layer_type == Types.POINTS:
            data.tofile(
                folder + f"/{self._project_info.layers_parameters[layer_id].name}.bin"
            )
        elif (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ):
            self.export_surface(layer_id, folder)

    @contextmanager
    def block_history(self):
        prev = self._state.block_history
        self._state.block_history = True
        try:
            yield
            if self._state.staged_history:
                self._state.history.append(self._state.staged_history)
                self._state.staged_history = []
        finally:
            self._state.block_history = prev

    def delete_layer(self, info: LayerInfo | NonEditableLayerInfo | int) -> set:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, LayerInfo):
            if info.id in self._state.deleted_layers:
                return set()
        if self._state.block_history:
            self._state.staged_history.append(info)
        else:
            self._state.history.append([info])
        deleted = set()
        if isinstance(info, LayerInfo):
            self._state.deleted_layers.add(info.id)
            deleted.add(info.id)
            parent_id = info.parent_id
            if parent_id is not None:
                self._project_info.data.layers[parent_id].child_layers.remove(info.id)
            queue = Queue()
            queue.put(info.id)
            while not queue.empty():
                info = self._project_info.data.layers[queue.get()]
                for children_id in info.child_layers:
                    queue.put(children_id)
                    self._state.deleted_layers.add(children_id)
                    deleted.add(children_id)
        return deleted

    def restore_last(self) -> list:
        last = self._state.history.pop()
        for info in last:
            if isinstance(info, LayerInfo):
                parent_id = info.parent_id
                if parent_id is not None:
                    self._project_info.data.layers[parent_id].child_layers.append(
                        info.id
                    )
                queue = Queue()
                self._state.deleted_layers.remove(info.id)
                queue.put(info.id)
                while not queue.empty():
                    info = self._project_info.data.layers[queue.get()]
                    for children_id in info.child_layers:
                        queue.put(children_id)
                        self._state.deleted_layers.remove(children_id)
        return last

    def save_layer_data(
        self,
        layer_id,
        data: (
            np.ndarray | tuple | SurfaceData | FinalSegmentationData | SegmentationData
        ),
    ) -> None:
        folder = self._project_info.folder + TMP_LAYERS_PATH

        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = f"layer_data_" + str(datetime.now()).replace(".", "_").replace(
            " ", "_"
        ).replace(":", "_")
        layer_type = self._project_info.data.layers[layer_id].type
        if (
            layer_type == Types.IMAGE
            or layer_type == Types.BACKGROUND_IMAGE
            or layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
        ):
            file = f"/{filename}.tif"
            imwrite(folder + file, data=data)
        elif layer_type == Types.POINTS:
            file = f"/{filename}.bin"
            data.tofile(folder + file)
        elif (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ):
            if isinstance(data, SurfaceData) or isinstance(data, SegmentationData):
                data = (
                    data.mesh_v_f[0],
                    data.mesh_v_f[1],
                    np.ones(len(data.mesh_v_f[0])),
                )
            elif isinstance(data, FinalSegmentationData):
                data = data.mesh_v_f_vv

            file = f"/{filename}.json"
            data = {
                "vertices": data[0].tolist(),
                "faces": data[1].tolist(),
                "vertex_values": data[2].tolist(),
            }
            f = open(folder + file, "w")
            dump(data, f)
            f.close()
        new_file = TMP_LAYERS_PATH + file
        if (
            self._project_info.layers_parameters[layer_id].tmp_file
            != self._project_info.layers_parameters[layer_id].file
        ):
            _remove_file(
                self._project_info.folder
                + self._project_info.layers_parameters[layer_id].tmp_file
            )
        self._project_info.layers_parameters[layer_id].tmp_file = new_file

    def fix_state(self, save=True) -> ProjectInfo:
        if save:
            for deleted in self._state.deleted_layers:
                layer_parameters = self._project_info.layers_parameters[deleted]
                layer_type = self._project_info.data.layers[deleted].type

                deleted_files = set()
                deleted_files.add(layer_parameters.file)
                deleted_files.add(layer_parameters.tmp_file)
                if layer_type == Types.POLYGON_MESH:
                    deleted_files.add(layer_parameters.mesh_file)
                    deleted_files.add(layer_parameters.tmp_mesh_file)
                    deleted_files.add(layer_parameters.mesh_source_tif_file)
                    deleted_files.add(layer_parameters.tmp_mesh_source_tif_file)
                elif layer_type == Types.POLYGON_MESH_SEGMENTATION:
                    deleted_files = deleted_files.union(
                        set(layer_parameters.spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.adjusted_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_adjusted_spines_files)
                    )
                elif layer_type == Types.FINAL_SEGMENTATION:
                    deleted_files.add(layer_parameters.mesh_file)
                    deleted_files.add(layer_parameters.tmp_mesh_file)
                    deleted_files = deleted_files.union(
                        set(layer_parameters.spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.adjusted_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_adjusted_spines_files)
                    )

                if (
                    layer_type == Types.BINARIZATION
                    or layer_type == Types.NECKS
                    or layer_type == Types.VOXEL_MESH_SEGMENTATION
                ):
                    deleted_files.add(
                        self._project_info.layers_parameters[deleted].parameters.get(
                            "colormap", ""
                        )
                    )
                    deleted_files.add(
                        self._project_info.layers_parameters[deleted].parameters.get(
                            "connected_components_colormap", ""
                        )
                    )

                deleted_files = deleted_files.union(
                    set(
                        self._project_info.layers_parameters[
                            deleted
                        ].additional_files.values()
                    )
                )
                deleted_files = deleted_files.union(
                    set(
                        self._project_info.layers_parameters[
                            deleted
                        ].tmp_additional_files.values()
                    )
                )

                del self._project_info.layers_parameters[deleted]

                if "" in deleted_files:
                    deleted_files.remove("")
                for file in deleted_files:
                    _remove_file(self._project_info.folder + file)
                del self._project_info.data.layers[deleted]

            for layer_parameters in self._project_info.layers_parameters.values():
                layer_parameters.preview = layer_parameters.tmp_preview
                layer_parameters.metadata = layer_parameters.tmp_metadata.copy()
                layer_parameters.preview_need_update = (
                    layer_parameters.tmp_preview_need_update
                )
                if layer_parameters.preview:
                    deleted_files = (
                        set(layer_parameters.spines_files).union(
                            set(layer_parameters.adjusted_spines_files)
                        )
                        - set(layer_parameters.tmp_spines_files)
                        - set(layer_parameters.tmp_adjusted_spines_files)
                    )
                    if "" in deleted_files:
                        deleted_files.remove("")
                    for file in deleted_files:
                        _remove_file(self._project_info.folder + file)
                    layer_parameters.preview_partially_fixed = (
                        layer_parameters.tmp_preview_partially_fixed
                    )
                    layer_parameters.deleted_spines = (
                        layer_parameters.tmp_deleted_spines.copy()
                    )
                    for i in range(len(layer_parameters.tmp_adjusted_spines_files)):
                        file = layer_parameters.tmp_adjusted_spines_files[i]
                        if file.startswith(TMP_LAYERS_PATH):
                            new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                            shutil.copy(
                                self._project_info.folder + file,
                                self._project_info.folder + new_file,
                            )
                            layer_parameters.tmp_adjusted_spines_files[i] = new_file
                    layer_parameters.adjusted_spines_files = (
                        layer_parameters.tmp_adjusted_spines_files.copy()
                    )
                    for i in range(len(layer_parameters.tmp_spines_files)):
                        file = layer_parameters.tmp_spines_files[i]
                        if file.startswith(TMP_LAYERS_PATH):
                            new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                            shutil.copy(
                                self._project_info.folder + file,
                                self._project_info.folder + new_file,
                            )
                            layer_parameters.tmp_spines_files[i] = new_file
                    layer_parameters.spines_files = (
                        layer_parameters.tmp_spines_files.copy()
                    )
                else:
                    deleted_files = set(layer_parameters.spines_files).union(
                        set(layer_parameters.tmp_spines_files)
                    ).union(set(layer_parameters.adjusted_spines_files)) - set(
                        layer_parameters.tmp_adjusted_spines_files
                    )
                    if "" in deleted_files:
                        deleted_files.remove("")
                    for file in deleted_files:
                        _remove_file(self._project_info.folder + file)
                    layer_parameters.preview_partially_fixed = False
                    layer_parameters.tmp_preview_partially_fixed = False
                    for deleted_surface in sorted(
                        layer_parameters.tmp_deleted_spines, reverse=True
                    ):
                        _remove_file(
                            self._project_info.folder
                            + layer_parameters.tmp_adjusted_spines_files[
                                deleted_surface
                            ]
                        )
                        del layer_parameters.tmp_adjusted_spines_files[deleted_surface]
                    layer_parameters.deleted_spines = set()
                    layer_parameters.tmp_deleted_spines = set()
                    for i in range(len(layer_parameters.tmp_adjusted_spines_files)):
                        file = layer_parameters.tmp_adjusted_spines_files[i]
                        if file.startswith(TMP_LAYERS_PATH):
                            new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                            shutil.copy(
                                self._project_info.folder + file,
                                self._project_info.folder + new_file,
                            )
                            layer_parameters.tmp_adjusted_spines_files[i] = new_file
                    layer_parameters.adjusted_spines_files = (
                        layer_parameters.tmp_adjusted_spines_files.copy()
                    )
                    layer_parameters.spines_files = (
                        layer_parameters.tmp_adjusted_spines_files.copy()
                    )
                    layer_parameters.tmp_spines_files = (
                        layer_parameters.tmp_adjusted_spines_files.copy()
                    )
                file = layer_parameters.tmp_file
                if file.startswith(TMP_LAYERS_PATH):
                    if layer_parameters.file != "":
                        _remove_file(self._project_info.folder + layer_parameters.file)
                    new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                    shutil.copy(
                        self._project_info.folder + file,
                        self._project_info.folder + new_file,
                    )
                    layer_parameters.tmp_file = new_file
                layer_parameters.file = layer_parameters.tmp_file
                file = layer_parameters.tmp_mesh_file
                if file.startswith(TMP_LAYERS_PATH):
                    if layer_parameters.mesh_file != "":
                        _remove_file(
                            self._project_info.folder + layer_parameters.mesh_file
                        )
                    new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                    shutil.copy(
                        self._project_info.folder + file,
                        self._project_info.folder + new_file,
                    )
                    layer_parameters.tmp_mesh_file = new_file
                layer_parameters.mesh_file = layer_parameters.tmp_mesh_file
                file = layer_parameters.tmp_mesh_source_tif_file
                if file.startswith(TMP_LAYERS_PATH):
                    if layer_parameters.mesh_source_tif_file != "":
                        _remove_file(
                            self._project_info.folder
                            + layer_parameters.mesh_source_tif_file
                        )
                    new_file = file.replace(TMP_LAYERS_PATH, LAYERS_PATH)
                    shutil.copy(
                        self._project_info.folder + file,
                        self._project_info.folder + new_file,
                    )
                    layer_parameters.tmp_mesh_source_tif_file = new_file
                layer_parameters.mesh_source_tif_file = (
                    layer_parameters.tmp_mesh_source_tif_file
                )
                for desc, file in layer_parameters.tmp_additional_files.items():
                    old_file = layer_parameters.additional_files.get(desc, "")
                    if file.startswith(TMP_AUXILIARY_PATH):
                        if old_file != "":
                            _remove_file(self._project_info.folder + old_file)
                        new_file = file.replace(TMP_AUXILIARY_PATH, AUXILIARY_PATH)
                        shutil.copy(
                            self._project_info.folder + file,
                            self._project_info.folder + new_file,
                        )
                        layer_parameters.additional_files[desc] = new_file
                layer_parameters.tmp_additional_files = (
                    layer_parameters.additional_files.copy()
                )
        else:
            self._project_info.shape = self._project_info.original_shape

            for restored in self._state.deleted_layers:
                if self._project_info.data.layers[restored].parent_id is not None:
                    self._project_info.data.layers[
                        self._project_info.data.layers[restored].parent_id
                    ].child_layers.append(self._project_info.data.layers[restored].id)

            for deleted in self._state.new_layers:
                layer_parameters = self._project_info.layers_parameters[deleted]
                layer_type = self._project_info.data.layers[deleted].type

                deleted_files = set()
                deleted_files.add(layer_parameters.tmp_file)
                if layer_type == Types.POLYGON_MESH:
                    deleted_files.add(layer_parameters.tmp_mesh_file)
                    deleted_files.add(layer_parameters.tmp_mesh_source_tif_file)
                elif layer_type == Types.POLYGON_MESH_SEGMENTATION:
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_adjusted_spines_files)
                    )
                elif layer_type == Types.FINAL_SEGMENTATION:
                    deleted_files.add(layer_parameters.tmp_mesh_file)
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_spines_files)
                    )
                    deleted_files = deleted_files.union(
                        set(layer_parameters.tmp_adjusted_spines_files)
                    )

                if (
                    layer_type == Types.BINARIZATION
                    or layer_type == Types.NECKS
                    or layer_type == Types.VOXEL_MESH_SEGMENTATION
                ):
                    deleted_files.add(
                        self._project_info.layers_parameters[deleted].parameters.get(
                            "colormap", ""
                        )
                    )
                    deleted_files.add(
                        self._project_info.layers_parameters[deleted].parameters.get(
                            "connected_components_colormap", ""
                        )
                    )

                deleted_files = deleted_files.union(
                    set(
                        self._project_info.layers_parameters[
                            deleted
                        ].tmp_additional_files.values()
                    )
                )

                del self._project_info.layers_parameters[deleted]

                if "" in deleted_files:
                    deleted_files.remove("")
                for file in deleted_files:
                    _remove_file(self._project_info.folder + file)

                if (
                    self._project_info.data.layers[deleted].parent_id
                    in self._project_info.data.layers
                ):
                    self._project_info.data.layers[
                        self._project_info.data.layers[deleted].parent_id
                    ].child_layers.remove(deleted)
                del self._project_info.data.layers[deleted]

            for layer_parameters in self._project_info.layers_parameters.values():
                layer_parameters.tmp_preview = layer_parameters.preview
                layer_parameters.tmp_metadata = layer_parameters.metadata.copy()
                layer_parameters.tmp_preview_need_update = (
                    layer_parameters.preview_need_update
                )
                layer_parameters.tmp_preview_partially_fixed = (
                    layer_parameters.preview_partially_fixed
                )
                layer_parameters.tmp_deleted_spines = (
                    layer_parameters.deleted_spines.copy()
                )
                deleted_files = (
                    set(layer_parameters.tmp_spines_files).union(
                        set(layer_parameters.tmp_adjusted_spines_files)
                    )
                    - set(layer_parameters.adjusted_spines_files)
                    - set(layer_parameters.spines_files)
                )
                if "" in deleted_files:
                    deleted_files.remove("")
                for file in deleted_files:
                    _remove_file(self._project_info.folder + file)
                layer_parameters.tmp_adjusted_spines_files = (
                    layer_parameters.adjusted_spines_files.copy()
                )
                layer_parameters.tmp_spines_files = layer_parameters.spines_files.copy()
                if layer_parameters.tmp_file != layer_parameters.file:
                    _remove_file(self._project_info.folder + layer_parameters.tmp_file)
                    layer_parameters.tmp_file = layer_parameters.file
                if layer_parameters.tmp_mesh_file != layer_parameters.mesh_file:
                    _remove_file(
                        self._project_info.folder + layer_parameters.tmp_mesh_file
                    )
                    layer_parameters.tmp_mesh_file = layer_parameters.mesh_file
                if (
                    layer_parameters.tmp_mesh_source_tif_file
                    != layer_parameters.mesh_source_tif_file
                ):
                    _remove_file(
                        self._project_info.folder
                        + layer_parameters.tmp_mesh_source_tif_file
                    )
                    layer_parameters.tmp_mesh_source_tif_file = (
                        layer_parameters.mesh_source_tif_file
                    )
                for desc, file in layer_parameters.tmp_additional_files.items():
                    if file != layer_parameters.additional_files.get(desc, ""):
                        _remove_file(self._project_info.folder + file)
                layer_parameters.tmp_additional_files = (
                    layer_parameters.additional_files.copy()
                )

        self._state.new_layers = set()
        self._state.deleted_layers = set()

        return self._project_info

    def create_image(
        self,
        info: LayerInfo | NonEditableLayerInfo,
        parameters: LayerParameters | None,
        data: np.ndarray,
    ) -> Image:
        if isinstance(info, NonEditableLayerInfo):
            id = info.layer_id
            name = self._project_info.layers_parameters[id].name
            parent_id = None
            need_link = True
            params = info.parameters
            layer_type = self._project_info.data.layers[id].type
        else:
            name = parameters.name
            parent_id = info.parent_id
            if parent_id is not None and parent_id in self._state.deleted_layers:
                return None
            self._register_layer(info, parameters)
            id = info.id
            need_link = False
            params = parameters.parameters
            layer_type = info.type

        image = Image(
            id,
            name,
            data,
            blending=params.get("blending", "translucent"),
            colormap=params.get("colormap", "gray"),
            contrast_limits=params.get("contrast_limits", None),
            gamma=params.get("gamma", 1.0),
            interpolation2d=params.get("interpolation2d", "nearest"),
            interpolation3d=params.get("interpolation3d", "linear"),
            iso_threshold=params.get("iso_threshold", None),
            metadata={"type": layer_type, "parent_id": parent_id},
            opacity=params.get("opacity", 1.0),
            visible=params.get("visible", True),
            max_projection=params.get("max_projection", False),
            scale=self._project_info.scale,
        )

        if need_link:
            image.metadata["link"] = None

        return image

    def load_image(self, info: LayerInfo | NonEditableLayerInfo | int) -> tuple:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, LayerInfo) and info.id in self._state.deleted_layers:
            return None
        if isinstance(info, NonEditableLayerInfo):
            file = self._project_info.layers_parameters[info.layer_id].tmp_file
            parameters = None
        else:
            file = self._project_info.layers_parameters[info.id].tmp_file
            parameters = self._project_info.layers_parameters[info.id]
        data = imread(self._project_info.folder + file).astype(np.uint8)
        return (info, parameters, data)

    def create_labels(
        self,
        info: LayerInfo | NonEditableLayerInfo,
        parameters: LayerParameters | None,
        data: np.ndarray,
    ) -> Labels:
        if isinstance(info, NonEditableLayerInfo):
            id = info.layer_id
            name = self._project_info.layers_parameters[id].name
            parent_id = None
            need_link = True
            layer_type = self._project_info.data.layers[id].type
            params = info.parameters
        else:
            name = parameters.name
            parent_id = info.parent_id
            if parent_id is not None and parent_id in self._state.deleted_layers:
                return None
            self._register_layer(info, parameters)
            id = info.id
            need_link = False
            layer_type = info.type
            params = parameters.parameters

        labels = Labels(
            id,
            name,
            data,
            blending=params.get("blending", "translucent_no_depth"),
            colormap=self.parameters_tracker.load_colormap(
                params.get("colormap", None)
            ),
            connected_components_colormap=self.parameters_tracker.load_colormap(
                params.get("connected_components_colormap", None)
            ),
            labels=params.get("labels", ["background", "general_label"]),
            metadata={"type": layer_type, "parent_id": parent_id},
            opacity=params.get("opacity", 0.7),
            brush_settings=params.get("brush_settings", None),
            visible=params.get("visible", True),
            fill_3d=params.get("fill_3d", False),
            scale=params.get("scale", self._project_info.scale),
        )

        labels._show_selected_label = params.get("show_selected_label", False)
        labels._preserve_background = params.get("preserve_background", False)

        if need_link:
            labels.metadata["link"] = None

        return labels

    def load_labels(self, info: LayerInfo | NonEditableLayerInfo | int) -> tuple:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, LayerInfo) and info.id in self._state.deleted_layers:
            return None
        if isinstance(info, NonEditableLayerInfo):
            file = self._project_info.layers_parameters[info.layer_id].tmp_file
            parameters = None
        else:
            file = self._project_info.layers_parameters[info.id].tmp_file
            parameters = self._project_info.layers_parameters[info.id]
        if file == "":
            data = np.zeros(self._project_info.shape, dtype=np.uint8)
        else:
            data = imread(self._project_info.folder + file).astype(np.uint8)
        return (info, parameters, data)

    def create_points(
        self,
        info: LayerInfo | NonEditableLayerInfo,
        parameters: LayerParameters | None,
        data: np.ndarray,
    ) -> Points:
        if isinstance(info, NonEditableLayerInfo):
            id = info.layer_id
            layer_parameters = self._project_info.layers_parameters[id]
            name = layer_parameters.name
            parent_id = None
            need_link = True
            params = info.parameters
        else:
            layer_parameters = parameters
            name = parameters.name
            parent_id = info.parent_id
            if parent_id is not None and parent_id in self._state.deleted_layers:
                return None

            self._register_layer(info, parameters)

            id = info.id
            need_link = False
            params = parameters.parameters

        points_metadata = {
            "type": Types.POINTS,
            "parent_id": parent_id,
        }

        pair_keys = (
            "neck_restoration_pairs_version",
            "coordinate_order",
            "shaft_points",
            "auto_shaft_points",
            "pair_active",
            "pair_source",
            "pair_edit_state",
            "spine_component_ids",
            "shaft_component_ids",
        )

        metadata_sources = []

        if layer_parameters is not None:
            metadata_sources.append(getattr(layer_parameters, "metadata", {}) or {})
            metadata_sources.append(getattr(layer_parameters, "tmp_metadata", {}) or {})
            metadata_sources.append(getattr(layer_parameters, "parameters", {}) or {})

        if isinstance(params, dict):
            metadata_sources.append(params)

        for source in metadata_sources:
            if not isinstance(source, dict):
                continue

            for key in pair_keys:
                if key in source:
                    points_metadata[key] = source[key]

            nested_params = source.get("params", {})
            if isinstance(nested_params, dict):
                for key in pair_keys:
                    if key in nested_params:
                        points_metadata[key] = nested_params[key]

        points = Points(
            id,
            name,
            data,
            blending=params.get("blending", "translucent"),
            metadata=points_metadata,
            opacity=params.get("opacity", 0.7),
            size=params.get("size", 20),
            face_color=params.get("face_color", "white"),
            edge_color=params.get("edge_color", "white"),
            edge_width=params.get("edge_width", 0.0),
            symbol=params.get("symbol", "disc"),
            out_of_slice_display=params.get("out_of_slice_display", True),
            visible=params.get("visible", True),
            scale=self._project_info.scale,
        )

        if need_link:
            points.metadata["link"] = None

        return points

    def load_points(self, info: LayerInfo | NonEditableLayerInfo | int) -> tuple:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, LayerInfo) and info.id in self._state.deleted_layers:
            return None
        if isinstance(info, NonEditableLayerInfo):
            file = self._project_info.layers_parameters[info.layer_id].tmp_file
            parameters = None
        else:
            file = self._project_info.layers_parameters[info.id].tmp_file
            parameters = self._project_info.layers_parameters[info.id]
        data = np.fromfile(self._project_info.folder + file, dtype=np.uint16)
        data = np.reshape(data, (-1, 3))
        return (info, parameters, data)

    def create_surface(
        self,
        info: LayerInfo | NonEditableLayerInfo,
        parameters: LayerParameters | None,
        data: tuple | list | SurfaceData | FinalSegmentationData | SegmentationData,
    ) -> Surface:
        if isinstance(info, NonEditableLayerInfo):
            id = info.layer_id
            name = self._project_info.layers_parameters[id].name
            parent_id = None
            need_link = True
            layer_type = self._project_info.data.layers[id].type
            params = info.parameters
        else:
            name = parameters.name
            parent_id = info.parent_id
            if parent_id is not None and parent_id in self._state.deleted_layers:
                return None
            self._register_layer(info, parameters)
            id = info.id
            need_link = False
            layer_type = info.type
            params = parameters.parameters

            if isinstance(data, SurfaceData) or isinstance(data, SegmentationData):
                data = data.mesh_v_f
            elif isinstance(data, FinalSegmentationData):
                data = data.mesh_v_f_vv

        default_colormap = "gray"
        colormaps_names = None

        if layer_type == Types.POLYGON_MESH:
            default_colormap = "green"
        elif layer_type == Types.POLYGON_MESH_SEGMENTATION:
            default_colormap = "plasma"
            colormaps_names = AVAILABLE_COLORMAPS_NAMES_SHORT_LIST
        elif layer_type == Types.FINAL_SEGMENTATION:
            default_colormap = "viridis"

        surface = Surface(
            id,
            name,
            data,
            blending=params.get("blending", "translucent"),
            colormap=params.get("colormap", default_colormap),
            contrast_limits=params.get("contrast_limits", None),
            gamma=params.get("gamma", 1.0),
            metadata={"type": layer_type, "parent_id": parent_id},
            opacity=params.get("opacity", 1.0),
            shading=params.get("shading", "none"),
            visible=params.get("visible", True),
            colormaps_names=colormaps_names,
            scale=self._project_info.scale,
        )

        if need_link:
            surface.metadata["link"] = None

        return surface

    def load_surface(self, info: LayerInfo | NonEditableLayerInfo | int) -> tuple:
        if isinstance(info, int):
            info = self._project_info.data.layers[info]
        if isinstance(info, LayerInfo) and info.id in self._state.deleted_layers:
            return None
        if isinstance(info, NonEditableLayerInfo):
            file = self._project_info.layers_parameters[info.layer_id].tmp_file
            parameters = None
        else:
            file = self._project_info.layers_parameters[info.id].tmp_file
            parameters = self._project_info.layers_parameters[info.id]
        with open(self._project_info.folder + file) as f:
            data = load(f)
        data = (
            np.array(data["vertices"], dtype=np.uint16),
            np.array(data["faces"], dtype=np.uint32),
            np.array(data["vertex_values"], dtype=np.float16),
        )
        return (info, parameters, data)

    def export_surface(self, layer_id: int, folder: str) -> None:
        layer_type = self._project_info.data.layers[layer_id].type
        layer_parameters = self._project_info.layers_parameters[layer_id]

        if layer_type == Types.POLYGON_MESH:
            surface_poly = Polyhedron_3(
                self._project_info.folder + layer_parameters.tmp_mesh_file
            )
            for v in surface_poly.vertices():
                v.set_point(Point_3(v.point().z(), v.point().y(), v.point().x()))
            surface_poly.write_to_file(folder + f"/surface_mesh.off")

        elif layer_type == Types.POLYGON_MESH_SEGMENTATION:
            k = 0
            for i, file in enumerate(layer_parameters.tmp_adjusted_spines_files):
                if i not in layer_parameters.tmp_deleted_spines and file != "":
                    surface_poly = Polyhedron_3(self._project_info.folder + file)
                    for h in surface_poly.halfedges():
                        if h.is_border():
                            surface_poly.fill_hole(h)
                            h.facet().set_id(0)
                    for facet in surface_poly.facets():
                        if not facet.is_triangle():
                            surface_poly.create_center_vertex(facet.halfedge())
                    for v in surface_poly.vertices():
                        v.set_point(
                            Point_3(v.point().z(), v.point().y(), v.point().x())
                        )
                    surface_poly.write_to_file(folder + f"/spine_{k}.off")
                    k += 1

        elif layer_type == Types.FINAL_SEGMENTATION:
            surface_poly = Polyhedron_3(
                self._project_info.folder + layer_parameters.tmp_mesh_file
            )
            for v in surface_poly.vertices():
                v.set_point(Point_3(v.point().z(), v.point().y(), v.point().x()))
            surface_poly.write_to_file(folder + f"/surface_mesh.off")

            k = 0
            for i, file in enumerate(layer_parameters.tmp_adjusted_spines_files):
                if i not in layer_parameters.tmp_deleted_spines and file != "":
                    surface_poly = Polyhedron_3(self._project_info.folder + file)
                    for h in surface_poly.halfedges():
                        if h.is_border():
                            surface_poly.fill_hole(h)
                            h.facet().set_id(0)
                    for facet in surface_poly.facets():
                        if not facet.is_triangle():
                            surface_poly.create_center_vertex(facet.halfedge())
                    for v in surface_poly.vertices():
                        v.set_point(
                            Point_3(v.point().z(), v.point().y(), v.point().x())
                        )
                    surface_poly.write_to_file(folder + f"/spine_{k}.off")
                    k += 1

def load_layers(
    layer_loader: LayerLoader, infos: list, queue_in: MPQueue, queue_out: MPQueue
) -> None:
    stop = False
    for info, metadata in infos:
        if not queue_in.empty():
            queue_in.get()
            stop = True
            break
        queue_out.put(TaskResult((layer_loader.load_layer(info), metadata)))
    queue_out.put(TaskResult(not stop))


def delete_layer(
    layer_loader: LayerLoader,
    layer_info: LayerInfo | NonEditableLayerInfo | int,
    queue: MPQueue,
) -> None:
    result = TaskResult(layer_loader.delete_layer(layer_info))
    result.layer_loader_state = layer_loader.state
    result.project_data = layer_loader._project_info.data
    queue.put(result)


def restore_layer(layer_loader: LayerLoader, queue: MPQueue) -> None:
    non_editable_layers = []
    layers = []
    restored = layer_loader.restore_last()
    for info in restored:
        if isinstance(info, LayerInfo):
            layer = layer_loader.load_layer(info)
            if layer:
                layers.append(layer)
        else:
            layer = layer_loader.load_layer(info)
            if layer:
                non_editable_layers.append(layer)

    queue.put(
        TaskResult(
            (layers, non_editable_layers),
            layer_loader.state,
            layer_loader._project_info.data,
        )
    )


def export_layer(
    layer_loader: LayerLoader,
    layer_id: int,
    layer_data: np.ndarray | tuple | None,
    folder: str,
    queue: MPQueue,
) -> None:
    layer_loader.export_layer(layer_id, layer_data, folder)

    queue.put(TaskResult(None))


def fix_project_state(
    layer_loader: LayerLoader, layers_data: list, save: bool, queue: MPQueue
) -> None:
    for layer_id, data in layers_data:
        layer_loader.save_layer_data(layer_id, data)

    result = TaskResult(layer_loader.fix_state(save))
    result.layer_loader_state = layer_loader.state
    queue.put(result)
