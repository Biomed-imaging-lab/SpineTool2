import os
from datetime import datetime
from functools import partial
from json import dump, load

import numpy as np

from projects.segmentation.utils.constants import (
    ADDITIONAL_COLORMAPS_PATH,
    COLORMAPS_PATH,
    Types,
)
from projects.segmentation.utils.project_info import ProjectInfo
from utils.colormaps.colormap import CyclicLabelColormap
from viewer.layers.base.base import Layer
from viewer.layers.image.image import Image
from viewer.layers.labels.labels import Labels
from viewer.layers.points.points import Points
from viewer.layers.surface.surface import Surface


class LayerParametersTracker:
    def __init__(self, project_info: ProjectInfo):
        self._project_info = project_info

    def _on_layer_parameters_changed(self, layer: Layer) -> None:
        self._project_info.layers_parameters[layer._id].name = layer._name
        file = self._project_info.layers_parameters[layer._id].parameters.get(
            "colormap", ""
        )
        try:
            os.remove(self._project_info.folder + file)
        except:
            pass
        file = self._project_info.layers_parameters[layer._id].parameters.get(
            "connected_components_colormap", ""
        )
        try:
            os.remove(self._project_info.folder + file)
        except:
            pass
        self._project_info.layers_parameters[layer._id].parameters = (
            self.get_parameters(layer)
        )

    def get_parameters(self, layer: Layer) -> dict:
        layer_type = layer.metadata["type"]
        parameters = {}
        if (
            layer_type == Types.IMAGE or layer_type == Types.BACKGROUND_IMAGE
        ) and isinstance(layer, Image):
            parameters["blending"] = layer.blending
            parameters["colormap"] = layer.colormap.name
            parameters["contrast_limits"] = layer.contrast_limits
            parameters["gamma"] = layer.gamma
            parameters["interpolation2d"] = layer.interpolation2d
            parameters["interpolation3d"] = layer.interpolation3d
            parameters["iso_threshold"] = layer.iso_threshold
            parameters["opacity"] = layer.opacity
            parameters["visible"] = layer.visible
            parameters["max_projection"] = layer.max_projection
        elif (
            layer_type == Types.MASK
            or layer_type == Types.BINARIZATION
            or layer_type == Types.NECKS
            or layer_type == Types.VOXEL_MESH_SEGMENTATION
        ) and isinstance(layer, Labels):
            parameters["blending"] = layer.blending
            parameters["colormap"] = self._save_colormap(layer)
            parameters["labels"] = layer.labels
            parameters["connected_components_colormap"] = self._save_colormap(
                layer, True
            )
            parameters["brush_settings"] = layer.brush_settings.as_dict()
            parameters["opacity"] = layer.opacity
            parameters["show_selected_label"] = layer.show_selected_label
            parameters["visible"] = layer.visible
            parameters["fill_3d"] = layer.fill_3d
            parameters["preserve_background"] = layer.preserve_background
        elif layer_type == Types.POINTS and isinstance(layer, Points):
            parameters["blending"] = layer.blending
            parameters["out_of_slice_display"] = layer.out_of_slice_display
            parameters["size"] = layer.size
            parameters["opacity"] = layer.opacity
            parameters["symbol"] = layer.symbol.value
            parameters["face_color"] = layer.face_color.tolist()
            parameters["edge_color"] = layer.edge_color.tolist()
            parameters["visible"] = layer.visible
        elif (
            layer_type == Types.POLYGON_MESH
            or layer_type == Types.POLYGON_MESH_SEGMENTATION
            or layer_type == Types.FINAL_SEGMENTATION
        ) and isinstance(layer, Surface):
            parameters["blending"] = layer.blending
            parameters["colormap"] = layer.colormap.name
            parameters["contrast_limits"] = layer.contrast_limits
            parameters["gamma"] = layer.gamma
            parameters["opacity"] = layer.opacity
            parameters["shading"] = layer.shading
            parameters["visible"] = layer.visible

        layer_parameters = self._project_info.layers_parameters.get(layer._id)
        if layer_parameters and "scale" in layer_parameters.parameters:
            parameters["scale"] = np.asarray(layer.scale).astype(float).tolist()
        return parameters

    def track_parameters(self, layer: Layer) -> None:
        if isinstance(layer, Image):
            layer.name_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.blending_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.colormap_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.contrast_limits_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.gamma_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.interpolation2d_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.interpolation3d_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.iso_threshold_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.opacity_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.visible_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.max_projection_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
        elif isinstance(layer, Labels):
            layer.name_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.blending_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.colormap_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.labels_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.opacity_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.brush_settings_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.show_selected_label_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.visible_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.fill_3d_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.preserve_background_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
        elif isinstance(layer, Points):
            layer.name_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.blending_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.size_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.opacity_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.face_color_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.edge_color_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.symbol_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.out_of_slice_display_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.visible_.connect(partial(self._on_layer_parameters_changed, layer))
        elif isinstance(layer, Surface):
            layer.name_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.blending_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.colormap_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.contrast_limits_.connect(
                partial(self._on_layer_parameters_changed, layer)
            )
            layer.gamma_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.opacity_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.shading_.connect(partial(self._on_layer_parameters_changed, layer))
            layer.visible_.connect(partial(self._on_layer_parameters_changed, layer))

    def _save_colormap(
        self, layer: Labels, connected_components_colormap: bool = False
    ) -> str:
        if not os.path.exists(self._project_info.folder + COLORMAPS_PATH):
            os.mkdir(self._project_info.folder + COLORMAPS_PATH)
        if not os.path.exists(self._project_info.folder + ADDITIONAL_COLORMAPS_PATH):
            os.mkdir(self._project_info.folder + ADDITIONAL_COLORMAPS_PATH)
        if not connected_components_colormap:
            colormap_desc = {
                "colors": layer._colormap.colors.tolist(),
                "controls": layer._colormap.controls.tolist(),
            }
            filename = (
                f"/colormap_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".json"
            )
            if "link" not in layer.metadata:
                filename = COLORMAPS_PATH + filename
            else:
                filename = ADDITIONAL_COLORMAPS_PATH + filename
        else:
            colormap_desc = {
                "colors": layer._connected_components_colormap.colors.tolist(),
                "controls": layer._connected_components_colormap.controls.tolist(),
            }
            filename = (
                f"/connected_components_colormap_"
                + str(datetime.now())
                .replace(".", "_")
                .replace(" ", "_")
                .replace(":", "_")
                + ".json"
            )
            if "link" not in layer.metadata:
                filename = COLORMAPS_PATH + filename
            else:
                filename = ADDITIONAL_COLORMAPS_PATH + filename
        file = open(self._project_info.folder + filename, "w")
        dump(colormap_desc, file)
        file.close()
        return filename

    def load_colormap(self, path=None) -> CyclicLabelColormap | None:
        if not path:
            return None
        try:
            file = open(self._project_info.folder + path, "r")
            colormap_desc = load(file)
            file.close()
            return CyclicLabelColormap(
                controls=np.array(colormap_desc["controls"]),
                colors=np.array(colormap_desc["colors"]),
            )
        except:
            return None

    def link_layers(self, layer1: Layer, layer2: Layer) -> None:
        if layer1.__class__ != layer2.__class__:
            return

        if layer1.metadata.get("link", None):
            return

        if isinstance(layer1, Surface):
            layer1._faces = layer2._faces
            layer1._vertices = layer2._vertices
            layer1._vertex_values = layer2._vertex_values.copy()
            file = open(
                self._project_info.folder
                + self._project_info.layers_parameters[layer2._id].tmp_additional_files[
                    "spines"
                ],
                "r",
            )
            spines = load(file)
            if layer2.metadata["type"] == Types.FINAL_SEGMENTATION:
                for spine in spines["spines"].values():
                    for index in spine["indices"]:
                        layer1._vertex_values[index] = 1.0
            else:
                deleted_spines = self._project_info.layers_parameters[
                    layer2._id
                ].tmp_deleted_spines
                for spine in spines["spines"].values():
                    value = 0.0 if spine["pos"] in deleted_spines else 1.0
                    for index in spine["indices"]:
                        layer1._vertex_values[index] = value
        else:
            layer1._data = layer2._data
            if isinstance(layer1, Labels):
                layer1._connected_components = layer2._connected_components
            elif isinstance(layer1, Points):
                layer1._shown = layer2._shown
            elif isinstance(layer1, Image):
                layer1._max_projection_images = layer2._max_projection_images
        layer1.editable = False
        layer1.refresh()
        layer1.metadata["link"] = layer2
        layer2.name_.connect(layer1.set_name)
        if isinstance(layer1, Points):
            layer1.metadata["connection"] = layer2.set_data_.connect(
                partial(self._update_linked_points, layer1, layer2)
            )
        elif isinstance(layer1, Labels):
            layer1.metadata["connection"] = layer2.set_data_.connect(
                partial(self._update_linked_labels, layer1, layer2)
            )
        if isinstance(layer1, Surface):
            layer1.metadata["connection"] = layer2.data_.connect(
                partial(self._update_linked_surface, layer1, layer2)
            )

    def _update_linked_labels(self, layer1: Labels, layer2: Labels):
        layer1._data = layer2._data
        layer1._connected_components = layer2._connected_components
        layer1.refresh()

    def _update_linked_points(self, layer1: Points, layer2: Points):
        layer1._data = layer2._data
        layer1._shown = layer2._shown
        layer1.refresh()

    def _update_linked_surface(self, layer1: Surface, layer2: Surface):
        layer1._faces = layer2._faces
        layer1._vertices = layer2._vertices
        layer1._vertex_values = layer2._vertex_values.copy()
        file = open(
            self._project_info.folder
            + self._project_info.layers_parameters[layer2._id].tmp_additional_files[
                "spines"
            ],
            "r",
        )
        spines = load(file)
        if layer2.metadata["type"] == Types.FINAL_SEGMENTATION:
            for spine in spines["spines"].values():
                for index in spine["indices"]:
                    layer1._vertex_values[index] = 1.0
        else:
            deleted_spines = self._project_info.layers_parameters[
                layer2._id
            ].tmp_deleted_spines
            for spine in spines["spines"].values():
                value = 0.0 if spine["pos"] in deleted_spines else 1.0
                for index in spine["indices"]:
                    layer1._vertex_values[index] = value
        layer1.refresh()

    def unlink_layers(self, layer: Layer) -> None:
        link: Layer = layer.metadata.get("link", None)
        if link:
            link.name_.disconnect(layer.set_name)
            layer.metadata["link"] = None
            if "connection" in layer.metadata:
                link.set_data_.disconnect(layer.metadata["connection"])
                del layer.metadata["connection"]
