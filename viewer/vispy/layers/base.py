from abc import ABC, abstractmethod

import numpy as np
from vispy.visuals.transforms import MatrixTransform

from viewer.layers.base.base import Layer
from viewer.vispy.utils.gl import BLENDING_MODES, get_max_texture_sizes


class VispyBaseLayer(ABC):
    def __init__(self, layer: Layer, node) -> None:
        super().__init__()

        self.layer = layer
        self._array_like = False
        self.node = node
        self.first_visible = False
        self.overlays = {}

        (
            self.MAX_TEXTURE_SIZE_2D,
            self.MAX_TEXTURE_SIZE_3D,
        ) = get_max_texture_sizes()

        self.layer.set_data_.connect(self._on_data_change)
        self.layer.visible_.connect(self._on_visible_change)
        self.layer.opacity_.connect(self._on_opacity_change)
        self.layer.blending_.connect(self._on_blending_change)
        self.layer.scale_.connect(self._on_matrix_change)

    @property
    def _master_transform(self):
        # whenever a new parent is set, the transform is reset
        # to a NullTransform so we reset it here
        if not isinstance(self.node.transform, MatrixTransform):
            self.node.transform = MatrixTransform()

        return self.node.transform

    @property
    def scale(self):
        matrix = self._master_transform.matrix[:-1, :-1]
        _, upper_tri = np.linalg.qr(matrix)
        return np.diag(upper_tri).copy()

    @property
    def order(self):
        return self.node.order

    @order.setter
    def order(self, order):
        self.node.order = order
        self._on_blending_change()

    @abstractmethod
    def _on_data_change(self):
        raise NotImplementedError

    def _on_visible_change(self):
        self.node.visible = self.layer.visible

    def _on_opacity_change(self):
        self.node.opacity = self.layer.opacity

    def _on_blending_change(self):
        blending = self.layer.blending
        blending_kwargs = BLENDING_MODES[blending].copy()

        if self.first_visible:
            # if the first layer, then we should blend differently
            # the goal is to prevent pathological blending with canvas
            # for minimum, use the src color, ignore alpha & canvas
            if blending == "minimum":
                src_color_blending = "one"
                dst_color_blending = "zero"
            # for additive, use the src alpha and blend to black
            elif blending == "additive":
                src_color_blending = "src_alpha"
                dst_color_blending = "zero"
            # for all others, use translucent blending
            else:
                src_color_blending = "src_alpha"
                dst_color_blending = "one_minus_src_alpha"
            blending_kwargs = {
                "depth_test": blending_kwargs["depth_test"],
                "cull_face": False,
                "blend": True,
                "blend_func": (
                    src_color_blending,
                    dst_color_blending,
                    "one",
                    "one",
                ),
                "blend_equation": "func_add",
            }

        self.node.set_gl_state(**blending_kwargs)
        self.node.update()

    def _on_matrix_change(self):
        scale = self.layer._scale.set_slice(self.layer._slice_input.displayed)
        matrix = np.diag(scale.scale[::-1])

        # Embed in the top left corner of a 4x4 affine matrix
        affine_matrix = np.eye(4)
        affine_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

        if self._array_like and self.layer._slice_input.ndisplay == 2:
            # Perform pixel offset to shift origin from top left corner
            # of pixel to center of pixel.
            # Note this offset is only required for array like data in
            # 2D.
            offset = -matrix @ np.ones(matrix.shape[1]) / 2
            # Convert NumPy axis ordering to VisPy axis ordering
            # and embed in full affine matrix
            affine_offset = np.eye(4)
            affine_offset[-1, : len(offset)] = offset[::-1]
            affine_matrix = affine_matrix @ affine_offset
        self._master_transform.matrix = affine_matrix

        child_matrix = np.eye(4)
        for child in self.node.children:
            child.transform.matrix = child_matrix

    def reset(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()
        self._on_matrix_change()

    def close(self):
        self.node.transforms = MatrixTransform()
        self.node.parent = None
