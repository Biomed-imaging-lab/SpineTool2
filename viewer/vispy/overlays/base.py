from vispy.visuals.transforms import MatrixTransform, STTransform

from viewer.vispy.utils.gl import BLENDING_MODES


class VispyBaseOverlay:
    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__()
        self.overlay = overlay

        self.node = node
        self.node.order = self.overlay.order

        self.overlay.visible_.connect(self._on_visible_change)
        self.overlay.opacity_.connect(self._on_opacity_change)
        self.overlay.blending_.connect(self._on_blending_change)

        if parent is not None:
            self.node.parent = parent

    def _on_visible_change(self):
        self.node.visible = self.overlay.visible

    def _on_opacity_change(self):
        self.node.opacity = self.overlay.opacity

    def _on_blending_change(self):
        self.node.set_gl_state(**BLENDING_MODES[self.overlay.blending])
        self.node.update()

    def reset(self):
        self._on_visible_change()
        self._on_opacity_change()
        self._on_blending_change()

    def close(self):
        self.overlay.visible_.disconnect(self._on_visible_change)
        self.overlay.opacity_.disconnect(self._on_opacity_change)
        self.overlay.blending_.disconnect(self._on_blending_change)
        self.node.transforms = MatrixTransform()
        self.node.parent = None


class VispyCanvasOverlay(VispyBaseOverlay):
    def __init__(self, *, overlay, node, parent=None) -> None:
        super().__init__(overlay=overlay, node=node, parent=None)

        # offsets and size are used to control fine positioning, and will depend
        # on the subclass and visual that needs to be rendered
        self.x_offset = 10
        self.y_offset = 10
        self.x_size = 0
        self.y_size = 0
        self.node.transform = STTransform()
        self.overlay.position_.connect(self._on_position_change)
        self.node.events.parent_change.connect(self._on_parent_change)

    def _on_parent_change(self, event):
        if event.new is not None and self.node.canvas is not None:
            # connect the canvas resize to recalculating the position
            event.new.canvas.events.resize.connect(self._on_position_change)

    def _on_position_change(self, event=None):
        pass

    def reset(self):
        super().reset()
        self._on_position_change()
