from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollBar, QStyle, QStyleOptionSlider

CC = QStyle.ComplexControl
SC = QStyle.SubControl


class ModifiedScrollBar(QScrollBar):
    def _move_to_mouse_position(self, event):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        point = (
            event.position().toPoint() if hasattr(event, "position") else event.pos()
        )
        control = self.style().hitTestComplexControl(CC.CC_ScrollBar, opt, point, self)
        if control not in {SC.SC_ScrollBarAddPage, SC.SC_ScrollBarSubPage}:
            return
        # scroll here
        gr = self.style().subControlRect(
            CC.CC_ScrollBar, opt, SC.SC_ScrollBarGroove, self
        )
        sr = self.style().subControlRect(
            CC.CC_ScrollBar, opt, SC.SC_ScrollBarSlider, self
        )
        if self.orientation() == Qt.Orientation.Horizontal:
            pos = point.x()
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
            if self.layoutDirection() == Qt.LayoutDirection.RightToLeft:
                opt.upsideDown = not opt.upsideDown
        else:
            pos = point.y()
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1
        self.setValue(
            QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                pos - slider_min - slider_length // 2,
                slider_max - slider_min,
                opt.upsideDown,
            )
        )

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            # dragging with the mouse button down should move the slider
            self._move_to_mouse_position(event)
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # clicking the mouse button should move slider to the clicked point
            self._move_to_mouse_position(event)
        return super().mousePressEvent(event)
