import re
from typing import Optional, Union

import numpy as np
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import (
    QColorDialog,
    QCompleter,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from vispy.color import get_color_dict

from utils.colormaps.colormap_utils import ColorType
from utils.colormaps.standardize_color import hex_to_name, rgb_to_hex, transform_color
from utils.qt_translater import Translater

rgba_regex = re.compile(r"\(?([\d.]+),\s*([\d.]+),\s*([\d.]+),?\s*([\d.]+)?\)?")

TRANSPARENT = np.array([0, 0, 0, 0], np.float32)
AnyColorType = Union[ColorType, QColor]


class QColorSwatchEdit(QWidget):
    color_changed = pyqtSignal(object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial_color: Optional[AnyColorType] = None,
        tooltip: Optional[str] = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setObjectName("QColorSwatchEdit")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setLayout(layout)

        self.line_edit = QColorLineEdit(self)
        self.line_edit.editingFinished.connect(self._on_line_edit_edited)

        self.color_swatch = QColorSwatch(parent, tooltip=tooltip)
        self.color_swatch.color_changed.connect(self._on_swatch_changed)
        self.setColor = self.color_swatch.setColor
        if initial_color is not None:
            self.setColor(initial_color)

        layout.addWidget(self.color_swatch)
        layout.addWidget(self.line_edit)

    @property
    def color(self):
        return self.color_swatch.color

    def _on_line_edit_edited(self):
        text = self.line_edit.text()
        rgb_match = rgba_regex.match(text)
        if rgb_match:
            text = [float(x) for x in rgb_match.groups() if x]
        self.color_swatch.setColor(text)

    def _on_swatch_changed(self, color: np.ndarray):
        self.line_edit.setText(color)
        self.color_changed.emit(color)


class QColorSwatch(QFrame):
    color_changed = pyqtSignal(object)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        tooltip: Optional[str] = None,
        initial_color: Optional[ColorType] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("colorSwatch")
        key = tooltip or "click to set color"
        self.setToolTip(Translater.instance().get_translation(key))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.color_changed.connect(self._update_swatch_style)
        self._color: np.ndarray = TRANSPARENT
        if initial_color is not None:
            self.setColor(initial_color)
        self.popup = None

    @property
    def color(self):
        return self._color

    def _update_swatch_style(self, color: np.ndarray) -> None:
        rgba = f'rgba({",".join(str(int(x*255)) for x in  self._color)})'
        self.setStyleSheet("#colorSwatch {background-color: " + rgba + ";}")

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.popup is None:
            initial = QColor(*(255 * self._color).astype("int"))
            self.popup = QColorPopup(self.parent(), initial)
            self.popup.colorSelected.connect(self.setColor)
            self.popup.closed.connect(self.resetPopup)
            self.popup.show()

    def resetPopup(self) -> None:
        if self.popup:
            self.popup.deleteLater()
            self.popup = None

    def setColor(self, color: AnyColorType) -> None:
        if self.popup:
            self.popup.deleteLater()
            self.popup = None
        if isinstance(color, QColor):
            _color = (np.array(color.getRgb()) / 255).astype(np.float32)
        else:
            try:
                _color = transform_color(color)[0]
            except ValueError:
                return self.color_changed.emit(self._color)
        emit = np.any(self._color != _color)
        self._color = _color
        if emit or np.all(_color == TRANSPARENT):
            self.color_changed.emit(_color)
            return None
        return None


class QColorLineEdit(QLineEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._compl = QCompleter([*get_color_dict(), "transparent"])
        self._compl.setCompletionMode(QCompleter.InlineCompletion)
        self.setCompleter(self._compl)
        self.setTextMargins(2, 2, 2, 2)

    def setText(self, color: ColorType):
        _rgb = transform_color(color)[0]
        _hex = rgb_to_hex(_rgb)[0]
        super().setText(hex_to_name.get(_hex, _hex))


class CustomColorDialog(QColorDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName("CustomColorDialog")

    def keyPressEvent(self, event: QEvent):
        event.ignore()


class QColorPopup(QDialog):
    currentColorChanged = pyqtSignal(object)
    colorSelected = pyqtSignal(object)
    closed = pyqtSignal()

    def __init__(
        self, parent: QWidget = None, initial_color: AnyColorType = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("QtColorPopup")
        self.color_dialog = CustomColorDialog(parent)

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.color_dialog.setOptions(
            QColorDialog.DontUseNativeDialog | QColorDialog.ShowAlphaChannel
        )
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.color_dialog)

        self.color_dialog.currentColorChanged.connect(self.currentColorChanged.emit)
        self.color_dialog.colorSelected.connect(self._on_color_selected)
        self.color_dialog.rejected.connect(self._on_rejected)
        self.color_dialog.setCurrentColor(QColor(initial_color))

    def _on_color_selected(self, color: QColor):
        self.colorSelected.emit(color)
        self.close()

    def _on_rejected(self):
        self.closed.emit()
        self.close()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self.color_dialog.accept()
        if event.key() == Qt.Key.Key_Escape:
            return self.color_dialog.reject()
        self.color_dialog.keyPressEvent(event)
        return None
