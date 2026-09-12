from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from widgets.qt_custom_label import QtLabel


def add_model_path_row(
    layout,
    params: dict,
    parent=None,
    key: str = "neural_model_path",
) -> None:
    model_path = params.get(key, "")
    if not model_path:
        return

    model_label = QLabel(Path(model_path).name, parent)
    model_label.setToolTip(model_path)
    model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    model_label.setWordWrap(True)
    layout.addRow(QtLabel("model weights", {"en", "ru"}, parent), model_label)
