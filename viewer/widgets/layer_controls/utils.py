from typing import Iterable

from PyQt5.QtWidgets import QGraphicsOpacityEffect, QWidget


def set_widgets_enabled_with_opacity(
    parent: QWidget, widgets: Iterable[QWidget], enabled: bool
):
    """Set enabled state on some widgets. If not enabled, decrease opacity."""
    for widget in widgets:
        op = QGraphicsOpacityEffect(parent)
        op.setOpacity(0.5)
        op.setEnabled(not enabled)
        widget.setEnabled(enabled)
        widget.setGraphicsEffect(op)
