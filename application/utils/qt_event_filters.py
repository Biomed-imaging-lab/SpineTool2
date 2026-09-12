"""Qt event filters providing custom handling of events."""

import html
import re

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QWidget

RICH_TEXT_PATTERN = re.compile("<[^\n]+>")


class QtToolTipEventFilter(QObject):
    """
    An event filter that converts all plain-text widget tooltips to rich-text
    tooltips.
    """

    def eventFilter(self, qobject: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ToolTipChange and isinstance(qobject, QWidget):
            tooltip = qobject.toolTip()
            if tooltip and not bool(RICH_TEXT_PATTERN.search(tooltip)):
                qobject.setToolTip(f"<qt>{html.escape(tooltip)}</qt>")
                return True

        return super().eventFilter(qobject, event)
