from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from application.utils.constants import (
    OPEN_PROJECT_SHORTCUT_TEXT,
    SHOW_ALL_SHORTCUTS_SHORTCUT_TEXT,
)
from widgets.qt_custom_label import QtLabel


class QtWelcomeLabel(QtLabel):
    """Labels used for main message in welcome page."""


class QtShortcutLabel(QtLabel):
    """Labels used for displaying shortcut information in welcome page."""


class QtWelcomeWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # Create colored icon using theme
        self._image = QLabel()
        self._image.setObjectName("logo_silhouette")
        self._image.setMinimumSize(300, 300)
        self._label = QtWelcomeLabel("welcome text")

        # Widget setup
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout
        text_layout = QVBoxLayout()
        text_layout.addWidget(self._label)

        shortcut_layout = QFormLayout()
        shortcut_layout.addRow(
            QtShortcutLabel(OPEN_PROJECT_SHORTCUT_TEXT),
            QtShortcutLabel("open project"),
        )
        shortcut_layout.addRow(
            QtShortcutLabel(SHOW_ALL_SHORTCUTS_SHORTCUT_TEXT),
            QtShortcutLabel("show all key bindings"),
        )
        shortcut_layout.setSpacing(0)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.setSpacing(30)
        layout.addWidget(self._image)
        layout.addLayout(text_layout)
        layout.addLayout(shortcut_layout)
        layout.addStretch()

        self.setLayout(layout)

    def minimumSizeHint(self):
        """
        Overwrite minimum size to allow creating small viewer instance
        """
        return QSize(100, 100)

    def paintEvent(self, event):
        """Override Qt method.

        Parameters
        ----------
        event : PyQt5.QtCore.QEvent
            Event from the Qt context.
        """
        option = QStyleOption()
        option.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, p, self)


class QtCentralWidget(QWidget):
    resized = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())

    def resizeEvent(self, event):
        self.resized.emit()
        return super().resizeEvent(event)
