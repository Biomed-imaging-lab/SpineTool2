from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QWidget


class QtViewerCentralWidget(QWidget):
    resized = pyqtSignal()
    leave = pyqtSignal()
    enter = pyqtSignal()

    def __init__(self, parent, widget) -> None:
        super().__init__(parent)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(widget)

    def resizeEvent(self, event):
        self.resized.emit()
        return super().resizeEvent(event)

    def enterEvent(self, event):
        self.enter.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.leave.emit()
        super().leaveEvent(event)
