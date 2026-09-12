from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QLineEdit


class QtLineEdit(QLineEdit):
    def __init__(self, text="", parent=None) -> None:
        super().__init__(text, parent)
        self._button_pressed = False
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._button_pressed = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._button_pressed = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() // 8
        if delta > 0:
            self.cursorForward(self._button_pressed, delta)
        else:
            self.cursorBackward(self._button_pressed, -delta)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.focusNextChild()
            event.accept()
            return
        return super().keyPressEvent(event)
