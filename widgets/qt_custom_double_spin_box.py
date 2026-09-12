from PyQt5.QtCore import QEvent, QLocale, QObject, Qt
from PyQt5.QtWidgets import QDoubleSpinBox


class QtEventFilter(QObject):
    def eventFilter(self, target, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            super().eventFilter(target, event)
            target.selectAll()
            return True
        return super().eventFilter(target, event)


class QtDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, value, decimals=4, min_value=0.0001, parent=None) -> None:
        super().__init__(parent)
        self.setDecimals(decimals)
        self.setValue(value)
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.setMinimum(min_value)
        self.setLocale(QLocale("C"))
        self.lineEdit().installEventFilter(QtEventFilter(self))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def wheelEvent(self, event):
        event.ignore()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.lineEdit().selectAll()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.focusNextChild()
            event.accept()
            return
        return super().keyPressEvent(event)
