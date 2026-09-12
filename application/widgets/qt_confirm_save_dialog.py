from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from application.settings import ApplicationSettings
from utils.qt_translater import Translater
from widgets.qt_custom_button import QtPushButton
from widgets.qt_custom_check_box import QtCheckBox
from widgets.qt_custom_label import QtLabel

if TYPE_CHECKING:
    from PyQt5.QtGui import QCloseEvent


class QtConfirmSaveDialog(QDialog):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        cancel_btn = QtPushButton("not save")
        save_btn = QtPushButton("save")
        icon_label = QWidget()

        self.do_not_ask = QtCheckBox("use autosave")

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowTitle(Translater.instance().get_translation("save project title"))
        cancel_btn.setObjectName("warning_icon_btn")
        icon_label.setObjectName("warning_icon_element")
        cancel_btn.setMinimumSize(80, 30)
        save_btn.setMinimumSize(80, 30)
        save_btn.setDefault(True)

        cancel_btn.clicked.connect(self.reject)
        save_btn.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.addWidget(icon_label)
        layout3 = QVBoxLayout()
        layout3.addWidget(QtLabel("save project text"))
        layout3.addWidget(self.do_not_ask)
        layout2.addLayout(layout3)
        layout4 = QHBoxLayout()
        layout4.addStretch(1)
        layout4.addWidget(cancel_btn)
        layout4.addWidget(save_btn)
        layout.addLayout(layout2)
        layout.addLayout(layout4)
        self.setLayout(layout)

    def accept(self):
        if self.do_not_ask.isChecked():
            settings = ApplicationSettings.instance()
            settings.update({"auto_saving_when_closing": True})
        super().accept()

    def closeEvent(self, event: "QCloseEvent") -> None:
        event.accept()
        self.accept()
