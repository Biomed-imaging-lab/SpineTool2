from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from utils.progress import progress
from widgets.qt_custom_label import QtLabel


class QtLabeledProgressBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, prog: progress = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.progress = prog

        self.qt_progress_bar = QProgressBar()
        self.description_label = QtLabel()
        self.elapsed_label = QtLabel("00:00")
        self.kill_button = QPushButton("Kill", self)
        self.kill_button.clicked.connect(self._confirm_kill)
        self._elapsed_seconds = 0
        self._elapsed_timer = QtCore.QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)
        base_layout = QVBoxLayout()

        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.description_label)
        pbar_layout.addWidget(self.qt_progress_bar)
        pbar_layout.addWidget(self.elapsed_label)
        pbar_layout.addWidget(self.kill_button)
        base_layout.addLayout(pbar_layout)

        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        line.setFixedHeight(1)
        base_layout.addWidget(line)

        self.setLayout(base_layout)

    def _update_elapsed(self):
        self._elapsed_seconds += 1
        hours, remainder = divmod(self._elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            text = f"{minutes:02d}:{seconds:02d}"
        self.elapsed_label.setText(text)

    def _confirm_kill(self):
        answer = QMessageBox.question(
            self,
            "Stop background task",
            "Stop this task and discard its unfinished result?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.kill_button.setEnabled(False)
            self.progress.request_cancel()

    def setRange(self, min_val, max_val):
        self.qt_progress_bar.setRange(min_val, max_val)

    def setValue(self, value):
        self.qt_progress_bar.setValue(value)
        QApplication.processEvents()

    def setDescription(self, value):
        self.description_label.setText(value)
        QApplication.processEvents()

    def _set_value(self, value):
        self.setValue(value)

    def _get_value(self):
        return self.qt_progress_bar.value()

    def _set_description(self, desc):
        self.setDescription(desc)

    def _make_indeterminate(self):
        self.setRange(0, 0)

    def _set_total(self, total):
        self.setRange(0, total)


class QtProgressBarGroup(QWidget):
    """One or more QtLabeledProgressBars with a QFrame line separator at the bottom"""

    def __init__(
        self,
        qt_progress_bar: QtLabeledProgressBar,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        pbr_group_layout = QVBoxLayout()
        pbr_group_layout.addWidget(qt_progress_bar)
        pbr_group_layout.setContentsMargins(0, 0, 0, 0)

        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        line.setFixedHeight(1)
        pbr_group_layout.addWidget(line)

        self.setLayout(pbr_group_layout)
