from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from application.settings import ApplicationSettings
from application.widgets.qt_progress_bar import QtLabeledProgressBar, QtProgressBarGroup
from utils.progress import progress
from widgets.qt_custom_button import QtToolButton
from widgets.qt_custom_label import QtLabel


class ActivityToggleItem(QWidget):
    """Toggle button for Activity Dialog.

    A progress indicator is displayed when there are active progress
    bars.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())

        self._activityBtn = QtToolButton("activity")
        self._activityBtn.setObjectName("QtActivityButton")
        self._activityBtn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._activityBtn.setArrowType(Qt.ArrowType.UpArrow)
        self._activityBtn.setIconSize(QSize(11, 11))
        self._activityBtn.setCheckable(True)

        self._inProgressIndicator = QtLabel("in progress")
        sp = self._inProgressIndicator.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self._inProgressIndicator.setSizePolicy(sp)
        load_gif = str(ApplicationSettings.RESOURCES_PATH + "/loading.gif")
        mov = QMovie(load_gif)
        mov.setScaledSize(QSize(18, 18))
        self._inProgressIndicator.setMovie(mov)
        self._inProgressIndicator.hide()

        self.layout().addWidget(self._inProgressIndicator)
        self.layout().addWidget(self._activityBtn)
        self.layout().setContentsMargins(0, 0, 0, 0)


class QtActivityDialog(QDialog):
    MIN_WIDTH = 250
    MIN_HEIGHT = 185

    def __init__(self, parent=None, toggle_button=None) -> None:
        super().__init__(parent)
        self._toggleButton = toggle_button

        self.setObjectName("Activity")
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setMaximumHeight(self.MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.setWindowFlags(
            Qt.WindowType.SubWindow | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setModal(False)

        opacityEffect = QGraphicsOpacityEffect(self)
        opacityEffect.setOpacity(0.8)
        self.setGraphicsEffect(opacityEffect)

        self._baseWidget = QWidget()

        self._activityLayout = QVBoxLayout()
        self._activityLayout.addStretch()
        self._baseWidget.setLayout(self._activityLayout)
        self._baseWidget.layout().setContentsMargins(0, 0, 0, 0)

        self._scrollArea = QScrollArea()
        self._scrollArea.setWidgetResizable(True)
        self._scrollArea.setWidget(self._baseWidget)

        self._titleBar = QLabel()
        self._titleBar.setMinimumHeight(30)

        title = QtLabel("activity", parent=self)
        title.setObjectName("QtCustomTitleLabel")
        title.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        )
        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        titleLayout = QHBoxLayout()
        titleLayout.setSpacing(4)
        titleLayout.setContentsMargins(8, 1, 8, 0)
        line.setFixedHeight(1)
        titleLayout.addWidget(line)
        titleLayout.addWidget(title)
        self._titleBar.setLayout(titleLayout)

        self._baseLayout = QVBoxLayout()
        self._baseLayout.addWidget(self._titleBar)
        self._baseLayout.addWidget(self._scrollArea)
        self.setLayout(self._baseLayout)
        self.resize(520, self.MIN_HEIGHT)
        self.move_to_bottom_right()

        progress.emitter.added.connect(self.make_new_pbar)
        progress.emitter.removed.connect(self.close_progress_bar)

    def make_new_pbar(self, prog: progress):
        # make and add progress bar
        pbar = QtLabeledProgressBar(prog=prog)
        self.add_progress_bar(pbar)

        # connect progress object events to updating progress bar
        prog.value.connect(pbar._set_value)
        prog.description.connect(pbar._set_description)
        prog.overflow.connect(pbar._make_indeterminate)
        prog.total_.connect(pbar._set_total)

        # connect pbar close method if we're closed
        self.destroyed.connect(prog.close)

        # set its range etc. based on progress object
        if prog.total is not None:
            pbar.setRange(prog.n, prog.total)
            pbar.setValue(prog.n)
        else:
            pbar.setRange(0, 0)
            prog.total = 0
        pbar.setDescription(prog.desc)

    def add_progress_bar(self, pbar):
        self._activityLayout.addWidget(pbar)

        # show progress indicator and start gif
        if self._toggleButton._inProgressIndicator.movie():
            self._toggleButton._inProgressIndicator.movie().start()
        self._toggleButton._inProgressIndicator.show()
        pbar.destroyed.connect(self.maybe_hide_progress_indicator)
        QApplication.processEvents()

    def get_pbar_from_prog(self, prog):
        if pbars := self._baseWidget.findChildren(QtLabeledProgressBar):
            for potential_parent in pbars:
                if potential_parent.progress is prog:
                    return potential_parent
        return None

    def close_progress_bar(self, prog: progress):
        current_pbar = self.get_pbar_from_prog(prog)
        if not current_pbar:
            return
        parent_widget = current_pbar.parent()
        current_pbar.close()
        current_pbar.deleteLater()
        if isinstance(parent_widget, QtProgressBarGroup):
            pbar_children = [
                child
                for child in parent_widget.children()
                if isinstance(child, QtLabeledProgressBar)
            ]
            # only close group if it has no visible progress bars
            if not any(child.isVisible() for child in pbar_children):
                parent_widget.close()

    def move_to_bottom_right(self, offset=(8, 8)):
        if not self.parent():
            return
        sz = self.parent().size() - self.size() - QSize(*offset)
        self.move(QPoint(sz.width(), sz.height()))

    def maybe_hide_progress_indicator(self):
        pbars = self._baseWidget.findChildren(QtLabeledProgressBar)
        pbar_groups = self._baseWidget.findChildren(QtProgressBarGroup)

        progress_visible = any(pbar.isVisible() for pbar in pbars)
        progress_group_visible = any(
            pbar_group.isVisible() for pbar_group in pbar_groups
        )
        if not progress_visible and not progress_group_visible:
            if self._toggleButton._inProgressIndicator.movie():
                self._toggleButton._inProgressIndicator.movie().stop()
            self._toggleButton._inProgressIndicator.hide()
