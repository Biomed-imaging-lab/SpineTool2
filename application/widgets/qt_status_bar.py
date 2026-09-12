"""Status bar widget on the viewer MainWindow"""

from typing import TYPE_CHECKING, Dict, Optional, cast

from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QResizeEvent
from PyQt5.QtWidgets import QLabel, QStatusBar, QWidget
from superqt import QElidingLabel

from application.widgets.qt_activity_dialog import ActivityToggleItem
from utils.qt_translater import Translater

if TYPE_CHECKING:
    from application.widgets.qt_main_window import QtMainWindow


class QtStatusBar(QStatusBar):
    activity_dock_toggled = pyqtSignal()

    def __init__(self, parent: "QtMainWindow") -> None:
        super().__init__(parent=parent)

        self._status_text = None
        self._status = QElidingLabel("")
        self._status.setContentsMargins(0, 0, 0, 0)

        self._source_text = None
        self._source = QElidingLabel("")
        self._source.setObjectName("source")
        self._source.setElideMode(Qt.TextElideMode.ElideMiddle)
        self._source.setMinimumSize(100, 16)
        self._source.setContentsMargins(0, 0, 0, 0)

        self._additional_info_text = None
        self._additional_info = QElidingLabel("")
        self._additional_info.setObjectName("additional info")
        self._additional_info.setMinimumSize(100, 16)
        self._additional_info.setContentsMargins(0, 0, 0, 0)

        self._help_text = None
        self._help = QElidingLabel("")
        self._help.setObjectName("help status")
        self._help.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        main_widget = QtStatusBarWidget(
            self._status,
            self._source,
            self._additional_info,
            self._help,
        )
        self.addWidget(main_widget, 1)

        self._activity_item = ActivityToggleItem()
        self._activity_item._activityBtn.clicked.connect(self._toggle_activity_dock)
        parent._activity_dialog._toggleButton = self._activity_item
        self.addPermanentWidget(self._activity_item)
        Translater.instance().language_changed_signal.connect(self.language_changed)

    def setHelpText(self, text: str) -> None:
        self._help_text = text
        self._help.setText(Translater.instance().get_translation(text))

    def setStatusText(self, status: Dict[str, str]) -> None:
        translater = Translater.instance()

        self._status_text = ""
        if "text" in status:
            self._status_text = status["text"]
        self._status.setText(translater.get_translation(self._status_text))

        self._source_text = ""
        if "source" in status:
            self._source.setVisible(True)
            self._source_text = status["source"]
        else:
            self._source.setVisible(False)
        self._source.setText(translater.get_translation(self._source_text))

        self._additional_info_text = ""
        if "additional_info" in status:
            self._additional_info.setVisible(True)
            self._additional_info_text = status["additional_info"]
        else:
            self._additional_info.setVisible(False)
        self._additional_info.setText(
            translater.get_translation(self._additional_info_text)
        )

    def language_changed(self):
        if self._help_text:
            self.setHelpText(self._help_text)
        status = {}
        if self._status_text:
            status["text"] = self._status_text
        if self._source_text:
            status["source"] = self._source_text
        if self._additional_info_text:
            status["additional_info"] = self._additional_info_text
        self.setStatusText(status)

    def _toggle_activity_dock(self, visible: Optional[bool] = None):
        par = cast("QtMainWindow", self.parent())
        if visible is None:
            visible = not par._activity_dialog.isVisible()
        if visible:
            par._activity_dialog.show()
            par._activity_dialog.raise_()
            self._activity_item._activityBtn.setArrowType(Qt.ArrowType.DownArrow)
        else:
            par._activity_dialog.hide()
            self._activity_item._activityBtn.setArrowType(Qt.ArrowType.UpArrow)
        self.activity_dock_toggled.emit()


class QtStatusBarWidget(QWidget):
    def __init__(
        self,
        status_label: QLabel,
        source_label: QLabel,
        additional_info_label: QLabel,
        help_label: QLabel,
        parent: QWidget = None,
    ) -> None:
        super().__init__(parent=parent)
        self._status_label = status_label
        self._source_label = source_label
        self._additional_info_label = additional_info_label
        self._help_label = help_label

        self._status_label.setParent(self)
        self._source_label.setParent(self)
        self._additional_info_label.setParent(self)
        self._help_label.setParent(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.do_layout()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.LayoutRequest:
            self.do_layout()
        return super().event(event)

    @staticmethod
    def _calc_width(fm: QFontMetrics, label: QLabel) -> int:
        # magical nuber +2 is from superqt code
        # magical number +12 is from experiments
        # Adding this values is required to avoid the text to be elided
        # if there is enough space to show it.
        return (
            (fm.boundingRect(label.text()).width() + label.margin() * 2 + 2 + 12)
            if label.isVisible()
            else 0
        )

    def do_layout(self):
        width = self.width()
        height = self.height()

        fm = QFontMetrics(self._status_label.font())

        status_width = self._calc_width(fm, self._status_label)
        layer_width = self._calc_width(fm, self._source_label)
        coordinates_width = self._calc_width(fm, self._additional_info_label)

        base_width = status_width + layer_width + coordinates_width

        help_width = max(0, width - base_width)

        if coordinates_width:
            help_width = 0

        if base_width > width:
            self._help_label.setVisible(False)
            layer_width = max(
                int((layer_width / base_width) * layer_width),
                min(self._source_label.minimumWidth(), layer_width),
            )
            coordinates_width = base_width - status_width - layer_width

        else:
            self._help_label.setVisible(True)

        self._status_label.setGeometry(0, 0, status_width, height)
        shift = status_width
        self._source_label.setGeometry(shift, 0, layer_width, height)
        shift += layer_width
        self._additional_info_label.setGeometry(shift, 0, coordinates_width, height)
        shift += coordinates_width
        self._help_label.setGeometry(shift, 0, help_width, height)
