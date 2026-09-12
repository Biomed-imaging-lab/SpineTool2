from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from viewer.components.layer_list.layerlist import LayerList
from viewer.widgets.layer_list.qt_layer_list import QtLayerList
from widgets.qt_custom_button import QtPushButton
from widgets.qt_form_layout import QtFormLayout


class QtNextStageListWidget(QWidget):
    def __init__(self, next_stage_list: LayerList, parent=None):
        QWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.next_stage_list = QtLayerList(next_stage_list, True, True)
        self.create_new_button = QtPushButton()
        self.run_options_widget = None

        layout = QtFormLayout()
        layout.setContentsMargins(3, 4, 5, 6)
        self.setLayout(layout)
        layout1 = QVBoxLayout()
        layout1.setContentsMargins(5, 0, 3, 0)
        layout1.addWidget(self.create_new_button)
        layout.addRow(layout1)
        layout.addWidget(self.next_stage_list)

    def set_run_options_widget(self, widget) -> None:
        if self.run_options_widget is not None:
            self.layout().removeWidget(self.run_options_widget)
            self.run_options_widget.setParent(None)
        self.run_options_widget = widget
        self.layout().insertRow(0, widget)

    def show_run_options(self, visible: bool) -> None:
        if self.run_options_widget is not None:
            self.run_options_widget.setVisible(visible)


class QtStageListWidget(QWidget):
    def __init__(
        self, stage_list: LayerList, dragAndDrop: bool, editable: bool, parent=None
    ):
        QWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.list = QtLayerList(stage_list, dragAndDrop, editable)

        layout = QtFormLayout()
        layout.setContentsMargins(3, 4, 5, 6)
        self.setLayout(layout)
        layout.addWidget(self.list)
