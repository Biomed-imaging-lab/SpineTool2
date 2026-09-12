from PyQt5.QtCore import QRegExp, pyqtSignal
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QScrollArea, QWidget, QCheckBox

from projects.segmentation.utils.project_info import ProjectInfo
from widgets.qt_custom_double_spin_box import QtDoubleSpinBox
from widgets.qt_custom_label import QtLabel
from widgets.qt_custom_line_edit import QtLineEdit
from widgets.qt_form_layout import QtFormLayout
from widgets.qt_line import QtHLine


class QtProjectInfo(QScrollArea):
    scale_changed_ = pyqtSignal(object)
    renamed_ = pyqtSignal(str)
    checkbox_changed_ = pyqtSignal(str)

    def __init__(self, project_info: ProjectInfo, parent=None):
        QScrollArea.__init__(self, parent)
        self.setMinimumHeight(52)
        self.setMinimumWidth(320)
        self.setMaximumHeight(300)
        self._project_info = project_info
        self._project_name_line = QtLineEdit(self._project_info.name)
        self._project_name_line.setValidator(
            QRegExpValidator(QRegExp("[a-zA-Z0-9_\-]+"))
        )
        self._project_name_line.textChanged.connect(self.rename)
        self._folder_line = QtLineEdit(self._project_info.folder)
        self._folder_line.setReadOnly(True)
        self._image_line = QtLineEdit(self._project_info.original_image)
        self._image_line.setReadOnly(True)
        self._x_scale = QtDoubleSpinBox(self._project_info.displayed_scale[0])
        self._x_scale.editingFinished.connect(self.change_scale)
        self._y_scale = QtDoubleSpinBox(self._project_info.displayed_scale[1])
        self._y_scale.editingFinished.connect(self.change_scale)
        self._z_scale = QtDoubleSpinBox(self._project_info.displayed_scale[2])
        self._z_scale.editingFinished.connect(self.change_scale)

        self._cuda_checkbox = QCheckBox()
        self._cuda_label = QtLabel("cuda")
        self._cuda_checkbox.toggled.connect(self._on_cuda_changed)

        project_info_widget = QWidget()
        form_layout = QtFormLayout()

        form_layout.addRow(QtLabel("project name"), self._project_name_line)
        form_layout.addRow(QtHLine())
        form_layout.addRow(QtLabel("folder"), self._folder_line)
        form_layout.addRow(QtLabel("original image"), self._image_line)
        form_layout.addRow(
            QtLabel("x scale"),
            QtLabel(f"{self._project_info.real_scale[2]:.4f}"),
        )
        form_layout.addRow(
            QtLabel("y scale"),
            QtLabel(f"{self._project_info.real_scale[1]:.4f}"),
        )
        form_layout.addRow(
            QtLabel("z scale"),
            QtLabel(f"{self._project_info.real_scale[0]:.4f}"),
        )
        form_layout.addRow(QtHLine())
        form_layout.addRow(QtLabel("x display scale", {"ru"}), self._x_scale)
        form_layout.addRow(QtLabel("y display scale", {"ru"}), self._y_scale)
        form_layout.addRow(QtLabel("z display scale", {"ru"}), self._z_scale)

        form_layout.addRow(QtHLine())
        form_layout.addRow(self._cuda_label, self._cuda_checkbox)

        project_info_widget.setLayout(form_layout)
        self.setWidget(project_info_widget)
        self.setWidgetResizable(True)

    def toggle_cuda_checkbox(self, checked: bool) -> None:
        self._cuda_checkbox.setVisible(checked)
        self._cuda_label.setVisible(checked)

    def rename(self) -> None:
        self._new_name = self._project_name_line.text()
        self._project_info.name = self._new_name
        old_folder: str = self._project_info.folder
        new_folder = "/".join(old_folder.split("/")[:-1]) + "/" + self._new_name
        self._folder_line.setText(new_folder)
        self.renamed_.emit(self._new_name)

    def change_scale(self) -> None:
        if (
            self._project_info.displayed_scale[0] != self._x_scale.value()
            or self._project_info.displayed_scale[1] != self._y_scale.value()
            or self._project_info.displayed_scale[2] != self._z_scale.value()
        ):
            self._project_info.scale = [
                self._x_scale.value(),
                self._y_scale.value(),
                self._z_scale.value(),
            ]
            self.scale_changed_.emit(self._project_info.scale)

    def _on_cuda_changed(self) -> None:
        if self._cuda_checkbox.isChecked():
            self._project_info.device = "cuda"
        else:
            self._project_info.device = "cpu"
        self.checkbox_changed_.emit(self._project_info.device)
