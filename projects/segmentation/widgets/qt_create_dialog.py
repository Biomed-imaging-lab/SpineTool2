import os

from PyQt5.QtCore import QRegExp, Qt, pyqtSignal
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QComboBox

import projects.segmentation.utils.constants as constants
from projects.segmentation.utils.history import (
    get_open_history_files,
    get_open_history_folders,
    update_open_history_files,
    update_open_history_folders,
)
from utils.notifications import show_warning
from utils.qt_translater import Translater
from widgets.qt_custom_button import QtPushButton
from widgets.qt_custom_double_spin_box import QtDoubleSpinBox
from widgets.qt_custom_label import QtLabel
from widgets.qt_custom_line_edit import QtLineEdit
from widgets.qt_form_layout import QtFormLayout
from widgets.qt_line import QtHLine


class CreateSegmentationDialog(QDialog):
    created = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        show_workflow_mode: bool = True,
        default_workflow_mode: str | None = None,
    ) -> None:
        super().__init__(parent)

        cancel_btn = QtPushButton("cancel")
        self.create_btn = QtPushButton("create")
        self.create_btn.setEnabled(False)

        self.project_name_line = QtLineEdit()
        self.project_name_line.setValidator(
            QRegExpValidator(QRegExp("[a-zA-Z0-9_\-]+"))
        )
        self.project_folder_line = QtLineEdit()
        self.project_folder_line.setValidator(
            QRegExpValidator(QRegExp("[a-zA-Z0-9_\-:\\/]+"))
        )
        self.project_image_line = QtLineEdit()
        self.project_mode_combo = QComboBox()
        self.project_mode_combo.addItem("Full Segmentation", constants.IMAGE)
        self.project_mode_combo.addItem(
            "Binary Voxel Correction", constants.BINARY_VOXEL_CORRECTION
        )
        if default_workflow_mode is None:
            self.project_mode_combo.setCurrentIndex(1)
        else:
            index = self.project_mode_combo.findData(default_workflow_mode)
            self.project_mode_combo.setCurrentIndex(max(index, 0))
        self.x_scale = QtDoubleSpinBox(1.0)
        self.y_scale = QtDoubleSpinBox(1.0)
        self.z_scale = QtDoubleSpinBox(1.0)

        select_folder_btn = QtPushButton(
            "select_folder", "select folder", True, self.select_folder
        )
        select_image_btn = QtPushButton(
            "select_image", "select image", True, self.select_image
        )
        select_folder_btn.setObjectName("ProjectButton")
        select_image_btn.setObjectName("ProjectButton")

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowTitle("create segmentation project")
        label = QtLabel("creation instruction", {"en", "ru"})
        warning = QtLabel("creation warning", {"en", "ru"})
        icon_label = QWidget()
        icon_label.setObjectName("warning_icon_element")

        self.project_name_line.textChanged.connect(self.check_content)
        self.project_folder_line.textChanged.connect(self.check_content)
        self.project_image_line.textChanged.connect(self.check_content)
        self.project_mode_combo.currentIndexChanged.connect(self.check_content)

        cancel_btn.clicked.connect(self.reject)
        self.create_btn.clicked.connect(self.accept)
        cancel_btn.setMinimumSize(80, 30)
        self.create_btn.setMinimumSize(80, 30)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        form_layout = QtFormLayout()
        form_layout.addRow(label)
        form_layout.addRow(QtLabel("project name"), self.project_name_line)
        layout2 = QHBoxLayout()
        layout2.addWidget(self.project_folder_line)
        layout2.addWidget(select_folder_btn)
        form_layout.addRow(QtLabel("folder"), layout2)
        layout3 = QHBoxLayout()
        layout3.addWidget(self.project_image_line)
        layout3.addWidget(select_image_btn)
        form_layout.addRow(QtLabel("original image"), layout3)
        if show_workflow_mode:
            form_layout.addRow(QtLabel("workflow mode"), self.project_mode_combo)
        form_layout.addRow(QtHLine())
        form_layout.addRow(QtLabel("x scale"), self.x_scale)
        form_layout.addRow(QtLabel("y scale"), self.y_scale)
        form_layout.addRow(QtLabel("z scale"), self.z_scale)
        layout4 = QHBoxLayout()
        layout4.addSpacing(20)
        layout4.addWidget(icon_label)
        layout4.addWidget(warning)
        form_layout.addRow(layout4)
        layout.addLayout(form_layout)
        layout5 = QHBoxLayout()
        layout5.setContentsMargins(0, 0, 8, 6)
        layout5.addStretch(1)
        layout5.addWidget(cancel_btn)
        layout5.addWidget(self.create_btn)
        layout.addStretch(1)
        layout.addLayout(layout5)
        self.setLayout(layout)

        self.setMinimumWidth(650)
        self.setMinimumHeight(300)

    def setWindowTitle(self, title):
        return super().setWindowTitle(Translater.instance().get_translation(title))

    def check_content(self):
        folder = self.project_folder_line.text()
        image = self.project_image_line.text()
        mode = self.project_mode_combo.currentData()
        supported_formats = (".tif", ".tiff", ".off")
        if mode == constants.BINARY_VOXEL_CORRECTION:
            supported_formats = (".tif", ".tiff")
        self.create_btn.setEnabled(
            bool(
                self.project_name_line.text()
                and (folder and os.path.isdir(folder))
                and (
                    image
                    and os.path.isfile(image)
                    and image.endswith(supported_formats)
                )
            )
        )

    def select_folder(self):
        dlg = QFileDialog(self)
        folders = get_open_history_folders()
        dlg.setHistory(folders)
        folder = dlg.getExistingDirectory(
            caption=Translater.instance().get_translation("select folder"),
            directory=folders[0],
        )
        if folder and folder != "":
            status, _, _ = self.project_folder_line.validator().validate(
                folder, len(folder)
            )
            if status > 0:
                self.project_folder_line.setText(folder)
                update_open_history_folders(folder)
            else:
                show_warning("Invalid characters in path")

    def select_image(self):
        dlg = QFileDialog(self)
        files = get_open_history_files()
        hist = []
        for file in files:
            hist.append(os.path.dirname(file))
        dlg.setHistory(hist)
        mode = self.project_mode_combo.currentData()
        files_filter = "Files (*.tiff *.tif *.off)"
        if mode == constants.BINARY_VOXEL_CORRECTION:
            files_filter = "Files (*.tiff *.tif)"
        path, folder = dlg.getOpenFileName(
            caption=Translater.instance().get_translation("select image"),
            directory=hist[0],
            filter=files_filter,
        )
        if path and path != "":
            self.project_image_line.setText(path)
            update_open_history_files(path)
