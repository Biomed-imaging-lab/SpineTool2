"""Small, standalone, sequential TIFF-to-SpineTool batch application (PyQt5)."""
import json
from pathlib import Path
import sys
import threading

import numpy as np
from tifffile import imread
from PyQt5.QtCore import Qt, QPointF, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_neural_project as cli


def number(value, low=0.000001, high=1000000, decimals=6):
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setValue(value)
    return spin


def project_folder(image, destination):
    """destination / immediate source directory name / TIFF stem."""
    image = Path(image)
    return Path(destination).resolve() / (image.parent.name or "root") / image.stem


class Canvas(QWidget):
    def __init__(self, projection, mask=None, editable=False):
        super().__init__()
        projection = np.ascontiguousarray(projection, dtype=np.uint8)
        h, w = projection.shape
        self.base = QImage(projection.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.mask = QImage(w, h, QImage.Format_ARGB32)
        self.mask.fill(Qt.transparent)
        if mask is not None:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask > 0] = [255, 90, 40, 120]
            self.mask = QImage(rgba.data, w, h, w*4, QImage.Format_RGBA8888).convertToFormat(QImage.Format_ARGB32)
        self.editable = editable
        self.radius = 30
        self.previous = None
        self.setMinimumSize(280, 230)
        self.setMouseTracking(True)

    def image_rect(self):
        scale = min(self.width()/self.base.width(), self.height()/self.base.height())
        w, h = self.base.width()*scale, self.base.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202226"))
        target = self.image_rect()
        painter.drawImage(target, self.base)
        painter.drawImage(target, self.mask)

    def draw(self, event):
        if not self.editable or not (event.buttons() & (Qt.LeftButton | Qt.RightButton)):
            return
        rect = self.image_rect()
        if not rect.contains(QPointF(event.pos())):
            self.previous = None
            return
        point = QPointF((event.x()-rect.left())*self.base.width()/rect.width(),
                        (event.y()-rect.top())*self.base.height()/rect.height())
        painter = QPainter(self.mask)
        if event.buttons() & Qt.RightButton:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.setPen(QPen(QColor(255, 90, 40, 120), self.radius*2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.previous is None:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 90, 40, 120))
            painter.drawEllipse(point, self.radius, self.radius)
        else:
            painter.drawLine(self.previous, point)
        painter.end()
        self.previous = point
        self.update()

    def mousePressEvent(self, event):
        self.previous = None
        self.draw(event)

    def mouseMoveEvent(self, event):
        self.draw(event)

    def mouseReleaseEvent(self, _):
        self.previous = None

    def array(self):
        bits = self.mask.bits()
        bits.setsize(self.mask.byteCount())
        pixels = np.frombuffer(bits, dtype=np.uint8).reshape(self.mask.height(), self.mask.bytesPerLine())
        # QImage ARGB32 is native-endian; alpha is the high byte on little-endian hosts.
        alpha_byte = 3 if sys.byteorder == "little" else 0
        return (pixels[:, alpha_byte:self.mask.width()*4:4] > 0).astype(np.uint8).copy()


class ROIEditor(QDialog):
    def __init__(self, projection, mask=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Area of interest — XY projection, all Z slices")
        self.resize(850, 700)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Left mouse: paint • Right mouse: erase • ROI applies through all Z slices"))
        self.canvas = Canvas(projection, mask, editable=True)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Brush radius (image pixels):"))
        radius = QSpinBox()
        radius.setRange(1, 2000)
        radius.setValue(30)
        radius.valueChanged.connect(lambda value: setattr(self.canvas, "radius", value))
        row.addWidget(radius)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        row.addWidget(clear)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_mask)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def clear(self):
        self.canvas.mask.fill(Qt.transparent)
        self.canvas.update()

    def accept_mask(self):
        if not self.canvas.array().any():
            QMessageBox.warning(self, "Empty ROI", "Draw a nonempty area or press Cancel.")
            return
        self.accept()


class Worker(QThread):
    log = pyqtSignal(str)
    status = pyqtSignal(int, str)
    finished_batch = pyqtSignal(dict)

    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs
        self.stop_event = threading.Event()

    def run(self):
        completed, failed, valid = [], [], []
        # Save every image's ROI before starting any model, including queued images.
        self.log.emit("Saving projects and all ROIs before inference…")
        for index, args in enumerate(self.jobs):
            self.status.emit(index, "Saving ROI")
            try:
                cli.run_regions(args, self.stop_event, self.log.emit, prepare_only=True)
                args.clear_run = False  # The formerly empty folder now contains its ROI.
                self.status.emit(index, "ROI saved")
            except Exception as error:
                self.log.emit(f"ERROR preparing {args.image}: {error}")
                self.status.emit(index, "Error")
                valid = [str(job.output_dir / job.image.stem) for job in self.jobs
                         if (job.output_dir / job.image.stem / "project_description.json").is_file()]
                self.finished_batch.emit({"complete": [], "failed": [str(args.image)], "valid": valid,
                                          "stopped": self.stop_event.is_set()})
                return
        for index, args in enumerate(self.jobs):
            if self.stop_event.is_set():
                break
            folder = args.output_dir / args.image.stem
            self.status.emit(index, "Running")
            try:
                cli.run_regions(args, self.stop_event, self.log.emit)
                completed.append(str(folder))
                self.status.emit(index, "Complete")
            except InterruptedError as error:
                self.log.emit(str(error))
                self.status.emit(index, "Stopped (saved)")
            except Exception as error:
                failed.append(str(args.image))
                self.log.emit(f"ERROR {args.image}: {error}")
                self.status.emit(index, "Error")
            if (folder / "project_description.json").is_file():
                valid.append(str(folder))
            if self.stop_event.is_set():
                break
        valid = [str(job.output_dir / job.image.stem) for job in self.jobs
                 if (job.output_dir / job.image.stem / "project_description.json").is_file()]
        self.finished_batch.emit({"complete": completed, "failed": failed, "valid": valid,
                                  "stopped": self.stop_event.is_set()})


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpineTool — Neural batch")
        self.resize(1150, 850)
        self.entries = []
        self.worker = None
        self.projection = None
        self.current_entry = None
        self.stage_count = 3
        self.progress_done = 0
        self.restored_destination = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.controls = QTabWidget()
        layout.addWidget(self.controls, 3)
        files = QWidget()
        file_layout = QVBoxLayout(files)
        self.controls.addTab(files, "1. TIFF files and areas")
        row = QHBoxLayout()
        add = QPushButton("Select TIFF files…")
        add.clicked.connect(self.add_files)
        remove = QPushButton("Remove selected file")
        remove.clicked.connect(self.remove_file)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        file_layout.addLayout(row)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["TIFF", "X µm", "Y µm", "Z µm", "ROIs", "Status"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(self.select_file)
        split = QSplitter()
        split.addWidget(self.table)
        preview = QWidget()
        self.preview_layout = QVBoxLayout(preview)
        self.preview_hint = QLabel("Select a TIFF. No ROI = the whole volume.")
        self.preview_hint.setWordWrap(True)
        self.preview_layout.addWidget(self.preview_hint)
        self.canvas = None
        self.areas = QListWidget()
        self.areas.setMaximumHeight(90)
        self.areas.currentRowChanged.connect(self.show_area)
        self.preview_layout.addWidget(self.areas)
        roi_row = QHBoxLayout()
        self.roi_buttons = []
        for title, action in (("Add area of interest", self.add_area), ("Edit", self.edit_area), ("Remove", self.remove_area)):
            button = QPushButton(title)
            button.clicked.connect(action)
            self.roi_buttons.append(button)
            roi_row.addWidget(button)
        self.preview_layout.addLayout(roi_row)
        split.addWidget(preview)
        split.setSizes([550, 500])
        file_layout.addWidget(split, 1)

        options = QWidget()
        form = QFormLayout(options)
        self.controls.addTab(options, "2. Output and processing")
        self.destination = QLineEdit()
        self.destination.editingFinished.connect(self.restore_areas)
        form.addRow("Output directory", self.path_row(self.destination, directory=True, callback=self.restore_areas))
        form.addRow(QLabel("Projects: output / source-parent-directory / TIFF-name / project_description.json"))
        common = QWidget()
        xyz = QHBoxLayout(common)
        xyz.setContentsMargins(0, 0, 0, 0)
        self.common_scale = []
        for label, value in zip("XYZ", (.025, .025, .1)):
            xyz.addWidget(QLabel(label))
            widget = number(value)
            self.common_scale.append(widget)
            xyz.addWidget(widget)
        form.addRow("Voxel size (µm)", common)
        self.individual = QCheckBox("Set individually (XYZ columns on file tab)")
        self.individual.toggled.connect(self.individual_changed)
        form.addRow(self.individual)
        self.stage = QComboBox()
        self.stage.addItems(["Through stage 3 — before shaft/spines", "Through stage 4 — include shaft/spines"])
        self.stage.currentIndexChanged.connect(self.update_options)
        form.addRow("Automatic pipeline", self.stage)
        self.device = QComboBox()
        self.device.addItems(["GPU if available, otherwise CPU", "CPU only", "Require CUDA GPU"])
        self.device.currentIndexChanged.connect(self.update_options)
        form.addRow("Inference device", self.device)
        threshold_row = QWidget()
        threshold_layout = QHBoxLayout(threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.override = QCheckBox("Override stage 1–3 defaults")
        self.threshold = number(.5, 0, 1, 3)
        self.threshold.setEnabled(False)
        self.override.toggled.connect(self.threshold.setEnabled)
        threshold_layout.addWidget(self.override)
        threshold_layout.addWidget(self.threshold)
        form.addRow("Probability threshold", threshold_row)
        self.clear_run = QCheckBox("Clear run — require empty target project folders (does not delete files)")
        form.addRow(self.clear_run)
        self.stage4 = QGroupBox("Stage 4 only")
        stage_form = QFormLayout(self.stage4)
        self.mode = QComboBox()
        self.mode.addItems(["trunk", "spine"])
        self.mode.currentIndexChanged.connect(self.update_options)
        stage_form.addRow("Threshold target", self.mode)
        self.trunk = number(.5, 0, 1, 3)
        self.spine = number(.8, 0, 1, 3)
        stage_form.addRow("Trunk threshold", self.trunk)
        stage_form.addRow("Spine threshold", self.spine)
        self.fill = QCheckBox("Fill enclosed holes before stage 4")
        self.fill.setChecked(True)
        stage_form.addRow(self.fill)
        self.overlap = QComboBox()
        self.overlap.addItems(["From application settings", "75", "50", "20"])
        stage_form.addRow("Overlap %", self.overlap)
        self.precision = QComboBox()
        self.precision.addItems(["From application settings", "Off", "On (CUDA)"])
        stage_form.addRow("Mixed precision", self.precision)
        self.prune = number(4, 0, 20, 1)
        stage_form.addRow("Skeleton prune length (µm)", self.prune)
        form.addRow(self.stage4)
        self.advanced_fields = {}
        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        for flag, title, directory in [
            ("settings", "Application settings JSON", False), ("models-dir", "Models directory", True),
            ("plugin-root", "AI plugin directory", True), ("python", "Inference Python executable", False),
            ("vsot-root", "VSOT root", True), ("vsot-matlab-dir", "VSOT src/+comSeg", True),
            ("octave", "Octave executable", False),
        ] + [(f"model-{stage}", f"Stage {stage} model override", False) for stage in range(1, 5)]:
            widget = QLineEdit()
            widget.setPlaceholderText("Use application default")
            self.advanced_fields[flag] = widget
            advanced_form.addRow(title, self.path_row(widget, directory))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(advanced)
        self.controls.addTab(scroll, "3. Runtime paths (optional)")
        self.update_options()

        self.overall = QProgressBar()
        self.overall.setFormat("Batch %p%")
        self.current = QProgressBar()
        self.current.setFormat("Current stage %p%")
        layout.addWidget(self.overall)
        layout.addWidget(self.current)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(10000)
        layout.addWidget(self.logs, 2)
        actions = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.start)
        self.resume_button = QPushButton("Continue saved batch…")
        self.resume_button.clicked.connect(self.resume_batch)
        self.stop_button = QPushButton("Stop and save")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        actions.addStretch()
        actions.addWidget(self.run_button)
        actions.addWidget(self.resume_button)
        actions.addWidget(self.stop_button)
        layout.addLayout(actions)

    def path_row(self, edit, directory=False, callback=None):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        browse = QPushButton("…")
        browse.setMaximumWidth(35)
        def choose():
            path = QFileDialog.getExistingDirectory(self, "Select directory", edit.text()) if directory else QFileDialog.getOpenFileName(self, "Select file", edit.text())[0]
            if path:
                edit.setText(path)
                if callback:
                    callback()
        browse.clicked.connect(choose)
        row.addWidget(browse)
        return container

    def individual_changed(self, enabled):
        for spin in self.common_scale:
            spin.setEnabled(not enabled)
        for row in range(self.table.rowCount()):
            for axis in range(3):
                widget = self.table.cellWidget(row, axis+1)
                if enabled and not self.entries[row].get("individual_set"):
                    widget.setValue(self.common_scale[axis].value())
                widget.setEnabled(enabled)
            if enabled:
                self.entries[row]["individual_set"] = True

    def update_options(self):
        enabled = self.stage.currentIndex() == 1
        self.stage4.setEnabled(enabled)
        self.trunk.setEnabled(self.mode.currentText() == "trunk")
        self.spine.setEnabled(self.mode.currentText() == "spine")
        if hasattr(self, "advanced_fields"):
            for key in ("model-4", "vsot-root", "vsot-matlab-dir", "octave"):
                self.advanced_fields[key].setEnabled(enabled)
        if self.device.currentIndex() == 1:
            self.precision.setCurrentIndex(1)
            self.precision.setEnabled(False)
        else:
            self.precision.setEnabled(True)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select 3D TIFF files", "", "TIFF (*.tif *.tiff *.TIF *.TIFF)")
        self.add_paths(paths)

    def add_paths(self, paths):
        known = {entry["path"] for entry in self.entries}
        for path in paths:
            path = str(Path(path).resolve())
            if path in known:
                continue
            known.add(path)
            row = self.table.rowCount()
            self.entries.append({"path": path, "masks": []})
            self.table.insertRow(row)
            item = QTableWidgetItem(Path(path).name)
            item.setToolTip(path)
            self.table.setItem(row, 0, item)
            for axis, common in enumerate(self.common_scale):
                spin = number(common.value())
                spin.setEnabled(self.individual.isChecked())
                self.table.setCellWidget(row, axis+1, spin)
            self.table.setItem(row, 4, QTableWidgetItem("Whole volume"))
            self.table.setItem(row, 5, QTableWidgetItem("Pending"))
        self.table.resizeColumnToContents(0)
        self.restore_areas()
        if self.entries:
            self.table.selectRow(0)

    def remove_file(self):
        row = self.table.currentRow()
        if row >= 0:
            self.entries.pop(row)
            self.table.removeRow(row)
            self.select_file(self.table.currentRow())

    def select_file(self, row, *_):
        self.projection = None
        self.current_entry = self.entries[row] if 0 <= row < len(self.entries) else None
        self.areas.clear()
        if not self.current_entry:
            self.replace_canvas(None)
            return
        try:
            image = imread(self.current_entry["path"])
            if image.ndim != 3 or image.dtype.kind not in "uif" or not np.isfinite(image).all():
                raise ValueError("Select a finite numeric 3D ZYX TIFF without channel or time axes")
            self.current_entry["shape"] = image.shape
            projection = image.max(axis=0).astype(np.float32)
            low, high = np.percentile(projection, [1, 99.5])
            self.projection = (np.clip((projection-low)/max(high-low, 1e-10), 0, 1)*255).astype(np.uint8)
            self.preview_hint.setText(f"{Path(self.current_entry['path']).name} • ZYX {image.shape}\nXY maximum projection; ROI extends through all Z.")
            self.refresh_areas()
        except Exception as error:
            self.preview_hint.setText(str(error))
            self.replace_canvas(None)

    def replace_canvas(self, mask):
        if self.canvas:
            self.preview_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        if self.projection is not None:
            self.canvas = Canvas(self.projection, mask)
            self.preview_layout.insertWidget(1, self.canvas, 1)

    def refresh_areas(self):
        self.update_roi_controls()
        self.areas.clear()
        if self.current_entry:
            for index in range(len(self.current_entry["masks"])):
                self.areas.addItem(f"Area {index+1}")
            row = self.table.currentRow()
            self.table.item(row, 4).setText(str(len(self.current_entry["masks"])) if self.current_entry["masks"] else "Whole volume")
            if self.current_entry["masks"]:
                self.areas.setCurrentRow(0)
            else:
                self.replace_canvas(None)

    def show_area(self, row):
        masks = self.current_entry["masks"] if self.current_entry else []
        self.replace_canvas(masks[row] if 0 <= row < len(masks) else None)

    def add_area(self):
        if self.projection is None or self.current_entry.get("roi_locked"):
            return
        editor = ROIEditor(self.projection, parent=self)
        if editor.exec_() == QDialog.Accepted:
            self.current_entry["masks"].append(editor.canvas.array())
            self.refresh_areas()
            self.areas.setCurrentRow(len(self.current_entry["masks"])-1)

    def edit_area(self):
        row = self.areas.currentRow()
        if self.projection is None or row < 0 or self.current_entry.get("roi_locked"):
            return
        editor = ROIEditor(self.projection, self.current_entry["masks"][row], self)
        if editor.exec_() == QDialog.Accepted:
            self.current_entry["masks"][row] = editor.canvas.array()
            self.show_area(row)

    def remove_area(self):
        row = self.areas.currentRow()
        if self.current_entry and row >= 0 and not self.current_entry.get("roi_locked"):
            self.current_entry["masks"].pop(row)
            self.refresh_areas()

    def update_roi_controls(self):
        editable = self.current_entry is not None and not self.current_entry.get("roi_locked")
        for button in self.roi_buttons:
            button.setEnabled(editable)
            button.setToolTip("Saved batch ROIs are read-only; use a new output folder to change them." if not editable else "")

    def restore_areas(self):
        if not self.destination.text().strip():
            return
        destination = str(Path(self.destination.text()).resolve())
        if self.restored_destination != destination:
            self.restored_destination = destination
            self.load_setup()
        for row, entry in enumerate(self.entries):
            folder = project_folder(entry["path"], self.destination.text())
            manifest = folder / "artifacts/plugin_runtime/batch.json"
            entry["roi_locked"] = manifest.is_file()
            if not manifest.is_file() or entry["masks"]:
                continue
            try:
                spec = json.loads(manifest.read_text(encoding="utf-8"))["specification"]
                if spec.get("explicit_areas"):
                    entry["masks"] = [imread(folder / f"artifacts/roi_runs/area_{i:03d}/layers/roi.tif")[0].astype(np.uint8)
                                      for i in range(1, len(spec["areas"])+1)]
                if not self.individual.isChecked():
                    self.individual.setChecked(True)
                entry["individual_set"] = True
                for axis, value in enumerate(spec["voxel_size"][::-1]):
                    self.table.cellWidget(row, axis+1).setValue(value)
                self.table.item(row, 4).setText(str(len(entry["masks"])) if entry["masks"] else "Whole volume")
            except (OSError, ValueError, KeyError, IndexError):
                continue
        self.select_file(self.table.currentRow())

    def setup_path(self):
        return Path(self.destination.text()) / ".spinetool-neural-batch.json"

    def save_setup(self, jobs=None):
        data = {"common_xyz": [spin.value() for spin in self.common_scale],
                "individual": self.individual.isChecked(), "stage": self.stage.currentIndex(),
                "device": self.device.currentIndex(), "override": self.override.isChecked(),
                "threshold": self.threshold.value(), "clear_run": self.clear_run.isChecked(),
                "mode": self.mode.currentIndex(), "trunk": self.trunk.value(), "spine": self.spine.value(),
                "fill": self.fill.isChecked(), "overlap": self.overlap.currentIndex(),
                "precision": self.precision.currentIndex(), "prune": self.prune.value(),
                "paths": {flag: widget.text() for flag, widget in self.advanced_fields.items()},
                "xyz": {entry["path"]: [self.table.cellWidget(row, axis+1).value() for axis in range(3)]
                        for row, entry in enumerate(self.entries)}}
        if jobs is not None:
            data["jobs"] = [job.command for job in jobs]
        cli.write_json(self.setup_path(), data)

    def load_setup(self):
        path = self.setup_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for spin, value in zip(self.common_scale, data["common_xyz"]):
                spin.setValue(float(value))
            for key in ("stage", "device", "mode", "overlap", "precision"):
                getattr(self, key).setCurrentIndex(int(data[key]))
            for key in ("individual", "override", "clear_run", "fill"):
                getattr(self, key).setChecked(bool(data[key]))
            for key in ("threshold", "trunk", "spine", "prune"):
                getattr(self, key).setValue(float(data[key]))
            for flag, value in data["paths"].items():
                if flag in self.advanced_fields:
                    self.advanced_fields[flag].setText(str(value))
            for row, entry in enumerate(self.entries):
                if entry["path"] in data["xyz"]:
                    entry["individual_set"] = True
                    for axis, value in enumerate(data["xyz"][entry["path"]]):
                        self.table.cellWidget(row, axis+1).setValue(float(value))
            self.update_options()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.logs.appendPlainText(f"Could not restore saved GUI settings: {error}")

    def make_jobs(self):
        if not self.entries or not self.destination.text().strip():
            raise ValueError("Select TIFF files and an output directory")
        jobs, targets = [], set()
        for row, entry in enumerate(self.entries):
            image = Path(entry["path"])
            target = project_folder(image, self.destination.text())
            key = str(target).casefold() if sys.platform in ("win32", "darwin") else str(target)
            if key in targets:
                raise ValueError(f"Two inputs map to the same project: {target}")
            targets.add(key)
            xyz = [self.table.cellWidget(row, axis+1).value() for axis in range(3)] if self.individual.isChecked() else [w.value() for w in self.common_scale]
            argv = [str(image), str(target.parent), "--voxel-size", *map(str, xyz[::-1]),
                    "--device", ["auto", "cpu", "cuda"][self.device.currentIndex()]]
            if self.stage.currentIndex() == 1:
                argv += ["--segment-spines", "--threshold-mode", self.mode.currentText()]
                argv += ["--trunk-threshold" if self.mode.currentText() == "trunk" else "--spine-threshold",
                         str(self.trunk.value() if self.mode.currentText() == "trunk" else self.spine.value())]
                argv += ["--prune-length-um", str(self.prune.value())]
                if not self.fill.isChecked():
                    argv.append("--no-fill-holes")
                if self.overlap.currentIndex():
                    argv += ["--overlap-percent", self.overlap.currentText()]
                if self.precision.currentIndex():
                    argv.append("--mixed-precision" if self.precision.currentIndex() == 2 else "--no-mixed-precision")
            if self.override.isChecked():
                argv += ["--threshold", str(self.threshold.value())]
            if self.clear_run.isChecked():
                argv.append("--clear-run")
            for flag, widget in self.advanced_fields.items():
                if widget.isEnabled() and widget.text().strip():
                    argv += ["--"+flag, widget.text().strip()]
            args = cli.parser().parse_args(argv)
            args.command = argv
            args.area_masks = [mask.copy() for mask in entry["masks"]]
            cli.check_directory(target, args.clear_run)
            cli.prepare(args)
            # Validate every image/ROI before launching any model.
            volume = imread(image)
            if volume.ndim != 3 or volume.dtype.kind not in "uif" or not np.isfinite(volume).all():
                raise ValueError(f"Invalid 3D ZYX TIFF: {image}")
            if any(mask.shape != volume.shape[1:] or not mask.any() for mask in args.area_masks):
                raise ValueError(f"Invalid ROI for {image}")
            jobs.append(args)
        return jobs

    def reset_progress(self):
        self.progress_done = 0
        for bar in (self.current, self.overall):
            bar.setRange(0, 100)
            bar.setValue(0)

    def start(self):
        try:
            jobs = self.make_jobs()
            self.save_setup(jobs)
        except Exception as error:
            self.reset_progress()
            self.logs.appendPlainText(f"ERROR: {error}")
            QMessageBox.warning(self, "Check settings", str(error))
            return
        self.begin_jobs(jobs)

    def resume_batch(self):
        directory = QFileDialog.getExistingDirectory(self, "Select output folder of the unfinished batch", self.destination.text())
        if not directory:
            return
        try:
            data = json.loads((Path(directory) / ".spinetool-neural-batch.json").read_text(encoding="utf-8"))
            commands = data.get("jobs", [])
            paths = [cli.parser().parse_args(command).image for command in commands] if commands else list(data["xyz"])
            if not paths or any(not Path(path).is_file() for path in paths):
                raise ValueError("Some original TIFF files are missing; restore them at their saved paths")
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            self.entries = []
            self.current_entry = None
            self.table.blockSignals(False)
            self.destination.setText(directory)
            self.restored_destination = None
            self.add_paths(paths)
            self.clear_run.setChecked(False)
            jobs = []
            for row, entry in enumerate(self.entries):
                folder = project_folder(entry["path"], directory)
                manifest = folder / "artifacts/plugin_runtime/batch.json"
                spec = json.loads(manifest.read_text(encoding="utf-8"))["specification"]
                if spec["explicit_areas"] and len(entry["masks"]) != len(spec["areas"]):
                    raise ValueError(f"Saved ROI masks are missing or unreadable: {folder}")
                entry["roi_locked"] = True
                if commands:
                    command = [arg for arg in commands[row] if arg != "--clear-run"]
                    args = cli.parser().parse_args(command)
                    args.output_dir = folder.parent
                    args.area_masks = [mask.copy() for mask in entry["masks"]]
                    cli.prepare(args)
                    jobs.append(args)
            if not commands:
                jobs = self.make_jobs()  # Compatibility with older saved GUI settings.
            self.update_roi_controls()
        except Exception as error:
            self.reset_progress()
            self.logs.appendPlainText(f"ERROR resuming batch: {error}")
            QMessageBox.warning(self, "Cannot resume batch", str(error))
            return
        self.begin_jobs(jobs)

    def begin_jobs(self, jobs):
        self.controls.setEnabled(False)
        self.run_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.logs.clear()
        self.reset_progress()
        self.stage_count = 4 if jobs[0].segment_spines else 3
        self.total_stages = sum(max(1, len(job.area_masks))*self.stage_count for job in jobs)
        self.worker = Worker(jobs)
        self.worker.log.connect(self.append_log)
        self.worker.status.connect(lambda row, status: self.table.item(row, 5).setText(status))
        self.worker.finished_batch.connect(self.finished)
        self.worker.start()

    def append_log(self, line):
        self.logs.appendPlainText(line)
        if line.startswith("SPINETOOL_PROGRESS "):
            try:
                value = json.loads(line.split(" ", 1)[1])
                self.current.setValue(int(100*value["value"]/max(1, value.get("total", 100))))
            except (ValueError, KeyError, TypeError):
                pass
        if line.startswith("Stage ") and "already completed" in line:
            self.progress_done += 1
        if line.startswith("SPINETOOL_STAGE_COMPLETE "):
            self.progress_done += 1
        if line.startswith("SPINETOOL_BRANCH_COMPLETE "):
            self.progress_done += int(line.rsplit(" ", 1)[1])
        self.overall.setValue(min(100, int(100*self.progress_done/max(1, self.total_stages))))

    def stop(self):
        if self.worker:
            self.worker.stop_event.set()
            self.stop_button.setEnabled(False)
            self.logs.appendPlainText("Stopping the active process tree; keeping the last valid checkpoint…")

    def finished(self, summary):
        self.worker.wait()
        self.controls.setEnabled(True)
        self.run_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not summary["stopped"] and not summary["failed"]:
            self.overall.setValue(100)
        else:
            self.reset_progress()
        self.restore_areas()
        text = (f"{'Stopped' if summary['stopped'] else 'Finished'}. Completed: {len(summary['complete'])}; "
                f"valid projects including partial: {len(summary['valid'])}; errors: {len(summary['failed'])}.\n"
                f"Output: {self.destination.text()}\n" + "\n".join(summary["valid"]))
        self.logs.appendPlainText(text)
        QMessageBox.information(self, "Batch summary", text)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop()
            event.ignore()
            self.logs.appendPlainText("Wait for Stop to finish, then close this window.")
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
