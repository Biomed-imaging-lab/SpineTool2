from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QWidget

from viewer.components.dims import Dims
from viewer.widgets.qt_scrollbar import ModifiedScrollBar


class QtDimSliderWidget(QWidget):
    def __init__(self, parent: QWidget, axis: int) -> None:
        super().__init__(parent=parent)
        self.axis = axis
        self.qt_dims = parent
        self.dims: Dims = parent.dims
        self.axis_label = QLabel(self)
        self.axis_label.setToolTip("Axis")
        self.axis_label.setText(self.dims.axis_labels[self.axis])
        self.slider = None
        self.curslice_label = QLineEdit(self)
        self.curslice_label.setToolTip(
            "Current slice for axis {axis}".format(axis=axis)
        )
        self.curslice_label.setValidator(QIntValidator(0, 999999))
        self.curslice_label.editingFinished.connect(self._set_slice_from_label)
        self.totslice_label = QLabel(self)
        self.totslice_label.setToolTip("Total slices for axis {axis}".format(axis=axis))
        self.curslice_label.setObjectName("slice_label")
        self.totslice_label.setObjectName("slice_label")
        sep = QFrame(self)
        sep.setFixedSize(1, 14)
        sep.setObjectName("slice_label_sep")

        layout = QHBoxLayout()
        self._create_range_slider_widget()

        layout.addWidget(self.axis_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.curslice_label)
        layout.addWidget(sep)
        layout.addWidget(self.totslice_label)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)
        self.dims.axis_labels_.connect(self._pull_label)

    def _set_slice_from_label(self):
        try:
            max_allowed = self.dims.nsteps[self.axis] - 1
        except IndexError:
            return

        val = int(self.curslice_label.text())
        if val > max_allowed:
            val = max_allowed
            self.curslice_label.setText(str(val))

        self.curslice_label.clearFocus()
        self.qt_dims.setFocus()
        self.dims.set_current_step(self.axis, val)

    def _value_changed(self, value):
        self.dims.set_current_step(self.axis, value)

    def _create_range_slider_widget(self):
        slider = ModifiedScrollBar(Qt.Orientation.Horizontal)
        slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        slider.setMinimum(0)
        slider.setMaximum(self.dims.nsteps[self.axis] - 1)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setValue(self.dims.current_step[self.axis])

        # Listener to be used for sending events back to model:
        slider.valueChanged.connect(self._value_changed)

        def slider_focused_listener():
            self.dims.last_used = self.axis

        # linking focus listener to the last used:
        slider.sliderPressed.connect(slider_focused_listener)
        self.slider = slider

    def _pull_label(self):
        label = self.dims.axis_labels[self.axis]
        self.axis_label.setText(label)

    def _update_range(self):
        displayed_sliders = self.qt_dims._displayed_sliders

        nsteps = self.dims.nsteps[self.axis] - 1
        if nsteps == 0:
            displayed_sliders[self.axis] = False
            self.qt_dims.last_used = 0
            self.hide()
        else:
            if (
                not displayed_sliders[self.axis]
                and self.axis not in self.dims.displayed
            ):
                displayed_sliders[self.axis] = True
                self.last_used = self.axis
                self.show()
            self.slider.setMinimum(0)
            self.slider.setMaximum(nsteps)
            self.slider.setSingleStep(1)
            self.slider.setPageStep(1)
            self.slider.setValue(self.dims.current_step[self.axis])
            self.totslice_label.setText(str(nsteps))
            self.totslice_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self._update_slice_labels()

    def _update_slider(self):
        self.slider.setValue(self.dims.current_step[self.axis])
        self._update_slice_labels()

    def _update_slice_labels(self):
        self.curslice_label.setText(str(self.dims.current_step[self.axis]))
        self.curslice_label.setAlignment(Qt.AlignmentFlag.AlignRight)
