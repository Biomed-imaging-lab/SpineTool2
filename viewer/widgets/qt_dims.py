import numpy as np
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import QLineEdit, QSizePolicy, QVBoxLayout, QWidget

from viewer.components.dims import Dims
from viewer.widgets.qt_dims_slider import QtDimSliderWidget


class QtDims(QWidget):
    def __init__(self, dims: Dims, parent=None) -> None:
        super().__init__(parent=parent)

        self.SLIDERHEIGHT = 22

        self.dims = dims
        self.slider_widgets = []
        self._displayed_sliders = []

        # Initialises the layout:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Update the number of sliders now that the dims have been added
        self._update_nsliders()
        self.dims.current_step_.connect(self._update_slider)
        self.dims.range_.connect(self._update_range)
        self.dims.ndisplay_.connect(self._update_display)
        self.dims.order_.connect(self._update_display)
        self.dims.last_used_.connect(self._on_last_used_changed)

    @property
    def nsliders(self):
        return len(self.slider_widgets)

    def _on_last_used_changed(self):
        for i, widget in enumerate(self.slider_widgets):
            sld = widget.slider
            sld.setProperty("last_used", i == self.dims.last_used)
            sld.style().unpolish(sld)
            sld.style().polish(sld)

    def _update_slider(self):
        for widget in self.slider_widgets:
            widget._update_slider()

    def _update_range(self):
        for widget in self.slider_widgets:
            widget._update_range()

        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()

    def _update_display(self):
        widgets = reversed(list(enumerate(self.slider_widgets)))
        nsteps = self.dims.nsteps
        for axis, widget in widgets:
            if axis in self.dims.displayed or nsteps[axis] <= 1:
                # Displayed dimensions correspond to non displayed sliders
                self._displayed_sliders[axis] = False
                self.dims.last_used = 0
                widget.hide()
            else:
                # Non displayed dimensions correspond to displayed sliders
                self._displayed_sliders[axis] = True
                self.dims.last_used = axis
                widget.show()
        nsliders = np.sum(self._displayed_sliders)
        self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_slice_labels()
        self._resize_axis_labels()

    def _update_nsliders(self):
        self._create_sliders(3)
        self._update_display()
        for i in range(3):
            self._update_range()
            if self._displayed_sliders[i]:
                self._update_slider()

    def _resize_axis_labels(self):
        displayed_labels = [
            self.slider_widgets[idx].axis_label
            for idx, displayed in enumerate(self._displayed_sliders)
            if displayed
        ]
        if displayed_labels:
            fm = QFontMetrics(QFont("", 0))
            labels = self.findChildren(QLineEdit, "axis_label")
            # set maximum width to no more than 20% of slider width
            maxwidth = int(self.slider_widgets[0].width() * 0.2)
            # set new base width to the width of the longest label being displayed
            newwidth = max(
                [int(fm.boundingRect(dlab.text()).width()) for dlab in displayed_labels]
            )

            for labl in labels:
                labl_width = min([newwidth + 10, maxwidth])
                labl.setFixedWidth(labl_width)

    def _resize_slice_labels(self):
        width = 0
        for ax, maxi in enumerate(self.dims.nsteps):
            if self._displayed_sliders[ax]:
                length = len(str(maxi - 1))
                if length > width:
                    width = length
        # gui width of a string of length `width`
        fm = QFontMetrics(QFont("", 0))
        width = fm.boundingRect("8" * width).width()
        for labl in self.findChildren(QWidget, "slice_label"):
            labl.setFixedWidth(width + 6)

    def _create_sliders(self, number_of_sliders: int):
        for slider_num in range(self.nsliders, number_of_sliders):
            dim_axis = number_of_sliders - slider_num - 1
            slider_widget = QtDimSliderWidget(self, dim_axis)
            self.layout().addWidget(slider_widget)
            self.slider_widgets.insert(0, slider_widget)
            self._displayed_sliders.insert(0, True)
            nsliders = np.sum(self._displayed_sliders)
            self.setMinimumHeight(nsliders * self.SLIDERHEIGHT)
        self._resize_axis_labels()

    def closeEvent(self, event):
        [w.deleteLater() for w in self.slider_widgets]
        self.deleteLater()
        event.accept()
