import weakref

from PyQt5.QtWidgets import QPushButton, QRadioButton

from utils.qt_translater import Translater


class QtModeRadioButton(QRadioButton):
    def __init__(
        self, layer, button_name, mode, *, tooltip=None, checked=False
    ) -> None:
        super().__init__()
        Translater.instance().language_changed_signal.connect(self.language_changed)

        self.layer_ref = weakref.ref(layer)
        self.tooltip = None
        if tooltip is not None:
            self.setToolTip(tooltip)
        self.setChecked(checked)
        self.setProperty("mode", button_name)
        self.setFixedWidth(28)
        self.mode = mode
        if mode is not None:
            self.toggled.connect(self._set_mode)

    def setToolTip(self, tooltip):
        self.tooltip = tooltip
        text = Translater.instance().get_translation(tooltip)
        super().setToolTip(text)

    def language_changed(self):
        if self.tooltip is not None:
            self.setToolTip(self.tooltip)

    def _set_mode(self, mode_selected):
        layer = self.layer_ref()
        if layer is None:
            return

        if mode_selected:
            layer.mode = self.mode


class QtModePushButton(QPushButton):
    def __init__(self, layer, button_name, *, slot=None, tooltip=None) -> None:
        super().__init__()

        self.layer = layer
        self.setProperty("mode", button_name)
        if tooltip is not None:
            self.setToolTip(tooltip)
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        if slot is not None:
            self.clicked.connect(slot)

    def setToolTip(self, tooltip):
        self.tooltip = tooltip
        text = Translater.instance().get_translation(tooltip)
        super().setToolTip(text)
