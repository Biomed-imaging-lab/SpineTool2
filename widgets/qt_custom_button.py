from PyQt5.QtWidgets import QPushButton, QToolButton

from utils.qt_translater import Translater


class QtPushButton(QPushButton):
    def __init__(self, key=None, tooltip=None, mode=False, slot=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self.key = None
        self.tooltip = None
        if mode:
            self.setProperty("mode", key)
            self.setProperty("with_mode", True)
        elif key is not None:
            self.setText(key)
        if tooltip is not None:
            self.setToolTip(tooltip)
        if slot is not None:
            self.clicked.connect(slot)

    def setText(self, key):
        self.key = key
        text = Translater.instance().get_translation(key)
        super().setText(text)

    def setToolTip(self, tooltip):
        self.tooltip = tooltip
        text = Translater.instance().get_translation(tooltip)
        super().setToolTip(text)

    def language_changed(self):
        if self.key:
            self.setText(self.key)
        if self.tooltip is not None:
            self.setToolTip(self.tooltip)


class QtToolButton(QToolButton):
    def __init__(self, key=None, tooltip=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self.key = None
        self.tooltip = None
        if key is not None:
            self.setText(key)
        if tooltip is not None:
            self.setToolTip(tooltip)

    def setText(self, key):
        self.key = key
        text = Translater.instance().get_translation(key)
        super().setText(text)

    def setToolTip(self, tooltip):
        self.tooltip = tooltip
        text = Translater.instance().get_translation(tooltip)
        super().setToolTip(text)

    def language_changed(self):
        if self.key:
            self.setText(self.key)
        if self.tooltip is not None:
            self.setToolTip(self.tooltip)
