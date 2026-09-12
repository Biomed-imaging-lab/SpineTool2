from PyQt5.QtWidgets import QCheckBox

from utils.qt_translater import Translater


class QtCheckBox(QCheckBox):
    def __init__(self, key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self.key = None
        if key is not None:
            self.setText(key)

    def setText(self, key):
        self.key = key
        text = Translater.instance().get_translation(key)
        super().setText(text)

    def language_changed(self):
        if self.key:
            self.setText(self.key)
