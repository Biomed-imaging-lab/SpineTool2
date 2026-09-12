from PyQt5.QtWidgets import QAction, QMenu

from utils.qt_translater import Translater


class QtMenu(QMenu):
    def __init__(self, title=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self._title = title

        if title:
            self.setTitle(title)

    def setTitle(self, title):
        self._title = title
        text = Translater.instance().get_translation(title)
        super().setTitle(text)

    def language_changed(self):
        if self._title:
            self.setTitle(self._title)


class QtAction(QAction):
    def __init__(self, key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self.key = None
        if key is not None:
            self.key = key
            self.setText(key)

    def setText(self, key):
        self.key = key
        text = Translater.instance().get_translation(key)
        super().setText(text)

    def language_changed(self):
        if self.key:
            self.setText(self.key)
