from PyQt5.QtWidgets import QLabel, QSizePolicy

from utils.qt_translater import Translater


class QtLabel(QLabel):
    def __init__(self, key=None, word_wrap=set(), parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        Translater.instance().language_changed_signal.connect(self.language_changed)
        self.word_wrap = word_wrap
        self.key = None
        if key is not None:
            self.setText(key)

    def setText(self, key):
        self.key = key
        translater = Translater.instance()
        text = translater.get_translation(key)
        super().setText(text)
        self.setWordWrap(translater.global_lang in self.word_wrap)

    def language_changed(self):
        if self.key:
            self.setText(self.key)
