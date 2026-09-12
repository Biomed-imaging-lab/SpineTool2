import yaml
from PyQt5.QtCore import QObject, pyqtSignal


class Translater(QObject):
    FILE_PATH = "translations.yml"

    _instance = None

    language_changed_signal = pyqtSignal()

    def __init__(self):
        if not Translater._instance:
            super().__init__()
            self.global_lang = None
            self.global_languages = None
            self.global_translations = None
            self.load_translations()

    def load_translations(self):
        file = self.load_translations_file(self.FILE_PATH)
        self.global_translations = file["translations"]
        self.global_languages = file["supported_languages"]
        self.global_lang = self.global_languages[0]

    def load_translations_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_translation(self, key, lang=None, translations=None):
        if not lang:
            lang = self.global_lang
        if not translations:
            translations = self.global_translations
        if key:
            if key in translations:
                if lang in translations[key]:
                    return translations[key][lang]
            return key

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = Translater()
        return cls._instance
