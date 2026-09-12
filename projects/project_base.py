from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from utils.settings import Settings
from utils.shortcuts import ShortcutsHandler
from utils.themes import Style

if TYPE_CHECKING:
    from application.widgets.qt_main_window import Window


class ProjectBase(ShortcutsHandler, QObject):
    closed = pyqtSignal()
    _style: Style = Style([], [])

    def __init__(self, window: "Window", settings: Optional[Settings] = None):
        ShortcutsHandler.__init__(self)
        QObject.__init__(self)
        self._window: Window = window
        self._settings: Optional[Settings] = settings

        if self._settings is not None:
            self._settings.load()
            self._settings.updated.connect(self.on_settings_update)
        self._window.project_save.connect(self.save)
        self._window.project_close.connect(self.close)
        self._window.block_project_specific_actions(False)

    @property
    def settings(self) -> Optional[Settings]:
        return self._settings

    @property
    def additional_settings(self) -> Dict[str, Settings]:
        return {}

    @property
    def type(self) -> str:
        raise NotImplementedError

    @property
    def saved(self) -> bool:
        raise NotImplementedError

    @classmethod
    def style(cls) -> Style:
        return cls._style

    @classmethod
    def check_file(cls, path: str) -> bool:
        raise NotImplementedError

    @classmethod
    def create(cls, window: "Window") -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def open(cls, window: "Window", path: str) -> Optional["ProjectBase"]:
        raise NotImplementedError

    def change_theme(self, theme_id) -> None:
        """To make changes not related to the style sheet installation"""

    def close(self) -> None:
        raise NotImplementedError

    def save(self, save_all: bool = True) -> None:
        raise NotImplementedError

    def on_settings_update(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError
