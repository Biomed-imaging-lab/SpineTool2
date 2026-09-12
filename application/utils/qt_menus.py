from typing import TYPE_CHECKING, List, Optional

from functools import partial

from PyQt5.QtCore import pyqtSignal

from application.utils import constants
from projects.project_types import TYPES
from widgets.qt_custom_menu import QtAction, QtMenu

if TYPE_CHECKING:
    from application.widgets.qt_main_window import Window


class QtFileMenu(QtMenu):
    project_create = pyqtSignal(str)
    project_open = pyqtSignal()
    preferences_open = pyqtSignal()
    project_save = pyqtSignal()
    project_close = pyqtSignal()

    def __init__(self, parent: "Window"):
        super().__init__("file", parent._qt_window)
        self._project_secific_actions: List[QtAction] = []
        submenu = QtMenu(parent=self)
        for project_type in TYPES:
            action = QtAction(project_type, self)
            action.triggered.connect(partial(self._create_project, project_type))
            submenu.addAction(action)
        action = QtAction("create new project", self)
        action.setMenu(submenu)
        self.addAction(action)
        action = QtAction("open project", self)
        action.triggered.connect(self._open_project)
        action.setShortcut(constants.OPEN_PROJECT_SHORTCUT_TEXT)
        self.addAction(action)
        action = QtAction("save project", self)
        action.triggered.connect(self._save_project)
        self._project_secific_actions.append(action)
        action.setShortcut(constants.SAVE_PROJECT_SHORTCUT_TEXT)
        self.addAction(action)
        self.addSeparator()
        action = QtAction("open preferences", self)
        action.triggered.connect(self._open_preferences)
        action.setShortcut(constants.OPEN_PREFERENCES_SHORTCUT_TEXT)
        self.addAction(action)
        self.addSeparator()
        action = QtAction("close project", self)
        action.triggered.connect(self._close_project)
        self._project_secific_actions.append(action)
        action.setShortcut(constants.CLOSE_PROJECT_SHORTCUT_TEXT)
        self.addAction(action)

    def _open_preferences(self):
        self.preferences_open.emit()

    def _open_project(self):
        self.project_open.emit()

    def _save_project(self):
        self.project_save.emit()

    def _create_project(self, project_type: str):
        self.project_create.emit(project_type)

    def _close_project(self):
        self.project_close.emit()

    def block_project_specific_actions(self, block: bool = True):
        for action in self._project_secific_actions:
            action.setEnabled(not block)


class QtWindowMenu(QtMenu):
    def __init__(self, parent: "Window"):
        super().__init__("window", parent._qt_window)
        self._window = parent
        action = QtAction("toggle full screen", self)
        action.setCheckable(True)
        action.setChecked(self._window.is_full_screen())
        action.setShortcut(constants.TOGGLE_FULL_SCREEN_SHORTCUT_TEXT)
        action.toggled.connect(self._window.on_toggle_fullscreen)
        self.addAction(action)
        self.addSeparator()
        self._toggle_activity_dock_action = QtAction("toggle activity dock", self)
        self._toggle_activity_dock_action.setCheckable(True)
        self._toggle_activity_dock_action.setChecked(
            self._window.is_activity_dock_visible()
        )
        self._toggle_activity_dock_action.setShortcut(
            constants.TOGGLE_ACTIVITY_DOCK_SHORTCUT_TEXT
        )
        self._toggle_activity_dock_action.toggled.connect(
            self._window.on_toggle_activity_dock
        )
        self._window.activity_dock_toggled.connect(self._on_toggle_activity_dock)
        self.addAction(self._toggle_activity_dock_action)

    def _on_toggle_activity_dock(self) -> None:
        self._toggle_activity_dock_action.toggled.disconnect(
            self._window.on_toggle_activity_dock
        )
        self._toggle_activity_dock_action.setChecked(
            self._window.is_activity_dock_visible()
        )
        self._toggle_activity_dock_action.toggled.connect(
            self._window.on_toggle_activity_dock
        )

    def addAction(self, action: Optional[QtAction]):
        if len(self.actions()) == 3:
            self._separator = self.addSeparator()
        super().addAction(action)

    def removeAction(self, action: Optional[QtAction]):
        super().removeAction(action)
        if len(self.actions()) == 4:
            super().removeAction(self._separator)
