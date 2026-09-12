import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QDir, QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog

from application.settings import ApplicationSettings
from application.utils import constants
from application.utils.history import get_open_history, update_open_history
from application.utils.qt_event_filters import QtToolTipEventFilter
from application.utils.utils import maybe_allow_interrupt
from application.widgets.qt_main_window import Window
from application.widgets.qt_preferences_dialog import QtPreferencesDialog
from application.widgets.qt_splash_screen import SplashScreen
from projects.project_base import ProjectBase
from projects.project_types import TYPES
from utils.notifications import Notification, NotificationManager, show_warning
from utils.progress import progress
from utils.qt_translater import Translater
from utils.shortcuts import Shortcut, ShortcutsHandler
from utils.themes import THEMES, Style, build_theme_svgs, get_stylesheet, theme_path


def set_app_id(app_id):
    if os.name == "nt" and app_id and not getattr(sys, "frozen", False):
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


class Application(ShortcutsHandler):
    _instance = None
    _app = None
    _style = Style(
        [
            str(x)
            for x in Path(ApplicationSettings.STYLES_PATH).resolve().iterdir()
            if x.suffix == ".qss"
        ],
        [
            str(x)
            for x in Path(ApplicationSettings.SVGS_PATH).resolve().iterdir()
            if x.suffix == ".svg"
        ],
    )

    def __init__(
        self,
        app_name: str = None,
        app_version: str = None,
        icon: str = None,
        org_name: str = None,
        org_domain: str = None,
        app_id: str = None,
        title: str = "SpineTool2.0",
        splash: bool = True,
    ):
        if not Application._instance:
            ShortcutsHandler.__init__(self)

            self._settings = ApplicationSettings.instance()
            self._settings.load()
            translater = Translater.instance()
            translater.global_lang = self._settings.state["language"]
            translater.language_changed_signal.emit()

            set_values = {
                "app_name": app_name if app_name else "SpineTool",
                "app_version": app_version if app_version else "2.0",
                "icon": (
                    icon if icon else ApplicationSettings.RESOURCES_PATH + "/logo.png"
                ),
                "org_name": org_name if org_name else "",
                "org_domain": org_domain if org_domain else "",
                "app_id": app_id if app_id else "spine_tool.2.0",
            }

            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

            argv = sys.argv.copy()
            self._app = QApplication(argv)
            self._app.setApplicationName(set_values.get("app_name"))
            self._app.setApplicationVersion(set_values.get("app_version"))
            self._app.setOrganizationName(set_values.get("org_name"))
            self._app.setOrganizationDomain(set_values.get("org_domain"))
            set_app_id(set_values.get("app_id"))
            self._app.setWindowIcon(QIcon(set_values.get("icon")))

            self._window: Window = Window(title)
            self._active_project: Optional[ProjectBase] = None
            self._preferences_dialog: Optional[QtPreferencesDialog] = None

            self._app.installEventFilter(self._window.get_event_filter())
            self._app.installEventFilter(QtToolTipEventFilter())

            if splash:
                self._splash_screen = SplashScreen()
            else:
                self._splash_screen = None

            self._window.block_project_specific_actions()
            self._window.exit.connect(self.on_exit)
            self._window.project_create.connect(self.on_project_create)
            self._window.project_open.connect(self.on_project_open)
            self._window.preferences_open.connect(self.on_preferences_open)
            self._window.save.connect(self.on_project_save)

            notification_manager = NotificationManager.instance()
            notification_manager.notification_ready.connect(
                self._window.show_notification
            )
            notification_manager.notification_ready.connect(
                self.show_console_notification
            )

            for name in THEMES:
                QDir.addSearchPath(f"theme_{name}", str(theme_path(name)))

            self._openning = None
            self._openning_pbar = None
            last_opened = get_open_history()[0]
            if last_opened != str(Path().home()):
                for type, project in TYPES.items():
                    if project.check_file(last_opened):
                        self.open_project(last_opened, type)

            self._set_shortcuts()
            shortcuts_handlers: List[ShortcutsHandler] = [self]
            if self._active_project:
                shortcuts_handlers.append(self._active_project)
            self._window.set_shortcuts_handlers(shortcuts_handlers)

            self._update_theme()

            self._settings.updated.connect(self.on_settings_update)

            self._running = False
            self._closing = False

    def _set_shortcuts(self):
        shortcuts = [
            Shortcut(
                constants.TOGGLE_FULL_SCREEN_SHORTCUT_TEXT,
                description="toggle full screen",
            ),
            Shortcut(
                constants.EXIT_FULL_SCREEN_SHORTCUT_TEXT,
                self.exit_full_screen,
                description="exit full screen",
            ),
            Shortcut(
                constants.TOGGLE_ACTIVITY_DOCK_SHORTCUT_TEXT,
                description="toggle activity dock",
            ),
            Shortcut(
                constants.SHOW_ALL_SHORTCUTS_SHORTCUT_TEXT,
                self.show_shortcuts,
                description="show shortcuts",
            ),
            Shortcut(constants.OPEN_PROJECT_SHORTCUT_TEXT, description="open project"),
            Shortcut(
                constants.OPEN_PREFERENCES_SHORTCUT_TEXT, description="show preferences"
            ),
            Shortcut(
                constants.SAVE_PROJECT_SHORTCUT_TEXT, description="save current project"
            ),
            Shortcut(
                constants.CLOSE_PROJECT_SHORTCUT_TEXT,
                description="close current project",
            ),
        ]
        self.register_shortcuts("", shortcuts)
        self.active_shortcuts = [""]

    def run(self) -> None:
        if self._running:
            return

        if self._splash_screen:
            self._splash_screen.close()
        self._window.show()
        self._running = True
        with NotificationManager.instance(), maybe_allow_interrupt(self._app):
            self._app.exec()

    def exit_full_screen(self) -> None:
        if self._window.is_full_screen():
            self._window.on_toggle_fullscreen()

    def on_exit(self):
        self._closing = True
        if self._active_project:
            self._active_project.close()
        else:
            QApplication.closeAllWindows()
            QApplication.quit()
            Application._instance = None
            del self

    def on_project_save(self) -> None:
        self._active_project.save()

    def on_project_create(self, type: str) -> None:
        self._update_theme(TYPES[type])
        path = TYPES[type].create(self._window)
        if path is not None:
            if self._active_project:
                self._openning = (path, type)
                self._openning_pbar = progress(total=0, desc="open project in queue")
                self._active_project.close()
            else:
                self.open_project(path, type)
        else:
            show_warning("project creation error")

    def on_project_open(self) -> None:
        dlg = QFileDialog(self._window._qt_window)
        files = get_open_history()
        hist = []
        for file in files:
            hist.append(os.path.dirname(file))
        dlg.setHistory(hist)
        path, folder = dlg.getOpenFileName(
            caption=Translater.instance().get_translation("select project file"),
            directory=hist[0],
        )
        if path and path != "":
            for type, project in TYPES.items():
                if project.check_file(path):
                    if self._active_project:
                        self._openning = (path, type)
                        self._openning_pbar = progress(
                            total=0, desc="open project in queue"
                        )
                        self._active_project.close()
                    else:
                        self.open_project(path, type)
                    return
            show_warning("not recognize project type")

    def open_project(self, path: str, type: str) -> None:
        self._openning = None
        if self._openning_pbar:
            self._openning_pbar.close()
            self._openning_pbar = None
        self._update_theme(TYPES[type])
        self._active_project = TYPES[type].open(self._window, path)
        if self._active_project:
            update_open_history(path)
            self._active_project.closed.connect(self.on_project_close)
            shortcuts_handlers = OrderedDict({"application": self})
            key = "project (" + self._active_project.type + ")"
            shortcuts_handlers[key] = self._active_project
            self._window.set_shortcuts_handlers([self, self._active_project])
            if self._preferences_dialog:
                settings = OrderedDict({"application": self._settings})
                if self._active_project.settings:
                    settings[key] = self._active_project.settings
                settings.update(self._active_project.additional_settings)
                self._preferences_dialog.rebuild_dialog(settings, shortcuts_handlers)
        else:
            show_warning("failed open project")
            if self._preferences_dialog:
                self._preferences_dialog.rebuild_dialog(
                    {"application": self._settings}, {"application": self}
                )
        self._update_theme()

    def on_preferences_open(self) -> None:
        if self._preferences_dialog is None:
            settings = OrderedDict({"application": self._settings})
            shortcuts_handlers = OrderedDict({"application": self})
            if self._active_project:
                key = "project (" + self._active_project.type + ")"
                if self._active_project.settings:
                    settings[key] = self._active_project.settings
                settings.update(self._active_project.additional_settings)
                shortcuts_handlers[key] = self._active_project
            win = QtPreferencesDialog(
                self._window._qt_window, settings, shortcuts_handlers
            )
            self._preferences_dialog = win

            if self._settings.state["preferences_size"]:
                win.resize(*self._settings.state["preferences_size"])

            @win.resized.connect
            def _save_size(sz: QSize):
                self._settings.update({"preferences_size": (sz.width(), sz.height())})

            win.finished.connect(self._clean_preferences_dialog)
            win.show()
        else:
            self._preferences_dialog.raise_()

    def _clean_preferences_dialog(self):
        self._preferences_dialog = None

    def on_project_close(self) -> None:
        self._window.set_shortcuts_handlers([self])
        self._active_project.closed.disconnect(self.on_project_close)
        self._active_project = None
        self._update_theme()
        if self._preferences_dialog:
            self._preferences_dialog.rebuild_dialog(
                {"application": self._settings}, {"application": self}
            )
        if self._closing:
            QApplication.closeAllWindows()
            QApplication.quit()
            Application._instance = None
            del self
        elif self._openning:
            self.open_project(self._openning[0], self._openning[1])

    def set_status_bar_visibility(self, visible: bool = True):
        self._window.set_status_bar_visibility(visible)

    def show_console_notification(self, notification: Notification) -> None:
        try:
            if (
                notification.severity
                < self._settings.state["console_notification_level"]
            ):
                return

            print(notification)
        except Exception:
            print(
                "An error occurred while trying to format an error and show it in console.\n"
            )
            raise

    @classmethod
    def instance(
        cls,
        app_name: str = None,
        app_version: str = None,
        icon: str = None,
        org_name: str = None,
        org_domain: str = None,
        app_id: str = None,
        title: str = "SpineTool2.0",
        splash: bool = True,
    ):
        if not cls._instance:
            cls._instance = Application(
                app_name, app_version, icon, org_name, org_domain, app_id, title, splash
            )
        return cls._instance

    def on_settings_update(self, state: Dict[str, Any]):
        if "language" in state:
            translater = Translater.instance()
            translater.global_lang = state["language"]
            translater.language_changed_signal.emit()
        if "theme" in state:
            self._update_theme()
        if "window_statusbar" in state:
            self.set_status_bar_visibility(self._settings.state["window_statusbar"])
        self._settings.save()

    def _update_theme(self, project: ProjectBase = None):
        theme_id = self._settings.state["theme"]
        files = self._style.stylesheets_paths
        build_theme_svgs(theme_id, self._style.icons_paths)
        if self._active_project:
            build_theme_svgs(theme_id, self._active_project.style().icons_paths)
            files.extend(self._active_project.style().stylesheets_paths)
            self._active_project.change_theme(theme_id)
        if project:
            build_theme_svgs(theme_id, project.style().icons_paths)
            files.extend(project.style().stylesheets_paths)
        self._window.set_style_sheet(get_stylesheet(theme_id, files))

    def show_shortcuts(self):
        self.on_preferences_open()
        pref_list = self._preferences_dialog._list
        for i in range(pref_list.count()):
            if pref_list.item(i).text() == Translater.instance().get_translation(
                "shortcuts"
            ):
                pref_list.setCurrentRow(i)
