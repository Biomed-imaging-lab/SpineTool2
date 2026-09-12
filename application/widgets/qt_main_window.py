import os
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from weakref import WeakValueDictionary

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtWidgets import QApplication, QDialog, QDockWidget, QMainWindow, QWidget

from application.settings import ApplicationSettings
from application.utils import qt_menus
from application.widgets.qt_activity_dialog import QtActivityDialog
from application.widgets.qt_central_widget import QtCentralWidget, QtWelcomeWidget
from application.widgets.qt_confirm_close_dialog import QtConfirmCloseDialog
from application.widgets.qt_confirm_close_project_dialog import (
    QtConfirmCloseProjectDialog,
)
from application.widgets.qt_dock_widget import QtDockWidget
from application.widgets.qt_notification import QtNotification
from application.widgets.qt_status_bar import QtStatusBar
from utils.notifications import Notification
from utils.shortcuts import ShortcutsHandler
from widgets.qt_custom_menu import QtMenu

if TYPE_CHECKING:
    from magicgui.widgets import Widget


class QtMainWindow(QMainWindow):
    ICON_PATH = ApplicationSettings.RESOURCES_PATH + "/logo.png"

    exit = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        self.setWindowIcon(QIcon(QtMainWindow.ICON_PATH))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._central = QtCentralWidget(self)
        self._central.layout().setContentsMargins(4, 0, 4, 0)
        self.setCentralWidget(self._central)

        self._welcome_widget = QtWelcomeWidget(self._central)
        self._central.layout().addWidget(self._welcome_widget)

        self.setWindowTitle(title)

        self._maximized_flag = False
        self._fullscreen_flag = False
        self._normal_geometry = QRect()
        self._window_size = None
        self._window_pos = None
        self._old_size = None
        self._positions = []

        act_dlg = QtActivityDialog(self._central)
        self._central.resized.connect(act_dlg.move_to_bottom_right)
        act_dlg.hide()
        self._activity_dialog = act_dlg

        self._status_bar = QtStatusBar(self)
        self.setStatusBar(self._status_bar)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._shortcuts_handlers = None

    def statusBar(self) -> QtStatusBar:
        return self._status_bar

    def isFullScreen(self):
        return self._fullscreen_flag

    def showNormal(self):
        self._fullscreen_flag = False
        if os.name == "nt":
            self.setWindowFlags(
                self.windowFlags() ^ (Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            )
            self.setGeometry(self._normal_geometry)
        if self._maximized_flag:
            super().showMaximized()
        else:
            super().showNormal()

    def showFullScreen(self):
        self._fullscreen_flag = True
        self._maximized_flag = self.isMaximized()
        if os.name == "nt":
            self.setWindowFlags(
                self.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            )
            super().showNormal()
            self._normal_geometry = self.normalGeometry()
            screen_rect = self.windowHandle().screen().geometry()
            self.setGeometry(
                screen_rect.left() - 1,
                screen_rect.top() - 1,
                screen_rect.width() + 2,
                screen_rect.height() + 2,
            )
        else:
            super().showFullScreen()

    def _set_central_widget(self, widget: QWidget = None) -> None:
        old_widget = self._central.layout().itemAt(0).widget()
        old_widget.hide()
        self._central.layout().removeWidget(old_widget)
        if not widget:
            self._central.layout().addWidget(self._welcome_widget)
            self._welcome_widget.show()
        else:
            self._central.layout().addWidget(widget)
            widget.setParent(self._central)

    def _load_window_settings(self):
        """
        Load window layout settings from configuration.
        """
        settings = ApplicationSettings.instance()
        window_position = settings.state["window_position"]

        if not window_position:
            window_position = (self.x(), self.y())
        else:
            origin_x, origin_y = window_position
            screen = QApplication.screenAt(QPoint(origin_x, origin_y))
            screen_geo = screen.geometry() if screen else None
            if not screen_geo:
                window_position = (self.x(), self.y())

        return (
            settings.state["window_size"],
            window_position,
            settings.state["window_maximized"],
            settings.state["window_fullscreen"],
        )

    def _get_window_settings(self):
        """Return current window settings.

        Symmetric to the 'set_window_settings' setter.
        """

        window_fullscreen = self.isFullScreen()
        if window_fullscreen:
            window_maximized = self._maximized_flag
        else:
            window_maximized = self.isMaximized()

        return (
            self._window_size or (self.width(), self.height()),
            self._window_pos or (self.x(), self.y()),
            window_maximized,
            window_fullscreen,
        )

    def _set_window_settings(
        self, window_size, window_position, window_maximized, window_fullscreen
    ) -> None:
        """
        Set window settings.

        Symmetric to the 'get_window_settings' accessor.
        """
        self.setUpdatesEnabled(False)

        if window_position:
            window_position = QPoint(*window_position)
            self.move(window_position)

        if window_size:
            window_size = QSize(*window_size)
            self.resize(window_size)

        self._maximized_flag = window_maximized
        if window_fullscreen:
            self.showFullScreen()
        elif window_maximized:
            self.setWindowState(Qt.WindowState.WindowMaximized)

        self.setUpdatesEnabled(True)

    def _save_current_window_settings(self):
        """Save the current geometry of the main window."""
        (
            window_size,
            window_position,
            window_maximized,
            window_fullscreen,
        ) = self._get_window_settings()

        settings = ApplicationSettings.instance()
        if settings.state["save_window_geometry"]:
            new_state = {
                "window_maximized": window_maximized,
                "window_fullscreen": window_fullscreen,
                "window_position": window_position,
                "window_size": window_size,
            }
            settings.update(new_state)

    def changeEvent(self, event):
        """Handle window state changes."""
        if event.type() == QEvent.Type.WindowStateChange:
            condition = self.isMaximized() if os.name == "nt" else self.isFullScreen()
            if condition and self._old_size is not None:
                if self._positions and len(self._positions) > 1:
                    self._window_pos = self._positions[-2]

                self._window_size = (
                    self._old_size.width(),
                    self._old_size.height(),
                )
            else:
                self._old_size = None
                self._window_pos = None
                self._window_size = None
                self._positions = []

        super().changeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if self._shortcuts_handlers is not None:
            for handler in self._shortcuts_handlers:
                if handler.on_key_pressed(event):
                    event.accept()
                    return

    def keyReleaseEvent(self, event: QKeyEvent):
        if self._shortcuts_handlers is not None:
            for handler in self._shortcuts_handlers:
                if handler.on_key_released(event):
                    event.accept()
                    return

    def resizeEvent(self, event):
        """Override to handle original size before maximizing."""
        if event.oldSize().isValid():
            self._old_size = event.oldSize()
            self._positions.append((self.x(), self.y()))

            if self._positions and len(self._positions) >= 2:
                self._window_pos = self._positions[-2]
                self._positions = self._positions[-2:]

        super().resizeEvent(event)

    def closeEvent(self, event):
        settings = ApplicationSettings.instance()
        if (
            event.spontaneous()
            and settings.state["confirm_close_window"]
            and self._central.layout().itemAt(0).widget() != self._welcome_widget
            and QtConfirmCloseDialog(self).exec() != QDialog.Accepted
        ):
            event.ignore()
            return

        # Close any floating dockwidgets
        for dock in self.findChildren(QtDockWidget):
            if isinstance(dock, QWidget) and dock.isFloating():
                dock.setFloating(False)

        self._save_current_window_settings()

        self.exit.emit()
        event.ignore()

    def sizeHint(self):
        szh = super().sizeHint()
        szh.setWidth(1200)
        szh.setHeight(900)
        return szh


class Window(QObject):
    exit = pyqtSignal()
    project_create = pyqtSignal(str)
    project_open = pyqtSignal()
    preferences_open = pyqtSignal()
    project_save = pyqtSignal()
    project_close = pyqtSignal()
    save = pyqtSignal()

    activity_dock_toggled = pyqtSignal()

    def __init__(self, title: str = ""):
        super().__init__()
        self._dock_widgets: Dict[str, QtDockWidget] = WeakValueDictionary()

        self._qt_window = QtMainWindow(title)
        self._title = title
        self._qt_window.exit.connect(self._on_exit)
        self._qt_window._status_bar.activity_dock_toggled.connect(
            self._on_activity_dock_toggled
        )

        settings = ApplicationSettings.instance()
        if settings.state["first_time"]:
            settings.update({"first_time": False})
            self._qt_window.resize(self._qt_window.sizeHint())
        else:
            try:
                if settings.state["save_window_geometry"]:
                    self._qt_window._set_window_settings(
                        *self._qt_window._load_window_settings()
                    )
            except Exception as err:
                import warnings

                warnings.warn(
                    "The window geometry settings could not be loaded due to the following error: {err}".format(
                        err=err,
                    ),
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        self._add_menus()

    def show(self):
        self._qt_window.show()
        self._qt_window.activateWindow()
        self._qt_window.raise_()

    def _add_menus(self):
        """Add menubar"""
        self.main_menu = self._qt_window.menuBar()
        self._additional_menus = []

        self.file_menu = qt_menus.QtFileMenu(self)
        self.main_menu.addMenu(self.file_menu)
        self.file_menu.preferences_open.connect(self._on_open_preferences_triggered)
        self.file_menu.project_create.connect(self._on_project_create_triggered)
        self.file_menu.project_save.connect(self._on_project_save_triggered)
        self.file_menu.project_open.connect(self._on_project_open_triggered)
        self.file_menu.project_close.connect(self._on_project_close_triggered)

        self.window_menu = qt_menus.QtWindowMenu(self)
        self.main_menu.addMenu(self.window_menu)

    def block_project_specific_actions(self, block: bool = True):
        self.file_menu.block_project_specific_actions(block)

    def _on_project_create_triggered(self, type: str) -> None:
        self.project_create.emit(type)

    def _on_project_open_triggered(self) -> None:
        self.project_open.emit()

    def _on_project_save_triggered(self) -> None:
        self.project_save.emit()

    def _on_project_close_triggered(self) -> None:
        settings = ApplicationSettings.instance()
        if (
            settings.state["confirm_close_project"]
            and QtConfirmCloseProjectDialog(self._qt_window).exec() != QDialog.Accepted
        ):
            return
        self.project_close.emit()

    def _on_open_preferences_triggered(self) -> None:
        self.preferences_open.emit()

    def _on_save(self) -> None:
        self.save.emit()

    def _on_activity_dock_toggled(self) -> None:
        self.activity_dock_toggled.emit()

    def on_toggle_fullscreen(self):
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
        else:
            self._qt_window.showFullScreen()

    def is_full_screen(self) -> bool:
        return self._qt_window.isFullScreen()

    def on_toggle_activity_dock(self):
        self._qt_window._status_bar._toggle_activity_dock()

    def is_activity_dock_visible(self) -> bool:
        return self._qt_window._activity_dialog.isVisible()

    def get_event_filter(self) -> QObject:
        return self._qt_window

    def set_menus(self, menus: List[QtMenu] = []) -> None:
        for action in self._additional_menus:
            self.main_menu.removeAction(action)
        self._additional_menus = []
        for menu in menus:
            self._additional_menus.append(self.main_menu.addMenu(menu))

    def _on_exit(self):
        self.exit.emit()

    def add_dock_widget(
        self,
        widget: Union[QWidget, "Widget"],
        name: str,
        area: str = "right",
        allowed_areas: Optional[Sequence[str]] = None,
        add_vertical_stretch: bool = True,
        tabify: bool = False,
    ):
        """Convenience method to add a QDockWidget to the main window.

        Parameters
        ----------
        widget : QWidget
            `widget` will be added as QDockWidget's main widget.
        name : str, optional
            Name of dock widget to appear in window menu.
        area : str
            Side of the main window to which the new dock widget will be added.
            Must be in {'left', 'right', 'top', 'bottom'}
        allowed_areas : list[str], optional
            Areas, relative to main window, that the widget is allowed dock.
            Each item in list must be in {'left', 'right', 'top', 'bottom'}
            By default, all areas are allowed.
        add_vertical_stretch : bool, optional
            Whether to add stretch to the bottom of vertical widgets (pushing
            widgets up towards the top of the allotted area, instead of letting
            them distribute across the vertical space). By default, True.
        tabify : bool
            Flag to tabify dockwidget or not.
        """
        dock_widget = QtDockWidget(
            self._qt_window,
            widget,
            name=name,
            area=area,
            allowed_areas=allowed_areas,
            add_vertical_stretch=add_vertical_stretch,
        )

        self._add_dock_widget(dock_widget, tabify=tabify)

        # Add dock widget to dictionary
        self._dock_widgets[name] = dock_widget

        return dock_widget

    def _add_dock_widget(self, dock_widget: QtDockWidget, tabify: bool = False):
        # Find if any other dock widgets are currently in area
        current_dws_in_area = [
            dw
            for dw in self._qt_window.findChildren(QDockWidget)
            if self._qt_window.dockWidgetArea(dw) == dock_widget.qt_area
        ]
        self._qt_window.addDockWidget(dock_widget.qt_area, dock_widget)

        # If another dock widget present in area then tabify
        if current_dws_in_area:
            if tabify:
                self._qt_window.tabifyDockWidget(current_dws_in_area[-1], dock_widget)
                dock_widget.show()
                dock_widget.raise_()
            elif dock_widget.area in ("right", "left"):
                _wdg = [*current_dws_in_area, dock_widget]
                # add sizes to push lower widgets up
                sizes = list(range(1, len(_wdg) * 4, 4))
                sizes[-1] = 100
                self._qt_window.resizeDocks(_wdg, sizes, Qt.Orientation.Vertical)

        action = dock_widget.toggleViewAction()
        self.window_menu.addAction(action)

        dock_widget.setFloating(False)

    def remove_dock_widget(self, widget: QWidget | str):
        if widget == "all":
            for dw in list(self._dock_widgets.values()):
                self.remove_dock_widget(dw)
            return

        if not isinstance(widget, QDockWidget):
            dw: QDockWidget
            for dw in self._qt_window.findChildren(QDockWidget):
                if dw.widget() is widget:
                    _dw: QDockWidget = dw
                    break
            else:
                raise LookupError(
                    "Could not find a dock widget containing: {widget}".format(
                        widget=widget,
                    )
                )
        else:
            _dw = widget

        if _dw.widget():
            _dw.widget().setParent(None)
        self._qt_window.removeDockWidget(_dw)
        self.window_menu.removeAction(_dw.toggleViewAction())

        self._dock_widgets.pop(_dw.name, None)
        _dw.deleteLater()

    def add_function_widget(
        self, function, magic_kwargs=None, name: str = "", area=None, allowed_areas=None
    ):
        from magicgui import magicgui

        if magic_kwargs is None:
            magic_kwargs = {
                "auto_call": False,
                "call_button": "run",
                "layout": "vertical",
            }

        widget = magicgui(function, **magic_kwargs or {})

        if area is None:
            area = "right" if str(widget.layout) == "vertical" else "bottom"
        if allowed_areas is None:
            allowed_areas = [area]

        return self.add_dock_widget(
            widget,
            name=name or function.__name__.replace("_", " "),
            area=area,
            allowed_areas=allowed_areas,
        )

    def resize(self, width, height):
        self._qt_window.resize(width, height)

    def set_geometry(self, left, top, width, height):
        self._qt_window.setGeometry(left, top, width, height)

    def geometry(self) -> Tuple[int, int, int, int]:
        rect = self._qt_window.geometry()
        return rect.left(), rect.top(), rect.width(), rect.height()

    def on_status_changed(self, status: Dict[str, str]):
        self._qt_window._status_bar.setStatusText(status)

    def set_title(self, title: str = ""):
        if title != "":
            self._qt_window.setWindowTitle(title)
        else:
            self._qt_window.setWindowTitle(self._title)

    def on_help_changed(self, help: str = ""):
        self._qt_window._status_bar.setHelpText(help)

    def show_notification(self, notification: Notification):
        QtNotification.show_notification(notification, self._qt_window._central)

    def set_central_widget(self, widget: QWidget = None):
        self._qt_window._set_central_widget(widget)

    def set_shortcuts_handlers(
        self, shortcuts_handlers: List[ShortcutsHandler] = None
    ) -> None:
        self._qt_window._shortcuts_handlers = shortcuts_handlers

    def set_style_sheet(self, stylesheet=None) -> None:
        self._qt_window.setStyleSheet(stylesheet)

    def set_status_bar_visibility(self, visible: bool = True):
        self._qt_window._status_bar.setVisible(visible)
