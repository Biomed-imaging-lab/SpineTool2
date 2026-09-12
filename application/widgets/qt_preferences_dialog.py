from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QVBoxLayout,
)

from application.qt_jsonschema_form.form import WidgetBuilder
from application.settings import ApplicationSettings
from application.widgets.qt_confirm_restore_settings import QtConfirmRestoreSettings
from utils.qt_translater import Translater
from utils.settings import Settings
from utils.shortcuts import ShortcutsHandler
from widgets.qt_custom_button import QtPushButton

if TYPE_CHECKING:
    from PyQt5.QtGui import QCloseEvent, QKeyEvent


class QtPreferencesDialog(QDialog):
    shortcuts_ui_schema = {
        "shortcuts": {"ui:widget": "shortcuts"},
    }
    shortcuts_schema = {
        "title": "shortcuts",
        "type": "object",
        "properties": {
            "shortcuts": {
                "title": "shortcuts",
                "description": "shortcuts description",
                "type": "object",
            }
        },
    }
    resized = pyqtSignal(QSize)

    def __init__(
        self,
        parent=None,
        settings: OrderedDict[str, Settings] = OrderedDict(),
        shortcuts_providers: OrderedDict[str, ShortcutsHandler] = OrderedDict(),
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._settings = settings
        self._shortcuts_providers = shortcuts_providers
        self._stack = QStackedWidget(self)
        self._list = QListWidget(self)
        self._list.setObjectName("Preferences")
        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Set up buttons
        self._button_cancel = QtPushButton("cancel")
        self._button_cancel.clicked.connect(self.reject)
        self._button_ok = QtPushButton("ok")
        self._button_ok.clicked.connect(self.accept)
        self._button_ok.setDefault(True)
        self._button_restore = QtPushButton("restore defaults")
        self._button_restore.clicked.connect(self._restore_default_dialog)

        # Layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._list)
        left_layout.addStretch()
        left_layout.addWidget(self._button_restore)
        left_layout.addWidget(self._button_cancel)
        left_layout.addWidget(self._button_ok)

        self.setLayout(QHBoxLayout())
        self.layout().addLayout(left_layout, 1)
        self.layout().addWidget(self._stack, 3)

        # Build dialog from settings
        self._rebuild_dialog()
        Translater.instance().language_changed_signal.connect(self._rebuild_dialog)

    def keyPressEvent(self, e: "QKeyEvent"):
        if e.key() == Qt.Key.Key_Escape:
            e.accept()
            self.accept()
            return
        super().keyPressEvent(e)

    def resizeEvent(self, event):
        """Override to emit signal."""
        self.resized.emit(event.size())
        super().resizeEvent(event)

    def _rebuild_dialog(self):
        """Removes settings not to be exposed to user and creates dialog pages."""
        self.setWindowTitle(Translater.instance().get_translation("preferences title"))

        self._starting_values = {}

        self._list.clear()
        while self._stack.count():
            self._stack.removeWidget(self._stack.currentWidget())

        for name, settings in self._settings.items():
            self._starting_values[name] = settings.state.copy()
            self._add_page(name, settings)

        self._add_shortcuts_page(self._shortcuts_providers)

        self._list.setCurrentRow(0)

    def get_translated_schema(self, schema):
        translater = Translater.instance()
        translated_schema = deepcopy(schema)
        translated_schema["title"] = translater.get_translation(
            translated_schema["title"]
        )
        for param in translated_schema["properties"]:
            translated_schema["properties"][param]["title"] = (
                translater.get_translation(
                    translated_schema["properties"][param]["title"]
                )
            )
            translated_schema["properties"][param]["description"] = (
                translater.get_translation(
                    translated_schema["properties"][param]["description"]
                )
            )
            if (
                "enum" in translated_schema["properties"][param]
                and "string" == translated_schema["properties"][param]["type"]
            ):
                items = []
                for item in translated_schema["properties"][param]["enum"]:
                    items.append((translater.get_translation(item), item))
                translated_schema["properties"][param]["enum"] = items
        return translated_schema

    def rebuild_dialog(
        self,
        settings: OrderedDict[str, Settings] = OrderedDict(),
        shortcuts_providers: OrderedDict[str, ShortcutsHandler] = OrderedDict(),
    ):
        self._settings = settings
        self._shortcuts_providers = shortcuts_providers
        self._rebuild_dialog()

    def _add_shortcuts_page(
        self, shortcuts_providers: OrderedDict[str, ShortcutsHandler]
    ):
        form = WidgetBuilder().create_form(
            self.get_translated_schema(self.shortcuts_schema),
            self.shortcuts_ui_schema,
            {"shortcuts": shortcuts_providers},
        )

        self._list.addItem(Translater.instance().get_translation("shortcuts"))
        self._stack.addWidget(form)

    def _add_page(self, name: str, settings: Settings):
        state = {}
        for name_, value in settings.state.items():
            if name_ not in settings.not_displayed:
                state[name_] = value
        form = WidgetBuilder().create_form(
            self.get_translated_schema(settings.schema), settings.ui_schema, state
        )
        form.widget.on_changed.connect(settings.update)

        self._list.addItem(Translater.instance().get_translation(name))
        self._stack.addWidget(form)

    def _restore_default_dialog(self):
        """Launches dialog to confirm restore settings choice."""
        if (
            ApplicationSettings.instance().state["confirm_restore_settings"]
            and QtConfirmRestoreSettings(self.parent()).exec() != QDialog.Accepted
        ):
            return

        for settings in self._settings.values():
            settings.reset()
        self._rebuild_dialog()

    def closeEvent(self, event: "QCloseEvent") -> None:
        event.accept()
        self.accept()

    def reject(self):
        """Restores the settings in place when dialog was launched."""
        for name, settings in self._settings.items():
            settings.update(self._starting_values[name])
        super().reject()
