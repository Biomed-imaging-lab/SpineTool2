from pathlib import Path

from utils.appdirs import config_dir
from utils.notifications import NotificationSeverity
from utils.settings import Settings
from utils.themes import THEMES


class ApplicationSettings(Settings):
    _instance = None

    FILE_NAME: str = config_dir() + "/application_settings.json"
    RESOURCES_PATH: str = str(Path(__file__).parent / "resources")
    SVGS_PATH: str = RESOURCES_PATH + "/icons"
    STYLES_PATH: str = RESOURCES_PATH + "/styles"

    def __init__(self):
        if not ApplicationSettings._instance:
            schema = {
                "title": "application settings",
                "type": "object",
                "properties": {
                    "language": {
                        "title": "language",
                        "description": "language description",
                        "type": "string",
                        "enum": ["en", "ru"],
                    },
                    "theme": {
                        "title": "theme",
                        "description": "theme description",
                        "type": "string",
                        "enum": list(THEMES.keys()),
                    },
                    "gui_notification_level": {
                        "title": "gui notification level",
                        "description": "gui notification level description",
                        "type": "string",
                        "enum": [s.value for s in NotificationSeverity],
                    },
                    "console_notification_level": {
                        "title": "console notification level",
                        "description": "console notification level description",
                        "type": "string",
                        "enum": [s.value for s in NotificationSeverity],
                    },
                    "window_statusbar": {
                        "title": "show status bar",
                        "description": "show status bar description",
                        "type": "boolean",
                    },
                    "save_window_geometry": {
                        "title": "save window geometry",
                        "description": "save window geometry description",
                        "type": "boolean",
                    },
                    "confirm_restore_settings": {
                        "title": "confirm settings restoring",
                        "description": "confirm settings restoring description",
                        "type": "boolean",
                    },
                    "confirm_close_window": {
                        "title": "confirm application closing",
                        "description": "confirm application closing description",
                        "type": "boolean",
                    },
                    "auto_saving_when_closing": {
                        "title": "save the project before closing",
                        "description": "save the project before closing description",
                        "type": "boolean",
                    },
                    "confirm_close_project": {
                        "title": "confirm project closing",
                        "description": "confirm project closing description",
                        "type": "boolean",
                    },
                },
            }
            ui_schema = {}
            default_state = {
                "language": "en",
                "theme": list(THEMES.keys())[0],
                "gui_notification_level": NotificationSeverity.INFO.value,
                "console_notification_level": NotificationSeverity.NONE.value,
                "window_statusbar": True,
                "save_window_geometry": True,
                "confirm_close_window": True,
                "confirm_restore_settings": True,
                "auto_saving_when_closing": False,
                "confirm_close_project": True,
                "first_time": True,
                "window_position": None,
                "window_size": None,
                "window_maximized": False,
                "window_fullscreen": False,
                "preferences_size": None,
                "open_history": [],
            }
            description = "general settings"
            not_displayed = [
                "first_time",
                "window_position",
                "window_size",
                "window_maximized",
                "window_fullscreen",
                "preferences_size",
                "open_history",
            ]
            super().__init__(
                schema,
                ui_schema,
                default_state,
                description,
                ApplicationSettings.FILE_NAME,
                not_displayed,
            )

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = ApplicationSettings()
        return cls._instance
