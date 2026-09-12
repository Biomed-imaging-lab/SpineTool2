from pathlib import Path

from utils.appdirs import config_dir
from utils.settings import Settings


class SegmentationSettings(Settings):
    _instance = None

    FILE_NAME: str = config_dir() + "/segmentation_settings.json"
    RESOURCES_PATH: str = str(Path(__file__).parent / "resources")
    SVGS_PATH: str = RESOURCES_PATH + "/icons"
    STYLES_PATH: str = RESOURCES_PATH + "/styles"

    def __init__(self):
        if not SegmentationSettings._instance:
            schema = {
                "title": "segmentation settings",
                "type": "object",
                "properties": {
                    "highlight_thickness": {
                        "title": "highlight thickness",
                        "description": "highlight thickness description",
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "save_active_stage": {
                        "title": "keep active stages",
                        "description": "keep active stages description",
                        "type": "boolean",
                    },
                    "save_non_editable": {
                        "title": "keep non-editable layers",
                        "description": "keep non-editable layers description",
                        "type": "boolean",
                    },
                },
            }
            ui_schema = {}
            default_state = {
                "highlight_thickness": 1,
                "save_active_stage": True,
                "save_non_editable": False,
                "open_history_files": [],
                "open_history_folders": [],
            }
            description = "segmentation project settings"
            not_displayed = ["open_history_files", "open_history_folders"]
            super().__init__(
                schema,
                ui_schema,
                default_state,
                description,
                SegmentationSettings.FILE_NAME,
                not_displayed,
            )

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = SegmentationSettings()
        return cls._instance
