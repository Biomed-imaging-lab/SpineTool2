from pathlib import Path

from utils.appdirs import config_dir
from utils.settings import Settings


class NeuralSegmentationSettings(Settings):
    _instance = None

    FILE_NAME: str = config_dir() + "/neural_segmentation_settings.json"
    RESOURCES_PATH: str = str(Path(__file__).parent / "resources")
    SVGS_PATH: str = RESOURCES_PATH + "/icons"
    STYLES_PATH: str = RESOURCES_PATH + "/styles"
    REPO_ROOT: Path = Path(__file__).resolve().parents[2]

    @classmethod
    def default_plugin_repo_root(cls) -> str:
        return str(cls.REPO_ROOT / "plugins" / "ai_segmentation")

    @classmethod
    def default_models_folder(cls) -> str:
        return str(Path(cls.default_plugin_repo_root()) / "models")

    def __init__(self):
        if not NeuralSegmentationSettings._instance:
            schema = {
                "title": "neural segmentation settings",
                "type": "object",
                "properties": {
                    "models_folder": {
                        "title": "AI models folder",
                        "description": "Root folder with stage_1, stage_2, stage_3, stage_4 model weights",
                        "type": "string",
                    },
                    "plugin_repo_root": {
                        "title": "AI runtime path",
                        "description": "Path to ai_segmentation runtime plugin",
                        "type": "string",
                    },
                    "plugin_python_executable": {
                        "title": "AI python executable",
                        "description": "Optional python executable for AI runtime. Empty means current environment",
                        "type": "string",
                    },
                    "vsot_matlab_dir": {
                        "title": "VSOT matlab dir",
                        "description": "Path to VSOT src/+comSeg directory",
                        "type": "string",
                    },
                    "vsot_root": {
                        "title": "VSOT root",
                        "description": "Path to the VSOT repository root (contains src and resources)",
                        "type": "string",
                    },
                    "octave_executable": {
                        "title": "Octave executable",
                        "description": "Full path to octave-cli.exe or octave.exe",
                        "type": "string",
                    },
                    "stage4_overlap_percent": {
                        "title": "Stage 4 inference overlap",
                        "description": "Sliding-window overlap for stage 4 inference",
                        "type": "integer",
                        "enum": [75, 50, 20],
                    },
                    "stage4_mixed_precision": {
                        "title": "Stage 4 mixed precision",
                        "description": "Use CUDA float16 autocast during stage 4 inference",
                        "type": "boolean",
                    },
                },
            }
            ui_schema = {}
            default_state = {
                "models_folder": self.default_models_folder(),
                "plugin_repo_root": self.default_plugin_repo_root(),
                "plugin_python_executable": "",
                "vsot_matlab_dir": "",
                "vsot_root": "",
                "octave_executable": "",
                "stage4_overlap_percent": 75,
                "stage4_mixed_precision": False,
                "open_history_files": [],
                "open_history_folders": [],
            }
            description = "neural segmentation project settings"
            not_displayed = [
                "stage4_overlap_percent",
                "stage4_mixed_precision",
                "stage4_skeleton_prune_length_um",
                "open_history_files",
                "open_history_folders",
            ]
            super().__init__(
                schema,
                ui_schema,
                default_state,
                description,
                NeuralSegmentationSettings.FILE_NAME,
                not_displayed,
            )

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = NeuralSegmentationSettings()
        return cls._instance

    def ensure_defaults(self) -> None:
        updates = {}
        if self.state.get("plugin_repo_root", "") == "":
            updates["plugin_repo_root"] = self.default_plugin_repo_root()
        if self.state.get("models_folder", "") == "":
            updates["models_folder"] = self.default_models_folder()
        if "stage4_overlap_percent" not in self.state:
            updates["stage4_overlap_percent"] = 75
        if "stage4_mixed_precision" not in self.state:
            updates["stage4_mixed_precision"] = False
        if updates:
            self.update(updates)
            Path(self.FILE_NAME).parent.mkdir(parents=True, exist_ok=True)
            self.save()
