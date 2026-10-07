import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from projects.segmentation.utils.constants import (
    EMPTY_PROJECT,
    EMPTY_PROJECT_DATA,
    EMPTY_LAYER_PARAMETERS,
    LayerParametersKeys as LPK,
    ProjectDescriptionkKeys as PDK,
)
from projects.segmentation.utils.project_info import ProjectInfo


def _project_info(folder, artifact_path):
    info = EMPTY_PROJECT.copy()
    info[PDK.DATA.value] = EMPTY_PROJECT_DATA.copy()
    layer = EMPTY_LAYER_PARAMETERS.copy()
    layer[LPK.LAYER_ID.value] = 0
    layer[LPK.FILE.value] = artifact_path
    info[PDK.LAYERS_PARAMETERS.value] = [layer]
    return ProjectInfo(info, str(folder))


class ProjectPathPortabilityTest(unittest.TestCase):
    def test_project_paths_survive_save_move_reopen_and_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            original_folder = tmp_path / "projects" / "sample"
            original_folder.mkdir(parents=True)
            artifact = original_folder / "layers" / "mask.tif"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")

            project = _project_info(original_folder, str(artifact))
            project.save()

            saved = json.loads(
                (original_folder / ProjectInfo.DESCRIPTION_FILENAME).read_text()
            )
            saved_path = saved[PDK.LAYERS_PARAMETERS.value][0][LPK.FILE.value]
            self.assertEqual(saved_path, "layers/mask.tif")
            self.assertEqual(saved[PDK.PROJECT_ROOT.value], str(original_folder.resolve()))

            reopened = ProjectInfo.load(project.filename)
            self.assertEqual(
                reopened.resolve_path(saved_path, must_exist=True), str(artifact.resolve())
            )

            moved_folder = tmp_path / "renamed-location" / "sample"
            moved_folder.parent.mkdir()
            shutil.move(str(original_folder), str(moved_folder))
            moved = ProjectInfo.load(str(moved_folder / ProjectInfo.DESCRIPTION_FILENAME))
            self.assertEqual(
                moved.resolve_path(saved_path, must_exist=True),
                str((moved_folder / "layers" / "mask.tif").resolve()),
            )

            moved.save()
            saved_after_move = json.loads(
                (moved_folder / ProjectInfo.DESCRIPTION_FILENAME).read_text()
            )
            self.assertEqual(
                saved_after_move[PDK.LAYERS_PARAMETERS.value][0][LPK.FILE.value],
                saved_path,
            )
            self.assertEqual(
                os.path.dirname(ProjectInfo.load(moved.filename).filename),
                str(moved_folder.resolve()),
            )


    def test_legacy_absolute_artifact_path_rebases_after_folder_move(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            original_folder = tmp_path / "old" / "sample"
            original_folder.mkdir(parents=True)
            artifact = original_folder / "layers" / "mask.tif"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")

            project = _project_info(original_folder, str(artifact))
            project.save()
            moved_folder = tmp_path / "new" / "sample"
            moved_folder.parent.mkdir()
            shutil.move(str(original_folder), str(moved_folder))

            # Simulate a project created before paths were made relative.
            project_file = moved_folder / ProjectInfo.DESCRIPTION_FILENAME
            saved = json.loads(project_file.read_text())
            saved[PDK.LAYERS_PARAMETERS.value][0][LPK.FILE.value] = str(artifact)
            project_file.write_text(json.dumps(saved))

            reopened = ProjectInfo.load(str(project_file))
            self.assertEqual(
                reopened.resolve_path(str(artifact), must_exist=True),
                str((moved_folder / "layers" / "mask.tif").resolve()),
            )
            self.assertEqual(reopened.portable_path(str(artifact)), "layers/mask.tif")

    def test_windows_absolute_path_rebases_on_any_host_os(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current_folder = Path(temp_dir)
            artifact = current_folder / "layers" / "mask.tif"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")

            project = _project_info(current_folder, "")
            project.project_root = r"C:\old\sample"
            old_absolute_path = r"C:\old\sample\layers\mask.tif"

            self.assertEqual(
                project.resolve_path(old_absolute_path, must_exist=True),
                str(artifact.resolve()),
            )

    def test_posix_absolute_path_rebases_on_any_host_os(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current_folder = Path(temp_dir)
            artifact = current_folder / "layers" / "mask.tif"
            artifact.parent.mkdir()
            artifact.write_bytes(b"artifact")

            project = _project_info(current_folder, "")
            project.project_root = "/old/sample"

            self.assertEqual(
                project.resolve_path(
                    "/old/sample/layers/mask.tif", must_exist=True
                ),
                str(artifact.resolve()),
            )
