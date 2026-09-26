import ntpath
import os
from pathlib import Path
import tempfile
import unittest

from projects.segmentation.utils.project_info import ProjectInfo
from utils.project_paths import resolve_project_path, is_absolute_path


class ProjectPathTests(unittest.TestCase):
    def test_external_absolute_paths(self):
        for value in ('/Users/example/probability.tif', '/tmp/external.tif',
                      'C:\\images\\input.tif', '\\\\server\\share\\input.tif'):
            with self.subTest(value=value):
                self.assertTrue(is_absolute_path(value))
                expected = ntpath.normpath(value) if ntpath.splitdrive(value)[0] else os.path.normpath(value)
                self.assertEqual(resolve_project_path('/project', value), expected)

    def test_legacy_artifacts_and_windows_relative_separators(self):
        for value in ('/layers/input.tif', '/tmp/layers/input.tif',
                      '/auxiliary/spines.json', '/artifacts/model_runs/probability.tif',
                      r'layers\input.tif', 'layers/input.tif'):
            with self.subTest(value=value):
                expected = os.path.abspath(os.path.join('/project', value.lstrip('/').replace('\\', os.sep)))
                self.assertEqual(resolve_project_path('/project', value), expected)

    def test_moved_project_and_portable_save(self):
        with tempfile.TemporaryDirectory() as folder:
            current = Path(folder) / 'new'
            (current / 'layers').mkdir(parents=True)
            artifact = current / 'layers' / 'image.tif'
            artifact.write_bytes(b'placeholder')
            for old_root, old_file in (
                ('/old/project', '/old/project/layers/image.tif'),
                (r'C:\old\project', r'C:\old\project\layers\image.tif'),
            ):
                with self.subTest(old_root=old_root):
                    info = ProjectInfo(folder=str(current))
                    info.project_root = old_root
                    self.assertEqual(info.resolve_path(old_file, must_exist=True), str(artifact))
                    self.assertEqual(info.portable_path(old_file), 'layers/image.tif')

    def test_recovers_absolute_path_clipped_by_old_macos_save(self):
        with tempfile.TemporaryDirectory() as folder:
            info = ProjectInfo(folder=folder)
            info.project_root = "/Users/example/old_project"
            artifact = Path(folder) / "artifacts" / "probability.tif"
            artifact.parent.mkdir()
            artifact.write_bytes(b"placeholder")
            clipped = "Users/example/old_project/artifacts/probability.tif"
            self.assertEqual(info.resolve_path(clipped, must_exist=True), str(artifact))
            self.assertEqual(info.portable_path(clipped), "artifacts/probability.tif")

    def test_missing_external_path_is_not_rebased(self):
        with tempfile.TemporaryDirectory() as folder:
            info = ProjectInfo(folder=folder)
            missing = '/Volumes/unplugged/probability.tif'
            self.assertEqual(info.resolve_path(missing), missing)
            self.assertEqual(info.portable_path(missing), missing)
            with self.assertRaises(FileNotFoundError):
                info.resolve_path(missing, must_exist=True)

    def test_existing_probability_path_survives_save_and_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            info = ProjectInfo(folder=folder)
            artifact = Path(folder) / 'artifacts' / 'probability.tif'
            artifact.parent.mkdir()
            artifact.write_bytes(b'placeholder')
            relative = info.portable_path(str(artifact))
            info.save()
            loaded = ProjectInfo.load(info.filename)
            self.assertEqual(loaded.resolve_path(relative, must_exist=True), str(artifact))
            self.assertEqual(loaded.resolve_path(str(artifact), must_exist=True), str(artifact))


if __name__ == '__main__':
    unittest.main()
