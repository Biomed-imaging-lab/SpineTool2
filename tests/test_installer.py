"""Installer regression checks; no network or environment mutation."""
import contextlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import install


class InstallerTests(unittest.TestCase):
    def test_miniforge_downloads_versioned_installer_and_checksum(self):
        for system, machine in [("Windows", "x86_64"), ("Darwin", "x86_64"), ("Darwin", "arm64")]:
            with self.subTest(system=system, machine=machine), tempfile.TemporaryDirectory() as temp, patch.object(install, "EXTERNAL", Path(temp)), patch.object(install, "find_conda", return_value=None):
                requests = []

                def download(url, target):
                    requests.append(url)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((install.hashlib.sha256(b"installer").hexdigest() + "  installer").encode() if url.endswith(".sha256") else b"installer")
                    return target

                def complete(*args, **kwargs):
                    executable = install.conda_path(Path(temp) / "miniforge", system)
                    executable.parent.mkdir(parents=True, exist_ok=True)
                    executable.touch()

                with patch.object(install, "download", side_effect=download), patch.object(install, "run", side_effect=complete), patch.object(install.subprocess, "run", side_effect=complete):
                    self.assertTrue(install.bootstrap_conda(system, machine).is_file())
                self.assertEqual(requests[1], requests[0] + ".sha256")
                self.assertIn(f"/download/{install.MINIFORGE_VERSION}/Miniforge3-{install.MINIFORGE_VERSION}-", requests[0])

    def test_bundled_windows_cgal_archive_installs_flat_package(self):
        archive = install.ROOT / "CGAL.zip"
        with tempfile.TemporaryDirectory() as temp, patch.object(install, "ROOT", Path(temp)), patch.object(install, "EXTERNAL", Path(temp)), patch.object(install, "download", return_value=archive):
            with patch.object(install, "run", side_effect=[install.subprocess.CompletedProcess([], 1), install.subprocess.CompletedProcess([], 0)]):
                install.install_cgal("conda", Path(temp), "Windows", "x86_64")
            package = Path(temp) / "CGAL"
            self.assertTrue((package / "CGAL_Kernel.py").is_file())
            self.assertTrue((package / "libgmp-10.dll").is_file())
            self.assertFalse((package / "CGAL").exists())

    def test_dry_run_does_not_bootstrap_or_create_directories(self):
        with patch.object(install, "bootstrap_conda") as bootstrap, patch.object(install.Path, "mkdir") as mkdir, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(install.main(["--dry-run"]), 0)
        bootstrap.assert_not_called()
        mkdir.assert_not_called()

    def test_platform_selection(self):
        for system, machine, expected in [("Windows", "AMD64", "x86_64"), ("Darwin", "arm64", "arm64"), ("Darwin", "x86_64", "x86_64")]:
            with self.subTest(system=system, machine=machine), patch.object(install.platform, "system", return_value=system), patch.object(install.platform, "machine", return_value=machine):
                self.assertEqual(install.target_platform(), (system, expected))
        with patch.object(install.platform, "system", return_value="Linux"):
            with self.assertRaises(RuntimeError):
                install.target_platform()

    def test_cuda_not_selected_on_mac(self):
        with self.assertRaises(RuntimeError):
            install.torch_packages("Darwin", "arm64", "cuda")
        self.assertIn("cu124", install.torch_packages("Windows", "x86_64", "cuda")[-1])
        self.assertEqual(install.torch_packages("Darwin", "x86_64", "cpu"), ["torch==2.2.2"])

    def test_conda_subdir_ignores_wrong_inherited_architecture(self):
        with patch.dict(install.os.environ, {"CONDA_SUBDIR": "osx-64"}):
            self.assertEqual(install.conda_environment("Darwin", "arm64")["CONDA_SUBDIR"], "osx-arm64")

    def test_archive_rejects_unsafe_paths_before_writing(self):
        for name in ["../outside", "/absolute", "C:/outside", "..\\outside"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                archive = Path(temp) / "archive.zip"
                target = Path(temp) / "target"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr("valid.txt", "valid")
                    bundle.writestr(name, "bad")
                with self.assertRaises(RuntimeError):
                    install.extract_zip(archive, target)
                self.assertFalse(target.exists())

    def test_archive_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "archive.zip"
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(info, "../outside")
            with self.assertRaises(RuntimeError):
                install.extract_zip(archive, Path(temp) / "target")

    def test_models_repeat_and_conflict_preserves_existing_weights(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(install, "ROOT", Path(temp)), patch.object(install, "EXTERNAL", Path(temp)):
            archive = Path(temp) / "models.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for i in range(1, 5):
                    bundle.writestr(f"models/stage_{i}/model.ckpt", b"weights")
            install.install_models(archive)
            install.install_models(archive)
            existing = Path(temp) / "plugins/ai_segmentation/models/stage_4/model.ckpt"
            existing.write_bytes(b"custom weights")
            with self.assertRaises(RuntimeError):
                install.install_models(archive)
            self.assertEqual(existing.read_bytes(), b"custom weights")

    def test_cgal_failed_probe_restores_previous_package(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(install, "ROOT", Path(temp)), patch.object(install, "EXTERNAL", Path(temp)):
            root = Path(temp)
            (root / "CGAL").mkdir()
            (root / "CGAL/original.txt").write_text("keep")
            with zipfile.ZipFile(root / "CGAL.zip", "w") as bundle:
                bundle.writestr("CGAL/CGAL_Kernel.py", "new")
            with patch.object(install, "run", side_effect=[install.subprocess.CompletedProcess([], 1), RuntimeError("bad binary")]):
                with self.assertRaisesRegex(RuntimeError, "bad binary"):
                    install.install_cgal("conda", root, "Windows", "x86_64")
            self.assertEqual((root / "CGAL/original.txt").read_text(), "keep")
            self.assertFalse((root / "CGAL/CGAL_Kernel.py").exists())


if __name__ == "__main__":
    unittest.main()
