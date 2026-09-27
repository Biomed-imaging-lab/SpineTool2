"""Install SpineTool2 locally. Bootstrap uses only Python's standard library."""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
RELEASES = "https://github.com/Biomed-imaging-lab/SpineTool2/releases/download"
VSOT_URL = "https://github.com/yu-lab-vt/VSOT/archive/refs/heads/main.zip"
OCTAVE_URL = "https://ftp.gnu.org/gnu/octave/windows/octave-10.3.0-w64.zip"
MODELS_URL = "https://drive.google.com/file/d/1Sq58lruLRNGSK9YELmpDl3Ar8qTT49lR/view"
MINIFORGE_VERSION = "26.7.2-0"


def target_platform():
    system, machine = platform.system(), platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    if (system, machine) not in {("Windows", "x86_64"), ("Darwin", "x86_64"), ("Darwin", "arm64")}:
        raise RuntimeError("Supported: Windows x64, macOS Intel and Apple Silicon (64-bit Python).")
    if sys.maxsize <= 2**32:
        raise RuntimeError("Please run the installer with 64-bit Python.")
    return system, machine


def run(args, *, check=True, env=None):
    args = [str(arg) for arg in args]
    print("+ " + subprocess.list2cmdline(args), flush=True)
    return subprocess.run(args, cwd=ROOT, env=env, check=check)


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print("Downloading " + url, flush=True)
    try:
        with urlopen(Request(url, headers={"User-Agent": "SpineTool2-installer"}), timeout=60) as source, partial.open("wb") as output:
            total = int(source.headers.get("Content-Length", 0))
            received, last_update = 0, time.monotonic()
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if time.monotonic() - last_update > 5:
                    print(f"  {received // 1048576} MiB" + (f" / {total // 1048576} MiB" if total else ""), flush=True)
                    last_update = time.monotonic()
            if total and received != total:
                raise RuntimeError("Incomplete download: " + url)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def extract_zip(archive, destination):
    """Validate every member before writing; reject traversal and symlinks."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
                raise RuntimeError("Unsafe archive member: " + info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError("Archive symlink is not supported: " + info.filename)
            resolved = (destination / str(path)).resolve()
            if destination != resolved and destination not in resolved.parents:
                raise RuntimeError("Archive member escapes destination: " + info.filename)
        bundle.extractall(destination)


def conda_path(prefix, system):
    return prefix / ("Scripts/conda.exe" if system == "Windows" else "bin/conda")


def find_conda(system):
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda"),
                  conda_path(EXTERNAL / "miniforge", system), conda_path(Path(sys.prefix), system),
                  conda_path(Path.home() / "miniforge3", system)]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and Path(candidate).suffix.lower() not in {".bat", ".cmd"}:
            return Path(candidate).resolve()
    return None


def bootstrap_conda(system, machine):
    existing = find_conda(system)
    if existing:
        return existing
    prefix = EXTERNAL / "miniforge"
    if prefix.exists():
        raise RuntimeError(f"Incomplete Miniforge installation: {prefix}. Move it aside and retry.")
    # Unversioned release aliases have no matching .sha256 asset. Pin both
    # downloads to one release so cached installers cannot drift from checksums.
    filename = f"Miniforge3-{MINIFORGE_VERSION}-{'Windows' if system == 'Windows' else 'MacOSX'}-{machine}.{'exe' if system == 'Windows' else 'sh'}"
    url = f"https://github.com/conda-forge/miniforge/releases/download/{MINIFORGE_VERSION}/" + filename
    installer = download(url, EXTERNAL / "downloads" / filename)
    checksum = download(url + ".sha256", installer.with_suffix(installer.suffix + ".sha256"))
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    if digest != checksum.read_text().split()[0].lower():
        raise RuntimeError(f"Miniforge checksum mismatch. Remove {installer} and {checksum}, then retry.")
    if system == "Windows":
        # NSIS requires /D to be last, unquoted, even when the path has spaces.
        command = subprocess.list2cmdline([str(installer), "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S"])
        subprocess.run(command + " /D=" + str(prefix), check=True)
    else:
        run(["bash", installer, "-b", "-p", prefix])
    result = conda_path(prefix, system)
    if not result.is_file():
        raise RuntimeError("Miniforge did not create " + str(result))
    return result


def conda_environment(system, machine):
    environment = os.environ.copy()
    environment["CONDA_SUBDIR"] = {("Windows", "x86_64"): "win-64", ("Darwin", "x86_64"): "osx-64", ("Darwin", "arm64"): "osx-arm64"}[(system, machine)]
    return environment


def python_command(conda, prefix, *args):
    return [conda, "run", "--no-capture-output", "--prefix", prefix, "python", *args]


def torch_packages(system, machine, backend):
    if system == "Darwin":
        if backend == "cuda":
            raise RuntimeError("CUDA is available only in the Windows installer. macOS uses CPU/MPS.")
        return ["torch==2.2.2" if machine == "x86_64" else "torch==2.11.0"]
    return ["torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/" + ("cu124" if backend == "cuda" else "cpu")]


def install_cgal(conda, prefix, system, machine):
    probe = "from CGAL.CGAL_Kernel import Point_3; from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization; print(Point_3(1,2,3))"
    if run(python_command(conda, prefix, "-c", probe), check=False).returncode == 0:
        return
    filename = "CGAL.zip" if system == "Windows" else "CGAL_macos_" + ("intel" if machine == "x86_64" else "arm") + ".zip"
    local = ROOT / filename
    archive = local if local.is_file() else download(f"{RELEASES}/{'0.1' if system == 'Windows' else '0.2'}/{filename}", EXTERNAL / "downloads" / filename)
    with tempfile.TemporaryDirectory(dir=EXTERNAL) as staging:
        extract_zip(archive, staging)
        package = Path(staging) / "CGAL"
        if not (package / "CGAL_Kernel.py").is_file():
            # The Windows 0.1 archive is flat; macOS archives include CGAL/.
            package = Path(staging)
        if not (package / "CGAL_Kernel.py").is_file():
            raise RuntimeError("CGAL archive must contain CGAL_Kernel.py at its root or inside CGAL/")
        destination = ROOT / "CGAL"
        backup = ROOT / f"CGAL.backup.{time.time_ns()}"
        if destination.exists():
            destination.rename(backup)
        try:
            shutil.move(str(package), destination)
            run(python_command(conda, prefix, "-c", probe))
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            if backup.exists():
                backup.rename(destination)
            raise


def install_vsot():
    destination = EXTERNAL / "VSOT"
    if not destination.exists():
        archive = download(VSOT_URL, EXTERNAL / "downloads" / "VSOT-main.zip")
        with tempfile.TemporaryDirectory(dir=EXTERNAL) as staging:
            extract_zip(archive, staging)
            source = Path(staging) / "VSOT-main"
            if not (source / "src/+comSeg").is_dir():
                raise RuntimeError("VSOT archive is missing src/+comSeg")
            shutil.move(str(source), destination)
    for relative in ["src/+comSeg", "src/misc", "resources/edt_mex/edt_mex", "resources/graph_related/graph_mex", "resources/src_mex/mex_EM_analysis/mex_EM_analysis"]:
        if not (destination / relative).is_dir():
            raise RuntimeError(f"Incomplete VSOT installation: {destination / relative}")
    return destination


def install_octave(conda, prefix, system, machine, supplied=None):
    if supplied:
        executable = Path(supplied).expanduser().resolve()
    elif system == "Darwin":
        octave_prefix = EXTERNAL / "octave-env"
        if not (octave_prefix / "bin/octave-cli").is_file():
            run([conda, "create", "--prefix", octave_prefix, "--override-channels", "-c", "conda-forge", "octave=10.3", "-y"], env=conda_environment(system, machine))
        executable = ROOT / "scripts/octave-cli"
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    else:
        destination = EXTERNAL / "octave-windows"
        if not destination.exists():
            archive = download(OCTAVE_URL, EXTERNAL / "downloads" / "octave-10.3.0-w64.zip")
            with tempfile.TemporaryDirectory(dir=EXTERNAL) as staging:
                extract_zip(archive, staging)
                if not list(Path(staging).rglob("octave-cli.exe")):
                    raise RuntimeError("Octave archive has no octave-cli.exe")
                shutil.move(staging, destination)
        candidates = sorted(destination.rglob("octave-cli.exe"))
        if len(candidates) != 1:
            raise RuntimeError("Cannot locate a unique octave-cli.exe; use --octave PATH.")
        executable = candidates[0]
        # Register the prebuilt packages shipped in the portable Windows ZIP.
        # This is the package-registration part of upstream post-install.bat.
        run([executable, "--no-gui", "--quiet", "--eval", "pkg rebuild -global;"])
    if not executable.is_file():
        raise RuntimeError("Octave executable not found: " + str(executable))
    check = [executable, "--no-gui", "--quiet", "--no-line-editing", "--eval", "pkg load image; disp('IMAGE_PACKAGE_OK');"]
    if run(check, check=False).returncode:
        if system == "Darwin" and not supplied:
            if subprocess.run(["xcrun", "--find", "clang"], capture_output=True).returncode:
                raise RuntimeError("Apple command-line tools are needed. Run xcode-select --install, then rerun install.py.")
            run(python_command(conda, prefix, str(ROOT / "scripts/install_octave_image.py")))
        else:
            run([executable, "--no-gui", "--quiet", "--eval", "pkg install -forge image;"])
        run(check)
    return executable


def install_models(archive):
    destination = ROOT / "plugins/ai_segmentation/models"
    with tempfile.TemporaryDirectory(dir=EXTERNAL) as staging:
        extract_zip(archive, staging)
        sources = [Path(staging), Path(staging) / "models"]
        source = next((p for p in sources if all((p / f"stage_{i}").is_dir() for i in range(1, 5))), None)
        if source is None:
            raise RuntimeError("Models ZIP must contain stage_1 through stage_4, optionally inside models/.")
        # Preserve existing weights; stop before copying if any file conflicts.
        files = [(p, destination / p.relative_to(source)) for i in range(1, 5) for p in (source / f"stage_{i}").rglob("*") if p.is_file()]
        for original, target in files:
            if target.exists() and (not target.is_file() or not filecmp.cmp(original, target, shallow=False)):
                raise RuntimeError(f"Model file already exists: {target}. Existing weights were preserved.")
        for original, target in files:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)


def verify(conda, prefix, octave=None):
    run(python_command(conda, prefix, "-m", "pip", "check"))
    run(python_command(conda, prefix, "-c", "import numpy, torch, easy3d, maxflow; from application.application import Application; from CGAL.CGAL_Kernel import Point_3; print('Application imports OK', numpy.__version__, torch.__version__, Point_3(0,0,0)); print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"))
    if octave:
        code = "import os, sys; os.environ['OCTAVE_EXECUTABLE']=sys.argv[1]; os.environ['PATH']=os.path.dirname(sys.argv[1])+os.pathsep+os.environ.get('PATH',''); from oct2py import Oct2Py; oc=Oct2Py(timeout=60); oc.eval('pkg load image'); assert oc.eval('1+1') == 2; oc.exit(); print('Python/Octave connection OK')"
        run(python_command(conda, prefix, "-c", code, str(octave)))


def configure(conda, prefix, vsot, octave):
    values = {"plugin_repo_root": str(ROOT / "plugins/ai_segmentation"), "models_folder": str(ROOT / "plugins/ai_segmentation/models")}
    if vsot:
        values.update(vsot_root=str(vsot), vsot_matlab_dir=str(vsot / "src/+comSeg"), octave_executable=str(octave))
    # Settings are scoped to the target Python environment. Keep user overrides.
    code = "from pathlib import Path; import json,sys; from projects.neural_segmentation.settings import NeuralSegmentationSettings; s=NeuralSegmentationSettings.instance(); s.load(); v=json.loads(sys.argv[1]); s.update({k:v for k,v in v.items() if not s.state.get(k)}); Path(s.FILE_NAME).parent.mkdir(parents=True,exist_ok=True); s.save(); print('Settings:', s.FILE_NAME)"
    run(python_command(conda, prefix, "-c", code, json.dumps(values)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=EXTERNAL / "python-env", help="Conda environment (default: external/python-env)")
    parser.add_argument("--models", type=Path, help="ZIP with stage_1 through stage_4 model weights")
    parser.add_argument("--torch-backend", choices=["cpu", "cuda"], default="cpu", help="Windows PyTorch variant; macOS installs CPU/MPS automatically")
    parser.add_argument("--skip-vsot", action="store_true", help="Skip Octave/VSOT (full stage 4 will be unavailable)")
    parser.add_argument("--octave", help="Use an existing Octave executable instead of downloading it")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without downloads or changes")
    parser.add_argument("--check", action="store_true", help="Check the selected environment without installing")
    args = parser.parse_args(argv)
    system, machine = target_platform()
    prefix = args.prefix.expanduser().resolve()
    torch_args = torch_packages(system, machine, args.torch_backend)
    print(f"SpineTool2: {system} {machine}\nEnvironment: {prefix}\nPyTorch: {' '.join(torch_args)}")
    print("Plan: Python 3.10, application/AI dependencies, Easy3D, CGAL" + (", Octave 10.3 + image, VSOT" if not args.skip_vsot else ""))
    if args.dry_run:
        print("Miniforge will be downloaded if conda is missing. Existing settings and models are preserved.\nRun after installation: python start.py")
        return 0
    if args.check:
        conda = find_conda(system)
        if not conda:
            raise RuntimeError("Conda not found; run install.py first.")
        verify(conda, prefix, args.octave)
        return 0
    if args.models and not args.models.is_file():
        raise RuntimeError("Models archive not found: " + str(args.models))
    EXTERNAL.mkdir(exist_ok=True)
    conda = bootstrap_conda(system, machine)
    python = prefix / ("python.exe" if system == "Windows" else "bin/python")
    if not python.is_file():
        if prefix.exists():
            raise RuntimeError(f"Incomplete environment at {prefix}; move it aside and retry.")
        run([conda, "create", "--prefix", prefix, "--override-channels", "-c", "conda-forge", "python=3.10", "pip", "-y"], env=conda_environment(system, machine))
    run(python_command(conda, prefix, "-c", f"import sys,platform; assert sys.version_info[:2] == (3,10), 'Python 3.10 required'; assert platform.machine().lower() in {['amd64', 'x86_64'] if machine == 'x86_64' else ['arm64', 'aarch64']!r}, 'Wrong environment architecture'"))
    pip = python_command(conda, prefix, "-m", "pip", "install")
    run([*pip, "-r", ROOT / "pip_requirements.txt"])
    run([*pip, *torch_args])
    run([*pip, "-r", ROOT / "plugins/ai_segmentation/requirements.txt"])
    wheel = "easy3d-2.6.1-cp310-cp310-" + ("win_amd64.whl" if system == "Windows" else "macosx_11_0_universal2.whl")
    wheel_source = ROOT / wheel
    run([*pip, wheel_source if wheel_source.is_file() else "https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/" + wheel])
    install_cgal(conda, prefix, system, machine)
    vsot = octave = None
    if not args.skip_vsot:
        vsot = install_vsot()
        octave = install_octave(conda, prefix, system, machine, args.octave)
    if args.models:
        install_models(args.models)
    verify(conda, prefix, octave)
    configure(conda, prefix, vsot, octave)
    manifest = EXTERNAL / "installation.json"
    manifest.write_text(json.dumps({"conda": str(conda), "prefix": str(prefix), "octave": str(octave) if octave else None}, indent=2), encoding="utf-8")
    print("\nInstallation verified. Start with: python start.py")
    model_root = ROOT / "plugins/ai_segmentation/models"
    missing = [str(i) for i in range(1, 5) if not any(p.is_file() and p.suffix.lower() in {".ckpt", ".pth", ".pt", ".onnx"} for p in (model_root / f"stage_{i}").glob("*"))]
    if missing:
        print("AI weights still needed for stages " + ", ".join(missing) + ". Download: " + MODELS_URL)
        print('Then rerun: python install.py --models "/path/to/models.zip"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print("\nInstallation failed: " + str(error) + "\nFix the reported problem and rerun the same command.", file=sys.stderr)
        raise SystemExit(1)
