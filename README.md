# SpineTool2

**Windows** is the primary platform. Instructions for macOS Intel and Apple Silicon are provided in a separate section below.

## Windows — primary platform

### Install
1. Download code
2. Download and Unzip [CGAL.zip](https://github.com/Biomed-imaging-lab/SpineTool2/releases/download/0.1/CGAL.zip) next to code, e.g. `PATH_TO_CODE\CGAL\...`
3. Install [Anaconda](https://www.anaconda.com/)
4. Open Anaconda
5. Execute
```cmd
cd PATH_TO_CODE
conda create --name spinetool2 --file conda_requirements.txt -y
conda activate spinetool2
pip install -r pip_requirements.txt
pip install -r plugins/ai_segmentation/requirements.txt 
pip install easy3d-2.6.1-cp310-cp310-win_amd64.whl
```
6. if use cuda, execute
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall
```
in not use cuda, execute
```cmd
pip install torch
```

### Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate spinetool2
python run.py
```

### Run AI segmentation and VSOT segmentation
1. install gnu octave: download https://octave.org/download zip, extract
2. clone vsot main branch https://github.com/yu-lab-vt/VSOT 
3. to run predictions faster, install cuda driver + cuda toolkit
4. download stage 1 to 4 [segmentation models](https://drive.google.com/file/d/1Sq58lruLRNGSK9YELmpDl3Ar8qTT49lR/view?usp=drive_link) and place to  plugins\ai_segmentation\models\stage_1\,  plugins\ai_segmentation\models\stage_2\, plugins\ai_segmentation\models\stage_3\, plugins\ai_segmentation\models\stage_4\
5. run
```cmd
cd PATH_TO_CODE
conda activate spinetool2
python run.py
```
6. Set Neural Segmentation settings to your VSOT root (`PATH_TO_VSOT_ROOT`), its MATLAB directory (`PATH_TO_VSOT_ROOT\src\+comSeg`), and your Octave executable (`PATH_TO_OCTAVE\bin\octave-cli.exe`). Replace these placeholders with your installation paths.


### Common errors

#### CUDA usage

RuntimeError: CUDA requested but is not available
steps:
1. check cuda availability manually
```
conda activate spinetool2
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```
2. result should be like:
```
2.x.x+cu...
12.x
True
NVIDIA GeForce RTX 2060
```
3. if output contains None/ False - check torch version, it should have postfix `cu`, like `2.4.1+cu124`, reinstall torch (switch off vpn)


RuntimeError: Numpy is not available
steps:
1. run 
```cmd
pip install --force-reinstall "numpy==1.26.4"
```

---

## macOS: Intel and Apple Silicon

Python **3.10** is required. Python, CGAL, and all other native extensions must use the same architecture.

| Mac | Python / CGAL | Miniforge |
| --- | --- | --- |
| Intel | `x86_64` | `Miniforge3-MacOSX-x86_64.sh` |
| Apple Silicon, native | `arm64` | `Miniforge3-MacOSX-arm64.sh` |
| Apple Silicon, through Rosetta | `x86_64` throughout the environment | Intel installer; do not mix with ARM CGAL |

### 1. Get the project

```bash
git clone https://github.com/Biomed-imaging-lab/SpineTool2.git
cd SpineTool2
```

If you already have the project, replace `PATH_TO_CODE` with its root directory:

```bash
cd PATH_TO_CODE
```

Run the following commands from the project root.

### 2. Install Miniforge and create an environment

Installers are available in the [official Miniforge repository](https://github.com/conda-forge/miniforge).
If conda is already installed, you do not need to run the installer again.

For Intel, download `Miniforge3-MacOSX-x86_64.sh`; for Apple Silicon, download `Miniforge3-MacOSX-arm64.sh`. Run the command for your architecture:

```bash
bash ~/Downloads/Miniforge3-MacOSX-x86_64.sh
# On Apple Silicon, use this instead:
# bash ~/Downloads/Miniforge3-MacOSX-arm64.sh
```

Open a new terminal after installation. If `conda` is not found:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Create the environment if it does not exist:

```bash
conda create --name spinetool2 -c conda-forge --file conda_macos_requirements.txt -y
conda activate spinetool2
python -c 'import sys, platform; print(sys.executable); print(sys.version); print(platform.machine())'
```

`conda_requirements.txt` contains Windows packages and Windows build identifiers. Use `conda_macos_requirements.txt` on macOS.
Do not set `CONDA_SUBDIR=osx-64` when creating a native ARM environment. If it remains set from another build, run `unset CONDA_SUBDIR` before creating the environment.

### 3. Install Python dependencies and PyTorch

The pip dependency lists are shared with Windows. The `PyQt5-Qt5==5.15.2` pin applies only to Windows; on macOS, PyQt5 selects a compatible Qt wheel for the architecture. Install PyTorch separately for your architecture and backend.

```bash
python -m pip install -r pip_requirements.txt
# Intel / x86_64 (CPU):
python -m pip install 'torch==2.2.2'
# Apple Silicon / native arm64 (MPS): use this instead of the Intel command:
# python -m pip install --upgrade 'torch==2.11.0'
python -m pip install -r plugins/ai_segmentation/requirements.txt
```

PyTorch 2.2.x is the last official release series supporting macOS Intel ([PyTorch announcement](https://pytorch.org/blog/pytorch2-2/)). Keep 2.2.2 for Intel CPU inference. For Apple Silicon MPS, use a native `arm64` environment and the 2.11.0 command above. The ARM/PyTorch 2.11.0 combination has not yet been validated with the full application.

#### MPS setup and verification

MPS is included in the macOS PyTorch package; there is no separate MPS pip package or CUDA toolkit to install. For the documented PyTorch 2.11.0 setup, Apple lists Apple Silicon, macOS 14.0 or later, Python 3.10 or later, and Xcode command-line tools ([Apple installation guide](https://developer.apple.com/metal/pytorch/)). Install the tools if missing:

```bash
xcode-select --install
```

If you previously installed PyTorch 2.2.2 on Apple Silicon, activate `spinetool2` and run the ARM upgrade command above. This project's models use 3D convolutions, which [PyTorch 2.2.2 does not support on MPS](https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/mps/operations/Convolution.mm#L70), even when MPS is reported as available.

Check using the same interpreter configured as **AI python executable**:

```bash
python - <<'PYTHON'
import platform
import torch
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    with torch.inference_mode():
        layer = torch.nn.Conv3d(1, 2, kernel_size=3).to("mps")
        result = layer(torch.ones(1, 1, 8, 8, 8, device="mps"))
        print("Conv3d check:", result.cpu().shape)
PYTHON
```

Both inference entry points accept `cpu`, `cuda`, `mps`, and `auto` in the request's `device` field. On macOS, `auto` selects MPS when available, otherwise CPU. Explicit `mps` raises an error if unavailable; explicit `cpu` stays on CPU. This availability fallback does not catch unsupported model operations. A successful check above does not replace full model validation.

Enable **Use GPU acceleration** in the project interface to send `device="auto"`: macOS uses MPS when available, other platforms use CUDA when available, and otherwise inference uses CPU. Uncheck it to force CPU. Saved `cuda` or `mps` choices are displayed as enabled acceleration and normalized to `auto` when opening the project. Keep CUDA mixed precision disabled on macOS; that option applies only to CUDA.

### 4. Install Easy3D for macOS

The Windows wheel in the repository root is not compatible with macOS. For Intel and ARM, use the official Python 3.10 `universal2` wheel from the [Easy3D 2.6.1 release](https://github.com/LiangliangNan/Easy3D/releases/tag/v2.6.1):

```bash
python -m pip install 'https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/easy3d-2.6.1-cp310-cp310-macosx_11_0_universal2.whl'
python -c 'import easy3d; print(easy3d.SurfaceMeshIO, easy3d.SurfaceMeshHoleFilling)'
```

### 5. Install CGAL for Python 3.10 and your architecture

The CGAL archive in the Windows instructions cannot be used on a Mac. You need a macOS build of [cgal-swig-bindings](https://github.com/DariaWelt/cgal-swig-bindings).

The build must contain `.py` wrappers, **26 `_CGAL_*.so` extensions**, internal CGAL libraries, and their `.dylib` dependencies. The `.cxx` files are source files and cannot replace the `.so` extensions.

Extract the prepared macOS packages into `external/cgal/intel` and `external/cgal/arm`, relative to the project root. Each package should have this layout:

```text
external/cgal/<variant>/lib/python3.10/site-packages/CGAL/
external/cgal/<variant>/lib/*.dylib
```

If you build CGAL yourself, follow the build instructions in the [cgal-swig-bindings repository](https://github.com/DariaWelt/cgal-swig-bindings). To make skeletonization compatible with `Polylines`, add `%import "SWIG_CGAL/Polygon_mesh_processing/CGAL_Polygon_mesh_processing.i"` after `%include "std_vector.i"` in `SWIG_CGAL/User_packages/Surface_mesh_skeletonization/CGAL_Surface_mesh_skeletonization.i` before building.

For packages built without that fix, place the corrected `_CGAL_Surface_mesh_skeletonization.so` in `external/cgal-fixes/intel` or `external/cgal-fixes/arm`. The corrected extension must match the package's architecture and Python version. If your package already includes the fix, skip the final copy command below.

Run these commands from the project root:

```bash
ARCH=$(python -c 'import platform; print(platform.machine())')
case "$ARCH" in
  x86_64) VARIANT=intel ;;
  arm64) VARIANT=arm ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
CGAL_DIST="./external/cgal/$VARIANT"
# Back up the existing package before replacing files.
if [ -d CGAL ]; then cp -R CGAL "CGAL.backup.$(date +%Y%m%d-%H%M%S)"; fi
mkdir -p CGAL
cp -p "$CGAL_DIST"/lib/python3.10/site-packages/CGAL/*.py CGAL/
cp -p "$CGAL_DIST"/lib/python3.10/site-packages/CGAL/*.so CGAL/
cp -p "$CGAL_DIST"/lib/*.dylib CGAL/
# Apply only if your package needs the separate Polylines fix:
# cp -p "./external/cgal-fixes/$VARIANT/_CGAL_Surface_mesh_skeletonization.so" CGAL/
python -c 'from CGAL.CGAL_Kernel import Point_3; print(Point_3(1, 2, 3))'
```

This package uses library search paths relative to the extensions. Copying files alone may not be sufficient for another build: inspect its dependencies with `otool -L CGAL/_CGAL_Kernel.so` and its architecture with `file CGAL/_CGAL_Kernel.so`.

### 6. Check the environment and run tests

```bash
python -m pip check
python -c 'import numpy, torch, easy3d; from CGAL.CGAL_Kernel import Point_3; print(numpy.__version__, torch.__version__, Point_3(0, 0, 0)); print("MPS:", torch.backends.mps.is_available())'
PYTHONPATH="$PWD" python -m unittest discover -s tests -t tests --quiet
```

In PyCharm, select the Python interpreter from `spinetool2`, set the working directory to the project root, and enable adding content roots to PYTHONPATH.

### 7. Start the application

```bash
conda activate spinetool2
cd PATH_TO_CODE
python run.py
```

Replace `PATH_TO_CODE` with your project directory. Run the application from a regular terminal with access to the desktop. Qt startup cannot be verified without access to a display.

A portable `projects/segmentation/utils/data_processing/graph.py` adapter was added because the missing original `graph` module prevented the application from importing. It reuses the existing A* implementation in `paired_necks.py`, searches 26 neighbouring voxels, accounts for intensity, and returns a path in voxel coordinates. The original extension was unavailable for comparison; segmentation on real data still needs separate validation.

## AI segmentation and VSOT on macOS

These additional files are not required to open the main window. Full inference requires model weights, VSOT, and Octave.

1. Install Octave in a separate environment from the project root. This keeps the application environment unchanged. Install Apple's command-line tools first if `clang` is unavailable:

   ```bash
   xcode-select --install
   ```

   Create the Octave environment and build its `image` package:

   ```bash
   mkdir -p external
   conda create --prefix "$PWD/external/octave-env" -c conda-forge --override-channels octave=10.3 -y
   conda activate spinetool2
   python scripts/install_octave_image.py
   ./scripts/octave-cli --no-gui --quiet --eval 'pkg load image; disp("image package OK")'
   ```

   `scripts/octave-cli` sets `OCTAVE_HOME` and `OCTAVE_EXEC_HOME` explicitly and disables line editing. This avoids padded paths and an Oct2Py/readline hang observed with the conda macOS package. The installer removes embedded NUL bytes from compiler flags and uses Apple Clang, C++17, and the environment's OpenMP runtime. Run it with the same architecture as the Octave environment. This installation was tested on Intel; native ARM execution still requires verification.

   Homebrew is an alternative listed on the [Octave download page](https://www.octave.org/download). For an existing Homebrew installation, use its `octave-cli` executable and install/load `image` there; the conda wrapper above is specific to `external/octave-env`.

2. Clone VSOT from the project root:

   ```bash
   mkdir -p external
   git clone --branch main https://github.com/yu-lab-vt/VSOT external/VSOT
   ```

3. Download and extract the [model weights for stages 1–4](https://drive.google.com/file/d/1Sq58lruLRNGSK9YELmpDl3Ar8qTT49lR/view?usp=drive_link). Preserve the directory structure and filenames:

   If the downloaded archive is `models.zip` with `stage_1` through `stage_4` at its root, run:

   ```bash
   mkdir -p plugins/ai_segmentation/models
   unzip /path/to/models.zip -d plugins/ai_segmentation/models
   ```

   Replace `/path/to/models.zip` with the archive location. Google Drive may require signing in through a browser.

   ```text
   plugins/ai_segmentation/models/stage_1/
   plugins/ai_segmentation/models/stage_2/
   plugins/ai_segmentation/models/stage_3/
   plugins/ai_segmentation/models/stage_4/
   ```

4. Resolve paths from the project root for Neural Segmentation settings:

   ```bash
   PROJECT_ROOT="$PWD"
   printf 'AI runtime path: %s/plugins/ai_segmentation\n' "$PROJECT_ROOT"
   printf 'AI models folder: %s/plugins/ai_segmentation/models\n' "$PROJECT_ROOT"
   printf 'VSOT root: %s/external/VSOT\n' "$PROJECT_ROOT"
   printf 'VSOT matlab dir: %s/external/VSOT/src/+comSeg\n' "$PROJECT_ROOT"
   python -c 'import sys; print("AI python executable:", sys.executable)'
   printf 'Octave executable: %s/scripts/octave-cli\n' "$PROJECT_ROOT"
   ```

   Enter the resolved paths in the settings fields. The locations relative to the project root are:

   | Setting | Location |
   | --- | --- |
   | AI runtime path | `./plugins/ai_segmentation` |
   | AI models folder | `./plugins/ai_segmentation/models` |
   | AI python executable | The interpreter path printed above, or leave blank to use the current Python |
   | VSOT root | `./external/VSOT` |
   | VSOT matlab dir | `./external/VSOT/src/+comSeg` |
   | Octave executable | The resolved path to `./scripts/octave-cli` for the conda setup above |

   For the conda setup, select the wrapper, not the binary inside `external/octave-env/bin`. For Homebrew, use the full path returned by `command -v octave-cli`, without `.exe`.

5. Run `python run.py`. Enable **Use GPU acceleration** for automatic MPS selection on a supported Mac, or disable it to use CPU. Keep CUDA mixed precision disabled on macOS, then start the required stage. Check MPS first as described in **MPS setup and verification** above.

VSOT includes MATLAB/MEX components; Windows MEX binaries do not run on macOS. `stage4_vsot.py` already provides an Octave compatibility backend for some missing MATLAB functions. This does not establish that the complete VSOT pipeline works on all data: run a full validation after installing the models and Octave.

## macOS verification status

- Intel `x86_64`, Python 3.10.21, NumPy 1.26.4, PyTorch 2.2.2.
- Installed environment dependencies are consistent: `pip check` passes.
- Easy3D 2.6.1 is installed from the official macOS wheel; the required classes are available.
- All 26 CGAL modules import successfully; the skeletonization test passes with the `Polylines` fix.
- All 25 project tests pass, including CGAL compatibility, platform shortcuts, and project path portability checks.
- The `SpineTool2.0` main window is created and visible; the verification instance closed automatically.
- The ARM CGAL fix binary was checked statically; the ARM application has not been tested at runtime.
- All four model checkpoints load and execute a small forward pass on CPU. Stage 1 CLI inference writes a probability TIFF from a synthetic volume.
- Octave 10.3, `image` 2.20.1, and the Python–Octave connection are installed and verified. A synthetic 48³ cylinder passes the VSOT adapter and writes shaft/spine TIFFs using the built-in compatibility backend.
- Full AI/VSOT inference on microscopy data and ARM runtime execution have not been verified.

## Common errors on macOS

- `cannot import name '_CGAL_Kernel'`: check that the `.so` extensions are present, Python is 3.10, and architectures match. A local `CGAL` directory can shadow the package installed in the environment.
- `Library not loaded`: check that all required `.dylib` files are copied and library search paths are portable.
- `No module named 'CGAL'` when running tests from `tests`: set `PYTHONPATH` to the project root as shown above.
- `CUDA requested but is not available`: use **Use GPU acceleration** for automatic backend selection, or use `auto` / `mps` in an inference request; the Windows CUDA commands do not apply to a Mac.
- `MPS requested but is not available`: run the MPS check above with the configured AI interpreter and verify the macOS version, native ARM architecture, and PyTorch installation. Use `cpu` if MPS is unavailable.
- `Conv3D is not supported on MPS`: on Apple Silicon, replace PyTorch 2.2.2 with the documented ARM version above. For Intel with PyTorch 2.2.2, use CPU.
- `Numpy is not available`: keep NumPy 1.26.4 when using PyTorch 2.2.2:

  ```bash
  python -m pip install --force-reinstall 'numpy==1.26.4'
  ```

- `invalid escape sequence` in Qt widgets is a warning about regular-expression strings in Python; it did not prevent the verified startup.

## Moving projects between Windows and macOS

Copy the entire project folder and open `project_description.json` from its new location. Files inside the project are saved with relative paths. Windows separators, historic `/layers/...` paths, and absolute paths beneath an older recorded project root are supported. Paths clipped by older macOS saves are recovered when they match the recorded root.

External files must remain available or be selected again at their new location. Configure Python, model weights, CGAL, and Octave for the destination machine and architecture. Path portability is covered by tests on macOS; a complete transfer and execution on Windows has not been verified.
