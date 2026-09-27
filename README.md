# SpineTool2

SpineTool2 is a desktop application for viewing, processing, and segmenting 3D microscopy of dendrites and dendritic spines. Windows x64, macOS Intel, and macOS Apple Silicon are supported.

## Requirements

- 64-bit operating system.
- Internet connection during installation.
- Several GB of free disk space.
- Python 3.10 for manual installation. The automatic installer creates its own Python 3.10 environment.
- Optional: an NVIDIA GPU on Windows or Apple Silicon GPU acceleration through MPS on macOS.

## Quick start: automatic installation

Download or clone the repository, open a terminal in its root directory, and run:

```bash
python install.py
python start.py
```

On macOS, use `python3` if `python` is unavailable. The installer downloads Miniforge when needed, creates an isolated environment, installs the dependencies, selects the correct Easy3D and CGAL packages, and configures Octave and VSOT.

To use AI segmentation, download the [stage 1–4 model archive](https://drive.google.com/file/d/1Sq58lruLRNGSK9YELmpDl3Ar8qTT49lR/view?usp=drive_link), then install it with:

```bash
python install.py --models "/path/to/models.zip"
```

Windows uses CPU inference by default. For an NVIDIA GPU with a compatible driver, use:

```bash
python install.py --torch-backend cuda --models "C:\path\to\models.zip"
```

Useful options:

```bash
python install.py --dry-run       # Show the installation plan
python install.py --skip-vsot     # Install without Octave and VSOT
python install.py --check         # Verify the installed environment
```

After a successful installation, always start the application from the repository root:

```bash
python start.py
```

## Example data

The [`example_data`](example_data) directory contains files for trying the main workflows and creating your first projects:

- [`example_zstack.tif`](example_data/example_zstack.tif) — a TIFF image for semi-automatic segmentation or AI segmentation;
- [`example_mesh.off`](example_data/example_mesh.off) — a surface mesh for semi-automatic mesh segmentation.

Start SpineTool2, select **File → Create new project**, choose **segmentation** for a semi-automatic project or **neural_segmentation** for an AI project, and select the corresponding example file as the original image. Save the project to a separate folder so the example files remain unchanged and can be reused.

## Manual installation

### Windows x64

1. Install [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda.
2. Extract the bundled `CGAL.zip` into the repository root. If the archive is missing, download it from the [SpineTool2 release](https://github.com/Biomed-imaging-lab/SpineTool2/releases/download/0.1/CGAL.zip).
3. Open a terminal in the repository root and run:

```cmd
conda create --name spinetool2 --file conda_requirements.txt -y
conda activate spinetool2
python -m pip install -r pip_requirements.txt
python -m pip install -r plugins/ai_segmentation/requirements.txt
python -m pip install easy3d-2.6.1-cp310-cp310-win_amd64.whl
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python run.py
```

For NVIDIA CUDA 12.4, replace the PyTorch command with:

```cmd
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### macOS Intel

1. Install the Intel (`x86_64`) build of [Miniforge](https://github.com/conda-forge/miniforge).
2. Download [CGAL_macos_intel.zip](https://github.com/Biomed-imaging-lab/SpineTool2/releases/download/0.2/CGAL_macos_intel.zip) and extract the `CGAL` folder into the repository root.
3. Run:

```bash
conda create --name spinetool2 -c conda-forge --file conda_macos_requirements.txt -y
conda activate spinetool2
python -m pip install -r pip_requirements.txt
python -m pip install -r plugins/ai_segmentation/requirements.txt
python -m pip install 'torch==2.2.2'
python -m pip install 'https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/easy3d-2.6.1-cp310-cp310-macosx_11_0_universal2.whl'
python run.py
```

### macOS Apple Silicon

1. Install the ARM64 build of [Miniforge](https://github.com/conda-forge/miniforge). Do not mix ARM64 and Intel/Rosetta packages.
2. Download [CGAL_macos_arm.zip](https://github.com/Biomed-imaging-lab/SpineTool2/releases/download/0.2/CGAL_macos_arm.zip) and extract the `CGAL` folder into the repository root.
3. Run:

```bash
unset CONDA_SUBDIR
conda create --name spinetool2 -c conda-forge --file conda_macos_requirements.txt -y
conda activate spinetool2
python -m pip install -r pip_requirements.txt
python -m pip install -r plugins/ai_segmentation/requirements.txt
python -m pip install 'torch==2.11.0'
python -m pip install 'https://github.com/LiangliangNan/Easy3D/releases/download/v2.6.1/easy3d-2.6.1-cp310-cp310-macosx_11_0_universal2.whl'
python run.py
```

## Manual AI segmentation setup

The automatic installer performs these steps for you. For a manual installation:

1. Extract the [model archive](https://drive.google.com/file/d/1Sq58lruLRNGSK9YELmpDl3Ar8qTT49lR/view?usp=drive_link) so that the repository contains:

```text
plugins/ai_segmentation/models/stage_1/
plugins/ai_segmentation/models/stage_2/
plugins/ai_segmentation/models/stage_3/
plugins/ai_segmentation/models/stage_4/
```

2. Install [GNU Octave](https://octave.org/download) with the `image` package.
3. Clone [VSOT](https://github.com/yu-lab-vt/VSOT).
4. In **Neural Segmentation settings**, set the AI models folder, VSOT root (`PATH_TO_VSOT_ROOT`), its MATLAB directory (`PATH_TO_VSOT_ROOT\src\+comSeg`), and your Octave executable (`PATH_TO_OCTAVE\bin\octave-cli.exe`). Replace these placeholders with your installation paths.

## Troubleshooting

### CGAL does not import

Make sure that:

- you downloaded the CGAL archive for your operating system and CPU architecture;
- the extracted directory is `<SpineTool2>/CGAL`;
- the active environment uses Python 3.10;
- Python and CGAL use the same architecture (`x86_64` or `arm64`).

If the supplied archive is not compatible with your system, build CGAL bindings locally using [cgal-swig-bindings](https://github.com/CGAL/cgal-swig-bindings).

### CUDA is unavailable

Check the installed PyTorch build and driver:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The CUDA build has a `+cu...` suffix and `torch.cuda.is_available()` should print `True`. Reinstall the CUDA 12.4 PyTorch package from the Windows instructions if necessary.

### `RuntimeError: Numpy is not available`

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

### macOS installer asks for command-line tools

Install them and rerun the installer:

```bash
xcode-select --install
python3 install.py
```
