# Helper scripts

## Neural Batch GUI

[`neural_batch_gui.py`](neural_batch_gui.py) is a standalone graphical utility for sequential batch AI segmentation of TIFF images. It supports multiple regions of interest (ROIs) per image, common or individual XYZ voxel sizes, CPU/CUDA selection, progress and logs, stopping with saved results, and resuming an unfinished batch.

Run from the repository root in the SpineTool environment after installing the dependencies and AI models described in the [main README](../README.md):

```bash
python run_neural_batch.py
```

Select TIFF files, optionally draw ROIs, set voxel sizes in micrometers, choose an output folder and processing stages, then click **Run**. Processing stops at stage 3 by default; stage 4 additionally requires its model, VSOT, and Octave. Saved projects can be opened in SpineTool2 for further editing.

See the [GUI guide](../docs/neural_batch_gui.md) for detailed instructions and the [CLI guide](../docs/neural_cli.md) for using [`create_neural_project.py`](create_neural_project.py) from the command line.
