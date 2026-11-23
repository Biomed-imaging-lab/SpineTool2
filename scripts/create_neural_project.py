"""Run SpineTool AI stages 1-3 and optional shaft/spines for a ZYX TIFF."""
import argparse
from copy import deepcopy
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import signal
import threading
import queue
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probability_threshold(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("threshold must be in [0, 1]")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--voxel-size", nargs=3, type=float, required=True,
                   metavar=("Z", "Y", "X"), help="voxel size in micrometers, Z Y X")
    p.add_argument("--draw-roi", action="store_true", help="paint an XY ROI on a maximum projection")
    p.add_argument("--roi", type=Path, help="existing 2D XY or 3D ZYX binary TIFF")
    p.add_argument("--areas", nargs="+", type=Path, help="independent ROI masks; one project tree with a branch per ROI")
    p.add_argument("--segment-spines", "--segment_spines", dest="segment_spines", action="store_true",
                   help="also run shaft/spines stage 4 (default: stop at stage 3)")
    p.add_argument("--threshold", type=probability_threshold, help="override thresholds for stages 1-3")
    p.add_argument("--threshold-mode", choices=["trunk", "spine"], default="trunk")
    p.add_argument("--trunk-threshold", type=probability_threshold, default=0.5)
    p.add_argument("--spine-threshold", type=probability_threshold, default=0.8)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--clear-run", action="store_true", help="require an empty project directory; never delete files")
    p.add_argument("--settings", type=Path, help="neural_segmentation_settings.json (otherwise app settings)")
    p.add_argument("--models-dir", type=Path)
    for stage in range(1, 5):
        p.add_argument(f"--model-{stage}", type=Path)
    p.add_argument("--plugin-root", type=Path)
    p.add_argument("--python", dest="python_executable")
    p.add_argument("--vsot-root", type=Path)
    p.add_argument("--vsot-matlab-dir", type=Path)
    p.add_argument("--octave", type=Path)
    p.add_argument("--overlap-percent", type=int, choices=[20, 50, 75])
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--mixed-precision", action="store_true", default=None)
    precision.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    p.add_argument("--no-fill-holes", action="store_true")
    p.add_argument("--prune-length-um", type=float, default=4.0)
    return p


def paint_roi(image):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    mask = np.zeros(image.shape[1:], dtype=np.uint8)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.23)
    ax.imshow(image.max(axis=0), cmap="gray")
    overlay = ax.imshow(np.ma.masked_equal(mask, 0), cmap="autumn", vmin=0, vmax=1, alpha=0.5)
    ax.set_title("ROI through all Z: left = paint, right = erase; Apply to continue")
    radius = Slider(fig.add_axes([0.2, 0.1, 0.5, 0.03]), "Radius (px)", 1, 200, valinit=30)
    button = Button(fig.add_axes([0.75, 0.04, 0.18, 0.06]), "Apply")
    accepted = [False]
    previous = [None]

    def draw(event):
        if event.inaxes != ax or event.button not in (1, 3) or event.xdata is None:
            previous[0] = None
            return
        x, y = event.xdata, event.ydata
        old = previous[0] or (x, y)
        steps = max(1, int(np.hypot(x-old[0], y-old[1]) / max(1, radius.val / 2)))
        for cx, cy in zip(np.linspace(old[0], x, steps+1), np.linspace(old[1], y, steps+1)):
            r = radius.val
            x0, x1 = max(0, int(cx-r)), min(mask.shape[1], int(cx+r)+1)
            y0, y1 = max(0, int(cy-r)), min(mask.shape[0], int(cy+r)+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            patch = mask[y0:y1, x0:x1]
            patch[(xx-cx)**2 + (yy-cy)**2 <= r*r] = int(event.button == 1)
        previous[0] = (x, y)
        overlay.set_data(np.ma.masked_equal(mask, 0))
        fig.canvas.draw_idle()

    def apply(_):
        if mask.any():
            accepted[0] = True
            plt.close(fig)
    fig.canvas.mpl_connect("button_press_event", draw)
    fig.canvas.mpl_connect("motion_notify_event", draw)
    fig.canvas.mpl_connect("button_release_event", lambda _: previous.__setitem__(0, None))
    button.on_clicked(apply)
    plt.show()
    if not accepted[0]:
        raise RuntimeError("ROI drawing cancelled or empty; project was not created")
    return np.broadcast_to(mask, image.shape).copy()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def resolve_device(python, requested):
    """Probe the actual inference environment, which may differ from this CLI's."""
    if requested == "cpu":
        return "cpu"
    result = subprocess.run(
        [str(python), "-c", "import torch; print('SPINETOOL_CUDA=' + str(int(torch.cuda.is_available())))"],
        capture_output=True, text=True, check=True,
    )
    available = "SPINETOOL_CUDA=1" in result.stdout.splitlines()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable in the inference Python environment")
    return "cuda" if available else "cpu"


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def check_directory(folder, clear_run):
    if folder.exists() and (not folder.is_dir() or (clear_run and any(folder.iterdir()))):
        raise FileExistsError(f"--clear-run requires an empty project directory: {folder}")


def same_request(saved, current):
    # A change of hardware must not invalidate a completed prediction.
    ignored = {"device"}
    return {k: v for k, v in saved.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored
    }


def cached_probability(stage_dir, request, read_probability):
    """A successful response alone is insufficient: verify the TIFF as well."""
    try:
        saved = json.loads((stage_dir / "request.json").read_text(encoding="utf-8"))
        response = json.loads((stage_dir / "response.json").read_text(encoding="utf-8"))
        if not same_request(saved, request) or response.get("status") != "ok":
            return None
        return read_probability(stage_dir / "probability.tif")
    except (OSError, ValueError, KeyError):
        return None


def invalidate_downstream(folder, stage):
    for downstream in range(stage, 5):
        (folder / f"artifacts/model_runs/stage_{downstream}/response.json").unlink(missing_ok=True)


def stop_process_tree(process):
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            # Terminate using the owned process handle even if taskkill is unavailable.
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            pass
        try:
            # Descendants may ignore SIGTERM even when the group leader exited.
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_process(command, environment, log_path, stop_event, emit):
    """Read subprocess output without blocking cancellation on a silent model/Octave."""
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", env=environment, **options)
        lines = queue.Queue()

        def read():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            while True:
                if stop_event.is_set():
                    raise InterruptedError("Stopped by user; last valid project checkpoint saved")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                emit(line.rstrip())
            return process.wait()
        except BaseException:
            stop_process_tree(process)
            raise
        finally:
            reader.join(timeout=3)
            process.stdout.close()


def compatible_configuration(saved, current):
    """Allow extending a three-stage run to stage 4, with identical upstream inputs."""
    saved, current = deepcopy(saved), deepcopy(current)
    count = min(len(saved.get("models", [])), len(current.get("models", [])))
    saved["models"] = saved.get("models", [])[:count]
    current["models"] = current.get("models", [])[:count]
    if count < 4:
        for key in ("threshold_mode", "trunk_threshold", "spine_threshold", "fill_holes", "overlap",
                    "mixed_precision", "prune_length_um", "vsot", "matlab"):
            saved.pop(key, None)
            current.pop(key, None)
    else:
        for configuration in (saved, current):
            inactive = "trunk_threshold" if configuration.get("threshold_mode") == "spine" else "spine_threshold"
            configuration.pop(inactive, None)
    return saved == current


def prepare(args):
    target_stage = 4 if args.segment_spines else 3
    if any(not math.isfinite(s) or s <= 0 for s in args.voxel_size):
        raise ValueError("Voxel sizes must be finite and positive")
    if not math.isfinite(args.prune_length_um) or args.prune_length_um < 0:
        raise ValueError("Prune length must be finite and nonnegative")
    if args.roi and args.draw_roi:
        raise ValueError("Choose either --roi or --draw-roi")
    if args.areas and (args.roi or args.draw_roi):
        raise ValueError("Choose --areas or a single ROI input, not both")
    settings = {}
    settings_path = args.settings
    if settings_path is None:
        from utils.appdirs import config_dir
        settings_path = Path(config_dir()) / "neural_segmentation_settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    elif args.settings:
        raise FileNotFoundError(settings_path)
    plugin = (args.plugin_root or Path(settings.get("plugin_repo_root") or ROOT / "plugins/ai_segmentation")).resolve()
    models = args.models_dir or Path(settings.get("models_folder") or plugin / "models")
    weights = []
    for stage in range(1, target_stage + 1):
        model = getattr(args, f"model_{stage}")
        if model is None:
            candidates = sorted(p for p in (models / f"stage_{stage}").rglob("*")
                                if p.suffix.lower() in (".ckpt", ".pth", ".pt", ".onnx"))
            if len(candidates) != 1:
                raise ValueError(f"Stage {stage}: found {len(candidates)} weights; specify --model-{stage}")
            model = candidates[0]
        model = model.resolve(strict=True)
        if not model.is_file():
            raise ValueError(f"Model is not a file: {model}")
        weights.append(model)
    vsot = (args.vsot_root or Path(settings.get("vsot_root") or "")).resolve()
    matlab = (args.vsot_matlab_dir or Path(settings.get("vsot_matlab_dir") or vsot / "src/+comSeg")).resolve()
    octave = args.octave or settings.get("octave_executable")
    if target_stage == 4 and settings.get("stage4_overlap_percent", 75) not in (20, 50, 75) and args.overlap_percent is None:
        raise ValueError("Stage 4 overlap must be 20, 50 or 75 percent")
    if target_stage == 4 and args.device == "cpu" and (
            args.mixed_precision if args.mixed_precision is not None else settings.get("stage4_mixed_precision", False)):
        raise ValueError("Mixed precision cannot be combined with CPU inference")
    if target_stage == 4 and (not octave or not Path(octave).is_file() or not matlab.is_dir() or not (vsot / "src").is_dir()):
        raise ValueError("Stage 4 requires valid --octave, --vsot-root and --vsot-matlab-dir (or app settings)")
    runner = plugin / "run_inference.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    return settings, plugin, weights, vsot, matlab, octave, runner


def run(args, stop_event=None, on_update=None, emit=print):
    import numpy as np
    from tifffile import imread, imwrite
    from projects.segmentation.utils.constants import EMPTY_LAYER_PARAMETERS, EMPTY_PROJECT
    from projects.segmentation.utils.project_info import ProjectInfo
    from projects.neural_segmentation.utils.data_processing.plugin_inference import (
        _labels_from_probability, _normalize_probability, _spatial_shape,
    )
    from projects.segmentation.utils.data_processing.voxel_cleanup import fill_small_enclosed_holes

    stop_event = stop_event or threading.Event()
    target_stage = 4 if args.segment_spines else 3
    if stop_event.is_set():
        raise InterruptedError("Stopped")
    settings, plugin, weights, vsot, matlab, octave, runner = prepare(args)
    image_path = args.image.resolve(strict=True)
    folder = getattr(args, "project_dir", None) or args.output_dir.resolve() / image_path.stem
    check_directory(folder, args.clear_run)
    manifest_path = folder / "artifacts/plugin_runtime/cli_run.json"
    resuming = folder.exists() and any(folder.iterdir())
    if resuming and not (folder / "layers/raw.tif").is_file():
        raise ValueError("Existing directory is not a resumable CLI project (raw.tif is missing)")
    image = imread(image_path)
    if image.ndim != 3 or image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError("Expected a finite numeric 3D TIFF in ZYX order; select channels/timepoints beforehand")
    area = np.ones(image.shape, dtype=np.uint8)
    if resuming and (folder / "layers/roi.tif").is_file():
        area = imread(folder / "layers/roi.tif")
        if area.shape != image.shape or not np.isfinite(area).all() or not np.isin(area, [0, 1]).all():
            raise ValueError("Saved ROI is invalid")
    elif args.draw_roi:
        area = paint_roi(image)
    elif args.roi:
        roi = imread(args.roi)
        if roi.shape not in (image.shape, image.shape[1:]):
            raise ValueError("ROI shape must match ZYX image or its XY plane")
        area = np.broadcast_to(roi > 0, image.shape).astype(np.uint8).copy()
    if not area.any():
        raise ValueError("ROI is empty")
    overlap = args.overlap_percent or settings.get("stage4_overlap_percent", 75)
    mixed_precision = args.mixed_precision if args.mixed_precision is not None else settings.get("stage4_mixed_precision", False)
    configuration = dict(image=file_identity(image_path), voxel_size=args.voxel_size,
                         models=[file_identity(w) for w in weights], plugin=str(plugin),
                         threshold=args.threshold, threshold_mode=args.threshold_mode,
                         trunk_threshold=args.trunk_threshold, spine_threshold=args.spine_threshold,
                         fill_holes=not args.no_fill_holes, overlap=overlap, mixed_precision=mixed_precision,
                         prune_length_um=args.prune_length_um, vsot=str(vsot), matlab=str(matlab),
                         roi_source=file_identity(args.roi) if args.roi else None,
                         draw_roi=args.draw_roi)
    saved_manifest = {}
    if manifest_path.is_file():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not compatible_configuration(saved_manifest.get("configuration", {}), configuration):
            raise ValueError("Run parameters/input/models changed; use a new output directory")
        if saved_manifest.get("roi") != file_identity(folder / "layers/roi.tif"):
            raise ValueError("Saved ROI changed; use a new output directory")
        if saved_manifest.get("raw") != file_identity(folder / "layers/raw.tif"):
            raise ValueError("Saved raw image changed; use a new output directory")
    elif resuming:
        # Compatibility with interrupted runs produced before checkpoint support.
        if not np.array_equal(imread(folder / "layers/raw.tif"), image):
            raise ValueError("Existing project's raw image differs from input")
        if args.roi:
            supplied_roi = np.broadcast_to(imread(args.roi) > 0, image.shape)
            if not np.array_equal(supplied_roi, area > 0):
                raise ValueError("ROI differs from the interrupted run")
    existing_description = folder / "project_description.json"
    completed_stage = saved_manifest.get("completed_stage", 0)
    if existing_description.exists() and "completed_stage" not in saved_manifest:
        old_info = json.loads(existing_description.read_text(encoding="utf-8"))
        completed_stage = max((p.get("metadata", {}).get("neural_stage", 0) for p in old_info["layers_parameters"]), default=0)
    if existing_description.exists() and completed_stage >= target_stage:
        emit(f"Project already completed; preserving manual corrections: {existing_description}")
        emit(f"SPINETOOL_BRANCH_COMPLETE {target_stage}")
        return folder
    for directory in ("layers", "colormaps", "additional_colormaps", "auxiliary", "tmp/layers", "tmp/auxiliary",
                      "artifacts/model_runs", "artifacts/corrections", "artifacts/plugin_runtime"):
        (folder / directory).mkdir(parents=True, exist_ok=True)
    if not resuming:
        shared_raw = getattr(args, "shared_raw", None)
        if shared_raw:
            try:
                os.link(shared_raw, folder / "layers/raw.tif")
            except OSError:
                shutil.copyfile(shared_raw, folder / "layers/raw.tif")
        else:
            imwrite(folder / "layers/raw.tif", image, photometric="minisblack")
    if not (folder / "layers/roi.tif").exists():
        imwrite(folder / "layers/roi.tif", area, photometric="minisblack")
    manifest = {"configuration": configuration, "roi": file_identity(folder / "layers/roi.tif"),
                "raw": file_identity(folder / "layers/raw.tif"), "completed_stage": completed_stage}
    write_json(manifest_path, manifest)
    python = args.python_executable or settings.get("plugin_python_executable") or sys.executable
    device = getattr(args, "resolved_device", None) or resolve_device(python, args.device)
    if target_stage == 4 and mixed_precision and device == "cpu":
        raise ValueError("Mixed precision requires CUDA; disable it or select a CUDA environment")
    emit(f"Inference device: {device} (Python: {python})")
    info = deepcopy(EMPTY_PROJECT)
    scale = [s / min(args.voxel_size) for s in args.voxel_size]
    info.update(name=image_path.stem, original_image=image_path.name, shape=list(image.shape),
                original_shape=list(image.shape), real_scale=args.voxel_size, scale=scale,
                project_root=str(folder), device=device, layers_parameters=[],
                data={"layers": [], "last_layer_id": 0})

    def add_layer(kind, name, file, metadata, shape, preview=False):
        index = len(info["layers_parameters"])
        params = deepcopy(EMPTY_LAYER_PARAMETERS)
        params.update(layer_id=index, name=name, file=file, metadata=metadata, preview=preview,
                      visualisation_parameters={"visible": index in (0, 5),
                          "scale": [s*a/b for s, a, b in zip(scale, image.shape, shape)]})
        if index:
            params["visualisation_parameters"]["labels"] = (["background", "trunk", "spines"] if index == 5
                                                           else ["background", "general_label"])
        info["layers_parameters"].append(params)
        if index:
            info["data"]["layers"][-1]["child_layers"] = [index]
        info["data"]["layers"].append(dict(id=index, type=kind, parent_id=index-1 if index else None, child_layers=[]))
        info["data"]["last_layer_id"] = index

    add_layer("image", "Image", "/layers/raw.tif", {"workflow_mode": "neural_cascade"}, image.shape)
    add_layer("mask", "ROI", "/layers/roi.tif", {}, image.shape)
    def checkpoint(stage):
        info["active_layers"] = list(range(stage + 1)) if stage else [0, 1]
        for params in info["layers_parameters"]:
            params["visualisation_parameters"]["visible"] = params["layer_id"] in (0, stage + 1)
        write_json(folder / "project_description.json", ProjectInfo(info, str(folder)).as_dict())
        manifest["completed_stage"] = stage
        write_json(manifest_path, manifest)
        if on_update:
            on_update(folder)
    # Keep the previous checkpoint intact until all its layers are reconstructed.
    if not existing_description.exists():
        checkpoint(0)
    probabilities = {}
    environment = os.environ.copy()
    if octave:
        environment["OCTAVE_EXECUTABLE"] = str(Path(octave).resolve())
        environment["PATH"] = str(Path(octave).resolve().parent) + os.pathsep + environment.get("PATH", "")
    reuse_upstream = True
    for stage in range(1, target_stage + 1):
        if stop_event.is_set():
            raise InterruptedError("Stopped; last valid project checkpoint saved")
        stage_dir = folder / f"artifacts/model_runs/stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        probability_path = stage_dir / "probability.tif"
        request = dict(stage=stage, repo_root=str(plugin), project_root=str(folder), ckpt_path=str(weights[stage-1]),
                       input_image_path=str(folder / "layers/raw.tif"), area_path=str(folder / "layers/roi.tif"),
                       current_scale=args.voxel_size, device=device, output_probability_path=str(probability_path))
        for previous in range(1, min(stage, 3)):
            request[f"stage{previous}_probability_path"] = str(probabilities[previous])
        if stage >= 3:
            request["expected_shape"] = list(image.shape)
        if stage == 4:
            stage3_path = folder / "layers/stage_3.tif"
            if not args.no_fill_holes:
                filled = fill_small_enclosed_holes(imread(stage3_path))
                filled[area == 0] = 0
                imwrite(stage3_path, filled.astype(np.uint8), photometric="minisblack")
                info["layers_parameters"][-1]["metadata"]["holes_filled_before_stage4"] = True
            request.update(stage3_mask_path=str(stage3_path), stage4_prepare_dir="/artifacts/model_runs/stage_4/preprocess",
                           vsot_root=str(vsot), vsot_matlab_dir=str(matlab), octave_executable=environment["OCTAVE_EXECUTABLE"],
                           patch_size=[64, 128, 128], overlap=[round(s*overlap/100) for s in (64, 128, 128)],
                           mixed_precision=mixed_precision,
                           skeleton_prune_length_nm=args.prune_length_um*1000)
        request_path = stage_dir / "request.json"
        response_path = stage_dir / "response.json"
        def read_probability(path):
            probability = imread(path)
            if stage == 4 and probability.shape == (2*image.shape[0], *image.shape[1:]):
                probability = probability.reshape((2, *image.shape))
            if not np.isfinite(probability).all():
                raise ValueError(f"Stage {stage} produced nonfinite probability")
            shape = _spatial_shape(probability)
            if len(shape) != 3 or (stage >= 3 and shape != image.shape):
                raise ValueError(f"Invalid stage {stage} output shape: {probability.shape}")
            if stage == 2 and shape != stage1_shape:
                raise ValueError("Stage 2 shape differs from stage 1")
            return probability

        probability = cached_probability(stage_dir, request, read_probability) if reuse_upstream else None
        if probability is not None:
            emit(f"Stage {stage}/{target_stage} already completed; reusing probability")
        else:
            reuse_upstream = False
            # Never allow an old successful response to survive an interrupted retry.
            invalidate_downstream(folder, stage)
            write_json(request_path, request)
            script = runner if stage < 4 else ROOT / "projects/neural_segmentation/plugins/ai_spines_stage4_inference_cli.py"
            emit(f"Stage {stage}/{target_stage}")
            code = run_process([str(python), str(script), "--request", str(request_path), "--response", str(response_path)],
                               environment, stage_dir / "inference.log", stop_event, emit)
            response = json.loads(response_path.read_text(encoding="utf-8")) if response_path.exists() else {}
            if code or response.get("status") != "ok":
                raise RuntimeError(f"Stage {stage} failed: {response.get('error', 'see ' + str(stage_dir / 'inference.log'))}")
            produced = Path(response.get("output_probability_path", probability_path))
            if not produced.is_absolute():
                produced = folder / str(produced).lstrip("/\\")
            probability = read_probability(produced)
        shape = _spatial_shape(probability)
        if stage == 1:
            stage1_shape = shape
        probability = _normalize_probability(probability)
        from scipy.ndimage import zoom
        spatial_area = area > 0 if shape == image.shape else zoom(
            area, [a/b for a, b in zip(shape, image.shape)], order=0, prefilter=False
        ) > 0
        probability[..., ~spatial_area] = 0
        if not reuse_upstream:
            imwrite(probability_path, probability, photometric="minisblack")
        probabilities[stage] = probability_path
        threshold = args.threshold if args.threshold is not None else (127/255 if stage == 1 else 0.5)
        labels = _labels_from_probability(probability, "stem_spines" if stage == 4 else "binary", threshold,
                                         args.trunk_threshold, args.spine_threshold, args.threshold_mode,
                                         imread(folder / "layers/stage_3.tif") > 0 if stage == 4 else None)
        labels[~spatial_area] = 0
        metadata = dict(workflow_mode="neural_cascade", neural_stage=stage, neural_model_path=str(weights[stage-1]),
                        neural_probability_path=f"/artifacts/model_runs/stage_{stage}/probability.tif",
                        neural_plugin_response_path=f"/artifacts/model_runs/stage_{stage}/response.json",
                        neural_imported_from=f"plugin:ai_spines:stage{stage}", neural_output_shape=list(shape),
                        neural_threshold=threshold, threshold=threshold*100, apply_user_threshold=False)
        if stage == 1:
            metadata["base_threshold"] = int(threshold*255)
        if stage == 4:
            metadata.update(threshold_mode=args.threshold_mode, trunk_threshold=args.trunk_threshold*100,
                            spine_threshold=args.spine_threshold*100, fill_holes_before_stage4=not args.no_fill_holes,
                            overlap_percent=overlap, mixed_precision=request["mixed_precision"],
                            skeleton_prune_length_nm=request["skeleton_prune_length_nm"])
        file = f"/layers/stage_{stage}.tif"
        label_path = folder / file.lstrip("/")
        if reuse_upstream and label_path.is_file():
            try:
                saved_labels = imread(label_path)
                if saved_labels.shape == labels.shape and np.isin(saved_labels, [0, 1, 2] if stage == 4 else [0, 1]).all():
                    labels = saved_labels
            except (OSError, ValueError):
                pass  # The probability is valid; reconstruct a damaged label TIFF.
        imwrite(label_path, labels, photometric="minisblack")
        if stage > 1:
            info["layers_parameters"][-1]["preview"] = False
        add_layer(["binarization", "necks", "voxel_mesh_segmentation", "voxel_mesh_segmentation"][stage-1],
                  ["Binarization", "Neck restoration", "Scale restoration", "Shaft and spines"][stage-1],
                  file, metadata, shape, preview=True)
        if stage >= completed_stage:
            checkpoint(stage)
        if not reuse_upstream:
            emit(f"SPINETOOL_STAGE_COMPLETE {stage}")
    emit(f"Ready for stage {target_stage} correction: {folder / 'project_description.json'}")
    return folder


def merge_branches(base, branches, folder):
    """Build one native SpineTool layer graph; branch artifacts stay project-relative."""
    result = deepcopy(base)
    root_layer = deepcopy(base["data"]["layers"][0])
    root_layer["child_layers"] = []
    result["data"] = {"layers": [root_layer], "last_layer_id": 0}
    result["layers_parameters"] = [deepcopy(base["layers_parameters"][0])]
    result["active_layers"] = [0]
    result["project_root"] = str(folder)
    for name, prefix, branch in branches:
        result["device"] = branch.get("device", result["device"])
        ids = {0: 0}
        for layer in branch["data"]["layers"]:
            if layer["id"]:
                ids[layer["id"]] = len(result["data"]["layers"]) + len(ids) - 1

        def path(value):
            if isinstance(value, str) and value.replace("\\", "/").lstrip("/").split("/")[0] in {
                    "layers", "artifacts", "auxiliary", "colormaps", "additional_colormaps", "tmp"}:
                return prefix + "/" + value.lstrip("/\\")
            return value
        for layer in branch["data"]["layers"]:
            if not layer["id"]:
                continue
            layer = deepcopy(layer)
            layer["id"] = ids[layer["id"]]
            layer["parent_id"] = ids[layer["parent_id"]]
            layer["child_layers"] = [ids[i] for i in layer["child_layers"]]
            if layer["parent_id"] == 0:
                root_layer["child_layers"].append(layer["id"])
            result["data"]["layers"].append(layer)
        for params in branch["layers_parameters"]:
            if not params["layer_id"]:
                continue
            params = deepcopy(params)
            params["layer_id"] = ids[params["layer_id"]]
            if params["layer_id"] == ids[1]:
                params["name"] = name
                params["metadata"]["area_name"] = name
            for key in ("file", "mesh_file", "mesh_source_tif_file"):
                params[key] = path(params[key])
            for key in ("spines_files", "adjusted_spines_files"):
                params[key] = [path(v) for v in params[key]]
            params["additional_files"] = {k: path(v) for k, v in params["additional_files"].items()}
            params["metadata"] = {k: path(v) if k.endswith("path") else v for k, v in params["metadata"].items()}
            for key in ("colormap", "connected_components_colormap"):
                if key in params["visualisation_parameters"]:
                    params["visualisation_parameters"][key] = path(params["visualisation_parameters"][key])
            result["layers_parameters"].append(params)
        if len(branches) == 1:
            result["active_layers"] = [ids[i] for i in branch["active_layers"]]
    result["data"]["last_layer_id"] = len(result["data"]["layers"]) - 1
    return result


def run_regions(args, stop_event=None, emit=print, prepare_only=False):
    """Run each ROI sequentially and publish their native branches after each stage."""
    import numpy as np
    from tifffile import imread, imwrite
    from projects.segmentation.utils.constants import EMPTY_PROJECT, EMPTY_LAYER_PARAMETERS
    stop_event = stop_event or threading.Event()
    settings, _, _, _, _, _, _ = prepare(args)
    python = args.python_executable or settings.get("plugin_python_executable") or sys.executable
    mixed_precision = args.mixed_precision if args.mixed_precision is not None else settings.get("stage4_mixed_precision", False)
    image_path = args.image.resolve(strict=True)
    folder = args.output_dir.resolve() / image_path.stem
    check_directory(folder, args.clear_run)
    image = imread(image_path)
    if image.ndim != 3 or image.dtype.kind not in "uif" or not np.isfinite(image).all():
        raise ValueError(f"Expected finite numeric 3D ZYX TIFF: {image_path}")
    if args.roi or args.draw_roi:
        raise ValueError("--areas cannot be combined with --roi or --draw-roi")
    masks = []
    for area_path in getattr(args, "area_masks", None) or args.areas or []:
        mask = area_path if isinstance(area_path, np.ndarray) else imread(area_path)
        if mask.shape not in (image.shape, image.shape[1:]) or not np.isfinite(mask).all() or not np.any(mask):
            raise ValueError("Empty, nonfinite or incorrectly shaped ROI")
        masks.append((mask > 0).astype(np.uint8))
    if not masks:
        masks = [np.ones(image.shape[1:], dtype=np.uint8)]
    batch_path = folder / "artifacts/plugin_runtime/batch.json"
    specification = {"image": file_identity(image_path), "voxel_size": args.voxel_size,
                     "areas": [hashlib.sha256(memoryview(np.ascontiguousarray(m))).hexdigest() for m in masks],
                     "explicit_areas": bool(getattr(args, "area_masks", None) or args.areas)}
    previous = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.exists() else None
    if previous:
        if previous["specification"] != specification:
            raise ValueError("Image, ROI list or voxel size changed; use a new output directory")
        description = folder / "project_description.json"
        if description.exists() and previous.get("description") != file_identity(description):
            raise ValueError("Project was modified outside the batch tool; use a new output directory to preserve corrections")
    elif folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Existing directory is not a batch project: {folder}")
    (folder / "layers").mkdir(parents=True, exist_ok=True)
    raw = folder / "layers/raw.tif"
    if not raw.exists():
        imwrite(raw, image, photometric="minisblack")
    base = deepcopy(EMPTY_PROJECT)
    scale = [s/min(args.voxel_size) for s in args.voxel_size]
    base.update(name=image_path.stem, original_image=image_path.name, shape=list(image.shape),
                original_shape=list(image.shape), real_scale=args.voxel_size, scale=scale,
                project_root=str(folder), layers_parameters=[], data={"layers": [], "last_layer_id": 0})
    root_params = deepcopy(EMPTY_LAYER_PARAMETERS)
    root_params.update(layer_id=0, name="Image", file="/layers/raw.tif", metadata={"workflow_mode": "neural_cascade"})
    base["layers_parameters"] = [root_params]
    base["data"]["layers"] = [dict(id=0, type="image", parent_id=None, child_layers=[])]
    branches = []
    for index, mask in enumerate(masks, 1):
        prefix = f"/artifacts/roi_runs/area_{index:03d}"
        region = folder / prefix.lstrip("/")
        (region / "layers").mkdir(parents=True, exist_ok=True)
        roi_path = region / "layers/roi.tif"
        if not roi_path.exists():
            imwrite(roi_path, np.broadcast_to(mask, image.shape), photometric="minisblack")
        # A placeholder makes even not-yet-started ROIs visible after Stop.
        branch = deepcopy(base)
        roi_params = deepcopy(EMPTY_LAYER_PARAMETERS)
        roi_params.update(layer_id=1, name=f"Area {index}", file="/layers/roi.tif")
        branch["layers_parameters"].append(roi_params)
        branch["data"]["layers"][0]["child_layers"] = [1]
        branch["data"]["layers"].append(dict(id=1, type="mask", parent_id=0, child_layers=[]))
        branch["active_layers"] = [0, 1]
        branches.append((f"Area {index}", prefix, branch))

    def publish(_=None):
        current = []
        for name, prefix, placeholder in branches:
            description = folder / prefix.lstrip("/") / "project_description.json"
            branch = json.loads(description.read_text(encoding="utf-8")) if description.exists() else placeholder
            current.append((name, prefix, branch))
        write_json(folder / "project_description.json", merge_branches(base, current, folder))
        write_json(batch_path, {"specification": specification,
                               "description": file_identity(folder / "project_description.json")})
    publish()
    if prepare_only:
        emit(f"ROI_SAVED {folder}")
        return folder
    args.resolved_device = resolve_device(python, args.device)
    if args.segment_spines and mixed_precision and args.resolved_device == "cpu":
        raise ValueError("Mixed precision requires CUDA; disable it or select a CUDA environment")
    for index, (_, prefix, _) in enumerate(branches, 1):
        if stop_event.is_set():
            raise InterruptedError("Stopped; project contains the last valid stages")
        emit(f"Area {index}/{len(branches)}: {image_path.name}")
        branch_args = deepcopy(args)
        branch_args.project_dir = folder / prefix.lstrip("/")
        branch_args.shared_raw = raw
        branch_args.roi = branch_args.project_dir / "layers/roi.tif"
        branch_args.areas = None
        branch_args.clear_run = False
        branch_args.draw_roi = False
        # The ROI placeholder is not yet a resumable run. Link the common image first.
        branch_raw = branch_args.project_dir / "layers/raw.tif"
        if not branch_raw.exists():
            try:
                os.link(raw, branch_raw)
            except OSError:
                shutil.copyfile(raw, branch_raw)
        try:
            run(branch_args, stop_event, publish, emit)
        finally:
            publish()
    emit(f"PROJECT_COMPLETE {folder}")
    return folder


def main():
    args = parser().parse_args()
    try:
        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        if args.areas:
            run_regions(args, stop_event)
        else:
            run(args, stop_event)
    except (Exception, KeyboardInterrupt) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
