import argparse
import inspect
import json
import os
import sys
import traceback
from typing import Any

import numpy as np
import torch
from tifffile import imread, imwrite


PROGRESS_PREFIX = "SPINETOOL_PROGRESS "


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _emit_progress(
    value: int,
    description: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    payload = {
        "stage": 4,
        "value": int(value),
        "total": 100,
        "description": description,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["items_total"] = int(total)
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def _required(request: dict[str, Any], key: str) -> Any:
    if key not in request:
        raise KeyError("Missing required key in request: " + key)
    return request[key]


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    arr = probability.astype(np.float32)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_min >= 0.0 and arr_max <= 1.0:
        return arr
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    arr.fill(0)
    return arr


def _resolve_device(device: str) -> str:
    value = str(device).lower()
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available")
        return "cuda"
    if value == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _import_preprocess_modules(repo_root: str):
    data_preprocessing_dir = os.path.join(repo_root, "data_preprocessing")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if data_preprocessing_dir not in sys.path:
        sys.path.insert(0, data_preprocessing_dir)

    from data_preprocessing.stage4_sceleton import scelete
    from data_preprocessing.stage4_distance import process_item as build_distance_maps
    from data_preprocessing.stage4_vsot import run_vsot_oct2py

    return scelete, run_vsot_oct2py, build_distance_maps


def _prepare_stage4_inputs(request: dict[str, Any]) -> dict[str, str]:
    octave_executable = _required(request, "octave_executable")
    os.environ["OCTAVE_EXECUTABLE"] = octave_executable
    octave_dir = os.path.dirname(octave_executable)
    os.environ["PATH"] = octave_dir + os.pathsep + os.environ.get("PATH", "")
    repo_root = _required(request, "repo_root")
    scelete, run_vsot_oct2py, build_distance_maps = _import_preprocess_modules(
        repo_root
    )

    _emit_progress(6, "stage 4: read user-approved stage 3 mask")
    stage3_mask_path = _required(request, "stage3_mask_path")
    mask = (imread(stage3_mask_path) > 0).astype(np.uint8)

    area_path = request.get("area_path")
    if isinstance(area_path, str) and area_path != "" and os.path.isfile(area_path):
        area = (imread(area_path) > 0).astype(np.uint8)
        if area.shape == mask.shape:
            mask = mask * area

    prepare_dir = _required(request, "stage4_prepare_dir")
    os.makedirs(prepare_dir, exist_ok=True)
    mask_path = os.path.join(prepare_dir, "stage3_binary_mask.tif")
    skeleton_path = os.path.join(prepare_dir, "skeleton.tif")
    segments_path = os.path.join(prepare_dir, "segments_mask.tif")
    shaft_vsot_path = os.path.join(prepare_dir, "shaft_vsot.tif")
    spine_vsot_path = os.path.join(prepare_dir, "spine_vsot.tif")
    shaft_dist_path = os.path.join(prepare_dir, "shaft_dist.tif")
    spine_dist_path = os.path.join(prepare_dir, "spine_dist.tif")
    imwrite(mask_path, mask.astype(np.uint8))

    current_scale = list(_required(request, "current_scale"))
    skeleton_item = {
        "name": "stage4_item",
        "stage3_mask": mask_path,
        "image": mask_path,
        "shaft_mask": mask_path,
        "skeleton": skeleton_path,
        "segments_mask": segments_path,
        "scale": current_scale,
        "skeleton_prune_length_nm": float(
            request.get("skeleton_prune_length_nm", 1000)
        ),
    }
    if isinstance(area_path, str) and area_path != "":
        skeleton_item["area_of_interest"] = area_path

    skeleton_json_path = os.path.join(prepare_dir, "stage4_skeleton_request.json")
    _write_json(skeleton_json_path, {"data": [skeleton_item]})
    _emit_progress(15, "stage 4: build dendrite skeleton")
    scelete(skeleton_json_path, base_path="", do_segmentation=True)

    if not os.path.isfile(skeleton_path) or not os.path.isfile(segments_path):
        raise RuntimeError(
            "stage4_skeleton did not create skeleton/segments files: "
            + skeleton_path
            + ", "
            + segments_path
        )

    vsot_json_path = os.path.join(prepare_dir, "stage4_vsot_request.json")
    _write_json(
        vsot_json_path,
        {
            "data": [
                {
                    "name": "stage4_item",
                    "general_mask": mask_path,
                    "skeleton": skeleton_path,
                    "segments_mask": segments_path,
                    "shaft_vsot": shaft_vsot_path,
                    "spine_vsot": spine_vsot_path,
                    "scale": current_scale,
                }
            ]
        },
    )

    run_vsot_signature = inspect.signature(run_vsot_oct2py)
    _emit_progress(30, "stage 4: run VSOT preprocessing")
    if "vsot_root" not in run_vsot_signature.parameters:
        raise RuntimeError(
            "The configured AI runtime is outdated: stage4_vsot.run_vsot_oct2py "
            "does not accept vsot_root"
        )
    run_vsot_oct2py(
        vsot_json_path,
        _required(request, "vsot_matlab_dir"),
        base_path="",
        vsot_root=_required(request, "vsot_root"),
    )
    _emit_progress(45, "stage 4: VSOT preprocessing finished")

    if not os.path.isfile(shaft_vsot_path) or not os.path.isfile(spine_vsot_path):
        raise RuntimeError(
            "stage4_vsot returned without an error but did not create shaft/spine files: "
            + shaft_vsot_path
            + ", "
            + spine_vsot_path
        )

    distance_json_path = os.path.join(prepare_dir, "stage4_distance_request.json")
    distance_item = {
        "name": "stage4_item",
        "skeleton": skeleton_path,
        "shaft_vsot": shaft_vsot_path,
        "spine_vsot": spine_vsot_path,
        "shaft_dist": shaft_dist_path,
        "spine_dist": spine_dist_path,
        "scale": current_scale,
    }
    _write_json(distance_json_path, {"data": [distance_item]})
    _emit_progress(46, "stage 4: build distance channels")
    build_distance_maps(distance_item, base_path="")
    _emit_progress(52, "stage 4: distance channels finished")

    if not os.path.isfile(shaft_dist_path) or not os.path.isfile(spine_dist_path):
        raise RuntimeError(
            "stage4_distance did not create shaft/spine distance files: "
            + shaft_dist_path
            + ", "
            + spine_dist_path
        )

    return {
        "mask_source": stage3_mask_path,
        "mask_path": mask_path,
        "skeleton_path": skeleton_path,
        "segments_mask_path": segments_path,
        "vsot_shaft_path": shaft_vsot_path,
        "vsot_spine_path": spine_vsot_path,
        "shaft_dist_path": shaft_dist_path,
        "spine_dist_path": spine_dist_path,
        "skeleton_json_path": skeleton_json_path,
        "vsot_json_path": vsot_json_path,
        "distance_json_path": distance_json_path,
    }


def _model_output_channels(model) -> int:
    try:
        return int(model.hparams.out_ch)
    except (AttributeError, TypeError, ValueError):
        return 1


def _predict_and_stitch_on_the_fly(
    model,
    datamodule,
    device: str,
    mixed_precision: bool = False,
) -> np.ndarray:
    dl = datamodule.predict_dataloader()
    original_shape = datamodule.predict_ds.shape
    use_cuda = device.startswith("cuda")
    accumulator_device = torch.device(device if use_cuda else "cpu")
    full = None
    counts = torch.zeros(
        original_shape,
        dtype=torch.float32,
        device=accumulator_device,
    )
    try:
        total_batches = len(dl)
    except Exception:
        total_batches = 0
    last_value = None

    model.eval()
    model.to(device)
    with torch.inference_mode():
        for batch_index, batch in enumerate(dl, start=1):
            coords = batch["coords"].numpy()
            valid_shapes = batch["valid_shape"].numpy()
            images_cpu = batch["image"]
            nonzero_mask = images_cpu.reshape(images_cpu.shape[0], -1).any(dim=1)
            active_indices = torch.nonzero(nonzero_mask, as_tuple=False).flatten().tolist()

            # Every window contributes to the averaging denominator. A skipped
            # all-zero window contributes an explicitly zero prediction.
            for idx in range(images_cpu.shape[0]):
                z, y, x = coords[idx]
                vz, vy, vx = valid_shapes[idx]
                counts[z : z + vz, y : y + vy, x : x + vx] += 1

            if active_indices:
                images = images_cpu[nonzero_mask].to(device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_cuda and mixed_precision,
                ):
                    predictions = torch.sigmoid(model(images))
                channel_count = predictions.shape[1]
                if full is None:
                    full_shape = (
                        original_shape
                        if channel_count == 1
                        else (channel_count, *original_shape)
                    )
                    full = torch.zeros(
                        full_shape,
                        dtype=torch.float32,
                        device=accumulator_device,
                    )

                for prediction_idx, batch_idx in enumerate(active_indices):
                    z, y, x = coords[batch_idx]
                    vz, vy, vx = valid_shapes[batch_idx]
                    prediction = predictions[prediction_idx]
                    if channel_count == 1:
                        full[z : z + vz, y : y + vy, x : x + vx] += prediction[
                            0, :vz, :vy, :vx
                        ].float()
                    else:
                        full[:, z : z + vz, y : y + vy, x : x + vx] += prediction[
                            :, :vz, :vy, :vx
                        ].float()
            if total_batches > 0:
                value = 62 + int(30 * batch_index / total_batches)
                if value != last_value:
                    _emit_progress(
                        value,
                        "stage 4: predict patches "
                        + str(batch_index)
                        + "/"
                        + str(total_batches),
                        batch_index,
                        total_batches,
                    )
                    last_value = value

    counts.clamp_min_(1)
    if full is None:
        channel_count = _model_output_channels(model)
        empty_shape = (
            original_shape
            if channel_count == 1
            else (channel_count, *original_shape)
        )
        return np.zeros(empty_shape, dtype=np.float32)
    if full.ndim == 3:
        probability = full / counts
    else:
        probability = full / counts[None, :, :, :]
    return probability.cpu().numpy()


def _run_stage4(request: dict[str, Any]) -> dict[str, Any]:
    repo_root = _required(request, "repo_root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _emit_progress(3, "stage 4: import model runtime")
    from src.datamodules.stage4_datamodule import Stage4DataModule
    from src.models.stage4_module import Stage4SegmentationModule

    prepared = _prepare_stage4_inputs(request)
    _emit_progress(54, "stage 4: resolve device")
    device = _resolve_device(request.get("device", "auto"))

    _emit_progress(56, "stage 4: prepare model input")
    datamodule = Stage4DataModule(
        vsot_shaft_path=prepared["vsot_shaft_path"],
        vsot_spine_path=prepared["vsot_spine_path"],
        shaft_dist_path=prepared["shaft_dist_path"],
        spine_dist_path=prepared["spine_dist_path"],
        area_path=request.get("area_path"),
        batch_size=int(request.get("batch_size", 1)),
        num_workers=int(request.get("num_workers", 1)),
        patch_size=request.get("patch_size", [64, 128, 128]),
        overlap=request.get("overlap", [48, 96, 96]),
    )
    datamodule.setup("predict")
    expected_shape = tuple(imread(prepared["mask_path"]).shape)
    if tuple(datamodule.predict_ds.shape) != expected_shape:
        raise RuntimeError(
            "Stage 4 input shape mismatch: dataset "
            + str(tuple(datamodule.predict_ds.shape))
            + ", user mask "
            + str(expected_shape)
        )

    _emit_progress(58, "stage 4: load model")
    model = Stage4SegmentationModule.load_from_checkpoint(
        _required(request, "ckpt_path"), weights_only=False
    )
    _emit_progress(62, "stage 4: predict patches")
    mixed_precision = bool(request.get("mixed_precision", False))
    probability = _predict_and_stitch_on_the_fly(
        model,
        datamodule,
        device,
        mixed_precision=mixed_precision,
    )
    probability_spatial_shape = (
        tuple(probability.shape[1:]) if probability.ndim == 4 else tuple(probability.shape)
    )
    if probability_spatial_shape != expected_shape:
        raise RuntimeError(
            "Stage 4 output shape mismatch: expected "
            + str(expected_shape)
            + ", got "
            + str(tuple(probability.shape))
        )
    _emit_progress(94, "stage 4: normalize probability")
    probability = _normalize_probability(probability)
    output_probability_path = _required(request, "output_probability_path")
    os.makedirs(os.path.dirname(output_probability_path), exist_ok=True)
    _emit_progress(97, "stage 4: write probability")
    if probability.ndim == 4:
        imwrite(
            output_probability_path,
            probability.astype(np.float32),
            metadata={"axes": "CZYX"},
            photometric="minisblack",
        )
    else:
        imwrite(output_probability_path, probability.astype(np.float32))
    _emit_progress(100, "stage 4: done")

    return {
        "status": "ok",
        "output_probability_path": output_probability_path,
        "shape": list(probability.shape),
        "min": float(np.min(probability)),
        "max": float(np.max(probability)),
        "device": device,
        "mixed_precision": mixed_precision,
        "overlap": list(datamodule.hparams.overlap),
        "prepared": prepared,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    try:
        request = _read_json(args.request)
        response = _run_stage4(request)
        _write_json(args.response, response)
        return 0
    except Exception as err:
        _write_json(
            args.response,
            {
                "status": "error",
                "error": str(err),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
