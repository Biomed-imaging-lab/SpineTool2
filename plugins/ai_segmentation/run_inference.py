import argparse
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from tifffile import imwrite


PROGRESS_PREFIX = "SPINETOOL_PROGRESS "


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _emit_progress(
    stage: int,
    value: int,
    description: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    payload = {
        "stage": stage,
        "value": int(value),
        "total": 100,
        "description": description,
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["items_total"] = int(total)
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def _resolve_device(device: str) -> str:
    value = str(device).lower()
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available")
        return "cuda"
    if value == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _stitch_probability(
    model: torch.nn.Module,
    dataloader,
    output_shape: tuple[int, int, int],
    device: str,
    stage: int,
    progress_start: int = 35,
    progress_end: int = 90,
) -> np.ndarray:
    full = None
    counts = np.zeros(output_shape, dtype=np.uint16)
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = 0
    last_value = None

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            images = batch["image"].to(device)
            predictions = torch.sigmoid(model(images)).cpu().numpy()
            channel_count = predictions.shape[1]
            if full is None:
                full_shape = output_shape if channel_count == 1 else (channel_count, *output_shape)
                full = np.zeros(full_shape, dtype=np.float32)
            coords = batch["coords"].cpu().numpy()
            valid_shapes = batch["valid_shape"].cpu().numpy()
            for idx in range(predictions.shape[0]):
                z, y, x = coords[idx]
                vz, vy, vx = valid_shapes[idx]
                if channel_count == 1:
                    patch = predictions[idx, 0]
                    full[z : z + vz, y : y + vy, x : x + vx] += patch[:vz, :vy, :vx]
                else:
                    patch = predictions[idx]
                    full[:, z : z + vz, y : y + vy, x : x + vx] += patch[
                        :, :vz, :vy, :vx
                    ]
                counts[z : z + vz, y : y + vy, x : x + vx] += 1
            if total_batches > 0:
                value = progress_start + int(
                    (progress_end - progress_start) * batch_index / total_batches
                )
                if value != last_value:
                    _emit_progress(
                        stage,
                        value,
                        "stage " + str(stage) + ": predict patches "
                        + str(batch_index)
                        + "/"
                        + str(total_batches),
                        batch_index,
                        total_batches,
                    )
                    last_value = value

    counts[counts == 0] = 1
    if full is None:
        return np.zeros(output_shape, dtype=np.float32)
    if full.ndim == 3:
        return full / counts.astype(np.float32)
    return full / counts.astype(np.float32)[None, :, :, :]


def _required(request: dict[str, Any], key: str) -> Any:
    if key not in request:
        raise KeyError("Missing required key in request: " + key)
    return request[key]


def _spatial_shape(probability: np.ndarray) -> tuple[int, int, int]:
    if probability.ndim == 4 and probability.shape[0] in (1, 2):
        return tuple(probability.shape[1:])
    if probability.ndim == 4 and probability.shape[1] in (1, 2):
        return (probability.shape[0], probability.shape[2], probability.shape[3])
    return tuple(probability.shape)


def _run_stage1(request: dict[str, Any], device: str, stage: int) -> np.ndarray:
    from src.datamodules.dendrite_datamodule import DendriteDataModule
    from src.models.segmentation_module import DendriteSegmentationModule

    _emit_progress(stage, 8, "stage " + str(stage) + ": prepare data")
    datamodule = DendriteDataModule(
        predict_tiff_path=_required(request, "input_image_path"),
        area_path=request.get("area_path"),
        current_scale=_required(request, "current_scale"),
        batch_size=int(request.get("batch_size", 1)),
        num_workers=int(request.get("num_workers", 0)),
        patch_size=request.get("patch_size", [64, 64, 64]),
        overlap=int(request.get("overlap", 8)),
    )
    datamodule.setup("predict")
    _emit_progress(stage, 25, "stage " + str(stage) + ": load model")
    model = DendriteSegmentationModule.load_from_checkpoint(
        _required(request, "ckpt_path"), weights_only=False
    )
    _emit_progress(stage, 35, "stage " + str(stage) + ": predict patches")

    stitched = _stitch_probability(
        model,
        datamodule.predict_dataloader(),
        tuple(datamodule.predict_ds.shape),
        device,
        stage,
    )

    _emit_progress(stage, 92, "stage " + str(stage) + ": normalize probability")

    # Stage 1 inference uses a symmetric context pad. Remove it before
    # returning the probability to SpineTool so downstream stages keep
    # the real normalized image geometry.
    pad = int(getattr(datamodule.predict_ds, "pad", 0))
    if pad > 0:
        if stitched.ndim == 3:
            stitched = stitched[pad:-pad, pad:-pad, pad:-pad]
        elif stitched.ndim == 4 and stitched.shape[0] in (1, 2):
            stitched = stitched[:, pad:-pad, pad:-pad, pad:-pad]
        elif stitched.ndim == 4 and stitched.shape[1] in (1, 2):
            stitched = stitched[pad:-pad, :, pad:-pad, pad:-pad]

    return _normalize_probability(stitched)


def _run_stage2(request: dict[str, Any], device: str, stage: int) -> np.ndarray:
    from src.datamodules.stage2_datamodule import Stage2DataModule
    from src.models.segmentation_module_stage2 import Stage2SegmentationModule

    _emit_progress(stage, 8, "stage " + str(stage) + ": prepare data")
    datamodule = Stage2DataModule(
        predict_tiff_path=_required(request, "input_image_path"),
        predict_l_dendrite_path=_required(request, "stage1_probability_path"),
        area_path=request.get("area_path"),
        current_scale=_required(request, "current_scale"),
        batch_size=int(request.get("batch_size", 1)),
        num_workers=int(request.get("num_workers", 0)),
        patch_size=request.get("patch_size", [64, 64, 64]),
        overlap=int(request.get("overlap", 16)),
    )
    datamodule.setup("predict")
    _emit_progress(stage, 25, "stage " + str(stage) + ": load model")
    model = Stage2SegmentationModule.load_from_checkpoint(
        _required(request, "ckpt_path"), weights_only=False
    )
    _emit_progress(stage, 35, "stage " + str(stage) + ": predict patches")
    stitched = _stitch_probability(
        model,
        datamodule.predict_dataloader(),
        tuple(datamodule.predict_ds.shape),
        device,
        stage,
    )
    _emit_progress(stage, 92, "stage " + str(stage) + ": normalize probability")
    return _normalize_probability(stitched)


def _run_stage3(request: dict[str, Any], device: str, stage: int) -> np.ndarray:
    from src.datamodules.stage3_datamodule import Stage3DataModule
    from src.models.stage3_module import Stage3SegmentationModule

    _emit_progress(stage, 8, "stage " + str(stage) + ": prepare data")
    datamodule = Stage3DataModule(
        predict_tiff_path=_required(request, "input_image_path"),
        predict_l_dendrite_path=_required(request, "stage1_probability_path"),
        predict_necks_path=_required(request, "stage2_probability_path"),
        area_path=request.get("area_path"),
        current_scale=_required(request, "current_scale"),
        batch_size=int(request.get("batch_size", 1)),
        num_workers=int(request.get("num_workers", 0)),
        patch_size=request.get("patch_size", [64, 128, 128]),
        overlap=request.get("overlap", [32, 64, 64]),
    )
    datamodule.setup("predict")
    _emit_progress(stage, 25, "stage " + str(stage) + ": load model")
    model = Stage3SegmentationModule.load_from_checkpoint(
        _required(request, "ckpt_path"), weights_only=False
    )
    _emit_progress(stage, 35, "stage " + str(stage) + ": predict patches")
    stitched = _stitch_probability(
        model,
        datamodule.predict_dataloader(),
        tuple(datamodule.predict_ds.shape),
        device,
        stage,
    )
    _emit_progress(stage, 92, "stage " + str(stage) + ": normalize probability")
    return _normalize_probability(stitched)


def _run_stage4(request: dict[str, Any], device: str, stage: int) -> np.ndarray:
    from src.datamodules.stage4_datamodule import Stage4DataModule
    from src.models.stage4_module import Stage4SegmentationModule

    _emit_progress(stage, 8, "stage " + str(stage) + ": prepare data")
    datamodule = Stage4DataModule(
        vsot_shaft_path=_required(request, "vsot_shaft_path"),
        vsot_spine_path=_required(request, "vsot_spine_path"),
        shaft_dist_path=_required(request, "shaft_dist_path"),
        spine_dist_path=_required(request, "spine_dist_path"),
        area_path=request.get("area_path"),
        batch_size=int(request.get("batch_size", 1)),
        num_workers=int(request.get("num_workers", 1)),
        patch_size=request.get("patch_size", [64, 128, 128]),
        overlap=request.get("overlap", [48, 96, 96]),
    )
    datamodule.setup("predict")
    _emit_progress(stage, 25, "stage " + str(stage) + ": load model")
    model = Stage4SegmentationModule.load_from_checkpoint(
        _required(request, "ckpt_path"), weights_only=False
    )
    _emit_progress(stage, 35, "stage " + str(stage) + ": predict patches")
    stitched = _stitch_probability(
        model,
        datamodule.predict_dataloader(),
        tuple(datamodule.predict_ds.shape),
        device,
        stage,
    )
    _emit_progress(stage, 92, "stage " + str(stage) + ": normalize probability")
    return _normalize_probability(stitched)


def _run(request: dict[str, Any]) -> dict[str, Any]:
    stage = int(_required(request, "stage"))
    output_path = _required(request, "output_probability_path")
    expected_shape = request.get("expected_shape")
    _emit_progress(stage, 1, "stage " + str(stage) + ": resolve device")
    #device = _resolve_device("cuda")
    device = _resolve_device(request.get("device", "auto"))
    started = time.time()

    if stage == 1:
        probability = _run_stage1(request, device, stage)
    elif stage == 2:
        probability = _run_stage2(request, device, stage)
    elif stage == 3:
        probability = _run_stage3(request, device, stage)
    elif stage == 4:
        probability = _run_stage4(request, device, stage)
    else:
        raise ValueError("Unsupported stage: " + str(stage))

    if expected_shape is not None and tuple(expected_shape) != _spatial_shape(probability):
        raise ValueError(
            "Shape mismatch: expected "
            + str(tuple(expected_shape))
            + ", got "
            + str(tuple(probability.shape))
        )

    _emit_progress(stage, 96, "stage " + str(stage) + ": write probability")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imwrite(output_path, probability.astype(np.float32))
    _emit_progress(stage, 100, "stage " + str(stage) + ": done")
    elapsed = time.time() - started
    return {
        "status": "ok",
        "stage": stage,
        "output_probability_path": output_path,
        "shape": list(probability.shape),
        "min": float(np.min(probability)),
        "max": float(np.max(probability)),
        "device": device,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    try:
        request = _read_json(args.request)
        response = _run(request)
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