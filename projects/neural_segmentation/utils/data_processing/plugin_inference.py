import json
import os
import subprocess
import traceback
from multiprocessing import Queue
from typing import Optional

import numpy as np
from tifffile import imread

from projects.segmentation.utils.data_processing.result import ProgressUpdate, Result


PROGRESS_PREFIX = "SPINETOOL_PROGRESS "


def _normalize_probability(volume: np.ndarray) -> np.ndarray:
    arr = volume.astype(np.float32)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_min >= 0.0 and arr_max <= 1.0:
        return arr
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    arr.fill(0)
    return arr


def _labels_from_probability(
    probability: np.ndarray,
    label_mode: str,
    threshold_0_1: float,
    trunk_threshold_0_1: float,
    spine_threshold_0_1: float,
    threshold_mode: str = "trunk",
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if label_mode == "stem_spines":
        trunk = min(max(float(trunk_threshold_0_1), 0.0), 1.0)
        spine = min(max(float(spine_threshold_0_1), 0.0), 1.0)
        if probability.ndim == 4 and probability.shape[0] == 2:
            trunk_probability = probability[0]
            spine_probability = probability[1]
        elif probability.ndim == 4 and probability.shape[1] == 2:
            trunk_probability = probability[:, 0]
            spine_probability = probability[:, 1]
        else:
            trunk_probability = probability
            spine_probability = probability
        mask = np.ones(trunk_probability.shape, dtype=bool)
        if base_mask is not None:
            mask = np.asarray(base_mask, dtype=bool)
        labels = np.zeros(trunk_probability.shape, dtype=np.uint8)
        if threshold_mode == "spine":
            labels[mask] = 1
            labels[mask & (spine_probability >= spine)] = 2
        else:
            labels[mask] = 2
            labels[mask & (trunk_probability >= trunk)] = 1
        return labels

    threshold = min(max(float(threshold_0_1), 0.0), 1.0)
    return (probability >= threshold).astype(np.uint8)


def _fallback_shape(expected_shape: Optional[tuple]) -> tuple:
    if expected_shape is None:
        return (1, 1, 1)
    return tuple(expected_shape)


def _spatial_shape(volume: np.ndarray) -> tuple:
    if volume.ndim == 4 and volume.shape[0] in (1, 2):
        return tuple(volume.shape[1:])
    if volume.ndim == 4 and volume.shape[1] in (1, 2):
        return (volume.shape[0], volume.shape[2], volume.shape[3])
    return tuple(volume.shape)


def _progress_update_from_line(line: str) -> ProgressUpdate | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
    except Exception:
        return None
    return ProgressUpdate(
        value=int(payload.get("value", payload.get("percent", 0))),
        total=int(payload.get("total", 100)),
        description=str(payload.get("description", "")),
    )


def _put_progress(
    queue_out: Queue,
    value: int,
    description: str,
    total: int = 100,
) -> None:
    queue_out.put(ProgressUpdate(value=value, total=total, description=description))


def _run_plugin_process(
    command: list[str],
    queue_in: Queue,
    queue_out: Queue,
    process_environment: Optional[dict[str, str]] = None,
) -> tuple[int | None, str, bool]:
    environment = os.environ.copy()
    if process_environment:
        environment.update(process_environment)
        octave_executable = process_environment.get("OCTAVE_EXECUTABLE", "")
        if octave_executable:
            environment["PATH"] = (
                os.path.dirname(octave_executable)
                + os.pathsep
                + environment.get("PATH", "")
            )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    stdout_lines = []

    if process.stdout is not None:
        for line in process.stdout:
            stdout_lines.append(line)
            progress_update = _progress_update_from_line(line.strip())
            if progress_update is not None:
                queue_out.put(progress_update)
            if not queue_in.empty():
                queue_in.get()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return process.returncode, "".join(stdout_lines), True

    return process.wait(), "".join(stdout_lines), False

def run_ai_spines_inference(
    python_executable: str,
    plugin_script_path: str,
    plugin_request: dict,
    plugin_response_path: str,
    output_probability_path: str,
    metadata: dict,
    queue_in: Queue,
    queue_out: Queue,
    expected_shape: Optional[tuple] = None,
    label_mode: str = "binary",
    threshold_0_1: float = 0.5,
    trunk_threshold_0_1: float = 0.5,
    spine_threshold_0_1: float = 0.8,
    threshold_mode: str = "trunk",
    additional_files: Optional[dict] = None,
    process_environment: Optional[dict[str, str]] = None,
):
    try:
        stage = plugin_request.get("stage", "")
        if not queue_in.empty():
            queue_in.get()
            queue_out.put(
                Result(
                    np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                    metadata,
                    stopped=True,
                )
            )
            return

        _put_progress(queue_out, 0, "stage " + str(stage) + ": prepare request")
        os.makedirs(os.path.dirname(plugin_response_path), exist_ok=True)
        request_path = plugin_response_path.replace(".json", "_request.json")
        with open(request_path, "w", encoding="utf-8") as file:
            json.dump(plugin_request, file)

        command = [
            python_executable,
            plugin_script_path,
            "--request",
            request_path,
            "--response",
            plugin_response_path,
        ]
        _put_progress(queue_out, 3, "stage " + str(stage) + ": start plugin")
        returncode, stdout, stopped = _run_plugin_process(
            command,
            queue_in,
            queue_out,
            process_environment,
        )

        metadata["neural_plugin_stdout"] = stdout
        metadata["neural_plugin_stderr"] = ""

        if stopped:
            queue_out.put(
                Result(
                    np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                    metadata,
                    stopped=True,
                )
            )
            return

        if not os.path.isfile(plugin_response_path):
            message = (
                "Plugin runner failed without response file. "
                + stdout
            )
            queue_out.put(
                Result(
                    np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                    metadata,
                    error=message,
                )
            )
            return

        _put_progress(queue_out, 94, "stage " + str(stage) + ": read plugin response")
        with open(plugin_response_path, "r", encoding="utf-8") as file:
            response = json.load(file)
        if response.get("status") != "ok":
            message = response.get("error", "Unknown plugin error")
            traceback_text = response.get("traceback", "")
            queue_out.put(
                Result(
                    np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                    metadata,
                    error=message + ("\n" + traceback_text if traceback_text else ""),
                )
            )
            return

        _put_progress(queue_out, 96, "stage " + str(stage) + ": read probability")
        produced_probability_path = response.get(
            "output_probability_path", output_probability_path
        )
        if not os.path.isfile(produced_probability_path):
            queue_out.put(
                Result(
                    np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                    metadata,
                    error="Plugin did not produce output probability file",
                )
            )
            return

        probability = imread(produced_probability_path)
        if (
            expected_shape is not None
            and probability.ndim == 3
            and probability.shape[0] == 2 * int(expected_shape[0])
            and tuple(probability.shape[1:]) == tuple(expected_shape[1:])
        ):
            probability = probability.reshape((2, *tuple(expected_shape)))
        if expected_shape is not None and _spatial_shape(probability) != tuple(expected_shape):
            queue_out.put(
                Result(
                    np.zeros(tuple(expected_shape), dtype=np.uint8),
                    metadata,
                    error=(
                        "Probability shape mismatch. Expected "
                        + str(tuple(expected_shape))
                        + ", got "
                        + str(tuple(probability.shape))
                    ),
                )
            )
            return

        probability = _normalize_probability(probability)
        _put_progress(queue_out, 98, "stage " + str(stage) + ": build labels")
        labels = _labels_from_probability(
            probability=probability,
            label_mode=label_mode,
            threshold_0_1=threshold_0_1,
            trunk_threshold_0_1=trunk_threshold_0_1,
            spine_threshold_0_1=spine_threshold_0_1,
            threshold_mode=threshold_mode,
            base_mask=(
                imread(plugin_request["stage3_mask_path"]) > 0
                if label_mode == "stem_spines"
                and plugin_request.get("stage3_mask_path")
                else None
            ),
        )
        _put_progress(queue_out, 100, "stage " + str(stage) + ": finished")
        queue_out.put(Result(labels, metadata))
    except Exception as err:
        queue_out.put(
            Result(
                np.zeros(_fallback_shape(expected_shape), dtype=np.uint8),
                metadata,
                error=str(err) + "\n" + traceback.format_exc(),
            )
        )
