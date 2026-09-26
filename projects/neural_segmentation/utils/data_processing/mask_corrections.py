import os
from typing import Optional

import numpy as np


def save_user_correction_state(
    path: str,
    painted_mask: np.ndarray,
    original_values: np.ndarray,
) -> None:
    """Atomically persist editable correction state in a compressed artifact."""
    painted_mask = np.asarray(painted_mask, dtype=bool)
    original_values = np.asarray(original_values)
    if painted_mask.shape != original_values.shape:
        raise ValueError("User-correction state shapes do not match")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp.npz"
    np.savez_compressed(
        temporary_path,
        painted_mask=painted_mask,
        original_values=original_values,
    )
    os.replace(temporary_path, path)


def load_user_correction_state(
    path: str, expected_shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Load correction arrays only when a caller needs to read or edit them."""
    with np.load(path, allow_pickle=False) as artifact:
        painted_mask = np.asarray(artifact["painted_mask"], dtype=bool)
        original_values = np.asarray(artifact["original_values"])
    if (
        painted_mask.shape != tuple(expected_shape)
        or original_values.shape != tuple(expected_shape)
    ):
        raise ValueError(
            "User-correction artifact shape mismatch: "
            + str(painted_mask.shape)
            + " vs "
            + str(tuple(expected_shape))
        )
    return painted_mask, original_values


def effective_user_corrections(
    labels: np.ndarray,
    painted_mask: Optional[np.ndarray],
    original_values: Optional[np.ndarray],
) -> Optional[dict[str, np.ndarray]]:
    """Return the final, non-undone user edits as binary constraints."""
    if painted_mask is None or original_values is None:
        return None
    labels = np.asarray(labels)
    painted_mask = np.asarray(painted_mask, dtype=bool)
    original_values = np.asarray(original_values)
    if painted_mask.shape != labels.shape or original_values.shape != labels.shape:
        raise ValueError("Label and user-correction shapes do not match")

    changed = painted_mask & (labels != original_values)
    if not np.any(changed):
        return None
    return {
        "mask": painted_mask,
        "foreground": labels > 0,
    }


def resize_binary_mask(mask: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    """Resize a correction mask with nearest-neighbour voxel interpolation."""
    mask = np.asarray(mask, dtype=bool)
    output_shape = tuple(int(size) for size in output_shape)
    if mask.ndim != len(output_shape) or any(size <= 0 for size in output_shape):
        raise ValueError(
            "Cannot resize mask " + str(mask.shape) + " to " + str(output_shape)
        )
    if mask.shape == output_shape:
        return mask.copy()

    result = mask
    for axis, output_size in enumerate(output_shape):
        input_size = result.shape[axis]
        coordinates = np.floor(
            (np.arange(output_size, dtype=np.float64) + 0.5)
            * input_size
            / output_size
        ).astype(np.intp)
        coordinates = np.clip(coordinates, 0, input_size - 1)
        result = np.take(result, coordinates, axis=axis)
    return result


def resize_user_corrections(
    corrections: Optional[dict[str, np.ndarray]],
    output_shape: tuple[int, ...],
) -> Optional[dict[str, np.ndarray]]:
    if corrections is None:
        return None
    return {
        "mask": resize_binary_mask(corrections["mask"], output_shape),
        "foreground": resize_binary_mask(corrections["foreground"], output_shape),
    }


def apply_local_percentile_probability_corrections(
    probability: np.ndarray,
    corrections: Optional[dict[str, np.ndarray]],
    threshold_0_1: float,
    *,
    window_size: int = 11,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
    threshold_margin: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Build an inference-only probability map with locally scaled corrections."""
    if corrections is None:
        return probability, 0
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("Local correction window size must be a positive odd number")

    probability = np.asarray(probability)
    mask = np.asarray(corrections["mask"], dtype=bool)
    foreground = np.asarray(corrections["foreground"], dtype=bool)
    probability_view = probability
    if probability.ndim == mask.ndim + 1:
        if probability.shape[0] == 1:
            probability_view = probability[0]
        elif probability.shape[1] == 1:
            probability_view = probability[:, 0]
    if probability_view.shape != mask.shape or foreground.shape != mask.shape:
        raise ValueError(
            "Probability and user-correction shapes do not match: "
            + str(probability.shape)
            + " vs "
            + str(mask.shape)
        )
    if mask.ndim != 3:
        raise ValueError("Local probability corrections require a 3D volume")

    correction_count = int(np.count_nonzero(mask))
    if correction_count == 0:
        return probability, 0

    result = probability.astype(np.float32, copy=True)
    if probability_view is probability:
        result_view = result
    elif probability.shape[0] == 1:
        result_view = result[0]
    else:
        result_view = result[:, 0]

    threshold = min(max(float(threshold_0_1), 0.0), 1.0)
    positive_floor = min(1.0, threshold + float(threshold_margin))
    negative_ceiling = max(0.0, threshold - float(threshold_margin))
    radius = window_size // 2
    source = probability_view
    for z, y, x in np.argwhere(mask):
        y0 = max(0, int(y) - radius)
        y1 = min(source.shape[1], int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(source.shape[2], int(x) + radius + 1)
        neighborhood = source[int(z), y0:y1, x0:x1]
        if foreground[z, y, x]:
            local_value = float(np.percentile(neighborhood, upper_percentile))
            result_view[z, y, x] = max(local_value, positive_floor)
        else:
            local_value = float(np.percentile(neighborhood, lower_percentile))
            result_view[z, y, x] = min(local_value, negative_ceiling)
    return result, correction_count


def apply_binary_label_corrections(
    labels: np.ndarray,
    corrections: Optional[dict[str, np.ndarray]],
) -> tuple[np.ndarray, int]:
    """Overlay binary ground truth without changing model probabilities."""
    if corrections is None:
        return labels, 0

    labels = np.asarray(labels)
    mask = np.asarray(corrections["mask"], dtype=bool)
    foreground = np.asarray(corrections["foreground"], dtype=bool)
    if labels.shape != mask.shape or foreground.shape != mask.shape:
        raise ValueError(
            "Label and user-correction shapes do not match: "
            + str(labels.shape)
            + " vs "
            + str(mask.shape)
        )
    correction_count = int(np.count_nonzero(mask))
    if correction_count == 0:
        return labels, 0
    result = labels.astype(np.uint8, copy=True)
    result[mask] = foreground[mask].astype(np.uint8)
    return result, correction_count
