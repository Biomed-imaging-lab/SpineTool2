from collections.abc import Iterable

import numpy as np


def format_float(value):
    return f"{value:0.3g}"


def status_format(value):
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return "[" + str.join(", ", [status_format(v) for v in value]) + "]"
    if value is None:
        return ""
    if isinstance(value, float) or np.issubdtype(type(value), np.floating):
        return format_float(value)

    return str(value)


def generate_layer_coords_status(position, value):
    if position is not None:
        full_coord = map(str, np.round(position).astype(int))
        msg = f" [{' '.join(full_coord)}]"
    else:
        msg = ""

    if value is not None:
        msg += f": {status_format(value)}"
    return msg
