from typing import Tuple, Union

import numpy as np

_np_uints = {
    8: np.uint8,
    16: np.uint16,
    32: np.uint32,
    64: np.uint64,
}

_np_ints = {
    8: np.int8,
    16: np.int16,
    32: np.int32,
    64: np.int64,
}

_np_floats = {
    16: np.float16,
    32: np.float32,
    64: np.float64,
}

_np_complex = {
    64: np.complex64,
    128: np.complex128,
}

_np_kinds = {
    "uint": _np_uints,
    "int": _np_ints,
    "float": _np_floats,
    "complex": _np_complex,
}


def _normalize_str_by_bit_depth(dtype_str, kind):
    if not any(str.isdigit(c) for c in dtype_str):  # Python 'int' or 'float'
        return np.dtype(kind).type
    bit_dict = _np_kinds[kind]
    if "128" in dtype_str:
        return bit_dict[128]
    if "8" in dtype_str:
        return bit_dict[8]
    if "16" in dtype_str:
        return bit_dict[16]
    if "32" in dtype_str:
        return bit_dict[32]
    if "64" in dtype_str:
        return bit_dict[64]
    return None


def normalize_dtype(dtype_spec):
    dtype_str = str(dtype_spec)
    if "uint" in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, "uint")
    if "int" in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, "int")
    if "float" in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, "float")
    if "complex" in dtype_str:
        return _normalize_str_by_bit_depth(dtype_str, "complex")
    if "bool" in dtype_str:
        return np.bool_

    return dtype_spec


vispy_texture_dtype = np.float32
