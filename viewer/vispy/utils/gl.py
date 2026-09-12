from contextlib import contextmanager
from functools import lru_cache
from typing import Tuple

import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas

texture_dtypes = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
]


@contextmanager
def _opengl_context():
    canvas = Canvas(show=False) if get_current_canvas() is None else None
    try:
        yield
    finally:
        if canvas is not None:
            canvas.close()


@lru_cache(maxsize=1)
def get_gl_extensions() -> str:
    with _opengl_context():
        return gl.glGetParameter(gl.GL_EXTENSIONS)


@lru_cache
def get_max_texture_sizes() -> Tuple[int, int]:
    with _opengl_context():
        max_size_2d = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)

    if max_size_2d == ():
        max_size_2d = None

    with _opengl_context():
        GL_MAX_3D_TEXTURE_SIZE = 32883
        max_size_3d = gl.glGetParameter(GL_MAX_3D_TEXTURE_SIZE)

    if max_size_3d == ():
        max_size_3d = None

    return max_size_2d, max_size_3d


def fix_data_dtype(data):
    dtype = np.dtype(data.dtype)
    if dtype in texture_dtypes:
        return data

    try:
        dtype_ = {
            "i": np.float32,
            "f": np.float32,
            "u": np.uint16,
            "b": np.uint8,
        }[dtype.kind]
        if dtype_ == np.uint16 and dtype.itemsize > 2:
            dtype_ = np.float32
    except KeyError as e:  # not an int or float
        raise TypeError(
            "type {dtype} not allowed for texture; must be one of {textures}".format(
                dtype=dtype,
                textures=set(texture_dtypes),
            )
        ) from e
    return data.astype(dtype_)


BLENDING_MODES = {
    "opaque": {
        "depth_test": True,
        "cull_face": False,
        "blend": False,
    },
    "translucent": {
        "depth_test": True,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "one_minus_src_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
    "translucent_no_depth": {
        "depth_test": False,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "one_minus_src_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
    "additive": {
        "depth_test": False,
        "cull_face": False,
        "blend": True,
        "blend_func": ("src_alpha", "dst_alpha", "one", "one"),
        "blend_equation": "func_add",
    },
}
