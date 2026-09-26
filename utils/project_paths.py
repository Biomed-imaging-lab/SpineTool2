"""Resolve native, Windows and historic project-relative artifact paths."""
import ntpath
import os

# Older project files stored these project-relative directories with a slash.
_LEGACY_ROOTS = (
    "layers", "auxiliary", "colormaps", "additional_colormaps", "artifacts",
    "tmp/layers", "tmp/auxiliary",
)


def is_absolute_path(path: str) -> bool:
    return os.path.isabs(path) or bool(ntpath.splitdrive(path)[0]) or path.startswith("\\\\")


def is_legacy_project_path(path: str) -> bool:
    value = str(path).replace("\\", "/")
    if not value.startswith("/") or value.startswith("//"):
        return False
    value = value[1:]
    return value == "tmp" or any(
        value == root or value.startswith(root + "/") for root in _LEGACY_ROOTS
    )


def normalize_absolute_path(path: str) -> str:
    if ntpath.splitdrive(path)[0] and os.name != "nt":
        return ntpath.normpath(path)
    return os.path.normpath(path)


def resolve_project_path(folder: str, path: str) -> str:
    if not path:
        return ""
    value = str(path)
    if is_absolute_path(value) and not is_legacy_project_path(value):
        return normalize_absolute_path(value)
    relative = value.lstrip("/\\").replace("\\", os.sep)
    return os.path.abspath(os.path.join(folder, relative))
