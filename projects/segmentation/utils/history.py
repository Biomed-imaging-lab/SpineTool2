from pathlib import Path

from projects.segmentation.settings import SegmentationSettings


def update_open_history_files(filename):
    settings = SegmentationSettings.instance()
    files = settings.state["open_history_files"]
    if filename in files:
        files.insert(0, files.pop(files.index(filename)))
    else:
        files.insert(0, filename)
    files = files[0:10]
    settings.update({"open_history_files": files})


def get_open_history_files():
    settings = SegmentationSettings.instance()
    files = settings.state["open_history_files"]
    return files or [str(Path.home())]


def update_open_history_folders(folder):
    settings = SegmentationSettings.instance()
    folders = settings.state["open_history_folders"]
    if folder in folders:
        folders.insert(0, folders.pop(folders.index(folder)))
    else:
        folders.insert(0, folder)
    folders = folders[0:10]
    settings.update({"open_history_folders": folders})


def get_open_history_folders():
    settings = SegmentationSettings.instance()
    folders = settings.state["open_history_folders"]
    return folders or [str(Path.home())]
