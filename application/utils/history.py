from pathlib import Path

from application.settings import ApplicationSettings


def update_open_history(filename):
    settings = ApplicationSettings.instance()
    files = settings.state["open_history"]
    if filename in files:
        files.insert(0, files.pop(files.index(filename)))
    else:
        files.insert(0, filename)
    files = files[0:10]
    settings.update({"open_history": files})
    settings.save()


def get_open_history():
    settings = ApplicationSettings.instance()
    files = settings.state["open_history"]
    return files or [str(Path.home())]
