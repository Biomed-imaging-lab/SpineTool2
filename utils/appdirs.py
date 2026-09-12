import hashlib
import os
import sys
from functools import partial
from typing import Callable

import appdirs

PREFIX_PATH = os.path.realpath(sys.prefix)

sha_short = (
    f"{os.path.basename(PREFIX_PATH)}_{hashlib.sha1(PREFIX_PATH.encode()).hexdigest()}"
)

_appname = "spine_tool_with_napari_ui"
_appauthor = False


config_dir: Callable[[], str] = partial(
    appdirs.user_config_dir, _appname, _appauthor, sha_short
)
data_dir: Callable[[], str] = partial(
    appdirs.user_data_dir, _appname, _appauthor, sha_short
)
