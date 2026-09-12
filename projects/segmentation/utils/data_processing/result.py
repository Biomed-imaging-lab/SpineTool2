from typing import Any

CHECK_QUEUE_TIME_INTERVAL = 0.1


class Result:
    def __init__(
        self,
        data: Any,
        metadata: dict,
        stopped: bool = False,
        files: dict = {},
        error: str = "",
    ):
        self.data = data
        self.stopped = stopped
        self.metadata = metadata
        self.additional_files = files
        self.error = error


class ProgressUpdate:
    def __init__(
        self,
        value: int,
        total: int = 100,
        description: str = "",
    ):
        self.value = value
        self.total = total
        self.description = description
