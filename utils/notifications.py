import sys
import threading
import warnings
from datetime import datetime
from enum import auto
from types import TracebackType
from typing import Callable, List, Optional, Sequence, Set, Tuple, Type, Union

from PyQt5.QtCore import QObject, pyqtSignal

from utils.misc import StringEnum

name2num = {
    "error": 30,
    "warning": 20,
    "info": 10,
    "none": 0,
}


class NotificationSeverity(StringEnum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    NONE = auto()

    def as_icon(self):
        return {
            self.ERROR: "ⓧ",
            self.WARNING: "⚠️",
            self.INFO: "ⓘ",
            self.NONE: "",
        }[self]

    def __lt__(self, other):
        return name2num[str(self)] < name2num[str(other)]

    def __le__(self, other):
        return name2num[str(self)] <= name2num[str(other)]

    def __gt__(self, other):
        return name2num[str(self)] > name2num[str(other)]

    def __ge__(self, other):
        return name2num[str(self)] >= name2num[str(other)]

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.value)


ActionSequence = Sequence[Tuple[str, Callable[[], None]]]


class Notification:
    def __init__(
        self,
        message: str,
        severity: Union[str, NotificationSeverity] = NotificationSeverity.WARNING,
        actions: ActionSequence = (),
    ):
        self.severity: NotificationSeverity = NotificationSeverity(severity)
        self.message: str = message
        self.actions: ActionSequence = actions
        self.date: datetime = datetime.now()

    @classmethod
    def from_exception(cls, exc: BaseException):
        return ErrorNotification(exc)

    @classmethod
    def from_warning(cls, warning: Warning):
        return WarningNotification(warning)

    def __str__(self):
        return f"{str(self.severity).upper()}: {self.message}"


class ErrorNotification(Notification):
    def __init__(self, exception: BaseException):
        msg = getattr(exception, "message", str(exception))
        actions = getattr(exception, "actions", ())
        super().__init__(msg, NotificationSeverity.ERROR, actions)
        self.exception: BaseException = exception

    def as_html(self) -> str:
        from utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=True)

    def as_text(self) -> str:
        from utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=False, color="NoColor")

    def __str__(self):
        from utils._tracebacks import get_tb_formatter

        fmt = get_tb_formatter()
        exc_info = (
            self.exception.__class__,
            self.exception,
            self.exception.__traceback__,
        )
        return fmt(exc_info, as_html=False)


class WarningNotification(Notification):
    def __init__(self, warning: Warning, filename=None, lineno=None):
        msg = getattr(warning, "message", str(warning))
        actions = getattr(warning, "actions", ())
        super().__init__(msg, NotificationSeverity.WARNING, actions)
        self.warning: Warning = warning
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        category = type(self.warning).__name__
        return f"{self.filename}:{self.lineno}: {category}: {self.warning}!"


class NotificationManager(QObject):
    _instance = None

    notification_ready = pyqtSignal(Notification)

    def __init__(self):
        if not NotificationManager._instance:
            super().__init__()
            self.records: List[Notification] = []
            self._originals_except_hooks: List[Callable] = []
            self._original_showwarnings_hooks: List[Callable] = []
            self._originals_thread_except_hooks: List[Callable] = []
            self._seen_warnings: Set[Tuple[str, Type, str, int]] = set()

    def __enter__(self):
        self.install_hooks()
        return self

    def __exit__(self, *args, **kwargs):
        self.restore_hooks()

    def install_hooks(self) -> None:
        self._originals_thread_except_hooks.append(threading.excepthook)
        threading.excepthook = self.receive_thread_error

        self._originals_except_hooks.append(sys.excepthook)
        self._original_showwarnings_hooks.append(warnings.showwarning)

        sys.excepthook = self.receive_error
        warnings.showwarning = self.receive_warning

    def restore_hooks(self) -> None:
        threading.excepthook = self._originals_thread_except_hooks.pop()
        sys.excepthook = self._originals_except_hooks.pop()
        warnings.showwarning = self._original_showwarnings_hooks.pop()

    def dispatch(self, notification: Notification) -> None:
        self.records.append(notification)
        self.notification_ready.emit(notification)

    def receive_thread_error(self, args: threading.ExceptHookArgs) -> None:
        self.receive_error(*args)

    def receive_error(
        self,
        exctype: Optional[Type[BaseException]] = None,
        value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
        thread: Optional[threading.Thread] = None,
    ) -> None:
        if isinstance(value, KeyboardInterrupt):
            sys.exit("Closed by KeyboardInterrupt")

        self.dispatch(Notification.from_exception(value))

    def receive_warning(
        self,
        message: Warning,
        category: Type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line=None,
    ) -> None:
        msg = message if isinstance(message, str) else message.args[0]
        if (msg, category, filename, lineno) in self._seen_warnings:
            return
        self._seen_warnings.add((msg, category, filename, lineno))
        self.dispatch(Notification.from_warning(message))

    def receive_info(self, message: str) -> None:
        self.dispatch(Notification(message, NotificationSeverity.INFO))

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = NotificationManager()
        return cls._instance


def show_info(message: str) -> None:
    NotificationManager.instance().dispatch(
        Notification(message, severity=NotificationSeverity.INFO)
    )


def show_warning(message: str) -> None:
    NotificationManager.instance().dispatch(
        Notification(message, severity=NotificationSeverity.WARNING)
    )


def show_error(message: str) -> None:
    NotificationManager.instance().dispatch(
        Notification(message, severity=NotificationSeverity.ERROR)
    )
