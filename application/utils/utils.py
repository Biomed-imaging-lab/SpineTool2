import collections.abc
import signal
import socket
from contextlib import contextmanager
from typing import Sequence, Union

from PyQt5.QtCore import QSocketNotifier
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget


@contextmanager
def maybe_allow_interrupt(qapp: QApplication):
    old_sigint_handler = signal.getsignal(signal.SIGINT)
    handler_args = None
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield
        return

    wsock, rsock = socket.socketpair()
    wsock.setblocking(False)
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    sn = QSocketNotifier(rsock.fileno(), QSocketNotifier.Type.Read)

    # Clear the socket to re-arm the notifier.
    sn.activated.connect(lambda *args: rsock.recv(1))

    def handle(*args):
        nonlocal handler_args
        handler_args = args
        qapp.exit()

    signal.signal(signal.SIGINT, handle)
    try:
        yield
    finally:
        wsock.close()
        rsock.close()
        sn.setEnabled(False)
        signal.set_wakeup_fd(old_wakeup_fd)
        signal.signal(signal.SIGINT, old_sigint_handler)
        if handler_args is not None:
            old_sigint_handler(*handler_args)


def is_sequence(arg):
    """Check if ``arg`` is a sequence like a list or tuple.

    return True:
        list
        tuple
    return False
        string
        numbers
        dict
        set
    """
    if isinstance(arg, collections.abc.Sequence) and not isinstance(arg, str):
        return True
    return False


def combine_widgets(
    widgets: Union[QWidget, Sequence[QWidget]], vertical: bool = False
) -> QWidget:
    """Combine a list of widgets into a single QWidget with Layout.

    Parameters
    ----------
    widgets : QWidget or sequence of QWidget
        A widget or a list of widgets to combine.
    vertical : bool, optional
        Whether the layout should be QVBoxLayout or not, by default
        QHBoxLayout is used

    Returns
    -------
    QWidget
        If ``widgets`` is a sequence, returns combined QWidget with `.layout`
        property, otherwise returns the original widget.

    Raises
    ------
    TypeError
        If ``widgets`` is neither a ``QWidget`` or a sequence of ``QWidgets``.
    """
    if isinstance(getattr(widgets, "native", None), QWidget):
        # compatibility with magicgui v0.2.0 which no longer uses QWidgets
        # directly. Like vispy, the backend widget is at widget.native
        return widgets.native  # type: ignore
    if isinstance(widgets, QWidget):
        return widgets
    if is_sequence(widgets):
        # the same as above, compatibility with magicgui v0.2.0
        widgets = [
            i.native if isinstance(getattr(i, "native", None), QWidget) else i
            for i in widgets
        ]
        if all(isinstance(i, QWidget) for i in widgets):
            container = QWidget()
            container.setLayout(QVBoxLayout() if vertical else QHBoxLayout())
            for widget in widgets:
                container.layout().addWidget(widget)
            return container
    raise TypeError('"widget" must be a QWidget or a sequence of QWidgets')
