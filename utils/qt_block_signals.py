from contextlib import contextmanager


@contextmanager
def qt_signals_blocked(obj):
    """Context manager to temporarily block signals from `obj`"""
    previous = obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(previous)
