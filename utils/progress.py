from PyQt5.QtCore import QObject, pyqtSignal


class Emitter(QObject):
    added = pyqtSignal(object)
    removed = pyqtSignal(object)


class progress(QObject):
    emitter = Emitter()
    value = pyqtSignal(int)
    description = pyqtSignal(str)
    overflow = pyqtSignal()
    total_ = pyqtSignal(int)
    cancel_requested = pyqtSignal()

    def __init__(self, desc: str = "", total: int = 0):
        QObject.__init__(self)
        self.desc = desc
        self._total = total
        self.n = 0
        progress.emitter.added.emit(self)

    def __repr__(self) -> str:
        return self.desc

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, total):
        self._total = total
        self.total_.emit(self.total)

    def update(self, n=1):
        self.n += n
        self.n = min(self.n, self._total)
        self.value.emit(self.n)

    def set_value(self, n):
        self.n = min(max(int(n), 0), self._total) if self._total > 0 else int(n)
        self.value.emit(self.n)

    def increment_with_overflow(self):
        if self.n == self.total:
            self.total = 0
            self.overflow.emit()
        else:
            self.update()

    def set_description(self, desc):
        self.desc = desc
        try:
            self.description.emit(desc)
        except:
            pass

    def close(self):
        try:
            progress.emitter.removed.emit(self)
        except:
            pass
        try:
            self.deleteLater()
        except:
            pass

    def request_cancel(self):
        self.cancel_requested.emit()
