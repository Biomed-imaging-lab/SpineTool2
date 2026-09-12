from enum import auto

from utils.misc import StringEnum


class Interpolation(StringEnum):
    """INTERPOLATION: Vispy interpolation mode."""

    BESSEL = auto()
    CUBIC = auto()
    LINEAR = auto()
    BLACKMAN = auto()
    CATROM = auto()
    GAUSSIAN = auto()
    HAMMING = auto()
    HANNING = auto()
    HERMITE = auto()
    KAISER = auto()
    LANCZOS = auto()
    MITCHELL = auto()
    NEAREST = auto()
    SPLINE16 = auto()
    SPLINE36 = auto()

    @classmethod
    def view_subset(cls):
        return (
            cls.LINEAR,
            cls.NEAREST,
        )
