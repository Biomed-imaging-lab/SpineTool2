import inspect as _inspect

from utils.misc import all_subclasses
from viewer.layers.base.base import Layer

# isabstact check is to exclude _ImageBase class
NAMES = {
    subclass.__name__.lower()
    for subclass in all_subclasses(Layer)
    if not _inspect.isabstract(subclass)
}
