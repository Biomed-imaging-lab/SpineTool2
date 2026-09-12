import builtins
from enum import Enum, EnumMeta
from typing import List, Sequence, Tuple, Type


class StringEnumMeta(EnumMeta):
    def __getitem__(self, item):
        if isinstance(item, str):
            item = item.upper()

        return super().__getitem__(item)

    def __call__(
        cls,
        value,
        names=None,
        *,
        module=None,
        qualname=None,
        type=None,
        start=1,
    ):
        if names is None:
            if isinstance(value, str):
                return super().__call__(value.lower())
            if isinstance(value, cls):
                return value

            raise ValueError(
                "{class_name} may only be called with a `str` or an instance of {class_name}. Got {dtype}".format(
                    class_name=cls,
                    dtype=builtins.type(value),
                )
            )

        return cls._create_(
            value,
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )

    def keys(self):
        return list(map(str, self))


class StringEnum(Enum, metaclass=StringEnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if type(self) is type(other):
            return self is other
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __hash__(self):
        return hash(str(self))


def all_subclasses(cls: Type) -> set:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def reorder_after_dim_reduction(order: Sequence[int]) -> Tuple[int, ...]:
    return tuple(argsort(argsort(order)))


def argsort(values: Sequence[int]) -> List[int]:
    return sorted(range(len(values)), key=values.__getitem__)


def ensure_n_tuple(val, n, fill=0):
    assert n > 0, "n must be greater than 0"
    tuple_value = tuple(val)
    return (fill,) * (n - len(tuple_value)) + tuple_value[-n:]
