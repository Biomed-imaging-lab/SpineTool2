from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Type, Union

from PyQt5.QtCore import QObject, pyqtSignal

from viewer.components.layer_list._typed import _L, _T, Index, TypedMutableSequence


class EventedListMeta(TypedMutableSequence[_T].__class__, QObject.__class__):
    pass


class EventedList(TypedMutableSequence[_T], QObject, metaclass=EventedListMeta):
    inserting_ = pyqtSignal(int)
    inserted_ = pyqtSignal(int, object)
    removing_ = pyqtSignal(int)
    removed_ = pyqtSignal(int, object)
    moving_ = pyqtSignal(int, int)
    moved_ = pyqtSignal(int, int, object)
    changed_ = pyqtSignal(int, object, object)
    reordered_ = pyqtSignal(object)

    def __init__(
        self,
        data: Iterable[_T] = (),
        *,
        basetype: Union[Type[_T], Sequence[Type[_T]]] = (),
        lookup: Dict[Type[_L], Callable[[_T], Union[_T, _L]]] = None,
    ) -> None:
        QObject.__init__(self)
        if lookup is None:
            lookup = {}
        TypedMutableSequence[_T].__init__(self, data, basetype=basetype, lookup=lookup)

    def __setitem__(self, key, value):
        old = self._list[key]
        if isinstance(key, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Can only assign an iterable to slice")
            value = list(value)  # make sure we don't empty generators and reuse them
            if value == old:
                return
            [self._type_check(v) for v in value]  # before we mutate the list
            if key.step is not None:  # extended slices are more restricted
                indices = list(range(*key.indices(len(self))))
                if not len(value) == len(indices):
                    raise ValueError(
                        "attempt to assign sequence of size {size} to extended slice of size {slice_size}".format(
                            size=len(value),
                            slice_size=len(indices),
                        )
                    )
                for i, v in zip(indices, value):
                    self.__setitem__(i, v)
            else:
                del self[key]
                start = key.start or 0
                for i, v in enumerate(value):
                    self.insert(start + i, v)
        else:
            if value is old:
                return
            super().__setitem__(key, value)
            self.changed_.emit(key, old, value)

    def _delitem_indices(self, key: Index) -> Iterable[Tuple["EventedList[_T]", int]]:
        # returning List[(self, int)] allows subclasses to pass nested members
        if isinstance(key, int):
            return [(self, key if key >= 0 else key + len(self))]
        if isinstance(key, slice):
            return [(self, i) for i in range(*key.indices(len(self)))]
        if type(key) in self._lookup:
            return [(self, self.index(key))]

        valid = {int, slice}.union(set(self._lookup))
        raise TypeError(
            "Deletion index must be {valid!r}, got {dtype}".format(
                valid=valid,
                dtype=type(key),
            )
        )

    def __delitem__(self, key: Index):
        # delete from the end
        for parent, index in sorted(self._delitem_indices(key), reverse=True):
            parent.removing_.emit(index)
            item = parent._list.pop(index)
            self._process_delete_item(item)
            parent.removed_.emit(index, item)

    def _process_delete_item(self, item: _T):
        """Allow process item in inherited class before event was emitted"""

    def insert(self, index: int, value: _T):
        self.inserting_.emit(index)
        super().insert(index, value)
        self.inserted_.emit(index, value)

    def move(
        self, src_index: int, dest_index: int = 0, reorder_signal: bool = True
    ) -> bool:
        if dest_index < 0:
            dest_index += len(self) + 1
        if dest_index in (src_index, src_index + 1):
            # this is a no-op
            return False

        self.moving_.emit(src_index, dest_index)
        item = self._list.pop(src_index)
        if dest_index > src_index:
            dest_index -= 1
        self._list.insert(dest_index, item)
        self.moved_.emit(src_index, dest_index, item)
        if reorder_signal:
            self.reordered_.emit(self)
        return True

    def move_multiple(self, sources: Iterable[Index], dest_index: int = 0) -> int:
        # calling list here makes sure that there are no index errors up front
        move_plan = list(self._move_plan(sources, dest_index))

        for src, dest in move_plan:
            self.move(src, dest, False)

        self.reordered_.emit(self)
        return len(move_plan)

    def _move_plan(self, sources: Iterable[Index], dest_index: int):
        if isinstance(dest_index, slice):
            raise TypeError("Destination index may not be a slice")

        to_move: List[int] = []
        for idx in sources:
            if isinstance(idx, slice):
                to_move.extend(list(range(*idx.indices(len(self)))))
            elif isinstance(idx, int):
                to_move.append(idx)
            else:
                raise TypeError(
                    "Can only move integer or slice indices, not {t}".format(
                        t=type(idx),
                    )
                )

        to_move = list(dict.fromkeys(to_move))

        if dest_index < 0:
            dest_index += len(self) + 1

        d_inc = 0
        popped: List[int] = []
        for i, src in enumerate(to_move):
            if src != dest_index:
                # we need to decrement the src_i by 1 for each time we have
                # previously pulled items out from in front of the src_i
                src -= sum(x <= src for x in popped)
                # if source is past the insertion point, increment src for each
                # previous insertion
                if src >= dest_index:
                    src += i
                yield src, dest_index + d_inc

            popped.append(src)
            # if the item moved up, icrement the destination index
            if dest_index <= src:
                d_inc += 1

    def reverse(self) -> None:
        self._list.reverse()
        self.reordered_.emit(self)
