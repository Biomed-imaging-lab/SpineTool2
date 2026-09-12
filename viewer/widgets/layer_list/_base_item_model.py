from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Any, Generic, Tuple, TypeVar, Union

from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt

from viewer.components.layer_list._evented_list import EventedList

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget


ItemType = TypeVar("ItemType")

ItemRole = Qt.UserRole
SortRole = Qt.UserRole + 1

_BASE_FLAGS = (
    Qt.ItemFlag.ItemIsSelectable
    | Qt.ItemFlag.ItemIsUserCheckable
    | Qt.ItemFlag.ItemIsEnabled
)


class _BaseEventedItemModel(QAbstractItemModel, Generic[ItemType]):
    _root: EventedList[ItemType]

    # ########## Reimplemented Public Qt Functions ##################

    def __init__(
        self,
        root: EventedList[ItemType],
        dragAndDropEnabled: bool = True,
        editable: bool = True,
        parent: "QWidget" = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setRoot(root)
        self.dragAndDropEnabled = dragAndDropEnabled
        self.editable = editable

    def parent(self, index):
        return QModelIndex()

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        if role == Qt.DisplayRole:
            return str(self.getItem(index))
        if role == ItemRole:
            return self.getItem(index)
        if role == SortRole:
            return index.row()
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = _BASE_FLAGS
        if self.dragAndDropEnabled:
            flags = flags | Qt.ItemFlag.ItemIsDragEnabled
        if self.editable:
            flags = flags | Qt.ItemFlag.ItemIsEditable
        if not index.isValid() or index.model() is not self:
            # we allow drops outside the items
            return Qt.ItemFlag.ItemIsDropEnabled
        if isinstance(self.getItem(index), MutableSequence):
            if self.dragAndDropEnabled:
                return flags | Qt.ItemFlag.ItemIsDropEnabled
            else:
                return flags
        return flags | Qt.ItemFlag.ItemNeverHasChildren

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def rowCount(self, parent: QModelIndex = None) -> int:
        if parent is None:
            parent = QModelIndex()
        try:
            return len(self.getItem(parent))
        except TypeError:
            return 0

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = None
    ) -> QModelIndex:
        if parent is None:
            parent = QModelIndex()

        return (
            self.createIndex(row, column, self.getItem(parent)[row])
            if self.hasIndex(row, column, parent)
            else QModelIndex()  # instead of index error, Qt wants null index
        )

    def supportedDropActions(self) -> Qt.DropActions:
        return Qt.MoveAction

    # ###### Non-Qt methods added for SelectableEventedList Model ############

    def setRoot(self, root: EventedList[ItemType]):
        if not isinstance(root, EventedList):
            raise TypeError(
                "root must be an instance of {class_name}".format(
                    class_name=EventedList,
                )
            )
        current_root = getattr(self, "_root", None)
        if root is current_root:
            return

        if current_root is not None:
            self._root.removing_.disconnect(self._on_begin_removing)
            self._root.removed_.disconnect(self._on_end_remove)
            self._root.inserting_.disconnect(self._on_begin_inserting)
            self._root.inserted_.disconnect(self._on_end_insert)
            self._root.moving_.disconnect(self._on_begin_moving)
            self._root.moved_.disconnect(self._on_end_move)

        self._root = root
        self._root.removing_.connect(self._on_begin_removing)
        self._root.removed_.connect(self._on_end_remove)
        self._root.inserting_.connect(self._on_begin_inserting)
        self._root.inserted_.connect(self._on_end_insert)
        self._root.moving_.connect(self._on_begin_moving)
        self._root.moved_.connect(self._on_end_move)

    def _split_nested_index(
        self, nested_index: Union[int, Tuple[int, ...]]
    ) -> Tuple[QModelIndex, int]:
        if isinstance(nested_index, int):
            return QModelIndex(), nested_index
        # Tuple indexes are used in NestableEventedList, so we support them
        # here so that subclasses needn't reimplmenet our _on_begin_* methods
        par = QModelIndex()
        *_p, idx = nested_index
        for i in _p:
            par = self.index(i, 0, par)
        return par, idx

    def _on_begin_inserting(self, index):
        par, idx = self._split_nested_index(index)
        self.beginInsertRows(par, idx, idx)

    def _on_end_insert(self, index, data):
        self._process_inserted_event(data)
        self.endInsertRows()

    def _on_begin_removing(self, index):
        par, idx = self._split_nested_index(index)
        self.beginRemoveRows(par, idx, idx)

    def _on_end_remove(self, index, data):
        self._process_removed_event(data)
        self.endRemoveRows()

    def _on_begin_moving(self, index, new_index):
        src_par, src_idx = self._split_nested_index(index)
        dest_par, dest_idx = self._split_nested_index(new_index)

        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _on_end_move(self, index, new_index, data):
        self.endMoveRows()

    def getItem(self, index: QModelIndex) -> ItemType:
        return self._root[index.row()] if index.isValid() else self._root

    def _process_inserted_event(self, data):
        # for subclasses to handle ItemType-specific data
        pass

    def _process_removed_event(self, data):
        # for subclasses to handle ItemType-specific data
        pass
