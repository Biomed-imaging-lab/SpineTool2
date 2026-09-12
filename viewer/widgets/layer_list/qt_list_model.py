import pickle
from typing import List, Optional, Sequence, TypeVar

from PyQt5.QtCore import QMimeData, QModelIndex, Qt

from viewer.widgets.layer_list._base_item_model import _BaseEventedItemModel

ListIndexMIMEType = "application/x-list-index"
ItemType = TypeVar("ItemType")


class QtListModel(_BaseEventedItemModel[ItemType]):
    def mimeTypes(self) -> List[str]:
        return [ListIndexMIMEType, "text/plain"]

    def mimeData(self, indices: List[QModelIndex]) -> Optional["QMimeData"]:
        if not indices:
            return None
        items, indices = zip(*[(self.getItem(i), i.row()) for i in indices])
        return ItemMimeData(items, indices)

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        if not data or action != Qt.DropAction.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        if isinstance(data, ItemMimeData):
            moving_indices = data.indices
            if len(moving_indices) == 1:
                return self._root.move(moving_indices[0], destRow)

            return bool(self._root.move_multiple(moving_indices, destRow))
        return False


class ItemMimeData(QMimeData):
    def __init__(self, items: Sequence[ItemType], indices: Sequence[int]) -> None:
        super().__init__()
        self.items = items
        self.indices = tuple(sorted(indices))
        if items:
            self.setData(ListIndexMIMEType, pickle.dumps(self.indices))
            self.setText(" ".join(str(item) for item in items))

    def formats(self) -> List[str]:
        return [ListIndexMIMEType, "text/plain"]
