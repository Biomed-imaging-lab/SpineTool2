from typing import TYPE_CHECKING, TypeVar

from PyQt5.QtWidgets import QListView

from viewer.widgets.layer_list._base_item_view import _BaseEventedItemView
from viewer.widgets.layer_list.qt_list_model import QtListModel

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget

    from viewer.components.layer_list._evented_list import EventedList

ItemType = TypeVar("ItemType")


class QtListView(_BaseEventedItemView[ItemType], QListView):
    _root: "EventedList[ItemType]"

    def __init__(
        self,
        root: "EventedList[ItemType]",
        dragAndDropEnabled: bool = True,
        editable: bool = True,
        parent: "QWidget" = None,
    ) -> None:
        _BaseEventedItemView.__init__(self)
        QListView.__init__(self, parent)
        self.setDragDropMode(QListView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setRoot(root, dragAndDropEnabled, editable)

    def model(self) -> QtListModel[ItemType]:
        return super().model()
