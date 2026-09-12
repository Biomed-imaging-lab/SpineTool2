from typing import TYPE_CHECKING, Generic, TypeVar

from PyQt5.QtCore import QItemSelection

from viewer.widgets.layer_list._base_item_model import ItemRole
from viewer.widgets.layer_list._factory import create_model

ItemType = TypeVar("ItemType")

if TYPE_CHECKING:
    from viewer.components.layer_list._evented_list import EventedList
    from viewer.widgets.layer_list._base_item_model import _BaseEventedItemModel


class _BaseEventedItemView(Generic[ItemType]):
    activeItem: ItemType

    def __init__(self):
        super().__init__()
        self.activeItem = None

    # ########## Reimplemented Public Qt Functions ##################

    def model(self) -> "_BaseEventedItemModel[ItemType]":  # for type hints
        return super().model()

    def selectionChanged(
        self: "_BaseEventedItemView",
        selected: QItemSelection,
        deselected: QItemSelection,
    ):
        sel = [i.data(ItemRole) for i in selected.indexes()]

        if len(sel) == 0:
            self.activeItem = None
        else:
            self.activeItem = sel[0]
        return super().selectionChanged(selected, deselected)

    # ###### Non-Qt methods added for EventedList Model ############

    def setRoot(
        self, root: "EventedList[ItemType]", dragAndDropEnabled: bool, editable: bool
    ):
        self._root = root
        self.setModel(create_model(root, dragAndDropEnabled, editable, self))
