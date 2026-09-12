from typing import TYPE_CHECKING

from PyQt5.QtCore import QSortFilterProxyModel, Qt
from PyQt5.QtGui import QKeyEvent

from viewer.layers.base.base import Layer
from viewer.widgets.layer_list._base_item_model import SortRole, _BaseEventedItemModel
from viewer.widgets.layer_list._layer_delegate import LayerDelegate
from viewer.widgets.layer_list.qt_list_view import QtListView

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget

    from viewer.components.layer_list.layerlist import LayerList


class ReverseProxyModel(QSortFilterProxyModel):
    def __init__(self, model: _BaseEventedItemModel) -> None:
        super().__init__()
        self.setSourceModel(model)
        self.setSortRole(SortRole)
        self.sort(0, Qt.SortOrder.DescendingOrder)

    def dropMimeData(self, data, action, destRow, col, parent):
        row = 0 if destRow == -1 else self.sourceModel().rowCount() - destRow
        return self.sourceModel().dropMimeData(data, action, row, col, parent)


class QtLayerList(QtListView[Layer]):
    def __init__(
        self,
        root: "LayerList",
        dragAndDropEnabled: bool = True,
        editable: bool = True,
        parent: "QWidget" = None,
    ) -> None:
        super().__init__(root, dragAndDropEnabled, editable, parent)
        self.setItemDelegate(LayerDelegate())
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.setModel(ReverseProxyModel(self.model()))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            return super().keyPressEvent(event)
        return event.ignore()
