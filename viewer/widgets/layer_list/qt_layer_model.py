import typing
from functools import partial

from PyQt5.QtCore import QModelIndex, QSize, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QWidget

from viewer.components.layer_list._evented_list import EventedList
from viewer.layers.base.base import Layer
from viewer.widgets.layer_list.qt_list_model import QtListModel

ThumbnailRole = Qt.UserRole + 2


class QtLayerListModel(QtListModel[Layer]):
    def __init__(
        self,
        root: EventedList[Layer],
        dragAndDropEnabled: bool = True,
        editable: bool = True,
        parent: QWidget = None,
    ):
        super().__init__(root, dragAndDropEnabled, editable, parent)
        self.slots = {}

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        if not index.isValid():
            return None
        layer = self.getItem(index)
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer._name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            return layer.get_source_str()
        if role == Qt.ItemDataRole.CheckStateRole:  # the "checked" state of this item
            return Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: typing.Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole:
            self.getItem(index).visible = Qt.CheckState(value) == Qt.CheckState.Checked
        elif role == Qt.ItemDataRole.EditRole:
            self.getItem(index).name = value
            role = Qt.ItemDataRole.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def _process_inserted_event(self, data: Layer):
        on_thumbnail_changed = partial(self._on_thumbnail_changed, data)
        on_visible_changed = partial(self._on_visible_changed, data)
        on_name_changed = partial(self._on_name_changed, data)
        self.slots[data] = {
            "thumbnail": on_thumbnail_changed,
            "visible": on_visible_changed,
            "name": on_name_changed,
        }
        data.thumbnail_.connect(on_thumbnail_changed)
        data.visible_.connect(on_visible_changed)
        data.name_.connect(on_name_changed)

    def _process_removed_event(self, data: Layer):
        data.thumbnail_.disconnect(self.slots[data]["thumbnail"])
        data.visible_.disconnect(self.slots[data]["visible"])
        data.name_.disconnect(self.slots[data]["name"])
        del self.slots[data]

    def _on_thumbnail_changed(self, layer: Layer):
        row = self.index(self._root.index(layer))
        self.dataChanged.emit(row, row, [ThumbnailRole])

    def _on_visible_changed(self, layer: Layer):
        row = self.index(self._root.index(layer))
        self.dataChanged.emit(row, row, [Qt.ItemDataRole.CheckStateRole])

    def _on_name_changed(self, layer: Layer):
        row = self.index(self._root.index(layer))
        self.dataChanged.emit(row, row, [Qt.ItemDataRole.DisplayRole])
