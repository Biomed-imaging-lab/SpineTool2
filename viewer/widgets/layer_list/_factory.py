from typing import TYPE_CHECKING

from viewer.components.layer_list._evented_list import EventedList
from viewer.components.layer_list.layerlist import LayerList

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget


def create_view(obj: EventedList, parent: "QWidget" = None):
    from viewer.widgets.layer_list.qt_layer_list import QtLayerList
    from viewer.widgets.layer_list.qt_list_view import QtListView

    if isinstance(obj, LayerList):
        return QtLayerList(obj, parent=parent)
    if isinstance(obj, EventedList):
        return QtListView(obj, parent=parent)
    raise TypeError(
        "Cannot create Qt view for obj: {obj}".format(
            obj=obj,
        )
    )


def create_model(
    obj: EventedList,
    dragAndDropEnabled: bool = True,
    editable: bool = True,
    parent: "QWidget" = None,
):
    from viewer.widgets.layer_list.qt_layer_model import QtLayerListModel
    from viewer.widgets.layer_list.qt_list_model import QtListModel

    if isinstance(obj, LayerList):
        return QtLayerListModel(obj, dragAndDropEnabled, editable, parent)
    if isinstance(obj, EventedList):
        return QtListModel(obj, dragAndDropEnabled, editable, parent)
    raise TypeError(
        "Cannot create Qt model for obj: {obj}".format(
            obj=obj,
        )
    )
