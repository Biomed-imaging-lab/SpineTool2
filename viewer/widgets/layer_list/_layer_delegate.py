from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QStyledItemDelegate

from utils.svg import QColoredSVGIcon
from viewer.utils.constants import SVGS_PATH
from viewer.widgets.layer_list._base_item_model import ItemRole
from viewer.widgets.layer_list.qt_layer_model import ThumbnailRole

if TYPE_CHECKING:
    from PyQt5 import QtCore
    from PyQt5.QtGui import QPainter
    from PyQt5.QtWidgets import QStyleOptionViewItem, QWidget


class LayerDelegate(QStyledItemDelegate):
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ):
        self.get_layer_icon(option, index)
        # paint the standard itemView (includes name, icon, and vis. checkbox)
        super().paint(painter, option, index)
        # paint the thumbnail
        self._paint_thumbnail(painter, option, index)

    def get_layer_icon(self, option: QStyleOptionViewItem, index: QtCore.QModelIndex):
        layer = index.data(ItemRole)
        if layer is None:
            return
        icon_name = f"new_{layer._type_string}"
        icon_path = SVGS_PATH + "/" + icon_name + ".svg"
        icon = QColoredSVGIcon(icon_path)
        # guessing theme rather than passing it through.
        bg = option.palette.color(option.palette.ColorRole.Window).red()
        option.icon = icon.colored(theme="dark" if bg < 128 else "light")
        option.decorationSize = QSize(18, 18)
        option.decorationPosition = option.Position.Right  # put icon on the right
        option.features |= option.ViewItemFeature.HasDecoration

    def _paint_thumbnail(self, painter, option, index):
        # paint the thumbnail
        # MAGICNUMBER: numbers from the margin applied in the stylesheet to
        # QtLayerTreeView::item
        thumb_rect = option.rect.translated(-2, 2)
        h = index.data(Qt.ItemDataRole.SizeHintRole).height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        image = index.data(ThumbnailRole)
        painter.drawPixmap(thumb_rect, QPixmap.fromImage(image))

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QWidget:
        # necessary for geometry, otherwise editor takes up full width.
        self.get_layer_icon(option, index)
        editor = super().createEditor(parent, option, index)
        # make sure editor has same alignment as the display name
        editor.setAlignment(
            Qt.AlignmentFlag(index.data(Qt.ItemDataRole.TextAlignmentRole))
        )
        return editor

    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> bool:
        # if the user clicks quickly on the visibility checkbox, we *don't*
        # want it to be interpreted as a double-click.  We want the visibilty
        # to simply be toggled.
        if event.type() == QMouseEvent.MouseButtonDblClick:
            self.initStyleOption(option, index)
            style = option.widget.style()
            check_rect = style.subElementRect(
                style.SubElement.SE_ItemViewItemCheckIndicator,
                option,
                option.widget,
            )
            if check_rect.contains(event.pos()):
                cur_state = index.data(Qt.ItemDataRole.CheckStateRole)
                if model.flags(index) & Qt.ItemFlag.ItemIsUserTristate:
                    state = Qt.CheckState((cur_state + 1) % 3)
                else:
                    state = (
                        Qt.CheckState.Unchecked if cur_state else Qt.CheckState.Checked
                    )
                return model.setData(index, state, Qt.ItemDataRole.CheckStateRole)
        # refer all other events to the QStyledItemDelegate
        return super().editorEvent(event, model, option, index)
