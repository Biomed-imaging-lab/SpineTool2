from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from utils.qt_translater import Translater
from utils.shortcuts import ShortcutsHandler


class ShortcutViewer(QWidget):
    def __init__(
        self,
        parent: QWidget = None,
        description: str = "",
        values: OrderedDict[str, ShortcutsHandler] = OrderedDict(),
    ):
        super().__init__(parent=parent)

        self.shortcuts_providers = values

        # widgets
        self.groups_combo_box = QComboBox(self)
        self._label = QLabel(Translater.instance().get_translation("group"))
        self._table = QTableWidget(self)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setShowGrid(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.groups_combo_box.currentIndexChanged.connect(self._set_table)

        self.set_data()

        # layout
        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self._label)
        hlayout1.addWidget(self.groups_combo_box)
        hlayout1.setContentsMargins(0, 0, 0, 0)
        hlayout1.setSpacing(20)
        hlayout1.addStretch(0)

        layout = QVBoxLayout()
        layout.addLayout(hlayout1)
        layout.addWidget(self._table)

        self.setLayout(layout)

    def set_data(self):
        items = []
        translater = Translater.instance()
        for key in list(self.shortcuts_providers.keys()):
            items.append(translater.get_translation(key))
        if len(items) > 0:
            self.groups_combo_box.clear()
            self.groups_combo_box.addItems(items)
            self.groups_combo_box.setCurrentIndex(0)
            self._set_table(self.groups_combo_box.currentIndex())

    def set_providers(
        self, shortcuts_providers: OrderedDict[str, ShortcutsHandler] = OrderedDict()
    ):
        if len(shortcuts_providers) > 0:
            self.shortcuts_providers = shortcuts_providers
            self.set_data()

    def _set_table(self, group_index):
        translater = Translater.instance()
        # Keep track of what is in each column.
        self._action_name_col = 0
        self._subgroup_col = 1
        self._shortcut_col = 2

        # Set header strings for table.
        header_strs = ["", "", ""]
        header_strs[self._action_name_col] = translater.get_translation("action")
        header_strs[self._subgroup_col] = translater.get_translation("subgroup")
        header_strs[self._shortcut_col] = translater.get_translation("keybinding")

        self._table.clearContents()

        # Table styling set up.
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setStyleSheet("border-bottom: 2px solid white;")

        # Get all actions for the group.
        actions = self.shortcuts_providers[
            list(self.shortcuts_providers.keys())[group_index]
        ].shortcuts_info

        if len(actions) > 0:
            # Set up table based on number of actions and needed columns.
            self._table.setRowCount(len(actions))
            self._table.setColumnCount(3)
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.horizontalHeader().setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft
            )
            self._table.verticalHeader().setVisible(False)

            # Column set up.
            self._table.setColumnWidth(self._action_name_col, 370)
            self._table.setColumnWidth(self._subgroup_col, 190)
            self._table.setColumnWidth(self._shortcut_col, 145)
            self._table.setWordWrap(True)

            # Add some padding to rows
            self._table.setStyleSheet("QTableView::item { padding: 6px; }")

            # Go through all the actions in the layer and add them to the table.
            for row, (description, subgroup, shortcut) in enumerate(actions):
                # Set action description.
                self._table.setItem(
                    row,
                    self._action_name_col,
                    QTableWidgetItem(translater.get_translation(description)),
                )
                # Ensure long descriptions can be wrapped in cells
                self._table.resizeRowToContents(row)
                self._table.setItem(
                    row,
                    self._subgroup_col,
                    QTableWidgetItem(translater.get_translation(subgroup)),
                )
                self._table.setItem(row, self._shortcut_col, QTableWidgetItem(shortcut))

        else:
            # Display that there are no actions for this group.
            self._table.setRowCount(1)
            self._table.setColumnCount(3)
            self._table.setHorizontalHeaderLabels(header_strs)
            self._table.verticalHeader().setVisible(False)
            self._table.setItem(
                0, 0, QTableWidgetItem(translater.get_translation("no key bindings"))
            )

    def value(self):
        return self.shortcuts_providers
