from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class QtScrollableContainer(QScrollArea):
    def __init__(
        self,
        widget=None,
        alignment=Qt.AlignmentFlag.AlignCenter,
        max_height=400,
        parent=None,
    ):
        QScrollArea.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(52)
        self.setMinimumWidth(280)
        self.setMaximumHeight(max_height)
        self.internal_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.internal_widget.setLayout(layout)
        self.setWidget(self.internal_widget)
        self.setWidgetResizable(True)
        if widget:
            self.current_widget = widget
            layout.addWidget(widget, alignment=alignment)
            self.current_widget.show()
        else:
            self.current_widget = None

    def replace_widget(
        self, widget, delete=True, alignment=Qt.AlignmentFlag.AlignCenter
    ) -> None:
        if self.current_widget:
            self.current_widget.hide()
            if delete:
                self.current_widget.setParent(None)
                self.current_widget.deleteLater()
            else:
                self.current_widget.setParent(None)
        self.current_widget = widget
        if widget:
            self.internal_widget.layout().addWidget(
                self.current_widget, alignment=alignment
            )
            self.current_widget.show()
