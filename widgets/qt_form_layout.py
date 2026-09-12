from PyQt5.QtWidgets import QFormLayout


class QtFormLayout(QFormLayout):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setContentsMargins(8, 4, 8, 6)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
